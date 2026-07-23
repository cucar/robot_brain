/// Thalamus — Brain's relay station for reference frame transfers.
///
/// Abstracts access to neurons, channels, and dimension mappings.
/// Handles bidirectional translation between external signals and internal neuron representations.
/// Named after the biological thalamus which routes sensory signals and translates reference frames.
///
/// Owns:
///   - Region[R] tree (each Region owns Column[C], each Column owns Neurons)
///   - Central metadata maps (neuronsByValue, baseNeurons, neuronParents, neuronLevels)
///   - Death ledger (scheduled neuron deaths)
///   - Channel / dimension registries
///   - Quantizer (scalar-to-bucket discretization)
///   - ID allocators (channel, dimension, neuron)

use rustc_hash::{FxHashMap, FxHashSet};

use crate::memory::InferredNeuron;
use crate::column::{
    ColumnProcessResult, ConnectionSpec, DeleteOp,
    NeuronCreateSpec,
};
use crate::context::{SpatialContext, TemporalContext};
use crate::diagnostics::{InferenceResultItem, InferenceType};
use crate::neuron::{
    ActiveNeuron, AgeState, ContextRefEntry, TemporalContextRefUpdate, SpatialContextRefUpdate,
    Correction, ErrorFeedback, TemporalNeuron, TemporalLearningWork,
    SerializedNeuron, Vote,
};
use crate::quantizer::{QuantizeMode, Quantizer};
use crate::region::Region;
use crate::types::{
    ChannelId, Coordinate, DimensionId, Distance, GroupMode, FrameNumber,
    Level, NeuronId, NeuronType, Reward,
};

// ── Supporting structs ──────────────────────────────────────────────────────

/// Metadata for a base (sensory/action) neuron — stored centrally in Thalamus.
#[derive(Debug, Clone)]
pub struct BaseNeuron {
    pub channel_id: ChannelId,
    pub neuron_type: NeuronType,
    pub coordinate: Coordinate,
}

/// Stored channel spec — immutable after registration.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: String,
    pub dimensions: Vec<DimSpec>,
    pub learn_action_sequences: bool,
}

/// Stored per-dimension spec within a channel.
#[derive(Debug, Clone)]
pub struct DimSpec {
    pub id: DimensionId,
    pub name: String,
    pub kind: DimKind,
    pub resolution: u32,
    pub mode: Option<String>,
    pub boundaries: Option<Vec<f64>>,
    pub actions: Option<Vec<i32>>,
    pub default_action: Option<i32>,
    pub warmup_samples: Option<usize>,
}

/// Dimension kind — input (observable) or action (choosable).
#[derive(Debug, Clone, PartialEq)]
pub enum DimKind {
    Input,
    Action,
}

/// Result of getNeuronIdForPoint — the neuron id and whether it was newly created.
#[derive(Debug, Clone)]
pub struct PointLookup {
    pub id: NeuronId,
    pub is_new: bool,
}

/// Result of registerChannelSpec — the allocated channel id and per-dimension id map.
#[derive(Debug, Clone)]
pub struct ChannelRegistration {
    pub channel_id: ChannelId,
    pub dimension_ids: FxHashMap<String, DimensionId>,
}

/// Per-age state tracked across frames in the level loop. Extends neuron::AgeState
/// with vote/context/threshold data that rides with the state between frames so
/// evaluateVoteError can judge prior-frame votes without reaching back into the neuron.
#[derive(Debug, Clone)]
pub struct LevelAgeState {
    /// Activated pattern id (set when a pattern fires at this age).
    pub activated_pattern_id: Option<NeuronId>,
    /// Votes cast at this age (populated by collectVotes, consumed by evaluateVoteError next frame).
    pub votes: Option<Vec<Vote>>,
    /// TemporalContext snapshot at vote time (for misprediction diagnostics).
    pub context: Option<Vec<ContextRefEntry>>,
    /// Error threshold at vote time (for next-frame error correction decision).
    pub threshold: Option<f64>,
}

impl Default for LevelAgeState {
    fn default() -> Self {
        Self {
            activated_pattern_id: None,
            votes: None,
            context: None,
            threshold: None,
        }
    }
}

/// An activation produced by recognition or error-correction.
#[derive(Debug, Clone)]
pub struct Activation {
    pub parent_id: NeuronId,
    pub pattern_id: NeuronId,
    pub age: Distance,
}

/// Results from processLevel — returned to the caller (Brain) for accumulation.
pub struct ProcessLevelResult {
    pub activations: Vec<Activation>,
    pub votes: Vec<FlatVote>,
    pub neuron_specs: Vec<NeuronCreateSpec>,
    pub results: Vec<ColumnProcessResult>,
    pub orchestration: OrchestrationTimings,
}

/// Output of one spatial level pass.
/// Spatial creates no neurons during the level loop; corrections come from a separate pass, settling
/// the per-neuron correction requests carried in the dispatch results.
pub struct SpatialLevelResult {
    pub activations: Vec<Activation>,
    pub results: Vec<crate::column::SpatialColumnResult>,
    pub orchestration: OrchestrationTimings,
}

/**
 * Wall-clock for the orchestration steps inside thalamus.process_level that
 * wrap the per-neuron dispatch. Lets us see how much of process_levels is
 * spent outside neuron op execution.
 */
#[derive(Debug, Clone, Copy, Default)]
pub struct OrchestrationTimings {
    pub get_level_tasks: f64,
    pub dispatch_frame: f64,
    pub collect_activations: f64,
    pub collect_votes: f64,
}

impl OrchestrationTimings {
    pub fn add(&mut self, other: &OrchestrationTimings) {
        self.get_level_tasks     += other.get_level_tasks;
        self.dispatch_frame      += other.dispatch_frame;
        self.collect_activations += other.collect_activations;
        self.collect_votes       += other.collect_votes;
    }
}

/// A flat vote with voter id attached — used for consensus after the level loop.
#[derive(Debug, Clone)]
pub struct FlatVote {
    pub voter_id: NeuronId,
    pub neuron_id: NeuronId,
    pub strength: f64,
    pub reward: f64,
    pub distance: Distance,
}

/// Snapshot of brain state for backup/restore.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub neurons: Vec<SnapshotNeuronEntry>,
    pub channel_name_to_id: FxHashMap<String, ChannelId>,
    pub dimension_name_to_id: FxHashMap<String, DimensionId>,
}

/// A single neuron entry in a snapshot — carries serialized neuron data plus resolved metadata.
#[derive(Debug, Clone)]
pub struct SnapshotNeuronEntry {
    pub neuron: SerializedNeuron,
    /// Temporal-hierarchy depth: 0 = sensory, 1+ = temporal pattern.
    pub temporal_level: Level,
    /// Spatial-hierarchy depth: 0 = sensory/base, 1+ = spatial correction pattern.
    /// Carried separately from `temporal_level` (all spatial patterns sit at temporal level 0).
    pub spatial_level: Level,
    pub base_neuron: Option<BaseNeuron>,
    pub parent_id: Option<NeuronId>,
}

/// Result of evaluateVoteError — whether to fire error correction and the observed error rate.
struct VoteErrorResult {
    fire: bool,
    error_rate: f64,
}

// ── Thalamus ────────────────────────────────────────────────────────────────

pub struct Thalamus {
    debug: bool,
    pattern_forget_rate: f64,
    context_length: u32,
    regions: usize,
    columns: usize,

    // ── Temporary experimental toggles ──────────────────────────────────────────
    // Each of these is a pending decision, not a feature: when its experiment concludes, the winning
    // behavior is hard-wired and the toggle is DELETED.

    /// Master learning toggle, fixed at construction and threaded to every region, column, and neuron
    /// like the other options.
    learning: bool,

    /// Coordinate → neuron id lookup (replaces JS neuronsByValue with string keys).
    /// Uses Coordinate directly as key since it derives Hash + Eq.
    neurons_by_value: FxHashMap<Coordinate, NeuronId>,

    /// Sensory neuron metadata: neuron id → { channel_id, type, coordinate }.
    base_neurons: FxHashMap<NeuronId, BaseNeuron>,

    /// Pattern neuron parent: neuron id → parent neuron id.
    neuron_parents: FxHashMap<NeuronId, NeuronId>,

    /// Neuron temporal level: neuron id → level (0 = sensory, 1+ = temporal pattern).
    /// Spatial corrections start at temporal level 0 (they enter temporal via the apex handoff
    /// at temporal_level_index[0]) regardless of their place in the spatial hierarchy.
    neuron_temporal_levels: FxHashMap<NeuronId, Level>,

    /// Neuron spatial level: neuron id → level in the spatial hierarchy (0 = sensory/base, 1+ = spatial correction).
    /// Stored separately from `neuron_temporal_levels` because a neuron occupies independent positions
    /// in the spatial and temporal hierarchies (see spatial-processing.md §3.3).
    /// Absent entries default to 0 — pre-spatial-era persisted neurons inherit this on load.
    neuron_spatial_levels: FxHashMap<NeuronId, Level>,

    /// Cumulative count of spatial correction patterns created by the spatial correction pass.
    /// Diagnostic — surfaced via Brain.get_spatial_correction_count() for harness validation.
    spatial_corrections_minted: u64,

    /// Cumulative count of spatial children retired by the one test's delete pass. Paired with
    /// `spatial_corrections_minted`, this is the cold-start churn the Phase-1 gate watches
    /// (docs/algorithm.md, "Risks"). Surfaced via Brain.get_spatial_deletion_count().
    spatial_corrections_deleted: u64,

    /// Death ledger: frame_number → set of neuron ids scheduled to die.
    death_ledger: FxHashMap<FrameNumber, FxHashSet<NeuronId>>,

    /// Reverse lookup: neuron id → scheduled death frame.
    neuron_death_frame: FxHashMap<NeuronId, FrameNumber>,

    /// Channel specs: channel id → stored spec.
    channel_specs: FxHashMap<ChannelId, ChannelSpec>,

    /// Dimension specs: dimension id → stored spec (flattened across all channels).
    dimension_specs: FxHashMap<DimensionId, DimSpec>,

    /// Per-channel action neuron ids — ordered Vec for deterministic exploration.
    channel_actions: FxHashMap<ChannelId, Vec<NeuronId>>,

    /// Per-channel default action neuron id.
    channel_default_actions: FxHashMap<ChannelId, NeuronId>,

    /// Channel name → id.
    channel_name_to_id: FxHashMap<String, ChannelId>,

    /// Channel id → name.
    channel_id_to_name: FxHashMap<ChannelId, String>,

    /// Per-channel SPATIAL neighbor sets, one per spatial level — restrict d=0 co-activation grouping
    /// (spatial pattern creation and the spatial neighborhood/context built by the correction pass)
    /// to the listed channels plus the channel itself.
    /// Index ℓ holds the neighbor set a LEVEL-ℓ neuron of this channel uses — the level-based radius:
    /// an encoder declares a growing neighborhood per level (e.g. radius 1 for L0, radius 2 for L1, …), so
    /// receptive fields widen with hierarchy depth instead of at the base.
    /// A level past the declared list reuses the LAST set (the neighborhoods stop growing once they cover the
    /// input), and a flat single-set declaration applies that one set at every level.
    /// Separate from the temporal set so a channel can group spatially with its near neighbors while
    /// still sequencing temporally against a different (or unrestricted) channel set.
    spatial_channel_neighbors: FxHashMap<ChannelId, Vec<FxHashSet<ChannelId>>>,

    /// Per-channel TEMPORAL neighbor set — restricts d>0 sequence learning (temporal connection
    /// pre-wiring, temporal pattern minting, vote-error evaluation, and the per-task temporal
    /// context) to the listed channels plus the channel itself.
    /// This is the set of channels whose past a channel may sequence against to predict the future.
    /// Separate from the spatial set per the split above.
    ///
    /// Shared semantics for both maps: the neighbor relationship is a property of the channel graph
    /// at the SENSORY level only.
    /// L1+ pattern neurons have NO channel — they emerge from cross-channel correlations and don't
    /// belong to any single channel.
    /// When a pattern is the "parent" of a filter lookup, `get_neuron_channel_id` returns None and
    /// the predicate falls through to all-pairs (no restriction).
    /// Channels NOT in a map have the default all-pairs neighborhood for that phase — preserving
    /// original pre-neighbor behavior for stocks, text, etc.
    temporal_channel_neighbors: FxHashMap<ChannelId, FxHashSet<ChannelId>>,

    /// Dimension name → id.
    dimension_name_to_id: FxHashMap<String, DimensionId>,

    /// Dimension id → name.
    dimension_id_to_name: FxHashMap<DimensionId, String>,

    /// Scalar-to-bucket discretizer.
    pub quantizer: Quantizer,

    /// ID allocators — advanced past max when restoring from a snapshot so newly
    /// allocated IDs don't collide with persisted ones.
    next_channel_id: ChannelId,
    next_dimension_id: DimensionId,
    next_neuron_id: NeuronId,

    /// Level counts — index = level, value = count of neurons at that level.
    /// Used for efficient max-level diagnostics lookup.
    temporal_level_counts: Vec<i64>,

    /// Region instances, indexed by region index.
    region_list: Vec<Region>,
}

impl Thalamus {
    pub fn new(
        debug: bool,
        pattern_forget_rate: f64,
        context_length: u32,
        group_threshold: f64,
        group_mode: GroupMode,
        regions: usize,
        columns: usize,
        learning: bool,
    ) -> Self {
        // construct the Region[R] tree — each Region constructs its Column[C]
        let channel_actions = FxHashMap::default();
        let channel_default_actions = FxHashMap::default();
        let mut region_list = Vec::with_capacity(regions);
        for _ in 0..regions {
            region_list.push(Region::new(
                columns,
                &channel_actions,
                &channel_default_actions,
                context_length,
                group_threshold,
                group_mode,
                learning,
            ));
        }

        Self {
            debug,
            pattern_forget_rate,
            context_length,
            regions,
            columns,
            learning,
            neurons_by_value: FxHashMap::default(),
            base_neurons: FxHashMap::default(),
            neuron_parents: FxHashMap::default(),
            neuron_temporal_levels: FxHashMap::default(),
            neuron_spatial_levels: FxHashMap::default(),
            spatial_corrections_minted: 0,
            spatial_corrections_deleted: 0,
            death_ledger: FxHashMap::default(),
            neuron_death_frame: FxHashMap::default(),
            channel_specs: FxHashMap::default(),
            dimension_specs: FxHashMap::default(),
            channel_actions,
            channel_default_actions: FxHashMap::default(),
            channel_name_to_id: FxHashMap::default(),
            channel_id_to_name: FxHashMap::default(),
            spatial_channel_neighbors: FxHashMap::default(),
            temporal_channel_neighbors: FxHashMap::default(),
            dimension_name_to_id: FxHashMap::default(),
            dimension_id_to_name: FxHashMap::default(),
            quantizer: Quantizer::new(),
            next_channel_id: 1,
            next_dimension_id: 1,
            next_neuron_id: 1,
            temporal_level_counts: Vec::new(),
            region_list,
        }
    }

    // ── Neuron coordinate lookup ────────────────────────────────────────────

    /// Get or create a sensory neuron ID from a frame point. coordinate form: {dim_id, bucket_id}
    /// Returns the neuron id and whether a new neuron was allocated.
    pub fn get_neuron_id_for_point(&mut self, coordinate: &Coordinate, channel_id: ChannelId, neuron_type: NeuronType) -> PointLookup {

        // try to find existing neuron — if found, return it
        if let Some(&neuron_id) = self.neurons_by_value.get(coordinate) {
            return PointLookup { id: neuron_id, is_new: false };
        }

        // allocate id and register metadata; Neuron construction deferred to create_neurons
        let id = self.allocate_sensory_neuron(coordinate, channel_id, neuron_type);
        if self.debug { println!("Created new sensory neuron {} for {}:{}", id, coordinate.dim_id, coordinate.bucket_id); }
        PointLookup { id, is_new: true }
    }

    /// Returns neuron ID by coordinate, or None if not found.
    pub fn get_neuron_id_by_coordinate(&self, coordinate: &Coordinate) -> Option<NeuronId> {
        self.neurons_by_value.get(coordinate).copied()
    }

    // ── Routing ─────────────────────────────────────────────────────────────

    /// Pure deterministic region-routing function. Maps a neuron id to its owning region.
    /// Interleaving (rather than chunking by id range) keeps id-bursts spread evenly:
    /// the error-correction pattern ids allocated in one frame spread evenly across regions
    /// instead of piling onto one.
    pub fn route_neuron(&self, neuron_id: NeuronId) -> usize {
        (neuron_id as usize) % self.regions
    }

    /// Bucket a flat batch by owning region using a key extractor.
    /// Returns a Vec indexed by region_idx, each entry the sub-list indices for that region.
    fn bucket_by_region_indices<T, F>(&self, batch: &[T], key: F) -> Vec<Vec<usize>>
    where
        F: Fn(&T) -> NeuronId,
    {
        let mut buckets: Vec<Vec<usize>> = (0..self.regions).map(|_| Vec::new()).collect();
        for (i, item) in batch.iter().enumerate() {
            let r = self.route_neuron(key(item));
            buckets[r].push(i);
        }
        buckets
    }

    // ── Neuron allocation ───────────────────────────────────────────────────

    /// Allocate a sensory neuron id and register its metadata. Does NOT construct
    /// the Neuron — that happens when the caller sends specs to create_neurons.
    fn allocate_sensory_neuron(&mut self, coordinate: &Coordinate, channel_id: ChannelId, neuron_type: NeuronType) -> NeuronId {
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;
        self.neuron_temporal_levels.insert(id, 0);
        self.neurons_by_value.insert(coordinate.clone(), id);
        self.base_neurons.insert(id, BaseNeuron { channel_id, neuron_type, coordinate: coordinate.clone() });
        self.increment_temporal_level_count(0);
        id
    }

    /// Allocate a TEMPORAL pattern neuron. Pre-wires d=1..age connections toward each per-age
    /// active set, filtered to the parent's neighbor channels — the new pattern only pre-wires
    /// toward channels in the parent's neighbor graph, matching how learn_temporal_connections
    /// restricts future strengthening. The pattern occupies temporal level=parent.level+1; its
    /// spatial_level defaults to 0.
    /// Does NOT touch the parent's routing table (that happens inside parent.process_temporal_frame
    /// via add_temporal_pattern) and does NOT register death (death frame is known only after
    /// parent.add_temporal_pattern runs).
    pub fn allocate_temporal_pattern_neuron(
        &mut self,
        level: Level,
        parent_id: NeuronId,
        age: Distance,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
    ) -> PatternNeuronSpec {

        // resolve connection spec using thalamus-local lookups (channel, reward)
        let parent_channel = self.get_neuron_channel_id(parent_id).unwrap_or(0);
        let mut connections = Vec::new();
        for a in 0..age.min(sensory_neurons.len() as u32) {
            let a_idx = a as usize;
            for &sensory_neuron_id in &sensory_neurons[a_idx] {
                let channel_id = self.get_neuron_channel_id(sensory_neuron_id).unwrap_or(0);
                if !self.is_temporal_neighbor_channel(parent_channel, channel_id) { continue; }
                let reward = rewards[a_idx].get(&channel_id).copied().unwrap_or(0.0);
                connections.push(ConnectionSpec {
                    distance: age - a,
                    to_neuron_id: sensory_neuron_id,
                    strength: 1.0,
                    reward,
                    channel_id,
                    dim_id: None,
                });
            }
        }

        // allocate id and build the spec for Column.create_neurons
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;

        // register metadata centrally (Neuron construction deferred to create_neurons).
        self.neuron_parents.insert(id, parent_id);
        self.neuron_temporal_levels.insert(id, level);
        self.increment_temporal_level_count(level);

        // Inherit the parent's full coordinate exactly like spatial corrections.
        // A temporal correction is a refinement of the parent inferences under observed context.
        // It asserts the same value, so the coordinate is a true invariant rather than an invented one.
        // Giving deeper temporal patterns a concrete channel lets neighbor filtering apply at higher levels.
        // Without this they fall back to all-pairs (no neighbor restriction), grouping across every channel.
        // NOT registered in neurons_by_value. That requires coordinate uniqueness.
        // Value → neuron resolution must always land on the base sensory/action neuron.
        // The inherited coordinate is metadata for neighbor filtering and consensus grouping only.
        // Refined tokens are reached solely via routing matches.
        // The parent is always registered in base_neurons, so this chains down to the original L0 coordinate.
        // On restore the coordinate is rederived by walking neuron_parents to the L0 ancestor (see restore_snapshot).
        let inherited = self.base_neurons.get(&parent_id)
            .unwrap_or_else(|| panic!(
                "allocate_temporal_pattern_neuron: parent {} has no base-neuron coordinate to inherit",
                parent_id
            ))
            .clone();
        self.base_neurons.insert(id, inherited);

        PatternNeuronSpec { id, forget_rate: self.pattern_forget_rate, connections }
    }

    /// The forget rate to stamp on base sensory/action neurons. Base neurons never die (no parent →
    /// never reaped), but this rate drives the decay of their hosted correction children, so it must
    /// be the brain-wide `pattern_forget_rate` — otherwise (rate 0.0) every child pattern hosted on a
    /// base neuron would be immortal, which was a bug.
    pub fn base_neuron_forget_rate(&self) -> f64 {
        self.pattern_forget_rate
    }

    /// Allocate a SPATIAL pattern neuron, seeded with the d=0 event connections it was minted to
    /// predict: the co-activation at its OWN level, observed on the birth frame, cut to its own
    /// neighborhood. This is the spatial half of what [allocate_temporal_pattern_neuron] already
    /// does — a pattern is born knowing the situation whose surprise paid its price, so it predicts
    /// on its first fire instead of firing blind, subsuming its parent, and leaving the frame
    /// unaccounted. Connectionless only when its level's neighborhood is genuinely empty at birth.
    /// Post-natal refinement is unchanged: `learn_spatial_event_connections` keeps upserting these
    /// edges on every frame the pattern is active, exactly as the temporal side does.
    /// The pattern occupies spatial_level=parent.spatial_level+1; its temporal level stays 0 (it
    /// enters temporal via the apex handoff at temporal_level_index[0]).
    /// Does NOT touch the parent's routing table (that happens inside column.install_spatial_corrections
    /// via add_spatial_pattern) and does NOT register death (death frame is known only after
    /// parent.add_spatial_pattern runs).
    pub fn allocate_spatial_pattern_neuron(
        &mut self,
        level: Level,
        parent_id: NeuronId,
        level_actives: &FxHashSet<NeuronId>,
    ) -> PatternNeuronSpec {

        // allocate id and build the spec for Column.create_neurons
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;

        // register metadata centrally (Neuron construction deferred to create_neurons).
        // Spatial pattern neurons sit at temporal level 0 — they enter the temporal sweep via the
        // apex handoff, NOT as a pattern at temporal level+1.
        self.neuron_parents.insert(id, parent_id);
        self.neuron_temporal_levels.insert(id, 0);
        self.neuron_spatial_levels.insert(id, level);
        self.increment_temporal_level_count(0);

        // Inherit the parent's full coordinate — channel, dimension, bucket and type.
        // A correction is a refinement of the parent observable, not a new one: it asserts the same
        // value under a specific neighborhood, so the coordinate is a true invariant rather than an
        // invented one. This keeps temporal level 0 a uniform coordinate-bearing interface (the apex
        // handoff injects these patterns there, where the per-dimension consensus and neighbor
        // filtering treat them like any L0 token), and is what stops aggregate_votes from panicking
        // on coordinate-less apex vote targets. The parent is always registered in base_neurons —
        // an L0 sensory directly, or, for deeper levels, an already-inherited pattern — so this
        // chains down to the original L0 coordinate.
        //
        // NOT registered in neurons_by_value: that map requires coordinate uniqueness, and
        // value→neuron resolution (action targets, event lookup) must always land on the L0
        // sensory/action neuron. The inherited coordinate is metadata for consensus grouping,
        // neighbor filtering, and vote dequantization only — refined tokens are reached solely via
        // routing matches. Persistence needs no new field: on restore the coordinate is derivable
        // by walking neuron_parents to the L0 ancestor.
        let inherited = self.base_neurons.get(&parent_id)
            .unwrap_or_else(|| panic!(
                "allocate_spatial_pattern_neuron: parent {} has no base-neuron coordinate to inherit",
                parent_id
            ))
            .clone();
        self.base_neurons.insert(id, inherited);

        // A newborn is created with NO connections (docs/algorithm.md, "Creating a child": "It is
        // created with no connections. Its connections belong to its own level, which has not been
        // observed at the moment of creation."). It learns its own model by ordinary counting as the
        // level above it populates. Seeding edges toward this frame's co-active neighbors — which may
        // themselves be churned away by the delete pass — is both off-spec and a source of dangling
        // connection targets on restore.
        let _ = level_actives;
        let connections = Vec::new();

        PatternNeuronSpec { id, forget_rate: self.pattern_forget_rate, connections }
    }

    // ── Neuron metadata getters ─────────────────────────────────────────────

    /// Get the channel id for a neuron. Sensory neurons return their registration channel.
    /// L1+ pattern neurons return None — they emerge from cross-channel correlations and have no
    /// channel of their own. Callers that use channel ids for neighbor lookups should treat None
    /// as "no neighbor restriction" (`is_spatial_neighbor_channel` / `is_temporal_neighbor_channel`
    /// already return true when the parent's channel has no neighbor list registered for that phase).
    pub fn get_neuron_channel_id(&self, neuron_id: NeuronId) -> Option<ChannelId> {
        self.base_neurons.get(&neuron_id).map(|b| b.channel_id)
    }

    /// Resolve a channel name plus a list of neighbor names into a channel id and a neighbor id set.
    /// Names not in the registry are silently ignored — encoders typically register every relevant
    /// channel first, then make a second pass to declare neighbor relationships (forward references
    /// are otherwise impossible during single-pass registration).
    /// Does NOT insert the channel into its own set — each setter below decides for itself.
    fn resolve_neighbor_ids(&self, name: &str, neighbor_names: &[String]) -> (ChannelId, FxHashSet<ChannelId>) {
        let channel_id = self.channel_name_to_id.get(name).copied()
            .unwrap_or_else(|| panic!("set neighbors: channel '{}' not registered", name));
        let neighbor_ids: FxHashSet<ChannelId> = neighbor_names.iter()
            .filter_map(|n| self.channel_name_to_id.get(n).copied())
            .collect();
        (channel_id, neighbor_ids)
    }

    /// Declare the SPATIAL (d=0 co-activation) neighbor channels for a registered channel.
    /// The list is used VERBATIM — the channel is NOT implicitly added to its own set. An empty list
    /// therefore disables spatial co-activation entirely: no cross-channel grouping AND no
    /// intra-channel grouping between multiple dims of the same channel. That is how a temporal-only
    /// workload turns spatial processing off. To keep intra-channel co-activation, list the channel
    /// itself. Channels with NO call retain the default all-pairs spatial neighborhood.
    pub fn set_spatial_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        let (channel_id, neighbor_ids) = self.resolve_neighbor_ids(name, neighbor_names);
        self.spatial_channel_neighbors.insert(channel_id, vec![neighbor_ids]);
    }

    /// Declare per-level SPATIAL neighbor sets for a registered channel — the level-based radius.
    /// `neighbor_names_by_level[ℓ]` is the neighbor list a LEVEL-ℓ neuron of this channel uses
    /// (e.g. the radius-(ℓ+1) neighborhood of a retinotopic pixel). Levels past the end of the list reuse
    /// the last declared set. Each list is used verbatim like `set_spatial_neighbors`.
    pub fn set_spatial_neighbor_levels(&mut self, name: &str, neighbor_names_by_level: &[Vec<String>]) {
        if neighbor_names_by_level.is_empty() { return; }
        let mut sets = Vec::with_capacity(neighbor_names_by_level.len());
        let mut channel = 0;
        for names in neighbor_names_by_level {
            let (channel_id, neighbor_ids) = self.resolve_neighbor_ids(name, names);
            channel = channel_id;
            sets.push(neighbor_ids);
        }
        self.spatial_channel_neighbors.insert(channel, sets);
    }

    /// Declare the TEMPORAL (d>0 sequence) neighbor channels for a registered channel.
    /// The channel is always added to its own set — a channel always sequences against its own past.
    /// Calling this with an empty list shrinks the temporal neighborhood to {itself}.
    /// Channels with NO call retain the default all-pairs temporal neighborhood.
    pub fn set_temporal_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        let (channel_id, mut neighbor_ids) = self.resolve_neighbor_ids(name, neighbor_names);
        neighbor_ids.insert(channel_id);
        self.temporal_channel_neighbors.insert(channel_id, neighbor_ids);
    }

    /// Declare the same neighbor set for BOTH phases — convenience for channels whose spatial and
    /// temporal neighbors coincide (e.g. retinotopic pixels). The channel is added to its own set in
    /// both maps (matching the original combined-neighbor behavior).
    pub fn set_channel_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        let (channel_id, mut neighbor_ids) = self.resolve_neighbor_ids(name, neighbor_names);
        neighbor_ids.insert(channel_id);
        self.spatial_channel_neighbors.insert(channel_id, vec![neighbor_ids.clone()]);
        self.temporal_channel_neighbors.insert(channel_id, neighbor_ids);
    }

    /// Test whether `target_channel` is in `parent_channel`'s SPATIAL neighbor set at the given
    /// spatial level — the level-based radius: a level-ℓ parent reads its channel's ℓ-th declared
    /// neighbor set (levels past the declared list reuse the last set, so neighborhoods stop growing once
    /// they cover the input). Returns true if `parent_channel` has no spatial neighbor list
    /// registered (default all-pairs).
    pub fn is_spatial_neighbor_channel(&self, parent_channel: ChannelId, parent_level: Level, target_channel: ChannelId) -> bool {
        match self.spatial_channel_neighbors.get(&parent_channel) {
            None => true,
            Some(sets) => {
                let idx = (parent_level as usize).min(sets.len() - 1);
                sets[idx].contains(&target_channel)
            }
        }
    }

    /// Keep only the top neuron over each region: drop any apex neuron a higher-level one covers.
    pub fn filter_apex_by_coverage(&self, apex: &FxHashSet<NeuronId>) -> FxHashSet<NeuronId> {

        // Coverage needs each neuron's position-channel and spatial level; patterns inherit the parent's.
        let members: Vec<(NeuronId, ChannelId, Level)> = apex.iter()
            .filter_map(|&id| self.base_neurons.get(&id).map(|b| (id, b.channel_id, self.get_neuron_spatial_level(id))))
            .collect();

        // filter the covered neurons
        let mut kept = FxHashSet::default();
        for &(bid, bch, blvl) in &members {

            // Covered when a strictly-higher neuron's level-based radius contains B's position.
            let covered = members.iter().any(|&(aid, ach, alvl)|
                aid != bid && alvl > blvl && self.is_spatial_neighbor_channel(ach, alvl, bch));
            if !covered { kept.insert(bid); }
        }
        kept
    }

    /// Test whether `target_channel` is in `parent_channel`'s TEMPORAL neighbor set.
    /// Returns true if `parent_channel` has no temporal neighbor list registered (default all-pairs).
    pub fn is_temporal_neighbor_channel(&self, parent_channel: ChannelId, target_channel: ChannelId) -> bool {
        match self.temporal_channel_neighbors.get(&parent_channel) {
            None => true,
            Some(set) => set.contains(&target_channel),
        }
    }

    /// Get the type for a neuron (Event or Action).
    pub fn get_neuron_type(&self, neuron_id: NeuronId) -> Option<NeuronType> {
        self.base_neurons.get(&neuron_id).map(|b| b.neuron_type)
    }

    /// Get the parent neuron ID for a pattern neuron.
    pub fn get_neuron_parent(&self, neuron_id: NeuronId) -> Option<NeuronId> {
        self.neuron_parents.get(&neuron_id).copied()
    }

    /// Get the temporal level for a neuron (0 = sensory, 1+ = temporal pattern).
    pub fn get_neuron_temporal_level(&self, neuron_id: NeuronId) -> Option<Level> {
        self.neuron_temporal_levels.get(&neuron_id).copied()
    }

    /// Get the spatial level for a neuron (0 = sensory/base or never spatially registered, 1+ = spatial correction).
    pub fn get_neuron_spatial_level(&self, neuron_id: NeuronId) -> Level {
        self.neuron_spatial_levels.get(&neuron_id).copied().unwrap_or(0)
    }

    /// Cumulative count of spatial corrections minted since brain start (or last hard reset).
    pub fn get_spatial_correction_count(&self) -> u64 {
        self.spatial_corrections_minted
    }

    /// Cumulative count of spatial children retired by the delete pass since brain start.
    pub fn get_spatial_deletion_count(&self) -> u64 {
        self.spatial_corrections_deleted
    }

    /// Count of neurons currently sitting above the base level in the spatial hierarchy.
    /// Counts unique correction neurons rather than mint events, so a correction that's later
    /// deleted via cascade doesn't show up.
    pub fn count_active_spatial_corrections(&self) -> usize {
        self.neuron_spatial_levels.values().filter(|&&lvl| lvl > 0).count()
    }

    /// Per-level count of correction neurons in the spatial hierarchy. Returns Vec where
    /// index = spatial level, value = count of alive neurons at that level. Level 0 is sensory
    /// (not corrections) and is omitted; the returned vec starts at level 1.
    pub fn spatial_level_counts(&self) -> Vec<u32> {
        let mut counts: Vec<u32> = Vec::new();
        for &lvl in self.neuron_spatial_levels.values() {
            if lvl == 0 { continue; }
            let idx = (lvl - 1) as usize;
            if idx >= counts.len() { counts.resize(idx + 1, 0); }
            counts[idx] += 1;
        }
        counts
    }

    /// Per-level count of PAID correction neurons. Under the womb every born pattern is paid — the
    /// unpaid hypotheses live in the womb as embryos, not the routing table — so this equals
    /// [spatial_level_counts]. Retained for API stability with the diagnostics harness.
    pub fn spatial_level_paid_counts(&self) -> Vec<u32> {
        self.spatial_level_counts()
    }

    /// Get the coordinate for a base (sensory/action) neuron.
    pub fn get_neuron_coordinate(&self, neuron_id: NeuronId) -> Option<&Coordinate> {
        self.base_neurons.get(&neuron_id).map(|b| &b.coordinate)
    }

    // ── Inspection ──────────────────────────────────────────────────────────

    /// Inspection helper: get the stored context entries for a child pattern
    /// (looked up via its parent's routing table). Returns Vec of
    /// (context_neuron_id, distance, strength) tuples, or None if the
    /// pattern_id has no recorded parent or the parent doesn't own it.
    pub fn get_pattern_context_entries(&self, pattern_id: NeuronId) -> Option<Vec<(NeuronId, Distance, f64)>> {
        let parent_id = self.get_neuron_parent(pattern_id)?;
        let r = self.route_neuron(parent_id);
        self.region_list[r].get_child_context_entries(parent_id, pattern_id)
    }

    /// Inspection: dump a neuron's outgoing connections (distance, target, strength, reward).
    pub fn get_neuron_connections(&self, neuron_id: NeuronId) -> Option<Vec<(Distance, NeuronId, f64, f64)>> {
        let r = self.route_neuron(neuron_id);
        self.region_list[r].get_neuron_connections(neuron_id)
    }

    // ── Inference results ───────────────────────────────────────────────────

    /// Per-frame inference performance bundle for diagnostics. Each item carries
    /// everything track_inference_performance needs (correctness flag, first-actual
    /// coord, pre-resolved reward) so the diagnostics call stays single-arg.
    pub fn get_inference_results(
        &self,
        active_neuron_ids: &FxHashSet<NeuronId>,
        inferred_neurons: &[InferredNeuron],
        rewards: &FxHashMap<ChannelId, Reward>,
    ) -> Vec<InferenceResultItem> {

        // per-channel first observed event coordinate (used to pair mispredictions)
        let mut first_actual_by_channel: FxHashMap<ChannelId, Coordinate> = FxHashMap::default();
        for &neuron_id in active_neuron_ids {
            let base = match self.base_neurons.get(&neuron_id) {
                Some(b) => b,
                None => continue,
            };
            if base.neuron_type != NeuronType::Event { continue; }
            first_actual_by_channel.entry(base.channel_id).or_insert_with(|| base.coordinate.clone());
        }

        let mut results = Vec::with_capacity(inferred_neurons.len());
        for inf in inferred_neurons {
            let neuron_id = inf.neuron_id;
            let neuron_type = match self.get_neuron_type(neuron_id) {
                Some(t) => t,
                None => continue,
            };
            let channel_id = match self.get_neuron_channel_id(neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let coordinate = self.get_neuron_coordinate(neuron_id).cloned();
            let actual_coord = if neuron_type == NeuronType::Event {
                first_actual_by_channel.get(&channel_id).cloned()
            } else {
                None
            };
            let is_correct = neuron_type == NeuronType::Event && active_neuron_ids.contains(&neuron_id);
            let reward = if neuron_type == NeuronType::Action {
                rewards.get(&channel_id).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            results.push(InferenceResultItem {
                neuron_type: match neuron_type {
                    NeuronType::Event => InferenceType::Event,
                    NeuronType::Action => InferenceType::Action,
                },
                is_correct,
                channel_id,
                coordinate,
                actual_coord,
                reward,
            });
        }
        results
    }

    // ── Channel registration ────────────────────────────────────────────────

    /// Register a channel spec with the brain. Allocates a channel ID and per-dimension IDs,
    /// populates the name↔id maps, registers each dimension with the quantizer, and pre-creates
    /// action neurons for action dims with explicit bucket IDs. Returns the allocated channel ID
    /// plus a dimension_ids lookup keyed by dim name.
    pub fn register_channel_spec(
        &mut self,
        name: &str,
        dimensions: Vec<DimSpecInput>,
        learn_action_sequences: bool,
    ) -> ChannelRegistration {
        if name.is_empty() { panic!("Thalamus: channel spec is missing required name"); }
        if self.channel_name_to_id.contains_key(name) {
            panic!("Thalamus: channel \"{}\" already registered", name);
        }

        // allocate channel id
        let channel_id = self.next_channel_id;
        self.next_channel_id += 1;

        // validate dimension specs
        for d in &dimensions {
            if d.name.is_empty() { panic!("Thalamus: dim on channel \"{}\" is missing a name", name); }
            if d.kind != DimKind::Input && d.kind != DimKind::Action {
                panic!("Thalamus: dim \"{}\" has invalid kind", d.name);
            }
        }

        // build the stored spec with allocated dimension ids baked in
        let mut dimension_ids = FxHashMap::default();
        let mut stored_dims = Vec::with_capacity(dimensions.len());
        for d in &dimensions {
            let dim_id = self.next_dimension_id;
            self.next_dimension_id += 1;
            dimension_ids.insert(d.name.clone(), dim_id);
            stored_dims.push(DimSpec {
                id: dim_id,
                name: d.name.clone(),
                kind: d.kind.clone(),
                resolution: d.resolution,
                mode: d.mode.clone(),
                boundaries: d.boundaries.clone(),
                actions: d.actions.clone(),
                default_action: d.default_action,
                warmup_samples: d.warmup_samples,
            });
        }

        let stored_spec = ChannelSpec {
            name: name.to_string(),
            dimensions: stored_dims,
            learn_action_sequences,
        };

        self.channel_specs.insert(channel_id, stored_spec.clone());

        // populate name↔id maps
        self.channel_name_to_id.insert(name.to_string(), channel_id);
        self.channel_id_to_name.insert(channel_id, name.to_string());

        // register dimensions with the quantizer and create action neurons in columns
        self.register_dimensions(&stored_spec);
        self.register_action_neurons(channel_id, &stored_spec);

        if self.debug { println!("Registered channel spec {} \"{}\" ({} dimensions)", channel_id, name, stored_spec.dimensions.len()); }
        ChannelRegistration { channel_id, dimension_ids }
    }

    /// Store dimension specs, register with quantizer, and populate dim name↔id maps.
    fn register_dimensions(&mut self, stored_spec: &ChannelSpec) {
        for dim in &stored_spec.dimensions {
            if self.dimension_specs.contains_key(&dim.id) {
                panic!("Thalamus: dimension {} already registered (channel \"{}\")", dim.id, stored_spec.name);
            }
            self.dimension_specs.insert(dim.id, dim.clone());
            self.dimension_name_to_id.insert(dim.name.clone(), dim.id);
            self.dimension_id_to_name.insert(dim.id, dim.name.clone());

            // determine quantizer mode from string
            let mode = match dim.mode.as_deref() {
                Some("static") => QuantizeMode::Static,
                Some("dynamic") => QuantizeMode::Dynamic,
                _ => QuantizeMode::Passthrough,
            };
            self.quantizer.register_dimension(dim.id, dim.resolution, mode, dim.boundaries.clone(), dim.warmup_samples);
        }
    }

    /// Pre-create action neurons for action dims so exploration can find them.
    fn register_action_neurons(&mut self, channel_id: ChannelId, stored_spec: &ChannelSpec) {
        let mut action_neurons: Vec<NeuronId> = self.channel_actions.get(&channel_id).cloned().unwrap_or_default();
        let mut new_neuron_specs = Vec::new();

        for dim in &stored_spec.dimensions {

            // only processing action dimensions
            if dim.kind != DimKind::Action { continue; }
            let action_buckets = match &dim.actions {
                Some(a) => a,
                None => continue,
            };

            // all action dimensions should have defaults
            if dim.default_action.is_none() {
                panic!("Invalid action dimension without default: {:?}", dim.name);
            }

            // allocate action neuron ids — push in registration order, skip duplicates
            for &bucket_id in action_buckets {
                let lookup = self.get_neuron_id_for_point(
                    &Coordinate { dim_id: dim.id, bucket_id },
                    channel_id,
                    NeuronType::Action,
                );
                if !action_neurons.contains(&lookup.id) {
                    action_neurons.push(lookup.id);
                }
                if lookup.is_new {
                    // Action neurons never die themselves, but their hosted correction children must
                    // decay — stamp the base-neuron forget rate, not 0.0 (which leaves them immortal).
                    new_neuron_specs.push(NeuronCreateSpec { id: lookup.id, forget_rate: self.base_neuron_forget_rate(), connections: None });
                }
            }

            // default action neuron already created in the actions loop above — get its id and save it
            let default_bucket = dim.default_action.unwrap();
            let default_lookup = self.get_neuron_id_for_point(
                &Coordinate { dim_id: dim.id, bucket_id: default_bucket },
                channel_id,
                NeuronType::Action,
            );
            self.channel_default_actions.insert(channel_id, default_lookup.id);
        }

        // index the action neurons
        if !action_neurons.is_empty() {
            self.channel_actions.insert(channel_id, action_neurons);
        }

        // sync action sets to all regions/columns so per-frame calls don't reach back to Thalamus
        for region in &mut self.region_list {
            region.update_action_sets(&self.channel_actions, &self.channel_default_actions);
        }

        // create action neurons in columns
        if !new_neuron_specs.is_empty() {
            self.create_neurons(&new_neuron_specs);
        }
    }

    /// Iterate all channel ids registered via channel specs.
    pub fn get_channel_ids(&self) -> Vec<ChannelId> {
        self.channel_specs.keys().copied().collect()
    }

    /// Get stored channel spec by ID. Test-only.
    #[cfg(test)]
    pub fn get_channel_spec(&self, channel_id: ChannelId) -> Option<&ChannelSpec> {
        self.channel_specs.get(&channel_id)
    }

    /// Translate an id-form coordinate to name-form.
    pub fn coordinate_id_to_name(&self, coordinate: &Coordinate) -> Option<(String, i32)> {
        self.dimension_id_to_name.get(&coordinate.dim_id)
            .map(|name| (name.clone(), coordinate.bucket_id))
    }

    // ── Level processing ────────────────────────────────────────────────────

    /// Process one level: aggregate the level view, dispatch processFrame (Op-3),
    /// and extract activations for the next level. Returns deferred work (neuron
    /// creation specs, raw dispatch results) alongside activations and votes.
    /// The caller accumulates deferred work across levels and flushes it once
    /// after the loop via apply_level_results.
    /// Spatial sweep dispatch for one level. Builds the spatial level_context, runs the d=0
    /// co-activation work on every active neuron, collects activations + votes.
    /// The caller accumulates deferred work across levels and flushes it once after the loop
    /// via apply_level_results.
    pub fn process_spatial_level(
        &mut self,
        level: Level,
        level_neuron_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
    ) -> SpatialLevelResult {

        let mut orchestration = OrchestrationTimings::default();

        // Aggregate the per-neuron work list and build the shared co-activation for this level.
        // Non-learning mode still builds it so pattern activation can run.
        let t = std::time::Instant::now();
        let (work_list, level_context) = self.get_spatial_level_tasks(level_neuron_ids, new_error_pattern_ids);
        orchestration.get_level_tasks = t.elapsed().as_secs_f64();

        // Dispatch the per-neuron frame pass — the only cross-region round-trip in the level loop.
        // The level's actives feed both roles: the co-activation identifies each neuron, and the
        // same set is what it predicts.
        let t = std::time::Instant::now();
        let results = self.dispatch_spatial_frame(&work_list, &level_context, new_error_pattern_ids, level_neuron_ids, frame_number);
        orchestration.dispatch_frame = t.elapsed().as_secs_f64();

        // Extract this level's pattern activations inline; they feed the next level up.
        let t = std::time::Instant::now();
        let mut activations = Vec::new();
        for result in &results {
            self.collect_spatial_activations(result, &mut activations);
        }
        orchestration.collect_activations = t.elapsed().as_secs_f64();

        if self.debug && !activations.is_empty() {
            let detail: Vec<String> = activations.iter()
                .map(|a| format!("parent={}, age={}, pattern={}", a.parent_id, a.age, a.pattern_id))
                .collect();
            println!("Spatial level {}: {} activations {}", level, activations.len(), detail.join("; "));
        }

        SpatialLevelResult { activations, results, orchestration }
    }

    /// Temporal sweep dispatch for one level. Builds the temporal level_context (mints temporal
    /// corrections for prior-frame vote misses), runs d>0 work per (neuron, age), collects votes.
    /// The caller accumulates deferred work across levels and flushes it once after the loop
    /// via apply_level_results.
    pub fn process_temporal_level(
        &mut self,
        level: Level,
        level_neurons: &mut FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        memory_depth: u32,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
    ) -> ProcessLevelResult {
        let mut orchestration = OrchestrationTimings::default();

        // Aggregate per-neuron work, build the shared level context, allocate error pattern specs.
        // Non-learning mode still builds the level context so pattern activation can run.
        // It skips error-correction allocation and per-age error feedback recording.
        let t = std::time::Instant::now();
        let (neurons, level_context, new_neuron_specs) =
            self.prepare_temporal_neurons(level, level_neurons, sensory_neurons, rewards, frame_number, new_error_pattern_ids);
        orchestration.get_level_tasks = t.elapsed().as_secs_f64();

        // Op-3: dispatch processTemporalFrame — the only cross-region round-trip in the level loop.
        let t = std::time::Instant::now();
        let results = self.dispatch_temporal_neurons(&neurons, memory_depth, &level_context, new_error_pattern_ids, frame_number);
        orchestration.dispatch_frame = t.elapsed().as_secs_f64();

        // extract activations inline — needed to feed the next level
        let t = std::time::Instant::now();
        let mut activations = Vec::new();
        for result in &results {
            self.collect_activations(result, &mut activations);
        }
        orchestration.collect_activations = t.elapsed().as_secs_f64();

        // collect votes inline — needed for consensus after the loop
        let t = std::time::Instant::now();
        Self::clear_stale_state(level_neurons);
        let mut votes = Vec::new();
        for result in &results {
            Self::collect_votes(result, level_neurons, &mut votes);
        }
        orchestration.collect_votes = t.elapsed().as_secs_f64();

        if self.debug && !activations.is_empty() {
            let detail: Vec<String> = activations.iter()
                .map(|a| format!("parent={}, age={}, pattern={}", a.parent_id, a.age, a.pattern_id))
                .collect();
            println!("Level {}: {} activations {}", level, activations.len(), detail.join("; "));
        }

        ProcessLevelResult { activations, votes, neuron_specs: new_neuron_specs, results, orchestration }
    }

    /// Flush deferred work accumulated across all levels in the level loop.
    /// Runs once per frame after the loop exits, replacing L per-level dispatches
    /// with a single batch for each operation.
    ///
    /// Op-4: Create error pattern neurons (batch across all levels)
    /// Op-5: Dispatch contextRef updates (batch across all levels)
    pub fn apply_level_results(&mut self, neuron_specs: &[NeuronCreateSpec], dispatch_results: &[Vec<ColumnProcessResult>]) {

        // Op-4: batch-create error pattern neurons accumulated across levels
        if !neuron_specs.is_empty() {
            self.create_neurons(neuron_specs);
        }

        // Op-5: batch contextRef updates across levels, then dispatch once
        let mut context_ref_updates: FxHashMap<NeuronId, Vec<TemporalContextRefUpdate>> = FxHashMap::default();
        for results in dispatch_results {
            for result in results {
                Self::collect_context_ref_updates(result, &mut context_ref_updates);
            }
        }

        if !context_ref_updates.is_empty() {
            let update_batch: Vec<(NeuronId, Vec<TemporalContextRefUpdate>)> = context_ref_updates.into_iter().collect();
            self.dispatch_temporal_context_ref_updates(&update_batch);
        }
    }

    // ── Spatial error pass (1c) ─────────────────────────────────────────────

    /// Settle this frame's spatial births. A request means an embryo in the neuron's womb covered
    /// its price and asks for the pattern it represents to be born. Every born pattern is paid — the
    /// womb owns pricing and clustering entirely — so this pass owns only what a neuron cannot decide
    /// locally: the subsumption filter, id allocation, and cross-neuron wiring.
    ///
    /// The CONTEXT of a born correction is the embryo's converged center, drawn from the parent's OWN
    /// level — for an Lk parent, the L(k+1) correction's context_entries are level-k co-actives. This
    /// is what lets the hierarchy grow: an L1 prediction failure births an L2 whose context is L1
    /// neighbors, so L2 will fire next frame when L1 patterns recur in a similar L1-neighborhood.
    pub fn create_spatial_corrections(
        &mut self,
        dispatch_results: &[Vec<crate::column::SpatialColumnResult>],
        subsumed_neurons: &FxHashSet<NeuronId>,
        fired_neurons: &FxHashSet<NeuronId>,
    ) -> (Vec<NeuronCreateSpec>, Vec<SpatialInstallOp>) {

        // Corrections born this frame are returned as creation specs plus install ops for their parents.
        let mut new_specs = Vec::new();
        let mut install_ops = Vec::new();

        // The frame's co-activation, split by level: a newborn's event connections are seeded from
        // the level it is born INTO, so each birth reads the bucket one above its parent's level.
        let actives_by_level = self.bucket_fired_by_spatial_level(fired_neurons);

        // process the birth requests of the spatial dispatch and create the corrections.
        for column_result in dispatch_results.iter().flatten() {
            let Some(request) = &column_result.correction_request else { continue };

            // A neuron represented by a fired higher-level pattern is subsumed — not to be corrected.
            // A subsumed neuron casts no votes and requests nothing, so this is a safety net.
            if subsumed_neurons.contains(&column_result.parent_id) { continue; }

            // Create one correction pattern for the add pass's birth request.
            let parent_id = column_result.parent_id;
            let parent_level = self.get_neuron_spatial_level(parent_id);

            // Phase 1 cap (docs/algorithm.md, "Implementation plan": capped at one level so the
            // recursion is not a variable yet): only base (level-0) neurons mint. Level-1 patterns
            // therefore never acquire children, so the hierarchy settles at depth 2 and the level-0
            // configuration loop is what gets measured. Lift this cap when contraction lands (Phase 2).
            if parent_level != 0 { continue; }

            let empty = FxHashSet::default();
            let level_actives = actives_by_level.get(&(parent_level + 1)).unwrap_or(&empty);
            self.create_spatial_correction(parent_id, parent_level, level_actives, &request.context_neighbors, &mut new_specs, &mut install_ops);
        }

        (new_specs, install_ops)
    }

    /// Retire the children the one test's delete pass flagged this frame. Each neuron already removed
    /// the child from its own routing table and index; here the thalamus releases the pattern neuron
    /// and scrubs its cross-neuron references through the delete cascade. Safe to run before the apex
    /// handoff: a retired child did not serve this frame (only one entry serves per neuron per frame),
    /// so it is not in this frame's fired/apex set. Returns the ids actually released.
    pub fn delete_spatial_children(
        &mut self,
        dispatch_results: &[Vec<crate::column::SpatialColumnResult>],
        current_frame: FrameNumber,
    ) -> Vec<NeuronId> {
        let mut to_delete: Vec<NeuronId> = Vec::new();
        for column_result in dispatch_results.iter().flatten() {
            to_delete.extend(column_result.deleted_children.iter().copied());
        }
        if to_delete.is_empty() { return Vec::new(); }
        let deleted = self.delete_patterns(&to_delete, current_frame);
        self.spatial_corrections_deleted += deleted.len() as u64;
        deleted
    }

    /// Bucket this frame's fired neurons by spatial level — the per-level co-activation a newborn
    /// seeds its event connections from.
    fn bucket_fired_by_spatial_level(&self, fired_neurons: &FxHashSet<NeuronId>) -> FxHashMap<Level, FxHashSet<NeuronId>> {
        let mut by_level: FxHashMap<Level, FxHashSet<NeuronId>> = FxHashMap::default();
        for &neuron_id in fired_neurons {
            by_level.entry(self.get_neuron_spatial_level(neuron_id)).or_default().insert(neuron_id);
        }
        by_level
    }

    /// Create one correction pattern for a paid-off embryo giving birth. `level_actives` is the
    /// co-activation at the newborn's own level — the events it is minted to predict.
    fn create_spatial_correction(
        &mut self,
        parent_id: NeuronId,
        parent_level: Level,
        level_actives: &FxHashSet<NeuronId>,
        context_neighbors: &[NeuronId],
        new_specs: &mut Vec<NeuronCreateSpec>,
        install_ops: &mut Vec<SpatialInstallOp>,
    ) -> NeuronId {

        // The correction lives one level deeper than the neuron it corrects.
        let spec = self.allocate_spatial_pattern_neuron(parent_level + 1, parent_id, level_actives);
        new_specs.push(NeuronCreateSpec {
            id: spec.id,
            forget_rate: spec.forget_rate,
            connections: Some(spec.connections),
        });

        // Installation wires the embryo's center in as the correction's context on its parent.
        install_ops.push(SpatialInstallOp {
            parent_id,
            pattern_id: spec.id,
            context_neuron_ids: context_neighbors.to_vec(),
        });
        self.spatial_corrections_minted += 1;
        spec.id
    }

    /// Install minted spatial corrections into their parents' routing tables. Each install adds
    /// the child pattern as an entry on the parent neuron and emits ContextRefUpdates so the
    /// target neurons know the parent references them. Death frames returned from add_pattern are
    /// registered in the death ledger.
    /// Corrections are NOT activated this frame — they're routing-table entries that match and fire
    /// on the next frame's spatial sweep (per spatial-processing.md §5.1).
    pub fn install_spatial_corrections(&mut self, install_ops: Vec<SpatialInstallOp>, frame_number: FrameNumber) {
        if install_ops.is_empty() { return; }

        // Bucket by owning region (route on parent_id — that's whose routing table changes).
        let mut by_region: Vec<Vec<SpatialInstallOp>> = (0..self.regions).map(|_| Vec::new()).collect();
        for op in install_ops {
            let r = self.route_neuron(op.parent_id);
            by_region[r].push(op);
        }

        // Dispatch and collect deaths + spatial context refs as flat lists.
        let mut all_deaths: Vec<(NeuronId, FrameNumber)> = Vec::new();
        let mut all_context_refs: Vec<SpatialContextRefUpdate> = Vec::new();
        for (r, region_ops) in by_region.into_iter().enumerate() {
            if region_ops.is_empty() { continue; }
            let result = self.region_list[r].install_spatial_corrections(region_ops, frame_number);
            all_deaths.extend(result.deaths);
            all_context_refs.extend(result.context_ref_updates);
        }

        for (id, df) in all_deaths {
            self.register_death(id, df);
        }

        // Bucket the flat list by target neuron_id at the dispatch boundary, then route.
        // Parallel to how `apply_level_results` buckets temporal updates before dispatching.
        if !all_context_refs.is_empty() {
            let mut by_target: FxHashMap<NeuronId, Vec<SpatialContextRefUpdate>> = FxHashMap::default();
            for upd in all_context_refs {
                by_target.entry(upd.neuron_id).or_insert_with(Vec::new).push(upd);
            }
            let update_batch: Vec<(NeuronId, Vec<SpatialContextRefUpdate>)> = by_target.into_iter().collect();
            self.dispatch_spatial_context_ref_updates(&update_batch);
        }
    }

    /// Spatial counterpart of `dispatch_temporal_context_ref_updates` — routes SpatialContextRefUpdates
    /// to the columns that own each target neuron, then dispatches in parallel.
    fn dispatch_spatial_context_ref_updates(&mut self, update_batch: &[(NeuronId, Vec<SpatialContextRefUpdate>)]) {
        let mut by_region: Vec<Vec<(NeuronId, Vec<SpatialContextRefUpdate>)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for (target_id, updates) in update_batch {
            let r = self.route_neuron(*target_id);
            by_region[r].push((*target_id, updates.clone()));
        }
        for (r, region_updates) in by_region.into_iter().enumerate() {
            if region_updates.is_empty() { continue; }
            self.region_list[r].update_spatial_context_refs(&region_updates);
        }
    }

    /// SPATIAL: walk the active neurons at this level and build the shared SpatialContext.
    /// Spatial corrections are created in a separate pass after the
    /// sweep, NOT here — so no corrections, no per-age error feedback, no new_neuron_specs.
    /// Action neurons are NOT skipped — spatial co-activation includes everything that fired.
    /// Build the spatial work list and the shared observed co-activation for one level.
    fn get_spatial_level_tasks(
        &mut self,
        level_neuron_ids: &FxHashSet<NeuronId>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
    ) -> (Vec<NeuronId>, SpatialContext) {

        let mut work_list = Vec::new();
        let mut level_context = SpatialContext::new();

        // Every active neuron contributes to the shared co-activation.
        // Freshly-created error patterns contribute context only: they have no children, history,
        // or votes in their birth frame, so they skip the dispatch.
        for &neuron_id in level_neuron_ids {
            level_context.add_neuron(neuron_id, 1.0);
            if new_error_pattern_ids.contains(&neuron_id) { continue; }
            work_list.push(neuron_id);
        }

        (work_list, level_context)
    }

    /// Build one `TemporalNeuron` per active neuron at this level for the dispatch, contributing each
    /// to the shared TemporalContext. On a learning pass each carries its `TemporalLearningWork`; a
    /// frozen pass carries none and never builds the neighbor actives.
    fn prepare_temporal_neurons(
        &mut self,
        level: Level,
        level_neurons: &FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
    ) -> (Vec<TemporalNeuron>, TemporalContext, Vec<NeuronCreateSpec>) {
        let mut neurons = Vec::new();
        let mut level_context = TemporalContext::new();
        let mut new_neuron_specs = Vec::new();

        // Neighbor actives are for connection learning only — decorate the age-0 set once, and only
        // when learning, so a frozen pass never pays for it.
        let decorated_actives = if self.learning {
            self.decorate_temporal_actives(&sensory_neurons[0], rewards.first())
        } else {
            Vec::new()
        };

        // Snapshot the entries first — the loop borrows self mutably (correction minting) while reading them.
        let neuron_entries: Vec<(NeuronId, FxHashMap<Distance, LevelAgeState>)> = level_neurons
            .iter()
            .map(|(&nid, ages)| (nid, ages.clone()))
            .collect();
        for (neuron_id, age_states) in &neuron_entries {

            // skip action neurons for learning or contexts if the channel learns without them
            if self.skip_action_neuron(*neuron_id) { continue; }

            // A new error pattern has no history or votes in its birth frame — it only feeds context.
            if new_error_pattern_ids.contains(neuron_id) {
                for (&age, _) in age_states {
                    if age > 0 { level_context.add_neuron(*neuron_id, age, 1.0); }
                }
                continue;
            }

            // Feed level_context and, when learning, mint this neuron's corrections + gather its feedback.
            let (corrections, error_feedback) = self.get_temporal_level_corrections(
                *neuron_id, level, &mut level_context, age_states, sensory_neurons, rewards, frame_number,
            );

            // Defer creation to the Op-4 batch, and mark each new pattern so it's skipped in its own birth frame.
            for correction in &corrections {
                new_neuron_specs.push(NeuronCreateSpec {
                    id: correction.pattern_id,
                    forget_rate: correction.forget_rate,
                    connections: Some(correction.connections.clone()),
                });
                new_error_pattern_ids.insert(correction.pattern_id);
            }

            // convert LevelAgeState to neuron::AgeState for dispatch
            let neuron_age_states: FxHashMap<Distance, AgeState> = age_states.iter()
                .map(|(&age, state)| (age, AgeState { activated_pattern_id: state.activated_pattern_id }))
                .collect();
            
            let learning_work = self.learning.then(||
                self.temporal_learning_work(*neuron_id, &decorated_actives, &corrections, error_feedback));
            neurons.push(TemporalNeuron { neuron_id: *neuron_id, age_states: neuron_age_states, learning_work });
        }

        (neurons, level_context, new_neuron_specs)
    }

    /// One neuron's learning work: its neighbor actives (filtered to the channels it learns from), the
    /// corrections to install, and the accuracy feedback to record.
    fn temporal_learning_work(&self, neuron_id: NeuronId, decorated_actives: &[ActiveNeuron], corrections: &[CorrectionSpec], error_feedback: Vec<ErrorFeedback>) -> TemporalLearningWork {
        let parent_channel = self.get_neuron_channel_id(neuron_id).unwrap_or(0);
        let neighbors = decorated_actives.iter()
            .filter(|n| self.is_temporal_neighbor_channel(parent_channel, n.channel_id))
            .cloned()
            .collect();
        let corrections = corrections.iter().map(|c| Correction {
            pattern_id: c.pattern_id,
            age: c.age,
            context_entries: c.context_entries.clone(),
        }).collect();
        TemporalLearningWork { neighbors, corrections, error_feedback }
    }

    /// Decorate active sensory neurons with channel, dimension, and pre-resolved reward (reward matters
    /// only for action neurons) — the neighbor actives connection learning reads.
    fn decorate_temporal_actives(&self, active_ids: &FxHashSet<NeuronId>, rewards: Option<&FxHashMap<ChannelId, Reward>>) -> Vec<ActiveNeuron> {
        active_ids.iter().map(|&neuron_id| {
            let (channel_id, neuron_type) = self.base_neurons.get(&neuron_id)
                .map(|b| (b.channel_id, Some(b.neuron_type)))
                .unwrap_or((0, None));
            let reward = match (rewards, neuron_type) {
                (Some(r), Some(NeuronType::Action)) => r.get(&channel_id).copied().unwrap_or(0.0),
                _ => 0.0,
            };
            ActiveNeuron { id: neuron_id, channel_id, reward }
        }).collect()
    }

    /// For a single active temporal neuron: add its age>0 entries to the shared level_context and
    /// create error-correction pattern neurons for ages whose previous votes mismatched reality.
    /// Non-learning mode still populates level_context (so pattern recognition at the next level
    /// has its context). It skips everything else.
    fn get_temporal_level_corrections(
        &mut self,
        neuron_id: NeuronId,
        level: Level,
        level_context: &mut TemporalContext,
        age_states: &FxHashMap<Distance, LevelAgeState>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
    ) -> (Vec<CorrectionSpec>, Vec<ErrorFeedback>) {
        let mut corrections = Vec::new();
        let mut error_feedback = Vec::new();

        // Resolve the parent's channel once — used for neighbor filtering of actuals, context
        // entries, and (inside allocate_temporal_pattern_neuron) the pre-wired connections of any
        // minted pattern. Channels without a registered neighbor list fall through to all-pairs.
        let parent_channel = self.get_neuron_channel_id(neuron_id).unwrap_or(0);

        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            let state = &age_states[&age];

            // Temporal context entries are at strictly positive ages — older neurons predicting the parent.
            if age > 0 { level_context.add_neuron(neuron_id, age, 1.0); }

            // Non-learning mode is done after level_context population — no accuracy stats, no error pattern allocation.
            if !self.learning { continue; }

            // skip if context is empty. empty context patterns can never match anything in future frames.
            // they would just keep regenerating useless siblings. we do this before vote error evaluation so that
            // empty context frames don't pollute the neuron's error-stats window either — Welford would
            // otherwise see misses the neuron had no chance to do better on, inflating future fire thresholds.
            if state.context.as_ref().map_or(true, |c| c.is_empty()) { continue; }

            // Build the actuals set for vote-error evaluation, filtered to the parent's neighbor
            // channels. Votes that hit a target outside the neighbor graph shouldn't have been cast
            // (connections are also neighbor-filtered at learn/allocate time) — but filter defensively.
            let actuals_filtered: FxHashSet<NeuronId> = sensory_neurons[0].iter()
                .copied()
                .filter(|&id| {
                    let target_ch = self.get_neuron_channel_id(id).unwrap_or(0);
                    self.is_temporal_neighbor_channel(parent_channel, target_ch)
                })
                .collect();

            // evaluate the prior-frame vote at this age (if any) and record feedback
            let result = match self.evaluate_vote_error(age, state, &actuals_filtered, frame_number) {
                Some(r) => r,
                None => continue,
            };
            error_feedback.push(ErrorFeedback { age, error_rate: result.error_rate });

            // skip if the error doesn't cross the dynamic threshold
            if !result.fire { continue; }

            // allocate an error correction pattern to be created after level processing
            let spec = self.allocate_temporal_pattern_neuron(level + 1, neuron_id, age, sensory_neurons, rewards);

            // Filter the correction's context_entries to neighbor channels — the new pattern only
            // matches against and references neighbor-channel neurons in its routing context.
            let context_entries_filtered: Vec<ContextRefEntry> = state.context.clone().unwrap()
                .into_iter()
                .filter(|e| {
                    let target_ch = self.get_neuron_channel_id(e.neuron_id).unwrap_or(0);
                    self.is_temporal_neighbor_channel(parent_channel, target_ch)
                })
                .collect();

            corrections.push(CorrectionSpec {
                pattern_id: spec.id,
                forget_rate: spec.forget_rate,
                connections: spec.connections,
                age,
                context_entries: context_entries_filtered,
            });
        }

        (corrections, error_feedback)
    }

    /// Spatial Op-3 dispatch — no rewards, no per-task channel-reward resolution.
    fn dispatch_spatial_frame(
        &mut self,
        work_list: &[NeuronId],
        level_context: &SpatialContext,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        inference_events: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<crate::column::SpatialColumnResult> {

        // What a neuron predicts and what identifies it are one and the same since d=0 (spatial processing)
        // so both are cut to the same level-scaled neighborhood — they differ only in payload:
        // inference carries each neighbor's position and reward, context carries only its strength.
        let inference_neurons = self.decorate_inference_neurons(inference_events);
        let inference_neighbors = self.select_inference_neighbors(work_list, &inference_neurons);
        let context_neighbors = self.select_context_neighbors(work_list, level_context);

        self.dispatch_to_regions(work_list, inference_neighbors, context_neighbors, new_error_pattern_ids, frame_number)
    }

    /// Resolve each active event's position once, up front — every neighbor list below reads it.
    fn decorate_inference_neurons(&self, inference_events: &FxHashSet<NeuronId>) -> Vec<ActiveNeuron> {
        inference_events.iter().map(|&neuron_id| {

            // Pattern neurons resolve here too: they inherit their parent's coordinate at birth.
            let channel_id = self.base_neurons.get(&neuron_id)
                .map(|b| b.channel_id)
                .unwrap_or(0);

            // Co-activation carries no reward semantics — reward lives on the payload channels.
            ActiveNeuron { id: neuron_id, channel_id, reward: 0.0 }
        }).collect()
    }

    /// Per neuron: the events it predicts and learns edges toward, cut to its own neighborhood.
    fn select_inference_neighbors(
        &self,
        work_list: &[NeuronId],
        inference_neurons: &[ActiveNeuron],
    ) -> Vec<Vec<ActiveNeuron>> {

        // Bucketing once lets each neuron walk whichever side is smaller, instead of every neuron
        // scanning the full set — that scan was quadratic in the active count.
        let by_channel = Self::bucket_neurons_by_channel(inference_neurons);
        work_list.iter()
            .map(|&neuron_id| match self.get_neighbor_channels(neuron_id) {
                Some(channels) if channels.len() < inference_neurons.len() =>
                    Self::gather_neurons_from_channels(channels, &by_channel),
                _ => self.scan_neurons_for_neighbors(neuron_id, inference_neurons),
            })
            .collect()
    }

    /// Group the frame's active neurons by channel, so a neighborhood walk can pull whole buckets.
    fn bucket_neurons_by_channel(inference_neurons: &[ActiveNeuron]) -> FxHashMap<ChannelId, Vec<ActiveNeuron>> {
        let mut by_channel: FxHashMap<ChannelId, Vec<ActiveNeuron>> = FxHashMap::default();
        for neuron in inference_neurons {
            by_channel.entry(neuron.channel_id).or_insert_with(Vec::new).push(neuron.clone());
        }
        by_channel
    }

    /// The narrow walk: pull each neighbor channel's bucket. Chosen when the neighborhood is smaller than the active set.
    fn gather_neurons_from_channels(
        channels: &FxHashSet<ChannelId>,
        by_channel: &FxHashMap<ChannelId, Vec<ActiveNeuron>>,
    ) -> Vec<ActiveNeuron> {
        channels.iter()
            .filter_map(|channel| by_channel.get(channel))
            .flat_map(|bucket| bucket.iter().cloned())
            .collect()
    }

    /// The wide walk: scan the active set and keep this neuron's neighbors. Chosen when the
    /// neighborhood is wider than the active set, or unrestricted.
    fn scan_neurons_for_neighbors(&self, neuron_id: NeuronId, inference_neurons: &[ActiveNeuron]) -> Vec<ActiveNeuron> {
        let (channel, level) = self.get_neighborhood_key(neuron_id);
        inference_neurons.iter()
            .filter(|neuron| self.is_spatial_neighbor_channel(channel, level, neuron.channel_id))
            .cloned()
            .collect()
    }

    /// Per neuron: the shared co-activation cut to its own neighborhood, minus itself. Without the
    /// cut every non-neighbor co-active counts as novel, dragging match scores below any sane bar —
    /// spatial matching would never fire.
    fn select_context_neighbors(&self, work_list: &[NeuronId], level_context: &SpatialContext) -> Vec<SpatialContext> {
        let by_channel = self.bucket_context_by_channel(level_context);
        work_list.iter()
            .map(|&neuron_id| match self.get_neighbor_channels(neuron_id) {
                Some(channels) if channels.len() < level_context.size() =>
                    Self::gather_context_from_channels(neuron_id, channels, &by_channel),
                _ => self.scan_context_for_neighbors(neuron_id, level_context),
            })
            .collect()
    }

    /// Group the co-activation by channel — the context-side counterpart of [bucket_neurons_by_channel].
    fn bucket_context_by_channel(&self, level_context: &SpatialContext) -> FxHashMap<ChannelId, Vec<(NeuronId, f64)>> {
        let mut by_channel: FxHashMap<ChannelId, Vec<(NeuronId, f64)>> = FxHashMap::default();
        for (&ctx_id, &strength) in level_context.entries() {
            let channel = self.get_neuron_channel_id(ctx_id).unwrap_or(0);
            by_channel.entry(channel).or_insert_with(Vec::new).push((ctx_id, strength));
        }
        by_channel
    }

    /// The narrow walk, context side: pull each neighbor channel's bucket, skipping the neuron itself.
    fn gather_context_from_channels(
        neuron_id: NeuronId,
        channels: &FxHashSet<ChannelId>,
        by_channel: &FxHashMap<ChannelId, Vec<(NeuronId, f64)>>,
    ) -> SpatialContext {
        let mut neighbors = SpatialContext::new();
        for (ctx_id, strength) in channels.iter()
            .filter_map(|channel| by_channel.get(channel))
            .flat_map(|bucket| bucket.iter())
            .filter(|&&(ctx_id, _)| ctx_id != neuron_id)
        {
            neighbors.add_neuron(*ctx_id, *strength);
        }
        neighbors
    }

    /// The wide walk, context side: scan the co-activation and keep this neuron's neighbors.
    fn scan_context_for_neighbors(&self, neuron_id: NeuronId, level_context: &SpatialContext) -> SpatialContext {
        let (channel, level) = self.get_neighborhood_key(neuron_id);
        let mut neighbors = SpatialContext::new();
        for (&ctx_id, &strength) in level_context.entries()
            .iter()
            .filter(|(&ctx_id, _)| ctx_id != neuron_id)
            .filter(|(&ctx_id, _)| {
                let target = self.get_neuron_channel_id(ctx_id).unwrap_or(0);
                self.is_spatial_neighbor_channel(channel, level, target)
            })
        {
            neighbors.add_neuron(ctx_id, strength);
        }
        neighbors
    }

    /// A neuron's declared neighbor channels at its own spatial level — the level-based radius, which
    /// widens with depth. Levels past the declared list reuse the last, fully-grown set.
    /// None means the channel declared no neighbors: unrestricted, every channel is a neighbor.
    fn get_neighbor_channels(&self, neuron_id: NeuronId) -> Option<&FxHashSet<ChannelId>> {
        let (channel, level) = self.get_neighborhood_key(neuron_id);
        self.spatial_channel_neighbors.get(&channel)
            .map(|sets| &sets[(level as usize).min(sets.len() - 1)])
    }

    /// The (channel, spatial level) pair that selects a neuron's neighborhood.
    fn get_neighborhood_key(&self, neuron_id: NeuronId) -> (ChannelId, Level) {
        (self.get_neuron_channel_id(neuron_id).unwrap_or(0), self.get_neuron_spatial_level(neuron_id))
    }

    /// Route each prepared task to its owning region and run the level there.
    /// The only cross-region round-trip in the level loop, so everything a neuron needs for the whole
    /// frame pass — its neighbors both ways — is assembled before this call and shipped in one go.
    fn dispatch_to_regions(
        &mut self,
        work_list: &[NeuronId],
        inference_neighbors: Vec<Vec<ActiveNeuron>>,
        context_neighbors: Vec<SpatialContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<crate::column::SpatialColumnResult> {

        // Route on the neuron's own id: a region owns the neuron's routing table, so its matching,
        // womb, and connection learning must all run where its state lives. The buckets hold indices
        // rather than ids because the two neighbor lists are positional — parallel to work_list.
        let indices_by_region = self.bucket_by_region_indices(work_list, |&id| id);

        let mut results = Vec::new();
        for (r, task_indices) in indices_by_region.iter().enumerate() {

            // Regions with nothing active this frame are skipped rather than sent an empty batch.
            if task_indices.is_empty() { continue; }

            // Re-pair each index into the (neuron, what it predicts, what identifies it) triple the
            // column expects. Cloned because the region takes ownership; the source vectors are
            // dropped at the end of the frame anyway.
            let region_tasks: Vec<_> = task_indices.iter()
                .map(|&i| (work_list[i], inference_neighbors[i].clone(), context_neighbors[i].clone()))
                .collect();

            // Freshly-minted patterns ride along so the region can skip them — they have no history
            // to learn from in the frame they were born.
            results.extend(self.region_list[r].process_spatial_level(&region_tasks, new_error_pattern_ids, frame_number));
        }
        results
    }

    /// Temporal Op-3 dispatch — route each prepared active neuron to its owning region and run its
    /// temporal frame there. Every neuron already carries its own learning work (or None), so this is
    /// pure routing: bucket by region, clone each region's slice, dispatch.
    fn dispatch_temporal_neurons(
        &mut self,
        neurons: &[TemporalNeuron],
        memory_depth: u32,
        level_context: &TemporalContext,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<ColumnProcessResult> {
        let indices_by_region = self.bucket_by_region_indices(neurons, |n| n.neuron_id);
        let level_context_opt = if level_context.size() > 0 { Some(level_context) } else { None };
        let mut results = Vec::new();
        for (r, indices) in indices_by_region.iter().enumerate() {
            if indices.is_empty() { continue; }
            let region_neurons: Vec<TemporalNeuron> = indices.iter().map(|&i| neurons[i].clone()).collect();
            let region_results = self.region_list[r].process_temporal_level(
                &region_neurons, memory_depth, level_context_opt, new_error_pattern_ids,
                frame_number,
            );
            results.extend(region_results);
        }
        results
    }

    /// Clear stale votes/context/threshold for this level's neurons so suppressed
    /// ages don't retain data from a previous frame.
    fn clear_stale_state(level_neurons: &mut FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>) {
        for age_states in level_neurons.values_mut() {
            for state in age_states.values_mut() {
                state.votes = None;
                state.context = None;
                state.threshold = None;
            }
        }
    }

    /// Register deaths and collect activations from recognition matches
    /// and error-correction patterns.
    /// Extract the fired pattern activations from one neuron's spatial result.
    fn collect_spatial_activations(&mut self, result: &crate::column::SpatialColumnResult, activations: &mut Vec<Activation>) {

        // Only recognition matches fire spatially; corrections install through their own pass.
        for m in &result.matches {
            if let Some(df) = m.death_frame {
                self.register_death(m.pattern_id, df);
            }
            if !m.activate { continue; }
            activations.push(Activation {
                parent_id: result.parent_id,
                pattern_id: m.pattern_id,
                age: m.age,
            });
        }
    }

    fn collect_activations(&mut self, result: &ColumnProcessResult, activations: &mut Vec<Activation>) {

        // recognition matches that fired
        for m in &result.matches {
            if m.activate {
                if let Some(df) = m.death_frame {
                    self.register_death(m.pattern_id, df);
                }
                activations.push(Activation {
                    parent_id: result.parent_id,
                    pattern_id: m.pattern_id,
                    age: m.age,
                });
            }
        }

        // error-correction patterns installed this frame
        for ca in &result.correction_activations {
            if let Some(df) = ca.death_frame {
                self.register_death(ca.pattern_id, df);
            }
            activations.push(Activation {
                parent_id: result.parent_id,
                pattern_id: ca.pattern_id,
                age: ca.age,
            });
        }
    }

    /// Write per-age votes back to level state and collect flat votes for consensus.
    /// Threshold is captured here so next frame's evaluate_vote_error can judge these
    /// votes without reaching back into the neuron.
    fn collect_votes(
        result: &ColumnProcessResult,
        level_neurons: &mut FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        votes: &mut Vec<FlatVote>,
    ) {
        if result.votes.is_empty() { return; }

        let age_states = match level_neurons.get_mut(&result.parent_id) {
            Some(a) => a,
            None => return,
        };

        for age_vote in &result.votes {
            if let Some(state) = age_states.get_mut(&age_vote.age) {
                state.votes = Some(age_vote.votes.clone());
                state.context = Some(age_vote.context.clone());
                state.threshold = Some(age_vote.threshold);
                for vote in &age_vote.votes {
                    votes.push(FlatVote {
                        voter_id: result.parent_id,
                        neuron_id: vote.neuron_id,
                        strength: vote.strength,
                        reward: vote.reward,
                        distance: vote.distance,
                    });
                }
            }
        }
    }

    /// Batch contextRef updates by target neuron for Op-5 dispatch.
    /// Fills in `parent_id` from the result (neuron-level code emits it as 0).
    fn collect_context_ref_updates(result: &ColumnProcessResult, per_target: &mut FxHashMap<NeuronId, Vec<TemporalContextRefUpdate>>) {
        for update in &result.context_ref_updates {
            let mut u = update.clone();
            u.parent_id = result.parent_id;
            per_target.entry(u.neuron_id)
                .or_insert_with(Vec::new)
                .push(u);
        }
    }

    /// Evaluate a single (neuron, age) prior-frame vote against current actuals.
    /// Returns the observed error rate (so it can be sent back to the neuron as
    /// feedback) and whether it crosses the threshold the neuron supplied when it
    /// cast the vote. The neuron owns its error stats — thalamus only judges.
    /// Scores the prior-frame EVENT prediction against this frame's events with the shared Jaccard-union
    /// error (`get_union_error`); actions are excluded because they are judged by reward, not hit/miss.
    fn evaluate_vote_error(&self, age: Distance, state: &LevelAgeState, actual_neuron_ids: &FxHashSet<NeuronId>, frame_number: FrameNumber) -> Option<VoteErrorResult> {

        // age=0 neurons cannot need correction because they are just voting now
        if age == 0 { return None; }

        // Warmup: don't fire error corrections before the context window has had a chance to fill up.
        // Without this, the first frames of a sequence (when level_context is mostly empty) generate empty context error patterns
        // that can never match anything on subsequent passes, producing unbounded creation of useless siblings episode after episode.
        // context_length frames is the natural horizon.
        if (frame_number as u32) < self.context_length { return None; }

        // if there are no votes from previous frame, no error to evaluate
        let votes = match &state.votes {
            Some(v) if !v.is_empty() => v,
            _ => return None,
        };

        // predicted = the prior-frame event votes
        let predicted: FxHashSet<NeuronId> = votes.iter()
            .filter(|v| self.get_neuron_type(v.neuron_id) == Some(NeuronType::Event))
            .map(|v| v.neuron_id)
            .collect();
        if predicted.is_empty() { return None; } // no predictions → nothing to evaluate

        // observed = this frame's event neurons
        let observed_events: FxHashSet<NeuronId> = actual_neuron_ids.iter()
            .copied()
            .filter(|&a| self.get_neuron_type(a) == Some(NeuronType::Event))
            .collect();

        // calculate the error rate between predicted and observed events
        let error_rate = crate::types::get_union_error(&predicted, &observed_events)?;

        // the threshold rode in with the vote when it was cast last frame
        let threshold = state.threshold.unwrap_or(0.5);
        let fire = error_rate > threshold;
        Some(VoteErrorResult { fire, error_rate })
    }

    /// Check if a neuron should be skipped (action neuron in a channel whose spec does
    /// not include action-sequence learning).
    fn skip_action_neuron(&self, neuron_id: NeuronId) -> bool {
        if self.neuron_temporal_levels.get(&neuron_id) != Some(&0) { return false; }
        if self.get_neuron_type(neuron_id) != Some(NeuronType::Action) { return false; }
        let channel_id = match self.get_neuron_channel_id(neuron_id) {
            Some(c) => c,
            None => return false,
        };
        match self.channel_specs.get(&channel_id) {
            Some(spec) => !spec.learn_action_sequences,
            None => false,
        }
    }

    // ── Brain.learn(): supervised wiring + read-only vote sweep ────────────

    /// Brain.learn() entry point.
    /// Routes a batch of supervised voter→action wirings to the owning region/column by voter_id.
    /// Each tuple is (voter_id, action_id, reward).
    /// Columns apply additive (strength += 1, reward += reward) onto the voter's distance-`distance` connection slot.
    pub fn learn_action_connections(&mut self, wirings: &[(NeuronId, NeuronId, Reward)], distance: Distance) {
        if wirings.is_empty() { return; }
        let mut by_region: Vec<Vec<(NeuronId, NeuronId, Reward)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for &w in wirings {
            let r = self.route_neuron(w.0);
            by_region[r].push(w);
        }
        for (r, region_wirings) in by_region.iter().enumerate() {
            if !region_wirings.is_empty() {
                self.region_list[r].learn_action_connections(region_wirings, distance);
            }
        }
    }

    /// Read-only vote sweep over (voter_id, age) pairs from currently-active non-suppressed voters.
    /// Routes each pair to its owning region by voter_id, calls `neuron.vote(age)` per pair, and flattens to FlatVotes.
    /// Mirrors what `Neuron::generate_votes` does inside process_frame.
    /// Every active voter at every valid age contributes its connections[age+1] entries.
    /// Used by Brain.learn() to produce the post-wire inference pass without touching any state.
    pub fn collect_votes_for_voter_ages(&self, voter_ages: &[(NeuronId, Distance)]) -> Vec<FlatVote> {
        if voter_ages.is_empty() { return Vec::new(); }
        let mut by_region: Vec<Vec<(NeuronId, Distance)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for &pair in voter_ages {
            let r = self.route_neuron(pair.0);
            by_region[r].push(pair);
        }
        let mut flat = Vec::new();
        for (r, region_pairs) in by_region.iter().enumerate() {
            if region_pairs.is_empty() { continue; }
            for (voter_id, _age, votes) in self.region_list[r].collect_votes_for_voter_ages(region_pairs) {
                for vote in votes {
                    flat.push(FlatVote {
                        voter_id,
                        neuron_id: vote.neuron_id,
                        strength: vote.strength,
                        reward: vote.reward,
                        distance: vote.distance,
                    });
                }
            }
        }
        flat
    }

    // ── Neuron creation / dispatch ──────────────────────────────────────────

    /// Op-1/Op-4: Route neuron specs to owning regions/columns for construction.
    pub fn create_neurons(&mut self, specs: &[NeuronCreateSpec]) {
        if specs.is_empty() { return; }

        // bucket specs by region
        let mut by_region: Vec<Vec<NeuronCreateSpec>> = (0..self.regions).map(|_| Vec::new()).collect();
        for spec in specs {
            let r = self.route_neuron(spec.id);
            by_region[r].push(spec.clone());
        }

        for (r, region_specs) in by_region.iter().enumerate() {
            if !region_specs.is_empty() {
                self.region_list[r].create_neurons(region_specs);
            }
        }
    }

    /// Op-5 (deferred): Route contextRef updates to owning regions/columns.
    fn dispatch_temporal_context_ref_updates(&mut self, update_batch: &[(NeuronId, Vec<TemporalContextRefUpdate>)]) {
        let mut by_region: Vec<Vec<(NeuronId, Vec<TemporalContextRefUpdate>)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for (neuron_id, updates) in update_batch {
            let r = self.route_neuron(*neuron_id);
            by_region[r].push((*neuron_id, updates.clone()));
        }
        for (r, region_updates) in by_region.iter().enumerate() {
            if !region_updates.is_empty() {
                self.region_list[r].update_temporal_context_refs(region_updates);
            }
        }
    }

    // ── Death management ────────────────────────────────────────────────────

    /// Register a neuron's scheduled death frame.
    pub fn register_death(&mut self, neuron_id: NeuronId, death_frame: FrameNumber) {
        // unregister old death frame if exists
        self.unregister_death(neuron_id);

        // register new death frame
        self.death_ledger.entry(death_frame).or_insert_with(FxHashSet::default).insert(neuron_id);
        self.neuron_death_frame.insert(neuron_id, death_frame);
    }

    /// Unregister a neuron from the death ledger.
    pub fn unregister_death(&mut self, neuron_id: NeuronId) {
        let old_frame = match self.neuron_death_frame.get(&neuron_id) {
            Some(&f) => f,
            None => return,
        };

        if let Some(set) = self.death_ledger.get_mut(&old_frame) {
            set.remove(&neuron_id);
            if set.is_empty() { self.death_ledger.remove(&old_frame); }
        }
        self.neuron_death_frame.remove(&neuron_id);
    }

    /// Reap neurons scheduled to die at the given frame.
    /// Returns array of dead neuron ids and removes them from the ledger.
    pub fn reap_dead_neurons(&mut self, current_frame: FrameNumber) -> Vec<NeuronId> {

        // get the neurons to be deleted in this frame
        let neuron_ids = match self.death_ledger.remove(&current_frame) {
            Some(ids) => ids,
            None => return Vec::new(), // nothing to do if no neurons dying
        };

        // reap the dead neuron ids and return them
        let mut dead = Vec::new();
        for neuron_id in &neuron_ids {
            if self.neuron_temporal_levels.contains_key(neuron_id) { dead.push(*neuron_id); }
            self.neuron_death_frame.remove(neuron_id);
        }
        dead
    }

    // ── Delete cascade ──────────────────────────────────────────────────────

    /// Op-2: Delete dead patterns via cascade pulses through region/column.
    /// Each pulse dispatches ops, collects results, cleans metadata, and feeds
    /// outbound ops + newly deletable ids into the next pulse until empty.
    pub fn delete_patterns(&mut self, pattern_ids: &[NeuronId], current_frame: FrameNumber) -> Vec<NeuronId> {
        let mut deleted_ids = Vec::new();
        let mut deleted_id_set = FxHashSet::default();
        let mut queued_ids: FxHashSet<NeuronId> = pattern_ids.iter().copied().collect();

        // seed the first pulse with DeleteNeuron ops for reaped patterns
        let mut inbound_ops = self.build_delete_neuron_ops(pattern_ids);

        // cascade: each pulse may produce outbound ops and new cascade candidates
        while !inbound_ops.is_empty() {
            let result = self.dispatch_delete_ops(&inbound_ops, current_frame);

            // remove destroyed neurons from Thalamus metadata
            for &id in &result.deleted_ids {
                if deleted_id_set.contains(&id) { continue; }
                deleted_id_set.insert(id);
                deleted_ids.push(id);
                self.cleanup_deleted_neuron_metadata(id);
            }

            // feed outbound ops into next pulse, plus new DeleteNeuron ops for cascade candidates
            inbound_ops = result.outbound_ops;
            for id in &result.newly_deletable_ids {
                if deleted_id_set.contains(id) || queued_ids.contains(id) { continue; }
                queued_ids.insert(*id);
                if let Some(&parent_id) = self.neuron_parents.get(id) {
                    inbound_ops.push(DeleteOp::DeleteNeuron { target_id: *id, parent_id });
                }
            }
        }

        deleted_ids
    }

    /// Build initial DeleteNeuron ops from a list of pattern ids.
    fn build_delete_neuron_ops(&self, pattern_ids: &[NeuronId]) -> Vec<DeleteOp> {
        let mut ops = Vec::new();
        for &id in pattern_ids {
            if let Some(&parent_id) = self.neuron_parents.get(&id) {
                ops.push(DeleteOp::DeleteNeuron { target_id: id, parent_id });
            }
            // sensory neurons have no parent — skip
        }
        ops
    }

    /// Fan out delete ops to regions, collect and merge results.
    fn dispatch_delete_ops(&mut self, ops: &[DeleteOp], current_frame: FrameNumber) -> DispatchDeleteResult {
        // bucket ops by region using target_id
        let mut by_region: Vec<Vec<DeleteOp>> = (0..self.regions).map(|_| Vec::new()).collect();
        for op in ops {
            let r = self.route_neuron(op.target_id());
            by_region[r].push(op.clone());
        }

        let mut outbound_ops = Vec::new();
        let mut deleted_ids = Vec::new();
        let mut newly_deletable_ids = Vec::new();

        for (r, region_ops) in by_region.iter().enumerate() {
            if region_ops.is_empty() { continue; }
            let result = self.region_list[r].delete_neurons(region_ops, current_frame);
            outbound_ops.extend(result.outbound_ops);
            deleted_ids.extend(result.deleted_ids);
            newly_deletable_ids.extend(result.newly_deletable_ids);
        }

        DispatchDeleteResult { outbound_ops, deleted_ids, newly_deletable_ids }
    }

    /// Remove a destroyed neuron from Thalamus-owned metadata maps.
    fn cleanup_deleted_neuron_metadata(&mut self, id: NeuronId) {
        self.unregister_death(id);
        let level = self.neuron_temporal_levels.remove(&id);
        self.neuron_spatial_levels.remove(&id);
        self.neuron_parents.remove(&id);
        if let Some(level) = level {
            self.decrement_temporal_level_count(level);
        }
    }

    // ── Snapshot / restore ──────────────────────────────────────────────────

    /// Get a self-contained snapshot of all brain state for external consumers (backup, dump).
    /// Each neuron entry carries serialized neuron data plus resolved metadata — consumers
    /// never need separate lookups or access to live Neuron objects.
    pub fn get_snapshot(&self) -> Snapshot {
        let mut neurons = Vec::new();
        for region in &self.region_list {
            for entry in region.get_snapshot() {
                let temporal_level = self.neuron_temporal_levels.get(&entry.id).copied().unwrap_or(0);
                // Only TRUE sensory neurons carry serializable base data. Spatial pattern neurons
                // also sit at temporal level 0 and, since fix 2.2, hold an inherited coordinate in
                // base_neurons — but that coordinate must NOT be snapshotted (restore would insert
                // it into neurons_by_value and clobber the L0 sensory mapping). It is derivable on
                // restore by walking neuron_parents to the L0 ancestor (fix 1.2). Distinguish by
                // spatial level: sensories are spatial_level 0, patterns are spatial_level >= 1.
                let spatial_level = self.get_neuron_spatial_level(entry.id);
                let is_sensory = temporal_level == 0 && spatial_level == 0;
                let base_neuron = if is_sensory { self.base_neurons.get(&entry.id).cloned() } else { None };
                let parent_id = self.neuron_parents.get(&entry.id).copied();
                neurons.push(SnapshotNeuronEntry {
                    neuron: entry.neuron,
                    temporal_level,
                    spatial_level,
                    base_neuron,
                    parent_id,
                });
            }
        }
        Snapshot {
            neurons,
            channel_name_to_id: self.channel_name_to_id.clone(),
            dimension_name_to_id: self.dimension_name_to_id.clone(),
        }
    }

    /// Restore brain state from a snapshot.
    /// Channel and dimension specs are expected to have been registered via
    /// register_channel_spec() BEFORE restore — the snapshot only carries the persisted
    /// id↔name maps so we can reconcile allocated-vs-persisted IDs and advance the
    /// counters past whatever was on disk.
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) {

        // advance ID counters past any persisted IDs
        for &id in snapshot.channel_name_to_id.values() {
            if id >= self.next_channel_id { self.next_channel_id = id + 1; }
        }
        for &id in snapshot.dimension_name_to_id.values() {
            if id >= self.next_dimension_id { self.next_dimension_id = id + 1; }
        }

        // reset neurons
        self.reset();

        // restore central metadata maps and bucket neurons by routing rule
        let mut buckets: Vec<Vec<Vec<SerializedNeuron>>> = Vec::with_capacity(self.regions);
        for _ in 0..self.regions {
            let mut region_buckets = Vec::with_capacity(self.columns);
            for _ in 0..self.columns { region_buckets.push(Vec::new()); }
            buckets.push(region_buckets);
        }

        // restore base neurons and their value maps
        for entry in &snapshot.neurons {
            let neuron_id = entry.neuron.id;
            if neuron_id >= self.next_neuron_id { self.next_neuron_id = neuron_id + 1; }
            self.neuron_temporal_levels.insert(neuron_id, entry.temporal_level);
            // Repopulate the spatial level so spatial pattern neurons keep their hierarchy depth.
            // reset() cleared neuron_spatial_levels; without this every neuron would read back as
            // spatial_level 0 (indistinguishable from a sensory) and the hierarchy would collapse.
            if entry.spatial_level != 0 {
                self.neuron_spatial_levels.insert(neuron_id, entry.spatial_level);
            }
            if let Some(parent_id) = entry.parent_id {
                self.neuron_parents.insert(neuron_id, parent_id);
            }
            self.increment_temporal_level_count(entry.temporal_level);
            if entry.temporal_level == 0 {
                if let Some(ref base) = entry.base_neuron {
                    self.neurons_by_value.insert(base.coordinate.clone(), neuron_id);
                    self.base_neurons.insert(neuron_id, base.clone());
                }
            }
            let r = self.route_neuron(neuron_id);
            let c = self.region_list[r].route_neuron(neuron_id);
            buckets[r][c].push(entry.neuron.clone());
        }

        // Rebuild inherited coordinates for spatial pattern neurons. Snapshots intentionally drop the
        // inherited (channel, dimension, bucket) coordinate of a spatial correction — snapshotting it
        // would clobber the L0 sensory mapping in neurons_by_value on restore. Instead it is rederived
        // here by chaining each correction to its parent's base coordinate, mirroring
        // allocate_spatial_pattern_neuron. Process in ascending spatial_level so an L2 inherits from its
        // already-rebuilt L1 parent, which in turn anchored at the original L0 sensory.
        let mut spatial_patterns: Vec<(NeuronId, Level)> = snapshot.neurons.iter()
            .filter(|e| e.spatial_level != 0)
            .map(|e| (e.neuron.id, e.spatial_level))
            .collect();
        spatial_patterns.sort_by_key(|&(_, level)| level);
        for (neuron_id, _) in spatial_patterns {
            let parent_id = match self.neuron_parents.get(&neuron_id) {
                Some(&p) => p,
                None => continue,
            };
            if let Some(inherited) = self.base_neurons.get(&parent_id).cloned() {
                // NOT inserted into neurons_by_value — refined tokens are reached only via routing
                // matches, and that map must keep resolving the coordinate to its L0 sensory.
                self.base_neurons.insert(neuron_id, inherited);
            }
        }

        // Rebuild inherited coordinates for temporal pattern neurons, mirroring the spatial rebuild
        // above and allocate_temporal_pattern_neuron. Snapshots drop the inherited coordinate for the
        // same reason (it would clobber the L0 sensory mapping in neurons_by_value on restore), so it
        // is rederived here by chaining each correction to its parent's base coordinate. Process in
        // ascending temporal_level so an L2 inherits from its already-rebuilt L1 parent, which in turn
        // anchored at the original L0 sensory. Temporal patterns are spatial_level 0 with
        // temporal_level >= 1 — the complement of the spatial-pattern filter above.
        let mut temporal_patterns: Vec<(NeuronId, Level)> = snapshot.neurons.iter()
            .filter(|e| e.spatial_level == 0 && e.temporal_level != 0)
            .map(|e| (e.neuron.id, e.temporal_level))
            .collect();
        temporal_patterns.sort_by_key(|&(_, level)| level);
        for (neuron_id, _) in temporal_patterns {
            let parent_id = match self.neuron_parents.get(&neuron_id) {
                Some(&p) => p,
                None => continue,
            };
            if let Some(inherited) = self.base_neurons.get(&parent_id).cloned() {
                self.base_neurons.insert(neuron_id, inherited);
            }
        }

        // distribute neurons to their owning columns
        for (r, region_buckets) in buckets.into_iter().enumerate() {
            self.region_list[r].restore_snapshot(region_buckets);
        }

        // Rebuild the position metadata of spatial connection targets on every restored neuron.
        // Snapshots do not persist the (channel, dimension) per target; lateral connection learning
        // wires toward same-level events — L0 sensories and spatial patterns alike — and both carry
        // coordinates in base metadata (patterns inherit their parent's), so the map is derivable.
        let target_meta: FxHashMap<NeuronId, (ChannelId, DimensionId)> = self.base_neurons.iter()
            .filter(|(id, b)| {
                b.neuron_type == NeuronType::Event
                    && self.neuron_temporal_levels.get(*id).copied().unwrap_or(0) == 0
            })
            .map(|(&id, b)| (id, (b.channel_id, b.coordinate.dim_id)))
            .collect();
        for region in &mut self.region_list {
            region.decorate_spatial_targets(&target_meta);
        }

        // rebuild the death ledger from materialized activation strengths —
        // collect first, then register (can't borrow self immutably and mutably at once)
        let death_frames: Vec<_> = self.region_list.iter()
            .flat_map(|region| region.collect_death_frames())
            .collect();
        for df_entry in death_frames {
            self.register_death(df_entry.pattern_id, df_entry.death_frame);
        }
    }

    /// Reset all neurons and neuron ID counter.
    pub fn reset(&mut self) {
        for region in &mut self.region_list { region.clear(); }
        self.neuron_temporal_levels.clear();
        self.neuron_spatial_levels.clear();
        self.neurons_by_value.clear();
        self.base_neurons.clear();
        self.neuron_parents.clear();
        self.death_ledger.clear();
        self.neuron_death_frame.clear();
        self.temporal_level_counts.clear();
        self.next_neuron_id = 1;
    }

    /// Materialize all lazy decay into actual values and reset timestamps.
    /// Re-registers death frames so pattern cleanup continues working.
    pub fn materialize_and_reset_neurons(&mut self, current_frame: FrameNumber) {
        self.death_ledger.clear();
        self.neuron_death_frame.clear();

        // collect death frames first, then register (borrow checker: can't mutate
        // self.death_ledger while iterating self.region_list mutably)
        let death_frames: Vec<_> = self.region_list.iter_mut()
            .flat_map(|region| region.materialize_and_reset_neurons(current_frame))
            .collect();
        for df_entry in death_frames {
            self.register_death(df_entry.pattern_id, df_entry.death_frame);
        }
    }

    // ── Level count diagnostics ─────────────────────────────────────────────

    /// Increment the neuron count at a given level.
    fn increment_temporal_level_count(&mut self, level: Level) {
        while self.temporal_level_counts.len() <= level as usize { self.temporal_level_counts.push(0); }
        self.temporal_level_counts[level as usize] += 1;
    }

    /// Decrement the neuron count at a given level.
    fn decrement_temporal_level_count(&mut self, level: Level) {
        if (level as usize) < self.temporal_level_counts.len() {
            self.temporal_level_counts[level as usize] -= 1;
        }
    }

    /// Get total number of neurons.
    pub fn get_neuron_count(&self) -> usize {
        self.neuron_temporal_levels.len()
    }

    /// Maximum live TEMPORAL level — depth of the temporal pattern hierarchy.
    /// 0 = sensory only; 1+ = temporal correction patterns exist at that level.
    pub fn get_max_temporal_level(&self) -> Level {
        for i in (0..self.temporal_level_counts.len()).rev() {
            if self.temporal_level_counts[i] > 0 { return i as Level; }
        }
        0
    }

    /// Maximum live SPATIAL level — depth of the spatial pattern hierarchy.
    /// 0 = sensory only; 1+ = spatial correction patterns exist at that level.
    pub fn get_max_spatial_level(&self) -> Level {
        self.neuron_spatial_levels.values().copied().max().unwrap_or(0)
    }

    /// Get the dimension_id → name mapping (for diagnostic display).
    pub fn get_dimension_id_to_name(&self) -> FxHashMap<DimensionId, String> {
        self.dimension_id_to_name.clone()
    }

    /// Get the dimension_name → id mapping (for diagnostic display).
    pub fn get_dimension_id_by_name(&self, name: &str) -> Option<DimensionId> {
        self.dimension_name_to_id.get(name).copied()
    }
}

// ── Internal task/spec types ────────────────────────────────────────────────

/// A task to be dispatched to a neuron during process_level.
/// Error-correction pattern spec — carries connection data for neuron creation.
#[derive(Debug, Clone)]
struct CorrectionSpec {
    pattern_id: NeuronId,
    forget_rate: f64,
    connections: Vec<ConnectionSpec>,
    age: Distance,
    context_entries: Vec<ContextRefEntry>,
}

/// Install op for a freshly-born spatial correction. Records the parent whose routing table gains
/// the pattern, the new pattern's id, and the d=0 context entries (the embryo's center) to bind
/// against. The pattern is born paid — its embryo already covered the price in the womb.
#[derive(Debug, Clone)]
pub struct SpatialInstallOp {
    pub parent_id: NeuronId,
    pub pattern_id: NeuronId,
    pub context_neuron_ids: Vec<NeuronId>,
}

/// Result of region.install_spatial_corrections — death frames for the death ledger plus
/// the spatial context-ref updates the install produced. Updates are emitted as a flat Vec
/// (each update carries its own target neuron_id) and bucketed at the thalamus boundary,
/// mirroring the temporal dispatch chain.
pub struct SpatialInstallResult {
    pub deaths: Vec<(NeuronId, FrameNumber)>,
    pub context_ref_updates: Vec<SpatialContextRefUpdate>,
}

/// Result of allocate_pattern_neuron.
pub struct PatternNeuronSpec {
    pub id: NeuronId,
    pub forget_rate: f64,
    pub connections: Vec<ConnectionSpec>,
}

/// An inferred action coordinate.
/// Input spec for registering a dimension (before allocation).
#[derive(Debug, Clone)]
pub struct DimSpecInput {
    pub name: String,
    pub kind: DimKind,
    pub resolution: u32,
    pub mode: Option<String>,
    pub boundaries: Option<Vec<f64>>,
    pub actions: Option<Vec<i32>>,
    pub default_action: Option<i32>,
    pub warmup_samples: Option<usize>,
}

/// Internal result from dispatch_delete_ops.
struct DispatchDeleteResult {
    outbound_ops: Vec<DeleteOp>,
    deleted_ids: Vec<NeuronId>,
    newly_deletable_ids: Vec<NeuronId>,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_thalamus() -> Thalamus {
        Thalamus::new(false, 0.1, 4, 0.5, GroupMode::Static, 1, 1, true)
    }

    #[test]
    fn test_allocate_sensory_neuron() {
        let mut t = make_thalamus();
        let coord = Coordinate { dim_id: 1, bucket_id: 3 };
        let lookup = t.get_neuron_id_for_point(&coord, 1, NeuronType::Event);
        assert!(lookup.is_new);
        assert_eq!(lookup.id, 1);

        // second lookup returns existing
        let lookup2 = t.get_neuron_id_for_point(&coord, 1, NeuronType::Event);
        assert!(!lookup2.is_new);
        assert_eq!(lookup2.id, 1);

        // metadata registered
        assert_eq!(t.get_neuron_channel_id(1), Some(1));
        assert_eq!(t.get_neuron_type(1), Some(NeuronType::Event));
        assert_eq!(t.get_neuron_temporal_level(1), Some(0));
    }

    #[test]
    fn test_routing() {
        let t = Thalamus::new(false, 0.1, 4, 0.5, GroupMode::Static, 3, 1, true);
        assert_eq!(t.route_neuron(1), 1);
        assert_eq!(t.route_neuron(2), 2);
        assert_eq!(t.route_neuron(3), 0);
        assert_eq!(t.route_neuron(6), 0);
    }

    #[test]
    fn test_register_channel_spec() {
        let mut t = make_thalamus();
        let reg = t.register_channel_spec(
            "test_channel",
            vec![
                DimSpecInput {
                    name: "price".to_string(),
                    kind: DimKind::Input,
                    resolution: 3,
                    mode: Some("static".to_string()),
                    boundaries: Some(vec![-0.5, 0.5]),
                    actions: None,
                    default_action: None,
                    warmup_samples: None,
                },
            ],
            false,
        );
        assert_eq!(reg.channel_id, 1);
        assert!(reg.dimension_ids.contains_key("price"));

        // channel lookups work
        assert_eq!(t.get_channel_ids().len(), 1);
        assert!(t.get_channel_spec(1).is_some());
    }

    #[test]
    fn test_register_channel_with_actions() {
        let mut t = make_thalamus();
        let reg = t.register_channel_spec(
            "action_channel",
            vec![
                DimSpecInput {
                    name: "trade".to_string(),
                    kind: DimKind::Action,
                    resolution: 3,
                    mode: None,
                    boundaries: None,
                    actions: Some(vec![1, 2, 3]),
                    default_action: Some(1),
                    warmup_samples: None,
                },
            ],
            false,
        );

        // action neurons should have been created
        assert!(t.channel_actions.contains_key(&reg.channel_id));
        assert_eq!(t.channel_actions.get(&reg.channel_id).unwrap().len(), 3);
        assert!(t.channel_default_actions.contains_key(&reg.channel_id));
        assert_eq!(t.get_neuron_count(), 3);
    }

    #[test]
    fn test_death_ledger() {
        let mut t = make_thalamus();

        // register a death
        t.register_death(42, 100);
        assert_eq!(t.neuron_death_frame.get(&42), Some(&100));

        // re-register at a different frame
        t.register_death(42, 200);
        assert_eq!(t.neuron_death_frame.get(&42), Some(&200));
        assert!(!t.death_ledger.contains_key(&100)); // old frame cleaned up

        // reap — neuron 42 not in neuron_temporal_levels so won't be returned
        let dead = t.reap_dead_neurons(200);
        assert!(dead.is_empty());

        // register with level to simulate actual neuron
        t.neuron_temporal_levels.insert(42, 1);
        t.register_death(42, 300);
        let dead = t.reap_dead_neurons(300);
        assert_eq!(dead, vec![42]);
    }

    #[test]
    fn test_reset() {
        let mut t = make_thalamus();
        let coord = Coordinate { dim_id: 1, bucket_id: 1 };
        t.get_neuron_id_for_point(&coord, 1, NeuronType::Event);
        assert_eq!(t.get_neuron_count(), 1);

        t.reset();
        assert_eq!(t.get_neuron_count(), 0);
        assert!(t.get_neuron_id_by_coordinate(&coord).is_none());
    }

    #[test]
    fn test_level_counts() {
        let mut t = make_thalamus();
        let coord1 = Coordinate { dim_id: 1, bucket_id: 1 };
        let coord2 = Coordinate { dim_id: 1, bucket_id: 2 };
        t.get_neuron_id_for_point(&coord1, 1, NeuronType::Event);
        t.get_neuron_id_for_point(&coord2, 1, NeuronType::Event);
        assert_eq!(t.get_max_temporal_level(), 0);
        assert_eq!(t.get_neuron_count(), 2);
    }

    #[test]
    fn test_skip_action_neuron() {
        let mut t = make_thalamus();

        // register a channel without action sequence learning
        t.register_channel_spec(
            "ch",
            vec![DimSpecInput {
                name: "act".to_string(),
                kind: DimKind::Action,
                resolution: 2,
                mode: None,
                boundaries: None,
                actions: Some(vec![1, 2]),
                default_action: Some(1),
                warmup_samples: None,
            }],
            false, // learn_action_sequences = false
        );

        // action neurons should be skipped
        let action_id = t.get_neuron_id_by_coordinate(&Coordinate { dim_id: 1, bucket_id: 1 }).unwrap();
        assert!(t.skip_action_neuron(action_id));
    }

    /// Helper: create a thalamus with event neurons registered at IDs 1 and 2.
    fn make_thalamus_with_events() -> Thalamus {
        let mut t = make_thalamus();
        // register neuron 1 as event at coord (dim=1, bucket=1)
        t.get_neuron_id_for_point(&Coordinate { dim_id: 1, bucket_id: 1 }, 1, NeuronType::Event);
        // register neuron 2 as event at coord (dim=1, bucket=2)
        t.get_neuron_id_for_point(&Coordinate { dim_id: 1, bucket_id: 2 }, 1, NeuronType::Event);
        t
    }

    #[test]
    fn test_evaluate_vote_error_no_votes() {
        let t = make_thalamus_with_events();
        let state = LevelAgeState::default();
        assert!(t.evaluate_vote_error(1, &state, &FxHashSet::default(), 1000).is_none());
    }

    #[test]
    fn test_evaluate_vote_error_age_zero() {
        let t = make_thalamus_with_events();
        let state = LevelAgeState {
            votes: Some(vec![Vote { neuron_id: 1, strength: 1.0, reward: 0.0, distance: 1 }]),
            ..Default::default()
        };
        // age 0 always returns None
        assert!(t.evaluate_vote_error(0, &state, &FxHashSet::default(), 1000).is_none());
    }

    #[test]
    fn test_evaluate_vote_error_fires() {
        let t = make_thalamus_with_events();
        let state = LevelAgeState {
            votes: Some(vec![
                Vote { neuron_id: 1, strength: 1.0, reward: 0.0, distance: 1 },
                Vote { neuron_id: 2, strength: 1.0, reward: 0.0, distance: 1 },
            ]),
            threshold: Some(0.3),
            ..Default::default()
        };
        // neither neuron is in actuals → 100% error → fires (> 0.3)
        let result = t.evaluate_vote_error(1, &state, &FxHashSet::default(), 1000).unwrap();
        assert!(result.fire);
        assert!((result.error_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_vote_error_does_not_fire() {
        let t = make_thalamus_with_events();
        let mut actuals = FxHashSet::default();
        actuals.insert(1);
        actuals.insert(2);
        let state = LevelAgeState {
            votes: Some(vec![
                Vote { neuron_id: 1, strength: 1.0, reward: 0.0, distance: 1 },
                Vote { neuron_id: 2, strength: 1.0, reward: 0.0, distance: 1 },
            ]),
            threshold: Some(0.5),
            ..Default::default()
        };
        // both correct → 0% error → does not fire
        let result = t.evaluate_vote_error(1, &state, &actuals, 1000).unwrap();
        assert!(!result.fire);
        assert!((result.error_rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_vote_error_counts_novel_actuals() {
        // Union behavior: an actual event the votes failed to predict counts as novel.
        // Predicted = {1}, actual events = {1, 2}: common=1, missing=0, novel=1 → error = 1/2 = 0.5.
        // The old containment form (missing/total = 0/1) would have scored 0 and never fired here.
        let t = make_thalamus_with_events();
        let mut actuals = FxHashSet::default();
        actuals.insert(1);
        actuals.insert(2);
        let state = LevelAgeState {
            votes: Some(vec![Vote { neuron_id: 1, strength: 1.0, reward: 0.0, distance: 1 }]),
            threshold: Some(0.3),
            ..Default::default()
        };
        let result = t.evaluate_vote_error(1, &state, &actuals, 1000).unwrap();
        assert!(result.fire);
        assert!((result.error_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_roundtrip_empty() {
        let mut t = make_thalamus();
        let snapshot = t.get_snapshot();
        assert!(snapshot.neurons.is_empty());

        // restore empty snapshot is a no-op
        t.restore_snapshot(&snapshot);
        assert_eq!(t.get_neuron_count(), 0);
    }

    #[test]
    fn test_get_inference_results() {
        let mut t = make_thalamus();

        // create two event neurons
        let coord1 = Coordinate { dim_id: 1, bucket_id: 1 };
        let coord2 = Coordinate { dim_id: 1, bucket_id: 2 };
        let lookup1 = t.get_neuron_id_for_point(&coord1, 1, NeuronType::Event);
        let lookup2 = t.get_neuron_id_for_point(&coord2, 1, NeuronType::Event);
        // must create the neurons in columns too
        t.create_neurons(&[
            NeuronCreateSpec { id: lookup1.id, forget_rate: 0.0, connections: None },
            NeuronCreateSpec { id: lookup2.id, forget_rate: 0.0, connections: None },
        ]);

        // neuron 1 is active, neuron 2 was predicted
        let mut active = FxHashSet::default();
        active.insert(lookup1.id);

        let inferred = vec![InferredNeuron {
            neuron_id: lookup2.id,
            coordinate: coord2.clone(),
            channel_id: 1,
            strength: 1.0,
            reward: 0.0,
            probability: 0.0,
        }]; // predicted neuron 2
        let rewards = FxHashMap::default();

        let results = t.get_inference_results(&active, &inferred, &rewards);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_correct); // predicted 2 but 1 was active
    }
}
