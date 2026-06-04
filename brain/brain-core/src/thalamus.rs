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
use crate::context::Context;
use crate::diagnostics::{InferenceResultItem, InferenceType};
use crate::neuron::{
    ActiveNeuron, AgeState, ContextRefEntry, ContextRefUpdate,
    Correction, ErrorFeedback,
    SerializedNeuron, Vote,
};
use crate::quantizer::{QuantizeMode, Quantizer};
use crate::region::Region;
use crate::types::{
    ChannelId, Coordinate, DimensionId, Distance, ErrorMode, FrameNumber,
    Level, NeuronId, NeuronType, Phase, Reward,
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
    /// Context snapshot at vote time (for misprediction diagnostics).
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
    pub level: Level,
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
    neuron_levels: FxHashMap<NeuronId, Level>,

    /// Neuron spatial level: neuron id → level in the spatial hierarchy (0 = sensory/base, 1+ = spatial correction).
    /// Stored separately from `neuron_levels` because a neuron occupies independent positions
    /// in the spatial and temporal hierarchies (see spatial-processing.md §3.3).
    /// Absent entries default to 0 — pre-spatial-era persisted neurons inherit this on load.
    neuron_spatial_levels: FxHashMap<NeuronId, Level>,

    /// Cumulative count of spatial correction patterns minted by `mint_spatial_corrections`.
    /// Diagnostic — surfaced via Brain.get_spatial_correction_count() for harness validation.
    spatial_corrections_minted: u64,

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
    level_counts: Vec<i64>,

    /// Region instances, indexed by region index.
    region_list: Vec<Region>,
}

impl Thalamus {
    pub fn new(
        debug: bool,
        pattern_forget_rate: f64,
        merge_threshold: f64,
        context_length: u32,
        error_mode: ErrorMode,
        error_threshold: f64,
        regions: usize,
        columns: usize,
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
                merge_threshold,
                error_mode,
                error_threshold,
            ));
        }

        Self {
            debug,
            pattern_forget_rate,
            context_length,
            regions,
            columns,
            neurons_by_value: FxHashMap::default(),
            base_neurons: FxHashMap::default(),
            neuron_parents: FxHashMap::default(),
            neuron_levels: FxHashMap::default(),
            neuron_spatial_levels: FxHashMap::default(),
            spatial_corrections_minted: 0,
            death_ledger: FxHashMap::default(),
            neuron_death_frame: FxHashMap::default(),
            channel_specs: FxHashMap::default(),
            dimension_specs: FxHashMap::default(),
            channel_actions,
            channel_default_actions: FxHashMap::default(),
            channel_name_to_id: FxHashMap::default(),
            channel_id_to_name: FxHashMap::default(),
            dimension_name_to_id: FxHashMap::default(),
            dimension_id_to_name: FxHashMap::default(),
            quantizer: Quantizer::new(),
            next_channel_id: 1,
            next_dimension_id: 1,
            next_neuron_id: 1,
            level_counts: Vec::new(),
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
        self.neuron_levels.insert(id, 0);
        self.neurons_by_value.insert(coordinate.clone(), id);
        self.base_neurons.insert(id, BaseNeuron { channel_id, neuron_type, coordinate: coordinate.clone() });
        self.increment_level_count(0);
        id
    }

    /// Allocate a pattern neuron id and build its creation spec. Resolves connection
    /// data (channel, reward) using Thalamus-local lookups so the spec is self-contained
    /// and can cross the MPI boundary. Does NOT construct the Neuron — that happens in
    /// Column.create_neurons via create_neurons.
    /// Does NOT touch the parent's routing table (that happens inside parent.process_frame
    /// via add_pattern) and does NOT register death (death frame is known only after
    /// parent.add_pattern runs).
    pub fn allocate_pattern_neuron(
        &mut self,
        level: Level,
        parent_id: NeuronId,
        age: Distance,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        phase: Phase,
    ) -> PatternNeuronSpec {

        // resolve connection spec using thalamus-local lookups (channel, reward).
        // Temporal corrections pre-wire d=1..age connections toward each per-age active set.
        // Spatial corrections are allocated at age=0 with empty connections; learn_connections
        // will fill connections[0] on future frames as the correction co-fires with others.
        let mut connections = Vec::new();
        for a in 0..age.min(sensory_neurons.len() as u32) {
            let a_idx = a as usize;
            for &sensory_neuron_id in &sensory_neurons[a_idx] {
                let channel_id = self.get_neuron_channel_id(sensory_neuron_id).unwrap_or(0);
                let reward = rewards[a_idx].get(&channel_id).copied().unwrap_or(0.0);
                connections.push(ConnectionSpec {
                    distance: age - a,
                    to_neuron_id: sensory_neuron_id,
                    strength: 1.0,
                    reward,
                    channel_id,
                });
            }
        }

        // allocate id and build the spec for Column.create_neurons
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;

        // register metadata centrally (Neuron construction deferred to create_neurons).
        // Temporal corrections occupy temporal level=parent.level+1; their spatial_level defaults to 0.
        // Spatial corrections occupy spatial_level=parent.spatial_level+1; their temporal level stays 0
        // (they enter temporal via the apex handoff at temporal_level_index[0]).
        self.neuron_parents.insert(id, parent_id);
        match phase {
            Phase::Temporal => {
                self.neuron_levels.insert(id, level);
                self.increment_level_count(level);
            }
            Phase::Spatial => {
                self.neuron_levels.insert(id, 0);
                self.neuron_spatial_levels.insert(id, level);
                self.increment_level_count(0);
            }
        }

        PatternNeuronSpec { id, forget_rate: self.pattern_forget_rate, connections }
    }

    // ── Neuron metadata getters ─────────────────────────────────────────────

    /// Get the channel id for a neuron.
    pub fn get_neuron_channel_id(&self, neuron_id: NeuronId) -> Option<ChannelId> {
        self.base_neurons.get(&neuron_id).map(|b| b.channel_id)
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
    pub fn get_neuron_level(&self, neuron_id: NeuronId) -> Option<Level> {
        self.neuron_levels.get(&neuron_id).copied()
    }

    /// Get the spatial level for a neuron (0 = sensory/base or never spatially registered, 1+ = spatial correction).
    pub fn get_neuron_spatial_level(&self, neuron_id: NeuronId) -> Level {
        self.neuron_spatial_levels.get(&neuron_id).copied().unwrap_or(0)
    }

    /// Cumulative count of spatial corrections minted since brain start (or last hard reset).
    pub fn get_spatial_correction_count(&self) -> u64 {
        self.spatial_corrections_minted
    }

    /// Count of neurons currently sitting above the base level in the spatial hierarchy.
    /// Counts unique correction neurons rather than mint events, so a correction that's later
    /// deleted via cascade doesn't show up.
    pub fn count_active_spatial_corrections(&self) -> usize {
        self.neuron_spatial_levels.values().filter(|&&lvl| lvl > 0).count()
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

    /// Get inferred actions grouped by channel from the given inferences.
    /// Returns a map of channel_id → list of {coordinate, strength, reward}.
    pub fn get_inferred_actions(&self, inferences: &[InferredNeuron]) -> FxHashMap<ChannelId, Vec<InferredAction>> {
        let mut channel_outputs: FxHashMap<ChannelId, Vec<InferredAction>> = FxHashMap::default();
        for inf in inferences {
            if self.get_neuron_type(inf.neuron_id) != Some(NeuronType::Action) { continue; }
            let channel_id = match self.get_neuron_channel_id(inf.neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let coordinate = match self.get_neuron_coordinate(inf.neuron_id) {
                Some(c) => c.clone(),
                None => continue,
            };
            channel_outputs.entry(channel_id)
                .or_insert_with(Vec::new)
                .push(InferredAction { coordinate });
        }
        channel_outputs
    }

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
                    new_neuron_specs.push(NeuronCreateSpec { id: lookup.id, forget_rate: 0.0, connections: None });
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
    pub fn process_level(
        &mut self,
        level: Level,
        level_neurons: &mut FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        memory_depth: u32,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
        learning: bool,
        phase: Phase,
    ) -> ProcessLevelResult {

        let mut orchestration = OrchestrationTimings::default();

        // Aggregate per-neuron work, build the shared level context, allocate error pattern specs.
        // Non-learning mode still builds the level context so pattern activation can run.
        // It skips error-correction allocation and per-age error feedback recording.
        let t = std::time::Instant::now();
        let (tasks, level_context, new_neuron_specs) =
            self.get_level_tasks(level, level_neurons, sensory_neurons, rewards, frame_number, new_error_pattern_ids, learning, phase);
        orchestration.get_level_tasks = t.elapsed().as_secs_f64();

        // Op-3: dispatch processFrame — the only cross-region round-trip in the level loop
        let t = std::time::Instant::now();
        let results = self.dispatch_frame(&tasks, memory_depth, &level_context, new_error_pattern_ids, &sensory_neurons[0], &rewards[0], frame_number, learning, phase);
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
        let mut context_ref_updates: FxHashMap<NeuronId, Vec<ContextRefUpdate>> = FxHashMap::default();
        for results in dispatch_results {
            for result in results {
                Self::collect_context_ref_updates(result, &mut context_ref_updates);
            }
        }

        if !context_ref_updates.is_empty() {
            let update_batch: Vec<(NeuronId, Vec<ContextRefUpdate>)> = context_ref_updates.into_iter().collect();
            self.dispatch_context_ref_updates(&update_batch);
        }
    }

    // ── Spatial error pass (1c) ─────────────────────────────────────────────

    /// Evaluate each spatial-fired neuron's d=0 votes against the observed L0 fired set.
    /// All spatial neurons predict L0 neighborhood (their `connections[0]` target L0 sensory),
    /// so error eval is always L0-predictions vs L0-observed.
    ///
    /// The CONTEXT for a minted correction, however, is drawn from the parent's OWN level —
    /// for an Lk parent, the L(k+1) correction's context_entries are the level-k fired set
    /// excluding the parent. This is what lets the hierarchy grow: an L1 partial-match → L0
    /// prediction error mints an L2 whose context is L1 neighbors, so L2 will fire next frame
    /// when L1 patterns recur in a similar L1-neighborhood.
    ///
    /// Symmetric mismatch — both missing predictions and novel observations count, per §4.3.
    pub fn mint_spatial_corrections(
        &mut self,
        spatial_dispatch_results: &[Vec<ColumnProcessResult>],
        spatial_fired: &FxHashSet<NeuronId>,
        rewards: &[FxHashMap<ChannelId, Reward>],
    ) -> (Vec<NeuronCreateSpec>, Vec<SpatialInstallOp>) {
        let mut new_specs = Vec::new();
        let mut install_ops = Vec::new();

        // Partition spatial_fired by spatial level. L0 keeps only event neurons (action neurons
        // aren't part of the co-activation neighborhood). L1+ are correction pattern neurons —
        // they have no NeuronType, so no type filter applies.
        let mut by_level: FxHashMap<Level, Vec<NeuronId>> = FxHashMap::default();
        for &id in spatial_fired {
            let level = self.get_neuron_spatial_level(id);
            if level == 0 && self.get_neuron_type(id) != Some(NeuronType::Event) { continue; }
            by_level.entry(level).or_insert_with(Vec::new).push(id);
        }

        // L0 event set — used for both error eval (predicted-vs-observed) and as the "actuals"
        // an Lk pattern's connections[0] is predicting.
        let l0_event_set: FxHashSet<NeuronId> = by_level.get(&0)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        // allocate_pattern_neuron at age=0 doesn't consume sensory_neurons (its inner loop is skipped),
        // so an empty slice is fine here.
        let empty: Vec<FxHashSet<NeuronId>> = Vec::new();

        for results in spatial_dispatch_results {
            for res in results {
                let parent_id = res.parent_id;
                let parent_level = self.get_neuron_spatial_level(parent_id);

                // Observed L0 events the parent's connections[0] should have predicted, minus the
                // parent itself (a neuron isn't its own co-activation partner — relevant only when
                // parent is L0).
                let observed_l0_minus_self: FxHashSet<NeuronId> = l0_event_set.iter()
                    .copied()
                    .filter(|&id| id != parent_id)
                    .collect();

                // Parent's neighborhood at its own spatial level — used as the context_entries of
                // any correction we mint for this parent. Excludes the parent itself.
                let neighborhood: Vec<NeuronId> = by_level.get(&parent_level)
                    .map(|v| v.iter().copied().filter(|&id| id != parent_id).collect())
                    .unwrap_or_default();

                for age_votes in &res.votes {
                    if age_votes.age != 0 { continue; }

                    // Predicted L0 events from the parent's connections[0].
                    let predicted_events: FxHashSet<NeuronId> = age_votes.votes.iter()
                        .filter(|v| self.get_neuron_type(v.neuron_id) == Some(NeuronType::Event))
                        .map(|v| v.neuron_id)
                        .collect();

                    // No predictions means no error to evaluate (bootstrap: parent has no d=0 connections yet).
                    if predicted_events.is_empty() { continue; }

                    let missing = predicted_events.difference(&observed_l0_minus_self).count();
                    let novel = observed_l0_minus_self.difference(&predicted_events).count();
                    let union_size = predicted_events.union(&observed_l0_minus_self).count();
                    if union_size == 0 { continue; }
                    let error_rate = (missing + novel) as f64 / union_size as f64;
                    if error_rate <= age_votes.threshold { continue; }

                    // Empty neighborhood means no signal to wire the correction's context against —
                    // it could never match anything in future frames. Skip (parallels the
                    // empty-state.context skip in get_level_corrections for temporal).
                    if neighborhood.is_empty() { continue; }

                    // Allocate the correction at one level deeper in the parent's spatial hierarchy.
                    let spec = self.allocate_pattern_neuron(
                        parent_level + 1,
                        parent_id,
                        0,
                        &empty,
                        rewards,
                        Phase::Spatial,
                    );

                    // Context entries: the parent's level-k neighborhood at distance=0.
                    let context_entries: Vec<ContextRefEntry> = neighborhood.iter()
                        .copied()
                        .map(|id| ContextRefEntry { neuron_id: id, distance: 0 })
                        .collect();

                    new_specs.push(NeuronCreateSpec {
                        id: spec.id,
                        forget_rate: spec.forget_rate,
                        connections: Some(spec.connections),
                    });
                    install_ops.push(SpatialInstallOp {
                        parent_id,
                        pattern_id: spec.id,
                        context_entries,
                    });
                    self.spatial_corrections_minted += 1;
                }
            }
        }

        (new_specs, install_ops)
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

        // Dispatch and collect deaths + context refs.
        let mut all_deaths: Vec<(NeuronId, FrameNumber)> = Vec::new();
        let mut all_context_refs: FxHashMap<NeuronId, Vec<ContextRefUpdate>> = FxHashMap::default();
        for (r, region_ops) in by_region.into_iter().enumerate() {
            if region_ops.is_empty() { continue; }
            let result = self.region_list[r].install_spatial_corrections(region_ops, frame_number);
            all_deaths.extend(result.deaths);
            for (target_id, updates) in result.context_ref_updates {
                all_context_refs.entry(target_id).or_insert_with(Vec::new).extend(updates);
            }
        }

        for (id, df) in all_deaths {
            self.register_death(id, df);
        }

        if !all_context_refs.is_empty() {
            let update_batch: Vec<(NeuronId, Vec<ContextRefUpdate>)> = all_context_refs.into_iter().collect();
            self.dispatch_context_ref_updates(&update_batch);
        }
    }

    /// Walk the active neurons at this level, contribute to the shared level context,
    /// pre-create error-correction pattern neurons for any (neuron, age) whose previous votes
    /// mismatched reality, and emit a task per neuron.
    fn get_level_tasks(
        &mut self,
        level: Level,
        level_neurons: &FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
        learning: bool,
        phase: Phase,
    ) -> (Vec<LevelTask>, Context, Vec<NeuronCreateSpec>) {
        let mut tasks = Vec::new();
        let mut level_context = Context::new();
        let mut new_neuron_specs = Vec::new();

        // collect neuron ids first to avoid borrow issues (level_neurons is immutable here,
        // but self is borrowed mutably for allocate_pattern_neuron)
        let neuron_entries: Vec<(NeuronId, FxHashMap<Distance, LevelAgeState>)> = level_neurons
            .iter()
            .map(|(&nid, ages)| (nid, ages.clone()))
            .collect();
        for (neuron_id, age_states) in &neuron_entries {

            // skip action neurons for learning or contexts if the channel learns without them
            if self.skip_action_neuron(*neuron_id) { continue; }

            // new error patterns only contribute to levelContext — they have no children,
            // history, or votes in their birth frame, so they skip dispatch and corrections.
            // Temporal context entries must be at age>0 (older than the parent's vote);
            // spatial context entries sit at age=0 (co-active on the current frame).
            if new_error_pattern_ids.contains(neuron_id) {
                for (&age, _) in age_states {
                    match phase {
                        Phase::Temporal => { if age > 0 { level_context.add_neuron(*neuron_id, age, 1.0); } }
                        Phase::Spatial => { if age == 0 { level_context.add_neuron(*neuron_id, age, 1.0); } }
                    }
                }
                continue;
            }

            // Populate level_context and (in learning mode) collect error corrections + per-age accuracy feedback.
            // 1b: spatial does not yet mint corrections — get_level_corrections returns empty for Spatial.
            let (corrections, error_feedback) = self.get_level_corrections(
                *neuron_id, level, &mut level_context, age_states, sensory_neurons, rewards, frame_number, learning, phase,
            );

            // extract creation specs for Op-4
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

            // emit the task to be dispatched to the neuron
            tasks.push(LevelTask {
                neuron_id: *neuron_id,
                age_states: neuron_age_states,
                corrections,
                error_feedback,
            });
        }

        (tasks, level_context, new_neuron_specs)
    }

    /// For a single active neuron: add its age>0 entries to the shared levelContext and create
    /// error-correction pattern neurons for ages whose previous votes mismatched reality.
    /// Non-learning mode still populates level_context (so pattern recognition at the next level has its context).
    /// It skips everything else — no accuracy-stats feedback, no error pattern allocation.
    fn get_level_corrections(
        &mut self,
        neuron_id: NeuronId,
        level: Level,
        level_context: &mut Context,
        age_states: &FxHashMap<Distance, LevelAgeState>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        learning: bool,
        phase: Phase,
    ) -> (Vec<CorrectionSpec>, Vec<ErrorFeedback>) {
        let mut corrections = Vec::new();
        let mut error_feedback = Vec::new();

        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            let state = &age_states[&age];

            // Temporal context entries are at strictly positive ages — older neurons predicting the parent.
            // Spatial context entries sit at age=0 — co-active on the current frame.
            match phase {
                Phase::Temporal => { if age > 0 { level_context.add_neuron(neuron_id, age, 1.0); } }
                Phase::Spatial => { if age == 0 { level_context.add_neuron(neuron_id, age, 1.0); } }
            }

            // 1b: spatial does not yet mint corrections or evaluate cross-frame votes.
            // Correction minting in process_spatial will land in 1c via a dedicated pass.
            if phase == Phase::Spatial { continue; }

            // Non-learning mode is done after level_context population — no accuracy stats, no error pattern allocation.
            if !learning { continue; }

            // skip if context is empty. empty context patterns can never match anything in future frames.
            // they would just keep regenerating useless siblings. we do this before vote error evaluation so that
            // empty context frames don't pollute the neuron's error-stats window either — Welford would
            // otherwise see misses the neuron had no chance to do better on, inflating future fire thresholds.
            if state.context.as_ref().map_or(true, |c| c.is_empty()) { continue; }

            // evaluate the prior-frame vote at this age (if any) and record feedback
            let result = match self.evaluate_vote_error(age, state, &sensory_neurons[0], frame_number) {
                Some(r) => r,
                None => continue,
            };
            error_feedback.push(ErrorFeedback { age, error_rate: result.error_rate });

            // skip if the error doesn't cross the dynamic threshold
            if !result.fire { continue; }

            // allocate an error correction pattern to be created after level processing
            let spec = self.allocate_pattern_neuron(level + 1, neuron_id, age, sensory_neurons, rewards, Phase::Temporal);
            corrections.push(CorrectionSpec {
                pattern_id: spec.id,
                forget_rate: spec.forget_rate,
                connections: spec.connections,
                age,
                context_entries: state.context.clone().unwrap(),
            });
        }

        (corrections, error_feedback)
    }

    /// Op-3 dispatch. Bucket tasks by region and dispatch to
    /// region.process_level; each region buckets per column.
    /// Concatenation order is region-index then column-index, stable across runs.
    fn dispatch_frame(
        &mut self,
        tasks: &[LevelTask],
        memory_depth: u32,
        level_context: &Context,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        age0: &FxHashSet<NeuronId>,
        current_rewards: &FxHashMap<ChannelId, Reward>,
        frame_number: FrameNumber,
        learning: bool,
        phase: Phase,
    ) -> Vec<ColumnProcessResult> {

        // decorate age=0 sensory neurons with channel id + pre-resolved reward
        let mut new_active_neurons = Vec::with_capacity(age0.len());
        for &neuron_id in age0 {
            let channel_id = self.get_neuron_channel_id(neuron_id).unwrap_or(0);
            let reward = if self.get_neuron_type(neuron_id) == Some(NeuronType::Action) {
                current_rewards.get(&channel_id).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            new_active_neurons.push(ActiveNeuron { id: neuron_id, channel_id, reward });
        }

        // bucket tasks by region
        let task_indices_by_region = self.bucket_by_region_indices(tasks, |t| t.neuron_id);

        // dispatch to each region, concatenate in region-index order
        let level_context_opt = if level_context.size() > 0 { Some(level_context) } else { None };
        let mut results = Vec::new();
        for (r, task_indices) in task_indices_by_region.iter().enumerate() {
            if task_indices.is_empty() { continue; }

            // build the tuples that Region.process_level expects
            let region_tasks: Vec<_> = task_indices.iter().map(|&i| {
                let task = &tasks[i];
                (task.neuron_id, task.age_states.clone(), task.corrections_as_neuron_corrections(), task.error_feedback.clone())
            }).collect();

            let region_results = self.region_list[r].process_level(
                &region_tasks, memory_depth, level_context_opt, new_error_pattern_ids,
                &new_active_neurons, frame_number, learning, phase,
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
    fn collect_context_ref_updates(result: &ColumnProcessResult, per_target: &mut FxHashMap<NeuronId, Vec<ContextRefUpdate>>) {
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

        // compare the inferred events to reality — events only, actions are
        // judged by reward not hit/miss
        let mut failed_events = 0u32;
        let mut total_events = 0u32;
        for vote in votes {
            if self.get_neuron_type(vote.neuron_id) == Some(NeuronType::Event) {
                total_events += 1;
                if !actual_neuron_ids.contains(&vote.neuron_id) { failed_events += 1; }
            }
        }
        if total_events == 0 { return None; }
        let error_rate = failed_events as f64 / total_events as f64;

        // the threshold rode in with the vote when it was cast last frame
        let threshold = state.threshold.unwrap_or(0.5);
        let fire = error_rate > threshold;
        Some(VoteErrorResult { fire, error_rate })
    }

    /// Check if a neuron should be skipped (action neuron in a channel whose spec does
    /// not include action-sequence learning).
    fn skip_action_neuron(&self, neuron_id: NeuronId) -> bool {
        if self.neuron_levels.get(&neuron_id) != Some(&0) { return false; }
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
    fn dispatch_context_ref_updates(&mut self, update_batch: &[(NeuronId, Vec<ContextRefUpdate>)]) {
        let mut by_region: Vec<Vec<(NeuronId, Vec<ContextRefUpdate>)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for (neuron_id, updates) in update_batch {
            let r = self.route_neuron(*neuron_id);
            by_region[r].push((*neuron_id, updates.clone()));
        }
        for (r, region_updates) in by_region.iter().enumerate() {
            if !region_updates.is_empty() {
                self.region_list[r].update_context_refs(region_updates);
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
            if self.neuron_levels.contains_key(neuron_id) { dead.push(*neuron_id); }
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
        let level = self.neuron_levels.remove(&id);
        self.neuron_spatial_levels.remove(&id);
        self.neuron_parents.remove(&id);
        if let Some(level) = level {
            self.decrement_level_count(level);
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
                let level = self.neuron_levels.get(&entry.id).copied().unwrap_or(0);
                let base_neuron = if level == 0 { self.base_neurons.get(&entry.id).cloned() } else { None };
                let parent_id = self.neuron_parents.get(&entry.id).copied();
                neurons.push(SnapshotNeuronEntry {
                    neuron: entry.neuron,
                    level,
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
            self.neuron_levels.insert(neuron_id, entry.level);
            if let Some(parent_id) = entry.parent_id {
                self.neuron_parents.insert(neuron_id, parent_id);
            }
            self.increment_level_count(entry.level);
            if entry.level == 0 {
                if let Some(ref base) = entry.base_neuron {
                    self.neurons_by_value.insert(base.coordinate.clone(), neuron_id);
                    self.base_neurons.insert(neuron_id, base.clone());
                }
            }
            let r = self.route_neuron(neuron_id);
            let c = self.region_list[r].route_neuron(neuron_id);
            buckets[r][c].push(entry.neuron.clone());
        }

        // distribute neurons to their owning columns
        for (r, region_buckets) in buckets.into_iter().enumerate() {
            self.region_list[r].restore_snapshot(region_buckets);
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
        self.neuron_levels.clear();
        self.neuron_spatial_levels.clear();
        self.neurons_by_value.clear();
        self.base_neurons.clear();
        self.neuron_parents.clear();
        self.death_ledger.clear();
        self.neuron_death_frame.clear();
        self.level_counts.clear();
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
    fn increment_level_count(&mut self, level: Level) {
        while self.level_counts.len() <= level as usize { self.level_counts.push(0); }
        self.level_counts[level as usize] += 1;
    }

    /// Decrement the neuron count at a given level.
    fn decrement_level_count(&mut self, level: Level) {
        if (level as usize) < self.level_counts.len() {
            self.level_counts[level as usize] -= 1;
        }
    }

    /// Get total number of neurons.
    pub fn get_neuron_count(&self) -> usize {
        self.neuron_levels.len()
    }

    /// Get the maximum level of any neuron currently in the registry.
    pub fn get_max_level(&self) -> Level {
        for i in (0..self.level_counts.len()).rev() {
            if self.level_counts[i] > 0 { return i as Level; }
        }
        0
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
struct LevelTask {
    neuron_id: NeuronId,
    age_states: FxHashMap<Distance, AgeState>,
    corrections: Vec<CorrectionSpec>,
    error_feedback: Vec<ErrorFeedback>,
}

impl LevelTask {
    /// Convert CorrectionSpecs to neuron::Corrections for dispatch.
    fn corrections_as_neuron_corrections(&self) -> Vec<Correction> {
        self.corrections.iter().map(|c| Correction {
            pattern_id: c.pattern_id,
            age: c.age,
            context_entries: c.context_entries.clone(),
        }).collect()
    }
}

/// Error-correction pattern spec — carries connection data for neuron creation.
#[derive(Debug, Clone)]
struct CorrectionSpec {
    pattern_id: NeuronId,
    forget_rate: f64,
    connections: Vec<ConnectionSpec>,
    age: Distance,
    context_entries: Vec<ContextRefEntry>,
}

/// Install op for a freshly-minted spatial correction. Records the parent whose routing table
/// gains the pattern, the new pattern's id, and the d=0 context entries to bind against.
#[derive(Debug, Clone)]
pub struct SpatialInstallOp {
    pub parent_id: NeuronId,
    pub pattern_id: NeuronId,
    pub context_entries: Vec<ContextRefEntry>,
}

/// Result of region.install_spatial_corrections — death frames for the death ledger plus
/// the context-ref updates the install produced (grouped by target neuron id).
pub struct SpatialInstallResult {
    pub deaths: Vec<(NeuronId, FrameNumber)>,
    pub context_ref_updates: FxHashMap<NeuronId, Vec<ContextRefUpdate>>,
}

/// Result of allocate_pattern_neuron.
pub struct PatternNeuronSpec {
    pub id: NeuronId,
    pub forget_rate: f64,
    pub connections: Vec<ConnectionSpec>,
}

/// An inferred action coordinate.
#[derive(Debug, Clone)]
pub struct InferredAction {
    pub coordinate: Coordinate,
}

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
        Thalamus::new(false, 0.1, 0.5, 4, ErrorMode::Static, 0.5, 1, 1)
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
        assert_eq!(t.get_neuron_level(1), Some(0));
    }

    #[test]
    fn test_routing() {
        let t = Thalamus::new(false, 0.1, 0.5, 4, ErrorMode::Static, 0.5, 3, 1);
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

        // reap — neuron 42 not in neuron_levels so won't be returned
        let dead = t.reap_dead_neurons(200);
        assert!(dead.is_empty());

        // register with level to simulate actual neuron
        t.neuron_levels.insert(42, 1);
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
        assert_eq!(t.get_max_level(), 0);
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
