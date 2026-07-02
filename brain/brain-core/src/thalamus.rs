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
use crate::footprint::{rebuild_footprint, Footprint};
use crate::neuron::{
    ActiveNeuron, AgeState, ContextRefEntry, TemporalContextRefUpdate, SpatialContextRefUpdate,
    Correction, ErrorFeedback,
    SerializedNeuron, Vote,
};
use crate::quantizer::{QuantizeMode, Quantizer};
use crate::region::Region;
use crate::types::{
    BucketId, ChannelId, Coordinate, DimensionId, Distance, GroupMode, FrameNumber,
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
/// Levels are not stored: a base neuron is one with `base_neuron` set; a correction is one with a
/// `parent_id`. Spatial vs temporal corrections are distinguished by the serialized child's `spatial`
/// flag, and a correction's depth is the wave's activation index, not a persisted field.
#[derive(Debug, Clone)]
pub struct SnapshotNeuronEntry {
    pub neuron: SerializedNeuron,
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

    /// Pattern neuron parents: correction id → the set of parent neurons that host it. A correction is
    /// minted with one parent; clustering and cross-frame reuse add more, so this is a set. A correction
    /// is reaped only when its parent set empties (refcounted reaping).
    neuron_parents: FxHashMap<NeuronId, FxHashSet<NeuronId>>,

    /// Cumulative count of spatial correction patterns minted by `mint_spatial_corrections`.
    /// Diagnostic — surfaced via Brain.get_spatial_correction_count() for harness validation.
    spatial_corrections_minted: u64,

    /// Spatial correction → the level it was minted at (parent.level + 1). The reuse lookup filters
    /// candidates to `mint_level == request.level + 1`, so a reused correction only ever gains parents
    /// one level below it and therefore stays single-level — reuse never creates the multi-depth case
    /// the spatial sweep can't represent. Transient (training-only): reuse runs only while learning, so
    /// this is rebuilt as corrections are minted and never serialized.
    correction_mint_level: FxHashMap<NeuronId, Level>,

    /// Per-neuron footprint: the set of base neurons a neuron ultimately covers, as a bitset over
    /// base-neuron bit indices. Base neuron = `{self}`; correction = `⋃ constituents` computed at mint.
    /// The universal neighborhood primitive — two neurons are neighbors iff their footprints touch in
    /// the base neighbor graph (`footprints_adjacent`), in both waves.
    neuron_footprints: FxHashMap<NeuronId, Footprint>,

    /// Dense base-neuron bit index: base neuron id → bit position in a footprint, assigned in
    /// allocation order. Base ids are interleaved with pattern ids, so footprints index off this dense
    /// position rather than the raw id.
    base_neuron_bit: FxHashMap<NeuronId, u32>,

    /// Reverse of `base_neuron_bit`: bit position → owning channel id. Indexed by bit; lets footprint
    /// adjacency resolve a set base bit back to its channel without a neuron lookup.
    base_bit_channel: Vec<ChannelId>,

    /// Reverse of `base_neuron_bit`: bit position → owning base neuron id.
    /// Indexed by bit; lets footprint resolution map a set base bit back to the coordinate-bearing neuron it stands for.
    /// This is how votes toward coordinate-less spatial apex events resolve to the base events that feed per-dimension consensus.
    base_bit_neuron: Vec<NeuronId>,

    /// Per-channel set of base-neuron bits — the bits of every base neuron registered in that channel.
    /// Footprint (spatial) adjacency dilates a parent footprint by unioning the base bits of its
    /// channels' base neighbors, which this provides directly.
    channel_base_bits: FxHashMap<ChannelId, Footprint>,

    /// Next base-neuron bit to assign. Advances by one per base neuron allocated; reset with `reset()`.
    next_base_bit: u32,

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

    /// THE base neighbor graph: per-channel set of neighbor channels at the base (sensory) level.
    /// There is one graph, used by footprint adjacency in BOTH waves — base neurons have a single
    /// (spatial) arrangement, so footprint touch is one relation, not a spatial and a temporal one.
    /// It is the set of channels a base neuron may neighbor — e.g. a pixel's adjacent pixels, or a set
    /// of genuinely correlated symbols. Channels NOT in the map default to all-pairs (no restriction),
    /// preserving original behavior for stocks/text. Declared via `set_spatial_neighbors`.
    base_neighbors: FxHashMap<ChannelId, FxHashSet<ChannelId>>,

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
            spatial_corrections_minted: 0,
            correction_mint_level: FxHashMap::default(),
            neuron_footprints: FxHashMap::default(),
            base_neuron_bit: FxHashMap::default(),
            base_bit_channel: Vec::new(),
            base_bit_neuron: Vec::new(),
            channel_base_bits: FxHashMap::default(),
            next_base_bit: 0,
            death_ledger: FxHashMap::default(),
            neuron_death_frame: FxHashMap::default(),
            channel_specs: FxHashMap::default(),
            dimension_specs: FxHashMap::default(),
            channel_actions,
            channel_default_actions: FxHashMap::default(),
            channel_name_to_id: FxHashMap::default(),
            channel_id_to_name: FxHashMap::default(),
            base_neighbors: FxHashMap::default(),
            dimension_name_to_id: FxHashMap::default(),
            dimension_id_to_name: FxHashMap::default(),
            quantizer: Quantizer::new(),
            next_channel_id: 1,
            next_dimension_id: 1,
            next_neuron_id: 1,
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
        self.neurons_by_value.insert(coordinate.clone(), id);
        self.base_neurons.insert(id, BaseNeuron { channel_id, neuron_type, coordinate: coordinate.clone() });
        self.assign_base_footprint(id, channel_id);
        id
    }

    /// Give a base neuron its dense bit and the singleton footprint `{self}`, and record the bit in its
    /// channel's base-bit set. Called when a base neuron is allocated and again on restore so the dense
    /// indexing and the per-channel base-bit sets are rebuilt before correction footprints are.
    fn assign_base_footprint(&mut self, id: NeuronId, channel_id: ChannelId) {
        let bit = self.next_base_bit;
        self.next_base_bit += 1;
        self.base_neuron_bit.insert(id, bit);
        if (bit as usize) >= self.base_bit_channel.len() {
            self.base_bit_channel.resize(bit as usize + 1, 0);
        }
        self.base_bit_channel[bit as usize] = channel_id;
        if (bit as usize) >= self.base_bit_neuron.len() {
            self.base_bit_neuron.resize(bit as usize + 1, 0);
        }
        self.base_bit_neuron[bit as usize] = id;
        self.channel_base_bits.entry(channel_id).or_insert_with(Footprint::new).set_bit(bit);
        self.neuron_footprints.insert(id, Footprint::single(bit));
    }

    /// Compute a correction's footprint as the union of its constituents' footprints: its parent plus
    /// the context neurons it binds. Constituents already carry footprints (they were allocated before
    /// this correction), so this is a straight union — base footprints ground it out.
    fn compute_correction_footprint(&self, parent_id: NeuronId, context_neuron_ids: &[NeuronId]) -> Footprint {
        let mut fp = Footprint::new();
        if let Some(p) = self.neuron_footprints.get(&parent_id) {
            fp.union_in_place(p);
        }
        for &c in context_neuron_ids {
            if let Some(cf) = self.neuron_footprints.get(&c) {
                fp.union_in_place(cf);
            }
        }
        fp
    }

    /// Whether two footprints are neighbors in the base neighbor graph — the SPATIAL locality test.
    ///
    /// Footprints are a spatial-only locality primitive: base neurons (pixels / sensory positions) have
    /// a single, spatial arrangement, so "do these two footprints touch" is a spatial question, and only
    /// the spatial wave uses it. Temporal has no neighborhood — it sequences against all active neurons —
    /// so this is never called on the temporal side.
    ///
    /// Implementation is the dilate-and-AND of [wavefront.md] without materializing the dilation: for
    /// each base bit of `parent_fp`, test the target against that bit's own footprint (overlap) and its
    /// neighbor channels' base bits. A parent base whose channel declared no neighbor list neighbors
    /// everything (the all-pairs default).
    pub fn footprints_adjacent(&self, parent_fp: &Footprint, target_fp: &Footprint) -> bool {
        if parent_fp.is_empty() || target_fp.is_empty() {
            return false;
        }
        // Overlapping footprints touch trivially (the "or equal" clause).
        if parent_fp.intersects(target_fp) {
            return true;
        }
        for bit in parent_fp.iter_bits() {
            let ch = self.base_bit_channel[bit as usize];
            match self.base_neighbors.get(&ch) {
                // No declared neighbor list ⇒ all-pairs: this base neighbors every channel.
                None => return true,
                Some(list) => {
                    for &c2 in list {
                        if let Some(cb) = self.channel_base_bits.get(&c2) {
                            if cb.intersects(target_fp) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Footprint for a neuron, or an empty footprint if it has none recorded.
    pub fn get_neuron_footprint(&self, neuron_id: NeuronId) -> Footprint {
        self.neuron_footprints.get(&neuron_id).cloned().unwrap_or_default()
    }

    /// Resolve a neuron's footprint to the (channel, bucket) of each base neuron it covers.
    /// Diagnostic for visualizing which sensory positions a spatial correction spans.
    /// Returns one entry per set footprint bit; empty if the neuron has no recorded footprint.
    pub fn footprint_bases(&self, neuron_id: NeuronId) -> Vec<(ChannelId, BucketId)> {
        let mut bases = Vec::new();
        if let Some(fp) = self.neuron_footprints.get(&neuron_id) {
            for bit in fp.iter_bits() {
                let base_id = self.base_bit_neuron[bit as usize];
                if let Some(b) = self.base_neurons.get(&base_id) {
                    bases.push((b.channel_id, b.coordinate.bucket_id));
                }
            }
        }
        bases
    }

    /// Invoke `f` once per base sensory/action neuron id in `neuron_id`'s footprint, allocation-free.
    /// A base neuron yields itself (`footprint == {self}`); a spatial apex event yields its constituent base events.
    /// This is how temporal votes toward the coordinate-less apex set fan out to base targets for per-dimension consensus.
    /// Only base neurons and spatial corrections carry footprints; temporal corrections intentionally carry none.
    /// A footprint-less neuron yields nothing — fine here, since the targets resolved are always base or spatial-apex,
    /// never temporal corrections (temporal connections wire only toward the temporal base level).
    pub fn for_each_base_neuron(&self, neuron_id: NeuronId, mut f: impl FnMut(NeuronId)) {
        if let Some(fp) = self.neuron_footprints.get(&neuron_id) {
            for bit in fp.iter_bits() {
                f(self.base_bit_neuron[bit as usize]);
            }
        }
    }

    /// Whether `target_id`'s footprint is adjacent to a precomputed parent footprint in the base
    /// neighbor graph. Resolves the target footprint by id so hot loops avoid cloning. A target with no
    /// footprint (should not occur — every neuron is given one) is treated as non-adjacent. Used by
    /// both waves: footprint adjacency is one relation, not a spatial and a temporal one.
    fn fp_adjacent_to(&self, parent_fp: &Footprint, target_id: NeuronId) -> bool {
        match self.neuron_footprints.get(&target_id) {
            Some(t) => self.footprints_adjacent(parent_fp, t),
            None => false,
        }
    }

    /// Transitively merge correction requests whose footprints touch in the base neighbor graph —
    /// connected components over the footprint-adjacency relation. This is the merge step of per-cluster
    /// minting: one coordinate-less correction is minted per returned component (its parents are the
    /// cluster's requests, its footprint the union of theirs). Returns clusters as index lists into
    /// `footprints`, each sorted ascending, in ascending first-member order — fully deterministic.
    /// Adjacency is directional (the all-pairs default makes a list-less channel a neighbor as a parent
    /// but not as a target), so membership uses the SYMMETRIC closure: i and j merge if either touches
    /// the other. Overlapping footprints merge trivially (`intersects` inside `footprints_adjacent`).
    #[allow(dead_code)] // superseded by cluster_by_target_similarity; kept (tested) for the locality variant
    fn cluster_by_footprint_adjacency(&self, footprints: &[Footprint]) -> Vec<Vec<usize>> {
        let n = footprints.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(i) = stack.pop() {
                comp.push(i);
                for j in 0..n {
                    if visited[j] { continue; }
                    if self.footprints_adjacent(&footprints[i], &footprints[j])
                        || self.footprints_adjacent(&footprints[j], &footprints[i]) {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
            comp.sort_unstable();
            clusters.push(comp);
        }
        clusters
    }

    /// Transitively merge requests whose L0 TARGET sets are similar — connected components over the
    /// `match_observed`-style Jaccard `|∩| / |∪|` ≥ the (pairwise-averaged) ADAPTIVE merge threshold.
    /// This is the grouping gate the clustering was missing: two requests merge only when they predict
    /// substantially the SAME correct L0, NOT merely because their footprints touch. Tightening the
    /// threshold makes clusters specific (less blur); loosening blurs them — this is the knob to sweep.
    /// Deterministic: components sorted ascending, in ascending first-member order.
    fn cluster_by_target_similarity(&self, targets: &[FxHashSet<NeuronId>], thresholds: &[f64]) -> Vec<Vec<usize>> {
        let n = targets.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(i) = stack.pop() {
                comp.push(i);
                for j in 0..n {
                    if visited[j] { continue; }
                    let inter = targets[i].intersection(&targets[j]).count();
                    if inter == 0 { continue; }
                    let union = targets[i].len() + targets[j].len() - inter;
                    let jaccard = inter as f64 / union as f64;
                    // Both requests must consider it a match — use the stricter (max) of their thresholds.
                    let thr = thresholds[i].max(thresholds[j]);
                    if jaccard >= thr {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
            comp.sort_unstable();
            clusters.push(comp);
        }
        clusters
    }

    /// Allocate a TEMPORAL pattern neuron. Pre-wires d=1..age connections toward each per-age
    /// active set, filtered by footprint adjacency to the parent — the new pattern only pre-wires
    /// toward footprint-adjacent targets, matching how learn_temporal_connections restricts future
    /// strengthening. The pattern's depth is the sweep loop variable, not a stored field.
    /// Does NOT touch the parent's routing table (that happens inside parent.process_temporal_frame
    /// via add_temporal_pattern) and does NOT register death (death frame is known only after
    /// parent.add_temporal_pattern runs).
    pub fn allocate_temporal_pattern_neuron(
        &mut self,
        parent_id: NeuronId,
        age: Distance,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
    ) -> PatternNeuronSpec {

        // Temporal has no neighborhood — a pattern sequences against ALL active neurons. Connections
        // pre-wire toward every active target across ages; footprints are a spatial-only locality
        // primitive and do not gate temporal sequencing.
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
        // A temporal correction is coordinate-less AND footprint-less — only an id and a parent.
        // Unlike a spatial correction it gets NO footprint, because footprints are a spatial-only locality primitive.
        // Temporal sequences against all active neurons with no neighborhood, so it needs no footprint to localize.
        // Output never resolves to a correction (votes toward coordinate-less targets are dropped before consensus).
        // It is never an apex or vote target either, so it needs no coordinate, channel, or type.
        // Reached solely via routing matches.
        // No stored level: a correction's depth is the sweep loop variable / activation index, not an intrinsic field.
        // The temporal sweep activates it at its parent's level + 1.
        self.neuron_parents.entry(id).or_default().insert(parent_id);

        PatternNeuronSpec { id, forget_rate: self.pattern_forget_rate, connections }
    }

    /// The forget rate to stamp on base sensory/action neurons. Base neurons never die (no parent →
    /// never reaped), but this rate drives the decay of their hosted correction children, so it must
    /// be the brain-wide `pattern_forget_rate` — otherwise (rate 0.0) every child pattern hosted on a
    /// base neuron would be immortal, which was a bug.
    pub fn base_neuron_forget_rate(&self) -> f64 {
        self.pattern_forget_rate
    }

    /// Allocate a SPATIAL pattern neuron. No pre-wired connections — spatial corrections are
    /// allocated with empty connections; learn_spatial_connections will fill them on future
    /// frames as the correction co-fires with others. Its depth in the spatial hierarchy is the sweep
    /// loop variable / activation index, not a stored field; it enters temporal via the apex handoff.
    /// Does NOT touch the parent's routing table (that happens inside column.install_spatial_corrections
    /// via add_spatial_pattern) and does NOT register death (death frame is known only after
    /// parent.add_spatial_pattern runs).
    pub fn allocate_spatial_pattern_neuron(
        &mut self,
        parent_id: NeuronId,
    ) -> PatternNeuronSpec {
        // allocate id and build the spec for Column.create_neurons
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;

        // register metadata centrally (Neuron construction deferred to create_neurons).
        // A correction is coordinate-less — id + footprint (set at mint by the caller), no base-neuron
        // entry. The apex handoff lifts it into the temporal base level as a coordinate-less token: it
        // carries the co-activation it represents, never an inherited pixel. Output never resolves to it
        // — votes toward coordinate-less targets are dropped before consensus — so it needs no
        // coordinate, channel, type, or stored level. Reached solely via routing matches; never
        // registered in neurons_by_value (value→neuron resolution must always land on the L0 base).
        self.neuron_parents.entry(id).or_default().insert(parent_id);

        PatternNeuronSpec { id, forget_rate: self.pattern_forget_rate, connections: Vec::new() }
    }

    // ── Neuron metadata getters ─────────────────────────────────────────────

    /// Get the channel id for a neuron. Sensory neurons return their registration channel.
    /// L1+ pattern neurons return None — they emerge from cross-channel correlations and have no
    /// channel of their own. Neighbor filtering no longer goes through channel ids — it uses footprint
    /// adjacency (`footprints_adjacent`) — so this is now used only for reward resolution, connection
    /// metadata, and consensus grouping.
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

    /// Declare the base-level neighbor channels for a registered channel — THE neighbor graph that
    /// footprint adjacency uses in both waves. The list is used VERBATIM — the channel is NOT
    /// implicitly added to its own set. An empty list therefore disables co-activation grouping for
    /// that channel entirely (no cross-channel AND no intra-channel grouping); to keep intra-channel
    /// grouping, list the channel itself. Channels with NO call retain the default all-pairs
    /// neighborhood. (Named `set_spatial_neighbors` for API stability — the base graph is the spatial
    /// adjacency of base neurons.)
    pub fn set_spatial_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        let (channel_id, neighbor_ids) = self.resolve_neighbor_ids(name, neighbor_names);
        self.base_neighbors.insert(channel_id, neighbor_ids);
    }

    /// Get the type for a neuron (Event or Action).
    pub fn get_neuron_type(&self, neuron_id: NeuronId) -> Option<NeuronType> {
        self.base_neurons.get(&neuron_id).map(|b| b.neuron_type)
    }

    /// A representative parent of a pattern neuron — any one of its parents. For diagnostics and the
    /// label/context walk, where any path up to a base neuron suffices. Use `get_neuron_parents` when the
    /// full set matters (install, reaping, delete cascade).
    pub fn get_neuron_parent(&self, neuron_id: NeuronId) -> Option<NeuronId> {
        self.neuron_parents.get(&neuron_id).and_then(|s| s.iter().next().copied())
    }

    /// The full set of parents that host a pattern neuron (correction). None for base neurons.
    pub fn get_neuron_parents(&self, neuron_id: NeuronId) -> Option<&FxHashSet<NeuronId>> {
        self.neuron_parents.get(&neuron_id)
    }

    /// Whether a neuron is a base (sensory/action) neuron — the explicit base-neuron predicate that
    /// replaces the old "temporal level == 0" test now that levels are not stored. A base neuron is
    /// exactly one with a base-neuron registry entry (equivalently, `footprint == {self}`).
    pub fn is_base_neuron(&self, neuron_id: NeuronId) -> bool {
        self.base_neurons.contains_key(&neuron_id)
    }

    /// Whether a neuron currently exists — a base neuron or a live correction (has a parent).
    pub fn neuron_exists(&self, neuron_id: NeuronId) -> bool {
        self.base_neurons.contains_key(&neuron_id) || self.neuron_parents.contains_key(&neuron_id)
    }

    /// Cumulative count of spatial corrections minted since brain start (or last hard reset).
    pub fn get_spatial_correction_count(&self) -> u64 {
        self.spatial_corrections_minted
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
        sensory_neurons: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
        learning: bool,
    ) -> ProcessLevelResult {
        let mut orchestration = OrchestrationTimings::default();

        // Aggregate per-neuron work and build the shared spatial level context.
        // Non-learning mode still builds the level context so pattern activation can run.
        let t = std::time::Instant::now();
        let (tasks, level_context) = self.get_spatial_level_tasks(level_neuron_ids, new_error_pattern_ids);
        orchestration.get_level_tasks = t.elapsed().as_secs_f64();

        // Op-3: dispatch processSpatialFrame — the only cross-region round-trip in the level loop.
        let t = std::time::Instant::now();
        let results = self.dispatch_spatial_frame(&tasks, &level_context, new_error_pattern_ids, sensory_neurons, frame_number, learning);
        orchestration.dispatch_frame = t.elapsed().as_secs_f64();

        // extract activations inline — needed to feed the next level
        let t = std::time::Instant::now();
        let mut activations = Vec::new();
        for result in &results {
            self.collect_activations(result, &mut activations);
        }
        orchestration.collect_activations = t.elapsed().as_secs_f64();

        // Spatial doesn't persist per-neuron state, so there's no "stale state" to clear and
        // no need to write back per-age votes — the spatial sweep doesn't reread them next frame.
        // We still collect votes for downstream consumers (e.g. mint_spatial_corrections reads
        // them through the dispatch results, not through level_neurons).
        let mut votes = Vec::new();
        for result in &results {
            for vote_bucket in &result.votes {
                for v in &vote_bucket.votes {
                    votes.push(FlatVote {
                        voter_id: result.parent_id,
                        neuron_id: v.neuron_id,
                        distance: v.distance,
                        strength: v.strength,
                        reward: v.reward,
                    });
                }
            }
        }

        // Reverse connection index (reuse Phase A): register each source on the reverse index of every
        // d=0 target it newly connected to this level. Routed by target at the orchestration boundary, so
        // a same-frame reuse lookup sees this level's edges. Unconsumed until reuse lookup (Phase C).
        let mut conn_sources_by_target: FxHashMap<NeuronId, Vec<NeuronId>> = FxHashMap::default();
        for result in &results {
            for &target in &result.new_spatial_connection_targets {
                conn_sources_by_target.entry(target).or_insert_with(Vec::new).push(result.parent_id);
            }
        }
        if !conn_sources_by_target.is_empty() {
            let batch: Vec<(NeuronId, Vec<NeuronId>)> = conn_sources_by_target.into_iter().collect();
            self.dispatch_spatial_connection_sources(&batch);
        }

        if self.debug && !activations.is_empty() {
            let detail: Vec<String> = activations.iter()
                .map(|a| format!("parent={}, age={}, pattern={}", a.parent_id, a.age, a.pattern_id))
                .collect();
            println!("Spatial level {}: {} activations {}", level, activations.len(), detail.join("; "));
        }

        ProcessLevelResult { activations, votes, neuron_specs: Vec::new(), results, orchestration }
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
        learning: bool,
    ) -> ProcessLevelResult {
        let mut orchestration = OrchestrationTimings::default();

        // Aggregate per-neuron work, build the shared level context, allocate error pattern specs.
        // Non-learning mode still builds the level context so pattern activation can run.
        // It skips error-correction allocation and per-age error feedback recording.
        let t = std::time::Instant::now();
        let (tasks, level_context, new_neuron_specs) =
            self.get_temporal_level_tasks(level, level_neurons, sensory_neurons, rewards, frame_number, new_error_pattern_ids, learning);
        orchestration.get_level_tasks = t.elapsed().as_secs_f64();

        // Op-3: dispatch processTemporalFrame — the only cross-region round-trip in the level loop.
        let t = std::time::Instant::now();
        let current_rewards = rewards.first();
        let results = self.dispatch_temporal_frame(&tasks, memory_depth, &level_context, new_error_pattern_ids, &sensory_neurons[0], current_rewards, frame_number, learning);
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
        spatial_subsumed: &FxHashSet<NeuronId>,
        spatial_levels: &FxHashMap<NeuronId, Level>,
    ) -> (Vec<NeuronCreateSpec>, Vec<SpatialInstallOp>, Vec<(NeuronId, f64)>) {
        let mut new_specs = Vec::new();
        let mut install_ops = Vec::new();

        // Per-parent error feedback for `record_spatial_errors` — collected for every fired neuron
        // that had any predictions, regardless of whether the error crosses the mint threshold.
        // This is what lets dynamic error modes (conservative/neutral/aggressive) adapt — without
        // these samples, get_spatial_error_threshold() stays in static-fallback mode forever.
        let mut error_feedback: Vec<(NeuronId, f64)> = Vec::new();

        // Partition spatial_fired by spatial level. L0 keeps only event neurons (action neurons
        // aren't part of the co-activation neighborhood). L1+ are correction pattern neurons —
        // they have no NeuronType, so no type filter applies.
        let mut by_level: FxHashMap<Level, Vec<NeuronId>> = FxHashMap::default();
        for &id in spatial_fired {

            // exclude action neurons
            if self.get_neuron_type(id) == Some(NeuronType::Action) { continue; }

            // Depth is activation-derived: the level the neuron fired at this frame, read from the
            // sweep's activation index rather than a stored field. Absent ⇒ base (level 0).
            let level = spatial_levels.get(&id).copied().unwrap_or(0);
            by_level.entry(level).or_insert_with(Vec::new).push(id);
        }

        // L0 event set — used for both error eval (predicted-vs-observed) and as the "actuals"
        // an Lk pattern's connections[0] is predicting.
        let l0_event_set: FxHashSet<NeuronId> = by_level.get(&0)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        // FIRST PASS — collect correction REQUESTS. A request is a non-subsumed parent whose d=0
        // prediction crossed the error threshold and has a non-empty neighborhood to condition on.
        // Error feedback is recorded PER PARENT (unchanged), so the Welford / dynamic-threshold stats
        // build up exactly as before regardless of clustering; only the MINT becomes per-cluster.
        // Each request: (parent, level, neighborhood, L0 targets, merge threshold). The targets and
        // threshold drive the Phase-C reuse lookup; neighborhood/level drive clustering and the mint.
        let mut requests: Vec<(NeuronId, Level, Vec<NeuronId>, FxHashSet<NeuronId>, f64)> = Vec::new();
        for level_results in spatial_dispatch_results {
            for neuron_result in level_results {
                let parent_id = neuron_result.parent_id;

                // if this parent was already explained by a higher-level pattern that fired this frame,
                // its role is represented — don't record an error sample and don't request a correction.
                if spatial_subsumed.contains(&parent_id) { continue; }

                let parent_level = spatial_levels.get(&parent_id).copied().unwrap_or(0);
                let parent_fp = self.get_neuron_footprint(parent_id);

                // Observed L0 events the parent's connections[0] should have predicted, minus the parent
                // and any L0 event whose footprint is not adjacent to the parent's (all-pairs default for
                // a channel with no declared neighbor list).
                let observed_l0_minus_self: FxHashSet<NeuronId> = l0_event_set.iter()
                    .copied()
                    .filter(|&id| id != parent_id)
                    .filter(|&id| self.fp_adjacent_to(&parent_fp, id))
                    .collect();

                // Parent's same-level neighborhood, footprint-adjacent and excluding itself.
                let neighborhood: Vec<NeuronId> = by_level.get(&parent_level)
                    .map(|v| v.iter().copied()
                        .filter(|&id| id != parent_id)
                        .filter(|&id| self.fp_adjacent_to(&parent_fp, id))
                        .collect())
                    .unwrap_or_default();

                for age_votes in &neuron_result.votes {
                    if age_votes.age != 0 { continue; }

                    // Predicted L0 events from connections[0], after per-position competition (the modal
                    // local patch): each (channel, dim) position keeps only its strongest-voted bucket, so
                    // the error below is a Hamming distance from the modal patch, not novelty-vs-everything.
                    let mut position_winners: FxHashMap<(ChannelId, DimensionId), (NeuronId, f64)> = FxHashMap::default();
                    for v in &age_votes.votes {
                        if self.get_neuron_type(v.neuron_id) != Some(NeuronType::Event) { continue; }
                        if !self.fp_adjacent_to(&parent_fp, v.neuron_id) { continue; }
                        let target_channel = self.get_neuron_channel_id(v.neuron_id).unwrap_or(0);
                        let dim_id = match self.get_neuron_coordinate(v.neuron_id) {
                            Some(c) => c.dim_id,
                            None => continue,
                        };
                        let key = (target_channel, dim_id);
                        match position_winners.get(&key) {
                            // Strongest bucket wins; tie-break on smaller neuron id for determinism.
                            Some(&(win_id, win_strength))
                                if win_strength > v.strength
                                    || (win_strength == v.strength && win_id <= v.neuron_id) => {}
                            _ => { position_winners.insert(key, (v.neuron_id, v.strength)); }
                        }
                    }
                    let predicted_events: FxHashSet<NeuronId> = position_winners.values().map(|&(id, _)| id).collect();

                    // No predictions means no error to evaluate (bootstrap: parent has no d=0 connections yet).
                    if predicted_events.is_empty() { continue; }

                    // Jaccard-union error over predicted-vs-observed L0 events (shared with temporal).
                    let error_rate = match Self::get_union_error(&predicted_events, &observed_l0_minus_self) {
                        Some(e) => e,
                        None => continue, // empty union → nothing to compare
                    };

                    // Per-parent error sample — recorded even below threshold so dynamic modes can adapt.
                    error_feedback.push((parent_id, error_rate));

                    if crate::config::trace_error() {
                        eprintln!(
                            "[error] parent={} parent_lvl={} predicted={} observed={} error={:.3} thr={:.3} {}",
                            parent_id, parent_level, predicted_events.len(), observed_l0_minus_self.len(),
                            error_rate, age_votes.threshold,
                            if error_rate > age_votes.threshold { "MINT" } else { "ok" }
                        );
                    }

                    if error_rate <= age_votes.threshold { continue; }
                    if neighborhood.is_empty() { continue; }

                    // A correction request — reuse/clustering/minting is deferred to the passes below.
                    // merge_threshold = 1 − error_threshold: a candidate must predict ≥ this fraction of
                    // the request's L0 targets to be reused (same threshold recognition uses).
                    requests.push((parent_id, parent_level, neighborhood.clone(), observed_l0_minus_self.clone(), 1.0 - age_votes.threshold));
                }
            }
        }

        // REUSE LOOKUP (Phase C) — before clustering, each request tries to REUSE an existing correction
        // that already predicts its L0 targets ≥ its merge threshold, at the level a fresh mint would
        // produce (mint_level == level + 1). A hit wires the request's parent to that correction and
        // installs it (no new neuron); a miss stays FRESH for clustering/minting. Cross-image reuse is
        // what bends per-image minting down — recurring structure lands on a shared correction instead of
        // a fresh one. (Footprint expansion of the reused pattern is deferred — an open knob per the doc.)
        // Gated by the `BRAIN_REUSE` ablation toggle (default ON): when off, every request stays FRESH
        // (skips the reuse lookup entirely) so each correction mints anew, isolating reuse's effect on count.
        let mut fresh: Vec<usize> = Vec::new();
        let reuse_on = crate::config::reuse_enabled();
        for i in 0..requests.len() {
            if !reuse_on { fresh.push(i); continue; }
            let parent_id = requests[i].0;
            let candidate_level = requests[i].1 + 1;
            let merge_threshold = requests[i].4;
            match self.find_reusable_correction(&requests[i].3, candidate_level, merge_threshold, parent_id) {
                Some(p) => {
                    self.neuron_parents.entry(p).or_default().insert(parent_id);
                    let mut context_neuron_ids = requests[i].2.clone();
                    context_neuron_ids.sort_unstable();
                    install_ops.push(SpatialInstallOp { parent_id, pattern_id: p, context_neuron_ids });
                }
                None => fresh.push(i),
            }
        }

        // SECOND PASS — per-cluster mint of the FRESH (non-reused) requests. Cluster them WITHIN each
        // spatial level by TARGET SIMILARITY (match_observed Jaccard ≥ adaptive threshold) — only requests
        // predicting substantially the same correct L0 merge, NOT merely footprint-adjacent ones — then
        // mint ONE coordinate-less correction per cluster: parents = the cluster's requests (multi-parent),
        // footprint = union of theirs, context = union of their neighborhoods minus their own parents.
        // A singleton cluster reproduces the old single-parent mint exactly. The threshold is the knob.
        let mut by_req_level: FxHashMap<Level, Vec<usize>> = FxHashMap::default();
        for &i in &fresh {
            by_req_level.entry(requests[i].1).or_insert_with(Vec::new).push(i);
        }
        let mut req_levels: Vec<Level> = by_req_level.keys().copied().collect();
        req_levels.sort_unstable();

        for level in req_levels {
            let req_indices = by_req_level.get(&level).cloned().unwrap_or_default();
            let target_sets: Vec<FxHashSet<NeuronId>> = req_indices.iter()
                .map(|&i| requests[i].3.clone())
                .collect();
            let thresholds: Vec<f64> = req_indices.iter()
                .map(|&i| requests[i].4)
                .collect();
            let clusters = self.cluster_by_target_similarity(&target_sets, &thresholds);

            for cluster in clusters {
                // Map local cluster indices back to request indices → the cluster's parents.
                let members: Vec<usize> = cluster.iter().map(|&ci| req_indices[ci]).collect();
                let parents: Vec<NeuronId> = members.iter().map(|&mi| requests[mi].0).collect();

                // Context = union of the cluster's neighborhoods minus the cluster's own parents (a
                // correction conditions on its NEIGHBORS, not on the units it corrects). Sorted for
                // deterministic install order.
                let mut context_set: FxHashSet<NeuronId> = FxHashSet::default();
                for &mi in &members {
                    for &n in &requests[mi].2 { context_set.insert(n); }
                }
                for &p in &parents { context_set.remove(&p); }
                if context_set.is_empty() { continue; }
                let mut context_neuron_ids: Vec<NeuronId> = context_set.into_iter().collect();
                context_neuron_ids.sort_unstable();

                // Allocate the shared correction; record EVERY cluster member as a parent.
                let spec = self.allocate_spatial_pattern_neuron(parents[0]);
                for &p in &parents[1..] {
                    self.neuron_parents.entry(spec.id).or_default().insert(p);
                }

                // Footprint = union of all constituents (every parent ∪ the bound context).
                let mut fp = self.compute_correction_footprint(parents[0], &context_neuron_ids);
                for &p in &parents[1..] { fp.union_in_place(&self.get_neuron_footprint(p)); }
                self.neuron_footprints.insert(spec.id, fp);

                // Record the mint level (one deeper than the cluster's parents) so the reuse lookup can
                // keep this correction single-level (it may only be reused by level-`level` requests).
                self.correction_mint_level.insert(spec.id, level + 1);

                new_specs.push(NeuronCreateSpec {
                    id: spec.id,
                    forget_rate: spec.forget_rate,
                    connections: Some(spec.connections),
                });
                // Install the ONE correction into EVERY parent's routing table (multi-parent).
                for &p in &parents {
                    install_ops.push(SpatialInstallOp {
                        parent_id: p,
                        pattern_id: spec.id,
                        context_neuron_ids: context_neuron_ids.clone(),
                    });
                }
                self.spatial_corrections_minted += 1;
            }
        }

        (new_specs, install_ops, error_feedback)
    }

    /// Find an existing correction to REUSE for a request predicting `targets` (its correct L0). A
    /// candidate qualifies iff it (a) was minted at `candidate_level` — the level a fresh mint would land
    /// at, which keeps reuse single-level — (b) is not the requesting parent, and (c) predicts ≥
    /// `merge_threshold` of the request's L0 targets, measured via the reverse connection index (how many
    /// of `targets` the candidate has a d=0 connection to). Returns the best-covering candidate
    /// (tie-break: smaller id) or None to mint fresh. Strength-blind — the reverse index is membership-only.
    fn find_reusable_correction(
        &self,
        targets: &FxHashSet<NeuronId>,
        candidate_level: Level,
        merge_threshold: f64,
        requesting_parent: NeuronId,
    ) -> Option<NeuronId> {
        if targets.is_empty() { return None; }

        // Per-candidate coverage = how many of the request's L0 targets the candidate connects to.
        let mut coverage: FxHashMap<NeuronId, usize> = FxHashMap::default();
        for &t in targets {
            for src in self.get_spatial_connection_sources(t) {
                if src == requesting_parent { continue; }
                if self.is_base_neuron(src) { continue; }
                if self.correction_mint_level.get(&src).copied() != Some(candidate_level) { continue; }
                *coverage.entry(src).or_insert(0) += 1;
            }
        }

        // Score each candidate by Jaccard similarity of its L0 prediction to the request's targets:
        // |∩| / |∪|, with |∪| = |candidate connections| + |targets| − |∩|. This PENALIZES over-general
        // candidates — a correction that has accumulated broad connections has a large union and so a low
        // score, so it is NOT reused. That is what stops the merge-runaway: only a candidate whose
        // prediction genuinely matches the request (score ≥ θ) is reused; otherwise the request mints its
        // own specific correction. Best score wins, tie-break on smaller id for determinism.
        let n_targets = targets.len();
        let mut best: Option<(NeuronId, f64)> = None;
        let mut best_raw: (f64, usize, usize) = (0.0, 0, 0); // (score, cov, cand_conns) ignoring threshold
        for (&cand, &cov) in &coverage {
            let cand_count = self.get_spatial_connection_count(cand);
            let union = cand_count + n_targets - cov; // cov ≤ min(cand_count, n_targets) ⇒ union ≥ both
            if union == 0 { continue; }
            let score = cov as f64 / union as f64;
            if score > best_raw.0 { best_raw = (score, cov, cand_count); }
            if score < merge_threshold { continue; }
            match best {
                Some((bid, bscore)) if bscore > score || (bscore == score && bid <= cand) => {}
                _ => best = Some((cand, score)),
            }
        }
        if crate::config::trace_reuse() {
            eprintln!(
                "[reuse] targets={} cands={} best_jac={:.3} (cov={} cand_conns={}) thr={:.3} {}",
                n_targets, coverage.len(), best_raw.0, best_raw.1, best_raw.2, merge_threshold,
                if best.is_some() { "HIT" } else { "MISS" }
            );
        }
        best.map(|(id, _)| id)
    }

    /// Dispatch spatial error-feedback samples to the owning columns, where each neuron records
    /// the rate into its `error_stats[0]` Welford bucket. Once the bucket has ≥3 samples, dynamic
    /// error modes (conservative/neutral/aggressive) start replacing the static fallback with the
    /// per-neuron adaptive threshold.
    pub fn record_spatial_errors(&mut self, feedback: &[(NeuronId, f64)]) {
        if feedback.is_empty() { return; }

        let mut by_region: Vec<Vec<(NeuronId, f64)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for &(id, rate) in feedback {
            let r = self.route_neuron(id);
            by_region[r].push((id, rate));
        }

        for (r, region_fb) in by_region.iter().enumerate() {
            if !region_fb.is_empty() {
                self.region_list[r].record_spatial_errors(region_fb);
            }
        }
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

    /// Route reverse-connection-index updates to the columns owning each target neuron — the connection
    /// analog of `dispatch_spatial_context_ref_updates`. Each entry is (target, sources): every source
    /// gains an edge into the target's `spatial_connection_sources`.
    fn dispatch_spatial_connection_sources(&mut self, update_batch: &[(NeuronId, Vec<NeuronId>)]) {
        let mut by_region: Vec<Vec<(NeuronId, Vec<NeuronId>)>> = (0..self.regions).map(|_| Vec::new()).collect();
        for (target_id, sources) in update_batch {
            let r = self.route_neuron(*target_id);
            by_region[r].push((*target_id, sources.clone()));
        }
        for (r, region_updates) in by_region.into_iter().enumerate() {
            if region_updates.is_empty() { continue; }
            self.region_list[r].update_spatial_connection_sources(&region_updates);
        }
    }

    /// Reuse candidate generator (Phase A): the source neurons with a d=0 connection to `target`, read
    /// from the target neuron's reverse connection index. The reuse lookup (Phase C) uses this to find
    /// existing patterns that already predict an L0 target. Empty if the target is unknown.
    pub fn get_spatial_connection_sources(&self, target: NeuronId) -> FxHashSet<NeuronId> {
        let r = self.route_neuron(target);
        self.region_list[r].get_spatial_connection_sources(target).unwrap_or_default()
    }

    /// Count of a candidate correction's d=0 connections — its prediction breadth, used as the Jaccard
    /// denominator in reuse scoring so an over-general candidate scores low and is not reused.
    pub fn get_spatial_connection_count(&self, neuron_id: NeuronId) -> usize {
        let r = self.route_neuron(neuron_id);
        self.region_list[r].spatial_connection_count(neuron_id).unwrap_or(0)
    }

    /// SPATIAL: walk the active neurons at this level and build the shared SpatialContext.
    /// Spatial corrections are minted in a separate pass (`mint_spatial_corrections`) after the
    /// sweep, NOT here — so no corrections, no per-age error feedback, no new_neuron_specs.
    /// Action neurons are NOT skipped — spatial co-activation includes everything that fired.
    fn get_spatial_level_tasks(
        &mut self,
        level_neuron_ids: &FxHashSet<NeuronId>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
    ) -> (Vec<LevelTask>, SpatialContext) {
        let _ = level_neuron_ids; // silence unused warning when no neurons fired
        let mut tasks = Vec::new();
        let mut level_context = SpatialContext::new();

        for &neuron_id in level_neuron_ids {

            // New error patterns only contribute to level_context — they have no children, history,
            // or votes in their birth frame, so they skip dispatch.
            if new_error_pattern_ids.contains(&neuron_id) {
                level_context.add_neuron(neuron_id, 1.0);
                continue;
            }

            // All other active neurons contribute to level_context AND get a dispatch task.
            level_context.add_neuron(neuron_id, 1.0);

            // Spatial age_states is always {0 → default}. Kept in the LevelTask shape so column
            // dispatch can stay one type; only the AgeState at age=0 is read inside the neuron.
            let mut age_states = FxHashMap::default();
            age_states.insert(0, AgeState { activated_pattern_id: None });
            tasks.push(LevelTask {
                neuron_id,
                age_states,
                corrections: Vec::new(),
                error_feedback: Vec::new(),
            });
        }

        (tasks, level_context)
    }

    /// TEMPORAL: walk the active neurons at this level, contribute to the shared TemporalContext,
    /// pre-create error-correction pattern neurons for any (neuron, age) whose previous votes
    /// mismatched reality, and emit a task per neuron.
    fn get_temporal_level_tasks(
        &mut self,
        level: Level,
        level_neurons: &FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        new_error_pattern_ids: &mut FxHashSet<NeuronId>,
        learning: bool,
    ) -> (Vec<LevelTask>, TemporalContext, Vec<NeuronCreateSpec>) {
        let mut tasks = Vec::new();
        let mut level_context = TemporalContext::new();
        let mut new_neuron_specs = Vec::new();

        // collect neuron ids first to avoid borrow issues (level_neurons is immutable here,
        // but self is borrowed mutably for allocate_temporal_pattern_neuron)
        let neuron_entries: Vec<(NeuronId, FxHashMap<Distance, LevelAgeState>)> = level_neurons
            .iter()
            .map(|(&nid, ages)| (nid, ages.clone()))
            .collect();
        for (neuron_id, age_states) in &neuron_entries {

            // skip action neurons for learning or contexts if the channel learns without them
            if self.skip_action_neuron(*neuron_id) { continue; }

            // New error patterns only contribute to level_context — they have no children, history,
            // or votes in their birth frame, so they skip dispatch and corrections.
            if new_error_pattern_ids.contains(neuron_id) {
                for (&age, _) in age_states {
                    if age > 0 { level_context.add_neuron(*neuron_id, age, 1.0); }
                }
                continue;
            }

            // Populate level_context and (in learning mode) collect error corrections + per-age accuracy feedback.
            let (corrections, error_feedback) = self.get_temporal_level_corrections(
                *neuron_id, level, &mut level_context, age_states, sensory_neurons, rewards, frame_number, learning,
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

            tasks.push(LevelTask {
                neuron_id: *neuron_id,
                age_states: neuron_age_states,
                corrections,
                error_feedback,
            });
        }

        (tasks, level_context, new_neuron_specs)
    }

    /// For a single active temporal neuron: add its age>0 entries to the shared level_context and
    /// create error-correction pattern neurons for ages whose previous votes mismatched reality.
    /// Non-learning mode still populates level_context (so pattern recognition at the next level
    /// has its context). It skips everything else.
    fn get_temporal_level_corrections(
        &mut self,
        neuron_id: NeuronId,
        _level: Level,
        level_context: &mut TemporalContext,
        age_states: &FxHashMap<Distance, LevelAgeState>,
        sensory_neurons: &[FxHashSet<NeuronId>],
        rewards: &[FxHashMap<ChannelId, Reward>],
        frame_number: FrameNumber,
        learning: bool,
    ) -> (Vec<CorrectionSpec>, Vec<ErrorFeedback>) {
        let mut corrections = Vec::new();
        let mut error_feedback = Vec::new();

        // Temporal has no neighborhood — actuals, context, and connections span ALL active neurons.
        // Footprints are a spatial-only locality primitive and do not gate temporal sequencing.

        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            let state = &age_states[&age];

            // Temporal context entries are at strictly positive ages — older neurons predicting the parent.
            if age > 0 { level_context.add_neuron(neuron_id, age, 1.0); }

            // Non-learning mode is done after level_context population — no accuracy stats, no error pattern allocation.
            if !learning { continue; }

            // skip if context is empty. empty context patterns can never match anything in future frames.
            // they would just keep regenerating useless siblings. we do this before vote error evaluation so that
            // empty context frames don't pollute the neuron's error-stats window either — Welford would
            // otherwise see misses the neuron had no chance to do better on, inflating future fire thresholds.
            if state.context.as_ref().map_or(true, |c| c.is_empty()) { continue; }

            // The actuals for vote-error evaluation are this frame's full active event set — temporal
            // sequences against all active neurons, no neighborhood restriction.
            let actuals: FxHashSet<NeuronId> = sensory_neurons[0].iter().copied().collect();

            // evaluate the prior-frame vote at this age (if any) and record feedback
            let result = match self.evaluate_vote_error(age, state, &actuals, frame_number) {
                Some(r) => r,
                None => continue,
            };
            error_feedback.push(ErrorFeedback { age, error_rate: result.error_rate });

            // skip if the error doesn't cross the dynamic threshold
            if !result.fire { continue; }

            // allocate an error correction pattern to be created after level processing
            let spec = self.allocate_temporal_pattern_neuron(neuron_id, age, sensory_neurons, rewards);

            // The correction's context is the full temporal context — no neighborhood restriction.
            let context_entries: Vec<ContextRefEntry> = state.context.clone().unwrap();

            corrections.push(CorrectionSpec {
                pattern_id: spec.id,
                forget_rate: spec.forget_rate,
                connections: spec.connections,
                age,
                context_entries,
            });
        }

        (corrections, error_feedback)
    }

    /// Spatial Op-3 dispatch — no rewards, no per-task channel-reward resolution.
    fn dispatch_spatial_frame(
        &mut self,
        tasks: &[LevelTask],
        level_context: &SpatialContext,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        age0: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
        learning: bool,
    ) -> Vec<ColumnProcessResult> {
        // Decorate age=0 sensory neurons. Spatial co-activation has no reward semantics, so every
        // active gets reward=0.0 regardless of neuron type.
        let mut full_actives = Vec::with_capacity(age0.len());
        for &neuron_id in age0 {
            let channel_id = self.get_neuron_channel_id(neuron_id).unwrap_or(0);
            full_actives.push(ActiveNeuron { id: neuron_id, channel_id, reward: 0.0 });
        }

        // Per-task actives filtered to footprints adjacent to the parent's.
        let task_actives: Vec<Vec<ActiveNeuron>> = tasks.iter().map(|t| {
            let parent_fp = self.get_neuron_footprint(t.neuron_id);
            full_actives.iter()
                .filter(|a| self.fp_adjacent_to(&parent_fp, a.id))
                .cloned()
                .collect()
        }).collect();

        // Per-task observed context — the shared level context filtered to footprints adjacent to the
        // parent's and minus the parent itself (fix 1.1). Without this, every neuron matched against
        // the unfiltered whole-level co-activation set, which counted every non-adjacent active as
        // novel and drove the Jaccard score below any sane threshold, so spatial matching never fired.
        // Mirrors the footprint-filtering already applied to `task_actives` and to the mint pass's
        // `observed_l0_minus_self` / `neighborhood`.
        let task_contexts: Vec<SpatialContext> = tasks.iter().map(|t| {
            let parent_fp = self.get_neuron_footprint(t.neuron_id);
            let mut ctx = SpatialContext::new();
            for (&neuron_id, &strength) in level_context.entries() {
                if neuron_id == t.neuron_id { continue; }
                if self.fp_adjacent_to(&parent_fp, neuron_id) {
                    ctx.add_neuron(neuron_id, strength);
                }
            }
            ctx
        }).collect();

        let task_indices_by_region = self.bucket_by_region_indices(tasks, |t| t.neuron_id);
        let mut results = Vec::new();
        for (r, task_indices) in task_indices_by_region.iter().enumerate() {
            if task_indices.is_empty() { continue; }
            let region_tasks: Vec<_> = task_indices.iter().map(|&i| {
                let task = &tasks[i];
                (task.neuron_id, task.age_states.clone(), task.corrections_as_neuron_corrections(), task.error_feedback.clone(), task_actives[i].clone(), task_contexts[i].clone())
            }).collect();
            let region_results = self.region_list[r].process_spatial_level(
                &region_tasks, new_error_pattern_ids,
                frame_number, learning,
            );
            results.extend(region_results);
        }
        results
    }

    /// Temporal Op-3 dispatch — resolves per-action rewards from the current frame's rewards map.
    fn dispatch_temporal_frame(
        &mut self,
        tasks: &[LevelTask],
        memory_depth: u32,
        level_context: &TemporalContext,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        age0: &FxHashSet<NeuronId>,
        current_rewards: Option<&FxHashMap<ChannelId, Reward>>,
        frame_number: FrameNumber,
        learning: bool,
    ) -> Vec<ColumnProcessResult> {
        // Decorate age=0 sensory neurons with channel id + pre-resolved reward.
        // Reward resolution is meaningful only for action neurons.
        let mut full_actives = Vec::with_capacity(age0.len());
        for &neuron_id in age0 {
            let channel_id = self.get_neuron_channel_id(neuron_id).unwrap_or(0);
            let reward = match (current_rewards, self.get_neuron_type(neuron_id)) {
                (Some(rewards), Some(NeuronType::Action)) => rewards.get(&channel_id).copied().unwrap_or(0.0),
                _ => 0.0,
            };
            full_actives.push(ActiveNeuron { id: neuron_id, channel_id, reward });
        }

        // Temporal has no neighborhood — every task sequences against ALL active neurons.
        let task_actives: Vec<Vec<ActiveNeuron>> = tasks.iter().map(|_| full_actives.clone()).collect();

        let task_indices_by_region = self.bucket_by_region_indices(tasks, |t| t.neuron_id);
        let level_context_opt = if level_context.size() > 0 { Some(level_context) } else { None };
        let mut results = Vec::new();
        for (r, task_indices) in task_indices_by_region.iter().enumerate() {
            if task_indices.is_empty() { continue; }
            let region_tasks: Vec<_> = task_indices.iter().map(|&i| {
                let task = &tasks[i];
                (task.neuron_id, task.age_states.clone(), task.corrections_as_neuron_corrections(), task.error_feedback.clone(), task_actives[i].clone())
            }).collect();
            let region_results = self.region_list[r].process_temporal_level(
                &region_tasks, memory_depth, level_context_opt, new_error_pattern_ids,
                frame_number, learning,
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
    fn collect_context_ref_updates(result: &ColumnProcessResult, per_target: &mut FxHashMap<NeuronId, Vec<TemporalContextRefUpdate>>) {
        for update in &result.context_ref_updates {
            let mut u = update.clone();
            u.parent_id = result.parent_id;
            per_target.entry(u.neuron_id)
                .or_insert_with(Vec::new)
                .push(u);
        }
    }

    /// Jaccard-union error between an inferred set and an observed set: `(missing + novel) / union`, where
    /// missing = inferred ∖ observed and novel = observed ∖ inferred. This is `1 − match` under the union
    /// denominator — the SAME grouping comparison recognition uses, here over predicted-vs-actual connection
    /// sets. Shared by spatial (`mint_spatial_corrections`) and temporal (`evaluate_vote_error`) correction;
    /// both pass sets already restricted to the comparable kind (events). Returns None when the union is
    /// empty (nothing to compare).
    fn get_union_error(inferred: &FxHashSet<NeuronId>, observed: &FxHashSet<NeuronId>) -> Option<f64> {
        let union_size = inferred.union(observed).count();
        if union_size == 0 { return None; }
        let missing = inferred.difference(observed).count();
        let novel = observed.difference(inferred).count();
        Some((missing + novel) as f64 / union_size as f64)
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
        let error_rate = Self::get_union_error(&predicted, &observed_events)?;

        // the threshold rode in with the vote when it was cast last frame
        let threshold = state.threshold.unwrap_or(0.5);
        let fire = error_rate > threshold;
        Some(VoteErrorResult { fire, error_rate })
    }

    /// Check if a neuron should be skipped (action neuron in a channel whose spec does
    /// not include action-sequence learning).
    fn skip_action_neuron(&self, neuron_id: NeuronId) -> bool {
        if !self.is_base_neuron(neuron_id) { return false; }
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
            if self.neuron_exists(*neuron_id) { dead.push(*neuron_id); }
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
                if let Some(parent_id) = self.get_neuron_parent(*id) {
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
            if let Some(parent_id) = self.get_neuron_parent(id) {
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
        self.neuron_parents.remove(&id);
        // Corrections carry a footprint but never a base bit (bases never die), so only the footprint
        // needs dropping here.
        self.neuron_footprints.remove(&id);
        self.correction_mint_level.remove(&id);
    }

    // ── Snapshot / restore ──────────────────────────────────────────────────

    /// Get a self-contained snapshot of all brain state for external consumers (backup, dump).
    /// Each neuron entry carries serialized neuron data plus resolved metadata — consumers
    /// never need separate lookups or access to live Neuron objects.
    pub fn get_snapshot(&self) -> Snapshot {
        let mut neurons = Vec::new();
        for region in &self.region_list {
            for entry in region.get_snapshot() {
                // Only TRUE base (sensory/action) neurons carry base data (coordinate, channel, type) —
                // they are exactly the ones in the base-neuron registry. Corrections are coordinate-less,
                // so they have no base-neuron entry to snapshot; their locality is the footprint, rebuilt
                // from the constituent graph on restore. Spatial vs temporal corrections are distinguished
                // by the serialized child's `spatial` flag, not a stored level.
                let base_neuron = self.base_neurons.get(&entry.id).cloned();
                // Single representative parent in the snapshot entry; the authoritative multi-parent
                // relation is serialized via patterns.csv (one row per parent) and rebuilt from children.
                let parent_id = self.get_neuron_parent(entry.id);
                neurons.push(SnapshotNeuronEntry {
                    neuron: entry.neuron,
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
            if let Some(parent_id) = entry.parent_id {
                self.neuron_parents.entry(neuron_id).or_default().insert(parent_id);
            }
            // Base (sensory/action) neurons are exactly the entries carrying base data. Corrections have
            // none — they are coordinate-less, with no stored level; their depth is the wave's activation
            // index and their locality is the footprint, rebuilt below.
            if let Some(ref base) = entry.base_neuron {
                self.neurons_by_value.insert(base.coordinate.clone(), neuron_id);
                self.base_neurons.insert(neuron_id, base.clone());
                // Reassign a dense base bit + the singleton footprint. Bit positions need not match
                // the original run — adjacency depends only on set membership, so any self-consistent
                // labelling restores identical neighborhoods.
                self.assign_base_footprint(neuron_id, base.channel_id);
            }
            let r = self.route_neuron(neuron_id);
            let c = self.region_list[r].route_neuron(neuron_id);
            buckets[r][c].push(entry.neuron.clone());
        }

        // Rebuild SPATIAL correction footprints from the constituent graph.
        // Footprints are never serialized: a spatial correction's footprint = ⋃ of its constituents (parent ∪ context).
        // The snapshot already carries that graph in each parent's children/context.
        // Only spatial children contribute: temporal corrections are footprint-less by design, so they are skipped.
        // Base footprints were assigned above; the memoized recursion grounds out there without needing stored levels.
        let mut constituents: FxHashMap<NeuronId, Vec<NeuronId>> = FxHashMap::default();
        for entry in &snapshot.neurons {
            let parent = entry.neuron.id;
            for child in &entry.neuron.children {
                // Skip temporal children — temporal corrections carry no footprint, matching the live mint path.
                if !child.spatial { continue; }
                let cons = constituents.entry(child.pattern_id).or_insert_with(|| vec![parent]);
                for ce in &child.context {
                    cons.push(ce.neuron_id);
                }
            }
        }
        let base_footprints = self.neuron_footprints.clone();
        let mut memo: FxHashMap<NeuronId, Footprint> = FxHashMap::default();
        let mut in_progress: std::collections::HashSet<NeuronId> = std::collections::HashSet::new();
        for &id in constituents.keys() {
            let fp = rebuild_footprint(id, &constituents, &base_footprints, &mut memo, &mut in_progress);
            self.neuron_footprints.insert(id, fp);
        }

        // distribute neurons to their owning columns
        for (r, region_buckets) in buckets.into_iter().enumerate() {
            self.region_list[r].restore_snapshot(region_buckets);
        }

        // rebuild the reverse connection index (derived, never serialized): gather every d=0 edge and
        // register each source on its target's `spatial_connection_sources`, routed by target.
        let conn_edges: Vec<(NeuronId, NeuronId)> = self.region_list.iter()
            .flat_map(|region| region.collect_spatial_connection_edges())
            .collect();
        let mut conn_by_target: FxHashMap<NeuronId, Vec<NeuronId>> = FxHashMap::default();
        for (source, target) in conn_edges {
            conn_by_target.entry(target).or_insert_with(Vec::new).push(source);
        }
        if !conn_by_target.is_empty() {
            let batch: Vec<(NeuronId, Vec<NeuronId>)> = conn_by_target.into_iter().collect();
            self.dispatch_spatial_connection_sources(&batch);
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
        self.neurons_by_value.clear();
        self.base_neurons.clear();
        self.neuron_parents.clear();
        self.neuron_footprints.clear();
        self.correction_mint_level.clear();
        self.base_neuron_bit.clear();
        self.base_bit_channel.clear();
        self.base_bit_neuron.clear();
        self.channel_base_bits.clear();
        self.next_base_bit = 0;
        self.death_ledger.clear();
        self.neuron_death_frame.clear();
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

    /// Total number of neurons: base (sensory/action) neurons plus live corrections (which carry a
    /// parent). With levels no longer stored, these two registries together are the neuron census.
    pub fn get_neuron_count(&self) -> usize {
        self.base_neurons.len() + self.neuron_parents.len()
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
        Thalamus::new(false, 0.1, 4, 0.5, GroupMode::Static, 1, 1)
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
        assert!(t.is_base_neuron(1));
    }

    #[test]
    fn test_routing() {
        let t = Thalamus::new(false, 0.1, 4, 0.5, GroupMode::Static, 3, 1);
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

        // reap — neuron 42 does not exist (no base entry, no parent) so won't be returned
        let dead = t.reap_dead_neurons(200);
        assert!(dead.is_empty());

        // register as a correction (give it a parent) to simulate an actual live neuron
        t.neuron_parents.entry(42).or_default().insert(7);
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
    fn test_neuron_count() {
        let mut t = make_thalamus();
        let coord1 = Coordinate { dim_id: 1, bucket_id: 1 };
        let coord2 = Coordinate { dim_id: 1, bucket_id: 2 };
        t.get_neuron_id_for_point(&coord1, 1, NeuronType::Event);
        t.get_neuron_id_for_point(&coord2, 1, NeuronType::Event);
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

    // ── Footprint tests (wave-front Stage 1) ────────────────────────────────

    /// Register a one-input-dimension channel and return (channel_id, dim_id).
    fn register_input_channel(t: &mut Thalamus, name: &str) -> (ChannelId, DimensionId) {
        let reg = t.register_channel_spec(
            name,
            vec![DimSpecInput {
                name: "v".to_string(),
                kind: DimKind::Input,
                resolution: 2,
                mode: None,
                boundaries: None,
                actions: None,
                default_action: None,
                warmup_samples: None,
            }],
            false,
        );
        (reg.channel_id, *reg.dimension_ids.get("v").unwrap())
    }

    /// Build a thalamus with base neurons over a known spatial neighbor graph:
    /// a ↔ b are neighbors, c is isolated (empty list), x has no list (all-pairs default).
    /// Returns the thalamus and the base neuron ids (a, b, c, x).
    fn make_footprint_graph() -> (Thalamus, NeuronId, NeuronId, NeuronId, NeuronId) {
        let mut t = make_thalamus();
        let (ca, da) = register_input_channel(&mut t, "a");
        let (cb, db) = register_input_channel(&mut t, "b");
        let (cc, dc) = register_input_channel(&mut t, "c");
        let (cx, dx) = register_input_channel(&mut t, "x");
        t.set_spatial_neighbors("a", &["b".to_string()]);
        t.set_spatial_neighbors("b", &["a".to_string()]);
        t.set_spatial_neighbors("c", &[]); // isolated — declared but empty
        // "x": no set_spatial_neighbors call → all-pairs default
        let a = t.get_neuron_id_for_point(&Coordinate { dim_id: da, bucket_id: 1 }, ca, NeuronType::Event).id;
        let b = t.get_neuron_id_for_point(&Coordinate { dim_id: db, bucket_id: 1 }, cb, NeuronType::Event).id;
        let c = t.get_neuron_id_for_point(&Coordinate { dim_id: dc, bucket_id: 1 }, cc, NeuronType::Event).id;
        let x = t.get_neuron_id_for_point(&Coordinate { dim_id: dx, bucket_id: 1 }, cx, NeuronType::Event).id;
        (t, a, b, c, x)
    }

    #[test]
    fn test_footprint_base_is_singleton() {
        let (t, a, b, c, x) = make_footprint_graph();
        for id in [a, b, c, x] {
            let fp = t.get_neuron_footprint(id);
            assert_eq!(fp.count_ones(), 1, "base neuron {} footprint must be a singleton", id);
        }
    }

    #[test]
    fn test_footprint_adjacency_matches_base_graph() {
        let (t, a, b, c, x) = make_footprint_graph();
        let (fa, fb, fc, fx) = (
            t.get_neuron_footprint(a),
            t.get_neuron_footprint(b),
            t.get_neuron_footprint(c),
            t.get_neuron_footprint(x),
        );

        // a ↔ b are neighbors (directional declaration is symmetric here).
        assert!(t.footprints_adjacent(&fa, &fb));
        assert!(t.footprints_adjacent(&fb, &fa));

        // c is isolated; a's list excludes c.
        assert!(!t.footprints_adjacent(&fa, &fc));
        assert!(!t.footprints_adjacent(&fc, &fa));

        // x has no list → all-pairs as a parent, but is not in a's list as a target.
        assert!(t.footprints_adjacent(&fx, &fa));
        assert!(!t.footprints_adjacent(&fa, &fx));
    }

    /// The channel-neighbor window that footprint adjacency replaced: target is in parent's spatial
    /// neighbor set, or the parent declared no list (all-pairs). Kept here as the test oracle for the
    /// L0-equivalence gate now that the production predicate is gone.
    fn channel_window_oracle(t: &Thalamus, parent_channel: ChannelId, target_channel: ChannelId) -> bool {
        match t.base_neighbors.get(&parent_channel) {
            None => true,
            Some(set) => set.contains(&target_channel),
        }
    }

    #[test]
    fn test_footprint_l0_equals_channel_window() {
        // The Stage-1 gate: at L0, footprint adjacency equals the channel-neighbor window for every
        // pair of DISTINCT base neurons (call sites always exclude the parent itself).
        let (t, a, b, c, x) = make_footprint_graph();
        let ids = [a, b, c, x];
        for &p in &ids {
            for &q in &ids {
                if p == q { continue; }
                let fp = t.get_neuron_footprint(p);
                let fq = t.get_neuron_footprint(q);
                let cp = t.get_neuron_channel_id(p).unwrap();
                let cq = t.get_neuron_channel_id(q).unwrap();
                assert_eq!(
                    t.footprints_adjacent(&fp, &fq),
                    channel_window_oracle(&t, cp, cq),
                    "footprint adjacency must equal channel window for ({}, {})", p, q,
                );
            }
        }
    }

    #[test]
    fn test_correction_is_coordinate_less() {
        // Wave-front Stage 3/4: a minted correction carries an id + footprint, never a coordinate or
        // a stored level.
        let (mut t, a, b, _c, _x) = make_footprint_graph();
        let spec = t.allocate_spatial_pattern_neuron(a);
        // No base-neuron entry → no coordinate, channel, or type; not a base neuron.
        assert!(t.get_neuron_coordinate(spec.id).is_none());
        assert!(t.get_neuron_channel_id(spec.id).is_none());
        assert!(t.get_neuron_type(spec.id).is_none());
        assert!(!t.is_base_neuron(spec.id));
        // Still a registered pattern: parented and counted as existing.
        assert_eq!(t.get_neuron_parent(spec.id), Some(a));
        assert!(t.neuron_exists(spec.id));
        // Footprint is set by the mint path (compute_correction_footprint), not allocate.
        let fp = t.compute_correction_footprint(a, &[b]);
        assert_eq!(fp.count_ones(), 2);
    }

    #[test]
    fn test_footprint_correction_is_union_of_constituents() {
        let (t, a, b, c, _x) = make_footprint_graph();
        // A correction with parent a and bound context {b} covers {a, b}.
        let fp_corr = t.compute_correction_footprint(a, &[b]);
        assert_eq!(fp_corr.count_ones(), 2);
        assert!(fp_corr.intersects(&t.get_neuron_footprint(a)));
        assert!(fp_corr.intersects(&t.get_neuron_footprint(b)));

        // Touches b (shared base) but not the isolated c.
        assert!(t.footprints_adjacent(&fp_corr, &t.get_neuron_footprint(b)));
        assert!(!t.footprints_adjacent(&fp_corr, &t.get_neuron_footprint(c)));
    }

    #[test]
    fn test_cluster_merges_adjacent_separates_isolated() {
        let (t, a, b, c, _x) = make_footprint_graph();
        // a↔b are neighbors → one component; c is isolated → its own.
        let fps = vec![t.get_neuron_footprint(a), t.get_neuron_footprint(b), t.get_neuron_footprint(c)];
        let clusters = t.cluster_by_footprint_adjacency(&fps);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.contains(&vec![0, 1]));
        assert!(clusters.contains(&vec![2]));
    }

    #[test]
    fn test_cluster_overlapping_footprints_merge() {
        let (t, a, _b, _c, _x) = make_footprint_graph();
        // Two copies of the same footprint overlap (intersects) → one component.
        let fps = vec![t.get_neuron_footprint(a), t.get_neuron_footprint(a)];
        let clusters = t.cluster_by_footprint_adjacency(&fps);
        assert_eq!(clusters, vec![vec![0, 1]]);
    }

    #[test]
    fn test_cluster_transitive_chain() {
        // a↔b adjacent and the correction {a,b} touches b, so [a, corr(a,b), b] is one transitive chain
        // even though a and b's roles differ — connectivity is via the shared/adjacent bases.
        let (t, a, b, c, _x) = make_footprint_graph();
        let fps = vec![
            t.get_neuron_footprint(a),
            t.compute_correction_footprint(a, &[b]),
            t.get_neuron_footprint(b),
            t.get_neuron_footprint(c), // isolated
        ];
        let clusters = t.cluster_by_footprint_adjacency(&fps);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.contains(&vec![0, 1, 2]));
        assert!(clusters.contains(&vec![3]));
    }

    #[test]
    fn test_cluster_empty_input() {
        let t = make_thalamus();
        let clusters = t.cluster_by_footprint_adjacency(&[]);
        assert!(clusters.is_empty());
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
