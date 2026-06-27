/// Neuron - Unified struct for all neurons (sensory and pattern)
///
/// All neurons have:
/// - connections: Vec<FxHashMap<NeuronId, ConnectionData>> - predictions (indexed by distance)
/// - routing_table: FxHashMap<PatternId, TemporalRoutingEntry> - child pattern contexts
///
/// All neuron metadata (level, coordinates, channel, type, parent) is stored externally
/// in Thalamus lookup tables. Neurons are pure data processors — they only store learned
/// associations and do pattern matching/voting based on numeric IDs and strengths.
///
/// Note: Active state (which neurons are active at which ages) and votes are managed
/// by the Brain, not stored on neurons. This allows efficient age-indexed queries.
///
/// Lazy Decay: Continuous decay based on frames elapsed since last activation.
/// We store last_activation_frame and compute effective strength on-demand:
/// effective_strength = strength - (frame_number - last_frame) * rate

use rustc_hash::{FxHashMap, FxHashSet};

use crate::context::{SpatialContext, TemporalContext};
use crate::types::*;

/// Minimum samples per (neuron, age) before dynamic modes switch off the warmup
/// fallback. Not a tunable knob — kept here to avoid a magic number in the error-threshold getters.
const ERROR_MIN_SAMPLES: u64 = 3;

/// Entry in the spatial routing table for a child spatial-correction pattern.
/// Spatial context has no distance dimension.
#[derive(Debug, Clone)]
pub struct SpatialRoutingEntry {
    pub context: SpatialContext,
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
}

/// Entry in the temporal routing table for a child temporal-correction pattern.
#[derive(Debug, Clone)]
pub struct TemporalRoutingEntry {
    pub context: TemporalContext,
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
}

/// A single vote cast by a neuron at a specific age.
#[derive(Debug, Clone)]
pub struct Vote {
    pub neuron_id: NeuronId,
    pub strength: Strength,
    pub reward: Reward,
    pub distance: Distance,
}

/// A recognition match result for a specific age.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub activate: bool,
    pub death_frame: Option<FrameNumber>,
}

/// A TEMPORAL context reference update.
/// `neuron_id` is the context neuron being referenced.
/// `parent_id` is the parent neuron that owns the temporal routing table referencing it.
/// `parent_id` is set to 0 when emitted from neuron-level code and filled in by thalamus during
/// `collect_context_ref_updates`.
#[derive(Debug, Clone)]
pub struct TemporalContextRefUpdate {
    pub neuron_id: NeuronId,
    pub distance: Distance,
    pub parent_id: NeuronId,
}

/// A SPATIAL context reference update — no distance dimension.
/// `neuron_id` is the context neuron being referenced (the target whose spatial_context_refs
/// gains an entry pointing at `parent_id`). Symmetric in shape to TemporalContextRefUpdate so
/// both dispatch chains bucket-by-target at the same boundary.
#[derive(Debug, Clone)]
pub struct SpatialContextRefUpdate {
    pub neuron_id: NeuronId,
    pub parent_id: NeuronId,
}

/// A context reference entry (neuron_id + distance).
#[derive(Debug, Clone)]
pub struct ContextRefEntry {
    pub neuron_id: NeuronId,
    pub distance: Distance,
}

/// An error-correction activation result.
#[derive(Debug, Clone)]
pub struct CorrectionActivation {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub death_frame: Option<FrameNumber>,
}

/// Per-age vote output from generate_votes.
#[derive(Debug, Clone)]
pub struct AgeVotes {
    pub age: Distance,
    pub votes: Vec<Vote>,
    pub context: Vec<ContextRefEntry>,
    pub threshold: f64,
}

/// Age state passed into process_frame (per-age activation info).
#[derive(Debug, Clone, Default)]
pub struct AgeState {
    pub activated_pattern_id: Option<NeuronId>,
}

/// Active sensory neuron info passed into learn_connections.
#[derive(Debug, Clone)]
pub struct ActiveNeuron {
    pub id: NeuronId,
    pub channel_id: ChannelId,
    pub reward: Reward,
}

/// Error feedback from prior-frame votes.
#[derive(Debug, Clone)]
pub struct ErrorFeedback {
    pub age: Distance,
    pub error_rate: f64,
}

/// Error-correction pattern to install as a child.
#[derive(Debug, Clone)]
pub struct Correction {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub context_entries: Vec<ContextRefEntry>,
}

/// Results from process_frame.
#[derive(Debug, Clone)]
pub struct ProcessFrameResult {
    pub matches: Vec<PatternMatch>,
    pub correction_activations: Vec<CorrectionActivation>,
    pub context_ref_updates: Vec<TemporalContextRefUpdate>,
    pub votes: Vec<AgeVotes>,
    pub timings: NeuronOpTimings,
}

/**
 * Per-neuron wall-clock for the 4 main ops inside process_frame, plus
 * recognize sub-buckets. Aggregated up through column → brain into the
 * frame timings. `recognize_candidates_evaluated` counts how many child
 * patterns were scored (post-index) so we can see search-space size per
 * frame.
 */
#[derive(Debug, Clone, Copy, Default)]
pub struct NeuronOpTimings {
    pub learn_connections: f64,
    pub recognize_patterns: f64,
    pub correct_errors: f64,
    pub generate_votes: f64,
    pub recognize_candidate_search: f64,
    pub recognize_candidate_eval: f64,
    pub recognize_candidates_evaluated: u64,
}

impl NeuronOpTimings {
    pub fn add(&mut self, other: &NeuronOpTimings) {
        self.learn_connections  += other.learn_connections;
        self.recognize_patterns += other.recognize_patterns;
        self.correct_errors     += other.correct_errors;
        self.generate_votes     += other.generate_votes;
        self.recognize_candidate_search   += other.recognize_candidate_search;
        self.recognize_candidate_eval     += other.recognize_candidate_eval;
        self.recognize_candidates_evaluated += other.recognize_candidates_evaluated;
    }
}

/// Results from recognize_patterns.
struct RecognizeResult {
    matches: Vec<PatternMatch>,
    context_ref_updates: Vec<TemporalContextRefUpdate>,
}

/// Results from correct_errors.
struct CorrectResult {
    correction_activations: Vec<CorrectionActivation>,
    context_ref_updates: Vec<TemporalContextRefUpdate>,
}

pub struct Neuron {
    /// Neuron ID — public because Column reads it after removing the neuron from its map.
    pub id: NeuronId,

    pattern_forget_rate: f64,

    /// The single brain-wide grouping coefficient θ, shared by spatial (d=0) and temporal (d>0). It is the
    /// SEED for the adaptive grouping threshold: when a unit has no error history, the live correction
    /// threshold is the derived complement `1 − θ` and recognition fires at similarity ≥ θ. Once error
    /// stats accrue, both sides float together off those stats — see [grouping_error_threshold], the sole
    /// reader of this field. One number, read from opposite sides; no separate merge knob, no phase split.
    group_threshold: f64,

    /// How the live grouping threshold adapts from this unit's running Welford error stats:
    /// Static keeps the seed `1 − θ`; the dynamic modes shift the correction side to mean ± σ (and
    /// recognition to `1 − that`). Shared across spatial and temporal — the error buckets differ, the
    /// adaptation policy does not.
    group_mode: GroupMode,

    /// Brain-wide context_length, replicated on each neuron so recognize_patterns can implement the warmup gate:
    /// skip matching until the context window has had a chance to fill up at the start of a sequence
    context_length: u32,

    /// Per-channel action neuron IDs — ordered Vec for deterministic alternative-action
    /// exploration (neurons are tried in registration order, not hash-iteration order).
    channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,

    // ── Spatial state (d=0 co-activation, no distance dimension) ────────────────

    /// Spatial outgoing connections: target_neuron_id → connection data.
    /// One flat map — there's no distance dimension because spatial is same-frame.
    spatial_connections: FxHashMap<NeuronId, ConnectionData>,

    /// Spatial routing table: child_pattern_id → SpatialRoutingEntry.
    /// Spatial correction patterns hosted by this neuron.
    spatial_routing_table: FxHashMap<NeuronId, SpatialRoutingEntry>,

    /// Spatial inverted index: ctx_neuron_id → set of pattern_ids that reference it.
    /// No distance keying — spatial context entries have no distance.
    spatial_context_index: FxHashMap<NeuronId, FxHashSet<NeuronId>>,

    /// Spatial context references: set of parent neurons whose spatial routing tables
    /// reference this neuron. No distance.
    spatial_context_refs: FxHashSet<NeuronId>,

    /// Spatial Welford error stats — single bucket (spatial has no age dimension).
    spatial_error_stats: Option<WelfordState>,

    // ── Temporal state (d>0 sequence, distance-keyed) ───────────────────────────

    /// Temporal outgoing connections by distance: temporal_connections[d] = target → connection.
    /// Index 0 is unused (spatial lives in `spatial_connections`); temporal uses index 1..context_length.
    temporal_connections: Vec<FxHashMap<NeuronId, ConnectionData>>,

    /// Temporal routing table: child_pattern_id → TemporalRoutingEntry.
    temporal_routing_table: FxHashMap<NeuronId, TemporalRoutingEntry>,

    /// Temporal inverted index: ctx_neuron_id → distance → set of pattern_ids.
    temporal_context_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>,

    /// Temporal context references: parent_id → set of distances that reference this neuron.
    temporal_context_refs: FxHashMap<NeuronId, FxHashSet<Distance>>,

    /// Temporal Welford error stats indexed by age. Used for dynamic error thresholds.
    temporal_error_stats: Vec<Option<WelfordState>>,
}

impl Neuron {
    /// Neuron id is allocated by the Thalamus (mirrors how channel and dimension
    /// ids are allocated) and passed in at construction. channel_action_ids are used
    /// for alternative-action lookup during learning (per-channel Vec iteration).
    /// Populated by register_channel_spec() and shared across all neurons.
    ///
    /// The group mode picks how the live grouping threshold adapts from past error rates (the temporal
    /// per-age bucket, or the single spatial bucket); recognition then uses `1 − that` for the same bucket:
    ///   Static       — fixed correction threshold = `1 − group_threshold`
    ///   Conservative — mean + σ of past error rates  (learn outliers)
    ///   Neutral      — mean
    ///   Aggressive   — mean − σ (learn aggressively)
    /// For dynamic modes, `1 − group_threshold` is the warmup fallback until ERROR_MIN_SAMPLES
    /// observations have been recorded in that bucket.
    pub fn new(
        id: NeuronId,
        pattern_forget_rate: f64,
        group_threshold: f64,
        group_mode: GroupMode,
        channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,
        context_length: u32,
    ) -> Self {
        Self {
            id,
            pattern_forget_rate,
            group_threshold,
            group_mode,
            context_length,
            channel_action_ids,
            spatial_connections: FxHashMap::default(),
            spatial_routing_table: FxHashMap::default(),
            spatial_context_index: FxHashMap::default(),
            spatial_context_refs: FxHashSet::default(),
            spatial_error_stats: None,
            temporal_connections: Vec::new(),
            temporal_routing_table: FxHashMap::default(),
            temporal_context_index: FxHashMap::default(),
            temporal_context_refs: FxHashMap::default(),
            temporal_error_stats: Vec::new(),
        }
    }

    // ── Grouping threshold ─────────────────────────────────────────────────────

    /// THE grouping operation's threshold, and the SOLE reader of `group_threshold` + `group_mode`.
    ///
    /// Given the relevant Welford error bucket (spatial: the single bucket; temporal: the per-age bucket),
    /// returns the correction (error) threshold E ∈ [0,1]: the static / warmup value is the derived
    /// complement `1 − group_threshold`; the dynamic modes shift it by the bucket's running error stats
    /// (mean ± σ) once ERROR_MIN_SAMPLES observations exist. Recognition treats two connection sets as
    /// "the same" when their match ratio ≥ `1 − E` ([grouping_merge_threshold]); correction mints when the
    /// observed error ratio > E. One number, read from opposite sides — there is no separate merge knob.
    fn grouping_error_threshold(&self, stats: Option<&WelfordState>) -> f64 {
        let fallback = 1.0 - self.group_threshold;
        if self.group_mode == GroupMode::Static { return fallback; }
        match stats {
            Some(s) if s.n >= ERROR_MIN_SAMPLES => {
                let sigma = s.std_dev();
                match self.group_mode {
                    GroupMode::Conservative => s.mean + sigma,
                    GroupMode::Neutral => s.mean,
                    GroupMode::Aggressive => s.mean - sigma,
                    GroupMode::Static => unreachable!(),
                }
            }
            _ => fallback,
        }
    }

    /// Recognition strictness: `1 − E` for the same error bucket. A reliable unit (low E) recognizes
    /// strictly; an unreliable one (high E) recognizes loosely. Derived, never tuned independently.
    fn grouping_merge_threshold(&self, stats: Option<&WelfordState>) -> f64 {
        1.0 - self.grouping_error_threshold(stats)
    }

    /// Temporal correction threshold for a given age — selects the per-age bucket and defers to
    /// [grouping_error_threshold]. Thin bucket-selector; the policy lives in one place.
    pub fn get_temporal_error_threshold(&self, age: Distance) -> f64 {
        self.grouping_error_threshold(self.temporal_error_stats.get(age as usize).and_then(|s| s.as_ref()))
    }

    /// Spatial correction threshold — selects the single spatial bucket and defers to
    /// [grouping_error_threshold].
    pub fn get_spatial_error_threshold(&self) -> f64 {
        self.grouping_error_threshold(self.spatial_error_stats.as_ref())
    }

    // ── Error stats ──────────────────────────────────────────────────────────

    /// Record an observed temporal error rate for a given age (Welford online update).
    /// Called after the threshold comparison so the current sample does not
    /// influence its own decision.
    pub fn record_temporal_error(&mut self, age: Distance, error_rate: f64) {
        let idx = age as usize;
        if idx >= self.temporal_error_stats.len() { self.temporal_error_stats.resize(idx + 1, None); }
        let stats = self.temporal_error_stats[idx].get_or_insert_with(WelfordState::new);
        stats.update(error_rate);
    }

    /// Record an observed spatial error rate (no age — single bucket).
    pub fn record_spatial_error(&mut self, error_rate: f64) {
        let stats = self.spatial_error_stats.get_or_insert_with(WelfordState::new);
        stats.update(error_rate);
    }

    /// Restoration entry point: install a fully-formed Welford bucket for a given
    /// temporal age. Used by Column.restore_neurons to rehydrate per-(neuron, age) error stats
    /// from serialized neuron. Does not validate the stats — the caller owns correctness.
    pub fn load_temporal_error_stats(&mut self, age: Distance, n: u64, mean: f64, m2: f64) {
        let idx = age as usize;
        if idx >= self.temporal_error_stats.len() { self.temporal_error_stats.resize(idx + 1, None); }
        self.temporal_error_stats[idx] = Some(WelfordState { n, mean, m2 });
    }

    /// Restoration entry point: install the spatial Welford bucket.
    pub fn load_spatial_error_stats(&mut self, n: u64, mean: f64, m2: f64) {
        self.spatial_error_stats = Some(WelfordState { n, mean, m2 });
    }

    // ── Serialization ────────────────────────────────────────────────────────

    /// Serialize persistent state into a plain object for snapshotting and backup.
    /// Returns everything needed to reconstruct this Neuron via Column.load_neuron.
    /// context_index is omitted — it is rebuilt from children's context entries on load.
    pub fn serialize(&self) -> SerializedNeuron {
        SerializedNeuron {
            id: self.id,
            pattern_forget_rate: self.pattern_forget_rate,
            connections: self.serialize_connections(),
            children: self.serialize_children(),
            context_refs: self.serialize_context_refs(),
            error_stats: self.serialize_error_stats(),
        }
    }

    /// Serialize directed connections — flattens spatial (d=0) and temporal (d>0) into a single
    /// distance-keyed array for snapshot back-compat.
    fn serialize_connections(&self) -> Vec<SerializedConnection> {
        let mut result = Vec::new();
        // spatial at distance 0
        for (&to_neuron_id, conn) in &self.spatial_connections {
            result.push(SerializedConnection {
                distance: 0,
                to_neuron_id,
                strength: conn.strength,
                reward: conn.reward,
            });
        }
        // temporal at distance > 0
        for (distance, targets) in self.temporal_connections.iter().enumerate() {
            if distance == 0 { continue; } // temporal slot 0 is unused
            for (&to_neuron_id, conn) in targets {
                result.push(SerializedConnection {
                    distance: distance as Distance,
                    to_neuron_id,
                    strength: conn.strength,
                    reward: conn.reward,
                });
            }
        }
        result
    }

    /// Serialize the routing tables (child patterns). Each child carries its activation
    /// strength, last activation frame, and flattened context entries. Spatial children's
    /// entries surface with distance=0 (a placeholder for the snapshot format).
    fn serialize_children(&self) -> Vec<SerializedChild> {
        let mut result = Vec::new();
        for (&pattern_id, entry) in &self.spatial_routing_table {
            result.push(SerializedChild {
                pattern_id,
                spatial: true,
                activation_strength: entry.activation_strength,
                last_activation_frame: entry.last_activation_frame,
                context: entry.context.get_entries(),
            });
        }
        for (&pattern_id, entry) in &self.temporal_routing_table {
            result.push(SerializedChild {
                pattern_id,
                spatial: false,
                activation_strength: entry.activation_strength,
                last_activation_frame: entry.last_activation_frame,
                context: entry.context.get_entries(),
            });
        }
        result
    }

    /// Serialize context references — both spatial and temporal flattened into a single
    /// {parent_id, distances} array. Spatial entries use a single-element [0] distances list
    /// as a placeholder for the snapshot format.
    fn serialize_context_refs(&self) -> Vec<SerializedContextRef> {
        let mut result = Vec::new();
        for &parent_id in &self.spatial_context_refs {
            result.push(SerializedContextRef {
                parent_id,
                spatial: true,
                distances: vec![0],
            });
        }
        for (&parent_id, distances) in &self.temporal_context_refs {
            result.push(SerializedContextRef {
                parent_id,
                spatial: false,
                distances: distances.iter().copied().collect(),
            });
        }
        result
    }

    /// Serialize per-age Welford error stats. The spatial bucket surfaces as age=0; temporal
    /// entries surface as their age index.
    /// Age 0 is reused for the spatial bucket because temporal age 0 never carries stats: error
    /// feedback only comes from evaluate_vote_error, which returns None for age 0 (an age-0 neuron is
    /// just voting now, nothing to correct). So temporal_error_stats[0] is always None and skipping it
    /// drops nothing — the debug_assert guards that invariant against future changes to the feedback path.
    fn serialize_error_stats(&self) -> Vec<SerializedErrorStats> {
        debug_assert!(
            self.temporal_error_stats.first().map_or(true, |s| s.is_none()),
            "temporal_error_stats[0] is unexpectedly populated — serialize aliases age 0 onto the spatial \
             bucket on the assumption it is always empty (see evaluate_vote_error's age-0 guard)"
        );
        let mut result = Vec::new();
        if let Some(s) = &self.spatial_error_stats {
            result.push(SerializedErrorStats { age: 0, n: s.n, mean: s.mean, m2: s.m2 });
        }
        for (age, stats) in self.temporal_error_stats.iter().enumerate() {
            if age == 0 { continue; }
            if let Some(s) = stats {
                result.push(SerializedErrorStats {
                    age: age as Distance,
                    n: s.n,
                    mean: s.mean,
                    m2: s.m2,
                });
            }
        }
        result
    }

    // ── Connections ──────────────────────────────────────────────────────────

    /// Returns if there is a connection at distance to a target neuron.
    /// Distance 0 routes to spatial_connections; distance > 0 routes to temporal_connections[d].
    pub fn has_connection(&self, distance: Distance, to_neuron_id: NeuronId) -> bool {
        if distance == 0 {
            return self.spatial_connections.contains_key(&to_neuron_id);
        }
        match self.temporal_connections.get(distance as usize) {
            Some(distance_map) => distance_map.contains_key(&to_neuron_id),
            None => false,
        }
    }

    /// Creates a connection at distance to target neuron.
    /// Distance 0 routes to spatial_connections; distance > 0 routes to temporal_connections[d].
    pub fn create_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, strength: Strength, reward: Reward) {
        if distance == 0 {
            self.spatial_connections.insert(to_neuron_id, ConnectionData { strength, reward });
            return;
        }
        let idx = distance as usize;
        if idx >= self.temporal_connections.len() { self.temporal_connections.resize_with(idx + 1, FxHashMap::default); }
        self.temporal_connections[idx].insert(to_neuron_id, ConnectionData { strength, reward });
    }

    /// Strengthen the connection if it already exists, otherwise create it with strength=1.0.
    /// Reward is folded in via strengthen_connection's exponential smoothing on update.
    /// On create, reward is stored as-is.
    /// This is the core (create-or-strengthen + smoothed reward) shared by temporal `upsert_connection` and supervised `Brain.learn`.
    /// Supervised callers use this directly because they don't need the alt-action lookup `upsert_connection` adds on top.
    pub fn strengthen_or_create_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, reward: Reward) {
        if self.has_connection(distance, to_neuron_id) {
            self.strengthen_connection(distance, to_neuron_id, reward);
        } else {
            self.create_connection(distance, to_neuron_id, 1.0, reward);
        }
    }

    /// Upsert a connection at distance to the target neuron: create if missing, else strengthen (smoothing the reward).
    /// If the resulting reward is negative, wire an alternative action with neutral reward so it can be tried next time.
    /// Used by temporal learning in process_frame, where reward exploration matters.
    pub fn upsert_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, channel_id: ChannelId, reward: Reward) {

        // strengthen or create the connection first
        self.strengthen_or_create_connection(distance, to_neuron_id, reward);

        // For actions with negative (smoothed) rewards, save an alternative with neutral reward — we'll try it next time.
        let conn_reward = self.get_connection_reward(distance, to_neuron_id).unwrap();
        if conn_reward < 0.0 {
            if let Some(alt) = self.find_alternative_action(distance, channel_id, to_neuron_id) {
                self.create_connection(distance, alt, 1.0, 0.0);
            }
        }
    }

    /// Read the stored reward on a connection at the given distance, if it exists.
    fn get_connection_reward(&self, distance: Distance, to_neuron_id: NeuronId) -> Option<f64> {
        if distance == 0 {
            return self.spatial_connections.get(&to_neuron_id).map(|c| c.reward);
        }
        self.temporal_connections.get(distance as usize)
            .and_then(|dm| dm.get(&to_neuron_id))
            .map(|c| c.reward)
    }

    /// Updates the connection at distance to target neuron - increments strength and updates reward.
    pub fn strengthen_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, reward: Reward) {
        let conn = if distance == 0 {
            self.spatial_connections.get_mut(&to_neuron_id).expect("Unknown connection")
        } else {
            let distance_map = self.temporal_connections.get_mut(distance as usize).expect("Unknown connection");
            distance_map.get_mut(&to_neuron_id).expect("Unknown connection")
        };

        // increment the strength of the connection
        conn.strength += 1.0;

        // update reward with dynamic exponential smoothing - calculates exact expected value based on means
        let alpha = 1.0 / conn.strength;
        conn.reward = alpha * reward + (1.0 - alpha) * conn.reward;
    }

    // ── Voting ───────────────────────────────────────────────────────────────

    /// Returns votes from this neuron at a specific age (temporal voting).
    /// Reads connections[age + 1] — predicting one frame past the neuron's activation age.
    pub fn vote(&self, age: Distance) -> Vec<Vote> {
        self.vote_at_distance(age + 1)
    }

    /// Returns votes from this neuron at a fixed distance.
    /// Distance 0 reads spatial_connections (co-activation predictions for the current frame).
    /// Distance > 0 reads temporal_connections[d] (next-frame and further predictions).
    pub fn vote_at_distance(&self, distance: Distance) -> Vec<Vote> {
        if distance == 0 {
            let mut result = Vec::new();
            for (&neuron_id, conn) in &self.spatial_connections {
                if conn.strength > 0.0 {
                    result.push(Vote { neuron_id, strength: conn.strength, reward: conn.reward, distance });
                }
            }
            return result;
        }

        // temporal connections
        let distance_map = match self.temporal_connections.get(distance as usize) {
            Some(dm) => dm,
            None => return Vec::new(),
        };
        let mut result = Vec::new();
        for (&neuron_id, conn) in distance_map {
            if conn.strength > 0.0 {
                result.push(Vote { neuron_id, strength: conn.strength, reward: conn.reward, distance });
            }
        }
        result
    }

    // ── Child pattern management ─────────────────────────────────────────────

    /// Get effective activation strength for a child pattern with lazy decay.
    /// Looks up in whichever routing table holds the pattern (spatial or temporal).
    pub fn get_child_effective_activation_strength(&self, pattern_id: NeuronId, current_frame: FrameNumber) -> f64 {
        if let Some(entry) = self.spatial_routing_table.get(&pattern_id) {
            return f64::max(0.0, entry.activation_strength - (current_frame - entry.last_activation_frame) as f64 * self.pattern_forget_rate);
        }
        if let Some(entry) = self.temporal_routing_table.get(&pattern_id) {
            return f64::max(0.0, entry.activation_strength - (current_frame - entry.last_activation_frame) as f64 * self.pattern_forget_rate);
        }
        0.0
    }

    /// Materialize lazy decay for a child pattern.
    pub fn materialize_child_strength(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) {
        let effective = self.get_child_effective_activation_strength(pattern_id, current_frame);
        if let Some(entry) = self.spatial_routing_table.get_mut(&pattern_id) {
            entry.activation_strength = effective;
            return;
        }
        if let Some(entry) = self.temporal_routing_table.get_mut(&pattern_id) {
            entry.activation_strength = effective;
        }
    }

    /// Fold pending lazy decay into every child's stored strength and rebase its activation clock to 0,
    /// across both the temporal and spatial routing tables.
    /// Returns each child's recomputed death frame so the caller can rebuild the death ledger on restore.
    /// Uses this neuron's own pattern_forget_rate — the same rate that drives its lazy-decay math.
    pub fn materialize_and_reset_children(&mut self, current_frame: FrameNumber) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();

        // Collect ids up front: materialize_child_strength borrows self mutably, so we can't hold
        // an iterator over the routing table while calling it.
        let temporal_ids: Vec<NeuronId> = self.temporal_routing_table.keys().copied().collect();
        for pattern_id in temporal_ids {

            // Fold any pending lazy decay into the stored strength so the snapshot is exact.
            self.materialize_child_strength(pattern_id, current_frame);
            if let Some(entry) = self.temporal_routing_table.get_mut(&pattern_id) {

                // Rebase the activation clock to 0 — the restored brain starts counting frames fresh.
                entry.last_activation_frame = 0;

                // Recompute the death frame from the now-materialized strength rather than persisting it.
                let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
                entries.push(DeathFrameEntry { pattern_id, death_frame });
            }
        }

        // Spatial children carry death frames too, so materialize and reset them identically.
        let spatial_ids: Vec<NeuronId> = self.spatial_routing_table.keys().copied().collect();
        for pattern_id in spatial_ids {

            // Fold any pending lazy decay into the stored strength so the snapshot is exact.
            self.materialize_child_strength(pattern_id, current_frame);
            if let Some(entry) = self.spatial_routing_table.get_mut(&pattern_id) {

                // Rebase the activation clock to 0 — the restored brain starts counting frames fresh.
                entry.last_activation_frame = 0;

                // Recompute the death frame from the now-materialized strength rather than persisting it.
                let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
                entries.push(DeathFrameEntry { pattern_id, death_frame });
            }
        }

        entries
    }

    /// Compute each child's death frame from its current activation strength, without mutating any state.
    /// Walks both the temporal and spatial routing tables, mirroring materialize_and_reset_children.
    /// Used on restore to rebuild the death ledger when strengths are already materialized.
    /// Uses this neuron's own pattern_forget_rate — the same rate that drives its lazy-decay math.
    pub fn compute_death_frames(&self) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();

        // Temporal children: a pattern survives until its activation strength has decayed away.
        // death_frame is how many frames that takes at the forget rate, rounded up.
        for (&pattern_id, entry) in &self.temporal_routing_table {
            let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
            entries.push(DeathFrameEntry { pattern_id, death_frame });
        }

        // Spatial children use the same strength/forget-rate decay model, so compute them identically.
        for (&pattern_id, entry) in &self.spatial_routing_table {
            let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
            entries.push(DeathFrameEntry { pattern_id, death_frame });
        }

        entries
    }

    /// Increments activation strength for a child pattern - materializes all owner-scoped lazy decay first.
    /// Returns death frame for pattern neurons.
    pub fn strengthen_child_activation(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) -> Option<FrameNumber> {
        if !self.spatial_routing_table.contains_key(&pattern_id) && !self.temporal_routing_table.contains_key(&pattern_id) { return None; }

        // update all strengths based on decay rate first
        self.materialize_child_strength(pattern_id, current_frame);

        let entry_strength = if let Some(entry) = self.spatial_routing_table.get_mut(&pattern_id) {
            entry.activation_strength += 1.0;
            entry.last_activation_frame = current_frame;
            entry.activation_strength
        } else {
            let entry = self.temporal_routing_table.get_mut(&pattern_id).unwrap();
            entry.activation_strength += 1.0;
            entry.last_activation_frame = current_frame;
            entry.activation_strength
        };

        Some(current_frame + (entry_strength / self.pattern_forget_rate).ceil() as i64)
    }

    /// Add a temporal child pattern to the temporal routing table and populate its context.
    /// Returns death frame for pattern neurons.
    pub fn add_temporal_pattern(&mut self, pattern_id: NeuronId, context: &[ContextRefEntry], current_frame: FrameNumber) -> Option<FrameNumber> {
        self.add_temporal_child(pattern_id, 0.0);
        for entry in context { self.add_temporal_context(pattern_id, entry.neuron_id, entry.distance, 1.0); }
        self.strengthen_child_activation(pattern_id, current_frame)
    }

    /// Add a spatial child pattern to the spatial routing table and populate its context.
    /// Spatial context has no distance dimension — entries are just (pattern, ctx_neuron) pairs.
    /// Returns death frame for pattern neurons.
    pub fn add_spatial_pattern(&mut self, pattern_id: NeuronId, context: &[NeuronId], current_frame: FrameNumber) -> Option<FrameNumber> {
        self.add_spatial_child(pattern_id, 0.0);
        for &ctx_neuron_id in context { self.add_spatial_context(pattern_id, ctx_neuron_id, 1.0); }
        self.strengthen_child_activation(pattern_id, current_frame)
    }

    /// Add a temporal child pattern to the temporal routing table (no context yet).
    pub fn add_temporal_child(&mut self, pattern_id: NeuronId, initial_strength: f64) {
        if !self.temporal_routing_table.contains_key(&pattern_id) {
            self.temporal_routing_table.insert(pattern_id, TemporalRoutingEntry {
                context: TemporalContext::new(),
                activation_strength: initial_strength,
                last_activation_frame: 0,
            });
        }
    }

    /// Add a spatial child pattern to the spatial routing table (no context yet).
    pub fn add_spatial_child(&mut self, pattern_id: NeuronId, initial_strength: f64) {
        if !self.spatial_routing_table.contains_key(&pattern_id) {
            self.spatial_routing_table.insert(pattern_id, SpatialRoutingEntry {
                context: crate::context::SpatialContext::new(),
                activation_strength: initial_strength,
                last_activation_frame: 0,
            });
        }
    }

    /// Check if a child pattern can be deleted (is a zombie).
    pub fn can_delete_child(&self, pattern_id: NeuronId, current_frame: FrameNumber) -> bool {
        // if the pattern has not been activated in some time, die!
        self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0
    }

    // ── TemporalContext management ───────────────────────────────────────────────────

    /// Adds an entry to a TEMPORAL pattern's context by (neuron_id, distance).
    pub fn add_temporal_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance, strength: Strength) {
        let entry = self.temporal_routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("add_temporal_context: pattern not found in temporal routing table: {}", pattern_id));

        if entry.context.has_key(neuron_id, distance) {
            entry.context.strengthen_neuron(neuron_id, distance);
        } else {
            entry.context.add_neuron(neuron_id, distance, strength);
        }

        self.add_temporal_context_index(neuron_id, distance, pattern_id);
    }

    /// Adds an entry to a SPATIAL pattern's context by neuron_id (no distance).
    pub fn add_spatial_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, strength: Strength) {
        let entry = self.spatial_routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("add_spatial_context: pattern not found in spatial routing table: {}", pattern_id));

        if entry.context.has_key(neuron_id) {
            entry.context.strengthen_neuron(neuron_id);
        } else {
            entry.context.add_neuron(neuron_id, strength);
        }

        self.add_spatial_context_index(neuron_id, pattern_id);
    }

    /// Adds a neuron to the temporal context index.
    fn add_temporal_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) {
        let dist_map = self.temporal_context_index.entry(neuron_id).or_insert_with(FxHashMap::default);
        let patterns = dist_map.entry(distance).or_insert_with(FxHashSet::default);
        patterns.insert(pattern_id);
    }

    /// Adds a neuron to the spatial context index.
    fn add_spatial_context_index(&mut self, neuron_id: NeuronId, pattern_id: NeuronId) {
        let patterns = self.spatial_context_index.entry(neuron_id).or_insert_with(FxHashSet::default);
        patterns.insert(pattern_id);
    }

    /// Removes an entry from a TEMPORAL child pattern's context.
    /// Returns true if no pattern references this neuron at this distance anymore (orphaned).
    pub fn remove_temporal_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {
        let entry = self.temporal_routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("remove_temporal_context: pattern {} not found in temporal routing table of neuron {}", pattern_id, self.id));
        entry.context.remove(neuron_id, distance);
        self.remove_temporal_context_index(neuron_id, distance, pattern_id)
    }


    /// Remove a pattern from the temporal context inverted index for a given neuron/distance.
    /// Returns true if no pattern references this neuron at this distance anymore (orphaned).
    pub fn remove_temporal_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) -> bool {
        let dist_map = self.temporal_context_index.get_mut(&neuron_id)
            .unwrap_or_else(|| panic!("remove_temporal_context_index: neuron {} not found in temporal_context_index of neuron {}", neuron_id, self.id));
        let patterns = dist_map.get_mut(&distance)
            .unwrap_or_else(|| panic!("remove_temporal_context_index: distance {} not found for neuron {} in temporal_context_index of neuron {}", distance, neuron_id, self.id));
        patterns.remove(&pattern_id);
        if patterns.is_empty() {
            dist_map.remove(&distance);
            if dist_map.is_empty() { self.temporal_context_index.remove(&neuron_id); }
            return true;
        }
        false
    }


    /// Remove all temporal references to a dying neuron from this parent's children's contexts.
    /// Returns pattern IDs whose context was modified (caller checks if deletable).
    pub fn remove_temporal_context_neuron(&mut self, neuron_id: NeuronId, distances: &FxHashSet<Distance>) -> FxHashSet<NeuronId> {
        let mut affected_patterns = FxHashSet::default();
        let dist_map = self.temporal_context_index.get(&neuron_id)
            .unwrap_or_else(|| panic!("remove_temporal_context_neuron: neuron {} not found in temporal_context_index of neuron {}", neuron_id, self.id));

        let mut ops: Vec<(Distance, Vec<NeuronId>)> = Vec::new();
        for &distance in distances {
            let patterns = dist_map.get(&distance)
                .unwrap_or_else(|| panic!("remove_temporal_context_neuron: distance {} for neuron {} not found in temporal_context_index of neuron {}", distance, neuron_id, self.id));
            ops.push((distance, patterns.iter().copied().collect()));
        }

        for (distance, pattern_ids) in &ops {
            for &pattern_id in pattern_ids {
                let entry = self.temporal_routing_table.get_mut(&pattern_id)
                    .unwrap_or_else(|| panic!("remove_temporal_context_neuron: pattern {} not found in temporal routing table of neuron {}", pattern_id, self.id));
                entry.context.remove(neuron_id, *distance);
                affected_patterns.insert(pattern_id);
            }
        }

        if let Some(dist_map) = self.temporal_context_index.get_mut(&neuron_id) {
            for &distance in distances {
                dist_map.remove(&distance);
            }
            if dist_map.is_empty() { self.temporal_context_index.remove(&neuron_id); }
        }

        affected_patterns
    }


    // ── TemporalContext references ───────────────────────────────────────────────────

    /// Add a TEMPORAL context reference from another neuron to this neuron.
    pub fn add_temporal_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        self.temporal_context_refs.entry(referencing_neuron_id).or_insert_with(FxHashSet::default).insert(distance);
    }

    /// Add a SPATIAL context reference from another neuron to this neuron.
    pub fn add_spatial_context_ref(&mut self, referencing_neuron_id: NeuronId) {
        self.spatial_context_refs.insert(referencing_neuron_id);
    }

    /// Remove a TEMPORAL context reference from another neuron to this neuron.
    pub fn remove_temporal_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        let distances = match self.temporal_context_refs.get_mut(&referencing_neuron_id) {
            Some(d) => d,
            None => return,
        };
        distances.remove(&distance);
        if distances.is_empty() { self.temporal_context_refs.remove(&referencing_neuron_id); }
    }

    /// Remove a SPATIAL context reference from another neuron to this neuron.
    pub fn remove_spatial_context_ref(&mut self, referencing_neuron_id: NeuronId) {
        self.spatial_context_refs.remove(&referencing_neuron_id);
    }

    /// Remove all SPATIAL references to a dying neuron from this parent's children's contexts.
    /// Returns pattern IDs whose context was modified (caller checks if deletable).
    pub fn remove_spatial_context_neuron(&mut self, neuron_id: NeuronId) -> FxHashSet<NeuronId> {
        let mut affected_patterns = FxHashSet::default();
        let pattern_ids: Vec<NeuronId> = match self.spatial_context_index.get(&neuron_id) {
            Some(ps) => ps.iter().copied().collect(),
            None => return affected_patterns,
        };
        for &pattern_id in &pattern_ids {
            let entry = self.spatial_routing_table.get_mut(&pattern_id)
                .unwrap_or_else(|| panic!("remove_spatial_context_neuron: pattern {} not found in spatial routing table of neuron {}", pattern_id, self.id));
            entry.context.remove(neuron_id);
            affected_patterns.insert(pattern_id);
        }
        self.spatial_context_index.remove(&neuron_id);
        affected_patterns
    }

    /// Check if a neuron exists in the spatial context index (for column purge ops).
    pub fn has_spatial_context_index_entry(&self, neuron_id: NeuronId) -> bool {
        self.spatial_context_index.contains_key(&neuron_id)
    }

    /// Remove a single context entry from a SPATIAL child pattern's context.
    /// Returns true if the context neuron is no longer referenced by any child spatial pattern.
    pub fn remove_spatial_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId) -> bool {
        if let Some(entry) = self.spatial_routing_table.get_mut(&pattern_id) {
            entry.context.remove(neuron_id);
        }
        // scrub the inverted index for (pattern, ctx_neuron)
        if let Some(patterns) = self.spatial_context_index.get_mut(&neuron_id) {
            patterns.remove(&pattern_id);
            if patterns.is_empty() {
                self.spatial_context_index.remove(&neuron_id);
                return true;
            }
        }
        false
    }

    /// Check if a context entry exists for a spatial child pattern (for same-pulse de-dup).
    pub fn has_spatial_context_key(&self, pattern_id: NeuronId, neuron_id: NeuronId) -> bool {
        self.spatial_routing_table.get(&pattern_id)
            .map_or(false, |e| e.context.has_key(neuron_id))
    }


    /// Apply a batch of TEMPORAL context-reference updates targeting this neuron.
    pub fn apply_temporal_context_ref_updates(&mut self, updates: &[TemporalContextRefUpdate]) {
        for update in updates {
            self.add_temporal_context_ref(update.parent_id, update.distance);
        }
    }

    /// Apply a batch of SPATIAL context-reference updates targeting this neuron.
    pub fn apply_spatial_context_ref_updates(&mut self, updates: &[SpatialContextRefUpdate]) {
        for update in updates {
            self.add_spatial_context_ref(update.parent_id);
        }
    }

    /// Get temporal context refs (for delete cascade).
    pub fn get_temporal_context_refs(&self) -> &FxHashMap<NeuronId, FxHashSet<Distance>> {
        &self.temporal_context_refs
    }

    /// Get spatial context refs (for spatial delete cascade — the set of parent neurons whose
    /// spatial routing tables reference this neuron).
    pub fn get_spatial_context_refs(&self) -> &FxHashSet<NeuronId> {
        &self.spatial_context_refs
    }

    /// Read-only access to the temporal routing table (for Column snapshot/death-frame/delete ops).
    pub fn get_temporal_routing_table(&self) -> &FxHashMap<NeuronId, TemporalRoutingEntry> {
        &self.temporal_routing_table
    }

    /// Read-only access to the spatial routing table (for delete cascade).
    pub fn get_spatial_routing_table(&self) -> &FxHashMap<NeuronId, SpatialRoutingEntry> {
        &self.spatial_routing_table
    }

    /// Inspection: flatten this neuron's outgoing connections into
    /// (distance, target_neuron_id, strength, reward) tuples. Spatial connections
    /// surface at distance=0; temporal at their distance index. Used by the diagnostic API
    /// to dump connection state for tipping-point analysis.
    pub fn get_connections(&self) -> Vec<(Distance, NeuronId, f64, f64)> {
        let mut out = Vec::new();
        for (&target_id, conn) in &self.spatial_connections {
            out.push((0, target_id, conn.strength, conn.reward));
        }
        for (idx, dist_map) in self.temporal_connections.iter().enumerate() {
            if idx == 0 { continue; }
            let dist = idx as Distance;
            for (&target_id, conn) in dist_map {
                out.push((dist, target_id, conn.strength, conn.reward));
            }
        }
        out
    }

    /// Mutable access to the temporal routing table (for Column restore — sets last_activation_frame).
    pub fn get_temporal_routing_table_mut(&mut self) -> &mut FxHashMap<NeuronId, TemporalRoutingEntry> {
        &mut self.temporal_routing_table
    }

    /// Mutable access to the spatial routing table (for Column restore — sets last_activation_frame).
    pub fn get_spatial_routing_table_mut(&mut self) -> &mut FxHashMap<NeuronId, SpatialRoutingEntry> {
        &mut self.spatial_routing_table
    }


    /// Remove a child pattern from whichever routing table holds it (for Column delete cascade).
    pub fn remove_routing_entry(&mut self, pattern_id: NeuronId) {
        if self.spatial_routing_table.remove(&pattern_id).is_some() { return; }
        self.temporal_routing_table.remove(&pattern_id);
    }

    /// Check if a temporal context key exists for a child pattern (for Column delete cascade).
    pub fn has_temporal_context_key(&self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {
        self.temporal_routing_table.get(&pattern_id)
            .map_or(false, |e| e.context.has_key(neuron_id, distance))
    }


    /// Check if a (neuron, distance) exists in the temporal context index (for Column purge).
    pub fn has_temporal_context_index_entry(&self, neuron_id: NeuronId, distance: Distance) -> bool {
        self.temporal_context_index.get(&neuron_id)
            .map_or(false, |dist_map| dist_map.contains_key(&distance))
    }


    // ── Frame processing ─────────────────────────────────────────────────────

    /// Process a frame for this neuron: derive per-age tasks, learn connections, match patterns,
    /// install pre-created error-correction patterns, and cast votes. One call per active neuron
    /// per frame. Matching is skipped when level_context is None/empty. Votes are cast for each
    /// eligible voting age unless suppressed by a this-frame match (activate=true) or
    /// error-correction activation at the same age.
    /// Spatial frame processing — no per-age iteration, no distance dimension.
    /// Queries `spatial_routing_table` and `spatial_context_index` for pattern matching.
    /// Writes co-activation predictions to `spatial_connections` for next-frame voting.
    pub fn process_spatial_frame(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        level_context: Option<&SpatialContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        actives: &[ActiveNeuron],
        current_frame: FrameNumber,
        corrections: &[Correction],
        _error_feedback: &[ErrorFeedback],
        learning: bool,
    ) -> ProcessFrameResult {
        let mut timings = NeuronOpTimings::default();

        // Check the age=0 state — the only meaningful slot for spatial.
        let state_at_0 = age_states.get(&0);
        let already_suppressed = state_at_0.map_or(false, |s| s.activated_pattern_id.is_some());

        // The "should learn" gate excludes new error patterns just minted this frame.
        let should_learn = learning && !new_error_pattern_ids.contains(&self.id);

        // Pattern recognition. For spatial, only age=0 is meaningful and only one pattern can match.
        let t = std::time::Instant::now();
        let (matches, match_refs) = if already_suppressed {
            (Vec::new(), Vec::new())
        } else {
            self.recognize_spatial_patterns(level_context, current_frame, learning, &mut timings)
        };
        timings.recognize_patterns = t.elapsed().as_secs_f64();

        // Install pre-created error-correction patterns. For spatial, `corrections` is currently
        // empty — spatial corrections come from a separate mint pass in brain's process_spatial_corrections,
        // not from inside process_spatial_frame. The empty-loop is kept for API symmetry.
        let t = std::time::Instant::now();
        let CorrectResult { correction_activations, context_ref_updates: correction_refs } = self.correct_errors(corrections, current_frame);
        timings.correct_errors = t.elapsed().as_secs_f64();

        // Cast spatial votes — single bucket at distance 0.
        // Suppressed if any pattern was matched this frame (matches.activate) or a correction installed.
        let t = std::time::Instant::now();
        let votes = self.generate_spatial_votes(state_at_0, level_context, &matches, &correction_activations);
        timings.generate_votes = t.elapsed().as_secs_f64();

        // Learn spatial connections AFTER vote generation so the vote reads the prior-frame
        // spatial_connections rather than the just-strengthened edges to this frame's co-actives.
        if should_learn {
            let t = std::time::Instant::now();
            self.learn_spatial_connections(actives);
            timings.learn_connections = t.elapsed().as_secs_f64();
        }

        let mut context_ref_updates = match_refs;
        context_ref_updates.extend(correction_refs);
        ProcessFrameResult { matches, correction_activations, context_ref_updates, votes, timings }
    }

    /// Spatial pattern recognition. Walks `spatial_routing_table` and finds the best matching child
    /// pattern given the observed co-activation `SpatialContext`.
    fn recognize_spatial_patterns(
        &mut self,
        level_context: Option<&SpatialContext>,
        current_frame: FrameNumber,
        learning: bool,
        timings: &mut NeuronOpTimings,
    ) -> (Vec<PatternMatch>, Vec<TemporalContextRefUpdate>) {
        let context_ref_updates = Vec::new();

        // Bail if no observed context.
        let observed = match level_context {
            Some(c) if c.size() > 0 => c,
            _ => return (Vec::new(), context_ref_updates),
        };

        // Candidate set via spatial_context_index — narrow to patterns that share at least one ctx neuron.
        let t = std::time::Instant::now();
        let mut candidates: FxHashSet<NeuronId> = FxHashSet::default();
        for (&neuron_id, _) in observed.entries() {
            if let Some(patterns) = self.spatial_context_index.get(&neuron_id) {
                for &pid in patterns { candidates.insert(pid); }
            }
        }
        timings.recognize_candidate_search += t.elapsed().as_secs_f64();
        if candidates.is_empty() { return (Vec::new(), context_ref_updates); }

        // Score candidates and pick the best.
        // Recognition strictness is the adaptive merge (1 − E) for the spatial bucket — same value correction reads.
        let merge_threshold = self.grouping_merge_threshold(self.spatial_error_stats.as_ref());
        let eval_start = std::time::Instant::now();
        let candidate_count = candidates.len();
        let mut best: Option<PartialMatch> = None;
        for pattern_id in candidates {
            if self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0 { continue; }
            let entry = self.spatial_routing_table.get(&pattern_id)
                .unwrap_or_else(|| panic!("recognize_spatial_patterns: pattern {} indexed but missing from spatial_routing_table", pattern_id));
            let m = match entry.context.match_observed(observed, merge_threshold) {
                Some(m) => m,
                None => continue,
            };
            if let Some(ref b) = best {
                if m.score < b.score { continue; }
                if m.score == b.score && pattern_id > b.pattern_id { continue; } // prefer smaller id
            }
            best = Some(PartialMatch { pattern_id, age: 0, score: m.score });
        }
        timings.recognize_candidate_eval += eval_start.elapsed().as_secs_f64();
        timings.recognize_candidates_evaluated += candidate_count as u64;

        let best = match best {
            Some(b) => b,
            None => return (Vec::new(), context_ref_updates),
        };

        // Activate the winning pattern (strengthen child activation in learning mode).
        let death_frame = if learning { self.strengthen_child_activation(best.pattern_id, current_frame) } else { None };
        let matches = vec![PatternMatch {
            pattern_id: best.pattern_id,
            age: 0,
            activate: true,
            death_frame,
        }];
        (matches, context_ref_updates)
    }

    /// Generate spatial vote — a single bucket at distance 0. Suppressed if the neuron already
    /// activated a higher-level pattern this frame, was matched by recognize_spatial_patterns,
    /// or had a correction installed.
    fn generate_spatial_votes(
        &self,
        state_at_0: Option<&AgeState>,
        level_context: Option<&SpatialContext>,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
    ) -> Vec<AgeVotes> {
        // Suppression checks.
        if state_at_0.map_or(false, |s| s.activated_pattern_id.is_some()) { return Vec::new(); }
        for m in matches { if m.activate && m.age == 0 { return Vec::new(); } }
        for c in correction_activations { if c.age == 0 { return Vec::new(); } }

        let cast_votes = self.vote_at_distance(0);

        // Context attached to this vote — the spatial co-active neurons.
        let ctx_entries: Vec<ContextRefEntry> = if let Some(lc) = level_context {
            lc.entries().keys()
                .map(|&nid| ContextRefEntry { neuron_id: nid, distance: 0 })
                .collect()
        } else { Vec::new() };

        vec![AgeVotes {
            age: 0,
            votes: cast_votes,
            context: ctx_entries,
            threshold: self.get_spatial_error_threshold(),
        }]
    }

    /// Spatial connection learning — strengthen edges to every co-active neuron.
    /// Writes to `spatial_connections` only. Self-connections are skipped.
    fn learn_spatial_connections(&mut self, actives: &[ActiveNeuron]) {
        for active in actives {
            if active.id == self.id { continue; }
            self.upsert_connection(0, active.id, active.channel_id, active.reward);
        }
    }

    /// Temporal frame processing — per-age iteration over the sliding window. Matching
    /// queries `temporal_routing_table` and produces d>0 sequence votes from `temporal_connections`.
    /// Connections are learned at the TOP of the frame: connections[age] is updated before
    /// the vote step which reads connections[age+1], so learn-then-vote doesn't echo back this
    /// frame's actives.
    pub fn process_temporal_frame(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        actives: &[ActiveNeuron],
        current_frame: FrameNumber,
        corrections: &[Correction],
        error_feedback: &[ErrorFeedback],
        learning: bool,
    ) -> ProcessFrameResult {

        // Fold prior-frame error feedback into per-age accuracy stats first.
        // That way the threshold attached to this frame's votes (computed in generate_votes) reflects the latest sample.
        // Non-learning mode skips this because accuracy stats are connection-substrate state.
        if learning {
            for fb in error_feedback { self.record_temporal_error(fb.age, fb.error_rate); }
        }
        let mut timings = NeuronOpTimings::default();

        let should_learn = learning && !new_error_pattern_ids.contains(&self.id);

        // Temporal learns connections[age>0] which the vote step (connections[age+1]) doesn't read,
        // so it's safe to run learn FIRST and have the vote read the same-frame updated edges only
        // for ages that don't vote at this distance.
        if should_learn {
            let t = std::time::Instant::now();
            self.learn_temporal_connections(age_states, actives);
            timings.learn_connections = t.elapsed().as_secs_f64();
        }

        // Match temporal patterns against the level_context if we have one and eligible ages.
        let t = std::time::Instant::now();
        let RecognizeResult { matches, context_ref_updates: match_refs } = self.recognize_temporal_patterns(age_states, memory_depth, level_context, new_error_pattern_ids, current_frame, learning, &mut timings);
        timings.recognize_patterns = t.elapsed().as_secs_f64();

        // Install pre-created temporal error-correction patterns as children and emit their contextRef adds.
        let t = std::time::Instant::now();
        let CorrectResult { correction_activations, context_ref_updates: correction_refs } = self.correct_errors(corrections, current_frame);
        timings.correct_errors = t.elapsed().as_secs_f64();

        // Cast temporal votes for each eligible age, suppressing any ages that activated a pattern.
        let t = std::time::Instant::now();
        let votes = self.generate_temporal_votes(age_states, memory_depth, level_context, &matches, &correction_activations);
        timings.generate_votes = t.elapsed().as_secs_f64();

        let mut context_ref_updates = match_refs;
        context_ref_updates.extend(correction_refs);
        ProcessFrameResult { matches, correction_activations, context_ref_updates, votes, timings }
    }

    // ── Pattern recognition ──────────────────────────────────────────────────

    /// Find matching patterns for this parent neuron given the observed level context.
    /// For each age that matches, the matched pattern's context is refined immediately.
    /// Cross-neuron contextRef additions and removals are returned for delivery in a separate
    /// post-match phase. If the same child pattern matches at multiple ages for this parent,
    /// all refined matches are returned in order so side effects remain identical, but only
    /// the first match is flagged as an activation candidate. Matching is skipped entirely for
    /// new error patterns, when level_context is None/empty, or when no age is eligible.
    /// Eligible ages: non-activated and not the oldest age in the sliding window (where no
    /// future context exists to match against).
    fn recognize_temporal_patterns(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        current_frame: FrameNumber,
        learning: bool,
        timings: &mut NeuronOpTimings,
    ) -> RecognizeResult {
        let mut matches = Vec::new();
        let context_ref_updates = Vec::new();

        // Warmup gate: skip pattern recognition until the context window has had a chance to fill up.
        // Without this, patterns whose stored contexts include entries at distances > current_frame are unfairly penalized.
        if (current_frame as u32) < self.context_length {
            return RecognizeResult { matches, context_ref_updates };
        }

        let ctx = match level_context {
            Some(c) if c.size() > 0 => c,
            _ => return RecognizeResult { matches, context_ref_updates },
        };
        let mut activated_pattern_ids = FxHashSet::default();

        // sort the ages so that we loop them in order
        let mut ages: Vec<Distance> = age_states.keys().copied().collect();
        ages.sort_unstable();

        for age in ages {
            let state = &age_states[&age];
            // Skip the oldest age — there's no future context to match against.
            if state.activated_pattern_id.is_some() { continue; }
            if age == memory_depth - 1 { continue; }

            // active ages are processed in ascending order (most recent first). The first age that
            // produces a match at that age is refined and preserved. More recent ages tend to have
            // the richest available context, so they are processed first.
            let best = match self.find_best_pattern_match_at_age(ctx, age, new_error_pattern_ids, current_frame, timings) {
                Some(b) => b,
                None => continue, // try older age if there is a match
            };

            // refine_context is intentionally disabled. Refining the matched pattern's
            // stored context mid-training made recognition non-reproducible: training-time
            // recognition saw "in-progress" patterns, later replays saw fully-refined ones,
            // so trajectories diverged. Every match now uses the pattern exactly as it
            // was created/installed. PartialMatch's common/missing/novel slices that
            // refine_context needed are gone with it.

            // activate the matched pattern if it was not elected to be activated already
            let activate = !activated_pattern_ids.contains(&best.pattern_id);
            activated_pattern_ids.insert(best.pattern_id);

            // Strengthen child activation automatically here, but only in learning mode.
            // Non-learning eval still activates the pattern but does not extend its life.
            let death_frame = if activate && learning { self.strengthen_child_activation(best.pattern_id, current_frame) } else { None };

            // include the best match for the age in the results
            matches.push(PatternMatch {
                pattern_id: best.pattern_id,
                age: best.age,
                activate,
                death_frame,
            });
        }
        RecognizeResult { matches, context_ref_updates }
    }

    /// Find the best matching temporal pattern for a specific active age.
    fn find_best_pattern_match_at_age(&self, observed: &TemporalContext, age: Distance, exclude_ids: &FxHashSet<NeuronId>, current_frame: FrameNumber, timings: &mut NeuronOpTimings) -> Option<PartialMatch> {
        let mut best: Option<PartialMatch> = None;

        // Use the inverted index to narrow the search to child patterns that share at least one
        // exact neuron/distance entry with the observed context at this active age.
        let t = std::time::Instant::now();
        let candidate_ids = self.get_pattern_candidates_at_age(observed, age);
        timings.recognize_candidate_search += t.elapsed().as_secs_f64();
        if candidate_ids.is_empty() { return None; }

        // Recognition strictness is the adaptive merge (1 − E) for this age's bucket — same value correction reads.
        let merge_threshold = self.grouping_merge_threshold(self.temporal_error_stats.get(age as usize).and_then(|s| s.as_ref()));
        let eval_start = std::time::Instant::now();
        let candidate_count = candidate_ids.len();
        // go through the candidate patterns and find the best match
        for pattern_id in candidate_ids {

            // check if pattern is still alive (skip functionally dead patterns)
            if self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0 { continue; }

            // get the context of the pattern
            let entry = self.temporal_routing_table.get(&pattern_id)
                .unwrap_or_else(|| panic!("Cannot find context for pattern: {}", pattern_id));

            // context.match_observed() handles the full scoring and threshold check;
            // the index only decides which child patterns are worth evaluating.
            // exclude_ids masks out brand-new neurons so they don't count as "novel" misses.
            let m = match entry.context.match_observed(observed, age, merge_threshold, Some(exclude_ids)) {
                Some(m) => m,
                None => continue, // nothing to do if there is no match
            };

            // if there is already a best match, check if this match is better
            if let Some(ref b) = best {

                // if the previous best is better than this match, skip
                if m.score < b.score { continue; }

                // to preserve determinism when multiple patterns achieve the exact same score, explicitly tie-break using pattern_id
                if m.score == b.score && pattern_id > b.pattern_id { continue; } // prefer smaller id (older)
            }

            // store the best match
            best = Some(PartialMatch {
                pattern_id,
                age,
                score: m.score,
            });
        }
        timings.recognize_candidate_eval += eval_start.elapsed().as_secs_f64();
        timings.recognize_candidates_evaluated += candidate_count as u64;
        best
    }

    /// Find candidate child patterns for a specific active age using the inverted index.
    /// Candidate patterns must share at least one exact neuron/distance context entry with the
    /// observed context after converting absolute observed ages into pattern-relative distances.
    ///
    /// The observed context is keyed by absolute age within the current frame snapshot, while each
    /// stored pattern context is keyed by distance relative to the parent's own activation age.
    /// For a parent active at `age`, an observed entry at absolute distance `d` maps to a pattern
    /// entry at relative distance `d - age`.
    ///
    /// Missing index entries are expected here: they just mean the observed neuron/distance pair is
    /// not referenced by any child pattern and therefore contributes no candidates.
    fn get_pattern_candidates_at_age(&self, observed: &TemporalContext, age: Distance) -> FxHashSet<NeuronId> {
        let mut candidates = FxHashSet::default();

        // Temporal context entries are at strictly positive relative distances — context neurons must be
        // older than the parent neuron itself.
        let min_distance: i64 = 1;

        for (&neuron_id, distance_map) in observed.entries() {

            // First narrow by exact context neuron ID. If this neuron does not appear in the index,
            // no child pattern references it anywhere in this parent's routing table.
            let indexed_distances = match self.temporal_context_index.get(&neuron_id) {
                Some(d) => d,
                None => continue,
            };

            for &absolute_distance in distance_map.keys() {

                // Convert the observed absolute age into the pattern-relative distance used by the
                // routing table and inverted index. The minimum admissible distance depends on phase
                // (temporal: ≥1; spatial: ≥0).
                let pattern_distance = absolute_distance as i64 - age as i64;
                if pattern_distance < min_distance { continue; }
                let pattern_distance = pattern_distance as Distance;

                // Then narrow by exact relative distance. Missing entries here are also expected and
                // simply mean no child pattern references this neuron at this particular distance.
                let pattern_ids = match indexed_distances.get(&pattern_distance) {
                    Some(p) => p,
                    None => continue,
                };

                // A candidate only needs one exact neuron/distance overlap to be worth a full score.
                for &pattern_id in pattern_ids { candidates.insert(pattern_id); }
            }
        }
        candidates
    }

    // ── Error correction ─────────────────────────────────────────────────────

    /// Install pre-created error-correction pattern neurons as children at the given ages.
    /// Each install adds a child pattern entry to the routing table and emits contextRef adds
    /// for every context entry so target neurons can track this parent.
    fn correct_errors(&mut self, corrections: &[Correction], current_frame: FrameNumber) -> CorrectResult {
        let mut correction_activations = Vec::new();
        let mut context_ref_updates = Vec::new();
        for correction in corrections {

            // add the pattern to the routing table
            let death_frame = self.add_temporal_pattern(correction.pattern_id, &correction.context_entries, current_frame);

            // add the pattern to the activations to be returned with its death frame
            correction_activations.push(CorrectionActivation { pattern_id: correction.pattern_id, age: correction.age, death_frame });

            // also update the context reference updates to be returned for the new patterns
            for entry in &correction.context_entries {
                context_ref_updates.push(TemporalContextRefUpdate { neuron_id: entry.neuron_id, distance: entry.distance, parent_id: 0 });
            }
        }
        CorrectResult { correction_activations, context_ref_updates }
    }

    // ── Vote generation ──────────────────────────────────────────────────────

    /// Cast one temporal vote per non-suppressed age; get_suppressed_ages owns the full inhibition rule.
    /// Runs for new error patterns too so their state.context gets populated for next-frame corrections.
    /// The per-age context is reshaped from level_context locally — context_by_age is a pure reshape
    /// (no extra info), so it's derived here instead of being shipped over the wire.
    fn generate_temporal_votes(
        &self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
    ) -> Vec<AgeVotes> {

        // Resolve every inhibited age once, then keep only the ages this neuron will actually vote on.
        let suppressed_ages = self.get_suppressed_ages(age_states, memory_depth, matches, correction_activations);
        let voting_ages: Vec<Distance> = age_states.keys().copied().filter(|age| !suppressed_ages.contains(age)).collect();

        // pre-bucket level_context into those voting ages so each cast picks its per-age context up with one lookup
        let context_by_age = self.derive_context_by_age(level_context, &voting_ages);

        // cast one vote per voting age
        let mut votes = Vec::with_capacity(voting_ages.len());
        for age in voting_ages {

            // Temporal votes predict the next frame via temporal_connections[age+1].
            let cast_votes = self.vote(age);
            votes.push(AgeVotes {
                age,
                votes: cast_votes,
                context: context_by_age.get(age as usize).cloned().unwrap_or_default(),
                threshold: self.get_temporal_error_threshold(age),
            });
        }
        votes
    }

    /// Collect every age whose vote is inhibited this frame, so generate_temporal_votes can skip them in one test.
    /// An age is suppressed when any of the following holds:
    /// - a recognized pattern activated at that age this frame — the firing parent represents it, so the subsumed
    ///   neuron must not also vote (only the activation candidate inhibits, not non-activating refinement matches),
    /// - an error-correction pattern was installed at that age this frame — the correction now owns that prediction,
    /// - the neuron was already marked subsumed at that age on a prior frame (activated_pattern_id is set),
    /// - it is the oldest slot in the window, with no future frame to vote toward.
    /// The oldest-slot rule applies only for context_length > 1; at context_length == 1 the single age 0 is the
    /// prediction-bearing age for single-frame episodes and must vote, otherwise process_frame emits no votes at all.
    fn get_suppressed_ages(
        &self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
    ) -> FxHashSet<Distance> {
        let mut suppressed_ages = FxHashSet::default();

        // if the neuron decided to activate a child pattern in this frame, its vote is suppressed in this frame
        for m in matches { if m.activate { suppressed_ages.insert(m.age); } }

        // if the neuron had a bad inference last frame from an age, and it needs to be corrected, suppress the vote for that age
        for c in correction_activations { suppressed_ages.insert(c.age); }

        // loop over the ages and check
        for (&age, state) in age_states {

            // if the neuron activated a child pattern in an age in a previous frame, that age stays suppressed
            if state.activated_pattern_id.is_some() { suppressed_ages.insert(age); continue; }

            // oldest age has no connections for next frame to vote, except age 0 at context_length 1, used for MNIST tests.
            if self.context_length > 1 && age >= memory_depth - 1 { suppressed_ages.insert(age); }
        }

        suppressed_ages
    }

    /// Reshape level_context into per-voting-age buckets, building only the supplied voting ages.
    /// For each entry at context-age `ctx_age`, emits it into every voting age < ctx_age with distance = ctx_age - age.
    /// Indexed by voting age so generate_temporal_votes picks up each per-age context with a single lookup —
    /// no per-age scan, no `ctx_age > age` branch. Suppressed ages get no bucket since their vote is never cast.
    /// Pure reshape kept inside the neuron to save per-frame MPI traffic.
    fn derive_context_by_age(&self, level_context: Option<&TemporalContext>, voting_ages: &[Distance]) -> Vec<Vec<ContextRefEntry>> {
        let mut context_by_age: Vec<Vec<ContextRefEntry>> = Vec::new();
        let ctx = match level_context {
            Some(c) => c,
            None => return context_by_age,
        };
        let voting: FxHashSet<Distance> = voting_ages.iter().copied().collect();
        for (&neuron_id, distance_map) in ctx.entries() {
            for &ctx_age in distance_map.keys() {
                for a in 0..ctx_age {
                    if !voting.contains(&a) { continue; }
                    let idx = a as usize;
                    if idx >= context_by_age.len() { context_by_age.resize_with(idx + 1, Vec::new); }
                    context_by_age[idx].push(ContextRefEntry { neuron_id, distance: ctx_age - a });
                }
            }
        }
        context_by_age
    }

    // ── Connection learning ──────────────────────────────────────────────────

    /// Update connections based on the currently observed actives. For each active age > 0:
    /// upsert a connection per active (create-or-strengthen + alt-action). Rewards are
    /// pre-resolved thalamus-side (0 for events, observed value for actions). Skipped entirely
    /// for new error patterns (they were just created this frame and have nothing yet to reinforce).
    ///
    /// Note: connections are no longer negatively reinforced. Previously, predictions at this
    /// distance that didn't occur were weakened (with eventual deletion at strength=0). This
    /// produced periodic discrete-death bursts during multi-episode training — a connection
    /// drifts down by ~1/episode until it finally hits 0, then dies abruptly, shifting the
    /// neuron's vote profile and triggering a cascade of error-pattern creation. For deterministic
    /// memorization scenarios we want predictions that ever occurred to remain available;
    /// non-occurrences should not erase them. Mirrors the action-connection behaviour, which
    /// was already kept-only-strengthen.
    /// Temporal connection learning — strengthen temporal_connections[age] for every active age>0.
    /// A neuron active at age=k learns a k-distance prediction toward each current actives (which
    /// are at age=0 in `actives`).
    fn learn_temporal_connections(&mut self, age_states: &FxHashMap<Distance, AgeState>, actives: &[ActiveNeuron]) {
        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            if age == 0 { continue; }
            for active in actives {
                self.upsert_connection(age, active.id, active.channel_id, active.reward);
            }
        }
    }

    /// Find an alternative action for a channel that hasn't been tried yet.
    pub(crate) fn find_alternative_action(&self, distance: Distance, channel_id: ChannelId, current_action_id: NeuronId) -> Option<NeuronId> {
        if let Some(action_ids) = self.channel_action_ids.get(&channel_id) {
            for &alt_neuron_id in action_ids {
                if alt_neuron_id != current_action_id && !self.has_connection(distance, alt_neuron_id) {
                    return Some(alt_neuron_id);
                }
            }
        }
        None
    }
}

// ── Serialization structs ────────────────────────────────────────────────────

/// Intermediate struct used during find_best_pattern_match_at_age before
/// the full PatternMatch is assembled.
struct PartialMatch {
    pattern_id: NeuronId,
    age: Distance,
    score: f64,
}

#[derive(Debug, Clone)]
pub struct SerializedNeuron {
    pub id: NeuronId,
    pub pattern_forget_rate: f64,
    pub connections: Vec<SerializedConnection>,
    pub children: Vec<SerializedChild>,
    pub context_refs: Vec<SerializedContextRef>,
    pub error_stats: Vec<SerializedErrorStats>,
}

#[derive(Debug, Clone)]
pub struct SerializedConnection {
    pub distance: Distance,
    pub to_neuron_id: NeuronId,
    pub strength: Strength,
    pub reward: Reward,
}

#[derive(Debug, Clone)]
pub struct SerializedChild {
    pub pattern_id: NeuronId,
    /// true = spatial routing entry (d=0 co-activation), false = temporal routing entry (d>0 sequence).
    /// Restore needs this discriminator because an empty-context spatial child is otherwise
    /// indistinguishable from a temporal one, and the back-compat shims route unconditionally to temporal.
    pub spatial: bool,
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
    pub context: Vec<ContextEntry>,
}

#[derive(Debug, Clone)]
pub struct SerializedContextRef {
    pub parent_id: NeuronId,
    /// true = spatial context ref (distances is the placeholder [0]), false = temporal.
    pub spatial: bool,
    pub distances: Vec<Distance>,
}

#[derive(Debug, Clone)]
pub struct SerializedErrorStats {
    pub age: Distance,
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_neuron(id: NeuronId) -> Neuron {
        Neuron::new(id, 0.01, 0.9, GroupMode::Static, FxHashMap::default(), 10)
    }

    fn make_neuron_with_actions(id: NeuronId, channel_id: ChannelId, action_ids: Vec<NeuronId>) -> Neuron {
        let mut channel_actions = FxHashMap::default();
        channel_actions.insert(channel_id, action_ids);
        Neuron::new(id, 0.01, 0.9, GroupMode::Static, channel_actions, 10)
    }

    #[test]
    fn test_create_and_has_connection() {
        let mut n = make_neuron(1);
        assert!(!n.has_connection(2, 10));
        n.create_connection(2, 10, 1.0, 0.0);
        assert!(n.has_connection(2, 10));
        assert!(!n.has_connection(2, 11));
        assert!(!n.has_connection(3, 10));
    }

    #[test]
    fn test_strengthen_connection_updates_reward() {
        let mut n = make_neuron(1);
        n.create_connection(1, 10, 1.0, 1.0);
        n.strengthen_connection(1, 10, 0.0);
        // strength should be 2, reward should be smoothed: (1/2)*0 + (1/2)*1 = 0.5
        let conn = &n.temporal_connections[1][&10];
        assert_eq!(conn.strength, 2.0);
        assert!((conn.reward - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vote_returns_connections_at_age_plus_one() {
        let mut n = make_neuron(1);
        n.create_connection(3, 10, 2.0, 0.5);
        n.create_connection(3, 11, 1.0, 0.0);
        let votes = n.vote(2); // age=2 → distance=3
        assert_eq!(votes.len(), 2);
    }

    #[test]
    fn test_vote_empty_when_no_connections() {
        let n = make_neuron(1);
        let votes = n.vote(5);
        assert!(votes.is_empty());
    }

    #[test]
    fn test_add_child_and_pattern() {
        let mut n = make_neuron(1);
        let context = vec![
            ContextRefEntry { neuron_id: 10, distance: 1 },
            ContextRefEntry { neuron_id: 11, distance: 2 },
        ];
        let death_frame = n.add_temporal_pattern(100, &context, 10);
        assert!(death_frame.is_some());

        // pattern should be in routing table with 2 context entries
        assert_eq!(n.temporal_routing_table[&100].context.size(), 2);

        // context index should have entries
        assert!(n.temporal_context_index.contains_key(&10));
        assert!(n.temporal_context_index.contains_key(&11));
    }

    #[test]
    fn test_lazy_decay() {
        let mut n = make_neuron(1);
        n.pattern_forget_rate = 0.1;
        n.add_temporal_child(100, 0.0);
        n.strengthen_child_activation(100, 0); // strength=1, last_frame=0

        // at frame 5: effective = 1 - 5*0.1 = 0.5
        assert!((n.get_child_effective_activation_strength(100, 5) - 0.5).abs() < 1e-10);

        // at frame 10: effective = 1 - 10*0.1 = 0.0
        assert!(n.get_child_effective_activation_strength(100, 10) <= 0.0);
        assert!(n.can_delete_child(100, 10));
    }

    #[test]
    fn test_error_threshold_static() {
        // group_threshold 0.9 → static correction threshold = 1 − 0.9 = 0.1.
        let n = make_neuron(1);
        assert!((n.get_temporal_error_threshold(0) - 0.1).abs() < 1e-10);
        assert!((n.get_temporal_error_threshold(5) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_error_threshold_dynamic_warmup() {
        let mut n = Neuron::new(1, 0.01, 0.9, GroupMode::Neutral, FxHashMap::default(), 10);
        // fewer than ERROR_MIN_SAMPLES → falls back to the derived 1 − group_threshold = 0.1
        n.record_temporal_error(0, 0.5);
        n.record_temporal_error(0, 0.5);
        assert!((n.get_temporal_error_threshold(0) - 0.1).abs() < 1e-10); // still warmup

        // after 3 samples → uses mean
        n.record_temporal_error(0, 0.5);
        assert!((n.get_temporal_error_threshold(0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_context_refs() {
        let mut n = make_neuron(1);
        n.add_temporal_context_ref(50, 2);
        n.add_temporal_context_ref(50, 3);
        n.add_temporal_context_ref(60, 1);

        assert_eq!(n.temporal_context_refs.len(), 2);
        assert_eq!(n.temporal_context_refs[&50].len(), 2);

        n.remove_temporal_context_ref(50, 2);
        assert_eq!(n.temporal_context_refs[&50].len(), 1);

        n.remove_temporal_context_ref(50, 3);
        assert!(!n.temporal_context_refs.contains_key(&50)); // auto-cleaned
    }

    #[test]
    fn test_find_alternative_action() {
        let mut n = make_neuron_with_actions(1, 0, vec![10, 11, 12]);
        n.create_connection(1, 10, 1.0, -0.5); // already tried action 10

        // should find 11 or 12 as alternative (not 10, already connected)
        let alt = n.find_alternative_action(1, 0, 10);
        assert!(alt.is_some());
        let alt_id = alt.unwrap();
        assert!(alt_id == 11 || alt_id == 12);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut n = make_neuron(1);
        n.create_connection(1, 10, 2.0, 0.5);
        n.create_connection(2, 11, 1.0, 0.0);
        n.add_temporal_child(100, 0.0);
        n.add_temporal_context(100, 20, 1, 1.0);
        n.add_temporal_context_ref(50, 2);
        // Spatial error stat lives in its own single bucket; the test originally used age=0 which
        // was the spatial slot under the unified-storage model.
        n.record_spatial_error(0.4);
        n.record_spatial_error(0.6);
        n.record_spatial_error(0.5);

        let s = n.serialize();
        assert_eq!(s.id, 1);
        assert_eq!(s.connections.len(), 2);
        assert_eq!(s.children.len(), 1);
        assert_eq!(s.children[0].context.len(), 1);
        assert_eq!(s.context_refs.len(), 1);
        assert_eq!(s.error_stats.len(), 1);
    }
}
