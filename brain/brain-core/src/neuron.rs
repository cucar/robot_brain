/// Neuron - Unified struct for all neurons (sensory and pattern)
///
/// All neurons have:
/// - connections: Vec<FxHashMap<NeuronId, ConnectionData>> - predictions (indexed by distance)
/// - routing_table: FxHashMap<PatternId, RoutingEntry> - child pattern contexts
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

use crate::context::Context;
use crate::types::*;

/// Minimum samples per (neuron, age) before dynamic modes switch off the warmup
/// fallback. Not a tunable knob — kept here to avoid a magic number in get_error_threshold.
const ERROR_MIN_SAMPLES: u64 = 3;

/// Entry in the routing table for a child pattern.
#[derive(Debug, Clone)]
pub struct RoutingEntry {
    pub context: Context,
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

/// A context reference update (add or remove).
/// `neuron_id` is the context neuron being referenced.
/// `parent_id` is the parent neuron that owns the routing table referencing it.
/// `parent_id` is set to 0 when emitted from neuron-level code and filled in
/// by thalamus during `collect_context_ref_updates`.
#[derive(Debug, Clone)]
pub struct ContextRefUpdate {
    pub neuron_id: NeuronId,
    pub distance: Distance,
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
    pub context_ref_updates: Vec<ContextRefUpdate>,
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
    context_ref_updates: Vec<ContextRefUpdate>,
}

/// Results from correct_errors.
struct CorrectResult {
    correction_activations: Vec<CorrectionActivation>,
    context_ref_updates: Vec<ContextRefUpdate>,
}

pub struct Neuron {
    /// Neuron ID — public because Column reads it after removing the neuron from its map.
    pub id: NeuronId,

    pattern_forget_rate: f64,
    merge_threshold: f64,
    error_mode: ErrorMode,
    error_threshold: f64,

    /// Brain-wide context_length, replicated on each neuron so recognize_patterns can implement the warmup gate:
    /// skip matching until the context window has had a chance to fill up at the start of a sequence
    context_length: u32,

    /// Per-channel action neuron IDs — ordered Vec for deterministic alternative-action
    /// exploration (neurons are tried in registration order, not hash-iteration order).
    channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,

    /// Inferences: Vec<FxHashMap<toNeuronId, ConnectionData>> indexed by distance.
    /// Distance 0 is unused (connections start at distance 1).
    connections: Vec<FxHashMap<NeuronId, ConnectionData>>,

    /// Routing table: FxHashMap<patternId, RoutingEntry> - child pattern contexts.
    routing_table: FxHashMap<NeuronId, RoutingEntry>,

    /// Inverted index: FxHashMap<neuronId, FxHashMap<distance, FxHashSet<patternId>>>.
    /// Speeds up pattern candidate lookup during recognition.
    context_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>,

    /// Context references: FxHashMap<parentId, FxHashSet<distance>>.
    /// Tracks which parent neurons reference this neuron in their children's contexts.
    context_refs: FxHashMap<NeuronId, FxHashSet<Distance>>,

    /// Per-age error rate stats: Vec<Option<WelfordState>> indexed by age.
    /// Used for dynamic error thresholds (Welford online variance).
    error_stats: Vec<Option<WelfordState>>,
}

impl Neuron {
    /// Neuron id is allocated by the Thalamus (mirrors how channel and dimension
    /// ids are allocated) and passed in at construction. channel_action_ids are used
    /// for alternative-action lookup during learning (per-channel Vec iteration).
    /// Populated by register_channel_spec() and shared across all neurons.
    ///
    /// error_mode picks the error-correction threshold function:
    ///   Static       — fixed threshold = error_threshold
    ///   Conservative — mean + σ of past per-age error rates  (learn outliers)
    ///   Neutral      — mean
    ///   Aggressive   — mean − σ (learn aggressively)
    /// For dynamic modes, error_threshold also serves as the warmup fallback until
    /// ERROR_MIN_SAMPLES observations have been recorded at that age.
    pub fn new(
        id: NeuronId,
        pattern_forget_rate: f64,
        merge_threshold: f64,
        error_mode: ErrorMode,
        error_threshold: f64,
        channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,
        context_length: u32,
    ) -> Self {
        Self {
            id,
            pattern_forget_rate,
            merge_threshold,
            error_mode,
            error_threshold,
            context_length,
            channel_action_ids,
            connections: Vec::new(),
            routing_table: FxHashMap::default(),
            context_index: FxHashMap::default(),
            context_refs: FxHashMap::default(),
            error_stats: Vec::new(),
        }
    }

    // ── Error threshold ──────────────────────────────────────────────────────

    /// Error-correction threshold for a given age, dispatched by self.error_mode.
    /// For dynamic modes, falls back to self.error_threshold during warmup
    /// (fewer than ERROR_MIN_SAMPLES observations at this age).
    pub fn get_error_threshold(&self, age: Distance) -> f64 {

        // if we're in static mode, return the error threshold from settings
        if self.error_mode == ErrorMode::Static { return self.error_threshold; }

        // if there are no stats yet, use the static threshold as fallback
        let stats = match self.error_stats.get(age as usize) {
            Some(Some(s)) => s,
            _ => return self.error_threshold,
        };

        // if there are not enough stats yet, use the static threshold as fallback
        if stats.n < ERROR_MIN_SAMPLES { return self.error_threshold; }

        // return the error threshold based on the mode
        let sigma = stats.std_dev();
        match self.error_mode {
            ErrorMode::Conservative => stats.mean + sigma,
            ErrorMode::Neutral => stats.mean,
            ErrorMode::Aggressive => stats.mean - sigma,
            ErrorMode::Static => unreachable!(),
        }
    }

    // ── Error stats ──────────────────────────────────────────────────────────

    /// Record an observed error rate for a given age (Welford online update).
    /// Called after the threshold comparison so the current sample does not
    /// influence its own decision.
    pub fn record_error(&mut self, age: Distance, error_rate: f64) {
        let idx = age as usize;

        // ensure the Vec is large enough
        if idx >= self.error_stats.len() { self.error_stats.resize(idx + 1, None); }

        let stats = self.error_stats[idx].get_or_insert_with(WelfordState::new);
        stats.update(error_rate);
    }

    /// Restoration entry point: install a fully-formed Welford bucket for a given
    /// age. Used by Column.restore_neurons to rehydrate per-(neuron, age) error stats
    /// from serialized neuron. Does not validate the stats — the caller owns correctness.
    pub fn load_error_stats(&mut self, age: Distance, n: u64, mean: f64, m2: f64) {
        let idx = age as usize;
        if idx >= self.error_stats.len() { self.error_stats.resize(idx + 1, None); }
        self.error_stats[idx] = Some(WelfordState { n, mean, m2 });
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

    /// Serialize directed connections. Flattens the nested Vec<FxHashMap<toNeuronId, conn>>
    /// into a flat array of {distance, to_neuron_id, strength, reward} entries.
    fn serialize_connections(&self) -> Vec<SerializedConnection> {
        let mut result = Vec::new();
        for (distance, targets) in self.connections.iter().enumerate() {
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

    /// Serialize the routing table (child patterns). Each child carries its activation
    /// strength, last activation frame, and flattened context entries from Context.get_entries().
    fn serialize_children(&self) -> Vec<SerializedChild> {
        let mut result = Vec::new();
        for (&pattern_id, entry) in &self.routing_table {
            result.push(SerializedChild {
                pattern_id,
                activation_strength: entry.activation_strength,
                last_activation_frame: entry.last_activation_frame,
                context: entry.context.get_entries(),
            });
        }
        result
    }

    /// Serialize context references. Converts FxHashMap<parentId, FxHashSet<distance>> into an array
    /// of {parent_id, distances} entries with distances expanded from the Set.
    fn serialize_context_refs(&self) -> Vec<SerializedContextRef> {
        let mut result = Vec::new();
        for (&parent_id, distances) in &self.context_refs {
            result.push(SerializedContextRef {
                parent_id,
                distances: distances.iter().copied().collect(),
            });
        }
        result
    }

    /// Serialize per-age Welford error stats. Converts Vec<Option<WelfordState>>
    /// into a flat array of {age, n, mean, m2} entries.
    fn serialize_error_stats(&self) -> Vec<SerializedErrorStats> {
        let mut result = Vec::new();
        for (age, stats) in self.error_stats.iter().enumerate() {
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
    pub fn has_connection(&self, distance: Distance, to_neuron_id: NeuronId) -> bool {
        match self.connections.get(distance as usize) {
            Some(distance_map) => distance_map.contains_key(&to_neuron_id),
            None => false,
        }
    }

    /// Creates a connection at distance to target neuron.
    pub fn create_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, strength: Strength, reward: Reward) {
        let idx = distance as usize;
        if idx >= self.connections.len() { self.connections.resize_with(idx + 1, FxHashMap::default); }
        self.connections[idx].insert(to_neuron_id, ConnectionData { strength, reward });
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
        let conn_reward = self.connections[distance as usize].get(&to_neuron_id).unwrap().reward;
        if conn_reward < 0.0 {
            if let Some(alt) = self.find_alternative_action(distance, channel_id, to_neuron_id) {
                self.create_connection(distance, alt, 1.0, 0.0);
            }
        }
    }

    /// Updates the connection at distance to target neuron - increments strength and updates reward.
    pub fn strengthen_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, reward: Reward) {
        let distance_map = self.connections.get_mut(distance as usize).expect("Unknown connection");

        // increment the strength of the connection
        let conn = distance_map.get_mut(&to_neuron_id).expect("Unknown connection");
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
    /// Spatial voting passes distance=0 to read co-activation predictions for the current frame;
    /// temporal voting goes through `vote(age)` which adds 1 to age.
    pub fn vote_at_distance(&self, distance: Distance) -> Vec<Vote> {

        // get connections at the distance - if there are none, no votes
        let distance_map = match self.connections.get(distance as usize) {
            Some(dm) => dm,
            None => return Vec::new(),
        };

        // create votes for all connections at the distance and return them
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
    pub fn get_child_effective_activation_strength(&self, pattern_id: NeuronId, current_frame: FrameNumber) -> f64 {
        let entry = match self.routing_table.get(&pattern_id) {
            Some(e) => e,
            None => return 0.0,
        };
        f64::max(0.0, entry.activation_strength - (current_frame - entry.last_activation_frame) as f64 * self.pattern_forget_rate)
    }

    /// Materialize lazy decay for a child pattern.
    pub fn materialize_child_strength(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) {
        let effective = self.get_child_effective_activation_strength(pattern_id, current_frame);
        if let Some(entry) = self.routing_table.get_mut(&pattern_id) {
            entry.activation_strength = effective;
        }
    }

    /// Increments activation strength for a child pattern - materializes all owner-scoped lazy decay first.
    /// Returns death frame for pattern neurons.
    pub fn strengthen_child_activation(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) -> Option<FrameNumber> {
        if !self.routing_table.contains_key(&pattern_id) { return None; }

        // update all strengths based on decay rate first
        self.materialize_child_strength(pattern_id, current_frame);

        let entry = self.routing_table.get_mut(&pattern_id).unwrap();

        // increment activation strength
        entry.activation_strength += 1.0;

        // remember when this happened for lazy decay
        entry.last_activation_frame = current_frame;

        // return death frame for pattern neurons
        Some(current_frame + (entry.activation_strength / self.pattern_forget_rate).ceil() as i64)
    }

    /// Add a child pattern to the routing table and populate its context.
    /// Returns death frame for pattern neurons.
    pub fn add_pattern(&mut self, pattern_id: NeuronId, context: &[ContextRefEntry], current_frame: FrameNumber) -> Option<FrameNumber> {
        self.add_child(pattern_id, 0.0);
        for entry in context { self.add_context(pattern_id, entry.neuron_id, entry.distance, 1.0); }
        self.strengthen_child_activation(pattern_id, current_frame)
    }

    /// Add a child pattern to the routing table.
    pub fn add_child(&mut self, pattern_id: NeuronId, initial_strength: f64) {
        if !self.routing_table.contains_key(&pattern_id) {
            self.routing_table.insert(pattern_id, RoutingEntry {
                context: Context::new(),
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

    // ── Context management ───────────────────────────────────────────────────

    /// Adds an entry to the pattern context by neuron ID.
    pub fn add_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance, strength: Strength) {

        // add the neuron to the pattern context at the given distance
        let entry = self.routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("add_context: pattern not found in routing table: {}", pattern_id));

        // when the (parent, age) dedupe reuses an existing pattern, the same context entry may be re-installed.
        // treat duplicates as a strengthen operation rather than a panic.
        if entry.context.has_key(neuron_id, distance) {
            entry.context.strengthen_neuron(neuron_id, distance);
        } else {
            entry.context.add_neuron(neuron_id, distance, strength);
        }

        // add the neuron to the context index so that we can search efficiently
        self.add_context_index(neuron_id, distance, pattern_id);
    }

    /// Adds a neuron to the context index.
    fn add_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) {
        let dist_map = self.context_index.entry(neuron_id).or_insert_with(FxHashMap::default);
        let patterns = dist_map.entry(distance).or_insert_with(FxHashSet::default);
        patterns.insert(pattern_id);
    }

    /// Removes an entry from a child pattern's context.
    /// Returns whether the neuron is no longer referenced by any child pattern in this parent
    /// (i.e., the caller should remove the contextRef on the target neuron).
    pub fn remove_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {

        // remove the neuron from the pattern context at the given distance
        let entry = self.routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("remove_context: pattern {} not found in routing table of neuron {}", pattern_id, self.id));
        entry.context.remove(neuron_id, distance);

        // remove the neuron from the context index
        self.remove_context_index(neuron_id, distance, pattern_id)
    }

    /// Remove a pattern from the context inverted index for a given neuron/distance.
    /// Returns true if no pattern references this neuron at this distance anymore (orphaned).
    pub fn remove_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) -> bool {
        let dist_map = self.context_index.get_mut(&neuron_id)
            .unwrap_or_else(|| panic!("remove_context_index: neuron {} not found in context_index of neuron {}", neuron_id, self.id));
        let patterns = dist_map.get_mut(&distance)
            .unwrap_or_else(|| panic!("remove_context_index: distance {} not found for neuron {} in context_index of neuron {}", distance, neuron_id, self.id));
        patterns.remove(&pattern_id);
        if patterns.is_empty() {
            dist_map.remove(&distance);
            if dist_map.is_empty() { self.context_index.remove(&neuron_id); }
            return true;
        }
        false
    }

    /// Remove all references to a dying neuron from this parent's children's contexts.
    /// Uses the context_index to find affected patterns in O(1) per entry.
    /// Returns pattern IDs whose context was modified (caller checks if deletable).
    pub fn remove_context_neuron(&mut self, neuron_id: NeuronId, distances: &FxHashSet<Distance>) -> FxHashSet<NeuronId> {
        let mut affected_patterns = FxHashSet::default();
        let dist_map = self.context_index.get(&neuron_id)
            .unwrap_or_else(|| panic!("remove_context_neuron: neuron {} not found in context_index of neuron {}", neuron_id, self.id));

        // collect pattern IDs before mutating (borrow checker)
        let mut ops: Vec<(Distance, Vec<NeuronId>)> = Vec::new();
        for &distance in distances {
            let patterns = dist_map.get(&distance)
                .unwrap_or_else(|| panic!("remove_context_neuron: distance {} for neuron {} not found in context_index of neuron {}", distance, neuron_id, self.id));
            ops.push((distance, patterns.iter().copied().collect()));
        }

        // remove from each pattern's context and collect affected pattern IDs
        for (distance, pattern_ids) in &ops {
            for &pattern_id in pattern_ids {
                let entry = self.routing_table.get_mut(&pattern_id)
                    .unwrap_or_else(|| panic!("remove_context_neuron: pattern {} not found in routing table of neuron {}", pattern_id, self.id));
                entry.context.remove(neuron_id, *distance);
                affected_patterns.insert(pattern_id);
            }
        }

        // clean up the index entries for these distances
        if let Some(dist_map) = self.context_index.get_mut(&neuron_id) {
            for &distance in distances {
                dist_map.remove(&distance);
            }
            if dist_map.is_empty() { self.context_index.remove(&neuron_id); }
        }

        affected_patterns
    }

    // ── Context references ───────────────────────────────────────────────────

    /// Add a context reference from another neuron to this neuron.
    /// Called when this neuron is added to another neuron's context.
    pub fn add_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        self.context_refs.entry(referencing_neuron_id).or_insert_with(FxHashSet::default).insert(distance);
    }

    /// Remove a context reference from another neuron to this neuron.
    /// Called when this neuron is removed from another neuron's context.
    pub fn remove_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        let distances = match self.context_refs.get_mut(&referencing_neuron_id) {
            Some(d) => d,
            None => return,
        };
        distances.remove(&distance);
        if distances.is_empty() { self.context_refs.remove(&referencing_neuron_id); }
    }

    /// Apply a batch of context-reference updates targeting this neuron.
    /// One call per target neuron per frame (callers aggregate by target).
    /// All updates are Adds — Remove is no longer produced upstream now that
    /// refine_context is disabled. Cascade-delete of context refs still flows
    /// directly via Neuron::remove_context_ref from Column::remove_context_ref_op.
    pub fn apply_context_ref_updates(&mut self, updates: &[ContextRefUpdate]) {
        for update in updates {
            self.add_context_ref(update.parent_id, update.distance);
        }
    }

    /// Get context refs (for delete cascade).
    pub fn get_context_refs(&self) -> &FxHashMap<NeuronId, FxHashSet<Distance>> {
        &self.context_refs
    }

    /// Read-only access to the routing table (for Column snapshot/death-frame/delete ops).
    pub fn get_routing_table(&self) -> &FxHashMap<NeuronId, RoutingEntry> {
        &self.routing_table
    }

    /// Inspection: flatten this neuron's outgoing connections into
    /// (distance, target_neuron_id, strength, reward) tuples. Distance is
    /// the slot index in the connections Vec. Used by the diagnostic API
    /// to dump connection state for tipping-point analysis.
    pub fn get_connections(&self) -> Vec<(Distance, NeuronId, f64, f64)> {
        let mut out = Vec::new();
        for (idx, dist_map) in self.connections.iter().enumerate() {
            let dist = idx as Distance;
            for (&target_id, conn) in dist_map {
                out.push((dist, target_id, conn.strength, conn.reward));
            }
        }
        out
    }

    /// Mutable access to the routing table (for Column restore — sets last_activation_frame).
    pub fn get_routing_table_mut(&mut self) -> &mut FxHashMap<NeuronId, RoutingEntry> {
        &mut self.routing_table
    }

    /// Remove a child pattern from the routing table (for Column delete cascade).
    pub fn remove_routing_entry(&mut self, pattern_id: NeuronId) {
        self.routing_table.remove(&pattern_id);
    }

    /// Check if a context key exists for a child pattern (for Column delete cascade).
    pub fn has_context_key(&self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {
        self.routing_table.get(&pattern_id)
            .map_or(false, |e| e.context.has_key(neuron_id, distance))
    }

    /// Check if a neuron+distance exists in the context index (for Column purge).
    pub fn has_context_index_entry(&self, neuron_id: NeuronId, distance: Distance) -> bool {
        self.context_index.get(&neuron_id)
            .map_or(false, |dist_map| dist_map.contains_key(&distance))
    }

    /// Pattern forget rate (for Column death-frame calculation).
    pub fn get_pattern_forget_rate(&self) -> f64 {
        self.pattern_forget_rate
    }

    // ── Frame processing ─────────────────────────────────────────────────────

    /// Process a frame for this neuron: derive per-age tasks, learn connections, match patterns,
    /// install pre-created error-correction patterns, and cast votes. One call per active neuron
    /// per frame. Matching is skipped when level_context is None/empty. Votes are cast for each
    /// eligible voting age unless suppressed by a this-frame match (activate=true) or
    /// error-correction activation at the same age.
    pub fn process_frame(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&Context>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        actives: &[ActiveNeuron],
        current_frame: FrameNumber,
        corrections: &[Correction],
        error_feedback: &[ErrorFeedback],
        learning: bool,
        phase: Phase,
    ) -> ProcessFrameResult {

        // Fold prior-frame error feedback into per-age accuracy stats first.
        // That way the threshold attached to this frame's votes (computed in generate_votes) reflects the latest sample.
        // Non-learning mode skips this because accuracy stats are connection-substrate state.
        if learning {
            for fb in error_feedback { self.record_error(fb.age, fb.error_rate); }
        }
        let mut timings = NeuronOpTimings::default();

        let should_learn = learning && !new_error_pattern_ids.contains(&self.id);

        // Temporal learns connections[age>0] which the vote step (connections[age+1]) doesn't read,
        // so the temporal flow runs learn first. Spatial learns connections[0] which IS what the vote
        // step reads, so spatial must run learn AFTER generate_votes — otherwise the just-strengthened
        // edges to this frame's novel co-actives get echoed straight back in the vote, and the spatial
        // error pass never sees a mismatch (see spatial-processing.md §4.5).
        if phase == Phase::Temporal && should_learn {
            let t = std::time::Instant::now();
            self.learn_connections(age_states, actives, phase);
            timings.learn_connections = t.elapsed().as_secs_f64();
        }

        // match patterns if we have context and eligible ages
        let t = std::time::Instant::now();
        let RecognizeResult { matches, context_ref_updates: match_refs } = self.recognize_patterns(age_states, memory_depth, level_context, new_error_pattern_ids, current_frame, learning, &mut timings, phase);
        timings.recognize_patterns = t.elapsed().as_secs_f64();

        // install pre-created error-correction patterns as children and emit their contextRef adds
        let t = std::time::Instant::now();
        let CorrectResult { correction_activations, context_ref_updates: correction_refs } = self.correct_errors(corrections, current_frame);
        timings.correct_errors = t.elapsed().as_secs_f64();

        // cast votes for each eligible age, suppressing any ages that activated a pattern.
        // For spatial this reads connections[0] BEFORE the d=0 learn step below modifies it.
        let t = std::time::Instant::now();
        let votes = self.generate_votes(age_states, memory_depth, level_context, &matches, &correction_activations, phase);
        timings.generate_votes = t.elapsed().as_secs_f64();

        // Spatial learn runs last so the vote above sees the prior-frame connections[0].
        if phase == Phase::Spatial && should_learn {
            let t = std::time::Instant::now();
            self.learn_connections(age_states, actives, phase);
            timings.learn_connections = t.elapsed().as_secs_f64();
        }

        // return frame processing results
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
    fn recognize_patterns(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&Context>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        current_frame: FrameNumber,
        learning: bool,
        timings: &mut NeuronOpTimings,
        phase: Phase,
    ) -> RecognizeResult {
        let mut matches = Vec::new();
        let context_ref_updates = Vec::new();

        // Warmup gate (temporal only): skip pattern recognition until the context window has had a chance to fill up.
        // Without this, patterns whose stored contexts include entries at distances > current_frame are unfairly penalized.
        // Spatial matching doesn't depend on past frames — co-activation context is fully observable on frame 1 — so no warmup needed.
        if phase == Phase::Temporal && (current_frame as u32) < self.context_length {
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
            // Temporal skips the oldest age (no future context to match). Spatial matches at age=0
            // against the current frame's co-activation set; the "oldest age" guard does not apply.
            if state.activated_pattern_id.is_some() { continue; }
            if phase == Phase::Temporal && age == memory_depth - 1 { continue; }

            // active ages are processed in ascending order (most recent first). The first age that
            // produces a match at that age is refined and preserved. More recent ages tend to have
            // the richest available context, so they are processed first.
            let best = match self.find_best_pattern_match_at_age(ctx, age, new_error_pattern_ids, current_frame, timings, phase) {
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

    /// Find the best matching pattern for a specific active age.
    fn find_best_pattern_match_at_age(&self, observed: &Context, age: Distance, exclude_ids: &FxHashSet<NeuronId>, current_frame: FrameNumber, timings: &mut NeuronOpTimings, phase: Phase) -> Option<PartialMatch> {
        let mut best: Option<PartialMatch> = None;

        // Use the inverted index to narrow the search to child patterns that share at least one
        // exact neuron/distance entry with the observed context at this active age.
        let t = std::time::Instant::now();
        let candidate_ids = self.get_pattern_candidates_at_age(observed, age, phase);
        timings.recognize_candidate_search += t.elapsed().as_secs_f64();
        if candidate_ids.is_empty() { return None; }

        let eval_start = std::time::Instant::now();
        let candidate_count = candidate_ids.len();
        // go through the candidate patterns and find the best match
        for pattern_id in candidate_ids {

            // check if pattern is still alive (skip functionally dead patterns)
            if self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0 { continue; }

            // get the context of the pattern
            let entry = self.routing_table.get(&pattern_id)
                .unwrap_or_else(|| panic!("Cannot find context for pattern: {}", pattern_id));

            // context.match_observed() handles the full scoring and threshold check;
            // the index only decides which child patterns are worth evaluating.
            // exclude_ids masks out brand-new neurons so they don't count as "novel" misses.
            let m = match entry.context.match_observed(observed, age, self.merge_threshold, Some(exclude_ids)) {
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
    fn get_pattern_candidates_at_age(&self, observed: &Context, age: Distance, phase: Phase) -> FxHashSet<NeuronId> {
        let mut candidates = FxHashSet::default();

        // Temporal context entries are at strictly positive relative distances — context neurons must be
        // older than the parent neuron itself. Spatial context entries sit at distance 0 — co-activation
        // with the parent on the same frame.
        let min_distance: i64 = match phase { Phase::Temporal => 1, Phase::Spatial => 0 };

        for (&neuron_id, distance_map) in observed.entries() {

            // First narrow by exact context neuron ID. If this neuron does not appear in the index,
            // no child pattern references it anywhere in this parent's routing table.
            let indexed_distances = match self.context_index.get(&neuron_id) {
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
            let death_frame = self.add_pattern(correction.pattern_id, &correction.context_entries, current_frame);

            // add the pattern to the activations to be returned with its death frame
            correction_activations.push(CorrectionActivation { pattern_id: correction.pattern_id, age: correction.age, death_frame });

            // also update the context reference updates to be returned for the new patterns
            for entry in &correction.context_entries {
                context_ref_updates.push(ContextRefUpdate { neuron_id: entry.neuron_id, distance: entry.distance, parent_id: 0 });
            }
        }
        CorrectResult { correction_activations, context_ref_updates }
    }

    // ── Vote generation ──────────────────────────────────────────────────────

    /// Cast votes for each eligible age, suppressing any age that activated a pattern this
    /// frame via either a recognition match (activate=true) or an error-correction install.
    /// Eligible ages: non-activated and younger than the oldest sliding-window slot. Runs for
    /// new error patterns too so their state.context gets populated for next-frame corrections.
    /// The per-age context is reshaped from level_context locally — context_by_age is a pure reshape
    /// (no extra info), so it's derived here instead of being shipped over the wire.
    fn generate_votes(
        &self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&Context>,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
        phase: Phase,
    ) -> Vec<AgeVotes> {

        // determine the suppressed ages based on recognized patterns and error creations
        let mut suppressed_ages = FxHashSet::default();
        for m in matches { if m.activate { suppressed_ages.insert(m.age); } }
        for c in correction_activations { suppressed_ages.insert(c.age); }

        // pre-bucket level_context by voting age so the loop below picks each per-age
        // context up with a single indexed lookup
        let context_by_age = self.derive_context_by_age(level_context);

        // cast votes for each eligible, non-suppressed age
        let mut votes = Vec::new();
        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            let state = &age_states[&age];
            if state.activated_pattern_id.is_some() { continue; }
            // Temporal skips the oldest age (no future frame to vote toward). Spatial votes at d=0
            // for the current frame, so the oldest-age guard doesn't apply.
            if phase == Phase::Temporal && age >= memory_depth - 1 { continue; }
            if suppressed_ages.contains(&age) { continue; }

            // Temporal votes predict the next frame via connections[age+1]; spatial votes predict
            // the same frame's co-activations via connections[0].
            let cast_votes = match phase {
                Phase::Temporal => self.vote(age),
                Phase::Spatial => self.vote_at_distance(0),
            };
            votes.push(AgeVotes {
                age,
                votes: cast_votes,
                context: context_by_age.get(age as usize).cloned().unwrap_or_default(),
                threshold: self.get_error_threshold(age),
            });
        }
        votes
    }

    /// Reshape level_context into per-voting-age buckets. For each entry at context-age
    /// `ctx_age`, emits it into every voting_age < ctx_age with distance = ctx_age - voting_age.
    /// Indexed by voting age so generate_votes picks up each per-age context with a single
    /// lookup — no per-age scan, no `ctx_age > age` branch. Pure reshape kept inside the
    /// neuron to save per-frame MPI traffic.
    fn derive_context_by_age(&self, level_context: Option<&Context>) -> Vec<Vec<ContextRefEntry>> {
        let mut context_by_age: Vec<Vec<ContextRefEntry>> = Vec::new();
        let ctx = match level_context {
            Some(c) => c,
            None => return context_by_age,
        };
        for (&neuron_id, distance_map) in ctx.entries() {
            for &ctx_age in distance_map.keys() {
                for a in 0..ctx_age {
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
    fn learn_connections(&mut self, age_states: &FxHashMap<Distance, AgeState>, actives: &[ActiveNeuron], phase: Phase) {
        match phase {

            // Temporal: strengthen connections[age] for every active age>0. A neuron active at age=k
            // learns a k-distance prediction toward each current actives (which are at age=0 in `actives`).
            Phase::Temporal => {
                let ages: Vec<Distance> = age_states.keys().copied().collect();
                for age in ages {
                    if age == 0 { continue; }
                    for active in actives {
                        self.upsert_connection(age, active.id, active.channel_id, active.reward);
                    }
                }
            }

            // Spatial: strengthen connections[0] toward every co-active neuron in the spatial fired set.
            // This neuron only fires in spatial at age=0; the upsert builds the d=0 co-activation graph.
            // Self-connections are skipped — a neuron isn't its own co-activation partner.
            Phase::Spatial => {
                for active in actives {
                    if active.id == self.id { continue; }
                    self.upsert_connection(0, active.id, active.channel_id, active.reward);
                }
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
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
    pub context: Vec<ContextEntry>,
}

#[derive(Debug, Clone)]
pub struct SerializedContextRef {
    pub parent_id: NeuronId,
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
        Neuron::new(id, 0.01, 0.9, ErrorMode::Static, 0.3, FxHashMap::default(), 10)
    }

    fn make_neuron_with_actions(id: NeuronId, channel_id: ChannelId, action_ids: Vec<NeuronId>) -> Neuron {
        let mut channel_actions = FxHashMap::default();
        channel_actions.insert(channel_id, action_ids);
        Neuron::new(id, 0.01, 0.9, ErrorMode::Static, 0.3, channel_actions, 10)
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
        let conn = &n.connections[1][&10];
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
        let death_frame = n.add_pattern(100, &context, 10);
        assert!(death_frame.is_some());

        // pattern should be in routing table with 2 context entries
        assert_eq!(n.routing_table[&100].context.size(), 2);

        // context index should have entries
        assert!(n.context_index.contains_key(&10));
        assert!(n.context_index.contains_key(&11));
    }

    #[test]
    fn test_lazy_decay() {
        let mut n = make_neuron(1);
        n.pattern_forget_rate = 0.1;
        n.add_child(100, 0.0);
        n.strengthen_child_activation(100, 0); // strength=1, last_frame=0

        // at frame 5: effective = 1 - 5*0.1 = 0.5
        assert!((n.get_child_effective_activation_strength(100, 5) - 0.5).abs() < 1e-10);

        // at frame 10: effective = 1 - 10*0.1 = 0.0
        assert!(n.get_child_effective_activation_strength(100, 10) <= 0.0);
        assert!(n.can_delete_child(100, 10));
    }

    #[test]
    fn test_error_threshold_static() {
        let n = make_neuron(1);
        assert_eq!(n.get_error_threshold(0), 0.3);
        assert_eq!(n.get_error_threshold(5), 0.3);
    }

    #[test]
    fn test_error_threshold_dynamic_warmup() {
        let mut n = Neuron::new(1, 0.01, 0.9, ErrorMode::Neutral, 0.3, FxHashMap::default(), 10);
        // fewer than ERROR_MIN_SAMPLES → falls back to error_threshold
        n.record_error(0, 0.5);
        n.record_error(0, 0.5);
        assert_eq!(n.get_error_threshold(0), 0.3); // still warmup

        // after 3 samples → uses mean
        n.record_error(0, 0.5);
        assert!((n.get_error_threshold(0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_context_refs() {
        let mut n = make_neuron(1);
        n.add_context_ref(50, 2);
        n.add_context_ref(50, 3);
        n.add_context_ref(60, 1);

        assert_eq!(n.context_refs.len(), 2);
        assert_eq!(n.context_refs[&50].len(), 2);

        n.remove_context_ref(50, 2);
        assert_eq!(n.context_refs[&50].len(), 1);

        n.remove_context_ref(50, 3);
        assert!(!n.context_refs.contains_key(&50)); // auto-cleaned
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
        n.add_child(100, 0.0);
        n.add_context(100, 20, 1, 1.0);
        n.add_context_ref(50, 2);
        n.record_error(0, 0.4);
        n.record_error(0, 0.6);
        n.record_error(0, 0.5);

        let s = n.serialize();
        assert_eq!(s.id, 1);
        assert_eq!(s.connections.len(), 2);
        assert_eq!(s.children.len(), 1);
        assert_eq!(s.children[0].context.len(), 1);
        assert_eq!(s.context_refs.len(), 1);
        assert_eq!(s.error_stats.len(), 1);
    }
}
