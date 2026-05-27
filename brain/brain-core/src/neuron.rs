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
    pub score: f64,
    pub common: Vec<ContextEntry>,
    pub missing: Vec<ContextEntry>,
    pub novel: Vec<ContextEntry>,
    pub removed_refs: Vec<ContextRefEntry>,
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
    pub update_type: ContextRefUpdateType,
    pub neuron_id: NeuronId,
    pub distance: Distance,
    pub parent_id: NeuronId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContextRefUpdateType {
    Add,
    Remove,
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

    /// Brain-wide context_length, replicated on each neuron so recognize_patterns
    /// can implement the warmup gate (skip matching until the context window
    /// has had a chance to fill up at the start of a sequence).
    context_length: u32,

    /// Per-channel action neuron IDs — ordered Vec for deterministic alternative-action
    /// exploration (neurons are tried in registration order, not hash-iteration order).
    channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,

    /// Flat union of all action neuron IDs across channels — used for O(1) is_action_neuron checks.
    action_ids: FxHashSet<NeuronId>,

    /// Optional fixed learning rate for action connection reward updates.
    /// None = dynamic alpha (1/strength), Some(v) = static alpha.
    action_alpha: Option<f64>,

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
    /// action_ids is the flat union of all action neuron ids across channels, used
    /// for O(1) is_action_neuron checks during connection learning. Both are populated
    /// by register_channel_spec() and shared across all neurons.
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
        action_ids: FxHashSet<NeuronId>,
        action_alpha: Option<f64>,
        context_length: u32,
    ) -> Self {
        Self {
            id,
            pattern_forget_rate,
            merge_threshold,
            error_mode,
            error_threshold,
            channel_action_ids,
            action_ids,
            action_alpha,
            context_length,
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

    /// Supervised action learning. Updates ONLY the X→correct_id connection at the given distance:
    pub fn learn_supervised_action(&mut self, distance: Distance, correct_id: NeuronId, _channel_id: ChannelId, reward: Reward) {
        let idx = distance as usize;
        if !self.has_connection(distance, correct_id) {
            self.create_connection(distance, correct_id, 0.0, 0.0);
        }

        // strength += 1 and reward += `reward` (additive accumulation, not smoothed).
        // Other action connections on this voter are untouched.
        let conn = self.connections[idx].get_mut(&correct_id).unwrap();
        conn.strength += 1.0;

        // Additive accumulation preserves the frequency signal: a voter wired to action A twice and
        // to action B once stores (A.strength=2, A.reward=2, B.strength=1, B.reward=1). At consensus
        // the action score is sum(strength × reward) / sum(strength) across voters, so A contributes
        // weight 2·2=4 and B contributes 1·1=1 — A wins 2-to-1, matching how event strengths
        // naturally weight by frequency. The earlier smoothed form (alpha = 1/strength) collapsed
        // reward to 1 regardless of frequency, hiding the signal.
        conn.reward += reward;
    }

    /// Creates or updates the connection at distance to a target neuron
    pub fn upsert_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, channel_id: ChannelId, reward: Reward) {
        if self.has_connection(distance, to_neuron_id) { self.strengthen_connection(distance, to_neuron_id, reward); }
        else { self.create_connection(distance, to_neuron_id, 1.0, reward); }

        // for actions with negative (smoothed) rewards, save an alternative with neutral reward - we'll try it next time
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

        // update reward: static alpha for action neurons (if configured),
        // dynamic alpha (1/strength) for events and when no static alpha is set
        let alpha = if self.action_ids.contains(&to_neuron_id) {
            self.action_alpha.unwrap_or(1.0 / conn.strength)
        } else {
            1.0 / conn.strength
        };
        conn.reward = alpha * reward + (1.0 - alpha) * conn.reward;
    }

    /// Weaken a connection via negative reinforcement (prediction didn't occur).
    /// Deletes the connection if strength drops to zero or below.
    pub fn weaken_connection(&mut self, distance: Distance, to_neuron_id: NeuronId) {
        let idx = distance as usize;
        let distance_map = match self.connections.get_mut(idx) {
            Some(dm) => dm,
            None => return,
        };
        let conn = match distance_map.get_mut(&to_neuron_id) {
            Some(c) => c,
            None => return,
        };
        conn.strength -= 1.0;
        if conn.strength <= 0.0 { self.delete_connection(distance, to_neuron_id); }
    }

    /// Delete connection at distance to target neuron.
    pub fn delete_connection(&mut self, distance: Distance, to_neuron_id: NeuronId) {
        let idx = distance as usize;
        if let Some(distance_map) = self.connections.get_mut(idx) {
            distance_map.remove(&to_neuron_id);
            // Note: we don't shrink the Vec even if distance_map is empty — sparse slots are cheap
        }
    }

    // ── Voting ───────────────────────────────────────────────────────────────

    /// Returns votes from this neuron at a specific age.
    pub fn vote(&self, age: Distance) -> Vec<Vote> {

        // use connections of distance one more than the age to get the inferences for the next frame
        let distance = age + 1;

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

    /// Remove a child pattern from this neuron's routing table.
    /// Called by thalamus when deleting a child pattern neuron.
    pub fn remove_child(&mut self, pattern_id: NeuronId) {

        // clean up both context and context index for all context entries of this pattern
        let entry = self.routing_table.get(&pattern_id)
            .unwrap_or_else(|| panic!("remove_child: pattern {} not found in routing table of neuron {}", pattern_id, self.id));
        let entries = entry.context.get_entries();
        for ctx_entry in &entries {
            self.remove_context(pattern_id, ctx_entry.neuron_id, ctx_entry.distance);
        }

        // remove the pattern from the routing table
        self.routing_table.remove(&pattern_id);
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

        // Idempotent: when the (parent, age) dedupe reuses an existing
        // pattern, the same context entry may be re-installed. Treat duplicate
        // entries as a strengthen rather than a panic.
        if entry.context.find(neuron_id, distance).is_some() {
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
    pub fn apply_context_ref_updates(&mut self, updates: &[ContextRefUpdate]) {
        for update in updates {
            if update.update_type == ContextRefUpdateType::Add { self.add_context_ref(update.parent_id, update.distance); }
            else { self.remove_context_ref(update.parent_id, update.distance); }
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
    pub fn dump_connections(&self) -> Vec<(Distance, NeuronId, f64, f64)> {
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
    ) -> ProcessFrameResult {

        // fold prior-frame error feedback into per-age stats first so the threshold attached
        // to this frame's votes (computed in generate_votes) reflects the latest sample
        for fb in error_feedback { self.record_error(fb.age, fb.error_rate); }

        // learn connections across all active ages (age=0 skipped internally)
        // but, if this neuron was just created, it was already created with the current connections - no need to learn again
        if learning && !new_error_pattern_ids.contains(&self.id) { self.learn_connections(age_states, actives); }

        // match patterns if we have context and eligible ages
        let RecognizeResult { matches, context_ref_updates: match_refs } = self.recognize_patterns(age_states, memory_depth, level_context, new_error_pattern_ids, current_frame, learning);

        // install pre-created error-correction patterns as children and emit their contextRef adds
        let CorrectResult { correction_activations, context_ref_updates: correction_refs } = self.correct_errors(corrections, current_frame);

        // cast votes for each eligible age, suppressing any ages that activated a pattern
        let votes = self.generate_votes(age_states, memory_depth, level_context, &matches, &correction_activations);

        // return frame processing results
        let mut context_ref_updates = match_refs;
        context_ref_updates.extend(correction_refs);
        ProcessFrameResult { matches, correction_activations, context_ref_updates, votes }
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
    ) -> RecognizeResult {
        let mut matches = Vec::new();
        let mut context_ref_updates = Vec::new();

        // Warmup gate: skip pattern recognition until the context window has
        // had a chance to fill up at the start of a sequence. Without this,
        // patterns whose stored contexts include entries at distances >
        // current_frame are unfairly penalized — their unreachable entries
        // count as "missing", letting smaller new patterns win matches by
        // default and displacing established ones (the root cause of the
        // training-degrades-accuracy regression).
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
            if state.activated_pattern_id.is_some() || age == memory_depth - 1 { continue; }

            // active ages are processed in ascending order (most recent first). The first age that
            // produces a match at that age is refined and preserved. More recent ages tend to have
            // the richest available context, so they are processed first.
            let best = match self.find_best_pattern_match_at_age(ctx, age, new_error_pattern_ids, current_frame) {
                Some(b) => b,
                None => continue, // try older age if there is a match
            };

            // refine_context is intentionally disabled here — refining the matched pattern's
            // stored context mid-training makes recognition non-reproducible (training-time
            // recognition sees "in-progress" patterns; later replays see fully-refined ones,
            // so trajectories diverge). With refinement off, every match uses the pattern
            // exactly as it was created/installed.
            let removed_refs: Vec<ContextRefEntry> = Vec::new();

            // activate the matched pattern if it was not elected to be activated already
            let activate = !activated_pattern_ids.contains(&best.pattern_id);
            activated_pattern_ids.insert(best.pattern_id);

            // strengthen child activation automatically here — also gated by learning so
            // inference-only replays don't bump activation strength on every match.
            let death_frame = if activate && learning { self.strengthen_child_activation(best.pattern_id, current_frame) } else { None };

            // include the best match for the age in the results
            matches.push(PatternMatch {
                pattern_id: best.pattern_id,
                age: best.age,
                score: best.score,
                common: best.common,
                missing: best.missing,
                novel: best.novel,
                removed_refs,
                activate,
                death_frame,
            });
        }
        RecognizeResult { matches, context_ref_updates }
    }

    /// Find the best matching pattern for a specific active age.
    fn find_best_pattern_match_at_age(&self, observed: &Context, age: Distance, exclude_ids: &FxHashSet<NeuronId>, current_frame: FrameNumber) -> Option<PartialMatch> {
        let mut best: Option<PartialMatch> = None;

        // Use the inverted index to narrow the search to child patterns that share at least one
        // exact neuron/distance entry with the observed context at this active age.
        let candidate_ids = self.get_pattern_candidates_at_age(observed, age);
        if candidate_ids.is_empty() { return None; }

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
                common: m.common,
                missing: m.missing,
                novel: m.novel,
            });
        }
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
    fn get_pattern_candidates_at_age(&self, observed: &Context, age: Distance) -> FxHashSet<NeuronId> {
        let mut candidates = FxHashSet::default();
        for (&neuron_id, distance_map) in observed.entries() {

            // First narrow by exact context neuron ID. If this neuron does not appear in the index,
            // no child pattern references it anywhere in this parent's routing table.
            let indexed_distances = match self.context_index.get(&neuron_id) {
                Some(d) => d,
                None => continue,
            };

            for &absolute_distance in distance_map.keys() {

                // Convert the observed absolute age into the pattern-relative distance used by the
                // routing table and inverted index. Distances < 1 are not valid pattern context entries
                // because context neurons must be older than the parent neuron itself.
                let pattern_distance = absolute_distance as i64 - age as i64;
                if pattern_distance < 1 { continue; }
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

    // ── Context refinement ───────────────────────────────────────────────────

    /// Refine the context of a pattern neuron based on the observed context.
    /// Strengthens common, adds novel, weakens/deletes missing.
    /// Returns list of context refs that should be removed (caller delivers to target neurons).
    fn refine_context(&mut self, pattern_id: NeuronId, common: &[ContextEntry], novel: &[ContextEntry], missing: &[ContextEntry]) -> Vec<ContextRefEntry> {

        // get the routing table entry for the pattern
        let entry = self.routing_table.get_mut(&pattern_id).expect("pattern not found in routing table.");

        // strengthen common context neurons
        for item in common { entry.context.strengthen_neuron(item.neuron_id, item.distance); }

        // add novel context neurons
        // (must drop mutable borrow of routing_table before calling add_context which also borrows it)
        let novel_entries: Vec<(NeuronId, Distance)> = novel.iter().map(|item| (item.neuron_id, item.distance)).collect();
        for (neuron_id, distance) in novel_entries { self.add_context(pattern_id, neuron_id, distance, 1.0); }

        // weaken missing — weaken_neuron auto-deletes at zero strength
        // collect deletions first, then update context index in a separate pass (borrow checker)
        let mut deleted_entries: Vec<(NeuronId, Distance)> = Vec::new();
        let entry = self.routing_table.get_mut(&pattern_id).expect("pattern not found in routing table.");
        for item in missing {
            let was_deleted = entry.context.weaken_neuron(item.neuron_id, item.distance);
            if was_deleted { deleted_entries.push((item.neuron_id, item.distance)); }
        }

        let mut removed_refs = Vec::new();
        for (neuron_id, distance) in deleted_entries {
            if self.remove_context_index(neuron_id, distance, pattern_id) {
                removed_refs.push(ContextRefEntry { neuron_id, distance });
            }
        }
        removed_refs
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
                context_ref_updates.push(ContextRefUpdate { update_type: ContextRefUpdateType::Add, neuron_id: entry.neuron_id, distance: entry.distance, parent_id: 0 });
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
            if state.activated_pattern_id.is_some() || age >= memory_depth - 1 { continue; }
            if suppressed_ages.contains(&age) { continue; }
            votes.push(AgeVotes {
                age,
                votes: self.vote(age),
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
    /// upsert a connection per active (create-or-strengthen + alt-action).
    /// Rewards are pre-resolved thalamus-side (0 for events, observed value for actions).
    /// Skipped entirely for new error patterns (they were just created this frame and have
    /// nothing yet to reinforce).
    ///
    /// Note: connections are no longer negatively reinforced. Previously, predictions at this
    /// distance that didn't occur were weakened (with eventual deletion at strength=0). This
    /// produced periodic discrete-death bursts during multi-image training (a connection drifts
    /// down by ~1/episode until it finally hits 0, then dies abruptly, shifting the neuron's
    /// vote profile and triggering a cascade of error-pattern creation). For deterministic
    /// memorization scenarios we want predictions that ever occurred to remain available;
    /// non-occurrences should not erase them. Mirrors the action-connection behaviour, which
    /// was already kept-only-strengthen.
    fn learn_connections(&mut self, age_states: &FxHashMap<Distance, AgeState>, actives: &[ActiveNeuron]) {
        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {

            // skip age 0 - connection learning only applies to context neurons (age > 0)
            if age == 0 { continue; }

            // learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
            for active in actives {
                self.upsert_connection(age, active.id, active.channel_id, active.reward);
            }
        }
    }

    /// Returns neuron IDs at a distance whose inferences did not occur.
    fn get_neurons_not_found(&self, distance: Distance, active_neuron_ids: &FxHashSet<NeuronId>) -> Vec<NeuronId> {
        let distance_map = match self.connections.get(distance as usize) {
            Some(dm) => dm,
            None => return Vec::new(),
        };
        let mut not_found = Vec::new();
        for &to_neuron_id in distance_map.keys() {
            if !active_neuron_ids.contains(&to_neuron_id) { not_found.push(to_neuron_id); }
        }
        not_found
    }

    /// Check if a neuron id is an action neuron in any channel. Uses the flat action_ids
    /// set maintained by Thalamus alongside channel_action_ids — both are kept in lockstep
    /// at registration time, so action detection stays aligned with alt-action lookup.
    fn is_action_neuron(&self, neuron_id: NeuronId) -> bool {
        self.action_ids.contains(&neuron_id)
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
/// the full PatternMatch (with removed_refs, activate, death_frame) is assembled.
struct PartialMatch {
    pattern_id: NeuronId,
    age: Distance,
    score: f64,
    common: Vec<ContextEntry>,
    missing: Vec<ContextEntry>,
    novel: Vec<ContextEntry>,
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
        Neuron::new(id, 0.01, 0.9, ErrorMode::Static, 0.3, FxHashMap::default(), FxHashSet::default(), None, 10)
    }

    fn make_neuron_with_actions(id: NeuronId, channel_id: ChannelId, action_ids: Vec<NeuronId>) -> Neuron {
        let action_set: FxHashSet<NeuronId> = action_ids.iter().copied().collect();
        let mut channel_actions = FxHashMap::default();
        channel_actions.insert(channel_id, action_ids);
        Neuron::new(id, 0.01, 0.9, ErrorMode::Static, 0.3, channel_actions, action_set, None, 10)
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
    fn test_weaken_connection_deletes_at_zero() {
        let mut n = make_neuron(1);
        n.create_connection(1, 10, 1.0, 0.0);
        n.weaken_connection(1, 10);
        assert!(!n.has_connection(1, 10));
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
    fn test_remove_child_cleans_up() {
        let mut n = make_neuron(1);
        let context = vec![
            ContextRefEntry { neuron_id: 10, distance: 1 },
        ];
        n.add_pattern(100, &context, 10);
        n.remove_child(100);

        // routing table and context index should be clean
        assert!(!n.routing_table.contains_key(&100));
        assert!(!n.context_index.contains_key(&10));
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
        let mut n = Neuron::new(1, 0.01, 0.9, ErrorMode::Neutral, 0.3, FxHashMap::default(), FxHashSet::default(), None, 10);
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
