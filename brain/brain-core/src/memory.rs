/// Memory — manages the temporal sliding window of active and inferred neurons.
///
/// **Temporal state** is keyed by activation frame number; age is a derived quantity
/// (`frame_number - activation_frame`) computed on read. `advance_temporal_window()` doesn't
/// have to migrate any per-neuron state Maps each frame — it just evicts whatever frame fell
/// off the back of the window. Public API for temporal (depth, get_level_neurons,
/// get_base_level, get_active_voter_ages, …) still speaks in ages.
///
/// **Spatial state** is age-free. Spatial is a same-frame computation: connections[0] predict
/// co-activations within the current frame, and the spatial index is wiped at the top of
/// every frame via `reset_spatial`. There is no spatial "history" to look back at, so spatial
/// activations don't touch `age_index`, `neuron_states`, or any frame-keyed storage. They
/// land directly in `spatial_level_index`, which is just `Level → set of neuron ids`.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::thalamus::LevelAgeState;
use crate::types::{ChannelId, Coordinate, Distance, FrameNumber, Level, NeuronId};

/// A single inferred neuron — winner from the voting consensus.
/// Carries the fields that buildFrame needs on the next frame:
/// strength for all neurons, reward for actions, probability for events.
#[derive(Debug, Clone)]
pub struct InferredNeuron {
    pub neuron_id: NeuronId,
    pub coordinate: Coordinate,
    pub channel_id: ChannelId,
    pub strength: f64,
    pub reward: f64,
    pub probability: f64,
}

pub struct Memory {
    /// Number of frames a base neuron stays active.
    context_length: u32,

    /// Current frame counter, advanced once per age() call. Activations are stored
    /// keyed by the frame they happened on; age = frame_number - activation_frame.
    frame_number: FrameNumber,

    /// Active neuron states by activation frame: neuron_id → frame → state.
    /// State objects are created at activation and never copied — external mutations
    /// (thalamus.process_level sets votes/context/threshold inline) stay attached for the
    /// life of the entry without per-frame rebuild.
    neuron_states: FxHashMap<NeuronId, FxHashMap<FrameNumber, LevelAgeState>>,

    /// Index for fast age queries — keyed by activation frame so eviction is a single
    /// delete-by-frame instead of a per-neuron shift.
    /// frame → set of neuron ids activated at that frame.
    age_index: FxHashMap<FrameNumber, FxHashSet<NeuronId>>,

    /// Spatial level index — `level → set of neuron ids`. Age-free.
    /// Wiped at the top of every frame by `reset_spatial`. Spatial activations land here and
    /// nowhere else — no age_index, no neuron_states. The spatial sweep reads this back via
    /// `get_spatial_level_neurons(level)` which fabricates a default per-neuron state
    /// at age=0 on demand (spatial never persists per-neuron state).
    spatial_level_index: FxHashMap<Level, FxHashSet<NeuronId>>,

    /// Temporal level index — `level → frame → set of neuron ids`.
    /// Frame keying lets the temporal sweep recover age (`age = frame_number - frame`) without
    /// migrating per-neuron Maps each frame. Eviction in `advance_temporal_window` is a single
    /// delete-by-frame.
    temporal_level_index: FxHashMap<Level, FxHashMap<FrameNumber, FxHashSet<NeuronId>>>,

    /// Current frame winning inferences.
    inferred_neurons: Vec<InferredNeuron>,

    /// Debug flag.
    debug: bool,
}

impl Memory {
    pub fn new(debug: bool, context_length: u32) -> Self {
        Self {
            context_length,
            frame_number: 0,
            neuron_states: FxHashMap::default(),
            age_index: FxHashMap::default(),
            spatial_level_index: FxHashMap::default(),
            temporal_level_index: FxHashMap::default(),
            inferred_neurons: Vec::new(),
            debug,
        }
    }

    /// Reset context (active neurons, inferred neurons, votes).
    pub fn reset(&mut self) {
        self.frame_number = 0;
        self.age_index.clear();
        self.spatial_level_index.clear();
        self.temporal_level_index.clear();
        self.neuron_states.clear();
        self.inferred_neurons.clear();
    }


    /// Sync memory's frame counter to the brain's. Memory acts as a slave clock — the brain owns
    /// the frame counter and passes it in so the two never drift. Called once at the top of every
    /// frame, before anything else reads `self.frame_number`.
    pub fn sync_frame(&mut self, frame_number: FrameNumber) {
        self.frame_number = frame_number;
    }

    /// Reset spatial state for a new frame. Spatial is fundamentally same-frame: connections[0]
    /// predict co-activations within a single frame's spatial phase. Carrying spatial activations
    /// across frames would cause learn_connections to strengthen d=0 edges between this frame's
    /// actives and last frame's neurons — spurious cross-frame co-activation. Clear the spatial
    /// index here at the top of process_spatial; activate_sensory_events_spatial (called next)
    /// repopulates spatial[0] with this frame's sensory. age_index, neuron_states, and
    /// temporal_level_index entries are untouched — those carry the temporal history.
    pub fn reset_spatial(&mut self) {
        self.spatial_level_index.clear();
    }

    /// Advance the temporal sliding window by one frame: evict whatever activation frame just fell
    /// off the back. With frame-keyed storage this is just a single eviction pass — no per-neuron
    /// Map rebuild, regardless of how many neurons are active. Must be called after `sync_frame`
    /// so the eviction target is computed from the current frame number.
    pub fn advance_temporal_window(&mut self) {
        if self.debug { println!("Aging neurons..."); }

        // evict the frame that just fell off the back of the window
        let evicted_frame = self.frame_number - self.context_length as i64;
        let evicted_neuron_ids = match self.age_index.remove(&evicted_frame) {
            Some(ids) => ids,
            None => return, // if there are no neurons in the evicted frame, nothing to do
        };

        // drop the evicted frame from every index in one pass
        for &neuron_id in &evicted_neuron_ids {
            if let Some(states) = self.neuron_states.get_mut(&neuron_id) {
                states.remove(&evicted_frame);
                if states.is_empty() { self.neuron_states.remove(&neuron_id); } // deactivate neuron if all ages are inactive
            }
        }
        // spatial_level_index is wiped each frame by reset_spatial — no per-frame eviction needed.
        for level_frames in self.temporal_level_index.values_mut() {
            level_frames.remove(&evicted_frame);
        }

        if self.debug && !evicted_neuron_ids.is_empty() {
            println!("Deactivated {} aged-out neurons", evicted_neuron_ids.len());
        }
    }

    /// Test convenience: do all three steps of a frame tick in one call (sync, reset spatial,
    /// advance temporal window). Production code calls the three methods individually at the
    /// appropriate points in `process_frame` / `process_spatial` / `process_temporal`.
    #[cfg(test)]
    pub fn age(&mut self, frame_number: FrameNumber) {
        self.sync_frame(frame_number);
        self.reset_spatial();
        self.advance_temporal_window();
    }

    /// Number of age slots currently in the sliding window.
    pub fn depth(&self) -> u32 {
        if self.age_index.is_empty() { return 0; }
        let min_frame = *self.age_index.keys().min().unwrap();
        ((self.frame_number - min_frame + 1) as u32).min(self.context_length)
    }

    /// Get the current frame number. Test-only.
    #[cfg(test)]
    pub fn get_frame_number(&self) -> FrameNumber {
        self.frame_number
    }

    /// Get the Set of neuron IDs at a specific age.
    pub fn get_neuron_ids_at_age(&self, age: Distance) -> FxHashSet<NeuronId> {
        let frame = self.frame_number - age as i64;
        self.age_index.get(&frame).cloned().unwrap_or_default()
    }

    /// Get the union of every active voter neuron id across all ages in the sliding window.
    /// A voter is an active neuron whose `activated_pattern_id` is None at that age.
    /// Neurons that activated a higher-level pattern are inhibited and do NOT vote.
    /// A neuron with mixed states (suppressed at some ages, voter at others) counts as a voter.
    /// Wiring is per-neuron, not per-age.
    /// Same neuron active at multiple activation frames is returned once (HashSet dedupes).
    pub fn get_active_voter_ids(&self) -> FxHashSet<NeuronId> {
        let mut ids = FxHashSet::default();
        for (&neuron_id, states) in &self.neuron_states {
            if states.values().any(|state| state.activated_pattern_id.is_none()) {
                ids.insert(neuron_id);
            }
        }
        ids
    }

    /// Get (voter_id, age) pairs for every non-suppressed active state in the sliding window.
    /// Same neuron active at multiple ages emits one pair per non-suppressed age.
    /// Mirrors how `Neuron::generate_votes` walks ages inside process_frame.
    /// Inhibited states (those whose `activated_pattern_id` is set) are skipped.
    /// Used by Brain.learn() to drive the post-wire vote sweep.
    /// Every active voter at every valid age contributes a vote.
    pub fn get_active_voter_ages(&self) -> Vec<(NeuronId, Distance)> {
        let mut pairs = Vec::new();
        for (&neuron_id, states) in &self.neuron_states {
            for (&frame, state) in states {
                if state.activated_pattern_id.is_some() { continue; }
                let age = (self.frame_number - frame) as Distance;
                pairs.push((neuron_id, age));
            }
        }
        pairs
    }

    /// Active neuron sets at the base level (level 0 — the sensory substrate).
    /// Spatial returns a single-element vec (just this frame's active sensory set).
    /// Temporal returns one entry per recency slot inside the sliding window — index 0 is the
    /// current frame, index 1 is the previous frame, and so on out to `depth() - 1`.
    /// Downstream code on the spatial side only ever reads index `[0]`; the recency dimension is
    /// there for temporal pattern allocation, which wires connections back across past frames.
    /// Active sensory event neurons at the spatial base level (level 0).
    /// Returns the single co-active set — no recency dimension, spatial is same-frame.
    pub fn get_spatial_base_level(&self) -> FxHashSet<NeuronId> {
        self.spatial_level_index.get(&0).cloned().unwrap_or_default()
    }

    /// Active sensory event neurons at the temporal base level (level 0), per recency slot.
    /// Index 0 is the current frame, index 1 is the previous frame, etc., out to `depth() - 1`.
    /// Used by temporal pattern allocation to wire connections back across past frames.
    pub fn get_temporal_base_level(&self) -> Vec<FxHashSet<NeuronId>> {
        let level_frames = match self.temporal_level_index.get(&0) {
            Some(lf) => lf,
            None => return Vec::new(),
        };
        let mut result = Vec::with_capacity(self.depth() as usize);
        for age in 0..self.depth() {
            let frame = self.frame_number - age as i64;
            result.push(level_frames.get(&frame).cloned().unwrap_or_default());
        }
        result
    }

    /// Active spatial neurons at a level — just the id set, no per-neuron state, no age dimension.
    /// Spatial neurons don't persist per-neuron state (nothing reads it across the level loop),
    /// so the sweep gets a fresh default state every time it asks for one.
    pub fn get_spatial_level_neurons(&self, level: Level) -> FxHashSet<NeuronId> {
        self.spatial_level_index.get(&level).cloned().unwrap_or_default()
    }

    /// Active temporal neurons at a level with their per-age states.
    /// Shape: FxHashMap<neuron_id, FxHashMap<age, LevelAgeState>>.
    /// Walks frame-keyed neuron_states and assembles the per-age map for each neuron active
    /// at this level anywhere in the sliding window.
    pub fn get_temporal_level_neurons(&self, level: Level) -> FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>> {
        let mut neurons: FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>> = FxHashMap::default();
        let level_frames = match self.temporal_level_index.get(&level) {
            Some(lf) => lf,
            None => return neurons,
        };

        // walk ages ascending so iteration order matches
        for age in 0..self.depth() {
            let frame = self.frame_number - age as i64;
            let neuron_ids = match level_frames.get(&frame) {
                Some(ids) => ids,
                None => continue,
            };
            for &neuron_id in neuron_ids {
                let state = self.neuron_states.get(&neuron_id)
                    .and_then(|states| states.get(&frame))
                    .cloned()
                    .unwrap_or_default();
                neurons.entry(neuron_id)
                    .or_insert_with(FxHashMap::default)
                    .insert(age, state);
            }
        }
        neurons
    }

    /// Write back modified per-neuron states after process_level — temporal only.
    /// The temporal level loop mutates votes/context/threshold on the states; those changes
    /// have to persist into frame-keyed neuron_states for next-frame evaluation.
    /// Spatial calls SHOULD NOT invoke this — spatial state is ephemeral by design.
    pub fn write_back_level_neurons(&mut self, level_neurons: &FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>>) {
        for (&neuron_id, age_states) in level_neurons {
            for (&age, state) in age_states {
                let frame = self.frame_number - age as i64;
                if let Some(states) = self.neuron_states.get_mut(&neuron_id) {
                    if let Some(stored) = states.get_mut(&frame) {
                        stored.votes = state.votes.clone();
                        stored.context = state.context.clone();
                        stored.threshold = state.threshold;
                        stored.activated_pattern_id = state.activated_pattern_id;
                    }
                }
            }
        }
    }

    /// Activate a sensory neuron in the spatial level index. Age-free, state-free — writes only
    /// to spatial_level_index. Nothing else needs to know that this neuron fired in spatial:
    /// the sweep reads spatial_level_index back, and the apex handoff promotes survivors into
    /// temporal via the separate temporal activation path.
    pub fn activate_spatial_neuron(&mut self, neuron_id: NeuronId, level: Level) {
        self.spatial_level_index.entry(level)
            .or_insert_with(FxHashSet::default)
            .insert(neuron_id);
    }

    /// Activate a neuron at age 0 in the temporal level index.
    /// Used by the apex handoff (sensory events + spatial corrections that survived subsumption)
    /// and by carry-forward action injection. Writes age_index + neuron_states + temporal_level_index
    /// because temporal needs all three for its sliding window machinery.
    pub fn activate_temporal_neuron(&mut self, neuron_id: NeuronId, level: Level) {
        self.activate_temporal_neuron_at_age(neuron_id, 0, level);
    }

    /// Activate a neuron at a specific age in the temporal level index. Internally stored keyed
    /// by activation frame = frame_number - age so the entry naturally moves to age+1 next frame
    /// without any rewrite.
    pub fn activate_temporal_neuron_at_age(&mut self, neuron_id: NeuronId, age: Distance, level: Level) {
        let frame = self.frame_number - age as i64;

        // add the neuron to the age index
        self.age_index.entry(frame).or_insert_with(FxHashSet::default).insert(neuron_id);

        // add the neuron to the temporal level index
        self.temporal_level_index.entry(level)
            .or_insert_with(FxHashMap::default)
            .entry(frame)
            .or_insert_with(FxHashSet::default)
            .insert(neuron_id);

        // add the active neuron to the neuron states with a new state (if not already present)
        self.neuron_states.entry(neuron_id)
            .or_insert_with(FxHashMap::default)
            .entry(frame)
            .or_insert_with(LevelAgeState::default);
    }

    /// Activate a spatial pattern neuron at the given level.
    /// Just adds the pattern id to spatial_level_index. No parent-state update —
    /// subsumption is tracked by the caller via a local set, not by reading
    /// activated_pattern_id back from neuron_states (spatial doesn't persist state).
    pub fn activate_spatial_pattern(&mut self, pattern_id: NeuronId, pattern_level: Level) {
        self.activate_spatial_neuron(pattern_id, pattern_level);
    }

    /// Activate a temporal pattern neuron at the given age/level and mark its parent's
    /// activated_pattern_id so vote generation knows to suppress the parent.
    pub fn activate_temporal_pattern(&mut self, pattern_id: NeuronId, pattern_level: Level, parent_id: NeuronId, age: Distance) {
        self.activate_temporal_neuron_at_age(pattern_id, age, pattern_level);

        // set the parent's activated_pattern_id at this age
        let frame = self.frame_number - age as i64;
        if let Some(states) = self.neuron_states.get_mut(&parent_id) {
            if let Some(state) = states.get_mut(&frame) {
                state.activated_pattern_id = Some(pattern_id);
            }
        }
    }

    /// Get all inferred neurons.
    pub fn get_inferred_neurons(&self) -> &[InferredNeuron] {
        &self.inferred_neurons
    }

    /// Clear all inferred neurons. Test-only.
    #[cfg(test)]
    pub fn clear_inferred_neurons(&mut self) {
        self.inferred_neurons.clear();
    }

    /// Save winning inferences to in-memory structures.
    pub fn save_inferred_neurons(&mut self, inferences: Vec<InferredNeuron>) {
        self.inferred_neurons = inferences;
        if self.debug { println!("Saved {} inferences", self.inferred_neurons.len()); }
    }

    /// Verify that none of the deleted pattern ids are currently active.
    pub fn assert_not_active(&self, deleted_pattern_ids: &[NeuronId]) {
        for &pattern_id in deleted_pattern_ids {
            if self.neuron_states.contains_key(&pattern_id) {
                panic!("BUG: deleting active neuron {}", pattern_id);
            }
        }
    }

    /// Maximum active spatial level.
    pub fn get_spatial_max_active_level(&self) -> usize {
        self.spatial_level_index.len()
    }

    /// Maximum active temporal level.
    pub fn get_temporal_max_active_level(&self) -> usize {
        self.temporal_level_index.len()
    }

    // ── Context snapshot ────────────────────────────────────────────────────

    /// Export active neuron activations with their full per-age state.
    /// Temporal-only — spatial state is wiped at the top of every frame and never persists, so
    /// there's nothing meaningful to snapshot. Each entry records (neuron_id, frame, level, state)
    /// for a temporal activation that's still inside the sliding window.
    pub fn get_context_snapshot(&self) -> Vec<(NeuronId, FrameNumber, Level, LevelAgeState)> {
        let mut entries = Vec::new();
        for (&frame, neuron_ids) in &self.age_index {
            for &neuron_id in neuron_ids {
                let state = self.neuron_states.get(&neuron_id)
                    .and_then(|states| states.get(&frame))
                    .cloned()
                    .unwrap_or_default();
                let level_opt = self.temporal_level_index.iter()
                    .find_map(|(&lvl, frames)| {
                        frames.get(&frame).and_then(|ids| {
                            if ids.contains(&neuron_id) { Some(lvl) } else { None }
                        })
                    });
                if let Some(level) = level_opt {
                    entries.push((neuron_id, frame, level, state.clone()));
                }
            }
        }
        entries
    }

    /// Restore temporal active neurons from a saved context snapshot (with full state).
    /// Spatial state is not part of the snapshot — it gets repopulated by the next
    /// `process_spatial` call from scratch.
    pub fn restore_context_snapshot(&mut self, frame_number: FrameNumber, entries: &[(NeuronId, FrameNumber, Level, LevelAgeState)]) {
        self.reset();
        self.frame_number = frame_number;
        for (neuron_id, activation_frame, level, state) in entries {
            self.age_index.entry(*activation_frame)
                .or_insert_with(FxHashSet::default)
                .insert(*neuron_id);
            self.temporal_level_index.entry(*level)
                .or_insert_with(FxHashMap::default)
                .entry(*activation_frame)
                .or_insert_with(FxHashSet::default)
                .insert(*neuron_id);
            self.neuron_states.entry(*neuron_id)
                .or_insert_with(FxHashMap::default)
                .insert(*activation_frame, state.clone());
        }
    }

    /// Export inferred neurons for context serialization.
    pub fn get_inferred_snapshot(&self) -> Vec<InferredNeuron> {
        self.inferred_neurons.clone()
    }

    /// Restore inferred neurons from a saved context snapshot.
    pub fn restore_inferred_snapshot(&mut self, inferred: Vec<InferredNeuron>) {
        self.inferred_neurons = inferred;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_memory_is_empty() {
        let m = Memory::new(false, 4);
        assert_eq!(m.depth(), 0);
        assert_eq!(m.get_frame_number(), 0);
        assert!(m.get_inferred_neurons().is_empty());
    }

    #[test]
    fn test_activate_and_age() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_temporal_neuron(10, 0);
        m.activate_temporal_neuron(20, 0);

        // at age 0, both neurons should be present
        let ids = m.get_neuron_ids_at_age(0);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert_eq!(m.depth(), 1);

        // advance to frame 2 — neurons move to age 1
        m.age(2);
        assert_eq!(m.depth(), 2);
        let ids_age1 = m.get_neuron_ids_at_age(1);
        assert!(ids_age1.contains(&10));
        let ids_age0 = m.get_neuron_ids_at_age(0);
        assert!(ids_age0.is_empty());
    }

    #[test]
    fn test_eviction() {
        let mut m = Memory::new(false, 3);

        // activate neuron 10 at frame 1
        m.age(1);
        m.activate_temporal_neuron(10, 0);

        // advance through context_length frames
        m.age(2);
        m.age(3);
        // neuron 10 is now at age 2 — still in window (depth=3, ages 0..2)
        assert!(m.get_neuron_ids_at_age(2).contains(&10));

        // advance past context_length — neuron 10 should be evicted
        m.age(4);
        // frame 1 is evicted (4 - 3 = 1)
        assert!(!m.get_neuron_ids_at_age(3).contains(&10));
        assert_eq!(m.neuron_states.len(), 0); // fully evicted
    }

    #[test]
    fn test_get_level_neurons() {
        let mut m = Memory::new(false, 4);

        // frame 1: activate neuron 10 at level 0
        m.age(1);
        m.activate_temporal_neuron(10, 0);

        // frame 2: activate neuron 20 at level 0
        m.age(2);
        m.activate_temporal_neuron(20, 0);

        // get level 0 neurons
        let level_neurons = m.get_temporal_level_neurons(0);
        assert!(level_neurons.contains_key(&10));
        assert!(level_neurons.contains_key(&20));

        // neuron 10 should have age 1, neuron 20 should have age 0
        assert!(level_neurons.get(&10).unwrap().contains_key(&1));
        assert!(level_neurons.get(&20).unwrap().contains_key(&0));
    }

    #[test]
    fn test_get_base_level() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_temporal_neuron(10, 0);
        m.age(2);
        m.activate_temporal_neuron(20, 0);

        let sets = m.get_temporal_base_level();
        assert_eq!(sets.len(), 2); // depth = 2
        assert!(sets[0].contains(&20)); // current frame
        assert!(sets[1].contains(&10)); // previous frame
    }

    #[test]
    fn test_activate_pattern() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_temporal_neuron(10, 0); // parent at age 0

        m.age(2);
        // parent is now at age 1, activate pattern as child
        m.activate_temporal_pattern(100, 1, 10, 1);

        // pattern should be active at level 1, age 1
        let level1 = m.get_temporal_level_neurons(1);
        assert!(level1.contains_key(&100));
        assert!(level1.get(&100).unwrap().contains_key(&1));

        // parent's state at age 1 should have activated_pattern_id
        let level0 = m.get_temporal_level_neurons(0);
        let parent_state = level0.get(&10).unwrap().get(&1).unwrap();
        assert_eq!(parent_state.activated_pattern_id, Some(100));
    }

    #[test]
    fn test_inferred_neurons() {
        let mut m = Memory::new(false, 4);
        m.save_inferred_neurons(vec![
            InferredNeuron { neuron_id: 1, coordinate: Coordinate { dim_id: 0, bucket_id: 1 }, channel_id: 0, strength: 0.8, reward: 0.0, probability: 0.0 },
            InferredNeuron { neuron_id: 2, coordinate: Coordinate { dim_id: 0, bucket_id: 2 }, channel_id: 0, strength: 0.6, reward: 0.0, probability: 0.0 },
        ]);
        assert_eq!(m.get_inferred_neurons().len(), 2);

        m.clear_inferred_neurons();
        assert!(m.get_inferred_neurons().is_empty());
    }

    #[test]
    fn test_reset() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_temporal_neuron(10, 0);
        m.save_inferred_neurons(vec![InferredNeuron { neuron_id: 1, coordinate: Coordinate { dim_id: 0, bucket_id: 1 }, channel_id: 0, strength: 1.0, reward: 0.0, probability: 0.0 }]);

        m.reset();
        assert_eq!(m.depth(), 0);
        assert_eq!(m.get_frame_number(), 0);
        assert!(m.get_inferred_neurons().is_empty());
        assert_eq!(m.neuron_states.len(), 0);
    }

    #[test]
    fn test_depth_grows_to_context_length() {
        let mut m = Memory::new(false, 3);
        assert_eq!(m.depth(), 0);
        m.age(1); m.activate_temporal_neuron(1, 0); assert_eq!(m.depth(), 1);
        m.age(2); m.activate_temporal_neuron(2, 0); assert_eq!(m.depth(), 2);
        m.age(3); m.activate_temporal_neuron(3, 0); assert_eq!(m.depth(), 3);
        m.age(4); m.activate_temporal_neuron(4, 0); assert_eq!(m.depth(), 3); // pinned at context_length
        m.age(5); m.activate_temporal_neuron(5, 0); assert_eq!(m.depth(), 3);
    }

    #[test]
    fn test_context_restore_with_negative_frames() {
        let mut m = Memory::new(false, 10);
        m.restore_context_snapshot(0, &[
            (10, -1, 0, LevelAgeState::default()),
            (20, -3, 0, LevelAgeState::default()),
            (30, -5, 0, LevelAgeState::default()),
        ]);
        assert_eq!(m.depth(), 6); // span: 0 - (-5) + 1 = 6
        assert_eq!(m.get_frame_number(), 0);

        // after first frame advance, depth grows by 1
        m.age(1);
        assert_eq!(m.depth(), 7); // span: 1 - (-5) + 1 = 7

        // restored neurons are reachable at correct ages
        let ids = m.get_neuron_ids_at_age(2); // frame 1 - 2 = -1
        assert!(ids.contains(&10));
        let ids = m.get_neuron_ids_at_age(4); // frame 1 - 4 = -3
        assert!(ids.contains(&20));
        let ids = m.get_neuron_ids_at_age(6); // frame 1 - 6 = -5
        assert!(ids.contains(&30));
    }

    #[test]
    #[should_panic(expected = "BUG: deleting active neuron")]
    fn test_assert_not_active() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_temporal_neuron(42, 0);
        m.assert_not_active(&[42]);
    }

    #[test]
    fn test_max_active_level() {
        let mut m = Memory::new(false, 4);
        assert_eq!(m.get_temporal_max_active_level(), 0);
        m.age(1);
        m.activate_temporal_neuron(10, 0);
        assert_eq!(m.get_temporal_max_active_level(), 1); // level 0 exists
        m.activate_temporal_neuron(20, 2);
        assert_eq!(m.get_temporal_max_active_level(), 2); // levels 0 and 2
    }
}
