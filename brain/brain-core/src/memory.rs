/// Memory — manages the temporal sliding window of active and inferred neurons.
///
/// Internally everything is keyed by **activation frame number**, not by age.
/// Age is a derived quantity (`frame_number - activation_frame`) computed on read.
/// This means `age()` doesn't have to migrate any per-neuron state Maps each frame —
/// it just bumps the frame counter and evicts whatever frame fell off the back of
/// the window. Public API (depth, get_level_neurons, get_level_ages, …) still speaks
/// in ages, so callers (brain, thalamus) are unchanged.

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

    /// Index for fast level queries — level → frame → set of neuron ids.
    level_index: FxHashMap<Level, FxHashMap<FrameNumber, FxHashSet<NeuronId>>>,

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
            level_index: FxHashMap::default(),
            inferred_neurons: Vec::new(),
            debug,
        }
    }

    /// Reset context (active neurons, inferred neurons, votes).
    pub fn reset(&mut self) {
        self.frame_number = 0;
        self.age_index.clear();
        self.level_index.clear();
        self.neuron_states.clear();
        self.inferred_neurons.clear();
    }

    /// Advance the sliding window to the given frame. Memory acts as a slave clock —
    /// the brain owns the frame counter and passes it in so the two never drift.
    /// With frame-keyed storage this is just a counter sync plus eviction of whatever
    /// activation frame fell off the back — no per-neuron Map rebuild, regardless of
    /// how many neurons are active.
    pub fn age(&mut self, frame_number: FrameNumber) {
        if self.debug { println!("Aging neurons..."); }

        // sync to the brain's frame number
        self.frame_number = frame_number;

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
        for level_frames in self.level_index.values_mut() {
            level_frames.remove(&evicted_frame);
        }

        if self.debug && !evicted_neuron_ids.is_empty() {
            println!("Deactivated {} aged-out neurons", evicted_neuron_ids.len());
        }
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

    /// Get the per-age Sets of neuron IDs for a level (index = age), in age-ascending order.
    /// Returns an empty vec if the level has no active neurons.
    /// Built fresh on each call by walking ages 0..depth-1 against the frame-keyed level index.
    pub fn get_level_ages(&self, level: Level) -> Vec<FxHashSet<NeuronId>> {
        let level_frames = match self.level_index.get(&level) {
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

    /// Returns the active neurons at a given level with their age-states.
    /// Shape: FxHashMap<neuron_id, FxHashMap<age, LevelAgeState>>.
    /// thalamus.process_level depends on this.
    pub fn get_level_neurons(&mut self, level: Level) -> FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>> {
        let mut neurons: FxHashMap<NeuronId, FxHashMap<Distance, LevelAgeState>> = FxHashMap::default();
        let level_frames = match self.level_index.get(&level) {
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

    /// Write back modified level age states after process_level. The level loop
    /// mutates votes/context/threshold on the states, and we need to persist those
    /// changes back into frame-keyed storage for next-frame evaluation.
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

    /// Activate a neuron at age 0.
    pub fn activate_neuron(&mut self, neuron_id: NeuronId, level: Level) {
        self.activate_neuron_at_age(neuron_id, 0, level);
    }

    /// Activate a neuron at a specific age.
    /// Internally stored keyed by activation frame = frame_number - age
    /// so the entry naturally moves to age+1 next frame without any rewrite.
    pub fn activate_neuron_at_age(&mut self, neuron_id: NeuronId, age: Distance, level: Level) {
        let frame = self.frame_number - age as i64;

        // add the neuron to the age index
        self.age_index.entry(frame).or_insert_with(FxHashSet::default).insert(neuron_id);

        // add the neuron to the level index
        self.level_index.entry(level)
            .or_insert_with(FxHashMap::default)
            .entry(frame)
            .or_insert_with(FxHashSet::default)
            .insert(neuron_id);

        // add the active neuron to the neuron states with a new state
        self.neuron_states.entry(neuron_id)
            .or_insert_with(FxHashMap::default)
            .insert(frame, LevelAgeState::default());
    }

    /// Activate a pattern neuron and link it to its parent.
    pub fn activate_pattern(&mut self, pattern_id: NeuronId, pattern_level: Level, parent_id: NeuronId, age: Distance) {
        self.activate_neuron_at_age(pattern_id, age, pattern_level);

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

    /// Get the maximum active level from the level index.
    pub fn get_max_active_level(&self) -> usize {
        self.level_index.len()
    }

    // ── Context snapshot ────────────────────────────────────────────────────

    /// Export active neuron activations with their full per-age state.
    pub fn get_context_snapshot(&self) -> Vec<(NeuronId, FrameNumber, Level, LevelAgeState)> {
        let mut entries = Vec::new();
        for (&frame, neuron_ids) in &self.age_index {
            for &neuron_id in neuron_ids {
                let level = self.level_index.iter()
                    .find_map(|(&lvl, frames)| {
                        frames.get(&frame).and_then(|ids| {
                            if ids.contains(&neuron_id) { Some(lvl) } else { None }
                        })
                    })
                    .unwrap_or(0);
                let state = self.neuron_states.get(&neuron_id)
                    .and_then(|states| states.get(&frame))
                    .cloned()
                    .unwrap_or_default();
                entries.push((neuron_id, frame, level, state));
            }
        }
        entries
    }

    /// Restore active neurons from a saved context snapshot (with full state).
    pub fn restore_context_snapshot(&mut self, frame_number: FrameNumber, entries: &[(NeuronId, FrameNumber, Level, LevelAgeState)]) {
        self.reset();
        self.frame_number = frame_number;
        for (neuron_id, activation_frame, level, state) in entries {
            self.age_index.entry(*activation_frame)
                .or_insert_with(FxHashSet::default)
                .insert(*neuron_id);
            self.level_index.entry(*level)
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
        m.activate_neuron(10, 0);
        m.activate_neuron(20, 0);

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
        m.activate_neuron(10, 0);

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
        m.activate_neuron(10, 0);

        // frame 2: activate neuron 20 at level 0
        m.age(2);
        m.activate_neuron(20, 0);

        // get level 0 neurons
        let level_neurons = m.get_level_neurons(0);
        assert!(level_neurons.contains_key(&10));
        assert!(level_neurons.contains_key(&20));

        // neuron 10 should have age 1, neuron 20 should have age 0
        assert!(level_neurons.get(&10).unwrap().contains_key(&1));
        assert!(level_neurons.get(&20).unwrap().contains_key(&0));
    }

    #[test]
    fn test_get_level_ages() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_neuron(10, 0);
        m.age(2);
        m.activate_neuron(20, 0);

        let ages = m.get_level_ages(0);
        assert_eq!(ages.len(), 2); // depth = 2
        assert!(ages[0].contains(&20)); // age 0 = current frame
        assert!(ages[1].contains(&10)); // age 1 = previous frame
    }

    #[test]
    fn test_activate_pattern() {
        let mut m = Memory::new(false, 4);
        m.age(1);
        m.activate_neuron(10, 0); // parent at age 0

        m.age(2);
        // parent is now at age 1, activate pattern as child
        m.activate_pattern(100, 1, 10, 1);

        // pattern should be active at level 1, age 1
        let level1 = m.get_level_neurons(1);
        assert!(level1.contains_key(&100));
        assert!(level1.get(&100).unwrap().contains_key(&1));

        // parent's state at age 1 should have activated_pattern_id
        let level0 = m.get_level_neurons(0);
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
        m.activate_neuron(10, 0);
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
        m.age(1); m.activate_neuron(1, 0); assert_eq!(m.depth(), 1);
        m.age(2); m.activate_neuron(2, 0); assert_eq!(m.depth(), 2);
        m.age(3); m.activate_neuron(3, 0); assert_eq!(m.depth(), 3);
        m.age(4); m.activate_neuron(4, 0); assert_eq!(m.depth(), 3); // pinned at context_length
        m.age(5); m.activate_neuron(5, 0); assert_eq!(m.depth(), 3);
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
        m.activate_neuron(42, 0);
        m.assert_not_active(&[42]);
    }

    #[test]
    fn test_max_active_level() {
        let mut m = Memory::new(false, 4);
        assert_eq!(m.get_max_active_level(), 0);
        m.age(1);
        m.activate_neuron(10, 0);
        assert_eq!(m.get_max_active_level(), 1); // level 0 exists
        m.activate_neuron(20, 2);
        assert_eq!(m.get_max_active_level(), 2); // levels 0 and 2
    }
}
