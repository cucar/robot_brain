use rustc_hash::FxHashMap;

use crate::types::{ContextEntry, NeuronId, Strength};

/// Spatial context: set of neurons co-active on the current frame with strengths.
/// No distance dimension. Storage is just `neuron_id → strength`.
#[derive(Debug, Clone)]
pub struct SpatialContext {
    entries: FxHashMap<NeuronId, Strength>,
}

impl SpatialContext {
    pub fn new() -> Self {
        Self { entries: FxHashMap::default() }
    }

    /// Add a neuron at a given strength.
    /// Panics if the entry already exists — caller must check first.
    pub fn add_neuron(&mut self, neuron_id: NeuronId, strength: Strength) {
        if self.entries.contains_key(&neuron_id) {
            panic!("SpatialContext entry already exists for neuron {}", neuron_id);
        }
        self.entries.insert(neuron_id, strength);
    }

    /// Check if an entry exists.
    pub fn has_key(&self, neuron_id: NeuronId) -> bool {
        self.entries.contains_key(&neuron_id)
    }

    /// Increment the strength of an existing entry by 1.
    /// Panics if the entry does not exist — callers must check has_key first.
    pub fn strengthen_neuron(&mut self, neuron_id: NeuronId) {
        let strength = self.entries.get_mut(&neuron_id).expect("SpatialContext entry not found for strengthening");
        *strength += 1.0;
    }

    /// Remove an entry explicitly. Used by the spatial death cascade to scrub a dying context
    /// neuron from a child pattern's stored context.
    pub fn remove(&mut self, neuron_id: NeuronId) {
        self.entries.remove(&neuron_id);
    }

    /// Direct access to the inner map.
    pub fn entries(&self) -> &FxHashMap<NeuronId, Strength> {
        &self.entries
    }

    /// Flatten all entries into a Vec for iteration / serialization.
    /// Each entry surfaces as a ContextEntry with `distance = 0` so the shared MatchResult /
    /// ContextEntry types stay reusable; readers should ignore the distance field.
    pub fn get_entries(&self) -> Vec<ContextEntry> {
        self.entries.iter()
            .map(|(&neuron_id, &strength)| ContextEntry { neuron_id, distance: 0, strength })
            .collect()
    }

}

