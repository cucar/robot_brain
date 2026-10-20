//! Column: owns a partition of the neurons and runs the per-neuron work on them. A column never
//! shares mutable state with another, so a region can run its columns in parallel; the machine
//! routes each neuron to its column by id and gets results back tagged by neuron.

use crate::neuron::*;
use crate::types::*;

/// One neuron's call at a level (R20): its rows and the context the machine hands it.
pub struct LevelTask {
    pub slot: usize,
    pub rows: Vec<RowInput>,
    pub built: std::sync::Arc<rustc_hash::FxHashMap<NeuronId, Vec<NeuronId>>>,
    pub ctx: Ctx,
}

/// What one neuron returned from its call: its bids, everything it retired, and the count added.
pub struct LevelResult {
    pub neuron: NeuronId,
    pub bids: Vec<Bid>,
    pub retired: Vec<Retired>,
    pub added: usize,
}

#[derive(Default)]
pub struct Column {
    /// The neurons this column owns, by slot; the thalamus maps ids to slots.
    pub neurons: Vec<Neuron>,
}

impl Column {
    pub fn get(&self, slot: usize) -> Option<&Neuron> { self.neurons.get(slot) }
    pub fn get_mut(&mut self, slot: usize) -> Option<&mut Neuron> { self.neurons.get_mut(slot) }
    pub fn push(&mut self, n: Neuron) { self.neurons.push(n); }
    pub fn len(&self) -> usize { self.neurons.len() }

    /// Run the call on every neuron this column was handed rows for.
    pub fn process_level(&mut self, tasks: Vec<LevelTask>) -> Vec<LevelResult> {
        let mut out = Vec::with_capacity(tasks.len());
        for t in tasks {
            let n = &mut self.neurons[t.slot];
            n.built = t.built;
            let (bids, retired, added) = n.process(t.rows, t.ctx);
            out.push(LevelResult { neuron: n.id, bids, retired, added });
        }
        out
    }

    /// Scrub a deleted neuron from every living neuron's patterns and saved activations; returns what
    /// each retired for it.
    pub fn delete_neighbor(&mut self, dead: NeuronId) -> Vec<(NeuronId, Retired)> {
        let mut out = Vec::new();
        for n in self.neurons.iter_mut().filter(|n| n.alive) {
            for r in n.delete_neighbor(dead) { out.push((n.id, r)); }
        }
        out
    }
}
