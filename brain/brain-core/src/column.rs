//! Column: owns a partition of the neurons and runs the per-neuron work on them. A column never
//! shares mutable state with another, so a region can run its columns in parallel; the machine
//! routes each neuron to its column by id and gets results back tagged by neuron.

use crate::neuron::*;
use crate::types::*;

/// One neuron's call for a wave (R20): its rows and the context the machine hands it.
pub struct WaveTask {
    pub slot: usize,
    pub rows: Vec<RowInput>,
    pub ctx: Ctx,
}

/// What one neuron returned from its call: bids, retirements, the count added, and the lines that
/// narrowed their uses with the child each holds.
pub struct WaveResult {
    pub neuron: NeuronId,
    pub bids: Vec<Bid>,
    pub retired: Vec<Retired>,
    pub added: usize,
    pub narrowed: Vec<(PatternId, NeuronId)>,
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
    pub fn process_wave(&mut self, tasks: Vec<WaveTask>) -> Vec<WaveResult> {
        let mut out = Vec::with_capacity(tasks.len());
        for t in tasks {
            let n = &mut self.neurons[t.slot];
            let (bids, retired, added) = n.process(t.rows, t.ctx);
            let narrowed = std::mem::take(&mut n.narrowed_children);
            out.push(WaveResult { neuron: n.id, bids, retired, added, narrowed });
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
