//! Column: owns a partition of the neurons and runs the per-neuron work on them. A column never
//! shares mutable state with another, so a region can run its columns in parallel; the machine
//! routes each neuron to its column by id and gets results back tagged by neuron.

use crate::neuron::*;
use crate::types::*;

/// One neuron's call at a level (R20): the rows of its activations.
pub struct LevelTask {
    pub slot: usize,
    pub rows: Vec<RowInput>,
}

/// What one neuron returned from its call (R21).
pub struct LevelResult {
    pub neuron: NeuronId,
    pub returns: Vec<ActReturn>,
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
            let returns = n.activate(t.rows);
            out.push(LevelResult { neuron: n.id, returns });
        }
        out
    }

    /// `forget neuron` on every living neuron this column owns (§4.1).
    pub fn forget(&mut self, dead: NeuronId) {
        for n in self.neurons.iter_mut().filter(|n| n.alive) { n.forget(dead); }
    }
}
