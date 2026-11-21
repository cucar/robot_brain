/// Column — owns a partition of the neurons and runs the per-neuron work of a batch on them.
///
/// A column never shares mutable state with another, so a region runs its columns in parallel.
/// Each task comes tagged with its place in the batch, and its result goes back with the same tag.

use rustc_hash::FxHashMap;

use crate::neuron::Neuron;
use crate::types::NeuronId;

#[derive(Default)]
pub struct Column {
    /// The sole storage for owned neurons, keyed by neuron id.
    neurons: FxHashMap<NeuronId, Neuron>,
}

impl Column {
    /// Create neurons in this column.
    pub fn create_neurons(&mut self, specs: Vec<(NeuronId, usize)>) {
        for (id, h) in specs { self.neurons.insert(id, Neuron::new(id, h)); }
    }

    /// Run a task on each neuron it names, in the order the tasks came, each result tagged with its task's place.
    pub fn run<T, R>(&mut self, tasks: Vec<(usize, NeuronId, T)>, op: &(impl Fn(&mut Neuron, T) -> R + Sync)) -> Vec<(usize, R)> {
        tasks.into_iter().map(|(at, id, task)| {
            let neuron = self.neurons.get_mut(&id).unwrap_or_else(|| panic!("Column.run: neuron {} not found", id));
            (at, op(neuron, task))
        }).collect()
    }

    /// Run an operation on every living neuron this column owns.
    pub fn run_living<R>(&mut self, op: &(impl Fn(&mut Neuron) -> R + Sync)) -> Vec<R> {
        self.neurons.values_mut().filter(|n| n.alive).map(|n| op(n)).collect()
    }
}
