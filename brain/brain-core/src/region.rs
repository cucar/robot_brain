/// Region — wraps the columns and fans their work out in parallel with Rayon.
///
/// Every batch follows three phases: route the tasks to columns, dispatch so each thread holds one column
/// exclusively, collect.
/// No locks, barriers or channels: columns never share mutable state.

use rayon::prelude::*;

use crate::column::Column;
use crate::neuron::Neuron;
use crate::types::NeuronId;

pub struct Region {
    columns: Vec<Column>,
}

impl Region {
    pub fn new(columns: usize) -> Self {
        Region { columns: (0..columns.max(1)).map(|_| Column::default()).collect() }
    }

    /// Map a neuron id to its owning column within this region.
    pub fn route_neuron(&self, neuron_id: NeuronId) -> usize {
        (neuron_id as usize) % self.columns.len()
    }

    /// Create neurons in their owning columns.
    pub fn create_neurons(&mut self, specs: Vec<(NeuronId, usize)>) {
        let mut per_column: Vec<Vec<(NeuronId, usize)>> = (0..self.columns.len()).map(|_| Vec::new()).collect();
        for spec in specs { per_column[self.route_neuron(spec.0)].push(spec); }
        self.columns.par_iter_mut().zip(per_column.into_par_iter()).for_each(|(col, specs)| col.create_neurons(specs));
    }

    /// Route tagged tasks to their columns and run the columns in parallel.
    pub fn run<T: Send, R: Send>(&mut self, tasks: Vec<(usize, NeuronId, T)>, op: &(impl Fn(&mut Neuron, T) -> R + Sync)) -> Vec<(usize, R)> {
        let mut per_column: Vec<Vec<(usize, NeuronId, T)>> = (0..self.columns.len()).map(|_| Vec::new()).collect();
        for task in tasks { let c = self.route_neuron(task.1); per_column[c].push(task); }
        let nested: Vec<Vec<(usize, R)>> = self.columns.par_iter_mut()
            .zip(per_column.into_par_iter())
            .map(|(col, tasks)| if tasks.is_empty() { Vec::new() } else { col.run(tasks, op) })
            .collect();
        nested.into_iter().flatten().collect()
    }

    /// Run an operation on every living neuron, in parallel over the columns.
    pub fn run_living<R: Send>(&mut self, op: &(impl Fn(&mut Neuron) -> R + Sync)) -> Vec<R> {
        let nested: Vec<Vec<R>> = self.columns.par_iter_mut().map(|col| col.run_living(op)).collect();
        nested.into_iter().flatten().collect()
    }
}
