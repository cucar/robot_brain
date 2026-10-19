//! Region: wraps the columns and fans their work out in parallel with Rayon.
//!
//! Every batch operation follows three phases: route items to columns (sequential), dispatch so
//! each thread holds one column exclusively, collect in column order. No locks, barriers or
//! channels: columns never share mutable state, and what they read in common is handed to them
//! immutably. A region is the unit that would become a process of its own.

use rayon::prelude::*;

use crate::column::{Column, WaveResult, WaveTask};
use crate::neuron::*;
use crate::types::*;

pub struct Region {
    pub columns: Vec<Column>,
}

impl Region {
    pub fn new(columns: usize) -> Self {
        Region { columns: (0..columns.max(1)).map(|_| Column::default()).collect() }
    }

    pub fn column_count(&self) -> usize { self.columns.len() }

    /// The call (R20) on every column, each with its own tasks, in parallel.
    pub fn process_wave(&mut self, tasks_by_column: Vec<Vec<WaveTask>>) -> Vec<WaveResult> {
        let nested: Vec<Vec<WaveResult>> = self.columns.par_iter_mut()
            .zip(tasks_by_column.into_par_iter())
            .map(|(col, tasks)| col.process_wave(tasks))
            .collect();
        nested.into_iter().flatten().collect()
    }

    /// Scrub a deleted neuron from every column's neurons, in parallel.
    pub fn delete_neighbor(&mut self, dead: NeuronId) -> Vec<(NeuronId, Retired)> {
        let nested: Vec<Vec<(NeuronId, Retired)>> = self.columns.par_iter_mut()
            .map(|col| col.delete_neighbor(dead))
            .collect();
        nested.into_iter().flatten().collect()
    }
}
