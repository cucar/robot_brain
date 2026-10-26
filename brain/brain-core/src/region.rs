//! Region: wraps the columns and fans their work out in parallel with Rayon.
//!
//! Every batch operation follows three phases: route items to columns (sequential), dispatch so
//! each thread holds one column exclusively, collect in column order. No locks, barriers or
//! channels: columns never share mutable state, and what they read in common is handed to them
//! immutably. A region is the unit that would become a process of its own.

use rayon::prelude::*;

use crate::column::{Column, LevelResult, LevelTask};
use crate::types::*;

pub struct Region {
    pub columns: Vec<Column>,
}

impl Region {
    pub fn new(columns: usize) -> Self {
        Region { columns: (0..columns.max(1)).map(|_| Column::default()).collect() }
    }

    pub fn column_count(&self) -> usize { self.columns.len() }

    /// The call (R20) at one level on every column, each with its own tasks, in parallel.
    pub fn process_level(&mut self, tasks_by_column: Vec<Vec<LevelTask>>) -> Vec<LevelResult> {
        let nested: Vec<Vec<LevelResult>> = self.columns.par_iter_mut()
            .zip(tasks_by_column.into_par_iter())
            .map(|(col, tasks)| col.process_level(tasks))
            .collect();
        nested.into_iter().flatten().collect()
    }

    /// Forget a decommissioned neuron in every column, in parallel.
    pub fn forget(&mut self, dead: NeuronId) {
        self.columns.par_iter_mut().for_each(|col| col.forget(dead));
    }
}
