//! Thalamus: the relay between the machine and its neurons. Owns the region tree (each region its
//! columns, each column its neurons), routes a neuron id to the column that owns it, allots ids, and
//! fans the per-neuron work out through the regions.
//!
//! Routing: with R regions and C columns per region, neuron id n lives in region n mod R, column
//! (n div R) mod C, at slot n div (R·C). Ids are allotted in order, so the columns fill evenly.

use rayon::prelude::*;
use std::ops::{Index, IndexMut};

use crate::column::{LevelResult, LevelTask};
use crate::neuron::*;
use crate::region::Region;
use crate::types::*;

pub struct Thalamus {
    pub regions: Vec<Region>,
    columns: usize,
    count: usize,
}

impl Thalamus {
    pub fn new(regions: usize, columns: usize) -> Self {
        let regions = regions.max(1);
        let columns = columns.max(1);
        Thalamus { regions: (0..regions).map(|_| Region::new(columns)).collect(), columns, count: 0 }
    }

    pub fn region_count(&self) -> usize { self.regions.len() }
    pub fn column_count(&self) -> usize { self.columns }

    /// Where a neuron lives: (region, column, slot).
    pub fn route(&self, id: NeuronId) -> (usize, usize, usize) {
        let n = id as usize;
        let r = self.regions.len();
        (n % r, (n / r) % self.columns, n / (r * self.columns))
    }

    /// The next id to be allotted.
    pub fn next_id(&self) -> NeuronId { self.count as NeuronId }

    /// How many neurons have been allotted, living or not.
    pub fn len(&self) -> usize { self.count }

    /// Take a neuron whose id is the next to be allotted.
    pub fn push(&mut self, n: Neuron) {
        assert_eq!(n.id as usize, self.count, "neurons are allotted in id order");
        let (r, c, _) = self.route(n.id);
        self.regions[r].columns[c].push(n);
        self.count += 1;
    }

    pub fn get(&self, id: NeuronId) -> Option<&Neuron> {
        if (id as usize) >= self.count { return None; }
        let (r, c, s) = self.route(id);
        self.regions[r].columns[c].get(s)
    }

    pub fn get_mut(&mut self, id: NeuronId) -> Option<&mut Neuron> {
        if (id as usize) >= self.count { return None; }
        let (r, c, s) = self.route(id);
        self.regions[r].columns[c].get_mut(s)
    }

    /// Every neuron in id order.
    pub fn iter(&self) -> impl Iterator<Item = &Neuron> + '_ {
        (0..self.count as NeuronId).map(move |id| &self[id])
    }

    /// The call (R20) at one level: route each neuron's task to its column, run the columns of every region in
    /// parallel, and hand the results back in the order the tasks came.
    pub fn process_level(&mut self, tasks: Vec<(NeuronId, Vec<RowInput>)>) -> Vec<LevelResult> {
        let order: Vec<NeuronId> = tasks.iter().map(|t| t.0).collect();
        let r = self.regions.len();
        let mut by_region: Vec<Vec<Vec<LevelTask>>> = (0..r).map(|_| (0..self.columns).map(|_| Vec::new()).collect()).collect();
        for (id, rows) in tasks {
            let (ri, ci, slot) = self.route(id);
            by_region[ri][ci].push(LevelTask { slot, rows });
        }
        let nested: Vec<Vec<LevelResult>> = self.regions.par_iter_mut()
            .zip(by_region.into_par_iter())
            .map(|(region, tasks)| region.process_level(tasks))
            .collect();
        let mut results: rustc_hash::FxHashMap<NeuronId, LevelResult> = nested.into_iter().flatten().map(|w| (w.neuron, w)).collect();
        order.into_iter().filter_map(|id| results.remove(&id)).collect()
    }

    /// Forget a decommissioned neuron in every living neuron, in parallel (§4.1).
    pub fn forget(&mut self, dead: NeuronId) {
        self.regions.par_iter_mut().for_each(|region| region.forget(dead));
    }
}

impl Index<NeuronId> for Thalamus {
    type Output = Neuron;
    fn index(&self, id: NeuronId) -> &Neuron { self.get(id).expect("no such neuron") }
}

impl IndexMut<NeuronId> for Thalamus {
    fn index_mut(&mut self, id: NeuronId) -> &mut Neuron { self.get_mut(id).expect("no such neuron") }
}
