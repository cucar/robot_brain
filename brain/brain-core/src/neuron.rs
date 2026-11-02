//! One neuron (§4, §5): a history of its activations, a relation index over it, and its connections. It
//! records, counts and returns; it decides nothing about what is covered. Its call during recognition is
//! `activate` (R20): refresh history, refresh index, return. What `count` answers (D47, R50) is read off the
//! rows by the machine.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::types::*;

pub use crate::activation::ActRef;
pub use crate::connection::*;

/// A relation as one neuron records it from its own activation (D47): the neighbor's neuron and its offset.
pub type RelKey = (NeuronId, Offset);

/// One neighbor of a row (D26): the activation, its neuron, and its exact offset from the row's activation (D6).
#[derive(Clone, Copy, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub neuron: NeuronId,
    pub off: Offset,
}

/// One row of the history (D18): an activation and the neighborhood it observed.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: u64,
    pub act: ActRef,
    pub nbs: Vec<Nb>,
}

/// What the machine hands a neuron for one activation of the level (R20).
#[derive(Clone, Debug)]
pub struct RowInput {
    pub act: ActRef,
    pub nbs: Vec<Nb>,
}

/// What the call returns for one activation (R21): each neighbor, with the strength of its relation.
#[derive(Clone, Debug)]
pub struct ActReturn {
    pub act: ActRef,
    pub rels: Vec<(Nb, u32)>,
}

pub struct Neuron {
    pub id: NeuronId,
    pub kind: Kind,
    pub dim: Option<DimId>,
    pub bucket: u32,
    pub channel: Option<ChannelId>,
    pub axes: u8,
    pub alive: bool,
    pub history: VecDeque<Row>,
    pub h: usize,
    /// The relation index (D47): every relation its rows hold, with its strength.
    pub index: FxHashMap<RelKey, u32>,
    pub connections: FxHashMap<(NeuronId, Offset), Conn>,
    next_row: u64,
}

fn keys_of(row: &Row) -> Vec<RelKey> {
    let mut k: Vec<RelKey> = row.nbs.iter().map(|nb| (nb.neuron, nb.off)).collect();
    k.sort_unstable();
    k.dedup();
    k
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8, h: usize) -> Self {
        Neuron { id, kind, dim: Some(dim), bucket, channel: Some(channel), axes, alive: true, history: VecDeque::new(), h, index: FxHashMap::default(), connections: FxHashMap::default(), next_row: 0 }
    }

    pub fn inter(id: NeuronId, axes: u8, h: usize) -> Self {
        Neuron { id, kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes, alive: true, history: VecDeque::new(), h, index: FxHashMap::default(), connections: FxHashMap::default(), next_row: 0 }
    }

    pub fn strength(&self, k: &RelKey) -> u32 { self.index.get(k).copied().unwrap_or(0) }

    /// `activate` (R20): evict the oldest rows as the frame needs, admit the frame's rows, and return each
    /// activation's relations with their strengths.
    pub fn activate(&mut self, inputs: Vec<RowInput>) -> Vec<ActReturn> {
        let incoming = inputs.len().min(self.h);
        while self.history.len() + incoming > self.h {
            let old = self.history.pop_front().unwrap();
            for k in keys_of(&old) {
                if let Some(v) = self.index.get_mut(&k) {
                    *v -= 1;
                    if *v == 0 { self.index.remove(&k); }
                }
            }
        }
        let mut admitted: Vec<(ActRef, Vec<Nb>)> = Vec::new();
        for inp in inputs.into_iter().take(incoming) {
            let row = Row { id: self.next_row, act: inp.act, nbs: inp.nbs };
            self.next_row += 1;
            for k in keys_of(&row) { *self.index.entry(k).or_insert(0) += 1; }
            admitted.push((row.act, row.nbs.clone()));
            self.history.push_back(row);
        }
        admitted.into_iter().map(|(act, nbs)| ActReturn { act, rels: nbs.iter().map(|nb| (*nb, self.strength(&(nb.neuron, nb.off)))).collect() }).collect()
    }

    /// What `activate` would return, admitting nothing: a frozen machine's call (the machine's test mode).
    pub fn peek(&self, inputs: Vec<RowInput>) -> Vec<ActReturn> {
        inputs.into_iter().map(|inp| ActReturn { act: inp.act, rels: inp.nbs.iter().map(|nb| (*nb, self.strength(&(nb.neuron, nb.off)) + 1)).collect() }).collect()
    }

    /// A relation seen from the other end after this activation's row was written (experiment): an activation that
    /// fired later in the same frame saw it, so its row holds the later one too, and the index counts it.
    pub fn add_seen(&mut self, act: ActRef, nb: Nb) {
        let Some(row) = self.history.iter_mut().rev().find(|r| r.act == act) else { return };
        let k = (nb.neuron, nb.off);
        let had = row.nbs.iter().any(|x| x.neuron == nb.neuron && x.off == nb.off);
        if row.nbs.iter().any(|x| x.act == nb.act) { return; }
        row.nbs.push(nb);
        if !had { *self.index.entry(k).or_insert(0) += 1; }
    }

    /// `forget neuron` (§4.1): remove a decommissioned neuron from every row and relation that names it.
    pub fn forget(&mut self, n: NeuronId) {
        self.index.retain(|k, _| k.0 != n);
        for row in self.history.iter_mut() { row.nbs.retain(|nb| nb.neuron != n); }
        self.connections.retain(|(t, _), _| *t != n);
    }
}
