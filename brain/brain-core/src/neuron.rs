//! One neuron (§4, §5): a history of its activations, a relation index over it, and its connections. It
//! records, counts and returns; it decides nothing about what is written. Its one call during recognition is
//! `activate` (R20): refresh history, refresh index, return.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::offsets::*;
use crate::types::*;

pub use crate::connection::*;
pub use crate::history::*;

/// A relation from a neuron's own activation (D34, D47): the neighbor's neuron, a dimension, and the offset in
/// it, written rounded.
pub type RelKey = (NeuronId, u8, i32);

/// What the machine hands a neuron for one activation of the level (R20).
#[derive(Clone, Debug)]
pub struct RowInput {
    pub act: ActRef,
    pub pos: Pos,
    pub own_cost: f64,
    pub nbs: Vec<Nb>,
    pub calls: Vec<CallInfo>,
}

/// What the call returns for one activation (R21): each relation from it, with the rows that hold it.
#[derive(Clone, Debug)]
pub struct ActReturn {
    pub act: ActRef,
    pub row: u64,
    pub rels: Vec<(RelKey, Vec<u64>)>,
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
    /// The relation index (D47): every relation its rows hold, each with the rows that hold it, by row id.
    pub index: FxHashMap<RelKey, Vec<u64>>,
    pub connections: FxHashMap<(NeuronId, Offset), Conn>,
    pub radii: Vec<i32>,
    next_row: u64,
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind, dim: Some(dim), bucket, channel: Some(channel), axes, alive: true, history: VecDeque::new(), h, index: FxHashMap::default(), connections: FxHashMap::default(), radii, next_row: 0 }
    }

    pub fn inter(id: NeuronId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes, alive: true, history: VecDeque::new(), h, index: FxHashMap::default(), connections: FxHashMap::default(), radii, next_row: 0 }
    }

    /// The relations a row holds (§5.2): one per neighbor per dimension the two share, written rounded (D6).
    pub fn relations_of(&self, row: &Row) -> Vec<RelKey> {
        let mut out: Vec<RelKey> = Vec::new();
        for nb in &row.nbs {
            let shared = (row.axes & nb.axes) | 1;
            for d in 0..self.radii.len().min(MAX_DIMS) {
                if shared & (1 << d) == 0 { continue; }
                out.push((nb.neuron, d as u8, round_offset(nb.delta[d], self.radii[d])));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// `activate` (R20): evict the oldest rows as the frame needs, admit the frame's rows, follow them in the
    /// index, and return the relations of every activation handed in, each with the rows that hold it.
    pub fn activate(&mut self, inputs: Vec<RowInput>) -> Vec<ActReturn> {
        let incoming = inputs.len().min(self.h);
        while self.history.len() + incoming > self.h {
            let old = self.history.pop_front().unwrap();
            for k in self.relations_of(&old) {
                if let Some(v) = self.index.get_mut(&k) {
                    v.retain(|r| *r != old.id);
                    if v.is_empty() { self.index.remove(&k); }
                }
            }
        }
        let mut admitted: Vec<(ActRef, u64)> = Vec::new();
        for inp in inputs.into_iter().take(incoming) {
            let id = self.next_row;
            self.next_row += 1;
            let row = Row { id, owner: self.id, act: inp.act, pos: inp.pos, axes: self.axes, own_cost: inp.own_cost, nbs: inp.nbs, calls: inp.calls };
            for k in self.relations_of(&row) { self.index.entry(k).or_default().push(id); }
            self.history.push_back(row);
            admitted.push((inp.act, id));
        }
        admitted.into_iter().map(|(act, id)| {
            let row = self.history.iter().rev().find(|r| r.id == id).unwrap();
            let rels = self.relations_of(row).into_iter().map(|k| (k, self.index.get(&k).cloned().unwrap_or_default())).collect();
            ActReturn { act, row: id, rels }
        }).collect()
    }

    pub fn row(&self, id: u64) -> Option<&Row> {
        // rows are admitted in id order, so the history is sorted by id
        let first = self.history.front()?.id;
        if id < first { return None; }
        self.history.get((id - first) as usize).filter(|r| r.id == id)
    }

    /// `forget neuron` (§4.1): remove a decommissioned neuron from every row and relation that names it.
    pub fn forget(&mut self, n: NeuronId) {
        self.index.retain(|k, _| k.0 != n);
        for row in self.history.iter_mut() {
            if row.nbs.iter().any(|nb| nb.neuron == n) { row.nbs.retain(|nb| nb.neuron != n); }
        }
        self.connections.retain(|(t, _), _| *t != n);
    }
}
