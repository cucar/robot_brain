//! One neuron (§4, §5): a history of its activations, a relation index over it, and its connections. It
//! records, counts and returns; it decides nothing about what is covered. Its call during recognition is
//! `activate` (R20): refresh history, refresh index, return. What `count` answers (D47, R50) is read off the
//! rows by the machine.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::types::*;

pub use crate::activation::ActRef;
pub use crate::connection::*;

/// A relation as one neuron records it from its own activation (D47, D34): its own pose, the neighbor in its
/// pose, and the offset.
pub type RelKey = (Sym, Sym, Offset);

/// One neighbor of a row (D26): the activation that stands there, the neuron in its pose, its exact offset from the
/// row's activation (D6), and the activation the row first saw there, which a call landing on it replaced (D10).
#[derive(Clone, Copy, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub sym: Sym,
    pub off: Offset,
    pub base: ActRef,
}

/// One row of the history (D18): an activation, the pose it fired in, and the neighborhood it observed, with
/// what followed it as it comes.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: u64,
    pub act: ActRef,
    pub own: Sym,
    pub nbs: Vec<Nb>,
}

/// What the machine hands a neuron for one activation of the level (R20).
#[derive(Clone, Debug)]
pub struct RowInput {
    pub act: ActRef,
    pub own: Sym,
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
    /// Per pose of its own, per neuron in its pose and offset, one connection (D25).
    pub connections: FxHashMap<(Sym, Sym, Offset), Conn>,
    next_row: u64,
}

fn keys_of(row: &Row) -> Vec<RelKey> {
    let mut k: Vec<RelKey> = row.nbs.iter().map(|nb| (row.own, nb.sym, nb.off)).collect();
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

    fn index_add(&mut self, keys: &[RelKey]) { for k in keys { *self.index.entry(*k).or_insert(0) += 1; } }

    fn index_sub(&mut self, keys: &[RelKey]) {
        for k in keys {
            if let Some(v) = self.index.get_mut(k) {
                *v -= 1;
                if *v == 0 { self.index.remove(k); }
            }
        }
    }

    /// `activate` (R20): evict the oldest rows as the frame needs, admit the frame's rows, and return each
    /// activation's relations with their strengths.
    pub fn activate(&mut self, inputs: Vec<RowInput>) -> Vec<ActReturn> {
        let incoming = inputs.len().min(self.h);
        while self.history.len() + incoming > self.h {
            let old = self.history.pop_front().unwrap();
            self.index_sub(&keys_of(&old));
        }
        let mut admitted: Vec<(ActRef, Sym, Vec<Nb>)> = Vec::new();
        for inp in inputs.into_iter().take(incoming) {
            let row = Row { id: self.next_row, act: inp.act, own: inp.own, nbs: inp.nbs };
            self.next_row += 1;
            self.index_add(&keys_of(&row));
            admitted.push((row.act, row.own, row.nbs.clone()));
            self.history.push_back(row);
        }
        admitted.into_iter().map(|(act, own, nbs)| ActReturn { act, rels: nbs.iter().map(|nb| (*nb, self.strength(&(own, nb.sym, nb.off)))).collect() }).collect()
    }

    /// What `activate` would return, admitting nothing: a frozen machine's call (the machine's test mode).
    pub fn peek(&self, inputs: Vec<RowInput>) -> Vec<ActReturn> {
        inputs.into_iter().map(|inp| ActReturn { act: inp.act, rels: inp.nbs.iter().map(|nb| (*nb, self.strength(&(inp.own, nb.sym, nb.off)) + 1)).collect() }).collect()
    }

    /// A call landed on an activation this row saw (D10): the row holds the call in its pose at its offset where
    /// it held the activation, one relation per call, and the index follows. Returns whether the row held it.
    pub fn replace_seen(&mut self, viewer: ActRef, old: ActRef, new: Nb) -> bool {
        let Some(pos) = self.history.iter().rposition(|r| r.act == viewer) else { return false };
        let row = &self.history[pos];
        let Some(base) = row.nbs.iter().find(|x| x.act == old || x.base == old).map(|x| x.base) else { return false };
        let before = keys_of(row);
        let row = &mut self.history[pos];
        row.nbs.retain(|x| x.act != old);
        if !row.nbs.iter().any(|x| x.act == new.act) { row.nbs.push(Nb { base, ..new }); }
        let after = keys_of(&self.history[pos]);
        let gone: Vec<RelKey> = before.iter().filter(|k| !after.contains(k)).copied().collect();
        let came: Vec<RelKey> = after.iter().filter(|k| !before.contains(k)).copied().collect();
        self.index_sub(&gone);
        self.index_add(&came);
        true
    }

    /// The row takes what followed its activation (D18), and the index counts it.
    pub fn add_followed(&mut self, viewer: ActRef, nbs: &[Nb]) -> bool {
        let Some(pos) = self.history.iter().rposition(|r| r.act == viewer) else { return false };
        let before = keys_of(&self.history[pos]);
        let row = &mut self.history[pos];
        for nb in nbs { if !row.nbs.iter().any(|x| x.act == nb.act) { row.nbs.push(*nb); } }
        let after = keys_of(&self.history[pos]);
        let came: Vec<RelKey> = after.iter().filter(|k| !before.contains(k)).copied().collect();
        self.index_add(&came);
        true
    }

    /// `forget neuron` (§4.1): remove a decommissioned neuron, in every pose it had, from every row and relation
    /// that names it.
    pub fn forget(&mut self, dead: &[Sym]) {
        self.index.retain(|k, _| !dead.contains(&k.0) && !dead.contains(&k.1));
        for row in self.history.iter_mut() { row.nbs.retain(|nb| !dead.contains(&nb.sym)); }
        self.connections.retain(|(o, t, _), _| !dead.contains(o) && !dead.contains(t));
    }
}
