//! A neuron's history (D18): rows, each an activation of the owner with the neighborhood it observed.

use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// One neighbor of a row (D26): the neuron of an adjacent activation at its exact offset, with what it
/// cost as it stood (D13).
#[derive(Clone, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub neuron: NeuronId,
    /// Exact offset from the owner's activation (D6), one component per axis.
    pub delta: Pos,
    /// What the activation cost as it stood: its whole record, or its neuron alone where a call had
    /// already covered its position (D10, D13).
    pub cost: f64,
    /// The part of `cost` that is its position: what a variable binding it covers (D22); zero where its
    /// position was covered already.
    pub pos: f64,
    /// For a child: the base symbols it stands for (R28), each at its offset from the owner's activation;
    /// in the relation index it is each of them (D47).
    pub covers: Vec<(NeuronId, Pos)>,
    /// For a child: every neuron its expansion passes through, interneurons and base symbols, each at its
    /// offset from the owner's activation; a fixed node naming one of them holds there (D54).
    pub stands_for: Vec<(NeuronId, Pos)>,
}

/// One row of the history: an activation of the owner with what it observed (D18).
#[derive(Clone, Debug)]
pub struct Row {
    pub act: ActRef,
    pub pos: Pos,
    /// What the owner's activation cost (D13).
    pub own_cost: f64,
    /// What a record cost when the row stood (D12, D13): present or absent, a neuron, a position. A call's
    /// record and an absence in this row are priced at it, as every neighbor is at what it cost then.
    pub rec: f64,
    pub nbs: Vec<Nb>,
}

#[inline]
pub fn add(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }
#[inline]
pub fn sub(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] -= b[d]; } p }
