//! A neuron's history (D18): rows, their neighbors, covers and owners.

use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// One neighbor of a row (D26): the neuron of an adjacent activation at its exact offset, with its cost.
#[derive(Clone, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub neuron: NeuronId,
    /// Exact offset from the owner's activation (D6), one component per axis.
    pub delta: Pos,
    /// What the activation cost (D13).
    pub cost: f64,
    /// For a child: the base symbols it stands for, its expansion (R28), each as (neuron, offset from the
    /// owner's activation); in the relation index it is each of them, its cost split equally (D47).
    pub covers: Vec<(NeuronId, Pos)>,
    /// For a child: every neuron its expansion passes through, interneurons and base symbols, each at its
    /// offset from the owner's activation; a function naming one of them matches the child (D22).
    pub stands_for: Vec<(NeuronId, Pos)>,
}

/// What one place of a function took in a row (D17, D19): for a fixed node the neighbors it covers, for
/// a variable the one neighbor it binds; none where the place failed.
#[derive(Clone, Debug)]
pub struct PlaceBind {
    pub place: usize,
    pub nbs: Vec<usize>,
}

/// One function taken into a row's cover (D17), with what stood in its variables (D37).
#[derive(Clone, Debug)]
pub struct CoverEntry {
    pub function: PatternId,
    pub binds: Vec<PlaceBind>,
    /// Per variable place, the neuron bound there.
    pub args: Vec<(usize, NeuronId)>,
    pub failed: u32,
    pub credited_owner: bool,
    /// What the variables cost to state: which member, per class place and once per parameter (D13).
    pub choice: f64,
}

/// One row of the history: an activation of the owner with what it observed (D18).
#[derive(Clone, Debug)]
pub struct Row {
    pub act: ActRef,
    pub pos: Pos,
    pub own_cost: f64,
    pub nbs: Vec<Nb>,
    pub cover: Vec<CoverEntry>,
    /// Per neighbor, the function of the cover credited with it, by covering or binding it (D19).
    pub owner: Vec<Option<PatternId>>,
    /// Per neighbor, whether the function credited with it binds it in a variable (D10).
    pub bound: Vec<bool>,
    pub owner_self: Option<PatternId>,
    pub dirty: bool,
}

impl Row {
    pub(crate) fn residual(&self, j: usize) -> bool { self.owner[j].is_none() }
    pub(crate) fn credit(&self, j: usize) -> f64 { self.nbs[j].cost }
    pub(crate) fn coord(&self, j: usize) -> Pos { add(&self.pos, &self.nbs[j].delta) }
}

#[inline]
pub fn add(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }
#[inline]
pub fn sub(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] -= b[d]; } p }
