//! A neuron's history (D18): rows, their neighbors, covers and owners.


use crate::neuron::*;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// What a pattern names (D16): the owner's own activation, a neuron, or a variable of the owner's table.

#[derive(Clone, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub neuron: NeuronId,
    /// Exact offset from the owner's activation (D6), one component per axis.
    pub delta: Pos,
    /// What the activation cost to write (D13).
    pub cost: f64,
    pub value: Option<Value>,
    /// For a child: what it stands for, its expansion (R28) to base symbols, each as (neuron, offset
    /// from the owner's activation); a node naming one of those neurons holds at the child (the departure).
    pub covers: Vec<(NeuronId, Pos)>,
}

/// The pattern bound to a node where a claimant was taken (D17, D19).
#[derive(Clone, Debug)]
pub struct NodeBind {
    pub node: Node,
    pub offset: Offset,
    /// The neighbors it took (indices into the row's `nbs`); empty where the node failed, or for the owner.
    pub nbs: Vec<usize>,
    /// For a variable node: the member that stood, and one neighbor per position (none where failed).
    pub var: Option<VarBind>,
}

#[derive(Clone, Debug)]
pub struct VarBind {
    pub var: PatternId,
    pub member: Option<NeuronId>,
    pub positions: Vec<Option<usize>>,
    pub failed: u32,
    pub choice: f64,
}

/// One claimant taken into a row's cover (D17).
#[derive(Clone, Debug)]
pub struct CoverEntry {
    pub pattern: PatternId,
    /// Absolute coordinate of the reference node: the use this entry is of (D27).
    pub place: Pos,
    pub nodes: Vec<NodeBind>,
    pub failed: u32,
    pub credited_owner: bool,
    pub names_owner: bool,
}

/// One row of the history: an activation of the owner with what it observed (D18).
#[derive(Clone, Debug)]
pub struct Row {
    pub act: ActRef,
    pub pos: Pos,
    pub own_cost: f64,
    /// The owner's value read through value neurons (D44), for the owner as a parameter position.
    pub owner_value: NeuronId,
    pub nbs: Vec<Nb>,
    pub cover: Vec<CoverEntry>,
    /// Per neighbor, the pattern credited with it (D19).
    pub owner: Vec<Option<PatternId>>,
    pub owner_self: Option<PatternId>,
    /// Per neighbor, the variable holding that coordinate, and which of its positions (D27, D33).
    pub claim: Vec<Option<(PatternId, usize)>>,
    pub dirty: bool,
}

impl Row {
    pub(crate) fn residual(&self, j: usize) -> bool { self.owner[j].is_none() }
    pub(crate) fn credit(&self, j: usize) -> f64 { self.nbs[j].cost }
    pub(crate) fn coord(&self, j: usize) -> Pos { add(&self.pos, &self.nbs[j].delta) }
    /// The value a parameter reads at a neighbor: through value neurons (D44).
    pub(crate) fn value_at(&self, j: usize) -> NeuronId { self.nbs[j].value.map(|v| v.member).unwrap_or(self.nbs[j].neuron) }
}

#[inline]
pub fn add(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }
#[inline]
pub fn sub(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] -= b[d]; } p }


pub(crate) fn nb_place(row: &Row, v: &VarBind) -> Option<Pos> {
    v.positions.first().copied().flatten().map(|j| row.coord(j))
}

