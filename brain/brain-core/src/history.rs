//! A neuron's history (D18): rows, each an activation of the neuron with the neighborhood it observed, every
//! neighbor at its exact offset and with what it cost in the file as it stood.

use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// How a neighbor stands beside the activation (D5, D47).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Via {
    /// On the apex, as itself.
    Direct,
    /// Covered by a call on the apex other than the activation's own (D5): worth its call's record together
    /// with everything else that call covers, and nothing singly (D22, R23).
    Inside(ActRef),
    /// A base symbol an interneuron on the apex stands for (D47): it holds a node and covers nothing.
    Under(ActRef),
}

/// One neighbor of a row (D26).
#[derive(Clone, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub neuron: NeuronId,
    /// Exact offset from the activation (D6), one component per axis.
    pub delta: Pos,
    pub axes: u8,
    /// What it costs in the file as it stands (D13): its record bare, its neuron alone where a call bound it,
    /// nothing where it stands inside or under a call.
    pub cost: f64,
    /// Its whole record, `d + 2`: what a fresh activation of it costs (D12).
    pub full: f64,
    pub via: Via,
}

/// A call on the apex whose covered activations stand in a row (R23): what it costs, and how many activations
/// it covers, so a fit can tell whether it lands on all of them.
#[derive(Clone, Copy, Debug)]
pub struct CallInfo {
    pub act: ActRef,
    pub size: usize,
    pub cost: f64,
}

/// One row of the history: an activation of the neuron with what it observed (D18).
#[derive(Clone, Debug)]
pub struct Row {
    pub id: u64,
    /// The neuron whose activation it is.
    pub owner: NeuronId,
    pub act: ActRef,
    pub pos: Pos,
    pub axes: u8,
    /// What the activation costs in the file (D13).
    pub own_cost: f64,
    pub nbs: Vec<Nb>,
    pub calls: Vec<CallInfo>,
}

#[inline]
pub fn add(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }
#[inline]
pub fn sub(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] -= b[d]; } p }

/// The spatial dimensions of a layout (D12).
#[inline]
pub fn spatial(axes: u8) -> u32 { (axes & !1).count_ones() }

/// A bare record (D12, D13): the neuron, the frame, and a position per spatial dimension.
#[inline]
pub fn record(axes: u8) -> f64 { spatial(axes) as f64 + 2.0 }
