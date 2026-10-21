//! Bids (D31, R21): what a neuron returns for each of the frame's activations, and what it retired.

use crate::neuron::*;
use crate::types::*;

/// One bid (D31): a function of the table that fits an activation's neighborhood, whole, with where each
/// node stood and what stood in each variable.
#[derive(Clone, Debug)]
pub struct Bid {
    pub owner: NeuronId,
    pub bidder: ActRef,
    pub bidder_pos: Pos,
    pub bidder_cost: f64,
    pub function: PatternId,
    pub child: Option<NeuronId>,
    /// The activations its fixed nodes stand on: what an accepted bid covers besides the bidder.
    pub covers: Vec<ActRef>,
    /// Per variable place, the activation bound there and its neuron: what an accepted bid binds (D10).
    pub binds: Vec<(usize, ActRef, NeuronId)>,
    pub failed: u32,
    pub p_size: u32,
    pub which_function_bits: f64,
    pub choice: f64,
}

impl Bid {
    /// The fields of its call (R22): what its bidder cost, which function, which member per variable, and
    /// which node failed per node that did not hold.
    pub fn price(&self) -> f64 {
        self.bidder_cost + self.which_function_bits + self.choice + self.failed as f64 * log2(self.p_size.max(1) as f64)
    }
}

/// What the call returns about a function it retired (§5.6).
#[derive(Clone, Debug)]
pub struct Retired {
    pub pattern: PatternId,
    pub child: Option<NeuronId>,
    /// The retired function itself: the ledger holds the function, not a handle to it, so what it fired
    /// can still be expanded until its windows close (R38).
    pub line: Option<Function>,
}

impl Neuron {
    // ───────────────────────────── return (§5.6, R21) ─────────────────────────────

    /// The bids of the frame's activations (R21): every function of the table that fits each one's
    /// neighborhood, in the cover or not (D31). A function fits where it pays there with nothing taken: one
    /// that does not cannot pay on the board, where less is free and the price is the same (R22).
    pub(crate) fn bids(&self, first_new: usize, ctx: Ctx) -> Vec<Bid> {
        let mut out = Vec::new();
        let wf = self.which_function_bits();
        for row in &self.history[first_new..] {
            let free = vec![false; row.nbs.len()];
            for f in &self.functions {
                let Some(c) = self.measure(row, f, &free, false, wf) else { continue };
                if !(c.coverage > c.price) { continue; }
                let mut covers: Vec<ActRef> = Vec::new();
                let mut binds: Vec<(usize, ActRef, NeuronId)> = Vec::new();
                for b in &c.entry.binds {
                    match f.places[b.place].0 {
                        Place::Fixed(_) => { for &j in &b.nbs { let a = row.nbs[j].act; if !covers.contains(&a) { covers.push(a); } } }
                        _ => { for &j in &b.nbs { binds.push((b.place, row.nbs[j].act, row.nbs[j].neuron)); } }
                    }
                }
                out.push(Bid {
                    owner: self.id, bidder: row.act, bidder_pos: row.pos, bidder_cost: row.own_cost,
                    function: f.id, child: f.child, covers, binds, failed: c.entry.failed,
                    p_size: f.p_size().max(1) as u32, which_function_bits: wf, choice: c.entry.choice,
                });
            }
        }
        let _ = ctx;
        out
    }
}
