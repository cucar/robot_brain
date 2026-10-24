//! Bids (D31, R21): what a neuron returns for each of the frame's activations, and what it retired.

use crate::neuron::*;
use crate::types::*;

/// One bid (D31): a function of the table that fits an activation's neighborhood, whole. It carries its
/// function, as it stands in the table, and its child; the machine resolves where each node stands and what
/// each variable binds on the board (R22).
#[derive(Clone, Debug)]
pub struct Bid {
    pub owner: NeuronId,
    pub bidder: ActRef,
    pub bidder_pos: Pos,
    pub function: Function,
    /// The members of each class the function uses, as the table holds them (D41).
    pub classes: Vec<(PatternId, Vec<NeuronId>)>,
    pub child: Option<NeuronId>,
}

/// What the call returns about a function it retired (§5.7).
#[derive(Clone, Debug)]
pub struct Retired {
    pub pattern: PatternId,
    pub child: Option<NeuronId>,
    /// The retired function itself: the ledger holds the function, not a handle to it, so what it fired
    /// can still be expanded until its windows close (R38).
    pub line: Option<Function>,
    /// Where merging retired it, the function of the table that took its place and its child (§5.5).
    pub heir: Option<PatternId>,
}

impl Neuron {
    // ───────────────────────────── return (§5.7, R21) ─────────────────────────────

    /// The bids of the frame's activations (R21): every function of the table that fits each one's row (D31,
    /// D54).
    pub(crate) fn bids(&self, first_new: usize, ctx: Ctx) -> Vec<Bid> {
        let mut out = Vec::new();
        for row in &self.history[first_new..] {
            for f in &self.functions {
                if !self.fits(row, f, ctx) { continue; }
                let classes: Vec<(PatternId, Vec<NeuronId>)> = f.places.iter().filter_map(|(p, _)| if let Place::Class(k) = p { self.class(*k).map(|c| (*k, c.members.iter().map(|m| m.neuron).collect())) } else { None }).collect();
                let mut function = f.clone();
                function.uses.clear();
                out.push(Bid { owner: self.id, bidder: row.act, bidder_pos: row.pos, function, classes, child: f.child });
            }
        }
        out
    }
}
