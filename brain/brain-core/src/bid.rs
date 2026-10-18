//! Bids (D31, R21): what a neuron returns from its covers, and what it retired.


use crate::neuron::*;
use crate::types::*;

#[derive(Clone, Debug)]
pub struct Bid {
    pub owner: NeuronId,
    pub bidder: ActRef,
    pub bidder_pos: Pos,
    pub bidder_cost: f64,
    pub pattern: PatternId,
    pub is_function: bool,
    pub child: Option<NeuronId>,
    pub names_owner: bool,
    /// Absolute coordinate of the reference node (D53: the child's frame is this one's).
    pub place: Pos,
    /// Every node with its offset from the reference and the activations that stood there.
    pub nodes: Vec<(Node, Offset, Vec<ActRef>)>,
    /// Per variable named: the member that stood, where (one activation per position that held), and
    /// its value neuron if the variable has one.
    pub values: Vec<BidValue>,
    pub failed: u32,
    pub p_size: u32,
    pub which_pattern_bits: f64,
    pub choice_bits: f64,
}

#[derive(Clone, Debug)]
pub struct BidValue {
    pub var: PatternId,
    pub is_class: bool,
    pub member: NeuronId,
    pub stood: Vec<ActRef>,
    pub value_neuron: Option<NeuronId>,
    pub choice: f64,
}

/// What the call returns about a pattern it retired (§5.6).
#[derive(Clone, Debug)]
pub struct Retired {
    pub pattern: PatternId,
    pub child: Option<NeuronId>,
    pub value_neurons: Vec<NeuronId>,
}

/// The measure of one claimant against one row in one round of the greedy cover (D22, D28).

impl Neuron {
    // ───────────────────────────── return (§5.6, R21) ─────────────────────────────

    pub(crate) fn bids(&self, first_new: usize) -> Vec<Bid> {
        let mut out = Vec::new();
        let wp = self.which_pattern_bits();
        for row in &self.history[first_new..] {
            for e in &row.cover {
                let Some(p) = self.pattern(e.pattern) else { continue };
                let is_function = p.is_function();
                let mut nodes: Vec<(Node, Offset, Vec<ActRef>)> = Vec::new();
                let mut values: Vec<BidValue> = Vec::new();
                let mut p_size = 0u32;
                let mut choice_bits = 0.0;
                for nb in &e.nodes {
                    let acts: Vec<ActRef> = nb.nbs.iter().map(|&j| row.nbs[j].act).collect();
                    match &nb.var {
                        Some(v) => {
                            let Some(vp) = self.pattern(v.var) else { continue };
                            let is_class = matches!(vp.body, Body::Class { .. });
                            p_size += vp.positions() as u32;
                            choice_bits += v.choice;
                            let stood: Vec<ActRef> = v.positions.iter().flatten().map(|&j| row.pos_act(j)).collect();
                            let value_neuron = v.member.and_then(|m| vp.member(m)).and_then(|m| m.value_neuron);
                            if let Some(m) = v.member { values.push(BidValue { var: v.var, is_class, member: m, stood, value_neuron, choice: v.choice }); }
                            nodes.push((nb.node, nb.offset, v.positions.iter().flatten().filter(|&&j| j != OWNER).map(|&j| row.nbs[j].act).collect()));
                        }
                        None => { p_size += 1; nodes.push((nb.node, nb.offset, acts)); }
                    }
                }
                out.push(Bid {
                    owner: self.id, bidder: row.act, bidder_pos: row.pos, bidder_cost: row.own_cost,
                    pattern: e.pattern, is_function, child: p.child(), names_owner: e.names_owner,
                    place: e.place, nodes, values, failed: e.failed, p_size: p_size.max(1),
                    which_pattern_bits: wp, choice_bits,
                });
            }
        }
        out
    }

}
