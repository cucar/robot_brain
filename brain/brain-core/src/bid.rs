//! Bids (D31, R21): what a neuron returns from its covers, and what it retired.


use crate::neuron::*;
use crate::types::*;

/// One bid (D31): what one activation offers from its cover. A function goes with the variables it
/// names, and functions of one cover that name the same variable go together as one bid, the variable
/// carried once; a parameter no function of the cover names goes alone.
#[derive(Clone, Debug)]
pub struct Bid {
    pub owner: NeuronId,
    pub bidder: ActRef,
    pub bidder_pos: Pos,
    pub bidder_cost: f64,
    /// The functions carried, or the lone parameter.
    pub parts: Vec<BidPart>,
    /// Per variable carried, per relation naming it: the member that stood, where, and its value neuron.
    pub values: Vec<BidValue>,
    pub which_pattern_bits: f64,
}

/// One function of a bid, or a parameter bid alone.
#[derive(Clone, Debug)]
pub struct BidPart {
    pub pattern: PatternId,
    pub is_function: bool,
    pub child: Option<NeuronId>,
    pub names_owner: bool,
    /// Absolute coordinate of the reference node (D53: the child's frame is this one's).
    pub place: Pos,
    /// Every node with its offset from the reference and the activations that stood there.
    pub nodes: Vec<(Node, Offset, Vec<ActRef>)>,
    /// The nodes that did not hold, and |p| (D22).
    pub failed: u32,
    pub p_size: u32,
    /// Indices into the bid's values: the variables this part names.
    pub values: Vec<usize>,
}

impl Bid {
    /// Whether any function carried names the owner: the bid then covers the bidder (D31).
    pub fn names_owner(&self) -> bool { self.parts.iter().any(|p| p.names_owner) }
    /// The child the tie-break reads (R24): the first function's.
    pub fn child(&self) -> Option<NeuronId> { self.parts.first().and_then(|p| p.child) }
    pub fn is_function(&self) -> bool { self.parts.first().map_or(false, |p| p.is_function) }
    /// What each variable held: its choice, once per relation naming it (D13).
    pub fn choice_bits(&self) -> f64 { self.values.iter().map(|v| v.choice).sum() }
    /// Which neighbor failed, per part (D13, D22).
    pub fn failure_bits(&self) -> f64 { self.parts.iter().map(|p| p.failed as f64 * log2(p.p_size.max(1) as f64)).sum() }
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
    /// A member left a variable that stays: only its value neuron is released (R38).
    pub member_only: bool,
    /// The retired line itself: the ledger holds the pattern, not a handle to it, so what it fired can
    /// still be expanded until its windows close (R38).
    pub line: Option<Pattern>,
}

/// The measure of one claimant against one row in one round of the greedy cover (D22, D28).

impl Neuron {
    // ───────────────────────────── return (§5.6, R21) ─────────────────────────────

    pub(crate) fn bids(&self, first_new: usize) -> Vec<Bid> {
        let mut out = Vec::new();
        let wp = self.which_pattern_bits();
        for row in &self.history[first_new..] {
            // the cover's entries, grouped: functions that name a variable in common go as one bid
            let n = row.cover.len();
            let mut group: Vec<usize> = (0..n).collect();
            fn find(g: &mut Vec<usize>, x: usize) -> usize { let mut r = x; while g[r] != r { r = g[r]; } let mut y = x; while g[y] != r { let z = g[y]; g[y] = r; y = z; } r }
            let vars_of = |e: &CoverEntry| -> Vec<PatternId> { e.nodes.iter().filter_map(|nb| nb.var.as_ref().map(|v| v.var)).collect() };
            for a in 0..n {
                if !self.pattern(row.cover[a].pattern).map_or(false, |p| p.is_function()) { continue; }
                for b in (a + 1)..n {
                    if !self.pattern(row.cover[b].pattern).map_or(false, |p| p.is_function()) { continue; }
                    let va = vars_of(&row.cover[a]);
                    if vars_of(&row.cover[b]).iter().any(|v| va.contains(v)) {
                        let (ra, rb) = (find(&mut group, a), find(&mut group, b));
                        if ra != rb { group[rb.max(ra)] = ra.min(rb); }
                    }
                }
            }
            let mut roots: Vec<usize> = Vec::new();
            for a in 0..n { let r = find(&mut group, a); if !roots.contains(&r) { roots.push(r); } }
            for r in roots {
                let mut parts: Vec<BidPart> = Vec::new();
                let mut values: Vec<BidValue> = Vec::new();
                for a in 0..n {
                    if find(&mut group, a) != r { continue; }
                    let e = &row.cover[a];
                    let Some(p) = self.pattern(e.pattern) else { continue };
                    let mut nodes: Vec<(Node, Offset, Vec<ActRef>)> = Vec::new();
                    let mut p_size = 0u32;
                    let mut mine: Vec<usize> = Vec::new();
                    for nb in &e.nodes {
                        match &nb.var {
                            Some(v) => {
                                let Some(vp) = self.pattern(v.var) else { continue };
                                p_size += vp.positions() as u32;
                                let stood: Vec<ActRef> = v.positions.iter().flatten().map(|&j| row.nbs[j].act).collect();
                                nodes.push((nb.node, nb.offset, stood.clone()));
                                let Some(m) = v.member else { continue };
                                // the variable carried once: one value per variable per place it stood
                                let k = match values.iter().position(|x| x.var == v.var && x.stood == stood) {
                                    Some(k) => k,
                                    None => {
                                        let is_class = matches!(vp.body, Body::Class { .. });
                                        let value_neuron = vp.member(m).and_then(|mm| mm.value_neuron);
                                        values.push(BidValue { var: v.var, is_class, member: m, stood, value_neuron, choice: v.choice });
                                        values.len() - 1
                                    }
                                };
                                if !mine.contains(&k) { mine.push(k); }
                            }
                            None => { p_size += 1; nodes.push((nb.node, nb.offset, nb.nbs.iter().map(|&j| row.nbs[j].act).collect())); }
                        }
                    }
                    parts.push(BidPart { pattern: e.pattern, is_function: p.is_function(), child: p.child(), names_owner: e.names_owner, place: e.place, nodes, failed: e.failed, p_size: p_size.max(1), values: mine });
                }
                if parts.is_empty() { continue; }
                out.push(Bid { owner: self.id, bidder: row.act, bidder_pos: row.pos, bidder_cost: row.own_cost, parts, values, which_pattern_bits: wp });
            }
        }
        out
    }

}
