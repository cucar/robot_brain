//! Recognition (§5.2): the greedy cover (D28) of a row by the table, each claimant measured (D22).

use rustc_hash::{FxHashSet};

use crate::neuron::*;
use crate::types::*;

pub(crate) struct Claim {
    pub(crate) entry: CoverEntry,
    pub(crate) coverage: f64,
    pub(crate) price: f64,
    pub(crate) is_function: bool,
}


impl Neuron {
    // ───────────────────────────── recognition (§5.2, D28) ─────────────────────────────

    /// Cover every dirty row by the greedy cover over the table. Returns the patterns whose uses changed.
    pub(crate) fn recognize(&mut self, ctx: Ctx) -> FxHashSet<PatternId> {
        let mut changed: FxHashSet<PatternId> = FxHashSet::default();
        for i in 0..self.history.len() {
            if !self.history[i].dirty { continue; }
            for e in &self.history[i].cover { changed.insert(e.pattern); }
            let cover = self.greedy_cover(i, ctx);
            self.install_cover(i, cover);
            for e in &self.history[i].cover { changed.insert(e.pattern); }
            self.history[i].dirty = false;
        }
        changed
    }

    pub(crate) fn install_cover(&mut self, i: usize, cover: Vec<CoverEntry>) {
        let row = &mut self.history[i];
        let n = row.nbs.len();
        row.owner = vec![None; n];
        row.claim = vec![None; n];
        row.owner_self = None;
        for e in &cover {
            if e.credited_owner && row.owner_self.is_none() { row.owner_self = Some(e.pattern); }
            for nb in &e.nodes {
                for &j in &nb.nbs { if row.owner[j].is_none() { row.owner[j] = Some(e.pattern); } }
                if let Some(v) = &nb.var {
                    for (k, pj) in v.positions.iter().enumerate() {
                        if let Some(j) = pj {
                            if row.owner[*j].is_none() { row.owner[*j] = Some(e.pattern); }
                            if row.claim[*j].is_none() { row.claim[*j] = Some((v.var, k)); }
                        }
                    }
                }
            }
        }
        row.cover = cover;
    }

    /// The greedy cover of one row (D28): functions with their variables first, then parameters alone
    /// on what no function took. `exclude` leaves one pattern out, for the margin's "without" (D30).
    pub(crate) fn greedy_cover(&self, i: usize, ctx: Ctx) -> Vec<CoverEntry> { self.greedy_cover_without(i, None, ctx) }

    pub(crate) fn greedy_cover_without(&self, i: usize, exclude: Option<PatternId>, ctx: Ctx) -> Vec<CoverEntry> {
        let row = &self.history[i];
        let mut taken = vec![false; row.nbs.len()];
        let mut owner_taken = false;
        let mut cover: Vec<CoverEntry> = Vec::new();
        let wp = self.which_pattern_bits();
        for phase in 0..2 {
            loop {
                let mut best: Option<Claim> = None;
                for p in &self.table {
                    if Some(p.id) == exclude { continue; }
                    if (phase == 0) != p.is_function() { continue; }
                    if phase == 1 && !matches!(p.body, Body::Parameter { .. }) { continue; }
                    if phase == 1 && cover.iter().any(|e| e.nodes.iter().any(|nb| nb.var.as_ref().map_or(false, |v| v.var == p.id))) { continue; }
                    for c in self.claimants(row, p, &taken, owner_taken, wp, ctx) {
                        let better = match &best {
                            None => true,
                            Some(b) => {
                                let r = c.coverage / c.price;
                                let br = b.coverage / b.price;
                                r > br || (r == br && (c.entry.pattern < b.entry.pattern || (c.entry.pattern == b.entry.pattern && c.entry.place < b.entry.place)))
                            }
                        };
                        if better { best = Some(c); }
                    }
                }
                let Some(b) = best else { break };
                if !(b.coverage > b.price) { break; }
                for nb in &b.entry.nodes {
                    for &j in &nb.nbs { taken[j] = true; }
                    if let Some(v) = &nb.var { for pj in v.positions.iter().flatten() { taken[*pj] = true; } }
                }
                if b.entry.credited_owner { owner_taken = true; }
                let _ = b.is_function;
                cover.push(b.entry);
            }
        }
        cover
    }

    /// Every placement of one pattern in a row, measured against the still-uncovered neighbors (D28).
    pub(crate) fn claimants(&self, row: &Row, p: &Pattern, taken: &[bool], owner_taken: bool, wp: f64, ctx: Ctx) -> Vec<Claim> {
        let mut out = Vec::new();
        match &p.body {
            Body::Function { nodes, .. } => {
                // a function left with fewer than two nodes covers nothing; retire removes it (R41)
                if nodes.len() < 2 { return out; }
                let names_owner = matches!(nodes.first(), Some((Node::Owner, _)));
                let anchors: Vec<Pos> = if names_owner { vec![[0; MAX_DIMS]] } else { self.anchors_for(row, &nodes[0].0, taken) };
                for anchor in anchors {
                    if let Some(c) = self.fit_function(row, p, nodes, anchor, taken, owner_taken, wp, ctx) { out.push(c); }
                }
            }
            Body::Parameter { offsets, .. } => {
                // alone: its first position at any residual neighbor (D31)
                let mut seen: FxHashSet<Pos> = FxHashSet::default();
                let mut anchors: Vec<Pos> = Vec::new();
                for j in 0..row.nbs.len() { if !taken[j] && row.residual(j) { anchors.push(row.nbs[j].delta); } }
                for anchor in anchors {
                    if !seen.insert(anchor) { continue; }
                    let Some(vb) = self.bind_parameter(row, p, offsets, &anchor, taken) else { continue };
                    if vb.failed > 0 { continue; }
                    let mut coverage = 0.0;
                    let credited_owner = false;
                    for pj in vb.positions.iter().flatten() { if !taken[*pj] { coverage += row.credit(*pj); } }
                    let price = row.own_cost + wp + vb.choice;
                    let failed = vb.failed;
                    let entry = CoverEntry { pattern: p.id, place: add(&row.pos, &anchor), nodes: vec![NodeBind { node: Node::Var(p.id), offset: [0; MAX_DIMS], nbs: Vec::new(), var: Some(vb) }], failed, credited_owner, names_owner: credited_owner };
                    out.push(Claim { entry, coverage, price, is_function: false });
                }
            }
            Body::Class { members, anchor } => {
                // alone: at its position from the owner, or from each neighbor of its anchor's neuron (D31, D41)
                let targets: Vec<Pos> = match anchor.0 {
                    Node::Owner => vec![anchor.1],
                    Node::Neuron(n) => (0..row.nbs.len()).filter(|&k| row.nbs[k].neuron == n).map(|k| add(&row.nbs[k].delta, &anchor.1)).collect(),
                    Node::Var(_) => Vec::new(),
                };
                let mut seen: FxHashSet<Pos> = FxHashSet::default();
                for target in targets {
                    if !seen.insert(target) { continue; }
                    let found: Vec<usize> = self.nbs_at(row, &target, None).into_iter().filter(|&j| !taken[j] && row.residual(j) && members.iter().any(|m| m.neuron == row.nbs[j].neuron)).collect();
                    let Some(&j) = found.first() else { continue };
                    let choice = p.choice();
                    let coverage = row.credit(j);
                    let price = row.own_cost + wp + choice;
                    let entry = CoverEntry { pattern: p.id, place: add(&row.pos, &target), nodes: vec![NodeBind { node: Node::Var(p.id), offset: target, nbs: Vec::new(), var: Some(VarBind { var: p.id, member: Some(row.nbs[j].neuron), positions: vec![Some(j)], failed: 0, choice }) }], failed: 0, credited_owner: false, names_owner: false };
                    out.push(Claim { entry, coverage, price, is_function: false });
                }
            }
        }
        let _ = ctx;
        out
    }

    /// Where a floating function's reference node may stand in a row (D15): every neighbor of that
    /// neuron, or holding a member of that variable.
    pub(crate) fn anchors_for(&self, row: &Row, reference: &Node, taken: &[bool]) -> Vec<Pos> {
        let mut out: Vec<Pos> = Vec::new();
        for j in 0..row.nbs.len() {
            if taken[j] { continue; }
            let ok = match reference {
                Node::Neuron(n) => row.nbs[j].neuron == *n,
                Node::Var(v) => match self.pattern(*v) {
                    Some(Pattern { body: Body::Class { members, .. }, .. }) => members.iter().any(|m| m.neuron == row.nbs[j].neuron),
                    Some(Pattern { body: Body::Parameter { .. }, .. }) => true,
                    _ => false,
                },
                Node::Owner => false,
            };
            if ok && !out.contains(&row.nbs[j].delta) { out.push(row.nbs[j].delta); }
        }
        out
    }

    /// Fit a function at one anchor (the reference node's offset from the owner's activation): each
    /// node holds or fails (D22), and the measure is over the neighbors not yet taken (D28).
    pub(crate) fn fit_function(&self, row: &Row, p: &Pattern, nodes: &[(Node, Offset)], anchor: Pos, taken: &[bool], owner_taken: bool, wp: f64, ctx: Ctx) -> Option<Claim> {
        let _ = ctx;
        let mut binds: Vec<NodeBind> = Vec::with_capacity(nodes.len());
        let mut failed = 0u32;
        let mut coverage = 0.0;
        let mut choices = 0.0;
        let mut p_size = 0u32;
        let mut credited_owner = false;
        let mut names_owner = false;
        let mut holds_any = false;
        // a neighbor bound by several nodes, a child standing for several parents, is credited once
        let mut credited: FxHashSet<usize> = FxHashSet::default();
        for (node, off) in nodes {
            let target = add(&anchor, off);
            match node {
                Node::Owner => {
                    names_owner = true;
                    p_size += 1;
                    if target != [0; MAX_DIMS] { failed += 1; }
                    else {
                        holds_any = true;
                        if !owner_taken { credited_owner = true; coverage += row.own_cost; }
                    }
                    binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: None });
                }
                Node::Neuron(n) => {
                    p_size += 1;
                    let found = self.nbs_at(row, &target, Some(*n));
                    if found.is_empty() { failed += 1; }
                    else {
                        holds_any = true;
                        for &j in &found { if !taken[j] && credited.insert(j) { coverage += row.credit(j); } }
                    }
                    binds.push(NodeBind { node: *node, offset: *off, nbs: found, var: None });
                }
                Node::Var(v) => {
                    let Some(vp) = self.pattern(*v) else { failed += 1; p_size += 1; binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: None }); continue };
                    match &vp.body {
                        Body::Class { members, .. } => {
                            p_size += 1;
                            let found: Vec<usize> = self.nbs_at(row, &target, None).into_iter().filter(|&j| members.iter().any(|m| m.neuron == row.nbs[j].neuron)).collect();
                            let choice = vp.choice();
                            choices += choice;
                            if found.is_empty() {
                                failed += 1;
                                binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: Some(VarBind { var: *v, member: None, positions: vec![None], failed: 1, choice }) });
                            } else {
                                holds_any = true;
                                let j = found[0];
                                if !taken[j] && credited.insert(j) { coverage += row.credit(j); }
                                binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: Some(VarBind { var: *v, member: Some(row.nbs[j].neuron), positions: vec![Some(j)], failed: 0, choice }) });
                            }
                        }
                        Body::Parameter { offsets, .. } => {
                            p_size += offsets.len() as u32;
                            let choice = vp.choice();
                            choices += choice;
                            match self.bind_parameter(row, vp, offsets, &target, taken) {
                                Some(vb) => {
                                    if vb.member.is_some() { holds_any = true; }
                                    failed += vb.failed;
                                    for pj in vb.positions.iter().flatten() { if !taken[*pj] && credited.insert(*pj) { coverage += row.credit(*pj); } }
                                    binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: Some(vb) });
                                }
                                None => {
                                    failed += offsets.len() as u32;
                                    binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: Some(VarBind { var: *v, member: None, positions: vec![None; offsets.len()], failed: offsets.len() as u32, choice }) });
                                }
                            }
                        }
                        Body::Function { .. } => { failed += 1; p_size += 1; binds.push(NodeBind { node: *node, offset: *off, nbs: Vec::new(), var: None }); }
                    }
                }
            }
        }
        // exact holding (D15): a node that fails means the function does not hold here
        if !holds_any || failed > 0 { return None; }
        let _ = p_size;
        let price = row.own_cost + wp + choices;
        let entry = CoverEntry { pattern: p.id, place: add(&row.pos, &anchor), nodes: binds, failed, credited_owner, names_owner };
        Some(Claim { entry, coverage, price, is_function: true })
    }

    /// The neighbors standing at a target offset from the owner's activation: every one whose exact
    /// offset rounds into the target's group (D26), of the given neuron where one is given.
    pub(crate) fn nbs_at(&self, row: &Row, target: &Pos, neuron: Option<NeuronId>) -> Vec<usize> {
        let tr = self.round(target);
        let mut out = Vec::new();
        for (j, nb) in row.nbs.iter().enumerate() {
            if neuron.map_or(false, |n| nb.neuron != n) { continue; }
            if self.round(&nb.delta) == tr { out.push(j); }
        }
        // the parent named may be matched by the child that covers it
        if out.is_empty() {
            if let Some(n) = neuron {
                for (j, nb) in row.nbs.iter().enumerate() {
                    if nb.covers.iter().any(|(m, d)| *m == n && self.round(d) == tr) { out.push(j); }
                }
            }
        }
        out
    }

    /// Bind a parameter with its first position at `anchor` (D44): one neighbor per position, the value
    /// the neuron at most of them, the latest on a tie, every other position a failed neighbor.
    pub(crate) fn bind_parameter(&self, row: &Row, vp: &Pattern, offsets: &[Offset], anchor: &Pos, taken: &[bool]) -> Option<VarBind> {
        let mut per_pos: Vec<Option<usize>> = Vec::with_capacity(offsets.len());
        for off in offsets {
            let target = add(anchor, off);
            let found = self.nbs_at(row, &target, None);
            // prefer an untaken residual neighbor, then any
            let pick = found.iter().copied().find(|&j| !taken[j] && row.residual(j)).or_else(|| found.first().copied());
            per_pos.push(pick);
        }
        if per_pos.iter().all(|p| p.is_none()) { return None; }
        // the value: the neuron at most positions, the latest (greatest time) on a tie
        let mut tally: Vec<(NeuronId, usize, i32)> = Vec::new();
        for pj in per_pos.iter().flatten() {
            let v = row.value_at(*pj);
            let t = row.nbs[*pj].delta[0];
            match tally.iter_mut().find(|e| e.0 == v) { Some(e) => { e.1 += 1; if t > e.2 { e.2 = t; } }, None => tally.push((v, 1, t)) }
        }
        tally.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        let value = tally[0].0;
        let mut failed = 0u32;
        let positions: Vec<Option<usize>> = per_pos.iter().map(|pj| match pj {
            Some(j) if row.value_at(*j) == value => Some(*j),
            _ => { failed += 1; None }
        }).collect();
        Some(VarBind { var: vp.id, member: Some(value), positions, failed, choice: vp.choice() })
    }

}
