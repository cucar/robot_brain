//! The election (R22, R24) and the children it creates and places (D53, R43).


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// A child to fire in the next wave, placed over what its bid covered (D53).
pub(crate) struct Pending {
    pub(crate) act: Act,
    pub(crate) covered: Vec<ActRef>,
}

impl Machine {
    // ───────────────────────────── election (R22, R24) ─────────────────────────────

    /// What a bid covers on the board and what it costs (R22).
    pub(crate) fn measure_bid(&self, bid: &Bid) -> (f64, f64, Vec<ActRef>, bool) {
        let mut covered: Vec<ActRef> = Vec::new();
        let mut coverage = 0.0;
        let mut strong = false;
        for (_, _, acts) in &bid.nodes {
            for r in acts {
                if let Some(a) = self.act(*r) {
                    if a.strong { strong = true; }
                    if a.covered_at.is_none() && !covered.contains(r) { covered.push(*r); coverage += a.cost; }
                }
            }
        }
        if bid.names_owner {
            if let Some(a) = self.act(bid.bidder) {
                if a.strong { strong = true; }
                if a.covered_at.is_none() { covered.push(bid.bidder); coverage += a.cost; }
            }
        }
        let price = bid.bidder_cost + bid.which_pattern_bits + bid.choice_bits;
        (coverage, price, covered, strong)
    }

    pub(crate) fn bid_before(&self, a: &Bid, b: &Bid) -> bool {
        match (a.child, b.child) {
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(x), Some(y)) if x != y => return x < y,
            _ => {}
        }
        if a.owner != b.owner { return a.owner < b.owner; }
        a.bidder_pos < b.bidder_pos
    }

    /// D28 over the board (R24), then children for the accepted bids (R43).
    pub(crate) fn elect(&mut self, wave: u32, bids: Vec<Bid>, pending: &mut Vec<Pending>) -> (usize, usize, usize) {
        let fno = self.frame_no;
        let mut remaining: Vec<Bid> = bids;
        let mut accepted = 0;
        let mut minted = 0;
        let mut reused = 0;
        loop {
            let mut best: Option<(usize, f64, f64)> = None;
            for (i, b) in remaining.iter().enumerate() {
                let (cov, price, _, _) = self.measure_bid(b);
                if !(cov > price) { continue; }
                let ratio = cov / price;
                let better = match best {
                    None => true,
                    Some((j, bc, bp)) => { let br = bc / bp; ratio > br || (ratio == br && self.bid_before(b, &remaining[j])) }
                };
                if better { best = Some((i, cov, price)); }
            }
            let Some((i, _, _)) = best else { break };
            let bid = remaining.remove(i);
            let (cov, price, covered, strong) = self.measure_bid(&bid);
            if self.trace { eprintln!("[accept f{} w{}] #{} p{} at {:?} cov {:.1} price {:.1} failed {} covers {}", fno, wave, bid.owner, bid.pattern, bid.bidder_pos, cov, price, bid.failed, covered.len()); }
            accepted += 1;
            // credit: covered at this frame, once (R23)
            for r in &covered {
                if let Some(a) = self.act_mut(*r) { if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); } }
            }
            // the child, reused or new (R43, D36)
            let (kids, m, u) = self.children_for(&bid, &covered, strong, wave + 1);
            minted += m;
            reused += u;
            pending.extend(kids);
        }
        (accepted, minted, reused)
    }

    /// The call neuron and value neurons an accepted bid fires, each placed over what it covered (D53).
    pub(crate) fn children_for(&mut self, bid: &Bid, covered: &[ActRef], strong: bool, wave: u32) -> (Vec<Pending>, usize, usize) {
        let mut out = Vec::new();
        let mut minted = 0;
        let mut reused = 0;
        let owner_axes = self.neurons[bid.owner as usize].axes;
        let height = 1 + bid.nodes.iter().flat_map(|(_, _, acts)| acts.iter()).filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        // value neurons first, so their ids are known to the child's values
        let mut values: Vec<(PatternId, NeuronId, Option<NeuronId>)> = Vec::new();
        for v in &bid.values {
            let vn = match v.value_neuron {
                Some(id) if self.neurons[id as usize].alive => id,
                _ => {
                    let key = self.value_key(bid.owner, v.var, v.member);
                    match self.index.get(&key).copied().filter(|id| self.neurons[*id as usize].alive) {
                        Some(id) => { reused += 1; id }
                        None => { minted += 1; self.mint(owner_axes, key, Some(Value { owner: bid.owner, var: v.var, member: self.follow(v.member) })) }
                    }
                }
            };
            self.neurons[bid.owner as usize].wire_value(v.var, v.member, vn);
            if !self.neurons[vn as usize].parents.contains(&(bid.owner, v.var)) { self.neurons[vn as usize].parents.push((bid.owner, v.var)); }
            values.push((v.var, v.member, Some(vn)));
            // placed: a class's where its member stood, a parameter's over its positions
            let stood: Vec<&Act> = v.stood.iter().filter_map(|r| self.act(*r)).collect();
            if stood.is_empty() { continue; }
            let (pos, span) = self.place_over(&stood, owner_axes, vn);
            let vstrong = stood.iter().any(|a| a.strong);
            let fb = FiredBy { owner: bid.owner, pattern: v.var, values: vec![] };
            let act = self.new_act(vn, pos, v.choice.max(0.0), vstrong, Some(fb), span, wave, height);
            out.push(Pending { act, covered: if bid.is_function { vec![] } else { covered.to_vec() } });
        }
        if bid.is_function {
            let child = match bid.child {
                Some(c) if self.neurons[c as usize].alive => c,
                _ => {
                    let key = self.child_key(bid.owner, bid.pattern);
                    match self.index.get(&key).copied().filter(|id| self.neurons[*id as usize].alive && self.neurons[*id as usize].value.is_none()) {
                        Some(id) => { reused += 1; id }
                        None => { minted += 1; self.mint(owner_axes, key, None) }
                    }
                }
            };
            self.neurons[bid.owner as usize].wire_child(bid.pattern, child);
            if !self.neurons[child as usize].parents.contains(&(bid.owner, bid.pattern)) { self.neurons[child as usize].parents.push((bid.owner, bid.pattern)); }
            let mut over: Vec<&Act> = bid.nodes.iter().flat_map(|(_, _, acts)| acts.iter()).filter_map(|r| self.act(*r)).collect();
            if bid.names_owner { if let Some(a) = self.act(bid.bidder) { over.push(a); } }
            if over.is_empty() { return (out, minted, reused); }
            let (mut pos, span) = self.place_over(&over, owner_axes, child);
            pos[0] = bid.place[0];
            let cost = bid.bidder_cost + bid.which_pattern_bits;
            let fb = FiredBy { owner: bid.owner, pattern: bid.pattern, values };
            let act = self.new_act(child, pos, cost, strong, Some(fb), span, wave, height);
            if self.trace { eprintln!("[fire f{}] #{} at {:?} from #{} p{} cost {:.1} span {:?} {}", self.frame_no, child, pos, bid.owner, bid.pattern, cost, span, if minted > 0 { "minted" } else { "" }); }
            out.push(Pending { act, covered: covered.to_vec() });
        }
        (out, minted, reused)
    }

    /// Where a child stands over a set of activations (D53): in time at the latest, in every other
    /// dimension at the center of their box, rounded toward the smaller coordinate on a half; and its
    /// span (D4), the farthest base activation under it.
    pub(crate) fn place_over(&self, over: &[&Act], axes: u8, child: NeuronId) -> (Pos, Pos) {
        let mut pos = [0i32; MAX_DIMS];
        pos[0] = over.iter().map(|a| a.pos[0]).max().unwrap_or(0);
        for d in 1..self.radii.len() {
            if axes & (1 << d) == 0 { continue; }
            let lo = over.iter().map(|a| a.pos[d]).min().unwrap_or(0);
            let hi = over.iter().map(|a| a.pos[d]).max().unwrap_or(0);
            pos[d] = lo + (hi - lo) / 2;
        }
        // the span is read by the descent expansion makes, once through each neuron, so a chain of
        // children closed on itself spans one turn and not the run (D4)
        let mut span = [0i32; MAX_DIMS];
        for a in over {
            let inner = self.span_by_descent(a.neuron, &mut vec![child]);
            for d in 0..self.radii.len() {
                let s = (a.pos[d] - pos[d]).abs() + inner[d];
                if s > span[d] { span[d] = s; }
            }
        }
        (pos, span)
    }

    /// How far the farthest base symbol under a neuron lies from it, read down its first parent's line
    /// once through each neuron (D4).
    pub(crate) fn span_by_descent(&self, n: NeuronId, path: &mut Vec<NeuronId>) -> Pos {
        let mut span = [0i32; MAX_DIMS];
        let nn = &self.neurons[n as usize];
        if nn.kind != Kind::Inter || !nn.alive || path.contains(&n) || path.len() > 64 { return span; }
        path.push(n);
        if let Some(v) = nn.value {
            if let Some(p) = self.neurons[v.owner as usize].pattern(v.var) {
                let inner = self.span_by_descent(v.member, path);
                let offsets: Vec<Offset> = match &p.body { Body::Parameter { offsets, .. } => offsets.clone(), _ => vec![[0; MAX_DIMS]] };
                for o in offsets { for d in 0..MAX_DIMS { span[d] = span[d].max(o[d].abs() + inner[d]); } }
            }
        } else if let Some(&(owner, pid)) = nn.parents.first() {
            if let Some(nodes) = self.neurons[owner as usize].function_nodes(pid) {
                let nodes: Vec<(Node, Offset)> = nodes.to_vec();
                for (node, off) in nodes {
                    let inner = match node {
                        Node::Owner => self.span_by_descent(owner, path),
                        Node::Neuron(m) => self.span_by_descent(m, path),
                        Node::Var(v) => {
                            let member = self.neurons[owner as usize].pattern(v).and_then(|q| q.frequent_member());
                            member.map(|m| self.span_by_descent(m, path)).unwrap_or([0; MAX_DIMS])
                        }
                    };
                    for d in 0..MAX_DIMS { span[d] = span[d].max(off[d].abs() + inner[d]); }
                }
            }
        }
        path.pop();
        span
    }

    pub(crate) fn mint(&mut self, axes: u8, key: String, value: Option<Value>) -> NeuronId {
        let id = self.neurons.len() as NeuronId;
        let mut n = Neuron::inter(id, axes, self.cfg.h, self.radii.clone());
        n.key = key.clone();
        n.value = value;
        self.neurons.push(n);
        self.spans.push([0; MAX_DIMS]);
        self.alive += 1;
        self.index.insert(key, id);
        id
    }

    /// A value followed through value neurons to a neuron that is not one (D45).
    pub(crate) fn follow(&self, mut n: NeuronId) -> NeuronId {
        for _ in 0..64 {
            match self.neurons[n as usize].value { Some(v) => n = v.member, None => break }
        }
        n
    }

}
