//! The election (R22, R24) and the children it creates and places (D53, R43).


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// A child to fire at the next level, placed over what its bid covered (D53).
pub(crate) struct Pending {
    pub(crate) act: Act,
    pub(crate) covered: Vec<ActRef>,
}

impl Machine {
    // ───────────────────────────── election (R22, R24) ─────────────────────────────

    /// What a bid covers on the board and what it costs (R22): the activations its patterns name that
    /// fired and no earlier bid covers, the bidder among them where a function names the owner, each at
    /// its cost; and the fields its call writes: the bidder's cost, which pattern per function, what each
    /// variable held, and which named neuron did not fire.
    pub(crate) fn measure_bid(&self, bid: &Bid) -> (f64, f64, Vec<ActRef>, bool) {
        let mut covered: Vec<ActRef> = Vec::new();
        let mut coverage = 0.0;
        let mut strong = false;
        for part in &bid.parts {
            for (_, _, acts) in &part.nodes {
                for r in acts {
                    if let Some(a) = self.act(*r) {
                        if a.gone { continue; }
                        if a.strong { strong = true; }
                        if a.covered_at.is_none() && !covered.contains(r) { covered.push(*r); coverage += a.cost; }
                    }
                }
            }
        }
        if bid.names_owner() {
            if let Some(a) = self.act(bid.bidder) {
                if a.strong { strong = true; }
                if a.covered_at.is_none() && !covered.contains(&bid.bidder) { covered.push(bid.bidder); coverage += a.cost; }
            }
        }
        let functions = bid.parts.iter().filter(|p| p.is_function).count().max(1) as f64;
        let price = bid.bidder_cost + functions * bid.which_pattern_bits + bid.choice_bits() + bid.failure_bits();
        (coverage, price, covered, strong)
    }

    /// The tie-break (R24): a bid carrying a child before one carrying none; between two with children,
    /// the older child; between two with none, the older bidder; then the earlier coordinate.
    pub(crate) fn bid_before(&self, a: &Bid, b: &Bid) -> bool {
        match (a.child(), b.child()) {
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(x), Some(y)) if x != y => return x < y,
            _ => {}
        }
        if a.owner != b.owner { return a.owner < b.owner; }
        a.bidder_pos < b.bidder_pos
    }

    /// D28 over the board (R24), then children for the accepted bids (R43).
    pub(crate) fn elect(&mut self, level: u32, bids: Vec<Bid>, pending: &mut Vec<Pending>) -> (usize, usize, usize) {
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
            if self.trace { eprintln!("[accept f{} l{}] #{} p{:?} at {:?} cov {:.1} price {:.1} failed {} covers {}", fno, level, bid.owner, bid.parts.iter().map(|p| p.pattern).collect::<Vec<_>>(), bid.bidder_pos, cov, price, bid.parts.iter().map(|p| p.failed).sum::<u32>(), covered.len()); }
            accepted += 1;
            // credit: covered at this frame, once (R23)
            for r in &covered {
                if let Some(a) = self.act_mut(*r) { if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); } }
            }
            // the child, reused or new (R43, D36)
            let (kids, m, u) = self.children_for(&bid, &covered, strong, level + 1);
            minted += m;
            reused += u;
            pending.extend(kids);
        }
        (accepted, minted, reused)
    }

    /// The call neurons and value neurons an accepted bid fires, each placed over what it covered (D53):
    /// a value neuron per variable carried, where its value stood, and the child of each function over
    /// the nodes it names, the bidder among them where it names the owner.
    pub(crate) fn children_for(&mut self, bid: &Bid, covered: &[ActRef], strong: bool, level: u32) -> (Vec<Pending>, usize, usize) {
        let mut out = Vec::new();
        let mut minted = 0;
        let mut reused = 0;
        let owner_axes = self.thalamus[bid.owner].axes;
        let height = 1 + bid.parts.iter().flat_map(|p| p.nodes.iter()).flat_map(|(_, _, acts)| acts.iter()).filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        let alone = !bid.is_function();
        // value neurons first, so their ids are known to the children's values
        let mut values: Vec<(PatternId, NeuronId, Option<NeuronId>)> = Vec::new();
        for v in &bid.values {
            let vn = match v.value_neuron {
                Some(id) if self.thalamus[id].alive => id,
                _ => {
                    let key = self.value_key(bid.owner, v.var, v.member);
                    match self.index.get(&key).copied().filter(|id| self.thalamus[*id].alive) {
                        Some(id) => { reused += 1; id }
                        None => { minted += 1; self.mint(owner_axes, key, Some(Value { owner: bid.owner, var: v.var, member: self.follow(v.member) })) }
                    }
                }
            };
            self.thalamus[bid.owner].wire_value(v.var, v.member, vn);
            if !self.thalamus[vn].parents.contains(&(bid.owner, v.var)) { self.thalamus[vn].parents.push((bid.owner, v.var)); }
            values.push((v.var, v.member, Some(vn)));
            // placed: a class's where its member stood, a parameter's over its positions
            let stood: Vec<&Act> = v.stood.iter().filter_map(|r| self.act(*r)).collect();
            if stood.is_empty() { continue; }
            let at: Vec<(Pos, Pos)> = stood.iter().map(|a| (a.pos, a.span)).collect();
            let (pos, span) = self.place_over(&at, owner_axes);
            let vstrong = stood.iter().any(|a| a.strong);
            // a value neuron beside a child costs its choice; a parameter's alone, the whole bid (D13)
            let cost = if alone { bid.bidder_cost + bid.which_pattern_bits + v.choice + bid.failure_bits() } else { v.choice.max(0.0) };
            let fb = FiredBy { owner: bid.owner, pattern: v.var, values: vec![] };
            let act = self.new_act(vn, pos, cost, vstrong, Some(fb), span, level, height);
            out.push(Pending { act, covered: if alone { covered.to_vec() } else { vec![] } });
        }
        for part in bid.parts.iter().filter(|p| p.is_function) {
            let child = match part.child {
                Some(c) if self.thalamus[c].alive => c,
                _ => {
                    let key = self.child_key(bid.owner, part.pattern);
                    match self.index.get(&key).copied().filter(|id| self.thalamus[*id].alive && self.thalamus[*id].value.is_none()) {
                        Some(id) => { reused += 1; id }
                        None => { minted += 1; self.mint(owner_axes, key, None) }
                    }
                }
            };
            self.thalamus[bid.owner].wire_child(part.pattern, child);
            if !self.thalamus[child].parents.contains(&(bid.owner, part.pattern)) { self.thalamus[child].parents.push((bid.owner, part.pattern)); }
            let mut mine: Vec<ActRef> = part.nodes.iter().flat_map(|(_, _, acts)| acts.iter().copied()).collect();
            if part.names_owner { mine.push(bid.bidder); }
            // where each node it names stood, and that node's span (D4, D53): a child standing in for the node
            // named is read as that node, at the coordinate its expansion puts it, with that node's span
            let mut at: Vec<(Pos, Pos)> = Vec::new();
            for (node, off, acts) in &part.nodes {
                for r in acts {
                    let Some(b) = self.act(*r) else { continue };
                    match node {
                        Node::Neuron(n) if b.neuron != *n => {
                            let c = add(&part.place, off);
                            at.push((c, self.line_span(*n)));
                        }
                        _ => at.push((b.pos, b.span)),
                    }
                }
            }
            if part.names_owner { if let Some(b) = self.act(bid.bidder) { at.push((b.pos, b.span)); } }
            if at.is_empty() { continue; }
            let (pos, span) = self.place_over(&at, owner_axes);
            // a function's child costs what its bidder cost, which pattern and the neighbors that failed (D13)
            let cost = bid.bidder_cost + bid.which_pattern_bits + part.failed as f64 * log2(part.p_size.max(1) as f64);
            let fv: Vec<(PatternId, NeuronId, Option<NeuronId>)> = part.values.iter().filter_map(|&k| values.get(k).copied()).collect();
            let fb = FiredBy { owner: bid.owner, pattern: part.pattern, values: fv };
            let act = self.new_act(child, pos, cost, strong, Some(fb), span, level, height);
            if self.trace { eprintln!("[fire f{}] #{} at {:?} from #{} p{} cost {:.1} span {:?}", self.frame_no, child, pos, bid.owner, part.pattern, cost, span); }
            // it covers what the election credited to the bid among the nodes it names
            let cov: Vec<ActRef> = covered.iter().copied().filter(|r| mine.contains(r)).collect();
            out.push(Pending { act, covered: cov });
        }
        (out, minted, reused)
    }

    /// Where a child stands over what its bid covered (D53): in time at the latest, in every other
    /// dimension at the center of their box, rounded toward the smaller coordinate on a half; and its span
    /// (D4): the largest, over what it names, of that node's distance from it plus that node's span.
    pub(crate) fn place_over(&self, over: &[(Pos, Pos)], axes: u8) -> (Pos, Pos) {
        let mut pos = [0i32; MAX_DIMS];
        pos[0] = over.iter().map(|(p, _)| p[0]).max().unwrap_or(0);
        for d in 1..self.radii.len() {
            if axes & (1 << d) == 0 { continue; }
            let lo = over.iter().map(|(p, _)| p[d]).min().unwrap_or(0);
            let hi = over.iter().map(|(p, _)| p[d]).max().unwrap_or(0);
            pos[d] = lo + (hi - lo).div_euclid(2);
        }
        let mut span = [0i32; MAX_DIMS];
        for (p, inner) in over {
            for d in 0..self.radii.len() {
                let s = (p[d] - pos[d]).abs() + inner[d];
                if s > span[d] { span[d] = s; }
            }
        }
        (pos, span)
    }

    /// How far the farthest base symbol of a neuron's expansion lies from it (D4, R28): zero for a base
    /// neuron.
    pub(crate) fn line_span(&self, n: NeuronId) -> Pos {
        let mut span = [0i32; MAX_DIMS];
        if self.thalamus.get(n).map_or(true, |x| x.kind != Kind::Inter) { return span; }
        let mut base = Vec::new();
        self.expand(n, [0; MAX_DIMS], &mut Vec::new(), &mut base, None);
        for (_, p) in base { for d in 0..MAX_DIMS { span[d] = span[d].max(p[d].abs()); } }
        span
    }

    pub(crate) fn mint(&mut self, axes: u8, key: String, value: Option<Value>) -> NeuronId {
        let id = self.thalamus.next_id();
        let mut n = Neuron::inter(id, axes, self.cfg.h, self.radii.clone());
        n.key = key.clone();
        n.value = value;
        self.thalamus.push(n);
        self.spans.push([0; MAX_DIMS]);
        self.alive += 1;
        self.index.insert(key, id);
        id
    }

    /// A value followed through value neurons to a neuron that is not one (D45).
    pub(crate) fn follow(&self, mut n: NeuronId) -> NeuronId {
        for _ in 0..64 {
            match self.thalamus[n].value { Some(v) => n = v.member, None => break }
        }
        n
    }

}
