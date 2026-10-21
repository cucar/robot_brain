//! The election (R22, R24), the children it creates (R43) and where they fire (D53), and span (D4).

use rustc_hash::FxHashMap;

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// A child to fire at the next level, with what its bid covered (§6.4).
pub(crate) struct Pending {
    pub(crate) act: Act,
    pub(crate) covered: Vec<ActRef>,
}

impl Machine {
    // ───────────────────────────── election (R22, R24) ─────────────────────────────

    /// What a bid is credited on the board and what it costs (R22): the activations it covers, its bidder
    /// and its fixed nodes, and those it binds in its variables, each at what it cost, where no earlier bid
    /// was credited it (D24); and the fields of its call.
    pub(crate) fn measure_bid(&self, bid: &Bid) -> (f64, f64, Vec<ActRef>, Vec<ActRef>, bool) {
        let mut covered: Vec<ActRef> = Vec::new();
        let mut bound: Vec<ActRef> = Vec::new();
        let mut coverage = 0.0;
        let mut strong = false;
        let free = |a: &Act| !a.gone && a.covered_at.is_none() && a.bound_at.is_none();
        let mut take = |r: ActRef, credit: Option<f64>, bind: bool, covered: &mut Vec<ActRef>, bound: &mut Vec<ActRef>| {
            if covered.contains(&r) || bound.contains(&r) { return; }
            let Some(a) = self.act(r) else { return };
            if a.strong { strong = true; }
            if !free(a) { return; }
            coverage += credit.map_or(a.cost, |c| c.min(a.cost));
            if bind { bound.push(r); } else { covered.push(r); }
        };
        take(bid.bidder, None, false, &mut covered, &mut bound);
        for (k, r) in bid.covers.iter().enumerate() { take(*r, bid.cover_credit.get(k).copied(), false, &mut covered, &mut bound); }
        for (_, r, _) in &bid.binds { take(*r, None, true, &mut covered, &mut bound); }
        (coverage, bid.price(), covered, bound, strong)
    }

    /// The tie-break (R24): a bid carrying a child before one carrying none; between two with children,
    /// the older child; between two with none, the older bidder; then the earlier coordinate.
    pub(crate) fn bid_before(&self, a: &Bid, b: &Bid) -> bool {
        match (a.child, b.child) {
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(x), Some(y)) if x != y => return x < y,
            _ => {}
        }
        if a.owner != b.owner { return a.owner < b.owner; }
        if a.bidder_pos != b.bidder_pos { return a.bidder_pos < b.bidder_pos; }
        a.function < b.function
    }

    /// D28 over the board (R24), then a child for each accepted bid (R43).
    pub(crate) fn elect(&mut self, level: u32, bids: Vec<Bid>, pending: &mut Vec<Pending>) -> (usize, usize, usize) {
        let fno = self.frame_no;
        let mut remaining: Vec<Bid> = bids;
        let mut accepted = 0;
        let mut minted = 0;
        let mut reused = 0;
        loop {
            let mut best: Option<(usize, f64, f64)> = None;
            for (i, b) in remaining.iter().enumerate() {
                let (cov, price, _, _, _) = self.measure_bid(b);
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
            let (cov, price, covered, bound, strong) = self.measure_bid(&bid);
            if self.trace {
                let show = |rs: &[ActRef]| rs.iter().filter_map(|r| self.act(*r).map(|a| format!("#{}@{}={:.1}", a.neuron, a.pos[0], a.cost))).collect::<Vec<_>>().join(" ");
                eprintln!("[accept f{} l{}] #{} p{} at {:?} cov {:.1} price {:.1} (bidder {:.1} which {:.1} choice {:.1} failed {}) covers [{}] binds [{}]", fno, level, bid.owner, bid.function, bid.bidder_pos, cov, price, bid.bidder_cost, bid.which_function_bits, bid.choice, bid.failed, show(&covered), show(&bound));
            }
            accepted += 1;
            // credit, once (R23): what it covers leaves the apex, what it binds stays (D10, D24)
            for r in &covered { if let Some(a) = self.act_mut(*r) { if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); } } }
            for r in &bound { if let Some(a) = self.act_mut(*r) { if a.bound_at.is_none() { a.bound_at = Some((fno - r.frame) as u32); } } }
            let (kid, m, u) = self.child_for(&bid, &covered, strong, level + 1);
            minted += m;
            reused += u;
            if let Some(k) = kid { pending.push(k); }
        }
        (accepted, minted, reused)
    }

    /// The call neuron an accepted bid fires (R16, R43, D53): the function's own child, the interneuron
    /// that stands for what its child would, or a new one; at the bidder's coordinate, as strong as what it
    /// covers or binds (D40), costing what its bidder cost, which function and its failed nodes (D13).
    pub(crate) fn child_for(&mut self, bid: &Bid, covered: &[ActRef], strong: bool, level: u32) -> (Option<Pending>, usize, usize) {
        let mut minted = 0;
        let mut reused = 0;
        let owner_axes = self.thalamus[bid.owner].axes;
        let child = match bid.child {
            Some(c) if self.thalamus[c].alive => c,
            _ => {
                let key = self.child_key(bid.owner, bid.function);
                match self.index.get(&key).copied().filter(|id| self.thalamus[*id].alive) {
                    Some(id) => { reused += 1; id }
                    None => { minted += 1; self.mint(owner_axes, key) }
                }
            }
        };
        self.thalamus[bid.owner].wire_child(bid.function, child);
        if !self.thalamus[child].parents.contains(&(bid.owner, bid.function)) { self.thalamus[child].parents.push((bid.owner, bid.function)); }
        let height = 1 + covered.iter().filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        let span = self.neuron_span(child, &mut FxHashMap::default(), &mut Vec::new());
        self.spans[child as usize] = span;
        let cost = bid.bidder_cost + bid.which_function_bits + bid.failed as f64 * log2(bid.p_size.max(1) as f64);
        let args: Vec<(usize, NeuronId, ActRef)> = bid.binds.iter().map(|(p, r, n)| (*p, *n, *r)).collect();
        let fb = FiredBy { owner: bid.owner, pattern: bid.function, args, failed: bid.failed_places.clone() };
        let act = self.new_act(child, bid.bidder_pos, cost, strong, Some(fb), span, level, height);
        if self.trace { eprintln!("[fire f{}] #{} at {:?} from #{} p{} cost {:.1} span {:?}", self.frame_no, child, bid.bidder_pos, bid.owner, bid.function, cost, span); }
        (Some(Pending { act, covered: covered.to_vec() }), minted, reused)
    }

    /// The span of a neuron (D4): zero for a base neuron; for an interneuron the largest, over the
    /// functions wired to it and every node each names, of the node's offset from the owner plus the node's
    /// own neuron's span, a variable's being the largest among its members.
    pub(crate) fn neuron_span(&self, n: NeuronId, memo: &mut FxHashMap<NeuronId, Pos>, path: &mut Vec<NeuronId>) -> Pos {
        let mut span = [0i32; MAX_DIMS];
        let Some(nn) = self.thalamus.get(n) else { return span };
        if nn.kind != Kind::Inter { return span; }
        if let Some(s) = memo.get(&n) { return *s; }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return span; }
        path.push(n);
        let parents = nn.parents.clone();
        for (owner, fid) in parents {
            let Some(f) = self.line(owner, fid).cloned() else { continue };
            for (place, off) in &f.places {
                let inner = match place {
                    Place::Owner => self.neuron_span(owner, memo, path),
                    Place::Fixed(m) => self.neuron_span(*m, memo, path),
                    Place::Class(k) => {
                        let members: Vec<NeuronId> = self.thalamus[owner].class(*k).map(|c| c.members.iter().map(|m| m.neuron).collect()).unwrap_or_default();
                        let mut s = [0i32; MAX_DIMS];
                        for m in members { let x = self.neuron_span(m, memo, path); for d in 0..MAX_DIMS { s[d] = s[d].max(x[d]); } }
                        s
                    }
                    Place::Param(k) => {
                        let members: Vec<NeuronId> = f.params.get(*k as usize).map(|p| p.members.iter().map(|m| m.neuron).collect()).unwrap_or_default();
                        let mut s = [0i32; MAX_DIMS];
                        for m in members { let x = self.neuron_span(m, memo, path); for d in 0..MAX_DIMS { s[d] = s[d].max(x[d]); } }
                        s
                    }
                };
                for d in 0..MAX_DIMS { span[d] = span[d].max(off[d].abs() + inner[d]); }
            }
        }
        path.pop();
        memo.insert(n, span);
        span
    }

    pub(crate) fn mint(&mut self, axes: u8, key: String) -> NeuronId {
        let id = self.thalamus.next_id();
        let mut n = Neuron::inter(id, axes, self.cfg.h, self.radii.clone());
        n.key = key.clone();
        self.thalamus.push(n);
        self.spans.push([0; MAX_DIMS]);
        self.alive += 1;
        self.index.insert(key, id);
        id
    }
}
