//! The election (R22, R24), the children it creates (R43) and where they fire (D53), and span (D4).

use rustc_hash::FxHashMap;

use crate::machine::*;
use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

/// A child to fire at the next level, with what its bid covered (§6.4).
pub(crate) struct Pending {
    pub(crate) act: Act,
    pub(crate) covered: Vec<ActRef>,
}

/// One bid resolved on the board from its bidder's coordinate (D31, R22).
#[derive(Clone, Debug)]
pub(crate) struct Resolved {
    pub(crate) coverage: f64,
    pub(crate) price: f64,
    pub(crate) covered: Vec<ActRef>,
    pub(crate) bound: Vec<(usize, ActRef)>,
    pub(crate) failed: Vec<usize>,
    pub(crate) strong: bool,
    pub(crate) span: Pos,
}

impl Machine {
    // ───────────────────────────── prices on the board (D13) ─────────────────────────────

    /// The window (D13): the longest reach the machine holds, in time and in every other dimension.
    pub(crate) fn window(&self) -> Pos {
        let mut w = [0; MAX_DIMS];
        w[0] = self.max_reach_t;
        for d in 1..self.radii.len() { w[d] = self.spans.iter().map(|s| s[d]).max().unwrap_or(0) + self.radii[d]; }
        w
    }

    /// A record (D12, D13): present or absent, a neuron, and its position in the window, for a neuron laid
    /// out over `axes`. Returns the record and the part of it that is the position.
    pub(crate) fn record_for(&self, axes: u8, _span: &Pos) -> (f64, f64) {
        let pos = window_bits(&self.window(), axes);
        (1.0 + log2(self.alive as f64) + pos, pos)
    }

    /// What is left of an activation to pay for (D24): nothing once covered, its neuron once bound.
    fn remaining(a: &Act) -> (f64, f64) {
        if a.gone || a.covered_at.is_some() { (0.0, 0.0) } else if a.bound_at.is_some() { (a.cost - a.pos_bits, 0.0) } else { (a.cost, a.pos_bits) }
    }

    /// The span a function's child has (D4), read with the spans as they stand.
    pub(crate) fn span_of_function(&self, owner: NeuronId, f: &Function, classes: &[(PatternId, Vec<NeuronId>)]) -> Pos {
        let sp = |n: NeuronId| self.spans.get(n as usize).copied().unwrap_or([0; MAX_DIMS]);
        let mut span = [0i32; MAX_DIMS];
        for (place, off) in &f.places {
            let inner = match place {
                Place::Owner => sp(owner),
                Place::Fixed(m) => sp(*m),
                Place::Class(k) => classes.iter().find(|(c, _)| c == k).map_or([0; MAX_DIMS], |(_, ms)| ms.iter().fold([0; MAX_DIMS], |s, m| max_pos(&s, &sp(*m)))),
                Place::Param(k) => f.params.get(*k as usize).map_or([0; MAX_DIMS], |p| p.members.iter().fold([0; MAX_DIMS], |s, m| max_pos(&s, &sp(m.neuron)))),
            };
            for d in 0..MAX_DIMS { span[d] = span[d].max(off[d].abs() + inner[d]); }
        }
        span
    }

    // ───────────────────────────── resolving a bid (D31, R22) ─────────────────────────────

    /// The activations on the board standing at a place of a bid (D6, D31), each with its reference.
    fn on_board(&self, bidder: &Act, off: &Offset, reach_t: i32) -> Vec<(ActRef, &Act)> {
        let mut out = Vec::new();
        for f in self.frames.iter() {
            let dt = bidder.pos[0] as i64 - f.no;
            if dt < 0 || dt > reach_t as i64 { continue; }
            for (i, a) in f.acts.iter().enumerate() {
                if a.gone { continue; }
                if round_delta(&sub(&a.pos, &bidder.pos), &self.radii, bidder.axes) != *off { continue; }
                out.push((ActRef { frame: f.no, index: i as u32 }, a));
            }
        }
        out
    }

    /// Of several activations that could answer one place, the one not yet covered first, then the one
    /// nearest the owner, then the earlier coordinate (D31).
    fn choose<'a>(&self, bidder: &Act, cands: &[(ActRef, &'a Act)]) -> Option<(ActRef, &'a Act)> {
        cands.iter().copied().min_by(|(_, a), (_, b)| {
            let near = |x: &Act| sub(&x.pos, &bidder.pos).iter().map(|v| v.abs()).sum::<i32>();
            a.covered_at.is_some().cmp(&b.covered_at.is_some()).then(near(a).cmp(&near(b))).then(a.pos.cmp(&b.pos))
        })
    }

    /// A child on the apex that stands for `n` at a place of the bid (D22): it holds the node there and is
    /// neither credited nor covered.
    fn stood_for(&self, bidder: &Act, n: NeuronId, off: &Offset, reach_t: i32) -> Option<ActRef> {
        for f in self.frames.iter() {
            let dt = bidder.pos[0] as i64 - f.no;
            if dt < 0 || dt > reach_t as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                if b.gone || b.covered_at.is_some() || b.kind != Kind::Inter { continue; }
                let mut all = Vec::new();
                let mut base = Vec::new();
                self.expand_into(b.neuron, b.pos, &mut Vec::new(), &mut base, Some(&mut all), b.fired_by.as_ref());
                if all.iter().any(|(m, p)| *m == n && round_delta(&sub(p, &bidder.pos), &self.radii, bidder.axes) == *off) { return Some(ActRef { frame: f.no, index: i as u32 }); }
            }
        }
        None
    }

    /// Resolve a bid on the board (R22): where each node stands from the bidder's coordinate; the coverage,
    /// what it covers that stands on the apex at what is left of it, its bidder among them, and the positions
    /// of what it binds; the price, its record and an absence per node named that did not stand.
    pub(crate) fn resolve(&self, bid: &Bid) -> Resolved {
        let Some(bidder) = self.act(bid.bidder) else { return Resolved { coverage: 0.0, price: f64::INFINITY, covered: vec![], bound: vec![], failed: vec![], strong: false, span: [0; MAX_DIMS] } };
        let axes = bidder.axes;
        let f = &bid.function;
        let reach_t = self.spans.get(bid.owner as usize).map_or(0, |s| s[0]) + self.radii[0];
        let sp = |n: NeuronId| self.spans.get(n as usize).copied().unwrap_or([0; MAX_DIMS]);
        let span = self.span_of_function(bid.owner, f, &bid.classes);
        let absence = |n: NeuronId| self.record_for(axes, &sp(n)).0;
        let mut price = self.record_for(axes, &span).0;
        let mut coverage = 0.0;
        let mut strong = bidder.strong;
        let mut covered: Vec<ActRef> = Vec::new();
        let mut bound: Vec<(usize, ActRef)> = Vec::new();
        let mut failed: Vec<usize> = Vec::new();
        let (own, _) = Self::remaining(bidder);
        if own > 0.0 { coverage += own; covered.push(bid.bidder); }
        let taken = |r: &ActRef, covered: &Vec<ActRef>, bound: &Vec<(usize, ActRef)>| covered.contains(r) || bound.iter().any(|(_, b)| b == r);
        for (pi, (place, off)) in f.places.iter().enumerate().skip(1) {
            match place {
                Place::Owner | Place::Param(_) => {}
                Place::Fixed(n) => {
                    let cands: Vec<(ActRef, &Act)> = self.on_board(bidder, off, reach_t).into_iter().filter(|(r, a)| a.neuron == *n && !taken(r, &covered, &bound)).collect();
                    if let Some((r, a)) = self.choose(bidder, &cands) {
                        if a.strong { strong = true; }
                        let (left, _) = Self::remaining(a);
                        if left > 0.0 { coverage += left; covered.push(r); }
                    } else if self.stood_for(bidder, *n, off, reach_t).is_some() {
                        // held by a child that stands for it: credited nothing, not covered (D22)
                    } else {
                        price += absence(*n);
                        failed.push(pi);
                    }
                }
                Place::Class(k) => {
                    let members: Vec<NeuronId> = bid.classes.iter().find(|(c, _)| c == k).map(|(_, m)| m.clone()).unwrap_or_default();
                    let cands: Vec<(ActRef, &Act)> = self.on_board(bidder, off, reach_t).into_iter().filter(|(r, a)| members.contains(&a.neuron) && !taken(r, &covered, &bound)).collect();
                    match self.choose(bidder, &cands) {
                        Some((r, a)) => {
                            if a.strong { strong = true; }
                            let (_, pos) = Self::remaining(a);
                            if pos > 0.0 { coverage += pos; }
                            bound.push((pi, r));
                        }
                        None => { price += absence(members.first().copied().unwrap_or(bid.owner)); failed.push(pi); }
                    }
                }
            }
        }
        // parameters: the value is the neuron at the earliest of their places (D44)
        for k in 0..f.params.len() as u32 {
            let mut places = f.param_places(k);
            places.sort_by(|a, b| f.places[*a].1.cmp(&f.places[*b].1));
            let value = places.iter().find_map(|&pi| {
                let cands: Vec<(ActRef, &Act)> = self.on_board(bidder, &f.places[pi].1, reach_t).into_iter().filter(|(r, _)| !taken(r, &covered, &bound)).collect();
                self.choose(bidder, &cands).map(|(_, a)| a.neuron)
            });
            for &pi in &places {
                let cands: Vec<(ActRef, &Act)> = value.map_or(Vec::new(), |v| self.on_board(bidder, &f.places[pi].1, reach_t).into_iter().filter(|(r, a)| a.neuron == v && !taken(r, &covered, &bound)).collect());
                match self.choose(bidder, &cands) {
                    Some((r, a)) => {
                        if a.strong { strong = true; }
                        let (_, pos) = Self::remaining(a);
                        if pos > 0.0 { coverage += pos; }
                        bound.push((pi, r));
                    }
                    None => { price += absence(value.unwrap_or(bid.owner)); failed.push(pi); }
                }
            }
        }
        Resolved { coverage, price, covered, bound, failed, strong, span }
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
        a.function.id < b.function.id
    }

    // ───────────────────────────── the election (R24) ─────────────────────────────

    /// D28 over the board (R24), then a child for each accepted bid (R43).
    pub(crate) fn elect(&mut self, level: u32, bids: Vec<Bid>, pending: &mut Vec<Pending>) -> (usize, usize, usize) {
        let fno = self.frame_no;
        let mut remaining: Vec<Bid> = bids;
        let mut accepted = 0;
        let mut minted = 0;
        let mut reused = 0;
        loop {
            let mut best: Option<(usize, f64)> = None;
            for (i, b) in remaining.iter().enumerate() {
                let r = self.resolve(b);
                if !(r.coverage > r.price) { continue; }
                let ratio = r.coverage / r.price;
                let better = match best {
                    None => true,
                    Some((j, br)) => ratio > br || (ratio == br && self.bid_before(b, &remaining[j])),
                };
                if better { best = Some((i, ratio)); }
            }
            let Some((i, _)) = best else { break };
            let bid = remaining.remove(i);
            let res = self.resolve(&bid);
            if self.trace {
                let show = |rs: &[ActRef]| rs.iter().filter_map(|r| self.act(*r).map(|a| format!("#{}@{}={:.1}", a.neuron, a.pos[0], a.cost))).collect::<Vec<_>>().join(" ");
                let bound: Vec<ActRef> = res.bound.iter().map(|(_, r)| *r).collect();
                eprintln!("[accept f{} l{}] #{} p{} at {:?} cov {:.1} price {:.1} failed {:?} covers [{}] binds [{}]", fno, level, bid.owner, bid.function.id, bid.bidder_pos, res.coverage, res.price, res.failed, show(&res.covered), show(&bound));
            }
            accepted += 1;
            // credit, once (R23): what it covers leaves the apex, what it binds stays with its position covered (D10, D24)
            for r in &res.covered { if let Some(a) = self.act_mut(*r) { if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); } } }
            for (_, r) in &res.bound { if let Some(a) = self.act_mut(*r) { if a.bound_at.is_none() && a.covered_at.is_none() { a.bound_at = Some((fno - r.frame) as u32); } } }
            let (kid, m, u) = self.child_for(&bid, &res, level + 1);
            minted += m;
            reused += u;
            if let Some(k) = kid { pending.push(k); }
        }
        (accepted, minted, reused)
    }

    /// The call neuron an accepted bid fires (R16, R43, D53): the function's own child, the interneuron that
    /// stands for what its child would, or a new one; at the bidder's coordinate, as strong as what it covers
    /// or binds (D40), its record its cost (D13). The bid sets the child's definition where it differs from it
    /// or the child is new, the first such bid of the frame (D36); the function is pointed at the child by
    /// update child once the frame's last level has run (R16).
    pub(crate) fn child_for(&mut self, bid: &Bid, res: &Resolved, level: u32) -> (Option<Pending>, usize, usize) {
        let mut minted = 0;
        let mut reused = 0;
        let owner_axes = self.thalamus[bid.owner].axes;
        let key = self.function_key(bid.owner, &bid.function, &bid.classes);
        let child = match bid.child.filter(|c| self.thalamus.get(*c).map_or(false, |n| n.alive)) {
            Some(c) => c,
            None => match self.index.get(&key).copied().filter(|id| self.thalamus.get(*id).map_or(false, |n| n.alive)) {
                Some(id) => { reused += 1; id }
                None => match self.reusable(bid.owner, &bid.function) {
                    Some(id) => { reused += 1; if self.trace { eprintln!("[closes f{}] #{} {} -> #{} {}", self.frame_no, bid.owner, describe_function(&bid.function), id, self.definitions.get(&id).map_or(String::new(), |(o, d)| format!("#{} {}", o, describe_function(d)))); } id }
                    None => { minted += 1; self.mint(owner_axes, key.clone()) }
                },
            },
        };
        // the definition (D36)
        let differs = self.definitions.get(&child).map_or(true, |(o, d)| *o != bid.owner || d.places != bid.function.places);
        if differs && !self.redefined.contains_key(&child) {
            let mut def = bid.function.clone();
            def.child = Some(child);
            self.redefined.insert(child, (bid.owner, def.clone(), bid.classes.clone()));
            self.definitions.insert(child, (bid.owner, def));
            self.def_classes.insert(child, bid.classes.clone());
        }
        if !self.won.contains(&(bid.owner, bid.function.id, child)) { self.won.push((bid.owner, bid.function.id, child)); }
        let covered: Vec<ActRef> = res.covered.clone();
        let height = 1 + covered.iter().chain(res.bound.iter().map(|(_, r)| r)).filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        let span = res.span;
        if (child as usize) < self.spans.len() { self.spans[child as usize] = max_pos(&self.spans[child as usize], &span); }
        let args: Vec<(usize, NeuronId, ActRef)> = res.bound.iter().filter_map(|(p, r)| self.act(*r).map(|a| (*p, a.neuron, *r))).collect();
        let fb = FiredBy { owner: bid.owner, pattern: bid.function.id, args, failed: res.failed.clone() };
        let act = self.new_act(child, bid.bidder_pos, res.strong, Some(fb), span, level, height);
        if self.trace { eprintln!("[fire f{}] #{} at {:?} from #{} p{} cost {:.1} span {:?}", self.frame_no, child, bid.bidder_pos, bid.owner, bid.function.id, act.cost, span); }
        (Some(Pending { act, covered }), minted, reused)
    }

    /// The span of an interneuron (D4): over its definition, every node's offset from the owner plus that
    /// node's own span, a variable's being the largest among its members.
    pub(crate) fn neuron_span(&self, n: NeuronId, memo: &mut FxHashMap<NeuronId, Pos>, path: &mut Vec<NeuronId>) -> Pos {
        let mut span = [0i32; MAX_DIMS];
        let Some(nn) = self.thalamus.get(n) else { return span };
        if nn.kind != Kind::Inter { return span; }
        if let Some(s) = memo.get(&n) { return *s; }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return span; }
        let Some((owner, f)) = self.definitions.get(&n).cloned() else { return span };
        let classes = self.def_classes.get(&n).cloned().unwrap_or_default();
        path.push(n);
        for (place, off) in &f.places {
            let inner = match place {
                Place::Owner => self.neuron_span(owner, memo, path),
                Place::Fixed(m) => self.neuron_span(*m, memo, path),
                // a member built over this neuron is the variable's value at another occasion, not part of what
                // this neuron stands for (D36): it adds nothing to the span
                Place::Class(k) => {
                    let members: Vec<NeuronId> = classes.iter().find(|(c, _)| c == k).map(|(_, m)| m.clone()).unwrap_or_default();
                    let members: Vec<NeuronId> = members.into_iter().filter(|m| !path.contains(m) && *m != n).collect();
                    members.iter().fold([0; MAX_DIMS], |s, m| max_pos(&s, &self.neuron_span(*m, memo, path)))
                }
                Place::Param(k) => {
                    let members: Vec<NeuronId> = f.params.get(*k as usize).map(|p| p.members.iter().map(|m| m.neuron).collect()).unwrap_or_default();
                    let members: Vec<NeuronId> = members.into_iter().filter(|m| !path.contains(m) && *m != n).collect();
                    members.iter().fold([0; MAX_DIMS], |s, m| max_pos(&s, &self.neuron_span(*m, memo, path)))
                }
            };
            for d in 0..MAX_DIMS { span[d] = span[d].max(off[d].abs() + inner[d]); }
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
