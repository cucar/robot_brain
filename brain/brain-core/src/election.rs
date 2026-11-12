//! The election (§6.2, R22, R24, D28, D24) and allocation (§6.3, R16): every fit of this level, taken by the largest
//! coverage less price, each while its coverage exceeds its price; each accepted fit fires its function's
//! interneuron at the next level, in the pose it read, at the position of the latest activation it landed on.

use rustc_hash::FxHashSet;

use crate::function::Func;
use crate::machine::*;
use crate::neuron::*;
use crate::recognize::{edge, Fit};
use crate::types::*;

impl Machine {
    /// What a fit would take off the board (R22, D28): each board relation its true relations land on, and the
    /// cost of each call it lands on, as a node or through its pose (D22): its failures, a call itself costing
    /// nothing (experiment); unless a call this election has taken already landed on it and took its cost off.
    fn coverage(&self, fit: &Fit) -> f64 {
        let mut seen: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        let mut c = 0.0;
        for &(a, b) in &fit.edges { if seen.insert(edge(a, b)) { c += 1.0; } }
        // credited the relations it read and nothing of a call (D22)
        c
    }

    /// What a fit costs (R22, D28): a failure per relation that did not come true, the call itself costing nothing
    /// (experiment); and one edge to each call this election has taken that stands for an activation it lands on
    /// (D10).
    fn price(&self, fit: &Fit) -> f64 {
        let mut taken: FxHashSet<ActRef> = FxHashSet::default();
        for (l, _) in &fit.landed {
            if let Some(a) = self.act(*l) { for c in &a.coverers { if self.electing.contains(c) { taken.insert(*c); } } }
        }
        fit.failed.len() as f64 + taken.len() as f64
    }

    /// The tie-break of two fits that measure the same (R24): a fit whose relations land on the board relations as
    /// they read before one landing a node on a call through its pose; experiment: the fit whose activations stand
    /// for more of the same activations underneath; a function with an interneuron before one without; the older
    /// interneuron; the function that joined first; the fit from the older activation.
    fn rank(&self, fit: &Fit) -> (bool, i64, u8, u64, u64, ActRef) {
        let f = self.dict.get(fit.func);
        let via_call = !fit.calls.is_empty() || fit.through;
        let shared = -(self.shared_under(fit) as i64);
        match f.and_then(|f| f.inter) {
            Some(n) => (via_call, shared, 0, n as u64, 0, fit.entry.1),
            None => (via_call, shared, 1, 0, f.map_or(u64::MAX, |f| f.joined), fit.entry.1),
        }
    }

    /// How many activations at the base stand under two or more of the activations a fit lands on (experiment).
    fn shared_under(&self, fit: &Fit) -> usize {
        let mut seen: rustc_hash::FxHashMap<ActRef, u32> = rustc_hash::FxHashMap::default();
        for (k, _) in fit.acts() {
            let Some(a) = self.act(k) else { continue };
            let mut base: Vec<ActRef> = if a.covers.is_empty() { vec![k] } else {
                self.under(a).into_iter().filter(|r| self.act(*r).map_or(false, |x| x.covers.is_empty())).collect()
            };
            base.sort();
            base.dedup();
            for b in base { *seen.entry(b).or_insert(0) += 1; }
        }
        seen.values().filter(|n| **n >= 2).count()
    }

    /// The reuse search (R43, D36): an interneuron that already stands for what this function's would, because the
    /// function's chain comes round to it. A function is never the interneuron of one of its own nodes, so no
    /// interneuron whose call is a node of the function is tried; only calls further down the chain, inside the calls
    /// the function's nodes are. Each such earlier call is compared, over its own length, with the function's
    /// expansion aligned at the present end (the origin). An interneuron is refused where any call of it, a node of
    /// the function or one met down its chain, stands on any activation of that window: it is present in what is
    /// read, not coming round. Where the two are the same, its interneuron is reused. The nearest meeting first,
    /// the older interneuron on a tie.
    fn reuse(&self, f: &Func) -> Option<NeuronId> {
        
        let lay = f.layout();
        let add = |a: &Pos, b: &Offset| { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p };
        // the function's expansion down to base neurons, and the calls met inside its nodes' calls, each at its place
        let mut out: Vec<(Sym, Pos)> = Vec::new();
        let mut met: Vec<(Sym, Pos)> = Vec::new();
        // every call in the function's expansion, its nodes and those down their chains, each at its place
        let mut calls: Vec<(Sym, Pos)> = Vec::new();
        let mut own_nodes: Vec<NeuronId> = Vec::new();
        for (i, m) in f.nodes.iter().enumerate() {
            let q = lay.position[i];
            if *m < SYM_CALL { out.push((*m, q)); continue; }
            own_nodes.push(self.dict.neuron_of(*m));
            calls.push((*m, q));
            for (s, d, _) in self.expansion_of(*m).iter() {
                let p = add(&q, d);
                if *s < SYM_CALL { out.push((*s, p)); } else { met.push((*s, p)); calls.push((*s, p)); }
            }
        }
        met.retain(|(s, _)| !own_nodes.contains(&self.dict.neuron_of(*s)));
        met.sort_by_key(|(s, p)| (p.iter().map(|x| x.unsigned_abs()).sum::<u32>(), self.dict.neuron_of(*s)));
        met.dedup();
        out.sort();
        out.dedup();
        let mut tried: Vec<(Sym, Pos)> = Vec::new();
        for (s, at) in met {
            if tried.contains(&(s, at)) { continue; }
            tried.push((s, at));
            let n = self.dict.neuron_of(s);
            if self.dict.of_inter(n).is_empty() || !self.thalamus.get(n).map_or(false, |x| x.alive) { continue; }
            // the earlier call's own expansion, down to base neurons, from its position
            let mut own: Vec<(Sym, Pos)> = self.expansion_of(s).iter().filter(|(x, _, _)| *x < SYM_CALL).map(|(x, d, _)| (*x, *d)).collect();
            if own.is_empty() { continue; }
            // its length in each dimension, and the function's expansion over that length, aligned at the present end
            let (mut lo, mut hi) = ([0i32; MAX_DIMS], [0i32; MAX_DIMS]);
            for (_, q) in &own { for d in 0..MAX_DIMS { lo[d] = lo[d].min(q[d]); hi[d] = hi[d].max(q[d]); } }
            let inside = |q: &Pos| (0..MAX_DIMS).all(|d| q[d] >= lo[d] && q[d] <= hi[d]);
            let mut a: Vec<(Sym, Pos)> = out.iter().filter(|(_, q)| inside(q)).copied().collect();
            own.sort();
            own.dedup();
            a.sort();
            // whether a call of this interneuron stands on any activation of the window; narrowed (switch 10), only
            // a call standing wholly inside it
            let window: Vec<Pos> = a.iter().map(|x| x.1).collect();
            let narrow = true;
            let present = calls.iter().filter(|(c, _)| self.dict.neuron_of(*c) == n).any(|(c, cp)| {
                let pts: Vec<Pos> = self.expansion_of(*c).iter().filter(|(x, _, _)| *x < SYM_CALL).map(|(_, d, _)| add(cp, d)).collect();
                if narrow { !pts.is_empty() && pts.iter().all(|p| window.contains(p)) } else { pts.iter().any(|p| window.contains(p)) }
            });
            if present {
                if self.trace { eprintln!("[reuse? f{}] {} meets {}: a call of #{} stands in the window", self.frame_no, self.dict.describe(f), self.dict.sym_text(s), n); }
                continue;
            }
            if self.trace { eprintln!("[reuse? f{}] {} meets {} | its own {:?} | the function's over that length {:?}", self.frame_no, self.dict.describe(f), self.dict.sym_text(s), own, a); }
            if a == own { return Some(n); }
        }
        None
    }

    /// The greedy cover over the fits of this level (D28, R24): each round the fit whose coverage exceeds its
    /// price by the most; a fit that read through a relation a taken fit covers is read again over what stands.
    /// Returns how many were accepted and allocated.
    pub(crate) fn elect(&mut self, level: u32, mut fits: Vec<Fit>) -> (usize, usize) {
        self.electing.clear();
        let (mut accepted, mut minted) = (0, 0);
        loop {
            let mut best: Option<(usize, f64)> = None;
            for (i, fit) in fits.iter().enumerate() {
                let net = self.coverage(fit) - self.price(fit);
                if !(net > 0.0) { continue; }
                let better = match best { None => true, Some((j, bn)) => net > bn || (net == bn && self.rank(fit) < self.rank(&fits[j])) };
                if better { best = Some((i, net)); }
            }
            let Some((i, _)) = best else { break };
            let fit = fits.swap_remove(i);
            let newly: FxHashSet<(ActRef, ActRef)> = fit.edges.iter().map(|&(a, b)| edge(a, b)).collect();
            minted += self.accept(&fit, level);
            accepted += 1;
            // read again what read through a relation the taken fit covers, or entered where it can no longer stand
            let mut next = Vec::with_capacity(fits.len());
            for g in fits {
                let stale = g.edges.iter().any(|&(a, b)| newly.contains(&edge(a, b)));
                if !stale { next.push(g); continue; }
                let Some(f) = self.dict.get(g.func) else { continue };
                if !self.act(g.entry.1).map_or(false, |a| self.landable(a)) { continue; }
                let again = self.fit(f, g.entry.0, g.entry.1);
                if !again.edges.is_empty() { next.push(again); }
            }
            fits = next;
        }
        self.electing.clear();
        (accepted, minted)
    }

    /// Accept a fit (R24, R16, D53): the call fires once at the next level, at the position of the latest activation
    /// the fit landed on, in the pose it read; it covers what the fit landed on and the relations it landed on
    /// (D24); each row that saw what it covers holds the call there instead (D10). Its function's interneuron,
    /// reused or allocated where it has none. Returns 1 where it allocated.
    fn accept(&mut self, fit: &Fit, level: u32) -> usize {
        let fno = self.frame_no;
        let Some(f) = self.dict.get(fit.func).cloned() else { return 0 };
        let all: Vec<(ActRef, usize)> = fit.acts().collect();
        // the latest activation it landed on, and the earliest (D8, D15)
        let mut latest = all[0];
        let mut earliest = all[0];
        for &(r, node) in &all {
            let (Some(p), Some(l), Some(e)) = (self.act(r).map(|a| a.pos), self.act(latest.0).map(|a| a.pos), self.act(earliest.0).map(|a| a.pos)) else { continue };
            if later(&p, &l) { latest = (r, node); }
            if later(&e, &p) { earliest = (r, node); }
        }
        let Some(origin) = self.act(latest.0).cloned() else { return 0 };
        let mut minted = 0;
        let own = f.inter.filter(|n| self.thalamus.get(*n).map_or(false, |x| x.alive));
        let found = if own.is_none() { self.reuse(&f) } else { None };
        let neuron = match own.or(found) {
            Some(n) if own.is_some() => n,
            Some(n) => {
                // an interneuron never holds two functions whose nodes are the same neurons: the older gives way,
                // its relations leaving with it (R43, R54, R18)
                let mut mine: Vec<NeuronId> = f.nodes.iter().map(|m| self.dict.neuron_of(*m)).collect();
                mine.sort_unstable();
                mine.dedup();
                let older: Vec<crate::function::FuncId> = self.dict.of_inter(n).iter().copied().filter(|g| {
                    self.dict.get(*g).map_or(false, |g| {
                        let mut v: Vec<NeuronId> = g.nodes.iter().map(|m| self.dict.neuron_of(*m)).collect();
                        v.sort_unstable();
                        v.dedup();
                        v == mine
                    })
                }).collect();
                for g in older {
                    if self.trace { if let Some(gf) = self.dict.get(g) { eprintln!("[gives way f{}] {} for {}", fno, self.dict.describe(gf), self.dict.describe(&f)); } }
                    if let Some(gone) = self.dict.remove(g) {
                        if self.trace_retire { eprintln!("[retire f{}] {} | gave way on reuse", self.frame_no, self.dict.describe(&gone)); }
                        self.forget_expansions(g);
                        self.retire(gone);
                    }
                }
                self.dict.set_inter(f.id, n);
                self.stats_reused += 1;
                if self.trace { eprintln!("[reuse f{}] {} -> #{}", fno, self.dict.describe(&f), n); }
                n
            }
            None => {
                minted = 1;
                let id = self.thalamus.next_id();
                self.thalamus.push(Neuron::inter(id, origin.axes, self.cfg.h));
                self.alive += 1;
                self.dict.set_inter(f.id, id);
                id
            }
        };
        self.ledger.retain(|x| *x != neuron);
        self.last_func.insert(neuron, f.id);
        // its pose (D53): from its latest node back to its earliest, round the laps it read; a call of its own
        // function it landed on adds what that call stood for, so its half lap is counted with the rest (R23)
        let order = f.back_order(latest.1);
        let pose = if !order.is_empty() && order.contains(&latest.1) {
            let mut len = 0usize;
            for (r, node) in fit.acts() {
                if !order.contains(&node) { continue; }
                let own = self.act(r).and_then(|a| if a.kind == Kind::Inter && a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { self.dict.pose_of(a.sym) } else { None });
                len += match own {
                    Some(p) => match (self.dict.node_by_neuron(&f, p.from), self.dict.node_by_neuron(&f, p.to)) { (Some(a), Some(b)) => f.stretch(a, b, p.laps), _ => 1 },
                    None => 1,
                };
            }
            let len = len.max(1);
            let c = order.len();
            Pose { from: self.dict.neuron_of(f.nodes[latest.1]), to: self.dict.neuron_of(f.nodes[order[(len - 1) % c]]), laps: ((len - 1) / c) as u32 }
        } else {
            let mut laps = fit.laps;
            for (c, _) in &fit.calls {
                if let Some(a) = self.act(*c) {
                    if a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { laps += self.dict.pose_of(a.sym).map_or(0, |p| p.laps); }
                }
            }
            Pose { from: self.dict.neuron_of(f.nodes[latest.1]), to: self.dict.neuron_of(f.nodes[earliest.1]), laps }
        };
        let sym = self.dict.sym_of(neuron, pose);
        let covers: Vec<ActRef> = all.iter().map(|x| x.0).collect();
        let strong = covers.iter().filter_map(|r| self.act(*r)).any(|a| a.strong);
        let height = 1 + covers.iter().filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        // what it stands for in space is the union of what it covers, and it sees one beyond that (D4)
        let (mut lo, mut hi) = (origin.pos, origin.pos);
        for c in covers.iter().filter_map(|r| self.act(*r)) { for d in 0..MAX_DIMS { lo[d] = lo[d].min(c.lo[d]); hi[d] = hi[d].max(c.hi[d]); } }
        let mut reach = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS {
            if d > 0 && origin.axes & (1 << d) == 0 { continue; }
            reach[d] = (origin.pos[d] - lo[d]).max(hi[d] - origin.pos[d]) + 1;
        }
        let act = Act {
            neuron, sym, kind: Kind::Inter, axes: origin.axes, pos: origin.pos, strong, reach, lo, hi,
            covered_at: None, coverers: Vec::new(), seen_by: Vec::new(), covers: covers.clone(),
            fired_by: Some(FiredBy { func: f.id, failed: fit.failed.clone() }),
            rels: Vec::new(), inferences: Vec::new(), opened_at: fno, level: level + 1, height, gone: false,
        };
        { let r = self.reach_of.entry(neuron).or_insert([0; MAX_DIMS]); for d in 0..MAX_DIMS { r[d] = r[d].max(act.reach[d]); } }
        let Some(fr) = self.frame_mut(fno) else { return minted };
        let me = ActRef { frame: fno, index: fr.acts.len() as u32 };
        fr.acts.push(act);
        self.electing.insert(me);
        for &(a, b) in &fit.edges {
            if self.covered_rel.insert(edge(a, b)) { self.covered_edges.push(edge(a, b)); }
        }
        // a relation read through a call and the relation to that call are one board relation: covered under
        // either form, it is covered under both (switch 11, R50, D5)
        {
            let mut more: Vec<(ActRef, ActRef)> = Vec::new();
            for &(a, b) in &fit.edges {
                for (x, y) in [(a, b), (b, a)] {
                    let Some(xa) = self.act(x) else { continue };
                    // read through a call beside x: the relation to that call is covered too
                    for br in &xa.rels {
                        let Some(c) = self.act(br.other) else { continue };
                        if c.covers.is_empty() || br.other == y { continue; }
                        if self.under(c).contains(&y) { more.push(edge(x, br.other)); }
                    }
                    // the relation to a call landed on as a node: every relation it stands for is covered too
                    if let Some(ya) = self.act(y) { if !ya.covers.is_empty() { for u in self.under(ya) { more.push(edge(x, u)); } } }
                }
            }
            {
                let mut t: Vec<ActRef> = fit.thru.clone();
                t.extend(fit.calls.iter().map(|x| x.0));
                t.extend(fit.landed.iter().map(|x| x.0).filter(|r| self.act(*r).map_or(false, |a| !a.covers.is_empty())));
                t.sort();
                t.dedup();
                self.call_thru.insert(me, t);
                self.call_fit.insert(me, fit.clone());
            }
            { let mut all: Vec<(ActRef, ActRef)> = fit.edges.iter().map(|&(a, b)| edge(a, b)).collect(); all.extend(more.iter().copied()); all.sort(); all.dedup(); self.call_edges.insert(me, all); }
            for e in more { if self.covered_rel.insert(e) { self.covered_edges.push(e); } }
        }
        for r in &covers {
            if let Some(a) = self.act_mut(*r) {
                if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); }
                if !a.coverers.contains(&me) { a.coverers.push(me); }
            }
        }
        // switch 11: what the call covered, each at its offset from it, kept for as long as rows may name it (R28)
        {
            if let Some(ca) = self.act(me) {
                let rec: Vec<(Sym, Offset)> = self.under(ca).into_iter().filter_map(|u| self.act(u).map(|ua| (ua.sym, offset_between(ca, ua)))).collect();
                self.covered_by.insert(me, rec);
            }
        }
        // every row that saw what it covers, outside what it stands for, holds the call in its pose at its offset
        // there (D10)
        if !self.frozen {
            let under = { let a = self.act(me).unwrap(); self.under(a) };
            let mut writes: Vec<(NeuronId, ActRef, ActRef, Nb)> = Vec::new();
            for r in &covers {
                let Some(x) = self.act(*r) else { continue };
                for &(vn, v) in &x.seen_by {
                    if v == me || under.contains(&v) { continue; }
                    let Some(va) = self.act(v) else { continue };
                    let off = offset_between(va, self.act(me).unwrap());
                    writes.push((vn, v, *r, Nb::seen(me, sym, off, *r)));
                }
            }
            let mut viewers: Vec<(NeuronId, ActRef)> = Vec::new();
            for (vn, v, old, nb) in writes {
                // switch 16: the row keeps what it held and takes the call beside it
                let took = { let _ = old; self.thalamus[vn].add_beside(v, nb) };
                if took && !viewers.contains(&(vn, v)) { viewers.push((vn, v)); }
            }
            if let Some(a) = self.act_mut(me) { a.seen_by = viewers; }
            // every row of what the call stands for keeps the edges it saved: those to what it also stands for (R50)
            
        }
        // where the report built the strong call of a function an inference named weakly, the weak one goes (D40)
        if strong {
            let pos = origin.pos;
            if let Some(fr) = self.frame_mut(fno) {
                for (i, w) in fr.acts.iter_mut().enumerate() {
                    if i as u32 != me.index && !w.strong && !w.gone && w.neuron == neuron && w.pos == pos { w.gone = true; }
                }
            }
        }
        if self.trace { eprintln!("[accept f{} l{}] {} at {:?} as {} covers {} failed {:?}", fno, level, self.dict.describe(&f), origin.pos, self.dict.sym_text(sym), covers.len(), fit.failed); }
        // with functions that never change, a fit that stepped onto its own earlier call counts that step as one
        // more row for the relation it stepped by, in the row of the activation it stepped from (switch 8)
        
        minted
    }
}
