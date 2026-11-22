/// The election (§6.2, R22, R24, D28, D24) and allocation (§6.3, R16): every fit of this level, taken by the largest
/// coverage less price, each while its coverage exceeds its price; each accepted fit fires its function's
/// interneuron at the next level, in the phase it read, at the position of the latest activation it landed on.

use rustc_hash::FxHashSet;

use crate::dictionary::Func;
use crate::brain::*;
use crate::recognize::{edge, Fit};
use crate::thalamus::NeuronFacts;
use crate::types::*;

impl Brain {
    /// What a fit would take off the board (R22, D28): each board relation its true relations land on, once. A
    /// relation refused or blocked is credited to no fit (D24).
    fn measure_coverage(&self, fit: &Fit) -> f64 {
        let mut seen: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        let mut c = 0.0;
        for &(a, b) in &fit.edges { if !self.is_blocked(a, b) && !self.refused.contains(&edge(a, b)) && seen.insert(edge(a, b)) { c += 1.0; } }
        // credited the relations it read and nothing of a call (D22)
        c
    }

    /// What a fit costs (R22, D28): a failure per relation that did not come true; the call itself costs nothing.
    fn price(&self, fit: &Fit) -> f64 {
        // its failures and its bindings, and nothing for a relation that comes into being only when its own
        // interneuron fires (D22, R24); there are no variables, so no bindings
        fit.failed.len() as f64
    }

    /// The tie-break of two fits that measure the same (R24): the fit whose activations stand for more of the same
    /// activations underneath; a function with an interneuron before one without; the older interneuron; the function
    /// that joined first; the fit from the older activation.
    fn rank(&self, fit: &Fit) -> (i64, u8, u64, u64, ActRef) {
        let f = self.thalamus.dict.get(fit.func);
        let shared = -(self.count_shared_under(fit) as i64);
        match f.and_then(|f| f.inter) {
            Some(n) => (shared, 0, n as u64, 0, fit.entry.1),
            None => (shared, 1, 0, f.map_or(u64::MAX, |f| f.joined), fit.entry.1),
        }
    }

    /// Whether a fit is inside the frame: its function holds relations inside the frame only (D15, D34).
    pub(crate) fn is_fit_inside(&self, fit: &Fit) -> bool {
        self.thalamus.dict.get(fit.func).map_or(false, |f| f.rels.iter().all(|r| r.off[0] == 0))
    }

    /// How many activations at the base stand under two or more of the activations a fit lands on.
    fn count_shared_under(&self, fit: &Fit) -> usize {
        let mut seen: rustc_hash::FxHashMap<ActRef, u32> = rustc_hash::FxHashMap::default();
        for (k, _) in fit.acts() {
            let Some(a) = self.find_act(k) else { continue };
            let mut base: Vec<ActRef> = if a.covers.is_empty() { vec![k] } else {
                self.list_under(a).into_iter().filter(|r| self.find_act(*r).map_or(false, |x| x.covers.is_empty())).collect()
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
    /// expansion aligned at the present end: its origin, or for a cycle, which has none, the node its call's phase
    /// starts at (D15, D37). An interneuron is refused where any call of it, a node of
    /// the function or one met down its chain, stands on any activation of that window: it is present in what is
    /// read, not coming round. Where the two are the same, its interneuron is reused. The nearest meeting first,
    /// the older interneuron on a tie.
    fn reuse(&self, f: &Func, root: usize) -> Option<NeuronId> {
        let lay = f.lay_out_from(root);
        let add = |a: &Pos, b: &Offset| a.plus(b);
        // the function's expansion down to base neurons, and the calls met inside its nodes' calls, each at its place
        let mut out: Vec<(Sym, Pos)> = Vec::new();
        let mut met: Vec<(Sym, Pos)> = Vec::new();
        // every call in the function's expansion, its nodes and those down their chains, each at its place
        let mut calls: Vec<(Sym, Pos)> = Vec::new();
        let mut own_nodes: Vec<NeuronId> = Vec::new();
        for (i, m) in f.nodes.iter().enumerate() {
            let q = lay.position[i];
            if *m < SYM_CALL { out.push((*m, q)); continue; }
            own_nodes.push(self.thalamus.dict.find_neuron_of(*m));
            calls.push((*m, q));
            for (s, d, _) in self.read_expansion(*m).iter() {
                let p = add(&q, d);
                if *s < SYM_CALL { out.push((*s, p)); } else { met.push((*s, p)); calls.push((*s, p)); }
            }
        }
        met.retain(|(s, _)| !own_nodes.contains(&self.thalamus.dict.find_neuron_of(*s)));
        met.sort_by_key(|(s, p)| ((0..p.len()).map(|d| p[d].unsigned_abs()).sum::<u32>(), self.thalamus.dict.find_neuron_of(*s)));
        met.dedup();
        out.sort();
        out.dedup();
        let mut tried: Vec<(Sym, Pos)> = Vec::new();
        for (s, at) in met {
            if tried.contains(&(s, at)) { continue; }
            tried.push((s, at));
            let n = self.thalamus.dict.find_neuron_of(s);
            if self.thalamus.dict.list_of_inter(n).is_empty() || !self.thalamus.registry.get(n).map_or(false, |x| x.alive) { continue; }
            // the earlier call's own expansion, down to base neurons, from its position
            let mut own: Vec<(Sym, Pos)> = self.read_expansion(s).iter().filter(|(x, _, _)| *x < SYM_CALL).map(|(x, d, _)| (*x, *d)).collect();
            if own.is_empty() { continue; }
            // its length in each dimension, and the function's expansion over that length, aligned at the present end
            let (mut lo, mut hi) = (Pos::zero(), Pos::zero());
            for (_, q) in &own { lo = lo.least(q); hi = hi.most(q); }
            let inside = |q: &Pos| (0..q.len().max(lo.len()).max(hi.len())).all(|d| q[d] >= lo[d] && q[d] <= hi[d]);
            let mut a: Vec<(Sym, Pos)> = out.iter().filter(|(_, q)| inside(q)).copied().collect();
            own.sort();
            own.dedup();
            a.sort();
            // whether a call of this interneuron stands wholly inside the window
            let window: Vec<Pos> = a.iter().map(|x| x.1).collect();
            let present = calls.iter().filter(|(c, _)| self.thalamus.dict.find_neuron_of(*c) == n).any(|(c, cp)| {
                let pts: Vec<Pos> = self.read_expansion(*c).iter().filter(|(x, _, _)| *x < SYM_CALL).map(|(_, d, _)| add(cp, d)).collect();
                !pts.is_empty() && pts.iter().all(|p| window.contains(p))
            });
            if present {
                continue;
            }
            if a == own { return Some(n); }
        }
        None
    }

    /// The greedy cover over the fits of this level (D28, R24): each round the fit whose coverage exceeds its
    /// price by the most; a fit that read through a relation a taken fit covers is read again over what stands.
    /// Returns how many were accepted and allocated.
    pub(crate) fn elect(&mut self, level: u32, mut fits: Vec<Fit>) -> (usize, usize) {
        self.electing.clear();
        // what joins two activations is what the calls taken at this level of this frame cover, across every election
        // of the level (D24, R24); a refused relation stays refused for as long as its two activations exist
        if self.joined_at != Some((self.frame_no, level)) {
            // a level visited twice in one frame, in the climb and after it, keeps what it joined across both visits
            match self.joined_at {
                Some((f, l)) if f == self.frame_no => {
                    let state = JoinState { joined: std::mem::take(&mut self.joined), parent: std::mem::take(&mut self.jparent), members: std::mem::take(&mut self.jmembers), dirty: std::mem::take(&mut self.jdirty), log_seen: self.jlog_seen };
                    self.joined_stash.insert(l, state);
                }
                _ => { self.joined_stash.clear(); self.rel_log.clear(); }
            }
            let state = self.joined_stash.remove(&level).unwrap_or_default();
            self.joined = state.joined;
            self.jparent = state.parent;
            self.jmembers = state.members;
            self.jdirty = state.dirty;
            self.jlog_seen = state.log_seen;
            self.joined_at = Some((self.frame_no, level));
        }
        let (mut accepted, mut minted) = (0, 0);
        loop {
            // a relation whose two ends the calls taken so far join would close a deadlock: it is refused (D24, D55)
            self.mark_refused();
            let mut best: Option<(usize, f64)> = None;
            // each fit's tie order read once per round, not at every comparison (speed only)
            let mut ranks: Vec<Option<(i64, u8, u64, u64, ActRef)>> = vec![None; fits.len()];
            for (i, fit) in fits.iter().enumerate() {
                let net = self.measure_coverage(fit) - self.price(fit);
                if !(net > 0.0) { continue; }
                // the frame before time (R24, D34): where one election holds both kinds, a fit inside the frame before
                // any fit in time; within a kind by net, then by rank
                let (ins_i, ins_j) = match best { Some((j, _)) => (self.is_fit_inside(fit), self.is_fit_inside(&fits[j])), None => (false, false) };
                let better = match best { None => true, Some(_) if ins_i != ins_j => ins_i, Some((j, bn)) => net > bn || (net == bn && {
                    if ranks[i].is_none() { ranks[i] = Some(self.rank(fit)); }
                    if ranks[j].is_none() { ranks[j] = Some(self.rank(&fits[j])); }
                    ranks[i] < ranks[j]
                }) };
                if better { best = Some((i, net)); }
            }
            let Some((i, _)) = best else { break };
            let fit = fits.swap_remove(i);
            let newly: FxHashSet<(ActRef, ActRef)> = fit.edges.iter().map(|&(a, b)| edge(a, b)).collect();
            minted += self.accept(&fit, level);
            accepted += 1;
            // covering takes effect as the fit is taken (D28, D24, R23): what it landed on and the taken fits now
            // cover is off the board from the next round
            let covered: FxHashSet<ActRef> = {
                let mut cands: Vec<ActRef> = fit.acts().map(|x| x.0).collect();
                for &(a, b) in &fit.edges { cands.push(a); cands.push(b); }
                self.cover_in_election(&cands).into_iter().collect()
            };
            // read again what read through a relation the taken fit covers, or through an activation now covered, or
            // entered where it can no longer stand
            let mut next = Vec::with_capacity(fits.len());
            for g in fits {
                let stale = g.edges.iter().any(|&(a, b)| newly.contains(&edge(a, b)))
                    || (!covered.is_empty() && (g.acts().any(|(r, _)| covered.contains(&r)) || g.edges.iter().any(|(a, b)| covered.contains(a) || covered.contains(b)) || g.thru.iter().any(|r| covered.contains(r))));
                if !stale { next.push(g); continue; }
                let Some(f) = self.thalamus.dict.get(g.func) else { continue };
                if !self.find_act(g.entry.1).map_or(false, |a| self.is_landable(a)) { continue; }
                let again = self.fit(f, g.entry.0, g.entry.1);
                if !again.edges.is_empty() { next.push(again); }
            }
            fits = next;
        }
        self.electing.clear();
        self.flush_lifts();
        (accepted, minted)
    }

    /// Accept a fit (R24, R16, D53): the call fires once at the next level, at the position of the latest activation
    /// the fit landed on, in the phase it read; it covers what the fit landed on and the relations it landed on
    /// (D24); each row that saw what it covers holds the call there instead (D10). Its function's interneuron,
    /// reused or allocated where it has none. Returns 1 where it allocated.
    fn accept(&mut self, fit: &Fit, level: u32) -> usize {
        let fno = self.frame_no;
        let Some(f) = self.thalamus.dict.get(fit.func).cloned() else { return 0 };
        let all: Vec<(ActRef, usize)> = fit.acts().collect();
        // the latest activation it landed on, and the earliest (D8, D15)
        let mut latest = all[0];
        let mut earliest = all[0];
        for &(r, node) in &all {
            let (Some(p), Some(l), Some(e)) = (self.find_act(r).map(|a| a.pos), self.find_act(latest.0).map(|a| a.pos), self.find_act(earliest.0).map(|a| a.pos)) else { continue };
            if later(&p, &l) { latest = (r, node); }
            if later(&e, &p) { earliest = (r, node); }
        }
        let Some(origin) = self.find_act(latest.0).cloned() else { return 0 };
        let mut minted = 0;
        let own = f.inter.filter(|n| self.thalamus.registry.get(*n).map_or(false, |x| x.alive));
        // read from its origin; a cycle has none, so from the node its call's phase starts at, the latest it landed on
        let found = if own.is_none() { self.reuse(&f, if f.has_cycle() { latest.1 } else { 0 }) } else { None };
        let neuron = match own.or(found) {
            Some(n) if own.is_some() => n,
            Some(n) => {
                // an interneuron never holds two functions whose nodes are the same neurons: the older gives way,
                // its relations leaving with it (R43, R54, R18)
                let mut mine: Vec<Sym> = f.nodes.clone();
                mine.sort_unstable();
                mine.dedup();
                let older: Vec<crate::dictionary::FuncId> = self.thalamus.dict.list_of_inter(n).iter().copied().filter(|g| {
                    self.thalamus.dict.get(*g).map_or(false, |g| {
                        let mut v: Vec<Sym> = g.nodes.clone();
                        v.sort_unstable();
                        v.dedup();
                        v == mine
                    })
                }).collect();
                for g in older {
                    if let Some(gone) = self.thalamus.dict.remove(g) {
                        self.forget_expansions(g);
                        self.retire(gone);
                    }
                }
                self.thalamus.dict.set_inter(f.id, n);
                self.stats_reused += 1;
                n
            }
            None => {
                minted = 1;
                let facts = NeuronFacts { kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes: origin.axes, alive: true };
                let id = self.thalamus.allot_neuron(facts, self.cfg.h);
                self.alive += 1;
                self.thalamus.dict.set_inter(f.id, id);
                id
            }
        };
        if self.frozen { self.frozen_last_func.insert(neuron, f.id); } else {
            self.ledger.retain(|x| *x != neuron);
            self.last_func.insert(neuron, f.id);
        }
        // its phase (D53): from its latest node back to its earliest, round the laps it read; a call of its own
        // function it landed on adds what that call stood for, so its half lap is counted with the rest (R23)
        let order = f.order_back(latest.1);
        let phase = if !order.is_empty() && order.contains(&latest.1) {
            let mut len = 0usize;
            for (r, node) in fit.acts() {
                if !order.contains(&node) { continue; }
                let own = self.find_act(r).and_then(|a| if a.kind == Kind::Inter && a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { self.thalamus.dict.find_pose(a.sym) } else { None });
                len += match own {
                    Some(p) => match (f.find_node(p.from), f.find_node(p.to)) { (Some(a), Some(b)) => f.stretch(a, b, p.laps), _ => 1 },
                    None => 1,
                };
            }
            let len = len.max(1);
            let c = order.len();
            Phase { from: f.nodes[latest.1], to: f.nodes[order[(len - 1) % c]], laps: ((len - 1) / c) as u32 }
        } else {
            let mut laps = fit.laps;
            for (c, _) in &fit.calls {
                if let Some(a) = self.find_act(*c) {
                    if a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { laps += self.thalamus.dict.find_pose(a.sym).map_or(0, |p| p.laps); }
                }
            }
            // where it took an earlier call of its own, the same call, its phase reaches as far back as that call's did (D54, D53)
            let mut to = f.nodes[earliest.1];
            {
                for (c, _) in &fit.calls {
                    if let Some(a) = self.find_act(*c) { if a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { if let Some(p) = self.thalamus.dict.find_pose(a.sym) { to = p.to; } } }
                }
            }
            Phase { from: f.nodes[latest.1], to, laps }
        };
        let sym = self.thalamus.dict.make_sym(neuron, phase);
        let covers: Vec<ActRef> = all.iter().map(|x| x.0).collect();
        let strong = covers.iter().filter_map(|r| self.find_act(*r)).any(|a| a.strong);
        let height = 1 + covers.iter().filter_map(|r| self.find_act(*r)).map(|a| a.height).max().unwrap_or(0);
        // what it stands for in space is the union of what it covers, and it sees one beyond that (D4)
        let (mut lo, mut hi) = (origin.pos, origin.pos);
        for c in covers.iter().filter_map(|r| self.find_act(*r)) { lo = lo.least(&c.lo); hi = hi.most(&c.hi); }
        // its reach goes back one lap: through its own earlier calls, to the one that stood at the node it stands at
        // now, where what it stands for comes back to itself (D4)
        let stands_at = f.nodes[latest.1];
        let (mut seen_lo, mut seen_hi) = (origin.pos, origin.pos);
        let mut walk: Vec<ActRef> = covers.clone();
        let mut walked: FxHashSet<ActRef> = FxHashSet::default();
        while let Some(r) = walk.pop() {
            if !walked.insert(r) { continue; }
            let Some(c) = self.find_act(r) else { continue };
            if c.neuron != neuron {
                seen_lo = seen_lo.least(&c.lo);
                seen_hi = seen_hi.most(&c.hi);
                continue;
            }
            // an earlier call of its own: the lap began where it stood at this node, and goes on through what it covered otherwise
            if self.thalamus.dict.find_pose(c.sym).map_or(true, |p| p.from == stands_at) { continue; }
            seen_lo = seen_lo.least(&c.pos);
            seen_hi = seen_hi.most(&c.pos);
            walk.extend(c.covers.iter().copied());
        }
        let reach = Pos::make(origin.axes.count(), |d| if origin.axes.has(d) { (origin.pos[d] - seen_lo[d]).max(seen_hi[d] - origin.pos[d]) + 1 } else { 0 });
        let act = Act {
            neuron, sym, kind: Kind::Inter, axes: origin.axes, pos: origin.pos, strong, reach, lo, hi,
            covered_at: None, coverers: Vec::new(), covers: covers.clone(),
            fired_by: Some(FiredBy { func: f.id, failed: fit.failed.clone() }),
            rels: Vec::new(), inferences: Vec::new(), opened_at: fno, level: level + 1, height, gone: false,
        };
        if !self.frozen { let r = self.reach_of.entry(neuron).or_insert(Pos::zero()); *r = r.most(&act.reach); self.reach_most = self.reach_most.max(act.reach[0]); }
        // the board span is the longest reach in time of any neuron, from the frame a call first has it (D9)
        self.span_s = self.span_s.max(act.reach[0]);
        let Some(fr) = self.find_frame_mut(fno) else { return minted };
        let me = ActRef { frame: fno, index: fr.acts.len() as u32 };
        fr.acts.push(act);
        self.electing.insert(me);
        for &(a, b) in &fit.edges {
            // a blocked relation is never covered; a covered one decides the kind of each end that has none (D24)
            if self.is_blocked(a, b) { continue; }
            // a refused relation is never covered (D24)
            if self.refused.contains(&edge(a, b)) { continue; }
            // the relation joins its two ends at this level of this frame (D24)
            self.joined.entry(a).or_default().push((b, me, f.id));
            self.joined.entry(b).or_default().push((a, me, f.id));
            let (ra, rb) = (self.find_joined_root(a), self.find_joined_root(b));
            // the two ends, and every activation of two sets that become one, are checked again for what to refuse
            self.jdirty.insert(a);
            self.jdirty.insert(b);
            if ra != rb {
                self.jparent.insert(ra, rb);
                let moved = self.jmembers.remove(&ra).unwrap_or_else(|| vec![ra]);
                let stay = self.jmembers.entry(rb).or_insert_with(|| vec![rb]);
                stay.extend(moved);
                self.jdirty.extend(stay.iter().copied());
            }
            self.decide_kind(a, b);
            if self.covered_rel.insert(edge(a, b)) {
                self.cov_inc.entry(a).or_default().push(b);
                self.cov_inc.entry(b).or_default().push(a);
            }
        }
        // a relation read through a call and the relation to that call are one board relation: covered under
        // either form, it is covered under both (R50, D5)
        {
            {
                let mut t: Vec<ActRef> = fit.thru.clone();
                t.extend(fit.calls.iter().map(|x| x.0));
                t.extend(fit.landed.iter().map(|x| x.0).filter(|r| self.find_act(*r).map_or(false, |a| !a.covers.is_empty())));
                t.sort();
                t.dedup();
                self.call_thru.insert(me, t);
                self.call_fit.insert(me, fit.clone());
            }
            { let mut all: Vec<(ActRef, ActRef)> = fit.edges.iter().map(|&(a, b)| edge(a, b)).collect(); all.sort(); all.dedup(); self.call_edges.insert(me, all); }
        }
        for r in &covers {
            if let Some(a) = self.find_act_mut(*r) {
                // it stands for the activation; the activation is covered once every relation of its own is (D24)
                if !a.coverers.contains(&me) { a.coverers.push(me); }
            }
        }
        // the call of the lap before took this lap: it is the same call, its phase that much longer (D53); the record
        // it grew from is out of the graph from now, and the grown record stands in its stead
        for &(c, _) in &fit.calls {
            let own = self.find_act(c).and_then(|ca| ca.fired_by.as_ref()).map_or(false, |fb| fb.func == f.id);
            if !own { continue; }
            self.grown.insert(c, me);
            if let Some(ca) = self.find_act_mut(c) { if ca.covered_at.is_none() { ca.covered_at = Some((fno - ca.pos[0] as i64).max(0) as u32); } }
        }
        // what the call covered, each at its offset from it, kept for as long as rows may name it (R28)
        {
            if let Some(ca) = self.find_act(me) {
                let rec: Vec<(Sym, Offset)> = self.list_under(ca).into_iter().filter_map(|u| self.find_act(u).map(|ua| (ua.sym, offset_between(ca, ua)))).collect();
                self.covered_by.insert(me, rec);
            }
        }
        // the rows that saw what it lands on take the call when that is covered, not before (D10, D24)
        // where the report built the strong call of a function an inference named weakly, the weak one goes (D40)
        if strong {
            let pos = origin.pos;
            if let Some(fr) = self.find_frame_mut(fno) {
                for (i, w) in fr.acts.iter_mut().enumerate() {
                    if i as u32 != me.index && !w.strong && !w.gone && w.neuron == neuron && w.pos == pos { w.gone = true; }
                }
            }
        }
        minted
    }
}
