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
        for (k, _) in fit.acts() {
            let Some(a) = self.act(k) else { continue };
            let Some(fb) = &a.fired_by else { continue };
            if a.coverers.iter().any(|x| self.electing.contains(x)) { continue; }
            c += fb.failed.len() as f64;
        }
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
    /// they read before one landing a node on a call through its pose; a function with an interneuron before one
    /// without; the older interneuron; the function that joined first; the fit from the older activation.
    fn rank(&self, fit: &Fit) -> (bool, u8, u64, u64, ActRef) {
        let f = self.dict.get(fit.func);
        let via_call = !fit.calls.is_empty();
        match f.and_then(|f| f.inter) {
            Some(n) => (via_call, 0, n as u64, 0, fit.entry.1),
            None => (via_call, 1, 0, f.map_or(u64::MAX, |f| f.joined), fit.entry.1),
        }
    }

    /// The reuse search (R43, D36): an interneuron that already stands for what this function's would. The
    /// function's expansion is read from its origin; an interneuron met in it, not at the origin, no further back
    /// than that interneuron's own length, that far included, whose expansion over exactly that length is the same as the function's. The
    /// nearest meeting first, the older interneuron on a tie.
    fn reuse(&self, f: &Func) -> Option<NeuronId> {
        let lay = f.layout();
        let mut out: Vec<(NeuronId, Pos)> = Vec::new();
        let mut met: Vec<(NeuronId, Pos)> = Vec::new();
        for (i, m) in f.nodes.iter().enumerate() {
            let mut q = [0i32; MAX_DIMS];
            for d in 0..MAX_DIMS { q[d] += lay.position[i][d]; }
            let mn = self.dict.neuron_of(*m);
            if self.thalamus.get(mn).map_or(false, |x| x.alive && x.kind == Kind::Inter) { met.push((mn, q)); }
            self.expand_from(*m, q, i64::MIN, false, &mut Vec::new(), &mut out, Some(&mut met));
        }
        met.retain(|(_, p)| *p != [0; MAX_DIMS]);
        met.sort_by_key(|(n, p)| (p.iter().map(|x| x.unsigned_abs()).sum::<u32>(), *n));
        met.dedup();
        let mut tried: Vec<NeuronId> = Vec::new();
        for (n, p) in met {
            if tried.contains(&n) { continue; }
            tried.push(n);
            if self.dict.of_inter(n).is_empty() { continue; }
            let mut own: Vec<(NeuronId, Pos)> = Vec::new();
            self.expand_from(n, [0; MAX_DIMS], i64::MIN, false, &mut Vec::new(), &mut own, None);
            if own.is_empty() { continue; }
            // its length in each dimension: the extent its expansion spans back from its origin
            let (mut lo, mut hi) = ([0i32; MAX_DIMS], [0i32; MAX_DIMS]);
            for (_, q) in &own { for d in 0..MAX_DIMS { lo[d] = lo[d].min(q[d]); hi[d] = hi[d].max(q[d]); } }
            if !(0..MAX_DIMS).all(|d| p[d].abs() <= hi[d] - lo[d] + 1) { continue; }
            let inside = |q: &Pos| (0..MAX_DIMS).all(|d| q[d] >= lo[d] && q[d] <= hi[d]);
            let mut a: Vec<(NeuronId, Pos)> = out.iter().filter(|(_, q)| inside(q)).copied().collect();
            let mut b = own;
            a.sort();
            b.sort();
            if a == b { return Some(n); }
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
        // its pose (D53): the laps it read, and those of a call of its own function it landed on, together (R23)
        let mut laps = fit.laps;
        for (c, _) in &fit.calls {
            if let Some(a) = self.act(*c) {
                if a.fired_by.as_ref().map_or(false, |fb| fb.func == f.id) { laps += self.dict.pose_of(a.sym).map_or(0, |p| p.laps); }
            }
        }
        let pose = Pose { from: f.nodes[latest.1], to: f.nodes[earliest.1], dir: if f.has_cycle() { 1 } else { 0 }, laps };
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
        for r in &covers {
            if let Some(a) = self.act_mut(*r) {
                if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); }
                if !a.coverers.contains(&me) { a.coverers.push(me); }
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
                    writes.push((vn, v, *r, Nb { act: me, sym, off, base: *r }));
                }
            }
            let mut viewers: Vec<(NeuronId, ActRef)> = Vec::new();
            for (vn, v, old, nb) in writes {
                if self.thalamus[vn].replace_seen(v, old, nb) && !viewers.contains(&(vn, v)) { viewers.push((vn, v)); }
            }
            if let Some(a) = self.act_mut(me) { a.seen_by = viewers; }
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
        minted
    }
}
