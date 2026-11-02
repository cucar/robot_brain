//! The election (§6.2, R22, R24, D28, D24) and allocation (§6.3, R16): every fit of this level, taken best
//! first, each while what it covers costs more than its call; each accepted fit fires its function's
//! interneuron at the next level, at the fit's origin.

use rustc_hash::FxHashSet;

use crate::function::Func;
use crate::machine::*;
use crate::neuron::*;
use crate::recognize::{edge, Fit};
use crate::types::*;

impl Machine {
    /// What a fit covers against the board as it stands (R22): each board relation its true relations stand in for
    /// that no accepted call covers, one edge each, and each call it lands on as a node, at that call's count.
    fn coverage(&self, fit: &Fit, edges: &FxHashSet<(ActRef, ActRef)>, calls: &FxHashSet<ActRef>) -> f64 {
        let mut c = 0.0;
        let mut seen: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        for &(a, b) in &fit.edges {
            let e = edge(a, b);
            if edges.contains(&e) || !seen.insert(e) { continue; }
            // experiment: within the election too, an activation is covered per combination of dimensions, so an
            // edge whose combination an accepted call already covers at either end earns nothing (D5)
            if let (Some(aa), Some(ab)) = (self.act(a), self.act(b)) {
                let off = offset_between(aa, ab);
                if aa.covered_for(&off) || ab.covered_for(&off) { continue; }
            }
            c += 1.0;
        }
        for k in &fit.calls {
            if calls.contains(k) { continue; }
            if let Some(a) = self.act(*k) { c += a.fired_by.as_ref().map_or(0.0, |fb| fb.failed.len() as f64); }
        }
        c
    }

    /// The tie-break of two fits that measure the same (R24): a function with an interneuron before one without,
    /// the older interneuron, the function that joined first, the fit from the older activation.
    fn rank(&self, fit: &Fit) -> (bool, u8, u64, u64, ActRef) {
        let f = self.dict.get(fit.func);
        // a fit whose relations land on the board relations as they read, before one landing a node on a call
        // through its origin (D54)
        let via_origin = !fit.calls.is_empty();
        match f.and_then(|f| f.inter) {
            Some(n) => (via_origin, 0, n as u64, 0, fit.origin),
            None => (via_origin, 1, 0, f.map_or(u64::MAX, |f| f.joined), fit.origin),
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
            for d in 0..MAX_DIMS { q[d] += lay.place[i][d]; }
            if self.thalamus.get(*m).map_or(false, |x| x.alive && x.kind == Kind::Inter) { met.push((*m, q)); }
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

    /// The greedy cover over the fits of this level (D28, R24). Returns how many were accepted and allocated.
    pub(crate) fn elect(&mut self, level: u32, mut fits: Vec<Fit>) -> (usize, usize) {
        let mut edges: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        let mut calls: FxHashSet<ActRef> = FxHashSet::default();
        let (mut accepted, mut minted) = (0, 0);
        loop {
            let mut best: Option<(usize, f64, f64)> = None;
            for (i, fit) in fits.iter().enumerate() {
                let cov = self.coverage(fit, &edges, &calls);
                let price = fit.failed.len() as f64;
                if !(cov > price) { continue; }
                let better = match best {
                    None => true,
                    Some((j, bc, bp)) => {
                        // a fit priced at nothing first, the highest coverage among them; then the best ratio
                        let key = |c: f64, p: f64| if p == 0.0 { (1, c) } else { (0, c / p) };
                        let (ka, kb) = (key(cov, price), key(bc, bp));
                        ka > kb || (ka == kb && self.rank(fit) < self.rank(&fits[j]))
                    }
                };
                if better { best = Some((i, cov, price)); }
            }
            let Some((i, _, _)) = best else { break };
            let fit = fits.swap_remove(i);
            for &(a, b) in &fit.edges { if edges.insert(edge(a, b)) { self.covered_edges.push(edge(a, b)); } }
            for k in &fit.calls { calls.insert(*k); }
            minted += self.accept(&fit, level);
            accepted += 1;
        }
        (accepted, minted)
    }

    /// Accept a fit (R24, R16): what its fixed nodes landed on and the calls it landed on are covered from this
    /// frame (D10, D24), and its function's interneuron fires at the next level at the fit's origin (D53), allocated
    /// where the function has none. Returns 1 where it allocated.
    fn accept(&mut self, fit: &Fit, level: u32) -> usize {
        let fno = self.frame_no;
        let Some(origin) = self.act(fit.origin).cloned() else { return 0 };
        let Some(f) = self.dict.get(fit.func).cloned() else { return 0 };
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
        let covers: Vec<ActRef> = fit.landed.iter().chain(fit.calls.iter()).copied().collect();
        let strong = covers.iter().filter_map(|r| self.act(*r)).any(|a| a.strong);
        let height = 1 + covers.iter().filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        // what it stands for in space is the union of what it covers, and it sees one beyond that (D4)
        let (mut lo, mut hi) = (origin.pos, origin.pos);
        for c in covers.iter().filter_map(|r| self.act(*r)) { for d in 1..MAX_DIMS { lo[d] = lo[d].min(c.lo[d]); hi[d] = hi[d].max(c.hi[d]); } }
        let mut reach = f.reach(origin.axes);
        for d in 1..MAX_DIMS { if origin.axes & (1 << d) != 0 { reach[d] = reach[d].max((origin.pos[d] - lo[d]).max(hi[d] - origin.pos[d]) + 1); } }
        let act = Act {
            neuron, kind: Kind::Inter, axes: origin.axes, pos: origin.pos, strong, reach, lo, hi,
            covered_at: None, coverers: Vec::new(), covered_in: 0, covers: covers.clone(),
            fired_by: Some(FiredBy { func: f.id, failed: fit.failed.clone() }),
            rels: Vec::new(), inferences: Vec::new(), opened_at: fno, level: level + 1, height, gone: false,
        };
        { let r = self.reach_of.entry(neuron).or_insert([0; MAX_DIMS]); for d in 0..MAX_DIMS { r[d] = r[d].max(act.reach[d]); } }
        let Some(fr) = self.frame_mut(fno) else { return minted };
        let me = ActRef { frame: fno, index: fr.acts.len() as u32 };
        fr.acts.push(act);
        for r in &covers {
            if let Some(a) = self.act_mut(*r) {
                if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); }
                if !a.coverers.contains(&me) { a.coverers.push(me); }
            }
        }
        // it covers each activation its true relations land at in the combinations those relations run in (D24)
        let marks: Vec<(ActRef, u8)> = fit.edges.iter().filter_map(|&(u, w)| {
            let (au, aw) = (self.act(u)?, self.act(w)?);
            Some([(u, combination(&offset_between(au, aw))), (w, combination(&offset_between(au, aw)))])
        }).flatten().collect();
        for (r, c) in marks { if let Some(a) = self.act_mut(r) { a.covered_in |= c; } }
        // where the report built the strong call of a function an inference placed weakly, the weak one goes (D40)
        if strong {
            let pos = origin.pos;
            if let Some(fr) = self.frame_mut(fno) {
                for (i, w) in fr.acts.iter_mut().enumerate() {
                    if i as u32 != me.index && !w.strong && !w.gone && w.neuron == neuron && w.pos == pos { w.gone = true; }
                }
            }
        }
        if self.trace { eprintln!("[accept f{} l{}] {} at {:?} covers {} failed {:?} -> #{}", fno, level, self.dict.describe(&f), origin.pos, covers.len(), fit.failed, neuron); }
        minted
    }
}
