//! The election (§6.2, R22, R24, D28, D24) and allocation (§6.3, R16): every fit of this level, taken best
//! first, each while what it covers costs more than its call; each accepted fit fires its function's
//! interneuron at the next level, at the fit's origin.

use rustc_hash::FxHashSet;

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
    fn rank(&self, fit: &Fit) -> (u8, u64, u64, ActRef) {
        let f = self.dict.get(fit.func);
        match f.and_then(|f| f.inter) {
            Some(n) => (0, n as u64, 0, fit.origin),
            None => (1, 0, f.map_or(u64::MAX, |f| f.joined), fit.origin),
        }
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
            for &(a, b) in &fit.edges { edges.insert(edge(a, b)); }
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
        let neuron = match f.inter.filter(|n| self.thalamus.get(*n).map_or(false, |x| x.alive)) {
            Some(n) => n,
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
        let act = Act {
            neuron, kind: Kind::Inter, axes: origin.axes, pos: origin.pos, strong, reach: f.reach(origin.axes),
            covered_at: None, coverer: None, covers: covers.clone(),
            fired_by: Some(FiredBy { func: f.id, failed: fit.failed.clone() }),
            rels: Vec::new(), inferences: Vec::new(), opened_at: fno, level: level + 1, height, gone: false,
        };
        let Some(fr) = self.frame_mut(fno) else { return minted };
        let me = ActRef { frame: fno, index: fr.acts.len() as u32 };
        fr.acts.push(act);
        for r in &covers {
            if let Some(a) = self.act_mut(*r) {
                if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); a.coverer = Some(me); }
            }
        }
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
