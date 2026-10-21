//! Recognition (§5.2): the greedy cover (D28) of a row by the table's functions, each measured (D22).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::neuron::*;
use crate::types::*;

pub(crate) struct Claim {
    pub(crate) entry: CoverEntry,
    pub(crate) coverage: f64,
    pub(crate) price: f64,
}

impl Neuron {
    // ───────────────────────────── recognition (§5.2, D28) ─────────────────────────────

    /// Cover every dirty row by the greedy cover over the table. Returns the functions whose rows changed.
    pub(crate) fn recognize(&mut self, ctx: Ctx) -> FxHashSet<PatternId> {
        let mut changed: FxHashSet<PatternId> = FxHashSet::default();
        for i in 0..self.history.len() {
            if !self.history[i].dirty { continue; }
            for e in &self.history[i].cover { changed.insert(e.function); }
            let cover = self.greedy_cover(i, None, ctx);
            self.install_cover(i, cover);
            for e in &self.history[i].cover { changed.insert(e.function); }
            self.history[i].dirty = false;
        }
        changed
    }

    /// Set a row's cover and its owners (D17, D19): the first function credited with each neighbor, by
    /// covering it at a fixed node or binding it in a variable.
    pub(crate) fn install_cover(&mut self, i: usize, cover: Vec<CoverEntry>) {
        let fixed_places: Vec<Vec<usize>> = cover.iter().map(|e| {
            let f = self.function(e.function);
            e.binds.iter().filter(|b| f.map_or(false, |f| matches!(f.places.get(b.place).map(|x| x.0), Some(Place::Fixed(_))))).map(|b| b.place).collect()
        }).collect();
        let row = &mut self.history[i];
        let n = row.nbs.len();
        row.owner = vec![None; n];
        row.bound = vec![false; n];
        row.owner_self = None;
        for (k, e) in cover.iter().enumerate() {
            if e.credited_owner && row.owner_self.is_none() { row.owner_self = Some(e.function); }
            for b in &e.binds {
                let fixed = fixed_places[k].contains(&b.place);
                for &j in &b.nbs {
                    if row.owner[j].is_none() { row.owner[j] = Some(e.function); row.bound[j] = !fixed; }
                }
            }
        }
        row.cover = cover;
    }

    /// The greedy cover of one row (D28) by every function of the table, `exclude` left out for the
    /// margin's "without" (D30).
    pub(crate) fn greedy_cover(&self, i: usize, exclude: Option<PatternId>, ctx: Ctx) -> Vec<CoverEntry> {
        let row = &self.history[i];
        let mut taken = vec![false; row.nbs.len()];
        let mut owner_taken = false;
        let mut cover: Vec<CoverEntry> = Vec::new();
        let wf = self.which_function_bits();
        let mut used: FxHashSet<PatternId> = FxHashSet::default();
        loop {
            let mut best: Option<Claim> = None;
            for f in &self.functions {
                if Some(f.id) == exclude || used.contains(&f.id) { continue; }
                let Some(c) = self.measure(row, f, &taken, owner_taken, wf) else { continue };
                let better = match &best {
                    None => true,
                    Some(b) => {
                        let (r, br) = (c.coverage / c.price, b.coverage / b.price);
                        r > br || (r == br && c.entry.function < b.entry.function)
                    }
                };
                if better { best = Some(c); }
            }
            let Some(b) = best else { break };
            if !(b.coverage > b.price) { break; }
            for pb in &b.entry.binds { for &j in &pb.nbs { taken[j] = true; } }
            if b.entry.credited_owner { owner_taken = true; }
            used.insert(b.entry.function);
            cover.push(b.entry);
        }
        let _ = ctx;
        cover
    }

    /// One function measured against a row (D22): the owner's activation where nothing took it, each fixed
    /// node covering the neighbors that stand at its offset, each variable binding what stands at its place;
    /// a place where nothing it could take stands fails, and costs `log₂ |p|`. The price is what the bidder
    /// cost, which function, which member per class place and once per parameter, and the failures. Nothing
    /// is measured where no place but the owner holds.
    pub(crate) fn measure(&self, row: &Row, f: &Function, taken: &[bool], owner_taken: bool, wf: f64) -> Option<Claim> {
        let mut binds: Vec<PlaceBind> = Vec::new();
        let mut args: Vec<(usize, NeuronId)> = Vec::new();
        let mut credited: FxHashSet<usize> = FxHashSet::default();
        let mut coverage = 0.0;
        let mut choice = 0.0;
        let mut failed = 0u32;
        let mut holds_any = false;
        for (pi, (place, off)) in f.places.iter().enumerate().skip(1) {
            match place {
                Place::Owner | Place::Param(_) => {}
                Place::Fixed(n) => {
                    let found = self.nbs_at(row, off, Some(*n));
                    if found.is_empty() { failed += 1; continue; }
                    holds_any = true;
                    for &j in &found { if !taken[j] && credited.insert(j) { coverage += row.credit(j); } }
                    binds.push(PlaceBind { place: pi, nbs: found });
                }
                Place::Class(k) => {
                    let Some(c) = self.class(*k) else { failed += 1; continue };
                    choice += c.choice();
                    let cands: Vec<usize> = self.nbs_at(row, off, None).into_iter().filter(|&j| c.has(row.nbs[j].neuron)).collect();
                    let pick = cands.iter().copied().find(|&j| !taken[j] && !credited.contains(&j)).or_else(|| cands.first().copied());
                    let Some(j) = pick else { failed += 1; continue };
                    holds_any = true;
                    if !taken[j] && credited.insert(j) { coverage += row.credit(j); }
                    binds.push(PlaceBind { place: pi, nbs: vec![j] });
                    args.push((pi, row.nbs[j].neuron));
                }
            }
        }
        // parameters: the value is the neuron at most of their places, the latest on a tie (D44)
        for (k, param) in f.params.iter().enumerate() {
            let pl = f.param_places(k as u32);
            choice += param.choice();
            let per: Vec<Vec<usize>> = pl.iter().map(|&pi| self.nbs_at(row, &f.places[pi].1, None)).collect();
            let mut tally: FxHashMap<NeuronId, (usize, i32)> = FxHashMap::default();
            for js in &per {
                let mut seen: Vec<NeuronId> = Vec::new();
                for &j in js {
                    let n = row.nbs[j].neuron;
                    if seen.contains(&n) { continue; }
                    seen.push(n);
                    let e = tally.entry(n).or_insert((0, i32::MIN));
                    e.0 += 1;
                    e.1 = e.1.max(row.nbs[j].delta[0]);
                }
            }
            let value = tally.iter().max_by(|a, b| (a.1).0.cmp(&(b.1).0).then((a.1).1.cmp(&(b.1).1)).then(b.0.cmp(a.0))).map(|(n, _)| *n);
            for (x, &pi) in pl.iter().enumerate() {
                let pick = value.and_then(|v| {
                    let cands: Vec<usize> = per[x].iter().copied().filter(|&j| row.nbs[j].neuron == v).collect();
                    cands.iter().copied().find(|&j| !taken[j] && !credited.contains(&j)).or_else(|| cands.first().copied())
                });
                match pick {
                    Some(j) => {
                        holds_any = true;
                        if !taken[j] && credited.insert(j) { coverage += row.credit(j); }
                        binds.push(PlaceBind { place: pi, nbs: vec![j] });
                        args.push((pi, row.nbs[j].neuron));
                    }
                    None => failed += 1,
                }
            }
        }
        if !holds_any { return None; }
        let credited_owner = !owner_taken;
        if credited_owner { coverage += row.own_cost; }
        let price = row.own_cost + wf + choice + failed as f64 * log2(f.p_size().max(1) as f64);
        binds.sort_by_key(|b| b.place);
        args.sort();
        Some(Claim { entry: CoverEntry { function: f.id, binds, args, failed, credited_owner, choice }, coverage, price })
    }

    /// The neighbors standing at an offset from the owner's activation: every one whose exact offset falls
    /// in the written offset's group (D6, D26). Where a neuron is named, a child standing for it at that
    /// offset matches as well (D22, R28).
    pub(crate) fn nbs_at(&self, row: &Row, off: &Offset, neuron: Option<NeuronId>) -> Vec<usize> {
        let mut out = Vec::new();
        for (j, nb) in row.nbs.iter().enumerate() {
            if neuron.map_or(false, |n| nb.neuron != n) { continue; }
            if self.round(&nb.delta) == *off { out.push(j); }
        }
        if let Some(n) = neuron {
            for (j, nb) in row.nbs.iter().enumerate() {
                if out.contains(&j) { continue; }
                if nb.stands_for.iter().any(|(m, d)| *m == n && self.round(d) == *off) { out.push(j); }
            }
        }
        out
    }
}
