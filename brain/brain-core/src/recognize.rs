//! Recognition (§6.1): finding (R53), placing (D31, D54), what the neurons answer at the count step (D47, R50),
//! and grouping: relations joining, leaving and starting functions by their worth (D57, R50, R54, R18).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::function::*;
use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// One fit of a function on the board (D15, D54): where its origin landed, the activations its fixed nodes came
/// true on, the calls it landed on as a node, the board relations its true relations stand in for, and the
/// relations that did not come true.
#[derive(Clone, Debug)]
pub(crate) struct Fit {
    pub func: FuncId,
    pub origin: ActRef,
    pub landed: Vec<ActRef>,
    pub calls: Vec<ActRef>,
    pub edges: Vec<(ActRef, ActRef)>,
    pub failed: Vec<usize>,
}

/// A relation a neuron returned at this level (R21): the activation that returned it, the other end, the
/// offset from the one to the other, its strength, and its one form.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fresh {
    pub near: ActRef,
    pub far: ActRef,
    pub off: Offset,
    pub strength: u32,
    pub rel: Rel,
}

fn add(a: &Pos, b: &Offset) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }
fn neg(o: &Offset) -> Offset { let mut r = *o; for x in r.iter_mut() { *x = -*x; } r }

pub(crate) fn edge(a: ActRef, b: ActRef) -> (ActRef, ActRef) { if a <= b { (a, b) } else { (b, a) } }

impl Machine {
    /// The neuron a call stands for (D54): the origin of the function that fired it.
    pub(crate) fn call_origin(&self, a: &Act) -> Option<NeuronId> {
        if a.kind != Kind::Inter { return None; }
        a.fired_by.as_ref().and_then(|fb| self.func(fb.func)).map(|f| f.origin())
    }

    /// The origins of an interneuron's functions, for reading a neighbor that is a call as its origin's neuron.
    pub(crate) fn inter_origins(&self, n: NeuronId) -> Vec<NeuronId> {
        let mut v: Vec<NeuronId> = self.dict.of_inter(n).iter().filter_map(|id| self.dict.get(*id)).map(|f| f.origin()).collect();
        if let Some(f) = self.last_func.get(&n).and_then(|id| self.func(*id)) { v.push(f.origin()); }
        v.sort_unstable();
        v.dedup();
        v
    }

    // ───────────────────────────── placing (D31, D54) ─────────────────────────────

    /// From a landed activation, the board relation at `delta` whose far end is `want`, or a call standing for
    /// it there (D54); the one an earlier frame holds first (R24). Whether it is a call.
    fn step(&self, from: ActRef, delta: Offset, want: NeuronId, used: &FxHashSet<ActRef>) -> Option<(ActRef, bool)> {
        let a = self.act(from)?;
        let fno = self.frame_no;
        let target = add(&a.pos, &delta);
        let mut others: Vec<ActRef> = a.rels.iter().filter(|br| br.off == delta).map(|br| br.other).collect();
        if let Some(fr) = self.frames.iter().find(|f| f.no == target[0] as i64) {
            for (i, w) in fr.acts.iter().enumerate() {
                if w.pos != target { continue; }
                let back = neg(&delta);
                if w.rels.iter().any(|br| br.other == from && br.off == back) { others.push(ActRef { frame: fr.no, index: i as u32 }); }
            }
        }
        others.sort();
        others.dedup();
        let mut best: Option<(ActRef, bool)> = None;
        for o in others {
            if used.contains(&o) { continue; }
            let Some(w) = self.act(o) else { continue };
            if !w.on_apex() || !self.stands(w, fno) { continue; }
            let is = if w.neuron == want { Some(false) } else if self.call_origin(w) == Some(want) { Some(true) } else { None };
            let Some(call) = is else { continue };
            let better = match best { None => true, Some((b, bc)) => (call, o) < (bc, b) };
            if better { best = Some((o, call)); }
        }
        best
    }

    /// Place a function at one activation that can stand at its origin (D31): read out from the origin relation by
    /// relation, round the cycle lap after lap while the relation that closes it lands, and the relations that
    /// hang off the cycle from their node on the last lap.
    pub(crate) fn fit(&self, f: &Func, origin: ActRef) -> Fit {
        let lay = f.layout();
        let n = f.nodes.len();
        let mut at: Vec<Option<ActRef>> = vec![None; n];
        let mut via_call = vec![false; n];
        at[0] = Some(origin);
        let mut used: FxHashSet<ActRef> = FxHashSet::default();
        used.insert(origin);
        let mut landed = vec![origin];
        let mut calls = Vec::new();
        let mut edges = Vec::new();
        let mut came = vec![false; f.rels.len()];
        // the cycle: its path once round, from the node the closing relation reaches back to where it starts
        let cycle_path: Vec<(usize, usize, usize, Offset)> = match lay.closing { Some((_, ua, ub, _)) => Func::tree_path(&lay, ub, ua), None => vec![] };
        let mut on_cycle: Vec<bool> = vec![false; n];
        if let Some((_, ua, ub, _)) = lay.closing { on_cycle[ua] = true; on_cycle[ub] = true; for s in &cycle_path { on_cycle[s.1] = true; on_cycle[s.2] = true; } }
        let path_rels: Vec<usize> = cycle_path.iter().map(|s| s.0).collect();
        let try_step = |this: &Machine, ri: usize, u: usize, v: usize, off: Offset, at: &mut Vec<Option<ActRef>>, via_call: &mut Vec<bool>, used: &mut FxHashSet<ActRef>, landed: &mut Vec<ActRef>, calls: &mut Vec<ActRef>, edges: &mut Vec<(ActRef, ActRef)>, came: &mut Vec<bool>| -> Option<bool> {
            let au = at[u]?;
            if via_call[u] { return None; }
            let (w, call) = this.step(au, off, f.nodes[v], used)?;
            came[ri] = true;
            edges.push((au, w));
            used.insert(w);
            at[v] = Some(w);
            via_call[v] = call;
            if call { calls.push(w); } else { landed.push(w); }
            Some(call)
        };
        // the tree, the relations hanging off the cycle left for after it has run
        let mut deferred = Vec::new();
        for &(ri, u, v, off) in &lay.tree {
            if lay.closing.is_some() && on_cycle[u] && !path_rels.contains(&ri) { deferred.push((ri, u, v, off)); continue; }
            if at[u].is_none() { deferred.push((ri, u, v, off)); continue; }
            try_step(self, ri, u, v, off, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came);
        }
        // round the cycle, lap after lap, while the closing relation lands (D54)
        if let Some((ri, ua, ub, off)) = lay.closing {
            let mut laps = 0;
            loop {
                laps += 1;
                if laps > 4096 { break; }
                let Some(au) = at[ua] else { break };
                if via_call[ua] { break; }
                let Some((w, call)) = self.step(au, off, f.nodes[ub], &used) else { break };
                came[ri] = true;
                edges.push((au, w));
                used.insert(w);
                if call { calls.push(w); at[ub] = Some(w); via_call[ub] = true; break; }
                landed.push(w);
                at[ub] = Some(w);
                via_call[ub] = false;
                // the lap before, from the node it reached
                let mut whole = true;
                for &(pri, pu, pv, poff) in &cycle_path {
                    if try_step(self, pri, pu, pv, poff, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came).is_none() { whole = false; break; }
                }
                if !whole { break; }
            }
        }
        // what hangs off the cycle, from its node on the last lap
        let mut progress = true;
        while progress && !deferred.is_empty() {
            progress = false;
            let mut rest = Vec::new();
            for (ri, u, v, off) in deferred {
                if at[u].is_some() {
                    try_step(self, ri, u, v, off, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came);
                    progress = true;
                } else { rest.push((ri, u, v, off)); }
            }
            deferred = rest;
        }
        // the relation that closes a cycle is never a failure: the fit ends where it stops landing, and a lap read
        // once is the cycle run once (D54)
        let closing_ri = lay.closing.map(|c| c.0);
        let failed: Vec<usize> = (0..f.rels.len()).filter(|ri| !came[*ri] && Some(*ri) != closing_ri).collect();
        Fit { func: f.id, origin, landed, calls, edges, failed }
    }

    /// Every fit of a function this frame (R53): one per activation of the present frame that can stand at its
    /// origin.
    pub(crate) fn place(&self, id: FuncId) -> Vec<Fit> {
        let Some(f) = self.dict.get(id) else { return vec![] };
        let fno = self.frame_no;
        let mut out: Vec<Fit> = Vec::new();
        let Some(fr) = self.frames.iter().find(|fr| fr.no == fno) else { return out };
        for (i, a) in fr.acts.iter().enumerate() {
            if a.neuron != f.origin() || !a.on_apex() { continue; }
            let fit = self.fit(f, ActRef { frame: fno, index: i as u32 });
            if fit.edges.is_empty() { continue; }
            let mut key: Vec<ActRef> = fit.landed.iter().chain(fit.calls.iter()).copied().collect();
            key.sort();
            if out.iter().any(|o| { let mut k: Vec<ActRef> = o.landed.iter().chain(o.calls.iter()).copied().collect(); k.sort(); k == key }) { continue; }
            out.push(fit);
        }
        out
    }

    /// Finding (R53): the functions that hold a relation returned at this level, as it is.
    pub(crate) fn find(&self, fresh: &[Fresh]) -> Vec<FuncId> {
        let mut v: Vec<FuncId> = fresh.iter().filter_map(|e| self.dict.holder(&e.rel)).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    // ───────────────────────────── what the neurons answer (D47, R50) ─────────────────────────────

    /// Whether a row holds a relation from its neuron (D47): the neighbor at that offset, or a call standing for
    /// it there (D54).
    fn row_holds(&self, row: &Row, far: NeuronId, off: &Offset, origins: &mut FxHashMap<NeuronId, Vec<NeuronId>>) -> bool {
        row.nbs.iter().any(|nb| nb.off == *off && (nb.neuron == far || origins.entry(nb.neuron).or_insert_with(|| self.inter_origins(nb.neuron)).contains(&far)))
    }

    /// The neuron that answers for a relation, from its own side (D47, R50): in time the later end; in space the
    /// end with more activations behind it.
    fn recorder(&self, r: &Rel) -> (NeuronId, NeuronId, Offset) {
        let mut best = r.recorders()[0];
        for c in r.recorders() {
            let len = |n: NeuronId| self.thalamus.get(n).map_or(0, |x| x.history.len());
            if len(c.0) > len(best.0) { best = c; }
        }
        best
    }

    /// How many rows of its neuron hold a relation (D47), reading a call as its origin's neuron.
    pub(crate) fn strength_of(&self, r: &Rel) -> u32 {
        let (n, far, off) = self.recorder(r);
        let Some(nn) = self.thalamus.get(n) else { return 0 };
        let mut origins = FxHashMap::default();
        nn.history.iter().filter(|row| self.row_holds(row, far, &off, &mut origins)).count() as u32
    }

    /// `with` and `without` for one relation of a set (R50): over the rows of the neuron that answers for it, how
    /// many hold the set's other relations at that neuron together with it, and how many hold them and not it.
    /// Where the neuron holds none of the others, every row that holds it is `with`, and nothing is `without`.
    pub(crate) fn answer(&self, rels: &[Rel], r: &Rel) -> (u32, u32) {
        let (n, far, off) = self.recorder(r);
        let Some(nn) = self.thalamus.get(n) else { return (0, 0) };
        let mut cond: Vec<(NeuronId, Offset)> = Vec::new();
        for s in rels {
            if s == r { continue; }
            for (m, f2, o2) in s.recorders() { if m == n { cond.push((f2, o2)); } }
        }
        let mut origins = FxHashMap::default();
        let (mut with, mut without) = (0, 0);
        for row in nn.history.iter() {
            if !cond.iter().all(|(f2, o2)| self.row_holds(row, *f2, o2, &mut origins)) { continue; }
            if self.row_holds(row, far, &off, &mut origins) { with += 1; } else if !cond.is_empty() { without += 1; }
        }
        (with, without)
    }

    /// A relation's worth in a set of relations (D57): `2 · with − without − 1` where the set holds others, its
    /// strength less one alone.
    pub(crate) fn worth_in(&self, rels: &[Rel], r: &Rel) -> (f64, f64) {
        if rels.len() <= 1 { let s = self.strength_of(r) as f64; return (s - 1.0, 1.0); }
        let (w, wo) = self.answer(rels, r);
        let near = if w + wo == 0 { 0.0 } else { w as f64 / (w + wo) as f64 };
        (2.0 * w as f64 - wo as f64 - 1.0, near)
    }

    // ───────────────────────────── grouping (R50, R54, R18) ─────────────────────────────

    /// The relation a board relation would be to a function at an activation it shares with the function's fit or
    /// call (R50): from the node there to the other end, a call read as its origin's neuron where that is a node
    /// of the function (D54). None where it is no candidate.
    fn candidate(&self, f: &Func, shared: ActRef, as_node: NeuronId, origin_pos: &Pos, other: ActRef) -> Option<Rel> {
        let (sa, oa) = (self.act(shared)?, self.act(other)?);
        let mut read = oa.neuron;
        if let Some(o) = self.call_origin(oa) { if f.node_of(o).is_some() { read = o; } }
        let mut off = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS { off[d] = oa.pos[d] - sa.pos[d]; }
        let rel = canon(as_node, read, off);
        if f.holds(&rel) || self.dict.holder(&rel).is_some() && self.dict.holder(&rel) == Some(f.id) { return None; }
        if f.node_of(read).is_some() {
            // it would close the cycle, where the function has none
            if f.has_cycle() { return None; }
        } else {
            // a new node may not be the origin: later in time, or at a lesser place in the origin's frame
            if oa.pos[0] > origin_pos[0] { return None; }
            if oa.pos[0] == origin_pos[0] && oa.pos[1..] < origin_pos[1..] { return None; }
            if f.has_cycle() {
                let lay = f.layout();
                if let Some((_, ua, ub, _)) = lay.closing {
                    let path = Func::tree_path(&lay, ub, ua);
                    let mut cyc: Vec<usize> = vec![ua, ub];
                    for s in &path { cyc.push(s.1); cyc.push(s.2); }
                    let node = f.node_of(as_node).unwrap_or(0);
                    let mut dims = 0u8;
                    for ri in path.iter().map(|s| s.0).chain(std::iter::once(lay.closing.unwrap().0)) {
                        for d in 0..MAX_DIMS { if f.rels[ri].off[d] != 0 { dims |= 1 << d; } }
                    }
                    let mut mine = 0u8;
                    for d in 0..MAX_DIMS { if off[d] != 0 { mine |= 1 << d; } }
                    if cyc.contains(&node) && dims & mine != 0 { return None; }
                }
            }
        }
        Some(rel)
    }

    /// Grouping at one level (R50, R54, R18): each candidate goes where its worth is highest, each relation of a
    /// function placed here leaves where it is worth more alone or nothing anywhere, and each recurring relation
    /// no function holds starts one. Returns the functions that changed.
    pub(crate) fn group(&mut self, fits: &[Fit], fresh: &[Fresh]) -> Vec<FuncId> {
        let mut changed: Vec<FuncId> = Vec::new();
        // the activations each function touches here: what its fits landed on, and its calls on the board
        let mut touch: FxHashMap<FuncId, Vec<(ActRef, NeuronId, Pos)>> = FxHashMap::default();
        // the board edges each function's fits already land on: one edge is never two of its relations
        let mut taken: FxHashMap<FuncId, FxHashSet<(ActRef, ActRef)>> = FxHashMap::default();
        for fit in fits {
            let Some(f) = self.dict.get(fit.func) else { continue };
            taken.entry(f.id).or_default().extend(fit.edges.iter().map(|&(a, b)| edge(a, b)));
            let opos = self.act(fit.origin).map(|a| a.pos).unwrap_or([0; MAX_DIMS]);
            for &l in &fit.landed { if let Some(a) = self.act(l) { touch.entry(f.id).or_default().push((l, a.neuron, opos)); } }
        }
        let fno = self.frame_no;
        for e in fresh {
            for end in [e.near, e.far] {
                let Some(a) = self.act(end) else { continue };
                if a.kind != Kind::Inter || !a.on_apex() || !self.stands(a, fno) { continue; }
                let Some(fb) = &a.fired_by else { continue };
                let Some(f) = self.dict.get(fb.func) else { continue };
                touch.entry(f.id).or_default().push((end, f.origin(), a.pos));
            }
        }
        // the candidates (R50)
        let mut cands: FxHashMap<Rel, Vec<FuncId>> = FxHashMap::default();
        let mut strength: FxHashMap<Rel, u32> = FxHashMap::default();
        for e in fresh {
            if e.strength < 2 { continue; }
            for (fid, list) in &touch {
                let Some(f) = self.dict.get(*fid) else { continue };
                for &(act, as_node, opos) in list {
                    let other = if e.near == act { e.far } else if e.far == act { e.near } else { continue };
                    if taken.get(fid).map_or(false, |t| t.contains(&edge(act, other))) { continue; }
                    let Some(rel) = self.candidate(f, act, as_node, &opos, other) else { continue };
                    let v = cands.entry(rel).or_default();
                    if !v.contains(fid) { v.push(*fid); }
                    let s = strength.entry(rel).or_insert(0);
                    *s = (*s).max(e.strength);
                }
            }
        }
        let mut joined: FxHashSet<Rel> = FxHashSet::default();
        let mut keys: Vec<Rel> = cands.keys().copied().collect();
        keys.sort();
        for rel in keys {
            let fs = cands[&rel].clone();
            // where it is: in a function, or alone
            let here = match self.dict.holder(&rel).and_then(|g| self.dict.get(g)) {
                Some(g) => self.worth_in(&g.rels, &rel).0,
                None => strength[&rel] as f64 - 1.0,
            };
            let mut best: Option<(f64, f64, u64, FuncId)> = None;
            for fid in fs {
                let Some(f) = self.dict.get(fid) else { continue };
                if f.holds(&rel) { continue; }
                // read against the function as it now stands, since joins and leaves before it may have changed it:
                // it touches a node, and closes no second cycle
                let (ka, kb) = (f.node_of(rel.a).is_some(), f.node_of(rel.b).is_some());
                if !(ka || kb) || (ka && kb && f.has_cycle()) { continue; }
                let mut rels = f.rels.clone();
                rels.push(rel);
                let (w, near) = self.worth_in(&rels, &rel);
                let better = match best { None => true, Some((bw, bn, bj, _)) => w > bw || (w == bw && (near > bn || (near == bn && f.joined < bj))) };
                if better { best = Some((w, near, f.joined, fid)); }
            }
            let Some((w, _, _, fid)) = best else { continue };
            if !(w > 0.0 && w > here) { continue; }
            if self.trace { eprintln!("[join f{}] {} into {} worth {:.0} over {:.0}", fno, describe_rel(&rel), self.dict.describe(self.dict.get(fid).unwrap()), w, here); }
            if let Some(g) = self.dict.holder(&rel) { self.leave(g, &rel, &mut changed); }
            if self.dict.get(fid).is_none() { continue; }
            self.dict.join(fid, rel);
            joined.insert(rel);
            if !changed.contains(&fid) { changed.push(fid); }
        }
        // leaving (R54, R18): one relation at a time, the one that gains most, priced again between
        let mut placed: Vec<FuncId> = fits.iter().map(|f| f.func).chain(changed.iter().copied()).collect();
        placed.sort_unstable();
        placed.dedup();
        for fid in placed {
            loop {
                let Some(f) = self.dict.get(fid) else { break };
                let rels = f.rels.clone();
                let mut worst: Option<(f64, Rel, bool)> = None;
                for r in &rels {
                    let s = self.strength_of(r) as f64;
                    let (inside, _) = self.worth_in(&rels, r);
                    let alone = s - 1.0;
                    if s < 2.0 { let gain = f64::INFINITY; if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, true)); } continue; }
                    if rels.len() > 1 && alone > inside {
                        let gain = alone - inside;
                        if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, false)); }
                    }
                }
                let Some((_, r, nothing)) = worst else { break };
                if self.trace { eprintln!("[leave f{}] {} from {}{}", fno, describe_rel(&r), self.dict.describe(self.dict.get(fid).unwrap()), if nothing { " for nothing" } else { "" }); }
                let one = rels.len() == 1;
                self.leave(fid, &r, &mut changed);
                if !nothing && !one {
                    let g = self.dict.start(r);
                    changed.push(g);
                }
            }
        }
        // a recurring relation no function holds starts one (R50)
        for e in fresh {
            if e.strength < 2 || joined.contains(&e.rel) || self.dict.holder(&e.rel).is_some() { continue; }
            // a relation to a call its own function already holds, read as the call's origin, is held
            if let Some(far) = self.act(e.far) {
                if let (Some(fb), Some(near)) = (&far.fired_by, self.act(e.near)) {
                    if let Some(g) = self.dict.get(fb.func) {
                        let read = canon(near.neuron, g.origin(), e.off);
                        if g.holds(&read) { continue; }
                    }
                }
            }
            if let Some(near) = self.act(e.near) {
                if let (Some(fb), Some(far)) = (&near.fired_by, self.act(e.far)) {
                    if let Some(g) = self.dict.get(fb.func) {
                        let read = canon(g.origin(), far.neuron, e.off);
                        if g.holds(&read) { continue; }
                    }
                }
            }
            let id = self.dict.start(e.rel);
            self.stats_added += 1;
            if self.trace { eprintln!("[new f{}] {}", fno, self.dict.describe(self.dict.get(id).unwrap())); }
            changed.push(id);
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// A relation leaves a function (R18, R54): what the origin no longer reaches leaves with it and starts a
    /// function of its own where it recurs; a function left with none leaves the dictionary.
    fn leave(&mut self, fid: FuncId, r: &Rel, changed: &mut Vec<FuncId>) {
        let (cut, gone) = self.dict.leave(fid, r);
        for c in cut {
            if self.strength_of(&c) >= 2 && self.dict.holder(&c).is_none() { let g = self.dict.start(c); changed.push(g); }
        }
        match gone {
            Some(f) => { self.retire(f); changed.retain(|x| *x != fid); }
            None => { if !changed.contains(&fid) { changed.push(fid); } }
        }
    }

    /// A function has left the dictionary (R18, R38): its interneuron, left with no function, waits on the ledger
    /// to be decommissioned when due, and the function stays readable while what it fired is open.
    pub(crate) fn retire(&mut self, f: Func) {
        self.stats_retired += 1;
        if let Some(n) = f.inter {
            if self.dict.of_inter(n).is_empty() && !self.ledger.contains(&n) { self.ledger.push(n); }
        }
        let until = self.frame_no + self.max_reach_t.max(self.span_s) as i64 + 1;
        self.retired_funcs.insert(f.id, (f, until));
    }

    /// A function as expansion reads it: the dictionary's, or the copy kept while what it fired is open (R38).
    pub(crate) fn func(&self, id: FuncId) -> Option<&Func> {
        self.dict.get(id).or_else(|| self.retired_funcs.get(&id).map(|x| &x.0))
    }
}
