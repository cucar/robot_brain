//! Recognition (§6.1): finding (R53), fitting (D31, D54), what the neurons answer at the count step (D47, R50),
//! and grouping: relations joining, leaving and starting functions by their worth (D57, R50, R54, R18).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::function::*;
use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// One fit of a function on the board (D15, D54): the node it entered at and the activation there, the
/// activations its fixed nodes came true on and the calls it landed on as a node, each with its node, the board
/// relations its true relations stand in for, the relations that did not come true, and the laps it read.
#[derive(Clone, Debug)]
pub(crate) struct Fit {
    pub func: FuncId,
    pub entry: (usize, ActRef),
    pub landed: Vec<(ActRef, usize)>,
    pub calls: Vec<(ActRef, usize)>,
    pub edges: Vec<(ActRef, ActRef)>,
    pub failed: Vec<usize>,
    pub laps: u32,
    /// it read a node through a call of another function, which stays where it stands (switch 11)
    pub through: bool,
    /// the calls it read a node through, its own lapped call among them (switch 11)
    pub thru: Vec<ActRef>,
}

impl Fit {
    /// Every activation it lands on, as a node or as a call.
    pub fn acts(&self) -> impl Iterator<Item = (ActRef, usize)> + '_ { self.landed.iter().chain(self.calls.iter()).copied() }
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

/// Whether two offsets lie along one line (D15: the direction a cycle advances in).
fn parallel(a: &Offset, b: &Offset) -> bool {
    if a.iter().all(|x| *x == 0) || b.iter().all(|x| *x == 0) { return false; }
    (0..MAX_DIMS).all(|d| (0..MAX_DIMS).all(|e| a[d] as i64 * b[e] as i64 == a[e] as i64 * b[d] as i64))
}

impl Machine {
    // ───────────────────────────── fitting (D31, D54) ─────────────────────────────

    /// Whether a fit may land on an activation (D28, D10): it is on the apex, or only calls this election has
    /// taken have landed on it, each covering relations of its own.
    pub(crate) fn landable(&self, w: &Act) -> bool {
        if w.gone || !self.stands(w, self.frame_no) { return false; }
        // with functions that never change, only what stands uncovered on the apex (switch 8)
        
        w.on_apex() || (!w.coverers.is_empty() && w.coverers.iter().all(|c| self.electing.contains(c)))
    }

    /// From a landed activation, the board relation at `delta` whose far end is `want`, or an earlier call of the
    /// function's own whose pose starts at `want` standing there (D54); none a call covers (D28); the older first
    /// (R24). Whether it is a call.
    fn step(&self, func: FuncId, from: ActRef, delta: Offset, want: Sym, used: &FxHashSet<ActRef>) -> Option<(ActRef, bool)> {
        self.step_through(func, from, delta, want, used).map(|(w, lap, _, _)| (w, lap))
    }

    /// A step under switch 11 (D54, D5): the far node lands on its own neuron standing bare at that position, or on
    /// it named there by a call standing on the apex beside the activation, at any depth; a relation already
    /// covered is not read (D28). Returns what the node stands on (the activation, or the call where the call is
    /// one of the fit's own function, which then takes the lap), whether it is that lap, the activation the
    /// relation was read to, and whether it was read through another function's call. Bare before through, the
    /// older first (R24).
    pub(crate) fn step_through(&self, func: FuncId, from: ActRef, delta: Offset, want: Sym, used: &FxHashSet<ActRef>) -> Option<(ActRef, bool, ActRef, Option<ActRef>)> {
        let a = self.act(from)?;
        let fno = self.frame_no;
        let target = add(&a.pos, &delta);
        // bare: what stands there on the apex, as the board relations have it
        let mut others: Vec<ActRef> = a.rels.iter().filter(|br| br.off == delta).map(|br| br.other).collect();
        if let Some(fr) = self.frames.iter().find(|f| f.no == target[0] as i64) {
            let back = neg(&delta);
            for (i, w) in fr.acts.iter().enumerate() {
                if w.pos != target { continue; }
                if w.rels.iter().any(|br| br.other == from && br.off == back) { others.push(ActRef { frame: fr.no, index: i as u32 }); }
            }
        }
        others.sort();
        others.dedup();
        for o in &others {
            if used.contains(o) || self.covered_rel.contains(&edge(from, *o)) { continue; }
            let Some(w) = self.act(*o) else { continue };
            if w.sym != want || !self.landable(w) { continue; }
            return Some((*o, false, *o, None));
        }
        // through a call beside it: every call standing on the apex that the activation sees, read as what its
        // function names, at any depth, each at its own position (D5)
        let mut calls: Vec<ActRef> = a.rels.iter().map(|br| br.other).filter(|r| self.act(*r).map_or(false, |c| !c.covers.is_empty())).collect();
        calls.sort();
        calls.dedup();
        let mut best: Option<(ActRef, bool, ActRef, Option<ActRef>)> = None;
        for c in calls {
            let Some(ca) = self.act(c) else { continue };
            if !ca.on_apex() || !self.stands(ca, fno) { continue; }
            let mut under: Vec<ActRef> = self.under(ca).into_iter().collect();
            under.sort();
            for u in under {
                if u == from || used.contains(&u) || self.covered_rel.contains(&edge(from, u)) { continue; }
                let Some(w) = self.act(u) else { continue };
                if w.pos != target || w.sym != want || w.gone || !self.stands(w, fno) { continue; }
                // read through a call of its own function going round: the call takes the lap (D53)
                let own = ca.fired_by.as_ref().map_or(false, |fb| fb.func == func);
                let cand = if own { (c, true, u, Some(c)) } else { (u, false, u, Some(c)) };
                if best.map_or(true, |b| (cand.2, cand.0) < (b.2, b.0)) { best = Some(cand); }
            }
        }
        best
    }

    /// Fit a function entered at node `root` on an activation (D31, D54): read out from there relation by
    /// relation, round the cycle lap after lap while the relation that closes it lands, and the relations that
    /// hang off the cycle from their node on the last lap.
    pub(crate) fn fit(&self, f: &Func, root: usize, entry: ActRef) -> Fit {
        let lay = f.layout_from(root);
        let n = f.nodes.len();
        let mut at: Vec<Option<ActRef>> = vec![None; n];
        let mut via_call = vec![false; n];
        at[root] = Some(entry);
        let mut used: FxHashSet<ActRef> = FxHashSet::default();
        used.insert(entry);
        // it starts on the node's own neuron in its pose, or on a call whose pose starts at the node, which it lands
        // on whole and reads the function's relations from (D54, D31)
        let on_call = self.act(entry).map_or(false, |a| a.sym != f.nodes[root]);
        let mut landed = if on_call { vec![] } else { vec![(entry, root)] };
        let mut calls = if on_call { vec![(entry, root)] } else { vec![] };
        let mut edges = Vec::new();
        let mut came = vec![false; f.rels.len()];
        // the nodes past a call the fit landed on: the lap stops at the call, and nothing past it fails (D54, D10)
        let mut beyond = vec![false; n];
        let mut inside = vec![false; f.rels.len()];
        let mut laps = 0u32;
        // the cycle: its path once round, from the node the closing relation reaches back to where it starts
        let cycle_path: Vec<(usize, usize, usize, Offset)> = match lay.closing { Some((_, ua, ub, _)) => Func::tree_path(&lay, ub, ua), None => vec![] };
        let mut on_cycle: Vec<bool> = vec![false; n];
        if let Some((_, ua, ub, _)) = lay.closing { on_cycle[ua] = true; on_cycle[ub] = true; for s in &cycle_path { on_cycle[s.1] = true; on_cycle[s.2] = true; } }
        let path_rels: Vec<usize> = cycle_path.iter().map(|s| s.0).collect();
        let through = std::cell::Cell::new(false);
        let thru_calls: std::cell::RefCell<Vec<ActRef>> = std::cell::RefCell::new(Vec::new());
        let s11 = true;
        // the step: what the node stands on, whether a call, and the activation the relation was read to
        let go = |this: &Machine, au: ActRef, off: Offset, want: Sym, used: &FxHashSet<ActRef>| -> Option<(ActRef, bool, ActRef)> {
            if s11 {
                let (w, lap, to, via) = this.step_through(f.id, au, off, want, used)?;
                if via.is_some() && !lap { through.set(true); }
                if let Some(c) = via { thru_calls.borrow_mut().push(c); }
                Some((w, lap, to))
            } else { let (w, c) = this.step(f.id, au, off, want, used)?; Some((w, c, w)) }
        };
        let try_step = |this: &Machine, ri: usize, u: usize, v: usize, off: Offset, at: &mut Vec<Option<ActRef>>, via_call: &mut Vec<bool>, used: &mut FxHashSet<ActRef>, landed: &mut Vec<(ActRef, usize)>, calls: &mut Vec<(ActRef, usize)>, edges: &mut Vec<(ActRef, ActRef)>, came: &mut Vec<bool>, beyond: &mut Vec<bool>, inside: &mut Vec<bool>| -> Option<bool> {
            if beyond[u] || (at[u].is_some() && via_call[u]) { inside[ri] = true; if at[v].is_none() { beyond[v] = true; } return None; }
            let au = at[u]?;
            let (w, call, to) = go(this, au, off, f.nodes[v], used)?;
            came[ri] = true;
            edges.push((au, to));
            used.insert(w);
            used.insert(to);
            at[v] = Some(w);
            via_call[v] = call;
            if call { calls.push((w, v)); } else { landed.push((w, v)); }
            Some(call)
        };
        // the tree, the relations hanging off the cycle left for after it has run
        let mut deferred = Vec::new();
        for &(ri, u, v, off) in &lay.tree {
            if lay.closing.is_some() && on_cycle[u] && !path_rels.contains(&ri) { deferred.push((ri, u, v, off)); continue; }
            if at[u].is_none() && !beyond[u] { deferred.push((ri, u, v, off)); continue; }
            try_step(self, ri, u, v, off, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came, &mut beyond, &mut inside);
        }
        // round the cycle, lap after lap, while the closing relation lands (D54)
        if let Some((ri, ua, ub, off)) = lay.closing {
            loop {
                if laps > 4096 { break; }
                let Some(au) = at[ua] else { break };
                if via_call[ua] { break; }
                let Some((w, call, to)) = go(self, au, off, f.nodes[ub], &used) else { break };
                came[ri] = true;
                edges.push((au, to));
                used.insert(w);
                used.insert(to);
                at[ub] = Some(w);
                if call { calls.push((w, ub)); via_call[ub] = true; break; }
                laps += 1;
                landed.push((w, ub));
                via_call[ub] = false;
                // the lap before, from the node it reached
                let mut whole = true;
                for &(pri, pu, pv, poff) in &cycle_path {
                    if try_step(self, pri, pu, pv, poff, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came, &mut beyond, &mut inside).is_none() { whole = false; break; }
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
                if at[u].is_some() || beyond[u] {
                    try_step(self, ri, u, v, off, &mut at, &mut via_call, &mut used, &mut landed, &mut calls, &mut edges, &mut came, &mut beyond, &mut inside);
                    progress = true;
                } else { rest.push((ri, u, v, off)); }
            }
            deferred = rest;
        }
        // the relation that closes a cycle is never a failure: the fit ends where it stops landing (D54)
        let closing_ri = lay.closing.map(|c| c.0);
        // group B: the fit of a cycle ends at the first of its relations that does not land, none a failure
        let cyc_ri: Vec<usize> = { match lay.closing { Some((ri, _, _, _)) => { let mut v = path_rels.clone(); v.push(ri); v } None => vec![] } };
        let failed: Vec<usize> = (0..f.rels.len()).filter(|ri| !came[*ri] && !inside[*ri] && Some(*ri) != closing_ri && !cyc_ri.contains(ri)).collect();
        let mut thru = thru_calls.into_inner();
        thru.sort();
        thru.dedup();
        Fit { func: f.id, entry: (root, entry), landed, calls, edges, failed, laps, through: through.get(), thru }
    }

    /// Every fit of a function this frame (R53, D54): one per node and activation of the present frame on the
    /// apex that the node admits, its own neuron in its pose or a call whose pose starts at it; the same landing
    /// reached from two entries is one fit.
    pub(crate) fn fits_of(&self, id: FuncId) -> Vec<Fit> {
        let Some(f) = self.dict.get(id) else { return vec![] };
        let fno = self.frame_no;
        let mut out: Vec<Fit> = Vec::new();
        let mut keys: FxHashSet<Vec<ActRef>> = FxHashSet::default();
        let Some(fr) = self.frames.iter().find(|fr| fr.no == fno) else { return out };
        for (root, node) in f.nodes.iter().enumerate() {
            for (i, a) in fr.acts.iter().enumerate() {
                // the node admits its own neuron in its pose, or a call whose pose starts at it (D54)
                let admits = a.sym == *node;
                if !admits || !a.on_apex() || a.gone { continue; }
                let fit = self.fit(f, root, ActRef { frame: fno, index: i as u32 });
                if fit.edges.is_empty() { continue; }
                let mut key: Vec<ActRef> = fit.acts().map(|x| x.0).collect();
                key.sort();
                if !keys.insert(key) { continue; }
                out.push(fit);
            }
        }
        out
    }

    /// Finding (R53): the functions that hold a relation returned at this level, as it is.
    pub(crate) fn find(&self, fresh: &[Fresh]) -> Vec<FuncId> {
        let mut v: Vec<FuncId> = fresh.iter().filter_map(|e| self.dict.holder(&e.rel)).collect();
        // a relation to a call is read as the nodes its function names, each at its own offset (switch 11, R53, D5)
        
        v.sort_unstable();
        v.dedup();
        v
    }

    // ───────────────────────────── what the neurons answer (D47, R50) ─────────────────────────────

    /// Every node a call in its pose expands to, at any depth, each at its offset from the call's position and with
    /// the interneurons of the calls that put it there, the call's own first (R28, R50): its function read from the
    /// node its pose starts at, its cycle round as many laps as the pose says, and each call among its nodes
    /// expanded in turn; none for a base neuron.
    pub(crate) fn expansion_of(&self, s: Sym) -> std::rc::Rc<Vec<(Sym, Offset, Vec<NeuronId>)>> {
        if let Some(v) = self.expansions.borrow().get(&s) { return v.clone(); }
        let mut out = Vec::new();
        let mut deps = Vec::new();
        self.expand_nodes(s, [0; MAX_DIMS], &mut Vec::new(), &mut out, &mut deps);
        out.sort_unstable();
        out.dedup();
        deps.sort_unstable();
        deps.dedup();
        let v = std::rc::Rc::new(out);
        self.expansions.borrow_mut().insert(s, v.clone());
        self.expansion_deps.borrow_mut().insert(s, deps);
        v
    }

    /// Drop every expansion kept that was read through a function, which has changed, and with them the
    /// strengths worked out this grouping.
    pub(crate) fn forget_expansions(&self, fid: FuncId) {
        self.held_memo.borrow_mut().clear();
        self.strength_memo.borrow_mut().clear();
        self.returned_memo.borrow_mut().clear();
        let gone: Vec<Sym> = self.expansion_deps.borrow().iter().filter(|(_, d)| d.contains(&fid)).map(|(s, _)| *s).collect();
        let (mut e, mut d) = (self.expansions.borrow_mut(), self.expansion_deps.borrow_mut());
        for s in gone { e.remove(&s); d.remove(&s); }
    }

    /// Drop every expansion kept.
    pub(crate) fn forget_all_expansions(&self) {
        self.held_memo.borrow_mut().clear();
        self.strength_memo.borrow_mut().clear();
        self.returned_memo.borrow_mut().clear();
        self.expansions.borrow_mut().clear();
        self.expansion_deps.borrow_mut().clear();
    }

    fn expand_nodes(&self, s: Sym, at: Offset, path: &mut Vec<NeuronId>, out: &mut Vec<(Sym, Offset, Vec<NeuronId>)>, deps: &mut Vec<FuncId>) {
        let Some(pose) = self.dict.pose_of(s) else { return };
        let n = self.dict.neuron_of(s);
        if path.contains(&n) || path.len() > 64 { return; }
        let Some(f) = self.func_for(s).cloned() else { return };
        deps.push(f.id);
        let root = self.dict.node_by_neuron(&f, pose.from).unwrap_or(0);
        let lay = f.layout_from(root);
        let mut nodes: Vec<(Sym, Offset)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, lay.position[i])).collect();
        if let Some(l) = f.lap() {
            // the cycle read back from the node the pose starts at, lap after lap, to the node it ends at (D53)
            let order = f.back_order(root);
            if !order.is_empty() {
                let to = self.dict.node_by_neuron(&f, pose.to).unwrap_or(order[order.len() - 1]);
                let total = f.stretch(root, to, pose.laps);
                let c = order.len();
                nodes.retain(|(m, _)| !order.iter().any(|v| f.nodes[*v] == *m));
                for i in 0..total {
                    let (v, k) = (order[i % c], (i / c) as i32);
                    let mut q = lay.position[v];
                    for d in 0..MAX_DIMS { q[d] += k * l[d]; }
                    nodes.push((f.nodes[v], q));
                }
            }
        }
        path.push(n);
        for (m, q) in nodes {
            let mut p = at;
            for d in 0..MAX_DIMS { p[d] += q[d]; }
            out.push((m, p, path.clone()));
            self.expand_nodes(m, p, path, out, deps);
        }
        path.pop();
    }

    /// Whether a row holds a relation to a node the way a call of the function it is read for could cover it there
    /// (R50): the node's neuron in its pose bare at that offset, or a call whose expansion puts it there with a call
    /// of `own`, the function's interneuron, anywhere in the chain. A row that holds it only through another
    /// function's call holds nothing for it; with no `own`, only bare.
    fn holds_for(&self, row: &Row, far: Sym, off: &Offset, own: Option<NeuronId>) -> bool {
        let _ = own;
        // a count never expands a call (D47): the neighbor itself, at that offset
        row.nbs.iter().any(|nb| nb.sym == far && nb.off == *off)
    }

    /// Whether a row holds a relation bare: the node's neuron in its pose itself at that offset.
    fn holds_bare(&self, row: &Row, far: Sym, off: &Offset) -> bool { row.nbs.iter().any(|nb| nb.sym == far && nb.off == *off) }

    /// Whether a row holds a relation for the count where it stands (R50): the way the function's own call could cover
    /// it (`holds_for`), or as an edge a call standing for both its ends saved, whichever function that call belongs
    /// to now. Used for `with`, `without` and worth; starting and a candidate's strength read `holds_for` alone.
    fn holds_in(&self, row: &Row, far: Sym, off: &Offset, own: Option<NeuronId>) -> bool {
        self.holds_for(row, far, off, own)
    }

    /// The rows of a symbol's neuron that it fired in that pose (D18).
    fn rows_of(&self, s: Sym) -> impl Iterator<Item = &Row> + '_ {
        let n = self.dict.neuron_of(s);
        self.thalamus.get(n).into_iter().flat_map(move |nn| nn.history.iter().filter(move |r| r.own == s))
    }

    /// Whether a row has held or failed a relation at offset `off` from its activation yet (R50, D18): one backward
    /// in time is on the row when it is written; one forward in time only once the row has taken what followed, at
    /// the connect step of the frame it reaches, so from the frame after that on.
    fn decided(&self, row: &Row, off: &Offset) -> bool { off[0] <= 0 || row.act.frame + (off[0] as i64) < self.frame_no }

    /// The neuron in its pose a relation is read from where it has no node to be counted at (D47, R50): an end whose
    /// rows hold it for `own`; where both do, the one with more activations behind it. None where no rows hold it.
    fn held_by(&self, r: &Rel, own: Option<NeuronId>) -> Option<(Sym, Sym, Offset)> {
        if let Some(h) = self.held_memo.borrow().get(&(*r, own)) { return *h; }
        let h = r.ends().into_iter()
            .filter(|c| self.rows_of(c.0).any(|row| self.decided(row, &c.2) && self.holds_in(row, c.1, &c.2, own)))
            .map(|c| (c, self.rows_count(c.0)))
            .max_by_key(|x| x.1).map(|x| x.0);
        self.held_memo.borrow_mut().insert((*r, own), h);
        h
    }

    /// A relation's strength as the neurons return it (D47, R21): at each end the rows that hold it for `own`, bare or
    /// under a call of that function, over every row; the larger of the two ends.
    pub(crate) fn returned_for(&self, r: &Rel, own: Option<NeuronId>) -> u32 {
        if let Some(s) = self.returned_memo.borrow().get(&(*r, own)) { return *s; }
        let s = r.ends().into_iter().map(|(n, far, off)| self.rows_of(n).filter(|row| self.holds_for(row, far, &off, own)).count() as u32).max().unwrap_or(0);
        self.returned_memo.borrow_mut().insert((*r, own), s);
        s
    }

    /// A returned relation's strength for starting a function (R50): by switch, rows left bare by their election (6),
    /// bare rows (4), rows through calls (1), or the strength the neuron returned (committed).
    fn start_strength(&self, r: &Rel, _returned: u32) -> u32 {
        self.returned_for(r, None)
    }

    /// How many rows a symbol's neuron holds in that pose: the activations behind that end.
    fn rows_count(&self, s: Sym) -> usize { self.rows_of(s).count() }

    /// `with` and `without` for one relation of a set read for the function whose interneuron is `own` (R50), over
    /// the rows of the node it is counted at: `with`, every row that holds it the way that function's call could
    /// cover it there, wherever it holds; `without`, every row where the set's other relations at that node held
    /// that way and something else stands where it expected its neuron, its neuron under another function's call
    /// included. Where nothing stands there, the row counts neither way; where the node holds none of the set's other
    /// relations, nothing is `without`.
    pub(crate) fn answer(&self, rels: &[Rel], r: &Rel, at: Option<Sym>, own: Option<NeuronId>) -> (u32, u32) {
        // at the node it attached at (R50), where it has one; else at an end whose rows hold it
        let end = at.and_then(|n| r.ends().into_iter().find(|c| c.0 == n)).or_else(|| self.held_by(r, own));
        let Some((n, far, off)) = end else { return (0, 0) };
        let mut cond: Vec<(Sym, Offset)> = Vec::new();
        for s in rels {
            if s == r { continue; }
            for (m, f2, o2) in s.ends() { if m == n { cond.push((f2, o2)); } }
        }
        let mut key_cond = cond.clone();
        key_cond.sort();
        let key = (n, far, off, key_cond);
        if let Some(m) = self.answer_memo.borrow().as_ref() { if let Some(v) = m.get(&key) { return *v; } }
        let (mut with, mut without) = (0, 0);
        {
            // the committed reading: a row counts where the set's other relations at that node hold
            for row in self.rows_of(n) {
                if !self.decided(row, &off) || !cond.iter().all(|(_, o2)| self.decided(row, o2)) { continue; }
                if !cond.iter().all(|(f2, o2)| self.holds_in(row, *f2, o2, own)) { continue; }
                if self.holds_in(row, far, &off, own) { with += 1; } else if !cond.is_empty() { without += 1; }
            }
        }
        if let Some(m) = self.answer_memo.borrow_mut().as_mut() { m.insert(key, (with, without)); }
        (with, without)
    }

    /// A relation's worth in a set read for `own` (D57): `with − without − 1`; with its nearness,
    /// `with / (with + without)`, and its `with`.
    pub(crate) fn worth_in(&self, rels: &[Rel], r: &Rel, at: Option<Sym>, own: Option<NeuronId>, closing: Option<Rel>) -> (f64, f64, u32) {
        let (w, wo) = self.answer(rels, r, at, own);
        let near = if w + wo == 0 { 0.0 } else { w as f64 / (w + wo) as f64 };
        // the committed build never counted against the relation that closes a cycle
        let x = if closing == Some(*r) { w as f64 - 1.0 } else { w as f64 - wo as f64 - 1.0 };
        (x, near, w)
    }

    /// The relation that closes the cycle of a set of relations listed from `origin` (D15), where it has one.
    fn closing_of(origin: Sym, rels: &[Rel]) -> Option<Rel> {
        let f = Func::from_rels(origin, rels);
        f.layout().closing.map(|c| rels[c.0])
    }

    /// A relation's contribution to the function it is in (D57, R54): the function's worth with it less without
    /// it, read for the function's calls; in a function of one relation, its worth there.
    /// A relation's contribution to its function under groups A and B: the function's worth with it less without
    /// it, the entry the same on both sides; as its function's only relation, that function's worth, its strength
    /// less two.
    pub(crate) fn contribution_ab(&self, f: &Func, r: &Rel) -> f64 {
        let (without, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
        if without.is_empty() { return self.halved(r) - 2.0; }
        let origin = if without.iter().any(|x| x.a == f.origin() || x.b == f.origin()) { f.origin() } else { without[0].a };
        self.worth_ab(f.origin(), &f.rels, &f.at, f.inter) - self.worth_ab(origin, &without, &wat, f.inter)
    }

    // ───────────────────────────── grouping (R50, R54, R18) ─────────────────────────────

    /// The relations a board relation would be to a function at an activation it shares with the function's fit or
    /// call (R50): from the node there to the other end; a call read as each node of the function its expansion puts
    /// where the shared activation sees, at that node's offset (R50, R28), and as itself where it puts none.
    fn candidates(&self, f: &Func, shared: ActRef, as_node: Sym, origin_pos: &Pos, other: ActRef) -> Vec<Rel> {
        let (Some(sa), Some(oa)) = (self.act(shared), self.act(other)) else { return vec![] };
        let off = offset_between(sa, oa);
        let mut reads: Vec<(Sym, Offset)> = Vec::new();
        reads.push((oa.sym, off));
        reads.into_iter().filter_map(|(read, q)| self.candidate(f, as_node, origin_pos, oa.pos, read, q)).collect()
    }

    /// One relation from node `as_node` to `read` at `off` as a candidate of a function (R50), the far end standing
    /// at `far_pos`. None where it is no candidate.
    fn candidate(&self, f: &Func, as_node: Sym, origin_pos: &Pos, far_pos: Pos, read: Sym, off: Offset) -> Option<Rel> {
        let rel = canon(as_node, read, off);
        if f.holds(&rel) { return None; }
        if f.node_of(read).is_some() {
            // it would close the cycle, where the function has none
            if f.has_cycle() { return None; }
        } else {
            // a new node may not be the origin: later in time, or at a lesser position in the origin's frame
            if later(&far_pos, origin_pos) { return None; }
            // nothing leads on from a node of the cycle in the direction it advances in (D15)
            if let Some(lap) = f.lap() {
                let node = f.node_of(as_node)?;
                if f.cycle_nodes().contains(&node) && parallel(&off, &lap) { return None; }
            }
        }
        Some(rel)
    }

    /// Grouping at one level (R50, R54, R18): each candidate goes where its gain is highest, each relation of a
    /// function that fit here leaves where it is worth more alone or nothing anywhere, and each recurring relation
    /// no function holds starts one. Returns the functions that changed.
    /// The relations standing on the board (R50, D9): every relation the neurons returned, this frame or at one
    /// before, whose two activations still stand on the apex, each edge once, at the strength its stronger end
    /// returned.
    pub(crate) fn board_rels(&self) -> Vec<Fresh> {
        let fno = self.frame_no;
        let mut out: FxHashMap<(ActRef, ActRef), Fresh> = FxHashMap::default();
        for fr in self.frames.iter() {
            for (i, a) in fr.acts.iter().enumerate() {
                let s15 = self.sw.s15_uncovered_relations;
                if (!s15 && !a.on_apex()) || a.gone || !self.stands(a, fno) { continue; }
                let near = ActRef { frame: fr.no, index: i as u32 };
                for br in &a.rels {
                    let Some(b) = self.act(br.other) else { continue };
                    if (!s15 && !b.on_apex()) || b.gone || !self.stands(b, fno) { continue; }
                    // a relation a call covers is off the board (switch 15)
                    if s15 && self.covered_rel.contains(&edge(near, br.other)) { continue; }
                    if !br.strength < 2 { continue; }
                    let k = edge(near, br.other);
                    if out.get(&k).map_or(false, |x| x.strength >= br.strength) { continue; }
                    out.insert(k, Fresh { near, far: br.other, off: br.off, strength: br.strength, rel: canon(a.sym, b.sym, br.off) });
                }
            }
        }
        let mut v: Vec<Fresh> = out.into_values().collect();
        v.sort_by(|a, b| (a.near, a.far).cmp(&(b.near, b.far)));
        v
    }

    pub(crate) fn group(&mut self, fits: &[Fit], fresh: &[Fresh], board: &[Fresh], fired: &FxHashSet<Sym>) -> Vec<FuncId> {
        // no row changes while grouping, so each relation's count is read once
        *self.halved_memo.borrow_mut() = Some(FxHashMap::default());
        *self.answer_memo.borrow_mut() = Some(FxHashMap::default());
        *self.share_memo.borrow_mut() = Some(FxHashMap::default());
        let out = self.group_inner(fits, fresh, board, fired);
        *self.halved_memo.borrow_mut() = None;
        *self.answer_memo.borrow_mut() = None;
        *self.share_memo.borrow_mut() = None;
        out
    }

    fn group_inner(&mut self, fits: &[Fit], fresh: &[Fresh], board: &[Fresh], fired: &FxHashSet<Sym>) -> Vec<FuncId> {
        
        self.forget_all_expansions();
        let mut changed: Vec<FuncId> = Vec::new();
        let fno = self.frame_no;
        // the activations each function touches here: what its fits landed on, each as its node, with the position
        // its origin stands at; and its calls on the board, as the node their pose starts at
        let mut touch: FxHashMap<FuncId, Vec<(ActRef, Sym, Pos)>> = FxHashMap::default();
        // the board edges each function's fits already land on: one edge is never two of its relations
        let mut taken: FxHashMap<FuncId, FxHashSet<(ActRef, ActRef)>> = FxHashMap::default();
        // switch 17: the fits this frame's election took touch the board as the fits of this grouping do
        let mut all_fits: Vec<Fit> = fits.to_vec();
        {
            let mut mine: Vec<(ActRef, Fit)> = self.call_fit.iter().map(|(r, f)| (*r, f.clone())).collect();
            mine.sort_by_key(|x| x.0);
            all_fits.extend(mine.into_iter().map(|x| x.1));
        }
        // for the count only: which touches came from the first election's fits (switch 17) and which from this
        // grouping's own fits or the board
        let mut touch_first: FxHashSet<(FuncId, ActRef)> = FxHashSet::default();
        let mut touch_own: FxHashSet<(FuncId, ActRef)> = FxHashSet::default();
        for (fi, fit) in all_fits.iter().enumerate() {
            let Some(f) = self.dict.get(fit.func) else { continue };
            for (l, _) in fit.acts() { if fi >= fits.len() { touch_first.insert((f.id, l)); } else { touch_own.insert((f.id, l)); } }
            taken.entry(f.id).or_default().extend(fit.edges.iter().map(|&(a, b)| edge(a, b)));
            let lay = f.layout();
            let Some(entry) = self.act(fit.entry.1) else { continue };
            let mut opos = entry.pos;
            for d in 0..MAX_DIMS { opos[d] -= lay.position[fit.entry.0][d]; }
            for (l, node) in fit.acts() { touch.entry(f.id).or_default().push((l, f.nodes[node], opos)); }
        }
        for e in board {
            for end in [e.near, e.far] {
                let Some(a) = self.act(end) else { continue };
                if a.kind != Kind::Inter || !a.on_apex() || !self.stands(a, fno) { continue; }
                let Some(fb) = &a.fired_by else { continue };
                let Some(f) = self.dict.get(fb.func) else { continue };
                let Some(n) = self.dict.start_of(a.sym) else { continue };
                let Some(node) = self.dict.node_by_neuron(f, n) else { continue };
                let start = f.nodes[node];
                let lay = f.layout();
                let mut opos = a.pos;
                for d in 0..MAX_DIMS { opos[d] -= lay.position[node][d]; }
                touch.entry(f.id).or_default().push((end, start, opos));
                touch_own.insert((f.id, end));
            }
        }
        let mut cand_first: FxHashSet<(Rel, FuncId)> = FxHashSet::default();
        let mut cand_own: FxHashSet<(Rel, FuncId)> = FxHashSet::default();
        // the candidates (R50)
        let mut cands: FxHashMap<Rel, Vec<(FuncId, Sym, Option<(Sym, Sym)>)>> = FxHashMap::default();
        let mut at_act: FxHashMap<ActRef, Vec<(FuncId, Sym, Pos)>> = FxHashMap::default();
        let mut fids: Vec<FuncId> = touch.keys().copied().collect();
        fids.sort_unstable();
        for fid in fids { for &(act, as_node, opos) in &touch[&fid] { at_act.entry(act).or_default().push((fid, as_node, opos)); } }
        let through = true;
        // switch 17: besides what the first election left on the board, each fit is offered the relations covered by
        // the calls it names or reads a node through
        let offered: Vec<(Fresh, Option<FuncId>)> = board.iter().map(|e| (*e, None)).collect();
        // switch 19: a relation under a call the fit touches, at the one activation it shares with the fit: the node
        // the fit read through the call, or the activation a call the fit landed on as a node stands at, with the
        // function read with that call's node as that activation
        let mut offered19: Vec<(Fresh, FuncId, ActRef, Sym, Option<(Sym, Sym)>, Pos, bool)> = Vec::new();
        {
            let mut seen: FxHashSet<(FuncId, (ActRef, ActRef), ActRef)> = FxHashSet::default();
            for (fi, fit) in all_fits.iter().enumerate() {
                let first = fi >= fits.len();
                let Some(f) = self.dict.get(fit.func) else { continue };
                let lay = f.layout();
                let Some(entry) = self.act(fit.entry.1) else { continue };
                let mut opos = entry.pos;
                for d in 0..MAX_DIMS { opos[d] -= lay.position[fit.entry.0][d]; }
                let mut offers: Vec<(ActRef, ActRef, Sym, Option<(Sym, Sym)>)> = Vec::new();
                // read through the call: the relations under it that share the node's activation
                for &c in &fit.thru {
                    let Some(ca) = self.act(c) else { continue };
                    let under = self.under(ca);
                    for &(l, node) in &fit.landed { if under.contains(&l) { offers.push((c, l, f.nodes[node], None)); } }
                }
                // landed on the call as a node: the relations under it that share the activation it stands at
                for &(c, node) in &fit.landed {
                    if f.nodes[node] < SYM_CALL { continue; }
                    let Some(ca) = self.act(c) else { continue };
                    if ca.covers.is_empty() { continue; }
                    let Some(p) = self.dict.pose_of(ca.sym) else { continue };
                    let at: Vec<ActRef> = ca.covers.iter().copied().filter(|r| self.act(*r).map_or(false, |x| x.neuron == p.from && x.pos == ca.pos)).collect();
                    if at.len() != 1 { if self.trace { eprintln!("[offer choice f{}] {} as a node of f{}: {} activations it stands at", fno, self.dict.sym_text(ca.sym), f.id, at.len()); } continue; }
                    let Some(sa) = self.act(at[0]) else { continue };
                    let cs = f.nodes[node];
                    if sa.sym != cs && f.node_of(sa.sym).is_some() { if self.trace { eprintln!("[offer choice f{}] {} as a node of f{}: the function already has a node {}", fno, self.dict.sym_text(cs), f.id, self.dict.sym_text(sa.sym)); } continue; }
                    offers.push((c, at[0], sa.sym, if sa.sym == cs { None } else { Some((cs, sa.sym)) }));
                }
                for (c, shared, as_node, rw) in offers {
                    let Some(es) = self.call_edges.get(&c) else { continue };
                    for &(p, q) in es {
                        if p != shared && q != shared { continue; }
                        if !seen.insert((fit.func, (p, q), shared)) { continue; }
                        let (Some(pa), Some(qa)) = (self.act(p), self.act(q)) else { continue };
                        let off = offset_between(pa, qa);
                        offered19.push((Fresh { near: p, far: q, off, strength: 2, rel: canon(pa.sym, qa.sym, off) }, fit.func, shared, as_node, rw, opos, first));
                    }
                }
            }
        }
        
        for (e, only) in offered.iter() {
            let e = *e;
            if !through && e.strength < 2 { continue; }
            // no function is built on a relation between two activations under one same activation (switch 11, R50);
            // switch 23: only where one stands over the other
            if self.share_under(e.near, e.far) { continue; }
            for (act, other) in [(e.near, e.far), (e.far, e.near)] {
                let Some(list) = at_act.get(&act) else { continue };
                for &(fid, as_node, opos) in list {
                    if only.map_or(false, |o| o != fid) { continue; }
                    let Some(f) = self.dict.get(fid) else { continue };
                    if taken.get(&fid).map_or(false, |t| t.contains(&edge(act, other))) { continue; }
                    for rel in self.candidates(f, act, as_node, &opos, other) {
                        // a candidate has strength 2 or more, read through calls (R50, D47)
                        if self.halved(&rel) < 2.0 { continue; }
                        // where both its ends are nodes of the function, it attaches at the end whose activation
                        // returned it, never at a node read through a call (R50)
                        let attach = if f.node_of(rel.a).is_some() && f.node_of(rel.b).is_some() {
                            [e.near, e.far].iter().filter_map(|r| self.act(*r)).map(|a| a.sym).find(|s| *s == rel.a || *s == rel.b).unwrap_or(as_node)
                        } else { as_node };
                        if touch_first.contains(&(fid, act)) && !touch_own.contains(&(fid, act)) { cand_first.insert((rel, fid)); } else { cand_own.insert((rel, fid)); }
                        let v = cands.entry(rel).or_default();
                        if !v.iter().any(|x| x.0 == fid) { v.push((fid, attach, None)); }
                    }
                }
            }
        }
        for &(e, fid, shared, as_node, rw, opos, first) in &offered19 {
            if self.share_under(e.near, e.far) { continue; }
            let other = if e.near == shared { e.far } else { e.near };
            let Some(f0) = self.dict.get(fid) else { continue };
            if taken.get(&fid).map_or(false, |t| t.contains(&edge(shared, other))) { continue; }
            let f = match rw { Some((c, n)) => Self::rewritten(f0, c, n), None => f0.clone() };
            for rel in self.candidates(&f, shared, as_node, &opos, other) {
                if self.halved(&rel) < 2.0 { continue; }
                let attach = if f.node_of(rel.a).is_some() && f.node_of(rel.b).is_some() {
                    [e.near, e.far].iter().filter_map(|r| self.act(*r)).map(|a| a.sym).find(|s| *s == rel.a || *s == rel.b).unwrap_or(as_node)
                } else { as_node };
                if self.trace { eprintln!("[offered f{}] {} to f{} at {}{}", fno, self.rel_text(&rel), fid, self.dict.sym_text(attach), rw.map_or(String::new(), |(c, n)| format!(" | the function read with {} as {}", self.dict.sym_text(c), self.dict.sym_text(n)))); }
                if first { cand_first.insert((rel, fid)); } else { cand_own.insert((rel, fid)); }
                let v = cands.entry(rel).or_default();
                if !v.iter().any(|x| x.0 == fid) { v.push((fid, attach, rw)); }
            }
        }
        // one that closes a cycle before one that adds a node (R50)
        let closes = |m: &Machine, rel: &Rel, fs: &[(FuncId, Sym, Option<(Sym, Sym)>)]| fs.iter().filter_map(|x| m.dict.get(x.0).map(|f| match x.2 { Some((c, n)) => Self::rewritten(f, c, n), None => f.clone() })).any(|f| f.node_of(rel.a).is_some() && f.node_of(rel.b).is_some());
        let mut keys: Vec<(bool, Rel)> = cands.iter().map(|(r, fs)| (!closes(self, r, fs), *r)).collect();
        keys.sort();
        let mut joined: FxHashSet<Rel> = FxHashSet::default();
        // switch 19: the calls each function has held from then on as the activations they stand at, this grouping
        let mut renamed: FxHashMap<FuncId, Vec<(Sym, Sym)>> = FxHashMap::default();
        for (_, rel) in keys {
            let fs = cands[&rel].clone();
            // its contribution where it stands: to its function, or alone, its strength less one; standing alone or
            // in a function of one relation, a tie joins, since it leaves fewer functions (R54)
            let holder = self.dict.holder(&rel).and_then(|g| self.dict.get(g));
            let lone = holder.map_or(true, |g| g.rels.len() == 1);
            let mut best: Option<(f64, f64, u64, FuncId, Sym, Option<(Sym, Sym)>, Rel)> = None;
            for (fid, attach, rw) in fs {
                let Some(f0) = self.dict.get(fid) else { continue };
                // read again against the function as it now stands (switch 19): a call it no longer has as a node is
                // the activation that call stands at; a rewrite already made is not made again, and none that would
                // give the function a second node of one neuron
                let (rel, attach, rw) = {
                    let mut r = rel;
                    let mut at = attach;
                    if let Some(list) = renamed.get(&fid) {
                        for &(c, n) in list {
                            if r.a == c || r.b == c { r = canon(if r.a == c { n } else { r.a }, if r.b == c { n } else { r.b }, r.off); }
                            if at == c { at = n; }
                        }
                    }
                    let rw = match rw {
                        Some((c, _)) if f0.node_of(c).is_none() => None,
                        Some((c, n)) if c != n && f0.node_of(n).is_some() => { self.reread_skips += 1; if self.trace { eprintln!("[reread f{}] {} to f{}: holding {} as {} would give it a second node {}", fno, self.rel_text(&r), fid, self.dict.sym_text(c), self.dict.sym_text(n), self.dict.sym_text(n)); } continue; }
                        x => x,
                    };
                    if r != rel {
                        if let Some(h) = self.dict.holder(&r) { if h != fid { self.reread_skips += 1; if self.trace { eprintln!("[reread f{}] {} read as {} for f{}: f{} holds that", fno, self.rel_text(&rel), self.rel_text(&r), fid, h); } continue; } }
                    }
                    (r, at, rw)
                };
                let fown = match rw { Some((c, n)) => Self::rewritten(f0, c, n), None => f0.clone() };
                let f = &fown;
                if f.holds(&rel) { continue; }
                // read against the function as it now stands: it touches a node, and closes no second cycle
                let (ka, kb) = (f.node_of(rel.a).is_some(), f.node_of(rel.b).is_some());
                if !(ka || kb) || (ka && kb && f.has_cycle()) { continue; }
                let w = {
                    let mut with = f.rels.clone();
                    with.push(rel);
                    let before = Func::meeting(&f.rels, &f.at, attach);
                    let mut at = before.clone();
                    at.push(Some(attach));
                    // switch 20: a function of one relation read at the node the candidate attaches at, over the
                    // same rows, where no cycle is involved
                    let lone_before = if f.rels.len() == 1 && Self::cycle_rels_any(f.origin(), &with).is_empty() {
                        self.alone_over(&f.rels[0], attach, &with, f.inter).map(|x| x - 1.0)
                    } else { None };
                    let wb = lone_before.unwrap_or_else(|| self.worth_ab(f.origin(), &f.rels, &before, f.inter));
                    if self.trace && lone_before.is_some() { eprintln!("   [at node f{}] {} alone in f{} at {}: {:.0} (its count {:.1})", fno, self.rel_text(&f.rels[0]), fid, self.dict.sym_text(attach), wb, self.halved(&f.rels[0])); }
                    self.worth_ab(f.origin(), &with, &at, f.inter) - wb
                };
                let mut rels = f.rels.clone();
                rels.push(rel);
                let near = self.worth_in(&rels, &rel, Some(attach), f.inter, None).1;
                if self.trace && rw.is_some() {
                    let mut with = f.rels.clone();
                    with.push(rel);
                    let before = Func::meeting(&f.rels, &f.at, attach);
                    let mut at = before.clone();
                    at.push(Some(attach));
                    let parts: Vec<String> = with.iter().zip(&at).map(|(r, a)| { let (x, y) = self.answer(&with, r, *a, f.inter); format!("{} at {}: with {} without {} count {:.1}", self.rel_text(r), a.map_or("-".into(), |z| self.dict.sym_text(z)), x, y, self.halved(r)) }).collect();
                    eprintln!("   [read as f{}] f{} {} | gain {:.0} | {}", fno, fid, self.dict.describe(f), w, parts.join("; "));
                }
                let better = match best { None => true, Some((bw, bn, bj, _, _, _, _)) => w > bw || (w == bw && (near > bn || (near == bn && f.joined < bj))) };
                if better { best = Some((w, near, f.joined, fid, attach, rw, rel)); }
            }
            let Some((w, _, _, fid, attach, rw, rel_t)) = best else { continue };
            // switch 21: a relation not at an end of the function it is in does not leave it, to join or otherwise
            {
                if let Some(g) = holder {
                    if !Self::at_an_end(g, &rel) {
                        let gid = g.id;
                        let here = self.contribution_split(g, &rel);
                        if w > here { let why = format!("would join f{}: gain {:.0} over its contribution {:.0}", fid, w, here); self.note_middle(gid, &rel, 2, why); }
                        continue;
                    }
                }
            }
            // alone, read at the node it would attach at (R50)
            // switch 20: alone, read at the node it would attach at, over the rows of the function it is compared with
            let here20 = if lone {
                self.dict.get(fid).and_then(|f0| {
                    let f = match rw { Some((c, n)) => Self::rewritten(f0, c, n), None => f0.clone() };
                    let mut with = f.rels.clone();
                    with.push(rel_t);
                    if !Self::cycle_rels_any(f.origin(), &with).is_empty() { return None; }
                    let own = holder.and_then(|g| g.inter);
                    self.alone_over(&rel_t, attach, &with, own).map(|x| x - 2.0)
                })
            } else { None };
            if self.trace { if let Some(h) = here20 { eprintln!("   [at node f{}] {} alone at {}: {:.0} (its count {:.1})", fno, self.rel_text(&rel), self.dict.sym_text(attach), h, self.halved(&rel)); } }
            let here = match holder {
                _ if here20.is_some() => here20.unwrap(),
                _ if lone => self.halved(&rel) - 2.0,
                Some(g) => self.contribution_ab(g, &rel),
                None => self.halved(&rel) - 2.0,
            };
            if self.trace { eprintln!("[cand f{}] {} best f{} gain {:.0} | where it stands f{:?} {:.0}", fno, self.rel_text(&rel), fid, w, self.dict.holder(&rel), here); }
            if !(w > here) { continue; }
            if self.trace { eprintln!("[join f{}] {} into {} gain {:.0} over {:.0}", fno, self.rel_text(&rel), self.dict.describe(self.dict.get(fid).unwrap()), w, here); }
            let closes_now = self.dict.get(fid).map_or(false, |f| !f.has_cycle() && f.node_of(rel_t.a).is_some() && f.node_of(rel_t.b).is_some());
            if self.trace_joins { eprintln!("[join-count f{}] {}{}", fno, if w == here { "tie" } else { "gain" }, if closes_now { " closing a cycle" } else { "" }); }
            // switch 19: the function's relation to the call is held from then on as the relation to the activation
            // the call stands at; where another function already holds that relation, nothing moves
            let rewrite = match rw {
                Some((c, n)) => {
                    let g = Self::rewritten(self.dict.get(fid).unwrap(), c, n);
                    let clash: Vec<Rel> = g.rels.iter().copied().filter(|r| self.dict.holder(r).map_or(false, |h| h != fid)).collect();
                    if !clash.is_empty() { self.reread_skips += 1; if self.trace { eprintln!("[join choice f{}] {} | read as {}, another function holds {}", fno, self.rel_text(&rel), self.dict.describe(&g), clash.iter().map(|r| self.rel_text(r)).collect::<Vec<_>>().join(", ")); } continue; }
                    Some((c, n, g))
                }
                None => None,
            };
            self.joined_ever.insert(rel);
            let from = self.dict.holder(&rel);
            if let Some(g) = from { self.leave(g, &rel, &mut changed, "its last relation joined another function"); }
            if self.dict.get(fid).is_none() { continue; }
            if let Some((c, n, g)) = rewrite {
                if self.trace { eprintln!("[held as f{}] f{} now {}", fno, fid, self.dict.describe(&g)); }
                self.dict.replace_body(fid, &g);
                renamed.entry(fid).or_default().push((c, n));
            }
            self.dict.join(fid, rel_t, attach);
            // for the count only: a relation that moved from one function into another, and whether only a fit of
            // the first election offered it (switch 17)
            if from.is_some() {
                self.moves_total += 1;
                if cand_first.contains(&(rel, fid)) && !cand_own.contains(&(rel, fid)) { self.moves_first += 1; }
                let k = self.move_count.entry(rel).or_insert(0);
                *k += 1;
                if *k > 1 { self.moves_again += 1; }
            }
            self.note_change(fid);
            self.forget_expansions(fid);
            joined.insert(rel);
            joined.insert(rel_t);
            if !changed.contains(&fid) { changed.push(fid); }
        }
        // leaving (R54, R18): one relation at a time, the one whose leaving raises the dictionary's worth most, the
        // function priced again between; over every function that fit here, changed, or that a symbol firing at
        // this level stands in (R18, R50)
        let standing: Vec<FuncId> = self.dict.funcs.values().filter(|f| f.nodes.iter().any(|n| fired.contains(n))).map(|f| f.id).collect();
        let mut moved: Vec<FuncId> = fits.iter().map(|f| f.func).chain(changed.iter().copied()).chain(standing).collect();
        moved.sort_unstable();
        moved.dedup();
        for fid in moved {
            loop {
                let Some(f) = self.dict.get(fid) else { break };
                let rels = f.rels.clone();
                let mut worst: Option<(f64, Rel, bool)> = None;
                let middle21 = true;
                let mut notes: Vec<(Rel, u8, String)> = Vec::new();
                for r in &rels {
                    // switch 21: a relation not at an end is not priced for leaving; what it would have done is counted
                    if middle21 && !Self::at_an_end(f, r) {
                        let h = self.halved(r);
                        if h < 2.0 { notes.push((*r, 0, format!("strength below the threshold ({:.1})", h))); }
                        else {
                            let at_r = f.rels.iter().zip(&f.at).find(|(x, _)| *x == r).and_then(|(_, a)| *a);
                            let alone = at_r.filter(|_| Self::cycle_rels_any(f.origin(), &f.rels).is_empty()).and_then(|a| self.alone_over(r, a, &f.rels, None)).map(|x| x - 2.0).unwrap_or(h - 2.0);
                            let inside = self.contribution_split(f, r);
                            if alone > inside { notes.push((*r, 1, format!("worth alone {:.0} above its contribution {:.0}", alone, inside))); }
                        }
                        continue;
                    }
                    // for nothing once the neurons return it below 2 (R18)
                    // the neurons return it with strength below 2 over every row that holds it (R18, D47)
                    // the neurons return it below 2, bare or under this function's calls, over every row (R18, R50)
                    let nothing = { let h = self.halved(r); if self.trace && (h < 2.0 || (rels.len() == 1 && h < 3.0)) { eprintln!("[leaves on f{}] {} count {:.1}", fno, self.rel_text(r), h); } h < 2.0 || (rels.len() == 1 && h < 3.0) };
                    if nothing { if self.trace { eprintln!("[price f{}] {} in f{}: returned below 2 ({})", fno, self.rel_text(r), fid, self.returned_for(r, f.inter)); }
                        if worst.map_or(true, |w| f64::INFINITY > w.0) { worst = Some((f64::INFINITY, *r, true)); } continue; }
                    // for a function of its own where its worth alone is higher than its contribution (R54, experiment)
                    if rels.len() > 1 {
                        let at_r = f.rels.iter().zip(&f.at).find(|(x, _)| *x == r).and_then(|(_, a)| *a);
                        let s20 = at_r.is_some() && Self::cycle_rels_any(f.origin(), &f.rels).is_empty();
                        let (inside, alone) = if s20 {
                            let at = at_r.unwrap();
                            let (wo, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
                            let origin = if wo.iter().any(|x| x.a == f.origin() || x.b == f.origin()) { f.origin() } else { wo[0].a };
                            let without = if wo.len() == 1 { self.alone_over(&wo[0], at, &f.rels, f.inter).map(|x| x - 1.0).unwrap_or_else(|| self.worth_ab(origin, &wo, &wat, f.inter)) } else { self.worth_ab(origin, &wo, &wat, f.inter) };
                            (self.worth_ab(f.origin(), &f.rels, &f.at, f.inter) - without, self.alone_over(r, at, &f.rels, None).map(|x| x - 2.0).unwrap_or(self.halved(r) - 2.0))
                        } else { (self.contribution_ab(f, r), self.halved(r) - 2.0) };
                        if self.trace { eprintln!("[price f{}] {} in f{}: contribution {:.0} alone {:.0} at {:?}", fno, self.rel_text(r), fid, inside, alone, f.rels.iter().zip(&f.at).find(|(x, _)| *x == r).and_then(|(_, a)| a.map(|a| self.dict.sym_text(a)))); }
                        if alone > inside {
                            let gain = alone - inside;
                            if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, false)); }
                        }
                    }
                }
                for (r, t, why) in notes { self.note_middle(fid, &r, t, why); }
                let Some((_, r, nothing)) = worst else { break };
                if self.trace { eprintln!("[leave f{}] {} from {}{}", fno, self.rel_text(&r), self.dict.describe(self.dict.get(fid).unwrap()), if nothing { " for nothing" } else { "" }); }
                let one = rels.len() == 1;
                self.leave(fid, &r, &mut changed, if nothing { "its last relation left for nothing" } else { "its last relation left for one of its own" });
                if !nothing && !one {
                    let g = self.dict.start(r);
                    changed.push(g);
                }
            }
        }
        // a recurring relation no function holds starts one (R50)
        for e in fresh {
            // switch 16: a relation to an activation under a call and the relation to that call are one board
            // relation; where either is at strength 3 and no function holds it, the function starts on the call
            {
                let covered = |r: ActRef| self.act(r).map_or(false, |a| !a.on_apex());
                let (nc, fc) = (covered(e.near), covered(e.far));
                // switch 18: a relation to an activation under a call starts nothing, whatever its count
                if nc || fc {
                    if self.trace && self.dict.holder(&e.rel).is_none() && self.halved(&e.rel) >= 3.0 { eprintln!("[no start f{}] {} | an end stands under a call ({:.1})", fno, self.rel_text(&e.rel), self.halved(&e.rel)); }
                    continue;
                }
                if nc || fc {
                    // both ends under one same activation: nothing is built on it (R50); under different calls, a choice
                    if nc && fc {
                        if !self.share_under(e.near, e.far) && self.trace { eprintln!("[start choice f{}] {} | both ends stand under different calls", fno, self.rel_text(&e.rel)); }
                        continue;
                    }
                    let (x, u) = if fc { (e.near, e.far) } else { (e.far, e.near) };
                    let mut calls = Vec::new();
                    self.apex_over(u, &mut calls, 0);
                    calls.sort();
                    calls.dedup();
                    if calls.len() != 1 { if self.trace { eprintln!("[start choice f{}] {} | {} calls stand over the covered end", fno, self.rel_text(&e.rel), calls.len()); } continue; }
                    let c = calls[0];
                    let (Some(xa), Some(ca)) = (self.act(x), self.act(c)) else { continue };
                    let rc = canon(xa.sym, ca.sym, offset_between(xa, ca));
                    if self.dict.holder(&e.rel).is_some() || self.dict.holder(&rc).is_some() || joined.contains(&rc) { continue; }
                    let (h_u, h_c) = (self.halved(&e.rel), self.halved(&rc));
                    if h_u < 3.0 && h_c < 3.0 { continue; }
                    let ec = Fresh { near: x, far: c, off: offset_between(xa, ca), strength: 0, rel: rc };
                    if self.is_lap_rel(&ec) || self.share_under(x, c) {
                        if self.trace { eprintln!("[no start f{}] {} for {} | {}", fno, self.rel_text(&rc), self.rel_text(&e.rel), if self.is_lap_rel(&ec) { "a lap of the call" } else { "both under one activation" }); }
                        continue;
                    }
                    let id = self.dict.start(rc);
                    self.stats_added += 1;
                    if self.trace { eprintln!("[new f{}] {} | on {} at {:.1} (the call's own {:.1})", fno, self.dict.describe(self.dict.get(id).unwrap()), self.rel_text(&e.rel), h_u, h_c); }
                    changed.push(id);
                    continue;
                }
            }
            let _st = { self.start_strength(&e.rel, e.strength).max(self.strength_through(e)) };
            let st = { if self.halved(&e.rel) >= 3.0 { 3 } else { 0 } };
            let need = { 3 };
            if joined.contains(&e.rel) || self.dict.holder(&e.rel).is_some() || st < need { continue; }
            // a lap of a call starts nothing, nor a relation between two activations under one same one (switch 11, R50)
            // switch 22: a relation between two activations over a shared activation starts like any other; one between an
            // activation and one it stands over still starts nothing
            let refused = { self.share_under(e.near, e.far) };
            if self.is_lap_rel(e) || refused {
                if self.trace { eprintln!("[no start f{}] {} | {}", fno, self.rel_text(&e.rel), if self.is_lap_rel(e) { "a lap of the call" } else { "both under one activation" }); }
                continue;
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

    /// A relation to a call read as one relation with the relations it stands for (switch 11, R50, D47): at each
    /// end that is not the call, the rows that hold the call at that offset, or hold bare a node the call's
    /// function names at its own offset; the larger of the ends. Zero where neither end is a call.
    pub(crate) fn strength_through(&self, e: &Fresh) -> u32 { self.rel_through(&e.rel) }

    /// The same for a relation as the dictionary holds it: at each end whose far end is a call.
    pub(crate) fn rel_through(&self, r: &Rel) -> u32 {
        let mut best = 0;
        for (n, c, off) in r.ends() {
            if c < SYM_CALL || n == c { continue; }
            let named: Vec<(Sym, Offset)> = self.expansion_of(c).iter().map(|(s, d, _)| { let mut q = off; for i in 0..MAX_DIMS { q[i] += d[i]; } (*s, q) }).filter(|(_, q)| q.iter().any(|x| *x != 0)).collect();
            let k = self.rows_of(n).filter(|row| self.holds_bare(row, c, &off) || named.iter().any(|(s, q)| self.holds_bare(row, *s, q))).count() as u32;
            best = best.max(k);
        }
        best
    }

    /// The calls standing on the apex over a covered activation, at any depth (switch 16).
    pub(crate) fn apex_over(&self, r: ActRef, out: &mut Vec<ActRef>, depth: usize) {
        let Some(a) = self.act(r) else { return };
        if depth > 256 { return; }
        for c in &a.coverers {
            let Some(ca) = self.act(*c) else { continue };
            if ca.gone || !self.stands(ca, self.frame_no) { continue; }
            if ca.on_apex() { out.push(*c); } else { self.apex_over(*c, out, depth + 1); }
        }
    }

    /// Whether two activations stand under one same activation, or one under the other (switch 11, R50, D10).
    pub(crate) fn share_under(&self, a: ActRef, b: ActRef) -> bool {
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(m) = self.share_memo.borrow().as_ref() { if let Some(v) = m.get(&key) { return *v; } }
        let v = self.share_under_read(a, b);
        if let Some(m) = self.share_memo.borrow_mut().as_mut() { m.insert(key, v); }
        v
    }

    fn share_under_read(&self, a: ActRef, b: ActRef) -> bool {
        let (Some(x), Some(y)) = (self.act(a), self.act(b)) else { return false };
        let mut ua = self.under(x);
        ua.insert(a);
        let mut ub = self.under(y);
        ub.insert(b);
        ua.iter().any(|r| ub.contains(r))
    }

    /// Whether a relation to a call is that call's own function going round its cycle (switch 11, R50, D53): the
    /// other end is a node of the call's function, and the function holds the relation from it to the node the call
    /// stands at, at that offset.
    pub(crate) fn is_lap_rel(&self, e: &Fresh) -> bool {
        for (o, c) in [(e.near, e.far), (e.far, e.near)] {
            let (Some(oa), Some(ca)) = (self.act(o), self.act(c)) else { continue };
            if ca.covers.is_empty() { continue; }
            let Some(fb) = &ca.fired_by else { continue };
            let Some(f) = self.func(fb.func) else { continue };
            if !f.has_cycle() { continue; }
            let Some(p) = self.dict.pose_of(ca.sym) else { continue };
            let Some(at) = self.dict.node_by_neuron(f, p.from) else { continue };
            if self.dict.node_by_neuron(f, oa.neuron).is_none() { continue; }
            if f.holds(&canon(oa.sym, f.nodes[at], offset_between(oa, ca))) { return true; }
        }
        false
    }

    /// Group A: a relation's strength, its two ends' counts added and halved (R21, R50): at each end the rows that
    /// hold it there, bare or through a call (as the count reads it), and an occurrence whose later end fired this
    /// frame counted at the earlier end too, whose row takes it only at the connect step.
    pub(crate) fn halved(&self, r: &Rel) -> f64 { let (a, b) = self.halved_ends(r); (a + b) as f64 / 2.0 }

    /// A relation's count at each of its two ends (group A): each end the neuron the relation is about, its rows
    /// that hold the relation, bare or through a call; an occurrence whose later end fired this frame is counted
    /// at the earlier end as well, whose row takes it only at the connect step. Where an end names a call the
    /// expansion gives no one neuron for, that end is read as the call itself.
    pub(crate) fn halved_ends(&self, r: &Rel) -> (u32, u32) {
        if let Some(m) = self.halved_memo.borrow().as_ref() { if let Some(v) = m.get(r) { return *v; } }
        let v = self.halved_ends_read(r);
        if let Some(m) = self.halved_memo.borrow_mut().as_mut() { m.insert(*r, v); }
        v
    }

    fn halved_ends_read(&self, r: &Rel) -> (u32, u32) {
        let fno = self.frame_no;
        let (a, b) = { (r.a, r.b) };
        let mut neg = r.off;
        for x in neg.iter_mut() { *x = -*x; }
        let count = |n: Sym, far: Sym, off: &Offset| -> (u32, u32) {
            let mut c = 0;
            let mut now = 0;
            for row in self.rows_of(n) {
                if !self.holds_for(row, far, off, None) { continue; }
                c += 1;
                if off[0] < 0 && row.act.frame == fno { now += 1; }
            }
            (c, now)
        };
        let (ca, na) = count(a, b, &r.off);
        let (cb, nb) = count(b, a, &neg);
        // this frame's occurrence, written so far only at its later end, counts at the earlier end too
        (ca + nb, cb + na)
    }

    /// The relations that go round a function's cycle, a relation of a neuron to itself included (group B).
    pub(crate) fn cycle_rels_any(origin: Sym, rels: &[Rel]) -> Vec<Rel> {
        let f = Func::from_rels(origin, rels);
        let lay = f.layout();
        let Some((ri, ua, ub, _)) = lay.closing else { return vec![] };
        let mut v: Vec<Rel> = Func::tree_path(&lay, ub, ua).iter().map(|s| rels[s.0]).collect();
        v.push(rels[ri]);
        v.sort();
        v.dedup();
        v
    }

    /// A function's worth less its entry under groups A and B (D57): a relation of its cycle is its strength less
    /// one, at no node; any other relation as before, at the node it shares with the function.
    pub(crate) fn worth_ab(&self, origin: Sym, rels: &[Rel], at: &[Option<Sym>], own: Option<NeuronId>) -> f64 {
        if rels.is_empty() { return 0.0; }
        if rels.len() == 1 { return self.halved(&rels[0]) - 1.0; }
        
        let cyc = Self::cycle_rels_any(origin, rels);
        let closing = { Self::closing_of(origin, rels) };
        // switch 24: in a function that is not only a cycle, a relation of its cycle counts only where the function
        // lands, read at its ends
        let only_cycle = rels.iter().all(|r| cyc.contains(r));
        rels.iter().zip(at).map(|(r, a)| if cyc.contains(r) {
            if !only_cycle { self.cycle_rel_at_ends(rels, r, own) } else { self.halved(r) - 1.0 }
        } else { self.worth_in(rels, r, *a, own, closing).0 }).sum()
    }

    /// Switch 24: a relation of a cycle in a function with other relations, read as a chain relation at each of its
    /// ends, over that node's rows with the function's relations at that node, with less without less one; the two
    /// readings added and halved, or one reading where both ends are one node.
    pub(crate) fn cycle_rel_at_ends(&self, rels: &[Rel], r: &Rel, own: Option<NeuronId>) -> f64 {
        let ends: Vec<Sym> = if r.a == r.b { vec![r.a] } else { vec![r.a, r.b] };
        let v: Vec<f64> = ends.iter().map(|n| { let (w, wo) = self.answer(rels, r, Some(*n), own); w as f64 - wo as f64 - 1.0 }).collect();
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// A function with a node read as another (switch 19): the call's node as the activation it stands at.
    pub(crate) fn rewritten(f: &Func, c: Sym, n: Sym) -> Func {
        let mut g = f.clone();
        for x in g.nodes.iter_mut() { if *x == c { *x = n; } }
        for r in g.rels.iter_mut() {
            if r.a == c || r.b == c {
                let a = if r.a == c { n } else { r.a };
                let b = if r.b == c { n } else { r.b };
                *r = canon(a, b, r.off);
            }
        }
        for a in g.at.iter_mut() { if *a == Some(c) { *a = Some(n); } }
        g
    }

    /// A relation alone, read at node `at` over the rows a set is read over there (switch 20): the rows of that node
    /// on which every relation of the set at it has held or failed, counting those that hold this one. None where
    /// the node is no end of it.
    pub(crate) fn alone_over(&self, r: &Rel, at: Sym, set: &[Rel], own: Option<NeuronId>) -> Option<f64> {
        let (n, far, off) = r.ends().into_iter().find(|c| c.0 == at)?;
        let mut offs: Vec<Offset> = vec![off];
        for x in set { for (m, _, o2) in x.ends() { if m == n { offs.push(o2); } } }
        let mut with = 0u32;
        for row in self.rows_of(n) {
            if !offs.iter().all(|o| self.decided(row, o)) { continue; }
            if self.holds_in(row, far, &off, own) { with += 1; }
        }
        Some(with as f64)
    }

    /// The groups of a set of relations that hold together, each as the indexes of its relations (switch 21).
    pub(crate) fn pieces(rels: &[Rel]) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        let mut done = vec![false; rels.len()];
        for i in 0..rels.len() {
            if done[i] { continue; }
            done[i] = true;
            let mut piece = vec![i];
            let mut nodes: Vec<Sym> = vec![rels[i].a, rels[i].b];
            loop {
                let mut grew = false;
                for j in 0..rels.len() {
                    if done[j] || !(nodes.contains(&rels[j].a) || nodes.contains(&rels[j].b)) { continue; }
                    done[j] = true;
                    piece.push(j);
                    for x in [rels[j].a, rels[j].b] { if !nodes.contains(&x) { nodes.push(x); } }
                    grew = true;
                }
                if !grew { break; }
            }
            out.push(piece);
        }
        out
    }

    /// Whether a relation is at an end of its function (switch 21): the function's other relations still hold
    /// together without it.
    pub(crate) fn at_an_end(f: &Func, r: &Rel) -> bool {
        let rest: Vec<Rel> = f.rels.iter().copied().filter(|x| x != r).collect();
        Self::pieces(&rest).len() <= 1
    }

    /// For switch 21's count only: a relation's contribution where taking it out splits its function, each piece
    /// priced as a set of its own. Nothing acts on it.
    pub(crate) fn contribution_split(&self, f: &Func, r: &Rel) -> f64 {
        let (wo, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
        let mut without = 0.0;
        for piece in Self::pieces(&wo) {
            let rs: Vec<Rel> = piece.iter().map(|i| wo[*i]).collect();
            let at: Vec<Option<Sym>> = piece.iter().map(|i| wat[*i]).collect();
            let origin = if rs.iter().any(|x| x.a == f.origin() || x.b == f.origin()) { f.origin() } else { rs[0].a };
            without += self.worth_ab(origin, &rs, &at, f.inter);
        }
        self.worth_ab(f.origin(), &f.rels, &f.at, f.inter) - without
    }

    /// For the count only: a function changed; counted where its interneuron already holds a connection to an action.
    pub(crate) fn note_change(&mut self, fid: FuncId) {
        let Some(n) = self.dict.get(fid).and_then(|f| f.inter) else { return };
        let Some(nn) = self.thalamus.get(n) else { return };
        let holds = nn.connections.keys().any(|(_, t, _)| *t < SYM_CALL && self.thalamus.get(*t).map_or(false, |x| x.kind == Kind::Action));
        if holds { self.changed_after_action += 1; self.changed_after_action_funcs.insert(fid); }
    }

    /// Switch 21: a relation not at an end that would have left, counted once per frame, function, relation and test.
    pub(crate) fn note_middle(&mut self, f: FuncId, r: &Rel, test: u8, why: String) {
        if !self.middle_seen.insert((self.frame_no, f, *r, test)) { return; }
        match test { 0 => self.middle_strength += 1, 1 => self.middle_worth += 1, _ => self.middle_join += 1 }
        if self.trace || self.trace_origin { eprintln!("[middle stays f{}] {} in {} | {}", self.frame_no, self.rel_text(r), self.dict.get(f).map_or(String::new(), |g| self.dict.describe(g)), why); }
    }

    /// A relation in the trace: each end with its neuron and pose.
    pub(crate) fn rel_text(&self, r: &Rel) -> String { format!("{}→{}{:?}", self.dict.sym_text(r.a), self.dict.sym_text(r.b), &r.off[..]) }

    /// A relation leaves a function (R18, R54): what the origin no longer reaches leaves with it and starts a
    /// function of its own where it recurs; a function left with none leaves the dictionary.
    fn leave(&mut self, fid: FuncId, r: &Rel, changed: &mut Vec<FuncId>, why: &str) {
        if self.trace_joins && self.joined_ever.contains(r) { eprintln!("[join-left f{}] {} | {}", self.frame_no, self.rel_text(r), why); }
        // switch 21: an end relation that takes the origin with it leaves the latest of the other nodes as the origin
        let mut origin_note: Option<(String, Sym, Sym, bool)> = None;
        {
            if let Some(f) = self.dict.get(fid) {
                let o = f.origin();
                let rest: Vec<Rel> = f.rels.iter().copied().filter(|x| x != r).collect();
                if !rest.is_empty() && !rest.iter().any(|x| x.a == o || x.b == o) {
                    let g = Func::from_rels(rest[0].a, &rest);
                    let lay = g.layout();
                    let mut order: Vec<usize> = (0..g.nodes.len()).collect();
                    order.sort_by(|&i, &j| { let (p, q) = (lay.position[i], lay.position[j]); q[0].cmp(&p[0]).then_with(|| p[1..].cmp(&q[1..])) });
                    let tie = order.len() > 1 && lay.position[order[0]] == lay.position[order[1]];
                    let no = g.nodes[order[0]];
                    self.origin_moves += 1;
                    if tie { self.origin_ties += 1; }
                    origin_note = Some((self.dict.describe(f), o, no, tie));
                    self.dict.set_origin(fid, no);
                }
            }
        }
        self.note_change(fid);
        let (cut, gone) = self.dict.leave(fid, r);
        if let Some((before, o, no, tie)) = origin_note {
            if self.trace || self.trace_origin {
                eprintln!("[origin leaves f{}] {} | before: {} origin {} | after: {} origin {}{}", self.frame_no, self.rel_text(r), before, self.dict.sym_text(o),
                    self.dict.get(fid).map_or("(gone)".to_string(), |g| self.dict.describe(g)), self.dict.sym_text(no), if tie { " | a tie for the latest node" } else { "" });
            }
        }
        self.forget_expansions(fid);
        for c in cut {
            let _st = { self.returned_for(&c, None) };
            let st = { if self.halved(&c) >= 3.0 { 3 } else { 0 } };
            let need = { 3 };
            if st >= need && self.dict.holder(&c).is_none() { let g = self.dict.start(c); changed.push(g); }
        }
        match gone {
            Some(f) => {
                if self.trace_retire { eprintln!("[retire f{}] {} | {}", self.frame_no, self.dict.describe(&f), why); }
                self.retire(f);
                changed.retain(|x| *x != fid);
            }
            None => { if !changed.contains(&fid) { changed.push(fid); } }
        }
    }

    /// A function has left the dictionary (R18, R38): its interneuron, left with no function, waits on the ledger
    /// to be decommissioned when due, and the function stays readable until what it fired has left the board.
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
