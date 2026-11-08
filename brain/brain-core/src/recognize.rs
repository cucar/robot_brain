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
        w.on_apex() || (!w.coverers.is_empty() && w.coverers.iter().all(|c| self.electing.contains(c)))
    }

    /// From a landed activation, the board relation at `delta` whose far end is `want`, or a call whose pose starts
    /// at `want` standing there (D54); none a call covers (D28); the older first (R24). Whether it is a call.
    fn step(&self, from: ActRef, delta: Offset, want: Sym, used: &FxHashSet<ActRef>) -> Option<(ActRef, bool)> {
        let a = self.act(from)?;
        let target = add(&a.pos, &delta);
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
        let mut best: Option<(ActRef, bool)> = None;
        for o in others {
            if used.contains(&o) || self.covered_rel.contains(&edge(from, o)) { continue; }
            let Some(w) = self.act(o) else { continue };
            if !self.landable(w) { continue; }
            let is = if w.sym == want { Some(false) } else if w.kind == Kind::Inter && self.dict.starts_at(w.sym, want) { Some(true) } else { None };
            let Some(call) = is else { continue };
            let better = match best { None => true, Some((b, bc)) => (call, o) < (bc, b) };
            if better { best = Some((o, call)); }
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
        let mut laps = 0u32;
        // the cycle: its path once round, from the node the closing relation reaches back to where it starts
        let cycle_path: Vec<(usize, usize, usize, Offset)> = match lay.closing { Some((_, ua, ub, _)) => Func::tree_path(&lay, ub, ua), None => vec![] };
        let mut on_cycle: Vec<bool> = vec![false; n];
        if let Some((_, ua, ub, _)) = lay.closing { on_cycle[ua] = true; on_cycle[ub] = true; for s in &cycle_path { on_cycle[s.1] = true; on_cycle[s.2] = true; } }
        let path_rels: Vec<usize> = cycle_path.iter().map(|s| s.0).collect();
        let try_step = |this: &Machine, ri: usize, u: usize, v: usize, off: Offset, at: &mut Vec<Option<ActRef>>, via_call: &mut Vec<bool>, used: &mut FxHashSet<ActRef>, landed: &mut Vec<(ActRef, usize)>, calls: &mut Vec<(ActRef, usize)>, edges: &mut Vec<(ActRef, ActRef)>, came: &mut Vec<bool>| -> Option<bool> {
            let au = at[u]?;
            if via_call[u] { return None; }
            let (w, call) = this.step(au, off, f.nodes[v], used)?;
            came[ri] = true;
            edges.push((au, w));
            used.insert(w);
            at[v] = Some(w);
            via_call[v] = call;
            if call { calls.push((w, v)); } else { landed.push((w, v)); }
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
            loop {
                if laps > 4096 { break; }
                let Some(au) = at[ua] else { break };
                if via_call[ua] { break; }
                let Some((w, call)) = self.step(au, off, f.nodes[ub], &used) else { break };
                came[ri] = true;
                edges.push((au, w));
                used.insert(w);
                at[ub] = Some(w);
                if call { calls.push((w, ub)); via_call[ub] = true; break; }
                laps += 1;
                landed.push((w, ub));
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
        // the relation that closes a cycle is never a failure: the fit ends where it stops landing (D54)
        let closing_ri = lay.closing.map(|c| c.0);
        let failed: Vec<usize> = (0..f.rels.len()).filter(|ri| !came[*ri] && Some(*ri) != closing_ri).collect();
        Fit { func: f.id, entry: (root, entry), landed, calls, edges, failed, laps }
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
                let admits = a.sym == *node || (a.kind == Kind::Inter && self.dict.starts_at(a.sym, *node));
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
        v.sort_unstable();
        v.dedup();
        v
    }

    // ───────────────────────────── what the neurons answer (D47, R50) ─────────────────────────────

    /// Every node a call in its pose expands to, at any depth, each at its offset from the call's position (R28,
    /// R50): its function read from the node its pose starts at, its cycle round as many laps as the pose says, and
    /// each call among its nodes expanded in turn; none for a base neuron.
    pub(crate) fn expansion_of(&self, s: Sym) -> std::rc::Rc<Vec<(Sym, Offset)>> {
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
        let gone: Vec<Sym> = self.expansion_deps.borrow().iter().filter(|(_, d)| d.contains(&fid)).map(|(s, _)| *s).collect();
        let (mut e, mut d) = (self.expansions.borrow_mut(), self.expansion_deps.borrow_mut());
        for s in gone { e.remove(&s); d.remove(&s); }
    }

    /// Drop every expansion kept.
    pub(crate) fn forget_all_expansions(&self) {
        self.held_memo.borrow_mut().clear();
        self.strength_memo.borrow_mut().clear();
        self.expansions.borrow_mut().clear();
        self.expansion_deps.borrow_mut().clear();
    }

    fn expand_nodes(&self, s: Sym, at: Offset, path: &mut Vec<NeuronId>, out: &mut Vec<(Sym, Offset)>, deps: &mut Vec<FuncId>) {
        let Some(pose) = self.dict.pose_of(s) else { return };
        let n = self.dict.neuron_of(s);
        if path.contains(&n) || path.len() > 64 { return; }
        let f = self.dict.of_inter(n).iter().filter_map(|id| self.dict.get(*id)).find(|f| self.dict.node_by_neuron(f, pose.from).is_some()).cloned()
            .or_else(|| self.func_of(n).cloned());
        let Some(f) = f else { return };
        deps.push(f.id);
        let root = self.dict.node_by_neuron(&f, pose.from).unwrap_or(0);
        let lay = f.layout_from(root);
        let mut nodes: Vec<(Sym, Offset)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, lay.position[i])).collect();
        if let Some(l) = f.lap() {
            let cyc = f.cycle_nodes();
            for k in 1..=pose.laps as i32 {
                for &v in &cyc { let mut q = lay.position[v]; for d in 0..MAX_DIMS { q[d] += k * l[d]; } nodes.push((f.nodes[v], q)); }
            }
        }
        path.push(n);
        for (m, q) in nodes {
            let mut p = at;
            for d in 0..MAX_DIMS { p[d] += q[d]; }
            out.push((m, p));
            self.expand_nodes(m, p, path, out, deps);
        }
        path.pop();
    }

    /// Whether a row holds a relation to a node (D47, R50): the node's neuron in its pose at that offset, or a call
    /// whose expansion puts that node at that offset from the row's activation.
    fn row_holds(&self, row: &Row, far: Sym, off: &Offset) -> bool {
        row.nbs.iter().any(|nb| {
            if nb.sym == far && nb.off == *off { return true; }
            if nb.sym < SYM_CALL { return false; }
            let mut d = [0i32; MAX_DIMS];
            for i in 0..MAX_DIMS { d[i] = off[i] - nb.off[i]; }
            self.expansion_of(nb.sym).iter().any(|(s, q)| *s == far && *q == d)
        })
    }

    /// The rows of a symbol's neuron that it fired in that pose (D18).
    fn rows_of(&self, s: Sym) -> impl Iterator<Item = &Row> + '_ {
        let n = self.dict.neuron_of(s);
        self.thalamus.get(n).into_iter().flat_map(move |nn| nn.history.iter().filter(move |r| r.own == s))
    }

    /// The neuron in its pose a relation is read from (D47, R50): an end whose rows hold it, at whatever offset
    /// (D10); where both ends hold it, the one with more activations behind it. None where no rows hold it.
    fn held_by(&self, r: &Rel) -> Option<(Sym, Sym, Offset)> {
        if let Some(h) = self.held_memo.borrow().get(r) { return *h; }
        let h = self.held_by_rows(r);
        self.held_memo.borrow_mut().insert(*r, h);
        h
    }

    fn held_by_rows(&self, r: &Rel) -> Option<(Sym, Sym, Offset)> {
        r.ends().into_iter()
            .filter(|c| self.rows_of(c.0).any(|row| self.row_holds(row, c.1, &c.2)))
            .map(|c| (c, self.rows_count(c.0)))
            .max_by_key(|x| x.1).map(|x| x.0)
    }

    /// How many rows a symbol's neuron holds in that pose: a base neuron has one pose, so all of its history.
    fn rows_count(&self, s: Sym) -> usize {
        if s < SYM_CALL { return self.thalamus.get(s).map_or(0, |n| n.history.len()); }
        self.rows_of(s).count()
    }

    /// How many rows of its neuron hold a relation (D47), reading a call as the node its pose starts at.
    pub(crate) fn strength_of(&self, r: &Rel) -> u32 {
        if let Some(s) = self.strength_memo.borrow().get(r) { return *s; }
        let s = match self.held_by(r) {
            Some((n, far, off)) => self.rows_of(n).filter(|row| self.row_holds(row, far, &off)).count() as u32,
            None => 0,
        };
        self.strength_memo.borrow_mut().insert(*r, s);
        s
    }

    /// `with` and `without` for one relation of a set (R50): over the rows of the end that answers for it, how
    /// many hold the set's other relations at that end together with it, and how many hold them and not it.
    /// Where that end holds none of the others, every row that holds it is `with`, and nothing is `without`.
    pub(crate) fn answer(&self, rels: &[Rel], r: &Rel, at: Option<Sym>) -> (u32, u32) {
        // at the node it attached at (R50), where it has one; else at an end whose rows hold it
        let end = at.and_then(|n| r.ends().into_iter().find(|c| c.0 == n)).or_else(|| self.held_by(r));
        let Some((n, far, off)) = end else { return (0, 0) };
        let mut cond: Vec<(Sym, Offset)> = Vec::new();
        for s in rels {
            if s == r { continue; }
            for (m, f2, o2) in s.ends() { if m == n { cond.push((f2, o2)); } }
        }
        let (mut with, mut without) = (0, 0);
        for row in self.rows_of(n) {
            if !cond.iter().all(|(f2, o2)| self.row_holds(row, *f2, o2)) { continue; }
            if self.row_holds(row, far, &off) { with += 1; } else if !cond.is_empty() { without += 1; }
        }
        (with, without)
    }

    /// The relation that closes the cycle of a set of relations listed from `origin` (D15), where it has one.
    fn closing_of(origin: Sym, rels: &[Rel]) -> Option<Rel> {
        let f = Func::from_rels(origin, rels);
        f.layout().closing.map(|c| rels[c.0])
    }

    /// A relation's worth in a set (D57): `with − without − 1`, and `with − 1` for the relation that closes a
    /// cycle; with its nearness, `with / (with + without)`, and its `with`.
    pub(crate) fn worth_in(&self, rels: &[Rel], r: &Rel, at: Option<Sym>, closing: Option<Rel>) -> (f64, f64, u32) {
        let (w, wo) = self.answer(rels, r, at);
        let near = if w + wo == 0 { 0.0 } else { w as f64 / (w + wo) as f64 };
        let x = if closing == Some(*r) { w as f64 - 1.0 } else { w as f64 - wo as f64 - 1.0 };
        (x, near, w)
    }

    /// A function's worth (D57): the sum of its relations' worths; a call costs nothing (experiment).
    pub(crate) fn worth_of(&self, origin: Sym, rels: &[Rel], at: &[Option<Sym>]) -> f64 {
        if rels.is_empty() { return 0.0; }
        let closing = Self::closing_of(origin, rels);
        rels.iter().zip(at).map(|(r, a)| self.worth_in(rels, r, *a, closing).0).sum()
    }

    /// A relation's worth in a function of its own (D57): its strength less one, a call costing nothing
    /// (experiment).
    pub(crate) fn alone(&self, r: &Rel) -> f64 { self.strength_of(r) as f64 - 1.0 }

    /// A relation's gain in a function it attaches to at node `attach` (D57, R54, R50): the function's worth with it
    /// less its worth without it, the relation counted at the node it attaches at.
    pub(crate) fn gain(&self, f: &Func, c: &Rel, attach: Sym) -> f64 {
        let mut with = f.rels.clone();
        with.push(*c);
        let mut at = Func::meeting(&f.rels, &f.at, attach);
        at.push(Some(attach));
        self.worth_of(f.origin(), &with, &at) - self.worth_of(f.origin(), &f.rels, &f.at)
    }

    /// A relation's contribution to the function it is in (D57, R54): the function's worth with it less without
    /// it; alone, its worth in a function of its own, its strength less one (experiment).
    pub(crate) fn contribution(&self, f: &Func, r: &Rel) -> f64 {
        let (without, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
        if without.is_empty() { return self.alone(r); }
        let origin = if without.iter().any(|x| x.a == f.origin() || x.b == f.origin()) { f.origin() } else { without[0].a };
        self.worth_of(f.origin(), &f.rels, &f.at) - self.worth_of(origin, &without, &wat)
    }

    // ───────────────────────────── grouping (R50, R54, R18) ─────────────────────────────

    /// The relations a board relation would be to a function at an activation it shares with the function's fit or
    /// call (R50): from the node there to the other end; a call read as each node of the function its expansion puts
    /// where the shared activation sees, at that node's offset (R50, R28), and as itself where it puts none.
    fn candidates(&self, f: &Func, shared: ActRef, as_node: Sym, origin_pos: &Pos, other: ActRef) -> Vec<Rel> {
        let (Some(sa), Some(oa)) = (self.act(shared), self.act(other)) else { return vec![] };
        let off = offset_between(sa, oa);
        let mut reads: Vec<(Sym, Offset)> = Vec::new();
        if oa.sym >= SYM_CALL {
            for (s, d) in self.expansion_of(oa.sym).iter() {
                if f.node_of(*s).is_none() { continue; }
                let mut q = off;
                for i in 0..MAX_DIMS { q[i] += d[i]; }
                if q.iter().all(|x| *x == 0) || !(0..MAX_DIMS).all(|i| q[i].abs() <= sa.reach[i]) { continue; }
                reads.push((*s, q));
            }
        }
        if reads.is_empty() { reads.push((oa.sym, off)); }
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
    pub(crate) fn group(&mut self, fits: &[Fit], fresh: &[Fresh], fired: &FxHashSet<Sym>) -> Vec<FuncId> {
        self.forget_all_expansions();
        let mut changed: Vec<FuncId> = Vec::new();
        let fno = self.frame_no;
        // the activations each function touches here: what its fits landed on, each as its node, with the position
        // its origin stands at; and its calls on the board, as the node their pose starts at
        let mut touch: FxHashMap<FuncId, Vec<(ActRef, Sym, Pos)>> = FxHashMap::default();
        // the board edges each function's fits already land on: one edge is never two of its relations
        let mut taken: FxHashMap<FuncId, FxHashSet<(ActRef, ActRef)>> = FxHashMap::default();
        for fit in fits {
            let Some(f) = self.dict.get(fit.func) else { continue };
            taken.entry(f.id).or_default().extend(fit.edges.iter().map(|&(a, b)| edge(a, b)));
            let lay = f.layout();
            let Some(entry) = self.act(fit.entry.1) else { continue };
            let mut opos = entry.pos;
            for d in 0..MAX_DIMS { opos[d] -= lay.position[fit.entry.0][d]; }
            for (l, node) in fit.acts() { touch.entry(f.id).or_default().push((l, f.nodes[node], opos)); }
        }
        for e in fresh {
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
            }
        }
        // the candidates (R50)
        let mut cands: FxHashMap<Rel, Vec<(FuncId, Sym)>> = FxHashMap::default();
        for e in fresh {
            if e.strength < 2 { continue; }
            for (fid, list) in &touch {
                let Some(f) = self.dict.get(*fid) else { continue };
                for &(act, as_node, opos) in list {
                    let other = if e.near == act { e.far } else if e.far == act { e.near } else { continue };
                    if taken.get(fid).map_or(false, |t| t.contains(&edge(act, other))) { continue; }
                    for rel in self.candidates(f, act, as_node, &opos, other) {
                        let v = cands.entry(rel).or_default();
                        if !v.iter().any(|x| x.0 == *fid) { v.push((*fid, as_node)); }
                    }
                }
            }
        }
        // one that closes a cycle before one that adds a node (R50)
        let closes = |m: &Machine, rel: &Rel, fs: &[(FuncId, Sym)]| fs.iter().filter_map(|x| m.dict.get(x.0)).any(|f| f.node_of(rel.a).is_some() && f.node_of(rel.b).is_some());
        let mut keys: Vec<(bool, Rel)> = cands.iter().map(|(r, fs)| (!closes(self, r, fs), *r)).collect();
        keys.sort();
        let mut joined: FxHashSet<Rel> = FxHashSet::default();
        for (_, rel) in keys {
            let fs = cands[&rel].clone();
            // its contribution where it stands: to its function, or alone, its strength less one (experiment)
            let here = match self.dict.holder(&rel).and_then(|g| self.dict.get(g)) {
                Some(g) => self.contribution(g, &rel),
                None => self.alone(&rel),
            };
            let mut best: Option<(f64, f64, u64, FuncId, Sym)> = None;
            for (fid, attach) in fs {
                let Some(f) = self.dict.get(fid) else { continue };
                if f.holds(&rel) { continue; }
                // read against the function as it now stands: it touches a node, and closes no second cycle
                let (ka, kb) = (f.node_of(rel.a).is_some(), f.node_of(rel.b).is_some());
                if !(ka || kb) || (ka && kb && f.has_cycle()) { continue; }
                let w = self.gain(f, &rel, attach);
                let mut rels = f.rels.clone();
                rels.push(rel);
                let near = self.worth_in(&rels, &rel, Some(attach), None).1;
                let better = match best { None => true, Some((bw, bn, bj, _, _)) => w > bw || (w == bw && (near > bn || (near == bn && f.joined < bj))) };
                if better { best = Some((w, near, f.joined, fid, attach)); }
            }
            let Some((w, _, _, fid, attach)) = best else { continue };
            if self.trace { eprintln!("[cand f{}] {} best f{} gain {:.0} | where it stands f{:?} {:.0}", fno, self.rel_text(&rel), fid, w, self.dict.holder(&rel), here); }
            if !(w > here) { continue; }
            if self.trace { eprintln!("[join f{}] {} into {} gain {:.0} over {:.0}", fno, self.rel_text(&rel), self.dict.describe(self.dict.get(fid).unwrap()), w, here); }
            if let Some(g) = self.dict.holder(&rel) { self.leave(g, &rel, &mut changed); }
            if self.dict.get(fid).is_none() { continue; }
            self.dict.join(fid, rel, attach);
            self.forget_expansions(fid);
            joined.insert(rel);
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
                for r in &rels {
                    // for nothing once its strength has slid below 2 (R18)
                    if self.strength_of(r) < 2 { if worst.map_or(true, |w| f64::INFINITY > w.0) { worst = Some((f64::INFINITY, *r, true)); } continue; }
                    // for a function of its own where its worth alone is higher than its contribution (R54, experiment)
                    if rels.len() > 1 {
                        let inside = self.contribution(f, r);
                        let alone = self.alone(r);
                        if alone > inside {
                            let gain = alone - inside;
                            if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, false)); }
                        }
                    }
                }
                let Some((_, r, nothing)) = worst else { break };
                if self.trace { eprintln!("[leave f{}] {} from {}{}", fno, self.rel_text(&r), self.dict.describe(self.dict.get(fid).unwrap()), if nothing { " for nothing" } else { "" }); }
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
            let id = self.dict.start(e.rel);
            self.stats_added += 1;
            if self.trace { eprintln!("[new f{}] {}", fno, self.dict.describe(self.dict.get(id).unwrap())); }
            changed.push(id);
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// A relation in the trace: each end with its neuron and pose.
    pub(crate) fn rel_text(&self, r: &Rel) -> String { format!("{}→{}{:?}", self.dict.sym_text(r.a), self.dict.sym_text(r.b), &r.off[..]) }

    /// A relation leaves a function (R18, R54): what the origin no longer reaches leaves with it and starts a
    /// function of its own where it recurs; a function left with none leaves the dictionary.
    fn leave(&mut self, fid: FuncId, r: &Rel, changed: &mut Vec<FuncId>) {
        let (cut, gone) = self.dict.leave(fid, r);
        self.forget_expansions(fid);
        for c in cut {
            if self.strength_of(&c) >= 2 && self.dict.holder(&c).is_none() { let g = self.dict.start(c); changed.push(g); }
        }
        match gone {
            Some(f) => { self.retire(f); changed.retain(|x| *x != fid); }
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
