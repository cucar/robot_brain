/// Recognition (§6.1): finding (R53), fitting (D31, D54), what the neurons answer at the count step (D47, R50),
/// and grouping: relations joining, leaving and starting functions by their worth (D57, R50, R54, R18).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::dictionary::*;
use crate::brain::*;
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
    /// the calls it read a node through, its own lapped call among them
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

impl Brain {
    // ───────────────────────────── fitting (D31, D54) ─────────────────────────────

    /// Whether a fit may land on an activation (D28, D10): it is on the apex, or only calls this election has
    /// taken have landed on it, each covering relations of its own.
    pub(crate) fn is_landable(&self, w: &Act) -> bool {
        if w.gone || !self.stands(w, self.frame_no) { return false; }
        // a fit does not read a covered activation (D24)
        w.covered_at.is_none()
    }

    /// A step (D54, D5): the far node lands on its own neuron standing bare at that position, or on
    /// it named there by a call standing on the apex beside the activation, at any depth; a relation already
    /// covered is not read (D28). Returns what the node stands on (the activation, or the call where the call is
    /// one of the fit's own function, which then takes the lap), whether it is that lap, the activation the
    /// relation was read to, and whether it was read through another function's call. Bare before through, the
    /// older first (R24).
    pub(crate) fn step_through(&self, func: FuncId, from: ActRef, delta: Offset, want: Sym, used: &FxHashSet<ActRef>) -> Option<(ActRef, bool, ActRef, Option<ActRef>)> {
        let a = self.find_act(from)?;
        let fno = self.frame_no;
        let target = add(&a.pos, &delta);
        // bare: what stands there on the apex, as the board relations have it
        let mut others: Vec<ActRef> = a.rels.iter().filter(|br| br.off == delta).map(|br| br.other).collect();
        if let Some(fr) = self.thalamus.frames.iter().find(|f| f.no == target[0] as i64) {
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
            let Some(w) = self.find_act(*o) else { continue };
            if w.sym != want || !self.is_landable(w) { continue; }
            // where the call of the lap before stands on that activation, its phase starting there, it takes the lap
            // (D53); nothing is read through a call (D5)
            let mut lap: Option<ActRef> = None;
            for &c in &w.coverers {
                if used.contains(&c) { continue; }
                let Some(ca) = self.find_act(c) else { continue };
                if ca.gone || ca.covered_at.is_some() || !self.stands(ca, fno) || ca.pos != w.pos { continue; }
                if ca.fired_by.as_ref().map_or(true, |fb| fb.func != func) || self.thalamus.dict.find_start(ca.sym) != Some(want) { continue; }
                if lap.map_or(true, |l| c > l) { lap = Some(c); }
            }
            return Some(match lap { Some(c) => (c, true, *o, None), None => (*o, false, *o, None) });
        }
        None
    }

    /// Fit a function entered at node `root` on an activation (D31, D54): read out from there relation by
    /// relation, round the cycle lap after lap while the relation that closes it lands, and the relations that
    /// hang off the cycle from their node on the last lap.
    pub(crate) fn fit(&self, f: &Func, root: usize, entry: ActRef) -> Fit {
        let lay = f.lay_out_from(root);
        let n = f.nodes.len();
        let mut at: Vec<Option<ActRef>> = vec![None; n];
        let mut via_call = vec![false; n];
        at[root] = Some(entry);
        let mut used: FxHashSet<ActRef> = FxHashSet::default();
        used.insert(entry);
        // it starts on the node's own pose, or on a call whose phase starts at the node, which it lands
        // on whole and reads the function's relations from (D54, D31)
        let on_call = self.find_act(entry).map_or(false, |a| a.sym != f.nodes[root]);
        let mut landed = if on_call { vec![] } else { vec![(entry, root)] };
        let mut calls = if on_call { vec![(entry, root)] } else { vec![] };
        let mut edges = Vec::new();
        let mut came = vec![false; f.rels.len()];
        // the nodes past a call the fit landed on: the lap stops at the call, and nothing past it fails (D54, D10)
        let mut beyond = vec![false; n];
        let mut inside = vec![false; f.rels.len()];
        let mut laps = 0u32;
        // the cycle: its path once round, from the node the closing relation reaches back to where it starts
        let cycle_path: Vec<(usize, usize, usize, Offset)> = match lay.closing { Some((_, ua, ub, _)) => Func::find_tree_path(&lay, ub, ua), None => vec![] };
        let mut on_cycle: Vec<bool> = vec![false; n];
        if let Some((_, ua, ub, _)) = lay.closing { on_cycle[ua] = true; on_cycle[ub] = true; for s in &cycle_path { on_cycle[s.1] = true; on_cycle[s.2] = true; } }
        let path_rels: Vec<usize> = cycle_path.iter().map(|s| s.0).collect();
        let thru_calls: std::cell::RefCell<Vec<ActRef>> = std::cell::RefCell::new(Vec::new());
        // a function in time is fit as it unfolds (D54, D31): entered at a node, the nodes later than it are still to
        // come, and the fit leaves them out with anything reached only through them, charging nothing for them
        let later: Vec<bool> = lay.position.iter().map(|p| lay.closing.is_none() && p[0] > 0).collect();
        // the step: what the node stands on, whether a call, and the activation the relation was read to
        let go = |this: &Brain, au: ActRef, off: Offset, want: Sym, used: &FxHashSet<ActRef>| -> Option<(ActRef, bool, ActRef)> {
            let (w, lap, to, via) = this.step_through(f.id, au, off, want, used)?;
            if let Some(c) = via { thru_calls.borrow_mut().push(c); }
            Some((w, lap, to))
        };
        let try_step = |this: &Brain, ri: usize, u: usize, v: usize, off: Offset, at: &mut Vec<Option<ActRef>>, via_call: &mut Vec<bool>, used: &mut FxHashSet<ActRef>, landed: &mut Vec<(ActRef, usize)>, calls: &mut Vec<(ActRef, usize)>, edges: &mut Vec<(ActRef, ActRef)>, came: &mut Vec<bool>, beyond: &mut Vec<bool>, inside: &mut Vec<bool>| -> Option<bool> {
            if beyond[u] || (at[u].is_some() && via_call[u]) { inside[ri] = true; if at[v].is_none() { beyond[v] = true; } return None; }
            if later[v] { inside[ri] = true; if at[v].is_none() { beyond[v] = true; } return None; }
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
        // the fit of a cycle ends at the first of its relations that does not land, none a failure
        let cyc_ri: Vec<usize> = { match lay.closing { Some((ri, _, _, _)) => { let mut v = path_rels.clone(); v.push(ri); v } None => vec![] } };
        let failed: Vec<usize> = (0..f.rels.len()).filter(|ri| !came[*ri] && !inside[*ri] && Some(*ri) != closing_ri && !cyc_ri.contains(ri)).collect();
        let mut thru = thru_calls.into_inner();
        thru.sort();
        thru.dedup();
        Fit { func: f.id, entry: (root, entry), landed, calls, edges, failed, laps, thru }
    }

    /// Every fit of a function this frame (R53, D54): one per node and activation of the present frame on the
    /// apex that the node admits, its own pose or a call whose phase starts at it; the same landing
    /// reached from two entries is one fit.
    pub(crate) fn find_fits(&self, id: FuncId) -> Vec<Fit> {
        let Some(f) = self.thalamus.dict.get(id) else { return vec![] };
        let fno = self.frame_no;
        let mut out: Vec<Fit> = Vec::new();
        let mut keys: FxHashSet<Vec<ActRef>> = FxHashSet::default();
        let Some(fr) = self.thalamus.frames.iter().find(|fr| fr.no == fno) else { return out };
        for (root, node) in f.nodes.iter().enumerate() {
            // a function is entered at any node that fires in the present frame (D54, D31, R53): in time the nodes later
            // than that node are still to come and the fit leaves them out; a cycle's fit ends where it stops landing
            for (i, a) in fr.acts.iter().enumerate() {
                // the node admits its own pose, or a call whose phase starts at it (D54)
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
        let mut v: Vec<FuncId> = fresh.iter().filter_map(|e| self.thalamus.dict.find_holder(&e.rel)).collect();
        // a relation to a call is read as the nodes its function names, each at its own offset (R53, D5)
        
        v.sort_unstable();
        v.dedup();
        v
    }

    // ───────────────────────────── what the neurons answer (D47, R50) ─────────────────────────────

    /// Every node a call in its phase expands to, at any depth, each at its offset from the call's position and with
    /// the interneurons of the calls that put it there, the call's own first (R28, R50): its function read from the
    /// node its phase starts at, its cycle round as many laps as the phase says, and each call among its nodes
    /// expanded in turn; none for a base neuron.
    pub(crate) fn read_expansion(&self, s: Sym) -> std::rc::Rc<Vec<(Sym, Offset, Vec<NeuronId>)>> {
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
        let gone: Vec<Sym> = self.expansion_deps.borrow().iter().filter(|(_, d)| d.contains(&fid)).map(|(s, _)| *s).collect();
        let (mut e, mut d) = (self.expansions.borrow_mut(), self.expansion_deps.borrow_mut());
        for s in gone { e.remove(&s); d.remove(&s); }
    }

    /// Drop every expansion kept.
    pub(crate) fn forget_all_expansions(&self) {
        self.expansions.borrow_mut().clear();
        self.expansion_deps.borrow_mut().clear();
    }

    fn expand_nodes(&self, s: Sym, at: Offset, path: &mut Vec<NeuronId>, out: &mut Vec<(Sym, Offset, Vec<NeuronId>)>, deps: &mut Vec<FuncId>) {
        let Some(phase) = self.thalamus.dict.find_pose(s) else { return };
        let n = self.thalamus.dict.find_neuron_of(s);
        if path.contains(&n) || path.len() > 64 { return; }
        let Some(f) = self.find_function_for(s).cloned() else { return };
        deps.push(f.id);
        let root = f.find_node(phase.from).unwrap_or(0);
        let lay = f.lay_out_from(root);
        let mut nodes: Vec<(Sym, Offset)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, lay.position[i])).collect();
        if let Some(l) = f.find_lap() {
            // the cycle read back from the node the phase starts at, lap after lap, to the node it ends at (D53)
            let order = f.order_back(root);
            if !order.is_empty() {
                let to = f.find_node(phase.to).unwrap_or(order[order.len() - 1]);
                let total = f.stretch(root, to, phase.laps);
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

    /// Ask one count of a pose's rows (D47, R50). The answer comes from this grouping's cache; one not yet answered
    /// is recorded for the next batch to the neurons and read as zero until then.
    fn ask(&self, q: CountQuery) -> (u32, u32) {
        assert!(self.counting, "counts are asked only while grouping");
        let n = self.thalamus.dict.find_neuron_of(q.read_pose());
        if (n as usize) >= self.thalamus.registry.len() { return (0, 0); }
        if let Some(a) = self.counts.borrow().get(&q) { return *a; }
        let mut missed = self.missed.borrow_mut();
        if !missed.iter().any(|(_, x)| *x == q) { missed.push((n, q)); }
        (0, 0)
    }

    /// Read a decision of grouping until every count it asks is answered (R50): the counts it asked that were not yet
    /// answered go to the neurons in one batch, and it is read again. No row changes while grouping, so each answer is
    /// the one a single reading would have had.
    pub(crate) fn settle<T>(&mut self, read: impl Fn(&Self) -> T) -> T {
        loop {
            let v = read(self);
            let missed = std::mem::take(&mut *self.missed.borrow_mut());
            if missed.is_empty() { return v; }
            let queries: Vec<CountQuery> = missed.iter().map(|(_, q)| q.clone()).collect();
            let answers = self.thalamus.count_relations(missed, self.frame_no);
            let mut counts = self.counts.borrow_mut();
            for (q, a) in queries.into_iter().zip(answers) { counts.insert(q, a); }
        }
    }

    /// The pose a relation is read from where it has no node to be counted at (D47, R50): an end whose
    /// rows hold it; where both do, the one with more activations behind it. None where no rows hold it.
    fn find_holding_end(&self, r: &Rel) -> Option<(Sym, Sym, Offset)> {
        r.ends().into_iter()
            .filter(|c| self.ask(CountQuery::Holding { own: c.0, far: c.1, off: c.2 }).0 > 0)
            .map(|c| (c, self.ask(CountQuery::Rows { own: c.0 }).0))
            .max_by_key(|x| x.1).map(|x| x.0)
    }

    /// `with` and `without` for one relation of a set (R50), over the rows of the node it is counted at: `with`, every
    /// row that holds it; `without`, every row where the set's other relations at that node held and something else
    /// stands where it expected its neuron. Where nothing stands there, the row counts neither way; where the node holds
    /// none of the set's other relations, nothing is `without`.
    pub(crate) fn count_with_without(&self, rels: &[Rel], r: &Rel, at: Option<Sym>) -> (u32, u32) {
        // at the node it attached at (R50), where it has one; else at an end whose rows hold it
        let end = at.and_then(|n| r.ends().into_iter().find(|c| c.0 == n)).or_else(|| self.find_holding_end(r));
        let Some((n, far, off)) = end else { return (0, 0) };
        let mut cond: Vec<(Sym, Offset)> = Vec::new();
        for s in rels {
            if s == r { continue; }
            for (m, f2, o2) in s.ends() { if m == n { cond.push((f2, o2)); } }
        }
        cond.sort();
        self.ask(CountQuery::Answer { own: n, far, off, cond })
    }

    /// A relation's worth in a set read for `own` (D57): `with − without − 1`; with its nearness,
    /// `with / (with + without)`, and its `with`.
    pub(crate) fn measure_worth_in(&self, rels: &[Rel], r: &Rel, at: Option<Sym>, closing: Option<Rel>) -> (f64, f64, u32) {
        let (w, wo) = self.count_with_without(rels, r, at);
        let near = if w + wo == 0 { 0.0 } else { w as f64 / (w + wo) as f64 };
        // nothing counts against the relation that closes a cycle
        let x = if closing == Some(*r) { w as f64 - 1.0 } else { w as f64 - wo as f64 - 1.0 };
        (x, near, w)
    }

    /// The relation that closes the cycle of a set of relations listed from `origin` (D15), where it has one.
    fn find_closing(origin: Sym, rels: &[Rel]) -> Option<Rel> {
        let f = Func::from_rels(origin, rels);
        f.lay_out().closing.map(|c| rels[c.0])
    }

    /// A relation's contribution to the function it is in, read at the node it attached at, over that node's rows
    /// (R54, R50): the function's worth with it less without it, a function of one relation left without it read
    /// alone at that node over the same rows. Where the relation attached at no node, or the function holds a cycle,
    /// its contribution read at no node.
    pub(crate) fn measure_contribution_at_node(&self, f: &Func, r: &Rel) -> f64 {
        let at_r = f.rels.iter().zip(&f.at).find(|(x, _)| *x == r).and_then(|(_, a)| *a);
        let Some(at) = at_r.filter(|_| f.rels.len() > 1 && Self::list_cycle_rels(f.read_root(), &f.rels).is_empty()) else { return self.measure_contribution(f, r) };
        let (wo, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
        let origin = if wo.iter().any(|x| x.a == f.read_root() || x.b == f.read_root()) { f.read_root() } else { wo[0].a };
        let without = if wo.len() == 1 { self.count_alone(&wo[0], at, &f.rels).map(|x| x - 1.0).unwrap_or_else(|| self.measure_worth(origin, &wo, &wat)) } else { self.measure_worth(origin, &wo, &wat) };
        self.measure_worth(f.read_root(), &f.rels, &f.at) - without
    }

    /// A relation's contribution to its function (D57, R54): the function's worth with it less without it, the entry
    /// the same on both sides; as its function's only relation, that function's worth, its strength less two.
    pub(crate) fn measure_contribution(&self, f: &Func, r: &Rel) -> f64 {
        let (without, wat): (Vec<Rel>, Vec<Option<Sym>>) = f.rels.iter().zip(&f.at).filter(|(x, _)| *x != r).map(|(x, a)| (*x, *a)).unzip();
        if without.is_empty() { return self.count_strength(r) - 2.0; }
        let origin = if without.iter().any(|x| x.a == f.read_root() || x.b == f.read_root()) { f.read_root() } else { without[0].a };
        self.measure_worth(f.read_root(), &f.rels, &f.at) - self.measure_worth(origin, &without, &wat)
    }

    // ───────────────────────────── grouping (R50, R54, R18) ─────────────────────────────

    /// The relations a board relation would be to a function at an activation it shares with the function's fit or
    /// call (R50): from the node there to the other end; a call read as each node of the function its expansion puts
    /// where the shared activation sees, at that node's offset (R50, R28), and as itself where it puts none.
    fn list_candidates(&self, f: &Func, shared: ActRef, as_node: Sym, origin_pos: &Pos, other: ActRef) -> Vec<Rel> {
        let (Some(sa), Some(oa)) = (self.find_act(shared), self.find_act(other)) else { return vec![] };
        let off = offset_between(sa, oa);
        let mut reads: Vec<(Sym, Offset)> = Vec::new();
        reads.push((oa.sym, off));
        reads.into_iter().filter_map(|(read, q)| self.make_candidate(f, as_node, origin_pos, oa.pos, read, q)).collect()
    }

    /// One relation from node `as_node` to `read` at `off` as a candidate of a function (R50), the far end standing
    /// at `far_pos`. None where it is no candidate.
    fn make_candidate(&self, f: &Func, as_node: Sym, origin_pos: &Pos, far_pos: Pos, read: Sym, off: Offset) -> Option<Rel> {
        let rel = canon(as_node, read, off);
        if f.holds(&rel) { return None; }
        if f.find_node(read).is_some() {
            // it would close the cycle, where the function has none
            if f.has_cycle() { return None; }
        } else {
            // a new node may not be the origin: later in time, or at a lesser position in the origin's frame
            if later(&far_pos, origin_pos) { return None; }
            // nothing leads on from a node of the cycle in the direction it advances in (D15)
            if let Some(lap) = f.find_lap() {
                let node = f.find_node(as_node)?;
                if f.list_cycle_nodes().contains(&node) && parallel(&off, &lap) { return None; }
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
    pub(crate) fn list_board_rels(&self) -> Vec<Fresh> {
        let fno = self.frame_no;
        let mut out: FxHashMap<(ActRef, ActRef), Fresh> = FxHashMap::default();
        for fr in self.thalamus.frames.iter() {
            for (i, a) in fr.acts.iter().enumerate() {
                if a.gone || !self.stands(a, fno) { continue; }
                let near = ActRef { frame: fr.no, index: i as u32 };
                for br in &a.rels {
                    let Some(b) = self.find_act(br.other) else { continue };
                    if b.gone || !self.stands(b, fno) { continue; }
                    // a relation a call covers is off the board
                    if self.covered_rel.contains(&edge(near, br.other)) || self.lifted.contains(&edge(near, br.other)) { continue; }
                    if !br.strength < 2 { continue; }
                    let k = edge(near, br.other);
                    if out.get(&k).map_or(false, |x| x.strength >= br.strength) { continue; }
                    out.insert(k, Fresh { near, far: br.other, strength: br.strength, rel: canon(a.sym, b.sym, br.off) });
                }
            }
        }
        let mut v: Vec<Fresh> = out.into_values().collect();
        v.sort_by(|a, b| (a.near, a.far).cmp(&(b.near, b.far)));
        v
    }

    pub(crate) fn group(&mut self, fits: &[Fit], fresh: &[Fresh], board: &[Fresh], fired: &FxHashSet<Sym>) -> Vec<FuncId> {
        // no row changes while grouping, so each count is read once and kept for the grouping
        assert!(self.lifts_due.is_empty(), "lifts are handed over before grouping reads the rows");
        self.counting = true;
        let out = self.group_inner(fits, fresh, board, fired);
        self.counting = false;
        self.counts.borrow_mut().clear();
        out
    }

    fn group_inner(&mut self, fits: &[Fit], fresh: &[Fresh], board: &[Fresh], fired: &FxHashSet<Sym>) -> Vec<FuncId> {
        
        self.forget_all_expansions();
        self.start_queue.clear();
        let mut changed: Vec<FuncId> = Vec::new();
        let fno = self.frame_no;
        // the activations each function touches here: what its fits landed on, each as its node, with the position
        // its origin stands at; and its calls on the board, as the node their phase starts at
        let mut touch: FxHashMap<FuncId, Vec<(ActRef, Sym, Pos)>> = FxHashMap::default();
        // the board edges each function's fits already land on: one edge is never two of its relations
        let mut taken: FxHashMap<FuncId, FxHashSet<(ActRef, ActRef)>> = FxHashMap::default();
        // the fits this frame's election took touch the board as the fits of this grouping do
        let mut all_fits: Vec<Fit> = fits.to_vec();
        {
            let mut mine: Vec<(ActRef, Fit)> = self.call_fit.iter().map(|(r, f)| (*r, f.clone())).collect();
            mine.sort_by_key(|x| x.0);
            all_fits.extend(mine.into_iter().map(|x| x.1));
        }
        for fit in all_fits.iter() {
            let Some(f) = self.thalamus.dict.get(fit.func) else { continue };
            taken.entry(f.id).or_default().extend(fit.edges.iter().map(|&(a, b)| edge(a, b)));
            let lay = f.lay_out();
            let Some(entry) = self.find_act(fit.entry.1) else { continue };
            let mut opos = entry.pos;
            for d in 0..MAX_DIMS { opos[d] -= lay.position[fit.entry.0][d]; }
            for (l, node) in fit.acts() { touch.entry(f.id).or_default().push((l, f.nodes[node], opos)); }
        }
        for e in board {
            for end in [e.near, e.far] {
                let Some(a) = self.find_act(end) else { continue };
                if a.kind != Kind::Inter || !a.on_apex() || !self.stands(a, fno) { continue; }
                let Some(fb) = &a.fired_by else { continue };
                let Some(f) = self.thalamus.dict.get(fb.func) else { continue };
                let Some(n) = self.thalamus.dict.find_start(a.sym) else { continue };
                let Some(node) = f.find_node(n) else { continue };
                let start = f.nodes[node];
                let lay = f.lay_out();
                let mut opos = a.pos;
                for d in 0..MAX_DIMS { opos[d] -= lay.position[node][d]; }
                touch.entry(f.id).or_default().push((end, start, opos));
            }
        }
        let mut at_act: FxHashMap<ActRef, Vec<(FuncId, Sym, Pos)>> = FxHashMap::default();
        let mut fids: Vec<FuncId> = touch.keys().copied().collect();
        fids.sort_unstable();
        for fid in fids { for &(act, as_node, opos) in &touch[&fid] { at_act.entry(act).or_default().push((fid, as_node, opos)); } }
        // besides what the first election left on the board, each fit is offered the relations covered by
        // the calls it names or reads a node through
        let offered: Vec<(Fresh, Option<FuncId>)> = board.iter().map(|e| (*e, None)).collect();
        // a relation under a call that also landed on an activation the fit landed on, at that activation;
        // where the fit landed on a call as a node, the relations under it are no candidates (R50)
        let mut offered19: Vec<(Fresh, FuncId, ActRef, Sym, Pos)> = Vec::new();
        {
            let mut seen: FxHashSet<(FuncId, (ActRef, ActRef), ActRef)> = FxHashSet::default();
            for fit in all_fits.iter() {
                let Some(f) = self.thalamus.dict.get(fit.func) else { continue };
                let lay = f.lay_out();
                let Some(entry) = self.find_act(fit.entry.1) else { continue };
                let mut opos = entry.pos;
                for d in 0..MAX_DIMS { opos[d] -= lay.position[fit.entry.0][d]; }
                let mut offers: Vec<(ActRef, ActRef, Sym)> = Vec::new();
                // a call that also landed on an activation the fit landed on: its relations that share that
                // activation (R50)
                for &(l, node) in &fit.landed {
                    let Some(la) = self.find_act(l) else { continue };
                    for &c in &la.coverers {
                        if self.find_act(c).and_then(|ca| ca.fired_by.as_ref()).map_or(true, |fb| fb.func == fit.func) { continue; }
                        offers.push((c, l, f.nodes[node]));
                    }
                }
                for (c, shared, as_node) in offers {
                    let Some(es) = self.call_edges.get(&c) else { continue };
                    for &(p, q) in es {
                        if p != shared && q != shared { continue; }
                        if !seen.insert((fit.func, (p, q), shared)) { continue; }
                        let (Some(pa), Some(qa)) = (self.find_act(p), self.find_act(q)) else { continue };
                        let off = offset_between(pa, qa);
                        offered19.push((Fresh { near: p, far: q, strength: 2, rel: canon(pa.sym, qa.sym, off) }, fit.func, shared, as_node, opos));
                    }
                }
            }
        }
        
        // the candidates (R50), read once every strength they ask is answered
        let cands: FxHashMap<Rel, Vec<(FuncId, Sym)>> = self.settle(|m| {
            let mut cands: FxHashMap<Rel, Vec<(FuncId, Sym)>> = FxHashMap::default();
            for (e, only) in offered.iter() {
                let e = *e;
                if m.half.map_or(false, |k| (e.rel.off[0] == 0) != k) { continue; }
                // two calls that landed on one activation relate like any two neighbors (D26, R50); a relation between
                // an activation and one it stands over is no relation
                if m.stands_over(e.near, e.far) { continue; }
                for (act, other) in [(e.near, e.far), (e.far, e.near)] {
                    let Some(list) = at_act.get(&act) else { continue };
                    for &(fid, as_node, opos) in list {
                        if only.map_or(false, |o| o != fid) { continue; }
                        let Some(f) = m.thalamus.dict.get(fid) else { continue };
                        if taken.get(&fid).map_or(false, |t| t.contains(&edge(act, other))) { continue; }
                        for rel in m.list_candidates(f, act, as_node, &opos, other) {
                            if m.half.map_or(false, |k| (rel.off[0] == 0) != k) { continue; }
                            // a candidate has strength 2 or more, read through calls (R50, D47)
                            if m.count_strength(&rel) < 2.0 { continue; }
                            // where both its ends are nodes of the function, it attaches at the end whose activation
                            // returned it, never at a node read through a call (R50)
                            let attach = if f.find_node(rel.a).is_some() && f.find_node(rel.b).is_some() {
                                [e.near, e.far].iter().filter_map(|r| m.find_act(*r)).map(|a| a.sym).find(|s| *s == rel.a || *s == rel.b).unwrap_or(as_node)
                            } else { as_node };
                            let v = cands.entry(rel).or_default();
                            if !v.iter().any(|x| x.0 == fid) { v.push((fid, attach)); }
                        }
                    }
                }
            }
            for &(e, fid, shared, as_node, opos) in &offered19 {
                if m.half.map_or(false, |k| (e.rel.off[0] == 0) != k) { continue; }
                if m.stands_over(e.near, e.far) { continue; }
                let other = if e.near == shared { e.far } else { e.near };
                let Some(f) = m.thalamus.dict.get(fid) else { continue };
                if taken.get(&fid).map_or(false, |t| t.contains(&edge(shared, other))) { continue; }
                for rel in m.list_candidates(f, shared, as_node, &opos, other) {
                    if m.half.map_or(false, |k| (rel.off[0] == 0) != k) { continue; }
                    if m.count_strength(&rel) < 2.0 { continue; }
                    let attach = if f.find_node(rel.a).is_some() && f.find_node(rel.b).is_some() {
                        [e.near, e.far].iter().filter_map(|r| m.find_act(*r)).map(|a| a.sym).find(|s| *s == rel.a || *s == rel.b).unwrap_or(as_node)
                    } else { as_node };
                    let v = cands.entry(rel).or_default();
                    if !v.iter().any(|x| x.0 == fid) { v.push((fid, attach)); }
                }
            }
            cands
        });
        // one that closes a cycle before one that adds a node (R50)
        let closes = |m: &Brain, rel: &Rel, fs: &[(FuncId, Sym)]| fs.iter().filter_map(|x| m.thalamus.dict.get(x.0)).any(|f| f.find_node(rel.a).is_some() && f.find_node(rel.b).is_some());
        let mut keys: Vec<(bool, Rel)> = cands.iter().map(|(r, fs)| (!closes(self, r, fs), *r)).collect();
        keys.sort();
        let mut joined: FxHashSet<Rel> = FxHashSet::default();
        for (_, rel) in keys {
            // the best function for it and its contribution where it stands, read once every count they ask is answered
            let weighed = self.settle(|m| {
                let fs = cands[&rel].clone();
                // its contribution where it stands: to its function, or alone, its strength less one; standing alone or
                // in a function of one relation, a tie joins, since it leaves fewer functions (R54)
                let holder = m.thalamus.dict.find_holder(&rel).and_then(|g| m.thalamus.dict.get(g));
                let lone = holder.map_or(true, |g| g.rels.len() == 1);
                let mut best: Option<(f64, f64, u64, FuncId, Sym)> = None;
                for (fid, attach) in fs {
                    let Some(f) = m.thalamus.dict.get(fid) else { continue };
                    if f.holds(&rel) { continue; }
                    // a function holds one kind, inside the frame or in time (D15, R50)
                    if f.rels.iter().any(|r| (r.off[0] == 0) != (rel.off[0] == 0)) { continue; }
                    // read against the function as it now stands: it touches a node, and closes no second cycle
                    let (ka, kb) = (f.find_node(rel.a).is_some(), f.find_node(rel.b).is_some());
                    if !(ka || kb) || (ka && kb && f.has_cycle()) { continue; }
                    // a cycle is a function of its own (D15, R50): a relation of a pose to itself joins no function, while one
                    // neuron in two phases is two nodes and a relation between them is a candidate like any other; no
                    // relation joins a cycle; a relation closes a cycle only where every relation is then on the cycle
                    if rel.a == rel.b { continue; }
                    if f.has_cycle() { continue; }
                    // a candidate that closes the cycle (R50): across frames any return closes it; inside the frame only
                    // the way back itself, from the node it attaches at to the other end, as the function's relations lay
                    // those nodes out, in every component, so it reaches the very activation the fit has there
                    if ka && kb && rel.off[0] == 0 {
                        let (u, v, seen) = if attach == rel.b { (rel.b, rel.a, neg(&rel.off)) } else { (rel.a, rel.b, rel.off) };
                        let lay = f.lay_out();
                        let (iu, iv) = (f.find_node(u).unwrap(), f.find_node(v).unwrap());
                        let mut back = [0i32; MAX_DIMS];
                        for d in 0..MAX_DIMS { back[d] = lay.position[iv][d] - lay.position[iu][d]; }
                        if seen != back { continue; }
                    }
                    if ka && kb {
                        let mut deg: FxHashMap<Sym, u32> = FxHashMap::default();
                        for r in f.rels.iter().chain(std::iter::once(&rel)) { *deg.entry(r.a).or_default() += 1; *deg.entry(r.b).or_default() += 1; }
                        if deg.values().any(|d| *d != 2) { continue; }
                    }
                    let w = {
                        let mut with = f.rels.clone();
                        with.push(rel);
                        let before = Func::find_meeting(&f.rels, &f.at, attach);
                        let mut at = before.clone();
                        at.push(Some(attach));
                        // a function of one relation read at the node the candidate attaches at, over the
                        // same rows, where no cycle is involved
                        let lone_before = if f.rels.len() == 1 && Self::list_cycle_rels(f.read_root(), &with).is_empty() {
                            m.count_alone(&f.rels[0], attach, &with).map(|x| x - 1.0)
                        } else { None };
                        let wb = lone_before.unwrap_or_else(|| m.measure_worth(f.read_root(), &f.rels, &before));
                        let wa = m.measure_worth(f.read_root(), &with, &at);
                        wa - wb
                    };
                    let mut rels = f.rels.clone();
                    rels.push(rel);
                    let near = m.measure_worth_in(&rels, &rel, Some(attach), None).1;
                    let better = match best { None => true, Some((bw, bn, bj, _, _)) => w > bw || (w == bw && (near > bn || (near == bn && f.joined < bj))) };
                    if better { best = Some((w, near, f.joined, fid, attach)); }
                }
                let Some((w, _, _, fid, attach)) = best else { return None };
                // a relation not at an end of the function it is in does not leave it, to join or otherwise
                if let Some(g) = holder { if !Self::is_at_an_end(g, &rel) { return None; } }
                // alone, read at the node it would attach at, over the rows of the function it is compared with (R50)
                let here20 = if lone {
                    m.thalamus.dict.get(fid).and_then(|f| {
                        let mut with = f.rels.clone();
                        with.push(rel);
                        // the candidate that closes a cycle is read alone at no node
                        if !Self::list_cycle_rels(f.read_root(), &with).is_empty() { return None; }
                        m.count_alone(&rel, attach, &with).map(|x| x - 2.0)
                    })
                } else { None };
                let own = match holder {
                    _ if here20.is_some() => here20.unwrap(),
                    _ if lone => m.count_strength(&rel) - 2.0,
                    // its contribution where it stands, read as leaving reads it, at the node it attached at (R54, R50)
                    Some(g) => m.measure_contribution_at_node(g, &rel),
                    None => m.count_strength(&rel) - 2.0,
                };
                // a function's last relation is weighed against everything that leaves with it: the function and its
                // subtree, every function that leaves the dictionary with its interneuron, and so on up (D57, R54, R38)
                let sub: Vec<(FuncId, f64)> = match holder { Some(g) if lone => m.list_subtree(g), _ => Vec::new() };
                let here = own + sub.iter().map(|x| x.1).sum::<f64>();
                // a candidate no function holds joins where its gain is above zero, and nothing else is asked of it: it is
                // not weighed against a function of its own, which it does not have (R50, R54)
                let here = if holder.is_none() { 0.0 } else { here };
                Some((w, fid, attach, here))
            });
            let Some((w, fid, attach, here)) = weighed else { continue };
            if !(w > here) { continue; }
            let from = self.thalamus.dict.find_holder(&rel);
            if let Some(g) = from { self.leave(g, &rel, &mut changed); }
            if self.thalamus.dict.get(fid).is_none() { continue; }
            self.thalamus.dict.join(fid, rel, attach);
            self.forget_expansions(fid);
            joined.insert(rel);
            if !changed.contains(&fid) { changed.push(fid); }
        }
        // leaving (R54, R18): one relation at a time, the one whose leaving raises the dictionary's worth most, the
        // function priced again between; over every function that fit here, changed, or that a symbol firing at
        // this level stands in (R18, R50)
        let standing: Vec<FuncId> = self.thalamus.dict.funcs.values().filter(|f| f.nodes.iter().any(|n| fired.contains(n))).map(|f| f.id).collect();
        let mut moved: Vec<FuncId> = fits.iter().map(|f| f.func).chain(changed.iter().copied()).chain(standing).collect();
        moved.sort_unstable();
        moved.dedup();
        if let Some(k) = self.half { moved.retain(|id| self.thalamus.dict.get(*id).map_or(false, |f| f.rels.iter().all(|r| r.off[0] == 0) == k)); }
        for fid in moved {
            loop {
                // the relation that leaves, read once every count it asks is answered
                let priced = self.settle(|m| {
                    let Some(f) = m.thalamus.dict.get(fid) else { return None };
                    let rels = f.rels.clone();
                    let mut worst: Option<(f64, Rel, bool)> = None;
                    // whether the relation chosen to leave leaves because it does not pay there (R54)
                    let mut worst_unpaid = false;
                    for r in &rels {
                        // a relation not at an end is not priced for leaving
                        if !Self::is_at_an_end(f, r) { continue; }
                        // for nothing once the neurons return it below 2 over every row that holds it (R18, D47)
                        let h = m.count_strength(r);
                        if h < 2.0 || (rels.len() == 1 && h < 3.0) {
                            if worst.map_or(true, |w| f64::INFINITY > w.0) { worst = Some((f64::INFINITY, *r, true)); }
                            continue;
                        }
                        if rels.len() > 1 {
                            let at_r = f.rels.iter().zip(&f.at).find(|(x, _)| *x == r).and_then(|(_, a)| *a);
                            if Self::list_cycle_rels(f.read_root(), &f.rels).is_empty() {
                                // a relation in a function of more than one relation leaves it where it does not pay there: its
                                // own worth, with less without less one at the node it is attached at, is not above zero; it is
                                // not weighed against a function of its own, and the one worth least leaves first (R50, R54)
                                let (wi, wo) = m.count_with_without(&f.rels, r, at_r);
                                let worth = wi as f64 - wo as f64 - 1.0;
                                if !(worth > 0.0) {
                                    let gain = -worth;
                                    if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, false)); worst_unpaid = true; }
                                }
                            } else {
                                // a relation of a cycle has no without: it leaves for a function of its own where its worth alone
                                // is higher than its contribution to the cycle (R50, R54)
                                let (inside, alone) = (m.measure_contribution(f, r), m.count_strength(r) - 2.0);
                                if alone > inside {
                                    let gain = alone - inside;
                                    if worst.map_or(true, |w| gain > w.0) { worst = Some((gain, *r, false)); worst_unpaid = false; }
                                }
                            }
                        }
                    }
                    Some((worst, worst_unpaid, rels.len()))
                });
                let Some((worst, worst_unpaid, count)) = priced else { break };
                let Some((_, r, nothing)) = worst else { break };
                let one = count == 1;
                let unpaid = !nothing && worst_unpaid;
                // a relation that leaves because it does not pay there is held by no function (R54)
                self.leave(fid, &r, &mut changed);
                if !nothing && !one && !unpaid {
                    let g = self.thalamus.dict.start(r);
                    changed.push(g);
                }
            }
        }
        // a recurring relation no function holds may start one (R50)
        let strengths: Vec<f64> = self.settle(|m| fresh.iter().map(|e| m.count_strength(&e.rel)).collect());
        for (e, strength) in fresh.iter().zip(strengths) {
            // a relation held 3 times or more may start a function, counted as itself; what stops it is that no
            // instance of it stands, never that a call has landed on one of its ends (R50, R18)
            if joined.contains(&e.rel) || self.thalamus.dict.find_holder(&e.rel).is_some() || strength < 3.0 { continue; }
            // a lap of a call starts nothing (R50, D53); two calls that landed on one activation relate like any two
            // neighbors and start a function as any relation does (D26, R50); one between an activation and one it
            // stands over starts nothing
            let refused = { self.stands_over(e.near, e.far) };
            if self.is_lap_rel(e) || refused {
                continue;
            }
            
            // functions start one at a time (R50): what may start one is queued, and starts after this half's second
            // election from a relation that still stands
            if self.half.is_some() { if !self.start_queue.contains(&e.rel) { self.start_queue.push(e.rel); } continue; }
            let id = self.thalamus.dict.start(e.rel);
            self.stats_added += 1;
            changed.push(id);
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// Whether one activation stands over the other (D5): one is under the other's call.
    pub(crate) fn stands_over(&self, a: ActRef, b: ActRef) -> bool {
        let (Some(x), Some(y)) = (self.find_act(a), self.find_act(b)) else { return false };
        self.list_under(x).contains(&b) || self.list_under(y).contains(&a)
    }

    /// Whether a relation to a call is that call's own function going round its cycle (R50, D53): the
    /// other end is a node of the call's function, and the function holds the relation from it to the node the call
    /// stands at, at that offset.
    pub(crate) fn is_lap_rel(&self, e: &Fresh) -> bool {
        for (o, c) in [(e.near, e.far), (e.far, e.near)] {
            let (Some(oa), Some(ca)) = (self.find_act(o), self.find_act(c)) else { continue };
            if ca.covers.is_empty() { continue; }
            let Some(fb) = &ca.fired_by else { continue };
            let Some(f) = self.find_function(fb.func) else { continue };
            if !f.has_cycle() { continue; }
            let Some(p) = self.thalamus.dict.find_pose(ca.sym) else { continue };
            let Some(at) = f.find_node(p.from) else { continue };
            if f.find_node(oa.sym).is_none() { continue; }
            if f.holds(&canon(oa.sym, f.nodes[at], offset_between(oa, ca))) { return true; }
        }
        false
    }

    /// Group A: a relation's strength, the larger of its two ends' counts (R21, R50): at each end the rows that
    /// hold it there, bare or through a call (as the count reads it), and an occurrence whose later end fired this
    /// frame counted at the earlier end too, whose row takes it only at the connect step.
    pub(crate) fn count_strength(&self, r: &Rel) -> f64 { let (a, b) = self.count_ends(r); a.max(b) as f64 }

    /// A relation's count at each of its two ends (R21, R50): at each end the rows of its pose that hold the relation,
    /// off the relation index; an occurrence whose later end fired this frame is counted at the earlier end as well,
    /// whose row takes it only at the connect step.
    pub(crate) fn count_ends(&self, r: &Rel) -> (u32, u32) {
        let mut neg = r.off;
        for x in neg.iter_mut() { *x = -*x; }
        let (ca, na) = self.ask(CountQuery::Ends { own: r.a, far: r.b, off: r.off });
        let (cb, nb) = self.ask(CountQuery::Ends { own: r.b, far: r.a, off: neg });
        (ca + nb, cb + na)
    }

    /// The functions that leave the dictionary with a function's interneuron, and so on up (R38): where the
    /// function is its interneuron's only one, every function that names that interneuron, and those that name the
    /// interneurons left with no function by that; each with its worth, its relations' worth less its entry (D57).
    pub(crate) fn list_subtree(&self, g: &Func) -> Vec<(FuncId, f64)> {
        let mut gone: FxHashSet<FuncId> = FxHashSet::default();
        gone.insert(g.id);
        let mut queue: Vec<NeuronId> = Vec::new();
        if let Some(n) = g.inter { if self.thalamus.dict.list_of_inter(n).iter().all(|x| gone.contains(x)) { queue.push(n); } }
        let mut seen_n: FxHashSet<NeuronId> = FxHashSet::default();
        let mut out = Vec::new();
        while let Some(n) = queue.pop() {
            if !seen_n.insert(n) { continue; }
            for id in self.thalamus.dict.list_naming(n) {
                if !gone.insert(id) { continue; }
                let Some(f) = self.thalamus.dict.get(id) else { continue };
                out.push((id, self.measure_worth(f.read_root(), &f.rels, &f.at) - 1.0));
                if let Some(m) = f.inter { if self.thalamus.dict.list_of_inter(m).iter().all(|x| gone.contains(x)) { queue.push(m); } }
            }
        }
        out
    }

    /// The relations that go round a function's cycle, a relation of a neuron to itself included.
    pub(crate) fn list_cycle_rels(origin: Sym, rels: &[Rel]) -> Vec<Rel> {
        let f = Func::from_rels(origin, rels);
        let lay = f.lay_out();
        let Some((ri, ua, ub, _)) = lay.closing else { return vec![] };
        let mut v: Vec<Rel> = Func::find_tree_path(&lay, ub, ua).iter().map(|s| rels[s.0]).collect();
        v.push(rels[ri]);
        v.sort();
        v.dedup();
        v
    }

    /// A function's worth less its entry (D57): a relation of a cycle is its strength less one, at no node; any other
    /// relation its worth at the node it shares with the function. `origin` is the node
    /// the relations are read from: the function's origin, or any node of a cycle, which reads the same from each.
    pub(crate) fn measure_worth(&self, origin: Sym, rels: &[Rel], at: &[Option<Sym>]) -> f64 {
        if rels.is_empty() { return 0.0; }
        if rels.len() == 1 { return self.count_strength(&rels[0]) - 1.0; }
        
        let cyc = Self::list_cycle_rels(origin, rels);
        let closing = { Self::find_closing(origin, rels) };
        // a function that is a cycle holds the cycle and nothing else (D15): each relation of it is counted at every
        // occurrence, its strength less one, with no without (D57, D54)
        rels.iter().zip(at).map(|(r, a)| if cyc.contains(r) { self.count_strength(r) - 1.0 } else { self.measure_worth_in(rels, r, *a, closing).0 }).sum()
    }


    /// A relation alone, read at node `at` over the rows a set is read over there: the rows of that node
    /// on which every relation of the set at it has held or failed, counting those that hold this one. None where
    /// the node is no end of it.
    pub(crate) fn count_alone(&self, r: &Rel, at: Sym, set: &[Rel]) -> Option<f64> {
        let (n, far, off) = r.ends().into_iter().find(|c| c.0 == at)?;
        let mut offs: Vec<Offset> = vec![off];
        for x in set { for (m, _, o2) in x.ends() { if m == n { offs.push(o2); } } }
        offs.sort();
        offs.dedup();
        Some(self.ask(CountQuery::Alone { own: n, far, off, offs }).0 as f64)
    }

    /// The groups of a set of relations that hold together, each as the indexes of its relations.
    pub(crate) fn split_pieces(rels: &[Rel]) -> Vec<Vec<usize>> {
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

    /// Whether a relation is at an end of its function: the function's other relations still hold
    /// together without it.
    pub(crate) fn is_at_an_end(f: &Func, r: &Rel) -> bool {
        let rest: Vec<Rel> = f.rels.iter().copied().filter(|x| x != r).collect();
        Self::split_pieces(&rest).len() <= 1
    }

    /// A relation leaves a function (R18, R54): what the origin no longer reaches leaves with it and starts a
    /// function of its own where it recurs; a function left with none leaves the dictionary.
    fn leave(&mut self, fid: FuncId, r: &Rel, changed: &mut Vec<FuncId>) {
        // an end relation that takes the origin with it leaves the latest of the other nodes as the origin
        {
            if let Some(f) = self.thalamus.dict.get(fid) {
                let o = f.read_root();
                let rest: Vec<Rel> = f.rels.iter().copied().filter(|x| x != r).collect();
                // a cycle has no origin: what is left of it once a relation leaves takes the latest of its nodes
                if !rest.is_empty() && (f.find_origin().is_none() || !rest.iter().any(|x| x.a == o || x.b == o)) {
                    let g = Func::from_rels(rest[0].a, &rest);
                    let lay = g.lay_out();
                    let mut order: Vec<usize> = (0..g.nodes.len()).collect();
                    order.sort_by(|&i, &j| { let (p, q) = (lay.position[i], lay.position[j]); q[0].cmp(&p[0]).then_with(|| p[1..].cmp(&q[1..])) });
                    let no = g.nodes[order[0]];
                    self.thalamus.dict.set_origin(fid, no);
                }
            }
        }
        let (cut, gone) = self.thalamus.dict.leave(fid, r);
        self.forget_expansions(fid);
        let strengths: Vec<f64> = self.settle(|m| cut.iter().map(|c| m.count_strength(c)).collect());
        for (c, strength) in cut.into_iter().zip(strengths) {
            if strength >= 3.0 && self.thalamus.dict.find_holder(&c).is_none() { let g = self.thalamus.dict.start(c); changed.push(g); }
        }
        match gone {
            Some(f) => {
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
            if self.thalamus.dict.list_of_inter(n).is_empty() && !self.ledger.contains(&n) { self.ledger.push(n); }
        }
        let until = self.frame_no + self.max_reach_t.max(self.span_s) as i64 + 1;
        self.retired_funcs.insert(f.id, (f, until));
    }

    /// A function as expansion reads it: the dictionary's, or the copy kept while what it fired is open (R38).
    pub(crate) fn find_function(&self, id: FuncId) -> Option<&Func> {
        self.thalamus.dict.get(id).or_else(|| self.retired_funcs.get(&id).map(|x| &x.0))
    }
}
