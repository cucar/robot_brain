//! Recognize and prune (§6.1): the margin (D30), grouping (R50), finding (R53), generalizing (R52) and pruning
//! (R18). The machine does all of it over what the neurons return and over their histories; no neuron holds a
//! function.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::fit::*;
use crate::function::*;
use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── the margin (D30) ─────────────────────────────

    /// The rows of the anchor's history a function may fit: those holding any of its relations to a fixed
    /// node, or every row where it has a variable (D47).
    pub(crate) fn candidate_rows(&self, f: &Func) -> Vec<u64> {
        let n = &self.thalamus[f.anchor];
        if f.rels.iter().any(|r| !r.node.is_fixed()) || f.nodes_for(f.anchor).len() > 1 { return n.history.iter().map(|r| r.id).collect(); }
        let mut rows: Vec<u64> = Vec::new();
        for r in &f.rels {
            if let Node::Fixed(m) = r.node { if let Some(v) = n.index.get(&(m, r.dim, r.off)) { rows.extend_from_slice(v); } }
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// Writing the anchor's history with `add` in the dictionary in place of `remove`, against as it stands:
    /// the cost saved over the rows where any of them fits, less the change in the dictionary's entries (D30).
    /// `new_classes` are the members of the classes `add` would bring into the dictionary.
    pub(crate) fn replacement_gain(&self, add: &[&Func], remove: &[FuncId], new_classes: &[Vec<NeuronId>]) -> f64 {
        let Some(anchor) = add.first().map(|f| f.anchor).or_else(|| remove.first().and_then(|id| self.dict.get(*id)).map(|f| f.anchor)) else { return 0.0 };
        let n = &self.thalamus[anchor];
        let radii = &self.radii;
        let before: Vec<&Func> = self.dict.standing(anchor).iter().filter_map(|id| self.dict.get(*id)).collect();
        let after: Vec<&Func> = before.iter().copied().filter(|f| !remove.contains(&f.id)).chain(add.iter().copied()).collect();
        let mut rows: Vec<u64> = Vec::new();
        for f in add.iter().copied().chain(remove.iter().filter_map(|id| self.dict.get(*id))) { rows.extend(self.candidate_rows(f)); }
        rows.sort_unstable();
        rows.dedup();
        let mut gain = 0.0;
        for (row, w) in by_sig(rows.iter().filter_map(|id| n.row(*id))) {
            let changed = add.iter().any(|f| fits(f, row, &self.dict, radii)) || remove.iter().filter_map(|i| self.dict.get(*i)).any(|f| fits(f, row, &self.dict, radii));
            if !changed { continue; }
            gain += w * (row_cost(row, &before, &self.dict, radii) - row_cost(row, &after, &self.dict, radii));
        }
        let removed: f64 = remove.iter().filter_map(|id| self.dict.get(*id)).map(|f| self.dict.entry(f, &[])).sum();
        let added: f64 = add.iter().enumerate().map(|(i, f)| self.dict.entry(f, if i == 0 { new_classes } else { &[] })).sum();
        gain + removed - added
    }

    /// Every margin of the functions anchored at a neuron at once (D30): each kind of row is covered with all
    /// of them and once without each that fits it.
    pub(crate) fn margins_of(&self, x: NeuronId) -> FxHashMap<FuncId, f64> {
        let n = &self.thalamus[x];
        let funcs: Vec<&Func> = self.dict.standing(x).iter().filter_map(|id| self.dict.get(*id)).collect();
        let mut m: FxHashMap<FuncId, f64> = funcs.iter().map(|f| (f.id, -self.dict.entry(f, &[]))).collect();
        let fresh_of = |row: &Row| -> Vec<(FuncId, Landing, u64)> {
            let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
            funcs.iter().filter_map(|f| land_best(f, x, &self.dict, &self.radii, row.axes, &row.nbs, &row.calls, &fresh).map(|l| (f.id, l, f.joined))).filter(|(_, l, _)| l.savings() > 0.0).collect()
        };
        // rows alike are priced once; what the cover leaves of each row is the assignment grouping reads
        let mut kinds: FxHashMap<u64, (usize, Vec<u64>)> = FxHashMap::default();
        let mut order: Vec<u64> = Vec::new();
        for (k, row) in n.history.iter().enumerate() {
            let s = row_sig(row);
            let e = kinds.entry(s).or_insert_with(|| { order.push(s); (k, Vec::new()) });
            e.1.push(row.id);
        }
        let mut assigned: FxHashMap<u64, (f64, Vec<f64>)> = FxHashMap::default();
        for s in order {
            let (k, ids) = &kinds[&s];
            let row = &n.history[*k];
            let w = ids.len() as f64;
            let ls = fresh_of(row);
            if ls.is_empty() { continue; }
            let all: Vec<(&Landing, u64)> = ls.iter().map(|(_, l, j)| (l, *j)).collect();
            let (with, _, left) = cover_left(row, &all);
            for id in ids { assigned.insert(*id, (left.anchor, left.nbs.clone())); }
            for k in 0..ls.len() {
                let rest: Vec<(&Landing, u64)> = ls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, (_, l, j))| (l, *j)).collect();
                *m.get_mut(&ls[k].0).unwrap() += w * (greedy(row, &rest) - with);
            }
        }
        self.assigned.borrow_mut().insert(x, assigned);
        m
    }

    // ───────────────────────────── grouping (R50): assign, update, seed ─────────────────────────────

    /// Group an activation's relations into functions, one online step of Lloyd's algorithm over its neuron's
    /// history (R50): the rows are the points and the functions the centres, and the distance from a row to a
    /// function is what the function saves writing it. Assign: the greedy cover writes the new row with the
    /// dictionary's functions (D28). Update: each function the cover took that is anchored here, of fixed nodes,
    /// moves one relation at a time, added or dropped, while that lowers the cost of writing the history (D30).
    /// Seed: the relations held in two or more rows that the cover left unwritten open one new function, trimmed
    /// the same way. Returns the seed where its margin is positive, for find (R53).
    pub(crate) fn group(&mut self, x: NeuronId, ret: &ActReturn) -> Vec<Vec<Rel>> {
        let (updates, seed) = {
            let Some(row) = self.thalamus[x].row(ret.row) else { return Vec::new() };
            // assign: the cover of the new row by every function the neuron reads
            let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
            let ls: Vec<(FuncId, Landing, u64)> = self.dict.standing(x).iter().filter_map(|id| self.dict.get(*id))
                .filter_map(|f| land_best(f, x, &self.dict, &self.radii, row.axes, &row.nbs, &row.calls, &fresh).map(|l| (f.id, l, f.joined)))
                .filter(|(_, l, _)| l.savings() > 0.0).collect();
            let refs: Vec<(&Landing, u64)> = ls.iter().map(|(_, l, j)| (l, *j)).collect();
            let (_, taken, now) = cover_left(row, &refs);
            let Some(t) = Table::new(self, x, ret, Some(&now)) else { return Vec::new() };
            // update: each function taken, anchored here, all of whose relations the row holds
            let mut updates: Vec<(FuncId, Vec<Rel>)> = Vec::new();
            let mut written: FxHashSet<usize> = FxHashSet::default();
            for &k in &taken {
                let (fid, l, _) = &ls[k];
                written.extend(l.covered.iter().copied());
                written.extend(l.bound.iter().map(|(i, _)| *i));
                written.extend(l.landed.iter().map(|(_, i)| *i).filter(|i| *i != usize::MAX));
                let Some(f) = self.dict.get(*fid) else { continue };
                if f.anchor != x { continue; }
                let Some(bits) = t.bits_of(f) else { continue };
                let moved = t.descend(bits, false);
                if moved != bits { updates.push((*fid, t.rels(moved))); }
            }
            // seed: what the cover left unwritten
            let open: u64 = (0..t.keys.len()).filter(|&b| !t.nbs_of(b).iter().any(|i| written.contains(i))).fold(0u64, |a, b| a | (1u64 << b));
            let seed = if open == 0 { None } else {
                let s = t.descend(open, true);
                if t.support(s) >= 2.0 { Some(t.rels(s)) } else { None }
            };
            (updates, seed)
        };
        for (fid, rels) in updates {
            if self.dict.find_key(x, &rels).is_some() { continue; }
            let moved = self.dict.draft(x, rels.clone(), 0);
            let gain = self.replacement_gain(&[&moved], &[fid], &[]);
            if gain > 1e-9 {
                if self.trace { eprintln!("[update f{}] {} -> {} gain {:.1}", self.frame_no, self.dict.describe(self.dict.get(fid).unwrap()), self.dict.describe(&moved), gain); }
                self.dict.rewrite(fid, rels);
            }
        }
        // the seed is priced once on its own, and kept where its margin is positive
        seed.into_iter().filter(|r| {
            if self.dict.find_key(x, r).is_some() { return true; }
            let f = self.dict.draft(x, r.clone(), 0);
            let g = self.replacement_gain(&[&f], &[], &[]);
            if self.trace { eprintln!("[seed f{}] {} margin {:.2}", self.frame_no, self.dict.describe(&f), g); }
            g > 1e-9
        }).collect()
    }

    // ───────────────────────────── finding (R53) and generalizing (R52) ─────────────────────────────

    /// Pair two functions' relations dimension by dimension and offset by offset; None where they do not
    /// have the same offsets at every relation.
    fn pair(a: &[Rel], b: &[Rel]) -> Option<Vec<(Rel, Rel)>> {
        if a.len() != b.len() { return None; }
        let mut ka: Vec<(u8, i32)> = a.iter().map(|r| (r.dim, r.off)).collect();
        let mut kb: Vec<(u8, i32)> = b.iter().map(|r| (r.dim, r.off)).collect();
        ka.sort_unstable();
        kb.sort_unstable();
        if ka != kb { return None; }
        let mut out = Vec::new();
        let mut used = vec![false; b.len()];
        for ra in a {
            // the same node first, then any at the same offset
            let j = (0..b.len()).find(|&j| !used[j] && b[j].dim == ra.dim && b[j].off == ra.off && b[j].node == ra.node)
                .or_else(|| (0..b.len()).find(|&j| !used[j] && b[j].dim == ra.dim && b[j].off == ra.off))?;
            used[j] = true;
            out.push((*ra, b[j]));
        }
        Some(out)
    }

    /// Whether a function of fixed nodes is fit by a function of the dictionary through its variables, so it is
    /// not new (R52, R53).
    fn fit_through_variables(&self, x: NeuronId, rels: &[Rel]) -> Option<FuncId> {
        for &id in self.dict.anchored(x) {
            let Some(f) = self.dict.get(id) else { continue };
            if f.rels.iter().all(|r| r.node.is_fixed()) { continue; }
            let Some(pairs) = Self::pair(rels, &f.rels) else { continue };
            let mut values: FxHashMap<u8, NeuronId> = FxHashMap::default();
            let ok = pairs.iter().all(|(g, d)| {
                let Node::Fixed(n) = g.node else { return false };
                match d.node {
                    Node::Fixed(m) => m == n,
                    Node::Class(k) => self.dict.class(k).contains(&n),
                    Node::Param(p) => *values.entry(p).or_insert(n) == n,
                }
            });
            if ok { return Some(id); }
        }
        None
    }

    /// Find each kept function in the dictionary (R53): the function itself, one a dictionary function fits
    /// through its variables, or new; a new one joins the dictionary and is generalized at once with every
    /// function that differs from it only at some nodes (R52). Returns the functions added.
    pub(crate) fn find(&mut self, x: NeuronId, kept: Vec<Vec<Rel>>) -> usize {
        let mut added = 0;
        for rels in kept {
            if self.dict.find_key(x, &rels).is_some() { continue; }
            if self.fit_through_variables(x, &rels).is_some() { continue; }
            let id = self.dict.add(self.dict.draft(x, rels, 0));
            added += 1;
            if self.trace { eprintln!("[new f{}] {}", self.frame_no, self.dict.describe(self.dict.get(id).unwrap())); }
            added += self.generalize(id);
        }
        added
    }

    /// Generalize a new function with each function of the dictionary that differs from it only in which
    /// neuron stands at some nodes (R52): fixed where they agree, a parameter where two or more nodes hold one
    /// neuron within each and differ across, a class otherwise; priced as a replacement of the two, and where it
    /// is cheaper it joins and the two leave. Returns the functions added.
    fn generalize(&mut self, gid: FuncId) -> usize {
        let Some(g) = self.dict.get(gid).cloned() else { return 0 };
        let partners: Vec<FuncId> = self.dict.anchored(g.anchor).iter().copied().filter(|id| *id != gid).collect();
        for fid in partners {
            let Some(f) = self.dict.get(fid).cloned() else { continue };
            let Some(pairs) = Self::pair(&g.rels, &f.rels) else { continue };
            // the nodes where they differ, as (g's neuron, f's node)
            let differ: Vec<(Rel, Rel)> = pairs.iter().copied().filter(|(a, b)| a.node != b.node).collect();
            if differ.is_empty() { continue; }
            let mut rels: Vec<Rel> = pairs.iter().filter(|(a, b)| a.node == b.node).map(|(a, _)| *a).collect();
            let mut params: Vec<(Node, Node)> = Vec::new();
            let mut new_classes: Vec<Vec<NeuronId>> = Vec::new();
            for (a, b) in &differ {
                let same = differ.iter().filter(|(c, d)| c.node == a.node && d.node == b.node).count();
                if same >= 2 {
                    let p = match params.iter().position(|x| *x == (a.node, b.node)) { Some(p) => p, None => { params.push((a.node, b.node)); params.len() - 1 } };
                    rels.push(Rel { dim: a.dim, off: a.off, node: Node::Param(p as u8) });
                    continue;
                }
                let mut members: Vec<NeuronId> = Vec::new();
                for node in [a.node, b.node] {
                    match node {
                        Node::Fixed(n) => members.push(n),
                        Node::Class(k) => members.extend_from_slice(self.dict.class(k)),
                        Node::Param(_) => {}
                    }
                }
                members.sort_unstable();
                members.dedup();
                if members.len() < 2 { continue; }
                if !self.dict.class_exists(&members) && !new_classes.contains(&members) { new_classes.push(members.clone()); }
                let k = self.dict.class_for(members);
                rels.push(Rel { dim: a.dim, off: a.off, node: Node::Class(k) });
            }
            if rels.len() != g.rels.len() { self.dict.drop_dead_classes(); continue; }
            let mut gen = self.dict.draft(g.anchor, rels, params.len());
            for (p, (a, b)) in params.iter().enumerate() {
                for node in [a, b] { if let Node::Fixed(n) = node { gen.param_members[p].push(*n); } }
            }
            if self.dict.find_key(gen.anchor, &gen.rels).is_some() { self.dict.drop_dead_classes(); continue; }
            let gain = self.replacement_gain(&[&gen], &[gid, fid], &new_classes);
            if gain > 1e-9 {
                if self.trace { eprintln!("[generalize f{}] {} + {} -> {} gain {:.1}", self.frame_no, self.dict.describe(&g), self.dict.describe(&f), self.dict.describe(&gen), gain); }
                self.dict.add(gen);
                self.retire_func(gid);
                self.retire_func(fid);
                self.dict.drop_dead_classes();
                return 1;
            }
            self.dict.drop_dead_classes();
        }
        0
    }

    // ───────────────────────────── prune (R18) ─────────────────────────────

    /// Price every function whose evidence changed at this level, and drop them one at a time, the lowest
    /// margin first while it is negative, repricing the rest between; on a tie the later to join leaves (R18).
    pub(crate) fn prune(&mut self, anchors: &FxHashSet<NeuronId>) -> usize {
        let mut retired = 0;
        // every history the functions in question are read over: each neuron that can stand in them (D30)
        let mut hist: Vec<NeuronId> = Vec::new();
        for a in anchors {
            hist.push(*a);
            for id in self.dict.standing(*a) {
                if let Some(f) = self.dict.get(*id) { for n in f.fixed_neurons() { if !f.nodes_for(n).is_empty() { hist.push(n); } } }
            }
        }
        hist.sort_unstable();
        hist.dedup();
        let mut margins: FxHashMap<NeuronId, FxHashMap<FuncId, f64>> = hist.iter().map(|a| (*a, self.margins_of(*a))).collect();
        let mut candidates: Vec<FuncId> = anchors.iter().flat_map(|a| self.dict.standing(*a).to_vec()).collect();
        candidates.sort_unstable();
        candidates.dedup();
        loop {
            // a function is kept while its margin is positive over at least one history it is read over
            let mut worst: Option<(FuncId, f64, u64)> = None;
            for &id in &candidates {
                let Some(f) = self.dict.get(id) else { continue };
                let m = hist.iter().filter_map(|h| margins.get(h).and_then(|ms| ms.get(&id))).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                if !m.is_finite() || m > -1e-9 { continue; }
                let take = match worst { None => true, Some((_, wm, wj)) => m < wm || (m == wm && f.joined > wj) };
                if take { worst = Some((id, m, f.joined)); }
            }
            let Some((id, m, _)) = worst else { break };
            if self.trace { eprintln!("[prune f{}] {} margin {:.1}", self.frame_no, self.dict.describe(self.dict.get(id).unwrap()), m); }
            let touched: Vec<NeuronId> = self.dict.get(id).map(|f| f.fixed_neurons()).unwrap_or_default();
            self.retire_func(id);
            // the rest are priced again before the next leaves (R18)
            for h in &hist { if touched.contains(h) { margins.insert(*h, self.margins_of(*h)); } }
            retired += 1;
        }
        self.dict.drop_dead_classes();
        retired
    }

    /// A function leaves the dictionary (R18, R52, R38): the ledger keeps it readable while what it fired is
    /// open, and its interneuron, left with no function, is decommissioned when due.
    pub(crate) fn retire_func(&mut self, id: FuncId) {
        let Some(f) = self.dict.remove(id) else { return };
        self.stats_retired += 1;
        if let Some(n) = f.inter {
            if self.dict.of_inter(n).is_empty() && !self.ledger.contains(&n) { self.ledger.push(n); }
        }
        let until = self.frame_no + self.max_reach_t as i64 + 1;
        self.retired_funcs.insert(id, (f, until));
    }

    /// A function as expansion reads it: the dictionary's, or the ledger's copy (R38).
    pub(crate) fn func(&self, id: FuncId) -> Option<&Func> {
        self.dict.get(id).or_else(|| self.retired_funcs.get(&id).map(|x| &x.0))
    }
}

/// One row of the history as grouping reads it (R50): which of the relations it holds, what its activation
/// costs and what the dictionary's cover left of it, and per relation it holds the neighbor that answers it,
/// with that neighbor's cost and what the cover left of it.
struct RowInfo {
    mask: u64,
    own: f64,
    own_left: f64,
    ans: Vec<(u8, u16, f64, f64)>,
}

/// One activation's relations against its neuron's history, as grouping reads them (R50): each relation held
/// in two or more rows is a bit, and each row of the history that holds any of them is read as a point.
struct Table<'a> {
    m: &'a Machine,
    x: NeuronId,
    row: &'a Row,
    keys: Vec<RelKey>,
    rows: Vec<RowInfo>,
}

impl<'a> Table<'a> {
    /// `now` is what the cover left of the new row; every other row reads what its cover left when it was last
    /// priced, or itself where nothing covered it.
    fn new(m: &'a Machine, x: NeuronId, ret: &ActReturn, now: Option<&Cover>) -> Option<Self> {
        let n = &m.thalamus[x];
        let row = n.row(ret.row)?;
        // a relation that reaches this activation's neighbor only as a base symbol under a call holds a node
        // and can cover nothing (D47, D22)
        let mut worth: Vec<RelKey> = Vec::new();
        for nb in row.nbs.iter().filter(|nb| !matches!(nb.via, Via::Under(_))) {
            let shared = (row.axes & nb.axes) | 1;
            for d in 0..m.radii.len().min(MAX_DIMS) {
                if shared & (1 << d) != 0 { worth.push((nb.neuron, d as u8, crate::offsets::round_offset(nb.delta[d], m.radii[d]))); }
            }
        }
        let mut rels: Vec<(RelKey, &Vec<u64>)> = ret.rels.iter().filter(|(k, rows)| rows.len() >= 2 && worth.contains(k)).map(|(k, r)| (*k, r)).collect();
        if rels.is_empty() { return None; }
        // at most 64, the most held first
        rels.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        rels.truncate(64);
        let mut held: FxHashMap<u64, u64> = FxHashMap::default();
        for (b, (_, rows)) in rels.iter().enumerate() { for r in rows.iter() { *held.entry(*r).or_insert(0) |= 1u64 << b; } }
        let keys: Vec<RelKey> = rels.into_iter().map(|(k, _)| k).collect();
        let bit: FxHashMap<RelKey, u8> = keys.iter().enumerate().map(|(b, k)| (*k, b as u8)).collect();
        let assigned = m.assigned.borrow();
        let left_of = assigned.get(&x);
        let mut ids: Vec<u64> = held.keys().copied().collect();
        ids.sort_unstable();
        let mut rows: Vec<RowInfo> = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(r) = n.row(id) else { continue };
            let mask = held[&id];
            let (own_left, nb_left): (f64, Vec<f64>) = if id == ret.row && now.is_some() {
                let c = now.unwrap();
                (c.anchor, c.nbs.clone())
            } else {
                left_of.and_then(|a| a.get(&id).cloned()).unwrap_or_else(|| (r.own_cost, r.nbs.iter().map(|nb| nb.cost).collect()))
            };
            let mut ans = Vec::new();
            for (i, nb) in r.nbs.iter().enumerate() {
                if matches!(nb.via, Via::Under(_)) { continue; }
                let shared = (r.axes & nb.axes) | 1;
                for d in 0..m.radii.len().min(MAX_DIMS) {
                    if shared & (1 << d) == 0 { continue; }
                    let Some(&b) = bit.get(&(nb.neuron, d as u8, crate::offsets::round_offset(nb.delta[d], m.radii[d]))) else { continue };
                    if mask >> b & 1 == 1 { ans.push((b, i as u16, nb.cost, nb_left.get(i).copied().unwrap_or(nb.cost))); }
                }
            }
            rows.push(RowInfo { mask, own: r.own_cost, own_left, ans });
        }
        drop(assigned);
        Some(Table { m, x, row, keys, rows })
    }

    fn each(&self, bits: u64) -> impl Iterator<Item = usize> + '_ { (0..self.keys.len()).filter(move |b| bits >> b & 1 == 1) }

    fn rels(&self, bits: u64) -> Vec<Rel> {
        self.each(bits).map(|b| { let (n, d, o) = self.keys[b]; Rel { dim: d, off: o, node: Node::Fixed(n) } }).collect()
    }

    /// A function anchored here as bits, where the row holds every relation of it.
    fn bits_of(&self, f: &Func) -> Option<u64> {
        let mut bits = 0u64;
        for r in &f.rels {
            let Node::Fixed(n) = r.node else { return None };
            let b = self.keys.iter().position(|k| *k == (n, r.dim, r.off))?;
            bits |= 1u64 << b;
        }
        Some(bits)
    }

    /// The neighbors of the new row that answer a relation.
    fn nbs_of(&self, b: usize) -> Vec<usize> {
        let (n, d, o) = self.keys[b];
        let d = d as usize;
        (0..self.row.nbs.len()).filter(|&i| {
            let nb = &self.row.nbs[i];
            nb.neuron == n && !matches!(nb.via, Via::Under(_)) && (d == 0 || nb.axes & (1 << d) != 0) && crate::offsets::round_offset(nb.delta[d], self.m.radii[d]) == o
        }).collect()
    }

    /// What the call of the function of these relations writes on the new row (D22): its price there.
    fn price(&self, bits: u64) -> f64 {
        let f = self.m.dict.draft(self.x, self.rels(bits), 0);
        let fresh = Fresh { own: self.row.own_cost, nbs: &self.row.nbs, calls: &self.row.calls };
        land(&f, &self.m.dict, &self.m.radii, self.row.axes, 0, &self.row.nbs, &self.row.calls, &fresh).price
    }

    /// How many rows hold every relation.
    fn support(&self, bits: u64) -> f64 { self.rows.iter().filter(|r| r.mask & bits == bits).count() as f64 }

    /// What writing the history with the function saves, read off the rows (D30): in each row that holds any of
    /// its relations, the activation and each neighbor answering one, as they cost or, `left`, as the cover
    /// left them, less its price and one number per relation that did not come true; nothing where that is not
    /// positive; less the function's entry.
    fn value(&self, bits: u64, price: f64, left: bool) -> f64 {
        if !price.is_finite() { return f64::NEG_INFINITY; }
        let mut v = 0.0;
        let mut seen: Vec<u16> = Vec::new();
        for r in &self.rows {
            let holds = bits & r.mask;
            if holds == 0 { continue; }
            let mut cov = if left { r.own_left } else { r.own };
            seen.clear();
            for &(b, i, cost, rest) in &r.ans {
                if holds >> b & 1 == 0 || seen.contains(&i) { continue; }
                seen.push(i);
                cov += if left { rest } else { cost };
            }
            let s = cov - price - (bits & !r.mask).count_ones() as f64;
            if s > 0.0 { v += s; }
        }
        let f = self.m.dict.draft(self.x, self.rels(bits), 0);
        v - self.m.dict.entry(&f, &[])
    }

    /// The update (R50): from `start`, add or drop the one relation that most raises the value, while one does.
    fn descend(&self, start: u64, left: bool) -> u64 {
        let mut cur = start;
        let mut val = self.value(cur, self.price(cur), left);
        for _ in 0..(2 * self.keys.len()) {
            let mut best: Option<(u64, f64)> = None;
            for b in 0..self.keys.len() {
                let next = cur ^ (1u64 << b);
                if next == 0 { continue; }
                let v = self.value(next, self.price(next), left);
                if v > val + 1e-9 && best.map_or(true, |(_, bv)| v > bv) { best = Some((next, v)); }
            }
            let Some((next, v)) = best else { break };
            cur = next;
            val = v;
        }
        cur
    }
}
