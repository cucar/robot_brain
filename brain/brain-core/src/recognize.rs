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

    /// The dictionary's functions anchored at a neuron, landed on one kind of its rows (D54): those that fit
    /// it, each with when it joined. Kept per kind of row while the dictionary stands.
    pub(crate) fn landed_on(&self, x: NeuronId, row: &Row, sig: u64) -> std::rc::Rc<Vec<(Landing, u64)>> {
        let version = self.dict.version;
        {
            let mut cache = self.landings.borrow_mut();
            let e = cache.entry(x).or_insert_with(|| (version, FxHashMap::default()));
            if e.0 != version { *e = (version, FxHashMap::default()); }
            if let Some(v) = e.1.get(&sig) { return v.clone(); }
        }
        let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
        let v: Vec<(Landing, u64)> = self.dict.standing(x).iter().filter_map(|id| self.dict.get(*id))
            .filter_map(|f| land_best(f, x, &self.dict, &self.radii, row.axes, &row.nbs, &row.calls, &fresh).map(|l| (l, f.joined)))
            .filter(|(l, _)| l.savings() > 0.0).collect();
        let v = std::rc::Rc::new(v);
        self.landings.borrow_mut().get_mut(&x).unwrap().1.insert(sig, v.clone());
        v
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
        for (row, w) in by_sig(n.history.iter()) {
            let ls = fresh_of(row);
            if ls.is_empty() { continue; }
            let all: Vec<(&Landing, u64)> = ls.iter().map(|(_, l, j)| (l, *j)).collect();
            let with = greedy(row, &all);
            for k in 0..ls.len() {
                let rest: Vec<(&Landing, u64)> = ls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, (_, l, j))| (l, *j)).collect();
                *m.get_mut(&ls[k].0).unwrap() += w * (greedy(row, &rest) - with);
            }
        }
        m
    }

    // ───────────────────────────── grouping (R50) ─────────────────────────────

    /// Group the relations an activation's neuron returned in two or more rows into functions of fixed
    /// nodes, merging while a merge lowers the cost of writing its history, and keep each whose margin is
    /// positive (R50).
    pub(crate) fn group(&self, x: NeuronId, ret: &ActReturn) -> Vec<Vec<Rel>> {
        let n = &self.thalamus[x];
        let radii = &self.radii;
        // a relation that reaches this activation's neighbor only as a base symbol under a call holds a node
        // and can cover nothing (D47, D22), so no merge with it lowers a cost
        let Some(me) = n.row(ret.row) else { return Vec::new() };
        let mut worth: Vec<RelKey> = Vec::new();
        for nb in me.nbs.iter().filter(|nb| !matches!(nb.via, Via::Under(_))) {
            let shared = (me.axes & nb.axes) | 1;
            for d in 0..radii.len().min(MAX_DIMS) {
                if shared & (1 << d) != 0 { worth.push((nb.neuron, d as u8, crate::offsets::round_offset(nb.delta[d], radii[d]))); }
            }
        }
        let groups: Vec<(Vec<Rel>, Vec<u64>)> = ret.rels.iter().filter(|(k, rows)| rows.len() >= 2 && worth.contains(k))
            .map(|((m, d, o), rows)| (vec![Rel { dim: *d, off: *o, node: Node::Fixed(*m) }], rows.clone())).collect();
        if groups.is_empty() { return Vec::new(); }
        let draft = |rels: &Vec<Rel>| self.dict.draft(x, rels.clone(), 0);
        // rows by signature: two rows alike cost the same, so each kind is read once (D30)
        let mut all_rows: Vec<u64> = groups.iter().flat_map(|(_, r)| r.iter().copied()).collect();
        all_rows.sort_unstable();
        all_rows.dedup();
        let mut sig_of: FxHashMap<u64, u64> = FxHashMap::default();
        let mut rep: FxHashMap<u64, u64> = FxHashMap::default();
        for id in &all_rows {
            if let Some(row) = n.row(*id) { let s = row_sig(row); sig_of.insert(*id, s); rep.entry(s).or_insert(*id); }
        }
        // per kind of row, the dictionary's functions that fit it, landed once; and each group as it lands there,
        // landed once (D28: a function's price is fixed by what fired)
        let mut fitting: FxHashMap<u64, std::rc::Rc<Vec<(Landing, u64)>>> = FxHashMap::default();
        for (s, id) in &rep {
            let Some(row) = n.row(*id) else { continue };
            fitting.insert(*s, self.landed_on(x, row, *s));
        }
        let group_land: std::cell::RefCell<FxHashMap<(u64, Vec<Rel>), Option<Landing>>> = std::cell::RefCell::new(FxHashMap::default());
        let cost_sig = |s: u64, gs: &[&Func]| -> f64 {
            let Some(row) = rep.get(&s).and_then(|id| n.row(*id)) else { return 0.0 };
            let mut owned: Vec<Landing> = Vec::new();
            for g in gs {
                let mut cache = group_land.borrow_mut();
                let l = cache.entry((s, g.rels.clone())).or_insert_with(|| {
                    let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
                    land_best(g, x, &self.dict, radii, row.axes, &row.nbs, &row.calls, &fresh).filter(|l| l.savings() > 0.0)
                });
                if let Some(l) = l { owned.push(l.clone()); }
            }
            let mut all: Vec<(&Landing, u64)> = fitting.get(&s).map(|v| v.iter().map(|(l, j)| (l, *j)).collect()).unwrap_or_default();
            all.extend(owned.iter().map(|l| (l, u64::MAX)));
            greedy(row, &all)
        };
        // merge, repeatedly, the two groups whose merge most lowers the cost; each pair is priced once and
        // priced again only when it comes to the top, as a merge elsewhere may have moved it
        let mut slots: Vec<Option<(Vec<Rel>, Vec<u64>, Func)>> = groups.into_iter().map(|(r, rows)| { let f = draft(&r); Some((r, rows, f)) }).collect();
        let delta_of = |slots: &Vec<Option<(Vec<Rel>, Vec<u64>, Func)>>, i: usize, j: usize, baseline: &mut FxHashMap<u64, f64>| -> f64 {
            let (Some((ri, rowi, _)), Some((rj, rowj, _))) = (&slots[i], &slots[j]) else { return 0.0 };
            // a merge gains only in the rows that hold both: elsewhere it writes what one wrote, less the other's
            // relations that did not come true
            let (mut a, mut b, mut both) = (0, 0, false);
            while a < rowi.len() && b < rowj.len() {
                if rowi[a] == rowj[b] { both = true; break; }
                if rowi[a] < rowj[b] { a += 1; } else { b += 1; }
            }
            if !both { return 0.0; }
            let alive: Vec<&Func> = slots.iter().flatten().map(|(_, _, f)| f).collect();
            let mut rels = ri.clone();
            rels.extend(rj.iter().copied());
            let merged_f = draft(&canonical(rels));
            let mut merged: Vec<&Func> = slots.iter().enumerate().filter(|(k, _)| *k != i && *k != j).filter_map(|(_, s)| s.as_ref().map(|(_, _, f)| f)).collect();
            merged.push(&merged_f);
            let mut kinds: FxHashMap<u64, f64> = FxHashMap::default();
            for id in rowi.iter().chain(rowj.iter()) { if let Some(s) = sig_of.get(id) { *kinds.entry(*s).or_insert(0.0) += 1.0; } }
            // a row on both lists was counted twice
            for id in rowi.iter().filter(|r| rowj.contains(r)) { if let Some(s) = sig_of.get(id) { *kinds.get_mut(s).unwrap() -= 1.0; } }
            let mut delta = 0.0;
            for (&s, &w) in &kinds {
                if w <= 0.0 { continue; }
                let before = *baseline.entry(s).or_insert_with(|| cost_sig(s, &alive));
                delta += w * (cost_sig(s, &merged) - before);
            }
            delta
        };
        #[derive(PartialEq)]
        struct Cand(f64, usize, usize);
        impl Eq for Cand {}
        impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
        impl Ord for Cand { fn cmp(&self, o: &Self) -> std::cmp::Ordering { o.0.partial_cmp(&self.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| (o.1, o.2).cmp(&(self.1, self.2))) } }
        let mut baseline: FxHashMap<u64, f64> = FxHashMap::default();
        let mut heap: std::collections::BinaryHeap<Cand> = std::collections::BinaryHeap::new();
        for i in 0..slots.len() { for j in (i + 1)..slots.len() {
            let d = delta_of(&slots, i, j, &mut baseline);
            if d < -1e-9 { heap.push(Cand(d, i, j)); }
        } }
        while let Some(Cand(d, i, j)) = heap.pop() {
            if slots[i].is_none() || slots[j].is_none() { continue; }
            let now = delta_of(&slots, i, j, &mut baseline);
            if !(now < -1e-9) { continue; }
            if now > d + 1e-9 { if let Some(top) = heap.peek() { if top.0 < now { heap.push(Cand(now, i, j)); continue; } } }
            let (ri, rowi, _) = slots[i].take().unwrap();
            let (rj, rowj, _) = slots[j].take().unwrap();
            let rels = canonical(ri.into_iter().chain(rj).collect());
            let rows: Vec<u64> = rowi.into_iter().filter(|r| rowj.contains(r)).collect();
            let f = draft(&rels);
            slots.push(Some((rels, rows, f)));
            baseline.clear();
            let k = slots.len() - 1;
            for l in 0..k {
                if slots[l].is_none() { continue; }
                let d = delta_of(&slots, l, k, &mut baseline);
                if d < -1e-9 { heap.push(Cand(d, l, k)); }
            }
        }
        let groups: Vec<(Vec<Rel>, Vec<u64>)> = slots.into_iter().flatten().map(|(r, rows, _)| (r, rows)).collect();
        // each group left is priced once on its own, and kept where its margin is positive
        groups.into_iter().map(|(r, _)| r).filter(|r| {
            if self.dict.find_key(x, r).is_some() { return true; }
            let f = draft(r);
            self.replacement_gain(&[&f], &[], &[]) > 1e-9
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
