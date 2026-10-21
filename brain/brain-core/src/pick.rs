//! The relation index (D47) and the greedy pick (D33): relations of the residual, grown into functions by
//! the collapse and kept on their margin (R14, R15).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::neuron::*;
use crate::types::*;

/// A relation of the index (D34, D47), with every place written from the owner (D15).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Relation {
    /// Node to node, the owner at one end.
    Owner(NeuronId, Offset),
    /// Node to node between two neighbors.
    Pair((NeuronId, Offset), (NeuronId, Offset)),
    /// Same to same: one neuron at both places, whichever.
    Same(Offset, Offset),
    /// Node to open, from the owner: different neurons at one place, `a` and `b` the two that stood there most.
    Open(Offset, NeuronId, NeuronId),
}

struct Tally { rows: Vec<usize>, save: f64, values: FxHashSet<NeuronId> }

impl Neuron {
    // ───────────────────────────── the greedy pick (D33) ─────────────────────────────

    /// Build functions out of the residual (D33): the relations of the index ranked by what each would
    /// save as it stands, taken in that order and grown by the collapse; a relation that builds nothing
    /// changes nothing, so the next in the ranking is the next the rule names, and one that builds changes
    /// the residual, so the index is read again. Every relation counted in two rows is tried. Returns how
    /// many functions were added.
    pub(crate) fn create(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        let mut tried: FxHashSet<Relation> = FxHashSet::default();
        loop {
            let ranked = self.relation_index(&tried);
            let mut built = false;
            for (rel, rows) in ranked {
                tried.insert(rel.clone());
                if self.build(&rel, &rows, ctx) { added += 1; built = true; break; }
            }
            if !built { break; }
        }
        added
    }

    /// The relation index (D47) over the residual of the history, ranked (D33): most saved as it stands,
    /// D22's savings of the relation taken as a whole function summed over the rows that hold it, then the
    /// higher count, the older neuron and the shorter offset. Each comes with the rows that hold it.
    pub(crate) fn relation_index(&self, tried: &FxHashSet<Relation>) -> Vec<(Relation, Vec<usize>)> {
        let wf = log2((self.functions.len() + 1) as f64);
        let mut index: FxHashMap<Relation, Tally> = FxHashMap::default();
        let mut open: FxHashMap<Offset, FxHashMap<NeuronId, (Vec<usize>, f64)>> = FxHashMap::default();
        for (i, row) in self.history.iter().enumerate() {
            // the owner's activation saves its cost where nothing was credited it first (D20)
            let own = if row.owner_self.is_none() { row.own_cost } else { 0.0 };
            let base = own - (row.own_cost + wf);
            // the residual read in nodes: each neighbor, and each base symbol a child stands in for (D47)
            let mut nodes: Vec<(NeuronId, Offset, f64, bool)> = Vec::new();
            for (j, nb) in row.nbs.iter().enumerate() {
                if !row.residual(j) { continue; }
                nodes.push((nb.neuron, self.round(&nb.delta), nb.cost, true));
                for (m, d, c) in &nb.covers { nodes.push((*m, self.round(d), *c, false)); }
            }
            let mut seen: FxHashSet<Relation> = FxHashSet::default();
            let mut add = |rel: Relation, save: f64, value: Option<NeuronId>, index: &mut FxHashMap<Relation, Tally>| {
                if !seen.insert(rel.clone()) { return; }
                let t = index.entry(rel).or_insert(Tally { rows: Vec::new(), save: 0.0, values: FxHashSet::default() });
                t.rows.push(i);
                t.save += save;
                if let Some(v) = value { t.values.insert(v); }
            };
            for a in 0..nodes.len() {
                let (na, oa, ca, da) = nodes[a];
                add(Relation::Owner(na, oa), base + ca, None, &mut index);
                if da {
                    let e = open.entry(oa).or_default().entry(na).or_insert((Vec::new(), 0.0));
                    if e.0.last() != Some(&i) { e.0.push(i); e.1 += ca; }
                }
                for b in (a + 1)..nodes.len() {
                    let (nb, ob, cb, db) = nodes[b];
                    if oa == ob && na == nb { continue; }
                    let (x, y) = if (oa, na) < (ob, nb) { ((na, oa), (nb, ob)) } else { ((nb, ob), (na, oa)) };
                    add(Relation::Pair(x, y), base + ca + cb, None, &mut index);
                    if da && db && na == nb && oa != ob {
                        let (p, q) = if oa < ob { (oa, ob) } else { (ob, oa) };
                        add(Relation::Same(p, q), base + ca + cb, Some(na), &mut index);
                    }
                }
            }
        }
        // node to open: the two neurons that stood at the place most
        for (off, by) in open {
            let mut order: Vec<(NeuronId, usize, f64)> = by.iter().map(|(n, (r, c))| (*n, r.len(), *c)).collect();
            if order.len() < 2 { continue; }
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let (a, b) = (order[0].0, order[1].0);
            let mut rows: Vec<usize> = by[&a].0.iter().chain(by[&b].0.iter()).copied().collect();
            rows.sort();
            rows.dedup();
            let save: f64 = rows.iter().map(|&i| {
                let row = &self.history[i];
                let own = if row.owner_self.is_none() { row.own_cost } else { 0.0 };
                let cost = row.nbs.iter().enumerate().filter(|(j, nb)| row.residual(*j) && self.round(&nb.delta) == off && (nb.neuron == a || nb.neuron == b)).map(|(_, nb)| nb.cost).fold(0.0, f64::max);
                own + cost - (row.own_cost + wf + 1.0)
            }).sum();
            let (x, y) = if a < b { (a, b) } else { (b, a) };
            index.insert(Relation::Open(off, x, y), Tally { rows, save, values: FxHashSet::default() });
        }
        let mut ranked: Vec<((f64, usize, NeuronId, i32), Relation, Vec<usize>)> = Vec::new();
        for (rel, t) in index {
            if t.rows.len() < 2 || tried.contains(&rel) { continue; }
            let mut save = t.save;
            if let Relation::Same(..) = rel { save -= t.rows.len() as f64 * log2(t.values.len() as f64); }
            let len = |o: &Offset| o.iter().map(|x| x.abs()).sum::<i32>();
            let (oldest, short) = match &rel {
                Relation::Owner(n, o) => (self.id.min(*n), len(o)),
                Relation::Pair((a, oa), (b, ob)) => (self.id.min(*a).min(*b), len(oa) + len(ob)),
                Relation::Same(p, q) => (self.id, len(p) + len(q)),
                Relation::Open(o, a, b) => (self.id.min(*a).min(*b), len(o)),
            };
            ranked.push(((save, t.rows.len(), oldest, short), rel, t.rows));
        }
        ranked.sort_by(|x, y| (y.0).0.partial_cmp(&(x.0).0).unwrap_or(std::cmp::Ordering::Equal)
            .then((y.0).1.cmp(&(x.0).1)).then((x.0).2.cmp(&(y.0).2)).then((x.0).3.cmp(&(y.0).3)).then(x.1.cmp(&y.1)));
        ranked.into_iter().map(|(_, r, rows)| (r, rows)).collect()
    }

    /// Grow one relation into a function by the collapse over the rows that hold it (D27), and keep it
    /// where its margin is positive (R14, R15): over each row, what the cover saves with it in it less what
    /// the cover as it stands saves, where that is positive, beats its entry in the table and the entries
    /// of the classes it made. A kept function joins the covers it was priced on.
    pub(crate) fn build(&mut self, rel: &Relation, rows: &[usize], ctx: Ctx) -> bool {
        let classes_before: Vec<PatternId> = self.classes.iter().map(|c| c.id).collect();
        let (places, params) = match rel {
            Relation::Owner(n, o) => (vec![(Place::Fixed(*n), *o)], Vec::new()),
            Relation::Pair((a, oa), (b, ob)) => (vec![(Place::Fixed(*a), *oa), (Place::Fixed(*b), *ob)], Vec::new()),
            Relation::Same(p, q) => (vec![(Place::Param(0), *p), (Place::Param(0), *q)], vec![Param::default()]),
            Relation::Open(o, a, b) => {
                let members = { let mut v = vec![*a, *b]; v.sort(); v };
                let existing = self.classes.iter().find(|c| { let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect(); m.sort(); m == members }).map(|c| c.id);
                let k = match existing {
                    Some(k) => k,
                    None => { let id = self.new_pid(); self.classes.push(Class { id, members: members.iter().map(|n| Member { neuron: *n, count: 0 }).collect() }); id }
                };
                (vec![(Place::Class(k), *o)], Vec::new())
            }
        };
        let fid = self.new_pid();
        self.functions.push(Function::new(fid, places, params));
        self.collapse(fid, rows, ctx, true);
        let ok = self.function(fid).map_or(false, |f| f.p_size() > 0) && self.identical(fid).is_none();
        let fail = |me: &mut Neuron| {
            me.functions.retain(|f| f.id != fid);
            me.classes.retain(|c| classes_before.contains(&c.id) || me.functions.iter().any(|f| f.uses_class(c.id)));
            false
        };
        if !ok { return fail(self); }
        // price it (R14, R15)
        let f = self.function(fid).unwrap().clone();
        let wf = self.which_function_bits();
        let mut benefit = 0.0;
        let mut joins: Vec<(usize, Vec<CoverEntry>)> = Vec::new();
        for i in 0..self.history.len() {
            // the row's cover as it stands with the candidate in it, owning only the residual it names (R14)
            let row = &self.history[i];
            let taken: Vec<bool> = row.owner.iter().map(|o| o.is_some()).collect();
            let Some(c) = self.measure(row, &f, &taken, row.owner_self.is_some(), wf) else { continue };
            if !(c.coverage > c.price) { continue; }
            let mut with = row.cover.clone();
            with.push(c.entry);
            let diff = self.savings_of_cover(i, &with) - self.savings_of_cover(i, &row.cover);
            if diff > 0.0 { benefit += diff; joins.push((i, with)); }
        }
        let made: f64 = self.classes.iter().filter(|c| !classes_before.contains(&c.id) && f.uses_class(c.id)).map(|c| self.class_entry(c, ctx)).sum();
        if benefit <= self.entry(&f, ctx) + made { return fail(self); }
        for (i, cover) in joins { self.install_cover(i, cover); }
        // classes the collapse made and the function does not use go
        self.classes.retain(|c| classes_before.contains(&c.id) || self.functions.iter().any(|g| g.uses_class(c.id)));
        true
    }
}
