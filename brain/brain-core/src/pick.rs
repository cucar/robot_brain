//! Seeds (D33) from the relation index (D47), grown into functions by the collapse and kept on their margin
//! (R14, R15).

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

impl Neuron {
    // ───────────────────────────── create (§5.4) ─────────────────────────────

    /// Try every seed once, best first (D33): grow it by the collapse over the rows that hold it, and keep it
    /// where its margin is positive (R15); a kept one is in the table for every seed after it. Returns how
    /// many functions were added.
    pub(crate) fn create(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        for (rel, rows) in self.seeds(ctx) {
            if self.writes(&rel) { continue; }
            if self.build(&rel, &rows, ctx) { added += 1; }
        }
        added
    }

    /// Whether a function of the table writes a relation between the same nodes at the same offset (D33).
    fn writes(&self, rel: &Relation) -> bool {
        self.functions.iter().any(|f| {
            let fixed_at = |n: NeuronId, o: &Offset| f.places.iter().any(|(p, q)| *p == Place::Fixed(n) && q == o);
            match rel {
                Relation::Owner(n, o) => fixed_at(*n, o),
                Relation::Pair((a, oa), (b, ob)) => fixed_at(*a, oa) && fixed_at(*b, ob),
                Relation::Same(p, q) => (0..f.params.len() as u32).any(|k| { let offs: Vec<Offset> = f.param_places(k).iter().map(|&pi| f.places[pi].1).collect(); offs.contains(p) && offs.contains(q) }),
                Relation::Open(o, _, _) => f.places.iter().any(|(p, q)| matches!(p, Place::Class(_)) && q == o),
            }
        })
    }

    /// The seeds (D33): every relation of the index (D47), over whole rows, counted in two or more rows, ranked
    /// by D22's savings of the relation taken as a whole function summed over the rows that hold it, then the
    /// higher count, the older neuron and the shorter offset. Each comes with the rows that hold it.
    pub(crate) fn seeds(&self, ctx: Ctx) -> Vec<(Relation, Vec<usize>)> {
        let mut index: FxHashMap<Relation, Vec<usize>> = FxHashMap::default();
        let mut open: FxHashMap<Offset, FxHashMap<NeuronId, Vec<usize>>> = FxHashMap::default();
        for (i, row) in self.history.iter().enumerate() {
            // the row read in nodes: each neighbor as itself, and each base symbol a child stands for (D47)
            let mut nodes: Vec<(NeuronId, Offset, bool)> = Vec::new();
            for nb in &row.nbs {
                nodes.push((nb.neuron, self.round(&nb.delta), true));
                for (m, d) in &nb.covers { nodes.push((*m, self.round(d), false)); }
            }
            nodes.sort();
            nodes.dedup();
            let mut seen: FxHashSet<Relation> = FxHashSet::default();
            let mut add = |rel: Relation, index: &mut FxHashMap<Relation, Vec<usize>>| { if seen.insert(rel.clone()) { index.entry(rel).or_default().push(i); } };
            for a in 0..nodes.len() {
                let (na, oa, da) = nodes[a];
                add(Relation::Owner(na, oa), &mut index);
                if da {
                    let e = open.entry(oa).or_default().entry(na).or_default();
                    if e.last() != Some(&i) { e.push(i); }
                }
                for b in (a + 1)..nodes.len() {
                    let (nb, ob, db) = nodes[b];
                    if oa == ob && na == nb { continue; }
                    let (x, y) = if (oa, na) < (ob, nb) { ((na, oa), (nb, ob)) } else { ((nb, ob), (na, oa)) };
                    add(Relation::Pair(x, y), &mut index);
                    if da && db && na == nb && oa != ob {
                        let (p, q) = if oa < ob { (oa, ob) } else { (ob, oa) };
                        add(Relation::Same(p, q), &mut index);
                    }
                }
            }
        }
        // node to open: the two neurons that stood at the place most, never both in one row
        for (off, by) in open {
            let mut order: Vec<(NeuronId, usize)> = by.iter().map(|(n, r)| (*n, r.len())).collect();
            if order.len() < 2 { continue; }
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let (a, b) = (order[0].0, order[1].0);
            if by[&a].iter().any(|r| by[&b].contains(r)) { continue; }
            let mut rows: Vec<usize> = by[&a].iter().chain(by[&b].iter()).copied().collect();
            rows.sort();
            rows.dedup();
            let (x, y) = if a < b { (a, b) } else { (b, a) };
            index.insert(Relation::Open(off, x, y), rows);
        }
        let mut ranked: Vec<((f64, usize, NeuronId, i32), Relation, Vec<usize>)> = Vec::new();
        for (rel, rows) in index {
            if rows.len() < 2 { continue; }
            let (f, new_class) = self.seed_function(&rel, PatternId::MAX);
            let save: f64 = rows.iter().map(|&i| {
                let row = &self.history[i];
                match (&rel, &new_class) {
                    // a class the table does not hold yet: the owner, and the position of the member that stood
                    (Relation::Open(o, _, _), Some(members)) => {
                        let p = row.nbs.iter().filter(|nb| self.round(&nb.delta) == *o && members.contains(&nb.neuron)).map(|nb| nb.pos).fold(0.0, f64::max);
                        row.own_cost + p - row.rec
                    }
                    _ => self.measure(row, &f, None, ctx).savings(),
                }
            }).sum();
            let len = |o: &Offset| o.iter().map(|x| x.abs()).sum::<i32>();
            let (oldest, short) = match &rel {
                Relation::Owner(n, o) => (self.id.min(*n), len(o)),
                Relation::Pair((a, oa), (b, ob)) => (self.id.min(*a).min(*b), len(oa) + len(ob)),
                Relation::Same(p, q) => (self.id, len(p) + len(q)),
                Relation::Open(o, a, b) => (self.id.min(*a).min(*b), len(o)),
            };
            ranked.push(((save, rows.len(), oldest, short), rel, rows));
        }
        ranked.sort_by(|x, y| (y.0).0.partial_cmp(&(x.0).0).unwrap_or(std::cmp::Ordering::Equal)
            .then((y.0).1.cmp(&(x.0).1)).then((x.0).2.cmp(&(y.0).2)).then((x.0).3.cmp(&(y.0).3)).then(x.1.cmp(&y.1)));
        ranked.into_iter().map(|(_, r, rows)| (r, rows)).collect()
    }

    /// The function a seed is, before the collapse grows it (D33), and the members of the class it would use.
    fn seed_function(&self, rel: &Relation, id: PatternId) -> (Function, Option<Vec<NeuronId>>) {
        match rel {
            Relation::Owner(n, o) => (Function::new(id, vec![(Place::Fixed(*n), *o)], Vec::new()), None),
            Relation::Pair((a, oa), (b, ob)) => (Function::new(id, vec![(Place::Fixed(*a), *oa), (Place::Fixed(*b), *ob)], Vec::new()), None),
            Relation::Same(p, q) => (Function::new(id, vec![(Place::Param(0), *p), (Place::Param(0), *q)], vec![Param::default()]), None),
            Relation::Open(o, a, b) => {
                let members = { let mut v = vec![*a, *b]; v.sort(); v };
                let k = self.classes.iter().find(|c| { let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect(); m.sort(); m == members }).map(|c| c.id);
                (Function::new(id, vec![(Place::Class(k.unwrap_or(PatternId::MAX)), *o)], Vec::new()), if k.is_none() { Some(members) } else { None })
            }
        }
    }

    /// Grow one seed into a candidate by the collapse over the rows that hold it (R14), and keep it where its
    /// benefit, over each row the row's cost without it less its cost with it where that is positive, exceeds
    /// its entry and the entries of the classes it made (R15, D30).
    pub(crate) fn build(&mut self, rel: &Relation, rows: &[usize], ctx: Ctx) -> bool {
        let classes_before: Vec<PatternId> = self.classes.iter().map(|c| c.id).collect();
        let fid = self.new_pid();
        let (mut f, new_class) = self.seed_function(rel, fid);
        if let Some(members) = new_class {
            let k = self.new_pid();
            self.classes.push(Class { id: k, members: members.iter().map(|n| Member { neuron: *n, count: 0 }).collect() });
            for (p, _) in f.places.iter_mut() { if let Place::Class(x) = p { if *x == PatternId::MAX { *x = k; } } }
        }
        self.functions.push(f);
        self.collapse(fid, rows, ctx);
        let fail = |me: &mut Neuron| {
            me.functions.retain(|f| f.id != fid);
            me.unsettled.remove(&fid);
            me.classes.retain(|c| classes_before.contains(&c.id) || me.functions.iter().any(|f| f.uses_class(c.id)));
            false
        };
        if self.function(fid).map_or(true, |f| f.p_size() == 0) || self.identical(fid).is_some() {
            if ctx.trace { eprintln!("[cand f{} #{}] {:?} over {} rows -> {}", ctx.frame, self.id, rel, rows.len(), if self.function(fid).map_or(true, |f| f.p_size() == 0) { "collapsed to nothing".to_string() } else { format!("identical to p{}", self.identical(fid).unwrap()) }); }
            return fail(self);
        }
        self.refit(fid, ctx);
        let f = self.function(fid).unwrap().clone();
        if f.uses.is_empty() {
            if ctx.trace { eprintln!("[cand f{} #{}] {:?} over {} rows -> {} fits no row", ctx.frame, self.id, rel, rows.len(), describe_function(&f)); }
            return fail(self);
        }
        let with: Vec<&Function> = self.functions.iter().collect();
        let without: Vec<&Function> = self.functions.iter().filter(|g| g.id != fid).collect();
        let index = self.row_index();
        let mut benefit = 0.0;
        for a in &f.uses {
            let Some(&i) = index.get(a) else { continue };
            let row = &self.history[i];
            let d = self.row_cost(row, &without, ctx) - self.row_cost(row, &with, ctx);
            if d > 0.0 { benefit += d; }
        }
        let made: f64 = self.classes.iter().filter(|c| !classes_before.contains(&c.id) && f.uses_class(c.id)).map(|c| self.class_entry(c, ctx)).sum();
        if ctx.trace { eprintln!("[cand f{} #{}] {:?} -> {} benefit {:.1} entry {:.1} classes {:.1}", ctx.frame, self.id, rel, describe_function(&f), benefit, self.entry(&f, ctx), made); }
        if benefit <= self.entry(&f, ctx) + made { return fail(self); }
        self.classes.retain(|c| classes_before.contains(&c.id) || self.functions.iter().any(|g| g.uses_class(c.id)));
        true
    }
}
