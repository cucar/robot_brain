//! The greedy pick (D33, D47, R14, R15): seeds, their collapse, their price, and the claims they leave.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

#[derive(Debug)]
pub(crate) enum VarSeed {
    Parameter { offset: Offset, uses: Vec<(Pos, Pos)> },
    Class { anchor: (Node, Offset), members: (NeuronId, NeuronId), uses: Vec<Pos> },
}

/// The rows of a history by the coordinates they hold, built once per call (see `coordinate_index`).
pub(crate) struct CoordIndex {
    pub fine: FxHashMap<Pos, Vec<usize>>,
    pub owner: FxHashMap<Pos, Vec<usize>>,
    pub coarse: Vec<(usize, FxHashSet<Offset>)>,
}

/// Whether `a` is later than `b`: later in time, then greater on the other axes.
pub(crate) fn later(a: &Pos, b: &Pos) -> bool { a > b }

impl Neuron {
    // ───────────────────────────── the greedy pick (D33) ─────────────────────────────

    pub(crate) fn new_pid(&mut self) -> PatternId { let id = self.next_pid; self.next_pid += 1; id }

    /// Build patterns out of the residual: variables first, then functions (D33). Returns how many were added.
    pub(crate) fn create(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        let first_pid = self.next_pid;
        // stage 1: variables; a seed that builds nothing is passed over and the next is tried (D33)
        let mut tried: FxHashSet<String> = FxHashSet::default();
        loop {
            let Some(seed) = self.best_variable_seed(ctx, &tried) else { break };
            let key = format!("{:?}", seed);
            tried.insert(key);
            let built = match seed {
                VarSeed::Parameter { offset, uses } => self.build_parameter(offset, &uses, ctx),
                VarSeed::Class { anchor, members, uses } => self.build_class(anchor, members, &uses, ctx),
            };
            if built { added += 1; }
        }
        let new_vars: Vec<PatternId> = self.table.iter().filter(|p| p.is_variable() && p.id >= first_pid).map(|p| p.id).collect();
        // stage 2: functions. The seeds are ranked once over the residual as it stands; a seed that
        // builds nothing changes nothing, so the next in the ranking is the next the rule names; one that
        // builds changes the residual, so the ranking is read again (D33). The rows are indexed by the
        // coordinates they hold once per call, for the uses of each seed tried.
        let at = self.coordinate_index();
        let mut tried: FxHashSet<(Node, Node, Offset)> = FxHashSet::default();
        loop {
            let ranked = self.ranked_function_seeds(ctx, &tried);
            let mut built = false;
            for ((a, b, off), pairs) in ranked {
                tried.insert((a, b, off));
                let uses = self.function_seed_places(a, pairs, &at);
                if self.build_function(a, b, off, &uses, ctx) { added += 1; built = true; break; }
            }
            if !built { break; }
        }
        // a variable no function names would write what it writes alone, and is not kept (D30)
        for v in new_vars {
            let named = self.table.iter().any(|p| matches!(&p.body, Body::Function { nodes, .. } if nodes.iter().any(|(n, _)| *n == Node::Var(v))));
            if !named {
                let r = self.remove_pattern(v);
                for row in self.history.iter_mut() { for c in row.claim.iter_mut() { if matches!(c, Some((x, _)) if *x == v) { *c = None; row.dirty = true; } } }
                let _ = r;
                added -= 1;
            }
        }
        added
    }

    /// The parameter seed (D47): pairs of residual coordinates holding one neuron, keyed by the offset
    /// between them; and the class seed: an offset from a node where neurons vary. The one that would
    /// save most as it stands (D33).
    pub(crate) fn best_variable_seed(&self, ctx: Ctx, tried: &FxHashSet<String>) -> Option<VarSeed> {
        let wp = log2((self.table.len() + 1) as f64);
        // parameter seeds
        struct PS { uses: FxHashMap<(Pos, Pos), (usize, f64)>, values: FxHashSet<NeuronId> }
        let mut ps: FxHashMap<Offset, PS> = FxHashMap::default();
        // class seeds: (anchor node, offset) -> per coordinate (use) the neuron standing, with credit
        struct CS { uses: FxHashMap<Pos, (NeuronId, f64, usize)> }
        let mut cs: FxHashMap<(Node, Offset), CS> = FxHashMap::default();
        for (i, row) in self.history.iter().enumerate() {
            let res: Vec<usize> = (0..row.nbs.len()).filter(|&j| row.residual(j) && row.claim[j].is_none()).collect();
            for &j in &res {
                let cj = row.coord(j);
                // class: from the owner
                let off = self.round(&row.nbs[j].delta);
                cs.entry((Node::Owner, off)).or_insert_with(|| CS { uses: FxHashMap::default() }).uses.entry(cj).or_insert((row.nbs[j].neuron, row.credit(j), i));
                for &k in &res {
                    if k == j { continue; }
                    let ck = row.coord(k);
                    let d = sub(&cj, &ck);
                    let off = self.round(&d);
                    // class: from a neighbor
                    cs.entry((Node::Neuron(row.nbs[k].neuron), off)).or_insert_with(|| CS { uses: FxHashMap::default() }).uses.entry(cj).or_insert((row.nbs[j].neuron, row.credit(j), i));
                    // parameter: the first position is the later one
                    if row.value_at(j) != row.value_at(k) { continue; }
                    if !later(&cj, &ck) { continue; }
                    let e = ps.entry(off).or_insert_with(|| PS { uses: FxHashMap::default(), values: FxHashSet::default() });
                    e.values.insert(row.value_at(j));
                    e.uses.entry((cj, ck)).or_insert((i, row.credit(j) + row.credit(k)));
                }
            }
        }
        let mut best: Option<(f64, usize, VarSeed)> = None;
        let consider = |best: &mut Option<(f64, usize, VarSeed)>, save: f64, count: usize, seed: VarSeed| {
            let better = match best { None => true, Some((bs, bc, _)) => save > *bs || (save == *bs && count > *bc) };
            if better { *best = Some((save, count, seed)); }
        };
        for (off, e) in &ps {
            if e.uses.len() < 2 { continue; }
            let choice = log2(e.values.len() as f64);
            let mut save = 0.0;
            for (_, (_, c)) in &e.uses { save += c - choice; }
            save -= ctx.offset_bits; // as it stands: one offset written
            let mut uses: Vec<(Pos, Pos)> = e.uses.keys().copied().collect();
            uses.sort();
            let seed = VarSeed::Parameter { offset: *off, uses };
            if tried.contains(&format!("{:?}", seed)) { continue; }
            consider(&mut best, save, e.uses.len(), seed);
        }
        for ((node, off), e) in &cs {
            if e.uses.len() < 2 { continue; }
            // the two neurons that stood there most, never together: coordinates are distinct uses here
            let mut tally: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
            for (_, (n, c, _)) in &e.uses { let t = tally.entry(*n).or_insert((0, 0.0)); t.0 += 1; t.1 += c; }
            if tally.len() < 2 { continue; }
            let mut order: Vec<(NeuronId, usize, f64)> = tally.iter().map(|(n, (c, cov))| (*n, *c, *cov)).collect();
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let (a, b) = (order[0], order[1]);
            let count = a.1 + b.1;
            let save = a.2 + b.2 - count as f64 * 1.0 - 2.0 * ctx.neuron_bits;
            let mut uses: Vec<Pos> = e.uses.iter().filter(|(_, (n, _, _))| *n == a.0 || *n == b.0).map(|(c, _)| *c).collect();
            uses.sort();
            let seed = VarSeed::Class { anchor: (*node, *off), members: (a.0, b.0), uses };
            if tried.contains(&format!("{:?}", seed)) { continue; }
            consider(&mut best, save, count, seed);
        }
        let _ = wp;
        best.map(|(_, _, s)| s)
    }

    pub(crate) fn build_parameter(&mut self, offset: Offset, uses: &[(Pos, Pos)], ctx: Ctx) -> bool {
        // an existing parameter with these offsets is used rather than a new one (D33)
        if self.table.iter().any(|p| matches!(&p.body, Body::Parameter { offsets, .. } if offsets.len() == 2 && offsets[1] == offset)) { return false; }
        let pid = self.new_pid();
        self.table.push(Pattern { id: pid, body: Body::Parameter { offsets: vec![[0; MAX_DIMS], offset], members: Vec::new() } });
        // the seed's uses, as places: the first position's coordinate, with the rows that saw it
        let mut places: Vec<(Pos, Vec<usize>)> = Vec::new();
        for (first, _) in uses {
            let rows: Vec<usize> = self.history.iter().enumerate().filter(|(_, r)| !self.nbs_at(r, &sub(first, &r.pos), None).is_empty()).map(|(i, _)| i).collect();
            places.push((*first, rows));
        }
        self.collapse_parameter(pid, &places, ctx);
        let ok = match self.pattern(pid) { Some(Pattern { body: Body::Parameter { offsets, .. }, .. }) => offsets.len() >= 2, _ => false };
        if !ok { self.table.retain(|p| p.id != pid); return false; }
        // margin as a seed: its part over its uses (D22, D30)
        let mut benefit = 0.0;
        for (first, rows) in &places {
            if let Some(&i) = rows.last() {
                if let Some((part, _)) = self.var_part_at(&self.history[i], pid, first) { benefit += part.max(0.0); }
            }
        }
        let line = self.line_cost(self.pattern(pid).unwrap(), ctx);
        if benefit <= line { self.table.retain(|p| p.id != pid); return false; }
        // claim the coordinates of its uses
        self.claim_parameter(pid);
        true
    }

    /// A kept parameter claims the coordinates of its uses (D33): every place in the residual where one
    /// neuron stands at all its positions.
    pub(crate) fn claim_parameter(&mut self, pid: PatternId) {
        let Some(p) = self.pattern(pid).cloned() else { return };
        let Body::Parameter { offsets, .. } = &p.body else { return };
        for i in 0..self.history.len() {
            let row = &self.history[i];
            let mut binds: Vec<(usize, usize)> = Vec::new();
            let taken: Vec<bool> = (0..row.nbs.len()).map(|j| !row.residual(j) || row.claim[j].is_some()).collect();
            let mut anchors: Vec<Pos> = Vec::new();
            for j in 0..row.nbs.len() { if !taken[j] { anchors.push(row.nbs[j].delta); } }
            for anchor in anchors {
                let Some(vb) = self.bind_parameter(row, &p, offsets, &anchor, &taken) else { continue };
                if vb.failed > 0 { continue; }
                for (k, pj) in vb.positions.iter().enumerate() { if let Some(x) = pj { if !taken[*x] && !binds.iter().any(|(y, _)| y == x) { binds.push((*x, k)); } } }
            }
            let row = &mut self.history[i];
            for (j, k) in binds { if row.claim[j].is_none() { row.claim[j] = Some((pid, k)); row.dirty = true; } }
        }
    }

    pub(crate) fn build_class(&mut self, anchor: (Node, Offset), members: (NeuronId, NeuronId), uses: &[Pos], ctx: Ctx) -> bool {
        let ms = { let mut v = vec![members.0, members.1]; v.sort(); v };
        if self.table.iter().any(|p| matches!(&p.body, Body::Class { members, .. } if members.iter().map(|m| m.neuron).collect::<Vec<_>>() == ms)) { return false; }
        let pid = self.new_pid();
        self.table.push(Pattern { id: pid, body: Body::Class { members: ms.iter().map(|n| Member { neuron: *n, count: 0, value_neuron: None }).collect(), anchor } });
        let mut places: Vec<(Pos, Vec<usize>)> = Vec::new();
        for c in uses {
            let rows: Vec<usize> = self.history.iter().enumerate().filter(|(_, r)| !self.nbs_at(r, &sub(c, &r.pos), None).is_empty()).map(|(i, _)| i).collect();
            places.push((*c, rows));
        }
        self.collapse_class(pid, &places, ctx);
        let ok = self.pattern(pid).map_or(false, |p| p.members().len() >= 2);
        if !ok { self.table.retain(|p| p.id != pid); return false; }
        let mut benefit = 0.0;
        for (c, rows) in &places {
            if let Some(&i) = rows.last() {
                if let Some((part, _)) = self.var_part_at(&self.history[i], pid, c) { benefit += part.max(0.0); }
            }
        }
        let line = self.line_cost(self.pattern(pid).unwrap(), ctx);
        if benefit <= line { self.table.retain(|p| p.id != pid); return false; }
        // claim: the coordinates of its uses
        for (c, rows) in &places {
            for &i in rows {
                let row = &mut self.history[i];
                let target = sub(c, &row.pos);
                let js: Vec<usize> = (0..row.nbs.len()).filter(|&j| row.claim[j].is_none() && row.residual(j) && ms.contains(&row.nbs[j].neuron) && round_delta(&row.nbs[j].delta, &self.radii, self.axes) == round_delta(&target, &self.radii, self.axes)).collect();
                if let Some(&j) = js.first() { row.claim[j] = Some((pid, 0)); row.dirty = true; }
            }
        }
        true
    }

    /// The function seeds (D47): two nodes standing together, read in nodes (D33), ranked by what each
    /// would save as it stands, most first. Each comes with the offset from the first to the second and
    /// the coordinates of its uses.
    pub(crate) fn ranked_function_seeds(&self, ctx: Ctx, tried: &FxHashSet<(Node, Node, Offset)>) -> Vec<((Node, Node, Offset), Vec<(Pos, Pos)>)> {
        let wp = log2((self.table.len() + 1) as f64);
        struct FS { uses: FxHashMap<(Pos, Pos), (usize, f64)> }
        let mut fs: FxHashMap<(Node, Node, Offset), FS> = FxHashMap::default();
        for (i, row) in self.history.iter().enumerate() {
            // nodes of the row: the owner at zero, residual neighbors, claimed coordinates as variables at their first position
            // the owner's activation is residual where no entry of the cover is credited it; named where it
            // is not, it is written and saves nothing
            let mut nodes: Vec<(Node, Pos, f64)> = vec![(Node::Owner, row.pos, if row.owner_self.is_none() { row.own_cost } else { 0.0 })];
            let mut seen_vars: FxHashSet<PatternId> = FxHashSet::default();
            for j in 0..row.nbs.len() {
                match row.claim[j] {
                    Some((v, 0)) => {
                        if !seen_vars.insert(v) { continue; }
                        let c = row.coord(j);
                        let part = self.var_part_at(row, v, &c).map(|(p, _)| p).unwrap_or(0.0).max(0.0);
                        nodes.push((Node::Var(v), c, part));
                    }
                    Some(_) => {}
                    None => {
                        if row.residual(j) {
                            nodes.push((Node::Neuron(row.nbs[j].neuron), row.coord(j), row.credit(j)));
                            // the child read as what it stands for, its credit spread over them
                            let nb = &row.nbs[j];
                            if !nb.covers.is_empty() {
                                let share = row.credit(j) / nb.covers.len() as f64;
                                for (m, d) in &nb.covers { nodes.push((Node::Neuron(*m), add(&row.pos, d), share)); }
                            }
                        }
                    }
                }
            }
            for a in 0..nodes.len() {
                for b in 0..nodes.len() {
                    if a == b { continue; }
                    let (na, ca, cra) = nodes[a];
                    let (nb, cb, crb) = nodes[b];
                    // the first is the later one; the owner is first where it is in the pair
                    if matches!(nb, Node::Owner) { continue; }
                    if !matches!(na, Node::Owner) && !later(&ca, &cb) { continue; }
                    let off = self.round(&sub(&cb, &ca));
                    if off[0] > 0 { continue; }
                    let e = fs.entry((na, nb, off)).or_insert_with(|| FS { uses: FxHashMap::default() });
                    e.uses.entry((ca, cb)).or_insert((i, cra + crb));
                }
            }
        }
        let mut ranked: Vec<(f64, usize, (Node, Node, Offset), Vec<(Pos, Pos)>)> = Vec::new();
        for (key, e) in fs {
            if e.uses.len() < 2 || tried.contains(&key) { continue; }
            // as it stands: covers both, writes the owner, which pattern and no failure; the owner cancels where named
            let mut save = 0.0;
            let mut count = 0;
            for (_, (i, cov)) in &e.uses {
                let row = &self.history[*i];
                let price = if matches!(key.0, Node::Owner) && row.owner_self.is_none() { wp } else { row.own_cost + wp };
                let s = cov - price;
                if s > 0.0 { save += s; count += 1; }
            }
            if count < 2 { continue; }
            ranked.push((save, count, key, e.uses.into_keys().collect()));
        }
        let _ = ctx;
        // most saved first, then most uses, then the lesser key
        ranked.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then(y.1.cmp(&x.1)).then(x.2.cmp(&y.2)));
        ranked.into_iter().map(|(_, _, key, pairs)| (key, pairs)).collect()
    }

    /// The rows by the coordinates their neighbors stand at, as `nbs_at` reads them (D6): a neighbor
    /// within the radius stands at its coordinate exactly; one beyond it stands at a rounded offset,
    /// which the rows holding any such offset keep by row, read against the rounded offset of the
    /// coordinate asked for.
    pub(crate) fn coordinate_index(&self) -> CoordIndex {
        let mut fine: FxHashMap<Pos, Vec<usize>> = FxHashMap::default();
        let mut owner: FxHashMap<Pos, Vec<usize>> = FxHashMap::default();
        let mut coarse: Vec<(usize, FxHashSet<Offset>)> = Vec::new();
        for (i, row) in self.history.iter().enumerate() {
            owner.entry(row.pos).or_default().push(i);
            let mut far: FxHashSet<Offset> = FxHashSet::default();
            for nb in &row.nbs {
                if self.within(&nb.delta) {
                    let v = fine.entry(add(&self.mask(&row.pos), &self.mask(&nb.delta))).or_default();
                    if v.last() != Some(&i) { v.push(i); }
                } else {
                    far.insert(self.round(&nb.delta));
                }
            }
            if !far.is_empty() { coarse.push((i, far)); }
        }
        CoordIndex { fine, owner, coarse }
    }

    /// A delta on the owner's axes alone, as rounding leaves it within the radius.
    pub(crate) fn mask(&self, d: &Pos) -> Pos {
        let mut out = [0; MAX_DIMS];
        for k in 0..MAX_DIMS.min(self.radii.len()) { if self.axes & (1 << k) != 0 { out[k] = d[k]; } }
        out
    }

    /// Whether a delta lies within the radius on every axis the owner has, where rounding is exact and a
    /// rounded offset beyond it never lands (D6).
    pub(crate) fn within(&self, d: &Pos) -> bool {
        (0..MAX_DIMS.min(self.radii.len())).all(|k| self.axes & (1 << k) == 0 || d[k].abs() <= self.radii[k])
    }

    /// A function seed's uses as places: the first node's coordinate, with the rows that saw it there.
    pub(crate) fn function_seed_places(&self, a: Node, pairs: Vec<(Pos, Pos)>, at: &CoordIndex) -> Vec<(Pos, Vec<usize>)> {
        let mut places: Vec<(Pos, Vec<usize>)> = Vec::new();
        for (ca, _) in pairs {
            let mut rows: Vec<usize> = match a {
                Node::Owner => at.owner.get(&ca).cloned().unwrap_or_default(),
                _ => {
                    let mut v = at.fine.get(&self.mask(&ca)).cloned().unwrap_or_default();
                    for (i, far) in &at.coarse {
                        let d = sub(&ca, &self.history[*i].pos);
                        if !self.within(&d) && far.contains(&self.round(&d)) { v.push(*i); }
                    }
                    v.sort_unstable();
                    v.dedup();
                    v
                }
            };
            rows.shrink_to_fit();
            if !rows.is_empty() { places.push((ca, rows)); }
        }
        places.sort();
        places.dedup_by(|x, y| x.0 == y.0);
        places
    }

    pub(crate) fn build_function(&mut self, a: Node, b: Node, off: Offset, uses: &[(Pos, Vec<usize>)], ctx: Ctx) -> bool {
        let pid = self.new_pid();
        let nodes = canonical(vec![(a, [0; MAX_DIMS]), (b, off)]);
        self.table.push(Pattern { id: pid, body: Body::Function { nodes, child: None } });
        self.collapse_function(pid, uses, ctx);
        let ok = self.pattern(pid).map_or(false, |p| matches!(&p.body, Body::Function { nodes, .. } if nodes.len() >= 2));
        if !ok { self.table.retain(|p| p.id != pid); return false; }
        // merge: an identical line already in the table takes the uses (D43)
        if let Some(existing) = self.identical(pid) {
            self.table.retain(|p| p.id != pid);
            let _ = existing;
            return false;
        }
        // price it (R14, R15): what each use's cover saves with it in it less what it saves as it
        // stands, each use once in its latest row, counting only where that is positive
        let p = self.pattern(pid).unwrap().clone();
        let mut benefit = 0.0;
        let mut joins: Vec<(usize, Vec<CoverEntry>)> = Vec::new();
        let mut seen: FxHashSet<usize> = FxHashSet::default();
        for (_, rows) in uses {
            let Some(&i) = rows.iter().max_by_key(|&&i| (self.history[i].pos[0], i)) else { continue };
            if !seen.insert(i) { continue; }
            // a claim enters a cover only where it pays with nothing taken; where none of the line's
            // placements does, the row's cover is as it stands and the line adds nothing there
            {
                let row = &self.history[i];
                let free = vec![false; row.nbs.len()];
                let wp = self.which_pattern_bits();
                if !self.claimants(row, &p, &free, false, wp, ctx).iter().any(|c| c.coverage > c.price) { continue; }
            }
            let with = self.greedy_cover_without(i, None, ctx);
            if !with.iter().any(|e| e.pattern == pid) { continue; }
            let as_is = self.greedy_cover_without(i, Some(pid), ctx);
            let diff = self.savings_of_cover(i, &with) - self.savings_of_cover(i, &as_is);
            if diff > 0.0 { benefit += diff; joins.push((i, with)); }
        }
        let line = self.line_cost(&p, ctx);
        if benefit <= line { self.table.retain(|p| p.id != pid); return false; }
        // join the covers it was priced on, owning the residual it names there (R15)
        for (i, cover) in joins { self.install_cover(i, cover); }
        true
    }

    /// What a cover saves over a row (D22 summed), read with its owners as that cover would set them.
    pub(crate) fn savings_of_cover(&self, i: usize, cover: &[CoverEntry]) -> f64 {
        let row = &self.history[i];
        let mut tmp = row.clone();
        let n = tmp.nbs.len();
        tmp.owner = vec![None; n]; tmp.claim = vec![None; n]; tmp.owner_self = None;
        for e in cover {
            if e.credited_owner && tmp.owner_self.is_none() { tmp.owner_self = Some(e.pattern); }
            for nb in &e.nodes {
                for &j in &nb.nbs { if tmp.owner[j].is_none() { tmp.owner[j] = Some(e.pattern); } }
                if let Some(v) = &nb.var { for pj in v.positions.iter().flatten() { if tmp.owner[*pj].is_none() { tmp.owner[*pj] = Some(e.pattern); } } }
            }
        }
        cover.iter().map(|e| self.entry_savings(&tmp, e)).sum()
    }

}
