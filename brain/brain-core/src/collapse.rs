//! The collapse (D27, D29): what a function names, decided place by place over its uses, and the classes
//! its places use.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::neuron::*;
use crate::types::*;

/// One thing standing at an offset from the owner in a row, as the collapse reads it.
#[derive(Clone, Copy)]
pub(crate) struct Around {
    pub(crate) off: Offset,
    pub(crate) neuron: NeuronId,
    pub(crate) cost: f64,
    pub(crate) j: usize,
    /// Owned by another function of the cover, or for a candidate by any (D19, D27).
    pub(crate) owned: bool,
    /// A base symbol a child stands in for, at its share of the child's cost (D22, D47).
    pub(crate) standin: bool,
}

/// What one place could be, and what it is worth over the uses (D27).
#[derive(Clone, Debug, PartialEq)]
enum Choice {
    Nothing,
    Fixed(NeuronId),
    Class(PatternId),
    NewClass(Vec<NeuronId>),
    Param(u32),
}

impl Neuron {
    // ───────────────────────────── what stands around a use (D27) ─────────────────────────────

    /// What stands at every offset from the owner in a row: each neighbor, and each base symbol a child
    /// neighbor stands in for.
    pub(crate) fn around(&self, i: usize, fid: Option<PatternId>, creating: bool) -> Vec<Around> {
        let row = &self.history[i];
        let mut out = Vec::new();
        for (j, nb) in row.nbs.iter().enumerate() {
            let owned = if creating { row.owner[j].is_some() } else { row.owner[j].map_or(false, |o| Some(o) != fid) };
            out.push(Around { off: self.round(&nb.delta), neuron: nb.neuron, cost: nb.cost, j, owned, standin: false });
            if !nb.covers.is_empty() {
                let share = nb.cost / nb.covers.len() as f64;
                for (m, d) in &nb.covers { out.push(Around { off: self.round(d), neuron: *m, cost: share, j, owned, standin: true }); }
            }
        }
        out
    }

    /// Whether every place of a function away from `off` holds in a row: each fixed node stands at its
    /// offset, each class has a member at its place (D27). Parameters are read with the place they tie.
    pub(crate) fn holds_rest(&self, row: &Row, f: &Function, off: &Offset) -> bool {
        f.places.iter().skip(1).filter(|(_, o)| o != off).all(|(p, o)| match p {
            Place::Fixed(n) => !self.nbs_at(row, o, Some(*n)).is_empty(),
            Place::Class(k) => self.class(*k).map_or(false, |c| self.nbs_at(row, o, None).into_iter().any(|j| c.has(row.nbs[j].neuron))),
            _ => true,
        })
    }

    /// What stands at `off` in the rows a function does not use but would where a variable widened there
    /// (D27): rows where the rest of it holds, each with the neurons standing at `off` that no other
    /// function of the cover owns (any, for a candidate), not through a child.
    pub(crate) fn widened(&self, f: &Function, uses: &[usize], off: &Offset, creating: bool) -> Vec<Vec<(NeuronId, f64)>> {
        let mut out = Vec::new();
        for (i, row) in self.history.iter().enumerate() {
            if uses.contains(&i) || !self.holds_rest(row, f, off) { continue; }
            let here: Vec<(NeuronId, f64)> = row.nbs.iter().enumerate()
                .filter(|(j, nb)| self.round(&nb.delta) == *off && (if creating { row.owner[*j].is_none() } else { row.owner[*j].map_or(true, |o| o == f.id) }))
                .map(|(_, nb)| (nb.neuron, nb.cost)).collect();
            if !here.is_empty() { out.push(here); }
        }
        out
    }

    /// A function's parameters with their records (D44): per parameter, the neurons that have stood at all
    /// of its places within the history, each with the rows it stood in.
    pub(crate) fn record_params(&mut self, fid: PatternId) {
        let Some(f) = self.function(fid) else { return };
        if f.params.is_empty() { return; }
        let mut records: Vec<Vec<Member>> = Vec::new();
        for k in 0..f.params.len() as u32 {
            let offs: Vec<Offset> = f.param_places(k).iter().map(|&pi| f.places[pi].1).collect();
            let mut tally: FxHashMap<NeuronId, usize> = FxHashMap::default();
            for row in &self.history {
                let mut at_all: Option<Vec<NeuronId>> = None;
                for off in &offs {
                    let here: Vec<NeuronId> = row.nbs.iter().filter(|nb| self.round(&nb.delta) == *off).map(|nb| nb.neuron).collect();
                    at_all = Some(match at_all { None => here, Some(v) => v.into_iter().filter(|n| here.contains(n)).collect() });
                }
                let mut v = at_all.unwrap_or_default();
                v.sort();
                v.dedup();
                for n in v { *tally.entry(n).or_default() += 1; }
            }
            let mut m: Vec<Member> = tally.into_iter().map(|(neuron, count)| Member { neuron, count }).collect();
            m.sort_by_key(|x| x.neuron);
            records.push(m);
        }
        if let Some(g) = self.function_mut(fid) { for (p, r) in g.params.iter_mut().zip(records) { p.members = r; } }
    }

    /// The uses of a function (D27): the rows it covers where every place holds.
    pub(crate) fn uses_of(&self, fid: PatternId) -> Vec<usize> {
        (0..self.history.len()).filter(|&i| self.history[i].cover.iter().any(|e| e.function == fid && e.failed == 0)).collect()
    }

    /// The rows a function covers (D29).
    pub(crate) fn rows_of(&self, fid: PatternId) -> Vec<usize> {
        (0..self.history.len()).filter(|&i| self.history[i].cover.iter().any(|e| e.function == fid)).collect()
    }

    // ───────────────────────────── collapse (D27) ─────────────────────────────

    /// Re-decide what a function names over its uses (D27): every place at an offset from the owner, one
    /// at a time and each on its own, among a fixed node, a class, a parameter tying it to a place already
    /// in, and nothing, whichever covers most over the uses that count there less a failed node in each of
    /// the others and its entry in the table. The uses that count at a place are those where no other
    /// function of the cover owns its coordinate; a candidate being created owns nothing, so every owned
    /// neighbor is skipped. A function with a child never names a neuron built over it. Returns true when
    /// the function changed.
    pub(crate) fn collapse(&mut self, fid: PatternId, rows: &[usize], ctx: Ctx, creating: bool) -> bool {
        let Some(f) = self.function(fid).cloned() else { return false };
        if rows.is_empty() { return false; }
        let child = f.child;
        let fail = log2(f.p_size().max(1) as f64);
        let nb_bits = ctx.neuron_bits;
        let ob = ctx.offset_bits;
        let arounds: Vec<Vec<Around>> = rows.iter().map(|&i| self.around(i, Some(fid), creating)).collect();
        let uses_idx: Vec<usize> = rows.to_vec();
        // each parameter's value per use: the neuron at most of its places (D44)
        let param_value = |k: u32, skip: Option<Offset>, r: usize| -> Option<NeuronId> {
            let mut tally: FxHashMap<NeuronId, usize> = FxHashMap::default();
            for &pi in &f.param_places(k) {
                let off = f.places[pi].1;
                if Some(off) == skip { continue; }
                let mut seen: Vec<NeuronId> = Vec::new();
                for a in arounds[r].iter().filter(|a| !a.standin && a.off == off) { if !seen.contains(&a.neuron) { seen.push(a.neuron); *tally.entry(a.neuron).or_default() += 1; } }
            }
            tally.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|(n, _)| n)
        };
        // every offset anything stood at, and every place the function has
        let mut offs: Vec<Offset> = arounds.iter().flatten().map(|a| a.off).collect();
        offs.extend(f.places.iter().skip(1).map(|(_, o)| *o));
        offs.sort();
        offs.dedup();
        let classes_now = self.classes.len();
        let mut chosen: Vec<(Offset, Choice)> = Vec::new();
        let mut fallback: FxHashMap<Offset, Choice> = FxHashMap::default();
        for off in offs {
            if off[0] > 0 { continue; }
            let current: Vec<Place> = f.places.iter().skip(1).filter(|(_, o)| *o == off).map(|(p, _)| *p).collect();
            // the uses that count here
            let counted: Vec<usize> = (0..rows.len()).filter(|&r| !arounds[r].iter().any(|a| !a.standin && a.off == off && a.owned)).collect();
            let s = counted.len();
            if s == 0 { continue; }
            let mut best: (f64, Choice) = (0.0, Choice::Nothing);
            let mut best_plain: (f64, Choice) = (0.0, Choice::Nothing);
            let consider = |v: f64, c: Choice, plain: bool, best: &mut (f64, Choice), best_plain: &mut (f64, Choice)| {
                let keep = |cur: &Choice| current.iter().any(|p| match (p, cur) { (Place::Fixed(a), Choice::Fixed(b)) => a == b, (Place::Class(a), Choice::Class(b)) => a == b, (Place::Param(a), Choice::Param(b)) => a == b, _ => false });
                if v > best.0 || (v == best.0 && v > 0.0 && keep(&c)) { *best = (v, c.clone()); }
                if plain && (v > best_plain.0 || (v == best_plain.0 && v > 0.0 && keep(&c))) { *best_plain = (v, c); }
            };
            // a fixed node
            let mut fixed: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
            for &r in &counted {
                let mut here: FxHashMap<NeuronId, f64> = FxHashMap::default();
                for a in arounds[r].iter().filter(|a| a.off == off && !a.owned) {
                    let e = here.entry(a.neuron).or_insert(0.0);
                    if a.cost > *e { *e = a.cost; }
                }
                for (n, c) in here { let e = fixed.entry(n).or_insert((0, 0.0)); e.0 += 1; e.1 += c; }
            }
            let mut fx: Vec<(NeuronId, usize, f64)> = fixed.iter().map(|(n, (c, v))| (*n, *c, *v)).collect();
            fx.sort_by(|a, b| a.0.cmp(&b.0));
            for (n, cnt, cov) in &fx {
                if child.map_or(false, |c| self.is_built_over(*n, c)) { continue; }
                let v = cov - (s - cnt) as f64 * fail - (nb_bits + ob);
                consider(v, Choice::Fixed(*n), true, &mut best, &mut best_plain);
            }
            // a class of the table, over the uses and the rows widening to it adds (D27)
            let direct: Vec<Vec<(NeuronId, f64)>> = counted.iter().map(|&r| arounds[r].iter().filter(|a| a.off == off && !a.owned && !a.standin).map(|a| (a.neuron, a.cost)).collect()).collect();
            let wide = self.widened(&f, &uses_idx, &off, creating);
            for c in &self.classes {
                let k = c.members.len();
                if k < 2 { continue; }
                let ch = c.choice();
                let mut cnt = 0;
                let mut cov = 0.0;
                for here in &direct {
                    if let Some((_, cost)) = here.iter().filter(|(n, _)| c.has(*n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cnt += 1; cov += cost - ch; }
                }
                for here in &wide {
                    if let Some((_, cost)) = here.iter().filter(|(n, _)| c.has(*n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cov += cost - ch; }
                }
                if cnt == 0 && cov <= 0.0 { continue; }
                let v = cov - (s - cnt) as f64 * fail - (self.which_class_bits() + ob);
                consider(v, Choice::Class(c.id), true, &mut best, &mut best_plain);
            }
            // a new class: its members taken in order of count, each while what it covers beats a neuron and
            // the wider choice it gives every use the class already has
            let mut cands: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
            for here in direct.iter().chain(wide.iter()) {
                let mut seen: Vec<NeuronId> = Vec::new();
                for (n, cost) in here { if seen.contains(n) { continue; } seen.push(*n); let e = cands.entry(*n).or_insert((0, 0.0)); e.0 += 1; e.1 += cost; }
            }
            let mut order: Vec<(NeuronId, usize, f64)> = cands.iter().map(|(n, (c, v))| (*n, *c, *v)).filter(|(n, _, _)| !child.map_or(false, |c| self.is_built_over(*n, c))).collect();
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut members: Vec<NeuronId> = Vec::new();
            let mut uses_k = 0usize;
            for (n, cnt, cov) in &order {
                let k = members.len();
                let widening = if k == 0 { 0.0 } else { uses_k as f64 * log2((k + 1) as f64 / k as f64) };
                if *cov > nb_bits + widening { members.push(*n); uses_k += cnt; } else if k > 0 { break; }
            }
            if members.len() >= 2 {
                members.sort();
                let existing = self.classes.iter().find(|c| { let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect(); m.sort(); m == members });
                if existing.is_none() {
                    let ch = log2(members.len() as f64);
                    let mut cnt = 0;
                    let mut cov = 0.0;
                    for here in &direct {
                        if let Some((_, cost)) = here.iter().filter(|(n, _)| members.contains(n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cnt += 1; cov += cost - ch; }
                    }
                    for here in &wide {
                        if let Some((_, cost)) = here.iter().filter(|(n, _)| members.contains(n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cov += cost - ch; }
                    }
                    let entry = log2((classes_now + 1) as f64) + ob + members.len() as f64 * nb_bits;
                    let v = cov - (s - cnt) as f64 * fail - entry;
                    consider(v, Choice::NewClass(members.clone()), true, &mut best, &mut best_plain);
                }
            }
            // a parameter tying this place to its places already in
            for k in 0..f.params.len() as u32 {
                let mut cnt = 0;
                let mut cov = 0.0;
                for &r in &counted {
                    let Some(v) = param_value(k, Some(off), r) else { continue };
                    if let Some(a) = arounds[r].iter().find(|a| a.off == off && !a.owned && !a.standin && a.neuron == v) { cnt += 1; cov += a.cost; }
                }
                if cnt == 0 { continue; }
                let v = cov - (s - cnt) as f64 * fail - ob;
                consider(v, Choice::Param(k), false, &mut best, &mut best_plain);
            }
            fallback.insert(off, best_plain.1.clone());
            if best.1 != Choice::Nothing { chosen.push((off, best.1)); }
        }
        // a parameter left with one place is a fixed node or nothing (R41)
        let mut param_count: FxHashMap<u32, usize> = FxHashMap::default();
        for (_, c) in &chosen { if let Choice::Param(k) = c { *param_count.entry(*k).or_default() += 1; } }
        for (off, c) in chosen.iter_mut() {
            if let Choice::Param(k) = c { if param_count[k] < 2 { *c = fallback.get(off).cloned().unwrap_or(Choice::Nothing); } }
        }
        // build the new body; a new class joins the table
        let mut rest: Vec<(Place, Offset)> = Vec::new();
        for (off, c) in chosen {
            let place = match c {
                Choice::Nothing => continue,
                Choice::Fixed(n) => Place::Fixed(n),
                Choice::Class(k) => Place::Class(k),
                Choice::Param(k) => Place::Param(k),
                Choice::NewClass(members) => {
                    let existing = self.classes.iter().find(|c| { let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect(); m.sort(); m == members }).map(|c| c.id);
                    let k = match existing {
                        Some(k) => k,
                        None => { let id = self.new_pid(); self.classes.push(Class { id, members: members.iter().map(|n| Member { neuron: *n, count: 0 }).collect() }); id }
                    };
                    Place::Class(k)
                }
            };
            rest.push((place, off));
        }
        let mut nf = Function { id: f.id, places: vec![(Place::Owner, [0; MAX_DIMS])], params: f.params.clone(), child: f.child };
        nf.places.extend(rest);
        nf.canonicalize();
        let changed = nf.places != f.places;
        if changed && ctx.trace { eprintln!("[collapse f{} #{}] p{} {} -> {} over {} rows", ctx.frame, self.id, fid, describe_function(&f), describe_function(&nf), rows.len()); }
        if changed { if let Some(g) = self.function_mut(fid) { g.places = nf.places; g.params = nf.params; } }
        self.record_params(fid);
        changed
    }

    /// Re-decide every class's members over everywhere it is used (D27, D41): what stands at the places
    /// functions use it, in the rows they cover. A member stays while what it covers beats a neuron and the
    /// wider choice it gives; a neuron that is not a member joins by the same test, in order of count. A
    /// class never takes a neuron built over the child of a function using it.
    pub(crate) fn update_classes(&mut self, ctx: Ctx) {
        let ids: Vec<PatternId> = self.classes.iter().map(|c| c.id).collect();
        for k in ids {
            let users: Vec<Function> = self.functions.iter().filter(|f| f.uses_class(k)).cloned().collect();
            if users.is_empty() { continue; }
            let mut cands: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
            let mut uses = 0usize;
            for f in &users {
                let offs: Vec<Offset> = f.places.iter().filter(|(p, _)| *p == Place::Class(k)).map(|(_, o)| *o).collect();
                let used = self.uses_of(f.id);
                for i in 0..self.history.len() {
                    let row = &self.history[i];
                    for off in &offs {
                        // a row it does not use counts where widening the class would make it one (D27)
                        if !used.contains(&i) && !self.holds_rest(row, f, off) { continue; }
                        uses += 1;
                        let mut seen: Vec<NeuronId> = Vec::new();
                        for (j, nb) in row.nbs.iter().enumerate() {
                            if self.round(&nb.delta) != *off { continue; }
                            if row.owner[j].map_or(false, |o| o != f.id) { continue; }
                            if seen.contains(&nb.neuron) { continue; }
                            seen.push(nb.neuron);
                            let e = cands.entry(nb.neuron).or_insert((0, 0.0));
                            e.0 += 1;
                            e.1 += nb.cost;
                        }
                    }
                }
            }
            let _ = uses;
            let banned = |n: NeuronId| users.iter().any(|f| f.child.map_or(false, |c| self.is_built_over(n, c)));
            let mut order: Vec<(NeuronId, usize, f64)> = cands.iter().map(|(n, (c, v))| (*n, *c, *v)).filter(|(n, _, _)| !banned(*n)).collect();
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut members: Vec<Member> = Vec::new();
            let mut uses_k = 0usize;
            for (n, cnt, cov) in &order {
                let m = members.len();
                let widening = if m == 0 { 0.0 } else { uses_k as f64 * log2((m + 1) as f64 / m as f64) };
                if *cov > ctx.neuron_bits + widening { members.push(Member { neuron: *n, count: *cnt }); uses_k += cnt; } else if m > 0 { break; }
            }
            members.sort_by_key(|m| m.neuron);
            if let Some(c) = self.class_mut(k) {
                if !members.is_empty() { c.members = members; } else { c.members.clear(); }
            }
        }
    }

    /// Re-center every function whose rows changed (D29), re-decide the classes, and re-cover the rows a
    /// moved function stands in.
    pub(crate) fn recenter(&mut self, changed: &FxHashSet<PatternId>, ctx: Ctx) {
        let mut ids: Vec<PatternId> = changed.iter().copied().filter(|f| self.function(*f).is_some()).collect();
        ids.sort();
        let mut moved: Vec<PatternId> = Vec::new();
        for fid in ids {
            let rows = self.uses_of(fid);
            if self.collapse(fid, &rows, ctx, false) { moved.push(fid); }
        }
        let before: Vec<(PatternId, Vec<NeuronId>)> = self.classes.iter().map(|c| (c.id, c.members.iter().map(|m| m.neuron).collect())).collect();
        self.update_classes(ctx);
        let after: Vec<(PatternId, Vec<NeuronId>)> = self.classes.iter().map(|c| (c.id, c.members.iter().map(|m| m.neuron).collect())).collect();
        for ((k, a), (_, b)) in before.iter().zip(after.iter()) {
            if a != b { for f in self.functions.iter().filter(|f| f.uses_class(*k)) { if !moved.contains(&f.id) { moved.push(f.id); } } }
        }
        if moved.is_empty() { return; }
        for i in 0..self.history.len() {
            if self.history[i].cover.iter().any(|e| moved.contains(&e.function)) { self.history[i].dirty = true; }
        }
        self.recognize(ctx);
    }
}
