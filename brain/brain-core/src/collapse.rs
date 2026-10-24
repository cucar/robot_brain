//! The collapse (D27) and re-centering (D29): what a function names, decided place by place over its uses,
//! and the members of the classes its places use.

use rustc_hash::FxHashMap;

use crate::neuron::*;
use crate::types::*;

/// One thing standing at an offset from the owner in a use, as the collapse reads it (D27): a neighbor at
/// what it cost and its position, or a neuron a child stands for there, which holds a node and is worth
/// nothing.
#[derive(Clone, Copy)]
pub(crate) struct Around {
    pub(crate) off: Offset,
    pub(crate) neuron: NeuronId,
    pub(crate) cost: f64,
    pub(crate) pos: f64,
    pub(crate) under: bool,
}

/// What one place could be (D27).
#[derive(Clone, Debug, PartialEq)]
enum Choice {
    Nothing,
    Fixed(NeuronId),
    Class(PatternId),
    NewClass(Vec<NeuronId>),
    Param(u32),
}

impl Neuron {
    /// Everything standing at an offset from the owner in a row: each neighbor, and each neuron a child
    /// stands for there (D27, D47).
    pub(crate) fn around(&self, row: &Row) -> Vec<Around> {
        let mut out = Vec::new();
        for nb in &row.nbs {
            out.push(Around { off: self.round(&nb.delta), neuron: nb.neuron, cost: nb.cost, pos: nb.pos, under: false });
            for (m, d) in &nb.stands_for { out.push(Around { off: self.round(d), neuron: *m, cost: 0.0, pos: 0.0, under: true }); }
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

    // ───────────────────────────── re-centering (D29) ─────────────────────────────

    /// Re-center a function whose uses changed (D29): the collapse over its uses as they stand, then its fit
    /// re-read over every row; where that changed its uses it re-centers again at the next call (D27).
    pub(crate) fn recenter(&mut self, fid: PatternId, ctx: Ctx) {
        let Some(f) = self.function(fid) else { return };
        let index = self.row_index();
        let rows: Vec<usize> = f.uses.iter().filter_map(|a| index.get(a).copied()).collect();
        if rows.is_empty() { return; }
        self.collapse(fid, &rows, ctx);
        if self.function(fid).map_or(false, |f| f.p_size() > 0) && self.refit(fid, ctx) { self.unsettled.insert(fid); }
    }

    // ───────────────────────────── collapse (D27) ─────────────────────────────

    /// Decide what a function names over its uses (D27): every place at an offset from the owner, each on its
    /// own, among a fixed node, a class, a parameter tying it to a place already in, and nothing, whichever
    /// covers most over the uses less an absence in each use where it does not hold and its entry. Nothing in
    /// a use is skipped because another function names it. A neuron built over the function's own child may
    /// be named, which is how a chain closes on itself (R43). Returns true when the function changed.
    pub(crate) fn collapse(&mut self, fid: PatternId, rows: &[usize], ctx: Ctx) -> bool {
        let Some(f) = self.function(fid).cloned() else { return false };
        if rows.is_empty() { return false; }
        let nb_bits = ctx.neuron_bits;
        let ob = self.offset_bits();
        let arounds: Vec<Vec<Around>> = rows.iter().map(|&i| self.around(&self.history[i])).collect();
        let s = rows.len();
        let recs: Vec<f64> = rows.iter().map(|&i| self.history[i].rec).collect();
        let all_recs: f64 = recs.iter().sum();
        // each parameter's value per use: the neuron at the earliest of its places, this one left out (D44)
        let param_value = |k: u32, skip: Offset, r: usize| -> Option<NeuronId> {
            let mut places: Vec<Offset> = f.param_places(k).iter().map(|&pi| f.places[pi].1).filter(|o| *o != skip).collect();
            places.sort();
            places.iter().find_map(|off| arounds[r].iter().filter(|a| !a.under && a.off == *off).min_by(|a, b| a.neuron.cmp(&b.neuron)).map(|a| a.neuron))
        };
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
            let mut best: (f64, Choice) = (0.0, Choice::Nothing);
            let mut best_plain: (f64, Choice) = (0.0, Choice::Nothing);
            let consider = |v: f64, c: Choice, plain: bool, best: &mut (f64, Choice), best_plain: &mut (f64, Choice)| {
                let keep = |cur: &Choice| current.iter().any(|p| match (p, cur) { (Place::Fixed(a), Choice::Fixed(b)) => a == b, (Place::Class(a), Choice::Class(b)) => a == b, (Place::Param(a), Choice::Param(b)) => a == b, _ => false });
                if v > best.0 || (v == best.0 && v > 0.0 && keep(&c)) { *best = (v, c.clone()); }
                if plain && (v > best_plain.0 || (v == best_plain.0 && v > 0.0 && keep(&c))) { *best_plain = (v, c); }
            };
            // a fixed node: what it costs where it stands, nothing where a child stands for it, an absence elsewhere
            let mut fixed: FxHashMap<NeuronId, (f64, f64)> = FxHashMap::default();
            for r in 0..s {
                let mut here: FxHashMap<NeuronId, f64> = FxHashMap::default();
                for a in arounds[r].iter().filter(|a| a.off == off) {
                    let e = here.entry(a.neuron).or_insert(0.0);
                    if a.cost > *e { *e = a.cost; }
                }
                for (n, c) in here { let e = fixed.entry(n).or_insert((0.0, 0.0)); e.0 += recs[r]; e.1 += c; }
            }
            let mut fx: Vec<(NeuronId, f64, f64)> = fixed.iter().map(|(n, (h, v))| (*n, *h, *v)).collect();
            fx.sort_by(|a, b| a.0.cmp(&b.0));
            for (n, held, cov) in &fx {
                let v = cov - (all_recs - held) - (nb_bits + ob);
                consider(v, Choice::Fixed(*n), true, &mut best, &mut best_plain);
            }
            // what stands here as itself in each use, for the variables: neuron and position
            let direct: Vec<Vec<(NeuronId, f64)>> = (0..s).map(|r| arounds[r].iter().filter(|a| a.off == off && !a.under).map(|a| (a.neuron, a.pos)).collect()).collect();
            // a class of the table: the position where a member stands
            for c in &self.classes {
                if c.members.len() < 2 { continue; }
                let mut cnt = 0;
                let mut cov = 0.0;
                let mut held = 0.0;
                for (r, here) in direct.iter().enumerate() {
                    if let Some((_, p)) = here.iter().filter(|(n, _)| c.has(*n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cnt += 1; cov += p; held += recs[r]; }
                }
                if cnt == 0 { continue; }
                let v = cov - (all_recs - held) - (self.which_class_bits() + ob);
                consider(v, Choice::Class(c.id), true, &mut best, &mut best_plain);
            }
            // a new class: members in order of count, never two that stood together in a use, each while what
            // it covers beats a neuron's entry; a member may be built over the function's own child, since a
            // variable is left open in what the child stands for (D36)
            let mut cands: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
            for here in &direct {
                let mut seen: Vec<NeuronId> = Vec::new();
                for (n, p) in here { if seen.contains(n) { continue; } seen.push(*n); let e = cands.entry(*n).or_insert((0, 0.0)); e.0 += 1; e.1 += p; }
            }
            let mut order: Vec<(NeuronId, usize, f64)> = cands.iter().map(|(n, (c, v))| (*n, *c, *v)).collect();
            order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut members: Vec<NeuronId> = Vec::new();
            for (n, _, cov) in &order {
                if direct.iter().any(|here| here.iter().any(|(m, _)| m == n) && here.iter().any(|(m, _)| members.contains(m))) { continue; }
                if *cov > nb_bits { members.push(*n); } else if !members.is_empty() { break; }
            }
            if members.len() >= 2 {
                members.sort();
                let exists = self.classes.iter().any(|c| { let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect(); m.sort(); m == members });
                if !exists {
                    let mut cov = 0.0;
                    let mut held = 0.0;
                    for (r, here) in direct.iter().enumerate() {
                        if let Some((_, p)) = here.iter().filter(|(n, _)| members.contains(n)).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) { cov += p; held += recs[r]; }
                    }
                    let entry = log2((classes_now + 1) as f64) + ob + members.len() as f64 * nb_bits;
                    let v = cov - (all_recs - held) - entry;
                    consider(v, Choice::NewClass(members.clone()), true, &mut best, &mut best_plain);
                }
            }
            // a parameter tying this place to its places already in: the position where the value stands here
            for k in 0..f.params.len() as u32 {
                let mut cnt = 0;
                let mut cov = 0.0;
                let mut miss = 0.0;
                for r in 0..s {
                    let value = param_value(k, off, r);
                    match value.and_then(|v| direct[r].iter().find(|(n, _)| *n == v)) {
                        Some((_, p)) => { cnt += 1; cov += p; }
                        None => miss += recs[r],
                    }
                }
                if cnt == 0 { continue; }
                let v = cov - miss - ob;
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
        // the new body; a new class joins the table
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
        let mut nf = Function { id: f.id, places: vec![(Place::Owner, [0; MAX_DIMS])], params: f.params.clone(), child: f.child, uses: f.uses.clone() };
        nf.places.extend(rest);
        nf.canonicalize();
        let changed = nf.places != f.places;
        if changed && ctx.trace { eprintln!("[collapse f{} #{}] p{} {} -> {} over {} uses", ctx.frame, self.id, fid, describe_function(&f), describe_function(&nf), rows.len()); }
        if changed { if let Some(g) = self.function_mut(fid) { g.places = nf.places; g.params = nf.params; } }
        self.record_params(fid);
        // the members of the classes it uses, decided wherever each is used
        let used: Vec<PatternId> = self.function(fid).map_or(Vec::new(), |g| g.places.iter().filter_map(|(p, _)| if let Place::Class(k) = p { Some(*k) } else { None }).collect());
        for k in used { self.decide_members(k, ctx); }
        changed
    }

    /// Re-decide a class's members over every place a function uses it, in that function's uses (D27, D41):
    /// a member stays, and a neuron that stood there joins, while what it covers beats a neuron's entry; two
    /// neurons that stood together in a use are never both members. A member may be built over the child of
    /// a function using the class: a variable is left open in what the child stands for (D36).
    pub(crate) fn decide_members(&mut self, k: PatternId, ctx: Ctx) {
        let users: Vec<Function> = self.functions.iter().filter(|f| f.uses_class(k)).cloned().collect();
        if users.is_empty() { return; }
        let index = self.row_index();
        let mut cands: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
        let mut together: Vec<Vec<NeuronId>> = Vec::new();
        for f in &users {
            let offs: Vec<Offset> = f.places.iter().filter(|(p, _)| *p == Place::Class(k)).map(|(_, o)| *o).collect();
            for a in &f.uses {
                let Some(&i) = index.get(a) else { continue };
                let row = &self.history[i];
                for off in &offs {
                    let mut seen: Vec<NeuronId> = Vec::new();
                    for nb in row.nbs.iter().filter(|nb| self.round(&nb.delta) == *off) {
                        if seen.contains(&nb.neuron) { continue; }
                        seen.push(nb.neuron);
                        let e = cands.entry(nb.neuron).or_insert((0, 0.0));
                        e.0 += 1;
                        e.1 += nb.pos;
                    }
                    together.push(seen);
                }
            }
        }
        let mut order: Vec<(NeuronId, usize, f64)> = cands.iter().map(|(n, (c, v))| (*n, *c, *v)).collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut members: Vec<Member> = Vec::new();
        for (n, cnt, cov) in &order {
            if together.iter().any(|t| t.contains(n) && t.iter().any(|m| members.iter().any(|x| x.neuron == *m))) { continue; }
            if *cov > ctx.neuron_bits { members.push(Member { neuron: *n, count: *cnt }); } else if !members.is_empty() { break; }
        }
        members.sort_by_key(|m| m.neuron);
        let changed = self.class(k).map_or(false, |c| c.members.iter().map(|m| m.neuron).collect::<Vec<_>>() != members.iter().map(|m| m.neuron).collect::<Vec<_>>());
        if let Some(c) = self.class_mut(k) { c.members = members; }
        if changed { for f in users { self.unsettled.insert(f.id); } }
    }
}
