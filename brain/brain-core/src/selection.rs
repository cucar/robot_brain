//! Selection (§8, R45–R48): the electorate, placements and resolution.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::machine::*;
use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── selection (R45–R48) ─────────────────────────────

    /// What one open activation places at the frame ahead (R46): base symbols its standing inferences expand to there.
    pub(crate) fn placements(&self, r: ActRef, ahead: Frame) -> Vec<(NeuronId, Pos, f64, f64)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut out = Vec::new();
        for inf in &a.inferences {
            if inf.struck || !self.thalamus[inf.target].alive { continue; }
            let mut at = a.pos;
            for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
            if (at[0] as i64) + self.spans[inf.target as usize][0] as i64 + 1 < ahead { continue; }
            let mut placed = Vec::new();
            self.expand(inf.target, at, &mut Vec::new(), &mut placed, None);
            for (n, p) in placed {
                if p[0] as i64 != ahead { continue; }
                let nn = &self.thalamus[n];
                let mut pp = p;
                for d in 1..MAX_DIMS { if nn.axes & (1 << d) == 0 { pp[d] = 0; } }
                out.push((n, pp, inf.strength, inf.estimate));
            }
        }
        out
    }

    pub(crate) fn select(&mut self) -> Vec<Winner> {
        let fno = self.frame_no;
        let ahead = fno + 1;
        self.match_all();
        // the electorate (R46, R48): uncovered open activations, and covered ones whose coverer has nothing
        let mut voters: Vec<(ActRef, Vec<(NeuronId, Pos, f64, f64)>)> = Vec::new();
        let mut stack: Vec<ActRef> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                let age = fno - a.pos[0] as i64;
                let open = (a.strong && age <= a.reach[0] as i64) || a.opened_at == fno;
                if !open || a.covered_at.is_some() { continue; }
                stack.push(ActRef { frame: f.no, index: i as u32 });
            }
        }
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        while let Some(r) = stack.pop() {
            if !seen.insert(r) { continue; }
            let p = self.placements(r, ahead);
            if !p.is_empty() { voters.push((r, p)); continue; }
            if let Some(a) = self.act(r) {
                for c in &a.covers {
                    if let Some(ca) = self.act(*c) {
                        let age = fno - c.frame;
                        let open = (ca.strong && age <= ca.reach[0] as i64) || ca.opened_at == fno;
                        if open { stack.push(*c); }
                    }
                }
            }
        }
        self.last_stats.voters = voters.len();
        if self.trace {
            for (r, placed) in &voters {
                let Some(a) = self.act(*r) else { continue };
                let acts: Vec<String> = placed.iter().filter(|(n, _, _, _)| self.thalamus[*n].kind == Kind::Action).map(|(n, _, s, e)| format!("#{} s={:.2} e={:+.2}", n, s, e)).collect();
                if !acts.is_empty() { eprintln!("[vote f{}] #{} at {:?} h{} : {}", fno, a.neuron, a.pos, a.height, acts.join(", ")); }
            }
        }
        self.last_stats.voters_above_base = voters.iter().filter(|(r, _)| self.act(*r).map_or(false, |a| a.height > 0)).count();
        // resolution (R47)
        struct Cand { share: f64, weighted: f64 }
        let mut table: FxHashMap<(DimId, Pos), FxHashMap<u32, Cand>> = FxHashMap::default();
        let mut voters_at: FxHashMap<(DimId, Pos), usize> = FxHashMap::default();
        for (_, placed) in &voters {
            let mut totals: FxHashMap<(DimId, Pos), f64> = FxHashMap::default();
            for (n, p, s, _) in placed {
                let nn = &self.thalamus[*n];
                let Some(d) = nn.dim else { continue };
                *totals.entry((d, *p)).or_default() += s;
            }
            for (k, _) in &totals { *voters_at.entry(*k).or_default() += 1; }
            for (n, p, s, e) in placed {
                let nn = &self.thalamus[*n];
                let Some(d) = nn.dim else { continue };
                let share = s / totals[&(d, *p)];
                let c = table.entry((d, *p)).or_default().entry(nn.bucket).or_insert(Cand { share: 0.0, weighted: 0.0 });
                c.share += share;
                c.weighted += share * e;
            }
        }
        let mut winners = Vec::new();
        let mut keys: Vec<(DimId, Pos)> = table.keys().copied().collect();
        keys.sort();
        for key in keys {
            let cands = &table[&key];
            let dim = &self.dims[key.0 as usize];
            let mut best: Option<(u32, f64, f64)> = None;
            for (b, c) in cands {
                let est = c.weighted / c.share;
                let better = match best {
                    None => true,
                    Some((bb, be, bs)) => match dim.kind {
                        Kind::Action => est > be || (est == be && (c.share > bs || (c.share == bs && *b < bb))),
                        _ => c.share > bs || (c.share == bs && *b < bb),
                    },
                };
                if better { best = Some((*b, est, c.share)); }
            }
            let Some((b, est, share)) = best else { continue };
            if dim.kind == Kind::Action && est < 0.0 && self.cfg.policy.action == ActionPolicy::Nothing { continue; }
            let mut pos = key.1;
            pos[0] = ahead as i32;
            winners.push(Winner { dim: key.0, bucket: b, pos, kind: dim.kind, estimate: est, share, voters: voters_at[&key] });
        }
        // execution activates what it expands where it won (R30): a pattern an inference's expansion passes
        // through fires weakly where its expansion completes at the frame ahead, provided the inference
        // stands and what it placed at that frame was selected
        let won: FxHashSet<(NeuronId, Pos)> = winners.iter().map(|w| (self.base_neuron(w.dim, w.bucket), w.pos)).collect();
        let mut weak_patterns: Vec<(NeuronId, Pos)> = Vec::new();
        for (r, _) in &voters {
            let Some(a) = self.act(*r) else { continue };
            for inf in &a.inferences {
                if inf.struck || self.thalamus[inf.target].kind != Kind::Inter || !self.thalamus[inf.target].alive { continue; }
                let mut at = a.pos;
                for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
                if at[0] as i64 > ahead { continue; }
                let mut base = Vec::new();
                let mut all = Vec::new();
                self.expand_into(inf.target, at, &mut Vec::new(), &mut base, Some(&mut all), None);
                let mut placed_ahead = base.iter().filter(|(_, p)| p[0] as i64 == ahead).map(|(n, p)| {
                    let mut pp = *p;
                    for d in 1..MAX_DIMS { if self.thalamus[*n].axes & (1 << d) == 0 { pp[d] = 0; } }
                    (*n, pp)
                }).peekable();
                if placed_ahead.peek().is_none() { continue; }
                if !placed_ahead.all(|x| won.contains(&x)) { continue; }
                if at[0] as i64 == ahead { weak_patterns.push((inf.target, at)); }
                for (n, p) in all { if self.thalamus[n].kind == Kind::Inter && p[0] as i64 == ahead { weak_patterns.push((n, p)); } }
            }
        }
        weak_patterns.sort();
        weak_patterns.dedup();
        self.pending_weak_patterns = weak_patterns;
        winners
    }

    // ───────────────────────────── the match (R49) ─────────────────────────────

    /// Read every standing inference that names a call neuron against the frames its line reaches back
    /// into, and strike it where the line would not pay there (R49).
    pub(crate) fn match_all(&mut self) {
        let fno = self.frame_no;
        // the apex of each open frame (D24), each activation with what it stands for: itself, and for a
        // child every base symbol its expansion places, at its share of the child's cost (D22, D47)
        let mut apex: FxHashMap<Frame, Vec<(NeuronId, Pos, f64, f64)>> = FxHashMap::default();
        for f in self.frames.iter() {
            let list = apex.entry(f.no).or_default();
            for b in f.acts.iter() {
                if b.gone || !Self::on_apex_at(b, f.no) || b.pos[0] as i64 != f.no { continue; }
                let (cost, pos) = if b.bound_at.is_some() { (b.cost - b.pos_bits, 0.0) } else { (b.cost, b.pos_bits) };
                list.push((b.neuron, b.pos, cost, pos));
                // a child holds what it stands for, and is worth nothing for it (D22, D47)
                if b.kind == Kind::Inter {
                    let mut base = Vec::new();
                    let mut all = Vec::new();
                    self.expand_into(b.neuron, b.pos, &mut Vec::new(), &mut base, Some(&mut all), b.fired_by.as_ref());
                    for (n, p) in all { list.push((n, p, 0.0, 0.0)); }
                }
            }
        }
        let mut strikes: Vec<(ActRef, usize)> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                for (k, inf) in a.inferences.iter().enumerate() {
                    if inf.struck { continue; }
                    let mut at = a.pos;
                    for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
                    if let Some(false) = self.match_one(inf.target, at, &apex, fno) { strikes.push((ActRef { frame: f.no, index: i as u32 }, k)); }
                }
            }
        }
        for (r, k) in strikes { if let Some(a) = self.act_mut(r) { a.inferences[k].struck = true; } }
    }

    /// The match of one inference completing at `at` (R49): the inferred call neuron's function, one
    /// expansion down, placed back from the completion; each fixed node in a frame that has happened credits
    /// what stands at its offset, credits that less an offset where it stands elsewhere within reach, and
    /// charges an absence where it stands nowhere; a variable is bound from what stands at its places that
    /// have happened, crediting their positions, a place holding nothing it could take charging an absence.
    /// Proposed when the credit exceeds the charges. None where nothing of the function has happened yet,
    /// or the target has no function to read.
    pub(crate) fn match_one(&self, target: NeuronId, at: Pos, apex: &FxHashMap<Frame, Vec<(NeuronId, Pos, f64, f64)>>, now: Frame) -> Option<bool> {
        let t = self.thalamus.get(target)?;
        if t.kind != Kind::Inter { return None; }
        let &(owner, pid) = t.parents.first()?;
        let f = self.line(owner, pid)?;
        let on = &self.thalamus[owner];
        let reach = self.reach_for(t.axes, &self.spans[target as usize]);
        let offset_bits = crate::offsets::offset_bits(&reach, &self.radii, t.axes);
        let absence = |n: NeuronId| self.record_for(t.axes, &self.spans.get(n as usize).copied().unwrap_or([0; MAX_DIMS])).0;
        let same_group = |a: &Pos, b: &Pos| round_delta(&sub(a, &at), &self.radii, t.axes) == round_delta(&sub(b, &at), &self.radii, t.axes);
        let mut credit = 0.0;
        let mut charge = 0.0;
        let mut judged = 0;
        for (place, off) in f.places.iter().skip(1) {
            let q = add(&at, off);
            if q[0] as i64 > now { continue; }
            match place {
                Place::Owner | Place::Param(_) => {}
                Place::Fixed(m) => {
                    judged += 1;
                    let here = apex.get(&(q[0] as i64)).and_then(|l| l.iter().filter(|(n, p, _, _)| n == m && same_group(p, &q)).map(|(_, _, c, _)| *c).fold(None, |b: Option<f64>, c| Some(b.map_or(c, |x| x.max(c)))));
                    if let Some(c) = here { credit += c; }
                    else if let Some((_, _, c, _)) = self.frames.iter().filter(|fr| (now - fr.no) <= reach[0] as i64).flat_map(|fr| apex.get(&fr.no).into_iter().flatten()).find(|(n, p, _, _)| n == m && (0..self.radii.len()).all(|d| (p[d] - q[d]).abs() <= reach[d])) { credit += c - offset_bits; }
                    else { charge += absence(*m); }
                }
                Place::Class(k) => {
                    let Some(c) = on.class(*k) else { continue };
                    judged += 1;
                    let found = apex.get(&(q[0] as i64)).and_then(|l| l.iter().find(|(n, p, _, _)| c.has(*n) && same_group(p, &q)));
                    match found { Some((_, _, _, pos)) => credit += pos, None => charge += absence(c.frequent().unwrap_or(owner)) }
                }
            }
        }
        // parameters: the value is the neuron at the earliest of the places that have happened (D44)
        for k in 0..f.params.len() as u32 {
            let mut places: Vec<Pos> = f.param_places(k).iter().map(|&qi| add(&at, &f.places[qi].1)).filter(|q| q[0] as i64 <= now).collect();
            places.sort();
            if places.is_empty() { continue; }
            judged += 1;
            let value = places.iter().find_map(|q| apex.get(&(q[0] as i64)).and_then(|l| l.iter().find(|(_, p, _, _)| same_group(p, q)).map(|(n, _, _, _)| *n)));
            for q in &places {
                match value.and_then(|v| apex.get(&(q[0] as i64)).and_then(|l| l.iter().find(|(n, p, _, _)| *n == v && same_group(p, q)))) {
                    Some((_, _, _, pos)) => credit += pos,
                    None => charge += absence(value.unwrap_or(owner)),
                }
            }
        }
        if judged == 0 { return None; }
        Some(credit > charge)
    }

}
