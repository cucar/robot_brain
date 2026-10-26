//! Selection (§8, R45–R48): the electorate, placements and resolution.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::machine::*;
use crate::function::*;
use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── selection (R45–R48) ─────────────────────────────

    /// What one open activation places at the frame ahead (R46): base symbols its standing inferences expand to there.
    pub(crate) fn placements(&self, r: ActRef, ahead: Frame, memo: &mut FxHashMap<(NeuronId, Pos), Vec<(NeuronId, Pos)>>) -> Vec<(NeuronId, Pos, f64, f64)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut out = Vec::new();
        for inf in &a.inferences {
            if inf.struck || !self.thalamus[inf.target].alive { continue; }
            let mut at = a.pos;
            for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
            // an expansion places back from where it completes, so one completed already places nothing ahead
            if (at[0] as i64) < ahead { continue; }
            let placed = memo.entry((inf.target, at)).or_insert_with(|| {
                let mut placed = Vec::new();
                self.expand_from(inf.target, at, ahead, &mut Vec::new(), &mut placed, None);
                placed
            }).clone();
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
        let t = std::time::Instant::now();
        self.match_all();
        self.timings[8] += t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
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
        let mut memo: FxHashMap<(NeuronId, Pos), Vec<(NeuronId, Pos)>> = FxHashMap::default();
        while let Some(r) = stack.pop() {
            if !seen.insert(r) { continue; }
            let p = self.placements(r, ahead, &mut memo);
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
        self.timings[9] += t.elapsed().as_secs_f64() * 1000.0;
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
        let t = std::time::Instant::now();
        let won: FxHashSet<(NeuronId, Pos)> = winners.iter().map(|w| (self.base_neuron(w.dim, w.bucket), w.pos)).collect();
        let mut weak_patterns: Vec<(NeuronId, Pos)> = Vec::new();
        for (r, _) in &voters {
            let Some(a) = self.act(*r) else { continue };
            for inf in &a.inferences {
                if inf.struck || self.thalamus[inf.target].kind != Kind::Inter || !self.thalamus[inf.target].alive { continue; }
                let mut at = a.pos;
                for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
                if at[0] as i64 != ahead { continue; }
                let mut base = Vec::new();
                let mut all = Vec::new();
                self.expand_from(inf.target, at, ahead, &mut Vec::new(), &mut base, Some(&mut all));
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
        self.timings[10] += t.elapsed().as_secs_f64() * 1000.0;
        winners
    }

    // ───────────────────────────── the match (R49) ─────────────────────────────

    /// Read every standing inference that names an interneuron against the frames its function reaches back
    /// into, and strike it where the function would not pay there (R49).
    pub(crate) fn match_all(&mut self) {
        let fno = self.frame_no;
        // the apex of each open frame (D49), each activation with what it costs, and for an interneuron every
        // base symbol it stands for, at nothing (D22, D47)
        let mut apex: FxHashMap<Frame, Vec<(NeuronId, Pos, f64)>> = FxHashMap::default();
        for f in self.frames.iter() {
            for (i, b) in f.acts.iter().enumerate() {
                if b.gone || b.neuron == NeuronId::MAX || !Self::on_apex_at(b, f.no) || b.pos[0] as i64 != f.no { continue; }
                apex.entry(f.no).or_default().push((b.neuron, b.pos, b.cost));
                if b.kind == Kind::Inter {
                    let mut base = Vec::new();
                    self.descend(ActRef { frame: f.no, index: i as u32 }, &mut Vec::new(), &mut base, None);
                    for r in base { if let Some(x) = self.act(r) { apex.entry(x.pos[0] as i64).or_default().push((x.neuron, x.pos, 0.0)); } }
                }
            }
        }
        let mut strikes: Vec<(ActRef, usize)> = Vec::new();
        let mut memo: FxHashMap<(NeuronId, Pos), Option<bool>> = FxHashMap::default();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                for (k, inf) in a.inferences.iter().enumerate() {
                    if inf.struck || self.thalamus.get(inf.target).map_or(true, |t| t.kind != Kind::Inter) { continue; }
                    let at = add(&a.pos, &inf.offset);
                    let verdict = *memo.entry((inf.target, at)).or_insert_with(|| self.match_one(inf.target, at, &apex, fno));
                    if let Some(false) = verdict { strikes.push((ActRef { frame: f.no, index: i as u32 }, k)); }
                }
            }
        }
        for (r, k) in strikes { if let Some(a) = self.act_mut(r) { a.inferences[k].struck = true; } }
    }

    /// The match of one inference completing at `at` (R49): the inferred interneuron's function, one expansion
    /// down, placed back from where its call would stand. A node in a frame that has happened credits what
    /// stands at its offset, credits that less an offset where it stands elsewhere within reach, and charges its
    /// relation where it stands nowhere; a variable credits the positions of what it binds. No record is
    /// charged. Proposed when the credit exceeds the charges; None where nothing of it has happened yet.
    pub(crate) fn match_one(&self, target: NeuronId, at: Pos, apex: &FxHashMap<Frame, Vec<(NeuronId, Pos, f64)>>, now: Frame) -> Option<bool> {
        let t = self.thalamus.get(target)?;
        if t.kind != Kind::Inter { return None; }
        let f = self.func_of(target)?;
        let reach = self.reach_for(t.axes, &self.spans[target as usize]);
        let anchor = add(&at, &Self::anchor_offset(&f));
        let same = |p: &Pos, q: &Pos, d: usize| -> bool { round_offset(p[d] - anchor[d], self.radii[d]) == round_offset(q[d] - anchor[d], self.radii[d]) };
        let mut credit = 0.0;
        let mut charge = 0.0;
        let mut judged = 0;
        let mut values: FxHashMap<u8, NeuronId> = FxHashMap::default();
        let mut nodes: Vec<(Node, Pos, usize)> = vec![(Node::Fixed(f.anchor), anchor, 0)];
        for r in &f.rels { let mut q = anchor; q[r.dim as usize] += r.off; nodes.push((r.node, q, r.dim as usize)); }
        for (node, q, d) in nodes {
            if q[0] as i64 > now { continue; }
            judged += 1;
            let here = apex.get(&(q[0] as i64));
            match node {
                Node::Fixed(m) => {
                    if let Some(c) = here.and_then(|l| l.iter().filter(|(n, p, _)| *n == m && same(p, &q, d)).map(|(_, _, c)| *c).fold(None, |b: Option<f64>, c| Some(b.map_or(c, |x| x.max(c))))) { credit += c; }
                    else if let Some((_, _, c)) = self.frames.iter().filter(|fr| (now - fr.no) <= reach[0] as i64).flat_map(|fr| apex.get(&fr.no).into_iter().flatten()).find(|(n, p, _)| *n == m && (0..self.radii.len()).all(|e| (p[e] - q[e]).abs() <= reach[e])) { credit += c - 1.0; }
                    else { charge += 1.0; }
                }
                Node::Class(k) => {
                    let members = self.dict.class(k);
                    match here.and_then(|l| l.iter().find(|(n, p, _)| members.contains(n) && same(p, &q, d))) { Some((_, _, c)) => credit += (c - 1.0).max(0.0), None => charge += 1.0 }
                }
                Node::Param(p) => {
                    let found = here.and_then(|l| l.iter().find(|(n, pp, _)| same(pp, &q, d) && values.get(&p).map_or(true, |v| v == n)));
                    match found { Some((n, _, c)) => { values.insert(p, *n); credit += (c - 1.0).max(0.0); } None => charge += 1.0 }
                }
            }
        }
        if judged == 0 { return None; }
        Some(credit > charge)
    }
}
