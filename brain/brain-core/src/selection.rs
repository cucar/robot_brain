//! Selection (§8, R45–R48): the electorate, placements and resolution.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── selection (R45–R48) ─────────────────────────────

    /// What one open activation places at the frame ahead (R46): base symbols its standing inferences expand to there.
    pub(crate) fn placements(&self, r: ActRef, ahead: Frame) -> Vec<(NeuronId, Pos, f64, f64)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut out = Vec::new();
        for inf in &a.inferences {
            if !self.thalamus[inf.target].alive { continue; }
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
        winners
    }

}
