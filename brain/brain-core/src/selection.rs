//! Selection (§8, R45–R49): the standing inferences, what they expand to at the frame ahead, the match, the
//! electorate and resolution.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::machine::*;
use crate::types::*;

fn add(a: &Pos, b: &Offset) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] += b[d]; } p }

impl Machine {
    /// What one open activation infers at the frame ahead (R46): the base symbols its standing inferences expand to
    /// there, each with the strength and estimate of its connection.
    pub(crate) fn inferred_ahead(&self, r: ActRef, ahead: Frame, memo: &mut FxHashMap<(NeuronId, Pos), Vec<(NeuronId, Pos)>>) -> Vec<(NeuronId, Pos, f64, f64)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut out = Vec::new();
        for inf in &a.inferences {
            if inf.struck || !self.thalamus.get(self.dict.neuron_of(inf.target)).map_or(false, |t| t.alive) { continue; }
            let at = add(&a.pos, &inf.offset);
            // an expansion reads back from where it completes, so one completed already names nothing ahead
            if (at[0] as i64) < ahead { continue; }
            let expanded = memo.entry((inf.target, at)).or_insert_with(|| {
                let mut expanded = Vec::new();
                self.expand_from(inf.target, at, ahead, true, &mut Vec::new(), &mut expanded, None);
                expanded
            }).clone();
            for (n, p) in expanded {
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
                if !self.is_open(a, fno) || a.covered_at.is_some() || a.neuron == NeuronId::MAX { continue; }
                stack.push(ActRef { frame: f.no, index: i as u32 });
            }
        }
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let mut memo: FxHashMap<(NeuronId, Pos), Vec<(NeuronId, Pos)>> = FxHashMap::default();
        while let Some(r) = stack.pop() {
            if !seen.insert(r) { continue; }
            let p = self.inferred_ahead(r, ahead, &mut memo);
            if !p.is_empty() { voters.push((r, p)); continue; }
            if let Some(a) = self.act(r) {
                for c in &a.covers {
                    if let Some(ca) = self.act(*c) { if self.is_open(ca, fno) { stack.push(*c); } }
                }
            }
        }
        self.last_stats.voters = voters.len();
        self.last_stats.voters_above_base = voters.iter().filter(|(r, _)| self.act(*r).map_or(false, |a| a.height > 0)).count();
        if self.trace {
            for (r, named) in &voters {
                let Some(a) = self.act(*r) else { continue };
                let s: Vec<String> = named.iter().map(|(n, p, st, e)| format!("#{}@{} s={:.0} e={:+.2}", n, p[0], st, e)).collect();
                eprintln!("[vote f{}] #{} at {:?} : {}", fno, a.neuron, a.pos, s.join(", "));
            }
        }
        // resolution (R47)
        struct Cand { share: f64, weighted: f64 }
        let mut table: FxHashMap<(DimId, Pos), FxHashMap<u32, Cand>> = FxHashMap::default();
        let mut voters_at: FxHashMap<(DimId, Pos), usize> = FxHashMap::default();
        for (_, named) in &voters {
            let mut totals: FxHashMap<(DimId, Pos), f64> = FxHashMap::default();
            for (n, p, s, _) in named {
                let Some(d) = self.thalamus[*n].dim else { continue };
                *totals.entry((d, *p)).or_default() += s;
            }
            for k in totals.keys() { *voters_at.entry(*k).or_default() += 1; }
            for (n, p, s, e) in named {
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

    // ───────────────────────────── the match (R49) ─────────────────────────────

    /// Whether a node stood at a position that has happened (R49): its neuron in its pose there, or a call whose pose
    /// starts at it (D54).
    pub(crate) fn stood(&self, n: Sym, q: &Pos) -> bool {
        let Some(f) = self.frames.iter().find(|f| f.no == q[0] as i64) else { return false };
        f.acts.iter().any(|a| {
            if a.gone { return false; }
            let mut p = a.pos;
            let mut qq = *q;
            for d in 1..MAX_DIMS { if a.axes & (1 << d) == 0 { p[d] = 0; qq[d] = 0; } }
            p == qq && (a.sym == n || self.dict.start_of(a.sym) == Some(n))
        })
    }

    /// Read every standing inference that names an interneuron against the frames its function reaches back into,
    /// and strike it where the function would not pay there (R49): each node that has happened credits where its
    /// neuron stands at its position and charges a failure where it does not; proposed while the credit exceeds the
    /// charges, or where nothing of it has happened yet.
    pub(crate) fn match_all(&mut self) {
        let fno = self.frame_no;
        let mut strikes: Vec<(ActRef, usize)> = Vec::new();
        let mut memo: FxHashMap<(NeuronId, Pos), bool> = FxHashMap::default();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                for (k, inf) in a.inferences.iter().enumerate() {
                    if inf.struck || self.thalamus.get(self.dict.neuron_of(inf.target)).map_or(true, |t| t.kind != Kind::Inter) { continue; }
                    let at = add(&a.pos, &inf.offset);
                    let ok = *memo.entry((inf.target, at)).or_insert_with(|| {
                        let (mut credit, mut charge, mut judged) = (0, 0, 0);
                        for (m, q) in self.nodes_at(inf.target, at) {
                            if q[0] as i64 > fno { continue; }
                            judged += 1;
                            if self.stood(m, &q) { credit += 1; } else { charge += 1; }
                        }
                        judged == 0 || credit > charge
                    });
                    if !ok { strikes.push((ActRef { frame: f.no, index: i as u32 }, k)); }
                }
            }
        }
        for (r, k) in strikes { if let Some(a) = self.act_mut(r) { a.inferences[k].struck = true; } }
    }
}
