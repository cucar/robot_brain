//! Selection (§8, R45–R49): the standing inferences, what they expand to at the frame ahead, the match, the
//! electorate and resolution.

use rustc_hash::FxHashMap;

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
        // the electorate (R46, R48): an open activation is heard unless it is inhibited, a relation of its own covered by
        // a call (D24) that has an inference for the frame ahead; where every call that inhibits it has none, it is heard,
        // and so on down to the base. A call that took a lap is the same call (D53): what covered a relation of the
        // record it grew from still does, and the record it grew from is not heard apart from it
        let mut voters: Vec<(ActRef, Vec<(NeuronId, Pos, f64, f64)>)> = Vec::new();
        let mut memo: FxHashMap<(NeuronId, Pos), Vec<(NeuronId, Pos)>> = FxHashMap::default();
        let mut inhib: FxHashMap<ActRef, Vec<ActRef>> = FxHashMap::default();
        for (c, es) in &self.call_edges {
            let cl = self.latest(*c);
            if self.act(cl).map_or(true, |x| x.gone) { continue; }
            for &(x, y) in es {
                for e in [x, y] {
                    let e = self.latest(e);
                    if e == cl { continue; }
                    let v = inhib.entry(e).or_default();
                    if !v.contains(&cl) { v.push(cl); }
                }
            }
        }
        let mut cands: Vec<ActRef> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if !self.is_open(a, fno) || a.neuron == NeuronId::MAX || self.grown.contains_key(&r) { continue; }
                cands.push(r);
            }
        }
        // the calls over an activation: every call that covers a relation of its own, and every call over one of
        // those, at any height (D24); it is silent in a neuron dimension where any of them has an inference for the
        // frame ahead in that dimension, and heard in a neuron dimension where none has (R46, R48, D1)
        let mut dims_of: FxHashMap<ActRef, Vec<DimId>> = FxHashMap::default();
        for r in cands {
            let p = self.inferred_ahead(r, ahead, &mut memo);
            if p.is_empty() { continue; }
            let mut ov: Vec<ActRef> = Vec::new();
            let mut stack: Vec<ActRef> = inhib.get(&r).cloned().unwrap_or_default();
            while let Some(c) = stack.pop() {
                if c == r || ov.contains(&c) || ov.len() > 4096 { continue; }
                ov.push(c);
                if let Some(up) = inhib.get(&c) { stack.extend(up.iter().copied()); }
            }
            let mut silenced: Vec<(DimId, ActRef)> = Vec::new();
            for &c in &ov {
                if !dims_of.contains_key(&c) {
                    let mut v: Vec<DimId> = self.inferred_ahead(c, ahead, &mut memo).iter().filter_map(|(n, _, _, _)| self.thalamus[*n].dim).collect();
                    v.sort();
                    v.dedup();
                    dims_of.insert(c, v);
                }
                for d in &dims_of[&c] { if !silenced.iter().any(|x| x.0 == *d) { silenced.push((*d, c)); } }
            }
            if self.trace_vote {
                let dn = |d: DimId| format!("{}.{}", self.channels[self.dims[d as usize].channel as usize].name, self.dims[d as usize].name);
                let who = |x: ActRef| { let Some(a) = self.act(x) else { return "?".to_string() }; format!("{}@{} L{} {}", self.dict.sym_text(a.sym), x.frame, a.level, a.fired_by.as_ref().map_or("base".to_string(), |fb| format!("call of f{}", fb.func))) };
                let infs = |v: &Vec<(NeuronId, Pos, f64, f64)>| v.iter().filter(|(n, _, _, _)| self.thalamus[*n].dim.map_or(false, |d| self.dims[d as usize].kind == Kind::Action)).map(|(n, _, st, e)| format!("{}={} s={:.0} e={:+.4}", dn(self.thalamus[*n].dim.unwrap()), self.thalamus[*n].bucket, st, e)).collect::<Vec<_>>().join(", ");
                let act_dims: Vec<DimId> = p.iter().filter_map(|(n, _, _, _)| self.thalamus[*n].dim).filter(|d| self.dims[*d as usize].kind == Kind::Action).collect();
                if !act_dims.is_empty() {
                    let heard: Vec<String> = { let mut v: Vec<DimId> = act_dims.iter().copied().filter(|d| !silenced.iter().any(|x| x.0 == *d)).collect(); v.sort(); v.dedup(); v.into_iter().map(|d| dn(d)).collect() };
                    let sil: Vec<String> = { let mut v: Vec<(DimId, ActRef)> = silenced.iter().copied().filter(|(d, _)| act_dims.contains(d)).collect(); v.sort(); v.into_iter().map(|(d, c)| { let ci = self.inferred_ahead(c, ahead, &mut FxHashMap::default()); let ci: Vec<_> = ci.into_iter().filter(|(n, _, _, _)| self.thalamus[*n].dim == Some(d)).collect(); format!("{} by {} [its inference: {}]", dn(d), who(c), infs(&ci)) }).collect() };
                    eprintln!("[hear f{}] {} | infers: {} | heard for: [{}] | silenced: [{}]", fno, who(r), infs(&p), heard.join(", "), sil.join("; "));
                }
            }
            let kept: Vec<(NeuronId, Pos, f64, f64)> = p.into_iter().filter(|(n, _, _, _)| self.thalamus[*n].dim.map_or(true, |d| !silenced.iter().any(|x| x.0 == d))).collect();
            if !kept.is_empty() { voters.push((r, kept)); }
        }
        self.last_heard = voters.iter().map(|(r, _)| *r).collect();
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
        if self.trace_vote {
            for ((d, _), cands) in &table {
                if self.dims[*d as usize].kind != Kind::Action { continue; }
                let mut v: Vec<String> = cands.iter().map(|(b, c)| format!("bucket {} share {:.4} estimate {:+.6}", b, c.share, c.weighted / c.share)).collect(); v.sort();
                eprintln!("[resolve f{}] {}.{} | {}", fno, self.channels[self.dims[*d as usize].channel as usize].name, self.dims[*d as usize].name, v.join(" | "));
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

    /// Whether a node stood at a position that has happened (R49): its pose there, or a call whose phase
    /// starts at it (D54).
    pub(crate) fn stood(&self, n: Sym, q: &Pos) -> bool {
        let Some(f) = self.frames.iter().find(|f| f.no == q[0] as i64) else { return false };
        f.acts.iter().any(|a| {
            if a.gone { return false; }
            let mut p = a.pos;
            let mut qq = *q;
            for d in 1..MAX_DIMS { if a.axes & (1 << d) == 0 { p[d] = 0; qq[d] = 0; } }
            p == qq && (a.sym == n || self.dict.starts_at(a.sym, n))
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
