//! The connect and infer calls (§7, R31, R33, R36, R37).

use crate::machine::*;
use crate::neuron::{Conn, Nb};
use crate::types::*;

impl Machine {
    // ───────────────────────────── connect (R31, R33) ─────────────────────────────

    /// The share of this frame's rewards a connection to `target` takes (R33): every reward that names no
    /// channel, and those that name the target's channel where it is a base action.
    fn share_now(&self, rewards: &[Reward], target: NeuronId) -> f64 {
        let mut total = 0.0;
        for rw in rewards {
            if rw.channels.is_empty() { total += rw.reward; continue; }
            let t = &self.thalamus[target];
            if t.kind == Kind::Action && t.channel.map_or(false, |ch| rw.channels.contains(&ch)) { total += rw.reward; }
        }
        total
    }

    /// The offset of a connection from an activation to one standing on the apex at `pos` (D25): the age in time,
    /// and the exact difference in every other dimension the two share; none beyond the holder's reach.
    fn conn_offset(&self, h: &Act, axes: u8, pos: &Pos, age: u32) -> Option<Offset> {
        let mut o = [0i32; MAX_DIMS];
        o[0] = age as i32;
        let shared = h.axes & axes;
        for d in 1..MAX_DIMS {
            if shared & (1 << d) == 0 { continue; }
            o[d] = pos[d] - h.pos[d];
            if o[d].abs() > h.reach[d] { return None; }
        }
        Some(o)
    }

    /// The connect call on every open activation past age 0 (§7.1, R31, R33): its row takes what stands on this
    /// frame's apex within its reach, as what followed it, covered or not (D18); one uncovered at this frame
    /// connects to every strong activation on this frame's apex within its reach that the policy admits; one
    /// covered this frame connects once more, to each call that covered it; and each takes this frame's reward,
    /// and the shares of earlier frames' at their distances.
    pub(crate) fn connect_all(&mut self) {
        if self.frozen { return; }
        let fno = self.frame_no;
        let span_default = self.max_reach_t.max(1) as u32;
        let apex: Vec<(ActRef, Sym, NeuronId, Kind, u8, Pos, bool)> = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate().map(move |(i, b)| (f.no, i, b)))
            .filter(|(_, _, b)| b.on_apex() && b.neuron != NeuronId::MAX).map(|(no, i, b)| (ActRef { frame: no, index: i as u32 }, b.sym, b.neuron, b.kind, b.axes, b.pos, b.strong)).collect();
        let rewards: Vec<Reward> = self.frames.iter().find(|f| f.no == fno).map(|f| f.rewards.clone()).unwrap_or_default();
        let mut writes: Vec<(NeuronId, Sym, Sym, Offset, f64)> = Vec::new();
        let mut shares: Vec<(NeuronId, Sym, Sym, Offset, f64)> = Vec::new();
        let mut followed: Vec<(NeuronId, ActRef, Vec<Nb>)> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                if !a.strong || a.gone || a.neuron == NeuronId::MAX { continue; }
                let age = fno - a.pos[0] as i64;
                if age <= 0 || age > a.reach[0] as i64 { continue; }
                let age = age as u32;
                // the row takes what followed (D18)
                let me = ActRef { frame: f.no, index: i as u32 };
                let mut nbs = Vec::new();
                for (br, bs, _, bk, baxes, bpos, _) in &apex {
                    if !self.cfg.policy.past[a.kind as usize][*bk as usize] { continue; }
                    let Some(o) = self.conn_offset(a, *baxes, bpos, age) else { continue };
                    nbs.push(Nb { act: *br, sym: *bs, off: o, base: *br });
                }
                if !nbs.is_empty() { followed.push((a.neuron, me, nbs)); }
                let covered_before = a.covered_at.map_or(false, |c| c < age);
                if covered_before { continue; }
                if a.covered_at == Some(age) {
                    // the last connection, to each call that covered it (D10)
                    for c in a.coverers.iter().filter_map(|c| self.act(*c)) {
                        if c.pos[0] as i64 != fno { continue; }
                        if c.strong && c.pos != a.pos && self.cfg.policy.future[a.kind as usize][c.kind as usize] {
                            if let Some(o) = self.conn_offset(a, c.axes, &c.pos, age) { writes.push((a.neuron, a.sym, c.sym, o, self.share_now(&rewards, c.neuron))); }
                        }
                    }
                } else {
                    for (_, bs, bn, bk, baxes, bpos, strong) in &apex {
                        if !strong || !self.cfg.policy.future[a.kind as usize][*bk as usize] { continue; }
                        let Some(o) = self.conn_offset(a, *baxes, bpos, age) else { continue };
                        writes.push((a.neuron, a.sym, *bs, o, self.share_now(&rewards, *bn)));
                    }
                }
                // shares for earlier frames (R33): at distance d, into the connection at offset age − d to the
                // action that ran then, where the activation was uncovered at that frame
                for rw in &rewards {
                    let s = if rw.frames == 0 { span_default } else { rw.frames };
                    for d in 1..s.min(age + 1) {
                        if a.covered_at.map_or(false, |c| c <= age - d) { continue; }
                        let share = rw.reward * (s - d) as f64 / s as f64;
                        let Some(fr) = self.frames.iter().find(|x| x.no == fno - d as i64) else { continue };
                        for (ch, ar) in &fr.actions_ran {
                            if !rw.channels.is_empty() && !rw.channels.contains(ch) { continue; }
                            let Some(b) = self.act(*ar) else { continue };
                            if !self.cfg.policy.future[a.kind as usize][b.kind as usize] { continue; }
                            let Some(o) = self.conn_offset(a, b.axes, &b.pos, age - d) else { continue };
                            shares.push((a.neuron, a.sym, b.sym, o, share));
                        }
                    }
                }
            }
        }
        for (n, me, nbs) in followed {
            if !self.thalamus[n].add_followed(me, &nbs) { continue; }
            for nb in &nbs { if let Some(b) = self.act_mut(nb.act) { b.seen_by.push((n, me)); } }
        }
        for (holder, own, target, off, share) in writes {
            let c = self.thalamus[holder].connections.entry((own, target, off)).or_default();
            c.strength += 1.0;
            if share != 0.0 { self.add_earnings(holder, own, target, off, share); }
        }
        for (holder, own, target, off, share) in shares {
            if self.thalamus[holder].connections.contains_key(&(own, target, off)) { self.add_earnings(holder, own, target, off, share); }
        }
    }

    /// A share of reward into one connection (R33); where its estimate turns negative, the walk (R37).
    pub(crate) fn add_earnings(&mut self, holder: NeuronId, own: Sym, target: Sym, off: Offset, share: f64) {
        let (before, after) = {
            let c = self.thalamus[holder].connections.entry((own, target, off)).or_default();
            let before = c.estimate();
            c.earnings += share;
            (before, c.estimate())
        };
        if after < 0.0 && before >= 0.0 { self.walk(holder, own, target, off); }
    }

    /// R37: a negative estimate wires the next untried base action, in each dimension the target acted in.
    pub(crate) fn walk(&mut self, holder: NeuronId, own: Sym, target: Sym, off: Offset) {
        let mut dims: Vec<DimId> = Vec::new();
        let t = &self.thalamus[self.dict.neuron_of(target)];
        if t.kind == Kind::Action { dims.push(t.dim.unwrap()); }
        else if t.kind == Kind::Inter {
            let mut expanded = Vec::new();
            self.expand_from(target, [0; MAX_DIMS], i64::MIN, false, &mut Vec::new(), &mut expanded, None);
            for (n, _) in expanded {
                let nn = &self.thalamus[n];
                if nn.kind == Kind::Action { if let Some(d) = nn.dim { if !dims.contains(&d) { dims.push(d); } } }
            }
        }
        for d in dims {
            let dim = &self.dims[d as usize];
            for b in 0..dim.resolution {
                let n = dim.first + b;
                let key = (own, n, off);
                if !self.thalamus[holder].connections.contains_key(&key) {
                    self.thalamus[holder].connections.insert(key, Conn { strength: 1.0, earnings: 0.0 });
                    break;
                }
            }
        }
    }

    // ───────────────────────────── infer (R36) ─────────────────────────────

    /// The infer call at age 0 (§7.2): each activation of this frame reads its neuron's connections.
    pub(crate) fn infer_all(&mut self) {
        let fno = self.frame_no;
        let mut reads: Vec<(Frame, usize, Vec<Inference>)> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                if a.opened_at != fno || a.gone || a.neuron == NeuronId::MAX { continue; }
                let n = &self.thalamus[a.neuron];
                let infs = n.connections.iter().filter(|((own, _, _), _)| *own == a.sym).map(|((_, t, o), c)| Inference { target: *t, offset: *o, strength: c.strength, estimate: c.estimate(), struck: false }).collect();
                reads.push((f.no, i, infs));
            }
        }
        for (no, i, infs) in reads { if let Some(f) = self.frame_mut(no) { f.acts[i].inferences = infs; } }
    }
}
