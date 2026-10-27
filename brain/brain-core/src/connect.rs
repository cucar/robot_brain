//! The connect and infer calls (§7, R31, R33, R36, R37).


use crate::machine::*;
use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── connect (R31, R33) ─────────────────────────────

    pub(crate) fn reward_share_frame0(&self, rewards: &[Reward], target: NeuronId) -> f64 {
        let mut total = 0.0;
        for rw in rewards {
            if rw.channels.is_empty() { total += rw.reward; continue; }
            let t = &self.thalamus[target];
            if t.kind == Kind::Action && t.channel.map_or(false, |ch| rw.channels.contains(&ch)) { total += rw.reward; }
        }
        total
    }

    /// Whether an activation stood on the apex of frame `at`.
    pub(crate) fn on_apex_at(a: &Act, at: Frame) -> bool {
        a.pos[0] as i64 <= at && a.covered_at.map_or(true, |c| c as i64 > at - a.pos[0] as i64)
    }

    pub(crate) fn connect_all(&mut self) {
        let fno = self.frame_no;
        // a test writes nothing: every activation is simply caught up
        if self.frozen {
            for f in self.frames.iter_mut() { for a in f.acts.iter_mut() { a.caught_up = fno; } }
            return;
        }
        let span_default = self.max_reach_t.max(1);
        let mut writes: Vec<(NeuronId, NeuronId, Offset, f64)> = Vec::new();
        let frames: Vec<(Frame, usize)> = self.frames.iter().map(|f| (f.no, f.acts.len())).collect();
        for (f_no, count) in frames {
            for i in 0..count {
                let r = ActRef { frame: f_no, index: i as u32 };
                let a = self.act(r).unwrap().clone();
                if !a.strong { continue; }
                // the frames to hand it: the ones since it last caught up, through this one
                let from = a.caught_up + 1;
                for at in from..=fno {
                    let age = (at - a.pos[0] as i64) as i64;
                    if age <= 0 || age > a.reach[0] as i64 { continue; }
                    let age = age as u32;
                    let at_rewards: Vec<Reward> = self.frames.iter().find(|f| f.no == at).map(|f| f.rewards.clone()).unwrap_or_default();
                    // the apex of frame `at` within reach, as the coverage set reconstructs it
                    let covered_then = a.covered_at.map_or(false, |c| (c as i64) < age as i64);
                    if !covered_then {
                        let apex: Vec<(NeuronId, Kind, u8, Pos)> = self.frames.iter().filter(|f| f.no == at).flat_map(|f| f.acts.iter()).filter(|b| b.strong && Self::on_apex_at(b, at) && b.pos[0] as i64 == at).map(|b| (b.neuron, b.kind, b.axes, b.pos)).collect();
                        for (bn, bk, baxes, bpos) in apex {
                            if !self.cfg.policy.future[a.kind as usize][bk as usize] { continue; }
                            let Some(off) = self.offset_between(&a, baxes, &bpos, age) else { continue };
                            let share = self.reward_share_frame0(&at_rewards, bn);
                            writes.push((a.neuron, bn, off, share));
                        }
                    }
                    // the last connection, to the call that covered it, and again to each call that took it back (D10, R23)
                    let mut coverers: Vec<ActRef> = Vec::new();
                    if a.covered_at == Some(age) { if let Some(c) = a.coverer { coverers.push(c); } }
                    for (g, c) in &a.retaken { if *g == age { coverers.push(*c); } }
                    for cr in coverers {
                        if let Some(cov) = self.act(cr).cloned() {
                            if cov.strong && cov.pos != a.pos && cov.neuron != NeuronId::MAX && self.cfg.policy.future[a.kind as usize][cov.kind as usize] {
                                if let Some(off) = self.offset_between(&a, cov.axes, &cov.pos, age) {
                                    let share = self.reward_share_frame0(&at_rewards, cov.neuron);
                                    writes.push((a.neuron, cov.neuron, off, share));
                                }
                            }
                        }
                    }
                    // shares for earlier frames (R33)
                    for rw in &at_rewards {
                        let s = if rw.frames == 0 { span_default as u32 } else { rw.frames };
                        let maxd = s.min(age + 1);
                        for d in 1..maxd {
                            if a.covered_at.map_or(false, |c| c <= age - d) { continue; }
                            let share = rw.reward * (s - d) as f64 / s as f64;
                            let f_then = at - d as i64;
                            let Some(fr) = self.frames.iter().find(|f| f.no == f_then) else { continue };
                            let ran: Vec<(ChannelId, ActRef)> = fr.actions_ran.clone();
                            for (ch, ar) in ran {
                                if !rw.channels.is_empty() && !rw.channels.contains(&ch) { continue; }
                                let Some(b) = self.act(ar).cloned() else { continue };
                                if !self.cfg.policy.future[a.kind as usize][b.kind as usize] { continue; }
                                let Some(off) = self.offset_between(&a, b.axes, &b.pos, age - d) else { continue };
                                if self.thalamus[a.neuron].connections.contains_key(&(b.neuron, off)) {
                                    self.add_earnings(a.neuron, b.neuron, off, share);
                                }
                            }
                        }
                    }
                }
                if let Some(am) = self.act_mut(r) { am.caught_up = fno; }
            }
        }
        for (holder, target, off, share) in writes {
            let c = self.thalamus[holder].connections.entry((target, off)).or_default();
            c.strength += 1.0;
            if share != 0.0 { self.add_earnings(holder, target, off, share); }
        }
    }

    /// The offset from an open activation to an apex activation: its age in time and the rounded
    /// distance in every other shared axis; none when out of reach.
    pub(crate) fn offset_between(&self, a: &Act, baxes: u8, bpos: &Pos, age: u32) -> Option<Offset> {
        let shared = a.axes & baxes;
        let mut delta = [0i32; MAX_DIMS];
        delta[0] = age as i32;
        for d in 1..self.radii.len() {
            if shared & (1 << d) == 0 { continue; }
            delta[d] = bpos[d] - a.pos[d];
            if delta[d].abs() > a.reach[d] { return None; }
        }
        Some(round_delta(&delta, &self.radii, shared | 1))
    }

    /// A share of reward into one connection (R33); where its estimate turns negative, the walk (R37).
    pub(crate) fn add_earnings(&mut self, holder: NeuronId, target: NeuronId, off: Offset, share: f64) {
        let (before, after) = {
            let c = self.thalamus[holder].connections.entry((target, off)).or_default();
            let before = c.estimate();
            c.earnings += share;
            (before, c.estimate())
        };
        if after < 0.0 && before >= 0.0 { self.walk(holder, target, off); }
    }

    /// R37: a negative estimate wires the next untried base action, in each dimension the target acted in.
    pub(crate) fn walk(&mut self, holder: NeuronId, target: NeuronId, off: Offset) {
        let mut dims: Vec<DimId> = Vec::new();
        let t = &self.thalamus[target];
        if t.kind == Kind::Action { dims.push(t.dim.unwrap()); }
        else if t.kind == Kind::Inter {
            let mut placed = Vec::new();
            self.expand(target, [0; MAX_DIMS], &mut Vec::new(), &mut placed, None);
            for (n, _) in placed {
                let nn = &self.thalamus[n];
                if nn.kind == Kind::Action { if let Some(d) = nn.dim { if !dims.contains(&d) { dims.push(d); } } }
            }
        }
        for d in dims {
            let dim = &self.dims[d as usize];
            for b in 0..dim.resolution {
                let n = dim.first + b;
                let key = (n, off);
                if !self.thalamus[holder].connections.contains_key(&key) {
                    self.thalamus[holder].connections.insert(key, Conn { strength: 1.0, earnings: 0.0 });
                    break;
                }
            }
        }
    }

    // ───────────────────────────── infer (R36) ─────────────────────────────

    pub(crate) fn infer_all(&mut self) {
        let fno = self.frame_no;
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() {
                if a.opened_at != fno { continue; }
                let n = &self.thalamus[a.neuron];
                a.inferences = n.connections.iter().map(|((t, o), c)| Inference { target: *t, offset: *o, strength: c.strength, estimate: c.estimate(), struck: false }).collect();
            }
        }
    }

}
