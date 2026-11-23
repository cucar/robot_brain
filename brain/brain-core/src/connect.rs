/// The connect and infer calls (§7, R31, R33, R36, R37).

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::brain::*;
use crate::neuron::{ConnectionWrite, Nb};
use crate::thalamus::ConnectTask;
use crate::types::*;

impl Brain {
    // ───────────────────────────── connect (R31, R33) ─────────────────────────────

    /// The share of this frame's rewards a connection to `target` takes (R33): every reward that names no
    /// channel, and those that name the target's channel where it is a base action.
    fn share_reward(&self, rewards: &[Reward], target: NeuronId) -> f64 {
        let mut total = 0.0;
        for rw in rewards {
            if rw.channels.is_empty() { total += rw.reward; continue; }
            let t = &self.thalamus.registry[target];
            if t.kind == Kind::Action && t.channel.map_or(false, |ch| rw.channels.contains(&ch)) { total += rw.reward; }
        }
        total
    }

    /// The offset of a connection from an activation to one standing on the apex at `pos` (D25): the age in time,
    /// and the exact difference in every other dimension the two share; none beyond the holder's reach.
    fn make_connection_offset(&self, h: &Act, axes: Axes, pos: &Pos, age: u32) -> Option<Offset> {
        for d in h.axes.space() {
            let d = *d as usize;
            if axes.has(d) && !h.sees(d, pos[d] - h.pos[d]) { return None; }
        }
        Some(Pos::make(h.pos.len().max(pos.len()), |d| if d == 0 { age as i32 } else if h.axes.has(d) && axes.has(d) { pos[d] - h.pos[d] } else { 0 }))
    }

    /// List what stood strong on the apex of an earlier frame when that frame's connections were written (R31).
    fn list_stood(&self, frame: FrameNumber) -> Vec<(Sym, Kind, Axes, Pos)> {
        let Some(fr) = self.thalamus.frames.iter().find(|x| x.no == frame) else { return Vec::new() };
        // on the apex then: fired by that frame, and covered in a later frame or not at all
        fr.acts.iter().filter(|b| b.strong && !b.gone && b.neuron != NeuronId::MAX && b.opened_at <= frame && b.covered_at.map_or(true, |c| c > 0))
            .filter(|b| (0..3).any(|h| self.cfg.policy.future[h][b.kind as usize]))
            .map(|b| (b.sym, b.kind, b.axes, b.pos)).collect()
    }

    /// The connect call on every open activation past age 0 (§7.1, R31, R33): its row takes what stands on this
    /// frame's apex within its reach, as what followed it, covered or not (D18); one uncovered at this frame
    /// connects to every strong activation on this frame's apex within its reach that the policy admits; one
    /// covered this frame connects once more, to each call that covered it; and each takes this frame's reward,
    /// and the shares of earlier frames' at their distances.
    pub(crate) fn connect_board(&mut self) {
        if self.frozen { return; }
        let fno = self.frame_no;
        let span_default = self.max_reach_t.max(1) as u32;
        let apex: Vec<(ActRef, Sym, NeuronId, Kind, Axes, Pos, bool)> = self.thalamus.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate().map(move |(i, b)| (f.no, i, b)))
            .filter(|(_, _, b)| b.on_apex() && b.neuron != NeuronId::MAX).map(|(no, i, b)| (ActRef { frame: no, index: i as u32 }, b.sym, b.neuron, b.kind, b.axes, b.pos, b.strong)).collect();
        let rewards: Vec<Reward> = self.thalamus.frames.iter().find(|f| f.no == fno).map(|f| f.rewards.clone()).unwrap_or_default();
        // what stood strong on the apex of each earlier frame a reward that names no channels pays, as it stood at
        // that frame: what the activations open then connected to (R31, R33)
        let far = rewards.iter().filter(|rw| rw.channels.is_empty()).map(|rw| if rw.frames == 0 { span_default } else { rw.frames }).max().unwrap_or(0);
        let stood: Vec<Vec<(Sym, Kind, Axes, Pos)>> = (1..far).map(|d| self.list_stood(fno - d as i64)).collect();
        let mut writes: Vec<(NeuronId, Sym, Sym, Offset, f64)> = Vec::new();
        let mut shares: Vec<(NeuronId, Sym, Sym, Offset, f64)> = Vec::new();
        let mut followed: Vec<(NeuronId, ActRef, Vec<Nb>)> = Vec::new();
        for f in self.thalamus.frames.iter() {
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
                    let Some(o) = self.make_connection_offset(a, *baxes, bpos, age) else { continue };
                    nbs.push(Nb { act: *br, sym: *bs, off: o });
                }
                if !nbs.is_empty() { followed.push((a.neuron, me, nbs)); }
                let covered_before = a.covered_at.map_or(false, |c| c < age);
                if covered_before { continue; }
                // a call that took a lap this frame is the same call, its phase that much longer (D53): as it stood before
                // the lap it connects to what follows it like any activation, under the key of the phase it had then (D25)
                let took_lap = a.covered_at == Some(age) && self.grown.contains_key(&me);
                if a.covered_at == Some(age) && !took_lap {
                    // the last connection, to each call that covered it (D10)
                    for c in a.coverers.iter().filter_map(|c| self.find_act(*c)) {
                        // a call that grew by a lap is the same call: it connects to nothing of itself (D53)
                        if c.pos[0] as i64 != fno || c.neuron == a.neuron { continue; }
                        if c.strong && c.pos != a.pos && self.cfg.policy.future[a.kind as usize][c.kind as usize] {
                            if let Some(o) = self.make_connection_offset(a, c.axes, &c.pos, age) { writes.push((a.neuron, a.sym, c.sym, o, self.share_reward(&rewards, c.neuron))); }
                        }
                    }
                } else {
                    for (_, bs, bn, bk, baxes, bpos, strong) in &apex {
                        if !strong || !self.cfg.policy.future[a.kind as usize][*bk as usize] { continue; }
                        let Some(o) = self.make_connection_offset(a, *baxes, bpos, age) else { continue };
                        writes.push((a.neuron, a.sym, *bs, o, self.share_reward(&rewards, *bn)));
                    }
                }
                // shares for earlier frames (R33): at distance d, into the connections at offset age − d
                // an activation covered before this frame was passed over above, and takes none
                for rw in &rewards {
                    let s = if rw.frames == 0 { span_default } else { rw.frames };
                    for d in 1..s.min(age + 1) {
                        if a.covered_at.map_or(false, |c| c <= age - d) { continue; }
                        let share = rw.reward * (s - d) as f64 / s as f64;
                        // a reward that names no channels: every connection the activation wrote at that frame
                        if rw.channels.is_empty() {
                            let Some(then) = stood.get(d as usize - 1) else { continue };
                            for (bs, bk, baxes, bpos) in then {
                                if !self.cfg.policy.future[a.kind as usize][*bk as usize] { continue; }
                                let Some(o) = self.make_connection_offset(a, *baxes, bpos, age - d) else { continue };
                                shares.push((a.neuron, a.sym, *bs, o, share));
                            }
                            continue;
                        }
                        // one that names channels: the action that ran then in each of them
                        let Some(fr) = self.thalamus.frames.iter().find(|x| x.no == fno - d as i64) else { continue };
                        for (ch, ar) in &fr.actions_ran {
                            if !rw.channels.contains(ch) { continue; }
                            let Some(b) = self.find_act(*ar) else { continue };
                            if !self.cfg.policy.future[a.kind as usize][b.kind as usize] { continue; }
                            let Some(o) = self.make_connection_offset(a, b.axes, &b.pos, age - d) else { continue };
                            shares.push((a.neuron, a.sym, b.sym, o, share));
                        }
                    }
                }
            }
        }
        // a call's connections are keyed by its phase less its laps, and a call a connection names is named the same way (D25)
        let (writes, shares) = {
            let mut w2 = Vec::with_capacity(writes.len());
            for (h, o, t, off, s) in writes { let (o, t) = (self.thalamus.dict.drop_laps(o), self.thalamus.dict.drop_laps(t)); w2.push((h, o, t, off, s)); }
            let mut s2 = Vec::with_capacity(shares.len());
            for (h, o, t, off, s) in shares { let (o, t) = (self.thalamus.dict.drop_laps(o), self.thalamus.dict.drop_laps(t)); s2.push((h, o, t, off, s)); }
            (w2, s2)
        };
        // each holder's part of the call: what followed its rows, then its writes and then its shares, in order
        let mut tasks: FxHashMap<NeuronId, ConnectTask> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        let mut walks: FxHashMap<Sym, Arc<Vec<Vec<NeuronId>>>> = FxHashMap::default();
        let none: Arc<Vec<Vec<NeuronId>>> = Arc::new(Vec::new());
        for (n, me, nbs) in followed {
            if !tasks.contains_key(&n) { order.push(n); }
            tasks.entry(n).or_default().followed.push((me, nbs));
        }
        for (exposure, list) in [(true, writes), (false, shares)] {
            for (holder, own, target, off, share) in list {
                let walk = if share != 0.0 {
                    if !walks.contains_key(&target) { let w = Arc::new(self.read_walk(target)); walks.insert(target, w); }
                    walks[&target].clone()
                } else { none.clone() };
                if !tasks.contains_key(&holder) { order.push(holder); }
                tasks.entry(holder).or_default().writes.push(ConnectionWrite { own, target, off, share, exposure, walk });
            }
        }
        let tasks: Vec<(NeuronId, ConnectTask)> = order.into_iter().map(|n| { let t = tasks.remove(&n).unwrap(); (n, t) }).collect();
        self.thalamus.connect_neurons(tasks);
    }

    /// R37's order of trial: per base action dimension the target acted in, its base actions in declared order; a
    /// negative estimate wires the first one with no connection at that offset.
    pub(crate) fn read_walk(&self, target: Sym) -> Vec<Vec<NeuronId>> {
        let mut dims: Vec<DimensionId> = Vec::new();
        let t = &self.thalamus.registry[self.thalamus.dict.find_neuron_of(target)];
        if t.kind == Kind::Action { dims.push(t.dim.unwrap()); }
        else if t.kind == Kind::Inter {
            let mut expanded = Vec::new();
            self.expand_from(target, Pos::zero(), FrameNumber::MIN, false, &mut Vec::new(), &mut expanded, None);
            for (n, _) in expanded {
                let nn = &self.thalamus.registry[n];
                if nn.kind == Kind::Action { if let Some(d) = nn.dim { if !dims.contains(&d) { dims.push(d); } } }
            }
        }
        dims.into_iter().map(|d| { let dim = &self.dims[d as usize]; (0..dim.resolution).map(|b| dim.first + b).collect() }).collect()
    }

    // ───────────────────────────── infer (R36) ─────────────────────────────

    /// The infer call at age 0 (§7.2): each activation of this frame reads its neuron's connections.
    pub(crate) fn infer_board(&mut self) {
        let fno = self.frame_no;
        let mut reads: Vec<(FrameNumber, usize, NeuronId, Sym)> = Vec::new();
        for f in self.thalamus.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                if a.opened_at != fno || a.gone || a.neuron == NeuronId::MAX { continue; }
                let mine = self.thalamus.dict.read_without_laps(a.sym);
                reads.push((f.no, i, a.neuron, mine));
            }
        }
        let read = self.thalamus.read_connections(reads.iter().map(|r| (r.2, r.3)).collect());
        for ((no, i, _, _), conns) in reads.into_iter().zip(read) {
            let infs = conns.into_iter().map(|(target, offset, strength, estimate)| Inference { target, offset, strength, estimate, struck: false }).collect();
            if let Some(f) = self.find_frame_mut(no) { f.acts[i].inferences = infs; }
        }
    }
}
