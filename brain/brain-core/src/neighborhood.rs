//! Neighborhoods (D5, D7, D26): what an activation observes.

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    /// Whether an activation is open, connecting, at the frame being processed: within its own reach in time (D9),
    /// and not a weak one of an earlier frame (D40).
    pub(crate) fn is_open(&self, a: &Act, at: Frame) -> bool {
        if a.gone { return false; }
        let age = at - a.pos[0] as i64;
        if !a.strong && age > 0 { return false; }
        age >= 0 && age <= a.reach[0] as i64
    }

    /// Whether an activation stands on the board at the frame being processed (D9): through S, the machine's longest
    /// reach in time, for a fit to land on it and an interneuron whose reach extends to it to see it; not a weak
    /// one of an earlier frame (D40).
    pub(crate) fn stands(&self, a: &Act, at: Frame) -> bool {
        if a.gone { return false; }
        let age = at - a.pos[0] as i64;
        if !a.strong && age > 0 { return false; }
        age >= 0 && age <= self.span_s as i64
    }

    /// What one activation observes (D5, D7): every activation in its frame or an earlier one, within its reach in
    /// every activation dimension the two share, not at its own place, standing on the board (D9), at an offset
    /// whose combination of dimensions no call has covered at it; not what the observer stands for, where it is a
    /// call; the call standing at an origin's place in place of that origin; that the policy admits. Each at its
    /// exact offset (D6).
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let fno = self.frame_no;
        let mut out = Vec::new();
        if !a.on_apex() { return out; }
        // a call never sees what it stands for, at any depth (D5)
        let mut under: Vec<ActRef> = Vec::new();
        let mut stack: Vec<ActRef> = a.covers.clone();
        while let Some(r) = stack.pop() {
            if under.contains(&r) { continue; }
            under.push(r);
            if let Some(c) = self.act(r) { stack.extend(c.covers.iter().copied()); }
        }
        for f in self.frames.iter() {
            if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me || b.neuron == NeuronId::MAX || !self.stands(b, fno) || under.contains(&r) { continue; }
                // where a call stands at its place, as the origin of its function, the call is the neighbor there
                if b.coverers.iter().filter_map(|c| self.act(*c)).any(|c| c.pos == b.pos && !c.gone) { continue; }
                // what fires above this level in this frame has not fired yet
                if b.opened_at == fno && b.level > a.level { continue; }
                if !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                let shared = a.axes & b.axes;
                let mut off = [0i32; MAX_DIMS];
                off[0] = b.pos[0] - a.pos[0];
                let mut within = true;
                for d in 1..MAX_DIMS {
                    if shared & (1 << d) == 0 { continue; }
                    off[d] = b.pos[d] - a.pos[d];
                    if off[d].abs() > a.reach[d] { within = false; }
                }
                if !within || off.iter().all(|x| *x == 0) { continue; }
                // a covered activation stays a neighbor at an offset whose combination no call has covered (D5)
                if b.covered_for(&off) { continue; }
                out.push(Nb { act: r, neuron: b.neuron, off });
            }
        }
        out.sort_by(|x, y| (x.off, x.neuron, x.act).cmp(&(y.off, y.neuron, y.act)));
        out
    }
}
