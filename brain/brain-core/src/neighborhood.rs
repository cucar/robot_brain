//! Neighborhoods (D5, D7, D26): what an activation observes.

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    /// Whether an activation is open at the frame being processed: within its window (D9), and not a weak one of
    /// an earlier frame (D40).
    pub(crate) fn is_open(&self, a: &Act, at: Frame) -> bool {
        if a.gone { return false; }
        let age = at - a.pos[0] as i64;
        if !a.strong && age > 0 { return false; }
        age >= 0 && age <= a.reach[0] as i64
    }

    /// What one activation observes (D5, D7): every activation in its frame or an earlier one, within its reach in
    /// every activation dimension the two share, not at its own place, both of them on the apex, that the policy
    /// admits. Each at its exact offset (D6).
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let fno = self.frame_no;
        let mut out = Vec::new();
        if !a.on_apex() { return out; }
        for f in self.frames.iter() {
            if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me || !b.on_apex() || b.neuron == NeuronId::MAX || !self.is_open(b, fno) { continue; }
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
                out.push(Nb { act: r, neuron: b.neuron, off });
            }
        }
        out.sort_by(|x, y| (x.off, x.neuron, x.act).cmp(&(y.off, y.neuron, y.act)));
        out
    }
}
