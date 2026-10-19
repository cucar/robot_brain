//! Neighborhoods (D5, D7): what an activation observes.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── neighborhoods (D5, D7) ─────────────────────────────

    /// What one activation observes: every activation within reach on every shared axis, not later
    /// than it, that fired before it and that the past table admits (D5), each with whether it stood
    /// on the apex (the recorded flag that seeds and credit read).
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let mut out = Vec::new();
        for f in self.frames.iter() {
            if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me { continue; }
                // fired before me: an earlier frame, or an earlier wave of the same frame
                if !(b.opened_at < a.opened_at || (b.opened_at == a.opened_at && b.wave < a.wave)) { continue; }
                if !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                let shared = a.axes & b.axes;
                let mut delta = [0i32; MAX_DIMS];
                delta[0] = b.pos[0] - a.pos[0];
                let mut ok = true;
                for d in 1..self.radii.len() {
                    if shared & (1 << d) == 0 { continue; }
                    delta[d] = b.pos[d] - a.pos[d];
                    if delta[d].abs() > a.reach[d] { ok = false; break; }
                }
                if !ok { continue; }
                // the apex alone (D18)
                if b.covered_at.is_some() { continue; }
                let value = self.neurons[b.neuron as usize].value;
                // what a child stands for, as base symbols at their offsets from the owner (R28), where a
                // pattern's parent may be matched by the child covering it
                let covers: Vec<(NeuronId, Pos)> = if b.kind == Kind::Inter {
                    let mut placed = Vec::new();
                    self.expand(b.neuron, b.pos, &mut Vec::new(), &mut placed, b.fired_by.as_ref());
                    placed.into_iter().filter(|(_, p)| p[0] <= a.pos[0]).map(|(n, p)| (n, sub(&p, &a.pos))).collect()
                } else { Vec::new() };
                out.push(Nb { act: r, neuron: b.neuron, delta, cost: b.cost, value, covers });
            }
        }
        out
    }

}
