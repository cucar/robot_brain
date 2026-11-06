//! Neighborhoods (D5, D7, D26): what an activation observes.

use rustc_hash::FxHashSet;

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

    /// Everything a call stands for, at any depth (D5, R28).
    pub(crate) fn under(&self, a: &Act) -> FxHashSet<ActRef> {
        let mut under: FxHashSet<ActRef> = FxHashSet::default();
        let mut stack: Vec<ActRef> = a.covers.clone();
        while let Some(r) = stack.pop() {
            if !under.insert(r) { continue; }
            if let Some(c) = self.act(r) { stack.extend(c.covers.iter().copied()); }
        }
        under
    }

    /// What stands where an activation stood, for an activation of `level` at the frame being processed (D5): the
    /// activation itself while it is on the apex; where calls have landed on it, each call that stands on the apex
    /// there, at any depth; and nothing that has not fired yet.
    fn standing_at(&self, r: ActRef, level: u32, out: &mut Vec<ActRef>, depth: usize) {
        let fno = self.frame_no;
        let Some(b) = self.act(r) else { return };
        if b.gone || (b.opened_at == fno && b.level > level) || !self.stands(b, fno) { return; }
        if b.on_apex() { out.push(r); return; }
        if depth > 256 { return; }
        for c in &b.coverers { self.standing_at(*c, level, out, depth + 1); }
    }

    /// What one activation observes (D5, D7): what stands in its frame or the one before within one position of
    /// it, in every activation dimension the two share, other than itself, on the board (D9); for a call, what
    /// stands that way beside each activation it stands for at the base, and nothing it stands for (D4, D5). Where
    /// calls have landed on an activation, each call is the neighbor there, in its pose (D5, D10). The policy
    /// admits each (D50). Each at its exact offset from the observer (D6).
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let mut out: Vec<Nb> = Vec::new();
        if !a.on_apex() { return out; }
        let under = self.under(a);
        // the activations at the base it stands for, or itself
        let bases: Vec<ActRef> = if a.covers.is_empty() { vec![me] } else {
            let mut v: Vec<ActRef> = under.iter().copied().filter(|r| self.act(*r).map_or(false, |x| x.covers.is_empty())).collect();
            v.sort();
            v
        };
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        for br in bases {
            let Some(b) = self.act(br) else { continue };
            for f in self.frames.iter() {
                if f.no > b.pos[0] as i64 || (b.pos[0] as i64 - f.no) > b.reach[0] as i64 { continue; }
                for (i, c) in f.acts.iter().enumerate() {
                    if c.neuron == NeuronId::MAX || !c.covers.is_empty() { continue; }
                    let shared = b.axes & c.axes;
                    let mut within = true;
                    for d in 1..MAX_DIMS {
                        if shared & (1 << d) == 0 { continue; }
                        if (c.pos[d] - b.pos[d]).abs() > b.reach[d] { within = false; break; }
                    }
                    if !within { continue; }
                    let cr = ActRef { frame: f.no, index: i as u32 };
                    if cr == me || under.contains(&cr) { continue; }
                    let mut there = Vec::new();
                    self.standing_at(cr, a.level, &mut there, 0);
                    for r in there {
                        if r == me || under.contains(&r) || !seen.insert(r) { continue; }
                        let Some(x) = self.act(r) else { continue };
                        if !self.cfg.policy.past[a.kind as usize][x.kind as usize] { continue; }
                        // only the activation itself is excluded: two at one position are neighbors at offset zero
                        let off = offset_between(a, x);
                        out.push(Nb { act: r, sym: x.sym, off, base: r });
                    }
                }
            }
        }
        out.sort_by(|x, y| (x.off, x.sym, x.act).cmp(&(y.off, y.sym, y.act)));
        out
    }
}
