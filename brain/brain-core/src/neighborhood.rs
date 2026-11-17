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

    /// The activations at the ends of the relations a call covers, over every record it grew from, each as the
    /// call it stands as now (D26, D53).
    pub(crate) fn covered_ends(&self, c: ActRef) -> FxHashSet<ActRef> {
        let mut recs = vec![c];
        for old in self.grown.keys() { if self.latest(*old) == c { recs.push(*old); } }
        let mut out: FxHashSet<ActRef> = FxHashSet::default();
        for r in recs { if let Some(es) = self.call_edges.get(&r) { for &(x, y) in es { out.insert(self.latest(x)); out.insert(self.latest(y)); } } }
        out
    }

    /// What one activation observes (D5, D7, D26). A base activation: what stands on the board within one position of
    /// it, in this frame and the one before, in every activation dimension the two share: an activation that is not
    /// covered as itself, and at a covered position each call that landed on the activation there (D4); nothing
    /// through a call. A call: the calls that landed on an activation it landed on, and the other ends of the
    /// relations lifted to it from the activations it covered (D26, D10, D55); nothing it stands for. The policy admits each (D50). Each
    /// at its exact offset from the observer (D6).
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let mut out: Vec<Nb> = Vec::new();
        if !a.on_apex() { return out; }
        let fno = self.frame_no;
        let under = self.under(a);
        let fired = |x: &Act| !x.gone && self.stands(x, fno) && !(x.opened_at == fno && x.level > a.level);
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let push = |r: ActRef, out: &mut Vec<Nb>, seen: &mut FxHashSet<ActRef>| {
            if r == me || under.contains(&r) || !seen.insert(r) { return; }
            let Some(x) = self.act(r) else { return };
            if !self.cfg.policy.past[a.kind as usize][x.kind as usize] { return; }
            out.push(Nb::seen(r, x.sym, offset_between(a, x), r));
        };
        if a.covers.is_empty() {
            for f in self.frames.iter() {
                if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
                for (i, c) in f.acts.iter().enumerate() {
                    if c.neuron == NeuronId::MAX || !c.covers.is_empty() { continue; }
                    let shared = a.axes & c.axes;
                    let mut within = true;
                    for d in 1..MAX_DIMS {
                        if shared & (1 << d) == 0 { continue; }
                        if (c.pos[d] - a.pos[d]).abs() > a.reach[d] { within = false; break; }
                    }
                    if !within { continue; }
                    let cr = ActRef { frame: f.no, index: i as u32 };
                    if cr == me { continue; }
                    let mut there = Vec::new();
                    if c.covered_at.is_none() { if fired(c) { there.push(cr); } } else { self.standing_at(cr, a.level, &mut there, 0); }
                    for r in there { push(r, &mut out, &mut seen); }
                }
            }
        } else {
            // the calls that landed on an activation it landed on; a call that grew by laps is one call (D26, D53)
            let mine: FxHashSet<ActRef> = a.covers.iter().map(|c| self.latest(*c)).collect();
            for f in self.frames.iter() {
                for (i, x) in f.acts.iter().enumerate() {
                    if x.covers.is_empty() || x.covered_at.is_some() || !fired(x) { continue; }
                    let xr = ActRef { frame: f.no, index: i as u32 };
                    if xr == me || under.contains(&xr) { continue; }
                    let theirs: Vec<ActRef> = x.covers.iter().map(|c| self.latest(*c)).collect();
                    // a call standing over this one, before or after it grew, is no neighbor of it
                    if theirs.contains(&me) || mine.contains(&xr) || self.under(x).contains(&me) { continue; }
                    if !theirs.iter().any(|c| mine.contains(c)) { continue; }
                    push(xr, &mut out, &mut seen);
                }
            }
            // the other ends of the relations lifted to it: an activation as itself, or at a covered one the calls on it (D10)
            if let Some(ends) = self.taken.get(&me) {
                for &e in ends {
                    let Some(x) = self.act(e) else { continue };
                    let mut there = Vec::new();
                    if x.covered_at.is_none() { if fired(x) { there.push(e); } } else { self.standing_at(e, a.level, &mut there, 0); }
                    for r in there { push(r, &mut out, &mut seen); }
                }
            }
        }
        out.sort_by(|x, y| (x.off, x.sym, x.act).cmp(&(y.off, y.sym, y.act)));
        out
    }
}
