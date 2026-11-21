/// The board: the open activations (D9), what fired them, the relations their neurons returned at them, what they
/// infer (R45), and what each observes (D5, D7, D26).

use rustc_hash::FxHashSet;

use crate::brain::Brain;
use crate::dictionary::FuncId;
use crate::neuron::*;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: FrameNumber,
    pub index: u32,
}

/// A relation the activation's neuron returned at it (R21): the other activation, at its offset from this one,
/// with its strength.
#[derive(Clone, Copy, Debug)]
pub struct BoardRel {
    pub other: ActRef,
    pub off: Offset,
    pub strength: u32,
}

/// Which function fired an interneuron's activation, and which of its relations did not come true (§6.4).
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub func: FuncId,
    pub failed: Vec<usize>,
}

/// One standing inference (R45): a connection as it was read at age 0.
#[derive(Clone, Copy, Debug)]
pub struct Inference {
    /// The pose the connection names (D25).
    pub target: Sym,
    pub offset: Offset,
    pub strength: f64,
    pub estimate: f64,
    /// Struck by the match (R49): it no longer stands.
    pub struck: bool,
}

/// An open activation as the machine holds it (D9).
#[derive(Clone, Debug)]
pub struct Act {
    pub neuron: NeuronId,
    /// Its pose (D8): its neuron's id for a base activation, the interneuron in the call's phase for a call.
    pub sym: Sym,
    pub kind: Kind,
    pub axes: u8,
    pub pos: Pos,
    pub strong: bool,
    pub reach: Pos,
    /// The box of positions it stands for in space (D4): its own position for a base activation, and for a call the
    /// union of what it covered.
    pub lo: Pos,
    pub hi: Pos,
    /// The age at which an accepted call first landed on it as a fixed node (D10, D24): from then it is off the apex.
    pub covered_at: Option<u32>,
    /// Every accepted call that landed on it as a fixed node (D24, D10).
    pub coverers: Vec<ActRef>,
    /// What its call covered (§6.4): the activations its fixed nodes landed on and the calls it landed on.
    pub covers: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    /// The relations its neuron returned at it (R21), the board's edges at this node (D55).
    pub rels: Vec<BoardRel>,
    pub inferences: Vec<Inference>,
    /// The frame it fired in.
    pub opened_at: FrameNumber,
    /// The level of its frame it fired at (R26): 0 for what the frame reported.
    pub level: u32,
    /// How many functions deep it stands, a diagnostic the machine never reads.
    pub height: u32,
    /// A weak activation the report replaced (D40): no longer there for anything.
    pub gone: bool,
}

impl Act {
    pub fn on_apex(&self) -> bool { self.covered_at.is_none() && !self.gone }

    /// Whether it sees an offset in dimension `d` (D4): in space, one position beyond the farthest of everything
    /// it stands for on that side; in time, its reach back, and the same number forward.
    pub fn sees(&self, d: usize, off: i32) -> bool {
        if d == 0 { return off.abs() <= self.reach[0]; }
        if off < 0 { -off <= self.pos[d] - self.lo[d] + 1 } else { off <= self.hi[d] - self.pos[d] + 1 }
    }
}

/// The offset from one activation to another in the dimensions both are laid out over, and time (D6).
pub fn offset_between(a: &Act, b: &Act) -> Offset {
    let shared = a.axes & b.axes;
    let mut o = [0i32; MAX_DIMS];
    o[0] = b.pos[0] - a.pos[0];
    for d in 1..MAX_DIMS { if shared & (1 << d) != 0 { o[d] = b.pos[d] - a.pos[d]; } }
    o
}

#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub no: FrameNumber,
    pub acts: Vec<Act>,
    /// The base actions that ran this frame, per channel (D24).
    pub actions_ran: Vec<(ChannelId, ActRef)>,
    pub rewards: Vec<Reward>,
}

impl Brain {
    /// Whether an activation is open, connecting, at the frame being processed: within its own reach in time (D9),
    /// and not a weak one of an earlier frame (D40).
    pub(crate) fn is_open(&self, a: &Act, at: FrameNumber) -> bool {
        if a.gone { return false; }
        let age = at - a.pos[0] as i64;
        if !a.strong && age > 0 { return false; }
        age >= 0 && age <= a.reach[0] as i64
    }

    /// Whether an activation stands on the board at the frame being processed (D9): through S, the machine's longest
    /// reach in time, for a fit to land on it and an interneuron whose reach extends to it to see it; not a weak
    /// one of an earlier frame (D40).
    pub(crate) fn stands(&self, a: &Act, at: FrameNumber) -> bool {
        if a.gone { return false; }
        let age = at - a.pos[0] as i64;
        if !a.strong && age > 0 { return false; }
        age >= 0 && age <= self.span_s as i64
    }

    /// Everything a call stands for, at any depth (D5, R28).
    pub(crate) fn list_under(&self, a: &Act) -> FxHashSet<ActRef> {
        let mut under: FxHashSet<ActRef> = FxHashSet::default();
        let mut stack: Vec<ActRef> = a.covers.clone();
        while let Some(r) = stack.pop() {
            if !under.insert(r) { continue; }
            if let Some(c) = self.find_act(r) { stack.extend(c.covers.iter().copied()); }
        }
        under
    }

    /// What stands where an activation stood, for an activation of `level` at the frame being processed (D5): the
    /// activation itself while it is on the apex; where calls have landed on it, each call that stands on the apex
    /// there, at any depth; and nothing that has not fired yet.
    fn standing_at(&self, r: ActRef, level: u32, out: &mut Vec<ActRef>, depth: usize) {
        let fno = self.frame_no;
        let Some(b) = self.find_act(r) else { return };
        if b.gone || (b.opened_at == fno && b.level > level) || !self.stands(b, fno) { return; }
        if b.on_apex() { out.push(r); return; }
        if depth > 256 { return; }
        for c in &b.coverers { self.standing_at(*c, level, out, depth + 1); }
    }

    /// What one activation observes (D5, D7, D26). A base activation: what stands on the board within one position of
    /// it, in this frame and the one before, in every activation dimension the two share: an activation that is not
    /// covered as itself, and at a covered position each call that landed on the activation there (D4); nothing
    /// through a call. A call: the calls that landed on an activation it landed on, and the other ends of the
    /// relations lifted to it from the activations it covered (D26, D10, D55); nothing it stands for. The policy admits each (D50). Each
    /// at its exact offset from the observer (D6).
    pub(crate) fn observe_neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.find_act(me) else { return vec![] };
        let mut out: Vec<Nb> = Vec::new();
        if !a.on_apex() { return out; }
        let fno = self.frame_no;
        let under = self.list_under(a);
        let fired = |x: &Act| !x.gone && self.stands(x, fno) && !(x.opened_at == fno && x.level > a.level);
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let push = |r: ActRef, out: &mut Vec<Nb>, seen: &mut FxHashSet<ActRef>| {
            if r == me || under.contains(&r) || !seen.insert(r) { return; }
            let Some(x) = self.find_act(r) else { return };
            if !self.cfg.policy.past[a.kind as usize][x.kind as usize] { return; }
            out.push(Nb { act: r, sym: x.sym, off: offset_between(a, x) });
        };
        if a.covers.is_empty() {
            for f in self.thalamus.frames.iter() {
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
            let mine: FxHashSet<ActRef> = a.covers.iter().map(|c| self.follow_laps(*c)).collect();
            for f in self.thalamus.frames.iter() {
                for (i, x) in f.acts.iter().enumerate() {
                    if x.covers.is_empty() || x.covered_at.is_some() || !fired(x) { continue; }
                    let xr = ActRef { frame: f.no, index: i as u32 };
                    if xr == me || under.contains(&xr) { continue; }
                    let theirs: Vec<ActRef> = x.covers.iter().map(|c| self.follow_laps(*c)).collect();
                    // a call standing over this one, before or after it grew, is no neighbor of it
                    if theirs.contains(&me) || mine.contains(&xr) || self.list_under(x).contains(&me) { continue; }
                    if !theirs.iter().any(|c| mine.contains(c)) { continue; }
                    push(xr, &mut out, &mut seen);
                }
            }
            // the other ends of the relations lifted to it: an activation as itself, or at a covered one the calls on it (D10)
            if let Some(ends) = self.taken.get(&me) {
                for &e in ends {
                    let Some(x) = self.find_act(e) else { continue };
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
