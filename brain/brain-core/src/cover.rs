/// Covering (D24, D28): an activation a call has landed on is covered once its covered relations of one kind reach
/// that kind's count; each relation of it no call covers is lifted to what stands over it (D55), and the row at the
/// other end takes the call (D10, D18). The kind each covered relation decides (D34), and the deadlocks refused where
/// they would close (D24).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::brain::*;
use crate::neuron::*;
use crate::recognize::edge;
use crate::types::*;

impl Brain {
    /// The cover check after a level (D24): each activation a call has landed on whose covered relations of one kind
    /// reach that kind's count (`cover_count`) is covered. From then it is out of the graph; the rows that saw it keep
    /// it as they saw it and take nothing for its calls (D10, D18).
    pub(crate) fn cover_board(&mut self) {
        let fno = self.frame_no;
        // covered: its covered relations of one kind reach that kind's count (D24)
        let mut newly: Vec<(ActRef, u32)> = Vec::new();
        for f in self.thalamus.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                if a.gone || a.covered_at.is_some() || a.coverers.is_empty() || a.neuron == NeuronId::MAX { continue; }
                let me = ActRef { frame: f.no, index: i as u32 };
                if self.reaches_cover_count(me) { newly.push((me, (fno - a.pos[0] as i64).max(0) as u32)); }
            }
        }
        for (me, age) in newly {
            // a row holds what stood on the board when it was written: it takes nothing for the call (D10, D18)
            if let Some(a) = self.find_act_mut(me) { a.covered_at = Some(age); }
            self.lift_relations(me);
        }
        self.flush_lifts();
    }

    /// Covering inside the election (D28, D24, R23): of the activations a fit just taken landed on, each whose covered
    /// relations of one kind now reach that kind's count (`cover_count`) is covered from this round: off the board for the rounds after,
    /// each relation of its own that no call covers lifted to the calls that landed on it. Returns what it covered.
    pub(crate) fn cover_in_election(&mut self, cands: &[ActRef]) -> Vec<ActRef> {
        let fno = self.frame_no;
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let mut newly: Vec<(ActRef, u32)> = Vec::new();
        for &me in cands {
            if !seen.insert(me) { continue; }
            let Some(a) = self.find_act(me) else { continue };
            if a.gone || a.covered_at.is_some() || a.coverers.is_empty() || a.neuron == NeuronId::MAX { continue; }
            let age = (fno - a.pos[0] as i64).max(0) as u32;
            if self.reaches_cover_count(me) { newly.push((me, age)); }
        }
        let mut out = Vec::new();
        for (me, age) in newly {
            if let Some(a) = self.find_act_mut(me) { a.covered_at = Some(age); }
            self.lift_relations(me);
            out.push(me);
        }
        out
    }

    /// The covered relations of an activation, one per other end, at their offsets as the board relations have them.
    pub(crate) fn list_covered_offsets(&self, me: ActRef) -> Vec<(ActRef, Offset)> {
        let mut out: Vec<(ActRef, Offset)> = Vec::new();
        let Some(a) = self.find_act(me) else { return out };
        let Some(others) = self.cov_inc.get(&me) else { return out };
        for &x in others {
            if out.iter().any(|(o, _)| *o == x) { continue; }
            let Some(xa) = self.find_act(x) else { continue };
            if xa.gone { continue; }
            if let Some(br) = a.rels.iter().find(|br| br.other == x) { out.push((x, br.off)); continue; }
            if let Some(br) = xa.rels.iter().find(|br| br.other == me) { out.push((x, br.off.negated())); }
        }
        out
    }

    /// Covering counts the two kinds of relation apart (D24, D34). Inside the frame (its offset in time zero, in space
    /// or at one position): one covered relation, so a call inside the frame covers every activation it landed on, at
    /// a relation it covered, as it is taken. In time: one covered relation to an earlier frame and one to a later,
    /// one to a later at the run's first frame. A relation of each kind never make a count together. Returns whether
    /// it was covered by one relation alone.
    pub(crate) fn reaches_cover_count(&self, me: ActRef) -> bool {
        let rels = self.list_covered_offsets(me);
        if rels.iter().any(|(_, o)| o[0] == 0) { return true; }
        let (hn, hp) = (rels.iter().any(|(_, o)| o[0] < 0), rels.iter().any(|(_, o)| o[0] > 0));
        (hn || hp) && (hn || !self.has_layout_side(me, 0, -1)) && (hp || !self.has_layout_side(me, 0, 1))
    }

    /// The records that stand for the same call as `r` after every lap it took (D53): the latest, and each it grew
    /// from.
    pub(crate) fn read_records(&self, r: ActRef) -> Vec<ActRef> {
        let l = self.follow_laps(r);
        let mut out = vec![l];
        for k in self.grown.keys() { if *k != l && self.follow_laps(*k) == l { out.push(*k); } }
        out
    }

    /// Make the task by which row `h` takes `nb` as a neighbor at offset `o` where it does not hold it there yet (D10):
    /// the row is found under its own record, or for a call that took a lap, under the record its row was written for
    /// (D18, D53). None where `h` has no activation.
    pub(crate) fn make_lift(&self, h: ActRef, nb: ActRef, o: Offset) -> Option<(NeuronId, LiftTask)> {
        let a = self.find_act(h)?;
        let sym = self.find_act(nb)?.sym;
        let row_records = self.read_records(h).into_iter().filter(|r| r.frame == h.frame).collect();
        Some((a.neuron, LiftTask { row: h, row_records, call_records: self.read_records(nb), call: Nb { act: nb, sym, off: o } }))
    }

    /// Hand the lifts collected since the last flush to the neurons in one batch (D10, D55); a lift whose row is not
    /// written yet waits for it (`lift_pending`).
    pub(crate) fn flush_lifts(&mut self) {
        if self.lifts_due.is_empty() { return; }
        let due = std::mem::take(&mut self.lifts_due);
        let (tasks, pending): (Vec<(NeuronId, LiftTask)>, Vec<Option<(ActRef, ActRef, Offset)>>) = due.into_iter().map(|(n, t, p)| ((n, t), p)).unzip();
        let found = self.thalamus.write_lifts(tasks);
        for (found, pending) in found.into_iter().zip(pending) {
            let Some((x, c, off)) = pending else { continue };
            if !found { self.lift_pending.entry(x).or_default().push((c, off)); }
        }
    }

    /// A relation of a covered activation that no call covers is lifted, from then on, to each call that landed on it,
    /// at the offset from that call's position; where its other end is covered too, it is the relation between the
    /// calls on the two, one edge however many such relations run between them (D24, D10, D26).
    pub(crate) fn lift_relations(&mut self, me: ActRef) {
        let Some(a) = self.find_act(me).cloned() else { return };
        let mut rels: Vec<(ActRef, u32)> = a.rels.iter().map(|br| (br.other, br.strength)).collect();
        for f in self.thalamus.frames.iter() { for (j, b) in f.acts.iter().enumerate() { if b.gone { continue; } for br in &b.rels { if br.other == me { rels.push((ActRef { frame: f.no, index: j as u32 }, br.strength)); } } } }
        // an end that is covered is the call that stands over it: the coverers followed upward to each call no call
        // covers now (D55)
        let coverers: Vec<ActRef> = self.find_standing_over(&a.coverers);
        for (o, s) in rels {
            let e = edge(me, o);
            // covered by a call as it stands, or in any relation it was lifted to (D55, D24)
            if self.is_covered_in_any_form(e, 0) { continue; }
            // a refused relation is never lifted, at any level (D24)
            if self.refused.contains(&e) { continue; }
            let Some(oa) = self.find_act(o).cloned() else { continue };
            if oa.gone || coverers.contains(&o) { continue; }
            // lifted already when its other end was covered: now that both ends are, it is lifted to the relation
            // between the calls on the two, one relation per pair of calls, whichever end was covered first (D26)
            if self.lifted.contains(&e) && oa.covered_at.is_none() { continue; }
            self.lifted.insert(e);
            // where the other end is covered, it is lifted to the relation between the calls on the two
            let others: Vec<ActRef> = if oa.covered_at.is_some() { self.find_standing_over(&oa.coverers) } else { vec![o] };
            for &c in &coverers {
                for &x in &others {
                    if x == c { continue; }
                    let (Some(ca), Some(xa)) = (self.find_act(c).cloned(), self.find_act(x).cloned()) else { continue };
                    let v = self.taken.entry(c).or_default();
                    let in_taken = v.contains(&x);
                    if !in_taken { v.push(x); }
                    let off = offset_between(&xa, &ca);
                    { let v = self.lift_forms.entry(e).or_default(); let f = edge(x, c); if !v.contains(&f) { v.push(f); } }
                    let held = self.find_act(c).map_or(false, |cm| cm.rels.iter().any(|br| br.other == x));
                    // what the edge stands for, and whether it stood on the board before (D24)
                    {
                        let f = edge(x, c);
                        let first = !self.form_src.contains_key(&f);
                        if first && (in_taken || held || xa.rels.iter().any(|br| br.other == c)) { self.form_pre.insert(f); }
                        let v = self.form_src.entry(f).or_default(); if !v.contains(&e) { v.push(e); }
                    }
                    if let Some(xm) = self.find_act_mut(x) { if !held && !xm.rels.iter().any(|br| br.other == c) { xm.rels.push(BoardRel { other: c, off, strength: s }); } }
                    // the row at the relation's other end holds it too (D18, D55, R50): the row of what stands there on the
                    // board, an activation no call has landed on, as itself, or the call that had already landed on that
                    // end, the row of the activation under it taking nothing. Inside the frame it takes the call as a
                    // neighbor at the call's offset from it, keeping what it saw, one neighbor per call; a row not yet
                    // written, a call of this frame, holds it when it is written. The row of the call the relation is
                    // lifted to takes nothing for the lift, and in time a row takes only what the connect call hands it
                    if off[0] == 0 && !self.frozen {
                        let waits = xa.fired_by.is_some() && xa.opened_at == self.frame_no;
                        match self.make_lift(x, c, off) {
                            Some(task) => self.lifts_due.push((task.0, task.1, if waits { Some((x, c, off)) } else { None })),
                            None => if waits { self.lift_pending.entry(x).or_default().push((c, off)); },
                        }
                    }
                }
            }
        }
    }

    /// The calls that stand over these coverers: each followed upward through its own coverers to a call on the board
    /// that no call covers now, one of each (D55). A call that has gone (D40) or no longer stands (D9) is passed over:
    /// it is no end, and the climb goes on through what landed on it.
    pub(crate) fn find_standing_over(&self, from: &[ActRef]) -> Vec<ActRef> {
        let fno = self.frame_no;
        let mut out: Vec<ActRef> = Vec::new();
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let mut stack: Vec<ActRef> = from.iter().map(|c| self.follow_laps(*c)).collect();
        while let Some(c) = stack.pop() {
            if !seen.insert(c) { continue; }
            let Some(ca) = self.find_act(c) else { continue };
            if ca.gone || !self.stands(ca, fno) {
                stack.extend(ca.coverers.iter().map(|x| self.follow_laps(*x)));
                continue;
            }
            if ca.covered_at.is_none() || ca.coverers.is_empty() { if !out.contains(&c) { out.push(c); } continue; }
            stack.extend(ca.coverers.iter().map(|x| self.follow_laps(*x)));
        }
        out.sort();
        out
    }

    /// Whether a call covers a relation as it stands or in any relation it was lifted to (D55, D24).
    pub(crate) fn is_covered_in_any_form(&self, e: (ActRef, ActRef), depth: usize) -> bool {
        if self.covered_rel.contains(&e) { return true; }
        assert!(depth <= MAX_CALL_DEPTH, "a lifted relation was followed up through more than {} calls", MAX_CALL_DEPTH);
        self.lift_forms.get(&e).map_or(false, |v| v.iter().any(|f| self.is_covered_in_any_form(*f, depth + 1)))
    }

    /// Whether the layout has a point on one side of an activation in a dimension, within its reach, among what it
    /// may see: in time, nothing ends after it (D24, D3, D9).
    pub(crate) fn has_layout_side(&self, me: ActRef, d: usize, sign: i32) -> bool {
        let Some(a) = self.find_act(me) else { return false };
        if d == 0 && sign > 0 { return true; }
        // in time, a position where something has fired is a point of the layout for good: the layout ends before an
        // activation only at the real start of the run, where nothing has fired before it (D24, D3)
        if d == 0 { return self.first_frame.map_or(false, |f0| (a.lo[0] as i64) > f0); }
        let under = self.list_under(a);
        for f in self.thalamus.frames.iter() {
            let dt = f.no - a.pos[0] as i64;
            if dt > 0 || -dt > a.reach[0] as i64 { continue; }
            for (i, x) in f.acts.iter().enumerate() {
                let xr = ActRef { frame: f.no, index: i as u32 };
                if xr == me || x.gone || under.contains(&xr) || x.neuron == NeuronId::MAX { continue; }
                if !self.cfg.policy.past[a.kind as usize][x.kind as usize] { continue; }
                let off = offset_between(a, x);
                let mut within = true;
                for k in a.axes.space() { let k = *k as usize; if x.axes.has(k) && off[k].abs() > a.reach[k] { within = false; } }
                if within && off[d].signum() == sign { return true; }
            }
        }
        false
    }
}

// ───── the two kinds and the kind decided (D34, D24) ─────
impl Brain {
    /// Whether a relation between two activations is inside the frame: its two ends in one frame (D34).
    pub(crate) fn is_rel_inside(&self, a: ActRef, b: ActRef) -> Option<bool> { Some(self.find_act(a)?.pos[0] == self.find_act(b)?.pos[0]) }

    /// A relation of the other kind than the one an end's first covered relation decided is blocked (D24).
    pub(crate) fn is_blocked(&self, a: ActRef, b: ActRef) -> bool {
        let Some(k) = self.is_rel_inside(a, b) else { return false };
        self.kind_of.get(&a).map_or(false, |x| *x != k) || self.kind_of.get(&b).map_or(false, |x| *x != k)
    }

    /// The kind an activation's first covered relation decided, if any: true inside the frame.
    pub fn read_kind_decided(&self, a: ActRef) -> Option<bool> { self.kind_of.get(&a).copied() }

    /// A call covers a relation between the two (D24): each end whose kind is not yet decided takes the relation's
    /// kind, and its relations of the other kind are blocked from now.
    pub(crate) fn decide_kind(&mut self, a: ActRef, b: ActRef) {
        let Some(k) = self.is_rel_inside(a, b) else { return };
        for x in [a, b] {
            if self.kind_of.contains_key(&x) { continue; }
            self.kind_of.insert(x, k);
        }
    }
}

// ───── a deadlock broken where it would close: joined, refused, eliminated (D24, D55) ─────
impl Brain {
    /// The root of an activation's set of activations joined in this election.
    pub(crate) fn find_joined_root(&self, mut x: ActRef) -> ActRef { while let Some(p) = self.jparent.get(&x) { if *p == x { break; } x = *p; } x }



    /// The calls on the shortest chain of relations covered in this election from `a` to `b`, if they are joined.
    pub(crate) fn find_joined_chain(&self, a: ActRef, b: ActRef) -> Option<Vec<ActRef>> {
        if a == b || self.find_joined_root(a) != self.find_joined_root(b) { return None; }
        let mut prev: FxHashMap<ActRef, (ActRef, ActRef)> = FxHashMap::default();
        let mut q = std::collections::VecDeque::new();
        q.push_back(a);
        prev.insert(a, (a, a));
        while let Some(x) = q.pop_front() {
            if x == b { break; }
            for &(y, call, _) in self.joined.get(&x).map(|v| v.as_slice()).unwrap_or(&[]) {
                if prev.contains_key(&y) { continue; }
                prev.insert(y, (x, call));
                q.push_back(y);
            }
        }
        if !prev.contains_key(&b) { return None; }
        let mut calls: Vec<ActRef> = Vec::new();
        let mut x = b;
        while x != a { let (p, c) = prev[&x]; if !calls.contains(&c) { calls.push(c); } x = p; }
        Some(calls)
    }

    /// A refused relation that had already ended at a call: that edge leaves the board, and every edge it became
    /// above. An edge that also stands for a relation that is not refused, or stood on the board before, stays (D24).
    pub(crate) fn remove_forms(&mut self, e: (ActRef, ActRef), depth: usize) {
        assert!(depth <= MAX_CALL_DEPTH, "a lifted relation was followed up through more than {} calls", MAX_CALL_DEPTH);
        let forms = self.lift_forms.get(&e).cloned().unwrap_or_default();
        for f in forms {
            if self.refused.contains(&f) { continue; }
            let shared = self.form_pre.contains(&f) || self.form_src.get(&f).map_or(false, |v| v.iter().any(|s| *s != e && !self.refused.contains(s)));
            if shared { continue; }
            self.refused.insert(f);
            let (x, c) = f;
            if let Some(xm) = self.find_act_mut(x) { xm.rels.retain(|br| br.other != c); }
            if let Some(cm) = self.find_act_mut(c) { cm.rels.retain(|br| br.other != x); }
            if let Some(v) = self.taken.get_mut(&c) { v.retain(|y| *y != x); }
            if let Some(v) = self.taken.get_mut(&x) { v.retain(|y| *y != c); }
            self.remove_forms(f, depth + 1);
        }
    }

    /// At the start of a round of the election: every relation on the board whose two ends the calls taken so far at
    /// this level of this frame join is refused, inside the frame or in time, whether or not a fit offers it or a
    /// function holds it (D24, D55). Returns the ends of the relations refused now.
    pub(crate) fn mark_refused(&mut self, fits: &[crate::recognize::Fit]) -> Vec<ActRef> {
        let mut ends = Vec::new();
        let mut cands: Vec<(ActRef, ActRef)> = Vec::new();
        let keys: Vec<ActRef> = self.joined.keys().copied().collect();
        for x in &keys {
            let Some(xa) = self.find_act(*x) else { continue };
            if xa.gone { continue; }
            for br in &xa.rels { if self.joined.contains_key(&br.other) { cands.push((*x, br.other)); } }
        }
        for fit in fits { cands.extend(fit.edges.iter().copied()); }
        let mut seen: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        for (a, b) in cands {
            let e = edge(a, b);
            if !seen.insert(e) { continue; }
            if self.refused.contains(&e) || self.is_blocked(a, b) || self.covered_rel.contains(&e) { continue; }
            if self.find_joined_chain(a, b).is_none() { continue; }
            self.refused.insert(e);
            // where it had already ended at a call, that edge leaves the board, with every edge above it
            if self.lifted.contains(&e) { self.remove_forms(e, 0); }
            ends.push(a);
            ends.push(b);
        }
        ends
    }
}
