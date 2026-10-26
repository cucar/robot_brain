//! Expansion (R28) and what an interneuron stands for (D36): a recognized call is read back through what
//! actually fired, its covers and its binds; an inferred call through its function in the dictionary. Either
//! way a descent stops where it comes back to an interneuron already on its path, so a ring stands for one turn.

use crate::function::*;
use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// Round half toward the smaller position (D53).
pub fn round_half_down(x: f64) -> i32 { (x - 0.5).ceil() as i32 }

impl Machine {
    /// Descend from an activation through what its call covered and bound, down to base symbols (R28). The base
    /// activations go to `base`, every interneuron activation passed below the root to `inters`. Where the
    /// descent comes back to an interneuron already on its path it stops there (D36).
    pub(crate) fn descend(&self, r: ActRef, path: &mut Vec<NeuronId>, base: &mut Vec<ActRef>, mut inters: Option<&mut Vec<ActRef>>) {
        let Some(a) = self.act(r) else { return };
        if a.kind != Kind::Inter { if !base.contains(&r) { base.push(r); } return; }
        if a.neuron != NeuronId::MAX && path.contains(&a.neuron) { return; }
        if !path.is_empty() || a.neuron == NeuronId::MAX { if let Some(v) = inters.as_deref_mut() { if a.neuron != NeuronId::MAX && !v.contains(&r) { v.push(r); } } }
        if path.len() > 4096 { return; }
        path.push(a.neuron);
        let below: Vec<ActRef> = a.covers.iter().chain(a.binds.iter()).copied().collect();
        for c in below { self.descend(c, path, base, inters.as_deref_mut()); }
        path.pop();
    }

    /// What an activation stands for, as base symbols at their offsets from it, within `horizon` (D36).
    pub(crate) fn window_of(&self, r: ActRef, horizon: &Pos) -> Vec<(NeuronId, Pos)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut base = Vec::new();
        let mut path = if a.neuron == NeuronId::MAX { Vec::new() } else { Vec::new() };
        self.descend(r, &mut path, &mut base, None);
        let mut out: Vec<(NeuronId, Pos)> = base.iter().filter_map(|b| self.act(*b)).map(|b| (b.neuron, sub(&b.pos, &a.pos)))
            .filter(|(_, o)| (0..MAX_DIMS).all(|d| o[d].abs() <= horizon[d])).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The span of an activation (D4): how far the farthest base symbol it stands for lies from where it
    /// stands, in each dimension.
    pub(crate) fn span_of(&self, r: ActRef) -> Pos {
        let Some(a) = self.act(r) else { return [0; MAX_DIMS] };
        let mut base = Vec::new();
        self.descend(r, &mut Vec::new(), &mut base, None);
        let mut span = [0i32; MAX_DIMS];
        for b in base.iter().filter_map(|b| self.act(*b)) {
            for d in 0..MAX_DIMS { span[d] = span[d].max((b.pos[d] - a.pos[d]).abs()); }
        }
        span
    }

    /// The function an interneuron is read through when nothing fired it there: the one that fired it last, or
    /// any of its functions (R28).
    pub(crate) fn func_of(&self, n: NeuronId) -> Option<&Func> {
        self.last_func.get(&n).and_then(|id| self.func(*id)).or_else(|| self.dict.of_inter(n).first().and_then(|id| self.dict.get(*id)))
    }

    /// Where a function's anchor stands relative to its call (D53): the call at the latest frame, the anchor's,
    /// and at the centroid of the nodes its relations pin in every other dimension.
    pub(crate) fn anchor_offset(f: &Func) -> Pos {
        let mut off = [0i32; MAX_DIMS];
        for d in 1..MAX_DIMS {
            let offs: Vec<i32> = f.rels.iter().filter(|r| r.dim as usize == d).map(|r| r.off).collect();
            let sum: i32 = offs.iter().sum();
            off[d] = -round_half_down(sum as f64 / (offs.len() + 1) as f64);
        }
        off
    }

    /// Expand a neuron placed at `at` to base symbols at their coordinates (R28): a recognized call standing
    /// there through what fired it, an inferred one through its function, a variable to its argument; every
    /// interneuron the descent passes through below the root to `all` where asked.
    pub fn expand(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, all: Option<&mut Vec<(NeuronId, Pos)>>) {
        self.expand_from(n, at, i64::MIN, path, out, all);
    }

    /// Expansion that reads nothing before frame `from`: an expansion places only at or before where a node
    /// stands, so a node earlier than `from` places nothing there (R28).
    pub fn expand_from(&self, n: NeuronId, at: Pos, from: Frame, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, all: Option<&mut Vec<(NeuronId, Pos)>>) {
        let mut seen: rustc_hash::FxHashSet<(NeuronId, Pos)> = rustc_hash::FxHashSet::default();
        self.expand_inner(n, at, from, path, out, all, &mut seen);
    }

    /// One expansion, each neuron at each place read once however many paths reach it.
    fn expand_inner(&self, n: NeuronId, at: Pos, from: Frame, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>, seen: &mut rustc_hash::FxHashSet<(NeuronId, Pos)>) {
        if (at[0] as i64) < from { return; }
        if !seen.insert((n, at)) { return; }
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { return; }
        if !path.is_empty() { if let Some(v) = all.as_deref_mut() { v.push((n, at)); } }
        // a call that fired there is read back through what fired it
        if let Some(r) = self.fired_at(n, &at) {
            let mut base = Vec::new();
            let mut inters = Vec::new();
            let mut p = path.clone();
            self.descend(r, &mut p, &mut base, Some(&mut inters));
            for b in base.iter().filter_map(|b| self.act(*b)) { out.push((b.neuron, b.pos)); }
            if let Some(v) = all.as_deref_mut() { for i in inters.iter().filter_map(|i| self.act(*i)) { v.push((i.neuron, i.pos)); } }
            return;
        }
        let Some(f) = self.func_of(n).cloned() else { return };
        path.push(n);
        let anchor = add(&at, &Self::anchor_offset(&f));
        self.expand_inner(f.anchor, anchor, from, path, out, all.as_deref_mut(), seen);
        for r in &f.rels {
            let mut q = anchor;
            q[r.dim as usize] += r.off;
            let m = match r.node {
                Node::Fixed(m) => Some(m),
                Node::Class(k) => self.dict.class(k).first().copied(),
                Node::Param(p) => self.standing_at(&q).or_else(|| f.param_members.get(p as usize).and_then(|v| v.first().copied())),
            };
            if let Some(m) = m { self.expand_inner(m, q, from, path, out, all.as_deref_mut(), seen); }
        }
        path.pop();
    }

    /// The activation of a neuron that fired at a coordinate, if one did.
    pub(crate) fn fired_at(&self, n: NeuronId, at: &Pos) -> Option<ActRef> {
        let f = self.frames.iter().find(|f| f.no == at[0] as i64)?;
        f.acts.iter().position(|a| !a.gone && a.neuron == n && a.pos == *at).map(|i| ActRef { frame: f.no, index: i as u32 })
    }

    /// What stands at a coordinate that has happened, for binding an inferred call's parameter (D44).
    pub(crate) fn standing_at(&self, q: &Pos) -> Option<NeuronId> {
        if q[0] as i64 > self.frame_no { return None; }
        let f = self.frames.iter().find(|f| f.no == q[0] as i64)?;
        f.acts.iter().find(|a| !a.gone && a.strong && a.pos == *q && a.covered_at.is_none()).or_else(|| f.acts.iter().find(|a| !a.gone && a.strong && a.pos == *q)).map(|a| a.neuron)
    }
}
