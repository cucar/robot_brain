//! The election (§6.2, R22, R23, R24, D24, D31) and allocation (§6.3, R16, R43): every function found for an
//! activation of the level is placed on the board, the greedy cover takes them best first, and each accepted
//! function fires its interneuron at the next level, reused where a stretch comes round.

use std::collections::BinaryHeap;

use rustc_hash::FxHashSet;

use crate::expansion::round_half_down;
use crate::fit::*;
use crate::function::*;
use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

/// One activation of the level with what it observes: the board around it (D49).
pub(crate) struct View {
    pub act: ActRef,
    pub nbs: Vec<Nb>,
    pub calls: Vec<CallInfo>,
}

/// The board as it stands, read live from the open activations (R23).
struct Board<'a> { m: &'a Machine, anchor: ActRef, nbs: &'a [Nb] }
impl<'a> State for Board<'a> {
    fn anchor(&self) -> f64 { self.m.act(self.anchor).map_or(0.0, |a| if a.on_apex() { a.cost } else { 0.0 }) }
    fn nb(&self, i: usize) -> f64 {
        let nb = &self.nbs[i];
        if nb.via != Via::Direct { return 0.0; }
        self.m.act(nb.act).map_or(0.0, |a| if a.on_apex() { a.cost } else { 0.0 })
    }
    fn call(&self, c: ActRef) -> f64 { self.m.act(c).map_or(0.0, |a| if a.on_apex() { a.cost } else { 0.0 }) }
}

/// A function placed on the board (D31).
struct Placement { func: FuncId, view: usize }

#[derive(PartialEq)]
struct Keyed { ratio: f64, rank: (u8, u64, Frame, Pos), idx: usize }
impl Eq for Keyed {}
impl PartialOrd for Keyed { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Keyed {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // the higher ratio first; on a tie, the smaller rank first (R24)
        self.ratio.partial_cmp(&o.ratio).unwrap_or(std::cmp::Ordering::Equal).then_with(|| o.rank.cmp(&self.rank))
    }
}

impl Machine {
    fn land_on_board(&self, f: &Func, v: &View) -> Option<Landing> {
        let a = self.act(v.act)?;
        land_best(f, a.neuron, &self.dict, &self.radii, a.axes, &v.nbs, &v.calls, &Board { m: self, anchor: v.act, nbs: &v.nbs })
    }

    /// The activations a landing writes for: the one read, what it covers and binds, and the calls it takes back.
    fn landed_acts(v: &View, l: &Landing) -> Vec<ActRef> {
        let mut acts: Vec<ActRef> = vec![v.act];
        acts.extend(l.covered.iter().map(|i| v.nbs[*i].act));
        acts.extend(l.bound.iter().map(|(i, _)| v.nbs[*i].act));
        acts.extend(l.takeback.iter().copied());
        acts.sort_unstable();
        acts.dedup();
        acts
    }

    /// The tie-break of two placements that measure the same (R24): a function with an interneuron before one
    /// without, the older interneuron, the function that joined first, the earlier coordinate.
    fn rank(&self, f: &Func, anchor: ActRef) -> (u8, u64, Frame, Pos) {
        let pos = self.act(anchor).map_or([0; MAX_DIMS], |a| a.pos);
        match f.inter.filter(|n| self.thalamus.get(*n).map_or(false, |x| x.alive)) {
            Some(n) => (0, n as u64, anchor.frame, pos),
            None => (1, f.joined, anchor.frame, pos),
        }
    }

    /// Place every function of the dictionary anchored at each activation's neuron where it fits the board, and
    /// run the greedy cover over them (R24, D28). Accepted functions fire their interneurons at the next level.
    /// Returns how many were accepted, allocated and reused.
    pub(crate) fn elect(&mut self, level: u32, views: &[View]) -> (usize, usize, usize) {
        let mut placements: Vec<Placement> = Vec::new();
        let mut seen: FxHashSet<(FuncId, Vec<ActRef>)> = FxHashSet::default();
        let mut heap: BinaryHeap<Keyed> = BinaryHeap::new();
        for (vi, v) in views.iter().enumerate() {
            let Some(a) = self.act(v.act) else { continue };
            let ids: Vec<FuncId> = self.dict.standing(a.neuron).to_vec();
            for id in ids {
                let Some(f) = self.dict.get(id) else { continue };
                // a test fires only the interneurons the machine already has
                if self.frozen && !f.inter.map_or(false, |n| self.thalamus.get(n).map_or(false, |x| x.alive)) { continue; }
                let Some(l) = self.land_on_board(f, v) else { continue };
                if !(l.coverage > l.price) { continue; }
                let acts = Self::landed_acts(v, &l);
                if !seen.insert((id, acts)) { continue; }
                let idx = placements.len();
                placements.push(Placement { func: id, view: vi });
                heap.push(Keyed { ratio: l.coverage / l.price, rank: self.rank(f, v.act), idx });
            }
        }
        let (mut accepted, mut minted, mut reused) = (0, 0, 0);
        // lazy greedy: coverage only falls, so a placement whose fresh ratio still leads is the best
        while let Some(top) = heap.pop() {
            let p = &placements[top.idx];
            let v = &views[p.view];
            let Some(f) = self.dict.get(p.func).cloned() else { continue };
            let Some(l) = self.land_on_board(&f, v) else { continue };
            if !(l.coverage > l.price) { continue; }
            let ratio = l.coverage / l.price;
            if let Some(next) = heap.peek() {
                if ratio < top.ratio && (ratio < next.ratio || (ratio == next.ratio && next.rank < top.rank)) {
                    heap.push(Keyed { ratio, rank: top.rank, idx: top.idx });
                    continue;
                }
            }
            let (m, u) = self.accept(&f, v, &l, level);
            accepted += 1;
            minted += m;
            reused += u;
        }
        (accepted, minted, reused)
    }

    /// Accept a placed function (R24, D24): what it covers leaves the apex, what it binds stays with its
    /// position covered, the calls it takes back leave with everything they covered (R23); its interneuron
    /// fires at the next level where what it covers stands (D53), allocated or reused (R16, R43).
    fn accept(&mut self, f: &Func, v: &View, l: &Landing, level: u32) -> (usize, usize) {
        let fno = self.frame_no;
        let Some(anchor) = self.act(v.act).cloned() else { return (0, 0) };
        // where the call stands: the latest frame, and the centroid of what it covers and binds in every other
        // dimension its function pins (D53)
        let mut pos = anchor.pos;
        pos[0] = fno as i32;
        for d in 1..self.radii.len() {
            if anchor.axes & (1 << d) == 0 { continue; }
            let mut ps: Vec<i32> = vec![anchor.pos[d]];
            for (i, pins) in &l.pinned {
                if pins & (1 << d) != 0 { if let Some(b) = self.act(v.nbs[*i].act) { ps.push(b.pos[d]); } }
            }
            pos[d] = round_half_down(ps.iter().sum::<i32>() as f64 / ps.len() as f64);
        }
        let covered: Vec<ActRef> = std::iter::once(v.act).chain(l.covered.iter().map(|i| v.nbs[*i].act)).chain(l.takeback.iter().copied()).collect();
        let bound: Vec<(ActRef, f64)> = l.bound.iter().map(|(i, left)| (v.nbs[*i].act, *left)).collect();
        let strong = covered.iter().chain(bound.iter().map(|(r, _)| r)).filter_map(|r| self.act(*r)).any(|a| a.strong);
        let height = 1 + covered.iter().chain(bound.iter().map(|(r, _)| r)).filter_map(|r| self.act(*r)).map(|a| a.height).max().unwrap_or(0);
        let fired_by = FiredBy {
            func: f.id,
            landed: l.landed.iter().map(|(ri, i)| (*ri, if *i == usize::MAX { v.act } else { v.nbs[*i].act })).collect(),
            failed: l.failed.clone(),
            args: l.args.clone(),
        };
        let act = Act {
            neuron: NeuronId::MAX, kind: Kind::Inter, axes: anchor.axes, pos, cost: l.price, full: l.price, strong,
            span: [0; MAX_DIMS], reach: [0; MAX_DIMS], covered_at: None, bound_at: None, coverer: None, retaken: Vec::new(),
            covers: covered.clone(), binds: bound.iter().map(|(r, _)| *r).collect(), fired_by: Some(fired_by),
            inferences: Vec::new(), opened_at: fno, level: level + 1, height, gone: false, caught_up: fno,
        };
        let Some(fr) = self.frame_mut(fno) else { return (0, 0) };
        let me = ActRef { frame: fno, index: fr.acts.len() as u32 };
        fr.acts.push(act);
        // the coverage set (D24)
        for r in &covered {
            let takeback = l.takeback.contains(r);
            let inner: Vec<ActRef> = if takeback { self.act(*r).map(|a| a.covers.clone()).unwrap_or_default() } else { Vec::new() };
            if let Some(a) = self.act_mut(*r) {
                if a.covered_at.is_none() { a.covered_at = Some((fno - r.frame) as u32); a.coverer = Some(me); }
            }
            for c in inner {
                if let Some(a) = self.act_mut(c) { a.retaken.push(((fno - c.frame) as u32, me)); }
            }
        }
        for (r, left) in &bound {
            if let Some(a) = self.act_mut(*r) {
                if a.bound_at.is_none() { a.bound_at = Some((fno - r.frame) as u32); }
                a.cost = a.cost.min(*left);
            }
        }
        // its interneuron: its own, one it comes round to, or a new one (R43, R16)
        let (mut minted, mut reused) = (0, 0);
        let neuron = match f.inter.filter(|n| self.thalamus.get(*n).map_or(false, |x| x.alive)) {
            Some(n) => n,
            None => match self.come_round(me) {
                Some(n) => { reused += 1; self.dict.set_inter(f.id, n); if self.trace { eprintln!("[ring f{}] {} -> #{}", fno, self.dict.describe(f), n); } n }
                None => {
                    minted += 1;
                    let id = self.thalamus.next_id();
                    self.thalamus.push(Neuron::inter(id, anchor.axes, self.cfg.h, self.radii.clone()));
                    self.spans.push([0; MAX_DIMS]);
                    self.alive += 1;
                    self.dict.set_inter(f.id, id);
                    id
                }
            },
        };
        if !self.frozen {
            self.ledger.retain(|x| *x != neuron);
            self.last_func.insert(neuron, f.id);
        }
        if let Some(a) = self.act_mut(me) { a.neuron = neuron; }
        let span = self.span_of(me);
        let reach = self.reach_for(anchor.axes, &span);
        if reach[0] > self.max_reach_t { self.max_reach_t = reach[0]; }
        if let Some(a) = self.act_mut(me) { a.span = span; a.reach = reach; }
        if !self.frozen { self.spans[neuron as usize] = span; }
        // where the report built the strong activation of a pattern an inference placed weakly, the weak one is
        // dropped (D40, R30)
        if strong {
            if let Some(fr) = self.frame_mut(fno) {
                for (i, w) in fr.acts.iter_mut().enumerate() {
                    if i as u32 != me.index && !w.strong && !w.gone && w.neuron == neuron && w.pos == pos { w.gone = true; }
                }
            }
        }
        if self.trace { eprintln!("[accept f{} l{}] {} at {:?} cov {:.1} price {:.1} failed {:?} -> #{}", fno, level, self.dict.describe(f), pos, l.coverage, l.price, l.failed, neuron); }
        (minted, reused)
    }

    /// The interneuron a new call comes round to (R43, D36): reading what it would stand for back, one met
    /// within that interneuron's own length, not where the call stands, over exactly that length the same.
    fn come_round(&self, me: ActRef) -> Option<NeuronId> {
        let a = self.act(me)?;
        let mut base = Vec::new();
        let mut inters = Vec::new();
        self.descend(me, &mut Vec::new(), &mut base, Some(&mut inters));
        let mut met: Vec<(NeuronId, ActRef)> = inters.iter().filter_map(|r| self.act(*r).map(|m| (m.neuron, *r))).collect();
        met.sort_unstable();
        for (n, r) in met {
            let m = self.act(r)?;
            if !self.thalamus.get(n).map_or(false, |x| x.alive) { continue; }
            let q = sub(&m.pos, &a.pos);
            if q == [0; MAX_DIMS] { continue; }
            if (0..MAX_DIMS).any(|d| q[d].abs() > m.span[d]) { continue; }
            let w = self.window_of(r, &m.span);
            if !w.is_empty() && self.window_of(me, &m.span) == w { return Some(n); }
        }
        None
    }
}
