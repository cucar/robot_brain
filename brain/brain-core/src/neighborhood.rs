//! Neighborhoods (D5, D7, D47): what an activation observes.

use rustc_hash::FxHashSet;

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    /// The call an activation is covered by as it stands: the last that took it, else its coverer (R23).
    pub(crate) fn current_coverer(a: &Act) -> Option<ActRef> {
        a.retaken.last().map(|(_, c)| *c).or(a.coverer)
    }

    /// What one activation observes (D5, D7): every activation in its frame or an earlier one, within its reach
    /// in every activation dimension they share, that stands on the apex, or is covered by a call on the apex
    /// other than its own; and, for each interneuron on the apex, the base symbols it stands for (D47). Each
    /// with what it costs in the file as it stands (D13). Returns the neighbors and the calls whose covered
    /// activations stand among them.
    pub(crate) fn neighborhood(&self, me: ActRef) -> (Vec<Nb>, Vec<CallInfo>) {
        let Some(a) = self.act(me) else { return (vec![], vec![]) };
        let mut out: Vec<Nb> = Vec::new();
        let mut calls: Vec<CallInfo> = Vec::new();
        let inside_box = |p: &Pos, baxes: u8| -> bool {
            let dt = a.pos[0] - p[0];
            if dt < 0 || dt > a.reach[0] { return false; }
            let shared = a.axes & baxes;
            (1..self.radii.len()).all(|d| shared & (1 << d) == 0 || (p[d] - a.pos[d]).abs() <= a.reach[d])
        };
        let delta = |p: &Pos, baxes: u8| -> Pos {
            let mut dlt = [0i32; MAX_DIMS];
            dlt[0] = p[0] - a.pos[0];
            for d in 1..self.radii.len() { if a.axes & baxes & (1 << d) != 0 { dlt[d] = p[d] - a.pos[d]; } }
            dlt
        };
        let mut apex_inters: Vec<usize> = Vec::new();
        for f in self.frames.iter() {
            if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me || b.gone || b.neuron == NeuronId::MAX { continue; }
                if b.opened_at == self.frame_no && b.level > a.level { continue; }
                if !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                if !inside_box(&b.pos, b.axes) { continue; }
                let via = if b.on_apex() {
                    Via::Direct
                } else {
                    // covered by a call on the apex other than its own (D5)
                    let Some(c) = Self::current_coverer(b) else { continue };
                    if c == me { continue; }
                    let Some(ca) = self.act(c) else { continue };
                    if !ca.on_apex() { continue; }
                    if !calls.iter().any(|x| x.act == c) { calls.push(CallInfo { act: c, size: ca.covers.len(), cost: ca.cost }); }
                    Via::Inside(c)
                };
                let cost = if via == Via::Direct { b.cost } else { 0.0 };
                if via == Via::Direct && b.kind == Kind::Inter { apex_inters.push(out.len()); }
                out.push(Nb { act: r, neuron: b.neuron, delta: delta(&b.pos, b.axes), axes: b.axes, cost, full: self.full_of(b), via });
            }
        }
        // an interneuron on the apex stands for the base symbols under it (D47)
        let present: FxHashSet<ActRef> = out.iter().map(|nb| nb.act).collect();
        for j in apex_inters {
            let c = out[j].act;
            let mut base: Vec<ActRef> = Vec::new();
            self.descend(c, &mut Vec::new(), &mut base, None);
            for r in base {
                if r == me || present.contains(&r) { continue; }
                let Some(b) = self.act(r) else { continue };
                if !inside_box(&b.pos, b.axes) || !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                out.push(Nb { act: r, neuron: b.neuron, delta: delta(&b.pos, b.axes), axes: b.axes, cost: 0.0, full: self.full_of(b), via: Via::Under(c) });
            }
        }
        // one order for every neighborhood, so two rows alike list their neighbors alike (D18)
        out.sort_by(|x, y| {
            let tag = |v: &Via| match v { Via::Direct => 0u8, Via::Inside(_) => 1, Via::Under(_) => 2 };
            (x.delta, x.neuron, tag(&x.via), x.act).cmp(&(y.delta, y.neuron, tag(&y.via), y.act))
        });
        let mut ordered: Vec<CallInfo> = Vec::new();
        for nb in &out {
            if let Via::Inside(c) = nb.via { if !ordered.iter().any(|x| x.act == c) { if let Some(ci) = calls.iter().find(|x| x.act == c) { ordered.push(*ci); } } }
        }
        (out, ordered)
    }

    /// An activation's whole record as it fired (D12): `d + 2` bare, its call's record above.
    pub(crate) fn full_of(&self, b: &Act) -> f64 {
        if b.kind == Kind::Inter { b.full } else { record(b.axes) }
    }
}
