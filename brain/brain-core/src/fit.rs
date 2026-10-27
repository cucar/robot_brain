//! Fit and savings (D54, D22), the greedy cover of a row (D28) and the margin (D30). One routine lands a
//! function's relations on a set of neighbors, against a row of a neuron's history or against the board, and
//! both read the same arithmetic.
//!
//! Bookkeeping: a value is counted with the activation it binds. A bound activation keeps one number, its
//! neuron, which is its value in the call's record (D10, D24); so binding a fresh activation is credited its
//! record less that one number, and the call's own record holds its `d + 2`, the positions no relation
//! fixes and the relations that did not come true. The difference A − B is D22's, number for number.

use rustc_hash::FxHashMap;

use crate::function::*;
use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

/// One function landed on a set of neighbors (D15, D22).
#[derive(Clone, Debug, Default)]
pub struct Landing {
    pub coverage: f64,
    pub price: f64,
    /// Per relation that came true, the neighbor it landed on.
    pub landed: Vec<(usize, usize)>,
    pub failed: Vec<usize>,
    /// Per relation through a variable, the neuron that stood there.
    pub args: Vec<(usize, NeuronId)>,
    /// Neighbors it covers, and how much of each it was credited.
    pub covered: Vec<usize>,
    /// Neighbors it binds, each with what is left of it once bound: its neuron, or nothing for a parameter's
    /// later nodes (D38).
    pub bound: Vec<(usize, f64)>,
    /// Calls it takes back whole (R23).
    pub takeback: Vec<ActRef>,
    /// Per neighbor landed, the dimensions its relations pin.
    pub pinned: Vec<(usize, u8)>,
}

impl Landing {
    pub fn savings(&self) -> f64 { self.coverage - self.price }
}

/// What is left of each thing in a row or on the board as a cover runs (D28).
pub trait State {
    fn anchor(&self) -> f64;
    fn nb(&self, i: usize) -> f64;
    fn call(&self, c: ActRef) -> f64;
}

/// The row as it stands before any function is taken.
pub struct Fresh<'a> { pub own: f64, pub nbs: &'a [Nb], pub calls: &'a [CallInfo] }
impl<'a> State for Fresh<'a> {
    fn anchor(&self) -> f64 { self.own }
    fn nb(&self, i: usize) -> f64 { self.nbs[i].cost }
    fn call(&self, c: ActRef) -> f64 { self.calls.iter().find(|x| x.act == c).map_or(0.0, |x| x.cost) }
}

/// A row partway through its greedy cover.
pub struct Cover { pub anchor: f64, pub nbs: Vec<f64>, pub calls: FxHashMap<ActRef, f64> }
impl State for Cover {
    fn anchor(&self) -> f64 { self.anchor }
    fn nb(&self, i: usize) -> f64 { self.nbs[i] }
    fn call(&self, c: ActRef) -> f64 { self.calls.get(&c).copied().unwrap_or(0.0) }
}

impl Cover {
    pub fn of(own: f64, nbs: &[Nb], calls: &[CallInfo]) -> Self {
        let mut c: FxHashMap<ActRef, f64> = calls.iter().map(|x| (x.act, x.cost)).collect();
        // a call standing in the row as a neighbor is the same number as the call it is
        for nb in nbs.iter().filter(|nb| nb.via == Via::Direct) { c.remove(&nb.act); }
        Cover { anchor: own, nbs: nbs.iter().map(|nb| nb.cost).collect(), calls: c }
    }
    /// What the row still costs: everything no function has taken, as itself.
    pub fn left(&self) -> f64 { self.anchor + self.nbs.iter().sum::<f64>() + self.calls.values().sum::<f64>() }
    /// Take a landed function: what it covers leaves, what it binds keeps its neuron (D28, D10).
    pub fn take(&mut self, l: &Landing, nbs: &[Nb]) {
        self.anchor = 0.0;
        for &i in &l.covered { self.nbs[i] = 0.0; }
        for &(i, left) in &l.bound { self.nbs[i] = self.nbs[i].min(left); }
        for c in &l.takeback {
            if let Some(v) = self.calls.get_mut(c) { *v = 0.0; }
            if let Some(i) = nbs.iter().position(|nb| nb.act == *c && nb.via == Via::Direct) { self.nbs[i] = 0.0; }
        }
    }
}

/// Of several neighbors that could answer one relation: the one not yet covered first, then the nearest,
/// then the earlier coordinate (D31).
fn better(nbs: &[Nb], st: &dyn State, a: usize, b: usize) -> bool {
    let free = |i: usize| st.nb(i) > 0.0 || nbs[i].via != Via::Direct;
    let near = |i: usize| nbs[i].delta.iter().map(|x| x.abs()).sum::<i32>();
    (free(a), -near(a), std::cmp::Reverse(nbs[a].delta)) > (free(b), -near(b), std::cmp::Reverse(nbs[b].delta))
}

/// Land a function on an activation's neighbors with the activation standing at node `at` (D15, D22, D31):
/// node 0 is the anchor, node `i + 1` the far node of relation `i`. Where the activation stands elsewhere than
/// at the anchor, the anchor is found among its neighbors by that relation, and every other relation lands from
/// the anchor. Each relation lands on the neighbor that satisfies it, a parameter's value the neuron at its
/// earliest node; then what the fit covers and binds and what its call writes: its record, a number for each
/// position of a landed activation that no relation landing on it fixes, and one per relation that did not
/// come true.
pub fn land(f: &Func, dict: &Dictionary, radii: &[i32], owner_axes: u8, at: usize, nbs: &[Nb], calls: &[CallInfo], st: &dyn State) -> Landing {
    let mut l = Landing::default();
    let off_ok = |d: usize, x: i32, want: i32| -> bool { d < radii.len() && round_offset(x, radii[d]) == want };
    // where the anchor stands, relative to the activation, and which neighbor it is
    let mut center: Option<usize> = None;
    let mut cd: Pos = [0; MAX_DIMS];
    if at > 0 {
        let r = &f.rels[at - 1];
        let d = r.dim as usize;
        let mut best: Option<usize> = None;
        for i in 0..nbs.len() {
            if nbs[i].neuron != f.anchor { continue; }
            if d > 0 && nbs[i].axes & (1 << d) == 0 { continue; }
            // the activation stands at the relation's offset from the anchor
            if !off_ok(d, -nbs[i].delta[d], r.off) { continue; }
            if best.map_or(true, |b| better(nbs, st, i, b)) { best = Some(i); }
        }
        let Some(c) = best else { l.price = f64::INFINITY; return l };
        center = Some(c);
        cd = nbs[c].delta;
    }
    let at_rel = |i: usize, r: &Rel| -> bool {
        let d = r.dim as usize;
        if d > 0 && nbs[i].axes & (1 << d) == 0 { return false; }
        if Some(i) == center { return false; }
        off_ok(d, nbs[i].delta[d] - cd[d], r.off)
    };
    let mut chosen: Vec<Option<usize>> = vec![None; f.rels.len()];
    for (ri, r) in f.rels.iter().enumerate() {
        if at > 0 && ri == at - 1 { continue; }
        let ok = |i: usize| -> bool {
            match r.node {
                Node::Fixed(n) => nbs[i].neuron == n,
                Node::Class(k) => !matches!(nbs[i].via, Via::Under(_)) && dict.class(k).contains(&nbs[i].neuron),
                Node::Param(_) => false,
            }
        };
        let mut best: Option<usize> = None;
        for i in 0..nbs.len() {
            if !ok(i) || !at_rel(i, r) { continue; }
            if best.map_or(true, |b| better(nbs, st, i, b)) { best = Some(i); }
        }
        chosen[ri] = best;
    }
    // parameters: the value is the neuron at the earliest of its nodes (D44)
    for p in 0..f.params() {
        let mut idx: Vec<usize> = (0..f.rels.len()).filter(|&ri| f.rels[ri].node == Node::Param(p as u8) && !(at > 0 && ri == at - 1)).collect();
        idx.sort_by_key(|&ri| (f.rels[ri].off, f.rels[ri].dim));
        let value = idx.iter().find_map(|&ri| {
            let r = &f.rels[ri];
            let mut best: Option<usize> = None;
            for i in 0..nbs.len() {
                if matches!(nbs[i].via, Via::Under(_)) || !at_rel(i, r) { continue; }
                if best.map_or(true, |b| better(nbs, st, i, b)) { best = Some(i); }
            }
            best.map(|i| nbs[i].neuron)
        });
        for &ri in &idx {
            let r = &f.rels[ri];
            let mut best: Option<usize> = None;
            if let Some(v) = value {
                for i in 0..nbs.len() {
                    if nbs[i].neuron != v || matches!(nbs[i].via, Via::Under(_)) || !at_rel(i, r) { continue; }
                    if best.map_or(true, |b| better(nbs, st, i, b)) { best = Some(i); }
                }
            }
            chosen[ri] = best;
        }
    }
    // what each landed neighbor is to the fit, and the dimensions its relations pin
    let mut fixed_on: FxHashMap<usize, u8> = FxHashMap::default();
    let mut var_on: FxHashMap<usize, (u8, Option<u8>)> = FxHashMap::default();
    let mut owner_pins: u8 = 0;
    let mut center_pins: u8 = 0;
    for (ri, c) in chosen.iter().enumerate() {
        let r = &f.rels[ri];
        if at > 0 && ri == at - 1 {
            // the activation itself stands at this node, and the anchor was found by it
            l.landed.push((ri, usize::MAX));
            owner_pins |= 1 << r.dim;
            center_pins |= 1 << r.dim;
            continue;
        }
        match c {
            None => l.failed.push(ri),
            Some(i) => {
                l.landed.push((ri, *i));
                if at == 0 { owner_pins |= 1 << r.dim; } else { center_pins |= 1 << r.dim; }
                match r.node {
                    Node::Fixed(_) => { *fixed_on.entry(*i).or_insert(0) |= 1 << r.dim; }
                    Node::Class(_) => { l.args.push((ri, nbs[*i].neuron)); let e = var_on.entry(*i).or_insert((0, None)); e.0 |= 1 << r.dim; }
                    Node::Param(p) => { l.args.push((ri, nbs[*i].neuron)); let e = var_on.entry(*i).or_insert((0, None)); e.0 |= 1 << r.dim; if e.1.is_none() { e.1 = Some(p); } }
                }
            }
        }
    }
    if let Some(c) = center { *fixed_on.entry(c).or_insert(0) |= center_pins; }
    let unpinned = |axes: u8, pins: u8| -> f64 { (((axes | 1) & !pins) & 0b111).count_ones() as f64 };
    let mut coverage = st.anchor();
    let mut price = record(owner_axes) + l.failed.len() as f64;
    if st.anchor() > 0.0 { price += unpinned(owner_axes, owner_pins); }
    let mut inside: FxHashMap<ActRef, Vec<usize>> = FxHashMap::default();
    let mut keys: Vec<usize> = fixed_on.keys().chain(var_on.keys()).copied().collect();
    keys.sort_unstable();
    keys.dedup();
    let mut param_seen: Vec<bool> = vec![false; f.params()];
    for i in keys {
        let nb = &nbs[i];
        let left = st.nb(i);
        let pins = fixed_on.get(&i).copied().unwrap_or(0) | var_on.get(&i).map_or(0, |v| v.0);
        l.pinned.push((i, pins));
        match nb.via {
            Via::Under(_) => {}
            Via::Inside(c) => { inside.entry(c).or_default().push(i); }
            Via::Direct => {
                let fresh = left >= nb.full - 1e-9 && left > 0.0;
                if fixed_on.contains_key(&i) {
                    if left > 0.0 { coverage += left; l.covered.push(i); if fresh { price += unpinned(nb.axes, pins); } }
                } else if let Some((_, p)) = var_on.get(&i) {
                    // a parameter writes its value once: every later node saves its neuron too (D38)
                    let later = match p { Some(p) => { let s = param_seen[*p as usize]; param_seen[*p as usize] = true; s } None => false };
                    if fresh {
                        coverage += if later { left } else { left - 1.0 };
                        price += unpinned(nb.axes, pins);
                        l.bound.push((i, if later { 0.0 } else { 1.0 }));
                    }
                }
            }
        }
    }
    // a call is taken back whole or not at all (R23, D22)
    for (c, members) in inside {
        let size = calls.iter().find(|x| x.act == c).map_or(usize::MAX, |x| x.size);
        let direct_landed = l.landed.iter().any(|(_, i)| *i != usize::MAX && nbs[*i].act == c && nbs[*i].via == Via::Direct) || center.map_or(false, |x| nbs[x].act == c);
        let mut acts: Vec<ActRef> = members.iter().map(|&i| nbs[i].act).collect();
        acts.sort_unstable();
        acts.dedup();
        if !direct_landed && acts.len() >= size {
            let v = st.call(c);
            if v > 0.0 { coverage += v; l.takeback.push(c); }
        }
    }
    l.coverage = coverage;
    l.price = price;
    l
}

/// The best landing of a function on an activation of `owner` (D31): over every node the activation can stand
/// at, the one that saves most.
pub fn land_best(f: &Func, owner: NeuronId, dict: &Dictionary, radii: &[i32], owner_axes: u8, nbs: &[Nb], calls: &[CallInfo], st: &dyn State) -> Option<Landing> {
    let mut best: Option<Landing> = None;
    for at in f.nodes_for(owner) {
        let l = land(f, dict, radii, owner_axes, at, nbs, calls, st);
        if !l.price.is_finite() { continue; }
        if best.as_ref().map_or(true, |b| l.savings() > b.savings()) { best = Some(l); }
    }
    best
}

/// What a landed function still covers as a cover runs (D28): its price is fixed by what fired, and only its
/// coverage falls as earlier rounds take what it covers.
pub fn cover_value(l: &Landing, nbs: &[Nb], st: &dyn State) -> f64 {
    let mut v = st.anchor();
    for &i in &l.covered { v += st.nb(i); }
    for &(i, left) in &l.bound {
        let r = st.nb(i);
        if r >= nbs[i].full - 1e-9 && r > 0.0 { v += r - left; }
    }
    for c in &l.takeback { v += st.call(*c); }
    v
}

/// The greedy cover of a row (D28) by functions already landed on it, each with when it joined the
/// dictionary for ties: what was paid, and what is left as itself.
pub fn greedy(row: &Row, landed: &[(&Landing, u64)]) -> f64 { cover_of(row, landed).0 }

/// The greedy cover of a row (D28): what it costs, and which of the landed functions it took, in order.
pub fn cover_of(row: &Row, landed: &[(&Landing, u64)]) -> (f64, Vec<usize>) { let (c, t, _) = cover_left(row, landed); (c, t) }

/// The greedy cover of a row (D28): what it costs, which functions it took, and what it left.
pub fn cover_left(row: &Row, landed: &[(&Landing, u64)]) -> (f64, Vec<usize>, Cover) {
    let mut taken: Vec<usize> = Vec::new();
    let mut st = Cover::of(row.own_cost, &row.nbs, &row.calls);
    let mut left: Vec<usize> = (0..landed.len()).collect();
    let mut paid = 0.0;
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (x, &k) in left.iter().enumerate() {
            let (l, joined) = landed[k];
            let cov = cover_value(l, &row.nbs, &st);
            if !(cov > l.price) { continue; }
            let ratio = cov / l.price;
            let take = match best { None => true, Some((y, br)) => ratio > br || (ratio == br && joined < landed[left[y]].1) };
            if take { best = Some((x, ratio)); }
        }
        let Some((x, _)) = best else { break };
        let (l, _) = landed[left[x]];
        paid += l.price;
        st.take(l, &row.nbs);
        taken.push(left[x]);
        left.remove(x);
    }
    (paid + st.left(), taken, st)
}

/// What the election would write for a row from a set of functions (D30): the greedy cover (D28) by those
/// that fit it, each taken as its call's record, and what none takes as itself. Ties go to the function that
/// joined the dictionary first.
pub fn row_cost(row: &Row, funcs: &[&Func], dict: &Dictionary, radii: &[i32]) -> f64 {
    let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
    let ls: Vec<(Landing, u64)> = funcs.iter().filter_map(|f| land_best(f, row.owner, dict, radii, row.axes, &row.nbs, &row.calls, &fresh).map(|l| (l, f.joined))).filter(|(l, _)| l.savings() > 0.0).collect();
    let refs: Vec<(&Landing, u64)> = ls.iter().map(|(l, j)| (l, *j)).collect();
    greedy(row, &refs)
}

/// Whether a function fits a row: its savings there are positive (D54).
pub fn fits(f: &Func, row: &Row, dict: &Dictionary, radii: &[i32]) -> bool {
    let fresh = Fresh { own: row.own_cost, nbs: &row.nbs, calls: &row.calls };
    land_best(f, row.owner, dict, radii, row.axes, &row.nbs, &row.calls, &fresh).map_or(false, |l| l.savings() > 0.0)
}

/// What a row is to the arithmetic, as one number: its own cost, and each neighbor's neuron, offset, cost and
/// how it stands, with the calls it stands inside named by their order in the row. Two rows with one
/// signature cost the same under every set of functions, so a sum over rows is a sum over signatures.
pub fn row_sig(row: &Row) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    row.own_cost.to_bits().hash(&mut h);
    row.axes.hash(&mut h);
    let local = |c: &ActRef| row.calls.iter().position(|x| x.act == *c).map_or(usize::MAX, |i| i) as u64;
    let mut items: Vec<(Pos, NeuronId, u64, u64, u8, u64, u64)> = row.nbs.iter().map(|nb| {
        let (tag, c) = match nb.via {
            Via::Direct => (0u8, row.calls.iter().position(|x| x.act == nb.act).map_or(u64::MAX, |i| i as u64)),
            Via::Inside(c) => (1u8, local(&c)),
            Via::Under(c) => (2u8, row.nbs.iter().position(|x| x.act == c).map_or(u64::MAX, |i| i as u64)),
        };
        (nb.delta, nb.neuron, nb.cost.to_bits(), nb.full.to_bits(), tag, c, nb.axes as u64)
    }).collect();
    items.sort_unstable();
    items.hash(&mut h);
    for c in &row.calls { c.size.hash(&mut h); c.cost.to_bits().hash(&mut h); }
    h.finish()
}

/// Rows by signature: one row to read for each, and how many rows share it.
pub fn by_sig<'a>(rows: impl Iterator<Item = &'a Row>) -> Vec<(&'a Row, f64)> {
    let mut at: FxHashMap<u64, usize> = FxHashMap::default();
    let mut out: Vec<(&'a Row, f64)> = Vec::new();
    for r in rows {
        let s = row_sig(r);
        match at.get(&s) { Some(&i) => out[i].1 += 1.0, None => { at.insert(s, out.len()); out.push((r, 1.0)); } }
    }
    out
}
