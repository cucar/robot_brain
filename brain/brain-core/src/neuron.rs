//! One neuron (§4, §5): its table, its history and its connections, and the `process functions`
//! call (R20) that reads them: refresh, fit, re-center, create, merge, retire, return. It follows
//! docs/algorithm.md at 699b1ca.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::offsets::*;
use crate::types::*;

pub use crate::bid::*;
pub use crate::connection::*;
pub use crate::history::*;
pub use crate::pattern::*;

/// What the machine hands a neuron with its call (R20): the count of neurons, as bits (D13).
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub neuron_bits: f64,
    pub frame: Frame,
    pub trace: bool,
}

/// What the machine's index says as a level begins, the same for every neuron of the level: every neuron's
/// span (D4), for the reach a function's child has; and the window, the longest reach the machine holds in each activation dimension, for what a position
/// costs (D13).
#[derive(Default)]
pub struct Shared {
    pub spans: Vec<Pos>,
    pub window: Pos,
}

/// What the machine hands a neuron for one activation of the frame (R20).
#[derive(Clone, Debug)]
pub struct RowInput {
    pub act: ActRef,
    pub pos: Pos,
    pub own_cost: f64,
    pub nbs: Vec<Nb>,
}

pub struct Neuron {
    pub id: NeuronId,
    pub kind: Kind,
    pub dim: Option<DimId>,
    pub bucket: u32,
    pub channel: Option<ChannelId>,
    pub axes: u8,
    pub alive: bool,
    /// The table (D32): its functions, and the classes they use.
    pub functions: Vec<Function>,
    pub classes: Vec<Class>,
    pub(crate) next_pid: PatternId,
    pub history: Vec<Row>,
    pub h: usize,
    pub connections: FxHashMap<(NeuronId, Offset), Conn>,
    /// An interneuron's parents in the index (D36): each neuron with a function pointing at it, and which.
    pub parents: Vec<(NeuronId, PatternId)>,
    /// What an interneuron stands for (D36), as the machine last wrote it.
    pub key: String,
    pub radii: Vec<i32>,
    pub shared: std::sync::Arc<Shared>,
    /// Functions whose uses a collapse changed: they re-center again at the next call (D27).
    pub(crate) unsettled: FxHashSet<PatternId>,
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind, dim: Some(dim), bucket, channel: Some(channel), axes, alive: true, functions: Vec::new(), classes: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), key: format!("b{}", id), radii, shared: Default::default(), unsettled: FxHashSet::default() }
    }

    pub fn inter(id: NeuronId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes, alive: true, functions: Vec::new(), classes: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), key: String::new(), radii, shared: Default::default(), unsettled: FxHashSet::default() }
    }

    pub fn function(&self, fid: PatternId) -> Option<&Function> { self.functions.iter().find(|f| f.id == fid) }
    pub fn function_mut(&mut self, fid: PatternId) -> Option<&mut Function> { self.functions.iter_mut().find(|f| f.id == fid) }
    pub fn class(&self, k: PatternId) -> Option<&Class> { self.classes.iter().find(|c| c.id == k) }
    pub fn class_mut(&mut self, k: PatternId) -> Option<&mut Class> { self.classes.iter_mut().find(|c| c.id == k) }
    pub(crate) fn new_pid(&mut self) -> PatternId { let id = self.next_pid; self.next_pid += 1; id }

    pub(crate) fn round(&self, delta: &Pos) -> Offset { round_delta(delta, &self.radii, self.axes) }

    // ───────────────────────────── prices (D13) ─────────────────────────────

    pub(crate) fn span_of(&self, n: NeuronId) -> Pos { self.shared.spans.get(n as usize).copied().unwrap_or([0; MAX_DIMS]) }

    /// The offsets of a dimension at a span's reach (D6, D13), summed over the neuron's axes.
    pub(crate) fn offsets_at(&self, span: &Pos) -> f64 {
        let mut reach = [0; MAX_DIMS];
        for d in 0..self.radii.len() { if self.axes & (1 << d) != 0 { reach[d] = span[d] + self.radii[d]; } }
        offset_bits(&reach, &self.radii, self.axes)
    }

    /// A position (D13): in time, a frame of the window; in every other dimension, a position within the
    /// window there; summed over the neuron's axes.
    pub(crate) fn position_bits(&self) -> f64 { window_bits(&self.shared.window, self.axes) }


    /// An offset in the table, at the owner's reach (D13).
    pub(crate) fn offset_bits(&self) -> f64 { self.offsets_at(&self.span_of(self.id)) }

    /// Which class: `log₂ |classes|`, nothing with one class (D13).
    pub fn which_class_bits(&self) -> f64 { log2(self.classes.len() as f64) }

    /// A function's entry in the table (D15): a neuron per fixed node, which class per class, and an offset
    /// per relation, one per place besides the owner.
    pub fn entry(&self, f: &Function, ctx: Ctx) -> f64 {
        f.fixed_count() as f64 * ctx.neuron_bits + f.class_places() as f64 * self.which_class_bits() + f.p_size() as f64 * self.offset_bits()
    }

    /// A class's entry in the table (D41): a neuron per member.
    pub fn class_entry(&self, c: &Class, ctx: Ctx) -> f64 { c.members.len() as f64 * ctx.neuron_bits }

    /// The index of a row by its activation.
    pub(crate) fn row_index(&self) -> FxHashMap<ActRef, usize> { self.history.iter().enumerate().map(|(i, r)| (r.act, i)).collect() }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    /// `process functions`: refresh, fit, re-center, create, merge, retire, return (R20). Returns the bids
    /// of the frame's activations, everything retired, and how many functions were added.
    pub fn process(&mut self, inputs: Vec<RowInput>, ctx: Ctx) -> (Vec<Bid>, Vec<Retired>, usize) {
        let mut retired: Vec<Retired> = Vec::new();
        // refresh history (§5.1): evict, then admit
        let incoming = inputs.len().min(self.h);
        let mut changed: FxHashSet<PatternId> = std::mem::take(&mut self.unsettled);
        let mut evicted: Vec<ActRef> = Vec::new();
        while self.history.len() + incoming > self.h { evicted.push(self.history.remove(0).act); }
        if !evicted.is_empty() {
            for f in self.functions.iter_mut() {
                let before = f.uses.len();
                f.uses.retain(|a| !evicted.contains(a));
                if f.uses.len() != before { changed.insert(f.id); }
            }
        }
        let first_new = self.history.len();
        for inp in inputs.into_iter().take(incoming) {
            let rec = 1.0 + ctx.neuron_bits + self.position_bits();
            self.history.push(Row { act: inp.act, pos: inp.pos, own_cost: inp.own_cost, rec, nbs: inp.nbs });
        }
        let ids: Vec<PatternId> = self.functions.iter().filter(|f| !f.params.is_empty()).map(|f| f.id).collect();
        for fid in ids { self.record_params(fid); }
        // fit (§5.2): every function against each admitted row
        for i in first_new..self.history.len() {
            let act = self.history[i].act;
            let fitting: Vec<PatternId> = self.functions.iter().filter(|f| self.fits(&self.history[i], f, ctx)).map(|f| f.id).collect();
            for fid in fitting {
                if let Some(f) = self.function_mut(fid) { f.uses.push(act); }
                changed.insert(fid);
            }
        }
        // re-center (§5.3): once per function whose uses changed
        let mut ids: Vec<PatternId> = changed.into_iter().filter(|f| self.function(*f).is_some()).collect();
        ids.sort();
        for fid in &ids { self.recenter(*fid, ctx); }
        // create (§5.4)
        let before = self.next_pid;
        let added = self.create(ctx);
        // merge (§5.5)
        retired.extend(self.merge(ctx));
        // retire (§5.6)
        retired.extend(self.retire(ctx));
        if ctx.trace {
            for f in self.functions.iter().filter(|f| f.id >= before) { eprintln!("[add f{} #{}] p{} {} margin {:.1}", ctx.frame, self.id, f.id, describe_function(f), self.margin(f.id, ctx)); }
            for r in retired.iter() { eprintln!("[retire f{} #{}] p{} child={:?}", ctx.frame, self.id, r.pattern, r.child); }
        }
        // return (§5.7)
        let bids = self.bids(first_new, ctx);
        (bids, retired, added)
    }

    /// Re-read a function's fit over every row (D27, D54): its uses as they now stand.
    pub(crate) fn refit(&mut self, fid: PatternId, ctx: Ctx) -> bool {
        let Some(f) = self.function(fid) else { return false };
        let uses: Vec<ActRef> = self.history.iter().filter(|r| self.fits(r, f, ctx)).map(|r| r.act).collect();
        let changed = uses != f.uses;
        if let Some(g) = self.function_mut(fid) { g.uses = uses; }
        changed
    }

    // ───────────────────────────── the machine's writes (§4.1) ─────────────────────────────

    pub fn wire_child(&mut self, fid: PatternId, child: NeuronId) {
        if let Some(f) = self.function_mut(fid) { f.child = Some(child); }
    }

    /// Update child (§4.1): point the function at the child where the definition is the function; else adapt
    /// the function to the definition or detach it from the child, whichever leaves the larger margin. The
    /// definition is the function that set it, anchored on its own owner; adapting is possible only where this
    /// neuron is one of its nodes, re-anchored on it.
    pub fn update_child(&mut self, fid: PatternId, child: NeuronId, def_owner: NeuronId, def: &Function, ctx: Ctx) -> ChildUpdate {
        let Some(f) = self.function(fid).cloned() else { return ChildUpdate::Gone };
        let adapted = self.anchor(def_owner, def);
        if let Some(a) = &adapted {
            if a.places == f.places && a.params.len() == f.params.len() {
                if let Some(g) = self.function_mut(fid) { g.child = Some(child); }
                return ChildUpdate::Pointed;
            }
        }
        let keep = self.margin(fid, ctx);
        if let Some(a) = adapted {
            let old = f.clone();
            if let Some(g) = self.function_mut(fid) { g.places = a.places.clone(); g.params = a.params.clone(); }
            self.refit(fid, ctx);
            let m = self.margin(fid, ctx);
            if m > keep {
                if let Some(g) = self.function_mut(fid) { g.child = Some(child); }
                return ChildUpdate::Adapted;
            }
            if let Some(g) = self.function_mut(fid) { g.places = old.places; g.params = old.params; g.uses = old.uses; }
        }
        if let Some(g) = self.function_mut(fid) { if g.child == Some(child) { g.child = None; } }
        ChildUpdate::Detached
    }

    /// A definition re-anchored on this neuron: the same nodes and relations, written from this neuron's
    /// place among them. None where this neuron is not one of its fixed nodes, or the definition uses a class
    /// this table does not hold.
    fn anchor(&self, def_owner: NeuronId, def: &Function) -> Option<Function> {
        if def_owner == self.id {
            let mut f = def.clone();
            f.uses.clear();
            if f.places.iter().any(|(p, _)| matches!(p, Place::Class(k) if self.class(*k).is_none())) { return None; }
            return Some(f);
        }
        if def.places.iter().any(|(p, _)| matches!(p, Place::Class(_))) { return None; }
        let (_, at) = def.places.iter().find(|(p, _)| *p == Place::Fixed(self.id))?;
        let at = *at;
        let mut rest: Vec<(Place, Offset)> = Vec::new();
        for (p, o) in &def.places {
            let mut off = *o;
            for d in 0..MAX_DIMS { off[d] -= at[d]; }
            match p {
                Place::Owner => rest.push((Place::Fixed(def_owner), off)),
                Place::Fixed(m) if *m == self.id && off == [0; MAX_DIMS] => {}
                other => rest.push((*other, off)),
            }
        }
        if rest.iter().any(|(_, o)| o[0] > 0) { return None; }
        Some(Function::new(def.id, rest, def.params.clone()))
    }

    /// Remove a deleted neuron from every pattern and saved activation that names it (R38). Returns the
    /// functions left with nothing besides the owner, which the machine treats as retired.
    pub fn delete_neighbor(&mut self, n: NeuronId) -> Vec<Retired> {
        let mut touched = false;
        for c in self.classes.iter_mut() { let before = c.members.len(); c.members.retain(|m| m.neuron != n); touched |= before != c.members.len(); }
        for f in self.functions.iter_mut() {
            for p in f.params.iter_mut() { p.members.retain(|m| m.neuron != n); }
            if f.places.iter().any(|(p, _)| *p == Place::Fixed(n)) {
                f.places.retain(|(p, _)| *p != Place::Fixed(n));
                f.canonicalize();
                touched = true;
                self.unsettled.insert(f.id);
            }
        }
        let mut out = Vec::new();
        if touched {
            out.extend(self.drop_dead_classes());
            let dead: Vec<PatternId> = self.functions.iter().filter(|f| f.p_size() == 0).map(|f| f.id).collect();
            for fid in dead { out.push(self.remove_function(fid)); }
        }
        for row in self.history.iter_mut() {
            if row.nbs.iter().any(|nb| nb.neuron == n) { row.nbs.retain(|nb| nb.neuron != n); }
        }
        self.connections.retain(|(t, _), _| *t != n);
        out
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        for c in &self.classes { s.push_str(&format!("  {}\n", describe_class(c))); }
        for f in &self.functions { s.push_str(&format!("  p{} {}\n", f.id, describe_function(f))); }
        s
    }
}

/// What update child did (§4.1), for the machine to keep the child's parents (D36).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildUpdate {
    Pointed,
    Adapted,
    Detached,
    Gone,
}

/// A position in the window (D13): `log₂` of the frames it holds in time, the reach plus the frame itself,
/// and of the positions it spans in every other dimension, both ways and the position itself.
pub fn window_bits(window: &Pos, axes: u8) -> f64 {
    let mut bits = 0.0;
    for d in 0..MAX_DIMS {
        if axes & (1 << d) == 0 { continue; }
        let n = if d == 0 { window[d] + 1 } else { 2 * window[d] + 1 };
        bits += log2(n.max(1) as f64);
    }
    bits
}

pub(crate) fn max_pos(a: &Pos, b: &Pos) -> Pos { let mut p = *a; for d in 0..MAX_DIMS { p[d] = p[d].max(b[d]); } p }
