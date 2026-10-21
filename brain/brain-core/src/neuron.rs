//! One neuron (§4, §5): its table, its history and its connections, and the `process functions`
//! call (R20) that reads them: refresh, recognize, delete, create, merge, return. It follows
//! docs/algorithm.md at 873967e.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::offsets::*;
use crate::types::*;

pub use crate::bid::*;
pub use crate::connection::*;
pub use crate::history::*;
pub use crate::pattern::*;

/// What the machine hands a neuron with its call (R20): the count of neurons and the offset alphabet
/// at the owner's reach (D13).
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    pub neuron_bits: f64,
    pub offset_bits: f64,
    pub frame: Frame,
    pub trace: bool,
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
    /// An interneuron's parents: every function wired to it (R43).
    pub parents: Vec<(NeuronId, PatternId)>,
    /// What an interneuron stands for (D36), as the machine last wrote it.
    pub key: String,
    pub radii: Vec<i32>,
    /// What every interneuron is built over (D36), as the machine's index stood when this call's level
    /// began, each a sorted list: no function names a neuron built over its own child (D27).
    pub built: std::sync::Arc<FxHashMap<NeuronId, Vec<NeuronId>>>,
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind, dim: Some(dim), bucket, channel: Some(channel), axes, alive: true, functions: Vec::new(), classes: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), key: format!("b{}", id), radii, built: Default::default() }
    }

    pub fn inter(id: NeuronId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes, alive: true, functions: Vec::new(), classes: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), key: String::new(), radii, built: Default::default() }
    }

    /// Whether `x` is `c` or built over it: what `x` stands for passes through `c` (D27, D36).
    pub(crate) fn is_built_over(&self, x: NeuronId, c: NeuronId) -> bool {
        x == c || self.built.get(&x).map_or(false, |v| v.binary_search(&c).is_ok())
    }

    pub fn function(&self, fid: PatternId) -> Option<&Function> { self.functions.iter().find(|f| f.id == fid) }
    pub fn function_mut(&mut self, fid: PatternId) -> Option<&mut Function> { self.functions.iter_mut().find(|f| f.id == fid) }
    pub fn class(&self, k: PatternId) -> Option<&Class> { self.classes.iter().find(|c| c.id == k) }
    pub fn class_mut(&mut self, k: PatternId) -> Option<&mut Class> { self.classes.iter_mut().find(|c| c.id == k) }
    pub(crate) fn new_pid(&mut self) -> PatternId { let id = self.next_pid; self.next_pid += 1; id }

    /// Which function: `log₂ |functions|` (D13).
    pub fn which_function_bits(&self) -> f64 { log2(self.functions.len() as f64) }
    /// Which class: `log₂ |classes|`, nothing with one class (D13).
    pub fn which_class_bits(&self) -> f64 { log2(self.classes.len() as f64) }

    pub(crate) fn round(&self, delta: &Pos) -> Offset { round_delta(delta, &self.radii, self.axes) }

    /// A function's entry in the table (D15): a neuron per fixed node, which class per class place, and an
    /// offset per relation, one per place besides the owner.
    pub fn entry(&self, f: &Function, ctx: Ctx) -> f64 {
        f.fixed_count() as f64 * ctx.neuron_bits + f.class_places() as f64 * self.which_class_bits() + f.p_size() as f64 * ctx.offset_bits
    }

    /// A class's entry in the table (D41): a neuron per member.
    pub fn class_entry(&self, c: &Class, ctx: Ctx) -> f64 { c.members.len() as f64 * ctx.neuron_bits }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    /// `process functions`: refresh, recognize, delete, create, merge, return (R20). Returns the bids of
    /// the frame's activations, everything retired, and how many functions were added.
    pub fn process(&mut self, inputs: Vec<RowInput>, ctx: Ctx) -> (Vec<Bid>, Vec<Retired>, usize) {
        let mut retired: Vec<Retired> = Vec::new();
        // refresh history (§5.1)
        let incoming = inputs.len().min(self.h);
        let mut evicted: FxHashSet<PatternId> = FxHashSet::default();
        while self.history.len() + incoming > self.h {
            let row = self.history.remove(0);
            for e in &row.cover { evicted.insert(e.function); }
        }
        let first_new = self.history.len();
        for inp in inputs.into_iter().take(incoming) {
            let n = inp.nbs.len();
            self.history.push(Row { act: inp.act, pos: inp.pos, own_cost: inp.own_cost, nbs: inp.nbs, cover: Vec::new(), owner: vec![None; n], bound: vec![false; n], owner_self: None, dirty: true });
        }
        // the parameters' records follow the history (D44)
        let ids: Vec<PatternId> = self.functions.iter().filter(|f| !f.params.is_empty()).map(|f| f.id).collect();
        for fid in ids { self.record_params(fid); }
        // recognize (§5.2): cover what changed, then re-center what those covers moved
        let mut changed = self.recognize(ctx);
        changed.extend(evicted);
        self.recenter(&changed, ctx);
        // delete (§5.3)
        retired.extend(self.retire(ctx));
        // create (§5.4)
        let before = self.next_pid;
        let added = self.create(ctx);
        // merge (§5.5)
        retired.extend(self.merge(ctx));
        if ctx.trace {
            for f in self.functions.iter().filter(|f| f.id >= before) { eprintln!("[add f{} #{}] p{} {} margin {:.1}", ctx.frame, self.id, f.id, describe_function(f), self.margin(f.id, ctx)); }
            for r in retired.iter() { eprintln!("[retire f{} #{}] p{} child={:?}", ctx.frame, self.id, r.pattern, r.child); }
        }
        // return (§5.6)
        let bids = self.bids(first_new, ctx);
        (bids, retired, added)
    }

    // ───────────────────────────── the machine's writes (§4.1) ─────────────────────────────

    pub fn wire_child(&mut self, fid: PatternId, child: NeuronId) {
        if let Some(f) = self.function_mut(fid) { f.child = Some(child); }
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
            }
        }
        let mut out = Vec::new();
        if touched {
            out.extend(self.drop_dead_classes());
            let dead: Vec<PatternId> = self.functions.iter().filter(|f| f.p_size() == 0).map(|f| f.id).collect();
            for fid in dead { out.push(self.remove_function(fid)); }
        }
        for row in self.history.iter_mut() {
            if row.nbs.iter().any(|nb| nb.neuron == n) {
                row.nbs.retain(|nb| nb.neuron != n);
                let m = row.nbs.len();
                row.owner = vec![None; m]; row.bound = vec![false; m]; row.owner_self = None; row.cover.clear();
                row.dirty = true;
            }
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
