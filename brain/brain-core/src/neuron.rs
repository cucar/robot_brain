//! One neuron (§4, §5): its table, its history and its connections, and the `process functions`
//! call (R20) that reads them: refresh, recognize, delete, create, merge, return.
//!
//! The departures from the document are deliberate and marked:
//! - A node naming a neuron holds where that neuron stands at its offset, or where a child stands that
//!   covers an activation of that neuron at the offset (the child's expansion, R28, read at the row).
//!   Matching (D22), seeding (D47) and the collapse (D27) read it the same way, and the bid then covers
//!   the child. A history row records the apex alone, as D18 says.
//! - Holding is exact (D15): a function holds only where every node holds and a variable only where every
//!   position holds; nothing is priced for a failed node, since nothing fails (against D22's failure term).
//! - The collapse admits nodes one at a time, and a node that holds in only some of the line's uses is
//!   taken where what it covers there beats what the line saves in the uses given up (D27 under D15);
//!   the uses given up fall to the residual. A line that narrows this way keeps its child's name, which
//!   the lines above name, and the child forgets its connections; a child the line shared with another
//!   line is let go, and the line is bought one by its new key (D36, R43).
//! - Base activations reported in one frame are neighbors of one another (`neighborhood.rs`), where D5
//!   admits only earlier waves.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::offsets::*;
use crate::types::*;

/// An open activation on the machine's side: the frame it fired in and its index there.
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
    pub owner_value: NeuronId,
    pub nbs: Vec<Nb>,
}

/// A bid (D31): one claimant of a cover of one of the frame's activations, as the neuron holds it.

pub struct Neuron {
    pub id: NeuronId,
    pub kind: Kind,
    pub dim: Option<DimId>,
    pub bucket: u32,
    pub channel: Option<ChannelId>,
    pub axes: u8,
    pub alive: bool,
    pub table: Vec<Pattern>,
    pub(crate) next_pid: PatternId,
    pub history: Vec<Row>,
    pub h: usize,
    pub connections: FxHashMap<(NeuronId, Offset), Conn>,
    /// An interneuron's parents: every pattern wired to it (R43); for a value neuron, (owner, variable).
    pub parents: Vec<(NeuronId, PatternId)>,
    /// A value neuron's value (D45).
    pub value: Option<Value>,
    /// What an interneuron stands for (D36), as the machine last wrote it.
    pub key: String,
    pub radii: Vec<i32>,
    /// Value neurons of members dropped from a variable since the last call, with the variable (R38).
    pub(crate) dropped_values: Vec<(PatternId, NeuronId)>,
    /// Children of lines that came to hold in fewer places since the last call: a child stands for what
    /// its line names (D36), so what it learned under the places the line gave up is let go (R31).
    pub(crate) narrowed_children: Vec<(PatternId, NeuronId)>,
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind, dim: Some(dim), bucket, channel: Some(channel), axes, alive: true, table: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), value: None, key: format!("b{}", id), radii, dropped_values: Vec::new(), narrowed_children: Vec::new() }
    }

    pub fn inter(id: NeuronId, axes: u8, h: usize, radii: Vec<i32>) -> Self {
        Neuron { id, kind: Kind::Inter, dim: None, bucket: 0, channel: None, axes, alive: true, table: Vec::new(), next_pid: 0, history: Vec::new(), h, connections: FxHashMap::default(), parents: Vec::new(), value: None, key: String::new(), radii, dropped_values: Vec::new(), narrowed_children: Vec::new() }
    }

    pub fn pattern(&self, pid: PatternId) -> Option<&Pattern> { self.table.iter().find(|p| p.id == pid) }
    pub fn pattern_mut(&mut self, pid: PatternId) -> Option<&mut Pattern> { self.table.iter_mut().find(|p| p.id == pid) }
    pub(crate) fn index_of(&self, pid: PatternId) -> Option<usize> { self.table.iter().position(|p| p.id == pid) }

    /// Which pattern: `log₂ |patterns|` (D13).
    pub fn which_pattern_bits(&self) -> f64 { log2(self.table.len() as f64) }

    pub(crate) fn round(&self, delta: &Pos) -> Offset { round_delta(delta, &self.radii, self.axes) }

    /// The line's cost (D13).
    pub fn line_cost(&self, p: &Pattern, ctx: Ctx) -> f64 {
        let wp = self.which_pattern_bits();
        match &p.body {
            Body::Function { nodes, .. } => {
                let mut c = 0.0;
                for (n, _) in nodes {
                    c += match n { Node::Owner => 0.0, Node::Neuron(_) => ctx.neuron_bits, Node::Var(_) => wp };
                }
                c + (nodes.len().saturating_sub(1)) as f64 * ctx.offset_bits
            }
            Body::Class { members, .. } => members.len() as f64 * ctx.neuron_bits,
            Body::Parameter { offsets, members } => (offsets.len().saturating_sub(1)) as f64 * ctx.offset_bits + members.len() as f64 * ctx.neuron_bits,
        }
    }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    /// `process functions`: refresh, recognize, delete, create, merge, return (R20). Returns the bids of
    /// the frame's rows, everything retired, and how many patterns were added.
    pub fn process(&mut self, inputs: Vec<RowInput>, ctx: Ctx) -> (Vec<Bid>, Vec<Retired>, usize) {
        let mut retired: Vec<Retired> = Vec::new();
        // refresh history (§5.1)
        let incoming = inputs.len().min(self.h);
        let mut evicted_patterns: FxHashSet<PatternId> = FxHashSet::default();
        while self.history.len() + incoming > self.h {
            let row = self.history.remove(0);
            for e in &row.cover { evicted_patterns.insert(e.pattern); }
        }
        let first_new = self.history.len();
        for inp in inputs.into_iter().take(incoming) {
            let n = inp.nbs.len();
            self.history.push(Row { act: inp.act, pos: inp.pos, own_cost: inp.own_cost, owner_value: inp.owner_value, nbs: inp.nbs, cover: Vec::new(), owner: vec![None; n], owner_self: None, claim: vec![None; n], dirty: true });
        }
        // recognize (§5.2): cover what changed, then re-center what those covers moved
        let mut changed = self.recognize(ctx);
        changed.extend(evicted_patterns);
        self.recenter(&changed, ctx);
        // delete (§5.3)
        retired.extend(self.retire(ctx));
        // create (§5.4)
        let before = self.next_pid;
        let added = self.create(ctx);
        // merge (§5.5)
        retired.extend(self.merge());
        for (pid, vn) in std::mem::take(&mut self.dropped_values) { retired.push(Retired { pattern: pid, child: None, value_neurons: vec![vn], member_only: true }); }
        if ctx.trace {
            for p in self.table.iter().filter(|p| p.id >= before) { eprintln!("[add f{} #{}] p{} {} margin {:.1}", ctx.frame, self.id, p.id, describe_body(&p.body), self.margin(p.id, ctx)); }
            for r in retired.iter().filter(|r| !r.member_only) { eprintln!("[retire f{} #{}] p{} child={:?}", ctx.frame, self.id, r.pattern, r.child); }
        }
        // return (§5.6)
        let bids = self.bids(first_new);
        (bids, retired, added)
    }

    // ───────────────────────────── the machine's writes (§4.1) ─────────────────────────────

    pub fn wire_child(&mut self, pid: PatternId, child: NeuronId) {
        if let Some(p) = self.pattern_mut(pid) { if let Body::Function { child: c, .. } = &mut p.body { *c = Some(child); } }
    }

    pub fn wire_value(&mut self, var: PatternId, member: NeuronId, value_neuron: NeuronId) {
        if let Some(p) = self.pattern_mut(var) {
            if p.is_variable() {
                match p.members_mut().iter_mut().find(|m| m.neuron == member) {
                    Some(m) => m.value_neuron = Some(value_neuron),
                    None => p.members_mut().push(Member { neuron: member, count: 0, value_neuron: Some(value_neuron) }),
                }
            }
        }
    }

    /// Remove a deleted neuron from every pattern and saved activation that names it (R38). Returns the
    /// patterns left degenerate, which the machine treats as retired.
    pub fn delete_neighbor(&mut self, n: NeuronId) -> Vec<Retired> {
        let mut out = Vec::new();
        let mut dropped: Vec<(PatternId, NeuronId)> = Vec::new();
        let ids: Vec<PatternId> = self.table.iter().map(|p| p.id).collect();
        for pid in ids {
            let degenerate = {
                let p = self.pattern_mut(pid).unwrap();
                match &mut p.body {
                    Body::Function { nodes, child } => {
                        if nodes.iter().any(|(x, _)| *x == Node::Neuron(n)) {
                            let keep: Vec<(Node, Offset)> = nodes.iter().copied().filter(|(x, _)| *x != Node::Neuron(n)).collect();
                            *nodes = canonical(keep);
                            let _ = child;
                        }
                        nodes.len() < 2
                    }
                    Body::Class { members, .. } => { for m in members.iter().filter(|m| m.neuron == n) { if let Some(v) = m.value_neuron { dropped.push((pid, v)); } } members.retain(|m| m.neuron != n); members.len() < 2 }
                    Body::Parameter { members, offsets } => { for m in members.iter().filter(|m| m.neuron == n) { if let Some(v) = m.value_neuron { dropped.push((pid, v)); } } members.retain(|m| m.neuron != n); offsets.len() < 2 }
                }
            };
            if degenerate { out.push(self.remove_pattern(pid)); }
        }
        for (pid, v) in dropped { if self.pattern(pid).is_some() { self.dropped_values.push((pid, v)); } }
        for row in self.history.iter_mut() {
            if row.nbs.iter().any(|nb| nb.neuron == n) {
                let keep: Vec<usize> = (0..row.nbs.len()).filter(|&j| row.nbs[j].neuron != n).collect();
                row.nbs = keep.iter().map(|&j| row.nbs[j].clone()).collect();
                let m = row.nbs.len();
                row.owner = vec![None; m]; row.claim = vec![None; m]; row.owner_self = None; row.cover.clear();
                row.dirty = true;
            }
        }
        self.connections.retain(|(t, _), _| *t != n);
        out
    }

    /// The nodes of a function with their offsets, for expansion (R28) and the index (D36).
    pub fn function_nodes(&self, pid: PatternId) -> Option<&[(Node, Offset)]> {
        match self.pattern(pid).map(|p| &p.body) { Some(Body::Function { nodes, .. }) => Some(nodes), _ => None }
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        for p in &self.table { s.push_str(&format!("  p{} {}\n", p.id, describe_body(&p.body))); }
        s
    }
}
