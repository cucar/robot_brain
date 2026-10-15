//! A neuron (§4, §5): its table, its history, its connections, and the `process functions` call.
//!
//! Scope of this implementation (deviations are listed in docs/ucar-results.md):
//! - functions name neighbors and variables (D38); variables are classes of one position (D41) and
//!   parameters (D44) whose positions are the two seed offsets.
//! - every row of the history is re-covered on every call rather than only the rows whose residual
//!   changed (§5.2); the result is the same cover the spec would keep where nothing changed.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::types::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent per phase of the call: refresh+recognize, re-center, delete, create, merge+bids.
pub static TRACE_NEURON: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
pub static PHASE_NS: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
fn tick(i: usize, t: &mut std::time::Instant) { PHASE_NS[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); *t = std::time::Instant::now(); }

/// A reference to an activation the machine holds (frame number and index in that frame).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// A neighbor as a row records it (D26): the neuron, its rounded offset and what it cost.
#[derive(Clone, Debug)]
pub struct Neighbor {
    pub neuron: NeuronId,
    pub offset: Offset,
    pub cost: f64,
    pub act: ActRef,
    /// For a value neuron standing as a neighbor: the value its variable held (D45), read through value neurons.
    pub value: Option<NeuronId>,
}

/// What a variable held in one row and what it was credited (D22's part).
#[derive(Clone, Debug)]
pub struct VarUse {
    pub pattern: PatternId,
    pub value: Vec<NeuronId>,
    pub owned: Vec<usize>,
    pub coverage: f64,
    pub choice: f64,
    pub failed: u32,
}

impl VarUse {
    /// `part(x, O)` (D22): what it holds less its choice and its failures.
    pub fn part(&self, positions: usize) -> f64 {
        self.coverage - self.choice - self.failed as f64 * log2(positions as f64)
    }
}

/// One pattern of a cover with what it was credited (D17, D19).
#[derive(Clone, Debug)]
pub struct CoverEntry {
    pub pattern: PatternId,
    /// Neighbor indices this entry owns directly (a function's own neighbors; a lone variable's positions).
    pub owned: Vec<usize>,
    pub vars: Vec<VarUse>,
    pub coverage: f64,
    pub price: f64,
    pub failed: u32,
    pub credited_bidder: bool,
    /// The which-pattern bits and the bidder's cost the entry was priced with; prices are read at the
    /// current alphabet (D13), so `price` and `coverage` are re-based on these when the entry is read.
    pub wp: f64,
    pub own_cost: f64,
}

impl CoverEntry {
    /// The entry's price at the current which-pattern bits (D13: a price is log2 of the alphabet as it stands).
    pub fn price_now(&self, wp_now: f64) -> f64 { self.price - self.wp + wp_now }
    /// What the entry covers, crediting the bidder's own instance to the first bid of the cover (D31).
    pub fn coverage_now(&self, credited_now: bool) -> f64 {
        self.coverage - if self.credited_bidder { self.own_cost } else { 0.0 } + if credited_now { self.own_cost } else { 0.0 }
    }
    pub fn savings_now(&self, wp_now: f64, credited_now: bool) -> f64 { self.coverage_now(credited_now) - self.price_now(wp_now) }
    pub fn from_measure(pattern: PatternId, m: Measure, own_cost: f64) -> CoverEntry {
        CoverEntry { pattern, owned: m.owned, vars: m.vars, coverage: m.coverage, price: m.price, failed: m.failed, credited_bidder: m.credited_bidder, wp: m.wp, own_cost }
    }
}

/// One activation of the history (D18): the neighborhood it observed, its own cost, and its cover.
#[derive(Clone, Debug)]
pub struct Row {
    pub act: ActRef,
    pub neighbors: Vec<Neighbor>,
    pub own_cost: f64,
    pub cover: Vec<CoverEntry>,
    /// Which pattern of the cover owns each neighbor (D19).
    pub owner: Vec<Option<PatternId>>,
    /// True for the frame's own activations in the call that admitted them.
    pub fresh: bool,
    /// True when the row's residual changed since it was last covered (§5.2).
    pub dirty: bool,
}

/// A dictionary line (D15, D38).
#[derive(Clone, Debug)]
pub enum Body {
    Function {
        neighbors: Vec<(NeuronId, Offset)>,
        variables: Vec<PatternId>,
        child: Option<NeuronId>,
    },
    Class {
        positions: Vec<Offset>,
        members: Vec<NeuronId>,
        values: FxHashMap<Vec<NeuronId>, NeuronId>,
    },
    Parameter {
        positions: Vec<Offset>,
        members: Vec<NeuronId>,
        values: FxHashMap<Vec<NeuronId>, NeuronId>,
    },
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub id: PatternId,
    pub body: Body,
}

impl Pattern {
    pub fn is_function(&self) -> bool { matches!(self.body, Body::Function { .. }) }
    pub fn is_variable(&self) -> bool { !self.is_function() }
    pub fn positions(&self) -> usize {
        match &self.body {
            Body::Function { neighbors, .. } => neighbors.len(),
            Body::Class { positions, .. } | Body::Parameter { positions, .. } => positions.len(),
        }
    }
    pub fn child(&self) -> Option<NeuronId> {
        match &self.body { Body::Function { child, .. } => *child, _ => None }
    }
    pub fn value_neurons(&self) -> Vec<NeuronId> {
        match &self.body {
            Body::Class { values, .. } | Body::Parameter { values, .. } => values.values().copied().collect(),
            _ => Vec::new(),
        }
    }
}

/// A connection (D25): strength and earnings; the estimate is read.
#[derive(Clone, Copy, Debug, Default)]
pub struct Conn {
    pub strength: f64,
    pub earnings: f64,
}

impl Conn {
    pub fn estimate(&self) -> f64 { if self.strength > 0.0 { self.earnings / self.strength } else { 0.0 } }
}

/// Prices the machine hands the neuron with the call (D13, R20).
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    /// log2 n, n the machine's count of its neurons.
    pub neuron_bits: f64,
    /// Bits of one offset field at the owner's reach.
    pub offset_bits: f64,
    /// D27: a position a variable holds is that variable's, so a variable may take a position a
    /// function named first and functions read it as the variable. Off, functions keep what they named.
    /// Whether the pick builds variables at all.
    pub variables: bool,
}

/// A variable carried by a bid (D31).
#[derive(Clone, Debug)]
pub struct BidVar {
    pub pattern: PatternId,
    pub value: Vec<NeuronId>,
    pub value_neuron: Option<NeuronId>,
    pub owned: Vec<usize>,
    pub choice: f64,
    pub failed: u32,
    pub positions: usize,
}

/// What an activation offers the machine (D31, R21).
#[derive(Clone, Debug)]
pub struct Bid {
    pub owner: NeuronId,
    pub bidder: ActRef,
    pub bidder_cost: f64,
    pub neighbors: Vec<Neighbor>,
    /// Functions bid together (sharing a variable), or empty for a variable alone.
    pub functions: Vec<PatternId>,
    pub children: Vec<Option<NeuronId>>,
    pub variables: Vec<BidVar>,
    /// Neighbor indices the functions own directly.
    pub named: Vec<usize>,
    pub failed: u32,
    pub which_pattern_bits: f64,
    /// |p|: what the bid names, neighbors and variable positions.
    pub p_size: usize,
}

#[derive(Clone, Debug)]
pub struct Retired {
    pub pattern: PatternId,
    pub child: Option<NeuronId>,
    pub value_neurons: Vec<NeuronId>,
}

#[derive(Default, Debug)]
pub struct CallResult {
    pub bids: Vec<Bid>,
    pub retired: Vec<Retired>,
    pub added: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Thing {
    Neighbor(NeuronId, Offset),
    Variable(PatternId),
}

#[derive(Clone, Copy, Debug)]
enum Claimant {
    Function(usize),
    Variable(usize),
}

pub struct Measure {
    coverage: f64,
    price: f64,
    failed: u32,
    owned: Vec<usize>,
    vars: Vec<VarUse>,
    credited_bidder: bool,
    wp: f64,
}

/// The neuron (D2, §4.1).
#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: NeuronId,
    pub kind: Kind,
    pub level: u32,
    pub axes: u8,
    pub dim: Option<DimId>,
    pub bucket: u32,
    pub channel: Option<ChannelId>,
    pub table: Vec<Pattern>,
    pub next_pattern: PatternId,
    pub history: VecDeque<Row>,
    pub connections: FxHashMap<(NeuronId, Offset), Conn>,
    pub span: Pos,
    /// The patterns wired to this neuron as their child: (owner, pattern). For a value neuron the
    /// variable, and the value it stands for is in `value`.
    pub parents: Vec<(NeuronId, PatternId)>,
    pub value: Option<Vec<NeuronId>>,
    pub alive: bool,
}

impl Neuron {
    pub fn base(id: NeuronId, kind: Kind, dim: DimId, bucket: u32, channel: ChannelId, axes: u8) -> Self {
        Neuron {
            id, kind, level: 0, axes, dim: Some(dim), bucket, channel: Some(channel),
            table: Vec::new(), next_pattern: 0, history: VecDeque::new(),
            connections: FxHashMap::default(), span: [0; MAX_DIMS], parents: Vec::new(), value: None, alive: true,
        }
    }

    pub fn inter(id: NeuronId, level: u32, axes: u8) -> Self {
        Neuron {
            id, kind: Kind::Inter, level, axes, dim: None, bucket: 0, channel: None,
            table: Vec::new(), next_pattern: 0, history: VecDeque::new(),
            connections: FxHashMap::default(), span: [0; MAX_DIMS], parents: Vec::new(), value: None, alive: true,
        }
    }

    pub fn pattern(&self, id: PatternId) -> Option<&Pattern> { self.table.iter().find(|p| p.id == id) }
    pub fn pattern_mut(&mut self, id: PatternId) -> Option<&mut Pattern> { self.table.iter_mut().find(|p| p.id == id) }
    fn pattern_index(&self, id: PatternId) -> Option<usize> { self.table.iter().position(|p| p.id == id) }

    fn which_pattern_bits(&self) -> f64 { log2(self.table.len() as f64) }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    /// `process functions`: refresh, recognize, re-center, delete, create, merge, return (R20).
    pub fn process_functions(&mut self, rows: Vec<Row>, h: usize, ctx: Ctx) -> CallResult {
        let mut result = CallResult::default();
        let mut t = std::time::Instant::now();
        for r in self.history.iter_mut() { r.fresh = false; }
        // refresh history (§5.1); patterns whose covered rows leave or arrive re-center (D29)
        let mut touched: FxHashSet<PatternId> = FxHashSet::default();
        for mut row in rows {
            if self.history.len() >= h {
                if let Some(old) = self.history.pop_front() {
                    for e in &old.cover { touched.insert(e.pattern); for v in &e.vars { touched.insert(v.pattern); } }
                }
            }
            row.fresh = true;
            row.dirty = true;
            self.history.push_back(row);
        }
        // recognize (§5.2): cover every row whose residual changed
        self.recognize_all(ctx);
        let check = std::env::var("UCAR_CHECK").is_ok();
        if check { self.check_covers("recognize", Some(ctx)); }
        for r in self.history.iter().filter(|r| r.fresh) {
            for e in &r.cover { touched.insert(e.pattern); for v in &e.vars { touched.insert(v.pattern); } }
        }
        tick(0, &mut t);
        // re-center (D29)
        self.recenter_all(ctx, &touched);
        if check { self.check_covers("recenter", Some(ctx)); }
        tick(1, &mut t);
        // delete (R18, R41)
        let retired = self.retire(ctx);
        if !retired.is_empty() {
            result.retired.extend(retired);
            self.recognize_all(ctx);
        }
        if check { self.check_covers("retire", Some(ctx)); }
        tick(2, &mut t);
        // create (D33)
        result.added = self.create(ctx);
        if check { self.check_covers("create", Some(ctx)); }
        tick(3, &mut t);
        // merge (D43)
        let merged = self.merge();
        if !merged.is_empty() {
            result.retired.extend(merged);
            self.recognize_all(ctx);
        }
        if check { self.check_covers("merge", Some(ctx)); }
        // return (§5.6)
        result.bids = self.bids();
        tick(4, &mut t);
        result
    }

    /// Debug invariant (UCAR_CHECK): every neighbor a cover entry owns is owned by that entry's pattern in
    /// the row's owner array and by no other entry; every owner array mark is held by some entry.
    fn check_covers(&self, phase: &str, ctx: Option<Ctx>) {
        for (ri, row) in self.history.iter().enumerate() {
            let mut seen: FxHashMap<usize, PatternId> = FxHashMap::default();
            for e in &row.cover {
                if self.pattern(e.pattern).is_none() { eprintln!("[check {}] row {} @{}: entry of dead p{}", phase, ri, row.act.frame, e.pattern); }
                let mut idx: Vec<usize> = e.owned.clone();
                for v in &e.vars { idx.extend(v.owned.iter().copied()); }
                idx.sort(); idx.dedup();
                for j in idx {
                    if let Some(prev) = seen.insert(j, e.pattern) {
                        eprintln!("[check {}] row {} @{}: neighbor {} ({:?}) owned by p{} and p{}", phase, ri, row.act.frame, j, (row.neighbors[j].neuron, row.neighbors[j].offset[0]), prev, e.pattern);
                    }
                    if row.owner[j] != Some(e.pattern) {
                        eprintln!("[check {}] row {} @{}: neighbor {} ({:?}) in entry p{} but owner[] = {:?}", phase, ri, row.act.frame, j, (row.neighbors[j].neuron, row.neighbors[j].offset[0]), e.pattern, row.owner[j]);
                    }
                }
            }
            for (j, o) in row.owner.iter().enumerate() {
                if let Some(p) = o { if seen.get(&j) != Some(p) { eprintln!("[check {}] row {} @{}: owner[{}] = p{} but no entry owns it", phase, ri, row.act.frame, j, p); } }
            }
            // staleness: an entry re-measured against what it owns plus the residual must come out the same
            if let Some(ctx) = ctx {
                for (k, e) in row.cover.iter().enumerate() {
                    let Some(pi) = self.pattern_index(e.pattern) else { continue };
                    let uncovered: Vec<bool> = row.owner.iter().map(|o| o.is_none() || *o == Some(e.pattern)).collect();
                    let credited = k > 0;
                    let taken_vars: FxHashSet<PatternId> = row.cover.iter().enumerate().filter(|(m, _)| *m != k).flat_map(|(_, x)| x.vars.iter().map(|v| v.pattern)).collect();
                    let claimant = if self.table[pi].is_function() { Claimant::Function(pi) } else { Claimant::Variable(pi) };
                    let m = self.measure(row, &uncovered, claimant, credited, ctx, &taken_vars);
                    let mut a: Vec<usize> = e.owned.iter().copied().chain(e.vars.iter().flat_map(|v| v.owned.iter().copied())).collect(); a.sort(); a.dedup();
                    let mut b: Vec<usize> = m.owned.iter().copied().chain(m.vars.iter().flat_map(|v| v.owned.iter().copied())).collect(); b.sort(); b.dedup();
                    let base_e = e.coverage - if e.credited_bidder { e.own_cost } else { 0.0 };
                    let base_m = m.coverage - if m.credited_bidder { row.own_cost } else { 0.0 };
                    if a != b || (base_m - base_e).abs() > 1e-6 {
                        eprintln!("[stale {}] row {} @{}: p{} entry owns {:?} cov {:.1} price {:.1}; re-measured owns {:?} cov {:.1} price {:.1} ({:?})", phase, ri, row.act.frame, e.pattern, a, e.coverage, e.price, b, m.coverage, m.price, self.table[pi].body);
                    }
                }
            }
        }
    }

    // ───────────────────────────── recognition (D28) ─────────────────────────────

    fn recognize_all(&mut self, ctx: Ctx) {
        let rederive = std::env::var("UCAR_REDERIVE").is_ok();
        for i in 0..self.history.len() {
            if !self.history[i].dirty && !rederive { continue; }
            let mut row = self.history[i].clone();
            if rederive { row.cover.clear(); row.owner = vec![None; row.neighbors.len()]; }
            let (cover, owner) = self.greedy_cover(&row, ctx);
            let r = &mut self.history[i];
            r.cover = cover;
            r.owner = owner;
            r.dirty = false;
        }
    }

    /// After a line moved: re-measure the pattern's entry in every row it covers against what it owned
    /// there plus the residual, so the owners follow the line (D29); an entry that holds nothing leaves,
    /// and what it released is residual for the next recognition.
    fn refresh_entries(&mut self, pid: PatternId, ctx: Ctx) {
        let Some(pi) = self.pattern_index(pid) else { return };
        let is_fn = self.table[pi].is_function();
        for r in 0..self.history.len() {
            let row = self.history[r].clone();
            let Some(k) = row.cover.iter().position(|e| e.pattern == pid) else { continue };
            let e = &row.cover[k];
            // what the entry may claim: its own neighbors and the residual
            let mut uncovered: Vec<bool> = row.owner.iter().map(|o| o.is_none() || *o == Some(pid)).collect();
            let _ = &mut uncovered;
            let credited = row.cover.iter().enumerate().any(|(m, _)| m < k);
            let taken_vars: FxHashSet<PatternId> = row.cover.iter().enumerate().filter(|(m, _)| *m != k).flat_map(|(_, x)| x.vars.iter().map(|v| v.pattern)).collect();
            let claimant = if is_fn { Claimant::Function(pi) } else { Claimant::Variable(pi) };
            let m = self.measure(&row, &uncovered, claimant, credited, ctx, &taken_vars);
            let holds = (m.owned.len() + m.vars.iter().map(|v| v.owned.len()).sum::<usize>()) > 0 && m.coverage > m.price;
            let rowm = &mut self.history[r];
            for &j in &e.owned { rowm.owner[j] = None; }
            for v in &e.vars { for &j in &v.owned { rowm.owner[j] = None; } }
            if holds {
                for &j in &m.owned { rowm.owner[j] = Some(pid); }
                for v in &m.vars { for &j in &v.owned { rowm.owner[j] = Some(pid); } }
                rowm.cover[k] = CoverEntry::from_measure(pid, m, row.own_cost);
            } else {
                rowm.cover.remove(k);
            }
            rowm.dirty = true;
        }
    }

    /// D28 over one row: claimants are functions with their variables, then variables alone.
    /// D28 over the residual of one row: what the row's cover already holds stays, and the claimants
    /// not yet in it compete for the uncovered neighbors (§5.2: a cover is never re-derived).
    fn greedy_cover(&self, row: &Row, ctx: Ctx) -> (Vec<CoverEntry>, Vec<Option<PatternId>>) {
        let mut owner = row.owner.clone();
        let mut cover: Vec<CoverEntry> = row.cover.iter().filter(|e| self.pattern(e.pattern).is_some()).cloned().collect();
        let mut taken: FxHashSet<PatternId> = cover.iter().map(|e| e.pattern).collect();
        let mut taken_vars: FxHashSet<PatternId> = cover.iter().flat_map(|e| e.vars.iter().map(|v| v.pattern)).collect();
        for e in &cover { if self.pattern(e.pattern).map_or(false, |p| p.is_variable()) { taken_vars.insert(e.pattern); } }
        let mut uncovered: Vec<bool> = owner.iter().map(|o| o.is_none()).collect();
        let mut credited = !cover.is_empty();
        loop {
            let mut best: Option<(f64, usize, Measure)> = None;
            for (i, p) in self.table.iter().enumerate() {
                if taken.contains(&p.id) { continue; }
                let claimant = if p.is_function() { Claimant::Function(i) } else {
                    if taken_vars.contains(&p.id) { continue; }
                    if matches!(&p.body, Body::Class { members, .. } if members.is_empty()) { continue; }
                    Claimant::Variable(i)
                };
                let m = self.measure(row, &uncovered, claimant, credited, ctx, &taken_vars);
                if m.coverage <= m.price || m.price <= 0.0 { continue; }
                if m.owned.is_empty() && m.vars.iter().all(|v| v.owned.is_empty()) { continue; }
                let ratio = m.coverage / m.price;
                let better = match &best {
                    None => true,
                    Some((r, bi, _)) => ratio > *r + 1e-12 || ((ratio - *r).abs() <= 1e-12 && p.id < self.table[*bi].id),
                };
                if better { best = Some((ratio, i, m)); }
            }
            let Some((_, i, m)) = best else { break };
            let pid = self.table[i].id;
            for &j in &m.owned { uncovered[j] = false; owner[j] = Some(pid); }
            for v in &m.vars {
                for &j in &v.owned { uncovered[j] = false; owner[j] = Some(pid); }
                taken_vars.insert(v.pattern);
            }
            taken.insert(pid);
            if self.table[i].is_variable() { taken_vars.insert(pid); }
            credited = true;
            cover.push(CoverEntry::from_measure(pid, m, row.own_cost));
        }
        (cover, owner)
    }

    /// Measure one claimant against the still-uncovered neighbors of a row (D28 step 1, D22).
    fn measure(&self, row: &Row, uncovered: &[bool], claimant: Claimant, credited: bool, ctx: Ctx, taken_vars: &FxHashSet<PatternId>) -> Measure {
        let _ = ctx;
        let wp = self.which_pattern_bits();
        match claimant {
            Claimant::Function(i) => {
                let Body::Function { neighbors, variables, .. } = &self.table[i].body else { unreachable!() };
                let mut owned = Vec::new();
                let mut coverage = 0.0;
                let mut failed = 0u32;
                for (n, off) in neighbors {
                    match Self::find_neighbor(row, uncovered, *n, off, &owned) {
                        Some(j) => { owned.push(j); coverage += row.neighbors[j].cost; }
                        None => failed += 1,
                    }
                }
                let mut vars: Vec<VarUse> = Vec::new();
                let mut choices = 0.0;
                let mut p_size = neighbors.len();
                for &vid in variables {
                    let Some(vi) = self.pattern_index(vid) else { continue };
                    let mut claimed: Vec<usize> = owned.clone();
                    for v in &vars { claimed.extend(v.owned.iter().copied()); }
                    let vu = self.measure_variable(row, uncovered, vi, &claimed);
                    p_size += self.table[vi].positions();
                    if taken_vars.contains(&vid) {
                        // a variable already taken alone cannot be carried: every position fails
                        failed += self.table[vi].positions() as u32;
                        continue;
                    }
                    coverage += vu.coverage;
                    choices += vu.choice;
                    failed += vu.failed;
                    vars.push(vu);
                }
                let credited_bidder = !credited;
                if credited_bidder { coverage += row.own_cost; }
                let price = row.own_cost + wp + choices + failed as f64 * log2(p_size as f64);
                Measure { coverage, price, failed, owned, vars, credited_bidder, wp }
            }
            Claimant::Variable(i) => {
                let vu = self.measure_variable(row, uncovered, i, &[]);
                let credited_bidder = !credited;
                let mut coverage = vu.coverage;
                if credited_bidder { coverage += row.own_cost; }
                let positions = self.table[i].positions();
                let price = row.own_cost + wp + vu.choice + vu.failed as f64 * log2(positions as f64);
                let failed = vu.failed;
                let owned = vu.owned.clone();
                Measure { coverage, price, failed, owned, vars: vec![vu], credited_bidder, wp }
            }
        }
    }

    fn find_neighbor(row: &Row, uncovered: &[bool], n: NeuronId, off: &Offset, claimed: &[usize]) -> Option<usize> {
        row.neighbors.iter().enumerate()
            .filter(|(j, nb)| uncovered[*j] && !claimed.contains(j) && nb.neuron == n && nb.offset == *off)
            .map(|(j, _)| j)
            .next()
    }

    /// The value standing at a neighbor for a parameter: read through value neurons (D44).
    fn neighbor_value(nb: &Neighbor) -> NeuronId { nb.value.unwrap_or(nb.neuron) }

    /// A class holds where a member stands at each position; a parameter where one neuron stands at all (D41, D44).
    fn measure_variable(&self, row: &Row, uncovered: &[bool], i: usize, claimed: &[usize]) -> VarUse {
        let p = &self.table[i];
        match &p.body {
            Body::Class { positions, members, values } => {
                let mut owned = Vec::new();
                let mut value = Vec::new();
                let mut coverage = 0.0;
                let mut failed = 0;
                for pos in positions {
                    let found = row.neighbors.iter().enumerate()
                        .filter(|(j, nb)| uncovered[*j] && !claimed.contains(j) && !owned.contains(j) && nb.offset == *pos && members.contains(&nb.neuron))
                        .max_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap());
                    match found {
                        Some((j, nb)) => { owned.push(j); value.push(nb.neuron); coverage += nb.cost; }
                        None => { failed += 1; value.push(members.first().copied().unwrap_or(0)); }
                    }
                }
                let choice = if positions.len() == 1 { log2(members.len() as f64) } else {
                    let stored = values.len().max(1);
                    log2(stored as f64)
                };
                VarUse { pattern: p.id, value, owned, coverage, choice, failed }
            }
            Body::Parameter { positions, members, .. } => {
                // the value is the neuron at most positions, the nearest on a tie
                let mut cands: Vec<(usize, NeuronId, f64)> = Vec::new();
                for pos in positions {
                    for (j, nb) in row.neighbors.iter().enumerate() {
                        if uncovered[j] && !claimed.contains(&j) && nb.offset == *pos { cands.push((j, Self::neighbor_value(nb), nb.cost)); }
                    }
                }
                let mut counts: FxHashMap<NeuronId, usize> = FxHashMap::default();
                for (_, v, _) in &cands { *counts.entry(*v).or_default() += 1; }
                let Some((&val, _)) = counts.iter().max_by_key(|(v, c)| (**c, std::cmp::Reverse(**v))) else {
                    return VarUse { pattern: p.id, value: vec![members.first().copied().unwrap_or(0)], owned: vec![], coverage: 0.0, choice: log2(members.len() as f64), failed: positions.len() as u32 };
                };
                let mut owned = Vec::new();
                let mut coverage = 0.0;
                let mut held = 0usize;
                for pos in positions {
                    if let Some((j, _, c)) = cands.iter().find(|(j, v, _)| *v == val && row.neighbors[*j].offset == *pos && !owned.contains(j)) {
                        owned.push(*j); coverage += c; held += 1;
                    }
                }
                let failed = (positions.len() - held) as u32;
                VarUse { pattern: p.id, value: vec![val], owned, coverage, choice: log2(members.len().max(1) as f64), failed }
            }
            Body::Function { .. } => unreachable!(),
        }
    }

    // ───────────────────────────── re-centering (D29, D27) ─────────────────────────────

    fn recenter_all(&mut self, ctx: Ctx, touched: &FxHashSet<PatternId>) {
        for i in 0..self.table.len() {
            let pid = self.table[i].id;
            if !touched.contains(&pid) && std::env::var("UCAR_ALL_TOUCH").is_err() { continue; }
            match &self.table[i].body {
                Body::Function { .. } => {
                    let uses: Vec<usize> = (0..self.history.len()).filter(|&r| self.history[r].cover.iter().any(|e| e.pattern == pid)).collect();
                    if uses.is_empty() { continue; }
                    let (neighbors, variables) = {
                        let Body::Function { neighbors, variables, .. } = &self.table[i].body else { unreachable!() };
                        (neighbors.clone(), variables.clone())
                    };
                    let seed: Vec<Thing> = neighbors.iter().map(|(n, o)| Thing::Neighbor(*n, *o)).chain(variables.iter().map(|v| Thing::Variable(*v))).collect();
                    let collapsed = self.collapse_function(&uses, &seed, Some(pid), ctx).unwrap_or((Vec::new(), Vec::new()));
                    let (nn, nv) = collapsed;
                    let changed = if let Body::Function { neighbors, variables, .. } = &mut self.table[i].body {
                        let c = *neighbors != nn || *variables != nv;
                        *neighbors = nn;
                        *variables = nv;
                        c
                    } else { false };
                    if changed { self.refresh_entries(pid, ctx); }
                }
                Body::Class { .. } => {
                    let before = self.table[i].body.clone();
                    self.collapse_class(i, ctx);
                    if !same_members(&before, &self.table[i].body) { self.refresh_entries(pid, ctx); }
                }
                Body::Parameter { .. } => {
                    let before = self.table[i].body.clone();
                    self.collapse_parameter_ctx(i, Some(ctx));
                    if !same_positions(&before, &self.table[i].body) { self.refresh_entries(pid, ctx); }
                }
            }
        }
        self.recognize_all(ctx);
    }

    /// True when a neighbor is evidence for a variable candidate: it is in the residual (D21, D33: variables
    /// are built out of the residual, and a candidate skips every owned neighbor, D27).
    fn claimable_by_variable(&self, row: &Row, j: usize) -> bool {
        row.owner[j].is_none()
    }

    /// True when a neighbor counts for an existing variable's collapse (D29): it is in the residual, the
    /// variable owns it where it bid alone, or a function of the cover carries the variable and holds the
    /// neighbor at one of its positions.
    fn held_by(&self, row: &Row, j: usize, pid: PatternId) -> bool {
        match row.owner[j] {
            None => true,
            Some(o) if o == pid => true,
            Some(o) => row.cover.iter().any(|e| e.pattern == o && e.vars.iter().any(|v| v.pattern == pid && v.owned.contains(&j))),
        }
    }

    /// The variable of the table that holds a neighbor's position and admits it, if any (D27).
    fn variable_holding(&self, nb: &Neighbor) -> Option<&Pattern> {
        self.table.iter().find(|p| match &p.body {
            Body::Class { positions, members, .. } => positions.contains(&nb.offset) && members.contains(&nb.neuron),
            Body::Parameter { positions, .. } => positions.contains(&nb.offset),
            _ => false,
        })
    }

    /// Things a row offers a function being collapsed: residual neighbors, what the pattern itself owns,
    /// and the variables standing in the row as things (D27, D33). A neighbor at a position a variable
    /// holds is read as that variable and never as the neuron standing there.
    fn row_things(&self, row: &Row, own: Option<PatternId>) -> Vec<(Thing, f64)> {
        let mut out: Vec<(Thing, f64)> = Vec::new();
        let mut seen: FxHashSet<Thing> = FxHashSet::default();
        for (j, nb) in row.neighbors.iter().enumerate() {
            let free = row.owner[j].is_none() || row.owner[j] == own;
            if !free { continue; }
            if let Some(v) = self.variable_holding(nb) {
                let t = Thing::Variable(v.id);
                if seen.insert(t) {
                    let choice = match &v.body { Body::Class { members, .. } => log2(members.len() as f64), Body::Parameter { members, .. } => log2(members.len().max(1) as f64), _ => 0.0 };
                    out.push((t, nb.cost - choice));
                }
                continue;
            }
            let t = Thing::Neighbor(nb.neuron, nb.offset);
            if seen.insert(t) { out.push((t, nb.cost)); }
        }
        for e in &row.cover {
            let Some(p) = self.pattern(e.pattern) else { continue };
            let inside_own = Some(e.pattern) == own;
            if p.is_variable() || inside_own {
                for v in &e.vars {
                    let Some(vp) = self.pattern(v.pattern) else { continue };
                    let t = Thing::Variable(v.pattern);
                    if seen.insert(t) { out.push((t, v.part(vp.positions()))); }
                }
            }
        }
        out
    }

    /// D27 for a function: every thing is taken when what it covers beats its failures and its entry.
    fn collapse_function(&self, uses: &[usize], seed: &[Thing], own: Option<PatternId>, ctx: Ctx) -> Option<(Vec<(NeuronId, Offset)>, Vec<PatternId>)> {
        let s = uses.len();
        if s == 0 { return None; }
        let mut count: FxHashMap<Thing, usize> = FxHashMap::default();
        let mut value: FxHashMap<Thing, f64> = FxHashMap::default();
        for &r in uses {
            for (t, v) in self.row_things(&self.history[r], own) {
                *count.entry(t).or_default() += 1;
                *value.entry(t).or_default() += v;
            }
        }
        let wp = log2((self.table.len() + if own.is_none() { 1 } else { 0 }) as f64);
        let entry = |t: &Thing| match t {
            Thing::Neighbor(..) => ctx.neuron_bits + ctx.offset_bits,
            Thing::Variable(_) => wp,
        };
        let mut p: Vec<Thing> = seed.to_vec();
        for _ in 0..4 {
            let p_size: usize = p.iter().map(|t| match t { Thing::Neighbor(..) => 1, Thing::Variable(v) => self.pattern(*v).map(|q| q.positions()).unwrap_or(1) }).sum::<usize>().max(1);
            let lp = log2(p_size as f64);
            let mut next: Vec<Thing> = Vec::new();
            for (t, c) in &count {
                let val = value[t];
                if val > (s - c) as f64 * lp + entry(t) { next.push(*t); }
            }
            next.sort_by_key(|t| match t { Thing::Neighbor(n, o) => (0, *n, *o), Thing::Variable(v) => (1, *v, [0; MAX_DIMS]) });
            if next == p { break; }
            p = next;
        }
        let neighbors: Vec<(NeuronId, Offset)> = p.iter().filter_map(|t| if let Thing::Neighbor(n, o) = t { Some((*n, *o)) } else { None }).collect();
        let variables: Vec<PatternId> = p.iter().filter_map(|t| if let Thing::Variable(v) = t { Some(*v) } else { None }).collect();
        if neighbors.is_empty() && variables.len() < 2 { return None; }
        Some((neighbors, variables))
    }

    /// D27 for a class: members by their count while each pays its entry and the widening; combinations stored.
    fn collapse_class(&mut self, i: usize, ctx: Ctx) {
        let pid = self.table[i].id;
        let (positions, members) = { let Body::Class { positions, members, .. } = &self.table[i].body else { unreachable!() }; (positions.clone(), members.clone()) };
        // candidates: neurons that have stood at the positions, in the residual or held by this class
        let mut count: FxHashMap<NeuronId, usize> = FxHashMap::default();
        let mut value: FxHashMap<NeuronId, f64> = FxHashMap::default();
        let mut rows_of: FxHashMap<NeuronId, Vec<usize>> = FxHashMap::default();
        for (r, row) in self.history.iter().enumerate() {
            for (j, nb) in row.neighbors.iter().enumerate() {
                if !self.held_by(row, j, pid) { continue; }
                if !positions.contains(&nb.offset) { continue; }
                *count.entry(nb.neuron).or_default() += 1;
                *value.entry(nb.neuron).or_default() += nb.cost;
                rows_of.entry(nb.neuron).or_default().push(r);
            }
        }
        let mut order: Vec<NeuronId> = count.keys().copied().collect();
        order.sort_by(|a, b| count[b].cmp(&count[a]).then(a.cmp(b)));
        let mut new_members: Vec<NeuronId> = Vec::new();
        let uses_of = |ms: &[NeuronId]| -> usize {
            let mut set: FxHashSet<usize> = FxHashSet::default();
            for m in ms { if let Some(rs) = rows_of.get(m) { set.extend(rs.iter().copied()); } }
            set.len()
        };
        // keep the seed members first (members stand in place of one another: never together in one row)
        let coexists = |a: NeuronId, b: NeuronId, rows_of: &FxHashMap<NeuronId, Vec<usize>>| -> bool {
            let (Some(ra), Some(rb)) = (rows_of.get(&a), rows_of.get(&b)) else { return false };
            ra.iter().any(|r| rb.contains(r))
        };
        // candidates join in order of count while each pays; the first that fails ends the joining,
        // and a member already in leaves only when what it covers no longer pays its entry (D27)
        let mut joining = true;
        for m in &order {
            if new_members.iter().any(|x| coexists(*x, *m, &rows_of)) { continue; }
            let is_member = members.contains(m);
            if !is_member && !joining { continue; }
            let k = new_members.len();
            let uses = uses_of(&new_members);
            let widening = if k == 0 { 0.0 } else { uses as f64 * log2((k + 1) as f64 / k as f64) };
            let pays = value[m] > ctx.neuron_bits + widening;
            if pays || (is_member && value[m] > ctx.neuron_bits) {
                new_members.push(*m);
            } else if !is_member {
                joining = false;
            }
        }
        new_members.sort();
        let mut passed: FxHashSet<Vec<NeuronId>> = FxHashSet::default();
        let mut use_rows: Vec<usize> = Vec::new();
        for (r, row) in self.history.iter().enumerate() {
            for e in &row.cover { for v in &e.vars { if v.pattern == pid { passed.insert(v.value.clone()); use_rows.push(r); } } }
        }
        // its positions: every offset is counted by the uses in which it holds a member, and is a position
        // while what it holds there beats the failures where it holds none and its place in the line (D27)
        let mut new_positions = positions.clone();
        if !use_rows.is_empty() && !new_members.is_empty() && std::env::var("UCAR_NO_POSITION_TESTS").is_err() {
            let mut count: FxHashMap<Offset, (usize, f64)> = FxHashMap::default();
            for &r in &use_rows {
                let row = &self.history[r];
                let mut seen: FxHashSet<Offset> = FxHashSet::default();
                for (j, nb) in row.neighbors.iter().enumerate() {
                    if !self.held_by(row, j, pid) { continue; }
                    if !new_members.contains(&nb.neuron) { continue; }
                    if seen.insert(nb.offset) { let e = count.entry(nb.offset).or_insert((0, 0.0)); e.0 += 1; e.1 += nb.cost; }
                }
            }
            let s_uses = use_rows.len();
            let mut cands: Vec<Offset> = count.keys().copied().collect();
            for p in &positions { if !cands.contains(p) { cands.push(*p); } }
            cands.sort();
            let lx = log2(positions.len().max(1) as f64);
            let mut kept = Vec::new();
            for d in cands {
                let (c, held) = count.get(&d).copied().unwrap_or((0, 0.0));
                let widening = if positions.contains(&d) { 0.0 } else { s_uses as f64 * log2(new_members.len() as f64) };
                if held > (s_uses - c) as f64 * lx + ctx.offset_bits + widening { kept.push(d); }
            }
            if !kept.is_empty() { new_positions = kept; }
        }
        let positions_changed = new_positions != positions;
        if let Body::Class { members, values, positions: ps, .. } = &mut self.table[i].body {
            *members = new_members;
            if positions_changed { *ps = new_positions; values.clear(); }
            else { values.retain(|combo, _| passed.contains(combo) || combo.iter().all(|m| members.contains(m))); }
        }
        if positions_changed { self.dirty_rows_for(pid); }
    }

    /// D27 for a parameter: members are whatever has stood at all its positions.
    fn collapse_parameter(&mut self, i: usize) {
        self.collapse_parameter_ctx(i, None);
    }

    /// D27 for a parameter: members are whatever has stood at all its positions; every offset is a position
    /// while what it holds over the uses beats the failures where it disagrees and its place in the line.
    fn collapse_parameter_ctx(&mut self, i: usize, ctx: Option<Ctx>) {
        let pid = self.table[i].id;
        let positions = { let Body::Parameter { positions, .. } = &self.table[i].body else { unreachable!() }; positions.clone() };
        // the uses: rows where the parameter is in the cover, each with the value it took there
        let mut uses: Vec<(usize, NeuronId, u32)> = Vec::new();
        for (r, row) in self.history.iter().enumerate() {
            for e in &row.cover {
                for v in &e.vars { if v.pattern == pid { uses.push((r, v.value[0], v.failed)); } }
            }
        }
        if let Some(ctx) = ctx {
            if !uses.is_empty() && std::env::var("UCAR_NO_POSITION_TESTS").is_err() {
                // count every offset by the uses in which it holds the value, and what it covers there
                let mut count: FxHashMap<Offset, (usize, f64)> = FxHashMap::default();
                for &(r, val, _) in &uses {
                    let row = &self.history[r];
                    let mut seen: FxHashSet<Offset> = FxHashSet::default();
                    for (j, nb) in row.neighbors.iter().enumerate() {
                        if !self.held_by(row, j, pid) { continue; }
                        if Self::neighbor_value(nb) != val { continue; }
                        if seen.insert(nb.offset) {
                            let e = count.entry(nb.offset).or_insert((0, 0.0));
                            e.0 += 1; e.1 += nb.cost;
                        }
                    }
                }
                let s_uses = uses.len();
                let mut new_positions: Vec<Offset> = Vec::new();
                let mut cands: Vec<Offset> = count.keys().copied().collect();
                for p in &positions { if !cands.contains(p) { cands.push(*p); } }
                cands.sort();
                let lx = log2(positions.len().max(2) as f64);
                for d in cands {
                    let (c, held) = count.get(&d).copied().unwrap_or((0, 0.0));
                    let keep = held > (s_uses - c) as f64 * lx + ctx.offset_bits;
                    if TRACE_NEURON.load(Ordering::Relaxed) == self.id {
                        let failed_uses = uses.iter().filter(|u| u.2 > 0).count();
                        eprintln!("    [position p{} δ={} uses={} (failed in {}) count={} held={:.1} vs {:.1} -> {}]", pid, d[0], s_uses, failed_uses, c, held, (s_uses - c) as f64 * lx + ctx.offset_bits, if keep { "keep" } else { "DROP" });
                    }
                    if keep { new_positions.push(d); }
                }
                if new_positions != positions {
                    if let Body::Parameter { positions: ps, .. } = &mut self.table[i].body { *ps = new_positions; }
                    self.dirty_rows_for(pid);
                }
            }
        }
        let mut members: FxHashSet<NeuronId> = FxHashSet::default();
        for &(_, val, failed) in &uses { if failed == 0 { members.insert(val); } }
        // a value neuron lives while some row of the history still passes its value (D27, R44)
        let passed: FxHashSet<Vec<NeuronId>> = uses.iter().map(|(_, v, _)| vec![*v]).collect();
        if let Body::Parameter { members: ms, values, .. } = &mut self.table[i].body {
            let mut v: Vec<NeuronId> = members.into_iter().collect();
            v.sort();
            if !v.is_empty() { *ms = v; }
            values.retain(|combo, _| passed.contains(combo));
        }
    }

    /// A pattern's line moved: every row it covers is re-measured at the next recognition.
    fn dirty_rows_for(&mut self, pid: PatternId) {
        for r in self.history.iter_mut() {
            if r.cover.iter().any(|e| e.pattern == pid || e.vars.iter().any(|v| v.pattern == pid)) { r.dirty = true; }
        }
    }

    // ───────────────────────────── margins and retirement (D30, R18) ─────────────────────────────

    fn line_cost(&self, p: &Pattern, ctx: Ctx) -> f64 {
        let wp = self.which_pattern_bits();
        match &p.body {
            Body::Function { neighbors, variables, .. } => neighbors.len() as f64 * (ctx.neuron_bits + ctx.offset_bits) + variables.len() as f64 * wp,
            Body::Class { positions, members, values } => positions.len() as f64 * ctx.offset_bits + members.len() as f64 * ctx.neuron_bits + if positions.len() > 1 { values.len() as f64 * positions.len() as f64 * log2(members.len() as f64) } else { 0.0 },
            Body::Parameter { positions, members, .. } => positions.len() as f64 * ctx.offset_bits + members.len() as f64 * ctx.neuron_bits,
        }
    }

    /// benefit(p) over the history (D30).
    fn benefit(&self, pid: PatternId) -> f64 {
        let wp = self.which_pattern_bits();
        let Some(p) = self.pattern(pid) else { return 0.0 };
        let mut total = 0.0;
        for row in &self.history {
            for (k, e) in row.cover.iter().enumerate() {
                if p.is_function() {
                    if e.pattern != pid { continue; }
                    let mut b = e.savings_now(wp, k == 0);
                    for v in &e.vars { b -= (self.part_now(v) - wp).max(0.0); }
                    total += b;
                } else {
                    for v in &e.vars {
                        if v.pattern == pid { total += self.part_now(v); }
                    }
                }
            }
        }
        total
    }

    pub fn margin_pub(&self, pid: PatternId, ctx: Ctx) -> f64 { self.margin(pid, ctx) }

    fn margin(&self, pid: PatternId, ctx: Ctx) -> f64 {
        let Some(p) = self.pattern(pid) else { return 0.0 };
        self.benefit(pid) - self.line_cost(p, ctx)
    }

    /// Every pattern's benefit (D30) in one pass over the history.
    fn benefits(&self) -> FxHashMap<PatternId, f64> {
        let wp = self.which_pattern_bits();
        let mut out: FxHashMap<PatternId, f64> = FxHashMap::default();
        for p in &self.table { out.insert(p.id, 0.0); }
        for row in &self.history {
            for (k, e) in row.cover.iter().enumerate() {
                let Some(p) = self.pattern(e.pattern) else { continue };
                if p.is_function() {
                    let mut b = e.savings_now(wp, k == 0);
                    for v in &e.vars { b -= (self.part_now(v) - wp).max(0.0); }
                    *out.entry(e.pattern).or_default() += b;
                }
                for v in &e.vars { *out.entry(v.pattern).or_default() += self.part_now(v); }
            }
        }
        out
    }

    /// A variable's `part` (D22) in one use at the variable's current line: its choice is read from the
    /// members and combinations it holds now, and a failure costs its current count of positions.
    fn part_now(&self, v: &VarUse) -> f64 {
        let Some(q) = self.pattern(v.pattern) else { return 0.0 };
        v.coverage - self.var_choice(q) - v.failed as f64 * log2(q.positions() as f64)
    }

    /// Which value, as a variable writes it now (D41, D44): a parameter and a one-position class choose a
    /// member, a class of several positions a stored combination.
    fn var_choice(&self, q: &Pattern) -> f64 {
        match &q.body {
            Body::Class { positions, members, values } => if positions.len() == 1 { log2(members.len() as f64) } else { log2(values.len().max(1) as f64) },
            Body::Parameter { members, .. } => log2(members.len().max(1) as f64),
            Body::Function { .. } => 0.0,
        }
    }

    fn retire(&mut self, ctx: Ctx) -> Vec<Retired> {
        let mut gone = Vec::new();
        let benefits = self.benefits();
        let mut dead: Vec<PatternId> = Vec::new();
        for p in &self.table {
            let degenerate = match &p.body {
                Body::Class { positions, members, .. } => members.len() < 2 || (positions.len() == 1 && members.len() == 1),
                Body::Parameter { positions, .. } => positions.len() < 2,
                Body::Function { neighbors, variables, .. } => neighbors.is_empty() && variables.len() < 2,
            };
            let margin = benefits.get(&p.id).copied().unwrap_or(0.0) - self.line_cost(p, ctx);
            if degenerate || margin < 0.0 { dead.push(p.id); }
        }
        for pid in dead { gone.push(self.remove_pattern(pid)); }
        gone
    }

    /// Delete a pattern from the table; functions naming a retired variable drop it (R18).
    pub fn remove_pattern(&mut self, pid: PatternId) -> Retired {
        let idx = self.pattern_index(pid).unwrap();
        let p = self.table.remove(idx);
        for q in self.table.iter_mut() {
            if let Body::Function { variables, .. } = &mut q.body { variables.retain(|v| *v != pid); }
        }
        for row in self.history.iter_mut() {
            if row.cover.iter().any(|e| e.pattern == pid || e.vars.iter().any(|v| v.pattern == pid)) { row.dirty = true; }
            row.cover.retain(|e| e.pattern != pid);
            for o in row.owner.iter_mut() { if *o == Some(pid) { *o = None; } }
        }
        Retired { pattern: pid, child: p.child(), value_neurons: p.value_neurons() }
    }

    // ───────────────────────────── creation (D33, R14, R15) ─────────────────────────────

    const MAX_TRIES: usize = 12;

    fn create(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        if ctx.variables && std::env::var("UCAR_NO_VARIABLES").is_err() { added += self.create_variables(ctx); }
        added += self.create_functions(ctx);
        added
    }

    fn residual_indices(row: &Row) -> Vec<usize> {
        (0..row.neighbors.len()).filter(|&j| row.owner[j].is_none()).collect()
    }

    /// Stage 1: classes from offsets that vary, parameters from pairs of offsets that agree.
    fn create_variables(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        let mut tries = 0;
        loop {
            if tries >= Self::MAX_TRIES { break; }
            // relations over the residual
            // class seeds: per offset, neurons by count
            let mut at: FxHashMap<Offset, FxHashMap<NeuronId, (usize, f64, Vec<usize>)>> = FxHashMap::default();
            // parameter seeds: per pair of offsets, rows where the same value stands at both
            let mut agree: FxHashMap<(Offset, Offset), (usize, f64)> = FxHashMap::default();
            for (r, row) in self.history.iter().enumerate() {
                let res = Self::residual_indices(row);
                for j in 0..row.neighbors.len() {
                    if !self.claimable_by_variable(row, j) { continue; }
                    let nb = &row.neighbors[j];
                    if self.variable_holding(nb).is_some() { continue; }
                    let e = at.entry(nb.offset).or_default().entry(nb.neuron).or_insert((0, 0.0, Vec::new()));
                    e.0 += 1; e.1 += nb.cost; e.2.push(r);
                }
                for a in 0..res.len() {
                    for b in (a + 1)..res.len() {
                        let na = &row.neighbors[res[a]];
                        let nb = &row.neighbors[res[b]];
                        if na.offset == nb.offset { continue; }
                        if Self::neighbor_value(na) != Self::neighbor_value(nb) { continue; }
                        let key = if na.offset < nb.offset { (na.offset, nb.offset) } else { (nb.offset, na.offset) };
                        let e = agree.entry(key).or_insert((0, 0.0));
                        e.0 += 1; e.1 += na.cost + nb.cost;
                    }
                }
            }
            // best class seed: two neurons that stood most at one offset and never together
            let mut best: Option<(f64, usize, SeedKind)> = None;
            for (off, neurons) in &at {
                let mut order: Vec<(&NeuronId, &(usize, f64, Vec<usize>))> = neurons.iter().collect();
                order.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(b.0)));
                if order.len() < 2 { continue; }
                let (a, ea) = order[0];
                let Some((b, eb)) = order[1..].iter().find(|(_, e)| !e.2.iter().any(|r| ea.2.contains(r))) else { continue };
                let count = ea.0 + eb.0;
                if count < 2 || ea.0 < 1 || eb.0 < 1 { continue; }
                let saving = ea.1 + eb.1 - count as f64 * 1.0; // which member, log2 2
                let cand = (saving, count, SeedKind::Class(*off, *a, **b));
                if best.as_ref().map_or(true, |x| cand.0 > x.0 || (cand.0 == x.0 && cand.1 > x.1)) { best = Some(cand); }
            }
            let no_parameters = std::env::var("UCAR_NO_PARAMETERS").is_ok(); // probe only
            for (&(o1, o2), &(count, cost)) in &agree {
                if no_parameters || count < 2 { continue; }
                let saving = cost - count as f64 * 1.0;
                let cand = (saving, count, SeedKind::Parameter(o1, o2));
                if best.as_ref().map_or(true, |x| cand.0 > x.0 || (cand.0 == x.0 && cand.1 > x.1)) { best = Some(cand); }
            }
            let Some((saving, count, seed)) = best else { break };
            if saving <= 0.0 { break; }
            tries += 1;
            let traced = TRACE_NEURON.load(Ordering::Relaxed) == self.id;
            if traced { eprintln!("      [seed n{}] {:?} saving {:.1} count {}", self.id, seed, saving, count); }
            let built = match seed {
                SeedKind::Class(off, a, b) => self.build_class(off, a, b, ctx),
                SeedKind::Parameter(o1, o2) => self.build_parameter(o1, o2, ctx),
            };
            if traced { eprintln!("      [seed n{}] -> {}", self.id, if built { "KEPT" } else { "dropped" }); }
            if built { added += 1; } else { break; }
        }
        added
    }

    /// Build a class at one offset from its seed (D27, R15). Returns whether it was kept.
    fn build_class(&mut self, off: Offset, a: NeuronId, b: NeuronId, ctx: Ctx) -> bool {
        // an existing class with this position: reuse it (D33)
        if self.table.iter().any(|p| matches!(&p.body, Body::Class { positions, .. } if positions.len() == 1 && positions[0] == off)) {
            return false;
        }
        let pid = self.next_pattern;
        self.next_pattern += 1;
        let mut members = vec![a, b];
        members.sort();
        self.table.push(Pattern { id: pid, body: Body::Class { positions: vec![off], members, values: FxHashMap::default() } });
        let i = self.table.len() - 1;
        self.collapse_class(i, ctx);
        let members = { let Body::Class { members, .. } = &self.table[i].body else { unreachable!() }; members.clone() };
        if members.len() < 2 { self.table.pop(); self.next_pattern -= 1; return false; }
        // join the covers where it pays, then price its margin (R15)
        let joined = self.join_variable_covers(pid, ctx);
        if joined == 0 || self.margin(pid, ctx) <= 0.0 {
            self.remove_pattern(pid);
            self.next_pattern -= 1;
            return false;
        }
        true
    }

    /// Build a parameter over two offsets that agree (D44, D27).
    fn build_parameter(&mut self, o1: Offset, o2: Offset, ctx: Ctx) -> bool {
        if self.table.iter().any(|p| matches!(&p.body, Body::Parameter { positions, .. } if positions.len() == 2 && positions[0] == o1 && positions[1] == o2)) {
            return false;
        }
        let pid = self.next_pattern;
        self.next_pattern += 1;
        self.table.push(Pattern { id: pid, body: Body::Parameter { positions: vec![o1, o2], members: Vec::new(), values: FxHashMap::default() } });
        let joined = self.join_variable_covers(pid, ctx);
        let i = self.table.len() - 1;
        self.collapse_parameter(i);
        let members_len = { let Body::Parameter { members, .. } = &self.table[i].body else { unreachable!() }; members.len() };
        if TRACE_NEURON.load(Ordering::Relaxed) == self.id {
            eprintln!("      [parameter n{}] ({},{}) joined {} rows, {:?}, margin {:.1} (line {:.1})", self.id, o1[0], o2[0], joined, self.table[i].body, self.margin(pid, ctx), self.line_cost(&self.table[i], ctx));
        }
        if joined == 0 || members_len == 0 || self.margin(pid, ctx) <= 0.0 {
            self.remove_pattern(pid);
            self.next_pattern -= 1;
            return false;
        }
        true
    }

    /// A new variable joins the cover of every row where it stands alone on the residual and pays (D33:
    /// it owns what it names there). It takes nothing another pattern of the cover owns.
    fn join_variable_covers(&mut self, pid: PatternId, ctx: Ctx) -> usize {
        let i = self.pattern_index(pid).unwrap();
        let mut joined = 0;
        for r in 0..self.history.len() {
            let row = self.history[r].clone();
            let uncovered: Vec<bool> = row.owner.iter().map(|o| o.is_none()).collect();
            let credited = !row.cover.is_empty();
            let m = self.measure(&row, &uncovered, Claimant::Variable(i), credited, ctx, &FxHashSet::default());
            if m.coverage <= m.price || m.vars[0].owned.is_empty() { continue; }
            let owned = m.vars[0].owned.clone();
            let rowm = &mut self.history[r];
            for &j in &owned { debug_assert!(rowm.owner[j].is_none()); rowm.owner[j] = Some(pid); }
            rowm.cover.push(CoverEntry::from_measure(pid, m, row.own_cost));
            joined += 1;
        }
        joined
    }

    /// Stage 2: functions over the rows read in things.
    fn create_functions(&mut self, ctx: Ctx) -> usize {
        let mut added = 0;
        let mut tries = 0;
        let mut rejected: FxHashSet<Vec<Thing>> = FxHashSet::default();
        let mut all_things: Vec<Vec<(Thing, f64)>> = self.history.iter().map(|row| self.row_things(row, None)).collect();
        loop {
            if tries >= Self::MAX_TRIES { break; }
            let wp = log2((self.table.len() + 1) as f64);
            // relations: owner with one thing, and two things in a row
            let mut single: FxHashMap<Thing, (usize, f64)> = FxHashMap::default();
            let mut pair: FxHashMap<(Thing, Thing), (usize, f64)> = FxHashMap::default();
            for things in all_things.iter() {
                for (t, v) in things {
                    let e = single.entry(*t).or_insert((0, 0.0));
                    e.0 += 1; e.1 += v;
                }
                for a in 0..things.len() {
                    for b in (a + 1)..things.len() {
                        let (ta, va) = things[a];
                        let (tb, vb) = things[b];
                        let key = if thing_key(&ta) < thing_key(&tb) { (ta, tb) } else { (tb, ta) };
                        let e = pair.entry(key).or_insert((0, 0.0));
                        e.0 += 1; e.1 += va + vb;
                    }
                }
            }
            let mut best: Option<(f64, usize, Vec<Thing>)> = None;
            let consider = |best: &mut Option<(f64, usize, Vec<Thing>)>, saving: f64, count: usize, seed: Vec<Thing>| {
                if count < 2 { return; }
                let better = match best { None => true, Some((s, c, _)) => saving > *s || (saving == *s && count > *c) };
                if better { *best = Some((saving, count, seed)); }
            };
            for (t, (c, v)) in &single {
                let seed = vec![*t];
                if rejected.contains(&seed) { continue; }
                consider(&mut best, v - *c as f64 * wp, *c, seed);
            }
            for ((a, b), (c, v)) in &pair {
                let seed = vec![*a, *b];
                if rejected.contains(&seed) { continue; }
                consider(&mut best, v - *c as f64 * wp, *c, seed);
            }
            let Some((saving, _, seed)) = best else { break };
            if saving <= 0.0 { break; }
            tries += 1;
            rejected.insert(seed.clone());
            let mut uses: Vec<usize> = (0..self.history.len()).filter(|&r| seed.iter().all(|t| all_things[r].iter().any(|(x, _)| x == t))).collect();
            if std::env::var("UCAR_ALL_ROWS").is_ok() && uses.len() >= 2 { uses = (0..self.history.len()).collect(); }
            if TRACE_NEURON.load(Ordering::Relaxed) == self.id {
                eprintln!("      [pick n{}] seed {:?} saving {:.1} uses {}", self.id, seed, saving, uses.len());
            }
            if let Some(joined) = self.build_function(&seed, &uses, ctx) {
                added += 1;
                for r in joined { all_things[r] = self.row_things(&self.history[r], None); }
            }
        }
        added
    }

    /// Build a function from a seed relation, price it by its margin over the rows holding the seed,
    /// and join the covers it pays in (R14, R15). Returns the rows it joined.
    fn build_function(&mut self, seed: &[Thing], uses: &[usize], ctx: Ctx) -> Option<Vec<usize>> {
        if uses.len() < 2 { return None; }
        let (neighbors, variables) = self.collapse_function(uses, seed, None, ctx)?;
        // identical to an existing function: nothing to add
        if self.table.iter().any(|p| matches!(&p.body, Body::Function { neighbors: n, variables: v, .. } if *n == neighbors && *v == variables)) {
            return None;
        }
        let pid = self.next_pattern;
        self.next_pattern += 1;
        self.table.push(Pattern { id: pid, body: Body::Function { neighbors: neighbors.clone(), variables: variables.clone(), child: None } });
        let i = self.table.len() - 1;
        // price: what each cover saves with C in it less what it saves as it stands
        let mut benefit = 0.0;
        let mut joins: Vec<(usize, Measure)> = Vec::new();
        for &r in uses {
            let row = &self.history[r];
            let named_vars_alone: Vec<usize> = row.cover.iter().enumerate()
                .filter(|(_, e)| self.pattern(e.pattern).map_or(false, |p| p.is_variable()) && variables.contains(&e.pattern))
                .map(|(k, _)| k).collect();
            let mut uncovered: Vec<bool> = row.owner.iter().map(|o| o.is_none()).collect();
            for &k in &named_vars_alone {
                let e = &row.cover[k];
                for &j in &e.owned { uncovered[j] = true; }
            }
            let credited = row.cover.iter().enumerate().any(|(k, _)| !named_vars_alone.contains(&k));
            let taken_vars: FxHashSet<PatternId> = row.cover.iter().enumerate().filter(|(k, _)| !named_vars_alone.contains(k)).flat_map(|(_, e)| e.vars.iter().map(|v| v.pattern)).collect();
            let m = self.measure(row, &uncovered, Claimant::Function(i), credited, ctx, &taken_vars);
            // what the variables it names would save alone there (D30: without f, they bid alone where that pays)
            let wp_now = log2(self.table.len() as f64);
            let alone: f64 = m.vars.iter().map(|v| (self.part_now(v) - wp_now).max(0.0)).sum();
            let gain = m.coverage - m.price - alone;
            if gain > 0.0 && (m.owned.len() + m.vars.iter().map(|v| v.owned.len()).sum::<usize>()) > 0 {
                benefit += gain;
                joins.push((r, m));
            }
        }
        let line = self.line_cost(&self.table[i], ctx);
        if TRACE_NEURON.load(Ordering::Relaxed) == self.id {
            eprintln!("      [build n{}] candidate {:?} vars {:?} benefit {:.1} over {} rows vs line {:.1} -> {}", self.id, neighbors, variables, benefit, joins.len(), line, if benefit > line { "ADD" } else { "no" });
        }
        if benefit <= line {
            self.table.pop();
            self.next_pattern -= 1;
            return None;
        }
        let mut joined = Vec::new();
        for (r, m) in joins {
            let row = &mut self.history[r];
            row.cover.retain(|e| !(variables.contains(&e.pattern)));
            for &j in &m.owned { row.owner[j] = Some(pid); }
            for v in &m.vars { for &j in &v.owned { row.owner[j] = Some(pid); } }
            let own_cost = row.own_cost;
            row.cover.push(CoverEntry::from_measure(pid, m, own_cost));
            joined.push(r);
        }
        Some(joined)
    }

    // ───────────────────────────── merging (D43) ─────────────────────────────

    fn merge(&mut self) -> Vec<Retired> {
        let mut gone = Vec::new();
        let mut i = 0;
        while i < self.table.len() {
            let mut j = i + 1;
            while j < self.table.len() {
                let same = match (&self.table[i].body, &self.table[j].body) {
                    (Body::Function { neighbors: a, variables: va, .. }, Body::Function { neighbors: b, variables: vb, .. }) => a == b && va == vb,
                    (Body::Class { positions: a, members: ma, .. }, Body::Class { positions: b, members: mb, .. }) => a == b && ma == mb,
                    (Body::Parameter { positions: a, .. }, Body::Parameter { positions: b, .. }) => a == b,
                    _ => false,
                };
                if same {
                    // keep the older (i), take the younger's child where the older has none
                    let younger = self.table[j].id;
                    let (yc, yv) = (self.table[j].child(), self.table[j].clone());
                    if let (Body::Function { child: oc, .. }, Some(c)) = (&mut self.table[i].body, yc) {
                        if oc.is_none() { *oc = Some(c); }
                    }
                    if let (Body::Class { values: ov, .. }, Body::Class { values: yvv, .. }) = (&mut self.table[i].body, &yv.body) {
                        for (k, v) in yvv { ov.entry(k.clone()).or_insert(*v); }
                    }
                    let kept_child = self.table[i].child();
                    let mut r = self.remove_pattern(younger);
                    if r.child == kept_child { r.child = None; }
                    gone.push(r);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        gone
    }

    // ───────────────────────────── bids (D31, R21) ─────────────────────────────

    fn bids(&self) -> Vec<Bid> {
        let mut out = Vec::new();
        let wp = self.which_pattern_bits();
        for row in self.history.iter().filter(|r| r.fresh) {
            // functions sharing a variable go as one bid
            let mut used = vec![false; row.cover.len()];
            for k in 0..row.cover.len() {
                if used[k] { continue; }
                let e = &row.cover[k];
                let Some(p) = self.pattern(e.pattern) else { continue };
                if p.is_variable() {
                    used[k] = true;
                    let v = &e.vars[0];
                    let value_neuron = self.value_neuron(p, &v.value);
                    out.push(Bid {
                        owner: self.id, bidder: row.act, bidder_cost: row.own_cost, neighbors: row.neighbors.clone(),
                        functions: vec![], children: vec![],
                        variables: vec![BidVar { pattern: p.id, value: v.value.clone(), value_neuron, owned: v.owned.clone(), choice: v.choice, failed: v.failed, positions: p.positions() }],
                        named: vec![], failed: 0, which_pattern_bits: wp, p_size: p.positions(),
                    });
                    continue;
                }
                let mut group = vec![k];
                used[k] = true;
                let mut vars: FxHashSet<PatternId> = e.vars.iter().map(|v| v.pattern).collect();
                let mut changed = true;
                while changed {
                    changed = false;
                    for m in 0..row.cover.len() {
                        if used[m] { continue; }
                        let f = &row.cover[m];
                        if self.pattern(f.pattern).map_or(true, |q| q.is_variable()) { continue; }
                        if f.vars.iter().any(|v| vars.contains(&v.pattern)) {
                            used[m] = true; group.push(m); changed = true;
                            vars.extend(f.vars.iter().map(|v| v.pattern));
                        }
                    }
                }
                let mut bid = Bid { owner: self.id, bidder: row.act, bidder_cost: row.own_cost, neighbors: row.neighbors.clone(), functions: vec![], children: vec![], variables: vec![], named: vec![], failed: 0, which_pattern_bits: 0.0, p_size: 0 };
                let mut seen_vars: FxHashSet<PatternId> = FxHashSet::default();
                for &m in &group {
                    let f = &row.cover[m];
                    let fp = self.pattern(f.pattern).unwrap();
                    bid.functions.push(f.pattern);
                    bid.children.push(fp.child());
                    bid.named.extend(f.owned.iter().copied());
                    bid.failed += f.failed;
                    bid.which_pattern_bits += wp;
                    if let Body::Function { neighbors, .. } = &fp.body { bid.p_size += neighbors.len(); }
                    for v in &f.vars {
                        if !seen_vars.insert(v.pattern) { continue; }
                        let Some(vp) = self.pattern(v.pattern) else { continue };
                        bid.p_size += vp.positions();
                        bid.variables.push(BidVar { pattern: v.pattern, value: v.value.clone(), value_neuron: self.value_neuron(vp, &v.value), owned: v.owned.clone(), choice: v.choice, failed: v.failed, positions: vp.positions() });
                    }
                }
                out.push(bid);
            }
        }
        out
    }

    fn value_neuron(&self, p: &Pattern, value: &[NeuronId]) -> Option<NeuronId> {
        match &p.body {
            Body::Class { values, .. } | Body::Parameter { values, .. } => values.get(value).copied(),
            _ => None,
        }
    }

    // ───────────────────────────── wiring (R16, R44) ─────────────────────────────

    pub fn wire_child(&mut self, pid: PatternId, child: NeuronId) {
        if let Some(Pattern { body: Body::Function { child: c, .. }, .. }) = self.pattern_mut(pid) { *c = Some(child); }
    }

    pub fn wire_value(&mut self, pid: PatternId, value: Vec<NeuronId>, neuron: NeuronId) {
        if let Some(p) = self.pattern_mut(pid) {
            if let Body::Class { values, .. } | Body::Parameter { values, .. } = &mut p.body { values.insert(value, neuron); }
        }
    }

    /// Remove every reference to a deleted neuron (§4.1 delete call).
    pub fn delete_neighbor(&mut self, dead: NeuronId) -> Vec<Retired> {
        let mut retired = Vec::new();
        let ids: Vec<PatternId> = self.table.iter().map(|p| p.id).collect();
        for pid in ids {
            let p = self.pattern_mut(pid).unwrap();
            let kill = match &mut p.body {
                Body::Function { neighbors, child, .. } => {
                    neighbors.retain(|(n, _)| *n != dead);
                    if *child == Some(dead) { *child = None; }
                    false
                }
                Body::Class { members, values, .. } => {
                    members.retain(|m| *m != dead);
                    values.retain(|k, v| !k.contains(&dead) && *v != dead);
                    members.len() < 2
                }
                Body::Parameter { members, values, .. } => {
                    members.retain(|m| *m != dead);
                    values.retain(|k, v| !k.contains(&dead) && *v != dead);
                    false
                }
            };
            if kill { retired.push(self.remove_pattern(pid)); }
        }
        for row in self.history.iter_mut() {
            let keep: Vec<bool> = row.neighbors.iter().map(|nb| nb.neuron != dead && nb.value != Some(dead)).collect();
            if keep.iter().all(|k| *k) { continue; }
            let mut nn = Vec::new();
            for (j, nb) in row.neighbors.iter().enumerate() { if keep[j] { nn.push(nb.clone()); } }
            row.neighbors = nn;
            row.owner = vec![None; row.neighbors.len()];
            row.cover.clear();
            row.dirty = true;
        }
        self.connections.retain(|(n, _), _| *n != dead);
        self.parents.retain(|(o, _)| *o != dead);
        retired
    }

    /// The things a pattern names with their offsets, for the span (D4) and expansion (R28).
    pub fn pattern_things(&self, pid: PatternId) -> Vec<(Option<NeuronId>, Offset)> {
        let Some(p) = self.pattern(pid) else { return vec![] };
        match &p.body {
            Body::Function { neighbors, variables, .. } => {
                let mut v: Vec<(Option<NeuronId>, Offset)> = neighbors.iter().map(|(n, o)| (Some(*n), *o)).collect();
                for vid in variables {
                    if let Some(vp) = self.pattern(*vid) {
                        if let Body::Class { positions, .. } | Body::Parameter { positions, .. } = &vp.body {
                            for o in positions { v.push((None, *o)); }
                        }
                    }
                }
                v
            }
            Body::Class { positions, .. } | Body::Parameter { positions, .. } => positions.iter().map(|o| (None, *o)).collect(),
        }
    }
}

fn same_positions(a: &Body, b: &Body) -> bool {
    match (a, b) {
        (Body::Parameter { positions: x, .. }, Body::Parameter { positions: y, .. }) => x == y,
        _ => true,
    }
}

fn same_members(a: &Body, b: &Body) -> bool {
    match (a, b) {
        (Body::Class { members: x, positions: px, .. }, Body::Class { members: y, positions: py, .. }) => x == y && px == py,
        _ => true,
    }
}

fn thing_key(t: &Thing) -> (u8, u32, Offset) {
    match t { Thing::Neighbor(n, o) => (0, *n, *o), Thing::Variable(v) => (1, *v, [0; MAX_DIMS]) }
}

#[derive(Clone, Copy, Debug)]
enum SeedKind {
    Class(Offset, NeuronId, NeuronId),
    Parameter(Offset, Offset),
}
