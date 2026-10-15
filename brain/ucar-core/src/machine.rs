//! The machine (§2, §3, §6, §7, §8): frames, activations, the level stack with its elections,
//! the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::neuron::*;
use crate::offsets::*;
use crate::types::*;

/// Which pattern fired an activation and what its variables held (§6.4).
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub owner: NeuronId,
    pub pattern: PatternId,
    pub values: Vec<(PatternId, Vec<NeuronId>)>,
}

/// One standing inference (R45): a connection as it was read at age 0.
#[derive(Clone, Copy, Debug)]
pub struct Inference {
    pub target: NeuronId,
    pub offset: Offset,
    pub strength: f64,
    pub estimate: f64,
}

/// An open activation as the machine holds it (D9).
#[derive(Clone, Debug)]
pub struct Act {
    pub neuron: NeuronId,
    pub kind: Kind,
    pub level: u32,
    pub axes: u8,
    pub pos: Pos,
    pub cost: f64,
    pub strong: bool,
    pub reach: Pos,
    pub covered_at: Option<u32>,
    pub coverer: Option<ActRef>,
    pub covers: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    pub inferences: Vec<Inference>,
}

#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub no: Frame,
    pub acts: Vec<Act>,
    /// The base actions that ran this frame, per channel (D24).
    pub actions_ran: Vec<(ChannelId, ActRef)>,
    pub rewards: Vec<Reward>,
}

/// One symbol the machine outputs for the frame ahead (R47).
#[derive(Clone, Debug)]
pub struct Winner {
    pub dim: DimId,
    pub bucket: u32,
    pub pos: Pos,
    pub kind: Kind,
    pub estimate: f64,
    pub share: f64,
    pub voters: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub frame: Frame,
    pub base_acts: usize,
    pub apex_acts: usize,
    pub apex_cost: f64,
    pub levels: usize,
    pub bids: usize,
    pub accepted: usize,
    pub minted: usize,
    pub retired: usize,
    pub added: usize,
    pub neurons: usize,
    pub patterns: usize,
    pub connections: usize,
    pub voters: usize,
    pub voters_above_base: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub h: usize,
    pub policy: Policy,
    /// Whether the greedy pick builds variables at all (D33 stage 1). Off, every pattern is a function.
    pub variables: bool,
}

struct PendingChild {
    act: Act,
    covered: Vec<ActRef>,
}

pub struct Machine {
    pub cfg: Config,
    pub act_dims: Vec<ActDim>,
    pub radii: Vec<i32>,
    pub dims: Vec<NeuronDim>,
    pub channels: Vec<Channel>,
    pub neurons: Vec<Neuron>,
    pub alive: usize,
    pub base_count: u32,
    pub frames: VecDeque<FrameRec>,
    pub frame_no: Frame,
    pending_weak: Vec<(NeuronId, Pos)>,
    pub max_reach_t: i32,
    ledger: Vec<NeuronId>,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    /// Accumulated wall-clock per phase, in ms: levels, connect, infer, reap, select.
    pub timings: [f64; 5],
}

impl Machine {
    pub fn new(h: usize, policy: Policy, time_radius: i32) -> Self {
        Machine {
            cfg: Config { h, policy, variables: true },
            act_dims: vec![ActDim { name: "time".into(), radius: time_radius.max(1) }],
            radii: vec![time_radius.max(1)],
            dims: Vec::new(),
            channels: Vec::new(),
            neurons: Vec::new(),
            alive: 0,
            base_count: 0,
            frames: VecDeque::new(),
            frame_no: 0,
            pending_weak: Vec::new(),
            max_reach_t: time_radius.max(1),
            ledger: Vec::new(),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            timings: [0.0; 5],
        }
    }

    /// Declare a spatial activation dimension (D1). Returns its index (1 or 2).
    pub fn declare_axis(&mut self, name: &str, radius: i32) -> usize {
        assert!(self.act_dims.len() < MAX_DIMS, "at most two spatial axes");
        self.act_dims.push(ActDim { name: name.into(), radius: radius.max(1) });
        self.radii.push(radius.max(1));
        self.act_dims.len() - 1
    }

    /// Declare a channel with an optional event dimension and action dimension (D3). Base neurons
    /// are minted in declaration order (D2).
    pub fn declare_channel(&mut self, name: &str, event: Option<(&str, u32)>, action: Option<(&str, u32)>, axes: &[usize]) -> ChannelId {
        assert!(self.frame_no == 0, "channels are declared before the first frame");
        let id = self.channels.len() as ChannelId;
        let mut mask: u8 = 1;
        for &a in axes { assert!(a < self.act_dims.len()); mask |= 1 << a; }
        let mut ev = None;
        let mut ac = None;
        if let Some((n, res)) = event { ev = Some(self.declare_dim(n, id, Kind::Event, res, mask)); }
        if let Some((n, res)) = action { ac = Some(self.declare_dim(n, id, Kind::Action, res, mask)); }
        self.channels.push(Channel { id, name: name.into(), event: ev, action: ac, axes: mask });
        id
    }

    fn declare_dim(&mut self, name: &str, channel: ChannelId, kind: Kind, resolution: u32, axes: u8) -> DimId {
        let id = self.dims.len() as DimId;
        let first = self.neurons.len() as NeuronId;
        for b in 0..resolution {
            self.neurons.push(Neuron::base(first + b, kind, id, b, channel, axes));
        }
        self.alive += resolution as usize;
        self.base_count += resolution;
        self.dims.push(NeuronDim { id, name: name.into(), channel, kind, resolution, first });
        id
    }

    pub fn base_neuron(&self, dim: DimId, bucket: u32) -> NeuronId { self.dims[dim as usize].first + bucket }

    fn base_bits(&self) -> f64 { log2(self.base_count as f64) }

    fn reach_of(&self, n: NeuronId) -> Pos {
        let nn = &self.neurons[n as usize];
        let mut r = [0; MAX_DIMS];
        for d in 0..self.radii.len() { if nn.axes & (1 << d) != 0 { r[d] = nn.span[d] + self.radii[d]; } }
        r
    }

    fn ctx_for(&self, n: NeuronId) -> Ctx {
        let reach = self.reach_of(n);
        Ctx { neuron_bits: log2(self.alive as f64), offset_bits: offset_bits(&reach, &self.radii, self.neurons[n as usize].axes), variables: self.cfg.variables }
    }

    fn act(&self, r: ActRef) -> Option<&Act> {
        self.frames.iter().find(|f| f.no == r.frame).and_then(|f| f.acts.get(r.index as usize))
    }

    fn act_mut(&mut self, r: ActRef) -> Option<&mut Act> {
        self.frames.iter_mut().find(|f| f.no == r.frame).and_then(|f| f.acts.get_mut(r.index as usize))
    }

    /// Clear every open activation and standing inference; keep the tables and connections.
    pub fn reset_context(&mut self) {
        self.frames.clear();
        self.pending_weak.clear();
    }

    // ───────────────────────────── one frame ─────────────────────────────

    /// Process one frame: what was observed and what ran, with any rewards (§2.3). Returns the
    /// machine's output for the frame ahead.
    pub fn process_frame(&mut self, inputs: &[Input], rewards: Vec<Reward>) -> Vec<Winner> {
        let start = std::time::Instant::now();
        self.frame_no += 1;
        let fno = self.frame_no;
        let mut stats = FrameStats { frame: fno, ..Default::default() };
        let base_bits = self.base_bits();

        // the frame's base activations: strong where the world reported, weak where only an inference placed (D40)
        let mut frame = FrameRec { no: fno, rewards, ..Default::default() };
        let mut reported: FxHashSet<(DimId, Pos)> = FxHashSet::default();
        for inp in inputs {
            let dim = &self.dims[inp.dim as usize];
            if inp.bucket >= dim.resolution { continue; }
            let mut pos = inp.pos;
            pos[0] = fno as i32;
            if !reported.insert((inp.dim, pos)) { continue; }
            let n = dim.first + inp.bucket;
            let idx = frame.acts.len() as u32;
            frame.acts.push(self.new_act(n, 0, pos, base_bits, true, None));
            if dim.kind == Kind::Action { frame.actions_ran.push((dim.channel, ActRef { frame: fno, index: idx })); }
        }
        let weak = std::mem::take(&mut self.pending_weak);
        for (n, p) in weak {
            let nn = &self.neurons[n as usize];
            let Some(dim) = nn.dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.new_act(n, 0, pos, base_bits, false, None));
        }
        stats.base_acts = frame.acts.len();
        self.frames.push_back(frame);

        // the level stack (§6, R26)
        let t_levels = std::time::Instant::now();
        let mut pending: FxHashMap<u32, Vec<PendingChild>> = FxHashMap::default();
        let mut level: u32 = 0;
        loop {
            // activate children bought for this level (§6.4)
            if let Some(kids) = pending.remove(&level) {
                for kid in kids {
                    let idx = self.current().acts.len() as u32;
                    let r = ActRef { frame: fno, index: idx };
                    let covered = kid.covered.clone();
                    let mut act = kid.act;
                    act.covers = covered.clone();
                    self.current_mut().acts.push(act);
                    for c in covered {
                        if let Some(a) = self.act_mut(c) { if a.coverer.is_none() { a.coverer = Some(r); } }
                    }
                }
            }
            let has_here = self.current().acts.iter().any(|a| a.level == level);
            if has_here {
                stats.levels += 1;
                // process functions (§5)
                let (bids, retired, added) = self.call_level(level);
                stats.bids += bids.len();
                stats.added += added;
                stats.retired += retired.len();
                for (owner, r) in retired { self.handle_retired(owner, r); }
                // elect (R24) and create children (R43)
                let (accepted, minted) = self.elect(level, bids, &mut pending);
                stats.accepted += accepted;
                stats.minted += minted;
            }
            level += 1;
            let more = self.current().acts.iter().any(|a| a.level >= level) || pending.keys().any(|&l| l >= level);
            if !more { break; }
            if level > 64 { break; }
        }

        self.timings[0] += t_levels.elapsed().as_secs_f64() * 1000.0;
        // connect and infer (§7)
        let t = std::time::Instant::now();
        self.connect_all();
        self.timings[1] += t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        self.infer_all();
        self.timings[2] += t.elapsed().as_secs_f64() * 1000.0;
        // delete children due (R38)
        let t = std::time::Instant::now();
        self.refresh_spans();
        self.reap();
        self.timings[3] += t.elapsed().as_secs_f64() * 1000.0;

        // selection (§8)
        let t = std::time::Instant::now();
        let winners = self.select();
        self.timings[4] += t.elapsed().as_secs_f64() * 1000.0;
        stats.voters = self.last_stats.voters;
        stats.voters_above_base = self.last_stats.voters_above_base;
        self.pending_weak = winners.iter().map(|w| (self.base_neuron(w.dim, w.bucket), w.pos)).collect();

        // stats and pruning
        // the settled frame: the oldest one no later bid can still cover (D14's body, per frame)
        let settled = fno - self.max_reach_t as i64;
        if let Some(sf) = self.frames.iter().find(|f| f.no == settled) {
            stats.apex_acts = sf.acts.iter().filter(|a| a.covered_at.is_none()).count();
            stats.apex_cost = sf.acts.iter().filter(|a| a.covered_at.is_none()).map(|a| a.cost).sum();
        }
        stats.neurons = self.alive;
        stats.patterns = self.neurons.iter().filter(|n| n.alive).map(|n| n.table.len()).sum();
        stats.connections = self.neurons.iter().filter(|n| n.alive).map(|n| n.connections.len()).sum();
        let keep_from = fno - self.max_reach_t as i64 - 1;
        while self.frames.front().map_or(false, |f| f.no < keep_from) { self.frames.pop_front(); }
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }

    fn current(&self) -> &FrameRec { self.frames.back().unwrap() }
    fn current_mut(&mut self) -> &mut FrameRec { self.frames.back_mut().unwrap() }

    fn new_act(&self, n: NeuronId, level: u32, pos: Pos, cost: f64, strong: bool, fired_by: Option<FiredBy>) -> Act {
        let nn = &self.neurons[n as usize];
        Act { neuron: n, kind: nn.kind, level, axes: nn.axes, pos, cost, strong, reach: self.reach_of(n), covered_at: None, coverer: None, covers: Vec::new(), fired_by, inferences: Vec::new() }
    }

    // ───────────────────────────── neighborhoods (D5, D7) ─────────────────────────────

    /// The neighborhood of one activation of the current frame: every uncovered activation within
    /// reach on every shared axis, not later in time, that the past table admits (D5).
    fn neighborhood(&self, me: ActRef) -> Vec<Neighbor> {
        let a = self.act(me).unwrap();
        let mut out = Vec::new();
        let reach_t = a.reach[0] as i64;
        for f in self.frames.iter() {
            if f.no < me.frame - reach_t || f.no > me.frame { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me { continue; }
                if b.covered_at.is_some() { continue; }
                if !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                let shared = a.axes & b.axes;
                let mut delta = [0i32; MAX_DIMS];
                let mut ok = true;
                for d in 0..self.radii.len() {
                    if shared & (1 << d) == 0 { continue; }
                    delta[d] = b.pos[d] - a.pos[d];
                    if delta[d].abs() > a.reach[d] { ok = false; break; }
                }
                if !ok { continue; }
                let offset = round_delta(&delta, &self.radii, shared);
                let value = self.value_through(b.neuron);
                out.push(Neighbor { neuron: b.neuron, offset, cost: b.cost, act: r, value });
            }
        }
        out
    }

    /// What a value neuron's activation stands for at its own coordinate, for a parameter reading it (D44,
    /// D45, D7): the bidder its bid covered there, followed through value neurons to a neuron that is not
    /// one. The variable's values sit at the positions the variable holds, not here. None for an activation
    /// that is not a value neuron's.
    #[allow(dead_code)]
    fn stands_for(&self, r: ActRef) -> Option<NeuronId> {
        let a = self.act(r)?;
        if self.neurons[a.neuron as usize].value.is_none() { return None; }
        let mut cur = a.clone();
        for _ in 0..64 {
            let owner = cur.fired_by.as_ref()?.owner;
            if self.neurons[owner as usize].value.is_none() { return Some(owner); }
            // the owner is itself a value neuron standing at the same coordinate: follow its own bidder
            let frame = self.frames.iter().find(|f| f.no == r.frame)?;
            let next = frame.acts.iter().find(|x| x.neuron == owner && x.pos == a.pos)?;
            cur = next.clone();
        }
        None
    }

    /// What a value neuron stands for, for a parameter reading it at one of its positions (D44, D45): the
    /// value its variable held, a member or a one-position combination, followed through value neurons to a
    /// neuron that is not one. A value neuron standing for a combination of several positions is read as
    /// itself. None for a neuron that is not a value neuron.
    fn value_through(&self, n: NeuronId) -> Option<NeuronId> {
        let one = |id: NeuronId| -> Option<NeuronId> {
            match self.neurons[id as usize].value.as_ref() {
                Some(v) if v.len() == 1 => Some(v[0]),
                _ => None,
            }
        };
        self.neurons[n as usize].value.as_ref()?;
        let mut cur = n;
        for _ in 0..64 {
            match one(cur) {
                Some(next) if next != cur => cur = next,
                _ => break,
            }
        }
        Some(cur)
    }

    /// Call every neuron with an activation at this level (R20).
    fn call_level(&mut self, level: u32) -> (Vec<Bid>, Vec<(NeuronId, Retired)>, usize) {
        let fno = self.frame_no;
        let mut by_neuron: FxHashMap<NeuronId, Vec<ActRef>> = FxHashMap::default();
        for (i, a) in self.current().acts.iter().enumerate() {
            if a.level == level { by_neuron.entry(a.neuron).or_default().push(ActRef { frame: fno, index: i as u32 }); }
        }
        let mut neurons: Vec<NeuronId> = by_neuron.keys().copied().collect();
        neurons.sort();
        let mut bids = Vec::new();
        let mut retired = Vec::new();
        let mut added = 0;
        let h = self.cfg.h;
        for n in neurons {
            let mut rows = Vec::new();
            for r in &by_neuron[&n] {
                let nbs = self.neighborhood(*r);
                if nbs.is_empty() { continue; }
                let a = self.act(*r).unwrap();
                let owner = vec![None; nbs.len()];
                rows.push(Row { act: *r, neighbors: nbs, own_cost: a.cost, cover: Vec::new(), owner, fresh: true, dirty: true });
            }
            if rows.is_empty() { continue; }
            if rows.len() > h { rows.truncate(h); }
            let ctx = self.ctx_for(n);
            let tv = std::env::var("UCAR_TRACE").ok();
            let trace = tv.as_deref().and_then(|v| v.parse::<u32>().ok()) == Some(n);
            crate::neuron::TRACE_NEURON.store(if trace { n } else { u32::MAX }, std::sync::atomic::Ordering::Relaxed);
            let result = self.neurons[n as usize].process_functions(rows, h, ctx);
            if tv.as_deref() == Some("churn") && (result.added > 0 || !result.retired.is_empty()) {
                let nn = &self.neurons[n as usize];
                eprintln!("[churn f{} n{} L{}] added={} retired={:?} table={}", self.frame_no, n, nn.level, result.added, result.retired.iter().map(|r| r.pattern).collect::<Vec<_>>(), nn.table.len());
                for p in nn.table.iter().rev().take(result.added) { eprintln!("    + p{} {:?}", p.id, p.body); }
            }
            if trace {
                let nn = &self.neurons[n as usize];
                eprintln!("[trace f{} n{}] rows={} added={} retired={:?} bids={}", self.frame_no, n, nn.history.len(), result.added, result.retired.iter().map(|r| r.pattern).collect::<Vec<_>>(), result.bids.len());
                for p in &nn.table {
                    let margin = nn.margin_pub(p.id, ctx);
                    let uses = nn.history.iter().filter(|r| r.cover.iter().any(|e| e.pattern == p.id || e.vars.iter().any(|v| v.pattern == p.id))).count();
                    eprintln!("    p{} uses={} margin={:.1} {:?}", p.id, uses, margin, p.body);
                }
                for r in nn.history.iter().rev().take(std::env::var("UCAR_TRACE_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1)) {
                    eprintln!("    row@{}: {:?} cover={:?}", r.act.frame, r.neighbors.iter().map(|x| (x.neuron, x.offset[0], (x.cost * 10.0).round() / 10.0)).collect::<Vec<_>>(), r.cover.iter().map(|e| (e.pattern, (e.coverage * 10.0).round() / 10.0, (e.price * 10.0).round() / 10.0, e.vars.iter().map(|v| (v.pattern, v.value.clone(), v.failed)).collect::<Vec<_>>())).collect::<Vec<_>>());
                }
            }
            bids.extend(result.bids);
            added += result.added;
            for r in result.retired { retired.push((n, r)); }
            // spans move when a line re-centers (D4)
            let kids: Vec<NeuronId> = self.neurons[n as usize].table.iter().filter_map(|p| p.child()).collect();
            for k in kids { self.update_span(k); }
        }
        (bids, retired, added)
    }

    fn update_span(&mut self, child: NeuronId) {
        let span = self.span_by_descent(child, &mut Vec::new());
        self.neurons[child as usize].span = span;
        let rt = span[0] + self.radii[0];
        if rt > self.max_reach_t { self.max_reach_t = rt; }
    }

    // ───────────────────────────── election (R22, R24) ─────────────────────────────

    fn measure_bid(&self, bid: &Bid) -> (f64, f64, Vec<ActRef>, bool, u32) {
        let mut coverage = 0.0;
        let mut covered = Vec::new();
        let mut strong = false;
        let mut max_level = 0;
        let free = |r: ActRef| self.act(r).filter(|a| a.covered_at.is_none());
        if let Some(a) = free(bid.bidder) { coverage += a.cost; covered.push(bid.bidder); strong |= a.strong; max_level = max_level.max(a.level); }
        let take = |j: usize, coverage: &mut f64, covered: &mut Vec<ActRef>, strong: &mut bool, max_level: &mut u32| {
            let r = bid.neighbors[j].act;
            if covered.contains(&r) { return; }
            if let Some(a) = free(r) { *coverage += a.cost; covered.push(r); *strong |= a.strong; *max_level = (*max_level).max(a.level); }
        };
        for &j in &bid.named { take(j, &mut coverage, &mut covered, &mut strong, &mut max_level); }
        let mut failed = bid.failed;
        let mut choices = 0.0;
        for v in &bid.variables {
            for &j in &v.owned { take(j, &mut coverage, &mut covered, &mut strong, &mut max_level); }
            failed += v.failed;
            choices += v.choice;
        }
        let price = bid.bidder_cost + bid.which_pattern_bits + choices + failed as f64 * log2(bid.p_size as f64);
        (coverage, price, covered, strong, max_level)
    }

    fn elect(&mut self, level: u32, bids: Vec<Bid>, pending: &mut FxHashMap<u32, Vec<PendingChild>>) -> (usize, usize) {
        let fno = self.frame_no;
        let mut taken = vec![false; bids.len()];
        let mut accepted: Vec<(usize, Vec<ActRef>, f64, bool, u32)> = Vec::new();
        loop {
            let mut best: Option<(f64, usize, f64, f64, Vec<ActRef>, bool, u32)> = None;
            let max_level: u32 = std::env::var("UCAR_MAX_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(u32::MAX);
            for (i, bid) in bids.iter().enumerate() {
                if taken[i] { continue; }
                let (coverage, price, covered, strong, ml) = self.measure_bid(bid);
                if price <= 0.0 || coverage <= price { continue; }
                if ml + 1 > max_level { continue; } // probe only: a cap on the stack
                let ratio = coverage / price;
                let better = match &best {
                    None => true,
                    Some((r, bi, ..)) => {
                        if (ratio - r).abs() > 1e-12 { ratio > *r } else { self.bid_before(bid, &bids[*bi]) }
                    }
                };
                if better { best = Some((ratio, i, coverage, price, covered, strong, ml)); }
            }
            let Some((_, i, _cov, price, covered, strong, ml)) = best else { break };
            taken[i] = true;
            for &r in &covered {
                let age = (fno - r.frame) as u32;
                if let Some(a) = self.act_mut(r) { a.covered_at = Some(age); }
            }
            accepted.push((i, covered, price, strong, ml));
        }
        let n_accepted = accepted.len();
        let mut minted = 0;
        // children (R16, R43): identical accepted bids share one child
        let mut shared: Vec<(Vec<ActRef>, NeuronId)> = Vec::new();
        for (i, covered, price, strong, ml) in accepted {
            let bid = &bids[i];
            let bidder = self.act(bid.bidder).map(|a| (a.pos, a.axes)).unwrap();
            let mut axes = bidder.1;
            for r in &covered { if let Some(a) = self.act(*r) { axes |= a.axes; } }
            let child_level = ml + 1;
            let mut key = covered.clone();
            key.sort_by_key(|r| (r.frame, r.index));
            let values: Vec<(PatternId, Vec<NeuronId>)> = bid.variables.iter().map(|v| (v.pattern, v.value.clone())).collect();
            for (fi, &fid) in bid.functions.iter().enumerate() {
                let child = match bid.children[fi] {
                    Some(c) if self.neurons[c as usize].alive => c,
                    _ => {
                        let reuse = shared.iter().find(|(k, _)| *k == key).map(|(_, c)| *c)
                            .or_else(|| self.find_reusable(bid));
                        let c = match reuse {
                            Some(c) => c,
                            None => { minted += 1; self.mint(child_level, axes, None) }
                        };
                        if reuse.is_some() && std::env::var("UCAR_MINT").map_or(false, |v| v.parse::<i64>().map_or(true, |from| fno >= from)) {
                            eprintln!("  [reuse f{}] #{} for owner #{} p{} covering {:?}", fno, c, bid.owner, fid, covered.iter().map(|r| { let a = self.act(*r).unwrap(); format!("#{}@{}", a.neuron, r.frame) }).collect::<Vec<_>>());
                        }
                        if std::env::var("UCAR_MINT").map_or(false, |v| v.parse::<i64>().map_or(true, |from| fno >= from)) {
                            let body = self.neurons[bid.owner as usize].pattern(fid).map(|p| format!("{:?}", p.body)).unwrap_or_default();
                            eprintln!("  [mint f{}] #{} L{} for owner #{} (L{}) p{} {} covering {:?}", fno, c, child_level, bid.owner, self.neurons[bid.owner as usize].level, fid, body, covered.iter().map(|r| { let a = self.act(*r).unwrap(); format!("#{}@{}", a.neuron, r.frame) }).collect::<Vec<_>>());
                        }
                        self.neurons[bid.owner as usize].wire_child(fid, c);
                        self.neurons[c as usize].parents.push((bid.owner, fid));
                        self.update_span(c);
                        c
                    }
                };
                if !shared.iter().any(|(k, _)| *k == key) { shared.push((key.clone(), child)); }
                let failed_bits = {
                    let f = self.neurons[bid.owner as usize].pattern(fid).map(|_| bid.failed).unwrap_or(0);
                    f as f64 * log2(bid.p_size as f64)
                };
                let cost = bid.bidder_cost + bid.which_pattern_bits / bid.functions.len().max(1) as f64 + failed_bits;
                // a neuron sits at one level (D2): the level it was minted at
                let lvl = self.neurons[child as usize].level.max(level + 1);
                let act = self.new_act(child, lvl, bidder.0, cost, strong, Some(FiredBy { owner: bid.owner, pattern: fid, values: values.clone() }));
                pending.entry(lvl).or_default().push(PendingChild { act, covered: covered.clone() });
            }
            for v in &bid.variables {
                let vn = match v.value_neuron {
                    Some(c) if self.neurons[c as usize].alive => c,
                    _ => {
                        minted += 1;
                        let c = self.mint(child_level, axes, Some(v.value.clone()));
                        if std::env::var("UCAR_MINT").map_or(false, |x| x.parse::<i64>().map_or(true, |from| fno >= from)) {
                            let body = self.neurons[bid.owner as usize].pattern(v.pattern).map(|p| format!("{:?}", p.body)).unwrap_or_default();
                            eprintln!("  [mint f{}] value neuron #{} L{} for owner #{} p{} value {:?} {} covering {:?}", fno, c, child_level, bid.owner, v.pattern, v.value, body, covered.iter().map(|r| { let a = self.act(*r).unwrap(); format!("#{}@{}", a.neuron, r.frame) }).collect::<Vec<_>>());
                        }
                        self.neurons[bid.owner as usize].wire_value(v.pattern, v.value.clone(), c);
                        self.neurons[c as usize].parents.push((bid.owner, v.pattern));
                        self.update_span(c);
                        c
                    }
                };
                let alone = bid.functions.is_empty();
                let cost = if alone { price } else { v.choice };
                let lvl = self.neurons[vn as usize].level.max(level + 1);
                let act = self.new_act(vn, lvl, bidder.0, cost, strong, Some(FiredBy { owner: bid.owner, pattern: v.pattern, values: vec![(v.pattern, v.value.clone())] }));
                let covered_by_value = if alone { covered.clone() } else { Vec::new() };
                pending.entry(lvl).or_default().push(PendingChild { act, covered: covered_by_value });
            }
        }
        (n_accepted, minted)
    }

    /// R24's tie order: a child before none, the older child, the older bidder, the earlier coordinate.
    fn bid_before(&self, a: &Bid, b: &Bid) -> bool {
        let ca = a.children.iter().flatten().min().copied();
        let cb = b.children.iter().flatten().min().copied();
        match (ca, cb) {
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(x), Some(y)) if x != y => return x < y,
            _ => {}
        }
        if a.owner != b.owner { return a.owner < b.owner; }
        let (pa, pb) = (self.act(a.bidder).map(|x| x.pos), self.act(b.bidder).map(|x| x.pos));
        pa < pb
    }

    fn mint(&mut self, level: u32, axes: u8, value: Option<Vec<NeuronId>>) -> NeuronId {
        let id = self.neurons.len() as NeuronId;
        let mut n = Neuron::inter(id, level, axes);
        n.value = value;
        self.neurons.push(n);
        self.alive += 1;
        id
    }

    fn handle_retired(&mut self, owner: NeuronId, r: Retired) {
        let mut orphans = Vec::new();
        if let Some(c) = r.child { orphans.push(c); }
        orphans.extend(r.value_neurons);
        for c in orphans {
            if c as usize >= self.neurons.len() { continue; }
            let n = &mut self.neurons[c as usize];
            n.parents.retain(|(o, p)| !(*o == owner && *p == r.pattern));
            if n.parents.is_empty() && n.alive && !self.ledger.contains(&c) {
                if std::env::var("UCAR_MINT").map_or(false, |v| v.parse::<i64>().map_or(true, |from| self.frame_no >= from)) {
                    eprintln!("  [orphan f{}] #{} after owner #{} retired p{}", self.frame_no, c, owner, r.pattern);
                }
                self.ledger.push(c);
            }
        }
    }

    /// Delete every neuron on the ledger whose last open activation has closed (R38).
    fn reap(&mut self) {
        if self.ledger.is_empty() { return; }
        let fno = self.frame_no;
        let mut i = 0;
        while i < self.ledger.len() {
            let c = self.ledger[i];
            if !self.neurons[c as usize].parents.is_empty() || !self.neurons[c as usize].alive { self.ledger.swap_remove(i); continue; }
            let open = self.frames.iter().any(|f| f.acts.iter().any(|a| a.neuron == c && (fno - f.no) <= a.reach[0] as i64));
            if open { i += 1; continue; }
            self.ledger.swap_remove(i);
            self.delete_neuron(c);
        }
    }

    fn delete_neuron(&mut self, c: NeuronId) {
        if std::env::var("UCAR_MINT").map_or(false, |v| v.parse::<i64>().map_or(true, |from| self.frame_no >= from)) {
            eprintln!("  [death f{}] #{} parents={:?} table={}", self.frame_no, c, self.neurons[c as usize].parents, self.neurons[c as usize].table.len());
        }
        self.neurons[c as usize].alive = false;
        self.alive -= 1;
        let own: Vec<PatternId> = self.neurons[c as usize].table.iter().map(|p| p.id).collect();
        for pid in own {
            let r = self.neurons[c as usize].remove_pattern(pid);
            self.handle_retired(c, r);
        }
        self.neurons[c as usize].connections.clear();
        self.neurons[c as usize].history.clear();
        let mut cascades = Vec::new();
        for n in 0..self.neurons.len() {
            if !self.neurons[n].alive || n == c as usize { continue; }
            let retired = self.neurons[n].delete_neighbor(c);
            for r in retired { cascades.push((n as NeuronId, r)); }
        }
        for (owner, r) in cascades { self.handle_retired(owner, r); }
    }

    // ───────────────────────────── connect (R31, R33) ─────────────────────────────

    fn reward_share_frame0(&self, target: NeuronId) -> f64 {
        let cur = self.current();
        let mut total = 0.0;
        for rw in &cur.rewards {
            if rw.channels.is_empty() { total += rw.reward; continue; }
            let t = &self.neurons[target as usize];
            if t.kind == Kind::Action && t.channel.map_or(false, |ch| rw.channels.contains(&ch)) { total += rw.reward; }
        }
        total
    }

    fn connect_all(&mut self) {
        let fno = self.frame_no;
        let apex: Vec<(ActRef, NeuronId, Kind, u8, Pos)> = self.current().acts.iter().enumerate()
            .filter(|(_, a)| a.covered_at.is_none() && a.strong)
            .map(|(i, a)| (ActRef { frame: fno, index: i as u32 }, a.neuron, a.kind, a.axes, a.pos)).collect();
        let span_default = self.max_reach_t.max(1);
        let rewards = self.current().rewards.clone();
        let mut writes: Vec<(NeuronId, NeuronId, Offset, f64)> = Vec::new();
        let frames: Vec<(Frame, usize)> = self.frames.iter().map(|f| (f.no, f.acts.len())).collect();
        for (f_no, count) in frames {
            if f_no >= fno { continue; }
            let age = (fno - f_no) as u32;
            for i in 0..count {
                let r = ActRef { frame: f_no, index: i as u32 };
                let a = self.act(r).unwrap().clone();
                if !a.strong || age as i32 > a.reach[0] { continue; }
                match a.covered_at {
                    None => {
                        for (_, bn, bk, baxes, bpos) in &apex {
                            if !self.cfg.policy.future[a.kind as usize][*bk as usize] { continue; }
                            let Some(off) = self.offset_between(&a, *baxes, bpos, age) else { continue };
                            let share = self.reward_share_frame0(*bn);
                            writes.push((a.neuron, *bn, off, share));
                        }
                    }
                    Some(c) if c == age => {
                        if let Some(cov) = a.coverer.and_then(|cr| self.act(cr)).cloned() {
                            if cov.strong && self.cfg.policy.future[a.kind as usize][cov.kind as usize] {
                                if let Some(off) = self.offset_between(&a, cov.axes, &cov.pos, age) {
                                    let share = self.reward_share_frame0(cov.neuron);
                                    writes.push((a.neuron, cov.neuron, off, share));
                                }
                            }
                        }
                    }
                    _ => {}
                }
                // shares for earlier frames (R33)
                for rw in &rewards {
                    let s = if rw.frames == 0 { span_default as u32 } else { rw.frames };
                    let maxd = s.min(age + 1);
                    for d in 1..maxd {
                        if a.covered_at.map_or(false, |c| c <= age - d) { continue; }
                        let share = rw.reward * (s - d) as f64 / s as f64;
                        let f_then = fno - d as i64;
                        let Some(fr) = self.frames.iter().find(|f| f.no == f_then) else { continue };
                        let ran: Vec<(ChannelId, ActRef)> = fr.actions_ran.clone();
                        for (ch, ar) in ran {
                            if !rw.channels.is_empty() && !rw.channels.contains(&ch) { continue; }
                            let Some(b) = self.act(ar).cloned() else { continue };
                            if !self.cfg.policy.future[a.kind as usize][b.kind as usize] { continue; }
                            let Some(off) = self.offset_between(&a, b.axes, &b.pos, age - d) else { continue };
                            let key = (b.neuron, off);
                            if self.neurons[a.neuron as usize].connections.contains_key(&key) {
                                self.add_earnings(a.neuron, b.neuron, off, share);
                            }
                        }
                    }
                }
            }
        }
        for (holder, target, off, share) in writes {
            let c = self.neurons[holder as usize].connections.entry((target, off)).or_default();
            c.strength += 1.0;
            if share != 0.0 { self.add_earnings(holder, target, off, share); }
        }
    }

    /// The offset from an open activation to an apex activation: its age in time and the rounded
    /// distance in every other shared axis; none when out of reach.
    fn offset_between(&self, a: &Act, baxes: u8, bpos: &Pos, age: u32) -> Option<Offset> {
        let shared = a.axes & baxes;
        let mut delta = [0i32; MAX_DIMS];
        delta[0] = age as i32;
        for d in 1..self.radii.len() {
            if shared & (1 << d) == 0 { continue; }
            delta[d] = bpos[d] - a.pos[d];
            if delta[d].abs() > a.reach[d] { return None; }
        }
        Some(round_delta(&delta, &self.radii, shared | 1))
    }

    fn add_earnings(&mut self, holder: NeuronId, target: NeuronId, off: Offset, share: f64) {
        let est = {
            let c = self.neurons[holder as usize].connections.entry((target, off)).or_default();
            c.earnings += share;
            c.estimate()
        };
        if est < 0.0 { self.walk(holder, target, off); }
    }

    /// R37: a negative estimate wires the next untried base action, in each dimension the target acted in.
    fn walk(&mut self, holder: NeuronId, target: NeuronId, off: Offset) {
        let mut dims: Vec<DimId> = Vec::new();
        let t = &self.neurons[target as usize];
        if t.kind == Kind::Action { dims.push(t.dim.unwrap()); }
        else if t.kind == Kind::Inter {
            let mut placed = Vec::new();
            self.expand(target, [0; MAX_DIMS], &mut Vec::new(), &mut placed, None);
            for (n, _) in placed {
                let nn = &self.neurons[n as usize];
                if nn.kind == Kind::Action { if let Some(d) = nn.dim { if !dims.contains(&d) { dims.push(d); } } }
            }
        }
        for d in dims {
            let dim = &self.dims[d as usize];
            for b in 0..dim.resolution {
                let n = dim.first + b;
                let key = (n, off);
                if !self.neurons[holder as usize].connections.contains_key(&key) {
                    self.neurons[holder as usize].connections.insert(key, Conn { strength: 1.0, earnings: 0.0 });
                    break;
                }
            }
        }
    }

    // ───────────────────────────── infer (R36) ─────────────────────────────

    fn infer_all(&mut self) {
        let fno = self.frame_no;
        let cur = self.frames.back_mut().unwrap();
        let _ = fno;
        for a in cur.acts.iter_mut() {
            let n = &self.neurons[a.neuron as usize];
            a.inferences = n.connections.iter().map(|((t, o), c)| Inference { target: *t, offset: *o, strength: c.strength, estimate: c.estimate() }).collect();
        }
    }

    // ───────────────────────────── expansion (R28) ─────────────────────────────

    /// Expand a neuron placed at a relative coordinate to base symbols at their relative coordinates (R28).
    /// `path` is the descent so far: the wave coming down passes through a neuron once, and a neuron met
    /// again on its own descent, as happens once a chain has closed on itself, is placed as its bidder's
    /// symbol at that coordinate and not expanded further.
    fn expand(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, fired_by: Option<&FiredBy>) {
        if path.len() > 64 || out.len() > 4096 { return; }
        let nn = &self.neurons[n as usize];
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        let Some(&(owner, pid)) = fired_by.map(|f| (f.owner, f.pattern)).as_ref().or(nn.parents.first()) else { return };
        let on = &self.neurons[owner as usize];
        let Some(p) = on.pattern(pid) else { return };
        let again = path.contains(&n);
        path.push(n);
        // the bidder at the neuron's own coordinate
        self.expand(owner, at, path, out, None);
        if !again {
            let add = |o: &Offset| { let mut p = at; for d in 0..MAX_DIMS { p[d] += o[d]; } p };
            match &p.body {
                Body::Function { neighbors, variables, .. } => {
                    for (m, o) in neighbors { self.expand(*m, add(o), path, out, None); }
                    for vid in variables {
                        let value = fired_by.and_then(|f| f.values.iter().find(|(v, _)| v == vid).map(|(_, val)| val.clone()));
                        self.expand_variable(on, *vid, value.as_deref(), at, path, out);
                    }
                }
                Body::Class { .. } | Body::Parameter { .. } => {
                    let value = nn.value.clone().or_else(|| fired_by.and_then(|f| f.values.first().map(|(_, v)| v.clone())));
                    self.expand_variable(on, pid, value.as_deref(), at, path, out);
                }
            }
        }
        path.pop();
    }

    fn expand_variable(&self, owner: &Neuron, vid: PatternId, value: Option<&[NeuronId]>, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>) {
        let Some(vp) = owner.pattern(vid) else { return };
        let add = |o: &Offset| { let mut p = at; for d in 0..MAX_DIMS { p[d] += o[d]; } p };
        match &vp.body {
            Body::Class { positions, members, .. } => {
                for (k, o) in positions.iter().enumerate() {
                    let m = value.and_then(|v| v.get(k).copied()).or_else(|| members.first().copied());
                    if let Some(m) = m { self.expand(m, add(o), path, out, None); }
                }
            }
            Body::Parameter { positions, members, .. } => {
                let m = value.and_then(|v| v.first().copied()).or_else(|| members.first().copied());
                if let Some(m) = m { for o in positions { self.expand(m, add(o), path, out, None); } }
            }
            Body::Function { .. } => {}
        }
    }

    // ───────────────────────────── windows and global reuse (R28, R43) ─────────────────────────────

    /// What an activation stands for, placed relative to `origin` and read within `horizon`: its neuron
    /// expanded with the values it fired with, cut at the horizon on every axis.
    fn window_of_act(&self, r: ActRef, origin: Pos, horizon: Pos, out: &mut Vec<(NeuronId, Pos)>) {
        let Some(a) = self.act(r) else { return };
        let mut at = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS { at[d] = a.pos[d] - origin[d]; }
        let mut raw = Vec::new();
        self.expand(a.neuron, at, &mut Vec::new(), &mut raw, a.fired_by.as_ref());
        out.extend(raw.into_iter().filter(|(_, p)| (0..MAX_DIMS).all(|d| p[d].abs() <= horizon[d])));
    }

    /// The window a function bid's child would stand for: the bidder and everything the bid names, read
    /// within the owner's reach.
    fn window_of_bid(&self, bid: &Bid, horizon: Pos) -> Vec<(NeuronId, Pos)> {
        let Some(b) = self.act(bid.bidder) else { return vec![] };
        let origin = b.pos;
        let mut out = Vec::new();
        self.window_of_act(bid.bidder, origin, horizon, &mut out);
        for &j in &bid.named { self.window_of_act(bid.neighbors[j].act, origin, horizon, &mut out); }
        for v in &bid.variables { for &j in &v.owned { self.window_of_act(bid.neighbors[j].act, origin, horizon, &mut out); } }
        out.sort(); out.dedup();
        out
    }

    /// The window a neuron stands for, by its own definition, read within `horizon`.
    fn window_of_neuron(&self, n: NeuronId, horizon: Pos) -> Vec<(NeuronId, Pos)> {
        let mut raw = Vec::new();
        self.expand(n, [0; MAX_DIMS], &mut Vec::new(), &mut raw, None);
        let mut out: Vec<(NeuronId, Pos)> = raw.into_iter().filter(|(_, p)| (0..MAX_DIMS).all(|d| p[d].abs() <= horizon[d])).collect();
        out.sort(); out.dedup();
        out
    }

    /// Global reuse: the living call neuron that stands, within the owner's reach, for exactly what this
    /// bid's child would stand for, if any. Matching is by what the neurons expand to, not by which
    /// lines state it, so a chain that comes round to a window it has stood for before fires the neuron
    /// it minted then instead of a new one.
    fn find_reusable(&self, bid: &Bid) -> Option<NeuronId> {
        if std::env::var("UCAR_NO_REUSE").is_ok() { return None; }
        let horizon = self.reach_of(bid.owner);
        let w = self.window_of_bid(bid, horizon);
        if w.is_empty() { return None; }
        self.neurons.iter()
            .filter(|n| n.alive && n.kind == Kind::Inter && n.value.is_none() && !n.parents.is_empty())
            .find(|n| self.window_of_neuron(n.id, horizon) == w)
            .map(|n| n.id)
    }

    /// A neuron's span by descent (D4): the farthest base symbol under it, where a neuron met again on its
    /// own descent counts as its bidder's symbol and no further, so a chain closed on itself spans one turn.
    fn span_by_descent(&self, n: NeuronId, path: &mut Vec<NeuronId>) -> Pos {
        let nn = &self.neurons[n as usize];
        let mut span = [0i32; MAX_DIMS];
        if nn.kind != Kind::Inter || !nn.alive || path.contains(&n) || path.len() > 64 { return span; }
        path.push(n);
        for &(owner, pid) in &nn.parents {
            if !self.neurons[owner as usize].alive { continue; }
            let os = self.span_by_descent(owner, path);
            for d in 0..MAX_DIMS { span[d] = span[d].max(os[d]); }
            for (m, off) in self.neurons[owner as usize].pattern_things(pid) {
                let inner = m.map(|x| self.span_by_descent(x, path)).unwrap_or([0; MAX_DIMS]);
                for d in 0..MAX_DIMS { span[d] = span[d].max(off[d].abs() + inner[d]); }
            }
        }
        path.pop();
        span
    }

    /// Recompute every living interneuron's span by descent (D4). Spans move only when lines move, so
    /// this runs once per frame after the waves.
    fn refresh_spans(&mut self) {
        let ids: Vec<NeuronId> = self.neurons.iter().filter(|n| n.alive && n.kind == Kind::Inter).map(|n| n.id).collect();
        for id in ids {
            let span = self.span_by_descent(id, &mut Vec::new());
            self.neurons[id as usize].span = span;
            let rt = span[0] + self.radii[0];
            if rt > self.max_reach_t { self.max_reach_t = rt; }
        }
    }

    // ───────────────────────────── selection (R45–R48) ─────────────────────────────

    /// What one open activation places at the frame ahead (R46): base symbols its standing inferences expand to there.
    fn placements(&self, r: ActRef, ahead: Frame) -> Vec<(NeuronId, Pos, f64, f64)> {
        let Some(a) = self.act(r) else { return vec![] };
        let mut out = Vec::new();
        for inf in &a.inferences {
            if !self.neurons[inf.target as usize].alive { continue; }
            let mut at = a.pos;
            for d in 0..MAX_DIMS { at[d] += inf.offset[d]; }
            // a cheap bound: nothing placed can reach ahead if the completion is far behind it
            if (at[0] as i64) + self.neurons[inf.target as usize].span[0] as i64 + 1 < ahead { continue; }
            let mut placed = Vec::new();
            self.expand(inf.target, at, &mut Vec::new(), &mut placed, None);
            for (n, p) in placed {
                if p[0] as i64 != ahead { continue; }
                let nn = &self.neurons[n as usize];
                let mut pp = p;
                for d in 1..MAX_DIMS { if nn.axes & (1 << d) == 0 { pp[d] = 0; } }
                out.push((n, pp, inf.strength, inf.estimate));
            }
        }
        out
    }

    fn select(&mut self) -> Vec<Winner> {
        let fno = self.frame_no;
        let ahead = fno + 1;
        // the electorate (R46, R48): uncovered open activations, and covered ones whose coverer has nothing
        let mut voters: Vec<(ActRef, Vec<(NeuronId, Pos, f64, f64)>)> = Vec::new();
        let mut stack: Vec<ActRef> = Vec::new();
        for f in self.frames.iter() {
            let age = fno - f.no;
            for (i, a) in f.acts.iter().enumerate() {
                let open = (a.strong && age <= a.reach[0] as i64) || age == 0;
                if !open || a.covered_at.is_some() { continue; }
                stack.push(ActRef { frame: f.no, index: i as u32 });
            }
        }
        // probe: UCAR_ALL_VOTE lets every open activation vote, covered or not (a deviation from D10)
        if std::env::var("UCAR_ALL_VOTE").is_ok() {
            stack.clear();
            for f in self.frames.iter() {
                let age = fno - f.no;
                for (i, a) in f.acts.iter().enumerate() {
                    let open = (a.strong && age <= a.reach[0] as i64) || age == 0;
                    if open { stack.push(ActRef { frame: f.no, index: i as u32 }); }
                }
            }
        }
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        while let Some(r) = stack.pop() {
            if !seen.insert(r) { continue; }
            let p = self.placements(r, ahead);
            if !p.is_empty() { voters.push((r, p)); continue; }
            if let Some(a) = self.act(r) {
                for c in &a.covers {
                    if let Some(ca) = self.act(*c) {
                        let age = fno - c.frame;
                        let open = (ca.strong && age <= ca.reach[0] as i64) || age == 0;
                        if open { stack.push(*c); }
                    }
                }
            }
        }
        if std::env::var("UCAR_VOTES").map_or(false, |v| v.parse::<i64>().map_or(true, |from| fno >= from)) {
            for (r, placed) in &voters {
                let a = self.act(*r).unwrap();
                let mut desc: Vec<String> = placed.iter().filter(|(n, ..)| self.neurons[*n as usize].kind == Kind::Action)
                    .map(|(n, _, s, e)| format!("#{}:s{:.0}/e{:+.2}", n, s, e)).collect();
                if desc.is_empty() { continue; }
                desc.sort();
                eprintln!("  [votes f{}] voter #{} L{} age{} {}{} -> {}", fno, a.neuron, a.level, fno - r.frame, if a.covered_at.is_some() { "(covered) " } else { "" }, if a.strong { "" } else { "(weak) " }, desc.join(" "));
            }
        }
        self.last_stats.voters = voters.len();
        self.last_stats.voters_above_base = voters.iter().filter(|(r, _)| self.act(*r).map_or(false, |a| a.level > 0)).count();
        // resolution (R47)
        struct Cand { share: f64, weighted: f64 }
        let mut table: FxHashMap<(DimId, Pos), FxHashMap<u32, Cand>> = FxHashMap::default();
        let mut voters_at: FxHashMap<(DimId, Pos), usize> = FxHashMap::default();
        for (_, placed) in &voters {
            let mut totals: FxHashMap<(DimId, Pos), f64> = FxHashMap::default();
            for (n, p, s, _) in placed {
                let nn = &self.neurons[*n as usize];
                let Some(d) = nn.dim else { continue };
                *totals.entry((d, *p)).or_default() += s;
            }
            for (k, _) in &totals { *voters_at.entry(*k).or_default() += 1; }
            for (n, p, s, e) in placed {
                let nn = &self.neurons[*n as usize];
                let Some(d) = nn.dim else { continue };
                let share = s / totals[&(d, *p)];
                let c = table.entry((d, *p)).or_default().entry(nn.bucket).or_insert(Cand { share: 0.0, weighted: 0.0 });
                c.share += share;
                c.weighted += share * e;
            }
        }
        let mut winners = Vec::new();
        let mut keys: Vec<(DimId, Pos)> = table.keys().copied().collect();
        keys.sort();
        for key in keys {
            let cands = &table[&key];
            let dim = &self.dims[key.0 as usize];
            let mut best: Option<(u32, f64, f64)> = None;
            for (b, c) in cands {
                let est = c.weighted / c.share;
                let better = match best {
                    None => true,
                    Some((bb, be, bs)) => match dim.kind {
                        Kind::Action => est > be || (est == be && (c.share > bs || (c.share == bs && *b < bb))),
                        _ => c.share > bs || (c.share == bs && *b < bb),
                    },
                };
                if better { best = Some((*b, est, c.share)); }
            }
            let Some((b, est, share)) = best else { continue };
            if dim.kind == Kind::Action && est < 0.0 && self.cfg.policy.action == ActionPolicy::Nothing { continue; }
            let mut pos = key.1;
            pos[0] = ahead as i32;
            winners.push(Winner { dim: key.0, bucket: b, pos, kind: dim.kind, estimate: est, share, voters: voters_at[&key] });
        }
        winners
    }

    // ───────────────────────────── inspection ─────────────────────────────

    pub fn level_counts(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for n in self.neurons.iter().filter(|n| n.alive) {
            let l = n.level as usize;
            if v.len() <= l { v.resize(l + 1, 0); }
            v[l] += 1;
        }
        v
    }

    pub fn describe_neuron(&self, id: NeuronId) -> String {
        let n = &self.neurons[id as usize];
        let mut s = match n.kind {
            Kind::Inter => format!("#{} L{} inter{} parents={:?}", id, n.level, n.value.as_ref().map(|v| format!(" value={:?}", v)).unwrap_or_default(), n.parents),
            _ => format!("#{} {:?} dim{} b{}", id, n.kind, n.dim.unwrap(), n.bucket),
        };
        s.push_str(&format!(" span={:?} H={} conns={}\n", n.span, n.history.len(), n.connections.len()));
        for p in &n.table {
            s.push_str(&format!("  p{} {:?}\n", p.id, p.body));
        }
        s
    }
}
