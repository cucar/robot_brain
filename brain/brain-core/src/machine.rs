//! The machine (§2, §3, §6, §7, §8): frames, open activations, the levels with their elections, the
//! children with their index, the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::neuron::*;
use crate::offsets::*;
use crate::thalamus::Thalamus;
use crate::types::*;

pub use crate::activation::*;
pub(crate) use crate::election::Pending;

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
    pub reused: usize,
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
    /// The dispatch: regions, and columns per region, over which the neurons are partitioned by id.
    pub regions: usize,
    pub columns: usize,
}


pub struct Machine {
    pub cfg: Config,
    pub act_dims: Vec<ActDim>,
    pub radii: Vec<i32>,
    pub dims: Vec<NeuronDim>,
    pub channels: Vec<Channel>,
    /// The neurons, owned by the columns of the regions and reached by id.
    pub thalamus: Thalamus,
    /// The largest span any activation of each neuron has had, for the owner's reach in prices (D13).
    pub spans: Vec<Pos>,
    pub alive: usize,
    pub base_count: u32,
    pub frames: VecDeque<FrameRec>,
    pub frame_no: Frame,
    pub(crate) pending_weak: Vec<(NeuronId, Pos)>,
    /// Patterns an inference's expansion completes at the frame ahead, where what it placed was selected:
    /// they fire weakly there (R30).
    pub(crate) pending_weak_patterns: Vec<(NeuronId, Pos)>,
    pub max_reach_t: i32,
    /// The interneuron index (D36): what each stands for, to its id.
    pub index: FxHashMap<String, NeuronId>,
    /// Children whose last parent retired, waiting for their windows to close (R38).
    pub ledger: Vec<NeuronId>,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    pub trace: bool,
    /// Retired lines the ledger holds until every window over what they fired has closed (R38), with
    /// the frame through which they are kept.
    pub(crate) retired_lines: FxHashMap<(NeuronId, PatternId), (Function, Frame)>,
    /// Times a descent met a neuron again: never, where no line names a neuron built over its own child
    /// (D27); counted so that a fault shows.
    pub cycle_guard: std::cell::Cell<u64>,
}

impl Machine {
    /// One region, a column per available core.
    pub fn new(h: usize, policy: Policy, time_radius: i32) -> Self {
        let columns = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::with_layout(h, policy, time_radius, 1, columns)
    }

    /// With the dispatch laid out: `regions` regions of `columns` columns each.
    pub fn with_layout(h: usize, policy: Policy, time_radius: i32, regions: usize, columns: usize) -> Self {
        Machine {
            cfg: Config { h, policy, regions: regions.max(1), columns: columns.max(1) },
            act_dims: vec![ActDim { name: "time".into(), radius: time_radius.max(1) }],
            radii: vec![time_radius.max(1)],
            dims: Vec::new(),
            channels: Vec::new(),
            thalamus: Thalamus::new(regions, columns),
            spans: Vec::new(),
            alive: 0,
            base_count: 0,
            frames: VecDeque::new(),
            frame_no: 0,
            pending_weak: Vec::new(),
            pending_weak_patterns: Vec::new(),
            max_reach_t: time_radius.max(1),
            index: FxHashMap::default(),
            ledger: Vec::new(),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            trace: std::env::var("UCAR_TRACE").is_ok(),
            retired_lines: FxHashMap::default(),
            cycle_guard: std::cell::Cell::new(0),
        }
    }

    /// Declare a spatial activation dimension (D1). Returns its index (1 or 2).
    pub fn declare_axis(&mut self, name: &str, radius: i32) -> usize {
        assert!(self.act_dims.len() < MAX_DIMS, "at most two spatial axes");
        self.act_dims.push(ActDim { name: name.into(), radius: radius.max(1) });
        self.radii.push(radius.max(1));
        self.act_dims.len() - 1
    }

    /// Declare a channel with an optional event dimension and action dimension (D3). Base neurons are
    /// minted in declaration order (D52).
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

    pub(crate) fn declare_dim(&mut self, name: &str, channel: ChannelId, kind: Kind, resolution: u32, axes: u8) -> DimId {
        let id = self.dims.len() as DimId;
        let first = self.thalamus.len() as NeuronId;
        for b in 0..resolution {
            self.thalamus.push(Neuron::base(first + b, kind, id, b, channel, axes, self.cfg.h, self.radii.clone()));
            self.spans.push([0; MAX_DIMS]);
        }
        self.alive += resolution as usize;
        self.base_count += resolution;
        self.dims.push(NeuronDim { id, name: name.into(), channel, kind, resolution, first });
        id
    }

    pub fn base_neuron(&self, dim: DimId, bucket: u32) -> NeuronId { self.dims[dim as usize].first + bucket }

    /// A function as the machine reads it for expansion and the index: the owner's table, or the ledger's
    /// copy of a retired function while what it fired is still open (R38).
    pub(crate) fn line(&self, owner: NeuronId, pid: PatternId) -> Option<&Function> {
        self.thalamus.get(owner).and_then(|n| n.function(pid)).or_else(|| self.retired_lines.get(&(owner, pid)).map(|x| &x.0))
    }

    /// What a base activation costs (D13): a neuron, `log₂ n` of the neurons the machine holds now.
    pub(crate) fn base_bits(&self) -> f64 { log2(self.alive as f64) }

    pub(crate) fn reach_for(&self, axes: u8, span: &Pos) -> Pos {
        let mut r = [0; MAX_DIMS];
        for d in 0..self.radii.len() { if axes & (1 << d) != 0 { r[d] = span[d] + self.radii[d]; } }
        r
    }

    pub(crate) fn ctx_for(&self, n: NeuronId) -> Ctx {
        let nn = &self.thalamus[n];
        let reach = self.reach_for(nn.axes, &self.spans[n as usize]);
        Ctx { neuron_bits: log2(self.alive as f64), offset_bits: offset_bits(&reach, &self.radii, nn.axes), frame: self.frame_no, trace: self.trace }
    }

    pub fn act(&self, r: ActRef) -> Option<&Act> {
        self.frames.iter().find(|f| f.no == r.frame).and_then(|f| f.acts.get(r.index as usize))
    }

    pub(crate) fn act_mut(&mut self, r: ActRef) -> Option<&mut Act> {
        self.frames.iter_mut().find(|f| f.no == r.frame).and_then(|f| f.acts.get_mut(r.index as usize))
    }

    pub(crate) fn frame_mut(&mut self, no: Frame) -> Option<&mut FrameRec> { self.frames.iter_mut().find(|f| f.no == no) }

    /// Clear every open activation and standing inference; keep the tables and connections.
    pub fn reset_context(&mut self) {
        self.frames.clear();
        self.pending_weak.clear();
        self.pending_weak_patterns.clear();
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
            frame.acts.push(self.new_act(n, pos, base_bits, true, None, [0; MAX_DIMS], 0, 0));
            if dim.kind == Kind::Action { frame.actions_ran.push((dim.channel, ActRef { frame: fno, index: idx })); }
        }
        let weak = std::mem::take(&mut self.pending_weak);
        for (n, p) in weak {
            let nn = &self.thalamus[n];
            let Some(dim) = nn.dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.new_act(n, pos, base_bits, false, None, [0; MAX_DIMS], 0, 0));
        }
        // the patterns the expansion completed here weakly, as inputs of the frame like the base (R30, D40)
        let weak_patterns = std::mem::take(&mut self.pending_weak_patterns);
        for (n, p) in weak_patterns {
            let Some(nn) = self.thalamus.get(n) else { continue };
            if !nn.alive || nn.kind != Kind::Inter { continue; }
            let mut pos = p;
            pos[0] = fno as i32;
            let wp = nn.parents.first().map(|(o, _)| self.thalamus[*o].which_function_bits()).unwrap_or(0.0);
            let span = self.spans[n as usize];
            frame.acts.push(self.new_act(n, pos, base_bits + wp, false, None, span, 0, 0));
        }
        stats.base_acts = frame.acts.len();
        self.frames.push_back(frame);

        // the levels (§6, R26)
        let mut level: u32 = 0;
        let mut pending: Vec<Pending> = Vec::new();
        loop {
            // activate children bought at the level before (§6.4)
            let fired = pending.len();
            for kid in pending.drain(..) {
                let fr = kid.act.pos[0] as i64;
                let Some(f) = self.frame_mut(fr) else { continue };
                let idx = f.acts.len() as u32;
                let r = ActRef { frame: fr, index: idx };
                let mut act = kid.act;
                act.covers = kid.covered.clone();
                // where the report built the strong activation of a pattern an inference placed weakly, the
                // weak one is dropped (D40, R30)
                if let Some(w) = f.acts.iter_mut().find(|w| !w.strong && !w.gone && w.neuron == act.neuron && w.pos == act.pos && w.opened_at == fno) {
                    if act.strong { w.gone = true; }
                }
                f.acts.push(act);
                for c in kid.covered {
                    if let Some(a) = self.act_mut(c) { if a.coverer.is_none() { a.coverer = Some(r); } }
                }
            }
            if level > 0 && fired == 0 { break; }
            stats.levels += 1;
            // process functions (§5), per neuron, with every activation of this level
            let (bids, retired, added) = self.call_level(level);
            stats.bids += bids.len();
            stats.added += added;
            stats.retired += retired.len();
            for (owner, r) in retired { self.handle_retired(owner, r); }
            // elect (R24) and create children (R43)
            let (accepted, minted, reused) = self.elect(level, bids, &mut pending);
            stats.accepted += accepted;
            stats.minted += minted;
            stats.reused += reused;
            level += 1;
            // nothing caps the depth (R26); a frame this deep means a fault, and is counted as one
            if level > 1024 { self.cycle_guard.set(self.cycle_guard.get() + 1); break; }
        }
        // wire (R16) and the index (D36)
        self.refresh_index();

        // connect and infer (§7)
        self.connect_all();
        self.infer_all();
        // delete children due (R38)
        self.reap();

        // selection (§8)
        let winners = self.select();
        stats.voters = self.last_stats.voters;
        stats.voters_above_base = self.last_stats.voters_above_base;
        self.pending_weak = winners.iter().map(|w| (self.base_neuron(w.dim, w.bucket), w.pos)).collect();

        // stats and pruning: the settled frame is the oldest one no later bid can still cover
        let settled = fno - self.max_reach_t as i64;
        if let Some(sf) = self.frames.iter().find(|f| f.no == settled) {
            stats.apex_acts = sf.acts.iter().filter(|a| a.covered_at.is_none()).count();
            stats.apex_cost = sf.acts.iter().filter(|a| a.covered_at.is_none()).map(|a| a.cost).sum();
        }
        stats.neurons = self.alive;
        stats.patterns = self.thalamus.iter().filter(|n| n.alive).map(|n| n.functions.len()).sum();
        stats.connections = self.thalamus.iter().filter(|n| n.alive).map(|n| n.connections.len()).sum();
        let keep_from = fno - self.max_reach_t as i64 - 1;
        while self.frames.front().map_or(false, |f| f.no < keep_from) { self.frames.pop_front(); }
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }

    pub(crate) fn current(&self) -> &FrameRec { self.frames.back().unwrap() }

    pub(crate) fn new_act(&self, n: NeuronId, pos: Pos, cost: f64, strong: bool, fired_by: Option<FiredBy>, span: Pos, level: u32, height: u32) -> Act {
        let nn = &self.thalamus[n];
        let reach = self.reach_for(nn.axes, &span);
        Act { neuron: n, kind: nn.kind, axes: nn.axes, pos, cost, strong, span, reach, covered_at: None, bound_at: None, coverer: None, covers: Vec::new(), fired_by, inferences: Vec::new(), opened_at: self.frame_no, level, height, gone: false, caught_up: pos[0] as i64 }
    }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    pub(crate) fn call_level(&mut self, level: u32) -> (Vec<Bid>, Vec<(NeuronId, Retired)>, usize) {
        let fno = self.frame_no;
        // route: the activations of this level, grouped by neuron, in the order they fired
        let mut by_neuron: FxHashMap<NeuronId, Vec<RowInput>> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        let refs: Vec<ActRef> = self.frames.iter().flat_map(|f| f.acts.iter().enumerate().filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone).map(|(i, _)| ActRef { frame: f.no, index: i as u32 }).collect::<Vec<_>>()).collect();
        for r in refs {
            let nbs = self.neighborhood(r);
            if nbs.is_empty() { continue; }
            let a = self.act(r).unwrap();
            if !by_neuron.contains_key(&a.neuron) { order.push(a.neuron); }
            by_neuron.entry(a.neuron).or_default().push(RowInput { act: r, pos: a.pos, own_cost: a.cost, nbs });
        }
        let tasks: Vec<(NeuronId, Vec<RowInput>, Ctx)> = order.iter().map(|&n| (n, by_neuron.remove(&n).unwrap(), self.ctx_for(n))).collect();
        // what every interneuron is built over as the index stands at this level, for D27 (D36)
        let built: FxHashMap<NeuronId, Vec<NeuronId>> = if tasks.is_empty() { FxHashMap::default() } else {
            self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter).map(|n| (n.id, self.through_of(n.id))).collect()
        };
        // dispatch: every neuron's call runs in its column, the columns in parallel; collect in the order above
        let results = self.thalamus.process_level(tasks, std::sync::Arc::new(built));
        let mut bids = Vec::new();
        let mut retired = Vec::new();
        let mut added = 0;
        for w in results {
            let n = w.neuron;
            if self.trace && (!w.bids.is_empty() || !w.retired.is_empty() || w.added > 0) {
                eprintln!("[call f{} l{} #{}] bids {} retired {} added {}", fno, level, n, w.bids.len(), w.retired.len(), w.added);
            }
            bids.extend(w.bids);
            retired.extend(w.retired.into_iter().map(|x| (n, x)));
            added += w.added;
        }
        (bids, retired, added)
    }

    // ───────────────────────────── inspection ─────────────────────────────

    /// How many living neurons stand at each height, read off their last activation.
    pub fn height_counts(&self) -> Vec<usize> {
        let mut heights: FxHashMap<NeuronId, u32> = FxHashMap::default();
        for f in self.frames.iter() { for a in &f.acts { let e = heights.entry(a.neuron).or_default(); if a.height > *e { *e = a.height; } } }
        let mut v = Vec::new();
        for n in self.thalamus.iter().filter(|n| n.alive) {
            let h = heights.get(&n.id).copied().unwrap_or(0) as usize;
            if v.len() <= h { v.resize(h + 1, 0); }
            v[h] += 1;
        }
        v
    }

    pub fn describe_neuron(&self, id: NeuronId) -> String {
        let n = &self.thalamus[id];
        let mut s = match n.kind {
            Kind::Inter => format!("#{} inter parents={:?} key={}", id, n.parents, n.key),
            _ => format!("#{} {:?} dim{} b{}", id, n.kind, n.dim.unwrap(), n.bucket),
        };
        s.push_str(&format!(" span={:?} H={} conns={}\n", self.spans[id as usize], n.history.len(), n.connections.len()));
        s.push_str(&n.describe());
        let mut conns: Vec<(&(NeuronId, Offset), &Conn)> = n.connections.iter().collect();
        conns.sort_by(|a, b| a.0.cmp(b.0));
        for ((t, o), c) in conns.into_iter().take(16) {
            s.push_str(&format!("    -> #{} @{:?} s={:.0} est={:+.2}\n", t, o, c.strength, c.estimate()));
        }
        s
    }
}
