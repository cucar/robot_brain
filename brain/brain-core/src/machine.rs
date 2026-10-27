//! The machine (§2, §3, §6, §7, §8): frames, open activations, the dictionary, the levels with recognition
//! and the election, the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::election::View;
use crate::function::*;
use crate::neuron::*;
use crate::thalamus::Thalamus;
use crate::types::*;

pub use crate::activation::*;

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
    /// The dictionary (D36): every function, its interneuron, and the classes.
    pub dict: Dictionary,
    /// Functions that left the dictionary, kept readable while what they fired is open (R38).
    pub(crate) retired_funcs: FxHashMap<FuncId, (Func, Frame)>,
    /// The assignment of grouping (R50): per neuron, per row of its history, what the dictionary's cover left
    /// of it when it was last priced, the activation and each neighbor. Read by grouping, refreshed by pruning.
    pub(crate) assigned: std::cell::RefCell<FxHashMap<NeuronId, FxHashMap<u64, (f64, Vec<f64>)>>>,
    /// Per interneuron, the function that fired it last, for inferring it (R28).
    pub(crate) last_func: FxHashMap<NeuronId, FuncId>,
    /// Interneurons whose last function has left, waiting for their windows to close (R38).
    pub ledger: Vec<NeuronId>,
    pub(crate) stats_added: usize,
    pub(crate) stats_retired: usize,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    pub trace: bool,
    /// A test: the machine recognizes and predicts but changes nothing of itself. No rows enter a history, no
    /// function is added, generalized or pruned, no interneuron is allocated or reused, no connection or reward
    /// is written, nothing is decommissioned.
    pub frozen: bool,
    /// Milliseconds spent per step since the machine began: neighborhoods and activate, group, find and
    /// generalize, elect and allocate, prune, connect and infer, decommission, select; and within select, the
    /// match, the voters' placements, and the patterns execution fires.
    pub timings: [f64; 11],
    /// Times a frame ran past a thousand levels: a fault, counted.
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
            dict: Dictionary::default(),
            retired_funcs: FxHashMap::default(),
            assigned: std::cell::RefCell::new(FxHashMap::default()),
            last_func: FxHashMap::default(),
            ledger: Vec::new(),
            stats_added: 0,
            stats_retired: 0,
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            trace: false,
            frozen: false,
            timings: [0.0; 11],
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

    pub(crate) fn reach_for(&self, axes: u8, span: &Pos) -> Pos {
        let mut r = [0; MAX_DIMS];
        for d in 0..self.radii.len() { if axes & (1 << d) != 0 { r[d] = span[d] + self.radii[d]; } }
        r
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
            frame.acts.push(self.new_act(n, pos, true, [0; MAX_DIMS]));
            if dim.kind == Kind::Action { frame.actions_ran.push((dim.channel, ActRef { frame: fno, index: idx })); }
        }
        let weak = std::mem::take(&mut self.pending_weak);
        for (n, p) in weak {
            let nn = &self.thalamus[n];
            let Some(dim) = nn.dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.new_act(n, pos, false, [0; MAX_DIMS]));
        }
        // the patterns the expansion completed here weakly, as inputs of the frame like the base (R30, D40)
        let weak_patterns = std::mem::take(&mut self.pending_weak_patterns);
        for (n, p) in weak_patterns {
            let Some(nn) = self.thalamus.get(n) else { continue };
            if !nn.alive || nn.kind != Kind::Inter { continue; }
            let mut pos = p;
            pos[0] = fno as i32;
            let span = self.spans[n as usize];
            frame.acts.push(self.new_act(n, pos, false, span));
        }
        stats.base_acts = frame.acts.len();
        self.frames.push_back(frame);

        // the levels (§6, R26): activate, recognize, elect, allocate, prune
        self.stats_added = 0;
        self.stats_retired = 0;
        let mut level: u32 = 0;
        loop {
            let refs: Vec<ActRef> = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate().filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone && a.neuron != NeuronId::MAX).map(|(i, _)| ActRef { frame: fno, index: i as u32 }).collect::<Vec<_>>()).collect();
            if refs.is_empty() { break; }
            stats.levels += 1;
            let (bids, accepted, minted, reused) = self.run_level(level, &refs);
            stats.bids += bids;
            stats.accepted += accepted;
            stats.minted += minted;
            stats.reused += reused;
            level += 1;
            // nothing caps the depth (R26); a frame this deep means a fault, and is counted as one
            if level > 1024 { self.cycle_guard.set(self.cycle_guard.get() + 1); break; }
        }
        stats.added = self.stats_added;
        stats.retired = self.stats_retired;

        // connect and infer (§7)
        let t = std::time::Instant::now();
        self.connect_all();
        self.infer_all();
        self.timings[5] += t.elapsed().as_secs_f64() * 1000.0;
        // decommission what is due (R38)
        let t = std::time::Instant::now();
        if !self.frozen { self.reap(); }
        self.timings[6] += t.elapsed().as_secs_f64() * 1000.0;

        // selection (§8)
        let t = std::time::Instant::now();
        let winners = self.select();
        self.timings[7] += t.elapsed().as_secs_f64() * 1000.0;
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
        stats.patterns = self.dict.len();
        stats.connections = self.thalamus.iter().filter(|n| n.alive).map(|n| n.connections.len()).sum();
        // the window the machine keeps: the longest reach in time among the activations still open (D24)
        let mut longest = self.radii[0];
        for f in self.frames.iter() { for a in &f.acts { if !a.gone && f.no + a.reach[0] as i64 >= fno { longest = longest.max(a.reach[0]); } } }
        self.max_reach_t = longest;
        let keep_from = fno - self.max_reach_t as i64 - 1;
        while self.frames.front().map_or(false, |f| f.no < keep_from) { self.frames.pop_front(); }
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }


    /// A bare activation of `n` at `pos` (D8), its record its cost (D12, D13).
    pub(crate) fn new_act(&self, n: NeuronId, pos: Pos, strong: bool, span: Pos) -> Act {
        let nn = &self.thalamus[n];
        let reach = self.reach_for(nn.axes, &span);
        let cost = record(nn.axes);
        Act { neuron: n, kind: nn.kind, axes: nn.axes, pos, cost, full: cost, strong, span, reach, covered_at: None, bound_at: None, coverer: None, retaken: Vec::new(), covers: Vec::new(), binds: Vec::new(), fired_by: None, inferences: Vec::new(), opened_at: self.frame_no, level: 0, height: 0, gone: false, caught_up: pos[0] as i64 }
    }

    // ───────────────────────────── one level (§6, R26) ─────────────────────────────

    /// One level: activate every neuron with an activation here (R20), recognize each activation in turn
    /// (group, find, generalize), elect over everything found (R24), allocate, and prune (R18).
    fn run_level(&mut self, level: u32, refs: &[ActRef]) -> (usize, usize, usize, usize) {
        // activate: each activation with its neighborhood, every neuron's call in its column (R20)
        let t = std::time::Instant::now();
        let mut views: Vec<View> = Vec::new();
        let mut by_neuron: FxHashMap<NeuronId, Vec<RowInput>> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        for &r in refs {
            let (nbs, calls) = self.neighborhood(r);
            let a = self.act(r).unwrap();
            if !by_neuron.contains_key(&a.neuron) { order.push(a.neuron); }
            by_neuron.entry(a.neuron).or_default().push(RowInput { act: r, pos: a.pos, own_cost: a.cost, nbs: nbs.clone(), calls: calls.clone() });
            views.push(View { act: r, nbs, calls });
        }
        let tasks: Vec<(NeuronId, Vec<RowInput>)> = order.iter().map(|&n| (n, by_neuron.remove(&n).unwrap())).collect();
        let results = if self.frozen { Vec::new() } else { self.thalamus.process_level(tasks) };
        self.timings[0] += t.elapsed().as_secs_f64() * 1000.0;
        // recognize, one activation at a time: group, find, generalize (R50, R53, R52)
        let mut anchors: FxHashSet<NeuronId> = FxHashSet::default();
        for res in &results {
            anchors.insert(res.neuron);
            for ret in &res.returns {
                let t = std::time::Instant::now();
                let kept = self.group(res.neuron, ret);
                self.timings[1] += t.elapsed().as_secs_f64() * 1000.0;
                let t = std::time::Instant::now();
                self.stats_added += self.find(res.neuron, kept);
                self.timings[2] += t.elapsed().as_secs_f64() * 1000.0;
            }
        }
        // elect and allocate (R24, R16, R43)
        let t = std::time::Instant::now();
        let (accepted, minted, reused) = self.elect(level, &views);
        self.timings[3] += t.elapsed().as_secs_f64() * 1000.0;
        // prune (R18)
        let t = std::time::Instant::now();
        if !self.frozen { self.prune(&anchors); }
        self.timings[4] += t.elapsed().as_secs_f64() * 1000.0;
        (results.iter().map(|r| r.returns.len()).sum(), accepted, minted, reused)
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
            Kind::Inter => format!("#{} inter", id),
            _ => format!("#{} {:?} dim{} b{}", id, n.kind, n.dim.unwrap(), n.bucket),
        };
        s.push_str(&format!(" span={:?} H={} conns={}\n", self.spans[id as usize], n.history.len(), n.connections.len()));
        for fid in self.dict.of_inter(id) { if let Some(f) = self.dict.get(*fid) { s.push_str(&format!("  is f{} {}\n", f.id, self.dict.describe(f))); } }
        for fid in self.dict.standing(id) { if let Some(f) = self.dict.get(*fid) { s.push_str(&format!("  f{} {}\n", f.id, self.dict.describe(f))); } }
        let mut conns: Vec<(&(NeuronId, Offset), &Conn)> = n.connections.iter().collect();
        conns.sort_by(|a, b| a.0.cmp(b.0));
        for ((t, o), c) in conns.into_iter().take(16) {
            s.push_str(&format!("    -> #{} @{:?} s={:.0} est={:+.2}\n", t, o, c.strength, c.estimate()));
        }
        s
    }
}
