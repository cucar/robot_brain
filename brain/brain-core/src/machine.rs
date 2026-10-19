//! The machine (§2, §3, §6, §7, §8): frames, open activations, the waves with their elections, the
//! children with their index, the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::neuron::*;
use crate::offsets::*;
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
    pub waves: usize,
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
}


pub struct Machine {
    pub cfg: Config,
    pub act_dims: Vec<ActDim>,
    pub radii: Vec<i32>,
    pub dims: Vec<NeuronDim>,
    pub channels: Vec<Channel>,
    pub neurons: Vec<Neuron>,
    /// The largest span any activation of each neuron has had, for the owner's reach in prices (D13).
    pub spans: Vec<Pos>,
    pub alive: usize,
    pub base_count: u32,
    pub frames: VecDeque<FrameRec>,
    pub frame_no: Frame,
    pub(crate) pending_weak: Vec<(NeuronId, Pos)>,
    pub max_reach_t: i32,
    /// The interneuron index (D36): what each stands for, to its id.
    pub index: FxHashMap<String, NeuronId>,
    /// Children whose last parent retired, waiting for their windows to close (R38).
    pub ledger: Vec<NeuronId>,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    pub trace: bool,
}

impl Machine {
    pub fn new(h: usize, policy: Policy, time_radius: i32) -> Self {
        Machine {
            cfg: Config { h, policy },
            act_dims: vec![ActDim { name: "time".into(), radius: time_radius.max(1) }],
            radii: vec![time_radius.max(1)],
            dims: Vec::new(),
            channels: Vec::new(),
            neurons: Vec::new(),
            spans: Vec::new(),
            alive: 0,
            base_count: 0,
            frames: VecDeque::new(),
            frame_no: 0,
            pending_weak: Vec::new(),
            max_reach_t: time_radius.max(1),
            index: FxHashMap::default(),
            ledger: Vec::new(),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            trace: std::env::var("UCAR_TRACE").is_ok(),
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
        let first = self.neurons.len() as NeuronId;
        for b in 0..resolution {
            self.neurons.push(Neuron::base(first + b, kind, id, b, channel, axes, self.cfg.h, self.radii.clone()));
            self.spans.push([0; MAX_DIMS]);
        }
        self.alive += resolution as usize;
        self.base_count += resolution;
        self.dims.push(NeuronDim { id, name: name.into(), channel, kind, resolution, first });
        id
    }

    pub fn base_neuron(&self, dim: DimId, bucket: u32) -> NeuronId { self.dims[dim as usize].first + bucket }

    pub(crate) fn base_bits(&self) -> f64 { log2(self.base_count as f64) }

    pub(crate) fn reach_for(&self, axes: u8, span: &Pos) -> Pos {
        let mut r = [0; MAX_DIMS];
        for d in 0..self.radii.len() { if axes & (1 << d) != 0 { r[d] = span[d] + self.radii[d]; } }
        r
    }

    pub(crate) fn ctx_for(&self, n: NeuronId) -> Ctx {
        let nn = &self.neurons[n as usize];
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
            let nn = &self.neurons[n as usize];
            let Some(dim) = nn.dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.new_act(n, pos, base_bits, false, None, [0; MAX_DIMS], 0, 0));
        }
        stats.base_acts = frame.acts.len();
        self.frames.push_back(frame);

        // the waves (§6, R26)
        let mut wave: u32 = 0;
        let mut pending: Vec<Pending> = Vec::new();
        loop {
            // activate children bought by the wave before (§6.4)
            let fired = pending.len();
            for kid in pending.drain(..) {
                let fr = kid.act.pos[0] as i64;
                let Some(f) = self.frame_mut(fr) else { continue };
                let idx = f.acts.len() as u32;
                let r = ActRef { frame: fr, index: idx };
                let mut act = kid.act;
                act.covers = kid.covered.clone();
                f.acts.push(act);
                for c in kid.covered {
                    if let Some(a) = self.act_mut(c) { if a.coverer.is_none() { a.coverer = Some(r); } }
                }
            }
            if wave > 0 && fired == 0 { break; }
            stats.waves += 1;
            // process functions (§5), per neuron, with every activation of this wave
            let (bids, retired, added) = self.call_wave(wave);
            stats.bids += bids.len();
            stats.added += added;
            stats.retired += retired.iter().filter(|(_, r)| !r.member_only).count();
            for (owner, r) in retired { self.handle_retired(owner, r); }
            // elect (R24) and create children (R43)
            let (accepted, minted, reused) = self.elect(wave, bids, &mut pending);
            stats.accepted += accepted;
            stats.minted += minted;
            stats.reused += reused;
            wave += 1;
            if wave > 64 { break; }
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
        stats.patterns = self.neurons.iter().filter(|n| n.alive).map(|n| n.table.len()).sum();
        stats.connections = self.neurons.iter().filter(|n| n.alive).map(|n| n.connections.len()).sum();
        let keep_from = fno - self.max_reach_t as i64 - 1;
        while self.frames.front().map_or(false, |f| f.no < keep_from) { self.frames.pop_front(); }
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }

    pub(crate) fn current(&self) -> &FrameRec { self.frames.back().unwrap() }

    pub(crate) fn new_act(&self, n: NeuronId, pos: Pos, cost: f64, strong: bool, fired_by: Option<FiredBy>, span: Pos, wave: u32, height: u32) -> Act {
        let nn = &self.neurons[n as usize];
        let reach = self.reach_for(nn.axes, &span);
        Act { neuron: n, kind: nn.kind, axes: nn.axes, pos, cost, strong, span, reach, covered_at: None, coverer: None, covers: Vec::new(), fired_by, inferences: Vec::new(), opened_at: self.frame_no, wave, height, caught_up: pos[0] as i64 }
    }

    // ───────────────────────────── the call (R20) ─────────────────────────────

    pub(crate) fn call_wave(&mut self, wave: u32) -> (Vec<Bid>, Vec<(NeuronId, Retired)>, usize) {
        let fno = self.frame_no;
        // the activations of this wave, grouped by neuron, in the order they fired
        let mut by_neuron: FxHashMap<NeuronId, Vec<RowInput>> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        let refs: Vec<ActRef> = self.frames.iter().flat_map(|f| f.acts.iter().enumerate().filter(|(_, a)| a.opened_at == fno && a.wave == wave).map(|(i, _)| ActRef { frame: f.no, index: i as u32 }).collect::<Vec<_>>()).collect();
        for r in refs {
            let nbs = self.neighborhood(r);
            if nbs.is_empty() { continue; }
            let a = self.act(r).unwrap();
            if !by_neuron.contains_key(&a.neuron) { order.push(a.neuron); }
            let owner_value = self.follow(a.neuron);
            by_neuron.entry(a.neuron).or_default().push(RowInput { act: r, pos: a.pos, own_cost: a.cost, owner_value, nbs });
        }
        let mut bids = Vec::new();
        let mut retired = Vec::new();
        let mut added = 0;
        for n in order {
            let rows = by_neuron.remove(&n).unwrap();
            let ctx = self.ctx_for(n);
            let (b, r, a) = self.neurons[n as usize].process(rows, ctx);
            // a line that narrowed: a child it shares with another line now stands for something else to
            // this line, which lets it go and is bought one by its new key (D36, R43); a child of its own
            // keeps its name, which the lines above name, and forgets what it learned under the places given up
            for (pid, c) in std::mem::take(&mut self.neurons[n as usize].narrowed_children) {
                if (c as usize) >= self.neurons.len() { continue; }
                if self.neurons[c as usize].parents.len() > 1 {
                    self.neurons[c as usize].parents.retain(|p| *p != (n, pid));
                    if let Some(p) = self.neurons[n as usize].pattern_mut(pid) { if let Body::Function { child, .. } = &mut p.body { *child = None; } }
                    if self.trace { eprintln!("[narrowed f{}] #{} p{} lets go of shared child #{}", fno, n, pid, c); }
                } else {
                    self.neurons[c as usize].connections.clear();
                    if self.trace { eprintln!("[narrowed f{}] #{} forgets its connections", fno, c); }
                }
            }
            if self.trace && (!b.is_empty() || !r.is_empty() || a > 0) {
                eprintln!("[call f{} w{} #{}] bids {} retired {} added {}", fno, wave, n, b.len(), r.len(), a);
            }
            bids.extend(b);
            retired.extend(r.into_iter().map(|x| (n, x)));
            added += a;
        }
        (bids, retired, added)
    }

    // ───────────────────────────── inspection ─────────────────────────────

    /// How many living neurons stand at each height, read off their last activation.
    pub fn height_counts(&self) -> Vec<usize> {
        let mut heights: FxHashMap<NeuronId, u32> = FxHashMap::default();
        for f in self.frames.iter() { for a in &f.acts { let e = heights.entry(a.neuron).or_default(); if a.height > *e { *e = a.height; } } }
        let mut v = Vec::new();
        for n in self.neurons.iter().filter(|n| n.alive) {
            let h = heights.get(&n.id).copied().unwrap_or(0) as usize;
            if v.len() <= h { v.resize(h + 1, 0); }
            v[h] += 1;
        }
        v
    }

    pub fn describe_neuron(&self, id: NeuronId) -> String {
        let n = &self.neurons[id as usize];
        let mut s = match n.kind {
            Kind::Inter => format!("#{} inter{} parents={:?} key={}", id, n.value.map(|v| format!(" value=({},p{},#{})", v.owner, v.var, v.member)).unwrap_or_default(), n.parents, n.key),
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
