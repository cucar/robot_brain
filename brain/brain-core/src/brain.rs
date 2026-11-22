/// The machine (§2, §3, §6, §7, §8): frames, open activations, the dictionary, the levels with recognition
/// and the election, the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::dictionary::*;
use crate::neuron::*;
use crate::recognize::{edge, Fresh};
use crate::thalamus::{NeuronFacts, Thalamus};
use crate::types::*;

pub use crate::board::*;

#[derive(Clone, Debug)]
pub struct Winner {
    pub dim: DimensionId,
    pub bucket: u32,
    pub pos: Pos,
    pub kind: Kind,
    pub estimate: f64,
    pub share: f64,
    pub voters: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub frame: FrameNumber,
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
    /// Per level: the activations it ran, the relations they returned, the fits found, accepted, and functions added.
    pub per_level: Vec<(usize, usize, usize, usize, usize, f64)>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub h: usize,
    pub policy: Policy,
    /// The dispatch: regions, and columns per region, over which the neurons are partitioned by id.
    pub regions: usize,
    pub columns: usize,
}

pub struct Brain {
    pub cfg: Config,
    pub act_dims: Vec<ActDim>,
    pub dims: Vec<NeuronDim>,
    pub channels: Vec<Channel>,
    /// The neurons, owned by the columns of the regions and reached by id.
    pub(crate) thalamus: Thalamus,
    pub alive: usize,
    pub base_count: u32,
    pub frame_no: FrameNumber,
    pub(crate) pending_weak: Vec<(NeuronId, Pos)>,
    pub max_reach_t: i32,
    /// S (D9): how long an activation stands on the board, the longest reach in time among the neurons the machine
    /// holds, one beyond the farthest node any function has from its origin. It moves with the dictionary.
    pub span_s: i32,
    /// Functions that left the dictionary, kept readable while what they fired is open (R38).
    pub(crate) retired_funcs: FxHashMap<FuncId, (Func, FrameNumber)>,
    /// Per interneuron, the function that fired it last, for inferring it (R28).
    pub(crate) last_func: FxHashMap<NeuronId, FuncId>,
    /// The same while frozen: what a frozen machine's calls would have written there, read before `last_func` and
    /// dropped once the machine learns again, so a test leaves `last_func` as it found it
    pub(crate) frozen_last_func: FxHashMap<NeuronId, FuncId>,
    /// Interneurons whose last function has left, waiting for their windows to close (R38).
    pub ledger: Vec<NeuronId>,
    /// Per interneuron on the ledger, the frame before which its subtree still stands on the board: it is not
    /// looked at again until then (R38).
    pub(crate) ledger_due: FxHashMap<NeuronId, FrameNumber>,
    pub(crate) stats_added: usize,
    pub(crate) stats_reused: usize,
    pub(crate) last_fits: usize,
    /// Per interneuron, the farthest it has seen in each dimension, over its activations (D4).
    pub(crate) reach_of: FxHashMap<NeuronId, Pos>,
    pub(crate) stats_retired: usize,
    /// The relations covered by the calls taken so far at one level of one frame, across every election of the
    /// level, each with the call and its function, as adjacency, and the sets they join (D24); the frame and level
    /// they are for
    pub(crate) joined: FxHashMap<ActRef, Vec<(ActRef, ActRef, FuncId)>>,
    pub(crate) jparent: FxHashMap<ActRef, ActRef>,
    pub(crate) joined_at: Option<(FrameNumber, u32)>,
    /// The relations refused because they would close a deadlock (D24, D55): refused for as long as their two
    /// activations exist
    pub(crate) refused: FxHashSet<(ActRef, ActRef)>,
    /// For each edge a relation became at a call, the relations it stands for; and the edges that stood on the board
    /// before any relation became them (D26, D24)
    pub(crate) form_src: FxHashMap<(ActRef, ActRef), Vec<(ActRef, ActRef)>>,
    pub(crate) form_pre: FxHashSet<(ActRef, ActRef)>,
    /// The relations that may start a function in this half of the level, in the order grouping read them (R50)
    pub(crate) start_queue: Vec<crate::dictionary::Rel>,
    /// The kind of relation this grouping reads, true inside the frame (R24, D34); None reads both
    pub(crate) half: Option<bool>,
    /// How many functions the last start step started
    pub(crate) last_start_rounds: usize,
    /// What each level of this frame has joined, kept while the frame visits another level (D24, R24, R26)
    pub(crate) joined_stash: FxHashMap<u32, (FxHashMap<ActRef, Vec<(ActRef, ActRef, FuncId)>>, FxHashMap<ActRef, ActRef>)>,
    /// Relations of covered activations that no call covered, lifted to their calls (D24, D10, D55)
    pub(crate) lifted: FxHashSet<(ActRef, ActRef)>,
    /// The covered relations at each activation: the other end of each, kept as each fit is accepted (D24, D28)
    pub(crate) cov_inc: FxHashMap<ActRef, Vec<ActRef>>,
    /// Each relation that was lifted, and the relations it was lifted to (D55, D24): covered by a call in any of those,
    /// it is covered, and is not lifted again
    pub(crate) lift_forms: FxHashMap<(ActRef, ActRef), Vec<(ActRef, ActRef)>>,
    /// The kind each activation's first covered relation decided (D24): true inside the frame, false in time
    pub(crate) kind_of: FxHashMap<ActRef, bool>,
    /// What each call took that way: the activations or calls at the other end of the relations lifted to it (D10, D26)
    pub(crate) taken: FxHashMap<ActRef, Vec<ActRef>>,
    /// Relations lifted inside the frame to a call whose other end's row is not written yet: per that activation, the
    /// call and its offset from it, held by the row when it is written; dropped at the frame's end (D10)
    pub(crate) lift_pending: FxHashMap<ActRef, Vec<(ActRef, Offset)>>,
    /// Lifts collected since the last flush, each with the pending entry it leaves where its row is not written yet
    pub(crate) lifts_due: Vec<(NeuronId, LiftTask, Option<(ActRef, ActRef, Offset)>)>,
    /// Whether a grouping is running: counts are asked only then (R50)
    pub(crate) counting: bool,
    /// The counts this grouping has had answered, and those it asked that wait for the next batch (R50)
    pub(crate) counts: std::cell::RefCell<FxHashMap<CountQuery, (u32, u32)>>,
    pub(crate) missed: std::cell::RefCell<Vec<(NeuronId, CountQuery)>>,
    /// The first frame of the run (D24), and every activation covered by one relation: the frame, and the activation
    pub first_frame: Option<FrameNumber>,
    /// A call that took a lap (D53): the record it grew from, to the record it stands as now.
    pub grown: FxHashMap<ActRef, ActRef>,
    /// Whether two activations stand over a shared one, remembered for one grouping step while nothing on the board
    /// changes (a speed-up only).
    /// Each function's reach in time with the nodes and relations it was read from, for S (D9)
    pub(crate) reach_memo: std::cell::RefCell<FxHashMap<FuncId, (Vec<Sym>, Vec<Rel>, i32)>>,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    /// The coverage set per relation (D24): every board relation an accepted call covers, over the frames on the
    /// board; nothing lands on one again (D28).
    pub(crate) covered_rel: FxHashSet<(ActRef, ActRef)>,
    /// Every call that fired, and what it covered, each at its offset from the call (R28): a call reads as
    /// this whether or not its interneuron still has a function
    pub(crate) covered_by: FxHashMap<ActRef, Vec<(Sym, Offset)>>,
    /// The board relations each call's fit covered
    pub(crate) call_edges: FxHashMap<ActRef, Vec<(ActRef, ActRef)>>,
    /// Per accepted call, the calls its fit named or read a node through
    pub(crate) call_thru: FxHashMap<ActRef, Vec<ActRef>>,
    /// This frame's accepted fits, by their calls, so grouping reads what they landed on
    pub(crate) call_fit: FxHashMap<ActRef, crate::recognize::Fit>,
    /// The calls the election running now has taken (D28): an activation they landed on may be landed on again
    /// in this election, each call covering relations of its own (D8, D10).
    pub(crate) electing: FxHashSet<ActRef>,
    /// Per call in its phase, every node its expansion puts back, at any depth, each at its offset from the call's
    /// position (R28, R50); read from the dictionary and dropped whenever the dictionary changes.
    pub(crate) expansions: std::cell::RefCell<FxHashMap<Sym, std::rc::Rc<Vec<(Sym, Offset, Vec<NeuronId>)>>>>,
    /// Per expansion kept, the functions it was read through: a change to one drops the expansions that used it.
    pub(crate) expansion_deps: std::cell::RefCell<FxHashMap<Sym, Vec<crate::dictionary::FuncId>>>,
    /// A test: the machine runs each level as it was trained to, the frame first and time after it, its elections
    /// covering, lifting, refusing and blocking as each fit is taken, its calls firing level after level, and infers
    /// and resolves as usual; but it changes nothing of itself. No rows enter a history and the relation index stays
    /// as it is, no function starts, joins, moves or leaves, no interneuron is allocated (so a fit of a function with
    /// no living interneuron is not elected), no connection or reward is written, nothing is decommissioned.
    pub(crate) frozen: bool,
}

impl Brain {
    /// Create a brain.
    ///
    /// * `h` — the depth of every neuron's history (R4)
    /// * `policy` — the neighborhood policy (D50) and the action policy (D51)
    /// * `regions` — R, the number of regions
    /// * `columns` — C, the number of columns per region
    /// * `learning` — fixed for the life of the instance: a brain built with learning off runs each frame as trained
    ///   and changes nothing of itself; it is loaded from a backup to be tested
    pub fn new(h: usize, policy: Policy, regions: usize, columns: usize, learning: bool) -> Self {
        Brain {
            cfg: Config { h, policy, regions: regions.max(1), columns: columns.max(1) },
            act_dims: vec![ActDim { name: "time".into(), radius: 1 }],
            dims: Vec::new(),
            channels: Vec::new(),
            thalamus: Thalamus::new(regions, columns),
            alive: 0,
            base_count: 0,
            frame_no: 0,
            pending_weak: Vec::new(),
            max_reach_t: 1,
            span_s: 1,
            retired_funcs: FxHashMap::default(),
            last_func: FxHashMap::default(),
            frozen_last_func: FxHashMap::default(),
            ledger: Vec::new(),
            ledger_due: FxHashMap::default(),
            stats_added: 0,
            stats_reused: 0,
            last_fits: 0,
            reach_of: FxHashMap::default(),
            stats_retired: 0,
            joined: FxHashMap::default(), jparent: FxHashMap::default(), joined_at: None, refused: FxHashSet::default(), form_src: FxHashMap::default(), form_pre: FxHashSet::default(),
            start_queue: Vec::new(), half: None, last_start_rounds: 0, joined_stash: FxHashMap::default(),
            grown: FxHashMap::default(),
            lifted: FxHashSet::default(),
            cov_inc: FxHashMap::default(),
            lift_forms: FxHashMap::default(),
            kind_of: FxHashMap::default(),
            taken: FxHashMap::default(),
            lift_pending: FxHashMap::default(),
            lifts_due: Vec::new(),
            counting: false,
            counts: std::cell::RefCell::new(FxHashMap::default()),
            missed: std::cell::RefCell::new(Vec::new()),
            first_frame: None,
            reach_memo: std::cell::RefCell::new(FxHashMap::default()),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            covered_rel: FxHashSet::default(),
            covered_by: FxHashMap::default(),
            call_edges: FxHashMap::default(),
            call_thru: FxHashMap::default(),
            call_fit: FxHashMap::default(),
            electing: FxHashSet::default(),
            expansions: std::cell::RefCell::new(FxHashMap::default()),
            expansion_deps: std::cell::RefCell::new(FxHashMap::default()),
            frozen: !learning,
            
        }
    }

    /// Declare a spatial activation dimension (D1). Returns its index (1 or 2). The radius is read by nothing (R4).
    pub fn declare_axis(&mut self, name: &str, _radius: i32) -> usize {
        assert!(self.act_dims.len() < MAX_DIMS, "at most two spatial axes");
        self.act_dims.push(ActDim { name: name.into(), radius: 1 });
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

    pub(crate) fn declare_dim(&mut self, name: &str, channel: ChannelId, kind: Kind, resolution: u32, axes: u8) -> DimensionId {
        let id = self.dims.len() as DimensionId;
        let first = self.thalamus.registry.len() as NeuronId;
        for b in 0..resolution {
            let facts = NeuronFacts { kind, dim: Some(id), bucket: b, channel: Some(channel), axes, alive: true };
            self.thalamus.allot_neuron(facts, self.cfg.h);
        }
        self.alive += resolution as usize;
        self.base_count += resolution;
        self.dims.push(NeuronDim { id, name: name.into(), channel, kind, resolution, first });
        id
    }

    pub fn find_base_neuron(&self, dim: DimensionId, bucket: u32) -> NeuronId { self.dims[dim as usize].first + bucket }

    pub fn find_act(&self, r: ActRef) -> Option<&Act> {
        self.thalamus.frames.iter().find(|f| f.no == r.frame).and_then(|f| f.acts.get(r.index as usize))
    }

    pub(crate) fn find_act_mut(&mut self, r: ActRef) -> Option<&mut Act> {
        self.thalamus.frames.iter_mut().find(|f| f.no == r.frame).and_then(|f| f.acts.get_mut(r.index as usize))
    }

    pub(crate) fn find_frame_mut(&mut self, no: FrameNumber) -> Option<&mut FrameRec> { self.thalamus.frames.iter_mut().find(|f| f.no == no) }

    /// Clear every open activation and standing inference; keep the dictionary and connections.
    pub fn reset_context(&mut self) {
        self.thalamus.frames.clear();
        self.pending_weak.clear();
    }

    // ───────────────────────────── one frame ─────────────────────────────

    /// Process one frame: what was observed and what ran, with any rewards (§2.3). Returns the
    /// machine's output for the frame ahead.
    pub fn process_frame(&mut self, inputs: &[Input], rewards: Vec<Reward>) -> Vec<Winner> {
        let start = std::time::Instant::now();
        // what a frozen machine named or recorded for its own calls goes once it learns again (`last_func`, poses);
        // the expansions kept may have been read through those poses
        self.thalamus.dict.frozen = self.frozen;
        if !self.frozen {
            self.frozen_last_func.clear();
            if self.thalamus.dict.drop_frozen_syms() { self.forget_all_expansions(); }
        }
        self.frame_no += 1;
        let fno = self.frame_no;
        let mut stats = FrameStats { frame: fno, ..Default::default() };

        // the frame's base activations: strong where the world reported, weak where only an inference named them (D40)
        let mut frame = FrameRec { no: fno, rewards, ..Default::default() };
        let mut reported: FxHashSet<(DimensionId, Pos)> = FxHashSet::default();
        for inp in inputs {
            let dim = &self.dims[inp.dim as usize];
            if inp.bucket >= dim.resolution { continue; }
            let mut pos = inp.pos;
            pos[0] = fno as i32;
            if !reported.insert((inp.dim, pos)) { continue; }
            let n = dim.first + inp.bucket;
            let idx = frame.acts.len() as u32;
            frame.acts.push(self.make_act(n, pos, true));
            if dim.kind == Kind::Action { frame.actions_ran.push((dim.channel, ActRef { frame: fno, index: idx })); }
        }
        // what an inference named peeks through where the report names nothing (R40)
        let weak = std::mem::take(&mut self.pending_weak);
        for (n, p) in weak {
            let Some(dim) = self.thalamus.registry[n].dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.make_act(n, pos, false));
        }
        stats.base_acts = frame.acts.len();
        self.thalamus.frames.push_back(frame);

        // the levels (§6, R26): activate, find, count, group, elect, allocate
        self.span_s = self.measure_board_span();
        self.stats_added = 0;
        self.stats_reused = 0;
        self.stats_retired = 0;
        if self.first_frame.is_none() { self.first_frame = Some(fno); }
        self.climb_levels(&mut stats);
        // and once more after the frame's last level (D24)
        self.cover_board();
        stats.added = self.stats_added;
        stats.reused = self.stats_reused;
        stats.retired = self.stats_retired;

        // connect and infer (§7), then decommission what is due (R38)
        self.connect_board();
        self.lift_pending.clear();
        self.infer_board();
        if !self.frozen { self.reap(); }

        // selection (§8)
        let winners = self.select();
        stats.voters = self.last_stats.voters;
        stats.voters_above_base = self.last_stats.voters_above_base;
        self.pending_weak = winners.iter().map(|w| (self.find_base_neuron(w.dim, w.bucket), w.pos)).collect();

        stats.apex_acts = self.thalamus.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter()).filter(|a| a.on_apex()).count();
        stats.neurons = self.alive;
        stats.patterns = self.thalamus.dict.len();
        stats.connections = self.thalamus.count_connections();
        // the window the machine keeps: the longest reach in time among the activations still open (D24)
        let mut longest = 1;
        for f in self.thalamus.frames.iter() { for a in &f.acts { if !a.gone && f.no + a.reach[0] as i64 >= fno { longest = longest.max(a.reach[0]); } } }
        self.max_reach_t = longest;
        let keep_from = fno - self.max_reach_t.max(self.span_s) as i64 - 1;
        while self.thalamus.frames.front().map_or(false, |f| f.no < keep_from) { self.thalamus.frames.pop_front(); }
        self.covered_rel.retain(|(a, b)| a.frame >= keep_from && b.frame >= keep_from);
        self.call_edges.retain(|c, _| c.frame >= keep_from);
        self.grown.retain(|c, _| c.frame >= keep_from);
        self.lifted.retain(|e| e.0.frame >= keep_from && e.1.frame >= keep_from);
        self.cov_inc.retain(|a, v| { v.retain(|b| b.frame >= keep_from); a.frame >= keep_from });
        self.kind_of.retain(|a, _| a.frame >= keep_from);
        self.lift_forms.retain(|e, v| { v.retain(|x| x.0.frame >= keep_from && x.1.frame >= keep_from); e.0.frame >= keep_from && e.1.frame >= keep_from });
        self.taken.retain(|c, _| c.frame >= keep_from);
        // a refused relation stays refused for as long as its two activations exist (D24)
        self.refused.retain(|e| e.0.frame >= keep_from && e.1.frame >= keep_from);
        self.form_src.retain(|e, v| { v.retain(|x| x.0.frame >= keep_from && x.1.frame >= keep_from); e.0.frame >= keep_from && e.1.frame >= keep_from });
        self.form_pre.retain(|e| e.0.frame >= keep_from && e.1.frame >= keep_from);
        self.call_thru.retain(|c, _| c.frame >= keep_from);
        self.call_fit.clear();
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }

    /// S (D9): one beyond the farthest node any function has from its origin in time, and at least the base reach.
    pub(crate) fn measure_board_span(&self) -> i32 {
        // a function's reach in time depends on its nodes and relations alone: it is read again only where they changed
        let mut memo = self.reach_memo.borrow_mut();
        let mut best = 1;
        for f in self.thalamus.dict.funcs.values() {
            let r = match memo.get(&f.id) {
                Some((n, rs, r)) if *n == f.nodes && *rs == f.rels => *r,
                _ => { let r = f.measure_reach(1)[0]; memo.insert(f.id, (f.nodes.clone(), f.rels.clone(), r)); r }
            };
            best = best.max(r);
        }
        if memo.len() > 2 * self.thalamus.dict.funcs.len() + 64 { memo.retain(|id, _| self.thalamus.dict.funcs.contains_key(id)); }
        best.max(1)
    }

    /// The record a call stands as now, after every lap it took (D53).
    pub(crate) fn follow_laps(&self, mut r: ActRef) -> ActRef {
        let mut n = 0;
        while let Some(x) = self.grown.get(&r) { r = *x; n += 1; if n > 4096 { break; } }
        r
    }

    pub(crate) fn make_act(&self, n: NeuronId, pos: Pos, strong: bool) -> Act {
        let nn = &self.thalamus.registry[n];
        let mut reach = [0; MAX_DIMS];
        for d in 0..MAX_DIMS { if d == 0 || nn.axes & (1 << d) != 0 { reach[d] = 1; } }
        Act { neuron: n, sym: n, kind: nn.kind, axes: nn.axes, pos, strong, reach, lo: pos, hi: pos, covered_at: None, coverers: Vec::new(), covers: Vec::new(), fired_by: None, rels: Vec::new(), inferences: Vec::new(), opened_at: self.frame_no, level: 0, height: 0, gone: false }
    }

    // ───────────────────────────── one level (§6, R26) ─────────────────────────────

    /// One level (R26): activate every neuron with an activation here (R20); find what the dictionary holds and
    /// fit it (R53, D31); count and group (R50, R54, R18), fitting again what changed; elect and allocate (R24,
    /// R16). Returns the relations returned, the fits accepted, and the interneurons allocated.
    /// The frame runs up all its levels before time runs at any (R26, R24). The climb: from level 0 up,
    /// each level's activations are activated and its frame half runs, until a level takes and starts nothing. Then
    /// from level 0 up each level as built: what is new at it since the climb is activated, its frame half runs over
    /// the relations only that brought, and its time half over every relation of the level.
    fn climb_levels(&mut self, stats: &mut FrameStats) {
        let fno = self.frame_no;
        let mut activated: FxHashSet<ActRef> = FxHashSet::default();
        let mut fresh_of: Vec<Vec<Fresh>> = Vec::new();
        // per level, what the climb did there: relations, fits, accepted, functions added, ms
        let mut climb: Vec<(usize, usize, usize, usize, f64)> = Vec::new();
        let at_level = |m: &Brain, level: u32| -> Vec<ActRef> { m.thalamus.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate()
            .filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone).map(|(i, _)| ActRef { frame: fno, index: i as u32 }).collect::<Vec<_>>()).collect() };
        let fired_of = |m: &Brain, refs: &[ActRef]| -> FxHashSet<Sym> { refs.iter().filter_map(|r| m.find_act(*r)).map(|a| a.sym).collect() };
        // the climb
        let mut level: u32 = 0;
        loop {
            let refs: Vec<ActRef> = at_level(self, level).into_iter().filter(|r| !activated.contains(r)).collect();
            if refs.is_empty() { break; }
            let lt = std::time::Instant::now();
            let added_before = self.stats_added;
            activated.extend(refs.iter().copied());
            let fresh = self.activate_level(&refs);
            while fresh_of.len() <= level as usize { fresh_of.push(Vec::new()); }
            fresh_of[level as usize] = fresh.clone();
            let fired = fired_of(self, &at_level(self, level));
            self.last_fits = 0;
            let (a, m, started) = if self.frozen { self.run_frozen_half(level, true, &fresh) } else { self.run_half(level, true, &fresh, &fired) };
            self.cover_board();
            while climb.len() <= level as usize { climb.push((0, 0, 0, 0, 0.0)); }
            climb[level as usize] = (fresh.len(), self.last_fits, a, self.stats_added - added_before, lt.elapsed().as_secs_f64() * 1000.0);
            stats.bids += fresh.len();
            stats.accepted += a;
            stats.minted += m;
            if a == 0 && started == 0 { break; }
            level += 1;
            if level > 1024 { break; }
        }
        // the levels as built, on the board the climb left
        let mut level: u32 = 0;
        loop {
            let all = at_level(self, level);
            if all.is_empty() { break; }
            let lt = std::time::Instant::now();
            let added_before = self.stats_added;
            let new: Vec<ActRef> = all.iter().copied().filter(|r| !activated.contains(r)).collect();
            activated.extend(new.iter().copied());
            let new_fresh = if new.is_empty() { Vec::new() } else { self.activate_level(&new) };
            while fresh_of.len() <= level as usize { fresh_of.push(Vec::new()); }
            fresh_of[level as usize].extend(new_fresh.iter().copied());
            let all_fresh = fresh_of[level as usize].clone();
            let fired = fired_of(self, &all);
            self.last_fits = 0;
            let (a1, m1, _) = if self.frozen { self.run_frozen_half(level, true, &new_fresh) } else { self.run_half(level, true, &new_fresh, &fired) };
            let (a2, m2, _) = if self.frozen { self.run_frozen_half(level, false, &all_fresh) } else { self.run_half(level, false, &all_fresh, &fired) };
            self.cover_board();
            stats.levels += 1;
            let c = climb.get(level as usize).copied().unwrap_or((0, 0, 0, 0, 0.0));
            stats.per_level.push((all.len(), c.0 + new_fresh.len(), c.1 + self.last_fits, c.2 + a1 + a2, c.3 + self.stats_added - added_before, c.4 + lt.elapsed().as_secs_f64() * 1000.0));
            stats.bids += new_fresh.len();
            stats.accepted += a1 + a2;
            stats.minted += m1 + m2;
            level += 1;
            if level > 1024 { break; }
        }
    }

    /// Activate a level's activations (§5): each row into its neuron's history, its relations returned, and the
    /// relations of this level in their one form (R21, D34).
    fn activate_level(&mut self, refs: &[ActRef]) -> Vec<Fresh> {
        assert!(self.lifts_due.is_empty(), "lifts are handed over before a level is activated");
        let mut by_neuron: FxHashMap<NeuronId, ActivateTask> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        for &r in refs {
            let nbs = self.observe_neighborhood(r);
            let (n, own) = { let a = self.find_act(r).unwrap(); (a.neuron, a.sym) };
            if !by_neuron.contains_key(&n) { order.push(n); }
            by_neuron.entry(n).or_default().rows.push(RowInput { act: r, own, nbs });
        }
        // a row written now holds the relations lifted to a call before it was written (D10)
        if !self.lift_pending.is_empty() {
            for &r in refs {
                for key in [r, self.follow_laps(r)] {
                    let Some(v) = self.lift_pending.remove(&key) else { continue };
                    for (nb, o) in v {
                        let Some((n, task)) = self.make_lift(r, nb, o) else { continue };
                        by_neuron.get_mut(&n).expect("a pending lift goes to a row of this level").lifts.push(task);
                    }
                }
            }
        }
        let tasks: Vec<(NeuronId, ActivateTask)> = order.iter().map(|&n| (n, by_neuron.remove(&n).unwrap())).collect();
        // a neuron records no more activations of one level than its history holds, and the rest return no relations (§5.1)
        for (n, t) in &tasks {
            if t.rows.len() > self.cfg.h { eprintln!("warning: frame {}: neuron #{} has {} activations at one level, more than the history depth of {}", self.frame_no, n, t.rows.len(), self.cfg.h); }
        }
        let returns: Vec<ActReturn> = self.thalamus.activate_neurons(tasks, !self.frozen).into_iter().flatten().collect();
        // the board's edges at each activation, and the relations of this level in their one form (R21, D34)
        let mut fresh: FxHashMap<(ActRef, ActRef), Fresh> = FxHashMap::default();
        for ret in &returns {
            let near_s = self.find_act(ret.act).unwrap().sym;
            let rels: Vec<BoardRel> = ret.rels.iter().map(|(nb, s)| BoardRel { other: nb.act, off: nb.off, strength: *s }).collect();
            for (nb, s) in &ret.rels {
                let e = Fresh { near: ret.act, far: nb.act, strength: *s, rel: canon(near_s, nb.sym, nb.off) };
                let k = edge(ret.act, nb.act);
                if fresh.get(&k).map_or(true, |x| x.strength < *s) { fresh.insert(k, e); }
            }
            if let Some(a) = self.find_act_mut(ret.act) { a.rels = rels; }
        }
        let mut fresh: Vec<Fresh> = fresh.into_values().collect();
        fresh.sort_by(|a, b| (a.near, a.far).cmp(&(b.near, b.far)));
        fresh
    }

    /// One half of a level on a frozen machine: the elections over the fits of the functions that exist, of one kind,
    /// as a learning machine runs them; it changes nothing that lasts: no function starts, joins, moves or leaves, and a fit
    /// of a function with no living interneuron is not in the election, since none is allocated.
    fn run_frozen_half(&mut self, level: u32, inside: bool, fresh: &[Fresh]) -> (usize, usize, usize) {
        let kind: Vec<Fresh> = fresh.iter().filter(|e| (e.rel.off[0] == 0) == inside).copied().collect();
        let found: Vec<FuncId> = self.find(&kind);
        let mine: Vec<_> = found.iter().filter(|id| self.thalamus.dict.get(**id).and_then(|f| f.inter).map_or(false, |n| self.thalamus.registry.is_alive(n)))
            .flat_map(|id| self.find_fits(*id)).collect();
        self.last_fits += mine.len();
        self.call_fit.clear();
        let (a, m) = self.elect(level, mine);
        (a, m, 0)
    }

    /// One half of a level (R24, R26, D34): find, elect, count, group, elect and start over the relations of one kind,
    /// inside the frame or in time, and the functions that hold them. Returns what its elections accepted and
    /// allocated, and how many functions it started.
    fn run_half(&mut self, level: u32, inside: bool, fresh: &[Fresh], fired: &FxHashSet<Sym>) -> (usize, usize, usize) {
        let (mut accepted, mut minted) = (0, 0);
            let kind: Vec<Fresh> = fresh.iter().filter(|e| (e.rel.off[0] == 0) == inside).copied().collect();
            // find (R53, D31): the functions that hold a relation of this kind returned at this level; a function
            // holds one kind (D15)
            let found: Vec<FuncId> = self.find(&kind);
            // elect (R24): the fits of the dictionary as it stands cover what it knows before anything in it changes
            let mine: Vec<_> = found.iter().filter(|id| self.thalamus.dict.get(**id).is_some()).flat_map(|id| self.find_fits(*id)).collect();
            self.last_fits += mine.len();
            // grouping reads the fits this election takes, and no other
            self.call_fit.clear();
            let (a1, m1) = self.elect(level, mine);
            accepted += a1;
            minted += m1;
            // what the election left on the board: every relation of this kind no call covers and that is not
            // lifted, its ends standing
            let left: Vec<Fresh> = kind.iter().filter(|e| !self.covered_rel.contains(&crate::recognize::edge(e.near, e.far)) && !self.lifted.contains(&crate::recognize::edge(e.near, e.far))
                && self.find_act(e.near).map_or(false, |a| !a.gone && a.covered_at.is_none() && self.stands(a, self.frame_no)) && self.find_act(e.far).map_or(false, |a| !a.gone && a.covered_at.is_none() && self.stands(a, self.frame_no))).copied().collect();
            // count and group (R50, R54, R18): over the relations of this kind and the functions that hold them
            let rest: Vec<_> = found.iter().filter(|id| self.thalamus.dict.get(**id).is_some()).flat_map(|id| self.find_fits(*id)).collect();
            let board: Vec<Fresh> = self.list_board_rels().into_iter().filter(|e| (e.rel.off[0] == 0) == inside).collect();
            self.half = Some(inside);
            let changed = self.group(&rest, &left, &board, fired);
            self.half = None;
            self.call_fit.clear();
            self.span_s = self.measure_board_span();
            // elect again (R24): a function that changed in group is fit on what the first election left
            let again: Vec<_> = changed.iter().filter(|id| self.thalamus.dict.get(**id).is_some()).flat_map(|id| self.find_fits(*id)).collect();
            self.last_fits += again.len();
            let (a2, m2) = self.elect(level, again);
            accepted += a2;
            minted += m2;
            // start (R50): one function at a time, from the relations that may start one and still stand
            let queue = std::mem::take(&mut self.start_queue);
            let (a3, m3) = self.start_one_at_a_time(level, &queue, &left);
            accepted += a3;
            minted += m3;
        (accepted, minted, self.last_start_rounds)
    }

    /// How many functions of the dictionary never took a call: no interneuron, or none alive.
    pub fn count_functions_never_called(&self) -> usize {
        self.thalamus.dict.funcs.values().filter(|f| f.inter.map_or(true, |n| !self.thalamus.registry.is_alive(n))).count()
    }

    /// Start (R50): functions start one at a time, and only from a relation that still stands. A relation stands
    /// where at least one instance of it on the board is covered by no call and is not lifted, refused or blocked,
    /// its ends standing (D24, D55). Of the relations that may start a function and stand, the one with the most
    /// instances standing starts first, and on a tie the one whose standing instance is oldest; the election takes
    /// its fits (R24), and the machine looks again at what stands. Returns what the elections accepted and allocated.
    fn start_one_at_a_time(&mut self, level: u32, queue: &[crate::dictionary::Rel], left: &[Fresh]) -> (usize, usize) {
        let (mut accepted, mut minted) = (0, 0);
        let mut round = 0;
        loop {
            let standing = |m: &Brain, e: &Fresh| -> bool {
                let k = crate::recognize::edge(e.near, e.far);
                if m.covered_rel.contains(&k) || m.lifted.contains(&k) || m.refused.contains(&k) || m.is_blocked(e.near, e.far) { return false; }
                let up = |r: ActRef| m.find_act(r).map_or(false, |a| !a.gone && a.covered_at.is_none() && m.stands(a, m.frame_no));
                up(e.near) && up(e.far)
            };
            let mut best: Option<((bool, std::cmp::Reverse<usize>, ActRef), crate::dictionary::Rel)> = None;
            for r in queue {
                if self.thalamus.dict.find_holder(r).is_some() { continue; }
                let inst: Vec<&Fresh> = left.iter().filter(|e| e.rel == *r && standing(self, e)).collect();
                if inst.is_empty() { continue; }
                let oldest = inst.iter().map(|e| e.near.min(e.far)).min().unwrap();
                let key = (r.off[0] != 0, std::cmp::Reverse(inst.len()), oldest);
                if best.as_ref().map_or(true, |(k, _)| key < *k) { best = Some((key, *r)); }
            }
            let Some((_, r)) = best else { break };
            round += 1;
            let id = self.thalamus.dict.start(r);
            self.stats_added += 1;
            let fits = self.find_fits(id);
            self.last_fits += fits.len();
            let (a, m) = self.elect(level, fits);
            accepted += a;
            minted += m;
        }
        self.last_start_rounds = round;
        (accepted, minted)
    }

    // ───────────────────────────── inspection ─────────────────────────────

    /// How many living neurons stand at each height, read off their last activation.
    pub fn count_heights(&self) -> Vec<usize> {
        let mut heights: FxHashMap<NeuronId, u32> = FxHashMap::default();
        for f in self.thalamus.frames.iter() { for a in &f.acts { let e = heights.entry(a.neuron).or_default(); if a.height > *e { *e = a.height; } } }
        let mut v = Vec::new();
        for (id, _) in self.thalamus.registry.iter().filter(|(_, n)| n.alive) {
            let h = heights.get(&id).copied().unwrap_or(0) as usize;
            if v.len() <= h { v.resize(h + 1, 0); }
            v[h] += 1;
        }
        v
    }
}

impl Brain {
    /// The dictionary (D36, D56).
    pub fn get_dictionary(&self) -> &Dictionary { &self.thalamus.dict }

    /// The board (D9): the frames still open, with their activations.
    pub fn get_board(&self) -> &VecDeque<FrameRec> { &self.thalamus.frames }

    /// Describe neurons: each one's kind, row count and connection count, the functions it is the interneuron of, and
    /// its first sixteen connections in key order.
    pub fn describe_neurons(&mut self, ids: &[NeuronId]) -> Vec<String> {
        let held = self.thalamus.inspect_neurons(ids.to_vec());
        let dict = &self.thalamus.dict;
        ids.iter().zip(held).map(|(&id, (rows, conns))| {
            let n = &self.thalamus.registry[id];
            let mut s = match n.kind {
                Kind::Inter => format!("#{} inter", id),
                _ => format!("#{} {:?} dim{} b{}", id, n.kind, n.dim.unwrap(), n.bucket),
            };
            s.push_str(&format!(" H={} conns={}
", rows, conns.len()));
            for fid in dict.list_of_inter(id) { if let Some(f) = dict.get(*fid) { s.push_str(&format!("  is {}
", dict.describe(f))); } }
            for ((own, t, o), c) in conns.into_iter().take(16) {
                s.push_str(&format!("    {} -> {} @{:?} s={:.0} est={:+.2}
", dict.describe_sym(own), dict.describe_sym(t), o, c.strength, c.estimate()));
            }
            s
        }).collect()
    }

    /// Every neuron's fixed facts, by id.
    pub fn get_registry(&self) -> &crate::thalamus::Registry { &self.thalamus.registry }
}
