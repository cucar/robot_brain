//! The machine (§2, §3, §6, §7, §8): frames, open activations, the dictionary, the levels with recognition
//! and the election, the connect and infer calls, and selection.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

use crate::function::*;
use crate::neuron::*;
use crate::recognize::{edge, Fresh};
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

/// The one choice the build leaves open (switch 15): whether a relation stays on the board, a candidate and a relation
/// that may start a function, while no call covers the relation itself, even where one of its ends now stands under a
/// call (R50, D5). On by default.
#[derive(Clone, Copy, Debug)]
pub struct Switches {
    pub s15_uncovered_relations: bool,
}

impl Default for Switches {
    fn default() -> Self { Switches { s15_uncovered_relations: true } }
}

pub struct Machine {
    pub cfg: Config,
    pub act_dims: Vec<ActDim>,
    pub dims: Vec<NeuronDim>,
    pub channels: Vec<Channel>,
    /// The neurons, owned by the columns of the regions and reached by id.
    pub thalamus: Thalamus,
    pub alive: usize,
    pub base_count: u32,
    pub frames: VecDeque<FrameRec>,
    pub frame_no: Frame,
    pub(crate) pending_weak: Vec<(NeuronId, Pos)>,
    pub max_reach_t: i32,
    /// S (D9): how long an activation stands on the board, the longest reach in time among the neurons the machine
    /// holds, one beyond the farthest node any function has from its origin. It moves with the dictionary.
    pub span_s: i32,
    /// The dictionary (D36, D56): every function and its interneuron.
    pub dict: Dictionary,
    /// Functions that left the dictionary, kept readable while what they fired is open (R38).
    pub(crate) retired_funcs: FxHashMap<FuncId, (Func, Frame)>,
    /// Per interneuron, the function that fired it last, for inferring it (R28).
    pub(crate) last_func: FxHashMap<NeuronId, FuncId>,
    /// Interneurons whose last function has left, waiting for their windows to close (R38).
    pub ledger: Vec<NeuronId>,
    /// Per interneuron on the ledger, the frame before which its subtree still stands on the board: it is not
    /// looked at again until then (R38).
    pub(crate) ledger_due: FxHashMap<NeuronId, Frame>,
    pub(crate) stats_added: usize,
    pub(crate) stats_reused: usize,
    pub(crate) last_fits: usize,
    /// Per interneuron, the farthest it has seen in each dimension, over its activations (D4).
    pub(crate) reach_of: FxHashMap<NeuronId, Pos>,
    pub(crate) stats_retired: usize,
    /// Switch 21: each time a relation not at an end would have left were it allowed, by the test it met: its
    /// strength below the threshold, its worth alone above its contribution, a higher gain in another function.
    pub middle_strength: u64,
    pub middle_worth: u64,
    pub middle_join: u64,
    /// Switch 21: each time an end relation's leaving took the function's origin with it.
    pub origin_moves: u64,
    /// Of those, the times two nodes tied for the latest; and whether the leaves at an origin and the relations that
    /// stay in the middle are printed without the full trace (off by default; prints only, changes nothing).
    pub origin_ties: u64,
    pub trace_origin: bool,
    pub(crate) middle_seen: FxHashSet<(Frame, FuncId, Rel, u8)>,
    /// For the count only: relations that moved from one function into another, those only a fit of the first
    /// election offered (switch 17), and moves of a relation that had moved before; candidates no longer read.
    pub moves_total: u64,
    pub moves_first: u64,
    pub moves_again: u64,
    pub move_count: FxHashMap<Rel, u32>,
    pub reread_skips: u64,
    /// For the count only: changes to a function whose interneuron already held a connection to an action.
    pub changed_after_action: u64,
    pub changed_after_action_funcs: FxHashSet<FuncId>,
    /// A relation's two end counts, remembered for one grouping step, while no row changes (a speed-up only).
    pub(crate) halved_memo: std::cell::RefCell<Option<FxHashMap<Rel, (u32, u32)>>>,
    /// A relation's with and without at a node, given the set's relations there, remembered for one grouping step
    /// while no row changes (a speed-up only).
    pub(crate) answer_memo: std::cell::RefCell<Option<FxHashMap<(Sym, Sym, Offset, Vec<(Sym, Offset)>), (u32, u32)>>>,
    /// Whether two activations stand over a shared one, remembered for one grouping step while nothing on the board
    /// changes (a speed-up only).
    pub(crate) share_memo: std::cell::RefCell<Option<FxHashMap<(ActRef, ActRef), bool>>>,
    pub last_stats: FrameStats,
    pub last_winners: Vec<Winner>,
    pub trace: bool,
    /// Whether a relation standing alone or in a function of one relation joins on a tie (R54), where switch 4 is on;
    /// with switch 4 off a tie joins, as in the build before it.
    pub tie_join: bool,
    /// print a line for every join, on a tie or by gain, and for every relation that joined and later left
    pub trace_joins: bool,
    /// relations that have joined a function, to tell a relation that joined and later left (diagnostic)
    pub(crate) joined_ever: rustc_hash::FxHashSet<crate::function::Rel>,
    /// The changes since the committed build, each on or off on its own, for measuring them (a diagnostic).
    pub sw: Switches,
    /// Print every function that leaves the dictionary, with why, a diagnostic.
    pub trace_retire: bool,
    /// Print each level's counts as it finishes, a diagnostic.
    pub trace_levels: bool,
    /// A diagnostic: run no level above this one in a frame.
    pub stop_after_level: Option<u32>,
    /// A diagnostic: the board edges the accepted fits of this frame covered.
    pub covered_edges: Vec<(ActRef, ActRef)>,
    /// The coverage set per relation (D24): every board relation an accepted call covers, over the frames on the
    /// board; nothing lands on one again (D28).
    pub(crate) covered_rel: FxHashSet<(ActRef, ActRef)>,
    /// switch 11: every call that fired, and what it covered, each at its offset from the call (R28): a call reads as
    /// this whether or not its interneuron still has a function
    pub(crate) covered_by: FxHashMap<ActRef, Vec<(Sym, Offset)>>,
    /// switch 17: the board relations each call's fit covered
    pub(crate) call_edges: FxHashMap<ActRef, Vec<(ActRef, ActRef)>>,
    /// switch 17: per accepted call, the calls its fit named or read a node through
    pub(crate) call_thru: FxHashMap<ActRef, Vec<ActRef>>,
    /// switch 17: this frame's accepted fits, by their calls, so grouping reads what they landed on
    pub(crate) call_fit: FxHashMap<ActRef, crate::recognize::Fit>,
    /// The calls the election running now has taken (D28): an activation they landed on may be landed on again
    /// in this election, each call covering relations of its own (D8, D10).
    pub(crate) electing: FxHashSet<ActRef>,
    /// Per call in its pose, every node its expansion puts back, at any depth, each at its offset from the call's
    /// position (R28, R50); read from the dictionary and dropped whenever the dictionary changes.
    pub(crate) expansions: std::cell::RefCell<FxHashMap<Sym, std::rc::Rc<Vec<(Sym, Offset, Vec<NeuronId>)>>>>,
    /// Per expansion kept, the functions it was read through: a change to one drops the expansions that used it.
    pub(crate) expansion_deps: std::cell::RefCell<FxHashMap<Sym, Vec<crate::function::FuncId>>>,
    /// Within one grouping, the histories stand still: per relation, the end it is read from and its strength,
    /// worked out once and dropped whenever a function changes.
    pub(crate) held_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), Option<(Sym, Sym, Offset)>>>,
    pub(crate) strength_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), u32>>,
    pub(crate) returned_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), u32>>,
    /// A test: the machine recognizes and predicts but changes nothing of itself. No rows enter a history, no
    /// function is added, moved or pruned, no interneuron is allocated, no connection or reward is written,
    /// nothing is decommissioned.
    pub frozen: bool,
    /// Experiment: at each level, elect the fits of the functions the dictionary already holds before grouping,
    /// then group over what that left uncovered, then fit what grouping changed and elect again over the rest.
    /// Off, a level is the spec's: find, count, group, elect.
    pub recognize_first: bool,
}

impl Machine {
    /// One region, a column per available core. The third argument is kept for callers and read by nothing: a
    /// base neuron sees one position in every dimension (D4, R4).
    pub fn new(h: usize, policy: Policy, _radius: i32) -> Self {
        let columns = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::with_layout(h, policy, 1, columns)
    }

    /// With the dispatch laid out: `regions` regions of `columns` columns each.
    pub fn with_layout(h: usize, policy: Policy, regions: usize, columns: usize) -> Self {
        Machine {
            cfg: Config { h, policy, regions: regions.max(1), columns: columns.max(1) },
            act_dims: vec![ActDim { name: "time".into(), radius: 1 }],
            dims: Vec::new(),
            channels: Vec::new(),
            thalamus: Thalamus::new(regions, columns),
            alive: 0,
            base_count: 0,
            frames: VecDeque::new(),
            frame_no: 0,
            pending_weak: Vec::new(),
            max_reach_t: 1,
            span_s: 1,
            dict: Dictionary::default(),
            retired_funcs: FxHashMap::default(),
            last_func: FxHashMap::default(),
            ledger: Vec::new(),
            ledger_due: FxHashMap::default(),
            stats_added: 0,
            stats_reused: 0,
            last_fits: 0,
            reach_of: FxHashMap::default(),
            stats_retired: 0,
            middle_strength: 0,
            middle_worth: 0,
            middle_join: 0,
            origin_moves: 0,
            origin_ties: 0,
            trace_origin: false,
            middle_seen: FxHashSet::default(),
            moves_total: 0,
            moves_first: 0,
            moves_again: 0,
            move_count: FxHashMap::default(),
            reread_skips: 0,
            changed_after_action: 0,
            changed_after_action_funcs: FxHashSet::default(),
            halved_memo: std::cell::RefCell::new(None),
            answer_memo: std::cell::RefCell::new(None),
            share_memo: std::cell::RefCell::new(None),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            tie_join: false,
            trace_joins: true,
            joined_ever: Default::default(),
            sw: Switches::default(),
            trace_retire: false,
            trace: false,
            trace_levels: false,
            stop_after_level: None,
            covered_edges: Vec::new(),
            covered_rel: FxHashSet::default(),
            covered_by: FxHashMap::default(),
            call_edges: FxHashMap::default(),
            call_thru: FxHashMap::default(),
            call_fit: FxHashMap::default(),
            electing: FxHashSet::default(),
            expansions: std::cell::RefCell::new(FxHashMap::default()),
            expansion_deps: std::cell::RefCell::new(FxHashMap::default()),
            held_memo: std::cell::RefCell::new(FxHashMap::default()),
            strength_memo: std::cell::RefCell::new(FxHashMap::default()),
            returned_memo: std::cell::RefCell::new(FxHashMap::default()),
            frozen: false,
            recognize_first: true,
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

    pub(crate) fn declare_dim(&mut self, name: &str, channel: ChannelId, kind: Kind, resolution: u32, axes: u8) -> DimId {
        let id = self.dims.len() as DimId;
        let first = self.thalamus.len() as NeuronId;
        for b in 0..resolution {
            self.thalamus.push(Neuron::base(first + b, kind, id, b, channel, axes, self.cfg.h));
        }
        self.alive += resolution as usize;
        self.base_count += resolution;
        self.dims.push(NeuronDim { id, name: name.into(), channel, kind, resolution, first });
        id
    }

    pub fn base_neuron(&self, dim: DimId, bucket: u32) -> NeuronId { self.dims[dim as usize].first + bucket }

    pub fn act(&self, r: ActRef) -> Option<&Act> {
        self.frames.iter().find(|f| f.no == r.frame).and_then(|f| f.acts.get(r.index as usize))
    }

    pub(crate) fn act_mut(&mut self, r: ActRef) -> Option<&mut Act> {
        self.frames.iter_mut().find(|f| f.no == r.frame).and_then(|f| f.acts.get_mut(r.index as usize))
    }

    pub(crate) fn frame_mut(&mut self, no: Frame) -> Option<&mut FrameRec> { self.frames.iter_mut().find(|f| f.no == no) }

    /// Clear every open activation and standing inference; keep the dictionary and connections.
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
        self.middle_seen.clear();
        let mut stats = FrameStats { frame: fno, ..Default::default() };

        // the frame's base activations: strong where the world reported, weak where only an inference named them (D40)
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
            frame.acts.push(self.new_act(n, pos, true));
            if dim.kind == Kind::Action { frame.actions_ran.push((dim.channel, ActRef { frame: fno, index: idx })); }
        }
        // what an inference named peeks through where the report names nothing (R40)
        let weak = std::mem::take(&mut self.pending_weak);
        for (n, p) in weak {
            let Some(dim) = self.thalamus[n].dim else { continue };
            let mut pos = p;
            pos[0] = fno as i32;
            if reported.contains(&(dim, pos)) { continue; }
            frame.acts.push(self.new_act(n, pos, false));
        }
        stats.base_acts = frame.acts.len();
        self.frames.push_back(frame);

        // the levels (§6, R26): activate, find, count, group, elect, allocate
        self.span_s = self.board_span();
        self.covered_edges.clear();
        self.stats_added = 0;
        self.stats_reused = 0;
        self.stats_retired = 0;
        let mut level: u32 = 0;
        loop {
            let refs: Vec<ActRef> = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate()
                .filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone).map(|(i, _)| ActRef { frame: fno, index: i as u32 }).collect::<Vec<_>>()).collect();
            if refs.is_empty() { break; }
            
            stats.levels += 1;
            let lt = std::time::Instant::now();
            let added_before = self.stats_added;
            let (bids, accepted, minted) = self.run_level(level, &refs);
            stats.per_level.push((refs.len(), bids, self.last_fits, accepted, self.stats_added - added_before, lt.elapsed().as_secs_f64() * 1000.0));
            if self.trace_levels {
                let (acts, rels, fits, acc, added, ms) = stats.per_level[stats.per_level.len() - 1];
                eprintln!("[level {}] activations {} | relations returned {} | fits {} | accepted {} | functions added {} | {:.0} ms", level, acts, rels, fits, acc, added, ms);
            }
            stats.bids += bids;
            stats.accepted += accepted;
            stats.minted += minted;
            if self.stop_after_level == Some(level) { break; }
            level += 1;
            // nothing caps the depth (R26); a frame this deep means a fault
            if level > 1024 { break; }
        }
        stats.added = self.stats_added;
        stats.reused = self.stats_reused;
        stats.retired = self.stats_retired;
        

        // connect and infer (§7), then decommission what is due (R38)
        self.connect_all();
        self.infer_all();
        if !self.frozen { self.reap(); }

        // selection (§8)
        let winners = self.select();
        stats.voters = self.last_stats.voters;
        stats.voters_above_base = self.last_stats.voters_above_base;
        self.pending_weak = winners.iter().map(|w| (self.base_neuron(w.dim, w.bucket), w.pos)).collect();

        stats.apex_acts = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter()).filter(|a| a.on_apex()).count();
        stats.neurons = self.alive;
        stats.patterns = self.dict.len();
        stats.connections = self.thalamus.iter().filter(|n| n.alive).map(|n| n.connections.len()).sum();
        // the window the machine keeps: the longest reach in time among the activations still open (D24)
        let mut longest = 1;
        for f in self.frames.iter() { for a in &f.acts { if !a.gone && f.no + a.reach[0] as i64 >= fno { longest = longest.max(a.reach[0]); } } }
        self.max_reach_t = longest;
        let keep_from = fno - self.max_reach_t.max(self.span_s) as i64 - 1;
        while self.frames.front().map_or(false, |f| f.no < keep_from) { self.frames.pop_front(); }
        self.covered_rel.retain(|(a, b)| a.frame >= keep_from && b.frame >= keep_from);
        self.call_edges.retain(|c, _| c.frame >= keep_from);
        self.call_thru.retain(|c, _| c.frame >= keep_from);
        self.call_fit.clear();
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.last_stats = stats;
        self.last_winners = winners.clone();
        winners
    }

    /// S (D9): one beyond the farthest node any function has from its origin in time, and at least the base reach.
    pub(crate) fn board_span(&self) -> i32 {
        self.dict.funcs.values().map(|f| f.reach(1)[0]).max().unwrap_or(1).max(1)
    }

    /// A bare activation of `n` at `pos` (D8): a base activation sees one position in every dimension (D4).
    pub(crate) fn new_act(&self, n: NeuronId, pos: Pos, strong: bool) -> Act {
        let nn = &self.thalamus[n];
        let mut reach = [0; MAX_DIMS];
        for d in 0..MAX_DIMS { if d == 0 || nn.axes & (1 << d) != 0 { reach[d] = 1; } }
        Act { neuron: n, sym: n, kind: nn.kind, axes: nn.axes, pos, strong, reach, lo: pos, hi: pos, covered_at: None, coverers: Vec::new(), seen_by: Vec::new(), covers: Vec::new(), fired_by: None, rels: Vec::new(), inferences: Vec::new(), opened_at: self.frame_no, level: 0, height: 0, gone: false }
    }

    // ───────────────────────────── one level (§6, R26) ─────────────────────────────

    /// One level (R26): activate every neuron with an activation here (R20); find what the dictionary holds and
    /// fit it (R53, D31); count and group (R50, R54, R18), fitting again what changed; elect and allocate (R24,
    /// R16). Returns the relations returned, the fits accepted, and the interneurons allocated.
    fn run_level(&mut self, level: u32, refs: &[ActRef]) -> (usize, usize, usize) {
        let mut by_neuron: FxHashMap<NeuronId, Vec<RowInput>> = FxHashMap::default();
        let mut order: Vec<NeuronId> = Vec::new();
        let mut seen: Vec<(ActRef, NeuronId, ActRef)> = Vec::new();
        for &r in refs {
            let nbs = self.neighborhood(r);
            let (n, own) = { let a = self.act(r).unwrap(); (a.neuron, a.sym) };
            for nb in &nbs { seen.push((nb.act, n, r)); }
            if !by_neuron.contains_key(&n) { order.push(n); }
            by_neuron.entry(n).or_default().push(RowInput { act: r, own, nbs });
        }
        // each neighbor learns whose row holds it, for the calls that land on it to be written there (D10)
        if !self.frozen { for (b, n, r) in seen { if let Some(x) = self.act_mut(b) { x.seen_by.push((n, r)); } } }
        let tasks: Vec<(NeuronId, Vec<RowInput>)> = order.iter().map(|&n| (n, by_neuron.remove(&n).unwrap())).collect();
        let returns: Vec<ActReturn> = if self.frozen {
            tasks.into_iter().flat_map(|(n, rows)| self.thalamus[n].peek(rows)).collect()
        } else {
            self.thalamus.process_level(tasks).into_iter().flat_map(|r| r.returns).collect()
        };
        // the board's edges at each activation, and the relations of this level in their one form (R21, D34)
        let mut fresh: FxHashMap<(ActRef, ActRef), Fresh> = FxHashMap::default();
        for ret in &returns {
            let near_s = self.act(ret.act).unwrap().sym;
            let rels: Vec<BoardRel> = ret.rels.iter().map(|(nb, s)| BoardRel { other: nb.act, off: nb.off, strength: *s }).collect();
            for (nb, s) in &ret.rels {
                let e = Fresh { near: ret.act, far: nb.act, off: nb.off, strength: *s, rel: canon(near_s, nb.sym, nb.off) };
                let k = edge(ret.act, nb.act);
                if fresh.get(&k).map_or(true, |x| x.strength < *s) { fresh.insert(k, e); }
            }
            if let Some(a) = self.act_mut(ret.act) { a.rels = rels; }
        }
        let mut fresh: Vec<Fresh> = fresh.into_values().collect();
        fresh.sort_by(|a, b| (a.near, a.far).cmp(&(b.near, b.far)));
        let bids = fresh.len();
        // find, and fit what was found (R53, D31)
        let mut found = self.find(&fresh);
        if self.trace { for e in &fresh { eprintln!("[fresh f{} l{}] {}→{}{:?} s={} held {:?}", self.frame_no, level, self.dict.sym_text(e.rel.a), self.dict.sym_text(e.rel.b), &e.rel.off[..], e.strength, self.dict.holder(&e.rel)); } }
        let mut fits: Vec<_> = found.iter().flat_map(|id| self.fits_of(*id)).collect();
        // two elections (§6, R24, R50): what the dictionary holds covers the board first; grouping reads only what
        // it left uncovered, and a function that changed or formed is fit on what it left and elected in the second
        if !self.frozen && self.recognize_first {
            let fired: FxHashSet<Sym> = refs.iter().filter_map(|r| self.act(*r)).map(|a| a.sym).collect();
            self.last_fits = fits.len();
            // switch 17: grouping reads the fits this level's first election takes, and no other
            self.call_fit.clear();
            let (mut accepted, mut minted) = self.elect(level, fits);
            let first_taken: Vec<ActRef> = self.call_fit.keys().copied().collect();
            // one count (switch 9): the level's rows are read as the first election left them, for starting and
            // retiring between the two elections; the frame's end judges them again
            
            let left: Vec<Fresh> = if self.sw.s15_uncovered_relations {
                // what the first election left on the board: every relation no call covers, its ends standing (switch 15)
                fresh.iter().filter(|e| !self.covered_rel.contains(&crate::recognize::edge(e.near, e.far))
                    && self.act(e.near).map_or(false, |a| !a.gone && self.stands(a, self.frame_no)) && self.act(e.far).map_or(false, |a| !a.gone && self.stands(a, self.frame_no))).copied().collect()
            } else {
                fresh.iter().filter(|e| self.act(e.near).map_or(false, |a| a.on_apex()) && self.act(e.far).map_or(false, |a| a.on_apex())).copied().collect()
            };
            let rest: Vec<_> = found.iter().flat_map(|id| self.fits_of(*id)).collect();
            let board = self.board_rels();
            let changed = self.group(&rest, &left, &board, &fired);
            let _ = first_taken;
            self.call_fit.clear();
            self.span_s = self.board_span();
            let again: Vec<_> = changed.iter().filter(|id| self.dict.get(**id).is_some()).flat_map(|id| self.fits_of(*id)).collect();
            if self.trace { for f in &again { eprintln!("[fit2 f{} l{}] f{} entry {:?} at node {} landed {} calls {} edges {} failed {:?} laps {}", self.frame_no, level, f.func, f.entry.1, f.entry.0, f.landed.len(), f.calls.len(), f.edges.len(), f.failed, f.laps); } }
            self.last_fits += again.len();
            let (a2, m2) = self.elect(level, again);
            accepted += a2;
            minted += m2;
            return (bids, accepted, minted);
        }
        // count and group (R50, R54, R18); what changed is fit again before the election
        if !self.frozen {
            let fired: FxHashSet<Sym> = refs.iter().filter_map(|r| self.act(*r)).map(|a| a.sym).collect();
            let changed = self.group(&fits, &fresh, &fresh, &fired);
            self.span_s = self.board_span();
            if !changed.is_empty() {
                found.extend(changed);
                found.retain(|id| self.dict.get(*id).is_some());
                found.sort_unstable();
                found.dedup();
                fits = found.iter().flat_map(|id| self.fits_of(*id)).collect();
            }
        } else {
            // a frozen machine allocates nothing
            fits.retain(|f| self.dict.get(f.func).and_then(|f| f.inter).is_some());
        }
        if self.trace { for f in &fits { eprintln!("[fit f{} l{}] f{} entry {:?} at node {} landed {} calls {} edges {} failed {:?} laps {}", self.frame_no, level, f.func, f.entry.1, f.entry.0, f.landed.len(), f.calls.len(), f.edges.len(), f.failed, f.laps); } }
        self.last_fits = fits.len();
        let (accepted, minted) = self.elect(level, fits);
        (bids, accepted, minted)
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

    /// Switch 21's count: the relations not at an end that would have left, by test, and the origins that moved.
    pub fn middle_summary(&self) -> String {
        let edges: usize = self.dict.funcs.values().map(|f| f.rels.len()).sum();
        format!("dictionary: {} functions, {} edges | not at an end, would have left: {} (strength below the threshold {}, worth alone above its contribution {}, to join another function {}) | relation at the origin left {} (ties for the latest node {})",
            self.dict.len(), edges, self.middle_strength + self.middle_worth + self.middle_join, self.middle_strength, self.middle_worth, self.middle_join, self.origin_moves, self.origin_ties)
    }

    pub fn describe_neuron(&self, id: NeuronId) -> String {
        let n = &self.thalamus[id];
        let mut s = match n.kind {
            Kind::Inter => format!("#{} inter", id),
            _ => format!("#{} {:?} dim{} b{}", id, n.kind, n.dim.unwrap(), n.bucket),
        };
        s.push_str(&format!(" H={} conns={}\n", n.history.len(), n.connections.len()));
        for fid in self.dict.of_inter(id) { if let Some(f) = self.dict.get(*fid) { s.push_str(&format!("  is {}\n", self.dict.describe(f))); } }
        let mut conns: Vec<(&(Sym, Sym, Offset), &Conn)> = n.connections.iter().collect();
        conns.sort_by(|a, b| a.0.cmp(b.0));
        for ((own, t, o), c) in conns.into_iter().take(16) {
            s.push_str(&format!("    {} -> {} @{:?} s={:.0} est={:+.2}\n", self.dict.sym_text(*own), self.dict.sym_text(*t), o, c.strength, c.estimate()));
        }
        s
    }
}
