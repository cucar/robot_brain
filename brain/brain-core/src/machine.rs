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
    /// The committed build's crossing charge: a fit pays one for each call taken earlier in the same election that
    /// landed on an activation it lands on.
    pub cross_charge: bool,
    /// Test: how a candidate that would close a cycle is weighed: 0 the function's worth as a cycle less its worth
    /// as it stands (the build); 1 (`round`) the candidate over the function's interneuron's rows: it joins where
    /// the rows that hold it, less one, is above zero, nothing else read
    pub closing: u8,
    /// Test only (off by default): a relation no function holds joins at a loss as it did before, its gain against its
    /// worth in a function of its own even where that is below zero; off, it joins only where its gain is above zero
    pub join_at_a_loss: bool,
}

impl Default for Switches {
    fn default() -> Self { Switches { s15_uncovered_relations: true, cross_charge: false, closing: 0, join_at_a_loss: false } }
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
    /// The same while frozen: what a frozen machine's calls would have written there, read before `last_func` and
    /// dropped once the machine learns again, so a test leaves `last_func` as it found it
    pub(crate) frozen_last_func: FxHashMap<NeuronId, FuncId>,
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
    /// Trace each activation as it is covered (D24).
    pub trace_cover: bool,
    /// The activations heard in the last selection, and whether to print the voting of a frame
    pub last_heard: Vec<ActRef>,
    /// The relations covered by the calls taken so far at one level of one frame, across every election of the
    /// level, each with the call and its function, as adjacency, and the sets they join (D24); the frame and level
    /// they are for
    pub(crate) joined: FxHashMap<ActRef, Vec<(ActRef, ActRef, FuncId)>>,
    pub(crate) jparent: FxHashMap<ActRef, ActRef>,
    pub(crate) joined_at: Option<(Frame, u32)>,
    /// The relations refused because they would close a deadlock (D24, D55): refused for as long as their two
    /// activations exist
    pub(crate) refused: FxHashSet<(ActRef, ActRef)>,
    /// For each edge a relation became at a call, the relations it stands for; and the edges that stood on the board
    /// before any relation became them (D26, D24)
    pub(crate) form_src: FxHashMap<(ActRef, ActRef), Vec<(ActRef, ActRef)>>,
    pub(crate) form_pre: FxHashSet<(ActRef, ActRef)>,
    /// Each base activation covered: its frame and how many calls had landed on it then
    pub cover_landed: Vec<(Frame, usize)>,
    /// The frames in which each function that is a cycle had a call, printed at the end where `print_cycles`
    pub cycle_calls: FxHashMap<FuncId, (String, Vec<Frame>)>,
    pub print_cycles: bool,
    /// The relations that may start a function in this half of the level, in the order grouping read them (R50)
    pub(crate) start_queue: Vec<crate::function::Rel>,
    /// The kind of relation this grouping reads, true inside the frame (R24, D34); None reads both
    pub(crate) half: Option<bool>,
    /// For the count only: joins (from no function) and moves (from another function), relations inside the frame
    /// first, in time second
    pub joins_by: [u64; 2],
    /// How many functions the last start step started
    pub(crate) last_start_rounds: usize,
    /// What each level of this frame has joined, kept while the frame visits another level (D24, R24, R26)
    pub(crate) joined_stash: FxHashMap<u32, (FxHashMap<ActRef, Vec<(ActRef, ActRef, FuncId)>>, FxHashMap<ActRef, ActRef>)>,
    /// For the report: the pixels (x, y, value) left bare after level 0 of the last frame
    pub bare_after_l0: Vec<(i32, i32, u32)>,
    pub moves_by: [u64; 2],
    pub trace_vote: bool,
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
    /// How many relations were lifted to a call, how many of them between two calls, and activations covered at age 0
    pub up_count: usize,
    pub up_pair_count: usize,
    pub cover_age0: usize,
    /// The first frame of the run (D24), and every activation covered by one relation: the frame, and the activation
    pub first_frame: Option<Frame>,
    pub one_side: Vec<(Frame, ActRef)>,
    /// What each cover check of the frame covered: the level it ran after (−1 after the last), base activations, calls.
    pub cover_log: Vec<(i64, usize, usize)>,
    /// A call that took a lap (D53): the record it grew from, to the record it stands as now.
    pub grown: FxHashMap<ActRef, ActRef>,
    pub(crate) middle_seen: FxHashSet<(Frame, FuncId, Rel, u8)>,
    /// For the count only: relations that moved from one function into another, those only a fit of the first
    /// election offered (switch 17), and moves of a relation that had moved before.
    pub moves_total: u64,
    pub moves_first: u64,
    pub moves_again: u64,
    pub move_count: FxHashMap<Rel, u32>,
    /// For the count only: changes to a function whose interneuron already held a connection to an action.
    pub changed_after_action: u64,
    pub changed_after_action_funcs: FxHashSet<FuncId>,
    /// A relation's two end counts, remembered for one grouping step, while no row changes (a speed-up only).
    pub(crate) end_counts_memo: std::cell::RefCell<Option<FxHashMap<Rel, (u32, u32)>>>,
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
    /// Test only: each candidate weighed for closing a cycle, its gains and what they were compared against
    pub trace_closing: bool,
    /// Test only: relations watched (`--watch`), each traced in detail where it is weighed, offered or priced
    pub watch_rels: FxHashSet<Rel>,
    /// For the count only: candidates weighed for closing a cycle, those that joined; under the build, the joins
    /// that the reading as it stands would refuse, attached at the end of the smaller gain (0) or the larger (1),
    /// and of those the largest gain the build read, with the gains read as it stands (smaller, larger)
    pub closing_weighed: u64,
    pub closing_joined: u64,
    pub closing_refused: [u64; 2],
    pub closing_max: Option<(f64, f64, f64)>,
    /// For the count only: closing weighs where the candidate's offset from the call lies beyond every offset the
    /// interneuron's rows hold anything at
    pub closing_beyond: u64,
    /// For the count only: closing weighs where neither end of the candidate is where the call stands
    pub closing_no_end: u64,
    /// For the count only: closing joins that took the relation out of a function holding other relations
    pub closing_from_multi: u64,
    /// For the count only: joins of a relation no function held that were refused because the gain was not above zero, and the lowest gain
    /// it refused
    pub help_refused: u64,
    pub help_refused_min: Option<f64>,
    /// Test only (`join_log`): every join (frame, relation, function, the function before it), every join
    /// refused that way (frame, relation, function, the function, gain, what it was compared against with the
    /// flag off, the count at each end, the node it would attach at, the count there), and each function's calls by
    /// frame
    pub join_log: bool,
    pub joins_seen: Vec<(Frame, String, FuncId, String)>,
    pub refused_seen: Vec<(Frame, String, FuncId, String, f64, f64, u32, u32, String, u32)>,
    pub func_calls: FxHashMap<FuncId, Vec<Frame>>,
    /// For the count only: closing weighs where one lap on cannot be worked out, the lap having a part in space
    pub closing_no_lap: u64,
    /// Test only, with `trace_closing`: each closing candidate weighed, with the function it was weighed for, that
    /// function's relations then, the frame of its first weigh, and at each later call of the motif whose reading is
    /// complete, the frame and what happened: the candidate beside the call or not, and if beside, weighed, offered
    /// but not weighed, or not offered and why
    pub closing_follow: Vec<(Rel, FuncId, Vec<Rel>, Frame, Vec<(Frame, String)>)>,
    pub closing_offered_now: FxHashSet<(Rel, FuncId)>,
    pub closing_weighed_now: FxHashSet<(Rel, FuncId)>,
    pub closing_follow_at: Frame,
    /// For the count only: each function closed into a cycle by a relation, the frame it closed in, the closing
    /// relation's strength and the strength of each relation it already held; and for each function that is a cycle,
    /// its calls on less than one lap, on one lap or more, and the most laps any stood on, a call counted at its
    /// latest record (a call that took a lap is the same call); the records counted so far, with their laps
    pub closed_cycles: Vec<(FuncId, Frame, f64, Vec<f64>)>,
    pub cycle_laps: FxHashMap<FuncId, (u64, u64, u32)>,
    pub(crate) cycle_counted: FxHashMap<ActRef, (FuncId, u32)>,
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
    /// Per call in its phase, every node its expansion puts back, at any depth, each at its offset from the call's
    /// position (R28, R50); read from the dictionary and dropped whenever the dictionary changes.
    pub(crate) expansions: std::cell::RefCell<FxHashMap<Sym, std::rc::Rc<Vec<(Sym, Offset, Vec<NeuronId>)>>>>,
    /// Per expansion kept, the functions it was read through: a change to one drops the expansions that used it.
    pub(crate) expansion_deps: std::cell::RefCell<FxHashMap<Sym, Vec<crate::function::FuncId>>>,
    /// Within one grouping, the histories stand still: per relation, the end it is read from and its strength,
    /// worked out once and dropped whenever a function changes.
    pub(crate) held_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), Option<(Sym, Sym, Offset)>>>,
    pub(crate) strength_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), u32>>,
    pub(crate) returned_memo: std::cell::RefCell<FxHashMap<(crate::function::Rel, Option<NeuronId>), u32>>,
    /// A test: the machine runs each level as it was trained to, the frame first and time after it, its elections
    /// covering, lifting, refusing and blocking as each fit is taken, its calls firing level after level, and infers
    /// and resolves as usual; but it changes nothing of itself. No rows enter a history and the relation index stays
    /// as it is, no function starts, joins, moves or leaves, no interneuron is allocated (so a fit of a function with
    /// no living interneuron is not elected), no connection or reward is written, nothing is decommissioned.
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
            frozen_last_func: FxHashMap::default(),
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
            trace_cover: false,
            last_heard: Vec::new(), trace_vote: false,
            joined: FxHashMap::default(), jparent: FxHashMap::default(), joined_at: None, refused: FxHashSet::default(), form_src: FxHashMap::default(), form_pre: FxHashSet::default(),
            cover_landed: Vec::new(),
            cycle_calls: FxHashMap::default(), print_cycles: false, start_queue: Vec::new(), half: None, last_start_rounds: 0, joined_stash: FxHashMap::default(), bare_after_l0: Vec::new(), joins_by: [0; 2], moves_by: [0; 2],
            grown: FxHashMap::default(),
            cover_log: Vec::new(),
            lifted: FxHashSet::default(),
            cov_inc: FxHashMap::default(),
            lift_forms: FxHashMap::default(),
            kind_of: FxHashMap::default(),
            taken: FxHashMap::default(),
            up_count: 0,
            up_pair_count: 0,
            cover_age0: 0,
            first_frame: None,
            one_side: Vec::new(),
            middle_seen: FxHashSet::default(),
            moves_total: 0,
            moves_first: 0,
            moves_again: 0,
            move_count: FxHashMap::default(),
            changed_after_action: 0,
            changed_after_action_funcs: FxHashSet::default(),
            end_counts_memo: std::cell::RefCell::new(None),
            answer_memo: std::cell::RefCell::new(None),
            share_memo: std::cell::RefCell::new(None),
            last_stats: FrameStats::default(),
            last_winners: Vec::new(),
            tie_join: false,
            trace_joins: true,
            trace_closing: false, watch_rels: FxHashSet::default(), closing_weighed: 0, closing_joined: 0, closing_refused: [0; 2], closing_max: None, closing_beyond: 0, closing_no_end: 0, closing_from_multi: 0, help_refused: 0, help_refused_min: None, join_log: false, joins_seen: Vec::new(), refused_seen: Vec::new(), func_calls: FxHashMap::default(), closing_no_lap: 0, closing_follow: Vec::new(), closing_offered_now: FxHashSet::default(), closing_weighed_now: FxHashSet::default(), closing_follow_at: Frame::MIN,
            closed_cycles: Vec::new(), cycle_laps: FxHashMap::default(), cycle_counted: FxHashMap::default(),
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
        // what a frozen machine named or recorded for its own calls goes once it learns again (`last_func`, poses);
        // the expansions kept may have been read through those poses
        self.dict.frozen = self.frozen;
        if !self.frozen {
            self.frozen_last_func.clear();
            if self.dict.drop_frozen_syms() { self.forget_all_expansions(); }
        }
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
        self.cover_log.clear();
        self.stats_added = 0;
        self.stats_reused = 0;
        self.stats_retired = 0;
        if self.first_frame.is_none() { self.first_frame = Some(fno); }
        let mut level: u32 = 0;
        if self.recognize_first {
            self.frame_up_levels(&mut stats);
            level = u32::MAX;
        }
        loop {
            if level == u32::MAX { break; }
            let refs: Vec<ActRef> = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate()
                .filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone).map(|(i, _)| ActRef { frame: fno, index: i as u32 }).collect::<Vec<_>>()).collect();
            if refs.is_empty() { break; }
            
            stats.levels += 1;
            let lt = std::time::Instant::now();
            let added_before = self.stats_added;
            let (bids, accepted, minted) = self.run_level(level, &refs);
            { let (b, c) = self.cover_pass(); self.cover_log.push((level as i64, b, c)); }
            if level == 0 {
                let thal = &self.thalamus;
                self.bare_after_l0 = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter()).filter(|a| a.level == 0 && a.kind == Kind::Event && !a.gone && a.covered_at.is_none()).map(|a| (a.pos[1], a.pos[2], thal[a.neuron].bucket)).collect();
            }
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
        // and once more after the frame's last level (D24)
        { let (b, c) = self.cover_pass(); self.cover_log.push((-1, b, c)); }
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
    pub(crate) fn board_span(&self) -> i32 {
        self.dict.funcs.values().map(|f| f.reach(1)[0]).max().unwrap_or(1).max(1)
    }

    /// The cover check after a level (D24): each activation a call has landed on whose covered relations of one kind
    /// reach that kind's count (`cover_count`) is covered. From then it is out of the graph; the rows that saw it keep
    /// it as they saw it and take nothing for its calls (D10, D18).
    pub(crate) fn cover_pass(&mut self) -> (usize, usize) {
        let fno = self.frame_no;
        // covered: its covered relations of one kind reach that kind's count (D24)
        let mut newly: Vec<(ActRef, u32)> = Vec::new();
        let mut one: Vec<(Frame, ActRef)> = Vec::new();
        for f in self.frames.iter() {
            for (i, a) in f.acts.iter().enumerate() {
                if a.gone || a.covered_at.is_some() || a.coverers.is_empty() || a.neuron == NeuronId::MAX { continue; }
                let me = ActRef { frame: f.no, index: i as u32 };
                if let Some(edge_only) = self.cover_count(me) {
                    if edge_only { one.push((fno, me)); }
                    newly.push((me, (fno - a.pos[0] as i64).max(0) as u32));
                }
            }
        }
        self.one_side.extend(one);
        let counts = (newly.iter().filter(|(r, _)| self.act(*r).map_or(false, |a| a.covers.is_empty())).count(),
            newly.iter().filter(|(r, _)| self.act(*r).map_or(false, |a| !a.covers.is_empty())).count());
        for (me, age) in newly {
            if self.trace_cover { eprintln!("[covered f{}] {} at frame {} age {}", fno, self.act(me).map_or("?".into(), |a| self.dict.sym_text(a.sym)), me.frame, age); }
            if let Some(a) = self.act(me) { if a.covers.is_empty() && a.kind == Kind::Event { self.cover_landed.push((me.frame, a.coverers.len())); } }
            if age == 0 { self.cover_age0 += 1; }
            // a row holds what stood on the board when it was written: it takes nothing for the call (D10, D18)
            if let Some(a) = self.act_mut(me) { a.covered_at = Some(age); }
            self.lift_relations(me);
        }
        counts
    }

    /// Covering inside the election (D28, D24, R23): of the activations a fit just taken landed on, each whose covered
    /// relations of one kind now reach that kind's count (`cover_count`) is covered from this round: off the board for the rounds after,
    /// each relation of its own that no call covers lifted to the calls that landed on it. Returns what it covered.
    pub(crate) fn cover_in_election(&mut self, cands: &[ActRef]) -> Vec<ActRef> {
        let fno = self.frame_no;
        let mut seen: FxHashSet<ActRef> = FxHashSet::default();
        let mut newly: Vec<(ActRef, u32)> = Vec::new();
        for &me in cands {
            if !seen.insert(me) { continue; }
            let Some(a) = self.act(me) else { continue };
            if a.gone || a.covered_at.is_some() || a.coverers.is_empty() || a.neuron == NeuronId::MAX { continue; }
            let age = (fno - a.pos[0] as i64).max(0) as u32;
            if let Some(one) = self.cover_count(me) {
                if one { self.one_side.push((fno, me)); }
                newly.push((me, age));
            }
        }
        let mut out = Vec::new();
        for (me, age) in newly {
            if self.trace_cover { eprintln!("[covered in the election f{}] {} at frame {} age {}", fno, self.act(me).map_or("?".into(), |a| self.dict.sym_text(a.sym)), me.frame, age); }
            if let Some(a) = self.act(me) { if a.covers.is_empty() && a.kind == Kind::Event { self.cover_landed.push((me.frame, a.coverers.len())); } }
            if age == 0 { self.cover_age0 += 1; }
            if let Some(a) = self.act_mut(me) { a.covered_at = Some(age); }
            self.lift_relations(me);
            out.push(me);
        }
        out
    }

    /// The covered relations of an activation, one per other end, at their offsets as the board relations have them.
    fn covered_offs(&self, me: ActRef) -> Vec<(ActRef, Offset)> {
        let mut out: Vec<(ActRef, Offset)> = Vec::new();
        let Some(a) = self.act(me) else { return out };
        let Some(others) = self.cov_inc.get(&me) else { return out };
        for &x in others {
            if out.iter().any(|(o, _)| *o == x) { continue; }
            let Some(xa) = self.act(x) else { continue };
            if xa.gone { continue; }
            if let Some(br) = a.rels.iter().find(|br| br.other == x) { out.push((x, br.off)); continue; }
            if let Some(br) = xa.rels.iter().find(|br| br.other == me) { let mut n = br.off; for v in n.iter_mut() { *v = -*v; } out.push((x, n)); }
        }
        out
    }

    /// Covering counts the two kinds of relation apart (D24, D34). Inside the frame (its offset in time zero, in space
    /// or at one position): one covered relation, so a call inside the frame covers every activation it landed on, at
    /// a relation it covered, as it is taken. In time: one covered relation to an earlier frame and one to a later,
    /// one to a later at the run's first frame. A relation of each kind never make a count together. Returns whether
    /// it was covered by one relation alone.
    pub(crate) fn cover_count(&self, me: ActRef) -> Option<bool> {
        let rels = self.covered_offs(me);
        let inside = rels.iter().filter(|(_, o)| o[0] == 0).count();
        if inside >= 1 { return Some(inside == 1); }
        let time: Vec<Offset> = rels.iter().filter(|(_, o)| o[0] != 0).map(|x| x.1).collect();
        let (hn, hp) = (time.iter().any(|x| x[0] < 0), time.iter().any(|x| x[0] > 0));
        if (hn || hp) && (hn || !self.layout_side(me, 0, -1)) && (hp || !self.layout_side(me, 0, 1)) { return Some(!(hn && hp)); }
        None
    }

    /// A relation of a covered activation that no call covers is lifted, from then on, to each call that landed on it,
    /// at the offset from that call's position; where its other end is covered too, it is the relation between the
    /// calls on the two, one edge however many such relations run between them (D24, D10, D26).
    fn lift_relations(&mut self, me: ActRef) {
        let Some(a) = self.act(me).cloned() else { return };
        let mut rels: Vec<(ActRef, u32)> = a.rels.iter().map(|br| (br.other, br.strength)).collect();
        for f in self.frames.iter() { for (j, b) in f.acts.iter().enumerate() { if b.gone { continue; } for br in &b.rels { if br.other == me { rels.push((ActRef { frame: f.no, index: j as u32 }, br.strength)); } } } }
        let coverers: Vec<ActRef> = a.coverers.iter().map(|c| self.latest(*c)).collect();
        for (o, s) in rels {
            let e = edge(me, o);
            // covered by a call as it stands, or in any relation it was lifted to (D55, D24)
            if self.covered_in_any_form(e, 0) { continue; }
            // a refused relation is never lifted, at any level (D24)
            if self.refused.contains(&e) { continue; }
            let Some(oa) = self.act(o).cloned() else { continue };
            if oa.gone || coverers.contains(&o) { continue; }
            // lifted already when its other end was covered: now that both ends are, it is lifted to the relation
            // between the calls on the two, one relation per pair of calls, whichever end was covered first (D26)
            if self.lifted.contains(&e) && oa.covered_at.is_none() { continue; }
            self.lifted.insert(e);
            self.up_count += 1;
            // where the other end is covered, it is lifted to the relation between the calls on the two
            let others: Vec<ActRef> = if oa.covered_at.is_some() { oa.coverers.iter().map(|c| self.latest(*c)).collect() } else { vec![o] };
            if oa.covered_at.is_some() { self.up_pair_count += 1; }
            for &c in &coverers {
                for &x in &others {
                    if x == c { continue; }
                    let (Some(ca), Some(xa)) = (self.act(c).cloned(), self.act(x).cloned()) else { continue };
                    let v = self.taken.entry(c).or_default();
                    let in_taken = v.contains(&x);
                    if !in_taken { v.push(x); }
                    let off = offset_between(&xa, &ca);
                    { let v = self.lift_forms.entry(e).or_default(); let f = edge(x, c); if !v.contains(&f) { v.push(f); } }
                    let held = self.act(c).map_or(false, |cm| cm.rels.iter().any(|br| br.other == x));
                    // what the edge stands for, and whether it stood on the board before (D24)
                    {
                        let f = edge(x, c);
                        let first = !self.form_src.contains_key(&f);
                        if first && (in_taken || held || xa.rels.iter().any(|br| br.other == c)) { self.form_pre.insert(f); }
                        let v = self.form_src.entry(f).or_default(); if !v.contains(&e) { v.push(e); }
                    }
                    if let Some(xm) = self.act_mut(x) { if !held && !xm.rels.iter().any(|br| br.other == c) { xm.rels.push(BoardRel { other: c, off, strength: s }); } }
                    if self.trace_cover { eprintln!("[lifted f{}] {} at frame {} -> {} lifted to {} | {}", self.frame_no, self.dict.sym_text(a.sym), me.frame, self.dict.sym_text(oa.sym), self.dict.sym_text(ca.sym), if oa.covered_at.is_some() { "between two calls" } else { "to the call" }); }
                }
            }
        }
    }

    /// Whether a call covers a relation as it stands or in any relation it was lifted to (D55, D24).
    fn covered_in_any_form(&self, e: (ActRef, ActRef), depth: usize) -> bool {
        if self.covered_rel.contains(&e) { return true; }
        if depth > 64 { return false; }
        self.lift_forms.get(&e).map_or(false, |v| v.iter().any(|f| self.covered_in_any_form(*f, depth + 1)))
    }

    /// Whether the layout has a point on one side of an activation in a dimension, within its reach, among what it
    /// may see: in time, nothing ends after it (D24, D3, D9).
    pub(crate) fn layout_side(&self, me: ActRef, d: usize, sign: i32) -> bool {
        let Some(a) = self.act(me) else { return false };
        if d == 0 && sign > 0 { return true; }
        // in time, a position where something has fired is a point of the layout for good: the layout ends before an
        // activation only at the real start of the run, where nothing has fired before it (D24, D3)
        if d == 0 { return self.first_frame.map_or(false, |f0| (a.lo[0] as i64) > f0); }
        let under = self.under(a);
        for f in self.frames.iter() {
            let dt = f.no - a.pos[0] as i64;
            if dt > 0 || -dt > a.reach[0] as i64 { continue; }
            for (i, x) in f.acts.iter().enumerate() {
                let xr = ActRef { frame: f.no, index: i as u32 };
                if xr == me || x.gone || under.contains(&xr) || x.neuron == NeuronId::MAX { continue; }
                if !self.cfg.policy.past[a.kind as usize][x.kind as usize] { continue; }
                let off = offset_between(a, x);
                let mut within = true;
                for k in 1..MAX_DIMS { if (a.axes & x.axes) & (1 << k) != 0 && off[k].abs() > a.reach[k] { within = false; } }
                if within && off[d].signum() == sign { return true; }
            }
        }
        false
    }

    /// The record a call stands as now, after every lap it took (D53).
    pub(crate) fn latest(&self, mut r: ActRef) -> ActRef {
        let mut n = 0;
        while let Some(x) = self.grown.get(&r) { r = *x; n += 1; if n > 4096 { break; } }
        r
    }

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
    /// The frame runs up all its levels before time runs at any (R26, R24). The climb: from level 0 up,
    /// each level's activations are activated and its frame half runs, until a level takes and starts nothing. Then
    /// from level 0 up each level as built: what is new at it since the climb is activated, its frame half runs over
    /// the relations only that brought, and its time half over every relation of the level.
    fn frame_up_levels(&mut self, stats: &mut FrameStats) {
        let fno = self.frame_no;
        let mut activated: FxHashSet<ActRef> = FxHashSet::default();
        let mut fresh_of: Vec<Vec<Fresh>> = Vec::new();
        // per level, what the climb did there: relations, fits, accepted, functions added, ms
        let mut climb: Vec<(usize, usize, usize, usize, f64)> = Vec::new();
        let at_level = |m: &Machine, level: u32| -> Vec<ActRef> { m.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter().enumerate()
            .filter(|(_, a)| a.opened_at == fno && a.level == level && !a.gone).map(|(i, _)| ActRef { frame: fno, index: i as u32 }).collect::<Vec<_>>()).collect() };
        let fired_of = |m: &Machine, refs: &[ActRef]| -> FxHashSet<Sym> { refs.iter().filter_map(|r| m.act(*r)).map(|a| a.sym).collect() };
        // the climb
        let mut level: u32 = 0;
        loop {
            let refs: Vec<ActRef> = at_level(self, level).into_iter().filter(|r| !activated.contains(r)).collect();
            if refs.is_empty() { break; }
            let lt = std::time::Instant::now();
            let added_before = self.stats_added;
            activated.extend(refs.iter().copied());
            let fresh = self.activate_level(level, &refs);
            while fresh_of.len() <= level as usize { fresh_of.push(Vec::new()); }
            fresh_of[level as usize] = fresh.clone();
            let fired = fired_of(self, &at_level(self, level));
            self.last_fits = 0;
            if self.trace { eprintln!("[climb f{} l{}] the frame half", fno, level); }
            let (a, m, started) = if self.frozen { self.frozen_half(level, true, &fresh) } else { self.run_half(level, true, &fresh, &fired) };
            { let (b, c) = self.cover_pass(); self.cover_log.push((level as i64, b, c)); }
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
            let new_fresh = if new.is_empty() { Vec::new() } else { self.activate_level(level, &new) };
            while fresh_of.len() <= level as usize { fresh_of.push(Vec::new()); }
            fresh_of[level as usize].extend(new_fresh.iter().copied());
            let all_fresh = fresh_of[level as usize].clone();
            let fired = fired_of(self, &all);
            self.last_fits = 0;
            if self.trace { eprintln!("[levels f{} l{}] the frame half over {} new relations, then time", fno, level, new_fresh.len()); }
            let (a1, m1, _) = if self.frozen { self.frozen_half(level, true, &new_fresh) } else { self.run_half(level, true, &new_fresh, &fired) };
            let (a2, m2, _) = if self.frozen { self.frozen_half(level, false, &all_fresh) } else { self.run_half(level, false, &all_fresh, &fired) };
            { let (b, c) = self.cover_pass(); self.cover_log.push((level as i64, b, c)); }
            if level == 0 {
                let thal = &self.thalamus;
                self.bare_after_l0 = self.frames.iter().filter(|f| f.no == fno).flat_map(|f| f.acts.iter()).filter(|a| a.level == 0 && a.kind == Kind::Event && !a.gone && a.covered_at.is_none()).map(|a| (a.pos[1], a.pos[2], thal[a.neuron].bucket)).collect();
            }
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
    fn activate_level(&mut self, level: u32, refs: &[ActRef]) -> Vec<Fresh> {
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
        for (b, n, r) in seen { if let Some(x) = self.act_mut(b) { x.seen_by.push((n, r)); } }
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
        if self.trace { for e in &fresh { eprintln!("[fresh f{} l{}] {}→{}{:?} s={} held {:?}", self.frame_no, level, self.dict.sym_text(e.rel.a), self.dict.sym_text(e.rel.b), &e.rel.off[..], e.strength, self.dict.holder(&e.rel)); } }
        fresh
    }

    fn run_level(&mut self, level: u32, refs: &[ActRef]) -> (usize, usize, usize) {
        let fresh = self.activate_level(level, refs);
        let bids = fresh.len();
        // find, and fit what was found (R53, D31)
        let mut found = self.find(&fresh);
        let mut fits: Vec<_> = found.iter().flat_map(|id| self.fits_of(*id)).collect();
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

    /// One half of a level on a frozen machine: the elections over the fits of the functions that exist, of one kind,
    /// as the build runs them; it changes nothing that lasts: no function starts, joins, moves or leaves, and a fit
    /// of a function with no living interneuron is not in the election, since none is allocated.
    fn frozen_half(&mut self, level: u32, inside: bool, fresh: &[Fresh]) -> (usize, usize, usize) {
        let kind: Vec<Fresh> = fresh.iter().filter(|e| (e.rel.off[0] == 0) == inside).copied().collect();
        let found: Vec<FuncId> = self.find(&kind);
        let mine: Vec<_> = found.iter().filter(|id| self.dict.get(**id).and_then(|f| f.inter).map_or(false, |n| self.thalamus.get(n).map_or(false, |x| x.alive)))
            .flat_map(|id| self.fits_of(*id)).collect();
        if self.trace { eprintln!("[half f{} l{}] {} (frozen) | fits of the functions that exist {}", self.frame_no, level, if inside { "the frame" } else { "time" }, mine.len()); }
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
            let mine: Vec<_> = found.iter().filter(|id| self.dict.get(**id).is_some()).flat_map(|id| self.fits_of(*id)).collect();
            if self.trace { eprintln!("[half f{} l{}] {} | fits of the functions that exist {}", self.frame_no, level, if inside { "the frame" } else { "time" }, mine.len()); }
            self.last_fits += mine.len();
            // grouping reads the fits this election takes, and no other (switch 17)
            self.call_fit.clear();
            let (a1, m1) = self.elect(level, mine);
            accepted += a1;
            minted += m1;
            // what the election left on the board: every relation of this kind no call covers and that is not
            // lifted, its ends standing (switch 15)
            let left: Vec<Fresh> = if self.sw.s15_uncovered_relations {
                kind.iter().filter(|e| !self.covered_rel.contains(&crate::recognize::edge(e.near, e.far)) && !self.lifted.contains(&crate::recognize::edge(e.near, e.far))
                    && self.act(e.near).map_or(false, |a| !a.gone && a.covered_at.is_none() && self.stands(a, self.frame_no)) && self.act(e.far).map_or(false, |a| !a.gone && a.covered_at.is_none() && self.stands(a, self.frame_no))).copied().collect()
            } else {
                kind.iter().filter(|e| self.act(e.near).map_or(false, |a| a.on_apex()) && self.act(e.far).map_or(false, |a| a.on_apex())).copied().collect()
            };
            // count and group (R50, R54, R18): over the relations of this kind and the functions that hold them
            let rest: Vec<_> = found.iter().filter(|id| self.dict.get(**id).is_some()).flat_map(|id| self.fits_of(*id)).collect();
            let board: Vec<Fresh> = self.board_rels().into_iter().filter(|e| (e.rel.off[0] == 0) == inside).collect();
            self.half = Some(inside);
            let changed = self.group(&rest, &left, &board, fired);
            self.half = None;
            self.call_fit.clear();
            self.span_s = self.board_span();
            // elect again (R24): a function that changed in group is fit on what the first election left
            let again: Vec<_> = changed.iter().filter(|id| self.dict.get(**id).is_some()).flat_map(|id| self.fits_of(*id)).collect();
            if self.trace { for f in &again { eprintln!("[fit2 f{} l{}] f{} entry {:?} at node {} landed {} calls {} edges {} failed {:?} laps {}", self.frame_no, level, f.func, f.entry.1, f.entry.0, f.landed.len(), f.calls.len(), f.edges.len(), f.failed, f.laps); } }
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
    pub fn funcs_never_called(&self) -> usize {
        self.dict.funcs.values().filter(|f| f.inter.map_or(true, |n| !self.thalamus.get(n).map_or(false, |x| x.alive))).count()
    }

    /// Start (R50): functions start one at a time, and only from a relation that still stands. A relation stands
    /// where at least one instance of it on the board is covered by no call and is not lifted, refused or blocked,
    /// its ends standing (D24, D55). Of the relations that may start a function and stand, the one with the most
    /// instances standing starts first, and on a tie the one whose standing instance is oldest; the election takes
    /// its fits (R24), and the machine looks again at what stands. Returns what the elections accepted and allocated.
    fn start_one_at_a_time(&mut self, level: u32, queue: &[crate::function::Rel], left: &[Fresh]) -> (usize, usize) {
        let (mut accepted, mut minted) = (0, 0);
        let mut round = 0;
        loop {
            let standing = |m: &Machine, e: &Fresh| -> bool {
                let k = crate::recognize::edge(e.near, e.far);
                if m.covered_rel.contains(&k) || m.lifted.contains(&k) || m.refused.contains(&k) || m.blocked(e.near, e.far) { return false; }
                let up = |r: ActRef| m.act(r).map_or(false, |a| !a.gone && a.covered_at.is_none() && m.stands(a, m.frame_no));
                up(e.near) && up(e.far)
            };
            let mut best: Option<((bool, std::cmp::Reverse<usize>, ActRef), crate::function::Rel)> = None;
            for r in queue {
                if self.dict.holder(r).is_some() { continue; }
                let inst: Vec<&Fresh> = left.iter().filter(|e| e.rel == *r && standing(self, e)).collect();
                if inst.is_empty() { continue; }
                let oldest = inst.iter().map(|e| e.near.min(e.far)).min().unwrap();
                let key = (r.off[0] != 0, std::cmp::Reverse(inst.len()), oldest);
                if best.as_ref().map_or(true, |(k, _)| key < *k) { best = Some((key, *r)); }
            }
            let Some((key, r)) = best else { break };
            round += 1;
            let id = self.dict.start(r);
            self.stats_added += 1;
            if self.trace { eprintln!("[start-one f{} l{}] round {}: started {} | {} | instances standing {}", self.frame_no, level, round, self.dict.describe(self.dict.get(id).unwrap()), if key.0 { "in time" } else { "inside the frame" }, key.1 .0); }
            let fits = self.fits_of(id);
            self.last_fits += fits.len();
            let (a, m) = self.elect(level, fits);
            accepted += a;
            minted += m;
            if self.trace {
                let rest: Vec<String> = queue.iter().filter(|q| self.dict.holder(q).is_none()).filter_map(|q| {
                    let n = left.iter().filter(|e| e.rel == *q && standing(self, e)).count();
                    if n > 0 { Some(format!("{} x{}", self.rel_text(q), n)) } else { None }
                }).collect();
                eprintln!("[start-one f{} l{}] round {}: taken {} | still standing bare: {}", self.frame_no, level, round, a, if rest.is_empty() { "nothing".to_string() } else { rest.join(", ") });
            }
        }
        self.last_start_rounds = round;
        (accepted, minted)
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

    /// Test only: the relations of any function whose text holds `pat` are watched from now on; one line for each
    /// watched relation: where it is held, its count at each end, and each end's neuron's rows.
    pub fn watch_lines(&mut self, pat: &str) -> Vec<String> {
        let found: Vec<Rel> = self.dict.funcs.values().flat_map(|f| f.rels.iter().copied()).filter(|r| self.rel_text(r).contains(pat)).collect();
        for r in found { self.watch_rels.insert(r); }
        let mut v: Vec<Rel> = self.watch_rels.iter().copied().collect();
        v.sort();
        v.into_iter().map(|r| {
            let (ca, cb) = self.end_counts(&r);
            let held = self.dict.holder(&r).and_then(|g| self.dict.get(g)).map_or("in no function".to_string(), |g| format!("in f{} ({} relations)", g.id, g.rels.len()));
            let rows = |s: Sym| { let n = self.dict.neuron_of(s); self.thalamus.get(n).map_or("-".to_string(), |nn| format!("#{} {} of {} rows", n, nn.history.len(), nn.h)) };
            format!("{} | {} | count at {} {} , at {} {} | strength {} | rows {} ; {}", self.rel_text(&r), held, self.dict.sym_text(r.a), ca, self.dict.sym_text(r.b), cb, ca.max(cb), rows(r.a), rows(r.b))
        }).collect()
    }

    /// For the count only: over every interneuron's history, the frames in which it has more than one call, and of
    /// those the frames in which a row holds another call of the same interneuron.
    pub fn twice_counts(&self) -> (u64, u64) {
        let (mut twice, mut seen) = (0u64, 0u64);
        for n in self.thalamus.iter() {
            if n.kind != Kind::Inter { continue; }
            let mut by_frame: FxHashMap<Frame, Vec<&crate::neuron::Row>> = FxHashMap::default();
            for row in n.history.iter() { by_frame.entry(row.act.frame).or_default().push(row); }
            for (_, rows) in by_frame {
                if rows.len() < 2 { continue; }
                twice += 1;
                if rows.iter().any(|r| r.nbs.iter().any(|nb| self.dict.neuron_of(nb.sym) == n.id)) { seen += 1; }
            }
        }
        (twice, seen)
    }

    /// Test only: one line per closing candidate followed (`closing_follow`).
    pub fn closing_follow_lines(&self) -> Vec<String> {
        self.closing_follow.iter().map(|(r, fid, _, first, ev)| {
            let beside = ev.iter().filter(|e| e.1 != "not beside" && e.1 != "motif changed" && e.1 != "motif gone").count();
            let mut counts: std::collections::BTreeMap<&str, u32> = Default::default();
            for e in ev { *counts.entry(e.1.as_str()).or_default() += 1; }
            format!("FOLLOWCLOSE {} to f{} | first weighed at frame {} | later calls read {} | candidate beside the call {} | {} | frames beside: {}", self.rel_text(r), fid, first, ev.len(), beside,
                counts.iter().map(|(k, v)| format!("{} {}", k, v)).collect::<Vec<_>>().join(", "),
                ev.iter().filter(|e| e.1 != "not beside" && e.1 != "motif changed" && e.1 != "motif gone").map(|e| format!("{}:{}", e.0, e.1)).take(12).collect::<Vec<_>>().join(" "))
        }).collect()
    }

    /// For the count only: one line per closing of a function into a cycle, with its calls' laps over the run.
    pub fn cycle_lap_lines(&self) -> Vec<String> {
        self.closed_cycles.iter().map(|(id, fr, cs, held)| {
            let (lt, ge, mx) = self.cycle_laps.get(id).copied().unwrap_or((0, 0, 0));
            let now = self.dict.get(*id).map_or("gone".to_string(), |f| format!("{}{}", self.dict.describe(f), if f.has_cycle() { "" } else { " (no longer a cycle)" }));
            format!("CYCLELAPS f{} | closed at frame {} | closing strength {:.0} | held {:?} | calls under one lap {} | one lap or more {} | most laps {} | now {}", id, fr, cs, held.iter().map(|x| *x as u32).collect::<Vec<_>>(), lt, ge, mx, now)
        }).collect()
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

// ───── the two kinds and the kind decided (D34, D24) ─────
impl Machine {
    /// Whether a relation between two activations is inside the frame: its two ends in one frame (D34).
    pub(crate) fn rel_inside(&self, a: ActRef, b: ActRef) -> Option<bool> { Some(self.act(a)?.pos[0] == self.act(b)?.pos[0]) }

    /// A relation of the other kind than the one an end's first covered relation decided is blocked (D24).
    pub(crate) fn blocked(&self, a: ActRef, b: ActRef) -> bool {
        let Some(k) = self.rel_inside(a, b) else { return false };
        self.kind_of.get(&a).map_or(false, |x| *x != k) || self.kind_of.get(&b).map_or(false, |x| *x != k)
    }

    /// The kind an activation's first covered relation decided, if any: true inside the frame.
    pub fn kind_decided(&self, a: ActRef) -> Option<bool> { self.kind_of.get(&a).copied() }

    /// A call covers a relation between the two (D24): each end whose kind is not yet decided takes the relation's
    /// kind, and its relations of the other kind are blocked from now.
    pub(crate) fn decide_kind(&mut self, a: ActRef, b: ActRef) {
        let Some(k) = self.rel_inside(a, b) else { return };
        for x in [a, b] {
            if self.kind_of.contains_key(&x) { continue; }
            self.kind_of.insert(x, k);
            if self.trace_cover {
                let Some(xa) = self.act(x) else { continue };
                let mut others: Vec<ActRef> = xa.rels.iter().map(|br| br.other).collect();
                for f in self.frames.iter() { for (j, y) in f.acts.iter().enumerate() { if y.gone { continue; } if y.rels.iter().any(|br| br.other == x) { others.push(ActRef { frame: f.no, index: j as u32 }); } } }
                others.sort(); others.dedup();
                let show = |r: &ActRef| format!("{}@{}", self.act(*r).map_or("?".into(), |y| self.dict.sym_text(y.sym)), r.frame);
                let bl: Vec<String> = others.iter().filter(|o| self.rel_inside(x, **o) == Some(!k)).map(|o| show(o)).collect();
                eprintln!("[kind f{} l{}] {} decided {} by {}–{} | blocked {}: {}", self.frame_no, xa.level, show(&x), if k { "inside the frame" } else { "across frames" }, show(&a), show(&b), bl.len(), bl.join(", "));
            }
        }
    }
}


// ───── a deadlock broken where it would close: joined, refused, eliminated (D24, D55) ─────
impl Machine {
    /// The root of an activation's set of activations joined in this election.
    pub(crate) fn jfind(&self, mut x: ActRef) -> ActRef { while let Some(p) = self.jparent.get(&x) { if *p == x { break; } x = *p; } x }



    /// The calls on the shortest chain of relations covered in this election from `a` to `b`, if they are joined.
    pub(crate) fn joined_chain(&self, a: ActRef, b: ActRef) -> Option<Vec<ActRef>> {
        if a == b || self.jfind(a) != self.jfind(b) { return None; }
        let mut prev: FxHashMap<ActRef, (ActRef, ActRef)> = FxHashMap::default();
        let mut q = std::collections::VecDeque::new();
        q.push_back(a);
        prev.insert(a, (a, a));
        while let Some(x) = q.pop_front() {
            if x == b { break; }
            for &(y, call, _) in self.joined.get(&x).map(|v| v.as_slice()).unwrap_or(&[]) {
                if prev.contains_key(&y) { continue; }
                prev.insert(y, (x, call));
                q.push_back(y);
            }
        }
        if !prev.contains_key(&b) { return None; }
        let mut calls: Vec<ActRef> = Vec::new();
        let mut x = b;
        while x != a { let (p, c) = prev[&x]; if !calls.contains(&c) { calls.push(c); } x = p; }
        Some(calls)
    }

    /// A refused relation that had already ended at a call: that edge leaves the board, and every edge it became
    /// above. An edge that also stands for a relation that is not refused, or stood on the board before, stays (D24).
    fn remove_forms(&mut self, e: (ActRef, ActRef), depth: usize) {
        if depth > 64 { return; }
        let forms = self.lift_forms.get(&e).cloned().unwrap_or_default();
        for f in forms {
            if self.refused.contains(&f) { continue; }
            let shared = self.form_pre.contains(&f) || self.form_src.get(&f).map_or(false, |v| v.iter().any(|s| *s != e && !self.refused.contains(s)));
            if shared { continue; }
            self.refused.insert(f);
            let (x, c) = f;
            if let Some(xm) = self.act_mut(x) { xm.rels.retain(|br| br.other != c); }
            if let Some(cm) = self.act_mut(c) { cm.rels.retain(|br| br.other != x); }
            if let Some(v) = self.taken.get_mut(&c) { v.retain(|y| *y != x); }
            if let Some(v) = self.taken.get_mut(&x) { v.retain(|y| *y != c); }
            if self.trace_cover { eprintln!("[form removed f{}] {:?}–{:?}", self.frame_no, x, c); }
            self.remove_forms(f, depth + 1);
        }
    }

    /// At the start of a round of the election: every relation on the board whose two ends the calls taken so far at
    /// this level of this frame join is refused, inside the frame or in time, whether or not a fit offers it or a
    /// function holds it (D24, D55). Returns the ends of the relations refused now.
    pub(crate) fn mark_refused(&mut self, fits: &[crate::recognize::Fit], level: u32) -> Vec<ActRef> {
        let fno = self.frame_no;
        let mut ends = Vec::new();
        let mut cands: Vec<(ActRef, ActRef)> = Vec::new();
        let keys: Vec<ActRef> = self.joined.keys().copied().collect();
        for x in &keys {
            let Some(xa) = self.act(*x) else { continue };
            if xa.gone { continue; }
            for br in &xa.rels { if self.joined.contains_key(&br.other) { cands.push((*x, br.other)); } }
        }
        for fit in fits { cands.extend(fit.edges.iter().copied()); }
        let mut seen: FxHashSet<(ActRef, ActRef)> = FxHashSet::default();
        for (a, b) in cands {
            let e = edge(a, b);
            if !seen.insert(e) { continue; }
            if self.refused.contains(&e) || self.blocked(a, b) || self.covered_rel.contains(&e) { continue; }
            let Some(chain) = self.joined_chain(a, b) else { continue };
            self.refused.insert(e);
            // where it had already ended at a call, that edge leaves the board, with every edge above it
            if self.lifted.contains(&e) { self.remove_forms(e, 0); }
            ends.push(a);
            ends.push(b);
            if self.trace_cover {
                let n = chain.len();
                let held = match (self.act(a), self.act(b)) { (Some(x), Some(y)) => self.dict.holder(&canon(x.sym, y.sym, offset_between(x, y))).is_some(), _ => false };
                let show = |x: &ActRef| format!("{}@{}", self.act(*x).map_or("?".into(), |q| self.dict.sym_text(q.sym)), x.frame);
                let offered: Vec<String> = fits.iter().filter(|f| f.edges.iter().any(|&(p, q)| edge(p, q) == e)).map(|f| format!("f{}", f.func)).collect();
                eprintln!("[refused f{} l{}] {}–{} | a deadlock of {} calls: {} | offered by [{}] | {}", fno, level, show(&a), show(&b), n + 1,
                    chain.iter().map(|c| format!("{} (call of f{})", show(c), self.act(*c).and_then(|x| x.fired_by.as_ref()).map_or(0, |fb| fb.func))).collect::<Vec<_>>().join(", "),
                    offered.join(" "), if held { "held by a function" } else { "held by no function" });
            }
        }
        ends
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        if self.print_cycles {
            let mut v: Vec<(&FuncId, &(String, Vec<Frame>))> = self.cycle_calls.iter().collect();
            v.sort_by(|a, b| a.0.cmp(b.0));
            for (id, (d, frames)) in v {
                let mut runs: Vec<String> = Vec::new();
                let mut i = 0;
                while i < frames.len() { let mut j = i; while j + 1 < frames.len() && frames[j + 1] == frames[j] + 1 { j += 1; } runs.push(if j > i { format!("{}-{}", frames[i], frames[j]) } else { frames[i].to_string() }); i = j + 1; }
                eprintln!("[cycle-calls] f{} | {} | {} frames: {}", id, d, frames.len(), runs.join(","));
            }
        }
    }
}
