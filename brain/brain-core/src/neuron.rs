/// Neuron - Unified struct for all neurons (sensory and pattern)
///
/// All neurons have:
/// - connections: Vec<FxHashMap<NeuronId, ConnectionData>> - predictions (indexed by distance)
/// - routing_table: FxHashMap<PatternId, TemporalRoutingEntry> - child pattern contexts
///
/// All neuron metadata (level, coordinates, channel, type, parent) is stored externally
/// in Thalamus lookup tables. Neurons are pure data processors — they only store learned
/// associations and do pattern matching/voting based on numeric IDs and strengths.
///
/// Note: Active state (which neurons are active at which ages) and votes are managed
/// by the Brain, not stored on neurons. This allows efficient age-indexed queries.
///
/// Lazy Decay: Continuous decay based on frames elapsed since last activation.
/// We store last_activation_frame and compute effective strength on-demand:
/// effective_strength = strength - (frame_number - last_frame) * rate

use rustc_hash::{FxHashMap, FxHashSet};

use crate::context::{SpatialContext, TemporalContext};
use crate::types::*;

/// Minimum samples per (neuron, age) before dynamic modes switch off the warmup
/// fallback. Not a tunable knob — kept here to avoid a magic number in the error-threshold getters.
const ERROR_MIN_SAMPLES: u64 = 3;

/// Who serves a frame in the level-0 configuration loop (UCAR, docs/algorithm.md): the neuron's
/// normal, or one of its children. Also the KEY of the spatial routing table: a routing-table entry
/// is the normal or a child (docs/algorithm.md, "The normal"), so `SpatialServer` names both the
/// winner of the one test and which entry a stored `SpatialRoutingEntry` belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialServer {
    /// The normal — serves the usual neighbourhood, has no pattern neuron, never propagates, is never
    /// deleted. Present as an entry in the routing table like any other.
    Normal,
    /// A child — when it serves it fires and delegates to the pattern one level up.
    Child(NeuronId),
}

/// A neuron's spatial history window, deduplicated by observed neighborhood (docs/algorithm.md, "The
/// history"). The design remembers "the frames it was active for, one horizon back" — but every frame
/// that observed the same neighborhood `O` carries the same observation, is served by the same entry,
/// and at the same distance, so the one test's sums (the add and delete passes) depend only on HOW MANY
/// frames each distinct `O` contributed, never on their identities. So the window is stored as a
/// histogram: one entry per distinct `O`, holding the frames it occurred on (for exact horizon
/// windowing) rather than a record per frame. Decisions are identical to the per-frame window; what
/// changes is that the observation and its cached server/distance are stored once per distinct
/// configuration instead of once per frame, and the add pass scans distinct configurations rather than
/// every remembered frame — the design's expensive step (docs/algorithm.md, "Risks").
#[derive(Debug, Clone, Default)]
struct SpatialHistory {
    /// observed neighborhood (sorted, deduplicated id list) → its evidence in the window.
    configs: FxHashMap<Vec<NeuronId>, ConfigEvidence>,
    /// The configuration recorded on the most recent `record` — the only entry a same-frame
    /// `drop_last_if_frame` may undo (contraction can retract this frame's record after it was written).
    last_key: Option<Vec<NeuronId>>,
}

/// The evidence one distinct neighborhood accumulated over the window: the frames it was observed on,
/// plus the cached server and served distance every per-frame record used to carry.
///
/// The served distance is cached because entries are frozen at their seed/mint configuration: a
/// configuration's distance to its server only changes when the SERVER changes (a mint pulls it onto the
/// newborn, a deletion drops it to its next-best), and both of those already rewrite `server` here — so
/// the cache stays exact with no separate invalidation. It is the FIRST-best, not a second-best cache
/// (the design rejects that, since a deletion promotes the second-best and no third-best was recorded);
/// the observations themselves are still held, so any next-best is recomputed exactly when deletion asks.
#[derive(Debug, Clone)]
struct ConfigEvidence {
    /// Frames within the window on which this exact configuration was observed. Its length is the
    /// multiplicity `n[O]`; aging drops the stale front, a retracted frame pops the tail.
    frames: Vec<FrameNumber>,
    /// The entry that serves this configuration — kept current by reassigning on a mint (configs the
    /// newborn pulls in) and on a deletion (the retired child's configs).
    server: SpatialServer,
    /// Distance from this configuration to `server`'s stored configuration — its current service cost.
    best_distance: u32,
}

impl SpatialHistory {
    /// True when no frame is remembered (every configuration has aged out, or none was ever recorded).
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.configs.values().all(|e| e.frames.is_empty())
    }

    /// Total frames across all configurations — the window's size, what a per-frame record list's
    /// `.len()` reported.
    #[cfg(test)]
    fn total_frames(&self) -> usize {
        self.configs.values().map(|e| e.frames.len()).sum()
    }

    /// The distinct configurations and their evidence — what the add/delete passes and refinement scan.
    fn iter(&self) -> impl Iterator<Item = (&Vec<NeuronId>, &ConfigEvidence)> {
        self.configs.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = (&Vec<NeuronId>, &mut ConfigEvidence)> {
        self.configs.iter_mut()
    }

    fn clear(&mut self) {
        self.configs.clear();
        self.last_key = None;
    }

    /// Append this frame's record. Identical observations fold into one entry with the frame added to
    /// its list; the cached server/distance are refreshed — unchanged for a recurring configuration
    /// (entries are frozen between structural edits), set for a first-seen one.
    fn record(&mut self, frame: FrameNumber, observed: &[NeuronId], server: SpatialServer, best_distance: u32) {
        let key = observed.to_vec();
        let ev = self.configs.entry(key.clone()).or_insert_with(|| ConfigEvidence {
            frames: Vec::new(), server, best_distance,
        });
        ev.frames.push(frame);
        ev.server = server;
        ev.best_distance = best_distance;
        self.last_key = Some(key);
    }

    /// Drop the record written this frame if it is still the tail — contraction retracts a frame whose
    /// neuron was subsumed by a neighbour (`drop_inhibited_spatial_frame`). Only the most recent `record`
    /// can be undone, and only if its frame matches.
    fn drop_last_if_frame(&mut self, frame: FrameNumber) {
        let Some(key) = self.last_key.take() else { return };
        if let Some(ev) = self.configs.get_mut(&key) {
            if ev.frames.last() == Some(&frame) {
                ev.frames.pop();
                if ev.frames.is_empty() { self.configs.remove(&key); }
            }
        }
    }

    /// Drop frames older than the cutoff; a configuration with no frames left leaves the window.
    fn age(&mut self, cutoff: FrameNumber) {
        self.configs.retain(|_, ev| {
            ev.frames.retain(|&f| f >= cutoff);
            !ev.frames.is_empty()
        });
    }

    /// Shift every remembered frame by `-delta` — the restore rebasing (materialize_and_reset_children).
    fn rebase(&mut self, delta: FrameNumber) {
        for ev in self.configs.values_mut() {
            for f in &mut ev.frames { *f -= delta; }
        }
    }
}

/// Sentinel for the distance cache: an uncomputed matrix cell, and an entry whose configuration has not
/// been interned yet. A real spatial distance over a finite neighbourhood is tiny, so `u32::MAX` can
/// never collide with one.
const DIST_SENTINEL: u32 = u32::MAX;

/// Per-neuron memoisation of [Neuron::spatial_distance] between neighbourhood configurations.
///
/// Since refinement was removed, every stored configuration (the normal, every child, every remembered
/// observation) is **frozen** for its lifetime — so the distance between any two configurations is a
/// constant. The route, delete and add passes recompute the same `d(O, config)` on every active frame
/// (a background neuron sees the same `O` and the same entries for thousands of consecutive frames), and
/// the delete pass alone is `O(D · E)` distance calls per frame. Caching turns all of that into O(1)
/// lookups after each distinct pair is seen once, with **no invalidation** needed for correctness: a
/// cell only goes stale if a configuration changes, and the one place that happens — a child's context
/// shrinking when a member is purged — resets the entry's interned id (so it re-interns as a new config).
///
/// Configurations are interned to dense per-neuron ids; distances live in a growable lower-triangular
/// matrix indexed by those ids. Bitmap/popcount packing is deliberately avoided so this works unchanged
/// at any bucket count, not just binary (docs/algorithm.md, "Open questions": configuration space beyond
/// the small case). The interned set is bounded by the distinct configurations the neuron actually sees
/// in its horizon (≤ 256 at radius 1 binary), so the matrix stays small.
#[derive(Debug, Clone, Default)]
struct SpatialDistanceCache {
    /// configuration (sorted, deduplicated id list) → its dense id.
    ids: FxHashMap<Vec<NeuronId>, u32>,
    /// dense id → the configuration, for computing a distance on a cache miss.
    configs: Vec<Vec<NeuronId>>,
    /// Lower-triangular distance matrix: `dist[hi][lo]` (hi ≥ lo) is `d(configs[hi], configs[lo])`,
    /// `DIST_SENTINEL` until first computed. Row `i` has length `i + 1`; the diagonal is 0.
    dist: Vec<Vec<u32>>,
}

impl SpatialDistanceCache {
    /// Intern a configuration to its dense id, assigning a fresh one (and a new matrix row) on first sight.
    fn intern(&mut self, config: &[NeuronId]) -> u32 {
        if let Some(&id) = self.ids.get(config) { return id; }
        let id = self.configs.len() as u32;
        self.ids.insert(config.to_vec(), id);
        self.configs.push(config.to_vec());
        let mut row = vec![DIST_SENTINEL; id as usize + 1];
        row[id as usize] = 0; // a configuration is at distance 0 from itself
        self.dist.push(row);
        id
    }

    /// The cached [Neuron::spatial_distance] between two interned configurations, computed once and
    /// stored. In debug builds a cache hit is re-verified against a fresh merge, so any missed
    /// invalidation (a config that changed without its id being reset) trips the assert immediately.
    fn distance(&mut self, a: u32, b: u32) -> u32 {
        let (lo, hi) = if a <= b { (a as usize, b as usize) } else { (b as usize, a as usize) };
        let cached = self.dist[hi][lo];
        if cached != DIST_SENTINEL {
            debug_assert_eq!(
                cached, Neuron::spatial_distance(&self.configs[hi], &self.configs[lo]),
                "spatial distance cache stale — a cached configuration changed without resetting its id",
            );
            return cached;
        }
        let d = Neuron::spatial_distance(&self.configs[hi], &self.configs[lo]);
        self.dist[hi][lo] = d;
        d
    }
}

/// A contraction bid (docs/algorithm.md, "Contraction: what a firing neuron offers"). A neuron that
/// served from a child offers to describe a patch of this level: its child names a whole
/// neighborhood, so the units it covers correctly can be recovered by expanding that one unit above.
/// The bid names exactly those correctly-covered active neurons — the bidder itself plus the neighbors
/// its child's configuration names that are actually present (`O ∩ config ∪ {self}`) — so the
/// thalamus's election can weigh this cover against its competitors. A neuron serving from its normal
/// makes no bid.
#[derive(Debug, Clone)]
pub struct SpatialBid {
    /// The bidding neuron (this level). It is covered by its own bid, so it appears in `covered`.
    pub bidder_id: NeuronId,
    /// The child pattern one level up that propagates as a single unit if this bid wins the election.
    pub pattern_id: NeuronId,
    /// The active neurons this bid covers correctly, sorted and deduplicated — the members whose role
    /// the promoted unit represents. Their level-below adjacency derives the unit's neighbors above.
    pub covered: Vec<NeuronId>,
}

/// Sentinel `pattern_id` on a [SpatialBid] that marks it as a request to create a **new** child (an
/// error correction) rather than to re-fire an existing one. An error correction is a child-activation
/// request just like recognition, so it competes in the same contraction election; the child is created
/// only if the request wins. `u64::MAX` never collides with a real allocated id, and being the largest
/// possible id it loses every election tie-break to an established child — the "consolidate onto older
/// patterns" rule (docs/algorithm.md, "Election": oldest id wins ties).
pub const NEW_CHILD_BID: NeuronId = NeuronId::MAX;

/// Per-neuron output of the spatial frame pass. Votes never leave the neuron — it routes its
/// neighborhood to the closest entry, then reconsiders its structure: a birth request from the add
/// pass and/or children retired by the delete pass, both decided by the one test over the history.
pub struct SpatialFrameResult {
    /// This neuron's single child-activation request for the election, or None for a pure normal-serve.
    /// A real `pattern_id` recognizes an existing child; `NEW_CHILD_BID` requests a new child at `O`.
    /// The request is a pure decision — nothing is recorded/mutated for it here; [Neuron::commit_spatial_frame]
    /// does that, and only if it wins. A pure normal-serve is finalized inline (no round-trip).
    pub bid: Option<SpatialBid>,
    /// The observation `O`, kept for the commit's record and (for a new-child winner) the child's config.
    pub observed: Vec<NeuronId>,
    /// Distance `O`→selected entry, for the commit's record.
    pub served_distance: u32,
    /// The child the delete pass would retire — applied by the commit only if the request wins. For a
    /// pure normal-serve it is applied inline and reported here for cross-neuron cleanup (bid is None).
    pub delete_candidate: Option<NeuronId>,
    pub timings: NeuronOpTimings,
}

/// Entry in the spatial routing table for a child spatial-correction pattern.
/// Spatial context has no distance dimension.
#[derive(Debug, Clone)]
pub struct SpatialRoutingEntry {
    pub context: SpatialContext,
    /// Decaying activation trace. Doubles as the candidacy gauge (a dead pattern stops being a
    /// recognition candidate) and, read through the same lazy decay, as the recognition trial
    /// count: with per-fire strengthening of common context entries, entry strength / this value
    /// approximates p(entry | pattern) — the likelihood model for likelihood-ratio recognition.
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
    /// This entry's configuration interned into the neuron's [SpatialDistanceCache], so the hot passes
    /// read its id without re-sorting/re-hashing the context every frame. `DIST_SENTINEL` means
    /// "not interned yet" — set lazily by `spatial_entry_ids`, and reset to it whenever the context is
    /// mutated (a purge shrinking the config) so the id re-derives against the new configuration.
    /// Not serialized: it is a pure cache, rebuilt on the first frame after a restore.
    pub config_id: u32,
}

/// Entry in the temporal routing table for a child temporal-correction pattern.
#[derive(Debug, Clone)]
pub struct TemporalRoutingEntry {
    pub context: TemporalContext,
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
}

/// A single vote cast by a neuron at a specific age.
#[derive(Debug, Clone)]
pub struct Vote {
    pub neuron_id: NeuronId,
    pub strength: Strength,
    pub reward: Reward,
    pub distance: Distance,
}

/// A recognition match result for a specific age.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub activate: bool,
    pub death_frame: Option<FrameNumber>,
}

/// A TEMPORAL context reference update.
/// `neuron_id` is the context neuron being referenced.
/// `parent_id` is the parent neuron that owns the temporal routing table referencing it.
/// `parent_id` is set to 0 when emitted from neuron-level code and filled in by thalamus during
/// `collect_context_ref_updates`.
#[derive(Debug, Clone)]
pub struct TemporalContextRefUpdate {
    pub neuron_id: NeuronId,
    pub distance: Distance,
    pub parent_id: NeuronId,
}

/// A SPATIAL context reference update — no distance dimension.
/// `neuron_id` is the context neuron being referenced (the target whose spatial_context_refs
/// gains an entry pointing at `parent_id`). Symmetric in shape to TemporalContextRefUpdate so
/// both dispatch chains bucket-by-target at the same boundary.
#[derive(Debug, Clone)]
pub struct SpatialContextRefUpdate {
    pub neuron_id: NeuronId,
    pub parent_id: NeuronId,
}

/// A context reference entry (neuron_id + distance).
#[derive(Debug, Clone)]
pub struct ContextRefEntry {
    pub neuron_id: NeuronId,
    pub distance: Distance,
}

/// An error-correction activation result.
#[derive(Debug, Clone)]
pub struct CorrectionActivation {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub death_frame: Option<FrameNumber>,
}

/// Per-age vote output from generate_votes.
#[derive(Debug, Clone)]
pub struct AgeVotes {
    pub age: Distance,
    pub votes: Vec<Vote>,
    pub context: Vec<ContextRefEntry>,
    pub threshold: f64,
}

/// Age state passed into process_frame (per-age activation info).
#[derive(Debug, Clone, Default)]
pub struct AgeState {
    pub activated_pattern_id: Option<NeuronId>,
}

/// Active neuron info for the co-activation neighborhood: its id, channel, and reward. (The old
/// per-target dimension — used by the removed prediction/vote substrate — is gone; the configuration
/// loop keys on neuron ids alone.)
#[derive(Debug, Clone)]
pub struct ActiveNeuron {
    pub id: NeuronId,
    pub channel_id: ChannelId,
    pub reward: Reward,
}

/// Error feedback from prior-frame votes.
#[derive(Debug, Clone)]
pub struct ErrorFeedback {
    pub age: Distance,
    pub error_rate: f64,
}

/// Error-correction pattern to install as a child.
#[derive(Debug, Clone)]
pub struct Correction {
    pub pattern_id: NeuronId,
    pub age: Distance,
    pub context_entries: Vec<ContextRefEntry>,
}

/// One active neuron to run through the temporal level pass — the dispatch subject.
#[derive(Debug, Clone)]
pub struct TemporalNeuron {
    pub neuron_id: NeuronId,
    /// What the neuron/window fired at each recency slot — recognition and voting read this every frame.
    pub age_states: FxHashMap<Distance, AgeState>,
    /// Some only on a learning pass; None on a frozen eval, so no learning work is carried or done.
    pub learning_work: Option<TemporalLearningWork>,
}

/// One neuron's learning-only work for this frame — recomputed each frame, NOT the static `self.learning` flag.
#[derive(Debug, Clone)]
pub struct TemporalLearningWork {
    /// Neighbor actives to learn connections from.
    pub neighbors: Vec<ActiveNeuron>,
    /// Error-correction patterns to install as children.
    pub corrections: Vec<Correction>,
    /// Last frame's accuracy samples to fold into the per-age stats.
    pub error_feedback: Vec<ErrorFeedback>,
}

/// Results from process_frame.
#[derive(Debug, Clone)]
pub struct ProcessFrameResult {
    pub matches: Vec<PatternMatch>,
    pub correction_activations: Vec<CorrectionActivation>,
    pub context_ref_updates: Vec<TemporalContextRefUpdate>,
    pub votes: Vec<AgeVotes>,
    pub timings: NeuronOpTimings,
}

/**
 * Per-neuron wall-clock for the 4 main ops inside process_frame, plus
 * recognize sub-buckets. Aggregated up through column → brain into the
 * frame timings. `recognize_candidates_evaluated` counts how many child
 * patterns were scored (post-index) so we can see search-space size per
 * frame.
 */
#[derive(Debug, Clone, Copy, Default)]
pub struct NeuronOpTimings {
    pub learn_connections: f64,
    pub recognize_patterns: f64,
    pub correct_errors: f64,
    pub generate_votes: f64,
    pub recognize_candidate_search: f64,
    pub recognize_candidate_eval: f64,
    pub recognize_candidates_evaluated: u64,
}

impl NeuronOpTimings {
    pub fn add(&mut self, other: &NeuronOpTimings) {
        self.learn_connections  += other.learn_connections;
        self.recognize_patterns += other.recognize_patterns;
        self.correct_errors     += other.correct_errors;
        self.generate_votes     += other.generate_votes;
        self.recognize_candidate_search   += other.recognize_candidate_search;
        self.recognize_candidate_eval     += other.recognize_candidate_eval;
        self.recognize_candidates_evaluated += other.recognize_candidates_evaluated;
    }
}

/// Results from recognize_patterns.
struct RecognizeResult {
    matches: Vec<PatternMatch>,
    context_ref_updates: Vec<TemporalContextRefUpdate>,
}

/// Results from correct_errors.
struct CorrectResult {
    correction_activations: Vec<CorrectionActivation>,
    context_ref_updates: Vec<TemporalContextRefUpdate>,
}

pub struct Neuron {
    /// Neuron ID — public because Column reads it after removing the neuron from its map.
    pub id: NeuronId,

    pattern_forget_rate: f64,

    /// The single brain-wide grouping coefficient θ, shared by spatial (d=0) and temporal (d>0). It is the
    /// SEED for the adaptive grouping threshold: when a unit has no error history, the live correction
    /// threshold is the derived complement `1 − θ` and recognition fires at similarity ≥ θ. Once error
    /// stats accrue, both sides float together off those stats — see [grouping_error_threshold], the sole
    /// reader of this field. One number, read from opposite sides; no separate merge knob, no phase split.
    group_threshold: f64,

    /// How the live grouping threshold adapts from this unit's running Welford error stats:
    /// Static keeps the seed `1 − θ`; the dynamic modes shift the correction side to mean ± σ (and
    /// recognition to `1 − that`). Shared across spatial and temporal — the error buckets differ, the
    /// adaptation policy does not.
    group_mode: GroupMode,

    /// Brain-wide context_length, replicated on each neuron so recognize_patterns can implement the warmup gate:
    /// skip matching until the context window has had a chance to fill up at the start of a sequence
    context_length: u32,

    /// Master learning toggle, fixed at construction like every other option.
    /// When false, recognition and voting still run but every substrate write is skipped.
    learning: bool,

    /// Per-channel action neuron IDs — ordered Vec for deterministic alternative-action
    /// exploration (neurons are tried in registration order, not hash-iteration order).
    channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,

    // ── Spatial state (d=0 co-activation, no distance dimension) ────────────────

    /// Spatial outgoing connections: target_neuron_id → connection data.
    /// One flat map — there's no distance dimension because spatial is same-frame.
    spatial_connections: FxHashMap<NeuronId, ConnectionData>,

    /// Position metadata per spatial connection target: target_neuron_id → (channel, dimension).
    /// Captured at connection-learn time (the target is active and decorated at that moment) and
    /// rebuilt from central metadata on restore. Prediction evaluation groups votes by position
    /// through this map; a target with no entry never entered the inference neighborhood through
    /// learning, so its votes are skipped — the same restriction the connection learning enforces.
    spatial_target_dims: FxHashMap<NeuronId, (ChannelId, DimensionId)>,

    /// Spatial routing table: every entry is a routing-table entry keyed by which one it is — the
    /// `Normal` or a `Child(pattern_id)` (docs/algorithm.md, "The normal": the normal is "a routing
    /// table entry like any other"). Each entry stores a **context** configuration recognition measures
    /// distance to, frozen at its seed/mint value. The `Normal` key is always
    /// present (inserted at construction); it has no pattern neuron, never propagates, and is never
    /// deleted — child-specific machinery (minting/releasing pattern neurons, serializing children,
    /// the inverted index) filters to `Child(_)` entries. Connections are a separate concern (for
    /// inference); on the d=0 axis the normal's context coincides with the connections, but that is an
    /// artifact of d=0, not relied on here — the normal is stored purely as context.
    spatial_routing_table: FxHashMap<SpatialServer, SpatialRoutingEntry>,

    /// Spatial inverted index: ctx_neuron_id → set of child pattern_ids that reference it. Children
    /// only — the normal is not indexed (nothing looks a pattern up through the normal).
    spatial_context_index: FxHashMap<NeuronId, FxHashSet<NeuronId>>,

    /// Spatial context references: set of parent neurons whose spatial routing tables
    /// reference this neuron. No distance.
    spatial_context_refs: FxHashSet<NeuronId>,

    // ── Level-0 configuration loop (UCAR, docs/algorithm.md) ────────────────────

    /// The sliding history window: the frames this neuron fired for, one horizon back, each holding
    /// the observed neighborhood and which entry served it. The one test (add/delete passes) reads
    /// this directly; nothing else accumulates — no balance, no running error.
    /// Serialized rebased to the save frame so the window survives a restore mid-run (see
    /// `materialize_and_reset_children`); it also refills as the neuron fires afterwards.
    spatial_history: SpatialHistory,

    /// Memoised pairwise distances between this neuron's neighbourhood configurations. A pure cache over
    /// frozen configs (see [SpatialDistanceCache]); the route/delete/add passes read it instead of
    /// recomputing [Neuron::spatial_distance] every active frame.
    spatial_dist_cache: SpatialDistanceCache,

    /// The horizon: how many frames of history the one test evaluates over — "how long the world is
    /// assumed to hold still" (docs/algorithm.md, "The cost"). This is its own knob, in frames, set at
    /// construction. Set it to the episode length and the sliding window holds exactly one episode of
    /// evidence, so no input is ever counted twice. It is decoupled from `pattern_forget_rate` (which
    /// now drives only the temporal decay clock); callers that leave it unset fall back to the old
    /// `round(1 / pattern_forget_rate)` timescale for backward compatibility.
    horizon: u32,

    // ── Temporal state (d>0 sequence, distance-keyed) ───────────────────────────

    /// Temporal outgoing connections by distance: temporal_connections[d] = target → connection.
    /// Index 0 is unused (spatial lives in `spatial_connections`); temporal uses index 1..context_length.
    temporal_connections: Vec<FxHashMap<NeuronId, ConnectionData>>,

    /// Temporal routing table: child_pattern_id → TemporalRoutingEntry.
    temporal_routing_table: FxHashMap<NeuronId, TemporalRoutingEntry>,

    /// Temporal inverted index: ctx_neuron_id → distance → set of pattern_ids.
    temporal_context_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>,

    /// Temporal context references: parent_id → set of distances that reference this neuron.
    temporal_context_refs: FxHashMap<NeuronId, FxHashSet<Distance>>,

    /// Temporal Welford error stats indexed by age. Used for dynamic error thresholds.
    temporal_error_stats: Vec<Option<WelfordState>>,
}

impl Neuron {
    /// Neuron id is allocated by the Thalamus (mirrors how channel and dimension
    /// ids are allocated) and passed in at construction. channel_action_ids are used
    /// for alternative-action lookup during learning (per-channel Vec iteration).
    /// Populated by register_channel_spec() and shared across all neurons.
    ///
    /// The group mode picks how the live grouping threshold adapts from past error rates (the temporal
    /// per-age bucket, or the single spatial bucket); recognition then uses `1 − that` for the same bucket:
    ///   Static       — fixed correction threshold = `1 − group_threshold`
    ///   Conservative — mean + σ of past error rates  (learn outliers)
    ///   Neutral      — mean
    ///   Aggressive   — mean − σ (learn aggressively)
    /// For dynamic modes, `1 − group_threshold` is the warmup fallback until ERROR_MIN_SAMPLES
    /// observations have been recorded in that bucket.
    pub fn new(
        id: NeuronId,
        pattern_forget_rate: f64,
        group_threshold: f64,
        group_mode: GroupMode,
        channel_action_ids: FxHashMap<ChannelId, Vec<NeuronId>>,
        context_length: u32,
        learning: bool,
        horizon: u32,
    ) -> Self {
        Self {
            id,
            pattern_forget_rate,
            group_threshold,
            group_mode,
            context_length,
            learning,
            channel_action_ids,
            spatial_connections: FxHashMap::default(),
            spatial_target_dims: FxHashMap::default(),
            // The routing table always contains the normal (an entry with no pattern neuron); children
            // are added as they are minted.
            spatial_routing_table: {
                let mut m: FxHashMap<SpatialServer, SpatialRoutingEntry> = FxHashMap::default();
                m.insert(SpatialServer::Normal, SpatialRoutingEntry {
                    context: SpatialContext::new(),
                    activation_strength: 0.0,
                    last_activation_frame: 0,
                    config_id: DIST_SENTINEL,
                });
                m
            },
            spatial_context_index: FxHashMap::default(),
            spatial_context_refs: FxHashSet::default(),
            spatial_history: SpatialHistory::default(),
            spatial_dist_cache: SpatialDistanceCache::default(),
            // The horizon is its own knob (the spatial evidence window in frames), set directly.
            horizon: horizon.max(1),
            temporal_connections: Vec::new(),
            temporal_routing_table: FxHashMap::default(),
            temporal_context_index: FxHashMap::default(),
            temporal_context_refs: FxHashMap::default(),
            temporal_error_stats: Vec::new(),
        }
    }

    // ── Grouping threshold ─────────────────────────────────────────────────────

    /// THE grouping operation's threshold, and the SOLE reader of `group_threshold` + `group_mode`.
    ///
    /// Given the relevant Welford error bucket (spatial: the single bucket; temporal: the per-age bucket),
    /// returns the correction (error) threshold E ∈ [0,1]: the static / warmup value is the derived
    /// complement `1 − group_threshold`; the dynamic modes shift it by the bucket's running error stats
    /// (mean ± σ) once ERROR_MIN_SAMPLES observations exist. Recognition treats two connection sets as
    /// "the same" when their match ratio ≥ `1 − E` ([grouping_merge_threshold]); correction mints when the
    /// observed error ratio > E. One number, read from opposite sides — there is no separate merge knob.
    fn grouping_error_threshold(&self, stats: Option<&WelfordState>) -> f64 {
        let fallback = 1.0 - self.group_threshold;
        if self.group_mode == GroupMode::Static { return fallback; }
        match stats {
            Some(s) if s.n >= ERROR_MIN_SAMPLES => {
                let sigma = s.std_dev();
                match self.group_mode {
                    GroupMode::Conservative => s.mean + sigma,
                    GroupMode::Neutral => s.mean,
                    GroupMode::Aggressive => s.mean - sigma,
                    GroupMode::Static => unreachable!(),
                }
            }
            _ => fallback,
        }
    }

    /// Recognition strictness: `1 − E` for the same error bucket. A reliable unit (low E) recognizes
    /// strictly; an unreliable one (high E) recognizes loosely. Derived, never tuned independently.
    fn grouping_merge_threshold(&self, stats: Option<&WelfordState>) -> f64 {
        1.0 - self.grouping_error_threshold(stats)
    }

    /// Temporal correction threshold for a given age — selects the per-age bucket and defers to
    /// [grouping_error_threshold]. Thin bucket-selector; the policy lives in one place.
    pub fn get_temporal_error_threshold(&self, age: Distance) -> f64 {
        self.grouping_error_threshold(self.temporal_error_stats.get(age as usize).and_then(|s| s.as_ref()))
    }

    // ── Error stats ──────────────────────────────────────────────────────────

    /// Record an observed temporal error rate for a given age (Welford online update).
    /// Called after the threshold comparison so the current sample does not
    /// influence its own decision.
    pub fn record_temporal_error(&mut self, age: Distance, error_rate: f64) {
        let idx = age as usize;
        if idx >= self.temporal_error_stats.len() { self.temporal_error_stats.resize(idx + 1, None); }
        let stats = self.temporal_error_stats[idx].get_or_insert_with(WelfordState::new);
        stats.update(error_rate);
    }

    /// Restoration entry point: install a fully-formed Welford bucket for a given
    /// temporal age. Used by Column.restore_neurons to rehydrate per-(neuron, age) error stats
    /// from serialized neuron. Does not validate the stats — the caller owns correctness.
    pub fn load_temporal_error_stats(&mut self, age: Distance, n: u64, mean: f64, m2: f64) {
        let idx = age as usize;
        if idx >= self.temporal_error_stats.len() { self.temporal_error_stats.resize(idx + 1, None); }
        self.temporal_error_stats[idx] = Some(WelfordState { n, mean, m2 });
    }

    // ── Serialization ────────────────────────────────────────────────────────

    /// Serialize persistent state into a plain object for snapshotting and backup.
    /// Returns everything needed to reconstruct this Neuron via Column.load_neuron.
    /// context_index is omitted — it is rebuilt from children's context entries on load.
    pub fn serialize(&self) -> SerializedNeuron {
        SerializedNeuron {
            id: self.id,
            pattern_forget_rate: self.pattern_forget_rate,
            connections: self.serialize_connections(),
            children: self.serialize_children(),
            context_refs: self.serialize_context_refs(),
            error_stats: self.serialize_error_stats(),
            // The normal's stored context configuration — its member neighbor ids. Persisted like a
            // child's context (in contexts.csv), just with no pattern neuron.
            normal_context: self.normal_entry().context.entries().keys().copied().collect(),
            spatial_history: self.serialize_spatial_history(),
        }
    }

    /// Flatten the history window for the snapshot, one row per remembered frame (the histogram expanded
    /// back to the `(frame, served_child, observed)` records the on-disk format keeps), oldest first so a
    /// restore rebuilds it in order.
    fn serialize_spatial_history(&self) -> Vec<SerializedHistoryRecord> {
        let mut out: Vec<SerializedHistoryRecord> = Vec::new();
        for (observed, ev) in self.spatial_history.iter() {
            let served_child = match ev.server { SpatialServer::Child(id) => Some(id), SpatialServer::Normal => None };
            for &frame in &ev.frames {
                out.push(SerializedHistoryRecord { frame, served_child, observed: observed.clone() });
            }
        }
        out.sort_by_key(|r| r.frame);
        out
    }

    /// Rebuild the history window from a snapshot. Records are folded back into the per-configuration
    /// histogram, so the restored neuron carries exactly the evidence it had when saved. The cached
    /// `best_distance` is derived, not serialized, so it is recomputed here as the configuration's
    /// distance to its stored server — the routing table (normal and children) is already loaded here.
    pub fn restore_spatial_history(&mut self, records: &[SerializedHistoryRecord]) {
        let cfg_of: FxHashMap<SpatialServer, Vec<NeuronId>> = self.spatial_entries().into_iter().collect();
        self.spatial_history.clear();
        for r in records {
            let server = match r.served_child { Some(id) => SpatialServer::Child(id), None => SpatialServer::Normal };
            let best_distance = cfg_of.get(&server)
                .map_or(0, |config| Self::spatial_distance(&r.observed, config));
            self.spatial_history.record(r.frame, &r.observed, server, best_distance);
        }
    }

    /// The normal routing-table entry — always present.
    fn normal_entry(&self) -> &SpatialRoutingEntry {
        self.spatial_routing_table.get(&SpatialServer::Normal).expect("normal routing entry must always exist")
    }

    /// Restore the normal's stored context configuration from its member neighbor ids.
    pub fn restore_spatial_normal(&mut self, context_ids: &[NeuronId]) {
        let mut ctx = SpatialContext::new();
        for &id in context_ids {
            if !ctx.has_key(id) { ctx.add_neuron(id, 1.0); }
        }
        let normal = self.spatial_routing_table.get_mut(&SpatialServer::Normal)
            .expect("normal routing entry must always exist");
        normal.context = ctx;
        normal.config_id = DIST_SENTINEL; // config changed — re-derive its interned distance id
    }

    /// Rebuild the position metadata of spatial connection targets from central base-neuron metadata.
    /// Snapshots do not persist the map — connection learning restricted every target to the inference
    /// neighborhood when the edge was created, so the metadata is derivable and re-attached on restore.
    pub fn decorate_spatial_targets(&mut self, meta: &FxHashMap<NeuronId, (ChannelId, DimensionId)>) {
        for &target_id in self.spatial_connections.keys() {
            if let Some(&m) = meta.get(&target_id) {
                self.spatial_target_dims.insert(target_id, m);
            }
        }
    }

    /// Serialize directed connections — flattens spatial (d=0) and temporal (d>0) into a single
    /// distance-keyed array for snapshot back-compat.
    fn serialize_connections(&self) -> Vec<SerializedConnection> {
        let mut result = Vec::new();
        // spatial at distance 0
        for (&to_neuron_id, conn) in &self.spatial_connections {
            result.push(SerializedConnection {
                distance: 0,
                to_neuron_id,
                strength: conn.strength,
                reward: conn.reward,
            });
        }
        // temporal at distance > 0
        for (distance, targets) in self.temporal_connections.iter().enumerate() {
            if distance == 0 { continue; } // temporal slot 0 is unused
            for (&to_neuron_id, conn) in targets {
                result.push(SerializedConnection {
                    distance: distance as Distance,
                    to_neuron_id,
                    strength: conn.strength,
                    reward: conn.reward,
                });
            }
        }
        result
    }

    /// Serialize the routing tables (child patterns). Each child carries its activation
    /// strength, last activation frame, and flattened context entries. Spatial children's
    /// entries surface with distance=0 (a placeholder for the snapshot format).
    fn serialize_children(&self) -> Vec<SerializedChild> {
        let mut result = Vec::new();
        // Children only — the normal entry is serialized separately (normal_context), not as a child.
        for (&server, entry) in &self.spatial_routing_table {
            let SpatialServer::Child(pattern_id) = server else { continue };
            result.push(SerializedChild {
                pattern_id,
                spatial: true,
                activation_strength: entry.activation_strength,
                last_activation_frame: entry.last_activation_frame,
                context: entry.context.get_entries(),
            });
        }
        for (&pattern_id, entry) in &self.temporal_routing_table {
            result.push(SerializedChild {
                pattern_id,
                spatial: false,
                activation_strength: entry.activation_strength,
                last_activation_frame: entry.last_activation_frame,
                context: entry.context.get_entries(),
            });
        }
        result
    }

    /// Serialize context references — both spatial and temporal flattened into a single
    /// {parent_id, distances} array. Spatial entries use a single-element [0] distances list
    /// as a placeholder for the snapshot format.
    fn serialize_context_refs(&self) -> Vec<SerializedContextRef> {
        let mut result = Vec::new();
        for &parent_id in &self.spatial_context_refs {
            result.push(SerializedContextRef {
                parent_id,
                spatial: true,
                distances: vec![0],
            });
        }
        for (&parent_id, distances) in &self.temporal_context_refs {
            result.push(SerializedContextRef {
                parent_id,
                spatial: false,
                distances: distances.iter().copied().collect(),
            });
        }
        result
    }

    /// Serialize the temporal per-age Welford error stats (age >= 1). Age 0 never carries stats —
    /// error feedback comes from evaluate_vote_error, which returns None for age 0 (an age-0 neuron is
    /// just voting now, nothing to correct) — so it is skipped. (The spatial axis has no error stats;
    /// the configuration loop is threshold-free.)
    fn serialize_error_stats(&self) -> Vec<SerializedErrorStats> {
        let mut result = Vec::new();
        for (age, stats) in self.temporal_error_stats.iter().enumerate() {
            if age == 0 { continue; }
            if let Some(s) = stats {
                result.push(SerializedErrorStats {
                    age: age as Distance,
                    n: s.n,
                    mean: s.mean,
                    m2: s.m2,
                });
            }
        }
        result
    }

    // ── Connections ──────────────────────────────────────────────────────────

    /// Returns if there is a connection at distance to a target neuron.
    /// Distance 0 routes to spatial_connections; distance > 0 routes to temporal_connections[d].
    pub fn has_connection(&self, distance: Distance, to_neuron_id: NeuronId) -> bool {
        if distance == 0 {
            return self.spatial_connections.contains_key(&to_neuron_id);
        }
        match self.temporal_connections.get(distance as usize) {
            Some(distance_map) => distance_map.contains_key(&to_neuron_id),
            None => false,
        }
    }

    /// Creates a connection at distance to target neuron.
    /// Distance 0 routes to spatial_connections; distance > 0 routes to temporal_connections[d].
    pub fn create_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, strength: Strength, reward: Reward) {
        if distance == 0 {
            self.spatial_connections.insert(to_neuron_id, ConnectionData { strength, reward });
            return;
        }
        let idx = distance as usize;
        if idx >= self.temporal_connections.len() { self.temporal_connections.resize_with(idx + 1, FxHashMap::default); }
        self.temporal_connections[idx].insert(to_neuron_id, ConnectionData { strength, reward });
    }

    /// Strengthen the connection if it already exists, otherwise create it with strength=1.0.
    /// Reward is folded in via strengthen_connection's exponential smoothing on update.
    /// On create, reward is stored as-is.
    /// This is the core (create-or-strengthen + smoothed reward) shared by temporal `upsert_connection` and supervised `Brain.learn`.
    /// Supervised callers use this directly because they don't need the alt-action lookup `upsert_connection` adds on top.
    pub fn strengthen_or_create_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, reward: Reward) {
        if self.has_connection(distance, to_neuron_id) {
            self.strengthen_connection(distance, to_neuron_id, reward);
        } else {
            self.create_connection(distance, to_neuron_id, 1.0, reward);
        }
    }

    /// Upsert a connection at distance to the target neuron: create if missing, else strengthen (smoothing the reward).
    /// If the resulting reward is negative, wire an alternative action with neutral reward so it can be tried next time.
    /// Used by temporal learning in process_frame, where reward exploration matters.
    pub fn upsert_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, channel_id: ChannelId, reward: Reward) {

        // strengthen or create the connection first
        self.strengthen_or_create_connection(distance, to_neuron_id, reward);

        // For actions with negative (smoothed) rewards, save an alternative with neutral reward — we'll try it next time.
        let conn_reward = self.get_connection_reward(distance, to_neuron_id).unwrap();
        if conn_reward < 0.0 {
            if let Some(alt) = self.find_alternative_action(distance, channel_id, to_neuron_id) {
                self.create_connection(distance, alt, 1.0, 0.0);
            }
        }
    }

    /// Read the stored reward on a connection at the given distance, if it exists.
    fn get_connection_reward(&self, distance: Distance, to_neuron_id: NeuronId) -> Option<f64> {
        if distance == 0 {
            return self.spatial_connections.get(&to_neuron_id).map(|c| c.reward);
        }
        self.temporal_connections.get(distance as usize)
            .and_then(|dm| dm.get(&to_neuron_id))
            .map(|c| c.reward)
    }

    /// Updates the connection at distance to target neuron - increments strength and updates reward.
    pub fn strengthen_connection(&mut self, distance: Distance, to_neuron_id: NeuronId, reward: Reward) {
        let conn = if distance == 0 {
            self.spatial_connections.get_mut(&to_neuron_id).expect("Unknown connection")
        } else {
            let distance_map = self.temporal_connections.get_mut(distance as usize).expect("Unknown connection");
            distance_map.get_mut(&to_neuron_id).expect("Unknown connection")
        };

        // increment the strength of the connection
        conn.strength += 1.0;

        // update reward with dynamic exponential smoothing - calculates exact expected value based on means
        let alpha = 1.0 / conn.strength;
        conn.reward = alpha * reward + (1.0 - alpha) * conn.reward;
    }

    // ── Voting ───────────────────────────────────────────────────────────────

    /// Returns votes from this neuron at a specific age (temporal voting).
    /// Reads connections[age + 1] — predicting one frame past the neuron's activation age.
    pub fn vote(&self, age: Distance) -> Vec<Vote> {
        self.vote_at_distance(age + 1)
    }

    /// Returns votes from this neuron at a fixed distance.
    /// Distance 0 reads spatial_connections (co-activation predictions for the current frame).
    /// Distance > 0 reads temporal_connections[d] (next-frame and further predictions).
    pub fn vote_at_distance(&self, distance: Distance) -> Vec<Vote> {
        if distance == 0 {
            let mut result = Vec::new();
            for (&neuron_id, conn) in &self.spatial_connections {
                if conn.strength > 0.0 {
                    result.push(Vote { neuron_id, strength: conn.strength, reward: conn.reward, distance });
                }
            }
            return result;
        }

        // temporal connections
        let distance_map = match self.temporal_connections.get(distance as usize) {
            Some(dm) => dm,
            None => return Vec::new(),
        };
        let mut result = Vec::new();
        for (&neuron_id, conn) in distance_map {
            if conn.strength > 0.0 {
                result.push(Vote { neuron_id, strength: conn.strength, reward: conn.reward, distance });
            }
        }
        result
    }

    // ── Child pattern management ─────────────────────────────────────────────

    /// Get effective activation strength for a child pattern with lazy decay.
    /// Looks up in whichever routing table holds the pattern (spatial or temporal).
    pub fn get_child_effective_activation_strength(&self, pattern_id: NeuronId, current_frame: FrameNumber) -> f64 {
        if let Some(entry) = self.spatial_routing_table.get(&SpatialServer::Child(pattern_id)) {
            return self.decay_activation(entry.activation_strength, entry.last_activation_frame, current_frame);
        }
        if let Some(entry) = self.temporal_routing_table.get(&pattern_id) {
            return self.decay_activation(entry.activation_strength, entry.last_activation_frame, current_frame);
        }
        0.0
    }

    /// Apply lazy time-decay to a stored activation strength, floored at zero.
    fn decay_activation(&self, activation_strength: f64, last_activation_frame: FrameNumber, current_frame: FrameNumber) -> f64 {
        f64::max(0.0, activation_strength - (current_frame - last_activation_frame) as f64 * self.pattern_forget_rate)
    }

    /// Materialize lazy decay for a child pattern.
    pub fn materialize_child_strength(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) {
        let effective = self.get_child_effective_activation_strength(pattern_id, current_frame);
        if let Some(entry) = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id)) {
            entry.activation_strength = effective;
            return;
        }
        if let Some(entry) = self.temporal_routing_table.get_mut(&pattern_id) {
            entry.activation_strength = effective;
        }
    }

    /// Fold pending lazy decay into every child's stored strength and rebase its activation clock to 0,
    /// across both the temporal and spatial routing tables.
    /// Returns each child's recomputed death frame so the caller can rebuild the death ledger on restore.
    /// Uses this neuron's own pattern_forget_rate — the same rate that drives its lazy-decay math.
    pub fn materialize_and_reset_children(&mut self, current_frame: FrameNumber) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();

        // Collect ids up front: materialize_child_strength borrows self mutably, so we can't hold
        // an iterator over the routing table while calling it.
        let temporal_ids: Vec<NeuronId> = self.temporal_routing_table.keys().copied().collect();
        for pattern_id in temporal_ids {

            // Fold any pending lazy decay into the stored strength so the snapshot is exact.
            self.materialize_child_strength(pattern_id, current_frame);
            if let Some(entry) = self.temporal_routing_table.get_mut(&pattern_id) {

                // Rebase the activation clock to 0 — the restored brain starts counting frames fresh.
                entry.last_activation_frame = 0;

                // Recompute the death frame from the now-materialized strength rather than persisting it.
                let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
                entries.push(DeathFrameEntry { pattern_id, death_frame });
            }
        }

        // Spatial children carry death frames too, so materialize and reset them identically.
        let spatial_ids: Vec<NeuronId> = self.spatial_routing_table.keys()
            .filter_map(|s| if let SpatialServer::Child(id) = s { Some(*id) } else { None }).collect();
        for pattern_id in spatial_ids {

            // Fold any pending lazy decay into the stored strength so the snapshot is exact.
            self.materialize_child_strength(pattern_id, current_frame);
            if let Some(entry) = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id)) {

                // Rebase the activation clock to 0 — the restored brain starts counting frames fresh.
                entry.last_activation_frame = 0;

                // Recompute the death frame from the now-materialized strength rather than persisting it.
                let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
                entries.push(DeathFrameEntry { pattern_id, death_frame });
            }
        }

        // Rebase the spatial history onto the same fresh clock. Each remembered frame becomes an offset
        // relative to the save frame — always ≤ 0 — so when the restored brain resumes at frame 0 and
        // ticks forward, the sliding horizon window expires them on the same schedule they would have
        // followed had the brain never stopped. Absolute frames would instead all sit far in the future
        // of a frame-0 restart and never expire, freezing the window.
        self.spatial_history.rebase(current_frame);

        entries
    }

    /// Compute each child's death frame from its current activation strength, without mutating any state.
    /// Walks both the temporal and spatial routing tables, mirroring materialize_and_reset_children.
    /// Used on restore to rebuild the death ledger when strengths are already materialized.
    /// Uses this neuron's own pattern_forget_rate — the same rate that drives its lazy-decay math.
    pub fn compute_death_frames(&self) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();

        // Temporal children: a pattern survives until its activation strength has decayed away.
        // death_frame is how many frames that takes at the forget rate, rounded up.
        for (&pattern_id, entry) in &self.temporal_routing_table {
            let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
            entries.push(DeathFrameEntry { pattern_id, death_frame });
        }

        // Spatial children use the same strength/forget-rate decay model, so compute them identically.
        // The normal has no pattern neuron and no death frame, so it is skipped.
        for (&server, entry) in &self.spatial_routing_table {
            let SpatialServer::Child(pattern_id) = server else { continue };
            let death_frame = (entry.activation_strength / self.pattern_forget_rate).ceil() as FrameNumber;
            entries.push(DeathFrameEntry { pattern_id, death_frame });
        }

        entries
    }

    /// Increments activation strength for a child pattern - materializes all owner-scoped lazy decay first.
    /// Returns death frame for pattern neurons.
    pub fn strengthen_child_activation(&mut self, pattern_id: NeuronId, current_frame: FrameNumber) -> Option<FrameNumber> {
        if !self.spatial_routing_table.contains_key(&SpatialServer::Child(pattern_id)) && !self.temporal_routing_table.contains_key(&pattern_id) { return None; }

        // update all strengths based on decay rate first
        self.materialize_child_strength(pattern_id, current_frame);

        let entry_strength = if let Some(entry) = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id)) {
            entry.activation_strength += 1.0;
            entry.last_activation_frame = current_frame;
            entry.activation_strength
        } else {
            let entry = self.temporal_routing_table.get_mut(&pattern_id).unwrap();
            entry.activation_strength += 1.0;
            entry.last_activation_frame = current_frame;
            entry.activation_strength
        };

        Some(current_frame + (entry_strength / self.pattern_forget_rate).ceil() as i64)
    }

    /// Add a temporal child pattern to the temporal routing table and populate its context.
    /// Returns death frame for pattern neurons.
    pub fn add_temporal_pattern(&mut self, pattern_id: NeuronId, context: &[ContextRefEntry], current_frame: FrameNumber) -> Option<FrameNumber> {
        self.add_temporal_child(pattern_id, 0.0);
        for entry in context { self.add_temporal_context(pattern_id, entry.neuron_id, entry.distance, 1.0); }
        self.strengthen_child_activation(pattern_id, current_frame)
    }

    /// Add a spatial child pattern to the spatial routing table and populate its context.
    /// Spatial context has no distance dimension — entries are just (pattern, ctx_neuron) pairs.
    /// The child was minted by the add pass, which already showed it pays for its storage, so it
    /// serves from its first frame.
    /// Returns death frame for pattern neurons.
    pub fn add_spatial_pattern(&mut self, pattern_id: NeuronId, context: &[NeuronId], current_frame: FrameNumber) -> Option<FrameNumber> {
        self.add_spatial_child(pattern_id, 0.0);
        for &ctx_neuron_id in context { self.add_spatial_context(pattern_id, ctx_neuron_id, 1.0); }
        let death = self.strengthen_child_activation(pattern_id, current_frame);
        // A new entry joined the table: frames it now wins move to it — reassign so the one test sees the
        // child from its first frame. The newborn's config is exactly `context`, sorted for the merge.
        let mut config: Vec<NeuronId> = context.to_vec();
        config.sort_unstable();
        config.dedup();
        self.reassign_after_mint(pattern_id, &config);
        death
    }

    /// Add a temporal child pattern to the temporal routing table (no context yet).
    pub fn add_temporal_child(&mut self, pattern_id: NeuronId, initial_strength: f64) {
        if !self.temporal_routing_table.contains_key(&pattern_id) {
            self.temporal_routing_table.insert(pattern_id, TemporalRoutingEntry {
                context: TemporalContext::new(),
                activation_strength: initial_strength,
                last_activation_frame: 0,
            });
        }
    }

    /// Add a spatial child pattern to the spatial routing table (no context yet).
    pub fn add_spatial_child(&mut self, pattern_id: NeuronId, initial_strength: f64) {
        if !self.spatial_routing_table.contains_key(&SpatialServer::Child(pattern_id)) {
            self.spatial_routing_table.insert(SpatialServer::Child(pattern_id), SpatialRoutingEntry {
                context: crate::context::SpatialContext::new(),
                activation_strength: initial_strength,
                last_activation_frame: 0,
                config_id: DIST_SENTINEL,
            });
        }
    }

    /// Check if a child pattern can be deleted (is a zombie).
    pub fn can_delete_child(&self, pattern_id: NeuronId, current_frame: FrameNumber) -> bool {
        // if the pattern has not been activated in some time, die!
        self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0
    }

    // ── TemporalContext management ───────────────────────────────────────────────────

    /// Adds an entry to a TEMPORAL pattern's context by (neuron_id, distance).
    pub fn add_temporal_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance, strength: Strength) {
        let entry = self.temporal_routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("add_temporal_context: pattern not found in temporal routing table: {}", pattern_id));

        if entry.context.has_key(neuron_id, distance) {
            entry.context.strengthen_neuron(neuron_id, distance);
        } else {
            entry.context.add_neuron(neuron_id, distance, strength);
        }

        self.add_temporal_context_index(neuron_id, distance, pattern_id);
    }

    /// Adds an entry to a SPATIAL pattern's context by neuron_id (no distance).
    pub fn add_spatial_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, strength: Strength) {
        let entry = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id))
            .unwrap_or_else(|| panic!("add_spatial_context: pattern not found in spatial routing table: {}", pattern_id));

        if entry.context.has_key(neuron_id) {
            entry.context.strengthen_neuron(neuron_id);
        } else {
            entry.context.add_neuron(neuron_id, strength);
        }
        // The configuration changed — drop its interned id so the distance cache re-derives it.
        entry.config_id = DIST_SENTINEL;

        self.add_spatial_context_index(neuron_id, pattern_id);
    }

    /// Adds a neuron to the temporal context index.
    fn add_temporal_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) {
        let dist_map = self.temporal_context_index.entry(neuron_id).or_insert_with(FxHashMap::default);
        let patterns = dist_map.entry(distance).or_insert_with(FxHashSet::default);
        patterns.insert(pattern_id);
    }

    /// Adds a neuron to the spatial context index.
    fn add_spatial_context_index(&mut self, neuron_id: NeuronId, pattern_id: NeuronId) {
        let patterns = self.spatial_context_index.entry(neuron_id).or_insert_with(FxHashSet::default);
        patterns.insert(pattern_id);
    }

    /// Removes an entry from a TEMPORAL child pattern's context.
    /// Returns true if no pattern references this neuron at this distance anymore (orphaned).
    pub fn remove_temporal_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {
        let entry = self.temporal_routing_table.get_mut(&pattern_id)
            .unwrap_or_else(|| panic!("remove_temporal_context: pattern {} not found in temporal routing table of neuron {}", pattern_id, self.id));
        entry.context.remove(neuron_id, distance);
        self.remove_temporal_context_index(neuron_id, distance, pattern_id)
    }


    /// Remove a pattern from the temporal context inverted index for a given neuron/distance.
    /// Returns true if no pattern references this neuron at this distance anymore (orphaned).
    pub fn remove_temporal_context_index(&mut self, neuron_id: NeuronId, distance: Distance, pattern_id: NeuronId) -> bool {
        let dist_map = self.temporal_context_index.get_mut(&neuron_id)
            .unwrap_or_else(|| panic!("remove_temporal_context_index: neuron {} not found in temporal_context_index of neuron {}", neuron_id, self.id));
        let patterns = dist_map.get_mut(&distance)
            .unwrap_or_else(|| panic!("remove_temporal_context_index: distance {} not found for neuron {} in temporal_context_index of neuron {}", distance, neuron_id, self.id));
        patterns.remove(&pattern_id);
        if patterns.is_empty() {
            dist_map.remove(&distance);
            if dist_map.is_empty() { self.temporal_context_index.remove(&neuron_id); }
            return true;
        }
        false
    }


    /// Remove all temporal references to a dying neuron from this parent's children's contexts.
    /// Returns pattern IDs whose context was modified (caller checks if deletable).
    pub fn remove_temporal_context_neuron(&mut self, neuron_id: NeuronId, distances: &FxHashSet<Distance>) -> FxHashSet<NeuronId> {
        let mut affected_patterns = FxHashSet::default();
        let dist_map = self.temporal_context_index.get(&neuron_id)
            .unwrap_or_else(|| panic!("remove_temporal_context_neuron: neuron {} not found in temporal_context_index of neuron {}", neuron_id, self.id));

        let mut ops: Vec<(Distance, Vec<NeuronId>)> = Vec::new();
        for &distance in distances {
            let patterns = dist_map.get(&distance)
                .unwrap_or_else(|| panic!("remove_temporal_context_neuron: distance {} for neuron {} not found in temporal_context_index of neuron {}", distance, neuron_id, self.id));
            ops.push((distance, patterns.iter().copied().collect()));
        }

        for (distance, pattern_ids) in &ops {
            for &pattern_id in pattern_ids {
                let entry = self.temporal_routing_table.get_mut(&pattern_id)
                    .unwrap_or_else(|| panic!("remove_temporal_context_neuron: pattern {} not found in temporal routing table of neuron {}", pattern_id, self.id));
                entry.context.remove(neuron_id, *distance);
                affected_patterns.insert(pattern_id);
            }
        }

        if let Some(dist_map) = self.temporal_context_index.get_mut(&neuron_id) {
            for &distance in distances {
                dist_map.remove(&distance);
            }
            if dist_map.is_empty() { self.temporal_context_index.remove(&neuron_id); }
        }

        affected_patterns
    }


    // ── TemporalContext references ───────────────────────────────────────────────────

    /// Add a TEMPORAL context reference from another neuron to this neuron.
    pub fn add_temporal_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        self.temporal_context_refs.entry(referencing_neuron_id).or_insert_with(FxHashSet::default).insert(distance);
    }

    /// Add a SPATIAL context reference from another neuron to this neuron.
    pub fn add_spatial_context_ref(&mut self, referencing_neuron_id: NeuronId) {
        self.spatial_context_refs.insert(referencing_neuron_id);
    }

    /// Remove a TEMPORAL context reference from another neuron to this neuron.
    pub fn remove_temporal_context_ref(&mut self, referencing_neuron_id: NeuronId, distance: Distance) {
        let distances = match self.temporal_context_refs.get_mut(&referencing_neuron_id) {
            Some(d) => d,
            None => return,
        };
        distances.remove(&distance);
        if distances.is_empty() { self.temporal_context_refs.remove(&referencing_neuron_id); }
    }

    /// Remove a SPATIAL context reference from another neuron to this neuron.
    pub fn remove_spatial_context_ref(&mut self, referencing_neuron_id: NeuronId) {
        self.spatial_context_refs.remove(&referencing_neuron_id);
    }

    /// Remove all SPATIAL references to a dying neuron from this parent's children's contexts.
    /// Returns pattern IDs whose context was modified (caller checks if deletable).
    pub fn remove_spatial_context_neuron(&mut self, neuron_id: NeuronId) -> FxHashSet<NeuronId> {
        let mut affected_patterns = FxHashSet::default();
        let pattern_ids: Vec<NeuronId> = match self.spatial_context_index.get(&neuron_id) {
            Some(ps) => ps.iter().copied().collect(),
            None => return affected_patterns,
        };
        for &pattern_id in &pattern_ids {
            let entry = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id))
                .unwrap_or_else(|| panic!("remove_spatial_context_neuron: pattern {} not found in spatial routing table of neuron {}", pattern_id, self.id));
            entry.context.remove(neuron_id);
            entry.config_id = DIST_SENTINEL; // config shrank — re-derive its interned distance id
            affected_patterns.insert(pattern_id);
        }
        self.spatial_context_index.remove(&neuron_id);
        affected_patterns
    }

    /// Check if a neuron exists in the spatial context index (for column purge ops).
    pub fn has_spatial_context_index_entry(&self, neuron_id: NeuronId) -> bool {
        self.spatial_context_index.contains_key(&neuron_id)
    }

    /// Remove a single context entry from a SPATIAL child pattern's context.
    /// Returns true if the context neuron is no longer referenced by any child spatial pattern.
    pub fn remove_spatial_context(&mut self, pattern_id: NeuronId, neuron_id: NeuronId) -> bool {
        if let Some(entry) = self.spatial_routing_table.get_mut(&SpatialServer::Child(pattern_id)) {
            entry.context.remove(neuron_id);
            entry.config_id = DIST_SENTINEL; // config shrank — re-derive its interned distance id
        }
        // scrub the inverted index for (pattern, ctx_neuron)
        if let Some(patterns) = self.spatial_context_index.get_mut(&neuron_id) {
            patterns.remove(&pattern_id);
            if patterns.is_empty() {
                self.spatial_context_index.remove(&neuron_id);
                return true;
            }
        }
        false
    }

    /// Check if a context entry exists for a spatial child pattern (for same-pulse de-dup).
    pub fn has_spatial_context_key(&self, pattern_id: NeuronId, neuron_id: NeuronId) -> bool {
        self.spatial_routing_table.get(&SpatialServer::Child(pattern_id))
            .map_or(false, |e| e.context.has_key(neuron_id))
    }


    /// Apply a batch of TEMPORAL context-reference updates targeting this neuron.
    pub fn apply_temporal_context_ref_updates(&mut self, updates: &[TemporalContextRefUpdate]) {
        for update in updates {
            self.add_temporal_context_ref(update.parent_id, update.distance);
        }
    }

    /// Apply a batch of SPATIAL context-reference updates targeting this neuron.
    pub fn apply_spatial_context_ref_updates(&mut self, updates: &[SpatialContextRefUpdate]) {
        for update in updates {
            self.add_spatial_context_ref(update.parent_id);
        }
    }

    /// Get temporal context refs (for delete cascade).
    pub fn get_temporal_context_refs(&self) -> &FxHashMap<NeuronId, FxHashSet<Distance>> {
        &self.temporal_context_refs
    }

    /// Get spatial context refs (for spatial delete cascade — the set of parent neurons whose
    /// spatial routing tables reference this neuron).
    pub fn get_spatial_context_refs(&self) -> &FxHashSet<NeuronId> {
        &self.spatial_context_refs
    }

    /// Read-only access to the temporal routing table (for Column snapshot/death-frame/delete ops).
    pub fn get_temporal_routing_table(&self) -> &FxHashMap<NeuronId, TemporalRoutingEntry> {
        &self.temporal_routing_table
    }

    /// Read-only access to the spatial routing table (for delete cascade).
    pub fn get_spatial_routing_table(&self) -> &FxHashMap<SpatialServer, SpatialRoutingEntry> {
        &self.spatial_routing_table
    }

    /// Inspection: flatten this neuron's outgoing connections into
    /// (distance, target_neuron_id, strength, reward) tuples. Spatial connections
    /// surface at distance=0; temporal at their distance index. Used by the diagnostic API
    /// to dump connection state for tipping-point analysis.
    pub fn get_connections(&self) -> Vec<(Distance, NeuronId, f64, f64)> {
        let mut out = Vec::new();
        for (&target_id, conn) in &self.spatial_connections {
            out.push((0, target_id, conn.strength, conn.reward));
        }
        for (idx, dist_map) in self.temporal_connections.iter().enumerate() {
            if idx == 0 { continue; }
            let dist = idx as Distance;
            for (&target_id, conn) in dist_map {
                out.push((dist, target_id, conn.strength, conn.reward));
            }
        }
        out
    }

    /// Mutable access to the temporal routing table (for Column restore — sets last_activation_frame).
    pub fn get_temporal_routing_table_mut(&mut self) -> &mut FxHashMap<NeuronId, TemporalRoutingEntry> {
        &mut self.temporal_routing_table
    }

    /// Mutable access to the spatial routing table (for Column restore — sets last_activation_frame).
    pub fn get_spatial_routing_table_mut(&mut self) -> &mut FxHashMap<SpatialServer, SpatialRoutingEntry> {
        &mut self.spatial_routing_table
    }


    /// Remove a child pattern from whichever routing table holds it (for Column delete cascade).
    pub fn remove_routing_entry(&mut self, pattern_id: NeuronId) {
        if self.spatial_routing_table.contains_key(&SpatialServer::Child(pattern_id)) {
            // Scrub the context index and re-assign the cache — same bookkeeping as a delete-pass retire.
            self.remove_spatial_child(pattern_id);
            return;
        }
        self.temporal_routing_table.remove(&pattern_id);
    }

    /// Check if a temporal context key exists for a child pattern (for Column delete cascade).
    pub fn has_temporal_context_key(&self, pattern_id: NeuronId, neuron_id: NeuronId, distance: Distance) -> bool {
        self.temporal_routing_table.get(&pattern_id)
            .map_or(false, |e| e.context.has_key(neuron_id, distance))
    }


    /// Check if a (neuron, distance) exists in the temporal context index (for Column purge).
    pub fn has_temporal_context_index_entry(&self, neuron_id: NeuronId, distance: Distance) -> bool {
        self.temporal_context_index.get(&neuron_id)
            .map_or(false, |dist_map| dist_map.contains_key(&distance))
    }


    // ── Frame processing ─────────────────────────────────────────────────────

    /// Process a frame for this neuron: derive per-age tasks, learn connections, match patterns,
    /// install pre-created error-correction patterns, and cast votes. One call per active neuron
    /// per frame. Matching is skipped when level_context is None/empty. Votes are cast for each
    /// eligible voting age unless suppressed by a this-frame match (activate=true) or
    /// error-correction activation at the same age.
    /// Spatial frame processing — the level-0 configuration loop (UCAR, docs/algorithm.md).
    /// The neuron observes its neighborhood `O`, routes it to the closest entry (its normal or one
    /// of its children), serves it, records the frame, then reconsiders its structure by the one
    /// test: the delete pass retires a child that no longer covers its storage, and the add pass
    /// mints a child at `O` when it would remove more history-error than it costs. No thresholds,
    /// no votes leaving the neuron, no per-age iteration — every decision is read off the history.
    pub fn process_spatial_frame(
        &mut self,
        level_neighbors: Option<&SpatialContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        _inference_neighbors: &[ActiveNeuron],
        current_frame: FrameNumber,
    ) -> SpatialFrameResult {
        let mut timings = NeuronOpTimings::default();

        // Freshly-created error patterns skip all substrate learning in their birth frame.
        let should_learn = self.learning && !new_error_pattern_ids.contains(&self.id);

        // O: the observed neighborhood — the active neighbors of this neuron's declared neighbor
        // channels, as a sorted set. Cold start (no neighbors seen yet) is silence: there is nothing
        // to route, serve, or decide (docs/algorithm.md, "The base model": cold start is silence).
        let observed: Vec<NeuronId> = match level_neighbors {
            Some(lc) if lc.size() > 0 => {
                let mut v: Vec<NeuronId> = lc.entries().keys().copied().collect();
                v.sort_unstable();
                v
            }
            _ => return SpatialFrameResult { bid: None, observed: Vec::new(), served_distance: 0, delete_candidate: None, timings },
        };

        // 1. Age.
        self.age_spatial_history(current_frame);

        // Seed the normal on the neuron's first observation (idempotent, election-independent).
        if self.learning && self.normal_entry().context.size() == 0 {
            let mut ctx = SpatialContext::new();
            for &id in &observed { ctx.add_neuron(id, 1.0); }
            let normal = self.spatial_routing_table.get_mut(&SpatialServer::Normal)
                .expect("normal routing entry must always exist");
            normal.context = ctx;
            normal.config_id = DIST_SENTINEL;
        }

        let observed_id = self.spatial_dist_cache.intern(&observed);
        let entries = self.spatial_entry_ids();

        // 2. Route.
        let t = std::time::Instant::now();
        let (server, served_distance) = self.spatial_route_cached(observed_id, &entries);
        timings.recognize_patterns += t.elapsed().as_secs_f64();

        // The one test as DECISIONS only. Delete candidate = worst-failing child (never the one that
        // served). The add pass (only when the normal served with error) decides whether a new child at O
        // pays; since the current frame is not recorded before it runs, it folds in this frame explicitly.
        let t = std::time::Instant::now();
        let serving = match server { SpatialServer::Child(pid) => Some(pid), SpatialServer::Normal => None };
        let delete_candidate = if should_learn { self.spatial_delete_candidate(serving, &entries) } else { None };

        // This neuron's single child-activation request for the election: recognize the selected child, or
        // (normal selected + add pays) request a new child at O. Recognition already covers O, so never both.
        let bid = match server {
            SpatialServer::Child(pid) => {
                let config = Self::config_sorted(&self.spatial_routing_table[&SpatialServer::Child(pid)].context);
                let mut covered = Self::sorted_intersection(&observed, &config);
                covered.push(self.id);
                covered.sort_unstable();
                covered.dedup();
                Some(SpatialBid { bidder_id: self.id, pattern_id: pid, covered })
            }
            SpatialServer::Normal => {
                if should_learn && served_distance > 0 && self.spatial_add_pass_pays(observed_id, &observed, served_distance) {
                    let mut covered = observed.clone();
                    covered.push(self.id);
                    covered.sort_unstable();
                    covered.dedup();
                    Some(SpatialBid { bidder_id: self.id, pattern_id: NEW_CHILD_BID, covered })
                } else {
                    None
                }
            }
        };
        timings.correct_errors += t.elapsed().as_secs_f64();

        // A pure normal-serve (learning, no request) does not enter the election — finalize it inline:
        // record it (server = Normal) and apply its delete now. If a neighbour's winning bid subsumes it
        // this frame, the sweep's inhibition-prune drops this record. A REQUEST records nothing here —
        // commit_spatial_frame does that, and only if it wins. (bid None + delete_candidate Some => the
        // delete was applied inline and only needs cross-neuron cleanup.)
        if should_learn && bid.is_none() {
            self.spatial_history.record(current_frame, &observed, server, served_distance);
            if let Some(pid) = delete_candidate { self.remove_spatial_child(pid); }
            return SpatialFrameResult { bid: None, observed: Vec::new(), served_distance, delete_candidate, timings };
        }

        SpatialFrameResult { bid, observed, served_distance, delete_candidate, timings }
    }

    /// Commit a child-activation request that won the election — called by the thalamus only for the
    /// winner (a recognized child, or a new child just created at `O`). Applies what the frame pass
    /// deferred: bump the serving child's clock, record `(frame, O, server)`, retire the delete candidate
    /// (never the child that served). A losing request is never committed, so nothing needs rolling back.
    pub fn commit_spatial_frame(
        &mut self,
        observed: &[NeuronId],
        server: SpatialServer,
        served_distance: u32,
        delete_candidate: Option<NeuronId>,
        current_frame: FrameNumber,
    ) {
        if let SpatialServer::Child(pid) = server {
            if let Some(entry) = self.spatial_routing_table.get_mut(&SpatialServer::Child(pid)) {
                entry.last_activation_frame = current_frame;
            }
        }
        self.spatial_history.record(current_frame, observed, server, served_distance);
        if let Some(pid) = delete_candidate {
            if SpatialServer::Child(pid) != server { self.remove_spatial_child(pid); }
        }
    }

    /// Drop this frame's record from the history — the neuron was inhibited (subsumed) by a neighbour's
    /// unit in contraction, so it contributed nothing to the file and this frame is not its evidence.
    /// Leaving it in would tell the one test the neuron is serving its neighbourhood badly (as `Normal`)
    /// and drive a child the neuron never needs, since a neighbour already covers it every such frame.
    ///
    /// The record to drop is the one just appended this frame — the tail. Matching the tail against the
    /// current frame guards against dropping an older record on a frame this neuron did not record.
    /// Removing a record does not move any entry, so no server reassignment is needed; the delete/add
    /// passes simply see one fewer frame next time.
    pub fn drop_inhibited_spatial_frame(&mut self, current_frame: FrameNumber) {
        self.spatial_history.drop_last_if_frame(current_frame);
    }

    /// Drop history records older than the horizon (docs/algorithm.md, "The frame, step by step":
    /// the neuron remembers **the frames it was active for**, one horizon back).
    ///
    /// This is a sliding window in frame numbers: keep every record within `horizon` frames of the
    /// current one, drop the rest. It requires the frame counter to climb monotonically — the host must
    /// not reset it per input, or a per-input restart would leave every record at the same low frame and
    /// the cutoff would never expire anything. Set the horizon to the episode length and the window holds
    /// exactly one episode, sliding one frame at a time, so no image is ever counted twice. A neuron that
    /// was not active at an expiring frame simply has no record for it; there is nothing to drop.
    ///
    /// An unbounded window breaks the one test outright: the add pass sums its benefit over every
    /// remembered frame while an entry's cost stays `1 + |config|`, so as episodes accumulate, benefit
    /// grows without bound and every configuration eventually looks worth its own child.
    fn age_spatial_history(&mut self, current_frame: FrameNumber) {
        let cutoff = current_frame - self.horizon as FrameNumber;
        self.spatial_history.age(cutoff);
    }

    /// Hamming distance between an observed neighborhood and a stored configuration: |O △ C| — the
    /// count of neurons the configuration would get wrong, which is exactly the corrections that
    /// would follow this activation (docs/algorithm.md, "The distance"). Service cost and match
    /// distance are the same number.
    ///
    /// Both operands are sorted, deduplicated id lists, so this is a single linear merge: no hashing and
    /// no allocation. The history-scan passes call it once per remembered frame, so a per-call heap set
    /// there dominated the sweep; a merge over two small slices costs nothing to set up.
    fn spatial_distance(observed: &[NeuronId], config: &[NeuronId]) -> u32 {
        let (mut i, mut j, mut inter) = (0usize, 0usize, 0u32);
        while i < observed.len() && j < config.len() {
            match observed[i].cmp(&config[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => { inter += 1; i += 1; j += 1; }
            }
        }
        (observed.len() + config.len()) as u32 - 2 * inter
    }

    /// The neighbors present in both sorted lists — a linear merge, used to build a serving child's bid
    /// cover (`O ∩ config`). The result stays sorted.
    fn sorted_intersection(a: &[NeuronId], b: &[NeuronId]) -> Vec<NeuronId> {
        let (mut i, mut j) = (0usize, 0usize);
        let mut out = Vec::new();
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => { out.push(a[i]); i += 1; j += 1; }
            }
        }
        out
    }

    /// A stored configuration as a SORTED list of its live neighbor ids. Used for both the normal and
    /// children — every routing-table entry is a context configuration of the same shape. Configurations
    /// are frozen after seed/mint, so this is built fresh each pass with no cache to keep in sync, and the
    /// sort is what lets [spatial_distance] merge instead of hash.
    fn config_sorted(context: &SpatialContext) -> Vec<NeuronId> {
        let mut v: Vec<NeuronId> = context.entries().iter().filter(|(_, &s)| s > 0.0).map(|(&id, _)| id).collect();
        v.sort_unstable();
        v
    }

    /// All of this neuron's routing entries — its normal and its children — as (server, sorted
    /// configuration) pairs. Every entry is a stored **context** configuration; the normal is one of
    /// them (docs/algorithm.md, "The normal"), competing on distance and able to serve, but with no
    /// pattern neuron, never propagating, and never deleted.
    fn spatial_entries(&self) -> Vec<(SpatialServer, Vec<NeuronId>)> {
        self.spatial_routing_table.iter()
            .map(|(&server, entry)| (server, Self::config_sorted(&entry.context)))
            .collect()
    }

    /// On a distance tie, does `candidate` outrank the current best `incumbent`? The normal outranks
    /// any child (it is free and never propagates, so grouping consolidates onto it); among children
    /// the smaller id outranks (docs/algorithm.md, "Election": oldest id). Order-independent, so the
    /// children's hash-map iteration order does not change the winner.
    fn spatial_outranks(candidate: SpatialServer, incumbent: SpatialServer) -> bool {
        match (candidate, incumbent) {
            (SpatialServer::Normal, SpatialServer::Child(_)) => true,
            (SpatialServer::Child(c), SpatialServer::Child(i)) => c < i,
            _ => false,
        }
    }

    /// Route an observation to the entry that serves it — the closest, ties broken by [spatial_outranks]
    /// — and its distance. One pass over all entries, the normal among them. Routing needs only the
    /// winner; the second-best is computed by the delete pass alone, so it is not returned here.
    ///
    /// Kept for the debug server invariant and tests — the hot per-frame path uses
    /// [spatial_route_cached] over interned ids instead.
    #[cfg(any(test, debug_assertions))]
    fn spatial_route(observed: &[NeuronId], entries: &[(SpatialServer, Vec<NeuronId>)]) -> (SpatialServer, u32) {
        let mut best: Option<(SpatialServer, u32)> = None;
        for (server, config) in entries {
            let d = Self::spatial_distance(observed, config);
            match best {
                Some((bserver, bd)) if d < bd || (d == bd && Self::spatial_outranks(*server, bserver)) => best = Some((*server, d)),
                None => best = Some((*server, d)),
                _ => {}
            }
        }
        best.expect("spatial_route: a neuron always has a normal entry")
    }

    /// Intern every routing entry's configuration into the distance cache (once per entry lifetime,
    /// since configs are frozen) and return the `(server, config id)` list the hot passes iterate.
    /// This replaces the per-frame `spatial_entries` sort-and-allocate: an entry keeps its interned id
    /// on `config_id` and only re-derives it when its context is mutated (which resets it to
    /// `DIST_SENTINEL`). Ids index [SpatialDistanceCache::configs], so a config's length (its storage
    /// cost) is read from there without touching the entry again.
    fn spatial_entry_ids(&mut self) -> Vec<(SpatialServer, u32)> {
        let Self { spatial_routing_table, spatial_dist_cache, .. } = self;
        let mut out = Vec::with_capacity(spatial_routing_table.len());
        for (&server, entry) in spatial_routing_table.iter_mut() {
            if entry.config_id == DIST_SENTINEL {
                let cfg = Self::config_sorted(&entry.context);
                entry.config_id = spatial_dist_cache.intern(&cfg);
            }
            out.push((server, entry.config_id));
        }
        out
    }

    /// Cached counterpart of [spatial_route]: same nearest-wins with the same [spatial_outranks]
    /// tie-break, but every distance is a lookup keyed on the interned observation and entry ids.
    fn spatial_route_cached(&mut self, observed_id: u32, entries: &[(SpatialServer, u32)]) -> (SpatialServer, u32) {
        let cache = &mut self.spatial_dist_cache;
        let mut best: Option<(SpatialServer, u32)> = None;
        for &(server, cid) in entries {
            let d = cache.distance(observed_id, cid);
            match best {
                Some((bserver, bd)) if d < bd || (d == bd && Self::spatial_outranks(server, bserver)) => best = Some((server, d)),
                None => best = Some((server, d)),
                _ => {}
            }
        }
        best.expect("spatial_route_cached: a neuron always has a normal entry")
    }

    /// Pull the configurations a freshly minted child now serves onto it. Used when a child is minted. A
    /// new entry can only LOWER a configuration's service distance, never raise it, so a configuration
    /// moves to the newborn exactly when the newborn is strictly closer than its cached best. Ties never
    /// move: the newborn holds the largest id, and the tie-break gives the incumbent (the normal, or any
    /// older child) the win — hence `<`, not `<=`. One merge per DISTINCT configuration against the
    /// newborn's config, where re-routing against every entry would be a factor of E slower. Deletion does
    /// NOT use this — removing a child can only change the winner of the configurations that child served,
    /// so it reassigns just those (see [remove_spatial_child]).
    fn reassign_after_mint(&mut self, child_id: NeuronId, config: &[NeuronId]) {
        let Self { spatial_history, spatial_dist_cache, .. } = self;
        let child_cid = spatial_dist_cache.intern(config);
        for (observed, ev) in spatial_history.iter_mut() {
            let oid = spatial_dist_cache.intern(observed);
            let d = spatial_dist_cache.distance(oid, child_cid);
            if d < ev.best_distance {
                ev.server = SpatialServer::Child(child_id);
                ev.best_distance = d;
            }
        }
    }

    /// Debug-only invariant: every record's stored `server` and `best_distance` must match a fresh route
    /// against the current entries. The delete pass groups frames by `server` and reads `best_distance`
    /// as the served cost, so a stale value would mis-attribute a frame's benefit.
    #[cfg(debug_assertions)]
    fn debug_assert_spatial_servers(&self) {
        let entries = self.spatial_entries();
        for (observed, ev) in self.spatial_history.iter() {
            let (server, best) = Self::spatial_route(observed, &entries);
            debug_assert_eq!(ev.server, server, "config server stale");
            debug_assert_eq!(ev.best_distance, best, "config best_distance stale");
        }
    }

    /// The one test's delete half — the SAME test as the add half, so nothing can be deleted then
    /// immediately re-added in a loop (docs/algorithm.md, "The one test"). For each child,
    /// benefit = Σ over the frames it serves of [next(O) − d(O, child)]; cost = 1 + |config|. `next(O)`
    /// — the closest other entry, the fallback if the child were gone — is computed here, and only for
    /// the frames a child served: the normal serves the rest and is never deleted, so those frames
    /// never need a second-best. A child that no longer spares the window enough error to cover its
    /// storage is retired — at most one per frame, the widest failure, because deletions interact.
    ///
    /// Reference implementation over freshly-merged distances; the per-frame path is
    /// [spatial_delete_pass_cached], which reads the same distances from the cache. Kept for the tests
    /// that assert the settled-state behaviour directly.
    #[cfg(test)]
    fn spatial_delete_pass(&mut self, serving: Option<NeuronId>) -> Option<NeuronId> {
        // The table always holds the normal; nothing to retire unless there is at least one child.
        if self.spatial_routing_table.len() <= 1 { return None; }
        #[cfg(debug_assertions)]
        self.debug_assert_spatial_servers();
        let entries = self.spatial_entries();

        // For each configuration a child served, its distance to that child (best) is the cached
        // `best_distance`; only next(O) — the closest OTHER entry, the fallback if the child were gone —
        // is computed here. The difference, times how many frames the configuration contributed, is the
        // error the child spares the window. Configurations the normal served are skipped without
        // touching the entries — they cannot support a deletion.
        let mut benefit: FxHashMap<NeuronId, f64> = FxHashMap::default();
        for (observed, ev) in self.spatial_history.iter() {
            let SpatialServer::Child(pid) = ev.server else { continue };
            let best = ev.best_distance; // cached distance to the child that served this configuration
            let mut next = u32::MAX; // distance to the closest other entry — next(O)
            for (server, config) in &entries {
                if *server == SpatialServer::Child(pid) { continue; } // its own server; best is cached
                next = next.min(Self::spatial_distance(observed, config));
            }
            *benefit.entry(pid).or_insert(0.0) += (next - best) as f64 * ev.frames.len() as f64;
        }

        // Retire the child that fails its storage by the widest margin, never the one serving this
        // frame (it has fired and been handed to the apex — releasing it now would dangle). The
        // normal is among `entries` but is never retired: it owns no cost line, so it never fails.
        let mut worst: Option<(NeuronId, f64)> = None;
        for (server, config) in &entries {
            let SpatialServer::Child(pid) = server else { continue };
            if Some(*pid) == serving { continue; }
            let cost = 1.0 + config.len() as f64;
            let margin = benefit.get(pid).copied().unwrap_or(0.0) - cost;
            if margin < 0.0 && worst.map_or(true, |(_, m)| margin < m) {
                worst = Some((*pid, margin));
            }
        }

        let (pid, _) = worst?;
        self.remove_spatial_child(pid);
        Some(pid)
    }

    /// Cached counterpart of [spatial_delete_pass]: identical benefit/margin computation and identical
    /// choice of the widest-failing child, but `next(O)` — the per-child, per-config scan over all other
    /// entries, the pass's `O(D · E)` core — is read from the distance cache keyed on interned ids
    /// instead of recomputing a merge for each pair every frame. `entries` is the interned `(server,
    /// config id)` list already built this frame by `spatial_entry_ids`; config lengths (the storage
    /// cost) come from the cache's stored configs.
    ///
    /// Returns the child to retire as a **decision only** — it does not remove it. The frame pass applies
    /// it inline for a pure normal-serve; for a request the commit applies it, and only if the request won.
    fn spatial_delete_candidate(&mut self, serving: Option<NeuronId>, entries: &[(SpatialServer, u32)]) -> Option<NeuronId> {
        // The table always holds the normal; nothing to retire unless there is at least one child.
        if self.spatial_routing_table.len() <= 1 { return None; }
        #[cfg(debug_assertions)]
        self.debug_assert_spatial_servers();

        {
            let Self { spatial_history, spatial_dist_cache, .. } = self;

            // For each configuration a child served, next(O) is the closest OTHER entry — computed from
            // cached distances. best is the stored `best_distance`. The gap, times the config's
            // multiplicity, is the error the child spares the window.
            let mut benefit: FxHashMap<NeuronId, f64> = FxHashMap::default();
            for (observed, ev) in spatial_history.iter() {
                let SpatialServer::Child(cpid) = ev.server else { continue };
                let oid = spatial_dist_cache.intern(observed);
                let best = ev.best_distance;
                let mut next = u32::MAX;
                for &(server, cid) in entries {
                    if server == SpatialServer::Child(cpid) { continue; } // its own server; best is stored
                    next = next.min(spatial_dist_cache.distance(oid, cid));
                }
                *benefit.entry(cpid).or_insert(0.0) += (next - best) as f64 * ev.frames.len() as f64;
            }

            // Retire the child failing its storage by the widest margin, never the one serving this frame.
            let mut worst: Option<(NeuronId, f64)> = None;
            for &(server, cid) in entries {
                let SpatialServer::Child(cpid) = server else { continue };
                if Some(cpid) == serving { continue; }
                let cost = 1.0 + spatial_dist_cache.configs[cid as usize].len() as f64;
                let margin = benefit.get(&cpid).copied().unwrap_or(0.0) - cost;
                if margin < 0.0 && worst.map_or(true, |(_, m)| margin < m) {
                    worst = Some((cpid, margin));
                }
            }
            worst.map(|(p, _)| p)
        }
    }

    /// The one test's add half: the candidate is this frame's observation O. Benefit = Σ over the
    /// whole history of [distance to the frame's server − distance to the candidate], counted only on
    /// the frames the candidate is strictly closer (the frames it would win); cost = 1 + |O|. Mint when
    /// benefit ≥ cost. This is the design's expensive step — but measuring the candidate against every
    /// distinct configuration rather than every remembered frame is exactly what the histogram buys
    /// (docs/algorithm.md, "Risks": the add pass).
    ///
    /// Reference implementation over freshly-merged distances; the per-frame path is
    /// [spatial_add_pass_pays]. Kept for the tests that assert the settled state directly.
    #[cfg(test)]
    fn spatial_add_pass(&mut self, observed: &[NeuronId]) -> bool {
        #[cfg(debug_assertions)]
        self.debug_assert_spatial_servers();
        // Each configuration's current best distance is the cached `best_distance` — no entry lookup, no
        // recompute. Only the candidate distance is measured, one linear merge per distinct configuration
        // over sorted slices, weighted by how many frames that configuration contributed.
        let cost = 1.0 + observed.len() as f64;
        let mut benefit = 0.0;
        for (obs, ev) in self.spatial_history.iter() {
            let d_cand = Self::spatial_distance(obs, observed);
            if d_cand < ev.best_distance { benefit += (ev.best_distance - d_cand) as f64 * ev.frames.len() as f64; }
        }
        // A child at O pays for its storage — the neuron would ask the thalamus to mint it.
        benefit >= cost
    }

    /// Does a new child at `O` pay for its storage? The add half of the one test as a decision: benefit
    /// sums, over distinct history configurations, the error a child at `O` would spare (cached
    /// `best_distance` minus candidate distance), plus this frame's own contribution `served_distance`
    /// (the current frame is not yet recorded — a request records nothing until it commits). Cost `1+|O|`.
    fn spatial_add_pass_pays(&mut self, observed_id: u32, observed: &[NeuronId], served_distance: u32) -> bool {
        #[cfg(debug_assertions)]
        self.debug_assert_spatial_servers();
        let cost = 1.0 + observed.len() as f64;
        let mut benefit = served_distance as f64;
        {
            let Self { spatial_history, spatial_dist_cache, .. } = self;
            for (obs, ev) in spatial_history.iter() {
                let oid = spatial_dist_cache.intern(obs);
                let d_cand = spatial_dist_cache.distance(oid, observed_id);
                if d_cand < ev.best_distance { benefit += (ev.best_distance - d_cand) as f64 * ev.frames.len() as f64; }
            }
        }
        benefit >= cost
    }

    /// Remove a retired child from this neuron's local routing structures. The thalamus releases the
    /// pattern neuron and scrubs its cross-neuron references from the id in `deleted_children`.
    fn remove_spatial_child(&mut self, pattern_id: NeuronId) {
        if let Some(entry) = self.spatial_routing_table.remove(&SpatialServer::Child(pattern_id)) {
            let ctx_ids: Vec<NeuronId> = entry.context.entries().keys().copied().collect();
            for ctx_id in ctx_ids {
                if let Some(set) = self.spatial_context_index.get_mut(&ctx_id) {
                    set.remove(&pattern_id);
                    if set.is_empty() { self.spatial_context_index.remove(&ctx_id); }
                }
            }
            // Only the configurations this child served can change hands — removing it cannot change the
            // winner of a configuration it did not win. Reassign just those to their next-best (now their
            // best), and refresh their cached distance to that new server. The child is already gone from
            // the table, so `spatial_entry_ids` returns exactly the surviving entries to route against.
            let entries = self.spatial_entry_ids();
            let Self { spatial_history, spatial_dist_cache, .. } = self;
            for (observed, ev) in spatial_history.iter_mut() {
                if ev.server != SpatialServer::Child(pattern_id) { continue; }
                let oid = spatial_dist_cache.intern(observed);
                let mut best: Option<(SpatialServer, u32)> = None;
                for &(server, cid) in &entries {
                    let d = spatial_dist_cache.distance(oid, cid);
                    match best {
                        Some((bserver, bd)) if d < bd || (d == bd && Self::spatial_outranks(server, bserver)) => best = Some((server, d)),
                        None => best = Some((server, d)),
                        _ => {}
                    }
                }
                let (server, b) = best.expect("remove_spatial_child: a neuron always has a normal entry");
                ev.server = server;
                ev.best_distance = b;
            }
        }
    }

    /// Temporal frame processing — per-age iteration over the sliding window. Matching
    /// queries `temporal_routing_table` and produces d>0 sequence votes from `temporal_connections`.
    /// Connections are learned at the TOP of the frame: connections[age] is updated before
    /// the vote step which reads connections[age+1], so learn-then-vote doesn't echo back this
    /// frame's actives.
    pub fn process_temporal_frame(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        learning_work: Option<&TemporalLearningWork>,
        current_frame: FrameNumber,
    ) -> ProcessFrameResult {
        let mut timings = NeuronOpTimings::default();

        if let Some(l) = learning_work {

            // Record before voting so this frame's vote threshold reflects the latest accuracy sample.
            for fb in &l.error_feedback { self.record_temporal_error(fb.age, fb.error_rate); }

            // Safe to learn before voting: the vote reads connections[age+1]; this writes connections[age>0].
            let t = std::time::Instant::now();
            self.learn_temporal_connections(age_states, &l.neighbors);
            timings.learn_connections = t.elapsed().as_secs_f64();
        }

        // Match temporal patterns against the level_context if we have one and eligible ages.
        let t = std::time::Instant::now();
        let RecognizeResult { matches, context_ref_updates: match_refs } = self.recognize_temporal_patterns(age_states, memory_depth, level_context, new_error_pattern_ids, current_frame, &mut timings);
        timings.recognize_patterns = t.elapsed().as_secs_f64();

        // Install pre-created temporal error-correction patterns as children and emit their contextRef adds.
        // Empty on a frozen pass, so correct_errors is a no-op there.
        let corrections: &[Correction] = learning_work.map_or(&[], |l| &l.corrections);
        let t = std::time::Instant::now();
        let CorrectResult { correction_activations, context_ref_updates: correction_refs } = self.correct_errors(corrections, current_frame);
        timings.correct_errors = t.elapsed().as_secs_f64();

        // Cast temporal votes for each eligible age, suppressing any ages that activated a pattern.
        let t = std::time::Instant::now();
        let votes = self.generate_temporal_votes(age_states, memory_depth, level_context, &matches, &correction_activations);
        timings.generate_votes = t.elapsed().as_secs_f64();

        let mut context_ref_updates = match_refs;
        context_ref_updates.extend(correction_refs);
        ProcessFrameResult { matches, correction_activations, context_ref_updates, votes, timings }
    }

    // ── Pattern recognition ──────────────────────────────────────────────────

    /// Find matching patterns for this parent neuron given the observed level context.
    /// For each age that matches, the matched pattern's context is refined immediately.
    /// Cross-neuron contextRef additions and removals are returned for delivery in a separate
    /// post-match phase. If the same child pattern matches at multiple ages for this parent,
    /// all refined matches are returned in order so side effects remain identical, but only
    /// the first match is flagged as an activation candidate. Matching is skipped entirely for
    /// new error patterns, when level_context is None/empty, or when no age is eligible.
    /// Eligible ages: non-activated and not the oldest age in the sliding window (where no
    /// future context exists to match against).
    fn recognize_temporal_patterns(
        &mut self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        current_frame: FrameNumber,
        timings: &mut NeuronOpTimings,
    ) -> RecognizeResult {
        let mut matches = Vec::new();
        let context_ref_updates = Vec::new();

        // Warmup gate: skip pattern recognition until the context window has had a chance to fill up.
        // Without this, patterns whose stored contexts include entries at distances > current_frame are unfairly penalized.
        if (current_frame as u32) < self.context_length {
            return RecognizeResult { matches, context_ref_updates };
        }

        let ctx = match level_context {
            Some(c) if c.size() > 0 => c,
            _ => return RecognizeResult { matches, context_ref_updates },
        };
        let mut activated_pattern_ids = FxHashSet::default();

        // sort the ages so that we loop them in order
        let mut ages: Vec<Distance> = age_states.keys().copied().collect();
        ages.sort_unstable();

        for age in ages {
            let state = &age_states[&age];
            // Skip the oldest age — there's no future context to match against.
            if state.activated_pattern_id.is_some() { continue; }
            if age == memory_depth - 1 { continue; }

            // active ages are processed in ascending order (most recent first). The first age that
            // produces a match at that age is refined and preserved. More recent ages tend to have
            // the richest available context, so they are processed first.
            let best = match self.find_best_pattern_match_at_age(ctx, age, new_error_pattern_ids, current_frame, timings) {
                Some(b) => b,
                None => continue, // try older age if there is a match
            };

            // activate the matched pattern if it was not elected to be activated already
            let activate = !activated_pattern_ids.contains(&best.pattern_id);
            activated_pattern_ids.insert(best.pattern_id);

            // Strengthen child activation automatically here, but only in learning mode.
            // Non-learning eval still activates the pattern but does not extend its life.
            let death_frame = if activate && self.learning { self.strengthen_child_activation(best.pattern_id, current_frame) } else { None };

            // include the best match for the age in the results
            matches.push(PatternMatch {
                pattern_id: best.pattern_id,
                age: best.age,
                activate,
                death_frame,
            });
        }
        RecognizeResult { matches, context_ref_updates }
    }

    /// Find the best matching temporal pattern for a specific active age.
    fn find_best_pattern_match_at_age(&self, observed: &TemporalContext, age: Distance, exclude_ids: &FxHashSet<NeuronId>, current_frame: FrameNumber, timings: &mut NeuronOpTimings) -> Option<PartialMatch> {
        let mut best: Option<PartialMatch> = None;

        // Use the inverted index to narrow the search to child patterns that share at least one
        // exact neuron/distance entry with the observed context at this active age.
        let t = std::time::Instant::now();
        let candidate_ids = self.get_pattern_candidates_at_age(observed, age);
        timings.recognize_candidate_search += t.elapsed().as_secs_f64();
        if candidate_ids.is_empty() { return None; }

        // Recognition strictness is the adaptive merge (1 − E) for this age's bucket — same value correction reads.
        let merge_threshold = self.grouping_merge_threshold(self.temporal_error_stats.get(age as usize).and_then(|s| s.as_ref()));
        let eval_start = std::time::Instant::now();
        let candidate_count = candidate_ids.len();
        // go through the candidate patterns and find the best match
        for pattern_id in candidate_ids {

            // check if pattern is still alive (skip functionally dead patterns)
            if self.get_child_effective_activation_strength(pattern_id, current_frame) <= 0.0 { continue; }

            // get the context of the pattern
            let entry = self.temporal_routing_table.get(&pattern_id)
                .unwrap_or_else(|| panic!("Cannot find context for pattern: {}", pattern_id));

            // context.match_observed() handles the full scoring and threshold check;
            // the index only decides which child patterns are worth evaluating.
            // exclude_ids masks out brand-new neurons so they don't count as "novel" misses.
            let m = match entry.context.match_observed(observed, age, merge_threshold, Some(exclude_ids)) {
                Some(m) => m,
                None => continue, // nothing to do if there is no match
            };

            // if there is already a best match, check if this match is better
            if let Some(ref b) = best {

                // if the previous best is better than this match, skip
                if m.score < b.score { continue; }

                // to preserve determinism when multiple patterns achieve the exact same score, explicitly tie-break using pattern_id
                if m.score == b.score && pattern_id > b.pattern_id { continue; } // prefer smaller id (older)
            }

            // store the best match
            best = Some(PartialMatch {
                pattern_id,
                age,
                score: m.score,
            });
        }
        timings.recognize_candidate_eval += eval_start.elapsed().as_secs_f64();
        timings.recognize_candidates_evaluated += candidate_count as u64;
        best
    }

    /// Find candidate child patterns for a specific active age using the inverted index.
    /// Candidate patterns must share at least one exact neuron/distance context entry with the
    /// observed context after converting absolute observed ages into pattern-relative distances.
    ///
    /// The observed context is keyed by absolute age within the current frame snapshot, while each
    /// stored pattern context is keyed by distance relative to the parent's own activation age.
    /// For a parent active at `age`, an observed entry at absolute distance `d` maps to a pattern
    /// entry at relative distance `d - age`.
    ///
    /// Missing index entries are expected here: they just mean the observed neuron/distance pair is
    /// not referenced by any child pattern and therefore contributes no candidates.
    fn get_pattern_candidates_at_age(&self, observed: &TemporalContext, age: Distance) -> FxHashSet<NeuronId> {
        let mut candidates = FxHashSet::default();

        // Temporal context entries are at strictly positive relative distances — context neurons must be
        // older than the parent neuron itself.
        let min_distance: i64 = 1;

        for (&neuron_id, distance_map) in observed.entries() {

            // First narrow by exact context neuron ID. If this neuron does not appear in the index,
            // no child pattern references it anywhere in this parent's routing table.
            let indexed_distances = match self.temporal_context_index.get(&neuron_id) {
                Some(d) => d,
                None => continue,
            };

            for &absolute_distance in distance_map.keys() {

                // Convert the observed absolute age into the pattern-relative distance used by the
                // routing table and inverted index. The minimum admissible distance depends on phase
                // (temporal: ≥1; spatial: ≥0).
                let pattern_distance = absolute_distance as i64 - age as i64;
                if pattern_distance < min_distance { continue; }
                let pattern_distance = pattern_distance as Distance;

                // Then narrow by exact relative distance. Missing entries here are also expected and
                // simply mean no child pattern references this neuron at this particular distance.
                let pattern_ids = match indexed_distances.get(&pattern_distance) {
                    Some(p) => p,
                    None => continue,
                };

                // A candidate only needs one exact neuron/distance overlap to be worth a full score.
                for &pattern_id in pattern_ids { candidates.insert(pattern_id); }
            }
        }
        candidates
    }

    // ── Error correction ─────────────────────────────────────────────────────

    /// Install pre-created error-correction pattern neurons as children at the given ages.
    /// Each install adds a child pattern entry to the routing table and emits contextRef adds
    /// for every context entry so target neurons can track this parent.
    fn correct_errors(&mut self, corrections: &[Correction], current_frame: FrameNumber) -> CorrectResult {
        let mut correction_activations = Vec::new();
        let mut context_ref_updates = Vec::new();
        for correction in corrections {

            // add the pattern to the routing table
            let death_frame = self.add_temporal_pattern(correction.pattern_id, &correction.context_entries, current_frame);

            // add the pattern to the activations to be returned with its death frame
            correction_activations.push(CorrectionActivation { pattern_id: correction.pattern_id, age: correction.age, death_frame });

            // also update the context reference updates to be returned for the new patterns
            for entry in &correction.context_entries {
                context_ref_updates.push(TemporalContextRefUpdate { neuron_id: entry.neuron_id, distance: entry.distance, parent_id: 0 });
            }
        }
        CorrectResult { correction_activations, context_ref_updates }
    }

    // ── Vote generation ──────────────────────────────────────────────────────

    /// Cast one temporal vote per non-suppressed age; get_suppressed_ages owns the full inhibition rule.
    /// Runs for new error patterns too so their state.context gets populated for next-frame corrections.
    /// The per-age context is reshaped from level_context locally — context_by_age is a pure reshape
    /// (no extra info), so it's derived here instead of being shipped over the wire.
    fn generate_temporal_votes(
        &self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
    ) -> Vec<AgeVotes> {

        // Resolve every inhibited age once, then keep only the ages this neuron will actually vote on.
        let suppressed_ages = self.get_suppressed_ages(age_states, memory_depth, matches, correction_activations);
        let voting_ages: Vec<Distance> = age_states.keys().copied().filter(|age| !suppressed_ages.contains(age)).collect();

        // pre-bucket level_context into those voting ages so each cast picks its per-age context up with one lookup
        let context_by_age = self.derive_context_by_age(level_context, &voting_ages);

        // cast one vote per voting age
        let mut votes = Vec::with_capacity(voting_ages.len());
        for age in voting_ages {

            // Temporal votes predict the next frame via temporal_connections[age+1].
            let cast_votes = self.vote(age);
            votes.push(AgeVotes {
                age,
                votes: cast_votes,
                context: context_by_age.get(age as usize).cloned().unwrap_or_default(),
                threshold: self.get_temporal_error_threshold(age),
            });
        }
        votes
    }

    /// Collect every age whose vote is inhibited this frame, so generate_temporal_votes can skip them in one test.
    /// An age is suppressed when any of the following holds:
    /// - a recognized pattern activated at that age this frame — the firing parent represents it, so the subsumed
    ///   neuron must not also vote (only the activation candidate inhibits, not non-activating refinement matches),
    /// - an error-correction pattern was installed at that age this frame — the correction now owns that prediction,
    /// - the neuron was already marked subsumed at that age on a prior frame (activated_pattern_id is set),
    /// - it is the oldest slot in the window, with no future frame to vote toward.
    /// The oldest-slot rule applies only for context_length > 1; at context_length == 1 the single age 0 is the
    /// prediction-bearing age for single-frame episodes and must vote, otherwise process_frame emits no votes at all.
    fn get_suppressed_ages(
        &self,
        age_states: &FxHashMap<Distance, AgeState>,
        memory_depth: u32,
        matches: &[PatternMatch],
        correction_activations: &[CorrectionActivation],
    ) -> FxHashSet<Distance> {
        let mut suppressed_ages = FxHashSet::default();

        // if the neuron decided to activate a child pattern in this frame, its vote is suppressed in this frame
        for m in matches { if m.activate { suppressed_ages.insert(m.age); } }

        // if the neuron had a bad inference last frame from an age, and it needs to be corrected, suppress the vote for that age
        for c in correction_activations { suppressed_ages.insert(c.age); }

        // loop over the ages and check
        for (&age, state) in age_states {

            // if the neuron activated a child pattern in an age in a previous frame, that age stays suppressed
            if state.activated_pattern_id.is_some() { suppressed_ages.insert(age); continue; }

            // oldest age has no connections for next frame to vote, except age 0 at context_length 1, used for MNIST tests.
            if self.context_length > 1 && age >= memory_depth - 1 { suppressed_ages.insert(age); }
        }

        suppressed_ages
    }

    /// Reshape level_context into per-voting-age buckets, building only the supplied voting ages.
    /// For each entry at context-age `ctx_age`, emits it into every voting age < ctx_age with distance = ctx_age - age.
    /// Indexed by voting age so generate_temporal_votes picks up each per-age context with a single lookup —
    /// no per-age scan, no `ctx_age > age` branch. Suppressed ages get no bucket since their vote is never cast.
    /// Pure reshape kept inside the neuron to save per-frame MPI traffic.
    fn derive_context_by_age(&self, level_context: Option<&TemporalContext>, voting_ages: &[Distance]) -> Vec<Vec<ContextRefEntry>> {
        let mut context_by_age: Vec<Vec<ContextRefEntry>> = Vec::new();
        let ctx = match level_context {
            Some(c) => c,
            None => return context_by_age,
        };
        let voting: FxHashSet<Distance> = voting_ages.iter().copied().collect();
        for (&neuron_id, distance_map) in ctx.entries() {
            for &ctx_age in distance_map.keys() {
                for a in 0..ctx_age {
                    if !voting.contains(&a) { continue; }
                    let idx = a as usize;
                    if idx >= context_by_age.len() { context_by_age.resize_with(idx + 1, Vec::new); }
                    context_by_age[idx].push(ContextRefEntry { neuron_id, distance: ctx_age - a });
                }
            }
        }
        context_by_age
    }

    // ── Connection learning ──────────────────────────────────────────────────

    /// Update connections based on the currently observed actives. For each active age > 0:
    /// upsert a connection per active (create-or-strengthen + alt-action). Rewards are
    /// pre-resolved thalamus-side (0 for events, observed value for actions). Skipped entirely
    /// for new error patterns (they were just created this frame and have nothing yet to reinforce).
    ///
    /// Note: connections are no longer negatively reinforced. Previously, predictions at this
    /// distance that didn't occur were weakened (with eventual deletion at strength=0). This
    /// produced periodic discrete-death bursts during multi-episode training — a connection
    /// drifts down by ~1/episode until it finally hits 0, then dies abruptly, shifting the
    /// neuron's vote profile and triggering a cascade of error-pattern creation. For deterministic
    /// memorization scenarios we want predictions that ever occurred to remain available;
    /// non-occurrences should not erase them. Mirrors the action-connection behaviour, which
    /// was already kept-only-strengthen.
    /// Temporal connection learning — strengthen temporal_connections[age] for every active age>0.
    /// A neuron active at age=k learns a k-distance prediction toward each current actives (which
    /// are at age=0 in `actives`).
    fn learn_temporal_connections(&mut self, age_states: &FxHashMap<Distance, AgeState>, actives: &[ActiveNeuron]) {
        let ages: Vec<Distance> = age_states.keys().copied().collect();
        for age in ages {
            if age == 0 { continue; }
            for active in actives {
                self.upsert_connection(age, active.id, active.channel_id, active.reward);
            }
        }
    }

    /// Find an alternative action for a channel that hasn't been tried yet.
    pub(crate) fn find_alternative_action(&self, distance: Distance, channel_id: ChannelId, current_action_id: NeuronId) -> Option<NeuronId> {
        if let Some(action_ids) = self.channel_action_ids.get(&channel_id) {
            for &alt_neuron_id in action_ids {
                if alt_neuron_id != current_action_id && !self.has_connection(distance, alt_neuron_id) {
                    return Some(alt_neuron_id);
                }
            }
        }
        None
    }
}

// ── Serialization structs ────────────────────────────────────────────────────

/// Intermediate struct used during find_best_pattern_match_at_age before
/// the full PatternMatch is assembled.
struct PartialMatch {
    pattern_id: NeuronId,
    age: Distance,
    score: f64,
}

#[derive(Debug, Clone)]
pub struct SerializedNeuron {
    pub id: NeuronId,
    pub pattern_forget_rate: f64,
    pub connections: Vec<SerializedConnection>,
    pub children: Vec<SerializedChild>,
    pub context_refs: Vec<SerializedContextRef>,
    pub error_stats: Vec<SerializedErrorStats>,
    /// The normal's stored context configuration — its member neighbor ids. A routing-table entry
    /// like a child's context, just with no pattern neuron (docs/algorithm.md, "The normal").
    pub normal_context: Vec<NeuronId>,
    /// The neuron's history window — the frames it was active for, each with the neighborhood observed
    /// and which entry served it. This is the evidence the one test reads, so it has to survive a
    /// restore: dropping it resets the neuron's accumulated benefit to zero and its structure stops
    /// growing, which is not what continuing to train from a snapshot is supposed to mean.
    pub spatial_history: Vec<SerializedHistoryRecord>,
}

/// One remembered frame, flattened for the snapshot. `served_child` is the child pattern that served
/// the frame, or `None` when the normal did (the normal has no pattern id of its own).
#[derive(Debug, Clone)]
pub struct SerializedHistoryRecord {
    pub frame: FrameNumber,
    pub served_child: Option<NeuronId>,
    pub observed: Vec<NeuronId>,
}

#[derive(Debug, Clone)]
pub struct SerializedConnection {
    pub distance: Distance,
    pub to_neuron_id: NeuronId,
    pub strength: Strength,
    pub reward: Reward,
}

#[derive(Debug, Clone)]
pub struct SerializedChild {
    pub pattern_id: NeuronId,
    /// true = spatial routing entry (d=0 co-activation), false = temporal routing entry (d>0 sequence).
    /// Restore needs this discriminator because an empty-context spatial child is otherwise
    /// indistinguishable from a temporal one, and the back-compat shims route unconditionally to temporal.
    pub spatial: bool,
    pub activation_strength: f64,
    pub last_activation_frame: FrameNumber,
    pub context: Vec<ContextEntry>,
}

#[derive(Debug, Clone)]
pub struct SerializedContextRef {
    pub parent_id: NeuronId,
    /// true = spatial context ref (distances is the placeholder [0]), false = temporal.
    pub spatial: bool,
    pub distances: Vec<Distance>,
}

#[derive(Debug, Clone)]
pub struct SerializedErrorStats {
    pub age: Distance,
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_neuron(id: NeuronId) -> Neuron {
        Neuron::new(id, 0.01, 0.9, GroupMode::Static, FxHashMap::default(), 10, true, 100)
    }

    fn make_neuron_with_actions(id: NeuronId, channel_id: ChannelId, action_ids: Vec<NeuronId>) -> Neuron {
        let mut channel_actions = FxHashMap::default();
        channel_actions.insert(channel_id, action_ids);
        Neuron::new(id, 0.01, 0.9, GroupMode::Static, channel_actions, 10, true, 100)
    }

    #[test]
    fn test_create_and_has_connection() {
        let mut n = make_neuron(1);
        assert!(!n.has_connection(2, 10));
        n.create_connection(2, 10, 1.0, 0.0);
        assert!(n.has_connection(2, 10));
        assert!(!n.has_connection(2, 11));
        assert!(!n.has_connection(3, 10));
    }

    #[test]
    fn test_strengthen_connection_updates_reward() {
        let mut n = make_neuron(1);
        n.create_connection(1, 10, 1.0, 1.0);
        n.strengthen_connection(1, 10, 0.0);
        // strength should be 2, reward should be smoothed: (1/2)*0 + (1/2)*1 = 0.5
        let conn = &n.temporal_connections[1][&10];
        assert_eq!(conn.strength, 2.0);
        assert!((conn.reward - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vote_returns_connections_at_age_plus_one() {
        let mut n = make_neuron(1);
        n.create_connection(3, 10, 2.0, 0.5);
        n.create_connection(3, 11, 1.0, 0.0);
        let votes = n.vote(2); // age=2 → distance=3
        assert_eq!(votes.len(), 2);
    }

    #[test]
    fn test_vote_empty_when_no_connections() {
        let n = make_neuron(1);
        let votes = n.vote(5);
        assert!(votes.is_empty());
    }

    #[test]
    fn test_add_child_and_pattern() {
        let mut n = make_neuron(1);
        let context = vec![
            ContextRefEntry { neuron_id: 10, distance: 1 },
            ContextRefEntry { neuron_id: 11, distance: 2 },
        ];
        let death_frame = n.add_temporal_pattern(100, &context, 10);
        assert!(death_frame.is_some());

        // pattern should be in routing table with 2 context entries
        assert_eq!(n.temporal_routing_table[&100].context.size(), 2);

        // context index should have entries
        assert!(n.temporal_context_index.contains_key(&10));
        assert!(n.temporal_context_index.contains_key(&11));
    }

    #[test]
    fn test_lazy_decay() {
        let mut n = make_neuron(1);
        n.pattern_forget_rate = 0.1;
        n.add_temporal_child(100, 0.0);
        n.strengthen_child_activation(100, 0); // strength=1, last_frame=0

        // at frame 5: effective = 1 - 5*0.1 = 0.5
        assert!((n.get_child_effective_activation_strength(100, 5) - 0.5).abs() < 1e-10);

        // at frame 10: effective = 1 - 10*0.1 = 0.0
        assert!(n.get_child_effective_activation_strength(100, 10) <= 0.0);
        assert!(n.can_delete_child(100, 10));
    }

    #[test]
    fn test_error_threshold_static() {
        // group_threshold 0.9 → static correction threshold = 1 − 0.9 = 0.1.
        let n = make_neuron(1);
        assert!((n.get_temporal_error_threshold(0) - 0.1).abs() < 1e-10);
        assert!((n.get_temporal_error_threshold(5) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_error_threshold_dynamic_warmup() {
        let mut n = Neuron::new(1, 0.01, 0.9, GroupMode::Neutral, FxHashMap::default(), 10, true, 100);
        // fewer than ERROR_MIN_SAMPLES → falls back to the derived 1 − group_threshold = 0.1
        n.record_temporal_error(0, 0.5);
        n.record_temporal_error(0, 0.5);
        assert!((n.get_temporal_error_threshold(0) - 0.1).abs() < 1e-10); // still warmup

        // after 3 samples → uses mean
        n.record_temporal_error(0, 0.5);
        assert!((n.get_temporal_error_threshold(0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_context_refs() {
        let mut n = make_neuron(1);
        n.add_temporal_context_ref(50, 2);
        n.add_temporal_context_ref(50, 3);
        n.add_temporal_context_ref(60, 1);

        assert_eq!(n.temporal_context_refs.len(), 2);
        assert_eq!(n.temporal_context_refs[&50].len(), 2);

        n.remove_temporal_context_ref(50, 2);
        assert_eq!(n.temporal_context_refs[&50].len(), 1);

        n.remove_temporal_context_ref(50, 3);
        assert!(!n.temporal_context_refs.contains_key(&50)); // auto-cleaned
    }

    #[test]
    fn test_find_alternative_action() {
        let mut n = make_neuron_with_actions(1, 0, vec![10, 11, 12]);
        n.create_connection(1, 10, 1.0, -0.5); // already tried action 10

        // should find 11 or 12 as alternative (not 10, already connected)
        let alt = n.find_alternative_action(1, 0, 10);
        assert!(alt.is_some());
        let alt_id = alt.unwrap();
        assert!(alt_id == 11 || alt_id == 12);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut n = make_neuron(1);
        n.create_connection(1, 10, 2.0, 0.5);
        n.create_connection(2, 11, 1.0, 0.0);
        n.add_temporal_child(100, 0.0);
        n.add_temporal_context(100, 20, 1, 1.0);
        n.add_temporal_context_ref(50, 2);
        // Temporal per-age Welford error stats (age >= 1) round-trip; the spatial axis has none.
        n.record_temporal_error(1, 0.4);
        n.record_temporal_error(1, 0.6);
        // The normal is a stored context configuration; it round-trips like a child's context.
        n.restore_spatial_normal(&[7, 8, 9]);

        let s = n.serialize();
        assert_eq!(s.id, 1);
        assert_eq!(s.connections.len(), 2);
        assert_eq!(s.children.len(), 1);
        assert_eq!(s.children[0].context.len(), 1);
        assert_eq!(s.context_refs.len(), 1);
        assert_eq!(s.error_stats.len(), 1);
        let mut normal = s.normal_context.clone();
        normal.sort_unstable();
        assert_eq!(normal, vec![7, 8, 9]);
    }

    /// Phase 1 compresses: on a stationary recurring input, the level-0 configuration loop settles to
    /// a local optimum of the per-neuron description length and shrinks it below the normal-only
    /// baseline. The neuron sees one usual configuration A (the normal) and one distinct, recurring
    /// deviation B; it must keep A as the normal, mint exactly one child at B, stop churning, and end
    /// at a state where no add or delete move improves L — a machine-checked local-optimum
    /// certificate — with L strictly below "no children" (the compression the design is for).
    ///
    /// This drives a single Neuron and simulates the thalamus install: a mint request installs a child
    /// with context = O one level up (add_spatial_pattern), exactly as create/install do; a deleted
    /// child is already scrubbed from the neuron locally, so there is nothing more to do here.
    #[test]
    fn test_phase1_compresses_recurring_configuration() {
        // Horizon 20 (its own parameter now). Period-5 input, run 6 horizons.
        let horizon: i64 = 20;
        let mut n = Neuron::new(1, 0.05, 0.9, GroupMode::Static, FxHashMap::default(), 10, true, horizon as u32);
        let config_a: Vec<NeuronId> = vec![10, 11, 12];       // usual: 3 of every 5 frames -> the normal
        let config_b: Vec<NeuronId> = vec![20, 21, 22, 23];   // deviation: 2 of every 5 -> should be a child

        let mut next_pid: NeuronId = 1000;
        let total_frames = 6 * horizon;

        let mut mints = 0usize;
        let (mut mints_settled, mut deletes_settled) = (0usize, 0usize);
        for f in 0..total_frames {
            let obs: &[NeuronId] = if f % 5 < 3 { &config_a } else { &config_b };
            let settled = f >= 4 * horizon; // watch the last two horizons for churn
            let (minted, deleted) = step(&mut n, obs, f, &mut next_pid);
            if minted.is_some() {
                mints += 1;
                if settled { mints_settled += 1; }
            }
            if settled { deletes_settled += deleted.len(); }
        }

        // 1. It built structure and then settled — no churn over the last two horizons.
        assert!(mints >= 1, "expected a child to be minted for the recurring deviation B");
        assert_eq!(mints_settled, 0, "structure must stop minting once settled");
        assert_eq!(deletes_settled, 0, "structure must stop deleting once settled");

        // 2. Exactly one child at B; the normal at A. Configs come back as sorted id lists.
        let entries = n.spatial_entries();
        let children: Vec<(NeuronId, Vec<NeuronId>)> = entries.iter()
            .filter_map(|(s, cfg)| match s { SpatialServer::Child(id) => Some((*id, cfg.clone())), _ => None })
            .collect();
        assert_eq!(children.len(), 1, "exactly one child should survive for the single recurring deviation");
        assert_eq!(children[0].1, config_b, "the child's configuration must be exactly B");
        let normal_cfg = entries.iter().find_map(|(s, cfg)| matches!(s, SpatialServer::Normal).then(|| cfg.clone())).unwrap();
        assert_eq!(normal_cfg, config_a, "the normal must be the usual configuration A");

        // 3. Local-optimum certificate: no add improves, no child fails delete.
        assert!(!n.spatial_add_pass(&config_a), "no improving add at the settled state (A is the normal)");
        assert!(!n.spatial_add_pass(&config_b), "no improving add at the settled state (B is a child)");
        assert!(n.spatial_delete_pass(None).is_none(), "no child fails the delete test at the settled state");

        // 4. Compression: L_local with the settled structure is strictly below the normal-only baseline.
        // L = Σ over entries (1 + |config|) + Σ over history frames (distance to the nearest entry).
        // The normal's (1 + |config|) term is identical in both models (same normal A), so it cancels —
        // the comparison isolates "children storage + reduced service" vs "service paid every frame".
        let final_entries = n.spatial_entries();
        let mut l_settled: u32 = final_entries.iter().map(|(_, c)| 1 + c.len() as u32).sum();
        for (obs, ev) in n.spatial_history.iter() {
            l_settled += Neuron::spatial_route(obs, &final_entries).1 * ev.frames.len() as u32;
        }
        // Normal-only baseline: a single normal = the median over ALL remembered frames, no children.
        let mut counts: FxHashMap<NeuronId, u64> = FxHashMap::default();
        for (obs, ev) in n.spatial_history.iter() { for &id in obs { *counts.entry(id).or_insert(0) += ev.frames.len() as u64; } }
        let total = n.spatial_history.total_frames() as u64;
        let mut baseline_normal: Vec<NeuronId> = counts.into_iter().filter(|(_, c)| c * 2 > total).map(|(id, _)| id).collect();
        baseline_normal.sort_unstable();
        let mut l_normal_only: u32 = 1 + baseline_normal.len() as u32;
        for (obs, ev) in n.spatial_history.iter() {
            l_normal_only += Neuron::spatial_distance(obs, &baseline_normal) * ev.frames.len() as u32;
        }
        eprintln!(
            "[phase1-compress] settled L={} vs normal-only L={} ({} children; window={} frames, {} distinct configs; mints total={})",
            l_settled, l_normal_only, children.len(), n.spatial_history.total_frames(), n.spatial_history.iter().count(), mints
        );
        assert!(l_settled < l_normal_only,
            "Phase 1 must compress: settled L={} should be below normal-only L={}", l_settled, l_normal_only);
    }

    // ── Phase 1 configuration-loop unit tests ──────────────────────────────────
    //
    // Each drives a single Neuron through process_spatial_frame and, when the add pass returns a
    // request, installs the child with context = O (add_spatial_pattern) exactly as the thalamus does.

    /// A configuration as the sorted, deduplicated id list the spatial helpers now speak.
    fn set(ids: &[NeuronId]) -> Vec<NeuronId> {
        let mut v = ids.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// A spatial neighborhood context over the given neighbor ids, each at strength 1.
    fn spatial_ctx(ids: &[NeuronId]) -> crate::context::SpatialContext {
        let mut c = crate::context::SpatialContext::new();
        for &id in ids { c.add_neuron(id, 1.0); }
        c
    }

    /// A level-0 neuron whose horizon is `round(1 / forget_rate)` — 0.1 → 10, 0.05 → 20. The horizon is
    /// now its own parameter, so these Phase-1 tests keep the old rate-derived value to preserve intent.
    fn phase1_neuron(forget_rate: f64) -> Neuron {
        let horizon = (1.0 / forget_rate).round().max(1.0) as u32;
        Neuron::new(1, forget_rate, 0.9, GroupMode::Static, FxHashMap::default(), 10, true, horizon)
    }

    /// Drive one learning frame through the compute→elect→commit path, standing in for the thalamus at
    /// n=1: a request wins the election iff its cover has ≥2 members. A winning new-child request creates
    /// and commits the child; a winning recognition commits; a losing request commits nothing; a pure
    /// normal-serve is already finalized inline by the frame pass. Returns the minted child id (if any)
    /// and the ids retired this frame.
    fn step(n: &mut Neuron, ids: &[NeuronId], frame: FrameNumber, next_pid: &mut NeuronId) -> (Option<NeuronId>, Vec<NeuronId>) {
        let ctx = spatial_ctx(ids);
        let empty: FxHashSet<NeuronId> = FxHashSet::default();
        let res = n.process_spatial_frame(Some(&ctx), &empty, &[], frame);
        let Some(bid) = res.bid else {
            // Pure normal-serve: already committed inline; any delete was applied inline too.
            return (None, res.delete_candidate.into_iter().collect());
        };
        if bid.covered.len() < 2 {
            return (None, Vec::new()); // loses the election (covers <2): nothing commits
        }
        let deleted: Vec<NeuronId> = res.delete_candidate.into_iter().collect();
        if bid.pattern_id == NEW_CHILD_BID {
            let pid = *next_pid;
            *next_pid += 1;
            n.add_spatial_pattern(pid, &res.observed, frame);
            n.commit_spatial_frame(&res.observed, SpatialServer::Child(pid), 0, res.delete_candidate, frame);
            (Some(pid), deleted)
        } else {
            let pid = bid.pattern_id;
            n.commit_spatial_frame(&res.observed, SpatialServer::Child(pid), res.served_distance, res.delete_candidate, frame);
            (None, deleted)
        }
    }

    /// The normal's current stored configuration (sorted id list).
    fn normal_cfg(n: &Neuron) -> Vec<NeuronId> {
        n.spatial_entries().into_iter().find_map(|(s, c)| matches!(s, SpatialServer::Normal).then_some(c)).unwrap()
    }

    /// Every child's current stored configuration (each a sorted id list).
    fn child_cfgs(n: &Neuron) -> Vec<Vec<NeuronId>> {
        n.spatial_entries().into_iter().filter_map(|(s, c)| matches!(s, SpatialServer::Child(_)).then_some(c)).collect()
    }

    /// Distance is Hamming (the symmetric-difference count), symmetric in its two arguments.
    #[test]
    fn test_spatial_distance_is_hamming() {
        assert_eq!(Neuron::spatial_distance(&set(&[1, 2, 3]), &set(&[1, 2, 3])), 0);
        assert_eq!(Neuron::spatial_distance(&set(&[1, 2, 3]), &set(&[4, 5, 6, 7])), 7); // disjoint: 3 + 4
        assert_eq!(Neuron::spatial_distance(&set(&[1, 2, 3]), &set(&[1, 2])), 1);       // one missing
        assert_eq!(Neuron::spatial_distance(&set(&[1, 2, 3]), &set(&[1, 2, 4])), 2);    // one swapped
        assert_eq!(Neuron::spatial_distance(&set(&[]), &set(&[1, 2])), 2);
        assert_eq!(
            Neuron::spatial_distance(&set(&[1, 2]), &set(&[1, 2, 3, 4])),
            Neuron::spatial_distance(&set(&[1, 2, 3, 4]), &set(&[1, 2])),
        );
    }

    /// Routing takes the closest entry; on a tie the normal outranks any child, and among children
    /// the smaller id outranks — so a recurring input always yields the same server.
    #[test]
    fn test_spatial_route_tiebreak_normal_then_smallest_id() {
        assert!(Neuron::spatial_outranks(SpatialServer::Normal, SpatialServer::Child(5)));
        assert!(!Neuron::spatial_outranks(SpatialServer::Child(5), SpatialServer::Normal));
        assert!(Neuron::spatial_outranks(SpatialServer::Child(3), SpatialServer::Child(9)));
        assert!(!Neuron::spatial_outranks(SpatialServer::Child(9), SpatialServer::Child(3)));

        // Normal and a child equidistant (both at 2) → the normal serves.
        let entries = vec![(SpatialServer::Normal, set(&[1, 2])), (SpatialServer::Child(5), set(&[1, 3]))];
        assert_eq!(Neuron::spatial_route(&set(&[1, 4]), &entries), (SpatialServer::Normal, 2));

        // Two children equidistant (both at 0) and beating the normal → the smaller id serves.
        let entries = vec![
            (SpatialServer::Normal, set(&[7, 8])),
            (SpatialServer::Child(9), set(&[1, 2])),
            (SpatialServer::Child(3), set(&[1, 2])),
        ];
        assert_eq!(Neuron::spatial_route(&set(&[1, 2]), &entries), (SpatialServer::Child(3), 0));
    }

    /// Cold start is silence: a neuron with no neighbors this frame routes, serves, and decides
    /// nothing, and writes no history.
    #[test]
    fn test_cold_start_is_silence() {
        let mut n = phase1_neuron(0.1);
        let empty: FxHashSet<NeuronId> = FxHashSet::default();
        let r = n.process_spatial_frame(None, &empty, &[], 0);
        assert!(r.bid.is_none() && r.delete_candidate.is_none());
        let r = n.process_spatial_frame(Some(&spatial_ctx(&[])), &empty, &[], 1);
        assert!(r.bid.is_none());
        assert!(n.spatial_history.is_empty(), "silence writes no history");
    }

    /// A stable world builds nothing: a single recurring configuration is served by the normal
    /// forever and never justifies a child (docs/algorithm.md, "The normal is always in the comparison").
    #[test]
    fn test_stable_world_creates_no_child() {
        let mut n = phase1_neuron(0.1);
        let mut pid = 1000;
        let c = [1, 2, 3];
        for f in 0..40 {
            let (minted, deleted) = step(&mut n, &c, f, &mut pid);
            assert!(minted.is_none(), "a stable configuration must never mint a child (frame {f})");
            assert!(deleted.is_empty());
        }
        assert!(child_cfgs(&n).is_empty());
        assert_eq!(normal_cfg(&n), set(&c), "the normal settles to the stable configuration");
    }

    /// A minted child's context is exactly the observation O that justified it (docs/algorithm.md,
    /// "Creating a child": its configuration starts as the observation that justified it).
    #[test]
    fn test_mint_context_is_the_observation() {
        let mut n = phase1_neuron(0.05);
        let (a, b) = ([10, 11, 12], [20, 21, 22, 23]);
        let empty: FxHashSet<NeuronId> = FxHashSet::default();
        let mut minted_cfg = None;
        for f in 0..40 {
            let obs = if f % 5 < 3 { &a[..] } else { &b[..] };
            let res = n.process_spatial_frame(Some(&spatial_ctx(obs)), &empty, &[], f);
            if let Some(bid) = &res.bid {
                if bid.pattern_id == NEW_CHILD_BID {
                    minted_cfg = Some(set(&res.observed));
                    n.add_spatial_pattern(1000, &res.observed, f);
                    break;
                }
            }
        }
        assert_eq!(minted_cfg, Some(set(&b)), "the first child minted takes the deviation B as its context");
    }

    /// Entries are frozen at their seed/mint configuration: the normal is seeded to A on the first frame
    /// and never moves, so minting B never disturbs it — the normal stays A throughout.
    #[test]
    fn test_mint_frame_does_not_pollute_the_normal() {
        let mut n = phase1_neuron(0.05);
        let mut pid = 1000;
        let (a, b) = ([10, 11, 12], [20, 21, 22, 23]);
        for f in 0..40 {
            let obs = if f % 5 < 3 { &a[..] } else { &b[..] };
            let (minted, _) = step(&mut n, obs, f, &mut pid);
            if minted.is_some() {
                assert_eq!(normal_cfg(&n), set(&a), "minting must not disturb the frozen normal (frame {f})");
            }
        }
        assert_eq!(child_cfgs(&n), vec![set(&b)], "settles to one child = B");
        assert_eq!(normal_cfg(&n), set(&a), "normal stays A");
    }

    /// A child is retired once its demand ages out of the window, and the normal — which owns no cost
    /// line — is never deleted (docs/algorithm.md, "The normal is always in the comparison").
    #[test]
    fn test_child_deleted_when_demand_disappears() {
        let mut n = phase1_neuron(0.1);
        let mut pid = 1000;
        let (a, b) = ([10, 11, 12], [20, 21, 22, 23]);
        for f in 0..40 {
            let obs = if f % 5 < 3 { &a[..] } else { &b[..] };
            step(&mut n, obs, f, &mut pid);
        }
        assert_eq!(child_cfgs(&n).len(), 1, "one child established for the recurring B");

        let mut deleted_any = false;
        for f in 40..70 {
            let (_m, del) = step(&mut n, &a, f, &mut pid);
            if !del.is_empty() { deleted_any = true; }
        }
        assert!(deleted_any, "the child is retired once its demand ages out of the window");
        assert!(child_cfgs(&n).is_empty(), "no child survives after its demand disappears");
        assert!(n.get_spatial_routing_table().contains_key(&SpatialServer::Normal), "the normal is never deleted");
        assert_eq!(normal_cfg(&n), set(&a));
    }

    /// A perfectly-served frame (distance 0) skips the add pass entirely — a settled neuron does no
    /// expensive work (docs/algorithm.md, "Risks": run the add pass only when the frame had error).
    #[test]
    fn test_no_add_pass_on_a_perfectly_served_frame() {
        let mut n = phase1_neuron(0.1);
        let empty: FxHashSet<NeuronId> = FxHashSet::default();
        let c = [1, 2, 3];
        n.process_spatial_frame(Some(&spatial_ctx(&c)), &empty, &[], 0); // frame 0 seeds the normal onto C
        for f in 1..10 {
            let r = n.process_spatial_frame(Some(&spatial_ctx(&c)), &empty, &[], f);
            assert!(r.bid.is_none(), "a distance-0 normal-serve must make no request (frame {f})");
        }
    }
}
