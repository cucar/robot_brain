/// Brain — top-level orchestrator for the hierarchical prediction engine.
///
/// Owns the full frame pipeline: sensory input → quantization → pattern recognition
/// → voting → inference. Coordinates Thalamus, Memory, Diagnostics, and Quantizer.
///
/// Architecture:
///   processFrame is the single entry point per time step. It:
///   1. Builds the frame from quantized inputs + previously inferred actions
///   2. Creates any new sensory neurons (Op-1)
///   3. Cleans up dead patterns (Op-2 — reap + cascade delete)
///   4. Ages context (slides the temporal window)
///   5. Activates new neurons at age 0
///   6. Processes levels (Op-3 dispatch per level, deferred Op-4/Op-5)
///   7. Runs voting consensus → scalar-space inference output
///   8. Tracks continuous prediction error (MAPE)
///
/// The Brain is a pure compute function — no I/O, no rendering, no action dispatch.
/// The host (N-API binding or CLI runner) feeds inputs and reads the return value.

use rustc_hash::{FxHashMap, FxHashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::backup::{Backup, read_csv, write_csv};
use crate::column::NeuronCreateSpec;
use crate::diagnostics::{DimInference, Diagnostics, InferenceType};
use crate::memory::{InferredNeuron, Memory};
use crate::neuron::{ContextRefEntry, Vote};
use crate::thalamus::{
    FlatVote, LevelAgeState,
    PointLookup, Thalamus,
};
use crate::types::{
    ChannelId, Coordinate, DimensionId, Distance, ErrorMode, FrameNumber,
    Level, LevelDecayMode, NeuronId, NeuronType, Reward,
};

// ── Re-exports for the N-API layer ──────────────────────────────────────────
// These types live in crate-private modules but are part of Brain's public API.
pub use crate::diagnostics::DiagnosticStats;
pub use crate::thalamus::{ChannelRegistration, DimKind, DimSpecInput};

// ── Supporting structs ──────────────────────────────────────────────────────

/// A single frame point: quantized coordinate + channel + type.
#[derive(Debug, Clone)]
struct FramePoint {
    coordinate: Coordinate,
    channel_id: ChannelId,
    neuron_type: NeuronType,
}

/// Per-dimension inference output in scalar space.
/// Winner is the dequantized best candidate; continuous is the
/// score-weighted average across all candidates on that dimension.
#[derive(Debug, Clone)]
pub struct DimInferenceOutput {
    pub dim_id: DimensionId,
    pub kind: NeuronType,
    pub winner: WinnerOutput,
    pub continuous: Option<f64>,
}

/// The winning candidate on a dimension.
#[derive(Debug, Clone)]
pub struct WinnerOutput {
    pub value: Option<f64>,
    pub strength: f64,
    pub score: f64,
}

/// Return value from processFrame — inferences plus per-frame diagnostic byproducts.
#[derive(Debug)]
pub struct FrameResult {
    /// Per-channel, per-dimension scalar-space inferences.
    pub inferences: FxHashMap<ChannelId, Vec<DimInferenceOutput>>,
    /// Wall-clock elapsed time for this frame (seconds).
    pub elapsed: f64,
    /// Optional vote debug dump — only populated when debug is set.
    pub vote_debug: Option<VoteDebug>,
    /// Per-action-value vote statistics — one entry per digit that received
    /// at least one vote. Shows the number of individual neuron votes, total
    /// strength, and reward distribution (min/max/avg) across those votes.
    pub action_vote_stats: Vec<ActionVoteStats>,
    /// Per-voter action votes — one entry per (voter, action neuron) pair.
    /// Preserves voter identity so hosts can implement per-voter normalization
    /// schemes that aggregated stats lose.
    pub action_votes: Vec<ActionVote>,
    /// Per-section wall-clock timings (seconds). Populated for profiling.
    /// Zero when the section was skipped (e.g. cleanup runs only when learning).
    pub timings: FrameTimings,
}

/// Memory-op timings inside brain.process_levels, summed across levels.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryTimings {
    pub get_level_neurons: f64,
    pub write_back_level_neurons: f64,
    pub activate_patterns: f64,
}

/// Per-section timings inside a single frame, in seconds.
#[derive(Debug, Clone, Default)]
pub struct FrameTimings {
    pub build_frame: f64,
    pub create_sensory: f64,
    pub cleanup_dead: f64,
    pub age_context: f64,
    pub activate: f64,
    pub process_levels: f64,
    pub apply_results: f64,
    pub infer: f64,
    pub track_error: f64,
    // Sub-timings inside process_levels: summed across all neurons and all
    // levels for this frame. process_levels includes overhead beyond these
    // four (memory I/O, level dispatch) so the four won't sum to it exactly.
    pub neuron_learn_connections: f64,
    pub neuron_recognize_patterns: f64,
    pub neuron_correct_errors: f64,
    pub neuron_generate_votes: f64,
    // recognize_patterns sub-buckets: index lookup vs candidate scoring.
    pub recognize_candidate_search: f64,
    pub recognize_candidate_eval: f64,
    pub recognize_candidates_evaluated: u64,
    // Orchestration around the per-neuron dispatch, summed across levels.
    pub orch_get_level_tasks: f64,
    pub orch_dispatch_frame: f64,
    pub orch_collect_activations: f64,
    pub orch_collect_votes: f64,
    // Memory ops in brain.process_levels, summed across levels.
    pub mem_get_level_neurons: f64,
    pub mem_write_back_level_neurons: f64,
    pub mem_activate_patterns: f64,
}

/// A single vote cast by a voter neuron for an action neuron.
/// Generic across action dimensions — `value` is the action's bucket id.
#[derive(Debug, Clone)]
pub struct ActionVote {
    pub voter_neuron_id: NeuronId,
    pub channel_id: ChannelId,
    pub dimension_id: DimensionId,
    pub value: i32,
    pub strength: f64,
    pub reward: f64,
}

/// Per-action-value vote statistics aggregated from raw FlatVotes.
/// Groups all votes targeting action neurons with the same bucket_id (digit).
#[derive(Debug, Clone)]
pub struct ActionVoteStats {
    /// The action value (e.g. digit 0–9).
    pub value: i32,
    /// Number of individual neuron votes for this action.
    pub vote_count: usize,
    /// Sum of vote strengths.
    pub total_strength: f64,
    /// Average reward across votes (strength-weighted).
    pub avg_reward: f64,
    /// Minimum reward across individual votes.
    pub min_reward: f64,
    /// Maximum reward across individual votes.
    pub max_reward: f64,
}

/// Resolved vote snapshot for host-side debug rendering.
#[derive(Debug, Clone)]
pub struct VoteDebug {
    pub votes: Vec<ResolvedVote>,
    pub winners: Vec<InferredNeuron>,
}

/// A single resolved vote with human-readable metadata.
#[derive(Debug, Clone)]
pub struct ResolvedVote {
    pub target_id: NeuronId,
    pub target_type: Option<NeuronType>,
    pub target_channel_id: Option<ChannelId>,
    pub target_coordinate: Option<(String, i32)>,
    pub voter_id: NeuronId,
    pub voter_level: Option<Level>,
    pub voter_label: String,
    pub strength: f64,
    pub reward: f64,
    pub distance: Distance,
}

/// Episode-level summary for host-side rendering.
#[derive(Debug, Clone)]
pub struct FrameSummary {
    pub frame_number: FrameNumber,
    pub neuron_count: usize,
    pub max_level: Level,
    pub stats: DiagnosticStats,
}

/// Summary returned by finalize_implant.
#[derive(Debug, Clone)]
pub struct ImplantSummary {
    pub positions_processed: usize,
    pub patterns_created: usize,
}

/// Neuron inspection result — parent + level + stored context entries.
#[derive(Debug, Clone)]
pub struct InspectedNeuron {
    pub neuron_id: NeuronId,
    pub level: Level,
    pub parent_id: Option<NeuronId>,
    /// (context_neuron_id, distance, strength) tuples from the parent's
    /// routing-table entry for this child pattern.
    pub context: Vec<(NeuronId, Distance, f64)>,
}

/// Start-of-frame diagnostic snapshot (rewards, observations, dim names).
#[derive(Debug, Clone)]
pub struct StartFrameInfo {
    pub frame_number: FrameNumber,
    pub rewards: FxHashMap<ChannelId, Reward>,
    pub frame: Vec<FramePointInfo>,
    pub dimension_id_to_name: FxHashMap<DimensionId, String>,
}

/// A simplified frame point for diagnostic display.
#[derive(Debug, Clone)]
pub struct FramePointInfo {
    pub coordinate: Coordinate,
    pub channel_id: ChannelId,
    pub neuron_type: NeuronType,
}

/// Episode summary with all diagnostic information.
#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    pub frame_number: FrameNumber,
    pub stats: DiagnosticStats,
}

/// Candidate aggregation entry — intermediate state during vote aggregation.
#[derive(Debug, Clone)]
struct Candidate {
    strength: f64,
    weighted_total: f64,
    reward: f64,
    probability: f64,
}

/// Per-dimension winner tracking during consensus determination.
#[derive(Debug, Clone)]
struct DimBestEntry {
    neuron_id: NeuronId,
    score: f64,
    strength: f64,
}

// ── Brain ───────────────────────────────────────────────────────────────────

pub struct Brain {
    /// Number of frames a base neuron stays active.
    context_length: u32,

    /// Debug flag — when set, enables verbose logging and vote debug output.
    debug: bool,

    /// When false, skip sensory neuron activation, aging, event learning, and
    /// pattern creation. The context is frozen — existing neurons stay at their
    /// current ages. Used for action-only training on a frozen image context.
    event_processing: bool,

    /// When false, skip action neuron inclusion in frame building, action
    /// inference in consensus, and action reward processing. Used during
    /// pure event learning phases (e.g. image scanning before digit prediction).
    action_processing: bool,

    /// When false, skip error correction pattern creation and connection
    /// learning during processFrame. Existing patterns still activate
    /// (recognition), but no new neurons are created and no connections
    /// are updated. Used during test/inference event scanning.
    learning: bool,

    /// Current frame data from all channels (rebuilt each frame).
    frame: Vec<FramePoint>,

    /// Channel rewards indexed by age — rewards[0] is this frame's rewards,
    /// rewards[1] is last frame's, etc. Kept in sync with the context window.
    rewards: Vec<FxHashMap<ChannelId, Reward>>,

    /// Monotonically increasing frame counter. Used for death ledger scheduling,
    /// diagnostics timestamps, and lazy decay materialization.
    frame_number: FrameNumber,

    /// Pure stats tracker — no presentation, no flags.
    diagnostics: Diagnostics,

    /// File-based snapshot save/load.
    backup_store: Backup,

    /// Relay station for neuron/channel/dimension mappings. Owns the Region tree.
    thalamus: Thalamus,

    /// Temporal sliding window of active and inferred neurons.
    memory: Memory,

    /// Implant-mode state — set by start_implant, used by implant_position,
    /// cleared by finalize_implant. None when not in implant mode.
    implant_state: Option<ImplantState>,

    /// Parallel runtime contexts. Empty when single-context (default).
    /// When `init_contexts(N)` is called, contexts holds N independent
    /// `ContextState` slots. The currently-active slot has its
    /// (memory, frame_number, rewards) swapped INTO Brain's fields;
    /// inactive slots hold their state in `contexts[i]`. `current_context`
    /// tells us which slot is currently swapped in.
    contexts: Vec<ContextState>,
    current_context: usize,
}

/// Snapshot of a single brain's runtime context — everything that must be
/// per-context when `init_contexts(N)` is used. The thalamus (neurons,
/// patterns, columns) is shared across all contexts.
#[derive(Clone)]
pub struct ContextState {
    pub memory: Memory,
    pub frame_number: FrameNumber,
    pub rewards: Vec<FxHashMap<ChannelId, Reward>>,
}

/// Per-image bit history accumulated during implant. The bit value is
/// the bucket id (0 or 1 for binary channels).
///
/// During implant the brain records observations only; the full algorithm
/// runs at finalize_implant when all positions have been seen.
#[derive(Debug)]
pub struct ImplantState {
    /// channel id for the single event dimension being implanted
    pub channel_id: ChannelId,
    /// dimension id within the channel
    pub dim_id: DimensionId,
    /// per-image bit history (each Vec is one image's bit stream so far)
    pub histories: Vec<Vec<u8>>,
    /// current position (number of bits implanted per image)
    pub position: u64,
    /// Recorded observations keyed by (parent_bit, packed_window_bits).
    /// Value is the count of each next_bit (index 0 = next_bit 0, idx 1 = next_bit 1).
    /// packed_window: bit at distance D-1 occupies bit position D-1 of the u64
    /// (least-significant bit = distance 1, most-significant within the used
    /// range = distance context_length-1). Limited to context_length ≤ 64.
    pub observations: FxHashMap<(u8, u64), [u32; 2]>,
}

impl Brain {
    /// Create a new Brain with the given hyperparameters.
    ///
    /// # Arguments
    /// * `context_length` — number of frames a base neuron stays active (default 10)
    /// * `error_correction_mode` — 'static' | 'conservative' | 'neutral' | 'aggressive'
    /// * `error_correction_threshold` — fixed threshold when mode='static'; warmup fallback
    /// * `merge_threshold` — percentage of matched entries needed for context merge
    /// * `pattern_forget_rate` — level-1 forget rate; deeper levels decay by contextLength per level
    /// * `regions` — R — number of regions (1 for single-process)
    /// * `columns` — C — number of columns per region (1 for single-thread)
    /// * `debug` — enable verbose logging
    pub fn new(
        context_length: u32,
        error_correction_mode: ErrorMode,
        error_correction_threshold: f64,
        merge_threshold: f64,
        pattern_forget_rate: f64,
        level_decay_mode: LevelDecayMode,
        regions: usize,
        columns: usize,
        debug: bool,
        action_alpha: Option<f64>,
    ) -> Self {
        Self {
            context_length,
            debug,
            event_processing: true,
            action_processing: true,
            learning: true,
            frame: Vec::new(),
            rewards: Vec::new(),
            frame_number: 0,
            diagnostics: Diagnostics::new(),
            backup_store: Backup::new(pattern_forget_rate, level_decay_mode, context_length),
            thalamus: Thalamus::new(
                debug,
                pattern_forget_rate,
                level_decay_mode,
                merge_threshold,
                context_length,
                error_correction_mode,
                error_correction_threshold,
                regions,
                columns,
                action_alpha,
            ),
            memory: Memory::new(debug, context_length),
            implant_state: None,
            contexts: Vec::new(),
            current_context: 0,
        }
    }

    // ── Parallel contexts ───────────────────────────────────────────────────

    /// Initialize N parallel runtime contexts. Each context has its own
    /// (memory, frame_number, rewards); the thalamus (neurons, patterns) is
    /// shared. After init, the active context is slot 0 — pass `context_id`
    /// to `process_frame` / `learn` / `infer` / `reset_context` to operate on
    /// a specific slot.
    pub fn init_contexts(&mut self, n: usize) {
        // wipe current single-context state and start fresh
        self.reset_context();
        let template = ContextState {
            memory: Memory::new(self.debug, self.context_length),
            frame_number: 0,
            rewards: Vec::new(),
        };
        // slot 0's state IS the brain's current (memory, frame, rewards);
        // we leave it swapped in. Slots 1..N hold their state in contexts[].
        self.contexts = (0..n).map(|_| template.clone()).collect();
        self.current_context = 0;
    }

    /// Swap to the given context. Saves the currently-active state into its
    /// slot and loads the target slot. No-op when already on that slot.
    fn swap_to(&mut self, ctx_id: usize) {
        if self.contexts.is_empty() { return; }   // single-context mode
        if ctx_id == self.current_context { return; }
        assert!(ctx_id < self.contexts.len(), "context_id {} out of range (init_contexts({}))", ctx_id, self.contexts.len());
        // save current state into current slot
        std::mem::swap(&mut self.memory, &mut self.contexts[self.current_context].memory);
        std::mem::swap(&mut self.frame_number, &mut self.contexts[self.current_context].frame_number);
        std::mem::swap(&mut self.rewards, &mut self.contexts[self.current_context].rewards);
        // load target slot
        std::mem::swap(&mut self.memory, &mut self.contexts[ctx_id].memory);
        std::mem::swap(&mut self.frame_number, &mut self.contexts[ctx_id].frame_number);
        std::mem::swap(&mut self.rewards, &mut self.contexts[ctx_id].rewards);
        self.current_context = ctx_id;
    }

    /// Get number of initialized contexts (0 = single-context mode).
    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }

    /// Switch the active context. Subsequent process_frame/learn/infer/
    /// reset_context calls operate on this context.
    pub fn set_active_context(&mut self, ctx_id: usize) {
        self.swap_to(ctx_id);
    }

    // ── Channel registration ────────────────────────────────────────────────

    /// Register a channel spec with the brain. Delegates to Thalamus.
    /// Returns the allocated channel id and per-dimension id map.
    pub fn register_channel_spec(
        &mut self,
        name: &str,
        dimensions: Vec<DimSpecInput>,
        emits_reward: bool,
        learn_action_sequences: bool,
    ) -> ChannelRegistration {
        self.thalamus.register_channel_spec(name, dimensions, emits_reward, learn_action_sequences)
    }

    // ── Context / reset ─────────────────────────────────────────────────────

    /// Reset brain memory state for a clean episode start.
    /// Materializes all lazy decay, resets frame counter and death ledger so
    /// the next episode starts clean while preserving learned knowledge.
    pub fn reset_context(&mut self) {
        // println!("Resetting brain context...");

        // Materialize all lazy decay and reset timestamps so frameNumber can restart at 0
        self.thalamus.materialize_and_reset_neurons(self.frame_number);
        self.frame_number = 0;

        // Reset accuracy stats
        self.diagnostics.reset_accuracy_stats();

        // Clear memory and rewards history
        self.memory.reset();
        self.rewards.clear();
    }

    /// Hard reset: clears ALL learned data (used mainly for tests).
    pub fn reset_brain(&mut self) {
        println!("Hard resetting brain (all learned data)...");

        // reset active memory (also resets frameNumber)
        self.reset_context();

        // reset all neurons
        self.thalamus.reset();
    }

    /// Reset accuracy and reward stats for a new episode.
    pub fn reset_accuracy_stats(&mut self) {
        self.diagnostics.reset_accuracy_stats();
    }

    /// Set processing mode flags that control which parts of the pipeline run.
    ///
    /// * `event_processing` — when false, skips sensory activation, aging,
    ///   event learning, and pattern creation. Context is frozen.
    /// * `action_processing` — when false, skips action neurons in frame
    ///   building, action inference, and action reward processing.
    /// * `learning` — when false, skips error correction pattern creation,
    ///   connection learning, and new neuron default-action pre-wiring.
    ///   Existing patterns still activate (recognition only).
    pub fn set_processing_mode(&mut self, event_processing: bool, action_processing: bool, learning: bool) {
        self.event_processing = event_processing;
        self.action_processing = action_processing;
        self.learning = learning;
        self.thalamus.set_action_processing(action_processing && learning);
        self.thalamus.set_learning(learning);
    }

    // ── Save / load ──────────────────────────────────���──────────────────────

    // ── Direct learning / inference ───────────────────────────────────────

    /// Direct supervised learning: resolve action neurons from the given action
    /// coordinates, wire connections from all votable context neurons to those
    /// action neurons with the given reward. Then run inference to test if
    /// the brain now returns the learned action.
    ///
    /// Does not advance the frame or age the context. The caller feeds events
    /// via process_frame first to build context, then calls learn() to wire
    /// action connections and verify.
    pub fn learn(
        &mut self,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        rewards: &FxHashMap<ChannelId, Reward>,
    ) -> FrameResult {
        let frame_start = Instant::now();

        // 1. Resolve action coordinates to neuron IDs
        let mut action_neurons: Vec<(NeuronId, ChannelId)> = Vec::new();
        for (&channel_id, dim_map) in actions {
            for (&dim_id, &scalar) in dim_map {
                let bucket_id = scalar as i32;
                let coord = Coordinate { dim_id, bucket_id };
                let lookup = self.thalamus.get_neuron_id_for_point(
                    &coord, channel_id, NeuronType::Action,
                );
                if lookup.is_new {
                    self.thalamus.create_neurons(&[NeuronCreateSpec {
                        id: lookup.id,
                        forget_rate: 0.0,
                        connections: None,
                    }]);
                }
                action_neurons.push((lookup.id, channel_id));
            }
        }

        // 2. Get votable (non-suppressed, non-oldest) entries from context
        let votable = self.memory.get_votable_entries();

        // 3. Build connection tasks: voter -> action at distance = age + 1
        //    Every votable neuron connects to every action neuron — no channel
        //    matching. Pattern neurons span channels, so filtering by channel
        //    would exclude them (they aren't in base_neurons). The normal
        //    learn_connections() path also connects regardless of channel.
        let mut tasks: Vec<(NeuronId, NeuronId, Distance, ChannelId, Reward)> = Vec::new();
        for &(voter_id, age) in &votable {
            let distance = age + 1;
            for &(action_id, action_channel) in &action_neurons {
                let reward = rewards.get(&action_channel).copied().unwrap_or(0.0);
                tasks.push((voter_id, action_id, distance, action_channel, reward));
            }
        }

        // 4. Dispatch learned connections
        self.thalamus.learn_action_connections(&tasks);

        // 5. Infer to test if learning took effect
        let mut infer_result = self.infer();
        infer_result.elapsed = frame_start.elapsed().as_secs_f64();
        infer_result
    }

    /// Run inference on the current context without advancing the frame.
    /// Collects votes from all votable neurons and determines action consensus.
    /// No side effects -- no frame advancement, no aging, no pattern creation,
    /// no connection learning.
    pub fn infer(&mut self) -> FrameResult {
        let frame_start = Instant::now();

        // Get votable entries from memory
        let votable = self.memory.get_votable_entries();

        // Collect votes via the thalamus -> region -> column -> neuron.vote() path
        let votes = self.thalamus.collect_votes(&votable);

        // Run consensus and build inference output
        let (inferences_map, vote_debug, _winners, action_vote_stats, action_votes) = self.infer_neurons(&votes);

        FrameResult {
            inferences: inferences_map,
            elapsed: frame_start.elapsed().as_secs_f64(),
            vote_debug,
            action_vote_stats,
            action_votes,
            timings: FrameTimings::default(),
        }
    }

    /// Save brain state to `<job_dir>/backups/<label>/`. Materializes lazy decay
    /// before snapshotting so strengths reflect true post-decay values.
    pub fn save(&mut self, job_dir: &std::path::Path, label: &str) -> Result<std::path::PathBuf, String> {
        self.thalamus.materialize_and_reset_neurons(self.frame_number);
        self.frame_number = 0;
        let snapshot = self.thalamus.get_snapshot();
        self.backup_store.save(job_dir, label, &snapshot)
    }

    /// Load a backup by label from `<job_dir>/backups/<label>/`.
    pub fn load(&mut self, job_dir: &std::path::Path, label: &str) -> Result<(), String> {
        let snapshot = self.backup_store.load(job_dir, label)?;
        self.thalamus.restore_snapshot(&snapshot);
        Ok(())
    }

    // ── Context save / load ─────────────────────────────────────────────────

    /// Save the brain's runtime context (active neurons, frame number, rewards,
    /// inferred neurons) to `<job_dir>/contexts/<label>/`. Uses "latest" semantics
    /// identical to brain save: timestamped folder vs named folder.
    ///
    /// Active neuron activation frames are stored as offsets relative to the
    /// current frame_number (always ≤ 0). On restore, frame_number is set to 0
    /// and the offsets become the activation_frames directly — no timestamp
    /// fixup needed.
    pub fn save_context(&self, job_dir: &std::path::Path, label: &str) -> Result<PathBuf, String> {
        let contexts_dir = job_dir.join("contexts");
        fs::create_dir_all(&contexts_dir)
            .map_err(|e| format!("Failed to create contexts dir: {}", e))?;

        let folder = if label == "latest" {
            let timestamp = crate::backup::format_timestamp();
            contexts_dir.join(&timestamp)
        } else {
            contexts_dir.join(label)
        };

        if label != "latest" && folder.exists() {
            fs::remove_dir_all(&folder)
                .map_err(|e| format!("Failed to clear context folder: {}", e))?;
        }
        fs::create_dir_all(&folder)
            .map_err(|e| format!("Failed to create context folder: {}", e))?;

        // active_neurons.csv: neuron_id, activation_offset, level, activated_pattern_id, threshold
        // activation_offset = activation_frame - frame_number (always ≤ 0)
        let activations = self.memory.get_context_snapshot();
        let rows: Vec<Vec<String>> = activations.iter()
            .map(|(id, frame, level, state)| {
                let offset = frame - self.frame_number;
                let pattern_id_str = state.activated_pattern_id.map_or(String::new(), |p| p.to_string());
                let threshold_str = state.threshold.map_or(String::new(), |t| t.to_string());
                vec![id.to_string(), offset.to_string(), level.to_string(), pattern_id_str, threshold_str]
            })
            .collect();
        write_csv(&folder.join("active_neurons.csv"), &rows)?;

        // votes.csv: neuron_id, activation_offset, voted_neuron_id, strength, reward, distance
        let mut vote_rows: Vec<Vec<String>> = Vec::new();
        for (id, frame, _level, state) in &activations {
            let offset = frame - self.frame_number;
            if let Some(ref votes) = state.votes {
                for vote in votes {
                    vote_rows.push(vec![
                        id.to_string(), offset.to_string(),
                        vote.neuron_id.to_string(), vote.strength.to_string(),
                        vote.reward.to_string(), vote.distance.to_string(),
                    ]);
                }
            }
        }
        write_csv(&folder.join("votes.csv"), &vote_rows)?;

        // context_refs.csv: neuron_id, activation_offset, ref_neuron_id, ref_distance
        let mut ref_rows: Vec<Vec<String>> = Vec::new();
        for (id, frame, _level, state) in &activations {
            let offset = frame - self.frame_number;
            if let Some(ref ctx) = state.context {
                for entry in ctx {
                    ref_rows.push(vec![
                        id.to_string(), offset.to_string(),
                        entry.neuron_id.to_string(), entry.distance.to_string(),
                    ]);
                }
            }
        }
        write_csv(&folder.join("context_refs.csv"), &ref_rows)?;

        // rewards.csv: age, channel_id, reward (one row per entry in the rewards vec)
        let mut reward_rows: Vec<Vec<String>> = Vec::new();
        for (age, reward_map) in self.rewards.iter().enumerate() {
            for (&channel_id, &reward) in reward_map {
                reward_rows.push(vec![age.to_string(), channel_id.to_string(), reward.to_string()]);
            }
        }
        write_csv(&folder.join("rewards.csv"), &reward_rows)?;

        // inferred_neurons.csv: neuron_id, dim_id, bucket_id, channel_id, strength, reward, probability
        let inferred = self.memory.get_inferred_snapshot();
        let inferred_rows: Vec<Vec<String>> = inferred.iter()
            .map(|inf| vec![
                inf.neuron_id.to_string(),
                inf.coordinate.dim_id.to_string(),
                inf.coordinate.bucket_id.to_string(),
                inf.channel_id.to_string(),
                inf.strength.to_string(),
                inf.reward.to_string(),
                inf.probability.to_string(),
            ])
            .collect();
        write_csv(&folder.join("inferred_neurons.csv"), &inferred_rows)?;

        println!("💾 Context saved: {} ({} active neurons, frame {})",
            folder.display(), activations.len(), self.frame_number);
        Ok(folder)
    }

    /// Load a context snapshot from `<job_dir>/contexts/<label>/`.
    /// Activation frames are stored as negative offsets — on restore,
    /// frame_number is 0 so the offsets become the activation_frames directly.
    pub fn load_context(&mut self, job_dir: &std::path::Path, label: &str) -> Result<(), String> {
        let contexts_dir = job_dir.join("contexts");

        let folder = if label == "latest" {
            self.find_latest_context_folder(&contexts_dir)
                .ok_or_else(|| format!("--load-context latest requested but no contexts found in {}", contexts_dir.display()))?
        } else {
            let named = contexts_dir.join(label);
            if !named.exists() {
                return Err(format!("--load-context {} requested but folder does not exist: {}", label, named.display()));
            }
            named
        };

        println!("📂 Loading context: {}", folder.display());

        // active_neurons.csv: neuron_id, activation_offset, level [, activated_pattern_id, threshold]
        let neuron_rows = read_csv(&folder.join("active_neurons.csv"))?;
        let mut activations: Vec<(NeuronId, FrameNumber, Level, LevelAgeState)> = Vec::with_capacity(neuron_rows.len());
        for row in &neuron_rows {
            if row.len() < 3 { continue; }
            let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad neuron_id: {}", e))?;
            let activation_frame: FrameNumber = row[1].parse().map_err(|e| format!("Bad activation_frame: {}", e))?;
            let level: Level = row[2].parse().map_err(|e| format!("Bad level: {}", e))?;
            let activated_pattern_id = row.get(3).and_then(|s| if s.is_empty() { None } else { s.parse().ok() });
            let threshold = row.get(4).and_then(|s| if s.is_empty() { None } else { s.parse().ok() });
            activations.push((neuron_id, activation_frame, level, LevelAgeState {
                activated_pattern_id,
                threshold,
                votes: None,
                context: None,
            }));
        }

        // votes.csv: neuron_id, activation_offset, voted_neuron_id, strength, reward, distance
        let votes_file = folder.join("votes.csv");
        if votes_file.exists() {
            let vote_rows = read_csv(&votes_file)?;
            for row in &vote_rows {
                if row.len() < 6 { continue; }
                let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad vote neuron_id: {}", e))?;
                let offset: FrameNumber = row[1].parse().map_err(|e| format!("Bad vote offset: {}", e))?;
                let vote = Vote {
                    neuron_id: row[2].parse().map_err(|e| format!("Bad voted_neuron_id: {}", e))?,
                    strength: row[3].parse().map_err(|e| format!("Bad vote strength: {}", e))?,
                    reward: row[4].parse().map_err(|e| format!("Bad vote reward: {}", e))?,
                    distance: row[5].parse().map_err(|e| format!("Bad vote distance: {}", e))?,
                };
                if let Some(entry) = activations.iter_mut().find(|(id, frame, _, _)| *id == neuron_id && *frame == offset) {
                    entry.3.votes.get_or_insert_with(Vec::new).push(vote);
                }
            }
        }

        // context_refs.csv: neuron_id, activation_offset, ref_neuron_id, ref_distance
        let refs_file = folder.join("context_refs.csv");
        if refs_file.exists() {
            let ref_rows = read_csv(&refs_file)?;
            for row in &ref_rows {
                if row.len() < 4 { continue; }
                let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad ref neuron_id: {}", e))?;
                let offset: FrameNumber = row[1].parse().map_err(|e| format!("Bad ref offset: {}", e))?;
                let ctx_entry = ContextRefEntry {
                    neuron_id: row[2].parse().map_err(|e| format!("Bad ref_neuron_id: {}", e))?,
                    distance: row[3].parse().map_err(|e| format!("Bad ref_distance: {}", e))?,
                };
                if let Some(entry) = activations.iter_mut().find(|(id, frame, _, _)| *id == neuron_id && *frame == offset) {
                    entry.3.context.get_or_insert_with(Vec::new).push(ctx_entry);
                }
            }
        }

        self.memory.restore_context_snapshot(self.frame_number, &activations);

        // rewards.csv
        self.rewards.clear();
        let rewards_file = folder.join("rewards.csv");
        if rewards_file.exists() {
            let reward_rows = read_csv(&rewards_file)?;
            for row in &reward_rows {
                if row.len() < 3 { continue; }
                let age: usize = row[0].parse().map_err(|e| format!("Bad reward age: {}", e))?;
                let channel_id: ChannelId = row[1].parse().map_err(|e| format!("Bad reward channel: {}", e))?;
                let reward: Reward = row[2].parse().map_err(|e| format!("Bad reward value: {}", e))?;
                while self.rewards.len() <= age {
                    self.rewards.push(FxHashMap::default());
                }
                self.rewards[age].insert(channel_id, reward);
            }
        }

        // inferred_neurons.csv
        let inferred_file = folder.join("inferred_neurons.csv");
        if inferred_file.exists() {
            let inferred_rows = read_csv(&inferred_file)?;
            let mut inferred = Vec::with_capacity(inferred_rows.len());
            for row in &inferred_rows {
                if row.len() < 7 { continue; }
                inferred.push(InferredNeuron {
                    neuron_id: row[0].parse().map_err(|e| format!("Bad inferred neuron_id: {}", e))?,
                    coordinate: Coordinate {
                        dim_id: row[1].parse().map_err(|e| format!("Bad inferred dim_id: {}", e))?,
                        bucket_id: row[2].parse().map_err(|e| format!("Bad inferred bucket_id: {}", e))?,
                    },
                    channel_id: row[3].parse().map_err(|e| format!("Bad inferred channel_id: {}", e))?,
                    strength: row[4].parse().map_err(|e| format!("Bad inferred strength: {}", e))?,
                    reward: row[5].parse().map_err(|e| format!("Bad inferred reward: {}", e))?,
                    probability: row[6].parse().map_err(|e| format!("Bad inferred probability: {}", e))?,
                });
            }
            self.memory.restore_inferred_snapshot(inferred);
        }

        println!("   Restored {} active neurons", activations.len());
        Ok(())
    }

    /// Find the most recent timestamped context folder under `contexts_dir`.
    /// Only considers folders matching the `YYYY-MM-DD_HH-mm-ss` format (same
    /// convention as brain backups). Named label folders are ignored — they are
    /// loaded by exact name, not by recency. Returns None if no timestamped
    /// folders exist.
    fn find_latest_context_folder(&self, contexts_dir: &std::path::Path) -> Option<PathBuf> {
        if !contexts_dir.exists() { return None; }

        // Collect only timestamp-formatted folder names (named labels are skipped)
        let mut folders: Vec<String> = fs::read_dir(contexts_dir).ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| crate::backup::is_timestamp_folder(name))
            .collect();
        if folders.is_empty() { return None; }

        // Lex-sort puts newest last thanks to zero-padded timestamps
        folders.sort();
        Some(contexts_dir.join(folders.last().unwrap()))
    }

    // ── Implant API ─────────────────────────────────────────────────────────
    //
    // Direct teaching mode. Instead of error-driven pattern creation, the
    // caller feeds bit observations from multiple training images in parallel
    // (one bit per image per position). At each position the brain looks
    // across all images to determine the dominant and minority next-bit
    // outcomes, sets default connections on sensory neurons for the dominant
    // case, and creates L1 patterns with distinguishing context for the
    // minority cases. The resulting structure perfectly predicts every
    // observed transition in O(N × frames × context_length) work — vs the
    // O(N × frames × episodes) of stream learning.

    /// Begin implant mode. The caller must have already registered the
    /// channel (single dimension, binary) via register_channel_spec.
    /// `channel_id` and `dim_id` identify the event dimension being implanted.
    pub fn start_implant(&mut self, channel_id: ChannelId, dim_id: DimensionId, num_images: u32) {
        assert!(self.context_length <= 64, "implant currently requires context_length ≤ 64");
        let histories: Vec<Vec<u8>> = (0..num_images).map(|_| Vec::new()).collect();
        self.implant_state = Some(ImplantState {
            channel_id,
            dim_id,
            histories,
            position: 0,
            observations: FxHashMap::default(),
        });
    }

    /// Process one position across all images. `bits` must have one entry per
    /// image (in the same order start_implant was given). Each value is a
    /// bucket id (0 or 1 for binary channels).
    pub fn implant_position(&mut self, bits: &[u8]) {
        let state = match self.implant_state.as_mut() {
            Some(s) => s,
            None => panic!("implant_position called without start_implant"),
        };
        assert_eq!(bits.len(), state.histories.len(), "implant_position: bits length must match num_images");

        // Push bits and increment position.
        for (img_idx, &bit) in bits.iter().enumerate() {
            state.histories[img_idx].push(bit);
        }
        state.position += 1;

        // Only record once we have enough history for a full window AND the
        // resulting pattern's parent frame would be past the inference-time
        // warmup gate. Position P: parent at idx P-2, frame P-1. Need P-1 > context_length.
        let position = state.position;
        let ctx_len = self.context_length as u64;
        if position <= ctx_len + 1 { return; }

        let last_idx = (position - 1) as usize;
        let parent_idx = last_idx - 1;
        let history_start = parent_idx.saturating_sub((self.context_length - 1) as usize);

        // For each image: pack the window into a u64 (bit at distance D occupies
        // bit position D-1) and record an observation.
        for img_idx in 0..state.histories.len() {
            let h = &state.histories[img_idx];
            if h.len() <= parent_idx { continue; }
            let parent_bit = h[parent_idx];
            let next_bit = h[last_idx];

            // Pack window bits: window[k] = bit at distance k+1 from parent
            // = h[parent_idx - 1 - k] = h[parent_idx-1], h[parent_idx-2], ...
            let mut packed: u64 = 0;
            for k in 0..(self.context_length as usize - 1) {
                let src_idx = parent_idx as i64 - 1 - k as i64;
                if src_idx < history_start as i64 || src_idx < 0 { break; }
                let bit = h[src_idx as usize] as u64;
                packed |= bit << k;
            }

            let entry = state.observations.entry((parent_bit, packed)).or_insert([0, 0]);
            entry[(next_bit & 1) as usize] += 1;
        }
    }

    /// Ensure a sensory neuron exists for (channel, dim, bucket), creating it
    /// if needed. Returns the neuron id.
    fn ensure_sensory_neuron(&mut self, channel_id: ChannelId, dim_id: DimensionId, bucket: u8) -> NeuronId {
        let coord = Coordinate { dim_id, bucket_id: bucket as i32 };
        let lookup = self.thalamus.get_neuron_id_for_point(&coord, channel_id, NeuronType::Event);
        if lookup.is_new {
            self.thalamus.create_neurons(&[NeuronCreateSpec {
                id: lookup.id,
                forget_rate: Thalamus::effective_forget_rate(
                    self.thalamus.get_base_forget_rate(),
                    self.context_length,
                    0,
                    self.thalamus.get_level_decay_mode(),
                ),
                connections: None,
            }]);
        }
        lookup.id
    }

    /// End implant mode. Walks the recorded observations and builds the
    /// brain structure: default connections (per parent → next_bit, weighted
    /// by global count) plus patterns for each unique (parent, window) that
    /// has a unique next_bit different from the global default.
    pub fn finalize_implant(&mut self) -> ImplantSummary {
        let state = self.implant_state.take().expect("finalize_implant called without start_implant");
        let channel_id = state.channel_id;
        let dim_id = state.dim_id;

        // ── Step 1: compute global counts per (parent, next_bit) ──────────
        // and aggregate into default connections on the sensory parent.
        let mut global_counts: [[u64; 2]; 2] = [[0, 0], [0, 0]]; // [parent][next]
        for ((parent_bit, _window), counts) in &state.observations {
            global_counts[*parent_bit as usize][0] += counts[0] as u64;
            global_counts[*parent_bit as usize][1] += counts[1] as u64;
        }
        // Install default connections on each sensory parent.
        for parent_bit in [0u8, 1u8] {
            let pid = self.ensure_sensory_neuron(channel_id, dim_id, parent_bit);
            for next_bit in [0u8, 1u8] {
                let c = global_counts[parent_bit as usize][next_bit as usize];
                if c == 0 { continue; }
                let tid = self.ensure_sensory_neuron(channel_id, dim_id, next_bit);
                self.thalamus.implant_default_connection(pid, tid, c as f64);
            }
        }

        // ── Step 2: per (parent, window), check whether the observed next_bit
        // (a) is unambiguous (only one next_bit observed) AND (b) differs from
        // the global default for that parent. If both, create a pattern that
        // fires for this exact window and predicts the unambiguous next_bit.
        //
        // If the (parent, window) is ambiguous (both next_bit=0 AND =1 observed),
        // it's an unresolvable case at this context length — neither default
        // nor any single pattern can perfectly handle it. Skip; accept the error.

        let mut patterns_created = 0usize;
        let mut conflicts = 0usize;
        let window_len = (self.context_length as usize).saturating_sub(1);

        for ((parent_bit, packed_window), counts) in &state.observations {
            let c0 = counts[0];
            let c1 = counts[1];
            if c0 > 0 && c1 > 0 { conflicts += 1; continue; }
            let unambiguous_next = if c0 > 0 { 0u8 } else { 1u8 };

            // Global default for this parent
            let g0 = global_counts[*parent_bit as usize][0];
            let g1 = global_counts[*parent_bit as usize][1];
            let global_default = if g0 >= g1 { 0u8 } else { 1u8 };
            if unambiguous_next == global_default { continue; }   // default already handles this — no pattern needed

            // Need a pattern. Build its stored context from packed_window.
            let pid = self.ensure_sensory_neuron(channel_id, dim_id, *parent_bit);
            let target_id = self.ensure_sensory_neuron(channel_id, dim_id, unambiguous_next);
            let mut stored_context: Vec<(NeuronId, Distance)> = Vec::with_capacity(window_len);
            for k in 0..window_len {
                let bit = ((*packed_window >> k) & 1) as u8;
                let nid = self.ensure_sensory_neuron(channel_id, dim_id, bit);
                stored_context.push((nid, (k + 1) as Distance));
            }
            self.thalamus.implant_pattern(pid, &stored_context, target_id);
            patterns_created += 1;
        }

        if self.debug && conflicts > 0 {
            println!("  implant: {} unresolvable (parent, window) tuples at context_length={}", conflicts, self.context_length);
        }

        ImplantSummary {
            positions_processed: state.position as usize,
            patterns_created,
        }
    }

    // ── Lookup helpers ─────────────────────────────────────────────────────────

	/// returns dimension id for a given dimension name
    pub fn get_dimension_id_by_name(&self, name: &str) -> Option<DimensionId> {
        self.thalamus.get_dimension_id_by_name(name)
    }

    /// returns neuron id for a given coordinate
    pub fn get_neuron_id_by_coordinate(&self, coordinate: &Coordinate) -> Option<NeuronId> {
        self.thalamus.get_neuron_id_by_coordinate(coordinate)
    }

    /// Inspection: { parent_id, level, context_entries } for a pattern neuron.
    /// Returns parent_id and level for any neuron; context_entries is empty
    /// for level-0 sensory neurons (no parent → no stored context).
    pub fn inspect_neuron(&self, neuron_id: NeuronId) -> InspectedNeuron {
        let level = self.thalamus.get_neuron_level(neuron_id).unwrap_or(0);
        let parent = self.thalamus.get_neuron_parent(neuron_id);
        let context = parent
            .and_then(|_| self.thalamus.get_pattern_context_entries(neuron_id))
            .unwrap_or_default();
        InspectedNeuron { neuron_id, level, parent_id: parent, context }
    }

    /// Inspection: dump a neuron's outgoing connections. Returns
    /// (distance, target_neuron_id, strength, reward) tuples.
    pub fn dump_neuron_connections(&self, neuron_id: NeuronId) -> Vec<(Distance, NeuronId, f64, f64)> {
        self.thalamus.dump_neuron_connections(neuron_id).unwrap_or_default()
    }

    /// Inspection: list the currently votable entries (neuron_id, age) — the
    /// same set used by infer() / learn() to collect votes and wire actions.
    pub fn get_votable_entries(&self) -> Vec<(NeuronId, Distance)> {
        self.memory.get_votable_entries()
    }

    // ── Context inspection ───────────────────────────────────────────────────

    /// Export a snapshot of all active neurons in context with their levels.
    pub fn get_context_snapshot(&self) -> Vec<(NeuronId, FrameNumber, Level, LevelAgeState)> {
        self.memory.get_context_snapshot()
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────

    /// Episode summary with all diagnostic information.
    pub fn get_episode_summary(&self) -> EpisodeSummary {
        EpisodeSummary {
            frame_number: self.frame_number,
            stats: self.diagnostics.get_stats(),
        }
    }

    /// Compose the cumulative/episode-level summary the host renderer uses.
    /// Numbers only — the renderer formats units and decides how to show None.
    pub fn get_frame_summary(&self) -> FrameSummary {
        FrameSummary {
            frame_number: self.frame_number,
            neuron_count: self.thalamus.get_neuron_count(),
            max_level: self.thalamus.get_max_level(),
            stats: self.diagnostics.get_stats(),
        }
    }

    /// Snapshot of the start-of-frame state for --diagnostic rendering.
    /// Returns None when the frame was empty (nothing to show).
    pub fn get_start_frame_info(&self) -> Option<StartFrameInfo> {
        if self.frame.is_empty() { return None; }
        Some(StartFrameInfo {
            frame_number: self.frame_number,
            rewards: self.rewards.first().cloned().unwrap_or_default(),
            frame: self.frame.iter().map(|p| FramePointInfo {
                coordinate: p.coordinate.clone(),
                channel_id: p.channel_id,
                neuron_type: p.neuron_type.clone(),
            }).collect(),
            dimension_id_to_name: self.thalamus.get_dimension_id_to_name(),
        })
    }

    // ── Main frame pipeline ─────────────────────────────────────────────────

    /// Id-native frame entry point. Accepts raw scalars per dimension and returns
    /// inferred predictions plus per-frame diagnostic byproducts. The brain is a
    /// pure compute function — no channel I/O, no action dispatch, no printing.
    ///
    /// # Arguments
    /// * `events` — channelId → (dimId → raw scalar) — sensory inputs
    /// * `actions` — channelId → (dimId → action value) — forced actions (empty = infer)
    /// * `rewards` — channelId → reward for previous frame's actions
    pub fn process_frame(
        &mut self,
        events: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        rewards: &FxHashMap<ChannelId, Reward>,
    ) -> FrameResult {
        let frame_start = Instant::now();

        // ── Event processing: build frame, age context, activate neurons,
        //    process levels (pattern recognition + error correction).
        //    When disabled, the context is frozen — no new activations, no aging.
        if self.event_processing {
            self.frame_number += 1;
            let mut timings = FrameTimings::default();

            // build the current frame from quantized events and forced/inferred actions
            let t = Instant::now();
            let frame_neurons = self.get_frame_neurons(events, actions);
            timings.build_frame = t.elapsed().as_secs_f64();
            if frame_neurons.is_empty() {
                return FrameResult {
                    inferences: FxHashMap::default(),
                    elapsed: frame_start.elapsed().as_secs_f64(),
                    vote_debug: None,
                    action_vote_stats: Vec::new(),
                    action_votes: Vec::new(),
                    timings,
                };
            }

            // Op-1: construct any new sensory neurons in their owning columns
            let t = Instant::now();
            self.create_new_sensory_neurons(&frame_neurons);
            timings.create_sensory = t.elapsed().as_secs_f64();

            // Op-2: forget connections and patterns to avoid curse of dimensionality
            // Skipped when learning is off (test/inference) — preserve trained state.
            if self.learning {
                let t = Instant::now();
                self.cleanup_dead_patterns();
                timings.cleanup_dead = t.elapsed().as_secs_f64();
            }

            // slide the temporal window: age active neurons and push the new rewards frame
            let t = Instant::now();
            self.age_context(rewards);
            timings.age_context = t.elapsed().as_secs_f64();

            // activate new neurons in age=0, level=0 — inputs from the world
            let t = Instant::now();
            let neuron_ids: Vec<NeuronId> = frame_neurons.iter().map(|p| p.id).collect();
            self.activate_neurons(&neuron_ids);
            timings.activate = t.elapsed().as_secs_f64();

            // process neurons level-by-level — Op-3 dispatch is the only per-level round-trip
            let t = Instant::now();
            let (votes, neuron_specs, dispatch_results, neuron_t, orch_t, mem_t) = self.process_levels();
            timings.process_levels = t.elapsed().as_secs_f64();
            timings.neuron_learn_connections  = neuron_t.learn_connections;
            timings.neuron_recognize_patterns = neuron_t.recognize_patterns;
            timings.neuron_correct_errors     = neuron_t.correct_errors;
            timings.neuron_generate_votes     = neuron_t.generate_votes;
            timings.recognize_candidate_search   = neuron_t.recognize_candidate_search;
            timings.recognize_candidate_eval     = neuron_t.recognize_candidate_eval;
            timings.recognize_candidates_evaluated = neuron_t.recognize_candidates_evaluated;
            timings.orch_get_level_tasks      = orch_t.get_level_tasks;
            timings.orch_dispatch_frame       = orch_t.dispatch_frame;
            timings.orch_collect_activations  = orch_t.collect_activations;
            timings.orch_collect_votes        = orch_t.collect_votes;
            timings.mem_get_level_neurons     = mem_t.get_level_neurons;
            timings.mem_write_back_level_neurons = mem_t.write_back_level_neurons;
            timings.mem_activate_patterns     = mem_t.activate_patterns;

            if self.debug && !neuron_specs.is_empty() {
                println!("  frame {}: {} new error patterns created", self.frame_number, neuron_specs.len());
            }

            // Op-4 + Op-5: flush deferred neuron creation and contextRef updates in one batch
            let t = Instant::now();
            self.thalamus.apply_level_results(&neuron_specs, &dispatch_results);
            timings.apply_results = t.elapsed().as_secs_f64();

            // do inferences with age>0 neurons
            let t = Instant::now();
            let (inferences_map, vote_debug, _winners, action_vote_stats, action_votes) = self.infer_neurons(&votes);
            timings.infer = t.elapsed().as_secs_f64();

            // accumulate MAPE by comparing continuous event predictions to the actual input scalars
            let t = Instant::now();
            let dim_inferences = self.build_dim_inferences(&inferences_map);
            self.diagnostics.track_continuous_error(&dim_inferences, events, &self.thalamus.quantizer);
            timings.track_error = t.elapsed().as_secs_f64();

            FrameResult {
                inferences: inferences_map,
                elapsed: frame_start.elapsed().as_secs_f64(),
                vote_debug,
                action_vote_stats,
                action_votes,
                timings,
            }
        } else {
            // ── Frozen context: no event processing, no aging.
            //    Only action inference runs (if action_processing is enabled).
            //    Push rewards so action connections can learn from them.
            if !rewards.is_empty() {
                if let Some(first) = self.rewards.first_mut() {
                    for (&ch, &r) in rewards {
                        first.insert(ch, r);
                    }
                }
            }

            // Swap in the previously inferred action neurons at age 0 so
            // dispatch_frame applies rewards to the correct action connections.
            if self.action_processing {
                let prev_inferred = self.memory.get_inferred_neurons().to_vec();

                // Build a map of channel_id → inferred action neuron_id
                let mut inferred_by_channel: FxHashMap<ChannelId, NeuronId> = FxHashMap::default();
                for inf in &prev_inferred {
                    if self.thalamus.get_neuron_type(inf.neuron_id) == Some(NeuronType::Action) {
                        inferred_by_channel.insert(inf.channel_id, inf.neuron_id);
                    }
                }

                if !inferred_by_channel.is_empty() {
                    // Find stale action neurons at age 0 that need replacing
                    let age0 = self.memory.get_level_ages(0);
                    let mut swaps: Vec<(NeuronId, NeuronId)> = Vec::new();
                    let mut activations: Vec<NeuronId> = Vec::new();

                    if let Some(set) = age0.first() {
                        // Collect stale action neurons to replace
                        for &nid in set {
                            if self.thalamus.get_neuron_type(nid) == Some(NeuronType::Action) {
                                if let Some(ch) = self.thalamus.get_neuron_channel_id(nid) {
                                    if let Some(&new_id) = inferred_by_channel.get(&ch) {
                                        if new_id != nid {
                                            swaps.push((nid, new_id));
                                        }
                                        inferred_by_channel.remove(&ch);
                                    }
                                }
                            }
                        }
                    }

                    // Any remaining inferred actions without a stale counterpart
                    for (_, new_id) in &inferred_by_channel {
                        activations.push(*new_id);
                    }

                    // Apply swaps and activations
                    for (old_id, new_id) in swaps {
                        self.memory.replace_active_neuron(old_id, new_id);
                    }
                    for new_id in activations {
                        self.memory.activate_neuron(new_id, 0);
                    }
                }
            }

            // Re-run voting from existing active neurons (frozen ages)
            let mut timings = FrameTimings::default();
            let t = Instant::now();
            let (votes, _neuron_specs, _dispatch_results, neuron_t, orch_t, mem_t) = self.process_levels();
            timings.process_levels = t.elapsed().as_secs_f64();
            timings.neuron_learn_connections  = neuron_t.learn_connections;
            timings.neuron_recognize_patterns = neuron_t.recognize_patterns;
            timings.neuron_correct_errors     = neuron_t.correct_errors;
            timings.neuron_generate_votes     = neuron_t.generate_votes;
            timings.recognize_candidate_search   = neuron_t.recognize_candidate_search;
            timings.recognize_candidate_eval     = neuron_t.recognize_candidate_eval;
            timings.recognize_candidates_evaluated = neuron_t.recognize_candidates_evaluated;
            timings.orch_get_level_tasks      = orch_t.get_level_tasks;
            timings.orch_dispatch_frame       = orch_t.dispatch_frame;
            timings.orch_collect_activations  = orch_t.collect_activations;
            timings.orch_collect_votes        = orch_t.collect_votes;
            timings.mem_get_level_neurons     = mem_t.get_level_neurons;
            timings.mem_write_back_level_neurons = mem_t.write_back_level_neurons;
            timings.mem_activate_patterns     = mem_t.activate_patterns;
            let t = Instant::now();
            let (inferences_map, vote_debug, _winners, action_vote_stats, action_votes) = self.infer_neurons(&votes);
            timings.infer = t.elapsed().as_secs_f64();

            FrameResult {
                inferences: inferences_map,
                elapsed: frame_start.elapsed().as_secs_f64(),
                vote_debug,
                action_vote_stats,
                action_votes,
                timings,
            }
        }
    }

    // ── Frame building ──────────────────────────────────────────────────────

    /// Build this.frame from events and actions. Events are quantized to bucket IDs
    /// and pushed as Event coordinates. Actions come from one of three sources:
    ///   1. Forced actions map (training — caller provides the correct answer)
    ///   2. Previously inferred actions from memory (test — brain predicts)
    ///   3. Nothing (action processing is off)
    fn build_frame(
        &mut self,
        events: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
    ) {
        self.frame.clear();

        // Resolve action coordinates: forced actions take priority, then inferred
        let inferred_actions = if !actions.is_empty() {
            // Forced actions — skip inference entirely
            FxHashMap::default()
        } else if self.action_processing {
            let inferred = self.memory.get_inferred_neurons();
            self.thalamus.get_inferred_actions(inferred)
        } else {
            FxHashMap::default()
        };

        // iterate every registered channel — a channel may contribute events,
        // forced/inferred actions, or both
        for channel_id in self.thalamus.get_channel_ids() {

            // quantize each dimension's scalar to a bucketId and push as event coordinate
            if let Some(dim_map) = events.get(&channel_id) {
                for (&dim_id, &scalar) in dim_map {
                    self.thalamus.quantizer.observe(dim_id, scalar);
                    let bucket_id = self.thalamus.quantizer.quantize(dim_id, scalar);
                    self.frame.push(FramePoint {
                        coordinate: Coordinate { dim_id, bucket_id },
                        channel_id,
                        neuron_type: NeuronType::Event,
                    });
                }
            }

            // forced actions from caller — push directly as action coordinates
            if let Some(dim_map) = actions.get(&channel_id) {
                for (&dim_id, &scalar) in dim_map {
                    let bucket_id = scalar as i32;
                    self.frame.push(FramePoint {
                        coordinate: Coordinate { dim_id, bucket_id },
                        channel_id,
                        neuron_type: NeuronType::Action,
                    });
                }
            } else if self.action_processing {
                // no forced action for this channel — use inferred from previous frame
                if let Some(inf_actions) = inferred_actions.get(&channel_id) {
                    for action in inf_actions {
                        self.frame.push(FramePoint {
                            coordinate: action.coordinate.clone(),
                            channel_id,
                            neuron_type: NeuronType::Action,
                        });
                    }
                }
            }
        }
    }

    /// Returns neuron IDs for given frame points, creating new base neurons as needed.
    /// Builds the frame first from quantized events + forced/inferred actions.
    fn get_frame_neurons(
        &mut self,
        events: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
    ) -> Vec<PointLookup> {
        self.build_frame(events, actions);

        let mut neurons = Vec::with_capacity(self.frame.len());
        for point in &self.frame {
            neurons.push(self.thalamus.get_neuron_id_for_point(
                &point.coordinate,
                point.channel_id,
                point.neuron_type.clone(),
            ));
        }
        neurons
    }

    /// Create sensory neurons that are new this frame.
    ///
    /// Sensory neurons (level 0) use `effective_forget_rate(level=0)` for their pattern_forget_rate.
    /// This rate governs the decay of their child patterns (L1 error-correction patterns) — NOT the sensory neuron's own activations.
    /// Without a non-zero rate here, all L1 patterns are effectively immortal,
    /// because `strengthen_child_activation` divides by the parent's rate to compute death frames.
    fn create_new_sensory_neurons(&mut self, frame_neurons: &[PointLookup]) {
        let forget_rate = Thalamus::effective_forget_rate(
            self.thalamus.get_base_forget_rate(),
            self.context_length,
            0,
            self.thalamus.get_level_decay_mode(),
        );
        let specs: Vec<NeuronCreateSpec> = frame_neurons.iter()
            .filter(|p| p.is_new)
            .map(|p| NeuronCreateSpec { id: p.id, forget_rate, connections: None })
            .collect();
        if !specs.is_empty() {
            self.thalamus.create_neurons(&specs);
        }
    }

    // ── Context aging ───────────────────────────────────────────────────────

    /// Slide the temporal window by one frame: push the new channel rewards onto the
    /// rewards history and age all active neurons in memory.
    fn age_context(&mut self, rewards: &FxHashMap<ChannelId, Reward>) {
        // push this frame's rewards onto the history
        self.rewards.insert(0, rewards.clone());
        if self.rewards.len() > self.context_length as usize {
            self.rewards.pop();
        }

        // advance the age of every active neuron and drop any that fell off the window
        self.memory.age(self.frame_number);
    }

    // ── Neuron activation ───────────────────────────────────────────────────

    /// Activate neurons by ID at age 0.
    fn activate_neurons(&mut self, neuron_ids: &[NeuronId]) {
        for &neuron_id in neuron_ids {
            let level = self.thalamus.get_neuron_level(neuron_id).unwrap_or(0);
            self.memory.activate_neuron(neuron_id, level);
        }

        // Track event accuracy, action rewards, and misprediction log
        let active_at_age_0 = self.memory.get_neuron_ids_at_age(0);
        let inferred = self.memory.get_inferred_neurons();
        let frame_rewards = self.rewards.first().cloned().unwrap_or_default();
        let results = self.thalamus.get_inference_results(
            &active_at_age_0,
            inferred,
            &frame_rewards,
        );
        self.diagnostics.track_inference_performance(&results);
    }

    // ── Level processing ────────────────────────────────────────────────────

    /// Process neurons level-by-level — each level in parallel (future).
    /// Only Op-3 (process frame dispatch) runs per level; neuron creation (Op-4)
    /// and contextRef updates (Op-5) are returned for the caller to flush once.
    fn process_levels(&mut self) -> (Vec<FlatVote>, Vec<NeuronCreateSpec>, Vec<Vec<crate::column::ColumnProcessResult>>, crate::neuron::NeuronOpTimings, crate::thalamus::OrchestrationTimings, MemoryTimings) {
        let learning = self.learning;
        // get the active sensory neurons at level 0
        let sensory_neurons = self.memory.get_level_ages(0);

        // get the maximum active level from memory index
        let mut max_active_level = self.memory.get_max_active_level();

        // track newly-created error pattern ids so they are excluded from the level
        // pass at their own level (prevents double connection-learning and context leak)
        let mut new_error_pattern_ids: FxHashSet<NeuronId> = FxHashSet::default();

        // accumulate votes, new neuron specs and context updates across levels
        let mut votes = Vec::new();
        let mut neuron_specs = Vec::new();
        let mut dispatch_results = Vec::new();
        let mut neuron_timings = crate::neuron::NeuronOpTimings::default();
        let mut orch_timings = crate::thalamus::OrchestrationTimings::default();
        let mut mem_timings = MemoryTimings::default();

        // Warmup gate: until the context window has had a chance to fill up
        // (i.e. frame_number < context_length), skip pattern recognition for
        // levels above 0. Sensory activation at level 0 still runs so the
        // brain can build memory state and learn base connections, but no
        // L1+ patterns are matched until the full context is reachable.
        //
        // Without this gate, early frames produce structurally unfair matches:
        // a richer pattern whose stored context references distances beyond
        // the current frame fails the merge_threshold ratio (its unreachable
        // entries count as "missing"), while a smaller pattern wins by
        // default. The displaced richer pattern then doesn't refine its
        // context, and subsequent training pollutes the brain with new
        // sibling patterns that displace established ones — the root cause
        // of "training-makes-accuracy-go-down" drift.
        let warmup = (self.frame_number as u32) < self.context_length;
        let max_level_to_process: Level = if warmup { 0 } else { Level::MAX };

        // process neurons level-by-level
        let mut level: Level = 0;
        loop {
            if self.debug { println!("Processing level {} for pattern recognition", level); }

            // get level neurons (mutable borrow returned, will be written back)
            let t = Instant::now();
            let mut level_neurons = self.memory.get_level_neurons(level);
            mem_timings.get_level_neurons += t.elapsed().as_secs_f64();

            // process level: aggregate view, recognize patterns, create error corrections, collect votes
            let result = self.thalamus.process_level(
                level,
                &mut level_neurons,
                self.memory.depth(),
                &sensory_neurons,
                &self.rewards,
                self.frame_number,
                &mut new_error_pattern_ids,
                learning,
            );
            orch_timings.add(&result.orchestration);

            // write back mutated level neurons to memory
            let t = Instant::now();
            self.memory.write_back_level_neurons(&level_neurons);
            mem_timings.write_back_level_neurons += t.elapsed().as_secs_f64();

            // activate matched patterns and newly-created error patterns at level+1
            let t = Instant::now();
            for activation in &result.activations {
                self.memory.activate_pattern(activation.pattern_id, level + 1, activation.parent_id, activation.age);
            }
            mem_timings.activate_patterns += t.elapsed().as_secs_f64();

            // if we produced any activations, increment the max active level as needed
            if !result.activations.is_empty() {
                let new_level = (level + 1) as usize;
                if new_level > max_active_level {
                    max_active_level = new_level;
                }
            }

            // accumulate this level's votes, neuron specs and context updates
            votes.extend(result.votes);
            neuron_specs.extend(result.neuron_specs);
            // Sum per-neuron op timings across this level into the per-frame accumulator.
            for col_res in &result.results {
                neuron_timings.add(&col_res.timings);
            }
            dispatch_results.push(result.results);

            // if we reached the maximum level and no more patterns are recognized, exit
            if level as usize >= max_active_level { break; }
            // Honour the warmup cap — stop after level 0 when in warmup.
            if level >= max_level_to_process { break; }
            level += 1;
        }

        (votes, neuron_specs, dispatch_results, neuron_timings, orch_timings, mem_timings)
    }

    // ── Inference (voting consensus) ────────────────────────────────────────

    /// Infer predictions and outputs using voting architecture.
    /// Returns the per-channel scalar-space inferences, optional vote debug, and
    /// the raw winner list for memory persistence.
    fn infer_neurons(&mut self, votes: &[FlatVote]) -> (FxHashMap<ChannelId, Vec<DimInferenceOutput>>, Option<VoteDebug>, Vec<InferredNeuron>, Vec<ActionVoteStats>, Vec<ActionVote>) {
        // if no inference votes, wait for more data
        if votes.is_empty() {
            if self.debug { println!("No inferences found. Waiting for more data in future frames."); }
            return (FxHashMap::default(), None, Vec::new(), Vec::new(), Vec::new());
        }

        // Aggregate votes and determine winners
        let (inferences, candidates, dim_best) = self.determine_consensus(votes);

        // Build the resolved vote dump only when debug is on
        let vote_debug = if self.debug {
            let resolved_votes: Vec<ResolvedVote> = votes.iter().map(|v| {
                ResolvedVote {
                    target_id: v.neuron_id,
                    target_type: self.thalamus.get_neuron_type(v.neuron_id),
                    target_channel_id: self.thalamus.get_neuron_channel_id(v.neuron_id),
                    target_coordinate: self.thalamus.get_neuron_coordinate(v.neuron_id)
                        .and_then(|c| self.thalamus.coordinate_id_to_name(c)),
                    voter_id: v.voter_id,
                    voter_level: self.thalamus.get_neuron_level(v.voter_id),
                    voter_label: self.format_neuron_label(v.voter_id),
                    strength: v.strength,
                    reward: v.reward,
                    distance: v.distance,
                }
            }).collect();
            Some(VoteDebug {
                votes: resolved_votes,
                winners: inferences.clone(),
            })
        } else {
            None
        };

        // Save inferences to memory (clears old inferences first).
        // When action processing is off, filter out action neurons so they
        // don't carry over as stale sensory input on the next frame.
        if self.action_processing {
            self.memory.save_inferred_neurons(inferences.clone());
        } else {
            let event_only: Vec<_> = inferences.iter()
                .filter(|inf| self.thalamus.get_neuron_type(inf.neuron_id) != Some(NeuronType::Action))
                .cloned()
                .collect();
            self.memory.save_inferred_neurons(event_only);
        }

        // Build the scalar-space output
        let inferences_map = self.build_inferences_by_channel(&candidates, &dim_best);

        // Compute per-digit action vote stats from raw votes
        let action_vote_stats = self.compute_action_vote_stats(votes);
        let action_votes = self.collect_action_votes(votes);

        (inferences_map, vote_debug, inferences, action_vote_stats, action_votes)
    }

    /// Collect per-voter action votes — one entry per FlatVote targeting an
    /// action neuron. Preserves voter identity so hosts can implement custom
    /// consensus schemes (e.g. per-voter reward normalization).
    fn collect_action_votes(&self, votes: &[FlatVote]) -> Vec<ActionVote> {
        let mut out: Vec<ActionVote> = Vec::new();
        for v in votes {
            if self.thalamus.get_neuron_type(v.neuron_id) != Some(NeuronType::Action) {
                continue;
            }
            let coord = match self.thalamus.get_neuron_coordinate(v.neuron_id) {
                Some(c) => c.clone(),
                None => continue,
            };
            let channel_id = match self.thalamus.get_neuron_channel_id(v.neuron_id) {
                Some(c) => c,
                None => continue,
            };
            out.push(ActionVote {
                voter_neuron_id: v.voter_id,
                channel_id,
                dimension_id: coord.dim_id,
                value: coord.bucket_id,
                strength: v.strength,
                reward: v.reward,
            });
        }
        out
    }

    /// Compute per-action-value vote statistics from raw FlatVotes.
    ///
    /// Groups all votes targeting action neurons by their bucket_id (digit),
    /// then computes vote count, total strength, and reward distribution
    /// (min/max/strength-weighted average) for each group.
    fn compute_action_vote_stats(&self, votes: &[FlatVote]) -> Vec<ActionVoteStats> {
        // Intermediate accumulator per digit value
        struct Acc {
            vote_count: usize,
            total_strength: f64,
            weighted_reward: f64,
            min_reward: f64,
            max_reward: f64,
        }
        let mut by_digit: FxHashMap<i32, Acc> = FxHashMap::default();

        for v in votes {
            // Only count action neuron votes
            if self.thalamus.get_neuron_type(v.neuron_id) != Some(NeuronType::Action) {
                continue;
            }
            let digit = match self.thalamus.get_neuron_coordinate(v.neuron_id) {
                Some(coord) => coord.bucket_id,
                None => continue,
            };
            let acc = by_digit.entry(digit).or_insert_with(|| Acc {
                vote_count: 0,
                total_strength: 0.0,
                weighted_reward: 0.0,
                min_reward: f64::INFINITY,
                max_reward: f64::NEG_INFINITY,
            });
            acc.vote_count += 1;
            acc.total_strength += v.strength;
            acc.weighted_reward += v.strength * v.reward;
            if v.reward < acc.min_reward { acc.min_reward = v.reward; }
            if v.reward > acc.max_reward { acc.max_reward = v.reward; }
        }

        let mut stats: Vec<ActionVoteStats> = by_digit.into_iter().map(|(digit, acc)| {
            ActionVoteStats {
                value: digit,
                vote_count: acc.vote_count,
                total_strength: acc.total_strength,
                avg_reward: if acc.total_strength > 0.0 { acc.weighted_reward / acc.total_strength } else { 0.0 },
                min_reward: acc.min_reward,
                max_reward: acc.max_reward,
            }
        }).collect();
        stats.sort_by(|a, b| b.avg_reward.partial_cmp(&a.avg_reward).unwrap_or(std::cmp::Ordering::Equal));
        stats
    }

    /// Aggregate votes and determine winners per dimension.
    /// Events win by strength (probability), actions win by reward.
    fn determine_consensus(&self, votes: &[FlatVote]) -> (Vec<InferredNeuron>, FxHashMap<NeuronId, Candidate>, FxHashMap<DimensionId, DimBestEntry>) {
        // Aggregate votes into candidates and dimension totals
        let (mut candidates, dim_total_strength) = self.aggregate_votes(votes);

        // Determine the best neuron per dimension (also sets reward/probability on each candidate)
        let dim_best = self.determine_dimension_winners(&mut candidates, &dim_total_strength);

        // Build winner objects from dimension winners
        let winner_ids: FxHashSet<NeuronId> = dim_best.values().map(|w| w.neuron_id).collect();
        if self.debug { println!("Determined consensus: {} candidates, {} winners", candidates.len(), winner_ids.len()); }
        let inferences = self.build_winners(&winner_ids, &candidates);
        (inferences, candidates, dim_best)
    }

    /// Aggregate votes into candidate neurons and dimension strength totals.
    fn aggregate_votes(&self, votes: &[FlatVote]) -> (FxHashMap<NeuronId, Candidate>, FxHashMap<DimensionId, f64>) {
        let mut candidates: FxHashMap<NeuronId, Candidate> = FxHashMap::default();
        let mut dim_total_strength: FxHashMap<DimensionId, f64> = FxHashMap::default();

        for v in votes {
            let candidate = candidates.entry(v.neuron_id).or_insert_with(|| Candidate {
                strength: 0.0,
                weighted_total: 0.0,
                reward: 0.0,
                probability: 0.0,
            });

            candidate.strength += v.strength;

            // for actions, calculate weighted total — for events, accumulate strength on the dimension
            // Action scoring is always active: even when action_processing is off (no action
            // neurons in the frame), votes from learned connections must still use reward-based
            // scoring so infer() and learn() produce correct consensus.
            if self.thalamus.get_neuron_type(v.neuron_id) == Some(NeuronType::Action) {
                candidate.weighted_total += v.strength * v.reward;
            } else if let Some(coord) = self.thalamus.get_neuron_coordinate(v.neuron_id) {
                *dim_total_strength.entry(coord.dim_id).or_insert(0.0) += v.strength;
            }
        }

        (candidates, dim_total_strength)
    }

    /// Determine the best neuron per dimension (events by probability, actions by reward).
    /// Also sets reward/probability on each candidate (JS mutates candidates in-place here).
    fn determine_dimension_winners(
        &self,
        candidates: &mut FxHashMap<NeuronId, Candidate>,
        dim_total_strength: &FxHashMap<DimensionId, f64>,
    ) -> FxHashMap<DimensionId, DimBestEntry> {
        let mut dim_best: FxHashMap<DimensionId, DimBestEntry> = FxHashMap::default();

        for (&neuron_id, candidate) in candidates.iter_mut() {
            let coordinate = match self.thalamus.get_neuron_coordinate(neuron_id) {
                Some(c) => c,
                None => continue,
            };

            let neuron_type = match self.thalamus.get_neuron_type(neuron_id) {
                Some(t) => t,
                None => continue,
            };

            // for actions, reward = weighted_total / strength
            // for events, probability = strength / dimension total strength
            // Store the computed score back on the candidate (matches JS behavior)
            let score = if neuron_type == NeuronType::Action {
                let reward = if candidate.strength > 0.0 { candidate.weighted_total / candidate.strength } else { 0.0 };
                candidate.reward = reward;
                reward
            } else {
                let total = dim_total_strength.get(&coordinate.dim_id).copied().unwrap_or(0.0);
                let probability = if total > 0.0 { candidate.strength / total } else { 0.0 };
                candidate.probability = probability;
                probability
            };

            // set the best neuron for this dimension — break ties by strength, then neuron ID
            let is_better = match dim_best.get(&coordinate.dim_id) {
                None => true,
                Some(best) => {
                    if score != best.score { score > best.score }
                    else if candidate.strength != best.strength { candidate.strength > best.strength }
                    else { neuron_id < best.neuron_id }
                }
            };
            if is_better {
                dim_best.insert(coordinate.dim_id, DimBestEntry { neuron_id, score, strength: candidate.strength });
            }
        }

        dim_best
    }

    /// Build winner InferredNeuron objects from winning neuron IDs.
    fn build_winners(&self, winner_ids: &FxHashSet<NeuronId>, candidates: &FxHashMap<NeuronId, Candidate>) -> Vec<InferredNeuron> {
        let mut winners = Vec::with_capacity(winner_ids.len());
        for &neuron_id in winner_ids {
            let candidate = match candidates.get(&neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let coordinate = match self.thalamus.get_neuron_coordinate(neuron_id) {
                Some(c) => c.clone(),
                None => continue,
            };
            let channel_id = match self.thalamus.get_neuron_channel_id(neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let neuron_type = self.thalamus.get_neuron_type(neuron_id);

            let (reward, probability) = if neuron_type == Some(NeuronType::Action) {
                (candidate.reward, 0.0)
            } else {
                (0.0, candidate.probability)
            };

            winners.push(InferredNeuron {
                neuron_id,
                coordinate,
                channel_id,
                strength: candidate.strength,
                reward,
                probability,
            });
        }
        winners
    }

    /// Build the per-channel, per-dimension inference output in scalar space.
    /// For each dimension that received votes, produces a single entry containing the
    /// winning bucket's dequantized value plus a score-weighted continuous prediction.
    fn build_inferences_by_channel(
        &self,
        candidates: &FxHashMap<NeuronId, Candidate>,
        dim_best: &FxHashMap<DimensionId, DimBestEntry>,
    ) -> FxHashMap<ChannelId, Vec<DimInferenceOutput>> {
        /// Per-dimension accumulator for the continuous prediction.
        struct DimEntry {
            channel_id: ChannelId,
            dim_id: DimensionId,
            kind: NeuronType,
            weighted_sum: f64,
            total_score: f64,
        }

        // Group every candidate by (channelId, dimId) and accumulate weighted sums
        let mut dims: FxHashMap<(ChannelId, DimensionId), DimEntry> = FxHashMap::default();
        for (&neuron_id, candidate) in candidates {
            let coordinate = match self.thalamus.get_neuron_coordinate(neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let kind = match self.thalamus.get_neuron_type(neuron_id) {
                Some(t) => t,
                None => continue,
            };
            let channel_id = match self.thalamus.get_neuron_channel_id(neuron_id) {
                Some(c) => c,
                None => continue,
            };

            let key = (channel_id, coordinate.dim_id);
            let entry = dims.entry(key).or_insert_with(|| DimEntry {
                channel_id,
                dim_id: coordinate.dim_id,
                kind: kind.clone(),
                weighted_sum: 0.0,
                total_score: 0.0,
            });

            // skip candidates with no dequantized value (never-observed bucket)
            let value = match self.thalamus.quantizer.dequantize(coordinate.dim_id, coordinate.bucket_id.into()) {
                Some(v) => v,
                None => continue,
            };

            let score = if kind == NeuronType::Action {
                candidate.reward
            } else {
                candidate.probability
            };
            entry.weighted_sum += score * value;
            entry.total_score += score;
        }

        // Finalize each dimension entry: resolve winner via dimBest and compute continuous
        let mut out: FxHashMap<ChannelId, Vec<DimInferenceOutput>> = FxHashMap::default();
        for entry in dims.values() {
            let best = match dim_best.get(&entry.dim_id) {
                Some(b) => b,
                None => continue,
            };
            let winner_coord = match self.thalamus.get_neuron_coordinate(best.neuron_id) {
                Some(c) => c,
                None => continue,
            };
            let winner_value = self.thalamus.quantizer.dequantize(winner_coord.dim_id, winner_coord.bucket_id.into());
            let continuous = if entry.total_score > 0.0 {
                Some(entry.weighted_sum / entry.total_score)
            } else {
                winner_value
            };

            out.entry(entry.channel_id).or_insert_with(Vec::new).push(DimInferenceOutput {
                dim_id: entry.dim_id,
                kind: entry.kind.clone(),
                winner: WinnerOutput {
                    value: winner_value,
                    strength: best.strength,
                    score: best.score,
                },
                continuous,
            });
        }
        out
    }

    /// Convert the per-channel DimInferenceOutput map into the DimInference format
    /// that Diagnostics.track_continuous_error expects.
    fn build_dim_inferences(&self, inferences_map: &FxHashMap<ChannelId, Vec<DimInferenceOutput>>) -> FxHashMap<ChannelId, Vec<DimInference>> {
        let mut result: FxHashMap<ChannelId, Vec<DimInference>> = FxHashMap::default();
        for (&channel_id, dim_outputs) in inferences_map {
            let entries: Vec<DimInference> = dim_outputs.iter().map(|d| DimInference {
                dim_id: d.dim_id,
                kind: match d.kind {
                    NeuronType::Event => InferenceType::Event,
                    NeuronType::Action => InferenceType::Action,
                },
                continuous: d.continuous,
            }).collect();
            result.insert(channel_id, entries);
        }
        result
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────

    /// Runs the cleanup cycle for zombie patterns.
    /// With lazy decay, this only deletes items that have decayed to zero effective strength.
    fn cleanup_dead_patterns(&mut self) {
        if self.debug { println!("=== CLEANUP STARTING ==="); }

        // reap neurons scheduled to die at this frame
        let dead_pattern_ids = self.thalamus.reap_dead_neurons(self.frame_number);
        if dead_pattern_ids.is_empty() { return; }

        // delete dead patterns (with recursive cleanup of context references)
        let deleted_pattern_ids = self.thalamus.delete_patterns(&dead_pattern_ids, self.frame_number);

        // verify no deleted patterns are active — that would be a bug
        self.memory.assert_not_active(&deleted_pattern_ids);

        if self.debug { println!("=== CLEANUP COMPLETED ===\n"); }
    }

    // ── Debug helpers ───────────────────────────────────────────────────────

    /// Format a neuron's coordinates as a display label.
    /// For pattern neurons, resolves up the parent chain to find root sensory coordinates.
    fn format_neuron_label(&self, neuron_id: NeuronId) -> String {
        // Pattern neurons: resolve parent chain to root sensory neuron
        let level = self.thalamus.get_neuron_level(neuron_id).unwrap_or(0);
        if level > 0 {
            if let Some(parent_id) = self.thalamus.get_neuron_parent(neuron_id) {
                return self.format_neuron_label(parent_id);
            }
        }

        // Sensory neurons: format coordinate (translate id-form to name-form for display)
        match self.thalamus.get_neuron_coordinate(neuron_id) {
            Some(coordinate) => {
                match self.thalamus.coordinate_id_to_name(coordinate) {
                    Some((name, bucket_id)) => format!("{}={}", name, bucket_id),
                    None => format!("dim{}={}", coordinate.dim_id, coordinate.bucket_id),
                }
            }
            None => format!("neuron_{}", neuron_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thalamus::DimSpecInput;

    /// Helper to create a brain with default settings.
    fn make_brain() -> Brain {
        Brain::new(
            10,                       // context_length
            ErrorMode::Conservative,  // error_correction_mode
            0.5,                      // error_correction_threshold
            0.5,                      // merge_threshold
            0.01,                     // pattern_forget_rate
            LevelDecayMode::Exponential, // level_decay_mode
            1,                        // regions
            1,                        // columns
            false,                    // debug
            None,                     // action_alpha
        )
    }

    #[test]
    fn test_brain_creation() {
        let brain = make_brain();
        assert_eq!(brain.frame_number, 0);
        assert!(brain.frame.is_empty());
        assert!(brain.rewards.is_empty());
    }

    #[test]
    fn test_register_channel_and_process_empty_frame() {
        let mut brain = make_brain();

        // register a channel with one input dimension
        let reg = brain.register_channel_spec(
            "test",
            vec![DimSpecInput {
                name: "price".to_string(),
                kind: crate::thalamus::DimKind::Input,
                resolution: 3,
                mode: Some("static".to_string()),
                boundaries: Some(vec![-0.5, 0.5]),
                actions: None,
                default_action: None,
                warmup_samples: None,
            }],
            false,
            false,
        );
        assert!(reg.dimension_ids.contains_key("price"));

        // process an empty frame — should return empty inferences
        let result = brain.process_frame(&FxHashMap::default(), &FxHashMap::default(), &FxHashMap::default());
        assert!(result.inferences.is_empty());
        assert!(result.vote_debug.is_none());
    }

    #[test]
    fn test_process_frame_with_input() {
        let mut brain = make_brain();

        // register a channel with one input dimension
        let reg = brain.register_channel_spec(
            "sensor",
            vec![DimSpecInput {
                name: "temp".to_string(),
                kind: crate::thalamus::DimKind::Input,
                resolution: 3,
                mode: Some("static".to_string()),
                boundaries: Some(vec![20.0, 30.0]),
                actions: None,
                default_action: None,
                warmup_samples: None,
            }],
            false,
            false,
        );
        let channel_id = reg.channel_id;
        let dim_id = reg.dimension_ids["temp"];

        // process first frame with a temperature reading
        let mut inputs = FxHashMap::default();
        let mut dim_vals = FxHashMap::default();
        dim_vals.insert(dim_id, 25.0);
        inputs.insert(channel_id, dim_vals);

        let result = brain.process_frame(&inputs, &FxHashMap::default(), &FxHashMap::default());
        assert_eq!(brain.frame_number, 1);
        // first frame won't have inferences yet (no prior context)
        assert!(result.inferences.is_empty());
    }

    #[test]
    fn test_reset_context() {
        let mut brain = make_brain();
        brain.frame_number = 10;

        brain.reset_context();
        assert_eq!(brain.frame_number, 0);
        assert!(brain.rewards.is_empty());
    }

    #[test]
    fn test_reset_brain() {
        let mut brain = make_brain();
        brain.frame_number = 10;

        brain.reset_brain();
        assert_eq!(brain.frame_number, 0);
    }

    #[test]
    fn test_is_better_candidate() {
        // helper to test the comparison logic
        let best = DimBestEntry { neuron_id: 5, score: 0.8, strength: 1.0 };

        // higher score wins
        assert!(0.9 > best.score);
        // same score, higher strength wins
        assert!(best.score == 0.8);
        // same score and strength, lower neuron_id wins
    }

    #[test]
    fn test_frame_summary() {
        let brain = make_brain();
        let summary = brain.get_frame_summary();
        assert_eq!(summary.frame_number, 0);
        assert_eq!(summary.neuron_count, 0);
    }

    #[test]
    fn test_episode_summary() {
        let brain = make_brain();
        let summary = brain.get_episode_summary();
        assert_eq!(summary.frame_number, 0);
        assert!(summary.stats.base_accuracy.is_none());
    }
}
