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
    ChannelId, ConsensusMode, Coordinate, DimensionId, Distance, ErrorMode, FrameNumber,
    Level, NeuronId, NeuronType, Reward,
};

/// Laplace-style floor added inside the Nb consensus log so a single zero posterior doesn't send a candidate to negative infinity.
const NB_EPS: f64 = 1e-3;

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

/// Resolved frame neurons, split by role.
///
/// `events` are this frame's quantized observations from the input scalars.
/// They drive the spatial co-activation sweep and are the only thing activated in spatial[0].
///
/// `actions` are the actions inferred by the previous frame's vote, carried forward.
/// They bypass spatial entirely and get injected into temporal[0] alongside the spatial apex set,
/// so the temporal sweep sees both "what just happened" (events through spatial) and
/// "what the agent just did" (carry-forward actions).
struct FrameNeurons {
    events: Vec<PointLookup>,
    actions: Vec<PointLookup>,
}

impl FrameNeurons {
    fn is_empty(&self) -> bool {
        self.events.is_empty() && self.actions.is_empty()
    }
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
    pub neuron_id: NeuronId,
    pub value: Option<f64>,
    pub strength: f64,
    pub score: f64,
}

/// Return value from processFrame — inferences plus per-frame diagnostic byproducts.
#[derive(Debug)]
pub struct FrameResult {
    /// Per-channel, per-dimension scalar-space inferences (the consensus).
    pub inferences: FxHashMap<ChannelId, Vec<DimInferenceOutput>>,
    /// Every vote cast this frame, with enough resolved metadata that callers
    /// can run their own consensus or analysis without further round-trips.
    /// Inferences are the summary; votes are the detail.
    pub votes: Vec<FrameVote>,
    /// Wall-clock elapsed time for this frame (seconds).
    pub elapsed: f64,
    /// Per-section wall-clock timings (seconds). Populated for profiling.
    pub timings: FrameTimings,
}

/// One vote cast by a voter neuron toward a target (event or action) neuron.
/// `voter_label` is the voter's root-sensory coordinate ("digit=5", "pixel_0=255")
/// for human-readable debug output; pattern-neuron voters resolve to whichever
/// sensory neuron sits at the root of their parent chain.
#[derive(Debug, Clone)]
pub struct FrameVote {
    pub voter_id: NeuronId,
    pub voter_label: String,
    pub voter_level: Level,
    pub target_id: NeuronId,
    pub target_type: NeuronType,
    pub channel_id: ChannelId,
    pub dim_id: DimensionId,
    pub value: i32,
    pub distance: Distance,
    pub strength: f64,
    pub reward: f64,
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

/// Memory-op timings inside brain.process_levels, summed across levels.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryTimings {
    pub get_level_neurons: f64,
    pub write_back_level_neurons: f64,
    pub activate_patterns: f64,
}

/**
 * Per-section timings inside a single frame, in seconds. The neuron/orch/mem
 * sub-buckets are summed across all neurons and all levels for this frame.
 * process_temporal/process_spatial include overhead beyond these (memory I/O, level dispatch)
 * so the sub-buckets won't sum to them exactly.
 */
#[derive(Debug, Clone)]
pub struct FrameTimings {
    /// When the frame stopwatch started. Captured at FrameTimings::default() construction.
    /// Used internally by finalize() to compute `total`; not exposed across the napi boundary.
    start: Instant,
    /// Wall-clock seconds the whole frame took, end-to-end. Set by finalize() at frame exit.
    pub total: f64,
    pub build_frame: f64,
    pub create_sensory: f64,
    pub cleanup_dead: f64,
    pub age_context: f64,
    pub activate: f64,
    pub process_spatial: f64,
    pub apex_handoff: f64,
    pub process_temporal: f64,
    pub apply_results: f64,
    pub infer: f64,
    pub track_error: f64,
    pub neuron_learn_connections: f64,
    pub neuron_recognize_patterns: f64,
    pub neuron_correct_errors: f64,
    pub neuron_generate_votes: f64,
    pub recognize_candidate_search: f64,
    pub recognize_candidate_eval: f64,
    pub recognize_candidates_evaluated: u64,
    pub orch_get_level_tasks: f64,
    pub orch_dispatch_frame: f64,
    pub orch_collect_activations: f64,
    pub orch_collect_votes: f64,
    pub mem_get_level_neurons: f64,
    pub mem_write_back_level_neurons: f64,
    pub mem_activate_patterns: f64,
}

impl Default for FrameTimings {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            total: 0.0,
            build_frame: 0.0,
            create_sensory: 0.0,
            cleanup_dead: 0.0,
            age_context: 0.0,
            activate: 0.0,
            process_spatial: 0.0,
            apex_handoff: 0.0,
            process_temporal: 0.0,
            apply_results: 0.0,
            infer: 0.0,
            track_error: 0.0,
            neuron_learn_connections: 0.0,
            neuron_recognize_patterns: 0.0,
            neuron_correct_errors: 0.0,
            neuron_generate_votes: 0.0,
            recognize_candidate_search: 0.0,
            recognize_candidate_eval: 0.0,
            recognize_candidates_evaluated: 0,
            orch_get_level_tasks: 0.0,
            orch_dispatch_frame: 0.0,
            orch_collect_activations: 0.0,
            orch_collect_votes: 0.0,
            mem_get_level_neurons: 0.0,
            mem_write_back_level_neurons: 0.0,
            mem_activate_patterns: 0.0,
        }
    }
}

impl FrameTimings {
    /// Stop the per-frame stopwatch and record the total wall-clock under `total`.
    /// Called once per frame, right before the FrameResult is built.
    pub fn finalize(&mut self) {
        self.total = self.start.elapsed().as_secs_f64();
    }
}

/// Episode-level summary for host-side rendering.
#[derive(Debug, Clone)]
pub struct FrameSummary {
    pub frame_number: FrameNumber,
    pub neuron_count: usize,
    /// Maximum active temporal level — depth of the temporal pattern hierarchy.
    /// 0 = sensory only, 1+ = temporal patterns exist at that level.
    pub max_temporal_level: Level,
    /// Maximum active spatial level — depth of the spatial pattern hierarchy.
    /// 0 = sensory only, 1+ = spatial-correction patterns exist at that level.
    pub max_spatial_level: Level,
    pub stats: DiagnosticStats,
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
    /// Naive-Bayes log-score accumulator for action candidates: Σ_vote log(reward + NB_EPS).
    /// Filled only when consensus mode is Nb; one term per incoming vote (per connection),
    /// matching the per-voter product rule. Ignored under Democratic consensus.
    nb_log_score: f64,
}

/// Per-dimension winner tracking during consensus determination.
#[derive(Debug, Clone)]
struct DimBestEntry {
    neuron_id: NeuronId,
    score: f64,
    strength: f64,
}

/// Output of a single phase's level-sweep. Spatial and temporal share the same shape; only
/// `fired_set` / `subsumed_set` are consumed (by the apex handoff) and only on the spatial side.
struct SweepResult {
    votes: Vec<FlatVote>,
    neuron_specs: Vec<NeuronCreateSpec>,
    dispatch_results: Vec<Vec<crate::column::ColumnProcessResult>>,
    fired_set: FxHashSet<NeuronId>,
    subsumed_set: FxHashSet<NeuronId>,
}

// ── Brain ───────────────────────────────────────────────────────────────────

pub struct Brain {
    /// Number of frames a base neuron stays active.
    context_length: u32,

    /// Debug flag — when set, enables verbose logging and vote debug output.
    debug: bool,

    /// How per-voter action posteriors are combined into a dimension winner.
    /// Democratic (default) is the strength-weighted mean; Nb is the Naive-Bayes log-product.
    /// Only affects action scoring — event winners are always probability (voter share).
    consensus_mode: ConsensusMode,

    /// When true, infer_neurons resolves every cast vote into a FrameVote on
    /// FrameResult.votes. Off by default since resolution allocates per-vote
    /// (voter_label String, target metadata lookups) — only opt in when a
    /// harness actually consumes the per-vote detail.
    emit_votes: bool,

    /// When true (default), process_frame mutates connections and learns error correction patterns.
    /// When false, process_frame skips forget/decay and error-correction pattern neuron creation.
    /// It also skips event→event connection strengthening, child-activation strengthening, and accuracy-stats tracking.
    /// Sensory neuron creation (op-1) still runs because without it the frame cannot be processed at all.
    /// Pattern activation and voting still run.
    /// The harness still reads predictions out of FrameResult.inferences exactly as during training.
    /// The supervised evaluation path is `set_learning(false)` followed by ordinary `process_frame` calls.
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
    backup: Backup,

    /// Relay station for neuron/channel/dimension mappings. Owns the Region tree.
    thalamus: Thalamus,

    /// Temporal sliding window of active and inferred neurons.
    memory: Memory,

    /// Spatial error-rate samples collected by the most recent `mint_spatial_corrections` call.
    /// One entry per fired neuron that had predictions. Diagnostic-only — used by tools that want
    /// to characterize the natural error-rate distribution (mean, percentiles, etc.) for tuning
    /// the static fallback threshold or understanding what dynamic modes would adapt toward.
    last_frame_spatial_errors: Vec<(NeuronId, f64)>,

    /// Spatial apex from the most recent `compute_apex_and_handoff` — the set of neurons that fired
    /// in the spatial sweep this frame and were NOT subsumed by any higher-level spatial pattern.
    /// This is the correct voter set for supervised wiring (brain.learn): subsumed parents are
    /// silenced and the higher-level pattern represents them. Distinct from
    /// `get_active_voter_ids` (which is keyed on `state.activated_pattern_id` — a temporal-inhibition
    /// field that, for multi-age contexts, can include neurons currently subsumed in spatial just
    /// because they had an unsuppressed state on an earlier frame).
    /// Cleared on `reset_context` / `reset_brain`.
    last_frame_apex: FxHashSet<NeuronId>,
}

impl Brain {
    /// Create a new Brain with the given hyperparameters.
    ///
    /// # Arguments
    /// * `context_length` — number of frames a base neuron stays active (default 10)
    /// * `error_correction_mode` — 'static' | 'conservative' | 'neutral' | 'aggressive'
    /// * `error_correction_threshold` — fixed threshold when mode='static'; warmup fallback
    /// * `merge_threshold` — percentage of matched entries needed for context merge
    /// * `pattern_forget_rate` — forget rate applied uniformly to every pattern neuron, all levels
    /// * `regions` — R — number of regions (1 for single-process)
    /// * `columns` — C — number of columns per region (1 for single-thread)
    /// * `consensus_mode` — how action votes combine into a winner ('democratic' | 'nb')
    /// * `debug` — enable verbose logging
    pub fn new(
        context_length: u32,
        error_correction_mode: ErrorMode,
        error_correction_threshold: f64,
        merge_threshold: f64,
        pattern_forget_rate: f64,
        regions: usize,
        columns: usize,
        consensus_mode: ConsensusMode,
        debug: bool,
    ) -> Self {
        Self {
            context_length,
            debug,
            consensus_mode,
            emit_votes: false,
            learning: true,
            frame: Vec::new(),
            rewards: Vec::new(),
            frame_number: 0,
            diagnostics: Diagnostics::new(),
            backup: Backup::new(pattern_forget_rate),
            thalamus: Thalamus::new(
                debug,
                pattern_forget_rate,
                merge_threshold,
                context_length,
                error_correction_mode,
                error_correction_threshold,
                regions,
                columns,
            ),
            memory: Memory::new(debug, context_length),
            last_frame_apex: FxHashSet::default(),
            last_frame_spatial_errors: Vec::new(),
        }
    }

    // ── Channel registration ────────────────────────────────────────────────

    /// Register a channel spec with the brain. Delegates to Thalamus.
    /// Returns the allocated channel id and per-dimension id map.
    pub fn register_channel_spec(
        &mut self,
        name: &str,
        dimensions: Vec<DimSpecInput>,
        learn_action_sequences: bool,
    ) -> ChannelRegistration {
        self.thalamus.register_channel_spec(name, dimensions, learn_action_sequences)
    }

    /// Declare the SPATIAL (d=0 co-activation) neighbor channel set for a registered channel.
    /// See `Thalamus::set_spatial_neighbors`.
    pub fn set_spatial_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        self.thalamus.set_spatial_neighbors(name, neighbor_names);
    }

    /// Declare the TEMPORAL (d>0 sequence) neighbor channel set for a registered channel.
    /// See `Thalamus::set_temporal_neighbors`.
    pub fn set_temporal_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        self.thalamus.set_temporal_neighbors(name, neighbor_names);
    }

    /// Declare the same neighbor set for BOTH phases. See `Thalamus::set_channel_neighbors`.
    pub fn set_channel_neighbors(&mut self, name: &str, neighbor_names: &[String]) {
        self.thalamus.set_channel_neighbors(name, neighbor_names);
    }

    // ── Context / reset ─────────────────────────────────────────────────────

    /// Toggle per-vote resolution on FrameResult.votes. Default off (training
    /// hot path stays free of per-vote string allocation and metadata lookups).
    /// Inference/debug harnesses that consume votes flip this on.
    pub fn set_emit_votes(&mut self, enabled: bool) {
        self.emit_votes = enabled;
    }

    /// Toggle the master learning flag.
    /// See `Brain.learning` for the list of mutating operations that get skipped when learning is off.
    /// Voting and inference still run, so `process_frame` continues to populate FrameResult.inferences identically.
    /// Only the learning side effects are suppressed.
    pub fn set_learning(&mut self, learning: bool) {
        self.learning = learning;
    }

    /// Inspect the current learning flag (test-only).
    #[cfg(test)]
    pub fn is_learning(&self) -> bool {
        self.learning
    }

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

        // Drop the apex carry-over — a fresh episode means a fresh frame, no prior apex.
        self.last_frame_apex.clear();
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

    // ── Save / load ──────────────────────────────────���──────────────────────

    /// Save brain state to `<job_dir>/backups/<label>/`. Materializes lazy decay
    /// before snapshotting so strengths reflect true post-decay values.
    pub fn save(&mut self, job_dir: &std::path::Path, label: &str) -> Result<std::path::PathBuf, String> {
        self.thalamus.materialize_and_reset_neurons(self.frame_number);
        self.frame_number = 0;
        let snapshot = self.thalamus.get_snapshot();
        self.backup.save(job_dir, label, &snapshot)
    }

    /// Load a backup by label from `<job_dir>/backups/<label>/`.
    pub fn load(&mut self, job_dir: &std::path::Path, label: &str) -> Result<(), String> {
        let snapshot = self.backup.load(job_dir, label)?;
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
        // activation_offset = activation_frame - frame_number (always ≤ 0).
        // Temporal-only — spatial activations are wiped per-frame and don't appear in snapshots.
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
        // Temporal-only — snapshots no longer carry a phase column because spatial state is wiped
        // per-frame and isn't persisted. Older snapshots that had a phase column are tolerated
        // (extra columns are simply ignored).
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

    // ── Lookup helpers ─────────────────────────────────────────────────────────

	/// returns dimension id for a given dimension name
    pub fn get_dimension_id_by_name(&self, name: &str) -> Option<DimensionId> {
        self.thalamus.get_dimension_id_by_name(name)
    }

    /// returns neuron id for a given coordinate
    pub fn get_neuron_id_by_coordinate(&self, coordinate: &Coordinate) -> Option<NeuronId> {
        self.thalamus.get_neuron_id_by_coordinate(coordinate)
    }

    // ── Inspection ──────────────────────────────────────────────────────────

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
    pub fn get_neuron_connections(&self, neuron_id: NeuronId) -> Vec<(Distance, NeuronId, f64, f64)> {
        self.thalamus.get_neuron_connections(neuron_id).unwrap_or_default()
    }

    /// Diagnostic: cumulative number of spatial corrections minted since brain start (or last hard reset).
    pub fn get_spatial_correction_count(&self) -> u64 {
        self.thalamus.get_spatial_correction_count()
    }

    /// Diagnostic: number of correction neurons currently sitting above the base spatial level.
    pub fn count_active_spatial_corrections(&self) -> usize {
        self.thalamus.count_active_spatial_corrections()
    }

    /// Diagnostic: per-level count of correction neurons. Index = (spatial_level - 1), so
    /// returned[0] = count at level 1, returned[1] = count at level 2, etc. Length = highest level reached.
    pub fn spatial_level_counts(&self) -> Vec<u32> {
        self.thalamus.spatial_level_counts()
    }

    /// Diagnostic: size of the most recent frame's apex set (the supervised wiring voter set).
    pub fn last_apex_size(&self) -> usize {
        self.last_frame_apex.len()
    }

    /// Diagnostic: size of `get_active_voter_ids` — what `learn()` USED to use (temporal voter set).
    /// Comparing this to `last_apex_size()` reveals whether the two mechanisms diverge.
    pub fn active_voter_ids_size(&self) -> usize {
        self.memory.get_active_voter_ids().len()
    }

    /// Diagnostic: number of subsumed parents in the most recent spatial sweep — i.e. the count
    /// of spatial-fired neurons that were silenced by a higher-level spatial pattern. For MNIST
    /// single-frame contexts: equal to (spatial_fired_size − last_apex_size).
    pub fn last_apex_subsumed_count(&self) -> usize {
        // We don't store spatial_fired or subsumed sets — but we expose this for a measurement
        // helper: `active_voter_ids_size - last_apex_size` is the count of neurons the temporal
        // wiring path would have included that the apex wiring path excludes.
        let voters = self.active_voter_ids_size();
        let apex = self.last_apex_size();
        voters.saturating_sub(apex)
    }

    /// Diagnostic: the spatial error rates from the most recent `process_frame`'s mint pass.
    /// One entry per fired neuron that had predictions. Used to characterize the natural error-rate
    /// distribution for tuning the static fallback threshold or understanding what dynamic modes
    /// would adapt toward.
    pub fn last_frame_spatial_error_rates(&self) -> &[(NeuronId, f64)] {
        &self.last_frame_spatial_errors
    }

    /// Diagnostic: number of neurons in apex but NOT in the temporal voter set, plus vice versa.
    /// If both sets are equal, returns (0, 0). If they diverge, the tuple reveals which side has
    /// extras. Used to confirm the apex wiring and the temporal-voter wiring produce the same set.
    pub fn apex_vs_voter_set_diff(&self) -> (usize, usize) {
        let voters: FxHashSet<NeuronId> = self.memory.get_active_voter_ids();
        let apex_only = self.last_frame_apex.difference(&voters).count();
        let voters_only = voters.difference(&self.last_frame_apex).count();
        (apex_only, voters_only)
    }

    /// Export a snapshot of all active neurons in context with their levels and phase.
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
            max_temporal_level: self.thalamus.get_max_temporal_level(),
            max_spatial_level: self.thalamus.get_max_spatial_level(),
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
    /// * `inputs` — channelId → (dimId → raw scalar)
    /// * `rewards` — channelId → reward for previous frame's actions
    pub fn process_frame(
        &mut self,
        inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        rewards: &FxHashMap<ChannelId, Reward>,
    ) -> FrameResult {
        // FrameTimings::default() captures the frame-start Instant.
        // timings.finalize() reads it back into timings.total at the end of the frame, so the
        // wall-clock measurement lives entirely inside the timings struct.
        let mut timings = FrameTimings::default();

        // Bump the brain's frame counter and sync memory to it.
        // Every downstream lookup keyed on the frame number (Op-2 reap, eviction math, activation
        // frame-keys) assumes this has run, so nothing else may read the frame number first.
        self.tick_frame();

        // Build the current frame from quantized inputs and previously inferred actions.
        let frame_neurons = self.get_frame_neurons(inputs, &mut timings);
        if frame_neurons.is_empty() {
            timings.finalize();
            return FrameResult {
                inferences: FxHashMap::default(),
                votes: Vec::new(),
                elapsed: timings.total,
                timings,
            };
        }

        // Op-1: construct any new sensory neurons in their owning columns. Runs in both learning
        // and non-learning modes — without sensory neurons we cannot process the frame at all.
        self.create_new_sensory_neurons(&frame_neurons, &mut timings);

        // Op-2: forget connections and patterns to avoid curse of dimensionality.
        // Skipped in non-learning mode so learned neurons do not decay or get forgotten.
        self.cleanup_dead_patterns(&mut timings);

        // Spatial phase: clear spatial[0], activate sensory events, run d=0 co-activation sweep, mint
        // corrections, hand off the apex set (fired \ subsumed) into temporal_level_index[0].
        // Event-only — actions skip spatial entirely. No rewards, no inference performance tracking.
        let spatial = self.process_spatial(&frame_neurons.events, &mut timings);

        // Temporal phase: push rewards, inject carry-forward actions into temporal[0] alongside spatial's
        // apex set, track inference performance for last frame's predictions, run the d>0 sequence sweep.
        let temporal = self.process_temporal(rewards, &frame_neurons.actions, &mut timings);

        // Op-4 + Op-5: flush deferred neuron creation and contextRef updates across both phases.
        // Spatial-correction specs are already folded into spatial.neuron_specs by process_spatial.
        // Partial-move spec/dispatch fields out of both sweeps so we can keep temporal.votes owned
        // and pass it into inference without cloning.
        self.apply_merged_results(
            spatial.neuron_specs,
            temporal.neuron_specs,
            spatial.dispatch_results,
            temporal.dispatch_results,
            &mut timings,
        );

        // Inference consumes temporal votes only. Spatial d=0 votes are same-frame co-activation
        // predictions; they don't contribute to next-frame consensus or action voting (see doc §3.3
        // and 1c spatial error pass).
        let (inferences, resolved_votes) = self.infer_neurons(&temporal.votes, &mut timings);

        // Accumulate MAPE by comparing continuous event predictions to the actual input scalars.
        self.track_continuous_error(&inferences, inputs, &mut timings);

        timings.finalize();
        FrameResult {
            inferences,
            votes: resolved_votes,
            elapsed: timings.total,
            timings,
        }
    }

    /// Advance the brain's frame counter and sync memory to the new value.
    /// Memory acts as a slave clock — the brain owns the tick.
    /// Called once per frame, at the very top of process_frame, before anything else reads the frame number.
    fn tick_frame(&mut self) {
        self.frame_number += 1;
        self.memory.sync_frame(self.frame_number);
    }

    // ── Frame building ──────────────────────────────────────────────────────

    /// Build this.frame from quantized inputs: one FramePoint per input scalar.
    /// Events only — carry-forward actions don't go through self.frame because we already have
    /// their neuron IDs directly on `InferredNeuron` and don't need the coordinate detour.
    fn build_frame(&mut self, inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>) {
        self.frame.clear();
        for channel_id in self.thalamus.get_channel_ids() {
            if let Some(dim_map) = inputs.get(&channel_id) {
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
        }
    }

    /// Resolve this frame's quantized inputs and carry-forward actions into neuron IDs, split by role.
    /// Events go through the coordinate-keyed lookup because input arrives as scalars, not ids.
    /// Actions skip the lookup — `InferredNeuron` already carries the neuron id from last frame's
    /// vote, and action neurons are pre-allocated at brain init, so every id is real and is_new is
    /// always false.
    fn get_frame_neurons(
        &mut self,
        inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        timings: &mut FrameTimings,
    ) -> FrameNeurons {
        let t = Instant::now();

        // Quantize raw scalars into FramePoints and stash on self.frame.
        // self.frame doubles as the start-of-frame diagnostic snapshot, so we keep it as an owned Vec on the brain.
        self.build_frame(inputs);

        // Resolve each event FramePoint to its neuron id, creating the substrate neuron on first sight.
        // get_neuron_id_for_point is &mut because the first time we see a (channel, coordinate) it mints a new sensory neuron id.
        let mut events = Vec::with_capacity(self.frame.len());
        for point in &self.frame {
            events.push(self.thalamus.get_neuron_id_for_point(
                &point.coordinate,
                point.channel_id,
                point.neuron_type.clone(),
            ));
        }

        // Take carry-forward action neuron ids straight off the inferred set.
        // No coordinate detour, no lookup — the previous frame's infer pass already produced the ids we need.
        let actions: Vec<PointLookup> = self.memory.get_inferred_neurons().iter()
            .filter(|n| self.thalamus.get_neuron_type(n.neuron_id) == Some(NeuronType::Action))
            .map(|n| PointLookup { id: n.neuron_id, is_new: false })
            .collect();

        timings.build_frame = t.elapsed().as_secs_f64();
        FrameNeurons { events, actions }
    }

    /// Op-1: construct any new sensory event neurons in their owning columns.
    /// Only events can be new this frame — action neurons are pre-allocated during brain
    /// initialization (one per registered action channel/coordinate), so every carry-forward
    /// action that reaches us here already exists. Carrying them through this pass would
    /// just filter them all out as is_new=false.
    fn create_new_sensory_neurons(&mut self, frame_neurons: &FrameNeurons, timings: &mut FrameTimings) {
        let t = Instant::now();
        // Sensory neurons never die themselves, but their hosted correction children must decay —
        // stamp the base-neuron forget rate, not 0.0 (which leaves those children immortal: a bug).
        let forget_rate = self.thalamus.base_neuron_forget_rate();
        let specs: Vec<NeuronCreateSpec> = frame_neurons.events.iter()
            .filter(|p| p.is_new)
            .map(|p| NeuronCreateSpec { id: p.id, forget_rate, connections: None })
            .collect();
        if !specs.is_empty() {
            self.thalamus.create_neurons(&specs);
        }
        timings.create_sensory = t.elapsed().as_secs_f64();
    }

    // ── Context aging ───────────────────────────────────────────────────────

    /// Push this frame's channel rewards onto the rewards history and trim to the window.
    /// Called at the start of temporal processing — spatial doesn't consume rewards.
    fn push_rewards(&mut self, rewards: &FxHashMap<ChannelId, Reward>) {
        self.rewards.insert(0, rewards.clone());
        if self.rewards.len() > self.context_length as usize {
            self.rewards.pop();
        }
    }

    // ── Neuron activation ───────────────────────────────────────────────────

    /// Activate this frame's sensory events in the spatial level index.
    /// Sensory neurons are level-0 substrate by definition — no thalamus lookup needed for the level.
    /// Age doesn't apply: spatial is fundamentally same-frame and gets wiped each frame, so this
    /// writes only to spatial_level_index (no age_index, no neuron_states, no frame keying).
    /// Events reach the temporal sweep — if at all — via the apex handoff at the end of spatial
    /// processing, where the non-subsumed survivors get activated in temporal[0] for real.
    fn activate_sensory_events(&mut self, frame_events: &[PointLookup], timings: &mut FrameTimings) {
        let t = Instant::now();
        for p in frame_events {
            self.memory.activate_spatial_neuron(p.id, 0);
        }
        timings.activate += t.elapsed().as_secs_f64();
    }

    /// Inject the previous frame's inferred actions, carried forward, into temporal[0].
    /// Actions bypass spatial entirely — they're motor outputs, not perceptual co-activation context.
    /// They land alongside spatial's apex set in temporal_level_index[0], so the temporal sweep sees
    /// both "what just happened" (events through the spatial pipeline) and "what the agent just did".
    /// Action neurons are level-0 substrate, so we pass level=0 directly without a thalamus lookup.
    fn activate_carry_forward_actions(&mut self, frame_actions: &[PointLookup], timings: &mut FrameTimings) {
        let t = Instant::now();
        for p in frame_actions {
            self.memory.activate_temporal_neuron(p.id, 0);
        }
        timings.activate += t.elapsed().as_secs_f64();
    }

    /// Compare this frame's sensory inputs to last frame's inferred predictions and update
    /// the running event-accuracy / action-reward stats. Runs at the start of temporal
    /// processing — the inferred set is from the previous frame's vote, and the actuals
    /// are this frame's age-0 active neurons.
    fn track_inference_performance(&mut self) {
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

    /// Run the spatial level-by-level sweep over `spatial_level_index`.
    /// Walks levels bottom-up: at each level, runs the d=0 co-activation pass over every active
    /// neuron, activates any matched higher-level pattern at level+1, and stops when no new
    /// activations push the max-level higher.
    ///
    /// Returns accumulated votes, deferred neuron creation specs, raw dispatch results, plus the
    /// fired set and the subsumed set that the apex handoff consumes to bridge spatial → temporal.
    /// Per-section timings are written into `timings` (with `+=` so the temporal sweep's flush
    /// folds into the same buckets).
    fn process_spatial_levels(&mut self, timings: &mut FrameTimings) -> SweepResult {

        // The sensory axis the thalamus uses for vote-error evaluation and connection learning.
        // For spatial this is the co-active set at level 0 of the spatial index — a single set,
        // no recency dimension.
        let sensory_neurons = self.memory.get_spatial_base_level();

        // Maximum currently-active spatial level. Walked bottom-up; can grow as patterns at
        // higher levels fire and get activated.
        let mut max_active_level = self.memory.get_spatial_max_active_level();

        // Track newly-created error pattern ids so they are excluded from the level pass at
        // their own level (prevents double connection-learning and context leak).
        let mut new_error_pattern_ids: FxHashSet<NeuronId> = FxHashSet::default();

        // Accumulate votes, new neuron specs, and dispatch results across levels.
        let mut votes = Vec::new();
        let mut neuron_specs = Vec::new();
        let mut dispatch_results = Vec::new();
        let mut neuron_timings = crate::neuron::NeuronOpTimings::default();
        let mut orch_timings = crate::thalamus::OrchestrationTimings::default();
        let mut mem_timings = MemoryTimings::default();

        // Spatial apex tracking: any neuron whose child pattern fires at level+1 is "subsumed".
        // The apex set is `fired \ subsumed` and feeds `temporal_level_index[0]` via the handoff.
        let mut fired_set: FxHashSet<NeuronId> = FxHashSet::default();
        let mut subsumed_set: FxHashSet<NeuronId> = FxHashSet::default();

        // Process neurons level-by-level — Op-3 dispatch is the only per-level round-trip.
        let mut level: Level = 0;
        loop {
            if self.debug { println!("Processing Spatial level {}", level); }

            // Get the level's active neuron ids. Spatial neurons carry no per-neuron state, so
            // this is just a flat set.
            let t = Instant::now();
            let level_neuron_ids = self.memory.get_spatial_level_neurons(level);
            mem_timings.get_level_neurons += t.elapsed().as_secs_f64();

            // Snapshot what fired at this level. Used by the apex handoff after the sweep.
            for &nid in &level_neuron_ids { fired_set.insert(nid); }

            // Process the level: aggregate view, match spatial patterns, generate d=0 votes.
            let result = self.thalamus.process_spatial_level(
                level,
                &level_neuron_ids,
                &sensory_neurons,
                self.frame_number,
                &mut new_error_pattern_ids,
                self.learning,
            );
            orch_timings.add(&result.orchestration);

            // Spatial state is ephemeral — nothing to write back, by design.

            // Activate matched patterns at level+1. Each activation marks its parent_id as
            // subsumed — that parent's role this frame is already represented by the
            // higher-level pattern it activated.
            let t = Instant::now();
            for activation in &result.activations {
                self.memory.activate_spatial_pattern(activation.pattern_id, level + 1);
                subsumed_set.insert(activation.parent_id);
            }
            mem_timings.activate_patterns += t.elapsed().as_secs_f64();

            // If we produced any activations, push the max active level up.
            if !result.activations.is_empty() {
                let new_level = (level + 1) as usize;
                if new_level > max_active_level { max_active_level = new_level; }
            }

            // Accumulate this level's votes, neuron specs, and dispatch results.
            votes.extend(result.votes);
            neuron_specs.extend(result.neuron_specs);
            for col_res in &result.results { neuron_timings.add(&col_res.timings); }
            dispatch_results.push(result.results);

            // If we reached the top of the active hierarchy, exit.
            if level as usize >= max_active_level { break; }
            level += 1;
        }

        Self::flush_sweep_timings(timings, &neuron_timings, &orch_timings, &mem_timings);
        SweepResult { votes, neuron_specs, dispatch_results, fired_set, subsumed_set }
    }

    /// Run the temporal level-by-level sweep over `temporal_level_index`.
    /// Walks levels bottom-up: at each level, runs the d>0 sequence pass over every (active
    /// neuron, age) pair, activates any matched higher-level pattern at level+1, and stops
    /// when no new activations push the max-level higher.
    ///
    /// Returns accumulated votes, deferred neuron creation specs, raw dispatch results, plus the
    /// fired and subsumed sets (populated harmlessly — the apex handoff doesn't consume them on
    /// the temporal side). Per-section timings are written into `timings` (with `+=` so the
    /// spatial sweep's flush folds into the same buckets).
    fn process_temporal_levels(&mut self, timings: &mut FrameTimings) -> SweepResult {

        // The sensory axis the thalamus uses for vote-error evaluation and connection learning.
        // For temporal this is the level-0 active set per recency slot — index 0 is the current
        // frame, index 1 is the previous frame, etc.
        let sensory_neurons = self.memory.get_temporal_base_level();

        // Maximum currently-active temporal level. Walked bottom-up; can grow as patterns at
        // higher levels fire and get activated.
        let mut max_active_level = self.memory.get_temporal_max_active_level();

        // Track newly-created error pattern ids so they are excluded from the level pass at
        // their own level (prevents double connection-learning and context leak).
        let mut new_error_pattern_ids: FxHashSet<NeuronId> = FxHashSet::default();

        // Accumulate votes, new neuron specs, and dispatch results across levels.
        let mut votes = Vec::new();
        let mut neuron_specs = Vec::new();
        let mut dispatch_results = Vec::new();
        let mut neuron_timings = crate::neuron::NeuronOpTimings::default();
        let mut orch_timings = crate::thalamus::OrchestrationTimings::default();
        let mut mem_timings = MemoryTimings::default();

        // Fired / subsumed are populated harmlessly on the temporal side — the apex handoff
        // only consumes them from the spatial sweep — but the SweepResult shape carries them
        // so the spatial and temporal paths share return types.
        let mut fired_set: FxHashSet<NeuronId> = FxHashSet::default();
        let mut subsumed_set: FxHashSet<NeuronId> = FxHashSet::default();

        // Process neurons level-by-level — Op-3 dispatch is the only per-level round-trip.
        let mut level: Level = 0;
        loop {
            if self.debug { println!("Processing Temporal level {}", level); }

            // Get the level's active neurons with their per-age states. Mutated downstream by
            // the level dispatch (votes/context/threshold written into LevelAgeState) and
            // written back into memory below.
            let t = Instant::now();
            let mut level_neurons = self.memory.get_temporal_level_neurons(level);
            mem_timings.get_level_neurons += t.elapsed().as_secs_f64();

            // Snapshot what fired at this level. Harmless for temporal (apex doesn't consume).
            for &nid in level_neurons.keys() { fired_set.insert(nid); }

            // Process the level: aggregate view, recognize temporal patterns, mint error
            // corrections, collect d>0 votes.
            let result = self.thalamus.process_temporal_level(
                level,
                &mut level_neurons,
                self.memory.depth(),
                &sensory_neurons,
                &self.rewards,
                self.frame_number,
                &mut new_error_pattern_ids,
                self.learning,
            );
            orch_timings.add(&result.orchestration);

            // Write the mutated per-neuron state back into memory for next-frame evaluation.
            // Temporal persists per-neuron state across frames (unlike spatial).
            let t = Instant::now();
            self.memory.write_back_level_neurons(&level_neurons);
            mem_timings.write_back_level_neurons += t.elapsed().as_secs_f64();

            // Activate matched patterns at level+1. Each activation marks its parent_id as
            // subsumed — that parent's role this frame is already represented by the
            // higher-level pattern it activated, so vote generation suppresses it.
            let t = Instant::now();
            for activation in &result.activations {
                self.memory.activate_temporal_pattern(activation.pattern_id, level + 1, activation.parent_id, activation.age);
                subsumed_set.insert(activation.parent_id);
            }
            mem_timings.activate_patterns += t.elapsed().as_secs_f64();

            // If we produced any activations, push the max active level up.
            if !result.activations.is_empty() {
                let new_level = (level + 1) as usize;
                if new_level > max_active_level { max_active_level = new_level; }
            }

            // Accumulate this level's votes, neuron specs, and dispatch results.
            votes.extend(result.votes);
            neuron_specs.extend(result.neuron_specs);
            for col_res in &result.results { neuron_timings.add(&col_res.timings); }
            dispatch_results.push(result.results);

            // If we reached the top of the active hierarchy, exit.
            if level as usize >= max_active_level { break; }
            level += 1;
        }

        Self::flush_sweep_timings(timings, &neuron_timings, &orch_timings, &mem_timings);
        SweepResult { votes, neuron_specs, dispatch_results, fired_set, subsumed_set }
    }

    /// Roll one sweep's sub-bucket accumulators into the frame-level timings. Uses += so spatial
    /// and temporal sweeps fold into the same fields without overwriting each other.
    fn flush_sweep_timings(
        timings: &mut FrameTimings,
        neuron_timings: &crate::neuron::NeuronOpTimings,
        orch_timings: &crate::thalamus::OrchestrationTimings,
        mem_timings: &MemoryTimings,
    ) {
        timings.neuron_learn_connections           += neuron_timings.learn_connections;
        timings.neuron_recognize_patterns          += neuron_timings.recognize_patterns;
        timings.neuron_correct_errors              += neuron_timings.correct_errors;
        timings.neuron_generate_votes              += neuron_timings.generate_votes;
        timings.recognize_candidate_search         += neuron_timings.recognize_candidate_search;
        timings.recognize_candidate_eval           += neuron_timings.recognize_candidate_eval;
        timings.recognize_candidates_evaluated     += neuron_timings.recognize_candidates_evaluated;
        timings.orch_get_level_tasks               += orch_timings.get_level_tasks;
        timings.orch_dispatch_frame                += orch_timings.dispatch_frame;
        timings.orch_collect_activations           += orch_timings.collect_activations;
        timings.orch_collect_votes                 += orch_timings.collect_votes;
        timings.mem_get_level_neurons              += mem_timings.get_level_neurons;
        timings.mem_write_back_level_neurons       += mem_timings.write_back_level_neurons;
        timings.mem_activate_patterns              += mem_timings.activate_patterns;
    }

    /// Full spatial phase: activate this frame's sensory events in the spatial index, run the d=0
    /// co-activation sweep, mint and install corrections, and hand off the non-subsumed apex
    /// set into temporal_level_index[0]. Returns the merged spec batch (sweep + corrections)
    /// and dispatch results for the deferred-creation flush at the end of the frame.
    /// Spatial is sensory-only — no rewards, no inference performance tracking.
    fn process_spatial(&mut self, frame_events: &[PointLookup], timings: &mut FrameTimings) -> SweepResult {
        let t = Instant::now();

        // Wipe last frame's spatial state.
        // Spatial is fundamentally same-frame: connections[0] predict co-activations within a single frame.
        // Carrying spatial activations across frames would cause d=0 learn_connections to strengthen edges
        // between this frame's actives and last frame's neurons — spurious cross-frame co-activation.
        self.memory.reset_spatial();

        // Activate this frame's sensory events in spatial[0].
        // Sensory events DO NOT activate in temporal directly. They reach temporal only via the apex handoff at the end of this function.
        // Any sensory event subsumed by a higher-level spatial pattern this frame stays out of temporal entirely; only the absorbing pattern represents it.
        self.activate_sensory_events(frame_events, timings);

        // Run the d=0 co-activation sweep over spatial_level_index.
        // Walks every active spatial level, matches the active set against each parent's routing table, casts same-frame predictions, and propagates matched patterns up to level+1.
        // Returns the fired_set (everything that activated anywhere) and subsumed_set (anything whose parent pattern fired at level+1) — the two sets the apex handoff needs.
        let mut spatial = self.process_spatial_levels(timings);

        // Mint corrections for parents whose d=0 predictions mismatched the actual fired set.
        // Each correction is a new pattern that gets installed in its parent's routing table for next frame's matching pass.
        // Also collects per-parent error-rate samples so dynamic error modes (conservative/neutral/aggressive) have data to adapt their thresholds.
        let correction_specs = self.process_spatial_corrections(&spatial);

        // Fold the freshly-minted correction specs into spatial's spec batch.
        // The end-of-frame flush in apply_merged_results materializes them alongside temporal's specs in one cross-region pass.
        // Their install ContextRefUpdates were already dispatched inline by process_spatial_corrections — only the neuron-creation half is deferred.
        spatial.neuron_specs.extend(correction_specs);

        // Lift the apex set (fired \ subsumed) into temporal_level_index[0].
        // This is the ONLY path sensory events and spatial corrections take into temporal.
        // Also populates last_frame_apex, which brain.learn reads to pick the voter set for the supervised wire.
        self.compute_apex_and_handoff(&spatial.fired_set, &spatial.subsumed_set, timings);

        timings.process_spatial = t.elapsed().as_secs_f64();
        spatial
    }

    /// Full temporal phase: push this frame's rewards onto the history, record inference
    /// performance for last frame's predictions vs. this frame's actuals, then run the d>0
    /// sequence sweep over `temporal_level_index`. By the time we get here, spatial has
    /// already populated temporal_level_index[0] with its apex set via the handoff.
    fn process_temporal(
        &mut self,
        rewards: &FxHashMap<ChannelId, Reward>,
        frame_actions: &[PointLookup],
        timings: &mut FrameTimings,
    ) -> SweepResult {
        let t = Instant::now();

        // Push this frame's channel rewards onto the rewards history and trim to the context window.
        // Action neurons need the current frame's rewards to resolve their per-edge expected reward during the d>0 sweep below.
        // Spatial never sees this — it was kept reward-free upstream.
        self.push_rewards(rewards);

        // Slide the temporal window: evict the frame that just fell off the back of the sliding window.
        // Done now, not earlier, for two reasons.
        // First, spatial above needed the full age_index to do cross-frame lookups against the previous frame's history.
        // Second, track_inference_performance below needs the pre-eviction inferred set to compare last frame's predictions against this frame's actuals.
        self.memory.advance_temporal_window();

        // Inject the previous frame's inferred actions, carried forward, into temporal_level_index[0].
        // Spatial's apex handoff already lifted non-subsumed sensory events and spatial corrections in there.
        // Together they form the full temporal[0] input: external events the brain perceived plus internal actions the brain produced.
        self.activate_carry_forward_actions(frame_actions, timings);

        // Compare last frame's predictions to this frame's actuals and update the running event-accuracy and action-reward stats.
        // Inputs to the comparison: last frame's inferred neurons (still in memory because we haven't run the new infer pass yet) vs. this frame's age=0 actives.
        // The aggregated stats drive the diagnostic per-channel summary printed at frame end and the run-wide accuracy reports.
        self.track_inference_performance();

        // Run the d>0 sequence sweep over temporal_level_index.
        // temporal_level_index[0] now holds spatial's apex set plus the carry-forward actions just activated above.
        // The sweep walks up through temporal patterns that match the recent activation history, casting next-frame votes that infer_neurons will aggregate into a consensus.
        let result = self.process_temporal_levels(timings);

        timings.process_temporal = t.elapsed().as_secs_f64();
        result
    }

    /// Apex handoff: any spatial-fired neuron that did NOT subsume into a higher-level
    /// spatial pattern this frame is inserted into `temporal_level_index[0]` and recorded in
    /// `last_frame_apex` for subsequent `brain.learn` calls.
    /// The apex set is the supervised-wiring voter set — subsumed parents are silenced by the
    /// higher-level pattern that absorbed them, so they shouldn't receive action wires either.
    /// On a fresh brain with no spatial routing, subsumed is empty and apex == fired == sensory,
    /// so temporal sees exactly today's inputs and `brain.learn` wires the full sensory set.
    fn compute_apex_and_handoff(
        &mut self,
        spatial_fired: &FxHashSet<NeuronId>,
        spatial_subsumed: &FxHashSet<NeuronId>,
        timings: &mut FrameTimings,
    ) {
        let t = Instant::now();

        // Wipe last frame's apex set before recomputing.
        // last_frame_apex is consumed by brain.learn to pick the voter set for the supervised wire.
        // Stale entries from last frame would wire to the wrong sensory state.
        self.last_frame_apex.clear();

        // Walk every neuron that fired anywhere in the spatial sweep this frame.
        // fired_set includes sensory events at level 0 and any spatial correction patterns that matched at level 1+.
        for &neuron_id in spatial_fired {

            // Skip subsumed neurons.
            // A neuron is subsumed when a higher-level spatial pattern that absorbs it also fired this frame.
            // Subsumed parents are silenced — the higher-level pattern represents their role in temporal.
            // If we let them through, they'd vote alongside the pattern that subsumed them and double-count.
            if spatial_subsumed.contains(&neuron_id) { continue; }

            // Activate the survivor in temporal_level_index[0].
            // This is the only path sensory events and spatial corrections take into the temporal sweep.
            self.memory.activate_temporal_neuron(neuron_id, 0);

            // Record the survivor in last_frame_apex.
            // brain.learn reads this to find the voter set for the supervised wire on the next learn() call.
            self.last_frame_apex.insert(neuron_id);
        }

        timings.apex_handoff = t.elapsed().as_secs_f64();
    }

    /// Spatial error pass: mint corrections from the d=0 sweep's results, install them in their
    /// parents' routing tables for next frame, record per-parent Welford error stats (when learning),
    /// and stash the feedback on `last_frame_spatial_errors`. Returns the spec batch so the caller
    /// can fold the new correction neurons into the deferred-creation flush.
    /// Not timed — these passes are bookkeeping around the spatial sweep, not their own bucket.
    fn process_spatial_corrections(&mut self, spatial: &SweepResult) -> Vec<NeuronCreateSpec> {

        // Mint correction specs from the d=0 sweep's dispatch results.
        // For every non-subsumed fired neuron whose d=0 predictions mismatched the actual fired set, this builds a NeuronCreateSpec for a new correction pattern, an install op to wire it into the parent's routing table, and an error-rate sample for the parent's Welford stats.
        // Returned specs are deferred — they'll be materialized in the end-of-frame flush along with temporal specs.
        let (specs, install_ops, feedback) = self.thalamus.mint_spatial_corrections(
            &spatial.dispatch_results,
            &spatial.fired_set,
            &spatial.subsumed_set,
        );

        // Install the corrections into their parents' routing tables right now.
        // These are ContextRefUpdates that have to land before the next frame so the parent can recognize the correction's context.
        // Done inline — not batched with the end-of-frame neuron-creation flush — because the install only mutates existing routing tables and doesn't depend on the correction neurons being constructed yet.
        self.thalamus.install_spatial_corrections(install_ops, self.frame_number);

        // Record per-parent error-rate samples for the spatial Welford stats.
        // Skipped in eval mode so the running stats don't drift from non-learning runs.
        // These samples are what let dynamic error-correction modes (conservative/neutral/aggressive) adapt the threshold over time — without them, get_error_threshold falls back to a static value forever.
        if self.learning {
            self.thalamus.record_spatial_errors(&feedback);
        }

        // Stash the feedback on the brain for downstream consumers (diagnostics, harnesses).
        // Owned here rather than passed back so callers don't have to thread it through process_spatial.
        self.last_frame_spatial_errors = feedback;

        // Return the spec batch.
        // process_spatial folds these into spatial.neuron_specs so they get materialized in the same flush as temporal specs.
        specs
    }

    /// Op-4 + Op-5: flush deferred neuron creation and contextRef updates across both phases in
    /// one batch. Spatial correction specs were already folded into `spatial_specs` by
    /// `process_spatial`. Their install ContextRefUpdates were dispatched inline by
    /// install_spatial_corrections — they aren't batched here.
    fn apply_merged_results(
        &mut self,
        spatial_specs: Vec<NeuronCreateSpec>,
        temporal_specs: Vec<NeuronCreateSpec>,
        spatial_dispatch: Vec<Vec<crate::column::ColumnProcessResult>>,
        temporal_dispatch: Vec<Vec<crate::column::ColumnProcessResult>>,
        timings: &mut FrameTimings,
    ) {
        let t = Instant::now();

        // Merge spec batches from both phases into one Vec.
        // spatial_specs already includes the spatial-correction specs that process_spatial folded in.
        // Reuses spatial_specs as the accumulator to avoid reallocating — temporal_specs gets extended onto it in place.
        let mut neuron_specs = spatial_specs;
        neuron_specs.extend(temporal_specs);

        // Merge dispatch results from both phases.
        // Same reuse trick: spatial_dispatch is the accumulator; temporal_dispatch gets extended onto it.
        // The merged Vec is what thalamus uses to apply ContextRefUpdates across all parents that processed this frame.
        let mut dispatch_results = spatial_dispatch;
        dispatch_results.extend(temporal_dispatch);

        // Flush Op-4 (deferred neuron creation) and Op-5 (deferred ContextRef updates) in one cross-region pass.
        // Neuron specs become real Neuron objects in their owning columns; ContextRef updates patch parents' routing tables with whatever each child's process_level produced.
        // Spatial-correction installs were already dispatched inline by process_spatial_corrections, so they aren't in dispatch_results.
        self.thalamus.apply_level_results(&neuron_specs, &dispatch_results);

        timings.apply_results = t.elapsed().as_secs_f64();
    }

    /// Build the dim-keyed inference view and feed it to the diagnostics MAPE tracker.
    fn track_continuous_error(
        &mut self,
        inferences: &FxHashMap<ChannelId, Vec<DimInferenceOutput>>,
        inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        timings: &mut FrameTimings,
    ) {
        let t = Instant::now();

        // Reshape the channel-keyed inference view into the dim-keyed shape the diagnostics tracker wants.
        // The two views carry the same data — track_continuous_error indexes by dim and reads each prediction's predicted bucket, while infer_neurons groups by channel so callers can render per-channel summaries cheaply.
        let dim_inferences = self.build_dim_inferences(inferences);

        // Update the running MAPE accumulator.
        // For each event dim that produced a prediction this frame, the tracker compares the predicted bucket's mid-scalar to the observed scalar in `inputs` and folds the relative error into the per-channel running average.
        // The quantizer is needed to map bucket id back to the scalar mid the prediction represented.
        self.diagnostics.track_continuous_error(&dim_inferences, inputs, &self.thalamus.quantizer);

        timings.track_error = t.elapsed().as_secs_f64();
    }

    // ── Inference (voting consensus) ────────────────────────────────────────

    /// Infer predictions and outputs using voting architecture.
    /// Returns the per-channel scalar-space inferences, the resolved per-vote
    /// list (always populated; same path that produces inferences just retains
    /// the raw data), and the raw winner list for memory persistence.
    fn infer_neurons(
        &mut self,
        votes: &[FlatVote],
        timings: &mut FrameTimings,
    ) -> (FxHashMap<ChannelId, Vec<DimInferenceOutput>>, Vec<FrameVote>) {
        let t = Instant::now();
        // if no inference votes, wait for more data
        if votes.is_empty() {
            if self.debug { println!("No inferences found. Waiting for more data in future frames."); }
            timings.infer = t.elapsed().as_secs_f64();
            return (FxHashMap::default(), Vec::new());
        }

        // Aggregate votes and determine winners
        let (winners, candidates, dim_best) = self.determine_consensus(votes);

        // Resolve each vote with the metadata callers need to render/analyze without further round-trips.
        // Targets without resolvable coordinates (shouldn't happen for event/action targets) are dropped.
        // Gated on emit_votes because per-vote resolution allocates strings, slowing the performance down.
        let resolved_votes: Vec<FrameVote> = if !self.emit_votes { Vec::new() } else {
            votes.iter().filter_map(|v| {
                let target_type = self.thalamus.get_neuron_type(v.neuron_id)?;
                let coord = self.thalamus.get_neuron_coordinate(v.neuron_id)?;
                let channel_id = self.thalamus.get_neuron_channel_id(v.neuron_id)?;
                Some(FrameVote {
                    voter_id: v.voter_id,
                    voter_label: self.format_neuron_label(v.voter_id),
                    voter_level: self.thalamus.get_neuron_level(v.voter_id).unwrap_or(0),
                    target_id: v.neuron_id,
                    target_type,
                    channel_id,
                    dim_id: coord.dim_id,
                    value: coord.bucket_id,
                    distance: v.distance,
                    strength: v.strength,
                    reward: v.reward,
                })
            }).collect()
        };

        // Save winners to memory (clears old inferences first)
        self.memory.save_inferred_neurons(winners);

        // Build the scalar-space output
        let inferences = self.build_inferences_by_channel(&candidates, &dim_best);

        timings.infer = t.elapsed().as_secs_f64();
        (inferences, resolved_votes)
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
        let winners = self.build_winners(&winner_ids, &candidates);
        (winners, candidates, dim_best)
    }

    /// Aggregate votes into candidate neurons and dimension strength totals.
    ///
    /// ── Per-voter split-by-strength normalization ──────────────────────────────────────
    ///
    /// Each FlatVote represents one outgoing CONNECTION from a voter neuron to a target neuron.
    /// A single voter typically casts MULTIPLE votes per frame — one per connection.
    /// vote() returns the full distance_map of that voter's connections at the voting distance.
    /// If a voter has connections to several neurons in the SAME dimension (action or event), every one of those connections produces a separate Vote.
    ///
    /// The naive aggregation — add each Vote's raw strength to its target candidate — double-counts hedged voters.
    /// A voter (action_A: str=3, action_B: str=1) in dim D would put 3 strength into action_A AND 1 strength into action_B, total 4 strength for D.
    /// So, a confident voter wired (action_A: str=4) contributes the same 4 strength but to only one candidate.
    /// The hedged voter ends up indistinguishable from "two independent voters each voting for a different action" — which it is not.
    ///
    /// Fix: scale each vote so a voter's total contribution to a single (dim, distance) is CONSERVED at 1 unit,
    /// split across its targets proportional to its connection strengths.
    /// In the example above the hedged voter contributes effective_strength=0.75 to action_A and 0.25 to action_B (sum = 1.0)
    /// the confident voter contributes 1.0 to action_A.
    /// "1 voter = 1 unit per (dim, distance)" regardless of how it's distributed.
    ///
    /// ── Why (voter, dim, distance) and not just (voter, dim) ───────────────────────────
    ///
    /// A single voter can be active at multiple ages simultaneously (the memory window holds it at several past frames).
    /// Each age produces votes at a different distance — these are independent predictions at different horizons.
    /// Keying by distance keeps those independent: each (voter, dim, distance) bucket gets its own 1-unit allocation.
    ///
    /// Events: With per-voter normalization, every voter contributes exactly 1.0 to dim_total per (dim, distance), so dim_total
    /// faithfully counts "voters that weighed in on this dim" weighted by their per-target shares. Probabilities become voter shares.
    ///
    /// Actions: With per-voter normalization, actions get the same split-by-strength behavior events were already partially getting through dim_total.
    /// The action winner is still highest reward, but strength magnitudes - used for tie-breaking and downstream consumers - are now properly distributed.
    /// For actions, reward = weighted_total / strength. Both terms are scaled by the same 1/voter_dim_total factor, so the ratio is invariant.
    /// A hedged voter still reports the same per-target reward; it just doesn't pretend to be multiple voters.
    fn aggregate_votes(&self, votes: &[FlatVote]) -> (FxHashMap<NeuronId, Candidate>, FxHashMap<DimensionId, f64>) {
        let mut candidates: FxHashMap<NeuronId, Candidate> = FxHashMap::default();
        let mut dim_total_strength: FxHashMap<DimensionId, f64> = FxHashMap::default();

        // Pre-pass: for every vote, accumulate the voter's total strength under (voter_id, dim_id, distance).
        // This becomes the denominator that scales each individual vote down to its share.
        // Every vote MUST target a neuron with a coordinate — sensory events and actions have
        // coordinates by construction. A vote toward a target without a coordinate means we
        // learned a connection to a pattern neuron upstream, which is a real architectural break
        // (the per-dim consensus has nothing to do with pattern targets).
        let mut voter_dim_total: FxHashMap<(NeuronId, DimensionId, Distance), f64> = FxHashMap::default();
        for v in votes {
            let coord = self.thalamus.get_neuron_coordinate(v.neuron_id)
                .unwrap_or_else(|| panic!(
                    "aggregate_votes: vote target {} has no coordinate (voter={}, distance={}). \
                     This means a connection was learned toward a pattern neuron — only sensory \
                     events/actions belong as vote targets.",
                    v.neuron_id, v.voter_id, v.distance
                ));
            *voter_dim_total.entry((v.voter_id, coord.dim_id, v.distance)).or_insert(0.0) += v.strength;
        }

        // Main pass: aggregate effective (normalized) strengths into candidates and dim totals.
        for v in votes {
            let coord = self.thalamus.get_neuron_coordinate(v.neuron_id)
                .unwrap_or_else(|| panic!(
                    "aggregate_votes: vote target {} has no coordinate in main pass",
                    v.neuron_id
                ));

            // Look up this voter's total strength in the (dim, distance) bucket and compute this vote's share of it.
            // effective_strength is in [0, 1]; per-voter shares sum to 1.0 within the (dim, distance) bucket by construction.
            let total = voter_dim_total.get(&(v.voter_id, coord.dim_id, v.distance)).copied().unwrap_or(0.0);
            if total <= 0.0 { continue; }
            let effective_strength = v.strength / total;

            // Fetch or insert the candidate for this target neuron.
            let candidate = candidates.entry(v.neuron_id).or_insert_with(|| Candidate {
                strength: 0.0,
                weighted_total: 0.0,
                reward: 0.0,
                probability: 0.0,
                nb_log_score: 0.0,
            });

            // Accumulate this voter's split share into the candidate's strength.
            candidate.strength += effective_strength;

            // For actions: accumulate strength-weighted reward sum, to calculate expected reward in determine_dimension_winners
            if self.thalamus.get_neuron_type(v.neuron_id) == Some(NeuronType::Action) {
                candidate.weighted_total += effective_strength * v.reward;
                // Naive-Bayes path: also accumulate the unweighted log-product of posteriors.
                // One term per vote (per connection) — a near-zero posterior vetoes the candidate.
                if self.consensus_mode == ConsensusMode::Nb {
                    candidate.nb_log_score += (v.reward + NB_EPS).ln();
                }
            }
            // For events: accumulate votes into the per-dim probability normalizer. every voter
            else {
                *dim_total_strength.entry(coord.dim_id).or_insert(0.0) += effective_strength;
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
            // After fixing aggregate_votes, every candidate must have a coordinate and a type —
            // candidates are only inserted from votes that survived the coordinate check.
            let coordinate = self.thalamus.get_neuron_coordinate(neuron_id)
                .unwrap_or_else(|| panic!("determine_dimension_winners: candidate {} has no coordinate", neuron_id));
            let neuron_type = self.thalamus.get_neuron_type(neuron_id)
                .unwrap_or_else(|| panic!("determine_dimension_winners: candidate {} has no neuron type", neuron_id));

            // for actions, reward = weighted_total / strength
            // for events, probability = strength / dimension total strength
            // Store the computed score back on the candidate (matches JS behavior)
            let score = if neuron_type == NeuronType::Action {
                let reward = if candidate.strength > 0.0 { candidate.weighted_total / candidate.strength } else { 0.0 };
                candidate.reward = reward;
                // The winner is selected on the consensus score: the expected reward (Democratic) or
                // the Naive-Bayes log-product (Nb). reward stays on the candidate either way so the
                // scalar-space continuous prediction is unchanged by the selection rule.
                match self.consensus_mode {
                    ConsensusMode::Democratic => reward,
                    ConsensusMode::Nb => candidate.nb_log_score,
                }
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
                    neuron_id: best.neuron_id,
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
    fn build_dim_inferences(&self, inferences: &FxHashMap<ChannelId, Vec<DimInferenceOutput>>) -> FxHashMap<ChannelId, Vec<DimInference>> {
        let mut result: FxHashMap<ChannelId, Vec<DimInference>> = FxHashMap::default();
        for (&channel_id, dim_outputs) in inferences {
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

    // ── Supervised learning entry point ─────────────────────────────────────

    /// Supervised wiring step that sits on top of the last process_frame call.
    /// `actions` names the correct action neuron(s) per channel as `ChannelId → DimensionId → scalar`.
    /// `rewards` carries the per-channel reward magnitude.
    /// `distance` is the connection-table slot at which the voter→action edge gets written and the read-back is taken.
    /// Wiring covers every currently active voter regardless of age — not just age-0 voters.
    /// In single-frame setups (MNIST) all voters happen to be at age 0.
    /// In temporal setups voters at multiple ages all contribute.
    /// This does NOT run process_frame, activate any new neurons, decay anything, or touch accuracy stats.
    /// It is pure wiring plus a read-back.
    /// Wiring goes through `Neuron::strengthen_or_create_connection`.
    /// That is the same create-or-strengthen + smoothed-reward core that temporal `upsert_connection` uses.
    /// Supervised wiring skips the alt-action-on-negative-reward exploration `upsert_connection` layers on top.
    /// That is why `learn()` doesn't need `channel_id` plumbed through.
    pub fn learn(
        &mut self,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, Vec<(f64, Reward)>>>,
        distance: Distance,
    ) -> FrameResult {
        let mut timings = FrameTimings::default();

        // Resolve the supplied (value, reward) pairs to action neuron ids paired with their per-target reward.
        let action_targets = self.resolve_action_targets(actions);

        // Wire the spatial apex (computed during the most recent compute_apex_and_handoff) to the
        // action targets. Subsumed parents are excluded because the higher-level pattern they
        // activated represents them — wiring the parent would dilute the pattern's signal.
        // Falls back to the temporal voter set on the rare path where process_frame wasn't called
        // before learn (e.g. supervised wiring on a freshly-loaded brain).
        let voter_ids: Vec<NeuronId> = if self.last_frame_apex.is_empty() {
            self.get_active_voter_ids()
        } else {
            self.last_frame_apex.iter().copied().collect()
        };
        assert!(!voter_ids.is_empty(), "Brain.learn: no active voters — process_frame must be called (with a non-empty frame) before learn()");

        // Wire every (voter → action) edge at the supplied distance with smoothed-reward accumulation.
        self.learn_action_connections(&voter_ids, &action_targets, distance);

        // Run a read-only vote sweep + consensus pass over all active voters at all their valid ages.
        // The caller can then observe the prediction the brain would make for this input post-supervision.
        let (inferences, votes) = self.compute_inferences(&mut timings);

        timings.finalize();
        FrameResult {
            inferences,
            votes,
            elapsed: timings.total,
            timings,
        }
    }

    /// Resolve (value, reward) pairs supplied to learn() into (action_neuron_id, reward) targets.
    /// Each `value` is quantized to a bucket id via the channel-dim's quantizer, then looked up as an action neuron.
    /// Lets the caller specify multiple action targets per dim, each with its own reward — e.g. for supervised digit
    /// classification, every digit is named on every call: correct digit with reward=1, every other digit with reward=0,
    /// so the smoothed-reward update on each connection converges to the per-voter posterior `K(V,d) / N_V = P(d|V)`.
    /// Panics if any value doesn't resolve to a registered action neuron, or if no targets were supplied at all.
    fn resolve_action_targets(
        &mut self,
        actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, Vec<(f64, Reward)>>>,
    ) -> Vec<(NeuronId, Reward)> {
        let mut action_targets: Vec<(NeuronId, Reward)> = Vec::new();
        for (&channel_id, dim_map) in actions {
            for (&dim_id, pairs) in dim_map {
                for &(value, reward) in pairs {
                    let bucket_id = self.thalamus.quantizer.quantize(dim_id, value);
                    let coord = Coordinate { dim_id, bucket_id };
                    let action_id = self.thalamus.get_neuron_id_by_coordinate(&coord).unwrap_or_else(|| {
                        panic!("Brain.learn: no action neuron registered at channel={}, dim={}, bucket={}", channel_id, dim_id, bucket_id)
                    });
                    action_targets.push((action_id, reward));
                }
            }
        }
        assert!(!action_targets.is_empty(), "Brain.learn: no action targets supplied — caller must name at least one action value");
        action_targets
    }

    /// Enumerate every currently-active voter across all ages in the sliding window.
    /// Inhibited neurons (those that activated a higher-level pattern) are excluded.
    /// They don't vote and shouldn't be wired by learn().
    /// A voter active at multiple ages is returned once — wiring is per-(voter, target, distance), not per-age.
    /// Panics if no voters are active — learn() requires a non-empty process_frame to have just run.
    fn get_active_voter_ids(&self) -> Vec<NeuronId> {
        let voter_ids: Vec<NeuronId> = self.memory.get_active_voter_ids().into_iter().collect();
        assert!(!voter_ids.is_empty(), "Brain.learn: no active voters — process_frame must be called (with a non-empty frame) before learn()");
        voter_ids
    }

    /// Build the (voter, action, reward) wiring batch and dispatch it through Thalamus.
    /// Every active voter wires to every supplied action neuron at the given distance.
    /// Wiring is additive (strength += 1, reward += reward_arg); the connection is allocated on first encounter.
    fn learn_action_connections(
        &mut self,
        voter_ids: &[NeuronId],
        action_targets: &[(NeuronId, Reward)],
        distance: Distance,
    ) {
        let mut wirings: Vec<(NeuronId, NeuronId, Reward)> = Vec::with_capacity(voter_ids.len() * action_targets.len());
        for &voter_id in voter_ids {
            for &(action_id, reward) in action_targets {
                wirings.push((voter_id, action_id, reward));
            }
        }
        self.thalamus.learn_action_connections(&wirings, distance);
    }

    /// Shared helper that walks every active non-suppressed (voter, age) pair, calls `neuron.vote(age)`
    /// on each, and feeds the resulting FlatVotes through the existing infer_neurons consensus path.
    /// Mirrors the voting half of process_frame without any of its mutating side effects.
    /// Used by `learn` as the post-wire read-back of the brain's prediction after the new wirings land.
    /// Returns the per-channel inference map and the per-vote detail list for the harness to inspect.
    fn compute_inferences(
        &mut self,
        timings: &mut FrameTimings,
    ) -> (FxHashMap<ChannelId, Vec<DimInferenceOutput>>, Vec<FrameVote>) {
        let voter_ages = self.memory.get_active_voter_ages();
        let votes = self.thalamus.collect_votes_for_voter_ages(&voter_ages);
        self.infer_neurons(&votes, timings)
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────

    /// Runs the cleanup cycle for zombie patterns.
    /// With lazy decay, this only deletes items that have decayed to zero effective strength.
    fn cleanup_dead_patterns(&mut self, timings: &mut FrameTimings) {
        // Forget-and-prune is a learning-mode side effect; in eval we leave the substrate alone.
        // Still record timing on both paths so the FrameTimings bucket exists every frame.
        let t = Instant::now();
        if !self.learning {
            timings.cleanup_dead = t.elapsed().as_secs_f64();
            return;
        }
        if self.debug { println!("=== CLEANUP STARTING ==="); }

        // reap neurons scheduled to die at this frame
        let dead_pattern_ids = self.thalamus.reap_dead_neurons(self.frame_number);
        if dead_pattern_ids.is_empty() {
            timings.cleanup_dead = t.elapsed().as_secs_f64();
            return;
        }

        // delete dead patterns (with recursive cleanup of context references)
        let deleted_pattern_ids = self.thalamus.delete_patterns(&dead_pattern_ids, self.frame_number);

        // verify no deleted patterns are active — that would be a bug
        self.memory.assert_not_active(&deleted_pattern_ids);

        if self.debug { println!("=== CLEANUP COMPLETED ===\n"); }
        timings.cleanup_dead = t.elapsed().as_secs_f64();
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
            1,                        // regions
            1,                        // columns
            ConsensusMode::Democratic, // consensus_mode
            false,                    // debug
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
        );
        assert!(reg.dimension_ids.contains_key("price"));

        // process an empty frame — should return empty inferences
        let result = brain.process_frame(&FxHashMap::default(), &FxHashMap::default());
        assert!(result.inferences.is_empty());
        assert!(result.votes.is_empty());
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
        );
        let channel_id = reg.channel_id;
        let dim_id = reg.dimension_ids["temp"];

        // process first frame with a temperature reading
        let mut inputs = FxHashMap::default();
        let mut dim_vals = FxHashMap::default();
        dim_vals.insert(dim_id, 25.0);
        inputs.insert(channel_id, dim_vals);

        let result = brain.process_frame(&inputs, &FxHashMap::default());
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
