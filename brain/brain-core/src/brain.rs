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
    Level, NeuronId, NeuronType, Reward,
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
        regions: usize,
        columns: usize,
        debug: bool,
    ) -> Self {
        Self {
            context_length,
            debug,
            frame: Vec::new(),
            rewards: Vec::new(),
            frame_number: 0,
            diagnostics: Diagnostics::new(),
            backup_store: Backup::new(pattern_forget_rate, context_length),
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

    // ── Save / load ──────────────────────────────────���──────────────────────

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

    // ── Lookup helpers ─────────────────────────────────────────────────────────

	/// returns dimension id for a given dimension name
    pub fn get_dimension_id_by_name(&self, name: &str) -> Option<DimensionId> {
        self.thalamus.get_dimension_id_by_name(name)
    }

    /// returns neuron id for a given coordinate
    pub fn get_neuron_id_by_coordinate(&self, coordinate: &Coordinate) -> Option<NeuronId> {
        self.thalamus.get_neuron_id_by_coordinate(coordinate)
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
    /// * `inputs` — channelId → (dimId → raw scalar)
    /// * `rewards` — channelId → reward for previous frame's actions
    pub fn process_frame(
        &mut self,
        inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
        rewards: &FxHashMap<ChannelId, Reward>,
    ) -> FrameResult {
        let frame_start = Instant::now();
        self.frame_number += 1;

        // build the current frame from quantized inputs and previously inferred actions
        let frame_neurons = self.get_frame_neurons(inputs);
        if frame_neurons.is_empty() {
            return FrameResult {
                inferences: FxHashMap::default(),
                elapsed: frame_start.elapsed().as_secs_f64(),
                vote_debug: None,
            };
        }

        // Op-1: construct any new sensory neurons in their owning columns
        self.create_new_sensory_neurons(&frame_neurons);

        // Op-2: forget connections and patterns to avoid curse of dimensionality
        self.cleanup_dead_patterns();

        // slide the temporal window: age active neurons and push the new rewards frame
        self.age_context(rewards);

        // activate new neurons in age=0, level=0 — inputs from the world
        let neuron_ids: Vec<NeuronId> = frame_neurons.iter().map(|p| p.id).collect();
        self.activate_neurons(&neuron_ids);

        // process neurons level-by-level — Op-3 dispatch is the only per-level round-trip
        let (votes, neuron_specs, dispatch_results) = self.process_levels();

        // Op-4 + Op-5: flush deferred neuron creation and contextRef updates in one batch
        self.thalamus.apply_level_results(&neuron_specs, &dispatch_results);

        // do inferences with age>0 neurons
        let (inferences_map, vote_debug, _winners) = self.infer_neurons(&votes);

        // accumulate MAPE by comparing continuous event predictions to the actual input scalars
        let dim_inferences = self.build_dim_inferences(&inferences_map);
        self.diagnostics.track_continuous_error(&dim_inferences, inputs, &self.thalamus.quantizer);

        FrameResult {
            inferences: inferences_map,
            elapsed: frame_start.elapsed().as_secs_f64(),
            vote_debug,
        }
    }

    // ── Frame building ──────────────────────────────────────────────────────

    /// Build this.frame from id-keyed inputs: quantize scalars to bucket IDs and push
    /// event coordinates, then append previously-inferred action coordinates from memory.
    fn build_frame(&mut self, inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>) {
        self.frame.clear();
        let inferred = self.memory.get_inferred_neurons();

        // convert InferredNeuron slice to the tuple format get_inferred_actions expects
        let frame_actions = self.thalamus.get_inferred_actions(inferred);

        // iterate every registered channel — a channel may contribute events,
        // carry-forward actions from the previous frame's inference, or both
        for channel_id in self.thalamus.get_channel_ids() {

            // quantize each dimension's scalar to a bucketId and push as event coordinate
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

            // include previously-inferred actions for this channel as sensory inputs
            if let Some(actions) = frame_actions.get(&channel_id) {
                for action in actions {
                    self.frame.push(FramePoint {
                        coordinate: action.coordinate.clone(),
                        channel_id,
                        neuron_type: NeuronType::Action,
                    });
                }
            }
        }
    }

    /// Returns neuron IDs for given frame points, creating new base neurons as needed.
    /// Builds the frame first from quantized inputs + inferred actions.
    fn get_frame_neurons(&mut self, inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>) -> Vec<PointLookup> {
        self.build_frame(inputs);

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
    fn create_new_sensory_neurons(&mut self, frame_neurons: &[PointLookup]) {
        let specs: Vec<NeuronCreateSpec> = frame_neurons.iter()
            .filter(|p| p.is_new)
            .map(|p| NeuronCreateSpec { id: p.id, forget_rate: 0.0, connections: None })
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
    fn process_levels(&mut self) -> (Vec<FlatVote>, Vec<NeuronCreateSpec>, Vec<Vec<crate::column::ColumnProcessResult>>) {
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

        // process neurons level-by-level
        let mut level: Level = 0;
        loop {
            if self.debug { println!("Processing level {} for pattern recognition", level); }

            // get level neurons (mutable borrow returned, will be written back)
            let mut level_neurons = self.memory.get_level_neurons(level);

            // process level: aggregate view, recognize patterns, create error corrections, collect votes
            let result = self.thalamus.process_level(
                level,
                &mut level_neurons,
                self.memory.depth(),
                &sensory_neurons,
                &self.rewards,
                self.frame_number,
                &mut new_error_pattern_ids,
            );

            // write back mutated level neurons to memory
            self.memory.write_back_level_neurons(&level_neurons);

            // activate matched patterns and newly-created error patterns at level+1
            for activation in &result.activations {
                self.memory.activate_pattern(activation.pattern_id, level + 1, activation.parent_id, activation.age);
            }

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
            dispatch_results.push(result.results);

            // if we reached the maximum level and no more patterns are recognized, exit
            if level as usize >= max_active_level { break; }
            level += 1;
        }

        (votes, neuron_specs, dispatch_results)
    }

    // ── Inference (voting consensus) ────────────────────────────────────────

    /// Infer predictions and outputs using voting architecture.
    /// Returns the per-channel scalar-space inferences, optional vote debug, and
    /// the raw winner list for memory persistence.
    fn infer_neurons(&mut self, votes: &[FlatVote]) -> (FxHashMap<ChannelId, Vec<DimInferenceOutput>>, Option<VoteDebug>, Vec<InferredNeuron>) {
        // if no inference votes, wait for more data
        if votes.is_empty() {
            if self.debug { println!("No inferences found. Waiting for more data in future frames."); }
            return (FxHashMap::default(), None, Vec::new());
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

        // Save inferences to memory (clears old inferences first)
        self.memory.save_inferred_neurons(inferences.clone());

        // Build the scalar-space output
        let inferences_map = self.build_inferences_by_channel(&candidates, &dim_best);

        (inferences_map, vote_debug, inferences)
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
        // Votes without a target coordinate (orphaned neurons that shouldn't exist in practice) contribute nothing here.
        let mut voter_dim_total: FxHashMap<(NeuronId, DimensionId, Distance), f64> = FxHashMap::default();
        for v in votes {
            if let Some(coord) = self.thalamus.get_neuron_coordinate(v.neuron_id) {
                *voter_dim_total.entry((v.voter_id, coord.dim_id, v.distance)).or_insert(0.0) += v.strength;
            }
        }

        // Main pass: aggregate effective (normalized) strengths into candidates and dim totals.
        for v in votes {

            // Skip votes without a target coordinate — they can't be aggregated into a dim
            // and downstream determine_dimension_winners would skip them anyway.
            let coord = match self.thalamus.get_neuron_coordinate(v.neuron_id) {
                Some(c) => c,
                None => continue,
            };

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
            });

            // Accumulate this voter's split share into the candidate's strength.
            candidate.strength += effective_strength;

            // For actions: accumulate strength-weighted reward sum, to calculate expected reward in determine_dimension_winners
            if self.thalamus.get_neuron_type(v.neuron_id) == Some(NeuronType::Action) {
                candidate.weighted_total += effective_strength * v.reward;
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
            1,                        // regions
            1,                        // columns
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
