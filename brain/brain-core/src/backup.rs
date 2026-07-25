/// Backup / restore for brain state.
///
/// Snapshots are written as a folder of CSVs (no header rows) that mirror the
/// MySQL schema, so the same files can be bulk-loaded with `LOAD DATA INFILE`
/// by the apps/db/import job.
///
/// Layout: `<job_dir>/backups/<YYYY-MM-DD_HH-mm-ss>/`
///   - `channels.csv`           id,name
///   - `dimensions.csv`         id,name
///   - `neurons.csv`            id,temporal_level,spatial_level
///   - `base_neurons.csv`       neuron_id,channel_id,type,dimension_id,val
///   - `connections.csv`        from_neuron_id,to_neuron_id,distance,strength,reward
///   - `patterns.csv`           pattern_neuron_id,parent_neuron_id,strength
///   - `contexts.csv`           pattern_neuron_id,context_neuron_id,context_age,strength — each routing
///                              entry's stored context. A child's rows are keyed by the child's pattern
///                              id (age 0 spatial, >=1 temporal); a neuron's NORMAL is keyed by the
///                              neuron's own id with context_age == u32::MAX (NORMAL_CONTEXT_AGE).
///   - `spatial_history.csv`   neuron_id,frame,served_child,observed — the history window: one row per
///                              remembered frame, `served_child` empty when the normal served, `observed`
///                              a space-separated neighbor id list. This is the evidence the one test reads.
///   - `neuron_error_stats.csv` neuron_id,age,n,mean,m2 — temporal per-age Welford error stats.

use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::neuron::{
    SerializedChild, SerializedConnection, SerializedContextRef,
    SerializedErrorStats, SerializedHistoryRecord, SerializedNeuron,
};

/// Sentinel `context_age` marking a `contexts.csv` row as a neuron's **normal** (its stored context
/// configuration) rather than a child's context. Real spatial contexts use age 0 and temporal ones a
/// small distance, so `u32::MAX` can never collide. The row's pattern-id column holds the owning
/// neuron's id, and the row routes straight to that neuron's normal — no parent/child lookup.
const NORMAL_CONTEXT_AGE: Distance = u32::MAX;
use crate::thalamus::{BaseNeuron, Snapshot, SnapshotNeuronEntry};
use crate::types::{
    ChannelId, ContextEntry, Coordinate, DimensionId, Distance,
    FrameNumber, Level, NeuronId, NeuronType,
};

/// Hard cap on retained backup folders. The 11th save evicts the oldest by
/// folder-name sort (timestamps sort lexicographically thanks to the format).
const MAX_BACKUPS: usize = 10;

pub struct Backup {
    /// Brain-wide forget rate — applied uniformly to every pattern neuron on
    /// load since the snapshot doesn't carry per-neuron rates.
    pattern_forget_rate: f64,
}

impl Backup {
    /// Create a new Backup carrying the brain-wide forget rate assigned to every pattern neuron on load.
    pub fn new(pattern_forget_rate: f64) -> Self {
        Self { pattern_forget_rate }
    }

    // ── Save ────────────────────────────────────────────────────────────────

    /// Save a snapshot under `<job_dir>/backups/<label>/`. When label is "latest",
    /// a timestamped subfolder is created and old ones are pruned. Any other label
    /// writes directly to that named folder (overwriting prior content).
    pub fn save(&self, job_dir: &Path, label: &str, snapshot: &Snapshot) -> Result<PathBuf, String> {
        let backups_dir = job_dir.join("backups");
        fs::create_dir_all(&backups_dir)
            .map_err(|e| format!("Failed to create backups dir: {}", e))?;

        let folder = if label == "latest" {
            let timestamp = format_timestamp();
            backups_dir.join(&timestamp)
        } else {
            backups_dir.join(label)
        };

        // For named labels, wipe the folder so stale CSVs from a prior save
        // don't bleed through if the new snapshot is smaller.
        if label != "latest" && folder.exists() {
            fs::remove_dir_all(&folder)
                .map_err(|e| format!("Failed to clear labeled backup folder: {}", e))?;
        }
        fs::create_dir_all(&folder)
            .map_err(|e| format!("Failed to create backup folder: {}", e))?;

        // Write each table in dependency order
        self.write_channels(&folder, snapshot)?;
        self.write_dimensions(&folder, snapshot)?;
        self.write_neurons(&folder, snapshot)?;
        self.write_base_neurons(&folder, snapshot)?;
        self.write_connections(&folder, snapshot)?;
        self.write_patterns(&folder, snapshot)?;
        self.write_contexts(&folder, snapshot)?;
        self.write_spatial_history(&folder, snapshot)?;
        self.write_neuron_error_stats(&folder, snapshot)?;

        println!("💾 Backup saved: {} ({} neurons)", folder.display(), snapshot.neurons.len());

        // Pruning only applies to timestamped "latest" saves
        if label == "latest" {
            self.prune_old_backups(&backups_dir);
        }

        Ok(folder)
    }

    // ── Load ────────────────────────────────────────────────────────────────

    /// Load a backup by label. When label is "latest", loads the newest
    /// timestamped folder. Any other label loads from that named folder directly.
    pub fn load(&self, job_dir: &Path, label: &str) -> Result<Snapshot, String> {
        let backups_dir = job_dir.join("backups");

        let folder = if label == "latest" {
            self.find_latest_backup(&backups_dir)
                .ok_or_else(|| format!("--load-brain latest requested but no backups found in {}", backups_dir.display()))?
        } else {
            let named = backups_dir.join(label);
            if !named.exists() {
                return Err(format!("--load-brain {} requested but folder does not exist: {}", label, named.display()));
            }
            named
        };

        println!("📂 Loading backup: {}", folder.display());

        // Channels: name→id
        let mut channel_name_to_id: FxHashMap<String, ChannelId> = FxHashMap::default();
        for row in read_csv(&folder.join("channels.csv"))? {
            if row.len() < 2 { continue; }
            let id: ChannelId = row[0].parse().map_err(|e| format!("Bad channel id: {}", e))?;
            channel_name_to_id.insert(row[1].clone(), id);
        }

        // Dimensions: name→id
        let mut dimension_name_to_id: FxHashMap<String, DimensionId> = FxHashMap::default();
        for row in read_csv(&folder.join("dimensions.csv"))? {
            if row.len() < 2 { continue; }
            let id: DimensionId = row[0].parse().map_err(|e| format!("Bad dim id: {}", e))?;
            dimension_name_to_id.insert(row[1].clone(), id);
        }

        // Neurons: build neuron shells keyed by id. The optional 3rd column carries spatial_level;
        // pre-spatial snapshots omit it, so it defaults to 0 (sensory / base of the spatial hierarchy).
        let mut neurons: FxHashMap<NeuronId, SerializedNeuron> = FxHashMap::default();
        let mut temporal_levels: FxHashMap<NeuronId, Level> = FxHashMap::default();
        let mut spatial_levels: FxHashMap<NeuronId, Level> = FxHashMap::default();
        for row in read_csv(&folder.join("neurons.csv"))? {
            if row.len() < 2 { continue; }
            let id: NeuronId = row[0].parse().map_err(|e| format!("Bad neuron id: {}", e))?;
            let temporal_level: Level = row[1].parse().map_err(|e| format!("Bad temporal level: {}", e))?;
            let spatial_level: Level = if row.len() >= 3 {
                row[2].parse().map_err(|e| format!("Bad spatial level: {}", e))?
            } else {
                0
            };
            neurons.insert(id, SerializedNeuron {
                id,
                pattern_forget_rate: self.pattern_forget_rate,
                connections: Vec::new(),
                children: Vec::new(),
                context_refs: Vec::new(),
                error_stats: Vec::new(),
                normal_context: Vec::new(),
                spatial_history: Vec::new(),
            });
            temporal_levels.insert(id, temporal_level);
            spatial_levels.insert(id, spatial_level);
        }

        // Base neurons: level-0 sensory metadata
        let mut base_neurons: FxHashMap<NeuronId, BaseNeuron> = FxHashMap::default();
        let base_file = folder.join("base_neurons.csv");
        if base_file.exists() {
            for row in read_csv(&base_file)? {
                if row.len() < 5 { continue; }
                let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad base neuron id: {}", e))?;
                let channel_id: ChannelId = row[1].parse().map_err(|e| format!("Bad channel id: {}", e))?;
                let neuron_type = match row[2].as_str() {
                    "action" => NeuronType::Action,
                    _ => NeuronType::Event,
                };
                let dim_id: DimensionId = row[3].parse().map_err(|e| format!("Bad dim id: {}", e))?;
                let bucket_id: i32 = row[4].parse().map_err(|e| format!("Bad bucket id: {}", e))?;
                base_neurons.insert(neuron_id, BaseNeuron {
                    channel_id,
                    neuron_type,
                    coordinate: Coordinate { dim_id, bucket_id },
                });
            }
        }

        // Connections: each row is a directed (from→to, distance) link
        let conn_file = folder.join("connections.csv");
        if conn_file.exists() {
            for row in read_csv(&conn_file)? {
                if row.len() < 5 { continue; }
                let from_id: NeuronId = row[0].parse().map_err(|e| format!("Bad conn from: {}", e))?;
                let to_id: NeuronId = row[1].parse().map_err(|e| format!("Bad conn to: {}", e))?;
                let distance: Distance = row[2].parse().map_err(|e| format!("Bad distance: {}", e))?;
                let strength: f64 = row[3].parse().map_err(|e| format!("Bad strength: {}", e))?;
                let reward: f64 = row[4].parse().map_err(|e| format!("Bad reward: {}", e))?;

                if !neurons.contains_key(&from_id) {
                    return Err(format!("Connection source neuron not found: {}", from_id));
                }
                if !neurons.contains_key(&to_id) {
                    return Err(format!("Connection target neuron not found: {}", to_id));
                }
                neurons.get_mut(&from_id).unwrap().connections.push(SerializedConnection {
                    distance, to_neuron_id: to_id, strength, reward,
                });
            }
        }

        // Patterns: register each pattern as a child of its parent neuron
        let mut neuron_parents: FxHashMap<NeuronId, NeuronId> = FxHashMap::default();
        // Tracks children for context attachment below
        let mut children_map: FxHashMap<NeuronId, FxHashMap<NeuronId, usize>> = FxHashMap::default();
        let patterns_file = folder.join("patterns.csv");
        if patterns_file.exists() {
            for row in read_csv(&patterns_file)? {
                if row.len() < 3 { continue; }
                let pattern_id: NeuronId = row[0].parse().map_err(|e| format!("Bad pattern id: {}", e))?;
                let parent_id: NeuronId = row[1].parse().map_err(|e| format!("Bad parent id: {}", e))?;
                let strength: f64 = row[2].parse().map_err(|e| format!("Bad strength: {}", e))?;
                // Extra trailing columns from older backups (a `fires` column, or the retired
                // evidence/price payment state) are ignored — the add pass prices a child before it is
                // created, so a routing-table child carries no payment state to restore.

                neuron_parents.insert(pattern_id, parent_id);

                let parent = neurons.get_mut(&parent_id)
                    .ok_or_else(|| format!("Pattern parent not found: {}", parent_id))?;
                // A spatial pattern neuron (spatial_level >= 1) restores as a spatial child of its parent;
                // a temporal pattern (spatial_level 0, temporal level >= 1) as a temporal child.
                let spatial = spatial_levels.get(&pattern_id).copied().unwrap_or(0) != 0;
                let child_idx = parent.children.len();
                parent.children.push(SerializedChild {
                    pattern_id,
                    spatial,
                    activation_strength: strength,
                    last_activation_frame: 0,
                    context: Vec::new(),
                });

                children_map.entry(parent_id)
                    .or_insert_with(FxHashMap::default)
                    .insert(pattern_id, child_idx);
            }
        }

        // Contexts: for each pattern, restore the context neurons and ages that define
        // when it should activate. Also build contextRefs on the context neurons.
        let contexts_file = folder.join("contexts.csv");
        if contexts_file.exists() {
            for row in read_csv(&contexts_file)? {
                if row.len() < 4 { continue; }
                let pattern_id: NeuronId = row[0].parse().map_err(|e| format!("Bad ctx pattern id: {}", e))?;
                let context_id: NeuronId = row[1].parse().map_err(|e| format!("Bad ctx neuron id: {}", e))?;
                let context_age: Distance = row[2].parse().map_err(|e| format!("Bad ctx age: {}", e))?;
                let strength: f64 = row[3].parse().map_err(|e| format!("Bad ctx strength: {}", e))?;

                // A normal row (sentinel age): the neuron's own stored context configuration, keyed by
                // its own id. Route straight to its normal — no parent/child lookup, and the normal is
                // never a delete-cascade target, so no contextRef is built for it.
                if context_age == NORMAL_CONTEXT_AGE {
                    let neuron = neurons.get_mut(&pattern_id)
                        .ok_or_else(|| format!("normal-context neuron not found: {}", pattern_id))?;
                    neuron.normal_context.push(context_id);
                    continue;
                }

                // Find the child entry on the parent and push the context entry
                let parent_id = neuron_parents.get(&pattern_id)
                    .ok_or_else(|| format!("contexts parent not found for pattern {}", pattern_id))?;
                let parent_children = children_map.get(parent_id)
                    .ok_or_else(|| format!("contexts parent children not found for pattern {}", pattern_id))?;
                let &child_idx = parent_children.get(&pattern_id)
                    .ok_or_else(|| format!("contexts child entry not found for pattern {}", pattern_id))?;

                let parent = neurons.get_mut(parent_id)
                    .ok_or_else(|| format!("contexts parent neuron not found: {}", parent_id))?;
                parent.children[child_idx].context.push(ContextEntry {
                    neuron_id: context_id,
                    distance: context_age,
                    strength,
                });

                // Build contextRef on the context neuron. Spatial contexts carry age 0; temporal
                // carry their real age (>= 1). Key the ref by (parent, spatial) so a neuron that is
                // both a spatial and a temporal context of the same parent keeps two distinct refs.
                let spatial = context_age == 0;
                let ctx_neuron = neurons.get_mut(&context_id)
                    .ok_or_else(|| format!("contexts context neuron not found: {}", context_id))?;
                let existing_ref = ctx_neuron.context_refs.iter_mut()
                    .find(|r| r.parent_id == *parent_id && r.spatial == spatial);
                match existing_ref {
                    Some(ref_entry) => {
                        if !ref_entry.distances.contains(&context_age) {
                            ref_entry.distances.push(context_age);
                        }
                    }
                    None => {
                        ctx_neuron.context_refs.push(SerializedContextRef {
                            parent_id: *parent_id,
                            spatial,
                            distances: vec![context_age],
                        });
                    }
                }
            }
        }

        // The history window. Rows are written in order per neuron, so appending as they are read
        // preserves it. Absent file = an older backup with no history — the neuron restores empty,
        // exactly as before this table existed.
        let history_file = folder.join("spatial_history.csv");
        if history_file.exists() {
            for row in read_csv(&history_file)? {
                if row.len() < 4 { continue; }
                let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad history neuron id: {}", e))?;
                let frame: FrameNumber = row[1].parse().map_err(|e| format!("Bad history frame: {}", e))?;
                let served_child: Option<NeuronId> = if row[2].is_empty() {
                    None
                } else {
                    Some(row[2].parse().map_err(|e| format!("Bad history server: {}", e))?)
                };
                let mut observed = Vec::new();
                for id in row[3].split_whitespace() {
                    observed.push(id.parse::<NeuronId>().map_err(|e| format!("Bad history observed id: {}", e))?);
                }
                let neuron = neurons.get_mut(&neuron_id)
                    .ok_or_else(|| format!("spatial_history neuron not found: {}", neuron_id))?;
                neuron.spatial_history.push(SerializedHistoryRecord { frame, served_child, observed });
            }
        }

        let error_stats_file = folder.join("neuron_error_stats.csv");
        if error_stats_file.exists() {
            for row in read_csv(&error_stats_file)? {
                if row.len() < 5 { continue; }
                let neuron_id: NeuronId = row[0].parse().map_err(|e| format!("Bad err neuron id: {}", e))?;
                let age: Distance = row[1].parse().map_err(|e| format!("Bad err age: {}", e))?;
                let n: u64 = row[2].parse().map_err(|e| format!("Bad err n: {}", e))?;
                let mean: f64 = row[3].parse().map_err(|e| format!("Bad err mean: {}", e))?;
                let m2: f64 = row[4].parse().map_err(|e| format!("Bad err m2: {}", e))?;

                let neuron = neurons.get_mut(&neuron_id)
                    .ok_or_else(|| format!("neuron_error_stats neuron not found: {}", neuron_id))?;
                neuron.error_stats.push(SerializedErrorStats { age, n, mean, m2 });
            }
        }

        // Assemble the snapshot shape Thalamus expects
        let mut neuron_entries = Vec::with_capacity(neurons.len());
        for (neuron_id, neuron) in neurons {
            let temporal_level = temporal_levels.get(&neuron_id).copied().unwrap_or(0);
            let spatial_level = spatial_levels.get(&neuron_id).copied().unwrap_or(0);
            // Only temporal_level 0 AND spatial_level 0 carry a stored base coordinate; spatial
            // patterns also sit at temporal_level 0 but their coordinate is rederived in Thalamus::restore_snapshot.
            let base_neuron = if temporal_level == 0 && spatial_level == 0 {
                base_neurons.get(&neuron_id).cloned()
            } else {
                None
            };
            let parent_id = neuron_parents.get(&neuron_id).copied();
            neuron_entries.push(SnapshotNeuronEntry {
                neuron,
                temporal_level,
                spatial_level,
                base_neuron,
                parent_id,
            });
        }

        println!("   Loaded {} neurons", neuron_entries.len());

        Ok(Snapshot {
            neurons: neuron_entries,
            channel_name_to_id,
            dimension_name_to_id,
        })
    }

    // ── Writers ─────────────────────────────────────────────────────────────

    /// Write the channel id↔name table. Sorted by id for stable output.
    fn write_channels(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut rows: Vec<Vec<String>> = snapshot.channel_name_to_id.iter()
            .map(|(name, &id)| vec![id.to_string(), name.clone()])
            .collect();
        rows.sort_by_key(|r| r[0].parse::<u32>().unwrap_or(0));
        write_csv(&folder.join("channels.csv"), &rows)
    }

    /// Write the dimension id↔name table. Sorted by id.
    fn write_dimensions(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut rows: Vec<Vec<String>> = snapshot.dimension_name_to_id.iter()
            .map(|(name, &id)| vec![id.to_string(), name.clone()])
            .collect();
        rows.sort_by_key(|r| r[0].parse::<u32>().unwrap_or(0));
        write_csv(&folder.join("dimensions.csv"), &rows)
    }

    /// Write the neuron id+temporal_level+spatial_level table. Sorted by id.
    fn write_neurons(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut rows: Vec<Vec<String>> = snapshot.neurons.iter()
            .map(|entry| vec![
                entry.neuron.id.to_string(),
                entry.temporal_level.to_string(),
                entry.spatial_level.to_string(),
            ])
            .collect();
        rows.sort_by_key(|r| r[0].parse::<u64>().unwrap_or(0));
        write_csv(&folder.join("neurons.csv"), &rows)
    }

    /// Write the temporal_level-0 sensory metadata table.
    fn write_base_neurons(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut rows: Vec<Vec<String>> = Vec::new();
        for entry in &snapshot.neurons {
            if entry.temporal_level != 0 { continue; }
            if let Some(base) = &entry.base_neuron {
                let type_str = match base.neuron_type {
                    NeuronType::Event => "event",
                    NeuronType::Action => "action",
                };
                rows.push(vec![
                    entry.neuron.id.to_string(),
                    base.channel_id.to_string(),
                    type_str.to_string(),
                    base.coordinate.dim_id.to_string(),
                    base.coordinate.bucket_id.to_string(),
                ]);
            }
        }
        rows.sort_by_key(|r| r[0].parse::<u64>().unwrap_or(0));
        write_csv(&folder.join("base_neurons.csv"), &rows)
    }

    /// Write the directed-connection table. Streamed row-by-row — this is the largest table
    /// (hundreds of MB on a big brain), so it must not be materialized in memory first.
    fn write_connections(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut w = open_csv(&folder.join("connections.csv"))?;
        for entry in &snapshot.neurons {
            for conn in &entry.neuron.connections {
                write_row(&mut w, &[
                    entry.neuron.id.to_string(),
                    conn.to_neuron_id.to_string(),
                    conn.distance.to_string(),
                    conn.strength.to_string(),
                    conn.reward.to_string(),
                ])?;
            }
        }
        w.flush().map_err(|e| format!("Failed to flush connections.csv: {}", e))
    }

    /// Write the pattern→parent table with activation strengths.
    fn write_patterns(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        // Build an id→neuron lookup so we can find activation strength on the parent side
        let neuron_map: FxHashMap<NeuronId, &SerializedNeuron> = snapshot.neurons.iter()
            .map(|e| (e.neuron.id, &e.neuron))
            .collect();

        let mut w = open_csv(&folder.join("patterns.csv"))?;
        for entry in &snapshot.neurons {
            // Every pattern neuron has a parent; true sensory/action neurons do not. Filter on parent
            // presence rather than temporal level, because spatial pattern neurons also sit at level 0
            // (their depth lives in spatial_level) and must still be recorded with their parent here.
            let parent_id = match entry.parent_id {
                Some(p) => p,
                None => continue,
            };
            let mut strength = 0.0;
            if let Some(parent) = neuron_map.get(&parent_id) {
                if let Some(child) = parent.children.iter().find(|c| c.pattern_id == entry.neuron.id) {
                    strength = child.activation_strength;
                }
            }
            write_row(&mut w, &[
                entry.neuron.id.to_string(),
                parent_id.to_string(),
                strength.to_string(),
            ])?;
        }
        w.flush().map_err(|e| format!("Failed to flush patterns.csv: {}", e))
    }

    /// Write the pattern context-entry table. Streamed row-by-row — on a deep brain this is the
    /// single largest table (can exceed the connection table), so it must not be buffered whole.
    fn write_contexts(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut w = open_csv(&folder.join("contexts.csv"))?;
        for entry in &snapshot.neurons {
            // Each child's stored context configuration, keyed by the child's pattern id.
            for child in &entry.neuron.children {
                for ctx in &child.context {
                    write_row(&mut w, &[
                        child.pattern_id.to_string(),
                        ctx.neuron_id.to_string(),
                        ctx.distance.to_string(),
                        ctx.strength.to_string(),
                    ])?;
                }
            }
            // The neuron's normal — a routing entry like a child, but with no pattern neuron, so it is
            // keyed by the owning neuron's own id and marked with the NORMAL_CONTEXT_AGE sentinel.
            for &ctx_id in &entry.neuron.normal_context {
                write_row(&mut w, &[
                    entry.neuron.id.to_string(),
                    ctx_id.to_string(),
                    NORMAL_CONTEXT_AGE.to_string(),
                    "1".to_string(),
                ])?;
            }
        }
        w.flush().map_err(|e| format!("Failed to flush contexts.csv: {}", e))
    }

    /// Write the spatial history window — the frames each neuron was active for, in order. This is the
    /// evidence the one test evaluates, so a snapshot that omitted it would restore a neuron that has
    /// forgotten everything it ever saw and stops growing structure (docs/algorithm.md, "The history").
    /// Streamed row-by-row: with a long horizon this is the largest table by far.
    /// Row: `neuron_id,frame,served_child,observed` where `served_child` is empty when the normal served
    /// and `observed` is a space-separated neighbor id list (kept in one field so the row stays fixed-arity).
    fn write_spatial_history(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut w = open_csv(&folder.join("spatial_history.csv"))?;
        for entry in &snapshot.neurons {
            for rec in &entry.neuron.spatial_history {
                let observed = rec.observed.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ");
                write_row(&mut w, &[
                    entry.neuron.id.to_string(),
                    rec.frame.to_string(),
                    rec.served_child.map_or(String::new(), |id| id.to_string()),
                    observed,
                ])?;
            }
        }
        w.flush().map_err(|e| format!("Failed to flush spatial_history.csv: {}", e))
    }

    fn write_neuron_error_stats(&self, folder: &Path, snapshot: &Snapshot) -> Result<(), String> {
        let mut rows: Vec<Vec<String>> = Vec::new();
        for entry in &snapshot.neurons {
            for stat in &entry.neuron.error_stats {
                rows.push(vec![
                    entry.neuron.id.to_string(),
                    stat.age.to_string(),
                    stat.n.to_string(),
                    stat.mean.to_string(),
                    stat.m2.to_string(),
                ]);
            }
        }
        rows.sort_by(|a, b| {
            let id_a: u64 = a[0].parse().unwrap_or(0);
            let id_b: u64 = b[0].parse().unwrap_or(0);
            let age_a: u32 = a[1].parse().unwrap_or(0);
            let age_b: u32 = b[1].parse().unwrap_or(0);
            id_a.cmp(&id_b).then(age_a.cmp(&age_b))
        });
        write_csv(&folder.join("neuron_error_stats.csv"), &rows)
    }

    // ── Folder management ───────────────────────────────────────────────────

    /// Find the most recent backup folder under `backups_dir` (or None).
    /// Folder names follow the `YYYY-MM-DD_HH-mm-ss` pattern; anything else is ignored.
    fn find_latest_backup(&self, backups_dir: &Path) -> Option<PathBuf> {
        if !backups_dir.exists() { return None; }

        let mut folders: Vec<String> = fs::read_dir(backups_dir).ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| is_timestamp_folder(name))
            .collect();

        if folders.is_empty() { return None; }

        // Lex-sort puts newest at the end thanks to zero-padded timestamps
        folders.sort();
        Some(backups_dir.join(folders.last().unwrap()))
    }

    /// Delete backup folders past the MAX_BACKUPS retention cap.
    fn prune_old_backups(&self, backups_dir: &Path) {
        let mut folders: Vec<String> = match fs::read_dir(backups_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|name| is_timestamp_folder(name))
                .collect(),
            Err(_) => return,
        };

        folders.sort();
        let stale_count = folders.len().saturating_sub(MAX_BACKUPS);
        for name in folders.iter().take(stale_count) {
            let path = backups_dir.join(name);
            if let Err(e) = fs::remove_dir_all(&path) {
                eprintln!("   Failed to prune backup {}: {}", name, e);
            } else {
                println!("   Pruned old backup: {}", name);
            }
        }
    }
}

// ── Free functions ──────────────────────────────────────────────────────────

/// Check if a folder name matches the timestamp format `YYYY-MM-DD_HH-mm-ss`.
pub(crate) fn is_timestamp_folder(name: &str) -> bool {
    if name.len() != 19 { return false; }
    let bytes = name.as_bytes();
    // Check pattern: NNNN-NN-NN_NN-NN-NN where N is digit
    bytes[4] == b'-' && bytes[7] == b'-' && bytes[10] == b'_'
        && bytes[13] == b'-' && bytes[16] == b'-'
        && bytes.iter().enumerate()
            .filter(|&(i, _)| i != 4 && i != 7 && i != 10 && i != 13 && i != 16)
            .all(|(_, &b)| b.is_ascii_digit())
}

/// Build a `YYYY-MM-DD_HH-mm-ss` timestamp using local time.
pub(crate) fn format_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Break epoch seconds into UTC date/time components manually.
    // Good enough for sortable folder names — no locale dependency needed.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Civil date from day count (algorithm from Howard Hinnant)
    let z = days as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", y, m, d, hours, minutes, seconds)
}

/// Open a buffered streaming writer for a CSV file. Used by the large tables (connections, contexts,
/// patterns) so they can emit rows incrementally instead of materializing the whole file in memory —
/// a multi-GB brain otherwise OOMs on the single contiguous allocation.
pub(crate) fn open_csv(filepath: &Path) -> Result<BufWriter<File>, String> {
    let file = File::create(filepath)
        .map_err(|e| format!("Failed to create {}: {}", filepath.display(), e))?;
    Ok(BufWriter::new(file))
}

/// Stream one CSV row (each cell escaped, comma-separated, newline-terminated) to a writer.
pub(crate) fn write_row<W: Write>(w: &mut W, fields: &[String]) -> Result<(), String> {
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            w.write_all(b",").map_err(|e| format!("CSV write failed: {}", e))?;
        }
        w.write_all(escape_field(field).as_bytes()).map_err(|e| format!("CSV write failed: {}", e))?;
    }
    w.write_all(b"\n").map_err(|e| format!("CSV write failed: {}", e))
}

/// Write `rows` to `filepath` as CSV, streaming row-by-row through a buffered writer. Each cell is
/// escaped via `escape_field`, fields joined by commas, lines by '\n'. Empty input produces a
/// zero-byte file. Never builds the whole file as one string, so it scales to multi-GB tables.
pub(crate) fn write_csv(filepath: &Path, rows: &[Vec<String>]) -> Result<(), String> {
    let mut w = open_csv(filepath)?;
    for row in rows {
        write_row(&mut w, row)?;
    }
    w.flush().map_err(|e| format!("Failed to flush {}: {}", filepath.display(), e))
}

/// Escape a single CSV cell. Wraps in double-quotes (and doubles up any embedded
/// quotes) when the value contains a delimiter, quote, or newline.
fn escape_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Read a CSV file, returning a vec of row vecs. Splits on '\n' matching the writer.
pub(crate) fn read_csv(filepath: &Path) -> Result<Vec<Vec<String>>, String> {
    let content = fs::read_to_string(filepath)
        .map_err(|e| format!("Failed to read {}: {}", filepath.display(), e))?;
    let mut rows = Vec::new();
    for line in content.split('\n') {
        if line.is_empty() { continue; }
        rows.push(parse_csv_line(line));
    }
    Ok(rows)
}

/// Parse one CSV line into a vec of field strings. Handles the same quoting rules
/// that `escape_field` produces — no multi-line quoted fields.
fn parse_csv_line(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut fields = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'"' {
            // Quoted field: scan until the closing quote, treating "" as literal "
            let mut value = String::new();
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'"' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    value.push('"');
                    i += 2;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    value.push(bytes[i] as char);
                    i += 1;
                }
            }
            fields.push(value);
            // Consume the trailing comma after the closing quote
            if i < bytes.len() && bytes[i] == b',' { i += 1; }
        } else {
            // Unquoted field: scan up to the next comma or EOL
            let mut value = String::new();
            while i < bytes.len() && bytes[i] != b',' {
                value.push(bytes[i] as char);
                i += 1;
            }
            fields.push(value);
            if i < bytes.len() && bytes[i] == b',' { i += 1; }
        }
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_escape_field_plain() {
        assert_eq!(escape_field("hello"), "hello");
        assert_eq!(escape_field("123"), "123");
    }

    #[test]
    fn test_escape_field_with_comma() {
        assert_eq!(escape_field("a,b"), "\"a,b\"");
    }

    #[test]
    fn test_escape_field_with_quotes() {
        assert_eq!(escape_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn test_parse_csv_line() {
        let fields = parse_csv_line("1,hello,world");
        assert_eq!(fields, vec!["1", "hello", "world"]);
    }

    #[test]
    fn test_parse_csv_line_quoted() {
        let fields = parse_csv_line("1,\"a,b\",3");
        assert_eq!(fields, vec!["1", "a,b", "3"]);
    }

    #[test]
    fn test_is_timestamp_folder() {
        assert!(is_timestamp_folder("2024-01-15_14-30-00"));
        assert!(!is_timestamp_folder("not-a-timestamp"));
        assert!(!is_timestamp_folder("2024-01-15"));
    }

    #[test]
    fn test_write_and_read_csv_roundtrip() {
        let dir = temp_dir().join("brain_test_csv");
        fs::create_dir_all(&dir).unwrap();
        let filepath = dir.join("test.csv");

        let rows = vec![
            vec!["1".to_string(), "hello".to_string()],
            vec!["2".to_string(), "world".to_string()],
        ];
        write_csv(&filepath, &rows).unwrap();
        let loaded = read_csv(&filepath).unwrap();
        assert_eq!(loaded, rows);

        // cleanup
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = temp_dir().join("brain_test_backup");
        fs::remove_dir_all(&dir).ok(); // clean from previous runs
        fs::create_dir_all(&dir).unwrap();

        let backup = Backup::new(0.01);

        // Build a minimal snapshot
        let snapshot = Snapshot {
            neurons: vec![
                SnapshotNeuronEntry {
                    neuron: SerializedNeuron {
                        id: 1,
                        pattern_forget_rate: 0.0,
                        connections: vec![SerializedConnection {
                            distance: 1,
                            to_neuron_id: 2,
                            strength: 0.5,
                            reward: 0.0,
                        }],
                        children: Vec::new(),
                        context_refs: Vec::new(),
                        error_stats: Vec::new(),
                        normal_context: Vec::new(),
                        spatial_history: Vec::new(),
                    },
                    temporal_level: 0,
                    spatial_level: 0,
                    base_neuron: Some(BaseNeuron {
                        channel_id: 1,
                        neuron_type: NeuronType::Event,
                        coordinate: Coordinate { dim_id: 1, bucket_id: 2 },
                    }),
                    parent_id: None,
                },
                SnapshotNeuronEntry {
                    neuron: SerializedNeuron {
                        id: 2,
                        pattern_forget_rate: 0.0,
                        connections: Vec::new(),
                        children: Vec::new(),
                        context_refs: Vec::new(),
                        error_stats: Vec::new(),
                        normal_context: Vec::new(),
                        spatial_history: Vec::new(),
                    },
                    temporal_level: 0,
                    spatial_level: 0,
                    base_neuron: Some(BaseNeuron {
                        channel_id: 1,
                        neuron_type: NeuronType::Event,
                        coordinate: Coordinate { dim_id: 1, bucket_id: 3 },
                    }),
                    parent_id: None,
                },
            ],
            channel_name_to_id: {
                let mut m = FxHashMap::default();
                m.insert("test_channel".to_string(), 1);
                m
            },
            dimension_name_to_id: {
                let mut m = FxHashMap::default();
                m.insert("price".to_string(), 1);
                m
            },
        };

        // Save
        let folder = backup.save(&dir, "latest", &snapshot).unwrap();
        assert!(folder.exists());

        // Load
        let loaded = backup.load(&dir, "latest").unwrap();
        assert_eq!(loaded.neurons.len(), 2);
        assert_eq!(loaded.channel_name_to_id.get("test_channel"), Some(&1));
        assert_eq!(loaded.dimension_name_to_id.get("price"), Some(&1));

        // Verify a connection was preserved
        let neuron1 = loaded.neurons.iter().find(|e| e.neuron.id == 1).unwrap();
        assert_eq!(neuron1.neuron.connections.len(), 1);
        assert_eq!(neuron1.neuron.connections[0].to_neuron_id, 2);

        // cleanup
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_reads_old_six_column_patterns_csv() {
        // Older backups wrote patterns.csv with extra trailing columns (a `fires` column, and the
        // retired evidence/price payment state — up to 6 columns). The add pass prices a child before
        // it is created, so restore ignores everything past strength but must still parse the leading
        // columns correctly rather than choking on the extra fields.
        let dir = temp_dir().join("brain_test_backup_legacy");
        fs::remove_dir_all(&dir).ok();
        let folder = dir.join("backups").join("legacy_test");
        fs::create_dir_all(&folder).unwrap();

        write_csv(&folder.join("channels.csv"), &[]).unwrap();
        write_csv(&folder.join("dimensions.csv"), &[]).unwrap();
        write_csv(&folder.join("neurons.csv"), &[
            vec!["1".to_string(), "0".to_string(), "0".to_string()],
            vec!["2".to_string(), "0".to_string(), "1".to_string()],
        ]).unwrap();
        write_csv(&folder.join("patterns.csv"), &[
            vec!["2".to_string(), "1".to_string(), "3.0".to_string(), "7".to_string(), "12.5".to_string(), "4.0".to_string()],
        ]).unwrap();

        let backup = Backup::new(0.01);
        let loaded = backup.load(&dir, "legacy_test").unwrap();

        let parent = loaded.neurons.iter().find(|e| e.neuron.id == 1).unwrap();
        let child = parent.neuron.children.iter().find(|c| c.pattern_id == 2).unwrap();
        assert_eq!(child.activation_strength, 3.0);

        // cleanup
        fs::remove_dir_all(&dir).ok();
    }
}
