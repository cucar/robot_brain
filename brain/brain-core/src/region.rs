/// Region — wraps Column[C] and parallelizes work across them using Rayon.
///
/// # Threading model
///
/// Every method that touches columns follows the same three-phase pattern:
///
///   1. **Route** — partition items into per-column work lists by `neuron_id % C`
///   2. **Dispatch** — hand each column its work list and let them run in parallel
///   3. **Collect** — gather results back in column-index order
///
/// No locks, barriers, or channels are needed because columns never share mutable
/// state — each column exclusively owns its neuron partition. Shared inputs
/// (level context, error pattern ids, active neurons) are passed as immutable
/// references, safe to read from any thread without synchronization.
///
/// # Rayon primitives used
///
/// The three phases map to Rayon calls like this (using process_level as example):
///
/// ```text
/// // Phase 1 — Route: build per-column task lists (sequential, before parallelism starts)
/// let column_tasks: Vec<Vec<Task>> = self.build_column_tasks(tasks);
///
/// // Phase 2 — Dispatch:
/// let nested: Vec<Vec<Result>> = self.columns
///     .par_iter_mut()                     // iterate columns in parallel; each thread
///                                         // gets exclusive &mut to one Column
///     .zip(column_tasks.into_par_iter())  // pair each column with its task list;
///                                         // into_par_iter consumes the Vec so each
///                                         // thread owns its tasks (no sharing)
///     .map(|(col, col_tasks)| {           // each thread runs this closure independently
///         col.process_level(&col_tasks, ...)
///     })
///     .collect();                         // gather Vec<Vec<Result>> in column-index order;
///                                         // Rayon preserves the original ordering
///
/// // Phase 3 — Collect: flatten nested results (sequential, parallelism is done)
/// nested.into_iter()                      // plain (non-parallel) iterator over outer Vec
///     .flatten()                          // unwrap each inner Vec into a flat stream
///     .collect()                          // gather into final Vec<Result>
/// ```
///
/// Becomes an MPI rank in a future phase; currently a pure router/aggregator.
/// Never exposes single-neuron access.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::column::{
    Column, ColumnProcessResult, DeleteOp, DeleteResult,
    NeuronCreateSpec, SnapshotEntry,
};
use crate::context::TemporalContext;
use crate::neuron::{
    ActiveNeuron, TemporalContextRefUpdate, TemporalNeuron,
    SerializedNeuron, Vote,
};
use crate::thalamus::{SpatialInstallOp, SpatialInstallResult};
use crate::types::{
    ChannelId, DeathFrameEntry, Distance, GroupMode, FrameNumber, NeuronId, Reward,
};

pub struct Region {
    /// Number of columns.
    c: usize,
    /// Column instances, indexed by column index.
    columns: Vec<Column>,
}

impl Region {
    pub fn new(
        c: usize,
        channel_actions: &FxHashMap<ChannelId, Vec<NeuronId>>,
        channel_default_actions: &FxHashMap<ChannelId, NeuronId>,
        context_length: u32,
        group_threshold: f64,
        group_mode: GroupMode,
        learning: bool,
        horizon: u32,
    ) -> Self {
        let mut columns = Vec::with_capacity(c);
        for _ in 0..c {
            columns.push(Column::new(
                channel_actions.clone(),
                channel_default_actions.clone(),
                context_length,
                group_threshold,
                group_mode,
                learning,
                horizon,
            ));
        }
        Self { c, columns }
    }

    // ── Routing ────────────────────────────────────────────────────────────

    /// Pure deterministic column-routing function.
    /// Maps a neuron id to its owning column within this region.
    pub fn route_neuron(&self, neuron_id: NeuronId) -> usize {
        (neuron_id as usize) % self.c
    }

    /// Partition a flat batch into per-column index lists using a key extractor.
    /// Returns a Vec indexed by column, each entry holding the source indices
    /// whose items route to that column.
    fn partition_by_column<T, F>(&self, batch: &[T], key: F) -> Vec<Vec<usize>>
    where
        F: Fn(&T) -> NeuronId,
    {
        let mut per_column: Vec<Vec<usize>> = (0..self.c).map(|_| Vec::new()).collect();
        for (i, item) in batch.iter().enumerate() {
            let col = self.route_neuron(key(item));
            per_column[col].push(i);
        }
        per_column
    }

    // ── Op-3: Process level (hot path) ─────────────────────────────────────

    /// SPATIAL sweep: route tasks to owning columns, fan out in parallel, concatenate results
    /// in column-index order (stable regardless of thread scheduling).
    pub fn process_spatial_level(
        &mut self,
        tasks: &[(NeuronId, Vec<ActiveNeuron>, crate::context::SpatialContext)],
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        spatial_capacity: usize,
        frame_number: FrameNumber,
    ) -> Vec<crate::column::SpatialColumnResult> {

        // Phase 1: Route — clone each task into its owning column's work list.
        // Must happen before the parallel fan-out so each column owns a list it can consume independently.
        let column_tasks = self.build_column_tasks(tasks, |t| t.0);

        // Phase 2: Dispatch — each column processes its tasks in parallel.
        // The per-task actives and neighbor-filtered observed co-activation travel inside each tuple,
        // so there is no shared mutable state to synchronize.
        let nested: Vec<Vec<crate::column::SpatialColumnResult>> = self.columns.par_iter_mut()
            .zip(column_tasks.into_par_iter())
            .map(|(col, col_tasks)| {
                if col_tasks.is_empty() { return Vec::new(); }
                col.process_spatial_level(&col_tasks, new_error_pattern_ids, spatial_capacity, frame_number)
            })
            .collect();

        // Phase 3: Collect — flatten in column-index order (the parallel iterator preserves it).
        nested.into_iter().flatten().collect()
    }

    /// TEMPORAL sweep: route tasks to owning columns, fan out in parallel, concatenate results
    /// in column-index order (stable regardless of thread scheduling).
    pub fn process_temporal_level(
        &mut self,
        temporal_neurons: &[TemporalNeuron],
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<ColumnProcessResult> {
        // Phase 1: Route — clone each neuron into its owning column's work list.
        // Must happen before par_iter so each column gets an owned Vec it can consume independently.
        let column_neurons = self.build_column_tasks(temporal_neurons, |n| n.neuron_id);

        // Phase 2: Dispatch — each column processes its neurons in parallel.
        // Shared refs (level_context, new_error_pattern_ids) are read-only and implement Sync,
        // so no synchronization needed. Each neuron carries its own learning work (or None).
        let nested: Vec<Vec<ColumnProcessResult>> = self.columns.par_iter_mut()
            .zip(column_neurons.into_par_iter())
            .map(|(col, col_neurons)| {
                if col_neurons.is_empty() { return Vec::new(); }
                col.process_temporal_level(
                    &col_neurons, memory_depth, level_context, new_error_pattern_ids,
                    frame_number,
                )
            })
            .collect();

        // Phase 3: Collect — flatten in column-index order (Rayon preserves it).
        nested.into_iter().flatten().collect()
    }

    /// Route tasks to their owning columns by neuron_id. Returns a Vec indexed
    /// by column, each entry holding the cloned tasks for that column.
    fn build_column_tasks<T: Clone>(
        &self,
        tasks: &[T],
        key: impl Fn(&T) -> NeuronId,
    ) -> Vec<Vec<T>> {
        let per_column_indices = self.partition_by_column(tasks, key);
        per_column_indices.iter()
            .map(|task_indices| {
                task_indices.iter().map(|&i| tasks[i].clone()).collect()
            })
            .collect()
    }

    // ── Spatial target decoration (restore) ────────────────────────────────

    /// Fan out the base-neuron position metadata so every column rebuilds its neurons' spatial
    /// target positions. Restore-only: at runtime the metadata is captured at connection-learn time.
    pub fn decorate_spatial_targets(&mut self, meta: &FxHashMap<NeuronId, ChannelId>) {
        self.columns.par_iter_mut()
            .for_each(|col| col.decorate_spatial_targets(meta));
    }

    // ── Spatial correction install (1c) ────────────────────────────────────

    /// Distribute spatial install ops to owning columns by parent_id, dispatch in parallel,
    /// and merge per-column results (deaths + context-ref updates).
    pub fn install_spatial_corrections(&mut self, ops: Vec<SpatialInstallOp>) -> SpatialInstallResult {
        let mut by_column: Vec<Vec<SpatialInstallOp>> = (0..self.c).map(|_| Vec::new()).collect();
        for op in ops {
            let c = self.route_neuron(op.parent_id);
            by_column[c].push(op);
        }

        let per_column: Vec<SpatialInstallResult> = self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .map(|(col, col_ops)| {
                if col_ops.is_empty() {
                    return SpatialInstallResult { deaths: Vec::new(), context_ref_updates: Vec::new() };
                }
                col.install_spatial_corrections(col_ops)
            })
            .collect();

        let mut deaths = Vec::new();
        let mut context_ref_updates: Vec<crate::neuron::SpatialContextRefUpdate> = Vec::new();
        for res in per_column {
            deaths.extend(res.deaths);
            context_ref_updates.extend(res.context_ref_updates);
        }
        SpatialInstallResult { deaths, context_ref_updates }
    }

    // ── Op-5: temporal context-ref updates ─────────────────────────────────────

    /// Apply TEMPORAL contextRef updates against owned Neurons. Updates are routed by
    /// neuron_id (the target neuron whose temporal_context_refs change).
    pub fn update_temporal_context_refs(&mut self, updates: &[(NeuronId, Vec<TemporalContextRefUpdate>)]) {
        let by_column = self.route_context_ref_updates(updates);
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_updates)| {
                if !col_updates.is_empty() {
                    col.update_temporal_context_refs(&col_updates);
                }
            });
    }

    /// Commit the winning child-activation requests this region owns, routed by parent neuron id.
    pub fn commit_spatial_frames(&mut self, ops: Vec<crate::column::SpatialCommitOp>, frame_number: FrameNumber) {
        let mut by_column: Vec<Vec<crate::column::SpatialCommitOp>> = (0..self.c).map(|_| Vec::new()).collect();
        for op in ops {
            let c = self.route_neuron(op.parent_id);
            by_column[c].push(op);
        }
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_ops)| {
                if !col_ops.is_empty() { col.commit_spatial_frames(&col_ops, frame_number); }
            });
    }

    /// Drop this frame from the history of every inhibited neuron this region owns, routed by neuron id.
    pub fn prune_inhibited_spatial_history(&mut self, ids: &[NeuronId], frame_number: FrameNumber) {
        let mut by_column: Vec<Vec<NeuronId>> = (0..self.c).map(|_| Vec::new()).collect();
        for &id in ids {
            by_column[self.route_neuron(id)].push(id);
        }
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_ids)| {
                if !col_ids.is_empty() { col.prune_inhibited_spatial_history(&col_ids, frame_number); }
            });
    }

    /// Apply SPATIAL context-ref updates against owned Neurons.
    pub fn update_spatial_context_refs(&mut self, updates: &[(NeuronId, Vec<crate::neuron::SpatialContextRefUpdate>)]) {
        let mut by_column: Vec<Vec<(NeuronId, Vec<crate::neuron::SpatialContextRefUpdate>)>> =
            (0..self.c).map(|_| Vec::new()).collect();
        for (neuron_id, upd) in updates {
            let col = self.route_neuron(*neuron_id);
            by_column[col].push((*neuron_id, upd.clone()));
        }
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_updates)| {
                if !col_updates.is_empty() {
                    col.update_spatial_context_refs(&col_updates);
                }
            });
    }

    /// Partition context ref updates into per-column work lists by target neuron_id.
    fn route_context_ref_updates(
        &self,
        updates: &[(NeuronId, Vec<TemporalContextRefUpdate>)],
    ) -> Vec<Vec<(NeuronId, Vec<TemporalContextRefUpdate>)>> {
        let mut by_column: Vec<Vec<(NeuronId, Vec<TemporalContextRefUpdate>)>> =
            (0..self.c).map(|_| Vec::new()).collect();
        for (neuron_id, upd) in updates {
            let col = self.route_neuron(*neuron_id);
            by_column[col].push((*neuron_id, upd.clone()));
        }
        by_column
    }

    // ── Op-1/Op-4: Neuron creation ─────────────────────────────────────────

    /// Construct new Neurons in their owning columns. Specs are routed by spec.id.
    pub fn create_neurons(&mut self, specs: &[NeuronCreateSpec]) {
        // Phase 1: Route specs to owning columns.
        let by_column = self.route_neuron_specs(specs);

        // Phase 2: Dispatch — each column creates its neurons in parallel.
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_specs)| {
                if !col_specs.is_empty() {
                    col.create_neurons(&col_specs);
                }
            });
    }

    /// Partition neuron creation specs into per-column work lists by neuron id.
    fn route_neuron_specs(&self, specs: &[NeuronCreateSpec]) -> Vec<Vec<NeuronCreateSpec>> {
        let mut by_column: Vec<Vec<NeuronCreateSpec>> =
            (0..self.c).map(|_| Vec::new()).collect();
        for spec in specs {
            let col = self.route_neuron(spec.id);
            by_column[col].push(spec.clone());
        }
        by_column
    }

    // ── Brain.learn(): supervised action wiring + read-only vote sweep ────

    /// Route voter→action wirings to owning columns by voter_id and dispatch in parallel.
    /// Each tuple is (voter_id, action_id, reward).
    /// The column applies additive learn_action_connection on the voter neuron at the supplied distance.
    pub fn learn_action_connections(&mut self, wirings: &[(NeuronId, NeuronId, Reward)], distance: Distance) {
        let mut by_column: Vec<Vec<(NeuronId, NeuronId, Reward)>> = (0..self.c).map(|_| Vec::new()).collect();
        for &w in wirings {
            let col = self.route_neuron(w.0);
            by_column[col].push(w);
        }
        self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .for_each(|(col, col_wirings)| {
                if !col_wirings.is_empty() {
                    col.learn_action_connections(&col_wirings, distance);
                }
            });
    }

    /// Route (voter_id, age) pairs to owning columns by voter_id and run a read-only vote sweep in parallel.
    /// Returns per-voter vote lists in column-index order.
    pub fn collect_votes_for_voter_ages(&self, voter_ages: &[(NeuronId, Distance)]) -> Vec<(NeuronId, Vec<Vote>)> {
        let mut by_column: Vec<Vec<(NeuronId, Distance)>> = (0..self.c).map(|_| Vec::new()).collect();
        for &pair in voter_ages {
            let col = self.route_neuron(pair.0);
            by_column[col].push(pair);
        }
        let nested: Vec<Vec<(NeuronId, Vec<Vote>)>> = self.columns.par_iter()
            .zip(by_column.into_par_iter())
            .map(|(col, col_pairs)| {
                if col_pairs.is_empty() { return Vec::new(); }
                col.collect_votes_for_voter_ages(&col_pairs)
            })
            .collect();
        nested.into_iter().flatten().collect()
    }

    // ── Op-2: Delete cascade ───────────────────────────────────────────────

    /// Apply a batch of delete operations across columns in parallel.
    /// Each column processes its ops independently; results are merged afterward.
    pub fn delete_neurons(&mut self, op_batch: &[DeleteOp], current_frame: FrameNumber) -> DeleteResult {
        // Phase 1: Route delete ops to owning columns.
        let by_column = self.route_delete_ops(op_batch);

        // Phase 2: Dispatch — each column runs its deletes in parallel,
        // producing outbound ops (for other columns), deleted ids, and
        // cascade candidates.
        let per_column_results: Vec<DeleteResult> = self.columns.par_iter_mut()
            .zip(by_column.into_par_iter())
            .map(|(col, col_ops)| {
                if col_ops.is_empty() {
                    return DeleteResult {
                        outbound_ops: Vec::new(),
                        deleted_ids: Vec::new(),
                        newly_deletable_ids: Vec::new(),
                    };
                }
                col.delete_neurons(&col_ops, current_frame)
            })
            .collect();

        // Phase 3: Merge — combine results from all columns. The caller
        // (Thalamus) feeds outbound_ops into the next cascade pulse.
        Self::merge_delete_results(per_column_results)
    }

    /// Partition delete ops into per-column work lists by target_id.
    fn route_delete_ops(&self, op_batch: &[DeleteOp]) -> Vec<Vec<DeleteOp>> {
        let mut by_column: Vec<Vec<DeleteOp>> =
            (0..self.c).map(|_| Vec::new()).collect();
        for op in op_batch {
            let col = self.route_neuron(op.target_id());
            by_column[col].push(op.clone());
        }
        by_column
    }

    /// Flatten per-column delete results into a single merged result.
    fn merge_delete_results(per_column: Vec<DeleteResult>) -> DeleteResult {
        let mut outbound_ops = Vec::new();
        let mut deleted_ids = Vec::new();
        let mut newly_deletable_ids = Vec::new();
        for result in per_column {
            outbound_ops.extend(result.outbound_ops);
            deleted_ids.extend(result.deleted_ids);
            newly_deletable_ids.extend(result.newly_deletable_ids);
        }
        DeleteResult { outbound_ops, deleted_ids, newly_deletable_ids }
    }

    // ── Inspection ─────────────────────────────────────────────────────────

    /// Look up the stored context entries for a child pattern on a given parent.
    /// Returns None if the parent doesn't live in this region or has no such child.
    pub fn get_child_context_entries(&self, parent_id: NeuronId, child_id: NeuronId) -> Option<Vec<(NeuronId, Distance, f64)>> {
        let c = self.route_neuron(parent_id);
        self.columns[c].get_child_context_entries(parent_id, child_id)
    }

    /// Dump a neuron's outgoing connections (distance, target, strength, reward).
    pub fn get_neuron_connections(&self, neuron_id: NeuronId) -> Option<Vec<(Distance, NeuronId, f64, f64)>> {
        let c = self.route_neuron(neuron_id);
        self.columns[c].get_neuron_connections(neuron_id)
    }

    // ── Snapshot / restore ─────────────────────────────────────────────────

    /// Collect serialized {id, neuron} entries from all columns for snapshotting.
    pub fn get_snapshot(&self) -> Vec<SnapshotEntry> {
        self.columns.par_iter()
            .flat_map_iter(|col| col.get_snapshot())
            .collect()
    }

    /// Distribute serialized neuron specs to columns for reconstruction on load.
    /// specs_by_column is a Vec indexed by column_idx, each entry a list of
    /// serialized neurons already routed to that column.
    pub fn restore_snapshot(&mut self, specs_by_column: Vec<Vec<SerializedNeuron>>) {
        self.columns.par_iter_mut()
            .zip(specs_by_column.into_par_iter())
            .for_each(|(col, specs)| {
                col.restore_snapshot(&specs);
            });
    }

    // ── Death ledger support ───────────────────────────────────────────────

    /// Collect computed death frames from all columns' routing tables.
    /// Read-only — does not mutate neuron state.
    pub fn collect_death_frames(&self) -> Vec<DeathFrameEntry> {
        self.columns.par_iter()
            .flat_map_iter(|col| col.collect_death_frames())
            .collect()
    }

    /// Materialize lazy decay across all columns and collect death frame entries
    /// so Thalamus can rebuild the death ledger.
    pub fn materialize_and_reset_neurons(&mut self, current_frame: FrameNumber) -> Vec<DeathFrameEntry> {
        self.columns.par_iter_mut()
            .flat_map_iter(|col| col.materialize_and_reset_neurons(current_frame))
            .collect()
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    /// Clear all neurons from all columns. Used during reset before restore.
    pub fn clear(&mut self) {
        self.columns.par_iter_mut().for_each(|col| col.clear());
    }

    /// Update shared action sets when a new channel is registered.
    /// Each column gets a clone of the current action sets so per-frame
    /// calls never reach back to Thalamus.
    pub fn update_action_sets(
        &mut self,
        channel_actions: &FxHashMap<ChannelId, Vec<NeuronId>>,
        channel_default_actions: &FxHashMap<ChannelId, NeuronId>,
    ) {
        self.columns.par_iter_mut().for_each(|col| {
            col.update_action_sets(channel_actions, channel_default_actions);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region() -> Region {
        Region::new(
            2, // 2 columns
            &FxHashMap::default(),
            &FxHashMap::default(),
            2,
            0.5,
            GroupMode::Static,
            true,
            100,
        )
    }

    #[test]
    fn test_route_neuron() {
        let r = make_region();
        // neuron 1 → column 1, neuron 2 → column 0, neuron 3 → column 1
        assert_eq!(r.route_neuron(1), 1);
        assert_eq!(r.route_neuron(2), 0);
        assert_eq!(r.route_neuron(3), 1);
    }

    #[test]
    fn test_create_neurons_routes_correctly() {
        let mut r = make_region();
        r.create_neurons(&[
            NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None },
            NeuronCreateSpec { id: 2, forget_rate: 0.0, connections: None },
            NeuronCreateSpec { id: 3, forget_rate: 0.0, connections: None },
        ]);
        let snapshot = r.get_snapshot();
        assert_eq!(snapshot.len(), 3);
    }

    #[test]
    fn test_clear() {
        let mut r = make_region();
        r.create_neurons(&[NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None }]);
        assert_eq!(r.get_snapshot().len(), 1);
        r.clear();
        assert_eq!(r.get_snapshot().len(), 0);
    }
}
