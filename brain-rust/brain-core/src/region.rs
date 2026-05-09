/// Region — wraps Column[C]. Becomes an MPI rank in Phase 2; in single-threaded
/// Phase 1 it's a pure router/aggregator. Never exposes single-neuron access.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::column::{
    Column, ColumnProcessResult, DeathFrameEntry, DeleteOp, DeleteResult,
    NeuronCreateSpec, SnapshotEntry,
};
use crate::context::Context;
use crate::neuron::{
    ActiveNeuron, AgeState, Correction, ContextRefUpdate, ErrorFeedback,
    SerializedNeuron,
};
use crate::types::{
    ChannelId, Distance, ErrorMode, FrameNumber, NeuronId,
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
        channel_actions: &FxHashMap<ChannelId, FxHashSet<NeuronId>>,
        action_ids: &FxHashSet<NeuronId>,
        merge_threshold: f64,
        error_mode: ErrorMode,
        error_threshold: f64,
    ) -> Self {
        let mut columns = Vec::with_capacity(c);
        for _ in 0..c {
            columns.push(Column::new(
                channel_actions.clone(),
                action_ids.clone(),
                merge_threshold,
                error_mode,
                error_threshold,
            ));
        }
        Self { c, columns }
    }

    /// Pure deterministic column-routing function.
    /// Maps a neuron id to its owning column within this region.
    pub fn route_neuron(&self, neuron_id: NeuronId) -> usize {
        (neuron_id as usize) % self.c
    }

    /// Bucket a flat batch by owning column using a key extractor function.
    /// Returns a Vec indexed by column_idx, each entry the sub-list for that column.
    fn bucket_by_column<T, F>(&self, batch: &[T], key: F) -> Vec<Vec<usize>>
    where
        F: Fn(&T) -> NeuronId,
    {
        let mut buckets: Vec<Vec<usize>> = (0..self.c).map(|_| Vec::new()).collect();
        for (i, item) in batch.iter().enumerate() {
            let col = self.route_neuron(key(item));
            buckets[col].push(i);
        }
        buckets
    }

    /// Op-3 down-trip. Bucket tasks by owning column, fan out, concatenate
    /// results in column-index order (stable regardless of thread scheduling).
    pub fn process_level(
        &mut self,
        tasks: &[(NeuronId, FxHashMap<Distance, AgeState>, Vec<Correction>, Vec<ErrorFeedback>)],
        memory_depth: u32,
        level_context: Option<&Context>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        new_active_neurons: &[ActiveNeuron],
        frame_number: FrameNumber,
    ) -> Vec<ColumnProcessResult> {
        // bucket task indices by column
        let indices_by_column = self.bucket_by_column(tasks, |t| t.0);

        let mut results = Vec::new();
        for (col_idx, task_indices) in indices_by_column.iter().enumerate() {
            if task_indices.is_empty() { continue; }

            // collect the tasks for this column
            let col_tasks: Vec<_> = task_indices.iter()
                .map(|&i| tasks[i].clone())
                .collect();

            let col_results = self.columns[col_idx].process_level(
                &col_tasks, memory_depth, level_context, new_error_pattern_ids,
                new_active_neurons, frame_number,
            );
            results.extend(col_results);
        }
        results
    }

    /// Op-5 (deferred): Apply contextRef updates against owned Neurons. Updates are routed by
    /// neuron_id (the target neuron whose contextRefs change).
    pub fn update_context_refs(&mut self, updates: &[(NeuronId, Vec<ContextRefUpdate>)]) {
        // bucket by column
        let mut by_column: Vec<Vec<(NeuronId, Vec<ContextRefUpdate>)>> = (0..self.c).map(|_| Vec::new()).collect();
        for (neuron_id, upd) in updates {
            let col = self.route_neuron(*neuron_id);
            by_column[col].push((*neuron_id, upd.clone()));
        }
        for (col_idx, col_updates) in by_column.iter().enumerate() {
            if !col_updates.is_empty() {
                self.columns[col_idx].update_context_refs(col_updates);
            }
        }
    }

    /// Op-1/Op-4: Construct new Neurons in their owning columns. Specs are routed by spec.id.
    pub fn create_neurons(&mut self, specs: &[NeuronCreateSpec]) {
        let mut by_column: Vec<Vec<NeuronCreateSpec>> = (0..self.c).map(|_| Vec::new()).collect();
        for spec in specs {
            let col = self.route_neuron(spec.id);
            by_column[col].push(spec.clone());
        }
        for (col_idx, col_specs) in by_column.iter().enumerate() {
            if !col_specs.is_empty() {
                self.columns[col_idx].create_neurons(col_specs);
            }
        }
    }

    /// Clear all neurons from all columns. Used during reset before restore.
    pub fn clear(&mut self) {
        for col in &mut self.columns {
            col.clear();
        }
    }

    /// Collect serialized {id, neuron} entries from columns for snapshotting.
    pub fn get_snapshot(&self) -> Vec<SnapshotEntry> {
        let mut entries = Vec::new();
        for col in &self.columns {
            entries.extend(col.get_snapshot());
        }
        entries
    }

    /// Distribute serialized neuron specs to columns for reconstruction on load.
    /// specs_by_column is a Vec indexed by column_idx, each entry a list of serialized neurons.
    pub fn restore_snapshot(&mut self, specs_by_column: Vec<Vec<SerializedNeuron>>) {
        for (col_idx, specs) in specs_by_column.into_iter().enumerate() {
            if col_idx < self.c {
                self.columns[col_idx].restore_snapshot(&specs);
            }
        }
    }

    /// Collect computed death frames from all columns' routing tables.
    /// Read-only — does not mutate neuron state.
    pub fn collect_death_frames(&self) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();
        for col in &self.columns {
            entries.extend(col.collect_death_frames());
        }
        entries
    }

    /// Materialize lazy decay across all columns and collect death frame entries
    /// so Thalamus can rebuild the death ledger.
    pub fn materialize_and_reset_neurons(&mut self, current_frame: FrameNumber) -> Vec<DeathFrameEntry> {
        let mut death_entries = Vec::new();
        for col in &mut self.columns {
            death_entries.extend(col.materialize_and_reset_neurons(current_frame));
        }
        death_entries
    }

    /// Op-2: Apply a batch of delete operations in the columns owned.
    pub fn delete_neurons(&mut self, op_batch: &[DeleteOp], current_frame: FrameNumber) -> DeleteResult {
        // bucket ops by column using target_id
        let mut by_column: Vec<Vec<DeleteOp>> = (0..self.c).map(|_| Vec::new()).collect();
        for op in op_batch {
            let col = self.route_neuron(op.target_id());
            by_column[col].push(op.clone());
        }

        let mut outbound_ops = Vec::new();
        let mut deleted_ids = Vec::new();
        let mut newly_deletable_ids = Vec::new();

        for (col_idx, col_ops) in by_column.iter().enumerate() {
            if col_ops.is_empty() { continue; }
            let result = self.columns[col_idx].delete_neurons(col_ops, current_frame);
            outbound_ops.extend(result.outbound_ops);
            deleted_ids.extend(result.deleted_ids);
            newly_deletable_ids.extend(result.newly_deletable_ids);
        }

        DeleteResult { outbound_ops, deleted_ids, newly_deletable_ids }
    }

    /// Update shared action sets when a new channel is registered.
    pub fn update_action_sets(
        &mut self,
        channel_actions: &FxHashMap<ChannelId, FxHashSet<NeuronId>>,
        action_ids: &FxHashSet<NeuronId>,
    ) {
        for col in &mut self.columns {
            col.update_action_sets(channel_actions, action_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region() -> Region {
        Region::new(
            2, // 2 columns
            &FxHashMap::default(),
            &FxHashSet::default(),
            0.5,
            ErrorMode::Static,
            0.5,
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
