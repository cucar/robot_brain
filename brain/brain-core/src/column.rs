/// Column — owns a partition of Neuron instances and exposes batch operations
/// on them. Becomes a worker thread in Phase 2; in single-threaded Phase 1 every
/// method is a synchronous local call.
///
/// Action sets are passed at init time so per-frame calls never reach back
/// to Thalamus. `self.neurons` is the sole storage for owned Neurons.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::context::TemporalContext;
use crate::neuron::{
    ActiveNeuron, AgeState, AgeVotes, Correction, TemporalContextRefUpdate, SpatialContextRefUpdate,
    CorrectionActivation, ErrorFeedback, Neuron, PatternMatch, Vote,
};
use crate::thalamus::{SpatialInstallOp, SpatialInstallResult};
use crate::types::{
    ChannelId, DeathFrameEntry, Distance, DimensionId, GroupMode, FrameNumber,
    NeuronId, Reward, Strength,
};

/// Result of processLevel — one per task (neuron). Tagged with parent_id so the
/// caller can associate results with the neuron that produced them.
#[derive(Debug, Clone)]
pub struct ColumnProcessResult {
    pub parent_id: NeuronId,
    pub matches: Vec<PatternMatch>,
    pub correction_activations: Vec<CorrectionActivation>,
    pub context_ref_updates: Vec<TemporalContextRefUpdate>,
    pub votes: Vec<AgeVotes>,
    pub timings: crate::neuron::NeuronOpTimings,
}

/// Per-neuron output of the spatial sweep, tagged with its owner. Votes never leave the neuron —
/// it evaluates its own prediction and surfaces only the resulting correction request.
pub struct SpatialColumnResult {
    pub parent_id: NeuronId,
    pub matches: Vec<PatternMatch>,
    pub correction_request: Option<crate::neuron::SpatialCorrectionRequest>,
    /// A pattern paid off by this frame's deposit — the thalamus adds it to the codebook count.
    pub promotion: Option<NeuronId>,
    pub timings: crate::neuron::NeuronOpTimings,
}

/// Result of a delete cascade pulse inside a column.
pub struct DeleteResult {
    pub outbound_ops: Vec<DeleteOp>,
    pub deleted_ids: Vec<NeuronId>,
    pub newly_deletable_ids: Vec<NeuronId>,
}

/// Operations that travel between columns during delete cascades.
/// Each variant carries the data needed to execute without reaching back to Thalamus.
/// Phase-specific variants exist where the work differs in shape (distance vs no-distance).
#[derive(Debug, Clone)]
pub enum DeleteOp {
    /// Destroy a dying neuron and emit cleanup ops for its connections.
    /// Generic — destroy walks both spatial and temporal state on the dying neuron.
    DeleteNeuron { target_id: NeuronId, parent_id: NeuronId },
    /// Remove a dead child pattern from its parent's routing table.
    /// Generic — parent.remove_routing_entry tries both spatial and temporal tables.
    RemovePattern { target_id: NeuronId, pattern_id: NeuronId },
    /// Scrub a dead neuron from a parent's TEMPORAL children's context entries.
    PurgeTemporalContextNeuron { target_id: NeuronId, dying_neuron_id: NeuronId, distances: Vec<Distance> },
    /// Scrub a dead neuron from a parent's SPATIAL children's context entries.
    PurgeSpatialContextNeuron { target_id: NeuronId, dying_neuron_id: NeuronId },
    /// Drop a stale TEMPORAL contextRef on a context neuron.
    RemoveTemporalContextRef { target_id: NeuronId, parent_id: NeuronId, distance: Distance },
    /// Drop a stale SPATIAL contextRef on a context neuron.
    RemoveSpatialContextRef { target_id: NeuronId, parent_id: NeuronId },
}

impl DeleteOp {
    /// Extract the routing key (target_id) for bucketing by region/column.
    pub fn target_id(&self) -> NeuronId {
        match self {
            DeleteOp::DeleteNeuron { target_id, .. } => *target_id,
            DeleteOp::RemovePattern { target_id, .. } => *target_id,
            DeleteOp::PurgeTemporalContextNeuron { target_id, .. } => *target_id,
            DeleteOp::PurgeSpatialContextNeuron { target_id, .. } => *target_id,
            DeleteOp::RemoveTemporalContextRef { target_id, .. } => *target_id,
            DeleteOp::RemoveSpatialContextRef { target_id, .. } => *target_id,
        }
    }
}

/// Serialized neuron snapshot entry.
#[derive(Debug, Clone)]
pub struct SnapshotEntry {
    pub id: NeuronId,
    pub neuron: crate::neuron::SerializedNeuron,
}

pub struct Column {
    /// Shared action neuron sets — per-channel action neuron ids in registration order.
    /// In single-threaded JS these are shared by reference with Thalamus; in Rust
    /// they're cloned at construction.
    channel_actions: FxHashMap<ChannelId, Vec<NeuronId>>,

    /// Per-channel default action neuron id.
    channel_default_actions: FxHashMap<ChannelId, NeuronId>,

    /// TemporalContext length — determines the range of distances (1..context_length-1)
    /// at which default action connections are pre-wired on new neurons.
    context_length: u32,

    /// The single brain-wide grouping coefficient θ, shared by spatial (d=0) and temporal (d>0).
    /// Recognition fires at similarity ≥ θ; correction fires at similarity < θ (error threshold = 1 − θ).
    group_threshold: f64,

    /// How the derived correction threshold adapts from per-unit Welford stats (shared spatial/temporal).
    group_mode: GroupMode,

    // ── Temporary experimental toggles ──────────────────────────────────────────
    // Handed to every neuron this column creates or loads; the field docs live on Neuron.
    match_info: bool,
    error_info: bool,
    trace_match: bool,
    trace_error: bool,

    /// Master learning toggle, fixed at construction and handed to every neuron this column creates or loads.
    learning: bool,

    /// The sole storage for owned Neurons. Keyed by neuron id.
    neurons: FxHashMap<NeuronId, Neuron>,
}

impl Column {
    pub fn new(
        channel_actions: FxHashMap<ChannelId, Vec<NeuronId>>,
        channel_default_actions: FxHashMap<ChannelId, NeuronId>,
        context_length: u32,
        group_threshold: f64,
        group_mode: GroupMode,
        learning: bool,
        match_info: bool,
        error_info: bool,
        trace_match: bool,
        trace_error: bool,
    ) -> Self {
        Self {
            channel_actions,
            channel_default_actions,
            context_length,
            group_threshold,
            group_mode,
            match_info,
            error_info,
            trace_match,
            trace_error,
            learning,
            neurons: FxHashMap::default(),
        }
    }

    /// Op-3 down-trip body for the SPATIAL sweep. Calls neuron.process_spatial_frame on every task
    /// and returns results parent_id-tagged in task order.
    pub fn process_spatial_level(
        &mut self,
        tasks: &[(NeuronId, Vec<ActiveNeuron>, crate::context::SpatialContext)],
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<SpatialColumnResult> {

        // Each task carries its own neighbor-filtered actives and observed co-activation.
        let mut results = Vec::with_capacity(tasks.len());
        for (neuron_id, actives, observed_context) in tasks {
            let neuron = self.neurons.get_mut(neuron_id)
                .unwrap_or_else(|| panic!("Column.process_spatial_level: neuron {} not found", neuron_id));
            let result = neuron.process_spatial_frame(
                Some(observed_context), new_error_pattern_ids, actives, frame_number,
            );
            results.push(SpatialColumnResult {
                parent_id: *neuron_id,
                matches: result.matches,
                correction_request: result.correction_request,
                promotion: result.promotion,
                timings: result.timings,
            });
        }
        results
    }

    /// Op-3 down-trip body for the TEMPORAL sweep. Calls neuron.process_temporal_frame on every
    /// task and returns results parent_id-tagged in task order.
    pub fn process_temporal_level(
        &mut self,
        tasks: &[(NeuronId, FxHashMap<Distance, AgeState>, Vec<Correction>, Vec<ErrorFeedback>, Vec<ActiveNeuron>)],
        memory_depth: u32,
        level_context: Option<&TemporalContext>,
        new_error_pattern_ids: &FxHashSet<NeuronId>,
        frame_number: FrameNumber,
    ) -> Vec<ColumnProcessResult> {
        let mut results = Vec::with_capacity(tasks.len());
        for (neuron_id, age_states, corrections, error_feedback, actives) in tasks {
            let neuron = self.neurons.get_mut(neuron_id)
                .unwrap_or_else(|| panic!("Column.process_temporal_level: neuron {} not found", neuron_id));
            let result = neuron.process_temporal_frame(
                age_states, memory_depth, level_context, new_error_pattern_ids,
                actives, frame_number, corrections, error_feedback,
            );
            results.push(ColumnProcessResult {
                parent_id: *neuron_id,
                matches: result.matches,
                correction_activations: result.correction_activations,
                context_ref_updates: result.context_ref_updates,
                votes: result.votes,
                timings: result.timings,
            });
        }
        results
    }

    /// Process a batch of delete operations against owned neurons.
    /// Returns outbound operations for other columns, deleted neuron ids,
    /// and neuron ids that just became deletable to cascade.
    pub fn delete_neurons(&mut self, op_batch: &[DeleteOp], current_frame: FrameNumber) -> DeleteResult {
        let mut outbound_ops = Vec::new();
        let mut deleted_ids = Vec::new();
        let mut newly_deletable_ids = Vec::new();

        // loop over requested operations and execute them on the neurons owned
        for op in op_batch {
            match op {

                // destroy a dying neuron: walk its state, emit cleanup ops, remove it
                DeleteOp::DeleteNeuron { target_id, parent_id } => {
                    if let Some(result) = self.delete_neuron(*target_id, *parent_id) {
                        outbound_ops.extend(result.outbound_ops);
                        deleted_ids.push(result.deleted_id);
                    }
                    // neuron already destroyed by an earlier op in this cascade if None
                }

                // remove a dead child pattern from its parent's routing table (spatial or temporal)
                DeleteOp::RemovePattern { target_id, pattern_id } => {
                    let ops = self.remove_pattern(*target_id, *pattern_id);
                    outbound_ops.extend(ops);
                }

                // scrub a dead neuron from a parent's temporal children's context entries
                DeleteOp::PurgeTemporalContextNeuron { target_id, dying_neuron_id, distances } => {
                    self.purge_temporal_context_neuron(*target_id, *dying_neuron_id, distances, current_frame, &mut newly_deletable_ids);
                }

                // scrub a dead neuron from a parent's spatial children's context entries
                DeleteOp::PurgeSpatialContextNeuron { target_id, dying_neuron_id } => {
                    self.purge_spatial_context_neuron(*target_id, *dying_neuron_id, current_frame, &mut newly_deletable_ids);
                }

                // drop a stale temporal contextRef on a context neuron
                DeleteOp::RemoveTemporalContextRef { target_id, parent_id, distance } => {
                    self.remove_temporal_context_ref_op(*target_id, *parent_id, *distance);
                }

                // drop a stale spatial contextRef on a context neuron
                DeleteOp::RemoveSpatialContextRef { target_id, parent_id } => {
                    self.remove_spatial_context_ref_op(*target_id, *parent_id);
                }
            }
        }

        DeleteResult { outbound_ops, deleted_ids, newly_deletable_ids }
    }

    /// Destroy a dying neuron. Walk its state to emit cleanup ops, then remove it.
    fn delete_neuron(&mut self, target_id: NeuronId, parent_id: NeuronId) -> Option<DeleteNeuronResult> {
        let neuron = self.neurons.remove(&target_id)?;

        let mut outbound_ops = Vec::new();

        // tell each TEMPORAL parent that referenced this neuron in its children's contexts to scrub it
        for (referencing_parent_id, distances) in neuron.get_temporal_context_refs() {
            outbound_ops.push(DeleteOp::PurgeTemporalContextNeuron {
                target_id: *referencing_parent_id,
                dying_neuron_id: neuron.id,
                distances: distances.iter().copied().collect(),
            });
        }

        // tell each SPATIAL parent that referenced this neuron in its children's contexts to scrub it
        for &referencing_parent_id in neuron.get_spatial_context_refs() {
            outbound_ops.push(DeleteOp::PurgeSpatialContextNeuron {
                target_id: referencing_parent_id,
                dying_neuron_id: neuron.id,
            });
        }

        // Orphan each TEMPORAL child: emit RemoveTemporalContextRef ops for each context entry,
        // then queue the child for deletion. We collect routing-table entries upfront because
        // neuron is moved out of self.neurons (we can't borrow it back mutably from the map).
        let temporal_children: Vec<(NeuronId, Vec<crate::types::ContextEntry>)> = neuron.get_temporal_routing_table()
            .iter()
            .map(|(&child_id, entry)| (child_id, entry.context.get_entries()))
            .collect();
        for (child_pattern_id, entries) in &temporal_children {
            for entry in entries {
                outbound_ops.push(DeleteOp::RemoveTemporalContextRef {
                    target_id: entry.neuron_id,
                    parent_id: neuron.id,
                    distance: entry.distance,
                });
            }
            outbound_ops.push(DeleteOp::DeleteNeuron {
                target_id: *child_pattern_id,
                parent_id: neuron.id,
            });
        }

        // Orphan each SPATIAL child: emit RemoveSpatialContextRef ops for each ctx neuron,
        // then queue the child for deletion. Spatial context has no distance dimension.
        let spatial_children: Vec<(NeuronId, Vec<NeuronId>)> = neuron.get_spatial_routing_table()
            .iter()
            .map(|(&child_id, entry)| (child_id, entry.context.entries().keys().copied().collect()))
            .collect();
        for (child_pattern_id, ctx_neuron_ids) in &spatial_children {
            for &ctx_neuron_id in ctx_neuron_ids {
                outbound_ops.push(DeleteOp::RemoveSpatialContextRef {
                    target_id: ctx_neuron_id,
                    parent_id: neuron.id,
                });
            }
            outbound_ops.push(DeleteOp::DeleteNeuron {
                target_id: *child_pattern_id,
                parent_id: neuron.id,
            });
        }

        // tell parent to remove this pattern from its routing table (tries both spatial and temporal)
        outbound_ops.push(DeleteOp::RemovePattern {
            target_id: parent_id,
            pattern_id: neuron.id,
        });

        Some(DeleteNeuronResult {
            outbound_ops,
            deleted_id: neuron.id,
        })
    }

    /// Remove a dead child pattern from a parent's routing table and context entries.
    /// Checks both spatial and temporal routing tables; emits orphan ops with the right shape
    /// based on which table held the pattern.
    fn remove_pattern(&mut self, target_id: NeuronId, pattern_id: NeuronId) -> Vec<DeleteOp> {
        let parent = match self.neurons.get_mut(&target_id) {
            Some(n) => n,
            None => return Vec::new(), // parent already destroyed in this cascade
        };

        let mut outbound_ops = Vec::new();

        // Try the SPATIAL side first.
        if let Some(entry) = parent.get_spatial_routing_table().get(&pattern_id) {
            let ctx_neuron_ids: Vec<NeuronId> = entry.context.entries().keys().copied().collect();
            for ctx_neuron_id in ctx_neuron_ids {
                if !parent.has_spatial_context_key(pattern_id, ctx_neuron_id) { continue; }
                let is_orphaned = parent.remove_spatial_context(pattern_id, ctx_neuron_id);
                if is_orphaned {
                    outbound_ops.push(DeleteOp::RemoveSpatialContextRef {
                        target_id: ctx_neuron_id,
                        parent_id: target_id,
                    });
                }
            }
            parent.remove_routing_entry(pattern_id);
            return outbound_ops;
        }

        // Otherwise the TEMPORAL side (or nothing — already removed).
        let entries = match parent.get_temporal_routing_table().get(&pattern_id) {
            Some(entry) => entry.context.get_entries(),
            None => return outbound_ops, // pattern already removed by parent's own DeleteNeuron cleanup
        };
        for entry in &entries {
            // same-pulse PurgeTemporalContextNeuron may have already removed this entry
            if !parent.has_temporal_context_key(pattern_id, entry.neuron_id, entry.distance) {
                continue;
            }
            let is_orphaned = parent.remove_temporal_context(pattern_id, entry.neuron_id, entry.distance);
            if is_orphaned {
                outbound_ops.push(DeleteOp::RemoveTemporalContextRef {
                    target_id: entry.neuron_id,
                    parent_id: target_id,
                    distance: entry.distance,
                });
            }
        }
        parent.remove_routing_entry(pattern_id);
        outbound_ops
    }

    /// Scrub a dead neuron from a parent's TEMPORAL children's context entries.
    /// Affected children whose activation strength decayed to zero become cascade candidates.
    fn purge_temporal_context_neuron(
        &mut self,
        target_id: NeuronId,
        dying_neuron_id: NeuronId,
        distances: &[Distance],
        current_frame: FrameNumber,
        newly_deletable_ids: &mut Vec<NeuronId>,
    ) {
        let parent = match self.neurons.get_mut(&target_id) {
            Some(n) => n,
            None => return,
        };

        // same-pulse RemovePattern may have already cleaned some distances
        let remaining_distances: FxHashSet<Distance> = distances.iter()
            .filter(|&&d| parent.has_temporal_context_index_entry(dying_neuron_id, d))
            .copied()
            .collect();
        if remaining_distances.is_empty() { return; }

        let affected_patterns = parent.remove_temporal_context_neuron(dying_neuron_id, &remaining_distances);
        for pattern_id in affected_patterns {
            if parent.can_delete_child(pattern_id, current_frame) {
                newly_deletable_ids.push(pattern_id);
            }
        }
    }

    /// Scrub a dead neuron from a parent's SPATIAL children's context entries.
    /// Affected children whose activation strength decayed to zero become cascade candidates.
    fn purge_spatial_context_neuron(
        &mut self,
        target_id: NeuronId,
        dying_neuron_id: NeuronId,
        current_frame: FrameNumber,
        newly_deletable_ids: &mut Vec<NeuronId>,
    ) {
        let parent = match self.neurons.get_mut(&target_id) {
            Some(n) => n,
            None => return,
        };
        // same-pulse RemovePattern may have already cleaned this entry
        if !parent.has_spatial_context_index_entry(dying_neuron_id) { return; }
        let affected_patterns = parent.remove_spatial_context_neuron(dying_neuron_id);
        for pattern_id in affected_patterns {
            if parent.can_delete_child(pattern_id, current_frame) {
                newly_deletable_ids.push(pattern_id);
            }
        }
    }

    /// Drop a single TEMPORAL contextRef entry on a context neuron.
    fn remove_temporal_context_ref_op(&mut self, target_id: NeuronId, parent_id: NeuronId, distance: Distance) {
        let neuron = match self.neurons.get_mut(&target_id) {
            Some(n) => n,
            None => return,
        };
        // same-pulse op may have already removed this ref
        let has_ref = neuron.get_temporal_context_refs()
            .get(&parent_id)
            .map_or(false, |distances| distances.contains(&distance));
        if !has_ref { return; }
        neuron.remove_temporal_context_ref(parent_id, distance);
    }

    /// Drop a SPATIAL contextRef entry on a context neuron.
    fn remove_spatial_context_ref_op(&mut self, target_id: NeuronId, parent_id: NeuronId) {
        let neuron = match self.neurons.get_mut(&target_id) {
            Some(n) => n,
            None => return,
        };
        // same-pulse op may have already removed this ref
        if !neuron.get_spatial_context_refs().contains(&parent_id) { return; }
        neuron.remove_spatial_context_ref(parent_id);
    }

    /// Inspection: dump a neuron's outgoing connections as
    /// (distance, target_id, strength, reward) tuples.
    pub fn get_neuron_connections(&self, neuron_id: NeuronId) -> Option<Vec<(Distance, NeuronId, f64, f64)>> {
        self.neurons.get(&neuron_id).map(|n| n.get_connections())
    }

    /// Return (parent_id, distance, strength) context entries for a child
    /// pattern stored under the given parent neuron's routing table. Used
    /// by the inspection API to dump a pattern's stored context.
    pub fn get_child_context_entries(&self, parent_id: NeuronId, child_id: NeuronId) -> Option<Vec<(NeuronId, Distance, f64)>> {
        let parent = self.neurons.get(&parent_id)?;
        let entry = parent.get_temporal_routing_table().get(&child_id)?;
        let mut out = Vec::new();
        for (nid, dist_map) in entry.context.entries() {
            for (dist, strength) in dist_map {
                out.push((*nid, *dist, *strength));
            }
        }
        Some(out)
    }

    /// Op-5 (deferred): Apply contextRef updates to owned neurons. Each entry carries the
    /// target neuron_id and a batch of updates for it.
    pub fn update_temporal_context_refs(&mut self, update_batch: &[(NeuronId, Vec<TemporalContextRefUpdate>)]) {
        for (neuron_id, updates) in update_batch {
            if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                neuron.apply_temporal_context_ref_updates(updates);
            }
        }
    }

    /// Rebuild the position metadata of spatial connection targets on every owned neuron.
    /// Restore-only: at runtime the metadata is captured at connection-learn time.
    pub fn decorate_spatial_targets(&mut self, meta: &FxHashMap<NeuronId, (ChannelId, crate::types::DimensionId)>) {
        for neuron in self.neurons.values_mut() {
            neuron.decorate_spatial_targets(meta);
        }
    }

    /// Diagnostic: list the UNPAID spatial patterns hosted by this column's neurons — hypotheses
    /// whose deposited evidence has not yet covered their price.
    pub fn collect_unpaid_spatial_patterns(&self) -> Vec<NeuronId> {
        let mut unpaid = Vec::new();
        for neuron in self.neurons.values() {
            for (&pattern_id, entry) in neuron.get_spatial_routing_table() {
                if !entry.is_paid() { unpaid.push(pattern_id); }
            }
        }
        unpaid
    }

    /// Spatial correction install — for each op, add the new pattern as a child on the
    /// parent neuron with the d=0 context entries, register the resulting death frame, and emit
    /// ContextRefUpdates for each context-entry target so they know this parent now references them.
    /// Corrections are NOT activated this frame; the routing-table entry will fire on next frame's
    /// spatial sweep via the recognize_patterns path.
    pub fn install_spatial_corrections(&mut self, ops: Vec<SpatialInstallOp>, frame_number: FrameNumber) -> SpatialInstallResult {
        let mut deaths = Vec::new();
        let mut context_ref_updates: Vec<SpatialContextRefUpdate> = Vec::new();

        for op in ops {
            // Locate the parent. If it's not in this column, the routing was wrong upstream — skip.
            let parent = match self.neurons.get_mut(&op.parent_id) {
                Some(n) => n,
                None => continue,
            };

            // Add the child pattern to the parent's spatial routing table with its payment state.
            let death_frame = parent.add_spatial_pattern(op.pattern_id, &op.context_neuron_ids, frame_number, op.evidence, op.price);
            if let Some(df) = death_frame { deaths.push((op.pattern_id, df)); }

            // For each context-neuron target, emit a SpatialContextRefUpdate so the target's
            // spatial_context_refs gains an entry pointing back to the parent.
            // Bucketing by target happens later in thalamus (parallel to the temporal path),
            // which is why each update carries its own neuron_id.
            for &ctx_neuron_id in &op.context_neuron_ids {
                context_ref_updates.push(SpatialContextRefUpdate {
                    neuron_id: ctx_neuron_id,
                    parent_id: op.parent_id,
                });
            }
        }

        SpatialInstallResult { deaths, context_ref_updates }
    }

    /// Spatial counterpart of update_temporal_context_refs — dispatches SpatialContextRefUpdates to the
    /// target neurons owned by this column.
    pub fn update_spatial_context_refs(&mut self, update_batch: &[(NeuronId, Vec<SpatialContextRefUpdate>)]) {
        for (target_id, updates) in update_batch {
            if let Some(neuron) = self.neurons.get_mut(target_id) {
                neuron.apply_spatial_context_ref_updates(updates);
            }
        }
    }

    /// Op-1/Op-4: Construct new Neuron instances from specs and store them locally.
    /// Each spec carries everything needed to build the Neuron without reaching back
    /// to Thalamus: id, forget_rate, connections, and shared config is on the Column.
    pub fn create_neurons(&mut self, specs: &[NeuronCreateSpec]) {
        for spec in specs {
            let mut neuron = Neuron::new(
                spec.id,
                spec.forget_rate,
                self.group_threshold,
                self.group_mode,
                self.channel_actions.clone(),
                self.context_length,
                self.learning,
                self.match_info,
                self.error_info,
                self.trace_match,
                self.trace_error,
            );
            // pre-wire default action connections at neutral reward across all voting distances
            for distance in 1..self.context_length {
                for &default_id in self.channel_default_actions.values() {
                    neuron.create_connection(distance, default_id, 1.0, 0.0);
                }
            }

            if let Some(ref connections) = spec.connections {
                let mut target_dims: FxHashMap<NeuronId, (ChannelId, DimensionId)> = FxHashMap::default();
                for conn in connections {
                    // save the event connections the pattern was created to infer
                    neuron.create_connection(conn.distance, conn.to_neuron_id, conn.strength, conn.reward);
                    if let Some(dim_id) = conn.dim_id {
                        target_dims.insert(conn.to_neuron_id, (conn.channel_id, dim_id));
                    }

                    // for actions with negative rewards, save an alternative with neutral reward
                    if conn.reward < 0.0 {
                        if let Some(alt) = neuron.find_alternative_action(conn.distance, conn.channel_id, conn.to_neuron_id) {
                            neuron.create_connection(conn.distance, alt, 1.0, 0.0);
                        }
                    }
                }
                // Founding spatial (d=0) connections need their target's position recorded, the
                // same way ordinary spatial learning does, or aggregate_spatial_prediction's
                // per-position winner competition silently drops their votes.
                if !target_dims.is_empty() { neuron.decorate_spatial_targets(&target_dims); }
            }
            self.neurons.insert(neuron.id, neuron);
        }
    }

    /// Serialize owned Neurons into plain data for snapshotting and backup.
    /// Returns {id, neuron} entries. Thalamus decorates each entry with
    /// metadata (level, parentId, baseNeuron) from its central maps before
    /// handing the assembled snapshot to Backup.
    pub fn get_snapshot(&self) -> Vec<SnapshotEntry> {
        self.neurons.values()
            .map(|neuron| SnapshotEntry { id: neuron.id, neuron: neuron.serialize() })
            .collect()
    }

    /// Reconstruct Neuron instances from serialized snapshot data on load.
    /// Each spec carries a plain serialized neuron that Thalamus has routed
    /// to this column via the routing rule.
    pub fn restore_snapshot(&mut self, specs: &[crate::neuron::SerializedNeuron]) {
        for data in specs {
            self.load_neuron(data);
        }
    }

    /// Construct a single Neuron from its serialized data, restoring connections,
    /// routing table with context entries, contextRefs, and error stats.
    /// Stores the reconstructed Neuron in this column's neuron map.
    fn load_neuron(&mut self, data: &crate::neuron::SerializedNeuron) {
        let mut neuron = Neuron::new(
            data.id,
            data.pattern_forget_rate,
            self.group_threshold,
            self.group_mode,
            self.channel_actions.clone(),
            self.context_length,
            self.learning,
            self.match_info,
            self.error_info,
            self.trace_match,
            self.trace_error,
        );

        // load directed connections (distance → target neuron id with strength and reward)
        for conn in &data.connections {
            neuron.create_connection(conn.distance, conn.to_neuron_id, conn.strength, conn.reward);
        }

        // load child patterns into the routing table with their activation strengths and context entries.
        // The serialized `spatial` discriminator decides which side each child restores to — spatial
        // children carry d=0 context entries, temporal children carry their real distances.
        for child in &data.children {
            if child.spatial {
                neuron.add_spatial_child(child.pattern_id, child.activation_strength);
                if let Some(entry) = neuron.get_spatial_routing_table_mut().get_mut(&child.pattern_id) {
                    entry.last_activation_frame = child.last_activation_frame;
                    entry.evidence = child.evidence;
                    entry.price = child.price;
                }
                for ctx in &child.context {
                    neuron.add_spatial_context(child.pattern_id, ctx.neuron_id, ctx.strength);
                }
            } else {
                neuron.add_temporal_child(child.pattern_id, child.activation_strength);
                if let Some(entry) = neuron.get_temporal_routing_table_mut().get_mut(&child.pattern_id) {
                    entry.last_activation_frame = child.last_activation_frame;
                }
                for ctx in &child.context {
                    neuron.add_temporal_context(child.pattern_id, ctx.neuron_id, ctx.distance, ctx.strength);
                }
            }
        }

        // load bidirectional context references, each back to its proper side.
        for ctx_ref in &data.context_refs {
            if ctx_ref.spatial {
                neuron.add_spatial_context_ref(ctx_ref.parent_id);
            } else {
                for &distance in &ctx_ref.distances {
                    neuron.add_temporal_context_ref(ctx_ref.parent_id, distance);
                }
            }
        }

        // load the local background models: context counts for likelihood-ratio recognition,
        // inference counts for information-priced correction creation.
        neuron.restore_spatial_context_counts(data.context_frames, &data.context_counts);
        neuron.restore_spatial_inference_counts(data.inference_frames, &data.inference_counts);
        neuron.restore_pending_spatial_mints(&data.pending_mints);

        // load per-(neuron, age) Welford error stats.
        // Spatial serializes as age=0; temporal serializes at its real age (>= 1).
        for stat in &data.error_stats {
            if stat.age == 0 {
                neuron.load_spatial_error_stats(stat.n, stat.mean, stat.m2);
            } else {
                neuron.load_temporal_error_stats(stat.age, stat.n, stat.mean, stat.m2);
            }
        }

        self.neurons.insert(neuron.id, neuron);
    }

    /// Walk routing tables and compute death frames from current activation strengths.
    /// Each neuron computes death frames for its own temporal and spatial tables in a single call.
    /// Used on restore to rebuild the death ledger without mutating any neuron state.
    pub fn collect_death_frames(&self) -> Vec<DeathFrameEntry> {
        let mut entries = Vec::new();
        for neuron in self.neurons.values() {
            entries.extend(neuron.compute_death_frames());
        }
        entries
    }

    /// Materialize lazy decay into actual activation strengths and reset last_activation_frame to 0 for all children.
    /// Each neuron materializes and resets its own temporal and spatial routing tables in a single call.
    /// Returns {pattern_id, death_frame} entries so Thalamus can rebuild the death ledger.
    pub fn materialize_and_reset_neurons(&mut self, current_frame: FrameNumber) -> Vec<DeathFrameEntry> {
        let mut death_entries = Vec::new();
        for neuron in self.neurons.values_mut() {
            death_entries.extend(neuron.materialize_and_reset_children(current_frame));
        }
        death_entries
    }

    /// Clear all neurons. Used during reset before restore.
    pub fn clear(&mut self) {
        self.neurons.clear();
    }

    /// Brain.learn(): apply a batch of supervised voter→action wirings against owned voter neurons.
    /// Each entry wires (voter_id) → (action_id) at the given distance.
    /// The wire uses additive (strength += 1, reward += reward_arg) semantics.
    /// The distance is caller-supplied so the same primitive can serve different supervised paths.
    pub fn learn_action_connections(&mut self, wirings: &[(NeuronId, NeuronId, Reward)], distance: Distance) {
        for &(voter_id, action_id, reward) in wirings {
            if let Some(neuron) = self.neurons.get_mut(&voter_id) {
                neuron.strengthen_or_create_connection(distance, action_id, reward);
            }
        }
    }

    /// Read-only vote sweep over (voter_id, age) pairs owned by this column.
    /// Each pair represents one active non-suppressed voter at a specific age.
    /// Mirrors how `Neuron::generate_votes` walks ages inside process_frame.
    /// Calls `neuron.vote(age)` for each pair and tags the resulting Votes with their natural distance = age + 1.
    /// Used by Brain.learn() to collect the post-wire inference vote pool without re-running process_frame.
    /// Returns (voter_id, age, votes) triples; empty vote lists are filtered out.
    pub fn collect_votes_for_voter_ages(&self, voter_ages: &[(NeuronId, Distance)]) -> Vec<(NeuronId, Distance, Vec<Vote>)> {
        let mut out = Vec::new();
        for &(voter_id, age) in voter_ages {
            if let Some(neuron) = self.neurons.get(&voter_id) {
                let votes = neuron.vote(age);
                if !votes.is_empty() {
                    out.push((voter_id, age, votes));
                }
            }
        }
        out
    }

    /// Update shared action sets when a new channel is registered. Called by Thalamus
    /// after registerChannelSpec to keep column-local copies in sync.
    pub fn update_action_sets(
        &mut self,
        channel_actions: &FxHashMap<ChannelId, Vec<NeuronId>>,
        channel_default_actions: &FxHashMap<ChannelId, NeuronId>,
    ) {
        self.channel_actions = channel_actions.clone();
        self.channel_default_actions = channel_default_actions.clone();
    }
}

/// Internal result from delete_neuron.
struct DeleteNeuronResult {
    outbound_ops: Vec<DeleteOp>,
    deleted_id: NeuronId,
}

/// Spec for creating a neuron in a column.
#[derive(Debug, Clone)]
pub struct NeuronCreateSpec {
    pub id: NeuronId,
    pub forget_rate: f64,
    pub connections: Option<Vec<ConnectionSpec>>,
}

/// Connection specification for neuron creation.
#[derive(Debug, Clone)]
pub struct ConnectionSpec {
    pub distance: Distance,
    pub to_neuron_id: NeuronId,
    pub strength: Strength,
    pub reward: Reward,
    pub channel_id: ChannelId,
    /// Target's position, spatial (distance 0) connections only — `aggregate_spatial_prediction`'s
    /// per-position winner competition needs it to consider this connection's votes at all. `None`
    /// for temporal (distance > 0) connections, which don't use position-based competition.
    pub dim_id: Option<DimensionId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_column() -> Column {
        Column::new(
            FxHashMap::default(),
            FxHashMap::default(),
            2,
            0.5,
            GroupMode::Static,
            true,
            false, false, false, false,
        )
    }

    #[test]
    fn test_create_and_snapshot() {
        let mut col = make_column();
        col.create_neurons(&[
            NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None },
            NeuronCreateSpec { id: 2, forget_rate: 0.1, connections: None },
        ]);
        assert_eq!(col.neurons.len(), 2);
        let snapshot = col.get_snapshot();
        assert_eq!(snapshot.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut col = make_column();
        col.create_neurons(&[NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None }]);
        assert_eq!(col.neurons.len(), 1);
        col.clear();
        assert_eq!(col.neurons.len(), 0);
    }

    #[test]
    fn test_create_with_connections() {
        let mut col = make_column();
        col.create_neurons(&[
            NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None },
            NeuronCreateSpec {
                id: 2,
                forget_rate: 0.1,
                connections: Some(vec![
                    ConnectionSpec { distance: 1, to_neuron_id: 1, strength: 1.0, reward: 0.0, channel_id: 0, dim_id: None },
                ]),
            },
        ]);
        let neuron = col.neurons.get(&2).unwrap();
        assert!(neuron.has_connection(1, 1));
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let mut col = make_column();
        col.create_neurons(&[
            NeuronCreateSpec { id: 1, forget_rate: 0.0, connections: None },
            NeuronCreateSpec {
                id: 2,
                forget_rate: 0.1,
                connections: Some(vec![
                    ConnectionSpec { distance: 1, to_neuron_id: 1, strength: 1.0, reward: 0.5, channel_id: 0, dim_id: None },
                ]),
            },
        ]);

        let snapshot = col.get_snapshot();
        let serialized: Vec<_> = snapshot.iter().map(|e| e.neuron.clone()).collect();

        let mut col2 = make_column();
        col2.restore_snapshot(&serialized);
        assert_eq!(col2.neurons.len(), 2);
        assert!(col2.neurons.get(&2).unwrap().has_connection(1, 1));
    }
}
