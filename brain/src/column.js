import { Neuron } from './neuron.js';

/**
 * Column — owns a partition of Neuron instances and exposes batch operations
 * on them. Becomes a worker thread in Phase 5; in single-process JS every
 * method is a synchronous local call.
 *
 * Action sets are passed at init time so per-frame calls never reach back
 * to Thalamus. `this.neurons` is the sole storage for owned Neurons.
 */
export class Column {

	constructor(channelActions, actionIds, channelDefaultActions, mergeThreshold, errorMode, errorThreshold) {
		this.channelActions = channelActions;
		this.actionIds = actionIds;
		this.channelDefaultActions = channelDefaultActions;
		this.mergeThreshold = mergeThreshold;
		this.errorMode = errorMode;
		this.errorThreshold = errorThreshold;
		this.neurons = new Map(); // id -> Neuron
	}

	/**
	 * Op-3 down-trip body. Calls neuron.processFrame on every task and returns
	 * results parentId-tagged in task order.
	 */
	processLevel(tasks, memoryDepth, levelContext, newErrorPatternIds, newActiveNeurons, frameNumber) {
		const results = [];
		for (const { neuronId, ageStates, corrections, errorFeedback } of tasks) {
			const result = this.neurons.get(neuronId).processFrame(
				ageStates, memoryDepth, levelContext, newErrorPatternIds,
				newActiveNeurons, frameNumber, corrections, errorFeedback
			);
			results.push({ parentId: neuronId, ...result });
		}
		return results;
	}

	/**
	 * Process a batch of delete operations against owned neurons.
	 * Returns outbound operations for other columns, deleted neuron ids, and neuron ids that just became deletable to cascade.
	 */
	deleteNeurons(opBatch, currentFrame) {
		const outboundOps = [];
		const deletedIds = [];
		const newlyDeletableIds = [];

		// loop over requested operations and execute them on the neurons owned
		for (const op of opBatch) {
			switch (op.type) {

				// destroy a dying neuron: walk its state, emit cleanup ops, remove it
				case 'DeleteNeuron': {
					const result = this.deleteNeuron(op);
					if (!result) break; // neuron already destroyed by an earlier op in this cascade
					for (const outboundOp of result.outboundOps) outboundOps.push(outboundOp);
					deletedIds.push(result.deletedId);
					break;
				}

				// remove a dead child pattern from its parent's routing table
				case 'RemovePattern':
					for (const outboundOp of this.removePattern(op)) outboundOps.push(outboundOp);
					break;

				// scrub a dead neuron from a parent's children's context entries
				case 'PurgeContextNeuron':
					this.purgeContextNeuron(op, currentFrame, newlyDeletableIds);
					break;

				// drop a stale contextRef on a context neuron
				case 'RemoveContextRef':
					this.removeContextRef(op);
					break;
			}
		}

		return { outboundOps, deletedIds, newlyDeletableIds };
	}

	/**
	 * Destroy a dying neuron. Walk its state to emit cleanup ops, then remove it.
	 */
	deleteNeuron(op) {
		const neuron = this.neurons.get(op.targetId);
		if (!neuron) return null;

		const outboundOps = [];

		// tell each parent that referenced this neuron in its children's contexts to scrub it
		for (const [referencingParentId, distances] of neuron.contextRefs)
			outboundOps.push({ type: 'PurgeContextNeuron', targetId: referencingParentId, dyingNeuronId: neuron.id, distances });

		// orphan each child: clean up context refs, then queue child for deletion
		for (const [childPatternId, tableEntry] of neuron.routingTable) {
			for (const entry of tableEntry.context.getEntries()) {
				const isOrphaned = neuron.removeContextIndex(entry.neuronId, entry.distance, childPatternId);
				if (isOrphaned)
					outboundOps.push({ type: 'RemoveContextRef', targetId: entry.neuronId, parentId: neuron.id, distance: entry.distance });
			}
			outboundOps.push({ type: 'DeleteNeuron', targetId: childPatternId, parentId: neuron.id });
		}

		// tell parent to remove this pattern from its routing table
		outboundOps.push({ type: 'RemovePattern', targetId: op.parentId, patternId: neuron.id });

		// free neuron memory
		this.neurons.delete(neuron.id);
		neuron.routingTable = null;
		neuron.contextIndex = null;
		neuron.contextRefs = null;
		neuron.connections = null;

		// return
		return { outboundOps, deletedId: neuron.id };
	}

	/**
	 * Remove a dead child pattern from a parent's routing table and context entries.
	 * Returns RemoveContextRef ops for orphaned context references.
	 */
	removePattern(op) {
		const parent = this.neurons.get(op.targetId);
		if (!parent) return []; // parent already destroyed in this cascade

		const entry = parent.routingTable.get(op.patternId);
		if (!entry) return []; // pattern already removed by parent's own DeleteNeuron cleanup

		const outboundOps = [];
		for (const ctxEntry of entry.context.getEntries()) {
			// same-pulse PurgeContextNeuron may have already removed this entry
			if (!entry.context.hasKey(ctxEntry.neuronId, ctxEntry.distance)) continue;
			const isOrphaned = parent.removeContext(op.patternId, ctxEntry.neuronId, ctxEntry.distance);
			if (isOrphaned)
				outboundOps.push({ type: 'RemoveContextRef', targetId: ctxEntry.neuronId, parentId: parent.id, distance: ctxEntry.distance });
		}

		parent.routingTable.delete(op.patternId);
		return outboundOps;
	}

	/**
	 * Scrub a dead neuron from a parent's children's context entries.
	 * Affected children whose activation strength decayed to zero become cascade candidates.
	 */
	purgeContextNeuron(op, currentFrame, newlyDeletableIds) {
		const parent = this.neurons.get(op.targetId);
		if (!parent) return;

		// same-pulse RemovePattern may have already cleaned some distances
		const distMap = parent.contextIndex.get(op.dyingNeuronId);
		if (!distMap) return;
		const remainingDistances = new Set();
		for (const d of op.distances)
			if (distMap.has(d)) remainingDistances.add(d);
		if (remainingDistances.size === 0) return;

		const affectedPatterns = parent.removeContextNeuron(op.dyingNeuronId, remainingDistances);
		for (const patternId of affectedPatterns)
			if (parent.canDeleteChild(patternId, currentFrame))
				newlyDeletableIds.push(patternId);
	}

	/**
	 * Drop a single contextRef entry on a context neuron.
	 */
	removeContextRef(op) {
		const neuron = this.neurons.get(op.targetId);
		if (!neuron) return;

		// same-pulse op may have already removed this ref
		const distances = neuron.contextRefs.get(op.parentId);
		if (!distances || !distances.has(op.distance)) return;

		neuron.removeContextRef(op.parentId, op.distance);
	}

	/**
	 * Op-5 (deferred): Apply contextRef updates to owned neurons. Each entry carries the
	 * target neuronId and a batch of {type, parentId, distance} updates for it.
	 */
	updateContextRefs(updateBatch) {
		for (const { neuronId, updates } of updateBatch)
			this.neurons.get(neuronId).applyContextRefUpdates(updates);
	}

	/**
	 * Op-1/Op-4: Construct new Neuron instances from specs and store them locally.
	 * Each spec carries everything needed to build the Neuron without reaching back
	 * to Thalamus: id, forgetRate, connections, and shared config is on the Column.
	 */
	createNeurons(specs) {
		const created = [];
		for (const spec of specs) {
			const neuron = new Neuron(
				spec.id, spec.forgetRate, this.mergeThreshold, this.errorMode,
				this.errorThreshold, this.channelActions, this.actionIds
			);
			if (spec.connections) neuron.initializeConnections(spec.connections);
			this.neurons.set(neuron.id, neuron);
			created.push(neuron);
		}
		return created;
	}

	/**
	 * Serialize owned Neurons into plain data for snapshotting and backup.
	 * Returns {id, neuron} entries. Thalamus decorates each entry with
	 * metadata (level, parentId, baseNeuron) from its central maps before
	 * handing the assembled snapshot to Backup.
	 */
	getSnapshot() {
		const entries = [];
		for (const neuron of this.neurons.values())
			entries.push({ id: neuron.id, neuron: neuron.serialize() });
		return entries;
	}

	/**
	 * Reconstruct Neuron instances from serialized snapshot data on load.
	 * Each spec carries a plain {neuron} object (from Neuron.serialize) that
	 * Thalamus has routed to this column via the routing rule.
	 */
	restoreSnapshot(specs) {
		for (const spec of specs) this.loadNeuron(spec.neuron);
	}

	/**
	 * Construct a single Neuron from its serialized data, restoring connections,
	 * routing table with context entries, contextRefs, and error stats.
	 * Stores the reconstructed Neuron in this column's neuron map.
	 */
	loadNeuron(data) {
		const neuron = new Neuron(
			data.id, data.patternForgetRate, this.mergeThreshold,
			this.errorMode, this.errorThreshold, this.channelActions, this.actionIds
		);
		this.loadConnections(neuron, data.connections);
		this.loadChildren(neuron, data.children);
		this.loadContextRefs(neuron, data.contextRefs);
		this.loadErrorStats(neuron, data.errorStats);
		this.neurons.set(neuron.id, neuron);
	}

	/**
	 * Load directed connections (distance → target neuron id with strength and reward).
	 */
	loadConnections(neuron, connections) {
		for (const conn of connections)
			neuron.createConnection(conn.distance, conn.toNeuronId, conn.strength, conn.reward);
	}

	/**
	 * Load child patterns into the routing table with their activation strengths and context entries.
	 */
	loadChildren(neuron, children) {
		for (const child of children) {
			neuron.addChild(child.patternId, child.activationStrength);
			const entry = neuron.routingTable.get(child.patternId);
			entry.lastActivationFrame = child.lastActivationFrame;
			for (const ctx of child.context)
				neuron.addContext(child.patternId, ctx.neuronId, ctx.distance, ctx.strength);
		}
	}

	/**
	 * Load bidirectional context references that track which parents reference this neuron.
	 */
	loadContextRefs(neuron, contextRefs) {
		for (const ref of contextRefs)
			for (const distance of ref.distances)
				neuron.addContextRef(ref.parentId, distance);
	}

	/**
	 * Load per-(neuron, age) Welford error stats for dynamic error correction modes.
	 */
	loadErrorStats(neuron, errorStats) {
		for (const stat of errorStats)
			neuron.loadErrorStats(stat.age, stat.n, stat.mean, stat.M2);
	}

	/**
	 * Walk routing tables and compute death frames from current activation
	 * strengths. Used on restore to rebuild the death ledger without mutating
	 * any neuron state.
	 */
	collectDeathFrames() {
		const entries = [];
		for (const neuron of this.neurons.values())
			for (const [patternId, ctx] of neuron.routingTable)
				entries.push({ patternId, deathFrame: Math.ceil(ctx.activationStrength / neuron.patternForgetRate) });
		return entries;
	}

	/**
	 * Materialize lazy decay into actual activation strengths and reset
	 * lastActivationFrame to 0 for all children. Returns {patternId, deathFrame}
	 * entries so Thalamus can rebuild the death ledger.
	 */
	materializeAndResetNeurons(currentFrame) {
		const deathEntries = [];
		for (const neuron of this.neurons.values()) {
			for (const [patternId, ctx] of neuron.routingTable) {
				neuron.materializeChildStrength(patternId, currentFrame);
				ctx.lastActivationFrame = 0;
				deathEntries.push({ patternId, deathFrame: Math.ceil(ctx.activationStrength / neuron.patternForgetRate) });
			}
		}
		return deathEntries;
	}
}
