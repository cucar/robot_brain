import { Column } from './column.js';

/**
 * Region — wraps Column[C]. Becomes an MPI rank in Phase 5; in single-process
 * JS it's a pure router/aggregator. Never exposes single-neuron access.
 */
export class Region {

	constructor(C, channelActions, actionIds, channelDefaultActions, mergeThreshold, errorMode, errorThreshold) {
		this.C = C;
		this.columns = [];
		for (let c = 0; c < C; c++)
			this.columns.push(new Column(channelActions, actionIds, channelDefaultActions, mergeThreshold, errorMode, errorThreshold));
	}

	/**
	 * Pure deterministic column-routing function.
	 * @param {number} neuronId
	 * @returns {number} columnIdx (region-local, 0..C-1)
	 */
	routeNeuron(neuronId) {
		return neuronId % this.C;
	}

	/**
	 * Bucket a flat batch by owning column using a per-method routing key extractor.
	 * Returns an array indexed by columnIdx, each entry the sub-list owned by that column.
	 */
	bucketByColumn(batch, key) {
		const buckets = Array.from({ length: this.C }, () => []);
		for (const item of batch) buckets[this.routeNeuron(item[key])].push(item);
		return buckets;
	}

	/**
	 * Op-3 down-trip. Bucket tasks by owning column, fan out, concatenate
	 * results in column-index order (stable regardless of thread scheduling).
	 * Broadcast params are passed by reference and not mutated inside the call.
	 */
	processLevel(tasks, memoryDepth, levelContext, newErrorPatternIds, newActiveNeurons, frameNumber) {
		const tasksByColumn = this.bucketByColumn(tasks, 'neuronId');
		const results = [];
		for (let c = 0; c < this.C; c++) {
			const colResults = this.columns[c].processLevel(
				tasksByColumn[c], memoryDepth, levelContext, newErrorPatternIds, newActiveNeurons, frameNumber
			);
			for (const r of colResults) results.push(r);
		}
		return results;
	}

	/**
	 * Op-5 (deferred): Apply contextRef updates against owned Neurons. Updates are routed by
	 * update.neuronId (the target neuron whose contextRefs change).
	 * No return — fire-and-forget, batched after the level loop.
	 */
	updateContextRefs(updates) {
		const updatesByColumn = this.bucketByColumn(updates, 'neuronId');
		for (let c = 0; c < this.C; c++)
			this.columns[c].updateContextRefs(updatesByColumn[c]);
	}

	/**
	 * Op-1/Op-4: Construct new Neurons in their owning columns. Specs are routed by spec.id
	 * (the freshly allocated neuron id).
	 */
	createNeurons(specs) {
		const specsByColumn = this.bucketByColumn(specs, 'id');
		const created = [];
		for (let c = 0; c < this.C; c++)
			for (const neuron of this.columns[c].createNeurons(specsByColumn[c]))
				created.push(neuron);
		return created;
	}

	/**
	 * Op-2: Apply a batch of delete operations in the columns owned
	 */
	deleteNeurons(opBatch, currentFrame) {
		const opsByColumn = this.bucketByColumn(opBatch, 'targetId');
		const outboundOps = [];
		const deletedIds = [];
		const newlyDeletableIds = [];

		for (let c = 0; c < this.C; c++) {
			const result = this.columns[c].deleteNeurons(opsByColumn[c], currentFrame);
			for (const op of result.outboundOps) outboundOps.push(op);
			for (const id of result.deletedIds) deletedIds.push(id);
			for (const id of result.newlyDeletableIds) newlyDeletableIds.push(id);
		}

		return { outboundOps, deletedIds, newlyDeletableIds };
	}

}
