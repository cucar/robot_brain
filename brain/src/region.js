import { Column } from './column.js';

/**
 * Region — wraps an array of Columns. Eventually represents an MPI rank;
 * in single-process JS it's a pure router/aggregator with no state of its own
 * beyond the column list.
 *
 * Per-frame methods take a flat batch (already filtered to ids this region owns)
 * and bucket internally by column. Each method knows which field on its batch
 * elements to route by — the shape of each method's inputs is its own contract.
 */
export class Region {

	constructor(C, channelActions, actionIds, channelDefaultActions) {
		this.C = C;
		this.columns = []; // index = columnIdx
		for (let c = 0; c < C; c++)
			this.columns.push(new Column(channelActions, actionIds, channelDefaultActions));
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
	 * Process one frame for every (neuron, age) task in this region's batch.
	 * Tasks are routed by task.neuronId.
	 */
	processLevel(tasks, sensoryNeurons, rewards, levelContext, newErrorPatternIds, frameNumber) {
		const tasksByColumn = this.bucketByColumn(tasks, 'neuronId');
		const results = [];
		for (let c = 0; c < this.C; c++) {
			const colResults = this.columns[c].processLevel(
				tasksByColumn[c], sensoryNeurons, rewards, levelContext, newErrorPatternIds, frameNumber
			);
			for (const r of colResults) results.push(r);
		}
		return results;
	}

	/**
	 * Apply contextRef updates against owned Neurons. Updates are routed by
	 * update.neuronId (the target neuron whose contextRefs change).
	 * No return — fire-and-forget within the level barrier.
	 */
	updateContextRefs(updates) {
		const updatesByColumn = this.bucketByColumn(updates, 'neuronId');
		for (let c = 0; c < this.C; c++)
			this.columns[c].updateContextRefs(updatesByColumn[c]);
	}

	/**
	 * Construct new Neurons in their owning columns. Specs are routed by spec.id
	 * (the freshly allocated neuron id).
	 */
	createNewNeurons(specs) {
		const specsByColumn = this.bucketByColumn(specs, 'id');
		for (let c = 0; c < this.C; c++)
			this.columns[c].createNewNeurons(specsByColumn[c]);
	}

	/**
	 * Apply a batch of delete operations. The receiver of an op depends on its type
	 * (DeleteSelf → neuronId, RemoveContextEntry → parentId, RemoveContextRef →
	 * ctxNeuronId, etc.), so the per-op-type bucketer lands when the op vocabulary
	 * is defined alongside the cleanup pulse loop.
	 */
	deleteNeurons(opBatch) {
		throw new Error('Region.deleteNeurons not yet implemented');
	}
}
