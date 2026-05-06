/**
 * Column — owns a partition of Neuron instances and exposes batch operations
 * on them. Becomes a worker thread in Phase 5; in single-process JS every
 * method is a synchronous local call.
 *
 * Action sets (channelActions, actionIds, channelDefaultActions) are passed
 * in at init time so per-frame calls never reach back to Thalamus for them.
 * `this.neurons` is the sole storage for owned Neurons.
 */
export class Column {

	constructor(channelActions, actionIds, channelDefaultActions) {
		this.channelActions = channelActions;
		this.actionIds = actionIds;
		this.channelDefaultActions = channelDefaultActions;
		this.neurons = new Map(); // id -> Neuron
	}

	/**
	 * Op-4 down-trip body. Calls neuron.processFrame on every task and returns
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
	 * Apply contextRef updates to owned Neurons. One call per target neuron
	 * the batch carries an update for; foreign updates are routed by the caller.
	 */
	updateContextRefs(updates) {
		throw new Error('Column.updateContextRefs not yet implemented');
	}

	/**
	 * Construct new Neuron instances locally from specs and store them in the
	 * owned neurons map. Used for both freshly observed sensory points and
	 * error-correction patterns.
	 */
	createNewNeurons(specs) {
		throw new Error('Column.createNewNeurons not yet implemented');
	}

	/**
	 * Apply a batch of delete operations against owned Neurons. Returns the
	 * outbound ops produced (to be routed by the caller to other columns) and
	 * the ids whose canDeleteChild flipped this round (to be re-queued as
	 * DeleteSelf next pulse).
	 */
	deleteNeurons(opBatch) {
		throw new Error('Column.deleteNeurons not yet implemented');
	}

	/**
	 * Serialize every owned Neuron's persistent state for snapshotting.
	 * Caller assembles the full snapshot by combining this output with the
	 * central metadata maps held by Thalamus.
	 */
	dumpAll() {
		throw new Error('Column.dumpAll not yet implemented');
	}

	/**
	 * Construct owned Neuron instances from serialized state on load. Caller
	 * has already routed each spec to its owning column via routeNeuron.
	 */
	restoreNeurons(specs) {
		throw new Error('Column.restoreNeurons not yet implemented');
	}
}
