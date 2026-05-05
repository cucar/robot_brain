/**
 * Column — owns a partition of Neuron instances and exposes batch operations
 * on them. Eventually becomes a worker thread; in single-process JS every
 * method is a synchronous local call.
 *
 * The constructor receives the read-only action sets needed by Neuron
 * internals (channel→action ids, the flat action id set) at init time so
 * per-frame calls never reach back to Thalamus for them.
 */
export class Column {

	constructor(channelActions, actionIds) {
		this.channelActions = channelActions;
		this.actionIds = actionIds;
		this.neurons = new Map(); // id -> Neuron
	}

	/**
	 * Process one frame for every (neuron, age) task in this batch.
	 * Each task is a self-contained payload describing one owned neuron's
	 * per-age work. Returns the per-neuron processFrame results.
	 */
	processLevel(tasks, sensoryNeurons, rewards, levelContext, newErrorPatternIds, frameNumber) {
		throw new Error('Column.processLevel not yet implemented');
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
