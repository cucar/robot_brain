/**
 * channels handle the interaction between the brain and a device. a device can be anything that has a consistent input/output mechanism.
 * examples of devices:
 * - eyes (input: visual data, output: eye movements - saccades)
 * - tongue (input: taste, output: tongue movements)
 * - ears (input: audio, output: ear movements - auriculomotor)
 * - arm/leg (input: touch, output: muscles)
 * - stocks (input: prices, output: buy/sell)
 * - text (input: characters, output: characters)
 * channels are like the adapters for the devices, similar to device drivers.
 * they do that by translating the streaming inputs and returning streaming outputs translated from brain action neurons.
 * this is the base class for all such channels. child classes implement different mediums like vision, audio, etc.
 */
export default class Channel {

	constructor(name) {
		this.name = name; // just for descriptions in debugging
		this.frameNumber = 0; // frame counter for channel-specific operations
		this.debug = true; // controls verbosity of channel output
		this.debug2 = false; // more detailed, verbose debug mode
		this.inferredActions = []; // actions selected by resolveConflicts
	}

	/**
	 * Initialize channel - override in subclasses if needed
	 * Called once during brain initialization
	 */
	async initialize() {
		// Default implementation does nothing
		// Child classes can override for channel-specific initialization
	}

	/**
	 * Execute outputs based on brain predictions - override in subclasses
	 * This method should execute actions, update channel state, and return final frame points
	 * @param {Array} outputs - Frame outputs from getFrameOutputs()
	 * @returns {void}
	 */
	async executeOutputs(outputs) {
		throw new Error('Channel must implement executeOutputs() method');
	}

	/**
	 * Get event dimension names - override in subclasses
	 */
	getEventDimensions() {
		throw new Error('Channel must implement getEventDimensions() method');
	}

	/**
	 * Get state dimension names - override in subclasses - default implementation returns empty array
	 */
	getStateDimensions() {
		return [];
	}

	/**
	 * Get output dimension names - override in subclasses  
	 */
	getOutputDimensions() {
		throw new Error('Channel must implement getOutputDimensions() method');
	}

	/**
	 * Get frame events data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameEvents() {
		throw new Error('Channel must implement getFrameEvents() method');
	}

	/**
	 * Get frame state data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameState() {
		throw new Error('Channel must implement getFrameState() method');
	}

	/**
	 * Get inferred actions scheduled for current frame
	 * Returns actions selected by resolveConflicts in previous frame
	 */
	async getFrameOutputs() {
		if (this.inferredActions.length === 0) return [];
		const actions = this.inferredActions;
		this.inferredActions = [];
		return actions;
	}

	/**
	 * Get feedback based on previous actions and current state - override in subclasses
	 * Returns reward factor: number (1.0 = neutral, >1.0 = positive, <1.0 = negative)
	 * This factor will be multiplied with existing neuron reward factors
	 */
	async getRewards() {
		return 1.0; // Neutral feedback by default
	}

	/**
	 * Returns an exploration action based on current channel state
	 * Child classes should override this to provide context-aware exploration
	 * Examples:
	 * - Stock: Can't sell if not owned
	 * - Arm: Can't move beyond joint limits
	 * - Eyes: Can't saccade outside visual field
	 */
	getExplorationAction() {
		// Child classes must implement this with state-aware logic
		throw new Error('Channel must implement getExplorationAction() method');
	}

	/**
	 * Resolve conflicts between multiple inferred base neurons for this channel
	 * Base implementation calls getResolvedInference and stores result automatically
	 * Child classes override getResolvedInference, not this method
	 * @param {Array} inference - Array of inference objects with structure: [{ neuron_id, coordinates: {dim1: val1, dim2: val2}, strength: number }]
	 * @returns {Array} - Array of selected predictions to execute (can be 0, 1, or many)
	 */
	resolveConflicts(inference) {

		// ask the child class to resolve conflicts and return the selected neurons
		const resolvedInference = this.getResolvedInference(inference);

		// store only output neurons for next frame execution
		const outputDims = new Set(this.getOutputDimensions());
		const inferredOutput = resolvedInference.filter(pred => Object.keys(pred.coordinates).some(dim => outputDims.has(dim)));
		this.inferredActions = inferredOutput.map(pred => pred.coordinates);

		return resolvedInference;
	}

	/**
	 * returns event and action inferences
	 * @param {Array} inferences - all inferences
	 * @returns {Object} - { events, actions }
	 */
	getEventsAndActions(inferences) {
		const eventDims = new Set(this.getEventDimensions());
		const outputDims = new Set(this.getOutputDimensions());
		const events = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => eventDims.has(dim)));
		const actions = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => outputDims.has(dim)));
		return { events, actions };
	}

	/**
	 * Get resolved inference for the channel
	 * Resolves conflicts for both event predictions and action inferences
	 * Conflict resolution logic is implemented in channel subclasses
	 * State neurons exist only to help guide actions, so they are filtered out
	 * @param {Array} inferences - all inferred neurons for this channel
	 * @returns {Array} - resolved neurons according to channel specific logic
	 */
	getResolvedInference(inferences) {

		// if there are no inferences, nothing to resolve
		if (!inferences || inferences.length === 0) return [];

		// Separate into input predictions and output inferences
		const { events, actions } = this.getEventsAndActions(inferences);

		// Resolve event predictions: select strongest for each input dimension
		const resolvedEvents = this.resolveEventPredictions(events);

		// Resolve action inferences: select strongest action
		const resolvedActions = this.resolveActionInferences(actions);

		// Combine and return
		const resolved = [...resolvedEvents, ...resolvedActions];
		if (this.debug) console.log('getResolvedInference', resolved);
		if (this.debug) this.logResolution(inferences.length, events.length, actions.length, resolved.length);
		return resolved;
	}

	/**
	 * Log resolution results for debugging
	 */
	logResolution(totalCount, inputCount, outputCount, resolvedCount) {
		console.log(`${this.name}: Resolved ${totalCount} inferences (${inputCount} inputs, ${outputCount} outputs) → ${resolvedCount} selected`);
	}

	/**
	 * Get resolved event predictions for this channel - override in subclasses
	 * @param {Array} events - predictions for event dimensions
	 * @returns {Array} - strongest prediction per event dimension
	 */
	resolveEventPredictions(events) {
		throw new Error('Channel must implement resolveEventPredictions(events) method');
	}

	/**
	 * Resolve action inferences: select strongest for each output dimension
	 * @param {Array} actions - inferences for output dimensions
	 * @returns {Array} - strongest inference per output dimension
	 */
	resolveActionInferences(actions) {
		throw new Error('Channel must implement resolveEventPredictions(actions) method');
	}

	/**
	 * Get output performance metrics - override in subclasses
	 * Returns channel-specific performance data (e.g., profit/loss, win rate, score, etc.)
	 * @returns {Object|null} - { value: number, label: string, format: string } or null
	 * Example: { value: 5.23, label: 'AAPL', format: 'currency' }
	 */
	getOutputPerformanceMetrics() {
		return null; // Default: no output performance tracking
	}
}