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
		this.debug = false; // controls verbosity of channel output
		this.debug2 = false; // controls detailed verbosity of channel output
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
	 * This method should execute actions and update channel state
	 */
	async executeOutputs() {
		throw new Error('Channel must implement executeOutputs() method');
	}

	/**
	 * Get input dimension names - override in subclasses
	 */
	getInputDimensions() {
		throw new Error('Channel must implement getInputDimensions() method');
	}

	/**
	 * Get output dimension names - override in subclasses  
	 */
	getOutputDimensions() {
		throw new Error('Channel must implement getOutputDimensions() method');
	}

	/**
	 * Get frame input data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameInputs() {
		throw new Error('Channel must implement getFrameInputs() method');
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
	async getFeedback() {
		return 1.0; // Neutral feedback by default
	}

	/**
	 * Get valid exploration actions based on current channel state
	 * Child classes should override this to provide context-aware exploration
	 * Examples:
	 * - Stock: Can't sell if not owned
	 * - Arm: Can't move beyond joint limits
	 * - Eyes: Can't saccade outside visual field
	 */
	getValidExplorationActions() {
		// Child classes must implement this with state-aware logic
		throw new Error('Channel must implement getValidExplorationActions() method');
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
	 * Separate inferences into input predictions and output inferences
	 * @param {Array} inferences - all inferences
	 * @returns {Object} - { inputPredictions, outputInferences }
	 */
	separateInputsAndOutputs(inferences) {
		const inputDims = new Set(this.getInputDimensions());
		const outputDims = new Set(this.getOutputDimensions());
		const inputPredictions = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => inputDims.has(dim)));
		const outputInferences = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => outputDims.has(dim)));
		return { inputPredictions, outputInferences };
	}

	/**
	 * Get resolved inference for this channel - override in subclasses
	 * Child classes implement their own conflict resolution logic here
	 * Examples:
	 * - Stock channel: Returns array with 1 prediction (can't buy and sell simultaneously)
	 * - Vision channel: Returns array with multiple predictions (can detect multiple objects)
	 */
	getResolvedInference() {
		throw new Error('Channel must implement getResolvedInference() method');
	}
}