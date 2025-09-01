/**
 * channels handle the interaction between the brain and a device. a device can be anything that has a consistent input/output mechanism.
 * examples of devices:
 * - eyes (input: visual data, output: eye movements - saccades)
 * - tounge (input: taste, output: tongue movements)
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
	}

	/**
	 * Initialize channel dimensions in brain
	 */
	async initialize(brain) {
		this.brain = brain;
		const dimensions = this.getDimensions();

		for (const dimName of dimensions) {
			await brain.conn.query(
				'INSERT IGNORE INTO dimensions (name) VALUES (?)',
				[dimName]
			);
		}

		await brain.loadDimensions();
		console.log(`Initialized ${dimensions.length} dimensions for ${this.name}`);
	}

	/**
	 * Build input frame for this channel - override in subclasses
	 */
	async buildFrame(data, frameNumber) {
		throw new Error('Channel must implement buildFrame() method');
	}

	/**
	 * Execute outputs based on brain predictions - override in subclasses
	 * This method should execute actions and update channel state
	 */
	async executeOutputs(predictions, frameNumber) {
		throw new Error('Channel must implement executeOutputs() method');
	}

	/**
	 * Get all dimension names for this channel (combines input, output, and feedback dimensions)
	 */
	getDimensions() {
		return [...this.getInputDimensions(), ...this.getOutputDimensions(), ...this.getFeedbackDimensions() ];
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
	 * Get feedback dimension names - override in subclasses
	 */
	getFeedbackDimensions() {
		throw new Error('Channel must implement getFeedbackDimensions() method');
	}

	/**
	 * Get frame data for processing - implemented in base class
	 * Orchestrates inputs, exploration, and feedback
	 * Returns array of frame objects: [{ [dim-name]: <value> }]
	 */
	async getFramePoints() {
		const frameData = [];

		// Get input neurons from child class
		const inputs = await this.getFrameInputs();
		if (inputs && inputs.length > 0) frameData.push(...inputs);

		// Exploration is handled by the brain when globally inactive
		// Brain will call getValidExplorationActions() when needed

		// Get feedback neurons from child class
		const feedback = await this.getFeedbackNeurons();
		if (feedback && feedback.length > 0) frameData.push(...feedback);

		return frameData;
	}

	/**
	 * Get frame input data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameInputs() {
		throw new Error('Channel must implement getFrameInputs() method');
	}

	/**
	 * Get feedback neurons based on previous actions and current state - override in subclasses
	 * Returns array of feedback neuron objects: [{ [feedback-dim]: value }]
	 */
	async getFeedbackNeurons() {
		// Default implementation returns empty array - will be overridden by child classes
		return [];
	}

	/**
	 * Get valid exploration actions based on current channel state
	 * Child classes should override this to provide context-aware exploration
	 * Examples:
	 * - Stock: Can't sell if not owned, can't buy if already owned
	 * - Arm: Can't move beyond joint limits  
	 * - Eyes: Can't saccade outside visual field
	 */
	getValidExplorationActions() {
		// Child classes must implement this with state-aware logic
		throw new Error('Channel must implement getValidExplorationActions() method');
	}

	/**
	 * Generate random exploration actions (legacy method - kept for compatibility)
	 */
	generateExploration(frameNumber, possibleActions) {
		const framesSinceOutput = frameNumber - this.lastOutputFrame;

		if (framesSinceOutput >= this.config.inactivityThreshold &&
			Math.random() < this.explorationRate) {

			const randomAction = possibleActions[
				Math.floor(Math.random() * possibleActions.length)
				];

			console.log(`${this.name}: Random exploration - ${randomAction}`);
			return [{ [randomAction]: 1.0 }];
		}

		return [];
	}

	/**
	 * create dimensions in the brain if not created before
	 */
	async init() {
		// Load dimension IDs from the brain for this channel's dimensions
		const dimIdMap = await this.brain.getDimensionIdMap();
		this.dimensions.forEach(dimName => {
			if (dimIdMap[dimName] === undefined) {
				console.warn(`Dimension '${dimName}' for channel not found in database.`);
				// In a real system, you might create it or throw an error.
			}
			this.dimensionNameToId[dimName] = dimIdMap[dimName];
		});
		console.log(`Channel initialized with dimensions: ${JSON.stringify(this.dimensionNameToId)}`);
	}

	/**
	 * Translates raw input data (e.g., pixel {x, y, r, g, b}) into
	 * an array of coordinate objects for activation.
	 * @param {Object} inputData - Raw input, e.g., { x: 10, y: 20, r: 255, g: 0, b: 0 }
	 * @returns {Array<Object>} - Array of { dimension_id, value } for the input.
	 */
	processInput(inputData) {
		const inputCoords = [];
		for (const dimName in inputData) {
			if (this.dimensionNameToId[dimName] !== undefined) {
				inputCoords.push({
					dimension_id: this.dimensionNameToId[dimName],
					value: inputData[dimName]
				});
			} else {
				console.warn(`Warning: Input dimension '${dimName}' not found in known dimensions.`);
			}
		}
		return inputCoords;
	}

	/**
	 * Abstract method: Processes raw input data into discretized coordinate pairs.
	 */
	async processRawInput(rawInput) {
		throw new Error("processRawInput must be implemented by child classes.");
	}
}