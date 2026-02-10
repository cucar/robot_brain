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

	static nextId = 1; // Start at 1 to match typical DB conventions

	constructor(name, debug, id = null, dimensions = null) {
		this.id = id !== null ? id : Channel.nextId++;
		this.name = name; // just for descriptions in debugging
		this.frameNumber = 0; // frame counter for channel-specific operations
		this.debug = debug; // controls verbosity of channel output

		// Update nextId if we're loading a channel with a specific ID
		if (id !== null && id >= Channel.nextId) Channel.nextId = id + 1;
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
	 * Invalid actions should be filtered during conflict resolution, so only valid actions should be received
	 */
	async executeOutputs() {
		throw new Error('Channel must implement executeOutputs() method');
	}

	/**
	 * Get event dimension names - override in subclasses
	 */
	getEventDimensions() {
		throw new Error('Channel must implement getEventDimensions() method');
	}

	/**
	 * Get output dimension names - override in subclasses
	 */
	getOutputDimensions() {
		throw new Error('Channel must implement getOutputDimensions() method');
	}

	/**
	 * Get all possible action neurons for this channel - override in subclasses
	 * Returns array of coordinate objects: [{ [action-dim]: value }, ...]
	 * These neurons are pre-created during brain init so exploration can find them
	 */
	getActions() {
		return []; // Default: no predefined actions
	}

	/**
	 * Get frame events data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameEvents() {
		throw new Error('Channel must implement getFrameEvents() method');
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
	 * Calculate continuous prediction error for this channel.
	 * Called by diagnostics to get channel-specific error metrics (e.g., price prediction MAPE).
	 * Override in subclasses that have continuous prediction tracking.
	 * @returns {number|null} - Error value or null if no continuous error calculation
	 */
	calculatePredictionError() {
		return null; // Default: no continuous error tracking
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