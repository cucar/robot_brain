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
export class Channel {

	static nextId = 1; // Start at 1 to match typical DB conventions

	/**
	 * dimensions are given as a 4th argument when loading from database - children may use them
	 */
	constructor(name, debug, id = null) {
		this.id = id !== null ? id : Channel.nextId++;
		this.name = name; // just for descriptions in debugging
		this.frameNumber = 0; // frame counter for channel-specific operations
		this.debug = debug; // controls verbosity of channel output
		this.actionSequences = true; // whether action neurons participate in learning context (override false for channels where actions don't affect events)

		// Update nextId if we're loading a channel with a specific ID
		if (id !== null && id >= Channel.nextId) Channel.nextId = id + 1;
	}

	/**
	 * Static method to reset channel-level context (shared state across all instances)
	 * Called once per episode reset before individual channel resetContext calls
	 * Override in subclasses if they have static/shared state to reset
	 */
	static resetChannelContext() {
		// Default implementation does nothing
		// Child classes can override for channel-specific static state reset
	}

	/**
	 * Static method called by Thalamus to execute actions for all channels of this type
	 * Default implementation: call each channel's executeOutputs individually
	 * Override in subclasses for coordinated execution (e.g., portfolio management)
	 * @param {Map<string, { channel, actions, events }>} channelInferences - Map of channel name to channel data
	 */
	static async executeChannelActions(channelInferences) {
		for (const [, { channel, actions, events }] of channelInferences)
			await channel.executeOutputs(actions, events);
	}

	/**
	 * Static method to get aggregate metrics across all channels of this type
	 * Override in subclasses to provide channel-type-specific aggregation (e.g., portfolio metrics)
	 * @param {Array} channels - Array of [channelName, channel] pairs
	 * @returns {Object|null} - Aggregate metrics or null if not applicable
	 */
	static getAggregateMetrics(channels) {
		return null;
	}

	/**
	 * Static method to get aggregate display string for frame summary
	 * Override in subclasses to provide channel-type-specific display
	 * @param {Array} channels - Array of [channelName, channel] pairs
	 * @returns {string|null} - Formatted display string or null if nothing to show
	 */
	static getAggregateDisplay(channels) {
		return null;
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
	getActionDimensions() {
		throw new Error('Channel must implement getActionDimensions() method');
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
	 * returns the coordinates of the channel default action - to be implemented by child classes
	 * this should correspond to an action that does nothing - trigger doing something when doing nothing is negatively rewarded
	 */
	getDefaultAction() {
		return null;
	}

	/**
	 * Get frame events data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 * @param {number} frameNumber - Current frame number (1-indexed)
	 */
	async getFrameEvents(frameNumber) {
		throw new Error('Channel must implement getFrameEvents() method');
	}

	/**
	 * Get feedback based on previous actions and current state - override in subclasses
	 * Returns reward factor: number (1.0 = neutral, >1.0 = positive, <1.0 = negative)
	 * This factor will be multiplied with existing neuron reward factors
	 */
	async getRewards() {
		return 0; // Neutral feedback by default
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

	/**
	 * Get channel metrics for diagnostic reporting
	 * Returns channel-specific state and performance metrics
	 * Override in subclasses to provide channel-specific metrics
	 * @returns {Object} - Channel metrics object
	 */
	getMetrics() {
		return {
			name: this.name
		};
	}

	/**
	 * Get short state display for frame summary
	 * Returns a brief string showing current channel state
	 * Override in subclasses for channel-specific state display
	 * @returns {string|null} - Short state string or null if nothing to display
	 */
	getStateDisplay() {
		return null;
	}

}