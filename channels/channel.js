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
	}

	/**
	 * Execute outputs based on brain predictions - override in subclasses
	 * This method should execute actions and update channel state
	 */
	async executeOutputs(predictions) {
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
	 * Get feedback based on previous actions and current state - override in subclasses
	 * Returns object with joy and pain values: { joy: number, pain: number }
	 * Joy represents positive feedback (rewards), Pain represents negative feedback (punishment)
	 */
	async getFeedback() {
		return { joy: 0, pain: 0 };
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
}