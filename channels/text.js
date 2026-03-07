import { Channel } from './channel.js';
import { Dimension } from './dimension.js';

/**
 * Text Channel - Handles character input for text sequence learning
 * Input: characters being read (ASCII codes)
 * Output: none for now (event inference only)
 * Goal: Test if brain can memorize and predict character sequences
 */
export class TextChannel extends Channel {

	/**
	 * Constructor for text channel - dimensions are given when loading from database
	 */
	constructor(name, debug, id = null, dimensions = null) {
		super(name, debug, id);

		// Initialize dimensions
		this.initializeDimensions(dimensions);

		// Training data / mode (set by setTraining)
		this.trainingData = null;
		this.trainingIndex = 0;

		// Initialize context
		this.resetContext();
	}

	/**
	 * Initialize dimensions - dimensions are given when loading from database
	 */
	initializeDimensions(dimensions) {
		if (dimensions && dimensions.length > 0) {
			// Loading from database - use provided dimensions
			this.charDim = dimensions.find(d => d.name === `${this.name}_char`);
			if (!this.charDim)
				throw new Error(`TextChannel ${this.name}: Missing required dimensions in database`);
		}
		else {
			// New channel - create dimensions with auto-increment IDs
			this.charDim = new Dimension(`${this.name}_char`);
		}
	}

	/**
	 * Set training data for this channel - switches channel to training mode
	 * @param {string} text - Text string to train on
	 * @param {number} iterations - Number of times to repeat the text
	 */
	setTraining(text, iterations = 1) {
		// Repeat the text for the specified number of iterations
		this.trainingData = text.repeat(iterations);
		this.trainingIndex = 0;
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets reading state)
	 */
	resetContext() {
		this.trainingIndex = 0;
	}

	/**
	 * Returns the input dimensions for the channel
	 */
	getEventDimensions() {
		return [this.charDim];
	}

	/**
	 * Returns the output dimensions for the channel - none for now
	 */
	getActionDimensions() {
		return [];
	}

	/**
	 * Get frame event data for this text channel
	 * @param {number} frameNumber - Current frame number (1-indexed)
	 */
	getFrameEvents(frameNumber) {
		if (this.trainingData === null)
			throw new Error(`${this.name}: Training data not set. Call setTraining() first.`);

		// Return empty if all characters consumed
		if (this.trainingIndex >= this.trainingData.length)
			return [];

		// Read current character and use ASCII code for dimension value
		const char = this.trainingData[this.trainingIndex++];
		const charCode = char.charCodeAt(0);
		if (this.debug)
			console.log(`${this.name}: Frame ${frameNumber} - char '${char}' (${charCode})`);

		return [{ [`${this.name}_char`]: charCode }];
	}

	/**
	 * Get feedback - neutral for text channel (no actions)
	 */
	async getRewards() {
		return 0; // Neutral - no actions to reward
	}

	/**
	 * Execute outputs - text channel has no outputs for now
	 */
	async executeOutputs() {
		// No outputs to execute
	}

	/**
	 * Get short state display for frame summary
	 * Shows current progress through the text
	 * @returns {string|null} - Progress display or null if no training data
	 */
	getStateDisplay() {
		if (!this.trainingData) return null;
		return `${this.name}:${this.trainingIndex}/${this.trainingData.length}`;
	}

	/**
	 * Get channel metrics for diagnostic reporting
	 * @returns {Object} - Text channel metrics
	 */
	getMetrics() {
		const totalChars = this.trainingData ? this.trainingData.length : 0;
		return {
			...super.getMetrics(),
			totalChars,
			currentIndex: this.trainingIndex,
			progress: totalChars > 0 ? `${this.trainingIndex}/${totalChars}` : 'N/A'
		};
	}
}
