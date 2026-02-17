import { Channel } from './channel.js';
import { Dimension } from './dimension.js';

/**
 * Text Channel - Handles character input and character output (like typing/speaking)
 * Input: characters being read/heard
 * Output: characters to type/speak
 * Feedback: language comprehension/generation rewards
 */
export class TextChannel extends Channel {

	constructor(name, pattern = 'cats', dimensions = null) {
		super(name);

		// Pattern to learn (can be set from job)
		this.pattern = pattern;

		this.currentLetterIndex = 0;
		this.patternIterations = 0;
		this.maxIterations = 10;

		// Create or use provided dimension objects for this channel
		if (dimensions) {
			// Loading from database - use provided dimensions
			this.charInputDim = dimensions.find(d => d.name === 'char_input');

			// Validate all required dimensions exist
			if (!this.charInputDim)
				throw new Error(`TextChannel ${name}: Missing required dimensions in database`);
		} else {
			// New channel - create dimensions with auto-increment IDs
			this.charInputDim = new Dimension('char_input');
		}
	}

	/**
	 * we only have one character input at a time
	 */
	getEventDimensions() {
		return [ this.charInputDim ];
	}

	/**
	 * text channel does not need any outputs - for now we are simply mimicking the inputs
	 * in the future, we may want to generate text based on conscious thoughts - we would add it then
	 * @returns {*[]}
	 */
	getOutputDimensions() {
		return [];
	}

	/**
	 * Get character input data
	 */
	async getFrameEvents() {

		// Stop if we have reached the maximum number of iterations
		if (this.patternIterations >= this.maxIterations) return [];

		// Reset to next iteration if current pattern is finished
		if (this.currentLetterIndex >= this.pattern.length) {
			this.currentLetterIndex = 0;
			this.patternIterations++;
			if (this.patternIterations >= this.maxIterations) return [];
		}

		// Use ASCII code for dimension value of the base neuron
		const charValue = this.pattern[this.currentLetterIndex].charCodeAt(0);

		// advance to next letter
		this.currentLetterIndex++;

		// Return character input neuron
		return [ { char_input: charValue } ];
	}

	/**
	 * Get feedback based on character prediction accuracy
	 */
	async getRewards() {
		return 1.0; // Neutral - no feedback for now since this is simply mimicking the input
	}

	/**
	 * Execute character output based on brain predictions
	 * Returns final frame points (inputs only, since text channel has no outputs)
	 */
	async executeOutputs(inputs, outputs) {
		// Text channel has no outputs - just return inputs as-is
		return inputs;
	}

	/**
	 * Get channel metrics for diagnostic reporting
	 * @returns {Object} - Text channel metrics
	 */
	getMetrics() {
		return {
			...super.getMetrics(),
			pattern: this.pattern,
			currentLetterIndex: this.currentLetterIndex,
			patternIterations: this.patternIterations,
			maxIterations: this.maxIterations,
			progress: `${this.patternIterations}/${this.maxIterations} iterations`
		};
	}
}
