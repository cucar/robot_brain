import Channel from './channel.js';

/**
 * Text Channel - Handles character input and character output (like typing/speaking)
 * Input: characters being read/heard
 * Output: characters to type/speak
 * Feedback: language comprehension/generation rewards
 */
export default class TextChannel extends Channel {

	constructor(name, pattern = 'cats') {
		super(name);

		// Pattern to learn (can be set from job)
		this.pattern = pattern;

		this.currentLetterIndex = 0;
		this.patternIterations = 0;
		this.maxIterations = 10;
		this.lastPredictedChar = null;
	}

	/**
	 * we only have one character input at a time
	 */
	getInputDimensions() {
		return [ 'char_input' ];
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
	async getFrameInputs() {

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
	async getFeedback() {
		return 1.0; // Neutral - no feedback for now since this is simply mimicking the input
	}

	/**
	 * Get valid exploration actions
	 */
	getValidExplorationActions() {
		return []; // return empty array since this is input-only - for now
	}

	/**
	 * Resolve conflicts between multiple character predictions
	 * Since we can only predict one character at a time, pick the strongest prediction
	 * @param {Array} predictions - Array of prediction objects with structure:
	 *   [{ neuron_id, coordinates: {char_input: asciiValue}, strength: number }]
	 * @returns {Array} - Array with single strongest prediction, or empty if no predictions
	 */
	resolveConflicts(predictions) {
		if (!predictions || predictions.length === 0) {
			this.lastPredictedChar = null;
			return [];
		}

		// Filter to only predictions that have char_input coordinate
		const charPredictions = predictions.filter(pred => 'char_input' in pred.coordinates);

		if (charPredictions.length === 0) {
			this.lastPredictedChar = null;
			return [];
		}

		// Find the strongest prediction
		let strongest = charPredictions[0];
		for (const pred of charPredictions)
			if (pred.strength > strongest.strength) strongest = pred;

		// Convert ASCII code to character for display
		const asciiCode = Math.round(strongest.coordinates.char_input);
		const predictedChar = String.fromCharCode(asciiCode);

		// Store for feedback calculation
		this.lastPredictedChar = strongest.coordinates.char_input;

		console.log(`text: Predicted '${predictedChar}' (ASCII: ${asciiCode}, strength: ${strongest.strength.toFixed(1)})`);

		return [strongest];
	}

	/**
	 * Execute character output based on brain predictions
	 */
	async executeOutputs(predictions) {
		// there should not be any outputs for this channel - for now
	}
}
