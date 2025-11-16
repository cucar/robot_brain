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
	 * Get resolved inference for text channel
	 * Resolves conflicts for input predictions only (no outputs for this channel)
	 * Input predictions: char_input (strongest wins)
	 * @param {Array} inferences - all inferred neurons for this channel
	 * @returns {Array} - strongest character prediction
	 */
	getResolvedInference(inferences) {

		// if there are no inferences, nothing to resolve
		if (!inferences || inferences.length === 0) {
			this.lastPredictedChar = null;
			return [];
		}

		// Separate into input predictions and output inferences
		const { inputPredictions } = this.separateInputsAndOutputs(inferences);

		// Resolve input predictions: select strongest character
		// No output resolution needed for text channel (no output dimensions)
		return this.resolveInputPredictions(inputPredictions);
	}

	/**
	 * Resolve input predictions: select strongest character prediction
	 * @param {Array} inputPredictions - predictions for char_input dimension
	 * @returns {Array} - strongest character prediction
	 */
	resolveInputPredictions(inputPredictions) {
		if (inputPredictions.length === 0) {
			this.lastPredictedChar = null;
			return [];
		}

		// Find strongest character prediction
		const strongest = this.findStrongest(inputPredictions);
		if (!strongest) {
			this.lastPredictedChar = null;
			return [];
		}

		// Store and log prediction
		this.storePrediction(strongest);
		this.logPrediction(strongest);

		return [strongest];
	}

	/**
	 * Find the strongest inference from a list
	 * @param {Array} inferences - list of inferences
	 * @returns {Object|null} - strongest inference or null if empty
	 */
	findStrongest(inferences) {
		if (inferences.length === 0) return null;

		let strongest = inferences[0];
		for (const inf of inferences)
			if (inf.strength > strongest.strength)
				strongest = inf;

		return strongest;
	}

	/**
	 * Store prediction for feedback calculation
	 * @param {Object} prediction - the prediction to store
	 */
	storePrediction(prediction) {
		this.lastPredictedChar = prediction.coordinates.char_input;
	}

	/**
	 * Log prediction for debugging
	 * @param {Object} prediction - the prediction to log
	 */
	logPrediction(prediction) {
		const asciiCode = Math.round(prediction.coordinates.char_input);
		const predictedChar = String.fromCharCode(asciiCode);
		console.log(`text: Predicted '${predictedChar}' (ASCII: ${asciiCode}, strength: ${prediction.strength.toFixed(1)})`);
	}

	/**
	 * Execute character output based on brain predictions
	 */
	async executeOutputs(predictions) {
		// there should not be any outputs for this channel - for now
	}
}
