import Channel from './channel.js';

/**
 * Text Channel - Handles character input and character output (like typing/speaking)
 * Input: characters being read/heard
 * Output: characters to type/speak
 * Feedback: language comprehension/generation rewards
 */
export default class TextChannel extends Channel {

	constructor(name) {
		super(name);
		
		// Sample text data from main.js text processing example
		this.trainingWords = ["cats", "dogs", "bird", "fish"];
		this.currentWord = "cats";
		this.currentLetterIndex = 0;
		this.wordIterations = 0;
		this.maxIterations = 10;
		
		// Character to coordinate mapping (simplified 1D space from main.js)
		this.letterCoords = {
			'a': 0.1, 'b': 0.2, 'c': 0.3, 'd': 0.4, 'e': 0.5,
			'f': 0.6, 'g': 0.7, 'h': 0.8, 'i': 0.9, 'j': 1.0,
			'k': 0.11, 'l': 0.21, 'm': 0.31, 'n': 0.41, 'o': 0.51,
			'p': 0.61, 'q': 0.71, 'r': 0.81, 's': 0.91, 't': 0.101,
			'u': 0.111, 'v': 0.121, 'w': 0.131, 'x': 0.141, 'y': 0.151, 'z': 0.161
		};
		
		this.lastPredictedChar = null;
		this.currentPosition = 0; // Position in text sequence
	}

	getInputDimensions() {
		return [
			'char_input', 'text_position' // Character input and position in sequence
		];
	}

	getOutputDimensions() {
		return [
			'char_output' // Character to output/predict
		];
	}

	/**
	 * Get character input data
	 */
	async getFrameInputs() {
		if (this.wordIterations >= this.maxIterations) {
			console.log(`${this.name}: Completed text learning iterations`);
			return [];
		}

		// Reset to next word/iteration if current word is finished
		if (this.currentLetterIndex >= this.currentWord.length) {
			this.currentLetterIndex = 0;
			this.wordIterations++;
			this.currentPosition = 0;
			
			if (this.wordIterations >= this.maxIterations) {
				return [];
			}
			
			console.log(`${this.name}: Starting iteration ${this.wordIterations + 1} of word "${this.currentWord}"`);
		}

		const currentChar = this.currentWord[this.currentLetterIndex];
		const charValue = this.letterCoords[currentChar] || 0;
		
		this.currentLetterIndex++;
		this.frameNumber++;
		
		console.log(`${this.name}: Reading character '${currentChar}' at position ${this.currentPosition}`);
		
		// Return character input neurons
		const inputs = [
			{ char_input: charValue },
			{ text_position: this.currentPosition / 10.0 } // Normalized position
		];
		
		this.currentPosition++;
		return inputs;
	}

	/**
	 * Get feedback based on character prediction accuracy
	 */
	async getFeedbackNeurons() {
		if (!this.lastPredictedChar) {
			return [];
		}

		// Get the expected next character
		const expectedCharIndex = this.currentLetterIndex; // Next character index
		if (expectedCharIndex >= this.currentWord.length) {
			return []; // End of word, no feedback
		}

		const expectedChar = this.currentWord[expectedCharIndex];
		const expectedValue = this.letterCoords[expectedChar] || 0;
		
		// Check if prediction was close to expected character
		const threshold = 0.05;
		const error = Math.abs(this.lastPredictedChar - expectedValue);
		
		let feedbackValue;

		if (error < threshold) {
			feedbackValue = 1; // Reward for correct prediction
			console.log(`${this.name}: REWARD! Correctly predicted next character '${expectedChar}'`);
		} else {
			feedbackValue = -1; // Penalty for wrong prediction
			console.log(`${this.name}: PENALTY! Wrong prediction for character '${expectedChar}' (error: ${error.toFixed(3)})`);
		}

		return [{ text_reward: feedbackValue }];
	}

	/**
	 * Execute character output based on brain predictions
	 */
	async executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) {
			return outputs;
		}

		// Extract character output predictions
		let charOutput = 0;
		let totalConfidence = 0;

		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				if (pred.coordinates.char_output !== undefined) {
					charOutput += pred.coordinates.char_output * pred.confidence;
					totalConfidence += pred.confidence;
				}
			});
		});

		if (totalConfidence > 0) {
			charOutput /= totalConfidence;
			
			// Find closest character to the predicted value
			let closestChar = 'a';
			let minDistance = Infinity;
			
			for (const [char, value] of Object.entries(this.letterCoords)) {
				const distance = Math.abs(charOutput - value);
				if (distance < minDistance) {
					minDistance = distance;
					closestChar = char;
				}
			}
			
			this.lastPredictedChar = charOutput;
			
			console.log(`${this.name}: PREDICTED CHARACTER '${closestChar}' (value: ${charOutput.toFixed(3)}, confidence: ${totalConfidence.toFixed(3)})`);
			
			outputs.actions.set('character', { 
				char: closestChar, 
				value: charOutput, 
				confidence: totalConfidence 
			});
		}

		return outputs;
	}
}
