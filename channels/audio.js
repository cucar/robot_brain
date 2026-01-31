import Channel from './channel.js';
import { Dimension } from './dimension.js';

/**
 * Ears Channel - Handles audio input and ear movement output (auriculomotor)
 * Input: audio data (frequency, amplitude, duration)
 * Output: ear movements (orientation towards sound)
 * Feedback: sound localization rewards
 */
export default class EarsChannel extends Channel {

	constructor(name) {
		super(name);

		// Sample audio data - frequencies, amplitudes, and durations
		this.audioData = [
			{ frequency: 440, amplitude: 0.8, duration: 0.5, direction: 0.2 }, // A note from right
			{ frequency: 523, amplitude: 0.6, duration: 0.3, direction: -0.3 }, // C note from left
			{ frequency: 659, amplitude: 0.9, duration: 0.4, direction: 0.0 }, // E note from center
			{ frequency: 440, amplitude: 0.7, duration: 0.6, direction: 0.1 }, // A note slight right
			{ frequency: 392, amplitude: 0.5, duration: 0.2, direction: -0.2 } // G note from left
		];
		this.currentDataIndex = 0;
		this.earOrientation = 0.0; // Current ear orientation (-1 left, +1 right)
		this.lastMovement = null;

		// Create dimension objects for this channel
		this.audioFrequencyDim = new Dimension('audio_frequency');
		this.audioAmplitudeDim = new Dimension('audio_amplitude');
		this.audioDurationDim = new Dimension('audio_duration');
		this.earOrientationDim = new Dimension('ear_orientation');
	}

	getEventDimensions() {
		return [ this.audioFrequencyDim, this.audioAmplitudeDim, this.audioDurationDim ];
	}

	getOutputDimensions() {
		return [ this.earOrientationDim ];
	}

	/**
	 * Get audio input data
	 */
	async getFrameEvents() {
		if (this.currentDataIndex >= this.audioData.length) {
			console.log(`${this.name}: No more audio data available`);
			return [];
		}

		const currentData = this.audioData[this.currentDataIndex];
		this.currentDataIndex++;
		this.frameNumber++;
		
		console.log(`${this.name}: Hearing audio:`, {
			frequency: currentData.frequency,
			amplitude: currentData.amplitude,
			duration: currentData.duration,
			from: currentData.direction > 0 ? 'right' : currentData.direction < 0 ? 'left' : 'center'
		});
		
		// Normalize inputs to 0-1 range
		return [
			{ audio_frequency: currentData.frequency / 1000.0 }, // Normalize frequency
			{ audio_amplitude: currentData.amplitude },
			{ audio_duration: currentData.duration }
		];
	}

	/**
	 * Get feedback based on ear movement accuracy for sound localization
	 */
	async getRewards() {
		if (!this.lastMovement) {
			return 1.0; // Neutral
		}

		// Get the actual sound direction
		const currentDataIndex = this.currentDataIndex - 1;
		if (currentDataIndex < 0 || currentDataIndex >= this.audioData.length) {
			return 1.0; // Neutral
		}

		const actualDirection = this.audioData[currentDataIndex].direction;
		const orientationError = Math.abs(this.earOrientation - actualDirection);
		const threshold = 0.2; // Acceptable localization error

		if (orientationError < threshold) {
			console.log(`${this.name}: SUCCESS! Accurately localized sound (error: ${orientationError.toFixed(3)})`);
			return 1.5; // Positive reward factor
		} else {
			console.log(`${this.name}: MISS! Poor sound localization (error: ${orientationError.toFixed(3)})`);
			return 0.5; // Negative reward factor
		}
	}

	/**
	 * Execute ear movements based on brain output
	 */
	async executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) {
			return outputs;
		}

		// Extract ear orientation predictions
		let earMovement = 0;
		let totalConfidence = 0;

		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				if (pred.coordinates.ear_orientation !== undefined) {
					earMovement += pred.coordinates.ear_orientation * pred.confidence;
					totalConfidence += pred.confidence;
				}
			});
		});

		if (totalConfidence > 0) {
			earMovement /= totalConfidence;
			
			// Update ear orientation
			this.earOrientation += earMovement * 0.1; // Scale movement
			
			// Keep within bounds (-1 to +1)
			this.earOrientation = Math.max(-1, Math.min(1, this.earOrientation));
			
			this.lastMovement = earMovement;
			
			const direction = this.earOrientation > 0.1 ? 'right' : this.earOrientation < -0.1 ? 'left' : 'center';
			console.log(`${this.name}: EXECUTED EAR MOVEMENT - Orientation: ${this.earOrientation.toFixed(3)} (${direction})`);
			
			outputs.actions.set('ear_movement', { 
				orientation: this.earOrientation, 
				movement: earMovement, 
				confidence: totalConfidence 
			});
		}

		return outputs;
	}
}