import Channel from './channel.js';

/**
 * Eyes Channel - Handles visual input and eye movement output (saccades)
 * Input: visual data (pixels, colors, positions)
 * Output: eye movements (saccade directions)
 * Feedback: visual attention rewards
 */
export default class EyesChannel extends Channel {
	
	constructor(name) {
		super(name);
		
		// Sample visual data - in real implementation this would come from camera/image processing
		this.visualData = [
			{ x: 0.1, y: 0.2, r: 1.0, g: 0.0, b: 0.0 }, // Red dot at 10,20
			{ x: 0.11, y: 0.21, r: 1.0, g: 0.0, b: 0.0 }, // Slightly shifted red dot
			{ x: 0.3, y: 0.4, r: 0.0, g: 1.0, b: 0.0 }, // Green dot at 30,40
			{ x: 0.1, y: 0.2, r: 1.0, g: 0.0, b: 0.0 }, // Red dot again - reinforcement
			{ x: 0.12, y: 0.22, r: 1.0, g: 0.0, b: 0.0 } // Another slightly shifted red dot
		];
		this.currentDataIndex = 0;
		this.eyePosition = { x: 0.0, y: 0.0 }; // Current eye position
		this.lastSaccade = null;
	}

	getInputDimensions() {
		return [
			'visual_x', 'visual_y', 'visual_r', 'visual_g', 'visual_b'
		];
	}

	getOutputDimensions() {
		return [
			'saccade_x', 'saccade_y' // Eye movement directions
		];
	}

	/**
	 * Get visual input data
	 */
	async getFrameInputs() {
		if (this.currentDataIndex >= this.visualData.length) {
			console.log(`${this.name}: No more visual data available`);
			return [];
		}

		const currentData = this.visualData[this.currentDataIndex];
		this.currentDataIndex++;
		this.frameNumber++;
		
		console.log(`${this.name}: Seeing visual data:`, currentData);
		
		// Return visual input neurons
		return [
			{ visual_x: currentData.x },
			{ visual_y: currentData.y },
			{ visual_r: currentData.r },
			{ visual_g: currentData.g },
			{ visual_b: currentData.b }
		];
	}

	/**
	 * Get feedback based on eye movements and visual targets
	 */
	async getFeedback() {
		if (!this.lastSaccade) {
			return 1.0; // Neutral
		}

		// Get current visual target
		const currentDataIndex = this.currentDataIndex - 1;
		if (currentDataIndex < 0 || currentDataIndex >= this.visualData.length) {
			return 1.0; // Neutral
		}

		const target = this.visualData[currentDataIndex];
		const distance = Math.sqrt(
			Math.pow(this.eyePosition.x - target.x, 2) +
			Math.pow(this.eyePosition.y - target.y, 2)
		);

		const threshold = 0.05; // How close the eye needs to be to the target

		if (distance < threshold) {
			console.log(`${this.name}: SUCCESS! Eye fixated on target (distance: ${distance.toFixed(3)})`);
			return 1.5; // Positive reward factor
		}
		else {
			console.log(`${this.name}: MISS! Eye missed target (distance: ${distance.toFixed(3)})`);
			return 0.5; // Negative reward factor
		}
	}

	/**
	 * Execute eye movements (saccades) based on brain output
	 */
	async executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) {
			return outputs;
		}

		// Extract saccade predictions
		let saccadeX = 0, saccadeY = 0;
		let totalConfidence = 0;

		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				if (pred.coordinates.saccade_x !== undefined) {
					saccadeX += pred.coordinates.saccade_x * pred.confidence;
					totalConfidence += pred.confidence;
				}
				if (pred.coordinates.saccade_y !== undefined) {
					saccadeY += pred.coordinates.saccade_y * pred.confidence;
					totalConfidence += pred.confidence;
				}
			});
		});

		if (totalConfidence > 0) {
			saccadeX /= totalConfidence;
			saccadeY /= totalConfidence;
			
			// Execute the saccade (move eyes)
			this.eyePosition.x += saccadeX * 0.1; // Scale movement
			this.eyePosition.y += saccadeY * 0.1;
			
			// Keep within bounds
			this.eyePosition.x = Math.max(0, Math.min(1, this.eyePosition.x));
			this.eyePosition.y = Math.max(0, Math.min(1, this.eyePosition.y));
			
			this.lastSaccade = { x: saccadeX, y: saccadeY };
			
			console.log(`${this.name}: EXECUTED SACCADE to (${this.eyePosition.x.toFixed(3)}, ${this.eyePosition.y.toFixed(3)})`);
			
			outputs.actions.set('saccade', { x: saccadeX, y: saccadeY, confidence: totalConfidence });
		}

		return outputs;
	}
}