import { Channel } from './channel.js';
import { Dimension } from './dimension.js';

/**
 * Tongue Channel - Handles taste input and tongue movement output
 * Input: taste data (sweet, sour, salty, bitter, umami)
 * Output: tongue movements (positioning, licking)
 * Feedback: taste preference rewards
 */
export class TongueChannel extends Channel {

	constructor(name, dimensions = null) {
		super(name);

		// Sample taste data - different flavors and intensities
		this.tasteData = [
			{ sweet: 0.8, sour: 0.1, salty: 0.2, bitter: 0.0, umami: 0.1, preference: 0.9 }, // Sweet treat
			{ sweet: 0.2, sour: 0.9, salty: 0.1, bitter: 0.0, umami: 0.0, preference: -0.5 }, // Sour lemon
			{ sweet: 0.0, sour: 0.0, salty: 0.9, bitter: 0.0, umami: 0.3, preference: 0.3 }, // Salty snack
			{ sweet: 0.1, sour: 0.2, salty: 0.1, bitter: 0.8, umami: 0.0, preference: -0.8 }, // Bitter medicine
			{ sweet: 0.3, sour: 0.0, salty: 0.4, bitter: 0.0, umami: 0.9, preference: 0.7 } // Savory umami
		];
		this.currentDataIndex = 0;
		this.tonguePosition = { x: 0.0, y: 0.0 }; // Current tongue position
		this.lastMovement = null;

		// Create or use provided dimension objects for this channel
		if (dimensions) {
			// Loading from database - use provided dimensions
			this.tasteSweetDim = dimensions.find(d => d.name === 'taste_sweet');
			this.tasteSourDim = dimensions.find(d => d.name === 'taste_sour');
			this.tasteSaltyDim = dimensions.find(d => d.name === 'taste_salty');
			this.tasteBitterDim = dimensions.find(d => d.name === 'taste_bitter');
			this.tasteUmamiDim = dimensions.find(d => d.name === 'taste_umami');
			this.tongueXDim = dimensions.find(d => d.name === 'tongue_x');
			this.tongueYDim = dimensions.find(d => d.name === 'tongue_y');

			// Validate all required dimensions exist
			if (!this.tasteSweetDim || !this.tasteSourDim || !this.tasteSaltyDim || !this.tasteBitterDim || !this.tasteUmamiDim || !this.tongueXDim || !this.tongueYDim)
				throw new Error(`TongueChannel ${name}: Missing required dimensions in database`);
		} else {
			// New channel - create dimensions with auto-increment IDs
			this.tasteSweetDim = new Dimension('taste_sweet');
			this.tasteSourDim = new Dimension('taste_sour');
			this.tasteSaltyDim = new Dimension('taste_salty');
			this.tasteBitterDim = new Dimension('taste_bitter');
			this.tasteUmamiDim = new Dimension('taste_umami');
			this.tongueXDim = new Dimension('tongue_x');
			this.tongueYDim = new Dimension('tongue_y');
		}
	}

	getEventDimensions() {
		return [ this.tasteSweetDim, this.tasteSourDim, this.tasteSaltyDim, this.tasteBitterDim, this.tasteUmamiDim ];
	}

	getActionDimensions() {
		return [ this.tongueXDim, this.tongueYDim ];
	}

	/**
	 * Get taste input data
	 */
	async getFrameEvents() {
		if (this.currentDataIndex >= this.tasteData.length) {
			console.log(`${this.name}: No more taste data available`);
			return [];
		}

		const currentData = this.tasteData[this.currentDataIndex];
		this.currentDataIndex++;
		this.frameNumber++;
		
		// Determine dominant taste
		const tastes = ['sweet', 'sour', 'salty', 'bitter', 'umami'];
		let dominantTaste = tastes[0];
		let maxIntensity = currentData[dominantTaste];
		
		for (const taste of tastes) {
			if (currentData[taste] > maxIntensity) {
				maxIntensity = currentData[taste];
				dominantTaste = taste;
			}
		}
		
		console.log(`${this.name}: Tasting ${dominantTaste} (intensity: ${maxIntensity.toFixed(2)})`);
		
		// Return taste input neurons
		return [
			{ taste_sweet: currentData.sweet },
			{ taste_sour: currentData.sour },
			{ taste_salty: currentData.salty },
			{ taste_bitter: currentData.bitter },
			{ taste_umami: currentData.umami }
		];
	}

	/**
	 * Get feedback based on taste preferences
	 */
	async getRewards() {
		if (!this.lastMovement) {
			return 1.0; // Neutral
		}

		// Get the taste preference for the current taste
		const currentDataIndex = this.currentDataIndex - 1;
		if (currentDataIndex < 0 || currentDataIndex >= this.tasteData.length) {
			return 1.0; // Neutral
		}

		const tastePreference = this.tasteData[currentDataIndex].preference;

		if (tastePreference > 0.5) {
			console.log(`${this.name}: PLEASANT! Good taste (preference: ${tastePreference.toFixed(2)})`);
			return 1.0 + Math.abs(tastePreference); // Scale positive reward by preference strength
		} else if (tastePreference < -0.5) {
			console.log(`${this.name}: UNPLEASANT! Bad taste (preference: ${tastePreference.toFixed(2)})`);
			return 1.0 - Math.abs(tastePreference); // Scale negative reward by preference strength
		} else {
			// Neutral taste, no strong feedback
			console.log(`${this.name}: Neutral taste (preference: ${tastePreference.toFixed(2)})`);
			return 1.0; // Neutral
		}
	}

	/**
	 * Execute tongue movements based on brain output
	 */
	async executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) return outputs;

		// Extract tongue movement predictions
		let tongueX = 0, tongueY = 0;
		let totalConfidence = 0;

		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				if (pred.coordinates.tongue_x !== undefined) {
					tongueX += pred.coordinates.tongue_x * pred.confidence;
					totalConfidence += pred.confidence;
				}
				if (pred.coordinates.tongue_y !== undefined) {
					tongueY += pred.coordinates.tongue_y * pred.confidence;
					totalConfidence += pred.confidence;
				}
			});
		});

		if (totalConfidence > 0) {
			tongueX /= totalConfidence;
			tongueY /= totalConfidence;
			
			// Execute the tongue movement
			this.tonguePosition.x += tongueX * 0.1; // Scale movement
			this.tonguePosition.y += tongueY * 0.1;
			
			// Keep within bounds (-1 to +1)
			this.tonguePosition.x = Math.max(-1, Math.min(1, this.tonguePosition.x));
			this.tonguePosition.y = Math.max(-1, Math.min(1, this.tonguePosition.y));
			
			this.lastMovement = { x: tongueX, y: tongueY };
			
			// Determine movement type
			let movementType = 'positioning';
			if (Math.abs(tongueX) > 0.5 || Math.abs(tongueY) > 0.5) {
				movementType = 'licking';
			}
			
			console.log(`${this.name}: EXECUTED TONGUE ${movementType.toUpperCase()} - Position: (${this.tonguePosition.x.toFixed(3)}, ${this.tonguePosition.y.toFixed(3)})`);
			
			outputs.actions.set('tongue_movement', { 
				x: tongueX, 
				y: tongueY, 
				position: { ...this.tonguePosition },
				type: movementType,
				confidence: totalConfidence 
			});
		}

		return outputs;
	}
}
