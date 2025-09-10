import Channel from './channel.js';

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);
		
		this.symbol = name; // Extract symbol from name (e.g., "AAPL" from name)

		// State tracking
		this.owned = false; // Simple owned flag
		this.entryPrice = null; // Price when we bought
		this.holdingFrames = 0; // How long we've held the position
		this.previousPrice = null; // Track previous price for change calculation
		this.previousVolume = null; // Track previous volume for change calculation

		// Sample data for testing - in real implementation this would come from CSV or API
		this.sampleData = [
			{ price: 150.0, volume: 1000000 },
			{ price: 152.5, volume: 1200000 },
			{ price: 148.0, volume: 800000 },
			{ price: 151.0, volume: 1100000 },
			{ price: 149.5, volume: 900000 },
			{ price: 153.0, volume: 1300000 },
			{ price: 147.0, volume: 900000 },
			{ price: 155.0, volume: 1500000 }
		];
		this.currentDataIndex = 0;

		// Exponential discretization buckets for percentage changes
		this.priceBuckets = [
			{ min: -Infinity, max: -50, value: -7 },  // -100%+ to -50%
			{ min: -50, max: -25, value: -6 },        // -50% to -25%
			{ min: -25, max: -12.5, value: -5 },      // -25% to -12.5%
			{ min: -12.5, max: -6.25, value: -4 },    // -12.5% to -6.25%
			{ min: -6.25, max: -3.125, value: -3 },   // -6.25% to -3.125%
			{ min: -3.125, max: -1.5625, value: -2 }, // -3.125% to -1.5625%
			{ min: -1.5625, max: -0.05, value: -1 },  // -1.5625% to -0.05%
			{ min: -0.05, max: 0.05, value: 0 },      // -0.05% to 0.05% (no change)
			{ min: 0.05, max: 1.5625, value: 1 },     // 0.05% to 1.5625%
			{ min: 1.5625, max: 3.125, value: 2 },    // 1.5625% to 3.125%
			{ min: 3.125, max: 6.25, value: 3 },      // 3.125% to 6.25%
			{ min: 6.25, max: 12.5, value: 4 },       // 6.25% to 12.5%
			{ min: 12.5, max: 25, value: 5 },         // 12.5% to 25%
			{ min: 25, max: 50, value: 6 },           // 25% to 50%
			{ min: 50, max: Infinity, value: 7 }      // 50%+ to 100%+
		];
	}

	getInputDimensions() {
		return [
			`${this.symbol}_price_change`, // Discretized percentage change in price
			`${this.symbol}_volume_change` // Discretized percentage change in volume
		];
	}

	getOutputDimensions() {
		return [
			`${this.symbol}_activity` // -1 for sell, +1 for buy
		];
	}

	getFeedbackDimensions() {
		return [
			`${this.symbol}_reward` // +1 for joy, -1 for pain
		];
	}

	/**
	 * Discretize percentage change into exponential buckets
	 */
	discretizePercentageChange(percentChange) {
		for (const bucket of this.priceBuckets)
			if (percentChange > bucket.min && percentChange <= bucket.max)
				return bucket.value;
		return 0; // Default to no change bucket
	}

	/**
	 * Get valid exploration actions based on current stock ownership
	 * Can't sell if not owned, can't buy if already owned
	 */
	getValidExplorationActions() {
		return [{ [`${this.symbol}_activity`]: (this.owned ? -1 : 1) }]
	}

	/**
	 * Get frame input data for this stock channel
	 */
	async getFrameData() {
		
		// Return current data point and advance index
		if (this.currentDataIndex >= this.sampleData.length) {
			console.log(`${this.symbol}: No more data available`);
			return [];
		}

		// get raw current stock data for the frame - TODO: make this read from a file or table
		const currentData = this.sampleData[this.currentDataIndex];
		this.currentDataIndex++;

		// if this is the first frame, return zero change (no previous data to compare)
		if (this.previousPrice !== null && this.previousVolume !== null) {
			console.log(`${this.symbol}: First frame - using zero change`);
			this.previousPrice = currentData.price;
			this.previousVolume = currentData.volume;
			return [
				{ [`${this.symbol}_price_change`]: 0 },
				{ [`${this.symbol}_volume_change`]: 0 }
			];
		}

		// Calculate percentage changes
		const priceChange = ((currentData.price - this.previousPrice) / this.previousPrice) * 100;
		const volumeChange = ((currentData.volume - this.previousVolume) / this.previousVolume) * 100;
		console.log(`${this.symbol}: Price: ${currentData.price} (${priceChange.toFixed(2)}%), Volume: ${currentData.volume} (${volumeChange.toFixed(2)}%)`);
		
		// Update previous values for next frame
		this.previousPrice = currentData.price;
		this.previousVolume = currentData.volume;
		
		// Increment holding counter if we own the stock
		if (this.owned) this.holdingFrames++;
		
		// Discretize the percentage changes
		const discretePriceChange = this.discretizePercentageChange(priceChange);
		const discreteVolumeChange = this.discretizePercentageChange(volumeChange);
		
		// Return discretized input neurons
		return [
			{ [`${this.symbol}_price_change`]: discretePriceChange },
			{ [`${this.symbol}_volume_change`]: discreteVolumeChange }
		];
	}

	/**
	 * Get feedback neurons based on position and price movement
	 */
	async getFeedbackNeurons() {

		// No feedback if we don't own stock
		if (!this.owned) return [];

		// Get current price from the most recent data
		const currentDataIndex = this.currentDataIndex - 1; // We already incremented in getFrameInputs
		if (currentDataIndex < 0 || currentDataIndex >= this.sampleData.length) return [];

		const currentPrice = this.sampleData[currentDataIndex].price;
		const priceChange = currentPrice - this.entryPrice;
		const percentChange = (priceChange / this.entryPrice) * 100;

		let feedbackValue = 0;
		if (priceChange > 0) {
			feedbackValue = 1; // Joy - stock went up since we bought it
			console.log(`${this.symbol}: JOY! Stock up ${percentChange.toFixed(2)}% since purchase (${priceChange.toFixed(2)} profit)`);
		} 
		else if (priceChange < 0) {
			feedbackValue = -1; // Pain - stock went down since we bought it
			console.log(`${this.symbol}: PAIN! Stock down ${percentChange.toFixed(2)}% since purchase (${Math.abs(priceChange).toFixed(2)} loss)`);
		}

		return feedbackValue === 0 ? [] : [{ [`${this.symbol}_reward`]: feedbackValue }];
	}

	async executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) {
			return outputs;
		}

		// Extract activity predictions from all prediction frames
		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				// Check for activity dimension (our new output format)
				if (pred.coordinates[`${this.symbol}_activity`] !== undefined) {
					const activityValue = pred.coordinates[`${this.symbol}_activity`];
					const confidence = pred.confidence;
					
					if (activityValue > 0) {
						// Positive activity = buy signal
						const current = outputs.actions.get('buy') || 0;
						outputs.actions.set('buy', current + confidence);
					} else if (activityValue < 0) {
						// Negative activity = sell signal
						const current = outputs.actions.get('sell') || 0;
						outputs.actions.set('sell', current + confidence);
					}
				}
			});
		});

		// Execute the strongest action (fake buying/selling)
		this.executeAction(outputs.actions, frameNumber);

		return outputs;
	}

	executeAction(actions, frameNumber) {
		let strongestAction = null;
		let maxStrength = 0;

		for (const [action, strength] of actions) {
			if (strength > maxStrength) {
				maxStrength = strength;
				strongestAction = action;
			}
		}

		if (strongestAction && maxStrength > 0) {

			// Get current price for position tracking
			const currentDataIndex = this.currentDataIndex - 1;
			const currentPrice = this.sampleData[currentDataIndex]?.price;

			if (strongestAction === 'buy') {
				if (!this.owned) {
					this.owned = true;
					this.entryPrice = currentPrice;
					this.holdingFrames = 0; // Reset holding counter
					console.log(`${this.symbol}: EXECUTED BUY at $${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
				}
			} else if (strongestAction === 'sell') {
				if (this.owned) {
					// Sell owned stock
					const profit = currentPrice - this.entryPrice;
					const percentReturn = (profit / this.entryPrice) * 100;
					
					console.log(`${this.symbol}: EXECUTED SELL at $${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
					console.log(`${this.symbol}: Profit/Loss: $${profit.toFixed(2)} (${percentReturn.toFixed(2)}%) over ${this.holdingFrames} frames`);
					
					this.owned = false;
					this.entryPrice = null;
					this.holdingFrames = 0; // Reset holding counter
				}
			}
		}
	}
}