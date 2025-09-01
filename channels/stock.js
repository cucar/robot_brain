import Channel from './channel.js';

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);
		
		this.symbol = name; // Extract symbol from name (e.g., "AAPL" from name)

		this.positionEntryPrice = null;
		this.positionEntryFrame = null;
		this.previousPrice = null; // Track previous price for feedback

		this.state = {}; // for tracking channel-specific state to determine action neurons results like owning a stock

		// Sample data for testing - in real implementation this would come from CSV or API
		this.sampleData = [
			{ price: 150.0, volume: 1000000 },
			{ price: 152.5, volume: 1200000 },
			{ price: 148.0, volume: 800000 },
			{ price: 151.0, volume: 1100000 },
			{ price: 149.5, volume: 900000 }
		];
		this.currentDataIndex = 0;
	}

	getInputDimensions() {
		return [
			`${this.symbol}_price`,
			`${this.symbol}_volume`
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
	 * Get valid exploration actions based on current stock position
	 * Can't sell if not owned, can't buy if already owned
	 */
	getValidExplorationActions() {
		const validActions = [];

		if (this.position !== 'long') {
			// Can buy if not already long
			validActions.push({ [`${this.symbol}_activity`]: 1 }); // Buy action
		}

		if (this.position === 'long') {
			// Can sell if currently long
			validActions.push({ [`${this.symbol}_activity`]: -1 }); // Sell action
		}

		// If no position, can also go short (sell first)
		if (this.position !== 'short' && this.position !== 'long') {
			validActions.push({ [`${this.symbol}_activity`]: -1 }); // Sell/short action
		}

		return validActions;
	}

	/**
	 * Get frame input data for this stock channel
	 */
	async getFrameInputs() {

		// Return current data point and advance index
		if (this.currentDataIndex >= this.sampleData.length) {
			console.log(`${this.symbol}: No more data available`);
			return [];
		}

		const currentData = this.sampleData[this.currentDataIndex];
		this.currentDataIndex++;
		this.frameNumber++;
		
		console.log(`${this.symbol}: Providing input data:`, currentData);
		
		// Store current price for feedback calculation in next frame
		this.previousPrice = currentData.price;
		
		// Return input neurons only
		return [
			{ [`${this.symbol}_price`]: currentData.price },
			{ [`${this.symbol}_volume`]: currentData.volume }
		];
	}

	/**
	 * Get feedback neurons based on position and price movement
	 */
	async getFeedbackNeurons() {
		// No feedback if we don't have a position or no previous price
		if (!this.position || this.previousPrice === null) {
			return [];
		}

		// Get current price from the most recent data
		const currentDataIndex = this.currentDataIndex - 1; // We already incremented in getFrameInputs
		if (currentDataIndex < 0 || currentDataIndex >= this.sampleData.length) {
			return [];
		}

		const currentPrice = this.sampleData[currentDataIndex].price;
		const priceChange = currentPrice - this.previousPrice;

		let feedbackValue = 0;

		if (this.position === 'long') { // We own the stock
			if (priceChange > 0) {
				feedbackValue = 1; // Joy - stock went up while we owned it
				console.log(`${this.symbol}: JOY! Owned stock went up by ${priceChange}`);
			} else if (priceChange < 0) {
				feedbackValue = -1; // Pain - stock went down while we owned it
				console.log(`${this.symbol}: PAIN! Owned stock went down by ${priceChange}`);
			}
		} else if (this.position === 'short') { // We're short the stock
			if (priceChange < 0) {
				feedbackValue = 1; // Joy - stock went down while we were short
				console.log(`${this.symbol}: JOY! Shorted stock went down by ${priceChange}`);
			} else if (priceChange > 0) {
				feedbackValue = -1; // Pain - stock went up while we were short
				console.log(`${this.symbol}: PAIN! Shorted stock went up by ${priceChange}`);
			}
		}

		if (feedbackValue !== 0) {
			return [{ [`${this.symbol}_reward`]: feedbackValue }];
		}

		return [];
	}

	async buildFrame(marketData, frameNumber) {
		const frame = [];

		// Add input neurons
		if (marketData.price !== undefined) {
			frame.push({ [`${this.symbol}_price`]: marketData.price });
		}
		if (marketData.volume !== undefined) {
			frame.push({ [`${this.symbol}_volume`]: marketData.volume });
		}

		// Add feedback neurons based on position and price movement
		if (this.position && marketData.priceChange !== undefined) {
			const feedbackNeuron = this.calculateFeedback(marketData.priceChange);
			if (feedbackNeuron) {
				frame.push(feedbackNeuron);
			}
		}

		// Add exploration actions if needed
		const explorationActions = this.generateExploration(
			frameNumber,
			[`action_buy_${this.symbol}`, `action_sell_${this.symbol}`]
		);
		frame.push(...explorationActions);

		return frame;
	}

	calculateFeedback(priceChange) {
		if (this.position === 'long') {
			if (priceChange > 0) {
				return { [`joy_owned_${this.symbol}_up`]: 1.0 };
			} else if (priceChange < 0) {
				return { [`pain_owned_${this.symbol}_down`]: 1.0 };
			}
		} else if (this.position === 'short') {
			if (priceChange < 0) {
				return { [`joy_short_${this.symbol}_down`]: 1.0 };
			} else if (priceChange > 0) {
				return { [`pain_short_${this.symbol}_up`]: 1.0 };
			}
		}
		return null;
	}

	async executeOutputs(predictions, frameNumber) {
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
				if (this.position !== 'long') {
					this.position = 'long';
					this.positionEntryPrice = currentPrice;
					this.positionEntryFrame = frameNumber;
					console.log(`${this.symbol}: EXECUTED BUY at ${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
				}
			} else if (strongestAction === 'sell') {
				if (this.position === 'long') {
					// Close long position
					const profit = currentPrice - this.positionEntryPrice;
					console.log(`${this.symbol}: EXECUTED SELL (close long) at ${currentPrice}, profit: ${profit.toFixed(2)} (strength: ${maxStrength.toFixed(3)})`);
					this.position = null;
				} else if (this.position !== 'short') {
					// Open short position
					this.position = 'short';
					this.positionEntryPrice = currentPrice;
					this.positionEntryFrame = frameNumber;
					console.log(`${this.symbol}: EXECUTED SELL (open short) at ${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
				}
			}
		}
	}

	async processRawInput(textString) {

		// Logic to convert time series data to standardized slope values between 0 and 1
		return [ { dimension_id: this.dimensionNameToId['slope'], value: textString.charCodeAt(0) } ];
	}
}