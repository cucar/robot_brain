import Channel from './channel.js';
import { createReadStream, mkdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// State tracking
		this.owned = false; // true = owned, false = sold (after first buy)
		this.entryPrice = null; // Price when we bought (for owned) or sold (for sold)
		this.holdingFrames = 0; // How long we've held the current position
		this.previousPrice = null; // Track previous price for change calculation
		this.previousVolume = null; // Track previous volume for change calculation
		this.hasTraded = false; // Track if we've ever made a trade

		// Episode metrics tracking
		this.totalProfit = 0; // Total profit from all trades in current episode
		this.totalLoss = 0; // Total loss from all trades in current episode
		this.totalTrades = 0; // Total number of trades in current episode
		this.profitableTrades = 0; // Number of profitable trades in current episode
		this.unrealizedProfit = 0; // Current unrealized profit/loss from open position
		this.lastUnrealizedProfit = 0; // Previous unrealized profit for tracking changes

		// Continuous prediction tracking for price
		this.lastPredictedPrice = null; // Predicted price from previous frame
		this.pricePredictionErrors = []; // Array of price prediction errors for calculating metrics
		this.priceForPrediction = null; // Price to use as base for next prediction (set before updating currentPrice)

		// Holdout configuration
		this.holdoutRows = 0; // Number of rows to hold out from training (set by job)
		this.isTrainingMode = true; // true = training (skip holdout rows), false = prediction (use only holdout rows)
		this.allRows = []; // Store all CSV rows for holdout management

		// CSV reading state
		this.csvPath = null;
		this.rl = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Fine-grained discretization for typical stock movements (-2% to +2%) with exponential for extremes
		this.priceBuckets = [
			{ min: -Infinity, max: -10, value: -10 },  // -100%+ to -10%
			{ min: -10, max: -5, value: -9 },          // -10% to -5%
			{ min: -5, max: -3, value: -8 },           // -5% to -3%
			{ min: -3, max: -2, value: -7 },           // -3% to -2%
			{ min: -2, max: -1, value: -6 },           // -2% to -1%
			{ min: -1, max: -0.5, value: -5 },         // -1% to -0.5%
			{ min: -0.5, max: -0.2, value: -4 },       // -0.5% to -0.2%
			{ min: -0.2, max: -0.05, value: -3 },      // -0.2% to -0.05%
			{ min: -0.05, max: -0.01, value: -2 },     // -0.05% to -0.01%
			{ min: -0.01, max: 0.01, value: 0 },       // -0.01% to 0.01% (no change)
			{ min: 0.01, max: 0.05, value: 2 },        // 0.01% to 0.05%
			{ min: 0.05, max: 0.2, value: 3 },         // 0.05% to 0.2%
			{ min: 0.2, max: 0.5, value: 4 },          // 0.2% to 0.5%
			{ min: 0.5, max: 1, value: 5 },            // 0.5% to 1%
			{ min: 1, max: 2, value: 6 },              // 1% to 2%
			{ min: 2, max: 3, value: 7 },              // 2% to 3%
			{ min: 3, max: 5, value: 8 },              // 3% to 5%
			{ min: 5, max: 10, value: 9 },             // 5% to 10%
			{ min: 10, max: Infinity, value: 10 }      // 10%+ to 100%+
		];

		// 15-part Exponential discretization buckets for percentage changes
		// this.priceBuckets = [
		// 	{ min: -Infinity, max: -50, value: -7 },  // -100%+ to -50%
		// 	{ min: -50, max: -25, value: -6 },        // -50% to -25%
		// 	{ min: -25, max: -12.5, value: -5 },      // -25% to -12.5%
		// 	{ min: -12.5, max: -6.25, value: -4 },    // -12.5% to -6.25%
		// 	{ min: -6.25, max: -3.125, value: -3 },   // -6.25% to -3.125%
		// 	{ min: -3.125, max: -1.5625, value: -2 }, // -3.125% to -1.5625%
		// 	{ min: -1.5625, max: -0.05, value: -1 },  // -1.5625% to -0.05%
		// 	{ min: -0.05, max: 0.05, value: 0 },      // -0.05% to 0.05% (no change)
		// 	{ min: 0.05, max: 1.5625, value: 1 },     // 0.05% to 1.5625%
		// 	{ min: 1.5625, max: 3.125, value: 2 },    // 1.5625% to 3.125%
		// 	{ min: 3.125, max: 6.25, value: 3 },      // 3.125% to 6.25%
		// 	{ min: 6.25, max: 12.5, value: 4 },       // 6.25% to 12.5%
		// 	{ min: 12.5, max: 25, value: 5 },         // 12.5% to 25%
		// 	{ min: 25, max: 50, value: 6 },           // 25% to 50%
		// 	{ min: 50, max: Infinity, value: 7 }      // 50%+ to 100%+
		// ];

		// 9-part exponential discretization buckets for percentage changes
		// this.priceBuckets = [
		// 	{ min: -Infinity, max: -50, value: -4 },
		// 	{ min: -50, max: -10, value: -3 },
		// 	{ min: -10, max: -3, value: -2 },
		// 	{ min: -3, max: -0.05, value: -1 },
		// 	{ min: -0.05, max: 0.05, value: 0 },
		// 	{ min: 0.05, max: 3, value: 1 },
		// 	{ min: 3, max: 10, value: 2 },
		// 	{ min: 10, max: 50, value: 3 },
		// 	{ min: 50, max: Infinity, value: 4 }
		// ];

		// 5-part exponential discretization buckets for percentage changes
		// this.priceBuckets = [
		// 	{ min: -Infinity, max: -10, value: -2 },  // -100%+ to -10%
		// 	{ min: -10, max: -0.05, value: -1 },      // -10% to -0.05%
		// 	{ min: -0.05, max: 0.05, value: 0 },      // -0.05% to 0.05% (no change)
		// 	{ min: 0.05, max: 10, value: 1 },         // 0.05% to 10%
		// 	{ min: 10, max: Infinity, value: 2 }      // 10%+ to 100%+
		// ];

		// 3-part buckets for percentage changes
		// this.priceBuckets = [
		// 	{ min: -Infinity, max: -0, value: -1 },   // -100%+ to -0%
		// 	{ min: -0, max: 0, value: 0 },            // -0% (no change)
		// 	{ min: 0, max: Infinity, value: 1 }    // 0% to 100%+
		// ];

		// Binary discretization: up (1) or down/flat (0)
		// this.priceBuckets = [
		// 	{ min: -Infinity, max: 0, value: 0 },     // Down or flat (<=0%)
		// 	{ min: 0, max: Infinity, value: 1 }       // Up (>0%)
		// ];
	}

	/**
	 * initialize this stock channel: open CSV and prepare iterator
	 */
	async initialize() {
		const baseDir = path.resolve(process.cwd(), 'data', 'stock');
		try { mkdirSync(baseDir, { recursive: true }); } catch {}
		this.csvPath = path.resolve(baseDir, `${this.symbol}.csv`);
		await this.loadAllRows();
		this.prepareDataIterator();
	}

	/**
	 * Load all CSV rows into memory for holdout management
	 */
	async loadAllRows() {
		this.allRows = [];
		const rl = createInterface({ input: createReadStream(this.csvPath), crlfDelay: Infinity });

		for await (const line of rl) {
			const parts = String(line).trim().split(',');
			if (parts.length < 2) throw new Error(`${this.symbol}: Invalid CSV line: ${value}`);
			const price = parseFloat(parts[0]);
			const volume = parseFloat(parts[1]);
			if (Number.isNaN(price) || Number.isNaN(volume)) throw new Error(`${this.symbol}: Invalid CSV values: '${value}'`);
			this.allRows.push({ price, volume });
		}

		if (this.debug) console.log(`${this.symbol}: Loaded ${this.allRows.length} rows from CSV`);
	}

	/**
	 * Prepare data iterator based on training/prediction mode
	 */
	prepareDataIterator() {

		// Training: use all rows except holdout
		if (this.isTrainingMode) {
			if (this.holdoutRows > 0) this.dataRows = this.allRows.slice(0, -this.holdoutRows);
			else this.dataRows = this.allRows; // Use all rows if no holdout
		}
		// Prediction: use only holdout rows
		else {
			if (this.holdoutRows > 0) this.dataRows = this.allRows.slice(-this.holdoutRows);
			else this.dataRows = []; // No prediction data if no holdout
		}

		this.currentRowIndex = 0;
		if (this.debug) console.log(`${this.symbol}: ${this.isTrainingMode ? 'Training' : 'Prediction'} mode - using ${this.dataRows.length} rows`);
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets trading state)
	 */
	resetEpisode() {

		// Reset trading state
		this.owned = false;
		this.entryPrice = null;
		this.holdingFrames = 0;
		this.hasTraded = false;
		this.previousPrice = null;
		this.previousVolume = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Reset episode metrics
		this.totalProfit = 0;
		this.totalLoss = 0;
		this.totalTrades = 0;
		this.profitableTrades = 0;
		this.unrealizedProfit = 0;
		this.lastUnrealizedProfit = 0;

		// Reset continuous prediction tracking
		this.lastPredictedPrice = null;
		this.pricePredictionErrors = [];

		// Reset data iterator to start from beginning
		this.prepareDataIterator();
	}

	/**
	 * Set prediction mode (uses only holdout rows)
	 */
	setPredictionMode() {
		this.isTrainingMode = false;
		this.prepareDataIterator();
	}

	/**
	 * Reads the next data row from the prepared dataset
	 * Returns null when all rows are consumed
	 */
	readNextRow() {
		if (this.currentRowIndex >= this.dataRows.length) return null;
		return this.dataRows[this.currentRowIndex++];
	}

	/**
	 * Compute discretized change inputs from previous → current and update state.
	 */
	computeChangeInputs() {
		const priceChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
		const volumeChange = ((this.currentVolume - this.previousVolume) / this.previousVolume) * 100;

		// Calculate prediction error if we had a price prediction from previous frame
		if (this.lastPredictedPrice !== null) {
			const actualChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
			const predictedChange = ((this.lastPredictedPrice - this.previousPrice) / this.previousPrice) * 100;

			// Error is the absolute difference between predicted and actual percentage changes
			const error = Math.abs(actualChange - predictedChange);
			this.pricePredictionErrors.push(error);

			// Calculate average error for this channel
			const avgError = this.pricePredictionErrors.reduce((sum, err) => sum + err, 0) / this.pricePredictionErrors.length;

			if (this.debug) {
				console.log(`${this.symbol}: Actual ${actualChange.toFixed(2)}% change → $${this.currentPrice.toFixed(2)}, Error ${error.toFixed(2)}pp, Avg Error ${avgError.toFixed(2)}pp`);
			}
			this.lastPredictedPrice = null;
		}

		if (this.debug) console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;
		return [
			{ [`${this.symbol}_price_change`]: this.discretizePercentageChange(priceChange) },
			{ [`${this.symbol}_volume_change`]: this.discretizePercentageChange(volumeChange) }
		];
	}

	/**
	 * returns the input dimensions for the channel
	 */
	getInputDimensions() {
		return [
			`${this.symbol}_price_change`, // Discretized percentage change in price
			`${this.symbol}_volume_change` // Discretized percentage change in volume
		];
	}

	/**
	 * returns the output dimensions for the channel
	 */
	getOutputDimensions() {
		return [
			`${this.symbol}_activity` // -1 for sell, +1 for buy
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
	 * Convert discretized bucket value back to approximate percentage change
	 * Uses the midpoint of the bucket range
	 */
	bucketValueToPercentage(bucketValue) {
		for (const bucket of this.priceBuckets) {
			if (bucket.value === bucketValue) {
				// Use midpoint of bucket range
				const min = bucket.min === -Infinity ? -100 : bucket.min;
				const max = bucket.max === Infinity ? 100 : bucket.max;
				return (min + max) / 2;
			}
		}
		return 0; // Default to no change
	}

	/**
	 * Get valid exploration actions based on current stock ownership
	 * Before first trade: only buy is valid
	 * After first trade: both buy and sell are valid (alternating states)
	 */
	getValidExplorationActions() {

		// Before first trade, can only buy
		if (!this.hasTraded) return [{ [`${this.symbol}_activity`]: 1 }];

		// After first trade, can do opposite of current state
		return [{ [`${this.symbol}_activity`]: (this.owned ? -1 : 1) }];
	}

	/**
	 * Get frame input data for this stock channel
	 */
	async getFrameInputs() {

		// Save current price as base for predictions BEFORE reading new price
		if (this.currentPrice !== null) this.priceForPrediction = this.currentPrice;

		// Read next data row
		const row = this.readNextRow();
		if (!row) return [];

		this.currentPrice = row.price;
		this.currentVolume = row.volume;

		// if this is the first frame, read another row so that we can start sending change stats
		if (this.previousPrice === null || this.previousVolume === null) {

			// seed baseline from the first observed row
			this.previousPrice = this.currentPrice;
			this.previousVolume = this.currentVolume;

			// read the next data row
			const nextRow = this.readNextRow();
			if (!nextRow) return [];

			this.currentPrice = nextRow.price;
			this.currentVolume = nextRow.volume;

			// compute and return discretized changes based on baseline → next
			return this.computeChangeInputs();
		}

		// Increment holding counter for current position
		this.holdingFrames++;

		// Compute and return discretized changes
		return this.computeChangeInputs();
	}

	/**
	 * Update unrealized profit/loss metrics for owned positions
	 * This is just for display/tracking purposes - actual profit/loss is recorded when selling
	 */
	updateUnrealizedProfitLoss() {

		// if we have not bought anything yet, initialize
		if (!this.owned || !this.entryPrice || !this.currentPrice) {
			this.unrealizedProfit = 0;
			return;
		}

		// Calculate current unrealized profit/loss (for display only)
		this.unrealizedProfit = this.currentPrice - this.entryPrice;
		this.lastUnrealizedProfit = this.unrealizedProfit;
	}

	/**
	 * Get feedback based on price movement using multiplicative reward factor
	 * For owned stocks: multiply by (new_price / old_price)
	 * For sold stocks: multiply by (old_price / new_price) - inverse relationship
	 * Returns 1.0 for no feedback (neutral)
	 */
	async getFeedback() {

		// Need both current and previous price for calculation
		const currentPrice = this.currentPrice;
		if (currentPrice === null || this.previousPrice === null) return 1.0;

		// No feedback if no price movement or haven't traded yet
		if (currentPrice === this.previousPrice || !this.hasTraded) return 1.0;

		// Update unrealized profit/loss metrics every frame
		this.updateUnrealizedProfitLoss();

		let rewardFactor;

		if (this.owned) {
			// For owned stocks: reward factor = new_price / old_price
			// If price goes up, factor > 1.0 (reward increases)
			// If price goes down, factor < 1.0 (reward decreases)
			rewardFactor = currentPrice / this.previousPrice;

			if (this.debug) {
				const totalChange = currentPrice - this.entryPrice;
				const percentChange = (totalChange / this.entryPrice) * 100;
				const recentChange = currentPrice - this.previousPrice;

				console.log(`${this.symbol}: OWNED - Price ${this.previousPrice.toFixed(2)} → ${currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
				console.log(`${this.symbol}: Reward factor: ${rewardFactor.toFixed(4)} | Total P&L: ${percentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
			}
		}
		else {
			// For sold stocks: provide inverse feedback
			// If price goes up after selling, factor < 1.0 (penalty for selling too early)
			// If price goes down after selling, factor > 1.0 (reward for good timing)
			rewardFactor = this.previousPrice / currentPrice;

			if (this.debug) {
				const totalChange = this.entryPrice - currentPrice; // Profit from selling high and price going lower
				const percentChange = (totalChange / this.entryPrice) * 100;
				const recentChange = currentPrice - this.previousPrice;

				console.log(`${this.symbol}: SOLD - Price ${this.previousPrice.toFixed(2)} → ${currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
				console.log(`${this.symbol}: Reward factor: ${rewardFactor.toFixed(4)} | Opportunity P&L: ${percentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
			}
		}

		return rewardFactor;
	}

	/**
	 * Resolve conflicts between multiple stock predictions
	 * For stocks: only one action can be taken (buy OR sell, not both)
	 * Prioritize price predictions over volume predictions, then select by strength
	 * Also calculates continuous price prediction (weighted average of predicted prices)
	 * IMPORTANT: Uses previousPrice (not currentPrice) to avoid lookahead bias - we're predicting
	 * the NEXT frame's price based on data available BEFORE reading the next frame.
	 * @returns {Array} - Array with single selected prediction
	 */
	resolveConflicts(predictions) {
		if (!predictions || predictions.length === 0) {
			this.lastPredictedPrice = null;
			return [];
		}

		const priceChangeDim = `${this.symbol}_price_change`;

		// Filter to only predictions that have price_change coordinate (not volume)
		const pricePredictions = predictions.filter(pred => priceChangeDim in pred.coordinates);

		// Calculate continuous price prediction as weighted average of predicted prices
		// Use priceForPrediction (the price from the CURRENT frame, which is the base for predicting NEXT frame)
		if (pricePredictions.length > 0 && this.priceForPrediction !== null) {
			let totalWeightedPrice = 0;
			let totalStrength = 0;
			const bucketDetails = [];

			for (const pred of pricePredictions) {
				const bucketValue = pred.coordinates[priceChangeDim];
				const percentageChange = this.bucketValueToPercentage(bucketValue);
				const predictedPrice = this.priceForPrediction * (1 + percentageChange / 100);
				totalWeightedPrice += predictedPrice * pred.strength;
				totalStrength += pred.strength;
				bucketDetails.push(`B${bucketValue}(${percentageChange.toFixed(2)}%):${pred.strength.toFixed(1)}`);
			}

			this.lastPredictedPrice = totalStrength > 0 ? totalWeightedPrice / totalStrength : null;
			if (this.lastPredictedPrice !== null && this.debug) {
				const predictedChange = ((this.lastPredictedPrice - this.priceForPrediction) / this.priceForPrediction) * 100;
				console.log(`${this.symbol}: Predicted ${predictedChange.toFixed(2)}% change (${bucketDetails.join(', ')}) → $${this.lastPredictedPrice.toFixed(2)} from $${this.priceForPrediction.toFixed(2)}`);
			}
		}
		else this.lastPredictedPrice = null;

		// Return strongest prediction for EACH dimension (price and volume separately)
		const volumeChangeDim = `${this.symbol}_volume_change`;
		const volumePredictions = predictions.filter(pred => volumeChangeDim in pred.coordinates);

		const result = [];

		// Add the strongest price prediction if any
		if (pricePredictions.length > 0) {
			let strongestPrice = pricePredictions[0];
			for (const pred of pricePredictions) if (pred.strength > strongestPrice.strength) strongestPrice = pred;
			result.push(strongestPrice);
		}

		// Add the strongest volume prediction if any
		if (volumePredictions.length > 0) {
			let strongestVolume = volumePredictions[0];
			for (const pred of volumePredictions) if (pred.strength > strongestVolume.strength) strongestVolume = pred;
			result.push(strongestVolume);
		}

		if (this.debug) {
			console.log(`${this.symbol}: Resolved ${predictions.length} predictions → returning ${result.length} (${pricePredictions.length > 0 ? 'price' : ''}${pricePredictions.length > 0 && volumePredictions.length > 0 ? '+' : ''}${volumePredictions.length > 0 ? 'volume' : ''})`);
		}

		return result;
	}

	/**
	 * Get continuous prediction error metrics for price predictions since last call
	 * Returns only NEW errors since last call, then clears the array
	 * @returns {Object} - { totalError: number, count: number } or null if no predictions
	 */
	getPredictionMetrics() {
		if (this.pricePredictionErrors.length === 0) return null;

		// Calculate total error for all predictions since last call
		const totalError = this.pricePredictionErrors.reduce((sum, err) => sum + err, 0);
		const count = this.pricePredictionErrors.length;

		// Clear the array so next call only returns new errors
		this.pricePredictionErrors = [];

		return {
			totalError,
			count,
			symbol: this.symbol
		};
	}

	/**
	 * Execute stock actions based on brain output coordinates
	 */
	async executeOutputs(coordinates) {

		if (!coordinates || Object.keys(coordinates).length === 0) return;

		// Extract activity value
		const activityValue = coordinates[`${this.symbol}_activity`];
		if (activityValue === undefined) return;

		// Get current price for position tracking
		const currentPrice = this.currentPrice;

		// Positive activity = buy signal (+1)
		if (activityValue > 0) {

			// if we already own the stock, nothing to do - just log it
			if (this.owned) {
				if (this.debug) console.log(`${this.symbol}: BUY SIGNAL IGNORED - Already owned at $${this.entryPrice}`);
				return;
			}

			// buy stock per request coming from the brain
			this.owned = true;
			this.entryPrice = currentPrice;
			this.holdingFrames = 0; // Reset holding counter
			this.hasTraded = true; // Mark that we've made our first trade

			// Reset unrealized profit tracking for new position
			this.unrealizedProfit = 0;
			this.lastUnrealizedProfit = 0;

			// Track trade metrics
			this.totalTrades++;

			if (this.debug) console.log(`${this.symbol}: EXECUTED BUY at $${currentPrice} (activity: ${activityValue})`);
		}
		// Negative activity = sell signal (-1)
		else if (activityValue < 0) {

			// if we don't own the stock, nothing to do - just log it
			if (!this.owned) {
				if (this.debug) console.log(`${this.symbol}: SELL SIGNAL IGNORED - Not owned`);
				return;
			}

			// Sell owned stock - calculate realized profit/loss
			const profit = currentPrice - this.entryPrice;
			const percentReturn = (profit / this.entryPrice) * 100;

			// Add realized profit/loss to totals
			if (profit > 0) {
				this.totalProfit += profit;
				this.profitableTrades++;
			} else if (profit < 0) {
				this.totalLoss += Math.abs(profit);
			}

			if (this.debug) {
				console.log(`${this.symbol}: EXECUTED SELL at $${currentPrice} (activity: ${activityValue})`);
				console.log(`${this.symbol}: Realized Profit/Loss: $${profit.toFixed(2)} (${percentReturn.toFixed(2)}%) over ${this.holdingFrames} frames`);
				console.log(`${this.symbol}: Episode totals: Profit $${this.totalProfit.toFixed(2)}, Loss $${this.totalLoss.toFixed(2)}, Net $${(this.totalProfit - this.totalLoss).toFixed(2)}`);
			}

			// Reset unrealized profit tracking since position is closed
			this.unrealizedProfit = 0;
			this.lastUnrealizedProfit = 0;

			// Switch to sold state
			this.owned = false;
			this.entryPrice = currentPrice; // Track sell price for sold position feedback
			this.holdingFrames = 0; // Reset holding counter
		}
	}
}