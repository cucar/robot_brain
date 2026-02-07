import Channel from './channel.js';
import { Dimension } from './dimension.js';
import { createReadStream, mkdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// Create dimension objects for this channel
		this.priceChangeDim = new Dimension(`${this.symbol}_price_change`);
		this.volumeChangeDim = new Dimension(`${this.symbol}_volume_change`);
		this.activityDim = new Dimension(`${this.symbol}_activity`);

		// Hyperparameters
		this.rewardAmplification = 1; // Power to raise reward ratios to (higher = stronger rewards/penalties)

		// State tracking
		this.owned = false; // true = owned, false = sold (after first buy)
		this.entryPrice = null; // Price when we bought (for owned) or sold (for sold)
		this.holdInFrames = 0; // How long we've been in owned state (for trigger-to-action)
		this.previousPrice = null; // Track previous price for change calculation
		this.previousVolume = null; // Track previous volume for change calculation

		// Episode metrics tracking
		this.initializeEpisodeMetrics();

		// Holdout configuration
		this.holdoutRows = 0; // Number of rows to hold out from training (set by job)
		this.isTrainingMode = true; // true = training (skip holdout rows), false = prediction (use only holdout rows)
		this.allRows = []; // Store all CSV rows for holdout management

		// CSV reading state
		this.csvPath = null;
		this.rl = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Unified discretization buckets for percentage changes (used for both price and volume)

		// Fine-grained for typical stock movements (-2% to +2%), exponential for extremes, extended for volume volatility
		// this.changeBuckets = [
		// 	{ min: -Infinity, max: -100, value: -20 }, // -100%+
		// 	{ min: -100, max: -90, value: -19 },       // -100% to -90%
		// 	{ min: -90, max: -80, value: -18 },        // -90% to -80%
		// 	{ min: -80, max: -70, value: -17 },        // -80% to -70%
		// 	{ min: -70, max: -60, value: -16 },        // -70% to -60%
		// 	{ min: -60, max: -50, value: -15 },        // -60% to -50%
		// 	{ min: -50, max: -40, value: -14 },        // -50% to -40%
		// 	{ min: -40, max: -30, value: -13 },        // -40% to -30%
		// 	{ min: -30, max: -20, value: -12 },        // -30% to -20%
		// 	{ min: -20, max: -10, value: -11 },        // -20% to -10%
		// 	{ min: -10, max: -5, value: -9 },          // -10% to -5%
		// 	{ min: -5, max: -3, value: -8 },           // -5% to -3%
		// 	{ min: -3, max: -2, value: -7 },           // -3% to -2%
		// 	{ min: -2, max: -1, value: -6 },           // -2% to -1%
		// 	{ min: -1, max: -0.5, value: -5 },         // -1% to -0.5%
		// 	{ min: -0.5, max: -0.2, value: -4 },       // -0.5% to -0.2%
		// 	{ min: -0.2, max: -0.05, value: -3 },      // -0.2% to -0.05%
		// 	{ min: -0.05, max: -0.01, value: -2 },     // -0.05% to -0.01%
		// 	{ min: -0.01, max: 0.01, value: 0 },       // -0.01% to 0.01% (no change)
		// 	{ min: 0.01, max: 0.05, value: 2 },        // 0.01% to 0.05%
		// 	{ min: 0.05, max: 0.2, value: 3 },         // 0.05% to 0.2%
		// 	{ min: 0.2, max: 0.5, value: 4 },          // 0.2% to 0.5%
		// 	{ min: 0.5, max: 1, value: 5 },            // 0.5% to 1%
		// 	{ min: 1, max: 2, value: 6 },              // 1% to 2%
		// 	{ min: 2, max: 3, value: 7 },              // 2% to 3%
		// 	{ min: 3, max: 5, value: 8 },              // 3% to 5%
		// 	{ min: 5, max: 10, value: 9 },             // 5% to 10%
		// 	{ min: 10, max: 20, value: 11 },           // 10% to 20%
		// 	{ min: 20, max: 30, value: 12 },           // 20% to 30%
		// 	{ min: 30, max: 40, value: 13 },           // 30% to 40%
		// 	{ min: 40, max: 50, value: 14 },           // 40% to 50%
		// 	{ min: 50, max: 60, value: 15 },           // 50% to 60%
		// 	{ min: 60, max: 70, value: 16 },           // 60% to 70%
		// 	{ min: 70, max: 80, value: 17 },           // 70% to 80%
		// 	{ min: 80, max: 90, value: 18 },           // 80% to 90%
		// 	{ min: 90, max: 100, value: 19 },          // 90% to 100%
		// 	{ min: 100, max: Infinity, value: 20 }     // 100%+
		// ];

		// Simple 2-bucket system: down (-1) and up (1), with 0 counting as down - we're looking for upside
		this.changeBuckets = [
			{ min: -Infinity, max: 0, value: -1 },     // Down (negative change)
			{ min: 0, max: Infinity, value: 1 }        // Up (zero or positive change)
		];

		// Build bucket-to-percentage mapping once (used in debug output)
		this.bucketToPercent = this.buildBucketPercentMap();
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
	 * Initialize episode metrics to zero
	 */
	initializeEpisodeMetrics() {
		this.totalProfit = 0; // Total profit from all trades in current episode
		this.totalLoss = 0; // Total loss from all trades in current episode
		this.totalTrades = 0; // Total number of trades in current episode
		this.profitableTrades = 0; // Number of profitable trades in current episode
		this.unrealizedProfit = 0; // Current unrealized profit/loss from open position
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
		this.holdInFrames = 0;
		this.previousPrice = null;
		this.previousVolume = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Reset episode metrics
		this.initializeEpisodeMetrics();

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
	 * Reads the next data row from the prepared dataset - returns true if we should continue to process
	 */
	readNextRow() {

		// return false when all rows are consumed - this will stop the processing loop
		if (this.currentRowIndex >= this.dataRows.length) return false;

		// save the current price/volume as previous before reading next row
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;

		// get the new row and update price/volume
		const row = this.dataRows[this.currentRowIndex++];
		this.currentPrice = row.price;
		this.currentVolume = row.volume;

		// return true to indicate that we have more data
		return true;
	}

	/**
	 * Compute discretized change inputs from previous → current and update state.
	 */
	computeChangeInputs() {
		const priceChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
		const volumeChange = ((this.currentVolume - this.previousVolume) / this.previousVolume) * 100;
		if (this.debug) console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		return [
			{ [`${this.symbol}_price_change`]: this.discretizeChange(priceChange) },
			{ [`${this.symbol}_volume_change`]: this.discretizeChange(volumeChange) }
		];
	}

	/**
	 * returns the input dimensions for the channel
	 */
	getEventDimensions() {
		return [ this.priceChangeDim, this.volumeChangeDim ];
	}

	/**
	 * returns the output dimensions for the channel
	 */
	getOutputDimensions() {
		return [ this.activityDim ];
	}

	/**
	 * Returns all possible actions for this channel.
	 * These are pre-created during brain init so exploration can find them.
	 */
	getActions() {
		const activityDim = `${this.symbol}_activity`;
		return [ { [activityDim]: POSITION_OUT }, { [activityDim]: POSITION_OWN } ];
	}

	/**
	 * Discretize percentage change into unified buckets
	 */
	discretizeChange(percentChange) {
		for (const bucket of this.changeBuckets)
			if (percentChange > bucket.min && percentChange <= bucket.max)
				return bucket.value;
		return 0; // Default to no change bucket
	}

	/**
	 * Convert discretized bucket value back to approximate percentage change
	 * Uses the midpoint of the bucket range
	 */
	bucketValueToPercentage(bucketValue) {
		for (const bucket of this.changeBuckets) {
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
	 * Get frame input data for this stock channel
	 */
	getFrameEvents() {

		// Read next data row - if none left, we're done
		if (!this.readNextRow()) return [];

		// if this is the first frame, read another row so that we can start sending change stats
		if (this.previousPrice === null || this.previousVolume === null) this.readNextRow();

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
	}

	/**
	 * Get feedback based on price movement using multiplicative reward factor
	 * For owned stocks: multiply by (new_price / old_price)
	 * For sold stocks: multiply by (old_price / new_price) - inverse relationship
	 * Returns 1.0 for no feedback (neutral)
	 *
	 * Returns additive reward (0 = neutral, positive = good, negative = bad):
	 * - Owned: positive if price went up, negative if price went down
	 * - Not owned: positive if price went down (good timing), negative if price went up (missed opportunity)
	 * Applies rewardAmplification to scale the reward signal.
	 */
	async getRewards() {

		// Need both current and previous price for calculation
		if (this.currentPrice === null || this.previousPrice === null) return 0;

		// Update unrealized profit/loss metrics every frame
		this.updateUnrealizedProfitLoss();

		// Calculate percentage change
		const percentChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;

		// For owned stocks: positive change = positive reward
		// For not owned: negative change = positive reward (good timing on selling)
		const reward = this.owned ? percentChange : -percentChange;

		// Amplify the reward signal and return it
		const amplifiedReward = reward * this.rewardAmplification;
		if (this.debug) this.debugRewards(amplifiedReward);
		return amplifiedReward;
	}

	/**
	 * Debug output for reward calculation
	 */
	debugRewards(amplifiedReward) {
		if (this.owned) {
			const totalChange = this.currentPrice - this.entryPrice;
			const totalPercentChange = (totalChange / this.entryPrice) * 100;
			const recentChange = this.currentPrice - this.previousPrice;

			console.log(`${this.symbol}: OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${amplifiedReward.toFixed(2)} (amp=${this.rewardAmplification}) | Total P&L: ${totalPercentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
		}
		else {
			const totalChange = this.entryPrice - this.currentPrice; // Profit from selling high and price going lower
			const totalPercentChange = (totalChange / this.entryPrice) * 100;
			const recentChange = this.currentPrice - this.previousPrice;

			console.log(`${this.symbol}: NOT OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${amplifiedReward.toFixed(2)} (amp=${this.rewardAmplification}) | Opportunity P&L: ${totalPercentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
		}
	}

	/**
	 * Calculate continuous prediction error for price predictions.
	 * Compares weighted predicted percentage change to actual percentage change.
	 * @param {Array} predictions - Array of {neuron, strength} for predicted event neurons
	 * @param {Array} actuals - Array of neurons that actually occurred
	 * @returns {number|null} - Absolute error in percentage points, or null if no price predictions
	 */
	calculatePredictionError(predictions, actuals) {
		const priceChangeDim = `${this.symbol}_price_change`;

		// Filter to price change predictions only
		const pricePredictions = predictions.filter(p => p.neuron.coordinates[priceChangeDim] !== undefined);
		if (pricePredictions.length === 0) return null;

		// Calculate weighted predicted percentage change
		let totalWeightedChange = 0;
		let totalStrength = 0;
		for (const pred of pricePredictions) {
			const bucketValue = pred.neuron.coordinates[priceChangeDim];
			const percentageChange = this.bucketValueToPercentage(bucketValue);
			totalWeightedChange += percentageChange * pred.strength;
			totalStrength += pred.strength;
		}
		if (totalStrength === 0) return null;
		const predictedChange = totalWeightedChange / totalStrength;

		// Find actual price change from actuals
		const actualNeuron = actuals.find(n => n.coordinates[priceChangeDim] !== undefined);
		if (!actualNeuron) return null;
		const actualChange = this.bucketValueToPercentage(actualNeuron.coordinates[priceChangeDim]);

		// Return absolute error in percentage points
		const error = Math.abs(predictedChange - actualChange);
		if (this.debug)
			console.log(`${this.symbol}: Predicted ${predictedChange.toFixed(2)}%, Actual ${actualChange.toFixed(2)}%, Error ${error.toFixed(2)}pp`);
		return error;
	}

	/**
	 * returns the label for an activity value
	 */
	getActionName(activityValue) {
		if (activityValue === POSITION_OWN) return 'POSITION_OWN';
		if (activityValue === POSITION_OUT) return 'POSITION_OUT';
		return 'UNKNOWN';
	}

	/**
	 * Get current profit/loss for performance tracking
	 * Returns net P&L including both realized and unrealized gains/losses
	 * @returns {Object} - { value: number, label: string, format: string }
	 */
	getOutputPerformanceMetrics() {
		// Net realized P&L from closed trades
		const realizedPL = this.totalProfit - this.totalLoss;

		// Add unrealized P&L from current open position
		const totalPL = realizedPL + this.unrealizedProfit;

		return {
			value: totalPL,
			label: this.symbol,
			format: 'currency'
		};
	}

	/**
	 * Execute stock actions based on brain output coordinates
	 * Conflict resolution ensures actions are position-aware, so we should only receive valid actions
	 * @param {Array} outputs - Frame outputs from getFrameOutputs()
	 */
	async executeOutputs(outputs) {

		// if no outputs, nothing to execute
		if (!outputs || outputs.length === 0) {
			this.lastAction = null;
			return;
		}

		// Extract coordinates from outputs array
		const output = outputs[0]; // should be single action neuron for stock channel - cannot be multiple - conflict resolution ensures it
		if (!output || Object.keys(output).length === 0) throw new Error('No coordinates in outputs');

		// extract activity value
		const activityValue = output[`${this.symbol}_activity`];
		if (activityValue === undefined) throw new Error('No activity found in outputs');

		// Determine action type: After conflict resolution, these should be position-appropriate
		if (activityValue === POSITION_OWN && !this.owned) this.executeBuy(activityValue);
		if (activityValue === POSITION_OWN && this.owned) this.executeHoldIn(activityValue);
		if (activityValue === POSITION_OUT && this.owned) this.executeSell(activityValue);
		if (activityValue === POSITION_OUT && !this.owned) this.executeHoldOut(activityValue);

		// show current status
		if (this.debug) console.log(`   ${this.symbol}: Owned: ${this.owned}, Entry Price: $${this.entryPrice?.toFixed(2) ?? 'N/A'}, Unrealized P&L: $${this.unrealizedProfit.toFixed(2)}`);
	}

	/**
	 * Execute a buy action - should only be called when not already owned (filtered in conflict resolution)
	 */
	executeBuy(activityValue) {

		// Safety check - should never happen since conflict resolution filters invalid actions
		if (this.owned) throw new Error(`${this.symbol}: BUY received when already owned - conflict resolution failed`);

		// buy stock per request coming from the brain
		this.owned = true;

		// prices are from inputs at the start of frame - we take action at the end of the frame as reaction to them
		this.entryPrice = this.currentPrice;

		// Reset unrealized profit tracking for new position
		this.unrealizedProfit = 0;
		this.holdInFrames = 0; // Reset hold-in counter (for profit reporting)

		// Track trade metrics
		this.totalTrades++;
		this.lastAction = POSITION_OWN;

		if (this.debug) console.log(`${this.symbol}: EXECUTED BUY at $${this.previousPrice} (activity: ${activityValue})`);
	}

	/**
	 * Execute a hold-in action
	 */
	executeHoldIn() {
		this.holdInFrames++; // Increment hold-in counter (for profit reporting)
		if (this.debug) console.log(`${this.symbol}: HOLD IN SIGNAL (Owned, ${this.holdInFrames} frames)`);
		this.lastAction = POSITION_OWN;
	}

	/**
	 * Execute a sell action - should only be called when owned (filtered in conflict resolution)
	 */
	executeSell(activityValue) {

		// Safety check - should never happen since conflict resolution filters invalid actions
		if (!this.owned) throw new Error(`${this.symbol}: SELL received when not owned - conflict resolution failed`);

		// Sell owned stock - calculate realized profit/loss - we are selling at the end of the frame with the price that was assumed valid throughout the frame
		const profit = this.currentPrice - this.entryPrice;
		const percentReturn = (profit / this.entryPrice) * 100;

		// Add realized profit/loss to totals
		if (profit > 0) {
			this.totalProfit += profit;
			this.profitableTrades++;
		}
		else if (profit < 0)
			this.totalLoss += Math.abs(profit);

		if (this.debug) {
			console.log(`${this.symbol}: EXECUTED SELL at $${this.previousPrice} (activity: ${activityValue})`);
			console.log(`${this.symbol}: Realized Profit/Loss: $${profit.toFixed(2)} (${percentReturn.toFixed(2)}%) over ${this.holdInFrames} frames`);
			console.log(`${this.symbol}: Episode totals: Profit $${this.totalProfit.toFixed(2)}, Loss $${this.totalLoss.toFixed(2)}, Net $${(this.totalProfit - this.totalLoss).toFixed(2)}`);
		}

		// Reset unrealized profit tracking since position is closed
		this.unrealizedProfit = 0;

		// Switch to sold state
		this.owned = false;
		this.entryPrice = this.currentPrice; // Track sell price for sold position feedback
		this.lastAction = POSITION_OUT;
	}

	/**
	 * Execute a hold-out action
	 */
	executeHoldOut() {
		if (this.debug) console.log(`${this.symbol}: HOLD OUT SIGNAL (Not Owned)`);
		this.lastAction = POSITION_OUT;
	}

	/**
	 * Format action label for debug output
	 * Converts raw coordinates to human-readable action names (e.g., "OWN", "OUT")
	 * @param {Object} coords - Coordinates object { dimension: value }
	 * @returns {string} Formatted action label
	 */
	formatActionLabel(coords) {
		const activityDim = `${this.symbol}_activity`;
		const activityVal = coords[activityDim];
		if (activityVal === POSITION_OWN) return 'OWN';
		if (activityVal === POSITION_OUT) return 'OUT';
		return JSON.stringify(coords);
	}

	/**
	 * Format coordinates string with percentage ranges where applicable
	 * Used by diagnostics for displaying event/action votes
	 * @param {string} coordsStr - Coordinates string (e.g., "dim1=val1, dim2=val2")
	 * @returns {string} Formatted coordinates with percentage ranges
	 */
	formatCoordinates(coordsStr) {
		return this.formatCoordsWithPercent(coordsStr, this.bucketToPercent);
	}

	/**
	 * Build a map from bucket values to percentage ranges for price/volume dimensions
	 */
	buildBucketPercentMap() {
		const map = new Map();
		for (const b of this.changeBuckets) {
			map.set(`${this.symbol}_price_change:${b.value}`, this.formatBucketRange(b.min, b.max));
			map.set(`${this.symbol}_volume_change:${b.value}`, this.formatBucketRange(b.min, b.max));
		}
		return map;
	}

	/**
	 * Format bucket range as readable string
	 */
	formatBucketRange(min, max) {
		if (min === -Infinity) return `<${max}%`;
		if (max === Infinity) return `>${min}%`;
		return `${min}%~${max}%`;
	}

	/**
	 * Format coordinates string with percentage ranges where applicable
	 */
	formatCoordsWithPercent(coordsStr, bucketToPercent) {
		if (!coordsStr) return '(no coords)';
		if (!bucketToPercent) return coordsStr;
		// Parse "TEST_price_change=5, TEST_volume_change=0" format
		return coordsStr.split(', ').map(part => {
			const [dimName, valStr] = part.split('=');
			const val = parseFloat(valStr);
			// Check if this dimension has a bucket mapping
			const key = `${dimName}:${val}`;
			const percentRange = bucketToPercent.get(key);
			if (percentRange) return `${dimName}=${val}(${percentRange})`;
			return part;
		}).join(', ');
	}
}