import Channel from './channel.js';
import { createReadStream, mkdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

const SELL = -1;
const HOLD_OUT = -0.5;
const HOLD_IN = 0.5;
const BUY = 1;

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// Hyperparameters
		this.rewardAmplification = 10; // Power to raise reward ratios to (higher = stronger rewards/penalties)

		// State tracking
		this.owned = false; // true = owned, false = sold (after first buy)
		this.entryPrice = null; // Price when we bought (for owned) or sold (for sold)
		this.holdingFrames = 0; // How long we've held the current position
		this.previousPrice = null; // Track previous price for change calculation
		this.previousVolume = null; // Track previous volume for change calculation

		// Deterministic exploration
		this.explorationSequence = [BUY, HOLD_IN, SELL, HOLD_OUT];
		this.explorationIndex = 0; // Current position in exploration sequence

		// exploration strategy - boltzmann or strongest selection
		this.explorationStrategy = 'strongest'; // boltzmann or strongest

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
	 * Initialize episode metrics to zero
	 */
	initializeEpisodeMetrics() {
		this.totalProfit = 0; // Total profit from all trades in current episode
		this.totalLoss = 0; // Total loss from all trades in current episode
		this.totalTrades = 0; // Total number of trades in current episode
		this.profitableTrades = 0; // Number of profitable trades in current episode
		this.unrealizedProfit = 0; // Current unrealized profit/loss from open position
		this.lastPredictedPrice = null; // Predicted price from previous frame
		this.pricePredictionErrors = []; // Array of price prediction errors for calculating metrics
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

		if (this.debug2) console.log(`${this.symbol}: Loaded ${this.allRows.length} rows from CSV`);
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
		if (this.debug2) console.log(`${this.symbol}: ${this.isTrainingMode ? 'Training' : 'Prediction'} mode - using ${this.dataRows.length} rows`);
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets trading state)
	 */
	resetEpisode() {

		// Reset trading state
		this.owned = false;
		this.entryPrice = null;
		this.holdingFrames = 0;
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

		// Calculate prediction error if we had a price prediction from previous frame
		if (this.lastPredictedPrice !== null) {
			const actualChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
			const predictedChange = ((this.lastPredictedPrice - this.previousPrice) / this.previousPrice) * 100;

			// Error is the absolute difference between predicted and actual percentage changes
			const error = Math.abs(actualChange - predictedChange);
			this.pricePredictionErrors.push(error);

			// Calculate average error for this channel
			const avgError = this.pricePredictionErrors.reduce((sum, err) => sum + err, 0) / this.pricePredictionErrors.length;

			if (this.debug2) console.log(`${this.symbol}: Actual ${actualChange.toFixed(2)}% change → $${this.currentPrice.toFixed(2)}, Error ${error.toFixed(2)}pp, Avg Error ${avgError.toFixed(2)}pp`);
			this.lastPredictedPrice = null;
		}

		if (this.debug) console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		return [
			{ [`${this.symbol}_price_change`]: this.discretizePercentageChange(priceChange) },
			{ [`${this.symbol}_volume_change`]: this.discretizePercentageChange(volumeChange) }
		];
	}

	/**
	 * returns the input dimensions for the channel
	 */
	getEventDimensions() {
		return [
			`${this.symbol}_price_change`,
			`${this.symbol}_volume_change`
		];
	}

	/**
	 * returns the output dimensions for the channel
	 */
	getOutputDimensions() {
		return [ `${this.symbol}_activity` ];
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
	 * Returns a valid exploration action based on current position
	 * Returns only feasible actions: if owned, can sell or hold-in; if not owned, can buy or hold-out
	 */
	getExplorationAction() {
		const activityDim = `${this.symbol}_activity`;
		const actions = [];
		if (this.owned) {
			actions.push({ [activityDim]: SELL });
			actions.push({ [activityDim]: HOLD_IN });
		}
		else {
			actions.push({ [activityDim]: BUY });
			actions.push({ [activityDim]: HOLD_OUT });
		}
		return actions[Math.floor(Math.random() * actions.length)];

		// NOTE: following code is deterministic to be able to troubleshoot output performance issues
		// Return deterministic action from sequence
		// const action = this.explorationSequence[this.explorationIndex];
		// this.explorationIndex = (this.explorationIndex + 1) % this.explorationSequence.length;
		// return { [activityDim]: action };
	}

	/**
	 * Get frame input data for this stock channel
	 */
	getFrameEvents() {

		// Read next data row - if none left, we're done
		if (!this.readNextRow()) return [];

		// if this is the first frame, read another row so that we can start sending change stats
		if (this.previousPrice === null || this.previousVolume === null) this.readNextRow();

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
	}

	/**
	 * Get feedback based on price movement using multiplicative reward factor
	 * For owned stocks: multiply by (new_price / old_price)
	 * For sold stocks: multiply by (old_price / new_price) - inverse relationship
	 * Returns 1.0 for no feedback (neutral)
	 *
	 * Applies rewardAmplification to amplify the reward signal:
	 * - rewardFactor = ratio^rewardAmplification
	 * - Higher amplification = stronger rewards and penalties
	 */
	async getRewards() {

		// Need both current and previous price for calculation
		if (this.currentPrice === null || this.previousPrice === null) return 1.0;

		// Update unrealized profit/loss metrics every frame
		this.updateUnrealizedProfitLoss();

		// For owned stocks: ratio = new_price / old_price
		// If price goes up, ratio > 1.0 (reward increases)
		// If price goes down, ratio < 1.0 (reward decreases)
		// For sold stocks: provide inverse feedback
		// If price goes up after selling, ratio < 1.0 (penalty for selling too early)
		// If price goes down after selling, ratio > 1.0 (reward for good timing)
		const ratio = this.owned ? (this.currentPrice / this.previousPrice) : (this.previousPrice / this.currentPrice);

		// Amplify the reward signal by raising to a power
		const rewardFactor = Math.pow(ratio, this.rewardAmplification);

		if (this.debug2) {
			if (this.owned) {
				const totalChange = this.currentPrice - this.entryPrice;
				const percentChange = (totalChange / this.entryPrice) * 100;
				const recentChange = this.currentPrice - this.previousPrice;

				console.log(`${this.symbol}: OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
				console.log(`${this.symbol}: Ratio: ${ratio.toFixed(4)} → Reward factor: ${rewardFactor.toFixed(4)} (amp=${this.rewardAmplification}) | Total P&L: ${percentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
			}
			else {
				const totalChange = this.entryPrice - this.currentPrice; // Profit from selling high and price going lower
				const percentChange = (totalChange / this.entryPrice) * 100;
				const recentChange = this.currentPrice - this.previousPrice;

				console.log(`${this.symbol}: NOT OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
				console.log(`${this.symbol}: Ratio: ${ratio.toFixed(4)} → Reward factor: ${rewardFactor.toFixed(4)} (amp=${this.rewardAmplification}) | Opportunity P&L: ${percentChange.toFixed(2)}% (${totalChange >= 0 ? '+' : ''}$${totalChange.toFixed(2)})`);
			}
		}

		return rewardFactor;
	}

	/**
	 * Resolve event predictions: select strongest for each input dimension. Also calculates continuous price prediction for accuracy tracking
	 * @param {Array} events - predictions for event dimensions
	 * @returns {Array} - strongest prediction per event dimension
	 */
	resolveEventPredictions(events) {

		// if there are no input predictions, nothing to resolve
		if (events.length === 0) {
			this.lastPredictedPrice = null;
			return [];
		}

		// Group by dimension
		const byDimension = this.groupByDimension(events);

		// Calculate continuous price prediction for tracking
		const priceChangeDim = `${this.symbol}_price_change`;
		if (byDimension.has(priceChangeDim)) this.calculateContinuousPricePrediction(byDimension.get(priceChangeDim));
		else this.lastPredictedPrice = null;

		// Select events for each dimension
		return this.selectPerDimension(byDimension);
	}

	/**
	 * Resolve action conflicts: select best action when multiple predicted
	 * @param {Array} actions - inferences for output dimensions
	 * @returns {Array} - resolved actions (one per dimension)
	 */
	resolveActionInferences(actions) {
		if (actions.length === 0) return [];

		// Group by dimension and select the action to execute
		const byDimension = this.groupByDimension(actions);
		return this.selectPerDimension(byDimension);
	}

	/**
	 * Correct invalid action inferences based on current position
	 * Maps position-agnostic actions to executable actions based on current position
	 * @param {Array} inferences - resolved inferences
	 * @returns {Array} - corrections: [{ originalNeuronId, correctedCoordinates }]
	 */
	correctActionInferences(inferences) {

		// get the action inferences
		const actions = this.getActions(inferences);
		if (actions.length === 0) return [];

		// Check each resolved action and correct if needed
		const activityDim = `${this.symbol}_activity`;
		const corrections = [];
		for (const action of actions) {

			// get the activity value
			const activity = action.coordinates[activityDim];
			if (activity === undefined) continue; // Not an action neuron

			// Determine if correction is needed based on position
			let correctedActivity = null;
			if (activity === HOLD_IN && !this.owned) correctedActivity = BUY;
			if (activity === HOLD_OUT && this.owned) correctedActivity = SELL;
			if (activity === BUY && this.owned) correctedActivity = HOLD_IN;
			if (activity === SELL && !this.owned) correctedActivity = HOLD_OUT;

			// if no correction is needed, skip
			if (correctedActivity === null) continue;

			// If correction needed, track it
			corrections.push({ originalNeuronId: action.neuron_id, correctedCoordinates: { [activityDim]: correctedActivity } });
		}

		return corrections;
	}

	/**
	 * returns the label for an activity value
	 */
	getActionName(activityValue) {
		if (activityValue === BUY) return 'BUY';
		if (activityValue === HOLD_IN) return 'HOLD_IN';
		if (activityValue === HOLD_OUT) return 'HOLD_OUT';
		if (activityValue === SELL) return 'SELL';
		return 'UNKNOWN';
	}

	/**
	 * Select one inference per dimension
	 * @param {Map} byDimension - Map of dimension name to array of inferences
	 * @returns {Array} - selected inference per dimension
	 */
	selectPerDimension(byDimension) {
		const resolved = [];
		for (const [_, inferences] of byDimension)
			resolved.push(this.explorationStrategy === 'boltzmann' ? this.boltzmannSelect(inferences) : this.findStrongest(inferences));
		return resolved;
	}

	/**
	 * Group inferences by dimension
	 * @param {Array} inferences - list of inferences
	 * @returns {Map} - Map of dimension name to array of inferences
	 */
	groupByDimension(inferences) {
		const groups = new Map();
		for (const inf of inferences)
			for (const dim of Object.keys(inf.coordinates)) {
				if (!groups.has(dim)) groups.set(dim, []);
				groups.get(dim).push(inf);
			}
		return groups;
	}

	/**
	 * Calculate continuous price prediction as weighted average
	 * Uses previousPrice as base for predicting next frame
	 * @param {Array} pricePredictions - predictions for price_change dimension
	 */
	calculateContinuousPricePrediction(pricePredictions) {
		if (pricePredictions.length === 0 || this.previousPrice === null) {
			this.lastPredictedPrice = null;
			return;
		}

		const priceChangeDim = `${this.symbol}_price_change`;
		let totalWeightedPrice = 0;
		let totalStrength = 0;
		const bucketDetails = [];

		for (const pred of pricePredictions) {
			const bucketValue = pred.coordinates[priceChangeDim];
			const percentageChange = this.bucketValueToPercentage(bucketValue);
			const predictedPrice = this.previousPrice * (1 + percentageChange / 100);

			totalWeightedPrice += predictedPrice * pred.strength;
			totalStrength += pred.strength;
			bucketDetails.push(`B${bucketValue}(${percentageChange.toFixed(2)}%):${pred.strength.toFixed(1)}`);
		}

		this.lastPredictedPrice = totalStrength > 0 ? totalWeightedPrice / totalStrength : null;

		if (this.lastPredictedPrice !== null && this.debug2) {
			const predictedChange = ((this.lastPredictedPrice - this.previousPrice) / this.previousPrice) * 100;
			console.log(`${this.symbol}: Predicted ${predictedChange.toFixed(2)}% change (${bucketDetails.join(', ')}) → $${this.lastPredictedPrice.toFixed(2)} from $${this.previousPrice.toFixed(2)}`);
		}
	}

	/**
	 * Boltzmann selection from a list of inferences
	 * Selects probabilistically using Boltzmann (softmax) distribution
	 * P(i) = exp(strength_i / T) / sum(exp(strength_j / T))
	 * Lower temperature = more deterministic (favors highest strength)
	 * Higher temperature = more random (approaches uniform distribution)
	 * @param {Array} inferences - list of inferences
	 * @param {number} temperature - temperature parameter (default: 1.0)
	 * @returns {Object|null} - selected inference or null if empty
	 */
	boltzmannSelect(inferences, temperature = 1.0) {
		if (inferences.length === 0) return null;
		if (inferences.length === 1) return inferences[0];

		// Calculate exponential weights (Boltzmann factors)
		const weights = inferences.map(inf => Math.exp(inf.strength / temperature));
		const totalWeight = weights.reduce((sum, w) => sum + w, 0);
		if (totalWeight <= 0 || !isFinite(totalWeight)) return inferences[0]; // Fallback for numerical issues

		// Sample from distribution
		const r = Math.random() * totalWeight;
		let cumulative = 0;
		for (let i = 0; i < inferences.length; i++) {
			cumulative += weights[i];
			if (r < cumulative) return inferences[i];
		}
		return inferences[inferences.length - 1]; // Fallback for floating point edge case
	}

	/**
	 * Find the strongest inference from a list
	 * @param {Array} inferences - list of inferences
	 * @returns {Object|null} - strongest inference or null if empty
	 */
	findStrongest(inferences) {
		if (inferences.length === 0) return null;
		let strongest = inferences[0];
		for (const inf of inferences) if (inf.strength > strongest.strength) strongest = inf;
		return strongest;
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
		if (activityValue === BUY) this.executeBuy(activityValue);
		if (activityValue === HOLD_IN) this.executeHoldIn(activityValue);
		if (activityValue === HOLD_OUT) this.executeHoldOut(activityValue);
		if (activityValue === SELL) this.executeSell(activityValue);

		// show current status
		if (this.diagnostic) console.log(`   ${this.symbol}: Owned: ${this.owned}, Entry Price: $${this.entryPrice?.toFixed(2) ?? 'N/A'}, Unrealized P&L: $${this.unrealizedProfit.toFixed(2)}`);
	}

	/**
	 * Execute a buy action - should only be called when not already owned (filtered in conflict resolution)
	 */
	executeBuy(activityValue) {

		// Safety check - should never happen since conflict resolution filters invalid actions
		if (this.owned) throw new Error(`${this.symbol}: BUY received when already owned - conflict resolution failed`);

		// buy stock per request coming from the brain
		this.owned = true;
		this.entryPrice = this.previousPrice; // prices are the end of day prices. we buy in the morning, at the close price of the previous day
		this.holdingFrames = 0; // Reset holding counter

		// Reset unrealized profit tracking for new position
		this.unrealizedProfit = 0;

		// Track trade metrics
		this.totalTrades++;
		this.lastAction = BUY;

		if (this.debug) console.log(`${this.symbol}: EXECUTED BUY at $${this.previousPrice} (activity: ${activityValue})`);
	}

	/**
	 * Execute a sell action - should only be called when owned (filtered in conflict resolution)
	 */
	executeSell(activityValue) {

		// Safety check - should never happen since conflict resolution filters invalid actions
		if (!this.owned) throw new Error(`${this.symbol}: SELL received when not owned - conflict resolution failed`);

		// Sell owned stock - calculate realized profit/loss - assume we are selling at the end of day price of yesterday (beginning price of today)
		const profit = this.previousPrice - this.entryPrice;
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
			console.log(`${this.symbol}: Realized Profit/Loss: $${profit.toFixed(2)} (${percentReturn.toFixed(2)}%) over ${this.holdingFrames} frames`);
			console.log(`${this.symbol}: Episode totals: Profit $${this.totalProfit.toFixed(2)}, Loss $${this.totalLoss.toFixed(2)}, Net $${(this.totalProfit - this.totalLoss).toFixed(2)}`);
		}

		// Reset unrealized profit tracking since position is closed
		this.unrealizedProfit = 0;

		// Switch to sold state
		this.owned = false;
		this.entryPrice = this.previousPrice; // Track sell price for sold position feedback
		this.holdingFrames = 0; // Reset holding counter
		this.lastAction = SELL;
	}

	/**
	 * Execute a hold-in action
	 */
	executeHoldIn() {
		if (this.debug) console.log(`${this.symbol}: HOLD IN SIGNAL (Owned)`);
		this.lastAction = HOLD_IN;
	}

	/**
	 * Execute a hold-out action
	 */
	executeHoldOut() {
		if (this.debug) console.log(`${this.symbol}: HOLD OUT SIGNAL (Not Owned)`);
		this.lastAction = HOLD_OUT;
	}
}