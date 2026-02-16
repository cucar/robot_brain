import Channel from './channel.js';
import { Dimension } from './dimension.js';
import { readFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	// total cash shared across all stock channel instances
	static initialCapital = 1000;
	static cash = StockChannel.initialCapital;

	// data configuration shared across all stock channel instances
	static holdoutRows = 0; // Number of rows to hold out from end (set by runtime options)
	static offsetRows = 0; // Number of rows to skip from start (set by runtime options)

	/**
	 * constructor for the stock channel - dimensions are given when loading from database
	 */
	constructor(name, debug, id = null, dimensions = null) {
		super(name, debug, id, dimensions);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// initialize dimensions
		this.initializeDimensions(dimensions);

		// initialize data to be read
		this.initializeData();

		// initialize buckets to be used for discretizing price and volume changes
		this.initializeBuckets();

		// initialize context
		this.resetContext(false);
	}

	/**
	 * initialize dimensions - dimensions are given when loading from database
	 */
	initializeDimensions(dimensions) {

		// Create or use provided dimension objects for this channel
		if (dimensions && dimensions.length > 0) {

			// Loading from database - use provided dimensions
			this.priceChangeDim = dimensions.find(d => d.name === `${this.symbol}_price_change`);
			this.volumeChangeDim = dimensions.find(d => d.name === `${this.symbol}_volume_change`);
			this.activityDim = dimensions.find(d => d.name === `${this.symbol}_activity`);

			// Validate all required dimensions exist
			if (!this.priceChangeDim || !this.volumeChangeDim || !this.activityDim)
				throw new Error(`StockChannel ${name}: Missing required dimensions in database`);
		}
		// New channel - create dimensions with auto-increment IDs
		else {
			this.priceChangeDim = new Dimension(`${this.symbol}_price_change`);
			this.volumeChangeDim = new Dimension(`${this.symbol}_volume_change`);
			this.activityDim = new Dimension(`${this.symbol}_activity`);
		}
	}

	/**
	 * Static method to initialize channel class with runtime options
	 * Called once before any channel instances are created
	 */
	static initialize(options = {}) {
		if (options.holdoutRows !== undefined) StockChannel.holdoutRows = options.holdoutRows;
		if (options.offsetRows !== undefined) StockChannel.offsetRows = options.offsetRows;
	}

	/**
	 * initialize data to be read from CSV file
	 */
	initializeData() {

		// Store all CSV rows for holdout/offset management
		this.allRows = [];

		// CSV reading state
		this.csvPath = null;
		this.rl = null;

		// Load CSV data
		const baseDir = path.resolve(process.cwd(), 'data', 'stock');
		try { mkdirSync(baseDir, { recursive: true }); } catch {}
		this.csvPath = path.resolve(baseDir, `${this.symbol}.csv`);
		this.loadAllRows();

		// Prepare data iterator
		this.prepareDataIterator();
	}

	/**
	 * initialize buckets to be used for discretizing price and volume changes
	 */
	initializeBuckets() {

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
	 * owned shorthand property
	 */
	get owned() {
		return this.shares > 0;
	}

	/**
	 * Static method to reset channel-level context (shared state across all instances)
	 * Called once per episode reset before individual channel resetContext calls
	 */
	static resetChannelContext() {
		StockChannel.cash = StockChannel.initialCapital;
	}

	/**
	 * Load all CSV rows into memory for holdout management
	 */
	loadAllRows() {
		this.allRows = [];
		const content = readFileSync(this.csvPath, 'utf-8');
		const lines = content.split('\n');

		for (const line of lines) {
			const trimmed = String(line).trim();
			if (!trimmed) continue;
			const parts = trimmed.split(',');
			if (parts.length < 2) throw new Error(`${this.symbol}: Invalid CSV line: ${line}`);
			const price = parseFloat(parts[0]);
			const volume = parseFloat(parts[1]);
			if (Number.isNaN(price) || Number.isNaN(volume)) throw new Error(`${this.symbol}: Invalid CSV values: '${line}'`);
			this.allRows.push({ price, volume });
		}

		if (this.debug) console.log(`${this.symbol}: Loaded ${this.allRows.length} rows from CSV`);
	}

	/**
	 * Prepare data iterator based on offset and holdout configuration
	 */
	prepareDataIterator() {

		// Calculate start and end indices using static properties
		const startIndex = StockChannel.offsetRows;
		const endIndex = StockChannel.holdoutRows > 0 ? this.allRows.length - StockChannel.holdoutRows : this.allRows.length;

		// Extract the slice
		this.dataRows = this.allRows.slice(startIndex, endIndex);

		this.currentRowIndex = 0;
		if (this.debug) console.log(`${this.symbol}: Using rows ${startIndex} to ${endIndex - 1} (${this.dataRows.length} rows)`);
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets trading state)
	 */
	resetContext() {

		// Reset trading state
		this.shares = 0;
		this.investment = 0; // Total amount invested in current position
		this.totalTrades = 0; // Total number of trades in current episode
		this.previousPrice = null;
		this.previousVolume = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Reset data iterator to start from beginning
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
	 * Static method for coordinated execution across all stock channels
	 * Handles portfolio allocation before executing individual channel actions
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @param {Map<string, Array>} actionsMap - Map of channel name to action data
	 */
	static async executeChannelActions(channels, actionsMap) {

		// nothing to do if there are no actions
		if (actionsMap.size === 0) return;

		// Calculate portfolio allocations
		const allocations = this.getAllocations(channels, actionsMap);

		// FIRST PASS: Execute sells to free up cash
		for (const [channelName, allocation] of allocations) {
			const channel = channels.get(channelName);

			// brain wants out - sell all shares if we have any
			if (allocation.action === POSITION_OUT) {
				if (channel.shares > 0) await channel.executeSell(channel.shares);
				continue;
			}

			// brain wants to own the stock, but how many shares? sell excess shares if current value exceeds target
			const targetShares = Math.floor(allocation.amount / channel.currentPrice);
			const sharesToSell = channel.shares - targetShares;
			if (sharesToSell > 0) await channel.executeSell(sharesToSell);
		}

		// SECOND PASS: Execute buys using freed cash
		for (const [channelName, allocation] of allocations) {
			const channel = channels.get(channelName);

			// only buy if we want to own the stock
			if (allocation.action !== POSITION_OWN) continue;

			// Buy additional shares if current value is below target
			const targetShares = Math.floor(allocation.amount / channel.currentPrice);
			const sharesToBuy = targetShares - channel.shares;
			if (sharesToBuy > 0) await channel.executeBuy(sharesToBuy);
		}
	}

	/**
	 * Calculate portfolio allocations for stock actions based on total portfolio value
	 * Uses softmax (exponential) weighting to handle negative rewards naturally
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @param {Map<string, Array>} actionsMap - Map of channel name to action data
	 * @returns {Map} - Map of channel name to { action, amount } allocation
	 */
	static getAllocations(channels, actionsMap) {

		// Calculate total portfolio value (cash + all current holdings)
		let totalPortfolioValue = this.cash;
		for (const [, channel] of channels)
			totalPortfolioValue += channel.shares * channel.currentPrice;

		// Collect all actions with their expected rewards
		const allActions = [];
		for (const [channelName, actions] of actionsMap) {
			const channel = channels.get(channelName);
			const actionData = actions[0]; // Single action per stock channel
			const action = actionData.coordinates[`${channel.symbol}_activity`];
			allActions.push({
				channelName: channelName,
				reward: actionData.reward,
				isOwn: action === POSITION_OWN
			});
		}

		// Filter to only POSITION_OWN actions for allocation
		const ownActions = allActions.filter(a => a.isOwn);

		// Calculate softmax weights - exp(reward[i]) / sum(exp(reward[j]))
		const expRewards = ownActions.map(a => ({ ...a, expReward: a.strength * Math.exp(a.reward) }));
		const totalExpReward = expRewards.reduce((sum, a) => sum + a.expReward, 0);

		// Allocate portfolio value proportional to softmax weights
		const allocations = new Map();
		for (const action of allActions) {
			if (action.isOwn) {
				const expAction = expRewards.find(e => e.channelName === action.channelName);
				const amount = totalExpReward > 0
					? (expAction.expReward / totalExpReward) * totalPortfolioValue
					: (totalPortfolioValue / ownActions.length);
				allocations.set(action.channelName, { action: POSITION_OWN, amount });
			}
			// POSITION_OUT gets 0 allocation
			else allocations.set(action.channelName, { action: POSITION_OUT, amount: 0 });
		}

		// Set OUT allocation for channels not in actionsMap (no brain prediction)
		for (const [channelName] of channels)
			if (!allocations.has(channelName))
				allocations.set(channelName, { action: POSITION_OUT, amount: 0 });

		return allocations;
	}

	/**
	 * Get portfolio-level metrics across all stock channels
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @returns {Object} - Portfolio metrics including total profit and per-channel unrealized profit
	 */
	static getPortfolioMetrics(channels) {
		let totalInvestments = 0;
		const channelProfits = new Map();

		for (const [channelName, channel] of channels) {
			const currentValue = channel.shares * channel.currentPrice;
			totalInvestments += channel.investment || 0;

			// Channel unrealized profit: current value - amount invested
			const channelProfit = currentValue - (channel.investment || 0);
			channelProfits.set(channelName, channelProfit);
		}

		// Total profit: (current cash + current investments) - original capital
		const totalProfit = (this.cash + totalInvestments) - this.initialCapital;

		return {
			cash: this.cash,
			totalInvestments,
			totalProfit,
			channelProfits
		};
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
	 * Get feedback based on price movement
	 * Returns additive reward (0 = neutral, positive = good, negative = bad):
	 * - Owned: positive if price went up, negative if price went down
	 * - Not owned: positive if price went down (good timing), negative if price went up (missed opportunity)
	 */
	async getRewards(actions) {

		// Need both current and previous price for calculation
		if (this.currentPrice === null || this.previousPrice === null) return 0;

		// if there were no actions, nothing to reward
		if (actions.length === 0) return 0;

		// Calculate percentage change
		const percentChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;

		// For owned stocks: positive change = positive reward
		// For not owned: negative change = positive reward (good timing on selling)
		const actionData = actions[0]; // Single action per stock channel
		const lastAction = actionData.coordinates[`${this.symbol}_activity`];
		const reward = (lastAction === POSITION_OWN) ? percentChange : -percentChange;
		if (this.debug) this.debugRewards(reward);
		return reward;
	}

	/**
	 * Debug output for reward calculation
	 */
	debugRewards(reward) {
		const recentChange = this.currentPrice - this.previousPrice;
		const currentValue = this.shares * this.currentPrice;
		const channelProfit = currentValue - this.investment;

		if (this.owned) {
			console.log(`${this.symbol}: OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${reward.toFixed(2)} | Unrealized P&L: ${channelProfit >= 0 ? '+' : ''}$${channelProfit.toFixed(2)}`);
		}
		else {
			console.log(`${this.symbol}: NOT OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${reward.toFixed(2)}`);
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
	 * Get current unrealized profit/loss for this channel
	 * Channel profit = current value - amount invested
	 * @returns {Object} - { value: number, label: string, format: string }
	 */
	getOutputPerformanceMetrics() {

		// Channel unrealized profit: current value - investment
		const currentValue = this.shares * this.currentPrice;
		const channelProfit = currentValue - this.investment;

		return {
			value: channelProfit,
			label: this.symbol,
			format: 'currency'
		};
	}

	/**
	 * Execute a buy action
	 * @param {number} sharesToBuy - Number of shares to buy
	 */
	async executeBuy(sharesToBuy) {

		const cost = sharesToBuy * this.currentPrice;

		// Check if we have enough cash
		if (StockChannel.cash < cost)
			throw new Error(`${this.symbol}: Insufficient cash to buy ${sharesToBuy} shares at $${this.currentPrice} (need $${cost.toFixed(2)}, have $${StockChannel.cash.toFixed(2)})`);

		// Deduct cash
		StockChannel.cash -= cost;

		// Add shares and track investment
		this.shares += sharesToBuy;
		this.investment += cost;

		// Track trade count
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: BOUGHT ${sharesToBuy} shares @ $${this.currentPrice.toFixed(2)} = $${cost.toFixed(2)} | Cash: $${StockChannel.cash.toFixed(2)}`);
	}

	/**
	 * Execute a sell action
	 * @param {number} sharesToSell - Number of shares to sell
	 */
	async executeSell(sharesToSell) {

		if (sharesToSell > this.shares)
			throw new Error(`${this.symbol}: Cannot sell ${sharesToSell} shares, only have ${this.shares}`);

		const proceeds = sharesToSell * this.currentPrice;
		const costBasis = (this.investment / this.shares) * sharesToSell;

		// Add cash
		StockChannel.cash += proceeds;

		// Reduce shares and investment
		this.shares -= sharesToSell;
		this.investment -= costBasis;

		// Track trade count
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: SOLD ${sharesToSell} shares @ $${this.currentPrice.toFixed(2)} = $${proceeds.toFixed(2)} | Cash: $${StockChannel.cash.toFixed(2)}`);
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