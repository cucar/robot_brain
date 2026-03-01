import { Channel } from './channel.js';
import { Dimension } from './dimension.js';

const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export class StockChannel extends Channel {

	// total cash shared across all stock channel instances
	static initialCapital = 15000;
	static cash = StockChannel.initialCapital;

	// trading mode - set by initialize() from command line options
	static eventTrading = false;

	// maximum number of positions to hold at once
	static maxPositions = 1;

	// maximum price limit for stocks
	static maxPrice = 5000;

	/**
	 * Initialize channel type with runtime options
	 * Called once during brain initialization
	 */
	static initialize(options) {
		StockChannel.eventTrading = options.eventTrading || false;
	}

	/**
	 * Static method to reset channel-level context (shared state across all instances)
	 * Called once per episode reset before individual channel resetContext calls
	 */
	static resetChannelContext() {
		StockChannel.cash = StockChannel.initialCapital;
	}

	/**
	 * constructor for the stock channel - dimensions are given when loading from database
	 */
	constructor(name, debug, id = null, dimensions = null) {
		super(name, debug, id);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;
		this.activityDimName = `${this.symbol}_activity`;

		// initialize dimensions
		this.initializeDimensions(dimensions);

		// training data / mode (set by setTraining)
		this.trainingData = null;
		this.trainingRow = 0;

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
			this.activityDim = dimensions.find(d => d.name === this.activityDimName);

			// Validate all required dimensions exist
			if (!this.priceChangeDim || !this.volumeChangeDim || !this.activityDim)
				throw new Error(`StockChannel ${name}: Missing required dimensions in database`);
		}
		// New channel - create dimensions with auto-increment IDs
		else {
			this.priceChangeDim = new Dimension(`${this.symbol}_price_change`);
			this.volumeChangeDim = new Dimension(`${this.symbol}_volume_change`);
			this.activityDim = new Dimension(this.activityDimName);
		}
	}

	/**
	 * Set training data for this channel - switches channel to training mode
	 * @param {Array<{price, volume}>} rows - Training data rows
	 */
	setTraining(rows) {
		this.trainingData = rows;
		this.trainingRow = 0;
	}

	/**
	 * initialize dynamic bucket categories
	 */
	initializeBuckets() {

		// price movements are best predicted from up/down - ideal split is 0.01% - favor profit making
		this.priceBoundaries = [0];

		// volume movements are best predicted as up/down
		this.volumeBoundaries = [0];

		// Build bucket-to-percentage mapping once (used in debug output)
		this.bucketToPercent = this.buildBucketPercentMap();
	}

	/**
	 * Build a map from bucket values to percentage ranges for price/volume dimensions
	 */
	buildBucketPercentMap() {
		const map = new Map();
		const categories = [
			{ dim: `${this.symbol}_price_change`,  boundaries: this.priceBoundaries },
			{ dim: `${this.symbol}_volume_change`, boundaries: this.volumeBoundaries }
		];
		for (const { dim, boundaries } of categories)
			for (let bucket = 1; bucket <= boundaries.length + 1; bucket++) {
				const { lo, hi } = this.getBucketRange(bucket, boundaries);
				map.set(`${dim}:${bucket}`, this.formatBucketRange(lo, hi));
			}
		return map;
	}

	/**
	 * Return the [lo, hi] boundary pair for a 1-indexed bucket in a category.
	 * lo = -Infinity for bucket 1; hi = Infinity for the last bucket.
	 */
	getBucketRange(bucketValue, boundaries) {
		const idx = bucketValue - 1;
		const lo = idx === 0 ? -Infinity : boundaries[idx - 1];
		const hi = idx >= boundaries.length ? Infinity : boundaries[idx];
		return { lo, hi };
	}

	/**
	 * Format bucket range as a readable string with 2 decimal places
	 */
	formatBucketRange(min, max) {
		if (min === -Infinity) return `<${max.toFixed(2)}%`;
		if (max === Infinity) return `>${min.toFixed(2)}%`;
		return `${min.toFixed(2)}%~${max.toFixed(2)}%`;
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets trading state)
	 */
	resetContext() {

		// Reset trading state
		this.shares = 0; // this is the actual number of shares we own - this is what will be adjusted by the actions
		this.investment = 0; // Total amount invested in current position
		this.totalTrades = 0; // Total number of trades in current episode
		this.previousPrice = null;
		this.previousVolume = null;
		this.currentPrice = null;
		this.currentVolume = null;
		this.lastAction = null; // Last action taken by brain

		// Reset data iterator to start from beginning
		this.trainingRow = 0;
	}

	/**
	 * Reads the next data row from the prepared dataset - returns true if we should continue to process
	 */
	readNextRow() {

		// return false when all rows are consumed - this will stop the processing loop
		if (this.trainingRow >= this.trainingData.length) return false;

		// save the current price/volume as previous before reading next row
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;

		// get the new row and update price/volume
		const row = this.trainingData[this.trainingRow++];
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
		const volumeChange = this.previousVolume === 0 ? 1000 : ((this.currentVolume - this.previousVolume) / this.previousVolume) * 100;
		if (this.debug) console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		return [
			{ [`${this.symbol}_price_change`]: this.discretizeChange(priceChange, this.priceBoundaries) },
			{ [`${this.symbol}_volume_change`]: this.discretizeChange(volumeChange, this.volumeBoundaries) }
		];
	}

	/**
	 * Discretize percentage change into unified buckets
	 */
	discretizeChange(value, boundaries) {
		for (let i = 0; i < boundaries.length; i++)
			if (value <= boundaries[i]) return i + 1;
		return boundaries.length + 1;
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
	getActionDimensions() {
		return [ this.activityDim ];
	}

	/**
	 * Returns all possible actions for this channel.
	 * These are pre-created during brain init so exploration can find them.
	 */
	getActions() {
		return [ { [this.activityDimName]: POSITION_OUT }, { [this.activityDimName]: POSITION_OWN } ];
	}

	/**
	 * returns the coordinates of the channel default action - for stock channels, this is "do nothing"
	 */
	getDefaultAction() {
		return { [this.activityDimName]: POSITION_OUT };
	}

	/**
	 * Get frame input data for this stock channel
	 */
	getFrameEvents() {

		// currently only the training mode is implemented
		if (this.trainingData === null) throw new Error(`${this.symbol}: live mode not implemented yet.`);

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
	async getRewards() {

		// Need both current and previous price for calculation
		if (this.currentPrice === null || this.previousPrice === null) return 0;

		// Calculate percentage change
		const percentChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;

		// For owned stocks: positive change = positive reward
		// For not owned: negative change = positive reward (good timing on selling)
		const reward = this.lastAction === POSITION_OWN ? percentChange : -percentChange;
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

		if (this.lastAction === POSITION_OWN) {
			console.log(`${this.symbol}: OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${reward} | Unrealized P&L: ${channelProfit >= 0 ? '+' : ''}$${channelProfit.toFixed(2)}`);
		}
		else {
			console.log(`${this.symbol}: NOT OWNED - Price ${this.previousPrice.toFixed(2)} → ${this.currentPrice.toFixed(2)} (${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(2)})`);
			console.log(`${this.symbol}: Reward: ${reward}`);
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
	 * Convert discretized bucket value back to approximate percentage change
	 * Uses the midpoint of the bucket range
	 */
	bucketValueToPercentage(bucketValue) {
		const { lo, hi } = this.getBucketRange(bucketValue, this.priceBoundaries);
		const loVal = lo === -Infinity ? hi - Math.abs(hi || 1) * 2 : lo;
		const hiVal = hi === Infinity ? lo + Math.abs(lo || 1) * 2 : hi;
		return (loVal + hiVal) / 2;
	}

	/**
	 * Static method for coordinated execution across all stock channels
	 * Handles portfolio allocation before executing individual channel actions
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @param {Map<string, Array>} actionsMap - Map of channel name to action data
	 * @param {Map<string, Array>} eventsMap - Map of channel name to event inference data
	 */
	static async executeChannelActions(channels, actionsMap, eventsMap) {

		// nothing to do if there are no actions
		if (actionsMap.size === 0) return;

		// Save last actions for rewarding in the next frame
		this.saveLastActions(channels, actionsMap);

		// Calculate portfolio allocations
		const allocations = this.getAllocations(channels, actionsMap, eventsMap);

		// Generate action plan based on differential between ideal and current allocations
		const actionPlan = this.getActionPlan(channels, allocations);

		// Execute the action plan
		await this.executeActionPlan(actionPlan);
	}

	/**
	 * Save last action for each channel for tracking purposes
	 */
	static saveLastActions(channels, actionsMap) {
		for (const [channelName, actions] of actionsMap) {
			if (actions.length === 0) continue;
			const channel = channels.get(channelName);
			const actionData = actions[0];
			channel.lastAction = actionData.coordinates[`${channel.symbol}_activity`];
		}
	}

	/**
	 * Calculate portfolio allocations for stock actions based on total portfolio value
	 * Uses softmax (exponential) weighting to handle negative rewards naturally
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @param {Map<string, Array>} actionsMap - Map of channel name to action data
	 * @param {Map<string, Array>} eventsMap - Map of channel name to event inference data
	 * @returns {Map} - Map of channel name to { action, amount } allocation
	 */
	static getAllocations(channels, actionsMap, eventsMap) {

		// Calculate total portfolio value (cash + all current holdings)
		const totalValue = this.getTotalValue(channels);

		// Collect all actions with their weights for allocations
		const actions = this.getActionsWeights(channels, actionsMap, eventsMap);

		// Allocate portfolio value proportional to the rewards
		const allocations = this.distributeAllocations(channels, actions, totalValue);

		// Set OUT allocation for channels not in actionsMap (no brain prediction)
		this.setMissingChannelAllocations(channels, allocations);

		return allocations;
	}

	/**
	 * returns total portfolio value (cash + all current holdings)
	 */
	static getTotalValue(channels) {
		let totalPortfolioValue = this.cash;
		for (const [, channel] of channels)
			totalPortfolioValue += channel.shares * channel.currentPrice;
		return totalPortfolioValue;
	}

	/**
	 * Calculate action weights for each channel
	 * - Event trading mode: uses event consensus inferences (price forecasts) to determine actions
	 * - Action trading mode: uses brain's action decisions with their rewards
	 */
	static getActionsWeights(channels, actionsMap, eventsMap) {
		const allActions = [];

		// Event-based trading: trade based on price predictions
		if (this.eventTrading) {
			for (const [channelName, events] of eventsMap) {
				const channel = channels.get(channelName);

				// Find price change prediction in event inferences
				const priceEvent = events.find(e => e.coordinates[`${channel.symbol}_price_change`] !== undefined);
				if (!priceEvent) continue;

				// Get the predicted price change bucket value (1 = down, 2 = up)
				// Determine action: bucket 2 (up) → buy, bucket 1 (down) → sell - use strength as weights
				const bucketValue = priceEvent.coordinates[`${channel.symbol}_price_change`];
				const action = bucketValue === 2 ? POSITION_OWN : POSITION_OUT;
				allActions.push({ channelName, isOwn: action === POSITION_OWN });
			}
		}
		// Action-based trading: use brain's action decisions
		else {
			for (const [channelName, actions] of actionsMap) {
				const channel = channels.get(channelName);
				const actionData = actions[0]; // Single action per stock channel
				const action = actionData.coordinates[`${channel.symbol}_activity`];
				allActions.push({ channelName, isOwn: action === POSITION_OWN });
			}
		}

		return allActions;
	}

	/**
	 * Allocate portfolio value proportional to softmax weights
	 */
	static distributeAllocations(channels, actions, totalValue) {

		// get the actions that want to own a stock
		let ownActions = actions.filter(a => a.isOwn);

		// limit to N positions
		if (ownActions.length > this.maxPositions) {
			ownActions = ownActions.filter(a => channels.get(a.channelName).currentPrice < this.maxPrice); // filter by price
			ownActions = ownActions.slice(0, this.maxPositions);
		}

		// create set of channel names that made the cut
		const ownChannels = new Set(ownActions.map(a => a.channelName));

		// allocate the stocks to portfolio - if we want to own the stock AND it's in top N, allocate it based on its weight, otherwise, 0
		const allocations = new Map();
		for (const action of actions) allocations.set(action.channelName, {
			action: (action.isOwn && ownChannels.has(action.channelName)) ? POSITION_OWN : POSITION_OUT,
			amount: (action.isOwn && ownChannels.has(action.channelName)) ? (1 / ownActions.length) * totalValue : 0
		});
		return allocations;
	}

	/**
	 * Set OUT allocation for channels not in actionsMap (no brain prediction)
	 */
	static setMissingChannelAllocations(channels, allocations) {
		for (const [channelName] of channels)
			if (!allocations.has(channelName))
				allocations.set(channelName, { action: POSITION_OUT, amount: 0 });
	}

	/**
	 * Generate action plan based on differential between ideal allocations and current holdings
	 * @param {Map<string, StockChannel>} channels - Map of channel name to channel instance
	 * @param {Map} allocations - Map of channel name to { action, amount } allocation
	 * @returns {Object} - Action plan with { sells: [...], buys: [...] }
	 */
	static getActionPlan(channels, allocations) {
		const sells = [];
		const buys = [];
		const state = { remainingCash: this.cash, cheapestOwnChannel: null };

		for (const [channelName, allocation] of allocations) {
			const channel = channels.get(channelName);

			// brain wants out - sell all shares if we have any
			if (allocation.action === POSITION_OUT) {
				if (channel.shares > 0) this.planSellAll(channel, sells, state);
				continue;
			}

			// track the cheapest stock we want to own so that we can fill leftover cash
			this.trackCheapestOwnChannel(channel, state);

			// brain wants to own the stock - calculate differential between target and current
			this.planPositionAdjustment(channel, allocation, sells, buys, state);
		}

		// use leftover cash to buy additional shares of the cheapest stock we want to own
		this.fillLeftoverCash(buys, state);

		// return the planned actions
		return { sells, buys };
	}

	/**
	 * called when the brain wants out - sell all shares if we have any
	 */
	static planSellAll(channel, sells, state) {
		sells.push({ channel, shares: channel.shares });
		state.remainingCash += channel.shares * channel.currentPrice;
	}

	/**
	 * track the cheapest stock we want to own so that we can fill leftover cash
	 */
	static trackCheapestOwnChannel(channel, state) {
		if (!state.cheapestOwnChannel || channel.currentPrice < state.cheapestOwnChannel.currentPrice)
			state.cheapestOwnChannel = channel;
	}

	/**
	 * brain wants to own the stock - calculate differential between target and current
	 */
	static planPositionAdjustment(channel, allocation, sells, buys, state) {
		const targetShares = Math.floor(allocation.amount / channel.currentPrice);
		const sharesDiff = targetShares - channel.shares;
		if (sharesDiff < 0) {
			sells.push({ channel, shares: -sharesDiff });
			state.remainingCash += (-sharesDiff) * channel.currentPrice;
		}
		else if (sharesDiff > 0) {
			buys.push({ channel, shares: sharesDiff });
			state.remainingCash -= sharesDiff * channel.currentPrice;
		}
	}

	/**
	 * Use leftover cash to buy additional shares of the cheapest stock we want to own
	 */
	static fillLeftoverCash(buys, state) {

		// if there is no cash to fill or no stock to buy, we can't fill left over cash
		if (!state.cheapestOwnChannel || state.remainingCash <= 0) return;

		// get the number of additional shares we can buy with the remaining cash - if none, nothing we can do
		const additionalShares = Math.floor(state.remainingCash / state.cheapestOwnChannel.currentPrice);
		if (additionalShares <= 0) return;

		// Find existing buy for this channel and increment it, or add new buy
		const existingBuy = buys.find(b => b.channel === state.cheapestOwnChannel);
		if (existingBuy) existingBuy.shares += additionalShares;
		else buys.push({ channel: state.cheapestOwnChannel, shares: additionalShares });
	}

	/**
	 * Execute action plan by performing sells first, then buys
	 * @param {Object} actionPlan - Action plan with { sells: [...], buys: [...] }
	 */
	static async executeActionPlan(actionPlan) {

		// FIRST PASS: Execute sells to free up cash
		for (const sell of actionPlan.sells)
			await sell.channel.executeSell(sell.shares);

		// SECOND PASS: Execute buys using freed cash
		for (const buy of actionPlan.buys)
			await buy.channel.executeBuy(buy.shares);
	}

	/**
	 * Execute a buy action
	 * @param {number} sharesToBuy - Number of shares to buy
	 */
	async executeBuy(sharesToBuy) {

		const cost = sharesToBuy * this.currentPrice;

		// Check if we have enough cash - give a dollar wiggle room for rounding and stuff
		if (StockChannel.cash < (cost - 1))
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
	 * Get portfolio-level metrics across all stock channels
	 * @param {Array} channels - array of channel name to channel instance
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
	 * returns the label for an activity value
	 */
	getActionName(activityValue) {
		if (activityValue === POSITION_OWN) return 'POSITION_OWN';
		if (activityValue === POSITION_OUT) return 'POSITION_OUT';
		return 'UNKNOWN';
	}

	/**
	 * Format action label for debug output
	 * Converts raw coordinates to human-readable action names (e.g., "OWN", "OUT")
	 * @param {Object} coords - Coordinates object { dimension: value }
	 * @returns {string} Formatted action label
	 */
	formatActionLabel(coords) {
		const activityVal = coords[this.activityDimName];
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
			let percentRange = bucketToPercent.get(key);
			// Fall back to matching by dimension suffix for cross-channel voters
			// (all StockChannels share identical boundaries so this is accurate)
			if (!percentRange) {
				const underscoreIdx = dimName.indexOf('_');
				if (underscoreIdx >= 0) {
					const suffix = dimName.substring(underscoreIdx);
					percentRange = bucketToPercent.get(`${this.symbol}${suffix}:${val}`);
				}
			}
			if (percentRange) return `${dimName}=${val}(${percentRange})`;
			return part;
		}).join(', ');
	}

	/**
	 * Get current unrealized profit/loss for this channel (profit = current value - amount invested)
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
	 * Get channel metrics for diagnostic reporting
	 * @returns {Object} - Stock channel metrics
	 */
	getMetrics() {
		const currentValue = this.shares * this.currentPrice;
		const unrealizedProfit = currentValue - this.investment;

		return {
			...super.getMetrics(),
			symbol: this.symbol,
			investment: this.investment,
			currentValue: currentValue,
			unrealizedProfit: unrealizedProfit,
			shares: this.shares,
			currentPrice: this.currentPrice,
			trades: this.totalTrades || 0,
			position: this.lastAction === POSITION_OWN ? 'OWNED' : 'NOT OWNED'
		};
	}
}