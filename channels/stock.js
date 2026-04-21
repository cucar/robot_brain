import { Channel } from './channel.js';
import { StockEncoder } from '../apps/stocks/encoder.js';
import { StockTrader } from '../apps/stocks/trader.js';

const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Stock Channel Implementation - thin adapter that delegates encoding to StockEncoder
 * and position/reward/execution to StockTrader. Preserves the channel surface the brain
 * currently relies on; will be retired once jobs talk to encoder/trader directly.
 */
export class StockChannel extends Channel {

	// --- portfolio-wide shared state (proxied to StockTrader for single-source-of-truth) ---
	static get initialCapital() { return StockTrader.initialCapital; }
	static set initialCapital(v) { StockTrader.initialCapital = v; }
	static get cash()           { return StockTrader.cash; }
	static set cash(v)          { StockTrader.cash = v; }
	static get maxPositions()   { return StockTrader.maxPositions; }
	static set maxPositions(v)  { StockTrader.maxPositions = v; }
	static get maxPrice()       { return StockTrader.maxPrice; }
	static set maxPrice(v)      { StockTrader.maxPrice = v; }

	/**
	 * Static method to reset channel-level context (shared state across all instances)
	 * Called once per episode reset before individual channel resetContext calls
	 */
	static resetChannelContext() {
		StockTrader.resetPortfolio();
	}

	/**
	 * constructor for the stock channel - dimensions are given when loading from database
	 */
	constructor(name, debug, id = null, dimensions = null) {
		super(name, debug, id);

		// Stock actions (buy/sell) don't affect event inputs (price/volume are independent of our trades)
		this.actionSequences = false;

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// Delegates: encoder handles inputs/dims/buckets, trader handles positions/reward/execution
		this.encoder = new StockEncoder(name, dimensions);
		this.trader = new StockTrader(name, debug);

		// Expose dimensions for the Channel framework
		this.activityDimName = this.encoder.activityDimName;
		this.priceChangeDim  = this.encoder.priceChangeDim;
		this.volumeChangeDim = this.encoder.volumeChangeDim;
		this.activityDim     = this.encoder.activityDim;

		// training data / mode (set by setTraining)
		this.trainingData = null;
		this.trainingRow = 0;

		// initialize context
		this.resetContext(false);
	}

	// --- per-instance state forwarding to trader (keeps the existing external surface working) ---
	get shares()          { return this.trader.shares; }
	set shares(v)         { this.trader.shares = v; }
	get investment()      { return this.trader.investment; }
	set investment(v)     { this.trader.investment = v; }
	get totalTrades()     { return this.trader.totalTrades; }
	set totalTrades(v)    { this.trader.totalTrades = v; }
	get lastKnownPrice()  { return this.trader.lastKnownPrice; }
	set lastKnownPrice(v) { this.trader.lastKnownPrice = v; }
	get previousPrice()   { return this.trader.previousPrice; }
	set previousPrice(v)  { this.trader.previousPrice = v; }
	get previousVolume()  { return this.trader.previousVolume; }
	set previousVolume(v) { this.trader.previousVolume = v; }
	get currentPrice()    { return this.trader.currentPrice; }
	set currentPrice(v)   { this.trader.currentPrice = v; }
	get currentVolume()   { return this.trader.currentVolume; }
	set currentVolume(v)  { this.trader.currentVolume = v; }
	get lastAction()      { return this.trader.lastAction; }
	set lastAction(v)     { this.trader.lastAction = v; }
	get hasData()         { return this.trader.hasData; }

	/**
	 * Set training data for this channel - switches channel to training mode
	 * @param {Array<{price, volume}>} rows - Training data rows
	 */
	setTraining(rows) {
		this.trainingData = rows;
		this.trainingRow = 0;
	}

	/**
	 * Reset channel state for new episode (keeps learned patterns but resets trading state)
	 */
	resetContext() {
		this.trader.resetContext();
		this.trainingRow = 0;
	}

	/**
	 * Reads the next data row from the prepared dataset - returns true if we should continue to process
	 */
	readNextRow() {

		// return false when all rows are consumed - this will stop the processing loop
		if (this.trainingRow >= this.trainingData.length) return false;

		// feed the next frame to the trader (advances previous → current)
		const row = this.trainingData[this.trainingRow++];
		this.trader.setFrame(row.price, row.volume);

		return true;
	}

	/**
	 * Legacy-path adapter: compute price/volume percent changes, bucketize through the
	 * encoder's static boundaries, and return name-keyed inputs in the shape the old
	 * buildFrame() path expects. The new spec-based path bypasses this — it hands raw
	 * scalars to brain.processInputs(), which bucketizes via the quantizer.
	 */
	computeChangeInputs() {
		const priceChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
		const volumeChange = this.previousVolume === 0 ? 1000 : ((this.currentVolume - this.previousVolume) / this.previousVolume) * 100;
		const priceBucket = this.encoder.discretizeChange(priceChange, this.encoder.priceBoundaries);
		const volumeBucket = this.encoder.discretizeChange(volumeChange, this.encoder.volumeBoundaries);
		if (this.debug) console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		return [
			{ dimension: this.encoder.priceChangeDim.name, value: priceBucket },
			{ dimension: this.encoder.volumeChangeDim.name, value: volumeBucket }
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
	getActionDimensions() {
		return [ this.activityDim ];
	}

	/**
	 * Returns all possible actions for this channel.
	 * These are pre-created during brain init so exploration can find them.
	 */
	getActions() {
		return [
			{ dimension: this.activityDimName, value: POSITION_OUT },
			{ dimension: this.activityDimName, value: POSITION_OWN }
		];
	}

	/**
	 * returns the coordinates of the channel default action - for stock channels, this is "do nothing"
	 */
	getDefaultAction() {
		return { dimension: this.activityDimName, value: POSITION_OUT };
	}

	/**
	 * Get frame input data for this stock channel
	 */
	getFrameEvents(frameNumber) {

		// currently only the training mode is implemented
		if (this.trainingData === null) throw new Error(`${this.symbol}: live mode not implemented yet.`);

		// Read next data row - if none left, we're done
		if (!this.readNextRow()) return [];

		// if this is the first frame, read another row so that we can start sending change stats
		if (frameNumber === 1) this.readNextRow();

		// if the channel does not have any data in this frame, nothing to report
		if (!this.hasData) return [];

		// Compute and return discretized changes
		return this.computeChangeInputs();
	}

	/**
	 * Get feedback based on price movement (delegates to trader).
	 */
	async getRewards() {
		return this.trader.getReward();
	}

	/**
	 * Calculate continuous prediction error for price predictions.
	 * Compares weighted predicted percentage change to actual percentage change.
	 */
	calculatePredictionError(predictions, actuals) {
		const priceChangeDim = `${this.symbol}_price_change`;

		// Filter to price change predictions only
		const pricePredictions = predictions.filter(p => p.coordinate.dimension === priceChangeDim);
		if (pricePredictions.length === 0) return null;

		// Calculate weighted predicted percentage change
		let totalWeightedChange = 0;
		let totalStrength = 0;
		for (const pred of pricePredictions) {
			const percentageChange = this.encoder.bucketValueToPercentage(pred.coordinate.value);
			totalWeightedChange += percentageChange * pred.strength;
			totalStrength += pred.strength;
		}
		if (totalStrength === 0) return null;
		const predictedChange = totalWeightedChange / totalStrength;

		// Find actual price change from actuals
		const actualCoord = actuals.find(c => c.dimension === priceChangeDim);
		if (!actualCoord) return null;
		const actualChange = this.encoder.bucketValueToPercentage(actualCoord.value);

		const error = Math.abs(predictedChange - actualChange);
		if (this.debug)
			console.log(`${this.symbol}: Predicted ${predictedChange.toFixed(2)}%, Actual ${actualChange.toFixed(2)}%, Error ${error.toFixed(2)}pp`);
		return error;
	}

	/**
	 * Effective price of the stock (delegates to trader).
	 */
	getCurrentPrice() {
		return this.trader.getCurrentPrice();
	}

	/**
	 * Static method for coordinated execution across all stock channels
	 * Handles portfolio allocation before executing individual channel actions
	 * @param {Map<string, { channel, actions, events }>} channelInferences - Map of channel name to channel data
	 */
	static async executeChannelActions(channelInferences) {

		// if there are no actions, nothing to do
		if (!this.hasActions(channelInferences)) return;

		// Save last actions for rewarding in the next frame
		this.saveLastActions(channelInferences);

		// Calculate portfolio allocations
		const allocations = this.getAllocations(channelInferences);

		// Generate action plan based on differential between ideal and current allocations
		const actionPlan = this.getActionPlan(channelInferences, allocations);

		// Execute the action plan
		await this.executeActionPlan(actionPlan);
	}

	/**
	 * Check if any channel has actions
	 */
	static hasActions(channelInferences) {
		for (const [, { actions }] of channelInferences) if (actions.length > 0) return true;
		return false;
	}

	/**
	 * Save last action for each channel for tracking purposes. Uses setAction() rather
	 * than apply() because this legacy path already extracted the action-kind winner into
	 * the actions array — we hand the bucket value and its expected reward directly.
	 */
	static saveLastActions(channelInferences) {
		for (const [, { channel, actions }] of channelInferences) {
			if (actions.length === 0) continue;
			channel.trader.setAction(actions[0].coordinate.value, actions[0].reward ?? 0);
		}
	}

	/**
	 * Calculate portfolio allocations for stock actions based on total portfolio value
	 * Uses softmax (exponential) weighting to handle negative rewards naturally
	 */
	static getAllocations(channelInferences) {
		const totalValue = this.getTotalValue(channelInferences);
		const actions = this.determineActions(channelInferences);
		const allocations = this.distributeAllocations(channelInferences, actions, totalValue);
		this.setMissingChannelAllocations(channelInferences, allocations);
		return allocations;
	}

	/**
	 * returns total portfolio value (cash + all current holdings)
	 */
	static getTotalValue(channelInferences) {
		let totalPortfolioValue = this.cash;
		for (const [, { channel }] of channelInferences)
			totalPortfolioValue += channel.shares * channel.getCurrentPrice();
		return totalPortfolioValue;
	}

	/**
	 * determines actions to be taken from the channel inferences
	 */
	static determineActions(channelInferences) {
		const allActions = [];
		for (const [channelName, { actions }] of channelInferences) {

			// get the brain desired action for the channel
			if (actions.length === 0) continue;
			const actionData = actions[0]; // Single action per stock channel
			const action = actionData.coordinate.value;
			const actionOwn = action === POSITION_OWN;

			// rank stocks by expected reward
			allActions.push({ channelName, rank: Math.exp(actionData.reward), isOwn: actionOwn });
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
			ownActions.sort((a, b) =>
				// sort by reward - prefer higher rewards
				b.rank - a.rank ||
				// if reward is the same, we're not sure - prefer more expensive, safer stocks
				channels.get(b.channelName).channel.getCurrentPrice() - channels.get(a.channelName).channel.getCurrentPrice() ||
				// if reward and price are the same, sort alphabetically to be deterministic
				a.channelName.localeCompare(b.channelName)
			);
			ownActions = ownActions.filter(a => channels.get(a.channelName).channel.getCurrentPrice() < this.maxPrice);
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
	 * Set OUT allocation for channels without predictions
	 * Channels with shares but no data still need OUT allocation so they can be sold using lastKnownPrice
	 */
	static setMissingChannelAllocations(channelInferences, allocations) {
		for (const [channelName] of channelInferences)
			if (!allocations.has(channelName))
				allocations.set(channelName, { action: POSITION_OUT, amount: 0 });
	}

	/**
	 * Generate action plan based on differential between ideal allocations and current holdings
	 */
	static getActionPlan(channelInferences, allocations) {
		const sells = [];
		const buys = [];
		const state = { remainingCash: this.cash, cheapestOwnChannel: null };

		for (const [channelName, allocation] of allocations) {
			const { channel } = channelInferences.get(channelName);

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

		return { sells, buys };
	}

	/**
	 * called when the brain wants out - sell all shares if we have any
	 */
	static planSellAll(channel, sells, state) {
		sells.push({ channel, shares: channel.shares });
		state.remainingCash += channel.shares * channel.getCurrentPrice();
	}

	/**
	 * track the cheapest stock we want to own so that we can fill leftover cash
	 */
	static trackCheapestOwnChannel(channel, state) {
		if (!state.cheapestOwnChannel || channel.getCurrentPrice() < state.cheapestOwnChannel.getCurrentPrice())
			state.cheapestOwnChannel = channel;
	}

	/**
	 * brain wants to own the stock - calculate differential between target and current
	 */
	static planPositionAdjustment(channel, allocation, sells, buys, state) {
		const targetShares = Math.floor(allocation.amount / channel.getCurrentPrice());
		const sharesDiff = targetShares - channel.shares;
		if (sharesDiff < 0) {
			sells.push({ channel, shares: -sharesDiff });
			state.remainingCash += (-sharesDiff) * channel.getCurrentPrice();
		}
		else if (sharesDiff > 0) {
			buys.push({ channel, shares: sharesDiff });
			state.remainingCash -= sharesDiff * channel.getCurrentPrice();
		}
	}

	/**
	 * Use leftover cash to buy additional shares of the cheapest stock we want to own
	 */
	static fillLeftoverCash(buys, state) {

		if (!state.cheapestOwnChannel || state.remainingCash <= 0) return;

		const additionalShares = Math.floor(state.remainingCash / state.cheapestOwnChannel.getCurrentPrice());
		if (additionalShares <= 0) return;

		const existingBuy = buys.find(b => b.channel === state.cheapestOwnChannel);
		if (existingBuy) existingBuy.shares += additionalShares;
		else buys.push({ channel: state.cheapestOwnChannel, shares: additionalShares });
	}

	/**
	 * Execute action plan by performing sells first, then buys
	 */
	static async executeActionPlan(actionPlan) {
		for (const sell of actionPlan.sells)
			await sell.channel.executeSell(sell.shares);
		for (const buy of actionPlan.buys)
			await buy.channel.executeBuy(buy.shares);
	}

	/**
	 * Execute a buy action (delegates to trader).
	 */
	async executeBuy(sharesToBuy) {
		return this.trader.executeBuy(sharesToBuy);
	}

	/**
	 * Execute a sell action (delegates to trader).
	 */
	async executeSell(sharesToSell) {
		return this.trader.executeSell(sharesToSell);
	}

	/**
	 * Get aggregate metrics across all stock channels (portfolio metrics)
	 */
	static getAggregateMetrics(channels) {
		let totalCurrentValue = 0;
		for (const [_, channel] of channels)
			totalCurrentValue += channel.shares * channel.getCurrentPrice();

		const totalProfit = (this.cash + totalCurrentValue) - this.initialCapital;

		return {
			cash: this.cash,
			totalInvestments: totalCurrentValue,
			totalProfit
		};
	}

	/**
	 * Get aggregate display string for frame summary (portfolio P&L)
	 */
	static getAggregateDisplay(channels) {
		const metrics = this.getAggregateMetrics(channels);
		const totalPL = metrics.totalProfit >= 0 ? '+' : '';
		return `Cash:${metrics.cash.toFixed(0)} | P&L:${totalPL}${metrics.totalProfit.toFixed(2)}`;
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
	 */
	formatActionLabel(coordinate) {
		if (coordinate.value === POSITION_OWN) return 'OWN';
		if (coordinate.value === POSITION_OUT) return 'OUT';
		return JSON.stringify(coordinate);
	}

	/**
	 * Format coordinates string with percentage ranges where applicable
	 */
	formatCoordinates(coordsStr) {
		return this.formatCoordsWithPercent(coordsStr, this.encoder.bucketToPercent);
	}

	/**
	 * Format coordinates string with percentage ranges where applicable
	 */
	formatCoordsWithPercent(coordsStr, bucketToPercent) {
		if (!coordsStr) return '(no coords)';
		if (!bucketToPercent) return coordsStr;
		return coordsStr.split(', ').map(part => {
			const [dimName, valStr] = part.split('=');
			const val = parseFloat(valStr);
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
	 * Get short state display for frame summary
	 */
	getStateDisplay() {
		if (this.shares === 0) return null;
		return `${this.symbol}:${this.shares}@$${this.getCurrentPrice()?.toFixed(2) ?? '?'}`;
	}

	/**
	 * Get channel metrics for diagnostic reporting
	 */
	getMetrics() {
		const currentValue = this.shares * this.getCurrentPrice();
		const unrealizedProfit = currentValue - this.investment;

		return {
			...super.getMetrics(),
			symbol: this.symbol,
			investment: this.investment,
			currentValue: currentValue,
			unrealizedProfit: unrealizedProfit,
			shares: this.shares,
			currentPrice: this.getCurrentPrice(),
			trades: this.totalTrades || 0,
			position: this.lastAction === POSITION_OWN ? 'OWNED' : 'NOT OWNED'
		};
	}
}
