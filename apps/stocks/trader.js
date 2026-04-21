const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Per-symbol trader. Owns position tracking, tick state, reward calculation, and buy/sell
 * execution. Shares portfolio-wide cash via class statics so a multi-symbol portfolio adds
 * up to a single bankroll. Configured for simulation today; Alpaca-backed later.
 */
export class StockTrader {

	// Portfolio-wide shared state
	static initialCapital = 15000;
	static cash = StockTrader.initialCapital;
	static maxPositions = 1;
	static maxPrice = 5000;

	/**
	 * Reset portfolio-wide shared state (called once per episode before per-symbol resets).
	 */
	static resetPortfolio() {
		StockTrader.cash = StockTrader.initialCapital;
	}

	constructor(symbol, debug = false) {
		this.symbol = symbol;
		this.debug = debug;
		this.resetContext();
	}

	resetContext() {
		this.shares = 0;
		this.investment = 0;
		this.totalTrades = 0;
		this.lastKnownPrice = null;
		this.previousPrice = null;
		this.previousVolume = null;
		this.currentPrice = null;
		this.currentVolume = null;
		this.lastAction = null;
	}

	/**
	 * Advance one tick: current becomes previous, then new values are applied.
	 */
	setTick(price, volume) {
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;
		this.currentPrice = price;
		this.currentVolume = volume;
		if (this.currentPrice !== null) this.lastKnownPrice = this.currentPrice;
	}

	get hasData() {
		return this.previousPrice !== null && this.currentPrice !== null && this.currentPrice > 0 && this.currentVolume > 0;
	}

	/**
	 * Effective price for valuation: current if available, else last known, else 0.
	 */
	getCurrentPrice() {
		return this.currentPrice || this.lastKnownPrice || 0;
	}

	/**
	 * Record the brain-chosen action for this symbol so the next-frame reward can use it.
	 */
	apply(actionValue) {
		this.lastAction = actionValue;
	}

	/**
	 * Additive reward based on last action and most recent price move.
	 * Owned: positive if price went up. Not owned: positive if price went down.
	 */
	getReward() {
		if (!this.hasData) return 0;
		const percentChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
		const reward = this.lastAction === POSITION_OWN ? percentChange : -percentChange;
		if (this.debug) this.debugReward(reward);
		return reward;
	}

	debugReward(reward) {
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

	async executeBuy(sharesToBuy) {
		const cost = sharesToBuy * this.getCurrentPrice();

		// Check if we have enough cash - give a dollar wiggle room for rounding and stuff
		if (StockTrader.cash < (cost - 1))
			throw new Error(`${this.symbol}: Insufficient cash to buy ${sharesToBuy} shares at $${this.getCurrentPrice()} (need $${cost.toFixed(2)}, have $${StockTrader.cash.toFixed(2)})`);

		StockTrader.cash -= cost;
		this.shares += sharesToBuy;
		this.investment += cost;
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: BOUGHT ${sharesToBuy} shares @ $${this.getCurrentPrice().toFixed(2)} = $${cost.toFixed(2)} | Cash: $${StockTrader.cash.toFixed(2)}`);
	}

	async executeSell(sharesToSell) {
		if (sharesToSell > this.shares)
			throw new Error(`${this.symbol}: Cannot sell ${sharesToSell} shares, only have ${this.shares}`);

		const proceeds = sharesToSell * this.getCurrentPrice();
		const costBasis = (this.investment / this.shares) * sharesToSell;

		StockTrader.cash += proceeds;
		this.shares -= sharesToSell;
		this.investment -= costBasis;
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: SOLD ${sharesToSell} shares @ $${this.getCurrentPrice().toFixed(2)} = $${proceeds.toFixed(2)} | Cash: $${StockTrader.cash.toFixed(2)}`);
	}
}
