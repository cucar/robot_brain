const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Per-symbol trader. Owns position tracking, frame state (current/previous price and volume),
 * reward calculation, and buy/sell execution. Shares portfolio-wide cash via class statics
 * so a multi-symbol portfolio adds up to a single bankroll. Configured for simulation today;
 * Alpaca-backed later.
 */
export class StockTrader {

	// Portfolio-wide shared state
	static initialCapital = 15000;
	static cash = StockTrader.initialCapital;
	static maxPositions = 1;
	static maxPrice = 5000;
	static transactionCost = 0;
	static totalTransactionCostPaid = 0;

	/**
	 * Reset portfolio-wide shared state (called once per episode before per-symbol resets).
	 */
	static resetPortfolio() {
		StockTrader.cash = StockTrader.initialCapital;
		StockTrader.totalTransactionCostPaid = 0;
	}

	/**
	 * Create a trader for the given ticker symbol. Position state is zeroed
	 * and the channelId is left null until bindChannelId() links this trader
	 * to its encoder's brain-allocated channel.
	 * @param {string} symbol
	 * @param {boolean} debug
	 */
	constructor(symbol, debug = false) {
		this.symbol = symbol;
		this.debug = debug;
		this.channelId = null; // assigned by bindChannelId() — same ID the encoder gets
		this.resetContext();
	}

	/**
	 * Called after the brain allocates the channel ID for this symbol. Trader borrows the
	 * encoder's ID so rewards, inputs, and inferences all key off a single number per symbol.
	 */
	bindChannelId(channelId) {
		this.channelId = channelId;
	}

	/**
	 * Zero out all per-episode mutable state: position, frame readings, and
	 * the last action/reward. Called at construction and once per episode.
	 */
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
		this.lastActionReward = 0;
	}

	/**
	 * Advance one frame: current becomes previous, then new values are applied. Called
	 * once per frame by the job so reward calculation has a full (prev, curr) pair and
	 * valuation can fall back to lastKnownPrice when a bar is missing.
	 */
	setFrame(price, volume) {
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;
		this.currentPrice = price;
		this.currentVolume = volume;
		if (this.currentPrice !== null) this.lastKnownPrice = this.currentPrice;
	}

	/**
	 * True when the trader has a valid (previous, current) frame pair with
	 * nonzero price and volume — the minimum needed for reward calculation.
	 */
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
	 * Apply the brain's inferences for this trader's channel. Extracts the action-kind
	 * winner and records it as the last action so next-frame reward can use it.
	 * Also captures the winning action's strength as its expected-reward proxy for
	 * portfolio ranking (mirrors the previous `actionData.reward` field).
	 * @param {Array<{dimId, kind, winner, continuous}>|undefined} inferences
	 */
	apply(inferences) {
		if (!inferences || inferences.length === 0) return;
		const actionInf = inferences.find(inf => inf.kind === 'action');
		if (!actionInf) return;
		this.lastAction = actionInf.winner.value;
		this.lastActionReward = actionInf.winner.score ?? actionInf.winner.strength ?? 0;
	}

	/**
	 * Legacy-path setter: record the action bucket value directly (no inference array).
	 * Used by StockChannel.saveLastActions while the channel-class path is still alive.
	 */
	setAction(value, reward = 0) {
		this.lastAction = value;
		this.lastActionReward = reward;
	}

	/**
	 * Additive reward based on last action and most recent price move.
	 * Owned: positive if price went up. Not owned: positive if price went down.
	 */
	getReward() {
		if (!this.hasData) return 0;
		const costMul = StockTrader.transactionCost / 100;
		const currentBid = this.currentPrice * (1 - costMul);
		const previousAsk = this.previousPrice * (1 + costMul);
		const percentChange = ((currentBid - previousAsk) / previousAsk) * 100;
		const reward = this.lastAction === POSITION_OWN ? percentChange : -percentChange;
		if (this.debug) this.debugReward(reward);
		return reward;
	}

	/**
	 * Log a human-readable reward breakdown to the console. Shows the price
	 * move, computed reward, and unrealized P&L when holding a position.
	 * @param {number} reward
	 */
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

	/**
	 * Buy shares at the current price, deducting cost from portfolio cash.
	 * Throws if the portfolio lacks sufficient funds.
	 * @param {number} sharesToBuy
	 */
	async executeBuy(sharesToBuy) {
		const price = this.getCurrentPrice();
		if (price <= 0) return; // exclude $0 stocks - they are de-listed
		const transactionPenalty = price * (StockTrader.transactionCost / 100);
		const effectivePrice = price + transactionPenalty;

		// Clamp to what we can actually afford — floating-point drift between the
		// planner's projection and execution can leave us a few cents short.
		const affordable = Math.floor(StockTrader.cash / effectivePrice);
		sharesToBuy = Math.min(sharesToBuy, affordable);
		if (sharesToBuy <= 0) return;

		const cost = sharesToBuy * effectivePrice;
		StockTrader.cash -= cost;
		StockTrader.totalTransactionCostPaid += sharesToBuy * transactionPenalty;
		this.shares += sharesToBuy;
		this.investment += cost;
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: BOUGHT ${sharesToBuy} shares @ $${effectivePrice.toFixed(2)} = $${cost.toFixed(2)} | Cash: $${StockTrader.cash.toFixed(2)}`);
	}

	/**
	 * Sell shares at the current price, crediting proceeds to portfolio cash
	 * and reducing the cost basis proportionally. Throws if the trader holds
	 * fewer shares than requested.
	 * @param {number} sharesToSell
	 */
	async executeSell(sharesToSell) {
		if (sharesToSell > this.shares)
			throw new Error(`${this.symbol}: Cannot sell ${sharesToSell} shares, only have ${this.shares}`);

		const price = this.getCurrentPrice();
		const transactionPenalty = price * (StockTrader.transactionCost / 100);
		const effectivePrice = price - transactionPenalty;
		const proceeds = sharesToSell * effectivePrice;
		const costBasis = (this.investment / this.shares) * sharesToSell;

		StockTrader.cash += proceeds;
		StockTrader.totalTransactionCostPaid += sharesToSell * transactionPenalty;
		this.shares -= sharesToSell;
		this.investment -= costBasis;
		this.totalTrades++;

		if (this.debug)
			console.log(`${this.symbol}: SOLD ${sharesToSell} shares @ $${effectivePrice.toFixed(2)} = $${proceeds.toFixed(2)} | Cash: $${StockTrader.cash.toFixed(2)}`);
	}

	/**
	 * Per-frame state blurb for the summary line's tail. Only emits when the trader
	 * holds a position — matches the legacy channel behavior of hiding idle symbols
	 * so the line stays compact in multi-symbol runs.
	 */
	getStateDisplay() {
		if (this.shares === 0) return null;
		return `${this.symbol}:${this.shares}@$${this.getCurrentPrice()?.toFixed(2) ?? '?'}`;
	}

	/**
	 * Build the tail suffix the Job appends to each per-frame summary: per-symbol
	 * holdings (via getStateDisplay) plus portfolio Cash/P&L. Returns an empty
	 * string when no trader holds anything and cash is at initial capital, so
	 * early frames before any trade don't clutter the log.
	 */
	static getSummaryTail(traders) {
		const stateParts = [];
		for (const trader of traders) {
			const display = trader.getStateDisplay();
			if (display) stateParts.push(display);
		}
		const state = stateParts.length > 0 ? stateParts.join(', ') : 'None';

		let totalCurrentValue = 0;
		for (const trader of traders) totalCurrentValue += trader.shares * trader.getCurrentPrice();
		const totalProfit = (StockTrader.cash + totalCurrentValue) - StockTrader.initialCapital;
		const sign = totalProfit >= 0 ? '+' : '';

		return `State: ${state} | Cash:${StockTrader.cash.toFixed(0)} | P&L:${sign}${totalProfit.toFixed(2)}`;
	}

	/**
	 * Aggregate portfolio profit across a set of traders:
	 *   (cash + market value) - initial capital
	 */
	static getPortfolioProfit(traders) {
		let totalCurrentValue = 0;
		for (const trader of traders) totalCurrentValue += trader.shares * trader.getCurrentPrice();
		return (StockTrader.cash + totalCurrentValue) - StockTrader.initialCapital;
	}

	/**
	 * Coordinated execution across a group of traders. Given each trader's inferences,
	 * allocates the portfolio, plans the differential, and executes sells-then-buys.
	 * Traders with no inferences get a default OUT allocation so lingering positions are sold.
	 * @param {StockTrader[]} traders
	 */
	static async executePortfolio(traders) {

		// Only traders whose brain produced an action participate in allocation. Traders
		// that didn't produce one still get a default OUT allocation below so any lingering
		// position is sold — we don't want stale holdings from a skipped frame.
		const acting = traders.filter(t => t.lastAction !== null);
		if (acting.length === 0) return;

		// Compute desired (action, dollar amount) per trader, then diff against current
		// holdings to produce a concrete sell/buy plan.
		const totalValue = this.getTotalValue(traders);
		const ranked = this.rankActions(acting);
		const allocations = this.distributeAllocations(traders, ranked, totalValue);
		this.setMissingAllocations(traders, allocations);

		// Sells run before buys so cash is freed before we try to spend it.
		const plan = this.getActionPlan(traders, allocations);
		await this.executeActionPlan(plan);
	}

	/**
	 * Total portfolio value: unallocated cash plus current market value of every trader's
	 * position. Used as the denominator for per-symbol allocation sizing.
	 */
	static getTotalValue(traders) {
		let total = this.cash;
		for (const trader of traders) total += trader.shares * trader.getCurrentPrice();
		return total;
	}

	/**
	 * Shape each trader into a ranking tuple. `rank = exp(lastActionReward)` gives an
	 * always-positive score suitable for sorting, and `isOwn` flags the ones that want
	 * to hold a position this frame.
	 */
	static rankActions(traders) {
		return traders.map(trader => ({
			trader,
			rank: Math.exp(trader.lastActionReward),
			isOwn: trader.lastAction === POSITION_OWN
		}));
	}

	/**
	 * Pick the winning OWN traders (respecting maxPositions and maxPrice), then split
	 * portfolio value equally across them. Traders not in the winning set get a zero-dollar OUT allocation.
	 * Returned Map is keyed by trader for O(1) lookup in the planner.
	 */
	static distributeAllocations(traders, ranked, totalValue) {
		let ownActions = ranked.filter(a => a.isOwn);

		// More OWN actions than we can hold: rank first by expected reward, then prefer
		// cheaper shares (more granularity when sizing), then alphabetic for determinism.
		// Filter out any symbol priced above maxPrice — too lumpy to allocate cleanly.
		if (ownActions.length > this.maxPositions) {
			ownActions.sort((a, b) =>
				b.rank - a.rank ||
				b.trader.getCurrentPrice() - a.trader.getCurrentPrice() ||
				a.trader.symbol.localeCompare(b.trader.symbol)
			);
			ownActions = ownActions.filter(a => {
				const p = a.trader.getCurrentPrice();
				return p > 0 && p < this.maxPrice; // exclude $0 stocks - they are de-listed
			});
			ownActions = ownActions.slice(0, this.maxPositions);
		}

		const ownSet = new Set(ownActions.map(a => a.trader));
		const allocations = new Map();
		for (const a of ranked) allocations.set(a.trader, {
			action: (a.isOwn && ownSet.has(a.trader)) ? POSITION_OWN : POSITION_OUT,
			amount: (a.isOwn && ownSet.has(a.trader)) ? (1 / ownActions.length) * totalValue : 0
		});
		return allocations;
	}

	/**
	 * Traders that didn't record an action (null lastAction) still need an allocation so
	 * the planner sees them and sells off any stale shares. Default them to OUT / $0.
	 */
	static setMissingAllocations(traders, allocations) {
		for (const trader of traders)
			if (!allocations.has(trader))
				allocations.set(trader, { action: POSITION_OUT, amount: 0 });
	}

	/**
	 * Turn (trader, allocation) pairs into a concrete {sells, buys} list by diffing target
	 * share count against current holdings. Tracks projected cash so we can sweep any
	 * leftover into the cheapest owned symbol (otherwise flooring to whole shares would
	 * leave dollars sitting idle).
	 */
	static getActionPlan(traders, allocations) {
		const sells = [];
		const buys = [];
		const transactionMul = this.transactionCost / 100;
		const state = { remainingCash: this.cash, cheapestOwnTrader: null };

		for (const trader of traders) {
			const allocation = allocations.get(trader);
			const price = trader.getCurrentPrice();

			// exclude $0 stocks - they are de-listed
			if (price <= 0) continue;

			// OUT allocation: liquidate any existing position and credit projected cash.
			// Sells receive price minus spread.
			if (allocation.action === POSITION_OUT) {
				if (trader.shares > 0) {
					sells.push({ trader, shares: trader.shares });
					state.remainingCash += trader.shares * price * (1 - transactionMul);
				}
				continue;
			}

			// Remember the cheapest OWN trader so we can dump leftover cash into it below.
			if (!state.cheapestOwnTrader || price < state.cheapestOwnTrader.getCurrentPrice())
				state.cheapestOwnTrader = trader;

			// Whole-share target from dollar allocation; buys cost price plus spread.
			const buyPrice = price * (1 + transactionMul);
			const targetShares = Math.floor(allocation.amount / buyPrice);
			const sharesDiff = targetShares - trader.shares;
			if (sharesDiff < 0) {
				sells.push({ trader, shares: -sharesDiff });
				state.remainingCash += (-sharesDiff) * price * (1 - transactionMul);
			}
			else if (sharesDiff > 0) {
				buys.push({ trader, shares: sharesDiff });
				state.remainingCash -= sharesDiff * buyPrice;
			}
		}

		// Flooring leaves sub-share dollars on the table; sweep them into the cheapest
		// owned symbol so we stay as fully invested as the allocation intended.
		if (state.cheapestOwnTrader && state.remainingCash > 0) {
			const sweepPrice = state.cheapestOwnTrader.getCurrentPrice() * (1 + transactionMul);
			const additional = Math.floor(state.remainingCash / sweepPrice);
			if (additional > 0) {
				const existing = buys.find(b => b.trader === state.cheapestOwnTrader);
				if (existing) existing.shares += additional;
				else buys.push({ trader: state.cheapestOwnTrader, shares: additional });
			}
		}

		return { sells, buys };
	}

	/**
	 * Execute the plan: sells first (frees cash), then buys (needs cash). Sequential
	 * awaits — simulation today, but the same ordering is required for a live broker
	 * so we don't overdraw.
	 */
	static async executeActionPlan(plan) {
		for (const sell of plan.sells) await sell.trader.executeSell(sell.shares);
		for (const buy of plan.buys) await buy.trader.executeBuy(buy.shares);
	}
}
