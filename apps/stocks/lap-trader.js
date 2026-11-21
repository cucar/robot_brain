/**
 * The trader of the stock laps: it holds cash and whole shares, and at each bar brings its holdings in line with the
 * actions the brain last said, at the price the bar ended at.
 * Each stock is owned (action 1) or out, at most `maxPositions` owned, the value split evenly among them, what cash is
 * left swept into the cheapest owned stock, a cost in percent on each trade.
 * Over a lap it keeps the portfolio's value after every bar, for the Sharpe ratio, and the cost it paid.
 */

// the trading bars in a year of daily data, for annualizing the Sharpe ratio
const BARS_PER_YEAR = 252;

// the yearly return of a riskless investment, which the Sharpe ratio measures the excess over
const RISK_FREE_RATE = 0.05;

export class LapTrader {

	/**
	 * @param {string[]} symbols - the stocks, in channel order
	 * @param {object} settings
	 * @param {number} settings.maxPositions - how many stocks it may own at once
	 * @param {number} settings.costPercent - the cost of a trade, in percent of its value
	 * @param {number} settings.initialCapital - the cash each lap starts with
	 * @param {number} settings.minPrice - where more stocks say own than there are positions, none priced below this is bought
	 * @param {number} settings.maxPrice - and none priced at this or above
	 */
	constructor(symbols, { maxPositions, costPercent, initialCapital, minPrice, maxPrice }) {
		this.symbols = symbols;
		this.maxPositions = maxPositions;
		this.costRate = costPercent / 100;
		this.initialCapital = initialCapital;
		this.minPrice = minPrice;
		this.maxPrice = maxPrice;
		this.startLap();
	}

	/**
	 * Start a lap: all cash, no shares, no trades, no cost paid, and the value so far only the starting cash.
	 */
	startLap() {
		this.cash = this.initialCapital;
		this.shares = this.symbols.map(() => 0);
		this.trades = 0;
		this.costPaid = 0;
		this.values = [this.initialCapital];
	}

	/**
	 * What the cash and the shares are worth at these prices.
	 */
	readValue(prices) {
		return this.cash + this.shares.reduce((sum, count, i) => sum + count * prices[i], 0);
	}

	/**
	 * What the trader made since the lap started, at these prices.
	 */
	readProfit(prices) {
		return this.readValue(prices) - this.initialCapital;
	}

	/**
	 * Note the portfolio's value at the end of a bar, at the bar's prices.
	 */
	recordValue(prices) {
		this.values.push(this.readValue(prices));
	}

	/**
	 * Trade to the actions: each stock's last action (1 own, 0 out, null none yet) and its estimate, at these prices.
	 */
	trade(actions, estimates, prices) {
		const owned = this.chooseOwned(actions, estimates, prices);
		const { sells, buys } = this.planTrades(owned, prices);
		for (const [i, count] of sells) this.sell(i, count, prices[i]);
		for (const [i, count] of buys) this.buy(i, count, prices[i]);
	}

	/**
	 * The stocks to own: those whose action is own, with a price.
	 * With more than the positions allow: the best by estimate, then the higher price, then the symbol, within the
	 * price limits.
	 */
	chooseOwned(actions, estimates, prices) {
		let owned = this.symbols.map((_, i) => i).filter(i => actions[i] === 1 && prices[i] > 0);
		if (owned.length <= this.maxPositions) return owned;
		owned.sort((a, b) => Math.exp(estimates[b]) - Math.exp(estimates[a]) || prices[b] - prices[a] || compareText(this.symbols[a], this.symbols[b]));
		owned = owned.filter(i => prices[i] >= this.minPrice && prices[i] < this.maxPrice);
		return owned.slice(0, this.maxPositions);
	}

	/**
	 * The trades that bring the holdings to the stocks owned: sell what is out, bring each owned stock to its equal
	 * share of the value, then sweep the cash left into the cheapest owned stock.
	 */
	planTrades(owned, prices) {
		const value = this.readValue(prices);
		let remaining = this.cash;
		const sells = [];
		const buys = [];
		let cheapest = null;
		this.symbols.forEach((_, i) => {
			if (!owned.includes(i)) {
				if (this.shares[i] > 0) {
					sells.push([i, this.shares[i]]);
					remaining += this.readProceeds(this.shares[i], prices[i]);
				}
				return;
			}
			if (cheapest === null || prices[i] < prices[cheapest]) cheapest = i;
			const target = Math.floor((value / owned.length) / this.readBuyPrice(prices[i]));
			const change = target - this.shares[i];
			if (change < 0) {
				sells.push([i, -change]);
				remaining += this.readProceeds(-change, prices[i]);
			}
			else if (change > 0) {
				buys.push([i, change]);
				remaining -= change * this.readBuyPrice(prices[i]);
			}
		});
		if (cheapest !== null && remaining > 0) this.sweepInto(buys, cheapest, Math.floor(remaining / this.readBuyPrice(prices[cheapest])));
		return { sells, buys };
	}

	/**
	 * Add shares of one stock to the buys, to its buy if it has one.
	 */
	sweepInto(buys, i, extra) {
		if (extra <= 0) return;
		const buy = buys.find(([j]) => j === i);
		if (buy) buy[1] += extra;
		else buys.push([i, extra]);
	}

	/**
	 * What selling shares brings, its cost taken out.
	 * The count is multiplied by the price before the cost, in the order the Rust trader multiplies, so the cents agree.
	 */
	readProceeds(count, price) {
		return count * price * (1 - this.costRate);
	}

	/**
	 * What a share costs to buy, its cost added.
	 */
	readBuyPrice(price) {
		return price * (1 + this.costRate);
	}

	/**
	 * The cost of trading shares at a price: the cost rate on their value.
	 */
	readCost(count, price) {
		return count * price * this.costRate;
	}

	sell(i, count, price) {
		this.cash += this.readProceeds(count, price);
		this.costPaid += this.readCost(count, price);
		this.shares[i] -= count;
		this.trades++;
	}

	/**
	 * Buy as many of the shares as the cash pays for.
	 */
	buy(i, count, price) {
		const affordable = Math.min(count, Math.floor(this.cash / this.readBuyPrice(price)));
		if (affordable <= 0) return;
		this.cash -= affordable * this.readBuyPrice(price);
		this.costPaid += this.readCost(affordable, price);
		this.shares[i] += affordable;
		this.trades++;
	}

	/* ---------- The lap's statistics ---------- */

	/**
	 * The return per bar that, compounded over the lap's bars, gives the lap's whole return.
	 */
	readPerBarReturn() {
		const bars = this.values.length - 1;
		const final = this.values[bars];
		return bars > 0 ? Math.pow(final / this.initialCapital, 1 / bars) - 1 : 0;
	}

	/**
	 * The Sharpe ratio of the lap: the yearly mean of the per-bar log returns, less the riskless rate, over their
	 * yearly standard deviation; null where there are too few bars or no spread.
	 */
	readSharpe() {
		const returns = this.listLogReturns();
		if (returns.length < 2) return null;
		const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
		const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
		const spread = Math.sqrt(variance);
		if (spread === 0) return null;
		return (mean * BARS_PER_YEAR - RISK_FREE_RATE) / (spread * Math.sqrt(BARS_PER_YEAR));
	}

	/**
	 * The log return of each bar, from the value before it to the value after; a bar after a value of nothing is left out.
	 */
	listLogReturns() {
		const returns = [];
		for (let i = 1; i < this.values.length; i++) {
			if (this.values[i - 1] > 0) returns.push(Math.log(this.values[i] / this.values[i - 1]));
		}
		return returns;
	}
}

/**
 * Compare two strings by their characters, as Rust compares them.
 */
function compareText(a, b) {
	return a < b ? -1 : a > b ? 1 : 0;
}
