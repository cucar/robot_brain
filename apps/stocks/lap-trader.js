/**
 * The trader of the stock laps: it holds cash and whole shares, and at each bar brings its holdings in line with the
 * actions the brain last said, at the price the bar ended at.
 * It trades as apps/stocks/trader.js does: each stock owned (action 1) or out, at most `maxPositions` owned, the value
 * split evenly among them, what cash is left swept into the cheapest owned stock, a cost in percent on each trade.
 */

// what the trader starts each lap with
export const STARTING_CASH = 15000;

// a stock priced at this or more is not bought where there are more owns than positions
const PRICE_CAP = 5000;

export class LapTrader {

	/**
	 * @param {string[]} symbols - the stocks, in channel order
	 * @param {number} maxPositions - how many stocks it may own at once
	 * @param {number} costPercent - the cost of a trade, in percent of its value
	 */
	constructor(symbols, maxPositions, costPercent) {
		this.symbols = symbols;
		this.maxPositions = maxPositions;
		this.costRate = costPercent / 100;
		this.startLap();
	}

	/**
	 * Start a lap: all cash, no shares, no trades.
	 */
	startLap() {
		this.cash = STARTING_CASH;
		this.shares = this.symbols.map(() => 0);
		this.trades = 0;
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
		return this.readValue(prices) - STARTING_CASH;
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
	 * With more than the positions allow, the best by estimate, then the higher price, then the symbol, under the cap.
	 */
	chooseOwned(actions, estimates, prices) {
		let owned = this.symbols.map((_, i) => i).filter(i => actions[i] === 1 && prices[i] > 0);
		if (owned.length <= this.maxPositions) return owned;
		owned.sort((a, b) => Math.exp(estimates[b]) - Math.exp(estimates[a]) || prices[b] - prices[a] || compareText(this.symbols[a], this.symbols[b]));
		owned = owned.filter(i => prices[i] < PRICE_CAP);
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

	sell(i, count, price) {
		this.cash += this.readProceeds(count, price);
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
		this.shares[i] += affordable;
		this.trades++;
	}
}

/**
 * Compare two strings by their characters, as Rust compares them.
 */
function compareText(a, b) {
	return a < b ? -1 : a > b ? 1 : 0;
}
