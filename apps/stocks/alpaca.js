import AlpacaApi from '@alpacahq/alpaca-trade-api';
import RateLimiter from './rate-limiter.js';

/**
 * Alpaca API wrapper for market data and trading
 */
export default class Alpaca {

	constructor() {
		this.alpaca = new AlpacaApi({
			keyId: process.env.ALPACA_KEY_ID,
			secretKey: process.env.ALPACA_SECRET_KEY,
			paper: true
		});
		this.rateLimiter = new RateLimiter();
	}

	/**
	 * returns the market data for a requested set of symbols
	 * @param symbols - symbol or array of symbols to get data for
	 * @param start - start date for the data request
	 * @param end - end date for the data request
	 * @param timeframe - timeframe for the data request (1Min, 5Min, 15Min, 1D, etc.)
	 * @returns [{ Timestamp, OpenPrice, ClosePrice, HighPrice, LowPrice, Volume, TradeCount, VWAP }]
	 */
	async getBars(symbols, start, end, timeframe = '1Min') {
		let barCount = 0;
		const bars = [];

		// limit 0 means return everything - pageLimit is the max number of bars per request
		const pageLimit = 10000; // use maximum limit for maximum efficiency
		const options = { start, end, timeframe, limit: 0, pageLimit, feed: 'iex', adjustment: 'all' };

		// getBarsV2 returns an async generator.  Internally the SDK fetches bars in pages from the REST API,
		// but it does NOT hand back pages — it yields individual bar objects one at a time.
		// So, the loop below receives individual bars for the requested data, even though the SDK makes multiple requests.
		for await (const bar of this.alpaca.getBarsV2(symbols, options)) {

			// The SDK won't fetch the NEXT page until we consume all bars from the current page.
			bars.push(bar);
			barCount++;

			// when api count hits a multiple of pageLimit we know the last bar of that page is consumed, and we can rate limit if needed
			// Every pageLimit bars = one page consumed = one API call was made. Pause if needed before the SDK fires the next API call.
			if (barCount % pageLimit === 0) await this.rateLimiter.wait();
		}

		return bars;
	}
}