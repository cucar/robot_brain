import AlpacaApi from '@alpacahq/alpaca-trade-api';

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
		const bars = [];
		// limit 0 means return everything - pageLimit is the max number of bars per request
		const options = { start, end, timeframe, limit: 0, pageLimit: 1000, feed: 'iex', adjustment: 'all' };
		for await (const bar of this.alpaca.getBarsV2(symbols, options))
			bars.push(bar);
		return bars;
	}
}