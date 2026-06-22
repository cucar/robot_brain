import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Stock Download Script (Yahoo Finance) - Downloads historical stock data from Yahoo's public chart
 * API and saves it as JSON in the SAME bar shape Alpaca produces ({ Timestamp, OpenPrice, HighPrice,
 * LowPrice, ClosePrice, Volume }), so the existing setup.js JSON→CSV step consumes it unchanged.
 *
 * Yahoo is the "longer data" provider: free daily history reaches back decades, far past Alpaca's
 * IEX window, which is what the roadmap's stock test calls for. This job is daily-only by design —
 * Yahoo caps intraday lookback (60m requests beyond ~730 days return HTTP 422), so hourly history is
 * unreliable and not worth supporting here. Use download-alpaca.js for intraday timeframes.
 *
 * Prices are split/dividend-adjusted to match Alpaca's `adjustment: 'all'`: Yahoo exposes an adjusted
 * close but not an adjusted open, so each bar's open/high/low are scaled by the per-bar factor
 * adjclose/close before being written. Daily open prices match Alpaca to within ~0.1% (the residual is
 * Alpaca's single-venue IEV feed vs Yahoo's consolidated tape). Volume differs by ~50-100x for the same
 * reason — Alpaca free reports IEX-only volume, Yahoo reports consolidated — so do NOT mix providers
 * within one dataset.
 *
 * Run with: node apps/stocks/jobs/download-yahoo.js --timeframe 1D --start 2005-01-01 --end 2026-05-13
 */

const config = {
	symbols: [
		// 100 stocks - expected to be good
		'SO', 'VALE', 'STLD', 'GOOGL', 'MU', 'PLTR', 'UUUU', 'PFE', 'CRM', 'HAL',
		'AWR', /* 'SAND', */ 'GM', 'EQIX', 'RTX', 'KGC', 'ALB', 'AAPL', 'CVX', 'HD',
		'WPM', 'BEP', 'AREC', 'JNJ', 'SLB', 'PLD', 'EXK', 'NVDA', 'CAT', 'WFC',
		'RGLD', 'WEAT', 'OXY', 'CEG', 'LOW', 'PAAS', 'MP', 'LMT', 'GS', 'COST',
		'AG', 'TECK', 'MRK', 'INTC', 'BIP', 'PSA', 'DVN', 'AVAV', 'PEP', 'CDE',
		'TSM', 'FCX', 'PM', 'NUE', 'LEU', 'AMT', 'WMT', 'MRVL', 'F', /* 'SILV', */
		'RIO', 'NOC', 'V', 'ENB', 'BTU', 'AEM', 'AMZN', 'KLAC', 'CLF', 'O',
		'NEM', 'GD', 'BAC', 'NEE', 'SQM', 'ABBV', 'AMAT', 'KMI', 'PG', 'UEC',
		'GOLD', 'BHP', 'CRML', 'LLY', 'AVGO', 'FNV', 'JPM', 'DE', 'TM', 'WM',
		'HL', 'CCJ', 'COP', 'USAR', 'XOM', 'AMD', 'LAC', 'MSFT', 'MUX', 'SPY'
	],
	timeframe: '1D',
	startDate: '2005-01-01',
	endDate: '2026-05-13'
};

/**
 * Parse command line arguments. Flags mirror download-alpaca.js so the two providers are
 * interchangeable in the download → setup → test pipeline.
 */
function parseArgs() {
	let timeframe = config.timeframe;
	const timeframeIndex = process.argv.indexOf('--timeframe');
	if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) timeframe = process.argv[timeframeIndex + 1];

	let startDate = config.startDate;
	const startIndex = process.argv.indexOf('--start');
	if (startIndex !== -1 && process.argv[startIndex + 1]) startDate = process.argv[startIndex + 1];

	let endDate = config.endDate;
	const endIndex = process.argv.indexOf('--end');
	if (endIndex !== -1 && process.argv[endIndex + 1]) endDate = process.argv[endIndex + 1];

	let symbols = config.symbols;
	const symbolsIndex = process.argv.indexOf('--symbols');
	if (symbolsIndex !== -1 && process.argv[symbolsIndex + 1]) symbols = process.argv[symbolsIndex + 1].split(',');

	return { timeframe, startDate, endDate, symbols };
}

/**
 * Convert a YYYY-MM-DD date string to a whole-second UNIX epoch (UTC midnight), the period
 * format Yahoo's chart API expects. The end period is bumped by one day so the final day is inclusive.
 */
function toEpochSeconds(dateStr) {
	return Math.floor(Date.parse(`${dateStr}T00:00:00Z`) / 1000);
}

/**
 * Fetch one symbol's bars from Yahoo's chart API and normalize them into the Alpaca JSON bar shape.
 * Split/dividend adjustment is applied per bar via the adjclose/close factor so prices match the
 * `adjustment: 'all'` series Alpaca returns. Bars with null fields (Yahoo emits these on gaps) are
 * skipped, since a filled bar has no real open to adjust.
 */
async function fetchYahooBars(symbol, interval, startEpoch, endEpoch) {
	const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}`
		+ `?period1=${startEpoch}&period2=${endEpoch}&interval=${interval}&events=div%2Csplits`;

	// Yahoo returns 429/forbidden without a browser-like User-Agent.
	const response = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 (robot-brain stock downloader)' } });
	if (!response.ok) throw new Error(`Yahoo HTTP ${response.status} for ${symbol}`);

	const payload = await response.json();
	if (payload.chart?.error) throw new Error(`Yahoo error for ${symbol}: ${payload.chart.error.description}`);

	const result = payload.chart?.result?.[0];
	if (!result || !result.timestamp) return [];

	const timestamps = result.timestamp;
	const quote = result.indicators.quote[0];
	const adjclose = result.indicators.adjclose?.[0]?.adjclose ?? null;

	const bars = [];
	for (let i = 0; i < timestamps.length; i++) {
		const open = quote.open[i];
		const close = quote.close[i];
		const high = quote.high[i];
		const low = quote.low[i];
		const volume = quote.volume[i];

		// A bar needs a tradeable open price; without it there's nothing to record, so skip it.
		// A missing volume is treated as no-trade (0) rather than a reason to drop the whole day —
		// dropping a day would knock this symbol out of date-alignment with the others. setup.js keeps
		// the date on the shared grid and the encoder skips volume-0 rows, so no spurious signal results.
		if (open == null) continue;
		const vol = volume == null ? 0 : volume;

		// Scale raw prices to split/dividend-adjusted space. When adjclose or close is missing, factor is 1.
		const factor = adjclose && adjclose[i] != null && close != null && close !== 0 ? adjclose[i] / close : 1;

		bars.push({
			Timestamp: new Date(timestamps[i] * 1000).toISOString(),
			OpenPrice: open * factor,
			HighPrice: high != null ? high * factor : open * factor,
			LowPrice: low != null ? low * factor : open * factor,
			ClosePrice: close != null ? close * factor : open * factor,
			Volume: vol
		});
	}
	return bars;
}

/**
 * Pause for the given milliseconds — a small courtesy delay between symbols so the script
 * doesn't hammer Yahoo's public endpoint and trip rate limiting.
 */
function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Download historical data for a single symbol from Yahoo and save it as JSON.
 */
async function downloadSymbol(symbol, timeframe, interval, startEpoch, endEpoch, dataDir, progress) {
	const bars = await fetchYahooBars(symbol, interval, startEpoch, endEpoch);

	// Save raw bars to JSON — same filename/shape as the Alpaca downloader so setup.js is provider-agnostic.
	const filePath = path.join(dataDir, `${symbol}.json`);
	fs.writeFileSync(filePath, JSON.stringify(bars, null, 2));

	console.log(`   ${progress} ✅ ${symbol}.json: ${bars.length} bars`);
}

async function main() {
	const { timeframe, startDate, endDate, symbols } = parseArgs();

	// Daily-only by design — Yahoo's intraday lookback is capped (60m beyond ~730 days returns 422),
	// so hourly history is unreliable. Use download-alpaca.js for intraday timeframes.
	if (timeframe !== '1D') {
		console.error(`❌ Error: download-yahoo.js supports daily data only (--timeframe 1D), got '${timeframe}'.`);
		console.error(`   For intraday timeframes use download-alpaca.js.`);
		process.exit(1);
	}
	const interval = '1d';

	console.log('📥 Stock Data Download (Yahoo Finance)');
	console.log(`📊 Symbols: ${symbols.join(', ')}`);
	console.log(`⏱️  Timeframe: ${timeframe} (Yahoo interval: ${interval})`);
	console.log(`📅 Date Range: ${startDate} to ${endDate}`);
	console.log('');

	// Create data directory
	const dataDir = path.join(__dirname, '..', 'data', timeframe);
	if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

	const startEpoch = toEpochSeconds(startDate);
	// Bump the end by one day so the requested end date is inclusive.
	const endEpoch = toEpochSeconds(endDate) + 86400;

	// Download each symbol with a small delay between requests to stay under Yahoo's rate limit.
	for (let i = 0; i < symbols.length; i++) {
		const progress = `[${i + 1}/${symbols.length}]`;
		try {
			await downloadSymbol(symbols[i], timeframe, interval, startEpoch, endEpoch, dataDir, progress);
		}
		catch (error) {
			console.error(`   ${progress} ❌ ${symbols[i]}: ${error.message}`);
		}
		if (i < symbols.length - 1) await sleep(250);
	}

	console.log('');
	console.log('✅ All data downloaded successfully!');
	console.log(`   Next: node apps/stocks/jobs/setup.js --timeframe ${timeframe} --start ${startDate} --end ${endDate}`);
}

// Run the script
main().catch(error => {
	console.error('❌ Error:', error.message);
	process.exit(1);
});
