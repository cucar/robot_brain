import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Stock Data Setup Script - Processes raw JSON bars downloaded from Alpaca into CSV
 * training files (price,volume per row, chronological order, no header).
 *
 * Requires download.js to have populated apps/stocks/data/{timeframe}/{symbol}.json first.
 * This is a plain script: no brain, no Job class — just file processing with its own
 * argument parsing, mirroring download.js.
 *
 * Run with: node apps/stocks/jobs/setup.js --timeframe=3H
 */

const config = {
	symbols: [
		'KGC', 'GLD', 'SPY', 'AAPL', 'NEM', 'GDX', 'NVDA', 'AMZN', 'MSFT', 'AMD',
		// 'META', 'JPM', 'BAC', 'QQQ', 'IWM', 'AEM', 'WPM', 'NG', 'GOOGL', 'XOM',
		// 'CVX', 'JNJ', 'UNH', 'PFE', 'WMT', 'COST', 'KO', 'CAT', 'XLF', 'DIA',
		// 'INTC', 'CRM', 'ORCL', 'IBM', 'CSCO', 'TGT', 'HD', 'MCD', 'NKE', 'SBUX',
		// 'ABBV', 'MRK', 'BMY', 'LLY', 'GILD', 'SLB', 'OXY', 'FCX', 'MOS', 'CLF',
		// 'ADBE', 'NFLX', 'PYPL', 'SHOP', 'UBER', 'ABNB', 'SNAP', 'PINS', 'ROKU', 'GS',
		// 'MS', 'C', 'WFC', 'AXP', 'V', 'MA', 'COF', 'SCHW', 'BLK', 'BA',
		// 'LMT', 'GE', 'UPS', 'FDX', 'DE', 'HON', 'RTX', 'UNP', 'DAL', 'DIS',
		// 'CMCSA', 'PEP', 'PM', 'MO', 'CL', 'PG', 'EL', 'LULU', 'F', 'COIN',
		// 'LCID', 'PLTR', 'SOFI', 'MARA', 'RIOT', 'GME', 'AMC', 'TWLO', 'ZM', 'SNOW'
	],
	startDate: '2026-01-22',
	endDate: '2026-02-22'
};

/**
 * Parse command line arguments. Flags mirror download.js so the two scripts can be
 * chained with the same invocation shape.
 */
function parseArgs() {
	let timeframe = '3H';
	const timeframeIndex = process.argv.indexOf('--timeframe');
	if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) timeframe = process.argv[timeframeIndex + 1];

	let startDate = config.startDate;
	const startIndex = process.argv.indexOf('--start');
	if (startIndex !== -1 && process.argv[startIndex + 1]) startDate = process.argv[startIndex + 1];

	let endDate = config.endDate;
	const endIndex = process.argv.indexOf('--end');
	if (endIndex !== -1 && process.argv[endIndex + 1]) endDate = process.argv[endIndex + 1];

	// Include pre-market and after-hours bars in the output when set. Off by default so
	// the brain only trains on regular-session data (9:30am–4:00pm ET).
	const extendedHours = process.argv.includes('--extended-hours');

	let symbols = config.symbols;
	const symbolsIndex = process.argv.indexOf('--symbols');
	if (symbolsIndex !== -1 && process.argv[symbolsIndex + 1]) symbols = process.argv[symbolsIndex + 1].split(',');

	return { timeframe, startDate, endDate, extendedHours, symbols };
}

/**
 * Check if a UTC timestamp falls within regular trading hours (9:30 AM - 4:00 PM ET).
 * Converts to ET via toLocaleString so DST is handled by the runtime rather than
 * hardcoded offsets.
 */
function isRegularHours(utcDate) {
	const etDate = new Date(utcDate.toLocaleString('en-US', { timeZone: 'America/New_York' }));
	const etTime = etDate.getHours() * 60 + etDate.getMinutes();
	const regularOpen = 9 * 60 + 30;   // 9:30 AM ET
	const regularClose = 16 * 60;      // 4:00 PM ET
	return etTime >= regularOpen && etTime < regularClose;
}

/**
 * Collect every timestamp across all symbols where at least one stock traded. Intraday
 * timeframes filter by market hours; the union is our set of "valid intervals" that
 * every symbol will be aligned against — so multi-symbol frames stay in lockstep even
 * when individual stocks have missing bars.
 */
function findValidIntervals(allBarMaps, extendedHours) {
	const intervals = new Set();
	for (const barMap of allBarMaps.values())
		for (const timestamp of barMap.keys())
			if (extendedHours || isRegularHours(new Date(timestamp + ':00Z')))
				intervals.add(timestamp);

	const hoursLabel = extendedHours ? 'extended hours' : 'regular hours';
	console.log(`   Found ${intervals.size} valid intervals (${hoursLabel}) where at least one stock has data`);
	return intervals;
}

/**
 * Find the first available price for a symbol in the valid intervals — used as the fill
 * value for missing bars at the start of the series (before this symbol's first trade).
 * Throws if the symbol has no bars at all, since there's nothing to fill with.
 */
function findFirstKnownPrice(symbol, barMap, sortedIntervals) {
	for (const interval of sortedIntervals) {
		const bar = barMap.get(interval);
		if (bar) return bar.open;
	}
	throw new Error(`No data at all for: ${symbol}`);
}

/**
 * Extract bars aligned to the full valid-interval grid. Missing bars are filled:
 * leading gaps use the symbol's first known price (fill-from-future), middle and
 * trailing gaps use the last known price (fill-from-past). Filled bars get volume=0
 * so the brain sees a 0% price change event rather than a real trade.
 */
function extractValidIntervals(symbol, barMap, validIntervals) {
	const sortedIntervals = Array.from(validIntervals).sort();
	const firstKnownPrice = findFirstKnownPrice(symbol, barMap, sortedIntervals);

	const result = [];
	let lastKnownPrice = firstKnownPrice;
	for (const interval of sortedIntervals) {
		const bar = barMap.get(interval);
		if (bar) {
			lastKnownPrice = bar.open;
			result.push({ open: bar.open, volume: bar.volume });
		}
		else result.push({ open: lastKnownPrice, volume: 0 });
	}
	return result;
}

/**
 * Process and save one symbol's data. Daily bars are filtered by date range and used
 * as-is (no gap filling — market holidays are genuine absences, not missing data).
 * Intraday bars are aligned to the shared valid-interval grid so every symbol's CSV
 * has identical row count and timestamp ordering.
 */
function processAndSaveSymbolData(symbol, barMap, dataDir, timeframe, startDate, endDate, validIntervals = null) {
	let filledData;
	if (timeframe === '1D') {
		const timestamps = Array.from(barMap.keys()).sort();
		const filteredTimestamps = timestamps.filter(timestamp => {
			const date = timestamp.substring(0, 10);
			return date >= startDate && date <= endDate;
		});
		filledData = filteredTimestamps.map(timestamp => ({
			open: barMap.get(timestamp).open,
			volume: barMap.get(timestamp).volume
		}));
	}
	else filledData = extractValidIntervals(symbol, barMap, validIntervals);

	// CSV format: price,volume per row. No header, no timestamp — the test job reads
	// rows in order and treats them as chronological frames.
	const rows = filledData.map(bar => `${bar.open},${bar.volume}`);
	const filePath = path.join(dataDir, `${symbol}.csv`);
	fs.writeFileSync(filePath, rows.join('\n'));

	console.log(`   ✅ ${symbol}.csv: ${rows.length} bars`);
}

async function main() {
	const { timeframe, startDate, endDate, extendedHours, symbols } = parseArgs();
	const dataDir = path.join(__dirname, '..', 'data', timeframe);

	console.log(`📊 Processing stock data (${timeframe})...`);
	console.log(`   Symbols: ${symbols.join(', ')}`);
	console.log('');

	// Verify JSON files exist before doing any work, so we fail fast with a clear
	// error pointing to the download step rather than mid-processing.
	console.log('📂 Checking for downloaded data...');
	for (const symbol of symbols) {
		const jsonPath = path.join(dataDir, `${symbol}.json`);
		if (!fs.existsSync(jsonPath)) {
			console.error(`❌ Error: ${symbol}.json not found in ${dataDir}`);
			console.error(`Please run: node apps/stocks/jobs/download.js --timeframe=${timeframe}`);
			process.exit(1);
		}
	}

	console.log('');
	console.log('📊 Processing data into training files...');

	// Load every symbol's bars up front into Maps keyed by truncated timestamp (minute-
	// precision) — lets findValidIntervals and extractValidIntervals do O(1) lookups.
	const allBarMaps = new Map();
	for (const symbol of symbols) {
		const jsonPath = path.join(dataDir, `${symbol}.json`);
		const bars = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

		const barMap = new Map();
		for (const bar of bars) {
			const timestamp = bar.Timestamp.substring(0, 16);
			barMap.set(timestamp, { open: bar.OpenPrice, volume: bar.Volume });
		}
		allBarMaps.set(symbol, barMap);
	}

	// Daily data has no intraday grid to align against; the date filter in
	// processAndSaveSymbolData handles range selection directly.
	let validIntervals = null;
	if (timeframe !== '1D') validIntervals = findValidIntervals(allBarMaps, extendedHours);

	for (const symbol of symbols) {
		const barMap = allBarMaps.get(symbol);
		processAndSaveSymbolData(symbol, barMap, dataDir, timeframe, startDate, endDate, validIntervals);
	}

	console.log('');
	console.log('✅ All data processed successfully!');
}

main().catch(error => {
	console.error('❌ Error:', error.message);
	console.error(error.stack);
	process.exit(1);
});
