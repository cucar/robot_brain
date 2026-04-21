import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from '#brain-node';
import { StockEncoder } from '../encoder.js';
import { StockTrader } from '../trader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Stock Test Job - Trains or tests the brain on stock symbols
 * Can be used for both training (with holdout) and prediction (with offset)
 * Each episode runs through the data, then resets context but keeps learned patterns
 */
export default class StockTestJob extends Job {

	constructor() {
		super();

		// Simple configuration - edit these values as needed
		this.config = {
			// Stock symbols to train on
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
				// 'LCID', 'PLTR', 'SOFI', 'MARA', 'RIOT', 'GME', /* 'AMC', */ 'TWLO', 'ZM', 'SNOW'
			],
			timeframe: '3H',                     // Timeframe for data (e.g., '1D', '1Min')
			startDate: '2026-01-22',             // Start date for data download
			endDate: '2026-02-22',               // End date for data download
			maxEpisodes: 1,                      // Number of training episodes (can be overridden with --episodes)
			holdoutRows: 0,                      // Number of rows to hold out from end for prediction testing (can be overridden with --holdout)
			offsetRows: 0,                       // Number of rows to skip from start (can be overridden with --offset)
			extendedHours: false                 // Include extended hours data (pre-market/after-hours) - use --extended-hours
		};

		this.encoders = [];
		this.traders = [];

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	/**
	 * Apply command line options to config
	 */
	applyOptions() {
		const episodesIndex = process.argv.indexOf('--episodes');
		if (episodesIndex !== -1 && process.argv[episodesIndex + 1]) this.config.maxEpisodes = parseInt(process.argv[episodesIndex + 1]);

		const holdoutIndex = process.argv.indexOf('--holdout');
		if (holdoutIndex !== -1 && process.argv[holdoutIndex + 1]) this.config.holdoutRows = parseInt(process.argv[holdoutIndex + 1]);

		const offsetIndex = process.argv.indexOf('--offset');
		if (offsetIndex !== -1 && process.argv[offsetIndex + 1]) this.config.offsetRows = parseInt(process.argv[offsetIndex + 1]);

		const timeframeIndex = process.argv.indexOf('--timeframe');
		if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) this.config.timeframe = process.argv[timeframeIndex + 1];

		const startIndex = process.argv.indexOf('--start');
		if (startIndex !== -1 && process.argv[startIndex + 1]) this.config.startDate = process.argv[startIndex + 1];

		const endIndex = process.argv.indexOf('--end');
		if (endIndex !== -1 && process.argv[endIndex + 1]) this.config.endDate = process.argv[endIndex + 1];

		if (process.argv.includes('--extended-hours')) this.config.extendedHours = true;

		const symbolsIndex = process.argv.indexOf('--symbols');
		if (symbolsIndex !== -1 && process.argv[symbolsIndex + 1]) this.config.symbols = process.argv[symbolsIndex + 1].split(',');

		const maxPositionsIndex = process.argv.indexOf('--max-positions');
		if (maxPositionsIndex !== -1 && process.argv[maxPositionsIndex + 1]) StockTrader.maxPositions = parseInt(process.argv[maxPositionsIndex + 1]);

		const maxPriceIndex = process.argv.indexOf('--max-price');
		if (maxPriceIndex !== -1 && process.argv[maxPriceIndex + 1]) StockTrader.maxPrice = parseFloat(process.argv[maxPriceIndex + 1]);

		const initialCapitalIndex = process.argv.indexOf('--initial-capital');
		if (initialCapitalIndex !== -1 && process.argv[initialCapitalIndex + 1]) {
			StockTrader.initialCapital = parseFloat(process.argv[initialCapitalIndex + 1]);
			StockTrader.cash = StockTrader.initialCapital;
		}
	}

	/**
	 * No Channel classes registered with the brain — this job owns encoders/traders directly
	 * and overrides registerBrainChannels() below to use the spec-based path. The Job base
	 * class still calls getChannels(), so return an empty list to opt out of the legacy path.
	 */
	getChannels() {
		return [];
	}

	/**
	 * Setup method - Processes JSON data into CSV training files.
	 * Called by run-setup.js; requires stock-download.js to have already fetched raw JSON.
	 */
	async setup() {
		const timeframe = this.config.timeframe;
		const dataDir = path.join(__dirname, '..', 'data', timeframe);

		console.log(`📊 Processing stock data (${timeframe})...`);
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log('');

		// Check if JSON files exist
		console.log('📂 Checking for downloaded data...');
		for (const symbol of this.config.symbols) {
			const jsonPath = path.join(dataDir, `${symbol}.json`);
			if (!fs.existsSync(jsonPath)) {
				console.error(`❌ Error: ${symbol}.json not found in ${dataDir}`);
				console.error(`Please run: node apps/stocks/jobs/download.js --timeframe=${timeframe}`);
				process.exit(1);
			}
		}

		console.log('');
		console.log('📊 Processing data into training files...');

		// Load all symbols' data first
		const allBarMaps = new Map();
		for (const symbol of this.config.symbols) {
			const jsonPath = path.join(dataDir, `${symbol}.json`);
			const bars = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

			// Convert bars array to Map for processing
			const barMap = new Map();
			for (const bar of bars) {
				const timestamp = bar.Timestamp.substring(0, 16);
				barMap.set(timestamp, { open: bar.OpenPrice, volume: bar.Volume });
			}
			allBarMaps.set(symbol, barMap);
		}

		// For minute data, find valid intervals (where ALL stocks have non-zero volume)
		let validIntervals = null;
		if (timeframe !== '1D')
			validIntervals = this.findValidIntervals(allBarMaps);

		// Process each symbol's data into CSV
		for (const symbol of this.config.symbols) {
			const barMap = allBarMaps.get(symbol);
			await this.processAndSaveSymbolData(symbol, barMap, dataDir, validIntervals);
		}

		console.log('');
		console.log('✅ All data processed successfully!');
	}

	/**
	 * Find all intervals where at least one stock has data
	 * By default only includes regular trading hours (9:30 AM - 4:00 PM ET)
	 * With extendedHours option, includes pre-market and after-hours data
	 * @param {Map<string, Map<string, {open: number, volume: number}>>} allBarMaps - Map of symbol -> barMap
	 * @returns {Set<string>} Set of valid interval timestamps (YYYY-MM-DDTHH:MM format)
	 */
	findValidIntervals(allBarMaps) {

		// Collect all unique intervals from all stocks (filtered by hours setting)
		// Alpaca only returns bars with actual trading data (non-zero values)
		const intervals = new Set();
		for (const barMap of allBarMaps.values())
			for (const timestamp of barMap.keys())
				if (this.config.extendedHours || this.isRegularHours(new Date(timestamp + ':00Z')))
					intervals.add(timestamp);

		const hoursLabel = this.config.extendedHours ? 'extended hours' : 'regular hours';
		console.log(`   Found ${intervals.size} valid intervals (${hoursLabel}) where at least one stock has data`);
		return intervals;
	}

	/**
	 * Check if a UTC timestamp falls within regular trading hours (9:30 AM - 4:00 PM ET)
	 * @param {Date} utcDate - UTC date to check
	 * @returns {boolean} True if within regular hours
	 */
	isRegularHours(utcDate) {
		const etDate = new Date(utcDate.toLocaleString('en-US', { timeZone: 'America/New_York' }));
		const etHour = etDate.getHours();
		const etMinutes = etDate.getMinutes();
		const etTime = etHour * 60 + etMinutes;
		const regularOpen = 9 * 60 + 30;   // 9:30 AM ET
		const regularClose = 16 * 60;      // 4:00 PM ET
		return etTime >= regularOpen && etTime < regularClose;
	}

	/**
	 * Process and save symbol data in the format expected by StockChannel
	 * Format: price,volume (no header, no timestamp, chronological order)
	 * For minute data: Only includes intervals where ALL stocks have data
	 * For daily data: Uses bars as-is without gap filling
	 * @param {string} symbol - Stock symbol
	 * @param {Map<string, {open: number, volume: number}>} barMap - Map of timestamp -> bar data
	 * @param {string} dataDir - Directory to save CSV files
	 * @param {Set<string>|null} validIntervals - Set of valid interval timestamps (for minute data only)
	 */
	async processAndSaveSymbolData(symbol, barMap, dataDir, validIntervals = null) {

		// For daily data, filter by startDate and endDate
		let filledData;
		if (this.config.timeframe === '1D') {
			const timestamps = Array.from(barMap.keys()).sort();
			const filteredTimestamps = timestamps.filter(timestamp => {
				const date = timestamp.substring(0, 10); // Extract YYYY-MM-DD
				return date >= this.config.startDate && date <= this.config.endDate;
			});
			filledData = filteredTimestamps.map(timestamp => ({
				open: barMap.get(timestamp).open,
				volume: barMap.get(timestamp).volume
			}));
		}
		// For minute data, only include valid intervals (where all stocks have data)
		else filledData = this.extractValidIntervals(symbol, barMap, validIntervals);

		// Format as CSV rows: price,volume (no timestamp)
		const rows = filledData.map(bar => `${bar.open},${bar.volume}`);

		// Write to CSV file (no header, chronological order)
		const filePath = path.join(dataDir, `${symbol}.csv`);
		fs.writeFileSync(filePath, rows.join('\n'));

		console.log(`   ✅ ${symbol}.csv: ${rows.length} bars`);
	}

	/**
	 * Extract bars for all valid intervals, using last known price with 0 volume for missing data
	 * This ensures the brain sees a "0% price change" event (categorized as down) for missing bars
	 * @param {string} symbol - Stock symbol
	 * @param {Map<string, {open: number, volume: number}>} barMap - Map of timestamp -> bar data
	 * @param {Set<string>} validIntervals - Set of valid interval timestamps (YYYY-MM-DDTHH:MM format)
	 * @returns {Array<{open: number, volume: number}>} Array of bars for valid intervals
	 */
	extractValidIntervals(symbol, barMap, validIntervals) {

		// Sort valid intervals chronologically
		const sortedIntervals = Array.from(validIntervals).sort();

		// Find the first available price (for filling gaps at the beginning)
		const firstKnownPrice = this.findFirstKnownPrice(symbol, barMap, sortedIntervals);

		// Extract bars for each valid interval
		// For missing data at beginning: use first known price (fill from future)
		// For missing data in middle/end: use last known price (fill from past)
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
	 * Find the first available price for a symbol in the valid intervals
	 * @param {string} symbol - Stock symbol (for error messages)
	 * @param {Map<string, {open: number, volume: number}>} barMap - Map of timestamp -> bar data
	 * @param {string[]} sortedIntervals - Chronologically sorted interval timestamps
	 * @returns {number} First available price
	 */
	findFirstKnownPrice(symbol, barMap, sortedIntervals) {
		for (const interval of sortedIntervals) {
			const bar = barMap.get(interval);
			if (bar) return bar.open;
		}
		throw new Error(`No data at all for: ${symbol}`);
	}

	/**
	 * Create an encoder + trader per symbol and register the encoder's spec with the brain.
	 * The trader borrows the encoder's channelId so rewards, inputs, and inferences all key
	 * off a single number per symbol — otherwise we'd need a second id→symbol lookup.
	 * Called before brain.init(), which then creates the dimensions/neurons from each spec.
	 */
	async registerBrainChannels() {
		for (const symbol of this.config.symbols) {
			const encoder = new StockEncoder(symbol);
			const trader = new StockTrader(symbol);
			trader.channelId = encoder.channelId;
			this.encoders.push(encoder);
			this.traders.push(trader);
			this.brain.registerChannelSpec(encoder.getChannelSpec());
		}
	}

	/**
	 * Hook: after brain init, load each symbol's CSV, slice off the holdout (from the tail)
	 * and offset (from the head), and hand the resulting chronological rows to the encoder.
	 * Runs once per job, before any episodes — rows are reused across episodes via resetFrames().
	 */
	async configureChannels() {
		const { timeframe, holdoutRows, offsetRows } = this.config;
		const dataDir = path.join(__dirname, '..', 'data', timeframe);

		for (const encoder of this.encoders) {
			const csvPath = path.join(dataDir, `${encoder.symbol}.csv`);
			const allRows = this.loadCsvRows(csvPath);

			// Offset trims from the start (skip warmup bars), holdout trims from the end
			// (reserved for out-of-sample prediction testing in a separate job invocation).
			const startIndex = offsetRows;
			const endIndex = holdoutRows > 0 ? allRows.length - holdoutRows : allRows.length;
			const rows = allRows.slice(startIndex, endIndex);

			encoder.setData(rows);
		}
	}

	/**
	 * Load and parse a CSV file into {price, volume} row objects. Rows are already in
	 * chronological order from processAndSaveSymbolData, so no sort is needed. Empty lines
	 * are filtered so a trailing newline doesn't produce a NaN row.
	 */
	loadCsvRows(csvPath) {
		const content = fs.readFileSync(csvPath, 'utf-8');
		return content.split('\n')
			.filter(line => line.trim())
			.map(line => {
				const parts = line.trim().split(',');
				return { price: parseFloat(parts[0]), volume: parseFloat(parts[1]) };
			});
	}

	/**
	 * Hook: Show startup information
	 */
	async showStartupInfo() {
		console.log(`🚀 Starting Stock Test Job`);
		console.log(`📊 Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`⏱️  Timeframe: ${this.config.timeframe}`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`📋 Holdout Rows: ${this.config.holdoutRows}`);
		console.log(`📋 Offset Rows: ${this.config.offsetRows}`);
		console.log('');
	}

	/**
	 * Hook: Execute main job logic - multi-episode training
	 */
	async executeJob() {
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {

			// Run episode
			await this.runEpisode();

			// if interrupt is received, stop processing
			if (this.isShuttingDown) return;

			// Show progress every 10 episodes or on last episode
			if (this.currentEpisode % 10 === 0 || this.currentEpisode === this.config.maxEpisodes)
				this.showProgress();
		}
	}

	/**
	 * Hook: Show results
	 */
	async showResults() {
		this.showFinalResults();
	}

	/**
	 * Run one episode: reset portfolio cash, rewind each encoder to frame 0, and wipe each
	 * trader's per-episode context (position, last action, etc.). Then stream frames through
	 * the brain, apply inferences back to the traders, and let the portfolio coordinate
	 * execution per frame. Learned brain state (dimensions, neuron weights) is preserved —
	 * only episode context is reset, so successive episodes build on prior learning.
	 */
	async runEpisode() {
		const startTime = Date.now();
		console.log(`📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes}... `);

		// Reset context but keep learned patterns
		this.brain.resetContext();
		StockTrader.resetPortfolio();
		for (const encoder of this.encoders) encoder.resetFrames();
		for (const trader of this.traders) trader.resetContext();

		// to test hard resets between episodes:
		// this.brain.thalamus.reset();
		// this.brain.thalamus.initializeActionNeurons();

		// Dump brain data at the beginning of each episode for debugging
		// this.brain.createDump();

		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			netProfit: 0,
			totalTrades: 0,
			channelResults: new Map(),
			baseAccuracy: null,
			overallAccuracy: null
		};

		// Warmup: consume the first frame per encoder so the first real processed frame
		// has a full (previous, current) pair. Mirrors the legacy frame-1 double-read in
		// StockChannel — without this, the first encoded frame would have previousPrice=null
		// and be skipped, and the trader would start one frame behind the encoder.
		for (let i = 0; i < this.encoders.length; i++) {
			const frame = this.encoders[i].nextFrame();
			if (frame) this.traders[i].setFrame(frame.price, frame.volume);
		}

		// Every encoder has the same row count by construction (aligned intervals). One
		// frame was consumed by the warmup above, so the main loop processes one fewer.
		const expectedFrames = this.encoders[0].rows.length - 1;

		// Process all frames for the episode duration
		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Process frame
			const hasMore = await this.runFrame();
			if (!hasMore) break;
			frameCount++;

			// Show progress every 100 frames
			if (frameCount % 100 === 0)
				process.stdout.write(`\r📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes} - Frame ${frameCount}/${expectedFrames}... `);

			// if interrupt is received, stop processing
			if (this.isShuttingDown) return;
		}

		// Clear progress line (only if stdout is a TTY)
		if (!process.stdout.isTTY) process.stdout.write(`\n`);
		else {
			process.stdout.write(`\r`);
			process.stdout.clearLine(0);
		}

		// Set frame count and duration first (needed for ROI calculation)
		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		// Collect episode results from all channels (includes ROI calculation)
		this.collectEpisodeResults(episodeMetrics);

		// Capture base level accuracy stats
		const summary = this.brain.getEpisodeSummary();
		if (summary.accuracy.total > 0)
			episodeMetrics.baseAccuracy = (summary.accuracy.correct / summary.accuracy.total * 100);
		this.episodeResults.push(episodeMetrics);

		// Dump brain data at the beginning of each episode for debugging
		// this.brain.createDump();

		// Format ROI output
		const roiStr = episodeMetrics.totalROIPercent >= 0 ? `+${episodeMetrics.totalROIPercent.toFixed(2)}%` : `${episodeMetrics.totalROIPercent.toFixed(2)}%`;
		const perFrameROIStr = episodeMetrics.perFrameROI !== undefined ? `, ${(episodeMetrics.perFrameROIPercent >= 0 ? '+' : '')}${episodeMetrics.perFrameROIPercent.toFixed(6)}%/frame` : '';

		console.log(`✅ Net: $${episodeMetrics.netProfit.toFixed(2)} | ROI: ${roiStr} over ${episodeMetrics.frames} frames${perFrameROIStr} (${episodeMetrics.totalTrades} trades, ${duration}ms)`);
	}

	/**
	 * Process one frame: pull the next frame per encoder, hand the encoded scalars and
	 * any per-trader rewards to the brain, apply the returned inferences back to each
	 * trader, and let the portfolio coordinate the resulting buys/sells.
	 * @returns {Promise<boolean>} false when all encoders are exhausted
	 */
	async runFrame() {
		const inputs = new Map();
		const rewards = new Map();

		// Pull one frame per encoder. Even frames that don't produce an encoded input
		// (e.g. zero-volume bar, first frame) still update the trader's price/volume so
		// valuation and reward math see the most recent reading.
		let anyFrames = false;
		for (let i = 0; i < this.encoders.length; i++) {
			const encoder = this.encoders[i];
			const trader = this.traders[i];
			const frame = encoder.nextFrame();
			if (!frame) continue;
			anyFrames = true;
			trader.setFrame(frame.price, frame.volume);
			const dimMap = encoder.encode(frame);
			if (dimMap) inputs.set(encoder.channelId, dimMap);
		}
		if (!anyFrames) return false;

		// Only report reward for traders that actually acted last frame. This matches the
		// legacy path, which called channel.getRewards() only when the channel had inferred
		// actions in the previous frame — reporting a reward for a trader that never acted
		// would credit/punish neurons that weren't responsible.
		for (const trader of this.traders)
			if (trader.lastAction !== null) rewards.set(trader.channelId, trader.getReward());

		// Brain returns inferences keyed by channelId. Each trader grabs its own (or an
		// empty array if the channel didn't fire) and records its last action for next frame.
		const inferences = this.brain.processInputs(inputs, rewards);

		for (const trader of this.traders)
			trader.apply(inferences.get(trader.channelId) ?? []);

		// Coordinated portfolio execution: ranks OWN actions, sizes positions, and runs
		// sells-then-buys so cash is freed before it's spent.
		await StockTrader.executePortfolio(this.traders);

		// Yield to the event loop so SIGINT handlers can fire between frames — without this,
		// a tight synchronous loop ignores Ctrl+C until the episode finishes.
		await new Promise(resolve => setImmediate(resolve));
		return true;
	}

	/**
	 * Populate episodeMetrics from the traders themselves — no Channel class aggregation.
	 * Computes portfolio-level ROI (total and compounded per-frame) plus per-symbol stats
	 * for the results display.
	 */
	collectEpisodeResults(episodeMetrics) {

		// Portfolio profit = (cash + market value of all positions) - initial capital.
		episodeMetrics.netProfit = StockTrader.getPortfolioProfit(this.traders);

		// Total ROI as a ratio (1.0 == breakeven) and as a percentage delta.
		const finalValue = StockTrader.initialCapital + episodeMetrics.netProfit;
		const totalROI = finalValue / StockTrader.initialCapital;
		episodeMetrics.totalROI = totalROI;
		episodeMetrics.totalROIPercent = (totalROI - 1) * 100;

		// Geometric per-frame return — the constant rate that, compounded over `frames`
		// frames, reproduces the episode's total ROI. Makes episodes of different lengths
		// comparable.
		if (episodeMetrics.frames > 0) {
			const perFrameROI = Math.pow(totalROI, 1 / episodeMetrics.frames) - 1;
			episodeMetrics.perFrameROI = perFrameROI;
			episodeMetrics.perFrameROIPercent = perFrameROI * 100;
		}

		// Per-symbol breakdown for the results table. Unrealized profit uses current market
		// price vs. cost basis — realized P&L is already baked into StockTrader.cash.
		for (const trader of this.traders) {
			const currentValue = trader.shares * trader.getCurrentPrice();

			const channelResult = {
				symbol: trader.symbol,
				investment: trader.investment,
				currentValue,
				unrealizedProfit: currentValue - trader.investment,
				trades: trader.totalTrades
			};

			episodeMetrics.channelResults.set(trader.symbol, channelResult);
			episodeMetrics.totalTrades += trader.totalTrades;
		}
	}

	/**
	 * Show training progress
	 */
	showProgress() {
		console.log(`\n📊 Training Progress (Episode ${this.currentEpisode}/${this.config.maxEpisodes}):`);

		if (this.episodeResults.length >= 10) {
			const recent10 = this.episodeResults.slice(-10);
			const avgProfit = recent10.reduce((sum, ep) => sum + ep.netProfit, 0) / recent10.length;
			const avgTrades = recent10.reduce((sum, ep) => sum + ep.totalTrades, 0) / recent10.length;

			console.log(`   Last 10 episodes avg: $${avgProfit.toFixed(2)} net profit, ${avgTrades.toFixed(1)} trades`);
		}

		const bestEpisode = this.episodeResults.reduce((best, ep) => ep.netProfit > best.netProfit ? ep : best);
		const worstEpisode = this.episodeResults.reduce((worst, ep) => ep.netProfit < worst.netProfit ? ep : worst);

		console.log(`   Best episode: #${bestEpisode.episode} ($${bestEpisode.netProfit.toFixed(2)})`);
		console.log(`   Worst episode: #${worstEpisode.episode} ($${worstEpisode.netProfit.toFixed(2)})`);
		console.log('');
	}

	/**
	 * Show final training results
	 */
	showFinalResults() {
		console.log(`\n🎯 Final Training Results (${this.config.maxEpisodes} episodes):`);
		console.log('='.repeat(60));

		const totalNetProfit = this.episodeResults.reduce((sum, ep) => sum + ep.netProfit, 0);
		const avgNetProfit = totalNetProfit / this.episodeResults.length;
		const totalTrades = this.episodeResults.reduce((sum, ep) => sum + ep.totalTrades, 0);
		const avgTrades = totalTrades / this.episodeResults.length;

		// Calculate average ROI metrics
		const avgTotalROI = this.episodeResults.reduce((sum, ep) => sum + (ep.totalROIPercent || 0), 0) / this.episodeResults.length;
		const avgPerFrameROI = this.episodeResults.reduce((sum, ep) => sum + (ep.perFrameROIPercent || 0), 0) / this.episodeResults.length;

		console.log(`📈 Overall Performance:`);
		console.log(`   Starting Capital: $${StockTrader.initialCapital.toFixed(2)}`);
		console.log(`   Total Net Profit: $${totalNetProfit.toFixed(2)}`);
		console.log(`   Average per Episode: $${avgNetProfit.toFixed(2)}`);
		console.log(`   Average ROI: ${avgTotalROI >= 0 ? '+' : ''}${avgTotalROI.toFixed(2)}%`);
		console.log(`   Average Per-Frame ROI: ${avgPerFrameROI >= 0 ? '+' : ''}${avgPerFrameROI.toFixed(6)}%`);
		console.log(`   Total Trades: ${totalTrades}`);
		console.log(`   Average Trades per Episode: ${avgTrades.toFixed(1)}`);

		// Show net profit and ROI per episode
		console.log(`\n💰 Net Profit & ROI by Episode:`);
		for (const ep of this.episodeResults) {
			const roiStr = ep.totalROIPercent >= 0 ? `+${ep.totalROIPercent.toFixed(2)}%` : `${ep.totalROIPercent.toFixed(2)}%`;
			const perFrameROIStr = ep.perFrameROI !== undefined ? `, ${(ep.perFrameROIPercent >= 0 ? '+' : '')}${ep.perFrameROIPercent.toFixed(6)}%/frame` : '';
			console.log(`   Episode ${ep.episode}: $${ep.netProfit.toFixed(2)} | ROI: ${roiStr}${perFrameROIStr} (${ep.totalTrades} trades)`);
		}

		// Show base level accuracy per episode
		console.log(`\n📊 Base Level Accuracy by Episode:`);
		for (const ep of this.episodeResults) {
			if (ep.baseAccuracy !== null)
				console.log(`   Episode ${ep.episode}: ${ep.baseAccuracy.toFixed(2)}%`);
			else
				console.log(`   Episode ${ep.episode}: N/A`);
		}

		// Show improvement trend
		if (this.episodeResults.length >= 20) {
			const first10 = this.episodeResults.slice(0, 10);
			const last10 = this.episodeResults.slice(-10);

			const firstAvg = first10.reduce((sum, ep) => sum + ep.netProfit, 0) / first10.length;
			const lastAvg = last10.reduce((sum, ep) => sum + ep.netProfit, 0) / last10.length;
			const improvement = lastAvg - firstAvg;

			const firstROI = first10.reduce((sum, ep) => sum + (ep.totalROIPercent || 0), 0) / first10.length;
			const lastROI = last10.reduce((sum, ep) => sum + (ep.totalROIPercent || 0), 0) / last10.length;
			const roiImprovement = lastROI - firstROI;

			console.log(`\n📊 Learning Progress:`);
			console.log(`   First 10 episodes avg: $${firstAvg.toFixed(2)} (${firstROI >= 0 ? '+' : ''}${firstROI.toFixed(2)}% ROI)`);
			console.log(`   Last 10 episodes avg: $${lastAvg.toFixed(2)} (${lastROI >= 0 ? '+' : ''}${lastROI.toFixed(2)}% ROI)`);
			console.log(`   Improvement: $${improvement.toFixed(2)}, ${roiImprovement >= 0 ? '+' : ''}${roiImprovement.toFixed(2)}pp ROI (${improvement >= 0 ? '📈' : '📉'})`);
		}

		console.log('\n🏆 Best Episodes (by ROI):');
		const sortedByROI = [...this.episodeResults].sort((a, b) => (b.totalROIPercent || 0) - (a.totalROIPercent || 0));
		for (let i = 0; i < Math.min(5, sortedByROI.length); i++) {
			const ep = sortedByROI[i];
			const roiStr = ep.totalROIPercent >= 0 ? `+${ep.totalROIPercent.toFixed(2)}%` : `${ep.totalROIPercent.toFixed(2)}%`;
			console.log(`   #${ep.episode}: ${roiStr} ROI ($${ep.netProfit.toFixed(2)}, ${ep.totalTrades} trades)`);
		}

		console.log('='.repeat(60));
	}
}

await runJob(import.meta, StockTestJob);