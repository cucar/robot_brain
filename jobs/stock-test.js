import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job } from './job.js';
import { StockChannel } from '../channels/stock.js';

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
				'META', 'JPM', 'BAC', 'QQQ', 'IWM', 'AEM', 'WPM', 'NG', 'GOOGL', 'XOM', 'CVX',
				'JNJ', 'UNH', 'PFE', 'WMT', 'COST', 'KO', 'CAT', 'XLF', 'DIA', 'INTC', 'CRM', 'ORCL',
				'IBM', 'CSCO', 'TGT', 'HD', 'MCD', 'NKE', 'SBUX', 'ABBV', 'MRK', 'BMY', 'LLY', 'GILD',
				'SLB', 'OXY', 'FCX', 'MOS', 'CLF'
			],
			timeframe: '1Min',                   // Timeframe for data (e.g., '1D', '1Min')
			startDate: '2021-02-22',             // Start date for data download
			endDate: '2026-02-22',               // End date for data download
			maxEpisodes: 1,                      // Number of training episodes (can be overridden with --episodes)
			holdoutRows: 0,                      // Number of rows to hold out from end for prediction testing (can be overridden with --holdout)
			offsetRows: 0                        // Number of rows to skip from start (can be overridden with --offset)
		};

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	/**
	 * Apply command line options to config
	 */
	applyOptions(options) {
		if (options.episodes !== null && options.episodes !== undefined) this.config.maxEpisodes = options.episodes;
		if (options.holdout !== null && options.holdout !== undefined) this.config.holdoutRows = options.holdout;
		if (options.offset !== null && options.offset !== undefined) this.config.offsetRows = options.offset;
		if (options.timeframe !== null && options.timeframe !== undefined) this.config.timeframe = options.timeframe;
		if (options.start !== null && options.start !== undefined) this.config.startDate = options.start;
		if (options.end !== null && options.end !== undefined) this.config.endDate = options.end;
	}

	/**
	 * Setup method - Processes JSON data into CSV training files
	 * Requires: Run node stock-download.js first to download data
	 */
	async setup() {
		const timeframe = this.config.timeframe;
		const dataDir = path.join(__dirname, '..', 'data', 'stock', timeframe);

		console.log(`📊 Processing stock data (${timeframe})...`);
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log('');

		// Check if JSON files exist
		console.log('📂 Checking for downloaded data...');
		for (const symbol of this.config.symbols) {
			const jsonPath = path.join(dataDir, `${symbol}.json`);
			if (!fs.existsSync(jsonPath)) {
				console.error(`❌ Error: ${symbol}.json not found in ${dataDir}`);
				console.error(`Please run: node stock-download.js --timeframe=${timeframe}`);
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

		// For minute data, find valid dates (where at least one stock has data)
		let validDates = null;
		if (timeframe !== '1D')
			validDates = this.findValidDates(allBarMaps);

		// Process each symbol's data into CSV
		for (const symbol of this.config.symbols) {
			const barMap = allBarMaps.get(symbol);
			await this.processAndSaveSymbolData(symbol, barMap, dataDir, validDates);
		}

		console.log('');
		console.log('✅ All data processed successfully!');
	}

	/**
	 * Find all valid dates where at least one stock has data
	 * @param {Map<string, Map<string, {open: number, volume: number}>>} allBarMaps - Map of symbol -> barMap
	 * @returns {Set<string>} Set of valid dates in YYYY-MM-DD format
	 */
	findValidDates(allBarMaps) {
		const validDates = new Set();

		// Collect all dates from all stocks
		for (const barMap of allBarMaps.values()) {
			for (const timestamp of barMap.keys()) {
				const date = timestamp.substring(0, 10); // Extract YYYY-MM-DD
				validDates.add(date);
			}
		}

		return validDates;
	}

	/**
	 * Process and save symbol data in the format expected by StockChannel
	 * Format: price,volume (no header, no timestamp, chronological order)
	 * For minute data: Only includes regular trading hours (9:30 AM - 4:00 PM ET)
	 * For daily data: Uses bars as-is without gap filling
	 * @param {string} symbol - Stock symbol
	 * @param {Map<string, {open: number, volume: number}>} barMap - Map of timestamp -> bar data
	 * @param {string} dataDir - Directory to save CSV files
	 * @param {Set<string>|null} validDates - Set of valid dates in YYYY-MM-DD format (for minute data only)
	 */
	async processAndSaveSymbolData(symbol, barMap, dataDir, validDates = null) {

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
		// For minute data, only include valid dates
		else filledData = this.fillRegularHoursOnly(barMap, validDates);

		// Format as CSV rows: price,volume (no timestamp)
		const rows = filledData.map(bar => `${bar.open},${bar.volume}`);

		// Write to CSV file (no header, chronological order)
		const filePath = path.join(dataDir, `${symbol}.csv`);
		fs.writeFileSync(filePath, rows.join('\n'));

		console.log(`   ✅ ${symbol}.csv: ${rows.length} bars`);
	}

	/**
	 * Generate complete grid of regular trading hours (9:30 AM - 4:00 PM ET) for valid trading days only
	 * Fills missing bars with previous price and 0 volume
	 * @param {Map<string, {open: number, volume: number}>} barMap - Map of timestamp -> bar data
	 * @param {Set<string>} validDates - Set of valid dates in YYYY-MM-DD format
	 * @returns {Array<{open: number, volume: number}>} Complete array of bars for regular hours only
	 */
	fillRegularHoursOnly(barMap, validDates) {

		// Parse start and end dates
		const start = new Date(this.config.startDate);
		const end = new Date(this.config.endDate);

		// Get the first bar's price to use as initial price for all stocks
		const timestamps = Array.from(barMap.keys()).sort();
		const firstBar = timestamps.length > 0 ? barMap.get(timestamps[0]) : null;
		let lastPrice = firstBar ? firstBar.open : null;

		// Determine time interval in minutes based on timeframe
		const intervalMinutes = this.getIntervalMinutes(this.config.timeframe);

		const result = [];

		// Loop through every interval from start to end
		const current = new Date(start);
		current.setUTCSeconds(0, 0);
		while (current <= end) {

			// Get the date for this timestamp
			const dateStr = current.toISOString().substring(0, 10);

			// Only process if this date has data for at least one stock
			if (validDates.has(dateStr)) {

				// Only output intervals within regular trading hours
				if (this.isRegularHours(current)) {

					// Check if we have a bar for this interval
					const timestampKey = current.toISOString().substring(0, 16);
					const bar = barMap.get(timestampKey);

					// if we do have a bar, add it to the result and update last price
					if (bar) {
						result.push({ open: bar.open, volume: bar.volume });
						lastPrice = bar.open;
					}
					// No bar - fill with last price and 0 volume
					else if (lastPrice !== null) result.push({ open: lastPrice, volume: 0 });
				}
			}

			// move to next interval
			current.setUTCMinutes(current.getUTCMinutes() + intervalMinutes);
		}

		return result;
	}

	/**
	 * Get the interval in minutes for a given timeframe
	 * @param {string} timeframe - Timeframe string (e.g., '1Min', '5Min', '15Min')
	 * @returns {number} Interval in minutes
	 */
	getIntervalMinutes(timeframe) {
		if (timeframe.endsWith('Min')) return parseInt(timeframe.replace('Min', ''));
		if (timeframe.endsWith('H')) return parseInt(timeframe.replace('H', '')) * 60;
		if (timeframe === '1D') return 1440;
		return 1; // default to 1 minute
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
	 * Returns the channels for the job - one channel per stock symbol
	 */
	getChannels() {
		return this.config.symbols.map(symbol => ({
			name: symbol,
			channelClass: StockChannel
		}));
	}

	/**
	 * Hook: Configure channels after brain init - load CSV data and call setTraining
	 */
	async configureChannels() {
		const { timeframe, holdoutRows, offsetRows } = this.config;
		const dataDir = path.join(__dirname, '..', 'data', 'stock', timeframe);

		for (const symbol of this.config.symbols) {
			const csvPath = path.join(dataDir, `${symbol}.csv`);
			const allRows = this.loadCsvRows(csvPath);

			const startIndex = offsetRows;
			const endIndex = holdoutRows > 0 ? allRows.length - holdoutRows : allRows.length;
			const rows = allRows.slice(startIndex, endIndex);

			this.brain.getChannel(symbol).setTraining(rows);
		}
	}

	/**
	 * Load and parse a CSV file into {price, volume} row objects
	 * Rows are already in chronological order from processAndSaveSymbolData
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
	 * Run a single training episode through all historical data
	 */
	async runEpisode() {
		const startTime = Date.now();
		console.log(`📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes}... `);

		// Reset context but keep learned patterns
		this.brain.resetContext();

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

		// Calculate expected number of frames based on data rows
		const stockChannel = [...this.brain.getChannels()][0][1];
		const expectedFrames = stockChannel.trainingData.length - 1; // -1 because first frame reads 2 rows

		// Process all frames for the episode duration
		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Process frame
			await this.brain.processFrame();
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
	 * Collect profit/loss results from all channels
	 */
	collectEpisodeResults(episodeMetrics) {
		// Get portfolio-level metrics from brain (via thalamus)
		const allPortfolioMetrics = this.brain.getEpisodeSummary().portfolioMetrics;
		const portfolioMetrics = allPortfolioMetrics ? allPortfolioMetrics.StockChannel : null;

		if (!portfolioMetrics) {
			console.error('Warning: getPortfolioMetrics returned null');
			episodeMetrics.netProfit = 0;
			episodeMetrics.totalROI = 1;
			episodeMetrics.totalROIPercent = 0;
			return;
		}

		// Store portfolio profit
		episodeMetrics.netProfit = portfolioMetrics.totalProfit;

		// Calculate ROI metrics
		const finalValue = StockChannel.initialCapital + portfolioMetrics.totalProfit;
		const totalROI = finalValue / StockChannel.initialCapital;
		episodeMetrics.totalROI = totalROI;
		episodeMetrics.totalROIPercent = (totalROI - 1) * 100;

		// Calculate per-frame ROI (assuming compounding returns)
		if (episodeMetrics.frames > 0) {
			const perFrameROI = Math.pow(totalROI, 1 / episodeMetrics.frames) - 1;
			episodeMetrics.perFrameROI = perFrameROI;
			episodeMetrics.perFrameROIPercent = perFrameROI * 100;
		}

		// Collect per-channel results from channel metrics
		for (const [channelName, channel] of this.brain.getChannels()) {
			const metrics = channel.getMetrics();

			const channelResult = {
				symbol: channelName,
				investment: metrics.investment,
				currentValue: metrics.currentValue,
				unrealizedProfit: metrics.unrealizedProfit,
				trades: metrics.trades
			};

			episodeMetrics.channelResults.set(channelName, channelResult);
			episodeMetrics.totalTrades += channelResult.trades;
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
		console.log(`   Starting Capital: $${StockChannel.initialCapital.toFixed(2)}`);
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