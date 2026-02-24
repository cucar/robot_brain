import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job } from './job.js';
import { StockChannel } from '../channels/stock.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Multi-Channel Test - Tests if brain can find optimal trading path for 12-day actual data with multiple stocks
 * Uses actual first 12 rows from each stock's CSV (like synthetic-extended-test) but with multiple stocks (like stock-training)
 * Runs continuously without episode resets to test learning across multiple channels
 */
export default class MultiChannelTest extends Job {

	/**
	 * Constructor - Initialize multi-channel test configuration
	 * Sets up symbols, cycle repeats, and source data storage
	 */
	constructor() {
		super();

		this.config = {
			symbols: ['KGC', 'GLD', 'SPY'],
			timeframe: '1D',                 // Timeframe for data (can be overridden with --timeframe)
			cycleRepeats: 20,
			sourceRows: 12 // First 12 rows from each stock's CSV
		};

		// Will be populated in setup() from actual CSV data
		this.sourceData = new Map(); // symbol -> array of {timestamp, price, volume}
	}

	/**
	 * Apply command line options to config
	 */
	applyOptions(options) {
		if (options.timeframe !== null && options.timeframe !== undefined) this.config.timeframe = options.timeframe;
	}

	/**
	 * Check if a timestamp falls within regular trading hours (9:30 AM - 4:00 PM ET)
	 * @param {string} timestamp - ISO timestamp string (e.g., "2026-01-22T14:30:00Z")
	 * @returns {boolean} True if within regular hours
	 */
	isRegularHours(timestamp) {
		const utcDate = new Date(timestamp);
		const etDate = new Date(utcDate.toLocaleString('en-US', { timeZone: 'America/New_York' }));
		const etHour = etDate.getHours();
		const etMinutes = etDate.getMinutes();
		const etTime = etHour * 60 + etMinutes;
		const regularOpen = 9 * 60 + 30;   // 9:30 AM ET
		const regularClose = 16 * 60;      // 4:00 PM ET
		return etTime >= regularOpen && etTime < regularClose;
	}

	/**
	 * Setup method - Load source data from JSON files and write to CSV files
	 * Reads from JSON files downloaded by stock-download
	 * Filters to regular trading hours only (9:30 AM - 4:00 PM ET)
	 * Writes CSV files in format: price,volume (no timestamps)
	 * @throws {Error} If JSON file not found for any symbol
	 */
	async setup() {
		console.log('📊 Loading first 12 rows from stock JSON files...');
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`   Timeframe: ${this.config.timeframe}`);
		console.log(`   Repeats: ${this.config.cycleRepeats}`);
		console.log('');

		const dataDir = path.join(__dirname, '..', 'data', 'stock', this.config.timeframe);

		for (const symbol of this.config.symbols) {
			const jsonPath = path.join(dataDir, `${symbol}.json`);
			if (!fs.existsSync(jsonPath)) {
				console.error(`❌ JSON file not found: ${jsonPath}`);
				console.error(`Please run: node stock-download.js --timeframe=${this.config.timeframe}`);
				process.exit(1);
			}

			const bars = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

			// Filter bars to regular trading hours only (9:30 AM - 4:00 PM ET)
			const regularHoursBars = this.config.timeframe === '1D'
				? bars
				: bars.filter(bar => this.isRegularHours(bar.Timestamp));

			// Take first sourceRows from regular hours data
			const rows = regularHoursBars.slice(0, this.config.sourceRows).map(bar => ({
				price: bar.OpenPrice,
				volume: bar.Volume
			}));

			this.sourceData.set(symbol, rows);

			// Write to CSV file: price,volume (no timestamps)
			const csvLines = rows.map(row => `${row.price},${row.volume}`);
			const csvPath = path.join(dataDir, `${symbol}_TEST.csv`);
			fs.writeFileSync(csvPath, csvLines.join('\n'));

			console.log(`   ✅ ${symbol}: Loaded ${rows.length} rows, wrote to ${symbol}_TEST.csv`);
		}

		console.log(`\n✅ Total Frames: ${this.config.cycleRepeats * (this.config.sourceRows - 1)}`);
	}

	/**
	 * Returns the channels for the job - one channel per stock symbol
	 * @returns {Array<Object>} Array of {name, channelClass} objects
	 */
	getChannels() {
		return this.config.symbols.map(symbol => ({
			name: symbol,
			channelClass: StockChannel
		}));
	}

	/**
	 * Hook: Configure channels after brain init - load CSV files and call setTraining
	 */
	async configureChannels() {
		const dataDir = path.join(__dirname, '..', 'data', 'stock', this.config.timeframe);

		for (const symbol of this.config.symbols) {
			const csvPath = path.join(dataDir, `${symbol}_TEST.csv`);
			const sourceRows = this.loadCsvRows(csvPath);

			// Store source data for optimal ownership calculation
			this.sourceData.set(symbol, sourceRows);

			// Generate cycled rows and set as training data
			this.brain.getChannel(symbol).setTraining(this.generateCycledRows(sourceRows));
		}
	}

	/**
	 * Load and parse a CSV file into {price, volume} row objects
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
	 * Generate cycled training rows from source rows
	 */
	generateCycledRows(sourceRows) {
		const rows = [];
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++)
			for (const row of sourceRows)
				rows.push({ price: row.price, volume: row.volume });
		rows.push({ price: sourceRows[0].price, volume: sourceRows[0].volume });
		return rows;
	}

	/**
	 * Hook: Show startup information
	 * Displays test configuration including symbols, cycles, and total frames
	 */
	async showStartupInfo() {
		const cycleLength = this.config.sourceRows - 1;
		console.log(`🧪 Multi-Channel Test (${this.config.symbols.length} stocks)`);
		console.log(`📊 Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * cycleLength}`);
		console.log('');
	}

	/**
	 * Hook: Execute main job logic - Run multi-channel test with optimality analysis
	 * Processes frames and tracks whether brain makes optimal buy/sell decisions
	 */
	async executeJob() {
		console.log('🚀 Running multi-channel continuous test...\n');

		this.brain.resetAccuracyStats();

		const firstChannel = [...this.brain.getChannels()][0][1];
		const expectedFrames = firstChannel.trainingData.length - 1;
		const cycleLength = this.config.sourceRows;

		// Pre-calculate optimal strategy per cycle frame for each symbol
		const optimalOwnership = new Map();
		for (const symbol of this.config.symbols)
			optimalOwnership.set(symbol, this.calculateOptimalOwnership(symbol));

		// Track decisions by cycle frame per channel
		const decisionStats = new Map();
		for (const symbol of this.config.symbols) {
			const stats = {};
			for (let i = 1; i <= cycleLength; i++)
				stats[i] = { optimal: 0, suboptimal: 0, details: [] };
			decisionStats.set(symbol, stats);
		}

		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Capture ownership BEFORE processing frame for all channels
			const ownedBeforeFrame = new Map();
			for (const [channelName, channel] of this.brain.getChannels())
				ownedBeforeFrame.set(channelName, channel.shares > 0);

			await this.brain.processFrame();
			frameCount++;
			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;

			// Track optimality for each channel
			for (const [channelName, _] of this.brain.getChannels()) {
				const actualOwned = ownedBeforeFrame.get(channelName);
				const optimalOwned = optimalOwnership.get(channelName)[cycleFrame];
				const stats = decisionStats.get(channelName);

				if (actualOwned === optimalOwned)
					stats[cycleFrame].optimal++;
				else {
					stats[cycleFrame].suboptimal++;
					stats[cycleFrame].details.push({ frame: frameCount, actual: actualOwned, optimal: optimalOwned });
				}
			}
		}

		console.log(`\n✅ Completed ${frameCount} frames\n`);
		await this.showOptimalityAnalysis(decisionStats, cycleLength);
	}

	/**
	 * Calculate optimal buy/sell decisions for a symbol based on price movements
	 * Determines whether to own the stock at each cycle frame based on next price
	 * @param {string} symbol - Stock symbol to analyze
	 * @returns {Object} Map of cycle frame number to boolean (true = own, false = out)
	 */
	calculateOptimalOwnership(symbol) {
		const ownership = {};
		const data = this.sourceData.get(symbol);

		for (let i = 0; i < data.length; i++) {
			const cycleFrame = i + 1;
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			ownership[cycleFrame] = nextPrice > currentPrice;
		}

		return ownership;
	}

	/**
	 * Display detailed optimality analysis showing how often brain made correct buy/sell decisions
	 * Compares actual decisions against optimal decisions for each cycle frame and symbol
	 * @param {Map} decisionStats - Map of channel name to decision statistics by cycle frame
	 * @param {number} cycleLength - Number of frames in each cycle
	 */
	async showOptimalityAnalysis(decisionStats, cycleLength) {
		console.log('='.repeat(80));
		console.log('📊 Optimality Analysis by Channel and Cycle Frame');
		console.log('='.repeat(80));

		let grandTotalOptimal = 0, grandTotalSuboptimal = 0;

		for (const symbol of this.config.symbols) {
			const stats = decisionStats.get(symbol);
			const data = this.sourceData.get(symbol);

			console.log(`\n📈 ${symbol}:`);
			console.log('CycleFrame | PriceChange | Optimal | OptimalRate | Suboptimal Frames');
			console.log('-----------|-------------|---------|-------------|------------------');

			let totalOptimal = 0, totalSuboptimal = 0;

			for (let i = 1; i <= cycleLength; i++) {
				const frameStats = stats[i];
				const total = frameStats.optimal + frameStats.suboptimal;
				const rate = total > 0 ? (frameStats.optimal / total * 100).toFixed(0) : 'N/A';

				const currentPrice = data[i - 1].price;
				const nextPrice = data[i % data.length].price;
				const priceChange = ((nextPrice - currentPrice) / currentPrice * 100);
				const optimal = nextPrice > currentPrice ? 'OWN' : 'OUT';

				totalOptimal += frameStats.optimal;
				totalSuboptimal += frameStats.suboptimal;

				const suboptimalFrames = frameStats.details.slice(0, 5).map(d => d.frame).join(', ');
				const moreCount = frameStats.details.length > 5 ? ` +${frameStats.details.length - 5} more` : '';

				console.log(`${String(i).padStart(10)} | ${priceChange.toFixed(2).padStart(10)}% | ${optimal.padStart(7)} | ${rate.padStart(10)}% | ${suboptimalFrames}${moreCount}`);
			}

			const overallRate = (totalOptimal / (totalOptimal + totalSuboptimal) * 100).toFixed(1);
			console.log(`   ${symbol} Optimal Rate: ${totalOptimal}/${totalOptimal + totalSuboptimal} = ${overallRate}%`);

			grandTotalOptimal += totalOptimal;
			grandTotalSuboptimal += totalSuboptimal;
		}

		const grandOverallRate = (grandTotalOptimal / (grandTotalOptimal + grandTotalSuboptimal) * 100).toFixed(1);
		console.log('\n' + '='.repeat(80));
		console.log(`🎯 Overall Optimal Rate: ${grandTotalOptimal}/${grandTotalOptimal + grandTotalSuboptimal} = ${grandOverallRate}%`);

		// Calculate total P&L using portfolio metrics
		const allPortfolioMetrics = this.brain.getEpisodeSummary().portfolioMetrics;
		const portfolioMetrics = allPortfolioMetrics ? allPortfolioMetrics.StockChannel : null;
		if (portfolioMetrics)
			console.log(`💰 Total P&L: $${portfolioMetrics.totalProfit.toFixed(2)}`);
		console.log('='.repeat(80));
	}
}