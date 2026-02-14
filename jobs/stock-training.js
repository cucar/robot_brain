import Job from './job.js';
import StockChannel from '../channels/stock.js';
import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Stock Training Job - Trains the brain on multiple stock symbols over many episodes
 * Each episode runs through the entire historical data, then resets context but keeps learned patterns
 */
export default class StockTrainingJob extends Job {

	constructor() {
		super();

		// Simple configuration - edit these values as needed
		this.config = {
			symbols: ['KGC', 'GLD', 'SPY'],        // Stock symbols to train on
			maxEpisodes: 1,                      // Number of training episodes (can be overridden with --episodes)
			holdoutRows: 50                     // Number of rows to hold out for prediction testing (can be overridden with --holdout)
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
	}

	/**
	 * Setup method - Downloads historical data from Alpha Vantage and processes it
	 * Run this with: node run-setup.js stock-training
	 */
	async setup() {
		console.log('📥 Downloading historical stock data from Alpha Vantage...');
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log('');

		// Ensure data directory exists
		const dataDir = path.join(__dirname, '..', 'data', 'stock');
		if (!fs.existsSync(dataDir)) {
			fs.mkdirSync(dataDir, { recursive: true });
			console.log(`✅ Created directory: ${dataDir}`);
		}

		// Remove old CSV files for symbols we're downloading
		for (const symbol of this.config.symbols) {
			const filePath = path.join(dataDir, `${symbol}.csv`);
			if (fs.existsSync(filePath)) {
				try {
					fs.unlinkSync(filePath);
				} catch (error) {
					console.log(`   ⚠️  Could not delete old ${symbol}.csv: ${error.message}`);
				}
			}
		}

		// Download JSON data for all symbols
		const symbolData = new Map();
		for (const symbol of this.config.symbols) {
			const data = await this.downloadSymbolData(symbol);
			symbolData.set(symbol, data);
		}

		console.log('');
		console.log('📊 Processing and aligning data...');

		// Find all dates that exist in ALL symbols (intersection)
		const commonDates = this.findCommonDates(symbolData);

		// Process and save each symbol's data using only common dates
		for (const symbol of this.config.symbols) {
			const data = symbolData.get(symbol);
			await this.processAndSaveSymbolData(symbol, data, commonDates, dataDir);
		}

		console.log('');
		console.log('✅ All data downloaded and processed successfully!');
	}

	/**
	 * Download historical data for a single symbol from Yahoo Finance Chart API
	 */
	async downloadSymbolData(symbol) {
		const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=0&period2=9999999999&interval=1d`;

		console.log(`📊 Downloading ${symbol}...`);

		return new Promise((resolve, reject) => {
			https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, response => {
				if (response.statusCode !== 200) {
					reject(new Error(`Failed to download ${symbol}: HTTP ${response.statusCode}`));
					return;
				}

				let data = '';
				response.on('data', chunk => data += chunk);

				response.on('end', () => {
					try {
						const json = JSON.parse(data);
						if (!json.chart || !json.chart.result || !json.chart.result[0]) {
							reject(new Error(`No chart data for ${symbol}`));
							return;
						}

						const result = json.chart.result[0];
						const timestamps = result.timestamp;
						const opens = result.indicators.quote[0].open;
						const volumes = result.indicators.quote[0].volume;

						const timeSeriesData = {};
						for (let i = 0; i < timestamps.length; i++) {
							const date = new Date(timestamps[i] * 1000).toISOString().split('T')[0];
							const open = opens[i];
							const volume = volumes[i];
							if (open != null && volume != null)
								timeSeriesData[date] = { '1. open': String(open), '5. volume': String(volume) };
						}

						const dateCount = Object.keys(timeSeriesData).length;
						console.log(`   ✅ ${symbol}: ${dateCount} days of data`);
						resolve(timeSeriesData);
					} catch (error) {
						reject(new Error(`Failed to parse data for ${symbol}: ${error.message}`));
					}
				});
			}).on('error', reject);
		});
	}

	/**
	 * Find all dates that exist in ALL symbols' data (intersection)
	 */
	findCommonDates(symbolData) {
		// Get all dates for each symbol as Sets
		const symbolDateSets = [];
		for (const [symbol, data] of symbolData) {
			const dates = new Set(Object.keys(data));
			symbolDateSets.push(dates);
			console.log(`   ${symbol}: ${dates.size} dates`);
		}

		// Find intersection of all date sets
		let commonDates = symbolDateSets[0];
		for (let i = 1; i < symbolDateSets.length; i++)
			commonDates = new Set([...commonDates].filter(date => symbolDateSets[i].has(date)));

		// Sort chronologically and return as array
		const sortedDates = [...commonDates].sort();
		console.log(`   Common dates: ${sortedDates.length} (${sortedDates[0]} to ${sortedDates[sortedDates.length - 1]})`);
		return sortedDates;
	}

	/**
	 * Process and save symbol data in the format expected by StockChannel
	 * Format: open,volume (no header, chronological order)
	 */
	async processAndSaveSymbolData(symbol, data, commonDates, dataDir) {
		// Extract open price and volume for each common date
		const rows = commonDates.map(date => {
			const dayData = data[date];
			const open = dayData['1. open'];
			const volume = dayData['5. volume'];
			return `${open},${volume}`;
		});

		// Write to CSV file (no header, chronological order)
		const filePath = path.join(dataDir, `${symbol}.csv`);
		fs.writeFileSync(filePath, rows.join('\n'));

		console.log(`   ✅ ${symbol}.csv: ${rows.length} days (${commonDates[0]} to ${commonDates[commonDates.length - 1]})`);
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
	 * Hook: Show startup information
	 */
	async showStartupInfo() {
		console.log(`🚀 Starting Stock Training Job`);
		console.log(`📊 Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`📋 Holdout Rows: ${this.config.holdoutRows}`);
		console.log('');
	}

	/**
	 * Hook: Configure channels after brain initialization
	 */
	async configureChannels() {
		// Set holdout rows for all channels
		for (const [_, channel] of this.brain.thalamus.getAllChannels()) {
			channel.holdoutRows = this.config.holdoutRows;
		}
	}

	/**
	 * Hook: Execute main job logic - multi-episode training
	 */
	async executeJob() {
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {

			// Run episode
			await this.runEpisode();

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
		await this.brain.resetContext();

		// Dump brain data at the beginning of each episode for debugging
		this.brain.createDump();

		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			totalProfit: 0,
			totalLoss: 0,
			netProfit: 0,
			totalTrades: 0,
			profitableTrades: 0,
			channelResults: new Map(),
			baseAccuracy: null,
			overallAccuracy: null
		};

		// Calculate expected number of frames based on data rows
		const stockChannel = [...this.brain.thalamus.getAllChannels()][0][1];
		const expectedFrames = stockChannel.dataRows.length - 1; // -1 because first frame reads 2 rows

		// Process all frames for the episode duration
		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Process frame
			await this.brain.processFrame();
			frameCount++;

			// Show progress every 100 frames
			if (frameCount % 100 === 0)
				process.stdout.write(`\r📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes} - Frame ${frameCount}/${expectedFrames}... `);
		}

		// Clear progress line (only if stdout is a TTY)
		if (!process.stdout.isTTY) process.stdout.write(`\n`);
		else {
			process.stdout.write(`\r`);
			process.stdout.clearLine(0);
		}

		// Collect episode results from all channels
		this.collectEpisodeResults(episodeMetrics);

		// Capture base level accuracy stats
		if (this.brain.diagnostics.accuracyStats.total > 0)
			episodeMetrics.baseAccuracy = (this.brain.diagnostics.accuracyStats.correct / this.brain.diagnostics.accuracyStats.total * 100);

		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		this.episodeResults.push(episodeMetrics);

		// Dump brain data at the beginning of each episode for debugging
		this.brain.createDump();

		console.log(`✅ Net: $${episodeMetrics.netProfit.toFixed(2)} (${episodeMetrics.totalTrades} trades, ${duration}ms)`);
	}

	/**
	 * Collect profit/loss results from all channels
	 */
	collectEpisodeResults(episodeMetrics) {
		for (const [channelName, channel] of this.brain.thalamus.getAllChannels()) {
			const channelResult = {
				symbol: channelName,
				profit: channel.totalProfit || 0,
				loss: channel.totalLoss || 0,
				trades: channel.totalTrades || 0,
				profitableTrades: channel.profitableTrades || 0
			};
			
			channelResult.netProfit = channelResult.profit - channelResult.loss;
			
			episodeMetrics.channelResults.set(channelName, channelResult);
			episodeMetrics.totalProfit += channelResult.profit;
			episodeMetrics.totalLoss += channelResult.loss;
			episodeMetrics.totalTrades += channelResult.trades;
			episodeMetrics.profitableTrades += channelResult.profitableTrades;
		}
		
		episodeMetrics.netProfit = episodeMetrics.totalProfit - episodeMetrics.totalLoss;
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

		console.log(`📈 Overall Performance:`);
		console.log(`   Total Net Profit: $${totalNetProfit.toFixed(2)}`);
		console.log(`   Average per Episode: $${avgNetProfit.toFixed(2)}`);
		console.log(`   Total Trades: ${totalTrades}`);
		console.log(`   Average Trades per Episode: ${avgTrades.toFixed(1)}`);

		// Show net profit per episode
		console.log(`\n💰 Net Profit by Episode:`);
		for (const ep of this.episodeResults)
			console.log(`   Episode ${ep.episode}: $${ep.netProfit.toFixed(2)} (${ep.totalTrades} trades)`);

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

			console.log(`\n📊 Learning Progress:`);
			console.log(`   First 10 episodes avg: $${firstAvg.toFixed(2)}`);
			console.log(`   Last 10 episodes avg: $${lastAvg.toFixed(2)}`);
			console.log(`   Improvement: $${improvement.toFixed(2)} (${improvement >= 0 ? '📈' : '📉'})`);
		}

		console.log('\n🏆 Best Episodes:');
		const sortedByProfit = [...this.episodeResults].sort((a, b) => b.netProfit - a.netProfit);
		for (let i = 0; i < Math.min(5, sortedByProfit.length); i++) {
			const ep = sortedByProfit[i];
			console.log(`   #${ep.episode}: $${ep.netProfit.toFixed(2)} (${ep.totalTrades} trades)`);
		}

		console.log('='.repeat(60));
	}
}