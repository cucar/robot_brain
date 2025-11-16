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
		this.hardReset = true; // Hard reset before first episode

		// Simple configuration - edit these values as needed
		this.config = {
			symbols: ['KGC', 'GLD', 'SPY'],       // Stock symbols to train on
			maxEpisodes: 20,                      // Number of training episodes
			holdoutRows: 5240,                       // Number of rows to hold out for prediction testing
			alphaVantageApiKey: '8DCVE4458VAJ8TUN' // Alpha Vantage API key
		};

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
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

		// Find oldest common date across all symbols
		const oldestCommonDate = this.findOldestCommonDate(symbolData);
		console.log(`   Oldest common date: ${oldestCommonDate}`);

		// Process and save each symbol's data
		for (const symbol of this.config.symbols) {
			const data = symbolData.get(symbol);
			await this.processAndSaveSymbolData(symbol, data, oldestCommonDate, dataDir);
		}

		console.log('');
		console.log('✅ All data downloaded and processed successfully!');
	}

	/**
	 * Download historical data for a single symbol as JSON
	 */
	async downloadSymbolData(symbol) {
		const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&outputsize=full&apikey=${this.config.alphaVantageApiKey}&datatype=json`;

		console.log(`📊 Downloading ${symbol}...`);

		return new Promise((resolve, reject) => {
			https.get(url, (response) => {
				if (response.statusCode !== 200) {
					reject(new Error(`Failed to download ${symbol}: HTTP ${response.statusCode}`));
					return;
				}

				let data = '';
				response.on('data', (chunk) => {
					data += chunk;
				});

				response.on('end', () => {
					try {
						const json = JSON.parse(data);

						// Check for API error
						if (json['Error Message']) {
							reject(new Error(`API error for ${symbol}: ${json['Error Message']}`));
							return;
						}

						if (json['Note']) {
							reject(new Error(`API limit reached: ${json['Note']}`));
							return;
						}

						if (!json['Time Series (Daily)']) {
							reject(new Error(`No time series data for ${symbol}`));
							return;
						}

						const timeSeriesData = json['Time Series (Daily)'];
						const dateCount = Object.keys(timeSeriesData).length;
						console.log(`   ✅ ${symbol}: ${dateCount} days of data`);

						resolve(timeSeriesData);
					} catch (error) {
						reject(new Error(`Failed to parse JSON for ${symbol}: ${error.message}`));
					}
				});
			}).on('error', (err) => {
				reject(err);
			});
		});
	}

	/**
	 * Find the oldest date that exists in all symbols' data
	 */
	findOldestCommonDate(symbolData) {
		// Get all dates for each symbol
		const symbolDates = new Map();
		for (const [symbol, data] of symbolData) {
			const dates = Object.keys(data).sort(); // Sort chronologically
			symbolDates.set(symbol, new Set(dates));
		}

		// Find the oldest date (earliest in chronological order)
		let oldestDate = null;
		for (const [_, dates] of symbolDates) {
			const symbolOldest = Array.from(dates).sort()[0];
			if (!oldestDate || symbolOldest > oldestDate) {
				oldestDate = symbolOldest;
			}
		}

		return oldestDate;
	}

	/**
	 * Process and save symbol data in the format expected by StockChannel
	 * Format: open,volume (no header, chronological order)
	 */
	async processAndSaveSymbolData(symbol, data, oldestCommonDate, dataDir) {
		// Get all dates and sort chronologically
		const dates = Object.keys(data).sort();

		// Filter to only dates >= oldestCommonDate
		const filteredDates = dates.filter(date => date >= oldestCommonDate);

		// Extract open price and volume for each date
		const rows = filteredDates.map(date => {
			const dayData = data[date];
			const open = dayData['1. open'];
			const volume = dayData['5. volume'];
			return `${open},${volume}`;
		});

		// Write to CSV file (no header, chronological order)
		const filePath = path.join(dataDir, `${symbol}.csv`);
		fs.writeFileSync(filePath, rows.join('\n'));

		console.log(`   ✅ ${symbol}.csv: ${rows.length} days (${filteredDates[0]} to ${filteredDates[filteredDates.length - 1]})`);
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
		for (const [_, channel] of this.brain.channels) {
			channel.holdoutRows = this.config.holdoutRows;
		}
	}

	/**
	 * Hook: Execute main job logic - multi-episode training
	 */
	async executeJob() {
		// Run training episodes
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {
			await this.runEpisode();

			// Show progress every 10 episodes or on last episode
			if (this.currentEpisode % 10 === 0 || this.currentEpisode === this.config.maxEpisodes) {
				this.showProgress();
			}
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
		process.stdout.write(`📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes}... `);

		// Reset context but keep learned patterns
		await this.brain.resetContext();

		// Reset accuracy stats for this episode
		this.brain.resetAccuracyStats();
		
		// Reset all channel states for new episode
		this.resetChannelStates();
		
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

		// Process all frames until all channels are exhausted
		let frameCount = 0;
		while (true) {
			// Get combined frame from all channels
			const frame = await this.brain.getFrame();

			// If no input data from any channel, episode is complete
			if (!frame || frame.length === 0)
				break;

			frameCount++;

			// Show progress every 100 frames
			if (frameCount % 100 === 0)
				process.stdout.write(`\r📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes} - Frame ${frameCount}/${5247}... `);

			// Get feedback and process frame
			const feedback = await this.brain.getFeedback();
			await this.brain.processFrame(frame, feedback);
		}

		// Clear progress line
		process.stdout.write(`\r`);
		process.stdout.clearLine(0);

		// Collect episode results from all channels
		this.collectEpisodeResults(episodeMetrics);

		// Capture base level accuracy stats
		const baseStats = this.brain.accuracyStats.get(0);
		if (baseStats && baseStats.total > 0)
			episodeMetrics.baseAccuracy = (baseStats.correct / baseStats.total * 100);

		// Capture overall accuracy across all levels
		let totalCorrect = 0;
		let totalPredictions = 0;
		for (const [_, stats] of this.brain.accuracyStats) {
			totalCorrect += stats.correct;
			totalPredictions += stats.total;
		}
		if (totalPredictions > 0)
			episodeMetrics.overallAccuracy = (totalCorrect / totalPredictions * 100);

		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		this.episodeResults.push(episodeMetrics);
		console.log(`✅ Net: $${episodeMetrics.netProfit.toFixed(2)} (${episodeMetrics.totalTrades} trades, ${duration}ms)`);
	}

	/**
	 * Reset all channel states for a new episode
	 */
	resetChannelStates() {
		for (const [_, channel] of this.brain.channels) {
			if (channel.resetEpisode)
				channel.resetEpisode();
			else {
				// Manual reset for stock channels
				channel.owned = false;
				channel.entryPrice = null;
				channel.holdingFrames = 0;
				channel.hasTraded = false;
				channel.previousPrice = null;
				channel.previousVolume = null;
				channel.currentPrice = null;
				channel.currentVolume = null;
				
				// Reset CSV iterator
				if (channel.rl)
					channel.rl.close();
				// Will be re-initialized on first getFrameInputs() call
				channel.rl = null;
				channel.lineIterator = null;
			}
		}
	}

	/**
	 * Collect profit/loss results from all channels
	 */
	collectEpisodeResults(episodeMetrics) {
		for (const [channelName, channel] of this.brain.channels) {
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

		// Show overall accuracy (all levels) per episode
		console.log(`\n📊 Overall Accuracy (All Levels) by Episode:`);
		for (const ep of this.episodeResults) {
			if (ep.overallAccuracy !== null)
				console.log(`   Episode ${ep.episode}: ${ep.overallAccuracy.toFixed(2)}%`);
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
