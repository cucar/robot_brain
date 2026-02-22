import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
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
			symbols: ['KGC', 'GLD', 'SPY'],        // Stock symbols to train on
			timeframe: '1D',                     // Timeframe for data (e.g., '1D', '1Min')
			maxEpisodes: 1,                      // Number of training episodes (can be overridden with --episodes)
			holdoutRows: 0,                     // Number of rows to hold out from end for prediction testing (can be overridden with --holdout)
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
	}

	/**
	 * Setup method - Downloads historical data from Alpha Vantage and processes it
	 * Run this with: node run-setup.js stock-test
	 */
	async setup() {
		const timeframe = this.config.timeframe;
		console.log(`📥 Downloading historical stock data (${timeframe})...`);
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log('');

		// Ensure timeframe-specific data directory exists
		const dataDir = path.join(__dirname, '..', 'data', 'stock', timeframe);
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
			const data = await this.downloadSymbolData(symbol, timeframe);
			symbolData.set(symbol, data);
		}

		console.log('');
		console.log('📊 Processing and aligning data...');

		// Find all timestamps that exist in ALL symbols (intersection)
		const commonDates = this.findCommonDates(symbolData);

		// Process and save each symbol's data using only common timestamps
		for (const symbol of this.config.symbols) {
			const data = symbolData.get(symbol);
			await this.processAndSaveSymbolData(symbol, data, commonDates, dataDir);
		}

		console.log('');
		console.log('✅ All data downloaded and processed successfully!');
	}

	/**
	 * Download historical data for a single symbol from Yahoo Finance Chart API
	 * @param {string} symbol - Stock symbol
	 * @param {string} timeframe - Timeframe (e.g. '1D', '1Min')
	 */
	async downloadSymbolData(symbol, timeframe) {
		const interval = this.getYahooInterval(timeframe);
		const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=0&period2=9999999999&interval=${interval}`;

		console.log(`📊 Downloading ${symbol} (${timeframe})...`);

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
							const key = this.getTimestampKey(timestamps[i], timeframe);
							const open = opens[i];
							const volume = volumes[i];
							if (open != null && volume != null)
								timeSeriesData[key] = { '1. open': String(open), '5. volume': String(volume) };
						}

						const count = Object.keys(timeSeriesData).length;
						console.log(`   ✅ ${symbol}: ${count} bars of data`);
						resolve(timeSeriesData);
					} catch (error) {
						reject(new Error(`Failed to parse data for ${symbol}: ${error.message}`));
					}
				});
			}).on('error', reject);
		});
	}

	/**
	 * Map our timeframe code to Yahoo Finance interval parameter
	 */
	getYahooInterval(timeframe) {
		const map = { '1D': '1d', '1Min': '1m', '5Min': '5m', '15Min': '15m', '1H': '1h' };
		return map[timeframe] || '1d';
	}

	/**
	 * Format a Unix timestamp as a key based on timeframe
	 * Daily → YYYY-MM-DD, intraday → YYYY-MM-DDTHH:MM
	 */
	getTimestampKey(timestamp, timeframe) {
		const iso = new Date(timestamp * 1000).toISOString();
		if (timeframe === '1D') return iso.split('T')[0];
		return iso.substring(0, 16);
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