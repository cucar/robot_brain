import Job from './job.js';
import StockChannel from '../channels/stock.js';

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
			symbols: ['AAPL', 'GOOGL', 'MSFT'],  // Stock symbols to train on
			maxEpisodes: 50,                      // Number of training episodes
			holdoutRows: 5                        // Number of rows to hold out for prediction testing
		};

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
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
	 * Hook: Handle brain reset strategy
	 */
	async handleBrainReset() {
		// Hard reset only before first episode
		if (this.hardReset) {
			console.log('🧠 Hard reset: Clearing all brain tables...');
			await this.brain.resetBrain();
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
			channelResults: new Map()
		};

		// Process all frames until all channels are exhausted
		let frameCount = 0;
		while (true) {
			// Get combined frame from all channels
			const frame = await this.brain.getFrame();

			// If no input data from any channel, episode is complete
			if (!frame || frame.length === 0) {
				break;
			}

			frameCount++;

			// Get feedback and process frame
			const feedback = await this.brain.getFeedback();
			await this.brain.processFrame(frame, feedback);
		}

		// Collect episode results from all channels
		this.collectEpisodeResults(episodeMetrics);
		
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
			if (channel.resetEpisode) {
				channel.resetEpisode();
			} else {
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
				if (channel.rl) {
					channel.rl.close();
				}
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
