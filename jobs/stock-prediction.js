import Job from './job.js';
import StockChannel from '../channels/stock.js';

/**
 * Stock Prediction Job - Tests the trained brain on holdout data
 * Uses the same symbols as training but only processes the holdout rows
 */
export default class StockPredictionJob extends Job {

	constructor() {
		super();
		this.hardReset = false; // Don't reset - use trained patterns
		
		// Configuration - should match training job
		this.config = {
			symbols: ['AAPL', 'GOOGL', 'MSFT'],  // Same symbols as training
			holdoutRows: 5                        // Same holdout size as training
		};
		
		// Prediction metrics
		this.predictionResults = {
			totalProfit: 0,
			totalLoss: 0,
			netProfit: 0,
			totalTrades: 0,
			profitableTrades: 0,
			channelResults: new Map()
		};
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
		console.log(`🔮 Starting Stock Prediction Job`);
		console.log(`📊 Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`📋 Holdout Rows: ${this.config.holdoutRows}`);
		console.log('');
	}

	/**
	 * Hook: Configure channels after brain initialization
	 */
	async configureChannels() {
		// Configure channels for prediction mode
		for (const [_, channel] of this.brain.channels) {
			channel.holdoutRows = this.config.holdoutRows;
			channel.setPredictionMode(); // Use only holdout rows
		}
	}

	/**
	 * Hook: Handle brain reset strategy
	 */
	async handleBrainReset() {
		// Reset context but keep learned patterns
		console.log('🧠 Resetting context (keeping learned patterns)...');
		await this.brain.resetContext();
	}

	/**
	 * Hook: Execute main job logic - prediction testing
	 */
	async executeJob() {
		await this.runPrediction();
	}

	/**
	 * Hook: Show results
	 */
	async showResults() {
		this.showPredictionResults();
	}

	/**
	 * Run prediction on holdout data
	 */
	async runPrediction() {
		const startTime = Date.now();
		console.log('🔮 Running prediction on holdout data...');

		// Process all frames until all channels are exhausted
		let frameCount = 0;
		while (true) {
			// Get combined frame from all channels
			const frame = await this.brain.getFrame();

			// If no input data from any channel, prediction is complete
			if (!frame || frame.length === 0) {
				break;
			}

			frameCount++;

			// Get feedback and process frame
			const feedback = await this.brain.getRewards();
			await this.brain.processFrame(frame, feedback);
		}

		// Collect prediction results from all channels
		this.collectPredictionResults();
		
		const duration = Date.now() - startTime;
		console.log(`✅ Prediction complete: ${frameCount} frames processed in ${duration}ms`);
	}

	/**
	 * Collect profit/loss results from all channels
	 */
	collectPredictionResults() {
		for (const [channelName, channel] of this.brain.channels) {
			const channelResult = {
				symbol: channelName,
				profit: channel.totalProfit || 0,
				loss: channel.totalLoss || 0,
				trades: channel.totalTrades || 0,
				profitableTrades: channel.profitableTrades || 0
			};
			
			channelResult.netProfit = channelResult.profit - channelResult.loss;
			
			this.predictionResults.channelResults.set(channelName, channelResult);
			this.predictionResults.totalProfit += channelResult.profit;
			this.predictionResults.totalLoss += channelResult.loss;
			this.predictionResults.totalTrades += channelResult.trades;
			this.predictionResults.profitableTrades += channelResult.profitableTrades;
		}
		
		this.predictionResults.netProfit = this.predictionResults.totalProfit - this.predictionResults.totalLoss;
	}

	/**
	 * Show prediction results
	 */
	showPredictionResults() {
		console.log(`\n🎯 Prediction Results (Holdout Data):`);
		console.log('='.repeat(50));
		
		console.log(`📈 Overall Performance:`);
		console.log(`   Net Profit: $${this.predictionResults.netProfit.toFixed(2)}`);
		console.log(`   Total Profit: $${this.predictionResults.totalProfit.toFixed(2)}`);
		console.log(`   Total Loss: $${this.predictionResults.totalLoss.toFixed(2)}`);
		console.log(`   Total Trades: ${this.predictionResults.totalTrades}`);
		console.log(`   Profitable Trades: ${this.predictionResults.profitableTrades}/${this.predictionResults.totalTrades}`);
		
		if (this.predictionResults.totalTrades > 0) {
			const winRate = (this.predictionResults.profitableTrades / this.predictionResults.totalTrades) * 100;
			console.log(`   Win Rate: ${winRate.toFixed(1)}%`);
		}
		
		console.log(`\n📊 Per-Symbol Results:`);
		for (const [symbol, result] of this.predictionResults.channelResults) {
			const winRate = result.trades > 0 ? (result.profitableTrades / result.trades) * 100 : 0;
			console.log(`   ${symbol}: $${result.netProfit.toFixed(2)} (${result.trades} trades, ${winRate.toFixed(1)}% win rate)`);
		}
		
		// Performance assessment
		console.log(`\n🏆 Assessment:`);
		if (this.predictionResults.netProfit > 0) {
			console.log(`   ✅ Profitable! The brain made money on unseen data.`);
		} else if (this.predictionResults.netProfit === 0) {
			console.log(`   ➖ Break-even. No profit or loss on unseen data.`);
		} else {
			console.log(`   ❌ Loss. The brain lost money on unseen data.`);
		}
		
		if (this.predictionResults.totalTrades === 0) {
			console.log(`   ⚠️  No trades executed. Brain may be too conservative or data insufficient.`);
		}
		
		console.log('='.repeat(50));
	}
}
