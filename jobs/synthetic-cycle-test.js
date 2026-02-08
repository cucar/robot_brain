import Job from './job.js';
import StockChannel from '../channels/stock.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Synthetic Cycle Test - Tests if brain can learn a perfectly repeating price pattern
 */
export default class SyntheticCycleTest extends Job {

	constructor() {
		super();

		this.config = {
			symbol: 'TEST',
			// up (p7), down (p8), down, up, up (p5), down (p6)
			cyclePattern: [0.009, -0.019, -0.029, 0.019, 0.029, -0.009], // +1%, -2%, -3%, +2%, +3%, -1%
			cycleRepeats: 50,
			startPrice: 100.00,
			startVolume: 100000
		};
	}

	/**
	 * Setup method - Generate synthetic CSV data
	 */
	async setup() {
		console.log('📊 Generating synthetic cycle data...');
		console.log(`   Pattern: ${this.config.cyclePattern.map(p => (p * 100).toFixed(0) + '%').join(', ')}`);
		console.log(`   Repeats: ${this.config.cycleRepeats}`);
		console.log('');

		// Ensure data directory exists
		const dataDir = path.join(__dirname, '..', 'data', 'stock');
		if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

		// Generate CSV data (format: open,volume - no header)
		const csvPath = path.join(dataDir, `${this.config.symbol}.csv`);
		const rows = [];

		let currentPrice = this.config.startPrice;
		let currentVolume = this.config.startVolume;

		// Generate data for each cycle
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++) {
			for (let i = 0; i < this.config.cyclePattern.length; i++) {
				const priceChange = this.config.cyclePattern[i];

				// Calculate new price and volume
				const newPrice = currentPrice * (1 + priceChange);
				const newVolume = currentVolume * (1 + priceChange); // change volume with price
				// const newVolume = currentVolume; // volume stays constant

				// Add row: open,volume
				rows.push(`${currentPrice.toFixed(2)},${Math.round(currentVolume)}`);

				// Update for next iteration
				currentPrice = newPrice;
				currentVolume = newVolume;
			}
		}

		// Add one final row to complete the last frame
		// (StockChannel needs N+1 rows to produce N frames because first frame consumes 2 rows)
		rows.push(`${currentPrice.toFixed(2)},${Math.round(currentVolume)}`);

		// Write CSV file
		fs.writeFileSync(csvPath, rows.join('\n'));
		console.log(`✅ Generated ${rows.length} rows of data`);
		console.log(`   Saved to: ${csvPath}`);
		console.log(`   Starting price: $${this.config.startPrice.toFixed(2)}`);
		console.log(`   Ending price: $${currentPrice.toFixed(2)}`);
		console.log(`   Total change: ${((currentPrice / this.config.startPrice - 1) * 100).toFixed(2)}%`);
	}

	/**
	 * Hook: Show startup information
	 */
	async showStartupInfo() {
		console.log(`🧪 Synthetic Cycle Test`);
		console.log(`📊 Symbol: ${this.config.symbol}`);
		console.log(`🔄 Pattern: ${this.config.cyclePattern.map(p => (p * 100).toFixed(0) + '%').join(' → ')}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * this.config.cyclePattern.length}`);
		console.log('');
	}

	/**
	 * Hook: Define channels
	 */
	getChannels() {
		return [
			{ name: this.config.symbol, channelClass: StockChannel }
		];
	}

	/**
	 * Hook: Execute main job logic
	 */
	async executeJob() {
		console.log('🚀 Running single episode...\n');

		// Get the stock channel
		const stockChannel = this.brain.getChannel(this.config.symbol);

		// Reset accuracy stats
		this.brain.resetAccuracyStats();

		// expected frames count is one less than the number of rows - first row is skipped to be able to start detecting changes
		const expectedFrames = stockChannel.dataRows.length - 2;

		// Track trades
		let trades = [];
		let frameCount = 0;

		// Process all frames
		while (frameCount < expectedFrames) {

			// Show cycle-specific debug info before processing frame
			if (this.brain.debug) {
				const cycleFrame = (frameCount % this.config.cyclePattern.length) + 1;
				const expectedChange = this.config.cyclePattern[cycleFrame - 1];
				console.log(`\n📍 Cycle Position: Frame ${cycleFrame}/6 (expecting ${(expectedChange * 100).toFixed(1)}% change)`);
			}

			// Process frame
			await this.brain.processFrame();
			frameCount++;

			// Track trades with source information
			if (stockChannel.lastAction !== null && stockChannel.lastAction !== 0) {
				const action = stockChannel.getActionName(stockChannel.lastAction);
				const priceChange = ((stockChannel.currentPrice - stockChannel.previousPrice) / stockChannel.previousPrice);
				const cycleFrame = (frameCount % this.config.cyclePattern.length) + 1;
				trades.push({
					frame: frameCount,
					cycleFrame: cycleFrame,
					action: action,
					priceChange: (priceChange * 100).toFixed(2) + '%'
				});
			}

			// Show progress every 25 frames
			if (frameCount % 25 === 0) console.log(`\rFrame ${frameCount}/${expectedFrames}... `);
		}

		console.log(`\r✅ Completed ${frameCount} frames\n`);

		// Show results
		this.showTestResults(trades, stockChannel);
	}

	/**
	 * Show test results
	 */
	showTestResults(trades, stockChannel) {
		console.log('='.repeat(60));
		console.log('📊 Test Results');
		console.log('='.repeat(60));

		// Accuracy
		const accuracy = this.brain.accuracyStats || { correct: 0, total: 0 };

		console.log(`\n🎯 Prediction Accuracy:`);
		if (accuracy.total > 0)
			console.log(`   ${accuracy.correct}/${accuracy.total} = ${(accuracy.correct / accuracy.total * 100).toFixed(2)}%`);

		// Trading performance
		console.log(`\n💰 Trading Performance:`);
		console.log(`   Total Trades: ${trades.length}`);
		console.log(`   Net Profit: $${(stockChannel.netProfit || 0).toFixed(2)}`);
		console.log(`   Position: ${stockChannel.position ? 'OWNED' : 'NOT OWNED'}`);

		// Show all trades
		console.log(`\n📋 Trade History:`);
		if (trades.length === 0) console.log('   No trades executed');
		else for (const trade of trades) console.log(`   Frame ${trade.frame} (Cycle ${trade.cycleFrame}): ${trade.action} at ${trade.priceChange}`);
	}
}

