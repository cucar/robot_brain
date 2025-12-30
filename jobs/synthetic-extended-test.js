import Job from './job.js';
import StockChannel from '../channels/stock.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Synthetic Extended Test - Replicates stock-training data pattern but runs continuously
 * Uses the actual first 12 rows from KGC.csv (the training data) repeated 25 times
 * This tests if episode boundaries (resetContext) are causing learning issues
 * 25 repeats × 11 frames = 275 frames (similar to 25 episodes worth)
 */
export default class SyntheticExtendedTest extends Job {

	constructor() {
		super();
		this.hardReset = true;

		this.config = {
			symbol: 'TEST',
			// Actual first 12 rows from KGC.csv (training data when holdoutRows=5240)
			sourceData: [
				{ price: 8.10, volume: 1447100 },
				{ price: 8.11, volume: 2112900 },
				{ price: 8.35, volume: 1411400 },
				{ price: 8.29, volume: 2091100 },
				{ price: 8.20, volume: 1247200 },
				{ price: 8.15, volume: 770000 },
				{ price: 8.19, volume: 1948400 },
				{ price: 8.05, volume: 2701100 },
				{ price: 7.94, volume: 1280800 },
				{ price: 7.86, volume: 3083500 },
				{ price: 7.22, volume: 2742800 },
				{ price: 7.51, volume: 1510600 }
			],
			cycleRepeats: 25 // 25 repeats × 11 frames = 275 frames
		};
	}

	async setup() {
		console.log('📊 Generating extended data from KGC training rows...');
		console.log(`   Source rows: ${this.config.sourceData.length}`);
		console.log(`   Repeats: ${this.config.cycleRepeats}`);
		console.log(`   Total Frames: ${this.config.cycleRepeats * (this.config.sourceData.length - 1)}`);
		console.log('');

		const dataDir = path.join(__dirname, '..', 'data', 'stock');
		if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

		const csvPath = path.join(dataDir, `${this.config.symbol}.csv`);
		const rows = [];

		// Repeat the source data pattern
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++)
			for (const row of this.config.sourceData)
				rows.push(`${row.price.toFixed(2)},${row.volume}`);

		// Add one final row to complete the last frame
		const lastRow = this.config.sourceData[0];
		rows.push(`${lastRow.price.toFixed(2)},${lastRow.volume}`);

		fs.writeFileSync(csvPath, rows.join('\n'));
		console.log(`✅ Generated ${rows.length} rows of data`);
		console.log(`   Saved to: ${csvPath}`);
	}

	async showStartupInfo() {
		const cycleLength = this.config.sourceData.length - 1; // 11 frames per cycle
		console.log(`🧪 Synthetic Extended Test (KGC training data pattern)`);
		console.log(`📊 Symbol: ${this.config.symbol}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * cycleLength}`);
		console.log('');
	}

	getChannels() {
		return [{ name: this.config.symbol, channelClass: StockChannel }];
	}

	async executeJob() {
		console.log('🚀 Running extended continuous episode...\n');

		const stockChannel = this.brain.channels.get(this.config.symbol);
		stockChannel.debug = false;
		stockChannel.debug2 = false;
		this.brain.debug = false;
		this.brain.debug2 = false;
		this.brain.resetAccuracyStats();

		const expectedFrames = stockChannel.dataRows.length - 1;
		const cycleLength = this.config.sourceData.length - 1; // 11 frames per cycle
		let frameCount = 0;

		// Print header
		console.log('Frame | CycleFrame | Price Change | Bucket | Action | Position | P&L');
		console.log('------|------------|--------------|--------|--------|----------|----');

		while (frameCount < expectedFrames) {
			const frame = await this.brain.getFrame();
			frameCount++;
			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;

			// Calculate price change and bucket
			const priceChange = stockChannel.previousPrice && stockChannel.currentPrice
				? ((stockChannel.currentPrice - stockChannel.previousPrice) / stockChannel.previousPrice * 100)
				: 0;
			const bucket = stockChannel.discretizePercentageChange ? stockChannel.discretizePercentageChange(priceChange) : 'N/A';

			// Get action name
			const action = stockChannel.lastAction !== null ? stockChannel.getActionName(stockChannel.lastAction) : 'NONE';
			const position = stockChannel.owned ? 'OWN' : 'OUT';
			const pnl = (stockChannel.totalProfit - stockChannel.totalLoss).toFixed(2);

			console.log(`${String(frameCount).padStart(5)} | ${String(cycleFrame).padStart(10)} | ${priceChange.toFixed(2).padStart(12)}% | ${String(bucket).padStart(6)} | ${action.padStart(12)} | ${position.padStart(8)} | $${pnl}`);

			await this.brain.processFrame(frame);

			// Stop after 10 cycles for analysis
			// if (frameCount >= cycleLength * 10) {
			// 	console.log('\n--- Stopping after 10 cycles for analysis ---');
			// 	break;
			// }
		}

		console.log(`\n✅ Completed ${frameCount} frames\n`);
	}

	showTestResults(trades, stockChannel) {
		console.log('='.repeat(60));
		console.log('📊 Extended Test Results');
		console.log('='.repeat(60));

		const baseAccuracy = this.brain.baseAccuracyStats || { correct: 0, total: 0 };
		console.log(`\n🎯 Prediction Accuracy:`);
		if (baseAccuracy.total > 0)
			console.log(`   Base Level: ${baseAccuracy.correct}/${baseAccuracy.total} = ${(baseAccuracy.correct / baseAccuracy.total * 100).toFixed(2)}%`);

		console.log(`\n💰 Trading Performance:`);
		console.log(`   Total Trades: ${trades.length}`);
		console.log(`   Net Profit: $${(stockChannel.totalProfit - stockChannel.totalLoss).toFixed(2)}`);
		console.log(`   Position: ${stockChannel.owned ? 'OWNED' : 'NOT OWNED'}`);

		// Show trade distribution by cycle frame
		const tradesByFrame = {};
		for (const trade of trades) {
			const key = `${trade.cycleFrame}-${trade.action}`;
			tradesByFrame[key] = (tradesByFrame[key] || 0) + 1;
		}
		console.log(`\n📋 Trade Distribution by Cycle Frame:`);
		for (const [key, count] of Object.entries(tradesByFrame).sort())
			console.log(`   ${key}: ${count} times`);
	}
}

