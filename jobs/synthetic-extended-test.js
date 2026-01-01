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
			cycleRepeats: 25 // 25 repeats × 12 frames = 300 frames
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

		this.brain.resetAccuracyStats();

		const expectedFrames = stockChannel.dataRows.length - 1;
		const cycleLength = this.config.sourceData.length; // 12 frames per cycle (not 11!)
		let frameCount = 0;

		// Pre-calculate optimal strategy per cycle frame
		// Optimal: own before positive frames, out before negative frames
		const optimalOwnership = this.calculateOptimalOwnership();
		console.log('Optimal ownership by cycle frame:', optimalOwnership);
		console.log('');

		// Track decisions by cycle frame
		const decisionStats = {};
		for (let i = 1; i <= cycleLength; i++)
			decisionStats[i] = { optimal: 0, suboptimal: 0, details: [] };

		// Print header
		console.log('Frame | CycleFrame | Price Change | Optimal | Actual | Match | P&L');
		console.log('------|------------|--------------|---------|--------|-------|----');

		while (frameCount < expectedFrames) {

			// this executes the inferred actions from previous frame in the current frame
			const frame = await this.brain.getFrame();

			frameCount++;
			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;

			// this.brain.waitForUserInput = cycleFrame === 2 || cycleFrame === 3;

			// Calculate price change
			const priceChange = stockChannel.previousPrice && stockChannel.currentPrice
				? ((stockChannel.currentPrice - stockChannel.previousPrice) / stockChannel.previousPrice * 100)
				: 0;

			// Get actual position (what we owned DURING this frame's price change)
			const actualOwned = stockChannel.owned;
			const optimalOwned = optimalOwnership[cycleFrame];
			const isOptimal = actualOwned === optimalOwned;

			// Track stats
			if (isOptimal)
				decisionStats[cycleFrame].optimal++;
			else {
				decisionStats[cycleFrame].suboptimal++;
				decisionStats[cycleFrame].details.push({ frame: frameCount, actual: actualOwned, optimal: optimalOwned });
			}

			const match = isOptimal ? '✓' : '✗';
			const realizedPL = stockChannel.totalProfit - stockChannel.totalLoss;

			console.log(`${String(frameCount).padStart(5)} | ${String(cycleFrame).padStart(10)} | ${priceChange.toFixed(2).padStart(12)}% | ${optimalOwned ? 'OWN' : 'OUT'.padStart(7)} | ${actualOwned ? 'OWN' : 'OUT'.padStart(6)} | ${match.padStart(5)} | $${realizedPL.toFixed(2)}`);

			await this.brain.processFrame(frame);
		}

		console.log(`\n✅ Completed ${frameCount} frames\n`);

		// Show analysis
		this.showOptimalityAnalysis(decisionStats, cycleLength, stockChannel);
	}

	/**
	 * Calculate optimal ownership for each cycle frame
	 * Optimal = own if price will go UP during this frame
	 */
	calculateOptimalOwnership() {
		const ownership = {};
		const data = this.config.sourceData;

		for (let i = 0; i < data.length; i++) {
			const cycleFrame = i + 1;
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			const priceChange = (nextPrice - currentPrice) / currentPrice;

			// Own if price goes up
			ownership[cycleFrame] = priceChange > 0;
		}

		return ownership;
	}

	/**
	 * Show detailed optimality analysis
	 */
	showOptimalityAnalysis(decisionStats, cycleLength, stockChannel) {
		console.log('='.repeat(70));
		console.log('📊 Optimality Analysis by Cycle Frame');
		console.log('='.repeat(70));

		const data = this.config.sourceData;
		let totalOptimal = 0, totalSuboptimal = 0;

		console.log('CycleFrame | PriceChange | Optimal | OptimalRate | Suboptimal Frames');
		console.log('-----------|-------------|---------|-------------|------------------');

		for (let i = 1; i <= cycleLength; i++) {
			const stats = decisionStats[i];
			const total = stats.optimal + stats.suboptimal;
			const rate = total > 0 ? (stats.optimal / total * 100).toFixed(0) : 'N/A';

			const currentPrice = data[i - 1].price;
			const nextPrice = data[i % data.length].price;
			const priceChange = ((nextPrice - currentPrice) / currentPrice * 100).toFixed(2);
			const optimal = nextPrice > currentPrice ? 'OWN' : 'OUT';

			totalOptimal += stats.optimal;
			totalSuboptimal += stats.suboptimal;

			const suboptimalFrames = stats.details.map(d => d.frame).join(', ');

			console.log(`${String(i).padStart(10)} | ${priceChange.padStart(11)}% | ${optimal.padStart(7)} | ${rate.padStart(10)}% | ${suboptimalFrames}`);
		}

		const overallRate = (totalOptimal / (totalOptimal + totalSuboptimal) * 100).toFixed(1);
		console.log('');
		console.log(`Overall Optimal Rate: ${totalOptimal}/${totalOptimal + totalSuboptimal} = ${overallRate}%`);

		// Calculate theoretical optimal profit
		let optimalProfit = 0;
		for (let i = 0; i < data.length; i++) {
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			if (nextPrice > currentPrice)
				optimalProfit += nextPrice - currentPrice;
		}
		const totalCycles = Math.floor((totalOptimal + totalSuboptimal) / cycleLength);
		const theoreticalOptimal = optimalProfit * totalCycles;

		console.log(`\n💰 Profit Analysis:`);
		console.log(`   Actual P&L: $${(stockChannel.totalProfit - stockChannel.totalLoss).toFixed(2)}`);
		console.log(`   Per-cycle optimal: $${optimalProfit.toFixed(2)}`);
		console.log(`   Theoretical optimal (${totalCycles} cycles): $${theoreticalOptimal.toFixed(2)}`);
		console.log(`   Efficiency: ${((stockChannel.totalProfit - stockChannel.totalLoss) / theoreticalOptimal * 100).toFixed(1)}%`);
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

