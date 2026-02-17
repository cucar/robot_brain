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

		this.config = {
			symbol: 'TEST',
			// Actual first 12 rows from KGC.csv (training data when holdoutRows=5240)
			// Frame 1 reads row 0→1, Frame 2 reads row 1→2, etc., Frame 12 reads row 11→0
			// Comments show frames in EXECUTION order (Frame 12 executes last but reads row 0)
			// Format: Frame N: price (neuron), vol (neuron), optimal action
			sourceData: [

				// detailed bucketing
				// { price: 8.10, volume: 1447100 },  // Frame 12: +7.86% (n17), -4.20% (n18), OWN
				// { price: 8.11, volume: 2112900 },  // Frame 1:  +0.12% (n1),  +46.01% (n2),  OWN
				// { price: 8.35, volume: 1411400 },  // Frame 2:  +2.96% (n3),  -33.20% (n4),  OWN
				// { price: 8.29, volume: 2091100 },  // Frame 3:  -0.72% (n5),  +48.16% (n2),  OUT
				// { price: 8.20, volume: 1247200 },  // Frame 4:  -1.09% (n7),  -40.36% (n8),  OUT
				// { price: 8.15, volume: 770000 },   // Frame 5:  -0.61% (n5),  -38.26% (n4),  OUT
				// { price: 8.19, volume: 1948400 },  // Frame 6:  +0.49% (n10), +153.04% (n11), OWN
				// { price: 8.05, volume: 2701100 },  // Frame 7:  -1.71% (n7),  +38.63% (n12), OUT
				// { price: 7.94, volume: 1280800 },  // Frame 8:  -1.37% (n7),  -52.58% (n13), OUT
				// { price: 7.86, volume: 3083500 },  // Frame 9:  -1.01% (n7),  +140.75% (n11), OUT
				// { price: 7.22, volume: 2742800 },  // Frame 10: -8.14% (n14), -11.05% (n15), OUT
				// { price: 7.51, volume: 1510600 }   // Frame 11: +4.02% (n16), -44.92% (n8),  OWN

				// binary buckets (up or down)
				// Neuron IDs: n1=price+1, n2=vol+1, n3=vol-1, n4=OUT, n5=price-1, n6=OWN
				{ price: 8.10, volume: 1447100 },  // Frame 12: price=1 (n1), vol=-1 (n3), OWN (n6)
				{ price: 8.11, volume: 2112900 },  // Frame 1:  price=1 (n1), vol=1  (n2), OWN (n6)
				{ price: 8.35, volume: 1411400 },  // Frame 2:  price=1 (n1), vol=-1 (n3), OWN (n6)
				{ price: 8.29, volume: 2091100 },  // Frame 3:  price=-1(n5), vol=1  (n2), OUT (n4)
				{ price: 8.20, volume: 1247200 },  // Frame 4:  price=-1(n5), vol=-1 (n3), OUT (n4)
				{ price: 8.15, volume: 770000 },   // Frame 5:  price=-1(n5), vol=-1 (n3), OUT (n4)
				{ price: 8.19, volume: 1948400 },  // Frame 6:  price=1 (n1), vol=1  (n2), OWN (n6)
				{ price: 8.05, volume: 2701100 },  // Frame 7:  price=-1(n5), vol=1  (n2), OUT (n4)
				{ price: 7.94, volume: 1280800 },  // Frame 8:  price=-1(n5), vol=-1 (n3), OUT (n4)
				{ price: 7.86, volume: 3083500 },  // Frame 9:  price=-1(n5), vol=1  (n2), OUT (n4)
				{ price: 7.22, volume: 2742800 },  // Frame 10: price=-1(n5), vol=-1 (n3), OUT (n4)
				{ price: 7.51, volume: 1510600 }   // Frame 11: price=1 (n1), vol=-1 (n3), OWN (n6)
			],
			cycleRepeats: 20 // 5 repeats × 12 frames = 60 frames (reduced for debugging)
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

		const stockChannel = this.brain.getChannel(this.config.symbol);

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
		console.log('Frame | CycleFrame | Price Change | Volume Change | Optimal | Actual | Match | P&L');
		console.log('------|------------|--------------|---------------|---------|--------|-------|----');

		while (frameCount < expectedFrames) {

			// Capture ownership BEFORE processing frame (this is what we owned during the price change)
			const ownedBeforeFrame = stockChannel.owned;

			// Process frame (includes getFrame, executeActions, getRewards, and all brain processing)
			await this.brain.processFrame();

			frameCount++;
			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;

			// Get actual position (what we owned DURING this frame's price change)
			const actualOwned = ownedBeforeFrame;
			const optimalOwned = optimalOwnership[cycleFrame];
			const isOptimal = actualOwned === optimalOwned;

			// Track stats
			if (isOptimal)
				decisionStats[cycleFrame].optimal++;
			else {
				decisionStats[cycleFrame].suboptimal++;
				decisionStats[cycleFrame].details.push({ frame: frameCount, actual: actualOwned, optimal: optimalOwned });
			}
		}

		console.log(`\n✅ Completed ${frameCount} frames\n`);

		// Show analysis
		await this.showOptimalityAnalysis(decisionStats, cycleLength, stockChannel);
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
	 * Get neuron ID for a specific dimension and value
	 */
	async getNeuronIdForDimensionValue(dimensionName, value) {
		return this.brain.getNeuronByCoordinates({ [dimensionName]: value }).id;
	}

	/**
	 * Show detailed optimality analysis
	 */
	async showOptimalityAnalysis(decisionStats, cycleLength, stockChannel) {
		console.log('='.repeat(70));
		console.log('📊 Optimality Analysis by Cycle Frame');
		console.log('='.repeat(70));

		const data = this.config.sourceData;
		let totalOptimal = 0, totalSuboptimal = 0;

		console.log('CycleFrame | PriceChange  | PriceNeuron | VolumeChange  | VolumeNeuron | Optimal | OptimalRate | Suboptimal Frames');
		console.log('-----------|--------------|-------------|---------------|--------------|---------|-------------|------------------');

		for (let i = 1; i <= cycleLength; i++) {
			const stats = decisionStats[i];
			const total = stats.optimal + stats.suboptimal;
			const rate = total > 0 ? (stats.optimal / total * 100).toFixed(0) : 'N/A';

			const currentPrice = data[i - 1].price;
			const nextPrice = data[i % data.length].price;
			const priceChange = ((nextPrice - currentPrice) / currentPrice * 100);
			const priceBucket = stockChannel.discretizeChange(priceChange);
			const priceNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_price_change`, priceBucket);

			const currentVolume = data[i - 1].volume;
			const nextVolume = data[i % data.length].volume;
			const volumeChange = ((nextVolume - currentVolume) / currentVolume * 100);
			const volumeBucket = stockChannel.discretizeChange(volumeChange);
			const volumeNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_volume_change`, volumeBucket);

			const optimal = nextPrice > currentPrice ? 'OWN' : 'OUT';

			totalOptimal += stats.optimal;
			totalSuboptimal += stats.suboptimal;

			const suboptimalFrames = stats.details.map(d => d.frame).join(', ');

			const priceNeuronStr = priceNeuronId ? String(priceNeuronId) : 'N/A';
			const volumeNeuronStr = volumeNeuronId ? String(volumeNeuronId) : 'N/A';

			console.log(`${String(i).padStart(10)} | ${priceChange.toFixed(2).padStart(11)}% | ${priceNeuronStr.padStart(11)} | ${volumeChange.toFixed(2).padStart(12)}% | ${volumeNeuronStr.padStart(12)} | ${optimal.padStart(7)} | ${rate.padStart(10)}% | ${suboptimalFrames}`);
		}

		const overallRate = (totalOptimal / (totalOptimal + totalSuboptimal) * 100).toFixed(1);
		console.log('');
		console.log(`Overall Optimal Rate: ${totalOptimal}/${totalOptimal + totalSuboptimal} = ${overallRate}%`);

		// Calculate theoretical optimal profit based on percentage returns
		let capitalMultiplier = 1.0;
		for (let i = 0; i < data.length; i++) {
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			const priceChange = (nextPrice - currentPrice) / currentPrice;
			if (priceChange > 0) capitalMultiplier *= (1 + priceChange);
		}

		// Get portfolio metrics for actual P&L
		const allPortfolioMetrics = this.brain.getEpisodeSummary().portfolioMetrics;
		console.log(`\n💰 Profit Analysis:`);
		console.log(`   Actual P&L: $${allPortfolioMetrics.StockChannel.totalProfit.toFixed(2)}`);

		// Show action neuron IDs
		console.log(`\n🎯 Action Neuron IDs:`);
		const ownNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_activity`, 1);
		const outNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_activity`, -1);
		console.log(`   OWN (activity=1):  Neuron ${ownNeuronId || 'N/A'}`);
		console.log(`   OUT (activity=-1): Neuron ${outNeuronId || 'N/A'}`);
	}
}

