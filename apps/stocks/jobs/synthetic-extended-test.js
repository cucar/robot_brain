import { Job, runJob } from '#brain-node';
import { StockEncoder } from '../encoder.js';
import { StockTrader } from '../trader.js';

/**
 * Synthetic Extended Test - Replicates the stock-training data pattern but runs continuously.
 * Uses the actual first 12 rows from KGC.csv (the training data) repeated N times to test
 * if episode boundaries (resetContext) were causing learning issues.
 *
 * Spec-based path: single StockEncoder + StockTrader, no Channel subclass. Source rows
 * live in code (no CSV is written) so the test is fully self-contained and reproducible.
 */
export default class SyntheticExtendedTest extends Job {

	constructor() {
		super();

		this.config = {
			symbol: 'TEST',
			// Actual first 12 rows from KGC.csv (training data when holdoutRows=5240).
			// Frame 1 reads row 0→1, Frame 2 reads row 1→2, etc., Frame 12 reads row 11→0.
			// Comments show frames in EXECUTION order (Frame 12 executes last but reads row 0).
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
			cycleRepeats: 20 // 20 repeats × 12 frames = 240 frames
		};

		// Single-symbol arrays — kept as arrays to mirror the multi-symbol shape.
		this.encoders = [];
		this.traders = [];
	}

	getChannels() {
		return [];
	}

	async registerBrainChannels() {
		const encoder = new StockEncoder(this.config.symbol);
		const trader = new StockTrader(this.config.symbol);
		const channelId = this.brain.registerChannelSpec(encoder.getChannelSpec());
		encoder.bindChannelId(channelId);
		trader.bindChannelId(channelId);
		this.encoders.push(encoder);
		this.traders.push(trader);
	}

	/**
	 * Build the cycled rows in memory. One trailing row (= first row of next cycle) is
	 * appended so the last frame still has a "next" reading to compute change against —
	 * mirrors the legacy behavior where the CSV had N+1 rows for N frames.
	 */
	async configureChannels() {
		const rows = [];
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++)
			for (const row of this.config.sourceData)
				rows.push({ price: row.price, volume: row.volume });
		rows.push({ price: this.config.sourceData[0].price, volume: this.config.sourceData[0].volume });

		for (const encoder of this.encoders) encoder.setData(rows);
	}

	async showStartupInfo() {
		const cycleLength = this.config.sourceData.length - 1; // 11 frames per cycle
		console.log(`🧪 Synthetic Extended Test (KGC training data pattern)`);
		console.log(`📊 Symbol: ${this.config.symbol}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * cycleLength}`);
		console.log('');
	}

	/**
	 * Single continuous episode (no resetContext between cycles). Tracks per-cycle-frame
	 * optimality so the results table can show whether the brain learned to OWN before
	 * up-moves and stay OUT before down-moves.
	 */
	async executeJob() {
		console.log('🚀 Running extended continuous episode...\n');

		StockTrader.resetPortfolio();
		for (const trader of this.traders) trader.resetContext();
		this.brain.resetAccuracyStats();

		// Warmup: see synthetic-cycle-test for rationale.
		for (let i = 0; i < this.encoders.length; i++) {
			const frame = this.encoders[i].nextFrame();
			if (frame) this.traders[i].setFrame(frame.price, frame.volume);
		}

		const expectedFrames = this.encoders[0].rows.length - 1;
		const cycleLength = this.config.sourceData.length; // 12 frames per cycle (not 11!)

		// Pre-calculate optimal strategy per cycle frame: own if price will rise, else out.
		const optimalOwnership = this.calculateOptimalOwnership();
		console.log('Optimal ownership by cycle frame:', optimalOwnership);
		console.log('');

		const decisionStats = {};
		for (let i = 1; i <= cycleLength; i++)
			decisionStats[i] = { optimal: 0, suboptimal: 0, details: [] };

		console.log('Frame | CycleFrame | Price Change | Volume Change | Optimal | Actual | Match | P&L');
		console.log('------|------------|--------------|---------------|---------|--------|-------|----');

		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Capture ownership BEFORE running the frame (this is what we owned during the
			// price change being evaluated). After runFrame the trader's lastAction reflects
			// the NEW decision for the upcoming frame, not the one we want to score here.
			const ownedBeforeFrame = this.traders[0].lastAction === 1; // POSITION_OWN = 1

			const hasMore = await this.runFrame();
			if (!hasMore) break;
			frameCount++;

			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;
			const optimalOwned = optimalOwnership[cycleFrame];
			const isOptimal = ownedBeforeFrame === optimalOwned;

			if (isOptimal) decisionStats[cycleFrame].optimal++;
			else {
				decisionStats[cycleFrame].suboptimal++;
				decisionStats[cycleFrame].details.push({ frame: frameCount, actual: ownedBeforeFrame, optimal: optimalOwned });
			}

			if (this.isShuttingDown) return;
		}

		console.log(`\n✅ Completed ${frameCount} frames\n`);
		await this.showOptimalityAnalysis(decisionStats, cycleLength);
	}

	/**
	 * One frame: same shape as the main stocks/jobs/test.js loop.
	 */
	async runFrame() {
		const inputs = new Map();
		const rewards = new Map();

		let anyFrames = false;
		for (let i = 0; i < this.encoders.length; i++) {
			const encoder = this.encoders[i];
			const trader = this.traders[i];
			const frame = encoder.nextFrame();
			if (!frame) continue;
			anyFrames = true;
			trader.setFrame(frame.price, frame.volume);
			const dimMap = encoder.encode(frame);
			if (dimMap) inputs.set(encoder.channelId, dimMap);
		}
		if (!anyFrames) return false;

		for (const trader of this.traders)
			if (trader.lastAction !== null) rewards.set(trader.channelId, trader.getReward());

		const inferences = this.brain.processInputs(inputs, rewards);

		for (const trader of this.traders)
			trader.apply(inferences.get(trader.channelId) ?? []);

		await StockTrader.executePortfolio(this.traders);
		await new Promise(resolve => setImmediate(resolve));
		return true;
	}

	/**
	 * Optimal at frame N = own iff price will rise into frame N+1 (cyclic).
	 */
	calculateOptimalOwnership() {
		const ownership = {};
		const data = this.config.sourceData;

		for (let i = 0; i < data.length; i++) {
			const cycleFrame = i + 1;
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			ownership[cycleFrame] = (nextPrice - currentPrice) / currentPrice > 0;
		}

		return ownership;
	}

	/**
	 * Look up a neuron by (dim name, bucket value). Used to display which neurons represent
	 * which price/volume/activity buckets in the optimality table.
	 */
	async getNeuronIdForDimensionValue(dimensionName, value) {
		const dimId = this.brain.thalamus.dimensionNameToId[dimensionName];
		return this.brain.thalamus.getNeuronIdByCoordinate({ dimId, bucketId: value })?.id;
	}

	async showOptimalityAnalysis(decisionStats, cycleLength) {
		console.log('='.repeat(70));
		console.log('📊 Optimality Analysis by Cycle Frame');
		console.log('='.repeat(70));

		const data = this.config.sourceData;
		const encoder = this.encoders[0];
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
			const priceBucket = encoder.discretizeChange(priceChange, encoder.priceBoundaries);
			const priceNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_price_change`, priceBucket);

			const currentVolume = data[i - 1].volume;
			const nextVolume = data[i % data.length].volume;
			const volumeChange = ((nextVolume - currentVolume) / currentVolume * 100);
			const volumeBucket = encoder.discretizeChange(volumeChange, encoder.volumeBoundaries);
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

		// Actual portfolio P&L (cash + market value − initial capital).
		const netProfit = StockTrader.getPortfolioProfit(this.traders);
		console.log(`\n💰 Profit Analysis:`);
		console.log(`   Actual P&L: $${netProfit.toFixed(2)}`);

		console.log(`\n🎯 Action Neuron IDs:`);
		const ownNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_activity`, 1);
		const outNeuronId = await this.getNeuronIdForDimensionValue(`${this.config.symbol}_activity`, -1);
		console.log(`   OWN (activity=1):  Neuron ${ownNeuronId || 'N/A'}`);
		console.log(`   OUT (activity=-1): Neuron ${outNeuronId || 'N/A'}`);
	}
}

await runJob(import.meta, SyntheticExtendedTest);
