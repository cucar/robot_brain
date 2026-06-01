import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from 'robot-brain';
import { StockEncoder } from '../encoder.js';
import { StockTrader } from '../trader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Multi-Channel Test - Verifies the brain can find an optimal trading path across multiple
 * stocks using real (not synthetic) data. Loads the first 12 rows from each symbol's JSON,
 * filters to regular hours, repeats them N times, and runs continuously without resetContext.
 *
 * Spec-based path: one StockEncoder + StockTrader per symbol, no Channel subclass. Source
 * rows live in memory after a one-time JSON load — no per-symbol _TEST.csv is written.
 */
export default class MultiChannelTest extends Job {

	constructor() {
		super();

		this.config = {
			symbols: ['KGC', 'GLD', 'SPY'],
			timeframe: '1D',
			cycleRepeats: 20,
			sourceRows: 12 // First 12 rows from each stock's JSON
		};

		// One encoder + trader per symbol.
		this.encoders = [];
		this.traders = [];

		// symbol -> array of {price, volume} source rows. Used by optimality analysis to
		// know what the "right" decision was at each cycle position.
		this.sourceData = new Map();
	}

	applyOptions() {
		const timeframeIndex = process.argv.indexOf('--timeframe');
		if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) this.config.timeframe = process.argv[timeframeIndex + 1];
	}

	/**
	 * Create encoder + trader for each symbol and register the encoder spec with the brain.
	 * The trader borrows the same channelId so rewards/inputs/inferences key off a single
	 * number per symbol.
	 */
	async initialize() {
		for (const symbol of this.config.symbols) {
			const encoder = new StockEncoder(symbol);
			const trader = new StockTrader(symbol);
			const { channelId, dimensionIds } = this.brain.registerChannelSpec(encoder.getChannelSpec());
			encoder.bindIds({ channelId, dimensionIds });
			trader.bindChannelId(channelId);
			this.encoders.push(encoder);
			this.traders.push(trader);
		}
	}

	/**
	 * Load the first N rows from each symbol's JSON, filter to regular trading hours
	 * (intraday only), generate the cycled training rows, and hand them to the encoder.
	 * sourceData stays around for the optimality analysis at the end.
	 */
	async configureChannels() {
		console.log('📊 Loading first 12 rows from stock JSON files...');
		console.log(`   Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`   Timeframe: ${this.config.timeframe}`);
		console.log(`   Repeats: ${this.config.cycleRepeats}`);
		console.log('');

		const dataDir = path.join(__dirname, '..', 'data', this.config.timeframe);

		for (let i = 0; i < this.config.symbols.length; i++) {
			const symbol = this.config.symbols[i];
			const jsonPath = path.join(dataDir, `${symbol}.json`);
			if (!fs.existsSync(jsonPath)) {
				console.error(`❌ JSON file not found: ${jsonPath}`);
				console.error(`Please run: node apps/stocks/jobs/download.js --timeframe ${this.config.timeframe}`);
				process.exit(1);
			}

			const bars = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

			// Daily bars are session-aggregated already; intraday needs to be filtered down
			// to regular hours so the cycle pattern doesn't include pre/after-market bars.
			const regularHoursBars = this.config.timeframe === '1D'
				? bars
				: bars.filter(bar => this.isRegularHours(bar.Timestamp));

			const sourceRows = regularHoursBars.slice(0, this.config.sourceRows).map(bar => ({
				price: bar.OpenPrice,
				volume: bar.Volume
			}));

			this.sourceData.set(symbol, sourceRows);

			// Generate cycled rows and hand them to the encoder.
			this.encoders[i].setData(this.generateCycledRows(sourceRows));
			console.log(`   ✅ ${symbol}: Loaded ${sourceRows.length} source rows, cycled into ${this.config.cycleRepeats * sourceRows.length + 1} frames`);
		}

		console.log(`\n✅ Total Frames: ${this.config.cycleRepeats * (this.config.sourceRows - 1)}`);
	}

	/**
	 * Regular trading hours filter — 9:30am to 4:00pm ET. Uses toLocaleString so DST is
	 * handled by the runtime rather than hardcoded offsets.
	 */
	isRegularHours(timestamp) {
		const utcDate = new Date(timestamp);
		const etDate = new Date(utcDate.toLocaleString('en-US', { timeZone: 'America/New_York' }));
		const etTime = etDate.getHours() * 60 + etDate.getMinutes();
		const regularOpen = 9 * 60 + 30;
		const regularClose = 16 * 60;
		return etTime >= regularOpen && etTime < regularClose;
	}

	/**
	 * Repeat the source rows N times and append one trailing row (= first row again) so
	 * the last frame still has a "next" reading — same N+1 shape the legacy CSV used.
	 */
	generateCycledRows(sourceRows) {
		const rows = [];
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++)
			for (const row of sourceRows)
				rows.push({ price: row.price, volume: row.volume });
		rows.push({ price: sourceRows[0].price, volume: sourceRows[0].volume });
		return rows;
	}

	async showStartupInfo() {
		const cycleLength = this.config.sourceRows - 1;
		console.log(`🧪 Multi-Channel Test (${this.config.symbols.length} stocks)`);
		console.log(`📊 Symbols: ${this.config.symbols.join(', ')}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * cycleLength}`);
		console.log('');
	}

	/**
	 * Run a single continuous test: warmup, then loop processing frames. Track per-symbol
	 * per-cycle-frame optimality so the analysis at the end can show which symbols/positions
	 * the brain learned to handle correctly.
	 */
	async executeJob() {
		console.log('🚀 Running multi-channel continuous test...\n');

		StockTrader.resetPortfolio();
		for (const trader of this.traders) trader.resetContext();
		this.brain.resetAccuracyStats();

		// Warmup: see synthetic-cycle-test for rationale.
		for (let i = 0; i < this.encoders.length; i++) {
			const frame = this.encoders[i].nextFrame();
			if (frame) this.traders[i].setFrame(frame.price, frame.volume);
		}

		const expectedFrames = this.encoders[0].rows.length - 1;
		const cycleLength = this.config.sourceRows;

		// Pre-calculate optimal strategy per cycle frame for each symbol.
		const optimalOwnership = new Map();
		for (const symbol of this.config.symbols)
			optimalOwnership.set(symbol, this.calculateOptimalOwnership(symbol));

		// Track decisions by cycle frame per symbol.
		const decisionStats = new Map();
		for (const symbol of this.config.symbols) {
			const stats = {};
			for (let i = 1; i <= cycleLength; i++)
				stats[i] = { optimal: 0, suboptimal: 0, details: [] };
			decisionStats.set(symbol, stats);
		}

		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Capture each trader's current action BEFORE running the frame — that's what
			// they "owned during" the price change about to happen. After runFrame their
			// lastAction reflects the next decision, not the one we're scoring.
			const actionBeforeFrame = new Map();
			for (const trader of this.traders)
				actionBeforeFrame.set(trader.symbol, trader.lastAction);

			const hasMore = await this.runFrame();
			if (!hasMore) break;
			frameCount++;
			const cycleFrame = ((frameCount - 1) % cycleLength) + 1;

			// Score each symbol's pre-frame action against the optimal decision for this
			// cycle position.
			for (const trader of this.traders) {
				const actualOwned = actionBeforeFrame.get(trader.symbol) === 1; // POSITION_OWN = 1
				const optimalOwned = optimalOwnership.get(trader.symbol)[cycleFrame];
				const stats = decisionStats.get(trader.symbol);

				if (actualOwned === optimalOwned) stats[cycleFrame].optimal++;
				else {
					stats[cycleFrame].suboptimal++;
					stats[cycleFrame].details.push({ frame: frameCount, actual: actualOwned, optimal: optimalOwned });
				}
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

		// Pull one frame per encoder. Even frames that don't produce an encoded input
		// still update the trader's price/volume so valuation/reward math see the latest reading.
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

		// Only emit a reward for traders that actually acted last frame — otherwise we'd
		// credit/punish neurons that weren't responsible for the current state.
		for (const trader of this.traders)
			if (trader.lastAction !== null) rewards.set(trader.channelId, trader.getReward());

		// Brain returns inferences keyed by channelId plus per-frame diagnostic data;
		// `frame` flows straight to the renderer.
		const frameResult = this.brain.processFrame(inputs, rewards);
		for (const trader of this.traders)
			trader.apply(frameResult.inferences.get(trader.channelId) ?? []);

		// Coordinated portfolio execution: ranks OWN actions, sizes positions, runs
		// sells-then-buys so cash is freed before it's spent.
		await StockTrader.executePortfolio(this.traders);

		// Host-side rendering happens here (after portfolio execution so the tail
		// reflects the positions the traders just took).
		this.renderFrame(frameResult);

		// Step-debug pause between frames (no-op unless --wait is set).
		await this.waitForUser('Press Enter to continue to next frame');

		// Yield to the event loop so SIGINT can fire between frames.
		await new Promise(resolve => setImmediate(resolve));
		return true;
	}

	/**
	 * Append per-symbol holdings + portfolio Cash/P&L to the base summary line.
	 * The brain doesn't know about traders, so the tail lives here.
	 */
	getFrameSummaryTail() {
		return StockTrader.getSummaryTail(this.traders);
	}

	/**
	 * Vote-dump formatters: encoders keyed by channelId. Each
	 * StockEncoder provides formatActionLabel + formatCoordinates so the renderer
	 * can label action votes (OWN/OUT) and append bucket percent ranges.
	 */
	getChannelFormatters() {
		const map = new Map();
		for (const encoder of this.encoders) map.set(encoder.channelId, encoder);
		return map;
	}

	/**
	 * Optimal at cycle frame N for a symbol = own iff the next price in the cycle is higher.
	 */
	calculateOptimalOwnership(symbol) {
		const ownership = {};
		const data = this.sourceData.get(symbol);
		for (let i = 0; i < data.length; i++) {
			const cycleFrame = i + 1;
			const currentPrice = data[i].price;
			const nextPrice = data[(i + 1) % data.length].price;
			ownership[cycleFrame] = nextPrice > currentPrice;
		}
		return ownership;
	}

	async showOptimalityAnalysis(decisionStats, cycleLength) {
		console.log('='.repeat(80));
		console.log('📊 Optimality Analysis by Channel and Cycle Frame');
		console.log('='.repeat(80));

		let grandTotalOptimal = 0, grandTotalSuboptimal = 0;

		for (const symbol of this.config.symbols) {
			const stats = decisionStats.get(symbol);
			const data = this.sourceData.get(symbol);

			console.log(`\n📈 ${symbol}:`);
			console.log('CycleFrame | PriceChange | Optimal | OptimalRate | Suboptimal Frames');
			console.log('-----------|-------------|---------|-------------|------------------');

			let totalOptimal = 0, totalSuboptimal = 0;

			for (let i = 1; i <= cycleLength; i++) {
				const frameStats = stats[i];
				const total = frameStats.optimal + frameStats.suboptimal;
				const rate = total > 0 ? (frameStats.optimal / total * 100).toFixed(0) : 'N/A';

				const currentPrice = data[i - 1].price;
				const nextPrice = data[i % data.length].price;
				const priceChange = ((nextPrice - currentPrice) / currentPrice * 100);
				const optimal = nextPrice > currentPrice ? 'OWN' : 'OUT';

				totalOptimal += frameStats.optimal;
				totalSuboptimal += frameStats.suboptimal;

				const suboptimalFrames = frameStats.details.slice(0, 5).map(d => d.frame).join(', ');
				const moreCount = frameStats.details.length > 5 ? ` +${frameStats.details.length - 5} more` : '';

				console.log(`${String(i).padStart(10)} | ${priceChange.toFixed(2).padStart(10)}% | ${optimal.padStart(7)} | ${rate.padStart(10)}% | ${suboptimalFrames}${moreCount}`);
			}

			const overallRate = (totalOptimal / (totalOptimal + totalSuboptimal) * 100).toFixed(1);
			console.log(`   ${symbol} Optimal Rate: ${totalOptimal}/${totalOptimal + totalSuboptimal} = ${overallRate}%`);

			grandTotalOptimal += totalOptimal;
			grandTotalSuboptimal += totalSuboptimal;
		}

		const grandOverallRate = (grandTotalOptimal / (grandTotalOptimal + grandTotalSuboptimal) * 100).toFixed(1);
		console.log('\n' + '='.repeat(80));
		console.log(`🎯 Overall Optimal Rate: ${grandTotalOptimal}/${grandTotalOptimal + grandTotalSuboptimal} = ${grandOverallRate}%`);

		// Portfolio P&L (cash + market value − initial capital).
		const netProfit = StockTrader.getPortfolioProfit(this.traders);
		console.log(`💰 Total P&L: $${netProfit.toFixed(2)}`);
		console.log('='.repeat(80));
	}
}

await runJob(import.meta, MultiChannelTest);
