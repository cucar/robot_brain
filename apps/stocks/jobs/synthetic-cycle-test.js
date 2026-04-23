import { Job, runJob } from '#brain-node';
import { StockEncoder } from '../encoder.js';
import { StockTrader } from '../trader.js';

/**
 * Synthetic Cycle Test - Tests if brain can learn a perfectly repeating price pattern.
 * Uses the spec-based registration path: a single StockEncoder + StockTrader for the
 * synthetic TEST symbol, no Channel subclass. The cycle pattern is generated in memory
 * and handed to the encoder via setData() — no CSV is written.
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

		// Single-symbol arrays — kept as arrays to mirror the multi-symbol shape used by
		// the other stocks jobs, so the runFrame loop is structurally identical.
		this.encoders = [];
		this.traders = [];
	}

	/**
	 * Opt out of the legacy Channel-class path; this job owns its encoder/trader directly.
	 */
	getChannels() {
		return [];
	}

	/**
	 * Create encoder + trader and register the encoder spec with the brain. The trader
	 * borrows the same channelId so rewards/inputs/inferences key off a single number.
	 */
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
	 * Generate the cycled rows in memory and hand them to the encoder. One trailing row
	 * is appended so the last cycle frame still has a "next" reading to compute change against.
	 */
	async configureChannels() {
		const rows = [];
		let currentPrice = this.config.startPrice;
		let currentVolume = this.config.startVolume;
		for (let cycle = 0; cycle < this.config.cycleRepeats; cycle++)
			for (const priceChange of this.config.cyclePattern) {
				rows.push({ price: currentPrice, volume: Math.round(currentVolume) });
				currentPrice = currentPrice * (1 + priceChange);
				currentVolume = currentVolume * (1 + priceChange);
			}
		rows.push({ price: currentPrice, volume: Math.round(currentVolume) });

		for (const encoder of this.encoders) encoder.setData(rows);
	}

	async showStartupInfo() {
		console.log(`🧪 Synthetic Cycle Test`);
		console.log(`📊 Symbol: ${this.config.symbol}`);
		console.log(`🔄 Pattern: ${this.config.cyclePattern.map(p => (p * 100).toFixed(0) + '%').join(' → ')}`);
		console.log(`🔁 Cycles: ${this.config.cycleRepeats}`);
		console.log(`📋 Total Frames: ${this.config.cycleRepeats * this.config.cyclePattern.length}`);
		console.log('');
	}

	/**
	 * Run a single continuous episode (no resetContext between cycles — the whole point
	 * of this test is to verify learning across repeated patterns without episode boundaries).
	 */
	async executeJob() {
		console.log('🚀 Running single episode...\n');

		StockTrader.resetPortfolio();
		for (const trader of this.traders) trader.resetContext();
		this.brain.resetAccuracyStats();

		// Warmup: consume the first frame so the first processed frame has a full
		// (previous, current) pair. Without this the first encoded frame is skipped
		// (previousPrice=null) and the trader runs one frame behind the encoder.
		for (let i = 0; i < this.encoders.length; i++) {
			const frame = this.encoders[i].nextFrame();
			if (frame) this.traders[i].setFrame(frame.price, frame.volume);
		}

		// One frame consumed by warmup, so main loop processes one fewer.
		const expectedFrames = this.encoders[0].rows.length - 1;
		const actions = [];
		let frameCount = 0;

		while (frameCount < expectedFrames) {

			if (this.brain.debug) {
				const cycleFrame = (frameCount % this.config.cyclePattern.length) + 1;
				const expectedChange = this.config.cyclePattern[cycleFrame - 1];
				console.log(`\n📍 Cycle Position: Frame ${cycleFrame}/${this.config.cyclePattern.length} (expecting ${(expectedChange * 100).toFixed(1)}% change)`);
			}

			const hasMore = await this.runFrame();
			if (!hasMore) break;
			frameCount++;

			// Record the action the trader just took (if any). Captures the cycle-frame
			// position so the results table can group actions by where they fell in the cycle.
			const trader = this.traders[0];
			if (trader.lastAction !== null) {
				const actionName = trader.lastAction === 1 ? 'OWN' : 'OUT';
				const priceChange = trader.previousPrice
					? ((trader.currentPrice - trader.previousPrice) / trader.previousPrice * 100).toFixed(2) + '%'
					: 'N/A';
				const cycleFrame = (frameCount % this.config.cyclePattern.length) + 1;
				actions.push({ frame: frameCount, cycleFrame, action: actionName, priceChange });
			}

			if (frameCount % 25 === 0) console.log(`\rFrame ${frameCount}/${expectedFrames}...`);
			if (this.isShuttingDown) return;
		}

		console.log(`\r✅ Completed ${frameCount} frames\n`);
		this.showTestResults(actions);
	}

	/**
	 * One-frame pipeline: pull next frame per encoder, build inputs/rewards keyed by
	 * channelId, ship to the brain, apply inferences back to traders, execute the
	 * portfolio. Identical shape to the main stocks/jobs/test.js loop.
	 */
	async runFrame() {
		const inputs = new Map();
		const rewards = new Map();

		// Pull next frame per encoder. Trader still gets price/volume even when the encoder
		// produces no input, so reward/valuation math always sees the latest reading.
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

		// Only report reward for traders that actually acted last frame — otherwise we'd
		// credit/punish neurons that didn't drive the current state.
		for (const trader of this.traders)
			if (trader.lastAction !== null) rewards.set(trader.channelId, trader.getReward());

		// Brain returns inferences keyed by channelId plus per-frame diagnostic data;
		// `frame` flows straight to the renderer.
		const { inferences, frame } = this.brain.processInputs(inputs, rewards);
		for (const trader of this.traders)
			trader.apply(inferences.get(trader.channelId) ?? []);

		// Coordinated portfolio execution: ranks OWN actions, sizes positions, runs
		// sells-then-buys so cash is freed before it's spent.
		await StockTrader.executePortfolio(this.traders);

		// Host-side rendering happens here (after portfolio execution so the tail
		// reflects the positions the traders just took).
		this.renderFrame(frame);

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
	 * Vote-dump formatters: encoders keyed by channel name (== symbol). Each
	 * StockEncoder provides formatActionLabel + formatCoordinates.
	 */
	getChannelFormatters() {
		const map = new Map();
		for (const encoder of this.encoders) map.set(encoder.symbol, encoder);
		return map;
	}

	showTestResults(actions) {
		console.log('='.repeat(60));
		console.log('📊 Test Results');
		console.log('='.repeat(60));

		const summary = this.brain.getEpisodeSummary();

		console.log(`\n🎯 Prediction Accuracy:`);
		if (summary.accuracy.total > 0)
			console.log(`   ${summary.accuracy.correct}/${summary.accuracy.total} = ${(summary.accuracy.correct / summary.accuracy.total * 100).toFixed(2)}%`);

		const trader = this.traders[0];
		const netProfit = StockTrader.getPortfolioProfit(this.traders);
		console.log(`\n💰 Trading Performance:`);
		console.log(`   Total Actions: ${actions.length}`);
		console.log(`   Actual Trades: ${trader.totalTrades}`);
		console.log(`   Net Profit: $${netProfit.toFixed(2)}`);
		console.log(`   Position: ${trader.shares > 0 ? `OWN (${trader.shares} shares)` : 'OUT'}`);

		console.log(`\n📋 Action History:`);
		if (actions.length === 0) console.log('   No actions executed');
		else for (const action of actions) console.log(`   Frame ${action.frame} (Cycle ${action.cycleFrame}): ${action.action} at ${action.priceChange}`);
	}
}

await runJob(import.meta, SyntheticCycleTest);
