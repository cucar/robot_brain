import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from 'robot-brain';
import { StockEncoder } from '../encoder.js';
import { StockTrader } from '../trader.js';

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
			// Stock symbols to train on
			symbols: [
				'KGC', 'GLD', 'SPY', 'AAPL', 'NEM', 'GDX', 'NVDA', 'AMZN', 'MSFT', 'AMD',
				// 'META', 'JPM', 'BAC', 'QQQ', 'IWM', 'AEM', 'WPM', 'NG', 'GOOGL', 'XOM',
				// 'CVX', 'JNJ', 'UNH', 'PFE', 'WMT', 'COST', 'KO', 'CAT', 'XLF', 'DIA',
				// 'INTC', 'CRM', 'ORCL', 'IBM', 'CSCO', 'TGT', 'HD', 'MCD', 'NKE', 'SBUX',
				// 'ABBV', 'MRK', 'BMY', 'LLY', 'GILD', 'SLB', 'OXY', 'FCX', 'MOS', 'CLF',
				// 'ADBE', 'NFLX', 'PYPL', 'SHOP', 'UBER', 'ABNB', 'SNAP', 'PINS', 'ROKU', 'GS',
				// 'MS', 'C', 'WFC', 'AXP', 'V', 'MA', 'COF', 'SCHW', 'BLK', 'BA',
				// 'LMT', 'GE', 'UPS', 'FDX', 'DE', 'HON', 'RTX', 'UNP', 'DAL', 'DIS',
				// 'CMCSA', 'PEP', 'PM', 'MO', 'CL', 'PG', 'EL', 'LULU', 'F', 'COIN',
				// 'LCID', 'PLTR', 'SOFI', 'MARA', 'RIOT', 'GME', /* 'AMC', */ 'TWLO', 'ZM', 'SNOW'
			],
			timeframe: '3H',                     // Timeframe for data (e.g., '1D', '1Min')
			maxEpisodes: 1,                      // Number of training episodes (can be overridden with --episodes)
			holdoutRows: 0,                      // Number of rows to hold out from end for prediction testing (can be overridden with --holdout)
			offsetRows: 0,                       // Number of rows to skip from start (can be overridden with --offset)
			extendedHours: false,                // Include extended hours data (pre-market/after-hours) - use --extended-hours
			randomBaseline: false                // Skip the brain entirely; pick own/out + symbol uniformly at random - use --random-baseline
		};

		this.encoders = [];
		this.traders = [];

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	/**
	 * Apply command line options to config
	 */
	applyOptions() {
		const episodesIndex = process.argv.indexOf('--episodes');
		if (episodesIndex !== -1 && process.argv[episodesIndex + 1]) this.config.maxEpisodes = parseInt(process.argv[episodesIndex + 1]);

		const holdoutIndex = process.argv.indexOf('--holdout');
		if (holdoutIndex !== -1 && process.argv[holdoutIndex + 1]) this.config.holdoutRows = parseInt(process.argv[holdoutIndex + 1]);

		const offsetIndex = process.argv.indexOf('--offset');
		if (offsetIndex !== -1 && process.argv[offsetIndex + 1]) this.config.offsetRows = parseInt(process.argv[offsetIndex + 1]);

		const timeframeIndex = process.argv.indexOf('--timeframe');
		if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) this.config.timeframe = process.argv[timeframeIndex + 1];

		if (process.argv.includes('--extended-hours')) this.config.extendedHours = true;

		const symbolsIndex = process.argv.indexOf('--symbols');
		if (symbolsIndex !== -1 && process.argv[symbolsIndex + 1]) this.config.symbols = process.argv[symbolsIndex + 1].split(',');

		const maxPositionsIndex = process.argv.indexOf('--max-positions');
		if (maxPositionsIndex !== -1 && process.argv[maxPositionsIndex + 1]) StockTrader.maxPositions = parseInt(process.argv[maxPositionsIndex + 1]);

		const maxPriceIndex = process.argv.indexOf('--max-price');
		if (maxPriceIndex !== -1 && process.argv[maxPriceIndex + 1]) StockTrader.maxPrice = parseFloat(process.argv[maxPriceIndex + 1]);

		const initialCapitalIndex = process.argv.indexOf('--initial-capital');
		if (initialCapitalIndex !== -1 && process.argv[initialCapitalIndex + 1]) {
			StockTrader.initialCapital = parseFloat(process.argv[initialCapitalIndex + 1]);
			StockTrader.cash = StockTrader.initialCapital;
		}

		if (process.argv.includes('--random-baseline')) this.config.randomBaseline = true;
	}

	/**
	 * Create an encoder + trader per symbol and register the encoder's spec with the brain.
	 * The trader borrows the encoder's channelId so rewards, inputs, and inferences all key
	 * off a single number per symbol — otherwise we'd need a second id→symbol lookup.
	 */
	async registerBrainChannels() {
		for (const symbol of this.config.symbols) {
			const encoder = new StockEncoder(symbol);
			const trader = new StockTrader(symbol);

			// Brain allocates the channel ID and per-dim IDs. Wire the channelId into
			// both encoder and trader so they share a single per-symbol key for
			// inputs/rewards/inferences; the encoder additionally stashes the dim IDs.
			const { channelId, dimensionIds } = this.brain.registerChannelSpec(encoder.getChannelSpec());
			encoder.bindIds({ channelId, dimensionIds });
			trader.bindChannelId(channelId);

			this.encoders.push(encoder);
			this.traders.push(trader);
		}
	}

	/**
	 * Hook: after brain init, load each symbol's CSV, slice off the holdout (from the tail)
	 * and offset (from the head), and hand the resulting chronological rows to the encoder.
	 * Runs once per job, before any episodes — rows are reused across episodes via resetFrames().
	 */
	async configureChannels() {
		const { timeframe, holdoutRows, offsetRows } = this.config;
		const dataDir = path.join(__dirname, '..', 'data', timeframe);

		for (const encoder of this.encoders) {
			const csvPath = path.join(dataDir, `${encoder.symbol}.csv`);
			const allRows = this.loadCsvRows(csvPath);

			// Offset trims from the start (skip warmup bars), holdout trims from the end
			// (reserved for out-of-sample prediction testing in a separate job invocation).
			const startIndex = offsetRows;
			const endIndex = holdoutRows > 0 ? allRows.length - holdoutRows : allRows.length;
			const rows = allRows.slice(startIndex, endIndex);

			encoder.setData(rows);
		}
	}

	/**
	 * Load and parse a CSV file into {price, volume} row objects. Rows are already in
	 * chronological order from processAndSaveSymbolData, so no sort is needed. Empty lines
	 * are filtered so a trailing newline doesn't produce a NaN row.
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
		if (this.config.randomBaseline) console.log(`🎲 Random baseline mode (brain disabled)`);
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
	 * Run one episode: reset portfolio cash, rewind each encoder to frame 0, and wipe each
	 * trader's per-episode context (position, last action, etc.). Then stream frames through
	 * the brain, apply inferences back to the traders, and let the portfolio coordinate
	 * execution per frame. Learned brain state (dimensions, neuron weights) is preserved —
	 * only episode context is reset, so successive episodes build on prior learning.
	 */
	async runEpisode() {
		const startTime = Date.now();
		console.log(`📈 Episode ${this.currentEpisode}/${this.config.maxEpisodes}... `);

		// Reset context but keep learned patterns
		this.brain.resetContext();
		StockTrader.resetPortfolio();
		for (const encoder of this.encoders) encoder.resetFrames();
		for (const trader of this.traders) trader.resetContext();

		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			netProfit: 0,
			totalTrades: 0,
			channelResults: new Map(),
			baseAccuracy: null,
			overallAccuracy: null
		};

		// Warmup: consume the first frame per encoder so the first real processed frame
		// has a full (previous, current) pair. Mirrors the legacy frame-1 double-read in
		// StockChannel — without this, the first encoded frame would have previousPrice=null
		// and be skipped, and the trader would start one frame behind the encoder.
		for (let i = 0; i < this.encoders.length; i++) {
			const frame = this.encoders[i].nextFrame();
			if (frame) this.traders[i].setFrame(frame.price, frame.volume);
		}

		// Every encoder has the same row count by construction (aligned intervals). One
		// frame was consumed by the warmup above, so the main loop processes one fewer.
		const expectedFrames = this.encoders[0].rows.length - 1;

		// Process all frames for the episode duration
		let frameCount = 0;
		while (frameCount < expectedFrames) {

			// Process frame
			const hasMore = await this.runFrame();
			if (!hasMore) break;
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

		// Format ROI output
		const roiStr = episodeMetrics.totalROIPercent >= 0 ? `+${episodeMetrics.totalROIPercent.toFixed(2)}%` : `${episodeMetrics.totalROIPercent.toFixed(2)}%`;
		const perFrameROIStr = episodeMetrics.perFrameROI !== undefined ? `, ${(episodeMetrics.perFrameROIPercent >= 0 ? '+' : '')}${episodeMetrics.perFrameROIPercent.toFixed(6)}%/frame` : '';

		console.log(`✅ Net: $${episodeMetrics.netProfit.toFixed(2)} | ROI: ${roiStr} over ${episodeMetrics.frames} frames${perFrameROIStr} (${episodeMetrics.totalTrades} trades, ${duration}ms)`);
	}

	/**
	 * Process one frame: pull the next frame per encoder, hand the encoded scalars and
	 * any per-trader rewards to the brain, apply the returned inferences back to each
	 * trader, and let the portfolio coordinate the resulting buys/sells.
	 * @returns {Promise<boolean>} false when all encoders are exhausted
	 */
	async runFrame() {
		const inputs = new Map();
		const rewards = new Map();
		const randomBaseline = this.config.randomBaseline;

		// Pull one frame per encoder. Even frames that don't produce an encoded input
		// (e.g. zero-volume bar, first frame) still update the trader's price/volume so
		// valuation and reward math see the most recent reading. In --random-baseline
		// mode the encoded scalars are never consumed, so skip the encode() call.
		let anyFrames = false;
		for (let i = 0; i < this.encoders.length; i++) {
			const encoder = this.encoders[i];
			const trader = this.traders[i];
			const frame = encoder.nextFrame();
			if (!frame) continue;
			anyFrames = true;
			trader.setFrame(frame.price, frame.volume);
			if (!randomBaseline) {
				const dimMap = encoder.encode(frame);
				if (dimMap) inputs.set(encoder.channelId, dimMap);
			}
		}
		if (!anyFrames) return false;

		let frame;
		if (randomBaseline) {
			// Skip the brain entirely. 50% chance to be fully out; otherwise pick one
			// trader uniformly to OWN. No reward collection, no processFrame, no encode —
			// this is the baseline against which the brain is measured, so it must not
			// share any of the brain's per-frame work.
			const ownThisFrame = Math.random() < 0.5;
			const ownedIndex = ownThisFrame ? Math.floor(Math.random() * this.traders.length) : -1;
			for (let i = 0; i < this.traders.length; i++)
				this.traders[i].setAction(i === ownedIndex ? 1 : -1, 0);
		}
		else {
			// Only report reward for traders that actually acted last frame. This matches the
			// legacy path, which called channel.getRewards() only when the channel had inferred
			// actions in the previous frame — reporting a reward for a trader that never acted
			// would credit/punish neurons that weren't responsible.
			for (const trader of this.traders)
				if (trader.lastAction !== null) rewards.set(trader.channelId, trader.getReward());

			// Brain returns inferences keyed by channelId plus per-frame diagnostic data
			// (timing, optional vote debug). Destructure so we can pass `frame` straight
			// to the renderer — no separate getter call into the brain.
			const result = this.brain.processFrame(inputs, rewards);
			frame = result.frame;
			for (const trader of this.traders)
				trader.apply(result.inferences.get(trader.channelId) ?? []);
		}

		// Coordinated portfolio execution: ranks OWN actions, sizes positions, and runs
		// sells-then-buys so cash is freed before it's spent.
		await StockTrader.executePortfolio(this.traders);

		// Host-side rendering happens here (after portfolio execution so the tail
		// reflects the positions the trader just took).
		this.renderFrame(frame);

		// Step-debug pause between frames (no-op unless --wait is set).
		await this.waitForUser('Press Enter to continue to next frame');

		// Yield to the event loop so SIGINT handlers can fire between frames — without this,
		// a tight synchronous loop ignores Ctrl+C until the episode finishes.
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
	 * Vote-dump formatters for the spec-based path: each StockEncoder owns its own
	 * action labels and bucket-to-percent map, so we hand the encoders to the
	 * renderer keyed by channel name (== symbol).
	 */
	getChannelFormatters() {
		const map = new Map();
		for (const encoder of this.encoders) map.set(encoder.symbol, encoder);
		return map;
	}

	/**
	 * Populate episodeMetrics from the traders themselves — no Channel class aggregation.
	 * Computes portfolio-level ROI (total and compounded per-frame) plus per-symbol stats
	 * for the results display.
	 */
	collectEpisodeResults(episodeMetrics) {

		// Portfolio profit = (cash + market value of all positions) - initial capital.
		episodeMetrics.netProfit = StockTrader.getPortfolioProfit(this.traders);

		// Total ROI as a ratio (1.0 == breakeven) and as a percentage delta.
		const finalValue = StockTrader.initialCapital + episodeMetrics.netProfit;
		const totalROI = finalValue / StockTrader.initialCapital;
		episodeMetrics.totalROI = totalROI;
		episodeMetrics.totalROIPercent = (totalROI - 1) * 100;

		// Geometric per-frame return — the constant rate that, compounded over `frames`
		// frames, reproduces the episode's total ROI. Makes episodes of different lengths
		// comparable.
		if (episodeMetrics.frames > 0) {
			const perFrameROI = Math.pow(totalROI, 1 / episodeMetrics.frames) - 1;
			episodeMetrics.perFrameROI = perFrameROI;
			episodeMetrics.perFrameROIPercent = perFrameROI * 100;
		}

		// Per-symbol breakdown for the results table. Unrealized profit uses current market
		// price vs. cost basis — realized P&L is already baked into StockTrader.cash.
		for (const trader of this.traders) {
			const currentValue = trader.shares * trader.getCurrentPrice();

			const channelResult = {
				symbol: trader.symbol,
				investment: trader.investment,
				currentValue,
				unrealizedProfit: currentValue - trader.investment,
				trades: trader.totalTrades
			};

			episodeMetrics.channelResults.set(trader.symbol, channelResult);
			episodeMetrics.totalTrades += trader.totalTrades;
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
		console.log(`   Starting Capital: $${StockTrader.initialCapital.toFixed(2)}`);
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

await runJob(import.meta, StockTestJob);