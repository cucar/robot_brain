import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from 'robot-brain';
import { TextEncoder } from '../encoder.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Text Test Job - Trains the brain on a repeating character pattern and reports per-episode
 * prediction accuracy. Uses the spec-based registration path (registerChannelSpec + an
 * Encoder it owns directly) — no Channel subclass.
 */
export default class TextTestJob extends Job {

	constructor() {
		super();

		// Simple configuration - edit these values as needed. The training text is
		// loaded from `file` at job start (default ../data/test.txt — the legacy
		// hard-coded sample). Override with --file <path> to point at any text
		// file (e.g. ../data/abramov.txt).
		this.config = {
			file: path.join(__dirname, '..', 'data', 'test.txt'),
			maxEpisodes: 5,
			iterationsPerEpisode: 1
		};

		// Single encoder for now — kept as an array to mirror stocks' multi-encoder shape,
		// so adding additional text streams later doesn't require restructuring.
		this.encoders = [];

		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	applyOptions() {
		const episodesIndex = process.argv.indexOf('--episodes');
		if (episodesIndex !== -1 && process.argv[episodesIndex + 1]) this.config.maxEpisodes = parseInt(process.argv[episodesIndex + 1]);

		// --file: bare names (no directory) resolve against ../data/, so `--file abramov.txt`
		// just works. Anything containing a slash/backslash is resolved against cwd as a path.
		const fileIndex = process.argv.indexOf('--file');
		if (fileIndex !== -1 && process.argv[fileIndex + 1]) {
			const arg = process.argv[fileIndex + 1];
			this.config.file = path.basename(arg) === arg
				? path.join(__dirname, '..', 'data', arg)
				: path.resolve(arg);
		}

		const iterationsIndex = process.argv.indexOf('--iterations');
		if (iterationsIndex !== -1 && process.argv[iterationsIndex + 1]) this.config.iterationsPerEpisode = parseInt(process.argv[iterationsIndex + 1]);
	}

	/**
	 * Create the encoder and hand its spec to the brain. The brain allocates the channel
	 * ID and per-dim IDs; we wire those back onto the encoder so encode() outputs key off
	 * the same numbers the job uses as Map keys.
	 */
	async initialize() {
		const encoder = new TextEncoder('text');
		const ids = this.brain.registerChannelSpec(encoder.getChannelSpec());
		encoder.bindIds(ids);
		this.encoders.push(encoder);
	}

	/**
	 * Load the training string from the configured file once per job and hand it
	 * to the encoder. resetFrames() is called per episode so the same sequence
	 * is re-streamed without re-reading from disk.
	 */
	async configureChannels() {
		this.pattern = fs.readFileSync(this.config.file, 'utf-8');
		const text = this.pattern.repeat(this.config.iterationsPerEpisode);
		for (const encoder of this.encoders) encoder.setData(text);
	}

	async showStartupInfo() {
		console.log(`🚀 Starting Text Test Job`);
		console.log(`📄 File: ${this.config.file}`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`🔁 Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log('');
	}

	async executeJob() {
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {
			await this.runEpisode();
			if (this.isShuttingDown) return;
			if (this.currentEpisode % 10 === 0 || this.currentEpisode === this.config.maxEpisodes)
				this.showProgress();
		}
	}

	async showResults() {
		this.showFinalResults();
	}

	/**
	 * One episode: rewind encoders, reset brain context (keeps learned patterns), then
	 * stream every character through the brain. No rewards (text channel doesn't emit any).
	 */
	async runEpisode() {
		const startTime = Date.now();
		console.log(`📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes}...`);

		this.brain.resetContext();
		for (const encoder of this.encoders) encoder.resetFrames();

		const episodeMetrics = {
			episode: this.currentEpisode,
			baseAccuracy: null
		};

		const expectedFrames = this.pattern.length * this.config.iterationsPerEpisode;

		let frameCount = 0;
		while (frameCount < expectedFrames) {
			const hasMore = await this.runFrame();
			if (!hasMore) break;
			frameCount++;

			if (frameCount % 100 === 0)
				process.stdout.write(`\r📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes} - Frame ${frameCount}/${expectedFrames}...`);

			if (this.isShuttingDown) return;
		}

		// Clear progress line (TTY only).
		if (!process.stdout.isTTY) process.stdout.write('\n');
		else {
			process.stdout.write('\r');
			process.stdout.clearLine?.(0);
		}

		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		const summary = this.brain.getEpisodeSummary();
		if (summary.accuracy.total > 0)
			episodeMetrics.baseAccuracy = (summary.accuracy.correct / summary.accuracy.total * 100);

		this.episodeResults.push(episodeMetrics);

		const accStr = episodeMetrics.baseAccuracy !== null
			? `${episodeMetrics.baseAccuracy.toFixed(2)}%`
			: 'N/A';
		console.log(`✅ Accuracy: ${accStr} (${frameCount} frames, ${duration}ms)`);

		this.showMispredictions(summary.mispredictions);
	}

	/**
	 * Pull one frame per encoder, build the inputs Map keyed by channelId, send it
	 * through the brain. Empty rewards Map — text doesn't reward.
	 * @returns {Promise<boolean>} false when all encoders are exhausted
	 */
	/**
	 * Vote-dump formatters: text encoders keyed by channel name. Text has no
	 * actions (event-only channel), so only `name` is needed — coords fall back
	 * to the raw string. The renderer's defaults handle the missing methods.
	 */
	getChannelFormatters() {
		const map = new Map();
		for (const encoder of this.encoders) map.set(encoder.channelId, encoder);
		return map;
	}

	async runFrame() {
		const inputs = new Map();
		const rewards = new Map();

		// Pull one character per encoder and key the encoded scalar map by channelId, the
		// same shape the brain uses for stocks. Empty frame → encoders are exhausted.
		let anyFrames = false;
		for (const encoder of this.encoders) {
			const frame = encoder.nextFrame();
			if (!frame) continue;
			anyFrames = true;
			const dimMap = encoder.encode(frame);
			if (dimMap) inputs.set(encoder.channelId, dimMap);
		}
		if (!anyFrames) return false;

		// Text doesn't reward (no environment feedback), so rewards stays empty. We only
		// need `frame` from the return — no actions to dispatch since there's no trader.
		const { frame } = this.brain.processFrame(inputs, rewards);

		// Host-side rendering: emits the per-frame summary line / vote debug / start-of-frame
		// info per the flags the host owns. Text channel has no app-layer tail to append.
		this.renderFrame(frameResult);

		// Step-debug pause between frames (no-op unless --wait is set).
		await this.waitForUser('Press Enter to continue to next frame');

		// Yield to the event loop so SIGINT can fire between frames.
		await new Promise(resolve => setImmediate(resolve));
		return true;
	}

	charFromCode(code) {
		if (code === 32) return '␣';
		if (code === 10) return '↵';
		if (code === 9) return '→';
		if (code < 32) return `\\x${code.toString(16).padStart(2, '0')}`;
		return String.fromCharCode(code);
	}

	showMispredictions(mispredictions) {
		if (!mispredictions || mispredictions.length === 0) return;

		const grouped = new Map();
		for (const m of mispredictions) {
			const predChar = m.predicted.bucketId;
			const actualChar = m.actual.bucketId;
			const key = `${predChar}→${actualChar}`;
			grouped.set(key, (grouped.get(key) || 0) + 1);
		}

		const items = [];
		for (const [key, count] of grouped) {
			const [pred, actual] = key.split('→').map(Number);
			const predStr = this.charFromCode(pred);
			const actualStr = this.charFromCode(actual);
			items.push(`'${predStr}'→'${actualStr}'${count > 1 ? `(×${count})` : ''}`);
		}

		console.log(`   ❌ Mispredictions: ${items.join(', ')}`);
	}

	showProgress() {
		console.log(`\n📊 Training Progress (Episode ${this.currentEpisode}/${this.config.maxEpisodes}):`);

		if (this.episodeResults.length >= 10) {
			const recent10 = this.episodeResults.slice(-10);
			const avgAcc = recent10.reduce((sum, ep) => sum + (ep.baseAccuracy || 0), 0) / recent10.length;
			console.log(`   Last 10 episodes avg accuracy: ${avgAcc.toFixed(2)}%`);
		}

		const validResults = this.episodeResults.filter(ep => ep.baseAccuracy !== null);
		if (validResults.length > 0) {
			const bestEpisode = validResults.reduce((best, ep) =>
				(ep.baseAccuracy || 0) > (best.baseAccuracy || 0) ? ep : best);
			const worstEpisode = validResults.reduce((worst, ep) =>
				(ep.baseAccuracy || 0) < (worst.baseAccuracy || 0) ? ep : worst);

			console.log(`   Best episode: #${bestEpisode.episode} (${bestEpisode.baseAccuracy?.toFixed(2)}%)`);
			console.log(`   Worst episode: #${worstEpisode.episode} (${worstEpisode.baseAccuracy?.toFixed(2)}%)`);
		}
		console.log('');
	}

	showFinalResults() {
		console.log(`\n🎯 Final Training Results (${this.config.maxEpisodes} episodes):`);
		console.log('='.repeat(60));

		const validResults = this.episodeResults.filter(ep => ep.baseAccuracy !== null);
		if (validResults.length === 0) {
			console.log('No valid accuracy data collected.');
			console.log('='.repeat(60));
			return;
		}

		const avgAccuracy = validResults.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / validResults.length;

		console.log(`📈 Overall Performance:`);
		console.log(`   File: ${this.config.file}`);
		console.log(`   Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log(`   Average Accuracy: ${avgAccuracy.toFixed(2)}%`);

		console.log(`\n📊 Accuracy by Episode:`);
		for (const ep of this.episodeResults) {
			const accStr = ep.baseAccuracy !== null ? `${ep.baseAccuracy.toFixed(2)}%` : 'N/A';
			console.log(`   Episode ${ep.episode}: ${accStr} (${ep.frames} frames)`);
		}

		// First-half / second-half comparison flags whether learning is improving.
		if (validResults.length >= 4) {
			const firstHalf = validResults.slice(0, Math.floor(validResults.length / 2));
			const secondHalf = validResults.slice(Math.floor(validResults.length / 2));

			const firstAvg = firstHalf.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / firstHalf.length;
			const secondAvg = secondHalf.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / secondHalf.length;
			const improvement = secondAvg - firstAvg;

			console.log(`\n📈 Learning Progress:`);
			console.log(`   First half avg: ${firstAvg.toFixed(2)}%`);
			console.log(`   Second half avg: ${secondAvg.toFixed(2)}%`);
			console.log(`   Improvement: ${improvement >= 0 ? '+' : ''}${improvement.toFixed(2)}pp ${improvement >= 0 ? '📈' : '📉'}`);
		}

		console.log('\n🏆 Best Episodes (by Accuracy):');
		const sortedByAccuracy = [...validResults].sort((a, b) => b.baseAccuracy - a.baseAccuracy);
		for (let i = 0; i < Math.min(5, sortedByAccuracy.length); i++) {
			const ep = sortedByAccuracy[i];
			console.log(`   #${ep.episode}: ${ep.baseAccuracy.toFixed(2)}%`);
		}

		console.log('='.repeat(60));
	}
}

await runJob(import.meta, TextTestJob);
