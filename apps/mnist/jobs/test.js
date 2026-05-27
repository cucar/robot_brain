import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Job, runJob} from 'robot-brain';
import {MNISTPixelChannelsEncoder} from '../encoders/pixel_channels_encoder.js';
import {loadImages, loadLabels} from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * MNIST job: 784-pixel-channel serial training. Each image is one frame
 * (all 784 channels fire simultaneously); the same frame is presented
 * `maxFrames` times per image so the brain converges on a static input.
 * After the frame loop, learn() wires actions to active voters and
 * infer() measures post-learn accuracy. Consensus is per-voter reward
 * normalized (each voter contributes total weight 1).
 */
export default class MNISTTestJob extends Job {

	/**
	 * Initialize job config with defaults. Real values arrive in applyOptions().
	 */
	constructor() {
		super();
		this.config = {
			buckets: 2,           // 2 = binary pixels (one of two sensory neurons fires per channel)
			maxImages: 1,         // start tiny — scale up once convergence is verified
			maxTestImages: 20,
			maxTrainTest: 20,     // also classify a subset of training images (memorization check)
			skipTest: false,
			episodes: 1,
			maxFrames: 6,         // each image presented this many times in a row
			dynamicFrames: false, // when true: stop the frame loop the moment Δneurons hits 0
		};

		this.encoder = null;
		this.trainImages = null;
		this.trainLabels = null;
		this.testImages = null;
		this.testLabels = null;
		this.trainBits = null;
		this.testBits = null;
	}

	/**
	 * Parse MNIST-specific CLI flags and stamp brain-default values for any
	 * brain flag the user did not explicitly set on the command line.
	 */
	applyOptions() {
		const num = (flag) => {
			const i = process.argv.indexOf(flag);
			return i !== -1 && process.argv[i + 1] !== undefined ? parseInt(process.argv[i + 1]) : null;
		};

		// MNIST-specific flags
		const buckets = num('--buckets');
		if (buckets !== null) this.config.buckets = buckets;
		const maxImages = num('--max-images');
		if (maxImages !== null) this.config.maxImages = maxImages;
		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;
		const maxTrainTest = num('--max-train-test');
		if (maxTrainTest !== null) this.config.maxTrainTest = maxTrainTest;
		const episodes = num('--episodes');
		if (episodes !== null) this.config.episodes = episodes;
		const maxFrames = num('--max-frames');
		if (maxFrames !== null) this.config.maxFrames = maxFrames;
		if (process.argv.includes('--skip-test')) this.config.skipTest = true;
		if (process.argv.includes('--dynamic-frames')) this.config.dynamicFrames = true;

		// Brain defaults — tiny context, loose merge, no forgetting, lax error correction.
		// The "??=" pattern lets `--context-length 5` on the command line override us.
		if (this.options.contextLength == null) this.options.contextLength = 3;
		if (this.options.mergeThreshold == null) this.options.mergeThreshold = 0.5;
		if (this.options.patternForgetRate == null) this.options.patternForgetRate = 0;
		// if (this.options.errorCorrectionMode == null) this.options.errorCorrectionMode = 'static';
		// if (this.options.errorCorrectionThreshold == null) this.options.errorCorrectionThreshold = 0.5;
	}

	/**
	 * Build the encoder and register its 784 pixel channels + 1 digit action
	 * channel with the brain.
	 */
	async initialize() {
		this.encoder = new MNISTPixelChannelsEncoder(this.config.buckets);
		this.encoder.registerChannels(this.brain);
	}

	/**
	 * Load MNIST IDX files (plain or .gz), apply size caps, and pre-quantize
	 * every image so the training loop doesn't pay encoding cost per frame.
	 */
	async configureChannels() {
		const dataDir = path.join(__dirname, '..', 'data');

		const trainImgPath = this.findDataFile(dataDir, 'train-images-idx3-ubyte');
		const trainLblPath = this.findDataFile(dataDir, 'train-labels-idx1-ubyte');
		const testImgPath = this.findDataFile(dataDir, 't10k-images-idx3-ubyte');
		const testLblPath = this.findDataFile(dataDir, 't10k-labels-idx1-ubyte');

		this.trainImages = loadImages(trainImgPath);
		this.trainLabels = loadLabels(trainLblPath);
		this.testImages = loadImages(testImgPath);
		this.testLabels = loadLabels(testLblPath);

		// Truncate datasets for fast iteration. maxImages=0 means "no cap".
		if (this.config.maxImages > 0) {
			this.trainImages = this.trainImages.slice(0, this.config.maxImages);
			this.trainLabels = this.trainLabels.slice(0, this.config.maxImages);
		}
		if (this.config.maxTestImages > 0) {
			this.testImages = this.testImages.slice(0, this.config.maxTestImages);
			this.testLabels = this.testLabels.slice(0, this.config.maxTestImages);
		}

		// Pre-quantize raw 0–255 pixels into bucket ids (0 or 1 in binary mode).
		this.trainBits = this.trainImages.map(px => this.encoder.buildBits(px));
		this.testBits = this.testImages.map(px => this.encoder.buildBits(px));
	}

	/**
	 * Try the plain IDX path first, fall back to its .gz variant.
	 */
	findDataFile(dataDir, baseName) {
		const plain = path.join(dataDir, baseName);
		if (fs.existsSync(plain)) return plain;
		const gz = path.join(dataDir, `${baseName}.gz`);
		if (fs.existsSync(gz)) return gz;
		throw new Error(`MNIST data not found: ${plain} (run download.js first)`);
	}

	/**
	 * Print the resolved configuration so the run is self-documenting in logs.
	 */
	async showStartupInfo() {
		console.log(`MNIST Digit Recognition — 784-Channel Per-Pixel, Serial Training`);
		console.log(`  Buckets: ${this.config.buckets}`);
		console.log(`  Context length: ${this.options.contextLength}`);
		console.log(`  Merge threshold: ${this.options.mergeThreshold}`);
		console.log(`  Forget rate: ${this.options.patternForgetRate}`);
		console.log(`  Error mode: ${this.options.errorCorrectionMode} @ ${this.options.errorCorrectionThreshold}`);
		console.log(`  Episodes: ${this.config.episodes}`);
		console.log(`  Max frames/image: ${this.config.maxFrames}${this.config.dynamicFrames ? ' (dynamic, early exit on convergence)' : ' (static)'}`);
		console.log(`  Training images cap: ${this.config.maxImages}`);
		console.log(`  Classify (train subset): ${this.config.maxTrainTest}`);
		console.log(`  Classify (held-out): ${this.config.skipTest ? 'skipped' : this.config.maxTestImages}`);
		console.log('');
	}

	/**
	 * Two phases: serial training (Phase 1) then cold-replay classification
	 * (Phase 2). Phase 1 is fully interruptable — Ctrl-C between images
	 * prints a partial summary and returns cleanly.
	 */
	async executeJob() {
		const N = this.trainImages.length;
		const EMPTY_MAP = new Map();

		console.log(`Phase 1: serial training — ${this.config.episodes} ep × ${N} img × ≤${this.config.maxFrames} frames`);
		console.log('');

		// Running totals across the whole training phase (kept outside the loops so
		// the shutdown path can print whatever we've accumulated so far).
		let postLearnCorrect = 0;
		let postLearnTotal = 0;
		const convergedAt = [];   // per-image frame index where Δneurons first hit 0 (-1 = never)
		const phaseStart = Date.now();
		// Accumulated brain-side timings across all training frames — printed in
		// the phase summary so we can see where the seconds go.
		const phaseTimings = this.newTimings();

		for (let ep = 0; ep < this.config.episodes; ep++) {
			console.log(`── episode ${ep + 1}/${this.config.episodes} ──`);
			for (let i = 0; i < N; i++) {
				// Per-image interrupt check — flush partial summary on the way out.
				if (this.isShuttingDown) { this.printPhase1Summary(postLearnCorrect, postLearnTotal, convergedAt, phaseStart); return; }

				const imgStart = Date.now();
				const bits = this.trainBits[i];
				const label = this.trainLabels[i];

				// Wipe the temporal window so this image starts from a clean context.
				// Then put the brain in event-only learning mode (actions off — we
				// don't want stale action neurons leaking into the frame loop).
				this.brain.resetContext();
				this.brain.setProcessingMode(true, false, true);
				const inputs = this.encoder.encodeImage(bits);

				// Per-frame convergence tracking: log Δneurons each frame so we can
				// see how many repetitions of a static image it takes to stop
				// creating new patterns.
				const frameLog = [];
				let prevNeurons = this.brain.getFrameSummary().neuronCount;
				let convFrame = -1;

				const imgTimings = this.newTimings();

				for (let f = 0; f < this.config.maxFrames; f++) {
					const fr = this.brain.processFrame(inputs, EMPTY_MAP, EMPTY_MAP);
					this.addTimings(imgTimings, fr.timings);
					const s = this.brain.getFrameSummary();
					const delta = s.neuronCount - prevNeurons;
					frameLog.push(`f${f + 1}+${delta}n/L${s.maxLevel}`);
					// First frame where no new neurons appear = converged on this image.
					if (delta === 0 && convFrame < 0) convFrame = f + 1;
					prevNeurons = s.neuronCount;
					if (this.isShuttingDown) { this.printPhase1Summary(postLearnCorrect, postLearnTotal, convergedAt, phaseStart, phaseTimings); return; }
					// Dynamic mode: stop the moment we converge, don't waste frames.
					if (this.config.dynamicFrames && delta === 0) break;
				}
				convergedAt.push(convFrame);
				this.addTimings(phaseTimings, imgTimings);

				// Wire digit actions to whichever voters are currently active.
				// Rewards are additive on the action connection per learn() call.
				this.encoder.resetActions();
				const actions = this.encoder.encodeAction(label);
				const rewards = this.encoder.buildRewards(label, 1.0, -1.0);
				const learnResult = this.brain.learn(actions, rewards);

				// Verification: did the connections we just wrote actually predict
				// the right digit? Uses per-voter reward normalization (each voter
				// contributes total weight 1 across the digits it votes for).
				const predicted = this.encoder.predictByVoterNormalizedConsensus(learnResult.actionVotes);
				const ok = predicted === label;
				if (ok) postLearnCorrect++;
				postLearnTotal++;

				const totalS = this.brain.getFrameSummary();
				const imgMs = Date.now() - imgStart;
				const convStr = convFrame > 0 ? `conv@${convFrame}` : 'no-conv';
				console.log(`  ep${ep + 1} img ${i + 1}/${N} lbl=${label}  ${frameLog.join(' ')}  ${convStr}  pred=${predicted}${ok ? '✓' : '✗'}  tot=${totalS.neuronCount}n/L${totalS.maxLevel}  ${imgMs}ms`);
				// Per-image timing breakdown — shows where the seconds went inside processFrame.
				console.log(`    timings: ${this.formatTimings(imgTimings)}`);
			}
		}

		this.printPhase1Summary(postLearnCorrect, postLearnTotal, convergedAt, phaseStart, phaseTimings);

		// Switch to read-only inference mode: events on so the test image still
		// activates patterns, learning off so we don't keep creating neurons.
		this.brain.setProcessingMode(true, false, false);

		// Re-classify a slice of training images first — this is the memorization
		// check (high acc here, low acc on test = overfit / brittle).
		const numTrainTest = Math.min(this.config.maxTrainTest, N);
		if (numTrainTest > 0) {
			console.log(`\nPhase 2a: classify ${numTrainTest} training images (cold replay)...`);
			this.trainResult = this.classifySet(this.trainBits.slice(0, numTrainTest), this.trainLabels.slice(0, numTrainTest), 'train');
			if (this.isShuttingDown) return;
		}

		if (!this.config.skipTest) {
			console.log(`\nPhase 2b: classify ${this.testImages.length} held-out test images...`);
			this.testResult = this.classifySet(this.testBits, this.testLabels, 'held');
		}
	}

	/**
	 * Print the running phase-1 totals — called both at the natural end of
	 * training and from each interrupt-check site so Ctrl-C doesn't lose
	 * whatever progress we already made.
	 */
	printPhase1Summary(correct, total, convergedAt, startMs, phaseTimings) {
		const ms = Date.now() - startMs;
		// Only count images that actually converged when computing "average frames-to-converge".
		const conv = convergedAt.filter(c => c > 0);
		const avgConv = conv.length > 0 ? (conv.reduce((a, b) => a + b, 0) / conv.length).toFixed(2) : '—';
		const convPct = total > 0 ? (conv.length / total * 100).toFixed(0) : '—';
		console.log('');
		console.log(`Phase 1 summary:`);
		console.log(`  Post-learn accuracy: ${total > 0 ? (correct / total * 100).toFixed(2) : '—'}% (${correct}/${total})`);
		console.log(`  Converged: ${convPct}% of images, avg frame-to-converge: ${avgConv}`);
		console.log(`  Elapsed: ${ms}ms`);
		if (phaseTimings) {
			console.log(`  Brain timings (total): ${this.formatTimings(phaseTimings)}`);
		}
	}

	/**
	 * Fresh per-section timing accumulator. Names match FrameTimings on the
	 * Rust side so addTimings() can do a straight field-wise add.
	 */
	newTimings() {
		return {
			buildFrame: 0, createSensory: 0, cleanupDead: 0, ageContext: 0,
			activate: 0, processLevels: 0, applyResults: 0, infer: 0, trackError: 0,
		};
	}

	/**
	 * Field-wise add of `delta` (e.g. one frame's timings) into `acc`.
	 */
	addTimings(acc, delta) {
		if (!delta) return;
		acc.buildFrame    += delta.buildFrame    ?? 0;
		acc.createSensory += delta.createSensory ?? 0;
		acc.cleanupDead   += delta.cleanupDead   ?? 0;
		acc.ageContext    += delta.ageContext    ?? 0;
		acc.activate      += delta.activate      ?? 0;
		acc.processLevels += delta.processLevels ?? 0;
		acc.applyResults  += delta.applyResults  ?? 0;
		acc.infer         += delta.infer         ?? 0;
		acc.trackError    += delta.trackError    ?? 0;
	}

	/**
	 * One-line timing breakdown in milliseconds. Sections are listed in
	 * pipeline order so the dominant cost stands out by position too.
	 */
	formatTimings(t) {
		const ms = (s) => (s * 1000).toFixed(1);
		const total = t.buildFrame + t.createSensory + t.cleanupDead + t.ageContext
		            + t.activate + t.processLevels + t.applyResults + t.infer + t.trackError;
		return `total=${ms(total)}ms  build=${ms(t.buildFrame)} sensory=${ms(t.createSensory)} cleanup=${ms(t.cleanupDead)} age=${ms(t.ageContext)} activate=${ms(t.activate)} levels=${ms(t.processLevels)} apply=${ms(t.applyResults)} infer=${ms(t.infer)} trackErr=${ms(t.trackError)}`;
	}

	/**
	 * Cold-replay classifier. For each image: reset context, present it for
	 * maxFrames repetitions in event-only mode, then call infer() and read
	 * the predicted digit out of actionVotes. Builds a confusion matrix
	 * along the way. Interruptable mid-loop.
	 */
	classifySet(bitsArr, labels, tag) {
		const EMPTY_MAP = new Map();
		let correct = 0;
		const perDigit = new Array(10).fill(0);
		const perDigitTotal = new Array(10).fill(0);
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
		const phaseTimings = this.newTimings();
		const t0 = Date.now();

		for (let i = 0; i < bitsArr.length; i++) {
			if (this.isShuttingDown) break;

			const imgStart = Date.now();
			this.brain.resetContext();
			const inputs = this.encoder.encodeImage(bitsArr[i]);
			const imgTimings = this.newTimings();
			// Same maxFrames as training so the brain reaches the same convergence
			// state it was in when learn() wired the actions.
			for (let f = 0; f < this.config.maxFrames; f++) {
				const fr = this.brain.processFrame(inputs, EMPTY_MAP, EMPTY_MAP);
				this.addTimings(imgTimings, fr.timings);
				if (this.isShuttingDown) break;
			}
			if (this.isShuttingDown) break;

			this.encoder.resetActions();
			const infer = this.brain.infer();
			this.addTimings(imgTimings, infer.timings);
			const predicted = this.encoder.predictByVoterNormalizedConsensus(infer.actionVotes);
			const label = labels[i];
			const ok = predicted === label;
			if (ok) { correct++; perDigit[label]++; }
			perDigitTotal[label]++;
			// Defensive: predictByVoterNormalizedConsensus can return -1 when no
			// voters cast action votes — exclude those from the confusion matrix.
			if (predicted >= 0 && predicted < 10) confusion[label][predicted]++;
			this.addTimings(phaseTimings, imgTimings);

			const imgMs = Date.now() - imgStart;
			console.log(`  ${tag} ${i + 1}/${bitsArr.length} lbl=${label} pred=${predicted}${ok ? '✓' : '✗'}  ${imgMs}ms  (running ${(correct / (i + 1) * 100).toFixed(1)}%)`);
			console.log(`    timings: ${this.formatTimings(imgTimings)}`);
		}

		const ms = Date.now() - t0;
		const total = bitsArr.length;
		const accuracy = total > 0 ? correct / total : 0;
		const perDigitStr = perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');
		console.log(`  ${tag} labelAcc: ${(accuracy * 100).toFixed(2)}% (${correct}/${total}) in ${ms}ms`);
		console.log(`    per-digit: ${perDigitStr}`);
		console.log(`    brain timings (total): ${this.formatTimings(phaseTimings)}`);
		return { accuracy, correct, total, perDigit, perDigitTotal, confusion };
	}

	/**
	 * Render the final accuracy block + confusion matrix. Skipped sections
	 * (--skip-test, maxTrainTest=0) just leave their rows out.
	 */
	async showResults() {
		console.log('\nResults');
		console.log('='.repeat(70));
		if (this.trainResult) {
			console.log(`  Training-set classify:  ${(this.trainResult.accuracy * 100).toFixed(2)}% (${this.trainResult.correct}/${this.trainResult.total})`);
		}
		if (this.testResult) {
			console.log(`  Held-out classify:      ${(this.testResult.accuracy * 100).toFixed(2)}% (${this.testResult.correct}/${this.testResult.total})`);
			console.log(`\n  Confusion (rows=actual, cols=predicted):`);
			let header = '       ';
			for (let p = 0; p < 10; p++) header += String(p).padStart(5);
			console.log(header);
			for (let a = 0; a < 10; a++) {
				let row = `   ${a}   `;
				for (let p = 0; p < 10; p++) row += String(this.testResult.confusion[a][p]).padStart(5);
				console.log(row);
			}
		}
		console.log('='.repeat(70));
	}
}

await runJob(import.meta, MNISTTestJob);
