import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Job, runJob} from 'robot-brain';
import {MNISTRowChannelsEncoder} from '../encoder.js';
import {loadImages, loadLabels} from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * MNIST Test Job — 28-channel column-scan training with parallel contexts.
 *
 * Architecture (the working recipe from implant_test):
 *   - 28 input channels, one per row. Each frame presents one column's pixels
 *     across all 28 channels. An image = 28 frames. context_length=30 covers
 *     the whole image, so L1 patterns natively encode columns and L2+ patterns
 *     encode whole-image column sequences.
 *   - Parallel-context training: brain.initContexts(N) creates one context per
 *     training image. We advance them column-by-column in lockstep with full
 *     event learning + error correction. The shared thalamus accumulates
 *     hierarchical patterns from all images. Refine_context and strengthen on
 *     match are gated by the learning flag in brain-core so cold replay is
 *     fully deterministic.
 *   - Action wiring (Phase 2): after training, for each context i, call
 *     brain.learn(label_i) so context i's end-of-image voters wire to action.
 *   - Classification: single-context replay (slot 0) with learning off. Read
 *     action consensus by SUM-OF-EVIDENCE (totalStrength × avgReward), not the
 *     brain's per-voter mean — additive reward + sum consensus preserve the
 *     frequency signal correctly.
 *
 * MNIST-specific flags:
 *   --buckets N            quantization level (default 2 = binary)
 *   --max-images N         cap training set (default: all 60K)
 *   --max-test-images N    cap test set (default 20)
 *   --max-train-test N     how many training images to also classify (default 20)
 *   --skip-test            skip held-out evaluation
 *
 * Brain flags inherited from base Job: --context-length, --merge-threshold,
 * --forget-rate, --error-mode, --error-threshold, --level-decay.
 */
export default class MNISTTestJob extends Job {

	constructor() {
		super();
		this.config = {
			buckets: 2,
			maxImages: 200,
			maxTestImages: 20,
			maxTrainTest: 20,
			skipTest: false,
		};

		this.encoder = null;
		this.trainImages = null;
		this.trainLabels = null;
		this.testImages = null;
		this.testLabels = null;
		this.trainBits = null;
		this.testBits = null;
	}

	applyOptions() {
		const num = (flag) => {
			const i = process.argv.indexOf(flag);
			return i !== -1 && process.argv[i + 1] !== undefined ? parseInt(process.argv[i + 1]) : null;
		};

		const buckets = num('--buckets');
		if (buckets !== null) this.config.buckets = buckets;

		const maxImages = num('--max-images');
		if (maxImages !== null) this.config.maxImages = maxImages;

		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;

		const maxTrainTest = num('--max-train-test');
		if (maxTrainTest !== null) this.config.maxTrainTest = maxTrainTest;

		if (process.argv.includes('--skip-test')) this.config.skipTest = true;

		if (this.options.contextLength == null) this.options.contextLength = 30;
		if (this.options.mergeThreshold == null) this.options.mergeThreshold = 1.0;
	}

	/**
	 * Register the single pixel channel with the brain.
	 */
	async initialize() {
		this.encoder = new MNISTRowChannelsEncoder(this.config.buckets);
		this.encoder.registerChannels(this.brain);
	}

	/**
	 * Load MNIST data from the IDX files in apps/mnist/data/. Accepts both
	 * plain and .gz variants — the loader auto-detects gzip compression.
	 * Optionally truncates the datasets per --max-images / --max-test-images
	 * for faster iteration during development. Pre-encodes every image's
	 * bit stream up front so episodes don't pay the quantize/trim cost.
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

		if (this.config.maxImages > 0) {
			this.trainImages = this.trainImages.slice(0, this.config.maxImages);
			this.trainLabels = this.trainLabels.slice(0, this.config.maxImages);
		}
		if (this.config.maxTestImages > 0) {
			this.testImages = this.testImages.slice(0, this.config.maxTestImages);
			this.testLabels = this.testLabels.slice(0, this.config.maxTestImages);
		}

		this.trainBits = this.trainImages.map(px => this.encoder.buildBits(px));
		this.testBits = this.testImages.map(px => this.encoder.buildBits(px));
	}

	/**
	 * Locate a data file, trying both uncompressed and .gz variants.
	 */
	findDataFile(dataDir, baseName) {
		const plain = path.join(dataDir, baseName);
		if (fs.existsSync(plain)) return plain;
		const gz = path.join(dataDir, `${baseName}.gz`);
		if (fs.existsSync(gz)) return gz;
		throw new Error(`MNIST data not found: ${plain} (run download.js first)`);
	}

	async showStartupInfo() {
		console.log(`MNIST Digit Recognition — 28-Channel Column Scan, Parallel Contexts`);
		console.log(`  Buckets: ${this.config.buckets}`);
		console.log(`  Context length: ${this.options.contextLength}`);
		console.log(`  Merge threshold: ${this.options.mergeThreshold}`);
		console.log(`  Forget rate: ${this.options.patternForgetRate}`);
		console.log(`  Error mode: ${this.options.errorCorrectionMode ?? 'conservative (default)'}`);
		console.log(`  Training images cap: ${this.config.maxImages}`);
		console.log(`  Classify (train subset): ${this.config.maxTrainTest}`);
		console.log(`  Classify (held-out): ${this.config.skipTest ? 'skipped' : this.config.maxTestImages}`);
		console.log('');
	}

	async executeJob() {
		const N = this.trainImages.length;
		const FRAMES = 28; // 28 columns

		// ── Phase 1: parallel-context training ──────────────────────────────
		console.log(`Phase 1: parallel training of ${N} contexts (${FRAMES} frames each)...`);
		this.brain.initContexts(N);
		this.brain.setProcessingMode(true, false, true);
		const EMPTY = new Map();
		const t0 = Date.now();
		for (let col = 0; col < FRAMES; col++) {
			for (let i = 0; i < N; i++) {
				this.brain.setActiveContext(i);
				this.brain.processFrame(this.encoder.encodeColumn(this.trainBits[i], col), EMPTY, EMPTY);
			}
			if (this.isShuttingDown) return;
		}
		const trainMs = Date.now() - t0;
		const summary = this.brain.getFrameSummary();
		console.log(`  trained in ${trainMs}ms, ${summary.neuronCount} neurons, max L${summary.maxLevel}`);

		// ── Phase 2: per-context action wiring ──────────────────────────────
		console.log(`\nPhase 2: action wiring per context...`);
		const t1 = Date.now();
		let postLearnCorrect = 0;
		for (let i = 0; i < N; i++) {
			const label = this.trainLabels[i];
			this.brain.setActiveContext(i);
			this.encoder.resetActions();
			const actions = this.encoder.encodeAction(label);
			const rewards = this.encoder.buildRewards(label, 1.0, -1.0);
			const learnResult = this.brain.learn(actions, rewards);
			const predicted = this.encoder.predictBySumConsensus(learnResult.actionVoteStats);
			if (predicted === label) postLearnCorrect++;
			if (this.isShuttingDown) return;
		}
		const wireMs = Date.now() - t1;
		console.log(`  wired ${N} contexts in ${wireMs}ms`);
		console.log(`  Post-learn infer (sum consensus): ${(postLearnCorrect / N * 100).toFixed(2)}% (${postLearnCorrect}/${N})`);

		// ── Phase 3: classify (single-context interface on slot 0) ──────────
		this.brain.setProcessingMode(true, false, false);
		this.brain.setActiveContext(0);

		// Classify a subset of training images (verification of memorization)
		const numTrainTest = Math.min(this.config.maxTrainTest, N);
		if (numTrainTest > 0) {
			console.log(`\nPhase 3a: classify ${numTrainTest} training images (cold replay)...`);
			const r = this.classifySet(this.trainBits.slice(0, numTrainTest), this.trainLabels.slice(0, numTrainTest), 'train');
			this.trainResult = r;
		}

		if (!this.config.skipTest) {
			console.log(`\nPhase 3b: classify ${this.testImages.length} held-out test images...`);
			const r = this.classifySet(this.testBits, this.testLabels, 'held');
			this.testResult = r;
		}
	}

	classifySet(bitsArr, labels, tag) {
		const FRAMES = 28;
		const EMPTY = new Map();
		let correct = 0;
		const perDigit = new Array(10).fill(0);
		const perDigitTotal = new Array(10).fill(0);
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
		const t0 = Date.now();

		for (let i = 0; i < bitsArr.length; i++) {
			this.brain.setActiveContext(0);
			this.brain.resetContext();
			for (let col = 0; col < FRAMES; col++) {
				this.brain.processFrame(this.encoder.encodeColumn(bitsArr[i], col), EMPTY, EMPTY);
			}
			this.encoder.resetActions();
			const infer = this.brain.infer();
			const predicted = this.encoder.predictBySumConsensus(infer.actionVoteStats);
			const label = labels[i];
			const ok = predicted === label;
			if (ok) { correct++; perDigit[label]++; }
			perDigitTotal[label]++;
			if (predicted >= 0 && predicted < 10) confusion[label][predicted]++;
			if (this.isShuttingDown) return null;
		}

		const ms = Date.now() - t0;
		const accuracy = correct / bitsArr.length;
		const perDigitStr = perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');
		console.log(`  ${tag} labelAcc: ${(accuracy * 100).toFixed(2)}% (${correct}/${bitsArr.length}) in ${ms}ms`);
		console.log(`    per-digit: ${perDigitStr}`);
		return { accuracy, correct, total: bitsArr.length, perDigit, perDigitTotal, confusion };
	}

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
