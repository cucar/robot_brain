import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EMPTY_REWARDS = new Map();

/**
 * Sensory-only MNIST app — the Naive Bayes baseline described in docs/mnist-merge.md.
 *
 * One channel per pixel position (retinotopic), all firing concurrently in a single frame per image.
 * Training: processFrame(image) populates sensory activations, then learn(actions, rewards, 1) wires every
 *   active sensory neuron to the labeled digit's action neuron at the same-frame voting slot.
 * Test: setLearning(false), processFrame(image), read the digit action winner from inferences.
 */
export default class MNISTTestJob extends Job {

	constructor() {
		super();
		this.config = {
			imageSize: 7, // by default, we do 7×7 binary run - small and fast
			buckets: 2,
			// Training is class-balanced by default: the same number of examples is used per digit.
			// The brain's per-voter normalization (K_{V,d} / Σ_d K_{V,d}) bakes the class prior into every voter's contribution.
			// Unbalanced training (natural MNIST has ~24% more 1s than 5s) leaks that prior tilt into every background voter and dominates the consensus.
			// Balanced training makes the prior uniform so the leak evaluates to 0.1 per class per voter — neutral — and lifts test accuracy from ~38% to ~76% at 28×28 binary.
			// perClass = 0 means "use the smallest class count available" (5421 with full MNIST); pass --per-class N to cap explicitly for faster iteration.
			perClass: 0,
			maxTestImages: 0,
			maxEpisodes: 10,
			skipTest: false,
			// split: split-MNIST mode — emit the balanced training set in digit order and train one episode per digit class.
			// Each episode sees only its own digit's samples; the final held-out test then reveals catastrophic forgetting.
			split: false,
		};
		this.encoder = null;
		this.trainImages = null;
		this.trainLabels = null;
		this.trainBits = null;
		this.testImages = null;
		this.testLabels = null;
		this.testBits = null;
		this.episodeResults = [];
		this.testResult = null;
	}

	/**
	 * Parse MNIST-specific flags and stamp NB-appropriate brain defaults.
	 * Context length 1 (single-frame episodes) and forget rate 0 (per-pixel-per-digit counts accumulate cleanly).
	 */
	applyOptions() {
		const num = (flag) => {
			const i = process.argv.indexOf(flag);
			return i !== -1 && process.argv[i + 1] !== undefined ? parseInt(process.argv[i + 1]) : null;
		};

		const imageSize = num('--image-size');
		if (imageSize !== null) this.config.imageSize = imageSize;
		const buckets = num('--buckets');
		if (buckets !== null) this.config.buckets = buckets;
		const perClass = num('--per-class');
		if (perClass !== null) this.config.perClass = perClass;
		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;
		const episodes = num('--episodes');
		if (episodes !== null) this.config.maxEpisodes = episodes;
		if (process.argv.includes('--skip-test')) this.config.skipTest = true;
		if (process.argv.includes('--split')) this.config.split = true;

		if (this.options.contextLength == null) this.options.contextLength = 1;
		if (this.options.patternForgetRate == null) this.options.patternForgetRate = 0;
	}

	/**
	 * Construct the pixel-channels encoder and register one channel per pixel position with the brain.
	 */
	async initialize() {
		this.encoder = new MNISTPixelChannelsEncoder(this.config.buckets, this.config.imageSize);
		this.encoder.registerChannels(this.brain);
	}

	/**
	 * Load MNIST train/test IDX files, build the class-balanced training subset, and pre-quantize every image into bits.
	 * Pre-quantization moves all encoding cost out of the training loop so per-frame work is just channel activation.
	 */
	async configureChannels() {

		// get the test and training files
		const dataDir = path.join(__dirname, '..', 'data');
		const trainImgPath = this.findDataFile(dataDir, 'train-images-idx3-ubyte');
		const trainLblPath = this.findDataFile(dataDir, 'train-labels-idx1-ubyte');
		const testImgPath = this.findDataFile(dataDir, 't10k-images-idx3-ubyte');
		const testLblPath = this.findDataFile(dataDir, 't10k-labels-idx1-ubyte');

		// load the test and training images
		const trainImages = loadImages(trainImgPath);
		const trainLabels = loadLabels(trainLblPath);
		this.testImages = loadImages(testImgPath);
		this.testLabels = loadLabels(testLblPath);

		// Class-balanced selection: walk the training set once in order, keep an image for each digit until that digit's per-class quota is filled.
		const { images: balancedImages, labels: balancedLabels } = this.selectClassBalanced(trainImages, trainLabels);
		this.trainImages = balancedImages;
		this.trainLabels = balancedLabels;

		// limit number of test images if requested
		if (this.config.maxTestImages > 0) {
			this.testImages = this.testImages.slice(0, this.config.maxTestImages);
			this.testLabels = this.testLabels.slice(0, this.config.maxTestImages);
		}

		// Pre-quantize so the training loop pays no encoding cost per frame if there are multiple episodes
		this.trainBits = this.trainImages.map(px => this.encoder.buildBits(px));
		this.testBits = this.testImages.map(px => this.encoder.buildBits(px));
	}

	/**
	 * Class-balanced selection: walk the training set once in order, keep an image for each digit until that digit's per-class quota is filled.
	 * perClass=0 means "use whatever the smallest class has available", which makes the balanced quota the natural floor without forcing the operator to know it.
	 * Returns {images, labels, cap} — picked indices are sorted ascending so the training order matches natural-MNIST order within the balanced subset, keeping the run deterministic and reproducible.
	 */
	selectClassBalanced(images, labels) {

		// Count how many examples each digit has, so we know the natural floor when perClass is left at 0.
		const counts = new Array(10).fill(0);
		for (const label of labels) counts[label]++;

		// Resolve the per-class quota: explicit override if set, otherwise the smallest class's count.
		const cap = this.config.perClass > 0 ? this.config.perClass : Math.min(...counts);

		// Walk the dataset once in order, taking the first `cap` examples of each digit.
		const indices = [];
		const picked = new Array(10).fill(0);
		for (let i = 0; i < images.length && indices.length < cap * 10; i++) {
			const digit = labels[i];
			if (picked[digit] < cap) { indices.push(i); picked[digit]++; }
		}

		// Order the indices. Split mode groups by digit (all 0s, then all 1s, …) for split-MNIST tests;
		// otherwise restore natural-MNIST ordering within the balanced subset for determinism and reproducibility.
		if (this.config.split) indices.sort((a, b) => labels[a] - labels[b] || a - b);
		else indices.sort((a, b) => a - b);

		// return the picked images and labels
		console.log(`  Balanced training set built: ${cap} per class × 10 = ${indices.length} total`);
		return { images: indices.map(i => images[i]), labels: indices.map(i => labels[i]), cap };
	}

	/**
	 * Locate a data file, trying the plain IDX path first and falling back to its .gz variant.
	 */
	findDataFile(dataDir, baseName) {
		const plain = path.join(dataDir, baseName);
		if (fs.existsSync(plain)) return plain;
		const gz = path.join(dataDir, `${baseName}.gz`);
		if (fs.existsSync(gz)) return gz;
		throw new Error(`MNIST data not found: ${plain} (run jobs/download.js first)`);
	}

	/**
	 * Print the resolved configuration before training begins so the run is self-documenting in the log.
	 */
	async showStartupInfo() {
		const phaseLabel = this.config.buckets === 2 ? 'A — binary'
			: this.config.buckets <= 4 ? 'B' : 'C';
		console.log('MNIST — sensory-only (Naive Bayes) baseline');
		console.log(`  Image size: ${this.config.imageSize}×${this.config.imageSize} (${this.config.imageSize * this.config.imageSize} channels)`);
		console.log(`  Buckets: ${this.config.buckets} (Phase ${phaseLabel})`);
		console.log(`  Context length: ${this.options.contextLength}`);
		console.log(`  Forget rate: ${this.options.patternForgetRate}`);
		console.log(`  Episodes: ${this.config.maxEpisodes}${this.config.split ? ' per digit task (split-MNIST)' : ''}`);
		console.log(`  Training: balanced, ${this.config.perClass > 0 ? this.config.perClass : 'auto'} per class`);
		console.log(`  Test images: ${this.config.skipTest ? 'skipped' : (this.config.maxTestImages || 'all')}`);
		console.log('');
	}

	/**
	 * Train, then evaluate once on the held-out set in non-learning mode.
	 */
	async executeJob() {

		// run the training first
		if (this.config.split) this.runSplitTraining();
		else this.runJointTraining();

		// now, run the tests evaluation
		if (!this.config.skipTest) this.runEvaluation();
	}

	/**
	 * Joint mode: train for maxEpisodes passes over the full balanced set.
	 */
	runJointTraining() {
		for (let ep = 1; ep <= this.config.maxEpisodes; ep++) {
			const result = this.runTraining(ep);
			this.episodeResults.push(result);
			if (this.isShuttingDown) return;
		}
	}

	/**
	 * Split mode: for each digit 0..9, train maxEpisodes passes over just that digit's samples before advancing.
	 * The brain keeps its learned state across tasks — the final held-out test reveals catastrophic forgetting (or its absence).
	 */
	runSplitTraining() {
		for (let digit = 0; digit < 10; digit++) {
			for (let ep = 1; ep <= this.config.maxEpisodes; ep++) {
				const result = this.runTraining(ep, digit);
				this.episodeResults.push(result);
				if (this.isShuttingDown) return;
			}
		}
	}

	/**
	 * Switch the brain into read-only inference and run the held-out test pass.
	 * Pattern activation + voting still run so inferences populate, but no decay, no error-correction neurons, no event-event strengthening.
	 */
	runEvaluation() {
		this.brain.setLearning(false);
		console.log('');
		this.testResult = this.runTest();
	}

	/**
	 * One pass through the training set. For each image:
	 *   resetContext → processFrame → record prediction (= training accuracy, pre-update) → learn(actions, rewards, 1).
	 * The training-accuracy number is the brain's prediction *before* the supervised wire lands.
	 * `digit` is optional: pass it in split mode to restrict the pass to that digit's slice; omit (or null) to walk the full balanced set.
	 */
	runTraining(episode, digit = null) {

		// Track wall-clock so we can report img/s for this pass.
		const startTime = Date.now();

		// Choose which training samples this pass walks: a single digit's slice in split mode, or the whole set otherwise.
		const indices = this.buildTrainIndices(digit);

		// Reused-per-image reward vector — built once outside the loop because every call uses the same +1 reward.
		const positiveReward = this.buildPositiveReward();

		// Accumulators for the per-pass training accuracy line: overall correct, per-digit correct, per-digit totals.
		const tally = this.newTally();

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = this.trainLabels[idx];

			// Predict *before* learning so the recorded accuracy reflects what the brain knew going in.
			const predicted = this.predictImage(this.trainBits[idx]);
			this.recordPrediction(tally, label, predicted);

			// Supervised wire: bind every active sensory neuron to the label's action neuron at the same-frame voting slot.
			this.brain.learn(this.encoder.encodeAction(label), positiveReward, 1);

			// Honor an in-flight shutdown without leaving the brain in a half-written state.
			if (this.isShuttingDown) break;
		}

		// Roll the tally up into a result, attach metadata (which task/episode, wall-clock), log a one-liner, and return.
		const duration = Date.now() - startTime;
		const result = { digit, episode, ...this.summarizeTally(tally), duration };
		this.logTrainingPass(result);
		return result;
	}

	/**
	 * Reward is +1 on every learn() call, and this is load-bearing for getting count-based voting out of the brain.
	 * The brain stores connection.reward as a running mean (alpha=1/strength), so reward stays at exactly 1.0 forever.
	 * That collapses every candidate's per-candidate reward score to 1.0, tying the brain's reward-based winner across all digits.
	 * The tie-break then falls through to argmax(candidate.strength), which is the per-voter posterior sum Σ_V K_{V,d}/total_V we want.
	 * A non-constant reward would re-enable the reward path and break this — a hard contract with the brain consensus, not a free parameter.
	 */
	buildPositiveReward() {
		return this.encoder.buildRewards(1);
	}

	/**
	 * Run one image through the brain and decode the predicted digit.
	 * Shared by training (pre-update prediction) and held-out test.
	 */
	predictImage(bits) {
		this.brain.resetContext();
		const inputs = this.encoder.encodeImage(bits);
		this.brain.processFrame(inputs, EMPTY_REWARDS);
		const inferResult = this.brain.infer();
		return this.encoder.decodeDigit(inferResult.inferences);
	}

	/**
	 * Fresh per-pass tally: aggregate correct, per-digit correct, per-digit totals.
	 */
	newTally() {
		return {
			correct: 0,
			perDigit: new Array(10).fill(0),
			perDigitTotal: new Array(10).fill(0),
		};
	}

	/**
	 * Update the tally with one (label, predicted) outcome.
	 */
	recordPrediction(tally, label, predicted) {
		if (predicted === label) { tally.correct++; tally.perDigit[label]++; }
		tally.perDigitTotal[label]++;
	}

	/**
	 * Collapse a tally into the result shape used by reports — accuracy, totals, and the per-digit arrays.
	 */
	summarizeTally(tally) {
		const total = tally.perDigitTotal.reduce((a, b) => a + b, 0);
		return {
			accuracy: total > 0 ? tally.correct / total : 0,
			correct: tally.correct,
			total,
			perDigit: tally.perDigit,
			perDigitTotal: tally.perDigitTotal,
		};
	}

	/**
	 * Print the one-line per-pass training log, picking the joint vs split label form.
	 */
	logTrainingPass({ digit, episode, accuracy, correct, total, duration, perDigit, perDigitTotal }) {
		const ips = (total / (duration / 1000)).toFixed(0);
		const perDigitStr = this.formatPerDigit(perDigit, perDigitTotal);
		const epLabel = this.config.split
			? `Task digit ${digit} — episode ${episode}/${this.config.maxEpisodes}`
			: `Episode ${episode}/${this.config.maxEpisodes}`;
		console.log(`  ${epLabel}: train=${(accuracy * 100).toFixed(2)}% (${correct}/${total}) | ${ips} img/s ${duration}ms | ${perDigitStr}`);
	}

	/**
	 * Held-out evaluation. setLearning(false) is already in effect.
	 * For each image: resetContext → processFrame → infer → read prediction. No learn() call.
	 * processFrame's vote generator suppresses age depth-1 (the only available age at context_length=1),
	 * so we read the prediction off brain.infer() which runs the same vote sweep without that guard.
	 * Accumulates aggregate accuracy, per-digit accuracy, and the 10×10 confusion matrix called out in the spec.
	 */
	runTest() {
		const startTime = Date.now();
		const tally = this.newTally();
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));

		for (let i = 0; i < this.testBits.length; i++) {
			const label = this.testLabels[i];
			const predicted = this.predictImage(this.testBits[i]);
			this.recordPrediction(tally, label, predicted);
			// decodeDigit() returns -1 when no action inference is present — keep those out of the confusion matrix.
			if (predicted >= 0 && predicted < 10) confusion[label][predicted]++;
			if (this.isShuttingDown) break;
		}

		const duration = Date.now() - startTime;
		const summary = this.summarizeTally(tally);
		const perDigitStr = this.formatPerDigit(summary.perDigit, summary.perDigitTotal);
		console.log(`  Test: ${(summary.accuracy * 100).toFixed(2)}% (${summary.correct}/${summary.total}) ${duration}ms | ${perDigitStr}`);

		return { ...summary, confusion };
	}

	/**
	 * Format the per-digit accuracy line, e.g. "0:85% 1:92% 2:71% ..."
	 */
	formatPerDigit(perDigit, perDigitTotal) {
		return perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');
	}

	/**
	 * Pick the training indices for a pass.
	 * Joint mode (digit=null): sequential walk over the full balanced training set.
	 * Split mode (digit=0..9): only the slice for that digit. The balanced set is sorted by digit
	 * with exactly `cap = N/10` samples per class, so digit d's slice is [d*cap, (d+1)*cap).
	 */
	buildTrainIndices(digit) {
		const N = this.trainBits.length;
		if (digit == null) return Array.from({ length: N }, (_, i) => i);
		const cap = N / 10;
		const start = digit * cap;
		return Array.from({ length: cap }, (_, i) => start + i);
	}

	/**
	 * Final summary: per-episode training accuracy, training trend, test accuracy, and the confusion matrix.
	 * The confusion matrix is the load-bearing artifact for the NB-band check — the 3/8/9 collapses are what motivate the spatial-processing workstream.
	 */
	async showResults() {
		console.log('\nResults');
		console.log('='.repeat(70));

		for (const ep of this.episodeResults) {
			const label = ep.digit != null
				? `Task digit ${ep.digit} ep ${ep.episode}`
				: `Episode ${ep.episode}`;
			console.log(`  ${label}: train=${(ep.accuracy * 100).toFixed(2)}% (${ep.duration}ms)`);
		}

		// Joint-mode-only: first vs last training accuracy is a meaningful learning curve.
		// In split mode, each entry measures pre-update accuracy on a different digit, so the delta is apples-to-oranges and would mislead — skip it.
		if (!this.config.split && this.episodeResults.length >= 2) {
			const first = this.episodeResults[0];
			const last = this.episodeResults[this.episodeResults.length - 1];
			const delta = ((last.accuracy - first.accuracy) * 100).toFixed(2);
			console.log(`\n  Training: ${(first.accuracy * 100).toFixed(2)}% → ${(last.accuracy * 100).toFixed(2)}% (${delta >= 0 ? '+' : ''}${delta}pp)`);
		}

		if (this.testResult) {
			console.log(`  Test:     ${(this.testResult.accuracy * 100).toFixed(2)}% (${this.testResult.correct}/${this.testResult.total})`);
			console.log('\n  Confusion (rows = actual, cols = predicted):');
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
