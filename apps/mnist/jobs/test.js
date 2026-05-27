import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Job, runJob} from 'robot-brain';
import {MNISTEncoder} from '../encoder.js';
import {loadImages, loadLabels} from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * MNIST Test Job — single-channel binary pixel stream.
 *
 * Each image is fed pixel-by-pixel through one channel. After the bit stream
 * plays through, brain.learn(label, +reward) wires the votable context
 * neurons to the correct digit-action neuron. At test time the same image
 * is replayed and brain.infer() returns the action winner — the predicted
 * digit.
 *
 * Defaults match the memorization recipe found via the text-test sweep:
 *   --context-length 28     row width + 1
 *   --merge-threshold 0.99
 *   --forget-rate 0
 *   --error-mode conservative (Brain default)
 *   trim leading/trailing all-black pixel runs per image (--no-trim to opt out)
 *
 * MNIST-specific flags:
 *   --buckets N            quantization level (default 2 = binary)
 *   --max-images N         cap training set (default: all 60K)
 *   --max-test-images N    cap test set (default: all 10K)
 *   --episodes N           training passes (default 10)
 *   --skip-test            skip held-out evaluation
 *   --shuffle              randomize image order each pass (default: in-order)
 *   --no-trim              keep the full 784-pixel stream (including borders)
 *   --test-train           test against training images instead of held-out set
 *   --correct-reward F     reward for correct digit prediction (default 1)
 *   --incorrect-reward F   reward for incorrect digit prediction (default -1)
 */
export default class MNISTTestJob extends Job {

	constructor() {
		super();

		this.config = {
			buckets: 2,
			maxImages: 0,            // 0 = use all 60K training images
			maxTestImages: 0,        // 0 = use all 10K test images
			maxEpisodes: 10,
			skipTest: false,
			shuffle: false,
			trim: true,
			correctReward: 1,
			incorrectReward: -1,
			testTrain: false
		};

		this.encoder = null;

		// Raw MNIST data — loaded once in configureChannels(), reused across episodes.
		this.trainImages = null;
		this.trainLabels = null;
		this.testImages = null;
		this.testLabels = null;

		// Pre-encoded bit streams so we don't requantize/trim every episode.
		this.trainBits = null;
		this.testBits = null;

		this.episodeResults = [];
		this.testResult = null;
	}

	/**
	 * Parse MNIST-specific command-line flags. Brain-level flags
	 * (--context-length, --forget-rate, --merge-threshold, --error-mode, etc.)
	 * are parsed by parseBrainArgs() in the Job base class. Override the
	 * brain defaults here to the memorization recipe when the user didn't
	 * specify them on the command line.
	 */
	applyOptions() {
		const num = (flag) => {
			const i = process.argv.indexOf(flag);
			return i !== -1 && process.argv[i + 1] !== undefined ? parseInt(process.argv[i + 1]) : null;
		};
		const flt = (flag) => {
			const i = process.argv.indexOf(flag);
			return i !== -1 && process.argv[i + 1] !== undefined ? parseFloat(process.argv[i + 1]) : null;
		};

		const buckets = num('--buckets');
		if (buckets !== null) this.config.buckets = buckets;

		const maxImages = num('--max-images');
		if (maxImages !== null) this.config.maxImages = maxImages;

		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;

		const episodes = num('--episodes');
		if (episodes !== null) this.config.maxEpisodes = episodes;

		if (process.argv.includes('--skip-test')) this.config.skipTest = true;
		if (process.argv.includes('--shuffle')) this.config.shuffle = true;
		if (process.argv.includes('--no-trim')) this.config.trim = false;
		if (process.argv.includes('--test-train')) this.config.testTrain = true;

		const correctReward = flt('--correct-reward');
		if (correctReward !== null) this.config.correctReward = correctReward;

		const incorrectReward = flt('--incorrect-reward');
		if (incorrectReward !== null) this.config.incorrectReward = incorrectReward;

		// use
		if (this.options.contextLength == null) this.options.contextLength = 30;
	}

	/**
	 * Register the single pixel channel with the brain.
	 */
	async initialize() {
		this.encoder = new MNISTEncoder(this.config.buckets, this.config.trim);
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

		// Per-label class weights: 1 / count(label). Used as the reward passed
		// to brain.learn() so that the SUM of reward into each digit-action
		// neuron across the training set is equal — preventing a digit with
		// more training examples from structurally outvoting rarer digits at
		// consensus on shared voters (e.g. the 4↔1 overlap failure).
		this.labelCounts = new Array(10).fill(0);
		for (const lbl of this.trainLabels) this.labelCounts[lbl]++;
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
		console.log(`MNIST Digit Recognition — Single-Channel Binary Stream`);
		console.log(`  Buckets: ${this.config.buckets}`);
		console.log(`  Context length: ${this.options.contextLength}  (row-width = 28)`);
		console.log(`  Merge threshold: ${this.options.mergeThreshold}`);
		console.log(`  Forget rate: ${this.options.patternForgetRate}`);
		console.log(`  Error mode: ${this.options.errorCorrectionMode ?? 'conservative (default)'}`);
		console.log(`  Trim borders: ${this.config.trim}`);
		console.log(`  Episodes: ${this.config.maxEpisodes}`);
		console.log(`  Shuffle: ${this.config.shuffle}`);
		console.log(`  Reward: correct=${this.config.correctReward}, incorrect=${this.config.incorrectReward}`);
		console.log('');
	}

	async executeJob() {
		for (let ep = 1; ep <= this.config.maxEpisodes; ep++) {
			const result = await this.runTrainingEpisode(ep);
			this.episodeResults.push(result);
			if (this.isShuttingDown) return;

			// Run a test pass after each training episode (unless --skip-test)
			// so the per-episode progression is visible, not just a single
			// snapshot at the end.
			if (!this.config.skipTest) {
				const t = await this.runTest({ silent: true });
				result.testLblAcc = t.accuracy;
				result.testEvtAcc = t.evtAcc;
				result.testCorrect = t.correct;
				result.testTotal = t.total;
				this.testResult = t;
				console.log(`  → after ep ${ep}: test lblAcc=${(t.accuracy * 100).toFixed(2)}% (${t.correct}/${t.total}) evtAcc=${t.evtAcc !== null ? (t.evtAcc * 100).toFixed(2) + '%' : '—'}\n`);
				if (this.isShuttingDown) return;
			}
		}
	}

	/**
	 * One pass through the training set. Each image: reset, feed bits, label.
	 *
	 * @param {number} episode
	 * @returns {Promise<{ episode, total, duration, trainAcc }>}
	 */
	async runTrainingEpisode(episode) {
		const startTime = Date.now();
		const indices = this.buildIndices(this.trainImages.length);

		let correct = 0;
		let evtSum = 0;
		let evtCount = 0;

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = this.trainLabels[idx];
			const { prediction, eventAccuracy } = this.runTrainImage(this.trainBits[idx], label);
			if (prediction === label) correct++;
			if (eventAccuracy !== null) { evtSum += eventAccuracy; evtCount++; }

			const summary = this.brain.getFrameSummary();
			const mark = prediction === label ? '✓' : '✗';
			const predStr = prediction === -1 ? '—' : String(prediction);
			const evtStr = eventAccuracy !== null ? (eventAccuracy * 100).toFixed(1) + '%' : '—';
			console.log(`  ep${episode} train ${String(i + 1).padStart(4)}/${indices.length} #${String(idx).padStart(5)} lbl=${label}→${predStr} ${mark}  evtAcc=${evtStr} neurons=${summary.neuronCount} L${summary.maxLevel}`);

			await new Promise(resolve => setImmediate(resolve));
			if (this.isShuttingDown) break;
		}

		const summary = this.brain.getFrameSummary();
		const duration = Date.now() - startTime;
		const ips = (indices.length / (duration / 1000)).toFixed(1);
		const labelAcc = (correct / indices.length * 100).toFixed(2);
		const evtAcc = evtCount > 0 ? (evtSum / evtCount * 100).toFixed(2) : '—';

		console.log(`  Episode ${episode}: labelAcc(post-learn)=${labelAcc}% evtAcc(stream)=${evtAcc}% | neurons=${summary.neuronCount} L${summary.maxLevel} | ${ips} img/s ${duration}ms`);

		return { episode, total: indices.length, correct, duration, labelAcc, evtAcc };
	}

	/**
	 * Train one image: feed every pixel as an event-only frame, then call
	 * brain.learn() with the digit label and reward so the votable context
	 * neurons wire to the correct action neuron. The returned prediction
	 * is the brain's pre-learning guess (from the verify-infer inside learn).
	 *
	 * @param {Uint8Array} bits — pre-encoded bit stream for this image
	 * @param {number} label
	 * @returns {number} predicted digit
	 */
	runTrainImage(bits, label) {
		this.brain.resetContext();
		this.brain.resetAccuracyStats();
		this.encoder.resetActions();

		this.brain.setProcessingMode(true, false, true);
		for (let i = 0; i < bits.length; i++) {
			this.brain.processFrame(this.encoder.encodePixel(bits[i]), new Map(), new Map());
			if (this.isShuttingDown) return { prediction: -1, eventAccuracy: null };
		}

		// Event accuracy = how well the brain predicted each next pixel of the
		// bit stream as it played through. This is the meaningful "memorized
		// the sequence?" signal — the post-learn label prediction below is a
		// freshly-wired self-check and not a recall metric.
		const evtSummary = this.brain.getFrameSummary();
		const eventAccuracy = evtSummary.accuracyTotal > 0
			? evtSummary.accuracyCorrect / evtSummary.accuracyTotal
			: null;

		// Class-balanced reward: each digit-action neuron should receive the
		// same TOTAL reward summed across the training set. A digit seen N
		// times in training gets reward = correctReward / N per call.
		const weight = this.labelCounts[label] > 0
			? this.config.correctReward / this.labelCounts[label]
			: this.config.correctReward;
		const actions = this.encoder.encodeAction(label);
		const rewards = this.encoder.buildRewards(label, weight, this.config.incorrectReward);
		const { inferences } = this.brain.learn(actions, rewards);
		const { predicted } = this.encoder.applyInferences(inferences);

		// Track for next-iteration buildRewards (won't see the same image, but
		// keeps the encoder's lastAction state consistent).
		this.encoder.setForcedAction(label);
		return { prediction: predicted, eventAccuracy };
	}

	/**
	 * Run held-out (or training-set) evaluation: feed bits without learning,
	 * infer the digit, compare to label.
	 *
	 * @returns {Promise<{ accuracy, correct, total, perDigit, perDigitTotal, confusion }>}
	 */
	async runTest({ silent = false } = {}) {
		if (!silent) console.log('');
		const images = this.config.testTrain ? this.trainImages : this.testImages;
		const bitsArr = this.config.testTrain ? this.trainBits : this.testBits;
		const labels = this.config.testTrain ? this.trainLabels : this.testLabels;
		if (!silent && this.config.testTrain) console.log('  (testing against training images)');
		const indices = this.buildIndices(images.length);

		let correct = 0;
		let evtSum = 0;
		let evtCount = 0;
		const perDigit = new Array(10).fill(0);
		const perDigitTotal = new Array(10).fill(0);
		// confusion[actual][predicted] = count
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = labels[idx];
			const result = this.runTestImage(bitsArr[idx], label);

			if (result.predicted >= 0 && result.predicted < 10) {
				confusion[label][result.predicted]++;
			}
			if (result.correct) {
				correct++;
				perDigit[label]++;
			}
			perDigitTotal[label]++;
			if (result.eventAccuracy !== null) { evtSum += result.eventAccuracy; evtCount++; }

			if (!silent) {
				const mark = result.correct ? '✓' : '✗';
				const predStr = result.predicted === -1 ? '—' : String(result.predicted);
				const running = (correct / (i + 1) * 100).toFixed(1);
				const evtStr = result.eventAccuracy !== null ? (result.eventAccuracy * 100).toFixed(1) + '%' : '—';
				console.log(`  test ${String(i + 1).padStart(4)}/${indices.length} #${String(idx).padStart(5)} lbl=${label}→${predStr} ${mark}  lblAcc=${running}% evtAcc=${evtStr} neurons=${result.neuronCount}`);
			}

			await new Promise(resolve => setImmediate(resolve));
			if (this.isShuttingDown) break;
		}

		const accuracy = correct / indices.length;
		const evtAcc = evtCount > 0 ? evtSum / evtCount : null;
		const perDigitStr = perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');

		if (!silent) console.log(`  Test: lblAcc=${(accuracy * 100).toFixed(2)}% (${correct}/${indices.length}) evtAcc=${evtAcc !== null ? (evtAcc * 100).toFixed(2) + '%' : '—'} | ${perDigitStr}`);

		return { accuracy, evtAcc, correct, total: indices.length, perDigit, perDigitTotal, confusion };
	}

	/**
	 * Test one image: feed bits with learning off, then infer.
	 *
	 * @param {Uint8Array} bits
	 * @param {number} label
	 * @returns {{ predicted, correct, neuronCount }}
	 */
	runTestImage(bits, label) {
		this.brain.resetContext();
		this.brain.resetAccuracyStats();
		this.encoder.resetActions();

		this.brain.setProcessingMode(true, false, false);
		for (let i = 0; i < bits.length; i++) {
			this.brain.processFrame(this.encoder.encodePixel(bits[i]), new Map(), new Map());
			if (this.isShuttingDown) return { predicted: -1, correct: false, neuronCount: 0, eventAccuracy: null };
		}

		// Capture event accuracy from the bit stream replay before infer().
		// Same metric as training-side evtAcc — directly comparable.
		const evtSummary = this.brain.getFrameSummary();
		const eventAccuracy = evtSummary.accuracyTotal > 0
			? evtSummary.accuracyCorrect / evtSummary.accuracyTotal
			: null;

		const { inferences } = this.brain.infer();
		const { predicted } = this.encoder.applyInferences(inferences);
		const summary = this.brain.getFrameSummary();

		return { predicted, correct: predicted === label, neuronCount: summary.neuronCount, eventAccuracy };
	}

	/**
	 * Build a shuffled (or sequential) index array.
	 */
	buildIndices(count) {
		const indices = new Array(count);
		for (let i = 0; i < count; i++) indices[i] = i;
		if (this.config.shuffle) {
			for (let i = count - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				const tmp = indices[i];
				indices[i] = indices[j];
				indices[j] = tmp;
			}
		}
		return indices;
	}

	async showResults() {
		console.log('\nResults');
		console.log('='.repeat(70));

		for (const ep of this.episodeResults) {
			const testLbl = ep.testLblAcc !== undefined ? (ep.testLblAcc * 100).toFixed(2) + '%' : '—';
			const testEvt = ep.testEvtAcc !== null && ep.testEvtAcc !== undefined ? (ep.testEvtAcc * 100).toFixed(2) + '%' : '—';
			const testCnt = ep.testCorrect !== undefined ? `${ep.testCorrect}/${ep.testTotal}` : '—';
			console.log(`  Ep ${String(ep.episode).padStart(3)}: train labelAcc=${ep.labelAcc}% evtAcc=${ep.evtAcc}% | test lblAcc=${testLbl} (${testCnt}) evtAcc=${testEvt} | ${ep.duration}ms`);
		}

		if (this.testResult) {
			const perDigitStr = this.testResult.perDigit.map((c, d) =>
				this.testResult.perDigitTotal[d] > 0 ? `${d}:${(c / this.testResult.perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
			).join(' ');
			const evtStr = this.testResult.evtAcc !== null && this.testResult.evtAcc !== undefined
				? (this.testResult.evtAcc * 100).toFixed(2) + '%' : '—';
			console.log(`\n  Test: lblAcc=${(this.testResult.accuracy * 100).toFixed(2)}% (${this.testResult.correct}/${this.testResult.total})  evtAcc=${evtStr}`);
			console.log(`    per-digit: ${perDigitStr}`);

			console.log(`\n  Confusion matrix (rows = actual, cols = predicted):`);
			let header = '       ';
			for (let p = 0; p < 10; p++) header += String(p).padStart(5);
			console.log(header);
			for (let a = 0; a < 10; a++) {
				let row = `   ${a}   `;
				for (let p = 0; p < 10; p++) {
					row += String(this.testResult.confusion[a][p]).padStart(5);
				}
				console.log(row);
			}
		}

		console.log('='.repeat(70));
	}
}

await runJob(import.meta, MNISTTestJob);
