import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Job, runJob} from 'robot-brain';
import {MNISTEncoder} from '../encoder.js';
import {loadImages, loadLabels} from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * MNIST Test Job — trains and evaluates Robot Brain on MNIST digit recognition.
 *
 * Architecture:
 * 784 parallel channels (one per pixel in the 28×28 grid), each with a quantized input dimension and a 10-way digit action dimension.
 * Images are presented as multi-frame episodes:
 * frame 1 populates the "retina"
 * subsequent frames enable cross-channel connection formation via temporal co-activation
 * final frame delivers the reward signal for the last informed prediction.
 *
 * Each image is a self-contained episode. resetContext() between images clears active neurons while preserving all learned connections.
 * The brain discovers spatial structure through co-activation statistics across channels, not through any geometric prior.
 *
 * MNIST-specific flags:
 *   --frames N             image presentations per episode (default 2, +1 reward frame)
 *   --buckets N            quantization level: 2=binary, 4, 8, 16 (default 2)
 *   --max-images N         cap training set for fast iteration (default: all 60K)
 *   --max-test-images N    cap test set (default: all 10K)
 *   --episodes N           training passes over the dataset (default 10)
 *   --skip-test            skip the test evaluation after training
 *   --no-shuffle           present images in original order
 *   --correct-reward F     reward for correct digit prediction (default 1)
 *   --incorrect-reward F   reward for incorrect prediction (default -1)
 */
export default class MNISTTestJob extends Job {

	constructor() {
		super();

		this.config = {
			framesPerImage: 2,       // image presentations per episode (+ 1 implicit reward frame)
			buckets: 2,              // quantization level — Phase A = 2 (binary)
			maxImages: 0,            // 0 = use all 60K training images
			maxTestImages: 0,        // 0 = use all 10K test images
			maxEpisodes: 10,         // training passes over the full dataset
			skipTest: false,         // skip the post-training test evaluation
			shuffle: true,           // randomize image order each pass
			correctReward: 1,        // reward delivered for correct digit prediction
			incorrectReward: -1      // reward delivered for incorrect digit prediction
		};

		this.encoder = null;

		// Raw MNIST data — loaded once in configureChannels(), reused across episodes.
		this.trainImages = null;
		this.trainLabels = null;
		this.testImages = null;
		this.testLabels = null;

		// Per-episode training accuracy for the final summary.
		this.episodeResults = [];

		// Post-training test result — populated by runTestPass() if not skipped.
		this.testResult = null;
	}

	/**
	 * Parse MNIST-specific command-line flags. Brain-level flags (--context-length,
	 * --forget-rate, --regions, --columns, etc.) are handled by parseBrainArgs()
	 * in the Job base class and passed to the Brain constructor.
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

		const frames = num('--frames');
		if (frames !== null) this.config.framesPerImage = frames;

		const buckets = num('--buckets');
		if (buckets !== null) this.config.buckets = buckets;

		const maxImages = num('--max-images');
		if (maxImages !== null) this.config.maxImages = maxImages;

		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;

		const episodes = num('--episodes');
		if (episodes !== null) this.config.maxEpisodes = episodes;

		if (process.argv.includes('--skip-test')) this.config.skipTest = true;
		if (process.argv.includes('--no-shuffle')) this.config.shuffle = false;

		const correctReward = flt('--correct-reward');
		if (correctReward !== null) this.config.correctReward = correctReward;

		const incorrectReward = flt('--incorrect-reward');
		if (incorrectReward !== null) this.config.incorrectReward = incorrectReward;
	}

	/**
	 * Register 784 pixel channels with the brain. Each pixel position becomes
	 * an independent channel with an input dim (quantized brightness) and an
	 * action dim (digit 0–9). Channel registration allocates all base neurons
	 * upfront — at binary quantization this is 9,408 neurons.
	 */
	async initialize() {
		this.encoder = new MNISTEncoder(this.config.buckets);
		this.encoder.registerChannels(this.brain);
	}

	/**
	 * Load MNIST data from the IDX files in apps/mnist/data/. Accepts both plain and .gz variants — the loader auto-detects gzip compression.
	 * Optionally truncates the datasets per --max-images / --max-test-images for faster iteration during development.
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
	}

	/**
	 * Locate a data file, trying both uncompressed and .gz variants.
	 * Throws with a clear "run download.js first" message if neither exists.
	 *
	 * @param {string} dataDir — directory to search in
	 * @param {string} baseName — filename without extension (e.g. 'train-images-idx3-ubyte')
	 * @returns {string} resolved path to the file
	 */
	findDataFile(dataDir, baseName) {
		const plain = path.join(dataDir, baseName);
		if (fs.existsSync(plain)) return plain;
		const gz = path.join(dataDir, `${baseName}.gz`);
		if (fs.existsSync(gz)) return gz;
		throw new Error(`MNIST data not found: ${plain} (run download.js first)`);
	}

	/**
	 * Print startup configuration so the operator can verify settings before
	 * a potentially long training run. Includes the quantization phase label
	 * (A/B/C) and the total frames per image including the reward frame.
	 */
	async showStartupInfo() {
		const phaseLabel = this.config.buckets === 2 ? 'A — binary'
			: this.config.buckets <= 4 ? 'B' : 'C';

		console.log(`MNIST Digit Recognition`);
		console.log(`  Buckets: ${this.config.buckets} (Phase ${phaseLabel})`);
		console.log(`  Frames per image: ${this.config.framesPerImage} (+1 reward frame = ${this.config.framesPerImage + 1} total)`);
		console.log(`  Episodes: ${this.config.maxEpisodes}`);
		console.log(`  Shuffle: ${this.config.shuffle}`);
		console.log(`  Reward: correct=${this.config.correctReward}, incorrect=${this.config.incorrectReward}`);
		console.log('');
	}

	/**
	 * Train for maxEpisodes passes over the training set, then run a single
	 * test evaluation on the held-out set. Training accuracy is reported per
	 * episode so the operator sees improvement during the run; the test pass
	 * runs once at the end to measure generalization.
	 */
	async executeJob() {
		// Training: one pass through the training set per episode.
		for (let ep = 1; ep <= this.config.maxEpisodes; ep++) {
			const result = await this.runTrainingPass(ep);
			this.episodeResults.push(result);

			if (this.isShuttingDown) return;
		}

		// Test: single evaluation on the held-out set after all training.
		if (!this.config.skipTest) this.testResult = await this.runTestPass();
	}

	/**
	 * One pass through the training set. Each image is a self-contained
	 * episode: reset context, present the image for N frames, deliver the
	 * reward, record whether the aggregate prediction was correct, move on.
	 *
	 * Prints a progress line every 1000 images with running accuracy, and
	 * a final summary line with overall accuracy, per-digit breakdown, and
	 * throughput (images/second).
	 *
	 * @param {number} episode — 1-based episode number (for display)
	 * @returns {{ episode, accuracy, correct, total, duration, perDigit, perDigitTotal }}
	 */
	async runTrainingPass(episode) {
		const startTime = Date.now();
		const numImages = this.trainImages.length;
		const indices = this.buildIndices(numImages);

		let correct = 0;
		const perDigit = new Array(10).fill(0);
		const perDigitTotal = new Array(10).fill(0);

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const result = this.runImage(this.trainImages[idx], this.trainLabels[idx], idx);
			if (result.correct) {
				correct++;
				perDigit[this.trainLabels[idx]]++;
			}
			perDigitTotal[this.trainLabels[idx]]++;

			// show progress every 1000 images
			// if ((i + 1) % 1000 === 0)
			console.log(`\r  Episode ${episode}/${this.config.maxEpisodes} — ${i + 1}/${indices.length} images, accuracy: ${(correct / (i + 1) * 100).toFixed(1)}%`);

			if (this.isShuttingDown) break;
		}

		const duration = Date.now() - startTime;
		const accuracy = correct / indices.length;
		const ips = (indices.length / (duration / 1000)).toFixed(0);

		// Clear the progress line before printing the final summary.
		if (process.stdout.isTTY) { process.stdout.write('\r'); process.stdout.clearLine(0); }
		else process.stdout.write('\n');

		// Per-digit accuracy breakdown: "0:85% 1:92% 2:71% ..."
		const perDigitStr = perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');

		console.log(`  Episode ${episode}: train=${(accuracy * 100).toFixed(2)}% (${correct}/${indices.length}) | ${ips} img/s ${duration}ms | ${perDigitStr}`);

		return { episode, accuracy, correct, total: indices.length, duration, perDigit, perDigitTotal };
	}

	/**
	 * Evaluate on the held-out test set after all training is complete. Uses
	 * the same frame structure as training — the brain continues learning
	 * during test (no freeze mode), but accuracy is measured on each image's
	 * FIRST prediction (before reward delivery), matching the roadmap's
	 * "first exposure" criterion.
	 *
	 * Images are shuffled so evaluation order doesn't systematically advantage
	 * later images that benefit from earlier test-set learning.
	 *
	 * @returns {{ accuracy, correct, total, perDigit, perDigitTotal }}
	 */
	async runTestPass() {
		console.log('');
		const numImages = this.testImages.length;
		const indices = this.buildIndices(numImages);

		let correct = 0;
		const perDigit = new Array(10).fill(0);
		const perDigitTotal = new Array(10).fill(0);

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const result = this.runImage(this.testImages[idx], this.testLabels[idx], idx);
			if (result.correct) {
				correct++;
				perDigit[this.testLabels[idx]]++;
			}
			perDigitTotal[this.testLabels[idx]]++;

			if (this.isShuttingDown) break;
		}

		const accuracy = correct / indices.length;

		// Per-digit accuracy breakdown for test results.
		const perDigitStr = perDigit.map((c, d) =>
			perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
		).join(' ');

		console.log(`  Test: ${(accuracy * 100).toFixed(2)}% (${correct}/${indices.length}) | ${perDigitStr}`);

		return { accuracy, correct, total: indices.length, perDigit, perDigitTotal };
	}

	/**
	 * Present one image to the brain as a multi-frame episode. This is the
	 * core method that implements the two-frame (or N-frame) episode structure
	 * described in the roadmap.
	 *
	 * Frame structure (framesPerImage=2 example):
	 *   f=1: present pixels, no reward, no prior context
	 *        → brain populates the retina and creates connections between co-active channels
	 *   f=2: present pixels, reward for f=1's prediction
	 *        → cross-channel context now available; first informed prediction
	 *   f=3: present pixels, reward for f=2's prediction
	 *        → brain learns from the reward; prediction captured for accuracy
	 *
	 * The inputs map is built once and reused across all frames — the image is
	 * identical each time; only the brain's internal context changes.
	 *
	 * @param {Uint8Array} pixels — 784 raw pixel values for one image
	 * @param {number} label — correct digit (0–9)
	 * @param {number} imageIndex — image id
	 * @returns {{ predicted: number, correct: boolean, confidence: number }}
	 */
	runImage(pixels, label, imageIndex) {
		this.brain.resetContext();
		this.encoder.resetActions();

		const verbose = this.diagnostic;
		if (verbose) console.log(`  Image #${imageIndex} label=${label}`);

		// Build inputs once — reused across all frames of this image.
		const inputs = this.encoder.encodeImage(pixels);
		let prediction = null;

		// framesPerImage image presentations + 1 reward delivery frame.
		const totalFrames = this.config.framesPerImage + 1;
		for (let f = 1; f <= totalFrames; f++) {

			// first frame has no prior action to reward. subsequent frames reward the previous frame's per-channel digit predictions.
			const rewards = f === 1
				? new Map()
				: this.encoder.buildRewards(label, this.config.correctReward, this.config.incorrectReward);

			const { inferences } = this.brain.processFrame(inputs, rewards);
			const result = this.encoder.applyInferences(inferences);

			if (verbose) {
				const mark = result.predicted === label ? '✓' : '✗';
				const tag = f > this.config.framesPerImage ? ' (reward)' : '';

				// Per-digit breakdown: votes (how many channels picked it) and
				// score (total strength behind those votes).
				const breakdown = Array.from(result.scores).map((s, d) =>
					result.voteCounts[d] > 0 ? `${d}:${result.voteCounts[d]}ch/${s.toFixed(1)}` : null
				).filter(Boolean).join('  ');

				console.log(`    f${f}: consensus=${result.predicted} ${mark}${tag}  [${breakdown}]`);
			}

			// Capture the prediction from the final frame (after reward delivery).
			// The reward for the informed prediction (frame framesPerImage) lands
			// on this frame, so the brain's inference here reflects what it learned.
			if (f === totalFrames) prediction = result;
		}

		return {
			predicted: prediction.predicted,
			correct: prediction.predicted === label,
			confidence: prediction.confidence
		};
	}

	/**
	 * Build a shuffled (or sequential) index array for iterating through a
	 * dataset. Fisher-Yates shuffle ensures uniform random permutation.
	 *
	 * @param {number} count — number of items (images) to index
	 * @returns {number[]} — indices 0..count-1, optionally shuffled
	 */
	buildIndices(count) {
		const indices = new Array(count);
		for (let i = 0; i < count; i++) indices[i] = i;
		if (this.config.shuffle) {
			// Fisher-Yates shuffle — O(n), uniform distribution.
			for (let i = count - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				const tmp = indices[i];
				indices[i] = indices[j];
				indices[j] = tmp;
			}
		}
		return indices;
	}

	/**
	 * Print a final summary table after all training and testing is complete.
	 * Shows per-episode training accuracy, the overall accuracy trend, and
	 * the final test result if one was run.
	 */
	async showResults() {
		console.log('\nResults');
		console.log('='.repeat(70));

		// Per-episode training accuracy.
		for (const ep of this.episodeResults) {
			console.log(`  Episode ${ep.episode}: train=${(ep.accuracy * 100).toFixed(2)}% (${ep.duration}ms)`);
		}

		// Training accuracy trend from first to last episode.
		if (this.episodeResults.length >= 2) {
			const first = this.episodeResults[0];
			const last = this.episodeResults[this.episodeResults.length - 1];
			const delta = ((last.accuracy - first.accuracy) * 100).toFixed(2);
			console.log(`\n  Training: ${(first.accuracy * 100).toFixed(2)}% -> ${(last.accuracy * 100).toFixed(2)}% (${delta >= 0 ? '+' : ''}${delta}pp)`);
		}

		// Test accuracy (single evaluation after training).
		if (this.testResult) {
			console.log(`  Test: ${(this.testResult.accuracy * 100).toFixed(2)}%`);
		}

		console.log('='.repeat(70));
	}
}

await runJob(import.meta, MNISTTestJob);
