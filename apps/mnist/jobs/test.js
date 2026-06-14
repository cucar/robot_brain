import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EMPTY_REWARDS = new Map();
const PROGRESS_EVERY = 100;

/**
 * Sensory-only MNIST app — the Naive Bayes baseline described in docs/mnist-merge.md.
 *
 * One channel per pixel position (retinotopic), all firing concurrently in a single frame per image.
 * Training: processFrame(image) populates sensory activations, then learn(actions, 1) wires every active
 *   sensory neuron to every digit action neuron with reward=1 on the correct digit and reward=0 on the rest.
 *   The smoothed-reward update on each connection converges to conn.reward(V,d) = P(d|V) — the per-voter posterior.
 * Test: setLearning(false), processFrame(image), read the digit action winner from inferences.
 */
export default class MNISTTestJob extends Job {

	constructor() {
		super();
		this.config = {
			imageSize: 7, // by default, we do 7×7 binary run - small and fast
			buckets: 2,
			// radius: spatial neighborhood radius declared per pixel — 1 = 3×3 (8 neighbors),
			// 2 = 5×5 (24), 3 = 7×7 (48). Larger = bigger L1 receptive field (more structural context)
			// but more specific patterns, more brittle matching, and far more sample-hungry.
			radius: 1,
			// Training is class-balanced by default: the same number of examples is used per digit.
			// The brain's per-voter normalization (K_{V,d} / Σ_d K_{V,d}) bakes the class prior into every voter's contribution.
			// Unbalanced training (natural MNIST has ~24% more 1s than 5s) leaks that prior tilt into every background voter and dominates the consensus.
			// Balanced training makes the prior uniform so the leak evaluates to 0.1 per class per voter — neutral — and lifts test accuracy from ~38% to ~76% at 28×28 binary.
			// perClass = 0 means "use the smallest class count available" (5421 with full MNIST); pass --per-class N to cap explicitly for faster iteration.
			perClass: 0,
			maxTestImages: 0,
			maxEpisodes: 1,
			skipTest: false,
			// split: split-MNIST mode — emit the balanced training set in digit order and train one episode per digit class.
			// Each episode sees only its own digit's samples; the final held-out test then reveals catastrophic forgetting.
			split: false,
			// noBalance: skip class-balanced selection and train on the full natural MNIST set (unbalanced).
			noBalance: false,
			// consensus: how the brain combines per-voter action posteriors into a digit winner. Passed
			// through to the brain (Rust) as a brain option — the decode now happens brain-side so votes
			// no longer cross the NAPI boundary each frame.
			//   'democratic' — strength-weighted ARITHMETIC mean of per-voter posteriors P(d|voter),
			//     argmax over digits (the brain's original consensus).
			//   'nb' — Naive-Bayes-style PRODUCT of per-voter posteriors: argmax_d Σ_voter log(P(d|voter)+eps).
			//     Sharper than the mean: a voter that confidently rules a digit out (P≈0 → log≈−large) heavily
			//     penalizes it, instead of being diluted by averaging. MNIST default.
			consensus: 'nb',
			// nbEps: Laplace-style floor added before the log so P(d|voter)=0 doesn't send a digit to −Infinity.
			// Passed through to the brain; only consulted under 'nb' consensus.
			nbEps: 1e-3,
			// errorCorrectRounds: N>0 → after normal training, run N discriminative passes over the
			// training set that reinforce (smoothed-reward learn()) only on mispredictions, with minting
			// off. Pushes training accuracy toward ~100%; test effect is the experiment.
			errorCorrectRounds: 0,
			// debugMiss: N>0 → during the held-out test (NB decode), analyze misclassified images: print
			// the first N with their per-digit NB log-scores, and aggregate over ALL misses the true
			// digit's rank, the winner−true margin, and voter counts (miss vs hit). Diagnoses whether
			// misses are near-ties (true digit rank 2, small margin) or confident errors (rank 3+).
			debugMiss: 0,
			// evalTrain: after training, run a clean frozen (no-learning) pass over the TRAINING set and
			// report its accuracy. Differs from the per-episode "train=" number, which is prequential
			// (each image scored by the model as it stood before that image's learn()). The frozen pass
			// scores every training image with the FINAL model — revealing apex-churn / wiring-lag loss.
			evalTrain: false,
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
		const radius = num('--radius');
		if (radius !== null) this.config.radius = radius;
		const perClass = num('--per-class');
		if (perClass !== null) this.config.perClass = perClass;
		const maxTestImages = num('--max-test-images');
		if (maxTestImages !== null) this.config.maxTestImages = maxTestImages;
		const episodes = num('--episodes');
		if (episodes !== null) this.config.maxEpisodes = episodes;
		if (process.argv.includes('--skip-test')) this.config.skipTest = true;
		if (process.argv.includes('--split')) this.config.split = true;
		if (process.argv.includes('--no-balance')) this.config.noBalance = true;
		if (process.argv.includes('--eval-train')) this.config.evalTrain = true;
		const missIdx = process.argv.indexOf('--debug-miss');
		if (missIdx !== -1 && process.argv[missIdx + 1] !== undefined) this.config.debugMiss = parseInt(process.argv[missIdx + 1]);
		const ecIdx = process.argv.indexOf('--error-correct-rounds');
		if (ecIdx !== -1 && process.argv[ecIdx + 1] !== undefined) this.config.errorCorrectRounds = parseInt(process.argv[ecIdx + 1]);

		const consensusIdx = process.argv.indexOf('--consensus');
		if (consensusIdx !== -1 && process.argv[consensusIdx + 1] !== undefined) this.config.consensus = process.argv[consensusIdx + 1];
		const epsIdx = process.argv.indexOf('--nb-eps');
		if (epsIdx !== -1 && process.argv[epsIdx + 1] !== undefined) this.config.nbEps = parseFloat(process.argv[epsIdx + 1]);

		if (this.options.contextLength == null) this.options.contextLength = 1;
		if (this.options.patternForgetRate == null) this.options.patternForgetRate = 0;
		// The decode now lives in the brain — hand it the consensus rule and Laplace floor so the
		// winner read out of `inferences` is already the chosen rule's pick (no votes marshalled).
		this.options.consensus = this.config.consensus;
		this.options.nbEps = this.config.nbEps;
	}

	/**
	 * Construct the pixel-channels encoder and register one channel per pixel position with the brain.
	 */
	async initialize() {
		this.encoder = new MNISTPixelChannelsEncoder(this.config.buckets, this.config.imageSize, this.config.radius);
		this.encoder.registerChannels(this.brain);
		// debug-miss option per-digit rank/margin analysis reaggregates the raw votes, so emit them
		if (this.config.debugMiss > 0) this.brain.setEmitVotes(true);
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

		// full data processing: not recommended but can be used for testing
		if (this.config.noBalance) {
			this.trainImages = trainImages;
			this.trainLabels = trainLabels;
			console.log(`  Unbalanced training set: ${this.trainImages.length} total (natural MNIST distribution)`);
		}
		// Class-balanced selection: walk the training set once in order, keep an image for each digit until that digit's per-class quota is filled.
		else {
			const { images: balancedImages, labels: balancedLabels } = this.selectClassBalanced(trainImages, trainLabels);
			this.trainImages = balancedImages;
			this.trainLabels = balancedLabels;
		}

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
		console.log(`  Consensus: ${this.config.consensus}${this.config.consensus === 'nb' ? ` (nb-eps ${this.config.nbEps})` : ''}`);
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

		// optional discriminative second phase: reinforce on training mispredictions only
		if (this.config.errorCorrectRounds > 0) this.runErrorCorrection();

		// now, run the tests evaluation
		if (!this.config.skipTest) this.runEvaluation();
	}

	/**
	 * Discriminative (error-driven) second phase. With minting OFF (setLearning(false), so NO new
	 * patterns are created — only rewards are adjusted), walk the training set; for each image, predict
	 * with the configured decode, and on a MISPREDICTION reinforce the correct digit via the smoothed-
	 * reward learn() wire (correct digit reward=1, all others 0). Repeated over rounds, this pushes the
	 * firing voters' posteriors to fix the training decision — Naive-Bayes → perceptron/logistic. Prints
	 * per-round training accuracy so the climb toward ~100% is visible. Expect train to approach 100%;
	 * test is the open question (overfitting risk vs. independence-violation recalibration).
	 */
	runErrorCorrection() {
		const N = this.trainBits.length;
		console.log('');
		this.brain.setLearning(false); // adjust rewards only — do not mint new patterns this phase
		for (let round = 1; round <= this.config.errorCorrectRounds; round++) {
			const start = Date.now();
			let correct = 0, corrected = 0;
			for (let i = 0; i < N; i++) {
				const label = this.trainLabels[i];
				if (this.predictImage(this.trainBits[i]) === label) {
					correct++;
				} else {
					this.brain.learn(this.encoder.encodeAction(label), 1);
					corrected++;
				}
				this.reportProgress('err-correct', i + 1, N, { correct }, start);
				if (this.isShuttingDown) { this.clearProgress(); return; }
			}
			this.clearProgress();
			console.log(`  Error-correct round ${round}/${this.config.errorCorrectRounds}: train=${(correct / N * 100).toFixed(2)}% (${correct}/${N}) | reinforced ${corrected} | ${Date.now() - start}ms`);
		}
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
		// Frozen pass over the training set, if requested — scores every training image with the
		// FINAL model (vs the prequential per-episode number). The gap measures apex-churn / wiring-lag.
		if (this.config.evalTrain) this.trainEvalResult = this.runTrainEval();
		this.testResult = this.runTest();
	}

	/**
	 * Clean frozen evaluation of the TRAINING set (setLearning(false) already in effect): predict every
	 * training image with the final model, no learning. Reported separately from the prequential
	 * per-episode "train=" number; a lower value here exposes how much the moving apex / one-exposure
	 * wiring lag cost relative to what the online number suggested.
	 */
	runTrainEval() {
		const startTime = Date.now();
		const tally = this.newTally();
		for (let i = 0; i < this.trainBits.length; i++) {
			const predicted = this.predictImage(this.trainBits[i]);
			this.recordPrediction(tally, this.trainLabels[i], predicted);
			this.reportProgress('train-eval', i + 1, this.trainBits.length, tally, startTime);
			if (this.isShuttingDown) break;
		}
		this.clearProgress();
		const duration = Date.now() - startTime;
		const summary = this.summarizeTally(tally);
		const perDigitStr = this.formatPerDigit(summary.perDigit, summary.perDigitTotal);
		console.log(`  Train (frozen, final model): ${(summary.accuracy * 100).toFixed(2)}% (${summary.correct}/${summary.total}) ${duration}ms | ${perDigitStr}`);
		return summary;
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

		// Accumulators for the per-pass training accuracy line: overall correct, per-digit correct, per-digit totals.
		const tally = this.newTally();

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = this.trainLabels[idx];

			// Predict *before* learning so the recorded accuracy reflects what the brain knew going in.
			const predicted = this.predictImage(this.trainBits[idx]);
			this.recordPrediction(tally, label, predicted);

			// Supervised wire: every active sensory neuron is bound to every digit's action neuron at the same-frame voting slot,
			// with reward=1 on the correct digit and reward=0 on every other digit. The brain's smoothed-reward update
			// makes conn.reward(V,d) converge to P(d|V) = K(V,d)/N_V — the per-voter posterior — directly in the reward field.
			this.brain.learn(this.encoder.encodeAction(label), 1);

			// Heartbeat every PROGRESS_EVERY images so long runs don't look frozen.
			this.reportProgress('train', i + 1, indices.length, tally, startTime);

			// Honor an in-flight shutdown without leaving the brain in a half-written state.
			if (this.isShuttingDown) break;
		}
		this.clearProgress();

		// Roll the tally up into a result, attach metadata (which task/episode, wall-clock), log a one-liner, and return.
		const duration = Date.now() - startTime;
		const result = { digit, episode, ...this.summarizeTally(tally), duration, spatial: this.captureSpatialDiagnostics() };
		this.logTrainingPass(result);
		return result;
	}

	/**
	 * Snapshot the spatial-hierarchy diagnostics after a training pass.
	 *   maxSpatialLevel   — depth of the spatial hierarchy (0 = sensory only, 1+ = correction levels exist).
	 *   levelCounts       — live correction neurons per spatial level (index 0 = level 1, …).
	 *   activeCorrections — total live correction neurons above the sensory base.
	 *   cumulativeMinted  — corrections minted since brain start (monotonic; plateaus once local statistics stabilize).
	 *   neuronCount       — total neurons (sensory + actions + all pattern neurons).
	 * These are the signals fixes 1.1/2.1/2.2 are meant to move: depth growing past 1, and minting
	 * plateauing instead of growing linearly with frames.
	 */
	captureSpatialDiagnostics() {
		const summary = this.brain.getFrameSummary();
		return {
			neuronCount: summary.neuronCount,
			maxSpatialLevel: summary.maxSpatialLevel,
			levelCounts: this.brain.spatialLevelCounts(),
			activeCorrections: this.brain.countActiveSpatialCorrections(),
			cumulativeMinted: this.brain.getSpatialCorrectionCount(),
		};
	}

	/**
	 * Format a spatial-diagnostics snapshot into a one-line summary.
	 */
	formatSpatial(s) {
		if (!s) return '';
		const levels = s.levelCounts.length
			? s.levelCounts.map((c, i) => `L${i + 1}:${c}`).join(' ')
			: '(no corrections)';
		return `depth=${s.maxSpatialLevel} | ${levels} | ${s.activeCorrections} active, ${s.cumulativeMinted} minted cum | ${s.neuronCount} neurons`;
	}

	/**
	 * Run one image through the brain and decode the predicted digit.
	 * Shared by training (pre-update prediction) and held-out test.
	 * The brain applies the configured consensus rule ('democratic' | 'nb') internally, so the
	 * winning digit comes straight off `inferences` — no per-vote marshalling on the hot path.
	 */
	predictImage(bits) {
		this.brain.resetContext();
		const inputs = this.encoder.encodeImage(bits);
		const inferResult = this.brain.processFrame(inputs, EMPTY_REWARDS);
		// --debug-miss reaggregates votes app-side for its rank/margin breakdown; stash them when on.
		if (this.config.debugMiss > 0) this._lastVotes = inferResult.votes;
		return this.encoder.decodeDigit(inferResult.inferences);
	}

	/**
	 * Per-digit NB log-score map (Σ_voter log(P(d|voter)+eps)) and the distinct voter count, computed
	 * app-side from emitted votes. Used only by the --debug-miss rank/margin analysis; the live decode
	 * now happens brain-side under the 'nb' consensus. Mirrors the brain's NB rule so the ranks line up.
	 */
	scoreDigitsNB(votes) {
		const logScore = new Map();
		const voters = new Set();
		if (!votes || votes.length === 0) return { logScore, voterCount: 0 };
		const eps = this.config.nbEps;
		const digitChannel = this.encoder.digitChannelId;
		for (const v of votes) {
			if (v.targetType !== 'action' || v.channelId !== digitChannel) continue;
			logScore.set(v.value, (logScore.get(v.value) || 0) + Math.log(v.reward + eps));
			voters.add(v.voterId);
		}
		return { logScore, voterCount: voters.size };
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
	 * Heartbeat line so long passes don't look frozen. Reprints on the same terminal line via \r every PROGRESS_EVERY images.
	 * Skipped between heartbeats and on the very last image (the final-result line takes over).
	 */
	reportProgress(phase, done, total, tally, startTime) {
		if (done === total) return;
		if (done % PROGRESS_EVERY !== 0) return;
		const elapsed = (Date.now() - startTime) / 1000;
		const ips = (done / elapsed).toFixed(0);
		const pct = (done / total * 100).toFixed(1);
		const acc = (tally.correct / done * 100).toFixed(1);
		process.stdout.write(`\r    ${phase} ${done}/${total} (${pct}%) | ${acc}% acc | ${ips} img/s   `);
	}

	/**
	 * Wipe the in-place progress line before the final result prints, so the result is not concatenated to the heartbeat.
	 */
	clearProgress() {
		process.stdout.write('\r' + ' '.repeat(80) + '\r');
	}

	/**
	 * Print the one-line per-pass training log, picking the joint vs split label form.
	 */
	logTrainingPass({ digit, episode, accuracy, correct, total, duration, perDigit, perDigitTotal, spatial }) {
		const ips = (total / (duration / 1000)).toFixed(0);
		const perDigitStr = this.formatPerDigit(perDigit, perDigitTotal);
		const epLabel = this.config.split
			? `Task digit ${digit} — episode ${episode}/${this.config.maxEpisodes}`
			: `Episode ${episode}/${this.config.maxEpisodes}`;
		console.log(`  ${epLabel}: train=${(accuracy * 100).toFixed(2)}% (${correct}/${total}) | ${ips} img/s ${duration}ms | ${perDigitStr}`);
		console.log(`    ↳ spatial: ${this.formatSpatial(spatial)}`);
	}

	/**
	 * Held-out evaluation. setLearning(false) is already in effect.
	 * For each image: resetContext → processFrame → read prediction off its FrameResult. No learn() call.
	 * At context_length=1 the vote generator now keeps the single available age, so processFrame's own
	 * votes/inferences carry the prediction directly — no second inference sweep needed.
	 * Accumulates aggregate accuracy, per-digit accuracy, and the 10×10 confusion matrix called out in the spec.
	 */
	runTest() {
		const startTime = Date.now();
		const tally = this.newTally();
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
		const miss = this.config.debugMiss && this.config.consensus === 'nb'
			? { count: 0, printed: 0, rank: new Array(11).fill(0), marginSum: 0, voterMissSum: 0, voterHitSum: 0, hits: 0 }
			: null;

		for (let i = 0; i < this.testBits.length; i++) {
			const label = this.testLabels[i];
			const predicted = this.predictImage(this.testBits[i]);
			this.recordPrediction(tally, label, predicted);
			// decodeDigit() returns -1 when no action inference is present — keep those out of the confusion matrix.
			if (predicted >= 0 && predicted < 10) confusion[label][predicted]++;
			if (miss) this.analyzeMiss(miss, label, predicted, this.testBits[i]);
			this.reportProgress('test', i + 1, this.testBits.length, tally, startTime);
			if (this.isShuttingDown) break;
		}
		this.clearProgress();

		const duration = Date.now() - startTime;
		const summary = this.summarizeTally(tally);
		const perDigitStr = this.formatPerDigit(summary.perDigit, summary.perDigitTotal);
		console.log(`  Test: ${(summary.accuracy * 100).toFixed(2)}% (${summary.correct}/${summary.total}) ${duration}ms | ${perDigitStr}`);
		if (miss) this.reportMissAnalysis(miss);

		return { ...summary, confusion };
	}

	/**
	 * Per-image miss bookkeeping for --debug-miss. Reads the votes stashed by predictImage, scores the
	 * digits, and records (for misses) the true digit's rank and the winner−true log-score margin, plus
	 * voter counts for both hits and misses. Prints the first `debugMiss` misses in detail.
	 */
	analyzeMiss(miss, label, predicted, bits) {
		const { logScore, voterCount } = this.scoreDigitsNB(this._lastVotes);
		if (predicted === label) { miss.voterHitSum += voterCount; miss.hits++; return; }
		const ranked = [...logScore.entries()].sort((a, b) => b[1] - a[1]);
		const trueRankIdx = ranked.findIndex(([d]) => d === label); // 0-based; -1 if true digit got no votes
		const trueScore = logScore.has(label) ? logScore.get(label) : -Infinity;
		const margin = (ranked.length ? ranked[0][1] : 0) - trueScore;
		miss.count++;
		miss.rank[trueRankIdx < 0 ? 10 : Math.min(trueRankIdx, 10)]++;
		if (Number.isFinite(margin)) miss.marginSum += margin;
		miss.voterMissSum += voterCount;
		if (miss.printed < this.config.debugMiss) {
			miss.printed++;
			const top3 = ranked.slice(0, 3).map(([d, s]) => `${d}:${s.toFixed(1)}`).join(' ');
			const trueStr = Number.isFinite(trueScore) ? trueScore.toFixed(1) : 'absent';
			console.log(`  MISS true=${label} pred=${predicted} | true rank=${trueRankIdx < 0 ? 'none' : trueRankIdx + 1} margin=${Number.isFinite(margin) ? margin.toFixed(2) : '∞'} voters=${voterCount} | top3 ${top3} | true(${label})=${trueStr}`);
			console.log(this.renderBits(bits));
		}
	}

	/**
	 * Render a quantized image as ASCII (one line per row) so a miss can be eyeballed — is it a
	 * genuinely ambiguous shape, or a clean digit the model fumbled? Binary uses '#'/'·'; grayscale
	 * uses a brightness ramp.
	 */
	renderBits(bits) {
		const sz = this.config.imageSize;
		const ramp = ' ·:-=+*#%@';
		const maxBucket = Math.max(1, this.config.buckets - 1);
		const lines = [];
		for (let y = 0; y < sz; y++) {
			let row = '      ';
			for (let x = 0; x < sz; x++) {
				const v = bits[y * sz + x];
				row += this.config.buckets === 2
					? (v ? '#' : '·')
					: ramp[Math.min(ramp.length - 1, Math.round((v / maxBucket) * (ramp.length - 1)))];
			}
			lines.push(row);
		}
		return lines.join('\n');
	}

	/**
	 * Aggregate miss summary: rank distribution (rank 2 = near-tie / true digit was runner-up), average
	 * winner−true margin, and average voters per image for misses vs hits (recognition-collapse proxy).
	 */
	reportMissAnalysis(miss) {
		if (miss.count === 0) { console.log('  MISS ANALYSIS: no misses'); return; }
		const rankStr = miss.rank.map((c, i) => c ? `${i === 10 ? 'r≥11/none' : 'r' + (i + 1)}:${c}` : '').filter(Boolean).join(' ');
		const nearTie = miss.rank[1]; // true digit was the runner-up (0-based index 1)
		console.log(`  MISS ANALYSIS (${miss.count} misses):`);
		console.log(`    true-digit rank: ${rankStr}  → near-ties (rank 2): ${nearTie} (${(100 * nearTie / miss.count).toFixed(0)}%)`);
		console.log(`    avg margin (winner−true, log): ${(miss.marginSum / miss.count).toFixed(2)} | avg voters/img: miss ${(miss.voterMissSum / miss.count).toFixed(0)} vs hit ${miss.hits ? (miss.voterHitSum / miss.hits).toFixed(0) : '-'}`);
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

		// Spatial-hierarchy recap: depth and neuron growth across episodes. The load-bearing check
		// for the spatial-processing fixes — depth should climb past 1 and `minted cum` should
		// plateau across episodes rather than grow linearly with the number of frames seen.
		if (this.episodeResults.some(ep => ep.spatial)) {
			console.log('\n  Spatial hierarchy (after each episode):');
			for (const ep of this.episodeResults) {
				if (!ep.spatial) continue;
				const label = ep.digit != null ? `Task ${ep.digit} ep ${ep.episode}` : `Episode ${ep.episode}`;
				console.log(`    ${label}: ${this.formatSpatial(ep.spatial)}`);
			}
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
