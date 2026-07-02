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
 * Each invocation runs one or more passes over a single dataset; learning and dataset are independent flags,
 * so the same job is used for training and for evaluation (the stocks-test interface):
 *   - default: train on the balanced training set (learning on), then `--save-brain <label>` persists it.
 *   - `--load-brain <label>`: resume from a saved brain — train another episode and save again, which is
 *     equivalent to having trained those episodes in one run.
 *   - `--disable-learning`: freeze the brain (setLearning(false)) — no wiring, no decay, no minting. Used for evaluation.
 *   - `--test-data`: run the pass over the held-out test set instead of the training set.
 * Training evaluation = load the trained brain, `--disable-learning`, re-run over the training data.
 * Testing evaluation  = load the trained brain, `--disable-learning --test-data`.
 *
 * Per image: resetContext → processFrame populates sensory activations → decode the winning digit. When
 * learning is on, learn(actions, 1) then wires every active sensory neuron to every digit action neuron with
 * reward=1 on the correct digit and reward=0 on the rest; the smoothed-reward update converges each
 * connection's reward to P(d|V) — the per-voter posterior.
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
			// disableLearning: freeze the brain for this run (setLearning(false)) — no wiring, decay, or minting.
			// This is how evaluation is done: load a trained brain, disable learning, and run a pass.
			disableLearning: false,
			// testData: run the pass over the held-out test set instead of the balanced training set.
			testData: false,
			// split: literature-standard class-incremental Split-MNIST — 5 sequential tasks of 2 classes each
			// ({0,1}{2,3}{4,5}{6,7}{8,9}), no task IDs, action space stays all 10 digits. After each task the
			// brain is frozen and tested on all 10 classes to build a retention matrix exposing forgetting.
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
			// debugMiss: N>0 → during the held-out test (NB decode), analyze misclassified images: print
			// the first N with their per-digit NB log-scores, and aggregate over ALL misses the true
			// digit's rank, the winner−true margin, and voter counts (miss vs hit). Diagnoses whether
			// misses are near-ties (true digit rank 2, small margin) or confident errors (rank 3+).
			debugMiss: 0,
		};
		this.encoder = null;
		this.trainImages = null;
		this.trainLabels = null;
		this.trainBits = null;
		this.testImages = null;
		this.testLabels = null;
		this.testBits = null;
		this.episodeResults = [];
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
		if (process.argv.includes('--disable-learning')) this.config.disableLearning = true;
		if (process.argv.includes('--test-data')) this.config.testData = true;
		if (process.argv.includes('--split')) this.config.split = true;
		if (process.argv.includes('--no-balance')) this.config.noBalance = true;
		const missIdx = process.argv.indexOf('--debug-miss');
		if (missIdx !== -1 && process.argv[missIdx + 1] !== undefined) this.config.debugMiss = parseInt(process.argv[missIdx + 1]);

		const consensusIdx = process.argv.indexOf('--consensus');
		if (consensusIdx !== -1 && process.argv[consensusIdx + 1] !== undefined) this.config.consensus = process.argv[consensusIdx + 1];

		// --forget-rate: brain-wide patternForgetRate. MNIST defaults to 0 (additive NB counts never decay).
		// A positive rate makes spatial-correction patterns decay and die when not re-activated, so the
		// network sheds rarely-seen corrections. Must be parsed before the null-default below so it sticks.
		const forgetIdx = process.argv.indexOf('--forget-rate');
		if (forgetIdx !== -1 && process.argv[forgetIdx + 1] !== undefined) this.options.patternForgetRate = parseFloat(process.argv[forgetIdx + 1]);

		// --refine none|context|connection|both: spatial pattern refinement ablation. Maps to env vars
		// brain-core reads once per process (see brain-core/src/config.rs); default (unset) is both ON.
		const refineIdx = process.argv.indexOf('--refine');
		if (refineIdx !== -1 && process.argv[refineIdx + 1] !== undefined) {
			const refine = process.argv[refineIdx + 1];
			process.env.BRAIN_REFINE_CONTEXT = (refine === 'context' || refine === 'both') ? '1' : '0';
			process.env.BRAIN_REFINE_CONNECTION = (refine === 'connection' || refine === 'both') ? '1' : '0';
		}

		if (this.options.contextLength == null) this.options.contextLength = 1;
		if (this.options.patternForgetRate == null) this.options.patternForgetRate = 0;
		// The decode now lives in the brain — hand it the consensus rule so the winner read out of
		// `inferences` is already the chosen rule's pick (no votes marshalled). The NB Laplace floor
		// is a baked brain constant, no longer a knob.
		this.options.consensus = this.config.consensus;
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
		console.log('MNIST — sensory-only (Naive Bayes) baseline');
		console.log(`  Image size: ${this.config.imageSize}×${this.config.imageSize} (${this.config.imageSize * this.config.imageSize} channels)`);
		console.log(`  Buckets: ${this.config.buckets}`);
		console.log(`  Context length: ${this.options.contextLength}`);
		console.log(`  Forget rate: ${this.options.patternForgetRate}`);
		console.log(`  Consensus: ${this.config.consensus}`);
		console.log(`  Dataset: ${this.config.testData ? 'test (held-out)' : 'training (balanced)'}`);
		console.log(`  Learning: ${this.config.disableLearning ? 'OFF (frozen — evaluation)' : 'ON'}`);
		if (this.config.split && !this.config.testData && !this.config.disableLearning)
			console.log(`  Mode: Split-MNIST (class-incremental, 5 tasks × 2 classes, sequential, no task IDs)`);
		else
			console.log(`  Episodes: ${this.config.maxEpisodes}`);
		console.log(`  Training selection: ${this.config.noBalance ? 'full' : 'balanced'}${this.config.perClass > 0 ? `, ${this.config.perClass} per class` : ''}`);
		console.log('');
	}

	/**
	 * Run the configured passes over the active dataset. Learning and dataset are independent flags:
	 * the default trains on the balanced set; --disable-learning freezes the brain for evaluation;
	 * --test-data swaps in the held-out set. There is no implicit second evaluation pass — evaluation
	 * is a separate invocation with the trained brain loaded and learning disabled.
	 */
	async executeJob() {

		// Freeze the brain up front when evaluating — no wiring, decay, or minting for the whole run.
		if (this.config.disableLearning) this.brain.setLearning(false);
		const learning = !this.config.disableLearning;

		// Split-MNIST is a sequential class-incremental training protocol — only meaningful with
		// learning on over the training set. It runs its own loop and reporting and returns early.
		if (this.config.split && !this.config.testData && learning) await this.runSplitMnist();
		else await this.runJointPasses(learning);
	}

	/**
	 * Yield to the event loop so a queued SIGINT handler can run and set isShuttingDown.
	 * The pass loops are otherwise fully synchronous NAPI calls, so without an occasional yield
	 * Ctrl+C is never serviced until the whole pass finishes — the isShuttingDown checks would be
	 * dead during a run. Yield on the heartbeat cadence so the cost is negligible per image.
	 */
	async yieldForSignals(done) {
		if (done % PROGRESS_EVERY === 0) await new Promise(resolve => setImmediate(resolve));
	}

	/**
	 * Joint mode: maxEpisodes passes over the whole active dataset.
	 */
	async runJointPasses(learning) {
		for (let ep = 1; ep <= this.config.maxEpisodes; ep++) {
			this.episodeResults.push(await this.runPass(ep, null, learning));
			if (this.isShuttingDown) return;
		}
	}

	/**
	 * Run literature-standard class-incremental Split-MNIST (van de Ven & Tolias 2019; Hsu et al. 2018).
	 */
	async runSplitMnist() {

		// Five sequential tasks of two classes each: T0={0,1} T1={2,3} T2={4,5} T3={6,7} T4={8,9}.
		// The balanced training set is sorted by digit (selectClassBalanced with split=true), so digit d's
		// samples occupy [d*cap, (d+1)*cap) and task t spans both of its digits at [2t*cap, (2t+2)*cap).
		const cap = this.trainBits.length / 10;
		this.splitMatrix = [];
		for (let task = 0; task < 5; task++) {

			// Strict sequential training: each task's data is seen once and never revisited.
			// There are no task IDs, and the action space stays all ten digits throughout — the digit action
			// channel is registered up front, so every learn() wires all ten (reward 1 on the true digit, 0 on the rest).
			this.brain.setLearning(true);
			const lo = 2 * task * cap;
			const hi = (2 * task + 2) * cap;
			const indices = Array.from({ length: hi - lo }, (_, k) => lo + k);
			await this.trainTaskPass(task, indices);
			if (this.isShuttingDown) return;

			// Freeze and evaluate the FULL 10-class test set, binned by task.
			// This row of the retention matrix exposes catastrophic forgetting (or its absence) across all tasks.
			this.brain.setLearning(false);
			const row = await this.evalTestByTask();
			this.splitMatrix.push(row);
			const rowStr = row.map((a, j) => `T${j}:${(a * 100).toFixed(1)}%`).join(' ');
			console.log(`    ↳ after task ${task}: ${rowStr}`);
			if (this.isShuttingDown) return;
		}
	}

	/**
	 * Train a single Split-MNIST task: one pass over its two digits' samples, learning on.
	 */
	async trainTaskPass(task, indices) {
		const startTime = Date.now();
		const labels = this.trainLabels;
		let correct = 0;
		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = labels[idx];

			// The recorded accuracy is prequential — the guess before this image's supervised wire lands.
			// Order within the task is deterministic; with forget rate 0 the additive NB counts are order-independent anyway.
			const predicted = this.classifyImage(this.trainBits[idx]);
			if (predicted === label) correct++;
			this.brain.learn(this.encoder.encodeAction(label), 1);

			this.reportProgress(`task${task}`, i + 1, indices.length, { correct }, startTime);
			await this.yieldForSignals(i + 1);
			if (this.isShuttingDown) break;
		}
		this.clearProgress();
		const duration = Date.now() - startTime;
		console.log(`  Task ${task} {${2 * task},${2 * task + 1}}: trained ${indices.length} imgs, prequential ${(100 * correct / indices.length).toFixed(2)}% (${duration}ms)`);
		console.log(`    ↳ spatial: ${this.formatSpatial(this.captureSpatialDiagnostics())}`);
	}

	/**
	 * Frozen evaluation over the FULL 10-class test set, returning per-task accuracy [5]. A test image's
	 * task is floor(label/2). setLearning(false) must already be in effect.
	 */
	async evalTestByTask() {
		const correct = new Array(5).fill(0);
		const total = new Array(5).fill(0);
		for (let i = 0; i < this.testBits.length; i++) {
			const label = this.testLabels[i];
			const task = Math.floor(label / 2);
			const predicted = this.classifyImage(this.testBits[i]);
			total[task]++;
			if (predicted === label) correct[task]++;
			await this.yieldForSignals(i + 1);
			if (this.isShuttingDown) break;
		}
		return correct.map((c, t) => (total[t] > 0 ? c / total[t] : 0));
	}

	/**
	 * One pass over the active dataset. For each image:
	 *   resetContext → processFrame → decode prediction → (if learning) learn(actions, 1).
	 * When learning is on the recorded accuracy is prequential — the brain's guess *before* this image's
	 * supervised wire lands. When learning is off it is a clean frozen evaluation with the fixed model.
	 * `digit` restricts a split-mode training pass to one digit's slice; null walks the whole dataset.
	 */
	async runPass(episode, digit, learning) {

		// Track wall-clock so we can report img/s for this pass.
		const startTime = Date.now();

		// Accumulators: overall/per-digit tally and the 10×10 confusion matrix.
		const tally = this.newTally();
		const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
		const miss = this.config.debugMiss && this.config.consensus === 'nb'
			? { count: 0, printed: 0, rank: new Array(11).fill(0), marginSum: 0, voterMissSum: 0, voterHitSum: 0, hits: 0 }
			: null;

		const { bits, labels } = this.activeDataset();
		const indices = this.buildPassIndices(digit);
		const phase = learning ? 'train' : 'eval';

		for (let i = 0; i < indices.length; i++) {
			const idx = indices[i];
			const label = labels[idx];

			// Classify the image (automatic inference), then record the predicted digit.
			const predicted = this.classifyImage(bits[idx]);
			this.recordPrediction(tally, label, predicted);

			// decodeDigit() returns -1 when no action inference is present — keep those out of the confusion matrix.
			if (predicted >= 0 && predicted < 10) confusion[label][predicted]++;
			if (miss) this.analyzeMiss(miss, label, predicted, bits[idx]);

			// Supervised wire (training only): every active sensory neuron is bound to every digit's action
			// neuron at the same-frame voting slot, reward=1 on the correct digit and 0 on the rest. The
			// smoothed-reward update makes conn.reward(V,d) converge to P(d|V) = K(V,d)/N_V — the per-voter posterior.
			if (learning) this.brain.learn(this.encoder.encodeAction(label), 1);

			// Heartbeat every so often so that long runs don't look frozen.
			this.reportProgress(phase, i + 1, indices.length, tally, startTime);

			// Yield to the event loop on the heartbeat cadence so a pending Ctrl+C can be serviced,
			// then honor an in-flight shutdown without leaving the brain in a half-written state.
			await this.yieldForSignals(i + 1);
			if (this.isShuttingDown) break;
		}
		this.clearProgress();

		// Roll the tally up into a result, attach metadata, log a one-liner, and return.
		const duration = Date.now() - startTime;
		const result = { digit, episode, learning, ...this.summarizeTally(tally), confusion, duration, spatial: this.captureSpatialDiagnostics() };
		this.logPass(result);
		if (miss) this.reportMissAnalysis(miss);
		return result;
	}

	/**
	 * Bits + labels for the active dataset: the held-out test set under --test-data, else the balanced training set.
	 */
	activeDataset() {
		return this.config.testData
			? { bits: this.testBits, labels: this.testLabels }
			: { bits: this.trainBits, labels: this.trainLabels };
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
	 * Shared by every pass — training (pre-update prediction) and frozen evaluation.
	 * The brain applies the configured consensus rule ('democratic' | 'nb') internally, so the
	 * winning digit comes straight off `inferences` — no per-vote marshalling on the hot path.
	 */
	classifyImage(bits) {
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
		// Mirrors the brain's baked NB_EPS constant so the app-side ranks line up with the brain decode.
		const eps = 1e-3;
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
	logPass({ digit, episode, learning, accuracy, correct, total, duration, perDigit, perDigitTotal, spatial }) {
		const ips = (total / (duration / 1000)).toFixed(0);
		const perDigitStr = this.formatPerDigit(perDigit, perDigitTotal);
		const epLabel = this.config.split && !this.config.testData
			? `Task digit ${digit} — episode ${episode}/${this.config.maxEpisodes}`
			: `Episode ${episode}/${this.config.maxEpisodes}`;
		// Prequential 'train=' under learning; frozen 'eval(train|test)=' otherwise.
		const tag = learning ? (this.config.testData ? 'test' : 'train') : `eval(${this.config.testData ? 'test' : 'train'})`;
		console.log(`  ${epLabel}: ${tag}=${(accuracy * 100).toFixed(2)}% (${correct}/${total}) | ${ips} img/s ${duration}ms | ${perDigitStr}`);
		console.log(`    ↳ spatial: ${this.formatSpatial(spatial)}`);
	}

	/**
	 * Per-image miss bookkeeping for --debug-miss. Reads the votes stashed by classifyImage, scores the
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
	 * Pick the image indices for a pass over the active dataset.
	 * Joint mode (digit=null): sequential walk over the whole dataset.
	 * Split mode (digit=0..9): only the slice for that digit. The balanced training set is sorted by
	 * digit with exactly `cap = N/10` samples per class, so digit d's slice is [d*cap, (d+1)*cap).
	 */
	buildPassIndices(digit) {
		const { bits } = this.activeDataset();
		const N = bits.length;
		if (digit == null) return Array.from({ length: N }, (_, i) => i);
		const cap = N / 10;
		const start = digit * cap;
		return Array.from({ length: cap }, (_, i) => start + i);
	}

	/**
	 * Split-MNIST report: the 5×5 retention matrix (rows = after training task i, cols = test accuracy on
	 * task j's two classes), the headline average accuracy after Task 5, and per-task forgetting
	 * (max-ever accuracy minus final). These are the standard class-incremental continual-learning
	 * metrics; the literature floor for naive backprop here is ~20%.
	 */
	showSplitResults() {
		const M = this.splitMatrix;
		console.log('\nSplit-MNIST (class-incremental — 5 tasks × 2 classes, no task IDs)');
		console.log('='.repeat(70));
		console.log('  Retention matrix — rows: after training task i; cols: frozen test acc on task j');

		let header = '              ';
		for (let j = 0; j < 5; j++) header += `T${j}(${2 * j}${2 * j + 1})`.padStart(9);
		console.log(header);
		for (let i = 0; i < M.length; i++) {
			let row = `  after T${i}    `;
			for (let j = 0; j < 5; j++) row += `${(M[i][j] * 100).toFixed(1)}%`.padStart(9);
			console.log(row);
		}

		// Headline: average accuracy over all classes after the final task (mean of the last row,
		// equal-sized tasks → the overall 10-class test accuracy).
		const last = M[M.length - 1];
		const avg = last.reduce((a, b) => a + b, 0) / last.length;
		console.log(`\n  Average accuracy after Task ${M.length - 1} (headline): ${(avg * 100).toFixed(2)}%`);

		// Forgetting: for each task, the drop from its best-ever accuracy (once trained) to its final accuracy.
		const forgets = [];
		for (let j = 0; j < 5; j++) {
			let best = 0;
			for (let i = j; i < M.length; i++) best = Math.max(best, M[i][j]);
			forgets.push(best - last[j]);
		}
		const avgForget = forgets.reduce((a, b) => a + b, 0) / forgets.length;
		console.log(`  Forgetting per task (max-ever − final): ${forgets.map((f, j) => `T${j}:${(f * 100).toFixed(1)}pp`).join(' ')}`);
		console.log(`  Average forgetting: ${(avgForget * 100).toFixed(2)}pp  (naive backprop floor ≈ ~20% avg acc / heavy forgetting)`);
		console.log('='.repeat(70));
	}

	/**
	 * Final summary: per-pass accuracy, the joint-mode learning curve, and the final pass's confusion matrix.
	 * The confusion matrix is the load-bearing artifact for the NB-band check — the 3/8/9 collapses are what
	 * motivate the spatial-processing workstream. For an evaluation run (--disable-learning [--test-data]) the
	 * single pass's accuracy + confusion are the held-out result.
	 */
	async showResults() {
		// Split-MNIST has its own retention-matrix report.
		if (this.splitMatrix) { this.showSplitResults(); return; }

		console.log('\nResults');
		console.log('='.repeat(70));

		const tag = this.config.testData ? 'test' : 'train';
		for (const ep of this.episodeResults) {
			const label = ep.digit != null
				? `Task digit ${ep.digit} ep ${ep.episode}`
				: `Episode ${ep.episode}`;
			const verb = ep.learning ? tag : `eval(${tag})`;
			console.log(`  ${label}: ${verb}=${(ep.accuracy * 100).toFixed(2)}% (${ep.correct}/${ep.total}, ${ep.duration}ms)`);
		}

		// Spatial-hierarchy recap: depth and neuron growth across passes. The load-bearing check
		// for the spatial-processing fixes — depth should climb past 1 and `minted cum` should
		// plateau across episodes rather than grow linearly with the number of frames seen.
		if (this.episodeResults.some(ep => ep.spatial)) {
			console.log('\n  Spatial hierarchy (after each pass):');
			for (const ep of this.episodeResults) {
				if (!ep.spatial) continue;
				const label = ep.digit != null ? `Task ${ep.digit} ep ${ep.episode}` : `Episode ${ep.episode}`;
				console.log(`    ${label}: ${this.formatSpatial(ep.spatial)}`);
			}
		}

		// Joint-mode-only: first vs last accuracy is a meaningful learning curve.
		// In split mode each entry measures accuracy on a different digit, so the delta is apples-to-oranges — skip it.
		if (!(this.config.split && !this.config.testData) && this.episodeResults.length >= 2) {
			const first = this.episodeResults[0];
			const last = this.episodeResults[this.episodeResults.length - 1];
			const delta = ((last.accuracy - first.accuracy) * 100).toFixed(2);
			console.log(`\n  Accuracy: ${(first.accuracy * 100).toFixed(2)}% → ${(last.accuracy * 100).toFixed(2)}% (${delta >= 0 ? '+' : ''}${delta}pp)`);
		}

		// Confusion matrix of the final pass — the headline artifact for an evaluation run.
		const last = this.episodeResults[this.episodeResults.length - 1];
		if (last && last.confusion) {
			console.log(`\n  Final pass: ${(last.accuracy * 100).toFixed(2)}% (${last.correct}/${last.total})`);
			console.log('  Confusion (rows = actual, cols = predicted):');
			let header = '       ';
			for (let p = 0; p < 10; p++) header += String(p).padStart(5);
			console.log(header);
			for (let a = 0; a < 10; a++) {
				let row = `   ${a}   `;
				for (let p = 0; p < 10; p++) row += String(last.confusion[a][p]).padStart(5);
				console.log(row);
			}
		}
		console.log('='.repeat(70));
	}
}

await runJob(import.meta, MNISTTestJob);
