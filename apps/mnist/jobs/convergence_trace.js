/**
 * Convergence Trace — feeds a single MNIST image as a binary pixel stream
 * (single channel, 1 dimension) and reports per-episode prediction accuracy.
 * Mirrors the text-test memorization recipe so the brain can converge to
 * 100% on a single image.
 *
 * Defaults are the "memorization recipe" found via the text-test sweep:
 *   --context-length 28     row-width: lets the brain see the pixel one row up
 *   --merge-threshold 0.99  tight merge for fast convergence
 *   --forget-rate 0         no decay — memorize, don't forget
 *   --trim                  strip leading/trailing all-black runs
 *
 * Usage: node apps/mnist/jobs/convergence_trace.js [--digit N] [--episodes N]
 *        [--context-length N] [--error-mode static|conservative]
 *        [--error-threshold N] [--merge-threshold N] [--forget-rate N]
 *        [--trim|--no-trim] [--verbose]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

// ── Parse args ──────────────────────────────────────────────────────────────

const num = (flag, fallback) => {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] ? parseInt(process.argv[i + 1]) : fallback;
};
const numFloat = (flag, fallback) => {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : fallback;
};
const str = (flag, fallback) => {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

const DIGIT = num('--digit', 0);
const EPISODES = num('--episodes', 10);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const TRIM = !process.argv.includes('--no-trim');
const VERBOSE = process.argv.includes('--verbose');

// ── Load data ───────────────────────────────────────────────────────────────

const findFile = (base) => {
	const plain = path.join(dataDir, base);
	if (fs.existsSync(plain)) return plain;
	return path.join(dataDir, `${base}.gz`);
};

const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

// Find first image with the requested digit.
let targetIdx = -1;
let targetPixels = null;
for (let i = 0; i < labels.length; i++) {
	if (labels[i] === DIGIT) {
		targetIdx = i;
		targetPixels = images[i];
		break;
	}
}

/**
 * Quantize a pixel to binary (0 or 1).
 *
 * @param {number} value — raw grayscale 0–255
 * @returns {number} 0 or 1
 */
function quantize(value) {
	return value >= 128 ? 1 : 0;
}

// Build the binary pixel sequence, optionally trimming leading/trailing zeros.
// The black preamble/postamble in MNIST is "easy" to predict but the transition
// from preamble to first white pixel was the one residual mispredict on the
// full 784-pixel stream — trimming removes it.
const allBits = new Uint8Array(784);
for (let i = 0; i < 784; i++) allBits[i] = quantize(targetPixels[i]);

let start = 0, end = 784;
if (TRIM) {
	while (start < end && allBits[start] === 0) start++;
	while (end > start && allBits[end - 1] === 0) end--;
}
const bits = allBits.subarray(start, end);

console.log('Convergence Trace — single channel, binary pixels');
console.log(`  Digit: ${DIGIT} (image #${targetIdx})`);
console.log(`  Episodes: ${EPISODES}`);
console.log(`  Context length: ${CONTEXT_LENGTH}`);
console.log(`  Error mode: ${ERROR_MODE}`);
console.log(`  Error threshold: ${ERROR_THRESHOLD}`);
console.log(`  Merge threshold: ${MERGE_THRESHOLD}`);
console.log(`  Forget rate: ${FORGET_RATE}`);
console.log(`  Trim: ${TRIM} (raw=784, trimmed=${bits.length}, leading=${start}, trailing=${784 - end})`);
console.log('');

// ── Create brain ────────────────────────────────────────────────────────────

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE,
	debug: VERBOSE
});

const spec = brain.registerChannelSpec({
	name: 'px',
	emitsReward: false,
	learnActionSequences: false,
	dimensions: [
		{
			name: 'px_val',
			kind: 'input',
			resolution: 2,
			mode: 'passthrough'
		}
	]
});

const channelId = spec.channelId;
const dimId = spec.dimensionIds['px_val'];

// ── Ctrl+C support ──────────────────────────────────────────────────────────

let shuttingDown = false;
process.on('SIGINT', () => {
	if (shuttingDown) process.exit(1);
	shuttingDown = true;
	console.log('\nShutting down...');
});

// ── Episodes ────────────────────────────────────────────────────────────────

for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;

	brain.resetContext();
	brain.resetAccuracyStats();
	brain.setProcessingMode(true, false, true);

	let prevNeuronCount = brain.getFrameSummary().neuronCount;
	const creations = [];

	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) break;
		const inputs = new Map();
		const dimMap = new Map();
		dimMap.set(dimId, bits[i]);
		inputs.set(channelId, dimMap);
		brain.processFrame(inputs, new Map(), new Map());

		if (VERBOSE) {
			const s = brain.getFrameSummary();
			const created = s.neuronCount - prevNeuronCount;
			prevNeuronCount = s.neuronCount;
			if (created > 0) creations.push({ frame: i, px: bits[i], created, total: s.neuronCount });
		}
	}

	const summary = brain.getFrameSummary();
	const acc = summary.accuracyTotal > 0
		? (summary.accuracyCorrect / summary.accuracyTotal * 100).toFixed(2)
		: 'N/A';
	const errs = summary.accuracyTotal - summary.accuracyCorrect;
	console.log(`  Episode ${String(ep).padStart(2)}/${EPISODES}: acc=${acc}%  (${summary.accuracyCorrect}/${summary.accuracyTotal}, ${errs} wrong)  neurons=${summary.neuronCount}  maxLevel=${summary.maxLevel}`);

	if (VERBOSE && creations.length > 0) {
		for (const c of creations) {
			console.log(`     frame ${String(c.frame).padStart(3)}: px=${c.px}  +${c.created} neurons  total=${c.total}`);
		}
	}
}

console.log('\nDone.');
