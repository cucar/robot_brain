/**
 * Multi-Digit Convergence Trace — feeds one image per digit (0–9) as binary
 * pixel streams through a single shared channel, episode by episode, and
 * reports per-digit memorization accuracy.
 *
 * Context is reset between images so each image is its own "thought" — the
 * test is whether each digit's patterns survive interleaving with the others.
 *
 * Defaults match the single-digit memorization recipe:
 *   --context-length 28     row width + 1
 *   --merge-threshold 0.99  tight merge for fast convergence
 *   --forget-rate 0         no decay
 *   --trim                  strip leading/trailing all-black runs per image
 *
 * Usage: node apps/mnist/jobs/multi_digit_trace.js [--digits 0,1,2,...]
 *        [--episodes N] [--context-length N] [--merge-threshold N]
 *        [--forget-rate N] [--error-mode static|conservative]
 *        [--error-threshold N] [--no-trim] [--verbose]
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

const DIGITS_ARG = str('--digits', '0,1,2,3,4,5,6,7,8,9');
const DIGITS = DIGITS_ARG.split(',').map(s => parseInt(s.trim()));
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

/**
 * Quantize a pixel to binary (0 or 1).
 *
 * @param {number} value — raw grayscale 0–255
 * @returns {number} 0 or 1
 */
function quantize(value) {
	return value >= 128 ? 1 : 0;
}

// Pick the first image per requested digit and build the trimmed bit stream.
const samples = DIGITS.map(digit => {
	let idx = -1;
	for (let i = 0; i < labels.length; i++) {
		if (labels[i] === digit) { idx = i; break; }
	}
	if (idx === -1) throw new Error(`no training image found for digit ${digit}`);
	const px = images[idx];
	const all = new Uint8Array(784);
	for (let i = 0; i < 784; i++) all[i] = quantize(px[i]);

	let start = 0, end = 784;
	if (TRIM) {
		while (start < end && all[start] === 0) start++;
		while (end > start && all[end - 1] === 0) end--;
	}
	return { digit, imageIndex: idx, bits: all.subarray(start, end), raw: all, lead: start, trail: 784 - end };
});

console.log('Multi-Digit Convergence Trace — single channel, binary pixels');
console.log(`  Digits: ${DIGITS.join(',')}`);
console.log(`  Episodes: ${EPISODES}`);
console.log(`  Context length: ${CONTEXT_LENGTH}`);
console.log(`  Error mode: ${ERROR_MODE}`);
console.log(`  Error threshold: ${ERROR_THRESHOLD}`);
console.log(`  Merge threshold: ${MERGE_THRESHOLD}`);
console.log(`  Forget rate: ${FORGET_RATE}`);
console.log(`  Trim: ${TRIM}`);
for (const s of samples) {
	console.log(`    digit ${s.digit}: image #${s.imageIndex}  bits=${s.bits.length}  (lead=${s.lead} trail=${s.trail})`);
}
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

/**
 * Feed one image (binary bits) through the brain as a fresh sequence and
 * return its per-frame prediction accuracy over the run.
 *
 * @param {Uint8Array} bits
 * @returns {{ correct: number, total: number, neurons: number }}
 */
function runImage(bits) {
	brain.resetContext();
	brain.resetAccuracyStats();
	brain.setProcessingMode(true, false, true);

	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return null;
		const inputs = new Map();
		const dimMap = new Map();
		dimMap.set(dimId, bits[i]);
		inputs.set(channelId, dimMap);
		brain.processFrame(inputs, new Map(), new Map());
	}
	const s = brain.getFrameSummary();
	return { correct: s.accuracyCorrect, total: s.accuracyTotal, neurons: s.neuronCount, maxLevel: s.maxLevel };
}

// ── Episodes ────────────────────────────────────────────────────────────────

const accHistory = samples.map(() => []);

for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;

	let finalNeurons = 0, finalMaxLevel = 0;
	const parts = [];

	for (let d = 0; d < samples.length; d++) {
		if (shuttingDown) break;
		const res = runImage(samples[d].bits);
		if (!res) break;
		const acc = res.total > 0 ? res.correct / res.total : 0;
		accHistory[d].push(acc);
		finalNeurons = res.neurons;
		finalMaxLevel = res.maxLevel;
		parts.push(`${samples[d].digit}:${(acc * 100).toFixed(1)}%`);
	}

	const avg = accHistory.reduce((sum, h) => sum + (h[h.length - 1] ?? 0), 0) / samples.length;
	console.log(`  Episode ${String(ep).padStart(2)}/${EPISODES}: avg=${(avg * 100).toFixed(2)}%  neurons=${finalNeurons}  maxLevel=${finalMaxLevel}  | ${parts.join(' ')}`);
}

// ── Final summary ───────────────────────────────────────────────────────────

console.log('\nPer-digit accuracy over episodes:');
console.log('  ep  ' + samples.map(s => `   d${s.digit}  `).join(''));
for (let ep = 0; ep < accHistory[0].length; ep++) {
	const row = String(ep + 1).padStart(4);
	const cells = samples.map((_, d) => {
		const a = accHistory[d][ep];
		return a === undefined ? '   --  ' : `${(a * 100).toFixed(1)}%`.padStart(7);
	}).join('');
	console.log('  ' + row + cells);
}

console.log('\nDone.');
