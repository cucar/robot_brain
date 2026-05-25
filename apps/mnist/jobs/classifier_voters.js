/**
 * Classifier Voter Analysis — runs the test.js training pipeline (feed bit
 * stream + brain.learn(digit-label, +reward) per image) and then, after
 * training, replays each training image cold (no learning) to collect the
 * set of L≥minLevel voter neurons that actually fire.
 *
 * Reports per-digit voter set sizes, pairwise overlap, and how many voters
 * appear across multiple digits — the cross-talk metric. If most voters fire
 * for every digit, the action votes aggregate to noise and classification
 * fails regardless of how well the digit-action wires were laid down.
 *
 * Defaults match the memorization recipe (single-channel binary pixel stream,
 * ctx=28, merge=0.99, forget=0, conservative error mode, trim on).
 *
 * Usage:
 *   node apps/mnist/jobs/classifier_voters.js [--digits 0,1,2,...]
 *     [--images-per-digit N] [--episodes N]
 *     [--context-length N] [--merge-threshold N] [--forget-rate N]
 *     [--error-mode static|conservative] [--error-threshold N]
 *     [--no-trim] [--min-level N]
 *     [--correct-reward F] [--incorrect-reward F]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoder.js';
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
const IMAGES_PER_DIGIT = num('--images-per-digit', 1);
const EPISODES = num('--episodes', 3);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const MIN_LEVEL = num('--min-level', 2);
const CORRECT_REWARD = numFloat('--correct-reward', 1);
const INCORRECT_REWARD = numFloat('--incorrect-reward', -1);
const TRIM = !process.argv.includes('--no-trim');

// ── Load data ───────────────────────────────────────────────────────────────

const findFile = (b) => {
	const plain = path.join(dataDir, b);
	if (fs.existsSync(plain)) return plain;
	return path.join(dataDir, `${b}.gz`);
};

const allImages = loadImages(findFile('train-images-idx3-ubyte'));
const allLabels = loadLabels(findFile('train-labels-idx1-ubyte'));

// Pick the first IMAGES_PER_DIGIT examples for each requested digit.
const samples = []; // { digit, idx, bits }
for (const d of DIGITS) {
	let picked = 0;
	for (let i = 0; i < allLabels.length && picked < IMAGES_PER_DIGIT; i++) {
		if (allLabels[i] !== d) continue;
		samples.push({ digit: d, idx: i, pixels: allImages[i] });
		picked++;
	}
	if (picked < IMAGES_PER_DIGIT) throw new Error(`only ${picked} images for digit ${d}`);
}

console.log('Classifier Voter Analysis');
console.log(`  Digits: ${DIGITS.join(',')}  images/digit: ${IMAGES_PER_DIGIT}  total: ${samples.length}`);
console.log(`  Episodes: ${EPISODES}`);
console.log(`  Context length: ${CONTEXT_LENGTH}`);
console.log(`  Error mode: ${ERROR_MODE}  threshold: ${ERROR_THRESHOLD}`);
console.log(`  Merge threshold: ${MERGE_THRESHOLD}  Forget rate: ${FORGET_RATE}`);
console.log(`  Trim: ${TRIM}  Min voter level: ${MIN_LEVEL}`);
console.log(`  Reward: correct=${CORRECT_REWARD} incorrect=${INCORRECT_REWARD}`);
console.log('');

// ── Brain + encoder ─────────────────────────────────────────────────────────

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE
});

const encoder = new MNISTEncoder(2, TRIM);
encoder.registerChannels(brain);

// Pre-encode bit streams.
for (const s of samples) s.bits = encoder.buildBits(s.pixels);

// ── Ctrl+C support ──────────────────────────────────────────────────────────

let shuttingDown = false;
process.on('SIGINT', () => {
	if (shuttingDown) process.exit(1);
	shuttingDown = true;
	console.log('\nShutting down...');
});

/**
 * Feed a bit stream through the brain (event-only frames). Returns the event
 * accuracy over the stream so we can sanity-check that memorization tracks.
 */
function feedBits(bits, learning) {
	brain.setProcessingMode(true, false, learning);
	brain.resetAccuracyStats();
	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return null;
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
	}
	const s = brain.getFrameSummary();
	return s.accuracyTotal > 0 ? s.accuracyCorrect / s.accuracyTotal : null;
}

/**
 * Train one image: feed bits with learning on, then apply the digit label
 * via brain.learn(actions, rewards). Mirrors test.js's runTrainImage.
 */
function trainImage(bits, label) {
	brain.resetContext();
	encoder.resetActions();
	feedBits(bits, true);

	const actions = encoder.encodeAction(label);
	const rewards = encoder.buildRewards(label, CORRECT_REWARD, INCORRECT_REWARD);
	brain.learn(actions, rewards);
	encoder.setForcedAction(label);
}

/**
 * Cold-replay one image with learning off and collect the set of L≥minLevel
 * voter neuron IDs that fire during the bit stream.
 */
function collectVoters(bits, minLevel) {
	brain.resetContext();
	encoder.resetActions();
	brain.setProcessingMode(true, false, false);
	brain.resetAccuracyStats();

	const voters = new Set();
	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return { voters, evtAcc: null };
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
		const active = brain.getActiveNeurons();
		for (let j = 0; j < active.length; j++) {
			if (active[j].level >= minLevel && !active[j].suppressed) {
				voters.add(active[j].neuronId);
			}
		}
	}
	const s = brain.getFrameSummary();
	const evtAcc = s.accuracyTotal > 0 ? s.accuracyCorrect / s.accuracyTotal : null;
	return { voters, evtAcc };
}

// ── Train ───────────────────────────────────────────────────────────────────

console.log('Training...');
for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	for (const s of samples) {
		if (shuttingDown) break;
		trainImage(s.bits, s.digit);
	}
	const summary = brain.getFrameSummary();
	console.log(`  Episode ${ep}: neurons=${summary.neuronCount} maxLevel=${summary.maxLevel}`);
}

// ── Cold-replay voter collection per image ──────────────────────────────────

console.log('\nCold-replay voter collection...');
const perImageVoters = []; // { digit, idx, voters, evtAcc }
for (const s of samples) {
	if (shuttingDown) break;
	const { voters, evtAcc } = collectVoters(s.bits, MIN_LEVEL);
	perImageVoters.push({ digit: s.digit, idx: s.idx, voters, evtAcc });
	console.log(`  digit ${s.digit} #${s.idx}: ${voters.size} voters  evtAcc=${evtAcc !== null ? (evtAcc * 100).toFixed(1) + '%' : '—'}`);
}

// ── Per-digit voter unions ──────────────────────────────────────────────────

const digitVoters = new Map(); // digit -> Set<neuronId>
for (const r of perImageVoters) {
	if (!digitVoters.has(r.digit)) digitVoters.set(r.digit, new Set());
	const u = digitVoters.get(r.digit);
	for (const id of r.voters) u.add(id);
}

// Universe of voters across all digits.
const universe = new Set();
for (const v of digitVoters.values()) for (const id of v) universe.add(id);

console.log('\nPer-digit voter union sizes:');
for (const d of DIGITS) {
	console.log(`  digit ${d}: ${digitVoters.get(d)?.size ?? 0} voters`);
}
console.log(`  Universe: ${universe.size} unique voters across all digits`);

// ── Pairwise overlap ────────────────────────────────────────────────────────

console.log('\nPairwise overlap matrix (rows/cols = digit, cells = shared voters):');
let header = '       ';
for (const d of DIGITS) header += String(d).padStart(7);
console.log(header);
for (const d1 of DIGITS) {
	let row = `   ${d1}   `;
	const v1 = digitVoters.get(d1);
	for (const d2 of DIGITS) {
		const v2 = digitVoters.get(d2);
		if (d1 === d2) {
			row += String(v1.size).padStart(7);
		} else {
			let overlap = 0;
			for (const id of v1) if (v2.has(id)) overlap++;
			row += String(overlap).padStart(7);
		}
	}
	console.log(row);
}

// ── Cross-talk histogram ────────────────────────────────────────────────────

// For each voter, count how many distinct digits it fires for.
const voterDigitCount = new Map(); // neuronId -> count of digits it fires for
for (const [d, voters] of digitVoters) {
	for (const id of voters) {
		voterDigitCount.set(id, (voterDigitCount.get(id) || 0) + 1);
	}
}

const histo = new Array(DIGITS.length + 1).fill(0);
for (const count of voterDigitCount.values()) {
	if (count <= DIGITS.length) histo[count]++;
}

console.log('\nVoter cross-talk histogram (how many digits each voter fires for):');
console.log('  digits  | voters');
let totalShared = 0;
for (let k = 1; k <= DIGITS.length; k++) {
	if (histo[k] === 0) continue;
	const bar = '█'.repeat(Math.min(40, Math.round(histo[k] / Math.max(1, universe.size) * 40)));
	console.log(`  ${String(k).padStart(2)}      | ${String(histo[k]).padStart(6)}  ${bar}`);
	if (k > 1) totalShared += histo[k];
}
console.log(`\n  Voters firing for only 1 digit: ${histo[1]} (${(histo[1] / universe.size * 100).toFixed(1)}%)`);
console.log(`  Voters firing for >=2 digits:   ${totalShared} (${(totalShared / universe.size * 100).toFixed(1)}%)`);

// ── Per-image stability (when images-per-digit > 1) ─────────────────────────

if (IMAGES_PER_DIGIT > 1) {
	console.log('\nPer-digit per-image Jaccard similarity (intra-digit voter consistency):');
	const byDigit = new Map();
	for (const r of perImageVoters) {
		if (!byDigit.has(r.digit)) byDigit.set(r.digit, []);
		byDigit.get(r.digit).push(r);
	}
	for (const d of DIGITS) {
		const list = byDigit.get(d) ?? [];
		if (list.length < 2) continue;
		let sumJ = 0, n = 0;
		for (let i = 0; i < list.length; i++) {
			for (let j = i + 1; j < list.length; j++) {
				let inter = 0;
				for (const id of list[i].voters) if (list[j].voters.has(id)) inter++;
				const union = list[i].voters.size + list[j].voters.size - inter;
				sumJ += union > 0 ? inter / union : 0;
				n++;
			}
		}
		console.log(`  digit ${d}: avg pairwise Jaccard = ${(sumJ / Math.max(1, n) * 100).toFixed(1)}% (over ${n} pairs)`);
	}
}

console.log('\nDone.');
