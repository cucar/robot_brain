/**
 * Voter Analysis — examines which pattern neurons fire for each digit.
 *
 * Picks one MNIST image per digit (0–9), feeds each as a 784-pixel binary
 * stream through a single channel (like text), repeats for several episodes,
 * then reports which high-level voter neurons are shared vs unique per digit.
 *
 * Usage: node apps/mnist/jobs/voter_analysis.js [--episodes N] [--context-length N]
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

const EPISODES = num('--episodes', 10);
const CONTEXT_LENGTH = num('--context-length', 28);
const MIN_LEVEL = num('--min-level', 2);
const SINGLE_DIGIT = num('--digit', -1);  // -1 = all digits

const numFloat = (flag, fallback) => {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : fallback;
};
const FORGET_RATE = numFloat('--forget-rate', 0);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const TRIM = !process.argv.includes('--no-trim');

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

/**
 * Build a trimmed binary bit stream from a raw 784-pixel image. Leading and
 * trailing all-black runs are stripped when TRIM is on — they are the source
 * of the residual transition mispredict on the single-digit run, and the
 * voter sets we collect should reflect the discriminative portion of each image.
 */
function buildBits(px) {
	const all = new Uint8Array(784);
	for (let i = 0; i < 784; i++) all[i] = quantize(px[i]);
	if (!TRIM) return all;
	let start = 0, end = 784;
	while (start < end && all[start] === 0) start++;
	while (end > start && all[end - 1] === 0) end--;
	return all.subarray(start, end);
}

// Pick the first image for each digit 0–9 and pre-build its trimmed bit stream.
const digitBits = new Array(10).fill(null);
const digitIndices = new Array(10).fill(-1);
for (let i = 0; i < labels.length; i++) {
	const d = labels[i];
	if (digitBits[d] === null) {
		digitBits[d] = buildBits(images[i]);
		digitIndices[d] = i;
	}
	if (digitBits.every(x => x !== null)) break;
}

const digits = SINGLE_DIGIT >= 0 ? [SINGLE_DIGIT] : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

console.log('Voter Analysis — single channel, binary pixels');
console.log(`  Episodes: ${EPISODES}`);
console.log(`  Context length: ${CONTEXT_LENGTH}`);
console.log(`  Min voter level: ${MIN_LEVEL}`);
console.log(`  Forget rate: ${FORGET_RATE}`);
console.log(`  Merge threshold: ${MERGE_THRESHOLD}`);
console.log(`  Trim: ${TRIM}`);
console.log(`  Mode: ${SINGLE_DIGIT >= 0 ? `single digit ${SINGLE_DIGIT}` : 'all digits'}`);
console.log(`  Images: ${digitIndices.map((idx, d) => `${d}:#${idx}(${digitBits[d].length}b)`).join(', ')}`);
console.log('');

// ── Create brain with single channel ────────────────────────────────────────

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	patternForgetRate: FORGET_RATE,
	mergeThreshold: MERGE_THRESHOLD
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

/**
 * Feed a pre-trimmed binary bit stream through the brain, collecting all
 * L≥minLevel neurons that fire at any point during the image.
 *
 * @param {Uint8Array} bits — trimmed binary pixel stream (0/1 values)
 * @param {number} minLevel — minimum level to track
 * @returns {{ voters: Set<number>, perFrame: Map<number, number>, aborted: boolean }}
 *   voters — set of all neuron IDs that fired at minLevel+
 *   perFrame — neuronId → count of frames it appeared in
 *   aborted — true if interrupted by Ctrl+C
 */
function feedImageAndCollect(bits, minLevel) {
	const voters = new Set();
	const perFrame = new Map();

	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return { voters, perFrame, aborted: true };

		const inputs = new Map();
		const dimMap = new Map();
		dimMap.set(dimId, bits[i]);
		inputs.set(channelId, dimMap);
		brain.processFrame(inputs, new Map(), new Map());

		const neurons = brain.getActiveNeurons();
		for (let j = 0; j < neurons.length; j++) {
			if (neurons[j].level >= minLevel && !neurons[j].suppressed) {
				const id = neurons[j].neuronId;
				voters.add(id);
				perFrame.set(id, (perFrame.get(id) || 0) + 1);
			}
		}
	}

	return { voters, perFrame, aborted: false };
}

// ── Ctrl+C support ──────────────────────────────────────────────────────────

let shuttingDown = false;
process.on('SIGINT', () => {
	if (shuttingDown) process.exit(1);
	shuttingDown = true;
	console.log('\nShutting down...');
});

// ── Run episodes ────────────────────────────────────────────────────────────

// Track voter sets per digit across episodes.
// After all episodes, a neuron is a "voter" for digit D if it appeared in
// the majority of episodes for that digit.
const voterHistory = new Array(10).fill(null).map(() => []);

// For single-digit mode, track exact voter IDs per episode for stability report.
const perEpisodeVoters = [];

for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;

	console.log(`Episode ${ep}/${EPISODES}`);

	for (const d of digits) {
		if (shuttingDown) break;

		brain.resetContext();
		brain.resetAccuracyStats();
		brain.setProcessingMode(true, false, true);

		const { voters, perFrame, aborted } = feedImageAndCollect(digitBits[d], MIN_LEVEL);
		if (aborted) break;
		voterHistory[d].push(voters);

		if (SINGLE_DIGIT >= 0) {
			perEpisodeVoters.push(voters);
		}

		const summary = brain.getFrameSummary();

		// Show top voters by frame count (most active across the image).
		const sorted = [...perFrame.entries()].sort((a, b) => b[1] - a[1]);
		const topN = sorted.slice(0, 10).map(([id, cnt]) => `${id}(${cnt})`).join(', ');
		console.log(`  digit ${d}: ${voters.size} voters (L${MIN_LEVEL}+)  neurons=${summary.neuronCount}`);
		console.log(`    top: ${topN}`);
	}

	console.log('');
}

// ── Analysis ────────────────────────────────────────────────────────────────

console.log('='.repeat(70));
console.log('VOTER ANALYSIS');
console.log('='.repeat(70));

// For each digit, find neurons that appeared in at least half the episodes.
const stableVoters = new Array(10).fill(null).map(() => new Set());
for (let d = 0; d < 10; d++) {
	const counts = new Map();
	for (const voters of voterHistory[d]) {
		for (const id of voters) {
			counts.set(id, (counts.get(id) || 0) + 1);
		}
	}
	const threshold = Math.ceil(voterHistory[d].length / 2);
	for (const [id, count] of counts) {
		if (count >= threshold) stableVoters[d].add(id);
	}
}

// Find neurons common to ALL digits.
let commonVoters = new Set(stableVoters[0]);
for (let d = 1; d < 10; d++) {
	const next = new Set();
	for (const id of commonVoters) {
		if (stableVoters[d].has(id)) next.add(id);
	}
	commonVoters = next;
}

console.log(`\nCommon voters (in all 10 digits): ${commonVoters.size}`);
if (commonVoters.size > 0 && commonVoters.size <= 50) {
	console.log(`  IDs: ${[...commonVoters].sort((a, b) => a - b).join(', ')}`);
}

// For each digit, find unique voters (not in any other digit's stable set).
console.log('\nPer-digit stable voters:');
for (let d = 0; d < 10; d++) {
	const unique = new Set();
	for (const id of stableVoters[d]) {
		let shared = false;
		for (let other = 0; other < 10; other++) {
			if (other !== d && stableVoters[other].has(id)) {
				shared = true;
				break;
			}
		}
		if (!shared) unique.add(id);
	}

	console.log(`  digit ${d}: ${stableVoters[d].size} stable, ${unique.size} unique, ${stableVoters[d].size - unique.size} shared`);
	if (unique.size > 0 && unique.size <= 30) {
		console.log(`    unique IDs: ${[...unique].sort((a, b) => a - b).join(', ')}`);
	}
}

// Pairwise overlap matrix.
console.log('\nPairwise overlap (shared stable voters):');
let header = '     ';
for (let d = 0; d < 10; d++) header += `  ${d}`.padStart(5);
console.log(header);
for (let d1 = 0; d1 < 10; d1++) {
	let row = `  ${d1}  `;
	for (let d2 = 0; d2 < 10; d2++) {
		if (d2 === d1) {
			row += `${stableVoters[d1].size}`.padStart(5);
		} else {
			let overlap = 0;
			for (const id of stableVoters[d1]) {
				if (stableVoters[d2].has(id)) overlap++;
			}
			row += `${overlap}`.padStart(5);
		}
	}
	console.log(row);
}

// ── Single-digit stability report ──────────────────────────────────────
if (SINGLE_DIGIT >= 0 && perEpisodeVoters.length > 1) {
	console.log('\nSINGLE-DIGIT STABILITY (episode-by-episode):');
	console.log('-'.repeat(70));

	// Show exact voter IDs per episode.
	for (let ep = 0; ep < perEpisodeVoters.length; ep++) {
		const ids = [...perEpisodeVoters[ep]].sort((a, b) => a - b);
		console.log(`  ep ${ep + 1}: [${ids.join(', ')}]  (${ids.length} voters)`);
	}

	// Pairwise Jaccard similarity between consecutive episodes.
	console.log('\nPairwise Jaccard similarity (consecutive episodes):');
	for (let i = 1; i < perEpisodeVoters.length; i++) {
		const prev = perEpisodeVoters[i - 1];
		const curr = perEpisodeVoters[i];
		let intersection = 0;
		for (const id of prev) {
			if (curr.has(id)) intersection++;
		}
		const union = new Set([...prev, ...curr]).size;
		const jaccard = union > 0 ? intersection / union : 0;
		console.log(`  ep ${i} → ${i + 1}: ${intersection} shared / ${union} union = ${(jaccard * 100).toFixed(1)}%`);
	}

	// Overall: neurons present in ALL episodes vs ANY episode.
	let inAll = new Set(perEpisodeVoters[0]);
	const inAny = new Set(perEpisodeVoters[0]);
	for (let i = 1; i < perEpisodeVoters.length; i++) {
		const next = new Set();
		for (const id of inAll) {
			if (perEpisodeVoters[i].has(id)) next.add(id);
		}
		inAll = next;
		for (const id of perEpisodeVoters[i]) {
			inAny.add(id);
		}
	}
	console.log(`\n  In ALL ${perEpisodeVoters.length} episodes: ${inAll.size} neurons`);
	if (inAll.size > 0 && inAll.size <= 50) {
		console.log(`    IDs: ${[...inAll].sort((a, b) => a - b).join(', ')}`);
	}
	console.log(`  In ANY episode: ${inAny.size} neurons`);
	console.log(`  Stability ratio: ${inAny.size > 0 ? (inAll.size / inAny.size * 100).toFixed(1) : 0}%`);
}

console.log('\n' + '='.repeat(70));
