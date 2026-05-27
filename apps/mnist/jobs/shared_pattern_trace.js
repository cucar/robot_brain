/**
 * Shared Pattern Trace — train on 2 images, then identify which patterns
 * fire for BOTH images on cold replay. For each shared pattern, dump its
 * level, parent, and stored context entries so we can see WHY it isn't
 * unique and whether deeper levels disambiguate.
 *
 * Defaults match the memorization recipe (single-channel, binary pixels,
 * ctx=28, merge=0.99, forget=0, conservative error mode, trim on).
 *
 * Usage: node apps/mnist/jobs/shared_pattern_trace.js [--digit-a N] [--digit-b N]
 *        [--episodes N] [--context-length N] [--merge-threshold N]
 *        [--forget-rate N] [--error-mode static|conservative]
 *        [--no-trim] [--min-level N] [--max-dump N]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoders/mnist_encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

// ── Args ────────────────────────────────────────────────────────────────────

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

const DIGIT_A = num('--digit-a', 5);
const DIGIT_B = num('--digit-b', 0);
const EPISODES = num('--episodes', 10);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const MIN_LEVEL = num('--min-level', 1);
const MAX_DUMP = num('--max-dump', 15);
const TRIM = !process.argv.includes('--no-trim');

// ── Load data ───────────────────────────────────────────────────────────────

const findFile = (b) => {
	const p = path.join(dataDir, b);
	if (fs.existsSync(p)) return p;
	return path.join(dataDir, `${b}.gz`);
};

const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

function firstWithLabel(target) {
	for (let i = 0; i < labels.length; i++) if (labels[i] === target) return { idx: i, pixels: images[i] };
	throw new Error(`no image with label ${target}`);
}

const sampleA = firstWithLabel(DIGIT_A);
const sampleB = firstWithLabel(DIGIT_B);

console.log('Shared Pattern Trace');
console.log(`  Digit A: ${DIGIT_A} (image #${sampleA.idx})`);
console.log(`  Digit B: ${DIGIT_B} (image #${sampleB.idx})`);
console.log(`  Episodes: ${EPISODES}  ctx=${CONTEXT_LENGTH}  errMode=${ERROR_MODE}  merge=${MERGE_THRESHOLD}  forget=${FORGET_RATE}  trim=${TRIM}`);
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

const bitsA = encoder.buildBits(sampleA.pixels);
const bitsB = encoder.buildBits(sampleB.pixels);

// ── Helpers ─────────────────────────────────────────────────────────────────

let shuttingDown = false;
process.on('SIGINT', () => { if (shuttingDown) process.exit(1); shuttingDown = true; });

/**
 * Feed bits through the brain (events only). `learning` controls pattern
 * creation. Returns nothing — caller reads getActiveNeurons after each frame.
 */
function feedBits(bits, learning, onFrame = null) {
	brain.resetContext();
	brain.resetAccuracyStats();
	encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return;
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
		if (onFrame) onFrame(i);
	}
}

/**
 * Cold-replay an image, collecting per-frame the set of non-suppressed
 * active neurons at L >= minLevel.
 *
 * @returns {Map<number, number[]>} neuronId → array of frame indices it fired in
 */
function collectActivations(bits, minLevel) {
	const firingFrames = new Map();
	feedBits(bits, false, (frame) => {
		const active = brain.getActiveNeurons();
		for (const n of active) {
			if (n.level < minLevel || n.suppressed) continue;
			let arr = firingFrames.get(n.neuronId);
			if (!arr) { arr = []; firingFrames.set(n.neuronId, arr); }
			arr.push(frame);
		}
	});
	return firingFrames;
}

// ── Train ───────────────────────────────────────────────────────────────────

console.log('Training...');
for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	// alternate to mirror normal training shuffle (deterministic here)
	feedBits(bitsA, true);
	feedBits(bitsB, true);
}
const trainSummary = brain.getFrameSummary();
console.log(`  After ${EPISODES} episodes: neurons=${trainSummary.neuronCount} maxLevel=${trainSummary.maxLevel}`);

// ── Cold-replay collection ──────────────────────────────────────────────────

console.log('\nCollecting cold-replay activations per image...');
const firingA = collectActivations(bitsA, MIN_LEVEL);
const firingB = collectActivations(bitsB, MIN_LEVEL);

console.log(`  digit ${DIGIT_A}: ${firingA.size} distinct L≥${MIN_LEVEL} neurons fired`);
console.log(`  digit ${DIGIT_B}: ${firingB.size} distinct L≥${MIN_LEVEL} neurons fired`);

// ── Intersection ────────────────────────────────────────────────────────────

const sharedIds = [];
for (const id of firingA.keys()) if (firingB.has(id)) sharedIds.push(id);

console.log(`\n  Shared (fire in BOTH): ${sharedIds.length}`);
console.log(`  Unique to ${DIGIT_A}:    ${firingA.size - sharedIds.length}`);
console.log(`  Unique to ${DIGIT_B}:    ${firingB.size - sharedIds.length}`);

// ── Per-level breakdown ─────────────────────────────────────────────────────

const byLevelA = new Map();
const byLevelB = new Map();
const byLevelShared = new Map();
for (const id of firingA.keys()) {
	const info = brain.inspectNeuron(id);
	byLevelA.set(info.level, (byLevelA.get(info.level) || 0) + 1);
}
for (const id of firingB.keys()) {
	const info = brain.inspectNeuron(id);
	byLevelB.set(info.level, (byLevelB.get(info.level) || 0) + 1);
}
for (const id of sharedIds) {
	const info = brain.inspectNeuron(id);
	byLevelShared.set(info.level, (byLevelShared.get(info.level) || 0) + 1);
}
const allLevels = new Set([...byLevelA.keys(), ...byLevelB.keys(), ...byLevelShared.keys()]);
const sortedLevels = [...allLevels].sort((a, b) => a - b);

console.log('\nPer-level breakdown (shared = fired in BOTH cold replays):');
console.log('  level  digit-A  digit-B  shared  shared%');
for (const lvl of sortedLevels) {
	const a = byLevelA.get(lvl) || 0;
	const b = byLevelB.get(lvl) || 0;
	const s = byLevelShared.get(lvl) || 0;
	const denom = Math.max(1, Math.min(a, b));
	const pct = (s / denom * 100).toFixed(1);
	console.log(`  ${String(lvl).padStart(5)}  ${String(a).padStart(7)}  ${String(b).padStart(7)}  ${String(s).padStart(6)}  ${pct.padStart(5)}%`);
}

// ── Dump some shared patterns with their context ────────────────────────────

console.log(`\nDumping up to ${MAX_DUMP} shared patterns (lowest level first), with parent + context:`);
const sharedSorted = sharedIds.map(id => ({ id, info: brain.inspectNeuron(id) }))
	.sort((a, b) => a.info.level - b.info.level || a.id - b.id);

for (let i = 0; i < Math.min(MAX_DUMP, sharedSorted.length); i++) {
	const { id, info } = sharedSorted[i];
	const framesA = firingA.get(id);
	const framesB = firingB.get(id);
	const parentInfo = info.parentId !== null && info.parentId !== undefined ? brain.inspectNeuron(info.parentId) : null;

	console.log(`\n  Pattern #${id}  L${info.level}  parent=${info.parentId ?? '—'}${parentInfo ? ` (L${parentInfo.level})` : ''}`);
	console.log(`    fires in A: ${framesA.length}× — frames [${framesA.slice(0, 8).join(',')}${framesA.length > 8 ? ',...' : ''}]`);
	console.log(`    fires in B: ${framesB.length}× — frames [${framesB.slice(0, 8).join(',')}${framesB.length > 8 ? ',...' : ''}]`);
	if (info.context.length === 0) {
		console.log(`    context: (none — base sensory or no stored context)`);
	} else {
		const ctxStr = info.context
			.sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId)
			.map(c => `${c.neuronId}@d${c.distance}${c.strength !== 1 ? `(s${c.strength.toFixed(1)})` : ''}`)
			.join(' ');
		console.log(`    context (${info.context.length} entries): ${ctxStr}`);
	}
}

console.log('\nDone.');
