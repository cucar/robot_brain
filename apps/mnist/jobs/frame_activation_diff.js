/**
 * Frame Activation Diff — train for many episodes; snapshot per-frame
 * active-neuron sets at the burst boundary (e.g. ep20 vs ep21); find the
 * first frame where the active sets diverge between the two episodes; dump
 * the neuron-level diff at that frame to identify the upstream root cause
 * of the burst.
 *
 * Default behaviour:
 *   1. Train for --episodes (default 22).
 *   2. During episodes within --diff-from..--diff-to (default 20..22),
 *      per-frame per-image snapshot getActiveNeurons() with learning ON.
 *   3. After training, for each consecutive episode pair in that window,
 *      walk image-by-image, frame-by-frame; report the FIRST frame where
 *      the active set differs (i.e. a neuron present at frame F in
 *      episode N is missing in episode N+1, or vice versa).
 *   4. Dump the diff: which neuron(s) appeared, which disappeared, with
 *      their level + parent.
 *
 * Usage: node apps/mnist/jobs/frame_activation_diff.js [--max-images N]
 *        [--episodes N] [--diff-from N] [--diff-to N]
 *        [--context-length N] [--merge-threshold N] [--forget-rate N]
 *        [--min-level N]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

const num = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseInt(process.argv[i + 1]) : d; };
const numFloat = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : d; };
const str = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : d; };

const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 22);
const DIFF_FROM = num('--diff-from', 20);
const DIFF_TO = num('--diff-to', 22);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const MIN_LEVEL = num('--min-level', 1);
const TRIM = !process.argv.includes('--no-trim');

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE
});
const encoder = new MNISTEncoder(2, TRIM);
encoder.registerChannels(brain);
const allBits = [];
for (let i = 0; i < MAX_IMAGES; i++) allBits.push(encoder.buildBits(images[i]));

let shuttingDown = false;
process.on('SIGINT', () => { if (shuttingDown) process.exit(1); shuttingDown = true; });

/**
 * Snapshot the active non-suppressed L≥MIN_LEVEL neuron ids at the current
 * instant. Returns a Set.
 */
function snapshotActive() {
	const set = new Set();
	const active = brain.getActiveNeurons();
	for (const n of active) {
		if (n.level >= MIN_LEVEL && !n.suppressed) set.add(n.neuronId);
	}
	return set;
}

/**
 * Run one image, snapshotting per-frame active sets if requested.
 * @returns Array<Set<NeuronId>> of length bits.length, or null if not snapshotting
 */
function runImage(bits, learning, snapshot) {
	brain.resetContext(); brain.resetAccuracyStats(); encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	const frames = snapshot ? new Array(bits.length) : null;
	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) break;
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
		if (snapshot) frames[i] = snapshotActive();
	}
	return frames;
}

// Per-episode (within DIFF_FROM..DIFF_TO) snapshots:
// snapshots[ep] = [perImage0Frames, perImage1Frames, ...]
// perImage*Frames is array of Sets, one per frame
const snapshots = new Map();
const prevTotal = [];
let lastNeuronId = 0;

console.log(`Frame Activation Diff — ${MAX_IMAGES} images, ${EPISODES} episodes, diffing ep${DIFF_FROM}..${DIFF_TO}`);
console.log(`  ctx=${CONTEXT_LENGTH} merge=${MERGE_THRESHOLD} forget=${FORGET_RATE}`);
console.log('');

let prevNeurons = 0;
for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	const shouldSnap = ep >= DIFF_FROM && ep <= DIFF_TO;
	const epFrames = shouldSnap ? [] : null;
	for (let im = 0; im < allBits.length; im++) {
		const frames = runImage(allBits[im], true, shouldSnap);
		if (shouldSnap) epFrames.push(frames);
	}
	if (shouldSnap) snapshots.set(ep, epFrames);
	const total = brain.getFrameSummary().neuronCount;
	const newOnes = total - prevNeurons;
	console.log(`  ep${String(ep).padStart(2)}: neurons=${total} (+${newOnes})${newOnes >= 5 ? ' ★BURST' : ''}${shouldSnap ? ' [snapshotted]' : ''}`);
	prevNeurons = total;
}

// ── Diff consecutive snapshotted episodes ──────────────────────────────────

function setDiff(a, b) {
	const onlyA = [], onlyB = [];
	for (const x of a) if (!b.has(x)) onlyA.push(x);
	for (const x of b) if (!a.has(x)) onlyB.push(x);
	return { onlyA, onlyB };
}

console.log('\nDiffing consecutive snapshotted episodes...');
const snappedEps = [...snapshots.keys()].sort((a, b) => a - b);
for (let i = 1; i < snappedEps.length; i++) {
	const epA = snappedEps[i - 1];
	const epB = snappedEps[i];
	const framesA = snapshots.get(epA);
	const framesB = snapshots.get(epB);
	console.log(`\n=== Diff ep${epA} → ep${epB} ===`);

	for (let im = 0; im < framesA.length; im++) {
		const imgA = framesA[im];
		const imgB = framesB[im];
		const numFrames = Math.min(imgA.length, imgB.length);
		let firstDiff = -1;
		for (let f = 0; f < numFrames; f++) {
			const d = setDiff(imgA[f], imgB[f]);
			if (d.onlyA.length > 0 || d.onlyB.length > 0) {
				firstDiff = f;
				break;
			}
		}
		if (firstDiff < 0) {
			console.log(`  image ${im}: identical across all ${numFrames} frames`);
			continue;
		}
		const d = setDiff(imgA[firstDiff], imgB[firstDiff]);
		console.log(`  image ${im}: first divergent frame = ${firstDiff}`);
		const fmtNeuron = (id) => {
			const info = brain.inspectNeuron(id);
			return `#${id}(L${info.level},p=${info.parentId})`;
		};
		console.log(`    in ep${epA} only: ${d.onlyA.length} neurons: ${d.onlyA.slice(0, 10).map(fmtNeuron).join(' ')}${d.onlyA.length > 10 ? '...' : ''}`);
		console.log(`    in ep${epB} only: ${d.onlyB.length} neurons: ${d.onlyB.slice(0, 10).map(fmtNeuron).join(' ')}${d.onlyB.length > 10 ? '...' : ''}`);

		// Walk forward a few more frames to see cascade
		let cascadeFrames = 0;
		for (let f = firstDiff + 1; f < numFrames && cascadeFrames < 5; f++) {
			const d2 = setDiff(imgA[f], imgB[f]);
			if (d2.onlyA.length + d2.onlyB.length === 0) continue;
			cascadeFrames++;
			console.log(`    frame ${f}: ep${epA}-only=${d2.onlyA.length} (${d2.onlyA.slice(0, 5).map(fmtNeuron).join(' ')}${d2.onlyA.length > 5 ? '...' : ''}), ep${epB}-only=${d2.onlyB.length} (${d2.onlyB.slice(0, 5).map(fmtNeuron).join(' ')}${d2.onlyB.length > 5 ? '...' : ''})`);
		}
	}
}

console.log('\nDone.');
