/**
 * Cold-Replay Frame Diff — train for N episodes; after each episode
 * snapshot per-frame active-neuron sets during a COLD REPLAY of one image
 * (learning off). Compare consecutive episode snapshots; find the first
 * frame where they diverge — that's where ep5 training changed the brain
 * such that cold replay now fires differently.
 *
 * Usage:
 *   node apps/mnist/jobs/cold_replay_diff.js [--max-images N] [--episodes N]
 *     [--target-image N] [--diff-from N] [--diff-to N]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoders/mnist_encoder.js';
import { loadImages } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

const num = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseInt(process.argv[i + 1]) : d; };
const numFloat = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : d; };

const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 5);
const TARGET_IMAGE = num('--target-image', 3);   // image #3 in the failing run
const DIFF_FROM = num('--diff-from', 4);
const DIFF_TO = num('--diff-to', 5);
const CONTEXT_LENGTH = num('--context-length', 30);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 1);

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));

const brain = new Brain({ contextLength: CONTEXT_LENGTH, mergeThreshold: MERGE_THRESHOLD, patternForgetRate: 0 });
const encoder = new MNISTEncoder(2, true);
encoder.registerChannels(brain);
const allBits = [];
for (let i = 0; i < MAX_IMAGES; i++) allBits.push(encoder.buildBits(images[i]));

function snap() {
	const set = new Set();
	for (const n of brain.getActiveNeurons()) {
		if (n.level >= 1 && !n.suppressed) set.add(n.neuronId);
	}
	return set;
}

function feed(bits, learning, snapshot = false) {
	brain.resetContext(); brain.resetAccuracyStats(); encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	const frames = snapshot ? new Array(bits.length) : null;
	for (let i = 0; i < bits.length; i++) {
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
		if (snapshot) frames[i] = snap();
	}
	return frames;
}

// snapshots[ep] = per-frame active sets for the COLD REPLAY of TARGET_IMAGE
const snapshots = new Map();

for (let ep = 1; ep <= EPISODES; ep++) {
	// Training pass (in order)
	for (const bits of allBits) feed(bits, true);

	if (ep >= DIFF_FROM && ep <= DIFF_TO) {
		// Cold-replay the target image, snapshot per frame.
		const frames = feed(allBits[TARGET_IMAGE], false, true);
		snapshots.set(ep, frames);
		const sum = brain.getFrameSummary();
		console.log(`After ep${ep} train: ${sum.neuronCount} neurons. Cold-replay image ${TARGET_IMAGE} snapshotted (${frames.length} frames).`);
	} else {
		const sum = brain.getFrameSummary();
		console.log(`After ep${ep} train: ${sum.neuronCount} neurons.`);
	}
}

// Diff consecutive snapshotted episodes
const eps = [...snapshots.keys()].sort((a, b) => a - b);
for (let i = 1; i < eps.length; i++) {
	const epA = eps[i - 1], epB = eps[i];
	const framesA = snapshots.get(epA);
	const framesB = snapshots.get(epB);
	const numFrames = Math.min(framesA.length, framesB.length);

	console.log(`\n=== Cold-replay diff image ${TARGET_IMAGE}: ep${epA} → ep${epB} ===`);
	const fmt = (id) => {
		const info = brain.inspectNeuron(id);
		return `#${id}(L${info.level},p=${info.parentId})`;
	};

	let firstDiff = -1;
	for (let f = 0; f < numFrames; f++) {
		const a = framesA[f], b = framesB[f];
		let differ = false;
		for (const x of a) if (!b.has(x)) { differ = true; break; }
		if (!differ) for (const x of b) if (!a.has(x)) { differ = true; break; }
		if (differ) { firstDiff = f; break; }
	}
	if (firstDiff < 0) {
		console.log(`  identical across all ${numFrames} frames`);
		continue;
	}

	// Show first few divergent frames
	let shown = 0;
	for (let f = firstDiff; f < numFrames && shown < 8; f++) {
		const a = framesA[f], b = framesB[f];
		const onlyA = [...a].filter(x => !b.has(x));
		const onlyB = [...b].filter(x => !a.has(x));
		if (onlyA.length === 0 && onlyB.length === 0) continue;
		shown++;
		console.log(`  frame ${f}:`);
		console.log(`    ep${epA} only (${onlyA.length}): ${onlyA.slice(0, 8).map(fmt).join(' ')}${onlyA.length > 8 ? '...' : ''}`);
		console.log(`    ep${epB} only (${onlyB.length}): ${onlyB.slice(0, 8).map(fmt).join(' ')}${onlyB.length > 8 ? '...' : ''}`);
	}
}

console.log('\nDone.');
