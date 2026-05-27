/**
 * Train 20 images per class (200 total). For each training context, capture
 * end-of-training voter set, grouped by class. For each test-5 image,
 * cold-replay and capture its voters. Then compute the overlap of each
 * test-5's voters with EACH class's training-voter pool.
 *
 * If the brain is class-discriminating, test-5 voters should overlap most
 * with the class-5 pool. If overlap is similar across classes, the brain
 * is not actually distinguishing classes at the pattern level.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');
const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const trainImages = loadImages(findFile('train-images-idx3-ubyte'));
const trainLabels = loadLabels(findFile('train-labels-idx1-ubyte'));
const testImages = loadImages(findFile('t10k-images-idx3-ubyte'));
const testLabels = loadLabels(findFile('t10k-labels-idx1-ubyte'));

const PER_CLASS = 20;

// Pick first PER_CLASS training images for each digit
const trainByClass = Array.from({length: 10}, () => []);
for (let i = 0; trainByClass.flat().length < 10 * PER_CLASS && i < trainLabels.length; i++) {
	const c = trainLabels[i];
	if (trainByClass[c].length < PER_CLASS) trainByClass[c].push(i);
}
for (let c = 0; c < 10; c++) {
	if (trainByClass[c].length < PER_CLASS) {
		console.warn(`Only found ${trainByClass[c].length} training images for class ${c}`);
	}
}

// Pick 20 test-5s
const test5Idxs = [];
for (let i = 0; test5Idxs.length < 20 && i < testLabels.length; i++) {
	if (testLabels[i] === 5) test5Idxs.push(i);
}

const encoder = new MNISTEncoder(2, true);
const brain = new Brain({ contextLength: 30, mergeThreshold: 0.5, patternForgetRate: 0 });
encoder.registerChannels(brain);

const channelId = encoder.channelId;
const inputDimId = encoder.inputDimId;
const buildEvents = (bit) => { const m = new Map(); const dm = new Map(); dm.set(inputDimId, bit); m.set(channelId, dm); return m; };
const EMPTY = new Map();

// Flat training list with class metadata
const trainCtxs = []; // {idx, label}
for (let c = 0; c < 10; c++) {
	for (const idx of trainByClass[c]) trainCtxs.push({ idx, label: c });
}
const N = trainCtxs.length; // 200

console.log(`Training on ${N} images (${PER_CLASS} per class).`);

const trainBitsArr = trainCtxs.map(t => Array.from(encoder.buildBits(trainImages[t.idx])));
const test5BitsArr = test5Idxs.map(i => Array.from(encoder.buildBits(testImages[i])));
const maxLen = Math.max(...trainBitsArr.map(b => b.length), ...test5BitsArr.map(b => b.length));

brain.initContexts(N);
const t0 = Date.now();
brain.setProcessingMode(true, false, true);
for (let pos = 0; pos < maxLen; pos++) {
	for (let i = 0; i < N; i++) {
		if (pos >= trainBitsArr[i].length) continue;
		brain.setActiveContext(i);
		brain.processFrame(buildEvents(trainBitsArr[i][pos]), EMPTY, EMPTY);
	}
}
console.log(`Trained in ${Date.now() - t0}ms. Neurons: ${brain.getFrameSummary().neuronCount}`);

// Capture per-class voter pools
const classPools = Array.from({length: 10}, () => new Set());
for (let i = 0; i < N; i++) {
	brain.setActiveContext(i);
	const voters = brain.getVotableEntries();
	for (const v of voters) classPools[trainCtxs[i].label].add(v.neuronId);
}
console.log(`\nPer-class voter pool sizes:`);
for (let c = 0; c < 10; c++) console.log(`  class ${c}: ${classPools[c].size} unique voter IDs`);

// Also compute pool overlap matrix between classes (sanity check on how
// specific each class's pool is)
console.log(`\nPairwise class-pool overlap (Jaccard, % of smaller set):`);
console.log(`     ` + Array.from({length:10}, (_,i) => `${i}`.padStart(5)).join(''));
for (let a = 0; a < 10; a++) {
	const row = [`${a}: `];
	for (let b = 0; b < 10; b++) {
		if (a === b) { row.push(` --- `); continue; }
		let common = 0;
		for (const id of classPools[a]) if (classPools[b].has(id)) common++;
		const smaller = Math.min(classPools[a].size, classPools[b].size);
		row.push(`${(common / smaller * 100).toFixed(0).padStart(4)}%`);
	}
	console.log(row.join(''));
}

// Cold-replay each test-5 and compute overlap with each class pool
console.log(`\nTest-5 cold replay (overlap with each class pool):`);
console.log(`  test_idx | predicted_class | ` + Array.from({length:10}, (_,i) => `c${i}`).join('  '));
brain.setProcessingMode(true, false, false);
for (let k = 0; k < test5BitsArr.length; k++) {
	brain.setActiveContext(0);
	brain.resetContext();
	const bits = test5BitsArr[k];
	for (let pos = 0; pos < bits.length; pos++) {
		brain.processFrame(buildEvents(bits[pos]), EMPTY, EMPTY);
	}
	const voters = brain.getVotableEntries();
	const overlaps = new Array(10).fill(0);
	for (const v of voters) {
		for (let c = 0; c < 10; c++) {
			if (classPools[c].has(v.neuronId)) overlaps[c]++;
		}
	}
	// Pick the class with highest overlap (our "predicted class" by overlap heuristic)
	let bestC = -1, bestOv = -1;
	for (let c = 0; c < 10; c++) if (overlaps[c] > bestOv) { bestOv = overlaps[c]; bestC = c; }
	const overlapStr = overlaps.map(o => `${o}`.padStart(3)).join(' ');
	const correct = bestC === 5 ? '✓' : '✗';
	console.log(`  test5[${String(test5Idxs[k]).padStart(3)}] | ${bestC} ${correct}      | ${overlapStr}  (${voters.length} voters)`);
}
