/**
 * Pick test5[8] (which had 28/29 voters in class-0 pool). Re-train + classify
 * and dump the stored contexts of its shared-with-class-0 voters, walking the
 * hierarchy down to see what history they actually encode.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoders/mnist_encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');
const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const trainImages = loadImages(findFile('train-images-idx3-ubyte'));
const trainLabels = loadLabels(findFile('train-labels-idx1-ubyte'));
const testImages = loadImages(findFile('t10k-images-idx3-ubyte'));

const PER_CLASS = 20;
const trainByClass = Array.from({length: 10}, () => []);
for (let i = 0; trainByClass.flat().length < 10 * PER_CLASS && i < trainLabels.length; i++) {
	const c = trainLabels[i];
	if (trainByClass[c].length < PER_CLASS) trainByClass[c].push(i);
}

const encoder = new MNISTEncoder(2, true);
const brain = new Brain({ contextLength: 30, mergeThreshold: 0.5, patternForgetRate: 0 });
encoder.registerChannels(brain);
const channelId = encoder.channelId;
const inputDimId = encoder.inputDimId;
const buildEvents = (bit) => { const m = new Map(); const dm = new Map(); dm.set(inputDimId, bit); m.set(channelId, dm); return m; };
const EMPTY = new Map();

const trainCtxs = [];
for (let c = 0; c < 10; c++) for (const idx of trainByClass[c]) trainCtxs.push({ idx, label: c });
const N = trainCtxs.length;
const trainBitsArr = trainCtxs.map(t => Array.from(encoder.buildBits(trainImages[t.idx])));

// Test image to inspect
const TEST_IDX = 8; // test5[8] from earlier — 28/29 voters in class-0 pool
const testBits = Array.from(encoder.buildBits(testImages[TEST_IDX]));
const maxLen = Math.max(...trainBitsArr.map(b => b.length), testBits.length);

brain.initContexts(N);
brain.setProcessingMode(true, false, true);
const t0 = Date.now();
for (let pos = 0; pos < maxLen; pos++) {
	for (let i = 0; i < N; i++) {
		if (pos >= trainBitsArr[i].length) continue;
		brain.setActiveContext(i);
		brain.processFrame(buildEvents(trainBitsArr[i][pos]), EMPTY, EMPTY);
	}
}
console.log(`Trained in ${Date.now() - t0}ms. Neurons: ${brain.getFrameSummary().neuronCount}`);

// Cold-replay the test image
brain.setProcessingMode(true, false, false);
brain.setActiveContext(0);
brain.resetContext();
for (let pos = 0; pos < testBits.length; pos++) {
	brain.processFrame(buildEvents(testBits[pos]), EMPTY, EMPTY);
}
const testVoters = brain.getVotableEntries();
console.log(`\nTest-5[${TEST_IDX}] cold-replay end voters: ${testVoters.length}`);

// Find each test-voter's level, parent, and stored context
console.log(`\nPer-voter detail (id, age, level, parent, context entries):`);
const voterDetails = [];
for (const v of testVoters) {
	const info = brain.inspectNeuron(v.neuronId);
	voterDetails.push({
		id: v.neuronId,
		age: v.age,
		level: info.level,
		parent: info.parentId,
		context: info.context,
	});
}

// Sort by level descending — look at deepest patterns first
voterDetails.sort((a, b) => b.level - a.level || a.age - b.age);

for (const v of voterDetails.slice(0, 15)) {
	console.log(`\n  voter ${v.id} (L${v.level}, age=${v.age}, parent=${v.parent}):`);
	console.log(`    stored context (${v.context.length} entries):`);
	for (const e of v.context.slice(0, 10)) {
		const ei = brain.inspectNeuron(e.neuronId);
		console.log(`      neuron ${e.neuronId} @ distance=${e.distance}  (L${ei.level})`);
	}
	if (v.context.length > 10) console.log(`      ... and ${v.context.length - 10} more`);
}
