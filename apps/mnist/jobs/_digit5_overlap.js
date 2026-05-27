/**
 * Pick 20 training-5s and 20 test-5s. Dump their bit streams as files for
 * inspection. Train the brain on the 20 training-5s in parallel contexts.
 * Then cold-replay each (both train and test) and capture end-of-replay
 * voters. Report:
 *   - bit-stream pairwise differences
 *   - per-image voter level distribution
 *   - voter ID overlap between training-class-5 and test-class-5
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoders/mnist_encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');
const outDir = path.join(__dirname, '_digit5');
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const trainImages = loadImages(findFile('train-images-idx3-ubyte'));
const trainLabels = loadLabels(findFile('train-labels-idx1-ubyte'));
const testImages = loadImages(findFile('t10k-images-idx3-ubyte'));
const testLabels = loadLabels(findFile('t10k-labels-idx1-ubyte'));

// Pick 20 of each
const N = 20;
const trainIdxs = [];
for (let i = 0; trainIdxs.length < N && i < trainLabels.length; i++) {
	if (trainLabels[i] === 5) trainIdxs.push(i);
}
const testIdxs = [];
for (let i = 0; testIdxs.length < N && i < testLabels.length; i++) {
	if (testLabels[i] === 5) testIdxs.push(i);
}

console.log(`Selected ${trainIdxs.length} training-5s: ${trainIdxs.join(',')}`);
console.log(`Selected ${testIdxs.length} test-5s    : ${testIdxs.join(',')}`);

const encoder = new MNISTEncoder(2, true);
const brain = new Brain({ contextLength: 30, mergeThreshold: 0.5, patternForgetRate: 0 });
encoder.registerChannels(brain);

const channelId = encoder.channelId;
const inputDimId = encoder.inputDimId;
const buildEvents = (bit) => { const m = new Map(); const dm = new Map(); dm.set(inputDimId, bit); m.set(channelId, dm); return m; };
const EMPTY = new Map();

// Build bits for all selected images and write to files
const trainBitsArr = trainIdxs.map(i => Array.from(encoder.buildBits(trainImages[i])));
const testBitsArr = testIdxs.map(i => Array.from(encoder.buildBits(testImages[i])));

function dumpImage(prefix, idx, bits) {
	fs.writeFileSync(path.join(outDir, `${prefix}_${idx}_bits.txt`), bits.join('\n'));
	const padded = bits.concat(new Array(Math.max(0, 784 - bits.length)).fill(0));
	const lines = [];
	for (let r = 0; r < 28; r++) {
		lines.push(padded.slice(r * 28, (r + 1) * 28).map(b => b ? '█' : '·').join(''));
	}
	lines.push(``);
	lines.push(`len=${bits.length} ones=${bits.filter(b => b).length}`);
	fs.writeFileSync(path.join(outDir, `${prefix}_${idx}_visual.txt`), lines.join('\n'));
}
trainBitsArr.forEach((b, k) => dumpImage('train5', trainIdxs[k], b));
testBitsArr.forEach((b, k) => dumpImage('test5', testIdxs[k], b));
console.log(`Wrote ${trainIdxs.length + testIdxs.length} bit/visual files to ${outDir}/`);

// Bit-stream stats
function avgPairwiseHamming(bitsArr) {
	let sum = 0, count = 0;
	for (let i = 0; i < bitsArr.length; i++) {
		for (let j = i + 1; j < bitsArr.length; j++) {
			const a = bitsArr[i], b = bitsArr[j];
			const len = Math.min(a.length, b.length);
			let d = Math.abs(a.length - b.length);
			for (let k = 0; k < len; k++) if (a[k] !== b[k]) d++;
			sum += d; count++;
		}
	}
	return count > 0 ? sum / count : 0;
}
const avgLenTrain = trainBitsArr.reduce((s, b) => s + b.length, 0) / trainBitsArr.length;
const avgLenTest = testBitsArr.reduce((s, b) => s + b.length, 0) / testBitsArr.length;
console.log(`\nBit-stream stats:`);
console.log(`  avg length: train=${avgLenTrain.toFixed(1)}, test=${avgLenTest.toFixed(1)}`);
console.log(`  avg pairwise Hamming distance: train↔train=${avgPairwiseHamming(trainBitsArr).toFixed(1)}, test↔test=${avgPairwiseHamming(testBitsArr).toFixed(1)}`);

// ── Train brain on training-5s in parallel contexts ────────────────────────
const maxLen = Math.max(...trainBitsArr.map(b => b.length), ...testBitsArr.map(b => b.length));
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
console.log(`\nTrained ${N} contexts in ${Date.now() - t0}ms. Neurons: ${brain.getFrameSummary().neuronCount}`);

// Capture each training context's end-of-training voter set (the patterns
// that fire at end of each training image's solo trajectory).
function captureVotersAtCtx(ctxId) {
	brain.setActiveContext(ctxId);
	const v = brain.getVotableEntries();
	return v.map(x => ({ id: x.neuronId, age: x.age, level: brain.inspectNeuron(x.neuronId).level }));
}

const trainVoters = [];
for (let i = 0; i < N; i++) trainVoters.push(captureVotersAtCtx(i));

// ── Cold-replay each test-5 in a fresh context, capture voters ─────────────
brain.setProcessingMode(true, false, false);
const testVoters = [];
for (let k = 0; k < N; k++) {
	brain.setActiveContext(0);
	brain.resetContext();
	const bits = testBitsArr[k];
	for (let pos = 0; pos < bits.length; pos++) {
		brain.processFrame(buildEvents(bits[pos]), EMPTY, EMPTY);
	}
	const v = brain.getVotableEntries();
	testVoters.push(v.map(x => ({ id: x.neuronId, age: x.age, level: brain.inspectNeuron(x.neuronId).level })));
}

// ── Compare voter sets ─────────────────────────────────────────────────────
function levelStr(voters) {
	const m = new Map();
	for (const v of voters) m.set(v.level, (m.get(v.level) || 0) + 1);
	return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([l, c]) => `L${l}:${c}`).join(' ');
}

console.log(`\nTraining-5 voter level distributions (end-of-training, post-learn state):`);
for (let i = 0; i < N; i++) console.log(`  train5[${trainIdxs[i]}] voters=${trainVoters[i].length} (${levelStr(trainVoters[i])})`);

console.log(`\nTest-5 voter level distributions (cold-replay end state):`);
for (let k = 0; k < N; k++) console.log(`  test5[${testIdxs[k]}] voters=${testVoters[k].length} (${levelStr(testVoters[k])})`);

// Pool all training-class-5 voter IDs into one set, then check how many of each
// test-5's voters are in that pool. This is the key generalization metric:
// "do test images activate the same neurons training images of the same class did?"
const trainPool = new Set();
for (const tv of trainVoters) for (const v of tv) trainPool.add(v.id);
console.log(`\nTraining-5 voter ID pool size: ${trainPool.size}`);

console.log(`\nTest-5 overlap with training-5 voter pool:`);
let totalOverlap = 0, totalVoters = 0;
for (let k = 0; k < N; k++) {
	const tv = testVoters[k];
	const inPool = tv.filter(v => trainPool.has(v.id)).length;
	console.log(`  test5[${testIdxs[k]}]: ${inPool} / ${tv.length} voters in training-5 pool (${(inPool / tv.length * 100).toFixed(1)}%)`);
	totalOverlap += inPool; totalVoters += tv.length;
}
console.log(`  TOTAL: ${totalOverlap} / ${totalVoters} = ${(totalOverlap / totalVoters * 100).toFixed(2)}% of test-5 voters overlap with training-5 voter pool`);

// Also: per-level overlap rate
console.log(`\nPer-level overlap rate (test-5 voter is in training-5 pool):`);
const perLevelTotal = new Map(), perLevelInPool = new Map();
for (const tv of testVoters) {
	for (const v of tv) {
		perLevelTotal.set(v.level, (perLevelTotal.get(v.level) || 0) + 1);
		if (trainPool.has(v.id)) perLevelInPool.set(v.level, (perLevelInPool.get(v.level) || 0) + 1);
	}
}
const levels = [...new Set([...perLevelTotal.keys()])].sort((a, b) => a - b);
for (const l of levels) {
	const t = perLevelTotal.get(l) || 0;
	const p = perLevelInPool.get(l) || 0;
	console.log(`  L${l}: ${p}/${t} (${(p / t * 100).toFixed(1)}%)`);
}
