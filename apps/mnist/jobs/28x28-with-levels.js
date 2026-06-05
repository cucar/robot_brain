// 28x28 binary MNIST with neighbors + adaptive errors. Moderate config so we get a result in
// reasonable wall time. Reports per-episode train accuracy, final test, and the spatial level
// distribution (how deep the pattern hierarchy goes).
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PER_CLASS = 300;
const TEST_IMAGES = 500;
const EPISODES = 3;
const IMAGE_SIZE = 28;
const BUCKETS = 2;
const RADIUS = 1;
const EMPTY_REWARDS = new Map();
const dataDir = path.join(__dirname, '..', 'data');

const trainAll = loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz'));
const trainLab = loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz'));
const testAll = loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz'));
const testLab = loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz'));

const tmpEnc = new MNISTPixelChannelsEncoder(BUCKETS, IMAGE_SIZE);
const picked = new Array(10).fill(0);
const trainIdx = [];
for (let i = 0; i < trainAll.length && trainIdx.length < PER_CLASS * 10; i++) {
	if (picked[trainLab[i]] < PER_CLASS) { trainIdx.push(i); picked[trainLab[i]]++; }
}
trainIdx.sort((a, b) => a - b);
const trainBits = trainIdx.map(i => tmpEnc.buildBits(trainAll[i]));
const trainL = trainIdx.map(i => trainLab[i]);
const testBits = testAll.slice(0, TEST_IMAGES).map(p => tmpEnc.buildBits(p));
const testL = testLab.slice(0, TEST_IMAGES);

const brain = new Brain({
	contextLength: 1, patternForgetRate: 0, columns: 20,
	errorCorrectionMode: 'aggressive', errorCorrectionThreshold: 0.5, mergeThreshold: 0.9,
});
const encoder = new MNISTPixelChannelsEncoder(BUCKETS, IMAGE_SIZE, RADIUS);
encoder.registerChannels(brain);

console.log(`MNIST 28×28 binary — ${trainBits.length} train × ${EPISODES} ep, ${testBits.length} test, r=${RADIUS}, aggressive err=0.50 merge=0.9`);
console.log('');

const start = Date.now();
for (let ep = 1; ep <= EPISODES; ep++) {
	const epStart = Date.now();
	let correct = 0;
	for (let i = 0; i < trainBits.length; i++) {
		brain.resetContext();
		brain.processFrame(encoder.encodeImage(trainBits[i]), EMPTY_REWARDS);
		const inf = brain.infer();
		const pred = encoder.decodeDigit(inf.inferences);
		if (pred === trainL[i]) correct++;
		brain.learn(encoder.encodeAction(trainL[i]), 1);
	}
	const epDuration = (Date.now() - epStart) / 1000;
	const trainAcc = correct / trainBits.length;
	const mints = brain.getSpatialCorrectionCount();
	const ips = (trainBits.length / epDuration).toFixed(0);
	console.log(`Episode ${ep}: train=${(trainAcc * 100).toFixed(2)}% mints=${mints} ${epDuration.toFixed(1)}s (${ips} img/s)`);
}

console.log('');
brain.setLearning(false);
const testStart = Date.now();
let testCorrect = 0;
const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
const perDigit = new Array(10).fill(0);
const perDigitTotal = new Array(10).fill(0);
for (let i = 0; i < testBits.length; i++) {
	brain.resetContext();
	brain.processFrame(encoder.encodeImage(testBits[i]), EMPTY_REWARDS);
	const inf = brain.infer();
	const pred = encoder.decodeDigit(inf.inferences);
	perDigitTotal[testL[i]]++;
	if (pred === testL[i]) { testCorrect++; perDigit[testL[i]]++; }
	if (pred >= 0 && pred < 10) confusion[testL[i]][pred]++;
}
const testDuration = (Date.now() - testStart) / 1000;
const testAcc = testCorrect / testBits.length;
console.log(`Test: ${(testAcc * 100).toFixed(2)}% (${testCorrect}/${testBits.length}) ${testDuration.toFixed(1)}s`);
console.log('Per-digit: ' + perDigit.map((c, d) => `${d}:${perDigitTotal[d] > 0 ? (c / perDigitTotal[d] * 100).toFixed(0) : '–'}%`).join(' '));

console.log('');
console.log('=== Spatial level distribution ===');
const levelCounts = brain.spatialLevelCounts();
if (levelCounts.length === 0) {
	console.log('  No spatial corrections at any level.');
} else {
	for (let i = 0; i < levelCounts.length; i++) {
		const lvl = i + 1;
		const bar = '█'.repeat(Math.min(60, Math.floor(levelCounts[i] / Math.max(...levelCounts) * 60)));
		console.log(`  L${lvl}: ${String(levelCounts[i]).padStart(6)}  ${bar}`);
	}
	console.log(`  Max spatial level reached: ${levelCounts.length}`);
	console.log(`  Total active corrections: ${levelCounts.reduce((a, b) => a + b, 0)}`);
}

console.log('');
console.log('=== Confusion (rows=actual, cols=predicted) ===');
let header = '       ';
for (let p = 0; p < 10; p++) header += String(p).padStart(5);
console.log(header);
for (let a = 0; a < 10; a++) {
	let row = `   ${a}   `;
	for (let p = 0; p < 10; p++) row += String(confusion[a][p]).padStart(5);
	console.log(row);
}

console.log('');
console.log(`Total runtime: ${((Date.now() - start) / 1000).toFixed(1)}s`);
