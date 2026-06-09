// 7×7 binary, 1000/class × 5 ep — fast config (~1m37s previously). Reports spatial level
// distribution so we can see how deep the hierarchy goes.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PER_CLASS = 1000;
const EPISODES = 5;
const EMPTY_REWARDS = new Map();
const dataDir = path.join(__dirname, '..', 'data');

const trainAll = loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz'));
const trainLab = loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz'));

const tmpEnc = new MNISTPixelChannelsEncoder(2, 7);
const picked = new Array(10).fill(0);
const trainIdx = [];
for (let i = 0; i < trainAll.length && trainIdx.length < PER_CLASS * 10; i++) {
	if (picked[trainLab[i]] < PER_CLASS) { trainIdx.push(i); picked[trainLab[i]]++; }
}
trainIdx.sort((a, b) => a - b);
const trainBits = trainIdx.map(i => tmpEnc.buildBits(trainAll[i]));
const trainL = trainIdx.map(i => trainLab[i]);

const brain = new Brain({
	contextLength: 1, patternForgetRate: 0, columns: 20,
	errorCorrectionMode: 'aggressive', errorCorrectionThreshold: 0.5, mergeThreshold: 0.9,
});
const encoder = new MNISTPixelChannelsEncoder(2, 7, 1);
encoder.registerChannels(brain);

const start = Date.now();
for (let ep = 1; ep <= EPISODES; ep++) {
	for (let i = 0; i < trainBits.length; i++) {
		brain.resetContext();
		brain.processFrame(encoder.encodeImage(trainBits[i]), EMPTY_REWARDS);
		brain.learn(encoder.encodeAction(trainL[i]), 1);
	}
	const counts = brain.spatialLevelCounts();
	const depth = counts.length;
	const total = counts.reduce((a, b) => a + b, 0);
	console.log(`Episode ${ep}: maxSpatialLevel=${depth}, total=${total}, by level: [${counts.join(', ')}]`);
}

console.log('');
console.log('=== Final spatial level distribution ===');
const counts = brain.spatialLevelCounts();
for (let i = 0; i < counts.length; i++) {
	const lvl = i + 1;
	const max = Math.max(...counts);
	const bar = '█'.repeat(Math.min(60, Math.floor(counts[i] / max * 60)));
	console.log(`  L${lvl}: ${String(counts[i]).padStart(6)}  ${bar}`);
}
console.log(`Max depth: ${counts.length}, total active: ${counts.reduce((a, b) => a + b, 0)}`);
console.log(`Runtime: ${((Date.now() - start) / 1000).toFixed(1)}s`);
