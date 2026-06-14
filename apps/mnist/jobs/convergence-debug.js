// Tiny-set convergence probe: 10 images, train for many episodes, report train accuracy each episode.
// If the architecture is sound it MUST converge to 100% on the training set.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PER_CLASS = parseInt(process.argv[2] || '1');
const EPISODES = parseInt(process.argv[3] || '50');
const MERGE_TH = parseFloat(process.argv[4] || '0.5');
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
	mergeThreshold: MERGE_TH,
});
const encoder = new MNISTPixelChannelsEncoder(2, 7, 1);
encoder.registerChannels(brain);

console.log(`Training set: ${trainBits.length} images (${PER_CLASS}/class), ${EPISODES} episodes, merge=${MERGE_TH}`);
console.log('');

for (let ep = 1; ep <= EPISODES; ep++) {
	let correct = 0;
	for (let i = 0; i < trainBits.length; i++) {
		brain.resetContext();
		const r = brain.processFrame(encoder.encodeImage(trainBits[i]), EMPTY_REWARDS);
		const pred = encoder.decodeDigit(r.inferences);
		if (pred === trainL[i]) correct++;
		brain.learn(encoder.encodeAction(trainL[i]), 1);
	}
	const levels = brain.spatialLevelCounts();
	const total = levels.reduce((a, b) => a + b, 0);
	console.log(`ep ${String(ep).padStart(2)}: train=${correct}/${trainBits.length} (${(correct/trainBits.length*100).toFixed(1)}%) depth=${levels.length} total=${total} levels=[${levels.join(',')}]`);
}

// Final test: present each training image and check prediction (read-only inference)
brain.setLearning(false);
let correct = 0;
const finalPreds = [];
for (let i = 0; i < trainBits.length; i++) {
	brain.resetContext();
	const r = brain.processFrame(encoder.encodeImage(trainBits[i]), EMPTY_REWARDS);
	const pred = encoder.decodeDigit(r.inferences);
	finalPreds.push(pred);
	if (pred === trainL[i]) correct++;
}
console.log('');
console.log(`Final (no-learn) pass: ${correct}/${trainBits.length} preds=[${finalPreds.join(',')}] labels=[${trainL.join(',')}]`);
const finalLevels = brain.spatialLevelCounts();
console.log(`Final levels: [${finalLevels.join(',')}], depth=${finalLevels.length}`);
