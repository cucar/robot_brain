// Run a single config and print the resulting spatial level distribution. Args: mode err merge.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const [, , mode, errArg, mergeArg] = process.argv;
const errTh = parseFloat(errArg);
const mergeTh = parseFloat(mergeArg);

const PER_CLASS = 500;
const EPISODES = 3;
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
	errorCorrectionMode: mode, errorCorrectionThreshold: errTh, mergeThreshold: mergeTh,
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
}
const counts = brain.spatialLevelCounts();
const elapsed = ((Date.now() - start) / 1000).toFixed(1);
const summary = counts.length === 0 ? 'no mints' : counts.map((n, i) => `L${i + 1}=${n}`).join(' ');
console.log(`${mode.padEnd(12)} err=${errTh.toFixed(2)} merge=${mergeTh.toFixed(1)}: depth=${counts.length}, ${summary} (${elapsed}s)`);
