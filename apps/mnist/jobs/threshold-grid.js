// Grid over error/merge thresholds AND error mode (static + adaptive) on MNIST 7×7 binary with
// 3×3 pixel neighborhoods. The post-neighbor error distribution is wider (mean 0.378, std 0.083),
// so adaptive modes finally have dynamic range to work with.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PER_CLASS = 100;
const TEST_IMAGES = 200;
const EPISODES = 2;
const RADIUS = 1;  // 3×3 neighborhoods — best from prior grid
const EMPTY_REWARDS = new Map();
const dataDir = path.join(__dirname, '..', 'data');

const trainAll = loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz'));
const trainLab = loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz'));
const testAll = loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz'));
const testLab = loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz'));

const tmpEnc = new MNISTPixelChannelsEncoder(2, 7);
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

async function runConfig(mode, errTh, mergeTh) {
	const start = Date.now();
	const brain = new Brain({
		contextLength: 1, patternForgetRate: 0, columns: 20,
		errorCorrectionMode: mode, errorCorrectionThreshold: errTh, mergeThreshold: mergeTh,
	});
	const encoder = new MNISTPixelChannelsEncoder(2, 7, RADIUS);
	encoder.registerChannels(brain);

	let lastTrain = 0;
	for (let ep = 1; ep <= EPISODES; ep++) {
		let correct = 0;
		for (let i = 0; i < trainBits.length; i++) {
			brain.resetContext();
			const inf = brain.processFrame(encoder.encodeImage(trainBits[i]), EMPTY_REWARDS);
			const pred = encoder.decodeDigit(inf.inferences);
			if (pred === trainL[i]) correct++;
			brain.learn(encoder.encodeAction(trainL[i]), 1);
		}
		lastTrain = correct / trainBits.length;
	}
	brain.setLearning(false);
	let testCorrect = 0;
	for (let i = 0; i < testBits.length; i++) {
		brain.resetContext();
		const inf = brain.processFrame(encoder.encodeImage(testBits[i]), EMPTY_REWARDS);
		const pred = encoder.decodeDigit(inf.inferences);
		if (pred === testL[i]) testCorrect++;
	}
	const testAcc = testCorrect / testBits.length;
	const mints = brain.getSpatialCorrectionCount();
	const elapsed = (Date.now() - start) / 1000;
	return { mode, errTh, mergeTh, train: lastTrain, test: testAcc, mints, elapsed };
}

const mergeThresholds = [0.5, 0.7, 0.9];
const configs = [];

// Static rows — fixed threshold across training.
for (const errTh of [0.40, 0.45, 0.50]) {
	for (const m of mergeThresholds) configs.push({ mode: 'static', errTh, mergeTh: m });
}

// Adaptive rows — warmup fallback is the static threshold used until Welford has ≥3 samples per
// (neuron, age). Set warmup = 0.50 (above the bulk) so early mints don't flood before adaptive
// kicks in.
for (const mode of ['conservative', 'neutral', 'aggressive']) {
	for (const m of mergeThresholds) configs.push({ mode, errTh: 0.50, mergeTh: m });
}

console.log(`Threshold grid — 7×7 binary, ${trainBits.length} train, ${testBits.length} test, ${EPISODES} eps, r=${RADIUS}`);
console.log('');

const results = [];
for (const cfg of configs) {
	const r = await runConfig(cfg.mode, cfg.errTh, cfg.mergeTh);
	results.push(r);
	const label = `${cfg.mode} err=${cfg.errTh.toFixed(2)} merge=${cfg.mergeTh.toFixed(1)}`;
	console.log(`${label.padEnd(34)}: train=${(r.train * 100).toFixed(1)}% test=${(r.test * 100).toFixed(1)}% mints=${String(r.mints).padStart(6)} ${r.elapsed.toFixed(1)}s`);
}

console.log('');
console.log('=== Test accuracy table ===');
let header = 'mode/err         ';
for (const m of mergeThresholds) header += ` merge=${m.toFixed(1)}`;
console.log(header);

const rows = [
	{ label: 'static err=0.40', mode: 'static', errTh: 0.40 },
	{ label: 'static err=0.45', mode: 'static', errTh: 0.45 },
	{ label: 'static err=0.50', mode: 'static', errTh: 0.50 },
	{ label: 'conservative     ', mode: 'conservative', errTh: 0.50 },
	{ label: 'neutral          ', mode: 'neutral', errTh: 0.50 },
	{ label: 'aggressive       ', mode: 'aggressive', errTh: 0.50 },
];
for (const row of rows) {
	let line = row.label.padEnd(17);
	for (const m of mergeThresholds) {
		const r = results.find(x => x.mode === row.mode && x.errTh === row.errTh && x.mergeTh === m);
		line += `   ${(r.test * 100).toFixed(1)}%   `;
	}
	console.log(line);
}
console.log('');
console.log(`Total time: ${results.reduce((s, r) => s + r.elapsed, 0).toFixed(1)}s`);
console.log('Reference: sensors-only baseline (no mints) gave ~55% test on this config.');
