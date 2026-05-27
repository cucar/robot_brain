/**
 * Dump pattern context for #3201 and #3838 (or any pair) at ep4 vs ep5 to
 * see what changed that flipped which one wins the match.
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

const A = num('--a', 3201);
const B = num('--b', 3838);
const MAX_IMAGES = num('--max-images', 5);
const EPISODES_BEFORE = num('--before', 4);
const EPISODES_AFTER = num('--after', 5);

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));

const brain = new Brain({ contextLength: 30, mergeThreshold: 1, patternForgetRate: 0 });
const encoder = new MNISTEncoder(2, true);
encoder.registerChannels(brain);
const allBits = [];
for (let i = 0; i < MAX_IMAGES; i++) allBits.push(encoder.buildBits(images[i]));

function feed(bits, learning) {
	brain.resetContext(); brain.resetAccuracyStats(); encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	for (let i = 0; i < bits.length; i++) brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
}

function dump(label) {
	for (const id of [A, B]) {
		const info = brain.inspectNeuron(id);
		const ctx = info.context.sort((x, y) => x.distance - y.distance).map(c => `${c.neuronId}@${c.distance}(s${c.strength.toFixed(0)})`).join(' ');
		const totalStrength = info.context.reduce((s, c) => s + c.strength, 0);
		console.log(`  [${label}] #${id}: L${info.level} parent=${info.parentId} entries=${info.context.length} totalStrength=${totalStrength.toFixed(0)}`);
		console.log(`    ctx: ${ctx}`);
	}
}

for (let ep = 1; ep <= EPISODES_BEFORE; ep++) for (const bits of allBits) feed(bits, true);
console.log(`After ep${EPISODES_BEFORE}: ${brain.getFrameSummary().neuronCount} neurons`);
dump(`ep${EPISODES_BEFORE}`);

for (let ep = EPISODES_BEFORE + 1; ep <= EPISODES_AFTER; ep++) for (const bits of allBits) feed(bits, true);
console.log(`\nAfter ep${EPISODES_AFTER}: ${brain.getFrameSummary().neuronCount} neurons`);
dump(`ep${EPISODES_AFTER}`);
