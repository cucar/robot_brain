/**
 * Inspect a pair of sibling pattern neurons after training.
 * Shows each's stored context (with strengths) so we can compare match scores.
 *
 * Usage: node apps/mnist/jobs/inspect_pair.js --a N --b N
 *        [--max-images N] [--episodes N] [--merge-threshold N]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

const num = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseInt(process.argv[i + 1]) : d; };
const numFloat = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : d; };

const A = num('--a', 2704);
const B = num('--b', 3818);
const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 4);
const MERGE = numFloat('--merge-threshold', 1);

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));

const brain = new Brain({ contextLength: 28, mergeThreshold: MERGE, patternForgetRate: 0 });
const encoder = new MNISTEncoder(2, true);
encoder.registerChannels(brain);
const allBits = [];
for (let i = 0; i < MAX_IMAGES; i++) allBits.push(encoder.buildBits(images[i]));

function feed(bits, learning) {
	brain.resetContext(); brain.resetAccuracyStats(); encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	for (let i = 0; i < bits.length; i++) brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
}

for (let ep = 1; ep <= EPISODES; ep++) for (const bits of allBits) feed(bits, true);

for (const id of [A, B]) {
	const info = brain.inspectNeuron(id);
	console.log(`\nNeuron #${id} L${info.level} parent=${info.parentId}`);
	console.log(`  context (${info.context.length} entries):`);
	const sorted = [...info.context].sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId);
	for (const c of sorted) console.log(`    ${c.neuronId}@d${c.distance}  strength=${c.strength}`);
}
