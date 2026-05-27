/**
 * One-off: dump all connections of a specific parent neuron grouped by
 * distance to see argmax/margin per slot. Used to investigate burst tipping.
 *
 * Usage: node apps/mnist/jobs/parent_conn_dump.js --parent N [--max-images N]
 *        [--episodes N] [--target-a 11 --target-b 12]
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

const PARENT = num('--parent', 224);
const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 20);
const TGT_A = num('--target-a', 11);
const TGT_B = num('--target-b', 12);

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));

const brain = new Brain({ contextLength: 28, mergeThreshold: 0.99, patternForgetRate: 0 });
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

const conns = brain.dumpNeuronConnections(PARENT);
const byDist = new Map();
for (const c of conns) {
	if (!byDist.has(c.distance)) byDist.set(c.distance, []);
	byDist.get(c.distance).push(c);
}

console.log(`Parent #${PARENT} all connections after ${EPISODES} episodes (${MAX_IMAGES} images):`);
console.log(`  dist | tgt=${TGT_A} (str/rew) | tgt=${TGT_B} (str/rew) | other targets | argmax | margin`);
for (const dist of [...byDist.keys()].sort((a, b) => a - b)) {
	const arr = byDist.get(dist);
	const cA = arr.find(x => x.targetId === TGT_A) || { strength: 0, reward: 0 };
	const cB = arr.find(x => x.targetId === TGT_B) || { strength: 0, reward: 0 };
	const others = arr.filter(x => x.targetId !== TGT_A && x.targetId !== TGT_B);
	const argmaxId = cA.strength > cB.strength ? TGT_A : cB.strength > cA.strength ? TGT_B : 'TIE';
	const margin = Math.abs(cA.strength - cB.strength);
	const othersStr = others.length === 0 ? '—' : others.map(o => `${o.targetId}:${o.strength.toFixed(0)}`).join(',');
	console.log(`  ${String(dist).padStart(4)} | ${(cA.strength.toFixed(0) + '/' + cA.reward.toFixed(2)).padStart(18)} | ${(cB.strength.toFixed(0) + '/' + cB.reward.toFixed(2)).padStart(18)} | ${othersStr.padStart(15)} | ${String(argmaxId).padStart(6)} | ${margin.toFixed(0)}`);
}
