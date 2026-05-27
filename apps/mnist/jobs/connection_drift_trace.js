/**
 * Connection Drift Trace — for one specific neuron, snapshot its outgoing
 * event connections per episode and report the strengths/rewards over time,
 * highlighting connections that get DELETED (drifted to strength 0).
 *
 * Usage:
 *   node apps/mnist/jobs/connection_drift_trace.js --neuron N
 *     [--max-images N] [--episodes N]
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Brain} from 'robot-brain';
import {MNISTEncoder} from '../encoders/mnist_encoder.js';
import {loadImages} from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');

const num = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseInt(process.argv[i + 1]) : d; };

const NEURON = num('--neuron', 3626);
const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 22);

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

// Snap helper: key by `${distance}:${targetId}` → {strength, reward}
function snap() {
	const m = new Map();
	for (const c of brain.dumpNeuronConnections(NEURON)) {
		m.set(`${c.distance}:${c.targetId}`, { strength: c.strength, reward: c.reward });
	}
	return m;
}

const snaps = new Map();
console.log(`Tracking event connections of neuron #${NEURON} across ${EPISODES} episodes`);

for (let ep = 1; ep <= EPISODES; ep++) {
	for (const bits of allBits) feed(bits, true);
	snaps.set(ep, snap());
}

// Find all keys that ever appeared
const allKeys = new Set();
for (const s of snaps.values()) for (const k of s.keys()) allKeys.add(k);
const sortedKeys = [...allKeys].sort((a, b) => {
	const [ad, at] = a.split(':').map(Number);
	const [bd, bt] = b.split(':').map(Number);
	return ad - bd || at - bt;
});

console.log(`\n#${NEURON}: ${allKeys.size} distinct connection slots ever existed.`);

// Find connections that got DELETED (existed earlier, missing later) or that
// hit strength 0 at some point.
const deleted = [];
const drifted = [];
for (const k of sortedKeys) {
	let firstEp = null, lastEp = null, firstStr = null, lastStr = null;
	let minStr = Infinity, maxStr = -Infinity;
	for (let ep = 1; ep <= EPISODES; ep++) {
		const v = snaps.get(ep).get(k);
		if (v === undefined) continue;
		if (firstEp === null) { firstEp = ep; firstStr = v.strength; }
		lastEp = ep; lastStr = v.strength;
		if (v.strength < minStr) minStr = v.strength;
		if (v.strength > maxStr) maxStr = v.strength;
	}
	const presentAtEnd = snaps.get(EPISODES).has(k);
	if (!presentAtEnd) {
		deleted.push({ k, firstEp, lastEp, firstStr, lastStr, minStr, maxStr });
	} else if (lastStr < firstStr) {
		drifted.push({ k, firstEp, lastEp, firstStr, lastStr, minStr, maxStr });
	}
}

console.log(`\nDELETED connections (existed at some point, gone by ep${EPISODES}): ${deleted.length}`);
for (const d of deleted.slice(0, 20)) {
	console.log(`  slot ${d.k.padEnd(8)}  appeared ep${d.firstEp} (str=${d.firstStr}), last seen ep${d.lastEp} (str=${d.lastStr})`);
}

console.log(`\nDRIFTED connections (still alive but final strength < first): ${drifted.length}`);
for (const d of drifted.slice(0, 20)) {
	console.log(`  slot ${d.k.padEnd(8)}  ep${d.firstEp}→${d.lastEp}: str ${d.firstStr.toFixed(1)} → ${d.lastStr.toFixed(1)} (min=${d.minStr.toFixed(1)})`);
}

// Per-episode strength of each slot — show how the alive ones move over time.
console.log(`\nPer-episode strength of each connection slot:`);
console.log(`  slot   | ` + Array.from({ length: EPISODES }, (_, i) => `ep${i + 1}`.padStart(4)).join(' '));
for (const k of sortedKeys) {
	let row = `  ${k.padEnd(6)} | `;
	for (let ep = 1; ep <= EPISODES; ep++) {
		const v = snaps.get(ep).get(k);
		row += (v === undefined ? '   .' : v.strength.toFixed(0).padStart(4)) + ' ';
	}
	console.log(row);
}

console.log('\nDone.');
