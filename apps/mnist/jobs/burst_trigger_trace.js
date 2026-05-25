/**
 * Burst Trigger Trace — train N images for many episodes; after each episode
 * record which parent neurons spawned new children; identify parents with
 * long quiet gaps between birth events (these are the tipping-point parents
 * causing bursts after stable periods); dump their event-connection state
 * across episodes to see what drifted.
 *
 * Usage: node apps/mnist/jobs/burst_trigger_trace.js [--max-images N]
 *        [--episodes N] [--context-length N] [--merge-threshold N]
 *        [--forget-rate N] [--gap-threshold N] [--dump-top N]
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
const str = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : d; };

const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 25);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const GAP_THRESHOLD = num('--gap-threshold', 5);
const DUMP_TOP = num('--dump-top', 2);
const TRIM = !process.argv.includes('--no-trim');

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

const samples = [];
for (let i = 0; i < MAX_IMAGES; i++) samples.push({ idx: i, pixels: images[i], label: labels[i] });

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE
});
const encoder = new MNISTEncoder(2, TRIM);
encoder.registerChannels(brain);
const allBits = samples.map(s => encoder.buildBits(s.pixels));

console.log('Burst Trigger Trace');
console.log(`  ${samples.length} samples: ${samples.map((s, i) => `lbl=${s.label}(${allBits[i].length}b)`).join(' ')}`);
console.log(`  ctx=${CONTEXT_LENGTH} merge=${MERGE_THRESHOLD} forget=${FORGET_RATE} errMode=${ERROR_MODE}`);
console.log('');

let shuttingDown = false;
process.on('SIGINT', () => { if (shuttingDown) process.exit(1); shuttingDown = true; });

function feedBits(bits, learning) {
	brain.resetContext();
	brain.resetAccuracyStats();
	encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	for (let i = 0; i < bits.length; i++) {
		if (shuttingDown) return;
		brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
	}
}

/**
 * Snapshot a neuron's event connections (target -> {strength, reward} per
 * distance). Used to diff across episodes for tipping-point neurons.
 */
function snapshotConnections(neuronId) {
	const conns = brain.dumpNeuronConnections(neuronId);
	// key: `${distance}:${targetId}` -> { strength, reward }
	const map = new Map();
	for (const c of conns) {
		map.set(`${c.distance}:${c.targetId}`, { strength: c.strength, reward: c.reward });
	}
	return map;
}

/**
 * Compute the diff between two snapshots: for each key (distance:target),
 * report the strength and reward change.
 */
function diffSnapshots(before, after) {
	const allKeys = new Set([...before.keys(), ...after.keys()]);
	const changes = [];
	for (const k of allKeys) {
		const b = before.get(k) || { strength: 0, reward: 0 };
		const a = after.get(k) || { strength: 0, reward: 0 };
		const ds = a.strength - b.strength;
		const dr = a.reward - b.reward;
		if (Math.abs(ds) > 1e-6 || Math.abs(dr) > 1e-6) {
			changes.push({ k, before: b, after: a, ds, dr });
		}
	}
	return changes;
}

// ── Train + snapshot ────────────────────────────────────────────────────────

// parent_id -> array of episodes in which they spawned a new child
const birthEpisodes = new Map();
// episode -> set of parent ids that fired this episode (gained ≥1 new child)
const perEpisodeParents = [];
// Per episode connection snapshots for ALL parents that have ever fired.
// We snapshot ALL their connections every episode so we can diff later.
// Map: parent_id -> Map<episode, snapshot>
const connSnapshots = new Map();

let lastNeuronId = 0;
let prevTotal = 0;

console.log('Episode | total neurons | new this ep | parents that fired');

for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	for (const bits of allBits) feedBits(bits, true);
	const total = brain.getFrameSummary().neuronCount;

	// Find new patterns and their parents.
	const firingParents = new Set();
	for (let id = lastNeuronId + 1; id <= total; id++) {
		const info = brain.inspectNeuron(id);
		if (info.level === 0) continue;
		firingParents.add(info.parentId);
		const arr = birthEpisodes.get(info.parentId) || [];
		arr.push(ep);
		birthEpisodes.set(info.parentId, arr);
	}
	lastNeuronId = total;
	perEpisodeParents.push({ ep, total, newOnes: total - prevTotal, firingParents });
	prevTotal = total;

	// Snapshot ALL ever-fired parents' connections this episode (small enough
	// for our use case since most parents are small-fanout L0/L1 neurons).
	for (const pid of birthEpisodes.keys()) {
		const map = connSnapshots.get(pid) || new Map();
		map.set(ep, snapshotConnections(pid));
		connSnapshots.set(pid, map);
	}

	const newOnes = perEpisodeParents[ep - 1].newOnes;
	const burstMark = newOnes > 20 ? ' ★BURST' : '';
	console.log(`  ${String(ep).padStart(5)} | ${String(total).padStart(13)} | ${String(newOnes).padStart(11)} | ${String(firingParents.size).padStart(18)}${burstMark}`);
}

// ── Identify tipping-point parents ─────────────────────────────────────────

// Parents whose birth events have a gap >= GAP_THRESHOLD episodes followed
// by a re-firing. Those are the ones that "wake up" after a quiet period.
const tippingPoints = [];
for (const [pid, eps] of birthEpisodes) {
	for (let i = 1; i < eps.length; i++) {
		const gap = eps[i] - eps[i - 1];
		if (gap >= GAP_THRESHOLD) {
			tippingPoints.push({ pid, lastQuietEp: eps[i - 1], reFireEp: eps[i], gap });
		}
	}
}
tippingPoints.sort((a, b) => b.gap - a.gap);

console.log(`\nTipping-point parents (gap ≥ ${GAP_THRESHOLD} episodes between birth events): ${tippingPoints.length}`);
for (let i = 0; i < Math.min(10, tippingPoints.length); i++) {
	const tp = tippingPoints[i];
	const parentInfo = brain.inspectNeuron(tp.pid);
	console.log(`  parent #${tp.pid} (L${parentInfo.level}): silent ep${tp.lastQuietEp + 1}..${tp.reFireEp - 1}, refired ep${tp.reFireEp} (gap=${tp.gap})`);
}

// ── Diff connections for top tipping-point parents ─────────────────────────

console.log(`\nFor the top ${DUMP_TOP} tipping-point parents, diff connections from quiet→fire:`);
for (let i = 0; i < Math.min(DUMP_TOP, tippingPoints.length); i++) {
	const tp = tippingPoints[i];
	const snaps = connSnapshots.get(tp.pid);
	const before = snaps.get(tp.reFireEp - 1);
	const after = snaps.get(tp.reFireEp);
	if (!before || !after) continue;

	const changes = diffSnapshots(before, after);
	console.log(`\n— parent #${tp.pid}: ep${tp.reFireEp - 1} → ep${tp.reFireEp} (just before fire → just after)`);
	console.log(`  ${changes.length} connection slot(s) changed`);
	const significant = changes.filter(c => Math.abs(c.ds) >= 1 || Math.abs(c.dr) > 0.05);
	console.log(`  ${significant.length} significant (|Δstrength|≥1 OR |Δreward|>0.05):`);
	for (const c of significant.slice(0, 15)) {
		console.log(`    ${c.k.padStart(8)}  strength ${c.before.strength.toFixed(1)} → ${c.after.strength.toFixed(1)} (Δ${c.ds >= 0 ? '+' : ''}${c.ds.toFixed(1)})  reward ${c.before.reward.toFixed(3)} → ${c.after.reward.toFixed(3)} (Δ${c.dr >= 0 ? '+' : ''}${c.dr.toFixed(3)})`);
	}

	// Also show what was drifting during the QUIET period leading up to the fire.
	// Pick the snapshot from 5 episodes before reFireEp to show longer drift.
	const longBefore = snaps.get(Math.max(1, tp.reFireEp - 5));
	if (longBefore) {
		const longChanges = diffSnapshots(longBefore, before);
		const longSig = longChanges.filter(c => Math.abs(c.ds) >= 0.1 || Math.abs(c.dr) > 0.01);
		console.log(`  during the QUIET period (ep${Math.max(1, tp.reFireEp - 5)} → ep${tp.reFireEp - 1}, ${longSig.length} slots drifted):`);
		for (const c of longSig.slice(0, 10)) {
			console.log(`    ${c.k.padStart(8)}  strength ${c.before.strength.toFixed(1)} → ${c.after.strength.toFixed(1)} (Δ${c.ds >= 0 ? '+' : ''}${c.ds.toFixed(2)})  reward ${c.before.reward.toFixed(3)} → ${c.after.reward.toFixed(3)} (Δ${c.dr >= 0 ? '+' : ''}${c.dr.toFixed(3)})`);
		}
	}
}

console.log('\nDone.');
