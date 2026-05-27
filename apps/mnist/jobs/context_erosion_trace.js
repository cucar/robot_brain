/**
 * Context Erosion Trace — train N images for many episodes; per episode
 * snapshot the stored context of every pattern neuron; identify a "burst"
 * episode (where new patterns are spawned after long quiet); then for the
 * specific parent that fired in the burst, dump per-episode context
 * evolution of all its sibling children — showing which entries got
 * weakened or deleted over time.
 *
 * The hypothesis: a sibling that was winning the match at some frame for
 * many episodes had its stored context eroded by refine_context's
 * weaken_neuron calls. Eventually a critical entry was deleted, the
 * sibling's match score dropped, and another sibling won — firing a wrong
 * prediction and triggering a burst.
 *
 * Usage:
 *   node apps/mnist/jobs/context_erosion_trace.js [--max-images N]
 *     [--episodes N] [--context-length N] [--merge-threshold N]
 *     [--forget-rate N] [--target-parent N]
 *
 * If --target-parent is given, dumps that specific neuron's children's
 * context evolution. Otherwise, finds the parent that fired in the latest
 * burst and dumps its children.
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
const numFloat = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : d; };
const str = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : d; };

const MAX_IMAGES = num('--max-images', 5);
const EPISODES = num('--episodes', 22);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const TARGET_PARENT = num('--target-parent', -1);
const TRIM = !process.argv.includes('--no-trim');

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE
});
const encoder = new MNISTEncoder(2, TRIM);
encoder.registerChannels(brain);
const allBits = [];
for (let i = 0; i < MAX_IMAGES; i++) allBits.push(encoder.buildBits(images[i]));

let shuttingDown = false;
process.on('SIGINT', () => { if (shuttingDown) process.exit(1); shuttingDown = true; });

function feed(bits, learning) {
	brain.resetContext(); brain.resetAccuracyStats(); encoder.resetActions();
	brain.setProcessingMode(true, false, learning);
	for (let i = 0; i < bits.length; i++) brain.processFrame(encoder.encodePixel(bits[i]), new Map(), new Map());
}

/**
 * Serialize a context entry list to a comparable string keyed by
 * "neuronId@distance" → strength.
 */
function ctxToMap(contextArr) {
	const m = new Map();
	for (const c of contextArr) m.set(`${c.neuronId}@${c.distance}`, c.strength);
	return m;
}

console.log(`Context Erosion Trace — ${MAX_IMAGES} images, ${EPISODES} episodes`);
console.log(`  ctx=${CONTEXT_LENGTH} merge=${MERGE_THRESHOLD} forget=${FORGET_RATE}`);
console.log('');

// Per-episode tracking. We snapshot ALL pattern (L>=1) contexts.
// Map: episode -> Map<neuron_id, contextMap>
const snapshots = new Map();
// neuron_id -> parent_id, level (cached)
const patternMeta = new Map();
// Map: parent_id -> set of episodes in which a new child was added under it
const fireEpisodes = new Map();
let lastNeuronId = 0;
let prevTotal = 0;

console.log('Episode | total | new');
for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	for (const bits of allBits) feed(bits, true);
	const total = brain.getFrameSummary().neuronCount;

	// detect new patterns and their parents
	for (let id = lastNeuronId + 1; id <= total; id++) {
		const info = brain.inspectNeuron(id);
		if (info.level === 0) continue;
		patternMeta.set(id, { parentId: info.parentId, level: info.level });
		const arr = fireEpisodes.get(info.parentId) || [];
		arr.push(ep);
		fireEpisodes.set(info.parentId, arr);
	}
	lastNeuronId = total;

	// Snapshot ALL L>=1 patterns' contexts this episode.
	const epSnap = new Map();
	for (const [id, _] of patternMeta) {
		const info = brain.inspectNeuron(id);
		epSnap.set(id, ctxToMap(info.context));
	}
	snapshots.set(ep, epSnap);

	const newOnes = total - prevTotal;
	prevTotal = total;
	const burst = newOnes >= 5 ? ' ★' : '';
	console.log(`  ${String(ep).padStart(5)} | ${String(total).padStart(5)} | ${String(newOnes).padStart(3)}${burst}`);
}

// ── Aggregate scan: did ANY pattern lose entries across the run? ───────────

console.log('\n── Aggregate erosion scan across ALL patterns ──');
const erodedPatterns = [];
for (const [pid, _] of patternMeta) {
	// find first and last snapshot for this pattern
	let firstSnap = null, firstEp = null, lastSnap = null, lastEp = null;
	for (let ep = 1; ep <= EPISODES; ep++) {
		const snap = snapshots.get(ep);
		if (!snap || !snap.has(pid)) continue;
		if (firstSnap === null) { firstSnap = snap.get(pid); firstEp = ep; }
		lastSnap = snap.get(pid); lastEp = ep;
	}
	if (!firstSnap || !lastSnap) continue;
	const removed = [...firstSnap.keys()].filter(k => !lastSnap.has(k));
	let weakened = 0;
	for (const [k, v] of firstSnap) {
		if (lastSnap.has(k) && lastSnap.get(k) < v) weakened++;
	}
	if (removed.length > 0 || weakened > 0) {
		erodedPatterns.push({ pid, removed: removed.length, weakened, firstEp, lastEp, removedKeys: removed });
	}
}
console.log(`  patterns with entries REMOVED or weakened: ${erodedPatterns.length} / ${patternMeta.size} total patterns`);
erodedPatterns.sort((a, b) => (b.removed + b.weakened) - (a.removed + a.weakened));
for (const e of erodedPatterns.slice(0, 10)) {
	const meta = patternMeta.get(e.pid);
	console.log(`  #${e.pid} (L${meta.level}, parent=${meta.parentId}): removed ${e.removed}, weakened ${e.weakened} entries (ep${e.firstEp}→${e.lastEp})`);
	if (e.removedKeys.length > 0) console.log(`    removed keys: ${e.removedKeys.join(', ')}`);
}

// ── Pick which parent to inspect ───────────────────────────────────────────

let parentId = TARGET_PARENT;
if (parentId < 0) {
	// Find latest burst episode and pick a parent that fired in it AND has a long quiet gap.
	let latestBurstEp = -1;
	for (let ep = EPISODES; ep >= 1; ep--) {
		const prev = snapshots.get(ep - 1);
		if (!prev) break;
		// parents whose # of children grew this ep
		const parentsFired = new Set();
		for (const [pid, eps] of fireEpisodes) {
			if (eps.includes(ep)) parentsFired.add(pid);
		}
		if (parentsFired.size > 0) {
			// pick the one with the longest gap to prior firing
			const sorted = [...parentsFired].map(pid => {
				const eps = fireEpisodes.get(pid);
				const idx = eps.indexOf(ep);
				const gap = idx > 0 ? ep - eps[idx - 1] : ep;
				return { pid, gap };
			}).sort((a, b) => b.gap - a.gap);
			if (sorted[0].gap >= 5) {
				latestBurstEp = ep;
				parentId = sorted[0].pid;
				break;
			}
		}
	}
	if (parentId < 0) {
		console.log('\nNo burst (with gap≥5) found. Try --episodes 25+ or --target-parent N.');
		process.exit(0);
	}
	console.log(`\nLatest burst at ep${latestBurstEp}, picking parent #${parentId} (gap=${fireEpisodes.get(parentId).slice(-2).reduce((a, b) => b - a)})`);
}

// ── Dump that parent's children's context evolution ─────────────────────────

console.log(`\nParent #${parentId} children — context evolution across episodes:`);

// Find all children of this parent (by walking patternMeta)
const children = [...patternMeta.entries()]
	.filter(([_, m]) => m.parentId === parentId)
	.map(([id, m]) => ({ id, level: m.level }))
	.sort((a, b) => a.id - b.id);

console.log(`  ${children.length} children of #${parentId}`);

for (const { id: cid } of children) {
	console.log(`\n  child #${cid} (L${patternMeta.get(cid).level}):`);
	let prevMap = null;
	let prevEp = null;
	const allKeys = new Set();
	// collect union of all keys ever in this child's context across episodes
	for (let ep = 1; ep <= EPISODES; ep++) {
		const snap = snapshots.get(ep);
		if (!snap || !snap.has(cid)) continue;
		for (const k of snap.get(cid).keys()) allKeys.add(k);
	}

	// Show summary: per episode the count of entries and changes vs prev
	let totalAdded = 0, totalRemoved = 0;
	for (let ep = 1; ep <= EPISODES; ep++) {
		const snap = snapshots.get(ep);
		if (!snap || !snap.has(cid)) continue;
		const cur = snap.get(cid);
		let added = 0, removed = 0, weakened = 0, strengthened = 0;
		if (prevMap) {
			for (const [k, v] of cur) {
				if (!prevMap.has(k)) added++;
				else if (prevMap.get(k) > v) weakened++;
				else if (prevMap.get(k) < v) strengthened++;
			}
			for (const k of prevMap.keys()) {
				if (!cur.has(k)) removed++;
			}
		}
		const change = (prevMap && (added || removed || weakened || strengthened))
			? ` [+${added} -${removed} weakened=${weakened} strengthened=${strengthened}]`
			: '';
		if (change || ep === 1) {
			console.log(`    ep${String(ep).padStart(2)}: ${cur.size} entries${change}`);
		}
		totalAdded += added; totalRemoved += removed;
		prevMap = cur; prevEp = ep;
	}
	if (totalRemoved > 0) {
		// show which keys were removed
		const firstSnap = (() => { for (let e = 1; e <= EPISODES; e++) { const s = snapshots.get(e); if (s && s.has(cid)) return s.get(cid); } return null; })();
		const lastSnap = snapshots.get(EPISODES)?.get(cid);
		if (firstSnap && lastSnap) {
			const removed = [...firstSnap.keys()].filter(k => !lastSnap.has(k));
			if (removed.length > 0) {
				console.log(`    REMOVED entries (first→last): ${removed.join(', ')}`);
			}
		}
	}
}

console.log('\nDone.');
