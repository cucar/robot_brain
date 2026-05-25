/**
 * Pattern Growth Trace — train 2 images for N episodes; after each episode
 * snapshot the set of pattern neurons, then per episode report:
 *   - how many NEW patterns were born this episode
 *   - how many distinct PARENTS those new patterns sit under
 *   - "fresh parents" (got their first child this episode) vs "repeat parents"
 *     (already had children from earlier episodes)
 *
 * The repeat-parent count is the smoking gun for the unbounded growth: if it
 * stays > 0 episode after episode, the brain keeps spawning new sibling
 * patterns under the same parents — meaning the same (parent, age) error
 * keeps firing and the prior pattern never resolves it.
 *
 * For each repeat parent we also dump:
 *   - how many siblings it now has under it
 *   - the context of its first and latest children, side by side, so you can
 *     compare how the contexts differ (are they subtly different versions of
 *     the same situation? completely different?)
 *
 * Usage: node apps/mnist/jobs/pattern_growth_trace.js [--digit-a N] [--digit-b N]
 *        [--episodes N] [--context-length N] [--merge-threshold N]
 *        [--forget-rate N] [--error-mode static|conservative]
 *        [--max-dump N]
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

const DIGIT_A = num('--digit-a', 5);
const DIGIT_B = num('--digit-b', 0);
const MAX_IMAGES = num('--max-images', 0); // 0 → use --digit-a/--digit-b
const EPISODES = num('--episodes', 10);
const CONTEXT_LENGTH = num('--context-length', 28);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = numFloat('--error-threshold', 0.5);
const MERGE_THRESHOLD = numFloat('--merge-threshold', 0.99);
const FORGET_RATE = numFloat('--forget-rate', 0);
const MAX_DUMP = num('--max-dump', 5);
const TRIM = !process.argv.includes('--no-trim');

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));
function firstWithLabel(t) { for (let i = 0; i < labels.length; i++) if (labels[i] === t) return { idx: i, pixels: images[i] }; throw new Error(`no image with label ${t}`); }
// Either two specific digits (default) or first MAX_IMAGES images from the
// training set in order.
let samples;
if (MAX_IMAGES > 0) {
	samples = [];
	for (let i = 0; i < MAX_IMAGES; i++) {
		samples.push({ idx: i, pixels: images[i], label: labels[i] });
	}
} else {
	const A = firstWithLabel(DIGIT_A);
	const B = firstWithLabel(DIGIT_B);
	samples = [{ ...A, label: DIGIT_A }, { ...B, label: DIGIT_B }];
}

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

console.log('Pattern Growth Trace');
console.log(`  ${samples.length} samples: ${samples.map((s, i) => `lbl=${s.label}(${allBits[i].length}b)`).join(' ')}`);
console.log(`  ctx=${CONTEXT_LENGTH} errMode=${ERROR_MODE} merge=${MERGE_THRESHOLD} forget=${FORGET_RATE}`);
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

// parent_id -> array of child pattern ids (in birth order, growing over time)
const childrenByParent = new Map();
// pattern_id -> { episode_born, parent_id, level, context }
const patternMeta = new Map();
let lastNeuronId = 0;

function snapshotNewPatterns(episode) {
	const total = brain.getFrameSummary().neuronCount;
	const newOnes = [];
	for (let id = lastNeuronId + 1; id <= total; id++) {
		const info = brain.inspectNeuron(id);
		if (info.level === 0) continue; // skip sensory
		patternMeta.set(id, { episodeBorn: episode, parentId: info.parentId, level: info.level, context: info.context });
		const arr = childrenByParent.get(info.parentId) || [];
		arr.push(id);
		childrenByParent.set(info.parentId, arr);
		newOnes.push({ id, ...info });
	}
	lastNeuronId = total;
	return newOnes;
}

// ── Train + snapshot ────────────────────────────────────────────────────────

console.log('Episode | new pat | fresh parents | repeat parents | repeat-parent new patterns | total neurons | maxLevel');

const perEpisodeStats = [];
const perEpisodeNew = []; // per-ep array of newly-created pattern ids
// VERBOSE_FROM: from this episode onward, print each new pattern individually.
// Defaults to 5 since by then we're in the steady-state regime.
const VERBOSE_FROM = num('--verbose-from', 5);

for (let ep = 1; ep <= EPISODES; ep++) {
	if (shuttingDown) break;
	for (const bits of allBits) feedBits(bits, true);

	const total = brain.getFrameSummary().neuronCount;
	const newOnes = [];
	const freshParents = new Set();
	const repeatParents = new Set();
	let repeatParentNewPatterns = 0;

	for (let id = lastNeuronId + 1; id <= total; id++) {
		const info = brain.inspectNeuron(id);
		if (info.level === 0) continue;
		const parent = info.parentId;
		const hadChildrenBefore = childrenByParent.has(parent);
		if (hadChildrenBefore) {
			repeatParents.add(parent);
			repeatParentNewPatterns++;
		} else {
			freshParents.add(parent);
		}
		const arr = childrenByParent.get(parent) || [];
		arr.push(id);
		childrenByParent.set(parent, arr);
		patternMeta.set(id, { episodeBorn: ep, parentId: parent, level: info.level, context: info.context });
		newOnes.push({ id, info });
	}
	lastNeuronId = total;
	perEpisodeNew.push(newOnes);

	const summary = brain.getFrameSummary();
	console.log(`  ${String(ep).padStart(5)} | ${String(newOnes.length).padStart(7)} | ${String(freshParents.size).padStart(13)} | ${String(repeatParents.size).padStart(14)} | ${String(repeatParentNewPatterns).padStart(27)} | ${String(summary.neuronCount).padStart(13)} | ${String(summary.maxLevel).padStart(8)}`);

	// In the steady-state regime, dump each newly-created pattern individually.
	if (ep >= VERBOSE_FROM) {
		for (const { id, info } of newOnes) {
			const ctx = info.context
				.sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId)
				.map(c => `${c.neuronId}@d${c.distance}`).join(' ');
			console.log(`        +#${id} L${info.level} parent=${info.parentId} ctx(${info.context.length}): ${ctx}`);
		}
	}

	perEpisodeStats.push({ ep, newPatterns: newOnes.length, freshParents: freshParents.size, repeatParents: repeatParents.size, repeatParentNewPatterns });
}

// ── Persistent spawners (steady-state regime only) ─────────────────────────

// Which parents got new children in episodes ≥ VERBOSE_FROM? These are the
// real steady-state offenders — they're firing the same error situation
// over and over without ever being resolved by their prior pattern.
const persistentSpawnerCounts = new Map(); // parent_id -> { steadyChildren: ids, allChildren: ids }
for (let i = VERBOSE_FROM - 1; i < perEpisodeNew.length; i++) {
	for (const { id, info } of perEpisodeNew[i]) {
		const pid = info.parentId;
		const entry = persistentSpawnerCounts.get(pid) || { steady: [], all: childrenByParent.get(pid) ?? [] };
		entry.steady.push({ id, ep: i + 1, context: info.context });
		entry.all = childrenByParent.get(pid) ?? entry.all;
		persistentSpawnerCounts.set(pid, entry);
	}
}

const persistentList = [...persistentSpawnerCounts.entries()]
	.sort((a, b) => b[1].steady.length - a[1].steady.length);

console.log(`\nPersistent spawners (parents that gained NEW children in ep≥${VERBOSE_FROM}): ${persistentList.length}`);
for (const [pid, entry] of persistentList) {
	const parentInfo = brain.inspectNeuron(pid);
	console.log(`  parent #${pid} (L${parentInfo.level}) — ${entry.all.length} children total, ${entry.steady.length} born in steady-state`);
	console.log(`    parent's own context (${parentInfo.context.length} entries):`);
	if (parentInfo.context.length > 0) {
		const ctxStr = parentInfo.context
			.sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId)
			.map(c => `${c.neuronId}@d${c.distance}`).join(' ');
		console.log(`      ${ctxStr}`);
	}

	// Dump all of this parent's children — ALL OF THEM, in birth order, with
	// their contexts. This is what we need to see: are the steady-state new
	// patterns near-duplicates of earlier siblings (broken dedupe), or are
	// they genuinely fresh contexts?
	console.log(`    all ${entry.all.length} children's contexts in birth order:`);
	for (const cid of entry.all) {
		const m = patternMeta.get(cid);
		if (!m) continue; // shouldn't happen
		const ctx = m.context
			.sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId)
			.map(c => `${c.neuronId}@d${c.distance}`).join(' ');
		const marker = m.episodeBorn >= VERBOSE_FROM ? '★' : ' ';
		console.log(`     ${marker} #${cid} ep${m.episodeBorn} L${m.level} ctx(${m.context.length}): ${ctx}`);
	}
}

// ── Repeat-parent analysis ──────────────────────────────────────────────────

const repeatParents = [...childrenByParent.entries()]
	.filter(([_, children]) => children.length > 1)
	.sort((a, b) => b[1].length - a[1].length);

console.log(`\nRepeat-spawning parents (parent has ≥2 children patterns under it): ${repeatParents.length}`);
console.log(`Top spawners (parent → child count):`);
for (let i = 0; i < Math.min(10, repeatParents.length); i++) {
	const [pid, children] = repeatParents[i];
	const parentInfo = brain.inspectNeuron(pid);
	const epoch = children.map(c => patternMeta.get(c).episodeBorn);
	const epRange = `ep${Math.min(...epoch)}-${Math.max(...epoch)}`;
	console.log(`  parent #${pid} (L${parentInfo.level}) — ${children.length} children, born ${epRange}`);
}

// ── Dump child contexts side-by-side for top repeat parents ────────────────

console.log(`\nFor up to ${MAX_DUMP} top repeat parents, dump children's contexts side-by-side:`);
for (let i = 0; i < Math.min(MAX_DUMP, repeatParents.length); i++) {
	const [pid, children] = repeatParents[i];
	const parentInfo = brain.inspectNeuron(pid);
	console.log(`\n— Parent #${pid} (L${parentInfo.level}) — ${children.length} children:`);
	const slice = children.slice(0, 5); // first 5 children
	for (const cid of slice) {
		const m = patternMeta.get(cid);
		const ctxStr = m.context
			.sort((x, y) => x.distance - y.distance || x.neuronId - y.neuronId)
			.map(c => `${c.neuronId}@d${c.distance}`)
			.join(' ');
		console.log(`    child #${cid}  born ep${m.episodeBorn}  L${m.level}  context(${m.context.length}): ${ctxStr || '(empty)'}`);
	}
	if (children.length > 5) console.log(`    ... and ${children.length - 5} more children`);
}

console.log('\nDone.');
