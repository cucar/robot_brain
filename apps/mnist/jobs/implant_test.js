/**
 * Implant Test (parallel-context training) — N parallel brain contexts for
 * training (one per training image), then classification uses the regular
 * single-context interface: resetContext + processFrame + infer for each
 * image.
 *
 * Reports both label accuracy (action consensus) and per-image event accuracy
 * (next-bit prediction across the classify pass).
 *
 * Usage:
 *   node apps/mnist/jobs/implant_test.js \
 *     [--max-images N] [--context-length N] [--epochs N] \
 *     [--merge-threshold F] [--error-mode static|conservative|neutral|aggressive] \
 *     [--error-threshold F] [--forget-rate F] [--level-decay exponential|linear|static] \
 *     [--no-class-balance] [--no-trim] [--test-held-out]
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
const flt = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? parseFloat(process.argv[i + 1]) : d; };
const str = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : d; };

const MAX_IMAGES = num('--max-images', 5);
const CONTEXT_LENGTH = num('--context-length', 30);
const EPOCHS = num('--epochs', 1);
const MERGE_THRESHOLD = flt('--merge-threshold', 1.0);
const ERROR_MODE = str('--error-mode', 'conservative');
const ERROR_THRESHOLD = flt('--error-threshold', 0.5);
const FORGET_RATE = flt('--forget-rate', 0);
const LEVEL_DECAY = str('--level-decay', 'exponential');
const TRIM = !process.argv.includes('--no-trim');
const TEST_HELD_OUT = process.argv.includes('--test-held-out');
const MAX_HELD_OUT = num('--max-held-out', MAX_IMAGES);
const MAX_CLASSIFY_TRAIN = num('--max-classify-train', MAX_IMAGES);
const WIRE_PER_EPOCH = process.argv.includes('--wire-per-epoch');

const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));
const testImages = TEST_HELD_OUT ? loadImages(findFile('t10k-images-idx3-ubyte')) : null;
const testLabels = TEST_HELD_OUT ? loadLabels(findFile('t10k-labels-idx1-ubyte')) : null;

const brain = new Brain({
	contextLength: CONTEXT_LENGTH,
	mergeThreshold: MERGE_THRESHOLD,
	patternForgetRate: FORGET_RATE,
	errorCorrectionMode: ERROR_MODE,
	errorCorrectionThreshold: ERROR_THRESHOLD,
	levelDecayMode: LEVEL_DECAY,
});
const encoder = new MNISTEncoder(2, TRIM);
encoder.registerChannels(brain);
const trainBits = [];
const trainLabelArr = [];
for (let i = 0; i < MAX_IMAGES; i++) {
	trainBits.push(encoder.buildBits(images[i]));
	trainLabelArr.push(labels[i]);
}

// Optionally classify only a subset of the trained images (saves time on huge N).
const numClassifyTrain = Math.min(MAX_CLASSIFY_TRAIN, MAX_IMAGES);
const classifyBits = trainBits.slice(0, numClassifyTrain).map(b => b);
const classifyLabels = trainLabelArr.slice(0, numClassifyTrain);
const classifyTags = classifyBits.map((_, i) => `train ${i}`);
if (TEST_HELD_OUT) {
	const numHeldOut = Math.min(MAX_HELD_OUT, testImages.length);
	for (let i = 0; i < numHeldOut; i++) {
		classifyBits.push(encoder.buildBits(testImages[i]));
		classifyLabels.push(testLabels[i]);
		classifyTags.push(`held  ${i}`);
	}
}

const maxLen = Math.max(...trainBits.map(b => b.length));
const N_TRAIN = MAX_IMAGES;
const N_CLASS = classifyBits.length;

console.log(`Config: N=${N_TRAIN} class=${N_CLASS} maxLen=${maxLen} ctxLen=${CONTEXT_LENGTH} epochs=${EPOCHS}`);
console.log(`        merge=${MERGE_THRESHOLD} errMode=${ERROR_MODE} errThresh=${ERROR_THRESHOLD} forget=${FORGET_RATE} levelDecay=${LEVEL_DECAY}`);

const channelId = encoder.channelId;
const inputDimId = encoder.inputDimId;

function buildEvents(bit) {
	const inputs = new Map();
	const dimMap = new Map();
	dimMap.set(inputDimId, bit);
	inputs.set(channelId, dimMap);
	return inputs;
}

// Compute action consensus from raw vote stats using SUM-OF-EVIDENCE
// (totalStrength × avgReward), not the brain's default per-voter mean.
// Returns the digit with the highest summed weighted vote, or -1.
function sumConsensusDigit(actionVoteStats) {
	let bestDigit = -1, bestScore = -Infinity;
	for (const s of actionVoteStats || []) {
		const score = s.totalStrength * s.avgReward; // = weighted_total
		if (score > bestScore) { bestScore = score; bestDigit = s.value; }
	}
	return bestDigit;
}

function extractEventPrediction(result) {
	const arr = result?.inferences?.get(channelId);
	if (!arr) return -1;
	for (const inf of arr) {
		if (inf.kind === 'event' && inf.dimId === inputDimId) {
			return inf.winner?.value ?? -1;
		}
	}
	return -1;
}

const EMPTY = new Map();

// ── Phase 1: parallel event training with full error correction ────────────

brain.initContexts(N_TRAIN);
const t0 = Date.now();
for (let epoch = 0; epoch < EPOCHS; epoch++) {
	if (epoch > 0) {
		for (let i = 0; i < N_TRAIN; i++) {
			brain.setActiveContext(i);
			brain.resetContext();
		}
	}
	brain.setProcessingMode(true, false, true);
	for (let pos = 0; pos < maxLen; pos++) {
		for (let i = 0; i < N_TRAIN; i++) {
			if (pos >= trainBits[i].length) continue;
			const bit = trainBits[i][pos];
			brain.setActiveContext(i);
			brain.processFrame(buildEvents(bit), EMPTY, EMPTY);
		}
	}
	// Optionally wire actions at end of every epoch (cumulative additive
	// reinforcement of voters that fire at end-of-image in each epoch).
	if (WIRE_PER_EPOCH) {
		for (let i = 0; i < N_TRAIN; i++) {
			const label = trainLabelArr[i];
			brain.setActiveContext(i);
			encoder.resetActions();
			const actions = encoder.encodeAction(label);
			const rewards = encoder.buildRewards(label, 1.0, -1.0);
			brain.learn(actions, rewards);
		}
	}
	if (EPOCHS > 1) {
		console.log(`  epoch ${epoch + 1}/${EPOCHS}: ${brain.getFrameSummary().neuronCount} total neurons${WIRE_PER_EPOCH ? ' (actions wired)' : ''}`);
	}
}
const trainMs = Date.now() - t0;
console.log(`Training: ${trainMs}ms (${EPOCHS} epoch${EPOCHS > 1 ? 's' : ''}, wire=${WIRE_PER_EPOCH ? 'per-epoch' : 'end-only'}), ${brain.getFrameSummary().neuronCount} total neurons`);

// ── Phase 2: per-context action wiring at end of training ──────────────────

// Per-image snapshot of voters + action stats captured at post-learn time,
// so we can compare to single-context replay state later.
const postLearnSnapshots = new Map(); // i → { voters, stats }

const t1 = Date.now();
let postLearnCorrect = 0;
const postLearnFails = [];
// If we wired per epoch, Phase 2 still runs a final wiring + post-learn infer
// so we can read the same accuracy metric. Additive accumulation means this
// just adds one more wiring on top — same relative ordering.
for (let i = 0; i < N_TRAIN; i++) {
	const label = trainLabelArr[i];
	brain.setActiveContext(i);
	encoder.resetActions();
	const actions = encoder.encodeAction(label);
	const rewards = encoder.buildRewards(label, 1.0, -1.0);
	const learnResult = brain.learn(actions, rewards);
	const predicted = sumConsensusDigit(learnResult.actionVoteStats);
	// snapshot ALL contexts' voter state at post-learn time (cheap)
	const voters = brain.getVotableEntries().map(v => ({
		id: v.neuronId,
		age: v.age,
		level: brain.inspectNeuron(v.neuronId).level,
	}));
	postLearnSnapshots.set(i, { voters, stats: learnResult.actionVoteStats });
	if (predicted === label) postLearnCorrect++;
	else {
		// capture voter set + level distribution at this fail moment
		const voters = brain.getVotableEntries();
		const levelCounts = new Map();
		const voterIds = [];
		for (const v of voters) {
			const info = brain.inspectNeuron(v.neuronId);
			levelCounts.set(info.level, (levelCounts.get(info.level) || 0) + 1);
			voterIds.push({ id: v.neuronId, age: v.age, level: info.level });
		}
		postLearnFails.push({ i, label, predicted, stats: learnResult.actionVoteStats, voters: voterIds, levelCounts });
	}
}
const wireMs = Date.now() - t1;
console.log(`Action wiring: ${wireMs}ms`);
console.log(`Post-learn infer accuracy: ${(postLearnCorrect / N_TRAIN * 100).toFixed(2)}% (${postLearnCorrect}/${N_TRAIN})`);
// Save voter sets for cross-fail intersection analysis
const failVoterSets = new Map(); // ctx i → Set of neuron ids
for (const f of postLearnFails) {
	const levelStr = [...f.levelCounts.entries()].sort((a, b) => a[0] - b[0]).map(([l, c]) => `L${l}:${c}`).join(' ');
	console.log(`\n  Post-learn FAIL: ctx ${f.i} lbl=${f.label} → ${f.predicted}  voters=${f.voters.length} (${levelStr})`);
	const ranked = (f.stats || [])
		.map(s => ({ digit: s.value, votes: s.voteCount, strength: s.totalStrength, avgReward: s.avgReward, score: s.totalStrength * s.avgReward }))
		.sort((a, b) => b.score - a.score);
	for (const r of ranked) {
		const tag = r.digit === f.label ? ' ←correct' : (r.digit === f.predicted ? ' ←chosen' : '');
		console.log(`    digit ${r.digit}: votes=${r.votes} strength=${r.strength.toFixed(2)} avgReward=${r.avgReward.toFixed(2)} weighted=${r.score.toFixed(2)}${tag}`);
	}
	failVoterSets.set(f.i, new Set(f.voters.map(v => v.id)));
}

// Cross-fail voter intersection — see if failing contexts share voters with each other
// AND with the class-3 (or whichever predicted-class) training contexts.
if (postLearnFails.length > 1) {
	console.log(`\n  Cross-fail voter overlap:`);
	const failCtxs = [...failVoterSets.keys()];
	for (let a = 0; a < failCtxs.length; a++) {
		for (let b = a + 1; b < failCtxs.length; b++) {
			const sa = failVoterSets.get(failCtxs[a]);
			const sb = failVoterSets.get(failCtxs[b]);
			let common = 0;
			for (const id of sa) if (sb.has(id)) common++;
			console.log(`    ctx ${failCtxs[a]} ∩ ctx ${failCtxs[b]}: ${common} / ${sa.size} shared neuron IDs`);
		}
	}
}

// ── Phase 3: classify each image via single-context interface (slot 0) ─────
//
// We reuse training slot 0 as our scratch single-context, resetting its
// per-context memory before each image. The shared thalamus (where all the
// trained patterns live) is unchanged.

// Resolve action neuron ids for digits 0..9 (created during Phase 2 wiring).
const actionNeuronIds = new Map();
for (let d = 0; d < 10; d++) {
	const nid = brain.getNeuronIdByCoordinate(encoder.actionDimId, d);
	if (nid != null) actionNeuronIds.set(d, Number(nid));
}

function dumpVoters(tag, label) {
	const voters = brain.getVotableEntries();
	const digitStrength = new Array(10).fill(0);
	const digitReward = new Array(10).fill(0);
	const digitVoteCount = new Array(10).fill(0);
	const perLevel = new Map();
	for (const v of voters) {
		const info = brain.inspectNeuron(v.neuronId);
		const lvl = info.level;
		perLevel.set(lvl, (perLevel.get(lvl) || 0) + 1);
		const conns = brain.dumpNeuronConnections(v.neuronId);
		for (const c of conns) {
			if (c.distance !== v.age + 1) continue;
			for (const [digit, aid] of actionNeuronIds) {
				if (c.targetId === aid) {
					digitStrength[digit] += c.strength;
					digitReward[digit] += c.strength * c.reward;
					digitVoteCount[digit]++;
				}
			}
		}
	}
	const levelStr = [...perLevel.entries()].sort((a, b) => a[0] - b[0]).map(([l, c]) => `L${l}:${c}`).join(' ');
	console.log(`  ${tag} lbl=${label} | voters=${voters.length} (${levelStr})`);
	const ranked = digitStrength.map((s, d) => ({ d, votes: digitVoteCount[d], strength: s, weighted: digitReward[d], score: s > 0 ? digitReward[d] / s : 0 }))
		.filter(x => x.votes > 0)
		.sort((a, b) => b.score - a.score);
	for (const r of ranked) {
		const correct = r.d === label ? ' ←correct' : '';
		console.log(`    digit ${r.d}: voters=${r.votes}  strength=${r.strength.toFixed(1)}  weighted=${r.weighted.toFixed(2)}  score=${r.score.toFixed(3)}${correct}`);
	}
}

const t2 = Date.now();
brain.setProcessingMode(true, false, false);
brain.setActiveContext(0);

console.log(`\nClassification (single-context interface on slot 0):`);
let trainCorrect = 0, heldCorrect = 0;
let trainTotal = 0, heldTotal = 0;
let trainEvtSum = 0, heldEvtSum = 0;
const perDigit = new Array(10).fill(0);
const perDigitTotal = new Array(10).fill(0);

for (let j = 0; j < N_CLASS; j++) {
	const bits = classifyBits[j];
	const label = classifyLabels[j];
	const tag = classifyTags[j];

	brain.resetContext();
	let lastPred = -1;
	let evtC = 0, evtT = 0;
	for (let pos = 0; pos < bits.length; pos++) {
		if (pos >= CONTEXT_LENGTH && lastPred !== -1) {
			evtT++;
			if (lastPred === bits[pos]) evtC++;
		}
		const result = brain.processFrame(buildEvents(bits[pos]), EMPTY, EMPTY);
		lastPred = extractEventPrediction(result);
	}

	// Voter dump at end-of-replay, before infer (infer is read-only).
	dumpVoters(tag, label);

	encoder.resetActions();
	const inferResult = brain.infer();
	const predicted = sumConsensusDigit(inferResult.actionVoteStats);
	const ok = predicted === label;
	// Capture per-image classify-time snapshot to compare against post-learn
	const replayVoters = brain.getVotableEntries().map(v => ({
		id: v.neuronId, age: v.age, level: brain.inspectNeuron(v.neuronId).level,
	}));
	if (!ok && j < N_TRAIN) {
		const pl = postLearnSnapshots.get(j);
		if (pl) {
			console.log(`\n  *** Diagnostic for failing ${tag} (lbl=${label} → ${predicted}):`);
			const plLevels = new Map(), rpLevels = new Map();
			for (const v of pl.voters) plLevels.set(v.level, (plLevels.get(v.level) || 0) + 1);
			for (const v of replayVoters) rpLevels.set(v.level, (rpLevels.get(v.level) || 0) + 1);
			const ls = m => [...m.entries()].sort().map(([l, c]) => `L${l}:${c}`).join(' ');
			console.log(`    post-learn  voters=${pl.voters.length} (${ls(plLevels)})`);
			console.log(`    cold-replay voters=${replayVoters.length} (${ls(rpLevels)})`);
			const plIds = new Set(pl.voters.map(v => v.id));
			const rpIds = new Set(replayVoters.map(v => v.id));
			let common = 0;
			for (const id of rpIds) if (plIds.has(id)) common++;
			console.log(`    overlap: ${common} / ${rpIds.size} replay voters were in post-learn set`);
			// Action vote comparison
			const fmt = (stats) => (stats || [])
				.map(s => ({ d: s.value, votes: s.voteCount, str: s.totalStrength, rew: s.avgReward, wt: s.totalStrength * s.avgReward }))
				.sort((a, b) => b.wt - a.wt)
				.map(x => `${x.d}:wt=${x.wt.toFixed(1)}(v=${x.votes},s=${x.str.toFixed(1)},r=${x.rew.toFixed(2)})`)
				.join(' | ');
			console.log(`    post-learn votes:  ${fmt(pl.stats)}`);
			console.log(`    cold-replay votes: ${fmt(inferResult.actionVoteStats)}`);
		}
	}
	const evtAcc = evtT > 0 ? evtC / evtT : 0;
	const isHeld = tag.startsWith('held');
	if (isHeld) {
		heldTotal++;
		heldEvtSum += evtAcc;
		if (ok) { heldCorrect++; perDigit[label]++; }
		perDigitTotal[label]++;
	} else {
		trainTotal++;
		trainEvtSum += evtAcc;
		if (ok) trainCorrect++;
	}
	console.log(`    → predicted ${predicted} ${ok ? '✓' : '✗'}  evt=${(evtAcc * 100).toFixed(1)}%`);
}
const replayMs = Date.now() - t2;
console.log(`\nClassification: ${replayMs}ms`);
console.log(`Train labelAcc: ${(trainCorrect / Math.max(1, trainTotal) * 100).toFixed(2)}% (${trainCorrect}/${trainTotal})  avg evtAcc: ${(trainEvtSum / Math.max(1, trainTotal) * 100).toFixed(2)}%`);
if (heldTotal > 0) {
	const perDigitStr = perDigit.map((c, d) =>
		perDigitTotal[d] > 0 ? `${d}:${(c / perDigitTotal[d] * 100).toFixed(0)}%` : `${d}:—`
	).join(' ');
	console.log(`Held-out labelAcc: ${(heldCorrect / heldTotal * 100).toFixed(2)}% (${heldCorrect}/${heldTotal})  avg evtAcc: ${(heldEvtSum / heldTotal * 100).toFixed(2)}%  per-digit: ${perDigitStr}`);
}
