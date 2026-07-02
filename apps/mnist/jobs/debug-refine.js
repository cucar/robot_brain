/**
 * Debug: does spatial CONNECTION refinement actually sharpen a correction's predictions on real images?
 *
 * Trains a fresh brain on a stream of real MNIST digits (default: 20 ones then 20 sevens) and, after each
 * image, measures — over the corrections active this frame — how many d=0 connections (predictions) they
 * hold and how "sharp" those predictions are (fraction with meaningful strength). It also tracks a handful
 * of long-lived corrections individually so we can watch one pattern's predictions evolve exposure by
 * exposure. Refinement should push unreliable connections toward 0 (sharp = few strong); without it they
 * only accumulate (blurry = many, all growing).
 *
 * Usage: node apps/mnist/jobs/debug-refine.js [--digits 1,7] [--per 20] [--track 3]
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const arg = (f, d) => { const i = process.argv.indexOf(f); return i !== -1 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : d; };
const SIZE = 7, BUCKETS = 2;
const digits = arg('--digits', '1,7').split(',').map(Number);
const PER = Number(arg('--per', '20'));
const TRACK = Number(arg('--track', '3'));

const encoder = new MNISTPixelChannelsEncoder(BUCKETS, SIZE, 1);
const brain = new Brain({ contextLength: 1, patternForgetRate: 0, consensus: 'nb', groupMode: 'static', groupThreshold: 0.9, columns: 20 });
encoder.registerChannels(brain);

const DATA = path.resolve(__dirname, '../data');
const images = loadImages(path.join(DATA, 'train-images-idx3-ubyte.gz'));
const labels = loadLabels(path.join(DATA, 'train-labels-idx1-ubyte.gz'));

const idxsOf = (d, n) => { const r = []; for (let i = 0; i < labels.length && r.length < n; i++) if (labels[i] === d) r.push(i); return r; };
const seq = digits.flatMap((d) => idxsOf(d, PER));

// d=0 connection stats for a correction: count, count with strength > 0.5 ("strong"), max/sum strength.
const stats = (id) => {
  const c = brain.getNeuronConnections(id).filter((x) => x.distance === 0);
  const s = c.map((x) => x.strength);
  return { n: c.length, strong: s.filter((v) => v > 0.5).length, max: s.length ? Math.max(...s) : 0, sum: s.reduce((a, b) => a + b, 0) };
};

const history = new Map(); // correction id -> [{img, digit, ...stats}]

console.log(`stream: ${digits.map((d) => `${PER}×${d}`).join(' then ')} = ${seq.length} images | groupMode=static θ=0.9\n`);
console.log('img digit | #active | avg d0-conns | avg strong(>0.5) | avg max-str');
for (let i = 0; i < seq.length; i++) {
  const lbl = labels[seq[i]];
  const bits = encoder.buildBits(images[seq[i]]);
  brain.resetContext();
  brain.processFrame(encoder.encodeImage(bits), new Map());

  const active = JSON.parse(brain.dumpActiveSpatialCorrections());
  let tn = 0, ts = 0, tm = 0;
  for (const a of active) {
    const st = stats(a.id);
    tn += st.n; ts += st.strong; tm += st.max;
    if (!history.has(a.id)) history.set(a.id, []);
    history.get(a.id).push({ img: i, digit: lbl, ...st });
  }
  const k = active.length || 1;
  console.log(`${String(i).padStart(3)}   ${lbl}   |  ${String(active.length).padStart(4)}  |  ${(tn / k).toFixed(1).padStart(6)}  |  ${(ts / k).toFixed(1).padStart(6)}  |  ${(tm / k).toFixed(1).padStart(6)}`);
}

// SELECTIVITY: does each correction fire on ONE digit (discriminative) or both (generic)? Selectivity =
// max per-digit firings / total firings. 1.0 = perfectly digit-selective, 0.5 = fires equally on both.
const recurring = [...history.entries()].filter(([, h]) => h.length >= 3);
const sel = recurring.map(([id, h]) => {
  const by = {};
  for (const e of h) by[e.digit] = (by[e.digit] || 0) + 1;
  const total = h.length, top = Math.max(...Object.values(by));
  return { id, total, by, selectivity: top / total };
});
const avgSel = sel.reduce((a, s) => a + s.selectivity, 0) / (sel.length || 1);
const buckets = { '≥0.9': 0, '0.7-0.9': 0, '0.5-0.7': 0 };
for (const s of sel) buckets[s.selectivity >= 0.9 ? '≥0.9' : s.selectivity >= 0.7 ? '0.7-0.9' : '0.5-0.7']++;
console.log(`\n── SELECTIVITY of ${sel.length} recurring corrections (fired ≥3×) ──`);
console.log(`avg selectivity: ${avgSel.toFixed(2)} (1.0 = fires on ONE digit only; 0.5 = fires equally on both)`);
console.log(`  digit-selective (≥0.9): ${buckets['≥0.9']} | mixed (0.7-0.9): ${buckets['0.7-0.9']} | generic (0.5-0.7): ${buckets['0.5-0.7']}`);

// Show the longest-lived corrections: how their prediction set evolves across the images they fired on.
const longlived = [...history.entries()].filter(([, h]) => h.length >= 5).sort((a, b) => b[1].length - a[1].length).slice(0, TRACK);
console.log(`\n── ${longlived.length} longest-lived corrections (n = d0-conn count, strong = strength>0.5, max = strongest edge) ──`);
for (const [id, h] of longlived) {
  console.log(`\ncorrection ${id} — fired on ${h.length} images:`);
  for (const e of h) console.log(`   img ${String(e.img).padStart(3)} (d${e.digit}): n=${String(e.n).padStart(3)} strong=${String(e.strong).padStart(3)} max=${e.max.toFixed(1)} sum=${e.sum.toFixed(1)}`);
}
