/**
 * Profile 28x28 radius-3 per-image cost: where does the time go, and does the neuron count run away?
 * Processes a handful of real training images one at a time with reuse+refinement on (defaults), printing
 * the per-section timings and the growth in neuron / minted-correction counts. Isolates whether the
 * bottleneck is recognition (matching), connection learning, or the mint pass (O(n^2) clustering + reuse).
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SIZE = 28, BUCKETS = 2, RADIUS = 3;
const N = Number(process.argv[2] || 15);

const encoder = new MNISTPixelChannelsEncoder(BUCKETS, SIZE, RADIUS);
const brain = new Brain({ contextLength: 1, patternForgetRate: 0, consensus: 'nb', groupMode: 'neutral', groupThreshold: 0.5, columns: 20 });
encoder.registerChannels(brain);

const DATA = path.resolve(__dirname, '../data');
const images = loadImages(path.join(DATA, 'train-images-idx3-ubyte.gz'));
const labels = loadLabels(path.join(DATA, 'train-labels-idx1-ubyte.gz'));

const ms = (s) => (s * 1000).toFixed(0);
console.log(`profiling ${N} images @ ${SIZE}x${SIZE} r${RADIUS}, reuse+refine on\n`);
console.log('img | total | procSpat recog candEval learnConn | neurons minted (Δmint)');

let prevMint = 0;
for (let i = 0; i < N; i++) {
  const bits = encoder.buildBits(images[i]);
  brain.resetContext();
  console.error(`=== IMG ${i} ===`);
  const t0 = Date.now();
  const r = brain.processFrame(encoder.encodeImage(bits), new Map());
  brain.learn(encoder.encodeAction(labels[i]), 1);
  const wall = Date.now() - t0;
  const t = r.frame.timings;
  const nc = brain.getFrameSummary().neuronCount;
  const minted = brain.getSpatialCorrectionCount();
  console.log(
    `${String(i).padStart(3)} | ${String(wall).padStart(5)}ms | ` +
    `${ms(t.processSpatial).padStart(6)} ${ms(t.neuronRecognizePatterns).padStart(5)} ${String(t.recognizeCandidatesEvaluated).padStart(7)} ${ms(t.neuronLearnConnections).padStart(6)} | ` +
    `${String(nc).padStart(7)} ${String(minted).padStart(6)} (+${minted - prevMint})`
  );
  prevMint = minted;
  if (i === N - 1) {
    console.log(`\n=== full timing breakdown for image ${i} (ms, sorted) ===`);
    const t2 = r.frame.timings;
    Object.entries(t2)
      .filter(([k]) => k !== 'recognizeCandidatesEvaluated')
      .map(([k, v]) => [k, v * 1000])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 14)
      .forEach(([k, v]) => console.log(`  ${k.padEnd(26)} ${v.toFixed(0)}ms`));
  }
}
