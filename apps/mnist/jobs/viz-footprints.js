/**
 * Visualize the spatial-correction hierarchy on a few 7×7 MNIST digits.
 *
 * Loads a trained brain, processes a sample image per digit, and renders each active spatial
 * correction's footprint (the base pixels it covers) on the 7×7 grid — so we can see whether the
 * L1/L2 patterns trace the digit's strokes rather than memorizing whole images.
 *
 * Usage: node apps/mnist/jobs/viz-footprints.js [--brain <label>] [--digits 1,7,0,2,4]
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTPixelChannelsEncoder } from '../encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const arg = (flag, dflt) => {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : dflt;
};
const SIZE = Number(arg('--image-size', '7'));
const brainLabel = arg('--brain', 'consv7b');
const wantDigits = arg('--digits', '1,7,0,2,4').split(',').map(Number);
const groupMode = arg('--group-mode', 'conservative');
const groupThreshold = Number(arg('--group-threshold', '0.5'));
const columns = Number(arg('--columns', '1'));
const buckets = Number(arg('--buckets', '2'));

// Build a brain matching the trained config (group mode/threshold and columns MUST match the training
// run or load/inference diverges), register the per-pixel channels, then load the backup. Channels
// register before load so the id↔name maps reconcile (same order as the harness).
const encoder = new MNISTPixelChannelsEncoder(buckets, SIZE, 1);
const brain = new Brain({ contextLength: 1, patternForgetRate: 0, consensus: 'nb', groupMode, groupThreshold, columns });
encoder.registerChannels(brain);
brain.load(path.resolve(__dirname, 'test'), brainLabel);
brain.setLearning(false);

// channelId → linear pixel index (p = y*SIZE + x), the inverse of the encoder's registration order.
const chToPixel = new Map();
encoder.pixelChannelIds.forEach((ch, p) => chToPixel.set(ch, p));

const DATA = path.resolve(__dirname, '../data');
const testImages = loadImages(path.join(DATA, 't10k-images-idx3-ubyte.gz'));
const testLabels = loadLabels(path.join(DATA, 't10k-labels-idx1-ubyte.gz'));

// Render a SIZE×SIZE grid by calling cell(x, y) for each position.
const grid = (cell) => {
	let s = '';
	for (let y = 0; y < SIZE; y++) {
		let row = '  ';
		for (let x = 0; x < SIZE; x++) row += cell(x, y) + ' ';
		s += row + '\n';
	}
	return s;
};

// Ink pixels (bucket ≥ 1 = "ink present") a correction's footprint covers, as a Set of pixel indices.
const inkPixels = (corr) => {
	const set = new Set();
	for (let i = 0; i < corr.ch.length; i++) {
		if (corr.bk[i] >= 1) {
			const p = chToPixel.get(corr.ch[i]);
			if (p !== undefined) set.add(p);
		}
	}
	return set;
};

for (const digit of wantDigits) {
	const idx = testLabels.findIndex((l) => l === digit);
	if (idx < 0) { console.log(`(no test image for digit ${digit})`); continue; }

	const bits = encoder.buildBits(testImages[idx]);
	brain.resetContext();
	brain.processFrame(encoder.encodeImage(bits), new Map());
	const dump = JSON.parse(brain.dumpActiveSpatialCorrections());

	const byLevel = new Map();
	for (const c of dump) {
		if (!byLevel.has(c.level)) byLevel.set(c.level, []);
		byLevel.get(c.level).push(c);
	}
	const levels = [...byLevel.keys()].sort((a, b) => a - b);
	const counts = levels.map((l) => `L${l}:${byLevel.get(l).length}`).join(' ');

	console.log(`\n${'='.repeat(40)}`);
	console.log(`digit ${digit} (test #${idx}) — active corrections: ${counts || 'none'}`);

	// Input ink.
	console.log(`\ninput:`);
	process.stdout.write(grid((x, y) => (bits[y * SIZE + x] >= 1 ? '#' : '·')));

	// Per-level ink-coverage heatmap: how many corrections at that level cover each pixel.
	for (const level of levels) {
		const heat = new Array(SIZE * SIZE).fill(0);
		for (const c of byLevel.get(level)) for (const p of inkPixels(c)) heat[p]++;
		console.log(`\nL${level} ink coverage (count of L${level} corrections covering each pixel):`);
		process.stdout.write(grid((x, y) => {
			const n = heat[y * SIZE + x];
			return n === 0 ? '·' : n > 9 ? '+' : String(n);
		}));
	}

	// A few individual L2 footprints, to see if a single pattern traces a stroke segment.
	const l2 = byLevel.get(2) || [];
	const show = l2.slice(0, 3);
	for (let k = 0; k < show.length; k++) {
		const ink = inkPixels(show[k]);
		console.log(`\nL2 #${show[k].id} footprint (O = covered ink, o = covered non-ink, # = digit ink uncovered):`);
		process.stdout.write(grid((x, y) => {
			const p = y * SIZE + x;
			const covered = ink.has(p);
			const isInk = bits[p] >= 1;
			if (covered && isInk) return 'O';
			if (covered && !isInk) return 'o';
			if (isInk) return '#';
			return '·';
		}));
	}
}
