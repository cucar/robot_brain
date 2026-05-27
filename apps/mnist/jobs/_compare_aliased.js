/**
 * Dump bit data for the N=20 failure images (13, 16, 18) and the class-3
 * training images they all alias to (7, 10, 12). Compare bit streams.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from 'robot-brain';
import { MNISTEncoder } from '../encoders/mnist_encoder.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, '..', 'data');
const findFile = (b) => fs.existsSync(path.join(dataDir, b)) ? path.join(dataDir, b) : path.join(dataDir, `${b}.gz`);
const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

const encoder = new MNISTEncoder(2, true);
const dummy = new Brain({ contextLength: 30, mergeThreshold: 1.0, patternForgetRate: 0 });
encoder.registerChannels(dummy);

// Failing images (predicted 3) and the class-3 training images they collide with.
const PAIR = [7, 10, 12, 13, 16, 18];

const bitsByIdx = new Map();
for (const idx of PAIR) {
	const bits = Array.from(encoder.buildBits(images[idx]));
	bitsByIdx.set(idx, bits);
	const outBits = path.join(__dirname, `_img${idx}_bits.txt`);
	const outVis = path.join(__dirname, `_img${idx}_visual.txt`);
	fs.writeFileSync(outBits, bits.join('\n'));
	const padded = bits.concat(new Array(Math.max(0, 784 - bits.length)).fill(0));
	const lines = [];
	for (let r = 0; r < 28; r++) {
		lines.push(padded.slice(r * 28, (r + 1) * 28).map(b => b ? '█' : '·').join(''));
	}
	lines.push('');
	lines.push(`label=${labels[idx]}  length=${bits.length}  ones=${bits.filter(b => b).length}`);
	fs.writeFileSync(outVis, lines.join('\n'));
}

console.log(`\nVisual comparison (28-col grids):\n`);
const visuals = [];
for (const idx of PAIR) {
	const bits = bitsByIdx.get(idx);
	const padded = bits.concat(new Array(Math.max(0, 784 - bits.length)).fill(0));
	const rows = [];
	for (let r = 0; r < 28; r++) {
		rows.push(padded.slice(r * 28, (r + 1) * 28).map(b => b ? '█' : '·').join(''));
	}
	visuals.push({ idx, label: labels[idx], length: bits.length, rows });
}

const HEADER = visuals.map(v => `idx ${String(v.idx).padStart(2)} lbl=${v.label} len=${v.length}`.padEnd(30)).join(' ');
console.log(HEADER);
for (let r = 0; r < 28; r++) {
	console.log(visuals.map(v => v.rows[r].padEnd(30)).join(' '));
}

console.log(`\n\nPairwise bit-stream comparison:`);
for (let i = 0; i < PAIR.length; i++) {
	for (let j = i + 1; j < PAIR.length; j++) {
		const ai = bitsByIdx.get(PAIR[i]);
		const aj = bitsByIdx.get(PAIR[j]);
		const minLen = Math.min(ai.length, aj.length);
		let diff = 0;
		for (let k = 0; k < minLen; k++) if (ai[k] !== aj[k]) diff++;
		const lenDiff = Math.abs(ai.length - aj.length);
		// Last-30-bits identity check (this is what end-of-training voters see)
		const tail = Math.min(30, ai.length, aj.length);
		const tailMatch = ai.slice(-tail).every((v, k) => v === aj.slice(-tail)[k]);
		console.log(`  idx ${PAIR[i]} (lbl=${labels[PAIR[i]]}) vs idx ${PAIR[j]} (lbl=${labels[PAIR[j]]}): ` +
			`lengths ${ai.length}/${aj.length} (diff ${lenDiff}), ${diff} differing bits in shared prefix, ` +
			`trailing-${tail} identical=${tailMatch}`);
	}
}
