/**
 * Dump a single MNIST image as a binary text file ('a' = dark pixel, 'b' = light pixel)
 * for testing through the text pipeline.
 *
 * Usage: node dump_image.js [imageIndex]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadImages, loadLabels } from './loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(__dirname, 'data');

const idx = parseInt(process.argv[2] || '0');

const findFile = (base) => {
	const plain = path.join(dataDir, base);
	if (fs.existsSync(plain)) return plain;
	return path.join(dataDir, `${base}.gz`);
};

const images = loadImages(findFile('train-images-idx3-ubyte'));
const labels = loadLabels(findFile('train-labels-idx1-ubyte'));

const pixels = images[idx];
const label = labels[idx];

// Binary quantize: >= 128 → 'b', < 128 → 'a'
let text = '';
for (let i = 0; i < 784; i++) {
	text += pixels[i] >= 128 ? 'b' : 'a';
}

const outPath = path.join(__dirname, '..', 'text', 'data', `mnist_img${idx}_label${label}.txt`);
fs.writeFileSync(outPath, text);

console.log(`Image #${idx} (label=${label}): ${text.length} chars`);
console.log(`Saved to: ${outPath}`);

// Show run-length stats
let runs = [];
let currentChar = text[0];
let runLen = 1;
for (let i = 1; i < text.length; i++) {
	if (text[i] === currentChar) {
		runLen++;
	} else {
		runs.push({ char: currentChar, len: runLen });
		currentChar = text[i];
		runLen = 1;
	}
}
runs.push({ char: currentChar, len: runLen });

const maxRun = Math.max(...runs.map(r => r.len));
const avgRun = (runs.reduce((s, r) => s + r.len, 0) / runs.length).toFixed(1);
const aCount = text.split('a').length - 1;
const bCount = text.split('b').length - 1;

console.log(`\nStats: ${runs.length} runs, max=${maxRun}, avg=${avgRun}`);
console.log(`  'a' (dark): ${aCount} pixels (${(aCount/784*100).toFixed(1)}%)`);
console.log(`  'b' (light): ${bCount} pixels (${(bCount/784*100).toFixed(1)}%)`);
console.log(`\nLongest runs:`);
runs.sort((a, b) => b.len - a.len);
for (let i = 0; i < Math.min(10, runs.length); i++) {
	console.log(`  '${runs[i].char}' × ${runs[i].len}`);
}
