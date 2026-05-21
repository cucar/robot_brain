/**
 * MNIST Data Setup — downloads the standard MNIST dataset (IDX format,
 * gzip-compressed) from Google's mirror into apps/mnist/data/.
 *
 * Run once before training:
 *   node apps/mnist/jobs/download.js
 *
 * Downloads four files (~11 MB total):
 *   train-images-idx3-ubyte.gz   60,000 training images (28×28 grayscale)
 *   train-labels-idx1-ubyte.gz   60,000 training labels (digit 0–9)
 *   t10k-images-idx3-ubyte.gz    10,000 test images
 *   t10k-labels-idx1-ubyte.gz    10,000 test labels
 *
 * Skips files that already exist so it's safe to re-run.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, '..', 'data');

// Google Cloud Storage mirror — the original site (yann.lecun.com) is
// sometimes unreliable; this mirror is stable and fast.
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist';

const FILES = [
	'train-images-idx3-ubyte.gz',
	'train-labels-idx1-ubyte.gz',
	't10k-images-idx3-ubyte.gz',
	't10k-labels-idx1-ubyte.gz'
];

/**
 * Download a single file from the MNIST mirror. Skips if the file already
 * exists on disk. Streams to disk via pipeline to avoid loading the entire
 * response into memory (though MNIST files are small enough that it
 * wouldn't matter — this is good hygiene).
 *
 * @param {string} filename — one of the four MNIST IDX filenames
 */
async function download(filename) {
	const dest = path.join(DATA_DIR, filename);
	if (fs.existsSync(dest)) {
		console.log(`  [skip] ${filename} already exists`);
		return;
	}

	const url = `${BASE_URL}/${filename}`;
	console.log(`  [download] ${filename} ...`);
	const response = await fetch(url);
	if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);

	await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(dest));
	const size = fs.statSync(dest).size;
	console.log(`  [done] ${filename} (${(size / 1024 / 1024).toFixed(1)} MB)`);
}

/**
 * Download MNIST files from the MNIST mirror.
 */
async function main() {
	fs.mkdirSync(DATA_DIR, { recursive: true });
	console.log(`Downloading MNIST dataset to ${DATA_DIR}\n`);

	for (const file of FILES) await download(file);

	console.log('\nMNIST data ready.');
}

main().catch(err => { console.error(err); process.exit(1); });