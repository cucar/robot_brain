/**
 * MNIST IDX file loader. Parses the standard IDX binary format defined by
 * Yann LeCun's MNIST dataset (http://yann.lecun.com/exdb/mnist/). Handles
 * both raw and gzip-compressed files — the setup script downloads .gz files,
 * but if someone decompresses them first the loader works either way.
 *
 * IDX format overview:
 *   IDX3 (images): magic(4) + count(4) + rows(4) + cols(4) + pixel_data
 *   IDX1 (labels): magic(4) + count(4) + label_data
 *   All multi-byte integers are big-endian.
 */
import fs from 'node:fs';
import { gunzipSync } from 'node:zlib';

/**
 * Read a file from disk and decompress if gzipped. Gzip detection is by
 * magic bytes (0x1f 0x8b) so the caller doesn't need to know the format.
 * @param {string} filePath — absolute or relative path to the IDX file
 * @returns {Buffer}
 */
function readIdxFile(filePath) {
	let buf = fs.readFileSync(filePath);

	// Gzip magic bytes: 0x1f 0x8b at the start of the file.
	if (buf[0] === 0x1f && buf[1] === 0x8b) buf = gunzipSync(buf);

	return buf;
}

/**
 * Load MNIST images from an IDX3 file (images). Each image is returned as a
 * Uint8Array of 784 bytes (28×28 pixels, row-major, grayscale 0–255). The
 * arrays are subarray views into the parsed buffer — no per-image copy.
 *
 * @param {string} filePath — path to train-images-idx3-ubyte[.gz] or t10k variant
 * @returns {Uint8Array[]} — one entry per image, each 784 bytes
 */
export function loadImages(filePath) {
	const buf = readIdxFile(filePath);

	// IDX3 magic: high bytes are zero/type, low two bytes are 0x0803.
	const magic = buf.readUInt32BE(0);
	if ((magic & 0xFFFF) !== 0x0803)
		throw new Error(`Not an IDX3 image file: magic=0x${magic.toString(16)}`);

	const count = buf.readUInt32BE(4);
	const rows = buf.readUInt32BE(8);
	const cols = buf.readUInt32BE(12);
	const size = rows * cols;

	// Return subarray views (no copy) — the underlying buffer stays in memory
	// for the lifetime of the process, which is fine for MNIST's ~47 MB.
	const images = new Array(count);
	for (let i = 0; i < count; i++) {
		const offset = 16 + i * size;
		images[i] = buf.subarray(offset, offset + size);
	}
	return images;
}

/**
 * Load MNIST labels from an IDX1 file. Returns a Uint8Array view over the
 * label bytes — each value is a digit 0–9.
 *
 * @param {string} filePath — path to train-labels-idx1-ubyte[.gz] or t10k variant
 * @returns {Uint8Array} — one byte per image, value 0–9
 */
export function loadLabels(filePath) {
	const buf = readIdxFile(filePath);

	// IDX1 magic: low two bytes are 0x0801.
	const magic = buf.readUInt32BE(0);
	if ((magic & 0xFFFF) !== 0x0801)
		throw new Error(`Not an IDX1 label file: magic=0x${magic.toString(16)}`);

	const count = buf.readUInt32BE(4);
	return buf.subarray(8, 8 + count);
}
