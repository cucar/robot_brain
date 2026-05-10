/**
 * Brain entry point — loads the Rust native addon when available, falls back to
 * the pure-JS implementation. The host code (`import Brain from 'brain'`) gets
 * the same API shape either way.
 */

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import JsBrain from './brain.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

let Brain;
try {
	const nativePath = path.resolve(__dirname, '../../brain-rust/brain-napi/brain-napi.node');
	const native = require(nativePath);
	Brain = native.Brain;
	console.log('🦀 Using Rust brain core');
} catch (e) {
	Brain = JsBrain;
	console.log('📦 Using JS brain core');
}

export default Brain;
