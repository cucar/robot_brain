/**
 * Brain entry point — loads the Rust native addon via N-API.
 */

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const nativePath = path.resolve(__dirname, '../brain-napi/brain-napi.node');
const native = require(nativePath);
const Brain = native.Brain;

export default Brain;
