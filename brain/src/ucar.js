/**
 * UCAR entry point — loads the ucar-core native addon via N-API (docs/algorithm.md).
 */

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const nativePath = path.resolve(__dirname, '../ucar-napi/ucar-napi.node');
const native = require(nativePath);

export const Ucar = native.Ucar;
export default Ucar;
