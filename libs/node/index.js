/**
 * Node lib surface for the machine-intelligence brain.
 *
 * While the core is still JavaScript, this file re-exports the current brain
 * modules so app code imports from a stable package name. Once the Rust core
 * lands, the underlying path flips to a compiled NAPI module and the public
 * surface here stays the same.
 */

export { default as Brain } from '../../brain/brain.js';
export { Quantizer } from '../../brain/quantizer.js';
export { Job } from './src/job.js';
export { runJob, parseBrainArgs } from './src/run.js';
