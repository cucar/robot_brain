/**
 * Node lib surface for the machine-intelligence brain.
 */

export { default as Brain } from 'brain';
export { Job } from './src/job.js';
export { runJob, executeJob, parseBrainArgs, readPositionals, readFlag, readNumber } from './src/run.js';
export { formatFixed, formatFloat } from './src/format.js';
