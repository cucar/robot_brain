/**
 * Compatibility entry point - `node run-brain.js <job-name> [flags]` still works.
 * Dynamic-imports the requested job module (which does NOT auto-run, because its
 * tail-call `runJob` guards on being the entry point) and then runs it via
 * `executeJob`. Prefer invoking job files directly:
 *   node jobs/stock-test.js --database
 */

import { executeJob } from '#brain-node';

const jobName = process.argv[2];
if (!jobName) {
	console.error('Usage: node run-brain.js <job-name> [flags]');
	process.exit(1);
}

let jobModule;
try {
	jobModule = await import(`./jobs/${jobName}.js`);
}
catch (error) {
	if (error.code === 'ERR_MODULE_NOT_FOUND') console.error(`Job not found: ./jobs/${jobName}.js`);
	else console.error(`Failed to load job: ${jobName}`, error);
	process.exit(1);
}

const JobClass = jobModule.default;
if (!JobClass) {
	console.error(`Job class (default export) not found in ./jobs/${jobName}.js`);
	process.exit(1);
}

await executeJob(JobClass);
