/**
 * Compatibility entry point - `node run-brain.js <job-name> [flags]` still works.
 * Each job file self-runs via its `runJob(...)` tail call, so all this has to do
 * is dynamic-import the requested job module. Prefer invoking job files directly:
 *   node jobs/stock-test.js --database
 */

const jobName = process.argv[2];
if (!jobName) {
	console.error('Usage: node run-brain.js <job-name> [flags]');
	process.exit(1);
}

try {
	await import(`./jobs/${jobName}.js`);
}
catch (error) {
	if (error.code === 'ERR_MODULE_NOT_FOUND') console.error(`Job not found: ./jobs/${jobName}.js`);
	else console.error(`Job failed: ${jobName}`, error);
	process.exit(1);
}
