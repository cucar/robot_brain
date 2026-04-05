/**
 * Job Runner - Entry point for running brain episodes
 * Usage: node run-brain.js <job-name> <options>
 */

class BrainRunner {
	async run(jobName, options = {}) {
		// console.log(`Starting job: ${jobName}`);

		try {
			// Dynamically import job class
			const jobModule = await import(`./jobs/${jobName}.js`);
			const JobClass = jobModule.default;
			if (!JobClass) throw new Error(`Job class not found in ./jobs/${jobName}.js`);

			// Create job and apply options to brain before running
			const job = new JobClass();

			// Store options on job so it can apply them to channels after they're registered
			job.options = options;

			await job.run();

			console.log(`Job completed: ${jobName}`);
			process.exit(0);
		}
		catch (error) {
			if (error.code === 'ERR_MODULE_NOT_FOUND') console.error(`Job not found: ./jobs/${jobName}.js`);
			else console.error(`Job failed: ${jobName}`, error);
			process.exit(1);
		}
	}
}

// Run if called directly
if (process.argv[2]) {
	const jobName = process.argv[2];

	// Parse --context-length
	let contextLength = null;
	const contextLengthIndex = process.argv.indexOf('--context-length');
	if (contextLengthIndex !== -1 && process.argv[contextLengthIndex + 1]) contextLength = parseInt(process.argv[contextLengthIndex + 1]);

	// Parse --forget-rate
	let patternForgetRate = null;
	const forgetRateIndex = process.argv.indexOf('--forget-rate');
	if (forgetRateIndex !== -1 && process.argv[forgetRateIndex + 1]) patternForgetRate = parseFloat(process.argv[forgetRateIndex + 1]);

	// Parse --error-threshold
	let errorCorrectionThreshold = null;
	const errorThresholdIndex = process.argv.indexOf('--error-threshold');
	if (errorThresholdIndex !== -1 && process.argv[errorThresholdIndex + 1]) errorCorrectionThreshold = parseFloat(process.argv[errorThresholdIndex + 1]);

	// Parse --merge-threshold
	let mergeThreshold = null;
	const mergeThresholdIndex = process.argv.indexOf('--merge-threshold');
	if (mergeThresholdIndex !== -1 && process.argv[mergeThresholdIndex + 1]) mergeThreshold = parseFloat(process.argv[mergeThresholdIndex + 1]);

	const options = {
		diagnostic: process.argv.includes('--diagnostic'),
		database: process.argv.includes('--database'),
		debug: process.argv.includes('--debug'),
		wait: process.argv.includes('--wait'),
		noSummary: process.argv.includes('--no-summary'),
		reset: process.argv.includes('--reset'),
		contextLength,
		patternForgetRate,
		errorCorrectionThreshold,
		mergeThreshold
	};

	const runner = new BrainRunner();
	runner.run(jobName, options);
}