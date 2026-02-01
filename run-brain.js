/**
 * Job Runner - Entry point for running brain episodes
 * Usage: node run-brain.js <job-name> [--diagnostic]
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

			// Apply options to brain
			if (options.diagnostic) job.brain.diagnostic = true;

			// Store options on job so it can apply them to channels after they're registered
			job.runnerOptions = options;

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
	const options = {
		diagnostic: process.argv.includes('--diagnostic'),
		mysql: process.argv.includes('--mysql'),
		database: process.argv.includes('--database'),
		debug: process.argv.includes('--debug'),
		noSummary: process.argv.includes('--no-summary')
	};

	const runner = new BrainRunner();
	runner.run(jobName, options);
}