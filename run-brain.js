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

	// Parse --episodes parameter
	let episodes = null;
	const episodesIndex = process.argv.indexOf('--episodes');
	if (episodesIndex !== -1 && process.argv[episodesIndex + 1]) episodes = parseInt(process.argv[episodesIndex + 1]);

	// Parse --holdout parameter
	let holdout = null;
	const holdoutIndex = process.argv.indexOf('--holdout');
	if (holdoutIndex !== -1 && process.argv[holdoutIndex + 1]) holdout = parseInt(process.argv[holdoutIndex + 1]);

	// Parse --offset parameter
	let offset = null;
	const offsetIndex = process.argv.indexOf('--offset');
	if (offsetIndex !== -1 && process.argv[offsetIndex + 1]) offset = parseInt(process.argv[offsetIndex + 1]);

	// Parse --timeframe parameter
	let timeframe = null;
	const timeframeIndex = process.argv.indexOf('--timeframe');
	if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) timeframe = process.argv[timeframeIndex + 1];

	const options = {
		diagnostic: process.argv.includes('--diagnostic'),
		database: process.argv.includes('--database'),
		debug: process.argv.includes('--debug'),
		noSummary: process.argv.includes('--no-summary'),
		reset: process.argv.includes('--reset'),
		episodes,
		holdout,
		offset,
		timeframe
	};

	const runner = new BrainRunner();
	runner.run(jobName, options);
}