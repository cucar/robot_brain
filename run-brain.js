/**
 * Job Runner - Entry point for running brain episodes
 * Usage: node run-brain.js <job-name>
 */

class BrainRunner {
	async run(jobName) {
		console.log(`Starting job: ${jobName}`);

		try {
			// Dynamically import job class
			const jobModule = await import(`./jobs/${jobName}.js`);
			const JobClass = jobModule.default;
			if (!JobClass) throw new Error(`Job class not found in ./jobs/${jobName}.js`);

			// Create and run job (job handles brain initialization)
			const job = new JobClass();
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
	const runner = new BrainRunner();
	runner.run(process.argv[2]);
}