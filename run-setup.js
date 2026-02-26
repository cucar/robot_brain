#!/usr/bin/env node

/**
 * Job Setup Runner
 *
 * Runs the setup() method for a job to download/prepare data.
 * This is separate from run-brain.js which runs the actual job.
 *
 * Usage:
 *   node run-setup.js <job-name>
 *
 * Example:
 *   node run-setup.js stock-test
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
	// Get job name from command line
	const jobName = process.argv[2];

	// Parse --timeframe parameter
	let timeframe = null;
	const timeframeIndex = process.argv.indexOf('--timeframe');
	if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) timeframe = process.argv[timeframeIndex + 1];

	// Parse --start parameter
	let start = null;
	const startIndex = process.argv.indexOf('--start');
	if (startIndex !== -1 && process.argv[startIndex + 1]) start = process.argv[startIndex + 1];

	// Parse --end parameter
	let end = null;
	const endIndex = process.argv.indexOf('--end');
	if (endIndex !== -1 && process.argv[endIndex + 1]) end = process.argv[endIndex + 1];

	if (!jobName) {
		console.error('❌ Error: Job name required');
		console.log('\nUsage: node run-setup.js <job-name>');
		console.log('\nAvailable jobs:');
		
		// List available jobs
		const jobsDir = path.join(__dirname, 'jobs');
		const files = fs.readdirSync(jobsDir);
		for (const file of files) {
			if (file.endsWith('.js') && file !== 'job.js') {
				const name = file.replace('.js', '');
				console.log(`  - ${name}`);
			}
		}
		
		process.exit(1);
	}
	
	// Load the job class
	const jobPath = path.join(__dirname, 'jobs', `${jobName}.js`);
	
	if (!fs.existsSync(jobPath)) {
		console.error(`❌ Error: Job file not found: ${jobPath}`);
		process.exit(1);
	}
	
	console.log(`🔧 Setting up job: ${jobName}`);
	console.log('');
	
	try {
		// Import the job class (convert to file:// URL for Windows compatibility)
		const jobUrl = new URL(`file:///${jobPath.replace(/\\/g, '/')}`);
		const jobModule = await import(jobUrl);
		const JobClass = jobModule.default;
		
		// Create job instance and apply options
		const job = new JobClass();
		const options = { timeframe, start, end };
		job.options = options;
		if (typeof job.applyOptions === 'function') job.applyOptions(options);

		// Check if job has setup method
		if (typeof job.setup !== 'function') {
			console.log(`ℹ️  Job '${jobName}' does not have a setup() method.`);
			console.log('   No setup required for this job.');
			process.exit(0);
		}
		
		// Run setup
		console.log(`⚙️  Running setup for '${jobName}'...`);
		console.log('');
		
		await job.setup();
		
		console.log('');
		console.log(`✅ Setup complete for '${jobName}'`);
		process.exit(0);
	}
	catch (error) {
		console.error('');
		console.error(`❌ Setup failed: ${error.message}`);
		console.error(error.stack);
		process.exit(1);
	}
}

main();

