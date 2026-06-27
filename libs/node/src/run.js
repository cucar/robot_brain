/**
 * Job runner entry point. A job file imports `runJob` from the lib, defines its
 * Job subclass, and ends with `await runJob(MyJob)` so running the file directly
 * (node path/to/job.js --flags) parses argv and executes the job.
 */

import process from 'node:process';
import { pathToFileURL } from 'node:url';

/**
 * Parse the well-known brain CLI flags out of argv into a plain options object.
 * Unknown args are ignored; the job subclass can read `process.argv` itself if
 * it needs job-specific flags.
 */
export function parseBrainArgs(argv = process.argv) {
	const has = flag => argv.includes(flag);
	const num = (flag, parser) => {
		const i = argv.indexOf(flag);
		return i !== -1 && argv[i + 1] !== undefined ? parser(argv[i + 1]) : null;
	};
	const str = flag => {
		const i = argv.indexOf(flag);
		return i !== -1 && argv[i + 1] !== undefined ? argv[i + 1] : null;
	};

	return {
		diagnostic: has('--diagnostic'),
		saveBrain: str('--save-brain'),
		loadBrain: str('--load-brain'),
		saveContext: str('--save-context'),
		loadContext: str('--load-context'),
		debug: has('--debug'),
		wait: has('--wait'),
		noSummary: has('--no-summary'),
		reset: has('--reset'),
		contextLength: num('--context-length', parseInt),
		patternForgetRate: num('--forget-rate', parseFloat),
		// The single grouping coefficient θ, shared by spatial and temporal: recognition / reuse fires at
		// similarity ≥ θ, correction fires at similarity < θ (error threshold = 1 − θ). groupMode
		// picks how the derived correction side adapts from per-unit Welford stats.
		groupThreshold: num('--group-threshold', parseFloat),
		groupMode: str('--group-mode'),
		regions: num('--regions', parseInt),
		columns: num('--columns', parseInt),
		consensus: str('--consensus')
	};
}

/**
 * Unconditionally run a Job class to completion and exit the process. Used by
 * external runners that dynamic-import a job module and drive it themselves.
 *
 * @param {Function} JobClass - Job subclass constructor
 * @param {object} [options] - if omitted, parsed from process.argv
 */
export async function executeJob(JobClass, options) {
	try {
		const job = new JobClass();
		job.options = options ?? parseBrainArgs();
		await job.run();
		process.exit(0);
	}
	catch (error) {
		console.error(`Job failed: ${JobClass.name}`, error);
		process.exit(1);
	}
}

/**
 * Run a Job class when its file is the program entry point. Meant to be called
 * at the bottom of a job file as `await runJob(import.meta, MyJob)`. If the
 * file was dynamic-imported by another module (e.g. run-setup.js), this is a
 * no-op so the caller can use the exported class without triggering a full run.
 *
 * @param {ImportMeta} meta - pass `import.meta` from the job file
 * @param {Function} JobClass - Job subclass constructor
 * @param {object} [options] - if omitted, parsed from process.argv
 */
export async function runJob(meta, JobClass, options) {

	// only run when this file is the entry point - dynamic imports should not auto-execute
	if (!meta || meta.url !== pathToFileURL(process.argv[1]).href) return;

	// Stash the job module URL so Job.getJobDir() can resolve <jobDir>/backups/.
	JobClass.moduleUrl = meta.url;
	await executeJob(JobClass, options);
}
