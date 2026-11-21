/**
 * The job runner.
 * A job file defines its Job subclass and ends with `await runJob(import.meta, MyJob)`, so running the file
 * (node path/to/job.js --flags) reads the command line and runs the job.
 */
import process from 'node:process';
import { pathToFileURL } from 'node:url';

/**
 * The value after a flag on the command line, or null when the flag is absent.
 */
function readFlagValue(argv, flag) {
	const at = argv.indexOf(flag);
	return at !== -1 && argv[at + 1] !== undefined ? argv[at + 1] : null;
}

/**
 * The value after a flag, as a whole number, or null when the flag is absent.
 */
function readWholeNumber(argv, flag) {
	const value = readFlagValue(argv, flag);
	return value === null ? null : parseInt(value);
}

/**
 * A job's own flag: the value after it on the command line, or the fallback where it is absent.
 */
export function readFlag(flag, fallback = null) {
	return readFlagValue(process.argv, flag) ?? fallback;
}

/**
 * A job's own flag as a number, or the fallback where it is absent.
 */
export function readNumber(flag, fallback = null) {
	const value = readFlagValue(process.argv, flag);
	return value === null ? fallback : parseFloat(value);
}

/**
 * Read the flags every job takes into an options object.
 * `--regions R` and `--columns C` set the dispatch, `--disable-learning` builds the brain with learning off,
 * and `--save-brain`, `--load-brain`, `--save-context`, `--load-context` name a backup or a context by its label.
 * A job reads its own arguments from `process.argv` itself.
 */
export function parseBrainArgs(argv = process.argv) {
	return {
		regions: readWholeNumber(argv, '--regions'),
		columns: readWholeNumber(argv, '--columns'),
		learning: argv.includes('--disable-learning') ? false : null,
		saveBrain: readFlagValue(argv, '--save-brain'),
		loadBrain: readFlagValue(argv, '--load-brain'),
		saveContext: readFlagValue(argv, '--save-context'),
		loadContext: readFlagValue(argv, '--load-context'),
	};
}

/**
 * The arguments that are not flags: everything left once each `--flag` and the value after it are taken out.
 * A flag that takes no value (`--disable-learning`) is named in `bareFlags`, so the argument after it is kept.
 */
export function readPositionals(argv = process.argv, bareFlags = ['--disable-learning']) {
	const args = argv.slice(2);
	const positionals = [];
	for (let i = 0; i < args.length; i++) {
		if (!args[i].startsWith('--')) positionals.push(args[i]);
		else if (!bareFlags.includes(args[i])) i++;
	}
	return positionals;
}

/**
 * Run a Job class to its end and exit the process: 0 when it ends, 1 when it fails.
 *
 * @param {Function} JobClass - the Job subclass
 * @param {object} [options] - read from the command line when absent
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
 * Run a Job class when its file is the program's entry point; do nothing when the file is imported.
 * The job file's URL is kept on the class, so the job finds its backups next to the file.
 *
 * @param {ImportMeta} meta - the job file's `import.meta`
 * @param {Function} JobClass - the Job subclass
 * @param {object} [options] - read from the command line when absent
 */
export async function runJob(meta, JobClass, options) {
	if (!meta || meta.url !== pathToFileURL(process.argv[1]).href) return;
	JobClass.moduleUrl = meta.url;
	await executeJob(JobClass, options);
}
