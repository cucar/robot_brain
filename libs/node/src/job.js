/**
 * The Job: what every program that runs a brain has in common.
 * A job subclass declares its channels and runs its frames; the Job builds the brain from the command-line options,
 * restores a saved brain or context when asked, saves them on the way out, and times the run.
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import Brain from 'brain';

export class Job {

	constructor() {
		// the command-line options, set by the runner before run() (see run.js)
		this.options = {};
		// the brain, built in run()
		this.brain = null;
		// each declared channel by name, read from the brain once its channels are declared or loaded
		this.channels = new Map();
		// set once the shutdown begins, so a second interrupt exits at once
		this.isShuttingDown = false;
		this.startTime = null;
	}

	/* ---------- The run ---------- */

	/**
	 * Run the job from start to end.
	 * The brain is built, its channels declared or loaded, the job's frames run, its results shown, and the brain saved.
	 */
	async run() {
		this.startTime = Date.now();
		this.listenForInterrupts();
		try {
			this.brain = this.createBrain();
			this.prepareBrain();
			await this.executeJob();
			await this.showResults();
		}
		finally {
			this.shutdown();
			this.showExecutionTime();
		}
	}

	/**
	 * Build a brain: the job's own options, the dispatch from the command line, and learning on unless turned off.
	 */
	createBrain(learning = this.options.learning ?? true) {
		return new Brain({
			...this.getBrainOptions(),
			regions: this.options.regions ?? 1,
			columns: this.options.columns ?? 1,
			learning,
			insideFrame: this.options.insideFrame ?? true,
		});
	}

	/**
	 * Give the brain its channels, from a backup when one is named, or by declaring them.
	 * A backup carries the channels it was saved with, and loads only into a brain that has declared nothing.
	 * The context, when named, is loaded after the backup it was saved with.
	 */
	prepareBrain() {
		if (this.options.loadBrain) this.brain.load(this.getJobDir(), this.options.loadBrain);
		else this.declareChannels();
		if (this.options.loadContext) this.brain.loadContext(this.getJobDir(), this.options.loadContext);
		this.channels = this.readChannels(this.brain);
	}

	/**
	 * Each channel of a brain, by name.
	 */
	readChannels(brain) {
		return new Map(brain.listChannels().map(channel => [channel.name, channel]));
	}

	/**
	 * Save what the command line asked to save.
	 * The context is saved before the brain, so both are written from the same frame.
	 */
	shutdown() {
		if (this.isShuttingDown || !this.brain) return;
		this.isShuttingDown = true;
		if (this.options.saveContext) this.brain.saveContext(this.getJobDir(), this.options.saveContext);
		if (this.options.saveBrain) this.brain.save(this.getJobDir(), this.options.saveBrain);
	}

	/* ---------- Interrupts ---------- */

	/**
	 * Save on Ctrl+C or a kill, as at a normal end.
	 */
	listenForInterrupts() {
		process.on('SIGINT', () => this.handleInterrupt('SIGINT'));
		process.on('SIGTERM', () => this.handleInterrupt('SIGTERM'));
	}

	/**
	 * The first interrupt saves and exits.
	 * A second one, while the save is still running, exits at once.
	 */
	handleInterrupt(signal) {
		if (this.isShuttingDown) {
			console.error(`\nSecond ${signal} received, forcing exit.`);
			process.exit(130);
		}
		console.error(`\nReceived ${signal}, shutting down...`);
		this.shutdown();
		this.showExecutionTime();
		process.exit(0);
	}

	/* ---------- Timing ---------- */

	/**
	 * Print how long the run took.
	 * It goes to stderr, so a job's stdout holds only its results and two runs compare line by line.
	 */
	showExecutionTime() {
		if (this.startTime === null) return;
		console.error(`\n⏱️  Total Execution Time: ${this.formatDuration(Date.now() - this.startTime)}`);
		this.startTime = null;
	}

	/**
	 * A duration in milliseconds, written in the largest units that fit.
	 */
	formatDuration(durationMs) {
		if (durationMs < 1000) return `${durationMs}ms`;
		const totalSeconds = Math.floor(durationMs / 1000);
		const hours = Math.floor(totalSeconds / 3600);
		const minutes = Math.floor((totalSeconds % 3600) / 60);
		const seconds = totalSeconds % 60;
		if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
		if (minutes > 0) return `${minutes}m ${seconds}s`;
		return `${(durationMs / 1000).toFixed(2)}s`;
	}

	/* ---------- Folders ---------- */

	/**
	 * The folder of this job's backups and contexts: the job file's folder, then a folder named after the job file.
	 * For apps/stocks/jobs/test.js it is apps/stocks/jobs/test/, so jobs in one folder do not share backups.
	 * Without a job file, it is the current folder.
	 */
	getJobDir() {
		const url = this.constructor.moduleUrl;
		if (!url) return process.cwd();
		const file = fileURLToPath(url);
		return path.join(path.dirname(file), path.basename(file, path.extname(file)));
	}

	/* ---------- A copy to test ---------- */

	/**
	 * A copy of the brain with learning off, to test it without changing it.
	 * The brain and its context are saved to a folder of this process's own, loaded into a new brain, and the folder removed.
	 */
	copyBrainWithoutLearning() {
		const folder = path.join(this.getJobDir(), `copy-${process.pid}`);
		this.brain.save(folder, 'copy');
		this.brain.saveContext(folder, 'copy');
		const copy = this.createBrain(false);
		copy.load(folder, 'copy');
		copy.loadContext(folder, 'copy');
		fs.rmSync(folder, { recursive: true, force: true });
		return copy;
	}

	/* ---------- Hooks ---------- */

	/**
	 * The brain's own options: its history depth, its neighborhood policy, its action policy.
	 * The dispatch and learning come from the command line.
	 */
	getBrainOptions() {
		return {};
	}

	/**
	 * Declare the job's axes and channels.
	 * It is not called when a backup is loaded: the backup carries them.
	 */
	declareChannels() {}

	/**
	 * Run the job's frames.
	 */
	async executeJob() {
		throw new Error('Job must implement executeJob() method');
	}

	/**
	 * Print the job's results once its frames have run.
	 */
	async showResults() {}
}
