/**
 * Base Job Class - Common functionality for all episodes
 */
import process from 'node:process';
import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';
import Brain from '../../../brain/brain.js';
import { formatFrameSummary, formatStartFrame, formatVoteDebug } from './renderer.js';

export class Job {

	constructor() {
		this.brain = null; // Brain instance will be created in run() based on options
		this.isShuttingDown = false;
		this.database = false; // Default: skip database backup/restore for jobs (tests)
		this.jobStartTime = null;
		this.hasShownExecutionTime = false;

		// Render flags — read from options in run(). These used to live on the brain but
		// they're presentation concerns, so they belong on the host where the console.log
		// actually happens. `frameSummary` defaults to on (matches legacy behavior);
		// `diagnostic` and `debug` are opt-in.
		this.frameSummary = true;
		this.diagnostic = false;
		this.debug = false;

		// Interactive step-debug: when --wait is set, runFrame should pause for the
		// user to hit Enter between frames. Lazy-initialized readline interface lives
		// here on the host (not the brain core) since it's terminal I/O.
		this.wait = false;
		this.rl = null;
	}

	/**
	 * Wait for the user to press Enter before continuing (used between frames when
	 * --wait is set). No-op when --wait is off so jobs can call it unconditionally.
	 * Lives on the Job because it's interactive host I/O, not part of the brain core.
	 */
	waitForUser(message = 'Press Enter to continue') {
		if (!this.wait) return Promise.resolve();
		if (!this.rl) this.rl = createInterface({ input: stdin, output: stdout });
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Main run method - template method pattern with hooks for customization
	 */
	async run() {
		this.jobStartTime = Date.now();

		// Set up signal handlers for graceful shutdown
		this.setupSignalHandlers();

		try {
			// Apply command line options to job config (if job has applyOptions method)
			if (this.applyOptions && this.options) this.applyOptions(this.options);

			// Pull render flags off options (set by parseBrainArgs in run.js).
			this.frameSummary = !(this.options?.noSummary);
			this.diagnostic = !!this.options?.diagnostic;
			this.debug = !!this.options?.debug;
			this.wait = !!this.options?.wait;

			// Create brain instance
			this.brain = new Brain(this.options);

			// Apply database option if provided (overrides default)
			if (this.options?.database !== undefined) this.database = this.options.database;

			// Allow jobs to show custom startup info
			await this.showStartupInfo();

			// get channels defined by child class and register them with brain
			await this.registerBrainChannels();

			// initialize database connection in the brain
			await this.brain.initDB();

			// Handle brain reset strategy
			await this.handleBrainReset();

			// initialize brain (this will initialize channels and create dimensions)
			await this.brain.init();

			// Allow jobs to configure channels after brain initialization
			await this.configureChannels();

			// Execute the main job logic
			await this.executeJob();

			// Allow jobs to show custom results
			await this.showResults();

			// Backup brain state to MySQL before exiting
			await this.shutdown();
			this.showExecutionTime();
		}
		catch (error) {
			console.error('Job execution failed:', error);
			await this.shutdown(); // try to back up on error
			this.showExecutionTime();
			throw error;
		}
	}

	/**
	 * Set up signal handlers for graceful shutdown (Ctrl+C, kill, etc.)
	 */
	setupSignalHandlers() {

		// SIGINT = Ctrl+C (works on Windows and Unix)
		process.on('SIGINT', () => this.handleInterrupt('SIGINT'));

		// SIGTERM = kill command (Unix only, ignored on Windows)
		process.on('SIGTERM', () => this.handleInterrupt('SIGTERM'));
	}

	/**
	 * Interrupt handler. First press starts a graceful shutdown; if the caller hits
	 * Ctrl+C again before it completes, force-exit immediately so we don't hang on
	 * a slow backup or a synchronous brain step.
	 */
	async handleInterrupt(signal) {
		if (this.isShuttingDown) {
			console.log(`\nSecond ${signal} received, forcing exit.`);
			process.exit(130);
		}
		console.log(`\nReceived ${signal}, shutting down gracefully...`);
		await this.shutdown();
		this.showExecutionTime();
		process.exit(0);
	}

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

	showExecutionTime() {
		if (this.hasShownExecutionTime || this.jobStartTime === null) return;

		this.hasShownExecutionTime = true;
		const totalExecutionTime = Date.now() - this.jobStartTime;
		console.log(`\n⏱️  Total Execution Time: ${this.formatDuration(totalExecutionTime)}`);
	}

	/**
	 * Graceful shutdown - backup brain state
	 */
	async shutdown() {
		if (this.isShuttingDown) return;
		this.isShuttingDown = true;
		// Close readline if --wait opened one; otherwise the process hangs on the
		// open stdin handle even after backup completes.
		if (this.rl) { this.rl.close(); this.rl = null; }
		if (this.brain && this.database) await this.brain.backup();
	}

	/* ---------- Hooks ---------- */

	async showStartupInfo() {}

	async configureChannels() {}

	async handleBrainReset() {
		if (this.options?.reset) {
			console.log('Hard reset requested. Clearing all tables...');
			await this.brain.resetBrain();
		}
	}

	async executeJob() {
		throw new Error('Job must implement executeJob() method');
	}

	async showResults() {}

	/**
	 * Per-frame host-side rendering entry point. Jobs call this after the brain
	 * processes a frame. Prints the start-of-frame diagnostic dump (if --diagnostic),
	 * the vote debug dump (if --debug), and the one-line summary (unless --no-summary).
	 * Any of the three is a no-op when its flag is off.
	 *
	 * @param {{ elapsed: number, voteDebug: object|null }} frame - per-frame byproducts
	 *   returned alongside inferences from brain.processFrame
	 */
	renderFrame(frame) {
		if (this.diagnostic) {
			const info = this.brain.getStartFrameInfo();
			const out = formatStartFrame(info);
			if (out) console.log(out);
		}

		if (this.debug) {
			const out = formatVoteDebug(frame?.voteDebug, this.getChannelFormatters());
			if (out) console.log(out);
		}

		if (this.frameSummary) {
			const summary = this.brain.getFrameSummary();
			const tail = this.getFrameSummaryTail();
			console.log(formatFrameSummary(summary, frame?.elapsed ?? 0, tail));
		}
	}

	/**
	 * Hook: app-layer state to append to the per-frame summary line. Default is
	 * empty (brain stats only). Stocks jobs override this to return the holdings
	 * list and portfolio P&L; text jobs override for per-symbol state; etc.
	 * @returns {string}
	 */
	getFrameSummaryTail() {
		return '';
	}

	/**
	 * Hook: return Map<channelName, formatter> the --debug vote-dump renderer uses
	 * to humanize action coordinates and bucket numbers. Each formatter exposes
	 * `name` and (optionally) `formatActionLabel(coord)` / `formatCoordinates(str)`.
	 *
	 * Default returns an empty map; jobs override to return a Map of their
	 * encoders/traders keyed by channel name.
	 * @returns {Map<string, object>}
	 */
	getChannelFormatters() {
		return new Map();
	}

	/**
	 * Register channels with the brain. Jobs override this to call
	 * brain.registerChannelSpec() for each channel they own.
	 */
	async registerBrainChannels() {}
}
