/**
 * Base Job Class - Common functionality for all episodes
 */
import path from 'node:path';
import process from 'node:process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { stdin, stdout } from 'node:process';
import Brain from 'brain';
import { formatFrameSummary, formatStartFrame, formatVotes } from './renderer.js';

export class Job {

	constructor() {
		this.brain = null; // Brain instance will be created in run() based on options
		this.isShuttingDown = false;
		this.saveBrain = null; // --save-brain <label>: write a backup on shutdown
		this.loadBrain = null; // --load-brain <label>: restore backup before executing
		this.saveContext = null; // --save-context <label>: save runtime context on shutdown
		this.loadContext = null; // --load-context <label>: restore runtime context
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

			// Debug rendering consumes per-vote detail; flip the brain's emit-votes
			// flag so processFrame populates result.votes. Without this the renderer
			// gets nothing and the dump is empty.
			if (this.debug) this.brain.setEmitVotes(true);

			// Backup options: --save-brain <label> writes a snapshot on shutdown,
			// --load-brain <label> restores a labeled backup before the first frame.
			// --save-context / --load-context do the same for runtime context.
			this.saveBrain = this.options?.saveBrain || null;
			this.loadBrain = this.options?.loadBrain || null;
			this.saveContext = this.options?.saveContext || null;
			this.loadContext = this.options?.loadContext || null;

			// Allow jobs to show custom startup info
			await this.showStartupInfo();

			// Job-specific initialization: register brain channels, configure statics, etc.
			await this.initialize();

			// Handle brain reset strategy
			await this.handleBrainReset();

			// --load-brain: restore a labeled backup into the brain. Channel
			// specs are already registered above; the backup just reconciles
			// id↔name maps and rehydrates neurons.
			if (this.loadBrain) this.brain.load(this.getJobDir(), this.loadBrain);

			// --load-context: restore the memory window (active neurons, rewards,
			// inferred neurons). Must happen after load-brain so neurons exist.
			if (this.loadContext) this.brain.loadContext(this.getJobDir(), this.loadContext);

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
		// Context must be saved BEFORE brain — brain.save() materializes decay
		// and resets frame_number to 0, which would destroy the context state.
		if (this.brain && this.saveContext) this.brain.saveContext(this.getJobDir(), this.saveContext);
		if (this.brain && this.saveBrain) this.brain.save(this.getJobDir(), this.saveBrain);
	}

	/* ---------- Hooks ---------- */

	async showStartupInfo() {}

	async configureChannels() {}

	async handleBrainReset() {
		if (this.options?.reset) {
			console.log('Hard reset requested. Clearing brain state...');
			this.brain.resetBrain();
		}
	}

	/**
	 * Folder where this job's backups live: <job-file's dir>/<job-file stem>/.
	 * For apps/stocks/jobs/test.js this resolves to apps/stocks/jobs/test/, so
	 * each job gets its own backup namespace and multiple jobs in the same
	 * directory don't collide. Resolved from the static `moduleUrl` set by
	 * runJob(). Falls back to cwd if not set.
	 */
	getJobDir() {
		const url = this.constructor.moduleUrl;
		if (!url) return process.cwd();
		const filepath = fileURLToPath(url);
		const dir = path.dirname(filepath);
		const stem = path.basename(filepath, path.extname(filepath));
		return path.join(dir, stem);
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
	 * @param {{ inferences: Map, votes: Array, frame: { elapsed: number, timings: object } }} result
	 *   the full FrameResult returned by brain.processFrame
	 */
	renderFrame(result) {
		const frame = result?.frame;
		if (this.diagnostic) {
			const info = this.brain.getStartFrameInfo();
			const out = formatStartFrame(info);
			if (out) console.log(out);
		}

		if (this.debug) {
			// The dim-name map is static after channel registration, but only
			// exposed via getStartFrameInfo (which is null on empty frames).
			// Cache the first non-null lookup so vote rendering still works
			// on frames where the snapshot would be unavailable.
			if (!this._dimNameCache) {
				const info = this.brain.getStartFrameInfo();
				if (info) this._dimNameCache = info.dimensionIdToName;
			}
			const out = formatVotes(
				result?.votes,
				result?.inferences ?? new Map(),
				this.getChannelFormatters(),
				this._dimNameCache ?? {},
			);
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
	 * Hook: return Map<channelId, formatter> the --debug vote-dump renderer uses
	 * to humanize action coordinates and bucket numbers. Each formatter exposes
	 * `name` and (optionally) `formatActionLabel(coord)` / `formatCoordinates(str)`.
	 *
	 * Default returns an empty map; jobs override to return a Map of their
	 * encoders/traders keyed by channelId.
	 * @returns {Map<number, object>}
	 */
	getChannelFormatters() {
		return new Map();
	}

	/**
	 * Job-specific initialization hook. Jobs override this to configure
	 * shared state, register brain channels, and create encoders/traders.
	 */
	async initialize() {}
}
