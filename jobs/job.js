/**
 * Base Job Class - Common functionality for all episodes
 */
import Brain from '../brain.js';
import BrainMySQL from '../brain-mysql.js';

export default class Job {

	constructor() {
		this.brain = null; // Brain instance will be created in run() based on options
		this.hardReset = false;
		this.isShuttingDown = false;
		this.database = false; // Default: skip database backup/restore for jobs (tests)
	}

	/**
	 * Main run method - template method pattern with hooks for customization
	 */
	async run() {

		// Set up signal handlers for graceful shutdown
		this.setupSignalHandlers();

		try {
			// Create brain instance based on mysql option
			this.brain = this.runnerOptions?.mysql ? new BrainMySQL() : new Brain(this.runnerOptions);

			// Apply database option if provided (overrides default)
			if (this.runnerOptions?.database !== undefined) this.database = this.runnerOptions.database;

			// Allow jobs to show custom startup info
			await this.showStartupInfo();

			// get channels defined by child class and register them with brain
			// console.log('Registering channels with brain...');
			for (const channel of this.getChannels()) this.brain.registerChannel(channel.name, channel.channelClass);

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
		}
		catch (error) {
			console.error('Job execution failed:', error);
			await this.shutdown(); // try to back up on error
			throw error;
		}
	}

	/**
	 * Set up signal handlers for graceful shutdown (Ctrl+C, kill, etc.)
	 */
	setupSignalHandlers() {
		const handleSignal = async (signal) => {
			console.log(`\nReceived ${signal}, shutting down gracefully...`);
			await this.shutdown();
			process.exit(0);
		};

		// SIGINT = Ctrl+C (works on Windows and Unix)
		process.on('SIGINT', () => handleSignal('SIGINT'));

		// SIGTERM = kill command (Unix only, ignored on Windows)
		process.on('SIGTERM', () => handleSignal('SIGTERM'));
	}

	/**
	 * Graceful shutdown - backup brain state
	 */
	async shutdown() {
		if (this.isShuttingDown) return;
		this.isShuttingDown = true;
		if (this.brain && this.database) await this.brain.backupBrain();
	}

	/**
	 * Hook: Show startup information (override in subclasses)
	 */
	async showStartupInfo() {
		// Default: no custom startup info
	}

	/**
	 * Hook: Configure channels after brain initialization (override in subclasses)
	 */
	async configureChannels() {
		// Default: no custom channel configuration
	}

	/**
	 * Hook: Handle brain reset strategy (override in subclasses)
	 */
	async handleBrainReset() {
		// if job requests a hard reset (mainly for tests), perform before init
		if (this.hardReset) {
			console.log('Job requests hard reset. Clearing all tables...');
			await this.brain.resetBrain();
		}
		// otherwise, just reset brain memory for clean episode
		else await this.brain.resetContext();
	}

	/**
	 * Hook: Execute main job logic (override in subclasses)
	 */
	async executeJob() {
		// Default: single episode processing
		console.log('Running episode...');
		await this.processFrames();
	}

	/**
	 * Hook: Show results (override in subclasses)
	 */
	async showResults() {
		// Default: no custom results display
	}

	/**
	 * Process frames in a loop until all channels are exhausted
	 * Channels execute their own outputs and provide feedback based on state changes
	 */
	async processFrames() {
		let continueProcessing = true;
		while (continueProcessing) continueProcessing = await this.brain.processFrame();
		console.log('Completed processing. no more channel data.');
	}

	/**
	 * Override this to define which channels the job uses
	 * Returns array of: { name, channelClass }
	 */
	getChannels() {
		throw new Error('Job must implement getChannels() method');
	}
}