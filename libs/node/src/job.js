/**
 * Base Job Class - Common functionality for all episodes
 */
import 'dotenv/config'; // Load environment variables from .env file
import process from 'node:process';
import Brain from '../../../brain/brain.js';

export class Job {

	constructor() {
		this.brain = null; // Brain instance will be created in run() based on options
		this.isShuttingDown = false;
		this.database = false; // Default: skip database backup/restore for jobs (tests)
		this.jobStartTime = null;
		this.hasShownExecutionTime = false;
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

			// Create brain instance
			this.brain = new Brain(this.options);

			// Apply database option if provided (overrides default)
			if (this.options?.database !== undefined) this.database = this.options.database;

			// Allow jobs to show custom startup info
			await this.showStartupInfo();

			// get channels defined by child class and register them with brain
			// console.log('Registering channels with brain...');
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
	 * event handler for interrupt signals
	 */
	async handleInterrupt(signal) {
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
		if (this.brain && this.database) await this.brain.backup();
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
		// Check command-line flags for reset strategy
		if (this.options?.reset) {
			console.log('Hard reset requested. Clearing all tables...');
			await this.brain.resetBrain();
		}
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
		while (continueProcessing && !this.isShuttingDown)
			continueProcessing = await this.brain.processFrame();
		if (this.isShuttingDown) console.log('Processing interrupted by shutdown signal.');
		else console.log('Completed processing. no more channel data.');
	}

	/**
	 * Override this to define which channels the job uses
	 * Returns array of: { name, channelClass }
	 */
	getChannels() {
		throw new Error('Job must implement getChannels() method');
	}

	/**
	 * Hook: Register channels with the brain. Default path registers each getChannels()
	 * entry as a Channel class. Jobs that own their encoders/traders directly can override
	 * this to call brain.registerChannelSpec() per channel instead.
	 */
	async registerBrainChannels() {
		for (const channel of this.getChannels()) this.brain.registerChannel(channel.name, channel.channelClass);
	}
}
