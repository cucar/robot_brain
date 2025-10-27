/**
 * Base Job Class - Common functionality for all episodes
 */
import Brain from '../brain-inmemory.js';  // Using in-memory brain for performance

export default class Job {
	constructor() {
		console.log('starting new job...');
		this.brain = new Brain();
		this.hardReset = false;
	}

	/**
	 * Main run method - template method pattern with hooks for customization
	 */
	async run() {
		try {
			// Allow jobs to show custom startup info
			await this.showStartupInfo();

			// get channels defined by child class and register them with brain
			console.log('Registering channels with brain...');
			for (const channel of this.getChannels()) this.brain.registerChannel(channel.name, channel.channelClass);

			// initialize brain (this will initialize channels and create dimensions)
			console.log('Initializing brain...');
			await this.brain.init();

			// Allow jobs to configure channels after brain initialization
			await this.configureChannels();

			// Handle brain reset strategy
			await this.handleBrainReset();

			// Execute the main job logic
			await this.executeJob();

			// Allow jobs to show custom results
			await this.showResults();
		}
		catch (error) {
			console.error('Job execution failed:', error);
			throw error;
		}
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
		while (true) {

			// Get combined frame from all channels
			const frame = await this.brain.getFrame();

			// If no input data from any channel, we're done
			if (!frame || frame.length === 0) {
				console.log('Completed processing. no more channel data.');
				return;
			}

			console.log(`Processing frame: ${frame.length} neurons`);

			// Get feedback from all channels for reward propagation
			const feedback = await this.brain.getFeedback();

			// Process the frame through the brain with feedback (executes outputs internally)
			await this.brain.processFrame(frame, feedback);
		}
	}

	/**
	 * Override this to define which channels the job uses
	 * Returns array of: { name, channelClass }
	 */
	getChannels() {
		throw new Error('Job must implement getChannels() method');
	}
}