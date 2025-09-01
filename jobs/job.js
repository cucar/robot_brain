/**
 * Base Job Class - Common functionality for all episodes
 */
import Brain from '../brain.js';

export default class Job {
	constructor() {
		this.brain = null;
	}

	/**
	 * Main run method - implemented in base class
	 */
	async run() {
		try {
			// Initialize brain
			console.log('Initializing brain...');
			this.brain = new Brain();

			// Get channels defined by child class and register them with brain
			console.log('Registering channels with brain...');
			const channels = this.getChannels();
			for (const channel of channels) this.brain.registerChannel(channel.name, channel.channelClass);

			// Initialize brain (this will call getDimensions on channels and create dimensions)
			await this.brain.init();

			// Reset brain context for clean episode
			await this.brain.resetContext();

			// process the job/episode
			console.log('Running episode...');
			await this.brain.processFrames();

		}
		catch (error) {
			console.error('Job execution failed:', error);
			throw error;
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