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
			console.log('starting new job...');
			this.brain = new Brain();

			// get channels defined by child class and register them with brain
			console.log('Registering channels with brain...');
			for (const channel of this.getChannels()) this.brain.registerChannel(channel.name, channel.channelClass);

			// initialize brain (this will initialize channels and create dimensions)
			console.log('Initializing brain...');
			await this.brain.init();

			// Reset brain context for clean episode
			await this.brain.resetContext();

			// process the job/episode
			console.log('Running episode...');
			await this.processFrames();
		}
		catch (error) {
			console.error('Job execution failed:', error);
			throw error;
		}
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

			// Process the frame through the brain and get the outputs that it deems fit
			const outputs = await this.brain.processFrame(frame);

			// now ask the channels to execute the outputs
			await this.brain.executeOutputs(outputs);
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