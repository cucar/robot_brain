import { Job } from './job.js';
import { TextChannel } from '../channels/text.js';

export default class TextJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'text',
				channelClass: TextChannel
			}
		];
	}

	async runTest() {
		console.log('Starting text processing and prediction learning...');
		
		// Process frames through the brain until text channel has no more data
		let frameCount = 0;
		const maxFrames = 50; // Allow for multiple iterations of word learning
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more text data available');
				break;
			}
			
			console.log(`Frame ${frameCount} outputs:`, outputs);
			frameCount++;
		}
		
		console.log(`Completed ${frameCount} frames of text learning`);
	}
}
