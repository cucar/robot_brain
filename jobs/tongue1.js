import Job from './job.js';
import TongueChannel from '../channels/tongue.js';

export default class TongueJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'tongue',
				channelClass: TongueChannel
			}
		];
	}

	async runTest() {
		console.log('Starting taste preference learning...');
		
		// Process frames through the brain until tongue channel has no more data
		let frameCount = 0;
		const maxFrames = 15; // Process all taste samples
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more taste data available from tongue');
				break;
			}
			
			console.log(`Frame ${frameCount} outputs:`, outputs);
			frameCount++;
		}
		
		console.log(`Completed ${frameCount} frames of taste learning`);
	}
}
