import { Job } from './job.js';
import { EyesChannel } from '../channels/vision.js'; // Will be renamed from vision.js to eyes.js

export default class VisionJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'eyes',
				channelClass: EyesChannel
			}
		];
	}

	async runTest() {
		console.log('Starting visual attention learning...');
		
		// Process frames through the brain until eyes channel has no more data
		let frameCount = 0;
		const maxFrames = 15; // Prevent infinite loops during testing
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more visual data available from eyes');
				break;
			}
			
			console.log(`Frame ${frameCount} outputs:`, outputs);
			frameCount++;
		}
		
		console.log(`Completed ${frameCount} frames of visual attention learning`);
	}
}
