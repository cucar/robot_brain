import Job from './job.js';
import EarsChannel from '../channels/audio.js'; // Will be renamed from audio.js to ears.js

export default class EarsJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'ears',
				channelClass: EarsChannel
			}
		];
	}

	async runTest() {
		console.log('Starting audio localization learning...');
		
		// Process frames through the brain until ears channel has no more data
		let frameCount = 0;
		const maxFrames = 15; // Process all audio samples
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more audio data available from ears');
				break;
			}
			
			console.log(`Frame ${frameCount} outputs:`, outputs);
			frameCount++;
		}
		
		console.log(`Completed ${frameCount} frames of audio localization learning`);
	}
}
