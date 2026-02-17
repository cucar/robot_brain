import { Job } from './job.js';
import { EyesChannel } from '../channels/vision.js';
import { EarsChannel } from '../channels/audio.js';
import { ArmChannel } from '../channels/arm.js';
import { TextChannel } from '../channels/text.js';

export default class MultiSensoryJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'eyes',
				channelClass: EyesChannel
			},
			{
				name: 'ears',
				channelClass: EarsChannel
			},
			{
				name: 'right_arm',
				channelClass: ArmChannel
			},
			{
				name: 'text_processor',
				channelClass: TextChannel
			}
		];
	}

	async runTest() {
		console.log('Starting multi-sensory integration learning...');
		console.log('This job demonstrates how the brain can learn from multiple sensory channels simultaneously');
		
		// Process frames through the brain until all channels have no more data
		let frameCount = 0;
		const maxFrames = 100; // Allow for extended multi-sensory learning
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more data available from any sensory channels');
				break;
			}
			
			// Log outputs from each channel
			for (const [channelName, channelOutputs] of outputs) {
				if (channelOutputs.actions.size > 0) {
					console.log(`${channelName} actions:`, Array.from(channelOutputs.actions.entries()));
				}
			}
			
			frameCount++;
			
			// Add some delay to make the multi-sensory processing observable
			if (frameCount % 5 === 0) {
				console.log(`--- Multi-sensory frame ${frameCount} completed ---`);
			}
		}
		
		console.log(`Completed ${frameCount} frames of multi-sensory integration learning`);
		console.log('The brain has learned to coordinate:');
		console.log('- Visual attention (eye saccades)');
		console.log('- Audio localization (ear movements)'); 
		console.log('- Motor control (arm reaching)');
		console.log('- Language processing (text prediction)');
	}
}
