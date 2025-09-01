import Job from './job.js';
import ArmChannel from '../channels/arm.js';

export default class ArmJob1 extends Job {

	getChannels() {
		return [
			{
				name: 'right_arm',
				channelClass: ArmChannel
			}
		];
	}

	async runTest() {
		console.log('Starting arm motor control learning...');
		
		// Process frames through the brain until arm channel has no more data
		let frameCount = 0;
		const maxFrames = 30; // Allow for multiple reaching trials
		
		while (frameCount < maxFrames) {
			const outputs = await this.processFrames();
			
			if (!outputs) {
				console.log('No more motor control data available from arm');
				break;
			}
			
			console.log(`Frame ${frameCount} outputs:`, outputs);
			frameCount++;
		}
		
		console.log(`Completed ${frameCount} frames of motor control learning`);
	}
}
