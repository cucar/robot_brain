import Brain from './brain.js';
import db from './db/db.js';

/**
 * main process feeding input to the brain
 */
async function main() {
	try {

		// initialize the brain with hyperparameters
		const brain = new Brain();

		// initialize database connection and load the dimensions in the brain
		await brain.init();

		// example frames of input data - assume normalized data coming from vision channel for now
		// const frames = new VisionChannel().processInput(inputData);
		const frames = [
			[ { x: 10 / 100, y: 20 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Red dot at 10,20
			[ { x: 11 / 100, y: 21 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Slightly shifted red dot
			[ { x: 30 / 100, y: 40 / 100, r: 0   / 255, g: 255 / 255, b: 0   / 255 } ], // Green dot at 30,40
			[ { x: 10 / 100, y: 20 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Red dot again - reinforcement
			[ { x: 12 / 100, y: 22 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Another slightly shifted red dot
			[ { x: 31 / 100, y: 41 / 100, r: 0   / 255, g: 255 / 255, b: 0   / 255 } ], // Green dot again
			[ { x: 10 / 100, y: 20 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Red dot again
			[ { x: 50 / 100, y: 60 / 100, r: 0   / 255, g: 0   / 255, b: 255 / 255 } ]  // Blue dot
		];

		// send the frames to the brain to be processed - brain will learn the patterns from this
		for (let i = 0; i < frames.length; i++) {
			console.log(`RUNNING FRAME ${i + 1} ---`);
			await brain.observeFrame(frames[i]);
		}
	}
	catch (error) {
		console.error('An unhandled error occurred:', error);
	}
	finally {
		await db.end(); // Close the database conn pool when done
		console.log('Database conn closed.');
	}
}

await main();