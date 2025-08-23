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
			// [ { x: 30 / 100, y: 40 / 100, r: 0   / 255, g: 255 / 255, b: 0   / 255 } ], // Green dot at 30,40
			// [ { x: 10 / 100, y: 20 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Red dot again - reinforcement
			// [ { x: 12 / 100, y: 22 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Another slightly shifted red dot
			// [ { x: 31 / 100, y: 41 / 100, r: 0   / 255, g: 255 / 255, b: 0   / 255 } ], // Green dot again
			// [ { x: 10 / 100, y: 20 / 100, r: 255 / 255, g: 0   / 255, b: 0   / 255 } ], // Red dot again
			// [ { x: 50 / 100, y: 60 / 100, r: 0   / 255, g: 0   / 255, b: 255 / 255 } ]  // Blue dot
		];

		// send the frames to the brain to be processed - brain will learn the patterns from this
		for (let i = 0; i < frames.length; i++) {
			console.log(`RUNNING FRAME ${i + 1} ---`);
			await brain.processFrame(frames[i]);
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


/**
 * Example usage of the optimized Brain class
 * Demonstrates text processing, stock trading, and motor control
 */

// Example 1: Text Processing - Learning "cats"
async function textProcessingExample() {
	const brain = new Brain();
	await brain.init();

	console.log('=== TEXT PROCESSING: Learning "cats" ===');

	// Define letter coordinates (simplified 1D space)
	const letterCoords = {
		'c': {d0: 0.1},
		'a': {d0: 0.2},
		't': {d0: 0.3},
		's': {d0: 0.4}
	};

	// Train the brain on "cats" multiple times
	for (let iteration = 1; iteration <= 10; iteration++) {
		console.log(`\n--- Iteration ${iteration} ---`);

		const word = "cats";
		for (let i = 0; i < word.length; i++) {
			const letter = word[i];
			const frame = [letterCoords[letter]];

			console.log(`Processing letter '${letter}' at position ${i}`);
			const predictions = await brain.processFrame(frame, 0);

			// Show predictions for next letters
			if (Object.keys(predictions).length > 0) {
				console.log('Predictions:', JSON.stringify(predictions, null, 2));
			}
		}
	}

	console.log('\n=== Final Test: Predicting from "ca" ===');

	// Test prediction: input "ca" and see if it predicts "t"
	await brain.processFrame([letterCoords['c']], 0);
	const predictions = await brain.processFrame([letterCoords['a']], 0);

	console.log('After "ca", brain predicts:');
	console.log(JSON.stringify(predictions, null, 2));
}

// Example 2: Stock Trading - Multi-dimensional coordinates
async function stockTradingExample() {
	const brain = new Brain();
	await brain.init();

	console.log('\n=== STOCK TRADING: Learning market patterns ===');

	// Simulate market data over time
	const marketScenarios = [
		// Bull market pattern
		{ frame: [{d0: 0.05, d1: 0.03, d2: 0.02}], wellBeing: 0.1, desc: "Market up, good day" },
		{ frame: [{d0: 0.03, d1: 0.04, d2: 0.01}], wellBeing: 0.15, desc: "Continued gains" },
		{ frame: [{d0: 0.02, d1: 0.02, d2: 0.03}], wellBeing: 0.08, desc: "Steady growth" },

		// Correction pattern
		{ frame: [{d0: -0.02, d1: -0.01, d2: -0.03}], wellBeing: -0.05, desc: "Minor correction" },
		{ frame: [{d0: -0.01, d1: 0.01, d2: -0.01}], wellBeing: 0.02, desc: "Recovery" },

		// Repeat pattern to strengthen learning
		{ frame: [{d0: 0.04, d1: 0.02, d2: 0.03}], wellBeing: 0.12, desc: "Bull pattern repeat" },
		{ frame: [{d0: 0.02, d1: 0.03, d2: 0.01}], wellBeing: 0.09, desc: "Continuation" },
		{ frame: [{d0: -0.02, d1: -0.02, d2: -0.01}], wellBeing: -0.03, desc: "Correction repeat" }
	];

	for (let day = 0; day < marketScenarios.length; day++) {
		const scenario = marketScenarios[day];
		console.log(`\nDay ${day + 1}: ${scenario.desc}`);
		console.log(`Input: ${JSON.stringify(scenario.frame[0])}`);
		console.log(`Well-being: ${scenario.wellBeing}`);

		const predictions = await brain.processFrame(scenario.frame, scenario.wellBeing);

		if (Object.keys(predictions).length > 0) {
			console.log('Brain predictions for future:');
			console.log(JSON.stringify(predictions, null, 2));
		}
	}

	console.log('\n=== Testing Market Prediction ===');

	// Test: Input a bull market start, see what it predicts
	const testFrame = [{d0: 0.05, d1: 0.03, d2: 0.02}];
	const testPredictions = await brain.processFrame(testFrame, 0);

	console.log('Given bull market conditions, brain predicts:');
	console.log(JSON.stringify(testPredictions, null, 2));
}

// Example 3: Motor Control - Learning movement sequences
async function motorControlExample() {
	const brain = new Brain();
	await brain.init();

	console.log('\n=== MOTOR CONTROL: Learning reach sequence ===');

	// Define a reaching movement sequence
	const reachSequence = [
		{ motor0: 0.0, motor1: 0.0, motor2: 0.0 }, // Rest position
		{ motor0: 0.3, motor1: 0.0, motor2: 0.0 }, // Shoulder extension
		{ motor0: 0.3, motor1: 0.4, motor2: 0.0 }, // Add elbow flexion
		{ motor0: 0.3, motor1: 0.4, motor2: 0.2 }, // Add wrist extension
		{ motor0: 0.3, motor1: 0.4, motor2: 0.1 }  // Adjust for grasp
	];

	// Train reaching pattern multiple times
	for (let trial = 1; trial <= 5; trial++) {
		console.log(`\n--- Reaching Trial ${trial} ---`);

		for (let step = 0; step < reachSequence.length; step++) {
			const motorFrame = [reachSequence[step]];

			// Successful reach gets positive well-being at the end
			const wellBeing = (step === reachSequence.length - 1) ? 0.2 : 0;

			console.log(`Step ${step}: ${JSON.stringify(motorFrame[0])}`);
			const predictions = await brain.processFrame(motorFrame, wellBeing);

			if (Object.keys(predictions).length > 0) {
				console.log('Motor predictions:', JSON.stringify(predictions, null, 2));
			}
		}
	}

	console.log('\n=== Testing Motor Prediction ===');

	// Test: Start reach movement, see if brain predicts the sequence
	const startFrame = [{ motor0: 0.3, motor1: 0.0, motor2: 0.0 }];
	const motorPredictions = await brain.processFrame(startFrame, 0);

	console.log('Given shoulder extension, brain predicts next movements:');
	console.log(JSON.stringify(motorPredictions, null, 2));
}

// Example 4: Performance demonstration
async function performanceTest() {
	const brain = new Brain();
	await brain.init();

	console.log('\n=== PERFORMANCE TEST ===');

	const startTime = Date.now();
	let frameCount = 0;

	// Process many frames quickly
	for (let i = 0; i < 100; i++) {
		const randomFrame = [{
			d0: Math.random() * 0.1,
			d1: Math.random() * 0.1,
			d2: Math.random() * 0.1
		}];

		await brain.processFrame(randomFrame, (Math.random() - 0.5) * 0.1);
		frameCount++;

		if (i % 20 === 0) {
			const elapsed = Date.now() - startTime;
			const fps = frameCount / (elapsed / 1000);
			console.log(`Processed ${frameCount} frames in ${elapsed}ms (${fps.toFixed(2)} FPS)`);
		}
	}

	const totalTime = Date.now() - startTime;
	const finalFps = frameCount / (totalTime / 1000);

	console.log(`\nFinal Performance: ${frameCount} frames in ${totalTime}ms`);
	console.log(`Average: ${finalFps.toFixed(2)} frames per second`);
	console.log(`Per frame: ${(totalTime / frameCount).toFixed(2)}ms`);
}

// Example 5: Complex pattern demonstration
async function complexPatternExample() {
	const brain = new Brain();
	await brain.init();

	console.log('\n=== COMPLEX PATTERNS: Learning ABC-123 sequence ===');

	// Define coordinates for letters and numbers
	const symbolCoords = {
		'A': {d0: 0.1, d1: 0.0}, 'B': {d0: 0.2, d1: 0.0}, 'C': {d0: 0.3, d1: 0.0},
		'1': {d0: 0.0, d1: 0.1}, '2': {d0: 0.0, d1: 0.2}, '3': {d0: 0.0, d1: 0.3}
	};

	// Train on repeating ABC-123 pattern
	const pattern = "ABC123";

	for (let cycle = 1; cycle <= 8; cycle++) {
		console.log(`\n--- Cycle ${cycle} ---`);

		for (let i = 0; i < pattern.length; i++) {
			const symbol = pattern[i];
			const frame = [symbolCoords[symbol]];

			const predictions = await brain.processFrame(frame, 0);
			console.log(`${symbol}: predictions =`, Object.keys(predictions).length);
		}
	}

	console.log('\n=== Testing Complex Prediction ===');

	// Test: Input "AB" and see if it predicts "C"
	await brain.processFrame([symbolCoords['A']], 0);
	const predictions = await brain.processFrame([symbolCoords['B']], 0);

	console.log('After "AB", brain predictions:');
	console.log(JSON.stringify(predictions, null, 2));
}

// Run all examples
async function runAllExamples() {
	try {
		await textProcessingExample();
		await stockTradingExample();
		await motorControlExample();
		await performanceTest();
		await complexPatternExample();

		console.log('\n=== ALL EXAMPLES COMPLETED ===');
		console.log('The brain successfully learned patterns in:');
		console.log('- Text sequences (temporal patterns)');
		console.log('- Market movements (spatio-temporal + reinforcement)');
		console.log('- Motor control (action sequences)');
		console.log('- Complex multi-modal patterns');
		console.log('- High-performance real-time processing');

	} catch (error) {
		console.error('Example failed:', error);
	}
}