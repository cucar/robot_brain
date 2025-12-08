/**
 * Test correction and exploration override mechanisms
 * Verifies that both correction and exploration properly clean up inference sources
 */

import BrainMySQL from '../brain-mysql.js';
import StockChannel from '../channels/stock.js';

async function testCorrectionAndExploration() {
	console.log('🧪 Testing Correction and Exploration Override Mechanisms\n');

	const brain = new BrainMySQL();
	brain.registerChannel('TEST', StockChannel);
	await brain.init();
	await brain.resetBrain(); // Clean slate

	// Get the channel instance
	const channel = brain.channels.get('TEST');
	channel.owned = false; // Start not owned

	try {
		console.log('1️⃣ Testing overrideInference (unified method)...');

		// Create test neurons
		const originalCoords = { TEST_activity: 1 }; // BUY action
		const overrideCoords = { TEST_activity: -1 }; // SELL action
		const [originalNeuronId] = await brain.getFrameNeurons([originalCoords]);
		const [overrideNeuronId] = await brain.getFrameNeurons([overrideCoords]);
		console.log(`   Created neurons: original=${originalNeuronId}, override=${overrideNeuronId}`);

		// Manually insert some inference sources for original neuron
		await brain.conn.query(`
			INSERT INTO connection_inference_sources (inferred_neuron_id, level, age, connection_id, prediction_strength)
			VALUES (?, 0, 0, 999, 10.0)
		`, [originalNeuronId]);

		await brain.conn.query(`
			INSERT INTO pattern_inference_sources (inferred_neuron_id, level, age, pattern_neuron_id, connection_id, prediction_strength)
			VALUES (?, 0, 0, 888, 999, 5.0)
		`, [originalNeuronId]);

		await brain.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			VALUES (?, 0, 0, 100)
		`, [originalNeuronId]);

		// Verify original sources exist
		const [connBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM connection_inference_sources WHERE inferred_neuron_id = ?', [originalNeuronId]);
		const [patternBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference_sources WHERE inferred_neuron_id = ?', [originalNeuronId]);
		const [inferredBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ?', [originalNeuronId]);

		console.log(`   Before override: connection=${connBefore[0].count}, pattern=${patternBefore[0].count}, inferred=${inferredBefore[0].count}`);

		// Call unified override method
		await brain.overrideInference(overrideNeuronId, originalNeuronId);

		// Verify original sources are deleted and override is in place
		const [connAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM connection_inference_sources WHERE inferred_neuron_id = ?', [originalNeuronId]);
		const [patternAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference_sources WHERE inferred_neuron_id = ?', [originalNeuronId]);
		const [inferredOriginal] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [originalNeuronId]);
		const [inferredOverride] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [overrideNeuronId]);
		const [exploreOverride] = await brain.conn.query('SELECT COUNT(*) as count FROM exploration_inference_sources WHERE inferred_neuron_id = ? AND age = 0', [overrideNeuronId]);

		console.log(`   After override: original inferred=${inferredOriginal[0].count}, override inferred=${inferredOverride[0].count}, override sources=${exploreOverride[0].count}`);

		if (connAfter[0].count === 0 && patternAfter[0].count === 0 && inferredOriginal[0].count === 0 && inferredOverride[0].count === 1)
			console.log('   ✅ overrideInference works correctly\n');
		else
			console.log('   ❌ overrideInference failed\n');

		console.log('2️⃣ Testing exploration (overrideInference without original)...');

		// Create exploration neuron
		const exploreCoords = { TEST_activity: 0 }; // HOLD
		const [exploreNeuronId] = await brain.getFrameNeurons([exploreCoords]);

		// Call overrideInference for exploration (no original neuron)
		await brain.overrideInference(exploreNeuronId);

		// Verify it was saved
		const [inferredCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [exploreNeuronId]);
		const [chainCheck] = await brain.conn.query('SELECT * FROM inference_chain WHERE base_neuron_id = ? AND age = 0', [exploreNeuronId]);
		const [exploreSourceCheck] = await brain.conn.query('SELECT * FROM exploration_inference_sources WHERE inferred_neuron_id = ? AND age = 0', [exploreNeuronId]);

		console.log(`   Inferred neurons: ${inferredCheck.length}, Inference chain: ${chainCheck.length}, Exploration sources: ${exploreSourceCheck.length}`);

		if (inferredCheck.length > 0 && chainCheck.length > 0)
			console.log('   ✅ Exploration override works correctly\n');
		else
			console.log('   ❌ Exploration override failed\n');

		console.log('3️⃣ Testing correctInferredActions (uses overrideInference)...');

		// Create new neurons for correction test
		const corrOriginalCoords = { TEST_activity: 1 }; // BUY
		const corrCorrectedCoords = { TEST_activity: -1 }; // SELL
		const [corrOriginalNeuronId] = await brain.getFrameNeurons([corrOriginalCoords]);

		// Clean up and add original to inferred_neurons with connection sources (not exploration)
		await brain.conn.query('DELETE FROM inferred_neurons WHERE neuron_id = ?', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES (?, 0, 0, 100)', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO connection_inference_sources (inferred_neuron_id, level, age, connection_id, prediction_strength) VALUES (?, 0, 0, 999, 10.0)', [corrOriginalNeuronId]);

		// Apply correction
		const corrections = [{ originalNeuronId: corrOriginalNeuronId, correctedCoordinates: corrCorrectedCoords }];
		await brain.correctInferredActions(corrections);

		// Verify original is removed and corrected is added
		const [originalCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [corrOriginalNeuronId]);
		const [correctedCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE age = 0 AND level = 0');

		console.log(`   Original neuron in inferred_neurons: ${originalCheck.length}`);
		console.log(`   Total inferred neurons: ${correctedCheck.length}`);

		if (originalCheck.length === 0 && correctedCheck.length > 0)
			console.log('   ✅ correctInferredActions works correctly\n');
		else
			console.log('   ❌ correctInferredActions failed\n');

		console.log('4️⃣ Testing that corrections skip exploration neurons...');

		// Create an exploration neuron
		const skipCoords = { TEST_activity: 0 }; // HOLD
		const [skipNeuronId] = await brain.getFrameNeurons([skipCoords]);

		// Manually mark it as exploration by inserting exploration_inference_sources
		// (simpler than setting up active neurons and connections)
		await brain.conn.query('DELETE FROM inferred_neurons WHERE neuron_id = ?', [skipNeuronId]);
		await brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES (?, 0, 0, 1000000)', [skipNeuronId]);
		await brain.conn.query('INSERT INTO exploration_inference_sources (inferred_neuron_id, age, connection_id, prediction_strength) VALUES (?, 0, 999, 10.0)', [skipNeuronId]);

		// Try to correct it (should be skipped)
		const skipCorrections = [{ originalNeuronId: skipNeuronId, correctedCoordinates: { TEST_activity: 1 } }];
		await brain.correctInferredActions(skipCorrections);

		// Verify the exploration neuron is still there (not corrected)
		const [skipCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [skipNeuronId]);
		const [exploreSourceStillThere] = await brain.conn.query('SELECT * FROM exploration_inference_sources WHERE inferred_neuron_id = ? AND age = 0', [skipNeuronId]);

		console.log(`   Exploration neuron still in inferred_neurons: ${skipCheck.length}`);
		console.log(`   Exploration sources still present: ${exploreSourceStillThere.length}`);

		if (skipCheck.length > 0 && exploreSourceStillThere.length > 0)
			console.log('   ✅ Corrections correctly skip exploration neurons\n');
		else
			console.log('   ❌ Corrections did not skip exploration neurons\n');

		console.log('✅ All tests completed!');

	} finally {
		if (brain.conn) await brain.conn.end();
	}
}

testCorrectionAndExploration().catch(console.error);

