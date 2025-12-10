/**
 * Test correction and exploration override mechanisms
 * Verifies that both correction and exploration properly clean up inference sources
 * Updated to use unified inference_sources table
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

		// Manually insert some inference sources for original neuron (using unified table)
		await brain.conn.query(`
			INSERT INTO inference_sources (age, base_neuron_id, source_type, source_id, inference_strength)
			VALUES (0, ?, 'connection', 999, 10.0)
		`, [originalNeuronId]);

		await brain.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			VALUES (?, 0, 0, 100)
		`, [originalNeuronId]);

		// Verify original sources exist
		const [sourcesBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM inference_sources WHERE base_neuron_id = ?', [originalNeuronId]);
		const [inferredBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ?', [originalNeuronId]);

		console.log(`   Before override: sources=${sourcesBefore[0].count}, inferred=${inferredBefore[0].count}`);

		// Call unified override method
		await brain.overrideInference(overrideNeuronId, originalNeuronId);

		// Verify original sources are deleted and override is in place
		const [sourcesOriginalAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM inference_sources WHERE base_neuron_id = ?', [originalNeuronId]);
		const [inferredOriginal] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [originalNeuronId]);
		const [inferredOverride] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [overrideNeuronId]);
		const [sourcesOverride] = await brain.conn.query('SELECT COUNT(*) as count FROM inference_sources WHERE base_neuron_id = ? AND age = 0', [overrideNeuronId]);

		console.log(`   After override: original sources=${sourcesOriginalAfter[0].count}, original inferred=${inferredOriginal[0].count}, override inferred=${inferredOverride[0].count}, override sources=${sourcesOverride[0].count}`);

		if (sourcesOriginalAfter[0].count === 0 && inferredOriginal[0].count === 0 && inferredOverride[0].count === 1)
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
		// Note: inference_sources may or may not have entries depending on whether connections exist

		console.log(`   Inferred neurons: ${inferredCheck.length}`);

		if (inferredCheck.length > 0)
			console.log('   ✅ Exploration override works correctly\n');
		else
			console.log('   ❌ Exploration override failed\n');

		console.log('3️⃣ Testing correctInferredActions (uses overrideInference)...');

		// Create new neurons for correction test
		const corrOriginalCoords = { TEST_activity: 1 }; // BUY
		const corrCorrectedCoords = { TEST_activity: -1 }; // SELL
		const [corrOriginalNeuronId] = await brain.getFrameNeurons([corrOriginalCoords]);

		// Clean up and add original to inferred_neurons with connection sources
		await brain.conn.query('DELETE FROM inferred_neurons WHERE neuron_id = ?', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES (?, 0, 0, 100)', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO inference_sources (age, base_neuron_id, source_type, source_id, inference_strength) VALUES (0, ?, \'connection\', 999, 10.0)', [corrOriginalNeuronId]);

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

		console.log('✅ All tests completed!');

	} finally {
		if (brain.conn) await brain.conn.end();
	}
}

testCorrectionAndExploration().catch(console.error);

