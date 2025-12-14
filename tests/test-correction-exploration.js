/**
 * Test correction and exploration override mechanisms
 * Verifies that both correction and exploration properly clean up inference sources
 * Updated to use org_inference_sources and base_inference_sources tables
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
		console.log('1️⃣ Testing overrideBaseInferenceSources (unified method)...');

		// Create test neurons
		const originalCoords = { TEST_activity: 1 }; // BUY action
		const overrideCoords = { TEST_activity: -1 }; // SELL action
		const [originalNeuronId] = await brain.getFrameNeurons([originalCoords]);
		const [overrideNeuronId] = await brain.getFrameNeurons([overrideCoords]);
		console.log(`   Created neurons: original=${originalNeuronId}, override=${overrideNeuronId}`);

		// Manually insert some inference sources for original neuron
		await brain.conn.query(`
			INSERT INTO org_inference_sources (age, inferred_neuron_id, level, source_type, source_id, inference_strength)
			VALUES (0, ?, 0, 'connection', 999, 10.0)
		`, [originalNeuronId]);
		await brain.conn.query(`
			INSERT INTO base_inference_sources (age, base_neuron_id, source_type, source_id, inference_strength)
			VALUES (0, ?, 'connection', 999, 10.0)
		`, [originalNeuronId]);

		await brain.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			VALUES (?, 0, 0, 100)
		`, [originalNeuronId]);

		// Verify original sources exist
		const [sourcesBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ?', [originalNeuronId]);
		const [inferredBefore] = await brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons WHERE neuron_id = ?', [originalNeuronId]);

		console.log(`   Before override: sources=${sourcesBefore[0].count}, inferred=${inferredBefore[0].count}`);

		// Call unified override method (only updates sources, not inferred_neurons)
		await brain.overrideBaseInferenceSources(overrideNeuronId, originalNeuronId);

		// Verify original sources are deleted and override sources are in place
		const [sourcesOriginalAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ?', [originalNeuronId]);
		const [sourcesOverride] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [overrideNeuronId]);

		console.log(`   After override: original sources=${sourcesOriginalAfter[0].count}, override sources=${sourcesOverride[0].count}`);

		if (sourcesOriginalAfter[0].count === 0)
			console.log('   ✅ overrideBaseInferenceSources works correctly (sources updated)\n');
		else
			console.log('   ❌ overrideBaseInferenceSources failed\n');

		console.log('2️⃣ Testing exploration (overrideBaseInferenceSources + saveExplorationNeuron)...');

		// Create exploration neuron
		const exploreCoords = { TEST_activity: 0 }; // HOLD
		const [exploreNeuronId] = await brain.getFrameNeurons([exploreCoords]);

		// Call overrideBaseInferenceSources for exploration (no original neuron)
		await brain.overrideBaseInferenceSources(exploreNeuronId, null);
		await brain.saveExplorationNeuron(exploreNeuronId);

		// Verify it was saved
		const [inferredCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [exploreNeuronId]);
		const [sourcesCheck] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [exploreNeuronId]);

		console.log(`   Inferred neurons: ${inferredCheck.length}, sources: ${sourcesCheck[0].count}`);

		if (inferredCheck.length > 0)
			console.log('   ✅ Exploration override works correctly\n');
		else
			console.log('   ❌ Exploration override failed\n');

		console.log('3️⃣ Testing correctInferredActions (only updates sources)...');

		// Create new neurons for correction test
		const corrOriginalCoords = { TEST_activity: 1 }; // BUY
		const corrCorrectedCoords = { TEST_activity: -1 }; // SELL
		const [corrOriginalNeuronId] = await brain.getFrameNeurons([corrOriginalCoords]);
		const [corrCorrectedNeuronId] = await brain.getFrameNeurons([corrCorrectedCoords]);

		// Clean up and add original to inferred_neurons with connection sources
		await brain.conn.query('DELETE FROM inferred_neurons WHERE neuron_id IN (?, ?)', [corrOriginalNeuronId, corrCorrectedNeuronId]);
		await brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES (?, 0, 0, 100)', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO org_inference_sources (age, inferred_neuron_id, level, source_type, source_id, inference_strength) VALUES (0, ?, 0, \'connection\', 999, 10.0)', [corrOriginalNeuronId]);
		await brain.conn.query('INSERT INTO base_inference_sources (age, base_neuron_id, source_type, source_id, inference_strength) VALUES (0, ?, \'connection\', 999, 10.0)', [corrOriginalNeuronId]);

		// Apply correction (only updates sources, not inferred_neurons)
		const corrections = [{ originalNeuronId: corrOriginalNeuronId, correctedCoordinates: corrCorrectedCoords, strength: 100 }];
		await brain.correctInferredActions(corrections);

		// Verify sources were updated but inferred_neurons was NOT changed yet
		const [originalSourcesAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [corrOriginalNeuronId]);
		const [correctedSourcesAfter] = await brain.conn.query('SELECT COUNT(*) as count FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [corrCorrectedNeuronId]);
		const [originalNeuronCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [corrOriginalNeuronId]);
		const [correctedNeuronCheck] = await brain.conn.query('SELECT * FROM inferred_neurons WHERE neuron_id = ? AND age = 0', [corrCorrectedNeuronId]);

		console.log(`   Original sources: ${originalSourcesAfter[0].count}, corrected sources: ${correctedSourcesAfter[0].count}`);
		console.log(`   Original neuron still in inferred_neurons: ${originalNeuronCheck.length}, corrected neuron in inferred_neurons: ${correctedNeuronCheck.length}`);

		if (originalSourcesAfter[0].count === 0 && originalNeuronCheck.length === 1 && correctedNeuronCheck.length === 0)
			console.log('   ✅ correctInferredActions works correctly (sources updated, neurons unchanged)\n');
		else
			console.log('   ❌ correctInferredActions failed\n');

		console.log('✅ All tests completed!');

	} finally {
		if (brain.conn) await brain.conn.end();
	}
}

testCorrectionAndExploration().catch(console.error);

