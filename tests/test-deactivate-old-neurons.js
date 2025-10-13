import Brain from '../brain.js';

async function testDeactivateOldNeurons() {
	console.log('Testing deactivateOldNeurons...\n');
	
	const brain = new Brain();
	await brain.init();
	
	try {
		// Clear tables
		await brain.conn.query('TRUNCATE active_connections');
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		await brain.conn.query('DELETE FROM patterns WHERE pattern_neuron_id > 0');
		
		// Test 1: Base level (level 0) - age-based deactivation
		console.log('Test 1: Base level age-based deactivation...');
		
		// Create neurons
		const [result1] = await brain.conn.query('INSERT INTO neurons VALUES (), (), ()');
		const neuron1 = result1.insertId;
		const neuron2 = neuron1 + 1;
		const neuron3 = neuron1 + 2;
		
		// Activate neurons with different ages
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 5), (?, 0, 9), (?, 0, 10)
		`, [neuron1, neuron2, neuron3]);
		
		// Deactivate old neurons (age >= 10)
		await brain.deactivateOldNeurons(0);
		
		// Check remaining neurons
		const [remaining1] = await brain.conn.query('SELECT neuron_id, age FROM active_neurons WHERE level = 0 ORDER BY neuron_id');
		console.log(`Remaining neurons: ${remaining1.length}`);
		console.log(`  Neuron ${neuron1} (age 5): ${remaining1.find(n => n.neuron_id === neuron1) ? '✓ kept' : '✗ removed'}`);
		console.log(`  Neuron ${neuron2} (age 9): ${remaining1.find(n => n.neuron_id === neuron2) ? '✓ kept' : '✗ removed'}`);
		console.log(`  Neuron ${neuron3} (age 10): ${remaining1.find(n => n.neuron_id === neuron3) ? '✗ kept' : '✓ removed'}`);
		console.log(remaining1.length === 2 ? '✓ PASS: Correctly removed age >= 10\n' : '✗ FAIL\n');
		
		// Test 2: Higher level - pattern-based deactivation
		console.log('Test 2: Higher level pattern-based deactivation...');
		
		// Clear and setup
		await brain.conn.query('TRUNCATE active_connections');
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		await brain.conn.query('DELETE FROM patterns WHERE pattern_neuron_id > 0');
		
		// Create base neurons and connections
		const [result2] = await brain.conn.query('INSERT INTO neurons VALUES (), (), (), ()');
		const base1 = result2.insertId;
		const base2 = base1 + 1;
		const pattern1 = base1 + 2;
		const pattern2 = base1 + 3;
		
		// Create connections at level 0 (different distances to avoid duplicate key)
		const [connResult] = await brain.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
			VALUES (?, ?, 0, 5.0), (?, ?, 1, 3.0)
		`, [base1, base2, base1, base2]);
		const conn1 = connResult.insertId;
		const conn2 = conn1 + 1;
		
		// Activate base neurons
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 0), (?, 0, 0)
		`, [base1, base2]);
		
		// Activate connections at level 0
		await brain.conn.query(`
			INSERT INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level)
			VALUES (?, ?, ?, 0), (?, ?, ?, 0)
		`, [conn1, base1, base2, conn2, base1, base2]);
		
		// Create pattern definitions
		await brain.conn.query(`
			INSERT INTO patterns (pattern_neuron_id, connection_id, strength)
			VALUES (?, ?, 1.0), (?, ?, 1.0)
		`, [pattern1, conn1, pattern2, conn2]);
		
		// Activate both patterns at level 1
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 1, 0), (?, 1, 0)
		`, [pattern1, pattern2]);
		
		console.log('Initial state: 2 patterns active at level 1');
		
		// Now remove one connection from active_connections
		await brain.conn.query('DELETE FROM active_connections WHERE connection_id = ?', [conn2]);
		console.log(`Removed connection ${conn2} from active_connections`);
		
		// Deactivate old patterns
		await brain.deactivateOldNeurons(1);
		
		// Check remaining patterns
		const [remaining2] = await brain.conn.query('SELECT neuron_id FROM active_neurons WHERE level = 1 ORDER BY neuron_id');
		console.log(`Remaining patterns: ${remaining2.length}`);
		console.log(`  Pattern ${pattern1} (conn ${conn1} still active): ${remaining2.find(n => n.neuron_id === pattern1) ? '✓ kept' : '✗ removed'}`);
		console.log(`  Pattern ${pattern2} (conn ${conn2} removed): ${remaining2.find(n => n.neuron_id === pattern2) ? '✗ kept' : '✓ removed'}`);
		console.log(remaining2.length === 1 && remaining2[0].neuron_id === pattern1 ? '✓ PASS: Correctly removed pattern with no active connections\n' : '✗ FAIL\n');
		
		// Test 3: Pattern with multiple connections - stays active if ANY connection is active
		console.log('Test 3: Pattern with multiple connections...');
		
		// Clear and setup
		await brain.conn.query('TRUNCATE active_connections');
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		await brain.conn.query('DELETE FROM patterns WHERE pattern_neuron_id > 0');
		
		// Create neurons and connections
		const [result3] = await brain.conn.query('INSERT INTO neurons VALUES (), (), (), ()');
		const n1 = result3.insertId;
		const n2 = n1 + 1;
		const n3 = n1 + 2;
		const patternMulti = n1 + 3;
		
		const [connResult2] = await brain.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
			VALUES (?, ?, 0, 5.0), (?, ?, 0, 3.0), (?, ?, 0, 2.0)
		`, [n1, n2, n2, n3, n1, n3]);
		const c1 = connResult2.insertId;
		const c2 = c1 + 1;
		const c3 = c1 + 2;
		
		// Activate base neurons
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 0), (?, 0, 0), (?, 0, 0)
		`, [n1, n2, n3]);
		
		// Activate all connections
		await brain.conn.query(`
			INSERT INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level)
			VALUES (?, ?, ?, 0), (?, ?, ?, 0), (?, ?, ?, 0)
		`, [c1, n1, n2, c2, n2, n3, c3, n1, n3]);
		
		// Create pattern with all 3 connections
		await brain.conn.query(`
			INSERT INTO patterns (pattern_neuron_id, connection_id, strength)
			VALUES (?, ?, 1.0), (?, ?, 1.0), (?, ?, 1.0)
		`, [patternMulti, c1, patternMulti, c2, patternMulti, c3]);
		
		// Activate pattern
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 1, 0)
		`, [patternMulti]);
		
		console.log('Pattern has 3 connections, all active');
		
		// Remove 2 out of 3 connections
		await brain.conn.query('DELETE FROM active_connections WHERE connection_id IN (?, ?)', [c1, c2]);
		console.log('Removed 2 out of 3 connections');
		
		// Deactivate old patterns
		await brain.deactivateOldNeurons(1);
		
		// Check if pattern is still active
		const [remaining3] = await brain.conn.query('SELECT neuron_id FROM active_neurons WHERE level = 1');
		console.log(`Pattern still active: ${remaining3.length > 0 ? 'YES' : 'NO'}`);
		console.log(remaining3.length === 1 ? '✓ PASS: Pattern stays active with 1 out of 3 connections\n' : '✗ FAIL\n');
		
		// Remove the last connection
		await brain.conn.query('DELETE FROM active_connections WHERE connection_id = ?', [c3]);
		console.log('Removed last connection');
		
		// Deactivate old patterns
		await brain.deactivateOldNeurons(1);
		
		// Check if pattern is now deactivated
		const [remaining4] = await brain.conn.query('SELECT neuron_id FROM active_neurons WHERE level = 1');
		console.log(`Pattern still active: ${remaining4.length > 0 ? 'YES' : 'NO'}`);
		console.log(remaining4.length === 0 ? '✓ PASS: Pattern deactivated when all connections removed\n' : '✗ FAIL\n');
		
		console.log('All deactivateOldNeurons tests completed!');
		
	} catch (error) {
		console.error('Test failed with error:', error);
		process.exit(1);
	} finally {
		await brain.conn.end();
		process.exit(0);
	}
}

testDeactivateOldNeurons();

