import BrainMySQL from '../brain-mysql.js';

async function testGetActiveConnections() {
	console.log('Testing getActiveConnections with active_connections table...\n');

	const brain = new BrainMySQL();
	await brain.init();
	
	try {
		// Clear tables first
		await brain.conn.query('TRUNCATE active_connections');
		await brain.conn.query('TRUNCATE active_neurons');

		// Setup: Create test neurons
		console.log('Setup: Creating test neurons...');
		const [result1] = await brain.conn.query('INSERT INTO neurons VALUES (), (), (), ()');
		const neuron1 = result1.insertId;
		const neuron2 = neuron1 + 1;
		const neuron3 = neuron1 + 2;
		const neuron4 = neuron1 + 3;
		
		// Create connections with different strengths
		// All distance=0 for spatial connections (will be active when neurons are age=0)
		await brain.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
			VALUES
				(?, ?, 0, 5.0),
				(?, ?, 0, 10.0),
				(?, ?, 0, 3.0)
		`, [neuron1, neuron2, neuron2, neuron3, neuron3, neuron4]);
		
		// Activate neurons at level 0 - all with age=0 initially
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 0), (?, 0, 0), (?, 0, 0), (?, 0, 0)
		`, [neuron1, neuron2, neuron3, neuron4]);

		// Use activateConnections to populate active_connections
		await brain.activateConnections(0);

		// Now age neuron1 to simulate temporal connections
		await brain.conn.query('UPDATE active_neurons SET age = 1 WHERE neuron_id = ?', [neuron1]);
		
		// Test 1: Get active connections for level 0
		console.log('Test 1: Get active connections for level 0...');
		const connections = await brain.getActiveConnections(0);
		
		console.log(`Found ${connections.length} active connections`);
		console.log(connections.length === 3 ? '✓ PASS: Found all 3 connections\n' : '✗ FAIL: Expected 3 connections\n');
		
		// Test 2: Verify connection details include strength from connections table
		console.log('Test 2: Verify connection details...');
		const conn1 = connections.find(c => c.from_neuron_id === neuron1 && c.to_neuron_id === neuron2);
		const conn2 = connections.find(c => c.from_neuron_id === neuron2 && c.to_neuron_id === neuron3);
		const conn3 = connections.find(c => c.from_neuron_id === neuron3 && c.to_neuron_id === neuron4);
		
		console.log('Connection 1 (neuron1 -> neuron2):');
		console.log(`  Strength: ${conn1?.strength} (expected: 5.0)`);
		console.log(`  Distance: ${conn1?.distance} (expected: 0)`);
		console.log(conn1?.strength === 5.0 && conn1?.distance === 0 ? '✓ PASS\n' : '✗ FAIL\n');
		
		console.log('Connection 2 (neuron2 -> neuron3):');
		console.log(`  Strength: ${conn2?.strength} (expected: 10.0)`);
		console.log(`  Distance: ${conn2?.distance} (expected: 0)`);
		console.log(conn2?.strength === 10.0 && conn2?.distance === 0 ? '✓ PASS\n' : '✗ FAIL\n');
		
		console.log('Connection 3 (neuron3 -> neuron4):');
		console.log(`  Strength: ${conn3?.strength} (expected: 3.0)`);
		console.log(`  Distance: ${conn3?.distance} (expected: 0)`);
		console.log(conn3?.strength === 3.0 && conn3?.distance === 0 ? '✓ PASS\n' : '✗ FAIL\n');
		
		// Test 3: Verify connections with strength <= 0 are filtered out
		console.log('Test 3: Verify connections with strength <= 0 are filtered...');
		await brain.conn.query('UPDATE connections SET strength = -1.0 WHERE from_neuron_id = ? AND to_neuron_id = ?', [neuron2, neuron3]);
		
		const connectionsAfterUpdate = await brain.getActiveConnections(0);
		console.log(`Found ${connectionsAfterUpdate.length} active connections after marking one as deleted`);
		console.log(connectionsAfterUpdate.length === 2 ? '✓ PASS: Filtered out connection with negative strength\n' : '✗ FAIL: Expected 2 connections\n');
		
		// Test 4: Test with multiple levels
		console.log('Test 4: Test with multiple levels...');
		
		// Create neurons at level 1
		const [result2] = await brain.conn.query('INSERT INTO neurons VALUES (), ()');
		const neuron5 = result2.insertId;
		const neuron6 = result2.insertId + 1;
		
		// Create connection at level 1
		await brain.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
			VALUES (?, ?, 0, 7.0)
		`, [neuron5, neuron6]);
		
		// Activate neurons at level 1
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 1, 1), (?, 1, 0)
		`, [neuron5, neuron6]);
		
		// Populate active connections for level 1
		await brain.activateConnections(1);
		
		// Get connections for level 1
		const level1Connections = await brain.getActiveConnections(1);
		console.log(`Found ${level1Connections.length} active connections for level 1`);
		console.log(level1Connections.length === 1 ? '✓ PASS: Found 1 connection at level 1\n' : '✗ FAIL: Expected 1 connection\n');
		
		// Verify level 0 connections are still separate
		const level0Connections = await brain.getActiveConnections(0);
		console.log(`Found ${level0Connections.length} active connections for level 0`);
		console.log(level0Connections.length === 2 ? '✓ PASS: Level 0 connections unchanged\n' : '✗ FAIL: Expected 2 connections\n');
		
		console.log('All getActiveConnections tests completed!');
		
	} catch (error) {
		console.error('Test failed with error:', error);
		process.exit(1);
	} finally {
		await brain.conn.end();
		process.exit(0);
	}
}

testGetActiveConnections();

