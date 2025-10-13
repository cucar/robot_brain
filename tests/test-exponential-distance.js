import Brain from '../brain.js';

async function testExponentialDistance() {
	console.log('Testing exponential distance rounding in reinforceConnections...\n');
	
	const brain = new Brain();
	await brain.init();
	
	try {
		// Clear tables
		await brain.conn.query('TRUNCATE active_connections');
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		
		// Create test neurons
		console.log('Setup: Creating test neurons...');
		const [result] = await brain.conn.query('INSERT INTO neurons VALUES (), ()');
		const neuron1 = result.insertId;
		const neuron2 = neuron1 + 1;
		
		// Test 1: Ages 1-9 should have exact distances
		console.log('Test 1: Ages 1-9 should have exact distances...');
		for (let age = 1; age <= 9; age++) {
			// Clear and setup
			await brain.conn.query('TRUNCATE active_neurons');
			await brain.conn.query('DELETE FROM connections WHERE from_neuron_id = ? OR to_neuron_id = ?', [neuron1, neuron2]);
			
			// Insert neurons with specific ages
			await brain.conn.query(`
				INSERT INTO active_neurons (neuron_id, level, age)
				VALUES (?, 0, ?), (?, 0, 0)
			`, [neuron1, age, neuron2]);
			
			// Reinforce connections
			await brain.reinforceConnections(0);
			
			// Check distance
			const [rows] = await brain.conn.query(`
				SELECT distance FROM connections 
				WHERE from_neuron_id = ? AND to_neuron_id = ?
			`, [neuron1, neuron2]);
			
			const expectedDistance = age;
			const actualDistance = rows[0]?.distance;
			console.log(`  Age ${age}: distance = ${actualDistance} (expected: ${expectedDistance}) ${actualDistance === expectedDistance ? '✓' : '✗'}`);
		}
		console.log();
		
		// Test 2: Ages 10-99 should round to multiples of 10
		console.log('Test 2: Ages 10-99 should round to multiples of 10...');
		const testCases10 = [
			{ age: 10, expected: 10 },
			{ age: 15, expected: 10 },
			{ age: 19, expected: 10 },
			{ age: 20, expected: 20 },
			{ age: 25, expected: 20 },
			{ age: 50, expected: 50 },
			{ age: 99, expected: 90 }
		];
		
		for (const { age, expected } of testCases10) {
			await brain.conn.query('TRUNCATE active_neurons');
			await brain.conn.query('DELETE FROM connections WHERE from_neuron_id = ? OR to_neuron_id = ?', [neuron1, neuron2]);
			
			await brain.conn.query(`
				INSERT INTO active_neurons (neuron_id, level, age)
				VALUES (?, 0, ?), (?, 0, 0)
			`, [neuron1, age, neuron2]);
			
			await brain.reinforceConnections(0);
			
			const [rows] = await brain.conn.query(`
				SELECT distance FROM connections 
				WHERE from_neuron_id = ? AND to_neuron_id = ?
			`, [neuron1, neuron2]);
			
			const actualDistance = rows[0]?.distance;
			console.log(`  Age ${age}: distance = ${actualDistance} (expected: ${expected}) ${actualDistance === expected ? '✓' : '✗'}`);
		}
		console.log();
		
		// Test 3: Ages 100-255 should round to multiples of 100 (TINYINT max is 255)
		console.log('Test 3: Ages 100-255 should round to multiples of 100...');
		const testCases100 = [
			{ age: 100, expected: 100 },
			{ age: 150, expected: 100 },
			{ age: 199, expected: 100 },
			{ age: 200, expected: 200 },
			{ age: 250, expected: 200 },
			{ age: 255, expected: 200 }
		];
		
		for (const { age, expected } of testCases100) {
			await brain.conn.query('TRUNCATE active_neurons');
			await brain.conn.query('DELETE FROM connections WHERE from_neuron_id = ? OR to_neuron_id = ?', [neuron1, neuron2]);
			
			await brain.conn.query(`
				INSERT INTO active_neurons (neuron_id, level, age)
				VALUES (?, 0, ?), (?, 0, 0)
			`, [neuron1, age, neuron2]);
			
			await brain.reinforceConnections(0);
			
			const [rows] = await brain.conn.query(`
				SELECT distance FROM connections 
				WHERE from_neuron_id = ? AND to_neuron_id = ?
			`, [neuron1, neuron2]);
			
			const actualDistance = rows[0]?.distance;
			console.log(`  Age ${age}: distance = ${actualDistance} (expected: ${expected}) ${actualDistance === expected ? '✓' : '✗'}`);
		}
		console.log();
		
		// Test 4: Cross-level connections should use same exponential rounding
		console.log('Test 4: Cross-level connections should use same exponential rounding...');
		
		// Create neurons at different levels
		const [result2] = await brain.conn.query('INSERT INTO neurons VALUES (), ()');
		const neuron3 = result2.insertId;
		const neuron4 = result2.insertId + 1;
		
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		
		// Level 0 neuron with age 25, Level 1 neuron with age 0
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 25), (?, 1, 0)
		`, [neuron3, neuron4]);
		
		await brain.reinforceConnections(1);
		
		const [crossLevelRows] = await brain.conn.query(`
			SELECT distance FROM connections 
			WHERE from_neuron_id = ? AND to_neuron_id = ?
		`, [neuron3, neuron4]);
		
		const crossLevelDistance = crossLevelRows[0]?.distance;
		console.log(`  Level 0 (age 25) -> Level 1 (age 0): distance = ${crossLevelDistance} (expected: 20) ${crossLevelDistance === 20 ? '✓' : '✗'}`);
		console.log();
		
		// Test 5: Age 0 should always have distance 0 (spatial connections)
		console.log('Test 5: Age 0 should always have distance 0 (spatial connections)...');
		await brain.conn.query('TRUNCATE active_neurons');
		await brain.conn.query('DELETE FROM connections WHERE id > 0');
		
		await brain.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES (?, 0, 0), (?, 0, 0)
		`, [neuron1, neuron2]);
		
		await brain.reinforceConnections(0);
		
		const [spatialRows] = await brain.conn.query(`
			SELECT distance FROM connections 
			WHERE from_neuron_id = ? AND to_neuron_id = ?
		`, [neuron1, neuron2]);
		
		const spatialDistance = spatialRows[0]?.distance;
		console.log(`  Age 0 -> Age 0: distance = ${spatialDistance} (expected: 0) ${spatialDistance === 0 ? '✓ PASS' : '✗ FAIL'}`);
		console.log();
		
		console.log('All exponential distance tests completed!');
		
	} catch (error) {
		console.error('Test failed with error:', error);
		process.exit(1);
	} finally {
		await brain.conn.end();
		process.exit(0);
	}
}

testExponentialDistance();

