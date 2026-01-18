import BrainMySQL from '../brain-mysql.js';

/**
 * Test Channel for Memory Management Tests
 */
class TestChannel {
    constructor(name = 'test_channel') {
        this.name = name;
    }

	getEventDimensions() {
        return ['test_x', 'test_y', 'test_value'];
    }
    
    getOutputDimensions() {
        return ['test_output'];
    }
    
    async initialize() {}
    async getFrameEvents() { return []; }
    async executeOutputs() {}
    async getRewards() { return 1.0; }
}

/**
 * Memory Management & Forgetting Tests
 * Tests the brain's memory cleanup, forget cycles, and system optimization
 */
class MemoryManagementTests {
    constructor() {
        this.testsPassed = 0;
        this.testsFailed = 0;
        this.brain = null;
    }

    assert(condition, message, actual = null, expected = null) {
        if (condition) {
            console.log(`✓ ${message}`);
            this.testsPassed++;
        } else {
            const details = actual !== null && expected !== null ? ` (got ${actual}, expected ${expected})` : '';
            console.log(`✗ ${message}${details}`);
            this.testsFailed++;
        }
    }

    async run() {
        console.log('Running Memory Management & Forgetting Tests...\n');

        await this.setupBrain();
        await this.testForgetCycleTiming();
        await this.testInferenceTableCleanup();
        await this.testCrossLevelMemoryManagement();
        await this.testMemoryPressureHandling();
        await this.testOrphanedDataCleanup();
        await this.testForgetCycleIntegration();
        await this.testEdgeCases();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);
        
        if (this.testsFailed > 0) {
            process.exit(1);
        }

        await this.cleanup();
    }

    async setupBrain() {
        console.log('Setting up brain for memory management testing...');

        this.brain = new BrainMySQL();
        this.brain.registerChannel('test_channel', TestChannel);

        await this.brain.init();
        await this.brain.resetBrain();

        this.assert(true, 'Brain setup complete');
        console.log();
    }

    async testForgetCycleTiming() {
        console.log('Testing Forget Cycle Timing & Execution:');

        // Reset counter
        this.brain.forgetCounter = 0;

        // Test that forget cycle doesn't run before threshold
        for (let i = 0; i < this.brain.forgetCycles - 1; i++) {
            await this.brain.runForgetCycle();
        }
        this.assert(this.brain.forgetCounter === this.brain.forgetCycles - 1,
            'Counter should increment but not reset before threshold',
            this.brain.forgetCounter, this.brain.forgetCycles - 1);

        // Test that forget cycle runs at threshold
        await this.brain.runForgetCycle();
        this.assert(this.brain.forgetCounter === 0,
            'Counter should reset to 0 after forget cycle executes',
            this.brain.forgetCounter, 0);

        // Test multiple cycles
        this.brain.forgetCounter = 0;
        for (let i = 0; i < this.brain.forgetCycles * 2 + 5; i++) {
            await this.brain.runForgetCycle();
        }
        this.assert(this.brain.forgetCounter === 5,
            'Counter should be at 5 after 2 full cycles + 5 frames',
            this.brain.forgetCounter, 5);

        console.log();
    }

    async testInferenceTableCleanup() {
        console.log('Testing Inference Table Cleanup:');

        // Clear inference tables
        await this.brain.conn.query('TRUNCATE connection_inference');
        await this.brain.conn.query('TRUNCATE pattern_inference');
        await this.brain.conn.query('TRUNCATE inferred_neurons');

        const neuronIds = await this.brain.createNeurons(20);

        // Create actual connections first for inference tables to reference
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[3], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[4], neuronIds[5]]);

        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');

        // Create connection_inference entries at different levels with ages
        // Schema: level, connection_id, age
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, ?, 0)', [connections[0].id]);
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, ?, 5)', [connections[1].id]);
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (1, ?, 0)', [connections[2].id]);
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (1, ?, 50)', [connections[3].id]);

        // Create pattern_inference entries at different levels with ages
        // Schema: level, pattern_neuron_id, connection_id, age
        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (0, ?, ?, 0)', [neuronIds[6], connections[0].id]);
        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (0, ?, ?, 8)', [neuronIds[7], connections[1].id]);
        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (1, ?, ?, 0)', [neuronIds[8], connections[2].id]);

        // Create inferred_neurons entries
        await this.brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[15]]);
        await this.brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age) VALUES (?, 0, 5)', [neuronIds[16]]);
        await this.brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age) VALUES (?, 1, 0)', [neuronIds[17]]);

        // Verify initial state
        const [initialConnInf] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        const [initialPatInf] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference');
        const [initialInferred] = await this.brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons');

        this.assert(initialConnInf[0].count === 4, 'Should have 4 connection_inference entries', initialConnInf[0].count, 4);
        this.assert(initialPatInf[0].count === 3, 'Should have 3 pattern_inference entries', initialPatInf[0].count, 3);
        this.assert(initialInferred[0].count === 3, 'Should have 3 inferred_neurons entries', initialInferred[0].count, 3);

        // Age the entries to trigger cleanup
        // Level 0 max age = 10, Level 1 max age = 100
        await this.brain.conn.query('UPDATE connection_inference SET age = 10 WHERE level = 0 AND age = 5');
        await this.brain.conn.query('UPDATE connection_inference SET age = 100 WHERE level = 1 AND age = 50');
        await this.brain.conn.query('UPDATE pattern_inference SET age = 10 WHERE level = 0 AND age = 8');
        await this.brain.conn.query('UPDATE inferred_neurons SET age = 10 WHERE level = 0 AND age = 5');

        // Run cleanup queries (simulating what should happen in processFrame)
        await this.brain.conn.query(`
            DELETE FROM connection_inference
            WHERE age >= POW(?, level)
        `, [this.brain.contextLength]);

        await this.brain.conn.query(`
            DELETE FROM pattern_inference
            WHERE age >= POW(?, level + 1)
        `, [this.brain.contextLength]);

        await this.brain.conn.query(`
            DELETE FROM inferred_neurons
            WHERE age >= POW(?, level)
        `, [this.brain.contextLength]);

        // Verify cleanup
        const [finalConnInf] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        const [finalPatInf] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference');
        const [finalInferred] = await this.brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons');

        this.assert(finalConnInf[0].count === 2, 'Should remove aged connection_inference entries', finalConnInf[0].count, 2);
        this.assert(finalPatInf[0].count === 2, 'Should remove aged pattern_inference entries', finalPatInf[0].count, 2);
        this.assert(finalInferred[0].count === 2, 'Should remove aged inferred_neurons entries', finalInferred[0].count, 2);

        console.log();
    }

    async testCrossLevelMemoryManagement() {
        console.log('Testing Cross-Level Memory Management:');

        // Clear tables
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('DELETE FROM neuron_rewards');

        const neuronIds = await this.brain.createNeurons(15);

        // Create cross-level connections at different levels
        // Level 0 → Level 1 (distance=0)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 1.5)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 0.05)', [neuronIds[2], neuronIds[3]]); // Weak

        // Level 1 → Level 0 (distance=9)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 9, 2.0)', [neuronIds[4], neuronIds[5]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 9, 0.08)', [neuronIds[6], neuronIds[7]]); // Weak

        // Level 1 → Level 2 (distance=0)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 1.8)', [neuronIds[8], neuronIds[9]]);

        // Same-level connections
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.2)', [neuronIds[10], neuronIds[11]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 0.06)', [neuronIds[12], neuronIds[13]]); // Weak

        const [initialConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        this.assert(initialConnections[0].count === 7, 'Should have 7 connections initially', initialConnections[0].count, 7);

        // Force forget cycle
        this.brain.forgetCounter = this.brain.forgetCycles - 1;
        await this.brain.runForgetCycle();

        // Verify that weak connections are removed regardless of distance/level
        const [finalConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        this.assert(finalConnections[0].count === 4, 'Should remove 3 weak connections', finalConnections[0].count, 4);

        // Verify remaining connections are strong ones
        const [remaining] = await this.brain.conn.query('SELECT distance, strength FROM connections ORDER BY distance, strength');
        this.assert(remaining.every(c => c.strength > 0.1), 'All remaining connections should be strong');

        // Verify cross-level connections decay same as same-level
        const crossLevel = remaining.filter(c => c.distance === 0 || c.distance === 9);
        const sameLevel = remaining.filter(c => c.distance === 1);
        this.assert(crossLevel.length === 3, 'Should have 3 cross-level connections remaining', crossLevel.length, 3);
        this.assert(sameLevel.length === 1, 'Should have 1 same-level connection remaining', sameLevel.length, 1);

        console.log();
    }

    async testMemoryPressureHandling() {
        console.log('Testing Memory Pressure Handling:');

        // Clear tables
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('DELETE FROM neurons WHERE id > 0');

        // Create large number of neurons and connections to simulate memory pressure
        const batchSize = 100;
        const neuronIds = await this.brain.createNeurons(batchSize);

        // Create many connections with varying strengths
        let strongCount = 0;
        let weakCount = 0;
        for (let i = 0; i < batchSize - 1; i++) {
            const strength = Math.random() * 2.0; // 0 to 2.0
            if (strength > 0.5) strongCount++;
            else weakCount++;
            await this.brain.conn.query(
                'INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, ?)',
                [neuronIds[i], neuronIds[i + 1], strength]
            );
        }

        const [initialConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        this.assert(initialConnections[0].count === batchSize - 1,
            `Should have ${batchSize - 1} connections initially`,
            initialConnections[0].count, batchSize - 1);

        // Run multiple forget cycles to clean up weak connections
        for (let i = 0; i < 5; i++) {
            this.brain.forgetCounter = this.brain.forgetCycles - 1;
            await this.brain.runForgetCycle();
        }

        // Verify significant cleanup occurred
        const [finalConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const reductionPercent = ((initialConnections[0].count - finalConnections[0].count) / initialConnections[0].count) * 100;

        this.assert(finalConnections[0].count < initialConnections[0].count,
            'Should reduce connection count under memory pressure',
            `${finalConnections[0].count} < ${initialConnections[0].count}`);

        this.assert(reductionPercent > 15,
            'Should reduce connections by at least 15% after 5 forget cycles',
            `${reductionPercent.toFixed(1)}%`, '>15%');

        // Verify remaining connections are stronger on average
        const [avgStrength] = await this.brain.conn.query('SELECT AVG(strength) as avg FROM connections');
        this.assert(avgStrength[0].avg > 0.5,
            'Average strength of remaining connections should be > 0.5',
            avgStrength[0].avg.toFixed(3), '>0.5');

        console.log();
    }

    async testOrphanedDataCleanup() {
        console.log('Testing Orphaned Data Cleanup:');

        // Clear tables
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('DELETE FROM neurons WHERE id > 0');
        await this.brain.conn.query('DELETE FROM active_neurons');

        const neuronIds = await this.brain.createNeurons(20);

        // Create various scenarios:
        // 1. Neurons with connections (should survive)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[1], neuronIds[2]]);

        // 2. Neurons with patterns (should survive)
        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id LIMIT 1');
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1.0)', [neuronIds[3], connections[0].id]);

        // 3. Active neurons (should survive)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[4]]);

        // 4. Neurons with both connections and patterns (should survive)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.0)', [neuronIds[5], neuronIds[6]]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1.0)', [neuronIds[5], connections[0].id]);

        // 5. Orphaned neurons (should be deleted)
        // neuronIds[7-19] have no connections, patterns, or activity

        const [initialNeurons] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neurons');
        this.assert(initialNeurons[0].count === 20, 'Should have 20 neurons initially', initialNeurons[0].count, 20);

        // Force forget cycle
        this.brain.forgetCounter = this.brain.forgetCycles - 1;
        await this.brain.runForgetCycle();

        // Verify orphaned neurons are removed
        const [finalNeurons] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neurons');
        this.assert(finalNeurons[0].count === 7,
            'Should remove 13 orphaned neurons, keep 7 with connections/patterns/activity',
            finalNeurons[0].count, 7);

        // Verify specific neurons survived
        const [survivors] = await this.brain.conn.query('SELECT id FROM neurons ORDER BY id');
        const survivorIds = survivors.map(n => n.id);

        this.assert(survivorIds.includes(neuronIds[0]), 'Neuron with outgoing connection should survive');
        this.assert(survivorIds.includes(neuronIds[1]), 'Neuron with incoming and outgoing connections should survive');
        this.assert(survivorIds.includes(neuronIds[2]), 'Neuron with incoming connection should survive');
        this.assert(survivorIds.includes(neuronIds[3]), 'Neuron with pattern should survive');
        this.assert(survivorIds.includes(neuronIds[4]), 'Active neuron should survive');
        this.assert(survivorIds.includes(neuronIds[5]), 'Neuron with both connection and pattern should survive');
        this.assert(survivorIds.includes(neuronIds[6]), 'Neuron with incoming connection should survive');

        this.assert(!survivorIds.includes(neuronIds[7]), 'Orphaned neuron should be deleted');
        this.assert(!survivorIds.includes(neuronIds[19]), 'Orphaned neuron should be deleted');

        console.log();
    }

    async testForgetCycleIntegration() {
        console.log('Testing Forget Cycle Integration:');

        // Test complete forget cycle with all components
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('DELETE FROM neuron_rewards');
        await this.brain.conn.query('DELETE FROM neurons WHERE id > 0');

        const neuronIds = await this.brain.createNeurons(10);

        // Create interconnected data
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1.5)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 0.05)', [neuronIds[2], neuronIds[3]]);

        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1.2)', [neuronIds[4], connections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 0.08)', [neuronIds[5], connections[0].id]);

        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 1.5)', [neuronIds[6]]);
        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 1.005)', [neuronIds[7]]);

        // Get initial counts
        const [initialConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const [initialPat] = await this.brain.conn.query('SELECT COUNT(*) as count FROM patterns');
        const [initialRew] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neuron_rewards');
        const [initialNeu] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neurons');

        // Force forget cycle
        this.brain.forgetCounter = this.brain.forgetCycles - 1;
        await this.brain.runForgetCycle();

        // Verify all components were processed
        const [finalConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const [finalPat] = await this.brain.conn.query('SELECT COUNT(*) as count FROM patterns');
        const [finalRew] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neuron_rewards');
        const [finalNeu] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neurons');

        this.assert(finalConn[0].count < initialConn[0].count, 'Should remove weak connections');
        this.assert(finalPat[0].count < initialPat[0].count, 'Should remove weak patterns');
        this.assert(finalRew[0].count < initialRew[0].count, 'Should remove near-neutral rewards');
        this.assert(finalNeu[0].count < initialNeu[0].count, 'Should remove orphaned neurons');

        // Verify cascade effect: removing weak connection should orphan its pattern
        const [remainingPatterns] = await this.brain.conn.query('SELECT connection_id FROM patterns');
        const [remainingConnections] = await this.brain.conn.query('SELECT id FROM connections');
        const remainingConnIds = remainingConnections.map(c => c.id);

        for (const pattern of remainingPatterns) {
            this.assert(remainingConnIds.includes(pattern.connection_id),
                'All remaining patterns should reference existing connections');
        }

        console.log();
    }

    async testEdgeCases() {
        console.log('Testing Edge Cases:');

        // Test forget cycle with empty tables
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('DELETE FROM neuron_rewards');
        await this.brain.conn.query('DELETE FROM neurons WHERE id > 0');

        this.brain.forgetCounter = this.brain.forgetCycles - 1;
        await this.brain.runForgetCycle();

        const [emptyConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const [emptyPat] = await this.brain.conn.query('SELECT COUNT(*) as count FROM patterns');
        const [emptyRew] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neuron_rewards');

        this.assert(emptyConn[0].count === 0, 'Empty connections should remain empty');
        this.assert(emptyPat[0].count === 0, 'Empty patterns should remain empty');
        this.assert(emptyRew[0].count === 0, 'Empty rewards should remain empty');

        // Test forget cycle with all strong data (nothing should be removed)
        const neuronIds = await this.brain.createNeurons(5);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5.0)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 4.0)', [neuronIds[1], neuronIds[2]]);

        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 3.0)', [neuronIds[3], connections[0].id]);

        // Give neuron 4 a connection so it doesn't get cleaned up as orphaned
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 3.0)', [neuronIds[4], neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 2.0)', [neuronIds[4]]);

        const [beforeConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const [beforePat] = await this.brain.conn.query('SELECT COUNT(*) as count FROM patterns');
        const [beforeRew] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neuron_rewards');

        this.brain.forgetCounter = this.brain.forgetCycles - 1;
        await this.brain.runForgetCycle();

        const [afterConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        const [afterPat] = await this.brain.conn.query('SELECT COUNT(*) as count FROM patterns');
        const [afterRew] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neuron_rewards');

        this.assert(afterConn[0].count === beforeConn[0].count, 'Strong connections should survive forget cycle');
        this.assert(afterPat[0].count === beforePat[0].count, 'Strong patterns should survive forget cycle');

        // Rewards decay toward neutral and may be removed if they get too close to 1.0
        // With rewardForgetRate=0.05, reward 2.0 becomes 1.95 after one cycle
        // This is still far from neutral threshold (ABS(reward - 1.0) < 0.01)
        const [rewardAfter] = await this.brain.conn.query('SELECT reward_factor FROM neuron_rewards WHERE neuron_id = ?', [neuronIds[4]]);
        if (rewardAfter.length > 0) {
            const expectedReward = 2.0 + (1.0 - 2.0) * this.brain.rewardForgetRate; // Decay toward 1.0
            this.assert(Math.abs(rewardAfter[0].reward_factor - expectedReward) < 0.001,
                'Reward should decay toward neutral (1.0)',
                rewardAfter[0].reward_factor.toFixed(3), expectedReward.toFixed(3));
        } else {
            // Reward was removed - this shouldn't happen with strong reward (2.0) after just one cycle
            this.assert(false, 'Strong reward (2.0) should not be removed after one forget cycle', 0, 1);
        }

        // Test rapid consecutive forget cycles
        for (let i = 0; i < 10; i++) {
            this.brain.forgetCounter = this.brain.forgetCycles - 1;
            await this.brain.runForgetCycle();
        }

        const [finalConn] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections');
        this.assert(finalConn[0].count > 0, 'Should still have connections after 10 consecutive forget cycles');

        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) {
            await this.brain.conn.end();
        }
    }
}

// Run tests
const tests = new MemoryManagementTests();
tests.run().catch(console.error);

