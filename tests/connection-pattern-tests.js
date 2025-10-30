import Brain from '../brain.js';

/**
 * Test Channel for Connection & Pattern Learning Tests
 */
class TestChannel {
    getInputDimensions() {
        return ['test_x', 'test_y', 'test_value'];
    }
    
    getOutputDimensions() {
        return ['test_output'];
    }
    
    async initialize() {}
    async getFrameInputs() { return []; }
    async executeOutputs() {}
    async getFeedback() { return 1.0; }
}

/**
 * Connection & Pattern Learning Tests
 * Tests the brain's ability to learn relationships and build knowledge hierarchies
 */
class ConnectionPatternTests {
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
        console.log('Running Connection & Pattern Learning Tests...\n');

        await this.setupBrain();
        await this.testActiveConnections();
        await this.testPeakDetection();
        await this.testDistanceWeighting();
        await this.testCrossLevelConnections();
        await this.testPatternActivation();
        await this.testHierarchicalLearning();
        await this.testLevel2PatternCreation();
        await this.testPatternReinforcement();
        await this.testConnectionStrengthUpdates();
        await this.testPatternMerging();
        await this.testPatternNonMerging();
        await this.testEdgeCases();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);
        
        if (this.testsFailed > 0) {
            process.exit(1);
        }

        await this.cleanup();
    }

    async setupBrain() {
        console.log('Setting up brain for testing...');
        
        this.brain = new Brain();
        this.brain.registerChannel('test_channel', TestChannel);
        await this.brain.init();
        await this.brain.resetBrain(); // Clean slate
        
        // Re-initialize after reset
        await this.brain.init();
        
        console.log('✓ Brain setup complete\n');
        this.testsPassed++;
    }

    async testActiveConnections() {
        console.log('Testing Active Connections:');

        // Clear and set up test data
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        
        const neuronIds = await this.brain.bulkInsertNeurons(4);
        
        // Create active neurons at different ages
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 2)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[3]]);
        
        // Create connections with proper distances
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 5)', [neuronIds[0], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 3)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 2)', [neuronIds[2], neuronIds[3]]);
        
        // Test getActiveConnections
        const connections = await this.brain.getActiveConnections(0);
        
        this.assert(connections.length === 3, 'Should return 3 active connections', connections.length, 3);
        
        // Verify connection details
        const conn1 = connections.find(c => c.from_neuron_id === neuronIds[0]);
        const conn2 = connections.find(c => c.from_neuron_id === neuronIds[1]);
        const conn3 = connections.find(c => c.from_neuron_id === neuronIds[2]);
        
        this.assert(conn1 && conn1.distance === 2, 'First connection should have distance 2');
        this.assert(conn1 && conn1.strength === 5, 'First connection should have strength 5');
        this.assert(conn2 && conn2.distance === 1, 'Second connection should have distance 1');
        this.assert(conn3 && conn3.distance === 0, 'Third connection should have distance 0');
        
        console.log();
    }

    async testPeakDetection() {
        console.log('Testing Peak Detection:');

        // Create test connections with different strengths and distances
        const testConnections = [
            { id: 1, from_neuron_id: 1, to_neuron_id: 2, distance: 0, strength: 5 },
            { id: 2, from_neuron_id: 2, to_neuron_id: 3, distance: 0, strength: 3 },
            { id: 3, from_neuron_id: 3, to_neuron_id: 4, distance: 0, strength: 1 },
            { id: 4, from_neuron_id: 1, to_neuron_id: 4, distance: 0, strength: 2 },
            { id: 5, from_neuron_id: 2, to_neuron_id: 4, distance: 0, strength: 4 }
        ];

        // Test getObservedPatterns method
        const peakConnections = this.brain.getObservedPatterns(testConnections);

        this.assert(peakConnections instanceof Map, 'getObservedPatterns should return a Map');
        this.assert(peakConnections.size > 0, 'Should detect at least one peak', peakConnections.size, '>0');

        // Verify peak detection logic - neuron 2 should be the peak (strength 12 > neighborhood avg 6.0)
        const peakNeuronIds = Array.from(peakConnections.keys());
        this.assert(peakNeuronIds.includes(2), 'Should detect neuron 2 as peak (highest strength vs neighborhood)');

        console.log();
    }

    async testDistanceWeighting() {
        console.log('Testing Distance Weighting in Peak Detection:');

        // Create test connections with varying distances to test weighting
        // baseNeuronMaxAge = 10, so weight = (10 - distance) / 10
        // distance=0 → weight=1.0, distance=5 → weight=0.5, distance=9 → weight=0.1
        const testConnections = [
            // Neuron 2 receives connections at different distances
            { id: 1, from_neuron_id: 1, to_neuron_id: 2, distance: 0, strength: 10 }, // weighted: 10 * 1.0 = 10.0
            { id: 2, from_neuron_id: 3, to_neuron_id: 2, distance: 5, strength: 10 }, // weighted: 10 * 0.5 = 5.0
            { id: 3, from_neuron_id: 4, to_neuron_id: 2, distance: 9, strength: 10 }, // weighted: 10 * 0.1 = 1.0
            // Neuron 5 receives same raw strength but all at distance=0
            { id: 4, from_neuron_id: 6, to_neuron_id: 5, distance: 0, strength: 10 }, // weighted: 10 * 1.0 = 10.0
            { id: 5, from_neuron_id: 7, to_neuron_id: 5, distance: 0, strength: 5 },  // weighted: 5 * 1.0 = 5.0
            { id: 6, from_neuron_id: 8, to_neuron_id: 5, distance: 0, strength: 1 }   // weighted: 1 * 1.0 = 1.0
        ];

        // Calculate neuron strengths with distance weighting
        const neuronStrengths = this.brain.getNeuronStrengths(testConnections);

        // Neuron 2: incoming = 10.0 + 5.0 + 1.0 = 16.0
        // Neuron 5: incoming = 10.0 + 5.0 + 1.0 = 16.0
        // Both should have same weighted strength despite different distance distributions
        this.assert(neuronStrengths.get(2) === 16.0, 'Neuron 2 weighted strength should be 16.0', neuronStrengths.get(2), 16.0);
        this.assert(neuronStrengths.get(5) === 16.0, 'Neuron 5 weighted strength should be 16.0', neuronStrengths.get(5), 16.0);

        // Test that distant connections have less impact
        const closeConnection = [
            { id: 1, from_neuron_id: 1, to_neuron_id: 2, distance: 0, strength: 10 }
        ];
        const distantConnection = [
            { id: 2, from_neuron_id: 1, to_neuron_id: 2, distance: 9, strength: 10 }
        ];

        const closeStrength = this.brain.getNeuronStrengths(closeConnection);
        const distantStrength = this.brain.getNeuronStrengths(distantConnection);

        // Close connection: 10 * 1.0 = 10.0
        // Distant connection: 10 * 0.1 = 1.0
        this.assert(closeStrength.get(2) === 10.0, 'Close connection (distance=0) should have full weight', closeStrength.get(2), 10.0);
        this.assert(distantStrength.get(2) === 1.0, 'Distant connection (distance=9) should have 0.1 weight', distantStrength.get(2), 1.0);
        this.assert(closeStrength.get(2) > distantStrength.get(2), 'Close connections should have more impact than distant ones');

        // Test peak detection with distance weighting
        // Neuron 10 has high raw strength but distant connections
        // Neuron 11 has lower raw strength but close connections
        const peakTestConnections = [
            { id: 1, from_neuron_id: 1, to_neuron_id: 10, distance: 9, strength: 100 }, // weighted: 100 * 0.1 = 10.0
            { id: 2, from_neuron_id: 2, to_neuron_id: 11, distance: 0, strength: 15 },  // weighted: 15 * 1.0 = 15.0
            { id: 3, from_neuron_id: 3, to_neuron_id: 11, distance: 0, strength: 15 }   // weighted: 15 * 1.0 = 15.0
        ];

        const peakStrengths = this.brain.getNeuronStrengths(peakTestConnections);

        // Neuron 10: 10.0 (distant, heavily discounted)
        // Neuron 11: 30.0 (close, full weight)
        this.assert(peakStrengths.get(10) === 10.0, 'Neuron 10 should have discounted strength', peakStrengths.get(10), 10.0);
        this.assert(peakStrengths.get(11) === 30.0, 'Neuron 11 should have full strength', peakStrengths.get(11), 30.0);
        this.assert(peakStrengths.get(11) > peakStrengths.get(10), 'Recent connections should dominate peak detection');

        console.log();
    }

    async testCrossLevelConnections() {
        console.log('Testing Cross-Level Connections:');

        // Clear and set up cross-level connection scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');

        const neuronIds = await this.brain.bulkInsertNeurons(10);

        // Create neurons at different levels
        // Level 0: base neurons
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 2)', [neuronIds[2]]);

        // Level 1: pattern neurons
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 0)', [neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 1)', [neuronIds[4]]);

        // Level 2: higher-level pattern
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 2, 0)', [neuronIds[5]]);

        // Create cross-level connections with proper distance encoding:
        // 1. Lower → Higher (level 0 → level 1): distance = 0
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 10)', [neuronIds[0], neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 8)', [neuronIds[1], neuronIds[3]]);

        // 2. Higher → Lower (level 1 → level 0): distance = baseNeuronMaxAge - 1 = 9
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 9, 5)', [neuronIds[3], neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 9, 6)', [neuronIds[4], neuronIds[0]]);

        // 3. Same-level connections for comparison (level 0 → level 0): distance = FLOOR(age/1)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[1], neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 4)', [neuronIds[2], neuronIds[0]]);

        // 4. Multi-level cross connections (level 0 → level 2)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 12)', [neuronIds[0], neuronIds[5]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 9)', [neuronIds[1], neuronIds[5]]);

        // Test getActiveConnections for level 0 (should include cross-level connections)
        const level0Connections = await this.brain.getActiveConnections(0);

        // Should get connections TO level 0 age=0 neurons FROM all active neurons
        // Expected: higher→lower (distance=9), same-level (distance=1,2)
        const higherToLower = level0Connections.filter(c => c.distance === 9);
        const sameLevelTemporal = level0Connections.filter(c => c.distance === 1 || c.distance === 2);

        this.assert(higherToLower.length === 2, 'Should have 2 higher→lower connections (distance=9)', higherToLower.length, 2);
        this.assert(sameLevelTemporal.length === 2, 'Should have 2 same-level temporal connections', sameLevelTemporal.length, 2);
        this.assert(level0Connections.length === 4, 'Should have 4 total connections to level 0', level0Connections.length, 4);

        // Test getActiveConnections for level 1 (should include lower→higher connections)
        const level1Connections = await this.brain.getActiveConnections(1);

        // Should get connections TO level 1 age=0 neurons FROM all active neurons
        // Expected: lower→higher (distance=0)
        const lowerToHigher = level1Connections.filter(c => c.distance === 0);

        this.assert(lowerToHigher.length === 2, 'Should have 2 lower→higher connections (distance=0)', lowerToHigher.length, 2);
        this.assert(level1Connections.every(c => c.to_neuron_id === neuronIds[3]), 'All connections should target level 1 age=0 neuron');

        // Test distance weighting with cross-level connections
        const allConnections = [
            { id: 1, from_neuron_id: neuronIds[0], to_neuron_id: neuronIds[3], distance: 0, strength: 10 }, // lower→higher: weight=1.0
            { id: 2, from_neuron_id: neuronIds[3], to_neuron_id: neuronIds[0], distance: 9, strength: 10 }, // higher→lower: weight=0.1
            { id: 3, from_neuron_id: neuronIds[1], to_neuron_id: neuronIds[0], distance: 1, strength: 10 }  // same-level: weight=0.9
        ];

        const strengths = this.brain.getNeuronStrengths(allConnections);

        // Neuron 0: incoming = 10*0.1 + 10*0.9 = 1.0 + 9.0 = 10.0, outgoing = 10*1.0 = 10.0, total = 20.0
        // Neuron 3: incoming = 10*1.0 = 10.0, outgoing = 10*0.1 = 1.0, total = 11.0
        // Neuron 1: outgoing = 10*0.9 = 9.0, total = 9.0
        this.assert(strengths.get(neuronIds[0]) === 20.0, 'Neuron 0 should have weighted strength 20.0', strengths.get(neuronIds[0]), 20.0);
        this.assert(strengths.get(neuronIds[3]) === 11.0, 'Neuron 3 should have weighted strength 11.0', strengths.get(neuronIds[3]), 11.0);
        this.assert(strengths.get(neuronIds[1]) === 9.0, 'Neuron 1 should have weighted strength 9.0', strengths.get(neuronIds[1]), 9.0);

        // Test that lower→higher connections have full weight (distance=0)
        const lowerToHigherWeight = (this.brain.baseNeuronMaxAge - 0) / this.brain.baseNeuronMaxAge;
        this.assert(lowerToHigherWeight === 1.0, 'Lower→higher connections should have full weight (distance=0)', lowerToHigherWeight, 1.0);

        // Test that higher→lower connections have minimal weight (distance=9)
        const higherToLowerWeight = (this.brain.baseNeuronMaxAge - 9) / this.brain.baseNeuronMaxAge;
        this.assert(higherToLowerWeight === 0.1, 'Higher→lower connections should have 0.1 weight (distance=9)', higherToLowerWeight, 0.1);

        // Verify that lower→higher connections dominate in peak detection
        this.assert(lowerToHigherWeight > higherToLowerWeight, 'Lower→higher should have more weight than higher→lower');
        this.assert(lowerToHigherWeight / higherToLowerWeight === 10, 'Lower→higher should be 10x stronger than higher→lower');

        // Test peak detection with mixed cross-level and same-level connections
        const mixedConnections = [
            // Neuron 0 receives from multiple levels
            { id: 1, from_neuron_id: neuronIds[3], to_neuron_id: neuronIds[0], distance: 9, strength: 100 }, // higher→lower: 100*0.1=10
            { id: 2, from_neuron_id: neuronIds[4], to_neuron_id: neuronIds[0], distance: 9, strength: 100 }, // higher→lower: 100*0.1=10
            { id: 3, from_neuron_id: neuronIds[1], to_neuron_id: neuronIds[0], distance: 1, strength: 20 },  // same-level: 20*0.9=18
            { id: 4, from_neuron_id: neuronIds[2], to_neuron_id: neuronIds[0], distance: 2, strength: 20 },  // same-level: 20*0.8=16
            // Neuron 3 receives from lower level
            { id: 5, from_neuron_id: neuronIds[0], to_neuron_id: neuronIds[3], distance: 0, strength: 15 },  // lower→higher: 15*1.0=15
            { id: 6, from_neuron_id: neuronIds[1], to_neuron_id: neuronIds[3], distance: 0, strength: 15 }   // lower→higher: 15*1.0=15
        ];

        const mixedStrengths = this.brain.getNeuronStrengths(mixedConnections);

        // Neuron 0: incoming = 10 + 10 + 18 + 16 = 54, outgoing = 15*1.0 = 15, total = 69
        // Neuron 3: incoming = 15 + 15 = 30, outgoing = 100*0.1 = 10, total = 40
        this.assert(mixedStrengths.get(neuronIds[0]) === 69, 'Neuron 0 should aggregate cross-level and same-level strengths', mixedStrengths.get(neuronIds[0]), 69);
        this.assert(mixedStrengths.get(neuronIds[3]) === 40, 'Neuron 3 should aggregate lower→higher strengths', mixedStrengths.get(neuronIds[3]), 40);

        // Despite higher raw strength from higher levels, same-level connections should contribute more due to weighting
        const higherLevelContribution = 10 + 10; // 20
        const sameLevelContribution = 18 + 16;   // 34
        this.assert(sameLevelContribution > higherLevelContribution, 'Same-level connections should contribute more than distant higher-level ones');

        console.log();
    }

    async testPatternActivation() {
        console.log('Testing Pattern Activation:');

        // Clear and set up complex pattern scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(6);

        // Create a clear pattern: 2 older neurons connecting to 1 new neuron (convergent pattern)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[2]]);

        // Create connections with proper distances (age=1 -> distance=1)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[0], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[1], neuronIds[2]]);

        // Get connection IDs for validation
        const [connectionRows] = await this.brain.conn.query('SELECT id, from_neuron_id, to_neuron_id FROM connections ORDER BY id');

        // Test activateLevelPatterns
        const patternsFound = await this.brain.activateLevelPatterns(0);

        this.assert(typeof patternsFound === 'boolean', 'activateLevelPatterns should return boolean');
        this.assert(patternsFound === true, 'Should find patterns with sufficient connections');

        // Validate peak detection - neuron 2 should be the peak (receives connections from 0 and 1)
        // Neuron 2 strength: 5 + 6 = 11, Neighborhood avg: (11 + 5) / 2 = 8, so 11 > 8 = peak
        const [observedPatterns] = await this.brain.conn.query('SELECT peak_neuron_id, connection_id FROM observed_patterns ORDER BY peak_neuron_id, connection_id');

        this.assert(observedPatterns.length === 2, 'Should observe 2 connections in the pattern', observedPatterns.length, 2);
        this.assert(observedPatterns.every(row => row.peak_neuron_id === neuronIds[2]), 'Peak should be the convergent neuron (neuron 2)');

        // Validate that the correct connections were identified
        const observedConnectionIds = observedPatterns.map(row => row.connection_id).sort();
        const expectedConnectionIds = connectionRows.map(row => row.id).sort();
        this.assert(JSON.stringify(observedConnectionIds) === JSON.stringify(expectedConnectionIds), 'Should identify both connections in the pattern');

        // Check pattern neurons were created at level 1
        const [level1Neurons] = await this.brain.conn.query('SELECT neuron_id FROM active_neurons WHERE level = 1');
        this.assert(level1Neurons.length === 1, 'Should create exactly 1 pattern neuron at level 1', level1Neurons.length, 1);

        const patternNeuronId = level1Neurons[0].neuron_id;

        // Validate pattern definitions in patterns table
        const [patternDefs] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns WHERE pattern_neuron_id = ? ORDER BY connection_id', [patternNeuronId]);

        this.assert(patternDefs.length === 2, 'Pattern should reference both connections', patternDefs.length, 2);

        // Pattern strength should be 2 (1 for creation + 1 for reinforcement during activation)
        this.assert(patternDefs.every(row => row.strength === 2), 'Pattern strength should be 2 (creation + reinforcement)', patternDefs[0].strength, 2);
        this.assert(patternDefs.every(row => expectedConnectionIds.includes(row.connection_id)), 'Pattern should reference the correct connections');

        console.log();
    }

    async testHierarchicalLearning() {
        console.log('Testing Hierarchical Learning:');

        // Clear and set up complex hierarchical scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(10);

        // Create base level neurons (level 0) - 3 older neurons, 2 new neurons
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[4]]);

        // Create Level 0 connections - two convergent patterns
        // Pattern 1: neurons 0,1 -> neuron 3 (convergent pattern)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[0], neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[1], neuronIds[3]]);

        // Pattern 2: neurons 1,2 -> neuron 4 (overlapping convergent pattern)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[1], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[2], neuronIds[4]]);

        // Process level 0 patterns first to validate level 0 behavior
        await this.brain.activateLevelPatterns(0);

        // === LEVEL 0 VALIDATION ===
        const [level0Neurons] = await this.brain.conn.query('SELECT neuron_id, age FROM active_neurons WHERE level = 0 ORDER BY neuron_id');
        this.assert(level0Neurons.length === 5, 'Should maintain all 5 base level neurons', level0Neurons.length, 5);

        // Validate Level 0 observed patterns - should detect 2 peaks (neurons 3 and 4)
        // Only the initial connections exist (reinforceConnections not called yet):
        // - 0→3, 1→3 (both distance=1)
        // - 1→4, 2→4 (both distance=1)
        // Peak 3 neighbors: {0,1} → connections: 0→3, 1→3 = 2 connections
        // Peak 4 neighbors: {1,2} → connections: 1→4, 2→4 = 2 connections
        const [level0ObservedPatterns] = await this.brain.conn.query('SELECT peak_neuron_id, COUNT(*) as connection_count FROM observed_patterns GROUP BY peak_neuron_id ORDER BY peak_neuron_id');
        this.assert(level0ObservedPatterns.length === 2, 'Should detect 2 convergent patterns at level 0', level0ObservedPatterns.length, 2);
        this.assert(level0ObservedPatterns[0].peak_neuron_id === neuronIds[3], 'First peak should be neuron 3');
        this.assert(level0ObservedPatterns[1].peak_neuron_id === neuronIds[4], 'Second peak should be neuron 4');
        this.assert(level0ObservedPatterns[0].connection_count === 2, 'Pattern 1 should have 2 connections', level0ObservedPatterns[0].connection_count, 2);
        this.assert(level0ObservedPatterns[1].connection_count === 2, 'Pattern 2 should have 2 connections', level0ObservedPatterns[1].connection_count, 2);

        // Continue hierarchical activation from level 1 onwards (level 0 already processed)
        // Process remaining levels manually since activatePatternNeurons would try to reprocess level 0
        for (let level = 1; level < this.brain.maxLevels; level++) {
            const hasPatterns = await this.brain.activateLevelPatterns(level);
            if (!hasPatterns) break;
        }

        // === LEVEL 1 VALIDATION ===
        const [level1Neurons] = await this.brain.conn.query('SELECT neuron_id, age FROM active_neurons WHERE level = 1 ORDER BY neuron_id');
        this.assert(level1Neurons.length === 2, 'Should create exactly 2 pattern neurons at level 1', level1Neurons.length, 2);
        this.assert(level1Neurons.every(n => n.age === 0), 'Level 1 neurons should have age 0');

        // Validate Level 1 pattern definitions
        const [level1Patterns] = await this.brain.conn.query(`
            SELECT p.pattern_neuron_id, COUNT(*) as connection_count, AVG(p.strength) as avg_strength
            FROM patterns p
            WHERE p.pattern_neuron_id IN (?, ?)
            GROUP BY p.pattern_neuron_id
            ORDER BY p.pattern_neuron_id
        `, [level1Neurons[0].neuron_id, level1Neurons[1].neuron_id]);

        this.assert(level1Patterns.length === 2, 'Should have pattern definitions for both level 1 neurons');
        this.assert(level1Patterns.every(p => p.connection_count === 2), 'Each pattern should reference 2 connections');
        this.assert(level1Patterns.every(p => p.avg_strength === 2), 'Pattern strengths should be 2 (creation + reinforcement)');

        // === LEVEL 1 CONNECTIONS VALIDATION ===
        // Check if level 1 neurons created connections between themselves
        const [level1Connections] = await this.brain.conn.query(`
            SELECT c.from_neuron_id, c.to_neuron_id, c.distance, c.strength
            FROM connections c
            JOIN active_neurons an1 ON c.from_neuron_id = an1.neuron_id AND an1.level = 1
            JOIN active_neurons an2 ON c.to_neuron_id = an2.neuron_id AND an2.level = 1
        `);

        if (level1Connections.length > 0) {
            console.log(`Found ${level1Connections.length} level 1 connections`);
            this.assert(level1Connections.every(c => c.distance === 0), 'Level 1 connections should have distance 0 (same age)');
            this.assert(level1Connections.every(c => c.strength > 0), 'Level 1 connections should have positive strength');
        }

        // === HIERARCHICAL STRUCTURE VALIDATION ===
        const [levelCounts] = await this.brain.conn.query('SELECT level, COUNT(*) as count FROM active_neurons GROUP BY level ORDER BY level');
        this.assert(levelCounts.length >= 2, 'Should create multiple hierarchical levels', levelCounts.length, '>=2');
        this.assert(levelCounts[0].level === 0 && levelCounts[0].count === 5, 'Level 0 should have 5 neurons');
        this.assert(levelCounts[1].level === 1 && levelCounts[1].count === 2, 'Level 1 should have 2 neurons');

        // Check if level 2 was created (if level 1 patterns were strong enough)
        const level2Count = levelCounts.find(lc => lc.level === 2);
        if (level2Count) {
            console.log(`Level 2 created with ${level2Count.count} neurons - hierarchical learning reached level 2!`);
            this.assert(level2Count.count > 0, 'Level 2 should have at least 1 neuron if created');
        }

        console.log();
    }

    async testLevel2PatternCreation() {
        console.log('Testing Level 2 Pattern Creation:');
        // This test follows the proper brain flow: use activatePatternNeurons() for complete hierarchy
        // rather than calling activateLevelPatterns() directly which violates calling assumptions

        // Clear and set up scenario that will definitely create level 2 patterns
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(12);

        // === LEVEL 0 SETUP ===
        // Create 6 base neurons: 4 older (age=1), 2 new (age=0)
        for (let i = 0; i < 4; i++)
            await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[i]]);
        for (let i = 4; i < 6; i++)
            await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[i]]);

        // Create Level 0 connections - 2 strong convergent patterns
        // Pattern A: neurons 0,1 -> neuron 4 (strong convergence)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[0], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 9)', [neuronIds[1], neuronIds[4]]);

        // Pattern B: neurons 2,3 -> neuron 5 (strong convergence)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[2], neuronIds[5]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[3], neuronIds[5]]);

        // Add more base level neurons and connections to create richer patterns
        for (let i = 6; i < 10; i++) {
            await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[i]]);
        }

        // Create additional connections that reinforce the existing patterns
        // More connections to pattern A (neuron 4) - making it a stronger convergent pattern
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[6], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[7], neuronIds[4]]);

        // More connections to pattern B (neuron 5) - making it a stronger convergent pattern
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[8], neuronIds[5]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 9)', [neuronIds[9], neuronIds[5]]);

        // === FRAME 1: INITIAL HIERARCHICAL ACTIVATION ===
        // Simulate first frame processing - this creates Level 0 and Level 1 patterns
        await this.brain.activatePatternNeurons();

        // Check Level 1 was created
        const [frame1Levels] = await this.brain.conn.query('SELECT level, COUNT(*) as count FROM active_neurons GROUP BY level ORDER BY level');
        console.log('After frame 1:', frame1Levels.map(lc => `Level ${lc.level}: ${lc.count} neurons`).join(', '));

        this.assert(frame1Levels.length >= 2, 'Should create at least 2 levels in frame 1');
        this.assert(frame1Levels.some(lc => lc.level === 1), 'Should create Level 1 patterns in frame 1');

        // Get Level 1 neurons for next frame setup
        const [level1Neurons] = await this.brain.conn.query('SELECT neuron_id FROM active_neurons WHERE level = 1 ORDER BY neuron_id');
        this.assert(level1Neurons.length >= 2, 'Need at least 2 Level 1 neurons for Level 2 patterns');

        // === SIMULATE FRAME TRANSITION ===
        // Age all neurons (this is what happens between frames)
        await this.brain.ageNeurons();

        // === FRAME 2: ADD NEW LEVEL 1 ACTIVITY ===
        // Add a new Level 1 neuron that creates a convergent pattern at Level 1
        const newLevel1NeuronId = neuronIds[10];
        await this.brain.activateNeurons([newLevel1NeuronId], 1);

        // Create strong connections from existing Level 1 neurons to new Level 1 neuron
        // This simulates Level 1 patterns that co-occur and should create Level 2
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 15)', [level1Neurons[0].neuron_id, newLevel1NeuronId]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 16)', [level1Neurons[1].neuron_id, newLevel1NeuronId]);

        // === FRAME 2: HIERARCHICAL ACTIVATION FOR LEVEL 2 ===
        // Process patterns starting from Level 1 (since Level 0 is already processed)
        // This should detect the Level 1 convergent pattern and create Level 2
        const level1Result = await this.brain.activateLevelPatterns(1);
        this.assert(level1Result === true, 'Level 1 should detect convergent pattern and create Level 2');

        // Check final levels
        const [finalLevels] = await this.brain.conn.query('SELECT level, COUNT(*) as count FROM active_neurons GROUP BY level ORDER BY level');
        console.log('After second activation:', finalLevels.map(lc => `Level ${lc.level}: ${lc.count} neurons`).join(', '));

        const maxLevel = Math.max(...finalLevels.map(lc => lc.level));
        const level2Created = maxLevel >= 2;

        // Early return on failure - Level 2 must be created for this test to pass
        if (!level2Created) {
            this.assert(false, 'FAILED: Level 2 should have been created with strong Level 1 convergent patterns');
            return;
        }

        console.log('✅ SUCCESS: Level 2 created through hierarchical learning!');

        // === LEVEL 2 VALIDATION ===
        const [level2Neurons] = await this.brain.conn.query('SELECT neuron_id, age FROM active_neurons WHERE level = 2 ORDER BY neuron_id');
        this.assert(level2Neurons.length >= 1, 'Should create at least 1 pattern neuron at level 2', level2Neurons.length, '>=1');
        // Level 2 neurons were created in frame 1 (age=0), then aged once (age=1), then possibly new ones created (age=0)
        // So we expect a mix of ages, with at least some neurons having age >= 1
        this.assert(level2Neurons.every(n => n.age >= 0), 'Level 2 neurons should have non-negative age');
        this.assert(level2Neurons.some(n => n.age >= 1), 'Some level 2 neurons should have age >= 1 (created in frame 1 and aged)');

        // Validate Level 2 pattern definitions
        const [level2Patterns] = await this.brain.conn.query(`
            SELECT p.pattern_neuron_id, COUNT(*) as connection_count, AVG(p.strength) as avg_strength
            FROM patterns p
            JOIN active_neurons an ON p.pattern_neuron_id = an.neuron_id AND an.level = 2
            GROUP BY p.pattern_neuron_id
        `);

        this.assert(level2Patterns.length >= 1, 'Should have pattern definitions for level 2 neurons');
        this.assert(level2Patterns.every(p => p.connection_count >= 1), 'Level 2 patterns should reference connections');
        this.assert(level2Patterns.every(p => p.avg_strength >= 1), 'Level 2 pattern strengths should be positive');

        // Validate multi-level hierarchy
        // With strong convergent patterns (strengths 6-9), the brain creates patterns up to maxLevels (6)
        // Each level creates strong patterns that trigger the next level
        // Expected: levels 0-6 (7 total levels) due to strong pattern strengths
        this.assert(finalLevels.length >= 3, 'Should have at least 3 hierarchical levels', finalLevels.length, '>=3');
        this.assert(finalLevels.some(lc => lc.level === 2), 'Should include level 2');
        this.assert(finalLevels.length <= 7, 'Should not exceed maxLevels+1 (0-6)', finalLevels.length, '<=7');

        // === FINAL VALIDATION ===
        // Validate the complete hierarchical structure
        this.assert(finalLevels.length >= 2, 'Should have at least 2 hierarchical levels', finalLevels.length, '>=2');
        this.assert(finalLevels[0].level === 0, 'First level should be level 0');
        this.assert(finalLevels[1].level === 1, 'Second level should be level 1');

        // Validate that base level has all neurons (original 6 + additional 4)
        this.assert(finalLevels[0].count === 10, 'Level 0 should have 10 neurons total', finalLevels[0].count, 10);

        // Validate level 1 has pattern neurons (original 2 + new convergent target)
        this.assert(finalLevels[1].count >= 3, 'Level 1 should have at least 3 neurons', finalLevels[1].count, '>=3');

        // === PATTERN QUALITY VALIDATION ===
        // Check that patterns were created with proper relationships across all levels
        const [patternQuality] = await this.brain.conn.query(`
            SELECT
                an.level,
                COUNT(DISTINCT p.pattern_neuron_id) as pattern_count,
                AVG(p.strength) as avg_strength,
                COUNT(p.connection_id) as total_connections
            FROM active_neurons an
            JOIN patterns p ON an.neuron_id = p.pattern_neuron_id
            WHERE an.level >= 1
            GROUP BY an.level
            ORDER BY an.level
        `);

        this.assert(patternQuality.length >= 1, 'Should have pattern definitions at level 1+');
        this.assert(patternQuality[0].pattern_count >= 2, 'Should have multiple patterns at level 1');
        this.assert(patternQuality[0].avg_strength >= 1, 'Pattern strengths should be positive');

        // If Level 2 was created, validate its pattern quality too
        if (level2Created && patternQuality.length >= 2) {
            this.assert(patternQuality[1].level === 2, 'Second pattern level should be level 2');
            this.assert(patternQuality[1].pattern_count >= 1, 'Should have at least 1 pattern at level 2');
            console.log(`✅ Level 2 pattern quality: ${patternQuality[1].pattern_count} patterns with avg strength ${patternQuality[1].avg_strength}`);
        }

        console.log(`Level 1 pattern quality: ${patternQuality[0].pattern_count} patterns with avg strength ${patternQuality[0].avg_strength}`);

        console.log();
    }

    async testPatternReinforcement() {
        console.log('Testing Pattern Reinforcement:');

        // Clear and set up comprehensive pattern reinforcement scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(8);

        // === SETUP MULTIPLE PATTERNS FOR COMPREHENSIVE TESTING ===
        // Create connections for two different patterns
        // Pattern A: neuronIds[0] -> neuronIds[1] -> neuronIds[2] (chain pattern)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 4)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[1], neuronIds[2]]);

        // Pattern B: neuronIds[3] -> neuronIds[4], neuronIds[5] -> neuronIds[4] (convergent pattern)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[3], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[5], neuronIds[4]]);

        // Get connection IDs for pattern definitions
        const [allConnections] = await this.brain.conn.query('SELECT id, from_neuron_id, to_neuron_id FROM connections ORDER BY id');
        this.assert(allConnections.length === 4, 'Should have 4 connections created');

        // Create pattern definitions with initial strength = 1
        // Pattern Neuron A (neuronIds[6]) represents Pattern A
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1)', [neuronIds[6], allConnections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1)', [neuronIds[6], allConnections[1].id]);

        // Pattern Neuron B (neuronIds[7]) represents Pattern B
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 2)', [neuronIds[7], allConnections[2].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 3)', [neuronIds[7], allConnections[3].id]);

        // === VALIDATE INITIAL STATE ===
        const [initialPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY pattern_neuron_id, connection_id');
        this.assert(initialPatterns.length === 4, 'Should have 4 initial pattern-connection relationships');

        // Pattern A should have strength 1 for both connections
        const patternAInitial = initialPatterns.filter(p => p.pattern_neuron_id === neuronIds[6]);
        this.assert(patternAInitial.length === 2, 'Pattern A should have 2 connections');
        this.assert(patternAInitial.every(p => p.strength === 1), 'Pattern A should start with strength 1');

        // Pattern B should have different initial strengths
        const patternBInitial = initialPatterns.filter(p => p.pattern_neuron_id === neuronIds[7]);
        this.assert(patternBInitial.length === 2, 'Pattern B should have 2 connections');
        this.assert(patternBInitial[0].strength === 2, 'Pattern B first connection should start with strength 2');
        this.assert(patternBInitial[1].strength === 3, 'Pattern B second connection should start with strength 3');

        // === SETUP REINFORCEMENT SCENARIO ===
        // Create peakPatterns map: peak_neuron_id -> [pattern_neuron_ids]
        const peakPatterns = new Map();
        peakPatterns.set(neuronIds[1], [neuronIds[6]]); // Peak neuron 1 activates Pattern A
        peakPatterns.set(neuronIds[4], [neuronIds[7]]); // Peak neuron 4 activates Pattern B

        // Create peakConnections map: peak_neuron_id -> [connection_ids]
        const peakConnections = new Map();
        peakConnections.set(neuronIds[1], [allConnections[0].id, allConnections[1].id]); // Pattern A connections
        peakConnections.set(neuronIds[4], [allConnections[2].id, allConnections[3].id]); // Pattern B connections

        // === EXECUTE PATTERN REINFORCEMENT ===
        await this.brain.reinforcePatterns(peakPatterns, peakConnections);

        // === VALIDATE REINFORCEMENT RESULTS ===
        const [reinforcedPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY pattern_neuron_id, connection_id');
        this.assert(reinforcedPatterns.length === 4, 'Should still have 4 pattern-connection relationships');

        // Pattern A should be reinforced from 1 to 2
        const patternAReinforced = reinforcedPatterns.filter(p => p.pattern_neuron_id === neuronIds[6]);
        this.assert(patternAReinforced.length === 2, 'Pattern A should still have 2 connections');
        this.assert(patternAReinforced.every(p => p.strength === 2), 'Pattern A should be reinforced to strength 2');

        // Pattern B should be reinforced from [2,3] to [3,4]
        const patternBReinforced = reinforcedPatterns.filter(p => p.pattern_neuron_id === neuronIds[7]);
        this.assert(patternBReinforced.length === 2, 'Pattern B should still have 2 connections');
        this.assert(patternBReinforced[0].strength === 3, 'Pattern B first connection should be reinforced to 3');
        this.assert(patternBReinforced[1].strength === 4, 'Pattern B second connection should be reinforced to 4');

        // === VALIDATE REINFORCEMENT SPECIFICITY ===
        // Check that only the connections involved in the observed patterns were reinforced
        const patternAConnections = patternAReinforced.map(p => p.connection_id).sort();
        const expectedPatternAConnections = [allConnections[0].id, allConnections[1].id].sort();
        this.assert(JSON.stringify(patternAConnections) === JSON.stringify(expectedPatternAConnections), 'Pattern A should reference correct connections');

        const patternBConnections = patternBReinforced.map(p => p.connection_id).sort();
        const expectedPatternBConnections = [allConnections[2].id, allConnections[3].id].sort();
        this.assert(JSON.stringify(patternBConnections) === JSON.stringify(expectedPatternBConnections), 'Pattern B should reference correct connections');

        // === VALIDATE REINFORCEMENT MATHEMATICS ===
        // Verify that reinforcement incremented by exactly 1 for each pattern-connection pair
        const strengthIncreases = [];
        for (let i = 0; i < initialPatterns.length; i++) {
            const initial = initialPatterns[i];
            const reinforced = reinforcedPatterns.find(p =>
                p.pattern_neuron_id === initial.pattern_neuron_id &&
                p.connection_id === initial.connection_id
            );
            strengthIncreases.push(reinforced.strength - initial.strength);
        }

        this.assert(strengthIncreases.every(increase => increase === 1), 'All pattern strengths should increase by exactly 1');
        this.assert(strengthIncreases.length === 4, 'Should have strength increases for all 4 pattern-connection pairs');

        console.log(`✅ Pattern reinforcement validated: ${strengthIncreases.length} patterns reinforced by +1 each`);
        console.log();
    }

    async testConnectionStrengthUpdates() {
        console.log('Testing Connection Strength Updates:');

        // Clear and set up
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');

        const neuronIds = await this.brain.bulkInsertNeurons(3);

        // Create initial connection
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 2)', [neuronIds[0], neuronIds[1]]);

        // Set up active neurons for reinforcement
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[1]]);

        // Test reinforceConnections - should strengthen existing connection
        await this.brain.reinforceConnections(0);

        // Check connection strength was updated
        const [connections] = await this.brain.conn.query('SELECT strength FROM connections WHERE from_neuron_id = ? AND to_neuron_id = ?', [neuronIds[0], neuronIds[1]]);

        this.assert(connections.length === 1, 'Should have 1 connection');
        this.assert(connections[0].strength === 3, 'Connection strength should be reinforced from 2 to 3', connections[0].strength, 3);

        // Test creating new connection
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[2]]);
        await this.brain.reinforceConnections(0);

        // Check new connection was created
        const [newConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connections WHERE to_neuron_id = ?', [neuronIds[2]]);
        this.assert(newConnections[0].count === 2, 'Should create 2 new connections to new neuron', newConnections[0].count, 2);

        console.log();
    }

    async testPatternMerging() {
        console.log('Testing Pattern Merging (>=66% match):');

        // Clear and set up comprehensive pattern merging scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(10);

        // === SETUP SCENARIO FOR SUCCESSFUL MERGING (>=66% OVERLAP) ===
        // Create a pattern with 3 connections: A->B, B->C, C->D
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 9)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 10)', [neuronIds[2], neuronIds[3]]);

        const [allConnections] = await this.brain.conn.query('SELECT id, from_neuron_id, to_neuron_id FROM connections ORDER BY id');
        this.assert(allConnections.length === 3, 'Should have 3 connections for pattern merging test');

        // Create existing pattern that uses 2 of the 3 connections (66.7% overlap when observed pattern has all 3)
        const existingPatternNeuronId = neuronIds[8];
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 3)', [existingPatternNeuronId, allConnections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 4)', [existingPatternNeuronId, allConnections[1].id]);

        // === VALIDATE INITIAL PATTERN STATE ===
        const [initialPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY connection_id');
        this.assert(initialPatterns.length === 2, 'Should have 2 initial pattern-connection relationships');
        this.assert(initialPatterns[0].strength === 3, 'First pattern connection should have strength 3');
        this.assert(initialPatterns[1].strength === 4, 'Second pattern connection should have strength 4');

        // === SETUP OBSERVED PATTERN THAT SHOULD MERGE ===
        // Observed pattern includes all 3 connections, so existing pattern (2/3 = 66.7%) should match
        const peakConnections = new Map();
        peakConnections.set(neuronIds[2], [allConnections[0].id, allConnections[1].id, allConnections[2].id]); // Peak at neuron C

        // === EXECUTE PATTERN MATCHING ===
        await this.brain.saveObservedPatterns(peakConnections);
        const peakPatterns = await this.brain.matchObservedPatterns(peakConnections);

        this.assert(peakPatterns instanceof Map, 'matchObservedPatterns should return a Map');
        this.assert(peakPatterns.size === 1, 'Should have 1 peak in the result');

        const peakNeuronId = neuronIds[2];
        this.assert(peakPatterns.has(peakNeuronId), 'Should have the correct peak neuron');

        const matchedPatterns = peakPatterns.get(peakNeuronId);
        this.assert(Array.isArray(matchedPatterns), 'Matched patterns should be an array');
        this.assert(matchedPatterns.length === 1, 'Should match exactly 1 existing pattern (>=66% overlap)');
        this.assert(matchedPatterns[0] === existingPatternNeuronId, 'Should match the correct existing pattern');

        // === EXECUTE PATTERN MERGING ===
        await this.brain.mergeMatchedPatterns(peakPatterns, peakConnections);

        // === VALIDATE MERGING RESULTS ===
        const [mergedPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY connection_id');

        // Should now have 3 pattern-connection relationships (original 2 + 1 new from merging)
        this.assert(mergedPatterns.length === 3, 'Should have 3 pattern-connection relationships after merging');

        // All should belong to the same pattern neuron
        this.assert(mergedPatterns.every(p => p.pattern_neuron_id === existingPatternNeuronId), 'All patterns should belong to the existing pattern neuron');

        // Validate strength updates
        const connection0Pattern = mergedPatterns.find(p => p.connection_id === allConnections[0].id);
        const connection1Pattern = mergedPatterns.find(p => p.connection_id === allConnections[1].id);
        const connection2Pattern = mergedPatterns.find(p => p.connection_id === allConnections[2].id);

        this.assert(connection0Pattern.strength === 4, 'First connection should be reinforced from 3 to 4');
        this.assert(connection1Pattern.strength === 5, 'Second connection should be reinforced from 4 to 5');
        this.assert(connection2Pattern.strength === 1, 'Third connection should be newly added with strength 1');

        // === VALIDATE MERGING SPECIFICITY ===
        // Check that the pattern now covers all observed connections
        const patternConnectionIds = mergedPatterns.map(p => p.connection_id).sort();
        const observedConnectionIds = [allConnections[0].id, allConnections[1].id, allConnections[2].id].sort();
        this.assert(JSON.stringify(patternConnectionIds) === JSON.stringify(observedConnectionIds), 'Pattern should now cover all observed connections');

        // === VALIDATE NO DUPLICATE PATTERNS CREATED ===
        const [patternNeuronCount] = await this.brain.conn.query('SELECT COUNT(DISTINCT pattern_neuron_id) as count FROM patterns');
        this.assert(patternNeuronCount[0].count === 1, 'Should still have only 1 pattern neuron (merged, not duplicated)');

        console.log(`✅ Pattern merging validated: 66.7% overlap triggered merge, 2→3 connections, strengths [3,4]→[4,5,1]`);
        console.log();
    }

    async testPatternNonMerging() {
        console.log('Testing Pattern Non-Merging (<66% match):');

        // Clear and set up scenario where pattern should NOT merge due to low overlap
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE observed_patterns');

        const neuronIds = await this.brain.bulkInsertNeurons(12);

        // === SETUP SCENARIO FOR FAILED MERGING (<66% OVERLAP) ===
        // Create 3 connections that will be observed
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 7)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[2], neuronIds[3]]);

        // Create 2 additional connections that are part of existing pattern but NOT observed
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 9)', [neuronIds[3], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 10)', [neuronIds[4], neuronIds[5]]);

        const [allConnections] = await this.brain.conn.query('SELECT id, from_neuron_id, to_neuron_id FROM connections ORDER BY id');
        this.assert(allConnections.length === 5, 'Should have 5 connections for non-merging test');

        // Create existing pattern that uses all 5 connections, but we'll only observe 2 of them (40% overlap)
        const existingPatternNeuronId = neuronIds[10];
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 5)', [existingPatternNeuronId, allConnections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 6)', [existingPatternNeuronId, allConnections[1].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 7)', [existingPatternNeuronId, allConnections[2].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 8)', [existingPatternNeuronId, allConnections[3].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 9)', [existingPatternNeuronId, allConnections[4].id]);

        // === VALIDATE INITIAL STATE ===
        const [initialPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY connection_id');
        this.assert(initialPatterns.length === 5, 'Should have 5 initial pattern-connection relationships');
        this.assert(initialPatterns.every(p => p.pattern_neuron_id === existingPatternNeuronId), 'All initial patterns should belong to existing pattern neuron');

        // === SETUP OBSERVED PATTERN THAT SHOULD NOT MERGE ===
        // Observed pattern includes only 2 connections, but existing pattern has 5 connections (2/5 = 40% < 66%)
        const peakConnections = new Map();
        peakConnections.set(neuronIds[3], [allConnections[0].id, allConnections[1].id]); // Only observe first 2 connections

        // === EXECUTE PATTERN MATCHING ===
        await this.brain.saveObservedPatterns(peakConnections);
        const peakPatterns = await this.brain.matchObservedPatterns(peakConnections);

        this.assert(peakPatterns instanceof Map, 'matchObservedPatterns should return a Map');
        this.assert(peakPatterns.size === 1, 'Should have 1 peak in the result');

        const peakNeuronId = neuronIds[3];
        this.assert(peakPatterns.has(peakNeuronId), 'Should have the correct peak neuron');

        const matchedPatterns = peakPatterns.get(peakNeuronId);
        this.assert(Array.isArray(matchedPatterns), 'Matched patterns should be an array');
        this.assert(matchedPatterns.length === 0, 'Should match 0 existing patterns (40% < 66% threshold)');

        // === EXECUTE PATTERN MERGING (SHOULD BE NO-OP) ===
        await this.brain.mergeMatchedPatterns(peakPatterns, peakConnections);

        // === VALIDATE NO MERGING OCCURRED ===
        const [unchangedPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY connection_id');

        // Should still have exactly the same 5 pattern-connection relationships
        this.assert(unchangedPatterns.length === 5, 'Should still have exactly 5 pattern-connection relationships (no merging)');
        this.assert(unchangedPatterns[0].strength === 5, 'First pattern strength should be unchanged (5)');
        this.assert(unchangedPatterns[1].strength === 6, 'Second pattern strength should be unchanged (6)');
        this.assert(unchangedPatterns[2].strength === 7, 'Third pattern strength should be unchanged (7)');
        this.assert(unchangedPatterns[3].strength === 8, 'Fourth pattern strength should be unchanged (8)');
        this.assert(unchangedPatterns[4].strength === 9, 'Fifth pattern strength should be unchanged (9)');
        this.assert(unchangedPatterns.every(p => p.pattern_neuron_id === existingPatternNeuronId), 'All patterns should still belong to original pattern neuron');

        // === VALIDATE PATTERN CREATION WILL HAPPEN INSTEAD ===
        // Since no merging occurred, createNewPatterns should create a new pattern for this peak
        // Let's simulate this by calling createNewPatterns directly
        await this.brain.createNewPatterns(peakPatterns, peakConnections);

        const [finalPatterns] = await this.brain.conn.query('SELECT pattern_neuron_id, connection_id, strength FROM patterns ORDER BY pattern_neuron_id, connection_id');

        // Should now have 5 (original) + 2 (new pattern) = 7 total pattern-connection relationships
        this.assert(finalPatterns.length === 7, 'Should have 7 total pattern-connection relationships after new pattern creation');

        // Should have 2 distinct pattern neurons
        const [patternNeuronCount] = await this.brain.conn.query('SELECT COUNT(DISTINCT pattern_neuron_id) as count FROM patterns');
        this.assert(patternNeuronCount[0].count === 2, 'Should have 2 distinct pattern neurons (original + new)');

        // Validate the new pattern was created with the 2 observed connections
        const newPatternConnections = finalPatterns.filter(p => p.pattern_neuron_id !== existingPatternNeuronId);
        this.assert(newPatternConnections.length === 2, 'New pattern should have the 2 observed connections');
        this.assert(newPatternConnections.every(p => p.strength === 1), 'New pattern connections should all have strength 1');

        console.log(`✅ Pattern non-merging validated: 40% overlap < 66% threshold → no merge, new pattern created instead`);
        console.log();
    }

    async testEdgeCases() {
        console.log('Testing Edge Cases:');

        // Test empty connections
        const emptyPeaks = this.brain.getObservedPatterns([]);
        this.assert(emptyPeaks instanceof Map, 'getObservedPatterns should handle empty input');
        this.assert(emptyPeaks.size === 0, 'Empty connections should return empty peaks');

        // Test single connection
        const singleConnection = [{ id: 1, from_neuron_id: 1, to_neuron_id: 2, distance: 0, strength: 1 }];
        const singlePeaks = this.brain.getObservedPatterns(singleConnection);
        this.assert(singlePeaks instanceof Map, 'getObservedPatterns should handle single connection');

        // Test activateLevelPatterns with no connections
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');

        const noPatterns = await this.brain.activateLevelPatterns(0);
        this.assert(noPatterns === false, 'Should return false when no connections exist');

        // Test getActiveConnections with no active neurons
        const noConnections = await this.brain.getActiveConnections(0);
        this.assert(Array.isArray(noConnections), 'getActiveConnections should return array');
        this.assert(noConnections.length === 0, 'Should return empty array when no active neurons');

        // Test activatePatternNeurons with no base neurons
        await this.brain.activatePatternNeurons();
        const [levelCheck] = await this.brain.conn.query('SELECT COUNT(*) as count FROM active_neurons WHERE level > 0');
        this.assert(levelCheck[0].count === 0, 'Should not create higher level neurons without base neurons');

        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) await this.brain.conn.release();
		process.exit(0);
    }
}

// Run the tests
const tests = new ConnectionPatternTests();
tests.run().catch(console.error);
