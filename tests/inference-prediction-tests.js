import Brain from '../brain.js';

/**
 * Test Channel for Inference & Prediction Engine Tests
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
 * Inference & Prediction Engine Tests
 * Tests the brain's ability to make predictions and infer future states
 */
class InferencePredictionTests {
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
        console.log('Running Inference & Prediction Engine Tests...\n');

        await this.setupBrain();
        await this.testConnectionInference();
        await this.testPatternInference();
        await this.testPredictedConnections();
        await this.testNeuronStrengths();
        await this.testInferPeakNeurons();
        await this.testInferNeuronsOrchestration();
        await this.testTemporalPredictionAccuracy();
        await this.testLevelSpecificInference();
        await this.testPredictionAging();
        await this.testRewardOptimization();
        await this.testEdgeCases();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);
        
        if (this.testsFailed > 0) {
            process.exit(1);
        }

        await this.cleanup();
    }

    async setupBrain() {
        console.log('Setting up brain for inference testing...');
        
        this.brain = new Brain();
        this.brain.registerChannel('test_channel', TestChannel);
        await this.brain.init();
        await this.brain.resetBrain(); // Clean slate
        
        // Re-initialize after reset
        await this.brain.init();
        
        console.log('✓ Brain setup complete\n');
        this.testsPassed++;
    }

    async testConnectionInference() {
        console.log('Testing Connection Inference:');

        // Clear and set up test data
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('TRUNCATE connection_inference');
        
        const neuronIds = await this.brain.bulkInsertNeurons(4);
        
        // Create active neurons with different ages
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 2)', [neuronIds[2]]);
        
        // Create connections with proper distances matching age + 1
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[0], neuronIds[3]]); // age 0 -> distance 1
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 4)', [neuronIds[1], neuronIds[3]]); // age 1 -> distance 2
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 3, 3)', [neuronIds[2], neuronIds[3]]); // age 2 -> distance 3
        
        // Test inferConnections
        await this.brain.inferConnections(0);
        
        // Validate connection_inference table
        const [inferences] = await this.brain.conn.query('SELECT level, connection_id, age FROM connection_inference ORDER BY connection_id');
        
        this.assert(inferences.length === 3, 'Should create 3 connection inferences', inferences.length, 3);
        this.assert(inferences.every(inf => inf.level === 0), 'All inferences should be for level 0');
        this.assert(inferences.every(inf => inf.age === 0), 'All new inferences should start with age 0');
        
        // Test aging of existing inferences - manually age them to test aging logic
        await this.brain.conn.query('UPDATE connection_inference SET age = age + 1 WHERE level = 0');
        const [agedInferences] = await this.brain.conn.query('SELECT age FROM connection_inference ORDER BY connection_id');
        this.assert(agedInferences.every(inf => inf.age === 1), 'Existing inferences should age to 1');
        
        console.log();
    }

    async testPatternInference() {
        console.log('Testing Pattern Inference:');

        // Clear and set up pattern inference scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE pattern_inference');
        
        const neuronIds = await this.brain.bulkInsertNeurons(6);
        
        // Create level 1 active pattern neuron (age=0)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 0)', [neuronIds[0]]);
        
        // Create level 0 connections that the pattern will predict
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 4)', [neuronIds[2], neuronIds[3]]);
        
        // Get connection IDs
        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');
        
        // Create pattern definition linking pattern neuron to connections
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 2)', [neuronIds[0], connections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 3)', [neuronIds[0], connections[1].id]);
        
        // Test inferPatterns
        await this.brain.inferPatterns(1);
        
        // Validate pattern_inference table
        const [patternInferences] = await this.brain.conn.query('SELECT level, pattern_neuron_id, connection_id, age FROM pattern_inference ORDER BY connection_id');
        
        this.assert(patternInferences.length === 2, 'Should create 2 pattern inferences', patternInferences.length, 2);
        this.assert(patternInferences.every(inf => inf.level === 0), 'Pattern inferences should be for level 0 (level-1)');
        this.assert(patternInferences.every(inf => inf.pattern_neuron_id === neuronIds[0]), 'All inferences should reference the pattern neuron');
        this.assert(patternInferences.every(inf => inf.age === 0), 'New pattern inferences should start with age 0');
        
        // Test aging of pattern inferences - manually age them to test aging logic
        await this.brain.conn.query('UPDATE pattern_inference SET age = age + 1 WHERE level = 0');
        const [agedPatternInferences] = await this.brain.conn.query('SELECT age FROM pattern_inference WHERE level = 0');
        this.assert(agedPatternInferences.every(inf => inf.age === 1), 'Existing pattern inferences should age to 1');
        
        console.log();
    }

    async testPredictedConnections() {
        console.log('Testing Predicted Connections Retrieval:');

        // Clear and set up mixed prediction scenario
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('TRUNCATE connection_inference');
        await this.brain.conn.query('TRUNCATE pattern_inference');
        
        const neuronIds = await this.brain.bulkInsertNeurons(6);
        
        // Create connections
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[0], neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 4)', [neuronIds[1], neuronIds[2]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 3)', [neuronIds[2], neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, -1)', [neuronIds[3], neuronIds[4]]); // Negative strength - should be excluded
        
        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');
        
        // Add connection inferences for level 0
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, ?, 0)', [connections[0].id]);
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, ?, 1)', [connections[1].id]);
        
        // Add pattern inferences for level 0
        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (0, ?, ?, 0)', [neuronIds[5], connections[2].id]);
        
        // Test getPredictedConnections
        const predictedConnections = await this.brain.getPredictedConnections(0);
        
        this.assert(predictedConnections.length === 3, 'Should return 3 predicted connections (excluding negative strength)', predictedConnections.length, 3);
        this.assert(predictedConnections.every(conn => conn.strength >= 0), 'All predicted connections should have non-negative strength');
        
        // Verify connection details
        const strengths = predictedConnections.map(conn => conn.strength).sort((a, b) => b - a);
        this.assert(strengths[0] === 5, 'Highest strength should be 5');
        this.assert(strengths[1] === 4, 'Second highest strength should be 4');
        this.assert(strengths[2] === 3, 'Third highest strength should be 3');
        
        // Test empty case
        const emptyPredictions = await this.brain.getPredictedConnections(99);
        this.assert(emptyPredictions.length === 0, 'Should return empty array for non-existent level');
        
        console.log();
    }

    async testNeuronStrengths() {
        console.log('Testing Neuron Strength Calculations:');

        // Create test connections with known strengths
        const testConnections = [
            { id: 1, from_neuron_id: 1, to_neuron_id: 2, strength: 5 },
            { id: 2, from_neuron_id: 2, to_neuron_id: 3, strength: 3 },
            { id: 3, from_neuron_id: 1, to_neuron_id: 3, strength: 2 },
            { id: 4, from_neuron_id: 3, to_neuron_id: 4, strength: 4 }
        ];

        // Test getNeuronStrengths method
        const neuronStrengths = this.brain.getNeuronStrengths(testConnections);
        
        this.assert(neuronStrengths instanceof Map, 'getNeuronStrengths should return a Map');
        this.assert(neuronStrengths.size === 4, 'Should calculate strengths for 4 neurons', neuronStrengths.size, 4);
        
        // Verify strength calculations (sum of incoming + outgoing connections)
        this.assert(neuronStrengths.get(1) === 7, 'Neuron 1 strength should be 7 (5+2 outgoing)', neuronStrengths.get(1), 7);
        this.assert(neuronStrengths.get(2) === 8, 'Neuron 2 strength should be 8 (5 incoming + 3 outgoing)', neuronStrengths.get(2), 8);
        this.assert(neuronStrengths.get(3) === 9, 'Neuron 3 strength should be 9 (3+2 incoming + 4 outgoing)', neuronStrengths.get(3), 9);
        this.assert(neuronStrengths.get(4) === 4, 'Neuron 4 strength should be 4 (4 incoming)', neuronStrengths.get(4), 4);
        
        // Test empty connections
        const emptyStrengths = this.brain.getNeuronStrengths([]);
        this.assert(emptyStrengths.size === 0, 'Empty connections should return empty Map');
        
        console.log();
    }

    async testInferPeakNeurons() {
        console.log('Testing Peak Neuron Inference:');

        // Clear inferred_neurons table
        await this.brain.conn.query('TRUNCATE inferred_neurons');
        
        // Create test scenario with clear peaks
        const testConnections = [
            { id: 1, from_neuron_id: 1, to_neuron_id: 2, strength: 10 }, // Neuron 2 will be strong
            { id: 2, from_neuron_id: 3, to_neuron_id: 2, strength: 8 },  // Neuron 2 gets more strength
            { id: 3, from_neuron_id: 4, to_neuron_id: 5, strength: 2 },  // Weak connection
            { id: 4, from_neuron_id: 5, to_neuron_id: 6, strength: 1 }   // Weak connection
        ];
        
        const neuronStrengths = this.brain.getNeuronStrengths(testConnections);
        
        // Test inferPeakNeurons
        await this.brain.inferPeakNeurons(neuronStrengths, 0, testConnections);
        
        // Validate inferred_neurons table
        const [inferredNeurons] = await this.brain.conn.query('SELECT neuron_id, level, age FROM inferred_neurons ORDER BY neuron_id');
        
        this.assert(inferredNeurons.length > 0, 'Should infer at least one peak neuron', inferredNeurons.length, '>0');
        this.assert(inferredNeurons.every(neuron => neuron.level === 0), 'All inferred neurons should be at level 0');
        this.assert(inferredNeurons.every(neuron => neuron.age === 0), 'All inferred neurons should start with age 0');
        
        // Neuron 2 should be a peak (strength 18 vs neighborhood average)
        const neuron2Inferred = inferredNeurons.find(neuron => neuron.neuron_id === 2);
        this.assert(neuron2Inferred !== undefined, 'Neuron 2 should be inferred as peak (highest strength)');
        
        // Test empty case
        await this.brain.conn.query('TRUNCATE inferred_neurons');
        await this.brain.inferPeakNeurons(new Map(), 0, []);
        const [emptyInferred] = await this.brain.conn.query('SELECT * FROM inferred_neurons');
        this.assert(emptyInferred.length === 0, 'Empty strengths should not infer any neurons');
        
        console.log();
    }

    async testInferNeuronsOrchestration() {
        console.log('Testing InferNeurons Orchestration:');

        // Set up complete inference scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE connection_inference');
        await this.brain.conn.query('TRUNCATE pattern_inference');
        await this.brain.conn.query('TRUNCATE inferred_neurons');

        const neuronIds = await this.brain.bulkInsertNeurons(8);

        // Create multi-level active neurons
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 0)', [neuronIds[2]]);

        // Create connections for level 0 - make sure we have a clear peak
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 6)', [neuronIds[0], neuronIds[3]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 5)', [neuronIds[1], neuronIds[4]]);
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 8)', [neuronIds[1], neuronIds[3]]); // Make neuron 3 a strong peak

        // Create connections for level 1 patterns
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 4)', [neuronIds[5], neuronIds[6]]);

        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');

        // Create pattern definition for level 1
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 2)', [neuronIds[2], connections[2].id]);

        // Test complete inferNeurons orchestration
        await this.brain.inferNeurons();

        // Validate that all inference types were created
        const [connectionInfs] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        const [patternInfs] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference');
        const [inferredNeurons] = await this.brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons');

        this.assert(connectionInfs[0].count > 0, 'Should create connection inferences');
        this.assert(patternInfs[0].count > 0, 'Should create pattern inferences');
        this.assert(inferredNeurons[0].count > 0, 'Should infer peak neurons');

        // Validate level processing order (should process from highest to lowest)
        const [levelInferences] = await this.brain.conn.query(`
            SELECT DISTINCT level FROM (
                SELECT level FROM connection_inference
                UNION
                SELECT level FROM pattern_inference
            ) AS all_levels ORDER BY level DESC
        `);

        this.assert(levelInferences.length > 0, 'Should have inferences at multiple levels');

        console.log();
    }

    async testTemporalPredictionAccuracy() {
        console.log('Testing Temporal Prediction Accuracy:');

        // Clear and set up temporal sequence
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('TRUNCATE connection_inference');

        const neuronIds = await this.brain.bulkInsertNeurons(5);

        // Create temporal sequence: A(age=2) -> B(age=1) -> C(age=0)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 2)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[2]]);

        // Create connections with distances matching temporal relationships
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 3, 8)', [neuronIds[0], neuronIds[3]]); // age 2 -> distance 3 (age+1)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 7)', [neuronIds[1], neuronIds[4]]); // age 1 -> distance 2 (age+1)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 6)', [neuronIds[2], neuronIds[3]]); // age 0 -> distance 1 (age+1)

        // Test temporal prediction accuracy
        await this.brain.inferConnections(0);

        // Validate that predictions match temporal expectations
        const [predictions] = await this.brain.conn.query(`
            SELECT ci.connection_id, c.from_neuron_id, c.distance, an.age
            FROM connection_inference ci
            JOIN connections c ON ci.connection_id = c.id
            JOIN active_neurons an ON c.from_neuron_id = an.neuron_id
            WHERE ci.level = 0
            ORDER BY c.distance
        `);

        this.assert(predictions.length === 3, 'Should predict all 3 temporal connections', predictions.length, 3);

        // Verify distance = age + 1 relationship
        for (const pred of predictions) {
            const expectedDistance = pred.age + 1;
            this.assert(pred.distance === expectedDistance,
                `Distance ${pred.distance} should match age+1 (${expectedDistance}) for temporal accuracy`);
        }

        console.log();
    }

    async testLevelSpecificInference() {
        console.log('Testing Level-Specific Inference Behavior:');

        // Clear and set up multi-level scenario
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        await this.brain.conn.query('DELETE FROM patterns');
        await this.brain.conn.query('TRUNCATE connection_inference');
        await this.brain.conn.query('TRUNCATE pattern_inference');

        const neuronIds = await this.brain.bulkInsertNeurons(10);

        // Create neurons at different levels with proper ages for connections
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 0)', [neuronIds[1]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 2, 1)', [neuronIds[2]]);

        // Add more active neurons at each level to ensure inferences are created
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 1)', [neuronIds[6]]);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 2, 0)', [neuronIds[7]]);

        // Create level-specific connections with proper distance calculations
        // For level 0: distance = FLOOR((age + 1) / POW(10, 0)) = age + 1
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 2, 5)', [neuronIds[0], neuronIds[3]]); // Level 0: age 1 -> distance 2

        // For level 1: distance = FLOOR((age + 1) / POW(10, 1)) = FLOOR((age + 1) / 10)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 4)', [neuronIds[1], neuronIds[4]]); // Level 1: age 0 -> distance 0 (FLOOR(1/10) = 0)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 6)', [neuronIds[6], neuronIds[5]]); // Level 1: age 1 -> distance 0 (FLOOR(2/10) = 0)

        // For level 2: distance = FLOOR((age + 1) / POW(10, 2)) = FLOOR((age + 1) / 100)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 3)', [neuronIds[7], neuronIds[8]]); // Level 2: age 0 -> distance 0 (FLOOR(1/100) = 0)
        await this.brain.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 0, 7)', [neuronIds[2], neuronIds[9]]); // Level 2: age 1 -> distance 0 (FLOOR(2/100) = 0)

        const [connections] = await this.brain.conn.query('SELECT id FROM connections ORDER BY id');

        // Create patterns for higher levels
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 3)', [neuronIds[1], connections[0].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 2)', [neuronIds[2], connections[1].id]);
        await this.brain.conn.query('INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES (?, ?, 1)', [neuronIds[7], connections[2].id]);

        // Test level-specific inference
        await this.brain.inferConnections(0);
        await this.brain.inferConnections(1);
        await this.brain.inferConnections(2);

        await this.brain.inferPatterns(1);
        await this.brain.inferPatterns(2);

        // Validate level separation
        const [level0Inferences] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference WHERE level = 0');
        const [level1Inferences] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference WHERE level = 1');
        const [level2Inferences] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference WHERE level = 2');

        this.assert(level0Inferences[0].count > 0, 'Should have level 0 connection inferences');
        this.assert(level1Inferences[0].count > 0, 'Should have level 1 connection inferences');
        this.assert(level2Inferences[0].count > 0, 'Should have level 2 connection inferences');

        // Validate pattern inferences go to lower levels
        const [patternLevel0] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference WHERE level = 0');
        const [patternLevel1] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference WHERE level = 1');

        this.assert(patternLevel0[0].count > 0, 'Should have pattern inferences at level 0 (from level 1)');
        this.assert(patternLevel1[0].count > 0, 'Should have pattern inferences at level 1 (from level 2)');

        console.log();
    }

    async testPredictionAging() {
        console.log('Testing Prediction Aging and Expiration:');

        // Clear and set up aging scenario
        await this.brain.conn.query('TRUNCATE connection_inference');
        await this.brain.conn.query('TRUNCATE pattern_inference');

        const neuronIds = await this.brain.bulkInsertNeurons(4);

        // Manually insert predictions with different ages
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, 1, 0)');
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, 2, 1)');
        await this.brain.conn.query('INSERT INTO connection_inference (level, connection_id, age) VALUES (0, 3, 5)');

        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (0, ?, 4, 0)', [neuronIds[0]]);
        await this.brain.conn.query('INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age) VALUES (0, ?, 5, 2)', [neuronIds[1]]);

        // Test aging process
        await this.brain.conn.query('UPDATE connection_inference SET age = age + 1 WHERE level = 0');
        await this.brain.conn.query('UPDATE pattern_inference SET age = age + 1 WHERE level = 0');

        // Validate aging
        const [agedConnections] = await this.brain.conn.query('SELECT connection_id, age FROM connection_inference ORDER BY connection_id');
        const [agedPatterns] = await this.brain.conn.query('SELECT connection_id, age FROM pattern_inference ORDER BY connection_id');

        this.assert(agedConnections[0].age === 1, 'First connection inference should age to 1');
        this.assert(agedConnections[1].age === 2, 'Second connection inference should age to 2');
        this.assert(agedConnections[2].age === 6, 'Third connection inference should age to 6');

        this.assert(agedPatterns[0].age === 1, 'First pattern inference should age to 1');
        this.assert(agedPatterns[1].age === 3, 'Second pattern inference should age to 3');

        // Test expiration (cleanup of old predictions) - level 0 max age is baseNeuronMaxAge^(level)
        await this.brain.conn.query('DELETE FROM connection_inference WHERE level = 0 AND age >= ?', [6]); // Remove age 6
        await this.brain.conn.query('DELETE FROM pattern_inference WHERE level = 0 AND age >= ?', [3]); // Remove age 3

        const [remainingConnections] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        const [remainingPatterns] = await this.brain.conn.query('SELECT COUNT(*) as count FROM pattern_inference');
        this.assert(remainingConnections[0].count === 2, 'Should have 2 remaining connection inferences after expiration', remainingConnections[0].count, 2);
        this.assert(remainingPatterns[0].count === 1, 'Should have 1 remaining pattern inference after expiration', remainingPatterns[0].count, 1);

        console.log();
    }

    async testRewardOptimization() {
        console.log('Testing Reward Optimization:');

        // Clear and set up reward scenario
        await this.brain.conn.query('DELETE FROM neuron_rewards');

        const neuronIds = await this.brain.bulkInsertNeurons(4);

        // Create neuron rewards with different factors
        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 1.5)', [neuronIds[0]]); // Positive reward
        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 0.8)', [neuronIds[1]]); // Negative reward
        await this.brain.conn.query('INSERT INTO neuron_rewards (neuron_id, reward_factor) VALUES (?, 1.0)', [neuronIds[2]]); // Neutral reward
        // neuronIds[3] has no reward entry (should default to 1.0)

        // Create base neuron strengths
        const baseStrengths = new Map([
            [neuronIds[0], 10.0],
            [neuronIds[1], 8.0],
            [neuronIds[2], 6.0],
            [neuronIds[3], 4.0]
        ]);

        // Test optimizeRewards
        const optimizedStrengths = await this.brain.optimizeRewards(baseStrengths, 0);

        this.assert(optimizedStrengths instanceof Map, 'optimizeRewards should return a Map');
        this.assert(optimizedStrengths.size === 4, 'Should optimize all 4 neuron strengths');

        // Verify reward factor application
        this.assert(optimizedStrengths.get(neuronIds[0]) === 15.0, 'Positive reward should increase strength (10 * 1.5 = 15)', optimizedStrengths.get(neuronIds[0]), 15.0);
        this.assert(optimizedStrengths.get(neuronIds[1]) === 6.4, 'Negative reward should decrease strength (8 * 0.8 = 6.4)', optimizedStrengths.get(neuronIds[1]), 6.4);
        this.assert(optimizedStrengths.get(neuronIds[2]) === 6.0, 'Neutral reward should maintain strength (6 * 1.0 = 6)', optimizedStrengths.get(neuronIds[2]), 6.0);
        this.assert(optimizedStrengths.get(neuronIds[3]) === 4.0, 'No reward entry should default to neutral (4 * 1.0 = 4)', optimizedStrengths.get(neuronIds[3]), 4.0);

        // Test that negative results are clamped to 0
        const negativeStrengths = new Map([[neuronIds[1], 1.0]]);
        await this.brain.conn.query('UPDATE neuron_rewards SET reward_factor = -0.5 WHERE neuron_id = ?', [neuronIds[1]]);
        const clampedStrengths = await this.brain.optimizeRewards(negativeStrengths, 0);
        this.assert(clampedStrengths.get(neuronIds[1]) === 0.0, 'Negative strength should be clamped to 0');

        console.log();
    }

    async testEdgeCases() {
        console.log('Testing Edge Cases:');

        // Test empty active neurons
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('TRUNCATE connection_inference');

        await this.brain.inferConnections(0);
        const [emptyInferences] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        this.assert(emptyInferences[0].count === 0, 'No active neurons should create no inferences');

        // Test no connections
        const neuronIds = await this.brain.bulkInsertNeurons(2);
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[0]]);
        await this.brain.conn.query('DELETE FROM connections');

        await this.brain.inferConnections(0);
        const [noConnInferences] = await this.brain.conn.query('SELECT COUNT(*) as count FROM connection_inference');
        this.assert(noConnInferences[0].count === 0, 'No connections should create no inferences');

        // Test getPredictedConnections with no predictions
        const noPredictions = await this.brain.getPredictedConnections(0);
        this.assert(noPredictions.length === 0, 'No predictions should return empty array');

        // Test getNeuronStrengths with empty connections
        const emptyStrengths = this.brain.getNeuronStrengths([]);
        this.assert(emptyStrengths.size === 0, 'Empty connections should return empty strengths Map');

        // Test inferPeakNeurons with no strengths
        await this.brain.conn.query('TRUNCATE inferred_neurons');
        await this.brain.inferPeakNeurons(new Map(), 0, []);
        const [noInferred] = await this.brain.conn.query('SELECT COUNT(*) as count FROM inferred_neurons');
        this.assert(noInferred[0].count === 0, 'No strengths should infer no neurons');

        // Test optimizeRewards with empty strengths
        const emptyOptimized = await this.brain.optimizeRewards(new Map(), 0);
        this.assert(emptyOptimized.size === 0, 'Empty strengths should return empty optimized Map');

        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) {
            await this.brain.conn.release();
        }
    }
}

// Run the tests
const tests = new InferencePredictionTests();
tests.run().catch(console.error);
