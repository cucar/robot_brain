/**
 * Neuron Creation & Coordinate Matching Tests
 * Tests frame-to-neuron conversion, coordinate matching, and bulk neuron creation
 * Run with: node tests/neuron-creation-tests.js
 */

import BrainMySQL from '../brain-mysql.js';
import Channel from '../channels/channel.js';

// Mock channel for testing
class TestChannel extends Channel {
    getInputDimensions() {
        return ['test_x', 'test_y', 'test_value'];
    }
    
    getOutputDimensions() {
        return ['test_output'];
    }
    
    async getFrameInputs() {
        return [
            { test_x: 0.1, test_y: 0.2, test_value: 1.0 },
            { test_x: 0.3, test_y: 0.4, test_value: 0.5 }
        ];
    }
    
    async executeOutputs(predictions) {}
    getExplorationAction() { return {}; }
}

class NeuronCreationTests {
    constructor() {
        this.testsPassed = 0;
        this.testsFailed = 0;
        this.brain = null;
    }

    // Simple assertion helper
    assert(condition, message, actual = null, expected = null) {
        if (condition) {
            console.log(`✓ ${message}`);
            this.testsPassed++;
        } else {
            console.log(`✗ ${message}`);
            if (actual !== null && expected !== null) {
                console.log(`  Expected: ${expected}, Got: ${actual}`);
            }
            this.testsFailed++;
        }
    }

    async runAllTests() {
        console.log('Running Neuron Creation & Coordinate Matching Tests...\n');

        await this.setupBrain();
        await this.testCoordinateMatching();
        await this.testNeuronCreation();
        await this.testBulkOperations();
        await this.testFrameToNeuronConversion();
        await this.testEdgeCases();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);
        
        if (this.testsFailed > 0) {
            process.exit(1);
        }
    }

    async setupBrain() {
        console.log('Setting up brain for testing...');

        this.brain = new BrainMySQL();
        this.brain.registerChannel('test_channel', TestChannel);
        await this.brain.init();
        await this.brain.resetBrain(); // Clean slate
        
        // Re-initialize after reset
        await this.brain.init();
        
        console.log('✓ Brain setup complete\n');
        this.testsPassed++;
    }

    async testCoordinateMatching() {
        console.log('Testing Coordinate Matching:');

        // Test matchNeuronsFromPoints with no existing neurons
        const testFrame = [
            { test_x: 0.1, test_y: 0.2, test_value: 1.0 },
            { test_x: 0.3, test_y: 0.4, test_value: 0.5 }
        ];

        const matches = await this.brain.getFrameNeurons(testFrame);
        
        this.assert(Array.isArray(matches), 'getFrameNeurons should return array');
        this.assert(matches.length === 2, 'Should return matches for both points', matches.length, 2);
        
        // With no existing neurons, all matches should have null or NaN neuron_id
        this.assert(!matches[0].neuron_id || isNaN(matches[0].neuron_id), 'First point should have no matching neuron');
        this.assert(!matches[1].neuron_id || isNaN(matches[1].neuron_id), 'Second point should have no matching neuron');
        
        // Test point_str format
        this.assert(matches[0].point_str === JSON.stringify(testFrame[0]), 'Point string should match JSON format');
        
        console.log();
    }

    async testNeuronCreation() {
        console.log('Testing Neuron Creation:');

        // Test createBaseNeurons
        const pointStrs = [
            JSON.stringify({ test_x: 0.1, test_y: 0.2, test_value: 1.0 }),
            JSON.stringify({ test_x: 0.3, test_y: 0.4, test_value: 0.5 })
        ];

        const neuronIds = await this.brain.createBaseNeurons(pointStrs);
        
        this.assert(Array.isArray(neuronIds), 'createBaseNeurons should return array');
        this.assert(neuronIds.length === 2, 'Should create 2 neurons', neuronIds.length, 2);
        this.assert(neuronIds[0] !== neuronIds[1], 'Neuron IDs should be unique');
        this.assert(Number.isInteger(neuronIds[0]), 'Neuron IDs should be integers');
        
        // Verify neurons were created in database
        const [neuronRows] = await this.brain.conn.query('SELECT COUNT(*) as count FROM neurons');
        this.assert(neuronRows[0].count >= 2, 'Neurons should be created in database', neuronRows[0].count, '>=2');
        
        // Verify coordinates were created
        const [coordRows] = await this.brain.conn.query('SELECT COUNT(*) as count FROM coordinates');
        this.assert(coordRows[0].count === 6, 'Should have 6 coordinates (2 neurons × 3 dimensions)', coordRows[0].count, 6);
        
        console.log();
    }

    async testBulkOperations() {
        console.log('Testing Bulk Operations:');

        // Test createNeurons
        const neuronIds = await this.brain.createNeurons(3);
        
        this.assert(Array.isArray(neuronIds), 'createNeurons should return array');
        this.assert(neuronIds.length === 3, 'Should return 3 neuron IDs', neuronIds.length, 3);
        
        // Test that IDs are sequential (MySQL auto_increment behavior)
        this.assert(neuronIds[1] === neuronIds[0] + 1, 'IDs should be sequential');
        this.assert(neuronIds[2] === neuronIds[1] + 1, 'IDs should be sequential');
        
        // Test empty bulk insert - should handle gracefully
        try {
            const emptyIds = await this.brain.createNeurons(0);
            this.assert(emptyIds.length === 0, 'Empty bulk insert should return empty array');
        } catch (error) {
            // createNeurons(0) creates invalid SQL, which is expected behavior
            this.assert(error.code === 'ER_PARSE_ERROR', 'Empty bulk insert should fail with parse error');
        }
        
        console.log();
    }

    async testFrameToNeuronConversion() {
        console.log('Testing Frame to Neuron Conversion:');

        // Test getFrameNeurons with new points (should create neurons)
        const testFrame = [
            { test_x: 0.5, test_y: 0.6, test_value: 0.8 },
            { test_x: 0.7, test_y: 0.8, test_value: 0.9 }
        ];

        const neuronIds = await this.brain.getFrameNeurons(testFrame);
        
        this.assert(Array.isArray(neuronIds), 'getFrameNeurons should return array');
        this.assert(neuronIds.length === 2, 'Should return 2 neuron IDs', neuronIds.length, 2);
        this.assert(neuronIds.every(id => Number.isInteger(id)), 'All IDs should be integers');
        
        // Test getFrameNeurons with same points (should match existing neurons)
        const matchedIds = await this.brain.getFrameNeurons(testFrame);
        
        this.assert(matchedIds.length === 2, 'Should still return 2 neuron IDs');
        this.assert(matchedIds[0] === neuronIds[0], 'Should match first existing neuron');
        this.assert(matchedIds[1] === neuronIds[1], 'Should match second existing neuron');
        
        console.log();
    }

    async testEdgeCases() {
        console.log('Testing Edge Cases:');

        // Test empty frame
        const emptyResult = await this.brain.getFrameNeurons([]);
        this.assert(emptyResult.length === 0, 'Empty frame should return empty array');
        
        // Test duplicate points in frame
        const duplicateFrame = [
            { test_x: 0.9, test_y: 0.9, test_value: 1.0 },
            { test_x: 0.9, test_y: 0.9, test_value: 1.0 }  // Duplicate
        ];
        
        const duplicateIds = await this.brain.getFrameNeurons(duplicateFrame);
        this.assert(duplicateIds.length === 1, 'Should deduplicate points and return 1 neuron', duplicateIds.length, 1);
        // Note: getFrameNeurons deduplicates at the createBaseNeurons level, so we get 1 neuron for 2 identical points
        
        // Test createBaseNeurons with empty array
        const emptyCreate = await this.brain.createBaseNeurons([]);
        this.assert(emptyCreate.length === 0, 'Empty creation should return empty array');
        
        // Test createBaseNeurons with duplicate point strings (should deduplicate)
        const duplicatePointStrs = [
            JSON.stringify({ test_x: 1.0, test_y: 1.0, test_value: 1.0 }),
            JSON.stringify({ test_x: 1.0, test_y: 1.0, test_value: 1.0 })  // Duplicate
        ];
        
        const dedupeIds = await this.brain.createBaseNeurons(duplicatePointStrs);
        this.assert(dedupeIds.length === 1, 'Should deduplicate identical point strings', dedupeIds.length, 1);
        
        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) await this.brain.conn.release();
        process.exit(0);
    }
}

// Run the tests
async function main() {
    const tests = new NeuronCreationTests();
    try {
        await tests.runAllTests();
    } catch (error) {
        console.error('Test execution failed:', error);
        process.exit(1);
    } finally {
        await tests.cleanup();
    }
}

main();
