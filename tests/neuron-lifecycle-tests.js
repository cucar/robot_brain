/**
 * Neuron Lifecycle & Activation Tests
 * Tests neuron activation, aging, and lifecycle management
 * Run with: node tests/neuron-lifecycle-tests.js
 */

import BrainMySQL from '../brain-mysql.js';
import Channel from '../channels/channel.js';

// Mock channel for testing
class TestChannel extends Channel {
	getEventDimensions() {
        return ['test_x', 'test_y', 'test_value'];
    }
    
    getOutputDimensions() {
        return ['test_output'];
    }
    
    async getFrameEvents() {
        return [
            { test_x: 0.1, test_y: 0.2, test_value: 1.0 }
        ];
    }

    async executeOutputs(inputs, outputs) { return inputs; }
    getExplorationAction() { return {}; }
}

class NeuronLifecycleTests {
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
        console.log('Running Neuron Lifecycle & Activation Tests...\n');

        await this.setupBrain();
        await this.testNeuronActivation();
        await this.testNeuronAging();
        await this.testLevelBasedAging();
        await this.testActivationIntegration();
        await this.testReinforceConnectionsDirect();
        await this.testConnectionReinforcement();

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

    async testNeuronActivation() {
        console.log('Testing Neuron Activation:');

        // Create some test neurons first
        const neuronIds = await this.brain.createNeurons(3);
        
        // Test insertActiveNeurons
        await this.brain.insertActiveNeurons(neuronIds, 0);
        
        // Verify neurons were activated
        const [activeRows] = await this.brain.conn.query('SELECT COUNT(*) as count FROM active_neurons WHERE level = 0');
        this.assert(activeRows[0].count === 3, 'Should have 3 active neurons at level 0', activeRows[0].count, 3);
        
        // Test activation at different level
        await this.brain.insertActiveNeurons([neuronIds[0]], 1);
        const [level1Rows] = await this.brain.conn.query('SELECT COUNT(*) as count FROM active_neurons WHERE level = 1');
        this.assert(level1Rows[0].count === 1, 'Should have 1 active neuron at level 1', level1Rows[0].count, 1);
        
        // Test empty activation (should not error)
        await this.brain.insertActiveNeurons([], 0);
        
        // Verify age is 0 for newly activated neurons
        const [ageRows] = await this.brain.conn.query('SELECT age FROM active_neurons WHERE level = 0');
        this.assert(ageRows.every(row => row.age === 0), 'All newly activated neurons should have age 0');
        
        console.log();
    }

    async testNeuronAging() {
        console.log('Testing Neuron Aging:');

        // Clear and set up test neurons
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM inferred_neurons');
        
        const neuronIds = await this.brain.createNeurons(2);
        await this.brain.insertActiveNeurons(neuronIds, 0);
        
        // Insert some inferred neurons too
        await this.brain.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[0]]);
        
        // Test aging
        await this.brain.ageNeurons();
        
        // Verify ages increased
        const [activeAges] = await this.brain.conn.query('SELECT age FROM active_neurons ORDER BY neuron_id');
        this.assert(activeAges.every(row => row.age === 1), 'All active neurons should age to 1');
        
        const [inferredAges] = await this.brain.conn.query('SELECT age FROM inferred_neurons ORDER BY neuron_id');
        this.assert(inferredAges.every(row => row.age === 1), 'All inferred neurons should age to 1');
        
        console.log();
    }

    async testLevelBasedAging() {
        console.log('Testing Level-Based Aging:');

        // Clear and set up neurons at different levels
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM inferred_neurons');
        
        const neuronIds = await this.brain.createNeurons(3);
        
        // Insert neurons at different levels with high ages
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 9)', [neuronIds[0]]);  // Close to level 0 max (10)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 1, 99)', [neuronIds[1]]); // Close to level 1 max (100)
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 5)', [neuronIds[2]]);  // Safe age
        
        // Age neurons
        await this.brain.ageNeurons();
        
        // Check which neurons survived (age < POW(baseNeuronMaxAge, level + 1))
        const [survivingRows] = await this.brain.conn.query('SELECT neuron_id, level, age FROM active_neurons ORDER BY neuron_id');
        
        // neuronIds[0] at level 0: age 9+1=10, max=POW(10,1)=10, should be deleted (age >= max)
        // neuronIds[1] at level 1: age 99+1=100, max=POW(10,2)=100, should be deleted (age >= max)  
        // neuronIds[2] at level 0: age 5+1=6, max=POW(10,1)=10, should survive (age < max)
        
        this.assert(survivingRows.length === 1, 'Only 1 neuron should survive aging', survivingRows.length, 1);
        this.assert(survivingRows[0].neuron_id === neuronIds[2], 'The safe-age neuron should survive');
        this.assert(survivingRows[0].age === 6, 'Surviving neuron should have age 6', survivingRows[0].age, 6);
        
        console.log();
    }

    async testActivationIntegration() {
        console.log('Testing Activation Integration:');

        // Test activateNeurons (which calls insertActiveNeurons + reinforceConnections)
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        
        const neuronIds = await this.brain.createNeurons(2);
        
        // Activate first neuron, then second (to create temporal connection)
        await this.brain.activateNeurons([neuronIds[0]], 0);
        
        // Age the first neuron so it's older
        await this.brain.conn.query('UPDATE active_neurons SET age = 1 WHERE neuron_id = ?', [neuronIds[0]]);
        
        // Activate second neuron (should create connection from first to second)
        await this.brain.activateNeurons([neuronIds[1]], 0);
        
        // Check that connection was created
        const [connectionRows] = await this.brain.conn.query('SELECT * FROM connections');
        this.assert(connectionRows.length >= 1, 'Should create at least 1 connection');
        
        if (connectionRows.length > 0) {
            const conn = connectionRows[0];
            this.assert(conn.from_neuron_id === neuronIds[0], 'Connection should be from first neuron');
            this.assert(conn.to_neuron_id === neuronIds[1], 'Connection should be to second neuron');
            this.assert(conn.strength === 1, 'Initial connection strength should be 1', conn.strength, 1);
        }
        
        console.log();
    }

    async testReinforceConnectionsDirect() {
        console.log('Testing reinforceConnections() Method Directly:');

        // Clear and set up
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');

        const neuronIds = await this.brain.createNeurons(3);

        // Set up active neurons with different ages
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 2)', [neuronIds[0]]); // Older
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[1]]); // Middle
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 0)', [neuronIds[2]]); // New

        // Call reinforceConnections directly
        await this.brain.reinforceConnections(0);

        // Check connections were created from older neurons to new neuron (age=0)
        const [connectionRows] = await this.brain.conn.query('SELECT * FROM connections ORDER BY from_neuron_id');

        // Should create connections: neuronIds[0] -> neuronIds[2] and neuronIds[1] -> neuronIds[2]
        // Distance should be FLOOR(age / POW(10, 0)) = age
        this.assert(connectionRows.length === 2, 'Should create 2 connections to new neuron', connectionRows.length, 2);

        const conn1 = connectionRows.find(c => c.from_neuron_id === neuronIds[0]);
        const conn2 = connectionRows.find(c => c.from_neuron_id === neuronIds[1]);

        this.assert(conn1 && conn1.to_neuron_id === neuronIds[2], 'Should connect from oldest to newest');
        this.assert(conn1.distance === 2, 'Distance should match age (2)', conn1.distance, 2);
        this.assert(conn1.strength === 1, 'Initial strength should be 1', conn1.strength, 1);

        this.assert(conn2 && conn2.to_neuron_id === neuronIds[2], 'Should connect from middle to newest');
        this.assert(conn2.distance === 1, 'Distance should match age (1)', conn2.distance, 1);
        this.assert(conn2.strength === 1, 'Initial strength should be 1', conn2.strength, 1);

        console.log();
    }

    async testConnectionReinforcement() {
        console.log('Testing Connection Reinforcement Integration:');

        // Clear and set up
        await this.brain.conn.query('DELETE FROM active_neurons');
        await this.brain.conn.query('DELETE FROM connections');
        
        const neuronIds = await this.brain.createNeurons(2);
        
        // Create initial connection with correct distance for reinforcement
        // For level 0, distance = FLOOR(age / POW(10, 0)) = FLOOR(age / 1) = age
        await this.brain.conn.query(
            'INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength) VALUES (?, ?, 1, 1)',
            [neuronIds[0], neuronIds[1]]
        );

        // Activate neurons to reinforce the connection
        await this.brain.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, 0, 1)', [neuronIds[0]]);
        await this.brain.activateNeurons([neuronIds[1]], 0); // This should reinforce the connection

        // Check connection strength increased
        const [connectionRows] = await this.brain.conn.query('SELECT strength FROM connections WHERE from_neuron_id = ? AND to_neuron_id = ?', [neuronIds[0], neuronIds[1]]);
        this.assert(connectionRows.length === 1, 'Should have 1 connection');
        this.assert(connectionRows[0].strength === 2, 'Connection strength should be reinforced to 2', connectionRows[0].strength, 2);
        
        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) await this.brain.conn.release();
		process.exit(0);
    }
}

// Run the tests
async function main() {
    const tests = new NeuronLifecycleTests();
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
