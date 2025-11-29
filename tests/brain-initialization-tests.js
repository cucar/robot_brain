/**
 * Brain Initialization & Configuration Tests
 * Tests brain constructor, hyperparameters, channel registration, and database initialization
 * Run with: node tests/brain-initialization-tests.js
 */

import BrainMySQL from '../brain-mysql.js';
import Channel from '../channels/channel.js';

// Mock channel for testing
class TestChannel extends Channel {
	getEventDimensions() {
        return ['test_input_1', 'test_input_2'];
    }
    
    getOutputDimensions() {
        return ['test_output_1'];
    }
    
    async getFrameEvents() {
        return [{ test_input_1: 0.5, test_input_2: 1.0 }];
    }

    async executeOutputs(inputs, outputs) {
        return [...inputs, ...(outputs || [])];
    }

    getExplorationAction() {
        return { test_output_1: 1 };
    }
}

class BrainInitializationTests {
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
        console.log('Running Brain Initialization & Configuration Tests...\n');

        await this.testChannelRegistration();
        await this.testDatabaseInitialization();
        await this.testResetOperations();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);

        if (this.testsFailed > 0) {
            process.exit(1);
        }
    }

    async testChannelRegistration() {
        console.log('Testing Channel Registration:');

        this.brain = new BrainMySQL();

        // Test that channel registration creates proper instances
        this.brain.registerChannel('test_channel', TestChannel);

        const channel = this.brain.channels.get('test_channel');
        this.assert(channel instanceof TestChannel, 'Registered channel should be instance of TestChannel');
        this.assert(channel.name === 'test_channel', 'Channel name should be set correctly', channel.name, 'test_channel');

        // Test that channels can provide their dimensions (critical for brain operation)
        const inputDims = channel.getInputDimensions();
        const outputDims = channel.getOutputDimensions();
        this.assert(Array.isArray(inputDims) && inputDims.length > 0, 'Channel should provide input dimensions');
        this.assert(Array.isArray(outputDims) && outputDims.length > 0, 'Channel should provide output dimensions');

        console.log();
    }

    async testDatabaseInitialization() {
        console.log('Testing Database Initialization:');
        
        try {
            // Test init method
            await this.brain.init();
            
            // Test database connection is established
            this.assert(this.brain.conn !== null, 'Database connection should be established');
            
            // Test dimensions are loaded
            this.assert(this.brain.dimensionNameToId !== undefined, 'dimensionNameToId should be loaded');
            this.assert(this.brain.dimensionIdToName !== undefined, 'dimensionIdToName should be loaded');
            this.assert(typeof this.brain.dimensionNameToId === 'object', 'dimensionNameToId should be an object');
            this.assert(typeof this.brain.dimensionIdToName === 'object', 'dimensionIdToName should be an object');
            
            // Test that test channel dimensions were created
            const expectedDimensions = ['test_input_1', 'test_input_2', 'test_output_1'];
            for (const dimName of expectedDimensions) {
                this.assert(
                    this.brain.dimensionNameToId[dimName] !== undefined, 
                    `Dimension ${dimName} should be loaded`,
                    this.brain.dimensionNameToId[dimName] !== undefined,
                    true
                );
            }
            
            // Test dimension mapping consistency
            for (const [name, id] of Object.entries(this.brain.dimensionNameToId)) {
                this.assert(
                    this.brain.dimensionIdToName[id] === name,
                    `Dimension mapping should be consistent for ${name}`,
                    this.brain.dimensionIdToName[id],
                    name
                );
            }
            
        } catch (error) {
            console.log(`✗ Database initialization failed: ${error.message}`);
            this.testsFailed++;
        }
        
        console.log();
    }

    async testResetOperations() {
        console.log('Testing Reset Operations:');
        
        try {
            // Test resetContext (memory tables only)
            await this.brain.resetContext();
            console.log('✓ resetContext executed without errors');
            this.testsPassed++;
            
            // Test resetBrain (all tables)
            await this.brain.resetBrain();
            console.log('✓ resetBrain executed without errors');
            this.testsPassed++;
            
            // After hard reset, dimensions should be cleared
            const [rows] = await this.brain.conn.query('SELECT COUNT(*) as count FROM dimensions');
            this.assert(rows[0].count === 0, 'Dimensions should be cleared after resetBrain', rows[0].count, 0);
            
        } catch (error) {
            console.log(`✗ Reset operations failed: ${error.message}`);
            this.testsFailed++;
        }
        
        console.log();
    }

    async cleanup() {
        if (this.brain && this.brain.conn) await this.brain.conn.release();
		process.exit(0);
	}
}

// Run the tests
async function main() {
    const tests = new BrainInitializationTests();
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
