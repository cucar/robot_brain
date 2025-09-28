/**
 * SQL Logic Tests - Critical Mathematical Formulas
 * Simple tests to validate the key SQL expressions used in brain.js
 * Run with: node tests/sql-logic-tests.js
 */

import db from '../db/db.js';

class SQLLogicTests {
    constructor() {
        this.conn = null;
        this.testsPassed = 0;
        this.testsFailed = 0;
    }

    async init() {
        this.conn = await db.getConnection();
        await this.conn.query('USE machine_intelligence');
    }

    async cleanup() {
        if (this.conn) await this.conn.release();
    }

    // Simple assertion helper with debug output
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

    // Floating point comparison with tolerance
    isEqual(actual, expected, tolerance = 1e-10) {
        return Math.abs(actual - expected) < tolerance;
    }

    async runAllTests() {
        console.log('Running SQL Logic Tests...\n');

        await this.testPOWCalculations();
        await this.testDistanceCalculations();
        await this.testRewardDecayFormulas();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);
        
        if (this.testsFailed > 0) {
            process.exit(1);
        }
    }

    async testPOWCalculations() {
        console.log('Testing POW Function Calculations:');
        const baseNeuronMaxAge = 10;

        // Test level-based max age calculations: POW(baseNeuronMaxAge, level + 1)
        const [rows0] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [baseNeuronMaxAge, 0]);
        this.assert(rows0[0].max_age === 10, 'Level 0 max age should be 10', rows0[0].max_age, 10);

        const [rows1] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [baseNeuronMaxAge, 1]);
        this.assert(rows1[0].max_age === 100, 'Level 1 max age should be 100', rows1[0].max_age, 100);

        const [rows2] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [baseNeuronMaxAge, 2]);
        this.assert(rows2[0].max_age === 1000, 'Level 2 max age should be 1000', rows2[0].max_age, 1000);

        console.log();
    }

    async testDistanceCalculations() {
        console.log('Testing Distance Calculations:');
        const baseNeuronMaxAge = 10;

        // Test distance formula: FLOOR(age / POW(baseNeuronMaxAge, level))

        // Level 0: FLOOR(age / 1) = age
        const [rows0] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [5, baseNeuronMaxAge, 0]);
        this.assert(rows0[0].distance === 5, 'Level 0 distance should equal age', rows0[0].distance, 5);

        // Level 1: FLOOR(age / 10)
        const [rows1a] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [9, baseNeuronMaxAge, 1]);
        this.assert(rows1a[0].distance === 0, 'Age 9 at level 1 should have distance 0', rows1a[0].distance, 0);

        const [rows1b] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [15, baseNeuronMaxAge, 1]);
        this.assert(rows1b[0].distance === 1, 'Age 15 at level 1 should have distance 1', rows1b[0].distance, 1);

        // Level 2: FLOOR(age / 100)
        const [rows2] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [250, baseNeuronMaxAge, 2]);
        this.assert(rows2[0].distance === 2, 'Age 250 at level 2 should have distance 2', rows2[0].distance, 2);

        console.log();
    }

    async testRewardDecayFormulas() {
        console.log('Testing Reward Decay Formulas:');
        const baseNeuronMaxAge = 10;

        // Test temporal decay formula: 1.0 + (globalReward - 1.0) * (1.0 - age/levelMaxAge)

        // Positive reward, zero age (full reward)
        const [rows1] = await this.conn.query(
            'SELECT 1.0 + (? - 1.0) * (1.0 - ? / POW(?, ? + 1)) as reward_factor',
            [1.5, 0, baseNeuronMaxAge, 0]
        );
        this.assert(this.isEqual(rows1[0].reward_factor, 1.5), 'Zero age should get full reward (1.5)', rows1[0].reward_factor, 1.5);

        // Positive reward, half age (half decay)
        const [rows2] = await this.conn.query(
            'SELECT 1.0 + (? - 1.0) * (1.0 - ? / POW(?, ? + 1)) as reward_factor',
            [1.5, 5, baseNeuronMaxAge, 0]
        );
        this.assert(this.isEqual(rows2[0].reward_factor, 1.25), 'Half age should get half-decayed reward (1.25)', rows2[0].reward_factor, 1.25);

        // Positive reward, max age (neutral)
        const [rows3] = await this.conn.query(
            'SELECT 1.0 + (? - 1.0) * (1.0 - ? / POW(?, ? + 1)) as reward_factor',
            [1.5, 10, baseNeuronMaxAge, 0]
        );
        this.assert(this.isEqual(rows3[0].reward_factor, 1.0), 'Max age should decay to neutral (1.0)', rows3[0].reward_factor, 1.0);

        // Negative reward
        const [rows4] = await this.conn.query(
            'SELECT 1.0 + (? - 1.0) * (1.0 - ? / POW(?, ? + 1)) as reward_factor',
            [0.5, 0, baseNeuronMaxAge, 0]
        );
        this.assert(this.isEqual(rows4[0].reward_factor, 0.5), 'Zero age should get full negative reward (0.5)', rows4[0].reward_factor, 0.5);

        // Neutral reward (should always be 1.0)
        const [rows5] = await this.conn.query(
            'SELECT 1.0 + (? - 1.0) * (1.0 - ? / POW(?, ? + 1)) as reward_factor',
            [1.0, 5, baseNeuronMaxAge, 0]
        );
        this.assert(this.isEqual(rows5[0].reward_factor, 1.0), 'Neutral reward should always be 1.0', rows5[0].reward_factor, 1.0);

        console.log();
    }
}

// Run the tests
async function main() {
    const tests = new SQLLogicTests();
    try {
        await tests.init();
        await tests.runAllTests();
    } catch (error) {
        console.error('Test execution failed:', error);
        process.exit(1);
    } finally {
        await tests.cleanup();
    }
}

main();
