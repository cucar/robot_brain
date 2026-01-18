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
        this.conn = await db();
        await this.conn.query('USE machine_intelligence');
    }

    async cleanup() {
        if (this.conn) await this.conn.end();
		process.exit(0);
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
        await this.testRewardDecayFormulasWithDifferentBase();

        console.log(`\nResults: ${this.testsPassed} passed, ${this.testsFailed} failed`);

        if (this.testsFailed > 0) {
            process.exit(1);
        }
    }

    async testPOWCalculations() {
        console.log('Testing POW Function Calculations:');
        const contextLength = 10;

        // Test level-based max age calculations: POW(contextLength, level + 1)
        const [rows0] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [contextLength, 0]);
        this.assert(rows0[0].max_age === 10, 'Level 0 max age should be 10', rows0[0].max_age, 10);

        const [rows1] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [contextLength, 1]);
        this.assert(rows1[0].max_age === 100, 'Level 1 max age should be 100', rows1[0].max_age, 100);

        const [rows2] = await this.conn.query('SELECT POW(?, ? + 1) as max_age', [contextLength, 2]);
        this.assert(rows2[0].max_age === 1000, 'Level 2 max age should be 1000', rows2[0].max_age, 1000);

        console.log();
    }

    async testDistanceCalculations() {
        console.log('Testing Distance Calculations:');
        const contextLength = 10;

        // Test distance formula: FLOOR(age / POW(contextLength, level))

        // Level 0: FLOOR(age / 1) = age
        const [rows0] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [5, contextLength, 0]);
        this.assert(rows0[0].distance === 5, 'Level 0 distance should equal age', rows0[0].distance, 5);

        // Level 1: FLOOR(age / 10)
        const [rows1a] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [9, contextLength, 1]);
        this.assert(rows1a[0].distance === 0, 'Age 9 at level 1 should have distance 0', rows1a[0].distance, 0);

        const [rows1b] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [15, contextLength, 1]);
        this.assert(rows1b[0].distance === 1, 'Age 15 at level 1 should have distance 1', rows1b[0].distance, 1);

        // Level 2: FLOOR(age / 100)
        const [rows2] = await this.conn.query('SELECT FLOOR(? / POW(?, ?)) as distance', [250, contextLength, 2]);
        this.assert(rows2[0].distance === 2, 'Age 250 at level 2 should have distance 2', rows2[0].distance, 2);

        console.log();
    }

    async testRewardDecayFormulas() {
        console.log('Testing Reward Decay Formulas (Exponential Temporal Decay):');
        const contextLength = 10;

        // Generalized formula with exponential decay matching connection distance grouping:
        // decayFactor = (N + 1 - bucketWithinTier) / POW(N, tier + 1)
        // Where:
        //   N = contextLength
        //   tier = FLOOR(LN(age) / LN(N))
        //   bucketWithinTier = CEIL(age / POW(N, tier))
        // Note: age >= 1 is guaranteed by the WHERE clause

        const rewardFormula = `
            1.0 + (? - 1.0) * (
                (? + 1 - CEIL(? / POW(?, FLOOR(LN(?) / LN(?)))))
                / POW(?, FLOOR(LN(?) / LN(?)) + 1)
            )
        `;

        // Test base tier (ages 1-10): decay from 1.0 to 0.1
        const [age1] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 1, contextLength, 1, contextLength, contextLength, 1, contextLength]
        );
        this.assert(this.isEqual(age1[0].reward_factor, 1.5), 'Age 1 should get full reward (decay=1.0)', age1[0].reward_factor, 1.5);

        const [age2] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 2, contextLength, 2, contextLength, contextLength, 2, contextLength]
        );
        this.assert(this.isEqual(age2[0].reward_factor, 1.45), 'Age 2 should get decay=0.9 (1.0 + 0.5*0.9 = 1.45)', age2[0].reward_factor, 1.45);

        const [age5] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 5, contextLength, 5, contextLength, contextLength, 5, contextLength]
        );
        this.assert(this.isEqual(age5[0].reward_factor, 1.3), 'Age 5 should get decay=0.6 (1.0 + 0.5*0.6 = 1.3)', age5[0].reward_factor, 1.3);

        const [age9] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 9, contextLength, 9, contextLength, contextLength, 9, contextLength]
        );
        this.assert(this.isEqual(age9[0].reward_factor, 1.1), 'Age 9 should get decay=0.2 (1.0 + 0.5*0.2 = 1.1)', age9[0].reward_factor, 1.1);

        const [age10] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 10, contextLength, 10, contextLength, contextLength, 10, contextLength]
        );
        this.assert(this.isEqual(age10[0].reward_factor, 1.05), 'Age 10 should get decay=0.1 (1.0 + 0.5*0.1 = 1.05)', age10[0].reward_factor, 1.05);

        // Test first exponential tier (ages 11-100): decay from 0.09 to 0.01
        const [age11] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 11, contextLength, 11, contextLength, contextLength, 11, contextLength]
        );
        this.assert(this.isEqual(age11[0].reward_factor, 1.045), 'Age 11 should get decay=0.09 (bucket 11-20)', age11[0].reward_factor, 1.045);

        const [age15] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 15, contextLength, 15, contextLength, contextLength, 15, contextLength]
        );
        this.assert(this.isEqual(age15[0].reward_factor, 1.045), 'Age 15 should get decay=0.09 (same bucket as 11-20)', age15[0].reward_factor, 1.045);

        const [age25] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 25, contextLength, 25, contextLength, contextLength, 25, contextLength]
        );
        this.assert(this.isEqual(age25[0].reward_factor, 1.04), 'Age 25 should get decay=0.08 (bucket 21-30)', age25[0].reward_factor, 1.04);

        const [age78] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 78, contextLength, 78, contextLength, contextLength, 78, contextLength]
        );
        this.assert(this.isEqual(age78[0].reward_factor, 1.015), 'Age 78 should get decay=0.03 (bucket 71-80)', age78[0].reward_factor, 1.015);

        const [age100] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 100, contextLength, 100, contextLength, contextLength, 100, contextLength]
        );
        this.assert(this.isEqual(age100[0].reward_factor, 1.005), 'Age 100 should get decay=0.01 (bucket 91-100)', age100[0].reward_factor, 1.005);

        // Test second exponential tier (ages 101-1000): decay from 0.009 to 0.001
        const [age101] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 101, contextLength, 101, contextLength, contextLength, 101, contextLength]
        );
        this.assert(this.isEqual(age101[0].reward_factor, 1.0045), 'Age 101 should get decay=0.009 (bucket 101-200)', age101[0].reward_factor, 1.0045);

        const [age250] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 250, contextLength, 250, contextLength, contextLength, 250, contextLength]
        );
        this.assert(this.isEqual(age250[0].reward_factor, 1.004), 'Age 250 should get decay=0.008 (bucket 201-300)', age250[0].reward_factor, 1.004);

        const [age1000] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 1000, contextLength, 1000, contextLength, contextLength, 1000, contextLength]
        );
        this.assert(this.isEqual(age1000[0].reward_factor, 1.0005), 'Age 1000 should get decay=0.001 (bucket 901-1000)', age1000[0].reward_factor, 1.0005);

        // Test with negative reward
        const [negAge1] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [0.5, contextLength, 1, contextLength, 1, contextLength, contextLength, 1, contextLength]
        );
        this.assert(this.isEqual(negAge1[0].reward_factor, 0.5), 'Age 1 with negative reward should get full penalty (0.5)', negAge1[0].reward_factor, 0.5);

        const [negAge50] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [0.5, contextLength, 50, contextLength, 50, contextLength, contextLength, 50, contextLength]
        );
        this.assert(this.isEqual(negAge50[0].reward_factor, 0.97), 'Age 50 with negative reward should get decay=0.06 (bucket 41-50: 1.0 + (-0.5)*0.06 = 0.97)', negAge50[0].reward_factor, 0.97);

        // Test neutral reward (should always be 1.0)
        const [neutral] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.0, contextLength, 50, contextLength, 50, contextLength, contextLength, 50, contextLength]
        );
        this.assert(this.isEqual(neutral[0].reward_factor, 1.0), 'Neutral reward should always be 1.0 regardless of age', neutral[0].reward_factor, 1.0);

        console.log();
    }

    async testRewardDecayFormulasWithDifferentBase() {
        console.log('Testing Reward Decay Formulas with contextLength = 5:');
        const contextLength = 5;

        // Same generalized formula, but with N=5 instead of N=10
        // This tests that the formula works for any contextLength value

        const rewardFormula = `
            1.0 + (? - 1.0) * (
                (? + 1 - CEIL(? / POW(?, FLOOR(LN(?) / LN(?)))))
                / POW(?, FLOOR(LN(?) / LN(?)) + 1)
            )
        `;

        // Test base tier (ages 1-5): decay from 1.0 to 0.2 (steps of 0.2)
        // decayFactor = (6 - bucket) / 5
        const [age1] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 1, contextLength, 1, contextLength, contextLength, 1, contextLength]
        );
        this.assert(this.isEqual(age1[0].reward_factor, 1.5), 'Age 1 should get full reward (decay=1.0 = 5/5)', age1[0].reward_factor, 1.5);

        const [age2] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 2, contextLength, 2, contextLength, contextLength, 2, contextLength]
        );
        this.assert(this.isEqual(age2[0].reward_factor, 1.4), 'Age 2 should get decay=0.8 (4/5: 1.0 + 0.5*0.8 = 1.4)', age2[0].reward_factor, 1.4);

        const [age3] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 3, contextLength, 3, contextLength, contextLength, 3, contextLength]
        );
        this.assert(this.isEqual(age3[0].reward_factor, 1.3), 'Age 3 should get decay=0.6 (3/5: 1.0 + 0.5*0.6 = 1.3)', age3[0].reward_factor, 1.3);

        const [age5] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 5, contextLength, 5, contextLength, contextLength, 5, contextLength]
        );
        this.assert(this.isEqual(age5[0].reward_factor, 1.1), 'Age 5 should get decay=0.2 (1/5: 1.0 + 0.5*0.2 = 1.1)', age5[0].reward_factor, 1.1);

        // Test first exponential tier (ages 6-25): decay from 0.16 to 0.04 (steps of 0.04)
        // Buckets: 6-10 (bucket 2), 11-15 (bucket 3), 16-20 (bucket 4), 21-25 (bucket 5)
        // decayFactor = (6 - bucket) / 25
        const [age6] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 6, contextLength, 6, contextLength, contextLength, 6, contextLength]
        );
        this.assert(this.isEqual(age6[0].reward_factor, 1.08), 'Age 6 should get decay=0.16 (4/25: bucket 6-10)', age6[0].reward_factor, 1.08);

        const [age10] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 10, contextLength, 10, contextLength, contextLength, 10, contextLength]
        );
        this.assert(this.isEqual(age10[0].reward_factor, 1.08), 'Age 10 should get decay=0.16 (same bucket as 6-10)', age10[0].reward_factor, 1.08);

        const [age15] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 15, contextLength, 15, contextLength, contextLength, 15, contextLength]
        );
        this.assert(this.isEqual(age15[0].reward_factor, 1.06), 'Age 15 should get decay=0.12 (3/25: bucket 11-15)', age15[0].reward_factor, 1.06);

        const [age25] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 25, contextLength, 25, contextLength, contextLength, 25, contextLength]
        );
        this.assert(this.isEqual(age25[0].reward_factor, 1.02), 'Age 25 should get decay=0.04 (1/25: bucket 21-25)', age25[0].reward_factor, 1.02);

        // Test second exponential tier (ages 26-125): decay from 0.032 to 0.008
        // Buckets: 26-50 (bucket 2), 51-75 (bucket 3), 76-100 (bucket 4), 101-125 (bucket 5)
        // decayFactor = (6 - bucket) / 125
        const [age26] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 26, contextLength, 26, contextLength, contextLength, 26, contextLength]
        );
        this.assert(this.isEqual(age26[0].reward_factor, 1.016), 'Age 26 should get decay=0.032 (4/125: bucket 26-50)', age26[0].reward_factor, 1.016);

        const [age75] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 75, contextLength, 75, contextLength, contextLength, 75, contextLength]
        );
        this.assert(this.isEqual(age75[0].reward_factor, 1.012), 'Age 75 should get decay=0.024 (3/125: bucket 51-75)', age75[0].reward_factor, 1.012);

        const [age125] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.5, contextLength, 125, contextLength, 125, contextLength, contextLength, 125, contextLength]
        );
        this.assert(this.isEqual(age125[0].reward_factor, 1.004), 'Age 125 should get decay=0.008 (1/125: bucket 101-125)', age125[0].reward_factor, 1.004);

        // Test with negative reward
        const [negAge1] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [0.5, contextLength, 1, contextLength, 1, contextLength, contextLength, 1, contextLength]
        );
        this.assert(this.isEqual(negAge1[0].reward_factor, 0.5), 'Age 1 with negative reward should get full penalty (0.5)', negAge1[0].reward_factor, 0.5);

        const [negAge15] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [0.5, contextLength, 15, contextLength, 15, contextLength, contextLength, 15, contextLength]
        );
        this.assert(this.isEqual(negAge15[0].reward_factor, 0.94), 'Age 15 with negative reward should get decay=0.12 (1.0 + (-0.5)*0.12 = 0.94)', negAge15[0].reward_factor, 0.94);

        // Test neutral reward
        const [neutral] = await this.conn.query(
            `SELECT ${rewardFormula} as reward_factor`,
            [1.0, contextLength, 50, contextLength, 50, contextLength, contextLength, 50, contextLength]
        );
        this.assert(this.isEqual(neutral[0].reward_factor, 1.0), 'Neutral reward should always be 1.0 regardless of age', neutral[0].reward_factor, 1.0);

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
