/**
 * Test Named Parameters in mysql2
 * Quick test to verify named parameters work with our database connection
 */

import db from '../db/db.js';

async function testNamedParameters() {
    console.log('Testing named parameters with mysql2...\n');
    
    const conn = await db();
    
    try {
        await conn.query('USE machine_intelligence');
        
        // Test 1: Simple named parameter
        console.log('Test 1: Simple named parameter');
        const [rows1] = await conn.execute(
            'SELECT :value as result',
            { value: 42 }
        );
        console.log('Expected: 42, Got:', rows1[0].result);
        console.log(rows1[0].result === 42 ? '✓ PASS\n' : '✗ FAIL\n');

        // Test 2: Multiple different named parameters
        console.log('Test 2: Multiple different named parameters');
        const [rows2] = await conn.execute(
            'SELECT :a + :b as sum',
            { a: 10, b: 20 }
        );
        console.log('Expected: 30, Got:', rows2[0].sum);
        console.log(rows2[0].sum === 30 ? '✓ PASS\n' : '✗ FAIL\n');

        // Test 3: Same named parameter used multiple times (the key test!)
        console.log('Test 3: Same named parameter used multiple times');
        const [rows3] = await conn.execute(
            'SELECT :N * 2 as doubled, :N * 3 as triple, POW(:N, 2) as squared',
            { N: 5 }
        );
        console.log('Expected: doubled=10, triple=15, squared=25');
        console.log('Got: doubled=' + rows3[0].doubled + ', triple=' + rows3[0].triple + ', squared=' + rows3[0].squared);
        console.log(rows3[0].doubled === 10 && rows3[0].triple === 15 && rows3[0].squared === 25 ? '✓ PASS\n' : '✗ FAIL\n');

        // Test 4: Complex formula similar to our reward formula
        console.log('Test 4: Complex formula with repeated named parameter');
        const N = 10;
        const age = 25;
        const [rows4] = await conn.execute(
            `SELECT
                (:N + 1 - CEIL(:age / POW(:N, FLOOR(LOG(:N, :age)))))
                / POW(:N, FLOOR(LOG(:N, :age)) + 1) as decay_factor`,
            { N, age }
        );
        console.log('Decay factor for age=' + age + ', N=' + N + ':', rows4[0].decay_factor);
        console.log('Expected: 0.08 (bucket 21-30)');
        console.log(Math.abs(rows4[0].decay_factor - 0.08) < 0.0001 ? '✓ PASS\n' : '✗ FAIL\n');

        // Test 5: Full reward formula
        console.log('Test 5: Full reward formula with named parameters');
        const globalReward = 1.5;
        const [rows5] = await conn.execute(
            `SELECT
                1.0 + (:globalReward - 1.0) * (
                    (:N + 1 - CEIL(:age / POW(:N, FLOOR(LOG(:N, :age)))))
                    / POW(:N, FLOOR(LOG(:N, :age)) + 1)
                ) as reward_factor`,
            { globalReward, N, age }
        );
        console.log('Reward factor for globalReward=' + globalReward + ', age=' + age + ', N=' + N + ':', rows5[0].reward_factor);
        console.log('Expected: 1.04 (1.0 + 0.5 * 0.08)');
        console.log(Math.abs(rows5[0].reward_factor - 1.04) < 0.0001 ? '✓ PASS\n' : '✗ FAIL\n');
        
        console.log('All tests completed!');
        
    } catch (error) {
        console.error('Test failed with error:', error);
        process.exit(1);
    } finally {
        await conn.end();
        process.exit(0);
    }
}

testNamedParameters();

