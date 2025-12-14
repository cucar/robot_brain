/**
 * Test the in-memory brain implementation
 */

import BrainMemory from './brain-memory.js';

console.log('Testing In-Memory Brain Implementation\n');
console.log('='.repeat(80));

const brain = new BrainMemory();

// Initialize
await brain.init();

// Setup test dimensions manually (simulating a simple channel)
console.log('\nSetting up test dimensions...');
await brain.conn.query("INSERT IGNORE INTO dimensions (name, channel, type) VALUES ('price_change', 'test', 'event')");
await brain.conn.query("INSERT IGNORE INTO dimensions (name, channel, type) VALUES ('volume_change', 'test', 'event')");
await brain.conn.query("INSERT IGNORE INTO dimensions (name, channel, type) VALUES ('action', 'test', 'action')");

console.log('\n1. Testing basic frame processing...');

// Frame 1: Simple input pattern
const inputs1 = [
	{ price_change: 1 },  // 1 = up
	{ volume_change: 1 }  // 1 = high
];
const outputs1 = [
	{ action: 1 }  // 1 = buy
];

await brain.processFrame(inputs1, outputs1);

console.log('\nBrain stats after frame 1:');
brain.printStats();

// Frame 2: Same pattern (should reinforce)
console.log('\n2. Testing pattern reinforcement...');
await brain.processFrame(inputs1, outputs1);

console.log('\nBrain stats after frame 2:');
brain.printStats();

// Frame 3: Different pattern
console.log('\n3. Testing new pattern...');
const inputs2 = [
	{ price_change: -1 },  // -1 = down
	{ volume_change: -1 }  // -1 = low
];
const outputs2 = [
	{ action: -1 }  // -1 = sell
];

await brain.processFrame(inputs2, outputs2);

console.log('\nBrain stats after frame 3:');
brain.printStats();

// Frame 4-10: Repeat patterns to build connections
console.log('\n4. Testing pattern learning (frames 4-10)...');
for (let i = 4; i <= 10; i++) {
	const usePattern1 = i % 2 === 0;
	await brain.processFrame(
		usePattern1 ? inputs1 : inputs2,
		usePattern1 ? outputs1 : outputs2
	);
}

console.log('\nBrain stats after 10 frames:');
brain.printStats();

// Test reward
console.log('\n5. Testing reward application...');
await brain.processFrame(inputs1, outputs1);
brain.applyReward(0.5); // Positive reward

console.log('\nBrain stats after reward:');
brain.printStats();

// Test forget cycle
console.log('\n6. Testing forget cycle...');
brain.forgetCounter = brain.forgetCycles - 1; // Force forget cycle on next frame
await brain.processFrame(inputs1, outputs1);

console.log('\nBrain stats after forget cycle:');
brain.printStats();

// Performance test
console.log('\n7. Performance test (100 frames)...');
const perfStart = performance.now();

for (let i = 0; i < 100; i++) {
	const usePattern1 = i % 2 === 0;
	await brain.processFrame(
		usePattern1 ? inputs1 : inputs2,
		usePattern1 ? outputs1 : outputs2
	);
}

const perfElapsed = performance.now() - perfStart;
const avgFrameTime = perfElapsed / 100;

console.log(`\nProcessed 100 frames in ${perfElapsed.toFixed(2)}ms`);
console.log(`Average frame time: ${avgFrameTime.toFixed(2)}ms`);
console.log(`Frames per second: ${(1000 / avgFrameTime).toFixed(2)}`);

console.log('\nFinal brain stats:');
brain.printStats();

// Close
await brain.close();

console.log('\n' + '='.repeat(80));
console.log('In-Memory Brain Test Complete! ✓');
console.log('='.repeat(80));

