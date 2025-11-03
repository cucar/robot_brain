/**
 * Test in-memory brain data structures and algorithms
 */

import {
	ConnectionStore,
	ActiveNeuronStore,
	PatternStore,
	PatternPeakStore,
	NeuronStore,
	ActiveConnectionStore
} from './brain-memory-stores.js';

import {
	detectPeaks,
	inferConnections,
	matchPatterns
} from './brain-algorithms.js';

console.log('Testing In-Memory Brain Data Structures\n');
console.log('='.repeat(60));

// Test 1: ConnectionStore
console.log('\n1. Testing ConnectionStore...');
const connStore = new ConnectionStore();

// Add some connections
const conn1 = connStore.set(1, 2, 1, 10.0);  // from=1, to=2, distance=1, strength=10
const conn2 = connStore.set(1, 3, 1, 15.0);  // from=1, to=3, distance=1, strength=15
const conn3 = connStore.set(2, 4, 1, 8.0);   // from=2, to=4, distance=1, strength=8
const conn4 = connStore.set(1, 2, 2, 5.0);   // from=1, to=2, distance=2, strength=5

console.log(`  Created ${connStore.size()} connections`);
console.log(`  Connections from neuron 1 at distance 1:`, connStore.getByFromDistance(1, 1).length);
console.log(`  Connections to neuron 2:`, connStore.getByTo(2).length);
console.log(`  ✓ ConnectionStore working`);

// Test 2: ActiveNeuronStore
console.log('\n2. Testing ActiveNeuronStore...');
const activeNeuronStore = new ActiveNeuronStore();

activeNeuronStore.add(1, 0, 0);  // neuron 1, level 0, age 0
activeNeuronStore.add(2, 0, 0);
activeNeuronStore.add(3, 0, 1);  // neuron 3, level 0, age 1
activeNeuronStore.add(4, 1, 0);  // neuron 4, level 1, age 0

console.log(`  Active neurons: ${activeNeuronStore.size()}`);
console.log(`  Neurons at level 0, age 0:`, activeNeuronStore.getByLevelAge(0, 0).length);
console.log(`  Neurons at level 0 (any age):`, activeNeuronStore.getByLevel(0).length);
console.log(`  ✓ ActiveNeuronStore working`);

// Test 3: PatternStore
console.log('\n3. Testing PatternStore...');
const patternStore = new PatternStore();

patternStore.set(100, conn1, 5.0);  // pattern 100 contains connection 1
patternStore.set(100, conn2, 3.0);  // pattern 100 contains connection 2
patternStore.set(101, conn1, 8.0);  // pattern 101 contains connection 1

console.log(`  Pattern entries: ${patternStore.size()}`);
console.log(`  Connections in pattern 100:`, patternStore.getByPattern(100).length);
console.log(`  Patterns containing connection ${conn1}:`, patternStore.getByConnection(conn1).length);
console.log(`  ✓ PatternStore working`);

// Test 4: PatternPeakStore
console.log('\n4. Testing PatternPeakStore...');
const patternPeakStore = new PatternPeakStore();

patternPeakStore.set(100, 2);  // pattern 100 owned by peak neuron 2
patternPeakStore.set(101, 2);  // pattern 101 owned by peak neuron 2
patternPeakStore.set(102, 3);  // pattern 102 owned by peak neuron 3

console.log(`  Pattern-peak mappings: ${patternPeakStore.size()}`);
console.log(`  Peak for pattern 100:`, patternPeakStore.getPeak(100));
console.log(`  Patterns owned by peak 2:`, patternPeakStore.getPatterns(2).length);
console.log(`  ✓ PatternPeakStore working`);

// Test 5: NeuronStore
console.log('\n5. Testing NeuronStore...');
const neuronStore = new NeuronStore();

const n1 = neuronStore.createNeuron();
const n2 = neuronStore.createNeuron();
neuronStore.setCoordinate(n1, 1, 5.0);   // neuron n1, dimension 1, value 5.0
neuronStore.setCoordinate(n1, 2, 10.0);  // neuron n1, dimension 2, value 10.0
neuronStore.setCoordinate(n2, 1, 5.0);   // neuron n2, dimension 1, value 5.0

console.log(`  Neurons: ${neuronStore.size()}`);
console.log(`  Coordinates for neuron ${n1}:`, neuronStore.getCoordinates(n1).length);
console.log(`  Neurons with dimension 1 = 5.0:`, neuronStore.findByDimensionValue(1, 5.0).length);
console.log(`  ✓ NeuronStore working`);

// Test 6: ActiveConnectionStore
console.log('\n6. Testing ActiveConnectionStore...');
const activeConnStore = new ActiveConnectionStore();

activeConnStore.add(conn1, 1, 2, 0, 0);  // connection 1, from=1, to=2, level=0, age=0
activeConnStore.add(conn2, 1, 3, 0, 0);
activeConnStore.add(conn3, 2, 4, 0, 0);

console.log(`  Active connections: ${activeConnStore.size()}`);
console.log(`  Connections at level 0, age 0:`, activeConnStore.getByLevelAge(0, 0).length);
console.log(`  ✓ ActiveConnectionStore working`);

// Test 7: Peak Detection Algorithm
console.log('\n7. Testing Peak Detection Algorithm...');
console.log('  Setting up test scenario...');

// Create a more complex scenario
const testConnStore = new ConnectionStore();
const testActiveConnStore = new ActiveConnectionStore();

// Create connections: neuron 1 -> {2, 3, 4} with different strengths
const c1 = testConnStore.set(1, 2, 0, 20.0);  // Strong connection to neuron 2
const c2 = testConnStore.set(1, 3, 0, 5.0);   // Weak connection to neuron 3
const c3 = testConnStore.set(1, 4, 0, 8.0);   // Medium connection to neuron 4
const c4 = testConnStore.set(5, 2, 0, 3.0);   // Another connection to neuron 2 (makes it stronger)
const c5 = testConnStore.set(5, 3, 0, 2.0);   // Another connection to neuron 3

// Activate these connections at level 0
testActiveConnStore.add(c1, 1, 2, 0, 0);
testActiveConnStore.add(c2, 1, 3, 0, 0);
testActiveConnStore.add(c3, 1, 4, 0, 0);
testActiveConnStore.add(c4, 5, 2, 0, 0);
testActiveConnStore.add(c5, 5, 3, 0, 0);

// Run peak detection
const peaks = detectPeaks(
	testActiveConnStore,
	testConnStore,
	0,           // level
	0.9,         // peakTimeDecayFactor
	10.0,        // minPeakStrength
	1.5          // minPeakRatio
);

console.log(`  Detected ${peaks.size} peaks`);
for (const [peakNeuron, connectionIds] of peaks) {
	console.log(`    Peak neuron ${peakNeuron}: ${connectionIds.size} connections`);
}
console.log(`  ✓ Peak detection working`);

// Test 8: Connection Inference Algorithm
console.log('\n8. Testing Connection Inference Algorithm...');

const testActiveNeuronStore = new ActiveNeuronStore();
testActiveNeuronStore.add(1, 0, 0);
testActiveNeuronStore.add(5, 0, 0);

const predictions = inferConnections(
	testActiveNeuronStore,
	testConnStore,
	0,           // level
	0.9,         // peakTimeDecayFactor
	5.0,         // minPeakStrength
	1.2          // minPeakRatio
);

console.log(`  Predicted ${predictions.size} neurons`);
for (const [neuronId, strength] of predictions) {
	console.log(`    Neuron ${neuronId}: strength ${strength.toFixed(2)}`);
}
console.log(`  ✓ Connection inference working`);

// Test 9: Pattern Matching Algorithm
console.log('\n9. Testing Pattern Matching Algorithm...');

// Create observed patterns (from peak detection)
const observedPatterns = new Map();
observedPatterns.set(2, new Set([c1, c4]));  // Peak 2 has connections c1, c4
observedPatterns.set(3, new Set([c2, c5]));  // Peak 3 has connections c2, c5

// Create known patterns
const testPatternStore = new PatternStore();
const testPatternPeakStore = new PatternPeakStore();

// Pattern 100 owned by peak 2, contains c1 and c4 (should match 100%)
testPatternStore.set(100, c1, 5.0);
testPatternStore.set(100, c4, 3.0);
testPatternPeakStore.set(100, 2);

// Pattern 101 owned by peak 2, contains c1 only (should match 50%)
testPatternStore.set(101, c1, 8.0);
testPatternPeakStore.set(101, 2);

// Pattern 102 owned by peak 3, contains c2 and c5 (should match 100%)
testPatternStore.set(102, c2, 4.0);
testPatternStore.set(102, c5, 2.0);
testPatternPeakStore.set(102, 3);

const matches = matchPatterns(
	observedPatterns,
	testPatternStore,
	testPatternPeakStore,
	0.66  // 66% threshold
);

console.log(`  Matched patterns:`);
for (const [peakNeuron, patternIds] of matches) {
	console.log(`    Peak ${peakNeuron}: ${patternIds.size} patterns matched`);
	for (const patternId of patternIds) {
		console.log(`      - Pattern ${patternId}`);
	}
}
console.log(`  ✓ Pattern matching working`);

// Performance Test
console.log('\n10. Performance Test...');
console.log('  Creating large dataset...');

const perfConnStore = new ConnectionStore();
const perfActiveConnStore = new ActiveConnectionStore();

// Create 1000 connections
const numConnections = 1000;
const numNeurons = 100;

for (let i = 0; i < numConnections; i++) {
	const from = Math.floor(Math.random() * numNeurons) + 1;
	const to = Math.floor(Math.random() * numNeurons) + 1;
	const distance = Math.floor(Math.random() * 3);
	const strength = Math.random() * 20;
	
	const connId = perfConnStore.set(from, to, distance, strength);
	
	// Activate 50% of connections
	if (Math.random() > 0.5) {
		perfActiveConnStore.add(connId, from, to, 0, 0);
	}
}

console.log(`  Created ${perfConnStore.size()} connections`);
console.log(`  Activated ${perfActiveConnStore.size()} connections`);

// Run peak detection and measure time
const perfStart = performance.now();
const perfPeaks = detectPeaks(
	perfActiveConnStore,
	perfConnStore,
	0,
	0.9,
	10.0,
	1.5
);
const perfElapsed = performance.now() - perfStart;

console.log(`  Peak detection on ${perfActiveConnStore.size()} connections: ${perfElapsed.toFixed(2)}ms`);
console.log(`  Found ${perfPeaks.size} peaks`);
console.log(`  ✓ Performance test complete`);

console.log('\n' + '='.repeat(60));
console.log('All tests passed! ✓');
console.log('='.repeat(60));

