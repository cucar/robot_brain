/**
 * In-Memory Brain Algorithms
 *
 * Pure JavaScript implementations of brain operations that were previously SQL queries.
 * These are optimized for speed using indexed data structures.
 */

/**
 * Detect peaks in active connections at a specific level
 *
 * Algorithm:
 * 1. Get all active connections at level (O(1) hash lookup)
 * 2. Calculate weighted strengths for each target neuron (O(N) single pass)
 * 3. Calculate average strength across all target neurons (O(N))
 * 4. Identify peaks: targets stronger than minPeakStrength AND stronger than avg * minPeakRatio (O(N))
 *
 * @param {ActiveConnectionStore} activeConnectionStore
 * @param {ConnectionStore} connectionStore
 * @param {number} level - The level to detect peaks for
 * @param {number} peakTimeDecayFactor - Decay factor for connection strength based on distance (e.g., 0.9)
 * @param {number} minPeakStrength - Minimum absolute strength to be considered a peak
 * @param {number} minPeakRatio - Minimum ratio vs average strength to be considered a peak
 * @returns {Map<peak_neuron_id, Set<connection_id>>} - Map of peak neurons to their connection IDs
 */
export function detectPeaks(activeConnectionStore, connectionStore, level, peakTimeDecayFactor, minPeakStrength, minPeakRatio) {
	const startTime = performance.now();

	// Step 1: Get all active connections at this level, age 0 - O(1)
	const activeConns = activeConnectionStore.getByLevelAge(level, 0);

	if (activeConns.length === 0) {
		console.log(`Found 0 peaks at level ${level} (no active connections)`);
		return new Map();
	}

	// Step 2: Calculate weighted strengths for each target neuron - O(N) single pass
	// Map<to_neuron_id, {total_strength, connections: Array<{connection_id, from_neuron_id, strength}>}>
	const targetData = new Map();

	for (const ac of activeConns) {
		const conn = connectionStore.get(ac.connection_id);
		if (!conn || conn.strength <= 0) continue;

		// Calculate weighted strength: strength * POW(decayFactor, distance)
		const weightedStrength = conn.strength * Math.pow(peakTimeDecayFactor, conn.distance);

		// Initialize target data if needed
		if (!targetData.has(conn.to_neuron_id)) {
			targetData.set(conn.to_neuron_id, {
				total_strength: 0,
				connections: []
			});
		}

		const target = targetData.get(conn.to_neuron_id);
		target.total_strength += weightedStrength;
		target.connections.push({
			connection_id: conn.id,
			from_neuron_id: conn.from_neuron_id,
			strength: weightedStrength
		});
	}

	// Step 3: Calculate average strength across all target neurons - O(N)
	let totalStrength = 0;
	for (const [, data] of targetData) {
		totalStrength += data.total_strength;
	}
	const avgStrength = targetData.size > 0 ? totalStrength / targetData.size : 0;

	// Step 4: Identify peaks - O(N)
	const peaks = new Map(); // Map<peak_neuron_id, Set<connection_id>>

	for (const [toNeuron, data] of targetData) {
		// Peak criteria: strength >= minPeakStrength AND strength > avg * minPeakRatio
		if (data.total_strength >= minPeakStrength &&
		    data.total_strength > avgStrength * minPeakRatio) {

			// This is a peak! Store all its connection IDs
			const connectionIds = new Set(data.connections.map(c => c.connection_id));
			peaks.set(toNeuron, connectionIds);
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Found ${peaks.size} peaks at level ${level} (${elapsed.toFixed(2)}ms, ${activeConns.length} active connections, ${targetData.size} targets, avg strength ${avgStrength.toFixed(2)})`);

	return peaks;
}

/**
 * Infer connections for next frame using peak detection
 *
 * Algorithm:
 * 1. Get all active neurons at this level (O(1))
 * 2. For each active neuron, get connections at distance = age + 1 (O(N) with index)
 * 3. Group by target neuron and calculate total strength (O(N))
 * 4. Calculate average strength across all candidates (O(N))
 * 5. Identify peaks: candidates stronger than minPeakStrength AND stronger than avg * minPeakRatio (O(N))
 *
 * @param {ActiveNeuronStore} activeNeuronStore
 * @param {ConnectionStore} connectionStore
 * @param {number} level
 * @param {number} peakTimeDecayFactor
 * @param {number} minPeakStrength
 * @param {number} minPeakRatio
 * @returns {Map<neuron_id, strength>} - Predicted neurons and their strengths
 */
export function inferConnections(activeNeuronStore, connectionStore, level, peakTimeDecayFactor, minPeakStrength, minPeakRatio) {
	const startTime = performance.now();

	// Step 1: Get all active neurons at this level (any age)
	const activeNeurons = activeNeuronStore.getByLevel(level);

	if (activeNeurons.length === 0) {
		console.log(`Level ${level}: Predicted 0 neurons for next frame (no active neurons)`);
		return new Map();
	}

	// Step 2-3: Build candidate connections and calculate total strengths - O(N)
	// Map<to_neuron_id, {total_strength, connections: Array<{from_neuron_id, connection_id, strength}>}>
	const candidates = new Map();

	for (const an of activeNeurons) {
		const distance = an.age + 1;
		const connections = connectionStore.getByFromDistance(an.neuron_id, distance);

		for (const conn of connections) {
			if (conn.strength <= 0) continue;

			const weightedStrength = conn.strength * Math.pow(peakTimeDecayFactor, conn.distance);

			if (!candidates.has(conn.to_neuron_id)) {
				candidates.set(conn.to_neuron_id, {
					total_strength: 0,
					connections: []
				});
			}

			const candidate = candidates.get(conn.to_neuron_id);
			candidate.total_strength += weightedStrength;
			candidate.connections.push({
				from_neuron_id: conn.from_neuron_id,
				connection_id: conn.id,
				strength: weightedStrength
			});
		}
	}

	// Step 4: Calculate average strength across all candidates - O(N)
	let totalStrength = 0;
	for (const [, data] of candidates) {
		totalStrength += data.total_strength;
	}
	const avgStrength = candidates.size > 0 ? totalStrength / candidates.size : 0;

	// Step 5: Identify peak predictions - O(N)
	const predictions = new Map(); // Map<neuron_id, strength>

	for (const [toNeuron, data] of candidates) {
		// Peak criteria: strength >= minPeakStrength AND strength > avg * minPeakRatio
		if (data.total_strength >= minPeakStrength &&
		    data.total_strength > avgStrength * minPeakRatio) {
			predictions.set(toNeuron, data.total_strength);
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Level ${level}: Predicted ${predictions.size} neurons for next frame (${elapsed.toFixed(2)}ms, ${activeNeurons.length} active neurons, ${candidates.size} candidates, avg strength ${avgStrength.toFixed(2)})`);

	return predictions;
}

/**
 * Match observed patterns to known patterns
 *
 * This replaces the pattern matching SQL query.
 *
 * @param {Map<peak_neuron_id, Set<connection_id>>} observedPatterns - From detectPeaks
 * @param {PatternStore} patternStore
 * @param {PatternPeakStore} patternPeakStore
 * @param {number} mergePatternThreshold - Minimum overlap ratio (e.g., 0.66 for 66%)
 * @returns {Map<peak_neuron_id, Set<pattern_neuron_id>>} - Matched patterns for each peak
 */
export function matchPatterns(observedPatterns, patternStore, patternPeakStore, mergePatternThreshold) {
	const startTime = performance.now();

	const matches = new Map(); // Map<peak_neuron_id, Set<pattern_neuron_id>>

	for (const [peakNeuronId, observedConnectionIds] of observedPatterns) {
		// Get all patterns owned by this peak
		const ownedPatterns = patternPeakStore.getPatterns(peakNeuronId);

		for (const patternNeuronId of ownedPatterns) {
			// Get all connections for this pattern
			const patternConnections = patternStore.getByPattern(patternNeuronId);

			if (patternConnections.length === 0) continue;

			// Calculate overlap: how many of the pattern's connections are in the observed pattern?
			let matchCount = 0;
			for (const pc of patternConnections) {
				if (pc.strength > 0 && observedConnectionIds.has(pc.connection_id)) {
					matchCount++;
				}
			}

			const overlapRatio = matchCount / patternConnections.length;

			// Match if overlap >= threshold
			if (overlapRatio >= mergePatternThreshold) {
				if (!matches.has(peakNeuronId)) {
					matches.set(peakNeuronId, new Set());
				}
				matches.get(peakNeuronId).add(patternNeuronId);
			}
		}
	}

	const totalMatches = Array.from(matches.values()).reduce((sum, set) => sum + set.size, 0);
	const elapsed = performance.now() - startTime;
	console.log(`Matched ${totalMatches} pattern-peak pairs (${elapsed.toFixed(2)}ms)`);

	return matches;
}

/**
 * Merge matched patterns with observed patterns
 *
 * This replaces the pattern merging SQL queries.
 *
 * @param {Map<peak_neuron_id, Set<pattern_neuron_id>>} matchedPatterns - From matchPatterns
 * @param {Map<peak_neuron_id, Set<connection_id>>} observedPatterns - From detectPeaks
 * @param {PatternStore} patternStore
 * @param {number} minConnectionStrength
 * @param {number} maxConnectionStrength
 * @param {number} patternNegativeReinforcement
 */
export function mergeMatchedPatterns(matchedPatterns, observedPatterns, patternStore, minConnectionStrength, maxConnectionStrength, patternNegativeReinforcement) {
	const startTime = performance.now();

	let addedCount = 0;
	let strengthenedCount = 0;
	let weakenedCount = 0;
	let deletedCount = 0;

	for (const [peakNeuronId, patternNeuronIds] of matchedPatterns) {
		const observedConnectionIds = observedPatterns.get(peakNeuronId);
		if (!observedConnectionIds) continue;

		for (const patternNeuronId of patternNeuronIds) {
			// Get all connections for this pattern
			const patternConnections = patternStore.getByPattern(patternNeuronId);

			// Positive reinforcement: strengthen observed connections
			for (const connectionId of observedConnectionIds) {
				const currentStrength = patternStore.get(patternNeuronId, connectionId);

				if (currentStrength === null) {
					// Add new connection to pattern
					patternStore.set(patternNeuronId, connectionId, 1.0);
					addedCount++;
				} else {
					// Strengthen existing connection (clamped)
					const newStrength = Math.max(minConnectionStrength, Math.min(maxConnectionStrength, currentStrength + 1.0));
					patternStore.set(patternNeuronId, connectionId, newStrength);
					strengthenedCount++;
				}
			}

			// Negative reinforcement: weaken unobserved connections
			for (const pc of patternConnections) {
				if (!observedConnectionIds.has(pc.connection_id)) {
					const newStrength = pc.strength - patternNegativeReinforcement;

					if (newStrength <= 0) {
						// Delete connection if strength goes negative
						patternStore.delete(patternNeuronId, pc.connection_id);
						deletedCount++;
					} else {
						// Weaken connection
						patternStore.set(patternNeuronId, pc.connection_id, newStrength);
						weakenedCount++;
					}
				}
			}
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Merged patterns: +${addedCount} added, +${strengthenedCount} strengthened, -${weakenedCount} weakened, -${deletedCount} deleted (${elapsed.toFixed(2)}ms)`);
}

/**
 * Create new patterns for peaks without matches
 *
 * @param {Map<peak_neuron_id, Set<connection_id>>} observedPatterns
 * @param {Map<peak_neuron_id, Set<pattern_neuron_id>>} matchedPatterns
 * @param {NeuronStore} neuronStore
 * @param {PatternStore} patternStore
 * @param {PatternPeakStore} patternPeakStore
 * @returns {Array<pattern_neuron_id>} - IDs of newly created patterns
 */
export function createNewPatterns(observedPatterns, matchedPatterns, neuronStore, patternStore, patternPeakStore) {
	const startTime = performance.now();

	const newPatternIds = [];

	for (const [peakNeuronId, observedConnectionIds] of observedPatterns) {
		// Skip if this peak already has matched patterns
		if (matchedPatterns.has(peakNeuronId)) continue;

		// Create new pattern neuron
		const patternNeuronId = neuronStore.createNeuron();

		// Map pattern to peak
		patternPeakStore.set(patternNeuronId, peakNeuronId);

		// Add all observed connections to the pattern
		for (const connectionId of observedConnectionIds) {
			patternStore.set(patternNeuronId, connectionId, 1.0);
		}

		newPatternIds.push(patternNeuronId);
	}

	const elapsed = performance.now() - startTime;
	console.log(`Created ${newPatternIds.length} new patterns for peaks without matches (${elapsed.toFixed(2)}ms)`);

	return newPatternIds;
}

/**
 * Activate connections between active neurons
 *
 * @param {ActiveNeuronStore} activeNeuronStore
 * @param {ConnectionStore} connectionStore
 * @param {ActiveConnectionStore} activeConnectionStore
 * @param {number} level
 */
export function activateConnections(activeNeuronStore, connectionStore, activeConnectionStore, level) {
	const startTime = performance.now();

	// Get all active neurons at this level
	const activeNeurons = activeNeuronStore.getByLevel(level);

	// Get neurons at age 0 for target matching
	const age0Neurons = new Set(
		activeNeuronStore.getByLevelAge(level, 0).map(n => n.neuron_id)
	);

	let activatedCount = 0;

	// For each active neuron, find connections to age-0 neurons
	for (const fromNeuron of activeNeurons) {
		const distance = fromNeuron.age;
		const connections = connectionStore.getByFromDistance(fromNeuron.neuron_id, distance);

		for (const conn of connections) {
			// Check if target is active at age 0 and same level
			if (!age0Neurons.has(conn.to_neuron_id)) continue;

			// Skip self-connections at same age
			if (conn.to_neuron_id === fromNeuron.neuron_id && fromNeuron.age === 0) continue;

			// Skip weak connections
			if (conn.strength <= 0) continue;

			// Activate this connection
			activeConnectionStore.add(conn.id, conn.from_neuron_id, conn.to_neuron_id, level, 0);
			activatedCount++;
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Activated ${activatedCount} connections at level ${level} (${elapsed.toFixed(2)}ms)`);
}

/**
 * Reinforce connections between co-active neurons
 *
 * @param {ActiveNeuronStore} activeNeuronStore
 * @param {ConnectionStore} connectionStore
 * @param {number} level
 * @param {number} minConnectionStrength
 * @param {number} maxConnectionStrength
 */
export function reinforceConnections(activeNeuronStore, connectionStore, level, minConnectionStrength, maxConnectionStrength) {
	const startTime = performance.now();

	// Get all active neurons at this level
	const activeNeurons = activeNeuronStore.getByLevel(level);

	// Get neurons at age 0 for target matching
	const age0Neurons = new Set(
		activeNeuronStore.getByLevelAge(level, 0).map(n => n.neuron_id)
	);

	let reinforcedCount = 0;
	let createdCount = 0;

	// For each active neuron, reinforce connections to age-0 neurons
	for (const fromNeuron of activeNeurons) {
		const distance = fromNeuron.age;

		for (const toNeuronId of age0Neurons) {
			// Skip self-connections at same age
			if (toNeuronId === fromNeuron.neuron_id && fromNeuron.age === 0) continue;

			// Find or create connection
			const existing = connectionStore.findByFromToDistance(fromNeuron.neuron_id, toNeuronId, distance);

			if (existing) {
				// Strengthen existing connection (clamped)
				const newStrength = Math.max(minConnectionStrength, Math.min(maxConnectionStrength, existing.strength + 1.0));
				connectionStore.set(fromNeuron.neuron_id, toNeuronId, distance, newStrength);
				reinforcedCount++;
			} else {
				// Create new connection
				connectionStore.set(fromNeuron.neuron_id, toNeuronId, distance, 1.0);
				createdCount++;
			}
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Reinforced ${reinforcedCount} connections, created ${createdCount} new connections at level ${level} (${elapsed.toFixed(2)}ms)`);
}

/**
 * Apply rewards to connections and patterns
 *
 * @param {ActiveConnectionStore} activeConnectionStore
 * @param {ConnectionStore} connectionStore
 * @param {PatternStore} patternStore
 * @param {number} rewardAdjustment - Additive adjustment (e.g., +0.5 for positive, -0.5 for negative)
 * @param {number} rewardTimeDecayFactor - Decay factor for older connections (e.g., 0.9)
 * @param {number} minConnectionStrength
 * @param {number} maxConnectionStrength
 */
export function applyRewards(activeConnectionStore, connectionStore, patternStore, rewardAdjustment, rewardTimeDecayFactor, minConnectionStrength, maxConnectionStrength) {
	const startTime = performance.now();

	if (Math.abs(rewardAdjustment) < 0.001) {
		console.log('Skipping reward application (neutral reward)');
		return;
	}

	let connectionCount = 0;
	let patternCount = 0;

	// Get all active connections (all levels, all ages)
	const allActiveConns = activeConnectionStore.byKey.values();

	for (const ac of allActiveConns) {
		const conn = connectionStore.get(ac.connection_id);
		if (!conn) continue;

		// Calculate time-decayed reward: adjustment * POW(decayFactor, age)
		const ageDecay = Math.pow(rewardTimeDecayFactor, ac.age);
		const adjustedReward = rewardAdjustment * ageDecay;

		// Apply reward to connection strength (additive, clamped)
		const newStrength = Math.max(minConnectionStrength, Math.min(maxConnectionStrength, conn.strength + adjustedReward));
		connectionStore.set(conn.from_neuron_id, conn.to_neuron_id, conn.distance, newStrength);
		connectionCount++;

		// Apply reward to patterns containing this connection
		const patternsWithConn = patternStore.getByConnection(ac.connection_id);
		for (const patternNeuronId of patternsWithConn) {
			const patternStrength = patternStore.get(patternNeuronId, ac.connection_id);
			if (patternStrength !== null) {
				const newPatternStrength = Math.max(minConnectionStrength, Math.min(maxConnectionStrength, patternStrength + adjustedReward));
				patternStore.set(patternNeuronId, ac.connection_id, newPatternStrength);
				patternCount++;
			}
		}
	}

	const elapsed = performance.now() - startTime;
	console.log(`Applied reward adjustment ${rewardAdjustment >= 0 ? '+' : ''}${rewardAdjustment.toFixed(3)}: ${connectionCount} connections, ${patternCount} pattern entries (${elapsed.toFixed(2)}ms)`);
}

