/**
 * In-Memory Brain Data Structures with Multi-Index Support
 *
 * This replaces MySQL MEMORY tables with JavaScript data structures
 * that maintain multiple indexes for fast lookups.
 */

/**
 * Multi-indexed collection for connections
 * Supports O(1) lookups by:
 * - connection_id
 * - from_neuron_id + distance
 * - to_neuron_id
 */
class ConnectionStore {
	constructor() {
		// Primary storage: Map<connection_id, {id, from_neuron_id, to_neuron_id, distance, strength}>
		this.byId = new Map();

		// Index: Map<from_neuron_id, Map<distance, Set<connection_id>>>
		this.byFromDistance = new Map();

		// Index: Map<to_neuron_id, Set<connection_id>>
		this.byTo = new Map();

		// Auto-increment ID
		this.nextId = 1;
	}

	/**
	 * Add or update a connection
	 */
	set(fromNeuronId, toNeuronId, distance, strength) {
		// Find existing connection
		const existing = this.findByFromToDistance(fromNeuronId, toNeuronId, distance);

		if (existing) {
			// Update existing
			existing.strength = strength;
			return existing.id;
		}

		// Create new connection
		const id = this.nextId++;
		const conn = { id, from_neuron_id: fromNeuronId, to_neuron_id: toNeuronId, distance, strength };

		// Add to primary storage
		this.byId.set(id, conn);

		// Add to from+distance index
		if (!this.byFromDistance.has(fromNeuronId)) {
			this.byFromDistance.set(fromNeuronId, new Map());
		}
		const distanceMap = this.byFromDistance.get(fromNeuronId);
		if (!distanceMap.has(distance)) {
			distanceMap.set(distance, new Set());
		}
		distanceMap.get(distance).add(id);

		// Add to to_neuron index
		if (!this.byTo.has(toNeuronId)) {
			this.byTo.set(toNeuronId, new Set());
		}
		this.byTo.get(toNeuronId).add(id);

		return id;
	}

	/**
	 * Get connection by ID - O(1)
	 */
	get(connectionId) {
		return this.byId.get(connectionId);
	}

	/**
	 * Find connection by from+to+distance - O(1) average
	 */
	findByFromToDistance(fromNeuronId, toNeuronId, distance) {
		const distanceMap = this.byFromDistance.get(fromNeuronId);
		if (!distanceMap) return null;

		const connectionIds = distanceMap.get(distance);
		if (!connectionIds) return null;

		// Check each connection for matching to_neuron
		for (const id of connectionIds) {
			const conn = this.byId.get(id);
			if (conn.to_neuron_id === toNeuronId) return conn;
		}
		return null;
	}

	/**
	 * Get all connections to a neuron - O(1)
	 */
	findByTo(toNeuronId) {
		const connectionIds = this.byTo.get(toNeuronId);
		if (!connectionIds) return [];

		return Array.from(connectionIds).map(id => this.byId.get(id));
	}

	/**
	 * Get all connections from a neuron at a specific distance - O(1)
	 * Returns: Array<{id, from_neuron_id, to_neuron_id, distance, strength}>
	 */
	getByFromDistance(fromNeuronId, distance) {
		const distanceMap = this.byFromDistance.get(fromNeuronId);
		if (!distanceMap) return [];

		const connectionIds = distanceMap.get(distance);
		if (!connectionIds) return [];

		return Array.from(connectionIds).map(id => this.byId.get(id));
	}

	/**
	 * Get all connections TO a neuron - O(1)
	 */
	getByTo(toNeuronId) {
		const connectionIds = this.byTo.get(toNeuronId);
		if (!connectionIds) return [];
		return Array.from(connectionIds).map(id => this.byId.get(id));
	}

	/**
	 * Delete connection and update indexes
	 */
	delete(connectionId) {
		const conn = this.byId.get(connectionId);
		if (!conn) return false;

		// Remove from primary storage
		this.byId.delete(connectionId);

		// Remove from from+distance index
		const distanceMap = this.byFromDistance.get(conn.from_neuron_id);
		if (distanceMap) {
			const connectionIds = distanceMap.get(conn.distance);
			if (connectionIds) {
				connectionIds.delete(connectionId);
				if (connectionIds.size === 0) distanceMap.delete(conn.distance);
			}
			if (distanceMap.size === 0) this.byFromDistance.delete(conn.from_neuron_id);
		}

		// Remove from to_neuron index
		const toSet = this.byTo.get(conn.to_neuron_id);
		if (toSet) {
			toSet.delete(connectionId);
			if (toSet.size === 0) this.byTo.delete(conn.to_neuron_id);
		}

		return true;
	}

	/**
	 * Get all connections (for iteration)
	 */
	getAll() {
		return Array.from(this.byId.values());
	}

	/**
	 * Clear all connections
	 */
	clear() {
		this.byId.clear();
		this.byFromDistance.clear();
		this.byTo.clear();
	}

	/**
	 * Get count
	 */
	size() {
		return this.byId.size;
	}
}

/**
 * Multi-indexed collection for active neurons
 * Supports O(1) lookups by:
 * - neuron_id + level + age (primary key)
 * - level + age (for batch queries)
 */
class ActiveNeuronStore {
	constructor() {
		// Primary storage: Map<"neuronId:level:age", {neuron_id, level, age}>
		this.byKey = new Map();

		// Index: Map<level, Map<age, Set<neuron_id>>>
		this.byLevelAge = new Map();
	}

	/**
	 * Create composite key
	 */
	_key(neuronId, level, age) {
		return `${neuronId}:${level}:${age}`;
	}

	/**
	 * Add active neuron
	 */
	add(neuronId, level, age = 0) {
		const key = this._key(neuronId, level, age);

		// Add to primary storage
		this.byKey.set(key, { neuron_id: neuronId, level, age });

		// Add to level+age index
		if (!this.byLevelAge.has(level)) {
			this.byLevelAge.set(level, new Map());
		}
		const ageMap = this.byLevelAge.get(level);
		if (!ageMap.has(age)) {
			ageMap.set(age, new Set());
		}
		ageMap.get(age).add(neuronId);
	}

	/**
	 * Check if neuron is active at specific level and age - O(1)
	 */
	has(neuronId, level, age) {
		return this.byKey.has(this._key(neuronId, level, age));
	}

	/**
	 * Get all neurons at specific level and age - O(1)
	 */
	getByLevelAge(level, age) {
		const ageMap = this.byLevelAge.get(level);
		if (!ageMap) return [];

		const neuronIds = ageMap.get(age);
		if (!neuronIds) return [];

		return Array.from(neuronIds).map(neuronId => ({
			neuron_id: neuronId,
			level,
			age
		}));
	}

	/**
	 * Get all neurons at a specific level (any age) - O(ages)
	 */
	getByLevel(level) {
		const ageMap = this.byLevelAge.get(level);
		if (!ageMap) return [];

		const result = [];
		for (const [age, neuronIds] of ageMap) {
			for (const neuronId of neuronIds) {
				result.push({ neuron_id: neuronId, level, age });
			}
		}
		return result;
	}

	/**
	 * Delete active neuron
	 */
	delete(neuronId, level, age) {
		const key = this._key(neuronId, level, age);

		// Remove from primary storage
		if (!this.byKey.delete(key)) return false;

		// Remove from level+age index
		const ageMap = this.byLevelAge.get(level);
		if (ageMap) {
			const neuronIds = ageMap.get(age);
			if (neuronIds) {
				neuronIds.delete(neuronId);
				if (neuronIds.size === 0) ageMap.delete(age);
			}
			if (ageMap.size === 0) this.byLevelAge.delete(level);
		}

		return true;
	}

	/**
	 * Clear all active neurons
	 */
	clear() {
		this.byKey.clear();
		this.byLevelAge.clear();
	}

	/**
	 * Get all active neurons
	 */
	getAll() {
		return Array.from(this.byKey.values());
	}

	/**
	 * Get count
	 */
	size() {
		return this.byKey.size;
	}
}

/**
 * Multi-indexed collection for patterns
 * Supports O(1) lookups by:
 * - pattern_neuron_id + connection_id (primary key)
 * - pattern_neuron_id (get all connections for a pattern)
 * - connection_id (find patterns containing a connection)
 */
class PatternStore {
	constructor() {
		// Primary storage: Map<"patternId:connectionId", {pattern_neuron_id, connection_id, strength}>
		this.byKey = new Map();

		// Index: Map<pattern_neuron_id, Map<connection_id, strength>>
		this.byPattern = new Map();

		// Index: Map<connection_id, Set<pattern_neuron_id>>
		this.byConnection = new Map();
	}

	/**
	 * Create composite key
	 */
	_key(patternNeuronId, connectionId) {
		return `${patternNeuronId}:${connectionId}`;
	}

	/**
	 * Add or update pattern connection
	 */
	set(patternNeuronId, connectionId, strength) {
		const key = this._key(patternNeuronId, connectionId);

		// Add to primary storage
		this.byKey.set(key, { pattern_neuron_id: patternNeuronId, connection_id: connectionId, strength });

		// Add to pattern index
		if (!this.byPattern.has(patternNeuronId)) {
			this.byPattern.set(patternNeuronId, new Map());
		}
		this.byPattern.get(patternNeuronId).set(connectionId, strength);

		// Add to connection index
		if (!this.byConnection.has(connectionId)) {
			this.byConnection.set(connectionId, new Set());
		}
		this.byConnection.get(connectionId).add(patternNeuronId);
	}

	/**
	 * Get strength for specific pattern+connection - O(1)
	 */
	get(patternNeuronId, connectionId) {
		const entry = this.byKey.get(this._key(patternNeuronId, connectionId));
		return entry ? entry.strength : null;
	}

	/**
	 * Get all connections for a pattern - O(1)
	 * Returns: Array<{pattern_neuron_id, connection_id, strength}>
	 */
	getByPattern(patternNeuronId) {
		const connMap = this.byPattern.get(patternNeuronId);
		if (!connMap) return [];

		return Array.from(connMap.entries()).map(([connectionId, strength]) => ({
			pattern_neuron_id: patternNeuronId,
			connection_id: connectionId,
			strength
		}));
	}

	/**
	 * Get all patterns containing a connection - O(1)
	 * Returns: Array<pattern_neuron_id>
	 */
	getByConnection(connectionId) {
		const patternIds = this.byConnection.get(connectionId);
		if (!patternIds) return [];
		return Array.from(patternIds);
	}

	/**
	 * Delete pattern connection
	 */
	delete(patternNeuronId, connectionId) {
		const key = this._key(patternNeuronId, connectionId);

		// Remove from primary storage
		if (!this.byKey.delete(key)) return false;

		// Remove from pattern index
		const connMap = this.byPattern.get(patternNeuronId);
		if (connMap) {
			connMap.delete(connectionId);
			if (connMap.size === 0) this.byPattern.delete(patternNeuronId);
		}

		// Remove from connection index
		const patternIds = this.byConnection.get(connectionId);
		if (patternIds) {
			patternIds.delete(patternNeuronId);
			if (patternIds.size === 0) this.byConnection.delete(connectionId);
		}

		return true;
	}

	/**
	 * Delete all connections for a pattern
	 */
	deletePattern(patternNeuronId) {
		const connMap = this.byPattern.get(patternNeuronId);
		if (!connMap) return 0;

		let count = 0;
		for (const connectionId of connMap.keys()) {
			if (this.delete(patternNeuronId, connectionId)) count++;
		}
		return count;
	}

	/**
	 * Clear all patterns
	 */
	clear() {
		this.byKey.clear();
		this.byPattern.clear();
		this.byConnection.clear();
	}

	/**
	 * Get all pattern entries
	 */
	getAll() {
		return Array.from(this.byKey.values());
	}

	/**
	 * Get count
	 */
	size() {
		return this.byKey.size;
	}
}

/**
 * Bidirectional mapping for pattern peaks
 * Supports O(1) lookups by:
 * - pattern_neuron_id (primary key) -> peak_neuron_id
 * - peak_neuron_id -> Set<pattern_neuron_id>
 */
class PatternPeakStore {
	constructor() {
		// Map<pattern_neuron_id, peak_neuron_id>
		this.patternToPeak = new Map();

		// Map<peak_neuron_id, Set<pattern_neuron_id>>
		this.peakToPatterns = new Map();
	}

	/**
	 * Add pattern-peak mapping
	 */
	set(patternNeuronId, peakNeuronId) {
		// Remove old mapping if exists
		const oldPeak = this.patternToPeak.get(patternNeuronId);
		if (oldPeak !== undefined) {
			const patterns = this.peakToPatterns.get(oldPeak);
			if (patterns) {
				patterns.delete(patternNeuronId);
				if (patterns.size === 0) this.peakToPatterns.delete(oldPeak);
			}
		}

		// Add new mapping
		this.patternToPeak.set(patternNeuronId, peakNeuronId);

		if (!this.peakToPatterns.has(peakNeuronId)) {
			this.peakToPatterns.set(peakNeuronId, new Set());
		}
		this.peakToPatterns.get(peakNeuronId).add(patternNeuronId);
	}

	/**
	 * Get peak for a pattern - O(1)
	 */
	getPeak(patternNeuronId) {
		return this.patternToPeak.get(patternNeuronId);
	}

	/**
	 * Get all patterns for a peak - O(1)
	 */
	getPatterns(peakNeuronId) {
		const patterns = this.peakToPatterns.get(peakNeuronId);
		return patterns ? Array.from(patterns) : [];
	}

	/**
	 * Delete pattern-peak mapping
	 */
	delete(patternNeuronId) {
		const peakNeuronId = this.patternToPeak.get(patternNeuronId);
		if (peakNeuronId === undefined) return false;

		this.patternToPeak.delete(patternNeuronId);

		const patterns = this.peakToPatterns.get(peakNeuronId);
		if (patterns) {
			patterns.delete(patternNeuronId);
			if (patterns.size === 0) this.peakToPatterns.delete(peakNeuronId);
		}

		return true;
	}

	/**
	 * Clear all mappings
	 */
	clear() {
		this.patternToPeak.clear();
		this.peakToPatterns.clear();
	}

	/**
	 * Get all mappings
	 */
	getAll() {
		return Array.from(this.patternToPeak.entries()).map(([pattern_neuron_id, peak_neuron_id]) => ({
			pattern_neuron_id,
			peak_neuron_id
		}));
	}

	/**
	 * Get count
	 */
	size() {
		return this.patternToPeak.size;
	}
}



/**
 * Store for neurons and their coordinates
 */
class NeuronStore {
	constructor() {
		// Map<neuron_id, {id}>
		this.neurons = new Map();

		// Map<neuron_id, Map<dimension_id, value>>
		this.coordinates = new Map();

		// Reverse index: Map<dimension_id, Map<value, Set<neuron_id>>>
		this.byDimensionValue = new Map();

		// Auto-increment ID
		this.nextId = 1;
	}

	/**
	 * Create a new neuron
	 */
	createNeuron() {
		const id = this.nextId++;
		this.neurons.set(id, { id });
		return id;
	}

	/**
	 * Create multiple neurons in bulk
	 */
	createNeurons(count) {
		const ids = [];
		for (let i = 0; i < count; i++) {
			ids.push(this.createNeuron());
		}
		return ids;
	}

	/**
	 * Set coordinate for a neuron
	 */
	setCoordinate(neuronId, dimensionId, value) {
		// Add to coordinates map
		if (!this.coordinates.has(neuronId)) {
			this.coordinates.set(neuronId, new Map());
		}

		// Remove old value from reverse index if exists
		const oldValue = this.coordinates.get(neuronId).get(dimensionId);
		if (oldValue !== undefined) {
			const dimMap = this.byDimensionValue.get(dimensionId);
			if (dimMap) {
				const neuronSet = dimMap.get(oldValue);
				if (neuronSet) {
					neuronSet.delete(neuronId);
					if (neuronSet.size === 0) dimMap.delete(oldValue);
				}
			}
		}

		// Set new value
		this.coordinates.get(neuronId).set(dimensionId, value);

		// Add to reverse index
		if (!this.byDimensionValue.has(dimensionId)) {
			this.byDimensionValue.set(dimensionId, new Map());
		}
		const dimMap = this.byDimensionValue.get(dimensionId);
		if (!dimMap.has(value)) {
			dimMap.set(value, new Set());
		}
		dimMap.get(value).add(neuronId);
	}

	/**
	 * Get coordinates for a neuron
	 */
	getCoordinates(neuronId) {
		const coords = this.coordinates.get(neuronId);
		if (!coords) return [];

		return Array.from(coords.entries()).map(([dimension_id, val]) => ({
			neuron_id: neuronId,
			dimension_id,
			val
		}));
	}

	/**
	 * Find neurons by dimension and value - O(1)
	 */
	findByDimensionValue(dimensionId, value) {
		const dimMap = this.byDimensionValue.get(dimensionId);
		if (!dimMap) return [];

		const neuronSet = dimMap.get(value);
		if (!neuronSet) return [];

		return Array.from(neuronSet);
	}

	/**
	 * Check if neuron exists
	 */
	has(neuronId) {
		return this.neurons.has(neuronId);
	}

	/**
	 * Delete neuron and all its coordinates
	 */
	delete(neuronId) {
		if (!this.neurons.has(neuronId)) return false;

		// Remove coordinates
		const coords = this.coordinates.get(neuronId);
		if (coords) {
			for (const [dimensionId, value] of coords) {
				const dimMap = this.byDimensionValue.get(dimensionId);
				if (dimMap) {
					const neuronSet = dimMap.get(value);
					if (neuronSet) {
						neuronSet.delete(neuronId);
						if (neuronSet.size === 0) dimMap.delete(value);
					}
					if (dimMap.size === 0) this.byDimensionValue.delete(dimensionId);
				}
			}
			this.coordinates.delete(neuronId);
		}

		this.neurons.delete(neuronId);
		return true;
	}

	/**
	 * Clear all neurons
	 */
	clear() {
		this.neurons.clear();
		this.coordinates.clear();
		this.byDimensionValue.clear();
	}

	/**
	 * Get count
	 */
	size() {
		return this.neurons.size;
	}
}

/**
 * Store for active connections (scratch table)
 */
class ActiveConnectionStore {
	constructor() {
		// Map<"connectionId:level:age", {connection_id, from_neuron_id, to_neuron_id, level, age}>
		this.byKey = new Map();

		// Index: Map<level, Map<age, Set<connection_id>>>
		this.byLevelAge = new Map();

		// Index: Map<from_neuron_id, Set<connection_id>>
		this.byFrom = new Map();

		// Index: Map<to_neuron_id, Set<connection_id>>
		this.byTo = new Map();
	}

	/**
	 * Create composite key
	 */
	_key(connectionId, level, age) {
		return `${connectionId}:${level}:${age}`;
	}

	/**
	 * Add active connection
	 */
	add(connectionId, fromNeuronId, toNeuronId, level, age = 0) {
		const key = this._key(connectionId, level, age);

		// Add to primary storage
		this.byKey.set(key, { connection_id: connectionId, from_neuron_id: fromNeuronId, to_neuron_id: toNeuronId, level, age });

		// Add to level+age index
		if (!this.byLevelAge.has(level)) {
			this.byLevelAge.set(level, new Map());
		}
		const ageMap = this.byLevelAge.get(level);
		if (!ageMap.has(age)) {
			ageMap.set(age, new Set());
		}
		ageMap.get(age).add(connectionId);

		// Add to from index
		if (!this.byFrom.has(fromNeuronId)) {
			this.byFrom.set(fromNeuronId, new Set());
		}
		this.byFrom.get(fromNeuronId).add(connectionId);

		// Add to to index
		if (!this.byTo.has(toNeuronId)) {
			this.byTo.set(toNeuronId, new Set());
		}
		this.byTo.get(toNeuronId).add(connectionId);
	}

	/**
	 * Get all connections at specific level and age - O(1)
	 */
	getByLevelAge(level, age) {
		const ageMap = this.byLevelAge.get(level);
		if (!ageMap) return [];

		const connectionIds = ageMap.get(age);
		if (!connectionIds) return [];

		return Array.from(connectionIds).map(connId => {
			// Find the entry (need to search by connection_id)
			for (const entry of this.byKey.values()) {
				if (entry.connection_id === connId && entry.level === level && entry.age === age) {
					return entry;
				}
			}
			return null;
		}).filter(e => e !== null);
	}

	/**
	 * Clear all active connections
	 */
	clear() {
		this.byKey.clear();
		this.byLevelAge.clear();
		this.byFrom.clear();
		this.byTo.clear();
	}

	/**
	 * Get count
	 */
	size() {
		return this.byKey.size;
	}
}

export {
	ConnectionStore,
	ActiveNeuronStore,
	PatternStore,
	PatternPeakStore,
	NeuronStore,
	ActiveConnectionStore
};

