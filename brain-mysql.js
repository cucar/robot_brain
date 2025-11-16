import Brain from './brain.js';

/**
 * MySQL-based Brain Implementation
 * Uses MySQL MEMORY tables for active context and persistent tables for learned data
 */
export default class BrainMySQL extends Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {
		super();
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (memory tables)...');
		await this.truncateTables([
			'active_neurons',
			'inferred_neurons',
			'inferred_neurons_resolved',
			'matched_patterns',
			'matched_pattern_connections',
			'active_connections',
			'connection_inference_sources',
			'pattern_inference_sources',
			'unpredicted_connections',
			'new_patterns'
		]);
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all learned data)...');
		await this.truncateTables([
			'active_neurons',
			'inferred_neurons',
			'inferred_neurons_resolved',
			'active_connections',
			'matched_patterns',
			'matched_pattern_connections',
			'connection_inference_sources',
			'pattern_inference_sources',
			'unpredicted_connections',
			'new_patterns',
			'pattern_peaks',
			'pattern_past',
			'pattern_future',
			'connections',
			'coordinates',
			'neurons'
		]);
	}

	/**
	 * truncates given tables for database reset
	 */
	async truncateTables(tables) {
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		await Promise.all(tables.map(table => this.conn.query(`TRUNCATE ${table}`)));
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= baseNeuronMaxAge.
	 */
	async ageNeurons() {
		if (this.debug) console.log('Aging active neurons, connections, and inferred neurons...');

		// age all neurons and connections - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE active_connections SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inferred_neurons_resolved SET age = age + 1 ORDER BY age DESC');

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Clean up inferred neurons after execution (age >= 2)
		// age=0: fresh predictions, age=1: executed this frame, age>=2: no longer needed
		const [inferredResult] = await this.conn.query('DELETE FROM inferred_neurons WHERE age >= 2');
		if (this.debug) console.log(`Cleaned up ${inferredResult.affectedRows} executed inferred neurons (age >= 2)`);

		const [resolvedResult] = await this.conn.query('DELETE FROM inferred_neurons_resolved WHERE age >= 2');
		if (this.debug) console.log(`Cleaned up ${resolvedResult.affectedRows} executed resolved neurons (age >= 2)`);
	}

	/**
	 * Get the highest level with active neurons (MySQL implementation)
	 * Queries active_neurons table to find the highest level with any active neurons (regardless of age)
	 */
	async getMaxActiveLevel() {
		const [maxLevelResult] = await this.conn.query('SELECT MAX(level) as max_level FROM active_neurons');
		return maxLevelResult[0].max_level || 0;
	}

	/**
	 * Clear inference scratch data for new predictions (MySQL implementation)
	 */
	async clearInferenceData() {
		await this.truncateTables([
			'inferred_neurons',
			'connection_inference_sources',
			'pattern_inference_sources'
		]);
	}

	/**
	 * Get inferred base neurons with their coordinates and group by channel (MySQL implementation)
	 * @returns {Promise<Map>} - Map of channel names to array of inference objects (neuron_id, strength, coordinates)
	 */
	async getChannelInferences() {

		// get inferred base neurons from mysql
		const [baseNeuronCoordinates] = await this.conn.query(`
			SELECT inf.neuron_id, inf.strength, c.dimension_id, c.val, d.name as dimension_name, d.channel
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0 AND inf.level = 0
			ORDER BY d.channel, inf.neuron_id
		`);

		// group inferred coordinates by channel, building complete prediction objects with coordinates
		const channelInferences = new Map();
		for (const coordinate of baseNeuronCoordinates) {

			// if the channel doesn't have a map yet, create one - otherwise get it from the map
			if (!channelInferences.has(coordinate.channel)) channelInferences.set(coordinate.channel, new Map());
			const channelMap = channelInferences.get(coordinate.channel);

			// if the neuron doesn't have a record yet, create one with no coordinates
			if (!channelMap.has(coordinate.neuron_id)) channelMap.set(coordinate.neuron_id, { neuron_id: coordinate.neuron_id, strength: coordinate.strength, coordinates: {} });
			// if the neuron is inferred through multiple paths, sum inference strengths
			else channelMap.get(coordinate.neuron_id).strength += coordinate.strength;

			// add the coordinate to the neuron's prediction
			channelMap.get(coordinate.neuron_id).coordinates[coordinate.dimension_name] = coordinate.val;
		}

		// go through the channel inferences, convert to array of objects and return them
		for (const [channel, neuronMap] of channelInferences) channelInferences.set(channel, Array.from(neuronMap.values()));
		return channelInferences;
	}

	/**
	 * Write resolved predictions to MySQL storage (implementation of abstract method)
	 * This is called after conflict resolution at base level (level 0)
	 */
	async writeResolvedPredictions(allSelectedPredictions) {
		// Batch insert all selected predictions at once (all at level 0)
		await this.conn.query(
			'INSERT INTO inferred_neurons_resolved (neuron_id, level, age, strength) VALUES ?',
			[allSelectedPredictions.map(p => [p.neuron_id, 0, 0, p.strength])]
		);
	}

	/**
	 * Copy higher level predictions from inferred_neurons to inferred_neurons_resolved
	 * For levels > 0, there's no conflict resolution, so predictions are copied directly
	 */
	async copyHigherLevelPredictions() {
		await this.conn.query(`
			INSERT INTO inferred_neurons_resolved (neuron_id, level, age, strength)
			SELECT neuron_id, level, age, strength
			FROM inferred_neurons
			WHERE level > 0 AND age = 0
			ON DUPLICATE KEY UPDATE strength = VALUES(strength)
		`);
	}

	/**
	 * Reports accuracy of neuron inference from the previous frame for ALL levels.
	 */
	async reportPredictionsAccuracy() {

		// get the inferred neurons and correct activations
		const [data] = await this.conn.query(`
			SELECT inf.level, COUNT(*) as total, SUM(IF(an.neuron_id IS NOT NULL, 1, 0)) as correct
			FROM inferred_neurons_resolved inf
			LEFT JOIN active_neurons an ON inf.neuron_id = an.neuron_id AND an.level = inf.level AND an.age = 0
			WHERE inf.age = 1
			GROUP BY inf.level
		`);

		// process the results to report accuracy
		for (const row of data) {

			// initialize cumulative stats for this level if needed
			if (!this.accuracyStats.has(row.level)) this.accuracyStats.set(row.level, { correct: 0, total: 0 });
			const cumulative = this.accuracyStats.get(row.level);

			// update and report - convert to Number to avoid BigInt issues
			cumulative.correct += Number(row.correct);
			cumulative.total += Number(row.total);
			const currentRate = (Number(row.correct) / Number(row.total) * 100).toFixed(1);
			const avgRate = (cumulative.correct / cumulative.total * 100).toFixed(1);
			if (this.debug)
				console.log(`Level ${row.level} prediction accuracy: ${row.correct}/${row.total} (${currentRate}%) | Avg: ${cumulative.correct}/${cumulative.total} (${avgRate}%)`);
		}
	}

	/**
	 * Apply negative reinforcement to failed connection predictions.
	 * Weakens connections that made incorrect predictions.
	 */
	async negativeReinforceConnections() {

		// Apply negative reinforcement to failed connection predictions
		// Failed = predicted but not observed
		// Only weaken connections whose predictions made it through conflict resolution
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN connection_inference_sources cis ON cis.connection_id = c.id
			JOIN inferred_neurons_resolved inf ON inf.neuron_id = cis.inferred_neuron_id AND inf.level = cis.level
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE cis.level = ?
			AND c.strength > 0
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = cis.inferred_neuron_id
				AND an.level = cis.level
				AND an.age = 0
			)
		`, [this.connectionNegativeReinforcement, this.lastInferenceLevel]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Level ${this.lastInferenceLevel}: Weakened ${result.affectedRows} failed connection predictions`);
	}

	/**
	 * Check if there are high-confidence failed predictions (neurons predicted with high strength but didn't activate).
	 * This is the entry criteria for error pattern creation - we only learn when surprised by confident predictions failing.
	 * Returns the count of high-confidence failed predictions.
	 */
	async countFailedPredictions() {

		// For pattern inference: predictions were made at lastInferenceLevel - 1
		// For connection inference: predictions were made at lastInferenceLevel
		const predictionLevel = this.lastInferenceType === 'pattern' ? this.lastInferenceLevel - 1 : this.lastInferenceLevel;

		const [result] = await this.conn.query(`
			SELECT COUNT(*) as count
			FROM inferred_neurons_resolved inf
			WHERE inf.level = ?
			AND inf.age = 1
			AND inf.strength >= ?
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = inf.neuron_id
				AND an.level = inf.level
				AND an.age = 0
			)
		`, [predictionLevel, this.minErrorPatternThreshold]);
		return result[0].count;
	}

	/**
	 * Populate unpredicted_connections with active connections at age=0 that were not predicted.
	 * These are connections that fired but were not in the inferred_neurons_resolved predictions.
	 * Works for both connection and pattern inference types.
	 * No strength filter - we want to learn ALL unpredicted connections when surprised.
	 * Returns the number of unpredicted connections found.
	 */
	async populateUnpredictedConnections() {
		await this.conn.query(`TRUNCATE unpredicted_connections`);

		// For pattern inference: predictions were made at lastInferenceLevel - 1
		// For connection inference: predictions were made at lastInferenceLevel
		const predictionLevel = this.lastInferenceType === 'pattern' ? this.lastInferenceLevel - 1 : this.lastInferenceLevel;

		const [result] = await this.conn.query(`
			INSERT INTO unpredicted_connections (connection_id, level, from_neuron_id, to_neuron_id, strength)
			SELECT ac.connection_id, ac.level, c.from_neuron_id, c.to_neuron_id, c.strength
			FROM active_connections ac
			JOIN connections c ON c.id = ac.connection_id
			WHERE ac.level = ?
			AND ac.age = 0
			AND NOT EXISTS (
				SELECT 1 
				FROM inferred_neurons_resolved inf
				WHERE inf.neuron_id = c.to_neuron_id
				AND inf.level = ac.level
				AND inf.age = 1
			)
		`, [predictionLevel]);
		return result.affectedRows;
	}

	/**
	 * Connection inference at a specific level (MySQL implementation). Returns number of predictions made.
	 */
	async inferConnectionsAtLevel(level) {

		// Populate details table with ALL individual connection predictions
		await this.conn.query(`
			INSERT INTO connection_inference_sources (inferred_neuron_id, level, connection_id, prediction_strength)
			SELECT c.to_neuron_id, ?, c.id, c.strength * POW(?, c.distance)
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [level, this.peakTimeDecayFactor, level]);

		// Aggregate details to create master table (only predictions meeting strength threshold)
		const [result] = await this.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			SELECT inferred_neuron_id, level, 0, SUM(prediction_strength) as total_strength
			FROM connection_inference_sources
			WHERE level = ?
			GROUP BY inferred_neuron_id, level
			HAVING total_strength >= ?
		`, [level, this.minPredictionStrength]);

		if (this.debug && result.affectedRows > 0)
			console.log(`Level ${level}: Connection inference predicted ${result.affectedRows} neurons`);

		return result.affectedRows;
	}

	/**
	 * Merge pattern_future with observed connections FROM the peak.
	 * Called during learning phase after pattern inference from previous frame.
	 * Uses pattern_inference_sources to know which patterns made predictions.
	 *
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future connections that were correctly predicted (predicted AND observed)
	 * 2. Negative: Weaken pattern_future connections that were incorrectly predicted (predicted but NOT observed)
	 * 3. Novel: Add new connections FROM peak that were observed but not predicted
	 */
	async mergePatternFuture() {

		// Pattern neurons are at lastInferenceLevel, but predictions were made at lastInferenceLevel - 1
		const predictionLevel = this.lastInferenceLevel - 1;

		if (this.debug) {
			const [pfCount] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_future');
			const [pisCount] = await this.conn.query('SELECT COUNT(DISTINCT pattern_neuron_id) as count FROM pattern_inference_sources WHERE level = ?', [predictionLevel]);
			const [acCount] = await this.conn.query('SELECT COUNT(*) as count FROM active_connections WHERE level = ? AND age = 0', [predictionLevel]);
			console.log(`Level ${this.lastInferenceLevel}: Before merge - pattern_future: ${pfCount[0].count}, patterns that predicted: ${pisCount[0].count}, active_connections: ${acCount[0].count}`);
		}

		// Get the peaks of patterns that made predictions in previous frame
		// We need to compare their pattern_future with what actually happened (active_connections FROM those peaks)

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted connections
		// Connection in pattern_future AND observed in active_connections FROM peak at age=0
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN pattern_inference_sources pis ON pis.pattern_neuron_id = pf.pattern_neuron_id
			JOIN inferred_neurons_resolved inf ON inf.neuron_id = pis.inferred_neuron_id AND inf.level = pis.level
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.connection_id = pf.connection_id AND ac.level = ? AND ac.age = 0
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength + 1))
			WHERE pis.level = ?
		`, [predictionLevel, this.minConnectionStrength, this.maxConnectionStrength, predictionLevel]);
		if (this.debug)
			console.log(`Level ${this.lastInferenceLevel}: Strengthened ${strengthenResult.affectedRows} correct pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections
		// Connection in pattern_future but NOT observed in active_connections FROM peak
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN pattern_inference_sources pis ON pis.pattern_neuron_id = pf.pattern_neuron_id
			JOIN inferred_neurons_resolved inf ON inf.neuron_id = pis.inferred_neuron_id AND inf.level = pis.level
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.connection_id = pf.connection_id AND ac.level = ? AND ac.age = 0
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength - ?))
			WHERE pis.level = ?
			AND ac.connection_id IS NULL
		`, [predictionLevel, this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement, predictionLevel]);
		if (this.debug)
			console.log(`Level ${this.lastInferenceLevel}: Weakened ${weakenResult.affectedRows} failed pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Observed but not predicted
		// Active connections FROM peaks of patterns that made predictions, but not in pattern_future
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [novelResult] = await this.conn.query(`
			INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pis.pattern_neuron_id, ac.connection_id, 1.0
			FROM pattern_inference_sources pis
			JOIN inferred_neurons_resolved inf ON inf.neuron_id = pis.inferred_neuron_id AND inf.level = pis.level
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pis.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.level = ? AND ac.age = 0
			LEFT JOIN pattern_future pf ON pf.pattern_neuron_id = pis.pattern_neuron_id AND pf.connection_id = ac.connection_id
			WHERE pis.level = ?
			AND pf.connection_id IS NULL
		`, [predictionLevel, predictionLevel]);
		if (this.debug)
			console.log(`Level ${this.lastInferenceLevel}: Added ${novelResult.affectedRows} novel connections to pattern_future`);

		if (this.debug) {
			const [pfCountAfter] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_future');
			console.log(`Level ${this.lastInferenceLevel}: After merge - pattern_future: ${pfCountAfter[0].count}`);
		}
	}

	/**
	 * Pattern inference from a source level (MySQL implementation)
	 * Returns count of predictions made.
	 */
	async inferPatternsFromLevel(sourceLevel) {

		// Populate details table with ALL individual pattern predictions
		await this.conn.query(`
			INSERT INTO pattern_inference_sources (inferred_neuron_id, level, pattern_neuron_id, connection_id, prediction_strength)
			SELECT c.to_neuron_id, ?, an.neuron_id, c.id, pf.strength * c.strength * POW(?, c.distance)
			FROM active_neurons an
			JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE an.level = ?
			AND an.age >= 0
			AND c.distance = an.age + 1
			AND pf.strength > 0
			AND c.strength > 0
		`, [sourceLevel - 1, this.peakTimeDecayFactor, sourceLevel]);

		// Aggregate details to create master table (only predictions meeting strength threshold)
		const [result] = await this.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			SELECT inferred_neuron_id, level, 0, SUM(prediction_strength) as total_strength
			FROM pattern_inference_sources
			WHERE level = ?
			GROUP BY inferred_neuron_id, level
			HAVING total_strength >= ?
		`, [sourceLevel - 1, this.minPredictionStrength]);

		if (this.debug && result.affectedRows > 0)
			console.log(`Level ${sourceLevel}: Pattern inference predicted ${result.affectedRows} neurons at level ${sourceLevel - 1}`);

		return result.affectedRows;
	}

	/**
	 * Unpack predictions from higher level to base level via peak chain (MySQL implementation)
	 * Follows pattern_neuron → peak_neuron → peak_neuron down to base.
	 */
	async unpackToBase(fromLevel, source) {
		if (this.debug) console.log(`Unpacking ${source} predictions from level ${fromLevel} to base`);

		// Unpack level by level, grouping at each step to handle multiple patterns sharing same peak
		// A pattern neuron at level N IS its peak neuron at level N-1
		// Loop from fromLevel down to level 1, unpacking each level to the one below
		for (let level = fromLevel; level > 0; level--) {

			// Unpack pattern neurons at this level to their peaks at level-1
			// Multiple patterns can share same peak, so GROUP BY to sum strengths
			await this.conn.query(`
				INSERT INTO inferred_neurons (neuron_id, level, age, strength)
				SELECT pp.peak_neuron_id, ?, 0, SUM(inf.strength)
				FROM inferred_neurons inf
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = inf.neuron_id
				WHERE inf.level = ?
				AND inf.age = 0
				GROUP BY pp.peak_neuron_id
			`, [level - 1, level]);
		}

		if (this.debug) console.log(`Unpacked to base level`);
	}

	/**
	 * Populate new_patterns table from unpredicted connections.
	 * Finds peak neurons (from_neurons of unpredicted connections) and creates one pattern per peak.
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		await this.conn.query(`TRUNCATE new_patterns`);
		const [insertResult] = await this.conn.query(`
			INSERT INTO new_patterns (peak_neuron_id)
			SELECT from_neuron_id FROM unpredicted_connections GROUP BY from_neuron_id
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Create pattern neurons and map them to new_patterns.
	 * @param {number} patternCount - Number of patterns to create neurons for
	 */
	async createPatternNeurons(patternCount) {

		// Create pattern neurons (one per peak)
		const patternNeuronIds = await this.createNeurons(patternCount);

		// Update new_patterns with pattern neuron IDs in bulk
		// seq_id auto-increments from 1, patternNeuronIds are sequential, so: pattern_neuron_id = firstNeuronId + (seq_id - 1)
		await this.conn.query('UPDATE new_patterns SET pattern_neuron_id = ? + (seq_id - 1)', [patternNeuronIds[0]]);
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future.
	 * @param {number} predictionLevel - Level where predictions were made
	 */
	async mergeNewPatterns(predictionLevel) {

		// Create pattern_peaks entries
		await this.conn.query(`
			INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength)
			SELECT np.pattern_neuron_id, np.peak_neuron_id, 1.0
			FROM new_patterns np
		`);

		// Create pattern_past entries (active connections at age=1 leading TO the peak)
		// This captures the context that was present when the peak was active
		// Peak neurons are at predictionLevel (where the failed predictions were)
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, ac.connection_id, 1.0
			FROM new_patterns np
			JOIN active_connections ac ON ac.to_neuron_id = np.peak_neuron_id
			WHERE ac.level = ?
			AND ac.age = 1
		`, [predictionLevel]);

		// Create pattern_future entries (unpredicted connections FROM peak)
		// This is what the pattern should predict when it matches
		// Use connection_id directly from unpredicted_connections
		await this.conn.query(`
			INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, uc.connection_id, 1.0
			FROM new_patterns np
			JOIN unpredicted_connections uc ON uc.from_neuron_id = np.peak_neuron_id
		`);
	}

	/**
	 * fetches all neuron coordinates that could potentially match any point in the frame
	 */
	async getFrameCoordinates(frame) {
		const allPairs = [];

		for (const point of frame)
			for (const [dimName, val] of Object.entries(point))
				allPairs.push([this.dimensionNameToId[dimName], val]);

		const [rows] = await this.conn.query(`
			SELECT neuron_id, dimension_id, val
			FROM coordinates
			WHERE (dimension_id, val) IN (?)
		`, [allPairs]);

		const neuronCoords = new Map();
		for (const row of rows) {
			if (!neuronCoords.has(row.neuron_id))
				neuronCoords.set(row.neuron_id, new Map());
			neuronCoords.get(row.neuron_id).set(row.dimension_id, row.val);
		}

		return neuronCoords;
	}

	/**
	 * Sets coordinates for neurons in batches to avoid query size limits
	 * @param {Array<{neuron_id: number, point: Object}>} neurons - Array of neuron_id and point pairs
	 */
	async setNeuronCoordinates(neurons) {

		// flatten to rows of [neuron_id, dimension_id, value]
		const rows = neurons.flatMap(({ neuron_id, point }) =>
			Object.entries(point).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));

		// Process in batches to avoid query size limits
		const batchSize = 5000;
		for (let i = 0; i < rows.length; i += batchSize) {
			const batch = rows.slice(i, i + batchSize);
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ? ON DUPLICATE KEY UPDATE val = VALUES(val)', [batch]);
		}
	}

	/**
	 * Creates new neurons and return their IDs.
	 * MySQL guarantees sequential auto-increment IDs.
	 * @param {number} count - Number of neurons to create
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async createNeurons(count) {
		const valuesSql = Array(count).fill('()').join(',');
		const insertNeuronsResult = await this.conn.query(`INSERT INTO neurons () VALUES ${valuesSql}`);
		const firstNeuronId = insertNeuronsResult[0].insertId;

		// Return array of sequential IDs
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * Reinforce connections between active neurons at the specified level.
	 * Creates connections from all active neurons to newly activated (age=0) neurons.
	 * With uniform aging (age always 0-9), distance is simply the source neuron's age.
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT f.neuron_id as from_neuron_id, t.neuron_id as to_neuron_id, f.age as distance, 1 as strength
			FROM active_neurons f
			CROSS JOIN active_neurons t
            WHERE t.age = 0  -- target neurons are newly activated
            AND t.level = :level  -- target neurons are at the specified level
            AND f.level = t.level  -- restrict to same level only
			-- we used to have the condition below that enabled us to do associative pooling, but I'm realizing that 
            -- it should really be used only for spatial processing - it's a special case of that 
			-- spatial pooling will look at the neighboring neurons in terms of X-Y coordinates
			-- associative pooling is just a special case of that where there are no X-Y coordinates	
            -- AND (t.neuron_id != f.neuron_id OR f.age > 0)  -- no self-connections at same age
            AND f.age > 0 -- at this point, we are only learning connections between different ages to use for inference 
			ON DUPLICATE KEY UPDATE strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, strength + VALUES(strength)))
		`, { level, minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });
	}

	/**
	 * activate neurons at a specified level & distance - age is always zero - not sending to save characters in query
	 */
	async activateNeurons(neuronIds, level = 0) {

		// insert given neurons to the active neurons table
		await this.insertActiveNeurons(neuronIds, level);

		// reinforce connections between active neurons in the level
		await this.reinforceConnections(level);

		// activate connections for the newly activated neurons at this level
		await this.activateConnections(level);
	}

	/**
	 * inserts neurons at a specified level & distance - age is always zero - not sending to save characters in query
	 */
	async insertActiveNeurons(neuronIds, level = 0) {
		if (neuronIds.length === 0) return;
		const activations = neuronIds.map(neuronId => [neuronId, level]);
		await this.conn.query(`INSERT INTO active_neurons (neuron_id, level) VALUES ?`, [activations]);
	}

	/**
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 * @returns {Promise<boolean>}
	 */
	async recognizeLevelPatterns(level) {
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// Match active connections to known patterns and write to matched_patterns table
		const matchCount = await this.matchObservedPatterns(level);
		if (matchCount === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// Merge matched patterns: add/strengthen observed connections, weaken unobserved connections
		await this.mergeMatchedPatterns();

		// Activate all pattern neurons (from matched_patterns table) at the next level
		const [patternNeurons] = await this.conn.query('SELECT DISTINCT pattern_neuron_id FROM matched_patterns');
		const patternNeuronIds = patternNeurons.map(row => row.pattern_neuron_id);
		if (patternNeuronIds.length > 0) await this.activateNeurons(patternNeuronIds, level + 1);

		return true;
	}

	/**
	 * Populate active_connections table for newly activated neurons at the specified level.
	 * This is called immediately after reinforceConnections in activateNeurons.
	 * Inserts connections from all active neurons to age=0 neurons at the specified level.
	 * With uniform aging, distance matching is simply c.distance = f.age.
	 * Connections are inserted with age=0 (matching the to_neuron age).
	 */
	async activateConnections(level) {
		await this.conn.query(`
			INSERT IGNORE INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level, age)
			SELECT c.id as connection_id, c.from_neuron_id, c.to_neuron_id, t.level, 0 as age
			FROM connections c
			JOIN active_neurons f ON c.from_neuron_id = f.neuron_id
			JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = :level
			WHERE c.distance = f.age
			AND f.level = t.level  -- restrict to same levels only
            -- we used to have the condition below that enabled us to do associative pooling, but I'm realizing that 
            -- it should really be used only for spatial processing - it's a special case of that 
            -- spatial pooling will look at the neighboring neurons in terms of X-Y coordinates
            -- associative pooling is just a special case of that where there are no X-Y coordinates	
            -- AND (t.neuron_id != f.neuron_id OR f.age > 0)  -- no self-connections at same age
            AND f.age > 0 -- at this point, we are only learning connections between different ages to use for inference 
			AND c.strength > 0  -- only connections that are not removed
		`, { level });
	}

	/**
	 * Match active connections directly to known patterns.
	 * No peak detection needed - we only check neurons that are already known peaks (in pattern_peaks).
	 * Writes results to matched_patterns memory table.
	 * Matches by connection_id (which encodes from_neuron + to_neuron + distance) to preserve temporal structure.
	 * Uses connection overlap (66% threshold) to determine if patterns match.
	 * @param {number} level - The level to match patterns for
	 * @returns {Promise<number>} - Number of matched patterns
	 */
	async matchObservedPatterns(level) {
		if (this.debug) console.log('Matching active connections to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_patterns');
		await this.conn.query('TRUNCATE matched_pattern_connections');

		// Determine which patterns matched based on overlap threshold
		// A pattern matches if at least 66% of its pattern_past connections are in active_connections
		const [result] = await this.conn.query(`
			INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
			SELECT pp.peak_neuron_id, pp.pattern_neuron_id
			FROM active_neurons an
			JOIN pattern_peaks pp ON an.neuron_id = pp.peak_neuron_id
			JOIN pattern_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.to_neuron_id = pp.peak_neuron_id AND ac.connection_id = p.connection_id AND ac.level = ? AND ac.age = 0
			WHERE an.level = ? AND an.age = 0
			GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
			HAVING COUNT(DISTINCT CASE WHEN ac.connection_id IS NOT NULL THEN p.connection_id END) >= COUNT(DISTINCT p.connection_id) * ?
		`, [level, level, this.mergePatternThreshold]);
		if (this.debug) console.log(`Matched ${result.affectedRows} pattern-peak pairs`);
		if (result.affectedRows === 0) return 0;

		// Now populate matched_pattern_connections ONLY for matched patterns
		// Pattern connections: LEFT JOIN to active to determine if common or missing
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, p.connection_id, IF(ac.connection_id IS NOT NULL, 'common', 'missing') as status
			FROM matched_patterns mp
			JOIN pattern_past p ON mp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.to_neuron_id = mp.peak_neuron_id AND ac.connection_id = p.connection_id AND ac.level = ? AND ac.age = 0
		`, [level]);

		// Novel connections: active connections not in pattern_past
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, ac.connection_id, 'novel' as status
			FROM matched_patterns mp
			JOIN active_connections ac ON ac.to_neuron_id = mp.peak_neuron_id AND ac.level = ? AND ac.age = 0
			LEFT JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id AND p.connection_id = ac.connection_id
			WHERE p.connection_id IS NULL
		`, [level]);

		return result.affectedRows;
	}

	/**
	 * Merge matched patterns using pre-analyzed connection sets.
	 * Uses matched_pattern_connections table populated by matchObservedPatterns:
	 * 1. Add novel connections (status='novel')
	 * 2. Strengthen common connections (status='common')
	 * 3. Weaken missing connections (status='missing')
	 */
	async mergeMatchedPatterns() {
		if (this.debug) console.log('merging matched patterns...');

		// Reinforce pattern_peaks strength for matched patterns
		await this.conn.query(`
			UPDATE pattern_peaks pp
			JOIN matched_patterns mp ON pp.pattern_neuron_id = mp.pattern_neuron_id
			SET pp.strength = LEAST(1000, pp.strength + 1.0)
		`);

		// Add novel connections: connections in active but not in pattern
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT pattern_neuron_id, connection_id, 1
			FROM matched_pattern_connections
			WHERE status = 'novel'
		`);

		// Strengthen common connections: connections in both pattern and active (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_connections mpc ON p.pattern_neuron_id = mpc.pattern_neuron_id AND p.connection_id = mpc.connection_id
			SET p.strength = GREATEST(?, LEAST(?, p.strength + 1))
			WHERE mpc.status = 'common'
		`, [this.minConnectionStrength, this.maxConnectionStrength]);

		// Weaken missing connections: connections in pattern but not in active (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_connections mpc ON p.pattern_neuron_id = mpc.pattern_neuron_id AND p.connection_id = mpc.connection_id
			SET p.strength = GREATEST(?, LEAST(?, p.strength - ?))
			WHERE mpc.status = 'missing'
		`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
	}

	/**
	 * runs the forget cycle, reducing reward factors, pattern strengths, connection strengths and deleting unused neurons
	 * also deletes obsolete (negative) connections - very important step that helps the system avoid curse of dimensionality
	 */
	async runForgetCycle() {

		// we run the forget cycle periodically for clean up
		this.forgetCounter++;
		if (this.forgetCounter % this.forgetCycles !== 0) return;
		this.forgetCounter = 0;

		if (this.debug) {
			console.log('=== FORGET CYCLE STARTING ===');
			const cycleStart = Date.now();

			// 1. PATTERN FORGETTING: Reduce pattern strengths and remove dead patterns (clamped between minConnectionStrength and maxConnectionStrength)
			console.log('Running forget cycle - pattern_past update...');
			let stepStart = Date.now();
			const [patternPastUpdateResult] = await this.conn.query(`UPDATE pattern_past SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
			console.log(`  Pattern_past UPDATE took ${Date.now() - stepStart}ms (updated ${patternPastUpdateResult.affectedRows} rows)`);

			console.log('Running forget cycle - pattern_future update...');
			stepStart = Date.now();
			const [patternFutureUpdateResult] = await this.conn.query(`UPDATE pattern_future SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
			console.log(`  Pattern_future UPDATE took ${Date.now() - stepStart}ms (updated ${patternFutureUpdateResult.affectedRows} rows)`);

			console.log('Running forget cycle - pattern_peaks update...');
			stepStart = Date.now();
			const [patternPeaksUpdateResult] = await this.conn.query(`UPDATE pattern_peaks SET strength = GREATEST(0, strength - ?) WHERE strength > 0`, [this.patternForgetRate]);
			console.log(`  Pattern_peaks UPDATE took ${Date.now() - stepStart}ms (updated ${patternPeaksUpdateResult.affectedRows} rows)`);

			// Delete patterns with zero strength
			console.log('Running forget cycle - pattern deletion...');
			stepStart = Date.now();
			const [patternPastDeleteResult] = await this.conn.query(`DELETE FROM pattern_past WHERE strength = ?`, [this.minConnectionStrength]);
			console.log(`  Pattern_past DELETE took ${Date.now() - stepStart}ms (deleted ${patternPastDeleteResult.affectedRows} rows)`);

			stepStart = Date.now();
			const [patternFutureDeleteResult] = await this.conn.query(`DELETE FROM pattern_future WHERE strength = ?`, [this.minConnectionStrength]);
			console.log(`  Pattern_future DELETE took ${Date.now() - stepStart}ms (deleted ${patternFutureDeleteResult.affectedRows} rows)`);

			stepStart = Date.now();
			const [patternPeaksDeleteResult] = await this.conn.query(`DELETE FROM pattern_peaks WHERE strength <= 0`);
			console.log(`  Pattern_peaks DELETE took ${Date.now() - stepStart}ms (deleted ${patternPeaksDeleteResult.affectedRows} rows)`);

			// 2. CONNECTION FORGETTING: Reduce connection strengths and remove dead connections (clamped between minConnectionStrength and maxConnectionStrength)
			console.log('Running forget cycle - connection update...');
			stepStart = Date.now();
			const [connectionUpdateResult] = await this.conn.query(`UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.connectionForgetRate]);
			console.log(`  Connection UPDATE took ${Date.now() - stepStart}ms (updated ${connectionUpdateResult.affectedRows} rows)`);

			// Delete connections with zero strength
			console.log('Running forget cycle - connection deletion...');
			stepStart = Date.now();
			const [connectionDeleteResult] = await this.conn.query(`DELETE FROM connections WHERE strength = ?`, [this.minConnectionStrength]);
			console.log(`  Connection DELETE took ${Date.now() - stepStart}ms (deleted ${connectionDeleteResult.affectedRows} rows)`);

			// 3. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
			console.log('Running forget cycle - orphaned neurons cleanup...');
			stepStart = Date.now();
			const [neuronDeleteResult] = await this.conn.query(`
				DELETE n FROM neurons n
				LEFT JOIN connections c1 ON c1.from_neuron_id = n.id
				LEFT JOIN connections c2 ON c2.to_neuron_id = n.id
				LEFT JOIN pattern_past pp ON pp.pattern_neuron_id = n.id
				LEFT JOIN pattern_future pf ON pf.pattern_neuron_id = n.id
				LEFT JOIN active_neurons an ON an.neuron_id = n.id
				WHERE c1.from_neuron_id IS NULL
				  AND c2.to_neuron_id IS NULL
				  AND pp.pattern_neuron_id IS NULL
				  AND pf.pattern_neuron_id IS NULL
				  AND an.neuron_id IS NULL
			`);
			console.log(`  Orphaned neurons DELETE took ${Date.now() - stepStart}ms (deleted ${neuronDeleteResult.affectedRows} rows)`);

			console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
		}
		else {
			// Run forget cycle without logging - parallelize independent UPDATE operations
			await Promise.all([
				this.conn.query(`UPDATE pattern_past SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]),
				this.conn.query(`UPDATE pattern_future SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]),
				this.conn.query(`UPDATE pattern_peaks SET strength = GREATEST(0, strength - ?) WHERE strength > 0`, [this.patternForgetRate]),
				this.conn.query(`UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.connectionForgetRate])
			]);

			// Delete operations after updates complete - parallelize independent DELETE operations
			await Promise.all([
				this.conn.query(`DELETE FROM pattern_past WHERE strength = ?`, [this.minConnectionStrength]),
				this.conn.query(`DELETE FROM pattern_future WHERE strength = ?`, [this.minConnectionStrength]),
				this.conn.query(`DELETE FROM pattern_peaks WHERE strength <= 0`),
				this.conn.query(`DELETE FROM connections WHERE strength = ?`, [this.minConnectionStrength])
			]);

			// Orphaned neuron cleanup - optimized with LEFT JOINs instead of NOT EXISTS
			await this.conn.query(`
				DELETE n FROM neurons n
				LEFT JOIN connections c1 ON c1.from_neuron_id = n.id
				LEFT JOIN connections c2 ON c2.to_neuron_id = n.id
				LEFT JOIN pattern_past pp ON pp.pattern_neuron_id = n.id
				LEFT JOIN pattern_future pf ON pf.pattern_neuron_id = n.id
				LEFT JOIN active_neurons an ON an.neuron_id = n.id
				WHERE c1.from_neuron_id IS NULL
				  AND c2.to_neuron_id IS NULL
				  AND pp.pattern_neuron_id IS NULL
				  AND pf.pattern_neuron_id IS NULL
				  AND an.neuron_id IS NULL
			`);
		}
	}

	/**
	 * Apply global reward to active connections that led to executed outputs.
	 * Uses multiplicative rewards with exponential temporal decay.
	 * Older connections get less reward/punishment (decay applied to the reward exponent).
	 */
	async applyRewards(globalReward) {

		if (globalReward === 1.0) {
			if (this.debug) console.log('Neutral global reward - no updates needed');
			return;
		}

		// Multiplicative reward: strength is multiplied by globalReward raised to a decayed power
		// globalReward = 1.5, age = 0 → multiply by 1.5^1.0 = 1.5 (50% increase)
		// globalReward = 1.5, age = 1 → multiply by 1.5^0.9 = 1.41 (41% increase)
		// globalReward = 0.5, age = 0 → multiply by 0.5^1.0 = 0.5 (50% decrease)
		// This is proportional to existing strength, avoiding saturation issues

		// Apply reward to active_connections with exponential temporal decay (clamped between minConnectionStrength and maxConnectionStrength)
		// Older connections (higher age) get less reward/punishment via decay applied to the exponent
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN active_connections ac ON c.id = ac.connection_id
			SET c.strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, c.strength * POW(:globalReward, POW(:rewardTimeDecayFactor, ac.age - 1))))
			WHERE ac.age > 0 -- skip the recently made connections - that's where the output was executed
		`, { globalReward, rewardTimeDecayFactor: this.rewardTimeDecayFactor, minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });

		if (this.debug) console.log(`Applied global reward ${globalReward.toFixed(3)} to ${result.affectedRows} active connections (multiplicative)`);
	}
}