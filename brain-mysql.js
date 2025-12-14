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
			'matched_patterns',
			'matched_pattern_connections',
			'active_connections',
			'org_inference_sources',
			'base_inference_sources',
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
			'active_connections',
			'matched_patterns',
			'matched_pattern_connections',
			'org_inference_sources',
			'base_inference_sources',
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
	 * Also ages inference source tables for temporal credit assignment.
	 */
	async ageNeurons() {
		if (this.debug2) console.log('Aging active neurons, connections, and inferred neurons...');

		// age all neurons and connections - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE active_connections SET age = age + 1 ORDER BY age DESC');

		// Age inference tables for temporal credit assignment
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE org_inference_sources SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE base_inference_sources SET age = age + 1 ORDER BY age DESC');

		// Skip deletions until we have enough frames
		if (this.frameNumber < this.baseNeuronMaxAge) return;

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Clean up inferred neurons after execution
		const [inferredResult] = await this.conn.query('DELETE FROM inferred_neurons WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${inferredResult.affectedRows} executed inferred neurons (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out inference sources (same lifecycle as neurons)
		const [orgSourcesResult] = await this.conn.query('DELETE FROM org_inference_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		const [baseSourcesResult] = await this.conn.query('DELETE FROM base_inference_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${orgSourcesResult.affectedRows} aged-out org inference sources, ${baseSourcesResult.affectedRows} base inference sources (age >= ${this.baseNeuronMaxAge})`);
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
	 * Get inferred base neurons with their coordinates and group by channel (MySQL implementation)
	 * @returns {Promise<Map>} - Map of channel names to array of inference objects (neuron_id, strength, coordinates)
	 */
	async getChannelBaseInferences() {

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
		const channelBaseInferences = new Map();
		for (const coordinate of baseNeuronCoordinates) {

			// if the channel doesn't have a map yet, create one - otherwise get it from the map
			if (!channelBaseInferences.has(coordinate.channel)) channelBaseInferences.set(coordinate.channel, new Map());
			const channelMap = channelBaseInferences.get(coordinate.channel);

			// if the neuron doesn't have a record yet, create one with no coordinates
			if (!channelMap.has(coordinate.neuron_id)) channelMap.set(coordinate.neuron_id, { neuron_id: coordinate.neuron_id, strength: coordinate.strength, coordinates: {} });
			// if the neuron is inferred through multiple paths, sum inference strengths
			else channelMap.get(coordinate.neuron_id).strength += coordinate.strength;

			// add the coordinate to the neuron's prediction
			channelMap.get(coordinate.neuron_id).coordinates[coordinate.dimension_name] = coordinate.val;
		}

		// go through the channel inferences, convert to array of objects and return them
		for (const [channel, neuronMap] of channelBaseInferences) channelBaseInferences.set(channel, Array.from(neuronMap.values()));
		return channelBaseInferences;
	}

	/**
	 * Save resolved inferences to MySQL storage (implementation of abstract method)
	 * This is called after conflict resolution at base level (level 0)
	 * Updates inferred_neurons in place (deletes invalid, inserts corrected)
	 */
	async saveResolvedInferences(resolvedInferences, deletedNeuronIds) {

		// Delete invalid inferences that were removed during conflict resolution
		if (deletedNeuronIds.length > 0) {
			await this.conn.query('DELETE FROM inferred_neurons WHERE neuron_id IN (?) AND level = 0 AND age = 0', [deletedNeuronIds]);
			// Also delete their inference sources
			await this.conn.query('DELETE FROM org_inference_sources WHERE inferred_neuron_id IN (?) AND age = 0 AND level = 0', [deletedNeuronIds]);
			await this.conn.query('DELETE FROM base_inference_sources WHERE base_neuron_id IN (?) AND age = 0', [deletedNeuronIds]);
		}

		// Insert corrected inferences (if any)
		if (resolvedInferences.length > 0)
			await this.insertInferredNeurons(resolvedInferences.map(p => [p.neuron_id, 0, 0, p.strength]), 'replace');
	}

	/**
	 * Reports accuracy of neuron inference from the previous frame for ALL levels.
	 * Only checks input predictions (event dimensions), not output predictions (action dimensions).
	 * Output predictions become self-fulfilled prophecies when executed, so they shouldn't be checked.
	 * Higher-level neurons (pattern neurons) are always checked since they don't have coordinates.
	 */
	async reportPredictionsAccuracy() {

		// Get the inferred neurons and correct activations
		// For level 0: only count neurons that have at least one event dimension
		// For level > 0: count all neurons (pattern neurons don't have coordinates)
		const [data] = await this.conn.query(`
			SELECT inf.level, COUNT(*) as total, SUM(IF(an.neuron_id IS NOT NULL, 1, 0)) as correct
			FROM inferred_neurons inf
			LEFT JOIN active_neurons an ON inf.neuron_id = an.neuron_id AND an.level = inf.level AND an.age = 0
			WHERE inf.age = 1
			AND (
				inf.level > 0
				OR EXISTS (
					SELECT 1 FROM coordinates c
					JOIN dimensions d ON c.dimension_id = d.id
					WHERE c.neuron_id = inf.neuron_id
					AND d.type = 'event'
				)
			)
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

		// Debug: Show which specific predictions failed at base level
		if (this.debug && data.some(row => row.level === 0)) {
			const [failed] = await this.conn.query(`
				SELECT inf.neuron_id, d.name as dimension_name, c.val
				FROM inferred_neurons inf
				JOIN coordinates c ON inf.neuron_id = c.neuron_id
				JOIN dimensions d ON c.dimension_id = d.id
				WHERE inf.level = 0 AND inf.age = 1
				AND NOT EXISTS (
					SELECT 1 FROM active_neurons an
					WHERE an.neuron_id = inf.neuron_id AND an.level = 0 AND an.age = 0
				)
				AND EXISTS (
					SELECT 1 FROM coordinates c2
					JOIN dimensions d2 ON c2.dimension_id = d2.id
					WHERE c2.neuron_id = inf.neuron_id
					AND d2.type IN ('event', 'state')
				)
				ORDER BY inf.neuron_id, d.name
			`);

			if (failed.length > 0) {
				const failedByNeuron = new Map();
				for (const row of failed) {
					if (!failedByNeuron.has(row.neuron_id)) failedByNeuron.set(row.neuron_id, []);
					failedByNeuron.get(row.neuron_id).push(`${row.dimension_name}=${row.val}`);
				}
				console.log(`   ❌ Failed predictions (L0):`);
				for (const [neuronId, coords] of failedByNeuron) {
					console.log(`      N${neuronId}: ${coords.join(', ')}`);
				}
			}
		}
	}

	/**
	 * Apply negative reinforcement to failed connection predictions.
	 * Weakens connections that made incorrect predictions at their original inference level.
	 * Uses org_inference_sources to find which connections made predictions and at what level.
	 */
	async negativeReinforceConnections() {

		// Apply negative reinforcement to failed INPUT predictions only - Failed = predicted but not observed
		const [result] = await this.conn.query(`
			UPDATE connections c
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE c.strength > 0
			-- penalize connections that were inferred in the previous frame
			AND c.id IN (SELECT source_id FROM org_inference_sources WHERE source_type = 'connection' AND age = 1)
			-- penalize the connections that did not come true
			AND c.id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
			-- negative reinforcement only applies to event predictions - not actions
			AND EXISTS (
				SELECT 1 FROM coordinates coord
				JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = c.to_neuron_id
				AND d.type = 'event'
			)
		`, [this.connectionNegativeReinforcement]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Weakened ${result.affectedRows} failed INPUT predictions`);
	}

	/**
	 * Check if there are high-confidence failed predictions (neurons predicted with high strength but didn't activate).
	 * This is the entry criteria for error pattern creation - we only learn when surprised by confident predictions failing.
	 * Returns the count of high-confidence failed predictions.
	 */
	async countFailedPredictions(predictionLevel) {

		const [result] = await this.conn.query(`
			SELECT COUNT(*) as count
			FROM inferred_neurons inf
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
	 * These are connections that fired but were not in the inferred_neurons predictions.
	 * Works for both connection and pattern inference types.
	 * No strength filter - we want to learn ALL unpredicted connections when surprised.
	 * Returns the number of unpredicted connections found.
	 */
	async populateUnpredictedConnections(predictionLevel) {
		await this.conn.query(`TRUNCATE unpredicted_connections`);

		const [result] = await this.conn.query(`
			INSERT INTO unpredicted_connections (connection_id, level, from_neuron_id, to_neuron_id, strength)
			SELECT ac.connection_id, ac.level, c.from_neuron_id, c.to_neuron_id, c.strength
			FROM active_connections ac
			JOIN connections c ON c.id = ac.connection_id
			WHERE ac.level = ?
			AND ac.age = 0
			AND NOT EXISTS (
				SELECT 1
				FROM inferred_neurons inf
				WHERE inf.neuron_id = c.to_neuron_id
				AND inf.level = ac.level
				AND inf.age = 1
			)
		`, [predictionLevel]);
		return result.affectedRows;
	}

	/**
	 * Connection inference at a specific level (MySQL implementation). Returns number of predictions made.
	 * Collects connection sources in memory, then saves to org_inference_sources and unpacks to base_inference_sources.
	 * @returns {Boolean} True if predictions were made, false if not
	 */
	async inferConnections(level) {

		// Get all connection inferences at this level
		const [inferences] = await this.conn.query(`
			SELECT c.to_neuron_id, c.id as connection_id, c.strength * c.reward * c.habituation * POW(?, c.distance - 1) as prediction_strength
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, level]);
		if (this.debug && inferences.length > 0) console.log(`Level ${level}: Connection inference found ${inferences.length} predictions`);
		if (inferences.length === 0) return false;

		// Process inferences using common logic
		return this.processInferences(inferences, level, 'connection', 'connection_id');
	}

	/**
	 * Pattern inference from a source level (MySQL implementation)
	 * Collects pattern sources in memory, then saves to org_inference_sources and unpacks to base_inference_sources.
	 * @returns {Boolean} True if predictions were made, false if not
	 **/
	async inferPatterns(sourceLevel) {

		// Get all pattern predictions at this level
		const [predictions] = await this.conn.query(`
			SELECT c.to_neuron_id, pf.id as pattern_future_id, pf.strength * pf.reward * pf.habituation * c.strength * c.reward * POW(?, c.distance - 1) as prediction_strength
			FROM active_neurons an
			JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE an.level = ?
			AND an.age >= 0
			AND c.distance = an.age + 1
			AND pf.strength > 0
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, sourceLevel]);
		if (this.debug && predictions.length > 0) console.log(`Level ${sourceLevel}: Pattern inference found ${predictions.length} predictions`);
		if (predictions.length === 0) return false;

		// Patterns predict one level down
		return this.processInferences(predictions, sourceLevel - 1, 'pattern', 'pattern_future_id');
	}

	/**
	 * Common logic for processing inferences from inferConnections and inferPatterns
	 * @param {Array} inferences - Array of prediction objects with to_neuron_id, source_id field, and prediction_strength
	 * @param {Number} targetLevel - Level where predictions should be inserted
	 * @param {String} sourceType - 'connection' or 'pattern'
	 * @param {String} sourceIdField - Field name containing the source ID (e.g., 'connection_id' or 'pattern_future_id')
	 * @returns {Boolean} True if predictions were made, false if not
	 */
	async processInferences(inferences, targetLevel, sourceType, sourceIdField) {

		// Build in-memory map: neuron_id → [{source_type, source_id, strength}]
		const neuronSources = new Map();
		for (const pred of inferences) {
			if (!neuronSources.has(pred.to_neuron_id)) neuronSources.set(pred.to_neuron_id, []);
			neuronSources.get(pred.to_neuron_id).push({
				source_type: sourceType,
				source_id: pred[sourceIdField],
				strength: pred.prediction_strength
			});
		}

		// Aggregate strengths per neuron and filter by threshold
		// Build a filtered map with only neurons that pass the threshold
		const inferredNeuronSources = new Map();
		const inferredNeurons = [];
		for (const [neuronId, sources] of neuronSources) {
			const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
			if (totalStrength >= this.minInferenceStrength) {
				inferredNeurons.push({ neuron_id: neuronId, strength: totalStrength });
				inferredNeuronSources.set(neuronId, sources);
			}
		}

		// if no neurons pass the threshold, return false to indicate that no inferences were made
		if (inferredNeurons.length === 0) return false;

		// Insert into inferred_neurons
		await this.insertInferredNeurons(inferredNeurons.map(n => [n.neuron_id, targetLevel, 0, n.strength]));
		if (this.debug) console.log(`Level ${targetLevel}: ${sourceType} inference predicted ${inferredNeurons.length} neurons`);

		// Save original inference sources at the level where inference was made (for learnFromErrors)
		await this.saveOriginalInferenceSources(inferredNeuronSources, targetLevel);

		// Unpack to base level and connect inference sources to base level inferences
		const baseLevelSources = await this.unpackInferences(targetLevel, inferredNeuronSources);

		// Save base level inference sources (for applyRewards)
		await this.saveBaseInferenceSources(baseLevelSources);

		// return true to indicate that we have inferences
		return true;
	}

	/**
	 * Unpack inferences from higher level to base level via peak chain.
	 * Follows pattern_neuron → peak_neuron → peak_neuron down to base.
	 * Tracks which sources (connection_id or pattern_future_id) led to which base outputs.
	 * @param {Number} fromLevel - the level where predictions were made
	 * @param {Map} neuronSources - Map of base level neuron_id → [{source_type, source_id, strength}]
	 */
	async unpackInferences(fromLevel, neuronSources) {
		if (this.debug2) console.log(`Unpacking inferences from level ${fromLevel} to base`);

		// if we're at the base level already, use the existing sources
		if (fromLevel === 0) return neuronSources;

		// Unpack level by level
		let currentLevelSources = neuronSources;
		for (let level = fromLevel; level > 0; level--) {

			// Get peaks from current level sources
			const peaks = await this.getPeaksFromSources(currentLevelSources);
			if (peaks.length === 0) throw new Error(`Failed to unpack level ${level}: no peaks found for ${currentLevelSources.size} pattern neurons`);

			// Get next level sources from peaks and current level sources
			const nextLevelSources = this.buildNextLevelSources(peaks, currentLevelSources);
			if (nextLevelSources.size === 0) throw new Error(`Cannot get next level sources at level ${level}: ${currentLevelSources.size} pattern neurons`);

			// Get next level inferred neurons
			const unpackedNeurons = this.buildInferredNeurons(nextLevelSources, level - 1);

			// Insert inferred neurons with additive strength
			await this.insertInferredNeurons(unpackedNeurons, 'add');

			// continue on to the next level
			currentLevelSources = nextLevelSources;
		}

		// return the base level sources
		return currentLevelSources;
	}

	/**
	 * Get peaks for pattern neurons from current level sources
	 * @param {Map} currentLevelSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @returns {Promise<Array>} Array of {pattern_neuron_id, peak_neuron_id}
	 */
	async getPeaksFromSources(currentLevelSources) {
		const neuronIds = [...currentLevelSources.keys()];
		const [peaks] = await this.conn.query(
			'SELECT pattern_neuron_id, peak_neuron_id FROM pattern_peaks WHERE pattern_neuron_id IN (?)',
			[neuronIds]
		);
		return peaks;
	}

	/**
	 * Build next level's mapping - peaks inherit sources from parent patterns
	 * If we inferred a pattern neuron at a higher level, we will connect it to the outputs via peaks
	 * @param {Array} peaks - Array of {pattern_neuron_id, peak_neuron_id}
	 * @param {Map} currentLevelSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @returns {Map} Map of peak_neuron_id → [{source_type, source_id, strength}]
	 */
	buildNextLevelSources(peaks, currentLevelSources) {
		const nextLevelSources = new Map();
		for (const peak of peaks) {
			const sources = currentLevelSources.get(peak.pattern_neuron_id);
			if (!sources) continue;
			if (!nextLevelSources.has(peak.peak_neuron_id)) nextLevelSources.set(peak.peak_neuron_id, []);
			nextLevelSources.get(peak.peak_neuron_id).push(...sources);
		}
		return nextLevelSources;
	}

	/**
	 * Build inferred neurons array from neuron sources
	 * @param {Map} neuronSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @param {Number} level - Level for the inferred neurons
	 * @returns {Array} Array of [neuron_id, level, age, strength]
	 */
	buildInferredNeurons(neuronSources, level) {
		const inferredNeurons = [];
		for (const [neuronId, sources] of neuronSources) {
			const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
			inferredNeurons.push([neuronId, level, 0, totalStrength]);
		}
		return inferredNeurons;
	}

	/**
	 * Insert inferred neurons into the database
	 * @param {Array} neurons - Array of [neuron_id, level, age, strength]
	 * @param {String} duplicateMode - How to handle duplicates: 'ignore', 'replace', 'add'
	 */
	async insertInferredNeurons(neurons, duplicateMode = 'none') {
		if (neurons.length === 0) return;

		let query = 'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ?';
		if (duplicateMode === 'ignore')
			query += ' ON DUPLICATE KEY UPDATE strength = strength';
		else if (duplicateMode === 'replace')
			query += ' ON DUPLICATE KEY UPDATE strength = VALUES(strength)';
		else if (duplicateMode === 'add')
			query += ' ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)';

		await this.conn.query(query, [neurons]);
	}

	/**
	 * Deduplicate and aggregate inference sources
	 * Same source can reach same neuron via multiple paths through the pattern hierarchy
	 * @param {Map} neuronSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @param {String} neuronKey - ase_neuron_id for base level, inferred_neuron_id for original level inferences
	 * @returns {Map} Map of unique key → {neuron_id, source_type, source_id, strength}
	 */
	deduplicateInferenceSources(neuronSources, neuronKey) {
		const deduped = new Map();
		for (const [neuronId, sources] of neuronSources) {
			for (const src of sources) {
				const key = `${neuronId}:${src.source_type}:${src.source_id}`;
				if (deduped.has(key)) deduped.get(key).strength += src.strength;
				else deduped.set(key, { [neuronKey]: neuronId, ...src });
			}
		}
		return deduped;
	}

	/**
	 * Save original inference sources to the database (at the level where inference was made)
	 * Used by learnFromErrors methods to validate predictions at their original level
	 * @param {Map} inferredNeuronSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @param {Number} level - The level where inference was made
	 */
	async saveOriginalInferenceSources(inferredNeuronSources, level) {

		// Dedupe and aggregate (same source can reach same neuron via multiple paths)
		const deduped = this.deduplicateInferenceSources(inferredNeuronSources, 'inferred_neuron_id');

		// Batch insert into org_inference_sources
		const rows = [...deduped.values()].map(s => [0, s.inferred_neuron_id, level, s.source_type, s.source_id, s.strength]);
		await this.conn.query('INSERT INTO org_inference_sources (age, inferred_neuron_id, level, source_type, source_id, inference_strength) VALUES ?', [rows]);
		if (this.debug2) console.log(`Saved ${rows.length} original inference sources at level ${level}`);
	}

	/**
	 * Save base inference sources to the database (unpacked to base level)
	 * Used by applyRewards to reward sources that led to outputs
	 * @param {Map} baseLevelSources - Map of base_neuron_id → [{source_type, source_id, strength}]
	 */
	async saveBaseInferenceSources(baseLevelSources) {

		// Dedupe and aggregate (same source can reach same base via multiple paths)
		const deduped = this.deduplicateInferenceSources(baseLevelSources, 'base_neuron_id');

		// Batch insert into base_inference_sources
		const rows = [...deduped.values()].map(s => [0, s.base_neuron_id, s.source_type, s.source_id, s.strength]);
		await this.conn.query('INSERT INTO base_inference_sources (age, base_neuron_id, source_type, source_id, inference_strength) VALUES ?', [rows]);
		if (this.debug2) console.log(`Saved ${rows.length} base inference sources`);
	}

	/**
	 * Override an inference with a new neuron (unified method for correction and exploration)
	 * Both correction and exploration are override mechanisms - the only difference is:
	 * - Correction: interprets existing inference to make it valid (preserves strength)
	 * - Exploration: randomly selects a valid action (uses maximum strength)
	 *
	 * This method:
	 * 1. Cleans up inference sources for original neuron (if any)
	 * 2. Finds connections that could have predicted the override neuron
	 * 3. Inserts override sources into both org_inference_sources and base_inference_sources
	 *
	 * Note: Does NOT update inferred_neurons - caller must handle that separately
	 *
	 * @param {Number} overrideNeuronId - neuron ID to use instead
	 * @param {Number|null} originalNeuronId - neuron ID being overridden (null for pure exploration)
	 */
	async overrideBaseInferenceSources(overrideNeuronId, originalNeuronId = null) {

		// 1. Clean up inference sources for original neuron (if any)
		if (originalNeuronId) {
			await this.conn.query('DELETE FROM org_inference_sources WHERE inferred_neuron_id = ? AND age = 0 AND level = 0', [originalNeuronId]);
			await this.conn.query('DELETE FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [originalNeuronId]);
		}

		// Also clean up any existing sources for the override neuron (in case it was already inferred)
		await this.conn.query('DELETE FROM org_inference_sources WHERE inferred_neuron_id = ? AND age = 0 AND level = 0', [overrideNeuronId]);
		await this.conn.query('DELETE FROM base_inference_sources WHERE base_neuron_id = ? AND age = 0', [overrideNeuronId]);

		// 2. Find connections that could have predicted the override neuron and insert as sources
		// Insert into org_inference_sources (level 0 since override is always at base level)
		await this.conn.query(`
			INSERT INTO org_inference_sources (age, inferred_neuron_id, level, source_type, source_id, inference_strength)
			SELECT 0, c.to_neuron_id, 0, 'connection', c.id, c.strength * c.reward * c.habituation * POW(?, c.distance - 1)
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = 0
			AND c.to_neuron_id = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, overrideNeuronId]);

		// Insert into base_inference_sources (same sources, for rewards)
		const [result] = await this.conn.query(`
			INSERT INTO base_inference_sources (age, base_neuron_id, source_type, source_id, inference_strength)
			SELECT 0, c.to_neuron_id, 'connection', c.id, c.strength * c.reward * c.habituation * POW(?, c.distance - 1)
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = 0
			AND c.to_neuron_id = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, overrideNeuronId]);

		if (this.debug && result.affectedRows > 0) console.log(`  Found ${result.affectedRows} possible connection sources for override`);
	}

	/**
	 * Save exploration neuron to inferred_neurons table
	 * Called after overrideBaseInferenceSources for exploration actions
	 * @param {Number} neuronId - neuron ID to save
	 */
	async saveExplorationNeuron(neuronId) {
		await this.insertInferredNeurons([[neuronId, 0, 0, 100000]], 'replace');
	}

	/**
	 * Correct inferred actions when conflict resolution changes them (MySQL implementation)
	 * Correction is an override mechanism that interprets existing connection/pattern inferences
	 * Only updates inference sources - inferred_neurons will be updated by saveResolvedInferences
	 * @param {Array} corrections - Array of { originalNeuronId, correctedCoordinates, strength }
	 */
	async correctInferredActions(corrections) {
		if (!corrections || corrections.length === 0) return;

		for (const correction of corrections) {
			const { originalNeuronId, correctedCoordinates, strength } = correction;

			// Find or create neuron for corrected action
			const [correctedNeuronId] = await this.getFrameNeurons([correctedCoordinates]);
			if (this.debug) console.log(`Correcting inference: neuron ${originalNeuronId} → ${correctedNeuronId} (strength: ${strength})`);

			// Update inference sources only - saveResolvedInferences will update inferred_neurons
			await this.overrideBaseInferenceSources(correctedNeuronId, originalNeuronId);
		}
	}

	/**
	 * Merge pattern_future with observed connections FROM the peak.
	 * Called during learning phase after pattern inference from previous frame.
	 * Uses org_inference_sources with source_type='pattern' to know which patterns made predictions and at what level.
	 * Pattern at level N predicts connections at level N-1 (where the peak is).
	 *
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future connections that were correctly predicted (predicted AND observed)
	 * 2. Negative: Weaken pattern_future connections that were incorrectly predicted (predicted but NOT observed)
	 * 3. Novel: Add new connections FROM peak that were observed but not predicted
	 */
	async mergePatternFuture() {

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted connections
		// Connection in pattern_future AND observed in active_connections FROM peak at age=0
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future
			SET strength = GREATEST(?, LEAST(?, strength + 1))
            WHERE strength > 0
			-- pattern future record should be inferred in the previous frame
			AND id IN (SELECT source_id FROM org_inference_sources WHERE age = 1 AND source_type = 'pattern')
			-- and pattern future connection should now be active (came true)
			AND connection_id IN (SELECT connection_id FROM active_connections WHERE age = 0)
		`, [this.minConnectionStrength, this.maxConnectionStrength]);
		if (this.debug && strengthenResult.affectedRows > 0)
			console.log(`Strengthened ${strengthenResult.affectedRows} correct pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections
		// Connection in pattern_future but NOT observed in active_connections FROM peak
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future
			SET strength = GREATEST(?, LEAST(?, strength - ?))
            WHERE strength > 0
            -- pattern future record should be inferred in the previous frame
            AND id IN (SELECT source_id FROM org_inference_sources WHERE age = 1 AND source_type = 'pattern')
			-- and pattern future connection should NOT be active (did NOT come true)
			AND connection_id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
		`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug && weakenResult.affectedRows > 0)
			console.log(`Weakened ${weakenResult.affectedRows} failed pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Observed but not predicted
		// For patterns that made predictions: add observed connections FROM their peaks that were NOT predicted (not in pattern_future)
		const [novelResult] = await this.conn.query(`
			INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pf_src.pattern_neuron_id, ac.connection_id, 1.0
			FROM org_inference_sources isrc
			JOIN pattern_future pf_src ON pf_src.id = isrc.source_id
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf_src.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.age = 0
			WHERE isrc.age = 1
			AND isrc.source_type = 'pattern'
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf
				WHERE pf.pattern_neuron_id = pf_src.pattern_neuron_id
				AND pf.connection_id = ac.connection_id
			)
		`);
		if (this.debug && novelResult.affectedRows > 0)
			console.log(`Added ${novelResult.affectedRows} novel connections to pattern_future`);
	}

	/**
	 * Get the prediction level from previous frame's inference.
	 * Returns null if no inference occurred (only exploration).
	 * Checks org_inference_sources to find the level where inference was made.
	 */
	async getPreviousInferenceLevel() {

		// Get prediction level from org_inference_sources (age=1 means previous frame)
		const [levelResult] = await this.conn.query(`
			SELECT MAX(level) as level FROM org_inference_sources WHERE age = 1
		`);

		return levelResult[0] && levelResult[0].level !== null ? levelResult[0].level : null;
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
		if (this.debug2) console.log(`Processing level ${level} for pattern recognition`);

		// Match active connections to known patterns and write to matched_patterns table
		const matchCount = await this.matchObservedPatterns(level);
		if (matchCount === 0) {
			if (this.debug2) console.log(`No pattern matches found at level ${level}`);
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
		if (this.debug2) console.log('Matching active connections to known patterns');

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
		if (this.debug && result.affectedRows > 0) console.log(`Matched ${result.affectedRows} pattern-peak pairs`);
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

			// 3. REWARD DECAY: Move reward factors back toward 1.0 (neutral)
			// Formula: reward = reward + (1.0 - reward) * rewardForgetRate
			// reward=2.0, rate=0.05 → 2.0 + (1.0-2.0)*0.05 = 1.95
			// reward=0.5, rate=0.05 → 0.5 + (1.0-0.5)*0.05 = 0.525
			console.log('Running forget cycle - connection reward decay...');
			stepStart = Date.now();
			const [connRewardResult] = await this.conn.query(`UPDATE connections SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]);
			console.log(`  Connection reward decay took ${Date.now() - stepStart}ms (updated ${connRewardResult.affectedRows} rows)`);

			console.log('Running forget cycle - pattern_future reward decay...');
			stepStart = Date.now();
			const [patternRewardResult] = await this.conn.query(`UPDATE pattern_future SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]);
			console.log(`  Pattern_future reward decay took ${Date.now() - stepStart}ms (updated ${patternRewardResult.affectedRows} rows)`);

			// 4. DISHABITUATION: Recover habituation toward 1.0
			// Formula: habituation = habituation + (1.0 - habituation) * dishabituationRate
			// habituation=0.5, rate=0.01 → 0.5 + (1.0-0.5)*0.01 = 0.505
			// habituation=0.9, rate=0.01 → 0.9 + (1.0-0.9)*0.01 = 0.901
			console.log('Running forget cycle - connection dishabituation...');
			stepStart = Date.now();
			const [connDishabResult] = await this.conn.query(`UPDATE connections SET habituation = habituation + (1.0 - habituation) * ? WHERE habituation < 1.0`, [this.dishabituationRate]);
			console.log(`  Connection dishabituation took ${Date.now() - stepStart}ms (updated ${connDishabResult.affectedRows} rows)`);

			console.log('Running forget cycle - pattern_future dishabituation...');
			stepStart = Date.now();
			const [patternDishabResult] = await this.conn.query(`UPDATE pattern_future SET habituation = habituation + (1.0 - habituation) * ? WHERE habituation < 1.0`, [this.dishabituationRate]);
			console.log(`  Pattern_future dishabituation took ${Date.now() - stepStart}ms (updated ${patternDishabResult.affectedRows} rows)`);

			// 5. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
			console.log('Running forget cycle - orphaned neurons cleanup...');
			stepStart = Date.now();
			const [neuronDeleteResult] = await this.conn.query(`
                DELETE
                FROM neurons n
                WHERE NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
				AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
				AND NOT EXISTS (SELECT 1 FROM pattern_past WHERE pattern_neuron_id = n.id)
				AND NOT EXISTS (SELECT 1 FROM pattern_future WHERE pattern_neuron_id = n.id)
				AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
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
				this.conn.query(`UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.connectionForgetRate]),
				this.conn.query(`UPDATE connections SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]),
				this.conn.query(`UPDATE pattern_future SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]),
				// Dishabituation: recover habituation toward 1.0
				this.conn.query(`UPDATE connections SET habituation = habituation + (1.0 - habituation) * ? WHERE habituation < 1.0`, [this.dishabituationRate]),
				this.conn.query(`UPDATE pattern_future SET habituation = habituation + (1.0 - habituation) * ? WHERE habituation < 1.0`, [this.dishabituationRate])
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
                DELETE
                FROM neurons n
                WHERE NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
                AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
                AND NOT EXISTS (SELECT 1 FROM pattern_past WHERE pattern_neuron_id = n.id)
                AND NOT EXISTS (SELECT 1 FROM pattern_future WHERE pattern_neuron_id = n.id)
                AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
			`);
		}
	}

	/**
	 * Apply channel-specific rewards to connections/patterns that led to executed outputs.
	 * Uses multiplicative rewards with exponential temporal decay.
	 * Older connections get less reward/punishment (decay applied to the reward exponent).
	 *
	 * Channel-Specific Credit Assignment:
	 * 1. Identify which channel each base-level output belongs to (via output dimensions)
	 * 2. Use base_inference_sources to find which connections/patterns led to each base output (unpacked from higher levels)
	 * 3. Apply channel-specific reward to the connections/patterns based on source_type
	 *
	 * Multiplicative reward: strength is multiplied by reward with time-decayed effect
	 * Formula: strength * (1 + (reward - 1) * POW(rewardTimeDecayFactor, age - 1))
	 * reward = 1.5, age = 1 → multiply by 1 + 0.5 * 0.9^0 = 1.5 (50% increase)
	 * reward = 1.5, age = 2 → multiply by 1 + 0.5 * 0.9^1 = 1.45 (45% increase)
	 * reward = 0.5, age = 1 → multiply by 1 + (-0.5) * 0.9^0 = 0.5 (50% decrease)
	 * reward = 0.5, age = 2 → multiply by 1 + (-0.5) * 0.9^1 = 0.55 (45% decrease)
	 * This is proportional to existing strength, avoiding saturation issues
	 */
	async applyRewards(channelRewards) {

		// it is not possible to apply rewards until at least we have 3 frames
		// first 2 frames build connections, we infer output in frame 3 using them, we reward them in frame 4
		if (this.frameNumber < 4) return;

		// nothing to update if there are no rewards
		if (channelRewards.size === 0) return;

		// Process each channel's rewards separately
		// Note: we process ALL channels (even neutral rewards) because habituation must be applied
		// whenever an action is executed, regardless of reward value
		let totalConnectionsRewarded = 0;
		let totalPatternsRewarded = 0;
		for (const [channelName, reward] of channelRewards) {

			if (this.debug) console.log(`Applying reward ${reward.toFixed(3)} and habituation for channel: ${channelName}`);

			// Get the output dimension IDs for this channel
			const outputDimIds = this.getChannelOutputDims(channelName);
			if (outputDimIds.length === 0) {
				console.warn(`Warning: No output dimensions found for channel ${channelName}`);
				continue;
			}

			// Reward and habituate connection-based inferences for this channel
			// source_type='connection' covers both regular connection inference and exploration
			const [connResult] = await this.conn.query(`
				UPDATE connections c
				JOIN base_inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
				JOIN coordinates coord ON coord.neuron_id = isrc.base_neuron_id AND coord.dimension_id IN (?)
				SET c.reward = GREATEST(?, LEAST(?, c.reward * (1 + (? - 1) * POW(?, isrc.age - 1)))),
				    c.habituation = c.habituation * ?
				WHERE isrc.age > 0 AND isrc.age <= ?
			`, [
				outputDimIds,
				this.minConnectionReward,
				this.maxConnectionReward,
				reward,
				this.rewardTimeDecayFactor,
				this.habituationDecay,
				this.maxRewardsAge
			]);
			totalConnectionsRewarded += connResult.affectedRows;

			// Reward and habituate pattern-based inferences for this channel
			// source_type='pattern' means source_id is pattern_future.id
			const [patternResult] = await this.conn.query(`
				UPDATE pattern_future pf
				JOIN base_inference_sources isrc ON pf.id = isrc.source_id AND isrc.source_type = 'pattern'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
				JOIN coordinates coord ON coord.neuron_id = isrc.base_neuron_id AND coord.dimension_id IN (?)
				SET pf.reward = GREATEST(?, LEAST(?, pf.reward * (1 + (? - 1) * POW(?, isrc.age - 1)))),
				    pf.habituation = pf.habituation * ?
				WHERE isrc.age > 0 AND isrc.age <= ?
			`, [
				outputDimIds,
				this.minConnectionReward,
				this.maxConnectionReward,
				reward,
				this.rewardTimeDecayFactor,
				this.habituationDecay,
				this.maxRewardsAge
			]);
			totalPatternsRewarded += patternResult.affectedRows;

			if (this.debug) console.log(`  ${channelName}: rewarded ${connResult.affectedRows} connections, ${patternResult.affectedRows} patterns: ${reward}`);
		}

		if (this.debug) console.log(`Total rewarded: ${totalConnectionsRewarded} connections, ${totalPatternsRewarded} patterns`);
		await this.waitForUser('Rewards applied');
	}

	/**
	 * returns channel output dimensions got a given channel name
	 */
	getChannelOutputDims(channelName) {
		const channel = this.channels.get(channelName);
		if (!channel) {
			console.warn(`Warning: No channel found: ${channelName}`);
			return [];
		}
		const outputDimNames = channel.getOutputDimensions();
		return outputDimNames.map(name => this.dimensionNameToId[name]).filter(id => id !== undefined);
	}

	/**
	 * Get detailed inference information for diagnostic output (MySQL implementation)
	 * Uses base_inference_sources table (for rewards tracking)
	 */
	async getInferenceDetails() {

		// Get inferred base neurons with their coordinates
		const [inferences] = await this.conn.query(`
			SELECT inf.neuron_id, inf.strength, c.dimension_id, c.val, d.name as dimension_name
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0 AND inf.level = 0
			ORDER BY inf.neuron_id, d.name
		`);
		if (inferences.length === 0) return [];

		// Group coordinates by neuron_id
		const neuronMap = new Map();
		for (const row of inferences) {
			if (!neuronMap.has(row.neuron_id)) neuronMap.set(row.neuron_id, { neuron_id: row.neuron_id, strength: row.strength, coordinates: {}, sources: [] });
			neuronMap.get(row.neuron_id).coordinates[row.dimension_name] = row.val;
		}

		// Get inference sources for base level neurons
		const neuronIds = [...neuronMap.keys()];

		// Get connection sources
		const [connSources] = await this.conn.query(`
			SELECT isrc.base_neuron_id, c.strength as conn_strength, c.reward as conn_reward, c.habituation as conn_habituation, isrc.inference_strength
			FROM base_inference_sources isrc
			JOIN connections c ON c.id = isrc.source_id
			WHERE isrc.age = 0 AND isrc.source_type = 'connection' AND isrc.base_neuron_id IN (?)
		`, [neuronIds]);

		// Group connection sources by neuron
		const connSourceMap = new Map();
		for (const row of connSources) {
			if (!connSourceMap.has(row.base_neuron_id)) connSourceMap.set(row.base_neuron_id, { type: 'connection', sources: [] });
			connSourceMap.get(row.base_neuron_id).sources.push({
				connection_strength: row.conn_strength,
				connection_reward: row.conn_reward,
				connection_habituation: row.conn_habituation,
				prediction_strength: row.inference_strength
			});
		}

		// Get pattern sources
		const [patternSources] = await this.conn.query(`
			SELECT isrc.base_neuron_id, c.strength as conn_strength, c.reward as conn_reward,
			       c.habituation as conn_habituation, pf.strength as pattern_strength,
			       pf.reward as pattern_reward, pf.habituation as pattern_habituation,
			       isrc.inference_strength
			FROM base_inference_sources isrc
			JOIN pattern_future pf ON pf.id = isrc.source_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE isrc.age = 0 AND isrc.source_type = 'pattern' AND isrc.base_neuron_id IN (?)
		`, [neuronIds]);

		// Group pattern sources by neuron
		const patternSourceMap = new Map();
		for (const row of patternSources) {
			if (!patternSourceMap.has(row.base_neuron_id)) patternSourceMap.set(row.base_neuron_id, { type: 'pattern', sources: [] });
			patternSourceMap.get(row.base_neuron_id).sources.push({
				connection_strength: row.conn_strength,
				connection_reward: row.conn_reward,
				connection_habituation: row.conn_habituation,
				pattern_strength: row.pattern_strength,
				pattern_reward: row.pattern_reward,
				pattern_habituation: row.pattern_habituation,
				prediction_strength: row.inference_strength
			});
		}

		// Attach sources to neurons
		for (const [neuronId, neuron] of neuronMap) {
			if (connSourceMap.has(neuronId)) neuron.sources.push(connSourceMap.get(neuronId));
			if (patternSourceMap.has(neuronId)) neuron.sources.push(patternSourceMap.get(neuronId));
		}

		return Array.from(neuronMap.values());
	}
}