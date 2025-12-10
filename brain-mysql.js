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
			'inference_sources',
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
			'inference_sources',
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
		await this.conn.query('UPDATE inference_sources SET age = age + 1 ORDER BY age DESC');

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
		const [sourcesResult] = await this.conn.query('DELETE FROM inference_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${sourcesResult.affectedRows} aged-out inference sources (age >= ${this.baseNeuronMaxAge})`);
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
	 * Check if a channel needs exploration (MySQL implementation)
	 * Uses probabilistic exploration inversely proportional to total inference strength.
	 * Higher confidence predictions = lower exploration probability.
	 * @param {string} channelName - name of the channel to check
	 * @param {string} actionNeuronId - action neuron id - used to check if it's already inferred or not
	 * @returns {Promise<boolean>} - true if channel needs exploration
	 */
	async channelNeedsExploration(channelName, actionNeuronId) {
		const channel = this.channels.get(channelName);
		if (!channel) return false;

		const outputDimNames = channel.getOutputDimensions();
		if (outputDimNames.length === 0) return false;

		// Get dimension IDs for this channel's output dimensions
		const outputDimIds = outputDimNames.map(name => this.dimensionNameToId[name]).filter(id => id !== undefined);
		if (outputDimIds.length === 0) return false;

		// Get total inference strength for output dimensions AND check if action neuron is among them
		const [result] = await this.conn.query(`
            SELECT COALESCE(SUM(inf.strength), 0) as total_strength, MAX(IF(inf.neuron_id = ?, 1, 0)) as action_neuron_inferred
            FROM inferred_neurons inf
            WHERE inf.age = 0
            AND inf.level = 0
            AND EXISTS (SELECT 1 FROM coordinates c WHERE c.neuron_id = inf.neuron_id AND c.dimension_id IN (?))
		`, [actionNeuronId, outputDimIds]);
		const totalInferenceStrength = result[0].total_strength || 0;
		const actionNeuronInferred = result[0].action_neuron_inferred === 1;

		// If action neuron is already inferred, no need for exploration
		if (actionNeuronInferred) return false;

		// If no outputs at all, must explore
		if (totalInferenceStrength === 0) return true;

		// Probabilistic exploration inversely proportional to confidence
		return this.decideExploration(totalInferenceStrength);
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
			await this.conn.query('DELETE FROM inference_sources WHERE base_neuron_id IN (?) AND age = 0', [deletedNeuronIds]);
		}

		// Insert corrected inferences (if any)
		if (resolvedInferences.length > 0)
			await this.conn.query(
				'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ? ON DUPLICATE KEY UPDATE strength = VALUES(strength)',
				[resolvedInferences.map(p => [p.neuron_id, 0, 0, p.strength])]
			);
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
	 * Weakens connections that made incorrect predictions.
	 * Uses inference_sources to find which connections made predictions.
	 */
	async negativeReinforceConnections() {

		// Apply negative reinforcement to failed INPUT predictions only - Failed = predicted but not observed
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN inference_sources isrc ON isrc.source_id = c.id AND isrc.source_type = 'connection'
			-- only weaken connections whose predictions made it through conflict resolution
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE isrc.age = 1
			AND c.strength > 0
			-- outputs are not penalized (they are actions, not predictions of world state)
			AND EXISTS (
				SELECT 1 FROM coordinates coord
				JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = isrc.base_neuron_id
				AND d.type != 'action'
			)
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = isrc.base_neuron_id
				AND an.level = 0
				AND an.age = 0
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
	 * Collects connection sources in memory, then unpacks to base level and saves to inference_sources.
	 */
	async inferConnectionsAtLevel(level) {

		// Get all connection predictions at this level
		const [predictions] = await this.conn.query(`
			SELECT c.to_neuron_id, c.id as connection_id, c.strength * c.reward * c.habituation * POW(?, c.distance - 1) as prediction_strength
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, level]);
		if (this.debug && predictions.length > 0) console.log(`Level ${level}: Connection inference found ${predictions.length} predictions`);
		if (predictions.length === 0) return 0;

		// Build in-memory map: neuron_id → [{source_type, source_id, strength}]
		const neuronSources = new Map();
		for (const pred of predictions) {
			if (!neuronSources.has(pred.to_neuron_id)) neuronSources.set(pred.to_neuron_id, []);
			neuronSources.get(pred.to_neuron_id).push({
				source_type: 'connection',
				source_id: pred.connection_id,
				strength: pred.prediction_strength
			});
		}

		// Aggregate strengths per neuron and filter by threshold
		// Build a filtered map with only neurons that pass the threshold
		const filteredNeuronSources = new Map();
		const aggregatedNeurons = [];
		for (const [neuronId, sources] of neuronSources) {
			const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
			if (totalStrength >= this.minInferenceStrength) {
				aggregatedNeurons.push({ neuron_id: neuronId, strength: totalStrength });
				filteredNeuronSources.set(neuronId, sources);
			}
		}
		if (aggregatedNeurons.length === 0) return 0;

		// Insert into inferred_neurons
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ?',
			[aggregatedNeurons.map(n => [n.neuron_id, level, 0, n.strength])]
		);
		if (this.debug) console.log(`Level ${level}: Connection inference predicted ${aggregatedNeurons.length} neurons`);

		// Unpack to base level and save inference sources (only for neurons that passed threshold)
		await this.unpackAndSaveInferenceSources(level, filteredNeuronSources);

		return aggregatedNeurons.length;
	}

	/**
	 * Save exploration inference to inferred_neurons and inference_sources (MySQL implementation)
	 * Called when exploration generates an action without prior inference
	 * Uses overrideInference with null originalNeuronId to find and save connection sources
	 * @param {Number} actionNeuronId - neuron ID for the exploration action
	 */
	async saveExplorationInference(actionNeuronId) {
		// Use overrideInference with null original - it will:
		// 1. Find connections that could have predicted this action
		// 2. Save them to inference_sources with source_type='connection'
		// 3. Insert into inferred_neurons
		await this.overrideInference(actionNeuronId, null);
	}

	/**
	 * Override an inference with a new neuron (unified method for correction and exploration)
	 * Both correction and exploration are override mechanisms - the only difference is:
	 * - Correction: interprets existing inference to make it valid
	 * - Exploration: randomly selects a valid action
	 *
	 * This method:
	 * 1. Cleans up inference_sources for original neuron (if any)
	 * 2. Finds connections that could have predicted the override neuron
	 * 3. Inserts override sources into inference_sources
	 * 4. Updates inferred_neurons (removes original, adds new)
	 *
	 * @param {Number} overrideNeuronId - neuron ID to use instead
	 * @param {Number|null} originalNeuronId - neuron ID being overridden (null for pure exploration)
	 */
	async overrideInference(overrideNeuronId, originalNeuronId = null) {

		// 1. Clean up inference sources for original neuron (if any)
		if (originalNeuronId)
			await this.conn.query('DELETE FROM inference_sources WHERE base_neuron_id = ? AND age = 0', [originalNeuronId]);

		// Also clean up any existing sources for the override neuron (in case it was already inferred)
		await this.conn.query('DELETE FROM inference_sources WHERE base_neuron_id = ? AND age = 0', [overrideNeuronId]);

		// 2. Find connections that could have predicted the override neuron and insert as sources
		const [result] = await this.conn.query(`
			INSERT INTO inference_sources (age, base_neuron_id, source_type, source_id, inference_strength)
			SELECT 0, c.to_neuron_id, 'connection', c.id, c.strength * c.reward * c.habituation * POW(?, c.distance - 1)
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = 0
			AND c.to_neuron_id = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, overrideNeuronId]);

		if (this.debug && result.affectedRows > 0) console.log(`  Found ${result.affectedRows} possible connection sources for override`);

		// 3. Update inferred_neurons (delete original, insert new)
		if (originalNeuronId) await this.conn.query('DELETE FROM inferred_neurons WHERE neuron_id = ? AND age = 0 AND level = 0', [originalNeuronId]);
		await this.conn.query('INSERT IGNORE INTO inferred_neurons (neuron_id, level, age, strength) VALUES (?, 0, 0, ?)', [overrideNeuronId, 1000000]);
	}

	/**
	 * Correct inferred actions when conflict resolution changes them (MySQL implementation)
	 * Correction is an override mechanism that interprets existing connection/pattern inferences
	 * @param {Array} corrections - Array of { originalNeuronId, correctedCoordinates }
	 */
	async correctInferredActions(corrections) {
		if (!corrections || corrections.length === 0) return;

		for (const correction of corrections) {
			const { originalNeuronId, correctedCoordinates } = correction;

			// Find or create neuron for corrected action
			const [correctedNeuronId] = await this.getFrameNeurons([correctedCoordinates]);

			if (this.debug) console.log(`Correcting inference: neuron ${originalNeuronId} → ${correctedNeuronId}`);

			// Use unified override method
			await this.overrideInference(correctedNeuronId, originalNeuronId);
		}
	}

	/**
	 * Merge pattern_future with observed connections FROM the peak.
	 * Called during learning phase after pattern inference from previous frame.
	 * Uses inference_sources with source_type='pattern' to know which patterns made predictions.
	 *
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future connections that were correctly predicted (predicted AND observed)
	 * 2. Negative: Weaken pattern_future connections that were incorrectly predicted (predicted but NOT observed)
	 * 3. Novel: Add new connections FROM peak that were observed but not predicted
	 */
	async mergePatternFuture() {

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted connections
		// Connection in pattern_future AND observed in active_connections FROM peak at age=0
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN inference_sources isrc ON isrc.source_id = pf.id AND isrc.source_type = 'pattern'
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.connection_id = pf.connection_id AND ac.level = 0 AND ac.age = 0
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength + 1))
			WHERE isrc.age = 1
		`, [this.minConnectionStrength, this.maxConnectionStrength]);
		if (this.debug && strengthenResult.affectedRows > 0)
			console.log(`Strengthened ${strengthenResult.affectedRows} correct pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections
		// Connection in pattern_future but NOT observed in active_connections FROM peak
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN inference_sources isrc ON isrc.source_id = pf.id AND isrc.source_type = 'pattern'
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.connection_id = pf.connection_id AND ac.level = 0 AND ac.age = 0
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength - ?))
			WHERE isrc.age = 1
			AND ac.connection_id IS NULL
		`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug && weakenResult.affectedRows > 0)
			console.log(`Weakened ${weakenResult.affectedRows} failed pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Observed but not predicted
		// Active connections FROM peaks of patterns that made predictions, but not in pattern_future
		// Only for patterns whose predictions were strong enough to be inferred and made it through conflict resolution
		const [novelResult] = await this.conn.query(`
			INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pf_src.pattern_neuron_id, ac.connection_id, 1.0
			FROM inference_sources isrc
			JOIN pattern_future pf_src ON pf_src.id = isrc.source_id AND isrc.source_type = 'pattern'
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.base_neuron_id AND inf.level = 0 AND inf.age = isrc.age
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf_src.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.level = 0 AND ac.age = 0
			LEFT JOIN pattern_future pf ON pf.pattern_neuron_id = pf_src.pattern_neuron_id AND pf.connection_id = ac.connection_id
			WHERE isrc.age = 1
			AND pf.connection_id IS NULL
		`);
		if (this.debug && novelResult.affectedRows > 0)
			console.log(`Added ${novelResult.affectedRows} novel connections to pattern_future`);
	}

	/**
	 * Get the prediction level from previous frame's inference.
	 * Returns null if no inference occurred (only exploration).
	 * With unified inference_sources, we check inferred_neurons at age=1.
	 */
	async getPreviousInferenceLevel() {

		// Get prediction level from inferred_neurons (age=1 means previous frame)
		const [levelResult] = await this.conn.query(`
			SELECT MAX(level) as level FROM inferred_neurons WHERE age = 1
		`);

		return levelResult[0] && levelResult[0].level !== null ? levelResult[0].level : null;
	}

	/**
	 * Pattern inference from a source level (MySQL implementation)
	 * Collects pattern sources in memory, then unpacks to base level and saves to inference_sources.
	 * Returns count of predictions made.
	 */
	async inferPatternsFromLevel(sourceLevel) {

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
		if (predictions.length === 0) return 0;

		// Build in-memory map: neuron_id → [{source_type, source_id, strength}]
		const neuronSources = new Map();
		for (const pred of predictions) {
			if (!neuronSources.has(pred.to_neuron_id)) neuronSources.set(pred.to_neuron_id, []);
			neuronSources.get(pred.to_neuron_id).push({
				source_type: 'pattern',
				source_id: pred.pattern_future_id,
				strength: pred.prediction_strength
			});
		}

		// Aggregate strengths per neuron and filter by threshold
		// Build a filtered map with only neurons that pass the threshold
		const filteredNeuronSources = new Map();
		const aggregatedNeurons = [];
		for (const [neuronId, sources] of neuronSources) {
			const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
			if (totalStrength >= this.minInferenceStrength) {
				aggregatedNeurons.push({ neuron_id: neuronId, strength: totalStrength });
				filteredNeuronSources.set(neuronId, sources);
			}
		}
		if (aggregatedNeurons.length === 0) return 0;

		// Insert into inferred_neurons at level sourceLevel - 1 (patterns predict one level down)
		const targetLevel = sourceLevel - 1;
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ?',
			[aggregatedNeurons.map(n => [n.neuron_id, targetLevel, 0, n.strength])]
		);
		if (this.debug) console.log(`Level ${sourceLevel}: Pattern inference predicted ${aggregatedNeurons.length} neurons at level ${targetLevel}`);

		// Unpack to base level and save inference sources (only for neurons that passed threshold)
		await this.unpackAndSaveInferenceSources(targetLevel, filteredNeuronSources);

		return aggregatedNeurons.length;
	}

	/**
	 * Unpack predictions from higher level to base level via peak chain (in-memory implementation)
	 * Follows pattern_neuron → peak_neuron → peak_neuron down to base.
	 * Tracks which sources (connection_id or pattern_future_id) led to which base outputs.
	 * @param {Number} fromLevel - the level where predictions were made
	 * @param {Map} neuronSources - Map of neuron_id → [{source_type, source_id, strength}]
	 */
	async unpackAndSaveInferenceSources(fromLevel, neuronSources) {
		if (this.debug2) console.log(`Unpacking inference sources from level ${fromLevel} to base`);

		// If already at base level, just save directly
		if (fromLevel === 0) {
			await this.saveInferenceSourcesToDb(neuronSources);
			return;
		}

		// Unpack level by level
		let currentLevelSources = neuronSources;

		for (let level = fromLevel; level > 0; level--) {
			// Get all neuron IDs at current level
			const neuronIds = [...currentLevelSources.keys()];
			if (neuronIds.length === 0) break;

			// Query pattern_peaks to get peak_neuron_id for each pattern_neuron_id
			const [peaks] = await this.conn.query(
				'SELECT pattern_neuron_id, peak_neuron_id FROM pattern_peaks WHERE pattern_neuron_id IN (?)',
				[neuronIds]
			);

			// Build next level's mapping - peaks inherit sources from parent patterns
			const nextLevelSources = new Map();
			for (const peak of peaks) {
				const sources = currentLevelSources.get(peak.pattern_neuron_id);
				if (!sources) continue;

				if (!nextLevelSources.has(peak.peak_neuron_id))
					nextLevelSources.set(peak.peak_neuron_id, []);
				nextLevelSources.get(peak.peak_neuron_id).push(...sources);
			}

			// Also insert unpacked neurons into inferred_neurons at level-1
			if (nextLevelSources.size > 0) {
				const unpackedNeurons = [];
				for (const [neuronId, sources] of nextLevelSources) {
					const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
					unpackedNeurons.push([neuronId, level - 1, 0, totalStrength]);
				}
				await this.conn.query(
					'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ? ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)',
					[unpackedNeurons]
				);
			}

			currentLevelSources = nextLevelSources;
		}

		// Now at base level - save to inference_sources
		await this.saveInferenceSourcesToDb(currentLevelSources);
	}

	/**
	 * Save inference sources to the database
	 * Deduplicates and aggregates sources that reach the same base neuron via multiple paths
	 * @param {Map} neuronSources - Map of base_neuron_id → [{source_type, source_id, strength}]
	 */
	async saveInferenceSourcesToDb(neuronSources) {
		// Dedupe and aggregate (same source can reach same base via multiple paths)
		const deduped = new Map();
		for (const [baseNeuronId, sources] of neuronSources) {
			for (const src of sources) {
				const key = `${baseNeuronId}:${src.source_type}:${src.source_id}`;
				if (deduped.has(key))
					deduped.get(key).strength += src.strength;
				else
					deduped.set(key, { base_neuron_id: baseNeuronId, ...src });
			}
		}

		if (deduped.size === 0) return;

		// Batch insert into inference_sources
		const rows = [...deduped.values()].map(s =>
			[0, s.base_neuron_id, s.source_type, s.source_id, s.strength]
		);
		await this.conn.query(
			'INSERT INTO inference_sources (age, base_neuron_id, source_type, source_id, inference_strength) VALUES ?',
			[rows]
		);
		if (this.debug) console.log(`Saved ${rows.length} inference sources to database`);
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
	 * 2. Use inference_sources to find which connections/patterns made predictions for each base output
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
				JOIN inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
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
				JOIN inference_sources isrc ON pf.id = isrc.source_id AND isrc.source_type = 'pattern'
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
	 * Uses the unified inference_sources table
	 */
	async getInferenceDetails(level) {

		// Get inferred neurons with their coordinates
		const [inferences] = await this.conn.query(`
			SELECT inf.neuron_id, inf.strength, c.dimension_id, c.val, d.name as dimension_name
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0 AND inf.level = ?
			ORDER BY inf.neuron_id, d.name
		`, [level]);

		if (inferences.length === 0) return [];

		// Group coordinates by neuron_id
		const neuronMap = new Map();
		for (const row of inferences) {
			if (!neuronMap.has(row.neuron_id)) {
				neuronMap.set(row.neuron_id, {
					neuron_id: row.neuron_id,
					strength: row.strength,
					coordinates: {},
					sources: []
				});
			}
			neuronMap.get(row.neuron_id).coordinates[row.dimension_name] = row.val;
		}

		// Get inference sources for base level neurons
		const neuronIds = [...neuronMap.keys()];
		if (neuronIds.length === 0) return Array.from(neuronMap.values());

		// Get connection sources
		const [connSources] = await this.conn.query(`
			SELECT isrc.base_neuron_id, c.strength as conn_strength, c.reward as conn_reward,
			       c.habituation as conn_habituation, isrc.inference_strength
			FROM inference_sources isrc
			JOIN connections c ON c.id = isrc.source_id
			WHERE isrc.age = 0 AND isrc.source_type = 'connection' AND isrc.base_neuron_id IN (?)
		`, [neuronIds]);

		// Group connection sources by neuron
		const connSourceMap = new Map();
		for (const row of connSources) {
			if (!connSourceMap.has(row.base_neuron_id)) {
				connSourceMap.set(row.base_neuron_id, {
					type: 'connection',
					sources: []
				});
			}
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
			FROM inference_sources isrc
			JOIN pattern_future pf ON pf.id = isrc.source_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE isrc.age = 0 AND isrc.source_type = 'pattern' AND isrc.base_neuron_id IN (?)
		`, [neuronIds]);

		// Group pattern sources by neuron
		const patternSourceMap = new Map();
		for (const row of patternSources) {
			if (!patternSourceMap.has(row.base_neuron_id)) {
				patternSourceMap.set(row.base_neuron_id, {
					type: 'pattern',
					sources: []
				});
			}
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