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
			'inferred_actions',
			'matched_patterns',
			'matched_pattern_connections',
			'active_connections',
			'inference_sources',
			'action_sources',
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
			'inferred_actions',
			'active_connections',
			'matched_patterns',
			'matched_pattern_connections',
			'inference_sources',
			'action_sources',
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
		await this.conn.query('UPDATE inferred_actions SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE action_sources SET age = age + 1 ORDER BY age DESC');

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

		// Clean up inferred actions after execution
		const [actionsResult] = await this.conn.query('DELETE FROM inferred_actions WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${actionsResult.affectedRows} executed inferred actions (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out inference sources (same lifecycle as neurons)
		const [infSourcesResult] = await this.conn.query('DELETE FROM inference_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		const [actionSourcesResult] = await this.conn.query('DELETE FROM action_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${infSourcesResult.affectedRows} aged-out inference sources, ${actionSourcesResult.affectedRows} action sources (age >= ${this.baseNeuronMaxAge})`);
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
	 * Get frame outputs for all channels from inferred_actions table (MySQL implementation)
	 * Reads output neurons (age=0, level=0) grouped by channel
	 * @returns {Promise<Map>} - Map of channel names to array of output coordinates
	 */
	async getFrameOutputs() {

		// Get all output neurons from inferred_actions table in one query
		const [rows] = await this.conn.query(`
			SELECT inf.neuron_id, c.dimension_id, c.val, d.name as dimension_name, d.channel
			FROM inferred_actions inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0
			ORDER BY d.channel, inf.neuron_id
		`);

		// Group by channel, then by neuron_id to build complete output objects
		const channelOutputs = new Map();
		for (const row of rows) {

			// Initialize channel map if needed
			if (!channelOutputs.has(row.channel)) channelOutputs.set(row.channel, new Map());
			const neuronMap = channelOutputs.get(row.channel);

			// Initialize neuron coordinates if needed
			if (!neuronMap.has(row.neuron_id)) neuronMap.set(row.neuron_id, {});
			neuronMap.get(row.neuron_id)[row.dimension_name] = row.val;
		}

		// Convert neuron maps to arrays of coordinate objects
		for (const [channel, neuronMap] of channelOutputs)
			channelOutputs.set(channel, Array.from(neuronMap.values()));

		return channelOutputs;
	}

	/**
	 * Apply negative reinforcement to failed connection predictions.
	 * Weakens connections that made incorrect predictions
	 * Uses inference_sources to find which connections made predictions.
	 */
	async negativeReinforceConnections() {

		// Apply negative reinforcement to failed event predictions - Failed = predicted but not observed
		const [result] = await this.conn.query(`
			UPDATE connections c
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE c.strength > 0
			-- penalize connections that were inferred in the previous frame
			AND c.id IN (SELECT source_id FROM inference_sources WHERE source_type = 'connection' AND age = 1)
			-- penalize the connections that did not come true
			AND c.id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
		`, [this.connectionNegativeReinforcement]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Weakened ${result.affectedRows} failed event predictions`);
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
	 * Connection inference at a specific level (MySQL implementation).
	 * Returns raw inferences with separate strength and reward values.
	 * @returns {Promise<Array|null>} Array of raw inference objects or null if no predictions
	 */
	async inferConnections(level) {

		// Get all connection inferences at this level with separate strength and reward
		const [inferences] = await this.conn.query(`
			SELECT c.to_neuron_id, c.id as source_id, c.strength * POW(?, c.distance - 1) as prediction_strength, c.reward
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = ?
			AND c.distance = an.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, level]);

		if (!inferences || inferences.length === 0) return null;

		if (this.debug) console.log(`Level ${level}: Connection inference found ${inferences.length} raw predictions`);

		// Return raw inferences with source_type added
		return inferences.map(inf => ({
			neuron_id: inf.to_neuron_id,
			level,
			source_type: 'connection',
			source_id: inf.source_id,
			strength: inf.prediction_strength,
			reward: inf.reward
		}));
	}

	/**
	 * Pattern inference from a source level (MySQL implementation)
	 * Returns raw inferences with separate strength and reward values.
	 * @returns {Promise<Array|null>} Array of raw inference objects or null if no predictions
	 **/
	async inferPatterns(sourceLevel) {

		// Get all pattern predictions at this level with separate strength and reward
		// strength = base prediction strength (without reward)
		const [inferences] = await this.conn.query(`
			SELECT c.to_neuron_id, pf.id as source_id, pf.strength * POW(?, c.distance - 1) as prediction_strength, pf.reward as reward
			FROM active_neurons an
			JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE an.level = ?
			AND c.distance = an.age + 1
			AND pf.strength > 0
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, sourceLevel]);

		if (!inferences || inferences.length === 0) return null;

		if (this.debug) console.log(`Level ${sourceLevel}: Pattern inference found ${inferences.length} raw predictions`);

		// Return raw inferences with source_type added
		return inferences.map(inf => ({
			neuron_id: inf.to_neuron_id,
			level: sourceLevel - 1,
			source_type: 'pattern',
			source_id: inf.source_id,
			strength: inf.prediction_strength,
			reward: inf.reward
		}));
	}

	/**
	 * Unpack aggregated inferences to base level without saving intermediate levels.
	 * Recursively follows pattern_peaks down to base level.
	 * @param {Map} aggregatedInferences - Map from aggregateInferences() at any level
	 * @returns {Promise<Map>} Map of base_neuron_id → {neuron_id, strength, reward, sources: [{source_type, source_id, strength, reward}]}
	 */
	async unpackToBase(aggregatedInferences) {
		if (this.debug) console.log(`Unpacking ${aggregatedInferences.size} inferences`);

		// Convert aggregated inferences to sources format
		let currentLevelSources = new Map();
		for (const [neuronId, neuron] of aggregatedInferences) currentLevelSources.set(neuronId, neuron.sources);

		// get peaks in the lower level from current level sources that lead to the current level's patterns
		let peaks = await this.getPeaksFromSources(currentLevelSources);

		// continue level by level until we reach the base level (no more peaks)
		while (peaks.length > 0) {

			// build next level sources (peaks inherit sources from parent patterns) and move down to next level
			currentLevelSources = this.buildNextLevelSources(peaks, currentLevelSources);

			// get peaks in the lower level from current level sources that lead to the current level's patterns
			peaks = await this.getPeaksFromSources(currentLevelSources);
		}

		// Convert base level sources back to aggregated format
		const baseInferences = new Map();
		for (const [neuronId, sources] of currentLevelSources) {
			const totalStrength = sources.reduce((sum, s) => sum + s.strength, 0);
			const avgReward = sources.reduce((sum, s) => sum + s.reward * s.strength, 0) / totalStrength;
			baseInferences.set(neuronId, { neuron_id: neuronId, strength: totalStrength, reward: avgReward, sources });
		}

		if (this.debug) console.log(`Unpacked to ${baseInferences.size} base level neurons`);
		return baseInferences;
	}

	/**
	 * Saves inferences to inferred_neurons and inference_sources (for pattern learning/negative reinforcement).
	 */
	async saveInferences(inferences) {
		const neurons = [];
		const sources = new Map();
		for (const [neuronId, neuron] of inferences) {
			neurons.push([neuronId, neuron.level, 0, neuron.strength, neuron.reward]);
			sources.set(neuronId, neuron.sources || []);
		}
		await this.insertInferredNeurons(neurons);
		await this.saveInferenceSources(sources);
		if (this.debug) console.log(`Inferences: Saved ${neurons.length} neurons`);
	}

	/**
	 * Get peaks for pattern neurons from current level sources
	 * @param {Map} currentLevelSources - Map of neuron_id → [{source_type, source_id, strength}]
	 * @returns {Promise<Array>} Array of {pattern_neuron_id, peak_neuron_id}
	 */
	async getPeaksFromSources(currentLevelSources) {
		const neuronIds = [...currentLevelSources.keys()];
		const [peaks] = await this.conn.query('SELECT pattern_neuron_id, peak_neuron_id FROM pattern_peaks WHERE pattern_neuron_id IN (?)', [neuronIds]);
		return peaks;
	}

	/**
	 * Build next level's mapping - peaks inherit sources from parent patterns
	 * If we inferred a pattern neuron at a higher level, we will connect it to the outputs via peaks
	 * @param {Array} peaks - Array of {pattern_neuron_id, peak_neuron_id}
	 * @param {Map} currentLevelSources - Map of neuron_id → [{source_type, source_id, strength, reward}]
	 * @returns {Map} Map of peak_neuron_id → [{source_type, source_id, strength, reward}]
	 */
	buildNextLevelSources(peaks, currentLevelSources) {
		const nextLevelSources = new Map();
		for (const peak of peaks) {
			const sources = currentLevelSources.get(peak.pattern_neuron_id);
			if (!sources) throw new Error(`Cannot get next level sources for ${peak.pattern_neuron_id} because it has no sources`);
			if (!nextLevelSources.has(peak.peak_neuron_id)) nextLevelSources.set(peak.peak_neuron_id, []);
			nextLevelSources.get(peak.peak_neuron_id).push(...sources);
		}
		return nextLevelSources;
	}

	/**
	 * Insert inferred neurons into the database
	 * @param {Array} neurons - Array of [neuron_id, level, age, strength, reward]
	 * @param {String} duplicateMode - How to handle duplicates: 'ignore', 'replace', 'add'
	 */
	async insertInferredNeurons(neurons, duplicateMode = 'none') {
		if (neurons.length === 0) return;

		let query = 'INSERT INTO inferred_neurons (neuron_id, level, age, strength, reward) VALUES ?';
		if (duplicateMode === 'ignore')
			query += ' ON DUPLICATE KEY UPDATE strength = strength, reward = reward';
		else if (duplicateMode === 'replace')
			query += ' ON DUPLICATE KEY UPDATE strength = VALUES(strength), reward = VALUES(reward)';
		else if (duplicateMode === 'add')
			query += ' ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength), reward = (reward * strength + VALUES(reward) * VALUES(strength)) / (strength + VALUES(strength))';

		await this.conn.query(query, [neurons]);
	}

	/**
	 * Deduplicate and aggregate inference sources
	 * Same source can reach same neuron via multiple paths through the pattern hierarchy
	 * @param {Map} neuronSources - Map of neuron_id → [{source_type, source_id, strength, reward}]
	 * @param {String} neuronKey - ase_neuron_id for base level, inferred_neuron_id for original level inferences
	 * @returns {Map} Map of unique key → {neuron_id, source_type, source_id, strength, reward}
	 */
	deduplicateInferenceSources(neuronSources, neuronKey) {
		const deduped = new Map();
		for (const [neuronId, sources] of neuronSources) {
			for (const src of sources) {
				const key = `${neuronId}:${src.source_type}:${src.source_id}`;
				if (deduped.has(key)) {
					const existing = deduped.get(key);
					const totalStrength = existing.strength + src.strength;
					existing.reward = (existing.reward * existing.strength + src.reward * src.strength) / totalStrength;
					existing.strength = totalStrength;
				}
				else deduped.set(key, { [neuronKey]: neuronId, ...src });
			}
		}
		return deduped;
	}

	/**
	 * Save inference sources to the database.
	 * @param {Map} sources - Map of neuron_id → [{source_type, source_id, strength}]
	 */
	async saveInferenceSources(sources) {

		// Dedupe and aggregate (same source can reach same neuron via multiple paths)
		const deduped = this.deduplicateInferenceSources(sources, 'neuron_id');
		if (deduped.size === 0) return;

		// Batch insert into inference_sources
		const rows = [...deduped.values()].map(s => [0, s.neuron_id, s.source_type, s.source_id, s.strength]);
		await this.conn.query('INSERT INTO inference_sources (age, neuron_id, source_type, source_id, inference_strength) VALUES ?', [rows]);
		if (this.debug2) console.log(`Saved ${rows.length} inference sources`);
	}

	/**
	 * Saves actions to inferred_actions and action_sources (for pattern learning/negative reinforcement).
	 */
	async saveActions(actions) {
		const neurons = [];
		const sources = new Map();
		for (const [neuronId, neuron] of actions) {
			neurons.push([neuronId, 0, neuron.strength, neuron.reward]);
			sources.set(neuronId, neuron.sources || []);
		}
		await this.insertInferredActions(neurons);
		await this.saveActionSources(sources);
		if (this.debug) console.log(`Actions: Saved ${neurons.length} neurons`);
	}

	/**
	 * Insert inferred actions into the database
	 * @param {Array} actions - Array of [neuron_id, level, age, strength, reward]
	 */
	async insertInferredActions(actions) {
		if (actions.length === 0) return;
		const query = 'INSERT INTO inferred_actions (neuron_id, age, strength, reward) VALUES ?';
		await this.conn.query(query, [actions]);
	}

	/**
	 * Saves action inferences to action_sources (for rewards). Used by applyRewards to reward sources that led to actions.
	 * @param {Map} sources - Map of action_neuron_id → [{source_type, source_id, strength, reward}]
	 */
	async saveActionSources(sources) {
		if (this.debug) console.log(`Actions: Saving ${sources.size} action sources`, sources);

		// Dedupe and aggregate (same source can reach same action via multiple paths)
		const deduped = this.deduplicateInferenceSources(sources, 'action_neuron_id');
		if (deduped.size === 0) return;

		// Batch insert into action_sources
		const rows = [...deduped.values()].map(s => [0, s.action_neuron_id, s.source_type, s.source_id, s.strength, s.reward]);
		await this.conn.query('INSERT INTO action_sources (age, action_neuron_id, source_type, source_id, inference_strength, reward) VALUES ?', [rows]);
		if (this.debug2) console.log(`Saved ${rows.length} action sources`);
	}

	/**
	 * Get connections that could have predicted an exploration neuron.
	 * Used for learning from exploration actions.
	 *
	 * For exploration, we look for EXISTING connections from active neurons to the exploration neuron.
	 * These connections may have been created in previous frames when this action was taken before.
	 * If no connections exist yet (first time taking this action), returns empty - the connections
	 * will be created when the action is observed, and can be rewarded in future frames.
	 *
	 * @param {Number} neuronId - exploration neuron ID
	 * @returns {Promise<Array>} Array of {source_type, source_id, strength, reward}
	 */
	async getExplorationSources(neuronId) {
		// Look for existing connections from active neurons to the exploration neuron
		// These exist if this action was taken before and connections were learned
		const [rows] = await this.conn.query(`
			SELECT 'connection' as source_type, c.id as source_id, c.strength * POW(?, c.distance - 1) as strength, c.reward
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE an.level = 0 
			AND c.to_neuron_id = ? 
			AND c.distance = an.age + 1 
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, neuronId]);

		return rows;
	}

	/**
	 * Merge pattern_future with observed connections FROM the peak.
	 * Called during learning phase after pattern inference from previous frame.
	 * Uses inference_sources with source_type='pattern' to know which patterns made predictions.
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
			AND id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
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
            AND id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
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
			FROM inference_sources isrc
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
	 * Checks inferred_neurons at age=1 to find the level where inference was made.
	 */
	async getPreviousInferenceLevel() {
		const [levelResult] = await this.conn.query(`
			SELECT MAX(level) as level FROM inferred_neurons WHERE age = 1
		`);
		return levelResult[0]?.level ?? null;
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
	 * Get coordinates for a list of neuron IDs with dimension info
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @returns {Promise<Map>} Map of neuron_id → Map of dimension_name → {type, value, channel}
	 */
	async getNeuronCoordinates(neuronIds) {
		if (neuronIds.length === 0) return new Map();

		const [rows] = await this.conn.query(`
			SELECT c.neuron_id, c.val, d.name as dimension_name, d.type, d.channel
			FROM coordinates c
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE c.neuron_id IN (?)
		`, [neuronIds]);

		const neuronCoords = new Map();
		for (const row of rows) {
			if (!neuronCoords.has(row.neuron_id)) neuronCoords.set(row.neuron_id, new Map());
			neuronCoords.get(row.neuron_id).set(row.dimension_name, { type: row.type, value: row.val, channel: row.channel });
		}

		return neuronCoords;
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

		// track time spent for the forget cycle
		const cycleStart = Date.now();
		if (this.debug) console.log('=== FORGET CYCLE STARTING ===');

		// 1. PATTERN FORGETTING: Reduce pattern strengths and remove dead patterns (clamped between minConnectionStrength and maxConnectionStrength)
		if (this.debug) console.log('Running forget cycle - pattern_past update...');
		let stepStart = Date.now();
		const [patternPastUpdateResult] = await this.conn.query(`UPDATE pattern_past SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
		if (this.debug) console.log(`  Pattern_past UPDATE took ${Date.now() - stepStart}ms (updated ${patternPastUpdateResult.affectedRows} rows)`);

		if (this.debug) console.log('Running forget cycle - pattern_future update...');
		stepStart = Date.now();
		const [patternFutureUpdateResult] = await this.conn.query(`UPDATE pattern_future SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
		if (this.debug) console.log(`  Pattern_future UPDATE took ${Date.now() - stepStart}ms (updated ${patternFutureUpdateResult.affectedRows} rows)`);

		if (this.debug) console.log('Running forget cycle - pattern_peaks update...');
		stepStart = Date.now();
		const [patternPeaksUpdateResult] = await this.conn.query(`UPDATE pattern_peaks SET strength = GREATEST(0, strength - ?) WHERE strength > 0`, [this.patternForgetRate]);
		if (this.debug) console.log(`  Pattern_peaks UPDATE took ${Date.now() - stepStart}ms (updated ${patternPeaksUpdateResult.affectedRows} rows)`);

		// Delete patterns with zero strength
		if (this.debug) console.log('Running forget cycle - pattern deletion...');
		stepStart = Date.now();
		const [patternPastDeleteResult] = await this.conn.query(`DELETE FROM pattern_past WHERE strength = ?`, [this.minConnectionStrength]);
		if (this.debug) console.log(`  Pattern_past DELETE took ${Date.now() - stepStart}ms (deleted ${patternPastDeleteResult.affectedRows} rows)`);

		stepStart = Date.now();
		const [patternFutureDeleteResult] = await this.conn.query(`DELETE FROM pattern_future WHERE strength = ?`, [this.minConnectionStrength]);
		if (this.debug) console.log(`  Pattern_future DELETE took ${Date.now() - stepStart}ms (deleted ${patternFutureDeleteResult.affectedRows} rows)`);

		stepStart = Date.now();
		const [patternPeaksDeleteResult] = await this.conn.query(`DELETE FROM pattern_peaks WHERE strength <= 0`);
		if (this.debug) console.log(`  Pattern_peaks DELETE took ${Date.now() - stepStart}ms (deleted ${patternPeaksDeleteResult.affectedRows} rows)`);

		// 2. CONNECTION FORGETTING: Reduce connection strengths and remove dead connections (clamped between minConnectionStrength and maxConnectionStrength)
		if (this.debug) console.log('Running forget cycle - connection update...');
		stepStart = Date.now();
		const [connectionUpdateResult] = await this.conn.query(`UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.connectionForgetRate]);
		if (this.debug) console.log(`  Connection UPDATE took ${Date.now() - stepStart}ms (updated ${connectionUpdateResult.affectedRows} rows)`);

		// Delete connections with zero strength
		if (this.debug) console.log('Running forget cycle - connection deletion...');
		stepStart = Date.now();
		const [connectionDeleteResult] = await this.conn.query(`DELETE FROM connections WHERE strength = ?`, [this.minConnectionStrength]);
		if (this.debug) console.log(`  Connection DELETE took ${Date.now() - stepStart}ms (deleted ${connectionDeleteResult.affectedRows} rows)`);

		// 3. REWARD DECAY: Move reward factors back toward 1.0 (neutral)
		// Formula: reward = reward + (1.0 - reward) * rewardForgetRate
		// reward=2.0, rate=0.05 → 2.0 + (1.0-2.0)*0.05 = 1.95
		// reward=0.5, rate=0.05 → 0.5 + (1.0-0.5)*0.05 = 0.525
		if (this.debug) console.log('Running forget cycle - connection reward decay...');
		stepStart = Date.now();
		const [connRewardResult] = await this.conn.query(`UPDATE connections SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]);
		if (this.debug) console.log(`  Connection reward decay took ${Date.now() - stepStart}ms (updated ${connRewardResult.affectedRows} rows)`);

		if (this.debug) console.log('Running forget cycle - pattern_future reward decay...');
		stepStart = Date.now();
		const [patternRewardResult] = await this.conn.query(`UPDATE pattern_future SET reward = reward + (1.0 - reward) * ?`, [this.rewardForgetRate]);
		if (this.debug) console.log(`  Pattern_future reward decay took ${Date.now() - stepStart}ms (updated ${patternRewardResult.affectedRows} rows)`);

		// 4. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
		if (this.debug) console.log('Running forget cycle - orphaned neurons cleanup...');
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
		if (this.debug) console.log(`  Orphaned neurons DELETE took ${Date.now() - stepStart}ms (deleted ${neuronDeleteResult.affectedRows} rows)`);

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Apply channel-specific rewards to connections/patterns that led to executed outputs.
	 * Uses multiplicative rewards with exponential temporal decay.
	 * Older connections get less reward/punishment (decay applied to the reward exponent).
	 *
	 * Channel-Specific Credit Assignment:
	 * 1. Identify which channel each base-level output belongs to (via output dimensions)
	 * 2. Use action_sources to find which connections/patterns led to each action (unpacked from higher levels)
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
		let totalConnectionsRewarded = 0;
		let totalPatternsRewarded = 0;
		for (const [channelName, reward] of channelRewards) {

			if (this.debug) console.log(`Applying reward ${reward.toFixed(3)} for channel: ${channelName}`);

			// Get the output dimension IDs for this channel
			const outputDimIds = this.getChannelOutputDims(channelName);
			if (outputDimIds.length === 0) {
				console.warn(`Warning: No output dimensions found for channel ${channelName}`);
				continue;
			}

			// Reward and habituate connection-based inferences for this channel
			// source_type='connection' covers both regular connection inference and exploration
			// Use EXISTS to filter to channel outputs (clearer than JOIN)
			const [connResult] = await this.conn.query(`
				UPDATE connections c
				JOIN action_sources asrc ON c.id = asrc.source_id AND asrc.source_type = 'connection'
				SET c.reward = GREATEST(?, LEAST(?, c.reward * (1 + (? - 1) * POW(?, asrc.age - 1))))
				WHERE asrc.age > 0 AND asrc.age <= ?
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = asrc.action_neuron_id AND coord.dimension_id IN (?)
				)
			`, [
				this.minConnectionReward,
				this.maxConnectionReward,
				reward,
				this.rewardTimeDecayFactor,
				this.maxRewardsAge,
				outputDimIds
			]);
			totalConnectionsRewarded += connResult.affectedRows;

			// Reward and habituate pattern-based inferences for this channel
			// source_type='pattern' means source_id is pattern_future.id
			const [patternResult] = await this.conn.query(`
				UPDATE pattern_future pf
				JOIN action_sources asrc ON pf.id = asrc.source_id AND asrc.source_type = 'pattern'
				SET pf.reward = GREATEST(?, LEAST(?, pf.reward * (1 + (? - 1) * POW(?, asrc.age - 1))))
				WHERE asrc.age > 0 AND asrc.age <= ?
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = asrc.action_neuron_id AND coord.dimension_id IN (?)
				)
			`, [
				this.minConnectionReward,
				this.maxConnectionReward,
				reward,
				this.rewardTimeDecayFactor,
				this.maxRewardsAge,
				outputDimIds
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
}