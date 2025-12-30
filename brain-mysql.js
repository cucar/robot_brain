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
			'new_pattern_connections',
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
			'new_pattern_connections',
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
		const [infSourcesResult] = await this.conn.query('DELETE FROM inference_sources WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Cleaned up ${infSourcesResult.affectedRows} aged-out inference sources (age >= ${this.baseNeuronMaxAge})`);
	}

	/**
	 * Get frame outputs for all channels from inferred_neurons table (MySQL implementation)
	 * Reads winning action neurons (age=0, level=0, is_winner=1) grouped by channel
	 * @returns {Promise<Map>} - Map of channel names to array of output coordinates
	 */
	async getFrameOutputs() {

		// Get all winning action neurons from inferred_neurons table
		// Only return neurons that have action dimensions (not event dimensions)
		const [rows] = await this.conn.query(`
			SELECT inf.neuron_id, c.dimension_id, c.val, d.name as dimension_name, d.channel
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0 AND inf.level = 0 AND inf.is_winner = 1 AND d.type = 'action'
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
	 * Count bad inferences: prediction errors OR negative reward actions.
	 * This is the entry criteria for error pattern creation.
	 * Returns 0 if ANY pattern inference occurred (pattern errors handled by mergeMatchedPatterns).
	 * Only creates new patterns when connection inference was used.
	 * Returns the count of bad inferences.
	 */
	async countBadInferences(predictionLevel) {

		// If any pattern inference occurred (won or lost), don't create new patterns
		// Pattern errors are handled by mergeMatchedPatterns, not by creating new patterns
		const [patternCheck] = await this.conn.query(`
			SELECT COUNT(*) as count FROM inference_sources WHERE age = 1 AND source_type = 'pattern'
		`);
		if (patternCheck[0].count > 0) return 0;

		// Bad inferences are winners that either:
		// 1. Didn't happen (prediction error)
		// 2. Got negative actual_reward (action regret)
		const [result] = await this.conn.query(`
			SELECT COUNT(*) as count
			FROM inferred_neurons inf
			WHERE inf.level = ?
			AND inf.age = 1
			AND inf.strength >= ?
			AND inf.is_winner = 1
			AND (
				NOT EXISTS (
					SELECT 1 FROM active_neurons an
					WHERE an.neuron_id = inf.neuron_id
					AND an.level = inf.level
					AND an.age = 0
				)
				OR inf.actual_reward < 1.0
			)
		`, [predictionLevel, this.minErrorPatternThreshold]);

		// Debug: show what was predicted vs what happened
		if (this.debug) {
			const [predicted] = await this.conn.query(`
				SELECT inf.neuron_id, inf.strength, inf.expected_reward, inf.actual_reward, inf.is_winner
				FROM inferred_neurons inf
				WHERE inf.level = ? AND inf.age = 1 AND inf.is_winner = 1
			`, [predictionLevel]);
			const [actual] = await this.conn.query(`
				SELECT an.neuron_id
				FROM active_neurons an
				WHERE an.level = ? AND an.age = 0
			`, [predictionLevel]);
			console.log(`Predicted winners: [${predicted.map(p => `${p.neuron_id}(exp=${p.expected_reward.toFixed(2)},act=${(p.actual_reward ?? 'N/A')})`).join(',')}], Actual: [${actual.map(a => a.neuron_id).join(',')}]`);
		}

		return result[0].count;
	}

	/**
	 * Populate new_pattern_connections with connections that should be predicted by new patterns.
	 * Two sources:
	 * 1. Prediction errors: active connections that weren't predicted as winners
	 * 2. Action regret: connections to best loser for dimensions where winner got negative reward
	 * Returns the number of new pattern connections found.
	 */
	async populateNewPatternConnections(predictionLevel) {
		await this.conn.query(`TRUNCATE new_pattern_connections`);
		await this.populatePredictionErrorConnections(predictionLevel);
		await this.populateActionRegretConnections(predictionLevel);
		const [countResult] = await this.conn.query(`SELECT COUNT(*) as count FROM new_pattern_connections`);
		return countResult[0].count;
	}

	/**
	 * Populate connections from prediction errors - active connections not predicted as winners.
	 */
	async populatePredictionErrorConnections(predictionLevel) {
		const [result] = await this.conn.query(`
			INSERT INTO new_pattern_connections (connection_id, level, from_neuron_id, to_neuron_id, strength)
			SELECT ac.connection_id, ac.level, c.from_neuron_id, c.to_neuron_id, c.strength
			FROM active_connections ac
			JOIN connections c ON c.id = ac.connection_id
			WHERE ac.level = ?
			AND ac.age = 0
			AND NOT EXISTS (
				SELECT 1 FROM inferred_neurons inf
				WHERE inf.neuron_id = c.to_neuron_id
				AND inf.level = ac.level
				AND inf.age = 1
				AND inf.is_winner = 1
			)
		`, [predictionLevel]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Found ${result.affectedRows} prediction error connections`);
	}

	/**
	 * Populate connections from action regret - best loser connections for dimensions where winner got negative reward.
	 */
	async populateActionRegretConnections(predictionLevel) {
		// Get loser votes (use expected_reward since actual_reward is only set for winners)
		const [loserVotes] = await this.conn.query(`
			SELECT inf.neuron_id, isrc.source_type, isrc.source_id, isrc.inference_strength as strength, inf.expected_reward as reward, inf.level
			FROM inferred_neurons inf
			JOIN inference_sources isrc ON isrc.neuron_id = inf.neuron_id AND isrc.age = inf.age
			WHERE inf.level = ? AND inf.age = 1 AND inf.is_winner = 0
		`, [predictionLevel]);

		if (loserVotes.length === 0) return;

		// Find best loser per dimension
		const bestLosers = await this.determineConsensus(loserVotes);
		const bestLoserWinners = bestLosers.filter(l => l.isWinner);

		// Filter to dimensions where winner got negative actual_reward
		const bestLoserNeuronIds = await this.filterToRegretDimensions(bestLoserWinners, predictionLevel);
		if (bestLoserNeuronIds.length === 0) return;

		// Insert connections to best losers
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_connections (connection_id, level, from_neuron_id, to_neuron_id, strength)
			SELECT c.id, ?, c.from_neuron_id, c.to_neuron_id, c.strength
			FROM inference_sources isrc
			JOIN connections c ON c.id = isrc.source_id
			WHERE isrc.age = 1 AND isrc.source_type = 'connection' AND isrc.neuron_id IN (?)
		`, [predictionLevel, bestLoserNeuronIds]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Found ${result.affectedRows} action regret connections`);
	}

	/**
	 * Filter the best losers to only those in dimensions where the winner got negative actual_reward.
	 */
	async filterToRegretDimensions(bestLoserWinners, predictionLevel) {
		const neuronIds = [];
		for (const loser of bestLoserWinners) {
			if (!loser.coordinates) continue;
			for (const dimName of Object.keys(loser.coordinates)) {
				const [winnerCheck] = await this.conn.query(`
					SELECT inf.actual_reward
					FROM inferred_neurons inf
					JOIN coordinates c ON c.neuron_id = inf.neuron_id
					JOIN dimensions d ON d.id = c.dimension_id
					WHERE inf.level = ? AND inf.age = 1 AND inf.is_winner = 1 AND d.name = ?
				`, [predictionLevel, dimName]);
				if (winnerCheck.length > 0 && winnerCheck[0].actual_reward < 1.0)
					neuronIds.push(loser.neuron_id);
			}
		}
		return neuronIds;
	}

	/**
	 * Collect votes from ALL levels in bulk queries.
	 * Returns all votes (actions and events) with level-weighted strengths.
	 * @returns {Promise<Array>} Array of {neuron_id, source_type, source_id, strength, reward, level}
	 */
	async collectVotes() {

		// Collect connection votes from ALL levels in one query
		// Vote weight = strength * POW(peakTimeDecayFactor, distance-1) * POW(levelVoteMultiplier, source_level)
		const [connectionVotes] = await this.conn.query(`
			SELECT c.to_neuron_id as neuron_id, 'connection' as source_type, c.id as source_id,
				c.strength * POW(?, c.distance - 1) * POW(?, an.level) as strength,
				c.reward, n.level
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			JOIN neurons n ON n.id = c.to_neuron_id
			WHERE c.distance = an.age + 1
		`, [this.peakTimeDecayFactor, this.levelVoteMultiplier]);

		// Collect pattern votes from ALL levels in one query
		// pattern_future stores connection_id (from peak neuron to target), patterns always predict next frame (age=0)
		const [patternVotes] = await this.conn.query(`
			SELECT c.to_neuron_id as neuron_id, 'pattern' as source_type, pf.connection_id as source_id,
				pf.strength * POW(?, c.distance - 1) * POW(?, an.level) as strength,
				pf.reward, n.level
			FROM active_neurons an
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = an.neuron_id
			JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
			JOIN connections c ON c.id = pf.connection_id
			JOIN neurons n ON n.id = c.to_neuron_id
			WHERE an.age = 0
		`, [this.peakTimeDecayFactor, this.levelVoteMultiplier]);

		// Combine all votes
		const allVotes = [...connectionVotes, ...patternVotes];
		if (this.debug) console.log(`Collected ${connectionVotes.length} connection + ${patternVotes.length} pattern = ${allVotes.length} total votes`);

		// Debug: show votes for action neurons (neuron 13 = OWN/1, neuron 14 = OUT/-1)
		if (this.debug2) {
			const actionVotes = allVotes.filter(v => v.neuron_id === 13 || v.neuron_id === 14);
			// Group by neuron_id and calculate totals
			const ownVotes = actionVotes.filter(v => v.neuron_id === 13); // neuron 13 = 1 = OWN
			const outVotes = actionVotes.filter(v => v.neuron_id === 14); // neuron 14 = -1 = OUT
			const ownStrength = ownVotes.reduce((s, v) => s + v.strength, 0);
			const outStrength = outVotes.reduce((s, v) => s + v.strength, 0);
			const ownReward = ownVotes.reduce((s, v) => s + v.reward, 0);
			const outReward = outVotes.reduce((s, v) => s + v.reward, 0);
			console.log(`Action votes: OWN(13): ${ownVotes.length} votes, strength=${ownStrength.toFixed(1)}, reward=${ownReward.toFixed(2)} | OUT(14): ${outVotes.length} votes, strength=${outStrength.toFixed(1)}, reward=${outReward.toFixed(2)}`);
		}

		return allVotes;
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

		// Batch insert into inference_sources (IGNORE duplicates from exploration matching voting)
		const rows = [...deduped.values()].map(s => [0, s.neuron_id, s.source_type, s.source_id, s.strength]);
		await this.conn.query('INSERT IGNORE INTO inference_sources (age, neuron_id, source_type, source_id, inference_strength) VALUES ?', [rows]);
		if (this.debug2) console.log(`Saved ${rows.length} inference sources`);
	}

	/**
	 * Save all inferences in one operation.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	async saveInferences(inferences) {
		if (inferences.length === 0) return;

		// Collect neurons and sources
		const neurons = [];
		const sources = new Map();

		for (const inf of inferences) {
			// is_winner: 1 for winners (highest reward then strength per dimension), 0 for losers
			const isWinner = inf.isWinner ? 1 : 0;
			// expected_reward is from connection/pattern sources at inference time
			// actual_reward is NULL until channel reward is applied
			neurons.push([inf.neuron_id, inf.level || 0, 0, inf.strength, inf.reward, isWinner]);
			if (inf.sources && inf.sources.length > 0) sources.set(inf.neuron_id, inf.sources);
		}

		// Save neurons and sources
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength, expected_reward, is_winner) VALUES ? ON DUPLICATE KEY UPDATE strength = strength',
			[neurons]
		);
		await this.saveInferenceSources(sources);

		if (this.debug) {
			const winnerCount = inferences.filter(i => i.isWinner).length;
			const loserCount = inferences.filter(i => !i.isWinner).length;
			console.log(`Saved ${inferences.length} inferences (${winnerCount} winners, ${loserCount} losers)`);
		}
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
	 * Merge pattern_future with observed connections.
	 * Called during learning phase after pattern inference from previous frame.
	 * Uses inference_sources with source_type='pattern' to know which patterns made predictions.
	 * pattern_future stores connection_id (from peak neuron to target).
	 *
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future connections that were correctly predicted (target now active)
	 * 2. Negative: Weaken pattern_future connections that were incorrectly predicted (target NOT active)
	 * 3. Novel: Add new connections from peak to newly observed neurons
	 */
	async mergePatternFuture() {

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted connections
		// Connection's target neuron is now active at age=0
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN connections c ON c.id = pf.connection_id
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength + 1))
			WHERE pf.strength > 0
			AND pf.connection_id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
			AND c.to_neuron_id IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
		`, [this.minConnectionStrength, this.maxConnectionStrength]);
		if (this.debug && strengthenResult.affectedRows > 0)
			console.log(`Strengthened ${strengthenResult.affectedRows} correct pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections
		// Connection's target neuron is NOT active
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN connections c ON c.id = pf.connection_id
			SET pf.strength = GREATEST(?, LEAST(?, pf.strength - ?))
			WHERE pf.strength > 0
			AND pf.connection_id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
			AND c.to_neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
		`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug && weakenResult.affectedRows > 0)
			console.log(`Weakened ${weakenResult.affectedRows} failed pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Connections from peak to newly active neurons not yet in pattern_future
		// Find patterns that made predictions, get their peak neurons, find connections from peak to active neurons
		const [novelResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pf_src.pattern_neuron_id, c.id, 1.0
			FROM inference_sources isrc
			JOIN pattern_future pf_src ON pf_src.connection_id = isrc.source_id
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf_src.pattern_neuron_id
			JOIN connections c ON c.from_neuron_id = pp.peak_neuron_id
			JOIN active_neurons an ON an.neuron_id = c.to_neuron_id AND an.age = 0
			WHERE isrc.age = 1
			AND isrc.source_type = 'pattern'
			AND c.strength > 0
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf
				WHERE pf.pattern_neuron_id = pf_src.pattern_neuron_id
				AND pf.connection_id = c.id
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
			SELECT from_neuron_id FROM new_pattern_connections GROUP BY from_neuron_id
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Create pattern neurons and map them to new_patterns.
	 * @param {number} patternCount - Number of patterns to create neurons for
	 * @param {number} level - Level of the pattern neurons
	 */
	async createPatternNeurons(patternCount, level) {

		// Create pattern neurons (one per peak) at the specified level
		const patternNeuronIds = await this.createNeurons(patternCount, level);

		// Update new_patterns with pattern neuron IDs in bulk
		// seq_id auto-increments from 1, patternNeuronIds are sequential, so: pattern_neuron_id = firstNeuronId + (seq_id - 1)
		await this.conn.query('UPDATE new_patterns SET pattern_neuron_id = ? + (seq_id - 1)', [patternNeuronIds[0]]);
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future.
	 * pattern_future contains:
	 * pattern_future stores connection_id (from peak neuron to target neurons).
	 * pattern_past includes cross-level connections (base→high, same-level).
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
		// Includes cross-level connections: same-level, base→high (level 0 → predictionLevel)
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, ac.connection_id, 1.0
			FROM new_patterns np
			JOIN active_connections ac ON ac.to_neuron_id = np.peak_neuron_id
			WHERE ac.level = ?
			AND ac.age = 1
		`, [predictionLevel]);

		// Create pattern_future entries: connections FROM peak to target neurons
		// Uses new_pattern_connections which contains connections the pattern should predict
		// These are the connections the pattern should predict in the future
		// Use INSERT IGNORE in case pattern already has this connection
		await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, npc.connection_id, 1.0
			FROM new_patterns np
			JOIN new_pattern_connections npc ON npc.from_neuron_id = np.peak_neuron_id
		`);

		// Add connections from peak to active action neurons (for voting)
		// These connections should already exist from reinforceConnections
		await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, c.id, 1.0
			FROM new_patterns np
			JOIN connections c ON c.from_neuron_id = np.peak_neuron_id
			JOIN active_neurons an ON an.neuron_id = c.to_neuron_id AND an.level = 0 AND an.age = 1
			JOIN coordinates coord ON coord.neuron_id = an.neuron_id
			JOIN dimensions d ON d.id = coord.dimension_id AND d.type = 'action'
			WHERE c.strength > 0
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
	 * @param {number} level - Level of the neurons (0 for base neurons, 1+ for pattern neurons)
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async createNeurons(count, level = 0) {
		const valuesSql = Array(count).fill(`(${level})`).join(',');
		const insertNeuronsResult = await this.conn.query(`INSERT INTO neurons (level) VALUES ${valuesSql}`);
		const firstNeuronId = insertNeuronsResult[0].insertId;

		// Return array of sequential IDs
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * Reinforce connections between active neurons.
	 * Creates connections from all active neurons to newly activated (age=0) neurons at the specified level.
	 * Cross-level connections allowed: same-level, high→base, and base→high.
	 * Distance is purely temporal (source neuron's age), regardless of level difference.
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT f.neuron_id as from_neuron_id, t.neuron_id as to_neuron_id, f.age as distance, 1 as strength
			FROM active_neurons f
			CROSS JOIN active_neurons t
            WHERE t.age = 0  -- target neurons are newly activated
            AND t.level = :level  -- target neurons are at the specified level
            AND (f.level = t.level OR t.level = 0 OR f.level = 0)  -- same level, high→base, or base→high
            AND f.age > 0 -- only learning connections between different ages for inference
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
	 * Cross-level connections allowed: same-level, high→base, and base→high.
	 * Distance matching is purely temporal: c.distance = f.age.
	 */
	async activateConnections(level) {
		await this.conn.query(`
			INSERT IGNORE INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level, age)
			SELECT c.id as connection_id, c.from_neuron_id, c.to_neuron_id, t.level, 0 as age
			FROM connections c
			JOIN active_neurons f ON c.from_neuron_id = f.neuron_id
			JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = :level
			WHERE c.distance = f.age
			AND (f.level = t.level OR t.level = 0 OR f.level = 0)  -- same level, high→base, or base→high
            AND f.age > 0 -- only connections between different ages for inference
			AND c.strength > 0  -- only connections that are not removed
		`, { level });
	}

	/**
	 * Match active connections directly to known patterns.
	 * No peak detection needed - we only check neurons that are already known peaks (in pattern_peaks).
	 * Writes results to matched_patterns memory table.
	 * Matches by connection_id (which encodes from_neuron + to_neuron + distance) to preserve temporal structure.
	 * Uses connection overlap (66% threshold) to determine if patterns match.
	 * Cross-level connections are included: pattern_past can contain connections from any level to the peak.
	 * @param {number} level - The level to match patterns for (peak neuron level)
	 * @returns {Promise<number>} - Number of matched patterns
	 */
	async matchObservedPatterns(level) {
		if (this.debug2) console.log('Matching active connections to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_patterns');
		await this.conn.query('TRUNCATE matched_pattern_connections');

		// Determine which patterns matched based on overlap threshold
		// A pattern matches if at least 66% of its pattern_past connections are in active_connections
		// active_connections.level = target level, so cross-level connections (base→high) are included
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
		// Cross-level connections included via active_connections.level = target level
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, p.connection_id, IF(ac.connection_id IS NOT NULL, 'common', 'missing') as status
			FROM matched_patterns mp
			JOIN pattern_past p ON mp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.to_neuron_id = mp.peak_neuron_id AND ac.connection_id = p.connection_id AND ac.level = ? AND ac.age = 0
		`, [level]);

		// Novel connections: active connections TO the peak not in pattern_past
		// Includes cross-level connections (base→high)
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
	 * 2. Use inferred_neurons (is_winner) + inference_sources to find which connections/patterns led to each action
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
		let totalWinnerConnections = 0, totalWinnerPatterns = 0;
		let totalLoserConnections = 0, totalLoserPatterns = 0;
		for (const [channelName, reward] of channelRewards) {

			if (this.debug) console.log(`Applying reward ${reward.toFixed(3)} for channel: ${channelName}`);

			// Get the output dimension IDs for this channel
			const outputDimIds = this.getChannelOutputDims(channelName);
			if (outputDimIds.length === 0) {
				console.warn(`Warning: No output dimensions found for channel ${channelName}`);
				continue;
			}

			// WINNERS: Apply actual reward to winning votes (is_winner=1)
			// Reward connection-based inferences
			const [winnerConnResult] = await this.conn.query(`
				UPDATE connections c
				JOIN inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET c.reward = GREATEST(?, LEAST(?, c.reward * (1 + (? - 1) * POW(?, isrc.age - 1))))
				WHERE isrc.age > 0 AND isrc.age <= ?
				AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (?)
				)
			`, [this.minConnectionReward, this.maxConnectionReward, reward, this.rewardTimeDecayFactor, this.maxRewardsAge, outputDimIds]);
			totalWinnerConnections += winnerConnResult.affectedRows;

			// Reward pattern-based inferences (pattern_future stores connection_id)
			const [winnerPatternResult] = await this.conn.query(`
				UPDATE pattern_future pf
				JOIN inference_sources isrc ON pf.connection_id = isrc.source_id AND isrc.source_type = 'pattern'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET pf.reward = GREATEST(?, LEAST(?, pf.reward * (1 + (? - 1) * POW(?, isrc.age - 1))))
				WHERE isrc.age > 0 AND isrc.age <= ?
				AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (?)
				)
			`, [this.minConnectionReward, this.maxConnectionReward, reward, this.rewardTimeDecayFactor, this.maxRewardsAge, outputDimIds]);
			totalWinnerPatterns += winnerPatternResult.affectedRows;

			// Update inferred_neurons.actual_reward for action winners so countBadInferences can detect negative rewards
			await this.conn.query(`
				UPDATE inferred_neurons inf
				SET inf.actual_reward = ?
				WHERE inf.age = 1 AND inf.level = 0 AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = inf.neuron_id AND coord.dimension_id IN (?)
				)
			`, [reward, outputDimIds]);

			// LOSERS: Apply inverse reward (1/reward) to losing votes (is_winner=0)
			// This trains dissenting levels even when outvoted
			// If winner got reward=1.5 (good), losers get 1/1.5=0.67 (punishment)
			// If winner got reward=0.5 (bad), losers get 1/0.5=2.0 (reward for being right)
			const inverseReward = 1.0 / reward;

			// Inverse reward connection-based inferences
			const [loserConnResult] = await this.conn.query(`
				UPDATE connections c
				JOIN inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET c.reward = GREATEST(?, LEAST(?, c.reward * (1 + (? - 1) * POW(?, isrc.age - 1))))
				WHERE isrc.age > 0 AND isrc.age <= ?
				AND inf.is_winner = 0
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (?)
				)
			`, [this.minConnectionReward, this.maxConnectionReward, inverseReward, this.rewardTimeDecayFactor, this.maxRewardsAge, outputDimIds]);
			totalLoserConnections += loserConnResult.affectedRows;

			// Inverse reward pattern-based inferences (pattern_future stores connection_id)
			const [loserPatternResult] = await this.conn.query(`
				UPDATE pattern_future pf
				JOIN inference_sources isrc ON pf.connection_id = isrc.source_id AND isrc.source_type = 'pattern'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET pf.reward = GREATEST(?, LEAST(?, pf.reward * (1 + (? - 1) * POW(?, isrc.age - 1))))
				WHERE isrc.age > 0 AND isrc.age <= ?
				AND inf.is_winner = 0
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (?)
				)
			`, [this.minConnectionReward, this.maxConnectionReward, inverseReward, this.rewardTimeDecayFactor, this.maxRewardsAge, outputDimIds]);
			totalLoserPatterns += loserPatternResult.affectedRows;

			if (this.debug) console.log(`  ${channelName}: winners=${winnerConnResult.affectedRows}c/${winnerPatternResult.affectedRows}p (${reward.toFixed(2)}), losers=${loserConnResult.affectedRows}c/${loserPatternResult.affectedRows}p (${inverseReward.toFixed(2)})`);
		}

		if (this.debug) console.log(`Total rewarded: winners=${totalWinnerConnections}c/${totalWinnerPatterns}p, losers=${totalLoserConnections}c/${totalLoserPatterns}p`);
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