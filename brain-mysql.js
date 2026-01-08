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

		// Cache for action neuron IDs (for debug output)
		this.actionNeuronCache = null;
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
			'new_pattern_future',
			'new_patterns'
		]);

		// Clear action neuron cache since neurons may have been recreated
		this.actionNeuronCache = null;
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
			'new_pattern_future',
			'new_patterns',
			'pattern_peaks',
			'pattern_past',
			'pattern_future',
			'connections',
			'coordinates',
			'neurons'
		]);

		// Clear action neuron cache since all neurons were deleted
		this.actionNeuronCache = null;
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
		// Only applies to events/patterns, NOT actions (actions are handled by reward system)
		const [result] = await this.conn.query(`
			UPDATE connections c
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE c.strength > 0
			-- penalize connections that were inferred in the previous frame
			AND c.id IN (SELECT source_id FROM inference_sources WHERE source_type = 'connection' AND age = 1)
			-- penalize the connections that did not come true
			AND c.id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
			-- only penalize event/pattern predictions, not actions
			AND NOT EXISTS (
				SELECT 1 FROM coordinates coord
				JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = c.to_neuron_id AND d.type = 'action'
			)
		`, [this.connectionNegativeReinforcement]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Weakened ${result.affectedRows} failed event predictions`);
	}

	/**
	 * Populate new_pattern_future with connections that should be in pattern_future of new patterns.
	 * Unified method that handles both prediction errors and action regret.
	 *
	 * Common filters for all pattern creation:
	 * - source_type = 'connection' (patterns only created from connection inference errors, not pattern inference)
	 * - inference_strength >= threshold (only surprising errors warrant new patterns)
	 * - Peak has context (connections TO it at age=1)
	 * - Peak didn't use Type 2 inference (wasn't itself pattern-inferred)
	 * - c.strength > 0 (only use valid connections)
	 *
	 * Two cases:
	 * 1. Prediction errors: inferred neuron NOT in active_neurons age=0 → connections from peak to what actually happened
	 * 2. Action regret: inferred action winner got negative reward → connections from peak to best loser action
	 *
	 * Returns the number of new pattern future connections found.
	 */
	async populateNewPatternFuture() {
		await this.conn.query(`TRUNCATE new_pattern_future`);
		await this.populatePredictionErrorFuture();
		await this.populateActionRegretFuture();
		const [countResult] = await this.conn.query(`SELECT COUNT(*) as count FROM new_pattern_future`);
		return countResult[0].count;
	}

	/**
	 * Populate prediction error connections into new_pattern_future.
	 * Peak predicted X via connection, but X didn't happen and Y happened instead.
	 * → Create pattern to predict Y from this context.
	 * Excludes peaks that have action regret (action regret takes priority).
	 */
	async populatePredictionErrorFuture() {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (connection_id)
			SELECT ac.connection_id
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			JOIN connections c_inferred ON c_inferred.id = isrc.source_id
			-- Find active connections FROM the same peak (what actually happened)
			-- Only distance=1: peak was at age=0 last frame, now at age=1, new observation at age=0
			JOIN active_connections ac ON ac.from_neuron_id = c_inferred.from_neuron_id AND ac.age = 0
			JOIN connections c ON c.id = ac.connection_id AND c.distance = 1
			WHERE isrc.age = 1
			AND isrc.source_type = 'connection'
			AND isrc.inference_strength >= ?
            -- The inferred neuron is an event or interneuron - not an action (those are handled separately below)
            AND NOT EXISTS (
                SELECT 1 FROM coordinates coord
                JOIN dimensions d ON d.id = coord.dimension_id
                WHERE coord.neuron_id = inf.neuron_id
                AND d.type = 'action'
            )
			-- The inference didn't come true (prediction error)
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = isrc.neuron_id
				AND an.age = 0
			)
			-- The peak neuron did NOT use Type 2 inference
			AND NOT EXISTS (
				SELECT 1 FROM inference_sources isrc2
				WHERE isrc2.neuron_id = c_inferred.from_neuron_id
				AND isrc2.age = 1
				AND isrc2.source_type = 'pattern'
			)
			-- The peak neuron has context (connections TO it from older neurons)
			AND EXISTS (
				SELECT 1 FROM active_connections context_ac
				WHERE context_ac.to_neuron_id = c_inferred.from_neuron_id
				AND context_ac.age = 1
			)
			-- Action neurons can NEVER be peaks (only events and interneurons can be peaks)
			AND NOT EXISTS (
				SELECT 1 FROM coordinates coord
				JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = c_inferred.from_neuron_id AND d.type = 'action'
			)
		`, [this.predictionErrorMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} prediction error connections`);
	}

	/**
	 * Populate action regret connections into new_pattern_future.
	 * Peak predicted action X (winner) via connection, but got negative reward.
	 * → Create pattern to predict the best loser action from this context.
	 * Uses determineConsensus to find the best alternative action (same logic as normal inference).
	 */
	async populateActionRegretFuture() {
		// Step 1: Find bad action inferences with their peaks
		const [badActionInferences] = await this.conn.query(`
			SELECT isrc.source_id as connection_id, c_inferred.from_neuron_id as peak_neuron_id, inf.neuron_id as bad_action_neuron_id
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			JOIN connections c_inferred ON c_inferred.id = isrc.source_id
			WHERE isrc.age = 1
			AND isrc.source_type = 'connection'
			AND isrc.inference_strength >= ?
			AND inf.is_winner = 1
			AND inf.actual_reward < ?
			-- The inferred neuron is an action
			AND EXISTS (
				SELECT 1 FROM coordinates coord
				JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = inf.neuron_id
				AND d.type = 'action'
			)
			-- The peak neuron did NOT use Type 2 inference
			AND NOT EXISTS (
				SELECT 1 FROM inference_sources isrc2
				WHERE isrc2.neuron_id = c_inferred.from_neuron_id
				AND isrc2.age = 1
				AND isrc2.source_type = 'pattern'
			)
			-- The peak neuron has context (connections TO it from older neurons)
			AND EXISTS (
				SELECT 1 FROM active_connections context_ac
				WHERE context_ac.to_neuron_id = c_inferred.from_neuron_id
				AND context_ac.age = 1
			)
			-- Action neurons can NEVER be peaks (only events and interneurons can be peaks)
			AND NOT EXISTS (
				SELECT 1 FROM coordinates coord
			    JOIN dimensions d ON d.id = coord.dimension_id
				WHERE coord.neuron_id = c_inferred.from_neuron_id AND d.type = 'action'
			)
		`, [this.actionRegretMinStrength, -this.actionRegretMinPain]);
		if (badActionInferences.length === 0) {
			if (this.debug) console.log('Action regret: no bad action inferences found');
			return;
		}
		if (this.debug) console.log(`Action regret: found ${badActionInferences.length} bad action inferences`);

		// Step 2: Get all loser votes from inference_sources (reconstruct the votes that lost)
		const [loserVotes] = await this.conn.query(`
			SELECT isrc.neuron_id, isrc.source_type, isrc.source_id, isrc.inference_strength as strength,
				inf.expected_reward as reward, inf.level, NULL as from_dim_type
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			WHERE isrc.age = 1
			AND inf.is_winner = 0
		`);
		if (loserVotes.length === 0) {
			if (this.debug) console.log('Action regret: no loser votes found');
			return;
		}
		if (this.debug) console.log(`Action regret: found ${loserVotes.length} loser votes`);

		// Step 3: Use determineConsensus to find what SHOULD have won among losers
		const consensusResults = await this.determineConsensus(loserVotes);
		const bestLoserNeuronIds = consensusResults.filter(r => r.isWinner).map(r => r.neuron_id);
		if (bestLoserNeuronIds.length === 0) {
			if (this.debug) console.log('Action regret: no consensus winners among losers');
			return;
		}
		if (this.debug) console.log(`Action regret: consensus found ${bestLoserNeuronIds.length} best loser actions`);

		// Step 4: Filter out peaks that already have an active pattern predicting the best loser
		const peakNeuronIds = [...new Set(badActionInferences.map(b => b.peak_neuron_id))];
		const [coveredPeaks] = await this.conn.query(`
			SELECT DISTINCT pp.peak_neuron_id
			FROM pattern_peaks pp
			JOIN active_neurons an ON an.neuron_id = pp.pattern_neuron_id AND an.age = 1
			JOIN pattern_future pf ON pf.pattern_neuron_id = pp.pattern_neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE pp.peak_neuron_id IN (?)
			AND c.to_neuron_id IN (?)
		`, [peakNeuronIds, bestLoserNeuronIds]);
		const coveredPeakIds = new Set(coveredPeaks.map(r => r.peak_neuron_id));
		const uncoveredPeakIds = peakNeuronIds.filter(id => !coveredPeakIds.has(id));
		if (uncoveredPeakIds.length === 0) {
			if (this.debug) console.log('Action regret: all peaks already covered by existing patterns');
			return;
		}
		if (this.debug) console.log(`Action regret: ${uncoveredPeakIds.length} uncovered peaks (${coveredPeakIds.size} already covered)`);

		// Step 5: Insert connections from uncovered peaks to the best losers
		// These connections exist but weren't active because we executed the wrong action
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (connection_id)
			SELECT c.id
			FROM connections c
			WHERE c.from_neuron_id IN (?)
			AND c.to_neuron_id IN (?)
			AND c.distance = 1
		`, [uncoveredPeakIds, bestLoserNeuronIds]);
		if (this.debug) console.log(`Found ${result.affectedRows} action regret connections`);
	}

	/**
	 * Collect votes from ALL levels in bulk queries.
	 * Returns all votes (actions and events) with level-weighted strengths.
	 * @returns {Promise<Array>} Array of {neuron_id, source_type, source_id, strength, reward, level}
	 */
	async collectVotes() {

		// Collect connection votes from ALL levels in one query
		// Vote weight = strength * POW(peakTimeDecayFactor, distance-1) * POW(levelVoteMultiplier, source_level)
		// from_dim_type: 'event'/'action' for base neurons, NULL for interneurons (level > 0)
		// Exclude neurons at max age - they'll be deactivated before we can populate inference_sources
		const [connectionVotes] = await this.conn.query(`
			SELECT c.from_neuron_id, c.to_neuron_id as neuron_id, 'connection' as source_type, c.id as source_id,
				c.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level) as strength,
                c.reward, n.level,
				(SELECT d.type FROM coordinates coord JOIN dimensions d ON d.id = coord.dimension_id WHERE coord.neuron_id = c.from_neuron_id LIMIT 1) as from_dim_type
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			JOIN neurons n ON n.id = c.to_neuron_id
			WHERE c.distance = an.age + 1
            AND c.strength > 0
			AND an.age < :maxAge
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier, maxAge: this.baseNeuronMaxAge });

		// Collect pattern votes from ALL levels and ages
		// pattern_future stores connections with various distances, patterns predict when distance matches age+1
		// from_dim_type: check underlying connection's from_neuron_id, NULL for interneurons
		// Exclude neurons at max age - they'll be deactivated before we can populate inference_sources
		const [patternVotes] = await this.conn.query(`
			SELECT c.from_neuron_id, c.to_neuron_id as neuron_id, 'pattern' as source_type, pf.connection_id as source_id,
				pf.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level) as strength,
                pf.reward, n.level,
				(SELECT d.type FROM coordinates coord JOIN dimensions d ON d.id = coord.dimension_id WHERE coord.neuron_id = c.from_neuron_id LIMIT 1) as from_dim_type
			FROM active_neurons an
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = an.neuron_id
			JOIN pattern_future pf ON pf.pattern_neuron_id = pp.pattern_neuron_id
			JOIN connections c ON c.id = pf.connection_id
			JOIN neurons n ON n.id = c.to_neuron_id
			WHERE c.distance = an.age + 1
			AND an.age = 0  
            AND c.strength > 0
            AND pf.strength > 0
			AND an.age < :maxAge
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier, maxAge: this.baseNeuronMaxAge });

		// Combine all votes
		const allVotes = [...connectionVotes, ...patternVotes];
		if (this.debug) console.log(`Collected ${connectionVotes.length} connection + ${patternVotes.length} pattern = ${allVotes.length} total votes`);

		// DEBUG: Check for level 1+ connection inferences (patterns doing connection inference)
		// Check if from_neuron is a pattern (level > 0) by checking if from_dim_type is NULL
		// const level1PlusConnectionVotes = connectionVotes.filter(v => v.from_dim_type === null);
		// if (level1PlusConnectionVotes.length > 0) {
		// 	console.error('FOUND LEVEL 1+ CONNECTION INFERENCES:');
		// 	console.error(level1PlusConnectionVotes);
		// 	process.exit(1);
		// }

		// Debug: show votes for action neurons
		if (this.debug2) await this.debugActionVotes(allVotes);

		return allVotes;
	}

	/**
	 * Populate action neuron cache by querying for OWN and OUT neurons
	 * Only caches if both neurons exist
	 */
	async populateActionNeuronCache() {
		const [actionNeurons] = await this.conn.query(`
			SELECT n.id as neuron_id, c.val as activity_value
			FROM neurons n
			JOIN coordinates c ON c.neuron_id = n.id
			JOIN dimensions d ON d.id = c.dimension_id
			WHERE d.name LIKE '%activity%' AND d.type = 'action'
			AND c.val IN (1, -1)
		`);

		const ownNeuronId = actionNeurons.find(n => n.activity_value === 1)?.neuron_id;
		const outNeuronId = actionNeurons.find(n => n.activity_value === -1)?.neuron_id;

		// Only cache if both neurons exist
		if (ownNeuronId && outNeuronId) this.actionNeuronCache = { ownNeuronId, outNeuronId };
	}

	/**
	 * Debug helper: show votes for OWN and OUT action neurons
	 * Shows which price/volume changes are voting for which action
	 * @param {Array} allVotes - Array of all votes
	 */
	async debugActionVotes(allVotes) {

		// Populate cache if needed
		if (!this.actionNeuronCache) await this.populateActionNeuronCache();
		if (!this.actionNeuronCache) return;

		const { ownNeuronId, outNeuronId } = this.actionNeuronCache;

		// Get source neuron coordinates for each vote
		const sourceNeuronIds = [...new Set(allVotes.map(v => v.source_id))];
		if (sourceNeuronIds.length === 0) return;

		// Query connections to get from_neuron coordinates
		const [connections] = await this.conn.query(`
			SELECT c.id, c.from_neuron_id, c.distance, c.strength as conn_strength, c.reward as conn_reward,
				GROUP_CONCAT(CONCAT(d.name, '=', coord.val) ORDER BY d.name SEPARATOR ', ') as from_coords
			FROM connections c
			JOIN coordinates coord ON coord.neuron_id = c.from_neuron_id
			JOIN dimensions d ON d.id = coord.dimension_id
			WHERE c.id IN (?)
			GROUP BY c.id
		`, [sourceNeuronIds]);

		const connMap = new Map(connections.map(c => [c.id, c]));

		// Get bucket-to-percentage mapping from stock channel if available
		const stockChannel = this.channels.get('TEST');
		const bucketToPercent = stockChannel ? this.buildBucketPercentMap(stockChannel) : null;

		// Calculate cycle frame (1-6) based on frame number
		const cycleFrame = ((this.frameNumber - 1) % 6) + 1;

		// Step 1: Aggregate by (from_neuron_id, to_neuron_id) - each source neuron votes once per target
		const aggregateBySource = (votes) => {
			const bySource = new Map();
			for (const v of votes) {
				const conn = connMap.get(v.source_id);
				if (!conn) continue;
				const key = conn.from_neuron_id;
				if (!bySource.has(key))
					bySource.set(key, { from_neuron_id: key, strength: 0, rewardSum: 0, rewardCount: 0, from_coords: conn.from_coords, distances: [] });
				const agg = bySource.get(key);
				agg.strength += v.strength;
				agg.rewardSum += v.reward;
				agg.rewardCount++;
				agg.distances.push(conn.distance);
			}
			// Calculate average reward per source
			for (const [_, agg] of bySource)
				agg.reward = agg.rewardCount > 0 ? agg.rewardSum / agg.rewardCount : 0;
			return [...bySource.values()];
		};

		const ownAgg = aggregateBySource(allVotes.filter(v => v.neuron_id === ownNeuronId));
		const outAgg = aggregateBySource(allVotes.filter(v => v.neuron_id === outNeuronId));

		// Format aggregated votes with source info
		const formatAggVotes = (aggVotes, label) => {
			if (aggVotes.length === 0) return `  ${label}: no votes`;
			const lines = [`  ${label}:`];
			for (const agg of aggVotes) {
				const coordsWithPercent = this.formatCoordsWithPercent(agg.from_coords, bucketToPercent);
				const distStr = agg.distances.length > 1 ? `d=[${agg.distances.join(',')}]` : `d=${agg.distances[0]}`;
				lines.push(`    ${coordsWithPercent} (${distStr}) → str=${agg.strength.toFixed(1)}, avgRwd=${agg.reward.toFixed(2)}`);
			}
			return lines.join('\n');
		};

		// Step 2: Calculate totals - all voters count in denominator for both actions
		// Collect all unique source neurons that voted for either action
		const allVoters = new Set();
		for (const agg of ownAgg) allVoters.add(agg.from_neuron_id);
		for (const agg of outAgg) allVoters.add(agg.from_neuron_id);
		const totalVoters = allVoters.size;

		// Reward = sum of rewards / total voters (missing votes count as 0)
		const ownTotal = {
			str: ownAgg.reduce((s, a) => s + a.strength, 0),
			rwd: totalVoters > 0 ? ownAgg.reduce((s, a) => s + a.rewardSum, 0) / totalVoters : 0
		};
		const outTotal = {
			str: outAgg.reduce((s, a) => s + a.strength, 0),
			rwd: totalVoters > 0 ? outAgg.reduce((s, a) => s + a.rewardSum, 0) / totalVoters : 0
		};

		// Calculate Boltzmann probabilities (exponential with temperature)
		const ownExp = Math.exp(ownTotal.rwd / this.boltzmannTemperature);
		const outExp = Math.exp(outTotal.rwd / this.boltzmannTemperature);
		const sumExp = ownExp + outExp;
		const ownProb = ownExp / sumExp;
		const outProb = outExp / sumExp;

		console.log(`\n=== ACTION VOTES (Cycle ${cycleFrame}/6) ===`);
		console.log(formatAggVotes(ownAgg, `OWN (${ownAgg.length}/${totalVoters} voters, str=${ownTotal.str.toFixed(1)}, avgRwd=${ownTotal.rwd.toFixed(2)}, prob=${(ownProb * 100).toFixed(1)}%)`));
		console.log(formatAggVotes(outAgg, `OUT (${outAgg.length}/${totalVoters} voters, str=${outTotal.str.toFixed(1)}, avgRwd=${outTotal.rwd.toFixed(2)}, prob=${(outProb * 100).toFixed(1)}%)`));
		console.log(`  SELECTION: Boltzmann (OWN ${(ownProb * 100).toFixed(1)}% vs OUT ${(outProb * 100).toFixed(1)}%)`);
		console.log(`===================\n`);
	}

	/**
	 * Build a map from bucket values to percentage ranges for price/volume dimensions
	 */
	buildBucketPercentMap(stockChannel) {
		const map = new Map();
		if (stockChannel.priceBuckets)
			for (const b of stockChannel.priceBuckets)
				map.set(`price_change:${b.value}`, this.formatBucketRange(b.min, b.max));
		if (stockChannel.volumeBuckets)
			for (const b of stockChannel.volumeBuckets)
				map.set(`volume_change:${b.value}`, this.formatBucketRange(b.min, b.max));
		return map;
	}

	/**
	 * Format bucket range as readable string
	 */
	formatBucketRange(min, max) {
		if (min === -Infinity) return `<${max}%`;
		if (max === Infinity) return `>${min}%`;
		return `${min}%~${max}%`;
	}

	/**
	 * Format coordinates string with percentage ranges where applicable
	 */
	formatCoordsWithPercent(coordsStr, bucketToPercent) {
		if (!bucketToPercent) return coordsStr;
		// Parse "TEST_price_change=5, TEST_volume_change=0" format
		return coordsStr.split(', ').map(part => {
			const [dimName, valStr] = part.split('=');
			const val = parseFloat(valStr);
			// Extract dimension type (price_change or volume_change)
			const dimType = dimName.includes('price_change') ? 'price_change' : dimName.includes('volume_change') ? 'volume_change' : null;
			if (dimType) {
				const percentRange = bucketToPercent.get(`${dimType}:${val}`);
				if (percentRange) return `${dimName}=${val}(${percentRange})`;
			}
			return part;
		}).join(', ');
	}

	/**
	 * Save all inferences in one operation.
	 * Sources are NOT saved here - they are populated after execution in populateInferenceSources.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	async saveInferences(inferences) {
		if (inferences.length === 0) return;

		// Collect neurons only (sources populated later after execution)
		const neurons = [];

		for (const inf of inferences) {
			// is_winner: 1 for winners (highest sum of strength * reward per dimension), 0 for losers
			const isWinner = inf.isWinner ? 1 : 0;
			// expected_reward is weighted average from connection/pattern sources at inference time
			// actual_reward is NULL until channel reward is applied
			neurons.push([inf.neuron_id, inf.level || 0, 0, inf.strength, inf.reward, isWinner]);
		}

		// Save neurons only
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength, expected_reward, is_winner) VALUES ? ON DUPLICATE KEY UPDATE is_winner = VALUES(is_winner), strength = VALUES(strength)',
			[neurons]
		);

		if (this.debug) {
			const winnerCount = inferences.filter(i => i.isWinner).length;
			const loserCount = inferences.filter(i => !i.isWinner).length;
			console.log(`Saved ${inferences.length} inferences (${winnerCount} winners, ${loserCount} losers)`);
		}
	}

	/**
	 * Populate inference_sources for all inferred neurons at age=1.
	 * Called after recognizeNeurons when connections have been created.
	 * Reverse-engineers collectVotes to find what would have predicted these neurons.
	 * Note: one frame later, so active_neurons are +1 age - use c.distance = an.age (not an.age + 1)
	 */
	async populateInferenceSources() {
		// Connection sources: active_neurons that have connections TO the inferred neuron
		// During inference: c.distance = an.age + 1, now an.age is +1, so c.distance = an.age
		const [connResult] = await this.conn.query(`
			INSERT INTO inference_sources (age, neuron_id, source_type, source_id, inference_strength)
			SELECT inf.age, inf.neuron_id, 'connection', c.id,
				c.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level)
			FROM inferred_neurons inf
			JOIN connections c ON c.to_neuron_id = inf.neuron_id
			JOIN active_neurons an ON an.neuron_id = c.from_neuron_id
			WHERE inf.age = 1
			AND c.distance = an.age
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier });

		// Pattern sources: active pattern peaks with pattern_future TO the inferred neuron
		// During inference: c.distance = an.age + 1, now an.age is +1, so c.distance = an.age
		// Multiple patterns can share the same connection - sum their strengths
		const [patternResult] = await this.conn.query(`
			INSERT INTO inference_sources (age, neuron_id, source_type, source_id, inference_strength)
			SELECT inf.age, inf.neuron_id, 'pattern', pf.connection_id,
				SUM(pf.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level))
			FROM inferred_neurons inf
			JOIN connections c ON c.to_neuron_id = inf.neuron_id
			JOIN pattern_future pf ON pf.connection_id = c.id
			JOIN active_neurons an ON an.neuron_id = pf.pattern_neuron_id
			WHERE inf.age = 1
			AND c.distance = an.age
			GROUP BY inf.age, inf.neuron_id, pf.connection_id
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier });

		if (this.debug) console.log(`Populated ${connResult.affectedRows} connection + ${patternResult.affectedRows} pattern inference sources`);
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
		// Connection is now active at age=0
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN active_connections ac ON ac.connection_id = pf.connection_id AND ac.age = 0
			SET pf.strength = LEAST(?, pf.strength + 1)
			WHERE pf.connection_id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
		`, [this.maxConnectionStrength]);
		if (this.debug) console.log(`Strengthened ${strengthenResult.affectedRows} correct pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections
		// Connection is NOT active
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			SET pf.strength = GREATEST(?, pf.strength - ?)
			WHERE pf.connection_id IN (SELECT source_id FROM inference_sources WHERE age = 1 AND source_type = 'pattern')
			AND pf.connection_id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug) console.log(`Weakened ${weakenResult.affectedRows} failed pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Active connections from peak not yet in pattern_future
		// Find patterns that made predictions, get their peak neurons, find active connections from peak
		const [novelResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pf_src.pattern_neuron_id, ac.connection_id, 1.0
			FROM inference_sources isrc
			JOIN pattern_future pf_src ON pf_src.connection_id = isrc.source_id
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
		if (this.debug) console.log(`Added ${novelResult.affectedRows} novel connections to pattern_future`);
	}

	/**
	 * Populate new_patterns table from new_pattern_future.
	 * Finds unique peak neurons (one pattern per peak).
	 * Action neurons can NEVER be peaks - only event neurons and interneurons can be peaks.
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		await this.conn.query(`TRUNCATE new_patterns`);
		const [insertResult] = await this.conn.query(`
			INSERT INTO new_patterns (peak_neuron_id)
			SELECT DISTINCT c.from_neuron_id
			FROM new_pattern_future npf
			JOIN connections c ON c.id = npf.connection_id
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Create pattern neurons and map them to new_patterns.
	 * Creates neurons at peak_level+1 for each peak neuron.
	 * @param {number} patternCount - Number of patterns to create neurons for
	 */
	async createPatternNeurons(patternCount) {

		// Get peak neurons with their levels - use LEFT JOIN to detect missing neurons
		const [peaks] = await this.conn.query(`
			SELECT np.seq_id, np.peak_neuron_id, n.level
			FROM new_patterns np
			LEFT JOIN neurons n ON n.id = np.peak_neuron_id
			ORDER BY n.level, np.seq_id
		`);

		// Check for missing peak neurons (data integrity issue)
		const missingPeaks = peaks.filter(p => p.level === null);
		if (missingPeaks.length > 0) {
			const missingIds = missingPeaks.map(p => p.peak_neuron_id);
			throw new Error(`Data integrity error: peak neurons not found in neurons table: [${missingIds.join(', ')}]`);
		}

		// Group by level
		const byLevel = new Map();
		for (const peak of peaks) {
			if (!byLevel.has(peak.level)) byLevel.set(peak.level, []);
			byLevel.get(peak.level).push(peak);
		}

		// Create pattern neurons for each level
		for (const [level, levelPeaks] of byLevel) {
			const patternNeuronIds = await this.createNeurons(levelPeaks.length, level + 1);

			// Update new_patterns with pattern neuron IDs
			for (let i = 0; i < levelPeaks.length; i++) {
				await this.conn.query(`UPDATE new_patterns SET pattern_neuron_id = ? WHERE seq_id = ?`,
					[patternNeuronIds[i], levelPeaks[i].seq_id]);
			}
		}
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future.
	 * Processes all levels in bulk.
	 */
	async mergeNewPatterns() {

		// Create pattern_peaks entries
		await this.conn.query(`
			INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength)
			SELECT np.pattern_neuron_id, np.peak_neuron_id, 1.0
			FROM new_patterns np
		`);

		// Create pattern_past entries (active connections at age=1 leading TO the peak)
		// This captures the context that was present when the peak was active
		// Includes cross-level connections (no level filter)
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, ac.connection_id, 1.0
			FROM new_patterns np
			JOIN active_connections ac ON ac.to_neuron_id = np.peak_neuron_id
			WHERE ac.age = 1
		`);

		// Create pattern_future entries: connections FROM peak to target neurons
		// Uses new_pattern_future which contains connection_ids
		await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, npf.connection_id, 1.0
			FROM new_patterns np
			JOIN connections c ON c.from_neuron_id = np.peak_neuron_id
			JOIN new_pattern_future npf ON npf.connection_id = c.id
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
	 * Action neurons can NEVER be sources - only event neurons and interneurons can predict.
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
            -- Action neurons can NEVER be sources (only events and interneurons can predict)
            AND NOT EXISTS (
                SELECT 1 FROM coordinates cf
                JOIN dimensions df ON df.id = cf.dimension_id
                WHERE cf.neuron_id = f.neuron_id AND df.type = 'action'
            )
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
			INSERT IGNORE INTO active_connections (connection_id, from_neuron_id, to_neuron_id, age)
			SELECT c.id as connection_id, c.from_neuron_id, c.to_neuron_id, 0 as age
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
		// A pattern matches if at least some percentage of its pattern_past connections are in active_connections
		const [result] = await this.conn.query(`
            INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
            SELECT pp.peak_neuron_id, pp.pattern_neuron_id
            FROM active_neurons an
			JOIN pattern_peaks pp ON an.neuron_id = pp.peak_neuron_id
			JOIN pattern_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.connection_id = p.connection_id AND ac.age = 0
            WHERE an.level = ?
            AND an.age = 0
            GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
            HAVING SUM(IF(ac.connection_id IS NOT NULL, 1, 0)) >= COUNT(*) * ?
		`, [level, this.mergePatternThreshold]);
		if (this.debug) console.log(`Matched ${result.affectedRows} pattern-peak pairs`);
		if (result.affectedRows === 0) return 0;

		// Now populate matched_pattern_connections ONLY for matched patterns
		// Pattern connections: LEFT JOIN to active to determine if common or missing
		// Cross-level connections included via active_connections.level = target level
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, p.connection_id, IF(ac.connection_id IS NOT NULL, 'common', 'missing') as status
			FROM matched_patterns mp
			JOIN pattern_past p ON mp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.connection_id = p.connection_id AND ac.age = 0
		`);

		// Novel connections: active connections TO the peak not in pattern_past
		// Includes cross-level connections (base→high)
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, ac.connection_id, 'novel' as status
			FROM matched_patterns mp
			JOIN active_connections ac ON ac.to_neuron_id = mp.peak_neuron_id AND ac.age = 0
			LEFT JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id AND p.connection_id = ac.connection_id
			WHERE p.connection_id IS NULL
		`);

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
			SET pp.strength = LEAST(?, pp.strength + 1)
		`, [this.maxConnectionStrength]);

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
			SET p.strength = LEAST(?, p.strength + 1)
			WHERE mpc.status = 'common'
		`, [this.maxConnectionStrength]);

		// Weaken missing connections: connections in pattern but not in active (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_connections mpc ON p.pattern_neuron_id = mpc.pattern_neuron_id AND p.connection_id = mpc.connection_id
			SET p.strength = GREATEST(?, p.strength - ?)
			WHERE mpc.status = 'missing'
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);
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

		// 3. REWARD DECAY: Move reward values back toward 0 (neutral)
		// Formula: reward = reward * (1 - rewardForgetRate)
		// reward=+10, rate=0.05 → 10 * 0.95 = 9.5
		// reward=-10, rate=0.05 → -10 * 0.95 = -9.5
		if (this.debug) console.log('Running forget cycle - connection reward decay...');
		stepStart = Date.now();
		const [connRewardResult] = await this.conn.query(`UPDATE connections SET reward = reward * (1 - ?)`, [this.rewardForgetRate]);
		if (this.debug) console.log(`  Connection reward decay took ${Date.now() - stepStart}ms (updated ${connRewardResult.affectedRows} rows)`);

		if (this.debug) console.log('Running forget cycle - pattern_future reward decay...');
		stepStart = Date.now();
		const [patternRewardResult] = await this.conn.query(`UPDATE pattern_future SET reward = reward * (1 - ?)`, [this.rewardForgetRate]);
		if (this.debug) console.log(`  Pattern_future reward decay took ${Date.now() - stepStart}ms (updated ${patternRewardResult.affectedRows} rows)`);

		// 4. PATTERN NEURON CLEANUP: Remove orphaned pattern neurons (level > 0) with no connections or pattern entries
		// Base neurons (level 0) are NEVER deleted - they are fundamental encoding units with coordinates
		// MEMORY engine doesn't enforce foreign keys, so deleting base neurons would leave orphaned coordinates
		// which would cause ghost neurons to be "found" and connections created to non-existent neurons
		if (this.debug) console.log('Running forget cycle - orphaned pattern neurons cleanup...');
		stepStart = Date.now();
		const [neuronDeleteResult] = await this.conn.query(`
			DELETE
			FROM neurons n
			WHERE n.level > 0
			AND NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM pattern_past WHERE pattern_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM pattern_future WHERE pattern_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM pattern_peaks WHERE pattern_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM pattern_peaks WHERE peak_neuron_id = n.id)
			AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
		`);
		if (this.debug) console.log(`  Orphaned pattern neurons DELETE took ${Date.now() - stepStart}ms (deleted ${neuronDeleteResult.affectedRows} rows)`);

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Apply channel-specific rewards to connections/patterns that led to executed outputs.
	 *
	 * Channel-Specific Credit Assignment:
	 * 1. Identify which channel each base-level output belongs to (via output dimensions)
	 * 2. Use inferred_neurons (is_winner) + inference_sources to find which connections/patterns led to each action
	 * 3. Apply channel-specific reward to the connections/patterns based on source_type
	 *
	 * Exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
	 * This converges to the expected reward for each connection.
	 * Neutral reward is 0, positive is good, negative is bad
	 */
	async applyRewards(channelRewards) {

		// nothing to update if there are no rewards
		if (channelRewards.size === 0) return;

		// Process each channel's rewards separately
		let totalWinnerConnections = 0, totalWinnerPatterns = 0;
		for (const [channelName, reward] of channelRewards) {

			if (this.debug) console.log(`Applying reward ${reward.toFixed(3)} for channel: ${channelName}`);

			// Get the output dimension IDs for this channel
			const outputDimIds = this.getChannelOutputDims(channelName);
			if (outputDimIds.length === 0) {
				console.warn(`Warning: No output dimensions found for channel ${channelName}`);
				continue;
			}

			// WINNERS: Apply actual reward to winning votes (is_winner=1)
			// Exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
			// Reward connection-based inferences
			const [winnerConnResult] = await this.conn.query(`
				UPDATE connections c
				JOIN inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET c.reward = :smooth * :reward + (1 - :smooth) * c.reward
				WHERE isrc.age > 0 AND isrc.age <= :maxAge
				AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (:dimIds)
				)
			`, { smooth: this.rewardExpSmooth, reward, maxAge: this.maxRewardsAge, dimIds: outputDimIds });
			totalWinnerConnections += winnerConnResult.affectedRows;

			// Reward pattern-based inferences (pattern_future stores connection_id)
			const [winnerPatternResult] = await this.conn.query(`
				UPDATE pattern_future pf
				JOIN inference_sources isrc ON pf.connection_id = isrc.source_id AND isrc.source_type = 'pattern'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				SET pf.reward = :smooth * :reward + (1 - :smooth) * pf.reward
				WHERE isrc.age > 0 AND isrc.age <= :maxAge
				AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = isrc.neuron_id AND coord.dimension_id IN (:dimIds)
				)
			`, { smooth: this.rewardExpSmooth, reward, maxAge: this.maxRewardsAge, dimIds: outputDimIds });
			totalWinnerPatterns += winnerPatternResult.affectedRows;

			// Update inferred_neurons.actual_reward for action winners so that we can detect negative rewards for new patterns
			await this.conn.query(`
				UPDATE inferred_neurons inf
				SET inf.actual_reward = ?
				WHERE inf.age = 1 AND inf.level = 0 AND inf.is_winner = 1
				AND EXISTS (
					SELECT 1 FROM coordinates coord
					WHERE coord.neuron_id = inf.neuron_id AND coord.dimension_id IN (?)
				)
			`, [reward, outputDimIds]);

			// LOSERS: Leave alone - we don't know what would have happened if they were executed

			if (this.debug) console.log(`  ${channelName}: winners=${winnerConnResult.affectedRows}c/${winnerPatternResult.affectedRows}p (${reward.toFixed(2)})`);
		}

		if (this.debug) console.log(`Total rewarded: winners=${totalWinnerConnections}c/${totalWinnerPatterns}p`);
		// await this.waitForUser('Rewards applied');
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