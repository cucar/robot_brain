import readline from 'node:readline';
import getMySQLConnection from './db/db.js';

/**
 * Artificial Brain
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.baseNeuronMaxAge = 5; // number of frames a base neuron stays active
		this.forgetCycles = 100; // number of frames between forget cycles
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle
		this.maxLevels = 6; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.20; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.minPeakStrength = 10.0; // minimum weighted strength for a neuron to be considered a peak (used for both pattern detection and prediction)
		this.minPeakRatio = 1.4; // minimum ratio of peak strength to neighborhood average to be considered a peak (used for both pattern detection and prediction)
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.9; // reward temporal decay = POW(rewardTimeDecayFactor, age)
		this.patternNegativeReinforcement = 1; // how much to weaken pattern connections that were not observed
		this.negativeLearningRate = 1; // how much to weaken connections when predictions fail

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;
		this.inactivityThreshold = 5; // frames of inactivity before exploration

		// Create readline interface for pausing between frames - used when debugging
		this.debug = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
	}

	/**
	 * Register a channel with the brain
	 */
	registerChannel(name, channelClass) {
		const channel = new channelClass(name);
		this.channels.set(name, channel);
		console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (memory tables)...');
		await this.truncateTables([
			'active_neurons',
			'connection_inference',
			'inferred_neurons',
			'observed_patterns',
			'active_connections'
		]);
	}

	/**
	 * Hard reset: clears ALL tables (used mainly for tests)
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all tables)...');
		await this.truncateTables([
			'active_neurons',
			'connection_inference',
			'inferred_neurons',
			'observed_patterns',
			'active_connections',
			'matched_patterns',
			'pattern_peaks',
			'patterns',
			'connections',
			'coordinates',
			'neurons',
			'dimensions'
		]);
	}

	/**
	 * truncates given tables for database reset
	 */
	async truncateTables(tables) {
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		for (const table of tables) await this.conn.query(`TRUNCATE ${table}`);
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async init() {

		// get new database connection
		this.conn = await getMySQLConnection();

		// create dimensions for all registered channels
		await this.initializeDimensions();

		// load the dimensions
		await this.loadDimensions();

		// initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) await channel.initialize();
	}

	/**
	 * Initialize dimensions for all registered channels
	 */
	async initializeDimensions() {
		console.log('Initializing dimensions for registered channels...');
		for (const [channelName, channel] of this.channels) {
			await this.insertChannelDimensions(channel.getInputDimensions(), channelName, 'input');
			await this.insertChannelDimensions(channel.getOutputDimensions(), channelName, 'output');
		}
	}

	/**
	 * inserts channel dimensions
	 */
	async insertChannelDimensions(dimensions, channelName, type){
		console.log(`Creating ${type} dimensions for ${channelName}:`, dimensions);
		for (const dimName of dimensions)
			await this.conn.query('INSERT IGNORE INTO dimensions (name, channel, type) VALUES (?, ?, ?)', [dimName, channelName, type]);
	}

	/**
	 * loads the dimensions to memory for input
	 */
	async loadDimensions() {
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};
		const [rows] = await this.conn.query('SELECT id, name FROM dimensions');
		console.log(rows);
		rows.forEach(row => {
			this.dimensionNameToId[row.name] = row.id;
			this.dimensionIdToName[row.id] = row.name;
		});
		console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * returns the current frame combined from all registered channels
	 */
	async getFrame() {
		const frame = [];

		// Increment frame counter to be able to track inactivity
		this.frameNumber++;

		// Get input data from all channels
		for (const [_, channel] of this.channels) {
			const channelInputs = await channel.getFrameInputs();
			if (channelInputs && channelInputs.length > 0) frame.push(...channelInputs);
		}

		return frame;
	}

	/**
	 * Get global feedback from all channels aggregated into a single reward factor
	 */
	async getFeedback() {
		console.log('Getting feedback from all channels...');
		let globalReward = 1.0; // Start with neutral
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const rewardFactor = await channel.getFeedback();
			if (rewardFactor !== 1.0) { // Only process non-neutral feedback
				console.log(`${channelName}: reward factor ${rewardFactor.toFixed(3)}`);
				globalReward *= rewardFactor; // Multiplicative aggregation
				feedbackCount++;
			}
		}

		if (feedbackCount > 0) console.log(`Total reward: ${globalReward.toFixed(3)} (${feedbackCount} channels)`);
		else console.log('No feedback from any channels');

		return globalReward;
	}

	/**
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and global reward factor from aggregated channel feedback
	 */
	async processFrame(frame, globalReward = 1.0) {
		console.log('******************************************************************');
		console.log(`OBSERVING NEW FRAME: ${JSON.stringify(frame)}`, this.frameNumber);
		console.log(`applying global reward: ${globalReward.toFixed(3)}`);
		console.log('******************************************************************');

		// do the whole thing as a transaction to avoid inconsistent database states
		await this.conn.beginTransaction();

		// if there's an error, we'll roll the transaction back
		try {

			// apply rewards to previously executed decisions (before aging them further)
			await this.applyRewards(globalReward);

			// age the active neurons in memory context - sliding the temporal window
			await this.ageNeurons();

			// execute previous frame's decisions + exploration if needed
			await this.executeOutputs();

			// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
			await this.recognizeNeurons(frame);

			// do predictions and outputs - what's going to happen next? and what's our best response?
			await this.inferNeurons();

			await this.conn.commit();
			console.log('Frame processed successfully.');
		}
		catch (error) {
			await this.conn.rollback();
			console.error('Error processing frame, transaction rolled back:', error);
			throw error;
		}

		// run forget cycle periodically and delete dead connections/neurons
		await this.runForgetCycle();

		// when debugging, wait for user to press Enter before continuing to next frame
		if (this.debug) await this.waitForUser('Press Enter to continue to next frame');
	}

	/**
	 * waits for user input to continue - used for debugging
	 */
	waitForUser(message) {
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Execute previous frame's decisions and exploration actions if needed
	 */
	async executeOutputs() {

		// Execute previous frame's decisions (age = 1)
		await this.executePreviousOutputs();

		// Execute exploration if brain is inactive
		await this.curiosityExploration();
	}

	/**
	 * Execute decisions from previous frame (age = 1)
	 */
	async executePreviousOutputs() {
		const [outputRows] = await this.conn.query(`
			SELECT inf.neuron_id, c.dimension_id, c.val, d.name as dimension_name, d.channel
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE d.type = 'output' AND inf.age = 1 AND inf.level = 0
			ORDER BY d.channel, d.name
		`);

		if (outputRows.length === 0) {
			console.log('No previous outputs to execute');
			return;
		}

		await this.executeOutputRows(outputRows);
	}

	/**
	 * Execute curiosity exploration if brain is inactive
	 */
	async curiosityExploration() {

		// Check if the brain is inactive - if active, no exploration needed
		if ((this.frameNumber - this.lastActivity) < this.inactivityThreshold) return;
		console.log('Brain inactive - executing curiosity exploration');

		// Get a random channel for exploration
		const channelNames = Array.from(this.channels.keys());
		const randomChannelName = channelNames[Math.floor(Math.random() * channelNames.length)];
		const randomChannel = this.channels.get(randomChannelName);

		// Get exploration actions for the channel
		const explorationActions = randomChannel.getValidExplorationActions();
		if (explorationActions.length === 0) {
			console.log(`No valid exploration actions for ${randomChannelName}`);
			return;
		}

		// Execute random exploration action
		const randomAction = explorationActions[Math.floor(Math.random() * explorationActions.length)];
		console.log(`${randomChannelName}: Executing exploration action:`, randomAction);

		await this.executeChannelOutputs(randomChannelName, randomAction);
	}

	/**
	 * Execute output rows grouped by channel
	 */
	async executeOutputRows(outputRows) {

		// Group outputs by channel
		const channelOutputs = new Map();

		for (const row of outputRows) {
			if (!channelOutputs.has(row.channel)) channelOutputs.set(row.channel, new Map());
			channelOutputs.get(row.channel).set(row.dimension_name, row.val);
		}

		// Execute outputs for each channel using unified method
		for (const [channelName, outputs] of channelOutputs) {
			const coordinates = Object.fromEntries(outputs);
			await this.executeChannelOutputs(channelName, coordinates);
		}
	}

	/**
	 * Unified method to execute outputs on a specific channel
	 */
	async executeChannelOutputs(channelName, coordinates) {
		const channel = this.channels.get(channelName);
		if (!channel) {
			console.log(`Warning: Channel ${channelName} not found`);
			return;
		}

		console.log(`${channelName}: Executing outputs:`, coordinates);
		await channel.executeOutputs(coordinates);

		// Track global activity
		this.lastActivity = this.frameNumber;
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= baseNeuronMaxAge.
	 */
	async ageNeurons() {
		console.log('Aging active neurons, connections, and inferred neurons...');

		// age all neurons and connections - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE active_connections SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.baseNeuronMaxAge]);
		console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.baseNeuronMaxAge]);
		console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Clean up inferred neurons after execution (age >= 2)
		// age=0: fresh predictions, age=1: executed this frame, age>=2: no longer needed
		const [inferredResult] = await this.conn.query('DELETE FROM inferred_neurons WHERE age >= 2');
		console.log(`Cleaned up ${inferredResult.affectedRows} executed inferred neurons (age >= 2)`);
	}

	/**
	 * recognizes and activates neurons from frame - returns the highest level of recognition reached
	 */
	async recognizeNeurons(frame) {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(frame);

		// bulk insert activations at base level
		await this.activateNeurons(neuronIds);

		// discover and activate patterns using connections - start recursion from base level
		await this.activatePatternNeurons();
	}

	/**
	 * Infer predictions and outputs starting from the highest active level down to base level.
	 * Connection inference: Predict next frame's neurons from connections (with negative reinforcement).
	 * Pattern inference: Predict lower-level peak neurons from higher-level pattern neurons.
	 */
	async inferNeurons() {

		// Get the highest level that is currently active
		const maxLevel = await this.getMaxActiveLevel();

		// Process levels in reverse: maxLevel, maxLevel-1, ..., 0
		for (let level = maxLevel; level >= 0; level--) {

			// Connection inference: Predict connections for age=-1 at this level
			await this.inferConnections(level);

			// Pattern inference: Predict peak neurons for the lower level
			if (level > 0) await this.inferPatterns(level);
		}
	}


	/**
	 * reports accuracy of neuron predictions from the previous frame for accuracy reporting.
	 * Check which predicted neurons (from inferred_neurons) actually got activated.
	 */
	async reportPredictionsAccuracy(level) {

		// Get predictions from previous frame (age=1 after aging)
		const [predictions] = await this.conn.query('SELECT neuron_id FROM inferred_neurons WHERE level = ? AND age = 1', [level]);
		if (predictions.length === 0) {
			console.log(`Level ${level}: No predictions to validate`);
			return;
		}

		// Find which predictions came true (exist in active_neurons at age=0)
		const [matches] = await this.conn.query(`
			SELECT inf.neuron_id
			FROM inferred_neurons inf
			JOIN active_neurons an ON inf.neuron_id = an.neuron_id
			WHERE inf.level = ?
			AND inf.age = 1
			AND an.level = ?
			AND an.age = 0
		`, [level, level]);

		const successRate = predictions.length > 0 ? (matches.length / predictions.length * 100).toFixed(1) : 0;
		console.log(`Level ${level}: Prediction accuracy: ${matches.length}/${predictions.length} (${successRate}%)`);
	}

	/**
	 * Validate connection predictions from the previous frame.
	 * Apply negative reinforcement to connections that predicted incorrectly.
	 */
	async validateConnectionPredictions(level) {

		// get the connections used for predictions from previous frame
		const [predictions] = await this.conn.query('SELECT connection_id FROM connection_inference WHERE level = ?', [level]);
		if (predictions.length === 0) return;

		// Find which predictions failed (not in active_connections)
		const [failures] = await this.conn.query(`
			SELECT ci.connection_id
			FROM connection_inference ci
			LEFT JOIN active_connections ac ON ci.connection_id = ac.connection_id AND ac.level = ? AND ac.age = 0
			WHERE ci.level = ?
			AND ac.connection_id IS NULL
		`, [level, level]);

		// if there are no failed predictions, no need to apply negative reinforcement
		if (failures.length === 0) return;

		// Apply negative reinforcement to failed predictions
		const failedConnectionIds = failures.map(f => f.connection_id);
		await this.conn.query('UPDATE connections SET strength = strength - ? WHERE id IN (?)', [this.negativeLearningRate, failedConnectionIds]);
		console.log(`Level ${level}: Applied negative reinforcement to ${failures.length} failed connection predictions`);
	}

	/**
	 * Connection inference: Predict next frame's neurons from active connections.
	 * Stores predicted connection_ids in connection_inference scratch table.
	 * Validates previous frame's predictions and applies negative reinforcement to failed predictions.
	 */
	async inferConnections(level) {

		// report the neuron prediction accuracy from previous frame
		await this.reportPredictionsAccuracy(level);

		// validate predictions from the previous frame (before clearing)
		await this.validateConnectionPredictions(level);

		// Clear previous predictions for this level
		await this.conn.query('DELETE FROM connection_inference WHERE level = ?', [level]);

		// Predict connections using the same peak detection algorithm as detectPeaks
		// but for one frame into the future, using all possible connections from active neurons
		await this.conn.query(`
			INSERT INTO connection_inference (level, connection_id)
			WITH candidate_connections AS (
				-- Get all possible connections from active neurons (one frame into the future)
				SELECT
					c.id as connection_id,
					c.from_neuron_id,
					c.to_neuron_id,
					c.strength * POW(:peakTimeDecayFactor, c.distance) as strength
				FROM active_neurons f
				JOIN connections c ON c.from_neuron_id = f.neuron_id
				WHERE f.level = :level
				AND c.distance = f.age + 1
				AND c.strength > 0
			),
			neuron_strengths AS (
				-- Calculate total strength for each neuron (sum of all connections it participates in)
				SELECT neuron_id, SUM(strength) as total_strength
				FROM (
					SELECT from_neuron_id as neuron_id, strength FROM candidate_connections
					UNION ALL
					SELECT to_neuron_id as neuron_id, strength FROM candidate_connections
				) all_neuron_connections
				GROUP BY neuron_id
			),
			neighborhood_strengths AS (
				-- Calculate average from_neuron strength for each to_neuron (neighborhood context)
				-- Use DISTINCT to count each from_neuron only once (in case of multiple connections)
				SELECT to_neuron_id, AVG(total_strength) as avg_neighborhood_strength
				FROM (
					SELECT DISTINCT cc.to_neuron_id, cc.from_neuron_id, ns.total_strength
					FROM candidate_connections cc
					JOIN neuron_strengths ns ON cc.from_neuron_id = ns.neuron_id
				) unique_neighbors
				GROUP BY to_neuron_id
			),
			peaks AS (
				-- Find peaks: to_neurons with strength >= minPeakStrength AND ratio > minPeakRatio
				-- Using multiplication instead of division for better index usage
				SELECT ns.neuron_id as peak_neuron_id
				FROM neuron_strengths ns
				JOIN neighborhood_strengths nhs ON ns.neuron_id = nhs.to_neuron_id
				WHERE ns.total_strength >= :minPeakStrength
				AND ns.total_strength > (nhs.avg_neighborhood_strength * :minPeakRatio)
			)
			-- Insert all connections for peak neurons
			SELECT :level, cc.connection_id
			FROM peaks p
			JOIN candidate_connections cc ON p.peak_neuron_id = cc.to_neuron_id
		`, {
			level,
			peakTimeDecayFactor: this.peakTimeDecayFactor,
			minPeakStrength: this.minPeakStrength,
			minPeakRatio: this.minPeakRatio
		});

		// Infer neurons from predicted connections (to_neuron_id at age=-1)
		// Use INSERT IGNORE because the same neuron may have been inferred from pattern inference
		await this.conn.query(`
			INSERT IGNORE INTO inferred_neurons (neuron_id, level, age)
			SELECT DISTINCT c.to_neuron_id, :level, 0
			FROM connection_inference ci
			JOIN connections c ON ci.connection_id = c.id
			WHERE ci.level = :level
		`, { level });

		const [result] = await this.conn.query(
			'SELECT COUNT(DISTINCT connection_id) as count FROM connection_inference WHERE level = ?',
			[level]
		);
		console.log(`Level ${level}: Predicted ${result[0].count} connections for next frame`);
	}

	/**
	 * Pattern inference: Predict lower-level peak neurons from higher-level pattern neurons.
	 * For each inferred pattern neuron at level N (from connection inference),
	 * predict its peak neuron at level N-1, age=-1.
	 * No negative reinforcement needed - pattern merge handles positive reinforcement,
	 * and connection inference handles negative reinforcement for pattern neuron predictions.
	 */
	async inferPatterns(level) {
		console.log(`Level ${level}: Inferring peak neurons for level ${level - 1}`);

		// Infer peak neurons from inferred pattern neurons
		// Pattern neuron at level N → Peak neuron at level N-1
		// This runs before inferConnections for the lower level, so no duplicates expected
		await this.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age)
			SELECT DISTINCT pp.peak_neuron_id, :targetLevel, 0
			FROM inferred_neurons inf
			JOIN pattern_peaks pp ON inf.neuron_id = pp.pattern_neuron_id
			WHERE inf.level = :level
			AND inf.age = 0
		`, { level, targetLevel: level - 1 });

		const [result] = await this.conn.query(`
			SELECT COUNT(DISTINCT neuron_id) as count
			FROM inferred_neurons
			WHERE level = ? AND age = 0
		`, [level - 1]);
		console.log(`Level ${level}: Predicted ${result[0].count} peak neurons for level ${level - 1}`);
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame
	 */
	async getFrameNeurons(frame) {

		// try tp get all the neurons that have coordinates in close ranges for each point - return format: [{ point_str, neuron_id }]
		const matches = await this.matchFrameNeurons(frame);
		console.log('pointNeuronMatches', matches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const neuronIds = matches.filter(p => p.neuron_id).map(p => p.neuron_id);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => !p.neuron_id);
		if (pointsNeedingNeurons.length > 0) {
			console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with dedupe.`);
			const createdNeuronIds = await this.createBaseNeurons(pointsNeedingNeurons.map(p => p.point_str));
			neuronIds.push(...createdNeuronIds);
		}

		// return matching neuron ids to given points
		console.log('frame neurons', neuronIds);
		return neuronIds;
	}

	/**
	 * matches base neurons from dimensional values for each point - return format: [{ point_str, neuron_id }]
	 */
	async matchFrameNeurons(frame) {
		if (frame.length === 0) return [];
		const neuronCoords = await this.getFrameCoordinates(frame);
		return this.findFrameMatches(frame, neuronCoords);
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
	 * finds exact neuron matches for each point in the frame
	 */
	findFrameMatches(frame, neuronCoords) {
		const results = [];

		for (const point of frame) {
			const pointStr = JSON.stringify(point);
			const matchedNeuronId = this.findPointMatch(point, neuronCoords);
			results.push({ point_str: pointStr, neuron_id: matchedNeuronId });
		}

		return results;
	}

	/**
	 * finds the neuron that exactly matches a single point
	 */
	findPointMatch(point, neuronCoords) {
		const pointStr = JSON.stringify(point);
		const expectedDimCount = Object.keys(point).length;
		let matchedNeuronId = null;

		for (const [neuronId, coords] of neuronCoords) {
			let matchCount = 0;
			for (const [dimName, val] of Object.entries(point)) {
				const dimId = this.dimensionNameToId[dimName];
				if (coords.has(dimId) && coords.get(dimId) === val) matchCount++;
			}

			if (matchCount === expectedDimCount) {
				if (matchedNeuronId !== null)
					throw new Error(`Multiple neuron matches for point: ${pointStr}`);
				matchedNeuronId = neuronId;
			}
		}

		return matchedNeuronId;
	}

	/**
	 * creates base neurons from a given set of points and returns their ids
	 */
	async createBaseNeurons(pointStrs) {

		// nothing to create if no points are sent - should not happen
		if (pointStrs.length === 0) return [];

		// deduplicate points and parse them
		const points = [...new Set(pointStrs)].map(pointStr => JSON.parse(pointStr));

		// bulk insert neurons and get their IDs
		const neuronIds = await this.bulkInsertNeurons(points.length);
		const created = points.map((point, idx) => ({ point, neuron_id: neuronIds[idx] }));

		// Rest of the coordinate insertion logic with batching for large frames
		const rows = created.flatMap(({ neuron_id, point }) =>
			Object.entries(point).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));

		// Process in batches to avoid query size limits
		const batchSize = 5000;
		for (let i = 0; i < rows.length; i += batchSize) {
			const batch = rows.slice(i, i + batchSize);
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [batch]);
		}

		// return the new neuron ids
		return neuronIds;
	}

	/**
	 * Bulk insert neurons and return their IDs.
	 * MySQL guarantees sequential auto-increment IDs.
	 * @param {number} count - Number of neurons to create
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async bulkInsertNeurons(count) {
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
            SELECT
                f.neuron_id as from_neuron_id,
                t.neuron_id as to_neuron_id,
                f.age as distance,
                1 as strength
			FROM active_neurons f
			CROSS JOIN active_neurons t
            WHERE t.age = 0  -- target neurons are newly activated
            AND t.level = :level  -- target neurons are at the specified level
            AND f.level = t.level  -- restrict to same level only
            AND (t.neuron_id != f.neuron_id OR f.age != 0)  -- no self-connections at same age
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)
		`, { level });
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
	 * detects all spatial levels in age=0 neurons using unified connections - start from base level and go as high as possible
	 */
	async activatePatternNeurons() {
		let level = 0;
		while (true) {

			// process the level to detect patterns - returns if there are patterns found or not
			const patternsFound = await this.activateLevelPatterns(level);

			// if no patterns are found in the level, nothing to do
			if (!patternsFound) break;

			// increment the level to start processing it
			level++;

			// if we exceeded the max level, give warning and stop
			if (level >= this.maxLevels) {
				console.error('Max level exceeded.');
				break;
			}
		}
	}

	/**
	 * Processes a level to detect patterns and activate them
	 * Returns true if patterns were found, false otherwise
	 */
	async activateLevelPatterns(level) {
		console.log(`Processing level ${level} for pattern recognition`);

		// Detect peaks and write to observed_patterns table
		const hasPeaks = await this.detectPeaks(level);
		if (!hasPeaks) {
			console.log(`No peaks found at level ${level}`);
			return false;
		}

		// Match observed patterns to known patterns and write to matched_patterns table
		await this.matchPatternNeurons();

		// Merge matched patterns: add/strengthen observed connections, weaken unobserved connections
		await this.mergeMatchedPatterns();

		// Create new patterns for peaks without matches (also adds to matched_patterns table)
		await this.createNewPatterns();

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
			AND (t.neuron_id != f.neuron_id OR f.age != 0)  -- no self-connections at same age
			AND c.strength > 0  -- only connections that are not removed
		`, { level });
	}

	/**
	 * Detect peaks and write directly to observed_patterns table
	 * Peaks are to_neurons whose strength exceeds their source neurons' average strength
	 * @param {number} level - The level to detect peaks for
	 */
	async detectPeaks(level) {

		// Clear observed_patterns table
		await this.conn.query('TRUNCATE observed_patterns');

		// Detect peaks and insert into observed_patterns in one query
		// Logic:
		// 1. Calculate neuron strengths (sum of all connection strengths each neuron participates in)
		// 2. Calculate neighborhood strength (average from_neuron strength for each to_neuron)
		// 3. Find peaks where to_neuron strength >= minPeakStrength AND ratio > minPeakRatio
		// 4. Insert all connections for those peaks into observed_patterns
		await this.conn.query(`
			INSERT INTO observed_patterns (peak_neuron_id, connection_id)
			WITH connection_data AS (
				-- Get all active connections with their strengths
				SELECT
					ac.connection_id,
					ac.from_neuron_id,
					ac.to_neuron_id,
					c.strength * POW(:peakTimeDecayFactor, c.distance) as strength
				FROM active_connections ac
				JOIN connections c ON ac.connection_id = c.id
				WHERE ac.level = :level
				AND ac.age = 0
				AND c.strength > 0
			),
			neuron_strengths AS (
				-- Calculate total strength for each neuron (sum of all connections it participates in)
				SELECT neuron_id, SUM(strength) as total_strength
				FROM (
					SELECT from_neuron_id as neuron_id, strength FROM connection_data
					UNION ALL
					SELECT to_neuron_id as neuron_id, strength FROM connection_data
				) all_neuron_connections
				GROUP BY neuron_id
			),
			neighborhood_strengths AS (
				-- Calculate average from_neuron strength for each to_neuron (neighborhood context)
				-- Use DISTINCT to count each from_neuron only once (in case of multiple connections)
				SELECT to_neuron_id, AVG(total_strength) as avg_neighborhood_strength
				FROM (
					SELECT DISTINCT cd.to_neuron_id, cd.from_neuron_id, ns.total_strength
					FROM connection_data cd
					JOIN neuron_strengths ns ON cd.from_neuron_id = ns.neuron_id
				) unique_neighbors
				GROUP BY to_neuron_id
			),
			peaks AS (
				-- Find peaks: to_neurons with strength >= minPeakStrength AND ratio > minPeakRatio
				-- Using multiplication instead of division for better index usage
				SELECT ns.neuron_id as peak_neuron_id
				FROM neuron_strengths ns
				JOIN neighborhood_strengths nhs ON ns.neuron_id = nhs.to_neuron_id
				WHERE ns.total_strength >= :minPeakStrength
				AND ns.total_strength > (nhs.avg_neighborhood_strength * :minPeakRatio)
			)
			-- Insert all connections for peak neurons
			SELECT p.peak_neuron_id, cd.connection_id
			FROM peaks p
			JOIN connection_data cd ON p.peak_neuron_id = cd.to_neuron_id
		`, {
			level,
			peakTimeDecayFactor: this.peakTimeDecayFactor,
			minPeakStrength: this.minPeakStrength,
			minPeakRatio: this.minPeakRatio
		});

		// Get count for logging
		const [result] = await this.conn.query('SELECT COUNT(DISTINCT peak_neuron_id) as peak_count FROM observed_patterns');
		const peakCount = result[0].peak_count;
		console.log(`Found ${peakCount} peaks at level ${level}`);

		return peakCount > 0;
	}

	/**
	 * Match observed patterns to known patterns owned by the peak neuron.
	 * Writes results to matched_patterns memory table.
	 * Each peak neuron only reviews patterns it learned before (via pattern_peaks table).
	 * Matches by connection_id (which encodes from_neuron + to_neuron + distance) to preserve temporal structure.
	 * Uses connection overlap (66% threshold) to determine if patterns match.
	 */
	async matchPatternNeurons() {

		// Clear matched_patterns table
		await this.conn.query('TRUNCATE matched_patterns');

		// Find matching patterns and insert into matched_patterns
		await this.conn.query(`
			INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
			WITH observed_pattern_matches AS (
				-- Start from observed patterns and find matching patterns owned by the peak
				-- PERFORMANCE: Starts from small set (current frame's observed patterns)
				-- PERFORMANCE: Only matches patterns owned by the peak neuron (via pattern_peaks)
				SELECT DISTINCT pp.pattern_neuron_id, op.peak_neuron_id
				FROM observed_patterns op
				JOIN patterns p ON op.connection_id = p.connection_id
				JOIN pattern_peaks pp ON p.pattern_neuron_id = pp.pattern_neuron_id AND pp.peak_neuron_id = op.peak_neuron_id
				WHERE p.strength > 0
			),
			candidate_pattern_connections AS (
				-- Get all connection_ids for each candidate pattern
				SELECT pattern_neuron_id, connection_id
				FROM patterns
				WHERE pattern_neuron_id IN (SELECT pattern_neuron_id FROM observed_pattern_matches)
				AND strength > 0
			)
			-- Calculate overlap percentage and return matching peak-pattern pairs
			-- at least 66% of the known pattern's connections should be part of the observed pattern to be matched
			SELECT opm.peak_neuron_id, opm.pattern_neuron_id
			FROM observed_pattern_matches opm
			JOIN candidate_pattern_connections cpc ON opm.pattern_neuron_id = cpc.pattern_neuron_id
			LEFT JOIN observed_patterns op ON opm.peak_neuron_id = op.peak_neuron_id AND op.connection_id = cpc.connection_id
			GROUP BY opm.peak_neuron_id, opm.pattern_neuron_id
			HAVING (COUNT(DISTINCT CASE WHEN op.connection_id IS NOT NULL THEN cpc.connection_id END) / COUNT(DISTINCT cpc.connection_id)) >= ?
		`, [this.mergePatternThreshold]);

		// Get count for logging
		const [result] = await this.conn.query('SELECT COUNT(*) as match_count FROM matched_patterns');
		console.log(`Matched ${result[0].match_count} pattern-peak pairs`);
	}

	/**
	 * Merge matched patterns with observed patterns using pure SQL:
	 * 1. Add new connections that weren't in the pattern before
	 * 2. Strengthen connections that were observed (positive reinforcement)
	 * 3. Weaken connections that were NOT observed (negative reinforcement)
	 */
	async mergeMatchedPatterns() {

		// Positive reinforcement: Add/strengthen observed connections
		await this.conn.query(`
			INSERT INTO patterns (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT mp.pattern_neuron_id, op.connection_id, 1
			FROM matched_patterns mp
			JOIN observed_patterns op ON mp.peak_neuron_id = op.peak_neuron_id
			ON DUPLICATE KEY UPDATE strength = strength + 1
		`);

		// Negative reinforcement: Weaken unobserved connections
		await this.conn.query(`
			UPDATE patterns p
			JOIN matched_patterns mp ON p.pattern_neuron_id = mp.pattern_neuron_id
			LEFT JOIN observed_patterns op ON mp.peak_neuron_id = op.peak_neuron_id AND p.connection_id = op.connection_id
			SET p.strength = p.strength - ?
			WHERE op.connection_id IS NULL
		`, [this.patternNegativeReinforcement]);
	}

	/**
	 * Create new pattern neurons for peaks that don't have any matching patterns.
	 * Leverages MySQL's sequential auto-increment IDs to map peaks to new pattern neurons.
	 * No scratch table needed - pattern_peaks table establishes the mapping directly.
	 */
	async createNewPatterns() {

		// Find peaks that need new patterns (peaks in observed_patterns but not in matched_patterns)
		// Order by peak_neuron_id for deterministic mapping
		const [peaksNeedingPatterns] = await this.conn.query(`
			SELECT DISTINCT op.peak_neuron_id
			FROM observed_patterns op
			LEFT JOIN matched_patterns mp ON op.peak_neuron_id = mp.peak_neuron_id
			WHERE mp.peak_neuron_id IS NULL
			ORDER BY op.peak_neuron_id
		`);
		const count = peaksNeedingPatterns.length;
		console.log(`Creating ${count} new patterns for peaks without matches`);
		if (peaksNeedingPatterns.length === 0) return;

		// Bulk create new pattern neurons - IDs are sequential
		const patternNeuronIds = await this.bulkInsertNeurons(count);

		// Insert pattern_peaks mappings using sequential IDs from bulkInsertNeurons
		const patternPeakMappings = [];
		for (let i = 0; i < count; i++)
			patternPeakMappings.push([ patternNeuronIds[i], peaksNeedingPatterns[i].peak_neuron_id ]);

		// Insert pattern_peaks - should never have duplicates since we're using fresh neuron IDs
		// If we get a duplicate key error, it means we have a bug (reusing a pattern neuron ID)
		await this.conn.query(
			'INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id) VALUES ?',
			[patternPeakMappings]
		);

		// Bulk insert pattern-connection relationships using explicit pattern neuron IDs
		// Safe for concurrent processes - uses exact IDs instead of range
		await this.conn.query(`
			INSERT INTO patterns (pattern_neuron_id, connection_id, strength)
			SELECT pp.pattern_neuron_id, op.connection_id, 1
			FROM pattern_peaks pp
			JOIN observed_patterns op ON pp.peak_neuron_id = op.peak_neuron_id
			WHERE pp.pattern_neuron_id IN (?)
		`, [patternNeuronIds]);

		// Add newly created patterns to matched_patterns so they get activated at the next level
		await this.conn.query(`
			INSERT INTO matched_patterns (pattern_neuron_id, peak_neuron_id)
			SELECT pattern_neuron_id, peak_neuron_id
			FROM pattern_peaks
			WHERE pattern_neuron_id IN (?)
		`, [patternNeuronIds]);
	}

	/**
	 * returns the maximum level from active neurons
	 */
	async getMaxActiveLevel() {
		const [rows] = await this.conn.query('SELECT MAX(level) as max_level FROM active_neurons');
		return rows[0].max_level || 0;
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

		console.log('=== FORGET CYCLE STARTING ===');
		const cycleStart = Date.now();

		// 1. PATTERN FORGETTING: Reduce pattern strengths and remove dead patterns
		console.log('Running forget cycle - pattern update...');
		let stepStart = Date.now();
		const [patternUpdateResult] = await this.conn.query(`UPDATE patterns SET strength = strength - ? WHERE strength > 0`, [this.patternForgetRate]);
		console.log(`  Pattern UPDATE took ${Date.now() - stepStart}ms (updated ${patternUpdateResult.affectedRows} rows)`);

		console.log('Running forget cycle - pattern delete...');
		stepStart = Date.now();
		const [patternDeleteResult] = await this.conn.query(`DELETE FROM patterns WHERE strength <= 0`);
		console.log(`  Pattern DELETE took ${Date.now() - stepStart}ms (deleted ${patternDeleteResult.affectedRows} rows)`);

		// 2. CONNECTION FORGETTING: Reduce connection strengths and remove dead connections
		console.log('Running forget cycle - connection update...');
		stepStart = Date.now();
		const [connectionUpdateResult] = await this.conn.query(`UPDATE connections SET strength = strength - ? WHERE strength > 0`, [this.connectionForgetRate]);
		console.log(`  Connection UPDATE took ${Date.now() - stepStart}ms (updated ${connectionUpdateResult.affectedRows} rows)`);

		console.log('Running forget cycle - connection delete...');
		stepStart = Date.now();
		const [connectionDeleteResult] = await this.conn.query(`DELETE FROM connections WHERE strength <= 0`);
		console.log(`  Connection DELETE took ${Date.now() - stepStart}ms (deleted ${connectionDeleteResult.affectedRows} rows)`);

		// 3. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
		// First, explicitly delete from pattern_peaks to avoid CASCADE timing issues
		console.log('Running forget cycle - pattern_peaks cleanup...');
		stepStart = Date.now();
		const [peaksDeleteResult] = await this.conn.query(`
			DELETE pp FROM pattern_peaks pp
			LEFT JOIN patterns p ON pp.pattern_neuron_id = p.pattern_neuron_id
			WHERE p.pattern_neuron_id IS NULL
		`);
		console.log(`  Pattern_peaks DELETE took ${Date.now() - stepStart}ms (deleted ${peaksDeleteResult.affectedRows} rows)`);

		// Then delete orphaned neurons
		console.log('Running forget cycle - orphaned neurons cleanup...');
		stepStart = Date.now();
		const [neuronDeleteResult] = await this.conn.query(`
			DELETE FROM neurons n
			WHERE NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM patterns WHERE pattern_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
		`);
		console.log(`  Orphaned neurons DELETE took ${Date.now() - stepStart}ms (deleted ${neuronDeleteResult.affectedRows} rows)`);

		console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Apply global reward to active connections that led to executed outputs.
	 * Strengthens connections for positive rewards, weakens for negative rewards.
	 * Uses exponential temporal decay - older connections get less reward/punishment.
	 */
	async applyRewards(globalReward) {
		if (globalReward === 1.0) {
			console.log('Neutral global reward - no updates needed');
			return;
		}

		// Calculate reward adjustment: positive reward strengthens, negative weakens
		// globalReward = 1.5 → adjustment = +0.5 per connection
		// globalReward = 0.5 → adjustment = -0.5 per connection
		const rewardAdjustment = globalReward - 1.0;

		// Apply reward to active_connections with exponential temporal decay
		// Older connections (higher age) get less reward/punishment
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN active_connections ac ON c.id = ac.connection_id
			SET c.strength = c.strength + (:rewardAdjustment * POW(:rewardTimeDecayFactor, ac.age))
			WHERE ac.age >= 0
		`, { rewardAdjustment, rewardTimeDecayFactor: this.rewardTimeDecayFactor });

		console.log(`Applied global reward ${globalReward.toFixed(3)} to ${result.affectedRows} active connections (adjustment: ${rewardAdjustment >= 0 ? '+' : ''}${rewardAdjustment.toFixed(3)})`);
	}
}