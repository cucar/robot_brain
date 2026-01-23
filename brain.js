import readline from 'node:readline';
import getMySQLConnection from './db/db.js';

/**
 * Brain Class
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		//************************************************************
		// hyperparameters
		//************************************************************

		// structural limits for the brain
		this.contextLength = 5; // number of frames a base neuron stays active - this determines the context length
		this.minConnectionStrength = 0; // minimum strength value for connections and patterns (clamped to prevent negative values)
		this.maxConnectionStrength = 100; // maximum strength value for connections and patterns (clamped to prevent overflow)

		// reward parameters
		this.rewardExpSmooth = 0.9; // exponential smoothing for rewards: new = smooth * observed + (1 - smooth) * old

		// pattern learning parameters
		this.predictionErrorMinStrength = 10.0; // minimum prediction strength to create error-driven pattern from connection inference
		this.patternErrorMinStrength = 10.0; // minimum prediction strength to create higher-level pattern from pattern inference error
		this.actionRegretMinStrength = 1; // minimum prediction strength to create action regret pattern (0 = always capture painful actions)
		this.actionRegretMinPain = 0; // minimum pain (negative reward magnitude) to create action regret pattern (0 = any negative reward triggers regret)
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.maxLevels = 10; // just to prevent against infinite recursion

		// voting parameters - simple strength-based voting, no weighting
		this.boltzmannTemperature = 0.1; // temperature for Boltzmann selection (lower = more aggressive, 1.0 = standard)

		// exploration parameters - probability inversely proportional to inference strength
		this.minExploration = 0.03; // minimum - never stop exploring
		this.maxExploration = 1.0; // 100% when totalStrength = 0
		this.explorationScale = 1000; // controls decay rate of exploration probability
		this.minExploredVotes = 2; // minimum votes to consider an action "explored" (excluding actions)

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.connectionForgetRate = 0.1; // how much connection strengths decay per forget cycle (reduced to preserve learned connections)
		this.patternForgetRate = 0.1; // how much pattern strengths decay per forget cycle
		this.rewardForgetRate = 0.05; // how much reward values decay toward 0 per forget cycle (0.05 = 5% decay toward neutral)

		// sliding window for max levels per age (for efficient voting)
		// maxLevelsByAge[0] = max level at age 0 (current frame)
		// maxLevelsByAge[1] = max level at age 1 (previous frame), etc.
		this.maxLevelsByAge = [];

		// current frame's highest active level (set by recognizePatternNeurons)
		this.currentMaxLevel = 0;

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// Prediction accuracy tracking (cumulative stats for base level only)
		this.accuracyStats = { correct: 0, total: 0 };

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

		// Debugging info and flags
		this.frameNumber = 0;
		this.frameSummary = true; // show frame summary or not
		this.debug = true;
		this.debug2 = true; // deeper, more verbose debug level
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info

		// Create readline interface for pausing between frames - used when debugging
		this.waitForUserInput = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });

		// Cache for action neuron IDs (for debug output)
		this.actionNeuronCache = null;
	}

	/**
	 * waits for user input to continue - used for debugging
	 */
	waitForUser(message) {
		if (!this.waitForUserInput) return Promise.resolve();
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Register a channel with the brain
	 */
	registerChannel(name, channelClass) {
		const channel = new channelClass(name);
		this.channels.set(name, channel);
		if (this.debug2) console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async initDB() {
		this.conn = await getMySQLConnection();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all learned data)...');
		await this.truncateTables([
			'channels',
			'dimensions',
			'neurons',
			'base_neurons',
			'coordinates',
			'connections',
			'active_neurons',
			'active_connections',
			'pattern_peaks',
			'pattern_past',
			'pattern_future',
			'matched_patterns',
			'matched_pattern_context',
			'new_pattern_future',
			'new_patterns',
			'inferred_neurons'
		]);

		// Clear action neuron cache since all neurons were deleted
		this.actionNeuronCache = null;
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (memory tables)...');
		await this.truncateTables([
			'active_neurons',
			'active_connections',
			'matched_patterns',
			'matched_pattern_connections',
			'new_pattern_future',
			'new_patterns',
			'inferred_neurons'
		]);

		// Clear action neuron cache since neurons may have been recreated
		this.actionNeuronCache = null;
	}

	/**
	 * Reset accuracy stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
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
	 * initializes the brain and loads dimensions
	 */
	async init() {
		// console.log('Initializing brain...');

		// register channels in DB and load channel IDs
		await this.initializeChannels();

		// create dimensions for all registered channels
		await this.initializeDimensions();

		// load the dimensions
		await this.loadDimensions();

		// initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) await channel.initialize();
	}

	/**
	 * Initialize channels in DB and load channel IDs
	 */
	async initializeChannels() {
		this.channelNameToId = {};
		this.channelIdToName = {};

		// Insert channels into DB
		for (const [channelName] of this.channels)
			await this.conn.query('INSERT IGNORE INTO channels (name) VALUES (?)', [channelName]);

		// Load channel IDs
		const [rows] = await this.conn.query('SELECT id, name FROM channels');
		rows.forEach(row => {
			this.channelNameToId[row.name] = row.id;
			this.channelIdToName[row.id] = row.name;
		});
		if (this.debug2) console.log('Channels loaded:', this.channelNameToId);
	}

	/**
	 * Initialize dimensions for all registered channels
	 */
	async initializeDimensions() {
		if (this.debug2) console.log('Initializing dimensions for registered channels...');
		for (const [, channel] of this.channels) {
			for (const dimName of channel.getEventDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (name) VALUES (?)', [dimName]);
			for (const dimName of channel.getOutputDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (name) VALUES (?)', [dimName]);
		}
	}

	/**
	 * loads the dimensions to memory (just id and name, no channel/type)
	 */
	async loadDimensions() {
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};

		const [rows] = await this.conn.query('SELECT id, name FROM dimensions');
		rows.forEach(row => {
			this.dimensionNameToId[row.name] = row.id;
			this.dimensionIdToName[row.id] = row.name;
		});
		if (this.debug2) console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * returns the current frame combined from all registered channels and executes previously inferred actions
	 * Each frame point includes: coordinates, channel, type
	 */
	async getFrameAndExecuteActions() {
		const frame = [];

		// Increment frame counter to be able to track inactivity
		this.frameNumber++;
		if (this.debug) console.log('******************************************************************');
		if (this.debug) console.log(`OBSERVING FRAME ${this.frameNumber}`);

		// Get all frame outputs from previous frame's inference (one query for all channels)
		const frameOutputs = await this.getFrameOutputs();

		// Process each channel: get inputs from channel, get outputs from brain tables, execute
		for (const [channelName, channel] of this.channels) {
			const channelId = this.channelNameToId[channelName];

			// Get the frame event inputs from the channel (current state before any outputs are executed)
			const channelEvents = await channel.getFrameEvents();
			for (const event of channelEvents)
				frame.push({ coordinates: event, channel: channelName, channel_id: channelId, type: 'event' });

			// Get last inferred actions to be executed in this frame (from brain's inference tables)
			const channelActions = frameOutputs.get(channelName) || [];
			for (const action of channelActions)
				frame.push({ coordinates: action, channel: channelName, channel_id: channelId, type: 'action' });

			// Execute actions - this updates channel state
			await channel.executeOutputs(channelActions);
		}

		if (this.debug) console.log(`frame points: ${JSON.stringify(frame)}`);
		if (this.debug) console.log('******************************************************************');
		return frame;
	}

	/**
	 * Get frame outputs for all channels from inferred_neurons table (MySQL implementation)
	 * Reads winning action neurons (age=0, level=0, is_winner=1) grouped by channel
	 * @returns {Promise<Map>} - Map of channel names to array of output coordinates
	 */
	async getFrameOutputs() {

		// Get all winning action neurons from inferred_neurons table
		// Only return neurons that are action type (from neurons table)
		const [rows] = await this.conn.query(`
			SELECT inf.neuron_id, c.dimension_id, c.val, d.name as dimension_name, ch.name as channel
			FROM inferred_neurons inf
			JOIN base_neurons b ON b.neuron_id = inf.neuron_id
			JOIN channels ch ON ch.id = b.channel_id
			JOIN coordinates c ON c.neuron_id = inf.neuron_id
			JOIN dimensions d ON d.id = c.dimension_id
			WHERE inf.is_winner = 1 AND b.type = 'action'
			ORDER BY ch.name, inf.neuron_id
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
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and channel-specific rewards (Map of channel_name -> reward)
	 */
	async processFrame(frame) {
		const frameStart = performance.now();

		// Display diagnostic frame header if enabled
		if (this.diagnostic) this.displayFrameHeader(frame);

		// get channel rewards
		const channelRewards = await this.getChannelRewards();

		// age the active neurons in memory context - sliding the temporal window
		await this.ageNeurons();

		// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
		await this.recognizeNeurons(frame);

		// learn from connection action inferences
		await this.rewardConnections(channelRewards);

		// refine the learned pattern definitions from prediction errors and action regret
		await this.refinePatterns(channelRewards);

		// learn new patterns from failed predictions and action regret
		await this.learnNewPatterns(channelRewards);

		// do predictions and outputs - what's going to happen next? and what's our best response?
		await this.inferNeurons();

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// used to avoid curse of dimensionality and delete dead connections/neurons
		await this.runForgetCycle();

		// show frame processing summary
		this.printFrameSummary(performance.now() - frameStart);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');
	}

	/**
	 * Display diagnostic frame header with frame number and observations
	 */
	displayFrameHeader(frame) {
		if (!this.diagnostic) return;

		// Build observation string from frame
		const observations = [];
		for (const point of frame)
			for (const [dim, val] of Object.entries(point))
				observations.push(`${dim}=${val}`);

		console.log(`\nF${this.frameNumber} | Obs: ${observations.join(', ')}`);
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getChannelRewards() {
		if (this.debug2) console.log('Getting rewards feedback from all channels...');
		const channelRewards = new Map();
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const reward = await channel.getRewards();
			if (reward !== 0) { // Only process non-neutral feedback (additive: 0 = neutral)
				if (this.debug2) console.log(`${channelName}: reward ${reward.toFixed(3)}`);
				channelRewards.set(channelName, reward);
				feedbackCount++;
			}
		}

		if (this.debug2) {
			if (feedbackCount > 0) console.log(`Received rewards from ${feedbackCount} channels`);
			else console.log('No rewards from any channels');
		}
		if (this.debug2 && feedbackCount > 0)
			console.log(`Channel rewards:`, Array.from(channelRewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));

		// Display reward information if diagnostic mode enabled
		if (this.diagnostic && channelRewards.size > 0) this.displayRewards(channelRewards);

		return channelRewards;
	}

	/**
	 * Display reward information for diagnostic output
	 */
	displayRewards(channelRewards) {
		if (!this.diagnostic) return;
		const rewardParts = [];
		for (const [channelName, reward] of channelRewards)
			rewardParts.push(`${channelName}:${reward.toFixed(3)}x`);
		console.log(`  Rewards: ${rewardParts.join(', ')}`);
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= contextLength.
	 * Also ages inference source tables for temporal credit assignment.
	 */
	async ageNeurons() {
		if (this.debug2) console.log('Aging active neurons, connections, and inferred neurons...');

		// age all neurons and connections - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE active_connections SET age = age + 1 ORDER BY age DESC');

		// Skip deletions until we have enough frames
		if (this.frameNumber < this.contextLength + 1) return;

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.contextLength})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.contextLength})`);
	}

	/**
	 * Recognizes and activates neurons from frame
	 * Common implementation for both MySQL and Memory backends
	 */
	async recognizeNeurons(frame) {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// insert the new base neurons to the active neurons table
		await this.insertActiveNeurons(neuronIds);

		// reinforce connections between active neurons in the base level
		await this.reinforceConnections();

		// activate connections for the newly activated neurons at the base level
		await this.activateConnections();

		// Validate event predictions and track accuracy stats
		await this.validatePredictions();

		// discover and activate patterns using connections - start recursion from base level
		await this.recognizePatternNeurons();
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame	 *  points have structure: { coordinates, channel, channel_id, type }
	 */
	async getFrameNeurons(frame) {

		// try to get all the neurons that have coordinates matching each point
		// return format: [{ point, neuron_id }] where point is the full frame point
		const matches = await this.matchFrameNeurons(frame);
		if (this.debug2) console.log('pointNeuronMatches', matches.map(match => JSON.stringify(match)).join('\n'));

		// matching neuron ids to be returned for each point of the frame
		const neuronIds = matches.filter(p => p.neuron_id).map(p => p.neuron_id);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => !p.neuron_id);
		if (pointsNeedingNeurons.length > 0) {
			if (this.debug2) console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with dedupe.`);
			const createdNeuronIds = await this.createBaseNeurons(pointsNeedingNeurons.map(p => p.point));
			neuronIds.push(...createdNeuronIds);
		}

		// return matching neuron ids to given points
		if (neuronIds.length === 0) throw new Error(`Failed to create neuron for exploration action: ${JSON.stringify(frame)}`);
		return neuronIds;
	}

	/**
	 * matches base neurons from dimensional values for each point
	 * Frame points have structure: { coordinates, channel, channel_id, type }
	 * Returns: [{ point, neuron_id }]
	 */
	async matchFrameNeurons(frame) {
		if (frame.length === 0) return [];
		const neuronCoords = await this.getFrameCoordinates(frame);
		return this.findFrameMatches(frame, neuronCoords);
	}

	/**
	 * fetches all neuron coordinates that could potentially match any point in the frame	 *  points have structure: { coordinates, channel, channel_id, type }
	 */
	async getFrameCoordinates(frame) {
		const allPairs = [];

		for (const point of frame)
			for (const [dimName, val] of Object.entries(point.coordinates))
				allPairs.push([this.dimensionNameToId[dimName], val]);

		const [rows] = await this.conn.query(`
			SELECT neuron_id, dimension_id, val
			FROM coordinates
			WHERE (dimension_id, val) IN (?)
		`, [allPairs]);

		const neuronCoords = new Map();
		for (const row of rows) {
			if (!neuronCoords.has(row.neuron_id)) neuronCoords.set(row.neuron_id, new Map());
			neuronCoords.get(row.neuron_id).set(row.dimension_id, row.val);
		}

		return neuronCoords;
	}

	/**
	 * finds exact neuron matches for each point in the frame	 *  points have structure: { coordinates, channel, channel_id, type }
	 */
	findFrameMatches(frame, neuronCoords) {
		const results = [];

		for (const point of frame) {
			const matchedNeuronId = this.findPointMatch(point.coordinates, neuronCoords);
			results.push({ point, neuron_id: matchedNeuronId });
		}

		return results;
	}

	/**
	 * finds the neuron that exactly matches a single point's coordinates
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
	 * creates base neurons from a given set of frame points and returns their ids
	 * Frame points have structure: { coordinates, channel, channel_id, type }
	 * Frame points are assumed to be unique, meaning no two points sent here should have the same coordinates
	 */
	async createBaseNeurons(framePoints) {
		if (framePoints.length === 0) return [];
		const neuronIds = await this.insertNeurons(framePoints.length);
		await this.insertBaseNeurons(neuronIds, framePoints);
		await this.setNeuronCoordinates(neuronIds, framePoints);
		return neuronIds;
	}

	/**
	 * Creates new neurons at a given level and return their IDs.
	 * MySQL guarantees sequential auto-increment IDs.
	 * @param {number} level
	 * @param {number} count
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async insertNeurons(count, level = 0) {
		if (count === 0) return [];
		const rows = Array.from({ length: count }, () => [level]);
		const [insertResult] = await this.conn.query('INSERT INTO neurons (level) VALUES ?', [rows]);
		const firstNeuronId = insertResult.insertId;
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * Insert base neuron metadata into base_neurons table
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @param {Array} framePoints - Array of frame points with channel_id and type
	 */
	async insertBaseNeurons(neuronIds, framePoints) {
		const infoValues = framePoints.map((point, idx) => [neuronIds[idx], point.channel_id, point.type]);
		await this.conn.query('INSERT INTO base_neurons (neuron_id, channel_id, type) VALUES ?', [infoValues]);
	}

	/**
	 * Sets coordinates for neurons
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @param {Array} framePoints - Array of frame points with coordinates
	 */
	async setNeuronCoordinates(neuronIds, framePoints) {

		// Map neuron IDs to coordinates
		const neurons = framePoints.map((point, idx) => ({ neuron_id: neuronIds[idx], coordinates: point.coordinates }));

		// flatten to rows of [neuron_id, dimension_id, value]
		const rows = neurons.flatMap(({ neuron_id, coordinates }) =>
			Object.entries(coordinates).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));

		// Insert all coordinates in a single batch
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ? ON DUPLICATE KEY UPDATE val = VALUES(val)', [rows]);
	}

	/**
	 * inserts active neurons at age=0
	 */
	async insertActiveNeurons(neuronIds) {
		if (neuronIds.length === 0) return;
		const activations = neuronIds.map(neuronId => [neuronId]);
		await this.conn.query(`INSERT INTO active_neurons (neuron_id) VALUES ?`, [activations]);
	}

	/**
	 * Reinforce connections between base level active neurons - from age > 0 to age = 0
	 */
	async reinforceConnections() {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT f.neuron_id as from_neuron_id, t.neuron_id as to_neuron_id, f.age as distance, 1 as strength
            -- build connections from event neurons in base level
			FROM active_neurons f
			JOIN neurons nf ON nf.id = f.neuron_id 
            JOIN base_neurons bnf ON bnf.neuron_id = nf.id AND bnf.type = 'event'
			-- build connections to base neurons (event or action)
			CROSS JOIN active_neurons t
			JOIN neurons nt ON nt.id = t.neuron_id AND nt.level = 0
            WHERE f.age > 0 -- build connections from event neurons in base level that are already active
            AND t.age = 0 -- build connections from base neurons that are just activated
            -- if the connection already exists, increment its strength 
			ON DUPLICATE KEY UPDATE connections.strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, connections.strength + VALUES(strength)))
		`, { minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });
	}

	/**
	 * Populate active_connections table for newly activated neurons at the base level
	 */
	async activateConnections() {
		await this.conn.query(`
			INSERT IGNORE INTO active_connections (connection_id, age)
			SELECT c.id as connection_id, 0 as age
			FROM active_neurons f
			-- connections from older base neurons to newly activated neurons (distance = age)
			JOIN connections c ON c.from_neuron_id = f.neuron_id AND c.distance = f.age AND c.strength > 0
			-- connections to newly activated base neurons (age=0)    
			JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0
		`);
	}

	/**
	 * Validate event predictions by comparing inferred_neurons (age=1) to active_neurons (age=0).
	 * Only validates base level (level 0) event predictions.
	 * Only validates winners (is_winner=1) since losers are alternative hypotheses that were rejected.
	 */
	async validatePredictions() {

		// Get all base level event predictions from previous frame and check if they came true (age=0)
		// Only validate winners - losers are just alternative hypotheses rejected during conflict resolution
		const [rows] = await this.conn.query(`
			SELECT IF(an.neuron_id IS NOT NULL, 1, 0) as is_correct
			FROM inferred_neurons inf
			JOIN base_neurons b ON b.neuron_id = inf.neuron_id
			LEFT JOIN active_neurons an ON an.neuron_id = inf.neuron_id AND an.age = 0
			WHERE b.type = 'event'
			AND inf.is_winner = 1
		`);

		// Count correct and total predictions
		const total = rows.length;
		const correct = rows.reduce((sum, row) => sum + row.is_correct, 0);

		// Update accuracyStats (cumulative across frames)
		this.accuracyStats.correct += correct;
		this.accuracyStats.total += total;

		if (this.debug && total > 0) {
			const accuracy = (correct / total * 100).toFixed(1);
			console.log(`Validated predictions: ${correct}/${total} (${accuracy}%)`);
		}
	}

	/**
	 * detects all spatial levels in age=0 neurons using unified connections - start from base level and go as high as possible
	 * Saves the highest activated level to this.currentMaxLevel for use by collectVotes
	 */
	async recognizePatternNeurons() {
		this.currentMaxLevel = 0;
		while (true) {

			// process the level to detect patterns - returns if there are patterns found or not
			const patternsFound = await this.recognizeLevelPatterns(this.currentMaxLevel);

			// if no patterns are found in the level, nothing to do
			if (!patternsFound) break;

			// increment the level to start processing it
			this.currentMaxLevel++;

			// if we exceeded the max level, give warning and stop
			if (this.currentMaxLevel >= this.maxLevels) {
				console.error('Max level exceeded.');
				break;
			}
		}
	}

	/**
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 */
	async recognizeLevelPatterns(level) {
		if (this.debug2) console.log(`Processing level ${level} for pattern recognition`);

		// Match active connections to known patterns and write to matched_patterns table
		const matchCount = await this.matchObservedPatterns(level);
		if (matchCount === 0) {
			if (this.debug2) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// strengthen the pattern peaks that are just activated
		await this.strengthenActivePatternPeaks();

		// update pattern_past for matched patterns - add/strengthen observed connections, weaken unobserved connections
		await this.refinePatternPast();

		// Activate all matched pattern neurons from matched_patterns table
		const [patternNeurons] = await this.conn.query('SELECT pattern_neuron_id FROM matched_patterns');
		const patternNeuronIds = patternNeurons.map(row => row.pattern_neuron_id);
		await this.insertActiveNeurons(patternNeuronIds);

		return true; // patterns found
	}

	/**
	 * Match active neurons to known pattern contexts and saves them in matched_patterns table.
	 * @param {number} level - The level to match patterns for (peak neuron level)
	 * @returns {Promise<number>} - Number of matched patterns
	 */
	async matchObservedPatterns(level) {
		if (this.debug2) console.log('Matching active connections to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_patterns');
		await this.conn.query('TRUNCATE matched_pattern_connections');

		// Determine which patterns matched based on overlap threshold
		const [result] = await this.conn.query(`
			INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
			SELECT pp.peak_neuron_id, pp.pattern_neuron_id
            FROM active_neurons an_peak
            JOIN neurons n_peak ON n_peak.id = an_peak.neuron_id AND n_peak.level = ?    
			JOIN pattern_peaks pp ON an_peak.neuron_id = pp.peak_neuron_id
            -- pattern_past only contains same-level connections (pattern identity is determined by same-level context)
            JOIN pattern_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
            LEFT JOIN active_neurons an_ctx ON an_ctx.neuron_id = p.context_neuron_id AND an_ctx.age = p.context_age
			WHERE an_peak.age = 0
			GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
			-- pattern matches if at least some percentage of its pattern_past connections are in active_connections
			HAVING SUM(IF(an_ctx.neuron_id IS NOT NULL, 1, 0)) >= COUNT(*) * ?
		`, [level, this.mergePatternThreshold]);

		// show matched pattern details for debugging
		if (this.debug) {
			const [matchedPairs] = await this.conn.query('SELECT * FROM matched_patterns');
			console.log(`Matched ${result.affectedRows} pattern-peak pairs:`,
				matchedPairs.map(p => `peak=${p.peak_neuron_id}, pattern=${p.pattern_neuron_id}`).join('; '));
		}

		// if no patterns matched, nothing to do
		if (result.affectedRows === 0) return 0;

		// populate context analysis for refinement of matched patterns
		await this.populateMatchedPatternContext(level);

		// return the number of matched patterns
		return result.affectedRows;
	}

	/**
	 * Populates the matched_pattern_context table with the context of the matched patterns
	 */
	async populateMatchedPatternContext() {

		// Common: context neurons that ARE active at correct age
		await this.conn.query(`
			INSERT INTO matched_pattern_context (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT p.pattern_neuron_id, p.context_neuron_id, p.context_age, 'common'
			FROM matched_patterns mp
			JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id
			JOIN active_neurons an ON an.neuron_id = p.context_neuron_id AND an.age = p.context_age
		`);

		// Missing: context neurons NOT active at correct age
		await this.conn.query(`
			INSERT INTO matched_pattern_context (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT p.pattern_neuron_id, p.context_neuron_id, p.context_age, 'missing'
			FROM matched_patterns mp
			JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id
			LEFT JOIN active_neurons an ON an.neuron_id = p.context_neuron_id AND an.age = p.context_age
			WHERE an.neuron_id IS NULL
		`);

		// Novel: active neurons at same level, not in pattern_past (all channels)
		await this.conn.query(`
			INSERT INTO matched_pattern_context (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT mp.pattern_neuron_id, an_context.neuron_id, an_context.age, 'novel'
			FROM matched_patterns mp
			JOIN neurons n_peak ON n_peak.id = mp.peak_neuron_id
			JOIN active_neurons an_context ON an_context.age > 0
            JOIN neurons n_context ON n_context.id = an_context.neuron_id AND n_context.level = n_peak.level
			LEFT JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id 
			    AND p.context_neuron_id = an_context.neuron_id AND p.context_age = an_context.age
			WHERE p.pattern_neuron_id IS NULL
		`);
	}

	/**
	 * update pattern_peaks that are just activated
	 */
	async strengthenActivePatternPeaks() {

		// Reinforce pattern_peaks strength for matched patterns
		await this.conn.query(`
			UPDATE pattern_peaks pp
			JOIN matched_patterns mp ON pp.pattern_neuron_id = mp.pattern_neuron_id
			SET pp.strength = LEAST(?, pp.strength + 1)
		`, [this.maxConnectionStrength]);
	}

	/**
	 * Refine matched patterns using pre-analyzed connection sets.
	 * Uses matched_pattern_connections table populated by matchObservedPatterns:
	 * 1. Add novel connections (status='novel')
	 * 2. Strengthen common connections (status='common')
	 * 3. Weaken missing connections (status='missing')
	 */
	async refinePatternPast() {
		if (this.debug) console.log('refining pattern_past...');

		// Add novel context neurons
		const [novelResult] = await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age, strength)
			SELECT pattern_neuron_id, context_neuron_id, context_age, 1.0
			FROM matched_pattern_context
			WHERE status = 'novel'
		`);

		// Strengthen common context
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_context mpc 
				ON p.pattern_neuron_id = mpc.pattern_neuron_id
				AND p.context_neuron_id = mpc.context_neuron_id
				AND p.context_age = mpc.context_age
			SET p.strength = LEAST(?, p.strength + 1)
			WHERE mpc.status = 'common'
		`, [this.maxConnectionStrength]);

		// Weaken missing context
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_context mpc
				ON p.pattern_neuron_id = mpc.pattern_neuron_id
				AND p.context_neuron_id = mpc.context_neuron_id
				AND p.context_age = mpc.context_age
			SET p.strength = GREATEST(?, p.strength - ?)
			WHERE mpc.status = 'missing'
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);

		if (this.debug) console.log(`Pattern context: +${novelResult.affectedRows} novel, ` +
			`↑${strengthenResult.affectedRows} strengthened, ↓${weakenResult.affectedRows} weakened`);
	}

	/**
	 * Apply rewards to action connections
	 */
	async rewardConnections(channelRewards) {

		// nothing to update if there are no rewards
		if (channelRewards.size === 0) return;

		// apply rewards reinforcement to executed actions via connection inference
		// winners were executed and added to frame, then activated when recognizing neurons
		// they appear in active_connections at age=0 (just activated)
		// exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
		// leave losers alone - we don't know what would have happened if they were executed
		const channelIds = Array.from(channelRewards.keys()).map(name => this.channelNameToId[name]);
		const rewardCase = this.buildChannelRewardCase(channelRewards);
		const [result] = await this.conn.query(`
            UPDATE connections c
            JOIN active_connections ac ON ac.connection_id = c.id AND ac.age = 0
            JOIN base_neurons b ON b.neuron_id = c.to_neuron_id
            SET c.reward = :smooth * (${rewardCase}) + (1 - :smooth) * c.reward
            WHERE b.type = 'action'
            AND b.channel_id IN (${channelIds.join(',')})
		`, { smooth: this.rewardExpSmooth });
		if (this.debug) console.log(`Rewarded ${result.affectedRows} connections`);
	}

	/**
	 * Build a CASE-WHEN SQL snippet for channel rewards.
	 * Maps channel_id to reward value for use in UPDATE statements.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 * @returns {string} SQL CASE statement like "CASE n.channel_id WHEN 1 THEN 0.5 WHEN 2 THEN -0.3 END"
	 */
	buildChannelRewardCase(channelRewards) {
		if (channelRewards.size === 0) return '';
		const caseWhen = Array.from(channelRewards.entries())
			.map(([name, reward]) => `WHEN ${this.channelNameToId[name]} THEN ${reward}`)
			.join(' ');
		return `CASE b.channel_id ${caseWhen} END`;
	}

	/**
	 * Get channel IDs with painful rewards (below actionRegretMinPain threshold).
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 * @returns {Array<number>} Array of channel IDs with negative rewards
	 */
	getPainfulChannelIds(channelRewards) {
		const painfulChannelIds = [];
		for (const [channelName, reward] of channelRewards)
			if (reward < this.actionRegretMinPain) painfulChannelIds.push(this.channelNameToId[channelName]);
		return painfulChannelIds;
	}

	/**
	 * Learns patterns from prediction errors and action regret and continues to refine them as they are observed
	 */
	async refinePatterns(channelRewards) {

		// update pattern_future for event patterns based on observations
		await this.refineEventPatternsFuture();

		// apply rewards to action predictions based on channel outcomes
		await this.rewardExecutedActionPatterns(channelRewards);

		// add alternative actions to action patterns in painful channels
		await this.addAlternativeActionsToPatterns(channelRewards);
	}

	/**
	 * Refine pattern_future for event patterns with observed connections.
	 * Called during learning phase after pattern inference.
	 * Refinement happens when age = distance (prediction outcome just observed).
	 * pattern_future stores inferred_neuron_id (base neuron that pattern predicts).
	 * This method only applies to EVENT patterns.
	 * Action pattern refinement happens in applyRewards (both reward-based and trial alternatives).
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future inferences that were correctly predicted (target now active)
	 * 2. Negative: Weaken pattern_future inferences that were incorrectly predicted (target NOT active)
	 * 3. Novel: Add new inferences from pattern to newly observed neurons
	 */
	async refineEventPatternsFuture() {

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted event neurons
		// Neuron is now active at age=0, refinement when age = distance
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
            JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN active_neurons an_target ON an_target.neuron_id = pf.inferred_neuron_id AND an_target.age = 0
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
			SET pf.strength = LEAST(?, pf.strength + 1)
			WHERE b.type = 'event'
		`, [this.maxConnectionStrength]);
		if (this.debug) console.log(`Strengthened ${strengthenResult.affectedRows} correct event pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted event neurons
		// Neuron is NOT active, refinement when age = distance
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
            JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
			SET pf.strength = GREATEST(?, pf.strength - ?)
			WHERE b.type = 'event'
			AND pf.inferred_neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug) console.log(`Weakened ${weakenResult.affectedRows} failed event pattern_future predictions`);

		// 3. ADD NOVEL NEURONS: Active event neurons not yet in pattern_future
		// Find event patterns that made predictions, find active base neurons in same channel
		// Pattern future must only contain same-channel predictions
		// Novel neurons are added at the same distance as the prediction that was made
		const [novelResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance)
			SELECT pf.pattern_neuron_id, an.neuron_id, pf.distance
			FROM pattern_future pf
			JOIN active_neurons ap ON ap.neuron_id = pf.pattern_neuron_id AND pf.distance = ap.age
			JOIN active_neurons an ON an.age = 0
            JOIN base_neurons b ON b.neuron_id = an.neuron_id AND b.type = 'event'
		`);
		if (this.debug) console.log(`Added ${novelResult.affectedRows} novel neurons to event pattern_future`);
	}

	/**
	 * Strengthen and reward executed action pattern_future predictions.
	 *
	 * Channel-Specific Credit Assignment:
	 * 1. Identify which channel each base-level output belongs to (via output dimensions)
	 * 2. Use pattern_future (inferred_neuron_id) to find which patterns led to each action
	 * 3. Apply channel-specific reward via CASE-WHEN on channel_id
	 *
	 * Rewards are applied via exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
	 * This converges to the expected reward for each connection.
	 * Neutral reward is 0, positive is good, negative is bad
	 *
	 * Also strengthens executed action patterns so they won't be forgotten.
	 */
	async rewardExecutedActionPatterns(channelRewards) {

		// nothing to do if there are no rewards
		if (channelRewards.size === 0) return;

		// Strengthen and reward action pattern_future in one query
		// Join chain: pattern_future → active neuron (just executed) → active pattern
		// Strength is always incremented; reward is updated only if channel has a reward
		const rewardCase = this.buildChannelRewardCase(channelRewards);
		const [result] = await this.conn.query(`
			UPDATE pattern_future pf
            JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN active_neurons an_target ON an_target.neuron_id = pf.inferred_neuron_id AND an_target.age = 0
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
			SET pf.strength = LEAST(?, pf.strength + 1), pf.reward = CASE ${rewardCase} ELSE pf.reward END
			WHERE b.type = 'action'
		`, [this.maxConnectionStrength]);

		if (this.debug) console.log(`Strengthened/rewarded ${result.affectedRows} action pattern_future predictions`);
	}

	/**
	 * Add alternative actions to action patterns in painful channels.
	 * When an action pattern prediction fails (painful reward), add one untried alternative.
	 * Key insight: Same pattern can be active at multiple ages (distances), each inferring different actions.
	 * We need to find alternatives per (pattern, distance, channel) tuple, not just per pattern.
	 */
	async addAlternativeActionsToPatterns(channelRewards) {

		// get the channels that executed painful actions - we will add alternative actions for them
		// if there are no painful channels, we don't need to add alternatives
		const painfulChannelIds = this.getPainfulChannelIds(channelRewards);
		if (painfulChannelIds.length === 0) return;

		// Find one untried action for each (pattern, distance, channel) that was executed in painful channels
		// Uses MIN(neuron_id) to pick one untried action per (pattern, distance, channel) tuple
		// NOT EXISTS filters out already-tried actions; if all actions tried, no rows remain for that group
		// INSERT IGNORE handles case where this alternative was already added in a previous frame
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, distance, inferred_neuron_id)
			SELECT pf.pattern_neuron_id, pf.distance, MIN(b_alt.neuron_id)
			FROM pattern_future pf
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
            JOIN active_neurons an_action ON an_action.neuron_id = pf.inferred_neuron_id AND an_action.age = 0
            JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN base_neurons b_alt ON b_alt.channel_id = b.channel_id AND b_alt.type = 'action'
			WHERE b.type = 'action'
			AND b.channel_id IN (${painfulChannelIds.join(',')})
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf_existing
				WHERE pf_existing.pattern_neuron_id = pf.pattern_neuron_id
				AND pf_existing.distance = pf.distance
				AND pf_existing.inferred_neuron_id = b_alt.neuron_id
			)
			GROUP BY pf.pattern_neuron_id, pf.distance, b.channel_id
		`);

		if (this.debug) console.log(`Added ${result.affectedRows} alternative actions to patterns`);
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Processes all levels in bulk, like inferNeurons.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 */
	async learnNewPatterns(channelRewards) {

		// do not start learning new patterns until we had enough frames in the context
		if (this.frameNumber < this.contextLength + 1) return;

		// Find the neurons that should be in pattern_future of new patterns
		// (prediction errors and action regret, unified in one method)
		const newPatternFutureCount = await this.populateNewPatternFuture(channelRewards);
		if (this.debug) console.log(`New pattern future count: ${newPatternFutureCount}`);
		if (newPatternFutureCount === 0) return;
		if (newPatternFutureCount > 0) this.waitForUserInput = true;

		// Populate new_patterns table with peaks from new pattern future inferences
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons();

		// Create new patterns in pattern_peaks, pattern_past, pattern_future
		await this.createNewPatterns();

		// Activate newly created pattern neurons so they can be refined in the same frame
		// and matched in future frames. Age = distance (when the peak was active)
		await this.activateNewPatterns();

		if (this.debug) console.log(`Created ${patternCount} error patterns`);
	}

	/**
	 * Populate new_pattern_future with neurons that should be in pattern_future of new patterns.
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
	 * 2. Action regret: inferred action winner got negative reward → pattern should infer another action
	 *
	 * Returns the number of new pattern future inferences found.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 */
	async populateNewPatternFuture(channelRewards) {
		await this.conn.query(`TRUNCATE new_pattern_future`);
		const eventCount = await this.populateNewEventPatterns();
		const actionCount = await this.populateNewActionPatterns(channelRewards);
		return eventCount + actionCount;
	}

	/**
	 * Populate event prediction error neurons into new_pattern_future.
	 * Peak predicted X, but X didn't happen and Y happened instead.
	 * → Create pattern to predict Y from this context.
	 *
	 * Handles both connection and pattern inference errors.
	 * Works for all levels: if a level N neuron predicts a base event,
	 * and it doesn't happen, create a level N+1 pattern to correct the prediction.
	 * @returns {Promise<number>} Number of event pattern futures created
	 */
	async populateNewEventPatterns() {
		const baseCount = await this.populateNewBaseEventPatterns();
		const highCount = await this.populateNewHighEventPatterns();
		return baseCount + highCount;
	}

	/**
	 * Populate base-level event prediction errors (from connection inferences).
	 * Peak is the base neuron that made the prediction.
	 * Find what actually happened (active neurons) instead of what was predicted.
	 * @returns {Promise<number>} Number of base event pattern futures created
	 */
	async populateNewBaseEventPatterns() {
		const [connResult] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT c.from_neuron_id, an.neuron_id, 1
			FROM inferred_neurons inf
            -- creating event patterns, so the inferred neuron is an event
			JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'event'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation 
            -- only peak neurons at age=1 learn the patterns, to ensure full context
            -- at this point (frameNumber >= contextLength + 1), age=1 peaks always have full context
			-- they will have additional distances as we observe them as added in when refining patterns
			JOIN connections c ON c.to_neuron_id = inf.neuron_id AND c.distance = 1 AND c.strength >= ?
			JOIN active_neurons an_peak ON an_peak.neuron_id = c.from_neuron_id AND an_peak.age = 1
            -- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
			-- note that this may be a stock specific thing - actions can learn patterns in robotics context    
            JOIN base_neurons b_peak ON b_peak.neuron_id = c.from_neuron_id AND b_peak.type = 'event'
			-- find the active neurons in same channel (what actually happened just now)
            -- we are creating patterns for events that happened in ANY channel to be able to generalize 
			JOIN active_neurons an ON an.age = 0
			JOIN base_neurons b_actual ON b_actual.neuron_id = an.neuron_id AND b_actual.type = 'event' 
			-- the inference did not come true (prediction error)
            WHERE inf.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
			-- don't create a pattern if there is already an active pattern for this peak 
			-- it means the peak recognized this situation - we'll refine it instead in that case
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an_pattern
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
				WHERE pp.peak_neuron_id = c.from_neuron_id
				AND an_pattern.age = 1
			)
		`, [this.predictionErrorMinStrength]);
		if (this.debug) console.log(`Found ${connResult.affectedRows} connection prediction error neurons`);
		return connResult.affectedRows;
	}

	/**
	 * Populate higher-level event prediction errors (from pattern inferences).
	 * Peak is the pattern neuron that made the prediction.
	 * Find what actually happened (active neurons) instead of what was predicted.
	 * Uses same logic as populateNewBaseEventPatterns but for pattern predictions.
	 * @returns {Promise<number>} Number of high event pattern futures created
	 */
	async populateNewHighEventPatterns() {
		const [patternResult] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT pf.pattern_neuron_id, an.neuron_id, 1
			FROM inferred_neurons inf
			-- creating event patterns, so the inferred neuron is an event
			JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'event'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			-- only create patterns from age=1 peaks to ensure full context
			-- at this point (frameNumber >= contextLength + 1), age=1 peaks always have full context
			JOIN pattern_future pf ON pf.inferred_neuron_id = inf.neuron_id AND pf.distance = 1 AND pf.strength >= ?
			JOIN active_neurons an_peak ON an_peak.neuron_id = pf.pattern_neuron_id AND an_peak.age = 1
			-- find the active neurons in same channel (what actually happened just now)
            -- we are creating patterns for events that happened in ANY channel to be able to generalize 
			JOIN active_neurons an ON an.age = 0
            JOIN base_neurons b_actual ON b_actual.neuron_id = an.neuron_id AND b_actual.type = 'event'
            -- the inference did not come true (prediction error)
			WHERE inf.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
			-- don't create a pattern if there is already an active pattern for this peak
			-- it means the peak recognized this situation - we'll refine it instead in that case
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an_pattern
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
				WHERE pp.peak_neuron_id = pf.pattern_neuron_id
				AND an_pattern.age = 1
			)
		`, [this.predictionErrorMinStrength]);
		if (this.debug) console.log(`Found ${patternResult.affectedRows} pattern prediction error neurons`);
		return patternResult.affectedRows;
	}

	/**
	 * Populate action regret neurons into new_pattern_future.
	 * When a connection or pattern predicts an action at distance=1 (full context),
	 * and it leads to pain, add one untried alternative action.
	 * Same logic as addAlternativeActionsToPatterns - just find something different to try.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 * @returns {Promise<number>} Number of action pattern futures created
	 */
	async populateNewActionPatterns(channelRewards) {

		// get the channels that executed painful actions - we will add alternative actions for them
		// if there are no painful channels, we don't need to create any action regret patterns
		const painfulChannelIds = this.getPainfulChannelIds(channelRewards);
		if (painfulChannelIds.length === 0) return 0;

		// process connection action inference regret: find connections that made painful predictions
		// and need to create a new pattern with an alternative action
		const baseCount = await this.populateNewBaseActionPatterns(painfulChannelIds);

		// Process pattern inference regret: find patterns that made painful predictions
		// and need to create a higher-level pattern with an alternative action
		const highCount = await this.populateNewHighActionPatterns(painfulChannelIds);

		return baseCount + highCount;
	}

	/**
	 * Handle action regret for connection inferences.
	 * Find connections that made painful action predictions and add one untried alternative action.
	 * Peak is the base neuron that made the wrong prediction.
	 * Uses same structure as populateNewBaseEventPatterns but for actions:
	 * - Events: write what actually happened (active neurons) instead of what was predicted
	 * - Actions: write an alternative action instead of what was executed (winner in painful channel)
	 * @param {Array} painfulChannelIds - Channel IDs with negative rewards
	 * @returns {Promise<number>} Number of base action pattern futures created
	 */
	async populateNewBaseActionPatterns(painfulChannelIds) {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT c.from_neuron_id, MIN(b_alt.neuron_id), 1
			FROM inferred_neurons inf
			-- creating action patterns, so the inferred neuron is an action that was executed (winner)
			JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'action'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			-- only peak neurons at age=1 learn the patterns, to ensure full context
			JOIN connections c ON c.to_neuron_id = inf.neuron_id AND c.distance = 1 AND c.strength >= ?
			JOIN active_neurons an_peak ON an_peak.neuron_id = c.from_neuron_id AND an_peak.age = 1
			-- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
            -- note that this may be a stock specific thing - actions can learn patterns in robotics context    
			JOIN base_neurons b_peak ON b_peak.neuron_id = c.from_neuron_id AND b_peak.type = 'event'
            -- find alternative actions in the same channel (what we should try instead)
			JOIN base_neurons b_alt ON b_alt.channel_id = b_inf.channel_id AND b_alt.type = 'action'
			-- the action was executed (winner) and resulted in pain
			WHERE inf.is_winner = 1
			AND b_inf.channel_id IN (${painfulChannelIds.join(',')})
			AND b_alt.neuron_id != inf.neuron_id  -- don't suggest the same action that just failed
		    -- don't create a pattern if there is already an active pattern for this peak 
		    -- it means the peak recognized this situation - we'll refine it instead in that case
			AND NOT EXISTS (
                SELECT 1 FROM active_neurons an_pattern
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
                WHERE pp.peak_neuron_id = c.from_neuron_id
                AND an_pattern.age = 1
			)
			GROUP BY c.from_neuron_id, b_inf.channel_id
		`, [this.actionRegretMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} connection action regret inferences`);
		return result.affectedRows;
	}

	/**
	 * Handle action regret for pattern inferences.
	 * Find patterns that made painful action predictions and add one untried alternative action.
	 * Peak is the pattern neuron that made the wrong prediction.
	 * Uses same structure as populateNewHighEventPatterns but for actions:
	 * - Events: write what actually happened (active neurons) instead of what was predicted
	 * - Actions: write an alternative action instead of what was executed (winner in painful channel)
	 * @param {Array} painfulChannelIds - Channel IDs with negative rewards
	 * @returns {Promise<number>} Number of high action pattern futures created
	 */
	async populateNewHighActionPatterns(painfulChannelIds) {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT pf.pattern_neuron_id, MIN(b_alt.neuron_id), 1
			FROM inferred_neurons inf
			-- creating action patterns, so the inferred neuron is an action that was executed (winner)
			JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'action'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			-- only create patterns from age=1 peaks to ensure full context
			JOIN pattern_future pf ON pf.inferred_neuron_id = inf.neuron_id AND pf.distance = 1 AND pf.strength >= ?
			JOIN active_neurons an_peak ON an_peak.neuron_id = pf.pattern_neuron_id AND an_peak.age = 1
			-- find alternative actions in the same channel (what we should try instead)
			JOIN base_neurons b_alt ON b_alt.channel_id = b_inf.channel_id AND b_alt.type = 'action'
			-- the action was executed (winner) and resulted in pain
			WHERE inf.is_winner = 1
			AND b_inf.channel_id IN (${painfulChannelIds.join(',')})
			AND b_alt.neuron_id != inf.neuron_id  -- don't suggest the same action that just failed
			-- don't create a pattern if there is already an active pattern for this peak
			-- it means the peak recognized this situation - we'll refine it instead in that case
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an_pattern
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
				WHERE pp.peak_neuron_id = pf.pattern_neuron_id
				AND an_pattern.age = 1
			)
			GROUP BY pf.pattern_neuron_id, b_inf.channel_id
		`, [this.patternErrorMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} pattern action regret inferences`);
		return result.affectedRows;
	}

	/**
	 * Populate new_patterns table from new_pattern_future.
	 * Finds unique (peak, type) combinations.
	 * Peak can be a base neuron (for connection errors) or pattern neuron (for pattern errors).
	 * All patterns are created at distance=1 (full context).
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		await this.conn.query(`TRUNCATE new_patterns`);
		// Create separate patterns for each (peak_neuron_id, type) combination
		// All patterns created at distance=1 for full context
		const [insertResult] = await this.conn.query(`
			INSERT INTO new_patterns (peak_neuron_id)
			SELECT DISTINCT peak_neuron_id
			FROM new_pattern_future
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Explore a channel by finding an unexplored action.
	 * Checks if all actions have been explored at all distances - only returns null if fully explored.
	 * @param {string} channelName - Channel name
	 * @param {Array} votedActions - All actions that received votes (have connections from current context)
	 * @returns {Promise<Object|null>} Exploration action or null if all actions explored at all distances
	 */
	async exploreChannel(channelName, votedActions) {
		const channel = this.channels.get(channelName);

		// Collect all actions that have been sufficiently explored (with event votes)
		const exploredActions = votedActions.filter(a => {
			if (!a.sources) return false;
			return a.sources.length >= this.minExploredVotes;
		});

		// Extract coordinates from explored actions
		const votedCoordinates = exploredActions.map(a => a.coordinates).filter(c => c);

		// Ask channel for an action that wasn't voted for
		const actionCoordinates = channel.getExplorationAction(votedCoordinates);
		if (!actionCoordinates || Object.keys(actionCoordinates).length === 0) {
			// All actions have been tried - now check if they've been tried at all distances
			// Group actions by distance
			const byDistance = new Map();
			for (const inf of exploredActions) {
				if (!inf.sources) continue;

				const distances = [...new Set(inf.sources.map(s => s.distance))];
				for (const distance of distances) {
					if (!byDistance.has(distance)) byDistance.set(distance, new Set());
					byDistance.get(distance).add(JSON.stringify(inf.coordinates));
				}
			}

			// Check if any distance is missing some actions
			const allActions = new Set(votedCoordinates.map(c => JSON.stringify(c)));
			for (const [distance, exploredAtDistance] of byDistance) {
				if (exploredAtDistance.size < allActions.size) {
					// This distance hasn't explored all actions - find one to explore
					for (const actionStr of allActions) {
						if (!exploredAtDistance.has(actionStr)) {
							const actionCoords = JSON.parse(actionStr);
							if (!actionCoords || Object.keys(actionCoords).length === 0) continue;

							const [actionNeuronId] = await this.getFrameNeurons([{
								coordinates: actionCoords,
								channel: channelName,
								channel_id: this.channelNameToId[channelName],
								type: 'action'
							}]);

							const exploration = {
								neuron_id: actionNeuronId,
								strength: 1,
								reward: 0,
								coordinates: actionCoords,
								type: 'action',
								isWinner: true
							};

							if (this.debug) console.log(`Exploration for ${channelName}: re-exploring action at distance ${distance}:`, actionCoords);
							return exploration;
						}
					}
				}
			}

			if (this.debug) console.log(`Exploration for ${channelName}: all actions explored at all distances`);
			return null; // All actions explored at all distances
		}

		// Find or create neuron for this action - wrap coordinates in frame point structure with channel metadata
		const [actionNeuronId] = await this.getFrameNeurons([{
			coordinates: actionCoordinates,
			channel: channelName,
			channel_id: this.channelNameToId[channelName],
			type: 'action'
		}]);

		// Return exploration action
		// Use low strength (below minErrorPatternThreshold) to avoid triggering error pattern creation
		// when exploration fails - exploration is random, not a confident prediction that was wrong
		const exploration = {
			neuron_id: actionNeuronId,
			strength: 1, // Low strength - exploration wins by marking others as losers, not by high strength
			reward: 0, // Neutral reward (additive: 0 = neutral)
			coordinates: actionCoordinates,
			type: 'action',
			isWinner: true
		};

		if (this.debug) console.log(`Exploration for ${channelName}:`, actionCoordinates, actionNeuronId);
		return exploration;
	}

	/**
	 * Apply exploration to action inferences.
	 * Modifies inferences array in place - may add exploration action and update isWinner flags.
	 * @param {Array} inferences - Array of inference objects (modified in place)
	 */
	async applyExploration(inferences) {

		// Group action inferences by channel (using channel from neuron, not dimension)
		const byChannel = new Map();
		for (const inf of inferences) {
			if (!inf.coordinates || inf.type !== 'action') continue;
			if (!byChannel.has(inf.channel)) byChannel.set(inf.channel, []);
			byChannel.get(inf.channel).push(inf);
		}

		// Process each channel
		for (const [channelName] of this.channels) {
			const channelInferences = byChannel.get(channelName) || [];
			const channelWinners = channelInferences.filter(inf => inf.isWinner);

			// Check if we should explore this channel
			if (!this.shouldExploreChannel(channelName, channelWinners)) continue;

			// Get exploration action - pass all channel inferences so channel knows what's been tried
			const exploration = await this.exploreChannel(channelName, channelInferences);
			if (!exploration) continue; // All actions have been tried

			// Mark previous winners as losers
			for (const winner of channelWinners)
				if (winner.neuron_id !== exploration.neuron_id)
					winner.isWinner = false;

			// Add exploration as winner
			inferences.push(exploration);
		}
	}

	/**
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {

		// Get base level (level 0) accuracy
		let baseAccuracy = 'N/A';
		if (this.accuracyStats.total > 0)
			baseAccuracy = `${(this.accuracyStats.correct / this.accuracyStats.total * 100).toFixed(1)}%`;

		// Collect continuous prediction metrics from channels (only new errors since last call)
		for (const [_, channel] of this.channels) {
			if (typeof channel.getPredictionMetrics === 'function') {
				const metrics = channel.getPredictionMetrics();
				if (metrics) {
					this.continuousPredictionMetrics.totalError += metrics.totalError;
					this.continuousPredictionMetrics.count += metrics.count;
				}
			}
		}

		// Calculate average MAPE (Mean Absolute Percentage Error) and format with count
		let mapeDisplay = 'N/A';
		if (this.continuousPredictionMetrics.count > 0) {
			const avgMAPE = (this.continuousPredictionMetrics.totalError / this.continuousPredictionMetrics.count).toFixed(2);
			mapeDisplay = `${avgMAPE}% (${this.continuousPredictionMetrics.count})`;
		}

		// Collect output performance metrics from channels
		const outputMetrics = [];
		for (const [_, channel] of this.channels) {
			if (typeof channel.getOutputPerformanceMetrics === 'function') {
				const metrics = channel.getOutputPerformanceMetrics();
				if (metrics) outputMetrics.push(metrics);
			}
		}

		// Format output performance display
		let outputDisplay = 'N/A';
		if (outputMetrics.length > 0) {
			outputDisplay = outputMetrics.map(m => {
				const formatted = m.format === 'currency'
					? `$${m.value >= 0 ? '+' : ''}${m.value.toFixed(2)}`
					: m.value.toFixed(2);
				return outputMetrics.length > 1 ? `${m.label}:${formatted}` : formatted;
			}).join(', ');
		}

		if (this.frameSummary) console.log(`Frame ${this.frameNumber} | Accuracy: ${baseAccuracy} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 */
	async inferNeurons() {

		// Collect ALL votes from ALL levels in one query
		const allVotes = await this.collectVotes();

		// If no votes, wait for more data
		if (allVotes.length === 0) {
			if (this.debug) console.log('No votes found. Waiting for more data in future frames.');
			return;
		}

		// Determine consensus - pick best per dimension (highest sum of strength * reward)
		// Returns array of inferences with isWinner flag
		const inferences = await this.determineConsensus(allVotes);

		// Apply exploration to actions (may override winners)
		await this.applyExploration(inferences);

		// Save all inferences
		await this.saveInferences(inferences);

		// Notify channels about winning event predictions for continuous tracking (e.g., price prediction)
		this.notifyChannelsOfEventPredictions(inferences);
	}

	/**
	 * Notify channels about winning event predictions for continuous tracking.
	 * Channels can use this to calculate continuous predictions (e.g., price prediction from buckets).
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	notifyChannelsOfEventPredictions(inferences) {

		// Filter to winning event predictions only
		const eventWinners = inferences.filter(inf => inf.isWinner && inf.type === 'event');
		if (eventWinners.length === 0) return;

		// Group by channel
		const byChannel = new Map();
		for (const winner of eventWinners) {
			if (!winner.channel) continue;
			if (!byChannel.has(winner.channel)) byChannel.set(winner.channel, []);
			byChannel.get(winner.channel).push(winner);
		}

		// Notify each channel
		for (const [channelName, winners] of byChannel) {
			const channel = this.channels.get(channelName);
			if (channel && typeof channel.onEventPredictions === 'function')
				channel.onEventPredictions(winners);
		}
	}

	/**
	 * Determine consensus from votes - sum strengths, strength-weighted average for rewards.
	 * All votes are for base level neurons (level 0) - pattern neurons are activated in pattern recognition
	 * Uses per-dimension conflict resolution.
	 * For actions: strength-weighted reward average ensures consistent cycle-based connections override noisy probabilistic ones.
	 * @param {Array} votes - Array of {from_neuron_id, neuron_id, source_type, source_id, strength, reward, distance, source_level}
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async determineConsensus(votes) {
		if (votes.length === 0) return [];

		// Aggregate votes by target neuron - sum strengths, strength-weighted average for rewards
		const aggregated = new Map();
		for (const vote of votes) {
			if (!aggregated.has(vote.neuron_id))
				aggregated.set(vote.neuron_id, { neuron_id: vote.neuron_id, strength: 0, weightedRewardSum: 0, sources: [], sourceNeurons: new Set() });

			const agg = aggregated.get(vote.neuron_id);
			agg.sources.push({ source_type: vote.source_type, source_id: vote.source_id, strength: vote.strength, reward: vote.reward, distance: vote.distance, source_level: vote.source_level });
			agg.sourceNeurons.add(vote.from_neuron_id);

			// Sum strengths, accumulate strength-weighted rewards
			agg.strength += vote.strength;
			agg.weightedRewardSum += vote.strength * vote.reward;
		}

		// Calculate strength-weighted average reward for each target neuron
		for (const [_, agg] of aggregated) agg.reward = agg.strength > 0 ? agg.weightedRewardSum / agg.strength : 0;

		// Add coordinates and group by dimension
		const byDimension = await this.groupVotesByDimension(aggregated);

		// Select winner for each dimension
		const winners = new Set();
		for (const [dimName, dimVotes] of byDimension) {
			const winner = this.selectWinnerForDimension(dimName, dimVotes);
			if (winner) winners.add(winner.neuron_id);
		}

		// Build inferences array with isWinner flag
		const inferences = [];
		for (const [neuronId, agg] of aggregated)
			inferences.push({ ...agg, isWinner: winners.has(neuronId) });

		return inferences;
	}

	/**
	 * Add coordinates to aggregated votes and group by dimension.
	 * Only called for base level (level 0) neurons which have coordinates.
	 * @param {Map} aggregated - Map from aggregateVotesByTarget
	 * @returns {Promise<Map>} Map of dimension name -> array of votes
	 */
	async groupVotesByDimension(aggregated) {
		const neuronIds = [...aggregated.keys()];
		const neuronInfo = await this.getNeuronCoordinates(neuronIds);

		// Add coordinates and neuron type for each neuron
		for (const [neuronId, agg] of aggregated) {
			const info = neuronInfo.get(neuronId);
			if (!info) {
				console.warn(`Warning: Base neuron ${neuronId} not found in getNeuronCoordinates - skipping`);
				continue;
			}
			agg.coordinates = {};
			for (const [dimName, value] of info.coordinates) agg.coordinates[dimName] = value;
			agg.type = info.type; // 'action' or 'event' (from neuron)
			agg.channel = info.channel;
			agg.channel_id = info.channel_id;
		}

		// Group by dimension
		const byDimension = new Map();
		for (const [, agg] of aggregated) {
			if (!agg.coordinates) continue;
			for (const dimName of Object.keys(agg.coordinates)) {
				if (!byDimension.has(dimName)) byDimension.set(dimName, []);
				byDimension.get(dimName).push(agg);
			}
		}

		return byDimension;
	}

	/**
	 * Select winner for a dimension using weighted voting.
	 * All levels contribute with level weighting (already applied in collectVotes).
	 * Events: select by highest weighted strength
	 * Actions: Boltzmann selection on weighted reward
	 * @param {string} dimName - Dimension name
	 * @param {Array} dimVotes - Array of aggregated votes for this dimension
	 * @returns {Object|null} Winning vote or null if no votes
	 */
	selectWinnerForDimension(dimName, dimVotes) {
		if (dimVotes.length === 0) return null;

		// Select winner based on neuron type (all votes for same dimension have same type)
		const isAction = dimVotes[0]?.type === 'action';

		// For actions: Boltzmann selection on weighted reward (already computed)
		if (isAction) {
			const winner = this.boltzmannSelect(dimVotes);
			if (this.debug) console.log(`Voting: ${dimName} (action) Boltzmann = ${winner.coordinates[dimName]} (n${winner.neuron_id}, rwd=${winner.reward.toFixed(2)}, str=${winner.strength.toFixed(2)}, ${dimVotes.length} cand)`);
			return winner;
		}

		// Events: deterministic selection by highest weighted strength
		dimVotes.sort((a, b) => b.strength - a.strength);
		const winner = dimVotes[0];
		if (this.debug) console.log(`Voting: ${dimName} (event) winner = ${winner.coordinates[dimName]} (n${winner.neuron_id}, str=${winner.strength.toFixed(2)}, ${dimVotes.length} cand)`);
		return winner;
	}

	/**
	 * Boltzmann selection from candidates based on reward values.
	 * Uses exponential Boltzmann where probability is proportional to exp(reward / temperature).
	 * Lower temperature = more aggressive (favors higher rewards more strongly).
	 * @param {Array} candidates - Array of objects with reward property
	 * @returns {Object} - Selected candidate
	 */
	boltzmannSelect(candidates) {
		if (candidates.length === 0) return null;
		if (candidates.length === 1) return candidates[0];

		// Exponential Boltzmann with temperature: probability proportional to exp(reward / temperature)
		const expValues = candidates.map(c => Math.exp(c.reward / this.boltzmannTemperature));
		const sum = expValues.reduce((s, v) => s + v, 0);

		// Calculate probabilities
		const probabilities = expValues.map(v => v / sum);

		// Sample from distribution
		const rand = Math.random();
		let cumulative = 0;
		for (let i = 0; i < candidates.length; i++) {
			cumulative += probabilities[i];
			if (rand < cumulative) return candidates[i];
		}

		// Fallback (shouldn't happen due to floating point)
		return candidates[candidates.length - 1];
	}

	/**
	 * Decide whether to explore a channel based on inference strength and reward
	 * Exploration probability is inversely proportional to total effective inference strength (strength * reward)
	 * Higher confidence predictions = lower exploration probability
	 * @param {string} channelName - name of the channel to check
	 * @param {Array} channelBaseInferences - array of channel base inferences with strength and reward
	 * @returns {boolean} - true if we should explore this channel
	 */
	shouldExploreChannel(channelName, channelBaseInferences) {

		// If there are no action inferences, exploration is needed
		if (!channelBaseInferences.length) return true;

		// Calculate total raw inference strength for this channel
		// Use raw strength (not strength * reward) because reward can compound to extreme values
		// that would suppress exploration even when the brain hasn't learned the correct pattern
		const totalInferenceStrength = channelBaseInferences.reduce((sum, inf) => sum + inf.strength, 0);

		// Linear decay from maxExploration to minExploration as strength increases
		// explorationScale defines the strength at which exploration reaches minimum
		const explorationRange = this.maxExploration - this.minExploration;
		let inferenceScale = totalInferenceStrength / this.explorationScale;
		if (inferenceScale > 1.0) inferenceScale = 1.0;
		const explorationProb = this.maxExploration - inferenceScale * explorationRange;

		// Randomly decide if we should explore based on probability
		const explore = Math.random() < explorationProb;
		if (this.debug && explore) console.log(`${channelName}: Total raw strength ${totalInferenceStrength.toFixed(2)} → Exploration prob ${explorationProb.toFixed(2)} → Exploring`);
		return explore;
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
	 * Collect votes from connections or patterns based on highest level PER AGE.
	 * For each age value:
	 *   - If highest level at that age > 0: use ONLY pattern votes from highest level
	 *   - If highest level at that age = 0: use ONLY connection votes from base level
	 * This ensures patterns override connections on a per-frame basis.
	 * Uses sliding window (maxLevelsByAge) to avoid querying max levels every frame.
	 * Uses this.currentMaxLevel (set by recognizePatternNeurons) for age=0.
	 * @returns {Promise<Array>} Array of {neuron_id, source_type, source_id, strength, reward, distance, source_level}
	 */
	async collectVotes() {
		const allVotes = [];

		// Update sliding window: remove oldest, shift all, add new age=0 max level
		// Use currentMaxLevel (set by recognizePatternNeurons) for age=0
		// Shift window: insert new age=0 at front
		if (this.maxLevelsByAge.length >= this.contextLength) this.maxLevelsByAge.pop(); // Remove oldest frame
		this.maxLevelsByAge.unshift(this.currentMaxLevel);

		// Build age groups for pattern and connection queries
		const patternAges = []; // ages where max_level > 0
		const patternLevels = []; // corresponding max levels
		const connectionAges = []; // ages where max_level = 0
		for (let age = 0; age < this.maxLevelsByAge.length && age < this.contextLength; age++) {
			const maxLevel = this.maxLevelsByAge[age];
			if (maxLevel === 0) connectionAges.push(age);
			else {
				patternAges.push(age);
				patternLevels.push(maxLevel);
			}
		}

		// Collect all pattern votes in one query (for ages with patterns)
		if (patternAges.length > 0) {
			const [patternVotes] = await this.conn.query(`
				SELECT pf.pattern_neuron_id as from_neuron_id, pf.inferred_neuron_id as neuron_id, 'pattern' as source_type, pf.id as source_id,
					pf.strength, pf.reward, pf.distance, an.level as source_level, an.age
				FROM active_neurons an
				JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
				WHERE pf.distance = an.age + 1
	            AND pf.strength > 0
				AND (${patternAges.map((age, i) => `(an.age = ${age} AND an.level = ${patternLevels[i]})`).join(' OR ')})
			`);
			allVotes.push(...patternVotes);
		}

		// Collect all connection votes in one query (for ages without patterns)
		if (connectionAges.length > 0) {
			const [connectionVotes] = await this.conn.query(`
				SELECT c.from_neuron_id, c.to_neuron_id as neuron_id, 'connection' as source_type, c.id as source_id,
					c.strength, c.reward, c.distance, an.level as source_level, an.age
				FROM active_neurons an
				JOIN connections c ON c.from_neuron_id = an.neuron_id
				WHERE c.distance = an.age + 1
	            AND c.strength > 0
				AND an.age IN (${connectionAges.join(',')})
				AND an.level = 0
			`);
			allVotes.push(...connectionVotes);
		}

		if (this.debug) console.log(`Collected ${allVotes.length} votes (${patternAges.length} pattern ages, ${connectionAges.length} connection ages)`);

		if (this.debug2) {
			await this.debugActionVotes(allVotes);
			await this.debugEventVotes(allVotes);
		}

		return allVotes;
	}

	/**
	 * Save all inferences in one operation.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	async saveInferences(inferences) {
		if (inferences.length === 0) return;

		// Collect neurons - is_winner: 1 for winners (highest sum of strength * reward per dimension), 0 for losers
		const neurons = [];
		for (const inf of inferences) neurons.push([inf.neuron_id, inf.level || 0, 0, inf.strength, inf.isWinner ? 1 : 0]);

		// Save inferred neurons
		await this.truncateTables([ 'inferred_neurons' ]);
		await this.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, strength, is_winner) VALUES ? 
			ON DUPLICATE KEY UPDATE is_winner = VALUES(is_winner), strength = VALUES(strength)
		`, [neurons]);

		if (this.debug) {
			const winnerCount = inferences.filter(i => i.isWinner).length;
			const loserCount = inferences.filter(i => !i.isWinner).length;
			console.log(`Saved ${inferences.length} inferences (${winnerCount} winners, ${loserCount} losers)`);
		}
	}

	/**
	 * Get coordinates for a list of neuron IDs with dimension info
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @returns {Promise<Map>} Map of neuron_id → {type, channel, channel_id, coordinates: Map of dimension_name → value}
	 */
	async getNeuronCoordinates(neuronIds) {
		if (neuronIds.length === 0) return new Map();

		// Get neuron type and channel from neurons table
		const [neuronRows] = await this.conn.query(`
			SELECT n.id as neuron_id, n.type, n.channel_id, ch.name as channel
			FROM neurons n
			JOIN channels ch ON ch.id = n.channel_id
			WHERE n.id IN (?)
		`, [neuronIds]);

		// Get coordinates from coordinates table
		const [coordRows] = await this.conn.query(`
			SELECT c.neuron_id, c.val, d.name as dimension_name
			FROM coordinates c
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE c.neuron_id IN (?)
		`, [neuronIds]);

		// Build result map
		const neuronCoords = new Map();
		for (const row of neuronRows)
			neuronCoords.set(row.neuron_id, { type: row.type, channel: row.channel, channel_id: row.channel_id, coordinates: new Map() });

		for (const row of coordRows) {
			const neuron = neuronCoords.get(row.neuron_id);
			if (neuron) neuron.coordinates.set(row.dimension_name, row.val);
		}

		return neuronCoords;
	}

	/**
	 * Create pattern neurons and map them to new_patterns.
	 * Creates neurons at peak_level+1 for each peak neuron.
	 * Pattern neurons use type from new_patterns (event/action) and channel_id from peak neuron.
	 */
	async createPatternNeurons() {

		// Get peak neurons with their levels, channel_id, and pattern type from new_patterns
		const [peaks] = await this.conn.query(`
			SELECT np.seq_id, np.peak_neuron_id, np.type as pattern_type, n.level, n.channel_id
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

		// Create pattern neurons with pattern type (from new_patterns) and channel_id (from peak)
		const neurons = peaks.map(p => [p.level + 1, p.pattern_type, p.channel_id]);
		const neuronIds = await this.createNeurons(neurons);

		// Bulk update new_patterns with pattern neuron IDs using CASE statement
		if (peaks.length > 0) {
			const caseWhen = peaks.map((p, i) => `WHEN ${p.seq_id} THEN ${neuronIds[i]}`).join(' ');
			const seqIds = peaks.map(p => p.seq_id);
			await this.conn.query(`
				UPDATE new_patterns
				SET pattern_neuron_id = CASE seq_id ${caseWhen} END
				WHERE seq_id IN (?)
			`, [seqIds]);
		}
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future.
	 * Handles both base-level peaks (connection errors) and pattern-level peaks (pattern errors).
	 *
	 * For connection errors: peak is base neuron, pattern_peaks points to base neuron
	 * For pattern errors: peak is pattern neuron, pattern_peaks points to pattern neuron
	 *   (the pattern's original base peak is accessed via the existing pattern_peaks entry)
	 */
	async createNewPatterns() {

		// Create pattern_peaks entries
		// For pattern errors, this creates a chain: new_pattern -> pattern_neuron -> base_neuron
		await this.conn.query(`
			INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength)
			SELECT np.pattern_neuron_id, np.peak_neuron_id, 1.0
			FROM new_patterns np
		`);

		// Create pattern_past entries (active connections at age=1 leading TO the peak)
		// This captures the full context that was present when the peak was active
		// All patterns created at distance=1 for full context
		// Only same-level connections: pattern identity is determined by same-level context only
		// Only event neurons in pattern_past: actions shouldn't be part of pattern identity
		// Works for both base peaks and pattern peaks - the level check ensures correct context
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, ac.connection_id, 1.0
			FROM new_patterns np
			JOIN neurons peak_n ON peak_n.id = np.peak_neuron_id
			JOIN active_connections ac ON ac.to_neuron_id = np.peak_neuron_id
			JOIN connections c ON c.id = ac.connection_id
			JOIN neurons from_n ON from_n.id = c.from_neuron_id
			WHERE ac.age = 1
			AND from_n.level = peak_n.level
			AND from_n.type = 'event'
		`);

		// Create pattern_future entries from new_pattern_future
		// inferred_neuron_id is always a base-level neuron (event or action)
		// Join on peak_neuron_id and type to match patterns to their future inferences
		// All entries in new_pattern_future have distance=1
		const [futureResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance, strength)
			SELECT np.pattern_neuron_id, npf.inferred_neuron_id, npf.distance, 1.0
			FROM new_patterns np
			JOIN new_pattern_future npf ON npf.peak_neuron_id = np.peak_neuron_id
				AND npf.type = np.type
		`);
		if (this.debug) console.log(`Created ${futureResult.affectedRows} pattern_future entries`);
	}

	/**
	 * Activate newly created pattern neurons at age=1.
	 * All patterns are created at distance=1 (full context).
	 * This allows the pattern to be refined in the same frame if there are multiple errors,
	 * and ensures the "already has active pattern" check works correctly in future frames.
	 */
	async activateNewPatterns() {
		// Get pattern neurons with their level
		const [patterns] = await this.conn.query(`
			SELECT np.pattern_neuron_id, n.level
			FROM new_patterns np
			JOIN neurons n ON n.id = np.pattern_neuron_id
		`);

		if (patterns.length === 0) return;

		// Insert into active_neurons with age = 1
		const activations = patterns.map(p => [p.pattern_neuron_id, p.level, 1]);
		await this.conn.query(`
			INSERT INTO active_neurons (neuron_id, level, age)
			VALUES ?
		`, [activations]);

		if (this.debug) console.log(`Activated ${patterns.length} new pattern neurons`);
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
			WHERE n.type = 'action'
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

		// Step 1: Aggregate by source neuron - sum strengths, strength-weighted average reward
		// This matches determineConsensus logic
		const aggregateBySource = (votes) => {
			const bySource = new Map();
			for (const v of votes) {
				const conn = connMap.get(v.source_id);
				if (!conn) continue;
				const key = conn.from_neuron_id;
				if (!bySource.has(key))
					bySource.set(key, { from_neuron_id: key, strength: 0, weightedRewardSum: 0, from_coords: conn.from_coords, distances: [] });
				const agg = bySource.get(key);
				agg.strength += v.strength;
				agg.weightedRewardSum += v.strength * v.reward;
				agg.distances.push(conn.distance);
			}
			// Calculate strength-weighted average reward per source (matches determineConsensus)
			for (const [_, agg] of bySource)
				agg.reward = agg.strength > 0 ? agg.weightedRewardSum / agg.strength : 0;
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

		// Step 2: Calculate totals - sum strengths, strength-weighted average reward (matches determineConsensus)
		const ownTotal = {
			str: ownAgg.reduce((s, a) => s + a.strength, 0),
			weightedRewardSum: ownAgg.reduce((s, a) => s + a.weightedRewardSum, 0)
		};
		const outTotal = {
			str: outAgg.reduce((s, a) => s + a.strength, 0),
			weightedRewardSum: outAgg.reduce((s, a) => s + a.weightedRewardSum, 0)
		};
		ownTotal.rwd = ownTotal.str > 0 ? ownTotal.weightedRewardSum / ownTotal.str : 0;
		outTotal.rwd = outTotal.str > 0 ? outTotal.weightedRewardSum / outTotal.str : 0;

		// Calculate Boltzmann probabilities (exponential with temperature) - matches boltzmannSelect
		const ownExp = Math.exp(ownTotal.rwd / this.boltzmannTemperature);
		const outExp = Math.exp(outTotal.rwd / this.boltzmannTemperature);
		const sumExp = ownExp + outExp;
		const ownProb = ownExp / sumExp;
		const outProb = outExp / sumExp;

		console.log(`\n=== ACTION VOTES (Cycle ${cycleFrame}/6) ===`);
		console.log(formatAggVotes(ownAgg, `OWN (${ownAgg.length} voters, str=${ownTotal.str.toFixed(1)}, avgRwd=${ownTotal.rwd.toFixed(2)}, prob=${(ownProb * 100).toFixed(1)}%)`));
		console.log(formatAggVotes(outAgg, `OUT (${outAgg.length} voters, str=${outTotal.str.toFixed(1)}, avgRwd=${outTotal.rwd.toFixed(2)}, prob=${(outProb * 100).toFixed(1)}%)`));
		console.log(`  SELECTION: Boltzmann (OWN ${(ownProb * 100).toFixed(1)}% vs OUT ${(outProb * 100).toFixed(1)}%)`);
		console.log(`===================\n`);
	}

	/**
	 * Debug helper: show votes for event neurons (price_change, volume_change, etc.)
	 * Shows which patterns/connections are voting for which event predictions
	 * @param {Array} allVotes - Array of all votes
	 */
	async debugEventVotes(allVotes) {

		// Get all event neuron IDs from votes
		const eventNeuronIds = [...new Set(allVotes.map(v => v.neuron_id))];
		if (eventNeuronIds.length === 0) return;

		// Get neuron info to identify event neurons and parse coordinates into dimensions
		const [neurons] = await this.conn.query(`
			SELECT n.id, n.type, n.channel_id,
				GROUP_CONCAT(CONCAT(d.name, '=', coord.val) ORDER BY d.name SEPARATOR ', ') as coords,
				d.name as dim_name, coord.val as dim_val
			FROM neurons n
			LEFT JOIN coordinates coord ON coord.neuron_id = n.id
			LEFT JOIN dimensions d ON d.id = coord.dimension_id
			WHERE n.id IN (?)
			GROUP BY n.id, d.name, coord.val
		`, [eventNeuronIds]);

		// Filter to event neurons only
		const eventNeurons = neurons.filter(n => n.type === 'event');
		if (eventNeurons.length === 0) return;

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

		// Build neuron map with dimension info
		const neuronMap = new Map();
		for (const n of eventNeurons) {
			if (!neuronMap.has(n.id))
				neuronMap.set(n.id, { id: n.id, type: n.type, coords: n.coords, dimensions: new Map() });
			if (n.dim_name)
				neuronMap.get(n.id).dimensions.set(n.dim_name, n.dim_val);
		}

		// Get bucket-to-percentage mapping from stock channel if available
		const stockChannel = this.channels.get('TEST');
		const bucketToPercent = stockChannel ? this.buildBucketPercentMap(stockChannel) : null;

		// Calculate cycle frame (1-6) based on frame number
		const cycleFrame = ((this.frameNumber - 1) % 6) + 1;

		// Group votes by event neuron
		const votesByEvent = new Map();
		for (const vote of allVotes) {
			const neuron = neuronMap.get(vote.neuron_id);
			if (!neuron) continue; // Skip non-event neurons

			if (!votesByEvent.has(vote.neuron_id))
				votesByEvent.set(vote.neuron_id, []);
			votesByEvent.get(vote.neuron_id).push(vote);
		}

		if (votesByEvent.size === 0) return;

		// Aggregate by source neuron for each event - matches determineConsensus aggregation
		const aggregateBySource = (votes) => {
			const bySource = new Map();
			for (const v of votes) {
				const conn = connMap.get(v.source_id);
				if (!conn) continue;
				const key = conn.from_neuron_id;
				if (!bySource.has(key))
					bySource.set(key, { from_neuron_id: key, strength: 0, from_coords: conn.from_coords, distances: [] });
				const agg = bySource.get(key);
				agg.strength += v.strength;
				agg.distances.push(conn.distance);
			}
			return [...bySource.values()];
		};

		// Group by dimension to find winners - sum all vote strengths per neuron (matches determineConsensus)
		const byDimension = new Map();
		for (const [neuronId, votes] of votesByEvent) {
			const neuron = neuronMap.get(neuronId);
			const totalStrength = votes.reduce((sum, v) => sum + v.strength, 0);

			for (const [dimName, dimVal] of neuron.dimensions) {
				if (!byDimension.has(dimName))
					byDimension.set(dimName, []);
				byDimension.get(dimName).push({ neuronId, neuron, votes, totalStrength, dimVal });
			}
		}

		// Format aggregated votes with source info
		const formatAggVotes = (aggVotes) => {
			if (aggVotes.length === 0) return '    no votes';
			const lines = [];
			for (const agg of aggVotes) {
				const coordsWithPercent = this.formatCoordsWithPercent(agg.from_coords, bucketToPercent);
				const distStr = agg.distances.length > 1 ? `d=[${agg.distances.join(',')}]` : `d=${agg.distances[0]}`;
				lines.push(`    ${coordsWithPercent} (${distStr}) → str=${agg.strength.toFixed(1)}`);
			}
			return lines.join('\n');
		};

		console.log(`\n=== EVENT VOTES (Cycle ${cycleFrame}/6) ===`);

		// Show votes per dimension with winner highlighted
		for (const [dimName, candidates] of byDimension) {
			// Sort by strength to find winner
			candidates.sort((a, b) => b.totalStrength - a.totalStrength);
			const winner = candidates[0];

			console.log(`  ${dimName} (${candidates.length} candidates):`);

			for (const cand of candidates) {
				const isWinner = cand.neuronId === winner.neuronId;
				const marker = isWinner ? '★ WINNER' : '';
				const coordsWithPercent = this.formatCoordsWithPercent(cand.neuron.coords, bucketToPercent);
				const aggVotes = aggregateBySource(cand.votes);

				console.log(`    ${coordsWithPercent} (n${cand.neuronId}) str=${cand.totalStrength.toFixed(1)} ${marker}`);
				console.log(formatAggVotes(aggVotes));
			}
		}

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
}