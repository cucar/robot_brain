import readline from 'node:readline';
import getMySQLConnection from './db/db.js';

/**
 * BrainMySQL Class - MySQL-based brain implementation
 * Uses MySQL for all data storage and processing
 */
export default class BrainMySQL {

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
		this.eventErrorMinStrength = 2; // minimum prediction strength to create error-driven patterns
		this.actionRegretMinStrength = 2; // minimum inference strength to create action regret pattern
		this.actionRegretMinPain = 0; // minimum pain (negative reward magnitude) to create action regret pattern (0 = any negative reward triggers regret)
		this.mergePatternThreshold = 0.5; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.maxLevels = 10; // just to prevent against infinite recursion

		// voting parameters
		this.levelVoteMultiplier = 3; // how much to weight votes from higher levels

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle (reduced to preserve learned connections)
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle

		//************************************************************
		// operational properties
		//************************************************************

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// Map of channel name -> Set of action neurons (for exploration)
		this.channelActions = new Map();

		// Prediction accuracy tracking (cumulative stats for base level only)
		this.accuracyStats = { correct: 0, total: 0 };

		// Action reward tracking (cumulative stats for base level only)
		this.rewardStats = { totalReward: 0, count: 0 };

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

		// Debugging info and flags
		this.frameNumber = 0;
		this.frameSummary = true; // show frame summary or not
		this.debug = false;
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info

		// Create readline interface for pausing between frames - used when debugging
		this.waitForUserInput = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
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
		if (this.debug) console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * Get channel instance by name
	 */
	getChannel(channelName) {
		return this.channels.get(channelName);
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async initDB() {
		this.conn = await getMySQLConnection();
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (memory tables)...');
		await this.truncateTables([
			'active_neurons',
			'matched_patterns',
			'matched_pattern_past',
			'new_pattern_future',
			'new_patterns',
			'inferred_neurons',
			'inference_votes'
		]);
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		await this.resetContext();
		console.log('Hard resetting brain (all learned data)...');
		await this.truncateTables([
			'channels',
			'dimensions',
			'neurons',
			'base_neurons',
			'coordinates',
			'connections',
			'pattern_peaks',
			'pattern_past',
			'pattern_future'
		]);
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
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

		// pre-create action neurons for all channels
		await this.initializeActionNeurons();

		// initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) await channel.initialize();
	}

	/**
	 * Pre-create action neurons for all channels.
	 * This ensures exploration can find action neurons even before any connections exist.
	 * Also builds the channelActions map for exploration.
	 */
	async initializeActionNeurons() {
		for (const [channelName, channel] of this.channels) {
			const actionCoords = channel.getActions();
			if (actionCoords.length === 0) continue;

			// Build frame points for action neurons
			const framePoints = actionCoords.map(coords => ({
				coordinates: coords,
				channel: channelName,
				channel_id: this.channelNameToId[channelName],
				type: 'action'
			}));

			// Create neurons (getFrameNeurons will create if they don't exist)
			const neuronIds = await this.getFrameNeurons(framePoints);

			// Build channelActions map for exploration
			const actionSet = new Set();
			for (let i = 0; i < neuronIds.length; i++)
				actionSet.add({ id: neuronIds[i], coordinates: actionCoords[i] });
			this.channelActions.set(channelName, actionSet);

			if (this.debug) console.log(`Created ${actionCoords.length} action neurons for ${channelName}`);
		}
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
		if (this.debug) console.log('Channels loaded:', this.channelNameToId);
	}

	/**
	 * Initialize dimensions for all registered channels
	 */
	async initializeDimensions() {
		if (this.debug) console.log('Initializing dimensions for registered channels...');
		for (const [, channel] of this.channels) {
			for (const dim of channel.getEventDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
			for (const dim of channel.getOutputDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
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
		if (this.debug) console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * Get neuron ID by dimension name and value.
	 * Used for diagnostic output to show which neurons correspond to which values.
	 * @param {string} dimensionName - The dimension name
	 * @param {number|string} value - The value to look up
	 * @returns {Promise<number|null>} - Neuron ID or null if not found
	 */
	async getNeuronIdByDimensionValue(dimensionName, value) {
		const dimensionId = this.dimensionNameToId[dimensionName];
		if (!dimensionId) return null;

		const [rows] = await this.conn.query(`
			SELECT neuron_id FROM coordinates
			WHERE dimension_id = ? AND val = ?
			LIMIT 1
		`, [dimensionId, value]);

		return rows.length > 0 ? rows[0].neuron_id : null;
	}

	/**
	 * Get frame outputs for all channels from inferred_neurons table (MySQL implementation)
	 * Reads winning action neurons (is_winner=1) grouped by channel
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
	 * @returns Promise<boolean> - true if frame was processed, false if no more data available
	 */
	async processFrame() {
		const frameStart = performance.now();

		// Handle all frame I/O: get frame, execute actions, get rewards, display header
		if (!await this.processFrameIO()) return false;

		// age the active neurons in memory context - sliding the temporal window
		// deletion of aged-out neurons is deferred to after pattern learning
		await this.ageNeurons();

		// activate base level neurons from the frame - what's happening right now?
		await this.processBaseNeurons();

		// recognize, refine and learn patterns from the base neurons
		await this.processPatternNeurons();

		// deactivate aged-out neurons AFTER pattern learning captured full context
		await this.deactivateOldNeurons();

		// do predictions and outputs - what's going to happen next? and what's our best response?
		await this.inferNeurons();

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// used to avoid curse of dimensionality and delete dead connections/neurons
		await this.runForgetCycle();

		// show frame processing summary
		this.printFrameSummary(performance.now() - frameStart);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');

		// return true to indicate that we have processed the frame successfully
		return true;
	}

	/**
	 * Handles all frame I/O: get frame, execute actions, get rewards, display header
	 * @returns Promise<boolean> - true if frame data available, false if no more data
	 */
	async processFrameIO() {

		// get the current frame from all channels - includes events and previously inferred actions
		await this.getFrame();
		if (!this.frame || this.frame.length === 0) return false;

		// execute the previously inferred actions in all channels
		await this.executeActions();

		// get rewards from all channels based on executed actions
		await this.getRewards();

		// display diagnostic frame header if enabled
		this.displayFrameHeader();

		return true;
	}

	/**
	 * Returns the current frame combined from all registered channels
	 * Each frame point includes: coordinates, channel, type
	 * Populates this.frame with events from channels and actions from previous inference
	 */
	async getFrame() {
		this.frame = [];

		// Increment frame counter to be able to track inactivity
		this.frameNumber++;
		if (this.debug) console.log('******************************************************************');
		if (this.debug) console.log(`OBSERVING FRAME ${this.frameNumber}`);

		// Get all frame outputs from previous frame's inference (one query for all channels)
		const frameOutputs = await this.getFrameOutputs();

		// Process each channel: get inputs from channel, get outputs from brain tables
		for (const [channelName, channel] of this.channels) {
			const channelId = this.channelNameToId[channelName];

			// Get the frame event inputs from the channel (current state before any outputs are executed)
			const channelEvents = await channel.getFrameEvents();
			for (const event of channelEvents)
				this.frame.push({ coordinates: event, channel: channelName, channel_id: channelId, type: 'event' });

			// Get inferred actions or explore if none
			const inferredActions = frameOutputs.get(channelName) || [];
			const channelActions = this.applyExploration(channelName, inferredActions);
			for (const action of channelActions)
				this.frame.push({ coordinates: action, channel: channelName, channel_id: channelId, type: 'action' });
		}

		if (this.debug) console.log(`Processing frame: ${this.frame.length} neurons`);
		if (this.debug) console.log(`frame points: ${JSON.stringify(this.frame)}`);
		if (this.debug) console.log('******************************************************************');
	}

	/**
	 * Execute previously inferred actions for all channels
	 */
	async executeActions() {
		const channelActions = this.getFrameActions();
		for (const [channelName, channel] of this.channels)
			await channel.executeOutputs(channelActions.get(channelName) || []);
	}

	/**
	 * Extract action outputs from this.frame grouped by channel
	 * @returns {Map} - Map of channel names to array of action coordinate objects
	 */
	getFrameActions() {
		const channelActions = new Map();
		for (const point of this.frame)
			if (point.type === 'action') {
				if (!channelActions.has(point.channel)) channelActions.set(point.channel, []);
				channelActions.get(point.channel).push(point.coordinates);
			}
		return channelActions;
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getRewards() {
		if (this.debug) console.log('Getting rewards feedback from all channels...');
		this.rewards = new Map();
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const reward = await channel.getRewards();
			if (reward !== 0) { // Only process non-neutral feedback (additive: 0 = neutral)
				if (this.debug) console.log(`${channelName}: reward ${reward.toFixed(3)}`);
				this.rewards.set(channelName, reward);
				feedbackCount++;
			}
		}

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Received rewards from ${feedbackCount} channels`);
			else console.log('No rewards from any channels');
		}
		if (this.debug && feedbackCount > 0)
			console.log(`Channel rewards:`, Array.from(this.rewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));
	}

	/**
	 * Display diagnostic frame header with frame number and observations
	 */
	displayFrameHeader() {
		if (!this.diagnostic) return;

		// Display reward information
		if (this.rewards.size > 0) this.displayRewards();

		// Build observation string from frame
		const observations = [];
		for (const point of this.frame)
			for (const [dim, val] of Object.entries(point.coordinates))
				observations.push(`${dim}=${val}`);

		console.log(`\nF${this.frameNumber} | Obs: ${observations.join(', ')}`);
	}

	/**
	 * Display reward information for diagnostic output
	 */
	displayRewards() {
		const rewardParts = [];
		for (const [channelName, reward] of this.rewards)
			rewardParts.push(`${channelName}:${reward.toFixed(3)}x`);
		console.log(`  Rewards: ${rewardParts.join(', ')}`);
	}

	/**
	 * Ages all neurons in the context by 1.
	 * Deletion of aged-out neurons is deferred to deactivateOldNeurons() at end of frame,
	 * so that pattern creation can capture the full context before neurons are deleted.
	 */
	async ageNeurons() {
		if (this.debug) console.log('Aging active neurons...');

		// age all neurons - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
	}

	/**
	 * Deactivates neurons that have aged out of the context window.
	 * Called at end of frame after pattern learning, so patterns can capture full context.
	 */
	async deactivateOldNeurons() {
		// Skip deletions until we have enough frames
		if (this.frameNumber < this.contextLength + 1) return;

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.contextLength})`);
	}

	/**
	 * recognizes and activates base level neurons from frame
	 */
	async processBaseNeurons() {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(this.frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// insert the new base neurons to the active neurons table
		await this.insertActiveNeurons(neuronIds);

		// reinforce connections between active neurons in the base level
		await this.reinforceConnections();

		// learn from connection action inferences
		await this.rewardConnections();

		// Track event prediction accuracy
		await this.trackEventAccuracy();

		// Track action reward stats
		await this.trackActionRewards();
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame	 *  points have structure: { coordinates, channel, channel_id, type }
	 */
	async getFrameNeurons(frame) {

		// try to get all the neurons that have coordinates matching each point
		// return format: [{ point, neuron_id }] where point is the full frame point
		const matches = await this.matchFrameNeurons(frame);
		if (this.debug) console.log('pointNeuronMatches', matches.map(match => JSON.stringify(match)).join('\n'));

		// matching neuron ids to be returned for each point of the frame
		const neuronIds = matches.filter(p => p.neuron_id).map(p => p.neuron_id);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => !p.neuron_id);
		if (pointsNeedingNeurons.length > 0) {
			if (this.debug) console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with dedupe.`);
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
		const neuronIds = await this.insertBaseNeurons(framePoints.length);
		await this.insertBaseNeuronMetadata(neuronIds, framePoints);
		await this.setNeuronCoordinates(neuronIds, framePoints);
		return neuronIds;
	}

	/**
	 * Creates new base neurons (level 0) in bulk and returns their IDs.
	 * MySQL guarantees sequential auto-increment IDs.
	 * @param {number} count - Number of neurons to create
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async insertBaseNeurons(count) {
		if (count === 0) return [];
		const rows = Array.from({ length: count }, () => [0]);
		const [insertResult] = await this.conn.query('INSERT INTO neurons (level) VALUES ?', [rows]);
		const firstNeuronId = insertResult.insertId;
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * Insert base neuron metadata into base_neurons table
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @param {Array} framePoints - Array of frame points with channel_id and type
	 */
	async insertBaseNeuronMetadata(neuronIds, framePoints) {
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
	 * Inserts active neurons at age 0
	 */
	async insertActiveNeurons(neuronIds) {
		if (neuronIds.length === 0) return;
		const activations = neuronIds.map(neuronId => [neuronId, 0]);
		await this.conn.query(`INSERT INTO active_neurons (neuron_id, age) VALUES ?`, [activations]);
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
            AND f.age < :contextLength -- exclude neurons about to be deleted (kept for pattern context only)
            AND t.age = 0 -- build connections from base neurons that are just activated
            -- if the connection already exists, increment its strength 
			ON DUPLICATE KEY UPDATE connections.strength = LEAST(:maxConnectionStrength, connections.strength + VALUES(strength))
		`, { maxConnectionStrength: this.maxConnectionStrength, contextLength: this.contextLength });
	}

	/**
	 * Apply rewards to action connections
	 */
	async rewardConnections() {

		// nothing to update if there are no rewards
		if (this.rewards.size === 0) return;

		// apply rewards reinforcement to executed actions via connection inference
		// winners were executed and added to frame, then activated when recognizing neurons
		// they appear in active_neurons at age=0 (just activated)
		// exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
		// leave losers alone - we don't know what would have happened if they were executed
		const channelIds = Array.from(this.rewards.keys()).map(name => this.channelNameToId[name]);
		const rewardCase = this.buildChannelRewardCase('c.reward');
		const [result] = await this.conn.query(`
            UPDATE connections c
            JOIN active_neurons an_to ON c.to_neuron_id = an_to.neuron_id AND an_to.age = 0
            JOIN active_neurons an_from ON c.from_neuron_id = an_from.neuron_id AND c.distance = an_from.age
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
	 * @param {string} rewardColumn - The column to use in ELSE clause (e.g., 'c.reward' or 'pf.reward')
	 * @returns {string} SQL CASE statement like "CASE b.channel_id WHEN 1 THEN 0.5 WHEN 2 THEN -0.3 ELSE c.reward END"
	 */
	buildChannelRewardCase(rewardColumn) {
		if (this.rewards.size === 0) return '';
		const caseWhen = Array.from(this.rewards.entries())
			.map(([name, reward]) => `WHEN ${this.channelNameToId[name]} THEN ${reward}`)
			.join(' ');
		return `CASE b.channel_id ${caseWhen} ELSE ${rewardColumn} END`;
	}

	/**
	 * Get channel IDs with painful rewards (below actionRegretMinPain threshold).
	 * @returns {Array<number>} Array of channel IDs with negative rewards
	 */
	getPainfulChannelIds() {
		const painfulChannelIds = [];
		for (const [channelName, reward] of this.rewards)
			if (reward < this.actionRegretMinPain) painfulChannelIds.push(this.channelNameToId[channelName]);
		return painfulChannelIds;
	}

	/**
	 * Track event prediction accuracy by comparing inferred_neurons (age=1) to active_neurons (age=0).
	 * Only validates base level (level 0) event predictions.
	 * Only validates winners (is_winner=1) since losers are alternative hypotheses that were rejected.
	 */
	async trackEventAccuracy() {

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
	 * Track action reward stats by summing rewards for executed action winners.
	 * Only tracks base level (level 0) action inferences.
	 * Only tracks winners (is_winner=1) since losers were not executed.
	 */
	async trackActionRewards() {
		if (this.rewards.size === 0) return;

		// Count action winners that were executed (now active at age=0)
		const [rows] = await this.conn.query(`
			SELECT COUNT(*) as count
			FROM inferred_neurons inf
			JOIN base_neurons b ON b.neuron_id = inf.neuron_id
			JOIN active_neurons an ON an.neuron_id = inf.neuron_id AND an.age = 0
			WHERE b.type = 'action'
			AND inf.is_winner = 1
		`);

		const actionCount = rows[0].count;
		if (actionCount === 0) return;

		// Sum rewards from all channels for this frame
		let frameReward = 0;
		for (const [_, reward] of this.rewards)
			frameReward += reward;

		// Update rewardStats (cumulative across frames)
		this.rewardStats.totalReward += frameReward;
		this.rewardStats.count += actionCount;

		if (this.debug && actionCount > 0)
			console.log(`Action rewards: ${frameReward.toFixed(3)} for ${actionCount} actions`);
	}

	/**
	 * recognize, refine and learn patterns from the base neurons
	 */
	async processPatternNeurons() {

		// discover and activate patterns using connections - start recursion from base level
		await this.recognizePatterns();

		// refine the learned pattern definitions from prediction errors and action regret
		await this.refinePatterns();

		// learn new patterns from failed predictions and action regret
		await this.learnNewPatterns();
	}

	/**
	 * detects all spatial levels in age=0 neurons using unified connections - start from base level and go as high as possible
	 */
	async recognizePatterns() {
		let level = 0; // contains the max level of neurons we were able to recognize and activate
		while (true) {

			// process the level to detect patterns - returns if there are patterns found or not
			const patternsFound = await this.recognizeLevelPatterns(level);

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
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 */
	async recognizeLevelPatterns(level) {
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// Match active connections to known patterns and write to matched_patterns table
		const matchCount = await this.matchObservedPatterns(level);
		if (matchCount === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// strengthen the pattern peaks that are just activated
		await this.strengthenActivePatterns();

		// update pattern_past for matched patterns - add/strengthen observed connections, weaken unobserved connections
		await this.refinePatternPast();

		// Activate all matched pattern neurons
		await this.activateMatchedPatterns();

		return true; // patterns found
	}

	/**
	 * Match active neurons to known pattern contexts and saves them in matched_patterns table.
	 * @param {number} level - The level to match patterns for (peak neuron level)
	 * @returns {Promise<number>} - Number of matched patterns
	 */
	async matchObservedPatterns(level) {
		if (this.debug) console.log('Matching active connections to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_patterns');
		await this.conn.query('TRUNCATE matched_pattern_past');

		// Dynamic-length pattern matching (like words in text):
		// 1. Threshold check: pattern matches if matched_count / pattern_past_count >= threshold
		// 2. Winner selection: among matched patterns for a peak, pick highest total strength of matching context
		// This builds a branching structure of switch statements - longer, more specific patterns win
		const [result] = await this.conn.query(`
			INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
			SELECT m.peak_neuron_id, m.pattern_neuron_id
			FROM (
				SELECT pp.peak_neuron_id, pp.pattern_neuron_id,
					ROW_NUMBER() OVER (
						PARTITION BY pp.peak_neuron_id
						ORDER BY SUM(IF(an_ctx.neuron_id IS NOT NULL, p.strength, 0)) DESC, pp.pattern_neuron_id
					) as rn
				FROM active_neurons an_peak
				JOIN neurons n_peak ON n_peak.id = an_peak.neuron_id AND n_peak.level = ?
				JOIN pattern_peaks pp ON an_peak.neuron_id = pp.peak_neuron_id
				JOIN pattern_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
				LEFT JOIN active_neurons an_ctx ON an_ctx.neuron_id = p.context_neuron_id AND an_ctx.age = p.context_age
				LEFT JOIN neurons n_ctx ON n_ctx.id = an_ctx.neuron_id AND n_ctx.level = n_peak.level
				WHERE an_peak.age = 0
				GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
				-- pattern matches if at least threshold percentage of its pattern_past neurons are active
				HAVING SUM(IF(an_ctx.neuron_id IS NOT NULL, 1, 0)) >= COUNT(*) * ?
			) m
			WHERE m.rn = 1
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
		await this.populateMatchedPatternContext();

		// return the number of matched patterns
		return result.affectedRows;
	}

	/**
	 * Populates the matched_pattern_past table with the context of the matched patterns
	 */
	async populateMatchedPatternContext() {

		// Common: context neurons that ARE active at correct age
		await this.conn.query(`
			INSERT INTO matched_pattern_past (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT p.pattern_neuron_id, p.context_neuron_id, p.context_age, 'common'
			FROM matched_patterns mp
			JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id
			JOIN active_neurons an ON an.neuron_id = p.context_neuron_id AND an.age = p.context_age
		`);

		// Missing: context neurons NOT active at correct age
		await this.conn.query(`
			INSERT INTO matched_pattern_past (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT p.pattern_neuron_id, p.context_neuron_id, p.context_age, 'missing'
			FROM matched_patterns mp
			JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id
			WHERE NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = p.context_neuron_id AND an.age = p.context_age
			)
		`);

		// Novel: active neurons at same level, not in pattern_past (all channels)
		// Exclude neurons at age >= contextLength (kept for pattern context only, not for learning new context)
		await this.conn.query(`
			INSERT INTO matched_pattern_past (pattern_neuron_id, context_neuron_id, context_age, status)
			SELECT mp.pattern_neuron_id, an_context.neuron_id, an_context.age, 'novel'
			FROM matched_patterns mp
			JOIN neurons n_peak ON n_peak.id = mp.peak_neuron_id
			JOIN active_neurons an_context ON an_context.age > 0 AND an_context.age < ?
            JOIN neurons n_context ON n_context.id = an_context.neuron_id AND n_context.level = n_peak.level
			-- pick the context neurons that are not already in the context in the age they are observed
			WHERE NOT EXISTS (
				SELECT 1 FROM pattern_past p
				WHERE p.pattern_neuron_id = mp.pattern_neuron_id
			    AND p.context_neuron_id = an_context.neuron_id
			    AND p.context_age = an_context.age
			)
		    -- exclude actions from context - note that this may be a stock specific thing - actions can learn patterns in robotics context
			AND NOT EXISTS (SELECT 1 FROM base_neurons b WHERE b.neuron_id = an_context.neuron_id AND b.type = 'action')
		`, [this.contextLength]);
	}

	/**
	 * update pattern_peaks that are just activated
	 */
	async strengthenActivePatterns() {

		// Reinforce pattern_peaks strength for matched patterns
		await this.conn.query(`
			UPDATE pattern_peaks pp
			JOIN matched_patterns mp ON pp.pattern_neuron_id = mp.pattern_neuron_id
			SET pp.strength = LEAST(?, pp.strength + 1)
		`, [this.maxConnectionStrength]);
	}

	/**
	 * Refine matched patterns using pre-analyzed connection sets.
	 * Uses matched_pattern_past table populated by matchObservedPatterns:
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
			FROM matched_pattern_past
			WHERE status = 'novel'
		`);

		// Strengthen common context
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_past mpc 
				ON p.pattern_neuron_id = mpc.pattern_neuron_id
				AND p.context_neuron_id = mpc.context_neuron_id
				AND p.context_age = mpc.context_age
			SET p.strength = LEAST(?, p.strength + 1)
			WHERE mpc.status = 'common'
		`, [this.maxConnectionStrength]);

		// Weaken missing context
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_past mpc
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
	 * Activate matched pattern neurons at age 0
	 */
	async activateMatchedPatterns() {
		await this.conn.query(`
			INSERT INTO active_neurons (neuron_id, age)
			SELECT pattern_neuron_id, 0
			FROM matched_patterns
		`);
	}

	/**
	 * Learns patterns from prediction errors and action regret and continues to refine them as they are observed
	 */
	async refinePatterns() {

		// update pattern_future with current events based on observations
		await this.refinePatternEventsFuture();

		// update pattern_future with actions based on rewards
		await this.refinePatternActionsFuture();
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
	async refinePatternEventsFuture() {

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

		// 3. ADD NOVEL NEURONS: Active event neurons not yet in pattern_future at this distance
		// When a pattern is active at age X, add currently active (age=0) event neurons at distance X
		// Patterns are cross-channel, so add all active event neurons regardless of channel
		// INSERT IGNORE prevents duplicates for (pattern_neuron_id, inferred_neuron_id, distance)
		// Exclude patterns at age >= contextLength (kept for pattern context only, not for learning new futures)
		const [novelResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance)
			SELECT ap.neuron_id as pattern_neuron_id, an.neuron_id as inferred_neuron_id, ap.age
			FROM active_neurons ap
			JOIN neurons n_pattern ON n_pattern.id = ap.neuron_id AND n_pattern.level > 0
			JOIN active_neurons an ON an.age = 0
			JOIN base_neurons b ON b.neuron_id = an.neuron_id AND b.type = 'event'
			WHERE ap.age > 0 AND ap.age < ?
		`, [this.contextLength]);
		if (this.debug) console.log(`Added ${novelResult.affectedRows} novel neurons to event pattern_future`);
	}

	/**
	 * Refine pattern action predictions based on executed actions and rewards.
	 */
	async refinePatternActionsFuture() {

		// pattern already had this action → update reward
		await this.rewardInferredActions();

		// pattern was active but didn't have this action → add with reward
		await this.learnNewActions();

		// painful action in pattern_future → add one untried alternative
		await this.addAlternativeActions();
	}

	/**
	 * Reward actions that patterns already had in pattern_future and were executed.
	 * Strengthens the pattern_future entry and updates reward via exponential smoothing.
	 */
	async rewardInferredActions() {
		if (this.rewards.size === 0) return;

		const rewardCase = this.buildChannelRewardCase('pf.reward');
		const [result] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN active_neurons an_target ON an_target.neuron_id = pf.inferred_neuron_id AND an_target.age = 0
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
			SET pf.strength = LEAST(?, pf.strength + 1),
			    pf.reward = ? * (${rewardCase}) + ? * pf.reward
			WHERE b.type = 'action'
		`, [this.maxConnectionStrength, this.rewardExpSmooth, 1 - this.rewardExpSmooth]);

		if (this.debug) console.log(`Rewarded ${result.affectedRows} inferred actions`);
	}

	/**
	 * Learn executed actions that active patterns didn't have in pattern_future.
	 * Only for channels with non-zero rewards (positive or negative).
	 */
	async learnNewActions() {
		if (this.rewards.size === 0) return;

		// Get all channel IDs with non-zero rewards
		const rewardedChannelIds = [];
		for (const [channelName, reward] of this.rewards) {
			const channelId = this.channelNameToId[channelName];
			if (channelId !== undefined && reward !== 0)
				rewardedChannelIds.push(channelId);
		}
		if (rewardedChannelIds.length === 0) return;

		const rewardCase = this.buildChannelRewardCase('0');
		const [result] = await this.conn.query(`
			INSERT INTO pattern_future (pattern_neuron_id, distance, inferred_neuron_id, reward)
			SELECT an_pattern.neuron_id, an_pattern.age, an_action.neuron_id, ${rewardCase}
			FROM active_neurons an_pattern
			JOIN neurons n_pattern ON n_pattern.id = an_pattern.neuron_id AND n_pattern.level > 0
			JOIN active_neurons an_action ON an_action.age = 0
			JOIN base_neurons b ON b.neuron_id = an_action.neuron_id AND b.type = 'action'
			WHERE an_pattern.age > 0 AND an_pattern.age < ?
			AND b.channel_id IN (${rewardedChannelIds.join(',')})
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf_existing
				WHERE pf_existing.pattern_neuron_id = an_pattern.neuron_id
				AND pf_existing.distance = an_pattern.age
				AND pf_existing.inferred_neuron_id = an_action.neuron_id
			)
			ON DUPLICATE KEY UPDATE
				strength = LEAST(?, strength + 1),
				reward = ? * VALUES(reward) + ? * reward
		`, [this.contextLength, this.maxConnectionStrength, this.rewardExpSmooth, 1 - this.rewardExpSmooth]);

		if (this.debug) console.log(`Learned ${result.affectedRows} new actions`);
	}

	/**
	 * Add one untried alternative action for each (pattern, distance, channel) with a painful action.
	 * After rewardInferredActions and learnNewActions, all executed actions are in pattern_future.
	 * This finds patterns with painful actions and suggests something else to try.
	 */
	async addAlternativeActions() {
		const painfulChannelIds = this.getPainfulChannelIds();
		if (painfulChannelIds.length === 0) return;

		const [result] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, distance, inferred_neuron_id)
			SELECT pf.pattern_neuron_id, pf.distance, MIN(b_alt.neuron_id)
			FROM pattern_future pf
			JOIN active_neurons an_pattern ON an_pattern.neuron_id = pf.pattern_neuron_id AND pf.distance = an_pattern.age
			JOIN active_neurons an_action ON an_action.neuron_id = pf.inferred_neuron_id AND an_action.age = 0
			JOIN base_neurons b ON b.neuron_id = pf.inferred_neuron_id
			JOIN base_neurons b_alt ON b_alt.channel_id = b.channel_id AND b_alt.type = 'action'
			WHERE b.type = 'action'
			AND an_pattern.age > 0 AND an_pattern.age < ?
			AND b.channel_id IN (${painfulChannelIds.join(',')})
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf_existing
				WHERE pf_existing.pattern_neuron_id = pf.pattern_neuron_id
				AND pf_existing.distance = pf.distance
				AND pf_existing.inferred_neuron_id = b_alt.neuron_id
			)
			GROUP BY pf.pattern_neuron_id, pf.distance, b.channel_id
		`, [this.contextLength]);

		if (this.debug) console.log(`Added ${result.affectedRows} alternative actions`);
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Processes all levels in bulk, like inferNeurons.
	 */
	async learnNewPatterns() {

		// Find the neurons that should be in pattern_future of new patterns
		// (prediction errors and action regret, unified in one method)
		const newPatternFutureCount = await this.populateNewPatternFuture();
		if (this.debug) console.log(`New pattern future count: ${newPatternFutureCount}`);
		if (newPatternFutureCount === 0) return;
		if (newPatternFutureCount > 0) {
			// this.waitForUserInput = true;
			// await this.waitForUser('New pattern future count > 0');
		}

		// Populate new_patterns table with peaks from new pattern future inferences
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons();

		// Create new patterns in neurons, pattern_peaks, pattern_past, pattern_future
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
	 */
	async populateNewPatternFuture() {
		await this.conn.query(`TRUNCATE new_pattern_future`);
		const eventCount = await this.populateNewPatternEvents();
		const actionCount = await this.populateNewPatternActions();
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
	async populateNewPatternEvents() {
		const baseCount = await this.populateNewPatternBaseEvents();
		const highCount = await this.populateNewPatternHighEvents();
		return baseCount + highCount;
	}

	/**
	 * Populate base-level event prediction errors (from connection inferences).
	 * Uses inference_votes which already excludes peaks with active patterns.
	 * Peak predicted X, but X didn't happen and Y happened instead.
	 * @returns {Promise<number>} Number of base event pattern futures created
	 */
	async populateNewPatternBaseEvents() {
		const [connResult] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT v.from_neuron_id, an.neuron_id as inferred_neuron_id, v.distance
			FROM inference_votes v
			-- get neuron info for the inferred neuron (target)
			JOIN base_neurons b_target ON b_target.neuron_id = v.neuron_id
			-- get neuron level for the voting neuron (to distinguish connection vs pattern)
			JOIN neurons n_voter ON n_voter.id = v.from_neuron_id
			-- find the active neurons (what actually happened just now) - create pattern to predict these
			JOIN active_neurons an ON an.age = 0
			JOIN base_neurons b_actual ON b_actual.neuron_id = an.neuron_id AND b_actual.type = 'event'
			-- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
			JOIN base_neurons b_peak ON b_peak.neuron_id = v.from_neuron_id AND b_peak.type = 'event'
			-- only connection votes (base level peaks)
			WHERE n_voter.level = 0
			-- only event predictions
			AND b_target.type = 'event'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			AND v.strength >= ?
			-- the inference did not come true (prediction error)
			AND v.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
		`, [this.eventErrorMinStrength]);
		if (this.debug) console.log(`Found ${connResult.affectedRows} connection prediction error neurons`);
		return connResult.affectedRows;
	}

	/**
	 * Populate higher-level event prediction errors (from pattern inferences).
	 * Uses inference_votes which already excludes peaks with active patterns.
	 * Peak predicted X, but X didn't happen and Y happened instead.
	 * @returns {Promise<number>} Number of high event pattern futures created
	 */
	async populateNewPatternHighEvents() {
		const [patternResult] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT v.from_neuron_id, an.neuron_id, v.distance
			FROM inference_votes v
			-- get neuron info for the inferred neuron (target)
			JOIN base_neurons b_target ON b_target.neuron_id = v.neuron_id
			-- get neuron level for the voting neuron (to distinguish connection vs pattern)
			JOIN neurons n_voter ON n_voter.id = v.from_neuron_id
			-- find the active neurons (what actually happened just now) - create pattern to predict these
			JOIN active_neurons an ON an.age = 0
			JOIN base_neurons b_actual ON b_actual.neuron_id = an.neuron_id AND b_actual.type = 'event'
			-- only pattern votes (higher level peaks)
			WHERE n_voter.level > 0
			-- only event predictions
			AND b_target.type = 'event'
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			AND v.strength >= ?
			-- the inference did not come true (prediction error)
			AND v.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
		`, [this.eventErrorMinStrength]);
		if (this.debug) console.log(`Found ${patternResult.affectedRows} pattern prediction error neurons`);
		return patternResult.affectedRows;
	}

	/**
	 * Populate action regret neurons into new_pattern_future.
	 * When a connection or pattern predicts an action at distance=1 (full context),
	 * and it leads to pain, add one untried alternative action.
	 * Same logic as addAlternativeActions - just find something different to try.
	 * @returns {Promise<number>} Number of action pattern futures created
	 */
	async populateNewPatternActions() {

		// get the channels that executed painful actions - we will add alternative actions for them
		// if there are no painful channels, we don't need to create any action regret patterns
		const painfulChannelIds = this.getPainfulChannelIds();
		if (painfulChannelIds.length === 0) return 0;

		// process connection action inference regret: find connections that made painful predictions
		// and need to create a new pattern with an alternative action
		const baseCount = await this.populateNewPatternBaseActions(painfulChannelIds);

		// Process pattern inference regret: find patterns that made painful predictions
		// and need to create a higher-level pattern with an alternative action
		const highCount = await this.populateNewPatternHighActions(painfulChannelIds);

		return baseCount + highCount;
	}

	/**
	 * Handle action regret for connection inferences.
	 * Uses inference_votes which already excludes peaks with active patterns.
	 * Find an alternative action to try instead of the one that led to pain.
	 * @param {Array} painfulChannelIds - Channel IDs with negative rewards
	 * @returns {Promise<number>} Number of base action pattern futures created
	 */
	async populateNewPatternBaseActions(painfulChannelIds) {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT v.from_neuron_id, MIN(b_alt.neuron_id) as inferred_neuron_id, v.distance
			FROM inference_votes v
			-- get neuron info for the inferred neuron (target)
			JOIN base_neurons b_target ON b_target.neuron_id = v.neuron_id
			-- get neuron level for the voting neuron (to distinguish connection vs pattern)
			JOIN neurons n_voter ON n_voter.id = v.from_neuron_id
			-- find alternative actions in the same channel (what we should try instead)
			JOIN base_neurons b_alt ON b_alt.channel_id = b_target.channel_id AND b_alt.type = 'action'
			-- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
			JOIN base_neurons b_peak ON b_peak.neuron_id = v.from_neuron_id AND b_peak.type = 'event'
			-- only connection votes (base level peaks)
			WHERE n_voter.level = 0
			-- only action predictions that were executed (winners) in painful channels
			AND b_target.type = 'action'
			AND b_target.channel_id IN (${painfulChannelIds.join(',')})
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			AND v.strength >= ?
			-- check if this was the winning action (need to verify it was executed)
			AND EXISTS (SELECT 1 FROM inferred_neurons inf WHERE inf.neuron_id = v.neuron_id AND inf.is_winner = 1)
			-- don't suggest the same action that just failed
			AND b_alt.neuron_id != v.neuron_id
			GROUP BY v.from_neuron_id, b_target.channel_id, v.distance
		`, [this.actionRegretMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} connection action regret inferences`);
		return result.affectedRows;
	}

	/**
	 * Handle action regret for pattern inferences.
	 * Uses inference_votes which already excludes peaks with active patterns.
	 * Find an alternative action to try instead of the one that led to pain.
	 * @param {Array} painfulChannelIds - Channel IDs with negative rewards
	 * @returns {Promise<number>} Number of high action pattern futures created
	 */
	async populateNewPatternHighActions(painfulChannelIds) {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (peak_neuron_id, inferred_neuron_id, distance)
			SELECT v.from_neuron_id, MIN(b_alt.neuron_id), v.distance
			FROM inference_votes v
			-- get neuron info for the inferred neuron (target)
			JOIN base_neurons b_target ON b_target.neuron_id = v.neuron_id
			-- get neuron level for the voting neuron (to distinguish connection vs pattern)
			JOIN neurons n_voter ON n_voter.id = v.from_neuron_id
			-- find alternative actions in the same channel (what we should try instead)
			JOIN base_neurons b_alt ON b_alt.channel_id = b_target.channel_id AND b_alt.type = 'action'
			-- only pattern votes (higher level peaks)
			WHERE n_voter.level > 0
			-- only action predictions that were executed (winners) in painful channels
			AND b_target.type = 'action'
			AND b_target.channel_id IN (${painfulChannelIds.join(',')})
			-- the inference must have been done with a strength above a threshold to trigger pattern creation
			AND v.strength >= ?
			-- check if this was the winning action (need to verify it was executed)
			AND EXISTS (SELECT 1 FROM inferred_neurons inf WHERE inf.neuron_id = v.neuron_id AND inf.is_winner = 1)
			-- don't suggest the same action that just failed
			AND b_alt.neuron_id != v.neuron_id
			GROUP BY v.from_neuron_id, b_target.channel_id, v.distance
		`, [this.actionRegretMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} pattern action regret inferences`);
		return result.affectedRows;
	}

	/**
	 * Populate new_patterns table from new_pattern_future. Peak can be a base neuron (for connection errors)
	 * or pattern neuron (for pattern errors). One pattern per peak - combines contexts from all ages where
	 * the peak was active and made bad predictions.
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		await this.truncateTables([ 'new_patterns' ]);
		const [insertResult] = await this.conn.query(`
			INSERT INTO new_patterns (peak_neuron_id)
			SELECT DISTINCT peak_neuron_id
			FROM new_pattern_future
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Creates pattern neurons from new_patterns at peak neuron level + 1
	 */
	async createPatternNeurons() {

		// Get peak neurons with their levels from new_patterns
		const [peaks] = await this.conn.query(`
			SELECT np.seq_id, np.peak_neuron_id, n.level as peak_level
			FROM new_patterns np
			JOIN neurons n ON n.id = np.peak_neuron_id
			ORDER BY np.seq_id
		`);

		if (peaks.length === 0) return;

		// Bulk insert pattern neurons - each at its peak neuron's level + 1
		const rows = peaks.map(p => [p.peak_level + 1]);
		const [insertResult] = await this.conn.query('INSERT INTO neurons (level) VALUES ?', [rows]);
		const firstNeuronId = insertResult.insertId;
		const neuronIds = Array.from({ length: peaks.length }, (_, idx) => firstNeuronId + idx);

		// Bulk update new_patterns with pattern neuron IDs using CASE statement
		const caseWhen = peaks.map((p, i) => `WHEN ${p.seq_id} THEN ${neuronIds[i]}`).join(' ');
		await this.conn.query(`UPDATE new_patterns SET pattern_neuron_id = CASE seq_id ${caseWhen} END`);
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future.
	 * Handles both base-level peaks (connection errors) and pattern-level peaks (pattern errors).
	 */
	async createNewPatterns() {

		// create pattern_peaks entries
		await this.conn.query(`
			INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id)
			SELECT pattern_neuron_id, peak_neuron_id
			FROM new_patterns
		`);

		// create pattern_past entries - combine contexts from ALL ages where the peak was active
		// If peak is active at ages 1, 4, 7 with contextLength=10:
		//   Age 1 context: neurons at ages 2-9 (relative ages 1-8)
		//   Age 4 context: neurons at ages 5-9 (relative ages 1-5)
		//   Age 7 context: neurons at ages 8-9 (relative ages 1-2)
		// All these (neuron, relative_age) tuples get combined into ONE pattern
		// the pattern context neurons must be at the same level as the peak neuron (one level lower than the pattern)
		// Get distinct peak ages that were actually voting (from new_pattern_future via inference_votes)
		// distance in new_pattern_future = peak age when it made the bad prediction
		await this.conn.query(`
			INSERT IGNORE INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age)
            SELECT np.pattern_neuron_id, ctx.neuron_id, ctx.age - npf.distance as context_age
			FROM new_patterns np
			JOIN neurons n_peak ON n_peak.id = np.peak_neuron_id
			-- only use ages where the peak was actually voting (not overridden by a pattern)
			JOIN (SELECT DISTINCT peak_neuron_id, distance FROM new_pattern_future) npf ON npf.peak_neuron_id = np.peak_neuron_id
            -- capture same-level active, but older neurons for each voting age
            -- context_age must be 1 to contextLength-1 (relative to peak age)
            JOIN active_neurons ctx ON ctx.age > npf.distance AND ctx.age < npf.distance + ?
            JOIN neurons ctx_n ON ctx_n.id = ctx.neuron_id AND ctx_n.level = n_peak.level
            -- exclude actions from context in base level
            -- note that this may be a stock specific thing - actions can learn patterns in robotics context
            WHERE NOT EXISTS (SELECT 1 FROM base_neurons b WHERE b.neuron_id = ctx.neuron_id AND b.type = 'action')
		`, [this.contextLength]);

		// Create pattern_future entries from new_pattern_future for cross-channel inferences
		// inferred_neuron_id is always a base-level neuron (event or action)
		// Join on peak_neuron_id to get all future inferences for this peak
		const [futureResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance)
			SELECT np.pattern_neuron_id, npf.inferred_neuron_id, npf.distance
			FROM new_patterns np
			JOIN new_pattern_future npf ON npf.peak_neuron_id = np.peak_neuron_id
		`);
		if (this.debug) console.log(`Created ${futureResult.affectedRows} pattern_future entries`);
	}

	/**
	 * Activate newly created pattern neurons at ALL ages where the peak was active.
	 * If peak was active at ages 1, 4, 7, the pattern is activated at ages 1, 4, and 7.
	 * This allows the pattern to be refined in the same frame if there are multiple errors,
	 * and ensures the "already has active pattern" check works correctly in future frames.
	 */
	async activateNewPatterns() {
		const [result] = await this.conn.query(`
			INSERT INTO active_neurons (neuron_id, age)
			SELECT DISTINCT np.pattern_neuron_id, npf.distance
			FROM new_patterns np
			JOIN new_pattern_future npf ON npf.peak_neuron_id = np.peak_neuron_id
		`);
		if (this.debug) console.log(`Activated ${result.affectedRows} new pattern neuron instances`);
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 */
	async inferNeurons() {

		// Collect votes and determine consensus (all in SQL)
		const inferences = await this.collectVotes();

		// If no inferences, wait for more data
		if (inferences.length === 0) {
			if (this.debug) console.log('No inferences found. Waiting for more data in future frames.');
			return;
		}

		// Save all inferences
		await this.saveInferences(inferences);

		// Notify channels about winning event predictions for continuous tracking (e.g., price prediction)
		this.notifyChannelsOfEventPredictions(inferences);
	}

	/**
	 * Collect votes and determine consensus in SQL.
	 * Pattern votes override their peak's connection votes.
	 * Aggregates by neuron, picks winners per dimension (highest strength for events, highest reward for actions).
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async collectVotes() {
		const timeDecay = 1 / this.contextLength;
		await this.conn.query('TRUNCATE inference_votes');

		// Insert connection votes (from base level neurons)
		const [connResult] = await this.conn.query(`
			INSERT INTO inference_votes (from_neuron_id, neuron_id, strength, reward, distance)
			SELECT c.from_neuron_id, c.to_neuron_id,
			       (1 + n.level * ?) * (1 - (c.distance - 1) * ?) * c.strength as effective_strength,
			       c.reward, c.distance
			FROM active_neurons an
			JOIN neurons n ON n.id = an.neuron_id AND n.level = 0
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			WHERE c.distance = an.age + 1 AND c.strength > 0
		`,[this.levelVoteMultiplier, timeDecay]);

		// Insert pattern votes (from pattern neurons at any level)
		const [patternResult] = await this.conn.query(`
			INSERT INTO inference_votes (from_neuron_id, neuron_id, strength, reward, distance)
			SELECT pf.pattern_neuron_id, pf.inferred_neuron_id,
			       (1 + pn.level * ?) * (1 - (pf.distance - 1) * ?) * pf.strength as effective_strength,
			       pf.reward, pf.distance
			FROM active_neurons an
			JOIN neurons pn ON pn.id = an.neuron_id AND pn.level > 0
			JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
			WHERE pf.distance = an.age + 1 AND pf.strength > 0
		`,[this.levelVoteMultiplier, timeDecay]);

		if (this.debug) console.log(`Inserted ${connResult.affectedRows} connection votes, ${patternResult.affectedRows} pattern votes`);

		// Delete overridden votes: votes from neurons that are peaks of other voting patterns
		const [deleteResult] = await this.conn.query(`
			DELETE v FROM inference_votes v
			JOIN pattern_peaks pp ON pp.peak_neuron_id = v.from_neuron_id
			JOIN inference_votes pv ON pv.from_neuron_id = pp.pattern_neuron_id
		`);

		if (this.debug) console.log(`Deleted ${deleteResult.affectedRows} overridden votes`);

		// Aggregate votes by neuron and determine winners per dimension
		const [inferences] = await this.conn.query(`
			-- aggregate votes for each candidate neuron
            WITH candidates AS (
            	SELECT v.neuron_id, SUM(v.strength) as strength, SUM(v.strength * v.reward) / SUM(v.strength) as reward,
					   b.type, b.channel_id, ch.name as channel,
					   GROUP_CONCAT(CONCAT(d.name, '|', coord.val) ORDER BY d.name SEPARATOR ',') as coordinates
				FROM inference_votes v
                JOIN base_neurons b ON b.neuron_id = v.neuron_id
                JOIN channels ch ON ch.id = b.channel_id
                JOIN coordinates coord ON coord.neuron_id = v.neuron_id
                JOIN dimensions d ON d.id = coord.dimension_id
                GROUP BY v.neuron_id, b.type, b.channel_id, ch.name
            ),
            -- join to coordinates and rank within each dimension (events by strength, actions by reward)
            ranked AS (
            	SELECT c.neuron_id, coord.dimension_id,
					RANK() OVER (PARTITION BY coord.dimension_id ORDER BY IF(c.type = 'action', c.reward, c.strength) DESC) as dim_rank
				FROM candidates c
                JOIN coordinates coord ON coord.neuron_id = c.neuron_id
			),
			-- collect neurons that are rank 1 in any dimension
            winners AS (
            	SELECT DISTINCT neuron_id
                FROM ranked
                WHERE dim_rank = 1
            )
            -- join candidates with winners to set is_winner flag
            SELECT c.*, IF(w.neuron_id IS NOT NULL, 1, 0) as is_winner
            FROM candidates c
            LEFT JOIN winners w ON w.neuron_id = c.neuron_id
		`);

		if (this.debug) console.log(`Collected ${inferences.length} inferences after consensus`);

		// Parse coordinates and convert is_winner to boolean
		return inferences.map(inf => ({
			...inf,
			coordinates: this.parseCoordinates(inf.coordinates),
			isWinner: inf.is_winner === 1
		}));
	}

	/**
	 * Parse coordinates string from GROUP_CONCAT into object.
	 * Input format: "dim1|val1,dim2|val2,..."
	 * Output format: { dim1: val1, dim2: val2, ... }
	 * @param {string} coordStr - Coordinates string from SQL
	 * @returns {Object} Coordinates object with dimension names as keys
	 */
	parseCoordinates(coordStr) {
		if (!coordStr) return {};
		const coords = {};
		for (const pair of coordStr.split(',')) {
			const [dim, val] = pair.split('|');
			coords[dim] = parseFloat(val);
		}
		return coords;
	}

	/**
	 * Apply exploration to a channel's inferred actions.
	 * Returns inferred actions unchanged if any exist, otherwise returns an exploration action.
	 * @param {string} channelName - Channel name
	 * @param {Array} inferredActions - Array of inferred action coordinates for this channel
	 * @returns {Array} - Action coordinates to use (either inferred or exploration)
	 */
	applyExploration(channelName, inferredActions) {
		if (inferredActions.length > 0)
			return inferredActions;

		// No inference - pick first action from channelActions
		const allActions = this.channelActions.get(channelName);
		if (!allActions || allActions.size === 0)
			return [];

		const firstAction = allActions.values().next().value;
		if (this.debug) console.log(`Exploration for ${channelName}:`, firstAction.coordinates);
		return [firstAction.coordinates];
	}

	/**
	 * Save all inferences in one operation.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	async saveInferences(inferences) {
		if (inferences.length === 0) return;

		// Collect neurons - is_winner: 1 for winners (highest sum of strength * reward per dimension), 0 for losers
		const neurons = [];
		for (const inf of inferences) neurons.push([inf.neuron_id, inf.strength, inf.isWinner ? 1 : 0]);

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
		const [patternPastUpdateResult] = await this.conn.query(`UPDATE pattern_past SET strength = GREATEST(?, strength - ?) WHERE strength > 0`, [this.minConnectionStrength, this.patternForgetRate]);
		if (this.debug) console.log(`  Pattern_past UPDATE took ${Date.now() - stepStart}ms (updated ${patternPastUpdateResult.affectedRows} rows)`);

		if (this.debug) console.log('Running forget cycle - pattern_future update...');
		stepStart = Date.now();
		const [patternFutureUpdateResult] = await this.conn.query(`UPDATE pattern_future SET strength = GREATEST(?, strength - ?) WHERE strength > 0`, [this.minConnectionStrength, this.patternForgetRate]);
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
		const [connectionUpdateResult] = await this.conn.query(`UPDATE connections SET strength = GREATEST(?, strength - ?) WHERE strength > 0`, [this.minConnectionStrength, this.connectionForgetRate]);
		if (this.debug) console.log(`  Connection UPDATE took ${Date.now() - stepStart}ms (updated ${connectionUpdateResult.affectedRows} rows)`);

		// Delete connections with zero strength
		if (this.debug) console.log('Running forget cycle - connection deletion...');
		stepStart = Date.now();
		const [connectionDeleteResult] = await this.conn.query(`DELETE FROM connections WHERE strength = ?`, [this.minConnectionStrength]);
		if (this.debug) console.log(`  Connection DELETE took ${Date.now() - stepStart}ms (deleted ${connectionDeleteResult.affectedRows} rows)`);

		// 3. PATTERN NEURON CLEANUP: Remove orphaned pattern neurons (level > 0) with no connections or pattern entries
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
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {

		// Get base level (level 0) accuracy
		let baseAccuracy = 'N/A';
		if (this.accuracyStats.total > 0)
			baseAccuracy = `${(this.accuracyStats.correct / this.accuracyStats.total * 100).toFixed(1)}%`;

		// Get average action reward
		let avgReward = 'N/A';
		if (this.rewardStats.count > 0)
			avgReward = `${(this.rewardStats.totalReward / this.rewardStats.count).toFixed(3)} (${this.rewardStats.count})`;

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

		if (this.frameSummary) console.log(`Frame ${this.frameNumber} | Accuracy: ${baseAccuracy} | Reward: ${avgReward} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}

}