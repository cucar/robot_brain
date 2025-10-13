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
		this.baseNeuronMaxAge = 10; // number of frames a base neuron stays active
		this.forgetCycles = 10; // number of frames between forget cycles
		this.connectionForgetRate = 0.1; // how much connection strengths decay per forget cycle
		this.patternForgetRate = 0.1; // how much pattern strengths decay per forget cycle
		this.rewardForgetRate = 0.05; // how much reward factors decay toward neutral (1.0) per forget cycle
		this.negativeLearningRate = 0.1; // how much pattern strengths will be decremented by when not accurate
		this.maxLevels = 6; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.minPeakStrength = 10.0; // minimum weighted strength for a neuron to be considered a peak (pattern)
		this.minPeakRatio = 1.0; // minimum ratio of peak strength to neighborhood average to be considered a peak (pattern)

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();
		
		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;
		this.inactivityThreshold = 5; // frames of inactivity before exploration

		// Create readline interface for pausing between frames - used when debugging
		this.debug = true;
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
			'pattern_inference',
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
			'pattern_inference',
			'connection_inference',
			'inferred_neurons',
			'observed_patterns',
			'active_connections',
			'patterns',
			'connections',
			'coordinates',
			'neuron_rewards',
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
	 * Ages all neurons in the context by 1, then deactivates aged-out neurons
	 * Processes levels in natural order (0 → maxLevel) so lower levels are cleaned up first
	 */
	async ageNeurons() {
		console.log('Aging active neurons and inferred neurons...');

		// age all neurons in the context - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');

		// get max levels independently for active and inferred neurons
		const [activeRows] = await this.conn.query('SELECT MAX(level) as max_level FROM active_neurons');
		const [inferredRows] = await this.conn.query('SELECT MAX(level) as max_level FROM inferred_neurons');
		const maxActiveLevel = activeRows[0].max_level || 0;
		const maxInferredLevel = inferredRows[0].max_level || 0;

		// deactivate old active neurons in natural order (0 → maxActiveLevel)
		for (let level = 0; level <= maxActiveLevel; level++)
			await this.deactivateOldNeurons(level);

		// deactivate old inferred neurons in natural order (0 → maxInferredLevel)
		for (let level = 0; level <= maxInferredLevel; level++)
			await this.deactivateOldInferences(level);
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
	 * infers predictions and outputs starting from the highest active level down to base level (reverse order)
	 */
	async inferNeurons() {

		// get the highest level that is currently active - that's where we will start from
		const maxLevel = await this.getMaxActiveLevel();

		// process levels in reverse: maxLevel, maxLevel-1, ..., 0
		for (let level = maxLevel; level >= 0; level--) {

			// do same-level connection predictions for the level
			await this.inferConnections(level);

			// do the pattern predictions for the lower level - these will be used in the next iteration
			if (level > 0) await this.inferPatterns(level);

			// get predicted connections for the level
			const predictedConnections = await this.getPredictedConnections(level);

			// skip if no predicted connections
			if (predictedConnections.length === 0) {
				console.log(`No predicted connections for level ${level}, skipping inference`);
				continue;
			}

			// determine predicted neuron strengths from connections
			let neuronStrengths = this.getNeuronStrengths(predictedConnections);
			if (neuronStrengths.size === 0) throw new Error('no neuron strengths.'); // should not happen
			console.log(`Calculated strengths for ${neuronStrengths.size} neurons at level ${level}`);

			// apply reward optimizations to neuron strengths
			neuronStrengths = await this.optimizeRewards(neuronStrengths, level);

			// determine peak neurons for the level using peak detection algorithm
			await this.inferPeakNeurons(neuronStrengths, level, predictedConnections);
		}
	}

	/**
	 * Infer same-level connections for predictions.
	 * Predicts ALL future connections from active neurons, weighted by distance to age=-1.
	 * No negative reinforcement - relies on forget cycles for cleanup.
	 * Rebuilds fresh each frame based on current context.
	 */
	async inferConnections(level) {

		// Clear previous predictions for this level
		await this.conn.query('DELETE FROM connection_inference WHERE level = ?', [level]);

		// Insert ALL future connections from active neurons at this level
		// weight_distance = distance to age=-1 = c.distance - (f.age + 1)
		// Only include connections where c.distance > f.age + 1 (future connections)
		await this.conn.query(`
			INSERT INTO connection_inference (level, connection_id, weight_distance)
			SELECT :level, c.id, c.distance - (f.age + 1) as weight_distance
			FROM active_neurons f
			JOIN connections c ON c.from_neuron_id = f.neuron_id
			WHERE f.level = :level
			AND c.distance > f.age + 1
			AND c.strength > 0
		`, { level });
	}

	/**
	 * Infer lower-level connections from active patterns.
	 * Predicts ALL pattern connections that aren't already active, weighted by distance to age=-1.
	 * No negative reinforcement - relies on forget cycles for cleanup.
	 * Rebuilds fresh each frame based on current context.
	 */
	async inferPatterns(level) {
		console.log(`Processing lower-level predictions for level ${level}`);

		// Clear previous pattern predictions for the lower level
		await this.conn.query('DELETE FROM pattern_inference WHERE level = ?', [level - 1]);

		// Insert ALL pattern connections from active patterns at this level
		// that are not already active in the lower level
		// weight_distance = distance to age=-1 = c.distance - (f.age + 1)
		// where f is the source neuron of the pattern's connection
		await this.conn.query(`
			INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, weight_distance)
			SELECT :targetLevel, p.pattern_neuron_id, p.connection_id, c.distance - (f.age + 1) as weight_distance
			FROM active_neurons an
			JOIN patterns p ON an.neuron_id = p.pattern_neuron_id
			JOIN connections c ON p.connection_id = c.id
			JOIN active_neurons f ON c.from_neuron_id = f.neuron_id
			WHERE an.level = :level
			AND p.strength > 0
			AND c.distance > f.age + 1
			AND NOT EXISTS (
				SELECT 1
				FROM active_connections ac
				WHERE ac.connection_id = p.connection_id
				AND ac.level = :targetLevel
			)
		`, { level, targetLevel: level - 1 });
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
	 * bulk insert neurons and return their IDs
	 */
	async bulkInsertNeurons(count) {

		// bulk insert neurons - mysql returns the first insert id - ids are continuous within the statement in the session
		const valuesSql = Array(count).fill('()').join(',');
		const insertNeuronsResult = await this.conn.query(`INSERT INTO neurons () VALUES ${valuesSql}`);
		const firstNeuronId = insertNeuronsResult[0].insertId;

		// compute ids from the first insert id (assume auto_increment step = 1)
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * reinforces the connections between newly active neurons (age = 0) at a level
	 * builds connections FROM all active neurons (adjacent levels only) TO age=0 neurons at the specified level
	 * Adjacent levels: same level, one level up, one level down (like cortical columns)
	 * Distance calculation uses exponential rounding based on source neuron age:
	 * - Ages 1-N: exact distance (1, 2, 3, ..., N-1)
	 * - Ages N to N²-1: rounded to multiples of N (N, 2N, 3N, ...)
	 * - Ages N² to N³-1: rounded to multiples of N² (N², 2N², 3N², ...)
	 * This allows neurons to remember relative time distances with decreasing precision
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT
                f.neuron_id as from_neuron_id,
                t.neuron_id as to_neuron_id,
                IF(f.age = 0, 0, FLOOR(f.age / POW(:N, FLOOR(LOG(:N, f.age)))) * POW(:N, FLOOR(LOG(:N, f.age)))) as distance,
                1 as strength
			FROM active_neurons f
			CROSS JOIN active_neurons t
            WHERE t.age = 0  -- target neurons are newly activated
            AND t.level = :level  -- target neurons are at the specified level
            AND ABS(f.level - t.level) <= 1  -- restrict to adjacent levels only (same, +1, -1)
            AND (t.neuron_id != f.neuron_id OR f.age != 0)  -- no self-connections at same age
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)
		`, { N: this.baseNeuronMaxAge, level });
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
	 * processes a level to detect patterns and activate them - returns if patterns found or not
	 */
	async activateLevelPatterns(level) {
		console.log(`processing level to activate patterns in it: ${level}`);

		// get all active connections between the newly activated neurons (age=0) and all active neurons in the requested level
		const connections = await this.getActiveConnections(level);
		console.log(`Found ${connections.length} active connections for level ${level}`);
		if (connections.length === 0) return false; // if there are no connections (there is only one neuron), nothing to do

		// cluster connections around peaks
		const peakConnections = await this.detectPeaks(connections);
		if (peakConnections.size === 0) return false; // if there are no peaks found in the level, no patterns to process - return false to indicate that we're done

		// save observed patterns (peak connections) to the observed_patterns table
		await this.saveObservedPatterns(peakConnections);

		// match observed patterns to known patterns using their connection ids. get all pattern neuron ids that use the connection ids.
		const peakPatterns = await this.matchPatternNeurons(peakConnections);

		// augment known patterns that were matched to observed patterns with the
		await this.mergeMatchedPatterns(peakPatterns, peakConnections);

		// create new patterns for the peaks that do not have any matching patterns
		await this.createNewPatterns(peakPatterns, peakConnections);

		// now strengthen the connections for each peak's pattern and its active connections that were just observed (increment by 1)
		await this.reinforcePatterns(peakPatterns, peakConnections);

		// activate the observed pattern neurons in the higher level (new or previously existing)
		await this.activateNeurons([...new Set(Array.from(peakPatterns.values()).flat())], level + 1);

		// return true to indicate that we need to process the next level
		return true;
	}

	/**
	 * returns active directed connections flowing INTO newly activated neurons (age=0) from all active neurons for spatio-temporal pooling
	 * Uses the active_connections table which is populated incrementally in activateConnections()
	 * connections are directed: older neurons → newer neurons. note that this includes connections between the age=0 neurons as well. that's the spatial pooling.
	 * the others are temporal. so, these connections form the basis of the spatio-temporal pooling. there may be connections between the same neuron when their ages
	 * are different. connections can span across levels, enabling multi-scale pattern recognition.
	 * there should not be any connections from the same neuron to itself within the same age and level.
	 */
	async getActiveConnections(level) {
		const [rows] = await this.conn.query(`
            SELECT
                ac.connection_id as id,
                ac.from_neuron_id,
                ac.to_neuron_id,
                c.distance,
                c.strength
            FROM active_connections ac
            JOIN connections c ON ac.connection_id = c.id
            WHERE ac.level = ?
            AND c.strength > 0  -- ignore connections that are scheduled to be deleted
		`, [level]);
		return rows;
	}

	/**
	 * Populate active_connections table for newly activated neurons at the specified level
	 * This is called immediately after reinforceConnections in activateNeurons
	 * Inserts connections from all active neurons to age=0 neurons at the specified level
	 * Uses exponential rounding for distance matching (same as reinforceConnections)
	 */
	async activateConnections(level) {
		await this.conn.query(`
			INSERT INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level)
			SELECT c.id as connection_id, c.from_neuron_id, c.to_neuron_id, t.level
			FROM connections c
			JOIN active_neurons f ON c.from_neuron_id = f.neuron_id
			JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = :level
			WHERE c.distance = IF(f.age = 0, 0, FLOOR(f.age / POW(:N, FLOOR(LOG(:N, f.age)))) * POW(:N, FLOOR(LOG(:N, f.age))))
			AND ABS(f.level - t.level) <= 1  -- restrict to adjacent levels only (same, +1, -1)
			AND (t.neuron_id != f.neuron_id OR f.age != 0)  -- no self-connections at same age
			AND c.strength > 0  -- only connections that are not removed
		`, { N: this.baseNeuronMaxAge, level });
	}

	/**
	 * Clean up active_connections when neurons are removed from active_neurons
	 * Removes connections where either the from_neuron or to_neuron is no longer active
	 * This should be called after neurons are aged out or removed from active_neurons
	 *
	 * @param {number} level - The level to clean up connections for (required)
	 */
	async cleanupActiveConnections(level) {
		await this.conn.query(`
			DELETE FROM active_connections
			WHERE level = ?
			AND (
				NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = from_neuron_id)
				OR NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = to_neuron_id)
			)
		`, [level]);
	}

	/**
	 * Deactivate (remove) old neurons from active_neurons at the specified level
	 * This implements variable time dilation:
	 * - Base level (0): Remove neurons when age >= baseNeuronMaxAge
	 * - Higher levels: Remove neurons when they have no active connections in the level
	 *
	 * Called from ageNeurons() in natural order (0 → maxLevel) after aging.
	 * Also cleans up orphaned active_connections.
	 *
	 * @param {number} level - The level to deactivate old neurons from (required)
	 */
	async deactivateOldNeurons(level) {

		// Base level: simple age-based cleanup
		if (level === 0) {
			const [result] = await this.conn.query('DELETE FROM active_neurons WHERE level = 0 AND age >= ?', [this.baseNeuronMaxAge]);
			console.log(`Deactivated ${result.affectedRows} aged-out neurons at level 0 (age >= ${this.baseNeuronMaxAge})`);
			await this.cleanupActiveConnections(level); // clean up any orphaned active connections
			return;
		}

		// Higher levels: remove pattern neurons whose defining connections are no longer active
		// Variable time dilation - patterns stay active as long as at least one of their connections is active
		const [result] = await this.conn.query(`
			DELETE FROM active_neurons
			WHERE level = ?
			AND NOT EXISTS (
				SELECT 1 FROM patterns p
				JOIN active_connections ac ON p.connection_id = ac.connection_id
				WHERE p.pattern_neuron_id = active_neurons.neuron_id
				AND ac.level = ?  -- connections are at the level below the pattern
			)
		`, [level, level - 1]);
		console.log(`Deactivated ${result.affectedRows} pattern neurons at level ${level} (no active connections in level ${level - 1})`);

		// Clean up orphaned active connections for this level
		await this.cleanupActiveConnections(level);
	}

	/**
	 * Deactivate (remove) old inferred neurons
	 * Parallel structure to deactivateOldNeurons:
	 * - Base level (0): Remove when age >= baseNeuronMaxAge
	 * - Higher levels: Remove when pattern has no connections in inference tables
	 *
	 * Called from ageNeurons() in natural order (0 → maxLevel) after aging.
	 *
	 * @param {number} level - The level to deactivate old inferences from (required)
	 */
	async deactivateOldInferences(level) {

		// Base level: simple age-based cleanup
		if (level === 0) {
			const [result] = await this.conn.query(`
				DELETE FROM inferred_neurons
				WHERE level = 0 AND age >= ?
			`, [this.baseNeuronMaxAge]);
			console.log(`Deactivated ${result.affectedRows} aged-out inferred neurons at level 0 (age >= ${this.baseNeuronMaxAge})`);
			return;
		}

		// Higher levels: remove pattern neurons whose defining connections are not in inference tables
		// A pattern stays inferred as long as at least one of its connections is still being predicted
		const [result] = await this.conn.query(`
			DELETE FROM inferred_neurons
			WHERE level = ?
			AND NOT EXISTS (
				SELECT 1 FROM patterns p
				WHERE p.pattern_neuron_id = inferred_neurons.neuron_id
				AND (
					EXISTS (
						SELECT 1 FROM connection_inference ci
						WHERE ci.connection_id = p.connection_id
						AND ci.level = ?
					)
					OR EXISTS (
						SELECT 1 FROM pattern_inference pi
						WHERE pi.connection_id = p.connection_id
						AND pi.level = ?
					)
				)
			)
		`, [level, level - 1, level - 1]);
		console.log(`Deactivated ${result.affectedRows} inferred pattern neurons at level ${level} (no connections in inference tables at level ${level - 1})`);
	}

	/**
	 * Generic peak detection that works for both active and predicted connections
	 * @param {Array} connections - Array of connection objects with strength property
	 * @returns {Map} Map of peak neuron IDs to their connection IDs
	 */
	async detectPeaks(connections) {

		// get the neuron strengths from the active connections (both ways - from and to)
		const neuronStrengths = this.getNeuronStrengths(connections);

		// get the neighborhood map for each neuron from its connections
		const neighborhoodMap = this.buildNeighborhoodMap(connections);

		// calculate the neighborhood average strength of each neuron. for each neuron, we need to calculate the strengths of its connected neurons.
		const neighborhoodStrengths = this.getNeighborhoodStrengths(neighborhoodMap, neuronStrengths);

		// now get the neuron ids whose strength exceeds its neighborhood - those are the peaks
		const peakNeuronIds = this.getPeakNeurons(neuronStrengths, neighborhoodStrengths);
		if (peakNeuronIds.length > 0 && this.debug) await this.waitForUser(`
			Found ${peakNeuronIds.length} peaks: [${peakNeuronIds.join(', ')}],
			Neuron Strengths: ${JSON.stringify(neuronStrengths)}, 
			Neighborhood Strengths: ${JSON.stringify(neighborhoodStrengths)}
		`);

		// get a map from peak neuron ids to the connection ids it uses and return them
		return this.getPeakNeuronsConnections(peakNeuronIds, neighborhoodMap, connections);
	}

	/**
	 * Returns the strength of each neuron from the sum of its incoming connections.
	 * Only counts destination neurons (to_neuron_id) - these are the neurons being activated.
	 * Applies linear distance weighting: closer connections (distance=0) have full weight (1.0),
	 * distant connections (distance=baseNeuronMaxAge-1) have minimal weight (0.1 if baseNeuronMaxAge=10).
	 */
	getNeuronStrengths(connections) {
		const neuronStrengths = new Map();

		for (const connection of connections) {
			const { to_neuron_id, strength, distance } = connection;

			// calculate linear distance weight: distance=0 → weight=1.0, distance=9 → weight=0.1
			// simple linear interpolation: (baseNeuronMaxAge - distance) / baseNeuronMaxAge
			const weight = (this.baseNeuronMaxAge - distance) / this.baseNeuronMaxAge;
			const weightedStrength = strength * weight;

			// add weighted strength for the destination neuron (incoming connection)
			if (neuronStrengths.has(to_neuron_id))
				neuronStrengths.set(to_neuron_id, neuronStrengths.get(to_neuron_id) + weightedStrength);
			else
				neuronStrengths.set(to_neuron_id, weightedStrength);
		}
		return neuronStrengths;
	}

	/**
	 * returns average strengths for each neuron's neighborhood (both incoming and outgoing connections)
	 */
	getNeighborhoodStrengths(neighborhoodMap, neuronStrengths) {
		const neighborhoodAverages = new Map();
		for (const [neuronId, neighbors] of neighborhoodMap) {
			let totalStrength = 0;
			let count = 0;

			for (const neighborId of neighbors) {
				const neighborStrength = neuronStrengths.get(neighborId) || 0;
				totalStrength += neighborStrength;
				count++;
			}

			const averageStrength = count > 0 ? (totalStrength / count) : 0;
			neighborhoodAverages.set(neuronId, averageStrength);
		}

		return neighborhoodAverages;
	}

	/**
	 * build a map of each neuron and its connected neighbors (bidirectional)
	 */
	buildNeighborhoodMap(connections) {
		const neighborhoodMap = new Map();
		for (const connection of connections) {
			const { from_neuron_id, to_neuron_id } = connection;

			// add to_neuron as neighbor of from_neuron (outgoing connection)
			if (!neighborhoodMap.has(from_neuron_id)) neighborhoodMap.set(from_neuron_id, new Set());
			neighborhoodMap.get(from_neuron_id).add(to_neuron_id);

			// add from_neuron as neighbor of to_neuron (incoming connection)
			if (!neighborhoodMap.has(to_neuron_id)) neighborhoodMap.set(to_neuron_id, new Set());
			neighborhoodMap.get(to_neuron_id).add(from_neuron_id);
		}
		return neighborhoodMap;
	}

	/**
	 * get neurons whose strength exceeds their neighborhood average strength (peaks).
	 * these are neurons that are stronger than their local neighborhood average.
	 * also requires minimum relative/absolute strength to avoid creating patterns from weak connections.
	 */
	getPeakNeurons(neuronStrengths, neighborhoodStrengths) {
		const peaks = [];
		for (const [neuronId, neuronStrength] of neuronStrengths) {
			const neighborhoodStrength = neighborhoodStrengths.get(neuronId) || 0;
			if (neuronStrength >= this.minPeakStrength && (neuronStrength / neighborhoodStrength) > this.minPeakRatio) peaks.push(neuronId);
		}
		return peaks;
	}

	/**
	 * returns a map of peak neurons to their connection IDs. for each peak neuron, returns all connection IDs where the neuron
	 * appears either as source or target using the neighborhoodMap.
	 */
	getPeakNeuronsConnections(peakNeuronIds, neighborhoodMap, connections) {
		const peakConnections = new Map();

		for (const peakNeuronId of peakNeuronIds) {
			const neighbors = neighborhoodMap.get(peakNeuronId);
			if (!neighbors) {
				peakConnections.set(peakNeuronId, []);
				continue;
			}

			const connectionIds = this.getPeakNeuronConnections(peakNeuronId, neighbors, connections);
			peakConnections.set(peakNeuronId, connectionIds);
		}

		return peakConnections;
	}

	/**
	 * returns the connection IDs for a given peak neuron and its neighbors
	 */
	getPeakNeuronConnections(peakNeuronId, neighbors, connections) {
		const connectionIds = [];
		for (const neighborId of neighbors) {

			// find connection from peak to neighbor
			const outgoingConnection = connections.find(c => c.from_neuron_id === peakNeuronId && c.to_neuron_id === neighborId);
			if (outgoingConnection) connectionIds.push(outgoingConnection.id);

			// find connection from neighbor to peak
			const incomingConnection = connections.find(c => c.from_neuron_id === neighborId && c.to_neuron_id === peakNeuronId);
			if (incomingConnection) connectionIds.push(incomingConnection.id);
		}

		// Deduplicate connection IDs (can have duplicates with cross-level connections)
		return [...new Set(connectionIds)];
	}

	/**
	 * saves observed patterns (peak connections) to the observed_patterns table
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async saveObservedPatterns(peakConnections) {
		const peakConnectionMappings = [];
		for (const [peakNeuronId, connectionIds] of peakConnections)
			for (const connectionId of connectionIds)
				peakConnectionMappings.push([peakNeuronId, connectionId]);

		// Clear and populate memory table for peak-connection mappings
		await this.conn.query('TRUNCATE observed_patterns');
		await this.conn.query('INSERT INTO observed_patterns VALUES ?', [peakConnectionMappings]);
	}

	/**
	 * matches observed patterns to known patterns based on neuron overlap (66% threshold)
	 * uses the observed_patterns table populated by saveObservedPatterns
	 * output format: Map(peak_neuron_id, pattern_neuron_ids)
	 */
	async matchPatternNeurons(peakConnections) {

		// Single efficient query that processes all peaks at once
		const [rows] = await this.conn.query(`
			WITH observed_neurons AS (
				-- All unique neurons for each peak from their observed connections
			    -- Note that these are not the same thing as age=0 neurons - they include all neurons that are connected to them as well
				SELECT DISTINCT op.peak_neuron_id, IF(v.direction = 'from', c.from_neuron_id, c.to_neuron_id) as neuron_id
				FROM observed_patterns op
				JOIN connections c ON op.connection_id = c.id
				CROSS JOIN (SELECT 'from' as direction UNION SELECT 'to' as direction) AS v
			),
			observed_pattern_matches AS (
				-- Get patterns that share connections with observed patterns (pattern-peak pairs)
				SELECT DISTINCT p.pattern_neuron_id, op.peak_neuron_id
				FROM patterns p 
				JOIN observed_patterns op ON p.connection_id = op.connection_id 
				WHERE p.strength > 0
			),
			candidate_pattern_neurons AS (
				-- Get all unique neurons for each candidate pattern (once per pattern, not duplicated per peak)
				SELECT DISTINCT p.pattern_neuron_id, IF(v.direction = 'from', c.from_neuron_id, c.to_neuron_id) as neuron_id
				FROM patterns p
				JOIN connections c ON p.connection_id = c.id
				CROSS JOIN (SELECT 'from' as direction UNION SELECT 'to' as direction) AS v
				WHERE p.pattern_neuron_id IN (SELECT DISTINCT pattern_neuron_id FROM observed_pattern_matches) 
				AND p.strength > 0
			)
			-- Calculate overlap percentage and return matching peak-pattern pairs
			-- at least 66% of the known pattern's neurons should be part of the observed pattern to be matched
			SELECT opm.peak_neuron_id, opm.pattern_neuron_id
			FROM observed_pattern_matches opm
			JOIN candidate_pattern_neurons cpn ON opm.pattern_neuron_id = cpn.pattern_neuron_id
			LEFT JOIN observed_neurons obs ON opm.peak_neuron_id = obs.peak_neuron_id AND obs.neuron_id = cpn.neuron_id
			GROUP BY opm.peak_neuron_id, opm.pattern_neuron_id
			HAVING (COUNT(DISTINCT CASE WHEN obs.neuron_id IS NOT NULL THEN cpn.neuron_id END) / COUNT(DISTINCT cpn.neuron_id)) >= ?
		`, [this.mergePatternThreshold]);

		// Convert to expected format
		const peakPatterns = new Map();
		
		// Initialize all peaks with empty arrays
		for (const [peakNeuronId] of peakConnections) peakPatterns.set(peakNeuronId, []);
		
		// Fill in matched patterns
		for (const row of rows) {
			const peakNeuronId = parseInt(row.peak_neuron_id);
			const patternNeuronId = parseInt(row.pattern_neuron_id);
			
			// Get existing array or create new one
			if (!peakPatterns.has(peakNeuronId)) peakPatterns.set(peakNeuronId, []);
			
			// Add the pattern to this peak's array
			peakPatterns.get(peakNeuronId).push(patternNeuronId);
		}
		
		return peakPatterns;
	}

	/**
	 * merges observed patterns into existing matched pattern definitions
	 * @param {Map} peakPatterns - Map of peak_neuron_id -> array of pattern_neuron_ids
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async mergeMatchedPatterns(peakPatterns, peakConnections) {
		
		// Create reverse mapping: pattern_neuron_id -> array of peak_neuron_ids
		const patternPeaks = new Map();
		for (const [peakNeuronId, patternNeuronIds] of peakPatterns) {
			for (const patternNeuronId of patternNeuronIds) {
				if (!patternPeaks.has(patternNeuronId)) patternPeaks.set(patternNeuronId, []);
				patternPeaks.get(patternNeuronId).push(peakNeuronId);
			}
		}

		// For each matched pattern, collect all connection IDs from the observed patterns
		const patternConnectionUpdates = [];
		for (const [patternNeuronId, peakNeuronIds] of patternPeaks) {

			// Get all connection IDs from the matched observed patterns
			const connectionIds = new Set();
			for (const peakNeuronId of peakNeuronIds) {
				const peakConnections_for_peak = peakConnections.get(peakNeuronId) || [];
				for (const connectionId of peakConnections_for_peak) connectionIds.add(connectionId);
			}

			// Add each connection to the pattern with strength increment of 1
			for (const connectionId of connectionIds)
				patternConnectionUpdates.push([patternNeuronId, connectionId, 1]);
		}

		// Batch insert/update all pattern-connection relationships
		if (patternConnectionUpdates.length > 0) {
			await this.conn.query(`
				INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES ?
				ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)
			`, [patternConnectionUpdates]);
		}
	}

	/**
	 * create new pattern neurons for peaks that don't have any matching patterns.
	 * updates the peakPatterns map to include the newly created pattern neurons.
	 * @param {Map} peakPatterns - Map of peak_neuron_id -> array of pattern_neuron_ids
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async createNewPatterns(peakPatterns, peakConnections) {

		// find peaks that need new patterns
		const peaksNeedingPatterns = [];
		for (const [peakNeuronId, connectionIds] of peakConnections) {
			const existingPatterns = peakPatterns.get(peakNeuronId) || [];
			if (existingPatterns.length === 0 && connectionIds.length > 0) peaksNeedingPatterns.push({ peakNeuronId, connectionIds });
		}

		// if all peaks have patterns, nothing to do - no new patterns to create
		if (peaksNeedingPatterns.length === 0) return;

		// bulk create new pattern neurons
		const newPatternNeuronIds = await this.bulkInsertNeurons(peaksNeedingPatterns.length);

		// prepare pattern-connection relationships with strength = 1.0
		const patternConnections = [];
		for (let i = 0; i < peaksNeedingPatterns.length; i++) {
			const { peakNeuronId, connectionIds } = peaksNeedingPatterns[i];
			const patternNeuronId = newPatternNeuronIds[i];

			// add pattern-connection relationships
			for (const connectionId of connectionIds) patternConnections.push([patternNeuronId, connectionId, 1]);

			// update the peakPatterns map with the new pattern neuron id
			peakPatterns.set(peakNeuronId, [patternNeuronId]);
		}

		// batch insert all pattern-connection relationships
		await this.conn.query(`INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES ?`, [patternConnections]);
	}

	/**
	 * reinforce the patterns for each peak by incrementing the strength of their pattern-connection relationships that were just observed.
	 * @param {Map} peakPatterns - Map of peak_neuron_id -> array of pattern_neuron_ids
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async reinforcePatterns(peakPatterns, peakConnections) {

		// use a Set to deduplicate pattern-connection pairs
		const uniquePairs = new Set();
		for (const [peakNeuronId, patternNeuronIds] of peakPatterns) {
			const connectionIds = peakConnections.get(peakNeuronId) || [];

			// create all combinations of pattern neurons and connections for this peak
			for (const patternNeuronId of patternNeuronIds)
				for (const connectionId of connectionIds)
					uniquePairs.add(`${patternNeuronId}-${connectionId}`);
		}

		// convert unique pairs back to array format for the query
		const patternConnections = Array.from(uniquePairs).map(pair => {
			const [patternNeuronId, connectionId] = pair.split('-').map(id => parseInt(id));
			return [patternNeuronId, connectionId, 1];
		});

		// batch insert/update all pattern-connection relationships
		await this.conn.query(`
        	INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES ?
        	ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)
    	`, [patternConnections]);
	}

	/**
	 * Get predicted connections from both connection_inference and pattern_inference tables
	 * Applies exponential distance weighting to connection strengths
	 * @param {number} level - The level to get predictions for
	 * @returns Promise<{Array}> Array of predicted connection objects with weighted strengths
	 */
	async getPredictedConnections(level) {
		console.log(`Getting predicted connections for level ${level}`);

		// Combine predictions from both sources and apply exponential distance weighting
		// Weight formula: (N - distance_within_tier) / POW(N, tier + 1)
		// where tier = FLOOR(LOG(N, weight_distance))
		// This gives: distance=0 → weight=1.0, distance=1 → weight=0.9, distance=10 → weight=0.1, etc.
		const [predictedConnections] = await this.conn.query(`
			SELECT
				c.id,
				c.from_neuron_id,
				c.to_neuron_id,
				c.distance,
				c.strength * (
					(:N - MOD(inf.weight_distance, POW(:N, FLOOR(LOG(:N, inf.weight_distance)) + 1)))
					/ POW(:N, FLOOR(LOG(:N, inf.weight_distance)) + 1)
				) as strength
			FROM connections c
			INNER JOIN (
				SELECT connection_id, weight_distance FROM connection_inference WHERE level = :level
				UNION ALL
				SELECT connection_id, weight_distance FROM pattern_inference WHERE level = :level
			) inf ON c.id = inf.connection_id
			WHERE c.strength > 0
			AND inf.weight_distance > 0
		`, { level, N: this.baseNeuronMaxAge });

		console.log(`Found ${predictedConnections.length} predicted connections for level ${level}`);
		return predictedConnections;
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
		console.log('Running forget cycle...');

		// 1. REWARD FORGETTING: Remove neutral rewards first, then decay toward neutral
		await this.conn.query(`DELETE FROM neuron_rewards WHERE ABS(reward_factor - 1.0) < 0.01`);
		await this.conn.query(`
			UPDATE neuron_rewards
			SET reward_factor = reward_factor + (1.0 - reward_factor) * ?
		`, [this.rewardForgetRate]);

		// 2. PATTERN FORGETTING: Reduce pattern strengths and remove dead patterns
		await this.conn.query(`UPDATE patterns SET strength = strength - ?`, [this.patternForgetRate]);
		await this.conn.query(`DELETE FROM patterns WHERE strength <= 0`);

		// 3. CONNECTION FORGETTING: Reduce connection strengths and remove dead connections
		await this.conn.query(`UPDATE connections SET strength = strength - ?`, [this.connectionForgetRate]);
		await this.conn.query(`DELETE FROM connections WHERE strength <= 0`);

		// 4. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
		await this.conn.query(`
			DELETE FROM neurons n
			WHERE NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM patterns WHERE pattern_neuron_id = n.id)
			  AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
		`);

		console.log('Forgetting cycle completed.');
	}

	/**
	 * Optimize neuron strengths based on reward factors
	 * Multiply base strength by reward factor (1.0 = neutral, >1.0 = boost, <1.0 = reduce)
	 */
	async optimizeRewards(neuronStrengths, level) {
		console.log(`Optimizing rewards for level ${level}`);

		// Get reward factors for all neurons in the strength map with batching for large sets
		const neuronIds = Array.from(neuronStrengths.keys());
		const neuronRewards = new Map();

		// Process in batches to avoid query size limits
		const batchSize = 1000;
		for (let i = 0; i < neuronIds.length; i += batchSize) {
			const batch = neuronIds.slice(i, i + batchSize);
			const [rewardRows] = await this.conn.query(`
				SELECT neuron_id, reward_factor
				FROM neuron_rewards
				WHERE neuron_id IN (${batch.map(() => '?').join(',')})
			`, batch);

			// Add batch results to the map
			for (const row of rewardRows) neuronRewards.set(row.neuron_id, row.reward_factor);
		}

		// Optimize neuron strengths based on reward factors
		const optimizedStrengths = new Map();
		let modifiedCount = 0;
		for (const [neuronId, baseStrength] of neuronStrengths) {
			const rewardFactor = neuronRewards.get(neuronId) || 1.0; // Default to neutral if no reward history

			// Multiply base strength by reward factor
			const adjustedStrength = baseStrength * rewardFactor;
			optimizedStrengths.set(neuronId, Math.max(0, adjustedStrength));
			if (rewardFactor !== 1.0) {
				console.log(`Adjusting neuron ${neuronId}: ${baseStrength.toFixed(3)} * ${rewardFactor.toFixed(3)} = ${adjustedStrength.toFixed(3)}`);
				modifiedCount++;
			}
		}

		console.log(`Optimized ${optimizedStrengths.size} neuron strengths (${modifiedCount} modified by reward factors)`);
		return optimizedStrengths;
	}

	/**
	 * Determine peak neurons from optimized strengths using peak detection algorithm
	 * and insert them into inferred_neurons table.
	 * Old predictions have already been aged and cleaned up by deactivateOldInferences.
	 */
	async inferPeakNeurons(neuronStrengths, level, predictedConnections) {
		console.log(`Inferring peak neurons for level ${level} from ${neuronStrengths.size} candidates`);

		// Use the same peak detection algorithm as detectPeaks
		// Build neighborhood map from predicted connections
		const neighborhoodMap = this.buildNeighborhoodMap(predictedConnections);

		// Get neighborhood average strengths
		const neighborhoodStrengths = this.getNeighborhoodStrengths(neighborhoodMap, neuronStrengths);

		// Get peak neuron IDs (neurons stronger than their neighborhood average)
		const peakNeuronIds = this.getPeakNeurons(neuronStrengths, neighborhoodStrengths);
		if (peakNeuronIds.length === 0) {
			console.log(`No peak neurons found for level ${level}`);
			return;
		}

		// Insert peak neurons into inferred_neurons table with age=0
		// Old age=0 predictions were already aged to age=1, so no conflict
		const peakNeurons = peakNeuronIds.map(neuronId => [neuronId, level, 0]); // neuron_id, level, age=0
		await this.conn.query('INSERT INTO inferred_neurons (neuron_id, level, age) VALUES ?', [peakNeurons]);
		console.log(`Inferred ${peakNeurons.length} peak neurons for level ${level}: [${peakNeuronIds.join(', ')}]`);
	}

	/**
	 * Apply global reward to executed inferred neurons with linear temporal decay
	 */
	async applyRewards(globalReward) {
		if (globalReward === 1.0) {
			console.log('Neutral global reward - no updates needed');
			return;
		}

		// Apply global reward to previously executed decisions with exponential temporal decay
		// age >= 1 means decisions from prior frames (not yet aged in current frame, but executed when age=1)
		//
		// Decay follows exponential rounding structure based on baseNeuronMaxAge (N):
		// - Ages 1-N: decay from 1.0 to 1/N (N steps)
		// - Ages (N+1)-(N²): decay from (N-1)/N² to 1/N² (N buckets of size N)
		// - Ages (N²+1)-(N³): decay from (N-1)/N³ to 1/N³ (N buckets of size N²)
		// - etc.
		//
		// Formula: decayFactor = (N + 1 - bucketWithinTier) / POW(N, tier + 1)
		// Where:
		//   tier = FLOOR(LOG(N, age))  -- which exponential tier (0 for 1-9, 1 for 10-99, etc.)
		//   bucketWithinTier = CEIL(age / POW(N, tier))  -- which bucket within the tier (1 to N)
		const [result] = await this.conn.execute(
			`INSERT INTO neuron_rewards (neuron_id, reward_factor)
			SELECT
				neuron_id,
				1.0 + (:globalReward - 1.0) * ((:N + 1 - CEIL(age / POW(:N, tier))) / POW(:N, tier + 1)) as reward_factor
			FROM (
				SELECT inf.neuron_id, inf.age, FLOOR(LOG(:N, inf.age)) as tier
				FROM inferred_neurons inf
				WHERE inf.age >= 1  -- Prior frames' decisions (not yet aged this frame, but executed when age=1)
			) AS aged_neurons
			ON DUPLICATE KEY UPDATE reward_factor = reward_factor * VALUES(reward_factor)`,
			{ globalReward, N: this.baseNeuronMaxAge }
		);

		console.log(`Applied global reward ${globalReward.toFixed(3)} to ${result.affectedRows} executed inferred neurons with exponential temporal decay`);
	}
}