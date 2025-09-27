import db from './db/db.js';

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
		this.forgetCycles = 1000; // number of frames between forget cycles
		this.forgetRate = 0.1; // how much the connection strengths will be decremented by at forget cycles
		this.negativeLearningRate = 0.1; // how much pattern strengths will be decremented by when not accurate
		this.maxLevels = 6; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();
		
		// track all executed output neurons per channel for reward feedback
		this.lastExecutedOutputNeurons = new Map();
		
		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;
		this.inactivityThreshold = 5; // frames of inactivity before exploration
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
		await this.conn.query('TRUNCATE active_neurons');
		await this.conn.query('TRUNCATE pattern_inference');
		await this.conn.query('TRUNCATE connection_inference');
		await this.conn.query('TRUNCATE inferred_neurons');
		await this.conn.query('TRUNCATE observed_patterns');
		await this.conn.query('TRUNCATE active_connections');
	}

	/**
	 * Hard reset: clears ALL tables (used mainly for tests)
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all tables)...');

		// truncate memory tables first (ENGINE=MEMORY) and then persistent ones
		const tables = [
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
		];
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		for (const table of tables) await this.conn.query(`TRUNCATE ${table}`);
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async init() {

		// get new connection to the database
		this.conn = await db.getConnection();

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
	 * Handle global exploration when ALL channels are inactive
	 */
	curiosityExploration() {

		// Check if the brain is inactive - if some channel is active, no exploration needed
		if ((this.frameNumber - this.lastActivity) < this.inactivityThreshold) return [];

		// brain has been inactive - trigger exploration on a random channel
		const channelNames = Array.from(this.channels.keys());
		const randomChannelName = channelNames[Math.floor(Math.random() * channelNames.length)];
		const randomChannel = this.channels.get(randomChannelName);

		// get exploration actions for the channel - if there are no valid actions, skip exploration for now, we'll try next time
		const validActions = randomChannel.getValidExplorationActions();
		if (validActions.length === 0) return [];

		// pick a random action
		const randomAction = validActions[Math.floor(Math.random() * validActions.length)];
		console.log(`Brain: Global inactivity detected, triggering exploration on ${randomChannelName}:`, randomAction);
		
		// return the random action - this is a point with a value on the output dimension of the channel
		return [randomAction];
	}

	/**
	 * returns the current frame combined from all registered channels
	 * includes exploration if all channels are inactive
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

		// add the previous outputs to the current frame so that they can be learned as patterns
		// const previousOutputs = await this.getPreviousOutputs();
		// if (previousOutputs.length > 0) frame.push(...previousOutputs);
		
		// Add exploration if globally inactive
		const explorationData = this.curiosityExploration();
		if (explorationData.length > 0) frame.push(...explorationData);

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
		console.log(`observing new frame: ${JSON.stringify(frame)}`);
		console.log(`applying global reward: ${globalReward.toFixed(3)}`);

		// do the whole thing as a transaction to avoid inconsistent database states
		await this.conn.beginTransaction();

		// if there's an error, we'll roll the transaction back
		try {

			// apply global reward to executed inferred neurons
			await this.applyRewards(globalReward);

			// age the active neurons in memory context - sliding the window
			await this.ageNeurons();

			// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
			await this.recognizeNeurons(frame);

			// populate active connections table for fast reward propagation
			await this.populateActiveConnections();

			// do predictions and outputs - what's going to happen next? and what's our best response?
			await this.inferNeurons();

			// now commit the transaction and return the new predictions
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

		// get the outputs from processing the frame - get it from the memory tables
		// habitual, pattern predicted and optimized outputs that have output dimension values
		return this.getFrameOutputs();
	}

	/**
	 * calls channels to execute the outputs from processing the frame
	 */
	async executeOutputs(outputs) {

		// Let each channel execute outputs and update their states
		for (const [channelName, channel] of this.channels) {

			// get the channel outputs - filter by the output dimensions of the channel
			const channelOutputs = this.getChannelOutputs(outputs, channel);

			// nothing to do if there are no actions to execute
			if (!channelOutputs || channelOutputs.actions.size === 0) continue;

			// Track all executed output neurons for this channel for reward feedback
			this.trackLastExecutedOutputNeurons(channelName, channelOutputs);

			// now ask the channel to execute the outputs
			await channel.executeOutputs(channelOutputs);

			// Track global activity - if any channel produced action outputs, mark brain as active
			this.lastActivity = this.frameNumber;
			console.log(`${channelName} executed actions:`, Array.from(channelOutputs.actions.keys()));
		}
	}

	/**
	 * Track all executed output neurons for a channel for future reward feedback
	 */
	trackLastExecutedOutputNeurons(channelName, channelOutputs) {

		const executedNeurons = [];
		for (const [neuronId] of channelOutputs.actions) executedNeurons.push(neuronId);
		
		if (executedNeurons.length > 0) {
			this.lastExecutedOutputNeurons.set(channelName, executedNeurons);
			console.log(`${channelName}: Tracking ${executedNeurons.length} output neurons for feedback: [${executedNeurons.join(', ')}]`);
		}
	}

	/**
	 * ages neurons in the context - sliding the window across frames
	 */
	async ageNeurons() {
		console.log('Aging active neurons and inferred neurons...');

		// age all neurons in the context
		await this.conn.query('UPDATE active_neurons SET age = age + 1');
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1');

		// deactivate aged-out neurons, but age higher level neurons slower than the lower level neurons
		// if level 0 max age = 10, level 1 max age = 100, level 2 is 1000, etc.
		await this.conn.query('DELETE FROM active_neurons WHERE age >= POW(?, level + 1)', [this.baseNeuronMaxAge]);
		await this.conn.query('DELETE FROM inferred_neurons WHERE age >= POW(?, level + 1)', [this.baseNeuronMaxAge]);
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

			// do the rewards optimizations - avoid pain, maximize joy
			neuronStrengths = await this.optimizeRewards(neuronStrengths, level);

			// determine peak neurons for the level using peak detection algorithm
			await this.inferPeakNeurons(neuronStrengths, level, predictedConnections);
		}
	}

	/**
	 * infer connections that will happen for the next cycle of the level from observations
	 * handles rolling window of predictions with proper time dilation for higher levels.
	 */
	async inferConnections(level) {
		
		// age all existing predictions in the connection_inference table
		await this.conn.query('UPDATE connection_inference SET age = age + 1 WHERE level = ?', [level]);
		
		// remove predictions that came true (predicted connection's to_neuron is now active at age 0)
		// we don't need to strengthen the predicted connections that came true - strength was already be increased simply because of observation
		await this.conn.query(`
			DELETE ci 
			FROM connection_inference ci
			JOIN connections c ON ci.connection_id = c.id
			JOIN active_neurons an ON c.to_neuron_id = an.neuron_id AND an.level = ci.level AND an.age = 0 
			WHERE ci.level = ? 
		`, [level]);

		// but if the predicted connections did not come true, and they are expiring at max age, reduce their strengths
		// these predictions did not come true and are about to be removed
		await this.conn.query(`
            UPDATE connections
            SET strength = strength - ?
            WHERE id IN (
            	SELECT ci.connection_id 
            	FROM connection_inference ci 
            	WHERE ci.level = ? 
            	AND ci.age >= POW(?, ci.level)
            )
		`, [this.negativeLearningRate, level, this.baseNeuronMaxAge]);

		// now that we used them for reinforcement, clear previous expired predictions for this level
		await this.conn.query('DELETE FROM connection_inference WHERE level = ? AND age >= POW(?, level)', [level, this.baseNeuronMaxAge]);

		// insert new predictions for the next cycle of this level from all active neurons
		// predict connections where distance matches the next temporal step for the neuron's age (age + 1)
		// distance calculation accounts for time dilation: level 0 = exact, higher levels = bucketed
		await this.conn.query(`
			INSERT IGNORE INTO connection_inference (level, connection_id, age)
			SELECT f.level, c.id, 0
			FROM active_neurons f
			JOIN connections c ON c.from_neuron_id = f.neuron_id AND c.distance = FLOOR((f.age + 1) / POW(?, f.level))
			WHERE f.level = ?
			AND c.strength >= 0
		`, [this.baseNeuronMaxAge, level]);
	}

	/**
	 * Lower-level predictions from active patterns at the current level.
	 * These predictions are valid for the reduced time-span of the lower level.
	 * If pattern is in level 1, predictions are valid for 10 cycles of level 0.
	 * If pattern is in level 2, predictions are valid for 10 cycles of level 1 (100 cycles of level 0).
	 */
	async inferPatterns(level) {
		console.log(`Processing lower-level predictions for level ${level}`);

		// age existing pattern predictions in the lower level
		await this.conn.query('UPDATE pattern_inference SET age = age + 1 WHERE level = ?', [level - 1]);

		// remove predictions that came true (predicted connections are now active at age=0 in the target level)
		// we don't need to strengthen the predicted connections that came true - strength was already be increased simply because of observation
		await this.conn.query(`
            DELETE pi
            FROM pattern_inference pi
            JOIN connections c ON pi.connection_id = c.id
            JOIN active_neurons an ON c.to_neuron_id = an.neuron_id AND an.level = pi.level AND an.age = 0
			WHERE pi.level = ? 
		`, [level - 1]);

		// but if the predicted connections did not come true, and they are expiring at max age, reduce their strengths
		// these predictions did not come true and are about to be removed
		await this.conn.query(`
            UPDATE patterns
            SET strength = strength - ?
            WHERE pattern_neuron_id IN (
            	SELECT pi.pattern_neuron_id 
            	FROM pattern_inference pi 
            	WHERE pi.level = ?
                AND pi.age >= POW(?, pi.level + 1)
			)
		`, [this.negativeLearningRate, level - 1, this.baseNeuronMaxAge]);

		// now that we used them for reinforcements, delete expired pattern predictions
		await this.conn.query('DELETE FROM pattern_inference WHERE level = ? AND age >= POW(?, level + 1)', [level - 1, this.baseNeuronMaxAge]);

		// create new predictions from patterns that were just activated in the higher level
		// if the higher level patterns are just activated (age=0) they must have been activated due to
		// some new neurons in the lower level (age=0 as well) - exclude them
		await this.conn.query(`
			INSERT INTO pattern_inference (level, pattern_neuron_id, connection_id, age)
			SELECT an.level - 1, p.pattern_neuron_id, p.connection_id, 0 
			FROM active_neurons an
			JOIN patterns p ON an.neuron_id = p.pattern_neuron_id
			JOIN connections c ON p.connection_id = c.id
			WHERE an.level = ?
			AND an.age = 0
			AND p.strength > 0
			AND NOT EXISTS (
                SELECT 1
                FROM active_neurons f
                JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = f.level
                WHERE c.from_neuron_id = f.neuron_id
                AND c.distance = FLOOR(f.age / POW(?, f.level))
                AND f.level = an.level - 1
                AND (t.neuron_id != f.neuron_id OR f.age != 0)
                AND c.strength >= 0
            )
		`, [level, this.baseNeuronMaxAge]);
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame
	 */
	async getFrameNeurons(points) {

		// try tp get all the neurons that have coordinates in close ranges for each point - return format: [{ point_str, neuron_id }]
		const matches = await this.matchNeuronsFromPoints(points);
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
	async matchNeuronsFromPoints(frame) {
		if (frame.length === 0) return [];

		// for each point in the frame, create a separate select query
		const unionQueries = [];
		for (const point of frame) {
			const dimensions = Object.keys(point);
			const dimCount = dimensions.length;
			
			// Build the dimension-value pairs for this point
			const dimensionValuePairs =
				Object.entries(point).map(([dimName, val]) => `(${this.dimensionNameToId[dimName]}, ${val})`);
			
			// Create a subquery for this point that finds neurons matching ALL its dimensions
			unionQueries.push(`
                SELECT '${JSON.stringify(point)}' as point_str, GROUP_CONCAT(neuron_id) as neuron_id
                FROM (
					SELECT DISTINCT neuron_id
					FROM coordinates
                    WHERE (dimension_id, val) IN (${dimensionValuePairs.join(', ')})
					GROUP BY neuron_id
					HAVING COUNT(*) = ${dimCount}
				) AS matched_neurons
			`);
		}

		// combine all point queries with UNION and get the matching neurons for each point
		const sql = unionQueries.join(' UNION ALL ');
		const [rows] = await this.conn.query(sql);

		// if there are any points with multiple matches, error out
		// this is theoretically possible, but should not happen as long as we are consistent with the input dimensions
		if (rows.find(row => row.neuron_id && row.neuron_id.includes(','))) throw new Error(`Multiple point matches: ${sql}`);

		// return results
		return rows.map(row => ({ point_str: row.point_str, neuron_id: Number(row.neuron_id) }));
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

		// Rest of the coordinate insertion logic...
		const rows = created.flatMap(({ neuron_id, point }) =>
			Object.entries(point).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [rows]);

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
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT 
                f.neuron_id as from_neuron_id, -- connection from the neuron id
                t.neuron_id as to_neuron_id, -- connection to the neuron id
                FLOOR(f.age / POW(?, f.level)) as distance, -- bucketed distance: level 0 = exact, higher levels = bucketed
                1 as strength -- each observation increases strength by 1 
			FROM active_neurons f -- connections are built from the older neurons to the new neurons
			JOIN active_neurons t ON t.level = f.level -- reinforcing connections only within the same level
            WHERE t.age = 0 -- reinforcing connections for the newly activated target neurons only
            AND f.level = ? -- get the active neurons in the given level
            AND (t.neuron_id != f.neuron_id OR f.age != 0) -- if it's the same neuron, it's gotta be an older one
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength) -- if connection exists, add on to it
		`, [this.baseNeuronMaxAge, level]);
	}

	/**
	 * activate neurons at a specified level & distance - age is always zero - not sending to save characters in query
	 */
	async activateNeurons(neuronIds, level = 0) {

		// insert given neurons to the active neurons table
		await this.insertActiveNeurons(neuronIds, level);

		// reinforce connections between active neurons in the level
		await this.reinforceConnections(level);
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
		if (connections.length === 0) return false; // if there are no connections (there is only one neuron), nothing to do

		// cluster connections around peaks
		const peakConnections = this.detectPeaks(connections);
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
	 * returns active directed connections flowing INTO newly activated neurons (age=0) from all active neurons at the specified level for spatio-temporal pooling
	 * connections are directed: older neurons → newer neurons. note that this includes connections between the age=0 neurons as well. that's the spatial pooling.
	 * the others are temporal. so, these connections form the basis of the spatio-temporal pooling. there may be connections between the same neuron when their ages
	 * are different. there may even be indirect connections between the same neuron on different levels (not sure how, but it looks possible).
	 * there should not be any connections from the same neuron to itself within the same age and level.
	 */
	async getActiveConnections(level) {
		const [rows] = await this.conn.query(`
            SELECT c.id, c.from_neuron_id, c.to_neuron_id, c.distance, c.strength
            FROM connections c
            -- source is the older neurons building connections from them to the new neurons
			JOIN active_neurons f 
				ON c.from_neuron_id = f.neuron_id 
			    AND c.distance = FLOOR(f.age / POW(?, f.level))
            -- getting active connections from older neurons to newly activated neurons within the same level
            JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = f.level
            WHERE f.level = ? -- get the active neurons in the given level
            AND (t.neuron_id != f.neuron_id OR f.age != 0) -- if it's the same neuron, it's gotta be an older one
            AND c.strength >= 0 -- ignore negative connections that are scheduled to be deleted 
		`, [this.baseNeuronMaxAge, level]);
		return rows;
	}

	/**
	 * Generic peak detection that works for both active and predicted connections
	 * @param {Array} connections - Array of connection objects with strength property
	 * @returns {Map} Map of peak neuron IDs to their connection IDs
	 */
	detectPeaks(connections) {

		// get the neuron strengths from the active connections (both ways - from and to)
		const neuronStrengths = this.getNeuronStrengths(connections);

		// get the neighborhood map for each neuron from its connections
		const neighborhoodMap = this.buildNeighborhoodMap(connections);

		// calculate the neighborhood average strength of each neuron. for each neuron, we need to calculate the strengths of its connected neurons.
		const neighborhoodStrengths = this.getNeighborhoodStrengths(neighborhoodMap, neuronStrengths);

		// now get the neuron ids whose strength exceeds its neighborhood - those are the peaks
		const peakNeuronIds = this.getPeakNeurons(neuronStrengths, neighborhoodStrengths);

		// get a map from peak neuron ids to the connection ids it uses and return them
		return this.getPeakNeuronsConnections(peakNeuronIds, neighborhoodMap, connections);
	}

	/**
	 * returns the strength of each neuron from the sum of its active connections strengths.
	 * this includes both incoming connections (to the neuron) and outgoing connections (from the neuron).
	 */
	getNeuronStrengths(connections) {
		const neuronStrengths = new Map();
		for (const connection of connections) {
			const { from_neuron_id, to_neuron_id, strength } = connection;

			// add strength for the source neuron (outgoing connection)
			if (neuronStrengths.has(from_neuron_id)) neuronStrengths.set(from_neuron_id, neuronStrengths.get(from_neuron_id) + strength);
			else neuronStrengths.set(from_neuron_id, strength);

			// add strength for the target neuron (incoming connection)
			if (neuronStrengths.has(to_neuron_id)) neuronStrengths.set(to_neuron_id, neuronStrengths.get(to_neuron_id) + strength);
			else neuronStrengths.set(to_neuron_id, strength);
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
	 */
	getPeakNeurons(neuronStrengths, neighborhoodStrengths) {
		const peaks = [];
		for (const [neuronId, neuronStrength] of neuronStrengths) {
			const neighborhoodStrength = neighborhoodStrengths.get(neuronId) || 0;
			if (neuronStrength > neighborhoodStrength) peaks.push(neuronId);
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
		return connectionIds;
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
	 * @param {number} level - The level to get predictions for
	 * @returns Promise<{Array}> Array of predicted connection objects
	 */
	async getPredictedConnections(level) {
		console.log(`Getting predicted connections for level ${level}`);

		// get both connection-level and pattern-level predictions in a single query
		const [predictedConnections] = await this.conn.query(`
			SELECT c.id, c.from_neuron_id, c.to_neuron_id, c.distance, c.strength
			FROM connections c
			WHERE c.strength >= 0 AND c.id IN (
				SELECT connection_id FROM connection_inference WHERE level = ?
				UNION
				SELECT connection_id FROM pattern_inference WHERE level = ?
			)
		`, [level, level]);
		
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
	 * runs the forget cycle, reducing conn weights and deleting unused neurons signifying unseen patterns
	 * also deletes obsolete (negative) connections - very important step that helps the system avoid curse of dimensionality
	 */
	async runForgetCycle() {

		// we run the forget cycle periodically for clean up
		this.forgetCounter++;
		if (this.forgetCounter % this.forgetCycles !== 0) return;
		this.forgetCounter = 0;
		console.log('Running forget cycle...');

		// reduce pattern strengths across the board and remove dead patterns
		await this.conn.query(`UPDATE patterns SET strength = strength - ?`, [this.forgetRate]);
		await this.conn.query(`DELETE FROM patterns WHERE strength <= 0`);

		// reduce connection strengths across the board and remove dead connections
		await this.conn.query(`UPDATE connections SET strength = strength - ?`, [this.forgetRate]);
		await this.conn.query(`DELETE FROM connections WHERE strength <= 0`);

		// remove orphaned neurons with no connections, no patterns, and not currently active
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
	 * Get frame outputs from inferred neurons that have output dimensions
	 */
	async getFrameOutputs() {
		console.log('Getting frame outputs from inferred neurons...');

		// Get frame outputs from inferred neurons that have output dimensions
		// Only Level 0 neurons can be outputs since they have coordinates
		const [rows] = await this.conn.query(`
			SELECT inf.neuron_id, inf.level, c.dimension_id, c.val, d.name as dimension_name, d.channel, d.type
			FROM inferred_neurons inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE d.type = 'output'
			AND inf.age = 0
			AND inf.level = 0
			ORDER BY inf.neuron_id
		`);
		if (rows.length === 0) {
			console.log('No output neurons found');
			return [];
		}

		// Group by neuron and create output objects
		const neuronOutputs = new Map();
		for (const row of rows) {
			const neuronId = row.neuron_id;
			if (!neuronOutputs.has(neuronId))
				neuronOutputs.set(neuronId, { neuron_id: neuronId, coordinates: {}, channel: row.channel });
			neuronOutputs.get(neuronId).coordinates[row.dimension_name] = row.val;
		}
		const outputs = Array.from(neuronOutputs.values());
		console.log(`Found ${outputs.length} output neurons`);
		return outputs;
	}

	/**
	 * Get channel-specific outputs filtered by the channel's output dimensions
	 */
	getChannelOutputs(outputs, channel) {
		if (!outputs || outputs.length === 0) return null;

		const channelOutputs = outputs.filter(output => output.channel === channel.name);
		if (channelOutputs.length === 0) return null;

		return {
			actions: new Map(channelOutputs.map(output => [output.neuron_id, output])),
			channelName: channel.name
		};
	}

	/**
	 * Optimize neuron strengths based on reward factors
	 * Multiply base strength by reward factor (1.0 = neutral, >1.0 = boost, <1.0 = reduce)
	 */
	async optimizeRewards(neuronStrengths, level) {
		console.log(`Optimizing rewards for level ${level}`);

		// Get reward factors for all neurons in the strength map
		const neuronIds = Array.from(neuronStrengths.keys());
		const [rewardRows] = await this.conn.query(`
			SELECT neuron_id, reward_factor
			FROM neuron_rewards
			WHERE neuron_id IN (${neuronIds.map(() => '?').join(',')})
		`, neuronIds);

		// Create a map of neuron reward factors
		const neuronRewards = new Map();
		for (const row of rewardRows) neuronRewards.set(row.neuron_id, row.reward_factor);

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
	 * and insert them into inferred_neurons table
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

		// Insert peak neurons into inferred_neurons table
		const peakNeurons = peakNeuronIds.map(neuronId => [neuronId, level, 0]); // neuron_id, level, age=0
		await this.conn.query('INSERT IGNORE INTO inferred_neurons (neuron_id, level, age) VALUES ?', [peakNeurons]);
		console.log(`Inferred ${peakNeurons.length} peak neurons for level ${level}: [${peakNeuronIds.join(', ')}]`);
	}

	/**
	 * Populate active_connections table for fast hierarchical reward propagation
	 */
	async populateActiveConnections() {
		console.log('Populating active connections table...');

		// Clear previous active connections
		await this.conn.query('TRUNCATE active_connections');

		// Populate active connections for all levels in one query
		await this.conn.query(`
			INSERT INTO active_connections (connection_id, from_neuron_id, to_neuron_id, level, strength)
			SELECT DISTINCT 
				c.id as connection_id,
				c.from_neuron_id,
				c.to_neuron_id,
				an_from.level,
				c.strength
			FROM connections c
			JOIN active_neurons an_from ON c.from_neuron_id = an_from.neuron_id
			JOIN active_neurons an_to ON c.to_neuron_id = an_to.neuron_id AND an_to.level = an_from.level
			WHERE c.strength > 0
		`);

		const [countRows] = await this.conn.query('SELECT COUNT(*) as count FROM active_connections');
		console.log(`Populated ${countRows[0].count} active connections`);
	}

	/**
	 * Apply global reward to executed inferred neurons with linear temporal decay
	 */
	async applyRewards(globalReward) {
		if (globalReward === 1.0) {
			console.log('Neutral global reward - no updates needed');
			return;
		}

		// Apply global reward to all executed decisions (age >= 1) with linear temporal decay
		// Recent decisions get full reward, older decisions get proportionally less, oldest get neutral (1.0)
		// Formula: 1.0 + (globalReward - 1.0) * (1.0 - age/levelMaxAge)
		const [result] = await this.conn.query(`
			INSERT INTO neuron_rewards (neuron_id, reward_factor)
			SELECT inf.neuron_id, 1.0 + (? - 1.0) * (1.0 - inf.age / POW(?, inf.level + 1))  -- Linear temporal decay
			FROM inferred_neurons inf
			WHERE inf.age >= 1  -- Executed decisions only, skip current frame
			ON DUPLICATE KEY UPDATE reward_factor = reward_factor * VALUES(reward_factor)
		`, [globalReward, this.baseNeuronMaxAge]);

		console.log(`Applied global reward ${globalReward.toFixed(3)} to ${result.affectedRows} executed inferred neurons with linear temporal decay`);
	}
}