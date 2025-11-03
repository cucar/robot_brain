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
		this.maxLevels = 10; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.10; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.minPeakStrength = 10.0; // minimum weighted strength for a neuron to be considered a peak (used for both pattern detection and prediction)
		this.minPeakRatio = 1.2; // minimum ratio of peak strength to neighborhood average to be considered a peak (used for both pattern detection and prediction)
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.9; // reward temporal decay = POW(rewardTimeDecayFactor, age)
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.negativeLearningRate = 0.1; // how much to weaken connections when predictions fail
		this.minConnectionStrength = 0; // minimum strength value for connections and patterns (clamped to prevent negative values)
		this.maxConnectionStrength = 1000; // maximum strength value for connections and patterns (clamped to prevent overflow)

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;
		this.inactivityThreshold = 5; // frames of inactivity before exploration

		// Prediction accuracy tracking (cumulative stats per level)
		this.accuracyStats = new Map(); // level -> { connection: {correct, total}, pattern: {correct, total}, resolved: {correct, total} }

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

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
			'observed_connections',
			'observed_neuron_strengths',
			'observed_peaks',
			'observed_patterns',
			'matched_peaks',
			'active_connections'
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
			'connection_inference',
			'connection_inferred_neurons',
			'pattern_inferred_neurons',
			'inferred_neurons',
			'observed_connections',
			'observed_neuron_strengths',
			'observed_peaks',
			'observed_patterns',
			'matched_peaks',
			'active_connections',
			'matched_patterns',
			'pattern_peaks',
			'patterns',
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
		if (this.debug) console.log('Getting feedback from all channels...');
		let globalReward = 1.0; // Start with neutral
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const rewardFactor = await channel.getFeedback();
			if (rewardFactor !== 1.0) { // Only process non-neutral feedback
				if (this.debug) console.log(`${channelName}: reward factor ${rewardFactor.toFixed(3)}`);
				globalReward *= rewardFactor; // Multiplicative aggregation
				feedbackCount++;
			}
		}

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Total reward: ${globalReward.toFixed(3)} (${feedbackCount} channels)`);
			else console.log('No feedback from any channels');
		}

		return globalReward;
	}

	/**
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and global reward factor from aggregated channel feedback
	 */
	async processFrame(frame, globalReward = 1.0) {
		const frameStart = performance.now();

		if (this.debug) {
			console.log('******************************************************************');
			console.log(`OBSERVING NEW FRAME: ${JSON.stringify(frame)}`, this.frameNumber);
			console.log(`applying global reward: ${globalReward.toFixed(3)}`);
			console.log('******************************************************************');
		}

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

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// run forget cycle periodically and delete dead connections/neurons
		await this.runForgetCycle();

		// show frame processing summary
		const frameElapsed = performance.now() - frameStart;
		this.printFrameSummary(frameElapsed);

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
			if (this.debug) console.log('No previous outputs to execute');
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
		if (this.debug) console.log('Brain inactive - executing curiosity exploration');

		// Get a random channel for exploration
		const channelNames = Array.from(this.channels.keys());
		const randomChannelName = channelNames[Math.floor(Math.random() * channelNames.length)];
		const randomChannel = this.channels.get(randomChannelName);

		// Get exploration actions for the channel
		const explorationActions = randomChannel.getValidExplorationActions();
		if (explorationActions.length === 0) {
			if (this.debug) console.log(`No valid exploration actions for ${randomChannelName}`);
			return;
		}

		// Execute random exploration action
		const randomAction = explorationActions[Math.floor(Math.random() * explorationActions.length)];
		if (this.debug) console.log(`${randomChannelName}: Executing exploration action:`, randomAction);

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
			if (this.debug) console.log(`Warning: Channel ${channelName} not found`);
			return;
		}

		if (this.debug) console.log(`${channelName}: Executing outputs:`, coordinates);
		await channel.executeOutputs(coordinates);

		// Track global activity
		this.lastActivity = this.frameNumber;
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= baseNeuronMaxAge.
	 */
	async ageNeurons() {
		if (this.debug) console.log('Aging active neurons, connections, and inferred neurons...');

		// age all neurons and connections - ORDER BY age DESC to avoid primary key collisions
		// (update highest ages first so age+1 doesn't collide with existing lower age+1 row)
		await this.conn.query('UPDATE active_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE active_connections SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE connection_inferred_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE pattern_inferred_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.baseNeuronMaxAge]);
		if (this.debug) console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Clean up inferred neurons after execution (age >= 2)
		// age=0: fresh predictions, age=1: executed this frame, age>=2: no longer needed
		const [connectionInferredResult] = await this.conn.query('DELETE FROM connection_inferred_neurons WHERE age >= 2');
		if (this.debug) console.log(`Cleaned up ${connectionInferredResult.affectedRows} executed connection inferred neurons (age >= 2)`);

		const [patternInferredResult] = await this.conn.query('DELETE FROM pattern_inferred_neurons WHERE age >= 2');
		if (this.debug) console.log(`Cleaned up ${patternInferredResult.affectedRows} executed pattern inferred neurons (age >= 2)`);

		const [inferredResult] = await this.conn.query('DELETE FROM inferred_neurons WHERE age >= 2');
		if (this.debug) console.log(`Cleaned up ${inferredResult.affectedRows} executed inferred neurons (age >= 2)`);
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
	 * Infer predictions and outputs using bulk processing for all levels.
	 * Connection inference handles validation, aging, and deletion internally for all levels.
	 * Pattern inference cascades predictions down all levels recursively.
	 */
	async inferNeurons() {

		// Report accuracy for all levels (from previous frame)
		await this.reportPredictionsAccuracy();

		// Connection inference for all levels (handles validation, aging, deletion)
		await this.inferConnections();

		// Pattern inference (recursive cascade down all levels)
		await this.inferPatterns();

		// Merge predictions for higher levels (level > 0)
		await this.mergeHigherLevelPredictions();

		// Resolve conflicts in input predictions at base level (level 0)
		await this.resolveInputPredictionConflicts();
	}

	/**
	 * Merge connection and pattern predictions for higher levels (level > 0).
	 * Higher levels don't use channel-based conflict resolution - just combine predictions by summing strengths.
	 * This allows the brain to learn connections between output neurons and high-level decision making.
	 */
	async mergeHigherLevelPredictions() {

		// Combine connection and pattern predictions for all levels > 0
		// Union both sources and sum strengths per neuron per level
		await this.conn.query(`
			INSERT INTO inferred_neurons (neuron_id, level, age, strength)
			SELECT neuron_id, level, age, SUM(strength) as strength
			FROM (
				SELECT neuron_id, level, age, strength
				FROM connection_inferred_neurons
				WHERE level > 0 AND age = 0

				UNION ALL

				SELECT neuron_id, level, age, strength
				FROM pattern_inferred_neurons
				WHERE level > 0 AND age = 0
			) AS combined_predictions
			GROUP BY neuron_id, level, age
		`);

		// Log predictions per level
		if (this.debug) {
			const [results] = await this.conn.query(`
				SELECT level, COUNT(DISTINCT neuron_id) as count
				FROM inferred_neurons
				WHERE age = 0 AND level > 0
				GROUP BY level
				ORDER BY level DESC
			`);
			for (const row of results)
				console.log(`Level ${row.level}: Merged ${row.count} predictions to inferred_neurons (connection + pattern)`);
		}
	}

	/**
	 * Resolve conflicts in input predictions per channel (level 0 only).
	 * Reads from connection_inferred_neurons and pattern_inferred_neurons,
	 * resolves conflicts using channel logic, and writes final predictions to inferred_neurons.
	 */
	async resolveInputPredictionConflicts() {

		// get the most recent predictions for the next frame at level 0 - if there are none, nothing to resolve
		const connectionRows = await this.getInputPredictions('connection_inferred_neurons');
		const patternRows = await this.getInputPredictions('pattern_inferred_neurons');
		if (connectionRows.length === 0 && patternRows.length === 0) return;

		// Group predictions by channel and resolve conflicts and write to the inferred_neurons table
		const channelPredictions = this.groupPredictionsByChannel(connectionRows, patternRows);
		await this.resolveAndWritePredictions(channelPredictions);
	}

	/**
	 * Get input predictions from a specific inference table
	 */
	async getInputPredictions(tableName) {
		const [rows] = await this.conn.query(`
			SELECT inf.neuron_id, inf.strength, c.dimension_id, c.val, d.name as dimension_name, d.channel
			FROM ${tableName} inf
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE d.type = 'input' AND inf.age = 0 AND inf.level = 0
			ORDER BY d.channel, inf.neuron_id
		`);
		return rows;
	}

	/**
	 * Group predictions by channel, building complete prediction objects with coordinates.
	 * If both connection and pattern inference predict the same neuron, their strengths are summed.
	 */
	groupPredictionsByChannel(connectionRows, patternRows) {
		const channelPredictions = new Map();
		this.addPredictionsToChannelMap(channelPredictions, connectionRows);
		this.addPredictionsToChannelMap(channelPredictions, patternRows);
		return channelPredictions;
	}

	/**
	 * Add predictions from rows to the channel map.
	 * If a neuron is already predicted, sum the strengths (both sources agree = higher confidence).
	 */
	addPredictionsToChannelMap(channelPredictions, rows) {
		for (const row of rows) {

			// if the channel doesn't have a map yet, create one - otherwise get it from the map
			if (!channelPredictions.has(row.channel)) channelPredictions.set(row.channel, new Map());
			const channelMap = channelPredictions.get(row.channel);

			// if the neuron doesn't have a prediction yet, create one
			if (!channelMap.has(row.neuron_id)) channelMap.set(row.neuron_id, { neuron_id: row.neuron_id, strength: row.strength, coordinates: {} });
			// Both connection and pattern predict this neuron - sum strengths
			else channelMap.get(row.neuron_id).strength += row.strength;

			// add the coordinate to the neuron's prediction
			channelMap.get(row.neuron_id).coordinates[row.dimension_name] = row.val;
		}
	}

	/**
	 * Resolve conflicts for each channel and write final predictions to inferred_neurons.
	 * Channels can return multiple predictions (e.g., vision detecting multiple objects).
	 */
	async resolveAndWritePredictions(channelPredictions) {

		// Resolve conflicts for each channel and collect selected predictions
		const allSelectedPredictions = [];
		for (const [channelName, predictionMap] of channelPredictions)
			allSelectedPredictions.push(...this.channels.get(channelName).resolveConflicts(Array.from(predictionMap.values())));

		// if there are no predictions, nothing to resolve
		if (allSelectedPredictions.length === 0) return;

		// Batch insert all selected predictions at once
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength) VALUES ?',
			[allSelectedPredictions.map(p => [p.neuron_id, 0, 0, p.strength])]
		);
		if (this.debug) console.log(`Resolved ${allSelectedPredictions.length} input predictions after conflict resolution`);
	}

	/**
	 * Reports accuracy of neuron predictions from the previous frame for ALL levels.
	 * Tracks accuracy for connection_inferred_neurons, pattern_inferred_neurons, and final inferred_neurons.
	 */
	async reportPredictionsAccuracy() {

		// Get all connection predictions and matches in bulk
		const [connectionData] = await this.conn.query(`
			SELECT cin.level, cin.neuron_id, IF(an.neuron_id IS NOT NULL, 1, 0) as matched
			FROM connection_inferred_neurons cin
			LEFT JOIN active_neurons an ON cin.neuron_id = an.neuron_id AND an.level = cin.level AND an.age = 0
			WHERE cin.age = 1
		`);

		// Get all pattern predictions and matches in bulk
		const [patternData] = await this.conn.query(`
			SELECT pin.level, pin.neuron_id, IF(an.neuron_id IS NOT NULL, 1, 0) as matched
			FROM pattern_inferred_neurons pin
			LEFT JOIN active_neurons an ON pin.neuron_id = an.neuron_id AND an.level = pin.level AND an.age = 0
			WHERE pin.age = 1
		`);

		// Get all resolved predictions and matches in bulk (all levels)
		const [resolvedData] = await this.conn.query(`
			SELECT inf.level, inf.neuron_id, IF(an.neuron_id IS NOT NULL, 1, 0) as matched
			FROM inferred_neurons inf
			LEFT JOIN active_neurons an ON inf.neuron_id = an.neuron_id AND an.level = inf.level AND an.age = 0
			WHERE inf.age = 1
		`);

		// Group by level
		const levelStats = new Map();

		for (const row of connectionData) {
			if (!levelStats.has(row.level))
				levelStats.set(row.level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			const stats = levelStats.get(row.level);
			stats.connection.total++;
			if (row.matched) stats.connection.correct++;
		}

		for (const row of patternData) {
			if (!levelStats.has(row.level))
				levelStats.set(row.level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			const stats = levelStats.get(row.level);
			stats.pattern.total++;
			if (row.matched) stats.pattern.correct++;
		}

		for (const row of resolvedData) {
			if (!levelStats.has(row.level))
				levelStats.set(row.level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			const stats = levelStats.get(row.level);
			stats.resolved.total++;
			if (row.matched) stats.resolved.correct++;
		}

		// Report and update cumulative stats for each level
		const levels = Array.from(levelStats.keys()).sort((a, b) => b - a);
		for (const level of levels) {
			const stats = levelStats.get(level);

			// Initialize cumulative stats for this level if needed
			if (!this.accuracyStats.has(level))
				this.accuracyStats.set(level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			const cumulative = this.accuracyStats.get(level);

			// Report connection accuracy
			if (stats.connection.total > 0) {
				cumulative.connection.correct += stats.connection.correct;
				cumulative.connection.total += stats.connection.total;
				const currentRate = (stats.connection.correct / stats.connection.total * 100).toFixed(1);
				const avgRate = (cumulative.connection.correct / cumulative.connection.total * 100).toFixed(1);
				if (this.debug)
					console.log(`Level ${level}: Connection prediction accuracy: ${stats.connection.correct}/${stats.connection.total} (${currentRate}%) | Avg: ${cumulative.connection.correct}/${cumulative.connection.total} (${avgRate}%)`);
			}

			// Report pattern accuracy
			if (stats.pattern.total > 0) {
				cumulative.pattern.correct += stats.pattern.correct;
				cumulative.pattern.total += stats.pattern.total;
				const currentRate = (stats.pattern.correct / stats.pattern.total * 100).toFixed(1);
				const avgRate = (cumulative.pattern.correct / cumulative.pattern.total * 100).toFixed(1);
				if (this.debug)
					console.log(`Level ${level}: Pattern prediction accuracy: ${stats.pattern.correct}/${stats.pattern.total} (${currentRate}%) | Avg: ${cumulative.pattern.correct}/${cumulative.pattern.total} (${avgRate}%)`);
			}

			// Report resolved accuracy (level 0 only)
			if (stats.resolved.total > 0) {
				cumulative.resolved.correct += stats.resolved.correct;
				cumulative.resolved.total += stats.resolved.total;
				const currentRate = (stats.resolved.correct / stats.resolved.total * 100).toFixed(1);
				const avgRate = (cumulative.resolved.correct / cumulative.resolved.total * 100).toFixed(1);
				if (this.debug)
					console.log(`Level ${level}: Resolved prediction accuracy: ${stats.resolved.correct}/${stats.resolved.total} (${currentRate}%) | Avg: ${cumulative.resolved.correct}/${cumulative.resolved.total} (${avgRate}%)`);
			}
		}
	}

	/**
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {
		// Get base level (level 0) accuracy
		let baseAccuracy = 'N/A';
		const baseCumulative = this.accuracyStats.get(0);
		if (baseCumulative && baseCumulative.connection.total > 0)
			baseAccuracy = `${(baseCumulative.connection.correct / baseCumulative.connection.total * 100).toFixed(1)}%`;

		// Get higher level accuracy (aggregate all levels > 0)
		let higherCorrect = 0;
		let higherTotal = 0;
		for (const [level, stats] of this.accuracyStats.entries()) {
			if (level > 0) {
				higherCorrect += stats.connection.correct;
				higherTotal += stats.connection.total;
			}
		}
		const higherAccuracy = higherTotal > 0 ? `${(higherCorrect / higherTotal * 100).toFixed(1)}%` : 'N/A';

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

		console.log(`Frame ${this.frameNumber} | Base: ${baseAccuracy} | Higher: ${higherAccuracy} | MAPE: ${mapeDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}

	/**
	 * Apply negative reinforcement to connections that predicted incorrectly.
	 * Validates connection predictions from the previous frame for ALL levels.
	 */
	async negativeReinforceConnections() {

		// Find which predictions failed (not in active_connections) across all levels
		const [failures] = await this.conn.query(`
			SELECT ci.level, ci.connection_id
			FROM connection_inference ci
			WHERE NOT EXISTS (
			    SELECT 1 
			    FROM active_connections ac
			    WHERE ci.connection_id = ac.connection_id 
			      AND ac.level = ci.level 
			      AND ac.age = 0
			)    
		`);

		if (failures.length === 0) return;

		// Apply negative reinforcement to failed predictions (clamped between minConnectionStrength and maxConnectionStrength)
		const failedConnectionIds = failures.map(f => f.connection_id);
		await this.conn.query(
			'UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE id IN (?)',
			[this.minConnectionStrength, this.maxConnectionStrength, this.negativeLearningRate, failedConnectionIds]);

		// Log per level
		const failuresByLevel = new Map();
		for (const f of failures)
			failuresByLevel.set(f.level, (failuresByLevel.get(f.level) || 0) + 1);
		if (this.debug)
			for (const [level, count] of failuresByLevel.entries())
				console.log(`Level ${level}: Applied negative reinforcement to ${count} failed connection predictions`);
	}

	/**
	 * Connection inference: Predict next frame's neurons from active connections for ALL levels at once.
	 * Handles validation and deletion of previous predictions.
	 * Uses average-based peak detection to identify strong predictions.
	 * Only predicts neurons that are known peaks (exist in pattern_peaks table).
	 * Uses scratch memory tables for better performance (similar to getObservedPatterns).
	 */
	async inferConnections() {

		// Validate predictions for all levels (from previous frame)
		await this.negativeReinforceConnections();

		// Clear previous predictions (connection_inference is scratch table, just truncate)
		await this.conn.query('TRUNCATE connection_inference');

		// Step 1: Materialize candidate connections
		await this.getInferredConnections();

		// Step 2: Aggregate per neuron per level
		await this.getInferredNeuronStrengths();

		// Step 3: Handle levels where neuron is peak by default (no neighbors)
		const defaultPeakCount = await this.handleSingleValueInference();

		// Step 4: Handle multi-value inference (peak detection with neighbors)
		const detectedPeakCount = await this.handleMultiValueInference();

		// Check if we have any predictions
		if (defaultPeakCount === 0 && detectedPeakCount === 0) {
			if (this.debug) console.log('No connection predictions found');
			return;
		}

		// Step 5: Convert connection predictions to neuron predictions
		await this.conn.query(`
			INSERT INTO connection_inferred_neurons (neuron_id, level, age, strength)
			SELECT to_neuron_id, level, 0, SUM(strength)
			FROM connection_inference
			GROUP BY to_neuron_id, level
		`);

		// Log predictions per level
		if (this.debug) {
			const [results] = await this.conn.query(`
				SELECT level, COUNT(DISTINCT neuron_id) as count
				FROM connection_inferred_neurons
				WHERE age = 0
				GROUP BY level
				ORDER BY level DESC
			`);
			for (const row of results)
				console.log(`Level ${row.level}: Predicted ${row.count} neurons for next frame (from connections)`);
		}
	}

	/**
	 * Handle inference for levels where the neuron is a peak by default (no neighbors to compare).
	 * When there's only one neuron at a level, it's automatically a peak.
	 * Returns count of default peak neurons.
	 */
	async handleSingleValueInference() {

		// Insert connections for levels where neuron is peak by default (no neighbors)
		await this.conn.query(`
			INSERT INTO connection_inference (level, connection_id, to_neuron_id, strength)
			SELECT ic.level, ic.connection_id, ic.to_neuron_id, ic.strength
			FROM inferred_connections ic
			WHERE ic.level IN (
				SELECT level
				FROM inferred_neuron_strengths
				GROUP BY level
				HAVING COUNT(*) = 1
			)
		`);

		// Count default peak neurons
		const [result] = await this.conn.query(`
			SELECT COUNT(DISTINCT to_neuron_id) as neuron_count
			FROM connection_inference
		`);

		const defaultPeakCount = result[0].neuron_count;
		if (this.debug && defaultPeakCount > 0)
			console.log(`Found ${defaultPeakCount} default peak(s) (no neighbors)`);

		return defaultPeakCount;
	}

	/**
	 * Handle multi-value inference using peak detection.
	 * Returns number of peaks found.
	 */
	async handleMultiValueInference() {

		// Calculate average strength per level
		await this.getInferredLevelStrengths();

		// Get inferred peaks
		const peakCount = await this.getInferredPeaks();

		// Populate connection_inference using inferred_peaks as filter
		if (peakCount > 0) await this.populateConnectionInference();

		return peakCount;
	}

	/**
	 * Materialize candidate connections from active neurons.
	 * Truncates and populates inferred_connections scratch table.
	 */
	async getInferredConnections() {
		// Clear and populate inferred_connections
		await this.conn.query('TRUNCATE inferred_connections');
		await this.conn.query(`
			INSERT INTO inferred_connections (level, connection_id, from_neuron_id, to_neuron_id, strength)
			SELECT
				f.level,
				c.id as connection_id,
				c.from_neuron_id,
				c.to_neuron_id,
				c.strength * POW(?, c.distance) as strength
			FROM active_neurons f
			JOIN connections c ON c.from_neuron_id = f.neuron_id
			WHERE c.distance = f.age + 1
			AND c.strength > 0
		`, [this.peakTimeDecayFactor]);
	}

	/**
	 * Aggregate connection strengths per neuron per level.
	 * Truncates and populates inferred_neuron_strengths scratch table.
	 */
	async getInferredNeuronStrengths() {
		// Clear and populate inferred_neuron_strengths
		await this.conn.query('TRUNCATE inferred_neuron_strengths');
		await this.conn.query(`
			INSERT INTO inferred_neuron_strengths (level, to_neuron_id, total_strength)
			SELECT level, to_neuron_id, SUM(strength) as total_strength
			FROM inferred_connections
			GROUP BY level, to_neuron_id
		`);
	}

	/**
	 * Calculate average strength per level.
	 * Truncates and populates inferred_level_strengths scratch table.
	 */
	async getInferredLevelStrengths() {
		// Clear and populate inferred_level_strengths
		await this.conn.query('TRUNCATE inferred_level_strengths');
		await this.conn.query(`
			INSERT INTO inferred_level_strengths (level, avg_strength)
			SELECT level, AVG(total_strength) as avg_strength
			FROM inferred_neuron_strengths
			GROUP BY level
		`);
	}

	/**
	 * Filter neurons for peaks based on strength thresholds per level.
	 * Truncates and populates inferred_peaks scratch table.
	 * Returns the total number of peaks found across all levels.
	 */
	async getInferredPeaks() {

		// Clear inferred_peaks
		await this.conn.query('TRUNCATE inferred_peaks');

		// Insert peaks using average strength per level from inferred_level_strengths
		await this.conn.query(`
			INSERT INTO inferred_peaks (level, peak_neuron_id, total_strength)
			SELECT ins.level, ins.to_neuron_id, ins.total_strength
			FROM inferred_neuron_strengths ins
			JOIN inferred_level_strengths ils ON ins.level = ils.level
			WHERE ins.total_strength >= ?
			AND ins.total_strength > (ils.avg_strength * ?)
		`, [this.minPeakStrength, this.minPeakRatio]);

		// Get count for logging
		const [result] = await this.conn.query('SELECT COUNT(*) as peak_count FROM inferred_peaks');
		const totalPeaks = result[0].peak_count;
		if (this.debug) console.log(`Found ${totalPeaks} peak predictions across all levels`);

		return totalPeaks;
	}

	/**
	 * Populate connection_inference table using inferred_peaks as filter.
	 */
	async populateConnectionInference() {
		await this.conn.query(`
			INSERT INTO connection_inference (level, connection_id, to_neuron_id, strength)
			SELECT ic.level, ic.connection_id, ic.to_neuron_id, ic.strength
			FROM inferred_connections ic
			JOIN inferred_peaks ip ON ic.level = ip.level AND ic.to_neuron_id = ip.peak_neuron_id
		`);
	}

	/**
	 * Pattern inference: Cascade predictions down all levels using recursive query.
	 * For each predicted neuron that is a pattern neuron, predict its peak at the level below.
	 * This continues recursively until we reach level 0.
	 */
	async inferPatterns() {

		// Use recursive CTE to cascade pattern predictions down all levels
		await this.conn.query(`
			INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
			WITH RECURSIVE pattern_cascade AS (
				-- Base case: For each level, check if predicted neurons are pattern neurons
				SELECT
					pp.peak_neuron_id as neuron_id,
					cin.level - 1 as level,
					0 as age,
					cin.strength
				FROM connection_inferred_neurons cin
				JOIN pattern_peaks pp ON cin.neuron_id = pp.pattern_neuron_id
				WHERE cin.age = 0
				AND cin.level > 0

				UNION ALL

				-- Recursive case: Pattern predictions can themselves be pattern neurons
				SELECT
					pp.peak_neuron_id as neuron_id,
					pc.level - 1 as level,
					0 as age,
					pc.strength
				FROM pattern_cascade pc
				JOIN pattern_peaks pp ON pc.neuron_id = pp.pattern_neuron_id
				WHERE pc.level > 0
			)
			SELECT neuron_id, level, age, SUM(strength) as strength
			FROM pattern_cascade
			GROUP BY neuron_id, level, age
		`);

		// Log predictions per level
		const [results] = await this.conn.query(`
			SELECT level, COUNT(DISTINCT neuron_id) as count
			FROM pattern_inferred_neurons
			WHERE age = 0
			GROUP BY level
			ORDER BY level DESC
		`);

		// Log in the format: "Level X: Predicted Y peak neurons for level X-1 (from patterns)"
		// We need to map the predictions to their source level
		const predictionsBySourceLevel = new Map();
		for (const row of results) {
			const sourceLevel = row.level + 1;
			predictionsBySourceLevel.set(sourceLevel, row.count);
		}

		// Get max level from connection_inferred_neurons to know the range
		const [maxLevelResult] = await this.conn.query('SELECT MAX(level) as max_level FROM connection_inferred_neurons');
		const maxLevel = maxLevelResult[0].max_level;

		// Log for each source level
		if (this.debug && maxLevel !== null)
			for (let level = maxLevel; level > 0; level--) {
				const count = predictionsBySourceLevel.get(level) || 0;
				console.log(`Level ${level}: Predicted ${count} peak neurons for level ${level - 1} (from patterns)`);
			}
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame
	 */
	async getFrameNeurons(frame) {

		// try to get all the neurons that have coordinates in close ranges for each point - return format: [{ point_str, neuron_id }]
		const matches = await this.matchFrameNeurons(frame);
		if (this.debug) console.log('pointNeuronMatches', matches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const neuronIds = matches.filter(p => p.neuron_id).map(p => p.neuron_id);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => !p.neuron_id);
		if (pointsNeedingNeurons.length > 0) {
			if (this.debug) console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with dedupe.`);
			const createdNeuronIds = await this.createBaseNeurons(pointsNeedingNeurons.map(p => p.point_str));
			neuronIds.push(...createdNeuronIds);
		}

		// return matching neuron ids to given points
		if (this.debug) console.log('frame neurons', neuronIds);
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
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// Detect peaks and write to observed_patterns table
		const hasPeaks = await this.getObservedPatterns(level);
		if (!hasPeaks) {
			if (this.debug) console.log(`No peaks found at level ${level}`);
			return false;
		}

		// Match observed patterns to known patterns and write to matched_patterns table
		await this.matchObservedPatterns();

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
	 * Materialize weighted connection data from active connections.
	 * Truncates and populates observed_connections scratch table.
	 * @param {number} level - The level to get connections for
	 */
	async getObservedConnections(level) {
		// Clear and populate observed_connections
		await this.conn.query('TRUNCATE observed_connections');
		await this.conn.query(`
			INSERT INTO observed_connections (to_neuron_id, connection_id, strength)
			SELECT ac.to_neuron_id, ac.connection_id, c.strength * POW(?, c.distance) as strength
			FROM active_connections ac
			JOIN connections c ON ac.connection_id = c.id
			WHERE ac.level = ?
			AND ac.age = 0
			AND c.strength > 0
		`, [this.peakTimeDecayFactor, level]);
	}

	/**
	 * Aggregate connection strengths per neuron.
	 * Truncates and populates observed_neuron_strengths scratch table.
	 */
	async getObservedNeuronStrengths() {
		// Clear and populate observed_neuron_strengths
		await this.conn.query('TRUNCATE observed_neuron_strengths');
		await this.conn.query(`
			INSERT INTO observed_neuron_strengths (to_neuron_id, total_strength, connection_count)
			SELECT to_neuron_id, SUM(strength) as total_strength, COUNT(*) as connection_count
			FROM observed_connections
			GROUP BY to_neuron_id
		`);
	}

	/**
	 * Filter neurons for peaks based on strength thresholds.
	 * Truncates and populates observed_peaks scratch table.
	 * returns the number of peaks found.
	 */
	async getObservedPeaks() {

		// Calculate average strength threshold
		const [statsResult] = await this.conn.query(`
			SELECT SUM(total_strength) as sum_strength, COUNT(*) as neuron_count 
			FROM observed_neuron_strengths
		`);
		const sumStrength = statsResult[0].sum_strength || 0;
		const neuronCount = statsResult[0].neuron_count || 0;

		// Clear and populate observed_peaks
		await this.conn.query('TRUNCATE observed_peaks');

		// no peaks possible if there are no observed neurons
		if (neuronCount === 0) return 0;

		// If only one neuron, set threshold to 0 (always include it as a peak)
		// Otherwise, calculate threshold based on average
		// const strengthThreshold = neuronCount <= 1 ? 0 : ((sumStrength / neuronCount) * this.minPeakRatio);
		const strengthThreshold = (sumStrength / neuronCount) * this.minPeakRatio;

		// populate the observed peaks
		await this.conn.query(`
			INSERT INTO observed_peaks (peak_neuron_id, total_strength, connection_count)
			SELECT to_neuron_id, total_strength, connection_count
			FROM observed_neuron_strengths
			WHERE total_strength >= ?
			AND total_strength > ?
			AND connection_count >= 2
		`, [this.minPeakStrength, strengthThreshold]);

		// Get count for logging
		const [result] = await this.conn.query('SELECT COUNT(*) as peak_count FROM observed_peaks');
		return result[0].peak_count;
	}

	/**
	 * Detect peaks and write directly to observed_peaks and observed_patterns tables
	 * Peaks are to_neurons whose strength exceeds their source neurons' average strength
	 * Uses scratch tables with indexes instead of CTEs for better performance
	 * Strategy: Materialize data once, aggregate once, then filter
	 * @param {number} level - The level to detect peaks for
	 */
	async getObservedPatterns(level) {
		if (this.debug) console.log('getting observed patterns');

		// Step 1: Materialize weighted connection data
		await this.getObservedConnections(level);

		// Step 2: Aggregate per neuron
		await this.getObservedNeuronStrengths();

		// Step 3: get observed peaks - if none found, return false, indicating no patterns found
		const peakCount = await this.getObservedPeaks();
		if (this.debug) console.log(`Found ${peakCount} peaks at level ${level}`);
		if (peakCount === 0) return false;

		// Step 4: Populate observed_patterns using observed_peaks as filter
		await this.conn.query('TRUNCATE observed_patterns');
		await this.conn.query(`
			INSERT INTO observed_patterns (peak_neuron_id, connection_id)
			SELECT oc.to_neuron_id, oc.connection_id
			FROM observed_connections oc
			JOIN observed_peaks opk ON oc.to_neuron_id = opk.peak_neuron_id
		`);

		return true;
	}

	/**
	 * Match observed patterns to known patterns owned by the peak neuron.
	 * Writes results to matched_patterns and matched_peaks memory tables.
	 * Each peak neuron only reviews patterns it learned before (via pattern_peaks table).
	 * Matches by connection_id (which encodes from_neuron + to_neuron + distance) to preserve temporal structure.
	 * Uses connection overlap (66% threshold) to determine if patterns match.
	 */
	async matchObservedPatterns() {
		if (this.debug) console.log('Matching observed patterns to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_peaks');
		await this.conn.query('TRUNCATE matched_patterns');

		// Find matching patterns and insert into matched_patterns
		// find matching patterns for each observed patterns by the peak
		// Calculate overlap percentage and return matching peak-pattern pairs
		// at least 66% of the known pattern's connections should be part of the observed pattern to be matched
		// Use observed_peaks for fast existence check instead of scanning observed_patterns
		await this.conn.query(`
			INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
			SELECT pp.peak_neuron_id, pp.pattern_neuron_id
			FROM pattern_peaks pp
			JOIN observed_peaks opk ON pp.peak_neuron_id = opk.peak_neuron_id
			JOIN patterns p ON pp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN observed_patterns op ON pp.peak_neuron_id = op.peak_neuron_id AND op.connection_id = p.connection_id
			GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
			HAVING COUNT(DISTINCT CASE WHEN op.connection_id IS NOT NULL THEN p.connection_id END) >= COUNT(DISTINCT p.connection_id) * ?
		`, [this.mergePatternThreshold]);

		// Populate matched_peaks with distinct peaks that have matches
		await this.conn.query(`
			INSERT INTO matched_peaks (peak_neuron_id)
			SELECT DISTINCT peak_neuron_id
			FROM matched_patterns
		`);

		// Get count for logging
		const [result] = await this.conn.query('SELECT COUNT(*) as match_count FROM matched_patterns');
		if (this.debug) console.log(`Matched ${result[0].match_count} pattern-peak pairs`);
	}

	/**
	 * Merge matched patterns with observed patterns using pure SQL:
	 * 1. Add new connections that weren't in the pattern before
	 * 2. Strengthen connections that were observed (positive reinforcement)
	 * 3. Weaken connections that were NOT observed (negative reinforcement)
	 */
	async mergeMatchedPatterns() {
		if (this.debug) console.log('merging matched patterns...');

		// Positive reinforcement: Add/strengthen observed connections (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			INSERT INTO patterns (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT mp.pattern_neuron_id, op.connection_id, 1
			FROM matched_patterns mp
			JOIN observed_patterns op ON mp.peak_neuron_id = op.peak_neuron_id
			ON DUPLICATE KEY UPDATE strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, strength + 1))
		`, { minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });

		// Negative reinforcement: Weaken unobserved connections (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE patterns p
			JOIN matched_patterns mp ON p.pattern_neuron_id = mp.pattern_neuron_id
			SET p.strength = GREATEST(?, LEAST(?, p.strength - ?))
            WHERE NOT EXISTS (
                SELECT 1
                FROM observed_patterns op
                WHERE op.peak_neuron_id = mp.peak_neuron_id
                  AND op.connection_id = p.connection_id
            )			
		`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
	}

	/**
	 * Create new pattern neurons for peaks that don't have any matching patterns.
	 * Leverages MySQL's sequential auto-increment IDs to map peaks to new pattern neurons.
	 * No scratch table needed - pattern_peaks table establishes the mapping directly.
	 */
	async createNewPatterns() {
		if (this.debug) console.log('creating new patterns...');

		// Find peaks that need new patterns (peaks in observed_peaks but not in matched_peaks)
		// Use matched_peaks for fast indexed lookup instead of scanning matched_patterns
		const [peaksNeedingPatterns] = await this.conn.query(`
			SELECT opk.peak_neuron_id
			FROM observed_peaks opk
			WHERE NOT EXISTS (
				SELECT 1
				FROM matched_peaks mpk
				WHERE mpk.peak_neuron_id = opk.peak_neuron_id
			)
		`);
		const count = peaksNeedingPatterns.length;
		if (this.debug) console.log(`Creating ${count} new patterns for peaks without matches`);
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
			console.log('Running forget cycle - pattern update...');
			let stepStart = Date.now();
			const [patternUpdateResult] = await this.conn.query(`UPDATE patterns SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
			console.log(`  Pattern UPDATE took ${Date.now() - stepStart}ms (updated ${patternUpdateResult.affectedRows} rows)`);

			// Delete patterns with zero strength
			console.log('Running forget cycle - pattern deletion...');
			stepStart = Date.now();
			const [patternDeleteResult] = await this.conn.query(`DELETE FROM patterns WHERE strength = ?`, [this.minConnectionStrength]);
			console.log(`  Pattern DELETE took ${Date.now() - stepStart}ms (deleted ${patternDeleteResult.affectedRows} rows)`);

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
		} else {
			// Run forget cycle without logging
			await this.conn.query(`UPDATE patterns SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.patternForgetRate]);
			await this.conn.query(`DELETE FROM patterns WHERE strength = ?`, [this.minConnectionStrength]);
			await this.conn.query(`UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE strength > 0`, [this.minConnectionStrength, this.maxConnectionStrength, this.connectionForgetRate]);
			await this.conn.query(`DELETE FROM connections WHERE strength = ?`, [this.minConnectionStrength]);
			await this.conn.query(`
				DELETE pp FROM pattern_peaks pp
				LEFT JOIN patterns p ON pp.pattern_neuron_id = p.pattern_neuron_id
				WHERE p.pattern_neuron_id IS NULL
			`);
			await this.conn.query(`
				DELETE FROM neurons n
				WHERE NOT EXISTS (SELECT 1 FROM connections WHERE from_neuron_id = n.id)
				  AND NOT EXISTS (SELECT 1 FROM connections WHERE to_neuron_id = n.id)
				  AND NOT EXISTS (SELECT 1 FROM patterns WHERE pattern_neuron_id = n.id)
				  AND NOT EXISTS (SELECT 1 FROM active_neurons WHERE neuron_id = n.id)
			`);
		}
	}

	/**
	 * Apply global reward to active connections that led to executed outputs.
	 * Strengthens connections for positive rewards, weakens for negative rewards.
	 * Uses exponential temporal decay - older connections get less reward/punishment.
	 */
	async applyRewards(globalReward) {
		if (globalReward === 1.0) {
			if (this.debug) console.log('Neutral global reward - no updates needed');
			return;
		}

		// Calculate reward adjustment: positive reward strengthens, negative weakens
		// globalReward = 1.5 → adjustment = +0.5 per connection
		// globalReward = 0.5 → adjustment = -0.5 per connection
		const rewardAdjustment = globalReward - 1.0;

		// Apply reward to active_connections with exponential temporal decay (clamped between minConnectionStrength and maxConnectionStrength)
		// Older connections (higher age) get less reward/punishment
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN active_connections ac ON c.id = ac.connection_id
			SET c.strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, c.strength + (:rewardAdjustment * POW(:rewardTimeDecayFactor, ac.age))))
			WHERE ac.age >= 0
		`, { rewardAdjustment, rewardTimeDecayFactor: this.rewardTimeDecayFactor, minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });

		if (this.debug) console.log(`Applied global reward ${globalReward.toFixed(3)} to ${result.affectedRows} active connections (adjustment: ${rewardAdjustment >= 0 ? '+' : ''}${rewardAdjustment.toFixed(3)})`);
	}
}