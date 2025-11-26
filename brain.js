import readline from 'node:readline';
import getMySQLConnection from './db/db.js';

/**
 * Base Brain Class - Common functionality for all brain implementations
 * Handles: channels, dimensions, feedback, execution, orchestration
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.baseNeuronMaxAge = 5; // number of frames a base neuron stays active
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle (reduced to preserve learned connections)
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle
		this.rewardForgetRate = 0.05; // how much reward factors decay toward 1.0 per forget cycle (0.05 = 5% decay toward neutral)
		this.maxLevels = 10; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.inactivityThreshold = 0; // frames of inactivity before exploration - require activity in every frame
		this.minPredictionStrength = 10.0; // minimum strength for a prediction to be made
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.9; // reward temporal decay = POW(rewardTimeDecayFactor, age)
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.connectionNegativeReinforcement = 1.0; // how much to weaken connections when predictions fail
		this.minErrorPatternThreshold = 5.0; // minimum prediction strength to create error-driven pattern
		this.minConnectionStrength = 0; // minimum strength value for connections and patterns (clamped to prevent negative values)
		this.maxConnectionStrength = 1000; // maximum strength value for connections and patterns (clamped to prevent overflow)
		this.maxConnectionReward = 10.0; // maximum reward factor for connections and patterns (clamped to prevent extreme values)
		this.maxRewardsAge = 1; // how far back in time to apply rewards (1 = only most recent outputs)

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;

		// Prediction accuracy tracking (cumulative stats per level)
		this.accuracyStats = new Map(); // level -> {correct, total}

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

		// Track last frame's inference for error-driven learning
		this.lastInferenceType = null; // 'connection' or 'pattern' or null
		this.lastInferenceLevel = null; // level where inference was made

		// Create readline interface for pausing between frames - used when debugging
		this.debug = false;
		this.waitForUserInput = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
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
		if (this.debug) console.log('Initializing dimensions for registered channels...');
		for (const [channelName, channel] of this.channels) {
			await this.insertChannelDimensions(channel.getInputDimensions(), channelName, 'input');
			await this.insertChannelDimensions(channel.getOutputDimensions(), channelName, 'output');
		}
	}

	/**
	 * inserts channel dimensions
	 */
	async insertChannelDimensions(dimensions, channelName, type){
		if (this.debug) console.log(`Creating ${type} dimensions for ${channelName}:`, dimensions);
		for (const dimName of dimensions)
			await this.conn.query('INSERT IGNORE INTO dimensions (name, channel, type) VALUES (?, ?, ?)', [dimName, channelName, type]);
	}

	/**
	 * loads the dimensions to memory with full info (id, name, channel, type)
	 */
	async loadDimensions() {
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};
		this.dimensions = new Map(); // Map<dimension_name, {id, channel, type}>

		const [rows] = await this.conn.query('SELECT id, name, channel, type FROM dimensions');
		rows.forEach(row => {
			this.dimensionNameToId[row.name] = row.id;
			this.dimensionIdToName[row.id] = row.name;
			this.dimensions.set(row.name, { id: row.id, channel: row.channel, type: row.type });
		});
		if (this.debug) console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * returns the current frame combined from all registered channels
	 */
	async getFrame() {
		const frame = [];

		// Increment frame counter to be able to track inactivity
		this.frameNumber++;

		// Get input and output data from all channels
		for (const [_, channel] of this.channels) {

			// get the frame inputs from the channel
			const channelInputs = await channel.getFrameInputs();
			if (channelInputs && channelInputs.length > 0) frame.push(...channelInputs);

			// get last inferred outputs to be executed in this frame in case there are any
			const channelOutputs = await channel.getFrameOutputs();
			if (channelOutputs && channelOutputs.length > 0) frame.push(...channelOutputs);
		}

		return frame;
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getFeedback() {
		if (this.debug) console.log('Getting feedback from all channels...');
		const channelRewards = new Map();
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const rewardFactor = await channel.getFeedback();
			if (rewardFactor !== 1.0) { // Only process non-neutral feedback
				if (this.debug) console.log(`${channelName}: reward factor ${rewardFactor.toFixed(3)}`);
				channelRewards.set(channelName, rewardFactor);
				feedbackCount++;
			}
		}

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Received feedback from ${feedbackCount} channels`);
			else console.log('No feedback from any channels');
		}

		return channelRewards;
	}

	/**
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and channel-specific rewards (Map of channel_name -> reward)
	 */
	async processFrame(frame, channelRewards = new Map()) {
		const frameStart = performance.now();
		if (this.debug) {
			console.log('******************************************************************');
			console.log(`OBSERVING NEW FRAME: ${JSON.stringify(frame)}`, this.frameNumber);
			console.log(`Channel rewards:`, Array.from(channelRewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));
			console.log('******************************************************************');
		}

		// apply rewards to previously executed decisions (before aging them further)
		await this.applyRewards(channelRewards);

		// age the active neurons in memory context - sliding the temporal window
		await this.ageNeurons();

		// execute previous frame's decisions + exploration if needed
		await this.executeOutputs(frame);

		// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
		await this.recognizeNeurons(frame);

		// learn from previous frame's prediction errors - this has to be done after recognizing and activating new neurons
		await this.learnFromPreviousFrameErrors();

		// do predictions and outputs - what's going to happen next? and what's our best response?
		await this.inferNeurons();

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// run forget cycle periodically and delete dead connections/neurons
		await this.runForgetCycle();

		// show frame processing summary
		const frameElapsed = performance.now() - frameStart;
		this.printFrameSummary(frameElapsed);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');
	}

	/**
	 * waits for user input to continue - used for debugging
	 */
	waitForUser(message) {
		if (!this.waitForUserInput) return Promise.resolve();
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Execute previous frame's decisions and exploration actions if needed
	 */
	async executeOutputs(frame) {

		// Execute previous frame's decisions (output neurons in current frame)
		await this.executeInferredActions(frame);

		// Execute exploration if brain is inactive
		await this.curiosityExploration(frame);
	}

	/**
	 * Execute decisions from previous frame (age = 1)
	 * Reads output neurons from the frame instead of querying database
	 */
	async executeInferredActions(frame){

		// Get channel outputs from frame
		const channelOutputs = this.getFrameChannelOutputs(frame);
		if (this.debug) console.log('channelOutputs', channelOutputs);

		// nothing to do if there are no outputs
		if (channelOutputs.size === 0) {
			if (this.debug) console.log('No inferred outputs to execute');
			return;
		}

		// execute outputs for each channel
		if (this.debug) console.log(`Executing ${channelOutputs.size} inferred outputs`, channelOutputs);
		for (const [channelName, coordinates] of channelOutputs)
			await this.executeChannelOutputs(channelName, coordinates);
	}

	/**
	 * Execute curiosity exploration for channels without inferred outputs
	 */
	async curiosityExploration(frame){

		// Check if the brain is inactive - if active, no exploration needed
		if ((this.frameNumber - this.lastActivity) < this.inactivityThreshold) return;

		// Get channel outputs from frame
		const channelOutputs = this.getFrameChannelOutputs(frame);

		// Explore channels without outputs
		let exploredCount = 0;
		for (const [channelName, channel] of this.channels) {

			// Skip if channel doesn't have output dimensions
			const channelOutputDims = channel.getOutputDimensions();
			if (channelOutputDims.length === 0) continue;

			// Skip if this channel already has output in frame
			if (channelOutputs.has(channelName)) continue;

			// Execute exploration for this channel
			const validActions = channel.getValidExplorationActions();
			if (validActions.length === 0) continue;

			// pick a random action from the valid ones
			const randomAction = validActions[Math.floor(Math.random() * validActions.length)];
			if (this.debug) console.log(`${channelName}: Curiosity exploration:`, randomAction);

			// execute the selected actions
			await this.executeChannelOutputs(channelName, randomAction);

			// add explored action to frame so it gets recognized as a neuron with coordinates
			frame.push(randomAction);
			exploredCount++;
		}

		if (this.debug && exploredCount > 0) console.log(`Explored ${exploredCount} channels without predictions`);
	}

	/**
	 * returns channel outputs from frame, grouped by channel
	 * @param {Array} frame - frame data points
	 * @returns {Map} - Map of channel name to coordinates object (empty object if channel has output but no values yet)
	 */
	getFrameChannelOutputs(frame) {
		const outputDimToChannel = this.getOutputDimensionToChannelMap();
		const channelOutputs = new Map();
		for (const point of frame)
			for (const [dim, value] of Object.entries(point))
				if (outputDimToChannel.has(dim)) {
					const channelName = outputDimToChannel.get(dim);
					if (!channelOutputs.has(channelName)) channelOutputs.set(channelName, {});
					channelOutputs.get(channelName)[dim] = value;
				}
		return channelOutputs;
	}

	/**
	 * Get mapping of output dimension names to channel names
	 * @returns {Map} - Map of dimension name to channel name
	 */
	getOutputDimensionToChannelMap() {
		const outputDimToChannel = new Map();
		for (const [channelName, channel] of this.channels)
			for (const dim of channel.getOutputDimensions())
				outputDimToChannel.set(dim, channelName);
		return outputDimToChannel;
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

		// get the channel object
		const channel = this.channels.get(channelName);
		if (!channel) {
			if (this.debug) console.log(`Warning: Channel ${channelName} not found`);
			return;
		}

		// execute outputs in the channel
		if (this.debug) console.log(`${channelName}: Executing outputs:`, coordinates);
		await channel.executeOutputs(coordinates);

		// Track global activity
		this.lastActivity = this.frameNumber;
	}

	/**
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {

		// Get base level (level 0) accuracy - use RESOLVED accuracy (after conflict resolution)
		let baseAccuracy = 'N/A';
		const baseCumulative = this.accuracyStats.get(0);
		if (baseCumulative && baseCumulative.total > 0)
			baseAccuracy = `${(baseCumulative.correct / baseCumulative.total * 100).toFixed(1)}%`;

		// Get higher level accuracy (aggregate all levels > 0)
		// Include both connection and pattern predictions
		let higherCorrect = 0;
		let higherTotal = 0;
		for (const [level, stats] of this.accuracyStats.entries())
			if (level > 0) {
				higherCorrect += stats.correct;
				higherTotal += stats.total;
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

		console.log(`Frame ${this.frameNumber} | Base: ${baseAccuracy} | Higher: ${higherAccuracy} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}

	// ========== ABSTRACT METHODS - Must be implemented by subclasses ==========

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		throw new Error('resetContext() must be implemented by subclass');
	}

	/**
	 * Reset accuracy stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats.clear();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 */
	async resetBrain() {
		throw new Error('resetBrain() must be implemented by subclass');
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 */
	async ageNeurons() {
		throw new Error('ageNeurons() must be implemented by subclass');
	}

	/**
	 * Get the highest level with active neurons (any age).
	 * Called during inference to determine iteration range.
	 * Implementation-specific: MySQL queries active_neurons table, Memory scans in-memory structure.
	 */
	async getMaxActiveLevel() {
		throw new Error('getMaxActiveLevel() must be implemented by subclass');
	}

	/**
	 * Recognizes and activates neurons from frame
	 * Common implementation for both MySQL and Memory backends
	 */
	async recognizeNeurons(frame) {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(frame);

		// bulk insert activations at base level
		await this.activateNeurons(neuronIds);

		// discover and activate patterns using connections - start recursion from base level
		await this.recognizePatternNeurons();
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
	 * creates base neurons from a given set of points and returns their ids
	 */
	async createBaseNeurons(pointStrs) {

		// nothing to create if no points are sent - should not happen
		if (pointStrs.length === 0) return [];

		// deduplicate points and parse them
		const points = [...new Set(pointStrs)].map(pointStr => JSON.parse(pointStr));

		// bulk insert neurons and get their IDs
		const neuronIds = await this.createNeurons(points.length);
		const created = points.map((point, idx) => ({ point, neuron_id: neuronIds[idx] }));

		// insert coordinates for the created neurons
		await this.setNeuronCoordinates(created);

		// return the new neuron ids
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
	 * detects all spatial levels in age=0 neurons using unified connections - start from base level and go as high as possible
	 */
	async recognizePatternNeurons() {
		let level = 0;
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
	 * Infer predictions and outputs
	 * Sequential inference with error-driven learning.
	 * Alternates between connection and pattern inference going down levels:
	 * connection@N, pattern@N-1, connection@N-1, pattern@N-2, ..., connection@0
	 * Note: inferred_neurons ages and gets cleaned up (age >= 2), no need to truncate
	 */
	async inferNeurons() {

		// Get max active level after recognition (includes all ages, not just age=0)
		const maxActiveLevel = await this.getMaxActiveLevel();

		// start from top level and alternate between connection and pattern inference, going down levels
		// stop at first successful inference (early return pattern)
		for (let level = maxActiveLevel; level >= 0; level--) {

			// Try connection inference at this level (gradual Hebbian learning)
			if (await this.inferConnections(level)) return;

			// Try pattern inference from this level (fast context-based override)
			// Patterns predict at level-1, so skip if already at base level
			if (level > 0 && await this.inferPatterns(level)) return;
		}

		// if we got to this level, it looks like there were no predictions - should be rare
		if (this.debug) console.log('No predictions at any level');
	}

	/**
	 * runs the forget cycle
	 */
	async runForgetCycle() {
		throw new Error('runForgetCycle() must be implemented by subclass');
	}

	/**
	 * Apply global reward to active connections
	 */
	async applyRewards() {
		throw new Error('applyRewards() must be implemented by subclass');
	}

	/**
	 * Learn from previous frame's predictions by comparing them to current observations.
	 * Called at start of new frame after observations are available.
	 * Uses cached inference type and level from previous frame.
	 */
	async learnFromPreviousFrameErrors() {

		// First frame - no previous predictions to learn from
		if (this.lastInferenceType === null) {
			if (this.debug) console.log('No previous predictions - first frame');
			return;
		}

		// Report accuracy from previous frame inference
		await this.reportPredictionsAccuracy();

		// if we had made the inference using connections, apply negative reinforcement to failed predictions
		// otherwise if we had made the inference using patterns, merge predictions with observations based on previous frame's inference type
		if (this.lastInferenceType === 'connection') await this.negativeReinforceConnections();
		else await this.mergePatternFuture();

		// now create patterns from failed predictions - error-driven pattern creation at level + 1
		await this.createErrorPatterns();
	}

	/**
	 * Primary inference mechanism using learned connections between neurons.
	 * Active neurons predict what should activate next based on connection strengths and temporal distances.
	 * Returns true if predictions were made, false if not.
	 */
	async inferConnections(level) {

		// Make predictions using connections from active neurons at this level
		const count = await this.inferConnectionsAtLevel(level);
		if (count === 0) return false;

		// Cache inference type and level for next frame's error learning
		this.lastInferenceType = 'connection';
		this.lastInferenceLevel = level;

		// If predictions are at higher level, unpack through pattern chain to base level
		await this.unpackToBase(level, 'connection');

		// Use channel logic to resolve conflicting inference at base level
		await this.resolveChannelInferenceConflicts();

		// For higher levels, copy predictions directly to resolved table (no conflict resolution at higher levels)
		if (level > 0) await this.copyHigherLevelPredictions();

		return true;
	}

	/**
	 * Override mechanism for fast adaptation to changing contexts.
	 * Patterns are meta-connections that capture full context (all connections to/from a peak).
	 * Connections learn gradually (Hebbian), patterns enable instant learning (error-driven).
	 * Returns true if predictions were made, false if not.
	 */
	async inferPatterns(level) {

		// Make predictions at level-1 using pattern_future from this level
		const count = await this.inferPatternsFromLevel(level);
		if (count === 0) return false;

		// Cache inference type and level for next frame's error learning
		this.lastInferenceType = 'pattern';
		this.lastInferenceLevel = level;

		// If predictions are at higher level, unpack through pattern chain to base level
		if (level > 0) await this.unpackToBase(level - 1, 'pattern');

		// Use channel logic to resolve conflicting predictions at base level
		await this.resolveChannelInferenceConflicts();

		// For higher levels, copy predictions directly to resolved table (no conflict resolution at higher levels)
		// Pattern predictions are at level-1, so copy if level-1 > 0
		if (level > 0) await this.copyHigherLevelPredictions();

		return true;
	}

	/**
	 * Resolve inference conflicts per channel (level 0 only).
	 * Reads from inferred_neurons table (which contains predictions from either connection or pattern inference).
	 */
	async resolveChannelInferenceConflicts() {

		// Read inferred base neurons from inferred_neurons table - if there are none, nothing to resolve
		const channelInferences = await this.getChannelInferences();
		if (channelInferences.size === 0) return;

		// Resolve conflicts for each channel and collect resolved inferences
		// Channels resolve conflicts and store final inferred neurons in their own memory as well to be executed in the next frame
		const resolvedInferences = [];
		for (const [channelName, channelInference] of channelInferences)
			resolvedInferences.push(...this.channels.get(channelName).resolveConflicts(channelInference));

		// Write selected predictions to be used later for accuracy reporting - implementation-specific
		await this.writeResolvedPredictions(resolvedInferences);

		if (this.debug) console.log(`Resolved ${resolvedInferences.length} neurons after conflict resolution`);
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Common orchestration logic with implementation-specific methods.
	 */
	async createErrorPatterns() {

		// For pattern inference: predictions were made at lastInferenceLevel - 1
		// For connection inference: predictions were made at lastInferenceLevel
		const predictionLevel = this.lastInferenceType === 'pattern' ? this.lastInferenceLevel - 1 : this.lastInferenceLevel;
		const newPatternLevel = predictionLevel + 1;

		// First check if we have failed predictions (surprising errors)
		const failedCount = await this.countFailedPredictions();
		if (this.debug) console.log(`Level ${this.lastInferenceLevel}: failed predictions count: ${failedCount}`);
		if (failedCount === 0) return;

		// We have errors, now find what we should have predicted instead
		const unpredictedCount = await this.populateUnpredictedConnections();
		if (this.debug) console.log(`Level ${this.lastInferenceLevel}: unpredicted connections count: ${unpredictedCount}`);
		if (unpredictedCount === 0) return;

		// Populate new_patterns table with peaks from unpredicted connections
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Level ${this.lastInferenceLevel}: Creating ${patternCount} error patterns at level ${newPatternLevel}`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons(patternCount);

		// Merge new patterns into pattern_peaks, pattern_past, pattern_future
		await this.mergeNewPatterns(predictionLevel);

		if (this.debug) console.log(`Level ${this.lastInferenceLevel}: Created ${patternCount} error patterns`);
	}

	/**
	 * Apply negative reinforcement to failed connection predictions (implementation-specific)
	 */
	async negativeReinforceConnections() {
		throw new Error('negativeReinforceConnections() must be implemented by subclass');
	}

	/**
	 * Merge pattern_future with observed connections (implementation-specific)
	 * Applies positive reinforcement, negative reinforcement, and adds novel connections
	 */
	async mergePatternFuture() {
		throw new Error('mergePatternFuture() must be implemented by subclass');
	}

	/**
	 * Report accuracy of predictions from previous frame (implementation-specific)
	 */
	async reportPredictionsAccuracy() {
		throw new Error('reportPredictionsAccuracy() must be implemented by subclass');
	}

	/**
	 * Connection inference at a specific level (implementation-specific)
	 * Returns count of predictions made.
	 */
	async inferConnectionsAtLevel() {
		throw new Error('inferConnectionsAtLevel() must be implemented by subclass');
	}

	/**
	 * Pattern inference from a source level (implementation-specific)
	 * Returns count of predictions made.
	 */
	async inferPatternsFromLevel() {
		throw new Error('inferPatternsFromLevel() must be implemented by subclass');
	}

	/**
	 * Unpack predictions from higher level to base level via peak chain (implementation-specific)
	 */
	async unpackToBase() {
		throw new Error('unpackToBase() must be implemented by subclass');
	}

	/**
	 * Get inferred	base neurons grouped by channels (implementation-specific)
	 */
	async getChannelInferences() {
		throw new Error('getChannelInferences() must be implemented by subclass');
	}

	/**
	 * Write resolved predictions to storage (implementation-specific)
	 */
	async writeResolvedPredictions() {
		throw new Error('writeResolvedPredictions() must be implemented by subclass');
	}

	/**
	 * Copy higher level predictions from inferred_neurons to inferred_neurons_resolved (implementation-specific)
	 * For levels > 0, there's no conflict resolution, so predictions are copied directly
	 */
	async copyHigherLevelPredictions() {
		throw new Error('copyHigherLevelPredictions() must be implemented by subclass');
	}

	/**
	 * fetches all neuron coordinates that could potentially match any point in the frame
	 */
	async getFrameCoordinates() {
		throw new Error('getFrameCoordinates(frame) must be implemented by subclass');
	}

	/**
	 * Sets coordinates for neurons in batches to avoid query size limits
	 */
	async setNeuronCoordinates() {
		throw new Error('setNeuronCoordinates(neurons) must be implemented by subclass');
	}

	/**
	 * Creates new neurons and return their IDs.
	 */
	async createNeurons() {
		throw new Error('createNeurons(count) must be implemented by subclass');
	}

	/**
	 * Activate neurons at base level (level 0)
	 */
	async activateNeurons() {
		throw new Error('activateNeurons() must be implemented by subclass');
	}

	/**
	 * Activate pattern neurons in a level
	 */
	async recognizeLevelPatterns() {
		throw new Error('recognizeLevelPatterns(level) must be implemented by subclass');
	}

	/**
	 * Count high-confidence failed predictions (implementation-specific)
	 * Returns the count of failed predictions.
	 */
	async countFailedPredictions() {
		throw new Error('countFailedPredictions() must be implemented by subclass');
	}

	/**
	 * Populate unpredicted_connections with active connections that were not predicted (implementation-specific)
	 * Returns the number of unpredicted connections found.
	 */
	async populateUnpredictedConnections() {
		throw new Error('populateUnpredictedConnections() must be implemented by subclass');
	}

	/**
	 * Populate new_patterns table from unpredicted connections (implementation-specific)
	 * Finds peak neurons (from_neurons of unpredicted connections) and creates one pattern per peak.
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		throw new Error('populateNewPatterns() must be implemented by subclass');
	}

	/**
	 * Create pattern neurons and map them to new_patterns (implementation-specific)
	 */
	async createPatternNeurons() {
		throw new Error('createPatternNeurons(patternCount) must be implemented by subclass');
	}

	/**
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future (implementation-specific)
	 */
	async mergeNewPatterns() {
		throw new Error('mergeNewPatterns(predictionLevel) must be implemented by subclass');
	}
}

