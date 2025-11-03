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
		this.maxLevels = 10; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.50; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.minPeakStrength = 1.0; // minimum weighted strength for a neuron to be considered a peak (reduced to allow more predictions)
		this.minPeakRatio = 1.0; // minimum ratio of peak strength to neighborhood average (1.0 = just above average)
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.9; // reward temporal decay = POW(rewardTimeDecayFactor, age)
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.negativeLearningRate = 1.0; // how much to weaken connections when predictions fail (match positive reinforcement)
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
		this.debug = true;
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
		if (this.waitForUserInput) await this.waitForUser('Press Enter to continue to next frame');
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

		// Write selected predictions - implementation-specific
		await this.writeResolvedPredictions(allSelectedPredictions);

		if (this.debug) console.log(`Resolved ${allSelectedPredictions.length} input predictions after conflict resolution`);
	}

	/**
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {
		// Get base level (level 0) accuracy - use RESOLVED accuracy (after conflict resolution)
		let baseAccuracy = 'N/A';
		const baseCumulative = this.accuracyStats.get(0);
		if (baseCumulative && baseCumulative.resolved.total > 0)
			baseAccuracy = `${(baseCumulative.resolved.correct / baseCumulative.resolved.total * 100).toFixed(1)}%`;

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

		// Show connection count every 100 frames in memory implementation
		let memoryStats = '';
		if (this.connections && this.connections.size && this.frameNumber % 100 === 0) {
			memoryStats = ` | Learned: ${this.connections.size()} conns, ${this.patterns ? this.patterns.size() : 0} patterns`;
		}

		console.log(`Frame ${this.frameNumber} | Base: ${baseAccuracy} | Higher: ${higherAccuracy} | MAPE: ${mapeDisplay} | Time: ${frameElapsed.toFixed(2)}ms${memoryStats}`);
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
	 * Execute decisions from previous frame (age = 1)
	 */
	async executePreviousOutputs() {
		throw new Error('executePreviousOutputs() must be implemented by subclass');
	}

	/**
	 * Recognizes and activates neurons from frame - returns the highest level of recognition reached
	 * Common implementation for both MySQL and Memory backends
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
	 * Infer predictions and outputs
	 */
	async inferNeurons() {
		throw new Error('inferNeurons() must be implemented by subclass');
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
	async applyRewards(globalReward) {
		throw new Error('applyRewards() must be implemented by subclass');
	}

	/**
	 * Write resolved predictions to storage (implementation-specific)
	 */
	async writeResolvedPredictions(allSelectedPredictions) {
		throw new Error('writeResolvedPredictions() must be implemented by subclass');
	}

	/**
	 * Get or create frame neurons from input points
	 */
	async getFrameNeurons(frame) {
		throw new Error('getFrameNeurons() must be implemented by subclass');
	}

	/**
	 * Activate neurons at base level (level 0)
	 */
	async activateNeurons(neuronIds) {
		throw new Error('activateNeurons() must be implemented by subclass');
	}

	/**
	 * Activate pattern neurons hierarchically
	 */
	async activatePatternNeurons() {
		throw new Error('activatePatternNeurons() must be implemented by subclass');
	}
}

