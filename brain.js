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

		//************************************************************
		// hyperparameters
		//************************************************************

		// structural limits for the brain
		this.baseNeuronMaxAge = 5; // number of frames a base neuron stays active - this determines the context length
		this.minConnectionStrength = 0; // minimum strength value for connections and patterns (clamped to prevent negative values)
		this.maxConnectionStrength = 100; // maximum strength value for connections and patterns (clamped to prevent overflow)

		// connection learning parameters
		this.connectionNegativeReinforcement = 0.25; // how much to weaken connections when predictions fail

		// pattern learning parameters
		this.predictionErrorMinStrength = 10.0; // minimum prediction strength to create error-driven pattern
		this.actionRegretMinStrength = 1; // minimum prediction strength to create action regret pattern (0 = always capture painful actions)
		this.actionRegretMinPain = 0; // minimum pain (negative reward magnitude) to create action regret pattern (0 = any negative reward triggers regret)
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.maxLevels = 10; // just to prevent against infinite recursion

		// inference parameters
		this.peakTimeDecayFactor = 0.75; // peak connection weight = POW(peakTimeDecayFactor, distance)

		// voting parameters - all levels vote for actions, higher levels have more temporal context
		this.levelVoteMultiplier = 1.2; // vote weight = strength * POW(levelVoteMultiplier, level)
		this.boltzmannTemperature = 0.1; // temperature for Boltzmann selection (lower = more aggressive, 1.0 = standard)

		// reward parameters
		this.maxRewardsAge = 1; // how far back in time to apply rewards (1 = only most recent outputs)
		this.rewardExpSmooth = 0.9; // exponential smoothing for rewards: new = smooth * observed + (1 - smooth) * old

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

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// Prediction accuracy tracking (cumulative stats per level)
		this.accuracyStats = new Map(); // level -> {correct, total}

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

		// Debugging info and flags
		this.frameNumber = 0;
		this.frameSummary = true; // show frame summary or not
		this.debug = false;
		this.debug2 = false; // deeper, more verbose debug level
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info

		// Create readline interface for pausing between frames - used when debugging
		this.waitForUserInput = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
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
	async init() {

		// get new database connection
		this.conn = await getMySQLConnection();

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
	 * returns the current frame combined from all registered channels
	 * Each frame point includes: coordinates, channel, type
	 */
	async getFrame() {
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
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and channel-specific rewards (Map of channel_name -> reward)
	 */
	async processFrame(frame) {
		const frameStart = performance.now();

		// Display diagnostic frame header if enabled
		if (this.diagnostic) this.displayFrameHeader(frame);

		// age the active neurons in memory context - sliding the temporal window
		await this.ageNeurons();

		// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
		await this.recognizeNeurons(frame);

		// populate inference sources for executed actions (age=1) using just-created connections
		await this.populateInferenceSources();

		// learn from base level: compare L0 inferences (age=1) to L0 observations (age=0)
		// - events: negative reinforcement for wrong predictions (strength)
		// - actions: apply rewards based on channel outcomes (reward)
		await this.learnFromBaseLevel();

		// learn from inference level: pattern-level learning
		// - mergePatternFuture: update pattern predictions based on observations
		// - createErrorPatterns: create new patterns from prediction errors
		await this.learnFromInferenceLevel();

		// do predictions and outputs - what's going to happen next? and what's our best response?
		await this.inferNeurons();

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// run forget cycle periodically and delete dead connections/neurons
		await this.runForgetCycle();

		// show frame processing summary
		this.printFrameSummary(performance.now() - frameStart);

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
	 * Explore a channel by asking for an unexplored action.
	 * The brain passes all voted actions for this channel (from connections/patterns).
	 * The channel returns an action that wasn't voted for (hasn't been tried from this context).
	 * @param {string} channelName - Channel name
	 * @param {Array} votedActions - All actions that received votes (have connections from current context)
	 * @returns {Promise<Object|null>} Exploration action or null if all actions already voted
	 */
	async exploreChannel(channelName, votedActions) {
		const channel = this.channels.get(channelName);

		// Filter to only actions with sufficient non-action votes
		// The votes from action neurons don't capture the situation (they just follow previous actions)
		// Only votes from event neurons count
		const exploredActions = votedActions.filter(a => {
			if (!a.sources) return false;
			const relevantVotes = a.sources.filter(s => s.from_type === 'event').length;
			return relevantVotes >= this.minExploredVotes;
		});

		// Extract coordinates from explored actions
		const votedCoordinates = exploredActions.map(a => a.coordinates).filter(c => c);

		// Ask channel for an action that wasn't voted for
		const actionCoordinates = channel.getExplorationAction(votedCoordinates);
		if (!actionCoordinates) {
			if (this.debug) console.log(`Exploration for ${channelName}: all actions already have votes`);
			return null; // All actions have been tried from this context
		}

		// Find or create neuron for this action - wrap coordinates in frame point structure with channel metadata
		const [actionNeuronId] = await this.getFrameNeurons([{
			coordinates: actionCoordinates,
			channel: channelName,
			channel_id: this.channelNameToId[channelName],
			type: 'action'
		}]);

		// Return exploration action - sources populated after execution in populateInferenceSources
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

		if (this.frameSummary) console.log(`Frame ${this.frameNumber} | Base: ${baseAccuracy} | Higher: ${higherAccuracy} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
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
	 * Recognizes and activates neurons from frame
	 * Common implementation for both MySQL and Memory backends
	 */
	async recognizeNeurons(frame) {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// bulk insert activations at base level
		await this.activateNeurons(neuronIds);

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
		if (this.debug2) console.log('pointNeuronMatches', matches);

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
	 * creates base neurons from a given set of frame points and returns their ids
	 * Frame points have structure: { coordinates, channel, channel_id, type }
	 */
	async createBaseNeurons(framePoints) {
		if (framePoints.length === 0) return [];

		// deduplicate points by coordinates (same coordinates = same neuron)
		const seen = new Map();
		const uniquePoints = [];
		for (const point of framePoints) {
			const coordStr = JSON.stringify(point.coordinates);
			if (!seen.has(coordStr)) {
				seen.set(coordStr, true);
				uniquePoints.push(point);
			}
		}

		// bulk insert neurons with type and channel, get their IDs
		const neurons = uniquePoints.map(p => [0, p.type, p.channel_id]);
		const neuronIds = await this.createNeurons(neurons);
		const created = uniquePoints.map((point, idx) => ({ coordinates: point.coordinates, neuron_id: neuronIds[idx] }));

		// insert coordinates for the created neurons
		await this.setNeuronCoordinates(created);

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
	 * Determine consensus from votes - aggregate by neuron, then pick winner per dimension.
	 * @param {Array} votes - Array of {from_neuron_id, neuron_id, source_type, source_id, strength, reward, level}
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async determineConsensus(votes) {
		if (votes.length === 0) return [];

		// Separate base level (0) from pattern levels (1+)
		// Pattern neurons don't have coordinates and don't compete - they all "win"
		const baseVotes = votes.filter(v => v.level === 0);
		const patternVotes = votes.filter(v => v.level > 0);

		// Process base level with per-dimension conflict resolution
		const baseInferences = await this.determineBaseConsensus(baseVotes);

		// Process pattern levels - all pattern inferences "win" (no conflict resolution)
		const patternInferences = this.aggregatePatternInferences(patternVotes);

		return [...baseInferences, ...patternInferences];
	}

	/**
	 * Determine consensus for base level (level 0) neurons.
	 * Uses per-dimension conflict resolution.
	 * @param {Array} votes - Votes for base level neurons only
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async determineBaseConsensus(votes) {
		if (votes.length === 0) return [];

		// Aggregate votes: first by source-target pair, then by target neuron
		const bySourceTarget = this.aggregateVotesBySourceTarget(votes);
		const aggregated = this.aggregateVotesByTarget(bySourceTarget);

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
	 * Aggregate pattern level (level > 0) inferences.
	 * Pattern neurons don't compete with each other - they all "win".
	 * Their predictions compete at the base level through pattern_future.
	 * @param {Array} votes - Votes for pattern neurons only
	 * @returns {Array} Array of inference objects with isWinner=true
	 */
	aggregatePatternInferences(votes) {
		if (votes.length === 0) return [];

		// Aggregate by neuron_id (pattern neurons don't have coordinates)
		const aggregated = new Map();
		for (const vote of votes) {
			if (!aggregated.has(vote.neuron_id))
				aggregated.set(vote.neuron_id, { neuron_id: vote.neuron_id, level: vote.level, strength: 0, rewardSum: 0, rewardCount: 0, sources: [], sourceNeurons: new Set() });

			const agg = aggregated.get(vote.neuron_id);
			agg.sources.push({ source_type: vote.source_type, source_id: vote.source_id, strength: vote.strength, reward: vote.reward, level: vote.level });
			agg.sourceNeurons.add(vote.from_neuron_id);
			agg.strength += vote.strength;
			agg.rewardSum += vote.reward;
			agg.rewardCount++;
		}

		// Calculate average reward and mark all as winners
		const inferences = [];
		for (const [neuronId, agg] of aggregated) {
			agg.reward = agg.rewardCount > 0 ? agg.rewardSum / agg.rewardCount : 0;
			inferences.push({ ...agg, isWinner: true }); // All pattern inferences "win"
		}

		return inferences;
	}

	/**
	 * Aggregate votes by (from_neuron_id, to_neuron_id) so each source neuron votes once per target.
	 * This prevents a single neuron with multiple distance connections from voting multiple times.
	 * @param {Array} votes - Raw votes from collectVotes
	 * @returns {Map} Map of "from:to" -> aggregated vote
	 */
	aggregateVotesBySourceTarget(votes) {
		const bySourceTarget = new Map();

		for (const vote of votes) {
			const key = `${vote.from_neuron_id}:${vote.neuron_id}`;
			if (!bySourceTarget.has(key))
				bySourceTarget.set(key, { from_neuron_id: vote.from_neuron_id, neuron_id: vote.neuron_id, strength: 0, rewardSum: 0, rewardCount: 0, sources: [], maxLevel: 0, hasPatternVotes: false, from_type: vote.from_type });

			const agg = bySourceTarget.get(key);
			agg.sources.push({ source_type: vote.source_type, source_id: vote.source_id, strength: vote.strength, reward: vote.reward, level: vote.level, from_type: vote.from_type });

			// Sum strength across distances, average reward
			agg.strength += vote.strength;
			agg.rewardSum += vote.reward;
			agg.rewardCount++;

			// Track max level and pattern votes
			if (vote.level > agg.maxLevel) agg.maxLevel = vote.level;
			if (vote.source_type === 'pattern') agg.hasPatternVotes = true;
		}

		// Calculate average reward per source-target pair
		for (const [key, agg] of bySourceTarget)
			agg.reward = agg.rewardCount > 0 ? agg.rewardSum / agg.rewardCount : 0;

		return bySourceTarget;
	}

	/**
	 * Aggregate source-target pairs by target neuron. Each source neuron counts once per target.
	 * If a neuron has pattern votes, connection votes are discarded (pattern inference overrides connection inference).
	 * @param {Map} bySourceTarget - Map from aggregateVotesBySourceTarget
	 * @returns {Map} Map of neuron_id -> aggregated vote with sourceNeurons set
	 */
	aggregateVotesByTarget(bySourceTarget) {
		const aggregated = new Map();

		for (const [key, srcAgg] of bySourceTarget) {
			if (!aggregated.has(srcAgg.neuron_id))
				aggregated.set(srcAgg.neuron_id, { neuron_id: srcAgg.neuron_id, strength: 0, rewardSum: 0, rewardCount: 0, reward: 0, sources: [], maxLevel: 0, hasPatternVotes: false, sourceNeurons: new Set() });

			const agg = aggregated.get(srcAgg.neuron_id);
			agg.sources.push(...srcAgg.sources);
			agg.sourceNeurons.add(srcAgg.from_neuron_id);

			// Sum strength, accumulate reward (averaging done later per dimension type)
			agg.strength += srcAgg.strength;
			agg.rewardSum += srcAgg.reward;
			agg.rewardCount++;

			// Track max level and pattern votes
			if (srcAgg.maxLevel > agg.maxLevel) agg.maxLevel = srcAgg.maxLevel;
			if (srcAgg.hasPatternVotes) agg.hasPatternVotes = true;
		}

		// Pattern inference overrides connection inference: if a neuron has pattern votes,
		// discard all connection votes. This ensures only pattern sources are saved to
		// inference_sources, preventing duplicate error pattern creation.
		for (const [neuronId, agg] of aggregated) {
			if (agg.hasPatternVotes) {
				// Filter to keep only pattern sources
				const patternSources = agg.sources.filter(s => s.source_type === 'pattern');
				// Recalculate strength and reward from pattern sources only
				agg.sources = patternSources;
				agg.strength = patternSources.reduce((sum, s) => sum + s.strength, 0);
				agg.rewardSum = patternSources.reduce((sum, s) => sum + s.reward, 0);
				agg.rewardCount = patternSources.length;
			}
		}

		// Calculate simple average reward (will be recalculated for actions per dimension)
		for (const [neuronId, agg] of aggregated)
			agg.reward = agg.rewardCount > 0 ? agg.rewardSum / agg.rewardCount : 0;

		return aggregated;
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
			if (info) {
				agg.coordinates = {};
				for (const [dimName, value] of info.coordinates)
					agg.coordinates[dimName] = value;
				agg.type = info.type; // 'action' or 'event' (from neuron)
				agg.channel = info.channel;
				agg.channel_id = info.channel_id;
			} else
				console.warn(`Warning: Base neuron ${neuronId} not found in getNeuronCoordinates - skipping`);
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
	 * Select winner for a dimension using level/pattern priority and Boltzmann (actions) or strength (events).
	 * @param {string} dimName - Dimension name
	 * @param {Array} dimVotes - Array of aggregated votes for this dimension
	 * @returns {Object|null} Winning vote or null if no votes
	 */
	selectWinnerForDimension(dimName, dimVotes) {
		if (dimVotes.length === 0) return null;

		// Filter to max level
		const maxLevel = Math.max(...dimVotes.map(v => v.maxLevel));
		let filtered = dimVotes.filter(v => v.maxLevel === maxLevel);

		// At max level, prefer pattern votes over connection votes
		const hasPatterns = filtered.some(v => v.hasPatternVotes);
		if (hasPatterns) filtered = filtered.filter(v => v.hasPatternVotes);

		// Select winner based on neuron type (all votes for same dimension have same type)
		const isAction = filtered[0]?.type === 'action';

		if (isAction) {
			// For actions: all voters count in denominator (missing votes = 0 reward)
			const allVoters = new Set();
			for (const v of filtered)
				for (const srcNeuron of v.sourceNeurons)
					allVoters.add(srcNeuron);
			const totalVoters = allVoters.size;

			// Recalculate reward with total voters as denominator
			for (const v of filtered)
				v.reward = totalVoters > 0 ? v.rewardSum / totalVoters : 0;

			const winner = this.boltzmannSelect(filtered);
			if (this.debug) console.log(`Voting: ${dimName} (action) Boltzmann = ${winner.coordinates[dimName]} (n${winner.neuron_id}, rwd=${winner.reward.toFixed(2)}, str=${winner.strength.toFixed(2)}, lvl=${winner.maxLevel}, pat=${winner.hasPatternVotes}, ${filtered.length} cand, ${totalVoters} voters)`);
			return winner;
		}

		// Events: deterministic selection by highest strength
		filtered.sort((a, b) => b.strength - a.strength);
		const winner = filtered[0];
		if (this.debug) console.log(`Voting: ${dimName} (event) winner = ${winner.coordinates[dimName]} (n${winner.neuron_id}, str=${winner.strength.toFixed(2)}, lvl=${winner.maxLevel}, pat=${winner.hasPatternVotes}, ${filtered.length} cand)`);
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
	 * runs the forget cycle
	 */
	async runForgetCycle() {
		throw new Error('runForgetCycle() must be implemented by subclass');
	}

	/**
	 * Apply global reward to active connections
	 */
	async applyRewards(channelRewards) {
		throw new Error('applyRewards(channelRewards) must be implemented by subclass');
	}

	/**
	 * Learn from base level: compare L0 inferences (age=1) to L0 observations (age=0).
	 * - Validation: track prediction accuracy for events
	 * - Events: negative reinforcement for wrong predictions (updates strength)
	 * - Actions: apply rewards based on channel outcomes (updates reward)
	 */
	async learnFromBaseLevel() {

		// Validate event predictions and track accuracy stats
		await this.validatePredictions();

		// Negative reinforcement for failed event predictions (strength decrease)
		await this.negativeReinforceConnections();

		// Apply rewards to action predictions based on channel outcomes
		const channelRewards = await this.getChannelRewards();
		await this.applyRewards(channelRewards);
	}

	/**
	 * Learn from inference level: pattern-level learning.
	 * Called after learnFromBaseLevel to update pattern predictions.
	 */
	async learnFromInferenceLevel() {

		// Pattern inference: merge pattern_future with observations
		await this.mergePatternFuture();

		// Create error patterns from failed predictions (all types)
		await this.createErrorPatterns();
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Processes all levels in bulk, like inferNeurons.
	 */
	async createErrorPatterns() {

		// Find connections that should be in pattern_future of new patterns
		// (prediction errors and action regret, unified in one method)
		const newPatternFutureCount = await this.populateNewPatternFuture();
		if (this.debug) console.log(`New pattern future count: ${newPatternFutureCount}`);
		if (newPatternFutureCount === 0) return;
		// if (newPatternFutureCount > 0) this.waitForUserInput = true;

		// Populate new_patterns table with peaks from new pattern future connections
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons(patternCount);

		// Create new patterns in pattern_peaks, pattern_past, pattern_future
		await this.createNewPatterns();

		if (this.debug) console.log(`Created ${patternCount} error patterns`);
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
	 * Collect votes from ALL levels in bulk for a specific dimension type (implementation-specific)
	 * @returns {Promise<Array>} Array of {neuron_id, source_type, source_id, strength, reward, level}
	 */
	async collectVotes() {
		throw new Error('collectVotes() must be implemented by subclass');
	}

	/**
	 * Save all inferences for learning (implementation-specific)
	 * @param {Array} inferences - inferences from determineConsensus
	 */
	async saveInferences(inferences) {
		throw new Error('saveInferences() must be implemented by subclass');
	}

	/**
	 * Get frame outputs for all channels from inferred_neurons table (implementation-specific)
	 * Reads output neurons (age=0, level=0) grouped by channel
	 * @returns {Promise<Map>} - Map of channel names to array of output coordinates
	 */
	async getFrameOutputs() {
		throw new Error('getFrameOutputs() must be implemented by subclass');
	}

	/**
	 * Populate inference_sources for executed actions (implementation-specific)
	 * Called after recognizeNeurons when connections have been created.
	 */
	async populateInferenceSources() {
		throw new Error('populateInferenceSources() must be implemented by subclass');
	}

	/**
	 * Get coordinates for a list of neuron IDs with dimension info
	 * @param {Array<number>} neuronIds - Array of neuron IDs
	 * @returns {Promise<Map>} Map of neuron_id → Map of dimension_name → {type, value, channel}
	 */
	async getNeuronCoordinates(neuronIds) {
		throw new Error('getNeuronCoordinates(neuronIds) must be implemented by subclass');
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
	 * @param {Array} neurons - Array of [level, type, channel_id] tuples
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async createNeurons(neurons) {
		throw new Error('createNeurons(neurons) must be implemented by subclass');
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
	 * Validate event predictions by comparing inferred_neurons (age=1) to active_neurons (age=0).
	 * Populates accuracyStats Map with {correct, total} counts per level.
	 * Only validates event predictions (not actions, which are validated via rewards).
	 */
	async validatePredictions() {
		throw new Error('validatePredictions() must be implemented by subclass');
	}

	/**
	 * Populate new_pattern_future with connections that should be in pattern_future of new patterns.
	 * Unified method handling both prediction errors and action regret.
	 * Returns the number of new pattern future connections found.
	 */
	async populateNewPatternFuture() {
		throw new Error('populateNewPatternFuture() must be implemented by subclass');
	}

	/**
	 * Populate new_patterns table from new_pattern_future (implementation-specific)
	 * Finds peak neurons (from_neuron_id of connections in new_pattern_future) and creates one pattern per peak.
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		throw new Error('populateNewPatterns() must be implemented by subclass');
	}

	/**
	 * Create pattern neurons and map them to new_patterns (implementation-specific)
	 * @param {number} patternCount - Number of patterns to create
	 */
	async createPatternNeurons(patternCount) {
		throw new Error('createPatternNeurons(patternCount) must be implemented by subclass');
	}

	/**
	 * Create new patterns in pattern_peaks, pattern_past, pattern_future (implementation-specific)
	 */
	async createNewPatterns() {
		throw new Error('createNewPatterns() must be implemented by subclass');
	}
}

