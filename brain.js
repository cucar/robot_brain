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
		this.maxConnectionReward = 10; // maximum reward factor for connections and patterns (wider range to overcome strength bias)
		this.minConnectionReward = 1 / this.maxConnectionReward; // minimum reward factor for connections and patterns (clamped to prevent extreme values)

		// connection learning parameters
		this.connectionNegativeReinforcement = 0.1; // how much to weaken connections when predictions fail

		// pattern learning parameters
		this.minErrorPatternThreshold = 10.0; // minimum prediction strength to create error-driven pattern
		this.mergePatternThreshold = 0.66; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.maxLevels = 10; // just to prevent against infinite recursion

		// inference parameters
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.95; // reward temporal decay = POW(rewardTimeDecayFactor, age)

		// voting parameters - all levels vote for actions, higher levels have more temporal context
		this.levelVoteMultiplier = 1.2; // vote weight = strength * POW(levelVoteMultiplier, level)

		// reward parameters
		this.maxRewardsAge = 1; // how far back in time to apply rewards (1 = only most recent outputs)

		// exploration parameters - probability inversely proportional to inference strength
		this.minExploration = 0.03; // minimum - never stop exploring
		this.maxExploration = 1.0; // 100% when totalStrength = 0
		this.explorationScale = 1000; // controls decay rate of exploration probability
		this.minExploredVotes = 2; // minimum votes to consider an action "explored" (excluding actions)

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle (reduced to preserve learned connections)
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle
		this.rewardForgetRate = 0.05; // how much reward factors decay toward 1.0 per forget cycle (0.05 = 5% decay toward neutral)

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
		if (this.debug2) console.log('Initializing dimensions for registered channels...');
		for (const [channelName, channel] of this.channels) {
			await this.insertChannelDimensions(channel.getEventDimensions(), channelName, 'event');
			await this.insertChannelDimensions(channel.getOutputDimensions(), channelName, 'action');
		}
	}

	/**
	 * inserts channel dimensions
	 */
	async insertChannelDimensions(dimensions, channelName, type){
		if (this.debug2) console.log(`Creating ${type} dimensions for ${channelName}:`, dimensions);
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
		if (this.debug2) console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * returns the current frame combined from all registered channels
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

			// Get the frame event inputs from the channel (current state before any outputs are executed)
			const channelEvents = await channel.getFrameEvents();

			// Get last inferred actions to be executed in this frame (from brain's inference tables)
			const channelActions = frameOutputs.get(channelName) || [];

			// Execute actions - this updates channel state
			await channel.executeOutputs(channelActions);

			// Add frame points to the frame
			frame.push(...[ ...channelEvents, ...channelActions ]);
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
			const rewardFactor = await channel.getRewards();
			if (rewardFactor !== 1.0) { // Only process non-neutral feedback
				if (this.debug2) console.log(`${channelName}: reward factor ${rewardFactor.toFixed(3)}`);
				channelRewards.set(channelName, rewardFactor);
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
		// Only votes from events (base level) or interneurons (level > 0, from_dim_type=null) count
		const exploredActions = votedActions.filter(a => {
			if (!a.sources) return false;
			const relevantVotes = a.sources.filter(s => s.from_dim_type !== 'action').length;
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

		// Find or create neuron for this action
		const [actionNeuronId] = await this.getFrameNeurons([actionCoordinates]);

		// Return exploration action - sources populated after execution in populateInferenceSources
		const exploration = {
			neuron_id: actionNeuronId,
			strength: 100000, // High strength for exploration
			reward: 1.0, // Neutral reward
			coordinates: actionCoordinates,
			dimType: 'action',
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

		// Group action inferences by channel
		const byChannel = new Map();
		for (const inf of inferences) {
			if (!inf.coordinates || inf.dimType !== 'action') continue;
			for (const dimName of Object.keys(inf.coordinates)) {
				const dimInfo = this.dimensions.get(dimName);
				if (dimInfo && dimInfo.type === 'action') {
					if (!byChannel.has(dimInfo.channel)) byChannel.set(dimInfo.channel, []);
					byChannel.get(dimInfo.channel).push(inf);
				}
			}
		}

		// Process each channel
		for (const [channelName] of this.channels) {
			const channelInferences = byChannel.get(channelName) || [];
			const channelWinners = channelInferences.filter(inf => inf.isWinner);

			// Check if we should explore this channel
			// if (!this.shouldExploreChannel(channelName, channelWinners)) continue;

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
	 * returns base neuron ids for given set of points coming from the frame
	 */
	async getFrameNeurons(frame) {

		// try to get all the neurons that have coordinates in close ranges for each point - return format: [{ point_str, neuron_id }]
		const matches = await this.matchFrameNeurons(frame);
		if (this.debug2) console.log('pointNeuronMatches', matches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const neuronIds = matches.filter(p => p.neuron_id).map(p => p.neuron_id);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => !p.neuron_id);
		if (pointsNeedingNeurons.length > 0) {
			if (this.debug2) console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with dedupe.`);
			const createdNeuronIds = await this.createBaseNeurons(pointsNeedingNeurons.map(p => p.point_str));
			neuronIds.push(...createdNeuronIds);
		}

		// return matching neuron ids to given points
		if (neuronIds.length === 0) throw new Error(`Failed to create neuron for exploration action: ${JSON.stringify(frame)}`);
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
	}

	/**
	 * Determine consensus from votes using voting architecture.
	 * All levels vote, winner is determined per dimension.
	 * Unified logic: pick the highest effectiveStrength (sum of strength * reward per neuron).
	 * @param {Array} votes - Array of {neuron_id, source_type, source_id, strength, reward, level}
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async determineConsensus(votes) {
		if (votes.length === 0) return [];

		// Aggregate votes by neuron_id
		// effectiveStrength = sum(strength * reward) for voting
		// strength = sum(strength) for database
		// reward = weighted average of rewards for database
		const aggregated = new Map();
		for (const vote of votes) {
			if (!aggregated.has(vote.neuron_id))
				aggregated.set(vote.neuron_id, { neuron_id: vote.neuron_id, effectiveStrength: 0, strength: 0, totalWeightedReward: 0, sources: [] });

			const agg = aggregated.get(vote.neuron_id);
			agg.sources.push({ source_type: vote.source_type, source_id: vote.source_id, strength: vote.strength, reward: vote.reward, level: vote.level, from_dim_type: vote.from_dim_type });

			// For voting: sum of (strength * reward)
			agg.effectiveStrength += vote.strength * vote.reward;

			// For database: accumulate weighted rewards
			agg.totalWeightedReward += vote.reward * vote.strength;
			agg.strength += vote.strength;
		}

		// Calculate final weighted average reward for each neuron
		for (const [neuronId, agg] of aggregated)
			agg.reward = agg.strength > 0 ? agg.totalWeightedReward / agg.strength : 1.0;

		// Get coordinates for all voted neurons to group by dimension
		const neuronIds = [...aggregated.keys()];
		const coordinates = await this.getNeuronCoordinates(neuronIds);

		// Add coordinates and determine dimension type for each neuron
		for (const [neuronId, agg] of aggregated) {
			const coords = coordinates.get(neuronId);
			if (coords) {
				agg.coordinates = {};
				for (const [dimName, dimInfo] of coords) {
					agg.coordinates[dimName] = dimInfo.value;
					agg.dimType = dimInfo.type; // 'action' or 'event'
				}
			}
		}

		// Group by dimension
		const byDimension = new Map();
		for (const [neuronId, agg] of aggregated) {
			if (!agg.coordinates) continue;
			for (const dimName of Object.keys(agg.coordinates)) {
				if (!byDimension.has(dimName)) byDimension.set(dimName, []);
				byDimension.get(dimName).push(agg);
			}
		}

		// Find winner per dimension: highest effectiveStrength (strength * reward)
		const inferences = [];
		const winners = new Set();

		for (const [dimName, dimVotes] of byDimension) {
			if (dimVotes.length === 0) continue;

			// Sort by effectiveStrength DESC
			dimVotes.sort((a, b) => b.effectiveStrength - a.effectiveStrength);
			const winner = dimVotes[0];
			winners.add(winner.neuron_id);

			if (this.debug) console.log(`Voting: ${dimName} winner = ${winner.coordinates[dimName]} (neuron ${winner.neuron_id}, effectiveStrength: ${winner.effectiveStrength.toFixed(2)}, ${dimVotes.length} candidates)`);
		}

		// Build inferences array with isWinner flag
		for (const [neuronId, agg] of aggregated)
			inferences.push({ ...agg, isWinner: winners.has(neuronId) });

		return inferences;
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
	 * - Events: negative reinforcement for wrong predictions (updates strength)
	 * - Actions: apply rewards based on channel outcomes (updates reward)
	 */
	async learnFromBaseLevel() {

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

		// Check if we have bad inferences across all levels
		const badCount = await this.countBadInferences();
		if (this.debug) console.log(`Bad inferences count: ${badCount}`);
		if (badCount === 0) return;

		// Find what we should have predicted instead (across all levels)
		const newPatternConnectionCount = await this.populateNewPatternConnections();
		if (this.debug) console.log(`New pattern connections count: ${newPatternConnectionCount}`);
		if (newPatternConnectionCount === 0) return;

		// Populate new_patterns table with peaks from new pattern connections
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons(patternCount);

		// Merge new patterns into pattern_peaks, pattern_past, pattern_future
		await this.mergeNewPatterns();

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
	 * @param {number} count - Number of neurons to create
	 * @param {number} level - Level of the neurons (0 for base, 1+ for patterns)
	 */
	async createNeurons(count, level) {
		throw new Error('createNeurons(count, level) must be implemented by subclass');
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
	 * Count bad inferences: prediction errors OR negative reward actions (implementation-specific)
	 * Returns the count of bad inferences.
	 */
	async countBadInferences() {
		throw new Error('countBadInferences() must be implemented by subclass');
	}

	/**
	 * Populate new_pattern_connections with connections that should be predicted by new patterns.
	 * Includes: (1) active connections not predicted (prediction errors), (2) best loser connections (action regret)
	 * Returns the number of new pattern connections found.
	 */
	async populateNewPatternConnections() {
		throw new Error('populateNewPatternConnections() must be implemented by subclass');
	}

	/**
	 * Populate new_patterns table from new pattern connections (implementation-specific)
	 * Finds peak neurons (from_neurons of new_pattern_connections) and creates one pattern per peak.
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
	 * Merge new patterns into pattern_peaks, pattern_past, pattern_future (implementation-specific)
	 */
	async mergeNewPatterns() {
		throw new Error('mergeNewPatterns() must be implemented by subclass');
	}
}

