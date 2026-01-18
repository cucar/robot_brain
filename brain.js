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
		this.levelVoteMultiplier = 2; // vote weight = strength * POW(levelVoteMultiplier, level)
		this.boltzmannTemperature = 0.1; // temperature for Boltzmann selection (lower = more aggressive, 1.0 = standard)

		// reward parameters
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

		// Cache for action neuron IDs (for debug output)
		this.actionNeuronCache = null;
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

		// get channel rewards
		const channelRewards = await this.getChannelRewards();

		// age the active neurons in memory context - sliding the temporal window
		await this.ageNeurons();

		// activate base neurons from the frame along with higher level patterns from them - what's happening right now?
		await this.recognizeNeurons(frame);

		// populate inference sources for executed actions (age=1) using just-created connections
		await this.populateInferenceSources();

		// learn from connection event and action inferences
		await this.refineConnections(channelRewards);

		// refine the learned pattern definitions from prediction errors and action regret
		await this.refinePatterns(channelRewards);

		// learn new patterns from failed predictions and action regret
		await this.learnNewPatterns(channelRewards);

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

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (memory tables)...');
		await this.truncateTables([
			'active_neurons',
			'inferred_neurons',
			'matched_patterns',
			'matched_pattern_connections',
			'active_connections',
			'inference_sources',
			'new_pattern_future',
			'new_patterns'
		]);

		// Clear action neuron cache since neurons may have been recreated
		this.actionNeuronCache = null;
	}

	/**
	 * Reset accuracy stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats.clear();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all learned data)...');
		await this.truncateTables([
			'active_neurons',
			'inferred_neurons',
			'active_connections',
			'matched_patterns',
			'matched_pattern_connections',
			'inference_sources',
			'new_pattern_future',
			'new_patterns',
			'pattern_peaks',
			'pattern_past',
			'pattern_future',
			'connections',
			'coordinates',
			'neurons',
			'dimensions'
		]);

		// Clear action neuron cache since all neurons were deleted
		this.actionNeuronCache = null;
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

		// Age inference tables for temporal credit assignment
		await this.conn.query('UPDATE inferred_neurons SET age = age + 1 ORDER BY age DESC');
		await this.conn.query('UPDATE inference_sources SET age = age + 1 ORDER BY age DESC');

		// Skip deletions until we have enough frames
		if (this.frameNumber < this.contextLength + 1) return;

		// Delete aged-out neurons from all levels at once
		const [neuronResult] = await this.conn.query('DELETE FROM active_neurons WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Deactivated ${neuronResult.affectedRows} aged-out neurons across all levels (age >= ${this.contextLength})`);

		// Delete aged-out connections from all levels at once
		const [connectionResult] = await this.conn.query('DELETE FROM active_connections WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Deactivated ${connectionResult.affectedRows} aged-out connections across all levels (age >= ${this.contextLength})`);

		// Clean up inferred neurons after execution
		const [inferredResult] = await this.conn.query('DELETE FROM inferred_neurons WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Cleaned up ${inferredResult.affectedRows} executed inferred neurons (age >= ${this.contextLength})`);

		// Delete aged-out inference sources (same lifecycle as neurons)
		const [infSourcesResult] = await this.conn.query('DELETE FROM inference_sources WHERE age >= ?', [this.contextLength]);
		if (this.debug) console.log(`Cleaned up ${infSourcesResult.affectedRows} aged-out inference sources (age >= ${this.contextLength})`);
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

		// Validate event predictions and track accuracy stats
		await this.validatePredictions();
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
	 * All votes are for base level neurons (level 0) - pattern neurons are activated via recognizeNeurons.
	 * Uses per-dimension conflict resolution.
	 * @param {Array} votes - Array of {from_neuron_id, neuron_id, source_type, source_id, strength, reward, distance, source_level, target_level}
	 * @returns {Promise<Array>} Array of inference objects with isWinner flag
	 */
	async determineConsensus(votes) {
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
	 * Aggregate votes by (from_neuron_id, to_neuron_id) so each source neuron votes once per target.
	 * This prevents a single neuron with multiple distance connections from voting multiple times.
	 * Uses weighted averaging with distance and level weights.
	 * @param {Array} votes - Raw votes from collectVotes with distance and source_level
	 * @returns {Map} Map of "from:to" -> aggregated vote with weighted strength/reward
	 */
	aggregateVotesBySourceTarget(votes) {
		const bySourceTarget = new Map();

		for (const vote of votes) {
			const key = `${vote.from_neuron_id}:${vote.neuron_id}`;
			if (!bySourceTarget.has(key))
				bySourceTarget.set(key, { from_neuron_id: vote.from_neuron_id, neuron_id: vote.neuron_id, strengthSum: 0, rewardSum: 0, weightSum: 0, sources: [], from_type: vote.from_type });

			const agg = bySourceTarget.get(key);

			// Calculate weights: distance_weight * level_weight
			const distanceWeight = Math.pow(this.peakTimeDecayFactor, vote.distance - 1);
			const levelWeight = Math.pow(this.levelVoteMultiplier, vote.source_level);
			const weight = distanceWeight * levelWeight;

			agg.sources.push({ source_type: vote.source_type, source_id: vote.source_id, strength: vote.strength, reward: vote.reward, distance: vote.distance, source_level: vote.source_level, from_type: vote.from_type });

			// Weighted sum for strength and reward
			agg.strengthSum += vote.strength * weight;
			agg.rewardSum += vote.reward * weight;
			agg.weightSum += weight;
		}

		// Calculate weighted average strength and reward per source-target pair
		for (const [key, agg] of bySourceTarget) {
			agg.strength = agg.weightSum > 0 ? agg.strengthSum / agg.weightSum : 0;
			agg.reward = agg.weightSum > 0 ? agg.rewardSum / agg.weightSum : 0;
		}

		return bySourceTarget;
	}

	/**
	 * Aggregate source-target pairs by target neuron. Each source neuron counts once per target.
	 * Both pattern and connection votes contribute with weighted averaging.
	 * @param {Map} bySourceTarget - Map from aggregateVotesBySourceTarget (already weighted)
	 * @returns {Map} Map of neuron_id -> aggregated vote with sourceNeurons set
	 */
	aggregateVotesByTarget(bySourceTarget) {
		const aggregated = new Map();

		for (const [key, srcAgg] of bySourceTarget) {
			if (!aggregated.has(srcAgg.neuron_id))
				aggregated.set(srcAgg.neuron_id, { neuron_id: srcAgg.neuron_id, strengthSum: 0, rewardSum: 0, count: 0, sources: [], sourceNeurons: new Set() });

			const agg = aggregated.get(srcAgg.neuron_id);
			agg.sources.push(...srcAgg.sources);
			agg.sourceNeurons.add(srcAgg.from_neuron_id);

			// Sum weighted strength and reward from each source
			agg.strengthSum += srcAgg.strength;
			agg.rewardSum += srcAgg.reward;
			agg.count++;
		}

		// Calculate average strength and reward across all sources
		for (const [neuronId, agg] of aggregated) {
			agg.strength = agg.count > 0 ? agg.strengthSum / agg.count : 0;
			agg.reward = agg.count > 0 ? agg.rewardSum / agg.count : 0;
		}

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
	 * Strengthen and reward executed action pattern_future predictions.
	 *
	 * Channel-Specific Credit Assignment:
	 * 1. Identify which channel each base-level output belongs to (via output dimensions)
	 * 2. Use active_connections (just executed) to find which patterns led to each action
	 * 3. Apply channel-specific reward via CASE-WHEN on channel_id
	 *
	 * Rewards are applied via exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
	 * This converges to the expected reward for each connection.
	 * Neutral reward is 0, positive is good, negative is bad
	 *
	 * Also strengthens executed action patterns so they won't be forgotten.
	 */
	async rewardExecutedActionPatterns(channelRewards) {

		// Build CASE-WHEN clause for channel rewards (empty string if no rewards)
		let caseWhen = '';
		for (const [channelName, reward] of channelRewards) caseWhen += `
			WHEN n_target.channel_id = ${this.channelNameToId[channelName]}
			THEN ${this.rewardExpSmooth} * ${reward} + (1 - ${this.rewardExpSmooth}) * pf.reward `;

		// Strengthen and reward action pattern_future in one query
		// Join chain: pattern_future → active_connection (just executed) → active pattern
		// Strength is always incremented; reward is updated only if channel has a reward
		const [result] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN neurons pn ON pn.id = pf.pattern_neuron_id
			JOIN active_connections ac ON ac.connection_id = pf.connection_id AND ac.age = 0
			JOIN active_neurons an ON an.neuron_id = pf.pattern_neuron_id
			JOIN neurons n_target ON n_target.id = ac.to_neuron_id
			SET pf.strength = LEAST(?, pf.strength + 1),
			    pf.reward = CASE ${caseWhen} ELSE pf.reward END
			WHERE pn.type = 'action'
		`, [this.maxConnectionStrength]);

		if (this.debug) console.log(`Strengthened/rewarded ${result.affectedRows} action pattern_future predictions`);
	}

	/**
	 * Add alternative actions to action patterns in painful channels.
	 * When an action pattern prediction fails (painful reward), find the best untried alternative.
	 * Creates connections if they don't exist yet.
	 */
	async addAlternativeActionsToPatterns(channelRewards) {

		// Find channels with negative rewards
		const painfulChannelIds = [];
		for (const [channelName, reward] of channelRewards)
			if (reward < this.actionRegretMinPain) painfulChannelIds.push(this.channelNameToId[channelName]);
		if (painfulChannelIds.length === 0) return;

		// Query 1: Get failed patterns with their failed action id
		// Each (pattern, distance) has exactly one failed action since channels execute one action per frame
		// Same pattern can be active at multiple ages, predicting at different distances
		// Use EXISTS for active_neurons to avoid row multiplication from multiple (level, age) combinations
		const [failedPatterns] = await this.conn.query(`
			SELECT pf.pattern_neuron_id, pp.peak_neuron_id, pn.channel_id, c.distance, ac.to_neuron_id as failed_action_id
			FROM pattern_future pf
			JOIN neurons pn ON pn.id = pf.pattern_neuron_id
			JOIN active_connections ac ON ac.connection_id = pf.connection_id AND ac.age = 0
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf.pattern_neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE pn.type = 'action'
			AND pn.channel_id IN (${painfulChannelIds.join(',')})
			AND EXISTS (SELECT 1 FROM active_neurons an WHERE an.neuron_id = pf.pattern_neuron_id)
		`);
		if (failedPatterns.length === 0) return;

		// Query 2: Get actions already in pattern_future for these patterns
		const patternIds = failedPatterns.map(fp => fp.pattern_neuron_id);
		const [existingFuture] = await this.conn.query(`
			SELECT pf.pattern_neuron_id, c.to_neuron_id as action_id
			FROM pattern_future pf
			JOIN connections c ON c.id = pf.connection_id
			WHERE pf.pattern_neuron_id IN (?)
		`, [patternIds]);

		// Build set of existing actions per pattern
		const existingByPattern = new Map();
		for (const ef of existingFuture) {
			if (!existingByPattern.has(ef.pattern_neuron_id)) existingByPattern.set(ef.pattern_neuron_id, new Set());
			existingByPattern.get(ef.pattern_neuron_id).add(ef.action_id);
		}

		// Query 3: Get all action neurons per channel with their connection rewards from peaks
		const channelIds = [...new Set(failedPatterns.map(fp => fp.channel_id))];
		const peakIds = [...new Set(failedPatterns.map(fp => fp.peak_neuron_id))];
		const [candidates] = await this.conn.query(`
			SELECT n.id as action_id, n.channel_id, c.from_neuron_id as peak_neuron_id, c.distance, c.reward
			FROM neurons n
			LEFT JOIN connections c ON c.to_neuron_id = n.id AND c.from_neuron_id IN (?)
			WHERE n.type = 'action' AND n.channel_id IN (?)
		`, [peakIds, channelIds]);

		// Build map: (peak, channel, distance, action) -> reward
		const rewardMap = new Map();
		const actionsByChannel = new Map();
		for (const c of candidates) {
			if (!actionsByChannel.has(c.channel_id))
				actionsByChannel.set(c.channel_id, new Set());
			actionsByChannel.get(c.channel_id).add(c.action_id);

			if (c.peak_neuron_id !== null)
				rewardMap.set(`${c.peak_neuron_id}:${c.distance}:${c.action_id}`, c.reward);
		}

		// In-memory: pick the best alternative per (pattern, distance)
		const best = [];
		for (const fp of failedPatterns) {
			const existingIds = existingByPattern.get(fp.pattern_neuron_id) || new Set();
			const channelActions = actionsByChannel.get(fp.channel_id) || new Set();

			let bestAlt = null;
			let bestReward = -Infinity;
			for (const actionId of channelActions) {
				if (actionId === fp.failed_action_id || existingIds.has(actionId)) continue;
				const reward = rewardMap.get(`${fp.peak_neuron_id}:${fp.distance}:${actionId}`) ?? -999;
				if (reward > bestReward || (reward === bestReward && (!bestAlt || actionId < bestAlt.alt_action_id))) {
					bestAlt = { ...fp, alt_action_id: actionId };
					bestReward = reward;
				}
			}
			if (bestAlt) best.push(bestAlt);
		}

		if (best.length === 0) return;

		// Bulk insert connections (ON DUPLICATE KEY for existing ones)
		const connValues = best.map(b => [b.peak_neuron_id, b.alt_action_id, b.distance, 1.0, 0.0]);
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength, reward)
			VALUES ? ON DUPLICATE KEY UPDATE strength = strength
		`, [connValues]);

		// Query to get connection IDs for the just-created/existing connections
		const connKeys = best.map(b => [b.peak_neuron_id, b.alt_action_id, b.distance]);
		const [connRows] = await this.conn.query(`
			SELECT id, from_neuron_id, to_neuron_id, distance FROM connections
			WHERE (from_neuron_id, to_neuron_id, distance) IN (?)
		`, [connKeys]);

		// Map connection keys to IDs
		const connIdMap = new Map();
		for (const c of connRows)
			connIdMap.set(`${c.from_neuron_id}:${c.to_neuron_id}:${c.distance}`, c.id);

		// Build pattern_future inserts
		const futureValues = [];
		for (const b of best) {
			const connId = connIdMap.get(`${b.peak_neuron_id}:${b.alt_action_id}:${b.distance}`);
			if (connId) futureValues.push([b.pattern_neuron_id, connId, 1.0, 0.0]);
		}

		if (futureValues.length === 0) return;

		const [result] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength, reward)
			VALUES ?
		`, [futureValues]);

		if (this.debug && result.affectedRows > 0)
			console.log(`Action pattern refinement: ${result.affectedRows} alternatives added`);
	}

	/**
	 * Learns patterns from prediction errors and action regret and continues to refine them as they are observed
	 */
	async refinePatterns(channelRewards) {

		// update pattern_past for action and event patterns both
		// add/strengthen observed connections, weaken unobserved connections
		await this.refinePatternPast();

		// update pattern_future for event patterns based on observations
		await this.refineEventPatternsFuture();

		// apply rewards to action predictions based on channel outcomes
		await this.rewardExecutedActionPatterns(channelRewards);
		await this.addAlternativeActionsToPatterns(channelRewards);
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Processes all levels in bulk, like inferNeurons.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 */
	async learnNewPatterns(channelRewards) {

		// do not start learning new patterns until we had enough frames in the context
		if (this.frameNumber < this.contextLength + 1) return;

		// Find connections that should be in pattern_future of new patterns
		// (prediction errors and action regret, unified in one method)
		const newPatternFutureCount = await this.populateNewPatternFuture(channelRewards);
		if (this.debug) console.log(`New pattern future count: ${newPatternFutureCount}`);
		if (newPatternFutureCount === 0) return;
		// if (newPatternFutureCount > 0) this.waitForUserInput = true;

		// Populate new_patterns table with peaks from new pattern future connections
		const patternCount = await this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		await this.createPatternNeurons();

		// Create new patterns in pattern_peaks, pattern_past, pattern_future
		await this.createNewPatterns();

		if (this.debug) console.log(`Created ${patternCount} error patterns`);
	}

	/**
	 * Apply negative reinforcement to failed connection predictions.
	 * Weakens connections that made incorrect predictions when age = distance (outcome observed).
	 * Uses inference_sources to find which connections made predictions.
	 */
	async refineConnections(channelRewards) {

		// Apply negative reinforcement to failed event predictions - Failed = predicted but not observed
		// Only applies to events/patterns, NOT actions (actions are handled by rewards below)
		// Reinforcement applied when age = distance (prediction outcome just observed)
		const [result] = await this.conn.query(`
			UPDATE connections c
			JOIN neurons n ON n.id = c.to_neuron_id
			JOIN inference_sources isrc ON isrc.source_id = c.id AND isrc.source_type = 'connection'
			SET c.strength = GREATEST(0, c.strength - ?)
			WHERE c.strength > 0
			-- penalize when prediction outcome is observed (age = distance)
			AND isrc.age = isrc.distance
			-- penalize the connections that did not come true
			AND c.id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
			-- only penalize event predictions, not actions
			AND n.type = 'event'
		`, [this.connectionNegativeReinforcement]);
		if (this.debug && result.affectedRows > 0)
			console.log(`Weakened ${result.affectedRows} failed event predictions`);

		// Apply rewards reinforcement to executed actions via connection inference
		let totalWinnerConnections = 0, totalWinnerPatterns = 0;
		for (const [channelName, reward] of channelRewards) {

			if (this.debug) console.log(`Applying reward ${reward.toFixed(3)} for channel: ${channelName}`);

			// WINNERS: Apply actual reward to winning votes (is_winner=1)
			// Rewards applied when age = distance (prediction outcome just observed)
			// Exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
			// ONLY ACTIONS get rewards - events get strength-based reinforcement instead
			// Reward connection-based inferences
			const [winnerConnResult] = await this.conn.query(`
				UPDATE connections c
				JOIN inference_sources isrc ON c.id = isrc.source_id AND isrc.source_type = 'connection'
				JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age AND inf.level = 0
				JOIN neurons n ON n.id = inf.neuron_id
				SET c.reward = :smooth * :reward + (1 - :smooth) * c.reward
				WHERE isrc.age = isrc.distance
				AND inf.is_winner = 1
				AND n.type = 'action'
				AND n.channel_id = (SELECT id FROM channels WHERE name = :channelName)
			`, { smooth: this.rewardExpSmooth, reward, channelName });
			totalWinnerConnections += winnerConnResult.affectedRows;

			// LOSERS: Leave alone - we don't know what would have happened if they were executed
		}

		if (this.debug) console.log(`Total connection rewarded winners=${totalWinnerConnections}`);
	}

	/**
	 * Refine pattern_future for event patterns with observed connections.
	 * Called during learning phase after pattern inference.
	 * Uses inference_sources with source_type='pattern' to know which patterns made predictions.
	 * Refinement happens when age = distance (prediction outcome just observed).
	 * pattern_future stores connection_id (from peak neuron to target).
	 * This method only applies to EVENT patterns.
	 * Action pattern refinement happens in applyRewards (both reward-based and trial alternatives).
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future connections that were correctly predicted (target now active)
	 * 2. Negative: Weaken pattern_future connections that were incorrectly predicted (target NOT active)
	 * 3. Novel: Add new connections from peak to newly observed neurons (distance=1 only)
	 */
	async refineEventPatternsFuture() {

		// 1. POSITIVE REINFORCEMENT: Strengthen correctly predicted connections (event patterns only)
		// Connection is now active at age=0, refinement when age = distance
		const [strengthenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN neurons pn ON pn.id = pf.pattern_neuron_id
			JOIN active_connections ac ON ac.connection_id = pf.connection_id AND ac.age = 0
			JOIN inference_sources isrc ON isrc.source_id = pf.id AND isrc.source_type = 'pattern'
			SET pf.strength = LEAST(?, pf.strength + 1)
			WHERE isrc.age = isrc.distance
			AND pn.type = 'event'
		`, [this.maxConnectionStrength]);
		if (this.debug) console.log(`Strengthened ${strengthenResult.affectedRows} correct event pattern_future predictions`);

		// 2. NEGATIVE REINFORCEMENT: Weaken incorrectly predicted connections (event patterns only)
		// Connection is NOT active, refinement when age = distance
		const [weakenResult] = await this.conn.query(`
			UPDATE pattern_future pf
			JOIN neurons pn ON pn.id = pf.pattern_neuron_id
			JOIN inference_sources isrc ON isrc.source_id = pf.id AND isrc.source_type = 'pattern'
			SET pf.strength = GREATEST(?, pf.strength - ?)
			WHERE isrc.age = isrc.distance
			AND pn.type = 'event'
			AND pf.connection_id NOT IN (SELECT connection_id FROM active_connections WHERE age = 0)
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);
		if (this.debug) console.log(`Weakened ${weakenResult.affectedRows} failed event pattern_future predictions`);

		// 3. ADD NOVEL CONNECTIONS: Active connections from peak not yet in pattern_future (event patterns only)
		// Find event patterns that made predictions, get their peak neurons, find active connections from peak
		// Pattern future must only contain same-channel predictions
		// Novel connections are added at the same distance as the prediction that was made
		const [novelResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT DISTINCT pf_src.pattern_neuron_id, ac.connection_id, 1.0
			FROM inference_sources isrc
			JOIN pattern_future pf_src ON pf_src.id = isrc.source_id
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = pf_src.pattern_neuron_id
			JOIN neurons pattern_n ON pattern_n.id = pf_src.pattern_neuron_id
			JOIN active_connections ac ON ac.from_neuron_id = pp.peak_neuron_id AND ac.age = 0
			JOIN connections c ON c.id = ac.connection_id
			JOIN neurons target_n ON target_n.id = c.to_neuron_id
			WHERE isrc.age = isrc.distance
			AND isrc.source_type = 'pattern'
			AND pattern_n.type = 'event'
			AND c.distance = isrc.distance
			AND target_n.type = 'event'
			AND target_n.channel_id = pattern_n.channel_id
			AND NOT EXISTS (
				SELECT 1 FROM pattern_future pf
				WHERE pf.pattern_neuron_id = pf_src.pattern_neuron_id
				AND pf.connection_id = ac.connection_id
			)
		`);
		if (this.debug) console.log(`Added ${novelResult.affectedRows} novel connections to event pattern_future`);
	}

	/**
	 * Collect votes from ALL levels in bulk queries.
	 * Returns raw strength/reward with distance and source_level for weighted averaging.
	 * @returns {Promise<Array>} Array of {neuron_id, source_type, source_id, strength, reward, distance, source_level, target_level}
	 */
	async collectVotes() {

		// Collect connection votes from ALL levels in one query
		// Returns raw strength/reward with distance and source_level for weighted averaging
		// from_type: 'event'/'action' for base neurons, NULL for interneurons (level > 0)
		// Exclude neurons at max age - they'll be deactivated before we can populate inference_sources
		const [connectionVotes] = await this.conn.query(`
			SELECT c.from_neuron_id, c.to_neuron_id as neuron_id, 'connection' as source_type, c.id as source_id,
				c.strength, c.reward, c.distance, an.level as source_level, n.level as target_level, nf.type as from_type
			FROM active_neurons an
			JOIN connections c ON c.from_neuron_id = an.neuron_id
			JOIN neurons n ON n.id = c.to_neuron_id
			JOIN neurons nf ON nf.id = c.from_neuron_id
			WHERE c.distance = an.age + 1
            AND c.strength > 0
			AND an.age < :maxAge
		`, { maxAge: this.contextLength });

		// Collect pattern votes from ALL levels and ages
		// pattern_future stores connections with various distances, patterns predict when distance matches age+1
		// Like connections, patterns can predict at any distance (age=0 predicts distance=1, age=1 predicts distance=2, etc.)
		// from_type: check underlying connection's from_neuron_id type
		// Exclude neurons at max age - they'll be deactivated before we can populate inference_sources
		// source_id is pattern_future.id (not connection_id) so each pattern prediction is tracked separately
		const [patternVotes] = await this.conn.query(`
			SELECT c.from_neuron_id, c.to_neuron_id as neuron_id, 'pattern' as source_type, pf.id as source_id,
				pf.strength, pf.reward, c.distance, an.level as source_level, n.level as target_level, nf.type as from_type
			FROM active_neurons an
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = an.neuron_id
			JOIN pattern_future pf ON pf.pattern_neuron_id = pp.pattern_neuron_id
			JOIN connections c ON c.id = pf.connection_id
			JOIN neurons n ON n.id = c.to_neuron_id
			JOIN neurons nf ON nf.id = c.from_neuron_id
			WHERE c.distance = an.age + 1
            AND c.strength > 0
            AND pf.strength > 0
			AND an.age < :maxAge
		`, { maxAge: this.contextLength });

		// Combine all votes
		const allVotes = [...connectionVotes, ...patternVotes];
		if (this.debug) console.log(`Collected ${connectionVotes.length} connection + ${patternVotes.length} pattern = ${allVotes.length} total votes`);

		// DEBUG: Check for level 1+ connection inferences (patterns doing connection inference)
		// Check if from_neuron is a pattern (level > 0) by checking if from_type is NULL
		// const level1PlusConnectionVotes = connectionVotes.filter(v => v.from_type === null);
		// if (level1PlusConnectionVotes.length > 0) {
		// 	console.error('FOUND LEVEL 1+ CONNECTION INFERENCES:');
		// 	console.error(level1PlusConnectionVotes);
		// 	process.exit(1);
		// }

		// Debug: show votes for action and event neurons
		if (this.debug2) {
			await this.debugActionVotes(allVotes);
			await this.debugEventVotes(allVotes);
		}

		return allVotes;
	}

	/**
	 * Save all inferences in one operation.
	 * Sources are NOT saved here - they are populated after execution in populateInferenceSources.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	async saveInferences(inferences) {
		if (inferences.length === 0) return;

		// Collect neurons only (sources populated later after execution)
		const neurons = [];

		for (const inf of inferences) {
			// is_winner: 1 for winners (highest sum of strength * reward per dimension), 0 for losers
			const isWinner = inf.isWinner ? 1 : 0;
			// expected_reward is weighted average from connection/pattern sources at inference time
			// actual_reward is NULL until channel reward is applied
			neurons.push([inf.neuron_id, inf.level || 0, 0, inf.strength, inf.reward, isWinner]);
		}

		// Save neurons only
		await this.conn.query(
			'INSERT INTO inferred_neurons (neuron_id, level, age, strength, expected_reward, is_winner) VALUES ? ON DUPLICATE KEY UPDATE is_winner = VALUES(is_winner), strength = VALUES(strength)',
			[neurons]
		);

		if (this.debug) {
			const winnerCount = inferences.filter(i => i.isWinner).length;
			const loserCount = inferences.filter(i => !i.isWinner).length;
			console.log(`Saved ${inferences.length} inferences (${winnerCount} winners, ${loserCount} losers)`);
		}
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
			JOIN neurons n ON n.id = inf.neuron_id
			JOIN channels ch ON ch.id = n.channel_id
			JOIN coordinates c ON inf.neuron_id = c.neuron_id
			JOIN dimensions d ON c.dimension_id = d.id
			WHERE inf.age = 0 AND inf.level = 0 AND inf.is_winner = 1 AND n.type = 'action'
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
	 * Populate inference_sources for all inferred neurons at age=1.
	 * Called after recognizeNeurons when connections have been created.
	 * Reverse-engineers collectVotes to find what would have predicted these neurons.
	 * Note: one frame later, so active_neurons are +1 age - use c.distance = an.age (not an.age + 1)
	 * Saves distance so rewards can be applied when age = distance (prediction outcome observed).
	 * Both pattern and connection sources are saved (both contribute to weighted average).
	 */
	async populateInferenceSources() {

		// Connection sources: active_neurons that have connections TO the inferred neuron
		// During inference: c.distance = an.age + 1, now an.age is +1, so c.distance = an.age
		const [connResult] = await this.conn.query(`
			INSERT INTO inference_sources (age, neuron_id, source_type, source_id, distance, inference_strength)
			SELECT inf.age, inf.neuron_id, 'connection', c.id, c.distance,
				c.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level)
			FROM inferred_neurons inf
			JOIN connections c ON c.to_neuron_id = inf.neuron_id
			JOIN active_neurons an ON an.neuron_id = c.from_neuron_id
			WHERE inf.age = 1
			AND c.distance = an.age
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier });

		// Pattern sources: active pattern peaks with pattern_future TO the inferred neuron
		// During inference: c.distance = an.age + 1, now an.age is +1, so c.distance = an.age
		// source_id is pattern_future.id (not connection_id) so each pattern prediction is tracked separately
		const [patternResult] = await this.conn.query(`
			INSERT INTO inference_sources (age, neuron_id, source_type, source_id, distance, inference_strength)
			SELECT inf.age, inf.neuron_id, 'pattern', pf.id, c.distance,
				pf.strength * POW(:decay, c.distance - 1) * POW(:levelMult, an.level)
			FROM inferred_neurons inf
			JOIN connections c ON c.to_neuron_id = inf.neuron_id
			JOIN pattern_future pf ON pf.connection_id = c.id
			JOIN active_neurons an ON an.neuron_id = pf.pattern_neuron_id
			WHERE inf.age = 1
			AND c.distance = an.age
		`, { decay: this.peakTimeDecayFactor, levelMult: this.levelVoteMultiplier });

		if (this.debug) console.log(`Populated ${connResult.affectedRows} connection + ${patternResult.affectedRows} pattern sources`);
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
			if (!neuronCoords.has(row.neuron_id))
				neuronCoords.set(row.neuron_id, new Map());
			neuronCoords.get(row.neuron_id).set(row.dimension_id, row.val);
		}

		return neuronCoords;
	}

	/**
	 * Sets coordinates for neurons in batches to avoid query size limits
	 * @param {Array<{neuron_id: number, coordinates: Object}>} neurons - Array of neuron_id and coordinates pairs
	 */
	async setNeuronCoordinates(neurons) {

		// flatten to rows of [neuron_id, dimension_id, value]
		const rows = neurons.flatMap(({ neuron_id, coordinates }) =>
			Object.entries(coordinates).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));

		// Process in batches to avoid query size limits
		const batchSize = 5000;
		for (let i = 0; i < rows.length; i += batchSize) {
			const batch = rows.slice(i, i + batchSize);
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ? ON DUPLICATE KEY UPDATE val = VALUES(val)', [batch]);
		}
	}

	/**
	 * Creates new neurons and return their IDs.
	 * MySQL guarantees sequential auto-increment IDs.
	 * @param {Array} neurons - Array of [level, type, channel_id] tuples
	 * @returns {Promise<Array<number>>} Array of neuron IDs
	 */
	async createNeurons(neurons) {
		if (neurons.length === 0) return [];
		const insertResult = await this.conn.query('INSERT INTO neurons (level, type, channel_id) VALUES ?', [neurons]);
		const firstNeuronId = insertResult[0].insertId;
		return Array.from({ length: neurons.length }, (_, idx) => firstNeuronId + idx);
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
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 * @returns {Promise<boolean>}
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

		// Activate all pattern neurons (from matched_patterns table) at the next level
		const [patternNeurons] = await this.conn.query('SELECT DISTINCT pattern_neuron_id FROM matched_patterns');
		const patternNeuronIds = patternNeurons.map(row => row.pattern_neuron_id);
		if (patternNeuronIds.length > 0) await this.activateNeurons(patternNeuronIds, level + 1);

		return true;
	}

	/**
	 * Reinforce connections between active neurons.
	 * Creates connections from all active neurons to newly activated (age=0) neurons at the specified level.
	 * Connection rules:
	 * - Events: same level → same level (for pattern context/recognition)
	 * - Events: any level → base level (for prediction/voting - higher levels predict base outcomes)
	 * - Actions: any level event → base actions only
	 * - Action neurons can NEVER be sources - only event neurons can predict
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT f.neuron_id as from_neuron_id, t.neuron_id as to_neuron_id, f.age as distance, 1 as strength
			FROM active_neurons f
			JOIN neurons nf ON nf.id = f.neuron_id
			CROSS JOIN active_neurons t
			JOIN neurons nt ON nt.id = t.neuron_id
            WHERE t.age = 0 AND t.level = :level -- learning connections to newly activated neurons at the requested level
            AND nf.type = 'event' AND f.age > 0 -- learning connections from older event neurons
            AND (
                (nt.type = 'event' AND f.level = t.level) -- same level events (for pattern context)
                OR
                (nt.type = 'event' AND t.level = 0) -- any level → base events (for prediction)
                OR
                (nt.type = 'action' AND t.level = 0) -- any level event → base actions only
            )
			ON DUPLICATE KEY UPDATE connections.strength = GREATEST(:minConnectionStrength, LEAST(:maxConnectionStrength, connections.strength + VALUES(strength)))
		`, { level, minConnectionStrength: this.minConnectionStrength, maxConnectionStrength: this.maxConnectionStrength });
	}

	/**
	 * Populate active_connections table for newly activated neurons at the specified level.
	 * This is called immediately after reinforceConnections in activateNeurons.
	 * Connection validity is enforced at creation time in reinforceConnections.
	 */
	async activateConnections(level) {
		await this.conn.query(`
			INSERT IGNORE INTO active_connections (connection_id, from_neuron_id, to_neuron_id, age)
			SELECT c.id as connection_id, c.from_neuron_id, c.to_neuron_id, 0 as age
			FROM active_neurons f
			-- connections from older neurons to newly activated neurons (distance = age)
			JOIN connections c ON c.from_neuron_id = f.neuron_id AND c.distance = f.age AND c.strength > 0
			-- connections to newly activated neurons at the given level (age=0)    
			JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = :level
		`, { level });
	}

	/**
	 * Match active connections directly to known patterns.
	 * No peak detection needed - we only check neurons that are already known peaks (in pattern_peaks).
	 * Writes results to matched_patterns memory table.
	 * Matches by connection_id (which encodes from_neuron + to_neuron + distance) to preserve temporal structure.
	 * Uses connection overlap (66% threshold) to determine if patterns match.
	 * Pattern_past only contains same-level connections (pattern identity is determined by same-level context).
	 * @param {number} level - The level to match patterns for (peak neuron level)
	 * @returns {Promise<number>} - Number of matched patterns
	 */
	async matchObservedPatterns(level) {
		if (this.debug2) console.log('Matching active connections to known patterns');

		// Clear scratch tables
		await this.conn.query('TRUNCATE matched_patterns');
		await this.conn.query('TRUNCATE matched_pattern_connections');

		// Determine which patterns matched based on overlap threshold
		// A pattern matches if at least some percentage of its pattern_past connections are in active_connections
		const [result] = await this.conn.query(`
            INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
            SELECT pp.peak_neuron_id, pp.pattern_neuron_id
            FROM active_neurons an
			JOIN pattern_peaks pp ON an.neuron_id = pp.peak_neuron_id
			JOIN pattern_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.connection_id = p.connection_id AND ac.age = 0
            WHERE an.level = ?
            AND an.age = 0
            GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
            HAVING SUM(IF(ac.connection_id IS NOT NULL, 1, 0)) >= COUNT(*) * ?
		`, [level, this.mergePatternThreshold]);

		if (this.debug) {
			const [matchedPairs] = await this.conn.query(`
				SELECT mp.peak_neuron_id, mp.pattern_neuron_id
				FROM matched_patterns mp
			`);
			console.log(`Matched ${result.affectedRows} pattern-peak pairs:`,
				matchedPairs.map(p => `peak=${p.peak_neuron_id}, pattern=${p.pattern_neuron_id}`).join('; '));
		}
		if (result.affectedRows === 0) return 0;

		// Now populate matched_pattern_connections ONLY for matched patterns
		// Pattern connections: LEFT JOIN to active to determine if common or missing
		// Cross-level connections included via active_connections.level = target level
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, p.connection_id, IF(ac.connection_id IS NOT NULL, 'common', 'missing') as status
			FROM matched_patterns mp
			JOIN pattern_past p ON mp.pattern_neuron_id = p.pattern_neuron_id
			LEFT JOIN active_connections ac ON ac.connection_id = p.connection_id AND ac.age = 0
		`);

		// Novel connections: active connections TO the peak not in pattern_past
		// Only same-level connections: pattern identity is determined by same-level context only
		await this.conn.query(`
			INSERT INTO matched_pattern_connections (pattern_neuron_id, connection_id, status)
			SELECT mp.pattern_neuron_id, ac.connection_id, 'novel' as status
			FROM matched_patterns mp
			JOIN neurons peak_n ON peak_n.id = mp.peak_neuron_id
			JOIN active_connections ac ON ac.to_neuron_id = mp.peak_neuron_id AND ac.age = 0
			JOIN connections c ON c.id = ac.connection_id
			JOIN neurons from_n ON from_n.id = c.from_neuron_id
			LEFT JOIN pattern_past p ON p.pattern_neuron_id = mp.pattern_neuron_id AND p.connection_id = ac.connection_id
			WHERE p.connection_id IS NULL
			AND from_n.level = peak_n.level
		`);

		return result.affectedRows;
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

		// Add novel connections: connections in active but not in pattern
		await this.conn.query(`
			INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
			SELECT pattern_neuron_id, connection_id, 1
			FROM matched_pattern_connections
			WHERE status = 'novel'
		`);

		// Strengthen common connections: connections in both pattern and active (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_connections mpc ON p.pattern_neuron_id = mpc.pattern_neuron_id AND p.connection_id = mpc.connection_id
			SET p.strength = LEAST(?, p.strength + 1)
			WHERE mpc.status = 'common'
		`, [this.maxConnectionStrength]);

		// Weaken missing connections: connections in pattern but not in active (clamped between minConnectionStrength and maxConnectionStrength)
		await this.conn.query(`
			UPDATE pattern_past p
			JOIN matched_pattern_connections mpc ON p.pattern_neuron_id = mpc.pattern_neuron_id AND p.connection_id = mpc.connection_id
			SET p.strength = GREATEST(?, p.strength - ?)
			WHERE mpc.status = 'missing'
		`, [this.minConnectionStrength, this.patternNegativeReinforcement]);
	}

	/**
	 * Validate event predictions by comparing inferred_neurons (age=1) to active_neurons (age=0).
	 * Populates accuracyStats Map with {correct, total} counts per level.
	 * Only validates event predictions (not actions, which are validated via rewards).
	 * Only validates winners (is_winner=1) since losers are alternative hypotheses that were rejected.
	 */
	async validatePredictions() {

		// Get all event predictions from previous frame (age=1) and check if they came true (age=0)
		// Only validate winners - losers are just alternative hypotheses rejected during conflict resolution
		const [rows] = await this.conn.query(`
			SELECT inf.level, inf.neuron_id, IF(an.neuron_id IS NOT NULL, 1, 0) as is_correct
			FROM inferred_neurons inf
			JOIN neurons n ON n.id = inf.neuron_id
			LEFT JOIN active_neurons an ON an.neuron_id = inf.neuron_id AND an.level = inf.level AND an.age = 0
			WHERE inf.age = 1
			AND n.type = 'event'
			AND inf.is_winner = 1
		`);

		// Aggregate by level
		const levelStats = new Map();
		for (const row of rows) {
			if (!levelStats.has(row.level)) levelStats.set(row.level, { correct: 0, total: 0 });
			const stats = levelStats.get(row.level);
			stats.total++;
			if (row.is_correct) stats.correct++;
		}

		// Update accuracyStats (cumulative across frames)
		for (const [level, stats] of levelStats) {
			if (!this.accuracyStats.has(level)) this.accuracyStats.set(level, { correct: 0, total: 0 });
			const cumulative = this.accuracyStats.get(level);
			cumulative.correct += stats.correct;
			cumulative.total += stats.total;
		}

		if (this.debug && levelStats.size > 0) {
			const debugParts = [];
			for (const [level, stats] of levelStats) {
				const accuracy = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : 'N/A';
				debugParts.push(`L${level}: ${stats.correct}/${stats.total} (${accuracy}%)`);
			}
			console.log(`Validated predictions: ${debugParts.join(', ')}`);
		}
	}

	/**
	 * Populate new_pattern_future with connections that should be in pattern_future of new patterns.
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
	 * 2. Action regret: inferred action winner got negative reward → connections from peak to best loser action
	 *
	 * Returns the number of new pattern future connections found.
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 */
	async populateNewPatternFuture(channelRewards) {
		await this.conn.query(`TRUNCATE new_pattern_future`);
		await this.populatePredictionErrorFuture();
		await this.populateActionRegretFuture(channelRewards);
		const [countResult] = await this.conn.query(`SELECT COUNT(*) as count FROM new_pattern_future`);
		return countResult[0].count;
	}

	/**
	 * Populate prediction error connections into new_pattern_future.
	 * Peak predicted X via connection, but X didn't happen and Y happened instead.
	 * → Create pattern to predict Y from this context.
	 *
	 * Only handles connection inference errors (source_type = 'connection').
	 * Pattern inference errors are handled by mergePatternFuture (strengthen/weaken).
	 *
	 * Works for all levels: if a level N pattern predicts a base event via connection,
	 * and it doesn't happen, create a level N+1 pattern to correct the prediction.
	 */
	async populatePredictionErrorFuture() {
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (connection_id, type)
			SELECT ac.connection_id, 'event'
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			JOIN neurons n_inferred ON n_inferred.id = inf.neuron_id
			JOIN connections c_inferred ON c_inferred.id = isrc.source_id
			JOIN neurons n_peak ON n_peak.id = c_inferred.from_neuron_id
			-- Find active connections FROM the same peak (what actually happened)
			-- Match the prediction distance: peak predicted at distance=D, now at age=D, observation at age=0
			JOIN active_connections ac ON ac.from_neuron_id = c_inferred.from_neuron_id AND ac.age = 0
			JOIN connections c ON c.id = ac.connection_id AND c.distance = isrc.distance
			JOIN neurons n_target ON n_target.id = c.to_neuron_id
			WHERE isrc.age = isrc.distance
			AND isrc.source_type = 'connection'
			AND isrc.inference_strength >= ?
			-- The inferred neuron is an event (action errors are handled by action regret)
			AND n_inferred.type = 'event'
			-- The inference didn't come true (prediction error)
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				WHERE an.neuron_id = isrc.neuron_id
				AND an.age = 0
			)
			-- The peak neuron did NOT use Type 2 inference (check at the distance when inference was made)
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an.neuron_id
				WHERE pp.peak_neuron_id = c_inferred.from_neuron_id
				AND an.age = isrc.distance
			)
			-- The peak neuron has context (connections TO it from older neurons at inference time)
			AND EXISTS (
				SELECT 1 FROM active_connections context_ac
				WHERE context_ac.to_neuron_id = c_inferred.from_neuron_id
				AND context_ac.age = isrc.distance
			)
			-- Action neurons can NEVER be peaks (only events can be peaks)
			AND n_peak.type = 'event'
			-- Only create patterns for event targets (action targets handled by action regret)
			AND n_target.type = 'event'
			-- Pattern future must only contain same-channel predictions
			AND n_target.channel_id = n_peak.channel_id
		`, [this.predictionErrorMinStrength]);
		if (this.debug) console.log(`Found ${result.affectedRows} prediction error connections`);
	}

	/**
	 * Populate action regret connections into new_pattern_future.
	 * Peak predicted action X (winner) via connection, but got negative reward.
	 * → Create pattern to predict the best loser action from this context.
	 *
	 * Only handles connection inference errors (source_type = 'connection').
	 * Pattern inference errors are handled by mergePatternFuture (strengthen/weaken).
	 *
	 * Works for all levels: if a level 1 pattern predicts an action via connection
	 * and it leads to pain, create a level 2 pattern.
	 *
	 * Uses determineConsensus to find the best alternative action (same logic as normal inference).
	 * @param {Map} channelRewards - Map of channel_name -> reward value
	 */
	async populateActionRegretFuture(channelRewards) {

		// Populate actual_reward for action winners so we can detect negative rewards
		// They are applied when age = distance (prediction outcome just observed)
		// Build CASE statement for all channels with rewards
		if (channelRewards.size > 0) {
			const caseWhen = Array.from(channelRewards.entries())
				.map(([name, reward]) => `WHEN ${this.channelNameToId[name]} THEN ${reward}`)
				.join(' ');
			const channelIds = Array.from(channelRewards.keys()).map(name => this.channelNameToId[name]);
			await this.conn.query(`
				UPDATE inferred_neurons inf
				JOIN neurons n ON n.id = inf.neuron_id
				JOIN inference_sources isrc ON isrc.neuron_id = inf.neuron_id AND isrc.age = inf.age
				SET inf.actual_reward = CASE n.channel_id ${caseWhen} END
				WHERE isrc.age = isrc.distance
				AND inf.level = 0 AND inf.is_winner = 1
				AND n.type = 'action'
				AND n.channel_id IN (?)
			`, [channelIds]);
		}

		// Step 1: Find bad action inferences with their peaks
		// Pattern creation happens when age = distance (prediction outcome just observed)
		const [badActionInferences] = await this.conn.query(`
			SELECT isrc.source_id as connection_id, c_inferred.from_neuron_id as peak_neuron_id, inf.neuron_id as bad_action_neuron_id, isrc.distance
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			JOIN neurons n_inferred ON n_inferred.id = inf.neuron_id
			JOIN connections c_inferred ON c_inferred.id = isrc.source_id
			JOIN neurons n_peak ON n_peak.id = c_inferred.from_neuron_id
			WHERE isrc.age = isrc.distance
			AND isrc.source_type = 'connection'
			AND isrc.inference_strength >= ?
			AND inf.is_winner = 1
			AND inf.actual_reward < ?
			-- The inferred neuron is an action
			AND n_inferred.type = 'action'
			-- The peak neuron did NOT use Type 2 inference (check at the distance when inference was made)
			AND NOT EXISTS (
				SELECT 1 FROM active_neurons an
				JOIN pattern_peaks pp ON pp.pattern_neuron_id = an.neuron_id
				WHERE pp.peak_neuron_id = c_inferred.from_neuron_id
				AND an.age = isrc.distance
			)
			-- The peak neuron has context (connections TO it from older neurons at inference time)
			AND EXISTS (
				SELECT 1 FROM active_connections context_ac
				WHERE context_ac.to_neuron_id = c_inferred.from_neuron_id
				AND context_ac.age = isrc.distance
			)
			-- Action neurons can NEVER be peaks (only events can be peaks)
			AND n_peak.type = 'event'
		`, [this.actionRegretMinStrength, this.actionRegretMinPain]);
		if (badActionInferences.length === 0) {
			if (this.debug) console.log('Action regret: no bad action inferences found');
			return;
		}
		if (this.debug) console.log(`Action regret: found ${badActionInferences.length} bad action inferences`);

		// Step 2: Get ACTION loser votes from inference_sources (reconstruct the votes that lost)
		// Only action neurons - event losers are handled by prediction error patterns
		// Get losers at the same distance as the bad inferences
		const [loserVotes] = await this.conn.query(`
			SELECT isrc.neuron_id, isrc.source_type, isrc.source_id, isrc.inference_strength as strength,
				inf.expected_reward as reward, inf.level, NULL as from_type
			FROM inference_sources isrc
			JOIN inferred_neurons inf ON inf.neuron_id = isrc.neuron_id AND inf.age = isrc.age
			JOIN neurons n ON n.id = isrc.neuron_id
			WHERE isrc.age = isrc.distance
			AND inf.is_winner = 0
			AND n.type = 'action'
		`);
		if (loserVotes.length === 0) {
			if (this.debug) console.log('Action regret: no action loser votes found');
			return;
		}
		if (this.debug) console.log(`Action regret: found ${loserVotes.length} action loser votes`);

		// Step 3: Use determineConsensus to find what SHOULD have won among losers
		const consensusResults = await this.determineConsensus(loserVotes);
		const bestLoserNeuronIds = consensusResults.filter(r => r.isWinner).map(r => r.neuron_id);
		if (bestLoserNeuronIds.length === 0) {
			if (this.debug) console.log('Action regret: no consensus winners among losers');
			return;
		}
		if (this.debug) console.log(`Action regret: consensus found ${bestLoserNeuronIds.length} best loser actions`);

		// Step 4: Filter out peaks that already have an active pattern predicting the best loser
		const peakNeuronIds = [...new Set(badActionInferences.map(b => b.peak_neuron_id))];
		const [coveredPeaks] = await this.conn.query(`
			SELECT DISTINCT pp.peak_neuron_id
			FROM pattern_peaks pp
			JOIN active_neurons an ON an.neuron_id = pp.pattern_neuron_id AND an.age = 1
			JOIN pattern_future pf ON pf.pattern_neuron_id = pp.pattern_neuron_id
			JOIN connections c ON c.id = pf.connection_id
			WHERE pp.peak_neuron_id IN (?)
			AND c.to_neuron_id IN (?)
		`, [peakNeuronIds, bestLoserNeuronIds]);
		const coveredPeakIds = new Set(coveredPeaks.map(r => r.peak_neuron_id));
		const uncoveredPeakIds = peakNeuronIds.filter(id => !coveredPeakIds.has(id));
		if (uncoveredPeakIds.length === 0) {
			if (this.debug) console.log('Action regret: all peaks already covered by existing patterns');
			return;
		}
		if (this.debug) console.log(`Action regret: ${uncoveredPeakIds.length} uncovered peaks (${coveredPeakIds.size} already covered)`);

		// Step 5: Insert connections from uncovered peaks to the best losers
		// These connections exist but weren't active because we executed the wrong action
		// Type is 'action' since we're predicting action neurons
		// Pattern future must only contain same-channel predictions
		const [result] = await this.conn.query(`
			INSERT IGNORE INTO new_pattern_future (connection_id, type)
			SELECT c.id, 'action'
			FROM connections c
			JOIN neurons n_peak ON n_peak.id = c.from_neuron_id
			JOIN neurons n_target ON n_target.id = c.to_neuron_id
			WHERE c.from_neuron_id IN (?)
			AND c.to_neuron_id IN (?)
			AND c.distance = 1
			AND n_target.channel_id = n_peak.channel_id
		`, [uncoveredPeakIds, bestLoserNeuronIds]);
		if (this.debug) console.log(`Found ${result.affectedRows} action regret connections`);
	}

	/**
	 * Populate new_patterns table from new_pattern_future.
	 * Finds unique peak neurons (one pattern per peak).
	 * Action neurons can NEVER be peaks - only event neurons and interneurons can be peaks.
	 * Pattern type is inherited from the peak neuron.
	 * Note: Duplicate patterns are prevented upstream by discarding connection inferences
	 * when pattern inferences exist (in aggregateVotesByTarget).
	 * Returns the number of patterns to create.
	 */
	async populateNewPatterns() {
		await this.conn.query(`TRUNCATE new_patterns`);
		// Create separate patterns for each (peak_neuron_id, type) combination
		// This ensures event and action patterns are separate even for the same peak
		const [insertResult] = await this.conn.query(`
			INSERT INTO new_patterns (peak_neuron_id, type)
			SELECT DISTINCT c.from_neuron_id, npf.type
			FROM new_pattern_future npf
			JOIN connections c ON c.id = npf.connection_id
		`);
		return insertResult.affectedRows;
	}

	/**
	 * Create pattern neurons and map them to new_patterns.
	 * Creates neurons at peak_level+1 for each peak neuron.
	 * Pattern neurons use type from new_patterns (event/action) and channel_id from peak neuron.
	 */
	async createPatternNeurons() {

		// Get peak neurons with their levels and channel_id, pattern type from new_patterns
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
	 * Processes all levels in bulk.
	 */
	async createNewPatterns() {

		// Create pattern_peaks entries
		await this.conn.query(`
			INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength)
			SELECT np.pattern_neuron_id, np.peak_neuron_id, 1.0
			FROM new_patterns np
		`);

		// Create pattern_past entries (active connections at age=1 leading TO the peak)
		// This captures the context that was present when the peak was active
		// Only same-level connections: pattern identity is determined by same-level context only
		// Only event neurons in pattern_past: actions shouldn't be part of pattern identity
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

		// Create pattern_future entries: connections FROM peak to target neurons
		// Uses new_pattern_future which contains connection_ids
		// Join on both peak_neuron_id AND type to ensure event/action separation
		const [futureResult] = await this.conn.query(`
			INSERT IGNORE INTO pattern_future (pattern_neuron_id, connection_id, strength)
			SELECT np.pattern_neuron_id, npf.connection_id, 1.0
			FROM new_patterns np
			JOIN connections c ON c.from_neuron_id = np.peak_neuron_id
			JOIN new_pattern_future npf ON npf.connection_id = c.id AND npf.type = np.type
		`);
		if (this.debug) console.log(`Created ${futureResult.affectedRows} pattern_future entries`);
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

		// Step 1: Aggregate by (from_neuron_id, to_neuron_id) - each source neuron votes once per target
		const aggregateBySource = (votes) => {
			const bySource = new Map();
			for (const v of votes) {
				const conn = connMap.get(v.source_id);
				if (!conn) continue;
				const key = conn.from_neuron_id;
				if (!bySource.has(key))
					bySource.set(key, { from_neuron_id: key, strength: 0, rewardSum: 0, rewardCount: 0, from_coords: conn.from_coords, distances: [] });
				const agg = bySource.get(key);
				agg.strength += v.strength;
				agg.rewardSum += v.reward;
				agg.rewardCount++;
				agg.distances.push(conn.distance);
			}
			// Calculate average reward per source
			for (const [_, agg] of bySource)
				agg.reward = agg.rewardCount > 0 ? agg.rewardSum / agg.rewardCount : 0;
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

		// Step 2: Calculate totals - all voters count in denominator for both actions
		// Collect all unique source neurons that voted for either action
		const allVoters = new Set();
		for (const agg of ownAgg) allVoters.add(agg.from_neuron_id);
		for (const agg of outAgg) allVoters.add(agg.from_neuron_id);
		const totalVoters = allVoters.size;

		// Reward = sum of rewards / total voters (missing votes count as 0)
		const ownTotal = {
			str: ownAgg.reduce((s, a) => s + a.strength, 0),
			rwd: totalVoters > 0 ? ownAgg.reduce((s, a) => s + a.rewardSum, 0) / totalVoters : 0
		};
		const outTotal = {
			str: outAgg.reduce((s, a) => s + a.strength, 0),
			rwd: totalVoters > 0 ? outAgg.reduce((s, a) => s + a.rewardSum, 0) / totalVoters : 0
		};

		// Calculate Boltzmann probabilities (exponential with temperature)
		const ownExp = Math.exp(ownTotal.rwd / this.boltzmannTemperature);
		const outExp = Math.exp(outTotal.rwd / this.boltzmannTemperature);
		const sumExp = ownExp + outExp;
		const ownProb = ownExp / sumExp;
		const outProb = outExp / sumExp;

		console.log(`\n=== ACTION VOTES (Cycle ${cycleFrame}/6) ===`);
		console.log(formatAggVotes(ownAgg, `OWN (${ownAgg.length}/${totalVoters} voters, str=${ownTotal.str.toFixed(1)}, avgRwd=${ownTotal.rwd.toFixed(2)}, prob=${(ownProb * 100).toFixed(1)}%)`));
		console.log(formatAggVotes(outAgg, `OUT (${outAgg.length}/${totalVoters} voters, str=${outTotal.str.toFixed(1)}, avgRwd=${outTotal.rwd.toFixed(2)}, prob=${(outProb * 100).toFixed(1)}%)`));
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

		// Group by dimension to find winners
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

		// Aggregate by source neuron for each event
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
