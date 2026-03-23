import { Memory } from './memory.js';
import { Database } from './database.js';
import { Diagnostics } from './diagnostics.js';
import { Dump } from './dump.js';
import { Neuron } from './neuron.js';
import { Thalamus } from './thalamus.js';

/**
 * Brain Class
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor(options) {

		// pattern learning parameters
		this.maxLevels = 150; // just to prevent against infinite recursion

		// frame number is used for death ledger and diagnostics
		this.frameNumber = 0;

		// Debugging info and flags
		this.debug = options.debug;
		this.database = options.database; // skip database backup/restore for tests
		this.diagnostic = options.diagnostic; // diagnostic mode - shows detailed inference/conflict resolution info
		this.frameSummary = !options.noSummary; // show frame summary or not
		this.waitForUserInput = options.wait;

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// Database - used for persistent storage - backup and restore
		this.db = this.database ? new Database(options) : null;

		// Diagnostics - used for debug methods and performance tracking
		this.diagnostics = new Diagnostics(this.diagnostic, this.frameSummary);

		// Dump - used for creating brain state dumps for debugging
		this.dump = new Dump();

		// Thalamus - relay station for neuron/channel/dimension mappings
		this.thalamus = new Thalamus(options);

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(options);
	}

	/**
	 * waits for user input to continue - used for debugging
	 */
	waitForUser(message) {
		if (!this.waitForUserInput) return Promise.resolve();
		return this.diagnostics.waitForUser(message);
	}

	/**
	 * Register a channel with the brain
	 */
	registerChannel(name, channelClass) {
		this.thalamus.registerChannel(name, channelClass);
	}

	/**
	 * Get channel by name
	 */
	getChannel(channelName) {
		return this.thalamus.getChannel(channelName);
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async initDB() {
		if (this.database) await this.db.initDB();
	}

	/**
	 * Reset brain memory state for a clean episode start.
	 * Materializes all lazy decay, resets frame counter and death ledger so
	 * the next episode starts clean while preserving learned knowledge.
	 */
	resetContext() {
		console.log('Resetting brain context...');

		// Materialize all lazy decay and reset timestamps so frameNumber can restart at 0
		this.thalamus.materializeAndResetNeurons(this.frameNumber);
		this.frameNumber = 0;

		// Reset accuracy stats
		this.resetAccuracyStats();

		// Clear memory
		this.memory.reset();

		// Reset channel class static state (once per class)
		const channelClasses = new Set();
		for (const [, channel] of this.thalamus.getChannels()) channelClasses.add(channel.constructor);
		for (const ChannelClass of channelClasses) ChannelClass.resetChannelContext();

		// Reset all channel instance states
		for (const [, channel] of this.thalamus.getChannels()) channel.resetContext();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all learned data)...');

		// reset active memory (also resets frameNumber)
		this.resetContext();

		// reset all neurons
		this.thalamus.reset();

		// Clear MySQL tables if using a database
		if (this.database) await this.db.reset();
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.diagnostics.resetAccuracyStats();
	}

	/**
	 * Get accuracy stats (for compatibility with jobs)
	 */
	get accuracyStats() {
		return this.diagnostics.accuracyStats;
	}

	/**
	 * Backup brain state from in-memory Neuron objects to MySQL.
	 * Called on shutdown or when job is interrupted.
	 */
	async backup() {
		if (!this.database) return;
		const neurons = this.thalamus.getNeurons();
		const channelNameToId = this.thalamus.getChannelNameToIdMap();
		const dimensionNameToId = this.thalamus.getDimensionNameToIdMap();
		const channels = this.thalamus.getChannels();
		await this.db.backupChannels(channels);
		await this.db.backupDimensions(channels);
		await this.db.backupNeurons(neurons, channelNameToId, dimensionNameToId);
	}

	/**
	 * Create a dump file with current brain state for debugging and comparison
	 */
	createDump() {
		const neurons = this.thalamus.getNeurons();
		const channelNameToId = this.thalamus.getChannelNameToIdMap();
		const dimensionNameToId = this.thalamus.getDimensionNameToIdMap();
		const channels = this.thalamus.getChannels();
		return this.dump.createDumpFile(neurons, channels, channelNameToId, dimensionNameToId);
	}

	/**
	 * initializes the brain and loads dimensions
	 */
	async init() {

		// Load channels from DB (if enabled)
		if (this.database) {
			const channelClasses = this.thalamus.getChannelClasses();
			const channels = await this.db.loadChannels(channelClasses);
			this.thalamus.setChannels(channels);
		}

		// Instantiate channels that did not come from the database
		this.thalamus.instantiateChannels();

		// Load dimension mappings BEFORE loading neurons (neurons need dimension name lookups)
		this.thalamus.loadDimensionMaps();

		// Load neurons from database (if enabled)
		if (this.database) {
			const channelIdToName = this.thalamus.getChannelIdToNameMap();
			const dimensionIdToName = this.thalamus.getDimensionIdToNameMap();
			const neurons = await this.db.loadNeurons(channelIdToName, dimensionIdToName);
			this.thalamus.setNeurons(neurons);
		}

		// Pre-create action neurons for all channels so that we can explore
		this.thalamus.initializeActionNeurons();

		// Build per-channel action sequence set and pass to memory
		this.memory.setNoActionSequenceChannels(this.thalamus.getNoActionSequenceChannels());
	}

	/**
	 * Get neuron ID by coordinates.
	 * Used for diagnostic output to show which neurons correspond to which values.
	 * @param {object} coordinates - Coordinate object with dimension-value pairs
	 * @returns {Neuron|null} - Neuron or null if not found
	 */
	getNeuronByCoordinates(coordinates) {
		return this.thalamus.getNeuronByCoordinates(coordinates);
	}

	/**
	 * Get all channels (public interface)
	 */
	getChannels() {
		return this.thalamus.getChannels();
	}

	/**
	 * Get episode summary with all diagnostic information
	 * @returns {Object} - Episode summary with accuracy, channel metrics, and aggregate metrics
	 */
	getEpisodeSummary() {
		return {
			frameNumber: this.frameNumber,
			accuracy: this.diagnostics.accuracyStats,
			mispredictions: this.diagnostics.mispredictions,
			channelMetrics: this.thalamus.getChannelMetrics(),
			aggregateMetrics: this.thalamus.getAggregateMetrics()
		};
	}

	/**
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and channel-specific rewards (Map of channel_name -> reward)
	 * @returns Promise<boolean> - true if frame was processed, false if no more data available
	 */
	async processFrame() {
		const frameStart = performance.now();

		// ---------------------------- FRAME I/O ----------------------------------

		// get the current frame from all channels - includes events and previously executed actions
		await this.getFrame();
		if (!this.frame || this.frame.length === 0) return false;

		// get rewards from all channels based on executed actions
		await this.getRewards();

		// display diagnostic frame start if enabled
		this.diagnostics.startFrame(this.frameNumber, this.rewards, this.frame);

		// age the active neurons in memory context - sliding the temporal window
		// also deactivates aged-out neurons (now that context was saved with votes in previous frame)
		this.memory.age();

		// activate sensory neurons in age=0, level=0 - inputs from the world
		this.activateSensors();

		// ---------------------------- PARALLEL PROCESSING START ----------------------------------

		// discover and activate patterns using connections in age=0 - start recursion from base level
		this.recognizePatterns();

		// update the age>0 neurons connections based on observations in age=0
		this.updateConnections();

		// learn new patterns in age>0 neurons from failed predictions and action regret
		this.learnNewPatterns();

		// do inferences with age>0 neurons - what's going to happen next? and what's our best response?
		this.inferNeurons();

		// ---------------------------- PARALLEL PROCESSING END ----------------------------------

		// execute the inferred actions in all channels
		await this.executeActions();

		// forget connections and patterns in all neurons to avoid curse of dimensionality
		// this should normally not be part of the frame processing and instead should be a separate thread
		this.cleanupDeadPatterns();

		// show frame processing summary
		this.diagnostics.endFrame(this.frameNumber, performance.now() - frameStart, this.thalamus.getChannels(), this.thalamus.getNeurons().length);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');

		// give a chance to the event loop to run other tasks
		await new Promise(resolve => setImmediate(resolve));

		// return true to indicate that we have processed the frame successfully
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

		// Get all frame actions from previous frame's inference (from in-memory inferredNeurons)
		const frameActions = this.memory.getInferredActions();

		// Process each channel: get inputs from channel, get outputs from previous inference
		for (const [channelName, channel] of this.thalamus.getChannels()) {

			// Get the frame event inputs from the channel
			const channelEvents = await channel.getFrameEvents(this.frameNumber);
			for (const event of channelEvents)
				this.frame.push({ coordinates: event, channel: channelName, type: 'event' });

			// Get actions from previous inference (guaranteed to exist after first frame)
			const channelActions = frameActions.get(channelName) || [];
			for (const action of channelActions)
				this.frame.push({ coordinates: action.coordinates, channel: channelName, type: 'action' });
		}

		if (this.debug) console.log(`Processing frame: ${this.frame.length} neurons`);
		if (this.debug) console.log(`frame points: ${JSON.stringify(this.frame)}`);
		if (this.debug) console.log('******************************************************************');
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getRewards() {
		if (this.debug) console.log('Getting rewards feedback from all channels...');
		this.rewards = new Map();
		let feedbackCount = 0;

		// Get all actions from previous frame's inference (from in-memory inferredNeurons)
		const frameActions = this.memory.getInferredActions();

		// Get reward for each channel
		for (const [channelName, channel] of this.thalamus.getChannels()) {

			// if there were no actions, nothing to reward
			if ((frameActions.get(channelName) || []).length === 0) continue;

			// get the reward for the channel
			const reward = await channel.getRewards();
			if (this.debug) console.log(`${channelName}: reward ${reward.toFixed(3)}`);
			this.rewards.set(channelName, reward);
			feedbackCount++;
		}

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Received rewards from ${feedbackCount} channels`);
			else console.log('No rewards from any channels');
		}
		if (this.debug && feedbackCount > 0)
			console.log(`Channel rewards:`, Array.from(this.rewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));
	}

	/**
	 * activates base level neurons from frame coordinates
	 */
	activateSensors() {

		// bulk find/create neurons for all input points
		const neurons = this.getFrameNeurons(this.frame);

		// activate the neurons in the in-memory context
		this.activateNeurons(neurons);

		// Track inference performance (event accuracy, action rewards, and continuous prediction errors)
		this.diagnostics.trackInferencePerformance(this.memory.getInferences(), this.memory.getNeuronsAtAge(0), this.rewards, this.thalamus.getChannels());
	}

	/**
	 * Returns neurons for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neurons = [];
		for (const point of frame) neurons.push(this.thalamus.getNeuronForPoint(point.coordinates, point.channel, point.type));
		if (neurons.length === 0) throw new Error(`Failed to get neurons for frame: ${JSON.stringify(frame)}`);
		// if (this.debug) console.log('frame neurons', neurons);
		return neurons;
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<Neuron>} neurons - Array of neurons to activate
	 */
	activateNeurons(neurons) {
		for (const neuron of neurons)
			this.memory.activateNeuron(neuron, this.frameNumber);
	}

	/**
	 * Detects patterns at all levels starting from base - goes as high as possible until no patterns found.
	 */
	recognizePatterns() {
		let level = 0;
		while (true) {
			const patternsFound = this.recognizeLevel(level);
			if (!patternsFound) break;

			level++;
			if (level >= this.maxLevels) {
				console.error('Max level exceeded.');
				break;
			}
		}
	}

	/**
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 */
	recognizeLevel(level) {
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// get active recognizer neurons at this level with their per-age contexts
		const recognizers = this.memory.getRecognizersWithContext(level);
		if (recognizers.length === 0) {
			if (this.debug) console.log(`No active neurons at level ${level}`);
			return false;
		}

		// Match patterns (parallelizable) - each recognizer uses its own age-appropriate context
		// A pattern can only be recognized once (by the first recognizer that matches it)
		const matchedPatterns = [];
		const recognizedPatterns = new Set();
		for (const { neuron: parent, age, context } of recognizers) {
			const match = parent.matchPattern(context, this.frameNumber);
			if (match && !recognizedPatterns.has(match.pattern)) {
				recognizedPatterns.add(match.pattern);
				matchedPatterns.push({ parent, age, match });
			}
		}

		// If no patterns matched, stop here
		if (matchedPatterns.length === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// activate the matched pattern neurons and refine their context
		for (const { parent, age, match } of matchedPatterns) {

			// activate the pattern neuron at the recognizer's age - not at age=0
			const deathFrame = this.memory.activatePattern(match.pattern, parent, age, this.frameNumber);
			this.thalamus.registerDeath(match.pattern, deathFrame);

			// refine the pattern context based on observations
			match.pattern.refineContext(match.common, match.novel, match.missing);
		}

		if (this.debug)
			console.log(`Matched ${matchedPatterns.length} patterns at level ${level}:`,
				matchedPatterns.map(m => `parent=${m.parent.id}, age=${m.age}, pattern=${m.match.pattern.id}`).join('; '));

		// return true to indicate patterns found
		return true;
	}

	/**
	 * updates neuron connections based on observations.
	 * Context neurons (age > 0) learn about newly active neurons (age = 0).
	 */
	updateConnections() {

		// Get newly active sensory neurons (age=0, level=0 - events and actions)
		const newActiveNeurons = this.memory.getNewSensoryNeurons();

		// Each context neuron learns connections at its own distance
		const channelActions = this.thalamus.getAllChannelActions();
		for (const { neuron, age } of this.memory.getContextNeurons()) {

			// do not allow action neurons to learn connections unless action sequences are enabled for their channel
			if (this.memory.skipActionNeuron(neuron)) continue;

			// learn connections from the context neuron to the newly active neurons
			neuron.learnConnections(age, newActiveNeurons, this.rewards, channelActions, this.frameNumber);
		}
	}

	/**
	 * Learn new patterns from prediction errors and action regret.
	 * Iterates over neurons that voted to find prediction errors.
	 */
	learnNewPatterns() {

		// Get active sensory neurons (level=0 - events and actions)
		const sensoryNeurons = this.memory.getSensoryNeurons();

		// ask each neuron if it needs a new error correction pattern
		const requests = this.findErrorCorrectionRequests(sensoryNeurons);

		// create pattern neurons and populate their connections from the future
		this.createErrorPatterns(requests, sensoryNeurons);

		if (this.debug && requests.length > 0) console.log(`Created ${requests.length} error patterns`);
	}

	/**
	 * Ask each neuron if it needs a new error correction pattern.
	 * Only sends newly active neurons (age=0) - that's all the neuron needs for error detection.
	 */
	findErrorCorrectionRequests(sensoryNeurons) {

		// get newly active sensory neurons (age=0, level=0 - events and actions)
		const newActiveNeurons = new Set(sensoryNeurons[0] || []);

		// call each neuron to ask if it needs a new error correction pattern (parallelizable)
		const channelActions = this.thalamus.getAllChannelActions();
		const requests = [];
		for (const { neuron, age, votes, context } of this.memory.getVotersWithContext())
			if (neuron.needsErrorCorrection(votes, newActiveNeurons, this.rewards, channelActions))
				requests.push({ neuron, age, context });
		return requests;
	}

	/**
	 * Create pattern neurons and populate their connections from the future.
	 */
	createErrorPatterns(requests, sensoryNeurons) {
		for (const { neuron, age, context } of requests) {

			// create the new pattern with its future connections
			const pattern = Neuron.createPattern(neuron.level + 1, neuron);
			for (let a = 0; a < age && a < sensoryNeurons.length; a++)
				for (const n of sensoryNeurons[a])
					pattern.createConnection(age - a, n, 1, 0);

			// index the new neuron with its id
			this.thalamus.addNeuron(pattern);

			// activate the pattern neuron at the parent's age and register the pattern for death
			// must happen before adding context — activation calls materializeStrengths which
			// would decay freshly added context entries from lastActivationFrame=0 to currentFrame
			const deathFrame = this.memory.activatePattern(pattern, neuron, age, this.frameNumber);
			this.thalamus.registerDeath(pattern, deathFrame);

			// Set context on patterns and add them to parent routing tables.
			// TODO: When context is moved to the parent, the brain will not need to set the context
			//  on the child. Instead, it will send the context to the parent along with the new pattern neuron id.
			for (const { neuron: ctxNeuron, distance } of context)
				if (ctxNeuron.level === neuron.level)
					pattern.addPatternContext(ctxNeuron, distance, 1);

			// add the pattern to parent routing table
			neuron.addChild(pattern);
		}
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 */
	inferNeurons() {

		// Collect votes from active neurons (suppression handled during collection)
		const votes = this.collectVotes();

		// If no inference votes, wait for more data
		if (votes.length === 0) {
			if (this.debug) console.log('No inferences found. Waiting for more data in future frames.');
			return;
		}

		// Aggregate votes and determine winners
		const inferences = this.determineConsensus(votes);

		// Ensure every channel has an action - explore if none inferred
		this.ensureChannelActions(inferences);

		// call diagnostics to show the debug logs for votes
		if (this.debug) this.diagnostics.debugVotes(votes, inferences, this.thalamus.channels);

		// Save inferences to memory (clears old inferences first)
		this.memory.saveInferences(inferences);
	}

	/**
	 * Collect votes from active neurons. Stores votes and context in activeNeurons for pattern learning.
	 * @returns {Array} Array of vote objects for consensus
	 */
	collectVotes() {
		const votes = [];

		// clear the previous votes before setting new ones
		this.memory.clearVotes();

		// Build all contexts once for all ages/levels
		const contexts = this.memory.getContexts();

		// Collect votes from neurons that can vote
		for (const { voter, age, state } of this.memory.getVotingNeurons()) {

			// if a pattern was activated by the neuron, its inference is suppressed - skip
			if (state.activatedPattern !== null) continue;

			// get the votes of the neuron
			const neuronVotes = voter.vote(age, this.frameNumber);

			// store votes and context in memory for learning if the inference ends up being bad (wrong/painful)
			this.memory.setVotes(voter, age, neuronVotes, contexts.get(`${age}:${voter.level}`) ?? []);

			// add the votes to the returned array
			for (const vote of neuronVotes) votes.push({ voter: voter, ...vote });
		}

		return votes;
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * For events, reward = strength / totalDimensionStrength (likelihood vs alternatives = safety score)
	 * @param {Array} votes - Array of vote objects from collectVotes
	 * @returns {Array} Array of winning inference objects {neuron_id, neuron, strength, reward}
	 */
	determineConsensus(votes) {

		// Aggregate votes into candidates and dimension totals
		const { candidates, dimTotalStrength } = this.aggregateVotes(votes);

		// Determine the best neuron per dimension
		const dimBest = this.determineDimensionWinners(candidates, dimTotalStrength);

		// Build winner objects from dimension winners
		const winnerIds = new Set([...dimBest.values()].map(w => w.neuronId));
		if (this.debug) console.log(`Determined consensus: ${candidates.size} candidates, ${winnerIds.size} winners`);
		return this.buildWinners(winnerIds, candidates);
	}

	/**
	 * Aggregate votes into candidate neurons and dimension strength totals.
	 * @returns {{ candidates: Map, dimTotalStrength: Map }}
	 */
	aggregateVotes(votes) {
		const candidates = new Map(); // neuronId -> {neuron, strength, weightedTotal}
		const dimTotalStrength = new Map(); // dimension -> totalStrength (for events only)
		for (const v of votes) {

			// add the neuron to the candidates if not seen before
			if (!candidates.has(v.neuron.id)) candidates.set(v.neuron.id, { neuron: v.neuron, strength: 0, weightedTotal: 0 });

			// update candidate total strength - this is needed for events and actions both
			const candidate = candidates.get(v.neuron.id);
			candidate.strength += v.strength;

			// for actions, calculate the weighted total - for events, calculate total strengths for each dimension
			if (v.neuron.type === 'action') candidate.weightedTotal += v.strength * v.reward;
			else this.addDimStrength(dimTotalStrength, v.neuron.coordinates, v.strength);
		}
		return { candidates, dimTotalStrength };
	}

	/**
	 * Determine the best neuron per dimension (events by probability, actions by reward).
	 * @returns {Map} dimension -> {neuronId, neuron, score, strength}
	 */
	determineDimensionWinners(candidates, dimTotalStrength) {
		const dimBest = new Map();
		for (const [neuronId, candidate] of candidates) {

			// for actions, calculate the reward as weighted total / strength - for events, calculate the likelihood of the event
			if (candidate.neuron.type === 'action') candidate.reward = candidate.strength > 0 ? candidate.weightedTotal / candidate.strength : 0;
			else candidate.probability = this.getEventProbability(candidate.strength, candidate.neuron.coordinates, dimTotalStrength);

			// set the best neuron for each dimension based on rewards or probabilities, break ties by strength
			for (const dim of Object.keys(candidate.neuron.coordinates)) {
				const best = dimBest.get(dim);
				const score = candidate.neuron.type === 'action' ? candidate.reward : candidate.probability;
				if (this.isBetterCandidate(score, candidate.strength, neuronId, best))
					dimBest.set(dim, { neuronId, neuron: candidate.neuron, score, strength: candidate.strength });
			}
		}
		return dimBest;
	}

	/**
	 * Check if a candidate beats the current best for a dimension.
	 * Compares by score first, then strength, then neuron ID as tiebreaker.
	 */
	isBetterCandidate(score, strength, neuronId, best) {
		if (!best) return true;
		if (score !== best.score) return score > best.score;
		if (strength !== best.strength) return strength > best.strength;
		return neuronId < best.neuronId;
	}

	/**
	 * Build winner inference objects from winning neuron IDs.
	 * @returns {Array} Array of winner objects
	 */
	buildWinners(winnerIds, candidates) {
		const winners = [];
		for (const neuronId of winnerIds) {
			const candidate = candidates.get(neuronId);
			const winner = {
				neuron_id: neuronId,
				neuron: candidate.neuron,
				coordinates: candidate.neuron.coordinates,
				channel: candidate.neuron.channel,
				strength: candidate.strength
			};
			if (candidate.neuron.type === 'action') winner.reward = candidate.reward;
			else winner.probability = candidate.probability;
			winners.push(winner);
		}
		return winners;
	}

	/**
	 * Add strength to dimension totals map
	 */
	addDimStrength(dimTotalStrength, coordinates, strength) {
		for (const dim of Object.keys(coordinates))
			dimTotalStrength.set(dim, (dimTotalStrength.get(dim) || 0) + strength);
	}

	/**
	 * Calculate likelihood (strength / total) averaged across dimensions
	 */
	getEventProbability(strength, coordinates, dimTotalStrength) {

		// if there are no dimensions, error out - this should not happen
		const dimensions = Object.keys(coordinates);
		const dimCount = dimensions.length;
		if (dimCount === 0) throw new Error('Neuron with no dimensions.');

		// calculate the total likelihood of the event and return it
		let totalLikelihood = 0;
		for (const dim of dimensions) {
			const total = dimTotalStrength.get(dim) || 0;
			if (total > 0) totalLikelihood += strength / total;
		}
		return totalLikelihood / dimCount;
	}

	/**
	 * Ensure every channel has an action in the inferences array.
	 * If a channel has no inferred action, add an exploration action.
	 * @param {Array} inferences - Array of winning inference objects to modify
	 */
	ensureChannelActions(inferences) {

		// Find which channels already have an action inferred
		const channelsWithActions = new Set();
		for (const inf of inferences)
			if (inf.neuron.type === 'action') channelsWithActions.add(inf.neuron.channel);

		// Add exploration action for channels without one
		for (const [channelName] of this.thalamus.getChannels()) {
			if (channelsWithActions.has(channelName)) continue;

			// Skip channels that have no actions defined
			const explorationAction = this.thalamus.getChannelDefaultAction(channelName);
			if (!explorationAction) continue;

			// No action inferred for this channel - use the default action for deterministic exploration
			inferences.push({
				neuron_id: explorationAction.id,
				neuron: explorationAction,
				coordinates: explorationAction.coordinates,
				strength: 0,
				reward: 0
			});
		}
	}

	/**
	 * Execute inferred actions for all channels
	 */
	async executeActions() {
		await this.thalamus.executeChannelActions(this.memory.getInferences());
	}

	/**
	 * Runs the cleanup cycle for zombie cleanup only.
	 * With lazy decay, this only deletes items that have decayed to zero effective strength.
	 * Critical for avoiding memory bloat from dead neurons.
	 */
	cleanupDeadPatterns() {
		const cycleStart = Date.now();
		if (this.debug) console.log('=== CLEANUP STARTING ===');

		// reap neurons scheduled to die at or before this frame
		const deadPatterns = this.thalamus.reapDeadNeurons(this.frameNumber);
		if (deadPatterns.length === 0) return;

		// delete dead patterns (with recursive cleanup of context references)
		const deletedPatterns = this.thalamus.deletePatterns(deadPatterns, this.frameNumber);

		// verify no deleted patterns are active - that would be a bug
		this.memory.assertNotActive(deletedPatterns);

		if (this.debug) console.log(`=== CLEANUP COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}
}