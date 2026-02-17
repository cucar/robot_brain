import { Memory } from './memory.js';
import { Database } from './database.js';
import { Diagnostics } from './diagnostics.js';
import { Dump } from './dump.js';
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
		this.maxLevels = 10; // just to prevent against infinite recursion

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.frameNumber = 0;

		// Debugging info and flags
		this.debug = options.debug;
		this.database = options.database; // skip database backup/restore for tests
		this.diagnostic = options.diagnostic; // diagnostic mode - shows detailed inference/conflict resolution info
		this.frameSummary = !options.noSummary; // show frame summary or not
		this.waitForUserInput = options.debug;

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// Database - used for persistent storage - backup and restore
		this.db = this.database ? new Database(this.debug) : null;

		// Diagnostics - used for debug methods and performance tracking
		this.diagnostics = new Diagnostics(this.diagnostic, this.frameSummary);

		// Dump - used for creating brain state dumps for debugging
		this.dump = new Dump();

		// Thalamus - relay station for neuron/channel/dimension mappings
		this.thalamus = new Thalamus(this.debug, options);

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(this.debug);
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
	 * Reset brain memory state for a clean episode start
	 */
	resetContext() {
		console.log('Resetting brain context...');

		// Reset frame counter for proper context window handling
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

		// reset active memory
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
		const neurons = this.thalamus.getAllNeurons();
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
		const neurons = this.thalamus.getAllNeurons();
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
	 * @returns {Object} - Episode summary with accuracy, channel metrics, and portfolio metrics
	 */
	getEpisodeSummary() {
		return {
			frameNumber: this.frameNumber,
			accuracy: this.diagnostics.accuracyStats,
			channelMetrics: this.thalamus.getChannelMetrics(),
			portfolioMetrics: this.thalamus.getPortfolioMetrics()
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

		// ---------------------------- FIRST LOOP ----------------------------------

		// age the active neurons in memory context - sliding the temporal window
		// also deactivates aged-out neurons (now that context was saved with votes in previous frame)
		this.memory.age();

		// activate sensory neurons in age=0, level=0 - inputs from the world
		this.activateSensors();

		// ---------------------------- SECOND LOOP ----------------------------------

		// discover and activate patterns using connections in age=0 - start recursion from base level
		this.recognizePatterns();

		// update the age>0 neurons connections based on observations in age=0
		this.updateConnections();

		// learn new patterns in age>0 neurons from failed predictions and action regret
		this.learnNewPatterns();

		// do inferences with age>0 neurons - what's going to happen next? and what's our best response?
		this.inferNeurons();

		// ---------------------------- END PROCESSING ----------------------------------

		// execute the inferred actions in all channels
		await this.executeActions();

		// forget connections and patterns in all neurons to avoid curse of dimensionality
		// this should normally not be part of the frame processing and instead should be a separate thread
		this.runForgetCycle();

		// show frame processing summary
		this.diagnostics.endFrame(this.frameNumber, performance.now() - frameStart, this.thalamus.getChannels());

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
			const channelEvents = await channel.getFrameEvents();
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
	 * Execute inferred actions for all channels
	 */
	async executeActions() {
		await this.thalamus.executeChannelActions(this.memory.getInferredActions());
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getRewards() {
		if (this.debug) console.log('Getting rewards feedback from all channels...');
		this.rewards = new Map();
		let feedbackCount = 0;

		const frameActions = this.memory.getInferredActions();
		for (const [channelName, channel] of this.thalamus.getChannels()) {
			const actions = frameActions.get(channelName) || [];
			const reward = await channel.getRewards(actions);
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
		for (const point of frame) neurons.push(this.thalamus.getNeuronForPoint(point));
		if (neurons.length === 0) throw new Error(`Failed to get neurons for frame: ${JSON.stringify(frame)}`);
		if (this.debug) console.log('frame neurons', neurons);
		return neurons;
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<Neuron>} neurons - Array of neurons to activate
	 */
	activateNeurons(neurons) {
		for (const neuron of neurons)
			this.memory.activateNeuron(neuron);
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

		// get the peaks and context for this level
		const {peaks, context} = this.memory.getPeaksAndContext(level);
		if (peaks.length === 0) {
			if (this.debug) console.log(`No newly activated neurons at level ${level}`);
			return false;
		}

		// Match patterns (parallelizable) - collect results with peak reference - this is parallelizable
		const matchedPeaks = peaks.map(peak => ({ peak, pattern: peak.matchPattern(context) }));
		const matchedPatterns = matchedPeaks.filter(p => p.pattern);

		// If no patterns matched, stop here
		if (matchedPatterns.length === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// activate the matched pattern neurons
		for (const { peak, pattern } of matchedPatterns)
			this.memory.activatePattern(pattern, peak, 0);

		if (this.debug)
			console.log(`Matched ${matchedPatterns.length} patterns at level ${level}:`,
				matchedPatterns.map(m => `peak=${m.peak.id}, pattern=${m.pattern.id}`).join('; '));

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
		for (const { neuron, age } of this.memory.getContextNeurons())
			neuron.learnConnections(age, newActiveNeurons, this.rewards, channelActions);
	}

	/**
	 * Learn new patterns from prediction errors and action regret.
	 * Iterates over neurons that voted to find prediction errors.
	 */
	learnNewPatterns() {

		// Get newly active sensory neurons (age=0, level=0 - events and actions)
		const newActiveNeurons = this.memory.getNewSensoryNeurons();

		// For each neuron that voted with its context
		const channelActions = this.thalamus.getAllChannelActions();
		let patternCount = 0;
		for (const { neuron, age, votes, context } of this.memory.getVotersWithContext()) {

			// Try to learn a pattern at this age
			const newPattern = neuron.learnNewPattern(age, context, votes, newActiveNeurons, this.rewards, channelActions);

			// if no pattern was learned, move on to the next neuron
			if (!newPattern) continue;

			// index the new neuron with its id
			this.thalamus.addNeuron(newPattern);

			// activate the pattern neuron at the peak's age
			this.memory.activatePattern(newPattern, neuron, age);

			patternCount++;
		}
		if (this.debug && patternCount > 0) console.log(`Created ${patternCount} error patterns`);
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

			// No action inferred for this channel - use the lowest ID action for deterministic exploration
			const actions = this.thalamus.getChannelActions(channelName);
			const explorationAction = [...actions].sort((a, b) => a.id - b.id)[0];
			inferences.push({ neuron_id: explorationAction.id, neuron: explorationAction, strength: 0, reward: 0 });
		}
	}

	/**
	 * Collect votes from active neurons. Stores votes and context in activeNeurons for pattern learning.
	 * @returns {Array} Array of vote objects for consensus
	 */
	collectVotes() {
		const votes = [];

		// Collect votes from neurons that can vote
		for (const { voter, age, state } of this.memory.getVotingNeurons()) {

			// if a pattern was activated by the neuron, its inference is suppressed - skip
			if (state.activatedPattern !== null) continue;

			// get the votes of the neuron
			const neuronVotes = voter.vote(age, 1 / this.memory.contextLength);

			// capture context at voting time for pattern learning
			const context = this.memory.getContextForAge(age, voter.level);

			// store votes and context in memory for learning if the inference ends up being bad (wrong/painful)
			this.memory.setVotes(voter, age, neuronVotes, context);

			// add the votes to the returned array
			for (const vote of neuronVotes) votes.push({ voter: voter, ...vote });
		}

		return votes;
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * @param {Array} votes - Array of vote objects from collectVotes
	 * @returns {Array} Array of winning inference objects {neuron_id, neuron, strength, reward}
	 */
	determineConsensus(votes) {

		// Aggregate candidate neurons
		const candidates = new Map(); // neuronId -> {neuron, strength, weightedReward}
		for (const v of votes) {
			if (!candidates.has(v.neuron.id)) candidates.set(v.neuron.id, { neuron: v.neuron, strength: 0, weightedReward: 0 });
			const candidate = candidates.get(v.neuron.id);
			candidate.strength += v.strength;
			candidate.weightedReward += v.strength * v.reward;
		}

		// Determine winners per dimension (events by strength, actions by reward)
		const dimBest = new Map(); // dimension -> {neuronId, score}
		for (const [neuronId, candidate] of candidates) {
			candidate.reward = candidate.weightedReward / candidate.strength;
			const score = candidate.neuron.type === 'action' ? candidate.reward : candidate.strength;
			for (const dim of Object.keys(candidate.neuron.coordinates))
				if (!dimBest.has(dim) || score > dimBest.get(dim).score)
					dimBest.set(dim, { neuronId, score });
		}

		// Return winners with neuron object reference and reward
		const winnerIds = new Set([...dimBest.values()].map(w => w.neuronId));
		if (this.debug) console.log(`Determined consensus: ${candidates.size} candidates, ${winnerIds.size} winners`);
		const winners = [];
		for (const neuronId of winnerIds) {
			const candidate = candidates.get(neuronId);
			winners.push({
				neuron_id: neuronId,
				neuron: candidate.neuron,
				strength: candidate.strength,
				reward: candidate.reward
			});
		}
		return winners;
	}

	/**
	 * Runs the forget cycle, reducing strengths and deleting unused connections/patterns/neurons.
	 * Critical for avoiding curse of dimensionality.
	 */
	runForgetCycle() {

		// Run periodically for cleanup
		if (this.frameNumber % this.forgetCycles !== 0) return;

		const cycleStart = Date.now();
		if (this.debug) console.log('=== FORGET CYCLE STARTING ===');

		// run forget on all neurons and collect patterns to be deleted after forgetting
		const deadPatterns = this.excludeActiveNeurons(this.thalamus.forgetNeurons());

		// delete dead patterns (with recursive cleanup of context references)
		const deletedPatterns = this.deletePatterns(deadPatterns);

		// clean up deleted patterns from active memory contexts
		for (const pattern of deletedPatterns) this.memory.cleanupDeletedNeuron(pattern);

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Delete dead pattern neurons (no content, no references, not active)
	 * Recursively deletes patterns that become deletable after cleanup
	 * @param {Array<Neuron>} patterns - Initial list of patterns to delete
	 * @returns {Array<Neuron>} - All deleted patterns (for memory cleanup)
	 */
	deletePatterns(patterns) {
		const toDelete = [...patterns];

		while (toDelete.length > 0) {
			const pattern = toDelete.shift();

			// Clean up context references and get newly deletable patterns
			const newlyDeletable = this.excludeActiveNeurons(this.thalamus.deletePattern(pattern));

			// Add newly deletable patterns to the queue
			toDelete.push(...newlyDeletable);
		}

		if (this.debug) console.log(`  Patterns deleted: ${patterns.length}`);
		return patterns;
	}

	/**
	 * excludes active neurons from delete list
	 */
	excludeActiveNeurons(neurons) {
		return neurons.filter(neuron => !this.memory.isNeuronActive(neuron));
	}
}