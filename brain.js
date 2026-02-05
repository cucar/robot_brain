import { Neuron } from './neurons/neuron.js';
import { Memory } from './memory.js';
import { BrainDB } from './brain-db.js';
import { BrainInit } from './brain-init.js';
import { BrainDiagnostics } from './brain-diagnostics.js';

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

		// pattern learning parameters
		this.maxLevels = 10; // just to prevent against infinite recursion

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)

		//************************************************************
		// persistent data structures (loaded from MySQL, saved on backup)
		//************************************************************

		// All neurons: Map of neuronId -> Neuron object (SensoryNeuron or PatternNeuron)
		// Each neuron contains its own connections/patterns as properties
		// Note: IDs are temporary during transition - will be removed later
		this.neurons = new Map();

		// Fast lookup for sensory neurons by coordinates: valueKey -> SensoryNeuron
		this.neuronsByValue = new Map();

		// initialize channel registry
		this.channels = new Map();

		// Map of channel name -> Set of action neurons (built once in initializeActionNeurons)
		this.channelActions = new Map();

		//************************************************************
		// frame processing scratch data (reset each frame or episode)
		//************************************************************

		// Debugging info and flags
		this.frameNumber = 0;
		this.debug = false;
		this.noDatabase = false; // skip database backup/restore for tests
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info
		this.waitForUserInput = false;

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(this.debug);

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// Helper classes
		this.db = new BrainDB();
		this.initializer = new BrainInit(this.debug);
		this.diagnostics = new BrainDiagnostics(this.debug);
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
		const channel = this.initializer.registerChannel(name, channelClass);
		this.channels.set(name, channel);
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async initDB() {
		if (!this.noDatabase) await this.db.initDB();
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain context...');

		// Reset frame counter for proper context window handling
		this.frameNumber = 0;

		// Clear memory
		this.memory.reset();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		await this.resetContext();
		console.log('Hard resetting brain (all learned data)...');

		// Clear in-memory neuron structures
		this.neurons.clear();
		this.neuronsByValue.clear();
		Neuron.nextId = 1;

		// Clear MySQL tables if using a database
		if (!this.noDatabase) await this.db.truncateTables();
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.diagnostics.resetAccuracyStats();
	}

	/**
	 * Backup brain state from in-memory Neuron objects to MySQL.
	 * Called on shutdown or when job is interrupted.
	 */
	async backupBrain() {
		if (!this.noDatabase)
			await this.db.backupBrain(this.neurons, this.channelNameToId, this.dimensionNameToId);
	}

	/**
	 * initializes the brain and loads dimensions
	 */
	async init() {

		// Initialize channels in DB and get mappings
		if (!this.noDatabase) await this.db.initializeChannels(this.channels);
		Object.assign(this, this.initializer.initializeChannels(this.channels));

		// Initialize dimensions in DB
		if (!this.noDatabase) await this.db.initializeDimensions(this.channels);

		// Load dimension mappings
		Object.assign(this, this.initializer.loadDimensions(this.channels));

		// Load learned data from MySQL
		if (!this.noDatabase)
			Object.assign(this, await this.db.loadAndPopulateNeurons(this.dimensionIdToName, this.neurons, this.neuronsByValue));

		// Pre-create action neurons for all channels
		this.channelActions = this.initializer.initializeActionNeurons(this.channels, this.channelNameToId, this.neurons, this.neuronsByValue);

		// Initialize all registered channels (channel-specific setup)
		await this.initializer.initializeAllChannels(this.channels);
	}

	/**
	 * Get neuron ID by dimension name and value.
	 * Used for diagnostic output to show which neurons correspond to which values.
	 * @param {string} dimensionName - The dimension name
	 * @param {number|string} value - The value to look up
	 * @returns {number|null} - Neuron ID or null if not found
	 */
	getNeuronIdByDimensionValue(dimensionName, value) {
		return this.diagnostics.getNeuronIdByDimensionValue(dimensionName, value, this.neurons);
	}

	/**
	 * Get frame actions for all channels from in-memory inferredNeurons Map.
	 * All entries in inferredNeurons are winners, so just filter by action type.
	 * @returns {Map} - Map of channel names to array of output coordinates
	 */
	getInferredActions() {
		const channelOutputs = new Map();

		for (const [neuronId] of this.memory.getInferences()) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron || neuron.level !== 0 || neuron.type !== 'action') continue;

			if (!channelOutputs.has(neuron.channel)) channelOutputs.set(neuron.channel, []);
			channelOutputs.get(neuron.channel).push(neuron.coordinates);
		}

		return channelOutputs;
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

		// execute the inferred actions in all channels
		await this.executeActions();

		// get rewards from all channels based on executed actions
		await this.getRewards();

		// display diagnostic frame header if enabled
		this.diagnostics.displayFrameHeader(this.frameNumber, this.rewards, this.frame);

		// ---------------------------- FIRST LOOP ----------------------------------

		// age the active neurons in memory context - sliding the temporal window
		// deletion of aged-out neurons is deferred to after pattern learning
		this.memory.age();

		// activate sensory neurons in age=0, level=0 - inputs from the world
		this.activateSensors();

		// ---------------------------- SECOND LOOP ----------------------------------

		// discover and activate patterns using connections in age=0 - start recursion from base level
		this.recognizePatterns();

		// update the (age>0 and age<=contextLength) neurons connections based on observations in age=0
		this.updateConnections();

		// learn new patterns in (age>0 and age<=contextLength) neurons from failed predictions and action regret
		this.learnNewPatterns();

		// deactivate aged-out neurons AFTER pattern learning captured full context (age>contextLength)
		this.memory.deactivateOld();

		// do predictions and outputs in (age>0 and age<=contextLength) neurons - what's going to happen next? and what's our best response?
		this.inferNeurons();

		// ---------------------------- END PROCESSING ----------------------------------

		// forget connections and patterns in (age>0 and age<=contextLength) neurons to avoid curse of dimensionality
		// this should normally not be part of the frame processing and instead should be a separate thread
		this.forgetNeurons();

		// show frame processing summary
		this.diagnostics.printFrameSummary(this.frameNumber, performance.now() - frameStart, this.channels);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');

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
		const frameActions = this.getInferredActions();

		// Process each channel: get inputs from channel, get outputs from previous inference
		for (const [channelName, channel] of this.channels) {
			const channelId = this.channelNameToId[channelName];

			// Get the frame event inputs from the channel
			const channelEvents = await channel.getFrameEvents();
			for (const event of channelEvents)
				this.frame.push({ coordinates: event, channel: channelName, channel_id: channelId, type: 'event' });

			// Get actions from previous inference (guaranteed to exist after first frame)
			const channelActions = frameActions.get(channelName) || [];
			for (const action of channelActions)
				this.frame.push({ coordinates: action, channel: channelName, channel_id: channelId, type: 'action' });
		}

		if (this.debug) console.log(`Processing frame: ${this.frame.length} neurons`);
		if (this.debug) console.log(`frame points: ${JSON.stringify(this.frame)}`);
		if (this.debug) console.log('******************************************************************');
	}

	/**
	 * Execute inferred actions for all channels
	 */
	async executeActions() {
		const channelActions = this.getInferredActions();
		for (const [channelName, channel] of this.channels) {
			const actions = channelActions.get(channelName) || [];
			await channel.executeOutputs(actions);
		}
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getRewards() {
		if (this.debug) console.log('Getting rewards feedback from all channels...');
		this.rewards = new Map();
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const reward = await channel.getRewards();
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
		const neuronIds = this.getFrameNeurons(this.frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// Track inference performance (event accuracy, action rewards, and continuous prediction errors)
		this.diagnostics.trackInferencePerformance(this.memory.getInferences(), this.memory.getNeuronsAtAge(0), this.rewards, this.neurons, this.channels);
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];

		for (const point of frame) {
			const valueKey = Neuron.makeValueKey(point.coordinates);
			let neuron = this.neuronsByValue.get(valueKey);

			// Create new neuron if not found
			if (!neuron) {
				neuron = Neuron.createSensory(point.channel, point.type, point.coordinates);
				const neuronId = neuron.id;
				this.neurons.set(neuronId, neuron);
				this.neuronsByValue.set(valueKey, neuron);
				if (this.debug) console.log(`Created new sensory neuron ${neuronId} for ${valueKey}`);
			}

			neuronIds.push(neuron.id);
		}

		if (neuronIds.length === 0) throw new Error(`Failed to get neurons for frame: ${JSON.stringify(frame)}`);
		return neuronIds;
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<number>} neuronIds - Array of neuron IDs to activate
	 */
	activateNeurons(neuronIds) {
		for (const neuronId of neuronIds) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron) throw new Error(`Neuron ${neuronId} not found in this.neurons`);
			this.memory.activateNeuron(neuron);
		}
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
		const matchedPatterns = peaks.map(peak => peak.matchBestPattern(context)).filter(m => m);

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
		const newActiveNeurons = new Set();
		for (const neuron of this.memory.getNeuronsAtAge(0).keys())
			if (neuron.level === 0) newActiveNeurons.add(neuron);
		if (newActiveNeurons.size === 0) return;

		// Each context neuron learns connections at its own distance
		for (const { neuron, age } of this.memory.getContextNeurons())
			neuron.learnConnectionsAtAge(age, newActiveNeurons, this.rewards, this.channelActions);
	}

	/**
	 * Learn new patterns from prediction errors and action regret.
	 * Iterates over inferringNeurons (neurons that voted for winners) to find prediction errors.
	 * Note: inferringNeurons is aged along with activeNeurons, so ages are aligned.
	 */
	learnNewPatterns() {

		// Get newly active sensory neurons (age=0, level=0 - events and actions)
		const newActiveNeurons = new Set();
		for (const neuron of this.memory.getNeuronsAtAge(0).keys())
			if (neuron.level === 0) newActiveNeurons.add(neuron);
		if (newActiveNeurons.size === 0) return;

		// For each inferring neuron with its context
		let patternCount = 0;
		for (const { neuron, age, votes, context } of this.memory.getInferringNeuronsWithContext()) {

			// Try to learn a pattern at this age
			const newPattern = neuron.learnNewPattern(age, votes, newActiveNeurons, this.rewards, this.channelActions, context);

			// if no pattern was learned, move on to the next neuron
			if (!newPattern) continue;

			// index the new neuron with its id
			this.neurons.set(newPattern.id, newPattern);

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

		// Aggregate votes and determine winners
		const inferences = this.determineConsensus(votes);

		// If no inferences, wait for more data
		if (inferences.length === 0) {
			if (this.debug) console.log('No inferences found. Waiting for more data in future frames.');
			return;
		}

		// Ensure every channel has an action - explore if none inferred
		this.ensureChannelActions(inferences);

		// Save inferences to memory (clears old inferences first)
		this.memory.saveInferences(inferences);
	}

	/**
	 * Ensure every channel has an action in the inferences array.
	 * If a channel has no inferred action, add an exploration action.
	 * @param {Array} inferences - Array of inference objects to modify
	 */
	ensureChannelActions(inferences) {

		// Find which channels already have an action inferred
		const channelsWithActions = new Set();
		for (const inf of inferences) {
			if (!inf.isWinner) continue;
			const neuron = this.neurons.get(inf.neuron_id);
			if (neuron && neuron.level === 0 && neuron.type === 'action')
				channelsWithActions.add(neuron.channel);
		}

		// Add exploration action for channels without one
		for (const [channelName] of this.channels) {
			if (channelsWithActions.has(channelName)) continue;
			// No action inferred for this channel - use first action as exploration
			const explorationAction = this.channelActions.get(channelName).values().next().value;
			inferences.push({ neuron_id: explorationAction.id, strength: 0, isWinner: true });
		}
	}

	/**
	 * Collect votes from active neurons.
	 * Populates inferringNeurons array for pattern learning.
	 * @returns {Array} Array of vote objects for consensus
	 */
	collectVotes() {
		const votes = [];
		let baseCount = 0, patternCount = 0;

		// Clear inferring neurons array
		this.memory.clearInferringNeurons();

		// Collect votes from neurons that can vote
		for (const { neuron, age, activatedPattern } of this.memory.getVotingNeurons()) {

			// Only event neurons (level 0) or pattern neurons can create patterns
			if (neuron.level === 0 && neuron.type !== 'event') continue;

			// if a pattern was activated by the neuron, its inference is suppressed - skip
			if (activatedPattern !== null) continue;

			// get the votes of the neuron
			const neuronVotes = neuron.collectVotesAtAge(age, this.memory.contextLength);
			if (neuronVotes.length === 0) continue;

			// Store votes in inferringNeurons for pattern learning
			this.memory.addVote(neuron, age, neuronVotes);

			// add the votes to the returned array
			for (const v of neuronVotes) {
				votes.push({ fromNeuronId: neuron.id, neuronId: v.toNeuron.id, strength: v.strength, reward: v.reward, distance: v.distance });
				if (neuron.isPattern) patternCount++;
				else baseCount++;
			}
		}

		if (this.debug) console.log(`Collected ${baseCount} base votes, ${patternCount} pattern votes`);
		return votes;
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * @param {Array} votes - Array of vote objects from collectVotes
	 * @returns {Array} Array of inference objects with isWinner flag
	 */
	determineConsensus(votes) {

		// Aggregate votes by neuron
		const aggregated = new Map(); // neuronId -> {neuron, strength, weightedReward}
		for (const v of votes) {
			const neuron = this.neurons.get(v.neuronId);
			if (!aggregated.has(v.neuronId))
				aggregated.set(v.neuronId, { neuron, strength: 0, weightedReward: 0 });
			const agg = aggregated.get(v.neuronId);
			agg.strength += v.strength;
			agg.weightedReward += v.strength * v.reward;
		}

		if (aggregated.size === 0) return [];

		// Build candidates with aggregated strength and reward
		const candidates = [];
		for (const [neuronId, agg] of aggregated) {
			const n = agg.neuron;
			candidates.push({
				neuron_id: neuronId,
				strength: agg.strength,
				reward: agg.weightedReward / agg.strength,
				type: n.type,
				channel: n.channel,
				coordinates: n.coordinates
			});
		}

		// Determine winners per dimension (events by strength, actions by reward)
		const dimBest = new Map(); // dimension -> {neuronId, score}
		for (const c of candidates) {
			const score = c.type === 'action' ? c.reward : c.strength;
			for (const dim of Object.keys(c.coordinates))
				if (!dimBest.has(dim) || score > dimBest.get(dim).score)
					dimBest.set(dim, { neuronId: c.neuron_id, score });
		}

		const winnerIds = new Set([...dimBest.values()].map(w => w.neuronId));

		if (this.debug) console.log(`Determined consensus: ${candidates.length} candidates, ${winnerIds.size} winners`);

		return candidates.map(c => ({ ...c, isWinner: winnerIds.has(c.neuron_id) }));
	}

	/**
	 * Runs the forget cycle, reducing strengths and deleting unused connections/patterns/neurons.
	 * Critical for avoiding curse of dimensionality.
	 */
	forgetNeurons() {

		// Run periodically for cleanup
		this.forgetCounter++;
		if (this.forgetCounter % this.forgetCycles !== 0) return;
		this.forgetCounter = 0;

		const cycleStart = Date.now();
		if (this.debug) console.log('=== FORGET CYCLE STARTING ===');

		// Single parallelizable loop - each neuron handles its own forgetting
		let connectionsUpdated = 0, connectionsDeleted = 0, contextDeleted = 0, peaksDeleted = 0;
		for (const neuron of this.neurons.values()) {
			const stats = neuron.forget();
			connectionsUpdated += stats.connectionsUpdated;
			connectionsDeleted += stats.connectionsDeleted;
			contextDeleted += stats.contextDeleted;
			if (stats.peakDeleted) peaksDeleted++;
		}
		if (this.debug) console.log(`  Connections: ${connectionsUpdated} weakened, ${connectionsDeleted} deleted`);
		if (this.debug) console.log(`  Patterns: context ${contextDeleted}, peaks ${peaksDeleted}`);

		// orphan cleanup (must be done after all neurons finish forgetting)
		this.deleteOrphanedPatterns();

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Delete orphaned pattern neurons (no content, no references, not active).
	 * @returns {number} Number of patterns deleted
	 */
	deleteOrphanedPatterns() {
		const toDelete = [];

		for (const neuron of this.neurons.values()) {
			if (neuron.level === 0) continue;
			if (!this.memory.isNeuronActive(neuron) && neuron.canDelete()) toDelete.push(neuron);
		}

		for (const neuron of toDelete) {
			neuron.cleanup();
			this.neurons.delete(neuron.id);
		}

		const orphanCount = toDelete.length;
		if (this.debug) console.log(`  Orphaned patterns deleted: ${orphanCount}`);
	}

}