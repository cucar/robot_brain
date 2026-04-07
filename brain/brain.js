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

		// set hyperparameters
		this.contextLength = options?.contextLength ?? 10; // number of frames a base neuron stays active
		this.errorCorrectionThreshold = options?.errorCorrectionThreshold ?? 0.5; // if the error is below this threshold, no need to create a new correction pattern
		this.mergeThreshold = options?.mergeThreshold ?? 0.5; // percentage of matched entries needed for context merge
		this.patternForgetRate = options?.patternForgetRate ?? 0.01; // how many frames will a pattern be remembered (inverse)

		// Debugging info and flags
		this.debug = options?.debug;
		this.database = options?.database; // skip database backup/restore for tests
		this.waitForUserInput = options?.wait;

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = []; // channel rewards indexed by age (array of Maps)
		this.frameNumber = 0; // frame number is used for death ledger and diagnostics

		// Database - used for persistent storage - backup and restore
		this.db = this.database ? new Database(this.debug, this.patternForgetRate, this.mergeThreshold) : null;

		// Diagnostics - used for debug methods and performance tracking
		this.diagnostics = new Diagnostics(options?.diagnostic, !options?.noSummary);

		// Dump - used for creating brain state dumps for debugging
		this.dump = new Dump();

		// Thalamus - relay station for neuron/channel/dimension mappings
		this.thalamus = new Thalamus(this.debug, this.patternForgetRate, this.mergeThreshold);

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(this.debug, this.contextLength);
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

		// Clear memory and rewards history
		this.memory.reset();
		this.rewards = [];

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
		await this.db.saveSnapshot(this.thalamus.getSnapshot());
	}

	/**
	 * Create a dump file with current brain state for debugging and comparison
	 */
	createDump() {
		return this.dump.saveSnapshot(this.thalamus.getSnapshot());
	}

	/**
	 * initializes the brain and loads dimensions
	 */
	async init() {

		// Load full snapshot from DB (channels, dimensions, neurons)
		if (this.database) {
			const snapshot = await this.db.loadSnapshot(this.thalamus.getChannelClasses());
			this.thalamus.restoreSnapshot(snapshot);
		}

		// Instantiate channels that did not come from the database
		this.thalamus.instantiateChannels();

		// Load dimension mappings for any new channels
		this.thalamus.loadDimensionMaps();

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
		this.diagnostics.startFrame(this.frameNumber, this.rewards[0], this.frame);

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
		this.diagnostics.endFrame(
			this.frameNumber,
			performance.now() - frameStart,
			this.thalamus.getChannels(),
			this.thalamus.getNeuronCount(),
			this.thalamus.getMaxLevel()
		);

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
		const frameActions = this.thalamus.getInferredActions(this.memory.getInferredNeurons());

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
		const rewards = new Map();
		let feedbackCount = 0;

		// Get all actions from previous frame's inference (from in-memory inferredNeurons)
		const frameActions = this.thalamus.getInferredActions(this.memory.getInferredNeurons());

		// Get reward for each channel
		for (const [channelName, channel] of this.thalamus.getChannels()) {

			// if there were no actions, nothing to reward
			if ((frameActions.get(channelName) || []).length === 0) continue;

			// get the reward for the channel
			const reward = await channel.getRewards();
			if (this.debug) console.log(`${channelName}: reward ${reward.toFixed(3)}`);
			rewards.set(channelName, reward);
			feedbackCount++;
		}

		// Age rewards: push current rewards to front, trim to context window
		this.rewards.unshift(rewards);
		if (this.rewards.length > this.contextLength) this.rewards.pop();

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Received rewards from ${feedbackCount} channels`);
			else console.log('No rewards from any channels');
		}
		if (this.debug && feedbackCount > 0)
			console.log(`Channel rewards:`, Array.from(rewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));
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
		const activeNeuronIds = new Set(this.memory.getNeuronsAtAge(0).keys());
		const actualEvents = this.thalamus.getActiveEvents(activeNeuronIds);
		const inferences = this.thalamus.getInferences(this.memory.getInferredNeurons());
		this.diagnostics.trackInferencePerformance(inferences, activeNeuronIds, actualEvents, this.rewards[0], this.thalamus.getChannels());
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
		}
	}

	/**
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 */
	recognizeLevel(level) {
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// Pass memory snapshot to thalamus — it owns all neuron access
		const matchedPatterns = this.thalamus.recognizeLevel(level, this.memory.activeNeurons, this.frameNumber);
		if (matchedPatterns.length === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// Activate matched patterns in memory and register death
		for (const { parent, age, match } of matchedPatterns) {
			const deathFrame = this.memory.activatePattern(match.pattern, parent, age, this.frameNumber);
			this.thalamus.registerDeath(match.pattern, deathFrame);
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
		const neurons = this.thalamus.neurons;

		// Get newly active sensory neurons (age=0, level=0) with metadata for connection learning
		const newActiveNeurons = [];
		const newActiveNeuronIds = new Set();
		for (const neuronId of this.memory.getNeuronsAtAge(0).keys()) {
			const neuron = neurons.get(neuronId);
			if (neuron && neuron.level === 0) {
				newActiveNeurons.push({ id: neuron.id, type: this.thalamus.getNeuronType(neuron.id), channel: this.thalamus.getNeuronChannel(neuron.id) });
				newActiveNeuronIds.add(neuronId);
			}
		}

		// Each context neuron (age > 0) learns connections at its own distance
		const channelActionIds = this.thalamus.getAllChannelActionIds();
		for (let age = 1; age < this.memory.depth; age++)
			for (const neuronId of this.memory.getNeuronsAtAge(age).keys()) {
				const neuron = neurons.get(neuronId);
				if (!neuron || this.thalamus.skipActionNeuron(neuron)) continue;
				neuron.learnConnections(age, newActiveNeurons, newActiveNeuronIds, this.rewards[0], channelActionIds);
			}
	}

	/**
	 * Learn new patterns from prediction errors and action regret.
	 * Iterates over neurons that voted to find prediction errors.
	 */
	learnNewPatterns() {
		const neurons = this.thalamus.neurons;

		// Get active sensory neurons (level=0) indexed by age
		const sensoryNeurons = [];
		for (let age = 0; age < this.memory.depth; age++) {
			const ageNeurons = [];
			for (const neuronId of this.memory.getNeuronsAtAge(age).keys()) {
				const neuron = neurons.get(neuronId);
				if (neuron && neuron.level === 0) ageNeurons.push({ id: neuron.id, type: this.thalamus.getNeuronType(neuron.id), channel: this.thalamus.getNeuronChannel(neuron.id) });
			}
			sensoryNeurons.push(ageNeurons);
		}

		// check for each neuron if it needs a new error correction pattern
		const corrections = this.getErrorCorrections(sensoryNeurons);

		// create pattern neurons and populate their connections from the future
		this.createErrorPatterns(corrections, sensoryNeurons);

		if (this.debug && corrections.length > 0) console.log(`Created ${corrections.length} error patterns`);
	}

	/**
	 * returns the error corrections we need for all voters
	 */
	getErrorCorrections(sensoryNeurons) {
		const neurons = this.thalamus.neurons;

		// get newly active events (age=0, level=0)
		const events = this.getActualEvents(sensoryNeurons);

		// check for each neuron if it needs a new error correction pattern
		const corrections = [];
		for (let age = 1; age < this.memory.depth; age++)
			for (const [neuronId, state] of this.memory.getNeuronsAtAge(age)) {
				if (!state.votes || state.votes.length === 0) continue;
				const neuron = neurons.get(neuronId);
				if (!neuron) continue;
				if (this.needsErrorCorrection(state.votes, events))
					corrections.push({ neuron, age, context: state.context });
			}
		return corrections;
	}

	/**
	 * returns the actual event neuron IDs from the given neurons (new frame)
	 */
	getActualEvents(sensoryNeurons) {
		const events = new Set();
		for (const neuron of sensoryNeurons[0]) if (neuron.type === 'event') events.add(neuron.id);
		return events;
	}

	/**
	 * returns if a neuron needs error correction
	 * based on its inferences in the previous frame and the actual events in the current frame
	 * @returns {boolean} Whether error correction is needed
	 */
	needsErrorCorrection(votes, actualEvents) {
		let failedEvents = 0;
		let totalEvents = 0;
		for (const vote of votes) {
			const votedNeuron = this.thalamus.neurons.get(vote.neuronId);
			if (votedNeuron && this.thalamus.getNeuronType(vote.neuronId) === 'event') {
				totalEvents++;
				if (!actualEvents.has(vote.neuronId)) failedEvents++;
			}
		}
		const eventError = failedEvents / totalEvents;
		// if (eventError > this.errorCorrectionThreshold) console.log('correctError', eventError, totalEvents);
		return eventError > this.errorCorrectionThreshold;
	}

	/**
	 * Create pattern neurons and populate their connections from the future.
	 */
	createErrorPatterns(corrections, sensoryNeurons) {
		const channelActionIds = this.thalamus.getAllChannelActionIds();
		for (const { neuron, age, context } of corrections) {

			// create the new pattern neuron
			const pattern = Neuron.createPattern(neuron.level + 1, neuron.id, this.patternForgetRate, this.mergeThreshold);

			// create the future connections of the pattern from currently observed neurons
			for (let a = 0; a < age && a < sensoryNeurons.length; a++)
				for (const n of sensoryNeurons[a]) {

					// save the event/action - include observed reward for actions - for events it's zero
					const nChannel = this.thalamus.getNeuronChannel(n.id);
					const reward = this.rewards[a].get(nChannel) || 0;
					pattern.createConnection(age - a, n.id, 1, reward);

					// for actions with negative rewards, save an alternative with neutral reward - we'll try it next time
					if (reward < 0) {
						const alt = pattern.findAlternativeAction(age - a, nChannel, n.id, channelActionIds);
						if (alt) pattern.createConnection(age - a, alt, 1, 0);
					}
				}

			// index the new neuron with its id
			this.thalamus.addNeuron(pattern);

			// activate the pattern neuron at the parent's age and register the pattern for death
			// must happen before adding context — activation calls materializeStrength which
			// would decay freshly added context entries from lastActivationFrame=0 to currentFrame
			const deathFrame = this.memory.activatePattern(pattern, neuron, age, this.frameNumber);
			this.thalamus.registerDeath(pattern, deathFrame);

			// Set context on patterns and add them to parent routing tables.
			// TODO: When context is moved to the parent, the brain will not need to set the context
			//  on the child. Instead, it will send the context to the parent along with the new pattern neuron id.
			for (const { neuron: ctxNeuron, distance } of context)
				if (ctxNeuron.level === neuron.level)
					pattern.addPatternContext(ctxNeuron, distance, 1);

			// add the pattern to parent routing table (by ID)
			neuron.addChild(pattern.id);
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

		// call diagnostics to show the debug logs for votes - pre-resolve all neuron data
		if (this.debug) {
			const resolvedVotes = votes.map(v => ({
				targetId: v.neuron.id,
				targetType: this.thalamus.getNeuronType(v.neuron.id),
				targetChannel: this.thalamus.getNeuronChannel(v.neuron.id),
				targetCoords: this.thalamus.getNeuronCoordinates(v.neuron.id),
				voterId: v.voter.id,
				voterLevel: v.voter.level,
				voterLabel: this.formatNeuronLabel(v.voter),
				strength: v.strength,
				reward: v.reward,
				distance: v.distance
			}));
			this.diagnostics.debugVotes(resolvedVotes, inferences, this.thalamus.channels);
		}

		// Save inferences to memory (clears old inferences first)
		this.memory.saveInferredNeurons(inferences);
	}

	/**
	 * Collect votes from active neurons. Stores votes and context in activeNeurons for pattern learning.
	 * @returns {Array} Array of vote objects for consensus
	 */
	collectVotes() {
		const votes = [];
		const neurons = this.thalamus.neurons;

		// clear the previous votes before setting new ones
		this.memory.clearVotes();

		// Build all contexts once for all ages/levels
		const contexts = new Map();
		for (let ctxAge = 1; ctxAge < this.memory.depth; ctxAge++)
			for (const neuronId of this.memory.getNeuronsAtAge(ctxAge).keys()) {
				const neuron = neurons.get(neuronId);
				if (!neuron || this.thalamus.skipActionNeuron(neuron)) continue;
				for (let age = 0; age < ctxAge; age++) {
					const key = `${age}:${neuron.level}`;
					if (!contexts.has(key)) contexts.set(key, []);
					contexts.get(key).push({ neuron, distance: ctxAge - age });
				}
			}

		// Collect votes from neurons that can vote (all ages except the oldest)
		for (let age = 0; age < this.memory.depth - 1; age++)
			for (const [neuronId, state] of this.memory.getNeuronsAtAge(age)) {
				const voter = neurons.get(neuronId);
				if (!voter || this.thalamus.skipActionNeuron(voter)) continue;

				// if a pattern was activated by the neuron, its inference is suppressed - skip
				if (state.activatedPattern !== null) continue;

				// get the votes of the neuron
				const neuronVotes = voter.vote(age);

				// store votes and context in memory for learning if the inference ends up being bad (wrong/painful)
				this.memory.setVotes(neuronId, age, neuronVotes, contexts.get(`${age}:${voter.level}`) ?? []);

				// add the votes to the returned array - resolve neuronId to neuron object for consensus
				for (const vote of neuronVotes) {
					const neuron = neurons.get(vote.neuronId);
					if (neuron) votes.push({ voter, neuron, ...vote });
				}
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
			if (this.thalamus.getNeuronType(v.neuron.id) === 'action') candidate.weightedTotal += v.strength * v.reward;
			else this.addDimStrength(dimTotalStrength, this.thalamus.getNeuronCoordinates(v.neuron.id), v.strength);
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
			const candidateType = this.thalamus.getNeuronType(neuronId);
			if (candidateType === 'action') candidate.reward = candidate.strength > 0 ? candidate.weightedTotal / candidate.strength : 0;
			else candidate.probability = this.getEventProbability(candidate.strength, this.thalamus.getNeuronCoordinates(neuronId), dimTotalStrength);

			// set the best neuron for each dimension based on rewards or probabilities, break ties by strength
			for (const dim of Object.keys(this.thalamus.getNeuronCoordinates(neuronId))) {
				const best = dimBest.get(dim);
				const score = candidateType === 'action' ? candidate.reward : candidate.probability;
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
				coordinates: this.thalamus.getNeuronCoordinates(neuronId),
				channel: this.thalamus.getNeuronChannel(neuronId),
				strength: candidate.strength
			};
			if (this.thalamus.getNeuronType(neuronId) === 'action') winner.reward = candidate.reward;
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
			if (this.thalamus.getNeuronType(inf.neuron.id) === 'action') channelsWithActions.add(this.thalamus.getNeuronChannel(inf.neuron.id));

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
				coordinates: this.thalamus.getNeuronCoordinates(explorationAction.id),
				strength: 0,
				reward: 0
			});
		}
	}

	/**
	 * Format a neuron's coordinates as a display label.
	 * For pattern neurons, resolves up the parent chain to find root sensory coordinates.
	 * @param {Neuron} neuron
	 * @returns {string}
	 */
	formatNeuronLabel(neuron) {
		// Pattern neurons: resolve parent chain to root sensory neuron
		if (neuron.level > 0 && neuron.parentId != null) {
			const parent = this.thalamus.neurons.get(neuron.parentId);
			if (parent) return this.formatNeuronLabel(parent);
			return `n${neuron.id}(parent:${neuron.parentId})`;
		}
		// Sensory neurons: format coordinates
		const coordinates = this.thalamus.getNeuronCoordinates(neuron.id);
		if (!coordinates) return `n${neuron.id}`;
		return Object.entries(coordinates)
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([k, v]) => `${k}=${v}`)
			.join(', ');
	}

	/**
	 * Execute inferred actions for all channels
	 */
	async executeActions() {
		await this.thalamus.executeChannelActions(this.memory.getInferredNeurons());
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