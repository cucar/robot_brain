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

		// reset neuron id counter
		Neuron.nextId = 1;

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

		// forget connections and patterns in all neurons to avoid curse of dimensionality
		this.cleanupDeadPatterns();

		// age the active neurons in memory context - sliding the temporal window
		// also deactivates aged-out neurons (now that context was saved with votes in previous frame)
		this.memory.age();

		// activate sensory neurons in age=0, level=0 - inputs from the world
		this.activateSensors();

		// ---------------------------- PARALLEL PROCESSING START ----------------------------------

		// process neurons level-by-level in parallel - collects votes inline per level
		const votes = this.processLevels();

		// do inferences with age>0 neurons - what's going to happen next? and what's our best response?
		this.inferNeurons(votes);

		// ---------------------------- PARALLEL PROCESSING END ----------------------------------

		// execute the inferred actions in all channels
		await this.executeActions();

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
		const neuronIds = this.getFrameNeurons(this.frame);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// Track inference performance (event accuracy, action rewards, and continuous prediction errors)
		const activeNeuronIds = new Set(this.memory.getNeuronIdsAtAge(0));
		const actualEvents = this.thalamus.getActiveEvents(activeNeuronIds);
		const inferences = this.thalamus.getInferences(this.memory.getInferredNeurons());
		this.diagnostics.trackInferencePerformance(inferences, activeNeuronIds, actualEvents, this.rewards[0], this.thalamus.getChannels());
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];
		for (const point of frame) neuronIds.push(this.thalamus.getNeuronIdForPoint(point.coordinates, point.channel, point.type));
		if (neuronIds.length === 0) throw new Error(`Failed to get neurons for frame: ${JSON.stringify(frame)}`);
		// if (this.debug) console.log('frame neurons', neuronIds);
		return neuronIds;
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<number>} neuronIds - Array of neuron IDs to activate
	 */
	activateNeurons(neuronIds) {
		for (const neuronId of neuronIds) this.memory.activateNeuron(neuronId, this.thalamus.getNeuronLevel(neuronId));
	}

	/**
	 * Detects patterns at all levels starting from base - goes as high as possible until no patterns found.
	 * Each level's processFrame pass produces both activations and votes; votes are accumulated
	 * across levels for consensus.
	 * @returns {Array} Accumulated votes across all processed levels
	 */
	processLevels() {
		const channelActionIds = this.thalamus.getChannelActionIds();

		// get the active sensory neurons at level 0
		const sensoryNeurons = this.memory.getLevelAges(0);

		// Get the maximum active level from memory index - this may increase as we process levels
		let maxActiveLevel = this.memory.getMaxActiveLevel();

		// track newly-created error pattern ids so they are excluded from the level
		// pass at their own level (prevents double connection-learning and context leak)
		const newErrorPatternIds = new Set();

		// accumulate votes across levels for consensus
		const votes = [];

		// process neurons level-by-level - each level in parallel
		let level = 0;
		while (true) {
			if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

			// process level: aggregate view, recognize patterns, create error corrections, collect votes
			const { activations, votes: levelVotes } = this.thalamus.processLevel(
				level, this.memory.getLevelNeurons(level), this.memory.depth,
				sensoryNeurons, this.rewards, channelActionIds, this.frameNumber,
				newErrorPatternIds, this.errorCorrectionThreshold
			);

			// activate matched patterns and newly-created error patterns at level+1
			for (const { parentId, patternId, age } of activations) this.memory.activatePattern(patternId, level + 1, parentId, age);

			// if we produced any activations, increment the max active level as needed
			if (activations.length > 0) maxActiveLevel = Math.max(maxActiveLevel, level + 1);

			// accumulate this level's votes for consensus
			votes.push(...levelVotes);

			// if we reached the maximum level and no more patterns are recognized, exit the level processing loop
			if (level >= maxActiveLevel) break;

			// otherwise, increment the level and process the next level
			level++;
		}

		return votes;
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 * @param {Array} votes - Accumulated votes from processLevels
	 */
	inferNeurons(votes) {

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
				voterLevel: this.thalamus.getNeuronLevel(v.voter.id),
				voterLabel: this.formatNeuronLabel(v.voter.id),
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
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * For events, reward = strength / totalDimensionStrength (likelihood vs alternatives = safety score)
	 * @param {Array} votes - Array of vote objects accumulated by processLevels
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
	 * @param {number} neuronId
	 * @returns {string}
	 */
	formatNeuronLabel(neuronId) {

		// Pattern neurons: resolve parent chain to root sensory neuron
		const parentId = this.thalamus.getNeuronParent(neuronId);
		if (this.thalamus.getNeuronLevel(neuronId) > 0 && parentId != null) return this.formatNeuronLabel(parentId);

		// Sensory neurons: format coordinates
		const coordinates = this.thalamus.getNeuronCoordinates(neuronId);
		if (!coordinates) return `n${neuronId}`;
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

		// reap neurons scheduled to die at this frame
		const deadPatterns = this.thalamus.reapDeadNeurons(this.frameNumber);
		if (deadPatterns.length === 0) return;

		// delete dead patterns (with recursive cleanup of context references)
		const deletedPatterns = this.thalamus.deletePatterns(deadPatterns, this.frameNumber);

		// verify no deleted patterns are active - that would be a bug
		this.memory.assertNotActive(deletedPatterns);

		if (this.debug) console.log(`=== CLEANUP COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}
}