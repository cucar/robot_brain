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

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = []; // channel rewards indexed by age (array of Maps)
		this.frameNumber = 0; // frame number is used for death ledger and diagnostics

		// Database - used for persistent storage - backup and restore
		this.db = this.database ? new Database(this.debug, this.patternForgetRate, this.mergeThreshold) : null;

		// Diagnostics - pure stats tracker. No flags — presentation flags live on the host.
		this.diagnostics = new Diagnostics();

		// Dump - used for creating brain state dumps for debugging
		this.dump = new Dump();

		// Thalamus - relay station for neuron/channel/dimension mappings
		this.thalamus = new Thalamus(this.debug, this.patternForgetRate, this.mergeThreshold, this.errorCorrectionThreshold);

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(this.debug, this.contextLength);
	}

	/**
	 * Register a channel with the brain (legacy class-based path).
	 */
	registerChannel(name, channelClass) {
		this.thalamus.registerChannel(name, channelClass);
	}

	/**
	 * Register a channel spec with the brain (new id-native path).
	 * The caller owns the encoder/trader and passes in a lightweight spec describing the
	 * channel's dimensions. See Thalamus.registerChannelSpec for the spec shape. Returns
	 * the allocated channel ID; dim IDs are written in place onto the Dimension instances
	 * the caller supplied. Coexists with registerChannel() during migration.
	 */
	registerChannelSpec(spec) {
		return this.thalamus.registerChannelSpec(spec);
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
			const snapshot = await this.db.loadSnapshot(this.thalamus.getChannelClasses(), this.thalamus.channelActions);
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
	 * Id-native frame entry point. Accepts raw scalars per dimension and returns
	 * inferred predictions plus per-frame diagnostic byproducts. The brain is a
	 * pure compute function here — no channel I/O, no action dispatch, no printing.
	 *
	 * The return shape bundles everything the host needs from this single call:
	 *   - `inferences`: channelId → per-dimension predictions (winner + continuous)
	 *   - `frame`: per-frame diagnostic data (elapsed, voteDebug). voteDebug is only
	 *     populated when this.debug is set; null otherwise (the resolution is
	 *     expensive, so it's skipped when nobody will read it). Keeping these in
	 *     the return value rather than as Brain instance fields means the host
	 *     never reaches into the core for "last frame" state — the data flows out
	 *     once, with the call that produced it.
	 *
	 * @param {Map<number, Map<number, number>>} inputs - channelId → (dimId → raw scalar)
	 * @param {Map<number, number>} rewards - channelId → reward for previous frame's actions
	 * @returns {{ inferences: Map, frame: { elapsed: number, voteDebug: object|null } }}
	 */
	processInputs(inputs, rewards) {
		const frameStart = performance.now();
		this.frameNumber++;

		// build the current frame from quantized inputs and previously inferred actions
		this.buildFrame(inputs);
		if (this.frame.length === 0)
			return { inferences: new Map(), frame: { elapsed: performance.now() - frameStart, voteDebug: null } };

		// forget connections and patterns in all neurons to avoid curse of dimensionality
		this.cleanupDeadPatterns();

		// slide the temporal window: age active neurons and push the new rewards frame
		this.ageContext(rewards);

		// activate sensory neurons in age=0, level=0 - inputs from the world
		this.activateSensors();

		// process neurons level-by-level in parallel - collects votes inline per level
		const votes = this.processLevels();

		// do inferences with age>0 neurons. Returns the scalar-space inferences plus an
		// optional voteDebug dump (populated only when this.debug is set, null otherwise).
		const { inferences, voteDebug } = this.inferNeurons(votes);

		// accumulate MAPE by comparing continuous event predictions to the actual input scalars
		// (skips dims not registered with the quantizer - those are tracked via channel callbacks)
		this.diagnostics.trackContinuousError(inferences, inputs, this.thalamus.quantizer);

		// Bundle the per-frame byproducts (timing + optional debug data) with the inferences
		// so the host gets everything from one call. No "last frame" state is kept on Brain.
		return { inferences, frame: { elapsed: performance.now() - frameStart, voteDebug } };
	}

	/**
	 * Compose the cumulative/episode-level summary the host renderer uses for the
	 * "Frame N | Neurons: ..." line. Numbers only; the renderer formats units and
	 * decides how to show null (== "no data yet"). Per-frame byproducts (elapsed,
	 * voteDebug) come back from processInputs directly — they are NOT in here.
	 * Apps can tack on app-layer state (e.g. the stocks portfolio P&L) by passing
	 * a tail string to formatFrameSummary().
	 */
	getFrameSummary() {
		return {
			frameNumber: this.frameNumber,
			neuronCount: this.thalamus.getNeuronCount(),
			maxLevel: this.thalamus.getMaxLevel(),
			...this.diagnostics.getStats()
		};
	}

	/**
	 * Snapshot of the start-of-frame state the --diagnostic renderer dumps: incoming
	 * rewards for the previous frame's actions, raw observations being sensed, and
	 * the dimId→name map so the host can humanize dim ids. Returns null when the
	 * frame was empty (nothing to show).
	 */
	getStartFrameInfo() {
		if (this.frame.length === 0) return null;
		return {
			frameNumber: this.frameNumber,
			rewards: this.rewards[0],
			frame: this.frame,
			dimensionIdToName: this.thalamus.dimensionIdToName
		};
	}

	/**
	 * Build this.frame[] from id-keyed inputs: quantize scalars to bucket IDs and push
	 * event coordinates, then append previously-inferred action coordinates from memory
	 * so they participate in this frame's pattern matching.
	 * @param {Map<number, Map<number, number>>} inputs - channelId → (dimId → raw scalar)
	 */
	buildFrame(inputs) {
		this.frame = [];
		const frameActions = this.thalamus.getInferredActions(this.memory.getInferredNeurons());

		// iterate every registered channel (instance or spec) - a channel may contribute events,
		// carry-forward actions from the previous frame's inference, or both
		for (const channelName of this.thalamus.getAllChannelNames()) {
			const channelId = this.thalamus.channelNameToId[channelName];
			const dimMap = inputs.get(channelId);

			// quantize each dimension's scalar to a bucketId and push as event coordinate
			if (dimMap) for (const [dimId, scalar] of dimMap) {
				this.thalamus.quantizer.observe(dimId, scalar);
				const bucketId = this.thalamus.quantizer.quantize(dimId, scalar);
				this.frame.push({ coordinate: { dimId, bucketId }, channel: channelName, type: 'event' });
			}

			// include previously-inferred actions for this channel as sensory inputs
			for (const action of frameActions.get(channelName) || [])
				this.frame.push({ coordinate: action.coordinate, channel: channelName, type: 'action' });
		}
	}

	/**
	 * Slide the temporal window by one frame: push the new channel rewards onto the
	 * rewards history and age all active neurons in memory context (which also
	 * deactivates any neurons that aged out of the window).
	 * @param {Map<number, number>} rewards - channelId → reward for previous frame's actions
	 */
	ageContext(rewards) {
		// push this frame's rewards onto the history, keyed by channel name for now
		const rewardsByName = new Map();
		for (const [channelId, reward] of rewards)
			rewardsByName.set(this.thalamus.channelIdToName[channelId], reward);
		this.rewards.unshift(rewardsByName);
		if (this.rewards.length > this.contextLength) this.rewards.pop();

		// advance the age of every active neuron and drop any that fell off the window
		this.memory.age();
	}

	/**
	 * Legacy entry point - thin shim around processInputs for channels that still own
	 * their I/O and bucketization. Pulls raw events and rewards from every registered
	 * channel, runs the pure compute step, then dispatches the inferred actions back
	 * to the channels. Used while channels migrate to the id-native processInputs path.
	 * @returns Promise<boolean> - true if a frame was processed, false if no data left
	 */
	async processFrame() {

		// collect per-channel, per-dim scalars and per-channel rewards from all channels
		const { inputs, rewards } = await this.collectChannelFrame();
		if (inputs.size === 0 && this.memory.getInferredNeurons().length === 0) return { processed: false, frame: null };

		// run the pure compute step - quantization, aging, pattern matching, inference, diagnostics
		const { inferences, frame } = this.processInputs(inputs, rewards);
		if (inferences.size === 0) return { processed: false, frame };

		// dispatch inferred actions back to the channels (reads the inferences memory saved above)
		await this.executeActions();

		// give the event loop a chance to run other tasks between frames
		await new Promise(resolve => setImmediate(resolve));

		// Return the per-frame diagnostic byproducts so the host renderer can pick them up
		// without reaching into Brain for "last frame" state.
		return { processed: true, frame };
	}

	/**
	 * Pull the upcoming frame's event scalars and reward signals from every channel
	 * and shape them into the id-keyed maps processInputs expects. The channel still
	 * emits name-form coordinates with discrete bucket values; we translate the dim
	 * name to its numeric ID and pass the bucket value through as the scalar (the
	 * Quantizer treats unregistered dims as passthrough).
	 * @returns {Promise<{inputs: Map<number, Map<number, number>>, rewards: Map<number, number>}>}
	 */
	async collectChannelFrame() {
		const inputs = new Map();
		const rewards = new Map();
		const frameActions = this.thalamus.getInferredActions(this.memory.getInferredNeurons());

		// channel.getFrameEvents takes the frame number the events belong to - processInputs
		// will increment this.frameNumber, so we anticipate it here to keep channels in sync
		const nextFrameNumber = this.frameNumber + 1;

		for (const [channelName, channel] of this.thalamus.getChannels()) {
			const channelId = this.thalamus.channelNameToId[channelName];

			// collect this channel's event scalars keyed by dimId (channel emits name-form)
			const dimMap = new Map();
			const channelEvents = await channel.getFrameEvents(nextFrameNumber);
			for (const event of channelEvents) {
				const { dimId, bucketId } = this.thalamus.coordinateNameToId(event);
				dimMap.set(dimId, bucketId);
			}
			if (dimMap.size > 0) inputs.set(channelId, dimMap);

			// request reward only when we actually executed actions for this channel last frame
			if ((frameActions.get(channelName) || []).length > 0)
				rewards.set(channelId, await channel.getRewards());
		}
		return { inputs, rewards };
	}

	/**
	 * activates base level neurons from frame coordinates
	 */
	activateSensors() {

		// bulk find/create neurons for all input points
		const neuronIds = this.getFrameNeurons(this.frame);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// Track inference performance (event accuracy, action rewards, and continuous prediction errors).
		// Diagnostics hands coordinates to channel.calculatePredictionError which expects name-form,
		// so we translate id-form → name-form at this boundary.
		const activeNeuronIds = new Set(this.memory.getNeuronIdsAtAge(0));
		const actualEvents = new Map();
		for (const [channelName, coords] of this.thalamus.getActiveEvents(activeNeuronIds))
			actualEvents.set(channelName, coords.map(c => this.thalamus.coordinateIdToName(c)));
		const inferences = this.thalamus.getInferences(this.memory.getInferredNeurons())
			.map(inf => ({ ...inf, coordinate: this.thalamus.coordinateIdToName(inf.coordinate) }));
		this.diagnostics.trackInferencePerformance(inferences, activeNeuronIds, actualEvents, this.rewards[0], this.thalamus.getChannels());
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];
		for (const point of frame) neuronIds.push(this.thalamus.getNeuronIdForPoint(point.coordinate, point.channel, point.type));
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
				sensoryNeurons, this.rewards, this.frameNumber, newErrorPatternIds
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
	 * @returns {{ inferences: Map, voteDebug: object|null }}
	 *   inferences: channelId → per-dimension inferences in scalar space. winner.value and
	 *     continuous are dequantized via the Quantizer. continuous is the score-weighted
	 *     average of all competing candidates on that dimension. Empty map if no votes.
	 *   voteDebug: fully-resolved vote snapshot for the host-side debug renderer, or null
	 *     when debug is off (the resolution is expensive; we only do it when asked).
	 */
	inferNeurons(votes) {

		// If no inference votes, wait for more data
		if (votes.length === 0) {
			if (this.debug) console.log('No inferences found. Waiting for more data in future frames.');
			return { inferences: new Map(), voteDebug: null };
		}

		// Aggregate votes and determine winners - returns winners plus full candidate map for continuous predictions
		const { inferences, candidates, dimBest } = this.determineConsensus(votes);

		// Ensure every channel has an action - explore if none inferred.
		// Also seeds candidates + dimBest so the scalar-space output includes the fallback action.
		this.ensureChannelActions(inferences, candidates, dimBest);

		// Build the resolved vote dump only when the host will actually render it. The
		// metadata lookups (neuron type/channel/coordinate name) are non-trivial.
		let voteDebug = null;
		if (this.debug) {
			const resolvedVotes = votes.map(v => ({
				targetId: v.neuronId,
				targetType: this.thalamus.getNeuronType(v.neuronId),
				targetChannel: this.thalamus.getNeuronChannel(v.neuronId),
				targetCoordinate: this.thalamus.coordinateIdToName(this.thalamus.getNeuronCoordinate(v.neuronId)),
				voterId: v.voterId,
				voterLevel: this.thalamus.getNeuronLevel(v.voterId),
				voterLabel: this.formatNeuronLabel(v.voterId),
				strength: v.strength,
				reward: v.reward,
				distance: v.distance
			}));
			// Pure data only — no Channel instance refs. The host owns label formatters
			// (encoders/traders/legacy Channel objects) and looks them up by channel name.
			voteDebug = { votes: resolvedVotes, winners: inferences };
		}

		// Save inferences to memory (clears old inferences first)
		this.memory.saveInferredNeurons(inferences);

		// Build the scalar-space output: per-channel, per-dimension winner and continuous prediction
		return { inferences: this.buildInferencesByChannel(candidates, dimBest), voteDebug };
	}

	/**
	 * Build the per-channel, per-dimension inference output in scalar space.
	 * For each dimension that received votes, produces a single entry containing the
	 * winning bucket's dequantized value plus a score-weighted continuous prediction
	 * across all candidates on that dimension. Bucket IDs never appear in the output.
	 *
	 * @param {Map<number, object>} candidates - neuronId → { strength, reward?, probability? }
	 * @param {Map<number, {neuronId, score, strength}>} dimBest - dimId → winning candidate
	 * @returns {Map<number, Array<{dimId, kind, winner: {value, strength, score}, continuous: number}>>}
	 */
	buildInferencesByChannel(candidates, dimBest) {

		// Group every candidate by (channelId, dimId) and accumulate weighted sums for the continuous prediction
		const dims = new Map(); // key `${channelId}:${dimId}` → { channelId, dimId, kind, weightedSum, totalScore }
		for (const [neuronId, candidate] of candidates) {
			const coordinate = this.thalamus.getNeuronCoordinate(neuronId);
			const kind = this.thalamus.getNeuronType(neuronId);
			const channelId = this.thalamus.channelNameToId[this.thalamus.getNeuronChannel(neuronId)];
			const score = kind === 'action' ? candidate.reward : candidate.probability;
			const value = this.thalamus.quantizer.dequantize(coordinate.dimId, coordinate.bucketId);

			const key = `${channelId}:${coordinate.dimId}`;
			let entry = dims.get(key);
			if (!entry) {
				entry = { channelId, dimId: coordinate.dimId, kind, weightedSum: 0, totalScore: 0 };
				dims.set(key, entry);
			}
			entry.weightedSum += score * value;
			entry.totalScore += score;
		}

		// Finalize each dimension entry: resolve winner via dimBest and compute the continuous prediction
		const out = new Map();
		for (const { channelId, dimId, kind, weightedSum, totalScore } of dims.values()) {
			const best = dimBest.get(dimId);
			const winnerCoord = this.thalamus.getNeuronCoordinate(best.neuronId);
			const winnerValue = this.thalamus.quantizer.dequantize(winnerCoord.dimId, winnerCoord.bucketId);
			const continuous = totalScore > 0 ? weightedSum / totalScore : winnerValue;

			if (!out.has(channelId)) out.set(channelId, []);
			out.get(channelId).push({
				dimId,
				kind,
				winner: { value: winnerValue, strength: best.strength, score: best.score },
				continuous
			});
		}
		return out;
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * For events, reward = strength / totalDimensionStrength (likelihood vs alternatives = safety score)
	 * @param {Array} votes - Array of vote objects accumulated by processLevels
	 * @returns {{ inferences: Array, candidates: Map, dimBest: Map }}
	 *   inferences: winner objects for memory/diagnostics.
	 *   candidates: neuronId → aggregated vote data (strength, reward/probability).
	 *   dimBest: dimId → winning candidate ({neuronId, score, strength}).
	 */
	determineConsensus(votes) {

		// Aggregate votes into candidates and dimension totals
		const { candidates, dimTotalStrength } = this.aggregateVotes(votes);

		// Determine the best neuron per dimension
		const dimBest = this.determineDimensionWinners(candidates, dimTotalStrength);

		// Build winner objects from dimension winners
		const winnerIds = new Set([...dimBest.values()].map(w => w.neuronId));
		if (this.debug) console.log(`Determined consensus: ${candidates.size} candidates, ${winnerIds.size} winners`);
		const inferences = this.buildWinners(winnerIds, candidates);
		return { inferences, candidates, dimBest };
	}

	/**
	 * Aggregate votes into candidate neurons and dimension strength totals.
	 * @returns {{ candidates: Map, dimTotalStrength: Map }}
	 */
	aggregateVotes(votes) {
		const candidates = new Map(); // neuronId -> {strength, weightedTotal}
		const dimTotalStrength = new Map(); // dimension -> totalStrength (for events only)
		for (const v of votes) {

			// add the neuron to the candidates if not seen before
			if (!candidates.has(v.neuronId)) candidates.set(v.neuronId, { strength: 0, weightedTotal: 0 });

			// update candidate total strength - this is needed for events and actions both
			const candidate = candidates.get(v.neuronId);
			candidate.strength += v.strength;

			// for actions, calculate the weighted total - for events, accumulate strength on the event's dimension
			if (this.thalamus.getNeuronType(v.neuronId) === 'action') candidate.weightedTotal += v.strength * v.reward;
			else this.addDimStrength(dimTotalStrength, this.thalamus.getNeuronCoordinate(v.neuronId), v.strength);
		}
		return { candidates, dimTotalStrength };
	}

	/**
	 * Determine the best neuron per dimension (events by probability, actions by reward).
	 * @returns {Map} dimension -> {neuronId, score, strength}
	 */
	determineDimensionWinners(candidates, dimTotalStrength) {
		const dimBest = new Map();
		for (const [neuronId, candidate] of candidates) {
			const coordinate = this.thalamus.getNeuronCoordinate(neuronId);

			// for actions, calculate the reward as weighted total / strength - for events, calculate the likelihood of the event
			const candidateType = this.thalamus.getNeuronType(neuronId);
			if (candidateType === 'action') candidate.reward = candidate.strength > 0 ? candidate.weightedTotal / candidate.strength : 0;
			else candidate.probability = this.getEventProbability(candidate.strength, coordinate, dimTotalStrength);

			// set the best neuron for this dimension based on reward or probability, break ties by strength
			const best = dimBest.get(coordinate.dimId);
			const score = candidateType === 'action' ? candidate.reward : candidate.probability;
			if (this.isBetterCandidate(score, candidate.strength, neuronId, best))
				dimBest.set(coordinate.dimId, { neuronId, score, strength: candidate.strength });
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
				neuronId,
				coordinate: this.thalamus.getNeuronCoordinate(neuronId),
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
	 * Add strength to the dimension total for the given coordinate
	 */
	addDimStrength(dimTotalStrength, coordinate, strength) {
		dimTotalStrength.set(coordinate.dimId, (dimTotalStrength.get(coordinate.dimId) || 0) + strength);
	}

	/**
	 * Calculate event likelihood (this neuron's strength / total strength on its dimension)
	 */
	getEventProbability(strength, coordinate, dimTotalStrength) {
		const total = dimTotalStrength.get(coordinate.dimId) || 0;
		return total > 0 ? strength / total : 0;
	}

	/**
	 * Ensure every channel has an action in the inferences array.
	 * If a channel has no inferred action, add an exploration action.
	 */
	ensureChannelActions(inferences, candidates, dimBest) {

		// Find which channels already have an action inferred
		const channelsWithActions = new Set();
		for (const inf of inferences)
			if (this.thalamus.getNeuronType(inf.neuronId) === 'action') channelsWithActions.add(this.thalamus.getNeuronChannel(inf.neuronId));

		// Add exploration action for channels without one (iterates instance- and spec-registered channels)
		for (const channelName of this.thalamus.getAllChannelNames()) {
			if (channelsWithActions.has(channelName)) continue;

			// Skip channels that have no actions defined
			const explorationActionId = this.thalamus.getChannelDefaultAction(channelName);
			if (!explorationActionId) continue;

			// No action inferred for this channel - use the default action for deterministic exploration
			const coordinate = this.thalamus.getNeuronCoordinate(explorationActionId);
			inferences.push({
				neuronId: explorationActionId,
				coordinate,
				channel: channelName,
				strength: 0,
				reward: 0
			});

			// also seed candidates + dimBest so buildInferencesByChannel surfaces this action
			// to processInputs callers (legacy path reads actions from memory; spec path reads the return map)
			if (candidates && !candidates.has(explorationActionId))
				candidates.set(explorationActionId, { strength: 0, reward: 0, weightedTotal: 0 });
			if (dimBest && !dimBest.has(coordinate.dimId))
				dimBest.set(coordinate.dimId, { neuronId: explorationActionId, score: 0, strength: 0 });
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

		// Sensory neurons: format coordinate (translate id-form to name-form for display)
		const coordinate = this.thalamus.getNeuronCoordinate(neuronId);
		const name = this.thalamus.dimensionIdToName[coordinate.dimId] ?? `dim${coordinate.dimId}`;
		return `${name}=${coordinate.bucketId}`;
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
		const deadPatternIds = this.thalamus.reapDeadNeurons(this.frameNumber);
		if (deadPatternIds.length === 0) return;

		// delete dead patterns (with recursive cleanup of context references)
		const deletedPatternIds = this.thalamus.deletePatterns(deadPatternIds, this.frameNumber);

		// verify no deleted patterns are active - that would be a bug
		this.memory.assertNotActive(deletedPatternIds);

		if (this.debug) console.log(`=== CLEANUP COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}
}