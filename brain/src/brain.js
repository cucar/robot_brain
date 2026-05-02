import { Memory } from './memory.js';
import { Backup } from './backup.js';
import { Diagnostics } from './diagnostics.js';
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
		this.errorCorrectionMode = options?.errorCorrectionMode ?? 'conservative'; // 'static' | 'conservative' | 'neutral' | 'aggressive' — see Neuron constructor
		this.errorCorrectionThreshold = options?.errorCorrectionThreshold ?? 0.5; // fixed threshold when mode='static'; warmup fallback for dynamic modes
		this.mergeThreshold = options?.mergeThreshold ?? 0.5; // percentage of matched entries needed for context merge
		this.patternForgetRate = options?.patternForgetRate ?? 0.01; // level-1 forget rate (inverse of frames remembered); deeper levels decay by contextLength per level

		// validate the error-correction mode early so a bad CLI flag fails fast
		const validModes = ['static', 'conservative', 'neutral', 'aggressive'];
		if (!validModes.includes(this.errorCorrectionMode))
			throw new Error(`Invalid errorCorrectionMode '${this.errorCorrectionMode}'. Expected one of: ${validModes.join(', ')}`);

		// Debugging info and flags
		this.debug = options?.debug;

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = []; // channel rewards indexed by age (array of Maps)
		this.frameNumber = 0; // frame number is used for death ledger and diagnostics

		// Diagnostics - pure stats tracker. No flags — presentation flags live on the host.
		this.diagnostics = new Diagnostics();

		// Backup - file-based snapshot save/load. Used by the Job runner via --save/--load.
		this.backupStore = new Backup(this.patternForgetRate, this.mergeThreshold, this.contextLength, this.errorCorrectionMode, this.errorCorrectionThreshold);

		// Thalamus - relay station for neuron/channel/dimension mappings
		this.thalamus = new Thalamus(this.debug, this.patternForgetRate, this.mergeThreshold, this.contextLength, this.errorCorrectionMode, this.errorCorrectionThreshold);

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory(this.debug, this.contextLength);
	}

	/**
	 * Register a channel spec with the brain. The caller owns the encoder/trader and
	 * passes in a lightweight spec describing the channel's dimensions. See
	 * Thalamus.registerChannelSpec for the spec shape. Returns `{ channelId, dimensionIds }`
	 * where `dimensionIds` is a map from dim name → allocated ID.
	 */
	registerChannelSpec(spec) {
		return this.thalamus.registerChannelSpec(spec);
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
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 */
	resetBrain() {
		console.log('Hard resetting brain (all learned data)...');

		// reset active memory (also resets frameNumber)
		this.resetContext();

		// reset all neurons
		this.thalamus.reset();
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.diagnostics.resetAccuracyStats();
	}

	/**
	 * Save brain state to a file-based backup under <jobDir>/backups/<timestamp>/.
	 * Save errors are caught and logged inside Backup.save so a failure here
	 * never throws during shutdown.
	 */
	save(jobDir) {
		// Materialize lazy decay before snapshotting so strengths in the file reflect
		// the true post-decay values. Without this, lastActivationFrame (which the
		// snapshot doesn't carry) gets implicitly reset to 0 on load and the next
		// resetContext() applies zero decay — making save/load diverge from a
		// continuous run by exactly the inter-episode decay step.
		this.thalamus.materializeAndResetNeurons(this.frameNumber);
		this.frameNumber = 0;
		return this.backupStore.save(jobDir, this.thalamus.getSnapshot());
	}

	/**
	 * Load the most recent backup from <jobDir>/backups/ and restore it into the
	 * Thalamus. Throws if no backup exists — --load is an explicit user request.
	 */
	load(jobDir) {
		const snapshot = this.backupStore.loadLatest(jobDir, this.thalamus.channelActions, this.thalamus.actionIds);
		this.thalamus.restoreSnapshot(snapshot);
	}

	/**
	 * Get episode summary with all diagnostic information
	 * @returns {Object} - Episode summary with accuracy and mispredictions
	 */
	getEpisodeSummary() {
		return {
			frameNumber: this.frameNumber,
			accuracy: this.diagnostics.accuracyStats,
			mispredictions: this.diagnostics.mispredictions
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
	processFrame(inputs, rewards) {
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
	 * voteDebug) come back from processFrame directly — they are NOT in here.
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
		for (const channelId of this.thalamus.getChannelIds()) {
			const dimMap = inputs.get(channelId);

			// quantize each dimension's scalar to a bucketId and push as event coordinate
			if (dimMap) for (const [dimId, scalar] of dimMap) {
				this.thalamus.quantizer.observe(dimId, scalar);
				const bucketId = this.thalamus.quantizer.quantize(dimId, scalar);
				this.frame.push({ coordinate: { dimId, bucketId }, channelId, type: 'event' });
			}

			// include previously-inferred actions for this channel as sensory inputs
			for (const action of frameActions.get(channelId) || [])
				this.frame.push({ coordinate: action.coordinate, channelId, type: 'action' });
		}
	}

	/**
	 * Slide the temporal window by one frame: push the new channel rewards onto the
	 * rewards history and age all active neurons in memory context (which also
	 * deactivates any neurons that aged out of the window).
	 * @param {Map<number, number>} rewards - channelId → reward for previous frame's actions
	 */
	ageContext(rewards) {

		// push this frame's rewards onto the history (channelId-keyed all the way through)
		this.rewards.unshift(rewards);
		if (this.rewards.length > this.contextLength) this.rewards.pop();

		// advance the age of every active neuron and drop any that fell off the window
		this.memory.age(this.frameNumber);
	}

	/**
	 * activates base level neurons from frame coordinates and tracks inference performance.
	 */
	activateSensors() {

		// bulk find/create neurons for all input points
		const neuronIds = this.getFrameNeurons(this.frame);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// Track event accuracy, action rewards, and misprediction log. Continuous prediction
		// error is tracked separately via diagnostics.trackContinuousError in processFrame().
		this.diagnostics.trackInferencePerformance(this.thalamus.getInferenceResults(
			this.memory.getNeuronIdsAtAge(0),
			this.memory.getInferredNeurons(),
			this.rewards[0]
		));
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinate, channelId, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];
		for (const point of frame) neuronIds.push(this.thalamus.getNeuronIdForPoint(point.coordinate, point.channelId, point.type));
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
			for (const vote of levelVotes) votes.push(vote);

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
				targetChannelId: this.thalamus.getNeuronChannelId(v.neuronId),
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

		// Group every candidate by (channelId, dimId) and accumulate weighted sums for the continuous prediction.
		// Candidates whose bucket has no observed samples yet contribute to the dim entry's existence (so the
		// winner below can still produce output) but not to the weighted sum — we can't mix in a scalar we don't have.
		const dims = new Map(); // key `${channelId}:${dimId}` → { channelId, dimId, kind, weightedSum, totalScore }
		for (const [neuronId, candidate] of candidates) {
			const coordinate = this.thalamus.getNeuronCoordinate(neuronId);
			const kind = this.thalamus.getNeuronType(neuronId);
			const channelId = this.thalamus.getNeuronChannelId(neuronId);

			const key = `${channelId}:${coordinate.dimId}`;
			let entry = dims.get(key);
			if (!entry) {
				entry = { channelId, dimId: coordinate.dimId, kind, weightedSum: 0, totalScore: 0 };
				dims.set(key, entry);
			}

			// Skip candidates with no dequantized value (never-observed bucket) — the dim entry
			// is already registered above so the winner's pass below still emits an inference.
			const value = this.thalamus.quantizer.dequantize(coordinate.dimId, coordinate.bucketId);
			if (value === null) continue;

			const score = kind === 'action' ? candidate.reward : candidate.probability;
			entry.weightedSum += score * value;
			entry.totalScore += score;
		}

		// Finalize each dimension entry: resolve winner via dimBest and compute the continuous prediction.
		// If no candidate contributed a dequantized value and the winner itself has none either, continuous
		// comes out null — downstream consumers (MAPE tracker, etc.) skip null rather than coerce it to zero.
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
				channelId: this.thalamus.getNeuronChannelId(neuronId),
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
			if (this.thalamus.getNeuronType(inf.neuronId) === 'action') channelsWithActions.add(this.thalamus.getNeuronChannelId(inf.neuronId));

		// Add exploration action for channels without one (iterates instance- and spec-registered channels)
		for (const channelId of this.thalamus.getChannelIds()) {
			if (channelsWithActions.has(channelId)) continue;

			// Skip channels that have no actions defined
			const explorationActionId = this.thalamus.getChannelDefaultAction(channelId);
			if (!explorationActionId) continue;

			// No action inferred for this channel - use the default action for deterministic exploration
			const coordinate = this.thalamus.getNeuronCoordinate(explorationActionId);
			inferences.push({
				neuronId: explorationActionId,
				coordinate,
				channelId,
				strength: 0,
				reward: 0
			});

			// also seed candidates + dimBest so buildInferencesByChannel surfaces this action
			// in the per-channel inference return map processFrame emits
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