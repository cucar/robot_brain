import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuronId, {strength, reward}>> - predictions
 * - routingTable: Map<patternId, Context> - child pattern contexts
 *
 * All neuron metadata (level, coordinates, channel, type, parent) is stored externally
 * in Thalamus lookup tables. Neurons are pure data processors — they only store learned
 * associations and do pattern matching/voting based on numeric IDs and strengths.
 *
 * Note: Active state (which neurons are active at which ages) and votes are managed
 * by the Brain, not stored on neurons. This allows efficient age-indexed queries.
 *
 * Lazy Decay: Continuous decay based on frames elapsed since last activation.
 * We store lastActivationFrame and compute effective strength on-demand:
 * effectiveStrength = strength - (currentFrame - lastFrame) * rate
 */
export class Neuron {

	// Static counter for assigning unique IDs to neurons
	static nextId = 1;

	/**
	 * constructor - id optional for loading from database
	 * channelActionIds are used for alternative-action lookup during learning. Static after initializeActionNeurons.
	 * shared across all neurons so the map is broadcast once at creation time (no per-frame traffic).
	 */
	constructor(patternForgetRate, mergeThreshold, channelActionIds, id = null) {

		// initialize neuron parameters
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;
		this.channelActionIds = channelActionIds;

		// initialize neuron id if given - update nextId if we're loading a neuron with a specific ID
		this.id = id || Neuron.nextId++;
		if (id && id >= Neuron.nextId) Neuron.nextId = id + 1;

		// initialize synapses
		this.connections = new Map(); // inferences: Map<distance, Map<toNeuronId, {strength, reward}>>
		this.routingTable = new Map(); // routing table: Map<patternId, Context>
		this.contextIndex = new Map(); // inverted index: Map<neuronId, Map<distance, Set<patternId>>>
		this.contextRefs = new Map(); // context references: Map<parentId, Set<distance>>
	}

	/**
	 * returns if there is a connection at distance to a target neuron
	 * @param {number} distance
	 * @param {number} toNeuronId - numeric neuron ID
	 */
	hasConnection(distance, toNeuronId) {
		if (!this.connections.has(distance)) return false;
		const distanceMap = this.connections.get(distance);
		return distanceMap.has(toNeuronId);
	}

	/**
	 * creates a connection at distance to target neuron
	 */
	createConnection(distance, toNeuronId, strength, reward) {
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		this.connections.get(distance).set(toNeuronId, { strength, reward });
	}

	/**
	 * Upsert a connection at distance to the target neuron: create if missing, else
	 * strengthen (smoothing the reward). If the resulting reward is negative, wire an
	 * alternative action with neutral reward so it can be tried next time.
	 * @param {number} distance - Connection distance (1+)
	 * @param {number} toNeuronId - Target neuron id
	 * @param {string} channel - Target neuron's channel (used only for alt-action lookup)
	 * @param {number} reward - Observed reward (0 for events, signed for actions)
	 */
	upsertConnection(distance, toNeuronId, channel, reward) {
		if (this.hasConnection(distance, toNeuronId)) this.strengthenConnection(distance, toNeuronId, reward);
		else this.createConnection(distance, toNeuronId, 1, reward);

		// for actions with negative (smoothed) rewards, save an alternative with neutral reward - we'll try it next time
		const conn = this.connections.get(distance).get(toNeuronId);
		if (conn.reward < 0) {
			const alt = this.findAlternativeAction(distance, channel, toNeuronId);
			if (alt) this.createConnection(distance, alt, 1, 0);
		}
	}

	/**
	 * Bulk-initialize this neuron's connections from a pre-built spec. Single entry point
	 * for wiring a freshly-created pattern neuron: all thalamus-side lookups (channel, reward)
	 * are already resolved in the spec; alternative-action substitution for negative rewards
	 * is resolved locally against this neuron's own (mutating) connection state. Each spec
	 * entry is a single observation, so connections are always created (strength=1) - never
	 * strengthened - even if a later entry targets a slot that a prior alt-action filled.
	 * @param {Array<{distance: number, toNeuronId: number, channel: string, reward: number}>} connections
	 */
	initializeConnections(connections) {
		for (const { distance, toNeuronId, channel, reward } of connections) {

			// save the event/action - include observed reward for actions - for events it's zero
			this.createConnection(distance, toNeuronId, 1, reward);

			// for actions with negative rewards, save an alternative with neutral reward - we'll try it next time
			if (reward < 0) {
				const alt = this.findAlternativeAction(distance, channel, toNeuronId);
				if (alt) this.createConnection(distance, alt, 1, 0);
			}
		}
	}

	/**
	 * updates the connection at distance to target neuron - increments strength and updates reward
	 */
	strengthenConnection(distance, toNeuronId, reward) {
		if (!this.connections.has(distance)) throw new Error('Unknown connection'); // should not happen

		// increment the strength of the connection
		const conn = this.connections.get(distance).get(toNeuronId);
		conn.strength++;

		// update reward with dynamic exponential smoothing - calculates exact expected value based on means
		const alpha = 1 / conn.strength;
		conn.reward = alpha * reward + (1 - alpha) * conn.reward;
	}

	/**
	 * Weaken a connection via negative reinforcement (prediction didn't occur).
	 * Deletes the connection if strength drops to zero or below.
	 * @param {number} distance
	 * @param {number} toNeuronId - numeric neuron ID
	 */
	weakenConnection(distance, toNeuronId) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return;
		const conn = distanceMap.get(toNeuronId);
		if (!conn) return;
		conn.strength--;
		if (conn.strength <= 0) this.deleteConnection(distance, toNeuronId);
	}

	/**
	 * Delete connection at distance to target neuron
	 * @param {number} distance
	 * @param {number} toNeuronId - numeric neuron ID
	 */
	deleteConnection(distance, toNeuronId) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap || !distanceMap.has(toNeuronId)) return false;
		distanceMap.delete(toNeuronId);
		if (distanceMap.size === 0) this.connections.delete(distance);
	}

	/**
	 * returns votes from this neuron at a specific age.
	 * @param {number} age - The age at which this neuron is active
	 * @returns {Array} Array of vote objects {neuronId, strength, reward, distance}
	 */
	vote(age) {

		// use connections of distance one more than the age to get the inferences for the next frame
		const distance = age + 1;

		// get connections at the distance - if there are none, no votes
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];

		// create votes for all connections at the distance and return them
		const result = [];
		for (const [neuronId, conn] of distanceMap)
			if (conn.strength > 0)
				result.push({ neuronId, strength: conn.strength, reward: conn.reward, distance });
		return result;
	}

	/**
	 * sets the activation strength of a child pattern - used when loading from database
	 */
	setChildActivationStrength(patternId, strength, lastFrame = 0) {
		const entry = this.routingTable.get(patternId);
		if (entry) {
			entry.activationStrength = strength;
			entry.lastActivationFrame = lastFrame;
		}
	}

	/**
	 * Get effective activation strength for a child pattern with lazy decay
	 */
	getChildEffectiveActivationStrength(patternId, currentFrame) {
		const entry = this.routingTable.get(patternId);
		if (!entry) return 0;
		return Math.max(0, entry.activationStrength - (currentFrame - entry.lastActivationFrame) * this.patternForgetRate);
	}

	/**
	 * Materialize lazy decay for a child pattern
	 */
	materializeChildStrength(patternId, currentFrame) {
		const entry = this.routingTable.get(patternId);
		if (entry) entry.activationStrength = this.getChildEffectiveActivationStrength(patternId, currentFrame);
	}

	/**
	 * increments activation strength for a child pattern - materializes all owner-scoped lazy decay first
	 * @param {number} patternId - ID of the child pattern
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @returns {number|null} Death frame for pattern neurons, null for sensory neurons
	 */
	strengthenChildActivation(patternId, currentFrame) {
		const entry = this.routingTable.get(patternId);
		if (!entry) return null;

		// update all strengths based on decay rate first
		this.materializeChildStrength(patternId, currentFrame);

		// increment activation strength
		entry.activationStrength++;

		// remember when this happened for lazy decay
		entry.lastActivationFrame = currentFrame;

		// return death frame for pattern neurons
		return currentFrame + Math.ceil(entry.activationStrength / this.patternForgetRate);
	}

	/**
	 * Add a child pattern to the routing table and populate its context
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 * @param {Array<{neuronId: number, distance: number}>} context - The context to add
	 * @param {number} currentFrame - Current frame number for lazy decay and strengthening
	 * @returns {number|null} Death frame for pattern neurons, null for sensory neurons
	 */
	addPattern(patternId, context, currentFrame) {
		this.addChild(patternId);
		for (const { neuronId, distance } of context) this.addContext(patternId, neuronId, distance);
		return this.strengthenChildActivation(patternId, currentFrame);
	}

	/**
	 * add a child pattern to the routing table
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 * @param {number} initialStrength - initial activation strength, useful when loading from database
	 */
	addChild(patternId, initialStrength = 0) {
		if (!this.routingTable.has(patternId)) {
			this.routingTable.set(patternId, {
				context: new Context(),
				activationStrength: initialStrength,
				lastActivationFrame: 0
			});
		}
	}

	/**
	 * Remove a child pattern from this neuron's routing table.
	 * Called by thalamus when deleting a child pattern neuron.
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 */
	removeChild(patternId) {

		// clean up both context and context index for all context entries of this pattern
		const entry = this.routingTable.get(patternId);
		if (!entry) throw new Error(`removeChild: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		for (const ctxEntry of entry.context.getEntries())
			this.removeContext(patternId, ctxEntry.neuronId, ctxEntry.distance);

		// remove the pattern from the routing table
		this.routingTable.delete(patternId);
	}

	/**
	 * returns pattern context entries
	 */
	getPatternContext(patternId) {
		const entry = this.routingTable.get(patternId);
		if (!entry) throw new Error(`getPatternContext: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		return entry.context.getEntries();
	}

	/**
	 * returns all pattern context entries for all patterns in the routing table
	 */
	getRoutingTable() {
		const result = [];
		for (const [patternId, entry] of this.routingTable)
			for (const ctxEntry of entry.context.getEntries())
				result.push({ patternId, ...ctxEntry });
		return result;
	}

	/**
	 * adds an entry to the pattern context by neuron ID
	 */
	addContext(patternId, neuronId, distance, strength = 1) {

		// add the neuron to the pattern context at the given distance
		const entry = this.routingTable.get(patternId);
		if (!entry) throw new Error(`addContext: pattern not found in routing table: ${patternId}`);
		entry.context.addNeuron(neuronId, distance, strength);

		// add the neuron to the context index so that we can search efficiently
		this.addContextIndex(neuronId, distance, patternId);
	}

	/**
	 * adds a neuron to the context index
	 */
	addContextIndex(neuronId, distance, patternId) {
		if (!this.contextIndex.has(neuronId)) this.contextIndex.set(neuronId, new Map());
		const distMap = this.contextIndex.get(neuronId);
		if (!distMap.has(distance)) distMap.set(distance, new Set());
		distMap.get(distance).add(patternId);
	}

	/**
	 * Removes an entry from a child pattern's context.
	 * Returns whether the neuron is no longer referenced by any child pattern in this parent
	 * (i.e., the caller should remove the contextRef on the target neuron).
	 * @param {number} patternId - child pattern ID
	 * @param {number} neuronId - context neuron ID to remove
	 * @param {number} distance - distance of the context entry
	 * @returns {boolean} true if no sibling pattern still references this neuron at this distance
	 */
	removeContext(patternId, neuronId, distance) {

		// remove the neuron from the pattern context at the given distance
		const entry = this.routingTable.get(patternId);
		if (!entry) throw new Error(`removeContext: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		entry.context.remove(neuronId, distance);

		// remove the neuron from the context index
		return this.removeContextIndex(neuronId, distance, patternId);
	}

	/**
	 * Remove a pattern from the context inverted index for a given neuron/distance.
	 * @returns {boolean} true if no pattern references this neuron at this distance anymore (orphaned)
	 */
	removeContextIndex(neuronId, distance, patternId) {
		const distMap = this.contextIndex.get(neuronId);
		if (!distMap) throw new Error(`removeContextIndex: neuron ${neuronId} not found in contextIndex of neuron ${this.id}`);
		const patterns = distMap.get(distance);
		if (!patterns) throw new Error(`removeContextIndex: distance ${distance} not found for neuron ${neuronId} in contextIndex of neuron ${this.id}`);
		patterns.delete(patternId);
		if (patterns.size === 0) {
			distMap.delete(distance);
			if (distMap.size === 0) this.contextIndex.delete(neuronId);
			return true;
		}
		return false;
	}

	/**
	 * Remove all references to a dying neuron from this parent's children's contexts.
	 * Uses the contextIndex to find affected patterns in O(1) per entry.
	 * @param {number} neuronId - ID of the dying context neuron
	 * @param {Set<number>} distances - distances at which the dying neuron was referenced
	 * @returns {Set<number>} pattern IDs whose context was modified (caller checks if deletable)
	 */
	removeContextNeuron(neuronId, distances) {
		const affectedPatterns = new Set();
		const distMap = this.contextIndex.get(neuronId);
		if (!distMap) throw new Error(`removeContextNeuron: neuron ${neuronId} not found in contextIndex of neuron ${this.id}`);

		for (const distance of distances) {
			const patterns = distMap.get(distance);
			if (!patterns) throw new Error(`removeContextNeuron: distance ${distance} for neuron ${neuronId} not found in contextIndex of neuron ${this.id}`);
			// remove from each pattern's context and collect affected pattern IDs
			for (const patternId of patterns) {
				const entry = this.routingTable.get(patternId);
				if (!entry) throw new Error(`removeContextNeuron: pattern ${patternId} not found in routing table of neuron ${this.id}`);
				entry.context.remove(neuronId, distance);
				affectedPatterns.add(patternId);
			}
			// clean up the index entries for this distance
			distMap.delete(distance);
		}
		if (distMap.size === 0) this.contextIndex.delete(neuronId);
		return affectedPatterns;
	}

	/**
	 * Add a context reference from another neuron to this neuron
	 * Called when this neuron is added to another neuron's context
	 */
	addContextRef(referencingNeuronId, distance) {
		this.contextRefs.set(referencingNeuronId, (this.contextRefs.get(referencingNeuronId) ?? new Set()).add(distance));
	}

	/**
	 * Remove a context reference from another neuron to this neuron
	 * Called when this neuron is removed from another neuron's context
	 */
	removeContextRef(referencingNeuronId, distance) {
		const distances = this.contextRefs.get(referencingNeuronId);
		if (!distances) throw new Error(`Context ref not found: ${referencingNeuronId}`);
		distances.delete(distance);
		if (distances.size === 0) this.contextRefs.delete(referencingNeuronId);
	}

	/**
	 * Apply a batch of context-reference updates targeting this neuron.
	 * One call per target neuron per frame (callers aggregate by target).
	 * @param {Array<{type: 'add'|'remove', parentId: number, distance: number}>} updates
	 */
	applyContextRefUpdates(updates) {
		for (const { type, parentId, distance } of updates)
			if (type === 'add') this.addContextRef(parentId, distance);
			else this.removeContextRef(parentId, distance);
	}

	/**
	 * Process a frame for this neuron: derive per-age tasks, learn connections, match patterns,
	 * install pre-created error-correction patterns, and cast votes. One call per active neuron
	 * per frame. Matching is skipped when levelContext is null/empty. Votes are cast for each
	 * eligible voting age unless suppressed by a this-frame match (activate=true) or
	 * error-correction activation at the same age.
	 * @param {Map<number, {activatedPatternId?: number}>} ageStates - This neuron's per-age state
	 * @param {number} memoryDepth - Sliding window depth (used to gate voting and recognizer ages)
	 * @param {Context|null} levelContext - Shared level context (all age>0 entries this frame,
	 *        including neurons created as error-corrections earlier this frame)
	 * @param {Set<number>} newErrorPatternIds - Ids of neurons created as error-corrections earlier
	 *        this frame. Used locally to (a) detect whether *this* neuron is one of them — in
	 *        which case learning, recognition, and correction are skipped this frame so only
	 *        voting runs — and (b) mask those ids out at match time so they don't look like
	 *        unexplained novel entries and unfairly penalize pattern scores. They remain in the
	 *        shared levelContext so their ids propagate into downstream state.context for
	 *        next-frame corrections.
	 * @param {Array<{id: number, channel: string, reward: number}>} actives - Age=0 sensory neurons with pre-resolved rewards
	 * @param {number} currentFrame - Current frame number
	 * @param {Array<{patternId: number, age: number, contextEntries: Array<{neuronId, distance}>}>} corrections
	 *        Pre-created error-correction pattern neurons to install as children at the given ages.
	 * @returns {{ matches, correctionActivations, contextRefUpdates, votes }}
	 *          votes: Array<{age, votes, context}> - one entry per non-suppressed voting age
	 */
	processFrame(ageStates, memoryDepth, levelContext, newErrorPatternIds, actives, currentFrame, corrections = []) {

		// derive locally — ships with the neuron instead of being re-sent each frame
		const isNewErrorPattern = newErrorPatternIds.has(this.id);

		// learn connections across all active ages (age=0 skipped internally)
		this.learnConnections(ageStates, isNewErrorPattern, actives);

		// match patterns if we have context and eligible ages
		const { matches, contextRefUpdates: matchRefs } = this.recognizePatterns(ageStates, memoryDepth, isNewErrorPattern, levelContext, newErrorPatternIds, currentFrame);

		// install pre-created error-correction patterns as children and emit their contextRef adds
		const { correctionActivations, contextRefUpdates: correctionRefs } = this.correctErrors(corrections, currentFrame);

		// cast votes for each eligible age, suppressing any ages that activated a pattern
		const votes = this.generateVotes(ageStates, memoryDepth, levelContext, matches, correctionActivations);

		// return frame processing results
		return { matches, correctionActivations, contextRefUpdates: [...matchRefs, ...correctionRefs], votes };
	}

	/**
	 * Find matching patterns for this parent neuron given the observed level context.
	 * For each age that matches, the matched pattern's context is refined immediately.
	 * Cross-neuron contextRef additions and removals are returned for delivery in a separate
	 * post-match phase. If the same child pattern matches at multiple ages for this parent,
	 * all refined matches are returned in order so side effects remain identical, but only
	 * the first match is flagged as an activation candidate. Matching is skipped entirely for
	 * new error patterns, when levelContext is null/empty, or when no age is eligible.
	 * Eligible ages: non-activated and not the oldest age in the sliding window (where no
	 * future context exists to match against).
	 * @param {Map<number, {activatedPatternId?: number}>} ageStates
	 * @param {number} memoryDepth - Sliding window depth
	 * @param {boolean} isNewErrorPattern - True if this neuron was just created this frame
	 * @param {Context|null} levelContext - Level context for matching, or null if recognition is off
	 * @param {Set<number>} newErrorPatternIds - Ids to mask out of the observed context during
	 *        match scoring (brand-new neurons created earlier this frame)
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @returns {{
	 *   matches: Array<{ patternId, age, score, common, missing, novel, removedRefs, activate, deathFrame }>,
	 *   contextRefUpdates: Array<{ type: 'add'|'remove', neuronId, distance }>
	 * }}
	 */
	recognizePatterns(ageStates, memoryDepth, isNewErrorPattern, levelContext, newErrorPatternIds, currentFrame) {
		const matches = [];
		const contextRefUpdates = [];
		if (isNewErrorPattern || !levelContext || levelContext.size === 0) return { matches, contextRefUpdates };
		const activatedPatternIds = new Set();
		for (const [age, state] of ageStates) {
			if (state.activatedPatternId || age === memoryDepth - 1) continue;

			// active ages are processed in ascending order (most recent first). The first age that
			// produces a match at that age is refined and preserved. More recent ages tend to have
			// the richest available context, so they are processed first.
			const best = this.findBestPatternMatchAtAge(levelContext, age, newErrorPatternIds, currentFrame);
			if (!best) continue; // try older age if there is a match

			// refine the context — returns cross-neuron contextRef side effects for later delivery
			best.removedRefs = this.refineContext(best.patternId, best.common, best.novel, best.missing);
			for (const entry of best.novel)
				contextRefUpdates.push({ type: 'add', neuronId: entry.neuronId, distance: entry.distance });
			for (const ref of best.removedRefs)
				contextRefUpdates.push({ type: 'remove', neuronId: ref.neuronId, distance: ref.distance });

			// activate the matched pattern if it was not elected to be activated already
			best.activate = !activatedPatternIds.has(best.patternId);
			activatedPatternIds.add(best.patternId);

			// strengthen child activation automatically here
			if (best.activate) best.deathFrame = this.strengthenChildActivation(best.patternId, currentFrame);

			// include the best match for the age in the results
			matches.push(best);
		}
		return { matches, contextRefUpdates };
	}

	/**
	 * Install pre-created error-correction pattern neurons as children at the given ages.
	 * Each install adds a child pattern entry to the routing table and emits contextRef adds
	 * for every context entry so target neurons can track this parent.
	 * @param {Array<{patternId: number, age, contextEntries: Array<{neuronId: number, distance: number }>}>} corrections
	 * @param {number} currentFrame - Current frame number for lazy decay / death scheduling
	 * @returns {{
	 *   correctionActivations: Array<{patternId, age, deathFrame}>,
	 *   contextRefUpdates: Array<{type: 'add', neuronId, distance}>
	 * }}
	 */
	correctErrors(corrections, currentFrame) {
		const correctionActivations = [];
		const contextRefUpdates = [];
		for (const { patternId, age, contextEntries } of corrections) {

			// add the pattern to the routing table
			const deathFrame = this.addPattern(patternId, contextEntries, currentFrame);

			// add the pattern to the activations to be returned with its death frame
			correctionActivations.push({ patternId, age, deathFrame });

			// also update the context reference updates to be returned for the new patterns
			for (const { neuronId, distance } of contextEntries)
				contextRefUpdates.push({ type: 'add', neuronId, distance });
		}
		return { correctionActivations, contextRefUpdates };
	}

	/**
	 * Cast votes for each eligible age, suppressing any age that activated a pattern this
	 * frame via either a recognition match (activate=true) or an error-correction install.
	 * Eligible ages: non-activated and younger than the oldest sliding-window slot. Runs for
	 * new error patterns too so their state.context gets populated for next-frame corrections.
	 * The per-age context is reshaped from levelContext locally — contextByAge is a pure reshape
	 * (no extra info), so it's derived here instead of being shipped over the wire.
	 * @param {Map<number, {activatedPatternId?: number}>} ageStates
	 * @param {number} memoryDepth - Sliding window depth
	 * @param {Context|null} levelContext - Shared level context (source for per-age context derivation)
	 * @param {Array<{patternId, age, activate}>} matches - This frame's recognition matches
	 * @param {Array<{patternId, age}>} correctionActivations - This frame's error-correction installs
	 * @returns {Array<{age, votes, context}>} One entry per non-suppressed voting age
	 */
	generateVotes(ageStates, memoryDepth, levelContext, matches, correctionActivations) {

		// determine the suppressed ages based on recognized patterns and error creations
		const suppressedAges = new Set();
		for (const m of matches) if (m.activate) suppressedAges.add(m.age);
		for (const c of correctionActivations) suppressedAges.add(c.age);

		// cast votes for each eligible, non-suppressed age
		const votes = [];
		for (const [age, state] of ageStates) {
			if (state.activatedPatternId || age >= memoryDepth - 1) continue;
			if (suppressedAges.has(age)) continue;
			votes.push({ age, votes: this.vote(age), context: this.deriveContextAtAge(levelContext, age) });
		}
		return votes;
	}

	/**
	 * Derive the per-age voting context from the shared levelContext. For voting age `a`,
	 * returns every levelContext entry whose context-age is > a, with distance shifted to
	 * be relative to age `a` (distance = ctxAge - a). This is a pure reshape — the caller
	 * could ship contextByAge pre-computed, but doing it locally saves per-frame wire traffic.
	 * @param {Context|null} levelContext
	 * @param {number} age
	 * @returns {Array<{neuronId: number, distance: number}>}
	 */
	deriveContextAtAge(levelContext, age) {
		const result = [];
		if (!levelContext) return result;
		for (const [neuronId, distanceMap] of levelContext.entries)
			for (const ctxAge of distanceMap.keys())
				if (ctxAge > age) result.push({ neuronId, distance: ctxAge - age });
		return result;
	}

	/**
	 * Find the best matching pattern for a specific active age.
	 * @param {Context} observed - The level context (distances are absolute ages)
	 * @param {number} age - The specific active age being evaluated
	 * @param {Set<number>} excludeIds - Neuron ids to mask out of the observed context (e.g.
	 *        brand-new error-correction patterns created earlier this frame)
	 * @param {number} currentFrame - The current frame number to check activation strength
	 * @returns {{ patternId, age, score, common, missing, novel }|null}
	 */
	findBestPatternMatchAtAge(observed, age, excludeIds, currentFrame) {
		let best = null; // { patternId, age, score, common, missing, novel }

		// Use the inverted index to narrow the search to child patterns that share at least one
		// exact neuron/distance entry with the observed context at this active age.
		const candidateIds = this.getPatternCandidatesAtAge(observed, age);
		if (candidateIds.size === 0) return null;

		// go through the candidate patterns and find the best match
		for (const patternId of candidateIds) {

			// check if pattern is still alive (skip functionally dead patterns)
			if (this.getChildEffectiveActivationStrength(patternId, currentFrame) <= 0) continue;

			// get the context of the pattern
			const entry = this.routingTable.get(patternId);
			if (!entry) throw new Error(`Cannot find context for pattern: ${patternId}`);

			// context.match() handles the full scoring and threshold check;
			// the index only decides which child patterns are worth evaluating.
			// excludeIds masks out brand-new neurons so they don't count as "novel" misses.
			const match = entry.context.match(observed, age, this.mergeThreshold, excludeIds);
			if (!match) continue; // nothing to do if there is no match

			// if there is already a best match, check if this match is better
			if (best) {

				// if the previous best is better than this match, skip
				if (match.score < best.score) continue;

				// to preserve determinism when multiple patterns achieve the exact same score, explicitly tie-break using patternId
				if (match.score === best.score && patternId > best.patternId) continue; // prefer smaller id (older)
			}

			// store the best match
			match.patternId = patternId;
			match.age = age;
			best = match;
		}
		return best;
	}

	/**
	 * Find candidate child patterns for a specific active age using the inverted index.
	 * Candidate patterns must share at least one exact neuron/distance context entry with the
	 * observed context after converting absolute observed ages into pattern-relative distances.
	 *
	 * The observed context is keyed by absolute age within the current frame snapshot, while each
	 * stored pattern context is keyed by distance relative to the parent's own activation age.
	 * For a parent active at `age`, an observed entry at absolute distance `d` maps to a pattern
	 * entry at relative distance `d - age`.
	 *
	 * Missing index entries are expected here: they just mean the observed neuron/distance pair is
	 * not referenced by any child pattern and therefore contributes no candidates.
	 * @param {Context} observed
	 * @param {number} age
	 * @returns {Set<number>}
	 */
	getPatternCandidatesAtAge(observed, age) {
		const candidates = new Set();
		for (const [neuronId, distanceMap] of observed.entries) {

			// First narrow by exact context neuron ID. If this neuron does not appear in the index,
			// no child pattern references it anywhere in this parent's routing table.
			const indexedDistances = this.contextIndex.get(neuronId);
			if (!indexedDistances) continue;

			for (const absoluteDistance of distanceMap.keys()) {

				// Convert the observed absolute age into the pattern-relative distance used by the
				// routing table and inverted index. Distances < 1 are not valid pattern context entries
				// because context neurons must be older than the parent neuron itself.
				const patternDistance = absoluteDistance - age;
				if (patternDistance < 1) continue;

				// Then narrow by exact relative distance. Missing entries here are also expected and
				// simply mean no child pattern references this neuron at this particular distance.
				const patternIds = indexedDistances.get(patternDistance);
				if (!patternIds) continue;

				// A candidate only needs one exact neuron/distance overlap to be worth a full score.
				for (const patternId of patternIds) candidates.add(patternId);
			}
		}
		return candidates;
	}

	/**
	 * Refine the context of a pattern neuron based on the observed context.
	 * Strengthens common, adds novel, weakens/deletes missing.
	 * Returns list of context refs that should be removed (caller delivers to target neurons).
	 * @param {number} patternId - ID of the pattern to refine
	 * @param {Array<{neuronId, distance, strength}>} common - Entries present in both known and observed
	 * @param {Array<{neuronId, distance, strength}>} novel - Entries in observed but not known
	 * @param {Array<{neuronId, distance, strength}>} missing - Entries in known but not observed
	 * @returns {Array<{neuronId, distance}>} context refs that should be removed on target neurons
	 */
	refineContext(patternId, common, novel, missing) {

		// get the routing table entry for the pattern
		const entry = this.routingTable.get(patternId);
		if (!entry) throw new Error('pattern not found in routing table.'); // should not happen

		// strengthen common context neurons
		for (const item of common) entry.context.strengthenNeuron(item.neuronId, item.distance);

		// add novel context neurons
		for (const item of novel) this.addContext(patternId, item.neuronId, item.distance, 1);

		// weaken missing — weakenNeuron auto-deletes at zero strength
		const removedRefs = [];
		for (const item of missing) {
			const wasDeleted = entry.context.weakenNeuron(item.neuronId, item.distance);
			if (wasDeleted && this.removeContextIndex(item.neuronId, item.distance, patternId))
				removedRefs.push({ neuronId: item.neuronId, distance: item.distance });
		}
		return removedRefs;
	}

	/**
	 * Update connections based on the currently observed actives. For each active age > 0:
	 * upsert a connection per active (create-or-strengthen + alt-action), then weaken
	 * predictions at that distance that did not occur. Rewards are pre-resolved thalamus-side
	 * (0 for events, observed value for actions). Skipped entirely for new error patterns
	 * (they were just created this frame and have nothing yet to reinforce).
	 * @param {Map<number, object>} ageStates - This neuron's per-age state
	 * @param {boolean} isNewErrorPattern - True if this neuron was just created this frame
	 * @param {Array<{id: number, channel: string, reward: number}>} neurons - Currently observed neurons (age=0) with pre-resolved rewards
	 */
	learnConnections(ageStates, isNewErrorPattern, neurons) {
		if (isNewErrorPattern) return;
		for (const age of ageStates.keys()) {

			// skip age 0 - connection learning only applies to context neurons (age > 0)
			if (age === 0) continue;

			// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
			const neuronIds = new Set();
			for (const { id, channel, reward } of neurons) {
				this.upsertConnection(age, id, channel, reward);
				neuronIds.add(id);
			}

			// negatively reinforce connections at this distance whose predictions didn't occur
			for (const neuronId of this.getNeuronsNotFound(age, neuronIds))
				this.weakenConnection(age, neuronId);
		}
	}

	/**
	 * returns neuron IDs at a distance whose inferences did not occur
	 * @param {number} distance
	 * @param {Set<number>} activeNeuronIds - Set of active neuron IDs
	 */
	getNeuronsNotFound(distance, activeNeuronIds) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];
		const notFound = [];
		for (const [toNeuronId] of distanceMap) if (!activeNeuronIds.has(toNeuronId)) notFound.push(toNeuronId);
		return notFound;
	}

	/**
	 * Find an alternative action for a channel that hasn't been tried yet.
	 * @param {number} age - The age at which to check for existing connections
	 * @param {string} channel - The channel name
	 * @param {number} currentActionId - The action neuron ID to find an alternative to
	 * @returns {number|null} An alternative action neuron ID, or null if none available
	 */
	findAlternativeAction(age, channel, currentActionId) {
		for (const altNeuronId of this.channelActionIds.get(channel))
			if (altNeuronId !== currentActionId && !this.hasConnection(age, altNeuronId))
				return altNeuronId;
		return null;
	}

	/**
	 * Check if a child pattern can be deleted (is a zombie)
	 * @param {number} patternId - ID of the child pattern
	 * @param {number} currentFrame - Current frame number
	 */
	canDeleteChild(patternId, currentFrame) {
		// if the pattern has not been activated in some time, die!
		return this.getChildEffectiveActivationStrength(patternId, currentFrame) <= 0;
	}
}