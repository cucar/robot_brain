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
	 */
	constructor(patternForgetRate, mergeThreshold, id = null) {

		// initialize neuron parameters
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;

		// initialize neuron id if given - update nextId if we're loading a neuron with a specific ID
		this.id = id || Neuron.nextId++;
		if (id && id >= Neuron.nextId) Neuron.nextId = id + 1;

		// initialize activation strength with delayed calculations based on frames
		this.activationStrength = 0; // incremented with activation, forgotten over time
		this.lastActivationFrame = 0; // frame when activation was last strengthened (for lazy decay)

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
	 * sets the activation strength of the neuron - used when loading from database
	 */
	setActivationStrength(strength, lastFrame = 0) {
		this.activationStrength = strength;
		this.lastActivationFrame = lastFrame;
	}

	/**
	 * Get effective activation strength with lazy decay
	 */
	getEffectiveActivationStrength(currentFrame) {
		return Math.max(0, this.activationStrength - (currentFrame - this.lastActivationFrame) * this.patternForgetRate);
	}

	/**
	 * Materialize lazy decay
	 */
	materializeStrength(currentFrame) {
		this.activationStrength = this.getEffectiveActivationStrength(currentFrame);
	}

	/**
	 * increments activation strength - materializes all owner-scoped lazy decay first
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @param {number} level - Neuron level (passed in from Thalamus)
	 * @returns {number|null} Death frame for pattern neurons, null for sensory neurons
	 */
	strengthenActivation(currentFrame, level) {

		// update all strengths based on decay rate first
		this.materializeStrength(currentFrame);

		// increment activation strength
		this.activationStrength++;

		// remember when this happened for lazy decay
		this.lastActivationFrame = currentFrame;

		// return death frame for pattern neurons (sensory neurons never die)
		if (level === 0) return null;
		return currentFrame + Math.ceil(this.activationStrength / this.patternForgetRate);
	}

	/**
	 * Add a child pattern to the routing table and populate its context
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 * @param {Array<{neuron: Neuron, distance: number}>} context - The context to add
	 */
	addPattern(patternId, context) {
		this.addChild(patternId);
		for (const { neuronId, distance } of context) this.addContext(patternId, neuronId, distance);
	}

	/**
	 * add a child pattern to the routing table
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 */
	addChild(patternId) {
		if (!this.routingTable.has(patternId)) this.routingTable.set(patternId, new Context());
	}

	/**
	 * Remove a child pattern from this neuron's routing table.
	 * Called by thalamus when deleting a child pattern neuron.
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 */
	removeChild(patternId) {

		// clean up both context and context index for all context entries of this pattern
		const context = this.routingTable.get(patternId);
		if (!context) throw new Error(`removeChild: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		for (const entry of context.getEntries())
			this.removeContext(patternId, entry.neuronId, entry.distance);

		// remove the pattern from the routing table
		this.routingTable.delete(patternId);
	}

	/**
	 * returns pattern context entries
	 */
	getPatternContext(patternId) {
		const ctx = this.routingTable.get(patternId);
		if (!ctx) throw new Error(`getPatternContext: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		return ctx.getEntries();
	}

	/**
	 * returns all pattern context entries for all patterns in the routing table
	 */
	getRoutingTable() {
		const result = [];
		for (const [patternId, context] of this.routingTable)
			for (const entry of context.getEntries())
				result.push({ patternId, ...entry });
		return result;
	}

	/**
	 * adds an entry to the pattern context by neuron ID
	 */
	addContext(patternId, neuronId, distance, strength = 1) {

		// add the neuron to the pattern context at the given distance
		const context = this.routingTable.get(patternId);
		if (!context) throw new Error(`addContext: pattern not found in routing table: ${patternId}`);
		context.addNeuron(neuronId, distance, strength);

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
		const context = this.routingTable.get(patternId);
		if (!context) throw new Error(`removeContext: pattern ${patternId} not found in routing table of neuron ${this.id}`);
		context.remove(neuronId, distance);

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
				const ctx = this.routingTable.get(patternId);
				if (!ctx) throw new Error(`removeContextNeuron: pattern ${patternId} not found in routing table of neuron ${this.id}`);
				ctx.remove(neuronId, distance);
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
	 * Find matching patterns for this parent neuron given the observed context.
	 * The neuron is called once per frame with all active ages. For each age that matches,
	 * the matched pattern's context is refined immediately. Cross-neuron contextRef additions
	 * and removals are returned for delivery in a separate post-match phase.
	 * If the same child pattern matches at multiple ages for this parent, all refined matches
	 * are returned in order so side effects remain identical, but only the first match is
	 * flagged as an activation candidate.
	 * @param {Context} observed - The level context (distances are absolute ages)
	 * @param {Array<number>} activeAges - Ages at which this neuron is active (sorted ascending)
	 * @returns {{
	 *   matches: Array<{ patternId, age, score, common, missing, novel, removedRefs, activate }>,
	 *   contextRefUpdates: Array<{ type: 'add'|'remove', neuronId, distance }>
	 * }}
	 */
	matchPatterns(observed, activeAges) {
		const matches = [];
		const contextRefUpdates = [];
		const activatedPatternIds = new Set();
		for (const age of activeAges) {

			// active ages are processed in ascending order (most recent first). The first age that
			// produces a match at that age is refined and preserved. More recent ages tend to have
			// the richest available context, so they are processed first.
			const best = this.findBestPatternMatchAtAge(observed, age);
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

			// include the best match for the age in the results
			matches.push(best);
		}
		return { matches, contextRefUpdates };
	}

	/**
	 * Find the best matching pattern for a specific active age.
	 * @param {Context} observed - The level context (distances are absolute ages)
	 * @param {number} age - The specific active age being evaluated
	 * @returns {{ patternId, age, score, common, missing, novel }|null}
	 */
	findBestPatternMatchAtAge(observed, age) {
		let best = null; // { patternId, age, score, common, missing, novel }

		// Use the inverted index to narrow the search to child patterns that share at least one
		// exact neuron/distance entry with the observed context at this active age.
		const candidateIds = this.getPatternCandidatesAtAge(observed, age);
		if (candidateIds.size === 0) return null;

		// go through the candidate patterns and find the best match
		for (const patternId of candidateIds) {

			// get the context of the pattern
			const context = this.routingTable.get(patternId);
			if (!context) throw new Error(`Cannot find context for pattern: ${patternId}`);

			// context.match() handles the full scoring and threshold check;
			// the index only decides which child patterns are worth evaluating.
			const match = context.match(observed, age, this.mergeThreshold);
			if (!match || (best && match.score <= best.score)) continue;

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
		const context = this.routingTable.get(patternId);
		if (!context) throw new Error('pattern not found in routing table.'); // should not happen

		// strengthen common context neurons
		for (const entry of common) context.strengthenNeuron(entry.neuronId, entry.distance);

		// add novel context neurons
		for (const entry of novel) this.addContext(patternId, entry.neuronId, entry.distance, 1);

		// weaken missing — weakenNeuron auto-deletes at zero strength
		const removedRefs = [];
		for (const entry of missing) {
			const wasDeleted = context.weakenNeuron(entry.neuronId, entry.distance);
			if (wasDeleted && this.removeContextIndex(entry.neuronId, entry.distance, patternId))
				removedRefs.push({ neuronId: entry.neuronId, distance: entry.distance });
		}
		return removedRefs;
	}

	/**
	 * Update connections at a specific age based on observations.
	 * Events: strengthen correct, weaken incorrect, add novel.
	 * Actions: update with rewards, add alternatives for painful actions.
	 * @param {number} age - The age at which this neuron is active
	 * @param {Array<{id: number, type: string, channel: string}>} newActiveNeurons - Newly active neurons at age=0 with metadata
	 * @param {Set<number>} newActiveNeuronIds - Set of neuron IDs for quick lookup
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<number>>} channelActionIds - Map of channel name to all action neuron IDs
	 */
	learnConnections(age, newActiveNeurons, newActiveNeuronIds, rewards, channelActionIds) {

		// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		for (const neuron of newActiveNeurons) {

			// get the reward for the neuron if it is an action
			const reward = neuron.type === 'action' ? (rewards.get(neuron.channel) || 0) : 0;

			// if the event/action was already known, strengthen the connection and update the reward
			if (this.hasConnection(age, neuron.id)) this.strengthenConnection(age, neuron.id, reward);
			// if the event/action was not known, add it to the connections with the current reward (learning from observation)
			else this.createConnection(age, neuron.id, 1, reward);

			// if the neuron is an action and the reward is below a threshold, add an alternative action for the channel
			const conn = this.connections.get(age).get(neuron.id);
			if (conn.reward < 0) {
				const altNeuronId = this.findAlternativeAction(age, neuron.channel, neuron.id, channelActionIds);
				if (altNeuronId) this.createConnection(age, altNeuronId, 1, 0);
			}
		}

		// negatively reinforce connections at this distance whose predictions didn't occur
		for (const neuronId of this.getNeuronsNotFound(age, newActiveNeuronIds))
			this.weakenConnection(age, neuronId);
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
	 * @param {Map<string, Set<number>>} channelActionIds - Map of channel name to all action neuron IDs
	 * @returns {number|null} An alternative action neuron ID, or null if none available
	 */
	findAlternativeAction(age, channel, currentActionId, channelActionIds) {
		for (const altNeuronId of channelActionIds.get(channel))
			if (altNeuronId !== currentActionId && !this.hasConnection(age, altNeuronId))
				return altNeuronId;
		return null;
	}

	/**
	 * Check if neuron can be deleted (is a zombie)
	 * @param {number} currentFrame - Current frame number
	 * @param {number} level - Neuron level (passed in from Thalamus)
	 */
	canDelete(currentFrame, level) {

		// sensory neurons cannot be deleted
		if (level === 0) return false;

		// if the pattern has not been activated in some time, die!
		return this.getEffectiveActivationStrength(currentFrame) <= 0;
	}
}