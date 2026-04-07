import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuronId, {strength, reward}>> - predictions
 * - children: Set<neuronId> - child pattern neuron IDs (routing table)
 *
 * Level 0 (sensory) neurons additionally have: type, coordinates
 * Channel is stored externally in Thalamus lookup tables (neuronId → channelName).
 * Level > 0 (pattern) neurons additionally have: parent
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
	 * Create a sensory neuron (level 0)
	 * Channel is stored externally in Thalamus lookup tables.
	 */
	static createSensory(type, coordinates, patternForgetRate, mergeThreshold) {
		const neuron = new Neuron(0, patternForgetRate, mergeThreshold);
		neuron.type = type;
		neuron.coordinates = coordinates;
		return neuron;
	}

	/**
	 * Create a pattern neuron (level > 0) - id optional for loading from database
	 */
	static createPattern(level, parentId, patternForgetRate, mergeThreshold) {
		const neuron = new Neuron(level, patternForgetRate, mergeThreshold);
		neuron.parentId = parentId;
		return neuron;
	}

	/**
	 * Create value key for neuron lookup
	 */
	static makeValueKey(coordinates) {
		const sorted = Object.keys(coordinates).sort();
		const obj = {};
		for (const k of sorted) obj[k] = coordinates[k];
		return JSON.stringify(obj);
	}

	/**
	 * constructor - id optional for loading from database
	 */
	constructor(level, patternForgetRate, mergeThreshold, id = null) {

		// initialize neuron parameters
		this.level = level;
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;

		// the context that activated this pattern - not used by sensory neurons
		// TODO: this will be moved to the parent routing tables instead of the child
		this.context = new Context();

		// initialize neuron id if given - update nextId if we're loading a neuron with a specific ID
		this.id = id || Neuron.nextId++;
		if (id && id >= Neuron.nextId) Neuron.nextId = id + 1;

		// initialize activation strength with delayed calculations based on frames
		this.activationStrength = 0; // incremented with activation, forgotten over time
		this.lastActivationFrame = 0; // frame when activation was last strengthened (for lazy decay)

		// initialize synapses
		this.connections = new Map(); // inferences: Map<distance, Map<toNeuronId, {strength, reward}>>
		this.children = new Set(); // child pattern neuron IDs (routing table)
		this.contextRefs = new Map(); // context references: Map<neuronId, Set<distance>>
	}

	/**
	 * Get value key for this neuron (sensory only)
	 */
	get valueKey() {
		return Neuron.makeValueKey(this.coordinates);
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
	 */
	strengthenActivation(currentFrame) {

		// update all strengths based on decay rate first
		this.materializeStrength(currentFrame);

		// increment activation strength
		this.activationStrength++;

		// remember when this happened for lazy decay
		this.lastActivationFrame = currentFrame;

		// return death frame for pattern neurons (sensory neurons never die)
		if (this.level === 0) return null;
		return currentFrame + Math.ceil(this.activationStrength / this.patternForgetRate);
	}

	/**
	 * add a child pattern to the routing table without context (used for load - it will be added later)
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 */
	addChild(patternId) {
		this.children.add(patternId);
	}

	/**
	 * Remove a child pattern from this neuron's routing table.
	 * Called by thalamus when deleting a child pattern neuron.
	 * @param {number} patternId - numeric neuron ID of the child pattern
	 */
	removeChild(patternId) {
		this.children.delete(patternId);
	}

	/**
	 * returns pattern context entries
	 */
	getPatternContext() {
		return this.context.getEntries();
	}

	/**
	 * adds a new entry to a pattern context
	 */
	addPatternContext(neuron, distance, strength = 1) {
		this.addContext(neuron.id, distance, strength);
		neuron.addContextRef(this.id, distance);
	}

	/**
	 * removes an entry from the pattern context
	 */
	removePatternContext(neuron, distance) {
		this.removeContext(neuron.id, distance);
		neuron.removeContextRef(this.id, distance);
	}

	/**
	 * adds an entry to the pattern context by neuron ID
	 */
	addContext(neuronId, distance, strength = 1) {
		this.context.addNeuron(neuronId, distance, strength);
	}

	/**
	 * removes an entry from the pattern context by neuron ID
	 */
	removeContext(neuronId, distance) {
		this.context.remove(neuronId, distance);
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
		this.contextRefs.get(referencingNeuronId).delete(distance);
		if (this.contextRefs.get(referencingNeuronId).size === 0) this.contextRefs.delete(referencingNeuronId);
	}

	/**
	 * Find the best matching pattern for this parent neuron given the observed context.
	 * @param {Context} observed - The observed context from brain
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @param {Map<number, Neuron>} neurons - Neuron lookup map (id → Neuron)
	 * @returns {Object|null} The matched pattern and match details, or null if no match
	 */
	matchPattern(observed, currentFrame, neurons) {

		// try to match the observed context to known patterns
		let best = null; // { pattern, score, common, missing, novel }
		for (const patternId of this.children) {
			const pattern = neurons.get(patternId);
			if (!pattern) continue;

			// if the pattern has been forgotten, ignore that - cleanup cycle will take care of it
			if (pattern.getEffectiveActivationStrength(currentFrame) === 0) continue;

			// get the match results for the pattern for the given context
			const match = pattern.context.match(observed, this.mergeThreshold);

			// if there is a match, and it's the best so far, store it
			if (match && (!best || match.score > best.score)) {
				match.pattern = pattern;
				best = match;
			}
		}
		if (!best) return null; // if there are no matches, return null

		// return the matched pattern and details so brain can activate first, then refine
		return best;
	}

	/**
	 * Refine the context of a pattern neuron based on the observed context.
	 * Strengthens common context neurons, adds novel ones, and weakens/ deletes missing ones.
	 * @param {Array<{neuronId, distance, strength}>} common - Entries present in both known and observed
	 * @param {Array<{neuronId, distance, strength}>} novel - Entries in observed but not known
	 * @param {Array<{neuronId, distance, strength}>} missing - Entries in known but not observed
	 * @param {Map<number, Neuron>} neurons - Neuron lookup map for resolving IDs to objects
	 */
	refineContext(common, novel, missing, neurons) {

		// strengthen common context neurons (only need ID)
		for (const entry of common) this.context.strengthenNeuron(entry.neuronId, entry.distance);

		// add novel context neurons (need Neuron object for contextRef)
		for (const entry of novel) {
			const neuron = neurons.get(entry.neuronId);
			if (neuron) this.addPatternContext(neuron, entry.distance, 1);
		}

		// Weaken missing and delete if necessary (need Neuron object for contextRef removal)
		for (const entry of missing) {
			const canDelete = this.context.weakenNeuron(entry.neuronId, entry.distance);
			if (canDelete) {
				const neuron = neurons.get(entry.neuronId);
				if (neuron) this.removePatternContext(neuron, entry.distance);
				else this.removeContext(entry.neuronId, entry.distance); // neuron already deleted, just remove context entry
			}
		}
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
	 */
	canDelete(currentFrame) {

		// sensory neurons cannot be deleted
		if (this.level === 0) return false;

		// if the pattern has not been activated in some time, die!
		return this.getEffectiveActivationStrength(currentFrame) <= 0;
	}
}