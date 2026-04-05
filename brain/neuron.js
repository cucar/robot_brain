import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuron, {strength, reward}>> - predictions
 * - children: Set<Neuron> - child pattern neurons (routing table)
 *
 * Level 0 (sensory) neurons additionally have: channel, type, coordinates
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
	 * Create a sensory neuron (level 0) - id optional for loading from database
	 */
	static createSensory(channel, type, coordinates, patternForgetRate, mergeThreshold) {
		const neuron = new Neuron(0, patternForgetRate, mergeThreshold);
		neuron.channel = channel;
		neuron.type = type;
		neuron.coordinates = coordinates;
		return neuron;
	}

	/**
	 * Create a pattern neuron (level > 0) - id optional for loading from database
	 */
	static createPattern(level, parent, patternForgetRate, mergeThreshold) {
		const neuron = new Neuron(level, patternForgetRate, mergeThreshold);
		neuron.parent = parent;
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
		this.connections = new Map(); // inferences: Map<distance, Map<toNeuron, {strength, reward}>>
		this.children = new Set(); // child pattern neurons (routing table)
		this.contextRefs = new Map(); // context references: Map<Neuron, Set<distance>>
	}

	/**
	 * Get value key for this neuron (sensory only)
	 */
	get valueKey() {
		return Neuron.makeValueKey(this.coordinates);
	}

	/**
	 * returns if there is a connection at distance to a target neuron
	 */
	hasConnection(distance, toNeuron) {
		if (!this.connections.has(distance)) return false;
		const distanceMap = this.connections.get(distance);
		return distanceMap.has(toNeuron);
	}

	/**
	 * creates a connection at distance to target neuron
	 */
	createConnection(distance, toNeuron, strength, reward) {
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		this.connections.get(distance).set(toNeuron, { strength, reward });
	}

	/**
	 * updates the connection at distance to target neuron - increments strength and updates reward
	 */
	strengthenConnection(distance, toNeuron, reward) {
		if (!this.connections.has(distance)) throw new Error('Unknown connection'); // should not happen

		// increment the strength of the connection
		const conn = this.connections.get(distance).get(toNeuron);
		conn.strength++;

		// update reward with dynamic exponential smoothing - calculates exact expected value based on means
		const alpha = 1 / conn.strength;
		conn.reward = alpha * reward + (1 - alpha) * conn.reward;
	}

	/**
	 * Weaken a connection via negative reinforcement (prediction didn't occur).
	 * Deletes the connection if strength drops to zero or below.
	 */
	weakenConnection(distance, toNeuron) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return;
		const conn = distanceMap.get(toNeuron);
		if (!conn) return;
		conn.strength--;
		if (conn.strength <= 0) this.deleteConnection(distance, toNeuron);
	}

	/**
	 * Delete connection at distance to target neuron
	 */
	deleteConnection(distance, toNeuron) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap || !distanceMap.has(toNeuron)) return false;
		distanceMap.delete(toNeuron);
		if (distanceMap.size === 0) this.connections.delete(distance);
	}

	/**
	 * returns votes from this neuron at a specific age.
	 * @param {number} age - The age at which this neuron is active
	 * @returns {Array} Array of vote objects {toNeuron, strength, reward, distance}
	 */
	vote(age) {

		// use connections of distance one more than the age to get the inferences for the next frame
		const distance = age + 1;

		// get connections at the distance - if there are none, no votes
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];

		// create votes for all connections at the distance and return them
		const result = [];
		for (const [neuron, conn] of distanceMap)
			if (conn.strength > 0)
				result.push({ neuron, strength: conn.strength, reward: conn.reward, distance });
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
	 */
	addChild(pattern) {
		this.children.add(pattern);
	}

	/**
	 * Remove a child pattern from this neuron's routing table.
	 * Called by thalamus when deleting a child pattern neuron.
	 */
	removeChild(pattern) {
		this.children.delete(pattern);
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
		this.addContext(neuron, distance, strength);
		neuron.addContextRef(this, distance);
	}

	/**
	 * removes an entry from the pattern context
	 */
	removePatternContext(neuron, distance) {
		this.removeContext(neuron, distance);
		neuron.removeContextRef(this, distance);
	}

	/**
	 * adds an entry from the pattern context
	 */
	addContext(neuron, distance, strength = 1) {
		this.context.addNeuron(neuron, distance, strength);
	}

	/**
	 * removes an entry from the pattern context
	 */
	removeContext(neuron, distance) {
		this.context.remove(neuron, distance);
	}

	/**
	 * Add a context reference from another neuron to this neuron
	 * Called when this neuron is added to another neuron's context
	 */
	addContextRef(referencingNeuron, distance) {
		this.contextRefs.set(referencingNeuron, (this.contextRefs.get(referencingNeuron) ?? new Set()).add(distance));
	}

	/**
	 * Remove a context reference from another neuron to this neuron
	 * Called when this neuron is removed from another neuron's context
	 */
	removeContextRef(referencingNeuron, distance) {
		this.contextRefs.get(referencingNeuron).delete(distance);
		if (this.contextRefs.get(referencingNeuron).size === 0) this.contextRefs.delete(referencingNeuron);
	}

	/**
	 * Find the best matching pattern for this parent neuron given the observed context.
	 * @param {Context} observed - The observed context from brain
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @returns {Object|null} The matched pattern and match details, or null if no match
	 */
	matchPattern(observed, currentFrame) {

		// try to match the observed context to known patterns
		let best = null; // { pattern, score, common, missing, novel }
		for (const pattern of this.children) {

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
	 */
	refineContext(common, novel, missing) {

		// strengthen common context neurons
		for (const entry of common) this.context.strengthenNeuron(entry.neuron, entry.distance);

		// add novel context neurons
		for (const entry of novel) this.addPatternContext(entry.neuron, entry.distance, 1);

		// Weaken missing and delete if necessary
		for (const entry of missing) {
			const canDelete = this.context.weakenNeuron(entry.neuron, entry.distance);
			if (canDelete) this.removePatternContext(entry.neuron, entry.distance);
		}
	}

	/**
	 * Update connections at a specific age based on observations.
	 * Events: strengthen correct, weaken incorrect, add novel.
	 * Actions: update with rewards, add alternatives for painful actions.
	 * @param {number} age - The age at which this neuron is active
	 * @param {Set<Neuron>} newActiveNeurons - Newly active action neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 */
	learnConnections(age, newActiveNeurons, rewards, channelActions) {

		// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		for (const neuron of newActiveNeurons) {

			// get the reward for the neuron if it is an action
			const reward = neuron.type === 'action' ? (rewards.get(neuron.channel) || 0) : 0;

			// if the event/action was already known, strengthen the connection and update the reward
			if (this.hasConnection(age, neuron)) this.strengthenConnection(age, neuron, reward);
			// if the event/action was not known, add it to the connections with the current reward (learning from observation)
			else this.createConnection(age, neuron, 1, reward);

			// if the neuron is an action and the reward is below a threshold, add an alternative action for the channel
			const conn = this.connections.get(age).get(neuron);
			if (conn.reward < 0) {
				const altNeuron = this.findAlternativeAction(age, neuron.channel, neuron, channelActions);
				if (altNeuron) this.createConnection(age, altNeuron, 1, 0);
			}
		}

		// negatively reinforce connections at this distance whose predictions didn't occur
		for (const neuron of this.getNeuronsNotFound(age, newActiveNeurons))
			this.weakenConnection(age, neuron);
	}

	/**
	 * returns neurons at a distance whose inferences did not occur
	 */
	getNeuronsNotFound(distance, activeNeurons) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];
		const notFound = [];
		for (const [toNeuron] of distanceMap) if (!activeNeurons.has(toNeuron)) notFound.push(toNeuron);
		return notFound;
	}

	/**
	 * Find an alternative action for a channel that hasn't been tried yet.
	 * @param {number} age - The age at which to check for existing connections
	 * @param {string} channel - The channel name
	 * @param {Neuron} currentAction - The action to find an alternative to
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @returns {Neuron|null} An alternative action neuron, or null if none available
	 */
	findAlternativeAction(age, channel, currentAction, channelActions) {
		for (const altNeuron of channelActions.get(channel))
			if (altNeuron !== currentAction && !this.hasConnection(age, altNeuron))
				return altNeuron;
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