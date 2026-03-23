import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuron, {strength, reward, lastFrame}>> - predictions with lazy decay
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

	// Hyperparameters
	static maxStrength = 100;
	static minStrength = 0;
	static rewardSmoothing = 0.9;
	static eventErrorMinStrength = 1;
	static actionRegretMinStrength = 4;
	static actionRegretMinPain = 0;
	static levelVoteMultiplier = 0;
	// use 0.001 or lower for text for all forget rates
	static connectionForgetRate = 0.009; // use 0.009 for stocks
	static contextForgetRate = 0.009; // use 0.009 for stocks
	static patternForgetRate = 0.013; // use 0.011 for stocks

	// static debug flag for the neuron
	static debug = false;

	/**
	 * Calculate continuous lazy decay amount based on frames elapsed
	 * @param {number} lastFrame - Frame when item was last updated
	 * @param {number} currentFrame - Current frame number
	 * @param {number} rate - Decay rate per frame
	 * @returns {number} Amount to subtract from strength
	 */
	static calculateDecay(lastFrame, currentFrame, rate) {
		return (currentFrame - lastFrame) * rate;
	}

	/**
	 * Get effective strength after lazy decay
	 */
	static getEffectiveStrength(strength, lastFrame, currentFrame, rate) {
		const decay = Neuron.calculateDecay(lastFrame, currentFrame, rate);
		return Math.max(Neuron.minStrength, strength - decay);
	}

	/**
	 * Create a sensory neuron (level 0) - id optional for loading from database
	 */
	static createSensory(channel, type, coordinates, id = null) {
		const neuron = new Neuron(0, id);
		neuron.channel = channel;
		neuron.type = type;
		neuron.coordinates = coordinates;
		return neuron;
	}

	/**
	 * Create a pattern neuron (level > 0) - id optional for loading from database
	 */
	static createPattern(level, parent, id = null) {
		const neuron = new Neuron(level, id);
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
	constructor(level = 0, id = null) {
		this.id = id !== null ? id : Neuron.nextId++;
		this.level = level;
		this.connections = new Map(); // inferences: Map<distance, Map<toNeuron, {strength, reward}>>
		this.children = new Set(); // child pattern neurons (routing table)
		this.context = new Context();  // the context that activated this pattern - not used by sensory neurons
		this.contextRefs = new Map(); // context references: Map<Neuron, Set<distance>>
		this.activationStrength = 0; // incremented with activation, forgotten over time
		this.lastActivationFrame = 0; // frame when activation was last strengthened (for lazy decay)

		// Update nextId if we're loading a neuron with a specific ID
		if (id !== null && id >= Neuron.nextId) Neuron.nextId = id + 1;
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
	 * Materializes lazy decay before incrementing strength using lastActivationFrame
	 */
	updateConnection(distance, toNeuron, reward, currentFrame) {
		if (!this.connections.has(distance)) throw new Error('Unknown connection'); // should not happen
		const conn = this.connections.get(distance).get(toNeuron);
		// Materialize lazy decay before strengthening using the neuron's true last activation frame
		const effectiveStrength = Neuron.getEffectiveStrength(conn.strength, this.lastActivationFrame, currentFrame, Neuron.connectionForgetRate);
		conn.strength = Math.min(Neuron.maxStrength, effectiveStrength + 1);
		conn.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * conn.reward;
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
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @returns {Array} Array of vote objects {toNeuron, strength, reward, distance}
	 */
	vote(age, currentFrame) {

		// use connections of distance one more than the age to get the inferences for the next frame
		const distance = age + 1;

		// level and age adjustments to the vote strength
		const levelWeight = 1 + this.level * Neuron.levelVoteMultiplier;

		// get connections at the distance - if there are none, no votes
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];

		// calculate neuron decay since last activation
		const decay = Neuron.calculateDecay(this.lastActivationFrame, currentFrame, Neuron.connectionForgetRate);

		// create votes for all connections at the distance and return them
		const result = [];
		for (const [neuron, conn] of distanceMap) {
			const effectiveStrength = Math.max(Neuron.minStrength, conn.strength - decay);
			if (effectiveStrength > 0)
				result.push({ neuron, strength: levelWeight * effectiveStrength, reward: conn.reward, distance });
		}
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
		return Neuron.getEffectiveStrength(this.activationStrength, this.lastActivationFrame, currentFrame, Neuron.patternForgetRate);
	}

	/**
	 * Materialize lazy decay for all owned connections.
	 */
	materializeConnections(currentFrame) {

		// Calculate decay based on when this neuron was last activated
		const decay = Neuron.calculateDecay(this.lastActivationFrame, currentFrame, Neuron.connectionForgetRate);
		if (decay <= 0) return;

		// update decayed strengths and collect dead connections
		const toDelete = [];
		for (const [distance, distanceMap] of this.connections)
			for (const [toNeuron, conn] of distanceMap) {
				const effectiveStrength = Math.max(Neuron.minStrength, conn.strength - decay);
				if (effectiveStrength <= 0) toDelete.push({ distance, toNeuron });
				else conn.strength = effectiveStrength;
			}

		// delete dead connections
		for (const { distance, toNeuron } of toDelete) this.deleteConnection(distance, toNeuron);
	}

	/**
	 * Materialize lazy decay for all owner-scoped values.
	 */
	materializeStrengths(currentFrame) {

		// decay own activation strength
		this.activationStrength = this.getEffectiveActivationStrength(currentFrame);

		// decay connection strengths and delete as needed
		this.materializeConnections(currentFrame);

		// decay context strengths and delete as needed
		const toDelete = this.context.materialize(this.lastActivationFrame, currentFrame, Neuron.contextForgetRate);
		for (const entry of toDelete) this.removePatternContext(entry.neuron, entry.distance);
	}

	/**
	 * increments activation strength - materializes all owner-scoped lazy decay first
	 */
	strengthenActivation(currentFrame) {

		// update all strengths based on decay rate first
		this.materializeStrengths(currentFrame);

		// increment activation strength
		this.activationStrength = Math.min(Neuron.maxStrength, this.activationStrength + 1);

		// remember when this happened for lazy decay
		this.lastActivationFrame = currentFrame;

		// return death frame for pattern neurons (sensory neurons never die)
		if (this.level === 0) return null;
		return currentFrame + Math.ceil(this.activationStrength / Neuron.patternForgetRate);
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
			const decay = Neuron.calculateDecay(pattern.lastActivationFrame, currentFrame, Neuron.contextForgetRate);
			const match = pattern.context.match(observed, decay);

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
	 * @param {number} currentFrame - Current frame number for lazy decay
	 */
	learnConnections(age, newActiveNeurons, rewards, channelActions, currentFrame) {

		// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		for (const neuron of newActiveNeurons) {

			// get the reward for the neuron if it is an action
			const reward = neuron.type === 'action' ? (rewards.get(neuron.channel) || 0) : 0;

			// if the event/action was already known, strengthen the connection and update the reward
			if (this.hasConnection(age, neuron)) this.updateConnection(age, neuron, reward, currentFrame);
			// if the event/action was not known, add it to the connections with the current reward (learning from observation)
			else this.createConnection(age, neuron, 1, reward);

			// if the neuron is an action and the reward is below a threshold, add an alternative action for the channel
			const conn = this.connections.get(age).get(neuron);
			if (conn.reward < Neuron.actionRegretMinPain) {
				const altNeuron = this.findAlternativeAction(age, neuron.channel, neuron, channelActions);
				if (altNeuron) this.createConnection(age, altNeuron, 1, 0);
			}
		}
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
	 * Check if there are any prediction errors or action regret that need correction.
	 * @returns {boolean} Whether any errors were found
	 */
	needsErrorCorrection(inferences, actualNeurons, rewards, channelActions) {

		// Categorize what actually happened into events and actions
		const { events: actualEvents, actions: executedActions } = this.categorizeActualNeurons(actualNeurons);

		// Identify channels with painful outcomes
		const painfulChannels = this.identifyPainfulChannels(rewards);

		// Pre-compute dimension signatures for fast lookup (performance optimization)
		const eventsByDimensions = this.groupEventsByDimensions(actualEvents);

		// Process each inference to find if any errors exist
		for (const inference of inferences) {

			// Check for event prediction errors
			if (this.hasEventErrors(inference, actualNeurons, eventsByDimensions)) return true;

			// Check for action regret
			if (this.hasActionRegret(inference, executedActions, painfulChannels, channelActions)) return true;
		}

		return false;
	}

	/**
	 * Categorize actual neurons into events and actions.
	 */
	categorizeActualNeurons(actualNeurons) {
		const events = new Set();
		const actions = new Set();
		for (const neuron of actualNeurons) {
			if (neuron.type === 'event') events.add(neuron);
			if (neuron.type === 'action') actions.add(neuron);
		}
		return { events, actions };
	}

	/**
	 * Identify channels that had painful outcomes.
	 */
	identifyPainfulChannels(rewards) {
		const painfulChannels = new Set();
		for (const [channelName, reward] of rewards)
			if (reward < Neuron.actionRegretMinPain) painfulChannels.add(channelName);
		return painfulChannels;
	}

	/**
	 * Group event neurons by their dimension signatures for fast lookup.
	 */
	groupEventsByDimensions(events) {
		const eventsByDimensions = new Map();
		for (const neuron of events) {
			const dimensionKey = Object.keys(neuron.coordinates).sort().join(',');
			if (!eventsByDimensions.has(dimensionKey)) eventsByDimensions.set(dimensionKey, []);
			eventsByDimensions.get(dimensionKey).push(neuron);
		}
		return eventsByDimensions;
	}

	/**
	 * Find corrections for event prediction errors.
	 * When a confident prediction didn't happen, find what actually happened with the same dimensions.
	 */
	hasEventErrors(prediction, actualNeurons, eventsByDimensions) {

		// Only process event predictions that were confident but didn't happen
		if (prediction.neuron.type !== 'event') return false;
		if (prediction.strength < Neuron.eventErrorMinStrength) return false;
		if (actualNeurons.has(prediction.neuron)) return false;

		// Check if there are actual events with the same dimensions as the failed prediction
		const failedDimensions = Object.keys(prediction.neuron.coordinates).sort().join(',');
		return eventsByDimensions.has(failedDimensions);
	}

	/**
	 * Check if a confident action was executed and resulted in pain.
	 */
	hasActionRegret(prediction, executedActions, painfulChannels, channelActions) {

		// Only process action predictions that were confident, executed, and painful
		if (prediction.neuron.type !== 'action') return false;
		if (prediction.strength < Neuron.actionRegretMinStrength) return false;
		if (!executedActions.has(prediction.neuron)) return false;
		if (!painfulChannels.has(prediction.neuron.channel)) return false;

		// Check if an alternative action exists for this channel
		const channelAlternatives = channelActions.get(prediction.neuron.channel);
		if (!channelAlternatives) return false;

		return [...channelAlternatives].some(n => n !== prediction.neuron);
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