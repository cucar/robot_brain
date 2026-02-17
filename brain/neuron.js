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
 */
export class Neuron {

	// Static counter for assigning unique IDs to neurons
	static nextId = 1;

	// Hyperparameters
	static maxStrength = 100;
	static minStrength = 0;
	static rewardSmoothing = 1;
	static eventErrorMinStrength = 2;
	static actionRegretMinStrength = 2;
	static actionRegretMinPain = 0;
	static levelVoteMultiplier = 3;
	static connectionForgetRate = 1;
	static contextForgetRate = 1;
	static patternForgetRate = 1;

	// static debug flag for the neuron
	static debug = false;

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
	 */
	updateConnection(distance, toNeuron, reward) {
		if (!this.connections.has(distance)) throw new Error('Unknown connection'); // should not happen
		const connection = this.connections.get(distance).get(toNeuron);
		connection.strength = Math.min(Neuron.maxStrength, connection.strength + 1);
		if (reward !== undefined) connection.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * connection.reward;
	}

	/**
	 * Delete connection at distance to target neuron
	 */
	deleteConnection(distance, toNeuron) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap || !distanceMap.has(toNeuron)) return false;
		distanceMap.delete(toNeuron);
		if (distanceMap.size === 0) this.connections.delete(distance);
		return true;
	}

	/**
	 * returns votes from this neuron at a specific age.
	 * @param {number} age - The age at which this neuron is active
	 * @param {number} timeDecay - Context window length for time decay
	 * @returns {Array} Array of vote objects {toNeuron, strength, reward, distance}
	 */
	vote(age, timeDecay) {

		// use connections of distance one more than the age to get the inferences for the next frame
		const distance = age + 1;

		// level and age adjustments to the vote strength
		const levelWeight = 1 + this.level * Neuron.levelVoteMultiplier;
		const timeWeight = 1 - age * timeDecay;

		// get connections at the distance - if there are none, no votes
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];

		// create votes for all connections at the distance and return them
		const result = [];
		for (const [neuron, conn] of distanceMap)
			result.push({ neuron, strength: levelWeight * timeWeight * conn.strength, reward: conn.reward, distance });
		return result;
	}

	/**
	 * sets the activation strength of the neuron - used when loading from database
	 */
	setActivationStrength(strength) {
		this.activationStrength = strength;
	}

	/**
	 * increments activation strength
	 */
	strengthenActivation() {
		this.activationStrength = Math.min(Neuron.maxStrength, this.activationStrength + 1);
	}

	/**
	 * reduces activation strength
	 */
	weakenActivation() {
		this.activationStrength = Math.max(Neuron.minStrength, this.activationStrength - Neuron.patternForgetRate);
	}

	/**
	 * add a child pattern to the routing table without context (used for load - it will be added later)
	 */
	addChild(pattern) {
		this.children.add(pattern);
	}

	/**
	 * returns pattern context entries
	 */
	getPatternContext() {
		return this.context.entries;
	}

	/**
	 * adds a new entry to a pattern context
	 */
	addPatternContext(neuron, distance, strength) {
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
	addContext(neuron, distance, strength) {
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
	 * Remove a child pattern from this neuron's routing table.
	 * Called by thalamus when deleting a child pattern neuron.
	 */
	removeChild(pattern) {
		this.children.delete(pattern);
	}

	/**
	 * Find the best matching pattern for this parent neuron given the observed context.
	 * @param {Context} observed - The observed context from brain
	 * @returns Neuron The matched pattern, or null if no match
	 */
	matchPattern(observed) {

		// try to match the observed context to known patterns
		let best = null; // { pattern, score, common, missing, novel }
		for (const pattern of this.children) {
			if (pattern.activationStrength === 0) continue;
			const match = pattern.context.match(observed);
			if (match && (!best || match.score > best.score)) best = { ...match, pattern };
		}
		if (!best) return null; // if there are no matches, return null

		// call the pattern to refine its context based on the observed context
		best.pattern.refineContext(best.common, best.novel, best.missing);

		// return the matched pattern with parent reference (brain will set activated pattern)
		return best.pattern;
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

		// action neurons cannot learn connections - they do not infer
		if (this.level === 0 && this.type === 'action') return;

		// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		for (const neuron of newActiveNeurons) {

			// get the reward for the neuron if it is an action
			const reward = neuron.type === 'action' ? rewards.get(neuron.channel) : undefined;

			// if the event/action was already known, strengthen the connection and update the reward
			if (this.hasConnection(age, neuron)) this.updateConnection(age, neuron, reward);
			// if the event/action was not known, add it to the connections with the current reward (learning from observation)
			else this.createConnection(age, neuron, 1, reward);

			// if the neuron is an action and the reward is below a threshold, add an alternative action for the channel
			const neuronReward = this.connections.get(age).get(neuron);
			if (neuronReward !== undefined && neuronReward < Neuron.actionRegretMinPain) {
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
	 * Learn new pattern from prediction errors and action regret at a specific age.
	 * Only called for ages where no pattern was activated.
	 * @param {number} age - The age at which this neuron made a bad inference
	 * @param {Array<{neuron: Neuron, distance: number}>} context - Active context neurons with distances
	 * @param {Array<{toNeuron, strength, reward}>} inferences - Inferences made by this neuron at this age
	 * @param {Set<Neuron>} actualNeurons - What actually happened at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @returns {Neuron|null} Newly created pattern, or null if no pattern needed
	 */
	learnNewPattern(age, context, inferences, actualNeurons, rewards, channelActions) {

		// Find corrections for prediction errors and action regret
		const errorCorrections = this.findErrorCorrections(inferences, actualNeurons, rewards, channelActions);

		// No errors to learn from
		if (errorCorrections.size === 0) return null;

		// Create pattern neuron at next level up
		const pattern = Neuron.createPattern(this.level + 1, this);
		for (const correctionNeuron of errorCorrections)
			pattern.createConnection(age, correctionNeuron, 1, 0);

		// add the new pattern to the routing table
		this.addChild(pattern);

		// Add the context to the new pattern (only same-level neurons)
		for (const { neuron, distance } of context)
			if (neuron.level === this.level)
				pattern.addPatternContext(neuron, distance, 1);

		// Return pattern - brain will handle activation
		return pattern;
	}

	/**
	 * Find corrections for prediction errors and action regret.
	 */
	findErrorCorrections(inferences, actualNeurons, rewards, channelActions) {
		const errorCorrections = new Set();

		// Categorize what actually happened into events and actions
		const { events: actualEvents, actions: executedActions } = this.categorizeActualNeurons(actualNeurons);

		// Identify channels with painful outcomes
		const painfulChannels = this.identifyPainfulChannels(rewards);

		// Pre-compute dimension signatures for fast lookup (performance optimization)
		const eventsByDimensions = this.groupEventsByDimensions(actualEvents);

		// Process each inference to find error corrections
		for (const inference of inferences) {

			// Handle event prediction errors
			this.findEventErrorCorrections(inference, actualNeurons, eventsByDimensions, errorCorrections);

			// Handle action regret
			this.findActionRegretCorrections(inference, executedActions, painfulChannels, channelActions, errorCorrections);
		}

		return errorCorrections;
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
			if (!eventsByDimensions.has(dimensionKey))
				eventsByDimensions.set(dimensionKey, []);
			eventsByDimensions.get(dimensionKey).push(neuron);
		}
		return eventsByDimensions;
	}

	/**
	 * Find corrections for event prediction errors.
	 * When a confident prediction didn't happen, find what actually happened with the same dimensions.
	 */
	findEventErrorCorrections(prediction, actualNeurons, eventsByDimensions, errorCorrections) {

		// Only process event predictions that were confident but didn't happen
		if (prediction.neuron.type !== 'event') return;
		if (prediction.strength < Neuron.eventErrorMinStrength) return;
		if (actualNeurons.has(prediction.neuron)) return;

		// Find actual events with the same dimensions as the failed prediction
		const failedDimensions = Object.keys(prediction.neuron.coordinates).sort().join(',');
		const matchingEvents = eventsByDimensions.get(failedDimensions);
		if (matchingEvents)
			for (const actualEvent of matchingEvents)
				errorCorrections.add(actualEvent);
	}

	/**
	 * When a confident action was executed and resulted in pain, suggest an alternative.
	 */
	findActionRegretCorrections(prediction, executedActions, painfulChannels, channelActions, errorCorrections) {

		// Only process action predictions that were confident, executed, and painful
		if (prediction.neuron.type !== 'action') return;
		if (prediction.strength < Neuron.actionRegretMinStrength) return;
		if (!executedActions.has(prediction.neuron)) return;
		if (!painfulChannels.has(prediction.neuron.channel)) return;

		// Find an alternative action for this channel
		const channelAlternatives = channelActions.get(prediction.neuron.channel);
		if (!channelAlternatives) return;

		const alternativeAction = [...channelAlternatives].find(n => n !== prediction.neuron);
		if (alternativeAction) errorCorrections.add(alternativeAction);
	}

	/**
	 * forget cycle - decay activation, connections and patterns - returns if it can be deleted after the forgetting
	 */
	forget() {

		// weaken activation strength
		this.weakenActivation();

		// forget the contexts which activate this pattern
		this.forgetContexts();

		// forget connections
		this.forgetConnections();

		// return if the neuron can be deleted or not
		return this.canDelete();
	}

	/**
	 * Forget context entries (routing tables) - decay strengths and delete weak entries.
	 */
	forgetContexts() {
		let contextsDeleted = 0, contextsUpdated = 0;

		// decay the context strengths
		const toDelete = [];
		for (const entry of this.context.entries) {
			const oldStrength = entry.strength;
			entry.strength = Math.max(Context.minStrength, entry.strength - Neuron.contextForgetRate);
			if (entry.strength < oldStrength) contextsUpdated++;
			if (entry.strength <= Context.minStrength) toDelete.push(entry);
		}

		// delete the weak context entries
		for (const entry of toDelete) {
			this.removePatternContext(entry.neuron, entry.distance);
			contextsDeleted++;
		}

		if (Neuron.debug) console.log(`  Contexts: ${contextsUpdated} weakened, ${contextsDeleted} deleted`);
	}

	/**
	 * Forget connections - decay strengths and delete weak connections.
	 */
	forgetConnections() {
		let connectionsUpdated = 0, connectionsDeleted = 0;
		const toDelete = [];

		// decay the connection strengths
		for (const [distance, distanceMap] of this.connections) {
			for (const [toNeuron, conn] of distanceMap) {
				const oldStrength = conn.strength;
				conn.strength = Math.max(Neuron.minStrength, conn.strength - Neuron.connectionForgetRate);
				if (conn.strength < oldStrength) connectionsUpdated++;
				if (conn.strength <= Neuron.minStrength) toDelete.push({ toNeuron, distance });
			}
		}

		// delete the weak connections
		for (const { toNeuron, distance } of toDelete) {
			this.deleteConnection(distance, toNeuron);
			connectionsDeleted++;
		}

		if (Neuron.debug) console.log(`  Connections: ${connectionsUpdated} weakened, ${connectionsDeleted} deleted`);
	}

	/**
	 * Check if neuron can be deleted
	 */
	canDelete() {

		// sensory neurons cannot be deleted
		if (this.level === 0) return false;

		// if the pattern has not been activated in some time, die!
		if (this.activationStrength === 0) return true;

		// if a pattern does not have any contexts (cannot be recognized), it cannot be activated - it needs to be deleted
		if (this.context.size === 0) return true;

		// if as a result of the forget or cleanup operation, we don't have any connections or children
		// we don't serve a purpose - it's detrimental - need to be deleted
		return this.connections.size === 0 && this.children.size === 0;
	}
}