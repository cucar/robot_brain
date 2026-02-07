import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuron, {strength, reward}>> - predictions
 * - contexts: Array<{context: Context, pattern: Neuron}> - routing table
 *
 * Level 0 (sensory) neurons additionally have: channel, type, coordinates
 * Level > 0 (pattern) neurons additionally have: peak
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
	static rewardSmoothing = 0.9;
	static eventErrorMinStrength = 2;
	static actionRegretMinStrength = 2;
	static actionRegretMinPain = 0;
	static levelVoteMultiplier = 3;
	static connectionForgetRate = 1;
	static contextForgetRate = 1;

	// static debug flag for the neuron
	static debug = false;

	/**
	 * Create a sensory neuron (level 0)
	 */
	static createSensory(channel, type, coordinates) {
		const neuron = new Neuron(0);
		neuron.channel = channel;
		neuron.type = type;
		neuron.coordinates = coordinates;
		return neuron;
	}

	/**
	 * Create a pattern neuron (level > 0)
	 */
	static createPattern(level, peak) {
		const neuron = new Neuron(level);
		neuron.peak = peak;
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

	constructor(level = 0) {
		this.id = Neuron.nextId++;
		this.level = level;
		this.connections = new Map(); // inferences: Map<distance, Map<toNeuron, {strength, reward}>>
		this.contexts = []; // Routing table: Array<{context: Context, pattern: Neuron}>
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
	createConnection(distance, toNeuron, reward) {
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		this.connections.get(distance).set(toNeuron, { strength: 1, reward });
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
		const timeWeight = 1 - (distance - 1) * timeDecay;

		// get connections at the distance - if there are none, no votes
		const distanceMap = this.connections.get(distance);
		if (!distanceMap) return [];

		// create votes for all connections at the distance and return them
		const result = [];
		for (const [toNeuron, conn] of distanceMap)
			result.push({ toNeuron, strength: levelWeight * timeWeight * conn.strength, reward: conn.reward, distance });
		return result;
	}

	/**
	 * Get or create context for a pattern in the routing table.
	 * @param {Neuron} pattern - The pattern neuron
	 * @returns {Context} The context for this pattern
	 */
	getOrCreateContext(pattern) {
		let entry = this.contexts.find(e => e.pattern === pattern);
		if (!entry) {
			entry = { context: new Context(), pattern };
			this.contexts.push(entry);
		}
		return entry.context;
	}

	/**
	 * Find the best matching pattern for this peak neuron given the observed context.
	 * @param {Context} observed - The observed context from brain
	 * @returns {{peak: Neuron, pattern: Neuron}|null} The matched pattern with peak reference, or null if no match
	 */
	matchBestPattern(observed) {

		// try to match the observed context to known patterns
		let best = null; // { context, pattern, score, common, missing, novel }
		for (const { context, pattern } of this.contexts) {
			const match = context.match(observed, pattern);
			if (match && (!best || match.score > best.score)) best = { ...match, context };
		}
		if (!best) return null; // if there are no matches, return null

		// refine the context of the best matching pattern based on observed context
		best.context.refine(best.common, best.novel, best.missing);

		// return the matched pattern with peak reference (brain will set activated pattern)
		return { peak: this, pattern: best.pattern };
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

		// Only event neurons (level 0) or pattern neurons can learn connections
		if (this.level === 0 && this.type !== 'event') return;

		// learn events and actions - age=distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		for (const neuron of newActiveNeurons) {

			// get the reward for the neuron if it is an action
			const reward = neuron.type === 'action' ? rewards.get(neuron.channel) : undefined;

			// if the event/action was already known, strengthen the connection and update the reward
			if (this.hasConnection(age, neuron)) this.updateConnection(age, neuron, reward);
			// if the event/action was not known, add it to the connections with the current reward (learning from observation)
			else this.createConnection(age, neuron, reward);

			// if the neuron is an action and the reward is below a threshold, add an alternative action for the channel
			// add the first unknown action to try next time with reward 0
			if (reward !== undefined && reward < Neuron.actionRegretMinPain)
				for (const altNeuron of channelActions.get(neuron.channel))
					if (!this.hasConnection(age, altNeuron)) {
						this.createConnection(age, altNeuron, 0);
						break;
					}
		}
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

		// Build context for the new pattern (only same-level neurons)
		const patternContext = this.buildPatternContext(context);

		// Create pattern neuron at next level up
		const pattern = Neuron.createPattern(this.level + 1, this);
		for (const correctionNeuron of errorCorrections)
			pattern.createConnection(age, correctionNeuron, 0);

		// Add the context to the peak's routing table for the pattern
		this.contexts.push({ context: patternContext, pattern });

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
		if (prediction.toNeuron.type !== 'event') return;
		if (prediction.strength < Neuron.eventErrorMinStrength) return;
		if (actualNeurons.has(prediction.toNeuron)) return;

		// Find actual events with the same dimensions as the failed prediction
		const failedDimensions = Object.keys(prediction.toNeuron.coordinates).sort().join(',');
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
		if (prediction.toNeuron.type !== 'action') return;
		if (prediction.strength < Neuron.actionRegretMinStrength) return;
		if (!executedActions.has(prediction.toNeuron)) return;
		if (!painfulChannels.has(prediction.toNeuron.channel)) return;

		// Find an alternative action for this channel
		const channelAlternatives = channelActions.get(prediction.toNeuron.channel);
		if (!channelAlternatives) return;

		const alternativeAction = [...channelAlternatives].find(n => n !== prediction.toNeuron);
		if (alternativeAction) errorCorrections.add(alternativeAction);
	}

	/**
	 * Build pattern context from observed context, filtering to same-level neurons only.
	 */
	buildPatternContext(context) {
		const patternContext = new Context();
		for (const { neuron, distance } of context)
			if (neuron.level === this.level) patternContext.add(neuron, distance, 1);
		return patternContext;
	}

	/**
	 * forget cycle - decay connections and patterns - returns if it can be deleted after the forgetting
	 */
	forget() {
		this.forgetPatterns();
		this.forgetConnections();
		return this.canDelete();
	}

	/**
	 * Forget context entries (routing tables) - decay strengths and delete weak entries.
	 */
	forgetPatterns() {
		let contextsDeleted = 0, contextsUpdated = 0;
		for (const { context } of this.contexts) {
			const toDelete = [];
			for (const entry of context.entries) {
				const oldStrength = entry.strength;
				entry.strength = Math.max(Context.minStrength, entry.strength - Neuron.contextForgetRate);
				if (entry.strength < oldStrength) contextsUpdated++;
				if (entry.strength <= Context.minStrength) toDelete.push(entry);
			}
			for (const entry of toDelete) {
				context.remove(entry.neuron, entry.distance);
				contextsDeleted++;
			}
		}
		if (Neuron.debug) console.log(`  Contexts: ${contextsUpdated} weakened, ${contextsDeleted} deleted`);
	}

	/**
	 * Forget connections - decay strengths and delete weak connections.
	 */
	forgetConnections() {
		let connectionsUpdated = 0, connectionsDeleted = 0;
		for (const [distance, distanceMap] of this.connections) {
			const toDelete = [];
			for (const [toNeuron, conn] of distanceMap) {
				const oldStrength = conn.strength;
				conn.strength = Math.max(Neuron.minStrength, conn.strength - Neuron.connectionForgetRate);
				if (conn.strength < oldStrength) connectionsUpdated++;
				if (conn.strength <= Neuron.minStrength) toDelete.push(toNeuron);
			}
			for (const toNeuron of toDelete) {
				this.deleteConnection(distance, toNeuron);
				connectionsDeleted++;
			}
		}
		if (Neuron.debug) console.log(`  Connections: ${connectionsUpdated} weakened, ${connectionsDeleted} deleted`);
	}

	/**
	 * Check if neuron can be deleted
	 */
	canDelete() {
		return this.level > 0 && // sensory neurons cannot be deleted
			this.connections.size === 0 && // no outgoing connections (no pattern future)
			this.contexts.length === 0; // no contexts (no known pattern)
	}

	/**
	 * Remove a pattern from this neuron's contexts (routing table).
	 * Called by thalamus when deleting a child pattern neuron.
	 * @param {Neuron} pattern - The pattern neuron to remove
	 */
	removePattern(pattern) {
		const idx = this.contexts.findIndex(e => e.pattern === pattern);
		if (idx !== -1) this.contexts.splice(idx, 1);
	}
}