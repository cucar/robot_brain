import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuron, {strength, reward}>> - predictions
 * - contexts: Array<{context: Context, pattern: Neuron}> - routing table
 *
 * Level 0 (sensory) neurons additionally have: channel, type, coordinates
 * Level > 0 (pattern) neurons additionally have: peak, peakStrength
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
	static negativeReinforcement = 0.1;
	static eventErrorMinStrength = 2;
	static actionRegretMinStrength = 2;
	static actionRegretMinPain = 0;
	static levelVoteMultiplier = 3;

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
		neuron.peakStrength = 1.0;
		peak.incomingCount++;
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
		this.incomingCount = 0;
		// Predictions: Map<distance, Map<toNeuron, {strength, reward}>>
		this.connections = new Map();
		// Routing table: Array<{context: Context, pattern: Neuron}>
		this.contexts = [];
	}

	get isSensory() { return this.level === 0; }
	get isPattern() { return this.level > 0; }

	/** Get value key for this neuron (sensory only) */
	get valueKey() {
		return Neuron.makeValueKey(this.coordinates);
	}

	/**
	 * Called when this pattern neuron is activated via match.
	 * Increases peak strength.
	 */
	strengthenPeak() {
		if (this.level > 0) this.peakStrength = Math.min(Neuron.maxStrength, this.peakStrength + 1);
	}

	/**
	 * Get or create connection at distance to target neuron
	 */
	getOrCreateConnection(distance, toNeuron) {
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		const distanceMap = this.connections.get(distance);
		if (!distanceMap.has(toNeuron)) {
			distanceMap.set(toNeuron, { strength: 1, reward: 0 });
			toNeuron.incomingCount++;
		}
		return distanceMap.get(toNeuron);
	}

	/**
	 * Delete connection at distance to target neuron
	 */
	deleteConnection(distance, toNeuron) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap || !distanceMap.has(toNeuron)) return false;
		distanceMap.delete(toNeuron);
		toNeuron.incomingCount--;
		if (distanceMap.size === 0) this.connections.delete(distance);
		return true;
	}

	/**
	 * Collect votes from this neuron at a specific age.
	 * @param {number} age - The age at which this neuron is active
	 * @param {number} contextLength - Context window length for time decay
	 * @returns {Array} Array of vote objects {toNeuron, strength, reward, distance}
	 */
	collectVotesAtAge(age, contextLength) {

		// use connections of distance one more than the age to get the inferences for the next frame
		const distance = age + 1;

		// level and age adjustments to the vote strength
		const timeDecay = 1 / contextLength;
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
	 * @param {Set<Neuron>} newEventNeurons - Newly active event neurons at age=0
	 * @param {Set<Neuron>} newActionNeurons - Newly active action neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 */
	learnConnectionsAtAge(age, newEventNeurons, newActionNeurons, rewards, channelActions) {

		// Only event neurons (level 0) or pattern neurons can learn connections
		if (this.level === 0 && this.type !== 'event') return;

		// get existing connections at the distance (if neuron is active at age=4, we are learning 4 steps into the future at age=0)
		const distance = age;
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		const distanceMap = this.connections.get(distance);

		// refine existing event predictions and add novel event predictions
		this.learnEvents(distanceMap, newEventNeurons, distance);

		// learn action inferences and add alternative actions for painful channels
		this.learnActions(distanceMap, newActionNeurons, rewards, channelActions);
	}

	/**
	 * Refine existing event predictions and add novel event predictions.
	 * @param {Map<Neuron, {strength, reward}>} distanceMap - Connections at this distance
	 * @param {Set<Neuron>} newEventNeurons - Newly active event neurons at age=0
	 * @param {number} distance - Temporal distance for these connections
	 */
	learnEvents(distanceMap, newEventNeurons, distance) {

		// refine existing event predictions (collect deletions to avoid modifying map while iterating)
		const toDelete = [];
		for (const [inferredNeuron, prediction] of distanceMap) {

			// refining event predictions
			if (inferredNeuron.level !== 0 || inferredNeuron.type !== 'event') continue;

			// if the predicted neuron did not come true, weaken the connection, and delete if it falls below min strength
			if (!newEventNeurons.has(inferredNeuron)) {
				prediction.strength -= Neuron.negativeReinforcement;
				if (prediction.strength <= Neuron.minStrength) toDelete.push(inferredNeuron);
				continue;
			}

			// if the predicted neuron is active at age=0, strengthen the connection
			prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
		}
		for (const inferredNeuron of toDelete) this.deleteConnection(distance, inferredNeuron);

		// Add novel event predictions
		for (const eventNeuron of newEventNeurons)
			if (!distanceMap.has(eventNeuron)) {
				distanceMap.set(eventNeuron, { strength: 1, reward: 0 });
				eventNeuron.incomingCount++;
			}
	}

	/**
	 * Learn action inferences and add alternative actions for painful channels.
	 * @param {Map<Neuron, {strength, reward}>} distanceMap - Connections at this distance
	 * @param {Set<Neuron>} newActionNeurons - Newly active action neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 */
	learnActions(distanceMap, newActionNeurons, rewards, channelActions) {

		// learn action inferences
		for (const actionNeuron of newActionNeurons) {

			// get the current rewards for the action - if there are no rewards for it, it's not worth learning
			const reward = rewards.get(actionNeuron.channel);
			if (reward === undefined) continue;

			// if the action was already known, strengthen the connection and update the reward
			if (distanceMap.has(actionNeuron)) {
				const prediction = distanceMap.get(actionNeuron);
				prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
				prediction.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * prediction.reward;
				continue;
			}

			// if the action was not known, add it to the connections with the current reward (learning from observation)
			distanceMap.set(actionNeuron, { strength: 1, reward });
			actionNeuron.incomingCount++;
		}

		// get painful channels
		const painfulChannels = new Set();
		for (const actionNeuron of newActionNeurons) {
			const reward = rewards.get(actionNeuron.channel);
			if (reward !== undefined && reward < Neuron.actionRegretMinPain) painfulChannels.add(actionNeuron.channel);
		}

		// Add alternative actions for painful channels
		for (const channel of painfulChannels) {

			// get all possible actions for the channel
			const allActions = channelActions.get(channel);
			if (!allActions) continue;

			// add the first unknown action to try next time with reward 0
			for (const altNeuron of allActions)
				if (!distanceMap.has(altNeuron)) {
					distanceMap.set(altNeuron, { strength: 1, reward: 0 });
					altNeuron.incomingCount++;
					break;
				}
		}
	}

	/**
	 * Learn new pattern from prediction errors and action regret at a specific age.
	 * Only called for ages where no pattern was activated.
	 * @param {number} age - The age at which this neuron made a bad inference
	 * @param {Array<{toNeuron, strength, reward}>} votes - Votes made by this neuron at this age
	 * @param {Set<Neuron>} newlyActiveNeurons - Currently active neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @param {Array<{neuron: Neuron, distance: number}>} context - Active context neurons with distances
	 * @returns {Neuron|null} Newly created pattern, or null if no pattern needed
	 */
	learnNewPattern(age, votes, newlyActiveNeurons, rewards, channelActions, context) {

		// Get active event neurons at age=0 (what actually happened)
		const activeEventNeurons = new Set();
		for (const neuron of newlyActiveNeurons)
			if (neuron.level === 0 && neuron.type === 'event')
				activeEventNeurons.add(neuron);

		// Get painful channels
		const painfulChannels = new Set();
		for (const [channelName, reward] of rewards)
			if (reward < Neuron.actionRegretMinPain)
				painfulChannels.add(channelName);

		// Get winning action neurons
		const winnerActions = new Set();
		for (const neuron of newlyActiveNeurons)
			if (neuron.level === 0 && neuron.type === 'action')
				winnerActions.add(neuron);

		// Collect errors at this distance (use Set to avoid duplicates)
		const patternConnections = new Set();
		for (const vote of votes) {

			// Event error: predicted event didn't happen (target not in newlyActiveNeurons)
			// Find actual events with exactly the same dimensions as the failed prediction
			if (vote.toNeuron.type === 'event' && vote.strength >= Neuron.eventErrorMinStrength && !newlyActiveNeurons.has(vote.toNeuron)) {
				const failedDims = Object.keys(vote.toNeuron.coordinates).sort().join(',');
				for (const actualNeuron of activeEventNeurons) {
					const actualDims = Object.keys(actualNeuron.coordinates).sort().join(',');
					if (failedDims === actualDims) patternConnections.add(actualNeuron);
				}
			}

			// Action regret: predicted action was executed and was painful
			if (vote.toNeuron.type === 'action' && vote.strength >= Neuron.actionRegretMinStrength)
				if (winnerActions.has(vote.toNeuron) && painfulChannels.has(vote.toNeuron.channel)) {
					const alts = channelActions.get(vote.toNeuron.channel);
					if (alts) {
						const altNeuron = [...alts].find(n => n !== vote.toNeuron);
						if (altNeuron) patternConnections.add(altNeuron);
					}
				}
		}

		if (patternConnections.size === 0) return null;

		// Add context to peak's routing table - only include context neurons at same level
		const patternContext = new Context();
		for (const { neuron, distance } of context) {
			if (neuron.level !== this.level) continue;
			patternContext.add(neuron, distance, 1);
		}

		// Create pattern - add connections to pattern (at distance = age)
		const pattern = Neuron.createPattern(this.level + 1, this);
		for (const inferredNeuron of patternConnections) pattern.getOrCreateConnection(age, inferredNeuron);

		// add the context to the peak's routing table for the pattern
		this.contexts.push({ context: patternContext, pattern });

		// Return pattern - brain will handle activation
		return pattern;
	}

	/**
	 * Check if neuron can be deleted
	 */
	canDelete(brain) {
		if (this.isSensory) return false;
		return this.incomingCount === 0 &&
			this.connections.size === 0 &&
			!brain.isNeuronActive(this);
	}

	/**
	 * Clean up all references when deleting this pattern
	 */
	cleanup() {
		if (!this.isPattern) return;

		// Remove from peak's routing table
		const idx = this.peak.contexts.findIndex(e => e.pattern === this);
		if (idx !== -1) {
			const entry = this.peak.contexts[idx];
			// Decrement incoming counts for context neurons
			for (const { neuron } of entry.context.entries)
				neuron.incomingCount--;
			this.peak.contexts.splice(idx, 1);
		}
		this.peak.incomingCount--;

		// Decrement incoming counts for connection targets
		for (const distanceMap of this.connections.values())
			for (const toNeuron of distanceMap.keys())
				toNeuron.incomingCount--;
	}
}