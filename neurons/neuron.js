import { Context } from './context.js';

/**
 * Neuron - Unified class for all neurons (sensory and pattern)
 *
 * All neurons have:
 * - connections: Map<distance, Map<toNeuron, {strength, reward}>> - predictions
 * - contexts: Array<{context: Context, pattern: Neuron}> - routing table
 * - activeAges: Map<age, patternNeuron|null> - active ages and which pattern (if any) was activated
 *
 * Level 0 (sensory) neurons additionally have: channel, type, coordinates
 * Level > 0 (pattern) neurons additionally have: peak, peakStrength
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
		// Active ages: Map<age, patternNeuron|null> - null means no pattern activated at that age
		this.activeAges = null;
		// Predictions: Map<distance, Map<toNeuron, {strength, reward}>>
		this.connections = new Map();
		// Routing table: Array<{context: Context, pattern: Neuron}>
		this.contexts = [];
		// Votes made this frame: Map<distance, Array<{target, strength, reward}>>
		this.votes = new Map();
	}

	get isActive() { return this.activeAges !== null && this.activeAges.size > 0; }
	get isNewlyActive() { return this.activeAges?.has(0) ?? false; }
	get isSensory() { return this.level === 0; }
	get isPattern() { return this.level > 0; }

	/** Get value key for this neuron (sensory only) */
	get valueKey() {
		return Neuron.makeValueKey(this.coordinates);
	}

	/**
	 * Age this neuron by incrementing all active ages by 1.
	 * @returns {Map<number, Neuron|null>} The new activeAges Map
	 */
	age() {
		if (!this.activeAges) return null;
		const newAges = new Map();
		for (const [a, pattern] of this.activeAges)
			newAges.set(a + 1, pattern);
		this.activeAges = newAges;
		return newAges;
	}

	/**
	 * Remove ages that have exceeded the context window.
	 * @param {number} contextLength - Maximum age before deactivation
	 * @returns {number} Number of ages removed
	 */
	deactivateAgedOut(contextLength) {
		if (!this.activeAges) return 0;
		let removed = 0;
		for (const a of this.activeAges.keys())
			if (a >= contextLength) {
				this.activeAges.delete(a);
				removed++;
			}
		if (this.activeAges.size === 0) this.activeAges = null;
		return removed;
	}

	/**
	 * Activate this neuron at age 0 with optional pattern.
	 */
	activate() {

		// activate this neuron at age 0
		if (!this.activeAges) this.activeAges = new Map();
		this.activeAges.set(0, null); // pattern will be activated later if there is a match

		// if this is a pattern, increase its peak strength since it must have been activated via match
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
	 * Collect votes from this neuron for inference.
	 * Skips ages where a pattern was activated (pattern votes for itself).
	 * @param {number} contextLength - Context window length for time decay
	 * @returns {Array} Array of vote objects {toNeuron, strength, reward, distance}
	 */
	collectVotes(contextLength) {
		this.votes.clear();
		if (!this.activeAges) return [];

		const result = [];
		const timeDecay = 1 / contextLength;

		for (const [age, activatedPattern] of this.activeAges) {
			// If a pattern was activated at this age, skip - the pattern votes when we iterate over it
			if (activatedPattern !== null) continue;

			const distance = age + 1;
			const levelWeight = 1 + this.level * Neuron.levelVoteMultiplier;
			const timeWeight = 1 - (distance - 1) * timeDecay;

			const distanceMap = this.connections.get(distance);
			if (!distanceMap) continue;

			for (const [toNeuron, conn] of distanceMap) {
				if (conn.strength <= 0) continue;
				const weightedStrength = levelWeight * timeWeight * conn.strength;
				result.push({ toNeuron, strength: weightedStrength, reward: conn.reward, distance });
				// Store vote for pattern learning
				if (!this.votes.has(distance)) this.votes.set(distance, []);
				this.votes.get(distance).push({ target: toNeuron, strength: weightedStrength, reward: conn.reward });
			}
		}

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
	 * @returns {Neuron|null} The matched pattern, or null if no match
	 */
	matchBestPattern(observed) {

		// try to match the observed context to known patterns
		let best = null; // { context, pattern, score, common, missing, novel }
		for (const { context, pattern } of this.contexts) {
			const match = context.match(observed, pattern);
			if (match && (!best || match.score > best.score)) best = { ...match, context };
		}
		if (!best) return null; // if there are no matches, return null

		// set the best matching pattern as active at age 0 - it will be aged later
		this.activeAges.set(0, best.pattern);

		// refine the context of the best matching pattern based on observed context
		best.context.refine(best.common, best.novel, best.missing);

		// return the best matched pattern
		return best.pattern;
	}

	/**
	 * Update connections based on observations at a distance.
	 * Events: strengthen correct, weaken incorrect, add novel.
	 * Actions: update with rewards, add alternatives for painful actions.
	 */
	updateConnections(newEventNeurons, newActionNeurons, rewards, channelActions) {

		// Only active neurons can update connections
		if (!this.activeAges) throw new Error(`Neuron ${this.id} in activeNeurons has no activeAges`);

		// Only event neurons (level 0) or pattern neurons can update connections
		if (this.level === 0 && this.type !== 'event') return;

		// Update connections at each active age > 0
		for (const age of this.activeAges.keys()) {

			// if the neuron is active at age=0, nothing to learn from that - no future connections from it yet
			// that said, if it was also active in older ages, it can update connections from those ages
			if (age === 0) continue;

			const distance = age;
			if (!this.connections.has(distance)) this.connections.set(distance, new Map());
			const distanceMap = this.connections.get(distance);
			const painfulChannels = new Set();

			// Refine existing event predictions
			for (const [inferredNeuron, prediction] of distanceMap) {
				if (inferredNeuron.level !== 0 || inferredNeuron.type !== 'event') continue;
				if (newEventNeurons.has(inferredNeuron)) {
					prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
				}
				else {
					prediction.strength -= Neuron.negativeReinforcement;
					if (prediction.strength <= Neuron.minStrength) this.deleteConnection(distance, inferredNeuron);
				}
			}

			// Add novel event predictions
			for (const eventNeuron of newEventNeurons)
				if (!distanceMap.has(eventNeuron)) {
					distanceMap.set(eventNeuron, { strength: 1, reward: 0 });
					eventNeuron.incomingCount++;
				}

			// Reward/learn action predictions
			for (const actionNeuron of newActionNeurons) {
				const reward = rewards.get(actionNeuron.channel);
				if (reward === undefined) continue;

				if (distanceMap.has(actionNeuron)) {
					const prediction = distanceMap.get(actionNeuron);
					prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
					prediction.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * prediction.reward;
					if (prediction.reward < Neuron.actionRegretMinPain) painfulChannels.add(actionNeuron.channel);
				}
				else if (reward !== 0) {
					distanceMap.set(actionNeuron, { strength: 1, reward });
					actionNeuron.incomingCount++;
					if (reward < Neuron.actionRegretMinPain) painfulChannels.add(actionNeuron.channel);
				}
			}

			// Add alternative actions for painful channels
			for (const channel of painfulChannels) {
				const allActions = channelActions.get(channel);
				if (!allActions) continue;
				for (const altNeuron of allActions)
					if (!distanceMap.has(altNeuron)) {
						distanceMap.set(altNeuron, { strength: 1, reward: 0 });
						altNeuron.incomingCount++;
						break;
					}
			}
		}
	}

	/**
	 * Learn new pattern from prediction errors and action regret at a specific age.
	 * Only called for ages where no pattern was activated (activeAges.get(age) === null).
	 * @param {number} age - The age at which this neuron made a bad inference
	 * @param {Set<Neuron>} newlyActiveNeurons - Currently active neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @param {Array<{neuron: Neuron, distance: number}>} context - Active context neurons with distances
	 * @returns {Neuron|null} Newly created pattern, or null if no pattern needed
	 */
	learnNewPattern(age, newlyActiveNeurons, rewards, channelActions, context) {

		// Only event neurons (level 0) or pattern neurons can create patterns
		if (this.level === 0 && this.type !== 'event') return null;

		// Get votes at this age (distance = age)
		const votes = this.votes.get(age);
		if (!votes || votes.length === 0) return null;

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

		// Collect errors at this distance
		const patternConnections = []; // Array of {inferredNeuron}
		let hasError = false;

		for (const vote of votes) {
			const target = vote.target;
			if (target.level !== 0) continue;

			// Event error: predicted event didn't happen
			if (target.type === 'event' && vote.strength >= Neuron.eventErrorMinStrength)
				if (!target.activeAges?.has(0)) {
					for (const actualNeuron of activeEventNeurons)
						patternConnections.push({ inferredNeuron: actualNeuron });
					hasError = true;
				}

			// Action regret: predicted action was executed and was painful
			if (target.type === 'action' && vote.strength >= Neuron.actionRegretMinStrength)
				if (winnerActions.has(target) && painfulChannels.has(target.channel)) {
					const alts = channelActions.get(target.channel);
					if (alts) {
						const altNeuron = [...alts].find(n => n !== target);
						if (altNeuron) {
							patternConnections.push({ inferredNeuron: altNeuron });
							hasError = true;
						}
					}
				}
		}

		if (!hasError) return null;

		// Create pattern
		const pattern = Neuron.createPattern(this.level + 1, this);

		// Add context to peak's routing table
		const patternContext = this.getOrCreateContext(pattern);
		for (const { neuron, distance } of context) {
			// Only include context neurons at same level, not actions
			if (neuron.level !== this.level) continue;
			if (neuron.level === 0 && neuron.type === 'action') continue;
			patternContext.add(neuron, distance, 1);
		}

		// Add connections to pattern (at distance = age)
		for (const { inferredNeuron } of patternConnections)
			pattern.getOrCreateConnection(age, inferredNeuron);

		// Activate pattern at current age
		this.activeAges.set(age, pattern);
		pattern.activate(null);

		return pattern;
	}

	/**
	 * Check if neuron can be deleted
	 */
	canDelete(brain) {
		if (this.isSensory) return false;
		return this.incomingCount === 0 &&
			this.connections.size === 0 &&
			!brain.activeNeurons.has(this);
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