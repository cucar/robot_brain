import { Neuron } from './neuron.js';

/**
 * PatternNeuron - Level > 0 neurons representing learned patterns
 * These have a peak neuron, context (past), and predictions (future)
 */
export class PatternNeuron extends Neuron {

	// Pattern-specific hyperparameters
	static mergeThreshold = 0.5;          // Fraction of context that must match to recognize pattern
	static negativeReinforcement = 0.1;   // How much to weaken missing context
	static actionRegretMinPain = 0;       // Threshold below which reward is painful (0 = any negative triggers regret)

	constructor(level, peak) {
		super(level);

		// Peak neuron this pattern differentiates (pointer to SensoryNeuron or PatternNeuron)
		this.peak = peak;
		this.peakStrength = 1.0;
		peak.incomingCount++;
		peak.patterns.add(this); // Register with peak for pattern matching

		// Context for matching: Map<contextNeuron, Map<age, strength>>
		// Age-first would be: Map<age, Map<contextNeuron, strength>> - but we iterate by neuron more often
		this.past = new Map();

		// Predictions: Map<distance, Map<inferredNeuron, {strength, reward}>>
		// Distance-first indexing (same as connections) for O(1) inference lookup
		this.future = new Map();
	}

	/**
	 * Add or strengthen context neuron at age
	 * @param {Neuron} contextNeuron - Context neuron
	 * @param {number} age - Context age
	 * @param {number} strength - Initial or added strength
	 */
	addContext(contextNeuron, age, strength = 1.0) {
		if (!this.past.has(contextNeuron)) {
			this.past.set(contextNeuron, new Map());
			contextNeuron.incomingCount++;
		}
		const ageMap = this.past.get(contextNeuron);
		if (ageMap.has(age))
			ageMap.set(age, ageMap.get(age) + strength);
		else
			ageMap.set(age, strength);
	}

	/**
	 * Remove context entry for neuron at age
	 * @returns {boolean} True if entry existed and was removed
	 */
	removeContext(contextNeuron, age) {
		const ageMap = this.past.get(contextNeuron);
		if (!ageMap || !ageMap.has(age)) return false;
		ageMap.delete(age);
		if (ageMap.size === 0) {
			this.past.delete(contextNeuron);
			contextNeuron.incomingCount--;
		}
		return true;
	}

	/**
	 * Check if this pattern matches the given active context.
	 * @param {Set<Neuron>} contextNeurons - Active neurons (each has .activeAges Set)
	 * @returns {{matches: boolean, score: number, common: Array, novel: Array, missing: Array}}
	 */
	matchContext(contextNeurons) {
		const common = [];  // In pattern AND active
		const missing = []; // In pattern but NOT active

		// Check each entry in pattern's past against active context
		let totalPast = 0;
		for (const [contextNeuron, ageMap] of this.past) {
			for (const [age, strength] of ageMap) {
				totalPast++;
				if (contextNeurons.has(contextNeuron) && contextNeuron.activeAges.has(age))
					common.push({ neuron: contextNeuron, age, strength });
				else
					missing.push({ neuron: contextNeuron, age, strength });
			}
		}

		// Novel: active context not in pattern's past
		const novel = [];
		for (const neuron of contextNeurons)
			for (const age of neuron.activeAges)
				if (age > 0 && !this.past.get(neuron)?.has(age))
					novel.push({ neuron, age });

		const matchRatio = totalPast > 0 ? common.length / totalPast : 1;
		const matches = matchRatio >= PatternNeuron.mergeThreshold;
		const score = common.reduce((sum, c) => sum + c.strength, 0);

		return { matches, score, common, novel, missing };
	}

	/**
	 * Refine this pattern based on match analysis.
	 * Strengthens peak, strengthens common context, adds novel context, weakens missing context.
	 * @param {{common: Array, novel: Array, missing: Array}} match - Match result from matchContext
	 */
	refine(match) {
		// Strengthen peak
		this.peakStrength = Math.min(Neuron.maxStrength, this.peakStrength + 1);

		// Strengthen common context
		for (const { neuron, age } of match.common) {
			const ageMap = this.past.get(neuron);
			const newStrength = Math.min(Neuron.maxStrength, ageMap.get(age) + 1);
			ageMap.set(age, newStrength);
		}

		// Add novel context
		for (const { neuron, age } of match.novel)
			this.addContext(neuron, age, 1);

		// Weaken missing context
		for (const { neuron, age } of match.missing) {
			const ageMap = this.past.get(neuron);
			const newStrength = ageMap.get(age) - PatternNeuron.negativeReinforcement;
			if (newStrength <= Neuron.minStrength)
				this.removeContext(neuron, age);
			else
				ageMap.set(age, newStrength);
		}
	}

	/**
	 * Refine event predictions in future based on current observations.
	 * Called when pattern is active at age > 0 (distance = age).
	 * @param {number} distance - The distance to refine (pattern's current age)
	 * @param {Set<Neuron>} activeEventNeurons - Currently active event neurons (age=0)
	 * @returns {{strengthened: number, weakened: number, novel: number}}
	 */
	refineEventFuture(distance, activeEventNeurons) {
		let strengthened = 0, weakened = 0, novel = 0;

		// Get or create distanceMap - we'll be adding novel entries anyway
		if (!this.future.has(distance)) this.future.set(distance, new Map());
		const distanceMap = this.future.get(distance);

		// Check existing predictions at this distance
		for (const [inferredNeuron, prediction] of distanceMap) {

			// Only refine event predictions (level 0, type 'event')
			if (inferredNeuron.level !== 0 || inferredNeuron.type !== 'event') continue;

			// Strengthen: prediction was correct
			if (activeEventNeurons.has(inferredNeuron)) {
				prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
				strengthened++;
			}
			// Weaken: prediction was wrong
			else {
				prediction.strength -= PatternNeuron.negativeReinforcement;
				if (prediction.strength <= Neuron.minStrength) this.deleteFuture(distance, inferredNeuron);
				weakened++;
			}
		}

		// Add novel: active event neurons not yet predicted at this distance
		for (const eventNeuron of activeEventNeurons) {
			if (!distanceMap.has(eventNeuron)) {
				distanceMap.set(eventNeuron, { strength: 1, reward: 0 });
				eventNeuron.incomingCount++;
				novel++;
			}
		}

		return { strengthened, weakened, novel };
	}

	/**
	 * Refine action predictions in future based on executed actions and rewards.
	 * Called when pattern is active at age > 0 (distance = age).
	 * @param {number} distance - The distance to refine (pattern's current age)
	 * @param {Set<Neuron>} activeActionNeurons - Currently active action neurons (age=0)
	 * @param {Map<string, number>} channelRewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @returns {{rewarded: number, learned: number, alternatives: number}}
	 */
	refineActionFuture(distance, activeActionNeurons, channelRewards, channelActions) {
		let rewarded = 0, learned = 0, alternatives = 0;

		// Get or create distanceMap
		if (!this.future.has(distance)) this.future.set(distance, new Map());
		const distanceMap = this.future.get(distance);

		// Track which channels have painful actions at this distance (for alternatives)
		const painfulChannels = new Set();

		// Reward existing action predictions that were executed
		for (const actionNeuron of activeActionNeurons) {
			const reward = channelRewards.get(actionNeuron.channel);
			if (reward === undefined) continue; // No reward for this channel

			// 1. Strengthen and update reward via exponential smoothing
			if (distanceMap.has(actionNeuron)) {
				const prediction = distanceMap.get(actionNeuron);
				prediction.strength = Math.min(Neuron.maxStrength, prediction.strength + 1);
				prediction.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * prediction.reward;
				rewarded++;

				// Track if this is painful
				if (prediction.reward < PatternNeuron.actionRegretMinPain) painfulChannels.add(actionNeuron.channel);
			}
			// 2. Learn new action (not in the future, has non-zero reward)
			else if (reward !== 0) {
				distanceMap.set(actionNeuron, { strength: 1, reward });
				actionNeuron.incomingCount++;
				learned++;

				// Track if this is painful
				if (reward < PatternNeuron.actionRegretMinPain) painfulChannels.add(actionNeuron.channel);
			}
		}

		// 3. Add alternative actions for painful channels
		for (const channel of painfulChannels) {
			const allActions = channelActions.get(channel);
			if (!allActions) continue;

			// Find one untried action in this channel
			for (const altNeuron of allActions) {
				if (!distanceMap.has(altNeuron)) {
					distanceMap.set(altNeuron, { strength: 1, reward: 0 });
					altNeuron.incomingCount++;
					alternatives++;
					break; // Only add one alternative per channel
				}
			}
		}

		return { rewarded, learned, alternatives };
	}

	/**
	 * Get or create future prediction at distance to inferred neuron
	 * @param {number} distance - Temporal distance
	 * @param {Neuron} inferredNeuron - Predicted neuron
	 * @returns {{strength: number, reward: number}} Prediction object
	 */
	getOrCreateFuture(distance, inferredNeuron) {
		if (!this.future.has(distance)) this.future.set(distance, new Map());
		const distanceMap = this.future.get(distance);
		if (!distanceMap.has(inferredNeuron)) {
			distanceMap.set(inferredNeuron, { strength: 1, reward: 0 });
			inferredNeuron.incomingCount++;
		}
		return distanceMap.get(inferredNeuron);
	}

	/**
	 * Delete future prediction at distance to inferred neuron
	 * @returns {boolean} True if prediction existed and was deleted
	 */
	deleteFuture(distance, inferredNeuron) {
		const distanceMap = this.future.get(distance);
		if (!distanceMap || !distanceMap.has(inferredNeuron)) return false;
		distanceMap.delete(inferredNeuron);
		inferredNeuron.incomingCount--;
		if (distanceMap.size === 0) this.future.delete(distance);
		return true;
	}

	/**
	 * Check if pattern can be deleted (orphaned)
	 * Pattern neurons can be deleted when they have no references and no content
	 */
	canDelete(brain) {
		return this.incomingCount === 0 &&
			this.past.size === 0 &&
			this.future.size === 0 &&
			!brain.activeNeurons.has(this);
	}

	/**
	 * Clean up all references when deleting this pattern
	 * Must be called before removing from brain.neurons
	 */
	cleanup() {
		// Unregister from peak's patterns Set
		this.peak.patterns.delete(this);
		this.peak.incomingCount--;

		// Decrement all context neurons' incoming counts
		for (const contextNeuron of this.past.keys())
			contextNeuron.incomingCount--;

		// Decrement all inferred neurons' incoming counts
		for (const distanceMap of this.future.values())
			for (const inferredNeuron of distanceMap.keys())
				inferredNeuron.incomingCount--;
	}
}