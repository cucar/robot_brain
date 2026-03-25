/**
 * Context - represents a set of neurons at distances with strengths.
 * Used both for observed context (from brain) and known contexts (in neuron routing tables).
 */
export class Context {

	// Hyperparameters (shared with Neuron)
	static maxStrength = 100;
	static minStrength = 0;
	static mergeThreshold = 1; // use 0.5 for stocks, 0.8 for text
	static negativeReinforcement = 0.1;

	constructor() {
		this.entries = new Map(); // Map<neuron, Map<distance, strength>>
	}

	get size() {
		let count = 0;
		for (const distanceMap of this.entries.values()) count += distanceMap.size;
		return count;
	}

	/**
	 * returns entries for the context as an array
	 * @returns Array<{neuron, distance, strength}>
	 */
	getEntries() {
		const result = [];
		for (const [neuron, distanceMap] of this.entries)
			for (const [distance, strength] of distanceMap)
				result.push({ neuron, distance, strength });
		return result;
	}

	/**
	 * Add or update an entry.
	 */
	addNeuron(neuron, distance, strength = 1) {
		if (!this.entries.has(neuron)) this.entries.set(neuron, new Map());
		const distanceMap = this.entries.get(neuron);
		if (distanceMap.has(distance)) throw new Error('Context entry already exists');
		distanceMap.set(distance, strength);
	}

	/**
	 * Find an entry by neuron and distance.
	 */
	find(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap) return null;
		const strength = distanceMap.get(distance);
		return strength !== undefined ? { neuron, distance, strength } : null;
	}

	/**
	 * increases the strength of an entry.
	 */
	strengthenNeuron(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for strengthening');
		const strength = distanceMap.get(distance);
		distanceMap.set(distance, Math.min(Context.maxStrength, strength + 1));
	}

	/**
	 * reduces the strength of an entry when not observed - returns if it can be deleted
	 */
	weakenNeuron(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for weakening');
		const strength = distanceMap.get(distance);
		const newStrength = Math.max(Context.minStrength, strength - Context.negativeReinforcement);
		distanceMap.set(distance, newStrength);
		return newStrength <= Context.minStrength; // return if the entry can be deleted or not
	}

	/**
	 * Materialize lazy decay for all entries using the owner's activation frame.
	 */
	materialize(patternLastActivationFrame, currentFrame, rate) {
		const decay = (currentFrame - patternLastActivationFrame) * rate;
		if (decay <= 0) return [];

		const toDelete = [];
		for (const [neuron, distanceMap] of this.entries)
			for (const [distance, strength] of distanceMap) {
				const effectiveStrength = Math.max(Context.minStrength, strength - decay);
				if (effectiveStrength <= 0) toDelete.push({ neuron, distance });
				else distanceMap.set(distance, effectiveStrength);
			}

		// return the entries that can be deleted
		return toDelete;
	}

	/**
	 * Remove an entry.
	 */
	remove(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for deletion');
		distanceMap.delete(distance);
		if (distanceMap.size === 0) this.entries.delete(neuron);
		return true;
	}

	/**
	 * Check if a key exists in this context.
	 */
	hasKey(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		return distanceMap ? distanceMap.has(distance) : false;
	}

	/**
	 * Match this known context against an observed context.
	 * Returns match result with score, or null if below threshold.
	 * Uses effective strengths (with lazy decay applied) for scoring.
	 * @param {Context} observed - The observed context to match against
	 * @param {number} decay - strength decay since last activation
	 * @param {number} contextLength - the context window length for distance similarity scaling
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed, decay, contextLength) {

		// match known context entries against observed
		const { commonStrength, knownStrength, common, missing } = this.matchKnown(observed, decay, contextLength);

		// if there are no common entries, there is no match
		if (knownStrength === 0) throw new Error('zero total strength');

		// get novel context entries that are not in the known context
		const { novelStrength, novel } = this.matchNovel(observed, decay);

		// calculate the normalized match score - round to 14 decimal places to avoid floating-point precision issues
		const score = Math.round(commonStrength / (knownStrength + novelStrength) * 1e14) / 1e14;

		// if the match score is less than the threshold, it's no a match - otherwise return match details
		return score < Context.mergeThreshold ? null : { score, common, missing, novel };
	}

	/**
	 * Score known context entries against observed context.
	 * Score is neuron-level (common vs missing). Return arrays are (neuron, distance)-level.
	 */
	matchKnown(observed, decay, contextLength) {
		let knownStrength = 0;
		let commonStrength = 0;
		const common = [];
		const missing = [];

		// loop over neurons in the known context
		for (const [neuron, distanceMap] of this.entries) {

			// apply the decay to get the effective distance map and total strength of the neuron in the known context
			const { effectiveDistanceMap, strength } = this.getEffectiveDistanceMap(distanceMap, decay);

			// if the neuron has no effective strength, it has been forgotten - ignore it
			if (strength === 0) continue;

			// add the neuron's known strength to the total strength - does not matter if missing or common
			knownStrength += strength;

			// get the distances for the neuron in the observed context
			const observedDistances = observed.entries.get(neuron);

			// add the neuron common/missing distances to the returned entries to merge if it's a match
			this.populateCommonMissing(neuron, effectiveDistanceMap, observedDistances, common, missing);

			// if the neuron is in the observed context, add the common strength
			if (observedDistances) commonStrength += this.getCommonStrength(effectiveDistanceMap, observedDistances, contextLength);
		}

		return { commonStrength, knownStrength, common, missing };
	}

	/**
	 * returns effective context distance map after applying decay
	 */
	getEffectiveDistanceMap(distanceMap, decay) {
		let strength = 0;
		const effectiveDistanceMap = new Map();
		for (const [distance, rawStrength] of distanceMap) {
			const effectiveStrength = Math.max(Context.minStrength, rawStrength - decay);
			if (effectiveStrength > 0) effectiveDistanceMap.set(distance, effectiveStrength);
			strength += effectiveStrength;
		}
		return { effectiveDistanceMap, strength };
	}

	/**
	 * populate common/missing entries for a known neuron
	 */
	populateCommonMissing(neuron, distanceMap, observedDistances, common, missing) {
		for (const [distance, strength] of distanceMap) {
			if (observedDistances?.has(distance)) common.push({ neuron, distance, strength });
			else missing.push({ neuron, distance, strength });
		}
	}

	/**
	 * returns the common strength for a known neuron in the observed context
	 * considers all pairs of known/observed distances, weighted by known strength
	 */
	getCommonStrength(knownDistanceMap, observedDistances, contextLength) {
		let weightedDeltaSum = 0;
		let totalWeight = 0;
		for (const [kDist, strength] of knownDistanceMap)
			for (const oDist of observedDistances.keys()) {
				weightedDeltaSum += strength * Math.abs(kDist - oDist);
				totalWeight += strength;
			}
		const avgDelta = weightedDeltaSum / totalWeight;
		let neuronStrength = 0;
		for (const s of knownDistanceMap.values()) neuronStrength += s;
		return neuronStrength * (1 - avgDelta / contextLength);
	}

	/**
	 * Find novel entries in observed context and compute novel penalty.
	 */
	matchNovel(observed, decay) {
		let novelStrength = 0;
		const novel = [];

		for (const { neuron, distance, strength } of observed.getEntries()) {
			if (this.neuronExists(neuron, distance, decay)) continue;
			novel.push({ neuron, distance, strength });
			novelStrength += strength;
		}

		return { novelStrength, novel };
	}

	/**
	 * Check if a (neuron, distance) entry exists and has positive effective strength.
	 */
	neuronExists(neuron, distance, decay) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) return false;
		return Math.max(Context.minStrength, distanceMap.get(distance) - decay) > 0;
	}
}