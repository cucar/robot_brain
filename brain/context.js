/**
 * Context - represents a set of neurons at distances with strengths.
 * Used both for observed context (from brain) and known contexts (in neuron routing tables).
 */
export class Context {

	// Hyperparameters (shared with Neuron)
	static maxStrength = 100;
	static minStrength = 0;
	static mergeThreshold = 0.5; // use 0.5 for stocks, 0.8 for text
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
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed, decay) {

		// Single pass: categorize into common/missing while computing score and counts
		const common = [];
		const missing = [];
		let totalCount = 0;
		let score = 0;

		// process all neurons in the known context
		for (const [neuron, distanceMap] of this.entries) {

			// get the observed distances for the neuron in the observed context
			const observedDistances = observed.entries.get(neuron);

			// process all distances for the neuron in the known context
			for (const [distance, strength] of distanceMap) {

				// calculate the effective strength of the entry - if it is zero or less, it will be deleted
				const effectiveStrength = Math.max(Context.minStrength, strength - decay);
				if (effectiveStrength <= 0) continue;
				totalCount++;

				// if the observed context has the neuron at the same distance, it is a common entry
				if (observedDistances && observedDistances.has(distance)) {
					common.push({ neuron, distance, strength: effectiveStrength });
					score += effectiveStrength;
				}
				// otherwise, it is a missing entry
				else missing.push({ neuron, distance, strength: effectiveStrength });
			}
		}

		// if there are no known context entries, there cannot be a match
		if (totalCount === 0) return null;

		// check match threshold to decide if there is a match or not
		if (common.length / totalCount < Context.mergeThreshold) return null;

		// match found - find the novel entries (in observed but not in this known context) using lookups
		const novel = [];
		for (const [neuron, distanceMap] of observed.entries) {
			const knownDistances = this.entries.get(neuron);
			for (const [distance, strength] of distanceMap)
				if (!knownDistances || !knownDistances.has(distance) || Math.max(Context.minStrength, knownDistances.get(distance) - decay) <= 0)
					novel.push({ neuron, distance, strength });
		}

		// Round to 14 decimal places to avoid floating-point precision issues
		score = Math.round(score * 1e14) / 1e14;

		return { score, common, missing, novel };
	}
}