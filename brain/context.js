/**
 * Context - represents a set of neurons at distances with strengths.
 * Used both for observed context (from brain) and known contexts (in neuron routing tables).
 */
export class Context {

	// Hyperparameters (shared with Neuron)
	static maxStrength = 100;
	static minStrength = 0;
	static mergeThreshold = 0.5;
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
	 * reduces the strength of an entry - returns if it can be deleted
	 */
	weakenNeuron(neuron, distance, rate) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for weakening');
		const strength = distanceMap.get(distance);
		const newStrength = Math.max(Context.minStrength, strength - rate);
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
	 * Get entries with effective strengths after lazy decay applied. Filters out entries that have decayed to zero.
	 */
	getEffectiveEntries(patternLastActivationFrame, currentFrame, decayRate) {
		const decay = (currentFrame - patternLastActivationFrame) * decayRate;
		const result = [];
		for (const [neuron, distanceMap] of this.entries)
			for (const [distance, strength] of distanceMap) {
				const effectiveStrength = Math.max(Context.minStrength, strength - decay);
				if (effectiveStrength > 0) result.push({ neuron, distance, strength: effectiveStrength });
			}
		return result;
	}

	/**
	 * Match this known context against an observed context.
	 * Returns match result with score, or null if below threshold.
	 * Uses effective strengths (with lazy decay applied) for scoring.
	 * @param {Context} observed - The observed context to match against
	 * @param {number} patternLastActivationFrame - When the owning pattern was last activated
	 * @param {number} currentFrame - Current frame number for lazy decay
	 * @param {number} decayRate - decay rate to use for lazy decay
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed, patternLastActivationFrame, currentFrame, decayRate) {

		// Get entries with decay applied, filtering out zero-strength
		const effectiveEntries = this.getEffectiveEntries(patternLastActivationFrame, currentFrame, decayRate);
		if (effectiveEntries.length === 0) return null;

		// Categorize into common (present in observed) and missing (not present)
		const common = [];
		const missing = [];
		for (const entry of effectiveEntries)
			if (observed.hasKey(entry.neuron, entry.distance)) common.push(entry);
			else missing.push(entry);

		// Check match threshold - if it's below the threshold, not matched at all
		if ((common.length / effectiveEntries.length) < Context.mergeThreshold) return null;

		// Find novel (in observed but not in this known context)
		const novel = [];
		for (const [neuron, distanceMap] of observed.entries)
			for (const [distance, strength] of distanceMap)
				if (!this.hasKey(neuron, distance))
					novel.push({ neuron, distance, strength });

		// Calculate score as sum of effective strengths of common entries
		// Round to 14 decimal places to avoid floating-point precision issues
		const score = Math.round(common.reduce((sum, e) => sum + e.strength, 0) * 1e14) / 1e14;

		// return the matched context with pattern and score
		return { score, common, missing, novel };
	}

	/**
	 * Get effective size (number of entries with positive effective strength)
	 */
	getEffectiveSize(patternLastActivationFrame, currentFrame, decayRate) {
		return this.getEffectiveEntries(patternLastActivationFrame, currentFrame, decayRate).length;
	}
}