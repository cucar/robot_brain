/**
 * Context - represents a set of neurons at distances with strengths.
 * Used both for observed context (from brain) and known contexts (in neuron routing tables).
 */
export class Context {

	constructor() {
		this.entries = new Map(); // Map<neuronId, Map<distance, strength>>
	}

	get size() {
		let count = 0;
		for (const distanceMap of this.entries.values()) count += distanceMap.size;
		return count;
	}

	/**
	 * returns entries for the context as an array
	 * @returns Array<{neuronId, distance, strength}>
	 */
	getEntries() {
		const result = [];
		for (const [neuronId, distanceMap] of this.entries)
			for (const [distance, strength] of distanceMap)
				result.push({ neuronId, distance, strength });
		return result;
	}

	/**
	 * Add or update an entry.
	 */
	addNeuron(neuronId, distance, strength = 1) {
		if (!this.entries.has(neuronId)) this.entries.set(neuronId, new Map());
		const distanceMap = this.entries.get(neuronId);
		if (distanceMap.has(distance)) throw new Error('Context entry already exists');
		distanceMap.set(distance, strength);
	}

	/**
	 * Find an entry by neuron ID and distance.
	 */
	find(neuronId, distance) {
		const distanceMap = this.entries.get(neuronId);
		if (!distanceMap) return null;
		const strength = distanceMap.get(distance);
		return strength !== undefined ? { neuronId, distance, strength } : null;
	}

	/**
	 * increases the strength of an entry.
	 */
	strengthenNeuron(neuronId, distance) {
		const distanceMap = this.entries.get(neuronId);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for strengthening');
		const strength = distanceMap.get(distance);
		distanceMap.set(distance, strength + 1);
	}

	/**
	 * Reduces the strength of an entry when not observed.
	 * Auto-deletes the entry when strength reaches zero.
	 * @returns {boolean} true if the entry was deleted, false if it was only weakened
	 */
	weakenNeuron(neuronId, distance) {
		const distanceMap = this.entries.get(neuronId);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for weakening');
		const strength = distanceMap.get(distance);
		const newStrength = strength - 1;
		if (newStrength <= 0) {
			distanceMap.delete(distance);
			if (distanceMap.size === 0) this.entries.delete(neuronId);
			return true;
		}
		distanceMap.set(distance, newStrength);
		return false;
	}

	/**
	 * Remove an entry.
	 */
	remove(neuronId, distance) {
		const distanceMap = this.entries.get(neuronId);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for deletion');
		distanceMap.delete(distance);
		if (distanceMap.size === 0) this.entries.delete(neuronId);
		return true;
	}

	/**
	 * Check if a key exists in this context.
	 */
	hasKey(neuronId, distance) {
		const distanceMap = this.entries.get(neuronId);
		return distanceMap ? distanceMap.has(distance) : false;
	}

	/**
	 * Score a known context entry against the observed context.
	 * Returns full strength for exact match, partial credit for distance mismatch, negative for missing.
	 */
	getMatchScore(strength, distance, observedDistances) {
		if (observedDistances?.has(distance)) return strength;
		if (!observedDistances) return -strength;
		let minDelta = Infinity;
		for (const observedDistance of observedDistances.keys()) {
			const delta = Math.abs(observedDistance - distance);
			if (delta < minDelta) minDelta = delta;
		}
		return strength / (1 + minDelta);
	}

	/**
	 * Score a novel observed entry. Returns 0 if the neuron has a partial match
	 * in the known context (already accounted for), otherwise returns negative strength.
	 */
	hasPartialMatch(distance, knownDistances) {
		if (knownDistances)
			for (const [d, strength] of knownDistances)
				if (d !== distance && strength > 0) return true;
		return false;
	}

	/**
	 * Match this known context against an observed context.
	 * Returns match result with score, or null if below threshold.
	 * Uses effective strengths (with lazy decay applied) for scoring.
	 * @param {Context} observed - The observed context to match against
	 * @param {number} offset - The parent's active age (shifts pattern distances to absolute)
	 * @param {number} mergeThreshold - minimum required percentage for merge (0-1)
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed, offset, mergeThreshold) {

		// Single pass: categorize into common/missing while computing score and counts
		const common = [];
		const missing = [];
		let totalCount = 0;
		let score = 0;

		// process all neurons in the known context (keyed by neuron ID)
		for (const [neuronId, distanceMap] of this.entries) {

			// get the observed distances for the neuron in the observed context
			const observedDistances = observed.entries.get(neuronId);

			// process all distances for the neuron in the known context
			for (const [distance, strength] of distanceMap) {

				// if the entry strength is zero or less, it will be deleted
				if (strength <= 0) continue;
				totalCount++;

				// convert pattern-relative distance to absolute distance for comparison
				const absoluteDistance = distance + offset;

				// if the observed context has the neuron at the absolute distance, it is a common entry - otherwise missing
				if (observedDistances?.has(absoluteDistance)) common.push({ neuronId, distance, strength });
				else missing.push({ neuronId, distance, strength });

				// add the match score for the entry (using absolute distance against observed)
				score += this.getMatchScore(strength, absoluteDistance, observedDistances);
			}
		}

		// if there are no known context entries, there cannot be a match
		if (totalCount === 0) return null;

		// check match threshold to decide if there is a match or not
		if (common.length / totalCount < mergeThreshold) return null;

		// match found - find the novel entries (in observed but not in this known context) using lookups
		const novel = [];
		for (const [neuronId, distanceMap] of observed.entries) {
			const knownDistances = this.entries.get(neuronId);
			for (const [absoluteDistance, strength] of distanceMap) {

				// convert absolute distance back to pattern-relative
				const patternDistance = absoluteDistance - offset;

				// context neurons must be older than the parent (distance >= 1 in pattern-relative terms)
				if (patternDistance < 1) continue;

				if (!knownDistances || !knownDistances.has(patternDistance) || knownDistances.get(patternDistance) <= 0)
					if (!this.hasPartialMatch(patternDistance, knownDistances)) {
						novel.push({ neuronId, distance: patternDistance, strength });
						score -= strength;
					}
			}
		}

		// Round to 14 decimal places to avoid floating-point precision issues
		score = Math.round(score * 1e14) / 1e14;

		return { score, common, missing, novel };
	}
}