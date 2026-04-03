import { Neuron } from './neuron.js';

/**
 * Context - represents a set of neurons at distances with strengths.
 * Used both for observed context (from brain) and known contexts (in neuron routing tables).
 */
export class Context {

	// Hyperparameters (shared with Neuron)
	static mergeThreshold = 0.5;

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
		distanceMap.set(distance, Math.min(Neuron.maxStrength, strength + Neuron.positiveReinforcement));
	}

	/**
	 * reduces the strength of an entry when not observed - returns if it can be deleted
	 */
	weakenNeuron(neuron, distance) {
		const distanceMap = this.entries.get(neuron);
		if (!distanceMap || !distanceMap.has(distance)) throw new Error('Context entry not found for weakening');
		const strength = distanceMap.get(distance);
		const newStrength = Math.max(Neuron.minStrength, strength - Neuron.negativeReinforcement);
		distanceMap.set(distance, newStrength);
		return newStrength <= Neuron.minStrength; // return if the entry can be deleted or not
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
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed) {

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
				const effectiveStrength = Math.max(Neuron.minStrength, strength);
				if (effectiveStrength <= 0) continue;
				totalCount++;

				// if the observed context has the neuron at the same distance, it is a common entry - otherwise missing
				if (observedDistances?.has(distance)) common.push({ neuron, distance, strength: effectiveStrength });
				else missing.push({ neuron, distance, strength: effectiveStrength });

				// add the match score for the entry
				score += this.getMatchScore(effectiveStrength, distance, observedDistances);
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
				if (!knownDistances || !knownDistances.has(distance) || Math.max(Neuron.minStrength, knownDistances.get(distance)) <= 0)
					if (!this.hasPartialMatch(distance, knownDistances)) {
						novel.push({ neuron, distance, strength });
						score -= strength;
					}
		}

		// Round to 14 decimal places to avoid floating-point precision issues
		score = Math.round(score * 1e14) / 1e14;

		return { score, common, missing, novel };
	}
}