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
		this.entries = []; // Array<{neuron, distance, strength}>
	}

	get size() { return this.entries.length; }

	/**
	 * Add or update an entry.
	 */
	add(neuron, distance, strength = 1) {
		const existing = this.find(neuron, distance);
		if (existing) {
			existing.strength = Math.min(Context.maxStrength, existing.strength + strength);
			return existing;
		}
		const entry = { neuron, distance, strength };
		this.entries.push(entry);
		neuron.incomingCount++;
		return entry;
	}

	/**
	 * Find an entry by neuron and distance.
	 */
	find(neuron, distance) {
		return this.entries.find(e => e.neuron === neuron && e.distance === distance) ?? null;
	}

	/**
	 * Remove an entry.
	 */
	remove(neuron, distance) {
		const idx = this.entries.findIndex(e => e.neuron === neuron && e.distance === distance);
		if (idx === -1) return false;
		this.entries.splice(idx, 1);
		neuron.incomingCount--;
		return true;
	}

	/**
	 * Build a lookup key for fast matching.
	 */
	buildKey(neuron, distance) {
		return `${neuron.id}:${distance}`;
	}

	/**
	 * Build a lookup set from entries for fast matching.
	 */
	buildLookupSet() {
		const set = new Set();
		for (const { neuron, distance } of this.entries)
			set.add(this.buildKey(neuron, distance));
		return set;
	}

	/**
	 * Match this known context against an observed context.
	 * Returns match result with score, or null if below threshold.
	 * @param {Context} observed - The observed context to match against
	 * @param {Neuron} pattern - The pattern this context maps to
	 * @returns {Object|null} { context, pattern, score, common, missing, novel } or null
	 */
	match(observed, pattern) {
		if (this.entries.length === 0) return null;

		const observedSet = observed.buildLookupSet();
		const common = [];
		const missing = [];

		// Check each entry in this known context and add to common or missing
		for (const entry of this.entries)
			if (observedSet.has(this.buildKey(entry.neuron, entry.distance))) common.push(entry);
			else missing.push(entry);

		// Check match threshold - if it's below the threshold, not matched at all
		if ((common.length / this.entries.length) < Context.mergeThreshold) return null;

		// Find novel (in observed but not in this known context)
		const novel = [];
		const knownSet = this.buildLookupSet();
		for (const entry of observed.entries)
			if (!knownSet.has(this.buildKey(entry.neuron, entry.distance)))
				novel.push(entry);

		// Calculate score as sum of strengths of common entries
		const score = common.reduce((sum, e) => sum + e.strength, 0);

		// return the matched context with pattern and score
		return { pattern, score, common, missing, novel };
	}

	/**
	 * Refine this context based on a match result.
	 * Strengthens common, adds novel, weakens missing.
	 */
	refine(common, novel, missing) {

		// Strengthen common context neurons
		for (const entry of common) entry.strength = Math.min(Context.maxStrength, entry.strength + 1);

		// Add novel context neurons
		for (const { neuron, distance } of novel) this.add(neuron, distance, 1);

		// Weaken missing and delete if necessary
		for (const entry of missing) {
			entry.strength -= Context.negativeReinforcement;
			if (entry.strength <= Context.minStrength) this.remove(entry.neuron, entry.distance);
		}
	}
}