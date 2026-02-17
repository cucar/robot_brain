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
		this.keys = new Set(); // Maintained incrementally for fast matching
	}

	get size() { return this.entries.length; }

	/**
	 * Add or update an entry.
	 */
	addNeuron(neuron, distance, strength = 1) {
		const existing = this.find(neuron, distance);
		if (existing) throw new Error('Context entry already exists');
		this.entries.push({ neuron, distance, strength });
		this.keys.add(this.buildKey(neuron, distance));
	}

	/**
	 * Find an entry by neuron and distance.
	 */
	find(neuron, distance) {
		return this.entries.find(e => e.neuron === neuron && e.distance === distance) ?? null;
	}

	/**
	 * Build a lookup key for fast matching.
	 */
	buildKey(neuron, distance) {
		return `${neuron.id}:${distance}`;
	}

	/**
	 * increases the strength of an entry.
	 */
	strengthenNeuron(neuron, distance) {
		const entry = this.find(neuron, distance);
		if (!entry) throw new Error('Context entry not found for strengthening');
		entry.strength = Math.min(Context.maxStrength, entry.strength + 1);
	}

	/**
	 * reduces the strength of an entry - returns if it can be deleted
	 */
	weakenNeuron(neuron, distance) {
		const entry = this.find(neuron, distance);
		if (!entry) throw new Error('Context entry not found for weakening');
		entry.strength -= Context.negativeReinforcement;
		return entry.strength <= Context.minStrength; // return if the entry can be deleted or not
	}

	/**
	 * Remove an entry.
	 */
	remove(neuron, distance) {
		const idx = this.entries.findIndex(e => e.neuron === neuron && e.distance === distance);
		if (idx === -1) throw new Error('Context entry not found for deletion');
		this.entries.splice(idx, 1);
		this.keys.delete(this.buildKey(neuron, distance));
		return true;
	}

	/**
	 * Check if a key exists in this context.
	 */
	hasKey(neuron, distance) {
		return this.keys.has(this.buildKey(neuron, distance));
	}

	/**
	 * Match this known context against an observed context.
	 * Returns match result with score, or null if below threshold.
	 * @param {Context} observed - The observed context to match against
	 * @returns {Object|null} { score, common, missing, novel } or null
	 */
	match(observed) {
		if (this.entries.length === 0) return null;

		const common = [];
		const missing = [];

		// Check each entry in this known context and add to common or missing
		for (const entry of this.entries)
			if (observed.hasKey(entry.neuron, entry.distance)) common.push(entry);
			else missing.push(entry);

		// Check match threshold - if it's below the threshold, not matched at all
		if ((common.length / this.entries.length) < Context.mergeThreshold) return null;

		// Find novel (in observed but not in this known context)
		const novel = [];
		for (const entry of observed.entries)
			if (!this.hasKey(entry.neuron, entry.distance))
				novel.push(entry);

		// Calculate score as sum of strengths of common entries
		const score = common.reduce((sum, e) => sum + e.strength, 0);

		// return the matched context with pattern and score
		return { score, common, missing, novel };
	}
}