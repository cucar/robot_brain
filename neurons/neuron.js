/**
 * Neuron - Abstract base class for all neurons
 * Contains properties common to both sensory and pattern neurons
 */
export class Neuron {

	// Static counter for assigning unique IDs to neurons
	// Note: IDs are temporary during transition - will be removed later
	static nextId = 1;

	// Hyperparameters shared by all neurons
	static maxStrength = 100;
	static minStrength = 0;
	static rewardSmoothing = 0.9; // exponential smoothing: new = smooth * observed + (1 - smooth) * old

	constructor(level = 0) {
		this.id = Neuron.nextId++;
		this.level = level;

		// Count of incoming references (connections, pattern_peaks, pattern_past, pattern_future)
		// Used for orphan detection - when this reaches 0 and neuron has no outgoing, it can be deleted
		this.incomingCount = 0;

		// Active ages: Set<age> when active, null when not active
		this.activeAges = null;

		// Patterns that have this neuron as their peak: Set<PatternNeuron>
		// Used for pattern matching - when this neuron activates, check which pattern matches context
		this.patterns = new Set();
	}

	get isActive() { return this.activeAges !== null && this.activeAges.size > 0; }
	get isNewlyActive() { return this.activeAges?.has(0) ?? false; }

	/**
	 * Age this neuron by incrementing all active ages by 1.
	 * Creates a new Set to maintain shared reference integrity.
	 * @returns {Set<number>} The new ages Set (for brain to update its map)
	 */
	age() {
		if (!this.activeAges) return null;
		const newAges = new Set();
		for (const a of this.activeAges) newAges.add(a + 1);
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
		for (const a of this.activeAges)
			if (a >= contextLength) {
				this.activeAges.delete(a);
				removed++;
			}
		if (this.activeAges.size === 0) this.activeAges = null;
		return removed;
	}

	/**
	 * Find the best matching pattern for this peak neuron given the active context.
	 * Iterates all patterns that have this neuron as peak, finds best match,
	 * and automatically refines/strengthens the matched pattern.
	 * Score = sum of strengths of matching context (same as SQL algorithm).
	 * @param {Set<Neuron>} contextNeurons - Active neurons at this level (each has .activeAges)
	 * @returns {PatternNeuron|null} The matched pattern, or null if no match
	 */
	matchBestPattern(contextNeurons) {
		if (this.patterns.size === 0) return null;

		let bestPattern = null;
		let bestScore = 0;
		let bestMatch = null;

		for (const pattern of this.patterns) {
			const match = pattern.matchContext(contextNeurons);
			if (match.matches && (bestPattern === null || match.score > bestScore)) {
				bestPattern = pattern;
				bestScore = match.score;
				bestMatch = match;
			}
		}

		if (bestPattern) bestPattern.refinePast(bestMatch);

		return bestPattern;
	}

	/**
	 * Update inferences for the neuron at the given distance.
	 * @param {number} distance - The distance to refine (pattern's current age)
	 * @param {Set<Neuron>} newlyActiveNeurons - Currently active neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Map of channel name to all action neurons
	 * @returns {{strengthened: number, weakened: number, novel: number, rewarded: number, learned: number, alternatives: number}}
	 */
	updateInferences(distance, newlyActiveNeurons, rewards, channelActions) {

	}
}
