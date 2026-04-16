/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * A dumb data store: stores neuron IDs and states, returns raw data.
 * All neuron-property-aware filtering is done by the caller (Brain).
 */
export class Memory {

	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

		// Active context indexed by level first, then neuron id, then age:
		// Map<level, Map<neuronId, Map<age, {activatedPatternId, votes, context}>>>
		// activatedPatternId: ID of pattern neuron activated by this neuron at that age, or null
		// votes: array of votes cast by this neuron at that age, or null if it hasn't voted yet
		// context: array of context neurons at voting time, or null if it hasn't voted yet
		this.activeNeurons = new Map();

		// Current size of the sliding window (number of age slots in use, capped at contextLength)
		this.depth = 0;

		// Current frame winning inferences: Array<{neuron, strength}>
		this.inferredNeurons = [];

		// carry over the debug flag
		this.debug = debug;
	}

	/**
	 * Reset context (active neurons, inferred neurons, votes)
	 */
	reset() {
		this.activeNeurons = new Map();
		this.depth = 0;
		this.inferredNeurons = [];
	}

	/**
	 * Age all neurons by shifting each neuron's age map up by one,
	 * dropping entries that age out of the context window.
	 */
	age() {
		if (this.debug) console.log('Aging neurons...');

		// re-build age maps of neurons at each level, increasing ages, pruning old ones
		let removedAges = 0;
		for (const [level, levelNeurons] of this.activeNeurons) {
			for (const [neuronId, oldAges] of levelNeurons) {

				// increment ages and prune old ages in the new collection
				const newAges = new Map();
				for (const [oldAge, state] of oldAges) {
					if (oldAge + 1 < this.contextLength) newAges.set(oldAge + 1, state);
					else removedAges++;
				}

				// if no active ages for the neuron, delete it - otherwise, update the age map
				if (newAges.size === 0) levelNeurons.delete(neuronId);
				else levelNeurons.set(neuronId, newAges);
			}
			if (levelNeurons.size === 0) this.activeNeurons.delete(level);
		}

		// Grow the sliding window up to contextLength
		if (this.depth < this.contextLength) this.depth++;

		if (this.debug && removedAges > 0) console.log(`Removed ${removedAges} ages of neurons`);
	}

	/**
	 * Get the levels currently present in memory (sorted ascending)
	 */
	getLevels() {
		return [...this.activeNeurons.keys()].sort((a, b) => a - b);
	}

	/**
	 * Get the maximum level currently present in activeNeurons
	 */
	getMaxLevel() {
		return Math.max(...this.activeNeurons.keys());
	}

	/**
	 * Get neurons active at a specific level
	 * @returns {Map<neuronId, Map<age, {activatedPatternId, votes, context}>>}
	 */
	getNeuronsAtLevel(level) {
		return this.activeNeurons.get(level) ?? new Map();
	}

	/**
	 * Get the newly activated sensory neuron IDs at age 0
	 */
	getNewSensoryNeuronIds() {
		const ids = new Set();
		for (const [neuronId, ageMap] of this.activeNeurons.get(0))
			if (ageMap.has(0)) ids.add(neuronId);
		return ids;
	}

	/**
	 * Activate a neuron at a specific age (keyed by level, neuron ID, age)
	 * @param {number} neuronId - Neuron ID to activate
	 * @param {number} level - Level of the neuron
	 * @param {number} age - Age slot
	 */
	activateNeuron(neuronId, level = 0, age = 0) {

		// get the neurons at level - add new map if the requested level did not exist yet
		let neurons = this.activeNeurons.get(level);
		if (!neurons) {
			neurons = new Map();
			this.activeNeurons.set(level, neurons);
		}

		// get the neuron state - add it if it did not exist yet
		const state = neurons.get(neuronId) ?? new Map();
		state.set(age, { activatedPatternId: null, votes: null, context: null });

		// keep ages ascending — required by Neuron.matchPatterns
		neurons.set(neuronId, new Map([...state].sort(([a], [b]) => a - b)));
	}

	/**
	 * Activate a pattern neuron and link it to its parent
	 * @param {number} patternId - The pattern neuron ID to activate
	 * @param {number} level - Level of the pattern neuron (parent level + 1)
	 * @param {number} parentId - The parent neuron ID that triggered the pattern
	 * @param {number} age - The age at which to activate
	 */
	activatePattern(patternId, level, parentId, age) {

		// activate the pattern
		this.activateNeuron(patternId, level, age);

		// update parent state so that its voting will be suppressed
		this.activeNeurons.get(level - 1).get(parentId).get(age).activatedPatternId = patternId;
	}

	/**
	 * Clear per-frame saved votes and contexts before recollecting them.
	 */
	clearVotes() {
		for (const [, neurons] of this.activeNeurons)
			for (const [, ageMap] of neurons)
				for (const [, state] of ageMap) {
					state.votes = null;
					state.context = null;
				}
	}

	/**
	 * Set votes and context for a neuron at a specific age
	 */
	setVotes(neuronId, level, age, votes, context) {
		const state = this.activeNeurons.get(level).get(neuronId).get(age);
		state.votes = votes;
		state.context = context;
	}

	/**
	 * Get all inferred neurons
	 */
	getInferredNeurons() {
		return this.inferredNeurons;
	}

	/**
	 * Clear all inferred neurons
	 */
	clearInferredNeurons() {
		this.inferredNeurons = [];
	}

	/**
	 * Save winning inferences to in-memory structures.
	 * @param {Array} inferences - Array of winning inference objects
	 */
	saveInferredNeurons(inferences) {
		this.clearInferredNeurons();
		for (const inference of inferences) this.inferredNeurons.push(inference);
		if (this.debug) console.log(`Saved ${this.inferredNeurons.length} inferences`);
	}

	/**
	 * Verify that none of the deleted patterns are currently active.
	 * @param {Array<Neuron>} deletedPatterns - Patterns that were deleted
	 */
	assertNotActive(deletedPatterns) {
		for (const pattern of deletedPatterns)
			for (const [, neurons] of this.activeNeurons)
				if (neurons.has(pattern.id))
					throw new Error(`BUG: deleting active neuron ${pattern.id}`);
	}
}