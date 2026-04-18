/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 */
export class Memory {

	/**
	 * @param {boolean} debug
	 * @param {number} contextLength - number of frames a base neuron stays active
	 */
	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

		// active neuron states by age - Map<neuronId, Map<age, state>> - state properties:
		// context: array of context neurons at voting time
		// votes: array of votes cast by this neuron
		// activatedPatternId: ID of pattern neuron activated by this neuron (recognized)
		this.neuronStates = new Map();

		// indexes to be able to retrieve neurons by age and level faster
		this.ageIndex = []; // Array<Set<neuronId>> - neurons at each age slot (order = activation order)
		this.levelIndex = new Map(); // Map<level, Array<Set<neuronId>>> - per-level, per-age neuron sets

		// Current frame winning inferences: Array<{neuron, strength}>
		this.inferredNeurons = [];

		// carry over the debug flag
		this.debug = debug;
	}

	/**
	 * Reset context (active neurons, inferred neurons, votes)
	 */
	reset() {
		this.ageIndex = [];
		this.levelIndex = new Map();
		this.neuronStates = new Map();
		this.inferredNeurons = [];
	}

	/**
	 * Age all neurons by shifting age keys and deactivate aged-out neurons
	 */
	age() {
		if (this.debug) console.log('Aging neurons...');

		// age index update for active neurons: add age=0 set and shift everything else
		this.ageIndex.unshift(new Set());

		// do the same for the level index - ages in each level
		for (const arr of this.levelIndex.values()) arr.unshift(new Set());

		// ages also exist in neuron states - shift every neuron's state age keys by +1 (preserves insertion order)
		for (const [neuronId, oldAges] of this.neuronStates) {
			const newAges = new Map();
			for (const [age, state] of oldAges) newAges.set(age + 1, state);
			this.neuronStates.set(neuronId, newAges);
		}

		// deactivate neurons that have aged out of the context window
		if (this.ageIndex.length > this.contextLength) this.deactivateOldNeurons();
	}

	/**
	 * deactivates aged-out neurons
	 */
	deactivateOldNeurons() {

		// we will deactivate the oldest age in the context
		const evictedAge = this.ageIndex.length - 1;

		// get rid of the oldest neurons in the age index, while retrieving their ids
		const evictedNeuronIds = this.ageIndex.pop();

		// get rid of the oldest neurons in the level index, for each age within them
		for (const levelAges of this.levelIndex.values()) levelAges.pop();

		// get rid of the oldest neuron ages in the neuron states themselves
		for (const id of evictedNeuronIds) {
			const ages = this.neuronStates.get(id);
			ages.delete(evictedAge);
			if (ages.size === 0) this.neuronStates.delete(id); // deactivate neuron if all ages are inactive
		}

		if (this.debug && evictedNeuronIds.size > 0) console.log(`Deactivated ${evictedNeuronIds.size} aged-out neurons`);
	}

	/**
	 * Number of age slots currently in the sliding window
	 */
	get depth() {
		return this.ageIndex.length;
	}

	/**
	 * Get the Set of neuron IDs at a specific age (insertion-order preserved).
	 */
	getNeuronIdsAtAge(age) {
		return this.ageIndex[age] ?? new Set();
	}

	/**
	 * Get the per-age Sets of neuron IDs for a level (index = age), in age-ascending order.
	 * Returns an empty array if the level has no active neurons.
	 */
	getLevelAges(level) {
		return this.levelIndex.get(level) ?? [];
	}

	/**
	 * Get the state for a neuron at a specific age.
	 */
	getState(neuronId, age) {
		return this.neuronStates.get(neuronId).get(age);
	}

	/**
	 * Activate a neuron at age 0
	 */
	activateNeuron(neuronId, level) {
		this.activateNeuronAtAge(neuronId, 0, level);
	}

	/**
	 * Activate a neuron at a specific age (keyed by neuron ID)
	 */
	activateNeuronAtAge(neuronId, age, level) {

		// add the neuron to the age index
		while (this.ageIndex.length <= age) this.ageIndex.push(new Set());
		this.ageIndex[age].add(neuronId);

		// add the neuron to the level index
		let levelAges = this.levelIndex.get(level);
		if (!levelAges) {
			levelAges = [];
			this.levelIndex.set(level, levelAges);
		}
		while (levelAges.length < this.ageIndex.length) levelAges.push(new Set());
		levelAges[age].add(neuronId);

		// add the active neuron to the neuron states with a new state
		let ages = this.neuronStates.get(neuronId);
		if (!ages) {
			ages = new Map();
			this.neuronStates.set(neuronId, ages);
		}
		ages.set(age, { activatedPatternId: null, votes: null, context: null });
	}

	/**
	 * Activate a pattern neuron and link it to its parent
	 */
	activatePattern(patternId, patternLevel, parentId, age) {
		this.activateNeuronAtAge(patternId, age, patternLevel);
		this.neuronStates.get(parentId).get(age).activatedPatternId = patternId;
	}

	/**
	 * Clear per-frame saved votes and contexts before recollecting them.
	 */
	clearVotes() {
		for (const inner of this.neuronStates.values())
			for (const state of inner.values()) {
				state.votes = null;
				state.context = null;
			}
	}

	/**
	 * Set votes and context for a neuron at a specific age
	 */
	setVotes(neuronId, age, votes, context) {
		const state = this.neuronStates.get(neuronId).get(age);
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
	 */
	saveInferredNeurons(inferences) {
		this.clearInferredNeurons();
		for (const inference of inferences) this.inferredNeurons.push(inference);
		if (this.debug) console.log(`Saved ${this.inferredNeurons.length} inferences`);
	}

	/**
	 * Verify that none of the deleted patterns are currently active.
	 */
	assertNotActive(deletedPatterns) {
		for (const pattern of deletedPatterns)
			if (this.neuronStates.has(pattern.id))
				throw new Error(`BUG: deleting active neuron ${pattern.id}`);
	}

	/**
	 * Get the maximum active level from the level index
	 * @returns {number} The highest level that has any active neurons
	 */
	getMaxActiveLevel() {
		return this.levelIndex.size;
	}
}