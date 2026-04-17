/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * A dumb data store: stores neuron IDs and states, returns raw data.
 * All neuron-property-aware filtering is done by the caller (Brain).
 *
 * Internal layout:
 *   ageIndex:     Array<Set<neuronId>>                    — neurons at each age slot (order = activation order)
 *   levelIndex:   Map<level, Array<Set<neuronId>>>        — per-level, per-age neuron sets (age arrays kept in sync with ageIndex length)
 *   neuronStates: Map<neuronId, Map<age, state>>          — per-neuron state, keyed by current age
 *
 * Aging shifts state keys by +1 and prepends empty sets. This is aggressive but keeps
 * the mental model simple; if it becomes a hotspot we can switch to a birth-frame scheme.
 */
export class Memory {

	/**
	 * @param {boolean} debug
	 * @param {number} contextLength - number of frames a base neuron stays active
	 */
	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

		// Active indexed context
		// activatedPatternId: ID of pattern neuron activated by this neuron, or null
		// votes: array of votes cast by this neuron, or null if hasn't voted yet
		// context: array of context neurons at voting time, or null if hasn't voted yet
		this.ageIndex = [];
		this.levelIndex = new Map();
		this.neuronStates = new Map();

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

		this.ageIndex.unshift(new Set());
		for (const arr of this.levelIndex.values()) arr.unshift(new Set());

		// Shift every neuron's state age keys by +1 (preserves insertion order within inner map)
		for (const [id, inner] of this.neuronStates) {
			const shifted = new Map();
			for (const [a, s] of inner) shifted.set(a + 1, s);
			this.neuronStates.set(id, shifted);
		}

		// Deactivate neurons that have aged out of the context window
		if (this.ageIndex.length > this.contextLength) {
			const evictedAge = this.ageIndex.length - 1;
			const evicted = this.ageIndex.pop();
			for (const arr of this.levelIndex.values()) arr.pop();
			for (const id of evicted) {
				const inner = this.neuronStates.get(id);
				inner.delete(evictedAge);
				if (inner.size === 0) this.neuronStates.delete(id);
			}
			if (this.debug && evicted.size > 0) console.log(`Deactivated ${evicted.size} aged-out neurons`);
		}
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

		while (this.ageIndex.length <= age) this.ageIndex.push(new Set());
		this.ageIndex[age].add(neuronId);

		let levelArr = this.levelIndex.get(level);
		if (!levelArr) {
			levelArr = [];
			this.levelIndex.set(level, levelArr);
		}
		while (levelArr.length < this.ageIndex.length) levelArr.push(new Set());
		levelArr[age].add(neuronId);

		let inner = this.neuronStates.get(neuronId);
		if (!inner) {
			inner = new Map();
			this.neuronStates.set(neuronId, inner);
		}
		inner.set(age, { activatedPatternId: null, votes: null, context: null });
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
}