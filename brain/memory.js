/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * A dumb data store: stores neuron IDs and states, returns raw data.
 * All neuron-property-aware filtering is done by the caller (Brain).
 */
export class Memory {

	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

		// Active context indexed by age: Array<Map<neuronId, {activatedPattern, votes, context}>>
		// activeNeurons[0] = age 0 (newest), activeNeurons[n] = age n (older)
		// activatedPattern: pattern neuron activated by this neuron, or null
		// votes: array of votes cast by this neuron, or null if hasn't voted yet
		// context: array of context neurons at voting time, or null if hasn't voted yet
		this.activeNeurons = [];

		// Current frame winning inferences: Array<{neuron, strength}>
		this.inferredNeurons = [];

		// carry over the debug flag
		this.debug = debug;
	}

	/**
	 * Reset context (active neurons, inferred neurons, votes)
	 */
	reset() {
		this.activeNeurons = [];
		this.inferredNeurons = [];
	}

	/**
	 * Age all neurons by shifting the age arrays and deactivate aged-out neurons
	 */
	age() {
		if (this.debug) console.log('Aging neurons...');
		this.activeNeurons.unshift(new Map());

		// Deactivate neurons that have aged out of the context window
		if (this.activeNeurons.length > this.contextLength) {
			const removed = this.activeNeurons.pop();
			if (this.debug && removed.size > 0) console.log(`Deactivated ${removed.size} aged-out neurons`);
		}
	}

	/**
	 * Get the number of age slots currently in the sliding window
	 */
	get depth() {
		return this.activeNeurons.length;
	}

	/**
	 * Get neuron IDs and states at a specific age
	 * @returns {Map<number, {activatedPattern, votes, context}>}
	 */
	getNeuronsAtAge(age) {
		return this.activeNeurons[age] ?? new Map();
	}

	/**
	 * Activate a neuron at age 0
	 */
	activateNeuron(neuron, currentFrame) {
		return this.activateNeuronAtAge(neuron, 0, currentFrame);
	}

	/**
	 * Activate a neuron at a specific age (keyed by neuron ID)
	 * @returns {number|null} death frame for pattern neurons, null for sensory
	 */
	activateNeuronAtAge(neuron, age, currentFrame) {
		if (!this.activeNeurons[age]) this.activeNeurons[age] = new Map();
		this.activeNeurons[age].set(neuron.id, { activatedPattern: null, votes: null, context: null });
		return neuron.strengthenActivation(currentFrame);
	}

	/**
	 * Activate a pattern neuron and link it to its parent
	 * @param {Neuron} pattern - The pattern neuron to activate
	 * @param {Neuron} parent - The parent neuron that triggered the pattern
	 * @param {number} age - The age at which to activate
	 * @param {number} currentFrame - Current frame number for lazy decay
	 */
	activatePattern(pattern, parent, age, currentFrame) {
		const deathFrame = this.activateNeuronAtAge(pattern, age, currentFrame);
		const neuronsAtAge = this.activeNeurons[age];
		const state = neuronsAtAge.get(parent.id);
		state.activatedPattern = pattern;
		return deathFrame;
	}

	/**
	 * Clear per-frame saved votes and contexts before recollecting them.
	 */
	clearVotes() {
		for (const neuronsAtAge of this.activeNeurons)
			for (const state of neuronsAtAge.values()) {
				state.votes = null;
				state.context = null;
			}
	}

	/**
	 * Set votes and context for a neuron at a specific age
	 */
	setVotes(neuronId, age, votes, context) {
		const state = this.activeNeurons[age].get(neuronId);
		state.votes = votes;
		state.context = context;
	}

	/**
	 * Get all inferred neurons
	 */
	getInferences() {
		return this.inferredNeurons;
	}

	/**
	 * Clear all inferred neurons
	 */
	clearInferences() {
		this.inferredNeurons = [];
	}

	/**
	 * Save winning inferences to in-memory structures.
	 * @param {Array} inferences - Array of winning inference objects
	 */
	saveInferences(inferences) {
		this.clearInferences();
		for (const inference of inferences) this.inferredNeurons.push(inference);
		if (this.debug) console.log(`Saved ${this.inferredNeurons.length} inferences`);
	}

	/**
	 * Verify that none of the deleted patterns are currently active.
	 * @param {Array<Neuron>} deletedPatterns - Patterns that were deleted
	 */
	assertNotActive(deletedPatterns) {
		for (const pattern of deletedPatterns)
			for (const neuronsAtAge of this.activeNeurons)
				if (neuronsAtAge.has(pattern.id))
					throw new Error(`BUG: deleting active neuron ${pattern.id} (level ${pattern.level})`);
	}
}