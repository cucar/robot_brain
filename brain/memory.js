/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * Encapsulates all access to the brain's short-term memory structures.
 */
export class Memory {

	constructor(debug) {

		// memory hyperparameters
		this.contextLength = 20; // number of frames a base neuron stays active

		// Active context indexed by age: Array<Map<Neuron, {activatedPattern, votes, context}>>
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
	 * Get neurons at a specific age
	 * @returns {Map<Neuron, {activatedPattern, votes, context}>}
	 */
	getNeuronsAtAge(age) {
		return this.activeNeurons[age] ?? new Map();
	}

	/**
	 * Activate a neuron at age 0
	 */
	activateNeuron(neuron, currentFrame) {
		this.activateNeuronAtAge(neuron, 0, currentFrame);
	}

	/**
	 * Activate a neuron at a specific age
	 */
	activateNeuronAtAge(neuron, age, currentFrame) {
		if (!this.activeNeurons[age]) this.activeNeurons[age] = new Map();
		this.activeNeurons[age].set(neuron, { activatedPattern: null, votes: null, context: null });
		neuron.strengthenActivation(currentFrame);
	}

	/**
	 * Check if a neuron is active at any age
	 */
	isNeuronActive(neuron) {
		for (const neuronsAtAge of this.activeNeurons)
			if (neuronsAtAge.has(neuron)) return true;
		return false;
	}

	/**
	 * Activate a pattern neuron and link it to its parent
	 * @param {Neuron} pattern - The pattern neuron to activate
	 * @param {Neuron} parent - The parent neuron that triggered the pattern
	 * @param {number} age - The age at which to activate
	 * @param {number} currentFrame - Current frame number for lazy decay
	 */
	activatePattern(pattern, parent, age, currentFrame) {
		this.activateNeuronAtAge(pattern, age, currentFrame);
		const neuronsAtAge = this.activeNeurons[age];
		const state = neuronsAtAge.get(parent);
		state.activatedPattern = pattern;
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
	setVotes(neuron, age, votes, context) {
		const neuronsAtAge = this.activeNeurons[age];
		const state = neuronsAtAge.get(neuron);
		state.votes = votes;
		state.context = context;
	}

	/**
	 * Add an inferred neuron
	 */
	addInference(neuron, strength, reward) {
		this.inferredNeurons.push({ neuron, strength, reward });
	}

	/**
	 * Get all inferred neurons
	 */
	getInferences() {
		return this.inferredNeurons;
	}

	/**
	 * Get inferred actions grouped by channel
	 * @returns {Map<string, Array>} - Map of channel names to array of action data {coordinates, strength, reward}
	 */
	getInferredActions() {
		const channelOutputs = new Map();
		for (const { neuron, strength, reward } of this.inferredNeurons) {
			if (neuron.type !== 'action') continue;
			if (!channelOutputs.has(neuron.channel)) channelOutputs.set(neuron.channel, []);
			channelOutputs.get(neuron.channel).push({ coordinates: neuron.coordinates, strength, reward });
		}
		return channelOutputs;
	}

	/**
	 * Clear all inferred neurons
	 */
	clearInferences() {
		this.inferredNeurons = [];
	}

	/**
	 * Get newly active sensory neurons (age=0, level=0 - events and actions)
	 * @returns {Set} Set of newly active sensory neurons
	 */
	getNewSensoryNeurons() {
		const newActiveNeurons = new Set();
		for (const neuron of this.getNeuronsAtAge(0).keys())
			if (neuron.level === 0) newActiveNeurons.add(neuron);
		return newActiveNeurons;
	}

	/**
	 * Build all contexts for all ages and levels in a single pass.
	 * Returns a map indexed by 'age:level' for O(1) lookup.
	 * @returns {Map<string, Array<{neuron, distance}>>} - Map of 'age:level' to context array
	 */
	getContexts() {
		const contexts = new Map();
		for (let ctxAge = 1; ctxAge < this.activeNeurons.length; ctxAge++)
			for (const neuron of (this.activeNeurons[ctxAge] ?? new Map()).keys()) {

				// actions cannot be in contexts
				if (neuron.level === 0 && neuron.type === 'action') continue;

				// Add this neuron to context for all ages before it
				for (let age = 0; age < ctxAge; age++) {
					const key = `${age}:${neuron.level}`;
					if (!contexts.has(key)) contexts.set(key, []);
					contexts.get(key).push({ neuron, distance: ctxAge - age });
				}
			}
		return contexts;
	}

	/**
	 * returns newly activated non-action neurons at age=0 (recognizers), filtered by level
	 */
	getRecognizerNeurons(level) {
		const newNeurons = [];
		for (const neuron of this.activeNeurons[0].keys()) {
			if (neuron.level === 0 && neuron.type === 'action') continue; // actions cannot be peaks
			if (neuron.level !== level) continue;
			newNeurons.push(neuron);
		}
		return newNeurons;
	}

	/**
	 * Get neurons that can vote (within context window)
	 * @returns {Array<{neuron, age, state}>}
	 */
	getVotingNeurons() {
		const result = [];
		// Iterate through all active neurons except the oldest (which don't have distance+1 connections)
		for (let age = 0; age < this.activeNeurons.length - 1; age++)
			for (const [voter, state] of this.activeNeurons[age])
				if (voter.level > 0 || voter.type !== 'action') // action neurons cannot vote - their inferences are too erratic
					result.push({ voter, age, state });
		return result;
	}

	/**
	 * Get context neurons (age > 0, age < contextLength), optionally filtered by level
	 * @param {number} [level] - Optional level to filter by
	 * @returns {Array<{neuron, age}>}
	 */
	getContextNeurons(level) {
		const filterByLevel = level !== undefined;
		const result = [];
		for (let age = 1; age < this.activeNeurons.length; age++)
			for (const neuron of this.activeNeurons[age].keys()) {
				if (neuron.level === 0 && neuron.type === 'action') continue; // actions cannot be in contexts
				if (filterByLevel && neuron.level !== level) continue;
				result.push({ neuron, age });
			}
		return result;
	}

	/**
	 * get neurons that voted with their context for pattern learning.
	 * Returns neurons that voted in the previous frame with their pre-saved context.
	 * @returns {Array<{neuron, age, votes, context}>}
	 */
	getVotersWithContext() {
		const result = [];
		for (let age = 1; age < this.activeNeurons.length; age++)
			for (const [neuron, state] of (this.activeNeurons[age] ?? new Map()))
				if (state.votes && state.votes.length > 0)
					result.push({ neuron, age, votes: state.votes, context: state.context });
		return result;
	}

	/**
	 * Save winning inferences to in-memory structures.
	 * @param {Array} inferences - Array of winning inference objects
	 */
	saveInferences(inferences) {
		this.clearInferences();
		for (const inf of inferences) this.addInference(inf.neuron, inf.strength, inf.reward);
		if (this.debug) console.log(`Saved ${this.inferredNeurons.length} inferences`);
	}

	/**
	 * Clean up a deleted neuron from all active contexts.
	 * This prevents learnNewPattern from creating patterns with zombie references.
	 * @param {Neuron} deletedNeuron - The neuron being deleted
	 */
	cleanupDeletedNeuron(deletedNeuron) {
		for (const neuronsAtAge of this.activeNeurons)
			for (const state of neuronsAtAge.values())
				if (state.context)
					state.context = state.context.filter(entry => entry.neuron !== deletedNeuron);
	}
}