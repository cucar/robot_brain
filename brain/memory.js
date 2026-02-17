import { Context } from './context.js';

/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * Encapsulates all access to the brain's short-term memory structures.
 */
export class Memory {

	constructor(debug) {

		// memory hyperparameters
		this.contextLength = 5; // number of frames a base neuron stays active

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
	 * Reset context (active neurons, inferred neurons)
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
	activateNeuron(neuron) {
		this.activateNeuronAtAge(neuron, 0);
	}

	/**
	 * Activate a neuron at a specific age
	 */
	activateNeuronAtAge(neuron, age) {
		if (!this.activeNeurons[age]) this.activeNeurons[age] = new Map();
		this.activeNeurons[age].set(neuron, { activatedPattern: null, votes: null, context: null });
		neuron.strengthenActivation();
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
	 * Activate a pattern neuron and link it to its peak
	 * @param {Neuron} pattern - The pattern neuron to activate
	 * @param {Neuron} peak - The peak neuron that triggered the pattern
	 * @param {number} age - The age at which to activate
	 */
	activatePattern(pattern, peak, age) {
		this.activateNeuronAtAge(pattern, age);
		const neuronsAtAge = this.activeNeurons[age];
		const state = neuronsAtAge.get(peak);
		state.activatedPattern = pattern;
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
	 * returns context array for a given age (older neurons relative to this age)
	 * @param {number} age - The age to build context for
	 * @param {number} level - The level to filter context neurons by (same level as peak neuron)
	 * @returns {Array<{neuron, distance}>}
	 */
	getContextForAge(age, level) {
		const result = [];
		for (let ctxAge = age + 1; ctxAge < this.activeNeurons.length; ctxAge++)
			for (const neuron of (this.activeNeurons[ctxAge] ?? new Map()).keys())
				if (neuron.level === level && (neuron.level > 0 || neuron.type !== 'action'))
					result.push({ neuron: neuron, distance: ctxAge - age });
		return result;
	}

	/**
	 * Get peaks (age=0) and context (age>0) neurons, optionally filtered by level
	 * @param {number} [level] - Optional level to filter by
	 * @returns {{peaks: Array<Neuron>, context: Context}}
	 */
	getPeaksAndContext(level) {
		const filterByLevel = level !== undefined;
		const peaks = [];
		const context = new Context();
		for (let age = 0; age < this.activeNeurons.length; age++)
			for (const neuron of this.activeNeurons[age].keys()) {
				if (neuron.level === 0 && neuron.type === 'action') continue; // actions cannot be in contexts and cannot be peaks
				if (filterByLevel && neuron.level !== level) continue;
				if (age === 0) peaks.push(neuron);
				else context.addNeuron(neuron, age, 1);
			}
		return { peaks, context };
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
	 * Get context neurons (age > 0, age < contextLength) for connection learning
	 * @returns {Array<{neuron, age}>}
	 */
	getContextNeurons() {
		const result = [];
		for (let age = 1; age < this.activeNeurons.length; age++)
			for (const neuron of this.activeNeurons[age].keys())
				if (neuron.level > 0 || neuron.type !== 'action') // action neurons cannot learn connections - they are predicted by events
					result.push({ neuron, age });
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
			for (const [neuron, state] of this.activeNeurons[age])
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