import { Context } from './context.js';

/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 * Encapsulates all access to the brain's short-term memory structures.
 */
export class Memory {

	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

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

		// Set of channel names that have action sequence learning disabled (populated by brain after init)
		this.noActionSequenceChannels = new Set();
	}

	/**
	 * Set the channels that have action sequence learning disabled
	 * @param {Set<string>} channels - Set of channel names
	 */
	setNoActionSequenceChannels(channels) {
		this.noActionSequenceChannels = channels;
	}

	/**
	 * Check if a neuron should be skipped from learning context (action neuron in a channel without action sequences)
	 */
	skipActionNeuron(neuron) {
		return neuron.level === 0 && neuron.type === 'action' && this.noActionSequenceChannels.has(neuron.channel);
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
		return this.activateNeuronAtAge(neuron, 0, currentFrame);
	}

	/**
	 * Activate a neuron at a specific age
	 * @returns {number|null} death frame for pattern neurons, null for sensory
	 */
	activateNeuronAtAge(neuron, age, currentFrame) {
		if (!this.activeNeurons[age]) this.activeNeurons[age] = new Map();
		this.activeNeurons[age].set(neuron, { activatedPattern: null, votes: null, context: null });
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
		const state = neuronsAtAge.get(parent);
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
	setVotes(neuron, age, votes, context) {
		const neuronsAtAge = this.activeNeurons[age];
		const state = neuronsAtAge.get(neuron);
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
	 * Get all active sensory neurons indexed by age.
	 * @returns {Array<Array<Neuron>>} Array where index is age, value is array of sensory neurons at that age
	 */
	getSensoryNeurons() {
		const result = [];
		for (let age = 0; age < this.activeNeurons.length; age++) {
			const neurons = [];
			for (const neuron of this.activeNeurons[age].keys())
				if (neuron.level === 0) neurons.push(neuron);
			result.push(neurons);
		}
		return result;
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

				// do not allow actions in contexts unless action sequences are enabled for their channel
				if (this.skipActionNeuron(neuron)) continue;

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
	 * Returns recognizer neurons with their per-age contexts for pattern matching.
	 * Each recognizer at age N gets context from neurons at ages > N at the same level.
	 * Contexts are shared across recognizers at the same age for efficiency.
	 * @returns {Array<{neuron, age, context: Context}>}
	 */
	getRecognizersWithContext(level) {

		// build context per age at this level (same fan-out pattern as getContexts)
		const contextByAge = new Map();
		for (let ctxAge = 1; ctxAge < this.activeNeurons.length; ctxAge++)
			for (const neuron of this.activeNeurons[ctxAge].keys()) {
				if (this.skipActionNeuron(neuron)) continue;
				if (neuron.level !== level) continue;
				for (let age = 0; age < ctxAge; age++) {
					if (!contextByAge.has(age)) contextByAge.set(age, new Context());
					contextByAge.get(age).addNeuron(neuron, ctxAge - age, 1);
				}
			}

		// collect recognizers and pair with pre-built contexts
		// skip if no context exists or if neuron already activated a pattern (recognition or error correction)
		const result = [];
		for (let age = 0; age < this.activeNeurons.length; age++) {
			const context = contextByAge.get(age);
			if (!context) continue;
			for (const [neuron, state] of this.activeNeurons[age]) {
				if (state.activatedPattern !== null) continue;
				if (this.skipActionNeuron(neuron)) continue;
				if (neuron.level !== level) continue;
				result.push({ neuron, age, context });
			}
		}
		return result;
	}

	/**
	 * Get neurons that can vote (within context window)
	 * @returns {Array<{neuron, age, state}>}
	 */
	getVotingNeurons() {
		const result = [];
		// Iterate through all active neurons except the oldest (which don't have distance+1 connections)
		for (let age = 0; age < this.activeNeurons.length - 1; age++)
			for (const [voter, state] of this.activeNeurons[age]) {

				// do not allow action neurons to vote unless action sequences are enabled for their channel
				if (this.skipActionNeuron(voter)) continue;

				// add the neuron to the list
				result.push({ voter, age, state });
			}
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

				// do not allow actions in contexts unless action sequences are enabled for their channel
				if (this.skipActionNeuron(neuron)) continue;

				// filter by level if requested and add to list
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
				if (neuronsAtAge.has(pattern))
					throw new Error(`BUG: deleting active neuron ${pattern.id} (level ${pattern.level})`);
	}
}