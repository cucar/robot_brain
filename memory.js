import { Context } from './neurons/context.js';

/**
 * Memory - manages the temporal sliding window of active, inferred, and inferring neurons.
 * Encapsulates all access to the brain's short-term memory structures.
 */
export class Memory {

	constructor(debug) {

		// memory hyperparameters
		this.contextLength = 5; // number of frames a base neuron stays active

		// Active context indexed by age: Array<Map<Neuron, pattern|null>>
		// activeNeurons[0] = age 0 (newest), activeNeurons[n] = age n (older)
		// pattern is the pattern neuron activated at that age, or null if no pattern
		this.activeNeurons = [];
		
		// Current frame winning inferences: neuronId -> {strength}
		this.inferredNeurons = new Map();
		
		// Inferring neurons indexed by age (winners only): Array<Map<Neuron, Array<{toNeuron, strength, reward}>>>
		// inferringNeurons[age] = Map of neurons to their votes at that age
		this.inferringNeurons = [];

		// carry over the debug flag
		this.debug = debug;
	}

	/**
	 * Reset context (active neurons, inferred neurons, inferring neurons)
	 */
	reset() {
		this.activeNeurons = [];
		this.inferredNeurons = new Map();
		this.inferringNeurons = [];
	}

	/**
	 * Age all neurons by shifting the age arrays
	 */
	age() {
		if (this.debug) console.log('Aging neurons...');
		this.activeNeurons.unshift(new Map());
		this.inferringNeurons.unshift(new Map());
	}

	/**
	 * Deactivate neurons that have aged out of the context window
	 */
	deactivateOld() {
		if (this.activeNeurons.length <= this.contextLength) return;
		this.inferringNeurons.pop();
		const removed = this.activeNeurons.pop();
		if (this.debug && removed.size > 0) console.log(`Deactivated ${removed.size} aged-out neurons`);
	}

	/**
	 * Get neurons at a specific age
	 * @returns {Map<Neuron, pattern|null>} Map of neurons to their activated pattern (or null)
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
		this.activeNeurons[age].set(neuron, null);
	}

	/**
	 * Check if a neuron is active at any age
	 */
	isNeuronActive(neuron) {
		for (const neurons of this.activeNeurons)
			if (neurons.has(neuron)) return true;
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
		if (neuronsAtAge?.has(peak)) neuronsAtAge.set(peak, pattern);
		pattern.strengthenPeak();
	}

	/**
	 * Add a vote from an inferring neuron at a specific age
	 */
	addVote(neuron, age, votes) {
		if (!this.inferringNeurons[age]) this.inferringNeurons[age] = new Map();
		this.inferringNeurons[age].set(neuron, votes);
	}

	/**
	 * Clear all inferring neurons
	 */
	clearInferringNeurons() {
		this.inferringNeurons = [];
	}

	/**
	 * Add an inferred neuron
	 */
	addInference(neuronId, strength) {
		this.inferredNeurons.set(neuronId, { strength });
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
		this.inferredNeurons.clear();
	}

	/**
	 * Get context neurons (age > 0, age < contextLength) for connection learning
	 * @returns {Array<{neuron, age}>}
	 */
	getContextNeurons() {
		const result = [];
		for (let age = 1; age < this.activeNeurons.length && age < this.contextLength; age++)
			for (const neuron of (this.activeNeurons[age] ?? new Map()).keys())
				result.push({ neuron, age });
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

		for (let age = 0; age < this.activeNeurons.length; age++) {
			const neurons = this.activeNeurons[age];
			if (!neurons) continue;
			for (const neuron of neurons.keys()) {
				if (filterByLevel && neuron.level !== level) continue;
				if (age === 0) peaks.push(neuron);
				else if (age < this.contextLength) context.add(neuron, age, 1, false);
			}
		}
		return { peaks, context };
	}

	/**
	 * Get neurons that can vote (age < contextLength - 1, not suppressed by pattern)
	 * @returns {Array<{neuron, age, activatedPattern}>}
	 */
	getVotingNeurons() {
		const result = [];
		for (let age = 0; age < this.activeNeurons.length && age < this.contextLength; age++)
			for (const [neuron, activatedPattern] of (this.activeNeurons[age] ?? new Map()))
				result.push({ neuron, age, activatedPattern });
		return result;
	}

	/**
	 * Get inferring neurons with their context for pattern learning
	 * @returns {Array<{neuron, age, votes, context}>}
	 */
	getInferringNeuronsWithContext() {
		const result = [];
		for (let age = 1; age < this.inferringNeurons.length; age++) {
			const votesAtAge = this.inferringNeurons[age];
			if (!votesAtAge || votesAtAge.size === 0) continue;

			// Build context array with distances relative to this age
			const context = [];
			for (let ctxAge = age + 1; ctxAge < this.activeNeurons.length; ctxAge++) {
				const ctxNeurons = this.activeNeurons[ctxAge];
				if (!ctxNeurons) continue;
				const distance = ctxAge - age;
				if (distance < this.contextLength)
					for (const ctxNeuron of ctxNeurons.keys())
						if (ctxNeuron.level > 0 || ctxNeuron.type !== 'action')
							context.push({ neuron: ctxNeuron, distance });
			}

			for (const [neuron, votes] of votesAtAge)
				result.push({ neuron, age, votes, context });
		}
		return result;
	}

	/**
	 * Filter inferring neurons to only keep votes that led to winners
	 * @param {Set} winnerIds - Set of winning neuron IDs
	 */
	filterInferringByWinners(winnerIds) {
		for (let age = 0; age < this.inferringNeurons.length; age++) {
			const ageMap = this.inferringNeurons[age];
			if (!ageMap) continue;

			const toDelete = [];
			for (const [neuron, votes] of ageMap) {
				const winningVotes = votes.filter(v => winnerIds.has(v.toNeuron.id));
				if (winningVotes.length === 0)
					toDelete.push(neuron);
				else
					ageMap.set(neuron, winningVotes);
			}
			for (const neuron of toDelete) ageMap.delete(neuron);
		}
	}

	/**
	 * Save winning inferences to in-memory structures.
	 * Only winners are saved - losers are discarded.
	 * Also filters inferringNeurons to only keep votes that led to winners.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	saveInferences(inferences) {

		// Get set of winning neuron IDs
		const winnerIds = new Set();
		for (const inf of inferences)
			if (inf.isWinner)
				winnerIds.add(inf.neuron_id);

		// Save only winners to inferredNeurons
		this.clearInferences();
		for (const inf of inferences)
			if (inf.isWinner)
				this.addInference(inf.neuron_id, inf.strength);

		// Filter inferringNeurons to only keep votes that led to winners
		this.filterInferringByWinners(winnerIds);

		if (this.debug) console.log(`Saved ${winnerIds.size} winning inferences`);
	}
}