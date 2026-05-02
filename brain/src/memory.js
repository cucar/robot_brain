/**
 * Memory - manages the temporal sliding window of active and inferred neurons.
 *
 * Internally everything is keyed by **activation frame number**, not by age.
 * Age is a derived quantity (`frameNumber - activationFrame`) computed on read.
 * This means `age()` doesn't have to migrate any per-neuron state Maps each frame —
 * it just bumps the frame counter and evicts whatever frame fell off the back of
 * the window. Public API (depth, getLevelNeurons, getLevelAges, …) still speaks in
 * ages, so callers (brain, thalamus) are unchanged.
 */
export class Memory {

	/**
	 * @param {boolean} debug
	 * @param {number} contextLength - number of frames a base neuron stays active
	 */
	constructor(debug, contextLength) {

		// number of frames a base neuron stays active
		this.contextLength = contextLength;

		// current frame counter, advanced once per age() call. Activations are stored
		// keyed by the frame they happened on; age = frameNumber - activationFrame.
		this.frameNumber = 0;

		// active neuron states by activation frame - Map<neuronId, Map<frame, state>>.
		// state properties:
		//   context: array of context neurons at voting time
		//   votes: array of votes cast by this neuron
		//   activatedPatternId: ID of pattern neuron activated by this neuron (recognized)
		// State objects are created at activation and never copied — external mutations
		// (thalamus.applyFrameResults sets votes/context/threshold) stay attached for the
		// life of the entry without per-frame rebuild.
		this.neuronStates = new Map();

		// indexes for fast age/level queries — both keyed by activation frame so eviction
		// is a single delete-by-frame instead of a per-neuron shift.
		this.ageIndex = new Map(); // Map<frame, Set<neuronId>>
		this.levelIndex = new Map(); // Map<level, Map<frame, Set<neuronId>>>

		// Current frame winning inferences: Array<{neuron, strength}>
		this.inferredNeurons = [];

		// carry over the debug flag
		this.debug = debug;
	}

	/**
	 * Reset context (active neurons, inferred neurons, votes)
	 */
	reset() {
		this.frameNumber = 0;
		this.ageIndex = new Map();
		this.levelIndex = new Map();
		this.neuronStates = new Map();
		this.inferredNeurons = [];
	}

	/**
	 * Advance the sliding window to the given frame. Memory acts as a slave clock —
	 * the brain owns the frame counter and passes it in so the two never drift.
	 * With frame-keyed storage this is just a counter sync plus eviction of whatever
	 * activation frame fell off the back — no per-neuron Map rebuild, regardless of
	 * how many neurons are active.
	 * @param {number} frameNumber - The brain's frame number for this frame
	 */
	age(frameNumber) {
		if (this.debug) console.log('Aging neurons...');

		// sync to the brain's frame number
		this.frameNumber = frameNumber;

		// nothing to evict until the window is full (frameNumber > contextLength)
		if (this.frameNumber <= this.contextLength) return;

		// we will deactivate the oldest age in the context
		const evictedFrame = this.frameNumber - this.contextLength;
		const evictedNeuronIds = this.ageIndex.get(evictedFrame);
		if (!evictedNeuronIds) return; // if there are no neurons in the evicted frame, nothing to do

		// drop the evicted frame from every index in one pass
		for (const neuronId of evictedNeuronIds) {
			const states = this.neuronStates.get(neuronId);
			states.delete(evictedFrame);
			if (states.size === 0) this.neuronStates.delete(neuronId); // deactivate neuron if all ages are inactive
		}
		this.ageIndex.delete(evictedFrame);
		for (const levelFrames of this.levelIndex.values()) levelFrames.delete(evictedFrame);

		if (this.debug && evictedNeuronIds.size > 0) console.log(`Deactivated ${evictedNeuronIds.size} aged-out neurons`);
	}

	/**
	 * Number of age slots currently in the sliding window
	 */
	get depth() {
		// grows from 0 up to contextLength as frames roll in, then stays pinned
		return Math.min(this.frameNumber, this.contextLength);
	}

	/**
	 * Get the Set of neuron IDs at a specific age (insertion-order preserved).
	 */
	getNeuronIdsAtAge(age) {
		return this.ageIndex.get(this.frameNumber - age) ?? new Set();
	}

	/**
	 * Get the per-age Sets of neuron IDs for a level (index = age), in age-ascending order.
	 * Returns an empty array if the level has no active neurons.
	 * Built fresh on each call by walking ages 0..depth-1 against the frame-keyed level index.
	 */
	getLevelAges(level) {
		const levelFrames = this.levelIndex.get(level);
		if (!levelFrames) return [];
		const result = [];
		for (let age = 0; age < this.depth; age++)
			result.push(levelFrames.get(this.frameNumber - age) ?? new Set());
		return result;
	}

	/**
	 * Returns the active neurons at a given level with their age-states.
	 * Shape: Map<neuronId, Map<age, state>>. The inner map is populated in
	 * age-ascending order (Map preserves insertion order), so iterating it
	 * yields ages ascending.
	 * thalamus.processLevel depends on this.
	 */
	getLevelNeurons(level) {
		const neurons = new Map();
		const levelFrames = this.levelIndex.get(level);
		if (!levelFrames) return neurons;

		// walk ages ascending so the inner Map's insertion order matches
		for (let age = 0; age < this.depth; age++) {
			const frame = this.frameNumber - age;
			const neuronIds = levelFrames.get(frame);
			if (!neuronIds) continue;
			for (const neuronId of neuronIds) {
				let ageStates = neurons.get(neuronId);
				if (!ageStates) {
					ageStates = new Map();
					neurons.set(neuronId, ageStates);
				}
				ageStates.set(age, this.neuronStates.get(neuronId).get(frame));
			}
		}
		return neurons;
	}

	/**
	 * Activate a neuron at age 0
	 */
	activateNeuron(neuronId, level) {
		this.activateNeuronAtAge(neuronId, 0, level);
	}

	/**
	 * Activate a neuron at a specific age (keyed by neuron ID)
	 * Internally stored keyed by activation frame = frameNumber - age
	 * so the entry naturally moves to age+1 next frame without any rewrite.
	 */
	activateNeuronAtAge(neuronId, age, level) {
		const frame = this.frameNumber - age;

		// add the neuron to the age index
		let ageSet = this.ageIndex.get(frame);
		if (!ageSet) { ageSet = new Set(); this.ageIndex.set(frame, ageSet); }
		ageSet.add(neuronId);

		// add the neuron to the level index
		let levelFrames = this.levelIndex.get(level);
		if (!levelFrames) {
			levelFrames = new Map();
			this.levelIndex.set(level, levelFrames);
		}
		let levelSet = levelFrames.get(frame);
		if (!levelSet) {
			levelSet = new Set();
			levelFrames.set(frame, levelSet);
		}
		levelSet.add(neuronId);

		// add the active neuron to the neuron states with a new state
		let states = this.neuronStates.get(neuronId);
		if (!states) {
			states = new Map();
			this.neuronStates.set(neuronId, states);
		}
		states.set(frame, { activatedPatternId: null, votes: null, context: null });
	}

	/**
	 * Activate a pattern neuron and link it to its parent
	 */
	activatePattern(patternId, patternLevel, parentId, age) {
		this.activateNeuronAtAge(patternId, age, patternLevel);
		this.neuronStates.get(parentId).get(this.frameNumber - age).activatedPatternId = patternId;
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
	 * Verify that none of the deleted pattern ids are currently active.
	 * @param {Array<number>} deletedPatternIds
	 */
	assertNotActive(deletedPatternIds) {
		for (const patternId of deletedPatternIds)
			if (this.neuronStates.has(patternId))
				throw new Error(`BUG: deleting active neuron ${patternId}`);
	}

	/**
	 * Get the maximum active level from the level index
	 * @returns {number} The highest level that has any active neurons
	 */
	getMaxActiveLevel() {
		return this.levelIndex.size;
	}
}