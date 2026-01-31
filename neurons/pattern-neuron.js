import { Neuron } from './neuron.js';

/**
 * PatternNeuron - Level > 0 neurons representing learned patterns
 * These have a peak neuron, context (past), and predictions (future)
 */
export class PatternNeuron extends Neuron {

	constructor(level, peak) {
		super(level);

		// Peak neuron this pattern differentiates (pointer to SensoryNeuron or PatternNeuron)
		this.peak = peak;
		this.peakStrength = 1.0;
		peak.incomingCount++; // Peak is referenced by this pattern

		// Context for matching: Map<contextNeuron, Map<age, strength>>
		// Age-first would be: Map<age, Map<contextNeuron, strength>> - but we iterate by neuron more often
		this.past = new Map();

		// Predictions: Map<distance, Map<inferredNeuron, {strength, reward}>>
		// Distance-first indexing (same as connections) for O(1) inference lookup
		this.future = new Map();
	}

	/**
	 * Add or strengthen context neuron at age
	 * @param {Neuron} contextNeuron - Context neuron
	 * @param {number} age - Context age
	 * @param {number} strength - Initial or added strength
	 */
	addContext(contextNeuron, age, strength = 1.0) {
		if (!this.past.has(contextNeuron)) {
			this.past.set(contextNeuron, new Map());
			contextNeuron.incomingCount++;
		}
		const ageMap = this.past.get(contextNeuron);
		if (ageMap.has(age))
			ageMap.set(age, ageMap.get(age) + strength);
		else
			ageMap.set(age, strength);
	}

	/**
	 * Remove context entry for neuron at age
	 * @returns {boolean} True if entry existed and was removed
	 */
	removeContext(contextNeuron, age) {
		const ageMap = this.past.get(contextNeuron);
		if (!ageMap || !ageMap.has(age)) return false;
		ageMap.delete(age);
		if (ageMap.size === 0) {
			this.past.delete(contextNeuron);
			contextNeuron.incomingCount--;
		}
		return true;
	}

	/**
	 * Get or create future prediction at distance to inferred neuron
	 * @param {number} distance - Temporal distance
	 * @param {Neuron} inferredNeuron - Predicted neuron
	 * @returns {{strength: number, reward: number}} Prediction object
	 */
	getOrCreateFuture(distance, inferredNeuron) {
		if (!this.future.has(distance)) this.future.set(distance, new Map());
		const distanceMap = this.future.get(distance);
		if (!distanceMap.has(inferredNeuron)) {
			distanceMap.set(inferredNeuron, { strength: 1, reward: 0 });
			inferredNeuron.incomingCount++;
		}
		return distanceMap.get(inferredNeuron);
	}

	/**
	 * Delete future prediction at distance to inferred neuron
	 * @returns {boolean} True if prediction existed and was deleted
	 */
	deleteFuture(distance, inferredNeuron) {
		const distanceMap = this.future.get(distance);
		if (!distanceMap || !distanceMap.has(inferredNeuron)) return false;
		distanceMap.delete(inferredNeuron);
		inferredNeuron.incomingCount--;
		if (distanceMap.size === 0) this.future.delete(distance);
		return true;
	}

	/**
	 * Check if pattern can be deleted (orphaned)
	 * Pattern neurons can be deleted when they have no references and no content
	 */
	canDelete(brain) {
		return this.incomingCount === 0 &&
			this.past.size === 0 &&
			this.future.size === 0 &&
			!brain.activeNeurons.has(this);
	}

	/**
	 * Clean up all references when deleting this pattern
	 * Must be called before removing from brain.neurons
	 */
	cleanup() {
		// Decrement peak's incoming count
		this.peak.incomingCount--;

		// Decrement all context neurons' incoming counts
		for (const contextNeuron of this.past.keys())
			contextNeuron.incomingCount--;

		// Decrement all inferred neurons' incoming counts
		for (const distanceMap of this.future.values())
			for (const inferredNeuron of distanceMap.keys())
				inferredNeuron.incomingCount--;
	}
}