import { Neuron } from './neuron.js';

/**
 * SensoryNeuron - Level 0 neurons representing sensory inputs and motor outputs
 * These have coordinates in dimension space and connections to other neurons
 */
export class SensoryNeuron extends Neuron {

	constructor(channel, type, coordinates) {
		super(0); // level 0

		this.channel = channel; // channel name (string)
		this.type = type; // 'event' or 'action'
		this.coordinates = coordinates; // { dimName: value, ... }

		// Outgoing connections: Map<distance, Map<toNeuron, {strength, reward}>>
		// Distance-first indexing for O(1) inference lookup
		// Only event neurons have outgoing connections (events predict events/actions)
		this.connections = new Map();
	}

	/**
	 * Create value key for neuron lookup
	 * @param {Object} coordinates - { dimName: value, ... }
	 * @returns {string} JSON string key for Map lookup
	 */
	static makeValueKey(coordinates) {
		const sorted = Object.keys(coordinates).sort(); // Sort keys for consistent ordering
		const obj = {};
		for (const k of sorted) obj[k] = coordinates[k];
		return JSON.stringify(obj);
	}

	/**
	 * Get value key for this neuron
	 */
	get valueKey() {
		return SensoryNeuron.makeValueKey(this.coordinates);
	}

	/**
	 * Get or create connection at distance to target neuron
	 * @param {number} distance - Temporal distance
	 * @param {Neuron} toNeuron - Target neuron
	 * @returns {{strength: number, reward: number}} Connection object
	 */
	getOrCreateConnection(distance, toNeuron) {
		if (!this.connections.has(distance)) this.connections.set(distance, new Map());
		const distanceMap = this.connections.get(distance);
		if (!distanceMap.has(toNeuron)) {
			distanceMap.set(toNeuron, { strength: 1, reward: 0 });
			toNeuron.incomingCount++;
		}
		return distanceMap.get(toNeuron);
	}

	/**
	 * Delete connection at distance to target neuron
	 * @param {number} distance - Temporal distance
	 * @param {Neuron} toNeuron - Target neuron
	 * @returns {boolean} True if connection existed and was deleted
	 */
	deleteConnection(distance, toNeuron) {
		const distanceMap = this.connections.get(distance);
		if (!distanceMap || !distanceMap.has(toNeuron)) return false;
		distanceMap.delete(toNeuron);
		toNeuron.incomingCount--;
		if (distanceMap.size === 0) this.connections.delete(distance);
		return true;
	}

	/**
	 * Check if this neuron has any outgoing connections
	 */
	hasOutgoingConnections() {
		return this.connections.size > 0;
	}

	/**
	 * Override canDelete - sensory neurons are never deleted (they have coordinates)
	 */
	canDelete() {
		return false; // Sensory neurons persist forever
	}

	/**
	 * Create/strengthen incoming connections from context neurons.
	 * Called when this neuron is newly activated (age=0).
	 * @param {Array<{neuron: SensoryNeuron, age: number}>} contextNeurons - Active event neurons at age > 0
	 */
	reinforceConnections(contextNeurons) {
		for (const { neuron: fromNeuron, age: distance } of contextNeurons) {
			const conn = fromNeuron.getOrCreateConnection(distance, this);
			conn.strength = Math.min(Neuron.maxStrength, conn.strength + 1);
		}
	}

	/**
	 * Apply reward to incoming connections from context neurons.
	 * Called when this neuron is a newly activated action with a reward.
	 * @param {Array<{neuron: SensoryNeuron, age: number}>} contextNeurons - Active event neurons at age > 0
	 * @param {number} reward - Reward value to apply
	 */
	applyReward(contextNeurons, reward) {
		const smoothing = Neuron.rewardSmoothing;
		for (const { neuron: fromNeuron, age: distance } of contextNeurons) {
			// Check if connection exists at this distance to this neuron
			const distanceMap = fromNeuron.connections.get(distance);
			if (!distanceMap) continue;
			const conn = distanceMap.get(this);
			if (!conn) continue;

			// Exponential smoothing: new_reward = smooth * observed + (1 - smooth) * old_reward
			conn.reward = smoothing * reward + (1 - smoothing) * conn.reward;
		}
	}
}

