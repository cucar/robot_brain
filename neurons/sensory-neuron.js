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
	 * Override canDelete - sensory neurons are never deleted (they have coordinates)
	 */
	canDelete() {
		return false; // Sensory neurons persist forever
	}

	/**
	 * Refine outgoing connections at this distance based on newly active neurons.
	 * Called when this neuron is active at age > 0 (context neuron doing the learning).
	 * Only event neurons have outgoing connections.
	 * @param {number} distance - The distance to refine (this neuron's current age)
	 * @param {Set<Neuron>} newlyActiveNeurons - Currently active neurons at age=0
	 * @param {Map<string, number>} rewards - Map of channel name to reward value
	 * @param {Map<string, Set<Neuron>>} channelActions - Unused, for interface compatibility with PatternNeuron
	 * @returns {{strengthened: number, rewarded: number}}
	 */
	refineInferences(distance, newlyActiveNeurons, rewards, channelActions) {

		// Only event neurons have outgoing connections
		if (this.type !== 'event') return { strengthened: 0, rewarded: 0 };

		let strengthened = 0, rewarded = 0;

		for (const targetNeuron of newlyActiveNeurons) {
			// Skip non-base neurons
			if (targetNeuron.level !== 0) continue;

			// Strengthen connection to this target
			const conn = this.getOrCreateConnection(distance, targetNeuron);
			conn.strength = Math.min(Neuron.maxStrength, conn.strength + 1);
			strengthened++;

			// Apply reward if this is an action with a reward
			if (targetNeuron.type === 'action') {
				const reward = rewards.get(targetNeuron.channel);
				if (reward !== undefined)
					conn.reward = Neuron.rewardSmoothing * reward + (1 - Neuron.rewardSmoothing) * conn.reward;
				rewarded++;
			}
		}

		return { strengthened, rewarded };
	}
}

