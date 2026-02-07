import { Neuron } from './neurons/neuron.js';

/**
 * Thalamus - Brain's relay station for reference frame transfers
 * 
 * Abstracts access to neurons, channels, and dimension mappings.
 * Handles bidirectional translation between external signals and internal neuron representations.
 * Named after the biological thalamus which routes sensory signals and translates reference frames.
 */
export class Thalamus {
	constructor(debug = false) {
		this.debug = debug;

		// Neuron registry
		this.neurons = new Map(); // neuronId -> Neuron
		this.neuronsByValue = new Map(); // valueKey -> SensoryNeuron

		// Channel registry
		this.channels = new Map(); // channelName -> Channel instance
		this.channelActions = new Map(); // channelName -> Set<Neuron>
		this.channelNameToId = {}; // channelName -> channelId
		this.channelIdToName = {}; // channelId -> channelName

		// Dimension mappings
		this.dimensionNameToId = {}; // dimensionName -> dimensionId
		this.dimensionIdToName = {}; // dimensionId -> dimensionName
	}

	// ============ NEURON OPERATIONS ============

	/**
	 * Get or create a sensory neuron ID from a frame point
	 * @param {object} point - Frame point with {coordinates, channel, type}
	 * @returns {number} - Neuron ID
	 */
	getNeuronIdForPoint(point) {

		// Try to find existing neuron - if found, return it
		let neuronId = this.getNeuronIdByCoordinates(point.coordinates);
		if (neuronId) return neuronId;

		// Create new neuron if not found
		const neuron = Neuron.createSensory(point.channel, point.type, point.coordinates);
		this.neurons.set(neuron.id, neuron);
		this.neuronsByValue.set(neuron.valueKey, neuron);
		neuronId = neuron.id;
		if (this.debug) console.log(`Created new sensory neuron ${neuronId} for ${neuron.valueKey}`);
		return neuronId;
	}

	/**
	 * Get neuron object by ID
	 * @param {number} neuronId - Neuron ID
	 * @returns {Neuron|undefined} - Neuron object or undefined
	 */
	getNeuron(neuronId) {
		return this.neurons.get(neuronId);
	}

	/**
	 * Get all neuron entries (id and neuron) for iteration
	 * @returns {IterableIterator<[number, Neuron]>} - Iterator of [neuronId, neuron] pairs
	 */
	getAllNeuronEntries() {
		return this.neurons.entries();
	}

	/**
	 * Add a neuron to the registry
	 * @param {Neuron} neuron - Neuron to add
	 */
	addNeuron(neuron) {
		this.neurons.set(neuron.id, neuron);
		if (neuron.level === 0) this.neuronsByValue.set(neuron.valueKey, neuron);
	}

	/**
	 * Get neuron count
	 * @returns {number} - Number of neurons
	 */
	getNeuronCount() {
		return this.neurons.size;
	}

	/**
	 * Reset all neurons and neuron ID counter
	 */
	reset() {
		this.neurons.clear();
		this.neuronsByValue.clear();
		Neuron.nextId = 1;
	}

	// ============ CHANNEL OPERATIONS ============

	/**
	 * Register a channel
	 * @param {string} name - Channel name
	 * @param {Channel} channelClass - Channel class constructor
	 * @returns {object} - Channel instance
	 */
	registerChannel(name, channelClass) {
		this.channels.set(name, new channelClass(name, this.debug));
		if (this.debug) console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * Get channel instance by name
	 * @param {string} channelName - Channel name
	 * @returns {object|undefined} - Channel instance or undefined
	 */
	getChannel(channelName) {
		return this.channels.get(channelName);
	}

	/**
	 * Get all channels for iteration
	 * @returns {IterableIterator<[string, object]>} - Iterator of [channelName, channel] pairs
	 */
	getAllChannels() {
		return this.channels.entries();
	}

	/**
	 * Get all channels with their IDs for iteration
	 * @returns {Array<{name: string, id: number, channel: object}>} - Array of channel objects
	 */
	getAllChannelsWithIds() {
		const result = [];
		for (const [channelName, channel] of this.channels)
			result.push({ name: channelName, id: this.channelNameToId[channelName], channel });
		return result;
	}

	/**
	 * Get channel ID by name
	 * @param {string} channelName - Channel name
	 * @returns {number|undefined} - Channel ID or undefined
	 */
	getChannelId(channelName) {
		return this.channelNameToId[channelName];
	}

	/**
	 * Get action neurons for a channel
	 * @param {string} channelName - Channel name
	 * @returns {Set<Neuron>|undefined} - Set of action neurons or undefined
	 */
	getChannelActions(channelName) {
		return this.channelActions.get(channelName);
	}

	/**
	 * Get all channel actions as a Map
	 * @returns {Map<string, Set<Neuron>>} - Map of channel name to action neurons
	 */
	getAllChannelActions() {
		return this.channelActions;
	}

	/**
	 * Execute actions for channels that have them
	 * @param {Map<string, Array>} channelActions - Map of channel name to action coordinates
	 */
	async executeChannelActions(channelActions) {
		for (const [channelName, actions] of channelActions)
			await this.channels.get(channelName).executeOutputs(actions);
	}

	// ============ DIMENSION OPERATIONS ============

	/**
	 * Get dimension name by ID
	 * @param {number} dimensionId - Dimension ID
	 * @returns {string|undefined} - Dimension name or undefined
	 */
	getDimensionName(dimensionId) {
		return this.dimensionIdToName[dimensionId];
	}

	/**
	 * Get dimension ID by name
	 * @param {string} dimensionName - Dimension name
	 * @returns {number|undefined} - Dimension ID or undefined
	 */
	getDimensionId(dimensionName) {
		return this.dimensionNameToId[dimensionName];
	}

	// ============ DIAGNOSTIC OPERATIONS ============

	/**
	 * returns neuron ID by coordinates (for diagnostics)
	 * @param {object} coordinates - Coordinate object with dimension-value pairs
	 * @returns {number|null} - Neuron ID or null if not found
	 */
	getNeuronIdByCoordinates(coordinates) {
		const valueKey = Neuron.makeValueKey(coordinates);
		const neuron = this.neuronsByValue.get(valueKey);
		return neuron ? neuron.id : null;
	}

	// ============ INITIALIZATION ============

	/**
	 * Initialize channels and set mappings
	 */
	initializeChannels() {
		const channelNameToId = {};
		const channelIdToName = {};

		for (const [channelName, channel] of this.getAllChannels()) {
			channelNameToId[channelName] = channel.id;
			channelIdToName[channel.id] = channelName;
		}

		this.setChannelMappings(channelNameToId, channelIdToName);
		if (this.debug) console.log('Channels loaded:', channelNameToId);
	}

	/**
	 * Load dimensions and set mappings
	 */
	loadDimensions() {
		const dimensionNameToId = {};
		const dimensionIdToName = {};

		for (const [, channel] of this.getAllChannels()) {
			for (const dim of channel.getEventDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
			for (const dim of channel.getOutputDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
		}

		this.setDimensionMappings(dimensionNameToId, dimensionIdToName);
		if (this.debug) console.log('Dimensions loaded:', dimensionNameToId);
	}

	/**
	 * Pre-create action neurons for all channels
	 */
	initializeActionNeurons() {
		const channelActions = new Map();

		for (const { name: channelName, id: channelId, channel } of this.getAllChannelsWithIds()) {

			const actionCoords = channel.getActions();

			// Build action points for action neurons
			const actionPoints = channel.getActions().map(coords => ({
				coordinates: coords,
				channel: channelName,
				channel_id: channelId,
				type: 'action'
			}));

			// Create neurons using thalamus
			const actionNeurons = new Set();
			for (const point of actionPoints) {
				const neuronId = this.getNeuronIdForPoint(point);
				const neuron = this.getNeuron(neuronId);
				actionNeurons.add(neuron);
			}

			channelActions.set(channelName, actionNeurons);
			if (this.debug) console.log(`Created ${actionCoords.length} action neurons for ${channelName}`);
		}

		this.setChannelActions(channelActions);
	}

	/**
	 * Initialize all channels (channel-specific setup)
	 */
	async initializeAllChannels() {
		for (const [, channel] of this.getAllChannels())
			await channel.initialize();
	}

	/**
	 * Set channel mappings (called during init)
	 * @param {object} nameToId - Channel name to ID mapping
	 * @param {object} idToName - Channel ID to name mapping
	 */
	setChannelMappings(nameToId, idToName) {
		this.channelNameToId = nameToId;
		this.channelIdToName = idToName;
	}

	/**
	 * Set dimension mappings (called during init)
	 * @param {object} nameToId - Dimension name to ID mapping
	 * @param {object} idToName - Dimension ID to name mapping
	 */
	setDimensionMappings(nameToId, idToName) {
		this.dimensionNameToId = nameToId;
		this.dimensionIdToName = idToName;
	}

	/**
	 * Set channel actions (called during init)
	 * @param {Map} channelActions - Map of channel name to Set of action neurons
	 */
	setChannelActions(channelActions) {
		this.channelActions = channelActions;
	}

	// ============ FORGET CYCLE ============

	/**
	 * Run forget cycle on all neurons and collect orphaned patterns.
	 * forget neuron patterns and connections and then delete if it can be deleted
	 * @returns {Array<Neuron>} - Array of neurons that can be deleted
	 */
	forgetNeurons() {
		const patterns = [];
		for (const neuron of this.neurons.values()) if (neuron.forget()) patterns.push(neuron);
		return patterns;
	}

	/**
	 * Delete a pattern neuron.
	 * Since canDelete() requires contexts.length === 0, the pattern has no children.
	 * @param {Neuron} pattern - Pattern neuron to delete
	 */
	deletePattern(pattern) {

		// Remove pattern from its peak's routing table (if peak still exists)
		// Peak might have been deleted already if both were in the deletion list
		if (pattern.peak && this.neurons.has(pattern.peak.id)) pattern.peak.removePattern(pattern);

		// Delete this pattern neuron from the index
		this.neurons.delete(pattern.id);
	}
}