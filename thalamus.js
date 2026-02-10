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
		this.channelClasses = new Map(); // channelName -> Channel class (not instantiated)
		this.channels = new Map(); // channelName -> Channel instance
		this.channelActions = new Map(); // channelName -> Set<Neuron>
		this.channelNameToId = {}; // channelName -> channelId
		this.channelIdToName = {}; // channelId -> channelName

		// Dimension mappings
		this.dimensionNameToId = {}; // dimensionName -> dimensionId
		this.dimensionIdToName = {}; // dimensionId -> dimensionName
	}

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
	 * returns all neurons as an array
	 */
	getAllNeurons() {
		return Array.from(this.neurons.values());
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
	 * Set neurons from a Map (used when loading from database)
	 * @param {Map<number, Neuron>} neurons - Map of neuron ID to neuron object
	 */
	setNeurons(neurons) {
		this.neurons = neurons;
	}

	/**
	 * Reset all neurons and neuron ID counter
	 */
	reset() {
		this.neurons.clear();
		this.neuronsByValue.clear();
		Neuron.nextId = 1;
	}

	/**
	 * Register a channel class (not instantiated yet)
	 * @param {string} name - Channel name
	 * @param {Class} channelClass - Channel class constructor
	 */
	registerChannel(name, channelClass) {
		this.channelClasses.set(name, channelClass);
		if (this.debug) console.log(`Registered channel class: ${name} (${channelClass.name})`);
	}

	/**
	 * Add an instantiated channel to the thalamus
	 * @param {string} name - Channel name
	 * @param {Channel} channelInstance - Instantiated channel object
	 */
	addChannel(name, channelInstance) {
		this.channels.set(name, channelInstance);
		this.channelNameToId[name] = channelInstance.id;
		this.channelIdToName[channelInstance.id] = name;
		if (this.debug) console.log(`Added channel instance: ${name}`);
	}

	/**
	 * Instantiate new channels (those registered but not yet in thalamus).
	 * Called after loadChannels (if DB) or standalone (if no DB).
	 */
	instantiateChannels() {
		for (const [channelName, channelClass] of this.channelClasses) {
			if (this.channels.has(channelName)) continue;
			const channel = new channelClass(channelName, this.debug);
			this.addChannel(channelName, channel);
			if (this.debug) console.log(`Created new channel: ${channelName} (id: ${channel.id})`);
		}
	}

	/**
	 * Get channel instance by name
	 */
	getChannel(channelName) {
		return this.channels.get(channelName);
	}

	/**
	 * Get all channels for iteration
	 */
	getAllChannels() {
		return Array.from(this.channels.entries());
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
	 * returns channel name map to id
	 */
	getChannelNameToIdMap() {
		return this.channelNameToId;
	}

	/**
	 * returns channel id map to name
	 */
	getChannelIdToNameMap() {
		return this.channelIdToName;
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

	/**
	 * Get dimension name by ID
	 * @param {number} dimensionId - Dimension ID
	 * @returns {string|undefined} - Dimension name or undefined
	 */
	getDimensionName(dimensionId) {
		return this.dimensionIdToName[dimensionId];
	}

	/**
	 * returns dimension name map to id
	 */
	getDimensionNameToIdMap() {
		return this.dimensionNameToId;
	}

	/**
	 * returns dimension id map to name
	 */
	getDimensionIdToNameMap() {
		return this.dimensionIdToName;
	}

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

	/**
	 * Load dimension name/id mappings from instantiated channels
	 */
	loadDimensionMaps() {
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
	 * Pre-create action neurons for all channels if they don't exist, so that we
	 */
	initializeActionNeurons() {
		const channelActions = new Map();

		for (const { name: channelName, id: channelId, channel } of this.getAllChannelsWithIds()) {

			// get points for the channel's action neurons
			const actionCoords = channel.getActions();

			const actionPoints = actionCoords.map(coords => ({
				coordinates: coords,
				channel: channelName,
				channel_id: channelId,
				type: 'action'
			}));

			// Get action neurons and add them
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

	/**
	 * Run forget cycle on all neurons and collect orphaned patterns.
	 * forget neuron patterns and connections and then delete if it can be deleted
	 * @returns {Array<Neuron>} - Array of neurons that can be deleted
	 */
	forgetNeurons() {
		const patterns = [];
		for (const neuron of this.getAllNeurons()) if (neuron.forget()) patterns.push(neuron);
		return patterns;
	}

	/**
	 * Delete a pattern neuron.
	 * Since canDelete() requires contexts.length === 0, the pattern has no children.
	 * @param {Neuron} pattern - Pattern neuron to delete
	 */
	deletePattern(pattern) {

		// remove all context references for this pattern
		// for (const [peak, peakRefs] of pattern.contextRefs)
		// 	for (const [peakPattern, distanceSet] of peakRefs)
		// 		for (const distance of distanceSet)
		// 			peak.removePatternContext(peakPattern, pattern, distance);

		// Remove pattern from its peak's routing table (if peak still exists)
		// Peak might have been deleted already if both were in the deletion list
		if (pattern.peak && this.neurons.has(pattern.peak.id)) pattern.peak.removePattern(pattern);

		// Delete this pattern neuron from the index
		this.neurons.delete(pattern.id);
	}
}