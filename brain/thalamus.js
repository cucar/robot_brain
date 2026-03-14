import { Neuron } from './neuron.js';

/**
 * Thalamus - Brain's relay station for reference frame transfers
 * 
 * Abstracts access to neurons, channels, and dimension mappings.
 * Handles bidirectional translation between external signals and internal neuron representations.
 * Named after the biological thalamus which routes sensory signals and translates reference frames.
 */
export class Thalamus {
	constructor(options = {}) {
		this.options = options; // Runtime options to pass to channel classes
		this.debug = options.debug;

		// Neuron registry
		this.neurons = new Map(); // neuronId -> Neuron
		this.neuronsByValue = new Map(); // valueKey -> SensoryNeuron

		// Death ledger - scheduled neuron deaths
		this.deathLedger = new Map(); // frameNumber -> Set<Neuron>
		this.neuronDeathFrame = new Map(); // neuron -> frameNumber (reverse lookup)

		// Channel registry
		this.channelClasses = new Map(); // channelName -> Channel class (not instantiated)
		this.channels = new Map(); // channelName -> Channel instance
		this.channelActions = new Map(); // channelName -> Set<Neuron>
		this.channelDefaultActions = new Map(); // channelName -> Neuron
		this.channelNameToId = {}; // channelName -> channelId
		this.channelIdToName = {}; // channelId -> channelName

		// Dimension mappings
		this.dimensionNameToId = {}; // dimensionName -> dimensionId
		this.dimensionIdToName = {}; // dimensionId -> dimensionName
	}

	/**
	 * Get or create a sensory neuron ID from a frame point
	 * @returns {Neuron} - Neuron
	 */
	getNeuronForPoint(coordinates, channel, type) {

		// Try to find existing neuron - if found, return it
		let neuron = this.getNeuronByCoordinates(coordinates);
		if (neuron) return neuron;

		// Create new neuron if not found
		neuron = Neuron.createSensory(channel, type, coordinates);
		this.neurons.set(neuron.id, neuron);
		this.neuronsByValue.set(neuron.valueKey, neuron);
		if (this.debug) console.log(`Created new sensory neuron ${neuron.id} for ${neuron.valueKey}`);
		return neuron;
	}

	/**
	 * returns neuron ID by coordinates (for diagnostics)
	 * @param {object} coordinates - Coordinate object with dimension-value pairs
	 * @returns {Neuron|null} - Neuron or null if not found
	 */
	getNeuronByCoordinates(coordinates) {
		return this.neuronsByValue.get(Neuron.makeValueKey(coordinates));
	}

	/**
	 * returns all neurons as an array
	 */
	getNeurons() {
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

		// Rebuild neuronsByValue map for base neurons
		this.neuronsByValue.clear();
		for (const neuron of neurons.values())
			if (neuron.level === 0)
				this.neuronsByValue.set(neuron.valueKey, neuron);
	}

	/**
	 * Reset all neurons and neuron ID counter
	 */
	reset() {
		this.neurons.clear();
		this.neuronsByValue.clear();
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
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
	 * Get registered channel classes
	 * @returns {Map} Map of channel name to channel class
	 */
	getChannelClasses() {
		return this.channelClasses;
	}

	/**
	 * Set channels from a Map (used when loading from database)
	 * @param {Map<string, Channel>} channels - Map of channel name to channel instance
	 */
	setChannels(channels) {
		for (const [channelName, channel] of channels) {
			this.addChannel(channelName, channel);
			if (this.debug) console.log(`Loaded channel from DB: ${channelName} (id: ${channel.id})`);
		}
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

			// protection to not instantiate channels that already exist - should not happen - just in case
			if (this.channels.has(channelName)) continue;

			// initialize channel class with runtime options
			channelClass.initialize(this.options);

			// create new channel instance and add it to the thalamus
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
	getChannels() {
		return Array.from(this.channels.entries());
	}

	/**
	 * Get channel metrics for all channels
	 * @returns {Array<Object>} - Array of channel metrics
	 */
	getChannelMetrics() {
		const metrics = [];
		for (const [, channel] of this.channels)
			metrics.push(channel.getMetrics());
		return metrics;
	}

	/**
	 * Get aggregate metrics by detecting distinct channel classes and calling their static methods
	 * @returns {Object|null} - Aggregate metrics keyed by channel class name, or null if none
	 */
	getAggregateMetrics() {
		if (this.channels.size === 0) return null;

		// Group channels by their class constructor
		const channelsByClass = new Map(); // ChannelClass → Map(channelName → channel)
		for (const [channelName, channel] of this.channels) {
			const ChannelClass = channel.constructor;
			if (!channelsByClass.has(ChannelClass)) channelsByClass.set(ChannelClass, new Map());
			channelsByClass.get(ChannelClass).set(channelName, channel);
		}

		// Call static getAggregateMetrics on each channel class
		const aggregateMetrics = {};
		for (const [ChannelClass, channelsOfType] of channelsByClass) {
			const metrics = ChannelClass.getAggregateMetrics(channelsOfType);
			if (metrics) {
				const className = ChannelClass.name;
				aggregateMetrics[className] = metrics;
			}
		}

		return Object.keys(aggregateMetrics).length > 0 ? aggregateMetrics : null;
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
	 * @returns {Neuron} - Set of action neurons or undefined
	 */
	getChannelDefaultAction(channelName) {
		return this.channelDefaultActions.get(channelName);
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
	 * Groups channels by type and calls static executeChannelActions on each channel class
	 * @param {Array} inferredNeurons - Array of { neuron, strength, reward } from memory
	 */
	async executeChannelActions(inferredNeurons) {

		// prepare the channels map that contains their event and action inferences
		const channelInferences = new Map(); // channelName → { actions, events }
		for (const channelName of this.channels.keys())
			channelInferences.set(channelName, { actions: [], events: [] });

		// Add inferred neurons to their channels
		for (const { neuron, strength, reward } of inferredNeurons) {
			const inferences = channelInferences.get(neuron.channel);
			const inference = { coordinates: neuron.coordinates, strength, reward };
			if (neuron.type === 'action') inferences.actions.push(inference);
			else if (neuron.type === 'event') inferences.events.push(inference);
		}

		// group by channel classes for action execution
		const channelTypes = new Map();
		for (const [channelName, inferences] of channelInferences) {
			const channel = this.channels.get(channelName);
			const ChannelClass = channel.constructor;
			if (!channelTypes.has(ChannelClass)) channelTypes.set(ChannelClass, new Map());
			channelTypes.get(ChannelClass).set(channelName, { channel, ...inferences });
		}

		// Dispatch to each channel class
		for (const [ChannelClass, classChannelData] of channelTypes)
			await ChannelClass.executeChannelActions(classChannelData);
	}

	/**
	 * Pre-create action neurons for all channels if they don't exist, so that we
	 */
	initializeActionNeurons() {

		// get points for the channel's action neurons and add them to the channel's action set for exploration
		const channelActions = new Map();
		for (const [channelName, channel] of this.getChannels()) {

			// get or create the action neurons for the channel
			const actionNeurons = new Set();
			for (const coordinates of channel.getActions())
				actionNeurons.add(this.getNeuronForPoint(coordinates, channelName, 'action'));

			// add channel's action neurons to the channelActions map
			channelActions.set(channelName, actionNeurons);
			if (this.debug) console.log(`Created ${actionNeurons.size} action neurons for ${channelName}`);

			// set the default action for the channel (if one exists)
			const defaultActionCoords = channel.getDefaultAction();
			if (defaultActionCoords !== null) {
				const defaultAction = this.getNeuronForPoint(defaultActionCoords, channelName, 'action');
				this.channelDefaultActions.set(channelName, defaultAction);
			}
		}

		// set channel actions for exploration
		this.channelActions = channelActions;
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
	 * Load dimension name/id mappings from instantiated channels
	 */
	loadDimensionMaps() {
		const dimensionNameToId = {};
		const dimensionIdToName = {};

		for (const [, channel] of this.getChannels()) {
			for (const dim of channel.getEventDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
			for (const dim of channel.getActionDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
		}

		this.setDimensionMappings(dimensionNameToId, dimensionIdToName);
		if (this.debug) console.log('Dimensions loaded:', dimensionNameToId);
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
	 * Register a neuron's scheduled death frame
	 */
	registerDeath(neuron, deathFrame) {
		// unregister old death frame if exists
		this.unregisterDeath(neuron);

		// register new death frame
		if (!this.deathLedger.has(deathFrame)) this.deathLedger.set(deathFrame, new Set());
		this.deathLedger.get(deathFrame).add(neuron);
		this.neuronDeathFrame.set(neuron, deathFrame);
	}

	/**
	 * Unregister a neuron from the death ledger
	 */
	unregisterDeath(neuron) {
		const oldFrame = this.neuronDeathFrame.get(neuron);
		if (oldFrame === undefined) return;

		const set = this.deathLedger.get(oldFrame);
		if (set) {
			set.delete(neuron);
			if (set.size === 0) this.deathLedger.delete(oldFrame);
		}
		this.neuronDeathFrame.delete(neuron);
	}

	/**
	 * Reap neurons scheduled to die at or before the given frame
	 * Returns array of dead neurons and removes them from the ledger
	 */
	reapDeadNeurons(currentFrame) {
		const dead = [];
		for (const [frame, neurons] of this.deathLedger) {
			if (frame > currentFrame) continue;
			for (const neuron of neurons) {
				dead.push(neuron);
				this.neuronDeathFrame.delete(neuron);
			}
			this.deathLedger.delete(frame);
		}
		return dead;
	}

	/**
	 * Delete dead pattern neurons with recursive cascade.
	 * When a pattern is deleted, other patterns that referenced it may become deletable too.
	 * @param {Array<Neuron>} patterns - Initial list of patterns to delete
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<Neuron>} - All deleted patterns
	 */
	deletePatterns(patterns, currentFrame) {
		const toDelete = [...patterns];
		const queuedIds = new Set(patterns.map(pattern => pattern.id));
		const deletedPatterns = [];
		const deletedIds = new Set();

		while (toDelete.length > 0) {
			const pattern = toDelete.shift();
			if (deletedIds.has(pattern.id)) continue;

			const newlyDeletable = this.deletePattern(pattern, currentFrame);
			deletedPatterns.push(pattern);
			deletedIds.add(pattern.id);

			for (const newlyDeletablePattern of newlyDeletable) {
				if (deletedIds.has(newlyDeletablePattern.id) || queuedIds.has(newlyDeletablePattern.id)) continue;
				toDelete.push(newlyDeletablePattern);
				queuedIds.add(newlyDeletablePattern.id);
			}
		}

		return deletedPatterns;
	}

	/**
	 * Delete a pattern neuron and clean up all references to it.
	 * Returns patterns that became deletable as a result of cleanup.
	 * @param {Neuron} pattern - Pattern to delete
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<Neuron>} - Patterns that became deletable after cleanup
	 */
	deletePattern(pattern, currentFrame) {

		// ignore double delete requests
		if (!this.neurons.has(pattern.id)) return [];

		// Clean up this pattern from other patterns' contexts
		const newlyDeletable = this.cleanupContextReferences(pattern, currentFrame);

		// Remove pattern from its parent's routing table (if parent still exists)
		if (pattern.parent && this.neurons.has(pattern.parent.id)) pattern.parent.removeChild(pattern);

		// Remove from death ledger
		this.unregisterDeath(pattern);

		// Delete this pattern neuron from the index
		this.neurons.delete(pattern.id);

		// memory cleanup
		pattern.parent = null;
		delete pattern.context;
		delete pattern.contextRefs;
		delete pattern.children;
		delete pattern.connections;
		pattern = null;

		return newlyDeletable;
	}

	/**
	 * Clean up context references when deleting a neuron/pattern.
	 * pattern.contextRefs tells us which patterns have this pattern in their context.
	 * We need to remove this pattern from those patterns' contexts.
	 * @param {Neuron} neuron - Neuron/pattern being deleted
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<Neuron>} - Patterns that became deletable after cleanup
	 */
	cleanupContextReferences(neuron, currentFrame) {
		const newlyDeletable = [];

		// clean up forward references (neurons this pattern referenced)
		// most of the time, this should be empty if the neuron is getting deleted, but it's possible for some left over
		for (const entry of neuron.context.getEntries())
			entry.neuron.removeContextRef(neuron, entry.distance);

		// for each pattern that has this neuron in their context, clean them up
		for (const [referencingPattern, distances] of neuron.contextRefs) {

			// Remove this neuron from that pattern's context
			for (const distance of distances)
				referencingPattern.removeContext(neuron, distance);

			// Check if the referencing pattern became deletable
			if (referencingPattern.canDelete(currentFrame))
				newlyDeletable.push(referencingPattern);
		}

		return newlyDeletable;
	}
}