import { Context } from './context.js';
import { Neuron } from './neuron.js';

/**
 * Thalamus - Brain's relay station for reference frame transfers
 * 
 * Abstracts access to neurons, channels, and dimension mappings.
 * Handles bidirectional translation between external signals and internal neuron representations.
 * Named after the biological thalamus which routes sensory signals and translates reference frames.
 */
export class Thalamus {
	constructor(debug, patternForgetRate, mergeThreshold) {
		this.debug = debug;
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;

		// Neuron registry
		this.neurons = new Map(); // neuronId -> Neuron
		this.neuronsByValue = new Map(); // valueKey -> SensoryNeuron

		// Neuron metadata lookup tables (immutable after creation)
		this.neuronChannel = new Map(); // neuronId -> channelName (sensory neurons only)
		this.neuronType = new Map(); // neuronId -> 'event'|'action' (sensory neurons only)

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

		// Level counts - tracks number of neurons at each level for efficient max level diagnostics lookup
		this.levelCounts = []; // index = level, value = count of neurons at that level
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
		neuron = Neuron.createSensory(coordinates, this.patternForgetRate, this.mergeThreshold);
		this.neurons.set(neuron.id, neuron);
		this.neuronsByValue.set(neuron.valueKey, neuron);
		this.neuronChannel.set(neuron.id, channel);
		this.neuronType.set(neuron.id, type);
		this.incrementLevelCount(neuron.level); // for diagnostics
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
		this.incrementLevelCount(neuron.level); // for diagnostics
	}

	/**
	 * Restore brain state from a snapshot (same format as getSnapshot).
	 * Channels and dimensions are restored if present in the snapshot.
	 * @param {{neurons: Array<{neuron: Neuron, channel: string|undefined}>, channels?: Map, channelNameToId?: Object, dimensionNameToId?: Object}} snapshot
	 */
	restoreSnapshot(snapshot) {
		// Restore channels and derive dimension mappings from them
		if (snapshot.channels) {
			this.setChannels(snapshot.channels);
			this.loadDimensionMaps();
		}

		// Restore neurons
		this.neurons.clear();
		this.neuronsByValue.clear();
		this.neuronChannel.clear();
		this.neuronType.clear();
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
		this.levelCounts = [];

		for (const { neuron, channel, type } of snapshot.neurons) {
			this.neurons.set(neuron.id, neuron);
			if (channel) this.neuronChannel.set(neuron.id, channel);
			if (type) this.neuronType.set(neuron.id, type);
			this.incrementLevelCount(neuron.level);
			if (neuron.level === 0)
				this.neuronsByValue.set(neuron.valueKey, neuron);
			else
				this.registerDeath(neuron, Math.ceil(neuron.activationStrength / neuron.patternForgetRate));
		}
	}

	/**
	 * Reset all neurons and neuron ID counter
	 */
	reset() {
		this.neurons.clear();
		this.neuronsByValue.clear();
		this.neuronChannel.clear();
		this.neuronType.clear();
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
		this.levelCounts = []; // for diagnostics
		Neuron.nextId = 1;
	}

	/**
	 * Get the channel name for a neuron
	 * @param {number} neuronId - Neuron ID
	 * @returns {string} Channel name
	 */
	getNeuronChannel(neuronId) {
		return this.neuronChannel.get(neuronId);
	}

	/**
	 * Get the type for a neuron ('event' or 'action')
	 * @param {number} neuronId - Neuron ID
	 * @returns {string} Neuron type
	 */
	getNeuronType(neuronId) {
		return this.neuronType.get(neuronId);
	}

	/**
	 * Get inferred actions grouped by channel from the given inferences.
	 * @param {Array<{neuron, strength, reward}>} inferences - Inferred neurons from memory
	 * @returns {Map<string, Array>} - Map of channel name to array of {coordinates, strength, reward}
	 */
	getInferredActions(inferences) {
		const channelOutputs = new Map();
		for (const { neuron, strength, reward } of inferences) {
			if (this.getNeuronType(neuron.id) !== 'action') continue;
			const channel = this.neuronChannel.get(neuron.id);
			if (!channelOutputs.has(channel)) channelOutputs.set(channel, []);
			channelOutputs.get(channel).push({ coordinates: neuron.coordinates, strength, reward });
		}
		return channelOutputs;
	}

	/**
	 * Get actual event coordinates grouped by channel from active neuron IDs.
	 * @param {Set<number>} activeNeuronIds - Set of neuron IDs active at age 0
	 * @returns {Map<string, Array>} - Map of channel name to array of coordinate objects
	 */
	getActiveEvents(activeNeuronIds) {
		const result = new Map();
		for (const neuronId of activeNeuronIds) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron || this.getNeuronType(neuronId) !== 'event') continue;
			const channel = this.neuronChannel.get(neuron.id);
			if (!result.has(channel)) result.set(channel, []);
			result.get(channel).push(neuron.coordinates);
		}
		return result;
	}

	/**
	 * Get inferences with channel metadata attached.
	 * @param {Array<{neuron, strength}>} inferredNeurons - Inferred neurons from memory
	 * @returns {Array<{neuron, strength, channel}>} - Inferences with channel
	 */
	getInferences(inferredNeurons) {
		const inferences = [];
		for (const { neuron, strength } of inferredNeurons)
			inferences.push({ neuron, strength, channel: this.neuronChannel.get(neuron.id), type: this.getNeuronType(neuron.id) });
		return inferences;
	}

	/**
	 * Get a self-contained snapshot of all brain state for external consumers (backup, dump).
	 * Each neuron entry carries its resolved metadata — consumers never need separate lookups.
	 * @returns {{neurons: Array<{neuron: Neuron, channel: string|undefined}>, channels: Array, channelNameToId: Object, dimensionNameToId: Object}}
	 */
	getSnapshot() {
		const neurons = [];
		for (const neuron of this.neurons.values()) {
			const entry = { neuron };
			if (neuron.level === 0) {
				entry.channel = this.neuronChannel.get(neuron.id);
				entry.type = this.neuronType.get(neuron.id);
			}
			neurons.push(entry);
		}
		return {
			neurons,
			channels: this.getChannels(),
			channelNameToId: this.channelNameToId,
			dimensionNameToId: this.dimensionNameToId,
		};
	}

	/**
	 * Materialize all lazy decay into actual values and reset timestamps.
	 * Re-registers death frames so pattern cleanup continues working.
	 */
	materializeAndResetNeurons(currentFrame) {
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
		for (const neuron of this.neurons.values()) {
			neuron.materializeStrength(currentFrame);
			neuron.lastActivationFrame = 0;
			if (neuron.level > 0)
				this.registerDeath(neuron, Math.ceil(neuron.activationStrength / neuron.patternForgetRate));
		}
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
	 * Recognize patterns at a given level using active neuron state from memory.
	 * Brain passes the raw memory snapshot; Thalamus owns all neuron access.
	 * @param {number} level - The level to process
	 * @param {Array<Map<number, {activatedPattern, votes, context}>>} activeNeuronsByAge - Memory snapshot
	 * @param {number} frameNumber - Current frame number for lazy decay
	 * @returns {Array<{parent, age, match}>} - Matched patterns to activate
	 */
	recognizeLevel(level, activeNeuronsByAge, frameNumber) {

		// get the contexts (neuron id and distance pairs) organized by age to be used for recognition
		const contextByAge = this.getContextByAge(level, activeNeuronsByAge);

		// get the active neurons that will do the recognition
		const recognizers = this.getRecognizers(level, contextByAge, activeNeuronsByAge);
		if (recognizers.length === 0) return [];

		// ask recognizers to match the contexts to known patterns
		const matchedPatterns = this.matchPatterns(recognizers, frameNumber);

		// Refine context on matched patterns
		// TODO: this will be done automatically by the parent when matching the pattern when we move contexts to parents
		for (const { match } of matchedPatterns)
			match.pattern.refineContext(match.common, match.novel, match.missing, this.neurons);

		return matchedPatterns;
	}

	/**
	 * Build context maps keyed by age for a given level.
	 * @returns {Map<number, Context>}
	 */
	getContextByAge(level, activeNeuronsByAge) {
		const contextByAge = new Map();
		for (let ctxAge = 1; ctxAge < activeNeuronsByAge.length; ctxAge++)
			for (const neuronId of (activeNeuronsByAge[ctxAge] ?? new Map()).keys()) {
				const neuron = this.neurons.get(neuronId);
				if (this.skipActionNeuron(neuron) || neuron.level !== level) continue;
				for (let age = 0; age < ctxAge; age++) {
					if (!contextByAge.has(age)) contextByAge.set(age, new Context());
					contextByAge.get(age).addNeuron(neuronId, ctxAge - age, 1);
				}
			}
		return contextByAge;
	}

	/**
	 * Check if a neuron should be skipped (action neuron in a channel without action sequences)
	 */
	skipActionNeuron(neuron) {
		if (neuron.level !== 0 || this.neuronType.get(neuron.id) !== 'action') return false;
		const channel = this.channels.get(this.neuronChannel.get(neuron.id));
		return channel && !channel.actionSequences;
	}

	/**
	 * Collect recognizer neurons at a given level, paired with their pre-built contexts.
	 * @returns {Array<{neuron, age, context: Context}>}
	 */
	getRecognizers(level, contextByAge, activeNeuronsByAge) {
		const recognizers = [];
		for (let age = 0; age < activeNeuronsByAge.length; age++) {
			const context = contextByAge.get(age);
			if (!context) continue;
			for (const [neuronId, state] of (activeNeuronsByAge[age] ?? new Map())) {
				const neuron = this.neurons.get(neuronId);
				if (!neuron || state.activatedPattern !== null) continue;
				if (this.skipActionNeuron(neuron) || neuron.level !== level) continue;
				recognizers.push({ neuron, age, context });
			}
		}
		return recognizers;
	}

	/**
	 * Match recognizers against their patterns. Each pattern can only be recognized once.
	 * @returns {Array<{parent, age, match}>}
	 */
	matchPatterns(recognizers, frameNumber) {
		const matchedPatterns = [];
		const recognizedPatterns = new Set();
		for (const { neuron: parent, age, context } of recognizers) {
			const match = parent.matchPattern(context, frameNumber, this.neurons);
			if (match && !recognizedPatterns.has(match.pattern)) {
				recognizedPatterns.add(match.pattern);
				matchedPatterns.push({ parent, age, match });
			}
		}
		return matchedPatterns;
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
	 * Get all channel action neuron IDs as a Map
	 * @returns {Map<string, Set<number>>} - Map of channel name to action neuron IDs
	 */
	getAllChannelActionIds() {
		const result = new Map();
		for (const [channelName, actionNeurons] of this.channelActions) {
			const ids = new Set();
			for (const neuron of actionNeurons) ids.add(neuron.id);
			result.set(channelName, ids);
		}
		return result;
	}

	/**
	 * Execute actions for channels that have them
	 * Groups channels by type and calls static executeChannelActions on each channel class
	 * @param {Array} inferredNeurons - Array of { neuron, strength, reward, probability } from memory
	 */
	async executeChannelActions(inferredNeurons) {

		// prepare the channels map that contains their event and action inferences
		const channelInferences = new Map(); // channelName → { actions, events }
		for (const channelName of this.channels.keys())
			channelInferences.set(channelName, { actions: [], events: [] });

		// Add inferred neurons to their channels
		for (const inference of inferredNeurons) {
			const inferences = channelInferences.get(this.neuronChannel.get(inference.neuron.id));
			const type = this.neuronType.get(inference.neuron.id);
			if (type === 'action') inferences.actions.push(inference);
			else if (type === 'event') inferences.events.push(inference);
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
		const { nameToId, idToName } = Thalamus.buildDimensionMaps(this.getChannels());
		this.setDimensionMappings(nameToId, idToName);
		if (this.debug) console.log('Dimensions loaded:', nameToId);
	}

	/**
	 * Build dimension name↔id maps from channel instances.
	 * @param {Iterable<[string, Channel]>} channels - Channel entries
	 * @returns {{nameToId: Object, idToName: Object}}
	 */
	static buildDimensionMaps(channels) {
		const nameToId = {};
		const idToName = {};
		for (const [, channel] of channels) {
			for (const dim of channel.getEventDimensions()) {
				nameToId[dim.name] = dim.id;
				idToName[dim.id] = dim.name;
			}
			for (const dim of channel.getActionDimensions()) {
				nameToId[dim.name] = dim.id;
				idToName[dim.id] = dim.name;
			}
		}
		return { nameToId, idToName };
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
		if (pattern.parentId) {
			const parentNeuron = this.neurons.get(pattern.parentId);
			if (parentNeuron) parentNeuron.removeChild(pattern.id);
		}

		// Remove from death ledger
		this.unregisterDeath(pattern);

		// Delete this pattern neuron from the index
		this.neurons.delete(pattern.id);

		// decrement level count for diagnostics
		this.decrementLevelCount(pattern.level);

		// memory cleanup
		pattern.parentId = null;
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
		for (const entry of neuron.context.getEntries()) {
			const ctxNeuron = this.neurons.get(entry.neuronId);
			if (ctxNeuron) ctxNeuron.removeContextRef(neuron.id, entry.distance);
		}

		// for each pattern that has this neuron in their context, clean them up
		for (const [referencingPatternId, distances] of neuron.contextRefs) {
			const referencingPattern = this.neurons.get(referencingPatternId);
			if (!referencingPattern) continue;

			// Remove this neuron from that pattern's context (by neuron ID)
			for (const distance of distances)
				referencingPattern.removeContext(neuron.id, distance);

			// Check if the referencing pattern became deletable
			if (referencingPattern.canDelete(currentFrame))
				newlyDeletable.push(referencingPattern);
		}

		return newlyDeletable;
	}

	/**
	 * Increment the neuron count at a given level for diagnostics
	 */
	incrementLevelCount(level) {
		while (this.levelCounts.length <= level) this.levelCounts.push(0);
		this.levelCounts[level]++;
	}

	/**
	 * Decrement the neuron count at a given level for diagnostics
	 */
	decrementLevelCount(level) {
		if (level < this.levelCounts.length) this.levelCounts[level]--;
	}

	/**
	 * Get total number of neurons for diagnostics
	 * @returns {number}
	 */
	getNeuronCount() {
		return this.neurons.size;
	}

	/**
	 * Get the maximum level of any neuron currently in the registry for diagnostics
	 * @returns {number}
	 */
	getMaxLevel() {
		for (let i = this.levelCounts.length - 1; i >= 0; i--)
			if (this.levelCounts[i] > 0) return i;
		return 0;
	}
}