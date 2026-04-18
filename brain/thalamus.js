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
		this.neuronsByValue = new Map(); // valueKey -> neuronId (coordinates -> neuronId lookup)
		this.neuronValues = new Map(); // neuronId -> coordinates (neuron -> coordinates reverse lookup)
		this.neuronChannel = new Map(); // neuronId -> channelName (sensory neurons only)
		this.neuronType = new Map(); // neuronId -> 'event'|'action' (sensory neurons only)
		this.neuronParents = new Map(); // neuronId -> parentNeuronId (pattern neurons only)
		this.neuronLevels = new Map(); // neuronId -> level (0 = sensory, 1+ = pattern)

		// Death ledger - scheduled neuron deaths
		this.deathLedger = new Map(); // frameNumber -> Set<number>
		this.neuronDeathFrame = new Map(); // neuronId -> frameNumber (reverse lookup)

		// Channel registry
		this.channelClasses = new Map(); // channelName -> Channel class (not instantiated)
		this.channels = new Map(); // channelName -> Channel instance
		this.channelActions = new Map(); // channelName -> Set<number> (neuron ids)
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
	 * Create value key for neuron coordinate lookup
	 */
	makeValueKey(coordinates) {
		const sorted = Object.keys(coordinates).sort();
		const obj = {};
		for (const k of sorted) obj[k] = coordinates[k];
		return JSON.stringify(obj);
	}

	/**
	 * Get or create a sensory neuron ID from a frame point
	 * @returns {number} - Neuron ID
	 */
	getNeuronIdForPoint(coordinates, channel, type) {

		// Try to find existing neuron - if found, return it
		let neuronId = this.getNeuronIdByCoordinates(coordinates);
		if (neuronId) return neuronId;

		// Create new neuron if not found
		const neuron = this.addSensoryNeuron(coordinates);
		this.neuronChannel.set(neuron.id, channel);
		this.neuronType.set(neuron.id, type);
		if (this.debug) console.log(`Created new sensory neuron ${neuron.id} for ${this.makeValueKey(coordinates)}`);
		return neuron.id;
	}

	/**
	 * returns neuron ID by coordinates (for diagnostics)
	 * @param {object} coordinates - Coordinate object with dimension-value pairs
	 * @returns {number|null} - Neuron ID or null if not found
	 */
	getNeuronIdByCoordinates(coordinates) {
		return this.neuronsByValue.get(this.makeValueKey(coordinates)) || null;
	}

	/**
	 * Create and add a new sensory neuron to the registry
	 * @param {object} coordinates - Coordinates for sensory neurons (level 0)
	 * @returns {Neuron} The newly created neuron
	 */
	addSensoryNeuron(coordinates) {
		const neuron = new Neuron(this.patternForgetRate, this.mergeThreshold);
		this.neurons.set(neuron.id, neuron);
		this.neuronLevels.set(neuron.id, 0);
		this.neuronsByValue.set(this.makeValueKey(coordinates), neuron.id);
		this.neuronValues.set(neuron.id, coordinates);
		this.incrementLevelCount(0); // for diagnostics
		return neuron;
	}

	/**
	 * Create a new pattern neuron and register it in the thalamus. Pure factory:
	 * wires the pattern's own connections to sensory history but does NOT touch the
	 * parent's routing table (that happens inside parent.processFrame via addPattern)
	 * and does NOT deliver contextRef updates or register death (death frame is known
	 * only after parent.addPattern runs).
	 * @param {number} level - Neuron level (1+ = pattern)
	 * @param {number} parentId - Parent neuron ID
	 * @param {number} age - Distance in time between the observation and the error
	 * @param {Array<Set<number>>} sensoryNeurons - Recent sensory neuron ids by age
	 * @param {Array<Map<string, number>>} rewards - Rewards by age
	 * @param {Map<string, Set<number>>} channelActionIds - Action neuron ids by channel
	 * @returns {number} The newly created pattern neuron
	 */
	createPatternNeuron(level, parentId, age, sensoryNeurons, rewards, channelActionIds) {

		// build the future connection spec of the pattern from currently observed neurons
		// (thalamus-side lookups - channel, reward - are resolved here so the neuron can be
		// initialized with a single self-contained call; MPI-ready boundary)
		const connections = [];
		for (let a = 0; a < age && a < sensoryNeurons.length; a++)
			for (const sensoryNeuronId of sensoryNeurons[a]) {
				const channel = this.getNeuronChannel(sensoryNeuronId);
				const reward = rewards[a].get(channel) || 0;
				connections.push({ distance: age - a, toNeuronId: sensoryNeuronId, channel, reward });
			}

		// create and initialize the neuron in a single call
		const neuron = new Neuron(this.patternForgetRate, this.mergeThreshold);
		neuron.initializeConnections(connections, channelActionIds);

		// register in the thalamus
		this.neurons.set(neuron.id, neuron);
		this.neuronLevels.set(neuron.id, level);
		this.neuronParents.set(neuron.id, parentId);
		this.incrementLevelCount(level); // for diagnostics

		return neuron.id;
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
		this.reset();

		for (const { neuron, level, channel, type, coordinates, parentId } of snapshot.neurons) {
			this.neurons.set(neuron.id, neuron);
			this.neuronLevels.set(neuron.id, level);
			if (channel) this.neuronChannel.set(neuron.id, channel);
			if (type) this.neuronType.set(neuron.id, type);
			if (parentId) this.neuronParents.set(neuron.id, parentId);
			this.incrementLevelCount(level);
			if (level === 0 && coordinates) {
				this.neuronsByValue.set(this.makeValueKey(coordinates), neuron.id);
				this.neuronValues.set(neuron.id, coordinates);
			}
			else if (level > 0 && parentId) {
				const parent = this.neurons.get(parentId);
				if (parent) {
					const ctx = parent.routingTable.get(neuron.id);
					if (ctx) this.registerDeath(neuron.id, Math.ceil(ctx.activationStrength / neuron.patternForgetRate));
				}
			}
		}
	}

	/**
	 * Reset all neurons and neuron ID counter
	 */
	reset() {
		this.neurons.clear();
		this.neuronLevels.clear();
		this.neuronsByValue.clear();
		this.neuronValues.clear();
		this.neuronChannel.clear();
		this.neuronType.clear();
		this.neuronParents.clear();
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
		this.levelCounts = []; // for diagnostics
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
	 * Get the parent neuron ID for a pattern neuron
	 * @param {number} neuronId - Neuron ID
	 * @returns {number|undefined} Parent neuron ID
	 */
	getNeuronParent(neuronId) {
		return this.neuronParents.get(neuronId);
	}

	/**
	 * Get the level for a neuron (0 = sensory, 1+ = pattern)
	 * @param {number} neuronId - Neuron ID
	 * @returns {number} Neuron level
	 */
	getNeuronLevel(neuronId) {
		return this.neuronLevels.get(neuronId);
	}

	/**
	 * Get the coordinates for a sensory neuron
	 * @param {number} neuronId - Neuron ID
	 * @returns {object|undefined} Coordinate object with dimension-value pairs
	 */
	getNeuronCoordinates(neuronId) {
		return this.neuronValues.get(neuronId);
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
			const channel = this.getNeuronChannel(neuron.id);
			if (!channelOutputs.has(channel)) channelOutputs.set(channel, []);
			channelOutputs.get(channel).push({ coordinates: this.getNeuronCoordinates(neuron.id), strength, reward });
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
			const channel = this.getNeuronChannel(neuron.id);
			if (!result.has(channel)) result.set(channel, []);
			result.get(channel).push(this.getNeuronCoordinates(neuronId));
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
			inferences.push({ neuron, strength, channel: this.getNeuronChannel(neuron.id), type: this.getNeuronType(neuron.id), coordinates: this.getNeuronCoordinates(neuron.id) });
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
			const level = this.neuronLevels.get(neuron.id);
			const entry = { neuron, level };
			if (level === 0) {
				entry.channel = this.getNeuronChannel(neuron.id);
				entry.type = this.getNeuronType(neuron.id);
				entry.coordinates = this.getNeuronCoordinates(neuron.id);
			}
			const parentId = this.neuronParents.get(neuron.id);
			if (parentId) entry.parentId = parentId;
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
			for (const [patternId, ctx] of neuron.routingTable) {
				neuron.materializeChildStrength(patternId, currentFrame);
				ctx.lastActivationFrame = 0;
				const pattern = this.neurons.get(patternId);
				if (pattern) this.registerDeath(pattern.id, Math.ceil(ctx.activationStrength / neuron.patternForgetRate));
			}
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
	 * Process one level end-to-end: aggregate the level view, match patterns, create
	 * error-correction pattern neurons, and return all activations at level+1.
	 * @param {number} level - Current level being processed
	 * @param {Map<number, Map<number, object>>} levelNeurons - Active neurons at this level: neuronId -> age -> state (ages ascending)
	 * @param {number} memoryDepth - Current sliding-window depth (age count)
	 * @param {Array<Set<number>>} sensoryNeurons - Active sensory neuron ids by age (level 0)
	 * @param {Array<Map>} rewards - Rewards by age (rewards[0] = current frame)
	 * @param {Map<string, Set<number>>} channelActionIds - Action neuron IDs by channel
	 * @param {number} frameNumber - Current frame number
	 * @param {Set<number>} newErrorPatternIds - Accumulator of error pattern ids created this frame (mutated)
	 * @param {number} errorCorrectionThreshold - Event-error ratio above which a correction is created
	 * @returns {Array<{parentId, patternId, age, deathFrame}>} Activations to apply at level+1
	 */
	processLevel(level, levelNeurons, memoryDepth, sensoryNeurons, rewards, channelActionIds, frameNumber, newErrorPatternIds, errorCorrectionThreshold) {

		// pass 1: aggregate per-neuron work, build level context, pre-create corrections
		const { tasks, levelContext } = this.buildLevelTasks(
			level, levelNeurons, memoryDepth, sensoryNeurons, rewards,
			channelActionIds, newErrorPatternIds, errorCorrectionThreshold
		);

		// pass 2: dispatchFrame - one neuron.processFrame call per active neuron
		const results = this.dispatchFrame(tasks, levelContext, sensoryNeurons[0], rewards[0], channelActionIds, frameNumber);

		// pass 3: applyFrameResults - batch contextRef updates by target, register deaths, collect activations
		const activations = this.applyFrameResults(results);

		if (this.debug && activations.length > 0)
			console.log(`Level ${level}: ${activations.length} activations`,
				activations.map(a => `parent=${a.parentId}, age=${a.age}, pattern=${a.patternId}`).join('; '));

		return activations;
	}

	/**
	 * walk the active neurons at this level, aggregate per-neuron dispatch data,
	 * inline-build the shared level context, and pre-create error-correction pattern neurons
	 * for any (neuron, age) whose previous votes mismatched reality. New correction pattern
	 * ids are added to newErrorPatternIds (mutated).
	 * @returns {{tasks: Array<{neuron, activeAges, recognizerAges, corrections}>, levelContext: Context}}
	 */
	buildLevelTasks(level, levelNeurons, memoryDepth, sensoryNeurons, rewards, channelActionIds, newErrorPatternIds, errorCorrectionThreshold) {
		const tasks = [];
		const levelContext = new Context();
		for (const [neuronId, ageStates] of levelNeurons) {

			// skip action neurons for learning or contexts if the channel learns without them
			if (this.skipActionNeuron(neuronId)) continue;

			// skip error patterns created earlier this frame (their own level pass is suppressed)
			if (newErrorPatternIds.has(neuronId)) continue;

			// loop through the ages and determine the task needed for the neuron
			const task = this.getNeuronTask(neuronId, level, levelContext, ageStates, memoryDepth, sensoryNeurons, rewards, channelActionIds, errorCorrectionThreshold);

			// also return the created pattern neuron id so that we can suppress it in higher level
			for (const correction of task.corrections) newErrorPatternIds.add(correction.patternId);

			// add the task to the returned tasks
			tasks.push(task);
		}
		return { tasks, levelContext };
	}

	/**
	 * returns a task to be executed by a neuron
	 */
	getNeuronTask(neuronId, level, levelContext, ageStates, memoryDepth, sensoryNeurons, rewards, channelActionIds, errorCorrectionThreshold) {
		const learnerAges = [];
		const recognizerAges = [];
		const corrections = [];
		for (const [age, state] of ageStates) {

			// every active entry (age > 0) contributes to the shared level context
			if (age > 0) levelContext.addNeuron(neuronId, age);

			// add active ages for the connections to be learned - age=0 cannot learn - just got activated
			if (age > 0) learnerAges.push(age);

			// add ages to be used for pattern recognition - if there was
			if (!state.activatedPatternId && age !== memoryDepth - 1) recognizerAges.push(age);

			// create an error correction pattern for this (neuron, age) if previous votes mismatched reality
			if (this.needsErrorCorrection(age, state.votes, sensoryNeurons[0], errorCorrectionThreshold)) {

				// create the error correction pattern neuron
				const contextEntries = state.context.map(c => ({ neuronId: c.neuron.id, distance: c.distance }));
				const patternId = this.createPatternNeuron(level + 1, neuronId, age, sensoryNeurons, rewards, channelActionIds);
				corrections.push({ patternId, age, contextEntries });
			}
		}
		return { neuronId, learnerAges, recognizerAges, corrections };
	}

	/**
	 * Pass 2: dispatch exactly one neuron.processFrame call per active neuron.
	 * Returns raw results tagged with parentId for post-processing.
	 */
	dispatchFrame(tasks, levelContext, age0, currentRewards, channelActionIds, frameNumber) {

		// decorate age=0 sensory neurons with channel + pre-resolved reward (MPI-ready: neuron doesn't need type)
		const newActiveNeurons = [];
		for (const neuronId of age0) {
			const channel = this.getNeuronChannel(neuronId);
			const reward = this.getNeuronType(neuronId) === 'action' ? (currentRewards.get(channel) || 0) : 0;
			newActiveNeurons.push({ id: neuronId, channel, reward });
		}

		// call each neuron to deliver the tasks to process the frame
		const results = [];
		for (const { neuronId, learnerAges, recognizerAges, corrections } of tasks) {
			const result = this.neurons.get(neuronId).processFrame(
				learnerAges, recognizerAges, levelContext, newActiveNeurons,
				channelActionIds, frameNumber, corrections
			);
			results.push({ parentId: neuronId, ...result });
		}
		return results;
	}

	/**
	 * Pass 3: deliver contextRef updates (one call per target neuron), register deaths
	 * for both recognition matches and error-correction patterns, and collect the unified
	 * activation list to feed into level+1.
	 * @returns {Array<{parentId, patternId, age, deathFrame}>}
	 */
	applyFrameResults(results) {
		const perTarget = new Map(); // targetNeuronId -> [{type, parentId, distance}]
		const activations = [];
		for (const { parentId, matches, correctionActivations, contextRefUpdates } of results) {

			// batch contextRef updates by target neuron (delivered in the loop below)
			for (const { type, neuronId, distance } of contextRefUpdates) {
				let list = perTarget.get(neuronId);
				if (!list) { list = []; perTarget.set(neuronId, list); }
				list.push({ type, parentId, distance });
			}

			// register deaths + collect activations for recognition matches
			for (const match of matches)
				if (match.activate) {
					this.registerDeath(match.patternId, match.deathFrame);
					activations.push({ parentId, patternId: match.patternId, age: match.age, deathFrame: match.deathFrame });
				}

			// register deaths + collect activations for error-correction patterns
			for (const { patternId, age, deathFrame } of correctionActivations) {
				this.registerDeath(patternId, deathFrame);
				activations.push({ parentId, patternId, age, deathFrame });
			}
		}

		// deliver contextRef updates: one call per target neuron
		for (const [targetId, updates] of perTarget)
			this.neurons.get(targetId).applyContextRefUpdates(updates);

		return activations;
	}

	/**
	 * returns if a neuron needs error correction
	 * based on its inferences in the previous frame and the actual events in the current frame
	 * @returns {boolean} Whether error correction is needed
	 */
	needsErrorCorrection(age, votes, actualNeuronIds, errorCorrectionThreshold) {

		// age=0 neurons cannot need correction because they are just voting now
		if (age === 0) return false;

		// if there are no votes from previous frame, no need for error correction
		if (!votes || votes.length === 0) return false;

		// compare the inferred events to reality to determine if we need error correction
		let failedEvents = 0;
		let totalEvents = 0;
		for (const vote of votes)
			if (this.getNeuronType(vote.neuronId) === 'event') {
				totalEvents++;
				if (!actualNeuronIds.has(vote.neuronId)) failedEvents++;
			}
		const eventError = failedEvents / totalEvents;
		return eventError > errorCorrectionThreshold;
	}

	/**
	 * Check if a neuron should be skipped (action neuron in a channel without action sequences)
	 */
	skipActionNeuron(neuronId) {
		if (this.neuronLevels.get(neuronId) !== 0 || this.getNeuronType(neuronId) !== 'action') return false;
		const channel = this.channels.get(this.getNeuronChannel(neuronId));
		return channel && !channel.actionSequences;
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
	 * Get action neurons for a channel
	 * @param {string} channelName - Channel name
	 * @returns {Neuron} - Set of action neurons or undefined
	 */
	getChannelDefaultAction(channelName) {
		return this.channelDefaultActions.get(channelName);
	}

	/**
	 * Get channel action neuron IDs as a Map
	 * @returns {Map<string, Set<number>>} - Map of channel name to action neuron IDs
	 */
	getChannelActionIds() {
		const result = new Map();
		for (const [channelName, actionNeurons] of this.channelActions) {
			const ids = new Set();
			for (const id of actionNeurons) ids.add(id);
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
			const inferences = channelInferences.get(this.getNeuronChannel(inference.neuron.id));
			const type = this.getNeuronType(inference.neuron.id);
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
				actionNeurons.add(this.getNeuronIdForPoint(coordinates, channelName, 'action'));

			// add channel's action neurons to the channelActions map
			channelActions.set(channelName, actionNeurons);
			if (this.debug) console.log(`Created ${actionNeurons.size} action neurons for ${channelName}`);

			// set the default action for the channel (if one exists)
			const defaultActionCoords = channel.getDefaultAction();
			if (defaultActionCoords !== null) {
				const defaultAction = this.neurons.get(this.getNeuronIdForPoint(defaultActionCoords, channelName, 'action'));
				this.channelDefaultActions.set(channelName, defaultAction);
			}
		}

		// set channel actions for exploration
		this.channelActions = channelActions;
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
	registerDeath(neuronId, deathFrame) {
		// unregister old death frame if exists
		this.unregisterDeath(neuronId);

		// register new death frame
		if (!this.deathLedger.has(deathFrame)) this.deathLedger.set(deathFrame, new Set());
		this.deathLedger.get(deathFrame).add(neuronId);
		this.neuronDeathFrame.set(neuronId, deathFrame);
	}

	/**
	 * Unregister a neuron from the death ledger
	 */
	unregisterDeath(neuronId) {
		const oldFrame = this.neuronDeathFrame.get(neuronId);
		if (oldFrame === undefined) return;

		const set = this.deathLedger.get(oldFrame);
		if (set) {
			set.delete(neuronId);
			if (set.size === 0) this.deathLedger.delete(oldFrame);
		}
		this.neuronDeathFrame.delete(neuronId);
	}

	/**
	 * Reap neurons scheduled to die at the given frame.
	 * Assumes cleanup runs for every frame in order with no skips.
	 * Returns array of dead neurons and removes them from the ledger.
	 */
	reapDeadNeurons(currentFrame) {

		// get the neurons to be deleted in this frame
		const neuronIds = this.deathLedger.get(currentFrame);
		if (!neuronIds) return []; // nothing to do if no neurons dying

		// reap the dead neurons and return them
		const dead = [];
		for (const neuronId of neuronIds) {
			const neuron = this.neurons.get(neuronId);
			if (neuron) dead.push(neuron);
			this.neuronDeathFrame.delete(neuronId);
		}
		this.deathLedger.delete(currentFrame);
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

		// Clean up forward references: remove this pattern from its parent's context entries
		const newlyDeletable = this.cleanupPatternFromParentContext(pattern);

		// Clean up reverse references: remove this pattern from other parents' child contexts
		const moreNewlyDeletable = this.cleanupPatternFromChildContexts(pattern, currentFrame);
		newlyDeletable.push(...moreNewlyDeletable);

		// Clean up contextRefs on context neurons for all remaining children in this neuron's routing table.
		// When this parent is deleted, orphaned children will later be processed by deletePatterns.
		// Their cleanupPatternFromParentContext will skip (parent gone), so we must handle it here.
		this.cleanupOrphanedChildren(pattern);

		// Remove pattern from its parent's routing table
		// Parent may already be deleted during cascade — its deletePattern cleaned up all children
		this.deregisterFromParent(pattern);

		// Remove from death ledger
		this.unregisterDeath(pattern.id);

		// Delete this pattern neuron from the index and decrement level count
		this.removeFromIndexes(pattern);

		// Free memory - clear internal properties
		this.freeMemory(pattern);

		return newlyDeletable;
	}

	/**
	 * Remove this pattern from its parent's context entries
	 * @param {Neuron} pattern - Pattern being deleted
	 * @returns {Array<Neuron>} - Patterns that became deletable after cleanup
	 */
	cleanupPatternFromParentContext(pattern) {
		const newlyDeletable = [];
		const parentId = this.neuronParents.get(pattern.id);
		if (!parentId) throw new Error(`Cannot find parent of pattern for cleanup: ${pattern.id}`);
		const parentNeuron = this.neurons.get(parentId);

		// If parent was already deleted during cascade, its deletePattern already cleaned up
		// all remaining children's contextRefs — nothing left for us to do.
		if (parentNeuron)
			for (const entry of parentNeuron.getPatternContext(pattern.id)) {
				const ctxNeuron = this.neurons.get(entry.neuronId);
				if (!ctxNeuron) continue;
				const isOrphaned = parentNeuron.removeContext(pattern.id, entry.neuronId, entry.distance);
				if (isOrphaned) ctxNeuron.removeContextRef(parentNeuron.id, entry.distance);
			}

		return newlyDeletable;
	}

	/**
	 * Remove this pattern from other parents' children's contexts
	 * @param {Neuron} pattern - Pattern being deleted
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<Neuron>} - Patterns that became deletable after cleanup
	 */
	cleanupPatternFromChildContexts(pattern, currentFrame) {
		const newlyDeletable = [];

		// Iterate through all parents that have this pattern in their children's contexts
		for (const [referencingParentId, distances] of pattern.contextRefs) {
			const referencingParent = this.neurons.get(referencingParentId);
			if (!referencingParent) continue;

			// Remove this pattern from the parent's context and get affected child patterns
			const affectedPatterns = referencingParent.removeContextNeuron(pattern.id, distances);
			for (const patternId of affectedPatterns) {
				const childPattern = this.neurons.get(patternId);
				const parentId = this.neuronParents.get(patternId);
				const parent = this.neurons.get(parentId);
				if (childPattern && parent && parent.canDeleteChild(patternId, currentFrame, this.neuronLevels.get(patternId)))
					newlyDeletable.push(childPattern);
			}
		}

		return newlyDeletable;
	}

	/**
	 * Clean up context references for all children in this pattern's routing table
	 * (needed because their parent is being deleted)
	 * @param {Neuron} pattern - Pattern being deleted
	 */
	cleanupOrphanedChildren(pattern) {
		for (const [childPatternId, tableEntry] of pattern.routingTable)
			for (const entry of tableEntry.context.getEntries()) {
				const isOrphaned = pattern.removeContextIndex(entry.neuronId, entry.distance, childPatternId);
				if (isOrphaned) {
					const ctxNeuron = this.neurons.get(entry.neuronId);
					if (ctxNeuron) ctxNeuron.removeContextRef(pattern.id, entry.distance);
				}
			}
	}

	/**
	 * Remove pattern from its parent's routing table
	 * @param {Neuron} pattern - Pattern being deleted
	 */
	deregisterFromParent(pattern) {
		const parentId = this.neuronParents.get(pattern.id);
		if (!parentId) throw new Error(`Cannot find parent of pattern for deletion: ${pattern.id}`);
		const parentNeuron = this.neurons.get(parentId);
		if (parentNeuron) parentNeuron.removeChild(pattern.id);
	}

	/**
	 * Remove pattern from all neuron indexes and decrement level count
	 * @param {Neuron} pattern - Pattern being deleted
	 */
	removeFromIndexes(pattern) {
		const level = this.neuronLevels.get(pattern.id);
		this.neurons.delete(pattern.id);
		this.neuronLevels.delete(pattern.id);
		this.neuronParents.delete(pattern.id);
		this.decrementLevelCount(level);
	}

	/**
	 * Free memory by clearing pattern's internal properties
	 * @param {Neuron} pattern - Pattern being deleted
	 */
	freeMemory(pattern) {
		delete pattern.routingTable;
		delete pattern.contextIndex;
		delete pattern.contextRefs;
		delete pattern.connections;
		pattern = null;
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