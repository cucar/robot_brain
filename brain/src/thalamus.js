import { Context } from './context.js';
import { Neuron } from './neuron.js';
import { Quantizer } from './quantizer.js';

/**
 * Thalamus - Brain's relay station for reference frame transfers
 * 
 * Abstracts access to neurons, channels, and dimension mappings.
 * Handles bidirectional translation between external signals and internal neuron representations.
 * Named after the biological thalamus which routes sensory signals and translates reference frames.
 */
export class Thalamus {
	constructor(debug, patternForgetRate, mergeThreshold, errorCorrectionThreshold) {
		this.debug = debug;
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;
		this.errorCorrectionThreshold = errorCorrectionThreshold;

		// Neuron registry
		this.neurons = new Map(); // neuronId -> Neuron
		this.neuronsByValue = new Map(); // valueKey -> neuronId (coordinate -> neuronId lookup)
		this.baseNeurons = new Map(); // neuronId -> { channel, type, coordinate } (sensory neurons only)
		this.neuronParents = new Map(); // neuronId -> parentNeuronId (pattern neurons only)
		this.neuronLevels = new Map(); // neuronId -> level (0 = sensory, 1+ = pattern)

		// Death ledger - scheduled neuron deaths
		this.deathLedger = new Map(); // frameNumber -> Set<number>
		this.neuronDeathFrame = new Map(); // neuronId -> frameNumber (reverse lookup)

		// Channel / dimension registries populated by registerChannelSpec(). Specs are the
		// single source of truth for channel and dimension metadata.
		this.channelSpecs = new Map(); // channelId -> ChannelSpec
		this.dimensionSpecs = new Map(); // dimensionId -> DimSpec (flattened across all channels)
		this.channelActions = new Map(); // channelName -> Set<number> (neuron ids)
		this.channelDefaultActions = new Map(); // channelName -> Neuron id
		this.channelNameToId = {}; // channelName -> channelId
		this.channelIdToName = {}; // channelId -> channelName

		// Dimension mappings
		this.dimensionNameToId = {}; // dimensionName -> dimensionId
		this.dimensionIdToName = {}; // dimensionId -> dimensionName

		this.quantizer = new Quantizer();

		// ID allocators for channel and dimension IDs. Advanced past max when restoring
		// from a snapshot so newly allocated IDs don't collide with persisted ones.
		this.nextChannelId = 1;
		this.nextDimensionId = 1;

		// Level counts - tracks number of neurons at each level for efficient max level diagnostics lookup
		this.levelCounts = []; // index = level, value = count of neurons at that level
	}

	/**
	 * Create value key for neuron coordinate lookup.
	 * Coordinates are id-form: {dimId, bucketId}. The key is purely numeric content
	 * encoded as a string for Map hashing — handles negative bucketIds naturally.
	 */
	makeValueKey(coordinate) {
		return `${coordinate.dimId}:${coordinate.bucketId}`;
	}

	/**
	 * Translate an id-form coordinate to name-form for channel-facing code.
	 * @param {{dimId: number, bucketId: number}} coordinate
	 * @returns {{dimension: string, value: number}}
	 */
	coordinateIdToName(coordinate) {
		return { dimension: this.dimensionIdToName[coordinate.dimId], value: coordinate.bucketId };
	}

	/**
	 * Translate a name-form coordinate to id-form for internal brain use.
	 * @param {{dimension: string, value: number}} coordinate
	 * @returns {{dimId: number, bucketId: number}}
	 */
	coordinateNameToId(coordinate) {
		return { dimId: this.dimensionNameToId[coordinate.dimension], bucketId: coordinate.value };
	}

	/**
	 * Get or create a sensory neuron ID from a frame point. coordinate form: {dimId, bucketId }
	 * @returns {number} - Neuron ID
	 */
	getNeuronIdForPoint(coordinate, channel, type) {

		// Try to find existing neuron - if found, return it
		let neuronId = this.getNeuronIdByCoordinate(coordinate);
		if (neuronId) return neuronId;

		// Create new neuron if not found
		const neuron = this.addSensoryNeuron(coordinate, channel, type);
		if (this.debug) console.log(`Created new sensory neuron ${neuron.id} for ${this.makeValueKey(coordinate)}`);
		return neuron.id;
	}

	/**
	 * returns neuron ID by coordinate
	 * @param {{dimId: number, bucketId: number}} coordinate - id-form coordinate
	 * @returns {number|null} - Neuron ID or null if not found
	 */
	getNeuronIdByCoordinate(coordinate) {
		return this.neuronsByValue.get(this.makeValueKey(coordinate)) || null;
	}

	/**
	 * Create and add a new sensory neuron to the registry
	 * @param {{dimId: number, bucketId: number}} coordinate - id-form coordinate
	 * @param {string} channel - Channel name
	 * @param {string} type - Neuron type ('event' or 'action')
	 * @returns {Neuron} The newly created neuron
	 */
	addSensoryNeuron(coordinate, channel, type) {
		const neuron = new Neuron(this.patternForgetRate, this.mergeThreshold, this.channelActions);
		this.neurons.set(neuron.id, neuron);
		this.neuronLevels.set(neuron.id, 0);
		this.neuronsByValue.set(this.makeValueKey(coordinate), neuron.id);
		this.baseNeurons.set(neuron.id, { channel, type, coordinate });
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
	 * @returns {number} The newly created pattern neuron
	 */
	createPatternNeuron(level, parentId, age, sensoryNeurons, rewards) {

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
		const neuron = new Neuron(this.patternForgetRate, this.mergeThreshold, this.channelActions);
		neuron.initializeConnections(connections);

		// register in the thalamus
		this.neurons.set(neuron.id, neuron);
		this.neuronLevels.set(neuron.id, level);
		this.neuronParents.set(neuron.id, parentId);
		this.incrementLevelCount(level); // for diagnostics

		return neuron.id;
	}

	/**
	 * Restore brain state from a snapshot (same format as getSnapshot).
	 * @param {{neurons: Array, channelNameToId?: Object, dimensionNameToId?: Object}} snapshot
	 *
	 * Channel and dimension specs are expected to have been registered via
	 * registerChannelSpec() BEFORE restore — the snapshot only carries the persisted
	 * id↔name maps so we can reconcile allocated-vs-persisted IDs and advance the
	 * counters past whatever was on disk.
	 */
	restoreSnapshot(snapshot) {
		if (snapshot.channelNameToId)
			for (const [, id] of Object.entries(snapshot.channelNameToId))
				if (id >= this.nextChannelId) this.nextChannelId = id + 1;
		if (snapshot.dimensionNameToId)
			for (const [, id] of Object.entries(snapshot.dimensionNameToId))
				if (id >= this.nextDimensionId) this.nextDimensionId = id + 1;

		// Restore neurons
		this.reset();

		for (const { neuron, level, baseNeuron, parentId } of snapshot.neurons) {
			this.neurons.set(neuron.id, neuron);
			this.neuronLevels.set(neuron.id, level);
			if (parentId) this.neuronParents.set(neuron.id, parentId);
			this.incrementLevelCount(level);
			if (level === 0) {
				this.neuronsByValue.set(this.makeValueKey(baseNeuron.coordinate), neuron.id);
				this.baseNeurons.set(neuron.id, baseNeuron);
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
		this.baseNeurons.clear();
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
		return this.baseNeurons.get(neuronId)?.channel;
	}

	/**
	 * Get the type for a neuron ('event' or 'action')
	 * @param {number} neuronId - Neuron ID
	 * @returns {string} Neuron type
	 */
	getNeuronType(neuronId) {
		return this.baseNeurons.get(neuronId)?.type;
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
	 * Get the coordinate for a base (sensory/action) neuron.
	 * Throws if called for an interneuron — interneurons have no coordinate.
	 * @param {number} neuronId - Neuron ID
	 * @returns {{dimId: number, bucketId: number}} Id-form coordinate
	 */
	getNeuronCoordinate(neuronId) {
		const baseNeuron = this.baseNeurons.get(neuronId);
		if (!baseNeuron) throw new Error(`getNeuronCoordinate called for non-base neuron ${neuronId}`);
		return baseNeuron.coordinate;
	}

	/**
	 * Get inferred actions grouped by channel from the given inferences.
	 * @param {Array<{neuronId, strength, reward}>} inferences - Inferred neurons from memory
	 * @returns {Map<string, Array>} - Map of channel name to array of {coordinates, strength, reward}
	 */
	getInferredActions(inferences) {
		const channelOutputs = new Map();
		for (const { neuronId, strength, reward } of inferences) {
			if (this.getNeuronType(neuronId) !== 'action') continue;
			const channel = this.getNeuronChannel(neuronId);
			if (!channelOutputs.has(channel)) channelOutputs.set(channel, []);
			channelOutputs.get(channel).push({ coordinate: this.getNeuronCoordinate(neuronId), strength, reward });
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
			const baseNeuron = this.baseNeurons.get(neuronId);
			if (baseNeuron.type !== 'event') continue;
			if (!result.has(baseNeuron.channel)) result.set(baseNeuron.channel, []);
			result.get(baseNeuron.channel).push(baseNeuron.coordinate);
		}
		return result;
	}

	/**
	 * Get inferences with channel metadata attached.
	 * @param {Array<{neuronId, strength}>} inferredNeurons - Inferred neurons from memory
	 * @returns {Array<{neuronId, strength, channel, type, coordinates}>} - Inferences with channel
	 */
	getInferences(inferredNeurons) {
		const inferences = [];
		for (const { neuronId, strength } of inferredNeurons)
			inferences.push({ neuronId, strength, channel: this.getNeuronChannel(neuronId), type: this.getNeuronType(neuronId), coordinate: this.getNeuronCoordinate(neuronId) });
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
			if (level === 0) entry.baseNeuron = this.baseNeurons.get(neuron.id);
			const parentId = this.neuronParents.get(neuron.id);
			if (parentId) entry.parentId = parentId;
			neurons.push(entry);
		}
		return {
			neurons,
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
				this.registerDeath(patternId, Math.ceil(ctx.activationStrength / neuron.patternForgetRate));
			}
		}
	}


	/**
	 * Register a channel spec with the brain. The Thalamus allocates a channel ID and
	 * per-dimension IDs, populates the name↔id maps, registers each dimension with the
	 * quantizer, and pre-creates action neurons for action dims with explicit bucket IDs.
	 * Returns the allocated channel ID plus a `dimensionIds` lookup keyed by dim name so
	 * the caller can wire IDs into whatever owns the spec (encoder, trader).
	 *
	 * @param {object} spec
	 * @param {string} spec.name - Channel name (used as the channel key in the frame pipeline)
	 * @param {Array<object>} spec.dimensions - Per-dimension specs
	 * @param {string} spec.dimensions[].name - Dimension name (must be unique across all channels)
	 * @param {string} spec.dimensions[].kind - 'input' | 'action'
	 * @param {number} spec.dimensions[].resolution - Number of buckets (>= 2)
	 * @param {string} [spec.dimensions[].mode='passthrough'] - Quantizer mode
	 * @param {number[]} [spec.dimensions[].boundaries] - Static mode boundaries (length = resolution - 1)
	 * @param {number[]} [spec.dimensions[].actionBuckets] - For action dims: explicit bucket IDs to pre-create neurons for
	 * @param {number} [spec.dimensions[].defaultBucket] - For action dims: bucket ID of the channel's default action
	 * @param {number} [spec.dimensions[].warmupSamples] - Dynamic mode warmup window
	 * @param {boolean} [spec.emitsReward=false] - Channel produces a reward signal each frame
	 * @param {boolean} [spec.learnActionSequences=false] - Channel's action neurons participate in pattern learning
	 * @returns {{ channelId: number, dimensionIds: Object<string, number> }}
	 */
	registerChannelSpec(spec) {
		if (!spec.name) throw new Error('Thalamus: channel spec is missing required name');
		if (this.channelNameToId[spec.name] !== undefined)
			throw new Error(`Thalamus: channel "${spec.name}" already registered`);

		// Allocate the channel ID.
		const channelId = this.nextChannelId++;

		// Validate dim specs up front.
		for (const d of spec.dimensions) {
			if (!d.name) throw new Error(`Thalamus: dim on channel "${spec.name}" is missing a name`);
			if (d.kind !== 'input' && d.kind !== 'action')
				throw new Error(`Thalamus: dim "${d.name}" has invalid kind '${d.kind}' (expected 'input' or 'action')`);
		}

		// Build the stored spec with allocated IDs baked in.
		const dimensionIds = {};
		const storedSpec = {
			id: channelId,
			name: spec.name,
			dimensions: spec.dimensions.map(d => {
				const id = this.nextDimensionId++;
				dimensionIds[d.name] = id;
				return {
					id,
					name: d.name,
					kind: d.kind,
					resolution: d.resolution,
					mode: d.mode,
					boundaries: d.boundaries,
					actionBuckets: d.actionBuckets,
					defaultBucket: d.defaultBucket,
					warmupSamples: d.warmupSamples
				};
			}),
			emitsReward: spec.emitsReward ?? false,
			learnActionSequences: spec.learnActionSequences ?? false
		};
		this.channelSpecs.set(channelId, storedSpec);

		// Populate name↔id maps so the frame pipeline can resolve this channel by name.
		this.channelNameToId[spec.name] = channelId;
		this.channelIdToName[channelId] = spec.name;

		// Register each dimension: store spec, register with quantizer, populate dim name↔id maps.
		for (const dim of storedSpec.dimensions) {
			if (this.dimensionSpecs.has(dim.id))
				throw new Error(`Thalamus: dimension ${dim.id} already registered (channel "${spec.name}")`);
			this.dimensionSpecs.set(dim.id, dim);
			this.dimensionNameToId[dim.name] = dim.id;
			this.dimensionIdToName[dim.id] = dim.name;
			this.quantizer.registerDimension(dim.id, {
				resolution: dim.resolution,
				mode: dim.mode,
				boundaries: dim.boundaries,
				warmupSamples: dim.warmupSamples
			});
		}

		// Pre-create action neurons for action dims with explicit bucket IDs so exploration can find them.
		const actionNeurons = this.channelActions.get(spec.name) || new Set();
		for (const dim of storedSpec.dimensions) {
			if (dim.kind !== 'action' || !Array.isArray(dim.actionBuckets)) continue;
			for (const bucketId of dim.actionBuckets)
				actionNeurons.add(this.getNeuronIdForPoint({ dimId: dim.id, bucketId }, spec.name, 'action'));
			if (dim.defaultBucket !== undefined)
				this.channelDefaultActions.set(spec.name, this.getNeuronIdForPoint({ dimId: dim.id, bucketId: dim.defaultBucket }, spec.name, 'action'));
		}
		if (actionNeurons.size > 0) this.channelActions.set(spec.name, actionNeurons);

		if (this.debug) console.log(`Registered channel spec ${channelId} "${spec.name}" (${storedSpec.dimensions.length} dimensions)`);
		return { channelId, dimensionIds };
	}

	/**
	 * Iterate all channel names registered via channel specs.
	 * Used by the frame pipeline to drive buildFrame over every channel the brain knows about.
	 * @returns {string[]}
	 */
	getAllChannelNames() {
		return [...this.channelSpecs.values()].map(spec => spec.name);
	}

	/**
	 * Get stored channel spec by ID.
	 */
	getChannelSpec(channelId) {
		return this.channelSpecs.get(channelId);
	}

	/**
	 * Get stored dimension spec by ID.
	 */
	getDimensionSpec(dimensionId) {
		return this.dimensionSpecs.get(dimensionId);
	}

	/**
	 * Process one level end-to-end: aggregate the level view, match patterns, create
	 * error-correction pattern neurons, collect votes, and return activations + votes.
	 * @param {number} level - Current level being processed
	 * @param {Map<number, Map<number, object>>} levelNeurons - Active neurons at this level: neuronId -> age -> state (ages ascending)
	 * @param {number} memoryDepth - Current sliding-window depth (age count)
	 * @param {Array<Set<number>>} sensoryNeurons - Active sensory neuron ids by age (level 0)
	 * @param {Array<Map>} rewards - Rewards by age (rewards[0] = current frame)
	 * @param {number} frameNumber - Current frame number
	 * @param {Set<number>} newErrorPatternIds - Accumulator of error pattern ids created this frame (mutated)
	 * @returns {{activations: Array<{parentId, patternId, age, deathFrame}>, votes: Array}}
	 */
	processLevel(level, levelNeurons, memoryDepth, sensoryNeurons, rewards, frameNumber, newErrorPatternIds) {

		// pass 1: aggregate per-neuron work, build the shared level context, pre-create corrections.
		// Per-age task derivation and per-age context reshape both happen inside the neuron —
		// the only cross-neuron data shipped is the shared levelContext and the small
		// newErrorPatternIds set (used by the neuron to mask brand-new ids out of matching).
		const { tasks, levelContext } = this.getLevelTasks(level, levelNeurons, sensoryNeurons, rewards, newErrorPatternIds);

		// pass 2: dispatchFrame - one neuron.processFrame call per active neuron (learn, match, correct, vote)
		const results = this.dispatchFrame(tasks, memoryDepth, levelContext, newErrorPatternIds, sensoryNeurons[0], rewards[0], frameNumber);

		// pass 3: applyFrameResults - batch contextRef updates by target, register deaths, collect activations + votes
		const { activations, votes } = this.applyFrameResults(results, levelNeurons);

		if (this.debug && activations.length > 0)
			console.log(`Level ${level}: ${activations.length} activations`,
				activations.map(a => `parent=${a.parentId}, age=${a.age}, pattern=${a.patternId}`).join('; '));

		return { activations, votes };
	}

	/**
	 * Walk the active neurons at this level, contribute to the shared level context,
	 * pre-create error-correction pattern neurons for any (neuron, age) whose previous votes
	 * mismatched reality, and emit a task per neuron. New correction pattern ids are added to
	 * newErrorPatternIds (mutated); error patterns created earlier this frame receive
	 * voting-only tasks (flagged via isNewErrorPattern).
	 *
	 * Per-age task derivation (learner/recognizer/voting ages) now happens inside
	 * neuron.processFrame from its own ageStates — this pass only does cross-neuron work
	 * (levelContext merge, id allocation for corrections) that must stay on the driver.
	 * @returns {{tasks: Array<{neuronId, ageStates, corrections}>, levelContext: Context}}
	 */
	getLevelTasks(level, levelNeurons, sensoryNeurons, rewards, newErrorPatternIds) {
		const tasks = [];
		const levelContext = new Context();
		for (const [neuronId, ageStates] of levelNeurons) {

			// skip action neurons for learning or contexts if the channel learns without them
			if (this.skipActionNeuron(neuronId)) continue;

			// get level error corrections
			const corrections = this.getLevelCorrections(
				neuronId, level, levelContext, ageStates, sensoryNeurons, rewards, newErrorPatternIds.has(neuronId)
			);

			// also return the created pattern neuron id so that we can suppress it in higher level
			for (const correction of corrections) newErrorPatternIds.add(correction.patternId);

			// emit the task - the neuron derives its own isNewErrorPattern from newErrorPatternIds
			tasks.push({ neuronId, ageStates, corrections });
		}
		return { tasks, levelContext };
	}

	/**
	 * For a single active neuron: add its age>0 entries to the shared levelContext and create
	 * error-correction pattern neurons for ages whose previous votes mismatched reality.
	 * Every age>0 entry contributes to levelContext, including new error patterns — their ids
	 * must propagate into downstream state.context for next-frame corrections. The neuron
	 * itself filters out newErrorPatternIds at match time so brand-new ids don't look like
	 * unexplained novel entries and unfairly penalize pattern scores.
	 * Correction id allocation stays central — the driver owns the neuron-id counter.
	 * @returns {Array<{patternId, age, contextEntries}>} corrections created for this neuron
	 */
	getLevelCorrections(neuronId, level, levelContext, ageStates, sensoryNeurons, rewards, isNewErrorPattern) {
		const corrections = [];
		for (const [age, state] of ageStates) {

			// every age > 0 entry contributes to the shared level context
			if (age > 0) levelContext.addNeuron(neuronId, age);

			// new error patterns skip further correction this frame
			if (isNewErrorPattern) continue;

			// create an error correction pattern for this (neuron, age) if previous votes mismatched reality
			if (this.needsErrorCorrection(age, state.votes, sensoryNeurons[0]))
				corrections.push({
					patternId: this.createPatternNeuron(level + 1, neuronId, age, sensoryNeurons, rewards),
					age,
					contextEntries: state.context.map(c => ({ neuronId: c.neuronId, distance: c.distance }))
				});
		}
		return corrections;
	}

	/**
	 * Pass 2: dispatch exactly one neuron.processFrame call per active neuron.
	 * Returns raw results tagged with parentId for post-processing.
	 */
	dispatchFrame(tasks, memoryDepth, levelContext, newErrorPatternIds, age0, currentRewards, frameNumber) {

		// decorate age=0 sensory neurons with channel + pre-resolved reward (MPI-ready: neuron doesn't need type)
		const newActiveNeurons = [];
		for (const neuronId of age0) {
			const channel = this.getNeuronChannel(neuronId);
			const reward = this.getNeuronType(neuronId) === 'action' ? (currentRewards.get(channel) || 0) : 0;
			newActiveNeurons.push({ id: neuronId, channel, reward });
		}

		// call each neuron to deliver the tasks to process the frame
		const results = [];
		for (const { neuronId, ageStates, corrections } of tasks) {
			const result = this.neurons.get(neuronId).processFrame(
				ageStates, memoryDepth, levelContext, newErrorPatternIds,
				newActiveNeurons, frameNumber, corrections
			);
			results.push({ parentId: neuronId, ...result });
		}
		return results;
	}

	/**
	 * Pass 3: deliver contextRef updates (one call per target neuron), register deaths
	 * for both recognition matches and error-correction patterns, collect the unified
	 * activation list to feed into level+1, and flush per-age votes onto neuron state
	 * while building the flat vote array for consensus. Clears state.votes/state.context
	 * for every level neuron first so suppressed ages don't retain stale data.
	 * @returns {{activations: Array<{parentId, patternId, age, deathFrame}>, votes: Array}}
	 */
	applyFrameResults(results, levelNeurons) {
		const perTarget = new Map(); // targetNeuronId -> [{type, parentId, distance}]
		const activations = [];
		const votes = [];

		// clear stale votes/context for this level's neurons so suppressed ages don't carry stale data
		for (const ageStates of levelNeurons.values())
			for (const state of ageStates.values()) {
				state.votes = null;
				state.context = null;
			}

		for (const { parentId, matches, correctionActivations, contextRefUpdates, votes: perAgeVotes } of results) {

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

			// flush per-age votes: write state.votes/state.context for storage, collect id-only votes for consensus
			if (perAgeVotes && perAgeVotes.length > 0) {
				const ageStates = levelNeurons.get(parentId);
				for (const { age, votes: ageVotes, context } of perAgeVotes) {
					const state = ageStates.get(age);
					state.votes = ageVotes;
					state.context = context;
					for (const vote of ageVotes) votes.push({ voterId: parentId, ...vote });
				}
			}
		}

		// deliver contextRef updates: one call per target neuron
		for (const [targetId, updates] of perTarget)
			this.neurons.get(targetId).applyContextRefUpdates(updates);

		return { activations, votes };
	}

	/**
	 * returns if a neuron needs error correction
	 * based on its inferences in the previous frame and the actual events in the current frame
	 * @returns {boolean} Whether error correction is needed
	 */
	needsErrorCorrection(age, votes, actualNeuronIds) {

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
		return eventError > this.errorCorrectionThreshold;
	}

	/**
	 * Check if a neuron should be skipped (action neuron in a channel whose spec does
	 * not include action-sequence learning).
	 */
	skipActionNeuron(neuronId) {
		if (this.neuronLevels.get(neuronId) !== 0 || this.getNeuronType(neuronId) !== 'action') return false;
		const channelId = this.channelNameToId[this.getNeuronChannel(neuronId)];
		const spec = channelId !== undefined ? this.channelSpecs.get(channelId) : null;
		return spec ? !spec.learnActionSequences : false;
	}

	/**
	 * Get default action neuron id for a channel
	 * @param {string} channelName - Channel name
	 * @returns {number|undefined} - Default action neuron id or undefined
	 */
	getChannelDefaultAction(channelName) {
		return this.channelDefaultActions.get(channelName);
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
	 * Returns array of dead neuron ids and removes them from the ledger.
	 * @returns {Array<number>}
	 */
	reapDeadNeurons(currentFrame) {

		// get the neurons to be deleted in this frame
		const neuronIds = this.deathLedger.get(currentFrame);
		if (!neuronIds) return []; // nothing to do if no neurons dying

		// reap the dead neuron ids and return them
		const dead = [];
		for (const neuronId of neuronIds) {
			if (this.neurons.has(neuronId)) dead.push(neuronId);
			this.neuronDeathFrame.delete(neuronId);
		}
		this.deathLedger.delete(currentFrame);
		return dead;
	}

	/**
	 * Delete dead pattern neurons with recursive cascade.
	 * When a pattern is deleted, other patterns that referenced it may become deletable too.
	 * @param {Array<number>} patternIds - Initial list of pattern ids to delete
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<number>} - All deleted pattern ids
	 */
	deletePatterns(patternIds, currentFrame) {
		const toDelete = [...patternIds];
		const queuedIds = new Set(patternIds);
		const deletedIds = [];
		const deletedIdSet = new Set();

		while (toDelete.length > 0) {
			const patternId = toDelete.shift();
			if (deletedIdSet.has(patternId)) continue;

			const newlyDeletable = this.deletePattern(patternId, currentFrame);
			deletedIds.push(patternId);
			deletedIdSet.add(patternId);

			for (const newlyDeletableId of newlyDeletable) {
				if (deletedIdSet.has(newlyDeletableId) || queuedIds.has(newlyDeletableId)) continue;
				toDelete.push(newlyDeletableId);
				queuedIds.add(newlyDeletableId);
			}
		}

		return deletedIds;
	}

	/**
	 * Delete a pattern neuron and clean up all references to it.
	 * Returns pattern ids that became deletable as a result of cleanup.
	 * @param {number} patternId - Id of the pattern to delete
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<number>} - Pattern ids that became deletable after cleanup
	 */
	deletePattern(patternId, currentFrame) {

		// ignore double delete requests - also the single cleanup-path lookup for this neuron
		const pattern = this.neurons.get(patternId);
		if (!pattern) return [];

		// Clean up forward references: remove this pattern from its parent's context entries
		const newlyDeletable = this.cleanupPatternFromParentContext(patternId);

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
		this.unregisterDeath(patternId);

		// Delete this pattern neuron from the index and decrement level count
		this.removeFromIndexes(pattern);

		// Free memory - clear internal properties
		this.freeMemory(pattern);

		return newlyDeletable;
	}

	/**
	 * Remove this pattern from its parent's context entries
	 * @param {number} patternId - Pattern being deleted
	 * @returns {Array<number>} - Pattern ids that became deletable after cleanup
	 */
	cleanupPatternFromParentContext(patternId) {
		const newlyDeletable = [];
		const parentId = this.neuronParents.get(patternId);
		if (!parentId) throw new Error(`Cannot find parent of pattern for cleanup: ${patternId}`);
		const parentNeuron = this.neurons.get(parentId);

		// If parent was already deleted during cascade, its deletePattern already cleaned up
		// all remaining children's contextRefs — nothing left for us to do.
		if (parentNeuron)
			for (const entry of parentNeuron.getPatternContext(patternId)) {
				const ctxNeuron = this.neurons.get(entry.neuronId);
				if (!ctxNeuron) continue;
				const isOrphaned = parentNeuron.removeContext(patternId, entry.neuronId, entry.distance);
				if (isOrphaned) ctxNeuron.removeContextRef(parentNeuron.id, entry.distance);
			}

		return newlyDeletable;
	}

	/**
	 * Remove this pattern from other parents' children's contexts
	 * @param {Neuron} pattern - Pattern being deleted
	 * @param {number} currentFrame - Current frame number for lazy decay checks
	 * @returns {Array<number>} - Pattern ids that became deletable after cleanup
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
				if (!this.neurons.has(patternId)) continue;
				const parent = this.neurons.get(this.neuronParents.get(patternId));
				if (parent && parent.canDeleteChild(patternId, currentFrame, this.neuronLevels.get(patternId)))
					newlyDeletable.push(patternId);
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