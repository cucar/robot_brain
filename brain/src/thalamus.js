import { Context } from './context.js';
import { Neuron } from './neuron.js';
import { Quantizer } from './quantizer.js';
import { Region } from './region.js';

/**
 * Thalamus - Brain's relay station for reference frame transfers
 * 
 * Abstracts access to neurons, channels, and dimension mappings.
 * Handles bidirectional translation between external signals and internal neuron representations.
 * Named after the biological thalamus which routes sensory signals and translates reference frames.
 */
export class Thalamus {
	constructor(debug, patternForgetRate, mergeThreshold, contextLength, errorMode, errorThreshold, regions = 1, columns = 1) {
		this.debug = debug;
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;
		this.contextLength = contextLength;
		this.errorMode = errorMode;
		this.errorThreshold = errorThreshold;
		this.regions = regions;
		this.columns = columns;

		// Neuron registry — Thalamus keeps a flat map for non-Op-4 code paths (delete
		// cascade, snapshot, contextRef delivery) that haven't been migrated to batch ops
		// yet. Column.neurons holds the same references for Op-4 dispatch. Kept in sync
		// via _syncToColumn/_removeFromColumn. Both go away when §3.11 lands.
		this.neurons = new Map(); // neuronId -> Neuron (flat, all neurons)
		this.neuronsByValue = new Map(); // valueKey -> neuronId (coordinate -> neuronId lookup)
		this.baseNeurons = new Map(); // neuronId -> { channelId, type, coordinate } (sensory neurons only)
		this.neuronParents = new Map(); // neuronId -> parentNeuronId (pattern neurons only)
		this.neuronLevels = new Map(); // neuronId -> level (0 = sensory, 1+ = pattern)

		// Death ledger - scheduled neuron deaths
		this.deathLedger = new Map(); // frameNumber -> Set<number>
		this.neuronDeathFrame = new Map(); // neuronId -> frameNumber (reverse lookup)

		// Channel / dimension registries populated by registerChannelSpec(). Specs are the
		// single source of truth for channel and dimension metadata.
		this.channelSpecs = new Map(); // channelId -> ChannelSpec
		this.dimensionSpecs = new Map(); // dimensionId -> DimSpec (flattened across all channels)
		this.channelActions = new Map(); // channelId -> Set<number> (neuron ids)
		this.actionIds = new Set(); // flat union of all action neuron ids — shared by-reference with neurons for O(1) isActionNeuron
		this.channelDefaultActions = new Map(); // channelId -> Neuron id
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
		this.nextNeuronId = 1;

		// Level counts - tracks number of neurons at each level for efficient max level diagnostics lookup
		this.levelCounts = []; // index = level, value = count of neurons at that level

		// Construct the Region[R] tree. Each Region constructs its Column[C]. Columns are
		// handed the action sets they need so per-frame calls don't reach back to Thalamus.
		// In single-process JS the Maps/Sets are shared-by-reference with Thalamus so
		// channel registration (which mutates them after this point) is reflected in every
		// column. In Phase 5 these become per-thread copies materialized at init.
		this.regionList = []; // index = regionIdx
		for (let r = 0; r < this.regions; r++)
			this.regionList.push(new Region(this.columns, this.channelActions, this.actionIds, this.channelDefaultActions, this.mergeThreshold, this.errorMode, this.errorThreshold));
	}

	// ── Internal sync: keep Column.neurons in lockstep with this.neurons ────
	// These are private plumbing — Region/Column never expose single-neuron APIs.

	_syncToColumn(id, neuron) {
		const r = this.routeNeuron(id);
		const c = this.regionList[r].routeNeuron(id);
		this.regionList[r].columns[c].neurons.set(id, neuron);
	}

	_removeFromColumn(id) {
		const r = this.routeNeuron(id);
		const c = this.regionList[r].routeNeuron(id);
		this.regionList[r].columns[c].neurons.delete(id);
	}

	_clearColumns() {
		for (const r of this.regionList)
			for (const col of r.columns) col.neurons.clear();
	}

	/**
	 * Effective forget rate for a neuron at a given level.
	 * Level 0 (sensory) is exempt from forgetting; level N decays by contextLength per level
	 * relative to the level-1 base rate, matching the geometric drop in observation frequency.
	 */
	static effectiveForgetRate(baseRate, contextLength, level) {
		if (level === 0) return 0;
		return baseRate / Math.pow(contextLength, level - 1);
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
	 * Get or create a sensory neuron ID from a frame point. coordinate form: {dimId, bucketId }
	 * @returns {{id: number, isNew: boolean}} id and whether a new neuron was allocated
	 */
	getNeuronIdForPoint(coordinate, channelId, type) {

		// Try to find existing neuron - if found, return it
		const neuronId = this.getNeuronIdByCoordinate(coordinate);
		if (neuronId) return { id: neuronId, isNew: false };

		// Allocate id and register metadata; Neuron construction deferred to createNeuronsInColumns
		const id = this.allocateSensoryNeuron(coordinate, channelId, type);
		if (this.debug) console.log(`Created new sensory neuron ${id} for ${this.makeValueKey(coordinate)}`);
		return { id, isNew: true };
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
	 * Pure deterministic region-routing function. Maps a neuron id to its owning region.
	 * Interleaving (rather than chunking by id range) keeps id-bursts spread evenly:
	 * the error-correction pattern ids allocated in one frame spread evenly across regions
	 * instead of piling onto one. Region.routeNeuron picks up where this leaves off and
	 * chooses a column within that region.
	 * Stable for the lifetime of a running process; snapshots do NOT persist R or C, so
	 * a brain saved with one (R, C) reroutes through the current run's on load.
	 * @param {number} neuronId
	 * @returns {number} regionIdx
	 */
	routeNeuron(neuronId) {
		return neuronId % this.regions;
	}

	/**
	 * Bucket a flat batch by owning region using a key extractor.
	 * Returns an array indexed by regionIdx, each entry the sub-list for that region.
	 */
	bucketByRegion(batch, key) {
		const buckets = Array.from({ length: this.regions }, () => []);
		for (const item of batch) buckets[this.routeNeuron(item[key])].push(item);
		return buckets;
	}

	/**
	 * Allocate a sensory neuron id and register its metadata. Does NOT construct
	 * the Neuron — that happens when the caller sends specs to createNeuronsInColumns.
	 * @returns {number} The allocated neuron id
	 */
	allocateSensoryNeuron(coordinate, channelId, type) {
		const id = this.nextNeuronId++;
		this.neuronLevels.set(id, 0);
		this.neuronsByValue.set(this.makeValueKey(coordinate), id);
		this.baseNeurons.set(id, { channelId, type, coordinate });
		this.incrementLevelCount(0);
		return id;
	}

	/**
	 * Allocate a pattern neuron id and build its creation spec. Resolves connection
	 * data (channel, reward) using Thalamus-local lookups so the spec is self-contained
	 * and can cross the MPI boundary. Does NOT construct the Neuron — that happens in
	 * Column.createNewNeurons via createNeuronsInColumns.
	 * Does NOT touch the parent's routing table (that happens inside parent.processFrame
	 * via addPattern) and does NOT register death (death frame is known only after
	 * parent.addPattern runs).
	 * @param {number} level - Neuron level (1+ = pattern)
	 * @param {number} parentId - Parent neuron ID
	 * @param {number} age - Distance in time between the observation and the error
	 * @param {Array<Set<number>>} sensoryNeurons - Recent sensory neuron ids by age
	 * @param {Array<Map<string, number>>} rewards - Rewards by age
	 * @returns {{id, forgetRate, connections}} creation spec for Column.createNewNeurons
	 */
	allocatePatternNeuron(level, parentId, age, sensoryNeurons, rewards) {

		// resolve connection spec using thalamus-local lookups (channel, reward)
		const connections = [];
		for (let a = 0; a < age && a < sensoryNeurons.length; a++)
			for (const sensoryNeuronId of sensoryNeurons[a]) {
				const channelId = this.getNeuronChannelId(sensoryNeuronId);
				const reward = rewards[a].get(channelId) || 0;
				connections.push({ distance: age - a, toNeuronId: sensoryNeuronId, channelId, reward });
			}

		// allocate id and build the spec for Column.createNewNeurons
		const id = this.nextNeuronId++;
		const forgetRate = Thalamus.effectiveForgetRate(this.patternForgetRate, this.contextLength, level);

		// register metadata centrally (Neuron construction deferred to createNeuronsInColumns)
		this.neuronLevels.set(id, level);
		this.neuronParents.set(id, parentId);
		this.incrementLevelCount(level);

		return { id, forgetRate, connections };
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
			if (neuron.id >= this.nextNeuronId) this.nextNeuronId = neuron.id + 1;
			this.neurons.set(neuron.id, neuron);
			this._syncToColumn(neuron.id, neuron);
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
		this._clearColumns();
		this.neuronLevels.clear();
		this.neuronsByValue.clear();
		this.baseNeurons.clear();
		this.neuronParents.clear();
		this.deathLedger.clear();
		this.neuronDeathFrame.clear();
		this.levelCounts = []; // for diagnostics
		this.nextNeuronId = 1;
	}

	/**
	 * Get the channel id for a neuron
	 * @param {number} neuronId - Neuron ID
	 * @returns {number} Channel id
	 */
	getNeuronChannelId(neuronId) {
		return this.baseNeurons.get(neuronId)?.channelId;
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
	 * @returns {Map<number, Array>} - Map of channel id to array of {coordinates, strength, reward}
	 */
	getInferredActions(inferences) {
		const channelOutputs = new Map();
		for (const { neuronId, strength, reward } of inferences) {
			if (this.getNeuronType(neuronId) !== 'action') continue;
			const channelId = this.getNeuronChannelId(neuronId);
			if (!channelOutputs.has(channelId)) channelOutputs.set(channelId, []);
			channelOutputs.get(channelId).push({ coordinate: this.getNeuronCoordinate(neuronId), strength, reward });
		}
		return channelOutputs;
	}

	/**
	 * Per-frame inference performance bundle for diagnostics. Each item carries
	 * everything trackInferencePerformance needs (correctness flag, first-actual
	 * coord, pre-resolved reward) so the diagnostics call stays single-arg.
	 * @param {Iterable<number>} activeNeuronIds - Sensory neuron ids active at age 0
	 * @param {Array<{neuronId, strength}>} inferredNeurons - Predictions cast last frame
	 * @param {Map<number, number>} rewards - channelId → reward for previous frame's actions
	 * @returns {Array<{type:string, isCorrect:boolean, channelId:number, coordinate:object,
	 *                  actualCoord:object|null, reward:number}>}
	 */
	getInferenceResults(activeNeuronIds, inferredNeurons, rewards) {
		const activeSet = activeNeuronIds instanceof Set ? activeNeuronIds : new Set(activeNeuronIds);

		// per-channel first observed event coordinate (used to pair mispredictions)
		const firstActualByChannel = new Map();
		for (const neuronId of activeSet) {
			const base = this.baseNeurons.get(neuronId);
			if (!base || base.type !== 'event') continue;
			if (!firstActualByChannel.has(base.channelId)) firstActualByChannel.set(base.channelId, base.coordinate);
		}

		const results = [];
		for (const { neuronId } of inferredNeurons) {
			const type = this.getNeuronType(neuronId);
			const channelId = this.getNeuronChannelId(neuronId);
			results.push({
				type,
				channelId,
				coordinate: this.getNeuronCoordinate(neuronId),
				isCorrect: type === 'event' ? activeSet.has(neuronId) : false,
				actualCoord: type === 'event' ? (firstActualByChannel.get(channelId) ?? null) : null,
				reward: type === 'action' ? (rewards?.get(channelId) ?? 0) : 0
			});
		}
		return results;
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
	 * @param {number[]} [spec.dimensions[].actions] - For action dims: explicit bucket IDs to pre-create neurons for
	 * @param {number} [spec.dimensions[].defaultAction] - For action dims: bucket ID of the channel's default action
	 * @param {number} [spec.dimensions[].warmupSamples] - Dynamic mode warmup window
	 * @param {boolean} [spec.emitsReward=false] - Channel produces a reward signal each frame
	 * @param {boolean} [spec.learnActionSequences=false] - Channel's action neurons participate in pattern learning
	 * @returns {{ channelId: number, dimensionIds: Object<string, number> }}
	 */
	registerChannelSpec(spec) {
		if (!spec.name) throw new Error('Thalamus: channel spec is missing required name');
		if (this.channelNameToId[spec.name] !== undefined)
			throw new Error(`Thalamus: channel "${spec.name}" already registered`);

		// allocate channel id and validate dimension specs
		const channelId = this.nextChannelId++;
		this.validateDimSpecs(spec);

		// build the stored spec with allocated dimension ids baked in
		const { storedSpec, dimensionIds } = this.buildStoredSpec(channelId, spec);
		this.channelSpecs.set(channelId, storedSpec);

		// populate name↔id maps so the frame pipeline can resolve this channel by name
		this.channelNameToId[spec.name] = channelId;
		this.channelIdToName[channelId] = spec.name;

		// register dimensions with the quantizer and create action neurons in columns
		this.registerDimensions(storedSpec);
		this.registerActionNeurons(channelId, storedSpec);

		if (this.debug) console.log(`Registered channel spec ${channelId} "${spec.name}" (${storedSpec.dimensions.length} dimensions)`);
		return { channelId, dimensionIds };
	}

	validateDimSpecs(spec) {
		for (const d of spec.dimensions) {
			if (!d.name) throw new Error(`Thalamus: dim on channel "${spec.name}" is missing a name`);
			if (d.kind !== 'input' && d.kind !== 'action')
				throw new Error(`Thalamus: dim "${d.name}" has invalid kind '${d.kind}' (expected 'input' or 'action')`);
		}
	}

	/**
	 * Allocate dimension ids and build the immutable stored spec for a channel.
	 */
	buildStoredSpec(channelId, spec) {
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
					actions: d.actions,
					defaultAction: d.defaultAction,
					warmupSamples: d.warmupSamples
				};
			}),
			emitsReward: spec.emitsReward ?? false,
			learnActionSequences: spec.learnActionSequences ?? false
		};
		return { storedSpec, dimensionIds };
	}

	/**
	 * Store dimension specs, register with quantizer, and populate dim name↔id maps.
	 */
	registerDimensions(storedSpec) {
		for (const dim of storedSpec.dimensions) {
			if (this.dimensionSpecs.has(dim.id))
				throw new Error(`Thalamus: dimension ${dim.id} already registered (channel "${storedSpec.name}")`);
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
	}

	/**
	 * Pre-create action neurons for action dims so exploration can find them.
	 * actionIds is the flat union — kept in lockstep with per-channel sets so
	 * neurons can do O(1) isActionNeuron lookups.
	 */
	registerActionNeurons(channelId, storedSpec) {
		const actionNeurons = this.channelActions.get(channelId) || new Set();
		const newNeuronSpecs = [];
		for (const dim of storedSpec.dimensions) {

			// only processing action dimensions
			if (dim.kind !== 'action' || !Array.isArray(dim.actions)) continue;

			// all action dimensions should have defaults
			if (!dim.defaultAction) throw new Error(`Invalid action dimension without default: ${JSON.stringify(dim)}`);

			// allocate action neuron ids
			for (const bucketId of dim.actions) {
				const { id, isNew } = this.getNeuronIdForPoint({ dimId: dim.id, bucketId }, channelId, 'action');
				actionNeurons.add(id);
				this.actionIds.add(id);
				if (isNew) newNeuronSpecs.push({ id, forgetRate: 0 });
			}

			// default action neuron already created in the actions loop above - get its id and save it
			const { id } = this.getNeuronIdForPoint({ dimId: dim.id, bucketId: dim.defaultAction }, channelId, 'action');
			this.channelDefaultActions.set(channelId, id);
		}

		// index the action neurons
		if (actionNeurons.size > 0) this.channelActions.set(channelId, actionNeurons);

		// important: create action neurons in columns in parallel
		if (newNeuronSpecs.length > 0) this.createNeuronsInColumns(newNeuronSpecs);
	}

	/**
	 * Iterate all channel ids registered via channel specs.
	 * Used by the frame pipeline to drive buildFrame over every channel the brain knows about.
	 * @returns {number[]}
	 */
	getChannelIds() {
		return [...this.channelSpecs.keys()];
	}

	/**
	 * Get stored channel spec by ID.
	 */
	getChannelSpec(channelId) {
		return this.channelSpecs.get(channelId);
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

		// aggregate per-neuron work, build the shared level context, allocate error pattern specs.
		// Per-age task derivation and per-age context reshape both happen inside the neuron.
		// the only cross-neuron data shipped is the shared levelContext and the small
		// newErrorPatternIds set (used by the neuron to mask brand-new ids out of matching).
		const { tasks, levelContext, newNeuronSpecs } = this.getLevelTasks(level, levelNeurons, sensoryNeurons, rewards, newErrorPatternIds);

		// Op-3 create new neurons: construct the new error pattern neurons in their owning columns.
		// Must complete before Op-4 so that processFrame can access the new neurons.
		if (newNeuronSpecs.length > 0) this.createNeuronsInColumns(newNeuronSpecs);

		// dispatchFrame - one neuron.processFrame call per active neuron (learn, match, correct, vote)
		const results = this.dispatchFrame(tasks, memoryDepth, levelContext, newErrorPatternIds, sensoryNeurons[0], rewards[0], frameNumber);

		// applyFrameResults - batch contextRef updates by target, register deaths, collect activations + votes
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
		const newNeuronSpecs = [];
		for (const [neuronId, ageStates] of levelNeurons) {

			// skip action neurons for learning or contexts if the channel learns without them
			if (this.skipActionNeuron(neuronId)) continue;

			// get level error corrections + per-age feedback for the neuron's own stats
			const { corrections, errorFeedback } = this.getLevelCorrections(
				neuronId, level, levelContext, ageStates, sensoryNeurons, rewards, newErrorPatternIds.has(neuronId)
			);

			// extract creation specs for Op-3, suppress new ids in higher levels
			for (const correction of corrections) {
				newNeuronSpecs.push({ id: correction.id, forgetRate: correction.forgetRate, connections: correction.connections });
				newErrorPatternIds.add(correction.patternId);
			}

			// emit the task - the neuron derives its own isNewErrorPattern from newErrorPatternIds.
			// errorFeedback rides in-band on the same per-frame call (no extra MPI round-trip).
			tasks.push({ neuronId, ageStates, corrections, errorFeedback });
		}
		return { tasks, levelContext, newNeuronSpecs };
	}

	/**
	 * For a single active neuron: add its age>0 entries to the shared levelContext and create
	 * error-correction pattern neurons for ages whose previous votes mismatched reality.
	 * Every age>0 entry contributes to levelContext, including new error patterns — their ids
	 * must propagate into downstream state.context for next-frame corrections. The neuron
	 * itself filters out newErrorPatternIds at match time so brand-new ids don't look like
	 * unexplained novel entries and unfairly penalize pattern scores.
	 * Correction id allocation stays central — the driver owns the neuron-id counter.
	 * @returns {{corrections: Array<{patternId, age, contextEntries}>, errorFeedback: Array<{age, errorRate}>}}
	 *          corrections: error-correction patterns created for this neuron
	 *          errorFeedback: observed error rates for every evaluable age — shipped back
	 *                        to the neuron in the same processFrame call so it can update
	 *                        its own stats with no extra round-trip.
	 */
	getLevelCorrections(neuronId, level, levelContext, ageStates, sensoryNeurons, rewards, isNewErrorPattern) {
		const corrections = [];
		const errorFeedback = [];
		for (const [age, state] of ageStates) {

			// every age > 0 entry contributes to the shared level context
			if (age > 0) levelContext.addNeuron(neuronId, age);

			// new error patterns skip further correction this frame
			if (isNewErrorPattern) continue;

			// evaluate the prior-frame vote at this age (if any) and record feedback
			const result = this.evaluateVoteError(age, state, sensoryNeurons[0]);
			if (!result) continue;
			errorFeedback.push({ age, errorRate: result.errorRate });

			// create an error correction pattern if the error crosses the dynamic threshold
			if (result.fire) {
				const spec = this.allocatePatternNeuron(level + 1, neuronId, age, sensoryNeurons, rewards);
				corrections.push({
					...spec,
					patternId: spec.id,
					age,
					contextEntries: state.context.map(c => ({ neuronId: c.neuronId, distance: c.distance }))
				});
			}
		}
		return { corrections, errorFeedback };
	}

	/**
	 * Pass 2: Op-4 dispatch. Bucket tasks by region and dispatch to
	 * region.processLevel; each region buckets per column. Result writes
	 * (applyFrameResults) run in Thalamus after every column has returned.
	 * Concatenation order is region-index then column-index, stable across runs.
	 */
	dispatchFrame(tasks, memoryDepth, levelContext, newErrorPatternIds, age0, currentRewards, frameNumber) {

		// decorate age=0 sensory neurons with channel id + pre-resolved reward (MPI-ready: neuron doesn't need type)
		const newActiveNeurons = [];
		for (const neuronId of age0) {
			const channelId = this.getNeuronChannelId(neuronId);
			const reward = this.getNeuronType(neuronId) === 'action' ? (currentRewards.get(channelId) || 0) : 0;
			newActiveNeurons.push({ id: neuronId, channelId, reward });
		}

		// bucket tasks by region, dispatch, concatenate in region-index order
		const tasksByRegion = this.bucketByRegion(tasks, 'neuronId');
		const results = [];
		for (let r = 0; r < this.regions; r++) {
			const regionResults = this.regionList[r].processLevel(
				tasksByRegion[r], memoryDepth, levelContext, newErrorPatternIds, newActiveNeurons, frameNumber
			);
			for (const x of regionResults) results.push(x);
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
				state.threshold = null;
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

			// flush per-age votes: write state.votes/state.context/state.threshold for storage,
			// collect id-only votes for consensus. threshold is captured here so next frame's
			// evaluateVoteError can judge these votes without reaching back into the neuron.
			if (perAgeVotes && perAgeVotes.length > 0) {
				const ageStates = levelNeurons.get(parentId);
				for (const { age, votes: ageVotes, context, threshold } of perAgeVotes) {
					const state = ageStates.get(age);
					state.votes = ageVotes;
					state.context = context;
					state.threshold = threshold;
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
	/**
	 * Evaluate a single (neuron, age) prior-frame vote against current actuals.
	 * Returns the observed error rate (so it can be sent back to the neuron as
	 * feedback) and whether it crosses the threshold the neuron supplied when it
	 * cast the vote. The neuron owns its error stats — thalamus only judges.
	 * @returns {{fire: boolean, errorRate: number} | null} null if not evaluable
	 */
	evaluateVoteError(age, state, actualNeuronIds) {

		// age=0 neurons cannot need correction because they are just voting now
		if (age === 0) return null;

		// if there are no votes from previous frame, no error to evaluate
		if (!state.votes || state.votes.length === 0) return null;

		// compare the inferred events to reality
		let failedEvents = 0;
		let totalEvents = 0;
		for (const vote of state.votes)
			if (this.getNeuronType(vote.neuronId) === 'event') {
				totalEvents++;
				if (!actualNeuronIds.has(vote.neuronId)) failedEvents++;
			}
		if (totalEvents === 0) return null;
		const errorRate = failedEvents / totalEvents;

		// the threshold rode in with the vote when it was cast last frame, so this
		// decision happens entirely from coordinator-side state (MPI-friendly: no
		// per-neuron index here, no callback into the neuron worker).
		const fire = errorRate > (state.threshold ?? 0.5);
		return { fire, errorRate };
	}

	/**
	 * Check if a neuron should be skipped (action neuron in a channel whose spec does
	 * not include action-sequence learning).
	 */
	skipActionNeuron(neuronId) {
		if (this.neuronLevels.get(neuronId) !== 0 || this.getNeuronType(neuronId) !== 'action') return false;
		const spec = this.channelSpecs.get(this.getNeuronChannelId(neuronId));
		return spec ? !spec.learnActionSequences : false;
	}

	/**
	 * Get default action neuron id for a channel
	 * @param {number} channelId - Channel id
	 * @returns {number|undefined} - Default action neuron id or undefined
	 */
	getChannelDefaultAction(channelId) {
		return this.channelDefaultActions.get(channelId);
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
	 * Op-2: Delete dead patterns via cascade pulses through region/column.
	 * Each pulse dispatches ops, collects results, cleans metadata, and feeds
	 * outbound ops + newly deletable ids into the next pulse until empty.
	 */
	deletePatterns(patternIds, currentFrame) {
		const allDeletedIds = [];
		const deletedIdSet = new Set();
		const queuedIds = new Set(patternIds);

		// seed the first pulse with DeleteNeuron ops for reaped patterns
		let inboundOps = this.buildDeleteNeuronOps(patternIds);

		// cascade: each pulse may produce outbound ops and new cascade candidates
		while (inboundOps.length > 0) {
			const { outboundOps, deletedIds, newlyDeletableIds } = this.dispatchDeleteOps(inboundOps, currentFrame);

			// remove destroyed neurons from Thalamus metadata
			for (const id of deletedIds) {
				if (deletedIdSet.has(id)) continue;
				deletedIdSet.add(id);
				allDeletedIds.push(id);
				this.cleanupDeletedNeuronMetadata(id);
			}

			// feed outbound ops into next pulse, plus new DeleteNeuron ops for cascade candidates
			inboundOps = outboundOps;
			for (const id of newlyDeletableIds) {
				if (deletedIdSet.has(id) || queuedIds.has(id)) continue;
				queuedIds.add(id);
				const parentId = this.neuronParents.get(id);
				if (parentId !== undefined)
					inboundOps.push({ type: 'DeleteNeuron', targetId: id, parentId });
			}
		}

		return allDeletedIds;
	}

	/**
	 * Build initial DeleteNeuron ops from a list of pattern ids.
	 */
	buildDeleteNeuronOps(patternIds) {
		const ops = [];
		for (const id of patternIds) {
			const parentId = this.neuronParents.get(id);
			if (!parentId) continue; // sensory neurons have no parent
			ops.push({ type: 'DeleteNeuron', targetId: id, parentId });
		}
		return ops;
	}

	/**
	 * Fan out delete ops to regions, collect and merge results.
	 */
	dispatchDeleteOps(ops, currentFrame) {
		const opsByRegion = this.bucketByRegion(ops, 'targetId');
		const outboundOps = [];
		const deletedIds = [];
		const newlyDeletableIds = [];

		for (let r = 0; r < this.regions; r++) {
			if (opsByRegion[r].length === 0) continue;
			const result = this.regionList[r].deleteNeurons(opsByRegion[r], currentFrame);
			for (const op of result.outboundOps) outboundOps.push(op);
			for (const id of result.deletedIds) deletedIds.push(id);
			for (const id of result.newlyDeletableIds) newlyDeletableIds.push(id);
		}

		return { outboundOps, deletedIds, newlyDeletableIds };
	}

	/**
	 * Op-1/Op-3: Route neuron specs to owning regions/columns for construction.
	 * Stores refs in the Thalamus flat map (temporary dual-map scaffolding).
	 */
	createNeuronsInColumns(specs) {
		const specsByRegion = this.bucketByRegion(specs, 'id');
		for (let r = 0; r < this.regions; r++) {
			if (specsByRegion[r].length === 0) continue;
			for (const neuron of this.regionList[r].createNewNeurons(specsByRegion[r]))
				this.neurons.set(neuron.id, neuron);
		}
	}

	/**
	 * Remove a destroyed neuron from Thalamus-owned metadata maps.
	 */
	cleanupDeletedNeuronMetadata(id) {
		this.unregisterDeath(id);
		this.neurons.delete(id);
		const level = this.neuronLevels.get(id);
		this.neuronLevels.delete(id);
		this.neuronParents.delete(id);
		if (level !== undefined) this.decrementLevelCount(level);
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