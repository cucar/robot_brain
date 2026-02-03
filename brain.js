import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';
import getMySQLConnection from './db/db.js';
import { Neuron } from './neurons/neuron.js';
import { Context } from './neurons/context.js';
import { Memory } from './memory.js';

/**
 * Brain Class
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		//************************************************************
		// hyperparameters
		//************************************************************

		// pattern learning parameters
		this.maxLevels = 10; // just to prevent against infinite recursion

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)

		//************************************************************
		// persistent data structures (loaded from MySQL, saved on backup)
		//************************************************************

		// All neurons: Map of neuronId -> Neuron object (SensoryNeuron or PatternNeuron)
		// Each neuron contains its own connections/patterns as properties
		// Note: IDs are temporary during transition - will be removed later
		this.neurons = new Map();

		// Fast lookup for sensory neurons by coordinates: valueKey -> SensoryNeuron
		this.neuronsByValue = new Map();

		// initialize channel registry
		this.channels = new Map();

		// Map of channel name -> Set of action neurons (built once in initializeActionNeurons)
		this.channelActions = new Map();

		//************************************************************
		// frame processing scratch data (reset each frame or episode)
		//************************************************************

		// Memory - manages temporal sliding window and inferred neurons
		this.memory = new Memory();

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// Prediction accuracy tracking (cumulative stats for base level only)
		this.accuracyStats = { correct: 0, total: 0 };

		// Action reward tracking (cumulative stats for base level only)
		this.rewardStats = { totalReward: 0, count: 0 };

		// Continuous prediction metrics (for channels that support it)
		this.continuousPredictionMetrics = { totalError: 0, count: 0 }; // Cumulative MAE across all channels

		// Debugging info and flags
		this.frameNumber = 0;
		this.frameSummary = true; // show frame summary or not
		this.debug = false;
		this.noDatabase = false; // skip database backup/restore for tests
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info

		// Create readline interface for pausing between frames - used when debugging
		this.waitForUserInput = false;
		this.rl = createInterface({ input: stdin, output: stdout });
	}

	/**
	 * waits for user input to continue - used for debugging
	 */
	waitForUser(message) {
		if (!this.waitForUserInput) return Promise.resolve();
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Register a channel with the brain
	 */
	registerChannel(name, channelClass) {
		const channel = new channelClass(name);
		this.channels.set(name, channel);
		if (this.debug) console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async initDB() {
		if (!this.noDatabase) this.conn = await getMySQLConnection();
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain context...');

		// Reset frame counter for proper context window handling
		this.frameNumber = 0;

		// Clear memory
		this.memory.reset();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 * Note: dimensions table is NOT truncated as it's schema-level configuration
	 */
	async resetBrain() {
		await this.resetContext();
		console.log('Hard resetting brain (all learned data)...');

		// Clear in-memory neuron structures
		this.neurons.clear();
		this.neuronsByValue.clear();
		Neuron.nextId = 1;

		// Clear MySQL tables if using a database
		if (!this.noDatabase) await this.truncateTables([
			'channels',
			'dimensions',
			'neurons',
			'base_neurons',
			'coordinates',
			'connections',
			'pattern_peaks',
			'pattern_past',
			'pattern_future'
		]);
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
	}

	/**
	 * Load neurons from MySQL into in-memory Neuron objects.
	 * Called during initialization to load previously learned data.
	 */
	async loadNeurons() {
		if (this.noDatabase) return;
		console.log('Loading neurons from MySQL...');

		// Clear all in-memory structures
		this.neurons.clear();
		this.neuronsByValue.clear();
		Neuron.nextId = 1;

		// Track max ID to set Neuron.nextId after restore
		let maxId = 0;

		// 1. Load base neurons (SensoryNeurons) with their coordinates
		const [baseRows] = await this.conn.query(`
			SELECT b.neuron_id, b.channel_id, b.type, c.name as channel_name
			FROM base_neurons b
			JOIN channels c ON c.id = b.channel_id
		`);
		const [valueRows] = await this.conn.query('SELECT neuron_id, dimension_id, val FROM coordinates');

		// Group coordinates by neuron_id
		const coordsByNeuron = new Map();
		for (const row of valueRows) {
			if (!coordsByNeuron.has(row.neuron_id))
				coordsByNeuron.set(row.neuron_id, {});
			// Convert dimension_id to dimension name for SensoryNeuron
			const dimName = this.dimensionIdToName[row.dimension_id];
			if (dimName) coordsByNeuron.get(row.neuron_id)[dimName] = row.val;
		}

		// Create sensory neuron objects
		for (const row of baseRows) {
			const coords = coordsByNeuron.get(row.neuron_id) || {};
			const neuron = Neuron.createSensory(row.channel_name, row.type, coords);
			this.neurons.set(row.neuron_id, neuron);
			this.neuronsByValue.set(neuron.valueKey, neuron);
			if (row.neuron_id > maxId) maxId = row.neuron_id;
		}
		console.log(`  Loaded ${baseRows.length} sensory neurons`);

		// 2. Load pattern neurons with their peaks
		// ORDER BY level ASC ensures lower-level patterns are created before higher-level ones
		// This matters because a pattern's peak can be another pattern (e.g., level 2 pattern peak is level 1)
		// so the peak must exist in this.neurons before we can reference it
		const [patternRows] = await this.conn.query(`
            SELECT n.id, n.level, pp.peak_neuron_id, pp.strength
            FROM neurons n
            JOIN pattern_peaks pp ON pp.pattern_neuron_id = n.id
            WHERE n.level > 0
            ORDER BY n.level
		`);

		// Create PatternNeuron objects
		for (const row of patternRows) {
			const peak = this.neurons.get(row.peak_neuron_id);
			if (!peak) {
				console.warn(`  Warning: Pattern ${row.id} references missing peak ${row.peak_neuron_id}`);
				continue;
			}
			const pattern = Neuron.createPattern(row.level, peak);
			pattern.peakStrength = row.strength;
			this.neurons.set(row.id, pattern);
			if (row.id > maxId) maxId = row.id;
		}
		console.log(`  Loaded ${patternRows.length} pattern neurons`);

		// 3. Load connections into sensory neuron connections
		const [connRows] = await this.conn.query('SELECT from_neuron_id, to_neuron_id, distance, strength, reward FROM connections');
		let connCount = 0;
		for (const row of connRows) {
			const fromNeuron = this.neurons.get(row.from_neuron_id);
			const toNeuron = this.neurons.get(row.to_neuron_id);
			if (!fromNeuron || !toNeuron) continue;
			if (fromNeuron.level !== 0) continue;

			if (!fromNeuron.connections.has(row.distance))
				fromNeuron.connections.set(row.distance, new Map());
			fromNeuron.connections.get(row.distance).set(toNeuron, {
				strength: row.strength,
				reward: row.reward
			});
			toNeuron.incomingCount++;
			connCount++;
		}
		console.log(`  Loaded ${connCount} connections`);

		// 4. Load pattern_past into peak's contexts routing table
		const [pastRows] = await this.conn.query('SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM pattern_past');
		let pastCount = 0;
		for (const row of pastRows) {
			const pattern = this.neurons.get(row.pattern_neuron_id);
			const contextNeuron = this.neurons.get(row.context_neuron_id);
			if (!pattern || !contextNeuron) continue;
			if (pattern.level === 0) continue;

			// Context is stored on the peak's routing table, not on the pattern
			const peak = pattern.peak;
			if (!peak) continue;
			const context = peak.getOrCreateContext(pattern);
			context.add(contextNeuron, row.context_age, row.strength);
			pastCount++;
		}
		console.log(`  Loaded ${pastCount} pattern context entries (from pattern_past)`);

		// 5. Load pattern_future into pattern.connections (future is now merged into connections)
		const [futureRows] = await this.conn.query('SELECT pattern_neuron_id, inferred_neuron_id, distance, strength, reward FROM pattern_future');
		let futureCount = 0;
		for (const row of futureRows) {
			const pattern = this.neurons.get(row.pattern_neuron_id);
			const inferredNeuron = this.neurons.get(row.inferred_neuron_id);
			if (!pattern || !inferredNeuron) continue;
			if (pattern.level === 0) continue;

			if (!pattern.connections.has(row.distance))
				pattern.connections.set(row.distance, new Map());
			pattern.connections.get(row.distance).set(inferredNeuron, {
				strength: row.strength,
				reward: row.reward
			});
			inferredNeuron.incomingCount++;
			futureCount++;
		}
		console.log(`  Loaded ${futureCount} pattern connections (from pattern_future)`);

		// Set next ID for new neuron creation
		Neuron.nextId = maxId + 1;

		console.log(`Neurons loaded: ${this.neurons.size} total, next ID: ${Neuron.nextId}`);
	}

	/**
	 * Backup brain state from in-memory Neuron objects to MySQL.
	 * Called on shutdown or when job is interrupted.
	 */
	async backupBrain() {
		console.log('Backing up brain to MySQL...');

		// Build neuron -> ID reverse mapping for connections/patterns
		const neuronToId = new Map();
		for (const [id, neuron] of this.neurons)
			neuronToId.set(neuron, id);

		// 1. Backup neurons table
		await this.conn.query('TRUNCATE neurons');
		if (this.neurons.size > 0) {
			const rows = [];
			for (const [id, neuron] of this.neurons)
				rows.push([id, neuron.level]);
			await this.conn.query('INSERT INTO neurons (id, level) VALUES ?', [rows]);
		}
		console.log(`  Saved ${this.neurons.size} neurons`);

		// 2. Backup base_neurons and coordinates
		await this.conn.query('TRUNCATE base_neurons');
		await this.conn.query('TRUNCATE coordinates');
		const baseRows = [];
		const valueRows = [];
		for (const [id, neuron] of this.neurons) {
			if (neuron.level !== 0) continue;
			const channelId = this.channelNameToId[neuron.channel];
			baseRows.push([id, channelId, neuron.type]);
			for (const [dimName, val] of Object.entries(neuron.coordinates)) {
				const dimId = this.dimensionNameToId[dimName];
				if (dimId !== undefined)
					valueRows.push([id, dimId, val]);
			}
		}
		if (baseRows.length > 0)
			await this.conn.query('INSERT INTO base_neurons (neuron_id, channel_id, type) VALUES ?', [baseRows]);
		if (valueRows.length > 0)
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [valueRows]);
		console.log(`  Saved ${baseRows.length} base neurons, ${valueRows.length} coordinates`);

		// 3. Backup connections
		await this.conn.query('TRUNCATE connections');
		const connRows = [];
		for (const [fromId, neuron] of this.neurons) {
			if (neuron.level !== 0) continue;
			for (const [distance, targets] of neuron.connections)
				for (const [toNeuron, conn] of targets) {
					const toId = neuronToId.get(toNeuron);
					if (toId)
						connRows.push([fromId, toId, distance, conn.strength, conn.reward]);
				}
		}
		if (connRows.length > 0)
			await this.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength, reward) VALUES ?', [connRows]);
		console.log(`  Saved ${connRows.length} connections`);

		// 4. Backup pattern_peaks
		await this.conn.query('TRUNCATE pattern_peaks');
		const peakRows = [];
		for (const [patternId, neuron] of this.neurons) {
			if (neuron.level === 0) continue;
			const peakId = neuronToId.get(neuron.peak);
			if (peakId)
				peakRows.push([patternId, peakId, neuron.peakStrength]);
		}
		if (peakRows.length > 0)
			await this.conn.query('INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength) VALUES ?', [peakRows]);
		console.log(`  Saved ${peakRows.length} pattern peaks`);

		// 5. Backup pattern context (from peak's routing table to pattern_past)
		await this.conn.query('TRUNCATE pattern_past');
		const pastRows = [];
		for (const [_, neuron] of this.neurons)
			for (const { context, pattern } of neuron.contexts) {
				const patternId = neuronToId.get(pattern);
				if (!patternId) continue;
				for (const { neuron: ctxNeuron, distance, strength } of context.entries) {
					const contextId = neuronToId.get(ctxNeuron);
					if (contextId)
						pastRows.push([patternId, contextId, distance, strength]);
				}
			}
		if (pastRows.length > 0)
			await this.conn.query('INSERT INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age, strength) VALUES ?', [pastRows]);
		console.log(`  Saved ${pastRows.length} pattern context entries (to pattern_past)`);

		// 6. Backup pattern connections (to pattern_future table for compatibility)
		await this.conn.query('TRUNCATE pattern_future');
		const futureRows = [];
		for (const [patternId, neuron] of this.neurons) {
			if (neuron.level === 0) continue;
			for (const [distance, targets] of neuron.connections)
				for (const [inferredNeuron, pred] of targets) {
					const inferredId = neuronToId.get(inferredNeuron);
					if (inferredId)
						futureRows.push([patternId, inferredId, distance, pred.strength, pred.reward]);
				}
		}
		if (futureRows.length > 0)
			await this.conn.query('INSERT INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance, strength, reward) VALUES ?', [futureRows]);
		console.log(`  Saved ${futureRows.length} pattern connections (to pattern_future)`);

		console.log('Brain backed up to MySQL.');
	}

	/**
	 * truncates given tables for database reset
	 */
	async truncateTables(tables) {
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		await Promise.all(tables.map(table => this.conn.query(`TRUNCATE ${table}`)));
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}

	/**
	 * initializes the brain and loads dimensions
	 */
	async init() {
		// console.log('Initializing brain...');

		// register channels in DB and load channel IDs
		await this.initializeChannels();

		// create dimensions for all registered channels
		await this.initializeDimensions();

		// load the dimensions
		await this.loadDimensions();

		// load learned data from MySQL into in-memory structures (skip if noDatabase flag set)
		await this.loadNeurons();

		// pre-create action neurons for all channels
		await this.initializeActionNeurons();

		// initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) await channel.initialize();
	}

	/**
	 * Pre-create action neurons for all channels.
	 * This ensures exploration can find action neurons even before any connections exist.
	 * Also builds the channelActions map for pattern refinement.
	 */
	async initializeActionNeurons() {
		for (const [channelName, channel] of this.channels) {
			const actionCoords = channel.getActionNeurons();
			if (actionCoords.length === 0) continue;

			// Build frame points for action neurons
			const framePoints = actionCoords.map(coords => ({
				coordinates: coords,
				channel: channelName,
				channel_id: this.channelNameToId[channelName],
				type: 'action'
			}));

			// Create neurons (getFrameNeurons will create if they don't exist)
			const neuronIds = this.getFrameNeurons(framePoints);

			// Build channelActions map for pattern refinement
			const actionSet = new Set();
			for (const neuronId of neuronIds)
				actionSet.add(this.neurons.get(neuronId));
			this.channelActions.set(channelName, actionSet);

			if (this.debug) console.log(`Created ${actionCoords.length} action neurons for ${channelName}`);
		}
	}

	/**
	 * Initialize channels in DB and load channel IDs, which come from static Channel.nextId counter
	 */
	async initializeChannels() {
		this.channelNameToId = {};
		this.channelIdToName = {};

		// Insert channels into DB with explicit IDs from Channel objects
		for (const [channelName, channel] of this.channels) {
			if (!this.noDatabase)
				await this.conn.query('INSERT IGNORE INTO channels (id, name) VALUES (?, ?)', [channel.id, channelName]);
			this.channelNameToId[channelName] = channel.id;
			this.channelIdToName[channel.id] = channelName;
		}
		if (this.debug) console.log('Channels loaded:', this.channelNameToId);
	}

	/**
	 * Initialize dimensions for all registered channels
	 * IDs come from static Dimension.nextId counter (not auto-increment)
	 */
	async initializeDimensions() {
		if (this.noDatabase) return;
		if (this.debug) console.log('Initializing dimensions for registered channels...');
		for (const [, channel] of this.channels) {
			for (const dim of channel.getEventDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
			for (const dim of channel.getOutputDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
		}
	}

	/**
	 * loads the dimensions to memory (just id and name, no channel/type)
	 * Populates mappings from Dimension objects owned by channels
	 */
	async loadDimensions() {
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};

		// Build mappings from Dimension objects (no need to query DB)
		for (const [, channel] of this.channels) {
			for (const dim of channel.getEventDimensions()) {
				this.dimensionNameToId[dim.name] = dim.id;
				this.dimensionIdToName[dim.id] = dim.name;
			}
			for (const dim of channel.getOutputDimensions()) {
				this.dimensionNameToId[dim.name] = dim.id;
				this.dimensionIdToName[dim.id] = dim.name;
			}
		}
		if (this.debug) console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * Get neuron ID by dimension name and value.
	 * Used for diagnostic output to show which neurons correspond to which values.
	 * @param {string} dimensionName - The dimension name
	 * @param {number|string} value - The value to look up
	 * @returns {number|null} - Neuron ID or null if not found
	 */
	getNeuronIdByDimensionValue(dimensionName, value) {
		// Search through all sensory neurons for one with this dimension value
		for (const [neuronId, neuron] of this.neurons)
			if (neuron.level === 0 && neuron.coordinates[dimensionName] === value)
				return neuronId;
		return null;
	}

	/**
	 * Get frame actions for all channels from in-memory inferredNeurons Map.
	 * All entries in inferredNeurons are winners, so just filter by action type.
	 * @returns {Map} - Map of channel names to array of output coordinates
	 */
	getInferredActions() {
		const channelOutputs = new Map();

		for (const [neuronId] of this.memory.getInferences()) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron || neuron.level !== 0 || neuron.type !== 'action') continue;

			if (!channelOutputs.has(neuron.channel)) channelOutputs.set(neuron.channel, []);
			channelOutputs.get(neuron.channel).push(neuron.coordinates);
		}

		return channelOutputs;
	}

	/**
	 * processes one frame of input values - [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 * and channel-specific rewards (Map of channel_name -> reward)
	 * @returns Promise<boolean> - true if frame was processed, false if no more data available
	 */
	async processFrame() {
		const frameStart = performance.now();

		// Handle all frame I/O: get frame, execute actions, get rewards, display header
		if (!await this.processFrameIO()) return false;

		// age the active neurons in memory context - sliding the temporal window
		// deletion of aged-out neurons is deferred to after pattern learning
		this.ageNeurons();

		// activate neurons that represent the current situation in age=0 - what's happening right now?
		this.recognizeNeurons();

		// update the age>0 neuron connections based on observations in age=0
		this.updateConnections();

		// learn new patterns from failed predictions and action regret
		this.learnNewPatterns();

		// deactivate aged-out neurons AFTER pattern learning captured full context
		this.deactivateOldNeurons();

		// do predictions and outputs - what's going to happen next? and what's our best response?
		this.inferNeurons();

		// at this point the frame is processed - the forget cycle is a periodic cleanup task
		// used to avoid curse of dimensionality and delete dead connections/neurons
		this.runForgetCycle();

		// show frame processing summary
		this.printFrameSummary(performance.now() - frameStart);

		// when debugging, wait for user to press Enter before continuing to next frame
		await this.waitForUser('Press Enter to continue to next frame');

		// return true to indicate that we have processed the frame successfully
		return true;
	}

	/**
	 * Handles all frame I/O: get frame, execute actions, get rewards, display header
	 * @returns Promise<boolean> - true if frame data available, false if no more data
	 */
	async processFrameIO() {

		// get the current frame from all channels - includes events and previously inferred actions
		await this.getFrame();
		if (!this.frame || this.frame.length === 0) return false;

		// execute the previously inferred actions in all channels
		await this.executeActions();

		// get rewards from all channels based on executed actions
		await this.getRewards();

		// display diagnostic frame header if enabled
		this.displayFrameHeader();

		return true;
	}

	/**
	 * Returns the current frame combined from all registered channels
	 * Each frame point includes: coordinates, channel, type
	 * Populates this.frame with events from channels and actions from previous inference
	 */
	async getFrame() {
		this.frame = [];

		// Increment frame counter to be able to track inactivity
		this.frameNumber++;
		if (this.debug) console.log('******************************************************************');
		if (this.debug) console.log(`OBSERVING FRAME ${this.frameNumber}`);

		// Get all frame actions from previous frame's inference (from in-memory inferredNeurons)
		const frameActions = this.getInferredActions();

		// Process each channel: get inputs from channel, get outputs from brain tables
		for (const [channelName, channel] of this.channels) {
			const channelId = this.channelNameToId[channelName];

			// Get the frame event inputs from the channel (current state before any outputs are executed)
			const channelEvents = await channel.getFrameEvents();
			for (const event of channelEvents)
				this.frame.push({ coordinates: event, channel: channelName, channel_id: channelId, type: 'event' });

			// Get last inferred actions to be executed in this frame (from brain's inference tables)
			const channelActions = frameActions.get(channelName) || [];
			for (const action of channelActions)
				this.frame.push({ coordinates: action, channel: channelName, channel_id: channelId, type: 'action' });
		}

		if (this.debug) console.log(`Processing frame: ${this.frame.length} neurons`);
		if (this.debug) console.log(`frame points: ${JSON.stringify(this.frame)}`);
		if (this.debug) console.log('******************************************************************');
	}

	/**
	 * Execute previously inferred actions for all channels
	 */
	async executeActions() {
		const channelActions = this.getFrameActions();
		for (const [channelName, channel] of this.channels)
			await channel.executeOutputs(channelActions.get(channelName) || []);
	}

	/**
	 * Extract action outputs from this.frame grouped by channel
	 * @returns {Map} - Map of channel names to array of action coordinate objects
	 */
	getFrameActions() {
		const channelActions = new Map();
		for (const point of this.frame)
			if (point.type === 'action') {
				if (!channelActions.has(point.channel)) channelActions.set(point.channel, []);
				channelActions.get(point.channel).push(point.coordinates);
			}
		return channelActions;
	}

	/**
	 * Get channel-specific feedback as a Map of channel_name -> reward
	 * Each channel provides its own reward signal based on its objectives
	 */
	async getRewards() {
		if (this.debug) console.log('Getting rewards feedback from all channels...');
		this.rewards = new Map();
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const reward = await channel.getRewards();
			if (reward !== 0) { // Only process non-neutral feedback (additive: 0 = neutral)
				if (this.debug) console.log(`${channelName}: reward ${reward.toFixed(3)}`);
				this.rewards.set(channelName, reward);
				feedbackCount++;
			}
		}

		if (this.debug) {
			if (feedbackCount > 0) console.log(`Received rewards from ${feedbackCount} channels`);
			else console.log('No rewards from any channels');
		}
		if (this.debug && feedbackCount > 0)
			console.log(`Channel rewards:`, Array.from(this.rewards.entries()).map(([ch, r]) => `${ch}: ${r.toFixed(3)}`).join(', '));
	}

	/**
	 * Display diagnostic frame header with frame number and observations
	 */
	displayFrameHeader() {
		if (!this.diagnostic) return;

		// Display reward information
		if (this.rewards.size > 0) this.displayRewards();

		// Build observation string from frame
		const observations = [];
		for (const point of this.frame)
			for (const [dim, val] of Object.entries(point.coordinates))
				observations.push(`${dim}=${val}`);

		console.log(`\nF${this.frameNumber} | Obs: ${observations.join(', ')}`);
	}

	/**
	 * Display reward information for diagnostic output
	 */
	displayRewards() {
		const rewardParts = [];
		for (const [channelName, reward] of this.rewards)
			rewardParts.push(`${channelName}:${reward.toFixed(3)}x`);
		console.log(`  Rewards: ${rewardParts.join(', ')}`);
	}

	/**
	 * Ages all neurons in the context by 1.
	 * Deletion of aged-out neurons is deferred to deactivateOldNeurons() at end of frame,
	 * so that pattern creation can capture the full context before neurons are deleted.
	 */
	ageNeurons() {
		if (this.debug) console.log('Aging neurons...');
		this.memory.age();
	}

	/**
	 * Deactivates neurons that have aged out of the context window.
	 * Called at end of frame after pattern learning, so patterns can capture full context.
	 */
	deactivateOldNeurons() {
		this.memory.deactivateOld(this.debug);
	}

	/**
	 * recognizes and activates base level neurons from frame
	 */
	recognizeNeurons() {

		// bulk find/create neurons for all input points
		const neuronIds = this.getFrameNeurons(this.frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// discover and activate patterns using connections - start recursion from base level
		this.recognizePatterns();

		// Track inference performance (event accuracy and action rewards)
		this.trackInferencePerformance();
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];

		for (const point of frame) {
			const valueKey = Neuron.makeValueKey(point.coordinates);
			let neuron = this.neuronsByValue.get(valueKey);

			// Create new neuron if not found
			if (!neuron) {
				neuron = Neuron.createSensory(point.channel, point.type, point.coordinates);
				const neuronId = neuron.id;
				this.neurons.set(neuronId, neuron);
				this.neuronsByValue.set(valueKey, neuron);
				if (this.debug) console.log(`Created new sensory neuron ${neuronId} for ${valueKey}`);
			}

			neuronIds.push(neuron.id);
		}

		if (neuronIds.length === 0) throw new Error(`Failed to get neurons for frame: ${JSON.stringify(frame)}`);
		return neuronIds;
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<number>} neuronIds - Array of neuron IDs to activate
	 */
	activateNeurons(neuronIds) {
		for (const neuronId of neuronIds) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron) throw new Error(`Neuron ${neuronId} not found in this.neurons`);
			this.memory.activateNeuron(neuron);
		}
	}

	/**
	 * Track inference performance for both events and actions.
	 * All entries in inferredNeurons are winners.
	 * Events: checks if prediction was correct (neuron became active)
	 * Actions: accumulates reward from the action's channel
	 */
	trackInferencePerformance() {
		let eventCorrect = 0;
		let eventTotal = 0;
		let actionReward = 0;
		let actionCount = 0;

		for (const [neuronId] of this.memory.getInferences()) {
			const neuron = this.neurons.get(neuronId);

			// Track event prediction accuracy
			if (neuron.type === 'event') {
				eventTotal++;
				if (this.memory.getNeuronsAtAge(0).has(neuron)) eventCorrect++;
			}
			// Track action reward from the action's channel
			else if (neuron.type === 'action') {
				const reward = this.rewards.get(neuron.channel);
				if (reward !== undefined) {
					actionReward += reward;
					actionCount++;
				}
			}
		}

		// Update cumulative stats
		this.accuracyStats.correct += eventCorrect;
		this.accuracyStats.total += eventTotal;
		this.rewardStats.totalReward += actionReward;
		this.rewardStats.count += actionCount;

		if (this.debug) {
			if (eventTotal > 0) {
				const accuracy = (eventCorrect / eventTotal * 100).toFixed(1);
				console.log(`Event predictions: ${eventCorrect}/${eventTotal} (${accuracy}%)`);
			}
			if (actionCount > 0)
				console.log(`Action rewards: ${actionReward.toFixed(3)} for ${actionCount} actions`);
		}
	}

	/**
	 * updates neuron connections based on observations.
	 * Context neurons (age > 0) learn about newly active neurons (age = 0).
	 */
	updateConnections() {

		// Get newly active sensory neurons (age=0, level=0 - events and actions)
		const newEventNeurons = new Set();
		const newActionNeurons = new Set();
		for (const neuron of this.memory.getNeuronsAtAge(0).keys()) {
			if (neuron.level > 0) continue;
			if (neuron.type === 'event') newEventNeurons.add(neuron);
			else if (neuron.type === 'action') newActionNeurons.add(neuron);
		}
		if (newEventNeurons.size === 0 && newActionNeurons.size === 0) return;

		// Each context neuron learns connections at its own distance
		for (const { neuron, age } of this.memory.getContextNeurons())
			neuron.learnConnectionsAtAge(age, newEventNeurons, newActionNeurons, this.rewards, this.channelActions);
	}

	/**
	 * Detects patterns at all levels starting from base - goes as high as possible until no patterns found.
	 */
	recognizePatterns() {
		let level = 0;
		while (true) {
			const patternsFound = this.recognizeLevelPatterns(level);
			if (!patternsFound) break;

			level++;
			if (level >= this.maxLevels) {
				console.error('Max level exceeded.');
				break;
			}
		}
	}

	/**
	 * Processes a level to detect patterns and activate them. Returns true if patterns were found, false otherwise.
	 */
	recognizeLevelPatterns(level) {
		if (this.debug) console.log(`Processing level ${level} for pattern recognition`);

		// get the peaks and context for this level
		const {peaks, context} = this.getPeaksAndContext(level);
		if (peaks.length === 0) {
			if (this.debug) console.log(`No newly activated neurons at level ${level}`);
			return false;
		}

		// Match patterns (parallelizable) - collect results with peak reference
		const matchedPatterns = peaks.map(peak => peak.matchBestPattern(context)).filter(m => m);

		// If no patterns matched, stop here
		if (matchedPatterns.length === 0) {
			if (this.debug) console.log(`No pattern matches found at level ${level}`);
			return false;
		}

		// activate the matched pattern neurons
		for (const { peak, pattern } of matchedPatterns)
			this.memory.activatePattern(pattern, peak, 0);

		if (this.debug)
			console.log(`Matched ${matchedPatterns.length} patterns at level ${level}:`,
				matchedPatterns.map(m => `peak=${m.peak.id}, pattern=${m.pattern.id}`).join('; '));

		// return true to indicate patterns found
		return true;
	}

	/**
	 * Get peaks (age=0) and context (age>0) neurons.
	 * @param {number} [level] - Optional level to filter by. If omitted, includes all levels.
	 * @returns {{peaks: Array<Neuron>, context: Context}}
	 */
	getPeaksAndContext(level) {
		return this.memory.getPeaksAndContext(level);
	}

	/**
	 * Learn new patterns from prediction errors and action regret.
	 * Iterates over inferringNeurons (neurons that voted for winners) to find prediction errors.
	 * Note: inferringNeurons is aged along with activeNeurons, so ages are aligned.
	 */
	learnNewPatterns() {

		// Get new activated neurons (age=0)
		const newActiveNeurons = new Set(this.memory.getNeuronsAtAge(0).keys());

		let patternCount = 0;

		// For each inferring neuron with its context
		for (const { neuron, age, votes, context } of this.memory.getInferringNeuronsWithContext()) {

			// Try to learn a pattern at this age
			const newPattern = neuron.learnNewPattern(age, votes, newActiveNeurons, this.rewards, this.channelActions, context);

			// if no pattern was learned, move on to the next neuron
			if (!newPattern) continue;

			// index the new neuron with its id
			this.neurons.set(newPattern.id, newPattern);

			// activate the pattern neuron at the peak's age
			this.memory.activatePattern(newPattern, neuron, age);

			patternCount++;
		}

		if (this.debug && patternCount > 0) console.log(`Created ${patternCount} error patterns`);
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 */
	inferNeurons() {

		// Collect votes from active neurons (suppression handled during collection)
		const votes = this.collectVotes();

		// Aggregate votes and determine winners
		const inferences = this.determineConsensus(votes);

		// If no inferences, wait for more data
		if (inferences.length === 0) {
			if (this.debug) console.log('No inferences found. Waiting for more data in future frames.');
			return;
		}

		// Apply exploration to actions (may override winners)
		this.applyExploration(inferences);

		// Save all inferences
		this.saveInferences(inferences);

		// Notify channels about winning event predictions for continuous tracking (e.g., price prediction)
		this.notifyChannelsOfEventPredictions(inferences);
	}

	/**
	 * Collect votes from active neurons.
	 * Populates inferringNeurons array for pattern learning.
	 * @returns {Array} Array of vote objects for consensus
	 */
	collectVotes() {
		const votes = [];
		let baseCount = 0, patternCount = 0;

		// Clear inferring neurons array
		this.memory.clearInferringNeurons();

		// Collect votes from neurons that can vote
		for (const { neuron, age, activatedPattern } of this.memory.getVotingNeurons()) {

			// Only event neurons (level 0) or pattern neurons can create patterns
			if (neuron.level === 0 && neuron.type !== 'event') continue;

			// if a pattern was activated by the neuron, its inference is suppressed - skip
			if (activatedPattern !== null) continue;

			// get the votes of the neuron
			const neuronVotes = neuron.collectVotesAtAge(age, this.memory.contextLength);
			if (neuronVotes.length === 0) continue;

			// Store votes in inferringNeurons for pattern learning
			this.memory.addVote(neuron, age, neuronVotes);

			// add the votes to the returned array
			for (const v of neuronVotes) {
				votes.push({ fromNeuronId: neuron.id, neuronId: v.toNeuron.id, strength: v.strength, reward: v.reward, distance: v.distance });
				if (neuron.isPattern) patternCount++;
				else baseCount++;
			}
		}

		if (this.debug) console.log(`Collected ${baseCount} base votes, ${patternCount} pattern votes`);
		return votes;
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * @param {Array} votes - Array of vote objects from collectVotes
	 * @returns {Array} Array of inference objects with isWinner flag
	 */
	determineConsensus(votes) {

		// Aggregate votes by neuron
		const aggregated = new Map(); // neuronId -> {neuron, strength, weightedReward}
		for (const v of votes) {
			const neuron = this.neurons.get(v.neuronId);
			if (!aggregated.has(v.neuronId))
				aggregated.set(v.neuronId, { neuron, strength: 0, weightedReward: 0 });
			const agg = aggregated.get(v.neuronId);
			agg.strength += v.strength;
			agg.weightedReward += v.strength * v.reward;
		}

		if (aggregated.size === 0) return [];

		// Build candidates with aggregated strength and reward
		const candidates = [];
		for (const [neuronId, agg] of aggregated) {
			const n = agg.neuron;
			candidates.push({
				neuron_id: neuronId,
				strength: agg.strength,
				reward: agg.weightedReward / agg.strength,
				type: n.type,
				channel: n.channel,
				coordinates: n.coordinates
			});
		}

		// Determine winners per dimension (events by strength, actions by reward)
		const dimBest = new Map(); // dimension -> {neuronId, score}
		for (const c of candidates) {
			const score = c.type === 'action' ? c.reward : c.strength;
			for (const dim of Object.keys(c.coordinates))
				if (!dimBest.has(dim) || score > dimBest.get(dim).score)
					dimBest.set(dim, { neuronId: c.neuron_id, score });
		}

		const winnerIds = new Set([...dimBest.values()].map(w => w.neuronId));

		if (this.debug) console.log(`Determined consensus: ${candidates.length} candidates, ${winnerIds.size} winners`);

		return candidates.map(c => ({ ...c, isWinner: winnerIds.has(c.neuron_id) }));
	}

	/**
	 * Apply exploration to action inferences.
	 * Modifies inferences array in place - may add exploration action and update isWinner flags.
	 * @param {Array} inferences - Array of inference objects (modified in place)
	 */
	applyExploration(inferences) {

		// Group action inferences by channel (using channel from neuron, not dimension)
		const byChannel = new Map();
		for (const inf of inferences) {
			if (!inf.coordinates || inf.type !== 'action') continue;
			if (!byChannel.has(inf.channel)) byChannel.set(inf.channel, []);
			byChannel.get(inf.channel).push(inf);
		}

		// Process each channel
		for (const [channelName] of this.channels) {
			const channelInferences = byChannel.get(channelName) || [];
			const channelWinners = channelInferences.filter(inf => inf.isWinner);

			// Check if we should explore this channel
			if (!this.shouldExploreChannel(channelName, channelWinners)) continue;

			// Get exploration action - pass all channel inferences so channel knows what's been tried
			const exploration = this.exploreChannel(channelName, channelInferences);
			if (!exploration) continue; // All actions have been tried

			// Mark previous winners as losers
			for (const winner of channelWinners)
				if (winner.neuron_id !== exploration.neuron_id)
					winner.isWinner = false;

			// Add exploration as winner
			inferences.push(exploration);
		}
	}

	/**
	 * Decide whether to explore a channel. Only explore when there are no action winners for the channel.
	 * @param {string} channelName - name of the channel to check
	 * @param {Array} channelWinners - array of winning action inferences for this channel
	 * @returns {boolean} - true if we should explore this channel
	 */
	shouldExploreChannel(channelName, channelWinners) {
		const shouldExplore = channelWinners.length === 0;
		if (this.debug && shouldExplore) console.log(`${channelName}: No action winners, exploring`);
		return shouldExplore;
	}

	/**
	 * Explore a channel by finding an unexplored action.
	 * @param {string} channelName - Channel name
	 * @param {Array} votedActions - All actions that received votes (have connections from current context)
	 * @returns {Object|null} Exploration action or null if all actions explored
	 */
	exploreChannel(channelName, votedActions) {
		const channel = this.channels.get(channelName);

		// Extract coordinates from voted actions
		const votedCoordinates = votedActions.map(a => a.coordinates).filter(c => c);

		// Ask channel for an action that wasn't voted for
		const actionCoordinates = channel.getExplorationAction(votedCoordinates);
		if (!actionCoordinates || Object.keys(actionCoordinates).length === 0) {
			if (this.debug) console.log(`Exploration for ${channelName}: all actions explored`);
			return null; // All actions explored
		}

		// Find or create neuron for this action - wrap coordinates in frame point structure with channel metadata
		const [actionNeuronId] = this.getFrameNeurons([{
			coordinates: actionCoordinates,
			channel: channelName,
			channel_id: this.channelNameToId[channelName],
			type: 'action'
		}]);

		// Return exploration action
		// Use low strength (below eventErrorMinStrength) to avoid triggering error pattern creation
		// when exploration fails - exploration is random, not a confident prediction that was wrong
		const exploration = {
			neuron_id: actionNeuronId,
			strength: 1, // Low strength - exploration wins by marking others as losers, not by high strength
			reward: 0, // Neutral reward (additive: 0 = neutral)
			coordinates: actionCoordinates,
			type: 'action',
			isWinner: true
		};

		if (this.debug) console.log(`Exploration for ${channelName}:`, actionCoordinates, actionNeuronId);
		return exploration;
	}

	/**
	 * Save winning inferences to in-memory structures.
	 * Only winners are saved - losers are discarded.
	 * Also filters inferringNeurons to only keep votes that led to winners.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	saveInferences(inferences) {

		// Get set of winning neuron IDs
		const winnerIds = new Set();
		for (const inf of inferences)
			if (inf.isWinner)
				winnerIds.add(inf.neuron_id);

		// Save only winners to inferredNeurons
		this.memory.clearInferences();
		for (const inf of inferences)
			if (inf.isWinner)
				this.memory.addInference(inf.neuron_id, inf.strength);

		// Filter inferringNeurons to only keep votes that led to winners
		this.memory.filterInferringByWinners(winnerIds);

		if (this.debug)
			console.log(`Saved ${winnerIds.size} winning inferences`);
	}

	/**
	 * Notify channels about winning event predictions for continuous tracking.
	 * Channels can use this to calculate continuous predictions (e.g., price prediction from buckets).
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	notifyChannelsOfEventPredictions(inferences) {

		// Filter to winning event predictions only
		const eventWinners = inferences.filter(inf => inf.isWinner && inf.type === 'event');
		if (eventWinners.length === 0) return;

		// Group by channel
		const byChannel = new Map();
		for (const winner of eventWinners) {
			if (!winner.channel) continue;
			if (!byChannel.has(winner.channel)) byChannel.set(winner.channel, []);
			byChannel.get(winner.channel).push(winner);
		}

		// Notify each channel
		for (const [channelName, winners] of byChannel) {
			const channel = this.channels.get(channelName);
			if (channel && typeof channel.onEventPredictions === 'function')
				channel.onEventPredictions(winners);
		}
	}

	/**
	 * Runs the forget cycle, reducing strengths and deleting unused connections/patterns/neurons.
	 * Critical for avoiding curse of dimensionality.
	 */
	runForgetCycle() {

		// Run periodically for cleanup
		this.forgetCounter++;
		if (this.forgetCounter % this.forgetCycles !== 0) return;
		this.forgetCounter = 0;

		const cycleStart = Date.now();
		if (this.debug) console.log('=== FORGET CYCLE STARTING ===');

		// 1. Single parallelizable loop - each neuron handles its own forgetting
		let connectionsUpdated = 0, connectionsDeleted = 0, contextDeleted = 0, peaksDeleted = 0;
		for (const neuron of this.neurons.values()) {
			const stats = neuron.forget();
			connectionsUpdated += stats.connectionsUpdated;
			connectionsDeleted += stats.connectionsDeleted;
			contextDeleted += stats.contextDeleted;
			if (stats.peakDeleted) peaksDeleted++;
		}
		if (this.debug) console.log(`  Connections: ${connectionsUpdated} weakened, ${connectionsDeleted} deleted`);
		if (this.debug) console.log(`  Patterns: context ${contextDeleted}, peaks ${peaksDeleted}`);

		// 2. Orphan cleanup (must be done after all neurons finish forgetting)
		const orphanCount = this.deleteOrphanedPatterns();
		if (this.debug) console.log(`  Orphaned patterns deleted: ${orphanCount}`);

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Delete orphaned pattern neurons (no content, no references, not active).
	 * @returns {number} Number of patterns deleted
	 */
	deleteOrphanedPatterns() {
		const toDelete = [];

		for (const neuron of this.neurons.values()) {
			if (neuron.level === 0) continue;
			if (!this.memory.isNeuronActive(neuron) && neuron.canDelete()) toDelete.push(neuron);
		}

		for (const neuron of toDelete) {
			neuron.cleanup();
			this.neurons.delete(neuron.id);
		}

		return toDelete.length;
	}

	/**
	 * Prints a one-line summary of the frame processing
	 */
	printFrameSummary(frameElapsed) {

		// Get base level (level 0) accuracy
		let baseAccuracy = 'N/A';
		if (this.accuracyStats.total > 0)
			baseAccuracy = `${(this.accuracyStats.correct / this.accuracyStats.total * 100).toFixed(1)}%`;

		// Get average action reward
		let avgReward = 'N/A';
		if (this.rewardStats.count > 0)
			avgReward = `${(this.rewardStats.totalReward / this.rewardStats.count).toFixed(3)} (${this.rewardStats.count})`;

		// Collect continuous prediction metrics from channels (only new errors since last call)
		for (const [_, channel] of this.channels) {
			if (typeof channel.getPredictionMetrics === 'function') {
				const metrics = channel.getPredictionMetrics();
				if (metrics) {
					this.continuousPredictionMetrics.totalError += metrics.totalError;
					this.continuousPredictionMetrics.count += metrics.count;
				}
			}
		}

		// Calculate average MAPE (Mean Absolute Percentage Error) and format with count
		let mapeDisplay = 'N/A';
		if (this.continuousPredictionMetrics.count > 0) {
			const avgMAPE = (this.continuousPredictionMetrics.totalError / this.continuousPredictionMetrics.count).toFixed(2);
			mapeDisplay = `${avgMAPE}% (${this.continuousPredictionMetrics.count})`;
		}

		// Collect output performance metrics from channels
		const outputMetrics = [];
		for (const [_, channel] of this.channels) {
			if (typeof channel.getOutputPerformanceMetrics === 'function') {
				const metrics = channel.getOutputPerformanceMetrics();
				if (metrics) outputMetrics.push(metrics);
			}
		}

		// Format output performance display
		let outputDisplay = 'N/A';
		if (outputMetrics.length > 0) {
			outputDisplay = outputMetrics.map(m => {
				const formatted = m.format === 'currency'
					? `$${m.value >= 0 ? '+' : ''}${m.value.toFixed(2)}`
					: m.value.toFixed(2);
				return outputMetrics.length > 1 ? `${m.label}:${formatted}` : formatted;
			}).join(', ');
		}

		if (this.frameSummary) console.log(`Frame ${this.frameNumber} | Accuracy: ${baseAccuracy} | Reward: ${avgReward} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}

}