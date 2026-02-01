import readline from 'node:readline';
import getMySQLConnection from './db/db.js';
import { Neuron, SensoryNeuron, PatternNeuron } from './neurons/index.js';

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

		// structural limits for the brain
		this.contextLength = 5; // number of frames a base neuron stays active - this determines the context length

		// pattern learning parameters
		this.eventErrorMinStrength = 10; // minimum prediction strength to create error-driven patterns
		this.actionRegretMinStrength = 1; // minimum inference strength to create action regret pattern
		this.maxLevels = 10; // just to prevent against infinite recursion

		// voting parameters
		this.levelVoteMultiplier = 3; // how much to weight votes from higher levels

		// forget cycle parameters - very important - fights curse of dimensionality
		this.forgetCycles = 100; // number of frames between forget cycles (increased to let connections stabilize)
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle (reduced to preserve learned connections)
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle

		//************************************************************
		// persistent data structures (loaded from MySQL, saved on backup)
		//************************************************************

		// All neurons: Map of neuronId -> Neuron object (SensoryNeuron or PatternNeuron)
		// Each neuron contains its own connections/patterns as properties
		// Note: IDs are temporary during transition - will be removed later
		this.neurons = new Map();

		// Fast lookup for sensory neurons by coordinates: valueKey -> SensoryNeuron
		this.neuronsByValue = new Map();

		//************************************************************
		// frame processing scratch data (reset each frame or episode)
		//************************************************************

		// Active context: Set of neurons currently in sliding window (ages stored on neurons)
		this.activeNeurons = new Set();

		// Current frame inference results: Neuron -> {strength, isWinner}
		this.inferredNeurons = new Map();

		// Inference votes for current frame: Array of {from, to, strength, reward, distance}
		this.inferenceVotes = [];

		// New pattern futures: Array of {peak, inferred, distance}
		this.newPatternFuture = [];

		// New patterns being created: Array of {peak, pattern}
		this.newPatterns = [];

		// Frame state - populated by processFrameIO methods
		this.frame = []; // current frame data from all channels
		this.rewards = new Map(); // channel rewards for current frame

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

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
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
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
		this.conn = await getMySQLConnection();
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain context...');

		// Reset frame counter for proper context window handling
		this.frameNumber = 0;

		// Clear activeAges on all neurons that were active
		for (const neuron of this.activeNeurons) neuron.activeAges = null;

		// Clear active neurons set
		this.activeNeurons.clear();

		// In-memory scratch data
		this.inferenceVotes = [];       // Array of {from, to, strength, reward, distance} (Neuron objects)
		this.inferredNeurons = new Map(); // Map<Neuron, {strength, isWinner}>
		this.newPatternFuture = [];     // Array of {peak, inferred, distance} (Neuron objects)
		this.newPatterns = [];          // Array of {peak, pattern} (Neuron objects)
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

		// Clear MySQL tables
		await this.truncateTables([
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
		const [coordRows] = await this.conn.query('SELECT neuron_id, dimension_id, val FROM coordinates');

		// Group coordinates by neuron_id
		const coordsByNeuron = new Map();
		for (const row of coordRows) {
			if (!coordsByNeuron.has(row.neuron_id))
				coordsByNeuron.set(row.neuron_id, {});
			// Convert dimension_id to dimension name for SensoryNeuron
			const dimName = this.dimensionIdToName[row.dimension_id];
			if (dimName) coordsByNeuron.get(row.neuron_id)[dimName] = row.val;
		}

		// Create SensoryNeuron objects
		for (const row of baseRows) {
			const coords = coordsByNeuron.get(row.neuron_id) || {};
			const neuron = new SensoryNeuron(row.channel_name, row.type, coords);
			this.neurons.set(row.neuron_id, neuron);
			this.neuronsByValue.set(neuron.valueKey, neuron);
			if (row.neuron_id > maxId) maxId = row.neuron_id;
		}
		console.log(`  Loaded ${baseRows.length} sensory neurons`);

		// 2. Load pattern neurons with their peaks
		// ORDER BY level ASC ensures lower-level patterns are created before higher-level ones
		// This matters because a pattern's peak can be another pattern (e.g., level 2 pattern's peak is level 1)
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
			const pattern = new PatternNeuron(row.level, peak);
			pattern.peakStrength = row.strength;
			this.neurons.set(row.id, pattern);
			if (row.id > maxId) maxId = row.id;
		}
		console.log(`  Loaded ${patternRows.length} pattern neurons`);

		// 3. Load connections into SensoryNeuron.connections
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

		// 4. Load pattern_past into PatternNeuron.past
		const [pastRows] = await this.conn.query('SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM pattern_past');
		let pastCount = 0;
		for (const row of pastRows) {
			const pattern = this.neurons.get(row.pattern_neuron_id);
			const contextNeuron = this.neurons.get(row.context_neuron_id);
			if (!pattern || !contextNeuron) continue;
			if (pattern.level === 0) continue;

			if (!pattern.past.has(contextNeuron)) {
				pattern.past.set(contextNeuron, new Map());
				contextNeuron.incomingCount++;
			}
			pattern.past.get(contextNeuron).set(row.context_age, row.strength);
			pastCount++;
		}
		console.log(`  Loaded ${pastCount} pattern_past entries`);

		// 5. Load pattern_future into PatternNeuron.future
		const [futureRows] = await this.conn.query('SELECT pattern_neuron_id, inferred_neuron_id, distance, strength, reward FROM pattern_future');
		let futureCount = 0;
		for (const row of futureRows) {
			const pattern = this.neurons.get(row.pattern_neuron_id);
			const inferredNeuron = this.neurons.get(row.inferred_neuron_id);
			if (!pattern || !inferredNeuron) continue;
			if (pattern.level === 0) continue;

			if (!pattern.future.has(row.distance))
				pattern.future.set(row.distance, new Map());
			pattern.future.get(row.distance).set(inferredNeuron, {
				strength: row.strength,
				reward: row.reward
			});
			inferredNeuron.incomingCount++;
			futureCount++;
		}
		console.log(`  Loaded ${futureCount} pattern_future entries`);

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
		const coordRows = [];
		for (const [id, neuron] of this.neurons) {
			if (neuron.level !== 0) continue;
			const channelId = this.channelNameToId[neuron.channel];
			baseRows.push([id, channelId, neuron.type]);
			for (const [dimName, val] of Object.entries(neuron.coordinates)) {
				const dimId = this.dimensionNameToId[dimName];
				if (dimId !== undefined)
					coordRows.push([id, dimId, val]);
			}
		}
		if (baseRows.length > 0)
			await this.conn.query('INSERT INTO base_neurons (neuron_id, channel_id, type) VALUES ?', [baseRows]);
		if (coordRows.length > 0)
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [coordRows]);
		console.log(`  Saved ${baseRows.length} base neurons, ${coordRows.length} coordinates`);

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

		// 5. Backup pattern_past
		await this.conn.query('TRUNCATE pattern_past');
		const pastRows = [];
		for (const [patternId, neuron] of this.neurons) {
			if (neuron.level === 0) continue;
			for (const [contextNeuron, ageMap] of neuron.past) {
				const contextId = neuronToId.get(contextNeuron);
				if (!contextId) continue;
				for (const [age, strength] of ageMap)
					pastRows.push([patternId, contextId, age, strength]);
			}
		}
		if (pastRows.length > 0)
			await this.conn.query('INSERT INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age, strength) VALUES ?', [pastRows]);
		console.log(`  Saved ${pastRows.length} pattern_past entries`);

		// 6. Backup pattern_future
		await this.conn.query('TRUNCATE pattern_future');
		const futureRows = [];
		for (const [patternId, neuron] of this.neurons) {
			if (neuron.level === 0) continue;
			for (const [distance, targets] of neuron.future)
				for (const [inferredNeuron, pred] of targets) {
					const inferredId = neuronToId.get(inferredNeuron);
					if (inferredId)
						futureRows.push([patternId, inferredId, distance, pred.strength, pred.reward]);
				}
		}
		if (futureRows.length > 0)
			await this.conn.query('INSERT INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance, strength, reward) VALUES ?', [futureRows]);
		console.log(`  Saved ${futureRows.length} pattern_future entries`);

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
		if (!this.noDatabase) await this.loadNeurons();

		// pre-create action neurons for all channels
		await this.initializeActionNeurons();

		// initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) await channel.initialize();
	}

	/**
	 * Pre-create action neurons for all channels.
	 * This ensures exploration can find action neurons even before any connections exist.
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
			this.getFrameNeurons(framePoints);

			if (this.debug) console.log(`Created ${actionCoords.length} action neurons for ${channelName}`);
		}
	}

	/**
	 * Initialize channels in DB and load channel IDs
	 * IDs come from static Channel.nextId counter (not auto-increment)
	 */
	async initializeChannels() {
		this.channelNameToId = {};
		this.channelIdToName = {};

		// Insert channels into DB with explicit IDs from Channel objects
		for (const [channelName, channel] of this.channels) {
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
	 * Get frame actions for all channels from in-memory inferredNeurons Map.
	 * Reads winning action neurons (isWinner=true) grouped by channel.
	 * @returns {Map} - Map of channel names to array of output coordinates
	 */
	getInferredActions() {
		const channelOutputs = new Map();

		for (const [neuronId, inf] of this.inferredNeurons) {
			if (!inf.isWinner) continue;

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

		// activate base level neurons from the frame - what's happening right now?
		this.processBaseNeurons();

		// recognize, refine and learn patterns from the base neurons
		this.processPatternNeurons();

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
		if (this.debug) console.log('Aging active neurons...');

		for (const neuron of this.activeNeurons)
			neuron.age();
	}

	/**
	 * Deactivates neurons that have aged out of the context window.
	 * Called at end of frame after pattern learning, so patterns can capture full context.
	 */
	deactivateOldNeurons() {
		if (this.frameNumber < this.contextLength + 1) return;

		let deactivatedCount = 0;

		for (const neuron of this.activeNeurons) {
			deactivatedCount += neuron.deactivateAgedOut(this.contextLength);
			if (!neuron.isActive) this.activeNeurons.delete(neuron);
		}

		if (this.debug) console.log(`Deactivated ${deactivatedCount} aged-out neurons across all levels (age >= ${this.contextLength})`);
	}

	/**
	 * recognizes and activates base level neurons from frame
	 */
	processBaseNeurons() {

		// bulk find/create neurons for all input points
		const neuronIds = this.getFrameNeurons(this.frame);
		if (this.debug) console.log('frame neurons', neuronIds);

		// activate the neurons in the in-memory context
		this.activateNeurons(neuronIds);

		// reinforce connections between active neurons in the base level
		this.reinforceConnections();

		// learn from connection action inferences
		this.rewardConnections();

		// Track event prediction accuracy
		this.trackEventAccuracy();

		// Track action reward stats
		this.trackActionRewards();
	}

	/**
	 * Returns neuron IDs for given frame points, creating new neurons as needed.
	 * Points have structure: { coordinates, channel, channel_id, type }
	 */
	getFrameNeurons(frame) {
		const neuronIds = [];

		for (const point of frame) {
			const valueKey = SensoryNeuron.makeValueKey(point.coordinates);
			let neuron = this.neuronsByValue.get(valueKey);

			// Create new neuron if not found
			if (!neuron) {
				neuron = new SensoryNeuron(point.channel, point.type, point.coordinates);
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
	 * Activate a single neuron at age 0.
	 * @param {Neuron} neuron - The neuron to activate
	 */
	activateNeuron(neuron) {
		if (!neuron.activeAges) neuron.activeAges = new Set();
		neuron.activeAges.add(0);
		this.activeNeurons.add(neuron);
	}

	/**
	 * Activate neurons by ID at age 0.
	 * @param {Array<number>} neuronIds - Array of neuron IDs to activate
	 */
	activateNeurons(neuronIds) {
		for (const neuronId of neuronIds) {
			const neuron = this.neurons.get(neuronId);
			if (!neuron) throw new Error(`Neuron ${neuronId} not found in this.neurons`);
			this.activateNeuron(neuron);
		}
	}

	/**
	 * Get active sensory neurons as context (age > 0).
	 * @param {boolean} includeAgedOut - If true, include neurons at age >= contextLength (for pattern learning)
	 * @returns {Array<{neuron: SensoryNeuron, age: number}>}
	 */
	getSensoryContext(includeAgedOut = false) {
		const result = [];
		for (const neuron of this.activeNeurons) {
			if (neuron.level !== 0) continue;
			for (const age of neuron.activeAges)
				if (age > 0 && (includeAgedOut || age < this.contextLength))
					result.push({ neuron, age });
		}
		return result;
	}

	/**
	 * Get newly activated sensory neurons (age = 0).
	 * @param {string} [type] - Optional filter: 'event' or 'action'
	 * @returns {Array<SensoryNeuron>}
	 */
	getNewlyActivatedSensory(type) {
		const result = [];
		for (const neuron of this.activeNeurons) {
			if (neuron.level !== 0) continue;
			if (!neuron.isNewlyActive) continue;
			if (type && neuron.type !== type) continue;
			result.push(neuron);
		}
		return result;
	}

	/**
	 * Get active neurons at a specific level (all ages).
	 * @param {number} level - The level to filter by
	 * @returns {Array<{neuron: Neuron, age: number}>}
	 */
	/**
	 * Reinforce connections between base level active neurons - from age > 0 to age = 0
	 */
	reinforceConnections() {
		// Get event context (age > 0, < contextLength)
		const contextNeurons = this.getSensoryContext().filter(c => c.neuron.type === 'event');

		// Each newly activated sensory neuron reinforces its own incoming connections
		for (const neuron of this.getNewlyActivatedSensory())
			neuron.reinforceConnections(contextNeurons);
	}

	/**
	 * Apply rewards to action connections
	 */
	rewardConnections() {
		if (this.rewards.size === 0) return;

		// Get full sensory context (includes age=0 for self-connections if any)
		const contextNeurons = this.getSensoryContext();

		// Each newly activated action neuron applies its reward to incoming connections
		for (const neuron of this.getNewlyActivatedSensory('action')) {
			if (!this.rewards.has(neuron.channel)) continue;
			const reward = this.rewards.get(neuron.channel);
			neuron.applyReward(contextNeurons, reward);
		}
	}

	/**
	 * Build a CASE-WHEN SQL snippet for channel rewards.
	 * Maps channel_id to reward value for use in UPDATE statements.
	 * @param {string} rewardColumn - The column to use in ELSE clause (e.g., 'c.reward' or 'pf.reward')
	 * @returns {string} SQL CASE statement like "CASE b.channel_id WHEN 1 THEN 0.5 WHEN 2 THEN -0.3 ELSE c.reward END"
	 */
	buildChannelRewardCase(rewardColumn) {
		if (this.rewards.size === 0) return '';
		const caseWhen = Array.from(this.rewards.entries())
			.map(([name, reward]) => `WHEN ${this.channelNameToId[name]} THEN ${reward}`)
			.join(' ');
		return `CASE b.channel_id ${caseWhen} ELSE ${rewardColumn} END`;
	}

	/**
	 * Track event prediction accuracy by comparing inferredNeurons to activeNeurons (age=0).
	 * Only validates base level (level 0) event predictions.
	 * Only validates winners (isWinner=true) since losers are alternative hypotheses that were rejected.
	 */
	trackEventAccuracy() {
		let correct = 0;
		let total = 0;

		for (const [neuronId, inf] of this.inferredNeurons) {
			if (!inf.isWinner) continue;

			const neuron = this.neurons.get(neuronId);
			if (neuron.level !== 0 || neuron.type !== 'event') continue;

			total++;

			// Check if this neuron is active at age=0
			if (neuron.isNewlyActive) correct++;
		}

		// Update accuracyStats (cumulative across frames)
		this.accuracyStats.correct += correct;
		this.accuracyStats.total += total;

		if (this.debug && total > 0) {
			const accuracy = (correct / total * 100).toFixed(1);
			console.log(`Validated predictions: ${correct}/${total} (${accuracy}%)`);
		}
	}

	/**
	 * Track action reward stats by summing rewards for executed actions.
	 * Counts action neurons active at age=0 (just executed).
	 */
	trackActionRewards() {
		if (this.rewards.size === 0) return;

		const actionCount = this.getNewlyActivatedSensory('action').length;
		if (actionCount === 0) return;

		// Sum rewards from all channels for this frame
		let frameReward = 0;
		for (const [_, reward] of this.rewards)
			frameReward += reward;

		// Update rewardStats (cumulative across frames)
		this.rewardStats.totalReward += frameReward;
		this.rewardStats.count += actionCount;

		if (this.debug && actionCount > 0)
			console.log(`Action rewards: ${frameReward.toFixed(3)} for ${actionCount} actions`);
	}

	/**
	 * recognize, refine and learn patterns from the base neurons
	 */
	processPatternNeurons() {

		// discover and activate patterns using connections - start recursion from base level
		this.recognizePatterns();

		// refine the learned pattern definitions from prediction errors and action regret
		this.refinePatterns();

		// learn new patterns from failed predictions and action regret
		this.learnNewPatterns();
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
		const { peaks, context } = this.getPeaksAndContext(level);
		if (peaks.length === 0) {
			if (this.debug) console.log(`No newly activated neurons at level ${level}`);
			return false;
		}

		// Match patterns (parallelizable) - collect results without activating
		const matchedPatterns = peaks
			.map(peak => peak.matchBestPattern(context))
			.filter(pattern => pattern !== null);

		// Activate matched patterns (sequential) - for next level processing
		for (const pattern of matchedPatterns) this.activateNeuron(pattern);

		if (this.debug && matchedPatterns.length > 0)
			console.log(`Matched ${matchedPatterns.length} patterns at level ${level}:`,
				matchedPatterns.map(p => `peak=${p.peak.id}, pattern=${p.id}`).join('; '));

		// return true to indicate patterns found
		return matchedPatterns.length > 0;
	}

	/**
	 * Get peaks (age=0) and context (age>0) neurons at a specific level.
	 * Single traversal over activeNeurons.
	 * @param {number} level - The level to filter by
	 * @returns {{peaks: Array<Neuron>, context: Set<Neuron>}}
	 */
	getPeaksAndContext(level) {
		const peaks = [];
		const context = new Set();
		for (const neuron of this.activeNeurons) {
			if (neuron.level !== level) continue;
			for (const age of neuron.activeAges)
				if (age === 0) peaks.push(neuron);
				else if (age < this.contextLength) context.add(neuron);
		}
		return { peaks, context };
	}

	/**
	 * Learns patterns from prediction errors and action regret and continues to refine them as they are observed
	 */
	refinePatterns() {

		// update pattern_future with current events based on observations
		this.refinePatternEventsFuture();

		// update pattern_future with actions based on rewards
		this.refinePatternActionsFuture();
	}

	/**
	 * Refine pattern_future for event patterns with observed connections.
	 * Called during learning phase after pattern inference.
	 * Refinement happens when age = distance (prediction outcome just observed).
	 * This method only applies to EVENT patterns.
	 * Action pattern refinement happens in refinePatternActionsFuture.
	 * Applies three types of reinforcement:
	 * 1. Positive: Strengthen pattern_future inferences that were correctly predicted (target now active)
	 * 2. Negative: Weaken pattern_future inferences that were incorrectly predicted (target NOT active)
	 * 3. Novel: Add new inferences from pattern to newly observed neurons
	 */
	refinePatternEventsFuture() {

		// Get active event neurons at age=0
		const activeEventNeurons = new Set();
		for (const neuron of this.activeNeurons)
			if (neuron.level === 0 && neuron.type === 'event' && neuron.activeAges.has(0))
				activeEventNeurons.add(neuron);

		if (activeEventNeurons.size === 0) return;

		// Aggregate results
		let totalStrengthened = 0, totalWeakened = 0, totalNovel = 0;

		// Iterate active pattern neurons (level > 0, age > 0, age < contextLength)
		for (const neuron of this.activeNeurons) {
			if (neuron.level === 0) continue;

			for (const age of neuron.activeAges) {
				if (age === 0 || age >= this.contextLength) continue;

				// Refine this pattern's future predictions at this distance
				const result = neuron.refineEventFuture(age, activeEventNeurons);
				totalStrengthened += result.strengthened;
				totalWeakened += result.weakened;
				totalNovel += result.novel;
			}
		}

		if (this.debug) {
			console.log(`Strengthened ${totalStrengthened} correct event pattern_future predictions`);
			console.log(`Weakened ${totalWeakened} failed event pattern_future predictions`);
			console.log(`Added ${totalNovel} novel neurons to event pattern_future`);
		}
	}

	/**
	 * Refine pattern action predictions based on executed actions and rewards.
	 * Combines three operations:
	 * 1. Reward existing action predictions that were executed
	 * 2. Learn new actions that weren't predicted but were executed
	 * 3. Add alternative actions for painful outcomes
	 */
	refinePatternActionsFuture() {
		if (this.rewards.size === 0) return;

		// Get active action neurons at age=0
		const activeActionNeurons = new Set();
		for (const neuron of this.activeNeurons)
			if (neuron.level === 0 && neuron.type === 'action' && neuron.activeAges.has(0))
				activeActionNeurons.add(neuron);

		if (activeActionNeurons.size === 0) return;

		// Build map of channel -> all action neurons (for alternatives)
		const channelActions = new Map();
		for (const neuron of this.neurons.values())
			if (neuron.level === 0 && neuron.type === 'action') {
				if (!channelActions.has(neuron.channel)) channelActions.set(neuron.channel, new Set());
				channelActions.get(neuron.channel).add(neuron);
			}

		// Aggregate results
		let totalRewarded = 0, totalLearned = 0, totalAlternatives = 0;

		// Iterate active pattern neurons (level > 0, age > 0, age < contextLength)
		for (const neuron of this.activeNeurons) {
			if (neuron.level === 0) continue;

			for (const age of neuron.activeAges) {
				if (age === 0 || age >= this.contextLength) continue;

				// Refine this pattern's action predictions at this distance
				const result = neuron.refineActionFuture(age, activeActionNeurons, this.rewards, channelActions);
				totalRewarded += result.rewarded;
				totalLearned += result.learned;
				totalAlternatives += result.alternatives;
			}
		}

		if (this.debug) {
			console.log(`Rewarded ${totalRewarded} inferred actions`);
			console.log(`Learned ${totalLearned} new actions`);
			console.log(`Added ${totalAlternatives} alternative actions`);
		}
	}

	/**
	 * Creates error-driven patterns from failed predictions.
	 * Processes all levels in bulk, like inferNeurons.
	 */
	learnNewPatterns() {

		// Find the neurons that should be in pattern_future of new patterns
		// (prediction errors and action regret, unified in one method)
		const newPatternFutureCount = this.populateNewPatternFuture();
		if (this.debug) console.log(`New pattern future count: ${newPatternFutureCount}`);
		if (newPatternFutureCount === 0) return;

		// Populate new_patterns table with peaks from new pattern future inferences
		const patternCount = this.populateNewPatterns();
		if (this.debug) console.log(`Creating ${patternCount} error patterns`);

		// Create pattern neurons and map them to new_patterns
		this.createPatternNeurons();

		// Create new patterns: populate pattern past and future
		this.createNewPatterns();

		// Activate newly created pattern neurons so they can be refined in the same frame
		// and matched in future frames. Age = distance (when the peak was active)
		this.activateNewPatterns();

		if (this.debug) console.log(`Created ${patternCount} error patterns`);
	}

	/**
	 * Populate new_pattern_future with neurons that should be in pattern_future of new patterns.
	 * Unified method that handles both prediction errors and action regret.
	 *
	 * Two cases:
	 * 1. Prediction errors: inferred neuron NOT active at age=0 → pattern to predict what actually happened
	 * 2. Action regret: inferred action winner got negative reward → pattern to infer alternative action
	 *
	 * Returns the number of new pattern future inferences found.
	 */
	populateNewPatternFuture() {
		this.newPatternFuture = [];
		const eventCount = this.populateNewPatternEvents();
		const actionCount = this.populateNewPatternActions();
		return eventCount + actionCount;
	}

	/**
	 * Populate event prediction error neurons into new_pattern_future.
	 * Peak predicted X, but X didn't happen and Y happened instead.
	 * → Create pattern to predict Y from this context.
	 *
	 * Handles both connection and pattern inference errors.
	 * Works for all levels: if a level N neuron predicts a base event,
	 * and it doesn't happen, create a level N+1 pattern to correct the prediction.
	 * @returns {number} Number of event pattern futures created
	 */
	populateNewPatternEvents() {
		// Filter votes by strength threshold
		const strongVotes = this.inferenceVotes.filter(v => v.strength >= this.eventErrorMinStrength);
		if (strongVotes.length === 0) return 0;

		// Get active event neurons at age=0 (what actually happened)
		const activeEventNeurons = new Set();
		for (const neuron of this.activeNeurons)
			if (neuron.level === 0 && neuron.type === 'event' && neuron.activeAges.has(0))
				activeEventNeurons.add(neuron);

		if (activeEventNeurons.size === 0) return 0;

		// Filter to valid votes: any level voter, event target, prediction error
		// Base voters (level 0) must be events; pattern voters (level > 0) can be any type
		const errorVotes = strongVotes.filter(v => {
			const voter = this.neurons.get(v.fromNeuronId);
			const target = this.neurons.get(v.neuronId);
			if (!voter || !target) return false;
			if (target.level > 0 || target.type !== 'event') return false;
			if (target.activeAges?.has(0)) return false;  // Not an error if target is active
			// Base voters must be events, pattern voters can be any type
			return voter.level > 0 || voter.type === 'event';
		});

		if (errorVotes.length === 0) return 0;

		// Create pattern futures: for each error vote, pair with each actual event
		let count = 0;
		for (const v of errorVotes)
			for (const actualNeuron of activeEventNeurons) {
				this.newPatternFuture.push({ peakNeuronId: v.fromNeuronId, inferredNeuronId: actualNeuron.id, distance: v.distance });
				count++;
			}

		if (this.debug) console.log(`Found ${count} event prediction error neurons`);
		return count;
	}

	/**
	 * Populate action regret neurons into new_pattern_future.
	 * When a connection or pattern predicts an action and it leads to pain,
	 * add one untried alternative action.
	 * @returns {number} Number of action pattern futures created
	 */
	populateNewPatternActions() {
		// Filter votes by strength threshold
		const strongVotes = this.inferenceVotes.filter(v => v.strength >= this.actionRegretMinStrength);
		if (strongVotes.length === 0) return 0;

		// Get painful channels - if none, no regret patterns needed
		const painfulChannels = new Set();
		for (const [channelName, reward] of this.rewards)
			if (reward < PatternNeuron.actionRegretMinPain) painfulChannels.add(channelName);
		if (painfulChannels.size === 0) return 0;

		// Get winning action neuron IDs
		const winnerIds = new Set();
		for (const [neuronId, inf] of this.inferredNeurons)
			if (inf.isWinner) winnerIds.add(neuronId);

		// Build map of channel -> all action neurons (for alternatives)
		const channelActions = new Map();
		for (const neuron of this.neurons.values())
			if (neuron.level === 0 && neuron.type === 'action') {
				if (!channelActions.has(neuron.channel)) channelActions.set(neuron.channel, []);
				channelActions.get(neuron.channel).push(neuron);
			}

		// Filter to valid regret votes:
		// - Voter: base events (level 0, type event) or patterns (level > 0)
		// - Target: action in painful channel that was a winner
		const regretVotes = strongVotes.filter(v => {
			const voter = this.neurons.get(v.fromNeuronId);
			const target = this.neurons.get(v.neuronId);
			if (!voter || !target) return false;
			if (target.level !== 0 || target.type !== 'action') return false;
			if (!painfulChannels.has(target.channel)) return false;
			if (!winnerIds.has(v.neuronId)) return false;
			// Base voters must be events, pattern voters can be any type
			return voter.level > 0 || voter.type === 'event';
		});

		if (regretVotes.length === 0) return 0;

		// Create pattern futures: for each regret vote, pick one alternative action
		let count = 0;
		for (const v of regretVotes) {
			const target = this.neurons.get(v.neuronId);
			const alts = channelActions.get(target.channel) || [];
			const altNeuron = alts.find(n => n.id !== v.neuronId);
			if (altNeuron) {
				this.newPatternFuture.push({ peakNeuronId: v.fromNeuronId, inferredNeuronId: altNeuron.id, distance: v.distance });
				count++;
			}
		}

		if (this.debug) console.log(`Found ${count} action regret inferences`);
		return count;
	}

	/**
	 * Populate new_patterns from new_pattern_future. Peak can be a base neuron (for connection errors)
	 * or pattern neuron (for pattern errors). One pattern per peak - combines contexts from all ages where
	 * the peak was active and made bad predictions.
	 * Returns the number of patterns to create.
	 */
	populateNewPatterns() {
		// Get distinct peak neuron IDs from newPatternFuture
		const distinctPeaks = [...new Set(this.newPatternFuture.map(f => f.peakNeuronId))];
		this.newPatterns = distinctPeaks.map(peakNeuronId => ({ peakNeuronId, patternNeuronId: null }));
		return this.newPatterns.length;
	}

	/**
	 * Creates pattern neurons from this.newPatterns at peak neuron level + 1
	 */
	createPatternNeurons() {
		if (this.newPatterns.length === 0) return;

		for (const p of this.newPatterns) {
			const peakNeuron = this.neurons.get(p.peakNeuronId);
			const pattern = new PatternNeuron(peakNeuron.level + 1, peakNeuron);
			this.neurons.set(pattern.id, pattern);
			p.patternNeuronId = pattern.id;
		}
	}

	/**
	 * Populate pattern past and future for newly created patterns.
	 * Peak is already set in PatternNeuron constructor.
	 */
	createNewPatterns() {
		if (this.newPatterns.length === 0) return;

		// Build map of peakNeuronId -> pattern for lookups
		const peakToPattern = new Map();
		for (const p of this.newPatterns)
			peakToPattern.set(p.peakNeuronId, this.neurons.get(p.patternNeuronId));

		// Get distinct (peakNeuronId, distance) pairs from newPatternFuture
		const peakDistances = new Map();
		for (const f of this.newPatternFuture) {
			const key = `${f.peakNeuronId}:${f.distance}`;
			if (!peakDistances.has(key)) peakDistances.set(key, { peakNeuronId: f.peakNeuronId, distance: f.distance });
		}

		// Populate pattern past - context neurons at appropriate ages
		for (const { peakNeuronId, distance } of peakDistances.values()) {
			const pattern = peakToPattern.get(peakNeuronId);
			const peakLevel = pattern.peak.level;

			// Get context neurons: same level as peak, age > distance and < distance + contextLength, not actions
			for (const neuron of this.activeNeurons) {
				if (neuron.level !== peakLevel) continue;
				if (neuron.level === 0 && neuron.type === 'action') continue;

				for (const age of neuron.activeAges) {
					if (age <= distance || age >= distance + this.contextLength) continue;
					const contextAge = age - distance;
					pattern.addContext(neuron, contextAge);
				}
			}
		}

		// Populate pattern future from newPatternFuture
		for (const f of this.newPatternFuture) {
			const pattern = peakToPattern.get(f.peakNeuronId);
			const inferredNeuron = this.neurons.get(f.inferredNeuronId);
			pattern.getOrCreateFuture(f.distance, inferredNeuron);
		}

		if (this.debug) console.log(`Created ${this.newPatternFuture.length} pattern_future entries`);
	}

	/**
	 * Activate newly created pattern neurons at ALL ages where the peak was active.
	 * If peak was active at ages 1, 4, 7, the pattern is activated at ages 1, 4, and 7.
	 * This allows the pattern to be refined in the same frame if there are multiple errors,
	 * and ensures the "already has active pattern" check works correctly in future frames.
	 */
	activateNewPatterns() {
		if (this.newPatterns.length === 0) return;

		// Build map of peakNeuronId -> pattern
		const peakToPattern = new Map();
		for (const p of this.newPatterns)
			peakToPattern.set(p.peakNeuronId, this.neurons.get(p.patternNeuronId));

		// Collect distinct (pattern, distance) pairs
		const patternAges = new Map();  // pattern -> Set of ages
		for (const f of this.newPatternFuture) {
			const pattern = peakToPattern.get(f.peakNeuronId);
			if (!patternAges.has(pattern)) patternAges.set(pattern, new Set());
			patternAges.get(pattern).add(f.distance);
		}

		// Activate each pattern at its ages
		let count = 0;
		for (const [pattern, ages] of patternAges) {
			pattern.activeAges = ages;
			this.activeNeurons.add(pattern);
			count += ages.size;
		}

		if (this.debug) console.log(`Activated ${count} new pattern neuron instances`);
	}

	/**
	 * Infer predictions and outputs using voting architecture.
	 * All levels vote for both actions and events.
	 */
	inferNeurons() {

		// Collect votes from active neurons
		this.collectVotes();

		// Filter out votes from peaks that have active patterns voting
		this.suppressOverriddenVotes();

		// Aggregate votes and determine winners
		const inferences = this.determineConsensus();

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
	 * Connection votes from base neurons, pattern votes from pattern neurons.
	 * Populates this.inferenceVotes array for use by pattern creation.
	 */
	collectVotes() {
		this.inferenceVotes = [];
		const timeDecay = 1 / this.contextLength;

		let connCount = 0, patternCount = 0;

		for (const neuron of this.activeNeurons) {
			for (const age of neuron.activeAges) {
				const distance = age + 1;
				const levelWeight = 1 + neuron.level * this.levelVoteMultiplier;
				const timeWeight = 1 - (distance - 1) * timeDecay;

				// Connection votes from base neurons
				if (neuron.level === 0) {
					const distanceMap = neuron.connections.get(distance);
					if (!distanceMap) continue;
					for (const [toNeuron, conn] of distanceMap) {
						if (conn.strength <= 0) continue;
						this.inferenceVotes.push({
							fromNeuronId: neuron.id,
							neuronId: toNeuron.id,
							strength: levelWeight * timeWeight * conn.strength,
							reward: conn.reward,
							distance
						});
						connCount++;
					}
				}
				// Pattern votes from pattern neurons
				else {
					const distanceMap = neuron.future.get(distance);
					if (!distanceMap) continue;
					for (const [inferredNeuron, pred] of distanceMap) {
						if (pred.strength <= 0) continue;
						this.inferenceVotes.push({
							fromNeuronId: neuron.id,
							neuronId: inferredNeuron.id,
							strength: levelWeight * timeWeight * pred.strength,
							reward: pred.reward,
							distance
						});
						patternCount++;
					}
				}
			}
		}

		if (this.debug) console.log(`Collected ${connCount} connection votes, ${patternCount} pattern votes`);
	}

	/**
	 * Filter out votes from neurons that are peaks of active voting patterns.
	 * Pattern votes override their peak's connection votes.
	 */
	suppressOverriddenVotes() {
		// Find peaks of all voting patterns
		const overriddenPeaks = new Set();
		for (const v of this.inferenceVotes) {
			const voter = this.neurons.get(v.fromNeuronId);
			if (voter.level > 0) overriddenPeaks.add(voter.peak.id);
		}

		if (overriddenPeaks.size === 0) return;

		const beforeCount = this.inferenceVotes.length;
		this.inferenceVotes = this.inferenceVotes.filter(v => !overriddenPeaks.has(v.fromNeuronId));
		if (this.debug) console.log(`Filtered ${beforeCount - this.inferenceVotes.length} overridden votes`);
	}

	/**
	 * Aggregate votes and determine winners per dimension.
	 * Events win by strength, actions win by reward.
	 * @returns {Array} Array of inference objects with isWinner flag
	 */
	determineConsensus() {

		// Aggregate votes by neuron
		const aggregated = new Map(); // neuronId -> {neuron, strength, weightedReward}
		for (const v of this.inferenceVotes) {
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
			for (const dim of Object.keys(c.coordinates)) {
				if (!dimBest.has(dim) || score > dimBest.get(dim).score)
					dimBest.set(dim, { neuronId: c.neuron_id, score });
			}
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
	 * Save all inferences to in-memory Map.
	 * @param {Array} inferences - Array of inference objects with isWinner flag
	 */
	saveInferences(inferences) {

		// Clear and populate in-memory map
		this.inferredNeurons.clear();
		for (const inf of inferences)
			this.inferredNeurons.set(inf.neuron_id, { strength: inf.strength, isWinner: inf.isWinner });

		if (this.debug) {
			const winnerCount = inferences.filter(i => i.isWinner).length;
			const loserCount = inferences.filter(i => !i.isWinner).length;
			console.log(`Saved ${inferences.length} inferences (${winnerCount} winners, ${loserCount} losers)`);
		}
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

		// 1. Connection forgetting
		const connStats = this.forgetConnections();
		if (this.debug) console.log(`  Connections: ${connStats.updated} weakened, ${connStats.deleted} deleted`);

		// 2. Pattern forgetting
		const patternStats = this.forgetPatterns();
		if (this.debug) console.log(`  Patterns: past ${patternStats.pastDeleted}, future ${patternStats.futureDeleted}, peaks ${patternStats.peaksDeleted}`);

		// 3. Orphan cleanup
		const orphanCount = this.deleteOrphanedPatterns();
		if (this.debug) console.log(`  Orphaned patterns deleted: ${orphanCount}`);

		if (this.debug) console.log(`=== FORGET CYCLE COMPLETED in ${Date.now() - cycleStart}ms ===\n`);
	}

	/**
	 * Reduce connection strengths and delete dead connections.
	 * @returns {{updated: number, deleted: number}} Stats
	 */
	forgetConnections() {
		let updated = 0, deleted = 0;

		for (const neuron of this.neurons.values()) {
			if (neuron.level !== 0) continue;

			for (const [distance, distanceMap] of neuron.connections) {
				const toDelete = [];
				for (const [toNeuron, conn] of distanceMap) {
					if (conn.strength <= 0) continue;
					conn.strength = Math.max(Neuron.minStrength, conn.strength - this.connectionForgetRate);
					updated++;
					if (conn.strength <= Neuron.minStrength) toDelete.push(toNeuron);
				}
				for (const toNeuron of toDelete) {
					neuron.deleteConnection(distance, toNeuron);
					deleted++;
				}
			}
		}

		return { updated, deleted };
	}

	/**
	 * Reduce pattern strengths and delete dead pattern entries.
	 * @returns {{pastDeleted: number, futureDeleted: number, peaksDeleted: number}} Stats
	 */
	forgetPatterns() {
		let pastDeleted = 0, futureDeleted = 0, peaksDeleted = 0;

		for (const neuron of this.neurons.values()) {
			if (neuron.level === 0) continue;

			// Forget pattern past
			const pastToDelete = [];
			for (const [contextNeuron, ageMap] of neuron.past) {
				for (const [age, strength] of ageMap) {
					const newStrength = Math.max(Neuron.minStrength, strength - this.patternForgetRate);
					if (newStrength <= Neuron.minStrength)
						pastToDelete.push({ contextNeuron, age });
					else
						ageMap.set(age, newStrength);
				}
			}
			for (const { contextNeuron, age } of pastToDelete) {
				neuron.removeContext(contextNeuron, age);
				pastDeleted++;
			}

			// Forget pattern future
			const futureToDelete = [];
			for (const [distance, distanceMap] of neuron.future) {
				for (const [inferredNeuron, pred] of distanceMap) {
					if (pred.strength <= 0) continue;
					pred.strength = Math.max(Neuron.minStrength, pred.strength - this.patternForgetRate);
					if (pred.strength <= Neuron.minStrength)
						futureToDelete.push({ distance, inferredNeuron });
				}
			}
			for (const { distance, inferredNeuron } of futureToDelete) {
				neuron.deleteFuture(distance, inferredNeuron);
				futureDeleted++;
			}

			// Forget peak strength
			if (neuron.peakStrength > 0) {
				neuron.peakStrength = Math.max(0, neuron.peakStrength - this.patternForgetRate);
				if (neuron.peakStrength <= 0) peaksDeleted++;
			}
		}

		return { pastDeleted, futureDeleted, peaksDeleted };
	}

	/**
	 * Delete orphaned pattern neurons (no content, no references, not active).
	 * @returns {number} Number of patterns deleted
	 */
	deleteOrphanedPatterns() {
		const toDelete = [];

		for (const neuron of this.neurons.values()) {
			if (neuron.level === 0) continue;
			if (neuron.canDelete(this)) toDelete.push(neuron);
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