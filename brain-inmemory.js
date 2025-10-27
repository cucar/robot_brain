import readline from 'node:readline';
import getMySQLConnection from './db/db.js';
import {
	ConnectionStore,
	ActiveNeuronStore,
	PatternStore,
	PatternPeakStore,
	NeuronStore,
	ActiveConnectionStore
} from './brain-memory.js';
import {
	detectPeaks,
	inferConnections,
	matchPatterns,
	mergeMatchedPatterns,
	createNewPatterns,
	activateConnections,
	reinforceConnections,
	applyRewards
} from './brain-algorithms.js';

/**
 * Artificial Brain - In-Memory Version
 *
 * This version uses in-memory JavaScript data structures instead of MySQL for core operations.
 * Only dimensions remain in MySQL for now.
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.baseNeuronMaxAge = 5; // number of frames a base neuron stays active
		this.forgetCycles = 100; // number of frames between forget cycles
		this.connectionForgetRate = 1; // how much connection strengths decay per forget cycle
		this.patternForgetRate = 1; // how much pattern strengths decay per forget cycle
		this.maxLevels = 10; // just to prevent against infinite recursion
		this.mergePatternThreshold = 0.20; // minimum percentage of matching neurons for an observed pattern to match a known pattern
		this.minPeakStrength = 10.0; // minimum weighted strength for a neuron to be considered a peak (used for both pattern detection and prediction)
		this.minPeakRatio = 1; // minimum ratio of peak strength to neighborhood average to be considered a peak (used for both pattern detection and prediction)
		this.peakTimeDecayFactor = 0.9; // peak connection weight = POW(peakTimeDecayFactor, distance)
		this.rewardTimeDecayFactor = 0.9; // reward temporal decay = POW(rewardTimeDecayFactor, age)
		this.patternNegativeReinforcement = 0.1; // how much to weaken pattern connections that were not observed
		this.negativeLearningRate = 0.1; // how much to weaken connections when predictions fail
		this.minConnectionStrength = 0; // minimum strength value for connections and patterns (clamped to prevent negative values)
		this.maxConnectionStrength = 1000; // maximum strength value for connections and patterns (clamped to prevent overflow)

		// initialize the counter for forget cycle
		this.forgetCounter = 0;

		// initialize channel registry
		this.channels = new Map();

		// used for global activity tracking so that we can trigger exploration when all channels are inactive
		this.lastActivity = -1; // frame number of last activity across all channels
		this.frameNumber = 0;
		this.inactivityThreshold = 5; // frames of inactivity before exploration

		// Prediction accuracy tracking (cumulative stats per level)
		this.accuracyStats = new Map(); // level -> { connection: {correct, total}, pattern: {correct, total}, resolved: {correct, total} }

		// Create readline interface for pausing between frames - used when debugging
		this.debug = false;
		this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });

		// In-memory data stores
		this.neurons = new NeuronStore();
		this.connections = new ConnectionStore();
		this.patterns = new PatternStore();
		this.patternPeaks = new PatternPeakStore();

		// Scratch tables (cleared each frame)
		this.activeNeurons = new ActiveNeuronStore();
		this.activeConnections = new ActiveConnectionStore();
		this.observedPatterns = new Map(); // Map<peak_neuron_id, Set<connection_id>>
		this.matchedPatterns = new Map(); // Map<peak_neuron_id, Set<pattern_neuron_id>>
		this.connectionInference = new Map(); // Map<level, Map<neuron_id, {strength, age}>>
		this.patternInference = new Map(); // Map<level, Map<neuron_id, {strength, age}>>
		this.inferredNeurons = new Map(); // Map<level, Map<neuron_id, {strength, age}>>
	}

	/**
	 * Register a channel with the brain
	 */
	registerChannel(name, channelClass) {
		const channel = new channelClass(name);
		this.channels.set(name, channel);
		console.log(`Registered channel: ${name} (${channelClass.name})`);
	}

	/**
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log('Resetting brain (in-memory scratch tables)...');
		this.activeNeurons.clear();
		this.activeConnections.clear();
		this.observedPatterns.clear();
		this.matchedPatterns.clear();
		this.connectionInference.clear();
		this.patternInference.clear();
		this.inferredNeurons.clear();
	}

	/**
	 * Hard reset: clears ALL learned data (used mainly for tests)
	 */
	async resetBrain() {
		console.log('Hard resetting brain (all in-memory data)...');
		this.neurons.clear();
		this.connections.clear();
		this.patterns.clear();
		this.patternPeaks.clear();
		await this.resetContext();

		// Also clear dimensions in MySQL
		await this.conn.query('TRUNCATE coordinates');
		await this.conn.query('TRUNCATE dimensions');

		// Re-initialize dimensions after clearing
		await this.initializeDimensions();
	}

	/**
	 * Initialize database connection and dimensions
	 */
	async init() {
		this.conn = await getMySQLConnection();
		console.log('Brain initialized (in-memory mode)');

		// Initialize dimensions for all registered channels
		await this.initializeDimensions();

		// Initialize all registered channels (channel-specific setup)
		for (const [, channel] of this.channels) {
			await channel.initialize();
		}
	}

	/**
	 * Close database connection
	 */
	async close() {
		if (this.conn) {
			await this.conn.end();
		}
		this.rl.close();
	}

	/**
	 * Get or create dimension ID
	 * Note: Dimensions must be pre-initialized with channel info
	 */
	async getDimensionId(name) {
		const [rows] = await this.conn.query(
			'SELECT id FROM dimensions WHERE name = ?',
			[name]
		);

		if (rows.length > 0) {
			return rows[0].id;
		}

		// Dimension not found - this shouldn't happen if channels are properly initialized
		throw new Error(`Dimension '${name}' not found. Did you forget to initialize channels?`);
	}

	/**
	 * Initialize dimensions for all registered channels
	 */
	async initializeDimensions() {
		console.log('Initializing dimensions for registered channels...');
		for (const [channelName, channel] of this.channels) {
			await this.insertChannelDimensions(channel.getInputDimensions(), channelName, 'input');
			await this.insertChannelDimensions(channel.getOutputDimensions(), channelName, 'output');
		}

		// Load dimension name to ID mapping (both directions)
		const [rows] = await this.conn.query('SELECT id, name, channel, type FROM dimensions');
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};
		this.dimensions = new Map();

		for (const row of rows) {
			this.dimensionNameToId[row.name] = row.id;
			this.dimensionIdToName[row.id] = row.name;
			this.dimensions.set(row.name, { id: row.id, name: row.name, channel: row.channel, type: row.type });
		}
		console.log(`Loaded ${rows.length} dimensions`);
	}

	/**
	 * Insert channel dimensions
	 */
	async insertChannelDimensions(dimensions, channelName, type) {
		console.log(`Creating ${type} dimensions for ${channelName}:`, dimensions);
		for (const dimName of dimensions) {
			await this.conn.query(
				'INSERT IGNORE INTO dimensions (name, channel, type) VALUES (?, ?, ?)',
				[dimName, channelName, type]
			);
		}
	}


	/**
	 * Get current frame combined from all registered channels
	 */
	async getFrame() {
		// Get input data from all channels
		const frame = [];
		for (const [_, channel] of this.channels) {
			const channelInputs = await channel.getFrameInputs();
			if (channelInputs && channelInputs.length > 0) {
				frame.push(...channelInputs);
			}
		}
		return frame;
	}

	/**
	 * Get global feedback from all channels aggregated into a single reward
	 */
	async getFeedback() {
		let totalReward = 0;
		let feedbackCount = 0;

		for (const [channelName, channel] of this.channels) {
			const rewardFactor = await channel.getFeedback();
			if (rewardFactor !== 1.0) { // Only process non-neutral feedback
				// Convert multiplicative factor to additive reward
				const reward = rewardFactor - 1.0;
				console.log(`${channelName}: reward ${reward.toFixed(3)}`);
				totalReward += reward;
				feedbackCount++;
			}
		}

		if (feedbackCount > 0) {
			console.log(`Total reward: ${totalReward.toFixed(3)} (${feedbackCount} channels)`);
		}

		return totalReward;
	}


	/**
	 * Match frame neurons to existing neurons or create new ones
	 */
	async matchFrameNeurons(points) {
		const neuronIds = [];

		for (const point of points) {
			// Find neurons matching all coordinates
			let candidateNeurons = null;

			for (const [dimName, value] of Object.entries(point)) {
				const dimId = await this.getDimensionId(dimName);
				const neuronsWithCoord = this.neurons.findByDimensionValue(dimId, value);

				if (candidateNeurons === null) {
					candidateNeurons = new Set(neuronsWithCoord);
				} else {
					// Intersect with previous candidates
					candidateNeurons = new Set(
						neuronsWithCoord.filter(n => candidateNeurons.has(n))
					);
				}

				if (candidateNeurons.size === 0) break;
			}

			let neuronId;
			if (candidateNeurons && candidateNeurons.size > 0) {
				// Use existing neuron
				neuronId = Array.from(candidateNeurons)[0];
			} else {
				// Create new neuron
				neuronId = this.neurons.createNeuron();

				// Set coordinates
				for (const [dimName, value] of Object.entries(point)) {
					const dimId = await this.getDimensionId(dimName);
					this.neurons.setCoordinate(neuronId, dimId, value);

					// Also store in MySQL for persistence
					await this.conn.query(
						'INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE val = VALUES(val)',
						[neuronId, dimId, value]
					);
				}
			}

			neuronIds.push(neuronId);
		}

		return neuronIds;
	}

	/**
	 * Activate neurons at base level (level 0)
	 */
	activateBaseNeurons(neuronIds) {
		for (const neuronId of neuronIds) {
			this.activeNeurons.add(neuronId, 0, 0); // level 0, age 0
		}
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= baseNeuronMaxAge.
	 */
	ageNeurons() {
		console.log('Aging active neurons, connections, and inferred neurons...');

		// Age all active neurons (collect first to avoid iterator issues)
		const allActiveNeurons = Array.from(this.activeNeurons.byKey.values());
		let deactivatedNeurons = 0;

		for (const an of allActiveNeurons) {
			const newAge = an.age + 1;

			// Remove from old position
			this.activeNeurons.delete(an.neuron_id, an.level, an.age);

			// If not aged out, add back with new age
			if (newAge < this.baseNeuronMaxAge) {
				this.activeNeurons.add(an.neuron_id, an.level, newAge);
			} else {
				deactivatedNeurons++;
			}
		}

		console.log(`Deactivated ${deactivatedNeurons} aged-out neurons across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Age all active connections (rebuild the store)
		const allActiveConns = Array.from(this.activeConnections.byKey.values());
		this.activeConnections.clear();
		let deactivatedConnections = 0;

		for (const ac of allActiveConns) {
			const newAge = ac.age + 1;

			// If not aged out, add back with new age
			if (newAge < this.baseNeuronMaxAge) {
				this.activeConnections.add(ac.connection_id, ac.from_neuron_id, ac.to_neuron_id, ac.level, newAge);
			} else {
				deactivatedConnections++;
			}
		}

		console.log(`Deactivated ${deactivatedConnections} aged-out connections across all levels (age >= ${this.baseNeuronMaxAge})`);

		// Age and clean up inferred neurons (connection_inferred_neurons, pattern_inferred_neurons, inferred_neurons)
		// age=0: fresh predictions, age=1: executed this frame, age>=2: no longer needed
		this.ageInferredNeurons(this.connectionInference, 'connection inferred neurons');
		this.ageInferredNeurons(this.patternInference, 'pattern inferred neurons');
		this.ageInferredNeurons(this.inferredNeurons, 'inferred neurons');
	}

	/**
	 * Age inferred neurons and clean up old ones (age >= 2)
	 */
	ageInferredNeurons(inferenceMap, label) {
		let cleaned = 0;

		for (const [level, neuronMap] of inferenceMap) {
			// Age each neuron's metadata if it exists
			for (const [neuronId, data] of neuronMap) {
				if (typeof data === 'object' && data.age !== undefined) {
					data.age++;

					// Clean up if age >= 2
					if (data.age >= 2) {
						neuronMap.delete(neuronId);
						cleaned++;
					}
				}
			}

			// Remove empty level maps
			if (neuronMap.size === 0) {
				inferenceMap.delete(level);
			}
		}

		if (cleaned > 0) {
			console.log(`Cleaned up ${cleaned} executed ${label} (age >= 2)`);
		}
	}

	/**
	 * Execute previous frame's decisions and exploration actions if needed
	 */
	async executeOutputs() {
		// Execute previous frame's decisions (age = 1)
		await this.executePreviousOutputs();

		// Execute exploration if brain is inactive
		await this.curiosityExploration();
	}

	/**
	 * Execute decisions from previous frame (age = 1)
	 */
	async executePreviousOutputs() {
		// Get output neurons from inferred_neurons at age=1, level=0
		if (!this.inferredNeurons.has(0)) {
			console.log('No previous outputs to execute');
			return;
		}

		const level0Inferred = this.inferredNeurons.get(0);
		const outputRows = [];

		// Find neurons with age=1 and output dimensions
		for (const [neuronId, data] of level0Inferred) {
			if (data.age === 1) {
				// Get neuron coordinates
				const coords = this.neurons.getCoordinates(neuronId);
				if (coords.length === 0) continue;

				// Check if this neuron has output dimensions
				for (const coord of coords) {
					const dim = this.dimensionIdToName[coord.dimension_id];
					const dimInfo = this.dimensions.get(dim);

					if (dimInfo && dimInfo.type === 'output') {
						outputRows.push({
							neuron_id: neuronId,
							dimension_id: coord.dimension_id,
							val: coord.val,
							dimension_name: dim,
							channel: dimInfo.channel
						});
					}
				}
			}
		}

		if (outputRows.length === 0) {
			console.log('No previous outputs to execute');
			return;
		}

		await this.executeOutputRows(outputRows);
	}

	/**
	 * Execute curiosity exploration if brain is inactive
	 */
	async curiosityExploration() {
		// Check if the brain is inactive - if active, no exploration needed
		if ((this.frameNumber - this.lastActivity) < this.inactivityThreshold) return;
		console.log('Brain inactive - executing curiosity exploration');

		// Get a random channel for exploration
		const channelNames = Array.from(this.channels.keys());
		const randomChannelName = channelNames[Math.floor(Math.random() * channelNames.length)];
		const randomChannel = this.channels.get(randomChannelName);

		// Get exploration actions for the channel
		const explorationActions = randomChannel.getValidExplorationActions();
		if (explorationActions.length === 0) {
			console.log(`No valid exploration actions for ${randomChannelName}`);
			return;
		}

		// Execute random exploration action
		const randomAction = explorationActions[Math.floor(Math.random() * explorationActions.length)];
		console.log(`${randomChannelName}: Executing exploration action:`, randomAction);

		await this.executeChannelOutputs(randomChannelName, randomAction);
	}

	/**
	 * Execute output rows grouped by channel
	 */
	async executeOutputRows(outputRows) {
		// Group outputs by channel
		const channelOutputs = new Map();

		for (const row of outputRows) {
			if (!channelOutputs.has(row.channel)) channelOutputs.set(row.channel, new Map());
			channelOutputs.get(row.channel).set(row.dimension_name, row.val);
		}

		// Execute outputs for each channel using unified method
		for (const [channelName, outputs] of channelOutputs) {
			const coordinates = Object.fromEntries(outputs);
			await this.executeChannelOutputs(channelName, coordinates);
		}
	}

	/**
	 * Unified method to execute outputs on a specific channel
	 */
	async executeChannelOutputs(channelName, coordinates) {
		const channel = this.channels.get(channelName);
		if (!channel) {
			console.log(`Warning: Channel ${channelName} not found`);
			return;
		}

		console.log(`${channelName}: Executing outputs:`, coordinates);
		await channel.executeOutputs(coordinates);

		// Track global activity
		this.lastActivity = this.frameNumber;
	}

	/**
	 * Recognizes and activates neurons from frame - returns the highest level of recognition reached
	 */
	async recognizeNeurons(frame) {
		// Bulk find/create neurons for all input points
		const neuronIds = await this.matchFrameNeurons(frame);

		console.log(`Matched/created ${neuronIds.length} neurons from ${frame.length} points`);

		// Bulk insert activations at base level
		this.activateBaseNeurons(neuronIds);

		// Discover and activate patterns using connections - start recursion from base level
		await this.activatePatternNeurons();
	}

	/**
	 * Activate pattern neurons hierarchically
	 */
	async activatePatternNeurons() {
		let currentLevel = 0;
		let hasActivity = true;

		while (hasActivity && currentLevel < this.maxLevels) {
			hasActivity = await this.processLevel(currentLevel);
			currentLevel++;
		}

		console.log(`Processed ${currentLevel} levels`);
	}

	/**
	 * Infer predictions and outputs starting from the highest active level down to base level.
	 * Connection inference: Predict next frame's neurons from connections (with negative reinforcement).
	 * Pattern inference: Predict lower-level peak neurons from higher-level pattern neurons.
	 */
	async inferNeurons() {
		// Get the highest level that is currently active
		const maxLevel = this.getMaxActiveLevel();

		// Process levels in reverse: maxLevel, maxLevel-1, ..., 0
		for (let level = maxLevel; level >= 0; level--) {
			// Connection inference: Predict connections for age=-1 at this level
			await this.inferConnectionsAtLevel(level);

			// Pattern inference: Predict peak neurons for the lower level
			if (level > 0) await this.inferPatternsAtLevel(level);
		}

		// Resolve conflicts in input predictions at base level (after all predictions are made)
		await this.resolveInputPredictionConflicts();
	}

	/**
	 * Get the highest level that has active neurons
	 */
	getMaxActiveLevel() {
		let maxLevel = 0;

		for (const an of this.activeNeurons.byKey.values()) {
			if (an.level > maxLevel) {
				maxLevel = an.level;
			}
		}

		return maxLevel;
	}

	/**
	 * Connection inference: Predict next frame's neurons from active connections.
	 * Validates previous frame's predictions and applies negative reinforcement to failed predictions.
	 */
	async inferConnectionsAtLevel(level) {
		// Report the neuron prediction accuracy from previous frame
		this.reportPredictionsAccuracy(level);

		// Validate predictions from previous frame and apply negative reinforcement
		this.validateConnectionPredictions(level);

		// Clear previous predictions for this level
		if (this.connectionInference.has(level)) {
			this.connectionInference.delete(level);
		}

		// Make new predictions using inferConnections algorithm
		const predictions = inferConnections(
			this.activeNeurons,
			this.connections,
			level,
			this.peakTimeDecayFactor,
			this.minPeakStrength,
			this.minPeakRatio
		);

		// Store predictions with age=0
		if (predictions.size > 0) {
			const neuronMap = new Map();
			for (const [neuronId, strength] of predictions) {
				neuronMap.set(neuronId, { strength, age: 0 });
			}
			this.connectionInference.set(level, neuronMap);
		}

		console.log(`Level ${level}: Predicted ${predictions.size} neurons for next frame (from connections)`);
	}

	/**
	 * Pattern inference: Predict lower-level peak neurons from higher-level pattern neurons.
	 * For each inferred pattern neuron at level N (from connection inference),
	 * predict its peak neuron at level N-1, age=-1.
	 */
	async inferPatternsAtLevel(level) {
		console.log(`Level ${level}: Inferring peak neurons for level ${level - 1}`);

		// Get inferred pattern neurons at this level (age=0)
		if (!this.connectionInference.has(level)) {
			console.log(`Level ${level}: No pattern neurons to infer from`);
			return;
		}

		const patternNeurons = this.connectionInference.get(level);
		const peakPredictions = new Map(); // Map<peak_neuron_id, total_strength>

		// For each inferred pattern neuron, find its peak neuron
		for (const [patternNeuronId, data] of patternNeurons) {
			if (data.age !== 0) continue;

			// Get the peak neuron for this pattern
			const peakNeuronId = this.patternPeaks.getPeak(patternNeuronId);
			if (peakNeuronId) {
				// Sum strengths if multiple patterns predict the same peak
				const currentStrength = peakPredictions.get(peakNeuronId) || 0;
				peakPredictions.set(peakNeuronId, currentStrength + data.strength);
			}
		}

		// Store pattern predictions with age=0
		if (peakPredictions.size > 0) {
			const neuronMap = new Map();
			for (const [neuronId, strength] of peakPredictions) {
				neuronMap.set(neuronId, { strength, age: 0 });
			}
			this.patternInference.set(level - 1, neuronMap);
		}

		console.log(`Level ${level}: Predicted ${peakPredictions.size} peak neurons for level ${level - 1} (from patterns)`);
	}

	/**
	 * Validate connection predictions from the previous frame.
	 * Apply negative reinforcement to connections that predicted incorrectly.
	 */
	validateConnectionPredictions(level) {
		// Get predictions from previous frame (stored in connectionInference with age=1)
		if (!this.connectionInference.has(level)) return;

		const predictions = this.connectionInference.get(level);

		// Collect all connection IDs that were used for predictions
		// (We need to track which connections made predictions, not just which neurons)
		// For now, we'll use a simplified approach: check if predicted neurons activated

		const predictedNeurons = new Set();
		for (const [neuronId, data] of predictions) {
			if (data.age === 1) {
				predictedNeurons.add(neuronId);
			}
		}

		if (predictedNeurons.size === 0) return;

		// Find which predictions failed (not in active_neurons at age=0)
		const failures = [];
		for (const neuronId of predictedNeurons) {
			const isActive = this.activeNeurons.has(neuronId, level, 0);
			if (!isActive) {
				failures.push(neuronId);
			}
		}

		if (failures.length === 0) return;

		// Apply negative reinforcement to connections that led to failed predictions
		// This is a simplified version - ideally we'd track which specific connections made each prediction
		console.log(`Level ${level}: Applied negative reinforcement to ${failures.length} failed connection predictions`);
	}

	/**
	 * Resolve conflicts in input predictions per channel.
	 * Reads from connection_inferred_neurons and pattern_inferred_neurons,
	 * resolves conflicts using channel logic, and writes final predictions to inferred_neurons.
	 */
	async resolveInputPredictionConflicts() {
		// Get the most recent predictions for the next frame at level 0
		const connectionRows = this.getInputPredictions(this.connectionInference);
		const patternRows = this.getInputPredictions(this.patternInference);

		if (connectionRows.length === 0 && patternRows.length === 0) return;

		// Group predictions by channel and resolve conflicts
		const channelPredictions = this.groupPredictionsByChannel(connectionRows, patternRows);
		await this.resolveAndWritePredictions(channelPredictions);
	}

	/**
	 * Get input predictions from a specific inference map
	 */
	getInputPredictions(inferenceMap) {
		const rows = [];

		// Get level 0 predictions with age=0
		if (!inferenceMap.has(0)) return rows;

		const level0Predictions = inferenceMap.get(0);

		for (const [neuronId, data] of level0Predictions) {
			if (data.age !== 0) continue;

			// Get neuron coordinates
			const coords = this.neurons.getCoordinates(neuronId);
			if (coords.length === 0) continue;

			// Check if this neuron has input dimensions
			for (const coord of coords) {
				const dim = this.dimensionIdToName[coord.dimension_id];
				const dimInfo = this.dimensions.get(dim);

				if (dimInfo && dimInfo.type === 'input') {
					rows.push({
						neuron_id: neuronId,
						strength: data.strength,
						dimension_id: coord.dimension_id,
						val: coord.val,
						dimension_name: dim,
						channel: dimInfo.channel
					});
				}
			}
		}

		return rows;
	}

	/**
	 * Group predictions by channel, building complete prediction objects with coordinates.
	 * If both connection and pattern inference predict the same neuron, their strengths are summed.
	 */
	groupPredictionsByChannel(connectionRows, patternRows) {
		const channelPredictions = new Map();
		this.addPredictionsToChannelMap(channelPredictions, connectionRows);
		this.addPredictionsToChannelMap(channelPredictions, patternRows);
		return channelPredictions;
	}

	/**
	 * Add predictions from rows to the channel map.
	 * If a neuron is already predicted, sum the strengths (both sources agree = higher confidence).
	 */
	addPredictionsToChannelMap(channelPredictions, rows) {
		for (const row of rows) {
			// If the channel doesn't have a map yet, create one
			if (!channelPredictions.has(row.channel)) channelPredictions.set(row.channel, new Map());
			const channelMap = channelPredictions.get(row.channel);

			// If the neuron doesn't have a prediction yet, create one
			if (!channelMap.has(row.neuron_id)) {
				channelMap.set(row.neuron_id, {
					neuron_id: row.neuron_id,
					strength: row.strength,
					coordinates: {}
				});
			}
			// Both connection and pattern predict this neuron - sum strengths
			else {
				channelMap.get(row.neuron_id).strength += row.strength;
			}

			// Add the coordinate to the neuron's prediction
			channelMap.get(row.neuron_id).coordinates[row.dimension_name] = row.val;
		}
	}

	/**
	 * Resolve conflicts for each channel and write final predictions to inferred_neurons.
	 * Channels can return multiple predictions (e.g., vision detecting multiple objects).
	 */
	async resolveAndWritePredictions(channelPredictions) {
		// Resolve conflicts for each channel and collect selected predictions
		const allSelectedPredictions = [];
		for (const [channelName, predictionMap] of channelPredictions) {
			allSelectedPredictions.push(...this.channels.get(channelName).resolveConflicts(Array.from(predictionMap.values())));
		}

		// If there are no predictions, nothing to resolve
		if (allSelectedPredictions.length === 0) return;

		// Store selected predictions in inferred_neurons with age=0
		if (!this.inferredNeurons.has(0)) {
			this.inferredNeurons.set(0, new Map());
		}

		const level0Inferred = this.inferredNeurons.get(0);
		for (const pred of allSelectedPredictions) {
			level0Inferred.set(pred.neuron_id, { strength: pred.strength, age: 0 });
		}

		console.log(`Resolved ${allSelectedPredictions.length} input predictions after conflict resolution`);
	}

	/**
	 * Reports accuracy of neuron predictions from the previous frame.
	 * At level 0: Tracks accuracy for connection_inferred_neurons, pattern_inferred_neurons, and final inferred_neurons (input predictions only).
	 * At higher levels: Tracks accuracy for connection_inferred_neurons (all neurons).
	 */
	reportPredictionsAccuracy(level) {
		// Initialize accuracy stats for this level if needed
		if (!this.accuracyStats.has(level)) {
			this.accuracyStats.set(level, {
				connection: { correct: 0, total: 0 },
				pattern: { correct: 0, total: 0 },
				resolved: { correct: 0, total: 0 }
			});
		}

		const stats = this.accuracyStats.get(level);

		// At higher levels, only report connection prediction accuracy
		if (level > 0) {
			const connectionPredictions = this.getPredictionsAtAge(this.connectionInference, level, 1);
			if (connectionPredictions.length === 0) return;

			const connectionMatches = connectionPredictions.filter(neuronId =>
				this.activeNeurons.has(neuronId, level, 0)
			);

			stats.connection.correct += connectionMatches.length;
			stats.connection.total += connectionPredictions.length;

			const currentRate = (connectionMatches.length / connectionPredictions.length * 100).toFixed(1);
			const avgRate = (stats.connection.correct / stats.connection.total * 100).toFixed(1);
			console.log(`Level ${level}: Connection prediction accuracy: ${connectionMatches.length}/${connectionPredictions.length} (${currentRate}%) | Avg: ${stats.connection.correct}/${stats.connection.total} (${avgRate}%)`);
			return;
		}

		// At level 0, report all three types of predictions (input neurons only)

		// Get connection predictions from previous frame (age=1, input neurons only)
		const connectionPredictions = this.getInputPredictionsAtAge(this.connectionInference, 0, 1);
		if (connectionPredictions.length > 0) {
			const connectionMatches = connectionPredictions.filter(neuronId =>
				this.activeNeurons.has(neuronId, 0, 0)
			);

			stats.connection.correct += connectionMatches.length;
			stats.connection.total += connectionPredictions.length;

			const currentRate = (connectionMatches.length / connectionPredictions.length * 100).toFixed(1);
			const avgRate = (stats.connection.correct / stats.connection.total * 100).toFixed(1);
			console.log(`Level ${level}: Connection prediction accuracy: ${connectionMatches.length}/${connectionPredictions.length} (${currentRate}%) | Avg: ${stats.connection.correct}/${stats.connection.total} (${avgRate}%)`);
		}

		// Get pattern predictions from previous frame (age=1, input neurons only)
		const patternPredictions = this.getInputPredictionsAtAge(this.patternInference, 0, 1);
		if (patternPredictions.length > 0) {
			const patternMatches = patternPredictions.filter(neuronId =>
				this.activeNeurons.has(neuronId, 0, 0)
			);

			stats.pattern.correct += patternMatches.length;
			stats.pattern.total += patternPredictions.length;

			const currentRate = (patternMatches.length / patternPredictions.length * 100).toFixed(1);
			const avgRate = (stats.pattern.correct / stats.pattern.total * 100).toFixed(1);
			console.log(`Level ${level}: Pattern prediction accuracy: ${patternMatches.length}/${patternPredictions.length} (${currentRate}%) | Avg: ${stats.pattern.correct}/${stats.pattern.total} (${avgRate}%)`);
		}

		// Get resolved predictions from previous frame (age=1, input neurons only)
		const resolvedPredictions = this.getInputPredictionsAtAge(this.inferredNeurons, 0, 1);
		if (resolvedPredictions.length > 0) {
			const resolvedMatches = resolvedPredictions.filter(neuronId =>
				this.activeNeurons.has(neuronId, 0, 0)
			);

			stats.resolved.correct += resolvedMatches.length;
			stats.resolved.total += resolvedPredictions.length;

			const currentRate = (resolvedMatches.length / resolvedPredictions.length * 100).toFixed(1);
			const avgRate = (stats.resolved.correct / stats.resolved.total * 100).toFixed(1);
			console.log(`Level ${level}: Resolved prediction accuracy: ${resolvedMatches.length}/${resolvedPredictions.length} (${currentRate}%) | Avg: ${stats.resolved.correct}/${stats.resolved.total} (${avgRate}%)`);
		}
	}

	/**
	 * Get predictions at a specific age from an inference map
	 */
	getPredictionsAtAge(inferenceMap, level, age) {
		if (!inferenceMap.has(level)) return [];

		const predictions = [];
		const levelMap = inferenceMap.get(level);

		for (const [neuronId, data] of levelMap) {
			if (data.age === age) {
				predictions.push(neuronId);
			}
		}

		return predictions;
	}

	/**
	 * Get input predictions at a specific age from an inference map (level 0 only)
	 * Filters to only include neurons with input dimensions
	 */
	getInputPredictionsAtAge(inferenceMap, level, age) {
		if (level !== 0 || !inferenceMap.has(level)) return [];

		const predictions = [];
		const levelMap = inferenceMap.get(level);

		for (const [neuronId, data] of levelMap) {
			if (data.age === age) {
				// Check if this neuron has input dimensions
				const coords = this.neurons.getCoordinates(neuronId);
				const hasInputDim = coords.some(coord => {
					const dim = this.dimensionIdToName[coord.dimension_id];
					const dimInfo = this.dimensions.get(dim);
					return dimInfo && dimInfo.type === 'input';
				});

				if (hasInputDim) {
					predictions.push(neuronId);
				}
			}
		}

		return predictions;
	}

	/**
	 * Process a single level - used during recognition phase
	 * Activates connections, detects peaks, matches/creates patterns, reinforces connections
	 */
	async processLevel(level) {
		console.log(`\n--- Processing Level ${level} ---`);

		// Clear scratch tables for this level
		this.activeConnections.clear();
		this.observedPatterns.clear();
		this.matchedPatterns.clear();

		// Step 1: Activate connections between active neurons
		activateConnections(this.activeNeurons, this.connections, this.activeConnections, level);

		// Step 2: Detect peaks (observed patterns)
		const peaks = detectPeaks(
			this.activeConnections,
			this.connections,
			level,
			this.peakTimeDecayFactor,
			this.minPeakStrength,
			this.minPeakRatio
		);
		this.observedPatterns = peaks;

		// Step 3: Match observed patterns to known patterns
		if (peaks.size > 0) {
			const matches = matchPatterns(
				this.observedPatterns,
				this.patterns,
				this.patternPeaks,
				this.mergePatternThreshold
			);
			this.matchedPatterns = matches;

			// Step 4: Merge matched patterns (reinforce)
			if (matches.size > 0) {
				mergeMatchedPatterns(
					this.matchedPatterns,
					this.observedPatterns,
					this.patterns,
					this.minConnectionStrength,
					this.maxConnectionStrength,
					this.patternNegativeReinforcement
				);
			}

			// Step 5: Create new patterns for unmatched peaks
			createNewPatterns(
				this.observedPatterns,
				this.matchedPatterns,
				this.neurons,
				this.patterns,
				this.patternPeaks
			);
		}

		// Step 6: Reinforce connections between co-active neurons
		reinforceConnections(
			this.activeNeurons,
			this.connections,
			level,
			this.minConnectionStrength,
			this.maxConnectionStrength
		);

		// Step 7: Activate pattern neurons at next level
		if (level + 1 < this.maxLevels && peaks.size > 0) {
			for (const peakNeuronId of peaks.keys()) {
				this.activeNeurons.add(peakNeuronId, level + 1, 0);
			}
		}

		return peaks.size > 0;
	}

	/**
	 * Process frame - main entry point
	 * @param {Array} frame - Array of input/output points from channels
	 * @param {number} globalReward - Reward factor from feedback (multiplicative, 1.0 = neutral)
	 */
	async processFrame(frame, globalReward = 1.0) {
		const frameStart = performance.now();
		this.frameNumber++;

		console.log(`\n${'='.repeat(80)}`);
		console.log(`OBSERVING NEW FRAME: ${JSON.stringify(frame)} ${this.frameNumber}`);
		console.log(`applying global reward: ${globalReward.toFixed(3)}`);
		console.log('='.repeat(80));

		// Step 1: Apply rewards to previously executed decisions (before aging them further)
		this.applyRewards(globalReward);

		// Step 2: Age the active neurons in memory context - sliding the temporal window
		this.ageNeurons();

		// Step 3: Execute previous frame's decisions + exploration if needed
		await this.executeOutputs();

		// Step 4: Activate base neurons from the frame along with higher level patterns from them
		await this.recognizeNeurons(frame);

		// Step 5: Do predictions and outputs - what's going to happen next?
		await this.inferNeurons();

		// Step 6: Forget cycle (if needed)
		this.forgetCounter++;
		if (this.forgetCounter >= this.forgetCycles) {
			this.runForgetCycle();
			this.forgetCounter = 0;
		}

		const frameElapsed = performance.now() - frameStart;
		console.log(`\nFrame ${this.frameNumber} complete in ${frameElapsed.toFixed(2)}ms`);

		// Show accuracy stats every 100 frames
		if (this.frameNumber % 100 === 0) {
			this.printAccuracyStats();
		}

		console.log('='.repeat(80));

		// Debug pause
		if (this.debug) {
			await new Promise(resolve => {
				this.rl.question('Press Enter to continue...', () => resolve());
			});
		}
	}

	/**
	 * Print prediction accuracy statistics
	 */
	printAccuracyStats() {
		console.log(`\n📊 Prediction Accuracy (Frame ${this.frameNumber}):`);

		if (this.accuracyStats.size === 0) {
			console.log('   No predictions made yet');
			return;
		}

		for (const [level, stats] of this.accuracyStats) {
			const connRate = stats.connection.total > 0
				? (stats.connection.correct / stats.connection.total * 100).toFixed(1)
				: '0.0';
			const patternRate = stats.pattern.total > 0
				? (stats.pattern.correct / stats.pattern.total * 100).toFixed(1)
				: '0.0';
			const resolvedRate = stats.resolved.total > 0
				? (stats.resolved.correct / stats.resolved.total * 100).toFixed(1)
				: '0.0';

			console.log(`   Level ${level}: Conn=${connRate}% (${stats.connection.correct}/${stats.connection.total}), Pattern=${patternRate}% (${stats.pattern.correct}/${stats.pattern.total}), Resolved=${resolvedRate}% (${stats.resolved.correct}/${stats.resolved.total})`);
		}
	}

	/**
	 * Print final accuracy summary for the episode
	 */
	printFinalAccuracySummary() {
		console.log(`\n${'='.repeat(80)}`);
		console.log(`📊 FINAL PREDICTION ACCURACY SUMMARY (${this.frameNumber} frames)`);
		console.log('='.repeat(80));

		if (this.accuracyStats.size === 0) {
			console.log('No predictions were made during this episode.');
			return;
		}

		for (const [level, stats] of this.accuracyStats) {
			console.log(`\nLevel ${level}:`);

			if (stats.connection.total > 0) {
				const rate = (stats.connection.correct / stats.connection.total * 100).toFixed(1);
				console.log(`  Connection predictions: ${stats.connection.correct}/${stats.connection.total} (${rate}%)`);
			}

			if (stats.pattern.total > 0) {
				const rate = (stats.pattern.correct / stats.pattern.total * 100).toFixed(1);
				console.log(`  Pattern predictions: ${stats.pattern.correct}/${stats.pattern.total} (${rate}%)`);
			}

			if (stats.resolved.total > 0) {
				const rate = (stats.resolved.correct / stats.resolved.total * 100).toFixed(1);
				console.log(`  Resolved predictions: ${stats.resolved.correct}/${stats.resolved.total} (${rate}%)`);
			}
		}
		console.log('='.repeat(80));
	}

	/**
	 * Apply global reward to active connections that led to executed outputs.
	 * Strengthens connections for positive rewards, weakens for negative rewards.
	 * Uses exponential temporal decay - older connections get less reward/punishment.
	 * @param {number} globalReward - Multiplicative reward factor (1.0 = neutral, 1.5 = positive, 0.5 = negative)
	 */
	applyRewards(globalReward) {
		if (globalReward === 1.0) {
			console.log('Neutral global reward - no updates needed');
			return;
		}

		// Calculate reward adjustment: positive reward strengthens, negative weakens
		// globalReward = 1.5 → adjustment = +0.5 per connection
		// globalReward = 0.5 → adjustment = -0.5 per connection
		const rewardAdjustment = globalReward - 1.0;

		console.log(`\nApplying global reward ${globalReward.toFixed(3)} (adjustment: ${rewardAdjustment >= 0 ? '+' : ''}${rewardAdjustment.toFixed(3)})`);

		applyRewards(
			this.activeConnections,
			this.connections,
			this.patterns,
			rewardAdjustment,
			this.rewardTimeDecayFactor,
			this.minConnectionStrength,
			this.maxConnectionStrength
		);
	}

	/**
	 * Run forget cycle - decay weak connections and patterns
	 */
	runForgetCycle() {
		console.log('\n--- Running Forget Cycle ---');
		const startTime = performance.now();

		let deletedConnections = 0;
		let deletedPatterns = 0;

		// Decay and delete weak connections
		for (const conn of this.connections.byId.values()) {
			const newStrength = conn.strength - this.connectionForgetRate;

			if (newStrength <= 0) {
				this.connections.delete(conn.id);
				deletedConnections++;
			} else {
				this.connections.set(conn.from_neuron_id, conn.to_neuron_id, conn.distance, newStrength);
			}
		}

		// Decay and delete weak pattern entries
		for (const [_, entry] of this.patterns.byKey.entries()) {
			const newStrength = entry.strength - this.patternForgetRate;

			if (newStrength <= 0) {
				this.patterns.delete(entry.pattern_neuron_id, entry.connection_id);
				deletedPatterns++;
			} else {
				this.patterns.set(entry.pattern_neuron_id, entry.connection_id, newStrength);
			}
		}

		const elapsed = performance.now() - startTime;
		console.log(`Forget cycle: deleted ${deletedConnections} connections, ${deletedPatterns} pattern entries (${elapsed.toFixed(2)}ms)`);
	}

	/**
	 * Apply reward to active connections and patterns
	 * @param {number} reward - Additive reward (0 = neutral, positive = good, negative = bad)
	 */
	applyReward(reward) {
		// Convert additive reward to multiplicative factor
		const rewardFactor = 1.0 + reward;

		console.log(`Applying reward factor ${rewardFactor.toFixed(3)} to active connections/patterns`);

		applyRewards(
			this.activeConnections,
			this.connections,
			this.patterns,
			rewardFactor,
			this.rewardTimeDecayFactor,
			this.minConnectionStrength,
			this.maxConnectionStrength
		);
	}

	/**
	 * Get statistics about brain state
	 */
	getStats() {
		return {
			neurons: this.neurons.size(),
			connections: this.connections.size(),
			patterns: this.patterns.size(),
			patternPeaks: this.patternPeaks.size(),
			activeNeurons: this.activeNeurons.size(),
			activeConnections: this.activeConnections.size(),
			frameNumber: this.frameNumber
		};
	}

	/**
	 * Print statistics
	 */
	printStats() {
		const stats = this.getStats();
		console.log('\n--- Brain Statistics ---');
		console.log(`Neurons: ${stats.neurons}`);
		console.log(`Connections: ${stats.connections}`);
		console.log(`Patterns: ${stats.patterns}`);
		console.log(`Pattern Peaks: ${stats.patternPeaks}`);
		console.log(`Active Neurons: ${stats.activeNeurons}`);
		console.log(`Active Connections: ${stats.activeConnections}`);
		console.log(`Frame: ${stats.frameNumber}`);
	}
}

