import Brain from './brain.js';
import {
	ConnectionStore,
	ActiveNeuronStore,
	PatternStore,
	PatternPeakStore,
	NeuronStore,
	ActiveConnectionStore
} from './brain-memory-stores.js';
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
export default class BrainMemory extends Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {
		super();

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
	 * Reset brain memory state for a clean episode start
	 */
	async resetContext() {
		console.log(`Resetting context... (Learned: ${this.neurons.size()} neurons, ${this.connections.size()} connections, ${this.patterns.size()} pattern entries)`);
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
	 * Close database connection
	 */
	async close() {
		if (this.conn) await this.conn.end();
		this.rl.close();
	}

	/**
	 * Get or create dimension ID
	 * Note: Dimensions must be pre-initialized with channel info
	 */
	async getDimensionId(name) {
		const [rows] = await this.conn.query('SELECT id FROM dimensions WHERE name = ?', [name]);
		if (rows.length > 0) return rows[0].id;

		// Dimension not found - this shouldn't happen if channels are properly initialized
		throw new Error(`Dimension '${name}' not found. Did you forget to initialize channels?`);
	}


	/**
	 * Match frame neurons to existing neurons or create new ones
	 * Uses fast O(1) coordinate set lookup instead of iterative dimension matching
	 */
	async getFrameNeurons(points) {
		const neuronIds = [];

		for (const point of points) {

			// Build coordinate map using dimension IDs
			const coordinateMap = new Map();
			for (const [dimName, value] of Object.entries(point)) {
				const dimId = await this.getDimensionId(dimName);
				coordinateMap.set(dimId, value);
			}

			// Fast O(1) lookup by complete coordinate set using the key
			const key = this.neurons.getCoordinateKey(coordinateMap);
			let neuronId = this.neurons.byCoordinateSet.get(key) || null;

			// Create new neuron if not found with coordinates
			if (neuronId === null) {
				neuronId = this.neurons.createNeuron();
				for (const [dimId, value] of coordinateMap) this.neurons.setCoordinate(neuronId, dimId, value);
			}

			neuronIds.push(neuronId);
		}

		if (this.debug) console.log(`Matched/created ${neuronIds.length} neurons from ${points.length} points`);
		return neuronIds;
	}

	/**
	 * Activate neurons at base level (level 0)
	 */
	async activateNeurons(neuronIds) {
		for (const neuronId of neuronIds) {
			this.activeNeurons.add(neuronId, 0, 0); // level 0, age 0
		}
	}

	/**
	 * Ages all neurons and connections in the context by 1, then deactivates aged-out items.
	 * With uniform aging, all levels are deactivated at once when age >= baseNeuronMaxAge.
	 */
	ageNeurons() {
		// Age all active neurons (collect first to avoid iterator issues)
		const allActiveNeurons = Array.from(this.activeNeurons.byKey.values());

		for (const an of allActiveNeurons) {
			const newAge = an.age + 1;

			// Remove from old position
			this.activeNeurons.delete(an.neuron_id, an.level, an.age);

			// If not aged out, add back with new age
			if (newAge < this.baseNeuronMaxAge) this.activeNeurons.add(an.neuron_id, an.level, newAge);
		}

		// Age all active connections (rebuild the store)
		const allActiveConns = Array.from(this.activeConnections.byKey.values());
		this.activeConnections.clear();

		for (const ac of allActiveConns) {
			const newAge = ac.age + 1;

			// If not aged out, add back with new age
			if (newAge < this.baseNeuronMaxAge) this.activeConnections.add(ac.connection_id, ac.from_neuron_id, ac.to_neuron_id, ac.level, newAge);
		}

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
		let aged = 0;

		for (const [level, neuronMap] of inferenceMap) {
			// Age each neuron's metadata if it exists
			for (const [neuronId, data] of neuronMap) {
				if (typeof data === 'object' && data.age !== undefined) {
					data.age++;
					aged++;

					// Clean up if age >= 2
					if (data.age >= 2) {
						neuronMap.delete(neuronId);
						cleaned++;
					}
				}
			}

			// Remove empty level maps
			if (neuronMap.size === 0) inferenceMap.delete(level);
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
		if (!this.inferredNeurons.has(0)) return;

		const level0Inferred = this.inferredNeurons.get(0);
		const outputRows = [];

		// Find neurons with age=1 and output dimensions
		for (const [neuronId, data] of level0Inferred) {
			if (data.age === 1) {
				// Get neuron coordinates
				const coords = this.neurons.getCoordinates(neuronId);
				if (coords.length === 0) continue;

				// Check if this neuron has action dimensions
				for (const coord of coords) {
					const dim = this.dimensionIdToName[coord.dimension_id];
					const dimInfo = this.dimensions.get(dim);

					if (dimInfo && dimInfo.type === 'action') {
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
			if (this.debug) console.log('No previous outputs to execute');
		}
	}

	/**
	 * Execute curiosity exploration if brain is inactive
	 */
	async curiosityExploration() {
		// Check if the brain is inactive - if active, no exploration needed
		if ((this.frameNumber - this.lastActivity) < this.inactivityThreshold) return;
		if (this.debug) console.log('Brain inactive - executing curiosity exploration');

		// Get a random channel for exploration
		const channelNames = Array.from(this.channels.keys());
		const randomChannelName = channelNames[Math.floor(Math.random() * channelNames.length)];
		const randomChannel = this.channels.get(randomChannelName);

		// Get exploration actions for the channel
		const explorationAction = randomChannel.getExplorationAction();
		if (explorationAction === null) {
			if (this.debug) console.log(`No exploration actions for ${randomChannelName}`);
		}
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
	}

	/**
	 * Infer predictions and outputs using bulk processing for all levels.
	 * Connection inference handles validation, aging, and deletion internally for all levels.
	 * Pattern inference cascades predictions down all levels recursively.
	 */
	async inferNeurons() {
		// Report accuracy for all levels (from previous frame)
		this.reportPredictionsAccuracy();

		// Connection inference for all levels (handles validation, aging, deletion)
		await this.inferConnections();

		// Pattern inference (recursive cascade down all levels)
		await this.inferPatterns();

		// Merge predictions for higher levels (level > 0)
		this.mergeHigherLevelPredictions();

		// Resolve conflicts in input predictions at base level (level 0)
		await this.resolveInputPredictionConflicts();
	}

	/**
	 * Get the highest level that has active neurons
	 */
	getMaxActiveLevel() {
		let maxLevel = 0;

		for (const an of this.activeNeurons.byKey.values()) {
			if (an.level > maxLevel) maxLevel = an.level;
		}

		return maxLevel;
	}

	/**
	 * Connection inference for all levels at once.
	 * Validates previous predictions, applies negative reinforcement, and makes new predictions.
	 */
	async inferConnections() {
		// Validate previous frame's predictions and apply negative reinforcement
		this.negativeReinforceConnections();

		// Clear previous connection predictions for all levels
		this.connectionInference.clear();

		// Get the highest level that is currently active
		const maxLevel = this.getMaxActiveLevel();

		// Make new predictions for all levels
		for (let level = maxLevel; level >= 0; level--) {
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
				if (this.debug) console.log(`Level ${level}: Predicted ${predictions.size} neurons for next frame (from connections)`);
			}
		}
	}

	/**
	 * Pattern inference for all levels - cascades predictions down from higher levels.
	 * For each inferred pattern neuron at level N, predict its peak neuron at level N-1.
	 */
	async inferPatterns() {

		// Clear previous pattern predictions
		this.patternInference.clear();

		// Get the highest level that is currently active
		const maxLevel = this.getMaxActiveLevel();

		// Process levels from high to low
		for (let level = maxLevel; level > 0; level--) {
			const targetLevel = level - 1;

			// Get inferred neurons at this level (age=0)
			if (!this.connectionInference.has(level)) continue;

			const inferredNeurons = this.connectionInference.get(level);
			const peakPredictions = new Map(); // Map<peak_neuron_id, total_strength>

			// For each inferred neuron, check if it's a pattern neuron and find its peak
			const debugPatternNeurons = [];
			for (const [neuronId, data] of inferredNeurons) {
				if (data.age !== 0) continue;

				// Check if this neuron is a pattern neuron (has a peak mapping)
				const peakNeuronId = this.patternPeaks.getPeak(neuronId);
				if (peakNeuronId) {
					// Sum strengths if multiple patterns predict the same peak
					const currentStrength = peakPredictions.get(peakNeuronId) || 0;
					peakPredictions.set(peakNeuronId, currentStrength + data.strength);

					if (this.debug && level === 1) debugPatternNeurons.push(`L${level}N${neuronId}→L${targetLevel}N${peakNeuronId}(${data.strength.toFixed(1)})`);
				}
			}

			if (this.debug && level === 1 && debugPatternNeurons.length > 0) {
				console.log(`   🔗 Level ${level} pattern neurons predicting level ${targetLevel}: ${debugPatternNeurons.join(', ')}`);
			}

			// Add pattern predictions with age=0
			if (peakPredictions.size > 0) {
				// Get or create the level map
				let neuronMap = this.patternInference.get(targetLevel);
				if (!neuronMap) {
					neuronMap = new Map();
					this.patternInference.set(targetLevel, neuronMap);
				}

				// Add new predictions with age=0
				for (const [neuronId, strength] of peakPredictions) {
					neuronMap.set(neuronId, { strength, age: 0 });
				}

				if (this.debug) console.log(`Level ${level}: Predicted ${peakPredictions.size} peak neurons for level ${targetLevel} (from patterns)`);
			}
		}
	}

	/**
	 * Merge connection and pattern predictions for higher levels (level > 0).
	 * Higher levels don't use channel-based conflict resolution - just combine predictions by summing strengths.
	 * This allows the brain to learn connections between output neurons and high-level decision making.
	 */
	mergeHigherLevelPredictions() {
		// Get the highest level that has predictions
		const maxLevel = this.getMaxActiveLevel();

		// Combine connection and pattern predictions for all levels > 0
		for (let level = maxLevel; level > 0; level--) {
			const mergedMap = new Map();

			// Add connection predictions
			if (this.connectionInference.has(level)) {
				const connMap = this.connectionInference.get(level);
				for (const [neuronId, data] of connMap) {
					if (data.age === 0) {
						mergedMap.set(neuronId, { strength: data.strength, age: 0 });
					}
				}
			}

			// Add pattern predictions (sum strengths if both sources predict same neuron)
			if (this.patternInference.has(level)) {
				const patternMap = this.patternInference.get(level);
				for (const [neuronId, data] of patternMap) {
					if (data.age === 0) {
						const existing = mergedMap.get(neuronId);
						if (existing) {
							existing.strength += data.strength;
						} else {
							mergedMap.set(neuronId, { strength: data.strength, age: 0 });
						}
					}
				}
			}

			// Store merged predictions in inferred_neurons
			if (mergedMap.size > 0) {
				this.inferredNeurons.set(level, mergedMap);
				if (this.debug) console.log(`Level ${level}: Merged ${mergedMap.size} predictions to inferred_neurons (connection + pattern)`);
			}
		}
	}

	/**
	 * Apply negative reinforcement to connections that predicted incorrectly.
	 * Validates connection predictions from the previous frame for ALL levels.
	 */
	negativeReinforceConnections() {
		let totalFailures = 0;

		// Process all levels
		for (const [level, predictions] of this.connectionInference) {
			// Collect all predicted neurons with age=1
			const predictedNeurons = new Set();
			for (const [neuronId, data] of predictions) {
				if (data.age === 1) predictedNeurons.add(neuronId);
			}

			if (predictedNeurons.size === 0) continue;

			// Find which predictions failed (not in active_neurons at age=0)
			const failures = [];
			for (const neuronId of predictedNeurons) {
				const isActive = this.activeNeurons.has(neuronId, level, 0);
				if (!isActive) failures.push(neuronId);
			}

			totalFailures += failures.length;
		}

		if (totalFailures > 0 && this.debug) {
			console.log(`Applied negative reinforcement to ${totalFailures} failed connection predictions across all levels`);
		}
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

		// Debug: Show pattern prediction strengths
		if (this.debug && patternRows.length > 0) {
			console.log(`   📊 Pattern predictions (L0): ${patternRows.map(r => `N${r.neuron_id}(${r.strength.toFixed(1)})`).join(', ')}`);
		}

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
	 * Check if a channel needs exploration (in-memory implementation)
	 * Returns true if channel has no inferred outputs OR if holding too long
	 * @param {string} channelName - name of the channel to check
	 * @returns {Promise<boolean>} - true if channel needs exploration
	 */
	async channelNeedsExploration(channelName) {
		const channel = this.channels.get(channelName);
		if (!channel) return false;

		// Check if channel is holding too long (if it has holdingFrames property)
		if (channel.holdingFrames !== undefined && channel.maxHoldingFrames !== undefined) {
			if (channel.holdingFrames > channel.maxHoldingFrames) return true;
		}

		const outputDimNames = channel.getOutputDimensions();
		if (outputDimNames.length === 0) return false;

		// Get dimension IDs for this channel's output dimensions
		const outputDimIds = new Set(outputDimNames.map(name => this.dimensionNameToId[name]).filter(id => id !== undefined));
		if (outputDimIds.size === 0) return false;

		// Check if any inferred neurons at level 0, age 0 have coordinates in these output dimensions
		const level0Inferred = this.inferredNeurons.get(0);
		if (!level0Inferred) return true; // No inferred neurons = needs exploration

		for (const [neuronId, data] of level0Inferred) {
			if (data.age !== 0) continue;

			// Get neuron coordinates
			const coords = this.neurons.getCoordinates(neuronId);
			if (coords.length === 0) continue;

			// Check if any coordinate is in the output dimensions
			for (const coord of coords) {
				if (outputDimIds.has(coord.dimension_id)) return false; // Has outputs = doesn't need exploration
			}
		}

		return true; // No outputs found = needs exploration
	}

	/**
	 * save resolved predictions to in-memory storage (implementation of abstract method)
	 */
	async saveResolvedPredictions(allSelectedPredictions) {
		// Store selected predictions in inferred_neurons with age=0
		if (!this.inferredNeurons.has(0)) this.inferredNeurons.set(0, new Map());
		const level0Inferred = this.inferredNeurons.get(0);
		for (const pred of allSelectedPredictions) level0Inferred.set(pred.neuron_id, { strength: pred.strength, age: 0 });
	}

	/**
	 * Reports accuracy of neuron predictions from the previous frame for ALL levels.
	 * Tracks accuracy for connection_inferred_neurons, pattern_inferred_neurons, and final inferred_neurons.
	 */
	reportPredictionsAccuracy() {
		// Group predictions by level
		const levelStats = new Map();

		// Debug: Track base level predictions vs activations
		let debugBasePredictions = [];
		let debugBaseActivations = [];

		// Process connection predictions
		for (const [level, predictions] of this.connectionInference) {
			if (!levelStats.has(level)) {
				levelStats.set(level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			}
			const stats = levelStats.get(level);

			for (const [neuronId, data] of predictions) {
				if (data.age === 1) {
					stats.connection.total++;
					const isCorrect = this.activeNeurons.has(neuronId, level, 0);
					if (isCorrect) stats.connection.correct++;

					// Debug base level - collect ALL predictions when debug is on
					if (level === 0 && this.debug) {
						debugBasePredictions.push({ neuronId, strength: data.strength, correct: isCorrect });
					}
				}
			}
		}

		// Debug: Collect base level activations and show detailed comparison
		if (this.debug) {
			// Get all active neurons at level 0, age 0
			const level0Map = this.activeNeurons.byLevelAge.get(0);
			if (level0Map) {
				const age0Set = level0Map.get(0);
				if (age0Set) debugBaseActivations = Array.from(age0Set);
			}

			if (debugBasePredictions.length > 0 || debugBaseActivations.length > 0) {
				console.log(`\n🔍 Base Level Debug (Frame ${this.frameNumber}):`);
				console.log(`   Predictions (age=1): ${debugBasePredictions.length} total`);
				console.log(`   Activations (age=0): ${debugBaseActivations.length} total`);
				console.log(`   Correct predictions: ${debugBasePredictions.filter(p => p.correct).length}`);

				// Show ALL predictions with their correctness
				if (debugBasePredictions.length > 0) {
					console.log(`   All predictions:`, debugBasePredictions.map(p => `${p.neuronId}(${p.strength.toFixed(1)})${p.correct ? '✓' : '✗'}`).join(', '));
				}

				// Show ALL activations
				if (debugBaseActivations.length > 0) {
					console.log(`   All activations:`, debugBaseActivations.join(', '));
				}

				// Show which activations were NOT predicted
				const predictedIds = new Set(debugBasePredictions.map(p => p.neuronId));
				const unpredicted = debugBaseActivations.filter(id => !predictedIds.has(id));
				if (unpredicted.length > 0) {
					console.log(`   ⚠️  Unpredicted activations: ${unpredicted.join(', ')}`);

					// Show coordinates for unpredicted neurons
					for (const neuronId of unpredicted) {
						const coords = this.neurons.getCoordinates(neuronId);
						const coordStr = coords.map(c => {
							const dimName = this.dimensions.get(c.dimension_id)?.name || `dim${c.dimension_id}`;
							return `${dimName}=${c.val}`;
						}).join(', ');
						console.log(`      Neuron ${neuronId}: ${coordStr}`);
					}

					// Check if there are any connections TO these neurons
					for (const neuronId of unpredicted) {
						const incomingConns = this.connections.findByTo(neuronId);
						console.log(`      Neuron ${neuronId}: ${incomingConns.length} incoming connections`);
						if (incomingConns.length > 0) {
							const sample = incomingConns.slice(0, 3).map(c =>
								`from=${c.from_neuron_id} dist=${c.distance} str=${c.strength.toFixed(1)}`
							).join(', ');
							console.log(`         Sample: ${sample}`);

							// Check if source neurons are active at the right age
							const distanceGroups = new Map();
							for (const conn of incomingConns) {
								if (!distanceGroups.has(conn.distance)) distanceGroups.set(conn.distance, []);
								distanceGroups.get(conn.distance).push(conn);
							}

							for (const [distance, conns] of distanceGroups) {
								const requiredAge = distance - 1;
								const activeSourceCount = conns.filter(c =>
									this.activeNeurons.has(c.from_neuron_id, 0, requiredAge)
								).length;
								console.log(`         Distance ${distance}: ${activeSourceCount}/${conns.length} source neurons active at age ${requiredAge}`);
							}
						}
					}
				}
			}
		}

		// Process pattern predictions
		for (const [level, predictions] of this.patternInference) {
			if (!levelStats.has(level)) {
				levelStats.set(level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			}
			const stats = levelStats.get(level);

			for (const [neuronId, data] of predictions) {
				if (data.age === 1) {
					stats.pattern.total++;
					if (this.activeNeurons.has(neuronId, level, 0)) stats.pattern.correct++;
				}
			}
		}

		// Process resolved predictions (all levels)
		for (const [level, predictions] of this.inferredNeurons) {
			if (!levelStats.has(level)) {
				levelStats.set(level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			}
			const stats = levelStats.get(level);

			for (const [neuronId, data] of predictions) {
				if (data.age === 1) {
					stats.resolved.total++;
					const isCorrect = this.activeNeurons.has(neuronId, level, 0);
					if (isCorrect) stats.resolved.correct++;

					// Debug: Show resolved predictions that were wrong
					if (!isCorrect && level === 0 && this.debug) {
						const coords = this.neurons.getCoordinates(neuronId);
						const coordStr = coords.map(c => {
							const dimName = this.dimensions.get(c.dimension_id)?.name || `dim${c.dimension_id}`;
							return `${dimName}=${c.val}`;
						}).join(', ');
						console.log(`   ❌ Resolved prediction WRONG: neuron ${neuronId} [${coordStr}] (strength ${data.strength.toFixed(1)}) was predicted but did NOT activate`);

						// Show which connections contributed to this wrong prediction
						const activeNeurons = this.activeNeurons.getByLevel(level);
						const contributingConns = [];
						for (const an of activeNeurons) {
							const distance = an.age + 1;
							const conn = this.connections.findByFromToDistance(an.neuron_id, neuronId, distance);
							if (conn && conn.strength > 0) {
								const anCoords = this.neurons.getCoordinates(an.neuron_id);
								const anCoordStr = anCoords.map(c => {
									const dimName = this.dimensions.get(c.dimension_id)?.name || `dim${c.dimension_id}`;
									return `${dimName}=${c.val}`;
								}).join(', ');
								const weightedStr = conn.strength * Math.pow(this.peakTimeDecayFactor, conn.distance);
								contributingConns.push(`N${an.neuron_id}[${anCoordStr}] age=${an.age} dist=${distance} str=${conn.strength.toFixed(1)} weighted=${weightedStr.toFixed(1)}`);
							}
						}
						if (contributingConns.length > 0) console.log(`      Contributing connections: ${contributingConns.slice(0, 5).join(', ')}`);
					}
				}
			}
		}

		// Report and update cumulative stats for each level
		const levels = Array.from(levelStats.keys()).sort((a, b) => b - a);
		for (const level of levels) {
			const stats = levelStats.get(level);

			// Initialize cumulative stats for this level if needed
			if (!this.accuracyStats.has(level)) {
				this.accuracyStats.set(level, { connection: { correct: 0, total: 0 }, pattern: { correct: 0, total: 0 }, resolved: { correct: 0, total: 0 } });
			}
			const cumulative = this.accuracyStats.get(level);

			// Report connection accuracy
			if (stats.connection.total > 0) {
				cumulative.connection.correct += stats.connection.correct;
				cumulative.connection.total += stats.connection.total;
				const currentRate = (stats.connection.correct / stats.connection.total * 100).toFixed(1);
				const avgRate = (cumulative.connection.correct / cumulative.connection.total * 100).toFixed(1);
				if (this.debug) {
					console.log(`Level ${level}: Connection prediction accuracy: ${stats.connection.correct}/${stats.connection.total} (${currentRate}%) | Avg: ${cumulative.connection.correct}/${cumulative.connection.total} (${avgRate}%)`);
				}
			}

			// Report pattern accuracy
			if (stats.pattern.total > 0) {
				cumulative.pattern.correct += stats.pattern.correct;
				cumulative.pattern.total += stats.pattern.total;
				const currentRate = (stats.pattern.correct / stats.pattern.total * 100).toFixed(1);
				const avgRate = (cumulative.pattern.correct / cumulative.pattern.total * 100).toFixed(1);
				if (this.debug) {
					console.log(`Level ${level}: Pattern prediction accuracy: ${stats.pattern.correct}/${stats.pattern.total} (${currentRate}%) | Avg: ${cumulative.pattern.correct}/${cumulative.pattern.total} (${avgRate}%)`);
				}
			}

			// Report resolved accuracy
			if (stats.resolved.total > 0) {
				cumulative.resolved.correct += stats.resolved.correct;
				cumulative.resolved.total += stats.resolved.total;
				const currentRate = (stats.resolved.correct / stats.resolved.total * 100).toFixed(1);
				const avgRate = (cumulative.resolved.correct / cumulative.resolved.total * 100).toFixed(1);
				if (this.debug) {
					console.log(`Level ${level}: Resolved prediction accuracy: ${stats.resolved.correct}/${stats.resolved.total} (${currentRate}%) | Avg: ${cumulative.resolved.correct}/${cumulative.resolved.total} (${avgRate}%)`);
				}
			}
		}
	}



	/**
	 * Process a single level - used during recognition phase
	 * Activates connections, detects peaks, matches/creates patterns, reinforces connections
	 */
	async processLevel(level) {
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

		if (this.debugPatterns) {
			console.log(`   🔍 Level ${level} peak detection: ${peaks.size} peaks found`);
			for (const [peakId, connIds] of peaks) {
				console.log(`      Peak N${peakId}: ${connIds.size} connections`);
			}
		}

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
		// Activate all pattern neurons from matched patterns (both matched and newly created)
		if (level + 1 < this.maxLevels && this.matchedPatterns.size > 0) {
			const patternNeuronIds = new Set();
			for (const patternSet of this.matchedPatterns.values()) {
				for (const patternNeuronId of patternSet) {
					patternNeuronIds.add(patternNeuronId);
				}
			}
			for (const patternNeuronId of patternNeuronIds) {
				this.activeNeurons.add(patternNeuronId, level + 1, 0);
			}
			if (this.debugPatterns) {
				console.log(`   → Level ${level} activated ${patternNeuronIds.size} pattern neurons at level ${level + 1}`);
			}
		}

		return peaks.size > 0;
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
			if (this.debug) console.log('Neutral global reward - no updates needed');
			return;
		}

		// Calculate reward adjustment: positive reward strengthens, negative weakens
		// globalReward = 1.5 → adjustment = +0.5 per connection
		// globalReward = 0.5 → adjustment = -0.5 per connection
		const rewardAdjustment = globalReward - 1.0;

		if (this.debug) console.log(`\nApplying global reward ${globalReward.toFixed(3)} (adjustment: ${rewardAdjustment >= 0 ? '+' : ''}${rewardAdjustment.toFixed(3)})`);

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
	 * Reduces strengths and deletes unused neurons (clamped between minConnectionStrength and maxConnectionStrength)
	 * Only runs every forgetCycles frames
	 */
	async runForgetCycle() {
		// Check if it's time to run the forget cycle
		this.forgetCounter = (this.forgetCounter || 0) + 1;
		if (this.forgetCounter < this.forgetCycles) return;

		// Reset counter
		this.forgetCounter = 0;

		if (this.debug) console.log('\n--- Running Forget Cycle ---');
		const startTime = performance.now();

		let deletedConnections = 0;
		let deletedPatterns = 0;
		let deletedPatternPeaks = 0;
		let deletedNeurons = 0;

		// 1. PATTERN FORGETTING: Reduce pattern strengths and remove dead patterns (clamped)
		for (const entry of this.patterns.byKey.values()) {
			const newStrength = Math.max(this.minConnectionStrength, Math.min(this.maxConnectionStrength, entry.strength - this.patternForgetRate));

			if (newStrength <= this.minConnectionStrength) {
				this.patterns.delete(entry.pattern_neuron_id, entry.connection_id);
				deletedPatterns++;
			} else {
				this.patterns.set(entry.pattern_neuron_id, entry.connection_id, newStrength);
			}
		}

		// 2. CONNECTION FORGETTING: Reduce connection strengths and remove dead connections (clamped)
		for (const conn of this.connections.byId.values()) {
			const newStrength = Math.max(this.minConnectionStrength, Math.min(this.maxConnectionStrength, conn.strength - this.connectionForgetRate));

			if (newStrength <= this.minConnectionStrength) {
				this.connections.delete(conn.id);
				deletedConnections++;
			} else {
				this.connections.set(conn.from_neuron_id, conn.to_neuron_id, conn.distance, newStrength);
			}
		}

		// 3. PATTERN_PEAKS CLEANUP: Remove pattern peaks for patterns with no connections
		const patternNeuronsWithConnections = new Set();
		for (const entry of this.patterns.byKey.values()) {
			patternNeuronsWithConnections.add(entry.pattern_neuron_id);
		}

		for (const mapping of this.patternPeaks.getAll()) {
			if (!patternNeuronsWithConnections.has(mapping.pattern_neuron_id)) {
				this.patternPeaks.delete(mapping.pattern_neuron_id);
				deletedPatternPeaks++;
			}
		}

		// 4. NEURON CLEANUP: Remove orphaned neurons with no connections, patterns, or activity
		const neuronsToDelete = [];
		for (const [neuronId] of this.neurons.neurons) {
			// Check if neuron has any connections (from or to)
			const hasFromConnections = this.connections.byFromDistance.has(neuronId);
			const hasToConnections = this.connections.byTo.has(neuronId);

			// Check if neuron is a pattern neuron
			const isPatternNeuron = this.patterns.byPattern.has(neuronId);

			// Check if neuron is currently active
			const isActive = this.activeNeurons.getAll().some(an => an.neuron_id === neuronId);

			if (!hasFromConnections && !hasToConnections && !isPatternNeuron && !isActive) {
				neuronsToDelete.push(neuronId);
			}
		}

		for (const neuronId of neuronsToDelete) {
			this.neurons.delete(neuronId);
			deletedNeurons++;
		}

		const elapsed = performance.now() - startTime;
		if (this.debug) {
			console.log(`Forget cycle: deleted ${deletedConnections} connections, ${deletedPatterns} pattern entries, ${deletedPatternPeaks} pattern peaks, ${deletedNeurons} neurons (${elapsed.toFixed(2)}ms)`);
		}
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

	/**
	 * Get detailed inference information for diagnostic output (Memory implementation)
	 * Note: This is a simplified stub for BrainMemory - full implementation would require
	 * tracking source information similar to BrainMySQL's inference_sources tables
	 */
	async getInferenceDetails(level) {
		// For now, return basic inference info without detailed source tracking
		// Full implementation would require adding source tracking to in-memory structures
		const details = [];

		if (!this.inferredNeurons.has(level)) return details;

		const levelInferences = this.inferredNeurons.get(level);
		for (const [neuronId, data] of levelInferences) {
			if (data.age !== 0) continue;

			const coords = this.neurons.getCoordinates(neuronId);
			const coordinates = {};
			for (const coord of coords) {
				const dimName = this.dimensionIdToName[coord.dimension_id];
				if (dimName) coordinates[dimName] = coord.val;
			}

			details.push({
				neuron_id: neuronId,
				strength: data.strength,
				coordinates,
				sources: [] // Would need source tracking for full implementation
			});
		}

		return details;
	}
}