import getMySQLConnection from './db/db.js';
import { Neuron } from './neurons/neuron.js';

/**
 * BrainDB - Database backup and restore operations for Brain
 * Handles persistence of neurons, connections, and patterns to MySQL
 */
export class BrainDB {
	constructor() {
		this.conn = null;
	}

	/**
	 * Initialize the database connection
	 */
	async initDB() {
		this.conn = await getMySQLConnection();
	}

	/**
	 * Initialize channels in DB
	 */
	async initializeChannels(channels) {
		for (const [channelName, channel] of channels)
			await this.conn.query('INSERT IGNORE INTO channels (id, name) VALUES (?, ?)', [channel.id, channelName]);
	}

	/**
	 * Initialize dimensions for all registered channels
	 */
	async initializeDimensions(channels) {
		for (const [, channel] of channels) {
			for (const dim of channel.getEventDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
			for (const dim of channel.getOutputDimensions())
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);
		}
	}

	/**
	 * Load neurons from MySQL and populate Brain's neuron maps.
	 * Clears existing neurons, loads from DB, and updates Neuron.nextId.
	 * @param {Map<string, number>} dimensionIdToName - Dimension ID to name mapping
	 * @param {Map} neurons - Brain's neurons map to populate
	 * @param {Map} neuronsByValue - Brain's neuronsByValue map to populate
	 * @returns {Promise<{neurons: Map, neuronsByValue: Map}>}
	 */
	async loadAndPopulateNeurons(dimensionIdToName, neurons, neuronsByValue) {
		// Clear existing neurons
		neurons.clear();
		neuronsByValue.clear();
		Neuron.nextId = 1;

		console.log('Loading neurons from MySQL...');

		// Track max ID
		let maxId = 0;

		// Load all components
		maxId = Math.max(maxId, await this.loadBaseNeurons(neurons, dimensionIdToName));
		maxId = Math.max(maxId, await this.loadPatternNeurons(neurons));
		await this.loadConnections(neurons);
		await this.loadPatternContext(neurons);
		await this.loadPatternConnections(neurons);

		// Populate neuronsByValue for base level neurons
		for (const neuron of neurons.values())
			if (neuron.level === 0)
				neuronsByValue.set(neuron.valueKey, neuron);

		// Update Neuron.nextId
		Neuron.nextId = maxId + 1;

		console.log(`Neurons loaded: ${neurons.size} total, max ID: ${maxId}`);
		return { neurons, neuronsByValue };
	}

	/**
	 * Load base neurons (SensoryNeurons) with their coordinates
	 * @param {Map} neurons - Map to populate with loaded neurons
	 * @param {Map<string, number>} dimensionIdToName - Dimension ID to name mapping
	 * @returns {Promise<number>} Maximum neuron ID found
	 */
	async loadBaseNeurons(neurons, dimensionIdToName) {
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
			const dimName = dimensionIdToName[row.dimension_id];
			if (dimName) coordsByNeuron.get(row.neuron_id)[dimName] = row.val;
		}

		// Create sensory neuron objects
		let maxId = 0;
		for (const row of baseRows) {
			const coords = coordsByNeuron.get(row.neuron_id) || {};
			const neuron = Neuron.createSensory(row.channel_name, row.type, coords);
			neurons.set(row.neuron_id, neuron);
			if (row.neuron_id > maxId) maxId = row.neuron_id;
		}
		console.log(`  Loaded ${baseRows.length} sensory neurons`);
		return maxId;
	}

	/**
	 * Load pattern neurons with their peaks
	 * @param {Map} neurons - Map to populate with loaded neurons
	 * @returns {Promise<number>} Maximum neuron ID found
	 */
	async loadPatternNeurons(neurons) {
		const [patternRows] = await this.conn.query(`
			SELECT n.id, n.level, pp.peak_neuron_id, pp.strength
			FROM neurons n
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = n.id
			WHERE n.level > 0
			ORDER BY n.level
		`);

		let maxId = 0;
		for (const row of patternRows) {
			const peak = neurons.get(row.peak_neuron_id);
			if (!peak) {
				console.warn(`  Warning: Pattern ${row.id} references missing peak ${row.peak_neuron_id}`);
				continue;
			}
			const pattern = Neuron.createPattern(row.level, peak);
			pattern.peakStrength = row.strength;
			neurons.set(row.id, pattern);
			if (row.id > maxId) maxId = row.id;
		}
		console.log(`  Loaded ${patternRows.length} pattern neurons`);
		return maxId;
	}

	/**
	 * Load connections into sensory neuron connections
	 * @param {Map} neurons - Map of loaded neurons
	 */
	async loadConnections(neurons) {
		const [connRows] = await this.conn.query('SELECT from_neuron_id, to_neuron_id, distance, strength, reward FROM connections');
		let connCount = 0;
		for (const row of connRows) {
			const fromNeuron = neurons.get(row.from_neuron_id);
			const toNeuron = neurons.get(row.to_neuron_id);
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
	}

	/**
	 * Load pattern_past into peak's contexts routing table
	 * @param {Map} neurons - Map of loaded neurons
	 */
	async loadPatternContext(neurons) {
		const [pastRows] = await this.conn.query('SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM pattern_past');
		let pastCount = 0;
		for (const row of pastRows) {
			const pattern = neurons.get(row.pattern_neuron_id);
			const contextNeuron = neurons.get(row.context_neuron_id);
			if (!pattern || !contextNeuron) continue;
			if (pattern.level === 0) continue;

			const peak = pattern.peak;
			if (!peak) continue;
			const context = peak.getOrCreateContext(pattern);
			context.add(contextNeuron, row.context_age, row.strength);
			pastCount++;
		}
		console.log(`  Loaded ${pastCount} pattern context entries (from pattern_past)`);
	}



	/**
	 * Load pattern_future into pattern.connections
	 * @param {Map} neurons - Map of loaded neurons
	 */
	async loadPatternConnections(neurons) {
		const [futureRows] = await this.conn.query('SELECT pattern_neuron_id, inferred_neuron_id, distance, strength, reward FROM pattern_future');
		let futureCount = 0;
		for (const row of futureRows) {
			const pattern = neurons.get(row.pattern_neuron_id);
			const inferredNeuron = neurons.get(row.inferred_neuron_id);
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
	}

	/**
	 * Backup brain state to MySQL.
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map<string, number>} channelNameToId - Channel name to ID mapping
	 * @param {Map<string, number>} dimensionNameToId - Dimension name to ID mapping
	 */
	async backupBrain(neurons, channelNameToId, dimensionNameToId) {
		console.log('Backing up brain to MySQL...');

		// Build neuron -> ID reverse mapping for connections/patterns
		const neuronToId = new Map();
		for (const [id, neuron] of neurons)
			neuronToId.set(neuron, id);

		// Backup all components
		await this.backupNeuronsTable(neurons);
		await this.backupBaseNeurons(neurons, channelNameToId, dimensionNameToId);
		await this.backupConnections(neurons, neuronToId);
		await this.backupPatternPeaks(neurons, neuronToId);
		await this.backupPatternContext(neurons, neuronToId);
		await this.backupPatternConnections(neurons, neuronToId);

		console.log('Brain backed up to MySQL.');
	}

	/**
	 * Backup neurons table
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 */
	async backupNeuronsTable(neurons) {
		await this.conn.query('TRUNCATE neurons');
		if (neurons.size > 0) {
			const rows = [];
			for (const [id, neuron] of neurons)
				rows.push([id, neuron.level]);
			await this.conn.query('INSERT INTO neurons (id, level) VALUES ?', [rows]);
		}
		console.log(`  Saved ${neurons.size} neurons`);
	}

	/**
	 * Backup base_neurons and coordinates
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map<string, number>} channelNameToId - Channel name to ID mapping
	 * @param {Map<string, number>} dimensionNameToId - Dimension name to ID mapping
	 */
	async backupBaseNeurons(neurons, channelNameToId, dimensionNameToId) {
		await this.conn.query('TRUNCATE base_neurons');
		await this.conn.query('TRUNCATE coordinates');
		const baseRows = [];
		const valueRows = [];
		for (const [id, neuron] of neurons) {
			if (neuron.level !== 0) continue;
			const channelId = channelNameToId[neuron.channel];
			baseRows.push([id, channelId, neuron.type]);
			for (const [dimName, val] of Object.entries(neuron.coordinates)) {
				const dimId = dimensionNameToId[dimName];
				if (dimId !== undefined)
					valueRows.push([id, dimId, val]);
			}
		}
		if (baseRows.length > 0)
			await this.conn.query('INSERT INTO base_neurons (neuron_id, channel_id, type) VALUES ?', [baseRows]);
		if (valueRows.length > 0)
			await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [valueRows]);
		console.log(`  Saved ${baseRows.length} base neurons, ${valueRows.length} coordinates`);
	}

	/**
	 * Backup connections
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupConnections(neurons, neuronToId) {
		await this.conn.query('TRUNCATE connections');
		const connRows = [];
		for (const [fromId, neuron] of neurons) {
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
	}

	/**
	 * Backup pattern_peaks
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternPeaks(neurons, neuronToId) {
		await this.conn.query('TRUNCATE pattern_peaks');
		const peakRows = [];
		for (const [patternId, neuron] of neurons) {
			if (neuron.level === 0) continue;
			const peakId = neuronToId.get(neuron.peak);
			if (peakId)
				peakRows.push([patternId, peakId, neuron.peakStrength]);
		}
		if (peakRows.length > 0)
			await this.conn.query('INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength) VALUES ?', [peakRows]);
		console.log(`  Saved ${peakRows.length} pattern peaks`);
	}

	/**
	 * Backup pattern context (from peak's routing table to pattern_past)
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternContext(neurons, neuronToId) {
		await this.conn.query('TRUNCATE pattern_past');
		const pastRows = [];
		for (const [_, neuron] of neurons)
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
	}

	/**
	 * Backup pattern connections (to pattern_future table for compatibility)
	 * @param {Map} neurons - Map of neuron ID to neuron object
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternConnections(neurons, neuronToId) {
		await this.conn.query('TRUNCATE pattern_future');
		const futureRows = [];
		for (const [patternId, neuron] of neurons) {
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
	}

	/**
	 * Truncate the brain tables for database reset
	 */
	async truncateTables() {
		const tables = [
			'channels',
			'dimensions',
			'neurons',
			'base_neurons',
			'coordinates',
			'connections',
			'pattern_peaks',
			'pattern_past',
			'pattern_future'
		];
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		await Promise.all(tables.map(table => this.conn.query(`TRUNCATE ${table}`)));
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}
}