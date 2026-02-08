import getMySQLConnection from './db/db.js';
import { Neuron } from './neurons/neuron.js';
import { Dimension } from './channels/dimension.js';
import Channel from './channels/channel.js';

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
	 * loads channels and dimensions between code and database.
	 * Loads channels from DB, validates against registered classes, instantiates channels,
	 * handles new channels, and updates thalamus with reconciled channels.
	 * @param {Thalamus} thalamus - Thalamus instance with registered channel classes
	 */
	async loadChannels(thalamus) {

		// Load channels and dimensions from database
		const [channelRows] = await this.conn.query('SELECT id, name FROM channels');
		const [dimensionRows] = await this.conn.query('SELECT id, name FROM dimensions');

		// load all dimensions and let each channel pick what it needs
		const dbDimensions = dimensionRows.map(row => new Dimension(row.name, row.id));

		// Track which channels have been processed
		const processedChannels = new Set();

		// Process each DB channel
		for (const channelRow of channelRows) {
			const channelName = channelRow.name;
			const channelId = channelRow.id;
			const channelClass = thalamus.channelClasses.get(channelName);

			// Fatal error if channel class not found
			if (!channelClass) throw new Error(`Channel class not found: ${channelName}. Code not compatible.`);

			// Instantiate channel with DB dimensions
			const channelInstance = new channelClass(channelName, dbDimensions);

			// Restore the channel ID from database
			channelInstance.id = channelId;
			if (channelId >= Channel.nextId) Channel.nextId = channelId + 1;

			// Add instantiated channel to thalamus
			thalamus.addChannel(channelName, channelInstance);
			processedChannels.add(channelName);

			if (thalamus.debug) console.log(`Loaded channel from DB: ${channelName} (id: ${channelId})`);
		}

		// Process new channels (registered but not in DB)
		for (const [channelName, channelClass] of thalamus.channelClasses) {
			if (processedChannels.has(channelName)) continue;

			// New channel - instantiate without dimensions (will create new ones)
			const channelInstance = new channelClass(channelName);
			thalamus.addChannel(channelName, channelInstance);

			// Save new channel to database
			await this.conn.query('INSERT INTO channels (id, name) VALUES (?, ?)', [channelInstance.id, channelName]);

			// Save new dimensions to database
			const dimensions = channelInstance.getEventDimensions().concat(channelInstance.getOutputDimensions());
			for (const dim of dimensions)
				await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES (?, ?)', [dim.id, dim.name]);

			if (thalamus.debug) console.log(`Added new channel to DB: ${channelName} (id: ${channelInstance.id})`);
		}

		// Set up channel name/id mappings (replaces old initializeChannels functionality)
		const channelNameToId = {};
		const channelIdToName = {};
		for (const [channelName, channel] of thalamus.getAllChannels()) {
			channelNameToId[channelName] = channel.id;
			channelIdToName[channel.id] = channelName;
		}
		thalamus.setChannelMappings(channelNameToId, channelIdToName);
		if (thalamus.debug) console.log('Channels reconciled:', channelNameToId);
	}

	/**
	 * Load neurons from MySQL and populate Thalamus.
	 * Clears existing neurons, loads from DB, and updates Neuron.nextId.
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async loadNeurons(thalamus) {

		// Clear existing neurons
		thalamus.reset();

		console.log('Loading neurons from MySQL...');

		// Load all components (Neuron.nextId is updated automatically during loading)
		await this.loadBaseNeurons(thalamus);
		await this.loadPatternNeurons(thalamus);
		await this.loadConnections(thalamus);
		await this.loadPatternContext(thalamus);
		await this.loadPatternConnections(thalamus);

		console.log(`Neurons loaded: ${thalamus.getNeuronCount()} total, next ID: ${Neuron.nextId}`);
	}

	/**
	 * Load base neurons (SensoryNeurons) with their coordinates
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @returns {Promise<number>} Maximum neuron ID found
	 */
	async loadBaseNeurons(thalamus) {
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
			const dimName = thalamus.getDimensionName(row.dimension_id);
			if (dimName) coordsByNeuron.get(row.neuron_id)[dimName] = row.val;
		}

		// Create sensory neuron objects with their database IDs
		let maxId = 0;
		for (const row of baseRows) {
			const coords = coordsByNeuron.get(row.neuron_id) || {};
			const neuron = Neuron.createSensory(row.channel_name, row.type, coords, row.neuron_id);
			thalamus.addNeuron(neuron);
			if (row.neuron_id > maxId) maxId = row.neuron_id;
		}
		console.log(`  Loaded ${baseRows.length} sensory neurons`);
		return maxId;
	}

	/**
	 * Load pattern neurons with their peaks
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @returns {Promise<number>} Maximum neuron ID found
	 */
	async loadPatternNeurons(thalamus) {
		const [patternRows] = await this.conn.query(`
			SELECT n.id, n.level, pp.peak_neuron_id
			FROM neurons n
			JOIN pattern_peaks pp ON pp.pattern_neuron_id = n.id
			WHERE n.level > 0
			ORDER BY n.level
		`);

		let maxId = 0;
		let skipped = 0;
		for (const row of patternRows) {
			const peak = thalamus.getNeuron(row.peak_neuron_id);
			if (!peak) {
				console.warn(`  Warning: Pattern ${row.id} references missing peak ${row.peak_neuron_id}`);
				skipped++;
				continue;
			}
			const pattern = Neuron.createPattern(row.level, peak, row.id);
			thalamus.addNeuron(pattern);
			if (row.id > maxId) maxId = row.id;
		}
		console.log(`  Loaded ${patternRows.length - skipped} pattern neurons (${skipped} skipped due to missing peaks)`);
		return maxId;
	}

	/**
	 * Load connections into sensory neuron connections
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async loadConnections(thalamus) {
		const [connRows] = await this.conn.query('SELECT from_neuron_id, to_neuron_id, distance, strength, reward FROM connections');
		let connCount = 0;
		for (const row of connRows) {
			const fromNeuron = thalamus.getNeuron(row.from_neuron_id);
			const toNeuron = thalamus.getNeuron(row.to_neuron_id);
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async loadPatternContext(thalamus) {
		const [pastRows] = await this.conn.query('SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM pattern_past');
		let pastCount = 0;
		for (const row of pastRows) {
			const pattern = thalamus.getNeuron(row.pattern_neuron_id);
			const contextNeuron = thalamus.getNeuron(row.context_neuron_id);
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async loadPatternConnections(thalamus) {
		const [futureRows] = await this.conn.query('SELECT pattern_neuron_id, inferred_neuron_id, distance, strength, reward FROM pattern_future');
		let futureCount = 0;
		for (const row of futureRows) {
			const pattern = thalamus.getNeuron(row.pattern_neuron_id);
			const inferredNeuron = thalamus.getNeuron(row.inferred_neuron_id);
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async backupBrain(thalamus) {
		console.log('Backing up brain to MySQL...');

		// Build neuron -> ID reverse mapping for connections/patterns
		const neuronToId = new Map();
		for (const [id, neuron] of thalamus.getAllNeuronEntries())
			neuronToId.set(neuron, id);

		// Backup all components
		await this.backupNeuronsTable(thalamus);
		await this.backupBaseNeurons(thalamus);
		await this.backupConnections(thalamus, neuronToId);
		await this.backupPatternPeaks(thalamus, neuronToId);
		await this.backupPatternContext(thalamus, neuronToId);
		await this.backupPatternConnections(thalamus, neuronToId);

		console.log('Brain backed up to MySQL.');
	}

	/**
	 * Backup neurons table
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async backupNeuronsTable(thalamus) {
		await this.conn.query('TRUNCATE neurons');
		const neuronCount = thalamus.getNeuronCount();
		if (neuronCount > 0) {
			const rows = [];
			for (const [id, neuron] of thalamus.getAllNeuronEntries())
				rows.push([id, neuron.level]);
			await this.conn.query('INSERT INTO neurons (id, level) VALUES ?', [rows]);
		}
		console.log(`  Saved ${neuronCount} neurons`);
	}

	/**
	 * Backup base_neurons and coordinates
	 * @param {Thalamus} thalamus - Thalamus instance
	 */
	async backupBaseNeurons(thalamus) {
		await this.conn.query('TRUNCATE base_neurons');
		await this.conn.query('TRUNCATE coordinates');
		const baseRows = [];
		const valueRows = [];
		for (const [id, neuron] of thalamus.getAllNeuronEntries()) {
			if (neuron.level !== 0) continue;
			const channelId = thalamus.getChannelId(neuron.channel);
			baseRows.push([id, channelId, neuron.type]);
			for (const [dimName, val] of Object.entries(neuron.coordinates)) {
				const dimId = thalamus.getDimensionId(dimName);
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupConnections(thalamus, neuronToId) {
		await this.conn.query('TRUNCATE connections');
		const connRows = [];
		for (const [fromId, neuron] of thalamus.getAllNeuronEntries()) {
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternPeaks(thalamus, neuronToId) {
		await this.conn.query('TRUNCATE pattern_peaks');
		const peakRows = [];
		for (const [patternId, neuron] of thalamus.getAllNeuronEntries()) {
			if (neuron.level === 0) continue;
			const peakId = neuronToId.get(neuron.peak);
			if (peakId)
				peakRows.push([patternId, peakId]);
		}
		if (peakRows.length > 0)
			await this.conn.query('INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id) VALUES ?', [peakRows]);
		console.log(`  Saved ${peakRows.length} pattern peaks`);
	}

	/**
	 * Backup pattern context (from peak's routing table to pattern_past)
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternContext(thalamus, neuronToId) {
		await this.conn.query('TRUNCATE pattern_past');
		const pastRows = [];
		for (const [_, neuron] of thalamus.getAllNeuronEntries())
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
	 * @param {Thalamus} thalamus - Thalamus instance
	 * @param {Map} neuronToId - Map of neuron object to ID
	 */
	async backupPatternConnections(thalamus, neuronToId) {
		await this.conn.query('TRUNCATE pattern_future');
		const futureRows = [];
		for (const [patternId, neuron] of thalamus.getAllNeuronEntries()) {
			if (neuron.level === 0) continue;
			for (const [distance, targets] of neuron.connections)
				for (const [inferredNeuron, pred] of targets) {
					const inferredId = neuronToId.get(inferredNeuron);
					if (inferredId)
						futureRows.push([patternId, inferredId, distance, pred.strength, pred.reward || 0]);
				}
		}
		if (futureRows.length > 0)
			await this.conn.query('INSERT INTO pattern_future (pattern_neuron_id, inferred_neuron_id, distance, strength, reward) VALUES ?', [futureRows]);
		console.log(`  Saved ${futureRows.length} pattern connections (to pattern_future)`);
	}

	/**
	 * Truncate the brain tables for database reset
	 */
	async reset() {
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