import getMySQLConnection from '../db/db.js';
import { Neuron } from './neuron.js';
import { Dimension } from '../channels/dimension.js';
import { Channel } from '../channels/channel.js';

/**
 * Database backup and restore operations for Brain
 * Handles persistence of neurons, connections, and patterns to MySQL
 */
export class Database {
	constructor(debug) {
		this.conn = null;
		this.debug = debug;
	}

	/**
	 * Initialize the database connection
	 */
	async initDB() {
		this.conn = await getMySQLConnection();
	}

	/**
	 * Load channels and dimensions from database and return instantiated channel objects.
	 * @param {Map} channelClasses - Map of channel name to channel class
	 * @returns {Promise<Map>} Map of channel name to channel instance
	 */
	async loadChannels(channelClasses) {

		// Load channels and dimensions from database
		const [channelRows] = await this.conn.query('SELECT id, name FROM channels order by id');
		const [dimensionRows] = await this.conn.query('SELECT id, name FROM dimensions order by id');

		// Create dimension objects
		const dbDimensions = dimensionRows.map(row => new Dimension(row.name, row.id));

		// Create channel instances
		const channels = new Map();
		let maxChannelId = 0;
		for (const row of channelRows) {

			// Validate that channel class exists
			const channelClass = channelClasses.get(row.name);
			if (!channelClass)
				throw new Error(`Channel class not found: ${row.name}. Code not compatible.`);

			// Instantiate channel with DB id and dimensions
			const channel = new channelClass(row.name, this.debug, row.id, dbDimensions);
			channels.set(row.name, channel);

			// Track max channel id
			if (row.id > maxChannelId) maxChannelId = row.id;
		}

		// Update Channel.nextId for new channels to be created after this
		if (maxChannelId >= Channel.nextId) Channel.nextId = maxChannelId + 1;

		console.log(`Channels loaded: ${channels.size} total, next ID: ${Channel.nextId}`);
		return channels;
	}

	/**
	 * Load neurons from MySQL and return neurons map.
	 * @returns {Promise<Map<number, Neuron>>} Map of neuron ID to neuron object
	 */
	async loadNeurons(channelIdToName, dimensionIdToName) {

		console.log('Loading neurons from MySQL...');

		// create the neurons first with their ids - they need to exist before we can build them
		const neurons = await this.loadNeuronsTable();

		// update base neurons and their coordinates
		await this.loadBaseNeurons(neurons, channelIdToName);
		await this.loadCoordinates(neurons, dimensionIdToName);

		// update connections used for inferences
		await this.loadConnections(neurons);

		// update patterns used for pattern matching
		await this.loadPatterns(neurons);
		await this.loadPatternContexts(neurons);

		// return the neurons to be loaded to thalamus
		return neurons;
	}

	/**
	 * Load neurons table
	 */
	async loadNeuronsTable() {

		// get the neurons from the database
		const [rows] = await this.conn.query('SELECT id, level FROM neurons');

		// create all neurons
		let maxId = 0;
		const neurons = new Map();
		for (const row of rows) {

			// create the neuron
			const neuron = new Neuron(row.level, row.id);
			neurons.set(row.id, neuron);

			// update max id
			if (row.id > maxId) maxId = row.id;
		}

		// Update Neuron.nextId for new neurons to be created after this
		if (maxId >= Neuron.nextId) Neuron.nextId = maxId + 1;

		console.log(`Neurons loaded: ${neurons.size} total, next ID: ${Neuron.nextId}`);
		return neurons;
	}

	/**
	 * Load base_neurons table
	 */
	async loadBaseNeurons(neurons, channelIdToName) {

		// get the base neuron data from the database
		const [rows] = await this.conn.query('SELECT neuron_id, channel_id, type FROM base_neurons');

		// update all base neurons
		for (const row of rows) {
			const neuron = neurons.get(row.neuron_id);
			neuron.channel = channelIdToName[row.channel_id];
			neuron.type = row.type;
		}
		console.log(`  Loaded ${rows.length} base neurons from table`);
	}

	/**
	 * Load coordinates table
	 */
	async loadCoordinates(neurons, dimensionIdToName) {

		// get the base neuron coordinates from the database
		const [rows] = await this.conn.query('SELECT neuron_id, dimension_id, val FROM coordinates');

		// update all base neuron coordinates
		for (const row of rows) {
			const neuron = neurons.get(row.neuron_id);
			if (!neuron.coordinates) neuron.coordinates = {};
			neuron.coordinates[dimensionIdToName[row.dimension_id]] = row.val;
		}

		console.log(`  Loaded ${rows.length} coordinates from table`);
	}

	/**
	 * Load connections from connections and pattern_future tables
	 */
	async loadConnections(neurons) {

		// get the connections from the database
		const [rows] = await this.conn.query(`
			SELECT from_neuron_id, to_neuron_id, distance, strength, reward 
			FROM connections
			UNION ALL
            SELECT pattern_neuron_id as from_neuron_id, inferred_neuron_id as to_neuron_id, distance, strength, reward
            FROM pattern_future
		`);

		// update all connections
		for (const row of rows) {

			// get the neuron that has the connection
			const fromNeuron = neurons.get(row.from_neuron_id);
			if (!fromNeuron) throw new Error('Connection source neuron not found');

			// get the neuron that is the target of the connection
			const toNeuron = neurons.get(row.to_neuron_id);
			if (!toNeuron) throw new Error('Connection target neuron not found');

			// add the connection
			fromNeuron.createConnection(row.distance, toNeuron, row.strength, row.reward);
		}

		console.log(`  Loaded ${rows.length} connections from table`);
		return rows;
	}

	/**
	 * Load patterns table
	 */
	async loadPatterns(neurons) {

		// get the patterns
		const [rows] = await this.conn.query('SELECT pattern_neuron_id, parent_neuron_id, strength FROM patterns');

		// create the pattern to parent mappings in neurons
		for (const row of rows) {

			// map the pattern to its parent
			const pattern = neurons.get(row.pattern_neuron_id);
			pattern.parent = neurons.get(row.parent_neuron_id);

			// set activation strength of the pattern
			pattern.setActivationStrength(row.strength);

			// add the pattern to the parent's children array without any context for now - will be loaded later
			pattern.parent.addChild(pattern);
		}
	}

	/**
	 * Load pattern_past table
	 */
	async loadPatternContexts(neurons) {

		// get the pattern contexts
		const [rows] = await this.conn.query(`
			SELECT pattern_neuron_id, context_neuron_id, context_age, strength 
			FROM pattern_past
		`);

		// load the pattern contexts in the parent
		for (const row of rows) {
			const pattern = neurons.get(row.pattern_neuron_id);
			const contextNeuron = neurons.get(row.context_neuron_id);
			if (!contextNeuron) throw new Error(`contextNeuron null: ${row.context_neuron_id}`);
			pattern.addPatternContext(contextNeuron, row.context_age, row.strength);
		}
		console.log(`  Loaded ${rows.length} pattern_past entries from table`);
	}

	/**
	 * Backup channels to MySQL.
	 */
	async backupChannels(channels) {
		const rows = [];
		for (const [channelName, channel] of channels)
			rows.push([channel.id, channelName]);
		if (rows.length === 0) return;
		await this.conn.query('INSERT IGNORE INTO channels (id, name) VALUES ?', [rows]);
		console.log(`  Saved ${rows.length} channels`);
	}

	/**
	 * Backup dimensions to MySQL.
	 */
	async backupDimensions(channels) {
		const rows = [];
		for (const [, channel] of channels) {
			const dimensions = channel.getEventDimensions().concat(channel.getOutputDimensions());
			for (const dim of dimensions) rows.push([dim.id, dim.name]);
		}
		if (rows.length === 0) return;
		await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES ?', [rows]);
		console.log(`  Saved ${rows.length} dimensions`);
	}

	/**
	 * Backup neurons state to MySQL.
	 */
	async backupNeurons(neurons, channelNameToId, dimensionNameToId) {
		console.log('Backing up brain to MySQL...');

		if (neurons.length === 0) return;

		await this.backupNeuronsTable(neurons);
		await this.backupBaseNeurons(neurons, channelNameToId, dimensionNameToId);
		await this.backupConnections(neurons);
		await this.backupPatterns(neurons);
		await this.backupPatternContext(neurons);
		await this.backupPatternConnections(neurons);

		console.log('Brain backed up to MySQL.');
	}

	/**
	 * Backup to neurons table
	 */
	async backupNeuronsTable(neurons) {
		await this.conn.query('TRUNCATE neurons');
		const rows = [];
		for (const neuron of neurons) rows.push([neuron.id, neuron.level]);
		await this.conn.query('INSERT INTO neurons (id, level) VALUES ?', [rows]);
		console.log(`  Saved ${rows.length} neurons`);
	}

	/**
	 * Backup base_neurons and coordinates
	 */
	async backupBaseNeurons(neurons, channelNameToId, dimensionNameToId) {
		await this.conn.query('TRUNCATE base_neurons');
		await this.conn.query('TRUNCATE coordinates');
		const baseRows = [];
		const valueRows = [];
		for (const neuron of neurons) {
			if (neuron.level !== 0) continue;
			baseRows.push([neuron.id, channelNameToId[neuron.channel], neuron.type]);
			for (const [dimName, val] of Object.entries(neuron.coordinates))
				valueRows.push([neuron.id, dimensionNameToId[dimName], val]);
		}
		await this.conn.query('INSERT INTO base_neurons (neuron_id, channel_id, type) VALUES ?', [baseRows]);
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [valueRows]);
		console.log(`  Saved ${baseRows.length} base neurons, ${valueRows.length} coordinates`);
	}

	/**
	 * Backup connections
	 */
	async backupConnections(neurons) {
		await this.conn.query('TRUNCATE connections');
		const connRows = [];
		for (const neuron of neurons)
			for (const [distance, targets] of neuron.connections)
				for (const [toNeuron, conn] of targets)
					connRows.push([neuron.id, toNeuron.id, distance, conn.strength, conn.reward || 0]);
		await this.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength, reward) VALUES ?', [connRows]);
		console.log(`  Saved ${connRows.length} connections`);
	}

	/**
	 * Backup patterns
	 */
	async backupPatterns(neurons) {
		await this.conn.query('TRUNCATE patterns');
		const patternRows = [];
		for (const pattern of neurons)
			if (pattern.level > 0)
				patternRows.push([pattern.id, pattern.parent.id, pattern.activationStrength]);
		if (patternRows.length === 0) return;
		await this.conn.query('INSERT INTO patterns (pattern_neuron_id, parent_neuron_id, strength) VALUES ?', [patternRows]);
		console.log(`  Saved ${patternRows.length} patterns`);
	}

	/**
	 * Backup pattern context (from peak's routing table to pattern_past)
	 */
	async backupPatternContext(neurons) {
		await this.conn.query('TRUNCATE pattern_past');
		const pastRows = [];
		for (const neuron of neurons)
			if (neuron.level > 0)
				for (const { neuron: ctxNeuron, distance, strength } of neuron.getPatternContext())
					pastRows.push([neuron.id, ctxNeuron.id, distance, strength]);
		if (pastRows.length === 0) return;
		await this.conn.query('INSERT INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age, strength) VALUES ?', [pastRows]);
		console.log(`  Saved ${pastRows.length} pattern context entries (to pattern_past)`);
	}

	/**
	 * Backup pattern connections (to pattern_future table for compatibility)
	 */
	async backupPatternConnections(neurons) {
		await this.conn.query('TRUNCATE pattern_future');
		const futureRows = [];
		for (const neuron of neurons)
			if (neuron.level > 0)
				for (const [distance, targets] of neuron.connections)
					for (const [inferredNeuron, pred] of targets)
						futureRows.push([neuron.id, inferredNeuron.id, distance, pred.strength, pred.reward || 0]);
		if (futureRows.length === 0) return;
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
			'patterns',
			'pattern_past',
			'pattern_future'
		];
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		await Promise.all(tables.map(table => this.conn.query(`TRUNCATE ${table}`)));
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}
}