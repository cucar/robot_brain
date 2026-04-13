import getMySQLConnection from '../db/db.js';
import { Neuron } from './neuron.js';
import { Thalamus } from './thalamus.js';
import { Dimension } from '../channels/dimension.js';
import { Channel } from '../channels/channel.js';

/**
 * Database backup and restore operations for Brain
 * Handles persistence of neurons, connections, and patterns to MySQL
 */
export class Database {
	constructor(debug, patternForgetRate, mergeThreshold) {
		this.conn = null;
		this.debug = debug;
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;
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
	 * @returns {Promise<{channels: Map, channelNameToId: Object, channelIdToName: Object}>}
	 */
	async loadChannels(channelClasses) {

		// Load channels and dimensions from database
		const [channelRows] = await this.conn.query('SELECT id, name FROM channels order by id');
		const [dimensionRows] = await this.conn.query('SELECT id, name FROM dimensions order by id');

		// Create dimension objects
		const dbDimensions = dimensionRows.map(row => new Dimension(row.name, row.id));

		// Create channel instances and build ID maps
		const channels = new Map();
		const channelNameToId = {};
		const channelIdToName = {};
		let maxChannelId = 0;
		for (const row of channelRows) {

			// Validate that channel class exists
			const channelClass = channelClasses.get(row.name);
			if (!channelClass)
				throw new Error(`Channel class not found: ${row.name}. Code not compatible.`);

			// Instantiate channel with DB id and dimensions
			const channel = new channelClass(row.name, this.debug, row.id, dbDimensions);
			channels.set(row.name, channel);
			channelNameToId[row.name] = row.id;
			channelIdToName[row.id] = row.name;

			// Track max channel id
			if (row.id > maxChannelId) maxChannelId = row.id;
		}

		// Update Channel.nextId for new channels to be created after this
		if (maxChannelId >= Channel.nextId) Channel.nextId = maxChannelId + 1;

		console.log(`Channels loaded: ${channels.size} total, next ID: ${Channel.nextId}`);
		return { channels, channelNameToId, channelIdToName };
	}

	/**
	 * Load a complete brain snapshot from MySQL (channels, dimensions, and neurons).
	 * Returns the same format as thalamus.getSnapshot().
	 * @param {Map} channelClasses - Map of channel name to channel class
	 * @returns {Promise<{neurons: Array<{neuron: Neuron, channel: string|undefined}>, channels: Map, channelNameToId: Object, dimensionNameToId: Object}>}
	 */
	async loadSnapshot(channelClasses) {

		const { channels, channelNameToId, channelIdToName } = await this.loadChannels(channelClasses);

		// Build dimension ID maps from channel instances
		const { nameToId: dimensionNameToId, idToName: dimensionIdToName } = Thalamus.buildDimensionMaps(channels);

		// Load neurons (dimension maps must be built first — coordinate loading needs dimension ID→name lookups)
		console.log('Loading neurons from MySQL...');
		const { neurons: neuronMap, levels: neuronLevelMap } = await this.loadNeuronsTable();

		const { neuronChannels, neuronTypes } = await this.loadBaseNeurons(channelIdToName);
		const neuronCoordinates = await this.loadCoordinates(dimensionIdToName);
		await this.loadConnections(neuronMap);
		const neuronParentMap = await this.loadPatterns(neuronMap);
		await this.loadPatternContexts(neuronMap);

		// Build snapshot
		const neurons = [];
		for (const [neuronId, neuron] of neuronMap) {
			const level = neuronLevelMap.get(neuronId);
			const entry = { neuron, level };
			if (level === 0) {
				entry.channel = neuronChannels.get(neuron.id);
				entry.type = neuronTypes.get(neuron.id);
				entry.coordinates = neuronCoordinates.get(neuron.id);
			}
			const parentId = neuronParentMap.get(neuron.id);
			if (parentId) entry.parentId = parentId;
			neurons.push(entry);
		}
		return { neurons, channels, channelNameToId, dimensionNameToId };
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
		const levels = new Map();
		for (const row of rows) {

			// create the neuron with its activation strength
			const neuron = new Neuron(this.patternForgetRate, this.mergeThreshold, row.id);
			neurons.set(row.id, neuron);
			levels.set(row.id, row.level);

			// update max id
			if (row.id > maxId) maxId = row.id;
		}

		// Update Neuron.nextId for new neurons to be created after this
		if (maxId >= Neuron.nextId) Neuron.nextId = maxId + 1;

		console.log(`Neurons loaded: ${neurons.size} total, next ID: ${Neuron.nextId}`);
		return { neurons, levels };
	}

	/**
	 * Load base_neurons table
	 */
	async loadBaseNeurons(channelIdToName) {

		// get the base neuron data from the database
		const [rows] = await this.conn.query('SELECT neuron_id, channel_id, type FROM base_neurons');

		// update all base neurons and collect channel/type metadata for caller
		const neuronChannels = new Map(); // neuronId → channelName
		const neuronTypes = new Map(); // neuronId → type
		for (const row of rows) {
			const channelName = channelIdToName[row.channel_id];
			neuronChannels.set(row.neuron_id, channelName);
			neuronTypes.set(row.neuron_id, row.type);
		}
		console.log(`  Loaded ${rows.length} base neurons from table`);
		return { neuronChannels, neuronTypes };
	}

	/**
	 * Load coordinates table
	 * @returns {Map<number, object>} neuronId → coordinates
	 */
	async loadCoordinates(dimensionIdToName) {

		// get the base neuron coordinates from the database
		const [rows] = await this.conn.query('SELECT neuron_id, dimension_id, val FROM coordinates');

		// build neuronId → coordinates map
		const neuronCoordinates = new Map(); // neuronId → coordinates
		for (const row of rows) {
			if (!neuronCoordinates.has(row.neuron_id)) neuronCoordinates.set(row.neuron_id, {});
			neuronCoordinates.get(row.neuron_id)[dimensionIdToName[row.dimension_id]] = row.val;
		}

		console.log(`  Loaded ${rows.length} coordinates from table`);
		return neuronCoordinates;
	}

	/**
	 * Load connections from connections table
	 */
	async loadConnections(neurons) {

		// get the connections from the database
		const [rows] = await this.conn.query(`
			SELECT from_neuron_id, to_neuron_id, distance, strength, reward 
			FROM connections
		`);

		// update all connections (using neuron IDs as keys)
		for (const row of rows) {

			// get the neuron that has the connection
			const fromNeuron = neurons.get(row.from_neuron_id);
			if (!fromNeuron) throw new Error('Connection source neuron not found');

			// verify the target neuron exists
			if (!neurons.has(row.to_neuron_id)) throw new Error('Connection target neuron not found');

			// add the connection using neuron ID as key
			fromNeuron.createConnection(row.distance, row.to_neuron_id, Number(row.strength), Number(row.reward));
		}

		console.log(`  Loaded ${rows.length} connections from table`);
		return rows;
	}

	/**
	 * Load patterns table
	 * @returns {Map<number, number>} neuronId -> parentNeuronId
	 */
	async loadPatterns(neurons) {

		// get the patterns
		const [rows] = await this.conn.query('SELECT pattern_neuron_id, parent_neuron_id, strength FROM patterns');

		// create the pattern to parent mappings (stored in Thalamus via snapshot)
		const neuronParentMap = new Map();
		for (const row of rows) {

			// map the pattern to its parent by ID
			neuronParentMap.set(row.pattern_neuron_id, row.parent_neuron_id);

			// add the pattern to the parent's routing table by ID
			const parent = neurons.get(row.parent_neuron_id);
			parent.addChild(row.pattern_neuron_id, Number(row.strength));
		}
		return neuronParentMap;
	}

	/**
	 * Load pattern_past table
	 */
	async loadPatternContexts(neurons) {

		// get the pattern contexts
		const [rows] = await this.conn.query(`
			SELECT p.parent_neuron_id, pp.pattern_neuron_id, pp.context_neuron_id, pp.context_age, pp.strength
			FROM pattern_past pp
			JOIN patterns p ON p.pattern_neuron_id = pp.pattern_neuron_id
		`);

		// load the pattern contexts in the parent
		for (const row of rows) {

			// get the context neuron
			const contextNeuron = neurons.get(row.context_neuron_id);
			if (!contextNeuron) throw new Error(`contextNeuron null: ${row.context_neuron_id}`);

			// get the parent neuron that will store the pattern route and context
			const parent = neurons.get(row.parent_neuron_id);
			if (!parent) throw new Error(`parent null: ${row.parent_neuron_id}`);

			// add the pattern to the parent routing table with the given distance
			parent.addContext(row.pattern_neuron_id, contextNeuron.id, row.context_age, Number(row.strength));

			// update the context references of the neuron used in pattern context
			contextNeuron.addContextRef(row.parent_neuron_id, row.context_age);
		}
		console.log(`  Loaded ${rows.length} pattern_past entries from table`);
	}

	/**
	 * Save a complete brain snapshot to MySQL.
	 * @param {Object} snapshot - Brain state snapshot from thalamus.getSnapshot()
	 */
	async saveSnapshot(snapshot) {
		await this.backupChannels(snapshot.channels);
		await this.backupDimensions(snapshot.channels);
		await this.backupNeurons(snapshot);
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
			const dimensions = channel.getEventDimensions().concat(channel.getActionDimensions());
			for (const dim of dimensions) rows.push([dim.id, dim.name]);
		}
		if (rows.length === 0) return;
		await this.conn.query('INSERT IGNORE INTO dimensions (id, name) VALUES ?', [rows]);
		console.log(`  Saved ${rows.length} dimensions`);
	}

	/**
	 * Backup neurons state to MySQL.
	 * @param {Object} snapshot - Brain state snapshot from thalamus.getSnapshot()
	 */
	async backupNeurons(snapshot) {
		console.log('Backing up brain to MySQL...');

		if (snapshot.neurons.length === 0) return;

		await this.backupNeuronsTable(snapshot.neurons);
		await this.backupBaseNeurons(snapshot);
		await this.backupConnections(snapshot.neurons);
		await this.backupPatterns(snapshot.neurons);
		await this.backupPatternContext(snapshot.neurons);

		console.log('Brain backed up to MySQL.');
	}

	/**
	 * Backup to neurons table
	 */
	async backupNeuronsTable(neurons) {
		await this.conn.query('TRUNCATE neurons');

		const rows = [];
		for (const { neuron, level } of neurons) {
			rows.push([neuron.id, level]);
		}

		await this.conn.query('INSERT INTO neurons (id, level) VALUES ?', [rows]);
		console.log(`  Saved ${rows.length} neurons`);
	}

	/**
	 * Backup base_neurons and coordinates
	 */
	async backupBaseNeurons(snapshot) {
		await this.conn.query('TRUNCATE base_neurons');
		await this.conn.query('TRUNCATE coordinates');
		const baseRows = [];
		const valueRows = [];
		for (const { neuron, level, channel, type, coordinates } of snapshot.neurons) {
			if (level !== 0) continue;
			baseRows.push([neuron.id, snapshot.channelNameToId[channel], type]);
			for (const [dimName, val] of Object.entries(coordinates))
				valueRows.push([neuron.id, snapshot.dimensionNameToId[dimName], val]);
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
		for (const { neuron } of neurons)
			for (const [distance, targets] of neuron.connections)
				for (const [toNeuronId, conn] of targets)
					connRows.push([neuron.id, toNeuronId, distance, conn.strength, conn.reward || 0]);
		if (connRows.length === 0) return;
		await this.conn.query('INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength, reward) VALUES ?', [connRows]);
		console.log(`  Saved ${connRows.length} connections`);
	}

	/**
	 * Backup patterns
	 */
	async backupPatterns(neurons) {
		await this.conn.query('TRUNCATE patterns');

		const neuronMap = new Map();
		for (const entry of neurons) neuronMap.set(entry.neuron.id, entry.neuron);

		const patternRows = [];
		for (const { neuron, level, parentId } of neurons) {
			if (level > 0) {
				let strength = 0;
				const parent = neuronMap.get(parentId);
				if (parent) {
					const ctx = parent.routingTable.get(neuron.id);
					if (ctx) strength = ctx.activationStrength;
				}
				patternRows.push([neuron.id, parentId, strength]);
			}
		}

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
		for (const { neuron } of neurons)
			for (const { patternId, neuronId, distance, strength } of neuron.getRoutingTable())
				pastRows.push([patternId, neuronId, distance, strength]);
		if (pastRows.length === 0) return;
		await this.conn.query('INSERT INTO pattern_past (pattern_neuron_id, context_neuron_id, context_age, strength) VALUES ?', [pastRows]);
		console.log(`  Saved ${pastRows.length} pattern context entries (to pattern_past)`);
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
			'pattern_past'
		];
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 0');
		await Promise.all(tables.map(table => this.conn.query(`TRUNCATE ${table}`)));
		await this.conn.query('SET FOREIGN_KEY_CHECKS = 1');
	}
}