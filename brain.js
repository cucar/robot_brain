import db from './db/db.js';

/**
 * Artificial Brain
 */
export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.baseNeuronMaxAge = 10; // number of frames a base neuron stays active
		this.forgetCycles = 1000; // number of frames between forget cycles
		this.forgetRate = 0.1; // how much the connection strengths will be decremented by at forget cycles
		this.positiveLearningRate = 0.1; // how much pattern strengths will be incremented by when accurate
		this.negativeLearningRate = 0.1; // how much pattern strengths will be decremented by when not accurate

		// initialize the counter for forget cycle
		this.forgetCounter = 0;
	}

	/**
	 * initializes the database connection and loads dimensions
	 */
	async init() {

		// get new connection to the database
		this.conn = await db.getConnection();

		// load the dimensions
		await this.loadDimensions();
	}

	/**
	 * loads the dimensions to memory for input
	 */
	async loadDimensions() {
		this.dimensionNameToId = {};
		this.dimensionIdToName = {};
		const [rows] = await this.conn.query('SELECT id, name FROM dimensions');
		console.log(rows);
		rows.forEach(row => {
			this.dimensionNameToId[row.name] = row.id;
			this.dimensionIdToName[row.id] = row.name;
		});
		console.log('Dimensions loaded:', this.dimensionNameToId);
	}

	/**
	 * processes one frame of input values - this is an array of points [{ [dim1-name]: <value>, [dim2-name]: <value>, ... }]
	 */
	async processFrame(frame, feedback = 0) {
		if (!frame || frame.length === 0) return; // ignore empty frames
		console.log(`observing new frame: ${JSON.stringify(frame)}`);

		// do the whole thing as a transaction to avoid inconsistent database states
		await this.conn.beginTransaction();

		// if there's an error, we'll roll the transaction back
		try {

			// age the active neurons in memory context - sliding the window
			await this.ageNeurons();

			// now, activate base neurons in the frame
			await this.activateBaseNeurons(frame);

			// discover and activate patterns using connections - start recursion from base level
			await this.activatePatternNeurons();

			// generate predictions
			await this.generatePredictions(feedback);

			// now that the connection strengths are updated, run forget cycle periodically and delete dead connections/neurons
			await this.runForgetCycle();

			// now commit the transaction and return the new predictions
			await this.conn.commit();
			console.log('Frame processed successfully.');
		}
		catch (error) {
			await this.conn.rollback();
			console.error('Error processing frame, transaction rolled back:', error);
			throw error;
		}
	}

	/**
	 * ages neurons in the context - sliding the window across frames
	 */
	async ageNeurons() {
		console.log('Aging active neurons and predictions...');

		// age all neurons in the context
		await this.conn.query('UPDATE active_neurons SET age = age + 1');

		// deactivate aged-out neurons, but age higher level neurons slower than the lower level neurons
		// if level 0 max age = 10, level 1 max age = 100, level 2 is 1000, etc.
		await this.conn.query('DELETE FROM active_neurons WHERE age >= pow(?, level + 1)', [this.baseNeuronMaxAge]);
	}

	/**
	 * activate base neurons from frame
	 */
	async activateBaseNeurons(frame) {

		// bulk find/create neurons for all input points
		const neuronIds = await this.getFrameNeurons(frame);

		// bulk insert activations at base level
		await this.activateNeurons(neuronIds);

		// reinforce connections between active neurons in the current level
		await this.reinforceConnections(0);
	}

	/**
	 * returns base neuron ids for given set of points coming from the frame
	 */
	async getFrameNeurons(points) {

		// try tp get all the neurons that have coordinates in close ranges for each point - return format: [{ point_str, neuron_id }]
		const matches = await this.matchNeuronsFromPoints(points);
		console.log('pointNeuronMatches', matches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const neuronIds = matches.filter(p => p.neuron_ids.length > 0);

		// create neurons for points with no matching neurons
		const pointsNeedingNeurons = matches.filter(p => p.neuron_ids.length === 0).map(p => p.point_str);
		if (pointsNeedingNeurons.length > 0) {
			console.log(`${pointsNeedingNeurons.length} points need new neurons. Creating neurons once with internal dedupe.`);
			const createdNeuronIds = await this.createBaseNeurons(pointsNeedingNeurons);
			neuronIds.push(...createdNeuronIds);
		}

		// return matching neuron ids to given points
		return neuronIds;
	}

	/**
	 * matches base neurons from dimensional values for each point - return format: [{ point_str, neuron_id }]
	 */
	async matchNeuronsFromPoints(frame) {
		if (frame.length === 0) return [];

		// for each point in the frame, create a separate select query
		const unionQueries = [];
		for (const point of frame) {
			const dimensions = Object.keys(point);
			const dimCount = dimensions.length;
			
			// Build the dimension-value pairs for this point
			const dimensionValuePairs =
				Object.entries(point).map(([dimName, val]) => `(${this.dimensionNameToId[dimName]}, ${val})`);
			
			// Create a subquery for this point that finds neurons matching ALL its dimensions
			unionQueries.push(`
				SELECT '${JSON.stringify(point)}' as point_str, group_concat(neuron_id) as neuron_id
				FROM coordinates
				WHERE (dimension_id, val) IN (${dimensionValuePairs.join(', ')})
				GROUP BY neuron_id
				HAVING COUNT(*) = ${dimCount}
			`);
		}

		// combine all point queries with UNION and get the matching neurons for each point
		const sql = unionQueries.join(' UNION ALL ');
		const [rows] = await this.conn.query(sql);

		// if there are any points with multiple matches, error out
		// this is theoretically possible, but should not happen as long as we are consistent with the input dimensions
		if (rows.find(row => row.neuron_id && row.neuron_id.includes(','))) throw new Error(`Multiple point matches: ${sql}`);

		// return results
		return rows;
	}

	/**
	 * creates base neurons from a given set of points and returns their ids
	 */
	async createBaseNeurons(pointStrs) {

		// nothing to create if no points are sent - should not happen
		if (pointStrs.length === 0) return [];

		// deduplicate points and parse them
		const points = [...new Set(pointStrs)].map(pointStr => JSON.parse(pointStr));

		// bulk insert neurons and get their IDs
		const neuronIds = await this.bulkInsertNeurons(points.length);
		const created = points.map((point, idx) => ({ point, neuron_id: neuronIds[idx] }));

		// Rest of the coordinate insertion logic...
		const rows = created.flatMap(({ neuron_id, point }) =>
			Object.entries(point).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]));
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [rows]);

		// return the new neuron ids
		return neuronIds;
	}

	/**
	 * bulk insert neurons and return their IDs
	 */
	async bulkInsertNeurons(count) {

		// bulk insert neurons - mysql returns the first insert id - ids are continuous within the statement in the session
		const valuesSql = Array(count).fill('()').join(',');
		const insertNeuronsResult = await this.conn.query(`INSERT INTO neurons () VALUES ${valuesSql}`);
		const firstNeuronId = insertNeuronsResult[0].insertId;

		// compute ids from the first insert id (assume auto_increment step = 1)
		return Array.from({ length: count }, (_, idx) => firstNeuronId + idx);
	}

	/**
	 * reinforces the connections between newly active neurons (age = 0) at a level
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
            SELECT 
                f.neuron_id as from_neuron_id, -- connection from the neuron id
                t.neuron_id as to_neuron_id, -- connection to the neuron id
                f.age as distance, -- taking the time distance between age=0 neurons, so from neuron age is the distance
                1 as strength -- each observation increases strength by 1 
			FROM active_neurons f -- connections are built from the older neurons to the new neurons
			JOIN active_neurons t ON t.level = f.level -- reinforcing connections only within the same level
            WHERE t.age = 0 -- reinforcing connections for the newly activated target neurons only
            AND f.level = ? -- get the active neurons in the given level
            AND (t.neuron_id != f.neuron_id OR f.age != 0) -- if it's the same neuron, it's gotta be an older one
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength) -- if connection exists, add on to it
		`, [level]);
	}

	/**
	 * activate neurons at a specified level & distance - age is always zero
	 */
	async activateNeurons(neuronIds, level = 0) {
		if (neuronIds.length === 0) return;
		const activations = neuronIds.map(neuronId => [neuronId, level]);
		await this.conn.query(`INSERT INTO active_neurons (neuron_id, level) VALUES ?`, [activations]);
	}

	/**
	 * detects all spatial levels in age=0 neurons using unified connections - start from base level and go as high as possible
	 */
	async activatePatternNeurons() {
		let level = 0;
		while (true) {
			const patternsFound = await this.processLevel(level);
			if (!patternsFound) break;
			level++;
		}
	}

	/**
	 * processes a level to detect patterns and activate them - returns if patterns found or not
	 */
	async processLevel(level) {
		console.log(`processing level: ${level}`);

		// get all active connections between the newly activated neurons (age=0) and all active neurons in the requested level
		const connections = await this.getActiveConnections(level);

		// get the neuron strengths from the active connections (both ways - from and to)
		const neuronStrengths = this.getNeuronStrengths(connections);

		// get the neighborhood map for each neuron from its connections
		const neighborhoodMap = this.buildNeighborhoodMap(connections);

		// calculate the neighborhood average strength of each neuron. for each neuron, we need to calculate the strengths of its connected neurons.
		const neighborhoodStrengths = this.getNeighborhoodStrengths(neighborhoodMap, neuronStrengths);

		// now get the neuron ids whose strength exceeds its neighborhood - those are the peaks
		const peakNeuronIds = this.getPeakNeurons(neuronStrengths, neighborhoodStrengths);

		// if there are no peaks found in the level, no patterns to process - return false to indicate that we're done
		if (peakNeuronIds.length === 0) return false;

		// get a map from peak neuron ids to the connection ids it uses
		const peakConnections = this.getPeakNeuronsConnections(peakNeuronIds, neighborhoodMap, connections);

		// match peak neurons with their connections to known patterns using their connection ids. get all pattern neuron ids that use the connection ids.
		const peakPatterns = await this.matchPatternNeurons(peakConnections);

		// create new patterns for the peaks that do not have any matching patterns
		await this.createNewPatterns(peakPatterns, peakConnections);

		// now strengthen the connections for each peak's pattern and its active connections that were just observed (increment by 1)
		await this.reinforcePatterns(peakPatterns, peakConnections);

		// activate the observed pattern neurons in the higher level (new or previously existing)
		await this.activateNeurons([...new Set(Array.from(peakPatterns.values()).flat())], level + 1);

		// reinforce connections between active neurons in the higher level
		await this.reinforceConnections(level + 1);

		// return true to indicate that we need to process the next level
		return true;
	}

	/**
	 * returns active directed connections flowing INTO newly activated neurons (age=0) from all active neurons at the specified level for spatio-temporal pooling
	 * connections are directed: older neurons → newer neurons. note that this includes connections between the age=0 neurons as well. that's the spatial pooling.
	 * the others are temporal. so, these connections form the basis of the spatio-temporal pooling. there may be connections between the same neuron when their ages
	 * are different. there may even be indirect connections between the same neuron on different levels (not sure how, but it looks possible).
	 * there should not be any connections from the same neuron to itself within the same age and level.
	 */
	async getActiveConnections(level) {
		const [rows] = await this.conn.query(`
            SELECT c.id, c.from_neuron_id, c.to_neuron_id, c.distance, c.strength
            FROM connections c
            -- source is the older neurons building connections from them to the new neurons
			JOIN active_neurons f ON c.from_neuron_id = f.neuron_id AND c.distance = f.age
            -- getting active connections from older neurons to newly activated neurons within the same level
            JOIN active_neurons t ON c.to_neuron_id = t.neuron_id AND t.age = 0 AND t.level = f.level
            WHERE f.level = ? -- get the active neurons in the given level
            AND (t.neuron_id != f.neuron_id OR f.age != 0) -- if it's the same neuron, it's gotta be an older one
            AND c.strength >= 0 -- ignore negative connections that are scheduled to be deleted 
		`, [level]);
		return rows;
	}

	/**
	 * returns the strength of each neuron from the sum of its active connections strengths.
	 * this includes both incoming connections (to the neuron) and outgoing connections (from the neuron).
	 */
	getNeuronStrengths(connections) {
		const neuronStrengths = new Map();
		for (const connection of connections) {
			const { from_neuron_id, to_neuron_id, strength } = connection;

			// add strength for the source neuron (outgoing connection)
			if (neuronStrengths.has(from_neuron_id)) neuronStrengths.set(from_neuron_id, neuronStrengths.get(from_neuron_id) + strength);
			else neuronStrengths.set(from_neuron_id, strength);

			// add strength for the target neuron (incoming connection)
			if (neuronStrengths.has(to_neuron_id)) neuronStrengths.set(to_neuron_id, neuronStrengths.get(to_neuron_id) + strength);
			else neuronStrengths.set(to_neuron_id, strength);
		}
		return neuronStrengths;
	}

	/**
	 * returns average strengths for each neuron's neighborhood (both incoming and outgoing connections)
	 */
	getNeighborhoodStrengths(neighborhoodMap, neuronStrengths) {
		const neighborhoodAverages = new Map();
		for (const [neuronId, neighbors] of neighborhoodMap) {
			let totalStrength = 0;
			let count = 0;

			for (const neighborId of neighbors) {
				const neighborStrength = neuronStrengths.get(neighborId) || 0;
				totalStrength += neighborStrength;
				count++;
			}

			const averageStrength = count > 0 ? totalStrength / count : 0;
			neighborhoodAverages.set(neuronId, averageStrength);
		}

		return neighborhoodAverages;
	}

	/**
	 * build a map of each neuron and its connected neighbors (bidirectional)
	 */
	buildNeighborhoodMap(connections) {
		const neighborhoodMap = new Map();
		for (const connection of connections) {
			const { from_neuron_id, to_neuron_id } = connection;

			// add to_neuron as neighbor of from_neuron (outgoing connection)
			if (!neighborhoodMap.has(from_neuron_id)) neighborhoodMap.set(from_neuron_id, new Set());
			neighborhoodMap.get(from_neuron_id).add(to_neuron_id);

			// add from_neuron as neighbor of to_neuron (incoming connection)
			if (!neighborhoodMap.has(to_neuron_id)) neighborhoodMap.set(to_neuron_id, new Set());
			neighborhoodMap.get(to_neuron_id).add(from_neuron_id);
		}
		return neighborhoodMap;
	}

	/**
	 * get neurons whose strength exceeds their neighborhood average strength (peaks).
	 * these are neurons that are stronger than their local neighborhood average.
	 */
	getPeakNeurons(neuronStrengths, neighborhoodStrengths) {
		const peaks = [];
		for (const [neuronId, neuronStrength] of neuronStrengths) {
			const neighborhoodStrength = neighborhoodStrengths.get(neuronId) || 0;
			if (neuronStrength > neighborhoodStrength) peaks.push(neuronId);
		}
		return peaks;
	}

	/**
	 * returns a map of peak neurons to their connection IDs. for each peak neuron, returns all connection IDs where the neuron
	 * appears either as source or target using the neighborhoodMap.
	 */
	getPeakNeuronsConnections(peakNeuronIds, neighborhoodMap, connections) {
		const peakConnections = new Map();

		for (const peakNeuronId of peakNeuronIds) {
			const neighbors = neighborhoodMap.get(peakNeuronId);
			if (!neighbors) {
				peakConnections.set(peakNeuronId, []);
				continue;
			}

			const connectionIds = this.getPeakNeuronConnections(peakNeuronId, neighbors, connections);
			peakConnections.set(peakNeuronId, connectionIds);
		}

		return peakConnections;
	}

	/**
	 * returns the connection IDs for a given peak neuron and its neighbors
	 */
	getPeakNeuronConnections(peakNeuronId, neighbors, connections) {
		const connectionIds = [];
		for (const neighborId of neighbors) {

			// find connection from peak to neighbor
			const outgoingConnection = connections.find(c => c.from_neuron_id === peakNeuronId && c.to_neuron_id === neighborId);
			if (outgoingConnection) connectionIds.push(outgoingConnection.id);

			// find connection from neighbor to peak
			const incomingConnection = connections.find(c => c.from_neuron_id === neighborId && c.to_neuron_id === peakNeuronId);
			if (incomingConnection) connectionIds.push(incomingConnection.id);
		}
		return connectionIds;
	}

	/**
	 * matches pattern neurons to peak neurons from the connection ids in their cluster
	 * input format: Map(peak_neuron_id, connection_ids)
	 * output format: Map(peak_neuron_id, pattern_neuron_ids)
	 */
	async matchPatternNeurons(peakConnections) {

		// for each peak, create a separate select query to get matching patterns
		const unionQueries = [];
		for (const [peakNeuronId, connectionIds] of peakConnections) {

			// create a select for this point that returns peak_neuron_id and pattern_neuron_ids
			// note that it is possible for connections to be shared between peaks, but hopefully that will not be too common
			unionQueries.push(`
				SELECT '${peakNeuronId}' AS peak_neuron_id, GROUP_CONCAT(pattern_neuron_id) as pattern_neuron_ids
				FROM patterns
				WHERE connection_id IN (${connectionIds.join(',')})
			`);
		}

		// combine all point queries with UNION and get the matching pattern neurons for each peak neuron
		const sql = unionQueries.join(' UNION ALL ');
		const [rows] = await this.conn.query(sql);

		// convert query results to expected Map format: Map(peak_neuron_id, pattern_neuron_ids)
		const peakPatterns = new Map();
		for (const row of rows) {
			const peakNeuronId = parseInt(row.peak_neuron_id);
			const patternNeuronIds = row.pattern_neuron_ids ? row.pattern_neuron_ids.split(',').map(id => parseInt(id)) : [];
			peakPatterns.set(peakNeuronId, patternNeuronIds);
		}
		return peakPatterns;
	}

	/**
	 * create new pattern neurons for peaks that don't have any matching patterns.
	 * updates the peakPatterns map to include the newly created pattern neurons.
	 * @param {Map} peakPatterns - Map of peak_neuron_id -> array of pattern_neuron_ids
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async createNewPatterns(peakPatterns, peakConnections) {

		// find peaks that need new patterns
		const peaksNeedingPatterns = [];
		for (const [peakNeuronId, connectionIds] of peakConnections) {
			const existingPatterns = peakPatterns.get(peakNeuronId) || [];
			if (existingPatterns.length === 0 && connectionIds.length > 0) peaksNeedingPatterns.push({ peakNeuronId, connectionIds });
		}

		// if all peaks have patterns, nothing to do - no new patterns to create
		if (peaksNeedingPatterns.length === 0) return;

		// bulk create new pattern neurons
		const newPatternNeuronIds = await this.bulkInsertNeurons(peaksNeedingPatterns.length);

		// prepare pattern-connection relationships with strength = 1.0
		const patternConnections = [];
		for (let i = 0; i < peaksNeedingPatterns.length; i++) {
			const { peakNeuronId, connectionIds } = peaksNeedingPatterns[i];
			const patternNeuronId = newPatternNeuronIds[i];

			// add pattern-connection relationships
			for (const connectionId of connectionIds) patternConnections.push([patternNeuronId, connectionId, 1]);

			// update the peakPatterns map with the new pattern neuron id
			peakPatterns.set(peakNeuronId, [patternNeuronId]);
		}

		// batch insert all pattern-connection relationships
		await this.conn.query(`INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES ?`, [patternConnections]);
	}

	/**
	 * reinforce the patterns for each peak by incrementing the strength of their pattern-connection relationships that were just observed.
	 * @param {Map} peakPatterns - Map of peak_neuron_id -> array of pattern_neuron_ids
	 * @param {Map} peakConnections - Map of peak_neuron_id -> array of connection_ids
	 */
	async reinforcePatterns(peakPatterns, peakConnections) {

		// use a Set to deduplicate pattern-connection pairs
		const uniquePairs = new Set();
		for (const [peakNeuronId, patternNeuronIds] of peakPatterns) {
			const connectionIds = peakConnections.get(peakNeuronId) || [];

			// create all combinations of pattern neurons and connections for this peak
			for (const patternNeuronId of patternNeuronIds)
				for (const connectionId of connectionIds)
					uniquePairs.add(`${patternNeuronId}-${connectionId}`);
		}

		// convert unique pairs back to array format for the query
		const patternConnections = Array.from(uniquePairs).map(pair => {
			const [patternNeuronId, connectionId] = pair.split('-').map(id => parseInt(id));
			return [patternNeuronId, connectionId, 1];
		});

		// batch insert/update all pattern-connection relationships
		await this.conn.query(`
        	INSERT INTO patterns (pattern_neuron_id, connection_id, strength) VALUES ?
        	ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)
    	`, [patternConnections]);
	}

	/**
	 * Generate predictions using unified connections table
	 */
	async generatePredictions(feedback) {

		// apply external feedback to previously predicted neurons
		await this.applyFeedback(feedback);

		// age predictions in memory and penalize the predictions that did not happen
		await this.agePredictions();

		// validate predictions against current activations
		await this.validateCurrentPredictions();

		// Get all currently active neurons across all levels
		const [activeNeurons] = await this.conn.query(`
            SELECT DISTINCT neuron_id, level
            FROM active_neurons
            WHERE age = 0
        `);

		if (activeNeurons.length === 0) return;

		const newPredictions = [];

		// For each active neuron, generate predictions using learned connections
		for (const activeNeuron of activeNeurons) {
			const predictions = await this.generateNeuronPredictions(activeNeuron);
			newPredictions.push(...predictions);
		}

		// Bulk insert predictions
		if (newPredictions.length > 0) {
			await this.conn.query(`
                INSERT INTO predicted_neurons 
                (neuron_id, predictor_neuron_id, spatial_level, temporal_level, distance, age, prediction_strength, frame_id)
                VALUES ?
                ON DUPLICATE KEY UPDATE 
                    prediction_strength = prediction_strength + VALUES(prediction_strength),
                    age = 0
            `, [newPredictions]);
		}

		// return current prediction landscape
		return this.getCurrentPredictions();
	}

	/**
	 * Validate current predictions against actual activations
	 */
	async validateCurrentPredictions() {
		// Get predictions that should happen now (distance = 0)
		const [currentPredictions] = await this.conn.query(`
            SELECT DISTINCT pn.neuron_id, pn.predictor_neuron_id, pn.spatial_level, pn.temporal_level
            FROM predicted_neurons pn
            WHERE pn.distance = 0
        `);

		if (currentPredictions.length === 0) return [];

		const feedback = [];

		for (const prediction of currentPredictions) {
			// Check if this neuron is actually active at the predicted level
			const [actualActivation] = await this.conn.query(`
                SELECT neuron_id
                FROM active_neurons
                WHERE neuron_id = ? 
                  AND spatial_level = ? 
                  AND temporal_level = ?
                  AND age = 0
            `, [prediction.neuron_id, prediction.spatial_level, prediction.temporal_level]);

			if (actualActivation.length > 0) {
				feedback.push({ neuron_id: prediction.neuron_id, feedback: 0.1 });
			} else {
				feedback.push({ neuron_id: prediction.neuron_id, feedback: -0.05 });
			}
		}

		// Remove validated predictions
		await this.conn.query(`DELETE FROM predicted_neurons WHERE distance = 0`);

		// apply validation feedback
		await this.applyFeedback(feedback);
	}

	/**
	 * Generate multi-level predictions using clustering and centroid mapping
	 */
	async generateMultiLevelPredictions(maxSpatial, maxTemporal) {
		const predictions = new Map();

		// For each spatio-temporal level, generate distance-based predictions
		const unionQueries = [];

		for (let sLevel = 0; sLevel <= maxSpatial; sLevel++) {
			for (let tLevel = 0; tLevel <= maxTemporal; tLevel++) {
				for (let distance = 1; distance <= this.maxTemporalDistance; distance++) {
					unionQueries.push(`
                        SELECT 
                            ${sLevel} as spatial_level,
                            ${tLevel} as temporal_level, 
                            ${distance} as prediction_distance,
                            target_coords.dimension_id,
                            AVG(target_coords.val) as predicted_val,
                            SUM(tp.strength) as confidence
                        FROM active_neurons current_an
                        JOIN temporal_patterns tp ON current_an.neuron_id = tp.from_neuron_id
                        JOIN coordinates target_coords ON tp.to_neuron_id = target_coords.neuron_id
                        WHERE current_an.spatial_level = ${sLevel}
                          AND current_an.temporal_level = ${tLevel}
                          AND current_an.age = 0
                          AND tp.distance = ${distance}
                          AND tp.strength >= ${this.minPatternStrength}
                        GROUP BY target_coords.dimension_id
                        HAVING confidence >= ${this.minPatternStrength * 2}
                    `);
				}
			}
		}

		if (unionQueries.length > 0) {
			const [allPredictions] = await this.conn.query(unionQueries.join(' UNION ALL '));

			// Group predictions by level and distance
			allPredictions.forEach(pred => {
				const levelKey = `s${pred.spatial_level}_t${pred.temporal_level}`;
				const distanceKey = `d${pred.prediction_distance}`;

				if (!predictions.has(levelKey)) predictions.set(levelKey, new Map());
				if (!predictions.get(levelKey).has(distanceKey)) {
					predictions.get(levelKey).set(distanceKey, new Map());
				}

				predictions.get(levelKey).get(distanceKey).set(pred.dimension_id, {
					predicted_val: pred.predicted_val,
					confidence: pred.confidence
				});
			});
		}

		return Object.fromEntries(predictions);
	}

	/**
	 * apply precise feedback to specific connections
	 */
	async applyFeedback(feedbackList) {

		// if no feedback given, nothing to apply
		if (feedbackList.length === 0) return;

		// create VALUES clause for feedback
		const feedbackValues = feedbackList.map(fb => `(${fb.neuron_id}, ${fb.feedback})`).join(',');

		// single query with VALUES join
		await this.conn.query(`
        	UPDATE connections c
        	JOIN predicted_neurons pn ON c.from_neuron_id = pn.predictor_neuron_id AND c.to_neuron_id = pn.neuron_id
        	JOIN (VALUES ${feedbackValues}) AS feedback(neuron_id, adjustment) ON pn.neuron_id = feedback.neuron_id
        	SET c.strength = c.strength + feedback.adjustment
    	`);
	}

	/**
	 * age predictions in memory and penalize the predictions that did not happen
	 */
	async agePredictions() {

		// age all predictions
		await this.conn.query('UPDATE predicted_neurons SET age = age + 1, distance = distance - 1 WHERE distance > 0');

		// penalize about-to-expire predictions in bulk - these predictions did not come true
		await this.conn.query(`
            UPDATE connections c
            JOIN predicted_neurons pn ON c.from_neuron_id = pn.predictor_neuron_id AND c.to_neuron_id = pn.neuron_id
            SET c.strength = c.strength + ?
            WHERE pn.age >= ?
		`, [this.agedOutPenalty, this.neuronMaxAge]);

		// delete aged-out predictions
		await this.conn.query(`DELETE FROM predicted_neurons WHERE age >= ?`, [this.neuronMaxAge]);
	}

	/**
	 * Generate predictions for a single neuron using unified connections
	 */
	async generateNeuronPredictions(activeNeuron) {
		// Find all learned connections from this neuron (any distance, any target level)
		const [learnedConnections] = await this.conn.query(`
            SELECT to_neuron_id, distance, strength
            FROM connections
            WHERE from_neuron_id = ?
              AND strength >= ?
        `, [activeNeuron.neuron_id, this.minPatternStrength]);

		// Convert to predictions - target level determined by looking up where to_neuron is typically active
		const predictions = [];

		for (const connection of learnedConnections) {
			// Get typical activation level for target neuron
			const [targetLevels] = await this.conn.query(`
                SELECT spatial_level, temporal_level, COUNT(*) as frequency
                FROM active_neurons 
                WHERE neuron_id = ?
                GROUP BY spatial_level, temporal_level
                ORDER BY frequency DESC
                LIMIT 1
            `, [connection.to_neuron_id]);

			if (targetLevels.length > 0) {
				predictions.push([
					connection.to_neuron_id,           // neuron_id
					activeNeuron.neuron_id,            // predictor_neuron_id
					targetLevels[0].spatial_level,     // spatial_level
					targetLevels[0].temporal_level,    // temporal_level
					connection.distance,               // distance
					0,                                 // age
					connection.strength,               // prediction_strength
					this.frameId                       // frame_id
				]);
			}
		}

		return predictions;
	}

	/**
	 * Get current predictions from memory table
	 */
	async getCurrentPredictions() {
		const predictions = {};

		for (let distance = 1; distance <= this.neuronMaxAge; distance++) {
			const [distancePredictions] = await this.conn.query(`
                SELECT 
                    pn.neuron_id,
                    pn.prediction_strength,
                    pn.spatial_level,
                    pn.temporal_level,
                    coords.dimension_id,
                    coords.val,
                    COUNT(*) as predictor_count,
                    SUM(pn.prediction_strength) as total_confidence
                FROM predicted_neurons pn
                JOIN coordinates coords ON pn.neuron_id = coords.neuron_id
                WHERE pn.distance = ?
                GROUP BY pn.neuron_id, coords.dimension_id
                HAVING total_confidence >= ?
                ORDER BY total_confidence DESC
            `, [distance, this.minPatternStrength]);

			if (distancePredictions.length > 0) {
				predictions[`distance_${distance}`] = this.groupPredictionsByDimension(distancePredictions);
			}
		}

		return predictions;
	}

	/**
	 * Group predictions by dimension into centroids
	 */
	groupPredictionsByDimension(predictions) {
		const dimensionGroups = predictions.reduce((groups, pred) => {
			const dimName = this.dimensionIdToName[pred.dimension_id];
			if (!groups[dimName]) groups[dimName] = [];
			groups[dimName].push(pred);
			return groups;
		}, {});

		const clusters = [];

		for (const [dimName, dimPredictions] of Object.entries(dimensionGroups)) {
			const totalWeight = dimPredictions.reduce((sum, p) => sum + p.total_confidence, 0);
			const weightedValue = dimPredictions.reduce((sum, p) => sum + (p.val * p.total_confidence), 0) / totalWeight;

			clusters.push({
				centroid: { [dimName]: weightedValue },
				confidence: totalWeight,
				supporting_predictions: dimPredictions.length,
				levels: dimPredictions.map(p => `(${p.spatial_level},${p.temporal_level})`).join(',')
			});
		}

		return clusters;
	}

	/**
	 * runs the forget cycle, reducing conn weights and deleting unused neurons signifying unseen patterns
	 * also deletes obsolete (negative) connections - very important step that helps the system avoid curse of dimensionality
	 */
	async runForgetCycle() {

		// we run the forget cycle periodically for clean up
		this.forgetCounter++;
		if (this.forgetCounter % this.forgetCycles !== 0) return;
		this.forgetCounter = 0;
		console.log('Running forget cycle...');

		// reduce pattern strengths across the board and remove dead patterns
		await this.conn.query(`UPDATE patterns SET strength = strength - ?`, [this.forgetRate]);
		await this.conn.query(`DELETE FROM patterns WHERE strength <= 0`);

		// reduce connection strengths across the board and remove dead connections
		await this.conn.query(`UPDATE connections SET strength = strength - ?`, [this.forgetRate]);
		await this.conn.query(`DELETE FROM connections WHERE strength <= 0`);

		// remove orphaned neurons with no connections, no patterns, and not currently active
		await this.conn.query(`
	        DELETE n FROM neurons n
    	    LEFT JOIN connections c1 ON n.id = c1.from_neuron_id
        	LEFT JOIN connections c2 ON n.id = c2.to_neuron_id
            LEFT JOIN patterns p ON n.id = p.pattern_neuron_id
        	LEFT JOIN active_neurons an ON n.id = an.neuron_id
        	WHERE c1.from_neuron_id IS NULL 
          	AND c2.to_neuron_id IS NULL
            AND p.pattern_neuron_id IS NULL
          	AND an.neuron_id IS NULL
    	`);

		console.log('Forgetting cycle completed.');
	}

}