import db from './db/db.js';

export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.decayRate = 1000; // run forget cycle every decay rate frames

		this.hp = {
			_alpha_: 1.5, // Peakiness multiplier
			_beta_: 5,   // Minimum weighted degree for a peak
			_min_conn_strength_: 2, // Min observation_count for an ingredient conn
			_match_threshold_: 1.0, // Max distance for pattern matching
			_t_cycles_: 10, // Frames a neuron stays active
			_max_connection_strength_: 100, // Max conn strength
			_learning_rate_coordinates_: 0.1, // How much interneuron coords adapt
		};

		// initialize the forget cycle counter
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
	 * processes one frame of input values - this is an array of tuples { [dimension-name]: <value>}
	 */
	async observeFrame(frame) {
		try {
			console.log(`observing new frame: ${JSON.stringify(frame)}`);

			// do the whole thing as a transaction to avoid inconsistent database states
			await this.conn.beginTransaction();

			// age the neurons in the frame - sliding the window
			await this.ageNeurons();

			// activate the neurons corresponding to the frame in level 0 with age 0
			await this.activateNeurons(frame, 0);

			// recursive Pattern Processing and Higher-Level Activation
			await this.processLevelsRecursively();

			// Phase C: Prediction and Inference
			// await this._makePredictions(connection);

			// TODO: run output neurons activation

			// increment the forget counter and run forget cycle if the limit is reached
			this.forgetCounter++;
			if (this.forgetCounter % this.decayRate === 0) await this.runForgetCycle();
			
			// now commit everything all at once
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
	 * truncates frame memory tables
	 */
	async resetFrameTables() {
		await this.conn.query('TRUNCATE TABLE suppressed_neurons');
		await this.conn.query('TRUNCATE TABLE potential_peaks');
		await this.conn.query('TRUNCATE TABLE observed_pattern_centroids');
		await this.conn.query('TRUNCATE TABLE observed_pattern_ingredients');
		console.log('Frame tables truncated.');
	}

	/**
	 * ages neurons in the context - sliding the window across frames
	 */
	async ageNeurons() {
		await this.conn.query('UPDATE active_neurons SET age = age + 1');
		await this.conn.query(`DELETE FROM active_neurons WHERE age > ?`, [this.hp._t_cycles_]);
		console.log('Active neurons aged and old ones deactivated.');
	}

	/**
	 * activates neurons corresponding to given set of points active for the frame
	 */
	async activateNeurons(frame, level) {
		console.log('activating neurons', level, frame);

		// get the neuron ids to activate for the given coordinates
		const neuronIds = await this.getFrameNeuronIds(frame, level);

		// activate the neurons by inserting to the context with age 0
		const activations = neuronIds.map(neuronId => [neuronId, level, 0]);
		await this.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES ?', [activations]);

		// reinforce connections between active neurons in the current level
		await this.reinforceConnections(level);
	}

	/**
	 * returns the best matching neuron ids for a given set of coordinates and level of resolution
	 */
	async getFrameNeuronIds(frame, level) {

		// we may relax this restriction later, but the inherent nature of levels and resolutions prevents infinite expansion
		if (level > 6) throw new Error('Level cannot exceed 6');

		// resolution is the matching threshold we will use when fetching this neuron
		// at the base level, this is very fine, to the point of making it exact (0)
		// as we go into higher levels, we need to be able to handle partial matches due to the different states of concepts
		// at level 1, the resolution is 10^-7, level 2 is 10^-6, etc.
		const resolution = level === 0 ? 0 : Math.pow(10, level - 8);

		// get all the neurons that have coordinates in close ranges, depending on the resolution
		const candidateNeuronIds = await this.getNeuronsFromFrame(frame, resolution);

		// if there are no neurons found, create neurons to signify current observations
		if (candidateNeuronIds.length === 0) return this.createFrameNeurons(frame);

		// get the best matching neurons among the matches
		return this.getBestMatchingNeuronIds(candidateNeuronIds, frame, resolution);
	}

	/**
	 * returns the best matching neurons to a given set of neurons and coordinates 
	 * For each dimension, finds the minimum distance to any active point in that dimension
	 */
	async getBestMatchingNeuronIds(neuronIds, frame, resolution) {

		// create a map of dimensions to their active values
		const dimensionToValues = new Map();
		
		// collect all values for each dimension from the frame
		for (const coordinates of frame) {
			for (const [dimName, value] of Object.entries(coordinates)) {
				const dimId = this.dimensionNameToId[dimName];
				if (!dimensionToValues.has(dimId)) dimensionToValues.set(dimId, []);
				dimensionToValues.get(dimId).push(value);
			}
		}

		const dimIds = Array.from(dimensionToValues.keys());

		// convert to the format needed for SQL distance calculation
		// for each dimension, find the minimum distance to any of its active values
		// note that we do not have an else condition for the case statement - if we encounter an unknown dimension, it should error out
		const distanceConditions = dimIds.map(dimId => {
			const values = dimensionToValues.get(dimId);
			const valueConditions = values.map(value => `POW(value - ${value}, 2)`).join(', ');
			if (values.length === 1) return `WHEN dimension_id = ${dimId} THEN ${valueConditions}`;
			return `WHEN dimension_id = ${dimId} THEN LEAST(${valueConditions})`;
		}).join(' ');

		// get the neurons with their distances
		const [distanceRows] = await this.conn.query(`
			SELECT neuron_id, SQRT(SUM(CASE ${distanceConditions} END)) AS distance
			FROM coordinates
			WHERE neuron_id IN (${neuronIds.join(',')})
			AND dimension_id IN (${dimIds.join(',')}) -- the dimensions of the neurons that are not seen in the frame are not relevant
			GROUP BY neuron_id
			HAVING distance <= ${resolution}
			ORDER BY distance
			LIMIT ${frame.length} -- select the same number of neurons as the frame points, one corresponding to each, hopefully in the best case
		`);

		// if after calculating the distances, we see none of the neurons is a good fit, create them
		if (distanceRows.length === 0) return this.createFrameNeurons(frame);

		// return all matching neuron ids
		const matchingNeuronIds = distanceRows.map(row => row.neuron_id);
		console.log(`found ${matchingNeuronIds.length} matching neurons (distances: ${distanceRows.map(r => r.distance.toFixed(4)).join(', ')}).`);
		return matchingNeuronIds;
	}

	/**
	 * creates new neurons from a given set of points in the frame
	 */
	async createFrameNeurons(frame) {
		const neuronIds = [];
		for (const coordinates of frame) {
			const neuronId = await this.createNeuron(coordinates);
			neuronIds.push(neuronId);
		}
		return neuronIds;
	}

	/**
	 * creates a new neuron with given coordinates
	 */
	async createNeuron(coordinates) {

		const result = await this.conn.query('INSERT INTO neurons () VALUES ()');
		const newNeuronId = result[0].insertId;
		console.log('newNeuronId', newNeuronId);

		const params = Object.entries(coordinates).map(([dimName, value]) => [newNeuronId, this.dimensionNameToId[dimName], value]);
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, value) VALUES ?', [params]);

		console.log(`created new neuron ${newNeuronId} with coordinates ${JSON.stringify(coordinates)}`);
		return newNeuronId;
	}

	/**
	 * returns the neurons within the range of dimensional values
	 */
	async getNeuronsFromFrame(frame, resolution) {

		// coordinate filters
		const valueRangeFilters = [];
		const valueRangeParams = [];

		// for each point in the frame, get the neurons that may be possible matches
		for (const point of frame)
			for (const [dimName, value] of Object.entries(point)) {
				valueRangeFilters.push(`(dimension_id = ? AND value BETWEEN ? AND ?)`);
				valueRangeParams.push(this.dimensionNameToId[dimName], value - resolution, value + resolution);
			}

		// get the matching neurons to the given values and return them
		const [rows] = await this.conn.query(`
			SELECT DISTINCT neuron_id
			FROM coordinates
			WHERE ${valueRangeFilters.join(' OR ')}
		`, valueRangeParams);
		return rows.map(row => row.neuron_id);
	}

	/**
	 * reinforces the connections between active neurons at a level
	 */
	async reinforceConnections(level) {
		console.log('reinforcing connections in level', level);
		await this.conn.query(`
			INSERT INTO connections (source_id, target_id, strength)
            SELECT s.neuron_id as source_id, t.neuron_id as target_id, 1 / (1 + t.age) as strength -- as the age difference increases, strength decreases
			FROM active_neurons s
			CROSS JOIN active_neurons t
			WHERE s.level = ? -- get the active neurons in the given level
            AND s.age = 0 -- reinforcing connections for the newly activated neurons only 
			AND t.level = s.level -- reinforcing connections only within the same level
			AND (t.neuron_id != s.neuron_id OR t.age != s.age) -- if it's the same neuron, it's gotta be an older one
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength) -- if connection exists, add on to it
		`, [level]);
	}


	async processLevelsRecursively() {
		console.log("Phase B: Starting recursive level processing.");
		let currentProcessingLevel = 0;
		const activeNeurons = await this.conn.query('SELECT COUNT(*) AS count FROM active_neurons WHERE level = 0');
		let newHigherLevelNeuronsActivated = activeNeurons[0][0].count;

		while (newHigherLevelNeuronsActivated > 0) {
			console.log(`\n  Processing Level: ${currentProcessingLevel}`);

			// Clear temporary tables for the current level's pattern processing
			await this.resetFrameTables();

			// B.1: Calculate Peakiness Scores of the current level
			await this.calculatePeakiness(currentProcessingLevel);

			// B.2: Sequential Peak Selection, Ingredient/Centroid, and Suppression
			const patternsObserved = await this._selectPeaksAndFormPatterns(connection);

			if (patternsObserved.length === 0) {
				console.log(`  No patterns observed at Level ${currentProcessingLevel}. Ending recursion.`);
				newHigherLevelNeuronsActivated = 0;
				break;
			}

			// B.3: Match Observed Patterns or Create New Interneurons and activate them
			const activatedNextLevelNeurons = await this._matchOrCreateInterneurons(connection, patternsObserved, currentProcessingLevel);

			// B.4: Reinforce Connections for newly activated neurons
			await this._reinforceLevelConnections(activatedNextLevelNeurons, currentProcessingLevel);

			if (activatedNextLevelNeurons.length === 0) {
				console.log(`  No new higher-level neurons activated from Level ${currentProcessingLevel}. Ending recursion.`);
				newHigherLevelNeuronsActivated = 0;
				break;
			}

			currentProcessingLevel++;
			newHigherLevelNeuronsActivated = activatedNextLevelNeurons.length; // Count new activations for next loop
		}
		console.log("Phase B: Recursive processing complete.");
	}

	async calculatePeakiness(level) {
		console.log(`calculating peakiness for active neurons at level ${level}`);

		// fetch all active neurons at the current level with their relevant connections to other active neurons.
		const [activeNeuronsAndConnections] = await this.conn.query(`
            SELECT s.neuron_id AS source_id, s.level AS source_level, c.target_id, c.strength
            FROM active_neurons s
            JOIN connections c ON s.neuron_id = c.source_id
            JOIN active_neurons t ON c.target_id = t.neuron_id
            WHERE s.level = ?
        `, [level]);

		// In-memory calculation of weighted_degree and neighbor_avg_wd
		const neuronWeightedDegrees = new Map(); // Map<neuronId, weightedDegree>
		const neuronActiveNeighbors = new Map(); // Map<neuronId, Set<neighborId>>

		activeNeuronsAndConnections.forEach(row => {
			const { source_id, target_id, observation_count } = row;

			// Calculate weighted_degree
			neuronWeightedDegrees.set(source_id, (neuronWeightedDegrees.get(source_id) || 0) + observation_count);

			// Store active neighbors for avg_neighbor_wd calculation
			if (!neuronActiveNeighbors.has(source_id)) {
				neuronActiveNeighbors.set(source_id, new Set());
			}
			neuronActiveNeighbors.get(source_id).add(target_id);
		});

		const peakinessScores = [];
		for (const [neuron_id, weighted_degree] of neuronWeightedDegrees.entries()) {
			let sumNeighborWeightedDegree = 0;
			let numActiveNeighbors = 0;

			const neighbors = neuronActiveNeighbors.get(neuron_id);
			if (neighbors) {
				for (const neighbor_id of neighbors) {
					if (neuronWeightedDegrees.has(neighbor_id)) { // Ensure neighbor itself has an active weighted_degree
						sumNeighborWeightedDegree += neuronWeightedDegrees.get(neighbor_id);
						numActiveNeighbors++;
					}
				}
			}

			const avg_neighbor_wd = numActiveNeighbors > 0 ? sumNeighborWeightedDegree / numActiveNeighbors : null;

			let peakiness_score = 0;
			if (weighted_degree >= this.hp._beta_) { // Must meet minimum influence
				if (avg_neighbor_wd === null || weighted_degree > this.hp._alpha_ * avg_neighbor_wd) {
					peakiness_score = weighted_degree - (this.hp._alpha_ * (avg_neighbor_wd || 0)); // If no avg_neighbor_wd, it implies high peakiness
				}
			}
			peakinessScores.push([neuron_id, peakiness_score]);
		}

		// Batch insert peakiness scores
		if (peakinessScores.length > 0) {
			await this.conn.query('INSERT INTO potential_peaks (neuron_id, peakiness_score) VALUES ?', [peakinessScores]);
			console.log(`  Calculated and inserted ${peakinessScores.length} potential peaks.`);
		}
		else console.log("  No potential peaks found for calculation.");
	}

	async _selectPeaksAndFormPatterns(connection) {
		console.log("  B.2: Selecting peaks and forming patterns.");
		const patternsObserved = []; // To return for B.3
		let patternIdCounter = (await connection.query('SELECT COALESCE(MAX(pattern_id), 0) AS max_id FROM observed_pattern_centroids'))[0][0].max_id;

		// Fetch peaks, ordered by score, not yet suppressed
		const [peaks] = await connection.query(`
            SELECT pp.neuron_id, pp.peakiness_score
            FROM potential_peaks pp
            LEFT JOIN suppressed_neurons sn ON pp.neuron_id = sn.neuron_id
            WHERE pp.peakiness_score > 0 AND sn.neuron_id IS NULL
            ORDER BY pp.peakiness_score DESC
        `);

		for (const peak of peaks) {
			const currentPeakId = peak.neuron_id;

			// Re-check suppression in case another peak in this batch suppressed it
			const [isSuppressed] = await connection.query('SELECT 1 FROM suppressed_neurons WHERE neuron_id = ?', [currentPeakId]);
			if (isSuppressed.length > 0) {
				continue;
			}

			patternIdCounter++;

			// B.2.a: Find Ingredient Neurons for the current peak.
			const [ingredients] = await connection.query(`
                SELECT c.target_id AS ingredient_id
                FROM connections c
                INNER JOIN active_neurons an_target ON c.target_id = an_target.neuron_id
                WHERE c.source_id = ? AND c.observation_count > ?
            `, [currentPeakId, this.hp._min_conn_strength_]);

			if (ingredients.length === 0) {
				console.log(`    Peak ${currentPeakId} has no significant active ingredients. Skipping.`);
				continue;
			}

			const ingredientIds = ingredients.map(i => i.ingredient_id);
			const ingredientInserts = ingredientIds.map(id => [patternIdCounter, id]);
			await connection.query('INSERT INTO observed_pattern_ingredients (pattern_id, ingredient_id) VALUES ?', [ingredientInserts]);

			// B.2.b: Calculate the Centroid of these active ingredients.
			const [centroidCoords] = await connection.query(`
                SELECT nc.dimension_id, AVG(nc.value) AS centroid_value
                FROM neuron_coordinates nc
                WHERE nc.neuron_id IN (?)
                GROUP BY nc.dimension_id
            `, [ingredientIds]);

			const centroidInserts = centroidCoords.map(c => [patternIdCounter, c.dimension_id, c.centroid_value]);
			await connection.query('INSERT INTO observed_pattern_centroids (pattern_id, dimension_id, value) VALUES ?', [centroidInserts]);

			// Store for later use
			patternsObserved.push({
				pattern_id: patternIdCounter,
				centroid: centroidCoords,
				ingredients: ingredientIds
			});

			// B.2.c: Suppress the selected peak and its direct, strong, active neighbors.
			const suppressedNeuronIds = [currentPeakId, ...ingredientIds];
			const suppressedInserts = suppressedNeuronIds.map(id => [id]);
			await connection.query('INSERT IGNORE INTO suppressed_neurons (neuron_id) VALUES ?', [suppressedInserts]);

			console.log(`    Pattern ${patternIdCounter} (from peak ${currentPeakId}) identified with ${ingredients.length} ingredients.`);
		}
		console.log(`  Found ${patternsObserved.length} unique patterns at this level.`);
		return patternsObserved;
	}

	async _matchOrCreateInterneurons(connection, patternsObserved, currentProcessingLevel) {
		console.log(`  B.3: Matching/Creating interneurons for patterns at level ${currentProcessingLevel + 1}.`);
		const activatedNextLevelNeurons = [];

		for (const pattern of patternsObserved) {
			const observedPatternId = pattern.pattern_id;
			const observedCentroid = pattern.centroid; // [{dimension_id, value}, ...]

			if (observedCentroid.length === 0) {
				console.warn(`    Pattern ${observedPatternId} has no centroid coordinates. Skipping match/create.`);
				continue;
			}

			// Try to find a matching existing interneuron
			const dimValuePairs = observedCentroid.map(c => `(nc_inter.dimension_id = ${c.dimension_id} AND nc_inter.value = ${c.value})`).join(' OR '); // This won't work for Euclidean distance
			// Euclidean distance matching needs a more complex query or in-app calculation.
			// Let's fetch all relevant next-level interneurons and calculate distances in Node.js.

			const [nextLevelInterneuronsRaw] = await connection.query(`
                SELECT n.id AS neuron_id, nc.dimension_id, nc.value
                FROM neurons n
                INNER JOIN neuron_coordinates nc ON n.id = nc.neuron_id
                WHERE n.level = ?;
            `, [currentProcessingLevel + 1]);

			const nextLevelInterneurons = new Map(); // Map<neuron_id, Map<dimension_id, value>>
			nextLevelInterneuronsRaw.forEach(row => {
				if (!nextLevelInterneurons.has(row.neuron_id)) {
					nextLevelInterneurons.set(row.neuron_id, new Map());
				}
				nextLevelInterneurons.get(row.neuron_id).set(row.dimension_id, row.value);
			});

			let bestMatchId = null;
			let minDistance = Infinity;

			for (const [interneuronId, interneuronCoords] of nextLevelInterneurons.entries()) {
				let sumSqDiff = 0;
				let hasAllDims = true; // Check if interneuron has all dimensions of observed pattern

				for (const obsCoord of observedCentroid) {
					if (interneuronCoords.has(obsCoord.dimension_id)) {
						const interCoordValue = interneuronCoords.get(obsCoord.dimension_id);
						sumSqDiff += Math.pow(interCoordValue - obsCoord.value, 2);
					} else {
						hasAllDims = false; // Interneuron doesn't cover all dimensions of the pattern
						break;
					}
				}

				if (hasAllDims) {
					const distance = Math.sqrt(sumSqDiff);
					if (distance < minDistance) {
						minDistance = distance;
						bestMatchId = interneuronId;
					}
				}
			}

			let activatedNeuronId;
			if (bestMatchId !== null && minDistance < this.hp._match_threshold_) {
				// B.3.a: Handle Matched Pattern
				activatedNeuronId = bestMatchId;
				console.log(`    Pattern ${observedPatternId} matched existing interneuron ${activatedNeuronId} (Distance: ${minDistance.toFixed(2)}).`);

				// Update coordinates of the matched interneuron (adaptive learning)
				const coordUpdates = observedCentroid.map(obsCoord => `
                    UPDATE neuron_coordinates
                    SET value = value * (1 - ${this.hp._learning_rate_coordinates_}) + ${obsCoord.value} * ${this.hp._learning_rate_coordinates_}
                    WHERE neuron_id = ${activatedNeuronId} AND dimension_id = ${obsCoord.dimension_id}
                `).join(';');
				// Need to ensure all dimensions exist for the interneuron first before updating.
				// This makes a batch update more complex or requires individual updates.
				// For now, let's assume they have the same set of relevant dimensions after creation.
				if (coordUpdates.length > 0) {
					await connection.query(coordUpdates); // Execute multiple updates
				}

			} else {
				// B.3.b: Create New Interneuron for Novel Pattern
				const newNeuronId = await this._getNextNeuronId();
				const newNeuronLevel = currentProcessingLevel + 1;

				await connection.query('INSERT INTO neurons (id, creation_time) VALUES (?, NOW())', [newNeuronId, newNeuronLevel]);

				const newCoordInserts = observedCentroid.map(c => [newNeuronId, c.dimension_id, c.value]);
				if (newCoordInserts.length > 0) {
					await connection.query('INSERT INTO coordinates (neuron_id, dimension_id, value) VALUES ?', [newCoordInserts]);
				}
				activatedNeuronId = newNeuronId;
				console.log(`    Pattern ${observedPatternId} created new interneuron ${activatedNeuronId} at level ${newNeuronLevel}.`);
			}

			// Activate the matched/new interneuron
			await connection.query(
				'INSERT INTO active_neurons (neuron_id, level, age) VALUES (?, ?, 0) ON DUPLICATE KEY UPDATE age = 0',
				[activatedNeuronId, currentProcessingLevel + 1]
			);
			activatedNextLevelNeurons.push(activatedNeuronId);
		}
		return activatedNextLevelNeurons;
	}

	async _getNextNeuronId() {
		const [rows] = await this.conn.query('SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM neurons');
		return rows[0].next_id;
	}

	async _reinforceLevelConnections(connection, activatedNextLevelNeurons, currentProcessingLevel) {
		console.log(`  B.4: Reinforcing connections from Level ${currentProcessingLevel} to Level ${currentProcessingLevel + 1}.`);
		if (activatedNextLevelNeurons.length === 0) {
			console.log("    No next-level neurons to connect to.");
			return;
		}

		const connectionsToReinforce = []; // [source_id, target_id] pairs
		const nextLevelNeuronIds = activatedNextLevelNeurons;

		// Get all patterns that led to these activatedNextLevelNeurons
		const [patternsInfo] = await connection.query(`
            SELECT
                opc.pattern_id,
                opc.dimension_id,
                opc.value AS centroid_value,
                opi.ingredient_id
            FROM observed_pattern_centroids opc
            INNER JOIN observed_pattern_ingredients opi ON opc.pattern_id = opi.pattern_id
        `);

		const patternsMap = new Map(); // Map<pattern_id, {centroid: [], ingredients: []}>
		patternsInfo.forEach(row => {
			if (!patternsMap.has(row.pattern_id)) {
				patternsMap.set(row.pattern_id, { centroid: [], ingredients: [] });
			}
			patternsMap.get(row.pattern_id).centroid.push({ dimension_id: row.dimension_id, value: row.centroid_value });
			if (!patternsMap.get(row.pattern_id).ingredients.includes(row.ingredient_id)) {
				patternsMap.get(row.pattern_id).ingredients.push(row.ingredient_id);
			}
		});

		// Loop through each newly activated next-level neuron and find its corresponding ingredients
		for (const nextLevelNeuronId of nextLevelNeuronIds) {
			// Find which pattern(s) this nextLevelNeuronId represents.
			// This needs to be more robust. If it was a match, it matched a pattern_id.
			// If it was new, its ID corresponds to one of the newly created patterns.
			// For simplicity, let's assume one-to-one mapping for patternsObserved and activatedNextLevelNeurons here.
			// In a real system, you'd track the pattern_id -> new_neuron_id mapping.

			// A more direct way is to fetch the ingredients linked to this higher-level neuron.
			// This implies the connections were already made (e.g. B.3). This is specifically REINFORCEMENT.
			// So, we need: active ingredients -> newly activated higher-level neuron.

			// Get the pattern_id corresponding to this `nextLevelNeuronId`
			// This is a complex join, simplifying: fetch ingredients for patterns matching these nextLevelNeuronIds.
			const [matchedPatternsData] = await connection.query(`
                SELECT
                    opi.ingredient_id,
                    an.neuron_id AS next_level_neuron_id
                FROM observed_pattern_ingredients opi
                INNER JOIN observed_pattern_centroids opc ON opi.pattern_id = opc.pattern_id
                INNER JOIN pattern_matches pm ON opc.pattern_id = pm.pattern_id
                INNER JOIN active_neurons an ON (pm.is_match = 1 AND an.neuron_id = pm.interneuron_id) OR
                                                (pm.is_match = 0 AND an.neuron_id = (SELECT n_new.id FROM neurons n_new
                                                                                        WHERE n_new.level = ? AND n_new.creation_time = (
                                                                                            SELECT MAX(creation_time) FROM neurons
                                                                                            WHERE level = ? AND id <= ? AND id >= (SELECT COALESCE(MAX(id),0) + 1 FROM neurons) - COUNT(*) FROM pattern_matches WHERE is_match = 0)
                                                                                        LIMIT 1
                                                                                        )))
                WHERE an.neuron_id IN (?)
            `, [currentProcessingLevel + 1, currentProcessingLevel + 1, nextLevelNeuronId, nextLevelNeuronId]); // Placeholder needs more thought for new neurons

			// Simpler approach for demo: Just reinforce all observed ingredients to all activated next-level neurons for those patterns.
			// This assumes the patternsObserved from _selectPeaksAndFormPatterns are in order and map directly.
			// A more robust system would store pattern_id -> activated_neuron_id mapping explicitly.

			// For each pattern that was processed and resulted in a next-level activation:
			const patternsThatActivated = patternsObserved.filter(p => nextLevelNeuronIds.includes(
				// Find the neuron ID that this pattern resulted in (this mapping is implicit currently)
				// This is the trickiest part for Node.js - reliably linking the pattern_id from B2 to the activated_neuron_id from B3
				// We'd need to return activatedNeuronId along with its pattern_id from _matchOrCreateInterneurons
				true // Placeholder
			));

			for (const pattern of patternsObserved) { // Iterating all patterns found in B2
				const correspondingActivatedNeuron = activatedNextLevelNeurons.find(neuronId => {
					// This logic is hard without a direct pattern_id to neuron_id map from B3.
					// For now, let's simplify by linking all ingredients to all *newly activated* neurons from B3.
					// A better approach would be to pass a map from B3: patternId -> activatedNeuronId.
					return true; // Simplified for demo.
				});

				if (!correspondingActivatedNeuron) continue;

				for (const ingredientId of pattern.ingredients) {
					connectionsToReinforce.push([ingredientId, correspondingActivatedNeuron]); // Ingredient -> Pattern
					connectionsToReinforce.push([correspondingActivatedNeuron, ingredientId]); // Pattern -> Ingredient
				}
			}
		}


		// Re-implementing the batch INSERT ... ON DUPLICATE KEY UPDATE with fetched data.
		if (connectionsToReinforce.length > 0) {
			const updates = [];
			const inserts = [];

			// Fetch existing counts for these potential connections
			const connectionPairs = connectionsToReinforce.map(pair => `(${pair[0]}, ${pair[1]})`).join(',');
			const [existingConnections] = await connection.query(`
                SELECT source_id, target_id, observation_count
                FROM connections
                WHERE (source_id, target_id) IN (${connectionPairs})
            `);

			const existingMap = new Map();
			existingConnections.forEach(conn => existingMap.set(`${conn.source_id}-${conn.target_id}`, conn.observation_count));

			for (const [src, tgt] of connectionsToReinforce) {
				const key = `${src}-${tgt}`;
				if (existingMap.has(key)) {
					const currentCount = existingMap.get(key);
					updates.push(`WHEN source_id = ${src} AND target_id = ${tgt} THEN LEAST(${this.hp._max_connection_strength_}, ${currentCount} + 1)`);
				} else {
					inserts.push(`(${src}, ${tgt}, ${LEAST(this.hp._max_connection_strength_, 1)})`);
				}
			}

			if (updates.length > 0) {
				await connection.query(`
                    UPDATE connections
                    SET observation_count = CASE
                        ${updates.join('\n')}
                        ELSE observation_count
                    END
                    WHERE (source_id, target_id) IN (${connectionPairs})
                `);
			}
			if (inserts.length > 0) {
				await connection.query(`
                    INSERT INTO connections (source_id, target_id, observation_count) VALUES ${inserts.join(',')}
                `);
			}
			console.log(`  Reinforced ${connectionsToReinforce.length / 2} level connections.`);
		}
	}


	async _makePredictions(connection) {
		console.log("Phase C: Making predictions.");

		const [highestLevelRow] = await connection.query('SELECT COALESCE(MAX(level), 0) AS max_level FROM active_neurons');
		const maxLevel = highestLevelRow[0].max_level;

		if (maxLevel === 0) {
			console.log("  No higher-level neurons active for prediction.");
			return [];
		}

		const [predictedCoords] = await connection.query(`
            SELECT
                nc.dimension_id,
                AVG(nc.value) AS predicted_value
            FROM
                active_neurons hl_an
            INNER JOIN
                connections c ON hl_an.neuron_id = c.source_id
            INNER JOIN
                neuron_coordinates nc ON c.target_id = nc.neuron_id
            WHERE
                hl_an.level = ? AND c.observation_count > ?
            GROUP BY
                nc.dimension_id;
        `, [maxLevel, this.hp._min_conn_strength_]);

		console.log("  Predicted pattern coordinates:", predictedCoords.map(c => `${this.dimensionIdToName[c.dimension_id]}:${c.predicted_value.toFixed(2)}`).join(', '));
		return predictedCoords;
	}

	/**
	 * runs the forget cycle, reducing conn weights and deleting unused neurons signifying unseen patterns
	 */
	async runForgetCycle() {
		console.log('Running forget cycle...');

		// reduce conn strengths across the board
		await this.conn.query(`UPDATE connections SET strength = strength - 1`);

		// now delete the connections that have gone to zero
		// TODO: get the neurons that have these connections (source and target) before deletion, then delete them if they have no connections left
		await this.conn.query('DELETE FROM connections WHERE strength <= 0');

		console.log("Forgetting cycle completed.");
	}
}