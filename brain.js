import db from './db/db.js';

export default class Brain {

	/**
	 * returns new brain instance
	 */
	constructor() {

		// set hyperparameters
		this.neuronMaxAge = 10; // Frames a neuron stays active
		this.decayRate = 1000; // run forget cycle every decay rate frames
		this.adaptationSpeed = 0.1; // how much neuron coordinates adapt towards observations
		this.maxLevel = 6; // maximum number of levels we will process patterns recursively
		this.maxResolution = 2; // at the highest level, this is the resolution used to match concepts (10^-maxResolution)
		this.peakMinStrength = 3; // minimum connection strength for a peak (pattern needs to observed at least this many times)
		this.peakMinRatio = 1.5; // minimum ratio required to exceed the neighborhood strength by to form a peak

		this.hp = {
			_min_conn_strength_: 2, // Min observation_count for an ingredient conn
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

			// activate the neurons corresponding to the frame in level 0 with age 0 (base neurons)
			await this.activateFrameNeurons(frame, 0);

			// now activate interneurons recursively (age = 0, level > 0)
			await this.activeInterneurons();

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
		await this.conn.query('TRUNCATE TABLE potential_peaks');
		await this.conn.query('TRUNCATE TABLE suppressed_neurons');
		await this.conn.query('TRUNCATE TABLE observed_pattern_centroids');
		await this.conn.query('TRUNCATE TABLE observed_pattern_ingredients');
		console.log('Frame tables truncated.');
	}

	/**
	 * ages neurons in the context - sliding the window across frames
	 */
	async ageNeurons() {
		await this.conn.query('UPDATE active_neurons SET age = age + 1');
		await this.conn.query(`DELETE FROM active_neurons WHERE age > ?`, [this.neuronMaxAge]);
		console.log('Active neurons aged and old ones deactivated.');
	}

	/**
	 * activates neurons corresponding to given set of points active for the frame
	 */
	async activateFrameNeurons(frame, level) {
		console.log('activating neurons', level, frame);

		// get the neurons to activate for the given coordinates as { point_str, neuron_id }
		const matches = await this.getFrameNeurons(frame, level);

		// activate the neurons of the frame
		await this.activateNeurons(matches, level);

		// reinforce connections between active neurons in the current level
		await this.reinforceConnections(level);
	}

	/**
	 * activates a given set of neurons at a given level with age 0 - matches is an array of { point_str, neuron_id }
	 */
	async activateNeurons(matches, level) {

		// derive unique neuron ids and activate with age 0
		const uniqueNeuronIds = Array.from(new Set((matches || []).map(m => m.neuron_id)));
		const activations = uniqueNeuronIds.map(neuronId => [neuronId, level, 0]);
		console.log('activations', activations);
		await this.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES ?', [activations]);

		// adapt coordinates of activated neurons towards their observed points
		await this.applyCoordinateAdaptation(matches);
	}

	/**
	 * returns the best matching neurons for given frame and level as [{ point_str, neuron_id }]
	 */
	async getFrameNeurons(frame, level) {

		// we may relax this restriction later, but the inherent nature of levels and resolutions prevents infinite expansion
		if (level > this.maxLevel) throw new Error(`Level cannot exceed ${this.maxLevel}.`);

		// resolution is the matching threshold we will use when fetching the neurons
		// at level 0, it's very fine, to the point of making it almost exact (10^-8)
		// at level 1, the resolution is 10^-7, level 2 is 10^-6, etc. level 6 would be 10^-2.
		// as we go into higher levels, we need to be able to handle partial matches due to the different states of concepts
		// that's why we increase the resolution to cast a wider net to group higher level concepts
		const resolution = Math.pow(10, level - (this.maxResolution + this.maxLevel));

		// get all the neurons that have coordinates in close ranges for each point
		const pointNeuronMatches = await this.getNeuronsFromFrame(frame, resolution);
		console.log('pointNeuronMatches', pointNeuronMatches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const matches = [];

		// create neurons for points with no matches
		const pointsWithoutMatches = pointNeuronMatches.filter(p => p.neuron_ids.length === 0);
		if (pointsWithoutMatches.length > 0) {
			const newNeurons = await this.createFrameNeurons(pointsWithoutMatches.map(p => p.point_str));
			console.log('newNeurons', newNeurons);
			matches.push(...newNeurons);
		}

		// if there are no points with matches, we are done
		const pointsWithMatches = pointNeuronMatches.filter(p => p.neuron_ids.length > 0);
		console.log('pointsWithMatches', pointsWithMatches);
		if (pointsWithMatches.length === 0) return matches;

		// find the best matching neurons for each point
		const bestMatchingResults = await this.getBestMatchingNeuronIds(pointsWithMatches, resolution);
		console.log('bestMatchingResults', bestMatchingResults);

		// add successful matches to the results
		matches.push(...bestMatchingResults.filter(result => result.neuron_id));

		// identify points that need new neurons (had candidates but none close enough)
		const pointsNeedingNeurons = bestMatchingResults.filter(r => !r.neuron_id).map(r => r.point_str);
		if (pointsNeedingNeurons.length > 0) {
			console.log(`${pointsNeedingNeurons.length} points had potential matches but none were close enough. Creating neurons for them.`);
			const additionalNeurons = await this.createFrameNeurons(pointsNeedingNeurons);
			matches.push(...additionalNeurons);
		}

		// return matches: [{ point_str, neuron_id }]
		return matches;
	}

	/**
	 * Applies coordinate adaptation for matched existing neurons, moving their coordinates
	 * towards the newly observed matched point values using learning rate.
	 * matches: Array<{ neuron_id: number, point: Record<dimName,string|number> }>
	 */
	async applyCoordinateAdaptation(matches) {

		// Purpose: Across one frame, multiple observed points can match the same neuron.
		// We aggregate observations per (neuron_id, dimension_id) and use the mean to
		// produce a single deterministic update per pair:
		//   new = old * (1 - lr) + observed_mean * lr
		// This avoids order-dependent multiple updates and stabilizes learning.

		// Expand matches into a flat list of valid (neuronId, dimId, value) entries
		const observedEntries = matches.flatMap(m => 
			Object.entries(JSON.parse(m.point_str)).map(([dimName, val]) => 
				({ neuronId: m.neuron_id, dimId: this.dimensionNameToId[dimName], val })));

		// Reduce to an aggregate map keyed by `${neuronId}:${dimId}` with sum and count
		const aggregate = observedEntries.reduce((map, { neuronId, dimId, val }) => ({
			...map,
			[`${neuronId}:${dimId}`]: {
				neuronId,
				dimId,
				sum: (map[`${neuronId}:${dimId}`]?.sum || 0) + val,
				count: (map[`${neuronId}:${dimId}`]?.count || 0) + 1,
			}
		}), {});

		// Now update the neuron coordinates for adaptation - use a single UPDATE with CASE expression
		const lr = this.adaptationSpeed; // hyperparameter to control the speed of adaptation
		const cases = [];
		const wherePairs = [];
		for (const { neuronId, dimId, sum, count } of Object.values(aggregate)) {
			const observedMean = sum / count;
			cases.push(`WHEN neuron_id = ${neuronId} AND dimension_id = ${dimId} THEN val * (1 - ${lr}) + ${observedMean} * ${lr}`);
			wherePairs.push(`(${neuronId}, ${dimId})`);
		}
		const sql = `UPDATE coordinates
			SET val = CASE ${cases.join(' ')} ELSE val END
			WHERE (neuron_id, dimension_id) IN (${wherePairs.join(',')})`;
		await this.conn.query(sql);
	}

	/**
	 * returns the best matching neurons for each point with their matching neurons array
	 * For each point, finds the closest neuron among its candidates
	 */
	async getBestMatchingNeuronIds(pointNeuronMatches, resolution) {
		const unionQueries = [];

		// for each point with its matching neurons, calculate distances
		for (let i = 0; i < pointNeuronMatches.length; i++) {
			const { point_str, neuron_ids } = pointNeuronMatches[i];
			
			// if no matching neurons for this point, skip it (will be handled later)
			if (neuron_ids.length === 0) continue;

			// create a map of dimensions to their values for this point
			const dimensionToValues = new Map();
			for (const [dimName, value] of Object.entries(JSON.parse(point_str))) {
				const dimId = this.dimensionNameToId[dimName];
				if (!dimensionToValues.has(dimId)) dimensionToValues.set(dimId, []);
				dimensionToValues.get(dimId).push(value);
			}

			// TODO: instead of filtering out the unobserved dimensions of matching neurons, they should increase
			//  the distance slightly, but not too much - hard to find the right amount - good enough for now
			const dimIds = Array.from(dimensionToValues.keys());

			// convert to the format needed for SQL distance calculation
			// for each dimension, calculate the distance to its value
			const distanceConditions = dimIds.map(dimId => {
				const value = dimensionToValues.get(dimId)[0]; // each dimension has exactly one value for a point
				return `WHEN dimension_id = ${dimId} THEN POW(val - ${value}, 2)`;
			}).join(' ');

			// create a select for this point that calculates distances to its matching neurons
			const pointQuery = `
				SELECT '${point_str}' as point_str, (
					SELECT CONCAT(neuron_id, '|', distance) as neuron_distance
					FROM (
						SELECT neuron_id, SQRT(SUM(CASE ${distanceConditions} END)) AS distance
						FROM coordinates
						WHERE neuron_id IN (${neuron_ids.join(',')})
						AND dimension_id IN (${dimIds.join(',')})
						GROUP BY neuron_id
						HAVING distance <= ${resolution}
						ORDER BY distance
						LIMIT 1
					) q
				) as neuron_distance
			`;

			unionQueries.push(pointQuery);
		}

		// if no queries to run, return empty array
		if (unionQueries.length === 0) return [];

		// combine all point queries with UNION
		const fullQuery = unionQueries.join(' UNION ALL ');

		// get the closest neurons for each point
		const [distanceRows] = await this.conn.query(fullQuery);
		console.log('neuron matches with distances', distanceRows, fullQuery);
		
		// return the closest neuron ids
		return distanceRows.map(r => ({ point_str: r.point_str, neuron_id: r.neuron_distance ? r.neuron_distance.split('|')[0] : null }));
	}

	/**
	 * creates new neurons from a given set of points in the frame
	 * TODO: this should do bulk insert, instead of inserting one neuron at a time - good enough for now
	 */
	async createFrameNeurons(points) {
		const created = [];
		for (const pointStr of points) {
			const neuronId = await this.createNeuron(JSON.parse(pointStr));
			created.push({ point_str: pointStr, neuron_id: neuronId });
		}
		return created;
	}

	/**
	 * creates a new neuron with given coordinates
	 */
	async createNeuron(coordinates) {

		const result = await this.conn.query('INSERT INTO neurons () VALUES ()');
		const newNeuronId = result[0].insertId;
		console.log('newNeuronId', newNeuronId);

		const params = Object.entries(coordinates).map(([dimName, value]) => [newNeuronId, this.dimensionNameToId[dimName], value]);
		await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [params]);

		console.log(`created new neuron ${newNeuronId} with coordinates ${JSON.stringify(coordinates)}`);
		return newNeuronId;
	}

	/**
	 * returns the neurons within the range of dimensional values for each point
	 * Returns an array of objects with point_str and neuron_ids for each point
	 */
	async getNeuronsFromFrame(frame, resolution) {
		const unionQueries = [];

		// for each point in the frame, create a separate select query
		for (const point of frame) {

			// coordinate filters for this specific point
			const valueRangeFilters = [];
			for (const [dimName, value] of Object.entries(point))
				valueRangeFilters.push(`(dimension_id = ${this.dimensionNameToId[dimName]} AND val BETWEEN ${value - resolution} AND ${value + resolution})`);

			// create a select for this point that returns point_str and neuron_ids
			const pointQuery = `
				SELECT '${JSON.stringify(point)}' AS point_str, COALESCE(JSON_ARRAYAGG(neuron_id), JSON_ARRAY()) AS neuron_ids
				FROM (
					SELECT DISTINCT neuron_id
					FROM coordinates
					WHERE ${valueRangeFilters.join(' OR ')}
				) AS distinct_neurons
			`;

			unionQueries.push(pointQuery);
		}

		// combine all point queries with UNION and get the matching neurons for each point
		const [rows] = await this.conn.query(unionQueries.join(' UNION ALL '));
		
		// return results
		return rows;
	}

	/**
	 * reinforces the connections between newly active neurons (age = 0) at a level
	 * note that the neuron connections are directionless. when we query the connections a neuron has, we don't care about direction.
	 * that means all connections should be inserted in the same way for a given pair, regardless of the direction.
	 */
	async reinforceConnections(level) {
		await this.conn.query(`
			INSERT INTO connections (neuron1_id, neuron2_id, strength)
            SELECT 
                LEAST(s.neuron_id, t.neuron_id) as neuron1_id, -- source is always the smaller neuron id
                GREATEST(s.neuron_id, t.neuron_id) as neuron2_id, -- target is always the bigger neuron id
                -- it's possible for a neuron to connect to an older target and that target to be also age 0 and connecting to an older source 
                -- we sum up in that case - this should not be a problem for neurons connecting to earlier activations of themselves - will not double count
                SUM(1 / (1 + t.age)) as strength -- as the age difference increases, strength decreases 
			FROM active_neurons s
			CROSS JOIN active_neurons t
			WHERE s.level = ? -- get the active neurons in the given level
            AND s.age = 0 -- reinforcing connections for the newly activated neurons only 
			AND t.level = s.level -- reinforcing connections only within the same level
			AND (t.neuron_id != s.neuron_id OR t.age != s.age) -- if it's the same neuron, it's gotta be an older one
			GROUP BY neuron1_id, neuron2_id
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength) -- if connection exists, add on to it
		`, [level]);
	}

	/**
	 * activates interneurons recursively based on currently observed patterns in level 0 (age = 0)
	 */
	async activeInterneurons() {
		console.log('activating interneurons...');

		// start with level 0 and loop until there are no neurons activated
		let level = 0;
		let newHigherLevelNeuronsActivated = await this.getActiveNeuronCount(level);
		while (newHigherLevelNeuronsActivated > 0) {
			console.log(`processing level: ${level}`);

			// Clear temporary tables for the current level's pattern processing
			await this.resetFrameTables();

			// calculate peakiness scores of the current level
			await this.calculatePeakiness(level);

			// select top peaks and get pattern centroids
			const patternsObserved = await this.selectPeaksAndFormPatterns();
			if (patternsObserved.length === 0) {
				console.log(`  No patterns observed at Level ${level}. Ending recursion.`);
				newHigherLevelNeuronsActivated = 0;
				break;
			}

			// activate neurons of observed patterns in the next level
			level++;
			await this.activateFrameNeurons(patternsObserved, level);

			// get the new active neuron count in the new level
			newHigherLevelNeuronsActivated = await this.getActiveNeuronCount(level);
		}
	}

	/**
	 * returns active neurons count at a given level
	 */
	async getActiveNeuronCount(level) {
		const activeNeurons = await this.conn.query(`SELECT COUNT(*) AS count FROM active_neurons WHERE level = ${level}`);
		return activeNeurons[0][0].count;
	}

	/**
	 * calculates the peakiness scores of newly activated neurons based on co-activations
	 */
	async calculatePeakiness(level) {
		console.log(`calculating peakiness for active neurons at level ${level}`);

		// fetch all active neurons at the current level with their relevant connections to other active neurons.
		const [activeConnections] = await this.conn.query(`
            SELECT c.neuron1_id, c.neuron2_id, SUM(c.strength) as strength
            FROM active_neurons s
            CROSS JOIN active_neurons t
            -- note that we join to the connections in a directionless way - smaller neuron id is always the source  
            JOIN connections c ON c.neuron1_id = LEAST(s.neuron_id, t.neuron_id) AND c.neuron2_id = GREATEST(s.neuron_id, t.neuron_id) 
            WHERE s.level = ? -- get the active neurons in the given level
            AND s.age = 0 -- the source neurons that are newly activated will be firing - we calculate their scores 
        	AND t.level = s.level -- we cluster within the same level
			AND (t.neuron_id != s.neuron_id OR t.age != s.age) -- if it's the same neuron, it's gotta be an older one
			GROUP BY c.neuron1_id, c.neuron2_id -- it's possible to see the same pair swapped as source and target, sum them up in that case
        `, [level]);

		// if there are no connections found between active neurons, error out - that's not normal - we must have added them even at first observation
		if (activeConnections.length === 0) throw new Error('No connections found for calculating peakiness.');

		// calculate active connection strengths (directionless) for each neuron from the query results
		const neuronStrengths = activeConnections.reduce((result, { neuron1_id, neuron2_id, strength }) => {
			result[neuron1_id] = (result[neuron1_id] || 0) + strength;
			result[neuron2_id] = (result[neuron2_id] || 0) + strength;
			return result;
		}, {});

		// calculate neighboring neurons for each neuron from the query results (directionless, from both sides)
		const neuronNeighbors = activeConnections.reduce((result, { neuron1_id, neuron2_id }) => {
			if (!result[neuron1_id]) result[neuron1_id] = new Set();
			if (!result[neuron2_id]) result[neuron2_id] = new Set();
			result[neuron1_id].add(neuron2_id);
			result[neuron2_id].add(neuron1_id);
			return result;
		}, {});

		// calculate the peakiness scores for each active neuron
		const peakinessScores = Object.entries(neuronStrengths).map(([neuronId, strength]) => {

			// get the neighbors of the neuron - if there are none, its strength will not be impacted
			const neighborIds = neuronNeighbors[neuronId] ? Array.from(neuronNeighbors[neuronId]) : [];
			if (neighborIds.length === 0) return [Number(neuronId), strength];

			// calculate the neighborhood average strength
			const neighborhoodStrengthSum = neighborIds.reduce((agg, neighborId) => agg + (neuronStrengths[neighborId] || 0), 0);
			const neighborhoodStrengthAvg = neighborhoodStrengthSum / neighborIds.size;

			// calculate the peakiness score for the neuron and return it
			const peakinessScore = (strength < this.peakMinStrength || strength < this.peakMinRatio * neighborhoodStrengthAvg) ? 0 :
				(strength - this.peakMinRatio * neighborhoodStrengthAvg);
			return [Number(neuronId), peakinessScore];
		});

		// batch insert peakiness scores
		await this.conn.query('INSERT INTO potential_peaks (neuron_id, peakiness_score) VALUES ?', [peakinessScores]);
		console.log(`calculated and inserted ${peakinessScores.length} potential peaks.`);
	}

	// Thoughts: should this be like activateNeurons, but with higher-level/lower-level connections between neurons?
	//  so, we will determine the peaks first, and their centroid will give us the new point,
	//  but then, after that, we use the coordinates to fetch the closest matching neuron to the centroid
	//  we could build and use connections for that to speed it up maybe?
	//  that would allow us to activate higher level neurons faster? Not sure. Never mind.
	async selectPeaksAndFormPatterns() {
		console.log("  B.2: Selecting peaks and forming patterns.");
		const patternsObserved = []; // To return for B.3
		let patternIdCounter = (await this.conn.query('SELECT COALESCE(MAX(pattern_id), 0) AS max_id FROM observed_pattern_centroids'))[0][0].max_id;

		// Fetch peaks, ordered by score, not yet suppressed
		const [peaks] = await this.conn.query(`
            SELECT pp.neuron_id, pp.peakiness_score
            FROM potential_peaks pp
            WHERE pp.peakiness_score > 0 AND pp.neuron_id NOT IN (SELECT neuron_id FROM suppressed_neurons) 
            ORDER BY pp.peakiness_score DESC
        `);

		// Thoughts: this neuron suppression makes it impossible to operate - I think we should just get all potential peaks
		// and each one activates its own higher level neuron based on its surroundings (active connections impacting the centroid)
		// so, each active neuron becomes like a switch - learning to activate different higher level patterns based on the surrounding neurons

		for (const peak of peaks) {
			const currentPeakId = peak.neuron_id;

			// Re-check suppression in case another peak in this batch suppressed it
			const [isSuppressed] = await this.conn.query('SELECT 1 FROM suppressed_neurons WHERE neuron_id = ?', [currentPeakId]);
			if (isSuppressed.length > 0) continue;

			patternIdCounter++;

			// B.2.a: Find Ingredient Neurons for the current peak.
			const [ingredients] = await this.conn.query(`
                SELECT c.target_id AS ingredient_id
                FROM connections c
                JOIN active_neurons an_target ON c.target_id = an_target.neuron_id
                WHERE c.source_id = ? AND c.strength > ?
            `, [currentPeakId, this.hp._min_conn_strength_]);

			if (ingredients.length === 0) {
				console.log(`    Peak ${currentPeakId} has no significant active ingredients. Skipping.`);
				continue;
			}

			const ingredientIds = ingredients.map(i => i.ingredient_id);
			const ingredientInserts = ingredientIds.map(id => [patternIdCounter, id]);
			await this.conn.query('INSERT INTO observed_pattern_ingredients (pattern_id, ingredient_id) VALUES ?', [ingredientInserts]);

			// B.2.b: Calculate the Centroid of these active ingredients.
			const [centroidCoords] = await this.conn.query(`
                SELECT dimension_id, AVG(val) AS centroid_value
                FROM coordinates c
                WHERE neuron_id IN (?)
                GROUP BY dimension_id
            `, [ingredientIds]);

			const centroidInserts = centroidCoords.map(c => [patternIdCounter, c.dimension_id, c.centroid_value]);
			await this.conn.query('INSERT INTO observed_pattern_centroids (pattern_id, dimension_id, value) VALUES ?', [centroidInserts]);

			// Store for later use
			patternsObserved.push({
				pattern_id: patternIdCounter,
				centroid: centroidCoords,
				ingredients: ingredientIds
			});

			// B.2.c: Suppress the selected peak and its direct, strong, active neighbors.
			const suppressedNeuronIds = [currentPeakId, ...ingredientIds];
			const suppressedInserts = suppressedNeuronIds.map(id => [id]);
			await this.conn.query('INSERT IGNORE INTO suppressed_neurons (neuron_id) VALUES ?', [suppressedInserts]);

			console.log(`    Pattern ${patternIdCounter} (from peak ${currentPeakId}) identified with ${ingredients.length} ingredients.`);
		}
		console.log(`  Found ${patternsObserved.length} unique patterns at this level.`);
		return patternsObserved;
	}


	async _getNextNeuronId() {
		const [rows] = await this.conn.query('SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM neurons');
		return rows[0].next_id;
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