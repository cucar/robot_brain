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

			// activate higher level neurons until there are no neurons activated
			const predictedNeuronIds = []; // predicted neuron ids that will light up soon
			let level = 0; // start with level 0 - base neurons corresponding to frame directly
			let levelPoints = frame; // the level points will change, but the initial points come from the frame
			let clusters = {}; // map of current level points to the neuron ids that resulted in that cluster { point_str: [neuron_id] } - empty for level 0
			while (true) {
				console.log(`processing level: ${level}`);

				// get the neurons to activate for the given coordinates as { point_str, neuron_id }
				const matches = await this.getFrameNeurons(levelPoints, level);

				// activate the neurons corresponding to the frame in the current level with age 0
				// returns the predicted neuron ids of the lower levels - that will be the output of the current level
				const predictions = await this.activateFrameNeurons(matches, level, clusters);
				predictedNeuronIds.push(...predictions);

				// extract the observed patterns in the frame - each pattern is { centroid, neuron_ids }
				// if no patterns found in the level, stop processing
				const patterns = await this.getFramePatterns(level);
				if (patterns.length === 0) {
					console.log(`no patterns observed at level ${level}. ending frame processing.`);
					break;
				}

				// now update the values for the next level iteration to observe the patterns found
				levelPoints = patterns.map(p => p.centroid);
				clusters = patterns.reduce((obj, p) => ({ ...obj, [JSON.stringify(p.centroid)]: p.neuron_ids }));
				level++;
			}

			// increment the forget counter and run forget cycle if the limit is reached
			this.forgetCounter++;
			if (this.forgetCounter % this.decayRate === 0) await this.runForgetCycle();

			// TODO: sort the predicted neuron ids based on most occurring to least occurring and return them as a map from neuron id to prediction count
			//  then, console log the predicted neuron ids
			console.log('predictedNeuronIds', predictedNeuronIds);

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
	 * ages neurons in the context - sliding the window across frames
	 */
	async ageNeurons() {
		await this.conn.query('UPDATE active_neurons SET age = age + 1');
		await this.conn.query(`DELETE FROM active_neurons WHERE age > ?`, [this.neuronMaxAge]);
		console.log('Active neurons aged and old ones deactivated.');
	}

	/**
	 * activates neurons corresponding to given set of points active for the frame. returns number of activated neurons.
	 */
	async activateFrameNeurons(matches, level, clusters) {
		console.log('activating neurons', level, matches, clusters);

		// activate the neurons of the frame and return the number of newly activated neurons
		await this.activateNeurons(matches, level);

		// adapt coordinates of activated neurons towards their observed points
		await this.applyCoordinateAdaptation(matches);

		// reinforce connections between active neurons in the current level
		await this.reinforceConnections(level);

		// reinforce inter-level pattern links using previous level clusters
		await this.reinforcePatterns(matches, level, clusters);

		// TODO: get the lower-level neurons that activated these newly higher-level neurons before (children in patterns table)
		//  the children neurons that are not activated yet will be the predictions. return them.

		// Placeholder until prediction flow is implemented
		return [];
	}

	/**
	 * Reinforces inter-level pattern links from previous-level clusters to the current level's matched neurons.
	 * For each matched parent neuron at level > 0, strengthens patterns(parent_id=parent, child_id in previous cluster).
	 */
	async reinforcePatterns(matches, level, clusters) {
		
		// Only meaningful for levels above 0 and when we have both matches and cluster context
		if (level <= 0 || !Array.isArray(matches) || matches.length === 0 || !clusters || Object.keys(clusters).length === 0)
			return;

		// Aggregate counts to avoid duplicate rows and to reflect multiple appearances within a frame
		const parentChildToCount = new Map(); // key: `${parentId}:${childId}` -> count
		for (const match of matches) {
			const parentNeuronId = Number(match.neuron_id);
			const childNeuronIds = clusters[match.point_str] || [];
			for (const childNeuronId of childNeuronIds) {
				const key = `${parentNeuronId}:${childNeuronId}`;
				parentChildToCount.set(key, (parentChildToCount.get(key) || 0) + 1);
			}
		}

		const bulkRows = Array.from(parentChildToCount.entries()).map(([key, count]) => {
			const [parentId, childId] = key.split(':').map(Number);
			return [parentId, childId, count];
		});

		if (bulkRows.length === 0) return;

		await this.conn.query(
			`INSERT INTO patterns (parent_id, child_id, strength) VALUES ? 
			ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength)`,
			[bulkRows]
		);
		console.log(`Reinforced ${bulkRows.length} inter-level pattern link(s).`);
	}

	/**
	 * activates a given set of neurons at a given level with age 0 - matches is an array of { point_str, neuron_id }
	 */
	async activateNeurons(matches, level) {
		const neuronIds = Array.from(new Set((matches || []).map(m => m.neuron_id))); // derive unique neuron ids
		const activations = neuronIds.map(neuronId => [neuronId, level, 0]); // activate with age 0
		console.log('activations', activations);
		await this.conn.query('INSERT INTO active_neurons (neuron_id, level, age) VALUES ?', [activations]);
	}

	/**
	 * returns the resolution for a level. resolution is the threshold we use when matching the neurons to points.
	 * at level 0, it's very fine, to the point of making it almost exact (10^-8)
	 * at level 1, the resolution is 10^-7, level 2 is 10^-6, etc. level 6 would be 10^-2.
	 * as we go into higher levels, we need to be able to handle partial matches due to the different states of concepts
	 * that's why we increase the resolution to cast a wider net to group higher level concepts
	 */
	getResolution(level) {
		return Math.pow(10, level - (this.maxResolution + this.maxLevel));
	}

	/**
	 * returns the best matching neurons for given frame and level as [{ point_str, neuron_id }]
	 */
	async getFrameNeurons(frame, level) {

		// we may relax this restriction later, but the inherent nature of levels and resolutions prevents infinite expansion
		if (level > this.maxLevel) throw new Error(`Level cannot exceed ${this.maxLevel}.`);

		// resolution is the matching threshold we will use when fetching the neurons
		const resolution = this.getResolution(level);

		// get all the neurons that have coordinates in close ranges for each point
		const pointNeuronMatches = await this.getNeuronsFromFrame(frame, resolution);
		console.log('pointNeuronMatches', pointNeuronMatches);

		// matching neuron ids to be returned for each point of the frame for adaptation { point_str, neuron_id }
		const matches = [];

		// separate points into with/without any candidate neurons
		const pointsWithoutMatches = pointNeuronMatches.filter(p => p.neuron_ids.length === 0);
		const pointsWithMatches = pointNeuronMatches.filter(p => p.neuron_ids.length > 0);
		console.log('pointsWithMatches', pointsWithMatches);

		// find the best matching neurons for points that had any candidates
		let bestMatchingResults = [];
		if (pointsWithMatches.length > 0) {
			bestMatchingResults = await this.getBestMatchingNeuronIds(pointsWithMatches, resolution);
			console.log('bestMatchingResults', bestMatchingResults);

			// add successful matches to the results
			matches.push(...bestMatchingResults.filter(result => result.neuron_id));
		}

		// points that still need new neurons: those with no candidates + those with candidates but no close enough match
		const pointsNeedingNeurons = [
			...pointsWithoutMatches.map(p => p.point_str),
			...bestMatchingResults.filter(r => !r.neuron_id).map(r => r.point_str)
		];
		if (pointsNeedingNeurons.length > 0) {
			console.log(`${pointsNeedingNeurons.length} points need new neurons (after matching). Creating neurons once with internal dedupe.`);
			const createdNeurons = await this.createFrameNeurons(pointsNeedingNeurons, resolution);
			matches.push(...createdNeurons);
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

		// Across one frame, multiple observed points can match the same neuron. We aggregate observations per (neuron_id, dimension_id)
		// and use the mean to produce a single deterministic update per pair: new = old * (1 - lr) + observed_mean * lr
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
	 */
	async createFrameNeurons(points, resolution) {
		
		// Deduplicate points within the batch
		const repsToCreate = this.dedupePoints(points, resolution);

		// Nothing to create
		if (repsToCreate.length === 0) return [];

		// 1) Bulk insert neurons. MySQL returns the first insert id; ids are contiguous within the statement.
		const valuesSql = repsToCreate.map(() => '()').join(',');
		const insertNeuronsResult = await this.conn.query(`INSERT INTO neurons () VALUES ${valuesSql}`);
		const firstNewId = insertNeuronsResult[0].insertId;

		// 2) Compute ids from the first insert id (assume auto_increment step = 1)
		const created = repsToCreate.map((centroid, idx) => ({
			point_str: JSON.stringify(centroid),
			neuron_id: firstNewId + idx
		}));

		// 3) Bulk insert coordinates for all created neurons
		const rows = created.flatMap(({ neuron_id, point_str }) => {
			const centroid = JSON.parse(point_str);
			return Object.entries(centroid).map(([dimName, value]) => [neuron_id, this.dimensionNameToId[dimName], value]);
		});
		if (rows.length > 0) await this.conn.query('INSERT INTO coordinates (neuron_id, dimension_id, val) VALUES ?', [rows]);

		return created;
	}

	/**
	 * Resolution-aware deduplication for a batch of points (JSON strings). Returns representative centroid objects.
	 */
	dedupePoints(points, resolution) {

		// 1) Parse and normalize input points
		//    - Keep dims sorted to form a stable dimension-key per point
		//    - Precompute grid coordinates per dimension using the provided resolution
		const parsed = points.map(str => {
			const point = JSON.parse(str);
			const dims = Object.keys(point).sort();
			return {
				point,
				dims,
				dimKey: dims.join('|'),
				grid: dims.map(d => Math.floor(point[d] / resolution))
			};
		});

		// 2) Group points by their dimension set so we only compare like-with-like
		const groups = parsed.reduce((acc, entry) => (
			acc.set(entry.dimKey, (acc.get(entry.dimKey) || []).concat(entry))
		), new Map());

		// Helper: generate neighbor offsets in {-1,0,1}^k functionally
		const enumerateOffsets = k => Array.from({ length: k })
			.reduce((acc) => [-1, 0, 1].flatMap(o => acc.map(a => [...a, o])), [[]]);

		// Helper: for a cells map, find an existing representative within resolution among neighboring cells
		const findMergeTarget = (cells, grid, dims, point) =>
			enumerateOffsets(dims.length)
				.map(offset => grid.map((g, i) => g + offset[i]).join(','))
				.map(key => cells.get(key) || [])
				.flat()
				.find(rep => dims.every(d => Math.abs(rep.centroid[d] - point[d]) <= resolution)) || null;

		// 3) For each dimension group, fold points into a sparse grid of representatives and return them.
		// Final list of de-duped centroids to create neurons for.
		return Array.from(groups.values()).flatMap(entries => {
			// cells: Map<cellKey, Array<{ centroid, weight }>>
			const cells = entries.reduce((cellMap, { point, dims, grid }) => {
				const target = findMergeTarget(cellMap, grid, dims, point);
				if (target) {
					// Incremental weighted mean (equal weights here)
					const w0 = target.weight || 1;
					const w1 = 1;
					const w = w0 + w1;
					target.centroid = dims.reduce((agg, d) => ({
						...agg,
						[d]: (target.centroid[d] * w0 + point[d] * w1) / w,
					}), target.centroid);
					target.weight = w;
					return cellMap;
				}
				const key = grid.join(',');
				const nextList = (cellMap.get(key) || []).concat({ centroid: { ...point }, weight: 1 });
				cellMap.set(key, nextList);
				return cellMap;
			}, new Map());

			return Array.from(cells.values()).flat().map(rep => rep.centroid);
		});
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
		return rows.map(row => ({ point_str: row.point_str, neuron_ids: JSON.parse(row.neuron_ids) }));
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
	 * Extract observed patterns at a given level from the currently active neurons.
	 *
	 * High-level algorithm per frame level:
	 * - Build a local graph among active neurons at this level using their recent co-activation strengths.
	 * - Compute each neuron's total strength (sum of incident connection strengths) and its neighbors.
	 * - Select peak candidates that are above an absolute and relative (to neighbor average) threshold.
	 * - For each peak, compute a weighted centroid of its ingredient neurons:
	 *   - The peak neuron is weighted by its own total strength in this frame.
	 *   - Each neighbor contributes with weight equal to the connection strength to the peak (soft membership).
	 *   - Centroid is computed per dimension using a streaming weighted mean for numerical stability.
	 * - Return patterns as { centroid, neuron_ids } where neuron_ids are the peak and its neighbors.
	 */
	async getFramePatterns(level) {
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

		// quick lookup for connection strengths between active neuron pairs (directionless)
		// We key by `${minId}:${maxId}` so the same pair maps to the same entry regardless of order.
		// These strengths represent observation counts of co-activation and are used as soft cluster weights.
		const pairStrength = new Map(); // key: `${minId}:${maxId}` -> strength
		for (const { neuron1_id, neuron2_id, strength } of activeConnections)
			pairStrength.set(`${neuron1_id}:${neuron2_id}`, strength);

		// select peaks among active neurons and calculate their centroids
		const patterns = [];
		for (const [neuronId, strength] of Object.entries(neuronStrengths)) {

			// if there are no neighbors for the neuron, it doesn't make sense - neuron strength comes from connection strengths to other neurons
			const peakId = Number(neuronId);
			if (!neuronNeighbors[peakId] || neuronNeighbors[peakId].size === 0)
				throw new Error(`Cannot find neighbors for an active neuron: ${peakId} (strength: ${strength}`);

			// calculate the neighborhood average strength for the neuron
			const neighborhoodStrengthSum = Array.from(neuronNeighbors[peakId]).reduce((agg, neighborId) => agg + (neuronStrengths[neighborId] || 0), 0);
			const neighborhoodStrengthAvg = neighborhoodStrengthSum / neuronNeighbors[peakId].size;

			// if the neuron strength does not qualify to be a peak, skip it
			// this is basically the suppression of neighbor neurons - something that is also seen in biology
			if (strength < this.peakMinStrength || strength < this.peakMinRatio * neighborhoodStrengthAvg) continue;

			// calculate the relative strength for the neuron compared to its neighborhood (not used anymore but good for debugging)
			// const peakinessScore = strength - this.peakMinRatio * neighborhoodStrengthAvg;

			// compute weighted centroid from the peak and its neighbors using connection strengths as soft weights
			// Ingredients for centroid: the peak itself plus its directly connected active neighbors.
			// - Peak weight = its total strength for this frame (strong peaks influence centroid more).
			// - Neighbor weight = connection strength to the peak (soft-membership based on co-activation).
			// - We ignore zero/negative weights to avoid polluting the centroid.
			const neighborIds = Array.from(neuronNeighbors[peakId]);
			const clusterStrengths = [
				{ neuron_id: peakId, strength: Number(strength) },
				...neighborIds.map(nId => {
					const a = Math.min(peakId, Number(nId));
					const b = Math.max(peakId, Number(nId));
					return { neuron_id: Number(nId), strength: pairStrength.get(`${a}:${b}`) || 0 };
				})
			].filter(m => m.strength > 0);

			// Fetch coordinates for all ingredient neurons.
			// Note: member ids are internal integers aggregated from DB results; composing the IN list is safe here.
			// If external input ever reaches this path, prefer placeholders to avoid SQL injection.
			const [rows] = await this.conn.query(`
				SELECT neuron_id, dimension_id, val 
				FROM coordinates 
				WHERE neuron_id IN (${clusterStrengths.map(m => m.neuron_id).join(',')})`
			);

			// Streaming weighted mean per dimension:
			// newMean = prevMean + (w * (x - prevMean)) / (prevWeight + w)
			// This avoids storing all points and is numerically stable compared to summing then dividing.
			const weights = Object.fromEntries(clusterStrengths.map(m => [m.neuron_id, m.strength]));
			const centroid = {};
			const dimWeights = new Map(); // dimId -> total weight so far
			for (const row of rows) {
				const dimName = this.dimensionIdToName[row.dimension_id];
				const oldWeight = dimWeights.get(row.dimension_id) || 0;
				const prevMean = centroid[dimName] ?? 0;
				const w = weights[row.neuron_id] || 0;
				if (!w) continue;
				const newWeight = oldWeight + w;
				const val = Number(row.val);
				centroid[dimName] = oldWeight === 0 ? val : prevMean + (w * (val - prevMean)) / newWeight;
				dimWeights.set(row.dimension_id, newWeight);
			}

			// store as pattern { centroid, neuron_ids }
			patterns.push({ centroid, neuron_ids: clusterStrengths.map(m => m.neuron_id) });
		}

		// return the patterns as { centroid, neuron_ids }
		console.log(`calculated ${patterns.length} potential patterns.`);
		return patterns;
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