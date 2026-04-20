/**
 * Quantizer - maps continuous scalars to discrete bucket IDs for neuron addressing.
 *
 * One quantizer instance lives inside the brain (owned by Thalamus). It holds
 * per-dimension bucket state. The algorithm is shared across all dimensions;
 * only the boundaries differ.
 *
 * Modes:
 *   'passthrough' - input is already an integer bucket ID in [1..resolution]; returned as-is.
 *                   Use this to reproduce pre-quantizer behavior when the encoder
 *                   outside the brain already emits discrete buckets.
 *   'static'      - fixed boundaries supplied at registration. Scalar → bucket via
 *                   boundary comparison. Equivalent to the old discretizeChange().
 *   'dynamic'     - boundaries adapt to observed values. Encoder emits raw scalars;
 *                   quantizer learns the split points. Skeleton present - not wired
 *                   into the frame pipeline yet.
 *
 * Buckets are 1-indexed: for resolution N, valid bucket IDs are 1..N.
 * With K boundaries, resolution = K + 1.
 */
export class Quantizer {

	/**
	 * returns new quantizer instance
	 */
	constructor() {
		this.dimensions = new Map(); // dimId → { mode, resolution, boundaries, samples? }
	}

	/**
	 * Register a dimension with the quantizer.
	 * @param {number} dimId - numeric dimension ID
	 * @param {object} spec
	 * @param {number} spec.resolution - number of buckets (>= 2)
	 * @param {string} [spec.mode='passthrough'] - 'passthrough' | 'static' | 'dynamic'
	 * @param {number[]} [spec.boundaries] - sorted ascending split points; length = resolution - 1 (required for 'static')
	 * @param {number} [spec.warmupSamples=1000] - samples to collect before first boundary computation (dynamic mode)
	 */
	registerDimension(dimId, spec) {
		const mode = spec.mode ?? 'passthrough';
		const resolution = spec.resolution;

		if (!Number.isInteger(resolution) || resolution < 2)
			throw new Error(`Quantizer: resolution must be integer >= 2, got ${resolution} for dim ${dimId}`);

		const state = { mode, resolution, boundaries: null };

		if (mode === 'static') {
			if (!Array.isArray(spec.boundaries) || spec.boundaries.length !== resolution - 1)
				throw new Error(`Quantizer: static mode requires ${resolution - 1} boundaries for dim ${dimId}, got ${spec.boundaries?.length}`);
			state.boundaries = [...spec.boundaries];
		}
		else if (mode === 'dynamic') {
			state.boundaries = null; // computed after warmup
			state.samples = []; // reservoir of observed values
			state.warmupSamples = spec.warmupSamples ?? 1000;
		}
		else if (mode !== 'passthrough')
			throw new Error(`Quantizer: unknown mode '${mode}' for dim ${dimId}`);

		this.dimensions.set(dimId, state);
	}

	/**
	 * Whether a dimension has been registered.
	 */
	has(dimId) {
		return this.dimensions.has(dimId);
	}

	/**
	 * Get the resolution (bucket count) for a dimension.
	 */
	getResolution(dimId) {
		const state = this.dimensions.get(dimId);
		if (!state) throw new Error(`Quantizer: dimension ${dimId} not registered`);
		return state.resolution;
	}

	/**
	 * Get current boundaries for a dimension (null if dynamic + pre-warmup).
	 */
	getBoundaries(dimId) {
		const state = this.dimensions.get(dimId);
		if (!state) throw new Error(`Quantizer: dimension ${dimId} not registered`);
		return state.boundaries ? [...state.boundaries] : null;
	}

	/**
	 * Feed a raw scalar to the quantizer - updates internal state for 'dynamic' mode.
	 * No-op for 'passthrough' and 'static'.
	 */
	observe(dimId, scalar) {
		const state = this.dimensions.get(dimId);
		if (!state) throw new Error(`Quantizer: dimension ${dimId} not registered`);
		if (state.mode !== 'dynamic') return;

		state.samples.push(scalar);

		// compute initial boundaries once we have enough warmup samples
		if (state.boundaries === null && state.samples.length >= state.warmupSamples)
			state.boundaries = this.computeQuantileBoundaries(state.samples, state.resolution);

		// TODO: incremental boundary refinement post-warmup (e.g. sliding window or t-digest)
	}

	/**
	 * Map a scalar to a 1-indexed bucket ID in [1..resolution].
	 * @param {number} dimId
	 * @param {number} scalar
	 * @returns {number} bucket ID
	 */
	quantize(dimId, scalar) {
		const state = this.dimensions.get(dimId);
		if (!state) throw new Error(`Quantizer: dimension ${dimId} not registered`);

		// input should already be an integer bucket ID - sign and magnitude don't matter,
		// only that the encoder uses integer IDs consistently across frames
		if (state.mode === 'passthrough') {
			if (!Number.isInteger(scalar))
				throw new Error(`Quantizer: passthrough dim ${dimId} expected integer bucket ID, got ${scalar}`);
			return scalar;
		}

		// dynamic mode before warmup completes: place everything in the middle bucket
		// this keeps the pipeline running without creating spurious neuron coverage
		if (state.boundaries === null)
			return Math.ceil(state.resolution / 2);

		return this.bucketize(scalar, state.boundaries);
	}

	/**
	 * Standard boundary comparison: returns 1-indexed bucket.
	 * Matches the semantics of the original discretizeChange() in stock.js.
	 */
	bucketize(value, boundaries) {
		for (let i = 0; i < boundaries.length; i++)
			if (value <= boundaries[i]) return i + 1;
		return boundaries.length + 1;
	}

	/**
	 * Compute equal-frequency (quantile) boundaries from a sample buffer.
	 * Produces resolution-1 split points that divide samples into equal-count buckets.
	 */
	computeQuantileBoundaries(samples, resolution) {
		const sorted = [...samples].sort((a, b) => a - b);
		const boundaries = [];
		for (let i = 1; i < resolution; i++) {
			const idx = Math.floor((sorted.length * i) / resolution);
			boundaries.push(sorted[Math.min(idx, sorted.length - 1)]);
		}
		return boundaries;
	}

	/**
	 * Serialize quantizer state for persistence.
	 * Only static boundaries and dynamic-mode learned boundaries are persisted;
	 * sample buffers are treated as transient.
	 */
	serialize() {
		const out = {};
		for (const [dimId, state] of this.dimensions)
			out[dimId] = {
				mode: state.mode,
				resolution: state.resolution,
				boundaries: state.boundaries ? [...state.boundaries] : null
			};
		return out;
	}

	/**
	 * Restore previously serialized state. Dimensions must be registered first
	 * with the same mode/resolution; this only reinstates boundaries.
	 */
	restore(snapshot) {
		for (const [dimIdStr, saved] of Object.entries(snapshot)) {
			const dimId = Number(dimIdStr);
			const state = this.dimensions.get(dimId);
			if (!state) continue; // unknown dimension - ignore
			if (saved.boundaries) state.boundaries = [...saved.boundaries];
		}
	}
}
