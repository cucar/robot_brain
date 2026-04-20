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

		// unregistered dims are owned by channels that bucketize on their own; ignore
		if (!state) return;
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

		// unregistered dim: channel already emits a discrete bucket ID, pass it through
		if (!state) {
			if (!Number.isInteger(scalar))
				throw new Error(`Quantizer: unregistered dim ${dimId} requires integer bucket ID, got ${scalar}`);
			return scalar;
		}

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
	 * Map a bucket ID back to a representative scalar in the dimension's input space.
	 * Accepts a fractional bucket ID so callers can pass the weighted average of a
	 * vote distribution and get a continuous scalar prediction back.
	 *
	 * passthrough: the bucket ID IS the scalar, returned unchanged.
	 * static / dynamic: interpolates across bucket midpoints. Interior midpoints are
	 * the average of adjacent boundaries; the two outer buckets are open-ended, so we
	 * reflect the nearest interior width to get a finite representative value.
	 *
	 * @param {number} dimId
	 * @param {number} bucketId - 1-indexed; may be fractional (e.g. 2.7 for weighted avg)
	 * @returns {number} representative scalar
	 */
	dequantize(dimId, bucketId) {
		const state = this.dimensions.get(dimId);

		// unregistered dims come from channels that still own their own bucketization;
		// treat the bucket ID as the scalar (same as passthrough mode)
		if (!state) return bucketId;

		if (state.mode === 'passthrough') return bucketId;

		// dynamic mode before warmup: we returned the middle bucket from quantize(),
		// so dequantize the same way - no boundaries means no meaningful scalar yet
		if (!state.boundaries) return 0;

		return this.interpolateBucketMidpoint(bucketId, state.boundaries);
	}

	/**
	 * Linear interpolation over a piecewise-linear map from bucket index to scalar.
	 * The map is anchored at the integer bucket midpoints; fractional bucket IDs
	 * interpolate between the two surrounding midpoints.
	 */
	interpolateBucketMidpoint(bucketId, boundaries) {
		const midpoints = this.bucketMidpoints(boundaries);
		const n = midpoints.length;
		if (bucketId <= 1) return midpoints[0];
		if (bucketId >= n) return midpoints[n - 1];
		const lo = Math.floor(bucketId);
		const frac = bucketId - lo;
		return midpoints[lo - 1] * (1 - frac) + midpoints[lo] * frac;
	}

	/**
	 * Build one representative scalar per bucket. Interior buckets use the mean of
	 * their two boundaries; the two outer open-ended buckets mirror the adjacent
	 * interior width so they contribute a finite anchor to interpolation.
	 */
	bucketMidpoints(boundaries) {
		const resolution = boundaries.length + 1;
		const mids = new Array(resolution);

		// interior buckets: midpoint between the two boundaries that enclose them
		for (let i = 1; i < resolution - 1; i++)
			mids[i] = (boundaries[i - 1] + boundaries[i]) / 2;

		// outer buckets: reflect the adjacent interior half-width to stay finite.
		// for resolution=2 there is no interior bucket, so we just offset by 1 unit.
		if (resolution === 2) {
			mids[0] = boundaries[0] - 0.5;
			mids[1] = boundaries[0] + 0.5;
		}
		else {
			mids[0] = 2 * boundaries[0] - mids[1];
			mids[resolution - 1] = 2 * boundaries[resolution - 2] - mids[resolution - 2];
		}

		return mids;
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
