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

		// bucketStats is lazy-initialized on first observed sample (see observe()). It
		// holds per-bucket {count, sum} so dequantize can return the empirical mean of
		// each bucket instead of its geometric midpoint — meaningful continuous output
		// even at low resolution, since the output lives in the actual input scale.
		const state = { mode, resolution, boundaries: null, bucketStats: null };

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
	 * Feed a raw scalar to the quantizer. Two jobs:
	 *   (1) for 'dynamic' mode, buffer samples and compute split points after warmup
	 *   (2) for any mode with known boundaries (static always, dynamic post-warmup),
	 *       accumulate a running {count, sum} per bucket so dequantize can return the
	 *       empirical mean of observations in that bucket instead of a geometric
	 *       midpoint. This lifts the dequantize output off the ±0.5 ceiling at res=2
	 *       and into the actual input scale (e.g. a "positive volume" bucket returns
	 *       ~+40% once it's seen typical positive-volume moves).
	 * No-op for 'passthrough' and for dims the caller hasn't registered.
	 */
	observe(dimId, scalar) {
		const state = this.dimensions.get(dimId);

		// unregistered dims are owned by channels that bucketize on their own; ignore
		if (!state) return;
		if (state.mode === 'passthrough') return;

		// dynamic: buffer and learn boundaries at warmup
		if (state.mode === 'dynamic') {
			state.samples.push(scalar);

			// compute initial boundaries once we have enough warmup samples
			if (state.boundaries === null && state.samples.length >= state.warmupSamples)
				state.boundaries = this.computeQuantileBoundaries(state.samples, state.resolution);

			// TODO: incremental boundary refinement post-warmup (e.g. sliding window or t-digest)
		}

		// accumulate per-bucket empirical mean. Requires boundaries, so dynamic mode
		// starts contributing only after warmup — pre-warmup samples are not attributed
		// to any bucket since the bucketing itself isn't defined yet.
		if (!state.boundaries) return;
		if (!state.bucketStats) {
			state.bucketStats = new Array(state.resolution);
			for (let i = 0; i < state.resolution; i++) state.bucketStats[i] = { count: 0, sum: 0 };
		}
		const bucketId = this.bucketize(scalar, state.boundaries);
		const stats = state.bucketStats[bucketId - 1];
		stats.count++;
		stats.sum += scalar;
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
	 * static / dynamic: looks up the empirical mean of observed samples per bucket
	 * (populated by observe()). Returns null when the bucket has never been observed
	 * — honest about the absence of data rather than fabricating a geometric midpoint.
	 * Callers must handle null (skip from weighted sums, skip from MAPE, etc.).
	 *
	 * @param {number} dimId
	 * @param {number} bucketId - 1-indexed; may be fractional (e.g. 2.7 for weighted avg)
	 * @returns {number|null} representative scalar, or null if unobserved
	 */
	dequantize(dimId, bucketId) {
		const state = this.dimensions.get(dimId);

		// unregistered dims come from channels that still own their own bucketization;
		// treat the bucket ID as the scalar (same as passthrough mode)
		if (!state) return bucketId;

		if (state.mode === 'passthrough') return bucketId;

		// dynamic mode before warmup, or static/dynamic with no samples yet: no data
		// to produce a scalar from. Null propagates through callers as "no prediction".
		if (!state.boundaries) return null;

		return this.interpolateRepresentative(bucketId, this.bucketRepresentatives(state));
	}

	/**
	 * Per-bucket representative scalars: the empirical mean of samples observed in
	 * each bucket (populated by observe()), or null for buckets we've never seen.
	 * Null is load-bearing — callers use it to know when to skip a contribution
	 * rather than consuming a fabricated midpoint that biases predictions.
	 */
	bucketRepresentatives(state) {
		if (!state.bucketStats) return new Array(state.resolution).fill(null);
		const out = new Array(state.resolution);
		for (let i = 0; i < state.resolution; i++) {
			const s = state.bucketStats[i];
			out[i] = s.count > 0 ? s.sum / s.count : null;
		}
		return out;
	}

	/**
	 * Linear interpolation over bucket representatives. Unseen buckets carry null;
	 * if one flanking rep is null the other is used as-is, and if both are null the
	 * result is null. Clamps at the ends so out-of-range bucketIds saturate instead
	 * of extrapolating.
	 */
	interpolateRepresentative(bucketId, reps) {
		const n = reps.length;
		if (bucketId <= 1) return reps[0];
		if (bucketId >= n) return reps[n - 1];
		const lo = Math.floor(bucketId);
		const frac = bucketId - lo;
		const a = reps[lo - 1];
		const b = reps[lo];
		if (a === null && b === null) return null;
		if (a === null) return b;
		if (b === null) return a;
		return a * (1 - frac) + b * frac;
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
	 * Persists boundaries (static + dynamic learned) and per-bucket empirical-mean
	 * accumulators — the latter is learned state that takes many frames to collect
	 * and would otherwise have to be rebuilt from scratch on every restart.
	 * Sample buffers (dynamic mode pre-warmup reservoir) are treated as transient.
	 */
	serialize() {
		const out = {};
		for (const [dimId, state] of this.dimensions)
			out[dimId] = {
				mode: state.mode,
				resolution: state.resolution,
				boundaries: state.boundaries ? [...state.boundaries] : null,
				bucketStats: state.bucketStats ? state.bucketStats.map(s => ({ count: s.count, sum: s.sum })) : null
			};
		return out;
	}

	/**
	 * Restore previously serialized state. Dimensions must be registered first
	 * with the same mode/resolution; this only reinstates boundaries and the
	 * per-bucket accumulators used for empirical-mean dequantization.
	 */
	restore(snapshot) {
		for (const [dimIdStr, saved] of Object.entries(snapshot)) {
			const dimId = Number(dimIdStr);
			const state = this.dimensions.get(dimId);
			if (!state) continue; // unknown dimension - ignore
			if (saved.boundaries) state.boundaries = [...saved.boundaries];
			if (saved.bucketStats) state.bucketStats = saved.bucketStats.map(s => ({ count: s.count, sum: s.sum }));
		}
	}
}
