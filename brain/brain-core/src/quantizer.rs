/// Quantizer - maps continuous scalars to discrete bucket IDs for neuron addressing.
///
/// One quantizer instance lives inside the brain (owned by Thalamus). It holds
/// per-dimension bucket state. The algorithm is shared across all dimensions;
/// only the boundaries differ.
///
/// Modes:
///   Passthrough - input is already an integer bucket ID in [1..resolution]; returned as-is.
///                 Use this to reproduce pre-quantizer behavior when the encoder
///                 outside the brain already emits discrete buckets.
///   Static      - fixed boundaries supplied at registration. Scalar → bucket via
///                 boundary comparison. Equivalent to the old discretizeChange().
///   Dynamic     - boundaries adapt to observed values. Encoder emits raw scalars;
///                 quantizer learns the split points. Skeleton present - not wired
///                 into the frame pipeline yet.
///
/// Buckets are 1-indexed: for resolution N, valid bucket IDs are 1..N.
/// With K boundaries, resolution = K + 1.

use rustc_hash::FxHashMap;

use crate::types::{BucketId, DimensionId};

/// Per-bucket empirical statistics: count and sum of observed samples.
#[derive(Debug, Clone)]
struct BucketStats {
    count: u64,
    sum: f64,
}

/// Quantization mode for a dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizeMode {
    Passthrough,
    Static,
    Dynamic,
}

/// Per-dimension quantizer state.
#[derive(Debug, Clone)]
struct DimensionState {
    mode: QuantizeMode,
    resolution: u32,
    boundaries: Option<Vec<f64>>,
    bucket_stats: Option<Vec<BucketStats>>,
    /// Dynamic mode only: reservoir of observed values before warmup.
    samples: Option<Vec<f64>>,
    /// Dynamic mode only: number of samples to collect before first boundary computation.
    warmup_samples: usize,
}

pub struct Quantizer {
    dimensions: FxHashMap<DimensionId, DimensionState>,
}

impl Quantizer {
    pub fn new() -> Self {
        Self { dimensions: FxHashMap::default() }
    }

    /// Register a dimension with the quantizer.
    /// * `dim_id` — numeric dimension ID
    /// * `resolution` — number of buckets (>= 2)
    /// * `mode` — Passthrough, Static, or Dynamic
    /// * `boundaries` — sorted ascending split points; length = resolution - 1 (required for Static)
    /// * `warmup_samples` — samples to collect before first boundary computation (Dynamic mode)
    pub fn register_dimension(&mut self, dim_id: DimensionId, resolution: u32, mode: QuantizeMode, boundaries: Option<Vec<f64>>, warmup_samples: Option<usize>) {
        if resolution < 2 { panic!("Quantizer: resolution must be >= 2, got {} for dim {}", resolution, dim_id); }

        // bucket_stats is lazy-initialized on first observed sample (see observe()). It
        // holds per-bucket {count, sum} so dequantize can return the empirical mean of
        // each bucket instead of its geometric midpoint — meaningful continuous output
        // even at low resolution, since the output lives in the actual input scale.
        let mut state = DimensionState {
            mode: mode.clone(),
            resolution,
            boundaries: None,
            bucket_stats: None,
            samples: None,
            warmup_samples: warmup_samples.unwrap_or(1000),
        };

        match mode {
            QuantizeMode::Static => {
                let b = boundaries.unwrap_or_else(|| panic!("Quantizer: static mode requires boundaries for dim {}", dim_id));
                if b.len() != (resolution - 1) as usize {
                    panic!("Quantizer: static mode requires {} boundaries for dim {}, got {}", resolution - 1, dim_id, b.len());
                }
                state.boundaries = Some(b);
            }
            QuantizeMode::Dynamic => {
                state.boundaries = None; // computed after warmup
                state.samples = Some(Vec::new()); // reservoir of observed values
            }
            QuantizeMode::Passthrough => {}
        }

        self.dimensions.insert(dim_id, state);
    }

    /// Whether a dimension has been registered.
    pub fn has(&self, dim_id: DimensionId) -> bool {
        self.dimensions.contains_key(&dim_id)
    }

    /// Feed a raw scalar to the quantizer. Two jobs:
    ///   (1) for Dynamic mode, buffer samples and compute split points after warmup
    ///   (2) for any mode with known boundaries (static always, dynamic post-warmup),
    ///       accumulate a running {count, sum} per bucket so dequantize can return the
    ///       empirical mean of observations in that bucket instead of a geometric
    ///       midpoint. This lifts the dequantize output off the ±0.5 ceiling at res=2
    ///       and into the actual input scale (e.g. a "positive volume" bucket returns
    ///       ~+40% once it's seen typical positive-volume moves).
    /// No-op for Passthrough and for dims the caller hasn't registered.
    pub fn observe(&mut self, dim_id: DimensionId, scalar: f64) {
        let state = match self.dimensions.get_mut(&dim_id) {
            Some(s) => s,
            // unregistered dims are owned by channels that bucketize on their own; ignore
            None => return,
        };
        if state.mode == QuantizeMode::Passthrough { return; }

        // dynamic: buffer and learn boundaries at warmup
        if state.mode == QuantizeMode::Dynamic {
            if let Some(ref mut samples) = state.samples {
                samples.push(scalar);

                // compute initial boundaries once we have enough warmup samples
                if state.boundaries.is_none() && samples.len() >= state.warmup_samples {
                    state.boundaries = Some(Self::compute_quantile_boundaries(samples, state.resolution));
                }
            }

            // TODO: incremental boundary refinement post-warmup (e.g. sliding window or t-digest)
        }

        // accumulate per-bucket empirical mean. Requires boundaries, so dynamic mode
        // starts contributing only after warmup — pre-warmup samples are not attributed
        // to any bucket since the bucketing itself isn't defined yet.
        let boundaries = match &state.boundaries {
            Some(b) => b,
            None => return,
        };
        if state.bucket_stats.is_none() {
            let mut stats = Vec::with_capacity(state.resolution as usize);
            for _ in 0..state.resolution { stats.push(BucketStats { count: 0, sum: 0.0 }); }
            state.bucket_stats = Some(stats);
        }
        let bucket_id = Self::bucketize(scalar, boundaries);
        let stats = &mut state.bucket_stats.as_mut().unwrap()[(bucket_id - 1) as usize];
        stats.count += 1;
        stats.sum += scalar;
    }

    /// Map a scalar to a 1-indexed bucket ID in [1..resolution].
    pub fn quantize(&self, dim_id: DimensionId, scalar: f64) -> BucketId {
        let state = match self.dimensions.get(&dim_id) {
            Some(s) => s,
            // unregistered dim: channel already emits a discrete bucket ID, pass it through
            None => return scalar as BucketId,
        };

        // input should already be an integer bucket ID - sign and magnitude don't matter,
        // only that the encoder uses integer IDs consistently across frames
        if state.mode == QuantizeMode::Passthrough { return scalar as BucketId; }

        // dynamic mode before warmup completes: place everything in the middle bucket
        // this keeps the pipeline running without creating spurious neuron coverage
        match &state.boundaries {
            None => ((state.resolution + 1) / 2) as BucketId,
            Some(b) => Self::bucketize(scalar, b),
        }
    }

    /// Map a bucket ID back to a representative scalar in the dimension's input space.
    /// Accepts a fractional bucket ID so callers can pass the weighted average of a
    /// vote distribution and get a continuous scalar prediction back.
    ///
    /// Passthrough: the bucket ID IS the scalar, returned unchanged.
    /// Static / Dynamic: looks up the empirical mean of observed samples per bucket
    /// (populated by observe()). Returns None when the bucket has never been observed
    /// — honest about the absence of data rather than fabricating a geometric midpoint.
    /// Callers must handle None (skip from weighted sums, skip from MAPE, etc.).
    pub fn dequantize(&self, dim_id: DimensionId, bucket_id: f64) -> Option<f64> {
        let state = match self.dimensions.get(&dim_id) {
            // unregistered dims come from channels that still own their own bucketization;
            // treat the bucket ID as the scalar (same as passthrough mode)
            None => return Some(bucket_id),
            Some(s) => s,
        };

        if state.mode == QuantizeMode::Passthrough { return Some(bucket_id); }

        // dynamic mode before warmup, or static/dynamic with no samples yet: no data
        // to produce a scalar from. None propagates through callers as "no prediction".
        if state.boundaries.is_none() { return None; }

        let reps = Self::bucket_representatives(state);
        Self::interpolate_representative(bucket_id, &reps)
    }

    /// Per-bucket representative scalars: the empirical mean of samples observed in
    /// each bucket (populated by observe()), or None for buckets we've never seen.
    /// None is load-bearing — callers use it to know when to skip a contribution
    /// rather than consuming a fabricated midpoint that biases predictions.
    fn bucket_representatives(state: &DimensionState) -> Vec<Option<f64>> {
        match &state.bucket_stats {
            None => vec![None; state.resolution as usize],
            Some(stats) => stats.iter().map(|s| {
                if s.count > 0 { Some(s.sum / s.count as f64) } else { None }
            }).collect(),
        }
    }

    /// Linear interpolation over bucket representatives. Unseen buckets carry None;
    /// if one flanking rep is None the other is used as-is, and if both are None the
    /// result is None. Clamps at the ends so out-of-range bucketIds saturate instead
    /// of extrapolating.
    fn interpolate_representative(bucket_id: f64, reps: &[Option<f64>]) -> Option<f64> {
        let n = reps.len();
        if bucket_id <= 1.0 { return reps[0]; }
        if bucket_id >= n as f64 { return reps[n - 1]; }
        let lo = bucket_id.floor() as usize;
        let frac = bucket_id - lo as f64;
        let a = reps[lo - 1];
        let b = reps[lo];
        match (a, b) {
            (None, None) => None,
            (None, Some(v)) => Some(v),
            (Some(v), None) => Some(v),
            (Some(va), Some(vb)) => Some(va * (1.0 - frac) + vb * frac),
        }
    }

    /// Standard boundary comparison: returns 1-indexed bucket.
    /// Matches the semantics of the original discretizeChange() in stock.js.
    fn bucketize(value: f64, boundaries: &[f64]) -> BucketId {
        for (i, &boundary) in boundaries.iter().enumerate() {
            if value <= boundary { return (i + 1) as BucketId; }
        }
        (boundaries.len() + 1) as BucketId
    }

    /// Compute equal-frequency (quantile) boundaries from a sample buffer.
    /// Produces resolution-1 split points that divide samples into equal-count buckets.
    fn compute_quantile_boundaries(samples: &[f64], resolution: u32) -> Vec<f64> {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut boundaries = Vec::new();
        for i in 1..resolution {
            let idx = (sorted.len() * i as usize) / resolution as usize;
            boundaries.push(sorted[idx.min(sorted.len() - 1)]);
        }
        boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_quantize() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 10, QuantizeMode::Passthrough, None, None);
        assert_eq!(q.quantize(0, 5.0), 5);
        assert_eq!(q.quantize(0, 1.0), 1);
    }

    #[test]
    fn test_passthrough_dequantize() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 10, QuantizeMode::Passthrough, None, None);
        assert_eq!(q.dequantize(0, 5.0), Some(5.0));
    }

    #[test]
    fn test_static_quantize() {
        let mut q = Quantizer::new();
        // 3 buckets: [-inf, -0.5], (-0.5, 0.5], (0.5, +inf]
        q.register_dimension(0, 3, QuantizeMode::Static, Some(vec![-0.5, 0.5]), None);
        assert_eq!(q.quantize(0, -1.0), 1);
        assert_eq!(q.quantize(0, -0.5), 1); // boundary inclusive
        assert_eq!(q.quantize(0, 0.0), 2);
        assert_eq!(q.quantize(0, 0.5), 2); // boundary inclusive
        assert_eq!(q.quantize(0, 1.0), 3);
    }

    #[test]
    fn test_static_dequantize_with_observations() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 3, QuantizeMode::Static, Some(vec![-0.5, 0.5]), None);

        // observe some values
        q.observe(0, -1.0); // bucket 1
        q.observe(0, -0.8); // bucket 1
        q.observe(0, 0.1);  // bucket 2
        q.observe(0, 0.3);  // bucket 2

        // dequantize should return empirical means
        let rep1 = q.dequantize(0, 1.0).unwrap();
        assert!((rep1 - (-0.9)).abs() < 1e-10); // mean of -1.0, -0.8

        let rep2 = q.dequantize(0, 2.0).unwrap();
        assert!((rep2 - 0.2).abs() < 1e-10); // mean of 0.1, 0.3

        // bucket 3 never observed
        assert!(q.dequantize(0, 3.0).is_none());
    }

    #[test]
    fn test_dynamic_warmup() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 3, QuantizeMode::Dynamic, None, Some(5));

        // before warmup: returns middle bucket
        assert_eq!(q.quantize(0, 42.0), 2);

        // feed samples
        for v in &[1.0, 2.0, 3.0, 4.0, 5.0] { q.observe(0, *v); }

        // after warmup: should have boundaries and quantize properly
        assert!(q.quantize(0, 1.0) >= 1);
        assert!(q.quantize(0, 5.0) <= 3);
    }

    #[test]
    fn test_unregistered_dim_passthrough() {
        let q = Quantizer::new();
        // unregistered dim: treat scalar as bucket ID
        assert_eq!(q.quantize(99, 7.0), 7);
        assert_eq!(q.dequantize(99, 7.0), Some(7.0));
    }

    #[test]
    fn test_has() {
        let mut q = Quantizer::new();
        assert!(!q.has(0));
        q.register_dimension(0, 5, QuantizeMode::Passthrough, None, None);
        assert!(q.has(0));
    }

    #[test]
    fn test_interpolate_fractional_bucket() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 3, QuantizeMode::Static, Some(vec![-0.5, 0.5]), None);

        q.observe(0, -1.0); // bucket 1
        q.observe(0, 0.0);  // bucket 2
        q.observe(0, 1.0);  // bucket 3

        // fractional bucket: interpolate between bucket 1 (-1.0) and bucket 2 (0.0)
        let rep = q.dequantize(0, 1.5).unwrap();
        assert!((rep - (-0.5)).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "resolution must be >= 2")]
    fn test_invalid_resolution() {
        let mut q = Quantizer::new();
        q.register_dimension(0, 1, QuantizeMode::Static, Some(vec![]), None);
    }

    #[test]
    fn test_bucketize() {
        // 4 buckets with 3 boundaries
        let boundaries = vec![10.0, 20.0, 30.0];
        assert_eq!(Quantizer::bucketize(5.0, &boundaries), 1);
        assert_eq!(Quantizer::bucketize(10.0, &boundaries), 1); // boundary inclusive
        assert_eq!(Quantizer::bucketize(15.0, &boundaries), 2);
        assert_eq!(Quantizer::bucketize(25.0, &boundaries), 3);
        assert_eq!(Quantizer::bucketize(35.0, &boundaries), 4);
    }

    #[test]
    fn test_compute_quantile_boundaries() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let boundaries = Quantizer::compute_quantile_boundaries(&samples, 4);
        // 4 buckets → 3 boundaries at ~25th, 50th, 75th percentiles
        assert_eq!(boundaries.len(), 3);
    }
}
