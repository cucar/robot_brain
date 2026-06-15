/// Shared type aliases and common structures used across the brain.
///
/// All cross-module references use integer IDs rather than pointers.
/// This keeps ownership simple and mirrors the JS implementation.

/// Unique identifier for a neuron (sensory or pattern).
pub type NeuronId = u64;

/// Unique identifier for a channel (e.g. a stock ticker, a text stream).
pub type ChannelId = u32;

/// Unique identifier for a dimension within a channel (e.g. "close", "volume").
pub type DimensionId = u32;

/// Bucket index within a dimension. Signed because action dimensions use negative
/// values (e.g. -1 for "OUT"). Event buckets are 1-indexed: [1..=resolution].
pub type BucketId = i32;

/// Temporal distance in the context window (0 = current frame, 1 = one frame ago, etc.).
pub type Distance = u32;

/// Hierarchical level (0 = sensory, 1+ = pattern).
pub type Level = u32;

/// Frame number — signed so context restore can use negative activation
/// frames (representing neurons activated before the current frame 0).
pub type FrameNumber = i64;

/// Connection strength (f64 to match JS behavior).
pub type Strength = f64;

/// Reward signal from the environment.
pub type Reward = f64;

/// A coordinate identifies a specific sensory neuron: (dimension, bucket).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Coordinate {
    pub dim_id: DimensionId,
    pub bucket_id: BucketId,
}

/// The type of a sensory neuron — either an observable event or a choosable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NeuronType {
    Event,
    Action,
}

/// Data stored per connection: how strongly neuron A predicts neuron B,
/// and the reward signal observed when this prediction fired.
#[derive(Debug, Clone)]
pub struct ConnectionData {
    pub strength: Strength,
    pub reward: Reward,
}

/// Result of matching a context against observed active neurons.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Overall match quality score.
    pub score: f64,
    /// Context entries that matched observed neurons.
    pub common: Vec<ContextEntry>,
    /// Context entries not found in observed neurons.
    pub missing: Vec<ContextEntry>,
    /// Observed neurons not in the context (potential additions).
    pub novel: Vec<ContextEntry>,
}

/// A single entry in a context: a neuron at a specific distance with a strength.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub neuron_id: NeuronId,
    pub distance: Distance,
    pub strength: Strength,
}

/// Welford online statistics for incremental mean/variance calculation.
/// Used for dynamic error thresholds per (neuron, age).
#[derive(Debug, Clone)]
pub struct WelfordState {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordState {
    pub fn new() -> Self {
        Self { n: 0, mean: 0.0, m2: 0.0 }
    }

    /// Record a new observation, updating running statistics.
    pub fn update(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance (returns 0.0 if fewer than 2 samples).
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Processing phase — distinguishes spatial (d=0 co-activation) from temporal (d>0 sequence) work.
/// Both phases use the same per-neuron prediction/error/correction mechanism; they differ only in
/// which connection-distance slot they read and which level index in Memory they iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Spatial,
    Temporal,
}

/// Consensus mode — determines how the per-voter action posteriors are combined into a winner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusMode {
    /// Strength-weighted arithmetic mean of the per-voter posteriors, argmax over the dimension.
    /// A soft ensemble: a candidate can win on a good average even when several voters contradict it.
    Democratic,
    /// Naive-Bayes product of the per-voter posteriors: argmax over Σ_voter log(P|voter + NB_EPS).
    /// Each voter's near-zero posterior acts as a veto, the correct rule for argmax over
    /// mutually-exclusive classes with roughly-independent evidence.
    Nb,
}


/// Error correction mode — determines how error thresholds are calculated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorMode {
    /// Fixed threshold (user-supplied value).
    Static,
    /// mean + 1 standard deviation (fewer corrections).
    Conservative,
    /// mean (moderate corrections).
    Neutral,
    /// mean - 1 standard deviation (more corrections).
    Aggressive,
}
