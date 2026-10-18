//! Shared types for the UCAR machine (docs/algorithm.md).

/// Time plus up to two spatial activation dimensions. Index 0 is always time (D1).
pub const MAX_DIMS: usize = 3;

/// A coordinate: frame in slot 0, spatial positions after it (D2).
pub type Pos = [i32; MAX_DIMS];
/// A rounded offset between two coordinates (D6).
pub type Offset = [i32; MAX_DIMS];

pub type NeuronId = u32;
pub type PatternId = u32;
pub type DimId = u32;
pub type ChannelId = u32;
pub type Frame = i64;

/// The three kinds the neighborhood policy distinguishes (D50).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Event = 0,
    Action = 1,
    Inter = 2,
}

/// A neuron dimension: an event or action alphabet of one channel (D1).
#[derive(Clone, Debug)]
pub struct NeuronDim {
    pub id: DimId,
    pub name: String,
    pub channel: ChannelId,
    pub kind: Kind,
    pub resolution: u32,
    /// Id of the base neuron for bucket 0; bucket b is `first + b`.
    pub first: NeuronId,
}

/// An activation dimension: an axis with its radius (D1, R4).
#[derive(Clone, Debug)]
pub struct ActDim {
    pub name: String,
    pub radius: i32,
}

/// A channel: its neuron dimensions and the axes it is laid out over (D3).
#[derive(Clone, Debug)]
pub struct Channel {
    pub id: ChannelId,
    pub name: String,
    pub event: Option<DimId>,
    pub action: Option<DimId>,
    /// Bitmask over activation dimensions; bit 0 (time) is always set.
    pub axes: u8,
}

/// What an action dimension outputs when every candidate is negative (D51).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionPolicy {
    LeastBad,
    Nothing,
}

/// The neighborhood policy (D50) and the action policy (D51).
#[derive(Clone, Debug)]
pub struct Policy {
    /// past[holder][target]: may a target of that kind be a neighbor of a holder of that kind.
    pub past: [[bool; 3]; 3],
    /// future[holder][target]: may a holder of that kind connect to a target of that kind.
    pub future: [[bool; 3]; 3],
    pub action: ActionPolicy,
}

impl Policy {
    pub fn all_on() -> Self {
        Policy { past: [[true; 3]; 3], future: [[true; 3]; 3], action: ActionPolicy::LeastBad }
    }
    /// The stocks/MNIST policy: no action is anyone's neighbor; only events and interneurons
    /// connect, and only to actions (algorithm-stocks.md §1).
    pub fn events_to_actions() -> Self {
        let mut past = [[true; 3]; 3];
        for k in 0..3 {
            past[k][Kind::Action as usize] = false;
            past[Kind::Action as usize][k] = false;
        }
        let mut future = [[false; 3]; 3];
        future[Kind::Event as usize][Kind::Action as usize] = true;
        future[Kind::Inter as usize][Kind::Action as usize] = true;
        Policy { past, future, action: ActionPolicy::LeastBad }
    }
}

/// log2 that prices a one-option alphabet at zero bits (D13).
#[inline]
pub fn log2(x: f64) -> f64 {
    if x <= 1.0 { 0.0 } else { x.log2() }
}

/// A reward as the world reports it (D35).
#[derive(Clone, Debug)]
pub struct Reward {
    pub reward: f64,
    /// Action channels it pays; empty means every channel.
    pub channels: Vec<ChannelId>,
    /// Span of frames, counted back from the arriving frame; 0 means the longest window.
    pub frames: u32,
}

/// One symbol reported by the world in a frame (§2.3).
#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub dim: DimId,
    pub bucket: u32,
    pub pos: Pos,
}
