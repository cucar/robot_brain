/// Shared types for the UCAR machine (docs/algorithm.md).

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use rustc_hash::FxHashSet;

/// A position: one number per activation dimension, time first (D1).
///
/// It has no fixed number of dimensions: a dimension it does not hold reads as zero.
/// Time is held in the position itself.
/// The numbers of space are held once for all the positions that share them, with the zeros after the last number
/// that is not zero left off.
#[derive(Clone, Copy)]
pub struct Pos {
    time: i32,
    space: &'static [i32],
}

/// An offset between two positions, exact in every dimension (D6).
pub type Offset = Pos;

/// The numbers of space every position shares: each list is kept once, for the life of the process.
static SPACES: OnceLock<RwLock<FxHashSet<&'static [i32]>>> = OnceLock::new();

/// The one kept copy of a list of numbers, the zeros at its end left off.
fn share_numbers(numbers: &[i32]) -> &'static [i32] {
    let mut held = numbers.len();
    while held > 0 && numbers[held - 1] == 0 { held -= 1; }
    let numbers = &numbers[..held];
    if numbers.is_empty() { return &[]; }
    let table = SPACES.get_or_init(|| RwLock::new(FxHashSet::default()));
    if let Some(kept) = table.read().unwrap().get(numbers) { return kept; }
    let mut table = table.write().unwrap();
    if let Some(kept) = table.get(numbers) { return kept; }
    let kept: &'static [i32] = Box::leak(numbers.to_vec().into_boxed_slice());
    table.insert(kept);
    kept
}

static ZERO: i32 = 0;

impl Pos {
    /// The position with zero in every dimension.
    pub fn zero() -> Pos { Pos { time: 0, space: &[] } }

    /// A position from its numbers, time first.
    pub fn from_slice(numbers: &[i32]) -> Pos {
        match numbers.split_first() { Some((time, space)) => Pos { time: *time, space: share_numbers(space) }, None => Pos::zero() }
    }

    /// A position of `count` dimensions, each number given by its dimension.
    pub fn make(count: usize, number: impl Fn(usize) -> i32) -> Pos {
        // the numbers are gathered on the stack where they are few, so making a position allocates nothing
        if count <= 8 {
            let mut numbers = [0i32; 8];
            for d in 0..count { numbers[d] = number(d); }
            return Pos::from_slice(&numbers[..count]);
        }
        let numbers: Vec<i32> = (0..count).map(number).collect();
        Pos::from_slice(&numbers)
    }

    /// How many dimensions it holds: time, and space up to its last number that is not zero.
    pub fn len(&self) -> usize { 1 + self.space.len() }

    /// Its numbers, time first, to `count` dimensions.
    pub fn to_vec(&self, count: usize) -> Vec<i32> { (0..count).map(|d| self[d]).collect() }

    /// The same position with one dimension set.
    pub fn with(&self, dimension: usize, number: i32) -> Pos {
        if dimension == 0 { return Pos { time: number, space: self.space }; }
        Pos::make(self.len().max(dimension + 1), |d| if d == dimension { number } else { self[d] })
    }

    /// This position moved by an offset.
    pub fn plus(&self, other: &Pos) -> Pos {
        if other.space.is_empty() { return Pos { time: self.time + other.time, space: self.space }; }
        Pos::make(self.len().max(other.len()), |d| self[d] + other[d])
    }

    /// The offset from another position to this one.
    pub fn minus(&self, other: &Pos) -> Pos {
        if other.space.is_empty() { return Pos { time: self.time - other.time, space: self.space }; }
        Pos::make(self.len().max(other.len()), |d| self[d] - other[d])
    }

    /// The opposite offset.
    pub fn negated(&self) -> Pos {
        if self.space.is_empty() { return Pos { time: -self.time, space: self.space }; }
        Pos::make(self.len(), |d| -self[d])
    }

    /// The lesser of the two in each dimension.
    pub fn least(&self, other: &Pos) -> Pos { Pos::make(self.len().max(other.len()), |d| self[d].min(other[d])) }

    /// The greater of the two in each dimension.
    pub fn most(&self, other: &Pos) -> Pos { Pos::make(self.len().max(other.len()), |d| self[d].max(other[d])) }

    /// Whether every dimension is zero.
    pub fn is_zero(&self) -> bool { self.time == 0 && self.space.is_empty() }

    /// The order of two positions in space alone, dimension by dimension in the order they are declared.
    pub fn compare_space(&self, other: &Pos) -> Ordering {
        for d in 1..self.len().max(other.len()) {
            let order = self[d].cmp(&other[d]);
            if order != Ordering::Equal { return order; }
        }
        Ordering::Equal
    }
}

impl std::ops::Index<usize> for Pos {
    type Output = i32;
    fn index(&self, dimension: usize) -> &i32 {
        if dimension == 0 { return &self.time; }
        self.space.get(dimension - 1).unwrap_or(&ZERO)
    }
}

impl PartialEq for Pos {
    fn eq(&self, other: &Pos) -> bool { self.time == other.time && (std::ptr::eq(self.space, other.space) || self.space == other.space) }
}

impl Eq for Pos {}

impl Hash for Pos {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.time.hash(state);
        self.space.hash(state);
    }
}

impl Ord for Pos {
    fn cmp(&self, other: &Pos) -> Ordering { self.time.cmp(&other.time).then_with(|| self.compare_space(other)) }
}

impl PartialOrd for Pos {
    fn partial_cmp(&self, other: &Pos) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl std::fmt::Debug for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_list().entries((0..self.len()).map(|d| self[d])).finish() }
}

/// The activation dimensions a channel is laid out over (D3): time always, and the dimensions of space it names.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Axes {
    space: &'static [i32],
}

impl Axes {
    /// Time alone.
    pub fn time() -> Axes { Axes { space: &[] } }

    /// Time and the named dimensions of space.
    pub fn from_dims(dimensions: &[usize]) -> Axes {
        let mut space: Vec<i32> = dimensions.iter().filter(|d| **d > 0).map(|d| *d as i32).collect();
        space.sort_unstable();
        space.dedup();
        Axes { space: share_numbers(&space) }
    }

    /// Whether it is laid out over a dimension.
    pub fn has(&self, dimension: usize) -> bool { dimension == 0 || self.space.contains(&(dimension as i32)) }

    /// Its dimensions of space, in the order they are declared.
    pub fn space(&self) -> &'static [i32] { self.space }

    /// One more than its last dimension: how many dimensions a position laid out over it holds at most.
    pub fn count(&self) -> usize { 1 + self.space.last().map_or(0, |d| *d as usize) }
}

/// Guards against a loop that never ends: where one is passed, the machine stops with a message.
/// The levels one frame climbs.
pub const MAX_LEVELS: u32 = 1024;
/// The laps one fit reads, the laps a call is followed through, and the laps an inferred cycle is read back.
pub const MAX_LAPS: u32 = 4096;
/// A function names a second and further node of one neuron by a symbol of its own (D15): it is kept in the table of
/// poses under this mark and its number, in the place of the laps, and has no phase.
pub const NODE_MARK: u32 = 1 << 31;
/// The calls standing over one activation.
pub const MAX_CALLS_OVER: usize = 4096;
/// The calls one under another, followed up from an activation or down through an expansion.
pub const MAX_CALL_DEPTH: usize = 256;

pub type NeuronId = u32;
pub type DimensionId = u32;
pub type ChannelId = u32;
pub type FrameNumber = i64;

/// A pose: a neuron in a phase (D8, D34). A base neuron has one phase, so its symbol is its id; a call's is
/// allotted from `SYM_CALL` up, one per interneuron and phase.
pub type Sym = u32;
pub const SYM_CALL: Sym = 1 << 28;

/// A call's phase (D8, D37, D53): the node the fit started at, the node it ended at, and its laps, how many times it
/// went round. A fit reads a cycle one way, the way its relations' offsets advance, so the phase names no
/// direction. The phase names its two nodes, each a node of the function (D15). So a phase is two nodes and a
/// number, and a ring of calls, each ending at the call before it, keeps the same names lap after lap.
/// A neuron in a phase is a pose.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
pub struct Phase {
    pub from: Sym,
    pub to: Sym,
    pub laps: u32,
}

/// Whether position `a` is later than position `b` (D15): later in time, and in one frame at the lesser position in the
/// order the dimensions are declared.
pub fn later(a: &Pos, b: &Pos) -> bool {
    if a[0] != b[0] { return a[0] > b[0]; }
    a.compare_space(b) == Ordering::Less
}

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
    pub id: DimensionId,
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
    pub event: Option<DimensionId>,
    pub action: Option<DimensionId>,
    /// The activation dimensions it is laid out over.
    pub axes: Axes,
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
    pub dim: DimensionId,
    pub bucket: u32,
    pub pos: Pos,
}
