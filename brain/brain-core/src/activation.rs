//! Open activations (D9), what fired them, the relations their neurons returned at them, and what they infer
//! (R45).

use crate::function::FuncId;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ActRef {
    pub frame: Frame,
    pub index: u32,
}

/// A relation the activation's neuron returned at it (R21): the other activation, at its offset from this one,
/// with its strength.
#[derive(Clone, Copy, Debug)]
pub struct BoardRel {
    pub other: ActRef,
    pub off: Offset,
    pub strength: u32,
}

/// Which function fired an interneuron's activation, and which of its relations did not come true (§6.4).
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub func: FuncId,
    pub failed: Vec<usize>,
}

/// One standing inference (R45): a connection as it was read at age 0.
#[derive(Clone, Copy, Debug)]
pub struct Inference {
    pub target: NeuronId,
    pub offset: Offset,
    pub strength: f64,
    pub estimate: f64,
    /// Struck by the match (R49): it no longer stands.
    pub struck: bool,
}

/// An open activation as the machine holds it (D9).
#[derive(Clone, Debug)]
pub struct Act {
    pub neuron: NeuronId,
    pub kind: Kind,
    pub axes: u8,
    pub pos: Pos,
    pub strong: bool,
    pub reach: Pos,
    /// The box of places it stands for in space (D4): its own place for a base activation, and for a call the
    /// union of what it covered.
    pub lo: Pos,
    pub hi: Pos,
    /// The age at which an accepted call covered it (D9, D10).
    pub covered_at: Option<u32>,
    pub coverer: Option<ActRef>,
    /// What its call covered (§6.4): the activations its fixed nodes landed on and the calls it landed on.
    pub covers: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    /// The relations its neuron returned at it (R21), the board's edges at this node (D55).
    pub rels: Vec<BoardRel>,
    pub inferences: Vec<Inference>,
    /// The frame it fired in.
    pub opened_at: Frame,
    /// The level of its frame it fired at (R26): 0 for what the frame reported.
    pub level: u32,
    /// How many functions deep it stands, a diagnostic the machine never reads.
    pub height: u32,
    /// A weak activation the report replaced (D40): no longer there for anything.
    pub gone: bool,
}

impl Act {
    pub fn on_apex(&self) -> bool { self.covered_at.is_none() && !self.gone }
}

#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub no: Frame,
    pub acts: Vec<Act>,
    /// The base actions that ran this frame, per channel (D24).
    pub actions_ran: Vec<(ChannelId, ActRef)>,
    pub rewards: Vec<Reward>,
}
