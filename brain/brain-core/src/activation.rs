//! Open activations (D9), what fired them, and what they infer (R45).


use crate::neuron::*;
use crate::types::*;

/// Which function fired an activation, and its arguments: per variable place, the neuron bound there and
/// its activation (§6.4, D37).
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub owner: NeuronId,
    pub pattern: PatternId,
    pub args: Vec<(usize, NeuronId, ActRef)>,
    /// The places that failed (D22): nothing of them was covered or bound, so expansion places nothing there.
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
    /// What its record costs (D12, D13).
    pub cost: f64,
    /// The part of `cost` that is its position: what a variable binding it covers (D10).
    pub pos_bits: f64,
    pub strong: bool,
    pub span: Pos,
    pub reach: Pos,
    pub covered_at: Option<u32>,
    /// The age at which an accepted bid bound it in a variable (D10, D24): credited, and still on the apex.
    pub bound_at: Option<u32>,
    pub coverer: Option<ActRef>,
    pub covers: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    pub inferences: Vec<Inference>,
    /// The frame it fired in, which is later than its coordinate for a child placed frames back (§6.4).
    pub opened_at: Frame,
    /// The level of its frame it fired at (R26): 0 for what the frame reported.
    pub level: u32,
    /// How many patterns deep it stands, a diagnostic the machine never reads.
    pub height: u32,
    /// A weak activation the report replaced (D40): no longer there for anything.
    pub gone: bool,
    /// Frames it has been handed by connect calls so far; a child placed frames back catches up (§7.1).
    pub caught_up: Frame,
}

#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub no: Frame,
    pub acts: Vec<Act>,
    /// The base actions that ran this frame, per channel (D24).
    pub actions_ran: Vec<(ChannelId, ActRef)>,
    pub rewards: Vec<Reward>,
}

