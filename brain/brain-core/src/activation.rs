//! Open activations (D9), what fired them, and what they infer (R45).


use crate::neuron::*;
use crate::types::*;

/// Which pattern fired an activation and what its variables held (§6.4).
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub owner: NeuronId,
    pub pattern: PatternId,
    /// Per variable named: (variable, member, value neuron).
    pub values: Vec<(PatternId, NeuronId, Option<NeuronId>)>,
}

/// One standing inference (R45): a connection as it was read at age 0.
#[derive(Clone, Copy, Debug)]
pub struct Inference {
    pub target: NeuronId,
    pub offset: Offset,
    pub strength: f64,
    pub estimate: f64,
}

/// An open activation as the machine holds it (D9).
#[derive(Clone, Debug)]
pub struct Act {
    pub neuron: NeuronId,
    pub kind: Kind,
    pub axes: u8,
    pub pos: Pos,
    pub cost: f64,
    pub strong: bool,
    pub span: Pos,
    pub reach: Pos,
    pub covered_at: Option<u32>,
    pub coverer: Option<ActRef>,
    pub covers: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    pub inferences: Vec<Inference>,
    /// The frame it fired in, which is later than its coordinate for a child placed frames back (§6.4).
    pub opened_at: Frame,
    pub wave: u32,
    pub height: u32,
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

