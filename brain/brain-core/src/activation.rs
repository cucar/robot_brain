//! Open activations (D9), what fired them, and what they infer (R45).

use crate::function::*;
use crate::neuron::*;
use crate::types::*;

/// Which function fired an activation of an interneuron, and how it fell (§6.4, D37): where each relation
/// landed, which did not come true, and the values its variables took.
#[derive(Clone, Debug)]
pub struct FiredBy {
    pub func: FuncId,
    /// Per relation that came true, the activation it landed on.
    pub landed: Vec<(usize, ActRef)>,
    /// The relations that did not come true (D12).
    pub failed: Vec<usize>,
    /// Per relation through a variable, the neuron bound there (D37).
    pub args: Vec<(usize, NeuronId)>,
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
    /// What it costs in the file as it stands (D13): its record, its neuron alone once bound.
    pub cost: f64,
    /// Its record as it fired (D12): `d + 2` bare, its call's record above.
    pub full: f64,
    pub strong: bool,
    pub span: Pos,
    pub reach: Pos,
    pub covered_at: Option<u32>,
    /// The age at which an accepted function bound it in a variable (D10, D24): credited, and still on the apex.
    pub bound_at: Option<u32>,
    pub coverer: Option<ActRef>,
    /// Later calls that took it back with everything its coverer covered, each at the age it happened (R23).
    pub retaken: Vec<(u32, ActRef)>,
    /// What its call covered and bound (§6.4).
    pub covers: Vec<ActRef>,
    pub binds: Vec<ActRef>,
    pub fired_by: Option<FiredBy>,
    pub inferences: Vec<Inference>,
    /// The frame it fired in.
    pub opened_at: Frame,
    /// The level of its frame it fired at (R26): 0 for what the frame reported.
    pub level: u32,
    /// How many functions deep it stands, a diagnostic the machine never reads.
    pub height: u32,
    /// A weak activation the report replaced (D40): no longer there for anything.
    pub gone: bool,
    /// Frames it has been handed by connect calls so far (§7.1).
    pub caught_up: Frame,
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
