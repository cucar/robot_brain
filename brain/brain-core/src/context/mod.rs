/// Context — observed co-activation / sequence structure that patterns match against.
///
/// Two distinct types because spatial and temporal have fundamentally different shapes:
///
/// * `SpatialContext` ([spatial]) is a flat set of neurons at the current frame with strengths.
///   No distance dimension — spatial is same-frame co-activation.
/// * `TemporalContext` ([temporal]) is a set of neurons at past distances with strengths.
///   Distance is the temporal recency: how many frames ago each context neuron fired.
///
/// Both are used in two roles:
/// 1. **Observed context** — built fresh each frame from the active set at that level.
/// 2. **Known context** — stored in neuron routing tables; what a pattern needs to see to match.
///
/// Both `match_observed` implementations score the same way: the Jaccard ratio
/// `common / (common + missing + novel)`, so a pattern that explains only a fraction of what is
/// active (or claims entries that are absent) cannot over-fire. The grouping operation is identical
/// across spatial and temporal; only the storage shape (flat vs distance-keyed) differs.

mod spatial;
mod temporal;

pub use spatial::SpatialContext;
pub use temporal::TemporalContext;
