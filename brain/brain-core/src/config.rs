//! Runtime ablation toggles for experimental spatial mechanisms, read once per process from the environment.
//!
//! These gate the spatial neuron-reuse and pattern-refinement paths so the training harness can run
//! controlled ablations without threading config through every construction layer (neuron ← column ←
//! region ← thalamus ← brain ← napi).
//! Each flag is read once via `OnceLock` on first access and cached for the process lifetime, so one
//! training run has a single fixed configuration — compare settings by running separate processes.
//! Defaults preserve current behavior (every mechanism ON), so a run with no flags set is unchanged.

use std::sync::OnceLock;

/// Read a boolean env flag: "0" / "false" / "off" / "no" → false, any other value → true, absent → `default`.
fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.as_str(), "0" | "false" | "off" | "no"),
        Err(_) => default,
    }
}

/// Spatial neuron reuse — the Phase C cross-frame reuse lookup in the mint path. Default ON.
/// When off, every correction request mints fresh (no reuse), isolating reuse's effect on the neuron count.
pub fn reuse_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_REUSE", true))
}

/// Spatial CONTEXT (sources) refinement — consolidating a matched pattern's inputs in recognition. Default ON.
pub fn refine_context_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_REFINE_CONTEXT", true))
}

/// Spatial CONNECTION (targets) refinement — pruning a pattern's unobserved predictions in connection
/// learning. Default ON.
pub fn refine_connection_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_REFINE_CONNECTION", true))
}

/// Diagnostic: when set, spatial `match_observed` prints its common/missing/novel/ratio/threshold per call
/// (to stderr). Off by default. Run over a tiny number of frames — it prints one line per candidate match.
pub fn trace_match() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_TRACE_MATCH", false))
}

/// Diagnostic: when set, spatial context refinement prints the pattern id, its resulting context size, and
/// the pixels added (novel) vs removed (missing) on that refinement. Shows whether a context inflates
/// toward a union blur or contracts toward a specific core over exposures. Off by default (to stderr).
pub fn trace_refine() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_TRACE_REFINE", false))
}

/// Diagnostic: when set, the spatial mint decision prints the predicted/observed L0 sizes, the measured
/// prediction error rate, and the adaptive error threshold it is compared against (per active neuron).
/// Shows whether errors cluster just under the adaptive threshold. Off by default (to stderr).
pub fn trace_error() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_TRACE_ERROR", false))
}

/// Diagnostic: when set, the reuse lookup prints per request the number of candidates found, the best
/// Jaccard achieved (even below threshold) with the winning candidate's coverage and connection count,
/// and the threshold — so we can see whether reuse misses for lack of candidates or lack of overlap.
pub fn trace_reuse() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_TRACE_REUSE", false))
}
