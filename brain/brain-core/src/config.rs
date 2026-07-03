//! Runtime toggles for the experimental spatial refinement mechanisms, read once per process from the
//! environment, plus diagnostic traces.
//!
//! These gate the spatial pattern-refinement paths so the training harness can run controlled ablations
//! without threading config through every construction layer (neuron ← column ← region ← thalamus ←
//! brain ← napi).
//! Each flag is read once via `OnceLock` on first access and cached for the process lifetime, so one
//! training run has a single fixed configuration — compare settings by running separate processes.
//! Defaults preserve current behavior (refinement ON, traces OFF), so a run with no flags set is unchanged.

use std::sync::OnceLock;

/// Read a boolean env flag: "0" / "false" / "off" / "no" → false, any other value → true, absent → `default`.
fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.as_str(), "0" | "false" | "off" | "no"),
        Err(_) => default,
    }
}

/// Read an integer env value, falling back to `default` when absent or unparseable.
fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or(default),
        Err(_) => default,
    }
}

/// MATURITY GATE for spatial correction minting: a neuron may not mint error-correction patterns until
/// its spatial error stats hold at least this many samples. Until then it only accumulates statistics
/// (error samples are recorded regardless of the mint decision) and keeps recognizing normally.
/// Rationale: "surprise" is only meaningful against an established distribution — newborn neurons
/// minting against 1-2 exposures of noise produce write-once corrections that never re-fire.
/// 0 disables the gate (previous behavior). Spatial only; temporal minting is untouched.
pub fn mint_min_samples() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| env_u64("BRAIN_MINT_MIN_SAMPLES", 10))
}

/// EXPERIMENTAL per-entry match statistics for spatial recognition: candidates are matched with no
/// shared threshold; the score-best candidate records its Jaccard ratio into its routing entry's match
/// stats and activates only if the ratio clears the entry's own bar (mean of its distribution, 0.5 until
/// warm). Default OFF = legacy recognition (shared adaptive merge threshold from the host's error stats).
/// Best measured form of the experiment; kept behind this flag after it traded too much selectivity
/// for its (large) compression — see the wavefront branch notes.
pub fn match_stats_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_MATCH_STATS", false))
}

/// MATCH-ALL spatial recognition: no match threshold at all — the best candidate (by score) always
/// fires, regardless of how well it matches. Recognition never rejects, so the vocabulary only grows
/// while a parent has no children yet. Experimental: tests whether refinement alone can keep pattern
/// identities discriminative without any rejection mechanism. Default OFF.
pub fn match_all_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_MATCH_ALL", false))
}

/// STATIC spatial match threshold: when set, legacy recognition uses this fixed merge threshold
/// instead of the adaptive coupling (1 − error threshold) — DECOUPLING recognition strictness from
/// the error side, which keeps its dynamic grouping for correction minting untouched. Unset = coupled.
pub fn match_threshold_override() -> Option<f64> {
    static V: OnceLock<Option<f64>> = OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_MATCH_THRESHOLD").ok().and_then(|v| v.parse().ok()))
}

/// INFORMATION-WEIGHTED spatial recognition (MDL): candidates are scored in BITS instead of entry
/// counts. Each context entry is weighted by its surprisal under the host's local activation
/// frequencies — common entries contribute -log2(p), missing entries cost -log2(1-p), and the winner
/// must SAVE bits net of a naming cost (no threshold at all: acceptance = "does this fold compress?").
/// Novel entries drop out of the criterion — a pattern is not punished for what its siblings explain.
/// Default OFF.
pub fn match_info_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_MATCH_INFO", false))
}

/// LIKELIHOOD-RATIO spatial recognition (MDL v2): candidates are scored as a hypothesis test — for
/// each stored context entry, log2 of the ratio between its probability under the candidate (from the
/// entry's per-fire co-occurrence counts) and under the background model (the host's local marginal
/// frequencies). Present entries credit log(p(e|C)/p(e|bg)); absent entries cost
/// log((1-p(e|C))/(1-p(e|bg))); novel entries cancel (siblings encode them). Acceptance: score > 0 —
/// the candidate must explain the observation better than "no pattern". No thresholds. Default OFF.
pub fn match_info2_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_flag("BRAIN_MATCH_INFO2", false))
}

/// Spatial CONTEXT (sources) refinement — consolidating a matched pattern's stored context toward the
/// observed configuration in recognition. Default ON.
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
