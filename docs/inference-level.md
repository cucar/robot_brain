# Inference Scope Experiment

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Status:** Pre-implementation
**Prerequisite:** [mnist-merge.md](./mnist-merge.md)
**Next:** [spatial-processing.md](./spatial-processing.md)

---

## Why

Today, every neuron's connections target only L=0 sensory neurons. This is because `thalamus.dispatch_frame` builds `new_active_neurons` from `age0` (the L=0 sensory set). Neurons at any level form their connections to sensory neurons only.

This may not be optimal. Before locking in any downstream architecture that depends on the inference-scope rule (spatial processing, neuron reuse), we want to pick the rule empirically on a known-good workload (stocks).

The winning rule applies uniformly to all downstream work.

---

## Three Variants

| Rule | `new_active_neurons` passed to a level-L neuron processing age=d | Hypothesis |
|---|---|---|
| `base` (today) | All L=0 neurons active at age=d | Cheapest; current behavior |
| `same-level` | L-resident neurons active at age=d | Cleaner hierarchy; each level predicts within its own abstraction |
| `all-levels` | All neurons active at age=d, any level | Maximum reuse potential; biologically closer to cortex |

---

## Phase 1 — Inference Scope Experiment (d>0 only, stocks)

**No spatial work, no architecture refactor.** Just gate `new_active_neurons` by scope rule and measure on the existing stocks pipeline.

### Code touched

- `brain/brain-core/src/brain.rs` — add `inference_scope: InferenceScope` to `BrainOptions`, defaulting to `Base` (zero behavior change).
- `brain/brain-core/src/types.rs` — add `enum InferenceScope { Base, SameLevel, AllLevels }`.
- `brain/brain-core/src/thalamus.rs::dispatch_frame` — branch on `inference_scope` when constructing `new_active_neurons`:
  - `Base`: existing logic (active at age=0 sensory).
  - `SameLevel`: filter the active set to neurons whose `level == current_processing_level`.
  - `AllLevels`: include every neuron active at the relevant age, regardless of level.
- `brain/brain-napi/src/lib.rs` — expose `inferenceScope` on the JS-facing brain options.

### Runs

1. `base` — control. Confirm reproduces current stocks baseline.
2. `same-level` — Variant A.
3. `all-levels` — Variant B.

Same data, same seed, same hyperparameters across all three.

### Metrics

- Directional accuracy.
- Per-episode ROI.
- Total neuron count at end of run.
- Total connection count at end of run.
- Per-frame runtime (mean, p99).
- Memory footprint at end of run.

### Acceptance

- All three runs complete without crashing on the stocks workload.
- A written decision lands at the end of this doc naming the winner and the metric that drove the call.
- Loser variants remain runnable via config — we don't delete them; they may be revisited if MNIST or another workload prefers a different rule.

### Notes / gotchas

- `SameLevel` filtering needs a per-level active set; the existing code already partitions by level via `memory.get_level_neurons(level)`. Use that.
- `AllLevels` may inflate per-frame runtime substantially on stocks — that's a data point, not a failure. Measure first, optimize later if it wins.
- If `base` wins outright, we still proceed with `base` semantics. The experiment is informative either way.
- The winning rule's name propagates into [spatial-processing.md](./spatial-processing.md) (used for d=0 scoping) and conceptually into [neuron-reuse.md](./neuron-reuse.md) (reuse pool is unaffected by scope; only the prediction-set construction is).

---

## Decision

*(To be filled in after Phase 1 runs.)*

**Winner:**
**Driving metric:**
**Notes:**
