# Inference Level Experiment

This document is the **inference-scope experiment**: which neurons a level-L neuron predicts over (`base` / `same-level` / `level-below` / `all-levels`), picked empirically on the stocks pipeline.

It **gates** the action-composition and reward-distribution designs in [action-composition.md](./action-composition.md): the action tower grounds `level-below` in the same units as the event tower with the arrow reversed, so the scope rule that wins here is the default there too. If `base` wins outright — predicting raw sensory beats predicting your own substrate — "levels infer levels" is undercut and the composition design needs rethinking before it is built.

---

## Why

Today, every neuron's connections target only L=0 sensory neurons. This is because `thalamus.dispatch_frame` builds `new_active_neurons` from `age0` (the L=0 sensory set). Neurons at any level form their connections to sensory neurons only.

This may not be optimal. Before locking in any downstream architecture that depends on the inference-scope rule (spatial processing, neuron reuse), we want to pick the rule empirically on a known-good workload (stocks).

The winning rule becomes the **default** for all downstream work, subject to re-validation on MNIST (see [Scope of the decision](#scope-of-the-decision)).

---

## Four Variants

| Rule | `new_active_neurons` passed to a level-L neuron processing age=d | Hypothesis |
|---|---|---|
| `base` (today) | All L=0 neurons active at age=d | Cheapest; current behavior |
| `same-level` | L-resident neurons active at age=d | Each level predicts within its own abstraction; **bootstrapping risk** (see below) |
| `level-below` | (L−1)-resident neurons active at age=d (L=1 sees L=0) | Classic cortical hierarchy rule; compositional abstraction at modest cost |
| `all-levels` | All neurons active at age=d, any level | Maximum reuse potential; risk of correlated double-counting |

### Variant-specific risks

- **`same-level` bootstrapping:** early in training, higher levels are sparsely populated, so a new level-L neuron has an almost-empty prediction substrate. The variant may lose by construction rather than by merit. If it underperforms, note in the decision whether starvation (low connection counts at L≥1) explains it before ruling the idea out.
- **`all-levels` double-counting:** a high-level neuron and its constituent low-level neurons are correlated evidence. Including both in the prediction set may distort voting/consensus precision even as raw coverage improves.
- **`level-below`** is the prior favorite: it builds genuine cross-level composition without the redundancy of `all-levels` or the starvation of `same-level`.

---

## What this experiment actually compares

Changing the scope rule changes the **learning substrate**, not just inference: connections formed under each rule produce different neuron populations over the course of training. We are therefore comparing whole training trajectories (rule + the hierarchy it grows), not a swapped-in inference rule on a fixed brain. The decision applies to the rule-as-trained, and any post-hoc analysis should keep this in mind.

---

## Phase 1 — Inference Scope Experiment (d>0 only, stocks)

**No spatial work, no architecture refactor.** Just gate `new_active_neurons` by scope rule and measure on the existing stocks pipeline.

### Code touched

- `brain/brain-core/src/brain.rs` — add `inference_scope: InferenceScope` to `BrainOptions`, defaulting to `Base` (zero behavior change).
- `brain/brain-core/src/types.rs` — add `enum InferenceScope { Base, SameLevel, LevelBelow, AllLevels }`.
- `brain/brain-core/src/thalamus.rs::dispatch_frame` — branch on `inference_scope` when constructing `new_active_neurons`:
  - `Base`: existing logic (active at age=0 sensory).
  - `SameLevel`: filter the active set to neurons whose `level == current_processing_level`.
  - `LevelBelow`: filter the active set to neurons whose `level == current_processing_level - 1` (for L=1 this is identical to `Base` restricted to actives; levels L≥2 differ).
  - `AllLevels`: include every neuron active at the relevant age, regardless of level.
- `brain/brain-napi/src/lib.rs` — expose `inferenceScope` on the JS-facing brain options.

### Runs

1. `base` — control. Confirm reproduces current stocks baseline.
2. `same-level` — Variant A.
3. `level-below` — Variant B.
4. `all-levels` — Variant C.

Same data and same hyperparameters across all variants. **Seeds:** 3 seeds per variant if a run is cheap enough to allow it; otherwise 1 seed per variant with the noise threshold below applied strictly.

### Metrics

- Directional accuracy. **(primary)**
- Per-episode ROI.
- Total neuron count at end of run.
- Total connection count at end of run (broken out per level — needed to diagnose `same-level` starvation).
- Per-frame runtime (mean, p99).
- Memory footprint at end of run.

### Decision rule (pre-registered)

Declared **before** any run to prevent post-hoc rationalization:

1. **Primary metric: directional accuracy**, subject to a runtime budget — per-frame p99 must stay within **3×** of `base`. A variant that exceeds the budget can only win if its accuracy gain is large enough that we'd commit to optimizing it (note this explicitly in the decision).
2. **Noise threshold:** accuracy deltas under **1 percentage point** (single-seed) or within the cross-seed spread (multi-seed) are treated as a tie.
3. **Tie-break order:** per-frame p99 runtime → connection count → memory footprint. Ties at every level default to `base` (cheapest, already validated).

### Acceptance

- All runs complete without crashing on the stocks workload.
- A written decision lands at the end of this doc naming the winner under the pre-registered rule above.
- Loser variants remain runnable via config — we don't delete them; they may be revisited if MNIST or another workload prefers a different rule.

### Notes / gotchas

- `SameLevel` and `LevelBelow` filtering need a per-level active set; the existing code already partitions by level via `memory.get_level_neurons(level)`. Use that.
- `AllLevels` may inflate per-frame runtime substantially on stocks — that's a data point, not a failure. Measure first, optimize later if it wins.
- If `base` wins outright, we still proceed with `base` semantics. The experiment is informative either way.

---

## Scope of the decision

This is a **d>0 temporal** experiment on stocks deciding a default that [spatial-processing.md](./spatial-processing.md) will apply to **d=0 spatial** scoping. The three inference regimes — `d=0` spatial, `d>0` event, `d<0` action — share one machinery, so the winning scope rule is the default for all three; but they are different regimes empirically, and the rule that wins here may not win elsewhere. Accordingly:

- The winner propagates into spatial processing as the **default**, to be **re-validated on MNIST** before being treated as settled for `d=0`.
- The winner is likewise the default for `d<0` action grounding (see [action-composition.md](./action-composition.md)), to be **re-validated once the action harness exists**. This gating is structural, not analogical: the action tower grounds level-below in the same units, with the arrow reversed.
- Conceptually the rule also informs [neuron-reuse.md](./neuron-reuse.md), but the reuse pool is unaffected by scope; only the prediction-set construction is.

---

## Decision

*(To be filled in after Phase 1 runs.)*

**Winner:**
**Result under pre-registered rule:** (accuracy deltas, runtime vs. budget, tie-breaks applied if any)
**Per-level connection counts:** (required if `same-level` underperformed — starvation or merit?)
**Notes:**
