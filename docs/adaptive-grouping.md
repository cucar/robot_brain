# Adaptive Grouping — one self-calibrating "sameness" threshold

This document is the **design** for collapsing the brain's grouping thresholds — the **merge threshold**
(recognition / reuse) and the **error threshold** (correction), each previously split into a **spatial** and a
**temporal** copy — into a single parameter `groupThreshold` (θ), and then making that single threshold
**adaptive** so it self-calibrates instead of being hand-tuned.

The collapse is total: the six historical knobs (`{spatial,temporal}MergeThreshold`,
`{spatial,temporal}ErrorCorrectionThreshold`, `{spatial,temporal}ErrorCorrectionMode`) become **one** shared
`groupThreshold` plus **one** shared `groupMode`. There is no per-phase split — a brain has one notion
of how-similar-is-the-same, and it is the same on the input (d=0) and sequence (d>0) sides.

It is a **foundation change to the existing brain**, independent of (and sequenced before) the wave-front and
reuse projects: it touches **recognition and error correction everywhere — spatial and temporal**. Every demo
must be re-validated after each stage (§5).

Two stages, in order:

1. **Unify the thresholds** — one `groupThreshold` (θ), shared spatial + temporal, with `error = 1 − θ`.
2. **Unified adaptive grouping** — derive that one coefficient per-unit from its own running error statistics,
   removing the last hand-tuned magnitude.

---

## 1. The insight: recognition and correction are one operation

The brain compares "are A and B the same?" in two places:

- **Recognition / reuse** — is the observed context the same as a stored pattern's context? If similar enough
  (≥ the **merge threshold** θ), the pattern fires / is reused
  ([context.rs `match_observed`](../brain/brain-core/src/context.rs),
  [neuron.rs `recognize_spatial_patterns`](../brain/brain-core/src/neuron.rs)).
- **Correction** — is the predicted L0 the same as the actual L0? If different enough (> the **error
  threshold**), the unit mints a correction request
  ([neuron.rs `get_spatial_error_threshold`](../brain/brain-core/src/neuron.rs),
  [thalamus.rs `mint_spatial_corrections`](../brain/brain-core/src/thalamus.rs)).

These are the **same operation** — a similarity test against one bar — pointed at two different operands (a
context on the input side, a prediction on the output side). Posit a single "sameness" threshold θ:

- fire / reuse when `similarity ≥ θ` ("same"),
- correct when `similarity < θ` ("different").

A brain plausibly has **one** notion of how-similar-is-the-same. Two independently-tuned numbers for what is
really one grouping decision is the least defensible option biologically and the most overfit-prone
engineering-wise.

---

## 2. The math: `error = 1 − merge` is an identity, not a coincidence

Both comparisons are the same Jaccard over the union. In the code:

- match score = `common / (common + missing + novel)`
- error rate  = `(missing + novel) / (common + missing + novel)`

They **sum to exactly 1** by construction: `error = 1 − matchScore` over any single comparison. So "correct when
`similarity < θ`" is exactly "correct when `error > 1 − θ`", i.e.

```
errorThreshold = 1 − mergeThreshold
```

is the **direct consequence** of there being one sameness threshold. The two thresholds were never independent;
the second is the first read from the other side.

---

## 3. Stage 1 — unify the thresholds (one `groupThreshold`, `error = 1 − θ`)

Replace the six parameters with one θ ([neuron.rs `group_threshold`](../brain/brain-core/src/neuron.rs)) plus the
shared error mode. Recognition reads θ directly ([neuron.rs `match_observed` call](../brain/brain-core/src/neuron.rs));
the correction side sets the error threshold to `1 − θ` wherever it is read
([neuron.rs `get_spatial_error_threshold` / `get_temporal_error_threshold`](../brain/brain-core/src/neuron.rs)) — its
**static / warmup** value becomes a derived quantity. Spatial and temporal collapse onto the *same* θ, so the
two correction getters survive only because they read different Welford buckets (spatial = one bucket, temporal =
per-age); the threshold magnitude and the adaptation mode they apply are now one shared pair.

Implemented across the whole plumbing chain — napi (`groupThreshold` option, retired keys warn-and-ignore) →
`Brain::new` → `Thalamus::new` → `Region::new` → `Column::new` → `Neuron::new`.

### Validation (reference simulation)

The simulation ([`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js), `couple` mode) swept
θ with `error = 1 − θ` head-to-head against the two-parameter decoupled grid, **same config** (14×14 binary
MNIST, train/test 400, reuse + refinement on):

| θ (merge) | error = 1−θ | test acc | train acc | patterns |
|---|---|---|---|---|
| 0.50 | 0.50 | 14.25% | 13.00% | 916 (degenerate) |
| 0.60 | 0.40 | 21.25% | 24.25% | 1091 |
| 0.70 | 0.30 | 43.75% | 58.25% | 811 |
| **0.80** | **0.20** | **50.75%** | 91.75% | 727 |
| 0.90 | 0.10 | 23.00% | 98.00% | 680 (overfit) |

**Best coupled = 50.75% (θ=0.8). Best decoupled at the same scale = 49.75% (merge 0.9 / error 0.4).** The
one-parameter model **ties — slightly edges — the two-parameter model** (the 1-point gap is within noise on 400
test images). The coupled curve has a clean single-parameter optimum at θ≈0.8 and rolls into overfitting at
0.9/0.1, exactly the shape a one-knob model should have. **Collapsing two thresholds into one costs nothing.**

---

## 4. Stage 2 — unified adaptive grouping (self-calibrating θ)

The single θ need not be hand-tuned. The brain **already** adapts the error threshold from each unit's running
**Welford** statistics of its observed error rate — `mean ± σ` under the conservative / neutral / aggressive
modes ([neuron.rs `get_spatial_error_threshold` / `apply_error_mode`](../brain/brain-core/src/neuron.rs)). With
the Stage-1 coupling, that single adaptive quantity drives **both** gates:

```
errorThreshold(unit) = mean(unit) + k·σ(unit)        // already implemented (k ≈ 1; mode picks the sign)
mergeThreshold(unit) = 1 − errorThreshold(unit)      // derived — recognition strictness from prediction reliability
```

This replaces a **dataset-specific magnitude** (a static `0.3` that has to be retuned per scale — the sim's
threshold sweet spot moved from ~0.66 at 60 images to ~0.8 at 400) with a **dimensionless, self-tuning** form:
"a unit's recognition strictness equals one minus how wrong it usually is." The
chicken-and-egg resolves cleanly: adapt from **error** (output side, always observable on every prediction),
**derive** merge on the input side.

### Risk: blur-runaway, and the adaptive-error-first order

The adaptation has a load-bearing sign. Under conservative (`mean + σ`):

- a **reliable** unit (low error) → low error threshold → high derived merge → fires **strictly** (good — precise
  predictors stay precise),
- an **unreliable** unit (high error) → high error threshold → low derived merge → fires **loosely**.

The second case is the danger: a bad unit could become *more* promiscuous and drag the field toward mush — the
"one big blurry pattern" collapse seen at low static θ in the sim. Whether refinement + reuse pull it back faster
than loose firing spreads is a **dynamics** question, not settled on paper. So Stage 2 lands in two steps:

1. **Adaptive error, static merge** — turn on per-unit adaptive error (already built) with the merge threshold
   still fixed; confirm the dynamics are stable and accuracy holds before removing the last guardrail.
2. **Fully adaptive coupled** — `merge = 1 − (mean + k·σ)` per unit. Recognition reads the observing unit's error
   stats. This is the zero-hand-tuned-threshold target.

---

## 5. Scope & mandatory re-validation

This is **not a reuse-only change** — the merge and error thresholds gate recognition and correction across the
**whole** brain, spatial **and** temporal. So after **each** stage, **all demos must be re-tested** and the
numbers re-recorded:

- **MNIST** (spatial) — accuracy at the representative stack ([96.44% config](./neuron-reuse.md)); confirm no
  regression vs the per-threshold baselines.
- **Stocks** (temporal + actions) — the [spatial sweep winner](./spatial-processing.md) and the full pipeline.
- Any other demo with tuned `merge_threshold` / `error_threshold` / error-mode flags — re-sweep the single θ.

Because Stage 1 removes one degree of freedom, expect to **re-find the operating point** (a single θ sweep)
rather than reuse the old two-number settings.

---

## 6. Acceptance gates

**Stage 1 (unify):**
- One `groupThreshold` (θ) shared brain-wide; the static / warmup error threshold is derived as `1 − θ`
  everywhere it was read (spatial + temporal), and the six retired knobs are gone (retired napi keys warn-and-ignore).
- A single-θ sweep on MNIST and stocks finds an operating point with accuracy **≥ the decoupled six-parameter
  baseline** (ties within noise are a pass — the win is the parameter, not the accuracy).
- If a coupled penalty appears, it is closed by **normalizing** the two comparisons, not by re-decoupling.

**Stage 2a (adaptive error, static merge):**
- Per-unit adaptive error on; dynamics stable (no pattern-count blow-up or collapse); accuracy ≥ Stage-1.

**Stage 2b (fully adaptive coupled):**
- `merge = 1 − adaptiveError` per unit; recognition reads the observing unit's error stats.
- No blur-runaway (footprint-size distribution stays regional, not collapsing to one blob); accuracy ≥ Stage-2a.
- **Zero hand-tuned thresholds** remain (only the scale-free σ multiplier / mode).

---

## 7. Open questions

- Test performance of static vs conservative vs neutral vs aggressive for the unified threshold.