# Neuron Reuse — Validation

**Cross-domain experiments that can only run after [Phase D](./neuron-reuse-final.md).** These are *not*
engineering phases — they ship no new mechanism. They are the measurement/tuning track that confirms reuse
does what the theory ([neuron-reuse.md](./neuron-reuse.md)) claims. Per-phase unit and acceptance gates live
in the phase docs; this doc holds only the experiments that need the full pipeline standing.

Each phase doc's gate proves *correctness*. This doc proves *value*.

---

## V1 — MNIST reuse validation + transfer effect

The headline claim: reused neurons cross task boundaries, so structure learned for early digits is reused
by later digits, yielding faster convergence and/or better generalization.

**Setup**: the MNIST single-frame harness ([spatial-processing.md §5.3](./spatial-processing.md)), reuse on.

**Measure**:

- **Neuron count**: total minted vs the Phase-A (mint-only / pre-reuse) baseline on the same data.
- **Accuracy**: ≥ the mint-only baseline.
- **Transfer**: train on digits **0–4**, then **5–9**, and compare 5–9 accuracy against training 5–9 **from
  scratch**. Also report neuron-count growth on the 5–9 block — expect it **sub-linear** vs from-scratch,
  because corrections minted for 0–4 sub-strokes get reused by 5–9.

**Scope caveat — transfer is co-located.** Spatial corrections inherit the parent coordinate
([spatial-processing.md §4.4](./spatial-processing.md)), so reuse fires for shared structure at the **same
receptive-field location**, not under translation. MNIST digits are centered, so co-located sub-strokes
(e.g. a shared top bar) *do* reuse; a stroke that appears at a different position in a different digit will
not. Frame the transfer result as co-located transfer — do not expect translation-invariant generalization
and do not read its absence as reuse failing.

**Gates**:

- Transfer effect detectable: 0–4 → 5–9 beats 5–9-from-scratch on 5–9 accuracy (or convergence speed) by a
  margin outside run-to-run noise.
- Neuron-count growth on 5–9 is sub-linear vs from-scratch.
- Reuse counts per neuron are observable and non-trivial (some neurons reused across the digit boundary).

---

## V2 — Stocks full-pipeline integration

Distinct from the spatial-only stocks integration ([spatial-processing.md §4](./spatial-processing.md), and
the spatial sweep winner recorded in project memory). That established a spatial baseline; this measures the
**additional** impact of reuse on top of the full spatial + temporal pipeline.

**Setup**: stocks with `process_spatial` on and reuse on. d=0 connections form across co-occurring top-level
patterns within each frame; spatial corrections feed the temporal phase in subsequent frames, building
spatio-temporal abstractions.

**Measure**: per-episode ROI and directional accuracy vs the spatial-only baseline; total neuron count vs
spatial-only-on-stocks.

**Tune**: the per-distance merge thresholds (`spatial_merge_threshold` and the temporal one) and the d=0
error threshold, if needed. Reuse is coupled to partial-context recognition through these thresholds
([neuron-reuse.md §2.5](./neuron-reuse.md)), so tune cautiously.

**Gates**:

- Directional accuracy improves over both prior baselines — target: lift off the historical 57–59% plateau.
- Neuron count significantly lower than spatial-only-on-stocks (reuse working as intended).

---

## V3 — Forget-rate / class-neuron generalization (long-run)

Validate and tune the generalization path from specific correction neurons to abstract class neurons over
long training (**10k+ frames**).

**Monitor**:

- Distribution of d=0 connection strengths over time per correction neuron — expect the structural core to
  strengthen and incidental edges to stay weak / decay (the reuse-then-decay sharpening of
  [neuron-reuse.md §1.3](./neuron-reuse.md)).
- Whether **reuse counts per neuron rise** over time — the signature of a correction becoming a class
  neuron.
- Whether **action bindings on heavily-reused correction neurons stay correct** — the dilution risk.

**Tune**: the global static forget rate if needed.

**Open**: when a correction neuron is heavily reused, action votes may need normalization across the many
contexts it now serves. Revisit if this experiment shows action-binding dilution; normalization is an
add-on, not assumed up front ([neuron-reuse.md §5.2](./neuron-reuse.md), action-binding-dilution risk).

**Gates**:

- Class neurons survive 10k+ frames with action binding intact.
- Heavily-reused neurons' strong d=0 connections converge toward a stable structural core (variance of the
  strong-edge set decreases over time).
