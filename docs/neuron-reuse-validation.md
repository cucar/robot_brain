# Neuron Reuse — Validation

**Experiments that can only run after [Phase D](./neuron-reuse-final.md).** They ship no new mechanism — they
confirm reuse does what the theory ([neuron-reuse.md](./neuron-reuse.md)) claims. Per-phase unit/acceptance
gates live in the phase docs; this doc holds the cross-domain experiments.

Reuse now applies at **all distances**, so both MNIST (spatial, d=0) and stocks (spatio-temporal) exercise it.
Each phase gate proves *correctness*; this doc proves *value*.

> First, a foundation check: [Phase A](./neuron-reuse-wavefront.md) is a rearchitecture (not bit-exact), so
> the **characterized regression** on MNIST + stocks is a prerequisite for everything here — confirm the
> wave-front learns comparably to the leveled baseline before measuring reuse on top.

---

## V1 — MNIST spatial reuse

The d=0 payoff: reuse collapses redundant spatial corrections within and across images.

**Setup**: the MNIST single-frame harness ([spatial-processing.md §5.3](./spatial-processing.md)), wave-front
+ reuse on.

**Measure**:
- **Neuron count** vs the Phase-A (no-reuse) baseline: within-frame batched mint dedups co-failers; cross-image
  lookup dedups recurring spatial structure.
- **Accuracy** ≥ baseline.
- **Transfer**: train digits **0–4**, then **5–9**; compare 5–9 accuracy vs training 5–9 from scratch.
  Sub-strokes shared across digits get reused → faster convergence / sub-linear neuron growth on 5–9.

**Scope caveat — co-located.** A correction fires only over its own footprint (specific base neurons), so
reuse captures a shared sub-stroke **at the same place**, not the same shape translated. MNIST is centered, so
co-located sub-strokes (e.g. a shared top bar) reuse; a stroke shifted in position does not. Frame the transfer
result as co-located, not translation-invariant. (Translation invariance needs relative connections — out of
scope, [neuron-reuse.md §5.1](./neuron-reuse.md).)

**Gates**: transfer effect detectable (0–4→5–9 beats from-scratch beyond noise); 5–9 neuron growth sub-linear;
reuse counts across the digit boundary non-trivial.

---

## V2 — Stocks full pipeline + transfer

Distinct from the spatial-only stocks baseline. Measures reuse on the full spatio-temporal wave-front.

**Setup**: stocks, wave-front + reuse on, d=0 and d>0.

**Measure**: per-episode ROI and directional accuracy vs the Phase-A baseline; total neuron count.

**Tune**: the per-distance merge thresholds and error thresholds; the strength-candidacy choice
([Phase B](./neuron-reuse-index.md)).

**Transfer** (the [future-work transfer-learning experiment](./future-work.md)): learn on stock set A, measure
set B before/after — reuse should let B converge faster if A and B share spatio-temporal structure.

**Gates**: directional accuracy improves over prior baselines (target: lift off the 57–59% plateau); neuron
count significantly lower than no-reuse; transfer effect present.

> Note the footprint-graded locality from Phase A: cross-channel (cross-stock) relationships now form only
> through grouping (as footprints grow), not as a raw cross-product. Watch whether that helps or limits
> cross-stock structure; it interacts with the reuse measurement.

---

## V3 — Forget-rate / class-neuron generalization (long-run)

Validate the path from specific corrections to abstract class neurons over long training (**10k+ frames**).

**Monitor**:
- Connection-strength distribution per correction — structural core strengthens, incidental edges stay
  weak / decay (the reuse-then-decay sharpening of [§1.3](./neuron-reuse.md)).
- **Reuse counts per neuron rising** — the signature of a correction becoming a class neuron.
- **Action bindings on heavily-reused neurons staying correct** — the dilution risk.

**Tune**: the brain-wide forget rate.

**Open**: heavily-reused neurons may need action-vote normalization across contexts. Revisit if dilution
appears.

**Gates**: class neurons survive 10k+ frames with action binding intact; heavily-reused neurons' strong
connections converge toward a stable structural core.

---

## V4 — Eventual: chatbot / text

Sequence reuse over tokens is where temporal reuse ultimately pays off — shared sub-sequence predictors across
many conversations. Once the text-channel-to-action chatbot harness exists ([roadmap §4](./roadmap.md)),
validate reuse there. Out of scope for the initial build, but the destination this design serves.
