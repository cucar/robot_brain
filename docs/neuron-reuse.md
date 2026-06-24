# Neuron Reuse on a Wave-Front Architecture

This document is the **theory** of the unified processing model and neuron reuse built on it. It mirrors
[spatial-processing.md](./spatial-processing.md): it describes *what* the model is and *why* it works, not the
build order. Implementation is split across phase docs plus a validation doc; see
[§6 Implementation](#6-implementation).

> **The milestone.** The processing **structure** is unchanged — spatial processing → apex handoff →
> temporal processing — but each stage becomes a **settling wave**, stored levels are gone, only base
> sensory/action neurons carry coordinates, and all corrections are coordinate-less with **footprints** for
> locality. On that foundation, **neuron reuse applies at every distance** (d=0 and d>0 alike) — the
> spatial/temporal asymmetry is gone. Phase A ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)) builds
> the foundation; the rest is reuse on top.

---

## 1. Motivation

### 1.1 The Problem

When the thalamus detects a prediction error at any distance, it always creates a brand new neuron. But the
inference needed may already exist somewhere in the network — a neuron created for a different context may
already produce exactly the required prediction. Without reuse, structure grows indefinitely: every error
mints a fresh neuron regardless of overlap.

Separately, the pre-wave-front architecture carried a **spatial/temporal asymmetry**: spatial corrections were
coordinate-anchored (for receptive-field growth), which made sharing them ambiguous (whose coordinate does a
shared correction take?), so reuse could only be done cleanly for temporal. The wave-front + footprint
foundation removes that asymmetry, so reuse becomes uniform.

### 1.2 The Solution: One Wave-Front + Reuse at Every Distance

Two pieces:

1. **Foundation (Phase A).** Collapse spatial and temporal into one wave-front. Drop coordinates from all
   corrections; carry locality as a **footprint** (the set of base sensory neurons a correction covers).
   Neighborhood at any level = footprint overlap. Details: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).
2. **Reuse (Phases B–D).** The reuse criterion is **inference-output match**: does an existing neuron's
   connection set already produce the inference a correction would need? Per (distance, footprint-region) per
   frame, query a reverse index for a match; on a hit, wire all co-failing neurons to the existing neuron; on
   a miss, mint **one** coordinate-less correction (footprint = union) and wire all co-failers to it.

Because corrections are coordinate-less, a shared correction takes the **union footprint** of its parents —
no anchor to be ambiguous about. That is what makes reuse clean at d=0, not just d>0.

### 1.3 Why Reuse Is Essential, Not an Optimization

Reuse is what makes generalization possible.

A single correction created from one event with no reuse only memorizes that event. When the same correction
is reused across many distinct error events, the connections shared across all those events accumulate
strength while per-event-specific connections stay weak. Over many reuses, the strong connections converge on
the structural core common to the equivalence class of triggering events. Decay then sharpens this: incidental
connections erode, reinforced structural ones persist. Reuse provides the cross-instance signal; decay
sharpens it. Both are required, at every distance.

---

## 2. The Wave-Front Foundation

### 2.1 One Operation, Two Waves

The same operation runs at every distance — **learn relationships at distance d**:

- **d = 0** — relationships within the current frame. This is `process_spatial`.
- **d = k** — relationships between the current frame and k frames back. This is `process_temporal`.

The **structure is unchanged**: `process_spatial` → apex handoff → `process_temporal`, each a separate
function. They stay separate because their control flow differs: spatial **settles within the frame** (a
multi-round wave to a fixpoint, deepening the within-frame hierarchy), while temporal **passes over the
materialized sliding window** (last frame's apex is already at age 1; its hierarchy deepens *across* frames,
not within one). That is the only irreducible spatial/temporal difference — space is the distance with no time
to traverse, time is the distance that takes frames to traverse. Everything else — footprints, coordinate-less
corrections, reuse — is shared. There is no `process_ages` collapse and no age-to-age chaining (§2.3).

### 2.2 Footprints — Locality Without Coordinates

Only base sensory/action neurons have coordinates. A correction has a **footprint** = the set of base sensory
neurons it covers (base: itself; correction: union of constituents). Two neurons are **neighbors** iff their
footprints touch in the base neighbor graph. This replaces coordinate-inheritance *and* channel-neighbor
filtering, at every distance. Locality is graded: low-level corrections are local, high-level ones span more
as footprints grow, so cross-region/cross-channel structure emerges through grouping rather than a raw
cross-product. Mechanics: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).

This is safe because **a correction is never a prediction output** — `aggregate_votes` already forbids a
pattern neuron from being a vote target ([brain.rs:1626-1638](../brain/brain-core/src/brain.rs)), so only
base neurons are ever dequantized. A correction's coordinate was only ever used for neighbor-filtering, which
footprints subsume.

### 2.3 The Apex Fans Out; Ages Do Not Chain

The spatial apex feeds temporal — and within temporal it feeds **every** distance in parallel (d=1, d=2, …),
not a chain where d=1 feeds d=2. The depth chain (output→input) lives inside each wave and across frames, not
along the age axis.

Cross-distance recurrence happens frame-to-frame: this frame's apex becomes next frame's age-1 context. Within
a single frame, the older ages are **materialized memory** of past outputs, not a pipeline to re-run.
Intra-frame age-chaining (feeding age d's output into age d+1's input in the same frame) would telescope the
time axis into the depth axis — double-counting the window and breaking the persistence that lets an activated
pattern vote every frame it is alive. So temporal relates the apex to each past age independently; there is no
`process_ages` cascade.

---

## 3. Reuse Mechanism (all distances)

### 3.1 One Observed Reality Per (Distance, Footprint-Region)

At a given distance d, within a footprint-region, the realized reality is **singular** — one actual
co-activation (d=0) or realized future (d>0) in that region this frame. Multiple neurons may err there, but
all erred by predicting different wrong things about the *same* observed set. Correction targets the observed
set. So there is at most one correction per (distance, observed-set) group, and the grouping key is the
observed set itself (well-defined: the set of observed neuron ids). Co-failers with the same observed set are
looking at the same base region, so their footprints overlap — grouping them and taking the **union footprint**
is unambiguous. There is **no anchor policy and no clustering problem** (footprints dissolved both).

### 3.2 Per-Group Lookup

For each (distance, observed-set) group with errors:

1. The thalamus knows the observed inference set for that group.
2. Query the **reverse inference index**: for each observed target T, which existing neurons have a connection
   to T at this distance?
3. Score each candidate's inference signature against the observed set (common/missing/novel, same as pattern
   recognition).
4. If the best scores ≥ the **merge threshold for this distance**, wire all co-failers in the group to it.
5. Else, mint (§3.3).

### 3.3 Batched Mint (Fallback)

On a miss, mint **one** coordinate-less correction for the group's observed reality, footprint = union of the
co-failers' footprints, and wire **all** co-failers to it. Within-frame dedup; no per-neuron duplicates.

### 3.4 Why Lookup Is Still Required

Batched mint is within-frame only. The same reality recurring next frame, to the mint path, is a fresh group
that mints another duplicate. Lookup is the cross-frame collapse — and *is* the generalization mechanism
(§1.3): a neuron reused across frames accumulates the structural core. Lookup first (cross-frame generalize),
batched mint as the fallback beneath it (within-frame dedup).

### 3.5 Merge Threshold

Reuse reads the **same merge threshold pattern recognition uses at that distance** (spatial threshold at d=0,
temporal at d>0). No separate `reuseMergeThreshold`. Setting a distance's threshold to 1.0 disables reuse and
partial recognition together.

---

## 4. Lifecycle, Activation, Inhibition

### 4.1 Multi-Parent References

Reuse and batched mint both wire **many** parents to one correction, which breaks the single-host ownership
model (today a correction's strength lives in its one host's routing entry and it dies when that entry decays
— [neuron.rs:64, 689-703, 709-714](../brain/brain-core/src/neuron.rs); the forget rate is brain-wide uniform —
[brain.rs:366](../brain/brain-core/src/brain.rs)). The clean shape: per-parent routing entries on a shared,
coordinate-less neuron; **reference-counted reaping** (alive while any parent references it); thalamus
**collapses multiple same-frame activations to one** (refractory) while crediting every activating parent
(all subsumed); `patterns.csv` serializes **many** `(parent, strength)` rows per pattern. Because corrections
are coordinate-less, there is **no anchor to reconcile** across parents — only lifecycle and activation
bookkeeping. Born with batched mint; mechanics in [neuron-reuse-frame.md](./neuron-reuse-frame.md).

### 4.2 Refractory and Correction-Wired Inhibition

Two per-frame tracking sets, cleared at frame end:

- **`fired_this_frame`** — refractory (each neuron fires at most once per frame). Load-bearing once a shared
  neuron can be activated from many parents. Lands with batched mint.
- **`correction_wired_this_frame`** — every correction target this frame that is a **reused pre-existing**
  neuron. A fresh mint already has the fresh-mint exemption ([spatial-processing.md §3.3 step 2](./spatial-processing.md));
  a reused neuron is not fresh, so it needs an explicit tag: learn the observed set, **don't vote** (layered
  on the existing `activated_pattern_id` suppression, so empty ⇒ bit-identical voting), **don't error-check**
  (its old connections must not spawn a fresh error cascade). Lands with the lookup.

---

## 5. Benefits & Risks

### 5.1 Benefits

- **Symmetry** — one operation, one neighborhood primitive, one reuse mechanism across all distances.
- **Compact structure at every level** — no redundant neurons computing the same inference, spatial or
  temporal.
- **Transfer / generalization** — a shared correction links the contexts that produce the same observed
  reality; structure transfers rather than being re-minted. (Cross-position transfer — the *same shape at a
  different place* — is **not** provided; a correction fires only over its own footprint. That ceiling is the
  retinotopic/absolute-connection model, common to both spatial and temporal.)
- **Content-addressable** — "is there a neuron that produces X?" answered via the reverse inference index.

### 5.2 Risks

- **Wave-front rearchitecture** (Phase A) is large and not bit-exact — characterized regression, not byte
  identity.
- **Footprint cost** at the apex (large footprints); bounded by bitsets, watch memory for huge base (vision).
- **Reverse-index cost** — one lookup per group, sharded by column, updates at orchestration boundaries.
- **Over-aggressive reuse** — too-low merge threshold; tune cautiously, lean on decay. See strength-candidacy
  ([neuron-reuse-index.md](./neuron-reuse-index.md)).
- **Action-binding dilution** — heavily-reused neurons; monitored long-run.

### 5.3 Decisions to settle before building

1. **Wave-front shape** — within-age settle vs single-pass for d>0; fixpoint + determinism
   ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md) OQ1–2).
2. **Mint-frame vs reuse-frame inhibition window** ([neuron-reuse-frame.md](./neuron-reuse-frame.md) DECIDE-THIS #1).
3. **Refractory vs cross-depth injection** for a reused neuron ([neuron-reuse-final.md](./neuron-reuse-final.md) DECIDE-THIS #2).
4. **Strength-candidacy** for the reuse index ([neuron-reuse-index.md](./neuron-reuse-index.md)).

---

## 6. Implementation

Build order: wave-front foundation → index → batched mint → lookup, then validation. Reuse applies at all
distances throughout.

| Phase | Doc | Goal | Gate |
|---|---|---|---|
| **A** | [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md) | **Foundation.** Turn `process_spatial` and `process_temporal` into settling waves (structure kept); remove stored levels; coordinate-less corrections; footprints for neighborhood in both waves; multi-depth memory. | Characterized regression (MNIST + stocks comparable); footprint/fixpoint units. **Not bit-exact.** |
| **B** | [neuron-reuse-index.md](./neuron-reuse-index.md) | Reverse **inference** index (target → distance → sources) over **all** connections (d=0 and d>0). Built, unit-tested, not yet consumed. Settles strength-candidacy. | Unit: target→sources correct across distances; size ∝ connection count. |
| **C** | [neuron-reuse-frame.md](./neuron-reuse-frame.md) | **Batched mint** at all distances (group by (distance, observed-set), mint one coordinate-less correction with union footprint, wire all) **+ multi-parent machinery** (refractory, shared activation, refcounted reaping, multi-parent serialization). Settles DECIDE-THIS #1. | Within-group dedup drops neuron count; multi-parent lifecycle units. |
| **D** | [neuron-reuse-final.md](./neuron-reuse-final.md) | Reuse **lookup** on top of C (consumes B) + `correction_wired_this_frame` + cross-frame accrual, all distances. Settles DECIDE-THIS #2. | Unit: cross-frame reuse via lookup; neuron count drops further; lookup < 20% per-frame overhead. |
| **Validation** | [neuron-reuse-validation.md](./neuron-reuse-validation.md) | MNIST (spatial reuse, within-frame + cross-image), stocks (full pipeline + transfer), long-run forget-rate. | Per-experiment gates in the doc. |

All phases depend on [spatial-processing](./spatial-processing.md) being complete (it is) — Phase A then
rebuilds its level model into the wave-front.
