# Neuron Reuse on a Wave-Front Architecture

This document is the **theory** of the unified processing model and neuron reuse built on it. 
It describes *what* the model is and *why* it should work.
Implementation is split across phase docs plus a validation doc; see [Implementation](#6-implementation).

This is an architectural milestone. 
Processing structure is unchanged (spatial processing → apex handoff → temporal processing) but each level becomes a **wave**. 
Stored levels will be gone. Only base sensory/action neurons will carry coordinates.
All corrections (patterns) are coordinate-less with **footprints** for locality. 
On that foundation, **neuron reuse applies at every distance** (d=0 and d>0 alike). 
Phase A ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)) builds the foundation; the rest is reuse on top.

---

## 1. Motivation

### 1.1 The Problem

Currently, when the thalamus detects a prediction error at any distance, it creates a brand-new neuron. 
But the inference needed may already exist somewhere in the network. 
A neuron created for a different context may already produce exactly the required prediction. 
Without reuse, structure grows indefinitely: every error mints a fresh neuron regardless of overlap.

Spatial corrections are coordinate-anchored for receptive-field growth with neighborhoods.
Spatial patterns currently inherit their parents' channels, dimensions and coordinates. 
This means temporal processing level 0 has channels, coordinates and dimensions, which are used for temporal neighborhoods. 
Temporal patterns currently inherit the channels as well, but they are not really used.

The problem is that with neuron reuse, it will be possible for pattern neurons to have multiple parents.
Having multiple parents kills the neighborhoods. Coordinates of patterns with multiple parents become undefined.
To handle multiple parents, spatial/temporal pattern neurons cannot have coordinates. 
The wave-front + footprint makes that possible, so reuse becomes uniform.

### 1.2 The Solution: One Wave-Front + Reuse at Every Distance

Two pieces:

1. **Foundation (Phase A).** Unify spatial and temporal under one wave-front operation (still two functions). Drop coordinates from all
   corrections; carry locality as a **footprint** (the set of base sensory neurons a correction covers).
   Neighborhood at any level = footprint overlap. Details: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).
2. **Reuse (Phases B–D).** The reuse criterion is **inference-output match**: does an existing neuron's
   connection set already produce the inference a correction would need? Per (distance, footprint-region) per
   frame, query a reverse index for a match (partial or exact); on a hit, wire all co-failing neurons to the existing neuron; on
   a miss, mint **one** coordinate-less correction (footprint = union) and wire all co-failers to it.

Because corrections are coordinate-less, a shared correction takes the **union footprint** of its parents —
no anchor to be ambiguous about. That is what makes reuse clean at d=0, not just d>0.

### 1.3 How Generalization Arises

Generalization is not produced by reuse alone — it is a stack of three contributions, and reuse is the
middle one.

1. **Substrate — the wave-front architecture.** Coordinate-less corrections with footprint-overlap
   neighborhoods make representations compositional: a single correction can span regions and channels, so
   structure emerges through grouping rather than a raw cross-product (§2.2). This is also what makes reuse
   *legal* — multi-parent corrections require coordinate-less identity. Without this substrate there is no
   cross-region generalization, and no reuse to begin with.
2. **Convergence — reuse.** A single correction created from one event with no reuse only memorizes that
   event. Reuse is what lands many distinct error events on the *same* neuron: connections shared across all
   those events accumulate strength while per-event-specific connections stay weak, so over many reuses the
   strong connections converge on the structural core common to the equivalence class of triggering events.
   The architecture makes representations mergeable; reuse is what actually merges the instances. Without it
   the coordinate-less network is just a neighbor-graph of single-event memorizers.
3. **Sharpening — refinement and decay.** Decay erodes incidental connections and keeps reinforced
   structural ones. Context/connection refinement ([refinement.md](./refinement.md)) consolidates a
   correction's sources *and* targets toward their common core, cleaning the class boundary. Reuse provides
   the cross-instance signal; refinement and decay sharpen it.

None of the three is sufficient alone: the architecture makes the right kind of generalization *possible*
and reuse *legal*, reuse converges instances onto shared structure, and refinement and decay sharpen the
result. All three are required, at every distance.

---

## 2. The Wave-Front Foundation

### 2.1 One Operation, Two Waves

The same operation runs at every distance — **learn relationships at distance d**:

- **d = 0** — relationships within the current frame. This is `process_spatial`.
- **d > 0** — relationships between the current frame and d frames back. This is `process_temporal`.

The **structure is unchanged**: `process_spatial` → apex handoff → `process_temporal`, each a separate function. 
Both already run almost identical bottom-up level-sweeps that settle until no higher level fires
(`process_spatial_levels` / `process_temporal_levels`, [brain.rs](../brain/brain-core/src/brain.rs)). 

The active set is fixed at the start, and freshly-minted corrections fire next frame. 
They stay two functions for the narrow reasons they are two today: 

- temporal **persists per-neuron state across frames** and reads d>0 connections
- spatial is **ephemeral** and reads d=0. 

That cross-frame persistence is the only irreducible spatial/temporal difference.
Space is the distance with no time to traverse. Time is the distance that takes frames to traverse. 
Everything else (sweep, footprints, coordinate-less corrections, reuse) is shared. 

### 2.2 Footprints: Neighborhoods Without Coordinates

Only base sensory/action neurons have coordinates. 
A correction has a **footprint** = the set of base sensory neurons it covers (base: itself; correction: union of constituents). 
Footprints are sensory-only for now; action footprints come with the action-composition project. 
Two neurons are **neighbors** if and only if their footprints intersect in the base neighbor graph. 
This replaces coordinate-inheritance *and* channel-neighbor filtering, at every distance. 
Locality is graded: low-level corrections are local, high-level ones span more as footprints grow.
So, cross-region/cross-channel structure emerges through grouping rather than a raw cross-product. 
Details: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).

Note that corrections are *not* the output; their base-targeting event and action connections are votes for the output.
`aggregate_votes` already forbids a pattern neuron from being a vote target ([brain.rs](../brain/brain-core/src/brain.rs)).
So, only the base neurons are ever dequantized. 

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
— [neuron.rs](../brain/brain-core/src/neuron.rs); the forget rate is brain-wide uniform —
[brain.rs](../brain/brain-core/src/brain.rs)). The clean shape: per-parent routing entries on a shared,
coordinate-less neuron; **reference-counted reaping** (alive while any parent references it); `patterns.csv`
serializes **many** `(parent, strength)` rows per pattern. Because corrections are coordinate-less, there is
**no anchor to reconcile** across parents — only lifecycle and activation bookkeeping. Born with batched mint;
mechanics in [neuron-reuse-frame.md](./neuron-reuse-frame.md).

### 4.2 Same-Frame Activation

A minted **or** reused correction is **installed into the erroring neuron's routing table and fires next
frame** — not the frame it is wired ([neuron.rs](../brain/brain-core/src/neuron.rs)). The erroring
parent's wrong vote this frame is already suppressed by the existing machinery (`get_suppressed_ages`,
[neuron.rs](../brain/brain-core/src/neuron.rs)). So reuse adds **no** same-frame activation to
inhibit: the originally-planned `correction_wired_this_frame` set has nothing to act on, and a reused neuron is
active this frame only via its **own** routing match (a normal recognition).

The genuinely-new activation case is **multi-parent routing**: a shared neuron can be routing-matched from
several parents at different depths in one sweep, activating it at each. Whether that fires **once**
(refractory) or **once per depth** (multi-depth memory) — and whether it needs any inhibition — is the open
**multi-parent activation model** (§5.3).

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
- **Over-aggressive reuse** — too-low merge threshold; tune cautiously. There is **no decay backstop** today
  (connections never weaken — [neuron.rs](../brain/brain-core/src/neuron.rs)), so a bad reuse
  persists. See strength-candidacy ([neuron-reuse-index.md](./neuron-reuse-index.md)).
- **Action-binding dilution** — heavily-reused neurons; monitored long-run.

### 5.3 Decisions to settle before building

1. **Strength-candidacy** for the reuse index — does a weak, incidental connection make a neuron a reuse
   candidate, or is membership gated/weighted by strength? ([neuron-reuse-index.md](./neuron-reuse-index.md)).
2. **Multi-parent activation model** — when reuse routing-matches a shared neuron from several parents at
   different depths in one sweep, how is that held (multi-depth `neuron_states`) and is it processed at each
   depth or collapsed? This also decides whether the `fired_this_frame` / `correction_wired_this_frame`
   inhibition sets are needed at all ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md),
   [neuron-reuse-final.md](./neuron-reuse-final.md)).

---

## 6. Implementation

Build order: wave-front foundation → index → batched mint → lookup, then validation. Reuse applies at all
distances throughout.

| Phase | Doc | Goal | Gate |
|---|---|---|---|
| **A** | [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md) | **Foundation.** Turn `process_spatial` and `process_temporal` into settling waves (structure kept); remove stored levels; coordinate-less corrections; footprints for neighborhood in both waves; multi-depth memory. | Characterized regression (MNIST + stocks comparable); footprint-adjacency units. **Not bit-exact.** |
| **B** | [neuron-reuse-index.md](./neuron-reuse-index.md) | **Two** reverse connection indexes — `spatial_connection_index` (target → sources) and `temporal_connection_index` (target → distance → sources), mirroring the context-index split. Built, unit-tested, not yet consumed. Settles strength-candidacy. | Unit: target→sources correct per index; stores isolated; size ∝ connection count. |
| **C** | [neuron-reuse-frame.md](./neuron-reuse-frame.md) | **Batched mint** at all distances (group by (distance, observed-set), mint one coordinate-less correction with union footprint, wire all) **+ multi-parent machinery** (refcounted reaping, multi-parent serialization, shared-neuron activation). | Within-group dedup drops neuron count; multi-parent lifecycle units. |
| **D** | [neuron-reuse-final.md](./neuron-reuse-final.md) | Reuse **lookup** on top of C (consumes B) + cross-frame accrual, all distances. | Unit: cross-frame reuse via lookup; neuron count drops further; lookup < 20% per-frame overhead. |
| **Validation** | [neuron-reuse-validation.md](./neuron-reuse-validation.md) | MNIST (spatial reuse, within-frame + cross-image), stocks (full pipeline + transfer), long-run forget-rate. | Per-experiment gates in the doc. |

All phases depend on [spatial-processing](./spatial-processing.md) being complete (it is) — Phase A then
rebuilds its level model into the wave-front.
