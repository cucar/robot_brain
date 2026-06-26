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
   Neighborhood at any level = footprint adjacency (footprints touch in the base neighbor graph). Details: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).
2. **Reuse (Phases B–D).** Each frame, recognized patterns predict L0 (§3.1); where a prediction is wrong, the
   request is **clustered with its neighbors** (transitive merge) and either **reuses** an existing pattern (a
   reverse-index match ≥ the merge threshold) or **mints** a new one. Reuse is the *merge* force — it lands many
   contexts on shared, general patterns; refinement and forgetting are the *split* force that carves them back
   toward regional patterns. Mechanism: §3.

Because corrections are coordinate-less, a reused or expanded pattern can take a **union footprint** across all
the parents that reuse it — no anchor to reconcile. That is what makes reuse clean at d=0, not just d>0.

### 1.3 How Generalization Arises

Generalization is not produced by reuse alone — it is a stack of three contributions, and reuse is the
middle one.

1. **Substrate — the wave-front architecture.** Coordinate-less corrections with footprint-adjacency
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
Two neurons are **neighbors** if and only if their footprints **touch** in the base neighbor graph — some base neuron in one is adjacent to (or equal to) a base neuron in the other, not only when they share a base neuron. 
This replaces coordinate-inheritance *and* channel-neighbor filtering, at every distance. 
Locality is graded: low-level corrections are local, high-level ones span more as footprints grow.
So, cross-region/cross-channel structure emerges through grouping rather than a raw cross-product. 
Details: [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md).

Note that corrections are *not* the output; their base-targeting event and action connections are votes for the output.
`aggregate_votes` already forbids a pattern neuron from being a vote target ([brain.rs](../brain/brain-core/src/brain.rs)).
So, only the base neurons are ever dequantized. 

### 2.3 The Apex Fans Out; Ages Do Not Chain

Since spatial is d=0 and temporal is d>0, it's tempting to chain them: d=0 feeds d=1, d=1 feeds d=2, and so on. 
But that does not work well. The design does not extend cleanly. 

Cross-distance recurrence happens frame-to-frame: this frame's apex becomes next frame's age-1 context. 
Within a single frame, the older ages are **materialized memory** of past outputs, not a pipeline to re-run.
Intra-frame age-chaining (feeding age d's output into age d+1's input in the same frame) would telescope the
time axis into the depth axis — double-counting the window and breaking the persistence that lets an activated
pattern vote every frame it is alive. So temporal relates the apex to each past age independently; there is no
`process_ages` cascade.

The spatial apex feeds every temporal distance in parallel (d=1, d=2, …).
The depth chain (output→input) lives inside each wave and across frames, not along the age axis.
This will not change.

---

## 3. Reuse Mechanism — recognize, mispredict, cluster, correct

This is the corrected model. (An earlier version of this section described a cold-start, single-frame
"group every neuron by its observed-set" mechanism — that was **wrong**, and so is the simulation built on
it. See §3.9 and [neuron-reuse-simulation.md](./neuron-reuse-simulation.md).)

Each frame, spatial processing runs one loop: **recognize → predict L0 → on error, cluster the requests →
mint/expand a more-contextual correction**. Reuse is the *merge* force woven through it; refinement and
forgetting are the *split* force. The whole thing is distance- and level-invariant.

### 3.1 Every level predicts L0

A pattern has two sides:

- its **context (sources)** — the neighbors at the level below that recognize it (L1's context = base
  neighbors; L2's context = L1 neighbors),
- its **prediction (targets)** — **always base (L0) neurons.** Only base neurons are ever a prediction output;
  corrections are never dequantized (§2.2).

So the hierarchy is **context depth grounding out in L0**, not a tower of abstractions. An L1 pattern predicts
the L0 reality in its region from its immediate context; an L2 predicts the *correct* L0 where L1 was wrong,
from *more* context. More context (higher level) → sharper L0 prediction. Every level answers the same
question — "what is the base reality here?" — with progressively more surrounding context.

### 3.2 Recognition: a context match fires a prediction

A unit checks its **context** — its active neighbors (the neighborhood). If that context matches a known
pattern's stored context **≥ the merge threshold**, the pattern is *inferred* (fires) and predicts L0.

- **base level:** context = the **coordinate neighborhood** (active neighbors within the radius).
- **higher levels:** context = **footprint-touch** neighbors (§2.2).

First exposure has nothing to recognize, so the active neurons just **learn** — record their context and take
the default. Recognition (and therefore the whole hierarchy) only exists because the threshold lets an
*approximate* context match fire a learned pattern; at exact match (θ=1.0) patterns almost never re-fire on new
input and nothing climbs (§3.8).

### 3.3 Error → a correction request

A fired pattern's **L0 prediction can be wrong** (beyond the error threshold) *even though its context
matched* — the pattern was too general for this instance. That mismatch is the error, and the neuron votes a
**correction request** to the thalamus. A novel image produces **hundreds** of these — wherever the image
deviates from the accumulated expectation. Those requests are what gets clustered.

### 3.4 Clustering the requests — transitive merge by neighborhood

The per-frame correction requests are grouped by **transitively merging neighbor-connected requests** (i.e.
connected components over the neighbor relation): two requests are in the same cluster if a chain of neighbor
links connects them.

- **neighbor relation:** base = coordinate neighborhood; higher levels = footprints touch.
- **one correction per connected cluster.** Its **footprint** = the cluster's coverage; its **context** = the
  cluster's neighbors; its **targets** = the correct L0 those neighbors should predict.
- **a correction needs ≥ 1 neighbor (context).** An isolated request with no neighbor has no context to
  condition on and **cannot form a higher correction** — there is nothing to predict *from*.

Early on, with few patterns and a low threshold, this merges aggressively → **large, general, blurry
corrections** (think "one blob in the middle of the image"). That is expected, and the split force (§3.6)
carves them back down over time.

### 3.5 Reuse — the merge force, across frames and within

Before minting, the requests consult the **reverse index** ([Phase B](./neuron-reuse-index.md)): does an
existing pattern already predict this region's L0 ≥ the merge threshold? Hit → **reuse** (the requests wire to
it as parents); miss → mint. Reuse lands many contexts on one shared, general pattern — that is the
cross-frame convergence (§1.3).

**Matched patterns and new requests merge in one pool (the unification — see §3.5.1).** A matched/reused
pattern that neighbors a new cluster of requests is pulled into the same cluster: the requests wire to it and
the pattern **expands** (its footprint/context grow to cover the new region). So reuse is not only
mint-or-reuse — it is also **grow the reused pattern.**

#### 3.5.1 Unifying matched patterns with new requests

The clustering pool is **(this frame's matched/reused patterns) ∪ (this frame's new requests)**, transitively
merged by neighborhood (§3.4). Three outcomes per connected cluster:

1. **only new requests** → mint one new correction (§3.4).
2. **contains a matched pattern, no others** → ordinary reuse; nothing to merge.
3. **a matched pattern adjacent to new requests** → the requests wire to the matched pattern as new parents,
   and the pattern's footprint/context **expand** to absorb the new region. One growing pattern, not a new
   neuron beside it.

Expansion mutates a *shared* pattern (it has many parents), so it drifts the pattern toward the union of the
contexts that reuse it — which is exactly the merge force. It is held in check by the split force (§3.6); the
two together set how big patterns get. **Open: the precise expansion rule** — how much a matched pattern may
grow per frame, and whether expansion is gated by an overlap floor — is a design proposal here, to be settled
in the simulation.

### 3.6 Merge vs split — the two forces that set pattern size

- **Merge** (grow / generalize): in-frame transitive clustering (§3.4) + inter-frame reuse incl. expansion
  (§3.5). Pulls toward big, shared, general patterns.
- **Split** (shrink / specialize): **refinement** ([refinement.md](./refinement.md)) consolidates a pattern
  toward the common core of the contexts it matches; **forgetting** decays incidental parts. Pulls toward
  smaller, regional patterns.

Where the equilibrium lands — **one big general pattern** vs **many regional patterns** — depends on the rates
(merge threshold, forget/refine rates, and **adaptive error thresholds**). The regional-pattern regime the
hierarchy needs is the *balanced* middle, and whether it exists is an **empirical** question, not one settled
on paper. The **[simulation](./neuron-reuse-simulation.md)** is the instrument built to answer it.

### 3.7 How the hierarchy climbs — how L2 forms

L2 cannot form inside one cold image; it needs L1 to already be a learned, *firing* vocabulary. Over frames:

1. **L1 accumulates** — early frames mint general L1 corrections (large, blurry).
2. **L1 fires by recognition** — on a later image, base contexts match L1 patterns ≥ threshold, so **several
   L1 patterns fire**, each predicting L0 over its region.
3. **A general L1 mispredicts L0** — its context matched, but it is too general, so its L0 prediction is wrong
   (§3.3). The active regional L1 patterns are the **active parents** of that large shared L1 (some parents may
   be inactive this frame).
4. **It mints an L2** that predicts the **correct L0**, conditioned on **more context: the active parent's
   *other* L1 neighbors.** (Needs ≥ 1 such neighbor — no neighbor, no context, no L2 — §3.4.)
5. **L2s are formed** by transitively merging the mispredicting L1 patterns by **footprint adjacency**.

Self-similar all the way up: recognize → mispredict L0 → cluster the mispredictors → mint a more-contextual
correction that gets L0 right.

### 3.8 The merge threshold is load-bearing at every level

Recognition is an *approximate* context match (≥ threshold), not equality. Without it, a learned pattern almost
never re-fires on a new image, so multiple patterns never co-fire, and a level can never produce the
co-occurring mispredictions the next level is built from. The threshold is therefore **not a late
optimization** — it is what lets the hierarchy form at all, and it is the same threshold for recognition and
reuse (no separate `reuseMergeThreshold`; θ=1.0 disables both).

### 3.9 The simulation is wrong; new spec

The current [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) models the **wrong**
mechanism: one cold frame, every neuron erroring, no recognition, no cross-frame state, grouping by
identical observed-set. None of that matches the model above (no recognition, no predict-L0, no
transitive-merge clustering, no merge/split). It must be rebuilt around recognize → mispredict L0 → cluster →
mint/expand, across frames. The new specification is
[neuron-reuse-simulation.md](./neuron-reuse-simulation.md); the existing sim should be treated as obsolete.

---

## 4. Lifecycle, Activation, Inhibition

### 4.1 Multi-Parent References

Reuse and cluster mint both wire **many** parents to one correction, which breaks the single-host ownership
model (today a correction's strength lives in its one host's routing entry and it dies when that entry decays
— [neuron.rs](../brain/brain-core/src/neuron.rs); the forget rate is brain-wide uniform —
[brain.rs](../brain/brain-core/src/brain.rs)). The clean shape: per-parent routing entries on a shared,
coordinate-less neuron; **reference-counted reaping** (alive while any parent references it); `patterns.csv`
serializes **many** `(parent, strength)` rows per pattern. Because corrections are coordinate-less, there is
**no anchor to reconcile** across parents — only lifecycle and activation bookkeeping. Born with cluster mint;
mechanics in [neuron-reuse-frame.md](./neuron-reuse-frame.md).

### 4.2 Same-Frame Activation

A minted **or** reused correction is **installed into the erroring neuron's routing table and fires next
frame** — not the frame it is wired ([neuron.rs](../brain/brain-core/src/neuron.rs)). The erroring
parent's wrong vote this frame is already suppressed by the existing machinery (`get_suppressed_ages`,
[neuron.rs](../brain/brain-core/src/neuron.rs)). So reuse adds **no** same-frame activation to
inhibit: the originally-planned `correction_wired_this_frame` set has nothing to act on, and a reused neuron is
active this frame only via its **own** routing match (a normal recognition).

The remaining activation case is **multi-parent routing**: a shared neuron can be routing-matched from several
parents at different depths in one sweep. It is activated **at each matched depth** — the multi-depth
`neuron_states` holds a neuron active across levels rather than collapsing it to one — and needs **no** extra
inhibition set; the existing newly-minted-correction inhibition (fires next frame, this-frame vote suppressed
via `get_suppressed_ages`) carries over unchanged.

---

## 5. Benefits & Risks

### 5.1 Benefits

- **Symmetry** — one operation, one neighborhood primitive, one reuse mechanism across all distances.
- **Compact structure at every level** — no redundant neurons computing the same inference, spatial or
  temporal.
- **Transfer / generalization** — a shared correction links the contexts that predict the same L0 region;
  structure transfers rather than being re-minted. (Cross-position transfer — the *same shape at a different
  place* — is **not** provided; a correction fires only over its own footprint. That ceiling is the
  retinotopic/absolute-connection model, common to both spatial and temporal.)
- **Content-addressable** — "is there a neuron that predicts this L0 region?" answered via the reverse index.

### 5.2 Risks

- **Wave-front rearchitecture** (Phase A) is large and not bit-exact — characterized regression, not byte
  identity.
- **Footprint cost** at the apex (large footprints); bounded by bitsets, watch memory for huge base (vision).
- **Reverse-index cost** — one lookup per group, sharded by column, updates at orchestration boundaries.
- **Over-aggressive reuse** — too-low merge threshold; tune cautiously. There is **no decay backstop** today
  (connections never weaken — [neuron.rs](../brain/brain-core/src/neuron.rs)), so a bad reuse
  persists. The **merge threshold is the only control** — candidacy is strength-blind (§3.2).
- **Action-binding dilution** — heavily-reused neurons; monitored long-run.
- **Merge/split equilibrium is unproven** — the merge force (clustering + reuse, §3.4–3.5) and the split force
  (refinement + forgetting, §3.6) must balance so patterns settle into *regional* sizes rather than collapsing
  to one big blurry pattern (or staying tiny and never reusing). Whether a stable middle exists depends on the
  rates (merge threshold, forget/refine, adaptive error) and is an **empirical** question — the
  [simulation](./neuron-reuse-simulation.md) is built to answer it.
- **Pattern expansion drift** — growing a matched pattern to absorb adjacent requests (§3.5.1) mutates a shared
  neuron for all its parents; the expansion rule and its overlap gate are unsettled.

---

## 6. Implementation

Build order: wave-front foundation → index → cluster + mint → lookup/expansion, then validation. Reuse applies
at all distances throughout. The **incremental migration + verification plan** (sim-as-oracle, step-by-step against
MNIST numbers, merge threshold deferred) is in
[neuron-reuse-wavefront-implementation.md](./neuron-reuse-wavefront-implementation.md).

| Phase | Doc | Goal | Gate |
|---|---|---|---|
| **A** | [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md) | **Foundation.** Turn `process_spatial` and `process_temporal` into settling waves (structure kept); remove stored levels; coordinate-less corrections; footprints for neighborhood in both waves; multi-depth memory. | Characterized regression (MNIST + stocks comparable); footprint-adjacency units. **Not bit-exact.** |
| **B** | [neuron-reuse-index.md](./neuron-reuse-index.md) | **Two** reverse connection indexes — `spatial_connection_index` (target → sources) and `temporal_connection_index` (target → distance → sources), mirroring the context-index split. Built, unit-tested, not yet consumed. Membership-only / strength-blind candidacy. | Unit: target→sources correct per index; stores isolated; size ∝ connection count. |
| **C** | [neuron-reuse-frame.md](./neuron-reuse-frame.md) | **Cluster + mint** — transitively merge neighbor-connected correction requests (coordinate neighborhood at base, footprint-touch higher); one correction per connected cluster, predicting the correct L0; **+ multi-parent machinery** (refcounted reaping, multi-parent serialization, shared-neuron activation). | Connected clusters mint one correction each; multi-parent lifecycle units. |
| **D** | [neuron-reuse-final.md](./neuron-reuse-final.md) | Reuse **lookup + expansion** on top of C (consumes B): a request reuses a pattern whose L0 prediction matches ≥ threshold; a matched pattern adjacent to new requests **expands** to absorb them. Cross-frame accrual, all distances. | Cross-frame reuse via lookup; expansion grows matched patterns; neuron count drops; lookup < 20% per-frame. |
| **Validation** | [neuron-reuse-validation.md](./neuron-reuse-validation.md) | MNIST (spatial reuse, within-frame + cross-image), stocks (full pipeline + transfer), long-run forget-rate. | Per-experiment gates in the doc. |

All phases depend on [spatial-processing](./spatial-processing.md) being complete (it is) — Phase A then
rebuilds its level model into the wave-front.

**Reference simulation.** The current [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js)
is **obsolete** — it models the wrong mechanism (cold-start, single-frame, no recognition, group-by-observed-set;
§3.9). The correct simulation is specified in [neuron-reuse-simulation.md](./neuron-reuse-simulation.md): a
multi-frame, recognition-based model (recognize → mispredict L0 → transitive-merge cluster → mint/expand) that
is both the instrument for the merge/split equilibrium question (§3.6) and the oracle for the eventual port.
