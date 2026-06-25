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
2. **Reuse (Phases B–D).** The reuse criterion is **inference-output match**: does an existing neuron's
   connection set already produce the inference a correction would need? Per (distance, footprint-region) per
   frame, query a reverse index for a match (partial or exact); on a hit, wire all co-failing neurons to the existing neuron; on
   a miss, mint **one** coordinate-less correction (footprint = union) and wire all co-failers to it.

Because corrections are coordinate-less, a shared correction takes the **union footprint** of its parents —
no anchor to be ambiguous about. That is what makes reuse clean at d=0, not just d>0.

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

## 3. Reuse Mechanism (all distances)

**Reuse precedence — four steps, in order.** Each frame, after the wave settles and errors are collected:

1. **Reuse lookup (per request).** Each erroring neuron queries the reverse index for an existing neuron whose
   inference output already covers its observed reality ≥ the merge threshold (§3.2). Hits are marked for reuse.
2. **Group the remainder.** The requests with no match group by (distance, observed-set) — co-failers about the
   same un-represented reality (§3.3).
3. **Mint per group.** One new coordinate-less correction per residual group (§3.3).
4. **Wire all together.** Every erroring neuron is then wired in one batched step — to its reused neuron (step
   1) or to its group's fresh mint (step 3).

Two invariants make this clean:

- **Grouping never gates reuse.** Grouping is only a batching step for the un-reusable residual; a neuron's
  group never constrains which existing pattern it may reuse. Reuse is decided per request, before any grouping.
- **Same reality → same pattern (automatic consolidation).** The lookup query *is* the observed reality, so two
  neurons observing the same reality issue the same query and resolve to the same neuron; and the residual
  groups by observed-set, so each distinct un-reusable reality mints exactly one neuron. Every distinct observed
  reality therefore maps to exactly one pattern — reused or minted — with no fragmentation and no clustering.

The four steps are **distance- and level-invariant** — identical at d=0 (spatial) and d>0 (temporal) and at
every level (§3.4). Executable reference:
[`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) (§3.6).

### 3.1 Observed reality — the per-neuron query and group key

For each erroring unit N, `observed(N)` = the co-active units whose footprint **touches** N's in the base
neighbor graph (adjacency, not set-overlap — §2.2). A unit predicts its neighbors, never itself, so N is
excluded from its own observed-set. This observed set is used twice: as the **lookup query** (§3.2) and, for
the residual, as the **group key** (§3.3). At a given (distance, footprint-region) the observed reality is
**singular** — one actual co-activation (d=0) or realized future (d>0) this frame — so the group key is exact
and well-defined, with no clustering algorithm and no anchor.

### 3.2 Lookup — reuse an existing pattern (step 1)

Per erroring neuron, query the **reverse inference index** ([Phase B](./neuron-reuse-index.md)): for each
observed target T, which existing neurons connect to T at this distance? Candidacy is **strength-blind** — a
neuron is a candidate if it has a connection to T, regardless of that connection's strength; strength governs
voting and recognition, not reuse. Score each candidate against the observed set (common/missing/novel, the
pattern-recognition scoring). If the best ≥ the **merge threshold for
this distance**, the neuron reuses that existing pattern. Because the query is purely the observed reality,
co-failers issue identical queries and resolve identically — same reality, same pattern. Lookup is the
cross-frame collapse and *is* the generalization mechanism (§1.3): a neuron reused across frames accumulates
the structural core common to the contexts that reuse it. Details:
[neuron-reuse-final.md](./neuron-reuse-final.md).

### 3.3 Group and mint the residual (steps 2–3)

The neurons that found no match group by (distance, observed-set) — co-failers about the same observed reality.
Per group, mint **one** coordinate-less correction (wired in step 4). Two rules:

1. **Footprint = the bound cluster.** `footprint = union(co-failers' footprints) ∪ observed-set's footprints`.
   The observed set must be included: co-failers' own footprints may be **disjoint** (four arms around a center
   share the center as their observed *target*, not their coverage), so unioning only the co-failers would
   leave a hole. Including the observed set keeps the footprint connected.
2. **Isolated units never merge.** A unit with an *empty* observed-set (no neighbors) is its own group; empties
   are never grouped together — that would bind disconnected structure and violate locality.

Minting is within-frame only, which is *why* lookup precedes it: the same reality recurring next frame is, to
the mint path alone, a fresh group that mints a duplicate; lookup is what collapses it across frames. Details:
[neuron-reuse-frame.md](./neuron-reuse-frame.md).

### 3.4 Levels — the same operation, self-similar, one ring at a time

The four steps run per level. At level 1 the units are base neurons (footprint = {self}); at level k the units
are the corrections produced at level k−1. Neighbor at every level is footprint-touch; observed-set, grouping,
and the union∪observed footprint are identical. Binding is **radius-local** — a correction binds a unit with
its footprint-adjacent co-active neighbors, one ring — so the footprint grows ~one ring per level. A
whole-shape pattern is therefore a *high-level* object reached after several binding steps, never a level-1
one; if level 1 swallowed the shape there would be nothing left to compose.

### 3.5 Merge threshold — and why exact match alone is brittle

Reuse reads the **same merge threshold pattern recognition uses at that distance** (spatial at d=0, temporal at
d>0). No separate `reuseMergeThreshold`; setting a distance's threshold to 1.0 disables reuse and partial
recognition together.

The threshold matters more than it first appears. Both the lookup and the residual grouping turn on **observed
reality**, and the simulation (§3.6) shows that *exact* match (threshold = 1.0) fires reuse only where there is
local symmetry: on a clean plus the arms share an observed center and reuse strongly, but on irregular or solid
shapes almost every unit sees a unique neighbor set, so within-frame reuse is near-zero (every unit a
singleton). On natural input (MNIST), exact-match reuse is therefore rare — almost all consolidation must come
from cross-frame **lookup** (§3.2) under a merge threshold < 1.0, plus refinement
([refinement.md](./refinement.md)). The threshold is what lets *partial*-overlap realities resolve to the same
pattern; exact match is the degenerate threshold=1.0 case.

### 3.6 Worked examples (and the simulation)

The reference simulation [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) runs the
grouping + mint path per level on ASCII shapes (`node apps/mnist/jobs/wavefront-sim.js`). It models one frame,
d=0, every unit erroring with **no index lookup** — i.e. steps 2–4 (group → mint → wire), the residual path on
its own. Key results:

- **Single plus** (center + 4 arms). The four arms each observe `{center}` → identical observed-set → **one
  shared correction, 4 parents**, footprint = arms ∪ center (the whole plus, center included). One frame's
  clearest within-frame sharing.
- **Plus-of-pluses** (five pluses arranged as a plus). **Level 1 stays local**: max footprint 5px of 21px —
  five small plus-patterns, never the whole shape. **Level 2** binds the four outer plus-patterns (they each
  observe the center plus-pattern) into the whole figure — the *same* arms-around-a-center grouping one level
  up. **Level 3** converges to one pattern. Self-similar, footprints 5 → 12 → 21.
- **Irregular shapes** (uneven blob, ragged patch, solid block, L). Level 1 stays local (≤5px) on all of them,
  but **shared groups drop to 0–1** — exact observed-set match barely fires without symmetry (§3.5). This is
  the empirical case for the merge threshold and cross-frame lookup.
- **Disconnected input** (two separate blobs, a 4-connected diagonal). Stays separate — the isolated-unit rule
  (§3.3) keeps locality from binding across gaps. Diagonals bind only under 8-connectivity, a base-graph knob
  worth choosing deliberately for MNIST's diagonal strokes.

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
  persists. The **merge threshold is the only control** — candidacy is strength-blind (§3.2).
- **Action-binding dilution** — heavily-reused neurons; monitored long-run.
- **Within-frame reuse is near-inert on irregular input** — exact observed-set match (§3.5) fires only on
  local symmetry; on natural/asymmetric shapes nearly every unit is a singleton (shown in the reference
  simulation, §3.6). Consolidation therefore leans almost entirely on cross-frame **lookup** + **refinement**,
  not on within-frame grouping. The merge threshold (< 1.0) is what makes partial-overlap reuse possible.

---

## 6. Implementation

Build order: wave-front foundation → index → batched mint → lookup, then validation. Reuse applies at all
distances throughout.

| Phase | Doc | Goal | Gate |
|---|---|---|---|
| **A** | [neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md) | **Foundation.** Turn `process_spatial` and `process_temporal` into settling waves (structure kept); remove stored levels; coordinate-less corrections; footprints for neighborhood in both waves; multi-depth memory. | Characterized regression (MNIST + stocks comparable); footprint-adjacency units. **Not bit-exact.** |
| **B** | [neuron-reuse-index.md](./neuron-reuse-index.md) | **Two** reverse connection indexes — `spatial_connection_index` (target → sources) and `temporal_connection_index` (target → distance → sources), mirroring the context-index split. Built, unit-tested, not yet consumed. Membership-only / strength-blind candidacy. | Unit: target→sources correct per index; stores isolated; size ∝ connection count. |
| **C** | [neuron-reuse-frame.md](./neuron-reuse-frame.md) | **Batched mint** at all distances (group by (distance, observed-set), mint one coordinate-less correction with union footprint, wire all) **+ multi-parent machinery** (refcounted reaping, multi-parent serialization, shared-neuron activation). | Within-group dedup drops neuron count; multi-parent lifecycle units. |
| **D** | [neuron-reuse-final.md](./neuron-reuse-final.md) | Reuse **lookup** on top of C (consumes B) + cross-frame accrual, all distances. | Unit: cross-frame reuse via lookup; neuron count drops further; lookup < 20% per-frame overhead. |
| **Validation** | [neuron-reuse-validation.md](./neuron-reuse-validation.md) | MNIST (spatial reuse, within-frame + cross-image), stocks (full pipeline + transfer), long-run forget-rate. | Per-experiment gates in the doc. |

All phases depend on [spatial-processing](./spatial-processing.md) being complete (it is) — Phase A then
rebuilds its level model into the wave-front.

**Reference simulation.** [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) is the
executable spec for the §3 grouping: a clean, dependency-free proof of concept that runs the per-level
operation on hand-drawn shapes (`node apps/mnist/jobs/wavefront-sim.js`). It currently models one spatial frame
— grouping + mint, no index lookup or merge threshold — and is the place to validate algorithm changes before
touching the brain. The intent is to grow it toward the port incrementally (add the index lookup, the
threshold, then temporal distances), keeping it faithful enough that it maps cleanly onto `process_spatial` /
the Phase-C/D code.
