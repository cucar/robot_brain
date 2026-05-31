# Spatial Processing

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Status:** Pre-implementation
**Prerequisites:** [mnist-merge.md](./mnist-merge.md), [inference-level.md](./inference-level.md)
**Companion:** [neuron-reuse.md](./neuron-reuse.md)

---

## 1. Motivation

### 1.1 The MNIST Experiment

The Robot Brain was tested on MNIST data using the following setup: 784 pixel channels (reduced to 49 for compute constraints) fed as independent input channels to the brain, with repeated frames shown until cortex stabilization, followed by action implantation for digit labels. The cortex-only approach produced correct but catastrophically inefficient results — requiring impractical computation time to build a useful representational tree.

### 1.2 The Core Insight

The cortex's pattern recognition mechanism is designed for temporal sequences — it detects "A then B" across time distances. MNIST digits have no temporal structure. Feeding the same frame repeatedly and asking the cortex to build hierarchy through temporal repetition forces it to represent spatial co-activation as a flat combinatorial expansion. This is equivalent to writing a logical expression without parentheses — the expression is valid but combinatorially intractable.

### 1.3 The Role of Spatial Processing (d=0)

Spatial connections (d=0) provide the "parentheses" — temporary storage for co-activation patterns that compress the representational space. They answer the question "what belongs together?" so that temporal connections (d>0) can efficiently answer "what follows what?" Without spatial grouping, the cortex must pattern-match over the full combinatorial space of raw inputs. With it, the cortex operates over a compressed vocabulary of pre-grouped units.

This is not an MNIST-specific optimization. It is core Brain functionality required for any domain where spatial or simultaneous structure exists — vision, point-in-time market states, sensor arrays, and any multi-channel sensory input.

### 1.4 Same Mechanism, Different Distance

Spatial processing is not a separate system bolted onto the cortex. It is the same mechanism — prediction, error detection, correction — applied at distance zero. A neuron predicting what comes next at d=3 and a neuron predicting what's active alongside it at d=0 use identical infrastructure: connections with strength, routing tables, thalamus orchestration, error-driven creation. The only difference is the distance parameter.

This eliminates the need for separate "moment neuron" and "pattern neuron" types. A neuron is a neuron. It has connections at various distances including zero. It fires when its predictions match. The distance determines whether the match is temporal or spatial.

---

## 2. Biological Parallels

### 2.1 Temporal Patterns as Pyramidal Cells

Cortical pyramidal cells are the canonical sequence detectors. They receive top-down context via apical dendrites and feedforward input via basal dendrites, learning temporal predictions. Temporal patterns in Robot Brain map directly to this: they detect "A then B" with context sensitivity, using routing tables with time-distance connections.

### 2.2 Spatial Corrections as Cortical Lateral Grouping

Within-frame grouping in the brain happens through cortical lateral and feedforward connectivity — V1 lateral inhibition groups edges, V2 groups edges into shapes, V4 groups shapes into objects, all within the time it takes a single visual frame to propagate up the ventral stream. Spatial corrections in this design play the same role: they fire when co-activations match, building hierarchical groupings within a single frame's spatial wavefront.

(The "moment neurons" framing from hippocampal literature describes a different mechanism — episodic encoding of "what happened at this moment" in the hippocampus proper. That's [hippocampus.md](./hippocampus.md). This document is about the cortical-lateral-grouping mechanism, not the hippocampal one.)

### 2.3 The Two Complementary Operations

- **Pyramidal cells / d>0 connections**: learn *when* (temporal sequences across frames).
- **Cortical lateral grouping / d=0 connections**: learn *what together* (spatial co-activation within a frame).

The brain requires both operations to build efficient representations.

---

## 3. Architecture

### 3.1 Core Principle: Unified Distance-Based Prediction

All processing uses a single mechanism — neurons predicting what other neurons will be active, differing only in the distance parameter:

| Property | Temporal (d > 0) | Spatial (d = 0) |
|---|---|---|
| **Prediction** | "What comes next at distance d" | "What's active alongside me right now" |
| **Connection semantics** | Time-distance connections to future events | Co-activation connections to concurrent neurons |
| **Error condition** | Predicted sequence didn't occur | Predicted co-activations didn't match |
| **Creation trigger** | Thalamus detects prediction error | Thalamus detects co-activation prediction error |
| **Routing** | Via thalamus, same infrastructure | Via thalamus, same infrastructure |
| **Neuron type** | Same | Same |
| **Forgetting** | Standard decay | Same decay (global static, set in [mnist-merge.md](./mnist-merge.md)) |

### 3.2 The Neuron's Perspective

Every neuron maintains connections indexed by distance. Connections at d>0 are temporal predictions ("neuron B will be active d frames from now"). Connections at d=0 are co-activation predictions ("neuron B should be active right now alongside me").

The neuron does not distinguish between temporal and spatial connections — they are entries in the same connection table.

### 3.3 Two-Phase Processing Model

Each frame is processed in two **sequential phases**: spatial first, temporal second.

The motivation is the feedforward sensory cascade. Object identification must precede motion tracking — you cannot have a d>0 prediction "ball moves left" until "ball" is a recognized entity in both frames. If temporal ran first, its d>0 predictions could only operate on raw sensory neurons (the only thing active at age=0 within the frame), which is the combinatorial-explosion bottleneck the design was meant to fix. By running spatial first, temporal sees the spatial hierarchy's apex as its input — objects and groupings, not raw pixels — and its d>0 predictions form between meaningful entities across frames.

Spatial and temporal are structurally symmetric: each has its own level hierarchy and its own per-frame processing loop. They differ only in distance — spatial uses d=0 (within-frame co-activation), temporal uses d>0 (across-frame sequence) — and in how their hierarchies are realized within a single frame:

- **Spatial:** wavefront within the frame. Seeds from L=0 sensory; routing matches and fresh mints propagate hierarchy upward wave by wave.
- **Temporal:** level-sweep within the frame (the existing `process_levels()` pipeline, renamed `process_temporal`). Iterates `for level in 0..max_active_level`; at each level, d>0 pattern recognition uses the prior-frame context window.

```
Frame arrives
    │
    ▼
Activate sensory neurons (L=0)
    │
    ▼
process_spatial  (new)
    │
    │   Input: L=0 sensory neurons.
    │
    │   Spatial wavefront:
    │     ├─ Wave 0: L=0 sensory fires. Their d=0 connections
    │     │           activate spatial corrections via routing.
    │     ├─ Wave 1: newly-activated corrections fire. Their d=0
    │     │           connections activate further corrections.
    │     ├─ Wave N: continues until no new activations.
    │     │           Refractory: each neuron fires at most once
    │     │           per spatial phase.
    │     ▼
    │   At spatial stabilization:
    │     • Evaluate d=0 predictions against the observed
    │       co-activation set (= fired neurons in this spatial
    │       phase, minus correction_wired_this_frame).
    │     • For each error: reuse lookup (see neuron-reuse.md),
    │       else mint correction neuron at
    │       erroring_neuron.activation_level + 1. Correction
    │       targets go into correction_wired_this_frame.
    │
    │   Handoff: spatial output = highest-level activations per age
    │            (= fired neurons in this spatial phase whose level
    │            equals the max active level for their age, minus
    │            correction_wired_this_frame).
    │
    ▼
process_temporal  (existing level-sweep, unchanged for d>0)
    │
    │   Input: sensory + spatial handoff (folded into the active
    │   set at each level).
    │
    │   For level 0..max_active_level:
    │     • d>0 pattern recognition
    │     • d>0 error detection and correction
    │
    ▼
Action voting
    │   Over the union of all voting neurons (fired_this_frame
    │   minus correction_wired_this_frame), across both phases.
    │
    ▼
Next frame
```

**Spatial inputs.** `process_spatial` seeds from L=0 sensory neurons activated this frame. The wavefront then propagates upward via existing d=0 routing matches: a fired neuron's routing table is consulted; matches activate the corresponding higher-level spatial correction; that correction fires in the next wave; its own routing is consulted; and so on. Each frame can deepen the spatial hierarchy by up to one fresh mint layer at the top.

**Spatial→temporal handoff.** `process_temporal`'s active-set construction is enriched with the spatial output: of the neurons fired in `process_spatial` that are eligible to vote (not in `correction_wired_this_frame`), the subset whose activation level equals the max active level for their age. For the MNIST first frame — no spatial hierarchy yet — the handoff is just the L=0 sensory neurons. For mature operation, the handoff is whatever apex spatial groupings emerged from the wavefront. Lower-level neurons that contributed to higher-level spatial corrections are excluded from the handoff; their relationships are already encoded by the corrections that absorbed them.

**Inference scope.** The validation set for a d=0 prediction — same-level fired neurons, base-level fired neurons, or all fired neurons regardless of level — is determined by the [inference-level experiment](./inference-level.md). The winning rule applies uniformly to d=0.

### 3.4 Refractory and Correction-Wiring Inhibition

Each neuron fires at most once per spatial phase. A `fired_this_frame: FxHashSet<NeuronId>` set tracks all neurons that have fired in either phase this frame, enforcing refractory within the spatial wavefront and preventing temporal re-firing of spatial activations. This bounds the spatial wavefront by total neuron count.

Additionally, a `correction_wired_this_frame: FxHashSet<NeuronId>` set tracks every neuron whose activation this frame is the result of being selected as a correction target — whether it was minted fresh or reused. Neurons in this set:

- **Learn from the current observed set.** They participate in connection strengthening so that the d=0 connections of a reused neuron gradually generalize toward the observed reality across reuse events (see §5 generalization, and [neuron-reuse.md](./neuron-reuse.md)).
- **Do not vote.** Their activation this frame is a wiring side-effect, not an inferential signal. Action voting excludes them.
- **Are not error-checked.** Their d=0 predictions are not evaluated this frame. This prevents a reused neuron — whose pre-existing d=0 set may not match the current observed set — from generating a fresh error and cascading into more corrections within the same phase.

The inhibition rule is unified: the trigger isn't novelty (newly minted), it's role this frame (correction-wired). Minted neurons were always a special case where their predictions trivially matched reality; reused neurons need explicit exclusion because their predictions do not trivially match.

Both sets clear at frame end.

### 3.5 Levels as Activation State, Not Neuron State

Levels are a property of *activations in active memory*, not of neurons. A neuron has no intrinsic level field. When a neuron is activated this frame, it is registered in active memory at `activating_neuron.activation_level + 1` (sensory neurons start at activation level 0). The temporal level-sweep iterates active-memory levels: at iteration L, it processes whoever is activated at level L this frame. The spatial wavefront propagates by routing match independent of level.

This design matters specifically because of reuse (see [neuron-reuse.md](./neuron-reuse.md)). A neuron R reused for an error at A (activation level 2) appears at activation level 3 this frame, regardless of where R was originally minted or where it was last activated. Next frame, the same R may be activated from a different routing source at a different level — appearing at a different activation level. R's identity is preserved; its "level" is per-frame contextual.

Without this design, cross-level reuse would either be unsafe (the level-sweep might never reach R's intrinsic level, leaving R's d>0 work and votes silently dropped) or require restricting reuse candidates to matching levels (shrinking the reuse pool significantly). With per-activation levels, neither problem exists.

---

## 4. Co-Activation Prediction and Error

### 4.1 Connection Building

Co-activation connections (d=0) are built through observation, identical to temporal connections. When neurons A and B are co-active in a frame's spatial phase, both strengthen their d=0 connections to each other. Over repeated exposures, strongly co-activating neurons develop strong mutual d=0 connections.

### 4.2 Co-Activation as Prediction

A neuron's d=0 connections constitute predictions: "when I fire, I expect these neurons to also be firing." This is the spatial analog of temporal prediction ("when I fire, I expect neuron B at distance d").

### 4.3 Error Detection

When a neuron fires during `process_spatial`, its d=0 connections constitute a predicted co-activation set. After spatial wavefront stabilization, the thalamus compares this predicted set against **reality = the neurons fired in this spatial phase, minus `correction_wired_this_frame`** (scoped per the [inference-level decision](./inference-level.md)). Missing predictions (expected neurons that didn't fire) and novel observations (unexpected neurons that did fire) both contribute to the mismatch rate. If the mismatch exceeds the error threshold, an error is generated. Neurons in `correction_wired_this_frame` do not produce predictions for evaluation.

### 4.4 Error Correction

The thalamus first checks for a reusable existing neuron via the inference index (see [neuron-reuse.md](./neuron-reuse.md)). If found, wires the erroring neuron's routing table to point at the reused neuron. If not, mints a fresh correction neuron capturing the actual co-activation context. The correction target is registered in active memory at `erroring_neuron.activation_level + 1`.

### 4.5 Bootstrap Dynamics

1. **First exposure**: Neurons A, B, C co-activate. No d=0 connections exist. No predictions, no errors. Thalamus records co-activations, builds d=0 connection entries: A↔B, A↔C, B↔C.
2. **Repeated exposures**: Same neurons co-activate. d=0 connections strengthen. Each neuron now predicts its co-activation partners. Predictions match reality. No error. No correction neurons needed.
3. **Conflicting exposure**: A new input activates A, D, E, F, G. Neuron A predicted B and C (its established d=0 partners), but sees D, E, F, G instead. Error exceeds threshold. Thalamus mints (or reuses) correction neuron C1 capturing the actual co-activation set. Updates A's routing table: in context of D, E, F, G, activate C1.
4. **Subsequent exposures**: When A, B, C recur — A predicts B and C correctly, no error, default wiring handles it. When A, D, E, F, G recur — A recognizes context, activates C1 directly.

### 4.6 Two Modes of Representation

- **First-learned patterns**: Represented in default d=0 connections of raw neurons. No correction neuron needed. Actions bind directly to the sensory neurons.
- **Subsequent conflicting patterns**: Represented by correction neurons created from prediction errors. Actions bind to the correction neurons.

---

## 5. Hierarchical Spatial Grouping

### 5.1 How Hierarchy Emerges

Spatial hierarchy emerges along two axes — within a single spatial phase (via the spatial wavefront and existing routing) and across frames (via the routing tables accumulating new entries over time).

**Within a spatial phase.** When `process_spatial` runs in frame N:

- Wave 0: L=0 sensory fires. d=0 routing matches activate L=1 spatial corrections (those minted in prior frames).
- Wave 1: L=1 corrections fire. d=0 routing matches on them activate L=2 spatial corrections.
- Wave K: continues until no new routing matches.
- At stabilization: d=0 predictions evaluated. For each error → mint or reuse correction at `erroring_neuron.activation_level + 1`. Fresh mints/reuses go into `correction_wired_this_frame` and do not fire this frame.

A single frame can therefore deepen the spatial hierarchy by at most one fresh layer at the top, plus however deep the existing routing structure already reaches.

**Across frames.** Frame N's fresh mints become routing-table entries in their source neurons. In frame N+1's spatial wavefront, those entries match and fire the corresponding correction neurons mid-wavefront — so the wavefront naturally reaches one level deeper than it did in frame N. Over many frames, the spatial hierarchy reaches arbitrary depth, even though any single frame only adds one fresh layer.

**Spatial→temporal handoff.** The highest-level activations per age from this frame's spatial wavefront are folded into `process_temporal`'s active set. Temporal then does its own level-sweep over the enriched set, learning d>0 sequences over spatial groupings rather than over raw pixels.

### 5.2 Walkthrough: MNIST Digits "1" and "7"

Using a simplified 3x3 grid:

**Learning "1"** (vertical stroke: pixels A, B, C):
- Multiple exposures build strong A↔B, A↔C, B↔C d=0 connections.
- No conflicts yet. Default wiring represents "1". Action "1" wired to A, B, C directly.

**Learning "7"** (horizontal top + diagonal, sharing pixel A with "1"):
- A fires with D, E, F, G instead of B, C. Error for A. Thalamus mints correction C1(D, A, E, F, G).
- A's routing table updated: context D, E, F, G → activate C1.
- Action "7" wired to C1.

**Inference on new "1"**: Pixels A, B, C fire → A predicts B and C, correct → default wiring → votes for action "1".

**Inference on new "7"**: Pixels D, A, E, F, G fire → A recognizes context → activates C1 → C1 votes for action "7".

### 5.3 Single-Frame MNIST Processing

With spatial-first, MNIST processing becomes:

1. Show one frame. L=0 pixel neurons activate.
2. `process_spatial` runs. Spatial wavefront seeds from the L=0 pixels. On first exposure, no corrections exist yet — wavefront stops at Wave 0, mints corrections from d=0 errors. On subsequent exposures, routing matches activate previously-minted corrections, the wavefront deepens, and possibly one new layer mints at the top.
3. `process_temporal` runs. The handoff becomes part of temporal's active set. For single-frame MNIST there's no cross-frame context, so temporal does no useful work, but it runs uniformly for architectural consistency.
4. Action voting runs across all voting neurons (sensory + spatial activations from routing matches), excluding `correction_wired_this_frame`.
5. `learn()` wires the digit action to the active voting set.

One frame per image. Training loop: show image → spatial + temporal phases run → learn digit → next image.

Inference is symmetric: on a test image, the spatial wavefront activates pre-trained correction neurons via routing matches (not as fresh wirings), and those activated neurons vote for the digit action.

### 5.4 Generalization via Reuse and Decay

Generalization requires both mechanisms working together:

1. **Reuse (primary driver).** When the same correction neuron is reused across many distinct error events (see [neuron-reuse.md](./neuron-reuse.md)), each reuse strengthens the connections shared across all those events and adds new connections specific to each event. Structurally-shared connections accumulate strength; per-event-specific connections remain weak.
2. **Decay (secondary driver).** Standard global static decay erodes weak per-event connections faster than reinforced structural ones.

Without reuse, a single correction neuron created from one event only memorizes that event — decay alone has no statistical basis to identify which connections are incidental. Reuse provides the cross-instance signal that decay then sharpens.

---

## 6. Implementation Plan

### Overview

| Phase | Goal | Validation gate |
|---|---|---|
| 1 | Spatial-phase scaffolding + intrinsic-level removal | Stocks regression: bit-exact vs inference-level winner |
| 2 | d=0 connection learning (per-neuron, parallel) | Unit test: co-active neurons mutually strengthen d=0 |
| 3 | Spatial wavefront orchestration | Unit test: wavefront terminates, predictions accumulate |
| 4 | d=0 error detection + correction minting (no reuse yet) | MNIST single-frame harness produces correction neurons on conflicting digits |
| 5 | MNIST single-frame harness, mint-only | >50% accuracy on test set with <1000 training images |
| 6 | Stocks integration (spatial-only) | Spatial wavefront active on stocks; directional accuracy ≥ baseline |
| 7 | Persistence / backup / import-export updates | Snapshot/restore round-trips d=0 connections and dynamic-level corrections |

Reuse is added in a separate workstream — see [neuron-reuse.md](./neuron-reuse.md). Phases 4 onward describe the "mint-only" fallback path; reuse plugs in ahead of mint when its workstream lands.

---

### Phase 1 — Spatial-Phase Scaffolding + Intrinsic-Level Removal

**Goal:** Two structural changes that need to land together because they touch the same dispatch path:

1. Set up the spatial-first two-phase pipeline: add an empty `process_spatial()` before the renamed `process_temporal()`. Plumb `fired_this_frame` and `correction_wired_this_frame`. Route action voting through the combined fired set.
2. Remove `neuron.level` as an intrinsic field. Level becomes a property of *activations in active memory*, not of neurons. The level-sweep iterates active-memory levels: "process whoever's active at level 0, then 1, then 2, …" Activation level is determined by the routing source (one above the activating neuron).

**No d=0 connection learning, no d=0 errors, no spatial wavefront body yet** — Phase 1 is structural scaffolding. The temporal pipeline's pattern-matching internals stay as-is; only the dispatch lookup and the per-neuron level field change.

#### Code touched

- `brain/brain-core/src/brain.rs`:
  - Rename `process_levels()` → `process_temporal()`. Internals unchanged.
  - Add `process_spatial()` — empty stub with a `// Phase 3 body lands here` marker.
  - Per-frame pipeline now calls `process_spatial()` **then** `process_temporal()` sequentially.
  - Add per-frame state `fired_this_frame: FxHashSet<NeuronId>` and `correction_wired_this_frame: FxHashSet<NeuronId>`. Both reset at frame start.
  - Both phases populate `fired_this_frame` as neurons activate.
  - Action voting reads `fired_this_frame \ correction_wired_this_frame` instead of per-level accumulation. With `process_spatial` empty and `correction_wired_this_frame` empty, behaviorally identical to today.
- `brain/brain-core/src/neuron.rs`:
  - **Remove `neuron.level` field.** Drop from struct, drop from serialization, drop from all per-neuron API surface.
  - No per-neuron forget-rate field — already a global static (from [mnist-merge](./mnist-merge.md)).
- `brain/brain-core/src/memory.rs`:
  - `level_index` becomes per-frame active-memory state, not persistent metadata. Reconstructed each frame from activations.
  - `get_level_neurons(level)` returns "neurons activated at this level this frame" rather than "neurons whose intrinsic level == L."
  - Sensory neurons activated at the start of a frame are at activation level 0.
  - For newly-minted correction neurons, activation level this frame is `activating_neuron.activation_level + 1`.
- `brain/brain-core/src/thalamus.rs`:
  - No signature changes for `process_level`. The `level` arg now means "current iteration level," consumed when reading active memory.
  - Add `correction_wired_this_frame` as a thalamus-owned field (mutated when corrections are wired; cleared per frame).
  - Existing d>0 correction path adds newly-minted/reused targets to `correction_wired_this_frame`.
- `brain/brain-core/src/column.rs`, `region.rs` — sweep for any code that references `neuron.level` and replace with activation-level lookup or remove if vestigial.

#### Acceptance

- Stocks pipeline output is **bit-exact** vs the inference-level winner on a fixed seed. Despite the substantial internal refactor, no neuron sees different inputs or emits different outputs when `process_spatial` is empty. Drift indicates a refactor bug.
- Zero new neurons created relative to the inference-level baseline.
- `process_spatial()` is reached every frame, runs in <1µs (empty stub), produces no side effects.
- `neuron.level` field gone from the struct and from serialized form. Loading a pre-Phase-1 snapshot errors clearly (decide during implementation; lean toward error + version bump).

#### Voting equivalence

Voting moves from per-level accumulation (today, via `level_age_state`) to a single pass over `fired_this_frame \ correction_wired_this_frame`. In the parallel-per-neuron model, voting is "each neuron emits its votes; thalamus aggregates" — today aggregation happens per-level, in Phase 1 it happens once over the union. With `correction_wired_this_frame` empty and the same neurons firing in both code paths, the aggregated vote set is identical by construction. The bit-exact gate above covers this; no separate unit test required.

#### Notes / gotchas

- The temporal level-sweep keeps its existing per-iteration dispatch and parallelism. What changes is the lookup at the top of each iteration.
- Activation-level assignment is "activating neuron's activation level + 1." A reused neuron R activated from A (at level 2) appears at level 3 this frame, even if R was originally minted at a different level for a different context. This is the design point that resolves the cross-level reuse problem.
- The renamed `process_temporal` keeps its existing public signature where exposed; only internal call sites change.
- Spatial-first ordering matters even with an empty stub — that's the order we're committing to. Don't let the empty body tempt anyone to flip the order "for now."

---

### Phase 2 — d=0 Connection Learning

**Goal:** During `process_spatial`'s wavefront (implemented in Phase 3), each neuron learns its own d=0 connections in parallel — strengthening edges from itself to the neurons co-active in this spatial phase. Phase 2 implements the per-neuron learning rule; Phase 3 implements the wavefront orchestration that drives it.

#### Parallel-per-neuron model

Spatial processing mirrors temporal: each neuron does its own work in parallel across columns, and the thalamus orchestrates cross-neuron operations at orchestration boundaries between dispatch waves. Per neuron, in parallel, the spatial frame does:

- Match own d=0 routing entries against the co-activation set captured at dispatch time; emit activations for matches.
- Emit own d=0 predictions (the d=0 connection set) for post-wavefront error evaluation.
- Strengthen own d=0 connections to neurons in the co-activation set (this phase).
- Emit index-update events for each connection created/strengthened (consumed by [neuron-reuse](./neuron-reuse.md)'s inference index at the next orchestration boundary).

Neurons in `correction_wired_this_frame` are still strengthened against — they appear as targets in other neurons' d=0 learning — but they do not run their own learning step this frame.

#### Spatial co-activation set

`process_spatial`'s wavefront seeds from L=0 sensory neurons that activated this frame. As the wavefront propagates, more neurons join `fired_this_frame` via routing matches. The co-activation set used for d=0 learning at any wave boundary is **the set of neurons fired during this spatial phase so far**, scoped per [inference-level winner](./inference-level.md).

#### Code touched

- `brain/brain-core/src/neuron.rs`:
  - `connections` field already has slot 0 reserved. Populate it.
  - Add `strengthen_d0(target_id, reward)` analogous to existing `upsert_connection` for d>0.
  - Extend per-neuron spatial-frame processing (added in Phase 3) to call `strengthen_d0` for each co-active target subject to the scoping rule. This work is per-neuron, parallel.
  - `learn_connections` is **not** modified — that's the d>0 path. Spatial d=0 learning is its own code path.
- `brain/brain-core/src/thalamus.rs` — no new "learn" method; learning is embedded in the per-neuron parallel spatial frame.

#### Acceptance

- Unit test: in a synthetic spatial phase with L=0 neurons {A, B, C} fired, after d=0 learning A's d=0 map contains B and C with positive strength; symmetric for B and C.
- Repeating the same frame strengthens existing entries (no duplicates).
- Stocks regression: same accuracy as Phase 1.

#### Notes / gotchas

- Reward propagation: existing d>0 learning tags `conn.reward` from the frame's reward. d=0 follows the same rule.
- Cost: d=0 strengthens up to (N-1)² connections per frame where N is the spatial fired-set size. On MNIST first frame, N=49 → ~2400 strengthens per frame. Profile early.

---

### Phase 3 — Spatial Wavefront Orchestration

**Goal:** Implement the wave-by-wave thalamus orchestration that drives `process_spatial`. Each wave is a parallel-per-neuron dispatch (mirroring temporal's `process_level`); waves continue until no new activations.

#### Per-neuron spatial frame (parallel work)

For each neuron in a wave, in parallel:

- Match own d=0 routing entries against the co-activation set captured at dispatch time; emit activations for matches.
- Emit own d=0 predictions (the d=0 connection set) for post-wavefront error evaluation (Phase 4).
- Strengthen own d=0 connections to neurons in the co-activation set (Phase 2).
- Emit index-update events for each connection created/strengthened.

#### Thalamus orchestration

```
pending = L=0 sensory neurons activated this frame
predicted_d0 = {}
while pending non-empty:
    wave = drain(pending)
    wave.retain(|n| !fired_this_frame.contains(n))
    fired_this_frame.extend(wave)
    results = region.dispatch_process_spatial_frame(wave)  // parallel, per-column
    apply_index_updates(results.index_updates)             // orchestration boundary
    predicted_d0.extend(results.predicted_d0)
    pending.extend(results.activations)
// Phase 4's error evaluation and correction wiring runs here.
```

#### Code touched

- `brain/brain-core/src/brain.rs::process_spatial` — implement the orchestration loop above.
- `brain/brain-core/src/neuron.rs`:
  - Add `process_spatial_frame(co_activation_set) -> SpatialFrameResult` returning emitted activations, predictions, and index-update events.
  - Add `vote_d0()` returning the neuron's d=0 connection set. Separate from `vote(age)` to keep the +1 distance convention clean.
- `brain/brain-core/src/column.rs::process_spatial` — analogous to existing `process_level`; iterates owned-neuron tasks and calls `neuron.process_spatial_frame`.
- `brain/brain-core/src/region.rs::dispatch_process_spatial_frame` — fan out across columns.
- `brain/brain-core/src/thalamus.rs` — own the per-frame `predicted_d0` accumulator. Reset at frame end.

#### Acceptance

- Unit test: pre-wire L=0 sensory neuron S with d=0 routing to existing L=1 correction C; feed S as a wavefront seed; C ends up in `fired_this_frame` and `predicted_d0[C]` contains C's d=0 connections.
- Unit test: spatial wavefront terminates when no new activations. Pre-wire a cycle (A→B, B→A) and verify refractory blocks the second pass.
- Unit test: handoff to temporal — verify the highest-level activations per age from the spatial phase are made available to `process_temporal`'s active-set construction.
- Stocks regression: same accuracy (predictions accumulate but aren't yet used).

---

### Phase 4 — d=0 Error Detection + Correction Minting

**Goal:** At spatial-wavefront stabilization, evaluate every fired neuron's d=0 predictions against the observed co-activation set, generate errors above threshold, mint correction neurons. **No reuse yet — every error mints fresh.** Reuse lands separately (see [neuron-reuse.md](./neuron-reuse.md)).

#### Code touched

- `brain/brain-core/src/thalamus.rs` — add `evaluate_d0_predictions()` called at the end of `process_spatial`.
  - Observed set = neurons fired during this spatial phase, minus `correction_wired_this_frame`, scoped per [inference-level winner](./inference-level.md).
  - For each predicting neuron N (not in `correction_wired_this_frame`):
    - Predicted set = `predicted_d0[N]`
    - Compute mismatch using existing common/missing/novel logic
    - If mismatch > error threshold → record d=0 error
  - For each error: call existing `allocate_pattern_neuron` path adapted for d=0 (connection specs carry distance=0; the new neuron's `connections[0]` is populated from the observed set).
  - Register minted neuron in active memory at `erroring_neuron.activation_level + 1`. Add to `correction_wired_this_frame` and `fired_this_frame`.
- `brain/brain-core/src/neuron.rs::correct_errors` — accept correction entries with d=0 context (routing-table key includes d=0 partial observed set).

#### Order of operations in `processFrame`

1. `process_spatial()`:
   a. Spatial wavefront seeds from L=0 sensory and drains; `predicted_d0` accumulated as neurons fire (Phase 3).
   b. Per-neuron d=0 learning happens in parallel inside each neuron's spatial frame (Phase 2).
   c. `evaluate_d0_predictions()` runs at spatial stabilization. Errors recorded.
   d. Op-4 batch creates correction neurons, registered in active memory at `erroring_neuron.activation_level + 1`. Routing tables updated.
   e. Handoff set computed: highest-level activations per age from the spatial fired set, minus `correction_wired_this_frame`.
2. `process_temporal()` runs (existing pipeline). Its active-set construction incorporates the handoff set.
3. Action voting over `fired_this_frame \ correction_wired_this_frame`.

#### Acceptance

- Unit test: train "digit-1-like" spatial input {A, B, C} across 10 frames → no errors after exposure 2-3. Switch to "digit-7-like" {A, D, E, F, G} → A errors on its d=0 predictions, correction neuron minted with d=0 connections to {A, D, E, F, G}.
- Inspect: correction neuron's routing table is wired into A's routing under d=0 context. Correction neuron is in `correction_wired_this_frame` (verified by checking it does not contribute to action voting this frame).
- Stocks regression: ≤ 5% accuracy delta. d=0 errors may produce new neurons in stocks too; expected, not a regression.

#### Notes / gotchas

- Error threshold mode: use the existing `errorCorrectionMode` for d=0. Parsimony of parameters — no separate `d0ErrorCorrectionMode`. If MNIST shows it needs different tuning, we'll revisit then.
- A correction neuron minted this frame is in `fired_this_frame` (for learning attachment) and in `correction_wired_this_frame` (inhibited from voting and error-checking). It becomes a normal voter starting next frame.
- The `correction_wired_this_frame` inhibition is the load-bearing termination rule (see §3.4 and §7): it prevents reuse-cascade in the reuse workstream and also makes Phase 4's mints behave consistently with future reuses. It also caps within-frame spatial deepening at one fresh layer.

---

### Phase 5 — MNIST Single-Frame Harness

**Goal:** Validate spatial processing on MNIST with the simplified loop.

#### Code touched

- `apps/mnist/test.js` — rewrite training loop:
  ```js
  for each training image:
      const inputs = encoder.encodeImage(image)
      const actions = encoder.encodeAction(label)
      const rewards = encoder.buildRewards(label)
      brain.processFrame(inputs, EMPTY_MAP, EMPTY_MAP)   // single frame
      brain.learn(actions, rewards)                       // wire digit to fired set
  ```
- `apps/mnist/test.js` — rewrite eval loop:
  ```js
  for each test image:
      const inputs = encoder.encodeImage(image)
      brain.processFrame(inputs, EMPTY_MAP, EMPTY_MAP)
      const result = brain.infer()
      prediction = consensus(result.actionVotes)
  ```
- Brain config: temporal context length can be set to 1 (or kept at default — doesn't matter, no temporal connections form on single-frame MNIST).

#### Acceptance

- Train on 1000 images (100 per digit, balanced), eval on 200 held-out → >50% accuracy.
- Inspect neuron counts: should be ≪ what cortex-only produced.
- Wavefront depth: log mean/p99 waves per frame. Should be small (≤10 expected).
- Per-image processing time: should be O(seconds) max, not O(minutes) as in cortex-only.

#### If Phase 5 fails

- **All-or-nothing memorization** (single huge correction neuron per digit): probe the merge threshold / error threshold; check correction neuron size distribution.
- **No correction neurons at all**: error threshold too lax, or d=0 connections strengthen too slowly. Lower threshold, increase initial connection strength.
- **Wavefront not terminating**: shouldn't be possible by §7 reasoning, but if it happens, log queue size per wave to find the source.

---

### Phase 6 — Stocks Integration (Spatial-Only)

**Goal:** Validate the spatial phase on the stocks workload, without reuse. Confirms that spatial processing doesn't regress accuracy and that the spatial fired-set populates as expected when temporal hierarchy is present.

This is distinct from the full stocks+reuse integration in [neuron-reuse.md](./neuron-reuse.md). Running this first isolates spatial's contribution from reuse's contribution.

#### Steps

- Run stocks with `process_spatial` enabled, reuse off. Spatial inputs initially = L=0 sensory; over many frames, the spatial wavefront begins activating correction neurons via routing.
- Measure: directional accuracy, per-episode ROI, neuron count growth, spatial fired-set size distribution, wavefront depth.
- Compare against the [inference-level winner](./inference-level.md) baseline.

#### Acceptance

- Stocks directional accuracy ≥ [inference-level winner](./inference-level.md) baseline (within ±2%).
- Spatial wavefront produces non-trivial corrections within reasonable training time.
- No runaway neuron count growth — corrections form at a sustainable rate.

#### Notes

- If accuracy regresses substantially, the most likely culprits are (a) d=0 error threshold too aggressive for stocks data noise, or (b) spatial corrections fragmenting representations that temporal had previously consolidated. Diagnose by tuning the error threshold first.
- If spatial corrections never form, the d=0 connection strengthen rate may be too slow for stocks' frame cadence; revisit Phase 2's strength scaling.

---

### Phase 7 — Persistence / Backup / Import-Export Updates

**Goal:** Make sure every persistence path round-trips the spatial-processing architecture cleanly.

#### Changes that need to be reflected

- **d=0 connections in `connections[0]`.** The connections vec already reserves slot 0 but it was unused before Phase 2. Serialization needs to round-trip it.
- **No `neuron.level` field.** Phase 1 removed it. Serialized form should not include it.
- **No per-neuron forget rate.** Global static from [mnist-merge](./mnist-merge.md); nothing to serialize per neuron.
- **Reverse inference index** (built in [neuron-reuse.md](./neuron-reuse.md)). Rebuild on load, do not persist.

#### Code touched

- `brain/brain-core/src/backup.rs` — extend `SerializedConnection` / `SerializedNeuron` so distance=0 entries round-trip. Verify the existing format already iterates over the full `connections` vec (slot 0 included) rather than skipping it. Confirm `neuron.level` is not in the serialized form (removed in Phase 1).
- `brain/brain-core/src/thalamus.rs` — on restore, rebuild the inference index from scratch by walking all neurons' connections. Add a `rebuild_inference_index()` method called once at load time (relevant once [neuron-reuse](./neuron-reuse.md)'s index is in place).
- DB import/export apps (anything under `apps/` that imports or exports brain state) — sweep for code that assumes connections start at distance 1 or that reads/writes `neuron.level`. Update accordingly.
- `correction_wired_this_frame` and `fired_this_frame` — **not** persisted. Per-frame state, resets at frame start.

#### Acceptance

- Round-trip test: train a brain on a few MNIST images (enough to mint d=0 corrections across multiple activation levels), snapshot, restore, verify (a) all d=0 connections present with correct strengths, (b) all correction neurons present and routable, (c) one more frame of training on the restored brain produces identical results as on the original.
- DB import/export apps: run a round-trip on a stocks brain with d=0 connections; verify byte-identical export after `import → export`.

#### Notes / gotchas

- Old (pre-Phase-1) backup format is incompatible — `neuron.level` removal alone breaks the layout. Bump format version, reject old backups with a clear error; migration isn't worth the complexity for the current scale of stored brains.
- Phase 7 can land after Phase 6 without blocking it. But before any production deployment that depends on backups, Phase 7 must close.

---

## 7. Risk Assessment

### 7.1 High Confidence

The architecture is correct. Distance-parameterized prediction error (d=0 spatial, d>0 temporal) is mathematically sound and biologically grounded. The temporal-only system already validates the core mechanism on stock data.

**Termination is self-guaranteed for both phases:**

- `process_temporal` is the existing level-sweep, bounded by `max_active_level` as it is today.
- `process_spatial`'s wavefront is bounded by refractory (each neuron fires at most once per spatial phase) plus the `correction_wired_this_frame` inhibition rule (§3.4). The latter is the load-bearing piece: a neuron in `correction_wired_this_frame` does not produce d=0 predictions for evaluation this frame, so it cannot generate fresh errors. Total spatial-phase work is bounded by the spatial hierarchy depth and the size of the input set, not by error-chain length.

No safety-valve wave-count limit needed.

### 7.2 Moderate Confidence

MNIST works within the timeline. The bootstrap dynamics — how many exposures are needed before d=0 connections form useful clusters — are unknown. Wavefront stabilization on high-dimensional spatial data is untested.

### 7.3 Key Risks

- **Bootstrap noise**: Early training may produce many correction neurons before d=0 connections have stabilized. Mitigation: the error threshold absorbs noise-level mismatches; reuse (see [neuron-reuse.md](./neuron-reuse.md)) prevents redundant minting even when errors are abundant.
- **Generalization failure**: d=0 correction neurons may over-memorize specific pixel configurations rather than generalizing to stroke-like features. Mitigation: reuse provides the cross-instance signal; decay sharpens it.
- **Performance**: per-neuron d=0 strengthening can be expensive at high spatial fired-set sizes. Profile early; optimize if it matters.

---

## 8. Open Items

1. **Bootstrap dynamics on MNIST**: how many exposures before d=0 stabilizes into useful clusters? Empirical only.
2. **Spatial wavefront depth on stocks**: how deep does the spatial hierarchy go in practice when temporal patterns feed it? Measure in Phase 6.
3. **Action binding through reuse generalization**: when a correction neuron is heavily reused, action votes may need normalization. See [neuron-reuse.md](./neuron-reuse.md).
