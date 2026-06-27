# Spatial Processing

---

## 1. Motivation

### 1.1 The MNIST Experiment

The Robot Brain was tested on MNIST data using the following setup: 784 pixel channels (reduced to 49 for compute constraints) fed as independent input channels to the brain, with repeated frames shown until cortex stabilization, followed by action implantation for digit labels. The temporal-only approach produced correct but catastrophically inefficient results — requiring impractical computation time to build a useful representational tree.

### 1.2 The Core Insight

The cortex's pattern recognition mechanism is designed for temporal sequences — it detects "A then B" across time distances. MNIST digits have no temporal structure. Feeding the same frame repeatedly and asking the cortex to build hierarchy through temporal repetition forces it to represent spatial co-activation as a flat combinatorial expansion. This is equivalent to writing a logical expression without parentheses — the expression is valid but combinatorially intractable.

### 1.3 The Role of Spatial Processing (d=0)

Spatial connections (d=0) provide the "parentheses" — storage for co-activation patterns that compress the representational space. They answer the question "what belongs together?" so that temporal connections (d>0) can efficiently answer "what follows what?" Without spatial grouping, the cortex must pattern-match over the full combinatorial space of raw inputs. With it, the cortex operates over a compressed vocabulary of pre-grouped units.

This is not an MNIST-specific optimization. It is core Brain functionality required for any domain where spatial or simultaneous structure exists — vision, point-in-time market states, sensor arrays, and any multi-channel sensory input.

### 1.4 Same Mechanism, Different Distance

Spatial processing is not a separate system bolted onto the cortex. It is the same mechanism — prediction, error detection, correction — applied at distance zero. A neuron predicting what comes next at d=3 and a neuron predicting what's active alongside it at d=0 use identical infrastructure: connections with strength, routing tables, thalamus orchestration, error-driven creation. The only difference is the distance parameter.

---

## 2. Biological Parallels

### 2.1 Temporal Patterns as Pyramidal Cells

Pyramidal cells are the canonical sequence detectors.
They receive context, learning temporal patterns and fire matching neurons.
Temporal patterns in Robot Brain map directly to this: they detect "A then B" with contexts, using routing tables with time-distance connections.

### 2.2 Spatial Patterns as Grid Cells

Within-frame grouping in the brain happens through co-activations.
V1 lateral inhibition groups edges, V2 groups edges into shapes, V4 groups shapes into objects.
It takes a single visual frame to propagate up the levels.
Spatial corrections in this design work the same way as temporal corrections, the only difference is d=0.

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
| **Forgetting** | Same | Same |

### 3.2 The Neuron's Perspective

Every neuron maintains connections indexed by distance. Connections at d>0 are temporal predictions ("neuron B will be active d frames from now"). Connections at d=0 are co-activation predictions ("neuron B should be active right now alongside me").

The neuron does not distinguish between temporal and spatial connections — they are entries in the same connection table.

### 3.3 Two-Phase Processing Model

Each frame runs the same processing twice in sequence: spatial first, temporal second. Spatial and temporal are not different mechanisms — they are the same level-sweep, the same per-neuron parallel work, the same prediction/error/correction loop. The only difference is which connection slot drives matching and which frame the resulting votes apply to: spatial reads `connections[0]` and votes for the current frame; temporal reads `connections[d>0]` and votes for frames d ahead. Mechanically identical; semantically one is "what's here now alongside me" and the other is "what comes next."

The motivation for running spatial first is the feedforward sensory cascade. Object identification must precede motion tracking — you cannot have a d>0 prediction "ball moves left" until "ball" is a recognized entity in both frames. If temporal ran first, its d>0 predictions could only operate on raw sensory neurons (the only thing active at age=0 within the frame), which is the combinatorial-explosion bottleneck the design was meant to fix. By running spatial first, temporal sees the spatial hierarchy's apex as its input — objects and groupings, not raw pixels — and its d>0 predictions form between meaningful entities across frames.

**Two level dimensions in active memory.** This is the principal new architectural piece. Each activation carries *two* levels, not one:

- `spatial_level` — depth in the spatial hierarchy. Sensory = 0; +1 per d=0 routing hop during `process_spatial`.
- `temporal_level` — depth in the temporal hierarchy. Apex spatial activations (sensory and spatial corrections that weren't subsumed by a higher spatial activation this frame) are inserted at `temporal_level = 0` regardless of their spatial level; +1 per d>0 routing hop during `process_temporal`. Because every spatial correction inherits its parent's coordinate (§4.4), all of these are coordinate-bearing tokens — so temporal level 0 is a uniform (channel, dimension, value) vocabulary whether a token is a raw sensory neuron or a context-refined correction, and the per-dimension temporal consensus needs no special case for spatial patterns.

A single neuron can hold both — e.g., a spatial correction at `spatial_level = 7` that's apex will *also* sit at `temporal_level = 0`. The two indexes mean different things and don't conflict.

**Why two dimensions, not one.** Apex spatial activations are at heterogeneous spatial depths — a cube might be apex at `spatial_level = 3`, a human at `spatial_level = 7`. From temporal's perspective both are "inputs," not depth-3 vs depth-7 things. Forcing a single level field would either misrepresent the cube as a "shallower input" than the human (it isn't — both are equal starting points for temporal sequencing) or require resetting the cube's spatial depth to 0 (losing the information that it's a depth-3 spatial construct). Two dimensions keep both facts intact: a neuron's place in the spatial hierarchy AND its starting point in the temporal hierarchy.

**Per-phase iteration.** Both phases iterate `for level in 0..max_active_level`, at each level fire whichever neurons are *activated at that level in this phase's index*, let their routing matches activate neurons at the next level, repeat until no new activations. Spatial drives off `spatial_level_index`; temporal drives off `temporal_level_index`. The existing `process_levels()` pipeline is renamed `process_temporal` and gains a filter to skip d=0 connections; `process_spatial` is the same loop pointed at d=0, reading from and writing to its own index.

**The handoff is explicit.** Between the two phases, the thalamus computes the apex set — neurons in the spatial fired set whose d=0 routing did *not* fire a higher-level spatial activation this frame — and inserts each into `temporal_level_index[0]`. Spatial corrections that did subsume into a higher spatial neuron are not handed off; their role is already filled by the higher-level correction that absorbed them. Sensory neurons that didn't contribute to any spatial correction this frame stay apex and feed temporal directly.

**Per-neuron, per-level order of operations** (lifted from the existing temporal path, [neuron.rs](../brain/brain-core/src/neuron.rs) — spatial follows the same sequence):

1. **Record error feedback** — fold prior-frame error rates into per-age accuracy stats (skipped in non-learning mode).
2. **Learn connections** — strengthen connections across active ages from this neuron (skipped for newly-minted error patterns; for spatial this is the d=0 learning step).
3. **Recognize patterns** — match stored contexts against `level_context`; refine on hit.
4. **Correct errors** — install pre-created error-correction patterns as children, emit their contextRef adds.
5. **Generate votes** — cast predictions for non-suppressed ages.

The thalamus orchestrates per-level dispatches around this; the per-neuron step is the loop body in both phases.

```
Frame arrives
    │
    ▼
Activate sensory neurons
    spatial_level_index[0]  ← sensory
    │
    ▼
process_spatial  (level-sweep over spatial_level_index, votes for current frame)
    │
    │   For level 0..max in spatial_level_index:
    │     • d=0 pattern recognition (routing matches on connections[0]
    │       activate neurons into spatial_level_index[level+1]).
    │     • d=0 error detection and correction.
    │     • Track which firing neurons subsumed into higher-level
    │       spatial activations (i.e., whose d=0 routing match fired
    │       a higher-level neuron this frame).
    │
    ▼
Apex handoff  (between phases)
    │
    │   apex = spatial fired set \ subsumed set
    │   for each neuron N in apex:
    │       temporal_level_index[0].insert(N)
    │
    ▼
process_temporal  (level-sweep over temporal_level_index, votes for future frames)
    │
    │   For level 0..max in temporal_level_index:
    │     • d>0 pattern recognition (routing matches on connections[d>0]
    │       activate neurons into temporal_level_index[level+1]).
    │     • d>0 error detection and correction.
    │
    ▼
Action voting (as-is — no changes)
    │
    ▼
Next frame
```

**Spatial→temporal handoff.** Explicit, not implicit. The two phases read separate level indexes in active memory — `spatial_level_index` and `temporal_level_index` — and the handoff is the apex computation that bridges them. Without it, temporal would start from an empty `temporal_level_index` and have nothing to process.

The two phases can't step on each other for two reasons: they read disjoint connection slots (spatial = `connections[0]`, temporal = `connections[d>0]`), and they iterate disjoint indexes. The same neuron can sit at very different levels in the two indexes — that's the point.

For the first frame on a fresh brain, no spatial corrections exist; the apex set is just the sensory neurons, and temporal behaves exactly as it does today. As spatial corrections accumulate, apex shifts from sensory pixels to spatial groupings, and temporal starts its sweep over those richer inputs.

---

## 4. Co-Activation Prediction and Error

### 4.1 Connection Building

Co-activation connections (d=0) are built through observation, identical to temporal connections. When neurons A and B are co-active in a frame's spatial phase, both strengthen their d=0 connections to each other. Over repeated exposures, strongly co-activating neurons develop strong mutual d=0 connections.

### 4.2 Co-Activation as Prediction

A neuron's d=0 connections constitute predictions: "when I fire, I expect these neurons to also be firing." This is the spatial analog of temporal prediction ("when I fire, I expect neuron B at distance d").

### 4.3 Error Detection

When a neuron fires during `process_spatial`, its d=0 connections constitute a predicted co-activation set. The thalamus compares this predicted set against the observed reality (the set of neurons fired in the same spatial phase) with the Jaccard-union error: missing predictions (expected neurons that didn't fire) and novel observations (unexpected neurons that did fire) both count toward the mismatch rate. If the mismatch exceeds the unit's correction threshold (`1 − groupThreshold`, adapted by `groupMode`), an error is generated.

This is the same error detection logic temporal uses for d>0 — the only change is the distance the predictions are read at.

### 4.4 Error Correction

The thalamus mints a correction neuron capturing the actual co-activation context. The new neuron's `spatial_level` is `erroring_neuron.spatial_level + 1` (its temporal_level is not set yet — that happens at the next frame's apex handoff if the correction fires and isn't itself subsumed). **The correction inherits the erroring (parent) neuron's full (channel, dimension, coordinate).** A correction is a refinement of its parent — "pixel A, but in this specific neighborhood configuration" — so what it asserts about the world is still A's value; the refined identity lives in the neuron id and routing context, not in the coordinate. Inheriting the coordinate does two things: it gives every level above L0 the parent's neighbor graph (so the correction's own d=0 learning and error evaluation stay filtered to the parent's neighborhood, and an L2 minted from an L1 anchored at position A takes as context only L1s anchored at A's neighbors — receptive fields grow one radius hop per level), and it keeps every apex token coordinate-bearing, so temporal level 0 stays a uniform interface (see §3.3, §5.1). The inherited coordinate is **not** registered in `neurons_by_value` — that map requires coordinate uniqueness, and value→neuron resolution must always land on the L0 sensory/action neuron; refined tokens are reached only via routing matches. The erroring neuron's routing table is updated so that the same context routes directly to the new correction next time. This is the same correction logic temporal uses for d>0; only the connection distance and the level dimension differ. **(Superseded by the wave-front redesign.** The [neuron-reuse plan](./neuron-reuse.md) keeps this structure — spatial processing → apex handoff → temporal processing — but turns each stage into a settling **wave**, removes stored levels, and makes **all** corrections coordinate-less: coordinate inheritance here is replaced by **footprints** (the set of base sensory neurons a correction covers) as the neighborhood primitive. Only base sensory/action neurons keep coordinates. This section describes the current implemented model; the wave-front is the planned successor.)

### 4.5 Bootstrap Dynamics

1. **First exposure**: Neurons A, B, C co-activate. No d=0 connections exist. No predictions, no errors. Thalamus records co-activations, builds d=0 connection entries: A↔B, A↔C, B↔C.
2. **Repeated exposures**: Same neurons co-activate. d=0 connections strengthen. Each neuron now predicts its co-activation partners. Predictions match reality. No error. No correction neurons needed.
3. **Conflicting exposure**: A new input activates A, D, E, F, G. Neuron A predicted B and C (its established d=0 partners), but sees D, E, F, G instead. Error exceeds threshold. Thalamus mints correction neuron C1 capturing the actual co-activation set. Updates A's routing table: in context of D, E, F, G, activate C1.
4. **Subsequent exposures**: When A, B, C recur — A predicts B and C correctly, no error, default wiring handles it. When A, D, E, F, G recur — A recognizes context, activates C1 directly.

### 4.6 Two Modes of Representation

- **First-learned patterns**: Represented in default d=0 connections of raw neurons. No correction neuron needed. Actions bind directly to the sensory neurons.
- **Subsequent conflicting patterns**: Represented by correction neurons created from prediction errors. Actions bind to the correction neurons.

---

## 5. Hierarchical Spatial Grouping

### 5.1 How Hierarchy Emerges

Spatial hierarchy emerges along two axes — within a single spatial phase (via the level-sweep and existing routing) and across frames (via the routing tables accumulating new entries over time).

**Within a spatial phase.** When `process_spatial` runs in frame N:

- `spatial_level_index[0]`: sensory fires. d=0 routing matches activate `spatial_level_index[1]` corrections (those minted in prior frames).
- `spatial_level_index[1]`: those corrections fire. d=0 routing matches activate `spatial_level_index[2]` corrections.
- Level K: continues until no new routing matches.
- At stabilization: d=0 predictions evaluated. For each error → mint correction at `erroring_neuron.spatial_level + 1`.

A single frame can therefore deepen the spatial hierarchy by at most one fresh layer at the top, plus however deep the existing routing structure already reaches.

**Across frames.** Frame N's fresh mints become routing-table entries in their source neurons. In frame N+1's spatial phase, those entries match and fire the corresponding correction neurons mid-sweep — so the sweep naturally reaches one level deeper than it did in frame N. Over many frames, the spatial hierarchy reaches arbitrary depth, even though any single frame only adds one fresh layer.

**Apex handoff to temporal.** At the end of the spatial phase, the apex set is computed: neurons that fired in spatial but did not have their d=0 routing fire any higher-level spatial neuron this frame. Each apex neuron is inserted into `temporal_level_index[0]` — regardless of its `spatial_level`. So a human-object at spatial_level 7 and a cube-object at spatial_level 3 both feed temporal at temporal_level 0, as equal inputs. Temporal then does its own level-sweep over `temporal_level_index`, learning d>0 sequences over spatial groupings rather than over raw pixels.

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

1. Show one frame. Pixel neurons activate into `spatial_level_index[0]`.
2. `process_spatial` runs. Level-sweep seeds from the sensory pixels. On first exposure, no corrections exist yet — sweep stops at level 0, mints corrections from d=0 errors. On subsequent exposures, routing matches activate previously-minted corrections, the sweep deepens, and possibly one new layer mints at the top.
3. Apex handoff: every spatial-fired neuron not subsumed by a higher-level spatial activation this frame is inserted into `temporal_level_index[0]`. On first exposure with no corrections, this is just the sensory pixels.
4. `process_temporal` runs over `temporal_level_index`. For single-frame MNIST there's no cross-frame context, so temporal does no useful work, but it runs uniformly for architectural consistency.
5. Action voting runs as it does today.
6. `learn()` wires the digit action to the active voting set.

One frame per image. Training loop: show image → spatial + temporal phases run → learn digit → next image.

Inference is symmetric: on a test image, the spatial sweep activates pre-trained correction neurons via routing matches (not as fresh wirings), and those activated neurons vote for the digit action.