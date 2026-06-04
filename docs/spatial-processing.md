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
- `temporal_level` — depth in the temporal hierarchy. Apex spatial activations (sensory and spatial corrections that weren't subsumed by a higher spatial activation this frame) are inserted at `temporal_level = 0` regardless of their spatial level; +1 per d>0 routing hop during `process_temporal`.

A single neuron can hold both — e.g., a spatial correction at `spatial_level = 7` that's apex will *also* sit at `temporal_level = 0`. The two indexes mean different things and don't conflict.

**Why two dimensions, not one.** Apex spatial activations are at heterogeneous spatial depths — a cube might be apex at `spatial_level = 3`, a human at `spatial_level = 7`. From temporal's perspective both are "inputs," not depth-3 vs depth-7 things. Forcing a single level field would either misrepresent the cube as a "shallower input" than the human (it isn't — both are equal starting points for temporal sequencing) or require resetting the cube's spatial depth to 0 (losing the information that it's a depth-3 spatial construct). Two dimensions keep both facts intact: a neuron's place in the spatial hierarchy AND its starting point in the temporal hierarchy.

**Per-phase iteration.** Both phases iterate `for level in 0..max_active_level`, at each level fire whichever neurons are *activated at that level in this phase's index*, let their routing matches activate neurons at the next level, repeat until no new activations. Spatial drives off `spatial_level_index`; temporal drives off `temporal_level_index`. The existing `process_levels()` pipeline is renamed `process_temporal` and gains a filter to skip d=0 connections; `process_spatial` is the same loop pointed at d=0, reading from and writing to its own index.

**The handoff is explicit.** Between the two phases, the thalamus computes the apex set — neurons in the spatial fired set whose d=0 routing did *not* fire a higher-level spatial activation this frame — and inserts each into `temporal_level_index[0]`. Spatial corrections that did subsume into a higher spatial neuron are not handed off; their role is already filled by the higher-level correction that absorbed them. Sensory neurons that didn't contribute to any spatial correction this frame stay apex and feed temporal directly.

**Per-neuron, per-level order of operations** (lifted from the existing temporal path, [neuron.rs:708](brain/brain-core/src/neuron.rs:708) — spatial follows the same sequence):

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

When a neuron fires during `process_spatial`, its d=0 connections constitute a predicted co-activation set. The thalamus compares this predicted set against the observed reality (the set of neurons fired in the same spatial phase). Missing predictions (expected neurons that didn't fire) and novel observations (unexpected neurons that did fire) both contribute to the mismatch rate. If the mismatch exceeds the error threshold, an error is generated.

This is the same error detection logic temporal uses for d>0 — the only change is the distance the predictions are read at.

### 4.4 Error Correction

The thalamus mints a correction neuron capturing the actual co-activation context. The new neuron's `spatial_level` is `erroring_neuron.spatial_level + 1` (its temporal_level is not set yet — that happens at the next frame's apex handoff if the correction fires and isn't itself subsumed). The erroring neuron's routing table is updated so that the same context routes directly to the new correction next time. This is the same correction logic temporal uses for d>0; only the connection distance and the level dimension differ.

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

---

## 6. Implementation Plan

### Overview

| Phase | Goal | Validation gate |
|---|---|---|
| 1 | `process_spatial` as a copy of `process_temporal`, restricted to d=0; `process_temporal` filtered to d>0; spatial runs first per frame | Stocks regression: behavior diverges only via d=0 work |
| 2 | MNIST validation run (existing harness) | Accuracy and confusion matrix improve over the sensory-only baseline |
| 3 | Stocks integration | Directional accuracy ≥ current baseline |
| 4 | Persistence / backup / import-export updates | Snapshot/restore round-trips d=0 connections |

---

### Phase 1 — Split `process_levels` into Spatial and Temporal

**Goal:** Stand up the two-phase pipeline with two level indexes and the apex handoff. Spatial and temporal share the same per-neuron parallel work and the same error-detection/correction; they differ in which connection slot they read (d=0 vs d>0) and which level index they iterate (`spatial_level_index` vs `temporal_level_index`).

#### Code touched

- `brain/brain-core/src/memory.rs`:
  - Split the existing `level_index` ([memory.rs:48](brain/brain-core/src/memory.rs:48)) into `spatial_level_index` and `temporal_level_index`. Both have the same shape as today's `level_index` (`FxHashMap<Level, FxHashMap<FrameNumber, FxHashSet<NeuronId>>>`).
  - `get_level_neurons` and `get_level_ages` take an additional phase parameter (`Spatial` | `Temporal`) and read the corresponding index.
  - When the spatial sweep activates a neuron, it writes into `spatial_level_index`. When the temporal sweep activates a neuron, it writes into `temporal_level_index`. Apex handoff (below) writes into `temporal_level_index[0]`.
- `brain/brain-core/src/brain.rs`:
  - Rename `process_levels()` → `process_temporal()`. Filter pattern-matching / error-detection to `connections[d>0]`. Drive iteration off `temporal_level_index`.
  - Add `process_spatial()` — same level-sweep body, restricted to `connections[0]`, driving iteration off `spatial_level_index`.
  - Add `compute_apex_and_handoff()` between the two phases. Walk the spatial fired set; for each neuron N, if N's d=0 routing did not fire any other neuron this spatial phase, mark N as apex. Insert all apex neurons into `temporal_level_index[0]`. (Subsumption is tracked incrementally during the spatial sweep — when a routing match fires neuron M from activator N, mark N as subsumed.)
  - Per-frame pipeline: `process_spatial()` → `compute_apex_and_handoff()` → `process_temporal()`.
- `brain/brain-core/src/neuron.rs`, `column.rs`, `region.rs`, `thalamus.rs` — wherever the level-sweep reads or matches connections, parameterize on distance and on which level index to read from. No new orchestration is needed beyond the apex handoff.
- `brain/brain-core/src/neuron.rs::get_pattern_candidates_at_age` ([line 919](brain/brain-core/src/neuron.rs:919)) — today this hard-rejects `pattern_distance < 1`. That guard needs to relax to `< 0` (or be parameterized on the phase) so spatial can match at d=0. This is the one place the existing code is explicitly d>0; everything else (connection storage already reserves slot 0, routing-table context is just a distance-keyed map) is general.
- Sensory neuron activation at frame start: insert into both `spatial_level_index[0]` *and* (after apex computation runs and confirms no spatial subsumption) `temporal_level_index[0]`. On a fresh brain with no spatial routing, every sensory neuron is apex by default and ends up at `temporal_level_index[0]`, matching today's behavior.

#### Notes / gotchas

- Voting (action voting and the existing higher-pattern-inhibition rule) reads the combined fired set across both phases, as today. The two-index split is about iteration, not about voting.

---

### Phase 2 — MNIST Validation Run

**Goal:** Run the existing MNIST harness ([apps/mnist/jobs/test.js](apps/mnist/jobs/test.js)) and see what spatial processing does to the accuracy and confusion matrix. No code changes expected — the harness already does the single-frame processFrame → learn loop and the held-out eval. Phases 1–2 plug spatial into the brain; this phase observes the effect.

#### Steps

- Run the existing balanced binary-7×7 default (`node apps/mnist/jobs/test.js`) as the baseline-after-spatial.
- Compare accuracy and confusion matrix against the pre-spatial numbers. The load-bearing artifact is whether the 3/8/9 collapses called out in the harness's `showResults` comment clear up — that's what motivated this workstream.
- Optionally scale: `--image-size 28 --buckets 2` and `--per-class N` to probe larger inputs once the binary-7 run is healthy.

#### Acceptance

- Test accuracy meaningfully above the sensory-only baseline (which already gets ~76% on 28×28 binary balanced).
- Confusion matrix shows the 3/8/9 group differentiating, not collapsing onto one cell.
- Per-image processing time stays in the same order as today — spatial shouldn't multiply per-frame cost by more than a small constant.

#### If results are bad

- **All-or-nothing memorization** (single huge correction neuron per digit): probe the merge threshold / error threshold; check correction neuron size distribution.
- **No correction neurons at all**: error threshold too lax, or d=0 connections strengthen too slowly. Lower threshold, increase initial connection strength.
- **Level-sweep not terminating**: log queue size per level to find the source — likely a routing-cycle bug.

---

### Phase 3 — Stocks Integration

**Goal:** Validate the spatial phase on the stocks workload.

#### Steps

- Run stocks with `process_spatial` enabled. Spatial inputs initially = L=0 sensory; over many frames, the spatial sweep begins activating correction neurons via routing.
- Measure: directional accuracy, per-episode ROI, neuron count growth.
- Compare against the current cortex-only stocks baseline.

#### Acceptance

- Stocks directional accuracy ≥ current baseline (within ±2%).
- Spatial sweep produces non-trivial corrections within reasonable training time.
- No runaway neuron count growth.

#### Notes

- If accuracy regresses substantially, the most likely culprits are (a) d=0 error threshold too aggressive for stocks data noise, or (b) spatial corrections fragmenting representations that temporal had previously consolidated. Diagnose by tuning the error threshold first.

---

### Phase 4 — Persistence / Backup / Import-Export Updates

**Goal:** Make sure every persistence path round-trips d=0 connections and the new spatial_level field.

#### Code touched

- `brain/brain-core/src/backup.rs` — extend `SerializedConnection` / `SerializedNeuron`:
  - `connections[0]` distance=0 entries must round-trip. Verify the existing format iterates the full `connections` vec rather than skipping slot 0.
  - The existing `neuron.level` field becomes the temporal intrinsic level. Add `neuron.spatial_level` alongside it. For pre-spatial-era neurons loaded from older snapshots, default `spatial_level` to 0 (they sit at the base of the spatial hierarchy because no spatial grouping built them).
  - The per-frame `spatial_level_index` / `temporal_level_index` are not persisted — same rule as today's `level_index`.
- DB import/export apps under `apps/` — sweep for code that assumes connections start at distance 1, and for code that reads/writes the single `neuron.level` field.

#### Acceptance

- Round-trip test: train a brain on a few MNIST images (enough to mint d=0 corrections across multiple spatial levels), snapshot, restore, verify (a) all d=0 connections present with correct strengths, (b) all correction neurons present and routable with their spatial_level intact, (c) one more frame of training on the restored brain produces identical results as on the original.
- DB import/export apps: run a round-trip on a stocks brain with d=0 connections; verify byte-identical export after `import → export`.
- Backward-compat test: load a pre-spatial stocks snapshot, verify all neurons get `spatial_level = 0`, run one frame, confirm output matches what the pre-spatial code would have produced (modulo the new d=0 work, which will start forming immediately).

---

## 7. Risk Assessment

### 7.1 High Confidence

The architecture is correct. Distance-parameterized prediction error (d=0 spatial, d>0 temporal) is mathematically sound and biologically grounded. The temporal-only system already validates the core mechanism on stock data; spatial is the same code path with the distance filter flipped, reading from its own level index.

Termination is self-guaranteed for both phases: each is bounded by the depth of its own level index. `process_spatial` terminates when no new d=0 routing matches activate further spatial neurons; `process_temporal` terminates by the same rule on `temporal_level_index`. The apex handoff between them is O(spatial fired set) and runs once.

### 7.2 Moderate Confidence

MNIST works within the timeline. The bootstrap dynamics — how many exposures are needed before d=0 connections form useful clusters — are unknown. Level-sweep stabilization on high-dimensional spatial data is untested.

### 7.3 Key Risks

- **Bootstrap noise**: Early training may produce many correction neurons before d=0 connections have stabilized. Mitigation: the error threshold absorbs noise-level mismatches.
- **Generalization failure**: d=0 correction neurons may over-memorize specific pixel configurations rather than generalizing to stroke-like features. Expected outcome at this stage; spatial alone gets the representational compression of co-activation grouping.
- **Performance**: per-neuron d=0 strengthening can be expensive at high spatial fired-set sizes. Profile early; optimize if it matters.

---

## 8. Open Items

1. **Bootstrap dynamics on MNIST**: how many exposures before d=0 stabilizes into useful clusters? Empirical only.
2. **Spatial sweep depth on stocks**: how deep does the spatial hierarchy go in practice when temporal patterns feed it? Measure in Phase 3.
3. **Apex subsumption granularity**: a neuron N is "subsumed" if its d=0 routing fires *any* higher-level spatial neuron this frame. Is that the right rule? Alternatives: (a) subsumed only if *all* of N's strong d=0 partners co-activated into the same higher neuron; (b) subsumed only if the higher neuron is itself apex (transitive). Phase 1 ships rule (current) — revisit if MNIST shows the apex set is too small (over-subsumption: the high-level neuron eats everything below it) or too big (under-subsumption: lower-level fragments still feed temporal alongside the higher abstraction).
