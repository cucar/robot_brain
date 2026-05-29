# Robot Brain: Spatial Processing & Neuron Reuse Design Document

**Date:** May 28, 2026  
**Author:** Cagdas Ucar  
**Status:** Draft — Pre-Implementation  

---

## 1. Motivation

### 1.1 The MNIST Experiment

The Robot Brain was tested on MNIST data using the following setup: 784 pixel channels (reduced to 49 for compute constraints) fed as independent input channels to the brain, with repeated frames shown until cortex stabilization, followed by action implantation for digit labels. The cortex-only approach produced correct but catastrophically inefficient results — requiring impractical computation time to build a useful representational tree.

### 1.2 The Core Insight

The cortex's pattern recognition mechanism is designed for temporal sequences — it detects "A then B" across time distances. MNIST digits have no temporal structure. Feeding the same frame repeatedly and asking the cortex to build hierarchy through temporal repetition forces it to represent spatial co-activation as a flat combinatorial expansion. This is equivalent to writing a logical expression without parentheses — the expression is valid but combinatorially intractable.

### 1.3 The Role of Spatial Processing (d=0)

Spatial connections (d=0) provide the "parentheses" — temporary storage for co-activation patterns that compress the representational space. They answer the question "what belongs together?" so that temporal connections (d>0) can efficiently answer "what follows what?" Without spatial grouping, the cortex must pattern-match over the full combinatorial space of raw inputs. With it, the cortex operates over a compressed vocabulary of pre-grouped units.

This is not an MNIST-specific optimization. It is core Brain functionality required for any domain where spatial or simultaneous structure exists — vision, point-in-time market states, sensor arrays, and any multi-channel sensory input.

### 1.4 The Unification Insight

Spatial processing is not a separate system bolted onto the cortex. It is the same mechanism — prediction, error detection, correction — applied at distance zero. A neuron predicting what comes next at d=3 and a neuron predicting what's active alongside it at d=0 use identical infrastructure: connections with strength, routing tables, thalamus orchestration, error-driven creation. The only difference is the distance parameter.

This insight eliminates the need for separate "moment neuron" and "pattern neuron" types. A neuron is a neuron. It has connections at various distances including zero. It fires when its predictions match. The distance determines whether the match is temporal or spatial.

---

## 2. Biological Parallels

### 2.1 Pattern Neurons as Pyramidal Cells

Cortical pyramidal cells are the canonical sequence detectors. They receive top-down context via apical dendrites and feedforward input via basal dendrites, learning temporal predictions. Pattern neurons in Robot Brain map directly to this: they detect "A then B" with context sensitivity, using routing tables with time-distance connections.

### 2.2 Moment Neurons as Grid Cells

Grid cells in entorhinal cortex fire in periodic spatial patterns, encoding position through co-activation. A set of grid cells firing together represents "where you are" in some abstract space — not through sequence, but through simultaneous activation. d=0 neurons serve the same role: they capture "what's active together right now" and mint a reusable reference to that state.

Grid cells sit in entorhinal cortex — the gateway to hippocampus — compressing high-dimensional cortical output into a spatial/relational code. Grid cells remap across environments, generalizing across contexts — analogous to moment neurons losing incidental connections and becoming class neurons that represent abstract categories rather than specific instances.

### 2.3 The Two Complementary Operations

- **Pyramidal cells / d>0 connections**: learn *when* (temporal sequences)
- **Grid cells / d=0 connections**: learn *what together* (spatial/relational co-activation)

The brain requires both operations to build efficient representations. Biologically, these are not separate processing phases — pyramidal cells and grid cells process simultaneously, feeding into each other within the same cycle.

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
| **Forgetting** | Standard decay | Same decay; repeated partial matches → class neuron generalization |

### 3.2 The Neuron's Perspective

Every neuron maintains connections indexed by distance. Connections at d>0 are temporal predictions ("neuron B will be active d frames from now"). Connections at d=0 are co-activation predictions ("neuron B should be active right now alongside me").

During frame processing, each neuron uses ALL its connections as predictions — d=0 and d>0 alike. Errors at any distance trigger the same correction mechanism. The neuron does not distinguish between temporal and spatial connections — they are entries in the same connection table.

### 3.3 Thalamus Role

The thalamus orchestration is identical for d=0 and d>0:

1. **Collect predictions**: Each active neuron has connections at various distances, including d=0.
2. **Compare against reality**: The thalamus compares predicted activations against actual activations at each distance.
3. **Detect errors**: If the mismatch exceeds the configured error threshold, that neuron is in error at that distance.
4. **Create correction neurons**: For errors, the thalamus creates a new neuron capturing the actual activation context — either the actual co-activation set (d=0) or the actual temporal sequence (d>0).
5. **Update routing tables**: The erroring neuron's routing table is updated so next time it recognizes the correct context and activates the correction neuron directly.

### 3.4 Wavefront Processing Model

The current level-sweep processing (`for level in 0..max_level`) is replaced by a wavefront propagation model. This is required for two reasons: (a) d=0 connections should activate neurons within the same frame without waiting for the next level pass, and (b) neuron reuse (Phase 2) means a neuron at any "level" can be activated from any other, making fixed level ordering meaningless.

The `level_index` in Memory is retained — context and co-activation are still scoped per level (a neuron's d=0 predictions are evaluated against the union of fired neurons at the same level during this frame's wavefront). What goes away is `neuron.level` as an intrinsic routing/dispatch property and the outer `for level in 0..max_level` loop. Levels remain a spatial organization of memory; they cease to be a temporal dispatch axis.

**Prediction evaluation timing.** d=0 predictions are evaluated against the union of all neurons fired during this frame's wavefront, **after stabilization, before action voting**. The validation set is built within the frame — there is no need to wait for the next frame. This is the critical distinction from d>0: temporal predictions are validated when their target frame arrives; spatial predictions are validated at end-of-wavefront in the same frame.

```
Frame arrives
    │
    ▼
Activate sensory neurons (age=0)
    │
    ▼
Wavefront Processing
    │
    ├─ Wave 0:
    │    Sensory neurons fire
    │    d=0 connections: activate co-activation neurons (spatial grouping)
    │    d>0 connections: learn/predict temporal sequences
    │    Errors at any distance → thalamus creates correction neurons
    │    Correction neurons added to next wave
    │
    ├─ Wave 1:
    │    Newly activated neurons (from d=0 or d>0 matches) fire
    │    Their d=0 and d>0 connections processed
    │    More corrections possible → next wave
    │
    ├─ Wave N:
    │    Continues until no new activations
    │    Refractory constraint: each neuron fires at most once per frame
    │
    ▼
Stabilization: no new activations
    │
    ▼
Action voting (all active neurons participate)
    │
    ▼
Next frame
```

### 3.5 Refractory Period

Each neuron can fire at most once per frame. This prevents infinite loops from circular activation chains. A `fired_this_frame: FxHashSet<NeuronId>` tracks which neurons have already fired. If an activation signal reaches a neuron that has already fired, it is ignored. This mirrors biological refractory periods and ensures wavefront termination.

---

## 4. Co-Activation Prediction and Error

### 4.1 Connection Building

Co-activation connections (d=0) are built through observation, identical to temporal connections. When neurons A and B are co-active in a frame, both strengthen their d=0 connections to each other. Over repeated exposures, strongly co-activating neurons develop strong mutual d=0 connections.

### 4.2 Co-Activation as Prediction

A neuron's d=0 connections constitute predictions: "when I fire, I expect these neurons to also be firing." This is the spatial analog of temporal prediction ("when I fire, I expect neuron B at distance d").

### 4.3 Error Detection

When a neuron fires, its d=0 connections constitute a predicted co-activation set. After wavefront stabilization, the thalamus compares this predicted set against **reality = the union of all neurons that fired anywhere in this frame's wavefront at the same level**. Missing predictions (expected neurons that didn't fire) and novel observations (unexpected neurons that did fire) both contribute to the mismatch rate. The intersection of predicted and observed can be empty — predictions are not constrained to be a subset of reality. If the mismatch exceeds the error threshold (same threshold mechanism as temporal errors, including dynamic modes), an error is generated.

### 4.4 Error Correction

The thalamus creates a correction neuron capturing the actual co-activation context. The erroring neuron's routing table is updated: next time it sees this co-activation context, it activates the correction neuron directly, which inhibits the neuron's default d=0 vote and instead the correction neuron votes for the correct co-activation set.

### 4.5 Bootstrap Dynamics

1. **First exposure**: Neurons A, B, C co-activate. No d=0 connections exist. No predictions, no errors. Thalamus records co-activations, builds d=0 connection entries: A↔B, A↔C, B↔C.

2. **Repeated exposures**: Same neurons co-activate. d=0 connections strengthen. Each neuron now predicts its co-activation partners. Predictions match reality. No error. No correction neurons needed — the base case lives in the default d=0 connection wiring.

3. **Conflicting exposure**: A new input activates A, D, E, F, G. Neuron A predicted B and C (its established d=0 partners), but sees D, E, F, G instead. Error exceeds threshold. Thalamus creates correction neuron C1 capturing the actual co-activation set. Updates A's routing table: in context of D, E, F, G, activate C1.

4. **Subsequent exposures**: When A, B, C recur — A predicts B and C correctly, no error, default wiring handles it. When A, D, E, F, G recur — A recognizes context, activates C1 directly.

### 4.6 Two Modes of Representation

This produces an asymmetric but efficient representation:

- **First-learned patterns**: Represented in default d=0 connections of raw neurons. No correction neuron needed. Actions bind directly to the sensory/pattern neurons.
- **Subsequent conflicting patterns**: Represented by correction neurons created from prediction errors. Actions bind to the correction neurons.

This mirrors temporal processing exactly: the first observed sequence becomes the default prediction; deviations create pattern neurons that correct the default in context.

---

## 5. Hierarchical Spatial Grouping

### 5.1 How Hierarchy Emerges

The wavefront naturally produces hierarchical groupings:

- Wave 0: Pixel neurons fire. d=0 connections activate correction neurons for learned co-activation groups (stroke-like groupings).
- Wave 1: Correction neurons from Wave 0 are now active. They have their own d=0 connections to other correction neurons. Errors at this level create higher-order correction neurons (digit-like compositions).
- Wave 2+: Continues until stabilization.

No explicit "levels" are assigned. The hierarchy is emergent from the wavefront propagation depth.

### 5.2 Walkthrough: MNIST Digits "1" and "7"

Using a simplified 3x3 grid:

**Learning "1"** (vertical stroke: pixels A, B, C):
- Multiple exposures build strong A↔B, A↔C, B↔C d=0 connections.
- No conflicts yet. Default wiring represents "1". Action "1" wired to A, B, C directly — their default d=0 vote becomes "1".

**Learning "7"** (horizontal top + diagonal, sharing pixel A with "1"):
- A fires with D, E, F, G instead of B, C. Error for A. Thalamus creates C1(D, A, E, F, G).
- A's routing table updated: context D, E, F, G → activate C1.
- Action "7" wired to C1.

**Inference on new "1"**: Pixels A, B, C fire → A predicts B and C, correct → default wiring → votes for action "1".

**Inference on new "7"**: Pixels D, A, E, F, G fire → A recognizes context → activates C1 → C1 votes for action "7".

### 5.3 Single-Frame MNIST Processing

With d=0 in the wavefront, MNIST processing becomes:

1. Show one frame. Pixel neurons fire.
2. Wavefront propagates: d=0 connections trigger spatial groupings within the same frame.
3. Higher-order groupings form in subsequent waves.
4. Stabilization: all co-activation predictions resolved.
5. Wire action via `learn()` with the digit label.

No repeated frames. No temporal context window. No episodes. One frame per image. The training loop is: show image → wavefront stabilizes → learn digit → next image.

### 5.4 Generalization via Reuse and Decay

Generalization emerges from two cooperating mechanisms:

1. **Reuse (primary driver).** When the same correction neuron is reused across many distinct error events (Phase 2), each reuse strengthens the connections shared across all those events and adds new connections specific to each event. The structurally-shared connections accumulate strength; the per-event-specific connections remain weak. Over many reuses, the neuron's strong connections converge on the structural core common to the equivalence class of triggering events.

2. **Decay (secondary driver).** Standard connection decay erodes weak per-event connections faster than reinforced structural ones, sharpening the same effect.

A class neuron for "vertical stroke" emerges when the same correction neuron is repeatedly reused for vertical-stroke-like co-activation errors across many digits — its core stays strong while incidental partners decay. A class neuron for "7" emerges from repeated reuse across "7" instances with varying pixel layouts.

This is why Phase 2 (reuse) is essential, not just an optimization: a single correction neuron created from one event with no reuse only memorizes that event — decay alone has no statistical basis to identify which connections are incidental. Reuse provides the cross-instance signal that decay then sharpens.

---

## 6. Neuron Reuse for Error Correction

### 6.1 The Problem

Currently, when the thalamus detects an error, it always creates a brand new neuron. But the inference needed — the prediction, the grouping — may already exist somewhere in the network. A neuron created for a completely different context might already have connections that express exactly the required prediction.

### 6.2 The Solution: Reverse Inference Index Lookup

Reuse applies to **all distances**, not just d=0. Any error (temporal or spatial) is a candidate for reuse before minting.

The reuse criterion is **inference-output match**, not context-match: does some existing neuron's connection set already produce the inference the correction would need? The candidate's own routing table and triggering context are irrelevant to the decision — only its output signature matters.

Mechanism per error:

1. Error detected at neuron A, distance d. Thalamus knows the **observed inference set** — the actual targets that should have been inferred (the correct co-activations for d=0; the correct sequence for d>0).
2. Thalamus queries a new **reverse inference index**: for each observed target T, which existing neurons have a connection to T? (This is the inverse of "neuron N infers targets {T1, T2, …}".)
3. Take the union/intersection of those candidate sets. For each candidate, score its inference signature against the observed set using the same common/missing/novel analysis as pattern recognition.
4. If a candidate scores above the existing **merge threshold** (the same parameter that governs partial-context matching for pattern recognition): wire the erroring neuron's routing table to defer to that candidate.
5. If no candidate qualifies: mint a new neuron as before.

The symmetry is intentional. Pattern recognition asks "does this observed context partially match a stored context?" Reuse asks "does this required inference partially match an existing neuron's inference?" Both are partial-set-overlap questions; they share the same threshold. Setting that threshold to 1.0 disables reuse entirely (and also disables partial-context recognition); lowering it enables both.

**Example.** Observed inference set = (A, B, C). Candidate neuron infers (B, C). Overlap 2/3 ≈ 0.67. If the merge threshold is below 0.67, reuse. The erroring neuron's routing entry now points to the candidate; when the same context recurs, the candidate fires and provides the (B, C) inference (missing A is accepted as the cost of reuse).

### 6.3 Intra-Level Connections

This creates connections that cross traditional level boundaries. A level-1 neuron might connect to a level-4 neuron. A spatial correction neuron might connect to an existing temporal pattern neuron. These are not bugs — they are features. The neuron doesn't care how it was activated. It fires, it has connections, it votes.

This mirrors biological reality: cortical connections are not neatly layered. Skip connections, lateral connections, and feedback loops across areas are ubiquitous. The apparent "messiness" of biological wiring is actually the brain reusing existing neurons instead of creating redundant ones.

### 6.4 Benefits

- **Neuron count reduction**: No redundant neurons computing the same inference. The network stays compact.
- **Transfer learning**: If two different contexts reuse the same neuron, they are inherently linked. Knowledge transfers across domains structurally, not through explicit transfer mechanisms.
- **Robustness**: Shared representations are stronger — reinforced from multiple activation pathways.
- **Convergence speed**: The system builds on existing structure rather than rebuilding from scratch in each context.

### 6.5 Interaction with Wavefront Processing

Neuron reuse requires the wavefront model (not the level-sweep model) because a reused neuron can sit at any depth in the hierarchy. The wavefront doesn't care — it processes whatever neurons are newly activated, regardless of where they sit. The refractory period prevents cycles.

### 6.6 Content-Addressable Network

With neuron reuse, the network becomes content-addressable. The thalamus can answer the question "is there a neuron that does X?" efficiently via reverse indexes. This is structurally similar to the existing content-hash addressing and context_index, extended to cover connections.

---

## 7. Integration: Temporal + Spatial + Reuse

### 7.1 The Unified System

When both spatial processing and neuron reuse are implemented, the system has a single unified processing model:

1. Frame arrives. Sensory neurons fire.
2. Wavefront propagates. Each neuron processes all its connections — d=0 (spatial) and d>0 (temporal).
3. Matching neurons fire. Some fire because of d>0 matches (temporal sequences across frames). Some fire because of d=0 matches (spatial co-activation within this frame). Some are reused neurons activated from either domain.
4. Errors at any distance trigger correction. The thalamus first checks for reusable existing neurons. If found, wires a connection. If not, creates a new neuron.
5. Wavefront continues until no new activations.
6. All active neurons vote for actions.

There is no distinction between "temporal processing" and "spatial processing" at runtime. There is no distinction between "pattern neurons" and "moment neurons" at the implementation level. There is one type of neuron, one processing loop, one error mechanism.

### 7.2 MNIST: Pure Spatial

Show an image → wavefront processes d=0 connections → spatial hierarchy forms → wire digit action. One frame.

### 7.3 Stocks: Primarily Temporal with Spatial Enhancement

Frame arrives with market data → d>0 connections predict temporal sequences across frames → d=0 connections group co-occurring market signals within this frame into reusable units → temporal patterns operate over these compressed units instead of raw signals → the accuracy plateau breaks because the system can now represent "pullback-after-gap-up" as a single reusable token.

### 7.4 Robotics / Video: Full Integration

Camera frame arrives → d=0 connections group spatial features within the frame → d>0 connections detect temporal sequences of spatial groupings across frames → "the ball is moving left" emerges from spatial ball-detection composed with temporal motion-detection. Both dimensions needed, both processed simultaneously in the same wavefront.

---

## 8. Open Design Questions

### 8.1 Wavefront Stabilization Dynamics

With d=0 connections, potentially dozens of neurons are co-active simultaneously, all predicting each other. The error landscape could be chaotic early on. The bootstrap dynamics need empirical validation: how many exposures are needed before d=0 connections form clean enough clusters to drive useful correction neuron creation? No separate connection-strength prediction gate is introduced — the error threshold itself is expected to absorb noise-level mismatches.

### 8.2 Error Threshold for d=0

The existing error threshold (static, conservative, neutral, aggressive modes with Welford online variance) may work directly for d=0 errors. Alternatively, d=0 errors might need different sensitivity. Empirical testing will determine this.

### 8.3 Reverse Inference Index Design

The lookup shape is: `target_neuron → set of (source_neuron, distance)` — for each target, which neurons have a connection to it at which distance. Updated on connection creation, strengthening, deletion. Sharded by the column owning the source neuron so the lookup parallelizes across regions. Open: storage layout (flat hashmap vs nested), update batching cadence, eviction of decayed connections.

### 8.4 Partial Match Scoring for Neuron Reuse

Reuse uses the **same merge threshold** as partial-context matching. The two operations are symmetric — observed-vs-stored-context overlap and required-vs-existing-inference overlap are the same partial-set-match shape — so they share the parameter. No separate `reuseMergeThreshold`. Tuning the global merge threshold is empirical and affects both behaviors together, which is the intended coupling.

### 8.5 Forget Rate for d=0 Connections

The current forget rate is level-dependent. Since `level_index` in Memory is retained as a spatial organization, level-based forget rates can still apply — d=0 connections inherit the forget rate of the level their source neuron lives in. Alternative anchors (depth in activation chain, uniform rate) remain on the table if level-based decay produces wrong gradients in practice.

### 8.6 Action Binding Durability Through Generalization

When a correction neuron generalizes via reuse + decay, does the action binding transfer cleanly? Or can it get diluted? Needs validation, especially for MNIST where the action binding is the core deliverable.

---

## 9. Expected Impact

### 9.1 MNIST

d=0 connections should make MNIST tractable with single-frame processing. Expected: digit recognition after training on hundreds of images (not thousands of repeated frame episodes). The training loop reduces from "show same image N times across temporal frames" to "show image once, wavefront stabilizes, wire action."

### 9.2 Stock Trading

The accuracy plateau (~57-59% directional prediction) may be a manifestation of the same bottleneck in milder form. d=0 connections allow the system to group co-occurring market signals into reusable tokens, giving temporal patterns a compressed vocabulary to work with. Neuron reuse further amplifies this by sharing learned structure across different market contexts.

### 9.3 General Architecture

The unified d=0/d>0 wavefront model transforms Robot Brain from a temporal-sequence-only architecture into one that handles both temporal and spatial structure simultaneously. Combined with neuron reuse, the network becomes a content-addressable, self-compressing prediction engine — the architectural prerequisite for general-purpose sensory processing.

---

## 10. Implementation Plan

### 10.1 Priority

Implement spatial processing and neuron reuse before the trading cloud infrastructure. The MNIST experiment demonstrates these are structural prerequisites, not optimizations. Estimated timeline: 2-3 weeks.

### 10.2 Phase 1: Spatial Processing (d=0 Connections)

**Goal:** Enable spatial co-activation processing within the existing architecture. Validate on MNIST.

**Step 1.1: Wavefront Processing Model**

Replace the level-sweep loop in `brain.process_levels()` with wavefront propagation. `level_index` in Memory stays — it remains the spatial organization for co-activation scoping and forget rates. What changes is the dispatch loop and the removal of `neuron.level` as an intrinsic routing property.

- Add `fired_this_frame: FxHashSet<NeuronId>` to track refractory state.
- Add `pending_activation: Vec<NeuronId>` as the wavefront queue.
- Process neurons from the queue in waves. Within-wave ordering is not required to be deterministic; parallel dispatch across regions/columns is preserved.
- Terminate when a wave produces no new activations.
- Neurons fire at most once per frame (refractory constraint).
- Action voting runs once, post-stabilization, over the union of all fired neurons.

Key code changes:
- `brain.rs`: Replace `process_levels()` loop with wavefront loop.
- `memory.rs`: Add wavefront queue management. Keep `level_index` for context scoping and forget rates.
- `thalamus.rs`: `process_level()` becomes `process_neurons()` — processes a batch of neurons regardless of `neuron.level`.

**Step 1.2: d=0 Connection Support**

Extend the connection infrastructure to handle distance zero:
- `neuron.rs`: `connections: Vec<FxHashMap<NeuronId, ConnectionData>>` currently starts at distance 1 (distance 0 unused). Enable d=0 entries.
- `neuron.rs`: `learn_connections()` currently skips age=0. Add d=0 co-activation learning: when a neuron is active at age 0, record d=0 connections to all other neurons active at age 0.
- `neuron.rs`: `vote()` currently returns connections at `age + 1`. Add d=0 voting: return d=0 connections when the neuron is active.

**Step 1.3: d=0 Error Detection and Correction**

Extend the error detection to handle d=0 prediction failures:
- `thalamus.rs`: In `process_level()` / the new `process_neurons()`, add d=0 error evaluation. When a neuron's d=0 predictions don't match reality, generate an error.
- `thalamus.rs`: `allocate_pattern_neuron()` adapted to create correction neurons for d=0 errors with co-activation context entries at distance 0.
- The correction neuron is wired into the erroring neuron's routing table with d=0 context.

**Step 1.4: MNIST Validation**

- Create MNIST test harness: single frame per image, wavefront processing, action wiring via `learn()`.
- Training loop: show image → wavefront stabilizes → learn digit → next image.
- Metrics: accuracy vs. number of training images, neuron count, wavefront depth, stabilization time.
- Success criterion: >50% accuracy on unseen digits with <1000 training images (proving the mechanism works, not competing with CNNs).

### 10.3 Phase 2: Neuron Reuse for Error Correction

**Goal:** Reduce redundant neuron creation by reusing existing neurons whose connections already express the needed inference.

**Step 2.1: Reverse Connection Index**

Build a reverse index for connections, analogous to `context_index`:
- `thalamus.rs` or `column.rs`: Add `connection_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>` — maps (target_neuron, distance) → set of neurons that have a connection to that target at that distance.
- Update on connection creation, strengthening, and deletion.

**Step 2.2: Reuse Lookup During Error Correction**

Modify the error correction path to check for reusable neurons:
- `thalamus.rs`: Before `allocate_pattern_neuron()`, query the reverse connection index for neurons whose connections match the required correction context.
- Score candidates using partial matching (same common/missing/novel analysis as pattern recognition).
- If a candidate scores above the merge threshold: wire a connection to it instead of creating a new neuron.
- If no candidate qualifies: create a new neuron as before.

**Step 2.3: Cross-Level Activation**

With neuron reuse, a neuron at any depth can be activated from any other:
- Verify the wavefront model handles this correctly (it should — fire once per frame, process from queue).
- Test for cycle handling: neuron A activates neuron B which (through its connections) would activate A again → refractory period blocks the cycle.
- Validate determinism under cross-level activation.

**Step 2.4: Validation**

- Run MNIST with reuse enabled. Compare neuron count vs. Phase 1 (expect significant reduction).
- Check for transfer learning effects: does training on digits 0-4 help with digits 5-9?
- Run stock trading with reuse. Evaluate whether shared neurons across symbols improve generalization.

### 10.4 Phase 3: Stock Trading Integration

**Goal:** Apply the unified architecture to the stock trading system.

- Review all changes in the `mnist` branch and merge only the necessary changes to `main` — review and decide which scaffolding/experiments stay in the branch and which become permanent.
- Enable d=0 connections on stock market data. Within each frame, co-occurring market signals across symbols are grouped via d=0 connections.
- Temporal patterns (d>0) operate over the compressed spatial groupings.
- Evaluate impact on directional prediction accuracy and per-episode ROI.
- Compare against the temporal-only baseline.

### 10.5 Phase 4: Forgetting and Class Neuron Generalization

**Goal:** Validate and tune the generalization path from specific correction neurons to abstract class neurons.

- Monitor connection decay on d=0 correction neurons across training.
- Verify that incidental connections decay while structural connections strengthen.
- Validate that class neurons (generalized correction neurons) maintain action bindings correctly.
- Tune decay rates if needed — d=0 connections may need different rates than d>0.

---

## 11. Risk Assessment

### 11.1 High Confidence (85-90%)

The architecture is correct. Unified d=0/d>0 processing through prediction error is mathematically sound and biologically grounded. The temporal-only system already validates the core mechanism on stock data.

Wavefront termination is self-guaranteed: refractory bounds firings at one per existing neuron per frame, and newly minted correction neurons cannot themselves error in the same frame (their d=0 connections are either empty or exactly equal to the co-activation set they were built from, so their predictions can't mismatch). No safety-valve wave-count limit needed.

### 11.2 Moderate Confidence (60-70%)

MNIST works within the timeline. The bootstrap dynamics — how many exposures are needed before d=0 connections form useful clusters — are unknown. Wavefront stabilization on high-dimensional spatial data is untested.

### 11.3 Key Risks

- **Bootstrap noise**: Early training may produce many correction neurons before d=0 connections have stabilized. Mitigation: the error threshold absorbs noise-level mismatches; the reuse mechanism (Phase 2) prevents redundant minting even when errors are abundant.
- **Generalization failure**: d=0 correction neurons may over-memorize specific pixel configurations rather than generalizing to stroke-like features. Mitigation: reuse provides the cross-instance signal; decay sharpens it. Tune merge threshold and decay rates if generalization is too slow or too aggressive.
- **Reverse-inference-index cost**: Per-frame reuse lookup could dominate runtime if poorly indexed. Mitigation: batch one lookup per level per frame (piggyback on existing op-3/op-4 dispatch); shard by column for parallel evaluation across regions.
- **Performance**: Wavefront model may be slower than level-sweep for temporal-only workloads (stocks). Mitigation: profile and optimize; wavefront with mostly d>0 connections should degrade to similar performance.
