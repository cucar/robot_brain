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

### 3.6 Processing Order and Determinism

Within each wave, neurons are processed in neuron-ID order to ensure deterministic behavior. This is important for reproducibility and debugging. The wavefront is essentially breadth-first propagation with deterministic ordering within each frontier.

---

## 4. Co-Activation Prediction and Error

### 4.1 Connection Building

Co-activation connections (d=0) are built through observation, identical to temporal connections. When neurons A and B are co-active in a frame, both strengthen their d=0 connections to each other. Over repeated exposures, strongly co-activating neurons develop strong mutual d=0 connections.

### 4.2 Co-Activation as Prediction

A neuron's d=0 connections constitute predictions: "when I fire, I expect these neurons to also be firing." This is the spatial analog of temporal prediction ("when I fire, I expect neuron B at distance d").

### 4.3 Error Detection

When a neuron fires and its d=0 predictions don't match reality — expected co-activations are absent or unexpected neurons are present — the mismatch rate is computed. If it exceeds the error threshold (same threshold mechanism as temporal errors, including dynamic modes), an error is generated.

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

### 5.4 Generalization via Class Neurons

A correction neuron initially captures a specific co-activation event. Over time and repeated partial matches, connections to neurons that consistently co-activate retain strength while connections to incidental partners decay. The correction neuron gradually loses specificity and retains only the structural core — becoming a class neuron.

A class neuron for "vertical stroke" fires for any vertical stroke regardless of exact pixel position. A class neuron for "7" fires for any digit with the right compositional structure. This generalization emerges from connection decay — the same mechanism as temporal pattern context refinement, no additional infrastructure needed.

---

## 6. Neuron Reuse for Error Correction

### 6.1 The Problem

Currently, when the thalamus detects an error, it always creates a brand new neuron. But the inference needed — the prediction, the grouping — may already exist somewhere in the network. A neuron created for a completely different context might already have connections that express exactly the required prediction.

### 6.2 The Solution: Reverse Index Lookup

Instead of always minting new neurons, the thalamus checks whether an existing neuron already performs the required inference. The mechanism:

1. Error detected at neuron A, distance d.
2. Thalamus computes the required inference: what connections does the correction neuron need?
3. Thalamus looks up the reverse connection index (analogous to the existing context_index for routing tables, but for connections).
4. Partial matching: find existing neurons whose connections best match the required inference. Same scoring logic as pattern recognition — exact match preferred, partial match with penalty acceptable.
5. If a sufficiently matching neuron exists: wire a connection to it instead of creating a new one. The erroring neuron's routing table points to the existing neuron.
6. If no match: create a new neuron as before.

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

With d=0 connections, potentially dozens of neurons are co-active simultaneously, all predicting each other. The error landscape could be chaotic early on — errors everywhere, correction neurons minting explosively before connections have stabilized. The bootstrap dynamics need empirical validation: how many exposures are needed before d=0 connections form clean enough clusters to drive useful correction neuron creation?

### 8.2 Connection Strength Threshold for d=0

When do d=0 connections become strong enough to constitute a prediction? This threshold determines sensitivity to noise and the number of exposures needed before the system can form spatial groupings. May need to be tuned independently from d>0 thresholds, or may work with the same parameters.

### 8.3 Error Threshold for d=0

The existing error threshold (static, conservative, neutral, aggressive modes with Welford online variance) may work directly for d=0 errors. Alternatively, d=0 errors might need different sensitivity than d>0 errors. Empirical testing will determine this.

### 8.4 Reverse Index Design for Neuron Reuse

The existing `context_index` provides fast lookup for pattern candidates during recognition. A similar reverse index is needed for connections to enable the thalamus to find reusable neurons. Design questions: index by connection targets? By connection hash? By partial connection signature? The lookup must be fast enough to not bottleneck correction.

### 8.5 Partial Match Scoring for Neuron Reuse

When the thalamus finds a candidate neuron for reuse, how good must the match be? Exact match only? Or accept partial matches with some threshold? Too strict means few reuse opportunities. Too loose means incorrect reuse that introduces errors. The scoring mechanism should mirror the existing pattern matching score (common/missing/novel analysis).

### 8.6 Forget Rate for d=0 Connections

The current forget rate is level-dependent (exponentially slower for deeper patterns). Without fixed levels, forget rate needs a different anchor — perhaps depth in the activation chain (hops to nearest sensory neuron), or a uniform rate for all d=0 connections, or the same decay as the neuron's d>0 connections.

### 8.7 Action Binding Durability Through Generalization

When a correction neuron generalizes into a class neuron through connection decay, does the action binding transfer cleanly? Or can it get diluted as incidental connections decay? This needs validation, especially for MNIST where the action binding is the core deliverable.

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

Replace the level-sweep loop in `brain.process_levels()` with wavefront propagation:
- Remove `level_index` from Memory as a dispatch mechanism (retain for diagnostics if needed).
- Add `fired_this_frame: FxHashSet<NeuronId>` to track refractory state.
- Add `pending_activation: Vec<NeuronId>` as the wavefront queue.
- Process neurons from the queue in waves, each wave processing in neuron-ID order for determinism.
- Terminate when a wave produces no new activations.
- Neurons fire at most once per frame (refractory constraint).

Key code changes:
- `brain.rs`: Replace `process_levels()` loop (lines 1520-1576) with wavefront loop.
- `memory.rs`: Add wavefront queue management. Simplify or remove `level_index` for dispatch.
- `thalamus.rs`: `process_level()` becomes `process_neurons()` — processes a batch of neurons regardless of level.

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

### 11.2 Moderate Confidence (60-70%)

MNIST works within the two-week window. The bootstrap dynamics — how many exposures are needed before d=0 connections form useful clusters — are unknown. Wavefront stabilization on high-dimensional spatial data is untested.

### 11.3 Key Risks

- **Bootstrap explosion**: Early training may produce too many correction neurons before d=0 connections have stabilized. Mitigation: connection strength threshold before prediction kicks in.
- **Wavefront non-termination**: Circular activation chains despite refractory period (shouldn't happen mathematically, but edge cases possible). Mitigation: hard wave-count limit as safety valve.
- **Generalization failure**: d=0 correction neurons may over-memorize specific pixel configurations rather than generalizing to stroke-like features. Mitigation: connection decay tuning, sufficient training variety.
- **Performance**: Wavefront model may be slower than level-sweep for temporal-only workloads (stocks). Mitigation: profile and optimize; wavefront with mostly d>0 connections should degrade to similar performance.
