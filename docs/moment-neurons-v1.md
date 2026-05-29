# Robot Brain: Moment Neuron Architecture Design Document

**Date:** May 28, 2026  
**Author:** Cagdas Ucar  
**Status:** Draft — Pre-Implementation  

---

## 1. Motivation

### 1.1 The MNIST Experiment

The Robot Brain was tested on MNIST data using the following setup: 784 pixel channels (reduced to 49 for compute constraints) fed as independent input channels to the brain, with repeated frames shown until cortex stabilization, followed by action implantation for digit labels. The cortex-only approach produced correct but catastrophically inefficient results — requiring impractical computation time to build a useful representational tree.

### 1.2 The Core Insight

The cortex's pattern recognition mechanism is designed for temporal sequences — it detects "A then B" across time distances. MNIST digits have no temporal structure. Feeding the same frame repeatedly and asking the cortex to build hierarchy through temporal repetition forces it to represent spatial co-activation as a flat combinatorial expansion. This is equivalent to writing a logical expression without parentheses — the expression is valid but combinatorially intractable.

### 1.3 The Role of Moment Neurons

Moment neurons provide the "parentheses" — temporary storage for co-activation patterns that compress the representational space. They answer the question "what belongs together?" so that the cortex can efficiently answer "what follows what?" Without moment neurons, the cortex must pattern-match over the full combinatorial space of raw inputs. With them, the cortex operates over a compressed vocabulary of pre-grouped units.

This is not an MNIST-specific optimization. It is core Brain functionality required for any domain where spatial or simultaneous structure exists — vision, point-in-time market states, sensor arrays, and any multi-channel sensory input.

---

## 2. Biological Parallels

### 2.1 Pattern Neurons as Pyramidal Cells

Cortical pyramidal cells are the canonical sequence detectors. They receive top-down context via apical dendrites and feedforward input via basal dendrites, learning temporal predictions. Pattern neurons in Robot Brain map directly to this: they detect "A then B" with context sensitivity, using routing tables with time-distance connections.

### 2.2 Moment Neurons as Grid Cells

Grid cells in entorhinal cortex fire in periodic spatial patterns, encoding position through co-activation. A set of grid cells firing together represents "where you are" in some abstract space — not through sequence, but through simultaneous activation. Moment neurons serve the same role: they capture "what's active together right now" and mint a reusable reference to that state.

Grid cells sit in entorhinal cortex — the gateway to hippocampus — compressing high-dimensional cortical output into a spatial/relational code. Moment neurons sitting between cortex and hippocampus proper occupy the same structural position, compressing co-active pattern neurons into grouped units.

### 2.3 The Two Complementary Operations

- **Pyramidal cells / Pattern neurons**: learn *when* (temporal sequences)
- **Grid cells / Moment neurons**: learn *what together* (spatial/relational co-activation)

The brain requires both operations to build efficient representations.

---

## 3. Architecture

### 3.1 Core Principle: Symmetry with Cortex

Moment processing uses the same fundamental mechanism as cortical processing. The key differences are:

| Property | Cortex (Pattern Neurons) | Moment Processing (Moment Neurons) |
|---|---|---|
| **Prediction axis** | Temporal — "what comes next" | Simultaneous — "what's active with me" |
| **Connection semantics** | Time-distance connections to future events | Co-activation connections to concurrent neurons |
| **Error condition** | Predicted sequence didn't occur | Predicted co-activations didn't match |
| **Creation trigger** | Thalamus detects prediction error | Thalamus detects co-activation prediction error |
| **Routing** | Via thalamus, same infrastructure | Via thalamus, same infrastructure |
| **Forgetting path** | Standard decay | Decay toward class neurons (generalization) |

### 3.2 The Neuron's Perspective

Every neuron maintains co-activation connections: a list of other neurons it has been active with, along with connection strengths that strengthen on co-activation and decay over time. This is structurally identical to the cortical routing table, but all time distances are zero.

During moment processing, each neuron uses these co-activation connections as predictions: "when I fire, I expect neurons B, C, D to also be firing." The neuron does not "claim" other neurons or reach down — it pushes activation up to any moment neurons already in its routing table, and reports its co-activation predictions to the thalamus.

### 3.3 Thalamus Role

The thalamus plays the same centralized orchestration role as in cortical processing:

1. **Collect reports**: Each active neuron reports its co-activation predictions (strongest N connections, analogous to context length).
2. **Compare against reality**: The thalamus compares predicted co-activations against actual co-activations.
3. **Detect errors**: If the mismatch exceeds the configured error threshold (e.g., X% of predicted co-activations are missing or unexpected neurons are present), that neuron is in error.
4. **Create moment neurons**: For error neurons, the thalamus creates a new moment neuron capturing the actual co-activation set the erroring neuron found itself in.
5. **Update routing tables**: The thalamus instructs the erroring neuron to add the new moment neuron to its routing table — next time it sees this co-activation context, it activates the moment neuron directly, which inhibits the neuron's default vote and instead the moment neuron votes for the correct co-activation set.
6. **Bind to pattern hierarchy**: At creation time, currently active pattern neurons bind the new moment neuron into their own routing tables, integrating moment neurons into the cortical vocabulary.

### 3.4 Processing Flow

```
Frame arrives
    │
    ▼
Cortex Processing (multiple levels until stabilization)
    │  Pattern neurons predict temporal sequences
    │  Errors create new pattern neurons
    │  Stabilization: no new firings, no new patterns
    │
    ▼
Moment Processing (multiple levels until stabilization)
    │
    ├─ Level 1:
    │    Active neurons push activation to known moment neurons
    │    Neurons report co-activation predictions to thalamus
    │    Thalamus compares predictions vs reality
    │    Errors → new moment neurons created
    │    Routing tables updated
    │
    ├─ Level 2:
    │    Active moment neurons from Level 1 push up
    │    Moment neurons report their co-activation predictions
    │    Thalamus compares, creates higher-order moment neurons on error
    │
    ├─ Level N:
    │    Continues until a round produces no changes
    │
    ▼
Stabilization: no new moment neurons, no new activations
    │
    ▼
Action voting (both pattern and moment neurons participate)
    │
    ▼
Next frame (moment neurons now available as cortex inputs)
```

### 3.5 Brain Parameter: When to Run Moment Processing

Moment processing frequency is controlled by a brain-level parameter, not hardcoded. The mechanism is identical regardless of frequency — only the timing varies:

- **Every frame**: Appropriate for recognition tasks (MNIST)
- **On salience signal**: Appropriate for streaming data (stock trading)
- **Every N frames**: Configurable periodic processing
- **On cortex stabilization**: Default — run after cortex settles

The parameter controls *when*, not *what*. What gets grouped is entirely determined by the co-activation connection landscape.

---

## 4. Bootstrap and Learning Dynamics

### 4.1 The Bootstrap Sequence

Moment neurons are not created on first exposure. The system must first build co-activation connections through observation before predictions (and therefore errors) can occur.

1. **First exposure**: Neurons A, B, C co-activate. No connections exist. No predictions, no errors, no moment neurons. Thalamus records co-activations, builds connection entries: A↔B, A↔C, B↔C. Strength initialized.

2. **Repeated exposures**: Same neurons co-activate. Connections strengthen. Each neuron now has co-activation predictions: A predicts B and C, etc. Predictions match reality. No error. No moment neurons needed — the base case lives in the default connection wiring.

3. **Conflicting exposure**: A new input activates A, D, E, F, G. Neuron A predicted B and C, but sees D, E, F, G instead. Error exceeds threshold. Thalamus creates moment neuron M1(A, D, E, F, G). Updates A's routing table: in context of D, E, F, G, activate M1.

4. **Subsequent exposures**: When the original pattern (A, B, C) recurs, A predicts B and C, prediction matches, no error, no moment neuron — default wiring handles it. When the new pattern (A, D, E, F, G) recurs, A recognizes context, activates M1 directly.

### 4.2 Two Modes of Representation

This produces an asymmetric but efficient representation:

- **First-learned patterns**: Represented in default co-activation connections of raw neurons. No moment neuron needed. Actions bind directly to the sensory/pattern neurons.
- **Subsequent conflicting patterns**: Represented by moment neurons created from prediction errors. Actions bind to the moment neurons.

This mirrors cortex exactly: the first observed sequence becomes the default prediction; deviations create pattern neurons that correct the default in context.

### 4.3 Scaling to Multiple Patterns (MNIST 10 Digits)

As more digits are learned:
- Neurons unique to a single digit never error and never need moment neurons.
- Neurons shared across digits accumulate context-dependent moment neuron activations.
- Each new digit that shares neurons with previous digits generates errors and moment neurons precisely at the points of conflict.
- The more digits learned, the richer the context routing becomes.

### 4.4 MNIST Single-Frame Processing

With moment neurons, MNIST processing simplifies dramatically:

1. Show one frame. Pixel neurons fire.
2. Moment level 1: groups co-active pixels into moment neurons (after sufficient connection building from prior exposures).
3. Moment level 2: groups co-active moment neurons into higher-order moment neurons.
4. Levels continue until stabilization.
5. Wire action to the active top-level moment/pattern neurons.

No repeated frames needed. No temporal processing of static data. The moment mechanism builds the compositional hierarchy in a single pass through levels, not through time.

---

## 5. Hierarchical Moment Creation

### 5.1 Walkthrough: Digits "1" and "7"

Using a simplified 3x3 grid:

**Learning "1"** (vertical stroke: pixels A, B, C):
- Multiple exposures build strong A↔B, A↔C, B↔C connections.
- No conflicts yet. Default wiring represents "1". Action "1" wired to A, B, C.

**Learning "7"** (horizontal top + diagonal, sharing pixel A with "1"):
- A fires with D, E, F, G instead of B, C. Error. Thalamus creates M1(D, A, E, F, G).
- A's routing table updated: context D, E, F, G → activate M1.
- Action "7" wired to M1.

**Learning "4"** (shares vertical stroke with "1", adds horizontal bar):
- Vertical stroke (A, B, C) activates — matches "1" default, no error at that level.
- Horizontal bar (B, E, H) introduces conflicts. B expected A and C, now also sees E and H.
- Error for B → thalamus creates M2 for B's new context.
- At moment level 2: M2 co-active with default vertical representation. New co-activation pattern → eventually creates M3 at level 2, representing the "4" composition.
- Action "4" wired to M3.

### 5.2 Generalization

When a slightly shifted "1" is shown (different raw pixels but similar vertical stroke), partial matching of co-activation predictions produces partial activation of the existing representations. Generalization emerges not from memorizing pixels but from connection strength decay eroding incidental position-specific connections while retaining the structural core.

---

## 6. Forgetting and Class Neurons

### 6.1 From Moment to Class

A moment neuron initially captures a specific co-activation event — the exact set of neurons that were active when it was created. Over time and repeated partial matches:

- Connections to neurons that consistently co-activate retain strength.
- Connections to neurons that only sometimes participate decay.
- The moment neuron gradually loses its specificity and retains only the core pattern.

This transforms a moment neuron (specific instance) into a class neuron (abstract category). The mechanism is connection strength decay — the same infrastructure as cortex, with potentially different decay rates as a parameter.

### 6.2 Implications for Recognition

Class neurons provide the generalization needed for robust recognition:
- A class neuron for "vertical stroke" fires for any vertical stroke regardless of exact pixel position.
- A class neuron for "7" fires for any digit that has the right compositional structure, tolerating variation in exact pixel layout.

---

## 7. Integration with Cortex

### 7.1 Moment Neurons as Cortex Vocabulary

Once created, moment neurons are available as inputs to cortical pattern processing on subsequent frames. The cortex can build temporal sequences over moment neurons:

- "Moment M1 at frame T, then moment M2 at frame T+1"
- The cortex never needs to see raw pixels for temporal reasoning — it works with compressed tokens.

### 7.2 Action Voting

Moment neurons participate in action voting identically to pattern neurons. The voting mechanism is unified:
- Pattern neurons vote based on temporal predictions.
- Moment neurons vote based on co-activation predictions.
- Both contribute to the action selection.

### 7.3 Bidirectional Binding

When a moment neuron is created:
- The moment neuron's routing table records its constituent neurons.
- The constituent neurons' routing tables are updated to include the moment neuron.
- Currently active pattern neurons bind the moment neuron into their own routing tables.

This ensures moment neurons are fully integrated into the existing hierarchy from the instant of creation.

---

## 8. Open Design Questions

### 8.1 Neuron Class: Same or Separate?

The moment neuron mechanism is nearly identical to the pattern neuron mechanism — same processFrame logic, same connection/routing table structure, same thalamus orchestration. The primary differences:

- **Prediction axis**: time-distance zero vs. time-distance > 0
- **Forgetting path**: decay toward class neuron (generalization) vs. standard decay

**Options**:
- Same base neuron class with a flag/subtype controlling decay behavior and prediction axis.
- Separate class inheriting from a shared base.

**Recommendation**: TBD — requirements analysis and implementation complexity should drive this decision.

### 8.2 Co-Activation Connection Strength Threshold

When do co-activation connections become strong enough to constitute a prediction? This determines:
- How many exposures before the system can error and create moment neurons.
- How fine or coarse the initial groupings are.
- Sensitivity to noise (weakly connected neurons that happen to co-activate).

### 8.3 Error Threshold

What percentage of predicted co-activations must be missing (or unexpected) to trigger moment neuron creation? This parallels the cortex error threshold and may use the same parameter or a separate one.

### 8.4 Cluster Overlap

A neuron can belong to multiple moment neurons (context-dependent prediction). The routing table handles this naturally — same as a pattern neuron participating in multiple sequences. No special mechanism needed; the context determines which moment neuron activates.

### 8.5 Action Binding Durability

When a moment neuron generalizes into a class neuron through forgetting, does the action binding transfer cleanly? Or can it get diluted as incidental connections decay? This needs validation.

### 8.6 d=0 Integration into Cortex Levels

An open question: should moment processing (d=0 connections) be integrated into each cortex processing level rather than running as a separate post-cortex pass? This could tighten the feedback loop but may complicate the processing model. Deferred for initial implementation.

---

## 9. Expected Impact

### 9.1 MNIST

Moment neurons should make MNIST tractable by providing the compositional grouping that the cortex cannot efficiently build from temporal repetition alone. Expected: digit recognition from single-frame processing after a training period to build co-activation connections and moment neuron hierarchy.

### 9.2 Stock Trading

The accuracy plateau (~57-59% directional prediction) may be a manifestation of the same bottleneck in milder form. Moment neurons could allow the system to mint reusable units like "pullback-after-gap-up" rather than re-detecting it from raw price deltas every frame. This could break the plateau and improve the action policy's compounding ROI.

### 9.3 General Architecture

Moment neurons transform Robot Brain from a temporal-sequence-only architecture into one that handles both temporal and spatial structure. This is the capability required for any multi-channel sensory processing — vision, audio, robotics, and beyond.

---

## 10. Implementation Plan

### 10.1 Priority

Implement moment neurons before the trading cloud infrastructure. The MNIST experiment demonstrates this is a structural prerequisite, not an optimization. Estimated timeline: 2 weeks.

### 10.2 Phased Approach

**Phase 1**: Co-activation connection tracking. Extend neuron routing tables to track d=0 connections with strength. Validate that connections build correctly from co-activation data.

**Phase 2**: Prediction and error detection. Implement co-activation prediction in processFrame for d=0 connections. Implement thalamus error detection for co-activation mismatches.

**Phase 3**: Moment neuron creation. Implement thalamus moment neuron minting on error. Implement routing table updates and integration with pattern hierarchy. Implement multi-level moment processing until stabilization.

**Phase 4**: Forgetting and class neuron transition. Implement differential decay for moment neuron connections. Validate generalization from specific instances to abstract categories.

**Phase 5**: MNIST validation. Run MNIST with moment processing enabled. Compare efficiency against cortex-only baseline. Validate digit-specific representations emerge.

**Phase 6**: Stock trading integration. Enable moment processing on stock data with salience-driven timing. Evaluate impact on directional prediction accuracy and ROI.
