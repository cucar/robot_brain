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

### 3.4 Two-Phase Processing Model

Each frame is processed in two **sequential phases**: spatial first, temporal second.

The motivation is the feedforward sensory cascade. Object identification must precede motion tracking — you cannot have a d>0 prediction "ball moves left" until "ball" is a recognized entity in both frames. If temporal ran first, its d>0 predictions could only operate on raw sensory neurons (the only thing active at age=0 within the frame), which is the combinatorial-explosion bottleneck the design was meant to fix. By running spatial first, temporal sees the spatial hierarchy's apex as its input — objects and groupings, not raw pixels — and its d>0 predictions form between meaningful entities across frames.

Spatial and temporal are structurally symmetric: each has its own level hierarchy and its own per-frame processing loop. They differ only in distance — spatial uses d=0 (within-frame co-activation), temporal uses d>0 (across-frame sequence) — and in how their hierarchies are realized within a single frame:

- **Spatial:** wavefront within the frame. Seeds from L=0 sensory; routing matches and fresh mints propagate hierarchy upward wave by wave.
- **Temporal:** level-sweep within the frame (the existing `process_levels()` pipeline, renamed). Iterates `for level in 0..max_active_level`; at each level, d>0 pattern recognition uses the prior-frame context window.

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
    │     │           activate L=1 spatial corrections via routing.
    │     ├─ Wave 1: L=1 corrections fire. Their d=0 connections
    │     │           activate L=2 corrections via routing.
    │     ├─ Wave N: continues until no new activations.
    │     │           Refractory: each neuron fires at most once
    │     │           per spatial phase.
    │     ▼
    │   At spatial stabilization:
    │     • Evaluate d=0 predictions against the observed
    │       co-activation set (= fired neurons in this spatial
    │       phase, minus correction_wired_this_frame).
    │     • For each error: reuse lookup, else mint correction
    │       neuron at (erroring_neuron.activation_level + 1). Newly-wired
    │       correction targets go into correction_wired_this_frame.
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
    │     • d>0 error detection and correction (mint or reuse)
    │
    ▼
Action voting
    │   Over the union of all voting neurons (fired_this_frame
    │   minus correction_wired_this_frame), across both phases.
    │
    ▼
Next frame
```

**Spatial inputs.** `process_spatial` seeds from L=0 sensory neurons activated this frame. The wavefront then propagates upward via existing d=0 routing matches: a fired neuron's routing table is consulted; matches activate the corresponding higher-level spatial correction; that correction fires in the next wave; its own routing is consulted; and so on. Each frame can deepen the spatial hierarchy by up to one fresh mint layer at the top (the correction-wired inhibition rule, §3.5, bounds same-frame deepening to one layer; cross-frame routing accumulation lets subsequent frames reach further).

**Spatial→temporal handoff.** `process_temporal`'s active-set construction is enriched with the spatial output: of the neurons fired in `process_spatial` that are eligible to vote (not in `correction_wired_this_frame`), the subset whose level equals the max active level for their age. For the MNIST first frame — no spatial hierarchy yet — the handoff is just the L=0 sensory neurons (which spatial fired but didn't deepen). For mature operation, the handoff is whatever apex spatial groupings emerged from the wavefront. Lower-level neurons that contributed to higher-level spatial corrections are excluded from the handoff; their relationships are already encoded by the corrections that absorbed them.

**Inference scope (the "co-activation reality" set).** What counts as the validation set for a d=0 prediction — same-level fired neurons, base-level fired neurons, or all fired neurons regardless of level — is determined by the **Phase 2 inference scope experiment** in the implementation plan. The experiment runs three variants (`base`, `same-level`, `all-levels`) on the stocks workload at d>0 and picks the winner; the same rule then applies uniformly to d=0. The body of this design uses "co-active neurons in this frame" as a placeholder; the concrete scope is whichever Phase 2 selects.

**Why the temporal pipeline stays put.** The existing `process_levels()` (renamed `process_temporal`) is well-tuned for d>0 on stocks. With spatial-first, it receives a richer active set (sensory + spatial apex) but its internal mechanism — level-sweep with d>0 pattern recognition — is unchanged.

### 3.5 Refractory and Correction-Wiring Inhibition

Each neuron fires at most once per spatial phase. A `fired_this_frame: FxHashSet<NeuronId>` set tracks all neurons that have fired in either phase this frame, and is also consulted by `process_spatial`'s wavefront to enforce refractory within the spatial phase. If an activation signal reaches a neuron that has already fired, it is ignored. This bounds the spatial wavefront by total neuron count.

Additionally, a `correction_wired_this_frame: FxHashSet<NeuronId>` set tracks every neuron whose activation this frame is the result of being selected as a correction target — whether it was minted fresh, or reused from an existing neuron. Neurons in this set:

- **Learn from the current observed set.** They participate in connection strengthening so that the d=0 connections of a reused neuron gradually generalize toward the observed reality across reuse events (§5.4).
- **Do not vote.** Their activation this frame is a wiring side-effect, not an inferential signal. Action voting excludes them.
- **Are not error-checked.** Their d=0 predictions are not evaluated this frame. This prevents a reused neuron — whose pre-existing d=0 set may not match the current observed set — from generating a fresh error and cascading into more corrections within the same phase.

The inhibition rule is unified: the trigger isn't novelty (newly minted), it's role this frame (correction-wired). Minted neurons were always a special case where their predictions trivially matched reality; reused neurons need explicit exclusion because their predictions do not trivially match.

Both sets clear at frame end.

### 3.6 Levels as Activation State, Not Neuron State

Levels are a property of *activations in active memory*, not of neurons. A neuron has no intrinsic level field. When a neuron is activated this frame, it is registered in active memory at `activating_neuron.activation_level + 1` (sensory neurons start at activation level 0). The temporal level-sweep iterates active-memory levels: at iteration L, it processes whoever is activated at level L this frame. The spatial wavefront propagates by routing match independent of level.

This design matters specifically because of reuse. A neuron R reused for an error at A (activation level 2) appears at activation level 3 this frame, regardless of where R was originally minted or where it was last activated. Next frame, the same R may be activated from a different routing source at a different level — appearing at a different activation level. R's identity is preserved; its "level" is per-frame contextual.

Without this design, cross-level reuse would either be unsafe (the level-sweep might never reach R's intrinsic level, leaving R's d>0 work and votes silently dropped) or require restricting reuse candidates to matching levels (shrinking the reuse pool significantly). With per-activation levels, neither problem exists.

---

## 4. Co-Activation Prediction and Error

### 4.1 Connection Building

Co-activation connections (d=0) are built through observation, identical to temporal connections. When neurons A and B are co-active in a frame, both strengthen their d=0 connections to each other. Over repeated exposures, strongly co-activating neurons develop strong mutual d=0 connections.

### 4.2 Co-Activation as Prediction

A neuron's d=0 connections constitute predictions: "when I fire, I expect these neurons to also be firing." This is the spatial analog of temporal prediction ("when I fire, I expect neuron B at distance d").

### 4.3 Error Detection

When a neuron fires during `process_spatial`, its d=0 connections constitute a predicted co-activation set. After spatial wavefront stabilization, the thalamus compares this predicted set against **reality = the neurons fired in this spatial phase, minus `correction_wired_this_frame`** (scoped per the Phase 2 inference scope decision). Missing predictions (expected neurons that didn't fire) and novel observations (unexpected neurons that did fire) both contribute to the mismatch rate. The intersection of predicted and observed can be empty — predictions are not constrained to be a subset of reality. If the mismatch exceeds the error threshold (same threshold mechanism as temporal errors, including dynamic modes), an error is generated. Neurons in `correction_wired_this_frame` do not produce predictions for evaluation (§3.5).

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

Spatial hierarchy emerges along two axes — within a single spatial phase (via the spatial wavefront and existing routing) and across frames (via the routing tables accumulating new entries over time).

**Within a spatial phase (spatial wavefront).** When `process_spatial` runs in frame N:

- Wave 0: L=0 sensory fires. d=0 routing matches on each sensory neuron activate L=1 spatial corrections (those minted in prior frames).
- Wave 1: L=1 corrections fire. d=0 routing matches on them activate L=2 spatial corrections.
- Wave K: continues until no new routing matches.
- At stabilization: d=0 predictions evaluated. For each error → mint correction at `erroring_neuron.activation_level + 1` (or reuse an existing neuron via the inference index — the reused neuron is registered at the same `erroring.activation_level + 1` this frame, regardless of where it was originally minted). Fresh mints/reuses go into `correction_wired_this_frame` and do not fire this frame.

A single frame can therefore deepen the spatial hierarchy by at most one fresh layer at the top, plus however deep the existing routing structure already reaches. The notion of "level" is a per-frame *activation level* in active memory, not an intrinsic neuron property — a neuron activated this frame from a level-2 source appears at level 3; the same neuron activated next frame from a level-4 source appears at level 5. This is what makes cross-context reuse work without level mismatches.

**Across frames (routing accumulation).** Frame N's fresh mints become routing-table entries in their source neurons. In frame N+1's spatial wavefront, those entries match and fire the corresponding correction neurons mid-wavefront — so the wavefront naturally reaches one level deeper than it did in frame N. Over many frames, the spatial hierarchy reaches arbitrary depth, even though any single frame only adds one fresh layer.

**Spatial→temporal handoff.** The highest-level activations per age from this frame's spatial wavefront are folded into `process_temporal`'s active set. Temporal then does its own level-sweep over the enriched set, learning d>0 sequences over spatial groupings rather than over raw pixels.

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

With spatial-first, MNIST processing becomes:

1. Show one frame. L=0 pixel neurons activate.
2. `process_spatial` runs. Spatial wavefront seeds from the L=0 pixels. On first exposure, no L=1 corrections exist yet — wavefront stops at Wave 0, mints L=1 correction(s) from d=0 errors. On subsequent exposures, routing matches activate previously-minted corrections, the wavefront deepens, and possibly one new layer mints at the top.
3. `process_temporal` runs. The handoff (highest-level spatial activations per age) becomes part of temporal's active set. For single-frame MNIST there's no cross-frame context, so temporal does no useful work, but it runs uniformly for architectural consistency.
4. Action voting runs across all voting neurons (sensory + spatial activations from routing matches), excluding `correction_wired_this_frame`.
5. `learn()` wires the digit action to the active voting set.

No repeated frames. No temporal context window. No episodes. One frame per image. The training loop is: show image → spatial + temporal phases run → learn digit → next image.

Inference is symmetric: on a test image, the spatial wavefront activates pre-trained correction neurons via routing matches (not as fresh wirings — routing-match activations are *not* in `correction_wired_this_frame`), and those activated neurons vote for the digit action. Only fresh mint-or-reuse events from this frame's errors are inhibited from voting.

### 5.4 Generalization via Reuse and Decay

Generalization emerges from two cooperating mechanisms:

1. **Reuse (primary driver).** When the same correction neuron is reused across many distinct error events (Phases 8-9), each reuse strengthens the connections shared across all those events and adds new connections specific to each event. The structurally-shared connections accumulate strength; the per-event-specific connections remain weak. Over many reuses, the neuron's strong connections converge on the structural core common to the equivalence class of triggering events.

2. **Decay (secondary driver).** Standard connection decay erodes weak per-event connections faster than reinforced structural ones, sharpening the same effect.

A class neuron for "vertical stroke" emerges when the same correction neuron is repeatedly reused for vertical-stroke-like co-activation errors across many digits — its core stays strong while incidental partners decay. A class neuron for "7" emerges from repeated reuse across "7" instances with varying pixel layouts.

This is why reuse (Phases 8-9) is essential, not just an optimization: a single correction neuron created from one event with no reuse only memorizes that event — decay alone has no statistical basis to identify which connections are incidental. Reuse provides the cross-instance signal that decay then sharpens.

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

### 6.5 Interaction with Two-Phase Processing

Reuse is the same operation at both d>0 (during `process_temporal` error correction) and d=0 (during `process_spatial` error correction). A reused neuron may sit at any level in the hierarchy — that's fine, because reuse wires the erroring neuron's routing table to point at the reused neuron, and the reused neuron then fires on subsequent frames when its routing context recurs. Within the frame where the reuse is wired, the reused neuron is added to `correction_wired_this_frame` (§3.5): it learns from the current observed set but does not vote and is not error-checked. This rule, not the wavefront model, is what bounds the system; the level-sweep for d>0 stays intact.

### 6.6 Content-Addressable Network

With neuron reuse, the network becomes content-addressable. The thalamus can answer the question "is there a neuron that does X?" efficiently via reverse indexes. This is structurally similar to the existing content-hash addressing and context_index, extended to cover connections.

---

## 7. Integration: Temporal + Spatial + Reuse

### 7.1 The Unified System

When spatial processing and neuron reuse are both implemented, the per-frame pipeline is:

1. Frame arrives. L=0 sensory neurons activate.
2. `process_spatial` runs. Spatial wavefront seeds from L=0 sensory. d=0 routing matches activate higher-level spatial corrections wave by wave. At stabilization, d=0 errors trigger correction — reuse lookup first, mint as fallback. Correction-wired neurons go into `correction_wired_this_frame`.
3. Handoff: highest-level activations per age from step 2 are folded into temporal's active set.
4. `process_temporal` runs (existing level-sweep). d>0 connections drive pattern recognition over the enriched active set (sensory + spatial handoff). d>0 errors trigger correction — reuse lookup first, mint as fallback.
5. Action voting over the union of voting neurons from both phases, excluding `correction_wired_this_frame`.

There is one neuron type, one error mechanism, one reuse mechanism, one routing infrastructure. The two phases are structurally symmetric — each has its own level hierarchy and its own per-frame processing — but they live at different distances (d=0 spatial, d>0 temporal) and run sequentially within a frame. Spatial precedes temporal because object identification must precede motion tracking: temporal's cross-frame predictions are over the entities spatial just identified, not over raw sensory inputs.

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

Forget rate is a global static (set in Phase 1, replacing the prior level-dependent rate). Experimentation on the `mnist` branch found no meaningful accuracy difference between level-dependent and static decay, and static is simpler and more biologically defensible (no reason higher-level neurons should follow different decay rules). One global parameter, no per-neuron forget-rate field, no per-level forget-rate table. d=0 connections decay at the same rate as d>0.

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

The detailed implementation plan lives in [moment-neurons-v2-impl.md](./moment-neurons-v2-impl.md) and is the source of truth for phasing, code touched, and acceptance criteria. This section is a one-line summary per phase.

| # | Phase | Goal |
|---|---|---|
| 1 | Branch Reconciliation | Merge keepable parts of `mnist` branch into `dev` |
| 2 | Inference Scope Experiment | Pick `base` / `same-level` / `all-levels` on stocks at d>0 |
| 3 | Spatial-Phase Scaffolding | Add empty `process_spatial()` before temporal; rename `process_levels` → `process_temporal`; plumb `fired_this_frame` and `correction_wired_this_frame` |
| 4 | d=0 Connection Learning | Per-neuron parallel d=0 strengthening during spatial frame |
| 5 | Spatial Wavefront Orchestration | Thalamus drives wave-by-wave dispatch seeded from L=0 sensory |
| 6 | d=0 Error Detection + Correction Minting | Evaluate predictions post-wavefront; mint corrections at level+1 |
| 7 | MNIST Single-Frame Harness | Validate on MNIST: >50% accuracy with <1000 training images |
| 8 | Reverse Inference Index | Build `target → distance → sources` lookup |
| 9 | Reuse Lookup in Correction Path | Reuse existing neurons whose inference signature matches |
| 10 | Reuse Validation | Neuron count drop + transfer test (0-4 → 5-9) |
| 11 | Stocks Integration | Run stocks with full pipeline; compare to baseline |
| 12 | Forget-Rate & Class-Neuron Generalization | Long-run training; tune decay if needed |
| 13 | Persistence / Backup / Import-Export | Round-trip d=0 connections and dynamic-level corrections |

Phases 1-2 are prerequisites that unblock everything downstream. Phases 3-7 deliver MNIST validation. Phases 8-10 add reuse and validate transfer. Phases 11-13 are post-MNIST.

---

## 11. Risk Assessment

### 11.1 High Confidence (85-90%)

The architecture is correct. Distance-parameterized prediction error (d=0 spatial, d>0 temporal) is mathematically sound and biologically grounded. The temporal-only system already validates the core mechanism on stock data.

**Termination is self-guaranteed for both phases:**

- `process_temporal` is the existing level-sweep, bounded by `max_active_level` as it is today.
- `process_spatial`'s wavefront is bounded by refractory (each neuron fires at most once per spatial phase) plus the `correction_wired_this_frame` inhibition rule (§3.5). The latter is the load-bearing piece: a neuron in `correction_wired_this_frame` does not produce d=0 predictions for evaluation this frame, so it cannot generate fresh errors. Both newly-minted and reused correction targets fall under this rule, so neither can trigger an error cascade within the spatial phase. Total spatial-phase work is bounded by the spatial hierarchy depth and the size of the input set, not by error-chain length.

No safety-valve wave-count limit needed.

### 11.2 Moderate Confidence (60-70%)

MNIST works within the timeline. The bootstrap dynamics — how many exposures are needed before d=0 connections form useful clusters — are unknown. Wavefront stabilization on high-dimensional spatial data is untested.

### 11.3 Key Risks

- **Bootstrap noise**: Early training may produce many correction neurons before d=0 connections have stabilized. Mitigation: the error threshold absorbs noise-level mismatches; the reuse mechanism (Phases 8-9) prevents redundant minting even when errors are abundant.
- **Generalization failure**: d=0 correction neurons may over-memorize specific pixel configurations rather than generalizing to stroke-like features. Mitigation: reuse provides the cross-instance signal; decay sharpens it. Tune merge threshold and decay rates if generalization is too slow or too aggressive.
- **Reverse-inference-index cost**: Per-frame reuse lookup could dominate runtime if poorly indexed. Mitigation: batch one lookup per level per frame (piggyback on existing op-3/op-4 dispatch); shard by column for parallel evaluation across regions.
- **Performance**: Wavefront model may be slower than level-sweep for temporal-only workloads (stocks). Mitigation: profile and optimize; wavefront with mostly d>0 connections should degrade to similar performance.
