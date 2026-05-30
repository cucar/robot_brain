# Neuron Reuse for Error Correction

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Status:** Pre-implementation
**Prerequisites:** [mnist-merge.md](./mnist-merge.md), [inference-level.md](./inference-level.md), [spatial-processing.md](./spatial-processing.md)

---

## 1. Motivation

### 1.1 The Problem

Currently, when the thalamus detects an error, it always creates a brand new neuron. But the inference needed — the prediction, the grouping — may already exist somewhere in the network. A neuron created for a completely different context might already have connections that express exactly the required prediction.

Without reuse, the network grows indefinitely: every error event mints a fresh neuron, regardless of structural overlap with existing neurons.

### 1.2 The Solution: Reverse Inference Index Lookup

Reuse applies to **all distances**, not just d=0. Any error (temporal or spatial) is a candidate for reuse before minting.

The reuse criterion is **inference-output match**, not context-match: does some existing neuron's connection set already produce the inference the correction would need? The candidate's own routing table and triggering context are irrelevant to the decision — only its output signature matters.

### 1.3 Why Reuse Is Essential, Not an Optimization

Reuse is what makes generalization possible.

A single correction neuron created from one event with no reuse only memorizes that event. Decay alone has no statistical basis to identify which connections are incidental.

When the same correction neuron is reused across many distinct error events, each reuse strengthens the connections shared across all those events and adds new connections specific to each event. The structurally-shared connections accumulate strength; the per-event-specific connections remain weak. Over many reuses, the neuron's strong connections converge on the structural core common to the equivalence class of triggering events.

Decay then sharpens this: incidental per-event connections erode while reinforced structural ones persist.

Reuse provides the cross-instance signal; decay sharpens it. Both are required.

---

## 2. Mechanism

### 2.1 Per-Error Lookup

For each error detected at neuron A, distance d:

1. The thalamus knows the **observed inference set** — the actual targets that should have been inferred (the correct co-activations for d=0; the correct sequence for d>0).
2. Query the **reverse inference index**: for each observed target T, which existing neurons have a connection to T? (This is the inverse of "neuron N infers targets {T1, T2, …}".)
3. Take the union of those candidate sets. For each candidate, score its inference signature against the observed set using the same common/missing/novel analysis as pattern recognition.
4. If a candidate scores above the existing **merge threshold** (the same parameter that governs partial-context matching for pattern recognition): wire the erroring neuron's routing table to defer to that candidate.
5. If no candidate qualifies: mint a new neuron as the fallback.

### 2.2 Symmetry with Pattern Recognition

The symmetry is intentional. Pattern recognition asks "does this observed context partially match a stored context?" Reuse asks "does this required inference partially match an existing neuron's inference?" Both are partial-set-overlap questions; they share the same threshold.

Setting the merge threshold to 1.0 disables reuse entirely (and also disables partial-context recognition); lowering it enables both. No separate `reuseMergeThreshold` parameter — the coupling is intentional.

### 2.3 Worked Example

Observed inference set = (A, B, C). Candidate neuron infers (B, C). Overlap 2/3 ≈ 0.67. If the merge threshold is below 0.67, reuse. The erroring neuron's routing entry now points to the candidate; when the same context recurs, the candidate fires and provides the (B, C) inference (missing A is accepted as the cost of reuse).

---

## 3. Interaction with Spatial Processing

### 3.1 Activation-Level Resolution

Reuse depends on the activation-level architecture from [spatial-processing.md](./spatial-processing.md#35-levels-as-activation-state-not-neuron-state). Neurons have no intrinsic level. When a reused neuron R is activated for an error at A (activation level 2), R appears at activation level 3 this frame, regardless of R's original mint level or where it was last activated.

This is what makes cross-context reuse safe. Without per-activation levels, a reused neuron's d>0 work could be dropped silently if its intrinsic level was never reached by the temporal level-sweep, or reuse would have to be restricted to same-level candidates (shrinking the reuse pool significantly).

### 3.2 Correction-Wired Inhibition

Reused neurons go into `correction_wired_this_frame` exactly like fresh mints. They:

- Learn from the current observed set (their d=0 connections strengthen toward the observed reality — this is how they gradually generalize across reuse events).
- Do not vote this frame (their activation is a wiring side-effect, not an inferential signal).
- Are not error-checked this frame (prevents a reused neuron whose pre-existing d=0 set doesn't match the current observed set from generating a fresh error and cascading).

This is the load-bearing termination rule. Without it, cross-frame reuse would risk runaway error cascades within a single spatial phase.

### 3.3 Reuse Across Both Phases

Reuse applies in both `process_spatial` (d=0 errors) and `process_temporal` (d>0 errors). Same mechanism, different phase. The reverse inference index is shared between phases.

---

## 4. Benefits

- **Neuron count reduction**: No redundant neurons computing the same inference. The network stays compact.
- **Transfer learning**: If two different contexts reuse the same neuron, they are inherently linked. Knowledge transfers across domains structurally, not through explicit transfer mechanisms.
- **Robustness**: Shared representations are stronger — reinforced from multiple activation pathways.
- **Convergence speed**: The system builds on existing structure rather than rebuilding from scratch in each context.
- **Content-addressable network**: The thalamus can answer the question "is there a neuron that does X?" efficiently via reverse indexes. Structurally similar to the existing content-hash addressing and `context_index`, extended to cover connections.

---

## 5. Implementation Plan

### Overview

| Phase | Goal | Validation gate |
|---|---|---|
| 1 | Reverse inference index | Unit test: target → set of source-neurons lookup |
| 2 | Reuse lookup in correction path | MNIST: neuron count drops vs spatial-mint-only baseline |
| 3 | Reuse validation | Transfer test: training on digits 0-4 helps with 5-9 |
| 4 | Stocks integration with full pipeline (spatial + reuse) | Directional accuracy lifts off the 57-59% plateau |
| 5 | Forget-rate / class-neuron generalization | Class neurons survive 10k+ frames with action binding intact |

Phases 1-2 can land in parallel with or after [spatial-processing](./spatial-processing.md) Phase 5. Phase 3 onward depends on spatial being in place.

---

### Phase 1 — Reverse Inference Index

**Goal:** Build the data structure that answers "which neurons have a connection to target T at distance d?"

#### Code touched

- `brain/brain-core/src/column.rs` or `thalamus.rs` — add `inference_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>`. Maps `target → distance → set of sources`. Sharded by the column owning the **source** neuron, so each column owns the index for its resident neurons.
- `brain/brain-core/src/neuron.rs` — on connection create/strengthen/decay/delete, emit an `IndexUpdate` event. Column applies the update to its local index.
- `brain/brain-core/src/region.rs` — expose a `query_inference_sources(targets: &[NeuronId], distance: Distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>` op that fans out across columns and merges per-target source sets.

#### Acceptance

- Unit test: create neurons N1→T1 and N2→T1 at distance 0; query for T1 at d=0 returns {N1, N2}.
- Decay a connection below threshold; index reflects removal.
- Index size stays proportional to total connection count (no bloat).

#### Notes / gotchas

- **Update batching at orchestration boundaries.** Don't update index per-connection-strengthen during a parallel dispatch. Each neuron emits index-update events as part of its per-neuron result; the thalamus applies them at the **orchestration boundary** that immediately follows the dispatch (between the parallel wave and the next orchestration step). This means the reuse lookup in Phase 2 sees the index reflecting this frame's deltas — no stale-index lag, no "defer to next frame" workaround. The pattern mirrors temporal: parallel-per-neuron work inside the dispatch, cross-neuron consolidation (allocate, route, index-update) sequentially between dispatches.
- Memory cost: index roughly doubles the connection-graph memory footprint. Acceptable for now; revisit if it becomes a bottleneck.
- Storage layout: flat `(target,distance) → sources` vs nested. Decide based on benchmark.
- Persistence: not serialized; rebuilt on load. See [spatial-processing](./spatial-processing.md) Phase 7.

---

### Phase 2 — Reuse Lookup in Correction Path

**Goal:** Before minting a correction neuron, query the reverse index for an existing neuron whose inference signature partially matches the required correction. If match score ≥ the existing merge threshold, wire reuse; otherwise mint. Reuse is always on — no enable flag.

#### Code touched

- `brain/brain-core/src/thalamus.rs` — new method `find_reusable(observed_targets, distance) -> Option<NeuronId>`:
  1. For each target T in observed_targets, query `inference_index[T][distance]` → set of candidate source neurons.
  2. Compute candidate score: `|candidate.connections ∩ observed_targets| / |observed_targets|` (or use existing common/missing/novel scoring from pattern matching).
  3. Filter candidates ≥ the existing global merge threshold (same one used by partial-context pattern recognition).
  4. Return best-scoring candidate, or None.
- `brain/brain-core/src/thalamus.rs::evaluate_*_predictions` correction path — for each error, call `find_reusable` first. If `Some(N)`: wire erroring neuron's routing entry to N, add N to `correction_wired_this_frame` (and to `fired_this_frame` for learning). If `None`: mint new neuron (existing path).
- Reuse applies to both phases:
  - In `process_temporal`, d>0 errors are followed by a reuse lookup before mint.
  - In `process_spatial`, d=0 errors are followed by a reuse lookup before mint.
  Each phase batches its own lookups; the inference index is shared.

#### Dispatch shape

```
errors = collect_errors_from_phase_stabilization()
candidate_queries = errors.map(|e| (e.observed_targets, e.distance))
candidate_results = region.query_inference_sources_batch(candidate_queries)   // parallel
for (error, candidates) in errors.zip(candidate_results):
    if let Some(reuse) = score_and_pick(candidates, error, merge_threshold):
        wire_reuse(error.erroring_neuron, reuse)
    else:
        mint_new(error)
```

#### Acceptance

- Unit test: train two errors with overlapping observed sets — second error reuses the first's correction neuron when overlap ≥ merge threshold.
- MNIST: rerun the spatial-processing [Phase 5](./spatial-processing.md#phase-5--mnist-single-frame-harness) harness with reuse enabled. Total neuron count should be significantly lower (target: ≥30% reduction). Accuracy should be ≥ Phase 5 baseline.
- Profile: reuse lookup adds < 20% to per-frame runtime.

#### Notes / gotchas

- Reuse applies at both distances — d>0 in `process_temporal`, d=0 in `process_spatial`. Same mechanism, different phase.
- Reuse reads the existing global merge threshold from brain options — no new parameter. Setting that threshold to 1.0 disables reuse (and also disables partial-context pattern recognition); the two behaviors are intentionally coupled.
- Edge case: erroring neuron N's reuse candidate is N itself. Filter self-matches.
- Edge case: reuse candidate is a neuron minted earlier *in the same phase, same frame*. Allow it. The reused neuron goes into `correction_wired_this_frame` as usual; the index update at the orchestration boundary keeps the reverse index consistent.
- Termination: reused neurons go into `correction_wired_this_frame` (per [spatial-processing](./spatial-processing.md#34-refractory-and-correction-wiring-inhibition)), so they cannot generate fresh errors this phase. This is the termination guarantee — refractory + correction-wired-inhibition bounds the phase regardless of reuse activity.

---

### Phase 3 — Reuse Validation

- Run MNIST with reuse: report neuron count, accuracy, transfer effect (train 0-4, eval on 5-9).
- Validate that the transfer effect appears — neurons learned for early digits get reused (at least partially) by later digits, producing faster convergence and/or better generalization on the second batch.

#### Acceptance

- Transfer effect detectable: training on digits 0-4 followed by 5-9 shows measurably better accuracy on 5-9 (relative to training 5-9 from scratch) due to neuron reuse.
- Neuron count growth on 5-9 is sub-linear vs digits-from-scratch.

---

### Phase 4 — Stocks Integration with Full Pipeline

**Goal:** Run the full pipeline (spatial + temporal + reuse) on stocks. This is distinct from the spatial-only stocks integration in [spatial-processing](./spatial-processing.md#phase-6--stocks-integration-spatial-only) — that established a spatial baseline; this measures the additional impact of reuse.

#### Steps

- Run stocks with `process_spatial` enabled and reuse on. d=0 connections form across co-occurring top-level patterns within each frame. Spatial corrections from one frame feed into the temporal phase in subsequent frames, building spatio-temporal abstractions.
- Compare per-episode ROI and directional accuracy against the [spatial-only baseline](./spatial-processing.md#phase-6--stocks-integration-spatial-only) and the [inference-level winner](./inference-level.md) baseline.
- Tune the global merge threshold and d=0 error threshold if needed.

#### Acceptance

- Directional accuracy improves over both prior baselines. Target: lift off the historical 57-59% plateau.
- Neuron count is significantly lower than spatial-only-on-stocks (reuse working as intended).

---

### Phase 5 — Forget-Rate & Class-Neuron Generalization

**Goal:** Validate and tune the generalization path from specific correction neurons to abstract class neurons.

- Long-running training (10k+ frames). Monitor:
  - Distribution of d=0 connection strengths over time per correction neuron.
  - Whether reuse counts per neuron rise (good — class behavior).
  - Whether action bindings on heavily-reused correction neurons stay correct.
- Tune the global static forget rate if needed.

#### Open

- When a correction neuron is heavily reused, action votes may need normalization. Revisit if Phase 5 shows action-binding dilution.

---

## 6. Risk Assessment

### 6.1 High Confidence

Reuse is correct in principle. The reverse-inference-index lookup is a straightforward extension of existing index machinery. The `correction_wired_this_frame` inhibition rule (see [spatial-processing](./spatial-processing.md#34-refractory-and-correction-wiring-inhibition)) makes reuse safe under within-frame error cascades.

### 6.2 Key Risks

- **Reverse-index cost**: per-frame reuse lookup could dominate runtime if poorly indexed. Mitigation: batch one lookup per error per phase; shard by column for parallel evaluation across regions; apply index updates at orchestration boundaries so lookups always see fresh data.
- **Over-aggressive reuse**: too-low merge threshold causes inappropriate reuse, polluting reused neurons with mismatched contexts. Mitigation: the shared global merge threshold means tuning is coupled to pattern recognition; default position is "tune cautiously and rely on decay to clean up bad reuses."
- **Action binding dilution**: heavily-reused correction neurons may have action votes diluted across many contexts. Mitigation: Phase 5 monitors this; normalization can be added if observed.
