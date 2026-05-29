# Moment Neurons v2 — Implementation Plan

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Design of record:** [moment-neurons-v2.md](./moment-neurons-v2.md)
**Status:** Pre-implementation. Use this as the working checklist next session.

---

## 0. Scope & Ordering

This plan implements the design in v2 in five sequential phases. Each phase ends with a runnable validation. Earlier phases must pass their validation before later phases begin.

| Phase | Goal | Validation gate |
|---|---|---|
| 1A | Wavefront dispatch infrastructure (no behavior change for d>0) | Stocks regression: directional accuracy ≥ current baseline |
| 1B | d=0 connection learning | Unit test: co-active neurons mutually strengthen d=0 |
| 1C | d=0 voting and predictions | Unit test: active neuron emits d=0 votes |
| 1D | d=0 error detection + correction neuron minting (no reuse yet) | MNIST single-frame harness produces correction neurons on conflicting digits |
| 1E | MNIST validation, mint-only | >50% accuracy on test set with <1000 training images |
| 2A | Reverse inference index | Unit test: target → set of source-neurons lookup |
| 2B | Reuse lookup in correction path | MNIST: neuron count drops vs Phase 1 baseline |
| 2C | Reuse validation | Transfer test: training on digits 0-4 helps with 5-9 |
| 3 | Stocks integration | Directional accuracy lifts off the 57-59% plateau |
| 4 | Forget-rate / class-neuron tuning | Class neurons survive 10k+ frames with action binding intact |

Estimated wall-clock: 2-3 weeks of focused work for Phases 1 + 2. Phases 3 + 4 are post-MNIST and may slip without blocking the MNIST milestone.

---

## Phase 1A — Wavefront Dispatch Infrastructure

**Goal:** Replace the level-sweep loop with wavefront propagation while preserving exact behavior for the stocks workload (d>0 only). No d=0 work yet. This is a pure refactor with a regression gate.

### Code touched

- `brain/brain-core/src/brain.rs` — replace `process_levels()` loop. Today's loop iterates `for level in 0..max_level` and calls thalamus per level. New loop: drain wavefront queue until empty.
- `brain/brain-core/src/memory.rs` — add wavefront state. Retain `level_index` exactly as is — it stays the spatial organization for co-activation scoping and forget rates.
- `brain/brain-core/src/thalamus.rs` — rename `process_level(level, …)` to `process_neurons(batch, …)`. Signature changes from "give me this level's active set" to "give me this wave's pending activations." Internal logic largely unchanged.
- `brain/brain-core/src/neuron.rs` — `neuron.level` field stays (still used for forget-rate lookup and diagnostics) but is no longer a routing axis.

### New state

```rust
// brain.rs, per-frame state during processFrame:
fired_this_frame: FxHashSet<NeuronId>   // refractory tracking
pending_activation: Vec<NeuronId>       // wavefront queue
```

### Per-frame pipeline (new)

1. Build frame from quantized inputs (unchanged).
2. Create sensory neurons (Op-1, unchanged).
3. Clean dead patterns (Op-2, unchanged).
4. Age context window (unchanged).
5. Push sensory neurons into `pending_activation`.
6. **Wavefront loop:**
   ```
   while pending_activation is non-empty:
       wave = drain(pending_activation)
       wave.retain(|n| !fired_this_frame.contains(n))
       fired_this_frame.extend(wave)
       results = region.dispatch_process_neurons(wave)   // parallel, per-column
       new_activations = results.flat_map(|r| r.activations)
       new_neurons     = results.flat_map(|r| r.new_neuron_specs)
       thalamus.apply_results(results)                   // routing-table updates, etc.
       pending_activation.extend(new_activations + new_neurons)
   ```
7. Action voting over `fired_this_frame` (single pass, post-stabilization).
8. Error tracking, decay, etc.

### Phase 1A acceptance

- Stocks pipeline (`apps/stocks`) runs and produces directional accuracy within ±1% of current baseline.
- No new neurons created in Phase 1A relative to current code (this is a pure refactor; behavior should match).
- Wavefront converges in ≤ current max_level waves on stocks data (since no d=0 connections exist yet, waves should be one-per-level equivalent).

### Notes / gotchas

- Parallelism: dispatch within each wave fans out across columns via existing Region/Rayon machinery. No within-wave ordering guarantee — design §3.6 removed the determinism requirement, so we don't sort by NeuronId.
- Action voting must move from "per level" to "once at end." Today voting accumulates per level via `level_age_state`. Refactor to single pass over `fired_this_frame`.
- `level_age_state` is still needed for d>0 next-frame error correction — don't delete, just feed it from `fired_this_frame` partitioned by neuron level.

---

## Phase 1B — d=0 Connection Learning

**Goal:** When co-active neurons fire at the same level in the same frame, mutually strengthen their d=0 connections.

### Code touched

- `brain/brain-core/src/neuron.rs::learn_connections` — currently iterates ages 1..N. Add age 0: for each neuron N active at age 0 (i.e., fired this frame), strengthen d=0 connections to all other neurons active at age 0 at the same level.
- `brain/brain-core/src/neuron.rs::connections` field — already a `Vec<FxHashMap<NeuronId, ConnectionData>>` with slot 0 reserved/unused. Populate slot 0 from now on.

### Scoping rule

A neuron's d=0 connections only target other neurons **at the same level**. Cross-level d=0 is forbidden (cross-level coupling happens via reuse, Phase 2, not via raw d=0 connections). This keeps the d=0 learning loop bounded.

### Phase 1B acceptance

- Unit test: feed a frame where neurons {A, B, C} at level 0 are co-active. After processing, A's d=0 map contains B and C with positive strength; symmetric for B and C.
- Repeating the same frame strengthens existing entries (not creates duplicates).
- Stocks regression: same accuracy as Phase 1A. d=0 connections build but no errors are detected yet (1D), so no behavior change in output.

### Notes / gotchas

- Reward propagation: existing `learn_connections` updates `conn.reward` per connection. d=0 should follow the same rule — if a reward was tagged on the frame, it propagates to d=0 connections too.
- Be conservative on strength scaling: d=0 has up to (N-1)² total connections per frame at level L (where N is per-level firing count). For a 49-pixel MNIST level-0 frame, that's ~2400 connections strengthened per frame. Profile.

---

## Phase 1C — d=0 Voting / Prediction

**Goal:** When a neuron fires during the wavefront, emit its d=0 connections as predicted co-activations into a per-level "predicted set" accumulator. No errors detected yet.

### Code touched

- `brain/brain-core/src/neuron.rs::vote` — currently returns connections at `age+1`. Add a `vote_d0` variant (or a flag on `vote`) returning d=0 connections when the neuron fires.
- `brain/brain-core/src/thalamus.rs` — collect d=0 votes into per-level `predicted_d0: FxHashMap<Level, FxHashMap<NeuronId, FxHashMap<NeuronId, Strength>>>`. Outer key = level, middle key = predicting neuron, inner = predicted target → strength.
- `brain/brain-core/src/brain.rs` — d=0 votes accumulate across waves within a frame; reset at frame end.

### Phase 1C acceptance

- Unit test: neuron N with d=0 connections to {A, B} fires; thalamus's `predicted_d0[level][N]` contains A and B with the recorded strengths.
- Stocks regression: same accuracy (predictions accumulate but aren't yet used for errors).

---

## Phase 1D — d=0 Error Detection + Correction Minting

**Goal:** At wavefront stabilization, evaluate every fired neuron's d=0 predictions against reality (union of fired neurons at same level), generate errors above threshold, mint correction neurons. No reuse yet — every error mints a new neuron.

### Code touched

- `brain/brain-core/src/thalamus.rs` — add `evaluate_d0_predictions(level_fired_sets)` called once per frame after wavefront stabilization, before action voting.
  - For each predicting neuron N at level L:
    - Predicted set = `predicted_d0[L][N]`
    - Observed set = `fired_this_frame ∩ neurons_at_level[L]`
    - Compute mismatch using existing common/missing/novel logic
    - If mismatch > error threshold → record error
- `brain/brain-core/src/thalamus.rs::allocate_pattern_neuron` — extend to accept d=0 specs. Connection specs carry distance=0; the new neuron's connections vec slot 0 is populated from the observed co-activation set.
- `brain/brain-core/src/neuron.rs::correct_errors` (routing table update) — accept correction entries with d=0 context. The erroring neuron's routing table now has an entry: `context_at_d0(observed_set) → activate(correction_neuron)`.

### Order of operations in `processFrame`

1. Wavefront drains (1A).
2. d=0 predictions accumulated as neurons fire (1C).
3. d>0 next-frame errors evaluated (existing).
4. **NEW:** d=0 errors evaluated against `fired_this_frame` partitioned by level.
5. Op-4 batch creates new neurons (now includes d=0 corrections).
6. Routing tables updated (now includes d=0 context entries).
7. Action voting over `fired_this_frame`.

### Phase 1D acceptance

- Unit test: train "digit-1-like" co-activation {A, B, C} across 10 frames → no errors after exposure 2-3. Switch to "digit-7-like" {A, D, E, F, G} → A errors on its d=0 predictions (predicted B, C; observed D, E, F, G), correction neuron minted with d=0 connections to {A, D, E, F, G}.
- Inspect: correction neuron's routing table is wired into A's routing under d=0 context.
- Stocks regression: ≤ 5% accuracy delta. d=0 errors may produce new neurons in stocks too; that's expected, not a regression.

### Notes / gotchas

- Error threshold mode: use the existing `errorCorrectionMode` (static / conservative / neutral / aggressive) — same parameter for d=0 and d>0 in Phase 1. Tunable separately later if needed.
- A correction neuron minted in this frame does NOT fire this frame. It's available starting next frame. Refractory + the fact that we mint after stabilization guarantee this.
- Correction neurons inherit the level of the erroring neuron.

---

## Phase 1E — MNIST Single-Frame Harness

**Goal:** Validate the design on MNIST with the simplified loop.

### Code touched

- `apps/mnist/test.js` — rewrite training loop:
  ```js
  for each training image:
      const inputs = encoder.encodeImage(image)
      const actions = encoder.encodeAction(label)
      const rewards = encoder.buildRewards(label)
      brain.processFrame(inputs, EMPTY_MAP, EMPTY_MAP)   // single frame, no actions
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
- `apps/mnist/test.js` — `resetContext()` between images is harmless but redundant (no temporal context being built). Keep it for safety.
- Brain config: temporal context length can be set to 1 (or kept at default — doesn't matter, no temporal connections form on single-frame MNIST).

### Phase 1E acceptance

- Train on 1000 images (100 per digit, balanced), eval on 200 held-out → >50% accuracy.
- Inspect neuron counts: should be ≪ what cortex-only produced.
- Wavefront depth: log mean/p99 waves per frame. Should be small (≤10 expected).
- Per-image processing time: should be O(seconds) max, not O(minutes) as in cortex-only.

### If 1E fails

Most likely failure modes and diagnostics:

- **All-or-nothing memorization** (single huge correction neuron per digit): probe the merge threshold / error threshold; check `mnist d0 correction neuron sizes` distribution.
- **No correction neurons at all**: error threshold too lax, or d=0 connections strengthen too slowly. Lower threshold, increase initial connection strength.
- **Wavefront not terminating**: shouldn't be possible by 11.1 reasoning, but if it happens, log queue size per wave to find the source.

---

## Phase 2A — Reverse Inference Index

**Goal:** Build the data structure that answers "which neurons have a connection to target T at distance d?"

### Code touched

- `brain/brain-core/src/column.rs` or `thalamus.rs` — add `inference_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>`. Maps `target → distance → set of sources`. Sharded by the column owning the **source** neuron, so each column owns the index for its resident neurons.
- `brain/brain-core/src/neuron.rs` — on connection create/strengthen/decay/delete, emit an `IndexUpdate` event. Column applies the update to its local index.
- `brain/brain-core/src/region.rs` — expose a `query_inference_sources(targets: &[NeuronId], distance: Distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>` op that fans out across columns and merges per-target source sets.

### Phase 2A acceptance

- Unit test: create neurons N1→T1 and N2→T1 at distance 0; query for T1 at d=0 returns {N1, N2}.
- Decay a connection below threshold; index reflects removal.
- Index size stays proportional to total connection count (no bloat).

### Notes / gotchas

- Update batching: don't update index per-connection-strengthen during a frame. Batch updates and apply at frame end (or at op-4 boundary). Per-frame strengthens are the dominant connection operation; batching is essential.
- Memory cost: index roughly doubles the connection-graph memory footprint. Acceptable for now; revisit if it becomes a bottleneck.

---

## Phase 2B — Reuse Lookup in Correction Path

**Goal:** Before minting a correction neuron, query the reverse index for an existing neuron whose inference signature partially matches the required correction. If match score ≥ the existing merge threshold (shared with partial-context recognition), wire reuse; otherwise mint. Reuse is always on — no enable flag.

### Code touched

- `brain/brain-core/src/thalamus.rs` — new method `find_reusable(observed_targets, distance) -> Option<NeuronId>`:
  1. For each target T in observed_targets, query `inference_index[T][distance]` → set of candidate source neurons.
  2. Compute candidate score: `|candidate.connections ∩ observed_targets| / |observed_targets|` (or use existing common/missing/novel scoring from pattern matching).
  3. Filter candidates ≥ the existing global merge threshold (same one used by partial-context pattern recognition).
  4. Return best-scoring candidate, or None.
- `brain/brain-core/src/thalamus.rs::evaluate_*_predictions` correction path — for each error, call `find_reusable` first. If Some(N): wire erroring neuron's routing entry to N. If None: mint new neuron (existing path).
- Batch the lookup once per frame after wavefront stabilization, not per error (single pass over all errors collected this frame).

### Dispatch shape

```
errors = collect_errors_from_wave_stabilization()
candidate_queries = errors.map(|e| (e.observed_targets, e.distance))
candidate_results = region.query_inference_sources_batch(candidate_queries)   // parallel
for (error, candidates) in errors.zip(candidate_results):
    if let Some(reuse) = score_and_pick(candidates, error, merge_threshold):
        wire_reuse(error.erroring_neuron, reuse)
    else:
        mint_new(error)
```

### Phase 2B acceptance

- Unit test: train two errors with overlapping observed sets — second error reuses the first's correction neuron when overlap ≥ merge threshold.
- MNIST: rerun 1E with reuse enabled. Total neuron count should be significantly lower (target: ≥30% reduction). Accuracy should be ≥ Phase 1E.
- Profile: reuse lookup adds < 20% to per-frame runtime.

### Notes / gotchas

- Reuse applies to **all distances**, not just d=0. A d>0 temporal error should also try reuse against the reverse index at that distance.
- Reuse reads the existing global merge threshold from brain options — no new parameter. Setting that threshold to 1.0 disables reuse (and also disables partial-context pattern recognition); the two behaviors are intentionally coupled.
- Edge case: erroring neuron N's reuse candidate is N itself. Filter self-matches.
- Edge case: reuse candidate is a neuron created earlier this frame (same frame as the error). Allow it; the index update at frame end will catch up next frame, but this frame's correction can still reuse newly-minted neurons if they happen to be in `pending_activation`.

---

## Phase 2C — Reuse Validation

- Run MNIST with reuse: report neuron count, accuracy, transfer effect (train 0-4, eval on 5-9).
- Stocks regression with reuse: directional accuracy should match or exceed Phase 1A baseline.

---

## Phase 3 — Stocks Integration

### 3.0 Branch Hygiene (Pre-Phase Gate)

Before touching stocks, audit the `mnist` branch and merge only what's needed into `main`.

- List every file changed on `mnist` vs `main`. Classify each as:
  - **Permanent** — core brain changes (wavefront, d=0, reuse, reverse index). Merge to `main`.
  - **MNIST app** — encoder updates, harness, jobs. Merge if reusable; otherwise keep on branch.
  - **Experimental scaffolding** — debug prints, throwaway harnesses, parameter sweeps. Leave on branch or delete.
- Verify stocks regression still passes on the merged `main` (Phase 1A gate replayed post-merge).
- Tag the pre-merge commit on `mnist` so MNIST validation results remain reproducible from the branch state.

### 3.1 Stocks Workload

- Run stocks with d=0 + reuse enabled. d=0 connections will form across symbols within each frame (co-occurring market signals).
- Compare per-episode ROI and directional accuracy against the temporal-only baseline.
- Tune the global merge threshold and d=0 error threshold if needed.

---

## Phase 4 — Forget-Rate & Class-Neuron Generalization

- Long-running training (10k+ frames). Monitor:
  - Distribution of d=0 connection strengths over time per correction neuron.
  - Whether reuse counts per neuron rise (good — class behavior).
  - Whether action bindings on heavily-reused correction neurons stay correct.
- Tune `d0ForgetRate` if d=0 decay differs meaningfully from d>0 in practice. Default: same as the neuron's level-based d>0 forget rate.

---

## Brain Options Added

No new brain options. Reuse is always on. Both reuse and partial-context pattern recognition read the existing global merge threshold — they are intentionally coupled. d=0 error threshold and d=0 forget rate default to the existing per-level settings used for d>0.

No breaking changes to NAPI surface. `processFrame`, `learn`, `infer` signatures unchanged.

---

## Open Items Carried Into Implementation

These were left open in the design and may surface as concrete decisions during coding:

1. **Inference-index storage layout** (§8.3): flat `(target,distance) → sources` vs nested. Decide in 2A based on benchmark.
2. **Update batching cadence** for the inference index: per-frame vs per-op-4. Decide in 2A based on profile.
3. **d=0 forget rate anchor** (§8.5): start with "same as level-based d>0." Revisit in Phase 4.
4. **Action binding through reuse**: when a correction neuron is heavily reused, action votes may need normalization. Revisit if Phase 4 shows dilution.

---

## Test Harness Additions

- `brain/brain-core/tests/d0_learning.rs` — Phase 1B unit tests.
- `brain/brain-core/tests/d0_errors.rs` — Phase 1D unit tests.
- `brain/brain-core/tests/wavefront.rs` — Phase 1A wavefront convergence + refractory tests.
- `brain/brain-core/tests/reuse.rs` — Phase 2B reuse decision tests.
- `apps/mnist/jobs/single-frame-eval.js` — Phase 1E + 2C harness.
- `apps/mnist/jobs/transfer.js` — Phase 2C transfer test (digits 0-4 → 5-9).