# Moment Neurons v2 — Implementation Plan

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Design of record:** [moment-neurons-v2.md](./moment-neurons-v2.md)
**Status:** Pre-implementation. Use this as the working checklist next session.

---

## 0. Scope & Ordering

This plan implements the design in v2 in sequential phases. Each phase ends with a runnable validation. Earlier phases must pass their validation before later phases begin.

| Phase | Goal | Validation gate |
|---|---|---|
| 1 | Merge `mnist` branch into `dev` | Stocks regression on merged `dev` ≥ current main baseline |
| 2 | Inference scope experiment (d>0 only, on stocks) | Pick winner among `base` / `same-level` / `all-levels` |
| 3 | Spatial-phase scaffolding + intrinsic-level removal | Stocks regression: bit-exact vs Phase 2 winner |
| 4 | d=0 connection learning (per-neuron, parallel) | Unit test: co-active neurons mutually strengthen d=0 |
| 5 | Spatial wavefront orchestration | Unit test: wavefront terminates, predictions accumulate |
| 6 | d=0 error detection + correction neuron minting (no reuse yet) | MNIST single-frame harness produces correction neurons on conflicting digits |
| 7 | MNIST single-frame harness, mint-only | >50% accuracy on test set with <1000 training images |
| 8 | Reverse inference index | Unit test: target → set of source-neurons lookup |
| 9 | Reuse lookup in correction path | MNIST: neuron count drops vs Phase 7 baseline |
| 10 | Reuse validation | Transfer test: training on digits 0-4 helps with 5-9 |
| 11 | Stocks integration | Directional accuracy lifts off the 57-59% plateau |
| 12 | Forget-rate / class-neuron tuning | Class neurons survive 10k+ frames with action binding intact |
| 13 | Persistence / backup / import-export updates | Snapshot/restore round-trips d=0 connections and dynamic-level corrections |

Phases 1 and 2 are prerequisites that unblock everything downstream:
- **Phase 1** clears the long-running `mnist` branch debt before any new structural work lands on top of it.
- **Phase 2** picks the inference-scope rule for d>0 on a known-good workload (stocks). The winning rule then propagates to d=0 in Phase 4+. Without this experiment, we'd be guessing the scope rule when we lock in d=0 semantics.

Estimated wall-clock: 2-3 weeks of focused work for Phases 3-10 after Phases 1-2 land. Phases 11-13 are post-MNIST and may slip without blocking the MNIST milestone.

---

## Phase 1 — Branch Reconciliation (mnist → dev)

**Goal:** Land the keepable parts of the `mnist` branch onto `dev` so subsequent phases have a clean base. The `mnist` branch has accumulated substantial work alongside v1 moment-neuron experiments; merging late would compound conflicts with the v2 work.

### Steps

1. List every file changed on `mnist` vs `main`. Classify each:
   - **Permanent — core** — brain/runtime changes that should ship regardless of v2 (e.g. bug fixes, perf improvements, infra cleanup). Merge to `dev`.
   - **Permanent — apps** — MNIST encoder, harness, jobs that will still be useful for Phase 7. Merge to `dev`.
   - **Experimental v1** — moment-neuron v1 scaffolding superseded by v2. Leave on `mnist`; do not merge.
   - **Throwaway** — debug prints, parameter-sweep scripts, ad-hoc harnesses. Leave on `mnist` or delete.
2. For every "Permanent — core" candidate, assess impact on the stocks pipeline before merging. Anything that changes stocks behavior gets called out explicitly so we know what the post-merge baseline is.
3. Open the merge into `dev` as a reviewable PR with the classification table in the description.
4. Tag the pre-merge tip of `mnist` (e.g. `mnist-v1-final`) so v1 MNIST experiments stay reproducible.

### Known permanent-core change to land here

- **Static forget rate.** Replace the existing level-dependent forget rate with a single global static forget rate. Experimentation on the `mnist` branch found no meaningful accuracy difference between level-dependent and static decay, and static is simpler and more biologically defensible (no reason higher-level neurons should follow different decay rules). This drops one parameter and removes a level-dependency that would otherwise complicate Phase 3's removal of `neuron.level`.
5. After merge, run the stocks pipeline on `dev` and record the new baseline (this is what Phase 2 measures against).

### Phase 1 acceptance

- Stocks directional accuracy on merged `dev` ≥ current `main` baseline (within ±1% noise).
- Brain unit tests pass on `dev`.
- Classification table is committed alongside the merge so future readers know what intentionally stayed on the branch.

### Notes

- All subsequent phases happen on `dev` or branches off `dev`. `mnist` becomes archival.
- If the merge surfaces work that's worth keeping but risky, land it as a follow-up PR after Phase 1 closes — keep the initial merge focused on no-regression changes.

---

## Phase 2 — Inference Scope Experiment (d>0 only, stocks)

**Goal:** Choose the inference scope rule that the rest of the v2 work commits to. Tested on stocks (d>0 only) because that's the workload we have a strong baseline for. The chosen rule then applies to d=0 in Phase 4+ uniformly.

### Background

Today, `thalamus.dispatch_frame` builds `new_active_neurons` from `age0`, which is the L=0 sensory set. Every neuron at every level therefore forms d>0 connections targeting only L=0. We want to know whether that's actually optimal vs two alternatives:

| Rule | `new_active_neurons` passed to a level-L neuron processing age=d | Hypothesis |
|---|---|---|
| `base` (today) | All L=0 neurons active at age=d | Cheapest; current behavior |
| `same-level` | L-resident neurons active at age=d | Cleaner hierarchy; each level predicts within its own abstraction |
| `all-levels` | All neurons active at age=d, any level | Maximum reuse potential; biologically closer to cortex |

### Code touched

- `brain/brain-core/src/brain.rs` — add `inference_scope: InferenceScope` to `BrainOptions`, defaulting to `Base` (zero behavior change).
- `brain/brain-core/src/types.rs` — add `enum InferenceScope { Base, SameLevel, AllLevels }`.
- `brain/brain-core/src/thalamus.rs::dispatch_frame` — branch on `inference_scope` when constructing `new_active_neurons`:
  - `Base`: existing logic (active at age=0 sensory).
  - `SameLevel`: filter `age0` (and the broader active set fed in via `age_states`) to neurons whose `level == current_processing_level`. Requires plumbing the level into the active-set builder.
  - `AllLevels`: include every neuron active at the relevant age, regardless of level. Requires extending `age0`-style state to cover non-sensory neurons (or pulling from the existing memory level index).
- `brain/brain-napi/src/lib.rs` — expose `inferenceScope` on the JS-facing brain options.

### Runs

1. `base` — control. Confirm reproduces current stocks baseline.
2. `same-level` — Variant A.
3. `all-levels` — Variant B.

Same data, same seed, same hyperparameters across all three.

### Metrics

- Directional accuracy.
- Per-episode ROI.
- Total neuron count at end of run.
- Total connection count at end of run.
- Per-frame runtime (mean, p99).
- Memory footprint at end of run.

### Phase 2 acceptance

- All three runs complete without crashing on the stocks workload.
- A written decision lands in this doc (and §3.4 of the design doc) naming the winner and the metric that drove the call.
- Loser variants remain runnable via config — we don't delete them; they may be revisited if MNIST or another workload prefers a different rule.

### Notes / gotchas

- **No wavefront refactor in this phase.** The level-sweep loop stays. Only `new_active_neurons` construction changes.
- **No d=0 work in this phase.** Strictly d>0.
- `SameLevel` filtering needs a per-level active set; the existing code already partitions by level via `memory.get_level_neurons(level)`. Use that.
- `AllLevels` may inflate per-frame runtime substantially on stocks — that's a data point, not a failure. Measure first, optimize later if it wins.
- If `base` wins outright, we still proceed to Phase 3 with `base` semantics for d>0 and design d=0 around it. The experiment is informative either way.
- The winning rule's name shows up in the §3.4 edit of the design doc and in Phase 4's "scoping" decision.

---

## Phase 3 — Spatial-Phase Scaffolding + Intrinsic-Level Removal

**Goal:** Two structural changes that need to land together because they touch the same dispatch path:

1. Set up the spatial-first two-phase pipeline: add an empty `process_spatial()` before the renamed `process_temporal()`. Plumb `fired_this_frame` and `correction_wired_this_frame`. Route action voting through the combined fired set.
2. Remove `neuron.level` as an intrinsic field. Level becomes a property of *activations in active memory*, not of neurons. The level-sweep iterates active-memory levels: "process whoever's active at level 0, then 1, then 2, …" The activation level is determined by the routing source (one above the activating neuron), so a reused neuron can appear at different activation levels in different frames without contradiction.

**No d=0 connection learning, no d=0 errors, no spatial wavefront body yet** — Phase 3 is structural scaffolding. The temporal pipeline's pattern-matching internals stay as-is; only the dispatch lookup and the per-neuron level field change.

### Code touched

- `brain/brain-core/src/brain.rs`:
  - Rename `process_levels()` → `process_temporal()`. Internals unchanged.
  - Add `process_spatial()` — empty stub for now, with a `// Phase 5 body lands here` marker.
  - Per-frame pipeline now calls `process_spatial()` **then** `process_temporal()` sequentially.
  - Add per-frame state `fired_this_frame: FxHashSet<NeuronId>` and `correction_wired_this_frame: FxHashSet<NeuronId>`. Both reset at frame start.
  - Both phases populate `fired_this_frame` as neurons activate.
  - Action voting reads `fired_this_frame \ correction_wired_this_frame` instead of per-level accumulation. With `process_spatial` empty and `correction_wired_this_frame` empty, behaviorally identical to today.
- `brain/brain-core/src/neuron.rs`:
  - **Remove `neuron.level` field.** Drop from struct, drop from serialization, drop from all per-neuron API surface.
  - No need for a per-neuron forget-rate field — Phase 1 made forget rate a global static.
- `brain/brain-core/src/memory.rs`:
  - `level_index` becomes per-frame active-memory state, not persistent metadata. Reconstructed each frame from activations.
  - `get_level_neurons(level)` returns "neurons activated at this level this frame" rather than "neurons whose intrinsic level == L." Activation level is recorded when a neuron is added to active memory via `activate_pattern(pattern_id, level, ...)`.
  - For sensory neurons activated at the start of a frame, the activation level is 0 (unchanged from today).
  - For newly-minted correction neurons (d>0 and d=0), the activation level this frame is `activating_neuron.activation_level + 1`. This replaces the old "neuron's intrinsic level + 1" convention.
- `brain/brain-core/src/thalamus.rs`:
  - No signature changes for `process_level` (still called from `process_temporal`). The `level` arg now means "current iteration level," consumed when reading active memory.
  - Add `correction_wired_this_frame` as a thalamus-owned field (mutated when corrections are wired; cleared per frame).
  - Existing d>0 correction path adds newly-minted/reused targets to `correction_wired_this_frame` (matters in Phase 9; no-op for plain mints).
- `brain/brain-core/src/column.rs`, `region.rs` — sweep for any code that references `neuron.level` and replace with activation-level lookup or remove if vestigial.

### Per-frame pipeline (new shape)

1. Build frame from quantized inputs (unchanged).
2. Create sensory neurons (Op-1, unchanged).
3. Clean dead patterns (Op-2, unchanged).
4. Age context window (unchanged).
5. Reset `fired_this_frame` and `correction_wired_this_frame`.
6. **`process_spatial()`** — no-op stub.
7. **`process_temporal()`** — existing level-sweep, identical behavior. Populates `fired_this_frame` as a side-effect.
8. Action voting over `fired_this_frame \ correction_wired_this_frame`.
9. Error tracking, decay, etc.

### Phase 3 acceptance

- Stocks pipeline output is **bit-exact** vs the Phase 2 winner on a fixed seed. Despite the substantial internal refactor (intrinsic level removed, active-memory-level dispatch, two-phase pipeline), the inputs to each neuron's per-frame work and the outputs they emit are unchanged when `process_spatial` is empty and `correction_wired_this_frame` is empty. Drift would indicate a refactor bug. (See "Voting equivalence" below.)
- Zero new neurons created relative to the Phase 2 baseline.
- `process_spatial()` is reached every frame, runs in <1µs (empty stub), produces no side effects.
- `neuron.level` field gone from the struct and from serialized form. Loading a pre-Phase-3 snapshot either migrates or errors clearly (decide during implementation; lean toward error + version bump).

### Voting equivalence

Voting moves from per-level accumulation (today, via `level_age_state`) to a single pass over `fired_this_frame \ correction_wired_this_frame`. In the parallel-per-neuron model, voting is "each neuron emits its votes; thalamus aggregates" — today aggregation happens per-level, in Phase 3 it happens once over the union. With `correction_wired_this_frame` empty and the same neurons firing in both code paths, the aggregated vote set is identical by construction. The bit-exact gate above covers this; no separate unit test required.

### Notes / gotchas

- The temporal level-sweep keeps its existing per-iteration dispatch and parallelism. What changes is the lookup at the top of each iteration: "who's active at level L?" now reads from active-memory rather than from intrinsic-level-indexed storage. The processing math inside `process_level` is unchanged.
- Activation-level assignment is "activating neuron's activation level + 1." Sensory neurons are at activation level 0. A reused neuron R activated from A (at level 2) appears at level 3 this frame, even if R was originally minted at a different level for a different context. This is the design point that resolves the cross-level reuse problem.
- `correction_wired_this_frame` exists but stays empty in Phase 3. Plumbing it now sets up Phase 9's reuse path cleanly.
- The renamed `process_temporal` keeps its existing public signature where exposed; only internal call sites change.
- Spatial-first ordering matters even with an empty stub — that's the order we're committing to. Don't let the empty body tempt anyone to flip the order "for now."
- Backup format version bumps in Phase 3 (intrinsic level removal) and again in Phase 13 (d=0 connection round-trip). Worth deciding now whether to consolidate to a single v2 format or accept two version bumps.

---

## Phase 4 — d=0 Connection Learning

**Goal:** During `process_spatial`'s wavefront (implemented in Phase 5), each neuron learns its own d=0 connections in parallel — strengthening edges from itself to the neurons co-active in this spatial phase. Phase 4 implements the per-neuron learning rule; Phase 5 implements the wavefront orchestration that drives it.

### Parallel-per-neuron model

Spatial processing mirrors temporal: each neuron does its own work in parallel across columns, and the thalamus orchestrates cross-neuron operations at orchestration boundaries between dispatch waves. Per neuron, in parallel, the spatial frame does:

- Match own d=0 routing entries against the co-activation set captured at dispatch time; emit activations for matches.
- Emit own d=0 predictions (the d=0 connection set) for post-wavefront error evaluation.
- Strengthen own d=0 connections to neurons in the co-activation set (this phase).
- Emit index-update events for each connection created/strengthened (consumed by Phase 8's inference index at the next orchestration boundary).

Neurons in `correction_wired_this_frame` are still strengthened against — i.e., they appear as targets in other neurons' d=0 learning — but they do not run their own learning step this frame. This is how reused correction neurons gradually generalize across reuse events (design §5.4) without themselves emitting fresh errors.

### Spatial co-activation set

`process_spatial`'s wavefront seeds from L=0 sensory neurons that activated this frame. As the wavefront propagates, more neurons join `fired_this_frame` via routing matches. The co-activation set used for d=0 learning at any wave boundary is **the set of neurons fired during this spatial phase so far**, scoped per Phase 2 winner.

### Code touched

- `brain/brain-core/src/neuron.rs`:
  - `connections` field already has slot 0 reserved. Populate it.
  - Add `strengthen_d0(target_id, reward)` analogous to existing `upsert_connection` for d>0.
  - Extend per-neuron spatial-frame processing (added in Phase 5) to call `strengthen_d0` for each co-active target subject to the scoping rule. This work is per-neuron, parallel.
  - `learn_connections` is **not** modified — that's the d>0 path. Spatial d=0 learning is its own code path.
- `brain/brain-core/src/thalamus.rs` — no new "learn" method; learning is embedded in the per-neuron parallel spatial frame. Thalamus role is to dispatch waves and collect emitted index-update events at orchestration boundaries.

### Scoping rule

A neuron's d=0 connections target neurons according to the **Phase 2 winning inference scope**:

- If `base` won: d=0 targets are L=0 neurons in the spatial fired set.
- If `same-level` won: d=0 targets are spatial-fired neurons at the same level as the source.
- If `all-levels` won: d=0 targets are all spatial-fired neurons regardless of level.

Neurons in `correction_wired_this_frame` participate in learning as targets *and* as sources for the d=0 strengthening pass — that's how reused correction neurons gradually generalize across reuse events (design §5.4).

### Phase 4 acceptance

- Unit test: in a synthetic spatial phase with L=0 neurons {A, B, C} fired, after `learn_d0_connections` A's d=0 map contains B and C with positive strength; symmetric for B and C.
- Repeating the same frame strengthens existing entries (no duplicates).
- Stocks regression: same accuracy as Phase 3. d=0 connections build but aren't yet read by predictions or errors (Phase 6).

### Notes / gotchas

- Reward propagation: existing d>0 learning tags `conn.reward` from the frame's reward. d=0 follows the same rule.
- Cost: d=0 strengthens up to (N-1)² connections per frame where N is the spatial fired-set size. On MNIST first frame, N=49 → ~2400 strengthens per frame. Profile early.
- For stocks, the spatial fired set is just L=0 sensory until d=0 connections start producing routing matches. Cost grows as the spatial hierarchy deepens; profile periodically.

---

## Phase 5 — Spatial Wavefront Orchestration

**Goal:** Implement the wave-by-wave thalamus orchestration that drives `process_spatial`. Each wave is a parallel-per-neuron dispatch (mirroring temporal's `process_level`); waves continue until no new activations.

### Per-neuron spatial frame (parallel work)

For each neuron in a wave, in parallel:

- Match own d=0 routing entries against the co-activation set captured at dispatch time; emit activations for matches.
- Emit own d=0 predictions (the d=0 connection set) for post-wavefront error evaluation (Phase 6).
- Strengthen own d=0 connections to neurons in the co-activation set (Phase 4).
- Emit index-update events for each connection created/strengthened (Phase 8 consumes these).

### Thalamus orchestration

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
// Phase 6's error evaluation and correction wiring runs here.
```

### Code touched

- `brain/brain-core/src/brain.rs::process_spatial` — implement the orchestration loop above.
- `brain/brain-core/src/neuron.rs`:
  - Add `process_spatial_frame(co_activation_set) -> SpatialFrameResult` returning emitted activations, predictions, and index-update events.
  - Add `vote_d0()` returning the neuron's d=0 connection set, used inside `process_spatial_frame`. Separate from `vote(age)` to keep the +1 distance convention clean.
- `brain/brain-core/src/column.rs::process_spatial` — analogous to existing `process_level`; iterates owned-neuron tasks and calls `neuron.process_spatial_frame`.
- `brain/brain-core/src/region.rs::dispatch_process_spatial_frame` — fan out across columns, like the existing temporal dispatch.
- `brain/brain-core/src/thalamus.rs` — own the per-frame `predicted_d0` accumulator. Reset at frame end. Apply index-update events at orchestration boundaries.

### Refractory and inhibition

- `fired_this_frame` is shared with `process_temporal` — a neuron that fired in spatial cannot re-fire in temporal (spatial runs first, so this matters mostly for temporal).
- `correction_wired_this_frame` is consulted but stays empty in Phase 5 (no errors detected yet).

### Phase 5 acceptance

- Unit test: pre-wire L=0 sensory neuron S with d=0 routing to existing L=1 correction C; feed S as a wavefront seed; C ends up in `fired_this_frame` and `predicted_d0[C]` contains C's d=0 connections.
- Unit test: spatial wavefront terminates when no new activations. Pre-wire a cycle (A→B, B→A) and verify refractory blocks the second pass.
- Unit test: handoff to temporal — verify the highest-level activations per age from the spatial phase are made available to `process_temporal`'s active-set construction (mechanism TBD in implementation; could be a thalamus-owned set the temporal pipeline reads).
- Stocks regression: same accuracy (predictions accumulate but aren't yet used).

---

## Phase 6 — d=0 Error Detection + Correction Minting

**Goal:** At spatial-wavefront stabilization, evaluate every fired neuron's d=0 predictions against the observed co-activation set, generate errors above threshold, mint correction neurons. No reuse yet — every error mints fresh.

### Code touched

- `brain/brain-core/src/thalamus.rs` — add `evaluate_d0_predictions()` called at the end of `process_spatial`, before returning control to `brain`.
  - Observed set = neurons that fired during this spatial phase, minus `correction_wired_this_frame`, scoped per Phase 2 rule.
  - For each predicting neuron N (not in `correction_wired_this_frame`):
    - Predicted set = `predicted_d0[N]`
    - Compute mismatch using existing common/missing/novel logic
    - If mismatch > error threshold → record d=0 error
  - For each error: call existing `allocate_pattern_neuron` path adapted for d=0 (connection specs carry distance=0; the new neuron's `connections[0]` is populated from the observed set).
  - Add minted neuron ID to `correction_wired_this_frame` and to `fired_this_frame` (it learned this frame; it does not vote or error-check this frame).
- `brain/brain-core/src/neuron.rs::correct_errors` — accept correction entries with d=0 context (routing-table key includes d=0 partial observed set).

### Order of operations in `processFrame`

1. `process_spatial()`:
   a. Spatial wavefront seeds from L=0 sensory and drains; `predicted_d0` accumulated as neurons fire (Phase 5).
   b. Per-neuron d=0 learning happens in parallel inside each neuron's spatial frame (Phase 4).
   c. **NEW:** `evaluate_d0_predictions()` runs at spatial stabilization. Errors recorded.
   d. Op-4 batch creates correction neurons for d=0 errors, registering them in active memory at `erroring_neuron.activation_level + 1`. Routing tables updated. New neurons join `fired_this_frame` and `correction_wired_this_frame`.
   e. Handoff set computed: highest-level activations per age from the spatial fired set, minus `correction_wired_this_frame`.
2. `process_temporal()` runs (existing pipeline). Its active-set construction incorporates the handoff set. d>0 next-frame error evaluation runs as today.
3. Action voting over `fired_this_frame \ correction_wired_this_frame`.

### Phase 6 acceptance

- Unit test: train "digit-1-like" spatial input {A, B, C} across 10 frames → no errors after exposure 2-3. Switch to "digit-7-like" {A, D, E, F, G} → A errors on its d=0 predictions (predicted B, C; observed D, E, F, G), correction neuron minted with d=0 connections to {A, D, E, F, G}.
- Inspect: correction neuron's routing table is wired into A's routing under d=0 context. Correction neuron is in `correction_wired_this_frame` (verified by checking it does not contribute to action voting this frame).
- Stocks regression: ≤ 5% accuracy delta. d=0 errors may produce new neurons in stocks too; expected, not a regression.

### Notes / gotchas

- Error threshold mode: use the existing `errorCorrectionMode` for d=0. Parsimony of parameters — no separate `d0ErrorCorrectionMode`. If MNIST shows it needs different tuning, we'll revisit then.
- A correction neuron minted this frame is in `fired_this_frame` (for learning attachment) and in `correction_wired_this_frame` (inhibited from voting and error-checking). It becomes a normal voter starting next frame.
- Correction neurons are registered in active memory at `erroring_neuron.activation_level + 1` — encoding the abstraction step. Neurons have no intrinsic level (Phase 3); the activation level is per-frame, determined by the routing source. Both d=0 and d>0 corrections follow the same `+1` rule.
- The `correction_wired_this_frame` inhibition is the load-bearing termination rule per design §3.5 / §11.1: it prevents reuse-cascade in Phase 9 and also makes Phase 6's mints behave consistently with reuses. It also caps within-frame spatial deepening at one fresh layer.

---

## Phase 7 — MNIST Single-Frame Harness

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

### Phase 7 acceptance

- Train on 1000 images (100 per digit, balanced), eval on 200 held-out → >50% accuracy.
- Inspect neuron counts: should be ≪ what cortex-only produced.
- Wavefront depth: log mean/p99 waves per frame. Should be small (≤10 expected).
- Per-image processing time: should be O(seconds) max, not O(minutes) as in cortex-only.

### If Phase 7 fails

Most likely failure modes and diagnostics:

- **All-or-nothing memorization** (single huge correction neuron per digit): probe the merge threshold / error threshold; check `mnist d0 correction neuron sizes` distribution.
- **No correction neurons at all**: error threshold too lax, or d=0 connections strengthen too slowly. Lower threshold, increase initial connection strength.
- **Wavefront not terminating**: shouldn't be possible by 11.1 reasoning, but if it happens, log queue size per wave to find the source.

---

## Phase 8 — Reverse Inference Index

**Goal:** Build the data structure that answers "which neurons have a connection to target T at distance d?"

### Code touched

- `brain/brain-core/src/column.rs` or `thalamus.rs` — add `inference_index: FxHashMap<NeuronId, FxHashMap<Distance, FxHashSet<NeuronId>>>`. Maps `target → distance → set of sources`. Sharded by the column owning the **source** neuron, so each column owns the index for its resident neurons.
- `brain/brain-core/src/neuron.rs` — on connection create/strengthen/decay/delete, emit an `IndexUpdate` event. Column applies the update to its local index.
- `brain/brain-core/src/region.rs` — expose a `query_inference_sources(targets: &[NeuronId], distance: Distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>` op that fans out across columns and merges per-target source sets.

### Phase 8 acceptance

- Unit test: create neurons N1→T1 and N2→T1 at distance 0; query for T1 at d=0 returns {N1, N2}.
- Decay a connection below threshold; index reflects removal.
- Index size stays proportional to total connection count (no bloat).

### Notes / gotchas

- Update batching: don't update index per-connection-strengthen during a parallel dispatch. Each neuron emits index-update events as part of its per-neuron result; the thalamus applies them at the **orchestration boundary** that immediately follows the dispatch (i.e., between the parallel wave and the next orchestration step). This means the reuse lookup in Phase 9 sees the index reflecting this frame's deltas — no stale-index lag, no "defer to next frame" workaround. The pattern mirrors temporal: parallel-per-neuron work inside the dispatch, cross-neuron consolidation (allocate, route, index-update) sequentially between dispatches.
- Memory cost: index roughly doubles the connection-graph memory footprint. Acceptable for now; revisit if it becomes a bottleneck.

---

## Phase 9 — Reuse Lookup in Correction Path

**Goal:** Before minting a correction neuron, query the reverse index for an existing neuron whose inference signature partially matches the required correction. If match score ≥ the existing merge threshold (shared with partial-context recognition), wire reuse; otherwise mint. Reuse is always on — no enable flag.

### Code touched

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

### Phase 9 acceptance

- Unit test: train two errors with overlapping observed sets — second error reuses the first's correction neuron when overlap ≥ merge threshold.
- MNIST: rerun Phase 7 with reuse enabled. Total neuron count should be significantly lower (target: ≥30% reduction). Accuracy should be ≥ Phase 7.
- Profile: reuse lookup adds < 20% to per-frame runtime.

### Notes / gotchas

- Reuse applies at both distances — d>0 in `process_temporal`, d=0 in `process_spatial`. Same mechanism, different phase.
- Reuse reads the existing global merge threshold from brain options — no new parameter. Setting that threshold to 1.0 disables reuse (and also disables partial-context pattern recognition); the two behaviors are intentionally coupled.
- Edge case: erroring neuron N's reuse candidate is N itself. Filter self-matches.
- Edge case: reuse candidate is a neuron minted earlier *in the same phase, same frame* (e.g. an earlier d=0 error in this spatial phase minted a neuron that's now a viable reuse target for a later d=0 error). Allow it. The reused neuron goes into `correction_wired_this_frame` as usual; the index update at frame end keeps the reverse index consistent.
- Termination: reused neurons go into `correction_wired_this_frame` (per Phase 6's rule), so they cannot generate fresh errors this phase. This is the design's §11.1 termination guarantee — refractory + correction-wired-inhibition bounds the phase regardless of reuse activity.

---

## Phase 10 — Reuse Validation

- Run MNIST with reuse: report neuron count, accuracy, transfer effect (train 0-4, eval on 5-9).
- Stocks regression with reuse: directional accuracy should match or exceed Phase 3 baseline.

---

## Phase 11 — Stocks Integration

Branch reconciliation already happened in Phase 1; no need to repeat here.

- Run stocks with `process_spatial` enabled and reuse on. Spatial inputs on stocks will be the highest-level temporal patterns per age — initially empty, then growing as the temporal hierarchy matures.
- d=0 connections form across co-occurring top-level patterns within each frame. Spatial corrections from one frame feed into the temporal phase in subsequent frames, building spatio-temporal abstractions.
- Compare per-episode ROI and directional accuracy against the Phase 2 winner baseline.
- Tune the global merge threshold and d=0 error threshold if needed.

---

## Phase 12 — Forget-Rate & Class-Neuron Generalization

- Long-running training (10k+ frames). Monitor:
  - Distribution of d=0 connection strengths over time per correction neuron.
  - Whether reuse counts per neuron rise (good — class behavior).
  - Whether action bindings on heavily-reused correction neurons stay correct.
- Forget rate is a single global static (set in Phase 1) — d=0 and d>0 connections use the same rate. If d=0-specific decay turns out to be needed, that's a follow-up; default position is parsimony.

---

## Phase 13 — Persistence / Backup / Import-Export Updates

**Goal:** Make sure every persistence path round-trips the new architecture cleanly. The v2 changes introduce these serialization concerns:

- **d=0 connections in `connections[0]`.** The connections vec already reserves slot 0 but it's been unused until Phase 4. Serialization needs to round-trip it.
- **No `neuron.level` field.** Phase 3 removed it from the struct. Serialized form should not include it. Backup-format version was bumped in Phase 3 — Phase 13 may bump again for d=0 connection round-trip, or consolidate to a single v2 bump if Phase 3 deferred its bump.
- **No per-neuron forget rate.** Phase 1 made forget rate a global static; nothing to serialize per neuron.
- **Reverse inference index (Phase 8).** Decide whether to persist it or rebuild on load. Recommended: rebuild on load — it's a derived index, persisting it doubles backup size and risks staleness.

### Code touched

- `brain/brain-core/src/backup.rs` — extend `SerializedConnection` / `SerializedNeuron` so distance=0 entries round-trip. Verify the existing format already iterates over the full `connections` vec (slot 0 included) rather than skipping it. Confirm `neuron.level` is not in the serialized form (removed in Phase 3).
- `brain/brain-core/src/thalamus.rs` — on restore, rebuild the inference index from scratch by walking all neurons' connections. Add a `rebuild_inference_index()` method called once at load time.
- DB import/export apps (e.g. anything under `apps/` that imports or exports brain state) — sweep for code that assumes connections start at distance 1 or that reads/writes `neuron.level`. Update accordingly.
- `correction_wired_this_frame` and `fired_this_frame` — **not** persisted. Per-frame state, resets at frame start.

### Phase 13 acceptance

- Round-trip test: train a brain on a few MNIST images (enough to mint d=0 corrections across multiple activation levels), snapshot, restore, verify (a) all d=0 connections present with correct strengths, (b) all correction neurons present and routable, (c) inference index rebuilt and queries return the same results as pre-snapshot, (d) one more frame of training on the restored brain produces identical results as on the original.
- DB import/export apps: run a round-trip on a stocks brain with d=0 connections; verify byte-identical export after `import → export`.

### Notes / gotchas

- The "rebuild inference index on load" choice means slightly slower restore but no risk of stale index. Worth it.
- Old (pre-Phase-3) backup format is incompatible — `neuron.level` removal alone breaks the layout. Bump format version, reject old backups with a clear error; migration isn't worth the complexity for the current scale of stored brains.
- Phase 13 can land after Phase 11 (Stocks integration) without blocking it. But before any production deployment that depends on backups, Phase 13 must close.

---

## Brain Options Added

No new brain options. Reuse is always on. Both reuse and partial-context pattern recognition read the existing global merge threshold — they are intentionally coupled. d=0 error threshold and d=0 forget rate default to the existing per-level settings used for d>0.

No breaking changes to NAPI surface. `processFrame`, `learn`, `infer` signatures unchanged.

---

## Open Items Carried Into Implementation

These were left open in the design and may surface as concrete decisions during coding:

1. **Inference-index storage layout** (§8.3): flat `(target,distance) → sources` vs nested. Decide in Phase 8 based on benchmark.
2. **Update batching cadence** for the inference index: per-frame vs per-op-4. Resolved — apply at orchestration boundaries between parallel dispatch waves (see Phase 8 notes).
3. **d=0 forget rate anchor** (§8.5): resolved — single global static rate (Phase 1). d=0 and d>0 share it. Revisit in Phase 12 only if empirical evidence demands d=0-specific decay.
4. **Action binding through reuse**: when a correction neuron is heavily reused, action votes may need normalization. Revisit if Phase 12 shows dilution.

---

## Test Harness Additions

- `brain/brain-core/tests/d0_learning.rs` — Phase 4 unit tests.
- `brain/brain-core/tests/d0_errors.rs` — Phase 6 unit tests.
- `brain/brain-core/tests/wavefront.rs` — Phase 5 wavefront convergence + refractory tests.
- `brain/brain-core/tests/reuse.rs` — Phase 9 reuse decision tests.
- `brain/brain-core/tests/persistence_roundtrip.rs` — Phase 13 round-trip tests.
- `apps/mnist/jobs/single-frame-eval.js` — Phase 7 + Phase 10 harness.
- `apps/mnist/jobs/transfer.js` — Phase 10 transfer test (digits 0-4 → 5-9).