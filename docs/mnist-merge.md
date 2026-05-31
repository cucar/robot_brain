# MNIST Branch Reconciliation

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Status:** Pre-implementation
**Next:** [inference-level.md](./inference-level.md)

---

## Why

The `mnist` branch has accumulated substantial work alongside experimental v1 moment-neuron scaffolding. Before any new structural work lands on top of it, the keepable parts must be merged into `dev`. Merging late would compound conflicts with later phases.

This is a prerequisite for everything downstream — inference-scope experimentation, spatial processing, neuron reuse.

---

## Phase 1 — Branch Reconciliation (mnist → dev)

### Process

1. Land the categorized changes below on `dev` in three groups, each its own commit:
   - **Group A — algorithmic core fixes** (warmup gate, refine_context disabled, negative-reinforcement removal, L1 decay, confidence-weighted error metric, idempotent `add_context`, activation_strength reset on restore).
   - **Group B — apps + tooling** (new MNIST encoders, `dump_image.js`, refactored `apps/mnist/jobs/test.js`, inspection API, frame timings instrumentation).
   - **Group C — static forget rate simplification** (collapse `LevelDecayMode` → single global rate; see below).
2. Roll back the **Experimental v1** group on the merge branch (implant API, parallel contexts, processing toggles, supervised action learning, `action_alpha`, `process_frame(events, actions, …)` signature change). Don't land any of it on `dev`.
3. Delete the **Throwaway** group on the merge branch (debug traces, `_digit5/` fixtures, sweep scripts).
4. Open the merge into `dev` as a reviewable PR with the table below in the description.
5. Tag the pre-merge tip of `mnist` as `mnist-v1-final` so v1 experiments stay reproducible.
6. After merge, run the stocks pipeline on `dev` and record the new baseline.

### Static forget rate

The mnist branch introduced a `LevelDecayMode` enum (Exponential | Linear | Static) to experiment with per-level decay schedules. The new spatial-processing + neuron-reuse design makes per-level decay obsolete: with intrinsic levels going away and the same neuron being reused across contexts, there is no longer a "deeper means longer-lived" distinction to honour. **A single global forget rate applies to every pattern, regardless of level.**

Concretely:
- Drop the `LevelDecayMode` enum and the `level_decay_mode` parameter from `Brain::new`, `Thalamus::new`, `Backup::new`.
- `Thalamus::effective_forget_rate` collapses to `base_rate` for all levels (matches what `LevelDecayMode::Static` does today).
- Remove `--level-decay` from `libs/node/src/run.js`.
- Note: this still implements the **L1-decays-at-base-rate** fix from the branch — L1 children of L0 sensories now decay (they were effectively immortal before). That is the behaviour change that ships to `dev`; the multi-mode scaffolding above it does not.

---

## Change Inventory

### Legend

- **PC** — Permanent / core (algorithmic fix or essential infra; ships to `dev`)
- **PA** — Permanent / apps (MNIST tooling worth keeping for later validation)
- **EV1** — Experimental v1 (moment-neuron v1 scaffolding; roll back, stays on `mnist`)
- **TW** — Throwaway (debug traces, fixtures, sweep scripts; delete)
- **★** — Affects stocks behaviour; post-merge baseline will shift

### brain-core/src/brain.rs (+977 lines)

| Change | Cat | Notes |
|---|---|---|
| Warmup gate in `process_levels` (skip L1+ until `frame >= context_length`) | PC ★ | Mirrors thalamus gate. Prevents empty-context error patterns at sequence start. |
| `refine_context` call removed from `recognize_patterns` | PC ★ | Mid-training context refinement made recognition non-reproducible. Keep disabled. |
| `learn_connections` no longer negatively reinforces unfired events | PC ★ | Removed `get_neurons_not_found` weakening — was causing discrete-death bursts during multi-pass training. Mirrors action-side behaviour. |
| L1 children of L0 now decay at base rate (via `effective_forget_rate`) | PC ★ | Was previously 0. Patterns can now expire. |
| Confidence-weighted error metric (per-dim argmax confidence vs raw miss rate) | PC ★ | Pattern-creation gate now reflects "when I was wrong, how confident was I" — prevents over-firing in early training. |
| Empty-context error patterns are skipped at allocation time | PC | Pulls context first, bails if empty. Stops sibling explosion. |
| `FrameTimings`, `MemoryTimings`, `NeuronOpTimings`, `OrchestrationTimings` | PA | Per-section wall-clock for profiling. Add as opt-in (or always on, cost is negligible). |
| `ActionVote`, `ActionVoteStats` on `FrameResult` | PA | Per-voter and aggregated vote stats — useful for analysis harnesses. |
| `inspect_neuron`, `dump_neuron_connections`, `get_votable_entries`, `InspectedNeuron` | PC | Inspection API. Cheap and useful. |
| `ImplantState`, `start_implant`, `implant_position`, `finalize_implant`, `ImplantSummary` | EV1 | Direct-teaching path superseded by spatial-processing design. Roll back. |
| `ContextState`, `init_contexts`, `swap_to`, `set_active_context`, `contexts` field | EV1 | Parallel-context training infra for v1 image scan. Roll back. |
| `event_processing` / `action_processing` / `learning` flags, `set_processing_mode` | EV1 | Frozen-context staged scan/predict flow. Roll back; revert to unconditional full pipeline. |
| `learn()`, `infer()` (direct supervised/inference-only paths) | EV1 | v1 supervised path. Roll back. |
| `action_alpha` ctor param + static alpha for action reward updates | EV1 | v1 supervised action learning. Roll back. |
| `LevelDecayMode` enum + ctor param | EV1 | Collapse to single global static rate (see "Static forget rate" above). |
| `process_frame(events, actions, rewards)` signature (was `(inputs, rewards)`) | EV1 ★ | Required by forced-action / implant flow. Revert to `(inputs, rewards)`. All apps (stocks, text, mnist) need their call sites adjusted. |

### brain-core/src/neuron.rs (+197 lines)

| Change | Cat | Notes |
|---|---|---|
| Warmup gate in `recognize_patterns` (skip until `current_frame >= context_length`) | PC ★ | See brain.rs note. |
| `learn_connections` simplified — no more `weaken_connection` on misses | PC ★ | See brain.rs note. |
| `add_context` idempotent (strengthen on duplicate instead of panic) | PC | Surfaced by `(parent, age)` dedupe re-installing the same entry. Safe correctness fix. |
| `NeuronOpTimings` + per-op timing instrumentation in `process_frame` | PA | Aggregates up to `FrameTimings`. |
| `dump_connections` | PC | Backs the inspection API. |
| `context_length` field on `Neuron` (needed for warmup gate) | PC | Required by gate. |
| `learn_supervised_action`, `action_alpha`, action-aware `strengthen_connection` (static vs dynamic alpha) | EV1 | Supervised action path. Roll back together. |
| `learning: bool` plumbed into `process_frame` / `recognize_patterns` | EV1 | Gates pattern creation + child-activation strengthening. Roll back. |

### brain-core/src/thalamus.rs (+363 lines)

| Change | Cat | Notes |
|---|---|---|
| Warmup gate in `evaluate_vote_error` | PC ★ | Stops error-pattern firing before the context window has filled. |
| Confidence-weighted error metric | PC ★ | See brain.rs note. |
| Empty-context skip in `get_level_corrections` | PC | See brain.rs note. |
| `OrchestrationTimings` (get_level_tasks / dispatch_frame / collect_activations / collect_votes) | PA | Profiling. |
| `collect_level_votes` rename (was `collect_votes`) | PC | Internal — needed because new public `collect_votes` exists for action-learning path. If we roll back action-learning, the rename can revert too; either way is fine. |
| `get_pattern_context_entries`, `dump_neuron_connections` | PC | Inspection API. |
| Verbose debug logging in error-correction path | PA | Useful when `debug` is on. Keep. |
| `LevelDecayMode` field + `effective_forget_rate(mode)` | EV1 | Collapse to single static rate. |
| `collect_votes`, `learn_action_connections` (public) — vote-collection + supervised-action wiring | EV1 | Roll back with supervised action learning. |
| `set_action_processing`, `set_learning` | EV1 | Processing toggles. Roll back. |
| `implant_default_connection`, `implant_pattern` | EV1 | Implant API. Roll back. |
| `process_level` / `get_level_tasks` / `get_level_corrections` carry `learning: bool` + `frame_number` | EV1 / PC | `frame_number` is needed for the warmup gate (PC). `learning` gate is EV1. Split when applying: keep the `frame_number` plumbing, drop the `learning` plumbing. |

### brain-core/src/column.rs (+131 lines)

| Change | Cat | Notes |
|---|---|---|
| `collect_votes`, `learn_action_connections` | EV1 | Roll back. |
| `install_implant_pattern`, `install_implant_default_connection` | EV1 | Roll back. |
| `dump_neuron_connections`, `get_child_context_entries` | PC | Inspection API. |
| `set_action_processing`, `set_learning` + `action_processing` / `learning` fields | EV1 | Roll back. |
| `action_alpha` ctor param + propagated to `Neuron::new` | EV1 | Roll back. |
| `context_length` param to `Neuron::new` | PC | Required by warmup gate. |
| Pre-wire default actions gated on `action_processing` | EV1 | Revert to unconditional pre-wire. |
| `activation_strength` reset to 1.0 in restore (`if entry.activation_strength > 1.0`) | PC | Prevents stale cross-episode accumulation from keeping patterns artificially alive. |

### brain-core/src/region.rs (+92 lines)

| Change | Cat | Notes |
|---|---|---|
| `collect_votes`, `learn_action_connections`, `route_vote_entries`, `route_learn_tasks` | EV1 | Roll back. |
| `install_implant_pattern`, `install_implant_default_connection` | EV1 | Roll back. |
| `get_child_context_entries`, `dump_neuron_connections` | PC | Inspection API. |
| `set_action_processing`, `set_learning` | EV1 | Roll back. |
| `action_alpha` propagated through ctor | EV1 | Roll back. |

### brain-core/src/memory.rs (+55 lines)

| Change | Cat | Notes |
|---|---|---|
| `replace_active_neuron` | EV1 | Used by frozen-context drilling. Roll back. |
| `get_votable_entries` | EV1 | Used by supervised action-learning path. Roll back (or keep if inspection finds use for it; currently nothing else calls it). |
| `#[derive(Clone)]` on `Memory` | EV1 | Required by `ContextState`. Roll back with parallel contexts. |

### brain-core/src/types.rs, backup.rs

| Change | Cat | Notes |
|---|---|---|
| `LevelDecayMode` enum (types.rs) | EV1 | Delete. |
| `level_decay_mode` plumbed through `Backup` | EV1 | Delete. |

### brain-napi (lib.rs, index.d.ts)

| Change | Cat | Notes |
|---|---|---|
| `processFrame(events, actions, rewards)` binding | EV1 ★ | Revert to `(inputs, rewards)`. |
| `learn`, `infer` bindings | EV1 | Remove. |
| `startImplant`, `implantPosition`, `finalizeImplant` | EV1 | Remove. |
| `initContexts`, `setActiveContext` | EV1 | Remove. |
| `setProcessingMode` | EV1 | Remove. |
| `LevelDecayMode` parsing + ctor option | EV1 | Remove. |
| `actionMode` / `actionAlpha` ctor options | EV1 | Remove. |
| `inspectNeuron`, `dumpNeuronConnections`, `getVotableEntries`, `getActiveNeurons` | PC | Keep (matches core inspection API). |
| `FrameResult` marshalling of `actionVoteStats`, `actionVotes`, `timings` | PA | Keep. |

### apps/stocks, apps/text, libs/node

| File | Change | Cat | Notes |
|---|---|---|---|
| `apps/stocks/jobs/*.js`, `apps/text/jobs/test.js` | `processFrame(inputs, new Map(), rewards)` adapter | EV1 ★ | Revert to `(inputs, rewards)` after the brain signature is reverted. No algorithm change in these files. |
| `libs/node/src/run.js` | `--level-decay` CLI arg | EV1 | Remove. |

### apps/mnist

| Path | Cat | Notes |
|---|---|---|
| `encoders/digits.js`, `encoders/mnist_encoder.js`, `encoders/pixel_channels_encoder.js`, `encoders/row_channels_encoder.js` | PA | Modular encoders replacing old single `encoder.js`. Useful for post-spatial-processing MNIST validation. |
| `dump_image.js` | PA | Converts MNIST images to text-pipeline format. Useful for cross-app validation. |
| `jobs/test.js` (refactor, +/- ~600 lines) | PA | Rewritten around the new encoders. Keep. |
| `encoder.js` (deleted) | PA | Replaced by `encoders/`. Deletion is correct. |
| `jobs/implant_test.js` | EV1 | Exercises the implant API. Delete. |
| `jobs/burst_trigger_trace.js`, `check_match_drift.js`, `classifier_voters.js`, `cold_replay_diff.js`, `connection_drift_trace.js`, `context_erosion_trace.js`, `convergence_trace.js`, `frame_activation_diff.js`, `inspect_pair.js`, `multi_digit_trace.js`, `parent_conn_dump.js`, `pattern_growth_trace.js`, `shared_pattern_trace.js`, `voter_analysis.js` | TW | Debug/trace jobs from v1 investigation. Delete. |
| `jobs/_sweep.sh`, `_compare_aliased.js`, `_inspect_shared_voter.js`, `_digit5_crossclass.js`, `_digit5_overlap.js` | TW | Ad-hoc scripts. Delete. |
| `jobs/_digit5/*.txt` (44 files), `jobs/_img*_bits.txt`, `_img*_visual.txt` | TW | Fixture dumps. Delete. |
| `apps/text/data/binary_*.txt`, `mnist_*.txt`, `sweep_*.txt` | TW | One-liner fixture dumps for text-app debugging. Delete unless any are referenced by `apps/text/jobs/test.js` after revert (verify). |
| `apps/mnist/performance.md` | PA | v1 performance notes — keep as a reference for what was measured, or delete if no future reader is expected. User call. |

---

## Test Plans

Group these by what's being verified — most "permanent core" changes are covered by one or two of the same tests.

### T1 — Stocks no-regression (covers all ★ changes)

The five ★ algorithmic changes (warmup gate, refine_context disabled, no negative reinforcement, L1 decay, confidence error metric) all affect stocks. They are co-dependent — testing them individually is impractical because they were developed together and partially compensate for each other. Test as a bundle.

- **Setup:** baseline directional accuracy + connection-count + pattern-count snapshot on `main` for `apps/stocks/jobs/test.js`, `synthetic-cycle-test.js`, `synthetic-extended-test.js`, `multi-channel-test.js`.
- **Run:** same configs on merged `dev`.
- **Pass:** directional accuracy within ±1% of `main` baseline on every harness. Per-direction accuracy (up vs down) also within ±1%.
- **Diagnostic:** if accuracy drops, inspect (a) pattern count at end of training — should be substantially lower since empty-context patterns are no longer created; (b) connection count — should be slightly higher since misses don't weaken; (c) error rate trajectory across training — should be smoother (no discrete-death bursts).
- **Record:** new baseline numbers in the PR description so future regressions are detectable.

### T2 — Brain unit tests

- Run `cargo test` in `brain/brain-core`.
- Pass: all green. The two test fns that take `Neuron::new(...)` were updated on `mnist` to pass `None, 10` — after the action_alpha rollback they take just the original arg set; after the `context_length` field stays (it's PC), the test helper passes a literal `10`. Verify the test file compiles cleanly after the rollback edits.

### T3 — Text pipeline no-regression

- Run `apps/text/jobs/test.js` on `main` and on merged `dev`. Memorization accuracy should match (deterministic — text is the most sensitive harness to the warmup-gate and refine_context changes because sequences are long).
- Pass: identical memorization accuracy. Identical neuron count within ±1%.

### T4 — Static forget rate behaviour

- With the `LevelDecayMode` collapse: confirm `effective_forget_rate(base, ctx, level)` returns `base` for level=0 and level≥1. (Was: `0.0` for level=0, then `base / ctx^(level-1)`.)
- On stocks, run a long synthetic-extended-test and inspect the final neuron-count distribution by level. With Exponential decay, deep-level neurons accumulated indefinitely. With static, deep levels should plateau.
- Pass: deep-level neuron counts stop growing past a stable saturation point (qualitative — confirm in the distribution, no precise threshold).

### T5 — Inspection API

- Spin up a small brain via NAPI, train briefly on a stocks fixture, then call `inspectNeuron`, `dumpNeuronConnections`, `getVotableEntries` on a known active neuron.
- Pass: returned values are well-formed (no panics, fields populated). This is a smoke test — the API is read-only and unlikely to break anything functionally.

### T6 — Frame timings

- Enable `debug` and run any one harness. Confirm `result.timings` is populated with non-zero values for `build_frame`, `activate`, `process_levels`, etc.
- Pass: fields present and roughly sum to elapsed (within the documented sub-bucket overhead). This is also a smoke test.

### T7 — MNIST encoder unit-level

- Run the new `apps/mnist/encoders/*.js` against a single canonical MNIST digit image (one from the standard test set, not a `_digit5` fixture).
- Pass: output bit vector matches the expected encoding for the chosen quantization. (No new spatial-processing logic depends on this yet — this is just confirming the encoders themselves work after the merge.)

### T8 — Backup / restore

- Train a small brain on stocks for ~100 frames, snapshot, restore, train another 100 frames.
- Pass: post-restore behaviour identical to never-restored baseline. The `activation_strength` reset on restore is the change that needs verification — if a pattern's strength was >1 at snapshot time, it should resume from 1.0 not from the stored value.

### T9 — Stocks baseline record (acceptance)

After all of the above pass, run all four stocks harnesses with default config and **record the new baseline** in the PR description. This is the post-merge truth that subsequent phases compare against.

---

## Acceptance

- **T1**: stocks directional accuracy ±1% of `main` (both directions independently).
- **T2**: `cargo test` green.
- **T3**: text memorization identical.
- **T4**: deep-level neuron count plateaus under static decay.
- **T5/T6**: inspection + timings smoke tests pass.
- **T7**: MNIST encoders produce expected outputs.
- **T8**: snapshot/restore cycle is behaviourally idempotent.
- **T9**: new stocks baseline committed to PR description.
- Classification table (this document) is the PR description.
- `LevelDecayMode` is gone from the codebase; no per-level decay scaling logic remains.

---

## Notes

- All subsequent phases happen on `dev` or branches off `dev`. `mnist` becomes archival under tag `mnist-v1-final`.
- If T1 fails, the most likely culprit is the confidence-weighted error metric — it's the change with the largest behavioural surface. Diagnostic path: temporarily revert just that one piece, rerun T1, isolate.
- If MNIST validation is wanted *before* spatial processing lands (sanity check that the new encoders work end-to-end), use `apps/mnist/jobs/test.js` against the **current** brain — accuracy will be low (the brain isn't actually equipped for spatial yet), but pipeline-level "does the run finish without errors" is what matters at this point.
