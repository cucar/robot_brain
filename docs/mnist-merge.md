# MNIST Branch Reconciliation

---

## Why

The `mnist` branch has accumulated substantial work alongside experimental v1 moment-neuron scaffolding. Before any new structural work lands on top of it, the keepable parts must be merged into `dev`. Merging late would compound conflicts with later phases.

This is a prerequisite for everything downstream — inference-scope experimentation, spatial processing, neuron reuse.

---

## How to read this document

Changes on `mnist` group into **coherent projects**. Each project is a single commit onto `dev`. Work them in order — top to bottom is the merge plan. Each project section contains:

- **Goal** — what the project was trying to accomplish.
- **Surface** — every file/symbol touched.
- **Decision** — Keep / Collapse / Roll back / Delete.
- **Test plan and acceptance** — what verifies it when the commit lands.

Categorization shorthand: 🟢 keep · 🟡 collapse · 🔴 roll back · ⚫ delete · ★ stocks-affecting.

**Workflow.** work on `dev` and pull each subsequent project's changes across from `mnist` one commit at a time, testing as you go. After all projects land, there is one **post-merge task** (delete the `mnist` branch).

---

## Project 1 — Small correctness fixes 🟢

**Goal.** Two unrelated paper-cuts surfaced during MNIST debugging.

**Fixes.**

1. **Idempotent `add_context`.** The `(parent, age)` dedupe path in pattern creation could re-install the same context entry, triggering a `panic!`. `add_context` now strengthens on duplicate instead.
   - `neuron.rs::add_context`.

2. **`activation_strength` reset on restore.** Cross-episode strength accumulation could leave patterns with very high activation_strength values. When forget_rate is 0 those values made patterns immortal across snapshot/restore. On restore, clamp strength to 1.0 if it was above. Pattern stays alive (>0); siblings get equal footing.
   - `column.rs::restore` (in the pattern-iteration loop).

**Decision: Keep both.**

**Test plan and acceptance.**
- **T-restore** — train a small brain ~100 frames, snapshot, restore, train another 100 frames. Pass: post-restore behaviour matches never-restored baseline. The clamp is the change to verify.
- **T-dedupe** — covered transitively by stocks/text runs (those exercise pattern creation). No separate test needed unless something explicitly regresses.
- `cargo test` green.

---

## Project 2 — Profiling instrumentation 🟢

**Goal.** Make MNIST training cost legible. Per-frame, per-section, per-neuron-op wall-clock measurements bubbled up through the call stack so a harness can read them off `FrameResult`.

**Surface.**
- `neuron.rs`: `NeuronOpTimings` (learn / recognize / correct / vote, plus recognize sub-buckets); plumbed through `process_frame`/`recognize_patterns`.
- `column.rs`: timings forwarded on `ColumnProcessResult`.
- `thalamus.rs`: `OrchestrationTimings` (get_level_tasks / dispatch_frame / collect_activations / collect_votes); forwarded on `ProcessLevelResult`.
- `brain.rs`: `FrameTimings` and `MemoryTimings` aggregating everything; `timings` field on `FrameResult`.
- `brain-napi`: marshalling of the timings struct into the JS result.
- Verbose error-correction debug logging in `thalamus.rs` (gated on `debug`).

**Decision: Keep all of it.** Cost is negligible (a few `Instant::now()` calls per neuron per frame); benefit is permanent. The verbose error-correction logging is `debug`-gated and stays off in normal runs.

**Test plan and acceptance.**
- **T-timings-smoke** — enable `debug`, run any one harness, confirm `result.timings` is populated with non-zero values across all sections and that they roughly sum to elapsed (within documented sub-bucket overhead). Smoke only.
- `cargo test` green.

---

## Project 3 — Inspection API 🟢

**Goal.** Read-only introspection for harness-side debugging: what does a neuron know, who does it connect to, what's in a pattern's stored context.

**Surface.**
- `neuron.rs`: `dump_connections`.
- `column.rs`: `dump_neuron_connections`, `get_child_context_entries`.
- `region.rs`: same two, routed.
- `thalamus.rs`: `dump_neuron_connections`, `get_pattern_context_entries`.
- `brain.rs`: `inspect_neuron` (the user-facing wrapper); `InspectedNeuron` result struct.
- `brain-napi`: `inspectNeuron`, `dumpNeuronConnections`, `getActiveNeurons` bindings.
- `brain.rs`: `ActionVote` / `ActionVoteStats` on `FrameResult` and the `compute_action_vote_stats` / `collect_action_votes` that populate them. Inspection-adjacent (per-voter and aggregated digit votes).

**Decision: Keep.** Pure-read APIs, no risk to behaviour.

**Excluded:** `Memory::get_votable_entries` was added for the supervised action path. Nothing else calls it. It goes with Project 10 (roll back).

**Test plan and acceptance.**
- **T-inspect-smoke** — train briefly via NAPI, call each inspection method on a known active neuron, confirm well-formed results (no panics, fields populated). Smoke only.
- `cargo test` green.

---

## Project 4 — Static forget rate 🟡 ★

**Goal.** The branch introduced a `LevelDecayMode` enum (Exponential | Linear | Static) to experiment with per-level decay schedules. The new spatial-processing + neuron-reuse design makes per-level decay obsolete: with intrinsic levels going away and the same neuron being reused across contexts, there is no longer a "deeper means longer-lived" distinction to honour. **A single global forget rate applies to every pattern, regardless of level.** The branch also surfaced a latent bug — the original code zeroed L0's forget rate, which (since L0 owns L1 children's death timing) made L1 patterns immortal. The L0→base behaviour fix is the only piece of this project that actually ships.

**Surface.**
- `types.rs`: `LevelDecayMode` enum — delete.
- `thalamus.rs`: `level_decay_mode` field, ctor param, `get_base_forget_rate`/`get_level_decay_mode` accessors — delete. `effective_forget_rate` collapses to `base_rate` for every level (matches what `LevelDecayMode::Static` did). **Keep `level.max(1)`-equivalent behaviour** — L0 now returns base, which is the L1-decay fix.
- `brain.rs`, `backup.rs`, `column.rs`, `region.rs`: drop the mode parameter from ctors.
- `brain-napi/src/lib.rs`, `index.d.ts`: drop `levelDecay` ctor option and enum parsing.
- `libs/node/src/run.js`: drop `--level-decay` CLI arg.

**Decision: Collapse to single static rate.** Document the L1-decay behaviour change explicitly in the commit message so future readers know L1 patterns decay where they didn't before.

**Test plan and acceptance.**
- **T-decay-1** — confirm `effective_forget_rate(base, ctx, level)` returns `base` for `level ∈ {0, 1, 2, 3, 4}`.
- **T-decay-2** — long run of `synthetic-extended-test.js` on `dev` vs `main`. Inspect final neuron count by level. On `main` deep-level counts grow unboundedly; on `dev` they should plateau. Qualitative — no precise threshold.
- **T-stocks (Project 2)** — re-run after this commit lands; the L1-decay change is also a ★ shift and folds into the post-Project-2 baseline.
- `cargo test` green.
- `LevelDecayMode` is gone from the codebase; no per-level decay scaling logic remains.

---

## Project 5 — Image implant / direct teaching 🔴

**Goal.** Bypass error-driven learning entirely and directly install L1 patterns from observed bit histories. The flow: for each pixel position in an image, record `(parent_bit, packed_context_bits) → next_bit_distribution`; once all positions are seen, materialize the majority transitions as L1 patterns with pre-baked stored context and outgoing prediction connections. A teaching shortcut that skipped the slow part of error-driven discovery.

**Surface.**
- `brain.rs`: `ImplantState` struct (per-image bit histories + observation map); `implant_state` field; `start_implant`/`implant_position`/`finalize_implant` methods; `ImplantSummary` return type.
- `thalamus.rs`: `implant_default_connection`, `implant_pattern` (build the L1 spec, allocate id, register parent/level, install via region).
- `region.rs`, `column.rs`: `install_implant_pattern`, `install_implant_default_connection` plumbing.
- `brain-napi`: `startImplant`/`implantPosition`/`finalizeImplant` bindings.
- `apps/mnist/jobs/implant_test.js`: the harness for it.

**Decision: Roll back entirely.** The spatial-processing design accomplishes the same outcome (durable spatial patterns) through the normal error-driven path with d=0 connections — there's no reason to ship a parallel shortcut.

**Roll back checklist:** all symbols above; ensure no remaining references in `apps/`.

**Test plan and acceptance.**
- `cargo test` green.
- `grep` confirms no remaining references to any of the above symbols.

---

## Project 6 — Parallel context training 🔴

**Goal.** Train multiple images concurrently by giving each one its own runtime state (memory, frame counter, rewards) while the thalamus (neurons, patterns, connections) stays shared. `init_contexts(N)` allocates N slots; `set_active_context(i)` swaps the active slot's state into the brain's live fields. Was meant to amortize thalamus-side parallelism overhead across more concurrent training streams.

**Surface.**
- `brain.rs`: `ContextState` struct; `contexts: Vec<ContextState>` and `current_context: usize` fields; `init_contexts`, `swap_to`, `set_active_context`, `num_contexts` methods.
- `memory.rs`: `#[derive(Clone)]` on `Memory` (required by `ContextState::clone`).
- `brain-napi`: `initContexts`, `setActiveContext` bindings.

**Decision: Roll back entirely.** Superseded by the spatial-processing design — each image becomes one frame across many parallel channels, not many serial frames in parallel contexts. No part of this is salvageable for spatial processing.

**Roll back checklist:** all symbols above. `Memory: Clone` can also be dropped (nothing else needs it).

**Test plan and acceptance.**
- `cargo test` green.
- `grep` confirms no remaining references to any of the above symbols.

---

## Project 7 — Frozen-context staged scan/predict 🔴 ★

**Goal.** Two-phase training flow: phase 1 scans the image's pixels (events fire, no action, no consensus); phase 2 predicts the digit on the now-frozen context (no event changes, action consensus runs, reward applied). Required toggling the pipeline's three sub-stages independently — event processing, action processing, and learning — and changing `process_frame` to take separate `events` and `actions` maps so phase 2 could force the action without re-driving the events.

**Surface.**
- `brain.rs`: `event_processing` / `action_processing` / `learning` fields on `Brain`; `set_processing_mode` method; **`process_frame(events, actions, rewards)` signature** (was `(inputs, rewards)`); refactored `build_frame` to accept forced actions; frozen-event branch inside `process_frame`.
- `column.rs`, `region.rs`: `action_processing` / `learning` fields; `set_action_processing` / `set_learning` methods; gating on default-action pre-wire (column).
- `thalamus.rs`: `set_action_processing` / `set_learning`; `learning: bool` plumbed into `process_level` → `get_level_tasks` → `get_level_corrections`. **Note:** these calls also gained a `frame_number` parameter — that piece belongs to Project 2 (warmup) and was already landed; only the `learning` half is rolled back here.
- `neuron.rs`: `learning: bool` parameter on `process_frame` / `recognize_patterns`; gates connection-learning and child-activation strengthening.
- `memory.rs`: `replace_active_neuron` (swap the forced action at age 0 during phase 2).
- `brain-napi`: `setProcessingMode` binding; `processFrame` signature change reverted.
- **Call-site reverts** — `process_frame(events, actions, rewards) → (inputs, rewards)` touches every app harness:
  - `apps/stocks/jobs/test.js`
  - `apps/stocks/jobs/multi-channel-test.js`
  - `apps/stocks/jobs/synthetic-cycle-test.js`
  - `apps/stocks/jobs/synthetic-extended-test.js`
  - `apps/text/jobs/test.js`
  - `apps/mnist/jobs/test.js`
  All `new Map()` adapter calls go away in this commit.

**Decision: Roll back entirely.** Spatial processing collapses the two phases into one frame across channels, so the staged flow has no purpose.

**Watch out:** keep the `frame_number` parameter on `get_level_tasks` / `get_level_corrections` — Project 2 (warmup gate) needs it. Don't strip the whole signature, just the `learning` half.

**Test plan and acceptance.**
- **T-stocks (Project 2)** — re-run; this commit reverts a ★ signature change that touched the harnesses, so re-confirm directional accuracy still matches the post-Project-2 baseline.
- **T-text (Project 2)** — same.
- `cargo test` green (the test helpers also pass `learning=true` today; verify they compile cleanly after revert).
- `grep` confirms no remaining references to `set_processing_mode`, `event_processing`, `action_processing`, `learning` field, `replace_active_neuron`.

---

## Project 8 — Supervised action learning 🔴

**Goal.** Replace the existing reward-shaped action path with direct supervised wiring: at each frame, collect every votable voter (every active non-suppressed neuron across levels), and additively accumulate `(voter → correct_action_id, distance, +reward)` connections. Frequency dominates: a voter wired to A twice and B once accumulates strength=2/reward=2 vs strength=1/reward=1, so A wins consensus by 4:1. Pairs with a static action-side learning rate (`action_alpha`) so the smoothing doesn't collapse frequency back to 1.

**Surface.**
- `brain.rs`: `learn()` (the public supervised entry); `infer()` (inference-only counterpart); `action_alpha` ctor param.
- `neuron.rs`: `learn_supervised_action` (additive accumulation); `action_alpha` field; action-aware branching inside `strengthen_connection` (static vs dynamic alpha).
- `column.rs`, `region.rs`: `action_alpha` propagated; `learn_action_connections` (route voter tasks per column); `collect_votes` (read-only vote sweep).
- `thalamus.rs`: public `collect_votes` and `learn_action_connections` (route across regions). This forced the internal `collect_votes` to be renamed `collect_level_votes` — revert the rename too.
- `memory.rs`: `get_votable_entries` (enumerate the voter pool).
- `brain-napi`: `learn`/`infer` bindings; `actionMode`/`actionAlpha` ctor options.
- `libs/node/src/run.js`: corresponding CLI flags.

**Decision: Roll back entirely.** The reverse-inference index from the neuron-reuse phase is the right way to make action learning targeted; this supervised shortcut was a stand-in.

**Test plan and acceptance.**
- `cargo test` green.
- After revert, verify `strengthen_connection`'s dynamic alpha branch is the only branch, and the action-side "never weakened" comment is restored.
- `grep` confirms no remaining references to `learn_supervised_action`, `action_alpha`, public `collect_votes`/`learn_action_connections`, `get_votable_entries`, `learn()`/`infer()` on Brain.

---

## Project 9 — MNIST apps refactor 🟢

**Goal.** The original `apps/mnist/encoder.js` was a single grayscale-byte encoder. The retinotopic-channels architecture in the roadmap needs per-pixel-position channels, plus row-aggregated variants for comparison, plus digit-label encoding. The branch split this into four focused encoders and rewrote `apps/mnist/jobs/test.js` around the new shape. `dump_image.js` was added to convert MNIST images into the text-pipeline format for cross-app validation.

**Surface.**
- `apps/mnist/encoders/digits.js` — digit label encoder.
- `apps/mnist/encoders/mnist_encoder.js` — grayscale-bucket encoder (the direct replacement for the old single file).
- `apps/mnist/encoders/pixel_channels_encoder.js` — one channel per pixel position.
- `apps/mnist/encoders/row_channels_encoder.js` — one channel per row.
- `apps/mnist/dump_image.js` — image → text-pipeline format.
- `apps/mnist/jobs/test.js` — rewritten around the new encoders (~600 lines diff, but mostly rewrite-in-place).
- `apps/mnist/encoder.js` — deleted (replaced by `encoders/`).

**Decision: Keep all.** These are the validation surface for spatial-processing's MNIST single-frame harness later in the roadmap. The encoders are independent of any brain-side changes being rolled back.

**Test plan and acceptance.**
- **T-encoder** — run each encoder against one canonical MNIST test image, confirm the output bit vector matches the expected encoding for the chosen quantization. Doesn't exercise the brain; just confirms the encoders themselves still work after the merge.

---

## Project 10 — Debug/trace tooling and fixtures ⚫

**Goal.** Investigation tooling produced during the MNIST scaling debug effort.

**Surface.**
- `apps/mnist/jobs/burst_trigger_trace.js`, `check_match_drift.js`, `classifier_voters.js`, `cold_replay_diff.js`, `connection_drift_trace.js`, `context_erosion_trace.js`, `convergence_trace.js`, `frame_activation_diff.js`, `inspect_pair.js`, `multi_digit_trace.js`, `parent_conn_dump.js`, `pattern_growth_trace.js`, `shared_pattern_trace.js`, `voter_analysis.js`.
- `apps/mnist/jobs/_sweep.sh`, `_compare_aliased.js`, `_inspect_shared_voter.js`, `_digit5_crossclass.js`, `_digit5_overlap.js`.
- `apps/mnist/jobs/_digit5/*.txt` (44 fixture files).
- `apps/mnist/jobs/_img*_bits.txt`, `_img*_visual.txt`.
- `apps/text/data/binary_*.txt`, `mnist_*.txt`, `sweep_*.txt` (one-line fixture dumps).
- `apps/mnist/performance.md` (v1 perf notes).

**Decision: Delete all.** Tooling that proves generally useful (e.g. `voter_analysis.js`-style harnesses) can be re-introduced on `dev` as a separate, intentional commit later.

**Verify before deleting.** `grep` `apps/text/jobs/test.js` for any reference to the `apps/text/data/*.txt` fixtures — if it loads any of them, those specific files need to stay.

**Test plan and acceptance.**
- `cargo test` green.
- All four stocks harnesses, text harness, and `apps/mnist/jobs/test.js` still run end-to-end (no broken references to deleted files).