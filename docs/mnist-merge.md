# MNIST Branch Reconciliation

---

## Why

The `mnist` branch has accumulated substantial work alongside experimental v1 moment-neuron scaffolding. Before any new structural work lands on top of it, the keepable parts must be merged into `main`. Merging late would compound conflicts with later phases.

This is a prerequisite for everything downstream — inference-scope experimentation, spatial processing, neuron reuse.

---

## How to read this document

Changes on `mnist` group into **coherent projects**. Each project listed below is a single commit onto `main`. Work them in order — top to bottom is the merge plan. Each project section contains:

- **Goal** — what the project was trying to accomplish.
- **Surface** — every file/symbol touched.
- **Decision** — Keep / Collapse.
- **Test plan and acceptance** — what verifies it when the commit lands.

Categorization shorthand: 🟢 keep · 🟡 collapse · ★ stocks-affecting.

Projects on `mnist` that are not listed below are **not being pulled** — they stay on the `mnist` branch and disappear when it's deleted. No work needed.

**Workflow.** work on `main` and pull each project's changes across from `mnist` one commit at a time, testing as you go. After all projects land, there is one **post-merge task** (delete the `mnist` branch).

---

## Supervised learn / inference-only infer 🟢 ★

**Goal.** Two entry points on Brain that the MNIST harness (and later the hippocampus path) needs:

- `learn(events, actions, rewards)` — supervised training step: process the frame with the correct action forced, learn connections, and additively wire active voter neurons to the correct action neuron.
- `infer(events)` — inference-only step: run pattern recognition and read out predictions without modifying any learned state. This is the 10k-test path.

At this phase the "voter pool" is just the active sensory neurons (no L1+ patterns exist without spatial processing). The same code wires correctly once spatial-processing patterns become voters.

**Surface (pulled from `mnist`).**
- `brain.rs`: `learn()` and `infer()` public methods; `action_alpha` ctor param.
- `neuron.rs`: `learn_supervised_action` (additive accumulation); `action_alpha` field; action-aware branching inside `strengthen_connection` (static alpha for action targets, dynamic alpha for events); `learning: bool` parameter on `process_frame` / `recognize_patterns` that gates connection-learning and child-activation strengthening.
- `column.rs`, `region.rs`: `action_alpha` propagated; `learn_action_connections` (route voter tasks per column); `collect_votes` (read-only vote sweep); `learning: bool` plumbed through `process_frame` calls.
- `thalamus.rs`: public `collect_votes` and `learn_action_connections` (route across regions); the internal `collect_votes` renamed to `collect_level_votes` to make room. `learning: bool` plumbed into `process_level` → `get_level_tasks` → `get_level_corrections`.
- `memory.rs`: `get_votable_entries` (enumerate the voter pool).
- `brain-napi`: `learn` / `infer` bindings; `actionAlpha` ctor option.
- `libs/node/src/run.js`: `--action-alpha` CLI flag if a harness uses it.

**Decision: Keep.** This is the pre-spatial-processing path for MNIST and the 10k test set. The supervised action wiring is also what hippocampus will use later.

**Do not pull.** The `mnist` branch also has phase-toggle machinery (`event_processing` / `action_processing` fields, `set_processing_mode`, the frozen-context branch in `process_frame`, `replace_active_neuron`) that was built around a staged scan/predict flow. None of it is needed once `learn()` / `infer()` exist — leave those symbols on the `mnist` branch.

**Signature note.** `process_frame(events, actions, rewards)` — the events/actions split comes with this project; `learn()` needs the `actions` arg. App harnesses gain `new Map()` adapter calls for the actions argument.

**Reward semantics.** The supervised path uses a static `action_alpha` and additive `(strength, reward)` accumulation, so frequency dominates ratio. This changes action-wiring math from the current dynamic-smoothed path — any harness running through `learn()` will produce different numbers than reward-shaped runs. Stocks/text harnesses that don't use `learn()` are unaffected. Note in commit message.

**Test plan and acceptance.**
- **T-mnist-learn-infer** — run the MNIST harness end-to-end: train via `learn()` on a small subset, evaluate on a held-out set via `infer()`, confirm `infer()` does not modify connection state (snapshot the brain before/after infer and diff).
- **T-stocks** — re-run; ★ change to action wiring (only affects paths that use `learn()`; reward-driven stocks harnesses should be unaffected). Update README baseline if anything shifts.
- `cargo test` green.

---

## MNIST apps refactor 🟢

**Goal.** The original `apps/mnist/encoder.js` was a single grayscale-byte encoder. The retinotopic-channels architecture in the roadmap needs per-pixel-position channels, plus row-aggregated variants for comparison, plus digit-label encoding. The branch split this into four focused encoders and rewrote `apps/mnist/jobs/test.js` around the new shape. `dump_image.js` was added to convert MNIST images into the text-pipeline format for cross-app validation.

**Surface.**
- `apps/mnist/encoders/digits.js` — digit label encoder.
- `apps/mnist/encoders/mnist_encoder.js` — grayscale-bucket encoder (the direct replacement for the old single file).
- `apps/mnist/encoders/pixel_channels_encoder.js` — one channel per pixel position.
- `apps/mnist/encoders/row_channels_encoder.js` — one channel per row.
- `apps/mnist/dump_image.js` — image → text-pipeline format.
- `apps/mnist/jobs/test.js` — rewritten around the new encoders (~600 lines diff, but mostly rewrite-in-place). Calls `learn()` / `infer()`, so **Supervised learn / infer** must land first.
- `apps/mnist/encoder.js` — deleted (replaced by `encoders/`).

**Decision: Keep all.** These are the validation surface for spatial-processing's MNIST single-frame harness later in the roadmap.

**Test plan and acceptance.**
- **T-encoder** — run each encoder against one canonical MNIST test image, confirm the output bit vector matches the expected encoding for the chosen quantization. Doesn't exercise the brain; just confirms the encoders themselves still work after the merge.
- **T-mnist-end-to-end** — run `apps/mnist/jobs/test.js`: train via `learn()`, evaluate via `infer()`, sanity-check the digit accuracy is non-random.
