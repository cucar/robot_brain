# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

## 1. Near-term engineering in MNIST test.js

App-side and small core cleanups that should land before (or alongside) the merge. Independent of
each other unless noted.

### 1.1 Eval-train refactor

- **Always run the frozen `runTrainEval()` at the end** (delete the `evalTrain` flag/config).
- **Add opt-in `--eval-train-per-episode`** that runs a frozen pass after each episode (off by default).

### 1.2. Persistence / Backup / Import-Export Updates

**Goal:** Make sure every persistence path round-trips d=0 connections and the new spatial_level field.

Make every persistence path round-trip d=0 connections and the spatial structure.

Serialization was updated for spatial but restore was not, so a save/load round-trip corrupts silently:

- Spatial levels are dropped (every neuron comes back at `spatial_level = 0`) and spatial routing entries land on the temporal side. Add a spatial/temporal
  discriminator to `SerializedChild` / `SerializedContextRef`, restore each to the proper side, and add `spatial_level` to `SnapshotNeuronEntry`. 
- **Spatial levels are dropped.** `SnapshotNeuronEntry` carries only the temporal `level`
  ([thalamus.rs:174-179](../brain/brain-core/src/thalamus.rs)); `get_snapshot`
  ([thalamus.rs:1851-1871](../brain/brain-core/src/thalamus.rs)) never reads
  `neuron_spatial_levels`, and `restore_snapshot` clears the map via `reset()`
  ([thalamus.rs:1938](../brain/brain-core/src/thalamus.rs)) and never repopulates it. Every
  neuron comes back at `spatial_level = 0`.
- **Spatial routing tables land on the temporal side.** `serialize_children` and
  `serialize_context_refs` flatten spatial and temporal entries into one undiscriminated list
  ([neuron.rs:404-443](../brain/brain-core/src/neuron.rs); spatial contexts marked
  `distance: 0`). `load_neuron` ([column.rs:600-645](../brain/brain-core/src/column.rs)) restores
  through `add_child` / `add_context` / `add_context_ref` — the back-compat shims that route
  **unconditionally to the temporal structures**
  ([neuron.rs:944-954](../brain/brain-core/src/neuron.rs)). After restore, every spatial routing
  entry and context ref sits in the temporal tables with d=0 contexts: spatial hierarchy gone,
  temporal tables polluted.
- Connections and Welford error stats *do* round-trip correctly (`create_connection` routes
  d=0 to `spatial_connections`; `age == 0` stats route to the spatial bucket,
  [column.rs:634-642](../brain/brain-core/src/column.rs)) — which deepens the false sense of
  safety.

**Fix:** add a spatial/temporal discriminator to `SerializedChild` and `SerializedContextRef`
(or split them into separate fields), restore each to the proper side; add `spatial_level` to
`SnapshotNeuronEntry`, populate in `get_snapshot`, restore into `neuron_spatial_levels`.
Then implement the Phase 4 acceptance test from the design doc (train → snapshot → restore →
identical next frame), which would have caught all of this.

#### Code touched

- `brain/brain-core/src/backup.rs` — extend `SerializedConnection` / `SerializedNeuron`:
    - `connections[0]` distance=0 entries must round-trip. Verify the existing format iterates the full `connections` vec rather than skipping slot 0.
    - The existing `neuron.level` field becomes the temporal intrinsic level. Add `neuron.spatial_level` alongside it. For pre-spatial-era neurons loaded from older snapshots, default `spatial_level` to 0 (they sit at the base of the spatial hierarchy because no spatial grouping built them).
    - The per-frame `spatial_level_index` / `temporal_level_index` are not persisted — same rule as today's `level_index`.
    - The inherited (channel, dimension, coordinate) of a correction (§4.4) needs no new snapshot field — it is derivable on restore by walking `neuron_parents` to the L0 ancestor (parent ids are already persisted). Deeper levels chain: an L2 inherits from its L1 parent, which anchors at the original L0 coordinate.
- DB import/export apps under `apps/` — sweep for code that assumes connections start at distance 1, and for code that reads/writes the single `neuron.level` field.

#### Acceptance

- Round-trip test: train a brain on a few MNIST images (enough to mint d=0 corrections across multiple spatial levels), snapshot, restore, verify (a) all d=0 connections present with correct strengths, (b) all correction neurons present and routable with their spatial_level intact, (c) one more frame of training on the restored brain produces identical results as on the original.
- DB import/export apps: run a round-trip on a stocks brain with d=0 connections; verify byte-identical export after `import → export`.
- Backward-compat test: load a pre-spatial stocks snapshot, verify all neurons get `spatial_level = 0`, run one frame, confirm output matches what the pre-spatial code would have produced (modulo the new d=0 work, which will start forming immediately).

- **Test imports / exports / backups** — implement the design's round-trip acceptance test (train →
  snapshot → restore → identical next frame), plus the DB import/export round-trip and the
  pre-spatial backward-compat load.

### 1.3 Fix the Ctrl+C bug

Shutdown handling is not working — `isShuttingDown` does not cleanly interrupt runs. Diagnose and
fix so long jobs can be cancelled without leaving the brain half-written.

---

## 3. Merge spatial branch to main

- **Review all the code** on the `spatial` branch.
- Merge once the near-term cleanups, the two review fixes, and stocks parity are in.

---

## 4. Optimize MNIST

Push past the 95.73% capstone and produce the publishable ablations. Full experiment log and the
detailed step list live in [mnist-spatial-experiments.md](mnist-spatial-experiments.md).

- **Update the experiments document** — prune obsolete tests (done).
- **Radius 3 at 28×28** — the radius-2 optimum was tuned at lower res; check radius 3 at full res.
- **Error / merge threshold re-tune at 28×28 (radius x)** — sweep the paired corners
  **0.1 / 0.9, 0.2 / 0.8, 0.3 / 0.7** and re-pick.
  - Anchor: `node apps/mnist/jobs/test.js --image-size 28 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --episodes 3 --error-mode static --error-threshold 0.1 --merge-threshold 0.9`
- **Literature-standard Split-MNIST** — the continual-learning headline. Details below.

### Context: the Naive-Bayes ladder

The sensory-only app is Naive Bayes by construction; the architecture must clear it *with a different
mechanism* (no backprop, no labels, demand-driven conjunctive features) for the result to mean
anything. Reference rungs on identical preprocessing:

- **Naive Bayes (independent pixels): ~83–84%** — the floor and the "did we just reimplement NB?"
  check.
- **Logistic regression / linear classifier: ~92%** — pixel-based but jointly weighted.
- **k-NN: ~97%** — pure template matching.
- **Simple MLP / small CNN: 98–99%+** — the gradient-trained ceiling, not the target.

The interesting result is **matching or beating the jointly-trained linear model (~92%) without joint
training, labels, or backprop**. The 95.73% capstone clears that rung and approaches k-NN — the climb
came mechanism by mechanism (spatial hierarchy +13pp over pixel-NB, NB readout +5pp over consensus,
radius 2 the unlock).

### Literature-standard Split-MNIST (class-incremental continual learning)

The **headline experiment for external positioning** — class-incremental, the hardest CL regime,
where naive backprop nets collapse to ~20%.

- **Use the standard 5 tasks × 2 classes** (0/1, 2/3, 4/5, 6/7, 8/9), citing van de Ven & Tolias 2019
  and Hsu et al. 2018 — **not** the current 10 tasks × 1 class. This makes our ~90–91% line up
  apples-to-apples against the cited floor.
- **No task IDs at train or test time.** Strict sequential training, one task's data at a time. Action
  space stays 10 digits throughout.
- **MLP class-incremental baseline:** a vanilla MLP trained under the *identical* split protocol
  (digits sequentially, test on all 10), expected to collapse to ~20%. ~30-line Python script,
  separate from the brain. This turns "we don't forget" into "we don't forget *where standard nets
  catastrophically do*" — the punch line.
- **Re-run our stack on the standard 5×2** at the new optimum (28×28, radius 2, merge 0.7, NB, full
  train, full 10K test) for the headline split number to sit beside the 95.73% joint.

**Why it should hold architecturally:** patterns formed for digits 0/1 have disjoint context
fingerprints from 2/3, so they don't fire — and aren't modified — during later tasks. Higher-level
patterns decay slower than lower ones, so consolidated digit patterns outlive subsequent tasks. The
empirical finding so far is *graceful, recency-biased degradation* (the earliest class's shared voters
get overwritten), not zero forgetting — still far above the ~20% backprop floor. Additive/local
learning gives stability; the hierarchy gives accuracy.

**Evaluation:** final 5×5 retention matrix, average accuracy after Task 5 (headline), and a forgetting
metric (max-ever minus final per task). Cite the literature baselines (~20% naive, ~20–25% EWC,
70–90% replay, ~98% joint upper bound) rather than re-running them.

---

## 5. Temporal channel inheritance experiment

Fix 2.2 gave *spatial* corrections their parent's full (channel, dimension, coordinate). Open
question: **should temporal corrections inherit channels the same way?** Symmetry argues yes, but
temporal corrections group across channels by design, so it may not apply cleanly. Test on MNIST and
stocks. Tracked in [spatial-processing.md §8](spatial-processing.md).

---

## 6. Inference Level Experiment

See **[inference-level.md](./inference-level.md)**.

Pick the inference scope rule by experiment on the stocks pipeline, comparing:

- **All levels**
- **Same level**
- **Lower level**

The decision propagates into spatial processing.

---

## 7. Neuron Re-use

See **[neuron-reuse.md](./neuron-reuse.md)**.

Allocates capacity onto the error manifold (residual-fitting). **Scope is reduced for now:** do
**just the first part** — *not* the reverse inference index and merge. Then:

- Test MNIST.
- Test stocks.

---

## 8. Re-introduce context refinement

Removed in commit `8a17f4d` to prevent pattern-identity drift. On a matched pattern, **strengthen**
common context entries, **add** novel, **weaken/delete** missing — so a pattern consolidates toward
the common core of the configs it matches instead of staying frozen at mint-time identity. This is the
missing abstraction/generalization step that would turn one-off corrections into general detectors and
let the hierarchy climb past depth 2.

- Add an **option** and put it back into temporal processing.
- Add the same logic to **spatial** processing behind the same flag.
- Guard for reproducibility: refine only during training, freeze for eval (or consolidate in a
  separate pass).
- Test MNIST performance.
- Test stock performance.

If the headline numbers land here:

- **LinkedIn post.**
- **Email investors.**

---

## 9. Calculate up/down accuracy separately

Report directional accuracy (up vs down) independently to identify prediction bias.

---

## 10. Neuron Limits

### Max neuron count hyperparameter
- Add a configurable cap on neuron count per region/column.

### Capacity enforcement
- When capacity is reached, stop learning new patterns.
- Once forgetting frees space, resume learning automatically.

### Overflow warning
- Emit a warning when capacity is hit so the operator knows the brain is saturated.
- Test that learning resumes correctly after decay opens capacity.

> Note: this is an opt-in *capacity* cap on neuron count, distinct from spatial-depth `MAX_LEVEL`
> caps — which we deliberately do **not** add (see Status snapshot).

---

## 11. Exponential Temporal Binning Test

Implement the cortical temporal binning scheme in
[experiment-temporal-binning.md](./experiment-temporal-binning.md).

Higher-level patterns currently store context at exact frame distances — meaningless precision at
their timescale. Exponential bins give every level the same number of bins but scale bin width with
level, letting higher-level patterns represent long-range temporal relationships without context
explosion.

- Context struct stores bin index instead of exact distance.
- Bin conversion: `distanceToBin(distance, level, contextLength, numBins)`.
- Pattern matching and voting use bin-space comparison.
- **Validation:** level-1 patterns behave nearly identically to current (regression); higher-level
  patterns form with fewer, coarser context entries; accuracy on existing benchmarks does not
  regress.

---

## 12. Documentation & Publish

- **Update all documentation** — sync docs with the current architecture post-Rust migration; update
  README demos and examples.
- **Fashion-MNIST** on the same stack — generality evidence for the paper.
- **Prior-art differentiation writeup** — HTM/Numenta, ART/Grossberg, predictive coding, growing
  neural gas.
- **npm package** — prepare and publish to the registry.
