# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

## Status snapshot

- **Spatial processing is landed and validated.** The review dependency chain (fixes 1.1 → 2.1 →
  2.2 in [spatial-processing-review.md](spatial-processing-review.md)) is in the `spatial` branch and
  produced **95.73% on full MNIST** (28×28 binary, radius 2, merge 0.7, NB readout) — see
  [mnist-spatial-experiments.md](mnist-spatial-experiments.md). The design is in
  [spatial-processing.md](spatial-processing.md).
- **What remains** before merge-to-main: near-term engineering cleanups, the two open review fixes
  (1.2 snapshot restore, 1.3 temporal guard), stocks integration, and a code review. Everything after
  the merge is optimization and research.
- **Research stance:** runaway spatial depth is **not** clamped by design — no `MAX_LEVEL` caps. We
  prefer to see depth stall and study it over constraining potential algorithmic breakthroughs.

---

## 1. Near-term engineering

App-side and small core cleanups that should land before (or alongside) the merge. Independent of
each other unless noted.

### 1.1 Remove `--error-correct-rounds`

The discriminative second phase pushed train→99% with **no test gain** (the ceiling is
representational, not readout/training — see experiment §9). Drop the option and the
`runErrorCorrection` path.

### 1.2 Eval-train refactor

- **Wire-only training by default:** `runTraining` calls `resetContext` + `processFrame` only — drop
  the per-image `infer()` / decode / tally. Faster, and it kills the prequential per-episode number.
- **Always run the frozen `runTrainEval()` at the end** (delete the `evalTrain` flag/config).
- **Add opt-in `--eval-train-per-episode`** that runs a frozen pass after each episode (off by
  default).
- Clean up `showResults` / `logTrainingPass` that depended on the prequential per-episode accuracy.

### 1.3 Fix the Ctrl+C bug

Shutdown handling is not working — `isShuttingDown` does not cleanly interrupt runs. Diagnose and
fix so long jobs can be cancelled without leaving the brain half-written.

---

## 2. Backups / imports / exports (Phase 4)

Make every persistence path round-trip d=0 connections and the spatial structure.

- **Fix review 1.2 — snapshot restore silently destroys spatial structure.** Serialization was
  updated for spatial but restore was not: spatial levels are dropped (every neuron comes back at
  `spatial_level = 0`) and spatial routing entries land on the temporal side. Add a spatial/temporal
  discriminator to `SerializedChild` / `SerializedContextRef`, restore each to the proper side, and
  add `spatial_level` to `SnapshotNeuronEntry`. Full detail in
  [spatial-processing-review.md §1.2](spatial-processing-review.md) and the Phase 4 acceptance test in
  [spatial-processing.md §6](spatial-processing.md).
- **Test imports / exports / backups** — implement the design's round-trip acceptance test (train →
  snapshot → restore → identical next frame), plus the DB import/export round-trip and the
  pre-spatial backward-compat load.

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
