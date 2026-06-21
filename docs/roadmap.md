# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

### 1. Fix the Ctrl+C bug in MNIST test.js

Shutdown handling is not working — `isShuttingDown` does not cleanly interrupt runs. Diagnose and
fix so long jobs can be cancelled without leaving the brain half-written.

---

## 2. Merge spatial branch to main

- **Review all the code** on the `spatial` branch.
- Merge once the near-term cleanups, the two review fixes, and stocks parity are in.

---

## 3. Split-MNIST MLP baseline

A vanilla MLP trained under the *identical* class-incremental protocol — the 5 tasks x 2 classes,
strictly sequential, tested on all 10 classes, no task IDs — expected to collapse to ~20%. A ~30-line
Python script, separate from the brain. This is the contrast figure for the Split-MNIST result in
[mnist-demos.md](mnist-demos.md): it turns "we don't forget" into "we don't forget *where standard
nets catastrophically do*." Pair it with the cited literature baselines (in mnist-demos.md), not the
naive-MLP collapse alone.

---

## 4. Optimize MNIST

Push past the 96.44% joint result. Current best results and runnable demos live in
[mnist-demos.md](mnist-demos.md). Three tests remain:

- **Radius 3 at 28×28** (error 0.1 / merge 0.9) — if it beats radius 2, re-run the two tests below at
  radius 3 instead of radius 2.
- **Error 0.2 / merge 0.8.**
- **Error 0.3 / merge 0.7.**

Each is one train + frozen-eval pair (the train -> evaluate workflow in mnist-demos.md); episodes
plateau after one pass, so a single episode suffices.

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

**Action pattern composition** (Part 2 of the doc) hangs off this experiment. Once the scope rule is
settled — composition assumes levels meaningfully predict levels, so it is **gated on a non-`base`
winner** — design the outcome-gated action hierarchy: action-pattern/action-moment neurons, mint by
event-level outcome + reward advantage (never frequency), event-context indexing for the high-event ->
high-action link, and commitment arbitration. Reverse-replay credit is a later accelerator, not a
precondition. The test harness is the long pole: no current domain exercises it — plan to convert the
**text channel to action-based output (a chatbot)** so emitting tokens becomes a composable action
sequence. Open questions and full dependencies are listed in the doc.

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
