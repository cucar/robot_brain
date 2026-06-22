# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

## 1. Optimize MNIST

Push past the 96.44% joint result. Current best results and runnable demos live in
[mnist-demos.md](mnist-demos.md). Three tests remain:

- **Radius 3 at 28×28** (error 0.1 / merge 0.9) — if it beats radius 2, re-run the two tests below at
  radius 3 instead of radius 2.
- **Error 0.2 / merge 0.8.**
- **Error 0.3 / merge 0.7.**

Each is one train + frozen-eval pair (the train -> evaluate workflow in mnist-demos.md); episodes
plateau after one pass, so a single episode suffices.

---

## 2. Inference Level Experiment

See **[inference-level.md](./inference-level.md)**.

Pick the inference scope rule by experiment on the stocks pipeline, comparing:

- **All levels**
- **Same level**
- **Lower level**

The decision propagates into spatial processing, and gates the action-composition work in §8.

---

## 3. Neuron Re-use

See **[neuron-reuse.md](./neuron-reuse.md)**.

Allocates capacity onto the error manifold (residual-fitting). **Scope is reduced for now:** do
**just the first part** — *not* the reverse inference index and merge. Then:

- Test MNIST.
- Test stocks.

---

## 4. Re-introduce context refinement

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

## 5. Documentation & Publish

- **Update all documentation** — sync docs with the current architecture post-Rust migration; update README demos and examples.
- **npm package** — prepare and publish to the registry.
- Refactor repo to have the apps be able to use npm or local packages.

---

## 6. Action composition

See **[action-composition.md](./action-composition.md)**. **Gated on §5** — composition assumes levels
meaningfully predict levels, so it waits on a non-`base` winner from the inference-scope experiment.

Grow the action hierarchy by the same machinery as events, run backward in time (`d<0`): after an
action fires it binds to its antecedents and mints an action pattern when backward inference error
crosses the existing Welford threshold (**mint by structure, survive by value** — the advantage test
lives in the Death Ledger, not the mint gate). Plus action-moment neurons, commitment / call-stack
arbitration, and a later reverse-replay credit accelerator.

The test harness is the long pole: no current domain exercises action composition — plan to convert the
**text channel to action-based output (a chatbot)** so emitting tokens becomes a composable action
sequence. Open questions and full dependencies are in the doc.

---

## 7. Global rewards

See **[global-rewards.md](./global-rewards.md)**. **Independent** of §5 and §8 — the reward distribution
policy holds with or without action composition, and can be decided separately. It meets composition at
exactly one point: reward credits the **apex active action**, not base neurons.

Move from the current **last-frame** policy to **per-span global rewards**: distribute reward back
across the apex actions active throughout the context span, weighted by **linear** (not exponential)
decay so distant frames keep nonzero credit under long-latency reward. Watch the length-bias assumption
(see doc) — span-normalized application may be needed if pattern spans vary widely.

---

## 8. Calculate up/down accuracy separately

Report directional accuracy (up vs down) independently to identify prediction bias.

---

## 9. Neuron Limits

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

## 10. Exponential Temporal Binning Test

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