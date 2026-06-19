# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

## 1. Temporal channel inheritance experiment

Spatial corrections have their parent's channel, dimension, and coordinate. 
Open question: **should temporal corrections inherit channels the same way?** 
Symmetry argues yes, but temporal corrections group across channels by design, so it may not apply cleanly. 
Test on MNIST and stocks.

---

## 2. Inference Level Experiment

See **[inference-level.md](./inference-level.md)**.

Pick the inference scope rule by experiment on the stocks pipeline, comparing:

- **All levels**
- **Same level**
- **Lower level**

The decision propagates into spatial processing.

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

## 5. Calculate up/down accuracy separately

Report directional accuracy (up vs down) independently to identify prediction bias.

---

## 6. Neuron Limits

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

## 7. Exponential Temporal Binning Test

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

## 8. Documentation & Publish

- **npm package** — prepare and publish to the registry.
- **Update all documentation** — sync docs with the current architecture post-Rust migration; update README demos and examples.