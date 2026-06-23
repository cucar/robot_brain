# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

## 1. Neuron Re-use

Theory in **[neuron-reuse.md](./neuron-reuse.md)**. Allocates capacity onto the error manifold
(residual-fitting).

Build order — each phase behind its own gate (theory §6):

1. **[Phase A — Levels](./neuron-reuse-levels.md)**: drop the persistent `neuron_spatial_levels` map;
   derive spatial level from activation. Bit-exact refactor + backup format bump.
2. **[Phase B — Index](./neuron-reuse-index.md)**: reverse inference index (target → distance → sources)
   over both connection stores. Built, unit-tested, not yet consumed.
3. **[Phase C — Frame](./neuron-reuse-frame.md)**: batched mint — group errors by (distance, neighborhood),
   mint one, wire all co-failers. First behavior change.
4. **[Phase D — Final](./neuron-reuse-final.md)**: reuse lookup on top of batched mint + the
   `fired_this_frame` / `correction_wired_this_frame` tracking sets.
5. **[Validation](./neuron-reuse-validation.md)**: MNIST transfer, stocks full-pipeline, forget-rate
   long-run.

Two semantics to settle before C/D: mint-frame vs reuse-frame inhibition window (Phase C DECIDE-THIS #1),
refractory vs cross-level injection (Phase D DECIDE-THIS #2).

---

## 2. Re-introduce context refinement

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

**Refine targets (connections) too, symmetric to sources (context).** The refinement above
consolidates a pattern's **context (sources)** — what activates it. A pattern has a second side: its
**target connections** — what it predicts and votes for (event connections, action connections). Today
those are refined only by strengthen-on-correct + mint-on-error (see
[error-driven-learning.md](./error-driven-learning.md) "Pattern Evolution"), not by the same
consolidate-toward-the-common-core logic. Apply the symmetric operation to the target side: on a matched
pattern, **strengthen** common targets, **add** novel targets observed, **weaken/delete** targets that
consistently fail to appear — so the pattern's *output* generalizes toward the common core, not just its
identity. Both ends refine under the same flag.

- Apply target refinement to **event** (prediction) connections — clean, structural, symmetric to
  context refinement.
- **Caveat for action connections:** action/reward connections are reward-smoothed and never-weakened
  by design (the forward *value* channel). Structural weaken/delete on them would fight the reward
  signal. So either restrict structural target refinement to event connections, or guard action
  connections so refinement never overrides reward-carried value. Flagged as the open design question
  for this extension.
- Test MNIST and stocks with target refinement on, both independently and combined with context
  refinement, to separate their contributions.

---

## 3. Documentation & Publish

- **Update all documentation** — sync docs with the current architecture post-Rust migration; update README demos and examples.
- **npm package** — prepare and publish to the registry.
- Refactor repo to have the apps be able to use npm or local packages.

---

## 4. Action composition

See **[action-composition.md](./action-composition.md)**.

**Inference-scope gate resolved.** The inference-scope experiment that previously gated this has been
concluded and removed (see [inference-level.md](./inference-level.md)): `base` is the ground-truth-anchored
default for *events*, and the compositional ambition relocates to the action side. Composition here does
**not** depend on events predicting events — it rests on **action→action** backward composition plus
**apex event ↔ apex action** coupling, anchored by reward and execution rather than by an event-side
cascade. The `base`-for-events conclusion therefore leaves this intact; the gate is removed, not failed.

Grow the action hierarchy by the same machinery as events, run backward in time (`d<0`): after an
action fires it binds to its antecedents and mints an action pattern when backward inference error
crosses the existing Welford threshold (**mint by structure, survive by value** — the advantage test
lives in the Death Ledger, not the mint gate). Plus action-moment neurons (the `d=0` action-spatial
pass), the per-frame offset pipeline, commitment / call-stack arbitration, and a later reverse-replay
credit accelerator.

The test harness is the long pole: no current domain exercises action composition — plan to convert the
**text channel to action-based output (a chatbot)** so emitting tokens becomes a composable action
sequence. Open questions and full dependencies are in the doc.

---

## 5. Global rewards

See **[global-rewards.md](./global-rewards.md)**. **Independent** of action composition (§4) — the reward
distribution policy holds with or without action composition, and can be decided separately. It meets
composition at exactly one point: reward credits the **apex active action**, not base neurons.

Move from the current **last-frame** policy to **per-span global rewards**: distribute reward back
across the apex actions active throughout the context span, weighted by **linear** (not exponential)
decay so distant frames keep nonzero credit under long-latency reward. Watch the length-bias assumption
(see doc) — span-normalized application may be needed if pattern spans vary widely.

Rewards are **global** in the sense that they apply across all channels, not just the channel that
produced the action, with a clean separation between immediate per-frame rewards and delayed
outcome-based rewards. This is foundational: everything downstream (imitation rewards, hippocampal
experimentation) benefits from the richer reward signal. The current per-frame system is sufficient for
stocks and MNIST but becomes a bottleneck for more complex behaviors.

Reference: https://claude.ai/share/f9732e46-a95c-44d2-8dee-b7217392834c

---

## 6. Calculate up/down accuracy separately

Report directional accuracy (up vs down) independently to identify prediction bias.

---

## 7. Neuron Limits

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

## 8. Exponential Temporal Binning Test

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
