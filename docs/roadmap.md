# Short-Term Roadmap

This is the **canonical ordered backlog**. Top-to-bottom is execution order. Each numbered section
is one workstream; sub-bullets are the concrete steps inside it.

---

## 1. Neuron Re-use (Wave-Front)

Theory in **[neuron-reuse.md](./neuron-reuse.md)**. Allocates capacity onto the error manifold
(residual-fitting). The processing **structure is kept** — spatial processing → apex handoff → temporal
processing — but each stage becomes a **settling wave**, stored levels are removed, only **base
sensory/action neurons carry coordinates**, and all corrections are coordinate-less with **footprints** (the
set of base neurons a correction covers) as the neighborhood primitive at every level. On that foundation,
**reuse applies at all distances** (d=0 and d>0) — the spatial/temporal asymmetry is gone, and the old
clustering/anchor problem dissolves (a shared correction takes the union footprint, no anchor).

Build order — each phase behind its own gate (theory §6):

1. **[Phase A — Wave-front](./neuron-reuse-wavefront.md)**: the foundation. Turn `process_spatial` and
   `process_temporal` into settling waves (structure unchanged); remove stored levels entirely;
   coordinate-less corrections; footprints for neighborhood in both waves; multi-depth memory.
   Rearchitecture — **not bit-exact** (characterized regression).
2. **[Phase B — Index](./neuron-reuse-index.md)**: **two** reverse connection indexes —
   `spatial_connection_index` (no distance) and `temporal_connection_index` (distance-keyed), mirroring the
   context-index split. Built, unit-tested, not yet consumed. Settles strength-candidacy.
3. **[Phase C — Frame](./neuron-reuse-frame.md)**: batched mint at all distances (group by (distance,
   observed-set), mint one coordinate-less correction with union footprint) **+ the multi-parent machinery**
   (refcounted reaping, multi-parent serialization, shared-neuron activation). Heaviest reuse phase.
4. **[Phase D — Final](./neuron-reuse-final.md)**: reuse lookup on top of C + cross-frame accrual, all
   distances. Light (reuse installs routing for next frame, so it needs no new same-frame tracking set).
5. **[Validation](./neuron-reuse-validation.md)**: MNIST spatial reuse + transfer, stocks full-pipeline +
   transfer, forget-rate long-run.

Decisions to settle before building: **strength-candidacy** for the reuse index (does a weak incidental
connection make a neuron a reuse candidate? — Phase B); **multi-parent activation model** (how a shared
neuron routing-matched from several parents at different depths in one sweep is held and processed — Phase
A/C/D). The clustering/anchor problem is gone (footprints + coordinate-less corrections), and the mint-frame
and cross-depth "decisions" turned out to be answered by the existing code (corrections install for next
frame, not the mint frame).

Knock-on: the wave-front removes channel-neighbor filtering (replaced by footprints), which removes the
cross-stream **isolation** knob parallel-stream learning relied on — that now needs a different primitive
(see [future-work.md](./future-work.md)).

---

## 2. Context & connection refinement

Design in **[refinement.md](./refinement.md)**. The missing abstraction/generalization step: on a matched
pattern, consolidate both its **sources** (context) and its **targets** (connections) toward the common
core of the configs it matches, instead of leaving it frozen at mint-time identity. This is what turns
one-off corrections into general detectors and lets the hierarchy climb past depth 2. Removed in commit
`8a17f4d` to prevent pattern-identity drift; reintroduced here behind a flag.

Steps:

1. **Context (sources).** Add an option and put context refinement back into temporal processing; add the
   same logic to spatial behind the same flag.
2. **Targets (connections).** Apply the symmetric operation to event (prediction) connections; settle the
   action-connection caveat (reward-smoothed, never-weakened) — restrict to event connections or guard
   action connections so refinement never overrides reward-carried value (open design question, see doc).
3. **Reproducibility guard.** Refine only during training, freeze for eval (or consolidate in a separate
   pass).
4. **Validate.** Test MNIST and stocks with context and target refinement, independently and combined, to
   separate their contributions.

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
