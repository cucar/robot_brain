# Inference Level Experiment — Concluded, then Superseded

> **Status: concluded 2026-06-22 (`base` as the event-tower default); superseded 2026-07-13 by the UCAR design
> ([algorithm.md](./algorithm.md)), which adopts same-level inference on every axis.** The analysis below was
> correct under its premises, and each premise has since been removed: minting no longer couples prediction-target
> to correction-target (subsumption accounting decouples them), readout is owned by the payload carve-out (action,
> reward, and label channels stay base-wired from every level), and bootstrap starvation is handled by the rent
> economics. What stands is the analysis's core invariant, which UCAR keeps: **no level ever trains against
> another level's predictions** — same-level targets are future or present *observed* activations, not the cascade of
> guesses this document rightly rejected. Implementation remains `base` until the staged port reaches the temporal
> tower (spatial is validated first, then events, then actions). The original reasoning and experiment spec are
> preserved below for the record.

This document originally proposed an empirical sweep over which neurons a level-L neuron predicts over
(`base` / `same-level` / `level-below` / `all-levels`), to be picked on the stocks pipeline and to gate
[action-composition.md](./action-composition.md). We concluded it conceptually instead. What follows is
the reasoning.

---

## The reframe: it is one axis, not two

The experiment was framed as "which neurons does a level-L neuron predict over" plus an unstated second
question, "how do the levels' predictions aggregate into one answer." Those are **not** two free axes.

Minting is **error-driven correction**: a new pattern is born precisely because some neuron's inference
was wrong, and it captures the pattern that should have been predicted. So **whatever a neuron predicts
over is, by construction, the same substrate its corrections are minted in** — prediction-target, error,
mint-target, and correction-target are one substrate. They cannot be chosen independently.

So the real question is not "what does a neuron predict over." It is:

> **What does a correction correct — i.e., in what substrate is the error measured?**

The mint mechanism is identical in every variant; only that substrate moves.

---

## The variants under that single question

A level-2 neuron is born from a level-1 neuron's error, so *where* the error is measured decides
everything.

- **`base`** — every level's error is measured against **sensory**. A level-2 correction is minted
  because a prediction *of sensory* was wrong, and it predicts *sensory* better. Every level, top to
  bottom, is anchored to **ground truth**. This is the whole reason `base` is self-consistent and easy:
  one substrate, and it is reality. Nothing composes, but nothing drifts.

- **`level-below` (the cascade)** — for it to stay self-consistent under "mint = correct the inference,"
  a level-2 neuron's error must be measured at **level-1** ("did the level-1 token I predicted actually
  fire?"), not at sensory. So level-1 is corrected against level-0 (reality), level-2 against
  **level-1's prediction**, level-3 against level-2's. **Only the bottom rung touches ground truth.**
  This composes (predictive coding), but higher levels chase lower-level predictions, so error
  **compounds** up the tower: a level-2 neuron is trained to predict a target that was itself a guess.

- **`same-level`** — a level-2 neuron predicts/corrects level-2 peers, but it was *born from a level-1
  error*. Its birth substrate and its prediction substrate do not even match. Coherent only if grounding
  to reality happens entirely separately at readout — the cascade with an extra disconnect — plus the
  bootstrapping starvation (early on, few level-L neurons to predict over).

- **`all-levels`** — error measured against everything active, sensory included. Most anchored, but a
  future moment is counted as sensory *and* as the level-1 token that grounds to the same sensory *and*
  as the level-2 token — double-counting. Coherent only with apex/subsumption dedup (highest non-subsumed
  neuron per region, constituents removed — the collapse the spatial path already computes).

---

## The actual trade-off

It was never cost vs. accuracy. It is **what anchors each level's correction**:

| Variant | What anchors each level | Consequence |
|---|---|---|
| **base** | ground truth at every level | no drift, no composition — parallel experts shouting at sensory |
| **level-below** | only the bottom rung sees reality; higher levels correct against lower **predictions** | composition, but **compounding-error risk** |
| **all-levels** | everything incl. ground truth, but **redundantly** | anchored but double-counts evidence |
| **same-level** | birth/prediction substrate mismatch | composition with an extra disconnect + starvation |

The moment levels predict levels, higher levels **stop being anchored to reality** and start being
anchored to the level below. That is the price of composition. `base` avoids it by anchoring everyone to
sensory — which is exactly *why* `base` cannot compose.

---

## Why the experiment as specified could not have answered this fairly

The original spec measured error/readout for every variant against the same target. If that target is
sensory — `base`'s home court — then `same-level` and `level-below` are crippled by construction (their
natural error substrate is a higher level, not sensory), `base` "wins" by measurement artifact, and that
spurious win would then propagate to kill action composition. To give the non-base variants a fair shot
you would have to build a per-level error substrate **and** an apex-grounded readout — substantial new
machinery — just to discover what the analysis above already shows: for the **event** tower, the
compounding-error problem has no anchor, so `base` stands.

---

## Conclusion / Decision (superseded — see status note at top)

**`base` is the default for the event tower.** Events are evaluated against the next sensory frame; that
is the only place `d>0` ground truth lives, so anchoring every level to sensory is correct and the
cascade's compounding error buys nothing the event tower can cash. "Levels infer levels" does **not** pay
for events.

**The compositional ambition relocates to the action side**, where it has an anchor the event cascade
lacks. Action composition (see [action-composition.md](./action-composition.md)) does not rely on events
predicting events. It rests on:

- **action→action backward composition** — chunks form from backward inference over *actions only*; and
- **apex event ↔ apex action coupling** — the existing forward value vote.

Both are anchored to reality the way `base` events are: emitted primitives are **real** (execution
grounds the action hierarchy at the bottom, the way sensory grounds events), and the **reward filter**
prunes the backward graph toward causality. So the `base`-for-events conclusion **leaves action
composition intact** — it removes the gate rather than failing it. The old worry ("if `base` wins, levels
infer levels is undercut") dissolves once composition is action→action plus apex coupling rather than
events predicting events.

**Net:** the four-variant sweep is abandoned; `base` is the event-tower default; composition is an
action-side mechanism; action-composition is no longer gated on this experiment.

---

## Original experiment design (preserved, not run)

*Kept for the record. None of the runs below were executed.*

### Four variants

| Rule | `new_active_neurons` passed to a level-L neuron processing age=d | Hypothesis |
|---|---|---|
| `base` (today) | All L=0 neurons active at age=d | Cheapest; current behavior |
| `same-level` | L-resident neurons active at age=d | Each level predicts within its own abstraction; bootstrapping risk |
| `level-below` | (L−1)-resident neurons active at age=d | Classic cortical hierarchy rule |
| `all-levels` | All neurons active at age=d, any level | Maximum reuse; risk of correlated double-counting |

### Phase 1 plan (not run)

Gate `new_active_neurons` by scope rule in `thalamus.rs::dispatch_frame`, add
`inference_scope: InferenceScope` to `BrainOptions` (default `Base`), expose on the JS brain options, and
measure directional accuracy / ROI / neuron+connection counts / runtime / memory on the stocks pipeline
across `base` / `same-level` / `level-below` / `all-levels`, with a per-frame p99 runtime budget of 3×
`base` and a 1pp noise threshold. The pre-registered tie-break order was p99 runtime → connection count →
memory, defaulting to `base`.

This plan is superseded by the conclusion above.
