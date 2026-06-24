# Context & Connection Refinement

This document is the **design** for refinement: the abstraction/generalization step that consolidates a
pattern toward the common core of the configurations it matches, instead of leaving it frozen at its
mint-time identity.
It is the **sharpening** stage of generalization in [neuron-reuse.md](./neuron-reuse.md) §1.3 — reuse
converges many error events onto one neuron; refinement (with decay) cleans the resulting class boundary on
both the source and target sides.

Refinement was removed in commit `8a17f4d` to prevent pattern-identity drift.
It is reintroduced behind a flag, with a reproducibility guard, and extended symmetrically from sources to
targets.

---

## 1. Why

A correction minted from one event encodes that event's exact configuration.
Without consolidation it stays frozen at mint-time identity: a one-off detector, not a general one.
Refinement lets a matched pattern move toward the structural core common to the configs it matches — the
missing generalization step that turns one-off corrections into general detectors and lets the hierarchy
climb past depth 2.

A pattern has two sides, and both refine under the same flag:

- **Sources (context)** — what activates the pattern.
- **Targets (connections)** — what it predicts and votes for.

---

## 2. Context Refinement (Sources)

On a matched pattern, update its context entries toward the common core of the configs it has matched:

- **Strengthen** context entries common to the match.
- **Add** novel entries observed in the match.
- **Weaken / delete** entries missing from the match.

So the pattern consolidates toward the common core of the configs it matches instead of staying frozen at
mint-time identity.

Steps:

- Add an **option** and put context refinement back into temporal processing.
- Add the same logic to **spatial** processing behind the same flag.

---

## 3. Target Refinement (Connections)

Symmetric to context.
Context refinement consolidates a pattern's **sources** — what activates it.
The **target connections** — what it predicts and votes for (event connections, action connections) — are
today refined only by strengthen-on-correct + mint-on-error (see
[error-driven-learning.md](./error-driven-learning.md) "Pattern Evolution"), not by the same
consolidate-toward-the-common-core logic.

Apply the symmetric operation to the target side. On a matched pattern:

- **Strengthen** common targets.
- **Add** novel targets observed.
- **Weaken / delete** targets that consistently fail to appear.

So the pattern's *output* generalizes toward the common core, not just its identity.
Both ends refine under the same flag.

### 3.1 Event connections

Apply target refinement to **event** (prediction) connections — clean, structural, symmetric to context
refinement.

### 3.2 Caveat: action connections

Action/reward connections are reward-smoothed and never-weakened by design (the forward *value* channel).
Structural weaken/delete on them would fight the reward signal.
So either:

- restrict structural target refinement to **event connections only**, or
- **guard** action connections so refinement never overrides reward-carried value.

This is the **open design question** for the target-refinement extension.

---

## 4. Reproducibility

Refinement mutates pattern identity across frames, so it must be controlled for reproducible eval:

- Refine **only during training**; **freeze** for eval — or consolidate in a separate pass.

---

## 5. Validation

- Test **MNIST** performance.
- Test **stock** performance.
- Test MNIST and stocks with **target refinement** on, both **independently** and **combined with context
  refinement**, to separate their contributions.
