# Global Rewards (Reward Distribution Policy)

How a reward signal is distributed backward over the actions that were active. This is **independent** of the inference-scope experiment ([inference-level.md](./inference-level.md)) and of action composition ([action-composition.md](./action-composition.md)) — it would hold with or without them, and can be decided separately. It meets action composition at exactly **one** point: reward is applied to the **apex active action**, not to base action neurons (below).

> **Status: design.** The last-frame policy is current; per-span global rewards are planned.

## Independence

The reward distribution policy is a separate concern from how the action hierarchy forms. The policy could be last-frame or global, per-frame or per-span, linearly or exponentially weighted — none of that is determined by the [action-composition](./action-composition.md) design, and composition does not depend on which is chosen. Keep them decoupled; do not let composition assumptions leak into the reward policy or the reverse.

## The one interface point: reward the apex, not the base

The single change [action composition](./action-composition.md) imposes on the reward policy: credit lands on the **apex active action** — the highest action pattern currently in control of the channel — falling back to the base primitive only when no higher action covers that frame.

- The apex is the unit that was actually **in control**. A committed chunk holds the channel and suppresses its constituents' votes; crediting the base primitives would credit suppressed subordinates instead of the decision-maker.
- It keeps **value at the same granularity as structure**. Composition builds structure at the chunk level; reward must accrue there too, or selection never sees it.
- Crediting primitives would reinforce primitive-level policy — the forward-frequency calcification composition avoids. Constituents still get credit, but only where *they* are the apex (no chunk covering them) — correct: a primitive is good *in this skill*, not universally.

This degrades gracefully: before any chunk exists the apex *is* the primitive, so the same rule holds across all of development.

## Policies

| Policy | Granularity | Status | Distribution |
|---|---|---|---|
| **Last-frame** | per-frame | **current** | Credit the apex action active in the immediately preceding frame. |
| **Global rewards** | per-span | **planned** | Distribute the reward back across the apex actions active throughout the context (context-decay span), weighted by **linear** decay. |

## Notes

- **Linear, not exponential.** Linear keeps nonzero credit on distant frames where exponential would zero them — better for long-latency reward ([action-composition.md](./action-composition.md) open #8). The cost: more distant, possibly spurious antecedents receive credit, so more thinning load falls on the reward / Death-Ledger filter ([action-composition.md](./action-composition.md) open #3). Accepted trade.
- **Length-bias assumption.** Under per-span global rewards, an action active for more frames takes more reward updates, and with the `α = 1/strength` running mean its estimate consolidates faster. Working assumption: patterns are active for roughly comparable spans, so this averages out and factors out. **To verify** — if span lengths turn out to vary widely, span-normalized application may be needed.
