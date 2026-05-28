# Hippocampus Phase 1 — Moments as Cortical Substrate

This is the **first** implementation phase toward the architecture in
[hippocampus.md](hippocampus.md). Scope is deliberately narrow: introduce
**moment neurons** into the cortical substrate as union-created nodes, and
get them participating in normal cortical activity.

**Out of scope for Phase 1:** replay, experiments, counterfactuals,
salience module, action temporal bins, sleep consolidation, multi-level
moment hierarchy as a designed feature, parallel hippocampal clock.
Anything from [hippocampus.md](hippocampus.md) not explicitly listed
below is deferred.

---

## Why now

The 60k/10k 7×7 MNIST run revealed the failure mode that motivates this:

- Post-learn (during training): 86.66%
- Cold-replay on training images: 57.00%
- Cold-replay on held-out: 13.49% (collapsed to predicting "5")

Sequence patterns drift as training proceeds — by the time we replay an
old image, the patterns that fired at training-time `learn()` no longer
all fire, so the action wires we wrote then don't sum to the right
prediction. The brain is memorizing sequences, not encoding moments.

Moments fix this by being **multi-parent conjunction nodes** wired to
the action at the same time `learn()` writes the action wires. Even when
some parent patterns drift, the moment still fires by partial-conjunction
match, preserving the index → action pointer.

---

## What a moment is

A moment is a cortical neuron, structurally indistinguishable from a
pattern at activation/voting time. It differs only in **how it was
created**:

- **Pattern creation rule (existing):** intersection. Statistical
  co-occurrence over many frames carves out a narrow context fingerprint.
- **Moment creation rule (new):** union. One-shot binding of every
  high-level pattern currently firing at mint time into a broad context
  fingerprint, all at distance 0.

Both kinds live in the existing `Neuron` struct, in cortical columns,
participate in `recognize_patterns`, vote into action selection, and decay
by the same death-ledger mechanism.

A `kind: NeuronKind { Pattern, Moment }` tag is added for observability
and debugging only. The activation/recognition code paths do **not**
branch on kind.

---

## Routing model (the d=0 question, resolved)

The existing code stores context entries with distance ≥ 1: "this
sub-neuron fired d frames before me." Moments need a way to encode "these
patterns are co-active with me, right now."

**Decision: use distance 0 as a valid context distance, in both directions.**

### Pattern → Moment

When pattern P fires at frame F and moment M includes P as a d=0 parent:

- M's context fingerprint contains `(P, distance: 0)`.
- P's `routing_table` contains an entry for M (P is context-for M).
- P's `context_index` (the inverted index queried by
  `get_pattern_candidates_at_age`) contains `M` under `(P, distance: 0)`.

When P fires, M becomes a recognition candidate this frame (same logic as
existing pattern activation, just with d=0 allowed).

### Moment → Moment

Moments can have other moments as d=0 parents too — this is what gives
the "intersections of unions" hierarchy. A moment M2 minted after moment
M1 (in a later image) can include M1 in its parent set if M1 is firing
at M2's mint frame. The same d=0 routing applies: M1's `routing_table`
gains an entry for M2, indexed under `(M1, distance: 0)`.

This is the lateral moment graph from [hippocampus.md](hippocampus.md),
but **emergent from the mint rule**, not built by an explicit graph
constructor. No `1/Δt` edge weighting in Phase 1 — just standard context
matching.

### Pattern → Pattern (unchanged)

Sequence patterns continue to use distance ≥ 1 only. The d=0 relaxation
is **opt-in** at neuron creation time: a pattern's context fingerprint
never gets a d=0 entry through normal cortex learning, only a moment's
does at mint.

### Cycle prevention

A moment's parents must already exist (and be firing) at mint time, so
parent IDs are always smaller than the new moment's ID. Combined with
processing moments after their parent level (see "Where moments live"),
this prevents same-frame cycles.

---

## Where moments live

Each moment is assigned to **level = max(parent.level) + 1** at mint
time. This guarantees the moment is processed after all its parents
within a frame, which is required for d=0 activation to be deterministic.

In MNIST the brain reaches L4 today. With moments, the first image's
mint creates an L5 moment. Subsequent images create L5 moments. If a
later moment binds an existing L5 moment as a parent, that new moment
lives at L6. This is how moments-of-moments would emerge naturally
without a separate Phase-10-style designed hierarchy.

---

## The four operations

### 1. Mint

Trigger (Phase 1): **end of every training image's frame loop**, called
explicitly from `apps/mnist/jobs/test.js` right before `learn()`.

For each mint:

1. Collect currently-active high-level patterns:
   - Definition: every neuron at the **highest level with any firing
     neuron this frame** that is currently firing above the existing
     activation threshold.
   - Open question: should this be top-K by activation strength rather
     than threshold-based? Defer until we observe parent counts.
2. Allocate a new neuron M with `kind = Moment`,
   `level = max(parent.level) + 1`.
3. For each parent pattern P:
   - Add `(P, distance: 0)` to M's context fingerprint.
   - Add the reverse routing entry on P (M as a child) with
     `(M, distance: 0)`.
   - Update P's `context_index` under `(P, distance: 0)` to include M.
4. Initialize M's activation strength from the mean (or weighted sum) of
   its parents' current activation strengths.
5. Mark M as firing this frame so it can be wired by `learn()` and so it
   can be a parent for any subsequent mint within the frame (relevant for
   multi-image batches; not used in the MNIST one-image-per-mint case).

Return M's neuron id so the caller can hand it to `learn()`.

### 2. Reactivate

Reactivation requires **no new code path**. A moment is a regular cortical
neuron with a context fingerprint of d=0 entries. When its parent patterns
fire on a later frame:

- The inverted index (after the d=0 audit, see implementation steps) lists
  M as a candidate via each firing parent.
- `find_best_pattern_match_at_age` (or equivalent — d=0 needs its own age
  handling, see implementation steps) scores M against the observed
  context.
- If enough parents are co-firing this frame, M's score crosses
  `merge_threshold` and M fires.

This is **pattern completion via partial-conjunction match**. Moments
survive sequence drift because a moment only needs *enough* of its
parents to still fire, not all of them.

### 3. Recursive processing (moments as substrate)

Moments are cortical neurons. Therefore:

- Higher-level pattern formation (the existing `learn_connections` /
  pattern-creation path) can include a firing moment in a new pattern's
  context fingerprint. The pattern just sees an active neuron at some
  distance — it doesn't care that the neuron is a moment.
- A later mint can include an existing firing moment as a parent (the
  Moment → Moment routing above).
- This is the "moments enter the cortical substrate and become raw
  material for further abstraction" property from [hippocampus.md](hippocampus.md),
  achieved at zero extra cost: it just falls out of treating moments as
  regular neurons.

### 4. Reintegration to active memory

When a moment fires in a frame, it must appear in the **active set** that
the rest of the brain machinery reads from:

- It contributes votes to action selection (via its action connections,
  which are written by `learn()` — see "Action wiring" below).
- It is available as context for subsequent pattern formation in the same
  frame and later frames.
- It is available as a parent for subsequent moment mints.
- It participates in the existing context-aging, forgetting, and
  death-ledger mechanisms.

No new "active memory" data structure is needed. The existing per-frame
active-neuron tracking already handles this once moments fire through the
normal pathway.

---

## Action wiring

When `learn(actions, rewards)` is called during MNIST training:

- **Existing path (kept):** wire the action to all currently-active
  high-level pattern neurons. Backward-compatible.
- **New path (added):** also wire the action to the moment minted moments
  ago in the same image. This is the moment-as-index pointer.

Both paths use the existing action-connection machinery. No temporal bins,
no exponential horizons — Phase 3 territory.

At inference time, both pattern action-wires and moment action-wires
contribute to the vote. We'll be able to A/B by toggling either path off
to measure each one's contribution.

---

## Implementation steps

Dependency order. Each step should end in a state where existing MNIST
runs still work (`--image-size 7` etc.) before moving to the next.

### S1. Allow distance 0 in the routing/indexing machinery

The audit pass for d=0. Every place that assumes `distance ≥ 1`:

- [brain/brain-core/src/neuron.rs](../brain/brain-core/src/neuron.rs):
  `get_pattern_candidates_at_age` line ~1001:
  `if pattern_distance < 1 { continue; }` — relax to `< 0`.
- Grep for `< 1`, `>= 1`, `distance == 0`, `distance > 0` across the
  brain crate and decide per occurrence whether the d=0 case is correct.
- `match_observed` — verify that scoring against a d=0 entry works
  (current observed-context entries are keyed by absolute age; an age=0
  entry maps to pattern_distance=0 when the parent is being matched at
  the current frame).
- Warmup gate at line ~860 (`if (current_frame as u32) < self.context_length`)
  — d=0 entries are always reachable, so a moment with only d=0 parents
  should be allowed to fire from frame 0 onward. Decide whether the gate
  applies per-pattern or per-entry.
- Context aging code — ensure d=0 entries don't get aged out incorrectly.
- Serialization — ensure d=0 round-trips cleanly.

Deliverable: a unit test that creates a pattern with one d=0 context
entry, fires the parent, and verifies the child becomes a candidate and
matches.

### S2. Introduce `NeuronKind`

Add `pub enum NeuronKind { Pattern, Moment }` to neuron.rs. Add a `kind`
field to `Neuron`, default `Pattern`. No behavior change yet — just the
tag for observability.

### S3. Implement `mint_moment` at the brain-core API level

New method on the brain (or thalamus, whichever owns neuron creation):

```rust
pub fn mint_moment(&mut self, parents: &[NeuronId], current_frame: FrameNumber) -> NeuronId
```

Implementation:
- Compute target level = max(parent.level) + 1.
- Create a new `Neuron { kind: Moment, level, ... }` in the appropriate
  column.
- For each parent, add the routing-table reverse entry and update the
  context index, all at distance 0.
- Mark the new moment as firing this frame (so it participates in
  downstream operations within this frame).
- Return the new id.

### S4. Expose a JS-callable hook on `brain-napi`

`brain.mintMoment(parentIds): neuronId` — thin wrapper around the
brain-core method. Plus a discovery helper:
`brain.getActiveHighLevelPatterns(): neuronId[]` so the JS caller can
collect parents.

### S5. Wire mint into the MNIST training loop

In `apps/mnist/jobs/test.js`, after the frame loop and before `learn()`:

```js
const activePatterns = this.brain.getActiveHighLevelPatterns();
const momentId = activePatterns.length > 0
    ? this.brain.mintMoment(activePatterns)
    : null;
```

Pass `momentId` into `learn()` so the action wire reaches it. (Either
extend `learn()` to take an optional extra-target list, or have
`mintMoment` mark the moment "active this frame" so `learn()` picks it
up naturally via the existing active-set path. The latter is cleaner.)

### S6. Verify reactivation on cold replay

No code change — just observation. Re-run the 60k/10k experiment with
moments enabled. Expectation:

- Phase-1 post-learn accuracy: unchanged (moments don't change training).
- Phase-2a cold-replay-train: rises substantially (moments fire via
  partial-conjunction match where sequence patterns drifted).
- Phase-2b held-out: rises (less collapse to "5", because moments wired
  per-image preserve the index even as actions accumulate).

If 2a doesn't rise, something is wrong with the routing/activation —
moments aren't firing on their own training images. That's the first
diagnostic.

---

## Open questions for Phase 1 (decide empirically)

1. **Parent-set definition at mint.** Top-K by activation strength,
   threshold-based, or all firing at max-level? Start with "all firing at
   max-level" because it matches the union principle; revisit if
   parent-counts explode.
2. **Moment activation threshold.** Does a moment fire only when ≥K of N
   parents fire? K=N (strict) is conservative but loses completion; K=1
   collapses to "any parent fires" which is too loose. Start with the
   existing `merge_threshold` (currently 0.5) applied to fraction of
   parents firing.
3. **Moment merge.** Two moments with overlapping parent sets — should
   they merge? Probably yes, by the same `merge_threshold`. Phase 1
   leaves merge behavior unchanged and observes whether duplicate moments
   pile up.
4. **Moment forget rate.** Per [hippocampus.md](hippocampus.md) moments
   age slower than patterns. Phase 1: reuse the same forget rate as
   patterns. Add a moment-specific rate later if observation warrants.
5. **Cold-start gating.** [hippocampus.md](hippocampus.md) requires K=3
   activations before a moment contributes to votes. Phase 1: **skip**
   this gate so we can measure the immediate effect on cold-replay. Add
   gating after Phase 1 if we see noise from one-shot moments.
6. **What level do moments live at?** `max(parent.level) + 1` per above.
   Open question: should moments share a level with other moments minted
   at the same parent-level, or each get a fresh level? Sharing seems
   right — they're peers.
7. **Reset behavior.** Does `resetContext()` clear moment activations?
   Yes — moments live in cortical columns and the existing reset path
   should handle them. Verify in S1's audit.

---

## Test plan

The MNIST cold-replay gap is the diagnostic experiment. With moments on
vs off, measure:

| Metric | Baseline (no moments) | With moments (target) |
|---|---|---|
| Phase-1 post-learn | 86.66% | ~same |
| Phase-2a cold-replay train | 57.00% | substantial rise (>75%) |
| Phase-2b held-out | 13.49% | substantial rise (>30%) |
| Predict-5 dominance | 89% of all predictions | drops materially |
| Final neuron count | 210k | 210k + #moments (one per image, so +60k) |

A null result (2a doesn't rise) means moments aren't firing on cold
replay — debug the d=0 routing first.

Add unit tests as we go:

- d=0 context entry creates an index hit and a successful match.
- `mint_moment` creates a neuron at level max+1 with N parents at d=0.
- A minted moment fires in a subsequent frame when its parents fire.
- A moment whose parents partially fire (3 of 5) still fires if the
  fraction exceeds `merge_threshold`.
- `learn()` wires the action to a freshly-minted moment.
- Moments survive `resetContext()` (they're durable cortical neurons).
- Moments appear in the active set for the frame in which they were
  minted (so context/voting picks them up immediately).

---

## What this phase does NOT do

Explicit reminder of deferrals so we don't scope-creep:

- **No replay.** No hippocampal executor, no experiment loop, no
  wavefront walk. Moments only fire through normal cortical activation.
- **No salience module.** Mint trigger is `end-of-image` for MNIST;
  later phases add z-scored reward/PE.
- **No action temporal bins.** Moments use the existing action-wire
  machinery, same as patterns.
- **No sleep / idle consolidation.** No high-temperature integrative
  replay.
- **No counterfactuals.** No parent-pattern-sibling traversal.
- **No designed multi-level hierarchy.** Moments-of-moments may emerge
  from the recursive mint rule, but no special Phase-10-style salience
  triggers per level.
- **No parallel hippocampal clock.** All operations run synchronously in
  the cortex frame loop.

Each of those is a separate phase. Phase 1 succeeds if moments mint,
reactivate, recursively participate, and reintegrate — and if the
MNIST cold-replay gap closes meaningfully as a result.
