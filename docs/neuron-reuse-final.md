# Neuron Reuse — Phase D: Reuse Lookup + Tracking Sets

**The feature.** Theory in [neuron-reuse.md §2.2, §2.4, §3.2](./neuron-reuse.md). This phase adds the
cross-frame **reuse lookup** on top of the batched-mint path from [Phase C](./neuron-reuse-frame.md),
consuming the reverse index from [Phase B](./neuron-reuse-index.md), and introduces the two per-frame
tracking sets that make reuse safe. After this phase, reuse is always on (no enable flag).

---

## Goal

Per (distance, neighborhood) group with errors this frame: query the reverse index **once** against the
group's observed reality for an existing neuron whose inference signature partially matches. If the best
candidate scores ≥ the merge threshold for this distance, wire all co-failers in the group to it; otherwise
fall through to the Phase-C batched mint. Plus: add `fired_this_frame` and `correction_wired_this_frame`,
and route the correction-wired voting/error-check exclusion through the latter.

---

## Design

### The lookup, slotted into Phase C's seam

Phase C's per-group loop body was `mint_one(observed, distance)`. Phase D wraps it:

```
errors = collect_errors_from_phase_stabilization()
by_group = errors.group_by(|e| (e.distance, e.observed_set))
queries  = by_group.keys().map(|(d, observed)| (observed, d))      // one query per group
results  = region.query_inference_sources_batch(queries)          // parallel, from Phase B index
for ((distance, observed), errs) in by_group:
    if let Some(reuse) = score_and_pick(results[(distance, observed)], observed, merge_threshold(distance)):
        for e in errs: wire_correction(e.erroring_neuron, reuse)
        mark_correction_wired(reuse)                                // §3.2: learn, no vote, no error-check
    else:
        n = mint_one(observed, distance)                           // Phase C fallback
        for e in errs: wire_correction(e.erroring_neuron, n)
        mark_correction_wired(n)
```

### `find_reusable` / `score_and_pick`

New method, e.g. `find_reusable(observed_targets, distance) -> Option<NeuronId>`:

1. For each target T in `observed_targets`, read candidate sources from the Phase-B index at this distance.
2. Score each candidate: `|candidate.connections ∩ observed_targets| / |observed_targets|`, or reuse the
   existing common/missing/novel scoring from pattern matching (`match_observed` at
   [neuron.rs:1140](../brain/brain-core/src/neuron.rs)) for exact symmetry with recognition.
3. Filter candidates ≥ `merge_threshold(distance)`:
   - d=0 → `spatial_merge_threshold` ([neuron.rs:203](../brain/brain-core/src/neuron.rs)).
   - d>0 → the temporal merge threshold.
   Setting a phase's threshold to 1.0 disables reuse for that phase (and partial-context recognition),
   intentionally coupled ([neuron-reuse.md §2.5](./neuron-reuse.md)).
4. Return the best-scoring candidate (tie-break to smaller id, matching
   [neuron.rs:1146](../brain/brain-core/src/neuron.rs)), or `None`.

### Tracking sets

Two new per-frame sets on the brain, cleared at frame end:

- **`fired_this_frame: FxHashSet<NeuronId>`** — populated as neurons activate in both phases. Enforces
  refractory: a neuron fires at most once per frame. Load-bearing for termination once an existing neuron
  can be wired from many sources in one frame.
- **`correction_wired_this_frame: FxHashSet<NeuronId>`** — every neuron selected as a correction target
  this frame (minted **or** reused). Members: learn the observed set, **do not vote**, **are not
  error-checked**.

### Voting: layer the exclusion, don't replace suppression

> This corrects the earlier plan, which said "route action voting through `fired_this_frame \
> correction_wired_this_frame` instead of per-level accumulation." That is wrong against the current code.
> Action voting today has **no** per-level accumulation and **no** subsumption filter: it collects every
> active (neuron, age) whose `activated_pattern_id` is `None` ([memory.rs:197-206](../brain/brain-core/src/memory.rs),
> aggregated at [brain.rs:1623-1699](../brain/brain-core/src/brain.rs)). `fired_this_frame` is a **superset**
> of that voter set (it includes pattern-suppressed activations). Routing voting through `fired_this_frame`
> would therefore *change* behavior by re-admitting suppressed voters.

Correct framing: **keep the existing `activated_pattern_id` suppression exactly as is, and subtract
`correction_wired_this_frame` from the voter set on top of it.** When `correction_wired_this_frame` is empty
(no reuse, no fresh correction active this frame), the voter set is bit-identical to today — which is the
property that lets this land without a stocks regression on the no-correction path.

Concretely: in `get_active_voter_ages` / the voter collection at
[memory.rs:197-206](../brain/brain-core/src/memory.rs), exclude any neuron in
`correction_wired_this_frame`. Do not otherwise change which ages vote.

---

## DECIDE-THIS #2 — Refractory vs cross-level injection

A reused neuron R can hit a conflict the mint path never does: R may **already have fired this frame** via
its own routing match (so it is in `fired_this_frame` at some activation level), **and then** be selected as
the correction target for a group whose erroring neuron sits at a higher level.

**The questions:**

1. **Does refractory block the second activation?** R already fired (refractory says once per frame). But
   correction-wiring wants R active *as a correction target* so it can learn the observed set and be the
   thing co-failers route to.
2. **At what level does R land** for the apex handoff / temporal entry — its original routing-match level,
   or `erroring_neuron_level + 1`?
3. **Can R even be injected** at `erroring_neuron_level + 1` if the spatial sweep is already past that level
   (the sweep is monotonically increasing — [brain.rs:933](../brain/brain-core/src/brain.rs))?

**Recommended resolution (decide before coding the wiring):**

- **Refractory governs *activation*, correction-wiring governs *routing*.** R is not re-activated. The
  correction wire is a routing-table edit on the erroring neurons (next time this context occurs, route to
  R) plus adding R to `correction_wired_this_frame` for *this frame's* learning. R's existing activation
  (from its own routing match) stands; it learns the observed set via correction-wiring but does not get a
  second activation.
- **R does not get re-leveled this frame.** Its apex/temporal participation, if any, is at the level it
  already fired at. The "appears at `source_level + 1` this frame" language in an earlier draft is dropped:
  the level relationship is realized **next** frame, when the erroring neuron's now-edited routing fires R
  at `source_activation_level + 1`. This sidesteps the monotonic-sweep injection problem entirely (Q3:
  there is no backward injection because there is no same-frame injection).
- **If R has *not* fired this frame** (the common case — R is a candidate found purely via the index, not
  itself routed-to this frame), then R *is* activated as a correction target this frame, like a fresh mint:
  added to `fired_this_frame` and `correction_wired_this_frame`, learns, no vote, no error-check. This is
  the symmetric case to DECIDE-THIS #1's mint-frame activation.

So the rule is: **reuse wires routing for next frame and tags R for this frame's learning; it never forces a
second activation or a backward level injection.** Confirm this is consistent with how `wire_correction`
ends up touching active memory, and write the resolved rule into [neuron-reuse.md §3.2](./neuron-reuse.md).

---

## Acceptance gates (inline)

- **Unit — within-frame still batches (regression on Phase C)**: two neurons erring against the same
  observed set at the same distance in one frame, with no prior matching neuron, still produce exactly
  **one** minted correction (lookup miss → Phase-C fallback intact).
- **Unit — cross-frame reuse**: frame 1 mints a correction for observed set S; frame 2 errs against an
  observed set overlapping S ≥ merge threshold; frame 2 **reuses** frame 1's neuron via the index (no new
  mint). Drop overlap below threshold → frame 2 mints fresh (boundary check).
- **Unit — self-match filtered**: an erroring neuron whose own connections match the observed set is not
  selected as its own reuse target.
- **Unit — same-frame, same-phase candidate**: a correction minted earlier this frame/phase is a legal
  reuse target for a later group at the same distance (its index update applied at the orchestration
  boundary makes it visible). Confirm this composes; see gotcha below.
- **Unit — voting bit-exact when no corrections active**: with `correction_wired_this_frame` empty, the
  voter set and action output match the pre-Phase-D build exactly.
- **Unit — correction-wired excluded from voting**: a reused/minted correction active this frame casts no
  action vote.
- **MNIST**: neuron count drops **further** vs the Phase-A baseline — target **≥30% reduction** overall —
  with accuracy ≥ the mint-only baseline.
- **Profile**: per-group reuse lookup adds **< 20%** to per-frame runtime.
- **Termination**: a frame with heavy cross-frame reuse terminates — refractory + correction-wired
  inhibition bound the phase regardless of reuse activity.

---

## Notes / gotchas

- **Same-frame same-phase reuse candidate (the merge-redundancy question).** If a correction minted for
  group G1 is reused by a later group G2 at the same distance, then G2's observed set overlapped G1's ≥
  threshold — which means G1 and G2 arguably *should have been one group*. Two readings: (a) the grouping
  was coarser than the reuse threshold (group key = neighborhood, reuse score = set overlap) and same-frame
  reuse legitimately collapses what grouping didn't; (b) grouping should have merged them and same-frame
  reuse is papering over it. Resolve by deciding whether the group key and the reuse-overlap threshold use
  the **same** equivalence. Recommend: **allow** same-frame reuse (reading (a)) — grouping is by exact
  observed-set identity, reuse is by *partial* overlap ≥ threshold, so they are different equivalences by
  design and same-frame reuse is the partial-overlap collapse grouping can't do. Keep the index update at
  the orchestration boundary so the just-minted G1 neuron is visible to G2's query.
- **One lookup *and* at most one mint per group per frame.** The observed reality is singular per group
  ([neuron-reuse.md §2.1](./neuron-reuse.md)); no finer clustering.
- **Reuse spans both phases**: d>0 lookup in `process_temporal`, d=0 in `process_spatial`. Each phase
  batches its own group queries against the shared index, using its own merge threshold.
- **Index freshness**: the Phase-B index updates apply at the orchestration boundary after each dispatch,
  so a lookup always sees this frame's deltas — including corrections minted earlier this frame.
