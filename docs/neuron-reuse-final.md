# Neuron Reuse — Phase D: Reuse Lookup

**The feature — and a comparatively light phase**, because [Phase C](./neuron-reuse-frame.md) already built the
multi-parent machinery and [Phase A](./neuron-reuse-wavefront.md) the coordinate-less wave-front substrate.
Theory in [neuron-reuse.md §3.2, §3.4, §4.2](./neuron-reuse.md). This phase adds the cross-frame **reuse
lookup** on top of Phase C's batched-mint path, consuming the reverse index from
[Phase B](./neuron-reuse-index.md), plus the one new tracking set reuse-of-existing-neurons needs. Applies at
**all distances**. After this phase, reuse is always on (no enable flag).

What's **already done**: coordinate-less corrections + footprints + settling-wave `process_spatial`/`process_temporal` (Phase A);
`fired_this_frame`/refractory, shared activation, refcounted reaping, multi-parent serialization, the
`parent_id` audit, batched mint (Phase C). Phase D does not redo them.

---

## Goal

Per (distance, observed-set) group with errors this frame: query the reverse index **once** against the
group's observed reality for an existing neuron whose inference signature partially matches. If the best
candidate scores ≥ the merge threshold for this distance, wire all co-failers to it; otherwise fall through to
the Phase-C batched mint. Plus: add `correction_wired_this_frame` and subtract it from the voter set.

---

## Design

### The lookup, slotted into Phase C's seam

```
by_group = errors.group_by(|e| (e.distance, e.observed))
queries  = by_group.keys().map(|(d, observed)| (observed, d))        // one query per group
results  = region.query_inference_sources_batch(queries)            // parallel, from Phase B index
for ((distance, observed), errs) in by_group:
    if let Some(reuse) = score_and_pick(results[(distance, observed)], observed, merge_threshold(distance)):
        for e in errs: wire_correction(e.erroring_neuron, reuse)
        mark_correction_wired(reuse)        // §4.2: learn, no vote, no error-check
    else:
        C = mint_one(observed, distance); C.footprint = ⋃ errs.footprint   // Phase C fallback
        for e in errs: wire_correction(e.erroring_neuron, C)
```

### `find_reusable` / `score_and_pick`

1. For each target T in `observed`, read candidate sources from the Phase-B index at this distance.
2. Score each candidate: `|candidate.connections ∩ observed| / |observed|`, or reuse the existing
   common/missing/novel scoring (`match_observed` [neuron.rs:1140](../brain/brain-core/src/neuron.rs)). (Strength
   weighting is the Phase-B strength-candidacy decision.)
3. Filter ≥ the **merge threshold for this distance** (spatial threshold at d=0, temporal at d>0; 1.0 disables
   reuse + partial recognition together).
4. Return the best (tie-break smaller id), or `None`. **Filter self-matches.**

The candidate's footprint need not match the group's — reuse is by inference output, not locality. A reused
correction simply gains another parent; on activation its footprint is whatever it already covers.

### Tracking set (the one D-only addition)

`fired_this_frame` and the multi-parent machinery landed in [Phase C](./neuron-reuse-frame.md). Phase D adds
one per-frame set, cleared at frame end:

- **`correction_wired_this_frame`** — every correction target this frame that is a **reused pre-existing**
  neuron. Members: learn the observed set, **do not vote**, **are not error-checked**.

Why D-only: a freshly minted correction already carries the fresh-mint exemption (Phase C). A reused neuron is
not fresh — it has a full prior connection set — so without an explicit tag it would vote (its activation is a
wiring side-effect) and be error-checked against the current observed set (which its old connections may not
match → spurious cascade). So this set is **the fresh-mint exemption extended to reused neurons**.

### Voting: layer the exclusion, don't replace suppression

Action voting collects every active (neuron, age) whose `activated_pattern_id` is `None`
([memory.rs:197-206](../brain/brain-core/src/memory.rs), aggregated at
[brain.rs:1623-1699](../brain/brain-core/src/brain.rs)). **Keep that suppression as is, and subtract
`correction_wired_this_frame` on top.** Empty set ⇒ voter set bit-identical to the pre-D build.

### Cross-frame accrual

The lookup wires a parent to a neuron that existed before this frame. The per-parent-entry / refcount /
serialization machinery from Phase C handles this unchanged; coordinate-less corrections mean no anchor
reconciliation across frames either.

---

## DECIDE-THIS #2 — Refractory vs cross-depth injection

A reused neuron R may **already have fired this frame** via its own routing match (in `fired_this_frame` at
some wave depth), **and then** be selected as the correction target for an error whose source sits deeper.

**What Phase A settles:** memory can hold R at multiple depths in one frame
([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)), so the representation is expressible. The remaining
question is **wave sequencing**: if R is picked for a depth the settling wave has already passed, R's work at
that depth can't run *this* frame even though memory can record the membership.

**Recommended resolution:** refractory governs *fresh activation*; correction-wiring is a separate role. R's
existing activation stands; record its correction-target depth in memory but **do not retro-run** a passed
wave step — full participation realized **next** frame when the edited routing fires R from the start of the
wave. If R has **not** fired this frame (the common case — found purely via the index), R *is* activated as a
correction target like a fresh mint (added to `fired_this_frame` + `correction_wired_this_frame`, learns, no
vote, no error-check). Confirm against `wire_correction`; write the resolved rule into
[neuron-reuse.md §4.2](./neuron-reuse.md).

---

## Acceptance gates (inline)

- **Unit — within-frame still batches**: same-observed-set co-failers with no prior match still produce
  exactly one minted correction (lookup miss → Phase-C fallback).
- **Unit — cross-frame reuse (d=0 and d>0)**: frame 1 mints a correction for observed set S at distance d;
  frame 2 errs at d against a set overlapping S ≥ merge threshold; frame 2 **reuses** via the index (no new
  mint). Drop overlap below threshold → frame 2 mints fresh.
- **Unit — self-match filtered**.
- **Unit — cross-frame multi-parent accrual**: reuse R (minted earlier) from a different parent this frame; R
  gains the new routing entry with independent strength; its connection set accumulates.
- **Unit — voting bit-exact when no corrections active**.
- **Unit — correction-wired excluded from voting**.
- **MNIST + stocks**: neuron count drops further vs the Phase-A baseline; accuracy ≥ baseline.
- **Profile**: per-group reuse lookup adds **< 20%** per-frame.
- **Termination**: heavy cross-frame reuse still terminates — refractory + correction-wired inhibition bound
  the frame.

---

## Notes / gotchas

- **Same-frame, same-group candidate**: a correction minted earlier this frame is a legal reuse target for a
  later group at the same distance (its index update at the orchestration boundary makes it visible). Allow it.
- **One lookup and at most one mint per group per frame** — observed reality singular per group
  ([neuron-reuse.md §3.1](./neuron-reuse.md)).
- **Index freshness**: Phase-B updates at the orchestration boundary, so a lookup always sees this frame's
  deltas.
