# Neuron Reuse — Phase D: Reuse Lookup

**The feature — and a comparatively light phase**, because [Phase C](./neuron-reuse-frame.md) already built the
multi-parent machinery and [Phase A](./neuron-reuse-wavefront.md) the coordinate-less wave-front substrate.
Theory in [neuron-reuse.md §3.2, §3.4](./neuron-reuse.md). This phase adds the cross-frame **reuse lookup** on
top of Phase C's batched-mint path, consuming the reverse index from [Phase B](./neuron-reuse-index.md).
Applies at **all distances**. After this phase, reuse is always on (no enable flag). It needs **no new
same-frame tracking set** — reuse installs routing for next frame, so there is no this-frame activation to
inhibit (see below).

What's **already done**: coordinate-less corrections + footprints + settling-wave `process_spatial`/`process_temporal` (Phase A);
multi-parent lifecycle (refcounted reaping, multi-parent serialization, shared-neuron activation), the
`parent_id` audit, batched mint (Phase C). Phase D does not redo them.

---

## Goal

Per (distance, observed-set) group with errors this frame: query the reverse index **once** against the
group's observed reality for an existing neuron whose inference signature partially matches. If the best
candidate scores ≥ the merge threshold for this distance, wire all co-failers to it; otherwise fall through to
the Phase-C batched mint.

---

## Design

### The lookup, slotted into Phase C's seam

```
by_group = errors.group_by(|e| (e.distance, e.observed))
queries  = by_group.keys().map(|(d, observed)| (observed, d))        // one query per group
results  = region.query_inference_sources_batch(queries)            // parallel, from Phase B index
for ((distance, observed), errs) in by_group:
    if let Some(reuse) = score_and_pick(results[(distance, observed)], observed, merge_threshold(distance)):
        for e in errs: wire_correction(e.erroring_neuron, reuse)   // installs routing for next frame
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

### No new same-frame tracking set is needed

Reuse **installs routing for next frame** (above) — it does **not** activate the reused neuron this frame. An
earlier plan added a `correction_wired_this_frame` set to suppress a reused neuron's "wiring-side-effect
activation," but there is no such activation, so the set has nothing to act on. A reused neuron is active this
frame only via its **own** routing match, in which case it votes and error-checks as a normal recognition
(existing behavior — nothing to suppress). So Phase D is essentially **just the lookup** plus the multi-parent
accrual that Phase C's machinery already handles.

The one genuinely-new activation question — a shared neuron routing-matched from several parents at different
depths in one sweep — is the **multi-parent activation model** ([neuron-reuse.md §5.3](./neuron-reuse.md));
whether it needs any inhibition falls out of that decision, not a separate Phase-D set.

### Cross-frame accrual

The lookup wires a parent to a neuron that existed before this frame. The per-parent-entry / refcount /
serialization machinery from Phase C handles this unchanged; coordinate-less corrections mean no anchor
reconciliation across frames either.

---

## Reuse wires routing for next frame; no same-frame injection

Like a mint, the lookup **installs** the erroring neurons' routing → R for the next time their context recurs
([neuron.rs:1455-1471](../brain/brain-core/src/neuron.rs)); it does **not** activate R this frame. So there is
no "inject R at a deeper level mid-sweep" problem — R is active this frame only if its own routing
independently matched.

The one place a shared neuron lands at **multiple depths in one frame** is multi-parent **routing**: under
reuse, R can be routing-matched from several parents at different levels in one sweep
(`activate_*_pattern(pattern_id, level+1)` from each), activating it at each level. Whether the temporal
`neuron_states` can hold that — and whether such activations are processed at each depth or collapsed — is the
multi-depth memory question in [Phase A](./neuron-reuse-wavefront.md), and the open **multi-parent activation
model** ([neuron-reuse.md §5.3](./neuron-reuse.md)). Any inhibition that's needed falls out of that decision.

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
- **MNIST + stocks**: neuron count drops further vs the Phase-A baseline; accuracy ≥ baseline.
- **Profile**: per-group reuse lookup adds **< 20%** per-frame.
- **Termination**: heavy cross-frame reuse still terminates — the level-sweep terminates as today
  ([brain.rs:1174, 1278](../brain/brain-core/src/brain.rs)), and reuse only adds routing entries (fired next
  frame), so it cannot extend the current frame's sweep.

---

## Notes / gotchas

- **Same-frame, same-group candidate**: a correction minted earlier this frame is a legal reuse target for a
  later group at the same distance (its index update at the orchestration boundary makes it visible). Allow it.
- **One lookup and at most one mint per group per frame** — observed reality singular per group
  ([neuron-reuse.md §3.1](./neuron-reuse.md)).
- **Index freshness**: Phase-B updates at the orchestration boundary, so a lookup always sees this frame's
  deltas.
