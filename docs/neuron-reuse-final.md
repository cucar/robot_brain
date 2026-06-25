# Neuron Reuse — Phase D: Reuse Lookup

**The feature — and a comparatively light phase**, because [Phase C](./neuron-reuse-frame.md) already built the
multi-parent machinery and [Phase A](./neuron-reuse-wavefront.md) the coordinate-less wave-front substrate.
Theory in [neuron-reuse.md §3.2, §3.5](./neuron-reuse.md). This phase adds the cross-frame **reuse lookup** on
top of Phase C's batched-mint path, consuming the reverse index from [Phase B](./neuron-reuse-index.md).
Applies at **all distances**. After this phase, reuse is always on (no enable flag). It needs **no new
same-frame tracking set** — reuse installs routing for next frame, so there is no this-frame activation to
inhibit (see below).

What's **already done**: coordinate-less corrections + footprints + settling-wave `process_spatial`/`process_temporal` (Phase A);
multi-parent lifecycle (refcounted reaping, multi-parent serialization, shared-neuron activation), the
`parent_id` audit, batched mint (Phase C). Phase D does not redo them.

---

## Goal

**Lookup precedes grouping.** Each erroring neuron queries the reverse index against its observed reality for
an existing neuron whose inference signature matches ≥ the merge threshold for this distance. The **hits reuse**
that neuron; only the **misses** flow into the Phase-C path — grouped by (distance, observed-set) and minted
one correction per group. Reused targets and fresh mints are then **wired together** in one batched step.

Because the lookup query is purely the observed reality, co-failers (same observed-set) issue the identical
query and resolve identically, so the per-request lookup may be **deduplicated to one query per distinct
observed-set** without changing the outcome — an efficiency optimization, not a different algorithm.

---

## Design

### Lookup first, group the residual, mint, then wire

```
errors  = collect_errors_from_wave_fixpoint()                        // each: (neuron, distance, observed, footprint)

// 1. Reuse lookup — per request. Queries dedup by (distance, observed); a result is observed-set-keyed,
//    so all co-failers with the same observed reality resolve identically.
queries = errors.map(|e| (e.observed, e.distance)).dedup()
results = region.query_inference_sources_batch(queries)              // parallel, from Phase B index
reused, residual = [], []
for e in errors:
    match score_and_pick(results[(e.distance, e.observed)], e.observed, merge_threshold(e.distance)):
        Some(R) => reused.push((e.neuron, R)),                       // hit: reuse existing
        None    => residual.push(e),                                 // miss: falls through to mint

// 2-3. Group only the residual (Phase C) and mint one coordinate-less correction per group.
mints = []
for ((distance, observed), errs) in residual.group_by(|e| (e.distance, e.observed)):
    C = mint_one(observed, distance); C.footprint = (⋃ errs.footprint) ∪ observed
    for e in errs: mints.push((e.neuron, C))

// 4. Wire everything together — reused targets and fresh mints alike. Installs routing for next frame.
for (neuron, target) in reused ++ mints:
    wire_correction(neuron, target)
```

### `find_reusable` / `score_and_pick`

1. For each target T in `observed`, read candidate sources from the Phase-B index at this distance.
2. Score each candidate: `|candidate.connections ∩ observed| / |observed|`, or reuse the existing
   common/missing/novel scoring (`match_observed` [neuron.rs](../brain/brain-core/src/neuron.rs)). Scoring is
   **strength-blind** — the index is membership-only ([Phase B](./neuron-reuse-index.md)).
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

The one remaining activation case — a shared neuron routing-matched from several parents at different depths in
one sweep — is activated **at each depth** (Phase A's multi-depth `neuron_states` holds it across levels), and
needs **no** separate Phase-D inhibition set.

### Cross-frame accrual

The lookup wires a parent to a neuron that existed before this frame. The per-parent-entry / refcount /
serialization machinery from Phase C handles this unchanged; coordinate-less corrections mean no anchor
reconciliation across frames either.

---

## Reuse wires routing for next frame; no same-frame injection

Like a mint, the lookup **installs** the erroring neurons' routing → R for the next time their context recurs
([neuron.rs](../brain/brain-core/src/neuron.rs)); it does **not** activate R this frame. So there is
no "inject R at a deeper level mid-sweep" problem — R is active this frame only if its own routing
independently matched.

The one place a shared neuron lands at **multiple depths in one frame** is multi-parent **routing**: under
reuse, R can be routing-matched from several parents at different levels in one sweep
(`activate_*_pattern(pattern_id, level+1)` from each), activating it at each level. The multi-depth
`neuron_states` ([Phase A](./neuron-reuse-wavefront.md)) **holds** a neuron active at several levels and each
activation is **processed at its depth** (not collapsed); no extra inhibition set is needed.

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
  ([brain.rs](../brain/brain-core/src/brain.rs)), and reuse only adds routing entries (fired next
  frame), so it cannot extend the current frame's sweep.

---

## Notes / gotchas

- **Same-frame, same-group candidate**: a correction minted earlier this frame is a legal reuse target for a
  later group at the same distance (its index update at the orchestration boundary makes it visible). Allow it.
- **One lookup and at most one mint per group per frame** — observed reality singular per group
  ([neuron-reuse.md §3.1](./neuron-reuse.md)).
- **Index freshness**: Phase-B updates at the orchestration boundary, so a lookup always sees this frame's
  deltas.
