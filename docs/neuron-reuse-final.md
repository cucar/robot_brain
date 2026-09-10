# Neuron Reuse — Phase C: Reuse Lookup + Expansion

> **⚠ Model corrected.** The reuse mechanism is now **recognize → predict L0 → on misprediction,
> transitively-merge-cluster the correction requests by neighborhood → reuse/expand a matched pattern or mint
> one**, balanced by refinement (the split force: specialize + reference-drain reaping). See [neuron-reuse.md §3](./neuron-reuse.md) and the corrected
> simulation spec [neuron-reuse-simulation.md](./neuron-reuse-simulation.md), now built and validated. This
> phase is **lookup + expansion**; the detailed mechanics below predate the correction and will firm up
> against the validated simulation. Note: the sim confirms reuse correctly matches by L0 output **across
> levels** (a higher request reusing a lower pattern is content-addressable reuse, not a bug).

**The feature — a comparatively light phase**, because [Phase B](./neuron-reuse-frame.md) built the
cluster+mint and multi-parent machinery (the coordinate-less substrate is a prerequisite). Theory in
[neuron-reuse.md §3.5, §3.5.1](./neuron-reuse.md). This phase adds the cross-frame **reuse
lookup AND pattern expansion** on top of Phase B, consuming the reverse index from
[Phase A](./neuron-reuse-index.md). Applies at all distances; always on (no enable flag). Reuse installs
routing for next frame, so there is no this-frame activation to inhibit.

---

## Goal

**Lookup precedes clustering, then the pool is unified.** Each correction request queries the reverse index for
an existing pattern whose **L0 prediction** covers its region ≥ the merge threshold. **Hits reuse** that
pattern. Then matched patterns and the **misses** are clustered in **one pool** (transitive merge, §3.5.1): a
matched pattern adjacent to a cluster of new requests **expands** to absorb them (its footprint/context grow,
the requests wire to it); a cluster of only-new requests mints. Reused, expanded, and freshly-minted targets
all install together.

---

## Design

### Lookup, unify the pool, expand or mint, then wire

```
requests = collect_correction_requests()                 // units whose L0 prediction missed (≥ error threshold)

// 1. Reuse lookup — per request, against the reverse index (Phase A): existing patterns predicting this L0 region.
matched, fresh = [], []
for r in requests:
    match score_and_pick(index.candidates(r.targets, r.distance), r, merge_threshold(r.distance)):
        Some(P) => matched.push((r, P)),                 // hit: reuse pattern P
        None    => fresh.push(r),                        // miss: a new request

// 2. Cluster the UNIFIED pool (matched patterns ∪ fresh requests) by transitive merge over neighborhoods.
for cluster in transitive_merge(matched.patterns ∪ fresh, neighbor):  // base = coord nbhd; higher = footprints touch
    if cluster.neighbors().is_empty(): continue          // need ≥1 neighbor for context
    if let Some(P) = cluster.matched_pattern():
        wire cluster.requests to P as parents
        expand(P, cluster)                               // P.footprint ∪= cluster.footprint; P.context ∪= cluster.context
    else:
        mint_one(cluster)                                // Phase-C fresh mint, predicting the correct L0

// installs fire NEXT frame, unchanged
```

`expand`'s growth cap and overlap gate are open knobs ([neuron-reuse-simulation.md §5.2](./neuron-reuse-simulation.md)).
### `find_reusable` / `score_and_pick`

A request carries the **L0 targets** it predicts (the correct base reality for its region). Lookup:

1. For each L0 target T the request predicts, read candidate sources from the Phase-B index at this distance.
2. Score each candidate against the request's L0 targets (`|candidate.L0_connections ∩ targets| / |targets|`,
   or the existing common/missing/novel scoring `match_observed` [neuron.rs](../brain/brain-core/src/neuron.rs)).
   Scoring is **strength-blind** — the index is membership-only ([Phase A](./neuron-reuse-index.md)).
3. Filter ≥ the **merge threshold for this distance** (spatial at d=0, temporal at d>0; 1.0 disables reuse +
   partial recognition together).
4. Return the best (tie-break smaller id), or `None`. **Filter self-matches.**

The candidate's footprint need not match the request's — reuse is by **L0 output**, not locality. A reused
correction gains another parent and may **expand** to absorb the request's region
([neuron-reuse.md §3.5.1](./neuron-reuse.md)); on activation its footprint is whatever it already covers.

### No new same-frame tracking set is needed

Reuse **installs routing for next frame** (above) — it does **not** activate the reused neuron this frame. An
earlier plan added a `correction_wired_this_frame` set to suppress a reused neuron's "wiring-side-effect
activation," but there is no such activation, so the set has nothing to act on. A reused neuron is active this
frame only via its **own** routing match, in which case it votes and error-checks as a normal recognition
(existing behavior — nothing to suppress). So Phase C is essentially **just the lookup** plus the multi-parent
accrual that Phase B's machinery already handles.

The one remaining activation case — a shared neuron routing-matched from several parents at different depths in
one frame — is activated **at each depth** (the multi-depth `neuron_states` holds it across levels), and
needs **no** separate Phase-D inhibition set.

### Cross-frame accrual

The lookup wires a parent to a neuron that existed before this frame. The per-parent-entry / refcount /
serialization machinery from Phase B handles this unchanged; coordinate-less corrections mean no anchor
reconciliation across frames either.

---

## Reuse wires routing for next frame; no same-frame injection

Like a mint, the lookup **installs** the erroring neurons' routing → R for the next time their context recurs
([neuron.rs](../brain/brain-core/src/neuron.rs)); it does **not** activate R this frame. So there is
no "inject R at a deeper level partway up" problem — R is active this frame only if its own routing
independently matched.

The one place a shared neuron lands at **multiple depths in one frame** is multi-parent **routing**: under
reuse, R can be routing-matched from several parents at different levels in one frame
(`activate_*_pattern(pattern_id, level+1)` from each), activating it at each level. The multi-depth
the multi-depth `neuron_states` **holds** a neuron active at several levels and each
activation is **processed at its depth** (not collapsed); no extra inhibition set is needed.

---

## Acceptance gates (inline)

- **Unit — cluster mint on miss**: a connected cluster of requests with no index match still mints exactly one
  correction (lookup miss → Phase-C cluster + mint).
- **Unit — cross-frame reuse (d=0 and d>0)**: frame 1 mints a correction predicting L0 region S at distance d;
  frame 2 errs at d against a region overlapping S ≥ merge threshold; frame 2 **reuses** via the index (no new
  mint). Drop overlap below threshold → frame 2 mints fresh.
- **Unit — expansion**: a matched pattern adjacent to a cluster of new requests absorbs them — its
  footprint/context grow and the requests wire to it as parents, with no new neuron minted.
- **Unit — self-match filtered**.
- **Unit — cross-frame multi-parent accrual**: reuse R (minted earlier) from a different parent this frame; R
  gains the new routing entry with independent strength; its connection set accumulates.
- **MNIST + stocks**: neuron count drops further vs the no-reuse baseline; accuracy ≥ baseline.
- **Profile**: per-request reuse lookup adds **< 20%** per-frame.
- **Termination**: heavy cross-frame reuse still terminates — the level loop terminates as today
  ([brain.rs](../brain/brain-core/src/brain.rs)), and reuse only adds routing entries (fired next
  frame), so it cannot extend the current frame's climb.

---

## Notes / gotchas

- **Same-frame candidate**: a correction minted earlier this frame is a legal reuse/expansion target for a
  later request at the same distance (its index update at the orchestration boundary makes it visible). Allow it.
- **One correction per connected cluster** — the cluster, not a per-neuron mint, is the unit of correction
  ([neuron-reuse.md §3.4](./neuron-reuse.md)).
- **Index freshness**: Phase-B updates at the orchestration boundary, so a lookup always sees this frame's
  deltas.
