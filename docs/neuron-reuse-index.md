# Neuron Reuse — Phase B: Reverse Inference Index

**Prerequisite phase.** Theory in [neuron-reuse.md §3.2](./neuron-reuse.md). This phase **builds and validates
the index but does not consume it** — the lookup that reads it lands in [Phase D](./neuron-reuse-final.md).
Separating bring-up from consumption means an index bug and a reuse bug can never be confused.

---

## Goal

Build reverse indexes that answer: **"which existing neurons have a connection to target T at distance d?"**
— the inverse of a neuron's own connection table, and the candidate-generation step for reuse lookup.

Because reuse applies at every distance, and the connection stores are split (`spatial_connections` flat,
`temporal_connections` distance-keyed), there are **two indexes** with different types, mirroring the existing
`spatial_context_index` / `temporal_context_index` split.

---

## These are connection indexes, not context indexes

The codebase already has two context indexes that map a **context neuron** → **patterns whose routing context
references it**: `spatial_context_index` (no distance) and `temporal_context_index` (distance-keyed)
([neuron.rs](../brain/brain-core/src/neuron.rs)). The reuse indexes are the different inverse —
**connection target → neurons whose outgoing connection set includes it** — and there are **two of them,
mirroring the context indexes' split types**, because the connection stores are themselves split
(`spatial_connections` flat, `temporal_connections` distance-keyed). No such index exists today.

---

## Design

### Shape — two indexes, mirroring the context indexes

```
spatial_connection_index:  FxHashMap<NeuronId /*target*/, FxHashSet<NeuronId /*source*/>>                       // d=0, no distance
temporal_connection_index: FxHashMap<NeuronId /*target*/, FxHashMap<Distance, FxHashSet<NeuronId /*source*/>>>  // d>0, distance-keyed
```

The spatial index has **no distance dimension** (d=0 co-activation is same-frame), exactly like
`spatial_context_index`; the temporal index keys distance, like `temporal_context_index`. Both live on the
**column**, sharded by the column owning the **source** neuron; a region-level fan-out query merges per-target
source sets. The Phase D lookup routes to the spatial index at d=0 and the temporal index at d>0.

### Membership, maintained on connection create

Spatial connections are created in `create_connection`'s spatial insert
([neuron.rs](../brain/brain-core/src/neuron.rs)) → update `spatial_connection_index`; temporal connections
in the temporal insert ([neuron.rs](../brain/brain-core/src/neuron.rs)) → update
`temporal_connection_index`. Both indexes are **membership-only** — they record *that* source→target exists,
not the strength — so they change **only on create**, never on `strengthen_connection`.

There is **no connection-delete/decay path today** (decay was removed —
[neuron.rs](../brain/brain-core/src/neuron.rs)), so every indexed edge is live and no removal hook is
needed now. **If decay/delete returns**, drop `source` from `target`'s set on delete — leave a marked stub.

### DECIDE-THIS — strength-blind candidacy

Membership ignores strength, and connections start at 1 and never decay, so a neuron that connected to T from
a **single incidental co-occurrence** is as much a reuse candidate as one strongly bound to T. Phase D's
default score is pure membership overlap, so many weak incidental edges can clear the merge threshold and
trigger a bad reuse — with **no decay** to clean it up. Decide before Phase D consumes the index:

1. **Membership-only** — cheapest; rely on the merge threshold (+ future decay). Risk: incidental edges drive
   bad reuse.
2. **Strength-gated membership** — an edge enters the index only once strength clears a floor (membership can
   then appear on *strengthen*).
3. **Strength-weighted scoring** — keep the index membership-only, weight matched targets by strength in
   Phase D's score.

Recommend (2) or (3) over (1), given no decay backstop. With two separate indexes (and reuse already using
different merge thresholds at d=0 vs d>0), this choice **can be made per index** if spatial co-activation and
temporal sequence edges turn out to want different policies — but the question is the same for both.

### Update batching at orchestration boundaries

Each neuron emits `IndexUpdate` events in its per-neuron result; the thalamus applies them at the
**orchestration boundary** right after the dispatch, so the Phase D lookup sees this frame's deltas (including
corrections minted earlier this frame). Connection learning already routes through `create_connection`, so new
edges hit the index by construction.

### Region query op

`query_inference_sources(targets, distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>` — fan out across
columns, merge per-target. Routes to `spatial_connection_index` when `distance == 0` and
`temporal_connection_index` otherwise. Built here (unconsumed) so it can be unit-tested in isolation.

### Persistence

Neither index is serialized — both rebuilt on load via `rebuild_connection_indexes()`, alongside the
connection-restore path.

---

## Acceptance gates (inline)

- **Unit — spatial index**: two sources to one target in `spatial_connections` → query at d=0 returns both.
- **Unit — temporal index distance keying**: sources to one target at d=2 vs d=3 resolve to the right sets.
- **Unit — stores isolated**: a spatial edge to T never appears in a temporal query for T, and vice versa.
- **Unit — rebuild**: snapshot → restore → rebuild → both indexes match a fresh build edge-for-edge.
- **Size**: each index's entry count ∝ its connection-store size.
- **No behavior change**: indexes built but unconsumed → no effect on the wave-front's output.

---

## Notes / gotchas

- **Shard by the source neuron's column**, not the target.
- **Self-edges** harmless here; Phase D filters self-matches at lookup.
- **Sequencing**: consumed only by Phase D, so B can sit anywhere before D.
