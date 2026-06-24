# Neuron Reuse — Phase B: Reverse Inference Index

**Prerequisite phase.** Theory in [neuron-reuse.md §3.2](./neuron-reuse.md). This phase **builds and validates
the index but does not consume it** — the lookup that reads it lands in [Phase D](./neuron-reuse-final.md).
Separating bring-up from consumption means an index bug and a reuse bug can never be confused.

---

## Goal

Build a reverse index that answers: **"which existing neurons have a connection to target T at distance d?"**
— the inverse of a neuron's own connection table, and the candidate-generation step for reuse lookup.

It covers **all distances** (d=0 and d>0), because reuse applies at every distance on the wave-front. Whatever
connection storage the [wave-front phase](./neuron-reuse-wavefront.md) leaves in place — a single distance-keyed
store, or separate d=0 / d>0 maps — the index spans all of it.

---

## This is a connection index, not a context index

The codebase already has context indexes (`spatial_context_index`, `temporal_context_index` —
[neuron.rs:227, 246](../brain/brain-core/src/neuron.rs)) that map a **context neuron** → **patterns whose
routing context references it**. The reuse index is a different inverse: **connection target → neurons whose
outgoing connection set includes it**. No such index exists today.

---

## Design

### Shape

```
inference_index: FxHashMap<NeuronId /*target*/, FxHashMap<Distance, FxHashSet<NeuronId /*source*/>>>
```

Distance-keyed (d=0 included). Live on the **column**, sharded by the column owning the **source** neuron; a
region-level fan-out query merges per-target source sets.

### Membership, maintained on connection create

Connections are created/strengthened in `create_connection` / `strengthen_connection` / `upsert_connection`
([neuron.rs:512-576](../brain/brain-core/src/neuron.rs)). The index is **membership-only** — it records *that*
source→target exists at distance d, not the strength — so it changes **only on create**, never on strengthen.

There is **no connection-delete/decay path today** (decay was removed —
[neuron.rs:1581-1587](../brain/brain-core/src/neuron.rs)), so every indexed edge is live and no removal hook is
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

Recommend (2) or (3) over (1), given no decay backstop.

### Update batching at orchestration boundaries

Each neuron emits `IndexUpdate` events in its per-neuron result; the thalamus applies them at the
**orchestration boundary** right after the dispatch, so the Phase D lookup sees this frame's deltas (including
corrections minted earlier this frame). Confirm the correction-wired learning path routes through
`create_connection` so its new edges hit the index.

### Region query op

`query_inference_sources(targets, distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>` — fan out across
columns, merge per-target. Built here (unconsumed) so it can be unit-tested in isolation.

### Persistence

Not serialized — rebuilt on load via `rebuild_inference_index()`, alongside the connection-restore path.

---

## Acceptance gates (inline)

- **Unit — distance keying** (incl. d=0): edges at different distances resolve to the right source sets.
- **Unit — union**: two sources to one target at the same distance → both returned.
- **Unit — rebuild**: snapshot → restore → rebuild → matches a fresh build edge-for-edge.
- **Size**: index entry count ∝ connection count.
- **No behavior change**: index built but unconsumed → no effect on the wave-front's output.

---

## Notes / gotchas

- **Shard by the source neuron's column**, not the target.
- **Self-edges** harmless here; Phase D filters self-matches at lookup.
- **Sequencing**: consumed only by Phase D, so B can sit anywhere before D.
