# Neuron Reuse — Phase B: Reverse Inference Index

**Prerequisite phase for reuse.** Theory in [neuron-reuse.md §2.2, §4](./neuron-reuse.md). This phase
**builds and validates the index but does not consume it** — the lookup that reads it lands in
[Phase D](./neuron-reuse-final.md). Keeping bring-up separate from consumption means an index bug and a
reuse bug can never be confused. This phase stays **bit-exact**: nothing reads the index yet.

---

## Goal

Build a reverse index that answers, efficiently: **"which existing neurons have an outgoing connection to
target T at distance d?"** This is the inverse of a neuron's own connection table ("neuron N infers
{T1, T2, …}"), and it is the candidate-generation step for reuse lookup.

It must span **both** connection stores, because connections are physically split:

- `spatial_connections: FxHashMap<NeuronId, ConnectionData>` — d=0, no distance keying
  ([neuron.rs:219](../brain/brain-core/src/neuron.rs)).
- `temporal_connections: Vec<FxHashMap<NeuronId, ConnectionData>>` — index d holds the d>0 connections;
  index 0 unused ([neuron.rs:240](../brain/brain-core/src/neuron.rs)).

---

## This is a connection index, not a context index

The codebase already has two **context** indexes, which are *not* what we need:

- `spatial_context_index: ctx_neuron → {pattern_ids that reference it}` ([neuron.rs:227](../brain/brain-core/src/neuron.rs)).
- `temporal_context_index: ctx_neuron → distance → {pattern_ids}` ([neuron.rs:246](../brain/brain-core/src/neuron.rs)).

Those map a **context neuron** to the **patterns whose routing context references it** — used by pattern
recognition to narrow candidates. The reuse index is a different inverse: it maps a **connection target**
to the **neurons whose outgoing connection set includes it**. No such index exists today
(confirmed: "cannot answer 'which neurons have a connection TO neuron X' efficiently"). It is genuinely new
— structurally parallel to the context indexes, but over connections.

---

## Design

### Shape

Two indexes mirroring the two stores. Recommend living on the **column** so each column owns the index for
its resident source neurons (shard by the column owning the **source** neuron), with a region-level fan-out
query that merges per-target source sets:

```
spatial_inference_index:  FxHashMap<NeuronId /*target*/, FxHashSet<NeuronId /*source*/>>
temporal_inference_index: FxHashMap<NeuronId /*target*/, FxHashMap<Distance, FxHashSet<NeuronId /*source*/>>>
```

The temporal index keys distance because a source may connect to the same target at multiple distances and
reuse is per-distance. The spatial index needs no distance dimension (d=0 only), matching
`spatial_connections`' own shape.

> Storage micro-decision (flat `(target,distance) → sources` vs nested `target → distance → sources`):
> defer to a benchmark on the temporal side. Start nested to mirror `temporal_context_index`; revisit if
> the merge step in the region query shows up in a profile.

### Membership, maintained on connection create

Connections are created/strengthened here:

- `create_connection()` — spatial insert at [neuron.rs:514](../brain/brain-core/src/neuron.rs),
  temporal insert at [neuron.rs:519](../brain/brain-core/src/neuron.rs).
- `strengthen_connection()` — [neuron.rs:563-576](../brain/brain-core/src/neuron.rs).
- `upsert_connection()` / `strengthen_or_create_connection()` —
  [neuron.rs:538-550](../brain/brain-core/src/neuron.rs).

The reverse index is **membership-only**: it records *that* source→target exists at distance d, not the
strength. Therefore it changes **only on create** (a new edge appears), never on strengthen (membership is
unchanged when an existing edge gets stronger). This is simpler than the earlier
"create/strengthen/decay/delete" framing.

**There is no connection-delete/decay path today** — connections persist; decay was removed in a prior
commit (see comment at [neuron.rs:1581-1587](../brain/brain-core/src/neuron.rs)). So:

- Every edge ever created is live (strength ≥ 1), and the membership index never needs a removal hook now.
- **If decay/delete is reintroduced later**, the index must drop `source` from `target`'s set on delete.
  Leave a clearly-marked hook stub at the (currently nonexistent) delete site so this isn't forgotten.
  Tracked also in [neuron-reuse.md §5.2](./neuron-reuse.md) (dead-edge pollution risk).

### Update batching at orchestration boundaries

Do not mutate the index per-connection-create during a parallel dispatch. Each neuron emits
`IndexUpdate` events as part of its per-neuron result; the thalamus applies them at the **orchestration
boundary** immediately following the dispatch (between the parallel wave and the next orchestration step).
This guarantees the Phase D lookup sees the index reflecting this frame's deltas — no stale-index lag, no
"defer to next frame" workaround. The pattern mirrors the existing temporal path: parallel per-neuron work
inside the dispatch, cross-neuron consolidation (allocate, route, index-update) sequentially between
dispatches.

### Region-level query op

Expose `query_inference_sources(targets: &[NeuronId], distance: Distance) -> FxHashMap<NeuronId, FxHashSet<NeuronId>>`
on the region: fan out across columns, each column answers from its local index for the source neurons it
owns, merge per-target source sets. Distance 0 routes to the spatial index, d>0 to the temporal index.
This is the op Phase D's lookup calls; building it here (unconsumed) lets it be unit-tested in isolation.

### Persistence

Not serialized — rebuilt on load. Add `rebuild_inference_index()` called once at load time, alongside the
existing connection-restore path. (Connections themselves are already serialized; the index is derived.)

---

## Acceptance gates (inline)

- **Unit — spatial**: create neurons N1→T1 and N2→T1 in `spatial_connections`; `query_inference_sources([T1], 0)`
  returns `{T1: {N1, N2}}`.
- **Unit — temporal distance keying**: N1→T1 at d=2 and N2→T1 at d=3; query at d=2 returns `{T1:{N1}}`,
  at d=3 returns `{T1:{N2}}`.
- **Unit — both stores isolated**: a spatial edge to T must not appear in a temporal query for T and vice
  versa.
- **Unit — rebuild**: snapshot a brain with connections, restore, `rebuild_inference_index()`, confirm the
  index matches a freshly-built one edge-for-edge.
- **Size**: index entry count is proportional to total connection count (no bloat); memory roughly doubles
  the connection-graph footprint — acceptable, revisit if it becomes a bottleneck.
- **Bit-exact**: with the index built but unconsumed, stocks regression stays byte-identical. The index is
  write-only this phase.

---

## Notes / gotchas

- **Sharding key is the source neuron's column.** A target T may be inferred by sources living in many
  columns; the region query merges across columns. Don't shard by target — that would put a single target's
  source set on one column and bottleneck writes from every other column.
- **Self-edges**: if a neuron ever connects to itself, it appears as its own source. Harmless for the
  index; Phase D filters self-matches at lookup time, not here.
- **Strength-0 edges**: none exist today (create starts at strength 1, strengthen only increases). If decay
  returns and can drive strength to 0 without deleting, decide then whether the index should track only
  edges above a liveness threshold — a membership index over dead edges would over-broaden Phase D's
  candidate set. Out of scope while there is no decay path.
