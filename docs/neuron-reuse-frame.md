# Neuron Reuse — Phase B: Cluster + Mint + Multi-Parent

> **⚠ Model corrected.** The reuse mechanism is now **recognize → predict L0 → on misprediction,
> transitively-merge-cluster the correction requests by neighborhood → reuse/expand a matched pattern or mint
> one**, balanced by refinement (the split force: specialize + reference-drain reaping). See [neuron-reuse.md §3](./neuron-reuse.md) and the corrected
> simulation spec [neuron-reuse-simulation.md](./neuron-reuse-simulation.md), now built and validated. This
> phase is the **cluster + mint** step; the detailed acceptance gates below predate the correction and will
> firm up against the validated simulation. The multi-parent machinery
> (ownership, refcount reaping, serialization, shared activation) is unchanged.

**The heaviest reuse phase.** Theory in
[neuron-reuse.md §3, §4](./neuron-reuse.md). Reshape the correction path: the per-frame **correction requests**
— units whose **L0 prediction was wrong** despite a context match — are **clustered by transitively merging
neighbor-connected requests**, and **one coordinate-less correction is minted per connected cluster**,
predicting the **correct L0**. Each cluster's requests become the correction's **parents**, so the
**multi-parent ownership/lifecycle/activation machinery lands here too**. Applies at all distances.

Does **not** touch the reverse index or cross-frame lookup/expansion (Phase C).

> **Clustering = transitive merge (connected components), not a bucket-by-observed-set.** Two requests join the
> same cluster if a chain of neighbor links connects them (base = coordinate neighborhood; higher = footprints
> touch). One correction per connected blob; its footprint = the blob's coverage. Corrections are
> coordinate-less, so the union footprint needs **no anchor** to reconcile. (The earlier
> *group-by-identical-observed-set* rule was wrong — see [neuron-reuse.md §3.9](./neuron-reuse.md).)

---

## Goal

Replace per-erroring-neuron minting with **per-cluster** minting, at every distance:

```
requests = collect_correction_requests()      // units whose L0 prediction missed (≥ error threshold)
clusters = transitive_merge(requests, neighbor) // connected components; neighbor = coord nbhd (base) / footprints touch (higher)
for cluster in clusters:
    if cluster.neighbors().is_empty(): continue // need ≥1 neighbor for context — no isolated correction
    C = mint_one(cluster)                       // ONE coordinate-less correction
    C.footprint = ⋃ cluster.footprints          // the cluster's coverage
    C.targets   = correct_L0(cluster)           // every pattern predicts L0
    C.context   = cluster.neighbors()           // the level-below neighbors that recognize it
    for r in cluster: wire_correction(r, C)     // C gets many parents — see Multi-parent
```

Phase C adds, **in front of this**, the reuse lookup: a request that matches an existing pattern (≥ threshold)
**reuses** it instead of minting, and a matched pattern adjacent to a cluster of new requests **expands** to
absorb them ([neuron-reuse-final.md](./neuron-reuse-final.md)). In Phase B alone there is no lookup — every
request mints into a fresh cluster. Per-request error feedback stays per-request (each records its own Welford
sample); clustering changes who-mints, not who-recorded. Iterate clusters in sorted key order for determinism.

This replaces today's per-erroring-neuron mints (spatial one-per-parent
[thalamus.rs](../brain/brain-core/src/thalamus.rs); temporal one-per-(neuron,age)
[thalamus.rs](../brain/brain-core/src/thalamus.rs)) — now unified and clustered.

---

## A minted correction fires next frame, not the mint frame

A minted correction is **installed into the erroring neuron's routing table** (`correct_errors` →
`add_temporal_pattern`, [neuron.rs](../brain/brain-core/src/neuron.rs)) and fires the **next** time
its context recurs — not the mint frame. The `correction_activations` value is **not** an activation of the new
neuron; it records the ages where this neuron just minted a correction so the erroring parent **suppresses its
own vote** at those ages this frame (`get_suppressed_ages`, [neuron.rs](../brain/brain-core/src/neuron.rs):
*"had a bad inference last frame from an age… suppress the vote for that age"*).

So a fresh cluster mint is **not active the mint frame**, and the erroring parent's wrong vote is already
suppressed by the existing machinery. This phase therefore needs **no** `correction_wired_this_frame` set —
fresh mints are covered. That set is only for Phase C, where a *reused, pre-existing* neuron may be
independently active this frame (via its own routing match) while also being wired as a correction target.

---

## Multi-parent: clustering is the first producer

Theory in [neuron-reuse.md §4.1](./neuron-reuse.md). Wiring a cluster's k requests to one correction makes it
multi-parent the moment it exists. Today a correction is owned by one host (strength in that host's routing
entry, dies when that entry decays — [neuron.rs](../brain/brain-core/src/neuron.rs)); the
forget rate is brain-wide uniform ([brain.rs](../brain/brain-core/src/brain.rs)), so per-parent entries
decay at the same rate, differing only in strength and `last_activation_frame`. Corrections are coordinate-less
so there is **no anchor** to reconcile across parents — only lifecycle and activation bookkeeping.

- **Install one child into many parents' routing tables.** The install path wires one correction into one
  parent today ([thalamus.rs](../brain/brain-core/src/thalamus.rs)); the cluster mint installs it into
  every request's table. The routing-table structure (child → entry,
  [neuron.rs](../brain/brain-core/src/neuron.rs)) already permits the same child id across many
  parents' maps — the work is install + lifecycle, not storage.
- **Thalamus activation of a shared neuron.** The frame after a cluster mint, more than one parent can
  match-and-activate the correction in one wave (`activate_*_pattern(pattern_id, level+1)` from each). Today an
  activation carries one `activation.parent_id` and marks exactly that parent subsumed
  ([brain.rs](../brain/brain-core/src/brain.rs)); with N activating parents, **all N** must
  be credited (each routing entry strengthened, each subsumed). The shared neuron is activated **at each
  matched depth** (held by the multi-depth `neuron_states`), **not** fired once — so there is **no**
  `fired_this_frame` set.
- **Refcounted reaping.** Reap only when **no** parent references the correction. The cascade path
  (`DeleteNeuron { target_id, parent_id }`, [column.rs](../brain/brain-core/src/column.rs),
  reaped around [thalamus.rs](../brain/brain-core/src/thalamus.rs)) carries a single `parent_id`; replace
  with a refcount over all referencing parents and scrub every parent's routing entry on death.
- **Multi-parent serialization.** `patterns.csv` = `pattern,parent,strength`
  ([backup.rs](../brain/brain-core/src/backup.rs)) serializes a pattern under one parent;
  a batched correction needs **many** `(parent, strength)` rows. Coordinate with the substrate's format bump.

### `parent_id` reader audit

The **reverse** direction (who references me as context) is already a set; the **forward** direction (a
pattern → its owning parent) is what breaks. Fix forward; leave reverse alone. **No coordinate/label
chain-walk concern** — corrections are coordinate-less, so the root-sensory coordinate walk
([brain.rs](../brain/brain-core/src/brain.rs)) doesn't apply to them; it only ever resolves base
neurons.

| Site | Multi-parent treatment |
|---|---|
| `patterns.csv` + `neuron_parents: pattern → parent` ([backup.rs](../brain/brain-core/src/backup.rs)) | **Breaks.** Many `(parent, strength)` rows per pattern; restore rebuilds an entry in every parent's table. |
| `DeleteNeuron { target_id, parent_id }` + cascade ([column.rs](../brain/brain-core/src/column.rs)) | **Breaks.** Reap over refcount; scrub all referencing parents' routing entries. |
| Subsumption marks `activation.parent_id` ([brain.rs](../brain/brain-core/src/brain.rs)) | Per-activation source is fine, but a shared neuron fired from N parents ⇒ N subsumptions. |
| `get_neuron_parent` ([brain.rs](../brain/brain-core/src/brain.rs)) → one parent | Return the set. |
| `InspectedNeuron.parent_id: Option` ([brain.rs](../brain/brain-core/src/brain.rs)) | Set of parents. Diagnostic only. |
| Context-ref scrub: `get_*_context_refs` ([column.rs](../brain/brain-core/src/column.rs); [neuron.rs](../brain/brain-core/src/neuron.rs)) | **No change** — already a set. |
| `ProcessResult.parent_id` ([column.rs](../brain/brain-core/src/column.rs)) | **No change** — labels the producing neuron. |

> Death-frame / persistence: recompute death frames per-(parent, child) on restore; reap decision over the
> **union** (alive if any entry alive).

---

## Acceptance gates (inline)

- **Unit — cluster mint (d=0 and d>0)**: a connected cluster of correction requests (transitively merged by
  the neighbor relation) produces **exactly one** coordinate-less correction predicting the correct L0,
  footprint = the cluster's coverage, with every request in the cluster wired to it as a parent.
- **Unit — disconnected requests stay separate**: requests with no neighbor chain between them form separate
  corrections.
- **Unit — isolated request makes no correction**: a request with no neighbor produces no correction (the
  ≥1-neighbor / no-context rule) and is never merged with other isolated requests.
- **Unit — shared activation, all parents credited**: the frame after a cluster mint, two parents both
  match-and-activate the correction. **Both** are subsumed and both routing entries strengthened, with the
  neuron activated **per matched depth** (multi-depth `neuron_states`).
- **Unit — refcounted reaping**: a correction with two parents survives one parent's entry dying; reaped only
  when the second dies. Not reaped on the first host's death.
- **Unit — multi-parent serialization round-trip**: snapshot/restore with both parents' routing entries intact
  (independent strengths), then continue identically.
- **MNIST + stocks**: total neuron count drops from clustering + reuse; accuracy ≥ the no-reuse baseline.

---

## Notes / gotchas

- **Wiring fan-out**: a cluster with k requests → k routing entries to **one** correction. The install path
  must take a list of parents for a single child.
- **Footprint of the correction** = the cluster's coverage (the union of the requests' footprints and the
  observed neighborhood the correction predicts). Corrections are coordinate-less, so the union needs no
  representative/anchor.
- **Isolated request**: a request with **no neighbor** (nothing in its neighborhood) has no context to
  condition on, so it forms **no** correction this frame — it is not merged with other isolated requests
  (which would bind disconnected structure and violate locality).
- **Determinism**: cluster iteration in sorted key order, not hash order.
- **Reference simulation**: the corrected spec is [neuron-reuse-simulation.md](./neuron-reuse-simulation.md)
  (now built and faithful — [neuron-reuse-simulation.md](./neuron-reuse-simulation.md)).
