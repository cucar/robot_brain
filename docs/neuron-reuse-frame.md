# Neuron Reuse — Phase C: Batched Mint + Multi-Parent

**The heaviest reuse phase** (Phase A, the wave-front, is heavier still). Theory in
[neuron-reuse.md §3, §4](./neuron-reuse.md). Reshape the correction path so co-failers are grouped and **at
most one coordinate-less correction is minted per (distance, observed-set) group**, with **all** co-failers
wired to it and the correction's footprint = the union. Because that wires many parents to one neuron, the
**multi-parent ownership/lifecycle/activation machinery lands here too**. Applies at **all distances** — d=0
and d>0 are the same code on the wave-front.

Does **not** touch the reverse index or cross-frame lookup (Phase D).

> **No clustering or anchor policy.** The clustering/anchor problem that coordinates created is gone. Corrections are
> coordinate-less (Phase A), so a shared correction takes the **union footprint** — no anchor to reconcile.
> Co-failers group by exact observed-set, which is well-defined. Their own footprints may be **disjoint**
> (they share the observed *target*, not necessarily their own coverage — think four arms around a center),
> so the correction's footprint = union of the co-failers **∪ the observed set** — the whole bound cluster,
> which is connected and unambiguous. No clustering policy, no anchor policy.

---

## Goal

Replace per-erroring-neuron minting with **per-group** minting, at every distance:

```
errors  = collect_errors_from_wave_fixpoint()           // each: (erroring_neuron, distance, observed, footprint)
by_group = errors.group_by(|e| (e.distance, e.observed)) // observed reality singular per (distance, region)
                                                         // empty observed ⇒ its OWN group (isolated, never merged)
for ((distance, observed), errs) in by_group:
    C = mint_one(observed, distance)                    // ONE coordinate-less correction
    C.footprint = (⋃ errs.footprint) ∪ observed         // co-failers ∪ observed set (the bound cluster)
    for e in errs: wire_correction(e.erroring_neuron, C) // C gets many parents — see Multi-parent
```

Phase D adds a **per-request reuse lookup in front of this path**: each erroring neuron first looks up an
existing neuron to reuse, and only the **misses** flow into the grouping + mint above — reused and freshly
minted targets are then wired together (the seam; [neuron-reuse-final.md](./neuron-reuse-final.md)). In Phase C
alone there is no lookup, so every error is residual: group all, mint per group, wire.
Per-neuron error feedback stays per-neuron (each co-failer records its own Welford sample); grouping changes
who-mints, not who-recorded. Iterate groups in sorted key order for determinism.

This replaces today's per-erroring-neuron mints (spatial one-per-parent
[thalamus.rs](../brain/brain-core/src/thalamus.rs); temporal one-per-(neuron,age)
[thalamus.rs](../brain/brain-core/src/thalamus.rs)) — now unified on the wave-front and batched.

---

## A minted correction fires next frame, not the mint frame

A minted correction is **installed into the erroring neuron's routing table** (`correct_errors` →
`add_temporal_pattern`, [neuron.rs](../brain/brain-core/src/neuron.rs)) and fires the **next** time
its context recurs — not the mint frame. The `correction_activations` value is **not** an activation of the new
neuron; it records the ages where this neuron just minted a correction so the erroring parent **suppresses its
own vote** at those ages this frame (`get_suppressed_ages`, [neuron.rs](../brain/brain-core/src/neuron.rs):
*"had a bad inference last frame from an age… suppress the vote for that age"*).

So a fresh batched mint is **not active the mint frame**, and the erroring parent's wrong vote is already
suppressed by the existing machinery. This phase therefore needs **no** `correction_wired_this_frame` set —
fresh mints are covered. That set is only for Phase D, where a *reused, pre-existing* neuron may be
independently active this frame (via its own routing match) while also being wired as a correction target.

---

## Multi-parent: batched mint is the first producer

Theory in [neuron-reuse.md §4.1](./neuron-reuse.md). Wiring k co-failers to one correction makes it
multi-parent the moment it exists. Today a correction is owned by one host (strength in that host's routing
entry, dies when that entry decays — [neuron.rs](../brain/brain-core/src/neuron.rs)); the
forget rate is brain-wide uniform ([brain.rs](../brain/brain-core/src/brain.rs)), so per-parent entries
decay at the same rate, differing only in strength and `last_activation_frame`. Corrections are coordinate-less
(Phase A), so there is **no anchor** to reconcile across parents — only lifecycle and activation bookkeeping.

- **Install one child into many parents' routing tables.** The install path wires one correction into one
  parent today ([thalamus.rs](../brain/brain-core/src/thalamus.rs)); batched mint installs it into
  every co-failer's table. The routing-table structure (child → entry,
  [neuron.rs](../brain/brain-core/src/neuron.rs)) already permits the same child id across many
  parents' maps — the work is install + lifecycle, not storage.
- **Thalamus activation of a shared neuron.** The frame after a batched mint, more than one parent can
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
  a batched correction needs **many** `(parent, strength)` rows. Coordinate with Phase A's format bump.

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

- **Unit — group batch (d=0 and d>0)**: co-failers with the same observed set at the same distance produce
  **exactly one** coordinate-less correction, footprint = union of co-failers ∪ observed set, all wired to it.
- **Unit — distinct observed-sets stay separate**: different observed sets → separate corrections.
- **Unit — shared activation, all parents credited**: the frame after a batched mint, two parents both
  match-and-activate the correction. **Both** are subsumed and both routing entries strengthened, with the
  neuron activated **per matched depth** (multi-depth `neuron_states`).
- **Unit — refcounted reaping**: a correction with two parents survives one parent's entry dying; reaped only
  when the second dies. Not reaped on the first host's death.
- **Unit — multi-parent serialization round-trip**: snapshot/restore with both parents' routing entries intact
  (independent strengths), then continue identically.
- **MNIST + stocks**: total neuron count drops from within-frame dedup; accuracy ≥ the Phase-A baseline.

---

## Notes / gotchas

- **Wiring fan-out**: a group with k co-failers → k routing entries to **one** correction. The install path
  must take a list of parents for a single child.
- **Footprint of the shared correction** = union of co-failers' footprints **∪ the observed set** (the bound
  cluster); co-failers' own footprints may be disjoint, so the observed set is what keeps it connected. No
  representative/anchor needed.
- **Isolated units**: a unit whose observed-set is **empty** (no footprint-adjacent co-active neighbor) is its
  own group — never grouped with other empties. Grouping all empties together binds disconnected structure
  (two separate blobs collapsing into one), which violates locality. Surfaced by the reference simulation.
- **Determinism**: group iteration in sorted key order, not hash order.
- **Executable spec**: [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) runs this
  grouping per level on ASCII shapes (`node apps/mnist/jobs/wavefront-sim.js`) — the reference for what
  batched mint should produce (local level-1 footprints, union∪observed, isolated units kept apart). It models
  one frame with no index lookup, so it exercises this phase's mint path, not Phase D reuse.
