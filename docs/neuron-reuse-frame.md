# Neuron Reuse — Phase C: Batched Mint + Multi-Parent

**The heaviest reuse phase** (Phase A, the wave-front, is heavier still). Theory in
[neuron-reuse.md §3, §4](./neuron-reuse.md). Reshape the correction path so co-failers are grouped and **at
most one coordinate-less correction is minted per (distance, observed-set) group**, with **all** co-failers
wired to it and the correction's footprint = the union. Because that wires many parents to one neuron, the
**multi-parent ownership/lifecycle/activation machinery lands here too**. Applies at **all distances** — d=0
and d>0 are the same code on the wave-front.

Does **not** touch the reverse index or cross-frame lookup (Phase D). Settles **DECIDE-THIS #1**.

> **No DECIDE-THIS #0.** The clustering/anchor problem that coordinates created is gone. Corrections are
> coordinate-less (Phase A), so a shared correction takes the **union footprint** — no anchor to reconcile.
> And co-failers group by exact observed-set, which is well-defined; same observed-set ⇒ overlapping
> footprints, so the union is unambiguous. No clustering policy, no anchor policy.

---

## Goal

Replace per-erroring-neuron minting with **per-group** minting, at every distance:

```
errors  = collect_errors_from_wave_fixpoint()           // each: (erroring_neuron, distance, observed, footprint)
by_group = errors.group_by(|e| (e.distance, e.observed)) // observed reality singular per (distance, region)
for ((distance, observed), errs) in by_group:
    C = mint_one(observed, distance)                    // ONE coordinate-less correction
    C.footprint = ⋃ errs.footprint                      // union; no anchor
    for e in errs: wire_correction(e.erroring_neuron, C) // C gets many parents — see Multi-parent
```

In Phase D the loop body becomes `lookup(observed, distance)` first, `mint_one` only on miss — the seam.
Per-neuron error feedback stays per-neuron (each co-failer records its own Welford sample); grouping changes
who-mints, not who-recorded. Iterate groups in sorted key order for determinism.

This replaces today's per-erroring-neuron mints (spatial one-per-parent
[thalamus.rs:1205-1229](../brain/brain-core/src/thalamus.rs); temporal one-per-(neuron,age)
[thalamus.rs:1471-1497](../brain/brain-core/src/thalamus.rs)) — now unified on the wave-front and batched.

---

## DECIDE-THIS #1 — Mint-frame vs reuse-frame inhibition window

The correction-wired inhibition ([neuron-reuse.md §4.2](./neuron-reuse.md)) says a corrected neuron learns the
observed set but does not vote and is not error-checked during its inhibition window. *Which frame?*

Corrections produce `correction_activations` out of the pass today
([neuron.rs:1101, 1461-1464](../brain/brain-core/src/neuron.rs)), suggesting they activate the mint frame.
**Recommended (confirm against code):** a freshly minted correction **does** activate its mint frame as a
tagged correction-activation — learns the group's observed set, does not vote, is not error-checked (its
one-frame window); on subsequent frames it fires via routing as a normal recognition. Trace
`correction_activations` through the thalamus to confirm, and write the answer into
[neuron-reuse.md §4.2](./neuron-reuse.md).

> Batched mint alone mints **fresh** neurons, which already carry the fresh-mint exemption
> ([spatial-processing.md §3.3 step 2](./spatial-processing.md)). So this phase needs no
> `correction_wired_this_frame` set; it becomes load-bearing only in Phase D for *reused* (non-fresh)
> neurons.

---

## Multi-parent: batched mint is the first producer

Theory in [neuron-reuse.md §4.1](./neuron-reuse.md). Wiring k co-failers to one correction makes it
multi-parent the moment it exists. Today a correction is owned by one host (strength in that host's routing
entry, dies when that entry decays — [neuron.rs:64, 689-703, 709-714](../brain/brain-core/src/neuron.rs)); the
forget rate is brain-wide uniform ([brain.rs:366](../brain/brain-core/src/brain.rs)), so per-parent entries
decay at the same rate, differing only in strength and `last_activation_frame`. Corrections are coordinate-less
(Phase A), so there is **no anchor** to reconcile across parents — only lifecycle and activation bookkeeping.

- **Install one child into many parents' routing tables.** The install path wires one correction into one
  parent today ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs)); batched mint installs it into
  every co-failer's table. The routing-table structure (child → entry,
  [neuron.rs:242-243](../brain/brain-core/src/neuron.rs)) already permits the same child id across many
  parents' maps — the work is install + lifecycle, not storage.
- **Thalamus activation of a shared neuron.** The frame after a batched mint, more than one parent can
  match-and-activate the correction in one wave. Collapse to one activation (refractory, via `fired_this_frame`)
  while crediting **every** activating parent: each routing entry strengthened, each subsumed. Today an
  activation carries one `activation.parent_id` and marks exactly that parent subsumed
  ([brain.rs:1151-1157, 1255-1261](../brain/brain-core/src/brain.rs)); with N activating parents, **all N**
  must be subsumed. → `fired_this_frame` lands here.
- **Refcounted reaping.** Reap only when **no** parent references the correction. The cascade path
  (`DeleteNeuron { target_id, parent_id }`, [column.rs:47, 218, 258-322](../brain/brain-core/src/column.rs),
  reaped around [thalamus.rs:1970](../brain/brain-core/src/thalamus.rs)) carries a single `parent_id`; replace
  with a refcount over all referencing parents and scrub every parent's routing entry on death.
- **Multi-parent serialization.** `patterns.csv` = `pattern,parent,strength`
  ([backup.rs:13, 204-232, 412-436](../brain/brain-core/src/backup.rs)) serializes a pattern under one parent;
  a batched correction needs **many** `(parent, strength)` rows. Coordinate with Phase A's format bump.

### `parent_id` reader audit

The **reverse** direction (who references me as context) is already a set; the **forward** direction (a
pattern → its owning parent) is what breaks. Fix forward; leave reverse alone. **No coordinate/label
chain-walk concern** — corrections are coordinate-less, so the root-sensory coordinate walk
([brain.rs:2048-2054](../brain/brain-core/src/brain.rs)) doesn't apply to them; it only ever resolves base
neurons.

| Site | Multi-parent treatment |
|---|---|
| `patterns.csv` + `neuron_parents: pattern → parent` ([backup.rs:13, 204-232](../brain/brain-core/src/backup.rs)) | **Breaks.** Many `(parent, strength)` rows per pattern; restore rebuilds an entry in every parent's table. |
| `DeleteNeuron { target_id, parent_id }` + cascade ([column.rs:47, 218, 258-322](../brain/brain-core/src/column.rs)) | **Breaks.** Reap over refcount; scrub all referencing parents' routing entries. |
| Subsumption marks `activation.parent_id` ([brain.rs:1151-1157, 1255-1261](../brain/brain-core/src/brain.rs)) | Per-activation source is fine, but a shared neuron fired from N parents ⇒ N subsumptions. |
| `get_neuron_parent` ([brain.rs:785, 2053](../brain/brain-core/src/brain.rs)) → one parent | Return the set. |
| `InspectedNeuron.parent_id: Option` ([brain.rs:137, 789](../brain/brain-core/src/brain.rs)) | Set of parents. Diagnostic only. |
| Context-ref scrub: `get_*_context_refs` ([column.rs:264-275](../brain/brain-core/src/column.rs); [neuron.rs:449-458](../brain/brain-core/src/neuron.rs)) | **No change** — already a set. |
| `ProcessResult.parent_id` ([column.rs:21-25, 163, 194](../brain/brain-core/src/column.rs)) | **No change** — labels the producing neuron. |

> Death-frame / persistence: recompute death frames per-(parent, child) on restore; reap decision over the
> **union** (alive if any entry alive).

---

## Acceptance gates (inline)

- **Unit — group batch (d=0 and d>0)**: co-failers with the same observed set at the same distance produce
  **exactly one** coordinate-less correction, footprint = union, all wired to it.
- **Unit — distinct observed-sets stay separate**: different observed sets → separate corrections.
- **Unit — shared activation, all parents subsumed**: the frame after a batched mint, two parents both
  match-and-activate the correction. It fires **once** (refractory); **both** subsumed; both routing entries
  strengthened.
- **Unit — refcounted reaping**: a correction with two parents survives one parent's entry dying; reaped only
  when the second dies. Not reaped on the first host's death.
- **Unit — multi-parent serialization round-trip**: snapshot/restore with both parents' routing entries intact
  (independent strengths), then continue identically.
- **MNIST + stocks**: total neuron count drops from within-frame dedup; accuracy ≥ the Phase-A baseline.

---

## Notes / gotchas

- **Wiring fan-out**: a group with k co-failers → k routing entries to **one** correction. The install path
  must take a list of parents for a single child.
- **Footprint of the shared correction** = union of co-failers' footprints; no representative/anchor needed.
- **Determinism**: group iteration in sorted key order, not hash order.
