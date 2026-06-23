# Neuron Reuse — Phase A: Levels as Activation State

**Prerequisite phase for reuse.** Theory in [neuron-reuse.md §3.1](./neuron-reuse.md). This phase is a
self-contained, bit-exact refactor with its own backup format bump. Nothing else in the reuse plan moves
while this lands — that isolation is the point.

---

## Goal

Make spatial level a property of a neuron's **activation this frame**, not a value stored per neuron, so
that a single neuron reused from different routing sources at different levels in different frames has no
stale intrinsic level to contradict its activation.

Concretely: **remove the persistent `neuron_spatial_levels` map from the thalamus** and move its eight
readers to either activation-derived values (read the level off the current-frame activation) or recomputed
diagnostics. Per-activation level infrastructure already exists — `spatial_level_index` and
`temporal_level_index` in active memory — so this phase is *deletion and redirection*, not new machinery.

This phase must be **bit-exact**: per-frame computation is unchanged. Nothing here depends on reuse; it is
purely the removal of a redundant persistent copy.

> Naming: this codebase distinguishes `spatial_level` and `temporal_level` and never uses a bare "level"
> for a neuron's hierarchy depth (see project conventions). This doc follows that. `temporal_level` is
> untouched by this phase.

---

## What already exists (do not rebuild)

- `Memory::spatial_level_index: FxHashMap<Level, FxHashSet<NeuronId>>` — per-frame, wiped each frame
  ([memory.rs:57](../brain/brain-core/src/memory.rs), reset at [memory.rs:112](../brain/brain-core/src/memory.rs)).
  Written by `activate_spatial_neuron(neuron_id, level)` ([memory.rs:300-302](../brain/brain-core/src/memory.rs)),
  read by `get_spatial_level_neurons(level)` ([memory.rs:240-241](../brain/brain-core/src/memory.rs)).
- `Memory::temporal_level_index` ([memory.rs:63](../brain/brain-core/src/memory.rs)) — analogous, frame-keyed.
- The spatial sweep already iterates `spatial_level_index` level-by-level
  ([brain.rs:933](../brain/brain-core/src/brain.rs)).

So a neuron's spatial level *as an activation* is already available wherever the neuron is active this
frame. The redundant copy is the persistent map below.

---

## The persistent copy to remove

`Thalamus::neuron_spatial_levels: FxHashMap<NeuronId, Level>` ([thalamus.rs:219](../brain/brain-core/src/thalamus.rs))
— "neuron id → level in the spatial hierarchy; absent entries default to 0." This is the intrinsic copy.
It has **eight readers**, each of which must move off it:

| # | Reader | Location | Replacement |
|---|---|---|---|
| 1 | `get_neuron_spatial_level()` getter | [thalamus.rs:640-641](../brain/brain-core/src/thalamus.rs) | Remove, or re-point to the activation level for the current frame (see Open Question 1). |
| 2 | `count_active_spatial_corrections` | [thalamus.rs:652-653](../brain/brain-core/src/thalamus.rs) | Recompute from the set of neurons that have a non-empty `spatial_routing_table` parent / are pattern (non-sensory) neurons — i.e. an identity property, not a level lookup. |
| 3 | `spatial_level_counts` histogram | [thalamus.rs:659-667](../brain/brain-core/src/thalamus.rs) | Recompute from activation history, or drop if diagnostics-only (see Open Question 2). |
| 4 | `mint_spatial_corrections` partition fired by level | [thalamus.rs:1109-1110](../brain/brain-core/src/thalamus.rs) | Read the parent's **activation level** from `spatial_level_index` for this frame — the parent fired this frame (that's why it can err), so its activation level is present. |
| 5 | `mint_spatial_corrections` parent level → child = parent+1 | [thalamus.rs:1130, 1213](../brain/brain-core/src/thalamus.rs) | Same: `child_level = parent_activation_level + 1`, computed from the activation, not stored on the child. The child isn't activated this frame anyway. |
| 6 | Snapshot generation writes `spatial_level` | [thalamus.rs:1993-1994](../brain/brain-core/src/thalamus.rs) | Drop the field from the snapshot (format bump, below). |
| 7 | Snapshot restoration reads `spatial_level` | [thalamus.rs:2047-2048](../brain/brain-core/src/thalamus.rs) | Drop; nothing to rehydrate. |
| 8 | `get_max_spatial_level()` sweep bound | [thalamus.rs:2187](../brain/brain-core/src/thalamus.rs) | Use `spatial_level_index.keys().max()` for the current frame instead of `neuron_spatial_levels.values().max()`. |
| — | Metadata cleanup on cascade delete | [thalamus.rs:1970](../brain/brain-core/src/thalamus.rs) | Delete with the map; no replacement needed. |

The critical readers are #4 and #5 — the mint path. Everything else is diagnostics, serialization, or the
sweep bound.

---

## Design

### Mint reads the activation, not the map

Today `mint_spatial_corrections` partitions the fired set by `neuron_spatial_levels` and derives the child
correction's level as `parent_level + 1`. After this phase:

- The parent fired this frame, so it is present in `spatial_level_index[L]` for some L. That L **is** the
  parent's level for this frame — read it from active memory.
- `child_level = L + 1` is recorded as an **activation fact** only when the child next fires (via its
  routing entry). The child is **not** activated the frame it is minted
  ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs): "patterns NOT activated this frame, match
  next frame"), so there is nothing to store now. Next frame, when a source routes to the child, the child
  is registered at `source_activation_level + 1` like any activation.

This is why the persistent copy is removable: the only thing that ever needed the child's level was the
*next* mint above it, and that reads from the activation index by then.

### Sweep bound from the index

`get_max_spatial_level()` currently scans every neuron's stored level. After this phase it reads the max
key present in this frame's `spatial_level_index` — which is exactly "how deep did the sweep get this
frame," the quantity the bound actually wants.

### Backup format bump

`neurons.csv` is `id,temporal_level,spatial_level` ([backup.rs:10, 358-369](../brain/brain-core/src/backup.rs)),
and load tolerates a missing 3rd column by defaulting to 0
([backup.rs:138-157](../brain/brain-core/src/backup.rs)). After this phase, `spatial_level` is no longer a
neuron property:

- **Write**: drop the `spatial_level` column → `neurons.csv` becomes `id,temporal_level`.
- **Read**: the existing `row.len() >= 3` guard already tolerates *absence*. The bump is the reverse —
  newer code reading an **older** backup that *has* a 3rd column must **ignore** it, not choke. Add an
  explicit version marker or simply read columns positionally and discard extras. Recommend a one-line
  format version in the backup header so the intent is explicit rather than relying on column counting.
- Old backups load (spatial_level ignored); new backups omit it. State trained under the old format still
  restores — spatial level is reconstructed from routing structure on the first frame, not from the file.

> Decision needed if regression baselines (the 95.73% MNIST run, stocks runs) are stored as backups and
> reloaded for comparison: confirm they restore cleanly under the new reader before relying on them as the
> bit-exact baseline. If `temporal_level` is *also* derivable, consider whether it stays — out of scope
> here; this phase touches spatial only.

---

## Acceptance gates (inline)

- **Bit-exact regression**: stocks demo produces byte-identical per-frame output and identical final
  metrics vs the pre-phase build. This is the primary gate — the whole phase is "remove a redundant copy
  with zero behavioral effect."
- **Backup round-trip**: train N frames → snapshot → restore → continue → identical trajectory to a run
  that never snapshotted. Both old-format (3-column) and new-format (2-column) `neurons.csv` load.
- **Diagnostics unchanged**: `count_active_spatial_corrections` and the per-level histogram report the
  same numbers as before on a fixed MNIST run (or are explicitly dropped — see Open Question 2 — and
  nothing downstream consumes them).
- **Unit**: mint a spatial correction from a parent activated at spatial level k; confirm that on the
  *next* frame the correction, when routed, registers at activation level k+1 in `spatial_level_index`.

---

## Open questions

1. **Does any non-mint caller need `get_neuron_spatial_level(id)` for a neuron that is *not active this
   frame*?** If yes, the activation index can't answer it (it's per-frame) and we'd need a fallback. Audit
   the getter's callers; the table above assumes all live uses are either mint (active this frame),
   diagnostics (recomputable), or serialization (dropped). Confirm before deleting the getter outright.
2. **Keep or drop the per-level histogram diagnostic?** `spatial_level_counts` is "diagnostic — surfaced
   via the harness." If the harness still wants it, recompute from activation history over a frame window;
   if it was only used during spatial-processing bring-up, drop it. Cheap either way; decide based on
   whether any current validation reads it.
