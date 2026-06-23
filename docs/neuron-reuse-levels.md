# Neuron Reuse — Phase A: Levels as Activation State

**Prerequisite phase for reuse.** Theory in [neuron-reuse.md §3.1](./neuron-reuse.md). Two moves, landed
together: **delete both intrinsic level collections from the thalamus**, and **extend active memory so one
neuron can be active at multiple levels at once**. Bit-exact, with a backup format bump. Nothing else in
the reuse plan moves while this lands — that isolation is the point.

---

## Goal

Make a neuron's level a property of its **activation**, never a value stored on the neuron. The driving
constraint is that a shared correction can be routed-to by **parents at different depths**. This first
becomes possible in **Phase C** (batched mint, if its clustering groups co-failers across spatial levels —
[DECIDE-THIS #0](./neuron-reuse-frame.md)) and again in **Phase D** (cross-frame reuse from a different-depth
source): **two parents at different levels route to one neuron R within a single frame**, so R is active at,
say, level 3 and level 6 *at the same time*. No per-neuron level field can represent that. Only the
activation can, and active memory must be able to hold the multiplicity.

Two concrete deliverables:

1. **Delete** `neuron_spatial_levels` *and* `neuron_temporal_levels` from the thalamus
   ([thalamus.rs:219](../brain/brain-core/src/thalamus.rs)). After this phase no neuron carries an intrinsic
   level of either kind. Every reader moves to an activation-derived or recomputed value.
2. **Extend active memory** to represent a neuron active at multiple levels simultaneously, for both
   hierarchies. The spatial index is already `level → {neuron}` and tolerates a neuron under several levels;
   the temporal side does **not** (see below) and must change.

This phase is **bit-exact** — but for a precise reason, and *not* because "each correction has one parent"
is a stable invariant. That single-parent property is itself something reuse abolishes: a reused neuron is,
by definition, routed-to by many parents (see [neuron-reuse.md §3.4](./neuron-reuse.md)). The bit-exactness
holds because **no multi-parent / multi-level producer exists until Phase C**. Phase A only builds the
capability — multi-level memory, no intrinsic level — and leaves it dormant. With no batched mint or lookup
yet, every correction is still minted and routed-to by exactly one parent this phase, so every neuron still
activates at one level and an activation-derived level equals the old intrinsic level for every neuron, every
frame. The capability does nothing until Phase C first exercises it.

> Naming: this codebase distinguishes `spatial_level` and `temporal_level` and never uses a bare "level"
> for a neuron's hierarchy depth (project convention). This doc follows that.

---

## Why per-frame rebuilding does not already solve it

An earlier draft of this plan claimed the per-frame `spatial_level_index` sidesteps the problem because it
is rebuilt each frame. **That is wrong.** The conflict is not across frames — it is *within* a single frame.
A neuron reused from two different-depth sources in one frame must be represented at two levels in *that*
frame's index. Rebuilding the index next frame is irrelevant; the multiplicity is needed now.

So "level is per-activation" is not satisfied for free by the existing index. The index has to *support*
multi-level membership, and the temporal representation currently doesn't.

---

## What memory must become

### Spatial — already close

`spatial_level_index: FxHashMap<Level, FxHashSet<NeuronId>>` ([memory.rs:57](../brain/brain-core/src/memory.rs))
is `level → set of neuron ids`, wiped each frame ([memory.rs:112](../brain/brain-core/src/memory.rs)).
Structurally a neuron can already be a member of several levels' sets, and the spatial sweep fabricates a
default per-neuron state at age 0 on demand rather than persisting one
([memory.rs:54-56](../brain/brain-core/src/memory.rs)). **Audit** the spatial sweep, the apex computation,
and the subsumption tracking ([spatial-processing.md §3.3](./spatial-processing.md)) for any spot that
assumes a fired neuron has a *single* spatial level — those assumptions must go even if the storage already
permits multiplicity.

### Temporal — must change

`neuron_states: FxHashMap<NeuronId, FxHashMap<FrameNumber, LevelAgeState>>`
([memory.rs:45](../brain/brain-core/src/memory.rs)) stores **one** `LevelAgeState` per `(neuron, frame)`.
That is a single level per neuron per activation-frame — it **cannot** represent R at two temporal levels in
one frame. This is the substantive representational change of the phase: allow multiple level states for one
`(neuron, frame)`, or re-key so level is part of the activation key rather than a single stored field.
`temporal_level_index: Level → FrameNumber → {NeuronId}`
([memory.rs:63](../brain/brain-core/src/memory.rs)) already keys level outermost and tolerates a neuron
under several levels; the blocker is the single-valued `neuron_states` entry, not the level index.

> Design decision for this phase: pick the multi-level temporal representation — (a) `(neuron, frame) →
> Vec<LevelAgeState>` / set keyed by level, vs (b) fold level into the activation key so `neuron_states`
> becomes `(neuron, frame, level) → state`. (b) is closer to how the spatial `level → set` already works
> and keeps eviction-by-frame intact. Decide on a benchmark / code-shape basis; flagged as Open Question 3.

---

## The intrinsic copies to remove

Both live on the thalamus, stored separately because a neuron occupies independent positions in the two
hierarchies:

- `neuron_spatial_levels: FxHashMap<NeuronId, Level>` ([thalamus.rs:219](../brain/brain-core/src/thalamus.rs)).
- `neuron_temporal_levels: FxHashMap<NeuronId, Level>` (sibling map, same site).

### Spatial readers (audited)

| # | Reader | Location | Replacement |
|---|---|---|---|
| 1 | `get_neuron_spatial_level()` getter | [thalamus.rs:640-641](../brain/brain-core/src/thalamus.rs) | Remove, or re-point to the current-frame activation level (Open Question 1). |
| 2 | `count_active_spatial_corrections` | [thalamus.rs:652-653](../brain/brain-core/src/thalamus.rs) | Recompute from an identity property (is the neuron a pattern vs sensory), not a level lookup. |
| 3 | `spatial_level_counts` histogram | [thalamus.rs:659-667](../brain/brain-core/src/thalamus.rs) | Recompute from activation history, or drop (Open Question 2). |
| 4 | `mint_spatial_corrections` partition fired by level | [thalamus.rs:1109-1110](../brain/brain-core/src/thalamus.rs) | Read the parent's **activation level** from `spatial_level_index` this frame (the parent fired → it's present). |
| 5 | `mint_spatial_corrections` child = parent+1 | [thalamus.rs:1130, 1213](../brain/brain-core/src/thalamus.rs) | `child_level = parent_activation_level + 1`, computed from the activation, never stored on the child. |
| 6 | Snapshot writes `spatial_level` | [thalamus.rs:1993-1994](../brain/brain-core/src/thalamus.rs) | Drop from the snapshot (format bump). |
| 7 | Snapshot restores `spatial_level` | [thalamus.rs:2047-2048](../brain/brain-core/src/thalamus.rs) | Drop; nothing to rehydrate. |
| 8 | `get_max_spatial_level()` sweep bound | [thalamus.rs:2187](../brain/brain-core/src/thalamus.rs) | `spatial_level_index.keys().max()` for this frame. |
| — | Cleanup on cascade delete | [thalamus.rs:1970](../brain/brain-core/src/thalamus.rs) | Deleted with the map. |

### Temporal readers (audit required)

The temporal-level readers were not enumerated as exhaustively as the spatial ones — **audit `neuron_temporal_levels`
callers before deleting**. Known sites and their replacements:

- **Snapshot** carries `temporal_level` ([thalamus.rs:179](../brain/brain-core/src/thalamus.rs)) and
  `neurons.csv` serializes it ([backup.rs:10, 358-369](../brain/brain-core/src/backup.rs)). Drop from both
  (format bump, below).
- **Temporal mint** derives a child correction's temporal level from its parent — same treatment as spatial
  #4/#5: read the parent's activation level from `temporal_level_index` this frame, child = parent + 1,
  never stored. Confirm the exact site in the temporal correction path
  ([thalamus.rs:431-487, 1471-1497](../brain/brain-core/src/thalamus.rs)).
- **Temporal sweep bound** — analog of `get_max_spatial_level`; use the max key in `temporal_level_index`.
- **Diagnostics / `depth` / `get_base_level`** — recompute from the activation index or drop.

> Action: run a reference search for every read of `neuron_temporal_levels` and slot each into the same
> three buckets the spatial table uses (mint → activation-derived, diagnostics → recompute/drop,
> serialization → drop). Open Question 4.

---

## Design notes

### Mint reads the activation, not the map (both hierarchies)

The parent erred *because it fired* this frame, so it is present in the relevant level index
(`spatial_level_index` or `temporal_level_index`) at some level L. Read L from there; the child correction's
level is `L + 1` as a derivation, recorded nowhere. When the child next fires via the routing entry just
installed, it is registered at `(whoever routed to it).activation_level + 1` — which, under reuse, may be a
*different* number than this mint's `L + 1`, and may be more than one number in one frame. That is exactly
the multiplicity the memory change above enables.

### Backup format bump

`neurons.csv` is `id,temporal_level,spatial_level` ([backup.rs:10, 358-369](../brain/brain-core/src/backup.rs)),
load tolerating a missing 3rd column ([backup.rs:138-157](../brain/brain-core/src/backup.rs)). After this
phase, **neither** level is a neuron property:

- **Write**: `neurons.csv` becomes `id` (plus whatever non-level per-neuron fields remain).
- **Read**: newer code reading an older backup with the extra level columns must **ignore** them, not choke.
  Recommend an explicit format-version marker in the backup header rather than relying on column counting,
  since two columns are being dropped at once.
- **Heads-up for the bump**: Phase C needs a *second* schema change — `patterns.csv` goes from
  `pattern,parent,strength` (one parent per pattern) to many `(parent, strength)` rows per pattern, for
  multi-parent batched mint ([neuron-reuse-frame.md](./neuron-reuse-frame.md), `parent_id` audit). Decide now
  whether to introduce the format-version marker here and widen `patterns.csv` once, rather than bumping the
  format twice across A and D.
- Old backups load (level columns ignored); levels reconstruct from routing structure as neurons fire, not
  from the file.

> If regression baselines (the 95.73% MNIST run, stocks runs) are stored as backups and reloaded for the
> bit-exact comparison, confirm they restore cleanly under the new reader before relying on them. Tracked
> in [neuron-reuse-validation.md](./neuron-reuse-validation.md).

---

## Acceptance gates (inline)

- **Bit-exact regression**: stocks demo is byte-identical per-frame and on final metrics vs the pre-phase
  build. Primary gate — behavior is unchanged; only the level *representation* moved.
- **Multi-level representable (structural unit test)**: directly activate one neuron at two levels in one
  frame in active memory (both spatial and temporal indexes) and confirm both memberships and both per-level
  states are retrievable. This proves the groundwork even though no production path exercises it yet.
- **Backup round-trip**: train → snapshot → restore → continue → identical trajectory to a never-snapshotted
  run. Old-format (with level columns) and new-format (without) both load.
- **Diagnostics unchanged or cleanly dropped**: `count_active_spatial_corrections` and the histograms report
  the same numbers, or are removed with nothing downstream consuming them.
- **Mint level unit**: mint a correction from a parent activated at level k; on the next frame, when routed,
  it registers at activation level k+1 — for both spatial and temporal.

---

## Open questions

1. **`get_neuron_spatial_level(id)` for an *inactive* neuron** — any caller needs a neuron's level when it
   is not active this frame? The activation index can't answer that (per-frame). Audit the getter's callers;
   the table assumes all live uses are mint (active), diagnostics (recomputable), or serialization (dropped).
2. **Keep or drop the per-level histogram diagnostic?** Recompute from activation history if the harness
   reads it; drop if it was only spatial-bring-up scaffolding.
3. **Multi-level temporal representation shape** — `(neuron, frame) → many states` vs `(neuron, frame, level)
   → state`. Pick on code-shape / eviction grounds (see "Temporal — must change").
4. **Full `neuron_temporal_levels` reader audit** — enumerate every caller and bucket it (mint / diagnostics
   / serialization) before deleting the map.
