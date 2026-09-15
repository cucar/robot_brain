# Wave-Front Processing + Footprints

The wave-front keeps the existing processing **structure** — spatial processing → apex handoff → temporal
processing — but turns each stage into a **settling wave**: it removes stored levels, makes **all** corrections
coordinate-less, and introduces **footprints** as the universal neighborhood primitive. A correction needs no
coordinate because output resolves only to base neurons and footprints carry locality.

It is **not bit-exact** — it restructures processing. Its gate is a *characterized* regression (MNIST + stocks
still learn comparably), not byte-identity.

---

## Structure is unchanged: two waves + a handoff

`process_spatial` and `process_temporal` **stay separate functions**. The structure is the same as today:

1. **`process_spatial`** (d=0) — relationships *within* the current frame.
2. **Apex handoff** — the spatial apex feeds temporal, exactly as today.
3. **`process_temporal`** (d>0) — relationships between the current frame and past frames.

**Both already run the same settling level loop** — `process_spatial_levels`
([brain.rs](../brain/brain-core/src/brain.rs)) and `process_temporal_levels`
([brain.rs](../brain/brain-core/src/brain.rs)) are the identical loop: walk levels bottom-up, each level's
matched patterns activate one level up, stop when a level produces no activations that push the hierarchy
higher. The active set is fixed at the start of the climb; freshly-minted error patterns are excluded from
their own level and fire *next* frame. So the "wave" is **not new** — it is this existing loop, just with the
level no longer stored as an intrinsic neuron field.

They stay two functions for the reasons they are two today, both narrow: temporal **persists per-neuron state
across frames** (`write_back_level_neurons`, [brain.rs](../brain/brain-core/src/brain.rs)) and reads d>0
connections; spatial is **ephemeral** ([brain.rs](../brain/brain-core/src/brain.rs)) and reads d=0. The
level loop itself is the same.

There is **no** `process_ages` collapse and **no** age-to-age chaining — the apex fans out to every temporal
distance in parallel; d=1 does not feed d=2.

---

## One correction per parent

Every correction is minted from exactly one erroring parent and lives in that parent's routing table — one
correction per erroring parent, as today (`neuron_parents` is scalar,
[thalamus.rs](../brain/brain-core/src/thalamus.rs)). This is what keeps per-neuron state simple: a
correction with a single parent has a single activation path, so the climb reaches it at exactly **one depth per
frame**. Per-neuron state therefore stays keyed by `(neuron, frame)` — no depth dimension. (That key already
handles the same neuron active at several *ages*; ages are the `frame` dimension, distinct from depth.)

---

## No stored levels

The level loop already settles bottom-up until no higher level fires (above). What changes here is that
**the level is no longer a stored intrinsic neuron field** — it is only the loop variable. The loop
already drives off `max_active_level` read from memory's activation index, not the stored
`neuron_spatial_levels` / `neuron_temporal_levels` maps. Those maps have four readers, all of which move to
activation-derived, recomputed, or predicate-based values:

1. **mint** (child level = parent level + 1) — under the wave the parent's level *is* the loop variable at the
   depth it fired, so mint reads it from the loop, not the map;
2. **diagnostics** (depth, per-level counts) — recomputed from the activation index;
3. **serialization** — the level columns are dropped from `neurons.csv` (below);
4. **base-neuron identity** — `skip_action_neuron` ([thalamus.rs](../brain/brain-core/src/thalamus.rs)) uses
   `neuron_temporal_levels.get(id) != Some(&0)` as a "is this a base neuron" test. This is **not** a level read
   in disguise; it needs an explicit base-neuron predicate (base-neuron registry membership, equivalently
   `footprint == {self}`). Replacing it is part of removing the maps — miss it and the action-skip silently
   breaks.

A neuron's depth becomes purely "how far the loop had climbed when it fired," never recorded on the neuron.

The loop, unchanged in shape (only neighborhood and coordinate handling differ — see below):

```
process_spatial (d=0):
  activate base sensory neurons          // each: footprint = {self}, coordinate + base-graph neighbors given
  repeat until fixpoint:
     fire d=0 routing matches            // activate existing corrections whose constituents are ready
     for each active neuron N:
        observed  = co-active neurons whose footprint is adjacent to N's    // neighborhood
        predicted = N's d=0 connection targets
        if mismatch > threshold:
           C = correction for N over its footprint-adjacent neighborhood    // coordinate-less
           C.footprint = footprint(N) ∪ ⋃ neighborhood footprints
           install C in N's routing table (context = neighborhood)
     stop when a pass fires nothing new and mints nothing

apex handoff: apex (non-subsumed spatial fired set) → temporal

process_temporal (d>0):
  same wave, reading d>0 connections over the materialized window; votes for future frames
```

---

## Footprints — neighborhood without coordinates

The mechanism that lets corrections be coordinate-less while keeping locality at every level.

- A neuron's **footprint** = the set of **base sensory neurons** it ultimately covers.
  - Base sensory neuron: `footprint = {itself}` (and it keeps its coordinate + base-graph neighbors).
  - Correction (any distance): `footprint = union of its constituents' footprints`, computed at mint. Its
    constituents are its **context set** — the parent plus the co-active neighbors the correction binds.
- **Neighborhood** at any level: A and B are neighbors iff their footprints **touch in the base neighbor
  graph** — `∃ base a ∈ footprint(A), base b ∈ footprint(B)` with `a`, `b` spatial-neighbors (or equal).
- This replaces **both** coordinate-inheritance (spatial RF growth) **and** channel-neighbor filtering
  (`is_spatial_neighbor_channel` / `is_temporal_neighbor_channel` — [thalamus.rs](../brain/brain-core/src/thalamus.rs)).
  Receptive fields still grow one ring per layer — the union footprint widens naturally.

**What this changes vs. today (the regression expectation).** The brain already has spatial locality — it is
just expressed at *channel* granularity. Retinotopic MNIST registers **one channel per pixel** and the encoder
wires each pixel's `(2r+1)²` window via `setSpatialNeighbors` ([encoder.js](../apps/mnist/encoder.js)), so
`is_spatial_neighbor_channel` *is* the pixel-radius neighborhood. Footprints are therefore **not** introducing
locality where there was none:

- **At L0**, footprint touch in the base neighbor graph ≈ today's channel-neighbor window. Near-equivalent — the
  characterized regression should be **tightest here**, and config parity on `radius` / connectivity is what
  makes that equivalence hold (see [wavefront-implementation.md](./wavefront-implementation.md) §2).
- **At L1+**, today's correction inherits its parent's *single pixel coordinate* and filters by that one pixel's
  window. Footprints replace that single-pixel anchor with the **union of covered pixels**, so the receptive
  field grows one ring per layer. **This is the real behavioral change**, and the intended one: graded locality.
  Expect the numbers to **move** here, not reproduce the leveled baseline — which is exactly why the gate is
  characterized regression, not bit-exactness.

**Representation:** footprint as a **bitset over base sensory neurons** (49–784 bits for MNIST — trivial).
Adjacency = dilate one footprint by the precomputed base neighbor-ring and AND with the other; nonzero ⇒
neighbors. (Conceptually a footprint is a cache of connection-graph reachability to base; the bitset makes
adjacency O(words).)

Footprint-neighborhoods apply in **both** waves — d>0 relationships are footprint-local too, so locality is
graded: low-level patterns are local, high-level ones span more as footprints grow. Cross-region /
cross-channel relationships emerge through grouping, not as a raw cross-product.

---

## Coordinates: only on base neurons

After this phase, **only base sensory/action neurons carry coordinates** — the neurons *input to spatial
processing*. All corrections (every distance, either wave) are coordinate-less — identified by id, with a
footprint for locality.

This is **already safe on the output path** — verified, not assumed. `aggregate_votes` already **forbids a
pattern neuron from being a vote target** ([brain.rs](../brain/brain-core/src/brain.rs)):

> "Every vote MUST target a neuron with a coordinate — sensory events and actions have coordinates by
> construction. A vote toward a target without a coordinate means we learned a connection to a pattern
> neuron upstream, which is a real architectural break."

So a correction is **never** a candidate → never a dimension winner → never dequantized
([brain.rs](../brain/brain-core/src/brain.rs)). Output already resolves exclusively to
base sensory/action neurons. Removing the correction coordinate touches nothing on the prediction/action layer;
the existing "no-coordinate ⇒ must not be a vote target" panic stays valid and becomes the load-bearing
invariant.

**Three current uses of a correction's inherited coordinate must each be retired** — the mint paths
([allocate_temporal_pattern_neuron](../brain/brain-core/src/thalamus.rs),
[allocate_spatial_pattern_neuron](../brain/brain-core/src/thalamus.rs)) inherit the parent's full coordinate into
`base_neurons` today, and the comments there cite three consumers:

1. **Neighbor filtering** — replaced by footprint adjacency (above). This is the only use the "coordinate-less"
   change strictly requires, and it goes away once footprints are load-bearing.
2. **Consensus grouping** — confirm `aggregate_votes` / consensus never reads a correction's coordinate (it
   shouldn't: corrections are forbidden as vote targets, so they never reach the per-dimension grouping). Verify,
   don't assume.
3. **Restore** — today restore rederives a correction's coordinate by **walking `neuron_parents` to the L0
   ancestor**. Coordinate-less corrections make this walk unnecessary, and the pattern's `base_neurons` insert at
   mint is dropped. The backup/restore changes ride with the stored-level removal (format-version bump, below).

Order matters: switch neighborhoods to footprint adjacency **first**, then drop the coordinate — otherwise
neighbor filtering loses its input mid-migration.

---

## Code touched (large — this is a rearchitecture)

- **`brain/brain-core/src/thalamus.rs`** — convert `process_spatial` and `process_temporal` to settling waves
  (propagate to fixpoint instead of a counted level loop). Delete **both** `neuron_spatial_levels` and
  `neuron_temporal_levels` ([thalamus.rs](../brain/brain-core/src/thalamus.rs)) and all four readers (mint
  child-level → loop variable; level bounds/diagnostics → activation-derived; snapshot → dropped columns;
  base-neuron identity in `skip_action_neuron` → explicit base predicate). Delete channel-neighbor machinery
  (`temporal_channel_neighbors`, `is_temporal_neighbor_channel`, `set_temporal_neighbors`, spatial
  equivalents) in favor of footprint adjacency.
- **`brain/brain-core/src/neuron.rs`** — drop coordinate/dimension inheritance for corrections
  ([neuron.rs](../brain/brain-core/src/neuron.rs)); add a `footprint` (bitset) to each neuron; correction
  mint computes `footprint = ⋃ constituents`. Connection/error/learn steps filter by footprint adjacency
  instead of channel-neighbor.
- **`brain/brain-core/src/memory.rs`** — **no shape change.** Each correction is reached at one depth per frame
  (one correction per parent), so the existing `LevelAgeState` per `(neuron, frame)`
  ([memory.rs](../brain/brain-core/src/memory.rs)) is sufficient — it already handles the same neuron at multiple
  *ages* (the `frame` dimension), which is distinct from depth. The level *indices* (`spatial_level_index`
  / `temporal_level_index`) remain as the wave's activation state.
- **`brain/brain-core/src/brain.rs`** — drive the two waves + apex handoff (structure unchanged); voting
  unchanged (already base-only).
- **`brain/brain-core/src/backup.rs`** — drop both level columns from `neurons.csv`; drop the correction
  coordinate (corrections are coordinate-less; only base neurons keep coordinates in `base_neurons.csv`); remove
  the restore parent-walk that rederived correction coordinates. Rebuild footprints from the connection graph on
  load (don't serialize them) — the rebuild must run in **dependency order** (a correction's footprint = ⋃ its
  constituents', so constituents must be built first); with stored levels gone there is no level column to sort
  by, so memoize a recursion from base outward (or topo-sort the constituent graph). Format-version bump.

---

## Acceptance gates (inline)

- **Characterized regression**: MNIST and stocks still learn and produce comparable accuracy / neuron counts
  to the leveled baseline. Not bit-exact — record the deltas and confirm no collapse.
- **Footprint adjacency unit**: build corrections with known constituents; confirm footprint = union and the
  adjacency test matches base-neighbor-graph reachability.
- **Wave fixpoint unit**: a stage settles deterministically to the same hierarchy regardless of mint order
  within a round.
- **Coordinate-less corrections unit**: corrections have no coordinate; voting/consensus/output unchanged
  (still base-only) on a fixed run; the `aggregate_votes` panic never fires.
- **Apex/handoff**: the spatial apex feeds temporal with the structure unchanged — characterized regression of
  demo-4 against the recorded baseline (verification approach in
  [wavefront-implementation.md](./wavefront-implementation.md) §1).

---

## Open questions

1. **Footprint growth at apex.** Top-level footprints can cover most of the input — fine for adjacency, watch
   memory if base is huge (vision). Bitsets scale, but revisit.
