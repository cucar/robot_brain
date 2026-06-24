# Neuron Reuse — Phase A: Wave-Front Processing + Footprints

**The foundation phase.** Theory in [neuron-reuse.md §2](./neuron-reuse.md). Everything else in the reuse
project sits on top of this. It keeps the existing processing **structure** — spatial processing → apex
handoff → temporal processing — but turns each stage into a **settling wave**, removes stored levels, makes
**all** corrections coordinate-less, and introduces **footprints** as the universal neighborhood primitive.
It subsumes the old "level as activation" idea — the wave removes levels outright rather than relocating them.

This phase is **not bit-exact** — it restructures processing. Its gate is a *characterized* regression
(MNIST + stocks still learn comparably), not byte-identity.

---

## Structure is unchanged: two waves + a handoff

`process_spatial` and `process_temporal` **stay separate functions**. The structure is the same as today:

1. **`process_spatial`** (d=0) — relationships *within* the current frame.
2. **Apex handoff** — the spatial apex feeds temporal, exactly as today.
3. **`process_temporal`** (d>0) — relationships between the current frame and past frames.

**Both already run the same settling level-sweep** — `process_spatial_levels`
([brain.rs:1126](../brain/brain-core/src/brain.rs)) and `process_temporal_levels`
([brain.rs:1222](../brain/brain-core/src/brain.rs)) are the identical loop: walk levels bottom-up, each level's
matched patterns activate one level up, stop when a level produces no activations that push the hierarchy
higher. The active set is fixed at the start of the sweep; freshly-minted error patterns are excluded from
their own level and fire *next* frame. So the "wave" is **not new** — it is this existing sweep, just with the
level no longer stored as an intrinsic neuron field.

They stay two functions for the reasons they are two today, both narrow: temporal **persists per-neuron state
across frames** (`write_back_level_neurons`, [brain.rs:1252](../brain/brain-core/src/brain.rs)) and reads d>0
connections; spatial is **ephemeral** ([brain.rs:1149](../brain/brain-core/src/brain.rs)) and reads d=0. The
sweep loop itself is the same.

There is **no** `process_ages` collapse and **no** age-to-age chaining — the apex fans out to every temporal
distance in parallel; d=1 does not feed d=2 (see [neuron-reuse.md §2.3](./neuron-reuse.md)).

---

## No stored levels

The level-sweep already settles bottom-up until no higher level fires (above). What changes here is that
**the level is no longer a stored intrinsic neuron field** — it is only the sweep's loop variable. The loop
already drives off `max_active_level` read from memory's activation index, not the stored
`neuron_spatial_levels` / `neuron_temporal_levels` maps; those maps are read only by mint (child = parent + 1),
diagnostics, and serialization, all of which move to activation-derived or recomputed values. A neuron's depth
becomes purely "how far the sweep had climbed when it fired," never recorded on the neuron.

The sweep, unchanged in shape (only neighborhood and coordinate handling differ — see below):

```
process_spatial (d=0):
  activate base sensory neurons          // each: footprint = {self}, coordinate + channel given
  repeat until fixpoint:
     fire d=0 routing matches            // activate existing corrections whose constituents are ready
     for each active neuron N:
        observed  = co-active neurons whose footprint is adjacent to N's    // neighborhood
        predicted = N's d=0 connection targets
        if mismatch > threshold:
           C = correction over the footprint-adjacent erroring cluster      // coordinate-less
           C.footprint = ⋃ cluster footprints
           wire cluster → C
     stop when a pass fires nothing new and mints nothing

apex handoff: apex (non-subsumed spatial fired set) → temporal

process_temporal (d>0):
  same wave, reading d>0 connections over the materialized window; votes for future frames
```

---

## Footprints — neighborhood without coordinates

The mechanism that lets corrections be coordinate-less while keeping locality at every level.

- A neuron's **footprint** = the set of **base sensory neurons** it ultimately covers.
  - Base sensory neuron: `footprint = {itself}` (and it keeps its coordinate + channel-neighbors).
  - Correction (any distance): `footprint = union of its constituents' footprints`, computed at mint.
- **Neighborhood** at any level: A and B are neighbors iff their footprints **touch in the base neighbor
  graph** — `∃ base a ∈ footprint(A), base b ∈ footprint(B)` with `a`, `b` spatial-neighbors (or equal).
- This replaces **both** coordinate-inheritance (spatial RF growth) **and** channel-neighbor filtering
  (`is_spatial_neighbor_channel` / `is_temporal_neighbor_channel` — [thalamus.rs:617](../brain/brain-core/src/thalamus.rs)).
  Receptive fields still grow one ring per layer — the union footprint widens naturally.

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
pattern neuron from being a vote target** ([brain.rs:1626-1638](../brain/brain-core/src/brain.rs)):

> "Every vote MUST target a neuron with a coordinate — sensory events and actions have coordinates by
> construction. A vote toward a target without a coordinate means we learned a connection to a pattern
> neuron upstream, which is a real architectural break."

So a correction is **never** a candidate → never a dimension winner → never dequantized
([brain.rs:1711-1746, 1832-1857](../brain/brain-core/src/brain.rs)). Output already resolves exclusively to
base sensory/action neurons. A correction's coordinate today is used **only** for neighbor-filtering — which
footprints replace. Removing it touches nothing on the prediction/action layer; the existing
"no-coordinate ⇒ must not be a vote target" panic stays valid and becomes the load-bearing invariant.

---

## Code touched (large — this is a rearchitecture)

- **`brain/brain-core/src/thalamus.rs`** — convert `process_spatial` and `process_temporal` to settling waves
  (propagate to fixpoint instead of a counted level-sweep). Delete **both** `neuron_spatial_levels` and
  `neuron_temporal_levels` ([thalamus.rs:219](../brain/brain-core/src/thalamus.rs)) and all readers (mint
  child-level, sweep bounds, diagnostics, snapshot). Delete channel-neighbor machinery
  (`temporal_channel_neighbors`, `is_temporal_neighbor_channel`, `set_temporal_neighbors`, spatial
  equivalents) in favor of footprint adjacency.
- **`brain/brain-core/src/neuron.rs`** — drop coordinate/dimension inheritance for corrections
  ([neuron.rs:468](../brain/brain-core/src/neuron.rs)); add a `footprint` (bitset) to each neuron; correction
  mint computes `footprint = ⋃ constituents`. Connection/error/learn steps filter by footprint adjacency
  instead of channel-neighbor.
- **`brain/brain-core/src/memory.rs`** — `neuron_states` must hold a neuron at multiple depths in one stage
  (a settling wave can reach a neuron at more than one depth); the single `LevelAgeState` per `(neuron, frame)`
  ([memory.rs:45](../brain/brain-core/src/memory.rs)) becomes multi-valued. Level indices become wave
  activation state.
- **`brain/brain-core/src/brain.rs`** — drive the two waves + apex handoff (structure unchanged); voting
  unchanged (already base-only).
- **`brain/brain-core/src/backup.rs`** — drop both level columns from `neurons.csv`; rebuild footprints from
  the connection graph on load (don't serialize them). Format-version bump.

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
- **Apex/handoff**: the spatial apex feeds temporal exactly as today (demo-4 behavior preserved).

---

## Open questions

> The settling sweep itself is **not** in question — it already exists and terminates (the loop exits when a
> level produces no activations that push `max_active_level` higher, [brain.rs:1174, 1278](../brain/brain-core/src/brain.rs)).
> Spatial and temporal run the identical loop today; this phase does not change it.

1. **Deterministic mint order under footprints.** Footprint-clustering and mint order must be deterministic
   for comparable runs (sorted keys, not hash order). New because footprints replace the coordinate-keyed
   grouping.
2. **Footprint growth at apex.** Top-level footprints can cover most of the input — fine for adjacency, watch
   memory if base is huge (vision). Bitsets scale, but revisit.
3. **Subsumption.** Definition unchanged (a neuron is subsumed if a covering correction fired this frame).
   No new definition needed.
4. **Multi-depth `neuron_states` shape** — `(neuron, frame) → {depth → state}` vs `(neuron, frame, depth) →
   state`. Needed only once reuse can activate a neuron from multiple parents at different depths (Phase C/D);
   groundwork here. Pick on eviction/code-shape grounds.
