# Wave-Front Implementation & Migration Plan

> **Spatial oracle is built.** The reference simulation specified in
> [neuron-reuse-simulation.md](./neuron-reuse-simulation.md) is **built and validated** —
> [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) implements the corrected model
> (recognize → predict L0 → cluster → reuse/mint, with refinement as the split force) and the merge/split
> equilibrium has been characterized. It is **spatial-only**, so it is the verification oracle for the
> **spatial** wave of this migration (the temporal wave is held to characterized regression against a recorded
> baseline — §1). The substrate migration is staged in §3; the verification approach (§1) and tracked numbers
> (§2) are its anchors.

The migration plan for the [wave-front foundation](./wavefront.md): move the brain from the current spatial
architecture to the wave-front incrementally, verifying each step against numbers from an MNIST verification
run. Theory: [neuron-reuse.md](./neuron-reuse.md) §2. The reuse phases that build on this foundation are
specs [A](./neuron-reuse-index.md), [B](./neuron-reuse-frame.md), [C](./neuron-reuse-final.md). This doc is
the *ordering and verification* layer over the wave-front rearchitecture ([wavefront.md](./wavefront.md)).

---

## 1. Verification approach (what survives the model correction)

The **concrete step sequence is the substrate migration in §3** (footprints → neighborhood switch →
coordinate-less → level-less), which follows from the wave-front substrate design
([wavefront.md](./wavefront.md)), **not** from the reuse mechanism. The rebuilt simulation
([neuron-reuse-simulation.md](./neuron-reuse-simulation.md)) is the **spatial verification oracle** for that
migration, not its step list. The committed **verification approach** is:

- **Sim as oracle — for the spatial wave.** The brain is verified against the **simulation**, never the old
  brain (different algorithms — they will not, and should not, match). Behavior-preserving refactors keep the
  brain's numbers ≈ the recorded baseline; an algorithm change moves them *toward* the sim's numbers. The sim is
  frozen as the spec; a brain↔sim disagreement is a brain bug — or a sim-spec gap, fixed in the sim first.
- **The sim is spatial-only (d=0).** It is the oracle for `process_spatial` and nothing else
  ([neuron-reuse-simulation.md §11](./neuron-reuse-simulation.md)). The **temporal wave** (`process_temporal`,
  d>0, the apex handoff) has **no sim oracle**, so its gate is **characterized regression against the recorded
  baseline** — structurally unchanged, numbers expected ≈ baseline. This is the one place we compare against the
  pre-migration brain, and it is legitimate precisely because the temporal structure is *not* changing
  algorithmically (only its neighborhood primitive and bookkeeping are). Verify each substrate change on the
  spatial side first (where the oracle exists), then transliterate the identical change to the temporal side and
  confirm it against baseline.
- **In-place edits, no runtime flag.** The old path lives in git history; the pre-migration baseline is commit
  **`b4dab30`** — re-run the old path by checking it out.
- **Structural + trend matching** on the §2 diagnostics — not bit-identical accuracy (JS↔Rust parity is not
  required).
- **Config parity is mandatory** — sim and brain must share image size, binarization, and radius/connectivity,
  or the oracle is meaningless.

What the correction **changes**:

- **The merge threshold is NOT deferred.** The earlier plan introduced it last; that is wrong — the threshold
  is intrinsic to **recognition** (an approximate context match), so without it the hierarchy never forms
  ([neuron-reuse.md §3.8](./neuron-reuse.md)). There is no "exact-match wave-front first" stage.
- **MNIST is one frame per image, but the hierarchy builds over many frames** via recognition, so the oracle is
  the multi-frame corrected sim — not a single-frame pass.

---

## 2. The numbers tracked at every step

Both the brain ([test.js](../apps/mnist/jobs/test.js) diagnostics) and the sim emit the same set, so they are
directly comparable:

- **accuracy** — prequential train + frozen eval, on the full set and on a fixed small subset (~50 images) for
  fast spot-checks.
- **depth** — max spatial level reached.
- **per-level pattern counts** — L1, L2, … live correction neurons per level.
- **active corrections / cumulative minted** — total live above base, and monotonic mint count.
- **neuron count** — total (base + actions + corrections).

A step is "verified" when these move as predicted: flat for a refactor, toward the sim for an algorithm change.
The small fixed subset is where divergences are localized (per-image apex/level inspection); the full set is
for the headline trend.

> **Config parity is mandatory.** The sim and the brain must use the **same** MNIST config — image size,
> binarization, and neighbor radius/connectivity — or the oracle is meaningless. Stage 0 confirms the sim's base
> neighbor graph matches the brain's `radius` (and the 4-vs-8 connectivity choice). Iterate on a small/fast
> config (e.g. 14×14); do the headline comparison on the representative stack (28×28 binary, radius 2 — the
> [96.44% configuration](./neuron-reuse.md)).

---

## 3. Migration stages

> **Scope reminder.** This is the migration for the **wave-front substrate only** — single-parent throughout
> ([wavefront.md](./wavefront.md), "Scope: single-parent is the dividing wall"). The
> recognition → predict-L0 → cluster → reuse/expand *mechanism* is the **reuse** project and is **not** staged
> here; an earlier draft of this section sequenced it by mistake. The wave-front delivers four substrate changes
> and nothing that requires a correction to have more than one parent.

The original detailed stages encoded the old, wrong model and are removed (the original **Stage A** "build the
sim" is superseded by [neuron-reuse-simulation.md](./neuron-reuse-simulation.md); the original **Stages C–D**
"exact-match wave-front then deferred threshold" are invalid — the threshold is intrinsic to recognition, which
is a reuse concern, not a wave-front one).

The migration is **five stages**, each behind its own gate. Every stage is an in-place edit (§1), verified on the
**spatial** side first (against the sim oracle + baseline) then transliterated to the **temporal** side (against
baseline only — the sim is spatial-only). The dependency order is deliberate: introduce the replacement
primitive, make it load-bearing, *then* delete the two stored fields it makes redundant.

### Stage 0 — Baseline (the re-baseline reference)

Run the representative MNIST and stocks configs and record the full §2 number set (accuracy, depth, per-level
counts, active/cumulative corrections, neuron count) into a checked-in results file. The **pre-migration baseline
is commit `b4dab30`** — every later stage diffs its numbers against this recorded set, and the old path is
re-runnable by checking that commit out (§5 risk: no live dual-run).

*Gate:* numbers recorded and committed; the sim reproduces its [§11](./neuron-reuse-simulation.md) results on the
same config (config parity confirmed — image size, binarization, radius/connectivity).

### Stage 1 — Footprints, additive (behavior-neutral)

Add a `footprint` bitset to every neuron: base = `{self}`; correction = `⋃ constituents` (the context set) at
mint. Precompute the base neighbor-ring from the encoder's declared neighborhoods (the `radius` window). Add the
footprint-adjacency test (dilate-and-AND). **Do not wire it into filtering yet** — channel-neighbor still drives
all behavior. Pure addition.

*Gate:* §2 numbers **flat** vs Stage 0 (nothing consumes footprints yet); footprint-adjacency unit passes
(footprint = union; adjacency = base-graph reachability); at L0, footprint adjacency **equals** the
channel-neighbor window on a fixed run (the equivalence Stage 2 relies on).

### Stage 2 — Switch neighborhood to footprint adjacency

Replace every `is_spatial_neighbor_channel` / `is_temporal_neighbor_channel` call site with footprint adjacency;
delete the channel-neighbor machinery (`*_channel_neighbors`, `is_*_neighbor_channel`, `set_*_neighbors`). This is
the **first real behavioral change**.

*Gate (characterized regression, not flat):* L0-dominated behavior ≈ baseline (footprint ≈ channel-neighbor at
base); L1+ **moves toward the sim** as receptive fields grow by union rather than single-pixel anchor
([wavefront.md](./wavefront.md), "What this actually changes"). Record the deltas; confirm no collapse and that
the depth/per-level trend tracks the sim's direction on the small fixed subset.

### Stage 3 — Coordinate-less corrections

Stop inheriting the parent coordinate at mint; corrections carry only `id` + `footprint`. Drop the pattern's
`base_neurons` insert and the restore parent-walk that rederived correction coordinates. Confirm consensus never
reads a correction coordinate (corrections are forbidden vote targets, so they never reach per-dimension
grouping). Backup: drop the correction coordinate; format-version bump.

*Gate:* coordinate-less-corrections unit (corrections have no coordinate; the `aggregate_votes`
"no-coordinate ⇒ never a vote target" panic never fires); numbers ≈ Stage 2 (the coordinate's only behavioral
use, neighbor filtering, was already retired in Stage 2 — this should be near-neutral). Save/restore round-trips
to identical numbers.

### Stage 4 — Remove stored levels (the wave is now pure)

Delete `neuron_spatial_levels` / `neuron_temporal_levels`. Repoint the four readers: mint reads the parent's
depth from the sweep loop variable; diagnostics recompute from the activation index; serialization drops the
level columns; `skip_action_neuron`'s base test becomes an explicit base-neuron predicate. The sweep is now a
settling wave driven solely by the loop variable. Backup: drop both level columns; rebuild footprints on load in
dependency order (memoized recursion from base / topo-sort the constituent graph); format-version bump.

*Gate:* wave-fixpoint unit (a stage settles deterministically to the same hierarchy regardless of mint order
within a round); numbers ≈ Stage 3; save/restore round-trips identically with footprints rebuilt, not
serialized; apex/handoff demo-4 ≈ baseline.

### Standing rules (carried from §1)

- **Sim stays the spec** for spatial — a brain↔sim disagreement is a brain bug, fixed in the sim first.
- **Keep every stage a separate, revertible commit**; localize divergences on the small fixed subset before
  touching the headline config.

---

## 4. Rust-port notes

Representation choices that survive the model correction:

- **Footprints are bitsets** over base sensory neurons; adjacency = dilate one footprint by the precomputed
  base neighbor-ring and AND with the other; nonzero ⇒ touch ([wavefront.md](./wavefront.md)).
- **Sets as sorted `Vec<u32>`** of indices, not delimited strings (footprint constituents, context sets).
- **Deterministic ordering** (sorted keys) everywhere, for reproducible diagnostics and to match the sim.
- *(Reuse-scope, not wave-front)* cluster membership via union-find over the neighbor graph (the transitive
  merge) belongs to the reuse project — listed here only so the port keeps the footprint representation
  union-find-friendly.
- **Neighbor adjacency is footprint *touch* at every level** — base-graph overlap-or-adjacency, *not* parent
  membership. Graded locality (footprints span more as you climb) is the intended behavior: it is what lets
  disjoint-but-abutting features become neighbors ([neuron-reuse.md §2, §3.4](./neuron-reuse.md)).

---

## 5. Risks

- **No live dual-run** (in-place choice): a divergence can only be compared against the *recorded* baseline and
  the sim, not a side-by-side run. Mitigation: keep every step a separate revertible commit, lean on a fixed
  small subset for localization.
- **Structural + trend matching can hide a real bug** inside the tolerance band. Mitigation: small-subset
  per-image spot-checks and the sim oracle catch what the aggregate trend misses.
- **Big algorithm steps** are the likeliest place to lose the thread; decompose on first sign of an
  unexplained divergence.
- **Sim must stay the spec** — when reality disagrees, fix the sim first, then the brain. A drifting sim is a
  useless oracle.
