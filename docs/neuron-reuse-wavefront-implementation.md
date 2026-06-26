# Neuron Reuse — Wave-Front Implementation & Migration Plan

> **⚠ Model corrected — this plan needs revision.** The reuse mechanism is now **recognize → predict L0 → on
> misprediction, transitively-merge-cluster the requests → reuse/expand or mint**, balanced by refinement +
> forgetting ([neuron-reuse.md §3](./neuron-reuse.md)). The "sim-as-oracle" premise still holds, but the oracle
> is the **rebuilt** sim specified in [neuron-reuse-simulation.md](./neuron-reuse-simulation.md), **not** the
> obsolete `wavefront-sim.js`. The old detailed stages are **removed** (§3); the verification approach (§1) and
> tracked numbers (§2) survive, and the brain migration cannot start until the rebuilt sim validates the
> merge/split equilibrium.

How to migrate the brain from the current spatial architecture to the wave-front incrementally, verifying each
step against numbers from an MNIST verification run, and deferring the merge threshold to the very end. Theory:
[neuron-reuse.md](./neuron-reuse.md). Phase specs: [A](./neuron-reuse-wavefront.md),
[B](./neuron-reuse-index.md), [C](./neuron-reuse-frame.md), [D](./neuron-reuse-final.md). This doc is the
*ordering and verification* layer over those.

---

## 1. Verification approach (what survives the model correction)

The corrected model — recognition-based, multi-frame, merge/split ([neuron-reuse.md §3](./neuron-reuse.md)) —
changes the migration substantially, and the **concrete step sequence is deferred** until the rebuilt
simulation ([neuron-reuse-simulation.md](./neuron-reuse-simulation.md)) is built and the merge/split
equilibrium is understood (§3). What survives is the **verification approach**:

- **Sim as oracle.** The brain is verified against the **simulation**, never the old brain (different
  algorithms — they will not, and should not, match). Behavior-preserving refactors keep the brain's numbers ≈
  a tagged baseline; an algorithm change moves them *toward* the sim's numbers. The sim is frozen as the spec;
  a brain↔sim disagreement is a brain bug — or a sim-spec gap, fixed in the sim first.
- **In-place edits, no runtime flag.** The old path lives in git history; tag the pre-migration commit and
  re-run by checkout.
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
> binarization, and neighbor radius/connectivity — or the oracle is meaningless. Stage A parameterizes the
> sim's base neighbor graph to match the brain's `radius` (and the 4-vs-8 connectivity choice). Iterate on a
> small/fast config (e.g. 14×14); do the headline comparison on the representative stack (28×28 binary,
> radius 2 — the [96.44% configuration](./neuron-reuse.md)).

---

## 3. Migration stages — deferred pending the simulation

The original detailed stages encoded the **old, wrong** model and are removed:

- the original **Stage A** (build the sim) is superseded by [neuron-reuse-simulation.md](./neuron-reuse-simulation.md);
- the original **Stages C–D** (an "exact-match wave-front by observed-set," then a deferred threshold) are
  invalid — the threshold is intrinsic to recognition, so there is no exact-match-first wave-front.

The two settled, model-agnostic anchors survive: **baseline the current brain and tag the commit** (the
re-baseline reference), and **freeze the rebuilt sim as the spec** (a brain↔sim disagreement is a brain bug,
fixed in the sim first).

The corrected migration sequence — recognition → predict-L0 → cluster → reuse/expand, with refinement and
forgetting — will be defined **once the rebuilt sim validates the model and the merge/split equilibrium**
([neuron-reuse.md §3.6](./neuron-reuse.md)) is understood. Until then, only the verification approach (§1) and
the tracked numbers (§2) are committed.

---

## 4. Rust-port notes

Representation choices that survive the model correction:

- **Footprints are bitsets** over base sensory neurons; adjacency = dilate one footprint by the precomputed
  base neighbor-ring and AND with the other; nonzero ⇒ touch ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)).
- **Cluster membership via union-find** over the neighbor graph (the transitive merge); represent any set as a
  sorted `Vec<u32>` of indices, not a delimited string.
- **Deterministic ordering** (sorted keys) everywhere, for reproducible diagnostics and to match the sim.
- **Neighbor adjacency is footprint *touch* at every level** — base-graph overlap-or-adjacency, *not* parent
  membership. Graded locality (footprints span more as you climb) is the intended behavior: it is what lets
  disjoint-but-abutting features become neighbors ([neuron-reuse.md §2.2, §3.4](./neuron-reuse.md)).

---

## 5. Risks

- **No live dual-run** (in-place choice): a divergence can only be compared against the *recorded* baseline and
  the sim, not a side-by-side run. Mitigation: tag every step, keep commits revertible, lean on a fixed small
  subset for localization.
- **Structural + trend matching can hide a real bug** inside the tolerance band. Mitigation: small-subset
  per-image spot-checks and the sim oracle catch what the aggregate trend misses.
- **Big algorithm steps** are the likeliest place to lose the thread; decompose on first sign of an
  unexplained divergence.
- **Sim must stay the spec** — when reality disagrees, fix the sim first, then the brain. A drifting sim is a
  useless oracle.
