# Neuron Reuse — Wave-Front Implementation & Migration Plan

How to migrate the brain from the current spatial architecture to the wave-front incrementally, verifying each
step against numbers from an MNIST verification run, and deferring the merge threshold to the very end. Theory:
[neuron-reuse.md](./neuron-reuse.md). Phase specs: [A](./neuron-reuse-wavefront.md),
[B](./neuron-reuse-index.md), [C](./neuron-reuse-frame.md), [D](./neuron-reuse-final.md). This doc is the
*ordering and verification* layer over those.

---

## 1. Why this is verifiable at all

The wave-front is **not bit-exact** with the current architecture, so at some point the MNIST numbers must
diverge. The trick that keeps the migration checkable up to and through that divergence is the **simulation as
an oracle**:

- **MNIST is single-frame, d=0 spatial** — one static fixation per image, no temporal pathway exercised. That
  is *exactly* the regime [`wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) models, so the sim can be a
  **complete** reference for MNIST, not a partial one.
- The brain is therefore verified against **two references, by step type**:
  - **Behavior-preserving refactor** → brain numbers stay ≈ the pre-change baseline (structural identity).
  - **Behavior-changing step** → brain numbers move *toward the sim's* numbers; after the full exact-match
    wave-front, brain ≈ sim.

The sim *predicts* what each wave-front change should do to the numbers. You are never verifying the new brain
against the *old* brain (different algorithms — they will not, and should not, match); you verify against the
baseline for refactors and against the sim for the algorithm change.

### Decisions baked into this plan

- **Sim role: full MNIST oracle.** The sim consumes real binary digit images and a supervised digit readout
  (NB-product decode, mirroring [test.js](../apps/mnist/jobs/test.js)) and emits an accuracy number plus the
  diagnostic set in §2.
- **Brain migration: in-place edits, no runtime flag.** The old path is not maintained behind a flag; it lives
  in **git history**. The pre-migration commit is tagged and is the re-baseline reference — re-run the old
  numbers by checking it out, not by a flag.
- **Match strictness: structural + trend.** Verification is on per-level pattern counts, depth, neuron count,
  and accuracy magnitude/direction — **not** bit-identical accuracy. JS↔Rust exact parity is not required.
- **Merge threshold deferred.** The entire migration runs at exact match (θ = 1.0) until Stage D; the threshold
  is introduced last and swept slowly.

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

## 3. Stage A — Make the sim a brain-shaped MNIST oracle (no brain changes)

Goal: a deterministic JS reference that runs real MNIST and emits §2's numbers, structured so the eventual
Rust read-across is mechanical.

1. **A1 — Restructure toward the brain's shape.** Units ≈ neurons carrying footprints; the level loop ≈
   `process_spatial_levels` (the settling sweep); the grouping + mint ≈ the Phase-C path. Naming and data flow
   mirror the Rust so the port is a transliteration. Keep iteration **deterministic** (sorted keys).
2. **A2 — Real MNIST input, scalably.** Feed binary digit images (reuse the encoder's bits / loader) instead
   of ASCII shapes. Add an **inverted `pixel → units` index** so adjacency is a local lookup, not an O(N²·K²)
   global scan — the JS analog of the Rust bitset `dilate + AND` ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)).
   Parameterize radius/connectivity to match the brain.
3. **A3 — Supervised readout.** Wire active patterns → digit action neurons, NB-product decode (mirror
   [test.js](../apps/mnist/jobs/test.js)), so the sim emits accuracy + the §2 diagnostics.
4. **A4 — Exact match only.** No merge threshold, no reuse lookup — single frame, d=0, every unit erroring.
   This is steps 2–4 of the §3 pipeline (group → mint → wire), the residual path on its own.

**Gate:** sim runs the full MNIST set in acceptable time; numbers are deterministic and stable across runs;
sanity holds (depth grows, L1 footprints stay local, accuracy in a sane band). Record these as the **migration
target**.

---

## 4. Stage B — Baseline and freeze the spec

1. **B1 — Baseline the current brain.** Record the current brain's §2 numbers at the iteration config and the
   representative config. **Tag the commit** — this is the re-baseline reference for the in-place migration.
2. **B2 — Freeze the sim as the spec.** From here, any brain↔sim disagreement is a **brain bug** — or a
   sim-spec gap, which gets fixed *in the sim first*, then mirrored into the brain. This ordering is what keeps
   debugging tractable without a runtime flag.
3. **B3 — Expected-deltas note.** Write down, per migration step, the expected *direction* of the number move
   (e.g. footprint neighbors → fewer/larger L1 groups than channel-neighbor filtering; batched mint → lower L1
   count). Knowing the expected sign in advance turns each step into a check, not a guess.

---

## 5. Stage C — Migrate to the exact-match wave-front (no threshold)

In-place, one commit per step. Behavior-preserving refactors first (numbers stay ≈ baseline), then the
diverging changes (numbers move toward the sim). The exact-match wave-front is the milestone where **brain ≈
sim**.

| Step | Change | Type | Verify |
|---|---|---|---|
| **C1** | Add a `footprint` (bitset) to every neuron, **computed but unused**; adjacency still via channel-neighbors. | preserving | numbers ≈ baseline (footprints inert) |
| **C2** | Settling-wave / multi-depth `neuron_states` restructure with **no change to what fires**. | preserving | ≈ baseline (or characterize a forced minor change) |
| **C3** | Swap neighbor filtering **channel-neighbor → footprint-adjacency** (touch in the base graph). | diverging | numbers move in the B3-predicted direction |
| **C4** | **Coordinate-less corrections** — drop coordinate inheritance. | diverging | near-neutral (coordinates were only neighbor-filtering, now footprints; corrections are never dequantized) |
| **C5** | **Batched mint by observed-set** (replace per-erroring-neuron mint) **+ multi-parent machinery** (refcounted reaping, multi-parent serialization, shared-neuron activation). | diverging | **brain ≈ sim** — full exact-match wave-front matches the oracle |

Notes:

- **C5 is the heavy step** (it is Phase C's batched mint + multi-parent lifecycle). If a divergence at C5 is
  hard to localize, decompose: batched-mint-grouping first (verify the L1 count drop), then the multi-parent
  lifecycle (verify serialization round-trip + refcount reaping units), then shared-neuron activation.
- **Re-baselining is `git checkout`**, not a flag — to compare against the old path, check out the B1 tag and
  re-run. Keep each C-step a clean, revertible commit.
- The **isolated-unit rule** and **footprint = co-failers ∪ observed** ([neuron-reuse.md §3.3](./neuron-reuse.md))
  must port exactly — they are what keep disconnected digits apart and footprints connected. The sim is the
  reference for both.

After C5 the brain is the **exact-match wave-front**: coordinate-less, footprint-local, batched mint,
multi-parent — and reproduces the sim's MNIST numbers (structural + trend). No reuse lookup, no threshold yet.

---

## 6. Stage D — Introduce the reverse index, lookup, and merge threshold (slowly)

| Step | Change | Verify |
|---|---|---|
| **D1** | Build the **reverse inference index** ([Phase B](./neuron-reuse-index.md)), unit-tested, **unconsumed**. | ≈ C5 (no behavior change) |
| **D2** | Add the **lookup at θ = 1.0** (exact match = no partial reuse). | ≈ C5 — proves the lookup plumbing changes nothing at θ = 1.0 |
| **D3** | **Lower θ in steps** — in the **sim first** (oracle), then the brain. Sweep accuracy vs neuron count. | brain tracks the sim's θ-sweep; compare against the 96.44% bar |
| **D4** | **Temporal (d>0) reuse.** | irrelevant to MNIST; validate on stocks ([validation](./neuron-reuse-validation.md)) |

D3 is where the "magic" is tested: the threshold is the only lever that turns exact-match memorization into
generalizing reuse ([neuron-reuse.md §3.5](./neuron-reuse.md)). Sweep it gently, watching whether reuse-groups
climb on real digits *without* collapsing discrimination — the open empirical question.

---

## 7. Relationship to the A/B/C/D phase docs

This plan **re-orders** the existing phases for verifiability; it does not change their content:

- **C1–C5** ≈ Phase A (wave-front foundation), pulling **Phase C** (batched mint + multi-parent) forward into
  C5, because batched mint needs no index.
- **D1** ≈ Phase B (the index, built unconsumed — exactly as that doc already frames it).
- **D2–D3** ≈ Phase D (lookup), with the threshold introduced last.
- **D4** ≈ temporal reuse + the stocks validation.

The only reordering is **index (Phase B) after batched mint (Phase C)** instead of before — harmless, since the
index is unconsumed until the lookup in D2 regardless.

---

## 8. Rust-port notes

Folding in the representation choices (some surfaced by external review of the sim):

- **Footprints are bitsets** over base sensory neurons; adjacency = dilate one footprint by the precomputed
  base neighbor-ring and AND with the other; nonzero ⇒ touch ([neuron-reuse-wavefront.md](./neuron-reuse-wavefront.md)).
  The sim's inverted `pixel → units` index is the JS stand-in.
- **Grouping key = a sorted `Vec<u32>` of unit indices**, not a delimited string. The sim's `join('|')` key is a
  JS convenience only — do not port it (delimiter-collision risk once ids are reformatted).
- **Deterministic ordering** (sorted keys) everywhere, for reproducible diagnostics and to match the sim.
- **Neighbor adjacency is footprint *touch* at every level** — base-graph overlap-or-adjacency, *not* parent
  membership. Graded locality (footprints span more as you climb) is the intended behavior, not a defect: it is
  what lets disjoint-but-abutting features become neighbors ([neuron-reuse.md §2.2, §3.3](./neuron-reuse.md)).

---

## 9. Risks

- **No live dual-run** (in-place choice): a divergence can only be compared against the *recorded* baseline and
  the sim, not a side-by-side run. Mitigation: tag every step, keep commits revertible, lean on the fixed small
  subset for localization.
- **Structural + trend matching can hide a real bug** inside the tolerance band. Mitigation: the small-subset
  per-image spot-checks and the sim oracle catch what the aggregate trend misses.
- **Big steps (C2, C5)** are the likeliest place to lose the thread; decompose on first sign of an
  unexplained divergence.
- **Sim must stay the spec** — when reality disagrees, fix the sim first, then the brain. A drifting sim is a
  useless oracle.
