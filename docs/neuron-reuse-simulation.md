# Neuron Reuse — Simulation Spec

The specification for the **corrected** reference simulation, and the results of running it.
[`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) is **built to this spec** (rewritten
in place — the obsolete group-by-observed-set version is gone) and **validated**: see §11 for results. It is
deterministic, dependency-free, and brain-shaped so the mechanism ports as a transliteration.

The simulation has three jobs, all now discharged:

1. **Validate** the corrected mechanism end to end ([neuron-reuse.md §3](./neuron-reuse.md)). ✓ — the
   recognize → predict-L0 → cluster → reuse/mint loop runs and climbs (§11).
2. **Answer the empirical question** the design cannot settle on paper: do patterns settle into *regional*
   sizes, or collapse into **one big blurry pattern**? (the merge/split equilibrium — §3.6 there). ✓ — a
   regional regime exists under refinement; merge-only runs away (§11).
3. **Oracle** for the eventual brain port (structures and flow mirror the brain). ✓ — reviewed faithful and
   portable (§11).

---

## 1. Why the old sim is wrong

The old sim modeled a **cold-start, single-frame** world: every active neuron errors, there is no recognition
of prior patterns, no cross-frame state, the active set is the ON pixels only, and requests are grouped by
*identical observed-set*. Every one of those is wrong against the real model:

- the real layer is **per-position black/white neurons** — the *whole* field is active, not just the stroke,
- patterns are **recognized** (approximate context match ≥ threshold) and **predict L0**; only the
  **mispredictions** become correction requests — not "every neuron errors,"
- requests are clustered by **transitive merge over neighborhoods**, not bucketed by identical observed-set,
- pattern size is set by a **merge vs split** balance across **many frames** — absent entirely from the old sim.

So it cannot be patched; it must be rebuilt around recognition + multi-frame dynamics.

---

## 2. The model to simulate

Exactly [neuron-reuse.md §3](./neuron-reuse.md). In brief, per frame and per level:

```
recognize (context match ≥ threshold)  →  predict L0
   →  where L0 prediction is wrong (≥ error threshold): a correction request
      →  cluster requests by transitive merge over neighborhoods
         →  reuse an existing pattern (index match) or mint a new one; matched patterns expand
            →  install (fires next frame)
```

with **every pattern predicting L0**, **merge** = clustering + reuse + expansion, and **split** = refinement +
forgetting.

---

## 3. Data model (mirror the brain)

- **Base layer.** Retinotopic. Each pixel position has neurons per value — **black and white** for binary.
  Exactly one fires per position per frame, so the **entire field** (`imageSize²` neurons) is active, one per
  position. (This replaces the old "active = ON pixels.")
- **Neuron** (`id`, `level`):
  - `footprint` — base pixels covered (base: its own position; correction: union of what it binds),
  - `context` — its learned expectation of its neighbors (the sources that recognize it): at the base, the
    coordinate neighborhood; higher, the footprint-touch neighbors,
  - `targets` — its **L0 prediction**: which base neurons it expects active (always base, never higher),
  - `parents` — the units that route to it (multi-parent under reuse),
  - `strength` / `last_active_frame` — for forgetting and the death ledger.
- **Reverse index** — `L0-target → patterns that predict it`, the candidate generator for reuse lookup.

Naming and decomposition mirror the brain (`process_frame` → `process_spatial` → `process_spatial_levels`; a
neuron store; verb-first methods; sorted-key determinism), so the port stays a transliteration.

---

## 4. The per-frame algorithm

```
processFrame(image):
  activateBaseNeurons(image)              // one neuron per position (black/white)
  for level in 0.. until settled:
    units = active units at this level
    requests = []
    for N in units:
      pattern = recognize(N.context, threshold)     // best context match ≥ merge threshold, else none
      if pattern: predictL0(pattern)                // pattern fires, predicts L0
      if firstExposure(N): learnDefault(N)          // record context, take default
      if pattern and L0Error(pattern) ≥ errorThreshold(level):
        requests.push(N)                            // mispredict despite context match
    corrections = clusterAndCorrect(requests, level)
    install(corrections)                            // fire next frame
  readout()                                         // supervised digit vote (§7)
```

### 4.1 Recognition

`recognize(context, threshold)` returns the existing pattern whose stored context best matches the unit's
current context **≥ the merge threshold**, else none. The matched pattern fires and emits its **L0**
prediction. First exposure (no match) just learns the context and takes the default digit.

### 4.2 Error

A fired pattern's L0 prediction is compared to the actual L0. If the mismatch ≥ the (adaptive — §5.3) error
threshold, the unit becomes a **correction request**, *even though its context matched* (it was too general).

### 4.3 Cluster and correct (the heart — issue #1 lives here)

```
clusterAndCorrect(requests, level):
  matched = patterns reused this frame at this level         // from the reverse-index lookup (§6)
  pool    = matched ∪ requests
  for each connected cluster in transitiveMerge(pool, neighborRelation(level)):
    if cluster has no neighbor: skip                         // need ≥1 neighbor for context
    if cluster contains a matched pattern P:
      wire the cluster's requests to P as parents
      expand P (grow footprint/context to cover the cluster) // EXPANSION — §5.2
    else:
      mint one correction over the cluster                   // footprint=coverage, targets=correct L0
```

- `neighborRelation(level)`: base = coordinate neighborhood (within radius); higher = footprints touch.
- `transitiveMerge` = connected components: union any two pool members that are neighbors.
- A correction's **targets = the correct L0** the cluster's context should predict; its **context = the
  cluster's neighbors**.

---

## 5. Merge and split

### 5.1 Merge (grow / generalize)
- **in-frame:** transitive clustering of requests (§4.3),
- **inter-frame:** reuse (a request matches an existing pattern via the index, §6),
- **expansion:** a matched pattern adjacent to new requests grows to absorb them (§5.2).

### 5.2 Expansion (issue #1 — proposal, to be tuned in-sim)
When a cluster contains a matched pattern `P` and adjacent new requests, the requests wire to `P` and `P`
expands: `footprint ← footprint ∪ cluster.footprint`, `context ← context ∪ cluster.context`. **Open knobs:**
a cap on growth per frame; an overlap floor below which a neighbor is *not* absorbed (to stop a pattern
swallowing the whole field in one step). Start with no cap, measure, then add a gate if it runs away.

### 5.3 Split (shrink / specialize)
- **refinement** ([refinement.md](./refinement.md)) — on a matched pattern, consolidate **context and targets**
  toward the common core of the instances it matched (strengthen common, weaken/drop incidental),
- **forgetting** — decay strength; reap via the death ledger when no parent keeps a pattern alive,
- **adaptive error threshold** — the lever the design hopes carves big general patterns into regional ones:
  as a pattern generalizes, its error tolerance adapts so it starts requesting corrections where it has become
  too vague, spawning more-specific children. Sweep this.

---

## 6. Reuse lookup (the merge force across frames)

Before clustering, each request queries the **reverse index**: existing patterns predicting its L0 region.
Score by context/target match; if best ≥ merge threshold → the request **reuses** that pattern (it joins the
`matched` set fed into §4.3, where it may also *expand*). Candidacy is membership-only (strength-blind).

---

## 7. Levels and the supervised readout

- **L1** forms from base mispredictors (clustered by coordinate neighborhood).
- **L2** forms from **co-firing, mispredicting L1 patterns** (clustered by footprint adjacency): a general L1
  fires (context matched) but mispredicts L0, and mints an L2 conditioned on its active parent's *other* L1
  neighbors that predicts the correct L0. Needs ≥ 1 L1 neighbor.
- **Readout (oracle).** Active patterns vote for the digit (supervised wire, NB-product decode, mirroring
  `test.js`). Train accumulates per-pattern digit counts; eval decodes. Produces train/test accuracy.

---

## 8. Diagnostics — the experiment that answers issue #2

The headline question is **one big blurry pattern vs many regional patterns**, so the sim must *measure
pattern size and the merge/split balance* over a stream of images, not just final accuracy:

- per level: **pattern count**, **footprint-size distribution** (the one-big-vs-regional signal), depth,
- **merge rate** (clusters merged, patterns expanded) vs **split rate** (refinements, reaps) per frame,
- **reuse rate** (requests that reused vs minted), **recognition rate** (units that fired a pattern),
- **accuracy** (train/eval) as the downstream check.

**Sweeps:** merge threshold, error threshold (incl. adaptive on/off), forget rate, refinement on/off,
radius/connectivity. The deliverable is the answer to: *under what rates do L1 footprints settle into a stable
regional distribution rather than collapsing to one big pattern (merge runaway) or staying singletons (no
reuse)?*

---

## 9. Build order

1. Base layer (black/white per position) + recognition + predict-L0 + error → requests (single level, no reuse).
2. Transitive-merge clustering + mint (Phase-C analog), with the ≥1-neighbor rule.
3. Reverse index + reuse lookup + expansion (Phase-D analog; issue #1).
4. Refinement + forgetting + adaptive error (the split force).
5. Multi-level (L2+) and the supervised readout.
6. The diagnostics + sweeps (§8) — the experiment.

Keep it deterministic (sorted keys), brain-shaped (so it ports), and dependency-free where possible.

---

## 10. Open questions (carried from the design)

- **Expansion rule (issue #1)** — growth cap and overlap gate (§5.2). *Resolved in-sim:* an overlap floor
  (`expansionOverlapFloor`) gates absorption; ~0.34 keeps the biggest pattern from swallowing the field.
- **Merge/split equilibrium (issue #2)** — does a regional regime exist, and under what rates (§8). *Resolved:*
  yes — see §11.
- **Adaptive error thresholds** — the hoped-for split lever; form and effect (§5.3). Implemented
  (`adaptiveError`, conservative/neutral/aggressive Welford modes); aggressive is the climb lever.

---

## 11. Results (the sim is built and validated)

Built in place in [`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) — deterministic,
dependency-free, brain-shaped. CLI modes: ASCII shapes (default), `mnist` (structure), `acc` (supervised
readout), `sweep` (merge × error), `lifecycle` (merge/split over a stream), `climb` (per-level funnel). All
results below are 14×14 binary MNIST, spatial-only (d=0).

**The mechanism works end to end.** Recognition fires existing patterns (Jaccard context match ≥ θ), they
predict L0 and mispredict, requests cluster by transitive merge (≥1-neighbor rule), and reuse-or-mint installs
for a later frame. The hierarchy climbs (L1 → L2 → L3+).

**Merge/split equilibrium (the §8 headline) — a regional regime exists.** Two regimes over the same stream:

- **Merge only** (no split): footprints run away — L1 median grows to ~54px and max saturates near the whole
  field; population grows unbounded; watched patterns grow and never come back down. The "one big blurry
  pattern" failure.
- **Merge + refine**: footprints settle **regional** — L1 median ~17–22px; individual patterns visibly **grow →
  shrink → survive → spawn higher-level children** (per-position reliability pruning specializes them); L2/L3
  climb. So the balanced middle the design hoped for is real, under refinement as the split force.

**Split force = refinement (no forgetting/decay).** Refinement (a) consolidates context/targets toward the
common core and prunes footprint positions a pattern predicts unreliably (specialize), and (b) **weakens parent
references for parents absent when the pattern fires; a pattern that drains its last reference is reaped** — the
multi-parent reference-count corollary (§4.1), not a time-based decay. Culling is therefore selective: a pattern
dies only when it drifts off all the contexts that minted it.

**The climb and reuse.** With reuse on, the hierarchy reaches L2–L3; the L2→L3 funnel shows valid requests that
**reuse** an existing lower pattern (by L0 output) rather than minting a new level — this is **correct**
content-addressable behavior, not a bug (with reuse forced off, the same requests mint and the hierarchy climbs
to the level cap). Reuse is the merge force; it is gated by the same θ as recognition.

**Accuracy (downstream check).** Supervised NB-product readout over fired patterns: **~50% test at 400 train,
~58% at 2000 train** (best operating point, still climbing with data) — well above chance (10%), below the full
spatial brain's ~96% as expected for a small single-distance sim. The merge threshold has a clear sweet spot:
**θ ≤ 0.5 is degenerate** (patterns merge to a blob, predict one digit, ~14%); **θ ≈ 0.7–0.9** discriminates;
**θ too high overfits** (memorizes train, no-match on test). Higher error tolerance keeps patterns general and
deepens the climb.

**Brain-applicability review.** No algorithm bugs; deterministic; eval freezes mutation. Every structure maps
to a brain op (whole-field black/white base ↔ encoder; Jaccard recognition ↔ `match_observed`; per-position
modal L0 + `(missing+novel)/union` error ↔ `mint_spatial_corrections`; transitive-merge cluster ↔ Phase B;
reverse target index ↔ Phase A; multi-parent refcount reaping ↔ §4.1). Port notes: per-position reliability
pruning and parent-reference draining are the concrete realization of target refinement (refinement.md §3) +
refcounted reaping; the sim's child index is an internal cascade accelerator (the brain uses the existing
`DeleteNeuron` cascade).
