# Wavefront Experiments — Test Runs

One row per run. Columns: **options · depth · neurons · level distribution · test acc · train acc**.
Common flags unless noted: `--buckets 2 --columns 20 --per-class 0 --episodes 1`.
Test = frozen held-out (`--disable-learning --test-data`) on the saved brain.

Each session appends its own runs below, newest first. Keep the columns consistent.

---

## Session 2026-07-23 — UCAR Phase 1 in brain.rs: the level-0 configuration loop

**First brain.rs run of the [UCAR](algorithm.md) design.** Replaced the spatial path's threshold-based
grouping (Jaccard `match_observed` + adaptive Welford thresholds) and the womb/embryo minting with the
**level-0 configuration loop**: a routing table (normal + children as stored configurations), a per-neuron
**history window** (horizon 1000), **Hamming distance** `|O △ C|`, the **one test** (add/delete over the
history), and **median refinement**. Minting is capped to base (level-0) parents, so the hierarchy settles at
**depth 2** (Phase 1: "capped at one level"). Newborns are created **with no connections** (per spec).
Temporal (`d>0`) is untouched — the design's open port.

Phase 1's gate is **affordability + churn + counts, not accuracy** (accuracy is the Phase-3 readout gate). All
runs `--image-size 7 --buckets 2 --columns 20`.

| options | depth | active L1 | minted / deleted cum | neurons | train acc | test acc |
|---|---|---|---|---|---|---|
| per-class 100 (1000 img), 1 ep, save | 2 (L1) | 396 | 592 / 196 | 483 | 56.3% | **54.4%** (frozen, 500 test) |
| per-class 200 (2000 img), 1 ep | 2 (L1) | 449 | 665 / 216 | 536 | 58.0% | — |

**Findings.**
- **The one test must be symmetric.** A first cut evaluated the delete pass over the frames a child was
  *recorded* serving while the add pass evaluated over frames a candidate *would win* — an asymmetry that made
  minting/deletion oscillate: **~21,900 mint + 21,800 delete over 2000 frames (~11/frame)**. Making both halves
  evaluate over "the frames the entry is currently the closest for, with `next(O)` the runner-up" (the design's
  "same test") collapsed churn to **665 mint / 216 delete (~0.3/frame)** and raised accuracy 49.7% → 58%. This
  is the load-bearing invariant of the design ("an entry cannot fail the test and then immediately pass it").
- **The add pass is the expensive step** (as the design flags). It is `O(history × entries)` per active neuron
  per frame; training throughput falls from ~340 img/s to ~60 img/s as the history fills toward the horizon and
  children accumulate (~450). Gating the add pass to error>0 frames helped little (most frames carry error).
  Frozen eval — no add pass — runs at ~600 img/s. Incremental per-child benefit sums (design: "the delete pass
  folds into the routing scan") are the obvious next optimization, deferred.
- **Two spec-conformance fixes surfaced via save/restore** (dangling connection targets to churned patterns):
  the old event-edge substrate (`learn_spatial_event_connections`) is now unread and not written; newborns are
  created with no connections. Save → frozen-load round-trips cleanly after both.
- **Depth-2 cap holds**; structure stabilizes (~400–450 active L1 patterns); accuracy (~54–58%) is far below the
  leveled baseline's 71% **as expected** — no contraction (Phase 2), no readout tuning (Phase 3).

*Deferred to Phase-1 follow-up:* remove the vestigial experimental toggles (`match_info`/`error_info`/`trace_*`)
and old spatial fields (Welford `spatial_error_stats`, `spatial_target_dims`, `spatial_inference_*`) that now
warn as dead — they ripple into the napi/JS constructor surface. Wire the horizon to a Brain construction option.

---

## Session 2026-06-25/26 — reference simulation (JS, not brain.rs): build + validate, then found the model wrong

**Not a brain.rs run.** These are runs of the standalone JS reference simulation
[`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js), built across this session's stages
A1 (restructure to mirror the brain's `Brain`/`process_spatial_levels` shape), A2 (real MNIST input + inverted
adjacency index), and A3 (supervised NB-product readout). All runs below are the **cold-start, single-frame,
group-by-observed-set** mechanism — later in the same session this was found to be the **wrong** model of
reuse (no recognition, no cross-frame state, doesn't match how the brain actually predicts L0 and clusters
mispredictions). The corrected model and simulation spec are in [neuron-reuse.md §3](./neuron-reuse.md) and
[neuron-reuse-simulation.md](./neuron-reuse-simulation.md); these numbers are historical — they characterize
the exact-match, no-recognition baseline, not the corrected mechanism.

### ASCII shapes — locality + reuse sanity (`node apps/mnist/jobs/wavefront-sim.js`)

Pre-restructure baseline (also reproduced byte-identically after the A1 `Brain`-class restructure — 0-line
diff):

| shape | L1 patterns | L1 max footprint | L1 reuse-groups | converges to |
|---|---|---|---|---|
| plus of pluses (21px) | 13 | 5px | 4 | 1 pattern @ L3 (9 parents) |
| uneven blob (18px) | 17 | 5px | 1 | stable, multiple patterns @ L4 |
| ragged patch (24px) | 23 | 5px | 1 | stable, multiple patterns @ L4 |
| L shape (7px) | 7 | 3px | 0 | 1 pattern @ L3 (5 parents) |
| solid 4×4 block (16px) | 16 | 5px | 0 | 1 pattern @ L3 (9 parents) |
| diagonal, 4-conn (5px) | — | — | — | **bug (pre-fix): wrongly merged to 1 pattern** — see below |
| two separate blobs (14px) | — | — | — | **bug (pre-fix): wrongly merged to 1 pattern, 2 parents** — see below |
| diagonal, 8-conn (5px) | 5 | 3px | 0 | 1 pattern @ L2 (5 parents) |

**Bug found and fixed:** isolated units (empty observed-set) were all grouped under one key, so disconnected
shapes wrongly bound together (two separate blobs → 1 pattern; a 4-connected diagonal → 1 pattern spanning all
5px, i.e. "level 1 = whole shape"). Fix: give every isolated unit a unique key. Re-run after the fix:

| shape | result after fix |
|---|---|
| diagonal, 4-conn | stays 5 disconnected 1px units (converged level 2, "footprints stable") |
| two separate blobs | stays 2 patterns, 7px each (converged level 3, "footprints stable") |

Locality held throughout: level-1 max footprint never approached the shape size on any input (clean or
irregular), confirming the hierarchy doesn't collapse to one level-1 blob.

### MNIST structural diagnostics (`node apps/mnist/jobs/wavefront-sim.js mnist <count> <imageSize> <radius>`)

No supervised readout — depth / pattern-count / footprint-size / neuron-count / timing only, real binary MNIST
test images, 8-connectivity (Moore, matching the brain's encoder).

| options | avg active px | avg depth | avg L1 max footprint | avg neurons | avg apex size | ms/img |
|---|---|---|---|---|---|---|
| 10 imgs, 14×14, r1 | 22px | 3.3 | 6.3px | 62 | — (apex-count bug, see below) | 3.3 |
| 10 imgs, 28×28, r2 (the 96.44% config) | 93px | 3.4 | 18.8px | 228 | — | 88.9 (tail to ~340ms on dense digits 0/9) |
| 6 imgs, 14×14, r1 (after apex fix) | 20px | 3.3 | 5.7px | 58 | 1 (2 on one image) | 3.6 |

**Bug found and fixed:** the top level's corrections were minted but never re-added to `fired_set` (the sweep
ends before reprocessing them), so `apex` always reported 0. Fixed by adding the final level's corrections to
`fired_set` before computing `apex = fired \ subsumed`.

**Finding:** level-1 stays local on real digits too (max ~19px of ~93 active at 28×28 r2, ≈20%), depth bounded
3–4 — matching the brain's own spatial-processing depth findings ([[project_spatial_mnist_findings]]).
28×28 r2 is slow for bulk sweeps (~89ms/img avg, worse tails) — fine for oracle spot-checks, not for a full
10k-image pass without a perf pass first.

### MNIST accuracy oracle (`node apps/mnist/jobs/wavefront-sim.js acc <trainCount> <testCount> <imageSize> <radius>`)

Supervised NB-product readout: every pattern (all levels) is a voter keyed by its **footprint signature**;
train accumulates per-signature digit counts, eval decodes via `argmax_d Σ log(P(d|voter)+ε)`. This is
**exact-match only** — no merge threshold, no recognition, cross-image "reuse" happens only when two images
produce byte-identical footprint signatures.

| train images | test images | image size | radius | distinct pattern signatures | train acc | test acc |
|---|---|---|---|---|---|---|
| 2000 | 1000 | 14×14 | 1 | 45,564 (4.2s to train) | 100.00% (1000/1000) | 82.60% (826/1000) |
| 5000 | 1000 | 14×14 | 1 | 104,049 (10.2s to train) | 100.00% (1000/1000) | 87.10% (871/1000) |

Train is always 100% (exact signature self-match). Test rises with data (82.6%→87.1%, 2k→5k train) — the
train/test gap is the **memorization signature** of exact-match reuse: patterns don't generalize, they just
recur exactly often enough across images to cover the test set. 87.1% at only 5k training images already
approaches the brain's 14×14 NB ceiling (~89–90% at full ~54k, see [[project_spatial_mnist_findings]]),
despite having zero recognition/tolerance mechanism — small local footprints simply repeat exactly across
digits more than expected.

**Why these numbers don't validate the real mechanism (found later this session):** the exact-match, no
recognition, single-frame model these runs characterize was subsequently determined to be wrong — real neurons
recognize context ≥ a merge threshold and predict L0; only mispredictions become correction requests, which are
then clustered by transitive merge over neighborhoods (not bucketed by identical observed-set), across many
frames. See [neuron-reuse.md §3](./neuron-reuse.md) for the corrected model and
[neuron-reuse-simulation.md](./neuron-reuse-simulation.md) for the rebuild spec. The numbers above remain useful
as an exact-match/no-recognition baseline (e.g. for comparison once the corrected sim adds a merge threshold),
but should not be read as evidence for the reuse mechanism itself.

---

## Session 2026-06-26 — corrected model built: mechanism bugs, merge/split, climb, threshold coupling

Follows directly from the A1–A3 entry below (2026-06-25, which built the **wrong**, obsolete cold-start
group-by-observed-set model). This entry covers the **corrected** model — recognize → predict L0 → on
misprediction cluster by transitive merge → reuse/mint, refinement as the split force — rewritten in place in
[`apps/mnist/jobs/wavefront-sim.js`](../apps/mnist/jobs/wavefront-sim.js) per
[neuron-reuse-simulation.md](./neuron-reuse-simulation.md). All runs are 14×14 binary MNIST unless noted.

### Bugs found and fixed while validating (in order)

1. **Global recognition** — a pattern's context was matched against the *whole* active frame instead of its
   local neighborhood; with the whole base field active, novel co-actives drowned every local pattern (only one
   giant pattern ever matched). Fixed to local recognition (`localObservedForRecognition`).
2. **Vote-collection id bug** — `firedPatternIds` collected `p.id` off the `{pattern, match, observedL0}`
   recognition-result wrapper (always `undefined`) instead of `r.pattern.id`. Only one phantom voter ever
   existed until fixed — accuracy jumped from ~10% (chance) to real signal the moment it was fixed.
3. **Arbitrary reference constants** — an initial reference-counting design used `refInit=2`, `refCap=5` with no
   derivation. Corrected to the brain's grounded arithmetic: references are born at 1, ±1 per refinement event,
   deleted at 0, **no cap** (matches `Context`/connection strengthening in the brain — no decay backstop).
   Re-running confirmed the cap was never the limiter (culling rate unchanged, 1 death either way) — the real
   reason culling is rare is structural: a pattern only fires (and thus only gets refined) on frames where its
   context matches, which correlates with its parents being *present*, so weakening events are inherently rare.

### Mechanism smoke tests

| run | result |
|---|---|
| ASCII shapes, default (`node wavefront-sim.js`) | 0 patterns minted on every shape — **correct**, not a bug: the same shape repeats every frame, so after frame 1 the learned prediction exactly matches and nothing errors. Minting needs cross-frame *variation* (real images), which the ASCII demo doesn't have. |
| `mnist 100 14 1` (structure only, pre-local-recognition-fix) | 108 patterns after 100 images. L0: reused 10/118 (8%); L1: 12 requests all reused (100%) — the local-recognition bug (below) meant almost nothing could ever recognize, but reuse (global, unaffected by the bug) still fired. |

### Supervised readout — `acc <train> <test> 14 [merge] [error] [refine]`

| train | test | merge | error | reuse | refine | patterns | voters | test acc | train acc | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 200 | 0.5 (default) | 0.3 (default) | on | off | 516 | 1 | 8.50% | 10.50% | pre-fix #1: global recognition, essentially nothing recognized |
| 300 | 150 | 0.5 | 0.3 | on | off | 326 | 1 | 8.50%→ still broken | — | pre-fix #2: local recognition fixed, but vote-collection id bug still zeroed voters |
| 300 | 150 | 0.5 | 0.3 | on | off | 326 | 326 | **28.00%** | **32.67%** | both bugs fixed — first valid accuracy number |
| 60 | 60 | 0.5 | 0.3 | on | off | — | — | — | — | (superseded by sweep below) |
| 2000 | 500 | 0.9 | 0.4 | on | on | 6,263 | 6,463 | **58.20%** | 76.20% | best-so-far config at scale; deepest sweep reached L4; 43 min wall-clock. L0 min 0% reuse, L1 31% reuse — reuse barely fires at θ=0.9 |
| 60 | 60 | 0.66 | 0.33 | on | on | 129 | 128 | 40.00% | 56.67% | best test acc seen at 60 images — but see next row |
| 500 | 500 | 0.66 | 0.33 | on | on | 1,002 | 1,001 | **27.00%** | 32.60% | same config at scale gets *worse* — 0.66 is left of the true optimum (see coupled sweep, peaks at θ=0.8); low θ degrades with more data as merge blurs patterns (train acc also collapsed 56.7%→32.6%, not overfitting) |

**Merge-threshold sweep — `sweep <train> <test> 14`** (reuse+refine on):

Smoke (60/60):

| merge | error | test acc | train acc | patterns | maxLvl |
|---|---|---|---|---|---|
| 0.30 | 0.20 | 10.00% | 15.00% | 59 | 2 |
| 0.30 | 0.40 | 10.00% | 15.00% | 87 | 2 |
| 0.50 | 0.20 | 10.00% | 15.00% | 62 | 3 |
| 0.50 | 0.40 | 10.00% | 15.00% | 159 | 3 |
| 0.70 | 0.20 | 35.00% | 56.67% | 107 | 3 |
| 0.70 | 0.40 | 36.67% | 60.00% | 234 | 4 |
| 0.90 | 0.20 | 21.67% | 98.33% | 105 | 3 |
| 0.90 | 0.40 | 33.33% | 91.67% | 217 | 3 |

Full (400/400):

| merge | error | test acc | train acc | patterns | maxLvl | no-match |
|---|---|---|---|---|---|---|
| 0.30 | 0.20 | 14.25% | 13.00% | 392 | 2 | 0/400 |
| 0.30 | 0.40 | 14.25% | 13.00% | 420 | 2 | 0/400 |
| 0.50 | 0.20 | 14.25% | 13.00% | 405 | 4 | 0/400 |
| 0.50 | 0.40 | 14.25% | 13.00% | 714 | 3 | 0/400 |
| 0.70 | 0.20 | 31.50% | 41.75% | 643 | 3 | 31/400 |
| 0.70 | 0.40 | 36.00% | 54.25% | 1,184 | 4 | 0/400 |
| 0.90 | 0.20 | 23.00% | 98.00% | 680 | 3 | 78/400 |
| **0.90** | **0.40** | **49.75%** | 82.75% | 1,345 | 3 | 1/400 |

**Reading:** merge ≤ 0.5 is degenerate (patterns merge into one blob, always predict the same digit — 14.25% is
that digit's base rate, not learning). θ 0.7–0.9 discriminates. θ=0.9/error=0.2 overfits (98% train, only 23%
test, 78 no-match — patterns too specific for unseen digits); error=0.4 keeps them general enough to fire.

### Merge/split equilibrium — `lifecycle <frames> 14 [only]`

Answers the design's open question (neuron-reuse.md §3.6): does pattern size settle *regional*, or collapse to
one big blurry pattern?

**merge-only** (no split force) — runs away, never settles:

| frame | live | L1# | L1 med/max | L2# | maxLvl |
|---|---|---|---|---|---|
| 20 | 28 | 19 | 20/113 | 6 | 2 |
| 100 | 65 | 58 | 18/151 | 7 | 2 |
| 200 | 167 | 155 | 26/162 | 12 | 2 |
| 300 | 276 | 255 | 32/164 | 21 | 2 |
| 500 | 460 | 437 | 24/169 | 23 | 2 |

Watched individual patterns only ever grow (`█████████████████████`, e.g. peak 140–161px, never shrink), L2
frozen at 14 once merge-only saturates, zero deaths/prunes ever.

**merge + refine** (refinement as the *only* split force — no forgetting/decay, per user correction; culling is
purely the multi-parent reference-drain corollary): footprints specialize, some patterns oscillate
grow↔shrink↔survive and spawn children, but population still grows (no forced turnover) since deaths are rare
(1 in 180–500 frames — expected: a pattern is only refined, hence only weakened, on frames where it fires, which
correlates with its parents being present, so weakening events are inherently uncommon under this design).

500-frame run, exact per-level neuron counts (this is also the answer to "how many neurons per level, how do
they change"):

| frame | L0 (base) | L1 | L2 | L3+ | total |
|---|---|---|---|---|---|
| 20 | 285 | 19 | 6 | 0 | 25 |
| 100 | 305 | 58 | 7 | 0 | 65 |
| 200 | 312 | 155 | 12 | 0 | 167 |
| 300 | 315 | 255 | 21 | 0 | 276 |
| 360 | — | 315 | **23** | 0 | 338 |
| 500 | 319 | 437 | 23 | 0 | 460 |

L0 saturates fast then flat (319 of a possible 392 = 2×196, borders never see both values); **L1 grows linearly
with no ceiling** (this is where essentially all population growth is — culling is too rare to bound it); L2
grows then plateaus at 23 once L1 specializes enough to stop mispredicting as often; L3+ never appears at this
scale/frame count (see climb experiment below for why).

Watched-pattern life traces (180-frame run, sampled across the size range, not just the biggest):

| pattern | trajectory | peak→trough | fate | children |
|---|---|---|---|---|
| L1#30 | `█▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆` | 18→12px | alive, sharpened | L1:17 L2:4 |
| L1#25 | `██▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆` | 14→10px | alive, sharpened | L1:11 L2:1 |
| L1#20 | `▆▆▅▇▆████████████████` | 155px, oscillates | alive (expansion regrows it) | L1:1 |
| L1#3 | `█████████████████████` | flat @2px | alive, tiny/stable | L1:6 L2:6 |

**Verdict:** merge-only collapses toward one big blurry pattern that never dies. Merge+refine settles toward
regional footprint sizes with real specialize-and-survive behavior — the hypothesis in
[neuron-reuse.md §3.6](./neuron-reuse.md) holds — but culling is rare rather than a strong population control,
which is an expected consequence of the design (see [neuron-reuse-simulation.md §11](./neuron-reuse-simulation.md)).

An earlier version of this experiment used a bolted-on `forgetRate` strength-decay as the split force (not in
the design — flagged and removed): `lifecycle 250/300/400` runs with `forgetRate=0.05–0.12` + aggressive
adaptive error showed forced turnover (deaths ~10/checkpoint, population plateauing ~150–200) and individual
patterns dying outright (`peak 153 → trough 113px → REAPED`). Superseded by the refinement-only mechanism above
per user correction; kept here only as a historical data point, not a valid model reference.

### Climb experiment — does L3+ ever form? (`climb <frames> 14 [reuse]`)

Built to check a suspected bug (L3 never appeared in the lifecycle runs above). Instruments a per-level funnel:
active → mispredict → (≥1-neighbor gate) → request → mint vs reuse.

**Reuse off** (forces every valid request to mint): climbs to the level cap (12) almost immediately —

| frame | L1 | L2 | L3 | L4 | … | L12 |
|---|---|---|---|---|---|---|
| 40 | 47 | 39 | 32 | 30 | … | 3 |
| 160 | 167 | 159 | 145 | 142 | … | 105 |
| 320 | 327 | 319 | 298 | 295 | … | 255 |

**Reuse on** (400 frames) — funnel shows exactly where L3 dies:

| level | active | mispredict | neighborless | requests | minted | reused |
|---|---|---|---|---|---|---|
| L0→L1 | 78,400 | 27,178 | 0 | 27,178 | 356 | 52 |
| L1→L2 | 35,164 | 22,793 | 4 | 22,789 | 23 | 376 |
| L2→L3 | 4,633 | 3,675 | 3 | 3,672 | **0** | **351** |

Live: 378 patterns (L1:355, L2:23), deepest sweep level 3, **0 L3+ patterns**.

**Verdict (per user: correct behavior, not a bug).** All 3,672 valid L2→L3 requests **reuse** an existing lower
pattern (matched by L0 output, across levels) instead of minting — genuine content-addressable reuse, not a
defect. The climb mechanics themselves are sound (proven by the reuse-off run reaching L12 instantly); reuse
just dominates at this θ/scale, which is expected design behavior, not something to "fix."

### Threshold-coupling discovery — `couple <train> <test> 14`

User's hypothesis: merge threshold and error threshold are the same "sameness" operation read from opposite
sides, so `error = 1 − merge` should hold as an identity (true by construction: both are the same Jaccard,
`common/(common+missing+novel)` vs `(missing+novel)/(...)`, which sum to exactly 1). Swept θ with error pinned
to `1−θ`, head-to-head against the decoupled grid above (train/test 400):

| θ (merge) | error = 1−θ | test acc | train acc | patterns |
|---|---|---|---|---|
| 0.50 | 0.50 | 14.25% | 13.00% | 916 (degenerate) |
| 0.60 | 0.40 | 21.25% | 24.25% | 1,091 |
| 0.70 | 0.30 | 43.75% | 58.25% | 811 |
| **0.80** | **0.20** | **50.75%** | 91.75% | 727 |
| 0.90 | 0.10 | 23.00% | 98.00% | 680 (overfit) |

**Coupled best (50.75% @ θ=0.8) ties/edges the decoupled best (49.75% @ merge 0.9/error 0.4)** — collapsing two
thresholds into one costs nothing at this scale. This is now its own project,
Adaptive grouping sequenced before the wave-front and reuse projects
since it touches recognition/correction brain-wide.

---

## Session 2026-06-27 — wave-front migration (Stages 1–4) + footprint/accuracy correctness fixes

Implemented the full wave-front migration per [wavefront.md](./wavefront.md) /
[wavefront-implementation.md](./wavefront-implementation.md): Stage 1 footprints (additive) → Stage 2
switch neighborhood to footprint adjacency → Stage 3 coordinate-less corrections → Stage 4 remove stored
levels. Gated on MNIST 7×7/14×14 and stocks demos 3/4. A subsequent manual code review of the diff (user,
file by file) found and fixed two real bugs: temporal processing was wrongly filtered by a footprint
neighbor graph (should have none — temporal sequences against all active neurons), and event accuracy was
scored against the post-subsumption spatial apex instead of the raw base sensory events. Also traced demo
4's episode-2 accuracy dip to a `resetContext` cold-start artifact, not real degradation. Rust unit tests
stayed green throughout (101 by the end, up from the pre-migration 89).

### MNIST 7×7 — wave-front gate (Stages 1–4)

```bash
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --episodes 1 --save-brain mnist7
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist7 --disable-learning --test-data
```

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 7×7 r1 (documented pre-migration baseline, not re-run) | — | — | — | 71.02 | — |
| 7×7 r1, wave-front Stages 1–3 | 3 | 378,722 / 378,723 | L1:744 L2:303,973 L3:73,909 | 69.56 (6956/10000) | 66.08 (35822/54210) |
| 7×7 r1, wave-front Stage 4 (levels removed) | 3 | 378,722 / 378,723 | L1:744 L2:303,973 L3:73,909 | 69.56 (6956/10000) | 66.08 (35822/54210) |

Stage 4 row is **bit-identical** to Stages 1–3 (same neuron counts, same accuracy to the exact fraction),
confirming removing stored levels is behavior-neutral. Train wall-clock 19m43s / 18m41s, eval 7m39s /
7m28s — roughly 5× the documented ~4-min baseline, from the L2/L3 explosion (see next section). Stage 4's
per-frame diagnostic line now reports the live activation-derived snapshot (e.g. `depth=2 | L1:36 L2:7 | 43
active`) rather than the old cumulative per-level population — the final `378,723 neurons` total still
matches exactly; only the presentation of the progress line changed.

### Footprint population analysis (mining the saved `mnist7` backup)

Wrote a standalone script (`analyze-mints.js`, scratchpad-only) to characterize the minted correction
population directly from `patterns.csv` + `contexts.csv`: per spatial level, context width (neighbors
bound) and how many of those context sets are **distinct** across the whole 54,210-image training set.

| level | corrections | avg ctx width | max ctx width | distinct ctx sets | distinct % |
|---|---|---|---|---|---|
| L1 | 744 | 7.4 | 8 | 673 | 90.5% |
| L2 | 303,973 | 24.3 | 46 | 303,850 | **100.0%** |
| L3 | 73,909 | 16.4 | 39 | 73,815 | 99.9% |

L1 stays bounded (small ≤8-neighbor contexts recur). L2's context width triples (7.4→24.3) and **100% of
L2 contexts are unique** — every one of the 303,973 L2 corrections has a context that never recurs anywhere
in training, so recognition can never match and a fresh correction mints every time. L2 alone is ~80% of
the total neuron count. Root cause is architectural (wide footprint "touch" meeting combinatorics at low
resolution), not a code bug; expected to be absorbed by neuron reuse + refinement, not by tuning footprint
growth.

### MNIST 14×14 — wave-front gate

```bash
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --episodes 1 --save-brain mnist14
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist14 --disable-learning --test-data
```

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 14×14 r1 (documented pre-migration baseline, not re-run) | — | — | — | 95.01 | — |
| 14×14 r1, wave-front (train) | 2 | 1,769,420 | L1:81 L2:20 | — | 89.08 (48290/54210) |
| 14×14 r1, wave-front (eval, first attempt) | — | — | — | **OOM crash** (`memory allocation of 466960 bytes failed`, mid-load) | — |
| 14×14 r1, wave-front (eval, re-run after increasing Windows pagefile; user-run) | 2 | 1,769,420 | — | **92.81** | — |

Train took 2h10m24s. The first eval attempt loaded the 1.77M-neuron brain and then OOM'd (a Rust/native
allocation failure — no test number produced). After the user increased Windows virtual memory and reran
manually, eval completed in ~1.25h at 92.81% (−2.19pp vs the 95.01% baseline) — a larger drop than 7×7's
71.02%→69.56% (−1.46pp), consistent with wider correction contexts at higher resolution producing more
one-off corrections and noisier NB voters.

### Stocks — demo 3 & demo 4 cross-domain confirmation (wave-front, Stages 1–4)

First attempt was piped through `tail`, which buffers until exit — no live progress was visible, the user
asked to kill it, and it was aborted with no valid data (see [[feedback_track_long_outputs]] equivalent:
capture output straight to a file for any run over ~30s). Relaunched trackable (direct output, no `tail`):

| demo | net profit | ROI | Sharpe | trades | notes |
|---|---|---|---|---|---|
| 3 — 50 symbols, 1 pass (documented baseline, not re-run) | +$2.56M | +17084% | 0.42 | — | reference |
| 3 — 50 symbols, 1 pass, wave-front | +$1,339,491.83 | +8929.95% | 0.33 | 28,675 | ~half baseline profit/Sharpe (accuracy 2.11%, pre accuracy-fix metric — not comparable, see below) |

| demo 4 episode | net profit | ROI | Sharpe | trades |
|---|---|---|---|---|
| 1 | $980,593.06 | +6537.29% | 0.26 | 8,419 |
| 2 | $31,834,578.74 | +212230.52% | 0.53 | 8,689 |
| 3 | $7,046,782,891.32 | +46978552.61% | 0.88 | 8,671 |
| 4 | $36,764,371,968.80 | +245095813.13% | 1.03 | 8,577 |
| 5 | $932,254,167,812.06 | +6215027785.41% | 1.23 | 8,620 |

Documented baseline Sharpe climbs to ~0.96 by episode 5; wave-front reaches **1.23** — improved.

### Bug 1 — temporal footprint over-restriction, and its fix

Diagnosed during the user's code review: footprint adjacency was unified onto one base graph and applied to
**both** spatial and temporal (matching the doc's stated intent), but this broke every config that declares
an *empty* spatial neighbor list to mean "spatial off, temporal all-pairs" (`setSpatialNeighbors(sym, [])`)
— under one graph, empty also restricted temporal to self-only.

| run | result |
|---|---|
| non-spatial stocks (empty-neighbor-list path), mid-fix | $0.00 net, 0 trades, 43.62% acc — flagged suspicious |
| `synthetic-extended-test.js --group-mode static --group-threshold 0.9`, mid-fix (user-run, regression report) | **58.3%** optimal rate (down from ~97%) |
| same command, after reverting temporal to "no neighborhood — sequences against all active neurons" | **233/240 = 97.1%** optimal rate (recovered) |

Resolution: footprint adjacency stayed a **spatial-only** locality primitive; temporal filtering was
removed entirely (temporal connections, actuals, and context now span every active neuron, no adjacency
gate).

### Bug 2 — event accuracy scored against the wrong set, and its fix

`track_inference_performance` compared inferred winners against the temporal age-0 set (the **post-
subsumption spatial apex**, containing correction patterns) instead of the raw spatial base level. A real
event absorbed into a spatial correction was invisible to the apex, so correct predictions were scored as
misses.

| run (`KGC,GOLD,SPY --context-length 3 --group-mode static --group-threshold 0.9 --spatial`) | net profit | Sharpe | trades | base accuracy |
|---|---|---|---|---|
| pre-fix (user-reported) | $13,901,536.87 | 0.69 | 5,821 | **1.88%** |
| post-fix (accuracy scored against `get_spatial_base_level`) | $13,901,536.87 (identical) | 0.69 (identical) | 5,821 (identical) | **51.54%** |

Trading is byte-identical; only the accuracy measurement changed — confirms it was a pure scoring bug.

### Demo 4 — full re-run, post both fixes (exact match to [stock-demos.md](./stock-demos.md))

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --context-length 3 --columns 20 --no-summary --episodes 5 --spatial
```

| episode | net profit | ROI | Sharpe | trades | base accuracy |
|---|---|---|---|---|---|
| 1 | $980,593.06 | +6537.29% | 0.26 | 8,419 | 50.26% |
| 2 | $31,834,578.74 | +212230.52% | 0.53 | 8,689 | 46.08% |
| 3 | $7,046,782,891.32 | +46978552.61% | 0.88 | 8,671 | 46.21% |
| 4 | $36,764,371,968.80 | +245095813.13% | 1.03 | 8,577 | 46.28% |
| 5 | $932,254,167,812.06 | +6215027785.41% | 1.23 | 8,620 | 46.44% |

Exact match to the documented values (profit, Sharpe, trades, and accuracy columns all reproduce to the
digit).

### Demo 4 — episode-2 accuracy dip root cause (2-episode full-log trace)

Ran demo 4 without `--no-summary` (full per-frame log, 10,746 frame lines) to get the instantaneous
(windowed) accuracy trajectory, not just the per-episode average:

| episode | frame | neurons | cumulative acc% | instantaneous acc% |
|---|---|---|---|---|
| 1 | 100 | 2,525 | 50.10 | 50.1 |
| 1 | 1000 | 4,809 | 50.60 | 51.1 |
| 1 | 3000 | 5,204 | 51.00 | 51.0 |
| 1 | 5373 (end) | 6,090 | 51.10 | 51.1 |
| 2 | 100 | 5,329 | 39.60 | 39.6 |
| 2 | 1000 | 5,179 | 43.20 | 45.9 |
| 2 | 3000 | 5,407 | 45.40 | 46.4 |
| 2 | 5373 (end) | 6,004 | 47.30 | 52.7 |

`resetContext()` runs at the start of every episode (`test.js:246`), clearing the temporal sliding window
but keeping learned patterns. Episode 1 starts nearly empty so the cold start costs almost nothing (flat
~51% the whole way). Episodes 2+ start with the full pattern set firing on a freshly-cleared context and
mispredict until the window re-warms — dragging the per-episode *average* down even though the
instantaneous/steady-state accuracy is actually **rising** across episodes (51.1% end of ep1 → 52.7% end of
ep2). Documented in [stock-demos.md](./stock-demos.md) under "Action Learning in Low Accuracy".

---

## Session 2026-06-27 — Adaptive Grouping: unify thresholds, unify the operation, mode sweep

Two-stage project (design in the now-folded [error-driven-learning.md](./error-driven-learning.md) "Grouping"
section; the standalone `adaptive-grouping.md` design doc was deleted after conclusion, roadmap §1 removed).
**Stage 1** collapsed the six knobs (`{spatial,temporal}MergeThreshold` / `…ErrorCorrectionThreshold` /
`…ErrorCorrectionMode`) into one `groupThreshold` + `groupMode`, with `error = 1 − θ` as a derived identity.
**Stage 2** made recognition and correction the same Jaccard-union comparison under one adaptive per-unit
threshold (`Neuron::grouping_error_threshold` as sole reader; `merge = 1 − E`), moved temporal correction from
containment to union, and — after re-testing — moved temporal recognition to union as well. Concluded and
committed this session; `groupMode` default flipped from `conservative` to `neutral`.

### Stage 1 demo re-validation (six knobs → `groupThreshold`/`groupMode`, static mode — bit-exact check)

Demos 1, 2, and 5 pin `--group-mode static`, so Stage 1 (a pure rename/derivation with no math change under
static) reproduces the prior decoupled baselines to the fraction:

| demo | metric | before | after (Stage 1) |
|---|---|---|---|
| 1 — single-channel synthetic cycle | optimal rate | 233/240 = 97.1% | 233/240 = 97.1% (exact match) |
| 2 — multi-channel synthetic cycle | optimal rate | 696/720 = 96.7% | 695/720 = 96.5% (−1 frame, noise) |
| 5 — sequence memorization, ep1 | base accuracy | 51.03% | 51.61% |

### Stage 2 re-test — temporal correction/recognition: containment → Jaccard union

A stale code comment claimed switching temporal to the union denominator hurt first-time accuracy. Re-tested
on the current architecture with the full stock pipeline (`test.js --no-summary --episodes 5 --symbols
KGC,GOLD,SPY --context-length 3 --forget-rate 0.0005 --group-mode static --group-threshold 0.9`):

| episode | base accuracy (containment) | base accuracy (union) |
|---|---|---|
| 1 (first-time) | 51.03% | **51.61%** (held, did not regress) |
| 2 | 55.54% | 62.41% |
| 3 | 57.72% | 65.13% |
| 4 | 58.46% | 66.00% |
| 5 | 58.90% | **66.59%** |

First-time accuracy held flat-to-better and cross-episode learning improved markedly (+7.7pp by episode 5);
Sharpe ep1 rose −0.02 → 0.29. The claimed regression did not reproduce — union adopted for spatial **and**
temporal, recognition and correction, with no remaining spatial/temporal asymmetry in the comparison.

### Stage 2 mode sweep — demo 4 (10 symbols, 5 episodes, `--spatial`), adaptively-coupled recognition

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --context-length 3 --columns 20 --no-summary --episodes 5 --spatial --group-mode <conservative|neutral|aggressive>
```

| mode | ep1 Sharpe | ep5 Sharpe | ep1 profit | ep5 profit | ep1 base acc | ep5 base acc |
|---|---|---|---|---|---|---|
| conservative | 0.42 | 0.84 | $12,586,724.46 | $4,343,981,330.88 | 14.76% | 13.67% |
| neutral | 0.22 | 0.96 | $830,302.80 | $29,132,233,788.59 | 10.97% | 8.29% |
| aggressive | 0.34 | 1.70 | $3,949,451.79 | $1,998,031,458,543,071.75 | 5.57% | 3.94% |

Clean generalize↔memorize spectrum: conservative (`mean+σ`) generalizes — highest event accuracy, lowest
profit, loosest recognition on noisy units; aggressive (`mean−σ`) memorizes — event accuracy collapses toward
chance while profit explodes (exploiting repeats, not predicting); neutral sits between. Demo 4 verdict: all
three modes keep learning across episodes (Sharpe climbs monotonically in every mode) — no blur-runaway /
"decapitation" here, contrary to first read; base accuracy staying flat while Sharpe climbs is deliberately
blurry base recognition with refinement happening above it, not stalled learning. Accepted as-is.

### Stage 2 default-mode decision — demo 3 (50 symbols, 1 pass, `--spatial`)

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL,AWR,GM,EQIX,RTX,KGC,ALB,AAPL,CVX,HD,WPM,BEP,AREC,JNJ,SLB,PLD,EXK,NVDA,CAT,WFC,RGLD,WEAT,OXY,CEG,LOW,PAAS,MP,LMT,GS,COST,AG,TECK,MRK,INTC,BIP,PSA,DVN,AVAV,PEP,CDE,TSM --context-length 3 --max-positions 3 --transaction-cost 0.02 --columns 20 --spatial --group-mode <conservative default|neutral>
```

| group-mode | result |
|---|---|
| conservative (Stage-2 default at the time) | disastrous — flagged as apparent "learning decapitated" (later reassessed as an artifact of comparing against the union-correction baseline, not a real regression; conservative still climbs Sharpe on demo 4 above) |
| **neutral** | **$2,562,690.23 net profit, +17084.60% ROI, Sharpe 0.42, 28,811 trades** — ≈28%/year reliable annualized return |

Neutral is the clear win on the flagship demo. Combined with it being the most parameter-free form (`k=0`,
no σ lean) of the adaptive threshold, `groupMode` default flipped from `conservative` to `neutral` in
`brain-napi/src/lib.rs` (both the option default and the no-options constructor path).

### Stage 2 default confirmation — MNIST 14×14, static vs conservative vs neutral

```bash
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --episodes 1 --group-mode <mode> [--group-threshold 0.9 for static] --save-brain mnist14
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist14 --disable-learning --test-data
```

| group-mode | test accuracy |
|---|---|
| static (θ=0.9, hand-tuned) | 94.88% |
| conservative (old default) | 94.51% |
| **neutral (new default)** | **95.01%** — beats both |

Neutral wins on MNIST too — the one pure-generalization task in the suite — settling the open question from
the design doc. `neutral` confirmed as the default with no MNIST regression. Committed.

---

## Session 2026-06-28 — resolve-to-base consensus (footprint fan-out) + perf fix

Mechanism change to `infer_neurons` (`brain.rs`): temporal votes toward the coordinate-less spatial apex
(event patterns) were previously **dropped** before consensus — only votes already targeting a base
neuron survived. Replaced with a **resolve-to-base fan-out**: each vote toward an apex event pattern now
contributes to every base neuron in that pattern's footprint (inheriting the vote's strength), so apex
event predictions reach per-dimension consensus instead of being discarded. Base actions and un-chunked
base events resolve to themselves and pass through unchanged. First implementation materialized one
`FlatVote` per footprint constituent (`resolve_to_base_neurons` + `flat_map`); this caused a severe
perf regression (fan-out scales with footprint size, which grows with the spatial hierarchy). Fixed by
folding the fan-out into an in-place `(voter, target, distance) → strength` accumulation
(`for_each_base_neuron` + `resolve_votes_to_base`), bounding cost by distinct triples instead of total
footprint area. Later refactored `infer_neurons` into smaller named helpers (`resolve_votes_to_base`,
`build_frame_votes`) and renamed `build_inferences_by_channel` → `build_inferences` for clarity — no
behavior change.

### Correctness — action policy and classification are byte-identical; event accuracy rises

Every run below reproduced net profit, ROI, Sharpe, and trade count **exactly** against the pre-change
baseline (actions are base neurons — resolution is identity for them). Base **event** accuracy rose in
every case, since previously-dropped apex-event votes now contribute:

| run | recorded (pre-change) | this session (post-change) |
|---|---|---|
| Demo 1 — single-channel synthetic cycle | 233/240 = 97.1% | 233/240 = 97.1% (exact match, unaffected) |
| Demo 2 — multi-channel synthetic cycle | 695/720 = 96.5% | 695/720 = 96.5% (exact match, unaffected) |
| MNIST 7×7, adaptive (default group-mode) | 69.56% (footprint-adjacency baseline) | 69.56% (6956/10000) — exact match; action/classification unaffected |
| Demo 4 ep1 (10 sym, `--spatial`) | $980,593.06, 50.26% acc | $980,593.06 (identical), **51.08%** acc (+0.82pp) |
| Demo 4 ep2 | $31,834,578.74, 46.08% acc | $31,834,578.74 (identical), **47.26%** acc (+1.18pp) |
| Demo 4 ep3 | $7,046,782,891.32, 46.21% acc | identical profit, **47.54%** acc (+1.33pp) |
| Demo 4 ep4 | $36,764,371,968.80, 46.28% acc | identical profit, **47.66%** acc (+1.38pp) |
| Demo 4 ep5 | $932,254,167,812.06, 46.44% acc | identical profit, **47.86%** acc (+1.42pp) |

### Root-cause investigation — why event accuracy still drops episode-over-episode

The resolve-to-base change lifted every episode ~+1–1.4pp but did **not** fix the pre-existing
ep1→ep2 accuracy drop (51.08%→47.26% full run). Traced via a bounded A/B (`--max-frames 3000`,
2 episodes, 10 symbols) isolating the `--spatial` flag:

| 3,000-frame run | ep1 | ep2 | direction |
|---|---|---|---|
| with `--spatial` | 51.02% | 50.23% | **degrades** (−0.79pp) |
| without `--spatial` | 51.66% | 54.19% | **improves** (+2.53pp) |

**Cause:** `pattern_forget_rate = 0` (default) means spatial co-activation corrections are never reaped —
they only accumulate. Each frame, more corrections fire and subsume their constituent base events out of
the temporal substrate (`fired \ subsumed`), shifting predictions from fine-grained per-symbol votes to
coarse multi-symbol co-activation votes. On noisy cross-sectional stock direction data, a joint
multi-symbol configuration recurring is a much weaker predictor than a single symbol's own next move, so
event accuracy falls as more corrections accumulate. Episode 2 starts with episode 1's full correction
population already in place, so the substrate is coarse from frame 1 — hence the step down. Confirmed by
the sign flip: no `--spatial` → no subsumption → substrate stays all-base → second pass genuinely
improves (ep1→ep2 up). Expected interaction with future work: **refinement** should reduce this (a
correction that's reliably conditioned on context stops overriding the base rate it subsumed); **reuse**
does not address it (attacks proliferation/size, not per-frame subsumption). Predicted asymmetry: MNIST
should not show this drop, since spatial patterns are *more* predictive than individual pixels there
(opposite of stocks) — not yet tested this session.

### Performance — naive fan-out vs. folded accumulation (50-symbol demo 3, `--spatial`)

Per-frame time, same neuron-count trajectory, before and after folding the fan-out:

| frame | neurons | naive (materialized `Vec<FlatVote>`) | folded (in-place accumulation) |
|---|---|---|---|
| 50 | ~6.3K | 97 ms | 23.8 ms |
| 500 | ~25K | 466 ms | 74.0 ms |
| 1000 | ~27K | 631 ms | 123 ms |
| 1800 | ~30K | ~673–977 ms (still climbing) | 153 ms (leveling off) |

Naive cost scaled with total footprint area and grew steeply with model size (projected >1hr for the
full 5,373-frame/50-symbol run). Folded cost scales with distinct `(voter, target, distance)` triples and
flattens as the hierarchy matures. Folded version reproduces demo 4 ep1 exactly ($980,593.06 / 51.08%,
25.7s wall-clock for 1 episode / 10 symbols).

### Notes

- MNIST 7×7 was also run once with `--group-mode static --group-threshold 0.9` (the demo's static params)
  and killed partway through (~86% trained) when the user asked for adaptive (default) params instead —
  not a valid data point, superseded by the adaptive 69.56% run above.
- The full 50-symbol demo 3 (all 5,373 frames) was not run to completion under the folded fix within this
  session — the bounded 1,800-frame sample above stands in for it. User indicated they would run the full
  demo 3 separately.

---

## Session 2026-06-29 — spatial hierarchy stall diagnosis: conservative grouping, forget rate, context refinement

Diagnosed why the spatial hierarchy stalls at level 1 under the default `neutral` group-mode, using a
(now-removed) per-level mint-gate funnel instrumented into `mint_spatial_corrections`. Finding: minting is
prolific (L1→L2 mints in the hundreds of thousands) but minted L2 corrections almost never re-fire — a
recognition/recurrence failure, not a minting failure. `group-mode conservative` fixes it by favoring
recognition of established patterns over minting near-duplicates. Confirmed on stocks and MNIST (7×7, 14×14),
including a footprint visualization showing L2 corrections tracing digit strokes (`apps/mnist/jobs/viz-footprints.js`,
`Brain.dumpActiveSpatialCorrections()`). Then tested whether context refinement (re-added behind
`--refine-context`, faithful to the removed temporal version) or forget-rate improves further — result:
refinement adds only overhead/drift with no benefit; the neuron-count plateau is entirely a forget-rate effect.
Both the mint-gate funnel and the refinement feature were rolled back/removed after the finding (see
[[project_conservative_grouping]] / [[project_refinement_findings]] in memory) — not present in current code.
Level/neuron/mint-count live tracking on the MNIST progress line, and the footprint dump/viz, were kept.

### Stocks — group-mode / threshold sweep, `--spatial` (10 symbols unless noted, `--context-length 3 --columns 20`)

| options | net profit | ROI | Sharpe | neurons | mint funnel (L1→L2 minted / L2 re-fires next frame) |
|---|---|---|---|---|---|
| demo 3 baseline, 50 symbols, neutral (default) | +$1,339,491.83 | +8929.95% | 0.33 | 30,674 | L1:213,683 minted → L2 considered only 558 (0.26% recur) |
| 10-symbol baseline, neutral (default) | +$980,593.06 | +6537.29% | 0.26 | 6,090 | L1:45,568 minted → L2 considered 1,480 (3.2% recur) |
| `--forget-rate 0` (immortal patterns) | -$4,756.83 | -31.71% | -0.11 | 160,550 (26×) | L1:43,234 minted → L2 considered 6,204 (14.3% recur) — helps recurrence, explodes substrate |
| `--group-threshold 0.3` (lower static seed) | +$11,296,071.94 | +75307.15% | 0.44 | 6,101 | L1:44,269 minted → L2 considered 829 (1.9% recur) — worse than baseline |
| **`--group-mode conservative`** | +$1,915,456.21 | +12769.71% | 0.28 | **4,984 (fewer)** | L1:12,304 minted (73% fewer) → L2 considered **12,122 (98.5% recur)** |

Conservative is the only setting that unsticks recurrence (3.2%→98.5%) while *reducing* neuron count and
minting — active spatial level shifts from S1-dominated (91% of frames) to S2-dominated (60%) at the same
10-symbol config. `forget-rate 0` and a lower static threshold were tested as alternative hypotheses and both
underperform conservative.

### MNIST 7×7, conservative (θ=0.5), radius 1

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 500/class (train) | 2 | 19,008 | L1:31 L2:10 | — | 60.02 |
| 500/class (test, full 10k) | 2 | 19,012 | L1:36 L2:8 | 64.33 | — |
| 2000/class (train) | 2 | 66,090 | L1:34 L2:15 | — | 63.78 |
| 2000/class (test, full 10k) | 2 | 66,091 | L1:36 L2:17 | 68.12 | — |

L2 footprints visually trace digit strokes (viz-footprints.js) even at this resolution, though thin strokes
(e.g. "1") mostly show covered-non-ink cells rather than covered ink.

### MNIST 14×14, conservative (θ=0.5), radius 1

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 1000/class (train) | 2 | 128,402 | L1:104 L2:24 | — | 84.38 |
| 1000/class (test, 3000 images) | 2 | 128,404 | L1:99 L2:43 | 87.33 | — |
| demo config (θ=0.9 static seed misapplied to conservative, killed @10.1%) | — | — | — | — | 82.5 (partial) |
| full 54,210, forget 0 (baseline, killed @40% to free build lock) | 2 | ~262k@40%, trending ~650k | L1/L2 growing linearly | — | ~87 (partial) |
| full 54,210, `--refine-context` only, forget 0 (killed @8.5%) | 2 | 80,510 | L1:107 L2:52 | — | 81.7 (partial) |
| full 54,210, `--refine-context --forget-rate 0.001` (killed @47.2%) | 2 | 31,231 | L1:105 L2:98 | — | 85.4 (partial) |
| **full 54,210, `--forget-rate 0.001`, no refine (completed)** | 2 | **31,675** | L1:98 L2:17 | — | **86.73** |

Viz on the 1000/class brain: L2 footprints decompose digits by stroke — e.g. a 7's top bar and diagonal get
separate L2 patterns; a 0's ring is traced by overlapping L2s along the loop.

Refinement-alone (forget 0) roughly matches refine+forget's neuron count at 2× the wall-clock and no
depth/accuracy gain — refinement's identity-drift churn is masked, not fixed, by forget rate. Forget-rate alone
achieves the same ~32K plateau as refine+forget, at 2× the speed (15 vs 8 img/s) and less cumulative minting —
so the plateau is a forget-rate effect, not a refinement effect. Depth stayed at 2 in every 14×14 variant tried
this session; none beat the recorded static-θ=0.9 demo baseline (94.9% test, 32K neurons, ~5 min).

### Note on the `θ=0.9` mistake

Two runs above used `--group-threshold 0.9` with `--group-mode conservative` (the demo's static-mode seed,
copied by habit). Seeding an *adaptive* mode at a *static-selective* value is self-contradictory and was
corrected to the default 0.5 (conservative's own baseline, untested prior to this session) — the θ=0.9 runs
are recorded above only as the "killed, wrong config" data point, not a valid conservative reference.

---

## Session 2026-07-04/05 — information stack (match-info2 + error-info / error-info2)

Recognition = likelihood ratio vs the neuron's background model (`--match-info2`); creation = accumulated
surprisal vs description cost (`--error-info`), or identity-priced description cost (`--error-info2`, the
parameter-free apex-tower fix). Common flags unless noted:
`--group-mode neutral --group-threshold 0.5 --refine connection --mint-min-samples 0 --match-info2`.

### 28×28 radius ladder (`--error-info`)

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| r1 | 4 | 25,539 | L1:23701 L2:295 L3:70 L4:5 | 93.42 | 88.19 |
| r2 | 23 | 64,051 | L1:62415 L2:34 L3:33 L4:15 … L18:2 … L23:1 (apex tower) | 95.66 | 93.41 |
| r3 | 9 | 104,271 | L1:102694 L2:24 L3:31 L4:20 L5:10 L6:5 L7:5 L8:5 L9:5 | **96.37** | 95.08 |
| r4 (eval abandoned: ~24h at 4 img/s; ~95.6 running at 52%) | 18 | 185,527 | L1:183605 L2:22 … L18:303 (apex tower) | — | 95.76 |

r3 beats the old single-episode record (96.26 at 2.16M neurons, legacy r2) with 20× fewer neurons and sits
0.07pp under the all-time 96.44 (which needed a second episode).
Accuracy rides the L1 vocabulary, not the hierarchy; the depth-23/18 shapes at r2/r4 are the apex-tower artifact.
Shuffled-order frozen eval of the r2 brain reproduced 9566/10000 and the confusion matrix exactly — the
rising in-pass accuracy of ordered evals is test-set ordering, not learning leakage (`--shuffle` flag added).

### 14×14 full data

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| error-info (reference) | 4 | 7,985 | L1:7372 L2:224 L3:20 L4:1 | 93.52 | 90.49 |
| error-info, episode 2 (load) | 4 | 8,147 | L1:7453 L2:275 L3:43 L4:8 | 93.62 | — |
| error-info, `--no-balance` | 4 | 8,053 | L1:7412 L2:240 L3:27 L4:6 | 91.97 | 90.86 |
| error-info, `--refine none` | 4 | 7,900 | L1:7291 L2:223 L3:17 L4:1 | 93.49 | 90.57 |
| error-info, `--refine context` | 2 | 21,909 | L1:21511 L2:30 | 94.87 | 92.73 |
| error-info, `--refine both` | 2 | 21,861 | L1:21467 L2:26 | 94.84 | 92.68 |
| error-info2 | 3 | 4,415 | L1:3986 L2:48 L3:13 | 92.42 | 88.17 |

Refinement ablation verdict: connection refinement is a no-op under the information stack; context
refinement buys +1.35pp only through a 2.7× larger L1 vocabulary and collapses consolidation (L2 224→30).
The no-balance loss (−1.55pp, damage ordered by class frequency) is a decode-side prior tilt; the substrate
is imbalance-blind (same size and shape).
error-info2 halves the substrate and removes the tower at a −1.1pp accuracy cost.

### Split-MNIST 14×14 (class-incremental, 5 tasks × 2 classes, no task IDs)

| options | avg acc after T4 | avg forgetting | neurons | structure |
|---|---|---|---|---|
| error-info | 92.58% | 4.35pp | 8,990 | depth 18 apex tower grew during task 0 |
| error-info2 | 88.02% | 9.18pp | 5,005 | depth 4, all levels genuine, no tower |

No catastrophic forgetting in either (naive backprop floor ≈ 20% average).
Joint training on the same data scores 93.52, so sequential learning costs ~1pp under error-info.
error-info2 kills the tower but retention worsens: the redundant voters it refuses to mint were ballast
against reward erosion (task 0 drops 99.9→83.0 at task 1, vs 100→98 under error-info).

### 7×7 determinism reference cells (per-class 200)

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| error-info | 2 | 1,057 | L1:966 L2:4 | 66.40 | 61.35 |
| error-info2 | 1 | 630 | L1:543 | — | 57.75 |

These two cells re-ran byte-identically after every refactor this session (constructor-option migration,
learning-state threading, recognition split, timing moves).

### Stocks demos (info stack vs legacy, both `--spatial`)

| demo | legacy spatial | info stack |
|---|---|---|
| 3 — trading, 50 symbols, 1 pass | +$3.18M, Sharpe 0.44, base acc 6.1%, S-depth 17 | +$219K, Sharpe 0.17, base acc 49.1%, S-depth 1–2 |
| 4 — action learning, 5 episodes | Sharpe 0.25→0.71→0.57, base acc ~5% | Sharpe 0.33→0.84→0.80, base acc ~34% |
| 5 — sequence memorization | Sharpe 2.00 | Sharpe 2.01 |

Demo 3 is the diagnostic loss: legacy out-earns 14× on a single pass despite near-random prediction, by
spraying redundant spatial features the reward layer exploits — reward-channel information is missing from
the creation accounting.
Side-finding: `--spatial` itself collapses demo 5 base accuracy from 66% (no-spatial reference) to ~10%.

---

## Session 2026-07-05 (spec) — wavefront migration benchmark baselines

A spec session (defined the wavefront migration plan in [wavefront.md](./wavefront.md) and
[wavefront-implementation.md](./wavefront-implementation.md)); no new experiments were run. The values below
are the **pre-migration baselines** the migration is gated against — each stage must hold MNIST ~70% and stocks
similar-or-better. Recorded here so the migration has a fixed reference.

### MNIST 7×7 — primary per-stage gate (~4 min)

```bash
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --episodes 1 --save-brain mnist7
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist7 --disable-learning --test-data
```

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 7×7 r1 (pre-migration baseline) | — | — | — | 71.02 | — |

Only test accuracy was reported this session (71.02%); depth / neurons / train not captured. Comparable 7×7 r1
cell from the prior session for reference: depth 3, 4,236 neurons, L1:2789 L2:1342 L3:9, test 70.90 / train 66.82.

### Stocks demos (both `--spatial`) — cross-domain confirmation

Documented baselines cited from [stock-demos.md](./stock-demos.md); **not re-run this session**.

| demo | command flags (after `apps/stocks/jobs/test.js --symbols …`) | baseline |
|---|---|---|
| 3 — trading, 50 symbols, 1 pass | `--context-length 3 --max-positions 3 --transaction-cost 0.02 --columns 20 --spatial` | +$2.56M net profit, +17084% ROI, Sharpe 0.42 |
| 4 — action learning, 10 symbols, 5 episodes | `--context-length 3 --columns 20 --no-summary --episodes 5 --spatial` | ROI climbs each episode; Sharpe 0.22→0.49→0.61→0.75→0.96 |

Note: last session recorded a different demo-3 figure on its branch (legacy spatial +$3.18M / Sharpe 0.44 — see
below), so treat the +$2.56M as the documented reference, not a bit-exact target. The wavefront gate is
similar-or-better, not identity.

---

## Session 2026-07-05 — consolidation → pooled-failure derivation

Mechanism arc this session: consolidated creation+recognition (neurons self-evaluate and
request corrections; thalamus prices against the paid codebook and allocates) → deleted the
`pending_mints` ledger and FIFO-16 cap → **pooled-failure derivation** (every failure pools
its surprisal + context-neighbor counts per parent; a correction mints when pooled evidence
covers the price of the *modal* failure context). Two efficiency add-ons tried and reverted:
divergence filter, coverage attribution.

All runs below: `--error-info --match-info`. "r" = base radius.

### Pure pooled-derivation (mechanism kept)

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 7×7 r1 | 3 | 4,236 | L1:2789 L2:1342 L3:9 | 70.90 | 66.82 |
| 14×14 r1 | 3 | 25,934 | L1:15714 L2:9835 L3:17 | 94.67 | 92.12 |
| 14×14 r1 ep2 (load mnist14pooled) | 3 | 32,369 | L1:18126 L2:13824 L3:51 | 94.82 | 94.73 |
| 14×14 r2 | 3 | 175,880 | L1:67508 L2:107831 L3:173 | 95.11 | 92.96 |

### Attributed pooling (efficiency attempt — reverted)

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 14×14 r1 | 3 | 21,002 | L1:12603 L2:8011 L3:20 | 94.48 | 91.18 |
| 14×14 r1 ep2 (load) | 3 | 26,165 | L1:14502 L2:11246 L3:49 | 94.73 | 94.32 |
| 14×14 r2 | 3 | 32,767 | L1:11047 L2:20471 L3:881 | 91.57 | 86.38 |
| 28×28 r1 | 3 | 63,296 | L1:34594 L2:27136 L3:98 | 94.35 | 89.09 |
| 28×28 r2 (killed at 81%) | 5 | 77,961+ | L1:21872 L2:52240 L3:2374 L4:10 L5:2 | — | 79.9 (partial) |

### Other runs this session (no held-out test pass saved)

| options | depth | neurons | level distribution | test | train |
|---|---|---|---|---|---|
| 7×7 r1, paid-vocab pricing (pre-pooled) | 3 | 19,791 | — | — | 66.92 |
| 14×14 r2, forget-rate 0.01 | 2 | 1,152 | L1:978 L2:78 | — | 65.02 |

**Reference points (from prior sessions, to beat):** 14×14 held-out 95.32% (MDL-v2 recognition);
28×28 95.64%/132K (mint-repeat r2); main-branch channel-neighbor baseline 96.44%.
Efficiency remains the open problem: neuron count scales positions × features.

---

## Session 2026-07-05/06 — likelihood-ratio pattern creation, local naming-cost pricing, backup bug fix

Ported the Jaccard/group-threshold-based pending-mint ledger (from the prior session's consolidation work) to
one that matches and mints purely on likelihood ratios, removing the ledger's remaining dependency on
`groupMode`/`groupThreshold`. Then iterated on the naming-cost pricing basis to recover accuracy lost to the
switch, and found/fixed a real backup/restore bug along the way. All runs `--match-info --error-info` (the
`match-info2`/`error-info2` names from prior sessions are now aliases of the same flags). "r" = base radius.

### Baseline — Jaccard-matched ledger, group-threshold-derived merge bar (start of session, committed)

| options | depth | neurons | test | train |
|---|---|---|---|---|
| 7×7 r1 | 3 | 2,483 | 70.85 | — |
| 14×14 r1 | 3 | 5,156 | 93.13 | — |
| 28×28 r1 | 3 | 13,949 | 92.95 | — |
| 28×28 r2 | 3 | 18,459 | 94.32 | — |
| 28×28 r3 (killed @47.8%) | 3 | 17,534+ | — | — (partial) |

### Failed experiment — MDL two-part-code merge test (reverted)

Replaced the Jaccard/group-threshold merge test with "merge into whichever ledger entry makes the *union*
cheaper to name than naming separately" (pure `name_bits`, no group-threshold dependency). Reverted: the
naming-cost function is strongly subadditive for small sets against a large vocabulary, so "union is cheaper"
came back true almost unconditionally — collapsing the ledger into a few catch-all buckets, the same failure
mode as the pre-ledger pooled-derivation bug.

| options | depth | neurons | test |
|---|---|---|---|
| 14×14 r1 | 4 | 20,352 | 94.18 |

### Likelihood-ratio ledger — exact-context match + likelihood-ratio scoring, global codebook price (ALPHA=0.9)

Pending-mint entries now track their own per-neighbor present/absent counts (an embryonic routing entry) and
merge by the same no-threshold `rank > 0` rule recognition uses, scored against background — not Jaccard.
`compute_spatial_evidence` also switched from plain self-information (`-log2(p)`) to a likelihood ratio against
a fresh one-occurrence hypothesis, so a mismatch on an already-common event can argue *against* minting, not
just for it.

| options | depth | neurons | test | train |
|---|---|---|---|---|
| 7×7 r1 | 3 | 1,187 | 69.78 | 67.11 (frozen) |
| 14×14 r1 | 3 | 4,786 | 92.85 | 92.12 (frozen) |
| 14×14 r1, 2 episodes | 4 | 5,865 | 92.90 | 91.77 |
| 28×28 r1 | 3 | 13,028 | 92.85 | 86.05 |
| 28×28 r2 | 3 | 9,341 | 92.79 | 85.61 |
| 28×28 r1, 4 buckets | 2 | 21,647 | 92.25 | 83.83 |

Roughly matches the Jaccard baseline's accuracy at ~50–90% of its neuron count, but 28×28 r2 is a real
regression from Jaccard's 94.32%. 4-buckets backfires again (more neurons, worse accuracy, shallower — matches
the historical "grayscale backfires at low res" finding). 2-episode run: nearly all the train-accuracy gain in
episode 2 was capacity growth with almost no test-accuracy gain (92.85→92.90, +0.05pp) — later shown to be a
real, reproducible pattern (see the corrected 5-episode 7×7 run below).

### Naming-cost pricing experiments (28×28 r2), chasing the historical 96.44% ceiling

The naming price was `name_bits(context) + name_bits(target)`, both priced against the whole brain's neuron
count — a category error, since context is drawn from the parent's own level/neighborhood and target from a
fixed base-sensory population, not the whole codebook. Tried, in order:

| approach | result | verdict |
|---|---|---|
| price context against this neuron's own children count | runaway — 158,612 neurons / depth 8 by 800 images | reverted, killed early |
| price against `sqrt(global codebook size)` | 14,283 neurons, depth 5, 93.62% test, 89.11% train | worked, but an arbitrary dampening exponent |
| price context against parent's own **level** population (brain-wide, `spatial_level_paid_counts()`) | 10,531 neurons, depth 3, 86.47% train (test not run) | modest gain, needed new cross-file plumbing |
| price context/target against **this neuron's own** tracked-neighbor counts (`spatial_context_counts.len()` / `spatial_inference_counts.len()`, already-persisted, zero new plumbing), ALPHA=0.9 | 18,412 neurons, depth 15, **94.14%** test, 90.06% train | adopted |
| same, **ALPHA=1.0** (plain Laplace, replaces an unexplained 0.9 tuning constant used in `compute_spatial_evidence`) | 28,800 neurons, depth 12, **94.74%** test, 90.97% train | **best result this session** |

The per-neuron-children attempt cascades: every new neuron starts with zero children, so it and its own
children are all cheap to mint in a chain with no global brake — exactly the "alias chains stack into towers"
failure the original global-codebook pricing existed to prevent. The winning design instead reuses
`spatial_context_counts`/`spatial_inference_counts` — per-neuron maps that already exist for the recognition
background model, growing from empty at birth — as the naming population, giving each neuron a naturally
adaptive, non-arbitrary, already-persisted pool with no new cross-file plumbing.

`ALPHA` (Laplace-style optimism, pre-existing at 0.9 in recognition's cold-start) was reused for the new
evidence/ledger math. At `ALPHA=1.0`, `rank_by_likelihood_ratio` and the ledger's `score_entry` stay
well-behaved (their `p_c` has a real `count` term), but `compute_spatial_evidence`'s formula degenerates
exactly to the old plain self-information surprisal (`log2(1.0/p) = -log2(p)`), losing the "a mismatch on a
common event can argue against minting" property — flagged as an open, not-yet-resolved tension.

### 5-episode 7×7, corrected (ALPHA=1.0, local pricing) — the accuracy-vs-neurons finding, now trustworthy

| episode | train | test | neurons |
|---|---|---|---|
| 1 | 66.91 | 71.05 | 3,082 |
| 2 | 68.52 | 71.00 | 3,323 |
| 3 | 68.70 | 70.97 | 3,612 |
| 4 | 68.73 | 70.94 | 3,946 |
| 5 | 68.78 | 71.10 | 4,333 |

Test accuracy is flat (71.0–71.1%, no trend) across all 5 episodes while train accuracy climbs and neuron count
grows +40% (3,082→4,333) — almost the entire train-accuracy gain lands in episode 2; episodes 3–5 add ~1,000
neurons for essentially no test-accuracy gain. Directly motivates a forgetting/rent mechanism (discussed,
not yet implemented): patterns should pay ongoing description-length rent and earn credit per accepted fire,
so low-value repeat-episode growth gets reaped instead of accumulating.

### Bug found and fixed — pending-mint ledger not persisted across save/restore

`pending_spatial_mints` (the per-neuron ledger of in-progress, not-yet-paid correction candidates) was never
part of `SerializedNeuron`/`backup.rs` — silently dropped on save, empty on load. Continuing training via
save→load→continue therefore lost all partially-accumulated evidence at each checkpoint, needing more total
occurrences to mint the same patterns than one uninterrupted run — a real, growing divergence, not noise:

| episode | continuous run | incremental, pre-fix | incremental, post-fix |
|---|---|---|---|
| 2 | 3,323 | 3,200 | 3,323 |
| 3 | 3,612 | 3,305 | 3,612 |
| 4 | 3,946 | 3,410 | 3,946 |
| 5 | 4,333 | 3,516 | 4,333 |

Fixed by adding `SerializedPendingMint` (context set, present/absent counts, occurrences, evidence) to
`SerializedNeuron`, and three new CSV tables (`pending_mint_meta.csv`, `pending_mint_context.csv`,
`pending_mint_counts.csv`) to `backup.rs`, keyed by `(neuron_id, ledger_entry_index)` since an entry holds a
set and two counted maps. Also fixed a dormant off-by-one in `patterns.csv` parsing (the `evidence` column's
bounds check required 6 columns instead of 5) — never triggered in practice since the writer always emits all
6, but inconsistent with the intended back-compat fallback. Post-fix, the incremental and continuous 5-episode
runs match neuron-for-neuron at every checkpoint (table above).

### Other cleanup this session

- Renamed newly-introduced pricing parameters away from "vocabulary"/"codebook" language (not used elsewhere
  in the signatures this session touched) to plain neuron-count names, then simplified further by computing
  them from existing per-neuron state instead of threading them as parameters at all.
- Removed a `.min(0.999)` clamp copied into the ledger's `score_entry` out of habit — it guards against
  `log2(0)` only where `p_c` is explicitly flipped via `(1 - p_c)` (recognition's absent-entries branch); the
  ledger's present/absent counts are tracked separately with no such flip, so the clamp did nothing there.
  Added an explanatory comment at the one place (recognition) where it's actually load-bearing.

## Session 2026-07-10 — womb conversion (now part of algorithm.md, the UCAR design)

The exact-context pending-mint ledger and the born-unpaid pattern machinery were replaced by the womb: each
parent neuron holds a small set of context-only embryos (per-neighbor count + occurrence total `N` + accumulated
benefit), a failure is served by the best embryo scoring positive under the same likelihood-ratio math
recognition uses (fuzzy assignment, not exact-context match), and a pattern is born only when an embryo's
evidence covers `price = |context| + 1`. Every born pattern is paid — there are no unpaid entries in the routing
table. Embryo eviction reuses `pattern_forget_rate`: effective evidence = deposited − decay, evicted at ≤ 0,
which unifies incoherence (negative-benefit deposits) and staleness (never re-served) into one rule. Recognition
and the Layer 1 net-benefit gate carry over unchanged. This is the `--error-info` path; recognition is
`--match-info`. Temporal (d>0) is untouched.

Held-out test, full balanced set (54,210 imgs), 1 episode, `--error-info --match-info`, forget rate 0 (so
embryos die by incoherence only, no staleness eviction):

| Config | Womb test | Legacy static-θ0.9 baseline | Womb neurons | Legacy neurons | Depth |
|---|---|---|---|---|---|
| 14×14 r1 | **94.52%** (9452/10000) | 94.88% | 18.1K | 32K | 5 |
| 28×28 r1 | **94.21%** (9421/10000) | 94.93% | 50.1K | 112K | 4 |
| 28×28 r2 | **94.26%** (9426/10000) | 96.44% | 131K | 2.16M | 7 |

**Finding:** the womb lands within ~0.4–0.7pp of the legacy averaged-threshold baseline while minting roughly
half the neurons (18K vs 32K; 50K vs 112K), and self-organizes a deeper hierarchy (L4–L7, no caps). Fuzzy
embryo pooling consolidates aggressively — fewer, better-earned patterns at a small accuracy cost at radius 1.
At 28×28 **radius 2 buys essentially nothing** (94.21%→94.26%, +0.05pp) whereas the legacy path gains +1.5pp
(94.93→96.44) — so the womb is not exploiting the larger receptive field, consistent with the aggressive
consolidation into fewer, broader patterns.

**Perf note (eval is slow at r2):** frozen eval of the r2 brain runs at ~1 img/s. Profiling the saved brains
(`apps/mnist/jobs/profile_recog.js`) showed the r2 cost is **spatial** recognition — ~15.6K candidate scorings
per frame at r1, exploding at r2 because radius-2 widens each pattern's stored context (13.5→63 entries) and
forget-0 keeps all 131K patterns as permanent candidates. Separately, the **temporal sweep is O(active²)**:
`dispatch_temporal_frame` rebuilds a per-neuron channel-filtered actives list by scanning the full active set
for every active neuron (784²≈614K clone-filters/frame at 28×28), run unconditionally even with zero temporal
patterns. It is a fixed ~14.5ms sidecar (dominates the fast r1/r14 frames, negligible next to r2's spatial cost).
The active set it runs over is the spatial apex, whose size is **structurally pinned to the sensory-grid size**
(784 at 28×28, 196 at 14×14): subsumption is strictly 1:1 — each firing pattern swaps out exactly its single
parent host (pixel→L1, L1→L2, …), one in one out — so the apex can never drop below the grid size, it only
trades pixels for patterns. Measured apex at 28×28 r1: 608 raw pixels + 176 fired patterns = 784 (14×14: 127 +
69 = 196). So the O(N²) always runs over N≈grid-size; it cannot be shrunk by "compressing the apex" because the
apex doesn't compress by design. (Earlier note that `suppressed=0` meant subsumption was broken was wrong —
`suppressed` is a temporal field, trivially 0 with no temporal patterns; spatial subsumption works, it's just 1:1.)

Backup format changed with the mechanism: `pending_mint_{meta,context,counts}.csv` (three tables) → `embryo_{meta,context}.csv` (two), and `patterns.csv` dropped its evidence/price columns (reader still tolerates the old wide rows).