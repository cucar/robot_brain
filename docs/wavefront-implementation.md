# Wave-Front Implementation & Migration Plan

The plan to move the brain from the current architecture to the wave-front ([wavefront.md](./wavefront.md))
incrementally — each step a small in-place edit, verified against a fixed benchmark before the next. This doc is
the *ordering and verification* layer; the architecture it implements is [wavefront.md](./wavefront.md).

---

## 1. Verification approach

- **Characterized regression against fixed benchmarks (§2), not bit-exactness.** The wave-front keeps the
  existing minting mechanism — one correction per erroring parent — and changes only the neighborhood primitive
  (footprints) and the bookkeeping (no stored levels, coordinate-less corrections). So the benchmark numbers
  should stay **≈ current**: minor movement is expected, collapse is not. For stocks, equal-or-better.
- **In-place edits, no runtime flag.** The old path lives in git history; the reference is the current
  pre-migration benchmark numbers (§2), recorded before the first edit.
- **Spatial first, then temporal.** Footprint adjacency at L1+ is the one real behavioral change (see
  [wavefront.md](./wavefront.md), Footprints); the temporal wave is structurally unchanged, so it should track
  the benchmark with no movement. Apply each substrate change on the spatial side, confirm the benchmark, then
  the temporal side.
- **Each stage is a separate, revertible commit**; localize any divergence before moving on.

---

## 2. Benchmarks (the gate)

A step is "verified" when the benchmark below holds. The MNIST 7×7 run is fast (~4 min) and is the **per-stage
gate**; the stocks demos are the **cross-domain confirmation**, run once the migration clears MNIST.

### MNIST 7×7 — the primary gate (must clear first, and the most important)

Train + save, then load and evaluate frozen on the test set:

```bash
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --episodes 1 --save-brain mnist7
node apps/mnist/jobs/test.js --image-size 7 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist7 --disable-learning --test-data
```

**Expected: ~70% test accuracy** (currently **71.02%**). The gate is "stays around 70%, no collapse," not an
exact match — minor changes are expected. ~4 minutes end to end, so this is the run to lean on for fast
per-stage iteration.

### Stocks — cross-domain confirmation (after MNIST clears)

Both exercise the spatial wave (`--spatial`); each should return **similar results or better**.

**Demo 3 — historical trading** ([stock-demos.md](./stock-demos.md), "Stock Trading"):

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL,AWR,GM,EQIX,RTX,KGC,ALB,AAPL,CVX,HD,WPM,BEP,AREC,JNJ,SLB,PLD,EXK,NVDA,CAT,WFC,RGLD,WEAT,OXY,CEG,LOW,PAAS,MP,LMT,GS,COST,AG,TECK,MRK,INTC,BIP,PSA,DVN,AVAV,PEP,CDE,TSM --context-length 3 --max-positions 3 --transaction-cost 0.02 --columns 20 --spatial
```

Expected ≈ **+$2.56M net profit (+17084% ROI, Sharpe 0.42)**.

**Demo 4 — action learning over episodes** ([stock-demos.md](./stock-demos.md), "Action Learning in Low
Accuracy"):

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --context-length 3 --columns 20 --no-summary --episodes 5 --spatial
```

Expected: per-episode ROI climbing across the 5 episodes (Sharpe rising to ~0.96 by episode 5).

### Diagnostics for localizing a divergence

`test.js` also emits structural diagnostics — use them to *localize* a divergence, not as the pass/fail: depth
(max spatial level), per-level pattern counts (L1, L2, …), active / cumulative corrections, and total neuron
count. Hold the MNIST 7×7 config fixed across all stages so these are comparable step to step.

---

## 3. Migration stages

> **Scope.** This migration moves the **substrate**: footprints, coordinate-less corrections, and the removal of
> stored levels ([wavefront.md](./wavefront.md)). It does **not** change what gets minted — corrections stay
> one-per-parent. The four substrate changes below are the whole of it.

The migration is **four stages**, each behind its own gate (the §2 benchmark). Every stage is an in-place edit,
verified on the **spatial** side first, then transliterated to the **temporal** side. The dependency order is
deliberate: introduce the replacement primitive, make it load-bearing, *then* delete the two stored fields it
makes redundant.

### Stage 1 — Footprints, additive (behavior-neutral)

Add a `footprint` bitset to every neuron: base = `{self}`; correction = `⋃ constituents` (the context set) at
mint. Precompute the base neighbor-ring from the encoder's declared neighborhoods (the `radius` window). Add the
footprint-adjacency test (dilate-and-AND). **Do not wire it into filtering yet** — channel-neighbor still drives
all behavior. Pure addition.

*Gate:* MNIST 7×7 **unchanged** (~71% — nothing consumes footprints yet); footprint-adjacency unit passes
(footprint = union; adjacency = base-graph reachability); at L0, footprint adjacency **equals** the
channel-neighbor window on a fixed run (the equivalence Stage 2 relies on).

### Stage 2 — Switch neighborhood to footprint adjacency

Replace every `is_spatial_neighbor_channel` / `is_temporal_neighbor_channel` call site with footprint adjacency;
delete the channel-neighbor machinery (`*_channel_neighbors`, `is_*_neighbor_channel`, `set_*_neighbors`). This is
the **first real behavioral change**.

*Gate:* MNIST 7×7 **stays ~70%** — L0 grouping is near-equivalent to the channel-neighbor window, while L1+
receptive fields grow by union rather than single-pixel anchor ([wavefront.md](./wavefront.md), Footprints), so
expect minor movement. Characterize the delta; confirm no collapse. If it moves more than expected, run stocks
demo 3 here to check the spatial change cross-domain before continuing.

### Stage 3 — Coordinate-less corrections

Stop inheriting the parent coordinate at mint; corrections carry only `id` + `footprint`. Drop the pattern's
`base_neurons` insert and the restore parent-walk that rederived correction coordinates. Confirm consensus never
reads a correction coordinate (corrections are forbidden vote targets, so they never reach per-dimension
grouping). Backup: drop the correction coordinate; format-version bump.

*Gate:* coordinate-less-corrections unit (corrections have no coordinate; the `aggregate_votes`
"no-coordinate ⇒ never a vote target" panic never fires); MNIST 7×7 ≈ Stage 2 (the coordinate's only behavioral
use, neighbor filtering, was already retired in Stage 2 — this should be near-neutral); save/restore round-trips
to identical numbers.

### Stage 4 — Remove stored levels (the wave is now pure)

Delete `neuron_spatial_levels` / `neuron_temporal_levels`. Repoint the four readers: mint reads the parent's
depth from the level loop variable; diagnostics recompute from the activation index; serialization drops the
level columns; `skip_action_neuron`'s base test becomes an explicit base-neuron predicate. The climb is now a
settling wave driven solely by the loop variable. Backup: drop both level columns; rebuild footprints on load in
dependency order (memoized recursion from base / topo-sort the constituent graph); format-version bump.

*Gate:* wave-fixpoint unit (a stage settles deterministically to the same hierarchy regardless of mint order
within a round); MNIST 7×7 ≈ Stage 3; save/restore round-trips identically with footprints rebuilt, not
serialized. **Then run the full cross-domain confirmation: stocks demo 3 + demo 4 — similar results or better.**

### Standing rule

- **Keep every stage a separate, revertible commit**; localize divergences on the fast MNIST 7×7 run before
  reaching for the slower stocks demos.

---

## 4. Rust-port notes

Representation choices for the port:

- **Footprints are bitsets** over base sensory neurons; adjacency = dilate one footprint by the precomputed
  base neighbor-ring and AND with the other; nonzero ⇒ touch ([wavefront.md](./wavefront.md)).
- **Sets as sorted `Vec<u32>`** of indices, not delimited strings (footprint constituents, context sets).
- **Deterministic ordering** (sorted keys) everywhere, for reproducible diagnostics step to step.
- **Neighbor adjacency is footprint *touch* at every level** — base-graph overlap-or-adjacency, *not* parent
  membership. Graded locality (footprints span more as you climb) is the intended behavior: it is what lets
  disjoint-but-abutting features become neighbors.

---

## 5. Risks

- **No live dual-run** (in-place choice): a divergence can only be compared against the recorded pre-migration
  benchmark numbers (§2), not a side-by-side run. Mitigation: keep every step a separate revertible commit; the
  fast MNIST 7×7 run localizes quickly.
- **Characterized regression can hide a real bug** inside the tolerance band. Mitigation: the unit gates
  (footprint adjacency, wave fixpoint, coordinate-less) catch what the aggregate accuracy misses.
- **Big steps** are the likeliest place to lose the thread; decompose on first sign of an unexplained
  divergence.
