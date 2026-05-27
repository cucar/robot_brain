# MNIST Performance Optimization Plan

The algorithm appears to work but is extremely slow. This document tracks
the disciplined plan for understanding *why* it's slow and *what to try*.
It is a living document — update findings as experiments complete.

## Current state (baseline)

`process_levels` is 95%+ of every frame. Cost grows superlinearly with brain
size (~7k neurons → 11× slower than ~800 neurons). Inside `process_levels`:

| Section | Time | % |
|---|---|---|
| Neuron ops (actual brain math) | 3661ms | 44% |
| `get_level_tasks` (orchestration) | 1769ms | 21% |
| `collect_votes` (orchestration) | 1146ms | 14% |
| Memory shuffling | 907ms | 11% |
| Unaccounted | ~650ms | 8% |

Inside neuron ops, **`recognize_patterns`** dominates (41% training, 71% cold
replay). The inverted index returns ~50% of all patterns as candidates per
query — see [`get_pattern_candidates_at_age`](../../brain/brain-core/src/neuron.rs):
candidate = any child pattern that shares **one** exact `(neuron_id, distance)`
pair with the observed context. With binary pixels and only ~784 possible
sensory neurons, this criterion is far too permissive.

Orchestration (~3.8s) ≈ actual neuron work (~3.7s). Half the time is
bookkeeping around the brain, not the brain itself.

---

## Experiments to run

Experiments come before code changes. Each one should produce a number we
can extrapolate from. Run them in roughly this order — cheap-and-informative
first.

### E1. Lower input resolution (14×14, then 7×7), binary

**Why:** We don't yet know whether the algorithm *converges* at MNIST scale
or just runs forever. Shrinking the input lets us see end-to-end behavior in
minutes instead of hours and gives us a scaling curve to extrapolate from.

**What to measure:**
- Wall-clock per image / per epoch at 7×7, 14×14, 28×28.
- Final neuron count at each resolution.
- Whether accuracy stabilizes (does it actually learn?).
- Shape of the scaling curve — is `process_levels` quadratic, cubic, worse?

**Decision it unblocks:** Whether 28×28 is even the right target right now,
or whether we should optimize at a tractable scale first and scale up only
after the code is fast.

### E2. Higher input resolution (256 values per pixel), binary task replaced with greyscale

**Why:** Tests the hypothesis that the index is slow because the sensory
vocabulary is too small. With 256 values per pixel, `(neuron_id, distance)`
pairs become rare → index buckets get thin → `recognize_patterns` should
speed up dramatically. But: higher-level patterns may stop forming because
nothing recurs across images. Trades generalization for speed and
memorization.

**What to measure:**
- Candidates per `recognize_patterns` call (already instrumented as
  `recognize_candidates_evaluated`).
- Total neuron count, broken down by level (L0 should balloon, L1+ should
  shrink).
- Accuracy on train vs held-out (expect train ↑, held-out ↓).
- Wall-clock per frame.

**Decision it unblocks:** How much of the perf problem is "index criterion
too loose" vs "vocabulary too small." If 256-value is dramatically faster,
the index criterion (E5) is the right structural fix. If it's roughly the
same, the cost is elsewhere (E6/E7).

### E3. Sweep `context_length` and `merge_threshold`

**Why:** Both parameters directly affect candidate-set size and pattern
formation rate, but we haven't measured their effect on perf — only on
behavior. A tighter `merge_threshold` may reduce pattern count
substantially; a shorter `context_length` reduces the number of
`(neuron, distance)` entries the index has to union over.

**What to measure:** wall-clock, neuron count, candidates evaluated, accuracy.

### E5. Sweep dynamics parameters

**Why:** Several parameters directly shape how many patterns get created,
how long they live, and how aggressively the brain reacts to error. They
all feed back into brain size and candidate-set width, but we don't have
measurements of their isolated perf effect.

**Parameters to sweep (one at a time):**
- **Number of frames per image (static vs dynamic).** Currently fixed. Try
  fewer frames (does the brain still converge?), more frames (does cost
  scale linearly with frames or worse?), and a dynamic policy that stops
  when activations stabilize.
- **Forget rate.** Higher forget rate → smaller live brain at any moment →
  fewer candidates in the index. Find the rate that keeps accuracy steady
  while shrinking the active pattern pool.
- **Level decay mode.** Different decay modes change which patterns stay
  "alive" enough to be scored (see the `get_child_effective_activation_strength > 0`
  filter in [neuron.rs:934](../../brain/brain-core/src/neuron.rs)). Mode choice
  directly affects candidate-eval cost.
- **Error mode.** Different error-detection modes produce different
  numbers of error patterns per frame, which feed back into pattern
  creation rate.
- **Error threshold (if static).** If currently fixed, sweep it. If
  dynamic, characterize what it adapts to and whether a static value
  performs comparably.

**What to measure (per parameter):** wall-clock per frame, neuron count,
candidates evaluated, accuracy. Look for the knee in the curve — the point
where loosening the parameter stops helping accuracy but keeps costing
perf.

**Decision it unblocks:** Which parameters are perf knobs we can tune
without changing code, vs which are structural and need code work to fix.

### E4. Cold replay vs training cost decomposition

**Why:** Cold replay is as slow as training (~3s/image either way) and
`recognize_patterns` is 71% of it. Learning is *not* the bottleneck. Confirm
this holds across brain sizes and rule out any per-frame learning overhead
we missed.

**What to measure:** Cold replay time per image at brain sizes from 1k to
10k neurons. Should be a clean function of neuron count.

---

## Code optimizations (after experiments inform priority)

### C1. Tighten index selectivity in `recognize_patterns`

**File:** [brain/brain-core/src/neuron.rs:984](../../brain/brain-core/src/neuron.rs)
(`get_pattern_candidates_at_age`)

**Problem:** A single shared `(neuron_id, distance)` pair makes a pattern a
candidate. With binary inputs this returns ~50% of the brain per call.

**Ideas to explore:**
- Require k ≥ 2 overlapping entries before a pattern becomes a candidate.
- Weight overlap by bucket rarity (TF-IDF-style) — entries shared by many
  patterns count less.
- Pre-filter by pattern size: if a pattern needs N entries to match above
  threshold and the observed context only overlaps in <N entries, skip.

**Risk:** Changing the criterion changes which patterns get scored, which
changes recognition output. Needs accuracy regression check after.

### C2. `get_level_tasks` — stop cloning every entry every frame

**File:** [brain/brain-core/src/thalamus.rs](../../brain/brain-core/src/thalamus.rs)
(around line 946, per analysis)

**Problem:** 1.8s / 21% of `process_levels`. The clone loop builds
`level_context` and pre-allocates error patterns by cloning all level neuron
entries every frame.

**Ideas to explore:**
- Borrow instead of clone where possible.
- Reuse buffers across frames.
- Only rebuild what actually changed since the previous frame.

**Risk:** Lower — this is pure bookkeeping with no semantic effect.

### C3. `collect_votes` — avoid re-keying

**File:** thalamus.rs (vote collection)

**Problem:** 1.1s / 14% of `process_levels`, mostly copying votes into level
state.

**Ideas to explore:**
- Use the same key shape end-to-end so no re-keying is needed.
- Move vote storage into the level state directly.

**Risk:** Low — also bookkeeping.

### C4. Memory shuffling (907ms / 11%)

`get_level_neurons` / `write_back_level_neurons` per level. Likely
reducible by holding references for the duration of the frame rather than
round-tripping through the memory layer.

### C5. The ~650ms unaccounted gap

Drill further with finer instrumentation before deciding what to do here.

---

## Methodology rules

- **One variable at a time.** Don't combine an experiment with a code change
  in the same measurement.
- **Always record:** wall-clock total, `process_levels`, neuron count,
  candidates evaluated, and accuracy. A perf win that tanks accuracy is not
  a win.
- **Extrapolate before committing.** If a code change saves 200ms on a 10s
  frame and we have a 4s structural problem, the structural problem comes
  first.
- **Cold replay is the cleanest benchmark.** No learning noise, no
  randomness — same input always produces the same neuron ops. Use it for
  perf comparisons unless we're specifically measuring training.
- **Record findings in this file** under each experiment as we go, so the
  plan stays current and we don't redo work.
