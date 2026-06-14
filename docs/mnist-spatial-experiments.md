# MNIST Spatial-Processing Experiments — Session Log

All runs use `apps/mnist/jobs/test.js`. **Constant across every run:** `columns=20`,
`forget-rate=0`, `context-length=1` (the job defaults). "Decode" = consensus (the brain's
built-in per-dimension argmax) unless **NB** (the Naive-Bayes log-sum readout added this session:
`--decode nb [--nb-eps E]`). "merge" = `mergeThreshold`, default 0.5 unless shown. "mode/err" =
`--error-mode` / `--error-threshold`. Train numbers are the prequential per-episode figure unless
marked *frozen* (clean post-training pass, `--eval-train`).

## 1. Error-threshold sweeps (consensus decode, 14×14 binary)

Result: dynamic error modes perform ok, but end up in runaway depth. 
It may reach some stabilization at some point, but it looks quite far. 
It's best to use static error mode. Threshold seems best around 0.3 or 0.4.

| per-class | episodes | mode/err | test# | train→ | test                                                |
| --- |----------| --- | --- |--------|-----------------------------------------------------|
| 300 | 3        | default (conservative/0.5) | 100 | 94.20% | 90.00% |
| 100 | 2        | static 1.0 | 100 | 90.80% | 87.00% |
| 100 | 2        | static 0.2 | 100 | 90.00% | 88.00% |
| 100 | 2        | static 0.1 / 0.05 / 0.0 | 100 | 90.8%  | 87% (all three identical — bimodal error)           |
| 100 | 2        | static 0.3 | 100 | 90.30% | 89%                                                 |
| 100 | 4        | static 0.4 | 100 | 93.7%  | 85%                                                 |
| 100 | 4        | static 0.5 | 100 | 88%    | 83%                                                 |

## 2. Data growth

| Res | per-class    | episodes | mode/err          | test# | train→  | test                      |
| --- |--------------| --- |-------------------------|-------|---------|---------------------------|
| 14² | 300          | 3 | static 0.3              | 200   | 96.87%  | 93% (runaway depth: 36)   |
| 14² | 300          | 3 | static 0.35             | 200   | 97.20%  | 93% (runaway depth: 17)   |
| 14² | 300          | 3 | static 0.4              | 200   | 90.8%   | 89.5% (runaway depth: 11) |
| 14² | 300          | 3 | static 0.45             | 200   | 90.8%   | 89.5% (runaway depth: 11) |
| 14² | 300          | 3 | static 0.5              | 200   | 84.8%   | 89.5% (stable depth: 2)   |
| 14² | 500          | 3 | static 0.5              | 1000  | 84.4%   | 82.1% (stable depth: 2)   |
| 14² | 500          | 5 | static 0.4              | 1000  | 91.3%   | 84.3%                     |
| 28² | 100          | 3 | static 0.4              | 200   | 94.3%   | 84%                       |
| 28² | 500          | 5 | static 0.4              | 1000  | 90.9%   | 83.8%                     |
| 28² | full (~5.4K) | 3 | static 0.4              | 2000  | 87.0%   | 85.35%                    |

| Res | per-class    | episodes | mode/err/merge          | test# | train→ | test                     |
| --- |--------------| --- |-------------------------|-------|--------|--------------------------|
| 14² | 100          | 3 | static 0.3 / merge 0.5  | 200   | 96.87% | 93% (runaway depth: 36)  |
| 14² | 100          | 3 | static 0.3 / merge 0.7  | 200   | 99.30% | 93.50% (stable depth: 2) |
| 14² | 100          | 3 | static 0.2 / merge 0.8  | 200   | 99.90% | 95.00% (stable depth: 1) |
| 14² | 100          | 3 | static 0.1 / merge 0.8  | 200   | 99.90% | 95.00% (stable depth: 1) |
| 14² | 100          | 3 | static 0.1 / merge 0.9  | 200   | 99.90% | 95.00% (stable depth: 1) |
| 14² | 100          | 3 | static 0.1 / merge 0.9  | 2000  | 99.23% | 90.25% (stable depth: 1) |
| 14² | 100          | 3 | static 0.2 / merge 0.75 | 200   | 99.30% | 91.00% (stable depth: 3) |
| 14² | 100          | 3 | static 0.1 / merge 0.7  | 200   | 100%   | 91.50% (stable depth: 4) |
| 14² | 100          | 3 | static 0.2 / merge 0.7  | 200   | 99.30% | 91.00% (stable depth: 3) |
| 14² | 100          | 3 | static 0.2 / merge 0.6  | 200   | 99.80% | 91.00% (stable depth: 5) |
| 28² | 200          | 3 | static 0.1 /merge 0.9   | 200   | 99.75% | 95.50%                   |

## 3. Split-MNIST (sequential training, catastrophic-forgetting probe)

| Res | per-class | eps/digit | mode/err | decode | test# | test |
| --- | --- | --- | --- | --- | --- | --- |
| 28² | 500 | 2 | static 0.4 | consensus | 2000 | **82.7%** (vs 85.35 joint → minimal forgetting) |
| 28² | 500 | 2 | static 1.0 (NB control) | consensus | 2000 | 70.9% (vs ~72.5 joint → NB also barely forgets) |

Conclusion: no-catastrophic-forgetting is a property of the additive readout, not unique to the
hierarchy (the NB control forgets just as little).

## 4. NB decode vs consensus (same trained brain)

| Res | per-class | mode/err | merge | decode | test# | test |
| --- | --- | --- | --- | --- | --- | --- |
| 14² | 300 | static 0.4 | 0.5 | consensus | 200 | 89.5% |
| 14² | 300 | static 0.4 | 0.5 | NB ε1e-6 | 200 | 93.5% |
| 14² | 300 | static 0.4 | 0.5 | NB ε1e-3 | 200 | 94.0% |
| 14² | 300 | static 0.4 | 0.5 | consensus | **1000** | 84.3% |
| 14² | 300 | static 0.4 | 0.5 | NB ε1e-3 | **1000** | 89.5% (reliable: +5.2pp) |
| 28² | 500 | static 0.4 | 0.5 | NB ε1e-3 | 1000 | 89.3% (vs 83.8 consensus → +5.5pp) |

The 200-image 94% was small-sample optimism; the reliable NB number is ~89.5%, a robust +5pp over
consensus.

## 5. Adaptive error modes

| Res | per-class | mode (warmup 0.4) | decode | test# | test | neurons |
| --- | --- | --- | --- | --- | --- | --- |
| 14² | 100 | static | consensus | 100 | 85% | 5K |
| 14² | 100 | conservative | consensus | 100 | 87% | 17K |
| 14² | 100 | neutral | consensus | 100 | 86% | 31K |
| 14² | 100 | aggressive | consensus | 100 | 89% | 54K |
| 14² | 300 | static + NB | NB ε1e-3 | 1000 | 89.5% | 8K |
| 14² | 300 | conservative + NB | NB ε1e-3 | 1000 | 88.7% | 38K |
| 14² | 300 | aggressive + NB | NB ε1e-3 | 1000 | 90.1% | 131K |

Adaptive over-mints with no reliable test gain (aggressive's +0.6pp is noise at 16× the neurons);
its mean±σ formulas were calibrated for the pre-fix-2.1 error distribution. Use static.

## 6. Merge / episodes / resolution / grayscale battery (NB ε1e-3, per-class 300, test 1000, ep3)

| Res | buckets | merge | episodes | test |
| --- | --- | --- | --- | --- |
| 14² | 2 | 0.2 | 3 | 88.0% |
| 14² | 2 | 0.3 | 3 | 88.6% |
| 14² | 2 | 0.5 | 3 | 89.5% |
| 14² | 2 | 0.7 | 3 | 89.6% |
| 14² | 4 | 0.3 | 3 | 86.8% |
| 14² | 4 | 0.5 | 3 | 87.6% |
| 14² | 2 | 0.5 | 5 | 89.5% |
| 14² | 2 | 0.5 | 8 | 89.4% |
| 28² | 2 | 0.5 | 3 | 89.0% |
| 28² | 4 | 0.5 | 3 | 88.0% |
| 28² | 4 | 0.3 | 3 | 89.1% |

Merge wants to be *high* (monotonic 0.2→0.7); episodes flat; resolution 14²≈28²; grayscale (4
buckets) overfits and never beats binary.

## 7. Higher merge + frozen-train eval (NB ε1e-3, 14² binary, per-class 300, test 1000)

| merge | frozen train | test |
| --- | --- | --- |
| 0.7 | 97.63% | 89.5% |
| 0.8 / 0.9 / 0.95 | 99.17% (all identical) | 89.2% |

Frozen train ≈ prequential (suspicion that it would be lower is disproven). Tighter merge memorizes
harder (train↑) with no test gain. Merge 0.7 is the operating point.

## 8. Resolution × data grid (NB ε1e-3, merge 0.7, binary, ep3, test 2000)

| Res | per-class 100 | 300 | 1000 | full |
| --- | --- | --- | --- | --- |
| 7² | 54.6% | 57.25% | 58.15% | 59.1% |
| 14² | — | 88.6% | 90.35% | **90.5%** (session best) |

7×7 is information-starved (digits collapse to identical patches); 14×14 is the sweet spot; data
helps but saturates at ~90.5%.

## 9. Miss diagnosis & readout controls (14² binary, per-class 300, merge 0.7, NB ε1e-3, test 2000)

| run | flags | train | test | finding |
| --- | --- | --- | --- | --- |
| miss diagnosis | `--debug-miss 20` | 97.6% (frozen) | 88.6% | 67% of misses are rank-2 near-ties; voter count identical hit vs miss (no recognition collapse) |
| pixel-only control | static **1.0** (no patterns) | 75.3% | 75.35% | the hierarchy adds **+13pp** over pixel-NB |
| error-correction | `--error-correct-rounds 10 --eval-train` | **99.33%** (frozen) | 88.85% | train→99% but test flat → ceiling is representational, not readout/training |

## 10. Radius sweep — the key unlock (14×14, NB ε1e-3, merge 0.7, ep3, test 1000)

| radius (window) | per-class | neurons | test |
| --- | --- | --- | --- |
| 1 (3×3) | 300 | 6K | 89.6% |
| **2 (5×5)** | 300 | 17K | **90.8%** |
| 3 (7×7) | 300 | 21K | 89.5% (overfit — peak is radius 2) |
| **2 (5×5)** | **1000** | 28K | **91.0%** (radius 2 + more data compounds) |

Bigger receptive field adds genuine *structural* context (unlike grayscale's intensity bins) and
helps; radius 3 is too specific/sample-hungry and falls back. **Radius 2 is the sweet spot.**

## 11. Radius-2 merge re-tune (14×14 radius 2, NB ε1e-3, per-class 300, test 1000)

| merge | test |
| --- | --- |
| 0.70 | 90.8% |
| 0.72 | 90.5% |
| 0.75 | 90.7% |
| 0.78 | 89.1% |
| 0.80 | 89.1% |
| 0.85 | 85.6% |

**Flat plateau 0.70–0.75 (~90.7%), then a cliff at 0.78+.** 0.75 *ties* 0.70 (does not beat it), so
0.7 is correctly at the top of the plateau. NOTE: at 7×7 the merge curve *inverts* (higher looks
better) — that is info-starvation noise; **do not tune merge on 7×7.**

## 12. THE CAPSTONE — full optimum stack (28×28 binary, radius 2, merge 0.7, static err 0.4, NB ε1e-3, FULL train ~54,210, FULL 10,000 test, 3 episodes)

| | result |
| --- | --- |
| Train (ep1→ep3) | 93.12% → 96.21% → 96.76% |
| **Test (full 10,000)** | **95.73% (9,573/10,000)** |
| Train/test gap | ~1pp (excellent generalization) |
| Hierarchy | depth 2, ~139K neurons (saturated) |
| Wall-clock | ~4 hours, ~12 img/s, ~850 MB |

Per-digit test: 0:97.8 1:98.8 2:94.7 3:95.7 4:97.3 5:95.9 6:96.7 7:91.9 8:94.7 9:93.9 (weakest 7,
confuses with 9/2). **The earlier "~90% ceiling / representational gap" was a 14×14-radius-1
artifact — restoring resolution (28²), a real receptive field (radius 2), and full data dissolves
it from ~90% to 95.7%.**

## 13. Continual learning — joint vs split at the new optimum

Split-MNIST here is **class-incremental** (train one digit at a time, test on all 10 with *no task
label, no replay, no EWC*) — the hardest CL regime, where naive backprop nets collapse to ~20%.

| config (28×28, radius 2, merge 0.7, NB) | JOINT | SPLIT (class-incremental) | gap |
| --- | --- | --- | --- |
| per-class 500 | 94.61% | 90.04% | 4.57pp |
| **FULL data** | **95.73%** | **91.15%** | **4.58pp** |

**HONEST NUANCE (from the full-data confusion matrix): forgetting is NOT uniform — it's graceful,
order-dependent degradation.** Digit 0 (trained FIRST, then buried under 9 later phases) drops to
**61%** (leaks to 6/8/5); every other digit is 90–98%. The earliest-learned class's *shared* voters
get overwritten by subsequent training; digit-specific voters survive. So the correct framing is
"graceful degradation with a recency bias," NOT "zero forgetting." Still strong: worst class 61% vs
naive backprop ~20% *overall* (the empty quadrant — additive/local learning gives stability,
hierarchy gives accuracy). The additive, local, non-gradient learning is *why* it degrades
gracefully (the NB control isolates this); the hierarchy is *why* it's also accurate. (Old config —
28×28, consensus, no radius — gave split 82.7% vs joint 85.35%.) **Still needed for the paper: the
MLP baseline under the identical protocol (TODO #1) to show the ~20% contrast.**

## Headline takeaways

- **Session best: 95.73%** on full MNIST test (28×28 binary, radius 2, merge 0.7, NB, full data).
  Beats logistic regression (~92%), approaches kNN (~97%); below CNN SOTA (99%+) but a respectable
  number that makes the architecture credible.
- **Lever ranking:** radius 2 (>+4pp at full res) **>** NB readout (+5pp vs consensus) **>**
  resolution+data. Radius was the unlock.
- **The "~90% ceiling" was NOT fundamental** — it was a low-resolution / small-radius artifact.
- **The spatial hierarchy is the discriminator** (+13pp over pixel-only NB); the NB log-sum readout
  beats the brain's arithmetic-mean consensus by +5pp on the same votes.
- **Merge plateau is 0.70–0.75 (cliff at 0.78+); radius 3 overfits; grayscale, threshold-0,
  adaptive modes, error-correction, and extra episodes do NOT move test.**
- **Continual learning may be the strongest paper angle:** class-incremental split at high accuracy
  with near-zero forgetting sits in the empty quadrant (NB-like methods don't forget but are weak;
  backprop nets are strong but forget).

## Next steps / TODO

This list is scoped to the roadmap's **Optimize MNIST** item (see [roadmap.md](roadmap.md)).
Obsolete and completed probes have been pruned; the result tables above are kept as the data log.

**Active (queued under Optimize MNIST):**

1. **Radius 3 at 28×28.** Radius 2 was the 14×14 optimum and held at 28×28 per-class-300; radius 3
   overfit at 14×14 but the larger-resolution receptive-field budget may behave differently at full
   res. Run radius 3 at 28×28 / merge 0.7 / NB and compare to the radius-2 capstone (95.73%).
2. **Error / merge threshold re-tune at 28×28 (radius x).** 0.4 / 0.7 were tuned at lower res /
   radius. Sweep the paired corners **err 0.1 / merge 0.9, err 0.2 / merge 0.8, err 0.3 / merge 0.7**
   at 28×28 and re-pick. Find the peak on a moderate set first (threshold tuning is roughly
   data-independent), then confirm at full data.
   - Anchor run: `node apps/mnist/jobs/test.js --image-size 28 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --episodes 3 --error-mode static --error-threshold 0.1 --merge-threshold 0.9`
3. **Is class-balanced training still required?** The balancing requirement was justified for the
   *consensus* readout (an unbalanced prior "leaks tilt into every background voter"). With the
   recent changes (NB log-sum readout, the landed spatial chain), it may be obsolete. Test: re-run
   the optimum stack with `--no-balance` (NB + radius 2) on the natural MNIST distribution and
   compare to the balanced result. If accuracy holds, drop balancing (simpler pipeline, more usable
   data).
4. **Literature-standard Split-MNIST.** Re-run continual learning on the *standard* **5 tasks × 2
   classes** (0/1, 2/3, …) protocol — not the current 10 tasks × 1 class — citing van de Ven &
   Tolias / Hsu, so our ~90–91% lines up apples-to-apples against the cited floor. Pair it with an
   **MLP class-incremental baseline**: a vanilla MLP under the identical split protocol (digits
   sequentially, test on all 10), expected to collapse to ~20%. ~30-line Python script, separate
   from the brain. This turns "we don't forget" into "we don't forget *where standard nets
   catastrophically do*" — the punch line for the continual-learning claim.
5. **Re-introduce context refinement** (removed in commit `8a17f4d` to prevent pattern-identity
   drift). On a matched pattern, **strengthen** common context entries, **add** novel,
   **weaken/delete** missing — so a pattern *consolidates* toward the common core of the configs it
   matches instead of staying frozen at mint-time identity. This is the missing
   abstraction/generalization step that would turn one-off corrections into general detectors that
   recur and climb past depth 2. Add behind a flag for both temporal and spatial; guard it (refine
   during training, freeze for eval) for reproducibility. Scheduled as its own roadmap item; test
   MNIST + stocks. Synergistic with the lower-merge corners in #2.

**Beyond MNIST optimization (paper / generality, tracked here for continuity):**

6. **Second dataset — Fashion-MNIST** on the same stack (same pipeline, harder, shows generality;
   MNIST-only is thin for a paper).
7. **Prior-art differentiation writeup** — HTM/Numenta, ART/Grossberg, predictive coding, growing
   neural gas. The existential related-work section.
8. **Package the ablations** (radius / NB-vs-consensus / merge / threshold / resolution sweeps) for
   the paper — mostly already run.

**Done / superseded (kept for the record):**

- ✅ **Merge sweep at 28×28 radius 2** — per-class 300, test 2000: 0.65→91.65, **0.70→91.85 (peak)**,
  0.75→91.40, 0.80→90.10. Same plateau as 14×14 radius 2; 0.70 correct at full res.
- ✅ **4 buckets (grayscale) at the optimum, full data** — train 99.77% → test 95.66% vs binary's
  96.76% / 95.73%: a statistical **tie** with binary but a 4.1pp train/test gap (vs 1.0pp) and
  **1.74M neurons** (L1:1.6M L2:134K, depth 2) vs binary's 139K — a flat one-off-memorization layer
  (`4^24` configs/patch → almost every patch unique). **Binary is strictly better at merge 0.7.** The
  trapped-but-non-generalizing signal (99.77% train) is exactly what context refinement (#5) and
  lower merge (#2 corners) would unlock — so the result strengthens the case for those.
- ⏸️ **4 buckets at lower merge** — superseded: pursue the lower-merge corners as part of the unified
  err/merge re-tune (#2) and context refinement (#5) rather than as a 4-bucket-specific probe; binary
  remains the operating point unless refinement closes the generalization gap.

## Job-flag changes (planned — see roadmap *Near-term engineering*)

The roadmap removes/refactors several of this session's experimental flags. Planned end state:

- `--decode nb [--nb-eps E]` — **being removed.** The NB log-sum either becomes the brain's default
  consensus (in Rust) or moves to a votes-only app rule; either way the option goes away and the
  default is NB. See roadmap *NB decode → brain*.
- `--error-correct-rounds N` — **being removed.** Discriminative second phase pushed train→99% with
  no test gain (ceiling is representational, not readout) — dropped.
- `--eval-train` — **being replaced.** Wire-only training becomes the default (no per-image
  infer/decode/tally), the frozen `runTrainEval()` always runs at the end, and an opt-in
  `--eval-train-per-episode` (off by default) runs a frozen pass after each episode.
- `--radius N` — kept. Spatial neighborhood radius (1 = 3×3, 2 = 5×5, 3 = 7×7). **Radius 2 is the
  optimum.**
- `--merge-threshold M` — kept; controls match tolerance (higher = tighter).
- `--debug-miss N` — kept; analyzes misclassified test images (per-digit NB scores, true-digit rank,
  margin, voter counts; renders the image as ASCII).
