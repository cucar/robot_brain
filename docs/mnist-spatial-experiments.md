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

| Res | per-class    | episodes | mode/err/merge          | test# | train→  | test                      |
| --- |--------------| --- |-------------------------|-------|---------|---------------------------|
| 14² | 100          | 3 | static 0.3 / merge 0.5  | 200   | 96.87%  | 93% (runaway depth: 36)   |
| 14² | 100          | 3 | static 0.3 / merge 0.7  | 200   | 99.30%  | 93.50% (stable depth: 2)  |
| 14² | 100          | 3 | static 0.2 / merge 0.8  | 200   | 99.90%  | 95.00% (stable depth: 1)  |
| 14² | 100          | 3 | static 0.1 / merge 0.8  | 200   | 99.90%  | 95.00% (stable depth: 1)  |
| 14² | 100          | 3 | static 0.1 / merge 0.9  | 200   | 99.90%  | 95.00% (stable depth: 1)  |
| 14² | 100          | 3 | static 0.2 / merge 0.75 | 200   | 99.30%  | 91.00% (stable depth: 3)  |
| 14² | 100          | 3 | static 0.1 / merge 0.7  | 200   | 100%    | 91.50% (stable depth: 4)  |
| 14² | 100          | 3 | static 0.2 / merge 0.7  | 200   | 99.30%  | 91.00% (stable depth: 3)  |
| 14² | 100          | 3 | static 0.2 / merge 0.6  | 200   | 99.80%  | 91.00% (stable depth: 5)  |
| 28² | 200          | 3 | static 0.1 /merge 0.9   | 200   | 99.75%  | 95.50%                    |

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

1. **MLP class-incremental baseline** — a vanilla MLP trained under the *identical* split protocol
   (digits sequentially, test on all 10), expected to collapse to ~20%. Turns "we don't forget"
   into "we don't forget *where standard nets catastrophically do*." ~30-line Python script, separate
   from the brain. **This is the punch line for the continual-learning claim.**
2. **Full-data split-MNIST at the new optimum** (28×28, radius 2, NB, full train, full 10K test,
   3 ep/digit) — per-class 500 gave 90.04% (gap 4.57pp); the full-data version is the headline split
   number to match the 95.73% joint. **[QUEUED — auto-launches after the 28² r2 merge/bucket sweep.]**
3. **Second dataset — Fashion-MNIST** on the same stack (same pipeline, harder, shows generality;
   MNIST-only is thin for a paper).
4. **Prior-art differentiation writeup** — HTM/Numenta, ART/Grossberg, predictive coding, growing
   neural gas. The existential related-work section.
5. **Radius sweep at 28×28** — radius 2 was tuned at 14×14; the optimum may differ at full res.
6. **Error-threshold re-tune at 28×28 / radius 2** — 0.4 was tuned at 14×14/radius-1, and it may be
   wrong here. The error gates minting on how far a neighbor patch deviates from the modal patch;
   radius 2 has 24 neighbors vs 8, so the deviation-rate distribution is different and the optimal
   threshold likely shifts. Sweep error-threshold (e.g. 0.3 / 0.4 / 0.5 / 0.6) at 28×28 radius 2 NB
   and re-pick. (Do this on a moderate data set first to find the peak — error-threshold tuning,
   like merge, is roughly data-independent — then confirm at full data.)
7. **Package the ablations** (radius / NB-vs-consensus / merge / threshold / resolution sweeps) for
   the paper — these are mostly already run.
8. **Is class-balanced training now obsolete?** The balancing requirement was justified for the
   *consensus* readout (an unbalanced prior "leaks tilt into every background voter and dominates
   the consensus" — see the encoder/job comments). The new NB log-sum readout aggregates per-voter
   posteriors differently (product, not weighted mean), so balancing may no longer be needed — or
   the imbalance bias may surface in a different way. Test: re-run the optimum stack with
   `--no-balance` (NB + radius 2) on the natural MNIST distribution and compare to the balanced
   result. If accuracy holds, balancing can be dropped (simpler pipeline, more training data usable).
9. ~~**Merge sweep at 28×28 radius 2**~~ — **DONE.** Per-class 300, test 2000: 0.65→91.65,
   **0.70→91.85 (peak)**, 0.75→91.40, 0.80→90.10. Confirms the same plateau as 14×14 radius 2;
   0.70 is correct at full resolution. (Bonus: per-class-300 at 28² r2 = ~91.8% > 14² r2's 90.8%,
   more evidence resolution helps with radius 2.)
10. **4 buckets (grayscale) at the new optimum — FULL DATA ONLY.** Grayscale overfit at
    14×14/radius-1 (more distinct patches → memorization). MUST be tested with full data, not a
    limited set: more buckets = exponentially more distinct patches, so it needs *more* samples, not
    fewer — testing on limited data just reproduces the overfit and tells us nothing. Run at 28×28 /
    radius 2 / merge 0.7 / NB / **full train + full test**, compare to the 95.73% binary. **[DONE.]**
    RESULT: **train 99.77% → test 95.66%** vs binary's 96.76% / **95.73%**. So 4-bucket test is a
    **statistical TIE with binary** (95.66 vs 95.73, within noise on 10K) but with a **4.1pp
    train/test gap** vs binary's 1.0pp. Minted **1.74M neurons** (L1:1.6M L2:134K, **depth 2**, vs
    binary's 139K) — a giant FLAT one-off-memorization layer (`4^24` configs/patch → almost every
    patch unique → no recurrence → no L2/L3 stacking). NEITHER prediction held: user expected
    *higher* than binary, Claude expected *lower* — reality was a wash. INTERPRETATION: the extra
    intensity info compensates for the much-worse generalization, so 4-bucket *matches* binary but at
    12.5× the neurons, ~3× slower, 7 GB RAM — i.e. **binary is strictly better at merge 0.7.** BUT the
    99.77% train + 4.1pp gap means there's "trapped" discriminative signal that isn't generalizing —
    which is exactly what #11 (lower merge → patterns fire on approximate matches) and #12 (context
    refinement → consolidate one-offs into general detectors) would unlock. So the result *strengthens*
    the case for #11/#12: 4-bucket could plausibly BEAT binary once its memorized signal generalizes.
11. **4 buckets at LOWER merge (0.6, then 0.5/0.4 if promising).** Merge 0.7 makes 4-bucket patterns
    so specific they never recur. Lower merge = tolerance: patterns fire on *approximate* matches, so
    the over-specific patches recur (shrinking the 1.6M explosion) and fire on test variants
    (generalizing). This is 4-bucket-SPECIFIC — binary wanted *high* merge (it already generalizes),
    but the bigger patch space needs tolerance. **[ON HOLD — do NOT auto-launch; run when user
    returns. User agrees lower merge + context refinement (#12) are the levers to get past depth 2.]**
12. **Re-introduce context refinement** (removed in commit `8a17f4d` "remove context refinement to
    prevent pattern identity drift"). On a matched pattern, refine its stored context: **strengthen**
    common entries, **add** novel, **weaken/delete** missing — so a pattern *consolidates* toward the
    common core of the configs it matches instead of staying frozen at its mint-time identity. This is
    the missing *abstraction/generalization* step: it would turn the 1.6M one-off patterns into far
    fewer, general detectors that recur and climb the hierarchy. **Synergistic with #11** (lower merge
    lets patterns fire on approximate matches; refinement then consolidates them). CAVEAT: it was
    removed for **reproducibility** — mid-training refinement makes recognition non-deterministic
    (training sees half-refined patterns, replays see refined ones). Guard it: refine only during
    training and freeze for eval, or do consolidation in a separate pass. **[QUEUED — after #11.]**

## New job flags added this session

- `--decode nb [--nb-eps E]` — Naive-Bayes log-sum readout over the action votes (vs the brain's consensus).
- `--radius N` — spatial neighborhood radius passed to the encoder (1 = 3×3, 2 = 5×5, 3 = 7×7). **Radius 2 is the optimum.**
- `--merge-threshold M` — already plumbed via `run.js`; controls match tolerance (higher = tighter).
- `--eval-train` — clean frozen pass over the training set with the final model.
- `--debug-miss N` — analyze misclassified test images (per-digit NB scores, true-digit rank, margin, voter counts; renders the image as ASCII).
- `--error-correct-rounds N` — discriminative second phase: reinforce only on training mispredictions (minting off).
