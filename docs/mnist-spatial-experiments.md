# MNIST Spatial-Processing Experiments — Session Log

All runs use `apps/mnist/jobs/test.js`. **Constant across every run:** `columns=20`,
`forget-rate=0`, `context-length=1` (the job defaults). "Decode" = consensus (the brain's
built-in per-dimension argmax) unless **NB** (the Naive-Bayes log-sum readout added this session:
`--decode nb [--nb-eps E]`). "merge" = `mergeThreshold`, default 0.5 unless shown. "mode/err" =
`--error-mode` / `--error-threshold`. Train numbers are the prequential per-episode figure unless
marked *frozen* (clean post-training pass, `--eval-train`).

## 1. Error-threshold sweeps (consensus decode, 14×14 binary)

| per-class | episodes | mode/err | test# | train→ | test |
| --- | --- | --- | --- | --- | --- |
| 300 | 3 | default (conservative/0.5) | 100 | 98.5% | **93%** (first run — default already engages spatial) |
| 100 | 2 | static 1.0 (pure NB, no patterns) | 100 | 52.7% | 75% |
| 100 | 2 | static 0.2 | 100 | — | 88% |
| 100 | 2 | static 0.1 / 0.05 / 0.0 | 100 | 90.8% | 87% (all three identical — bimodal error) |
| 100 | 4 | static 0.3 | 100 | — | 89% |
| 100 | 4 | static 0.4 | 100 | 93.7% | 85% |
| 100 | 4 | static 0.5 | 100 | 88% | 83% |

## 2. Data growth (consensus, static 0.4 unless noted)

| Res | per-class | episodes | mode/err | test# | train→ | test |
| --- | --- | --- | --- | --- | --- | --- |
| 14² | 300 | 3 | static 1.0 (pure NB) | 200 | 62.8% | 72.5% |
| 14² | 300 | 3 | static 0.4 | 200 | 90.8% | 89.5% |
| 14² | 300 | 3 | static 0.3 | 200 | 96.9% | 93% |
| 14² | 500 | 5 | static 0.4 | 1000 | 91.3% | 84.3% |
| 28² | 100 | 3 | static 0.4 | 200 | 94.3% | 84% |
| 28² | 500 | 5 | static 0.4 | 1000 | 90.9% | 83.8% |
| 28² | full (~5.4K) | 3 | static 0.4 | 2000 | 87.0% | 85.35% |

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

## Headline takeaways

- **Session best: 90.5%** (14×14 full set, NB, merge 0.7).
- **The spatial hierarchy is the discriminator:** +13pp over pixel-only NB.
- **NB readout** beats the brain's consensus by **+5pp** on the same votes.
- **Merge wants to be high (~0.7); episodes, adaptive modes, grayscale, and error-correction do not
  move test** — so the remaining gap is **representational** (feature/matching), not readout or
  training duration.
- Misses are **near-ties on classically-confusable digits** (4↔9, 7↔9, 3↔5), with the true digit
  sitting at rank 2 by a hair on two-thirds of errors.
- Data helps but saturates (~90.5% at 14×14); 7×7 is too coarse; 28×28 ≈ 14×14.

## New job flags added this session

- `--decode nb [--nb-eps E]` — Naive-Bayes log-sum readout over the action votes (vs the brain's consensus).
- `--merge-threshold M` — already plumbed via `run.js`; controls match tolerance (higher = tighter).
- `--eval-train` — clean frozen pass over the training set with the final model.
- `--debug-miss N` — analyze misclassified test images (per-digit NB scores, true-digit rank, margin, voter counts).
- `--error-correct-rounds N` — discriminative second phase: reinforce only on training mispredictions (minting off).
