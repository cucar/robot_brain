# MNIST Demos — Vision & Continual Learning

A sensory-only MNIST classifier built on the brain's count-based voting. One channel per pixel position
(retinotopic — 196 channels at 14×14, 784 at 28×28), all firing concurrently in a single frame per
image. Supervision lands on a separate `digit` action channel via `brain.learn()`. With
`patternForgetRate = 0` and a constant reward of 1, the per-voter consensus reduces to a Naive-Bayes
posterior — every `learn()` call increments per-pixel-per-digit counts, and inference picks the argmax.

**Why this matters.** The interesting result is not the raw accuracy — MNIST is solved by CNNs at
99%+. It is *how* the accuracy is reached: **no backpropagation, no gradient-trained features, a single
online pass, local/additive learning, and demand-driven conjunctive features** (the spatial hierarchy).
Measured against same-preprocessing baselines, the architecture clears the jointly-trained linear model
and approaches k-NN — with a fundamentally different mechanism — and then does **class-incremental
continual learning** where naive backprop nets collapse.

---

## Quick start

Download the MNIST data first (one-time, ~11 MB into `apps/mnist/data/`):

```bash
node apps/mnist/jobs/download.js
```

This fetches the four standard IDX files (60k train, 10k test, gzipped) from Google's MNIST mirror.
Files that already exist are skipped, so it's safe to re-run.

### The train → evaluate workflow

Training and evaluation are **separate runs**. First train and save the brain; the saved backup carries
its own hyperparameters, so evaluation just loads it and turns learning off — no need to repeat the
training flags. Evaluation is a true **frozen** pass: `--disable-learning` stops all weight updates *and*
correction minting, and each test image is classified independently (no test-set adaptation, no leakage).

The fastest meaningful run (~5–8 minutes, ~94.9% held-out) is 14×14:

```bash
# 1. Train one episode and save
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --episodes 1 --group-mode static --group-threshold 0.9 --save-brain mnist14

# 2. Evaluate on the held-out test set (frozen)
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --load-brain mnist14 --disable-learning --test-data
```

Drop `--test-data` to evaluate on the training set instead. The eval run prints
`🧩 Restored brain config …` — confirmation that the matching thresholds were loaded from the backup
rather than silently defaulting.

---

## The Naive-Bayes ladder

The sensory-only app is Naive Bayes by construction, so the architecture must clear that floor *with a
different mechanism* for the result to mean anything. Reference rungs on identical preprocessing:

| Method | Held-out test |
| --- | --- |
| Naive Bayes (independent pixels) — the floor | ~83–84% |
| Jointly-trained linear / logistic regression | ~92% |
| **This architecture (28×28, radius 2)** | **96.44%** |
| k-NN (pure template matching) | ~97% |
| LeNet-5 (LeCun et al., 1998) | ~99.0% |
| Modern CNN | ~99.7% |

The interesting band is **matching/beating the jointly-trained linear model and approaching k-NN, with
no joint training, no labels for feature learning, and no backprop.** The spatial hierarchy buying
~13pp over pixel-only Naive Bayes is the evidence it is not "just NB with extra steps."

---

## The accuracy ladder (joint training)

Each rung is a single training episode, frozen held-out test on all 10k test images. Higher resolution
and a real receptive field (`--radius 2`) drive the climb.

| Config | Held-out test | Train pass | Model size |
| --- | --- | --- | --- |
| 14×14, radius 1 | 94.88% | ~5 min | 32K neurons |
| 28×28, radius 1 | 94.93% | ~45 min | 112K neurons |
| **28×28, radius 2** | **96.26%** (ep1) | ~1.5 hr | 2.16M neurons |

The headline run (28×28, radius 2, merge 0.9):

```bash
node apps/mnist/jobs/test.js --image-size 28 --buckets 2 --columns 20 --per-class 0 --radius 2 --episodes 1 --group-mode static --group-threshold 0.9 --save-brain mnist28
node apps/mnist/jobs/test.js --image-size 28 --buckets 2 --columns 20 --per-class 0 --radius 2 --max-test-images 0 --load-brain mnist28 --disable-learning --test-data
```

**Expected output of the evaluation run:**
```
🧩 Restored brain config from backup 'mnist28': groupMode=static, groupThreshold=0.9
  Episode 1/1: eval(test)=96.26% (9626/10000) | 0:96% 1:99% 2:93% 3:94% 4:95% 5:86% 6:96% 7:95% 8:87% 9:91%

Results
======================================================================
  Final pass: 96.26% (9626/10000)
  Confusion (rows = actual, cols = predicted):
           0    1    2    3    4    5    6    7    8    9
   0     963    0    1    1    0    3    7    1    4    0
   1       0 1122    3    3    0    0    5    0    2    0
   2       8    0  983    4    6    0    2    8   20    1
   3       1    0    9  966    0    8    1   12    9    4
   4       0    1    3    0  963    0    5    0    3    7
   5       3    2    2    8    0  865    9    0    1    2
   6       7    4    0    0    4   12  928    0    3    0
   7       1   11   15    1    6    2    0  970    2   20
   8      10    0    7   14    4    3    1    5  921    9
   9      12    8    2   10   11    4    0   10    7  945
======================================================================
```

**96.26% on the held-out test set from a single training pass** — a frozen, leakage-free evaluation
(no learning, no minting, each image classified on its own). The spatial hierarchy reaches depth 3 at
radius 2. Remaining confusion is between digits whose pixel marginals overlap (4/9, 7/9, 3/5).

### Multiple episodes plateau

Resuming the saved brain to train further (`--load-brain mnist28 … --save-brain …`) moves the held-out
number only marginally — the per-pixel counts saturate after one pass:

| Episode | Held-out test | Prequential train |
| --- | --- | --- |
| 1 | 96.26% | 92.85% |
| **2** | **96.44%** | 98.52% |
| 3 | 96.43% | 99.72% |

The training accuracy keeps climbing toward 100% (the model memorizing the training set harder), but the
test number plateaus at **96.44%** — extra episodes do not generalize. Best joint result: **96.44%**.

### Group threshold sweep (28×28, radius 2)

Loosening the group threshold mints fewer corrections, trading a little accuracy for a much leaner model.
The first four rows are each one **frozen, single-episode** train→eval pair at 28×28 radius 2 (recorded
2026-06-22):

| Group threshold | Held-out test | Neurons | Depth |
|-----------------| --- | --- | --- |
| 0.9             | 96.26% (ep1) | 2.16M | 3 |
| 0.8             | 96.15% | 1.25M | 3 |
| 0.7             | 95.68% | 0.61M | 3 |
| 0.6             | 94.63% | 0.32M | 3 |

### Radius 3

**Radius 3 at 28×28 — abandoned (2026-06-22).** 
Two attempts at the radius-3 gate (group 0.9). 
The model exhibits runaway growth: throughput fell from ~16 img/s to under 1 img/s (~0.85 img/s measured) 
by the halfway point and kept decelerating, projecting a >12 hr single training pass. 
Prequential accuracy was not pulling away from the radius-2 trajectory by anything close enough to justify the ~10× cost. 
This is the runaway-depth behavior at high resolution.
Planned to be revisited after neuron reuse project.  

### Forget rate is not an accuracy lever here

`patternForgetRate` (CLI: `--forget-rate`) decays the activation strength of **pattern neurons** — in
MNIST, the **spatial corrections** — and reaps them when strength hits 0. It does **not** touch the
Naive-Bayes counts, which live in reward-smoothed action connections on the base sensory neurons. So
turning it up only erodes the spatial hierarchy that lifts accuracy above pixel-only NB. Measured at
14×14 (`--per-class 200`):

| `--forget-rate` | Surviving neurons | Minted (cum) | Train acc |
| --- | --- | --- | --- |
| 0 | 15,788 | 15,440 | 81.95% |
| 0.05 | 1,790 | 79,906 | 68.85% |

Rare corrections decay and die, then re-mint when their pixel pattern recurs (mint → die → re-mint
**thrash** — 79,906 mints to end with 1,442 survivors), so accuracy drops and the model does not even
compact usefully. **Keep `--forget-rate` at 0 for all joint runs.** Its real use is the
continual-learning / capacity experiments — where controlled forgetting is the *subject*, and where it
pairs with neuron-reuse (re-bind to the surviving neuron instead of re-minting) rather than running
alone.

### Inference cost (honest note)

The architecture is *not* fast at inference, and cost grows with model size (not training-set size like
k-NN). Measured single-threaded, current implementation:

| Config | Per image | Model size |
| --- | --- | --- |
| 14×14 r1 | ~3.8 ms | 32K neurons |
| 28×28 r1 | ~25 ms | 112K neurons |
| 28×28 r2 | ~143 ms | 2.16M neurons |

At MNIST scale k-NN is ~1–5 ms/query, so the brain is *slower* at the 96% config. The architectural
differences from k-NN are that it does not store the dataset, learns online in a single pass, and learns
continually — not that it infers faster.

---

## Split-MNIST — class-incremental continual learning

The **headline experiment for external positioning.** Standard 5 tasks × 2 classes
(`{0,1} {2,3} {4,5} {6,7} {8,9}`, citing van de Ven & Tolias 2019 and Hsu et al. 2018), trained strictly
sequentially — each task once, never revisited, **no task IDs**, the action space staying all 10 digits.
After each task the brain is frozen and tested on the full 10-class test set to build a retention matrix.
This is the hardest CL regime, where naive backprop nets and regularization methods (EWC) collapse to
~20%.

```bash
node apps/mnist/jobs/test.js --image-size 28 --buckets 2 --columns 20 --per-class 0 --radius 2 --split --group-mode static --group-threshold 0.9
```

A fast version (14×14, 200/class, 2k test) for a ~45-second illustration of the behavior:

```
Retention matrix  T0(01)   T1(23)   T2(45)   T3(67)   T4(89)
  after T0        99.8%     0.0%     0.0%     0.0%     0.0%
  after T1        97.3%    96.2%     0.0%     0.0%     0.0%
  after T2        95.1%    89.9%    97.0%     0.0%     0.0%
  after T3        93.4%    87.3%    95.2%    89.6%     0.0%
  after T4        93.4%    84.0%    88.1%    84.3%    83.9%

  Average accuracy after Task 4 (headline): 86.77%
  Average forgetting: 6.52pp   (naive backprop floor ≈ ~20% avg acc)
```

The full 28×28 / radius-2 / full-data result:

```
Retention matrix  T0(01)   T1(23)   T2(45)   T3(67)   T4(89)
  after T0        99.9%     0.0%     0.0%     0.0%     0.0%
  after T1        99.0%    98.6%     0.0%     0.0%     0.0%
  after T2        98.9%    97.5%    99.0%     0.0%     0.0%
  after T3        98.4%    96.5%    98.5%    96.7%     0.0%
  after T4        98.1%    95.0%    97.2%    95.5%    95.0%

  Average accuracy after Task 4 (headline): 96.15%
  Forgetting per task (max-ever − final): T0:1.8pp T1:3.7pp T2:1.8pp T3:1.2pp T4:0.0pp
  Average forgetting: 1.69pp  (naive backprop floor ≈ ~20% avg acc / heavy forgetting)
```

The behavior is textbook well-behaved class-incremental learning:

- **Upper triangle is 0%** — the model never predicts a class it has not yet been trained on (10-way
  action space, but no wiring to unseen digits). No task-ID cheating.
- **Negligible forgetting** — 1.69pp average across tasks; task 0 drifts from 99.9% to 98.1% after
  four more tasks arrive, rather than collapsing.
- **96.15% average** — far above the ~20% floor where naive backprop lands in this regime, and
  competitive with replay-based methods that store and replay old examples.

**Literature baselines:** naive backprop ~20%, EWC / regularization methods
~20–25%, replay-based methods 70–90% (but they store and replay old examples), joint-training upper
bound ~98%. Citations: van de Ven & Tolias 2019; Hsu et al. 2018.

### Vanilla-MLP baseline — the contrast figure

A plain 2-layer MLP run through the *identical* protocol makes the result concrete: standard SGD with no
continual-learning machinery collapses, where the brain holds. It is a self-contained NumPy script (no
torch, separate from the brain) reading the same gzipped IDX files:

```bash
python apps/mnist/jobs/split_mnist_mlp.py
```

```
Joint training (all 10 classes at once, same MLP): 94.52%   <- the ceiling

Retention matrix  T0(0,1)   T1(2,3)   T2(4,5)   T3(6,7)   T4(8,9)
  after T0         99.9%     0.0%     0.0%     0.0%     0.0%
  after T1          0.0%    98.0%     0.0%     0.0%     0.0%
  after T2          0.0%     0.0%    99.2%     0.0%     0.0%
  after T3          0.0%     0.0%     0.0%    99.5%     0.0%
  after T4          0.0%     0.0%     0.0%     0.2%    97.7%

  Average accuracy after Task 4 (headline): 19.58%

The collapse: 94.5% (joint) -> 19.4% (sequential), a 75pp drop to the ~20% chance floor.
```

The script trains the **same MLP twice** so the collapse has a ceiling to fall from: jointly on all 10
classes shuffled together (**94.5%** — the upper bound), then class-incrementally two classes at a time.
The diagonal tells the story — each task hits ~98–99% the moment it is trained (see the live value on its
own diagonal), then drops to **0%** as the next task overwrites those weights. Only the most recent two
classes survive, so the average sits at the ~20% chance floor. Place this next to the brain's matrix
(96.15% average, 1.69pp forgetting): same protocol, same data, same 10-way output, *both architectures
capable of ~95% jointly* — the only difference is the learning rule. That turns "we don't forget" into
"we don't forget *where standard nets catastrophically do*."

**Why it holds architecturally:** patterns formed for digits 0/1 have disjoint context fingerprints from
2/3, so they don't fire — and aren't modified — during later tasks. The additive/local learning gives
stability; the hierarchy gives accuracy. (The honest caveat for a paper: additive/instance-based learners
sidestep catastrophic forgetting partly by construction, so the result should be baselined against k-NN
and replay methods — not only against the naive-MLP collapse.)

