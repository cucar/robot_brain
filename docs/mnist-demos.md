# MNIST Demos — Vision & Continual Learning

Handwritten digits on the brain. The image is one channel laid out over x and y, with one event dimension: a
pixel's value, black (0) or white (1) by default. Every pixel fires in every frame. A second channel, `digit`, holds
ten digit actions. For the other demos see [stock-demos.md](stock-demos.md), [text-demos.md](text-demos.md),
[shape-demos.md](shape-demos.md) and [tap-demos.md](tap-demos.md).

Each training image is two frames: the image, then the image's digit, run as a lesson and rewarded +1, scoped to the
digit channel and that one frame. The context is reset after each image, so no image sees another. An evaluation
loads a trained brain with learning off: each test image's answer runs, unrewarded, and nothing of the brain changes.

There is no backpropagation and no gradient: the brain learns online, one image at a time, by naming the patterns
of pixels that recur and recording which digit followed them.

---

## Quick start

Download the MNIST data first (one-time, ~11 MB into `apps/mnist/data/`):

```bash
node apps/mnist/jobs/download.js
```

This fetches the four standard IDX files (60k train, 10k test) from Google's MNIST mirror. Files that already exist
are skipped, so it is safe to re-run.

### The train → evaluate workflow

Training and evaluation are **separate runs**. First train and save the brain; then load it with learning off and
evaluate it on the held-out test set. Each test image is classified on its own: no learning, no adaptation to the
test set.

```bash
# 1. Train on 500 images of each digit at 14x14, and save
node apps/mnist/jobs/test.js --image-size 14 --per-class 500 --save-brain mnist14

# 2. Evaluate on the 10,000 held-out test images, frozen
node apps/mnist/jobs/test.js --image-size 14 --per-class 500 --load-brain mnist14 --disable-learning --test-data
```

The options:

| option | |
|---|---|
| `--image-size <n>` | the image side: 28, 14, 7 or 4 (each block of the 28×28 image averaged); 7 by default |
| `--buckets <n>` | a pixel's values; 2 by default (black and white: white where the block's mean reaches 128) |
| `--per-class <n>` | train on the first n images of each digit, in the file's order; 0, by default, takes as many as the smallest digit has (5,421, so 54,210 images) |
| `--train-images <n>` | train on the first n images instead, unbalanced |
| `--no-balance` | train on the whole training set |
| `--episodes <n>` | passes over the training images; 1 by default |
| `--max-test-images <n>` | evaluate on at most n images; 0, by default, is all |
| `--test-data` | evaluate on the held-out test set; without it, on the training images |
| `--split` | Split-MNIST (below) |

**Expected output of the evaluation run:**
```
mnist: 14x14, black and white pixels, the brain loaded from mnist14 on 10000 test images, learning off
  test: 10000 images | spoke 10000 | right 5667 (56.7%) | apex/image 10.9 | neurons 5670 functions 6595 | 121s
  confusion (row = digit, column = said):
    0 [688, 2, 40, 22, 18, 65, 45, 11, 77, 12]
    1 [28, 865, 36, 11, 21, 58, 36, 31, 38, 11]
    2 [144, 2, 391, 118, 21, 178, 72, 13, 83, 10]
    3 [66, 3, 66, 502, 16, 193, 22, 7, 94, 41]
    4 [81, 12, 29, 47, 459, 98, 47, 18, 85, 106]
    5 [51, 2, 59, 125, 14, 499, 22, 12, 89, 19]
    6 [78, 4, 63, 13, 18, 67, 641, 6, 58, 10]
    7 [43, 15, 54, 41, 43, 85, 14, 492, 36, 205]
    8 [93, 4, 33, 61, 31, 68, 44, 5, 576, 59]
    9 [65, 7, 24, 29, 80, 77, 26, 64, 83, 554]
```

**56.7% on the held-out test set from 5,000 training images**, one pass. The 1s are read best (865 of 1,135); the 2s
worst, often taken for 0, 3 or 5. Training takes about 3 minutes and the evaluation about 2.

---

## Results

Each row is one training pass, then a frozen evaluation on held-out test images. Training is kept to runs of a few
minutes: the time per training image grows with the dictionary, at 14×14 from 24 ms an image at image 1,000 to 34 ms
at image 5,000 and 61 ms at image 10,000.

| image | training images | test images | held-out test | train pass | evaluation | functions |
| --- | --- | --- | --- | --- | --- | --- |
| 14×14 | 5,000 (500 per digit) | 10,000 | **56.7%** | 178 s | 121 s (12 ms an image) | 6,595 |
| 28×28 | 1,000 (100 per digit) | 2,000 | **39.8%** | 144 s | 193 s (97 ms an image) | 2,582 |

```bash
node apps/mnist/jobs/test.js --image-size 28 --per-class 100 --save-brain mnist28
node apps/mnist/jobs/test.js --image-size 28 --per-class 100 --load-brain mnist28 --disable-learning --test-data --max-test-images 2000
```

The two rows differ in training size as well as in resolution, so they do not compare one to the other.

Reference results on MNIST, for scale (each trained on the full 60,000 images):

| Method | Held-out test |
| --- | --- |
| Naive Bayes (independent pixels) | ~83–84% |
| Linear / logistic regression | ~92% |
| k-NN (pure template matching) | ~97% |
| LeNet-5 (LeCun et al., 1998) | ~99.0% |
| Modern CNN | ~99.7% |

---

## Split-MNIST — class-incremental continual learning

Five tasks of two digits each, `{0,1} {2,3} {4,5} {6,7} {8,9}` (citing van de Ven & Tolias 2019 and Hsu et al.
2018), trained strictly in turn: each task once, never revisited, **no task given**, the answers always all ten
digits. After each task a copy of the brain with learning off is tested on the test set, building a retention matrix:
the share right on each task's test digits after each task.

```bash
node apps/mnist/jobs/test.js --image-size 14 --split --per-class 1000 --max-test-images 2000
```

1,000 images of each digit (10,000 in all), 2,000 test images after each task:

```
Retention matrix  T0(01)   T1(23)   T2(45)   T3(67)   T4(89)
  after T0        93.6%    0.0%     0.0%     0.0%     0.0%
  after T1        66.0%    80.0%    0.0%     0.0%     0.0%
  after T2        52.1%    50.0%    77.8%    0.0%     0.0%
  after T3        43.8%    42.5%    51.8%    83.3%    0.0%
  after T4        39.4%    29.1%    26.5%    50.7%    75.6%
  Average accuracy after the last task: 44.26%
  Forgetting per task (best less last): T0:54.3pp T1:50.9pp T2:51.3pp T3:32.6pp T4:0.0pp
  Average forgetting: 37.82pp
```

- **The upper triangle is 0%**: the brain never answers a digit it has not yet been taught. No task-ID cheating.
- **Each task is learned when it is taught**: 76–94% on its own digits right after its task.
- **Earlier tasks fade as new ones arrive**: the first task falls from 93.6% to 39.4% after four more, and the average
  forgetting is 37.82 points. After the last task the average is 44.26%: above the ~20% floor where naive backprop
  nets land in this regime, well below joint training.

The run takes about 9 minutes.

**Literature baselines:** naive backprop ~20%, EWC / regularization methods ~20–25%, replay-based methods 70–90% (but
they store and replay old examples), joint-training upper bound ~98%. Citations: van de Ven & Tolias 2019; Hsu et al.
2018.

### Vanilla-MLP baseline — the contrast figure

A plain 2-layer MLP run through the *identical* protocol shows the collapse standard SGD suffers with no
continual-learning machinery. It is a self-contained NumPy script (no torch, separate from the brain) reading the same
gzipped IDX files:

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

Each task hits ~98–99% the moment it is trained, then drops to **0%** as the next task overwrites those weights. The
brain's earlier tasks fade too, but they do not drop to zero: after all five tasks its oldest task still stands at
39.4%, where the MLP's stands at 0%.

---

## The MNIST benchmarks

Six runs at 14×14 on the first training images in the file's order, each image taught, no test images. They are
benchmarks (see [Benchmarks](../README.md#benchmarks)); their full outputs are saved in
[benchmarks/output](benchmarks/output).

```bash
node apps/mnist/jobs/test.js --image-size 14 --train-images <n> --episodes <e>
```

| benchmark | training images × episodes | right | functions | the dictionary last changed | run time |
|---|---|---|---|---|---|
| `mnist-1x60` | 1 × 60 | 1/1 in the last episode; 59/60 over the run | 40 | episode 25 | 1.0 s |
| `mnist-10x20` | 10 × 20 | 10/10 in the last episode; 187/200 over the run | 220 | episode 20 (still changing) | 3.4 s |
| `mnist-10x300` | 10 × 300 | 10/10 in the last episode; 2,987/3,000 over the run | 229 | episode 298 | 56.7 s |
| `mnist-100` | 100 × 1 | 19/100 | 289 | still changing | 1.8 s |
| `mnist-255` | 255 × 1 | 96/255 | 641 | still changing | 5.2 s |
| `mnist-1000` | 1,000 × 1 | 435/1,000 | 2,055 | still changing | 23.3 s |

One image shown again and again is said right from the second episode on, and its dictionary settles at episode 25;
ten images are all right from the third episode on. The saved outputs of all but `mnist-100` list the dictionary's largest functions with
functions of equal size in another order than the job prints them (in `mnist-10x20` a different one of the tied
functions takes the last place), so the runner reports them as `different` on those lines alone.

The data are the four raw (gunzipped) idx files of the standard MNIST set; their MD5 sums:

| file | MD5 |
|---|---|
| `train-images-idx3-ubyte` | `6bbc9ace898e44ae57da46a324031adb` |
| `train-labels-idx1-ubyte` | `a25bea736e30d166cdddb491f175f624` |
| `t10k-images-idx3-ubyte` | `2646ac647ad5339dbf082846283269ea` |
| `t10k-labels-idx1-ubyte` | `27ae3e4e09519cfbb04c329615203637` |
