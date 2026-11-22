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
  test: 10000 images | spoke 10000 | right 5692 (56.9%) | apex/image 10.9 | neurons 5668 functions 6598 | 145s
  confusion (row = digit, column = said):
    0 [689, 3, 43, 18, 20, 66, 43, 7, 80, 11]
    1 [35, 864, 35, 9, 23, 56, 38, 27, 37, 11]
    2 [132, 2, 398, 116, 23, 174, 76, 11, 88, 12]
    3 [65, 1, 74, 503, 16, 192, 21, 11, 91, 36]
    4 [87, 13, 29, 49, 463, 92, 46, 21, 82, 100]
    5 [49, 2, 62, 122, 14, 496, 19, 9, 97, 22]
    6 [74, 5, 61, 10, 22, 69, 646, 4, 58, 9]
    7 [46, 16, 51, 49, 46, 84, 12, 486, 31, 207]
    8 [94, 4, 35, 59, 33, 68, 42, 6, 574, 59]
    9 [67, 5, 26, 31, 77, 76, 23, 68, 63, 573]
```

**56.9% on the held-out test set from 5,000 training images**, one pass. The 1s are read best (864 of 1,135); the 2s
worst, often taken for 0, 3 or 5. Training takes about 3½ minutes and the evaluation about 2½.

---

## Results

Each row is one training pass, then a frozen evaluation on held-out test images. Training is kept to runs of a few
minutes: the time per training image grows with the dictionary, at 14×14 from about 28 ms an image at image 1,000 to about
39 ms at image 5,000 and about 70 ms at image 10,000.

| image | training images | test images | held-out test | train pass | evaluation | functions |
| --- | --- | --- | --- | --- | --- | --- |
| 14×14 | 5,000 (500 per digit) | 10,000 | **56.9%** | about 200 s | about 145 s (about 14 ms an image) | 6,598 |
| 28×28 | 1,000 (100 per digit) | 2,000 | **39.5%** | about 170 s | about 240 s (about 120 ms an image) | 2,591 |

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
digits. Then a copy of the brain with learning off is tested on the test set.

**Forgetting is measured against regular training.** The same 10,000 images, 1,000 of each digit, are taught to a
second brain with the digits mixed, in the file's order, and that brain is tested on the same test images. The two
brains see the same training images and the same test images. Only the order differs. What regular training gets
right and split training does not is the forgetting.

```bash
# split: two digits at a time
node apps/mnist/jobs/test.js --image-size 14 --split --per-class 1000 --max-test-images 2000

# regular: the same images, the digits mixed
node apps/mnist/jobs/test.js --image-size 14 --per-class 1000 --save-brain mnist14x1000
node apps/mnist/jobs/test.js --image-size 14 --per-class 1000 --load-brain mnist14x1000 --disable-learning --test-data --max-test-images 2000
```

| training on the same 10,000 images | right, of the same 2,000 test images | functions |
|---|---|---|
| regular, the digits mixed | 1,057 (52.9%) | 11,765 |
| split, two digits at a time | 907 (45.4%) | 11,760 |
| **forgetting** | 150 (**7.5 points**) | |

Learning the digits two at a time builds a dictionary of the same size as learning them mixed, and loses 150 of the
1,057 answers.

The split run prints its figure on the line `TEST after task 4`, and the regular test on its `test:` line.

The split run takes about 10 minutes. The regular run takes about 9½ minutes to train and half a minute to test.

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

Each task hits ~98–99% the moment it is trained, then drops to **0%** as the next task overwrites those weights.

By the measure above, the MLP forgets 75 points: 94.5% trained on all ten digits at once, 19.6% trained in turn. The
brain forgets 7.5 points: 52.9% with the digits mixed, 45.4% in turn.

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
| `mnist-1x60` | 1 × 60 | 1/1 in the last episode; 59/60 over the run | 34 | episode 34 | about 1 s |
| `mnist-10x20` | 10 × 20 | 10/10 in the last episode; 186/200 over the run | 216 | episode 20 (still changing) | about 3.5 s |
| `mnist-10x300` | 10 × 300 | 10/10 in the last episode; 2,986/3,000 over the run | 219 | episode 295 | about 1 minute |
| `mnist-100` | 100 × 1 | 18/100 | 289 | still changing | about 2 s |
| `mnist-255` | 255 × 1 | 95/255 | 642 | still changing | about 5 s |
| `mnist-1000` | 1,000 × 1 | 433/1,000 | 2,069 | still changing | about 25 s |

One image shown again and again is said right from the second episode on, and its dictionary settles at episode 34;
ten images are all right from the fourth episode on.

The data are the four raw (gunzipped) idx files of the standard MNIST set; their MD5 sums:

| file | MD5 |
|---|---|
| `train-images-idx3-ubyte` | `6bbc9ace898e44ae57da46a324031adb` |
| `train-labels-idx1-ubyte` | `a25bea736e30d166cdddb491f175f624` |
| `t10k-images-idx3-ubyte` | `2646ac647ad5339dbf082846283269ea` |
| `t10k-labels-idx1-ubyte` | `27ae3e4e09519cfbb04c329615203637` |
