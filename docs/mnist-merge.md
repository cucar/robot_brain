# MNIST Branch Reconciliation

---

## Why

The `mnist` branch has accumulated substantial work alongside experimental v1 moment-neuron scaffolding. Before any new structural work lands on top of it, the keepable parts must be merged into `main`. Merging late would compound conflicts with later phases.

This is a prerequisite for everything downstream — inference-scope experimentation, spatial processing, neuron reuse.

---

## What this initial app is: Naive Bayes

The MNIST work being merged here is the **sensory-only** iteration. Spatial processing and neuron reuse are not yet implemented, so the only voters are the active sensory neurons — one channel per pixel. Each neuron learns, per digit, the fraction of that digit's training images in which it appears, and at inference every active pixel casts a vote weighted by those learned percentages. Votes are summed across channels and the highest-scoring digit wins.

Structurally, this is **Naive Bayes** with independent Bernoulli pixel features. Each pixel contributes evidence about the class on its own; nothing in this iteration represents the *joint* — there is no mechanism for "pixel A and pixel B together mean something different than either alone." That is the defining limitation of the naive conditional-independence assumption, and the sensory-only app sits exactly on it by construction.

This is worth naming explicitly because it sets the honest reference frame for everything downstream:

- **The sensory-only app is the zeroth-order / degenerate case of the full architecture.** It is what Robot Brain reduces to when neurons are forbidden from binding combinations (no spatial processing) and forbidden from re-allocating to error patterns (no reuse). The two deferred workstreams are precisely what *relax* the independence assumption — spatial processing manufactures conjunctive features (encoding `P(class | A∧B)` rather than `P(class|A)·P(class|B)`), and neuron reuse allocates representational capacity onto the error manifold (residual-fitting, boosting-like, but done structurally through neuron allocation rather than reweighting).
- **The accuracy ceiling of this iteration is the Naive Bayes ceiling, not the architecture's.** Plain NB on full-resolution 28×28 MNIST lands in the **low-to-mid 80s** (~83–84% test, varying a few points with binarization threshold and smoothing). That is the bar this app is structurally capped at, and the bar the full architecture must *beat using a different mechanism* to make its point.

### Initial test result

The initial sensory-only run at **7×7 binary** behaves as the Naive Bayes characterization predicts: it works (accuracy meaningfully above chance) but is inefficient and well short of full-resolution NB. At 7×7 binary, downsampling discards most of the discriminative signal (3/8/9 collapse into similar blobs) and independent voting can't recover configural structure, so the expected band is roughly **~45–60% test**, with train only marginally higher — 49 independent histograms have almost no capacity to overfit 60k examples, so the train/test gap stays small. This is consistent with what was observed: correct in mechanism, capped in accuracy, and confirming the pipeline rather than the architecture.

**Implication for validation.** A result near the NB band confirms *correctness of the merge and the voting integration* — nothing more. The architectural accuracy gain does not come from resolution (7×7 → 28×28) or color depth (binary → 256), but almost entirely from the spatial processing and neuron reuse deferred to later workstreams. Treat this iteration's number as a plumbing check, and log the **per-digit confusion matrix** rather than only aggregate accuracy: with independent voting, the digit pairs that collapse are exactly the evidence for *why* spatial processing is needed.

> Open verification before leaning on the "degenerate case of NB" framing in any external writeup: confirm that the cortex's intersection/union operation genuinely composes joint statistics rather than re-summing marginals. The cheapest falsifying test is a two-pixel XOR case (A and B individually uninformative, their XOR perfectly predictive) computed by hand — the canonical problem NB cannot learn and a conjunctive feature can. If a moment neuron solves XOR, the framing is load-bearing; if it can't, the gain mechanism is additive under the hood and the story needs revising. This belongs in spatial-processing validation, not here, but it is the hinge the whole NB-relaxation narrative rests on.

---

## Naive MNIST app

**Goal.** Stand up the **sensory-only (Naive Bayes) MNIST app** described at the top of this document. The original `apps/mnist/encoder.js` was a single-channel pixel-stream encoder that fed pixels temporally one frame at a time — a sequence-learning shape that doesn't match how the NB voting setup needs to see an image. The refactor replaces it with a per-pixel-position (retinotopic) encoder, so a whole image becomes **one frame** in which every pixel channel fires concurrently and votes are aggregated across channels into the shared digit action — the exact voting structure the intro characterizes as Naive Bayes. The branch also adds two siblings — a row-channel variant and the digit label encoder — and rewrites `apps/mnist/jobs/test.js` around the new shape. `dump_image.js` was added to convert MNIST images into the text-pipeline format for cross-app inspection.

This is what the merged app does **now**, in its sensory-only form. It is the degenerate, pre-spatial-processing iteration of the full retinotopic-channels architecture targeted in roadmap step 5 (Vanilla MNIST). The same channel layout, encoder, and harness carry through unchanged; the accuracy gain in step 5 comes from spatial processing and reuse landing in steps 3–4, not from further encoder work.

### Retinotopic parallel channels

Rather than scanning pixels sequentially (which imposes an arbitrary temporal order on spatial data), every pixel position is its own parallel channel — analogous to a retinotopic map where each spatial position has a dedicated cortical column.

* **One channel per pixel position**, all running in parallel. At full resolution that's **784 channels** (28×28); at 7×7 it's 49.
* **Sensory neurons per channel** = the quantization bucket count (see below).
* **10 shared action neurons** for digit classification (digits 0–9), aggregating votes from every channel.

Each pixel-column doesn't "know" it's part of a grid — it only knows what value it sees, and what reward followed when it fired during a labeled training image. This mirrors how the visual cortex maps spatial positions to cortical columns.

### Single-frame episode structure

The earlier plan presented each image *twice* — a two-frame repetition trick to fake temporal co-occurrence so a sequence-learning architecture could ingest spatial data. That is no longer needed: a whole image is **one frame**, every pixel channel fires its quantized value simultaneously, and the harness reads the brain's digit prediction (during eval) or supplies the correct digit via `learn()` (during training). There is no closed-loop reward delivery through `process_frame` — supervision happens entirely through `learn(actions, rewards, distance=1)` with the one-entry digit-channel map, which accumulates a positive reward onto the active-voter → correct-digit connections at the standard same-frame voting slot (distance 1). The brain's actual prediction during training does not influence wiring: every training image gets a `learn()` call with the labeled correct digit, regardless of what the brain predicted.

In this sensory-only iteration there is no inter-channel connection formation — that's what step 3 (spatial processing) unlocks. The only connections being learned in this app are **sensory → action**: each pixel-value sensory neuron strengthens its weighted vote toward the correct digit action whenever it co-fires with a positive-reward label. Because the wiring is additive cumulative-reward accumulation (see the **Supervised learn() + non-learning mode** project), the strength a given pixel-value→digit connection accumulates is just a count — effectively the *fraction of that digit's training images in which that pixel-value fired*, which is exactly the per-class frequency framing the intro uses to characterize this as Naive Bayes. Independent per-pixel evidence summed across channels into a shared digit decision.

### Phased sensory quantization — binary first

Robot Brain learns through co-activation frequency, not gradient averaging. Sensory precision trades off directly against pattern stability: with 256 grayscale buckets, two handwritten "3"s that differ by a few brightness levels at a few pixels become completely different activation sets, fragmenting the representation. With binary, those same two "3"s collapse into the identical activation pattern, and one training example reinforces the next.

This mirrors biological vision: retinal ganglion cells don't transmit raw luminance — they transmit contrast, edges, on/off transitions. The brain aggressively compresses before pattern formation.

**Phase A — Binary (2 buckets).** Threshold to black/white. At 28×28 that's 1,568 sensory neurons total (784 × 2) — orders of magnitude more tractable than 200K. Maximum overlap between examples of the same digit, minimal entropy, fastest stabilization. The architectural proof of concept. If binary fails, the issue is architectural; if binary succeeds, the core mechanism is validated. The initial run is at **7×7 binary** (49 channels × 2 = 98 sensory neurons), which is what the intro's ~45–60% expected band is calibrated to.

**Phase B — 4 or 8 buckets.** Black / dark gray / light gray / white, adding stroke thickness and anti-aliasing structure without exploding the representation space. ~3,136 neurons at 4 buckets, ~6,272 at 8 (full 28×28).

**Phase C — 16 buckets.** Likely enough precision for anything useful in MNIST. ~12,544 neurons at full resolution. 256 buckets are unlikely to help and may actively hurt generalization through fragmentation.

**The quantization curve itself is a publishable result** — it characterizes how non-gradient architectures interact with sensory resolution. Note that the *NB-ceiling* portion of that curve is what this step measures; the *post-NB climb* portion is step 5's job.

### Output mechanism — shared action neurons

A single set of 10 action neurons (digits 0–9) aggregates votes from every pixel channel. Individual pixel positions carry vastly different amounts of information about digit identity — a corner pixel that's always black has no discriminative power. The shared voting pool lets the system naturally weight contributions: channels with strong, reward-reinforced action connections dominate; uninformative channels contribute noise that washes out.

This mirrors the stock experiment, where multiple stock channels vote on a shared up/down action space and the consensus extracts signal from the aggregate.

### Training and evaluation

1. Generate episodes from the 60,000 MNIST training images — each image becomes one single-frame episode across all pixel channels.
2. Train: for each image, call `process_frame(image)` (activates the corresponding pixel-value sensory neurons), then call `learn(actions, rewards, 1)` with the digit channel mapping to the labeled correct digit and a positive reward to additively wire every currently-active sensory neuron to that digit's action neuron at distance 1. Multiple passes (10–100×) over the training set with forget rate 0 (see hyperparameters).
3. **Training accuracy**: percentage of training episodes where the brain's post-`process_frame` prediction matches the label, measured *before* the `learn()` call updates the wiring.
4. **Test accuracy**: switch to non-learning mode, then present the 10,000 held-out test images via `process_frame` (no state change). Accuracy measured on first exposure to each test image, randomized order.

### Compute requirements

Compute scales with quantization level and resolution:

* **7×7 binary** (initial run): 98 sensory neurons, ~1k sensory→action connections. Trivial; runs in seconds on CPU.
* **28×28 binary**: 1,568 sensory neurons. 60,000 images × 100 passes = 6M frames. Highly tractable with the small neuron count; fast enough to iterate on hyperparameters rapidly.
* **Higher buckets**: scales linearly. 4 buckets ≈ 2× binary; 16 buckets ≈ 8× binary.
* Requires Rust + Rayon threading at full resolution. Start small (e.g. 1,000 images at 7×7) to calibrate timing.
* **For comparison**: conventional CNNs train on MNIST in 2–5 minutes on GPU. The compute gap is expected and irrelevant to the architectural claim.

### Recommended hyperparameters (starting point)

* Context length: 1 (single frame per episode)
* Forget rate: **0**. Initial runs do no forgetting at all — at 28×28 binary the entire sensory population is 1,568 known-in-advance neurons, so capacity is not a concern and we want the per-pixel-per-digit counts to accumulate cleanly.
* Action wiring: additive cumulative-reward accumulation via `learn()` (set by the **Supervised learn() + non-learning mode** project) — no static alpha, no smoothing.
* Error threshold and merge threshold are pattern-formation knobs that only come into play with spatial processing (step 3). Leave them at defaults for this app — they have no effect at this phase.

### Surface

- `apps/mnist/encoders/pixel_channels_encoder.js` — **the workhorse for the sensory-only NB app.** One channel per pixel position (28→784, 14→196, 7→49), block-average downsampling for smaller sizes, configurable bucket count. A whole image is one frame; every pixel channel fires its quantized value concurrently; the shared digit channel carries the action. The initial 7×7 binary run uses this encoder.
- `apps/mnist/encoders/row_channels_encoder.js` — 28 row-channels presented column-by-column over 28 frames. A temporal-shape comparison variant; not the NB setup, kept for the column-scan experiment.
- `apps/mnist/encoders/digits.js` — shared digit label / action constants used by every encoder.
- `apps/mnist/encoder.js` — deleted. The old single-channel pixel-stream encoder; superseded by the per-pixel-channel encoder, which is the right shape for sensory-only voting.
- `apps/mnist/jobs/test.js` — rewritten around the new encoders (~600 lines diff, mostly rewrite-in-place). Calls `learn()` and uses non-learning mode for evaluation, so **Supervised learn() + non-learning mode** must land first.
- `apps/mnist/dump_image.js` — image → text-pipeline format, for cross-app inspection of individual digits.

**Decision: Keep all.** `pixel_channels_encoder` is the sensory-only NB app itself; the others are the surrounding scaffolding (label encoder, comparison variant, inspection tool) that the same harness needs.

### Test plan and acceptance

- **T-encoder** — run `pixel_channels_encoder` against one canonical MNIST test image at 7×7 binary, confirm the output bit vector matches the expected encoding. Doesn't exercise the brain; just confirms the encoder is producing the right frame shape for NB voting.
- **T-mnist-nb-baseline** — run `apps/mnist/jobs/test.js` at 7×7 binary: train via `process_frame` + `learn()`, evaluate via `process_frame` in non-learning mode, confirm test accuracy lands in the predicted **~45–60%** band with a small train/test gap. This is the Naive-Bayes plumbing check described in the intro, not an architectural validation. Log the per-digit confusion matrix alongside the aggregate number — the 3/8/9 collapses are the evidence that motivates the spatial-processing workstream.
