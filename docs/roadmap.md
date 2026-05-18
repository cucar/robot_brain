# Short-Term Roadmap

---

## Metrics & Quick Wins

### Calculate up/down accuracy separately
- Report directional accuracy (up vs down) independently to identify prediction bias

### Rename test.js → run.js
- Rename `apps/stocks/jobs/test.js` to `run.js` to reflect its role as the primary entry point

---

## Neuron Limits

### Max neuron count hyperparameter
- Add configurable cap on neuron count per region/column

### Capacity enforcement
- When capacity is reached, stop learning new patterns
- Once forgetting frees space, resume learning automatically

### Overflow warning
- Emit warning when capacity is hit so the operator knows the brain is saturated
- Test that learning resumes correctly after decay opens capacity

---

## Exponential Temporal Binning Test

Implement the cortical temporal binning scheme described in [experiment-temporal-binning.md](./experiment-temporal-binning.md).

### Summary
Higher-level patterns currently store context at exact frame distances — meaningless precision at their timescale. Exponential bins give every level the same number of bins but scale bin width with level, letting higher-level patterns represent long-range temporal relationships without explosion in context entries.

### Key changes
- Context struct stores bin index instead of exact distance
- Bin conversion: `distanceToBin(distance, level, contextLength, numBins)`
- Pattern matching uses bin-space comparison
- Voting carries bin index instead of exact distance

### Validation
- Level-1 patterns behave nearly identically to current (regression)
- Higher-level patterns form with fewer, coarser context entries
- Prediction accuracy on existing benchmarks does not regress

---

## MNIST: Vision + Continual Learning Benchmark

### Why this matters

MNIST is the most widely recognized benchmark in machine learning. We use it twice, in sequence, to make two distinct architectural claims:

1. **Vanilla MNIST** — demonstrates that one prediction-only architecture handles stocks, text, and vision with zero modifications. Validates the "one substrate, multiple domains" claim.
2. **Split-MNIST (class-incremental)** — demonstrates continual learning without replay buffers, task IDs, regularizers, or any of the workarounds gradient-based models require. This is the headline result for external positioning.

The deeper claim Split-MNIST validates: **Robot Brain is constitutionally immune to catastrophic forgetting.** Neurons aren't overwritten by gradient updates; they're added and decayed independently. Transformers and MLPs trained sequentially on disjoint class subsets collapse to ~20% accuracy on old tasks (Hsu et al. 2018; van de Ven & Tolias 2019). This architecture should retain old-task accuracy by construction — patterns formed for digits 0/1 are not modified when the system later sees digits 2/3, because their context fingerprints don't match.

### Phase 1 — Vanilla MNIST (validate vision works at all)

Before tackling continual learning, confirm the architecture handles 10-way digit classification at all.

#### Approach — sequential pixel reading with action-based classification

Robot Brain cannot process static images. Each MNIST image (28×28 grayscale) is presented as a **784-frame episode**, reading pixels sequentially in raster scan order. Each frame contains a single grayscale value (0–255).

> **Single-dim model note**: under the current single-dimension-per-base-neuron rule, each pixel position is its own dimension (e.g. `pixel_12_5`), and a frame emits exactly one base neuron `{dimension: pixel_x_y, value: brightness}`. Multi-dim packing of position + color into one neuron is no longer supported.

Classification is modeled as an **action selection problem**:

* The brain has 10 possible actions (digits 0–9)
* At each frame (after passing the context length threshold), the brain outputs a digit prediction as its action
* During training, the correct digit action receives positive reward; incorrect actions receive negative reward
* Over many episodes, the brain learns which pixel-value sequences correlate with which digit actions
* The brain discovers spatial structure implicitly — e.g., dark pixels at positions spaced 28 apart represent a vertical line — without any geometric encoding

This is analogous to the stock channel (sequential price events → directional actions with reward) and the text channel (sequential character events → character actions with reward). Same mechanism, different domain.

#### Spatial feature discovery

The system has no knowledge that pixels are arranged in a 28×28 grid. All spatial relationships must be discovered through temporal prediction:

* **Horizontal adjacency**: consecutive pixels in the sequence (distance 1)
* **Vertical adjacency**: pixels 28 frames apart (distance 28) — requires sufficient context length
* **Diagonal features**: pixels at distance 27 or 29
* **Larger structures**: hierarchical pattern neurons combine lower-level detections across longer timescales

Context length directly determines what spatial features the system can discover. A context length of 100+ is recommended to capture cross-row relationships.

#### Parallelization variant — row-at-a-time (28 channels)

For practical compute reasons, a parallelized variant reads one full row per frame across 28 simultaneous channels (one per column position). This reduces episode length from 784 to 28 frames while preserving the need to discover vertical relationships across frames. This maps naturally to the multi-channel architecture already validated with 30 stock channels.

#### Training and evaluation

1. Generate episodes from the 60,000 MNIST training images — each image becomes one 784-frame episode (or 28-frame episode in the 28-channel variant)
2. Run multiple training passes (episodes repeated 10–100 times) with low forget rate to build stable representations
3. **Training accuracy**: percentage of training-set episodes where the brain's final-frame action matches the correct digit
4. **Test accuracy (generalization)**: present the 10,000 held-out test images as new episodes the brain has never seen. Brain continues learning during test (no freeze mode) — accuracy measured on **first exposure** to each test image, randomized order.

#### Compute requirements

* **Single-channel (784 frames/episode)**: 60,000 images × 100 episodes × 784 frames ≈ 4.7B frames. At 0.007ms/frame (Rust target) ≈ 9 hours. Requires Rust + Rayon threading.
* **28-channel (28 frames/episode)**: 60,000 images × 100 episodes × 28 frames ≈ 168M frames. Significantly more tractable — under 1 hour in Rust.
* **For comparison**: conventional CNNs train on MNIST in 2–5 minutes on GPU. The compute gap is expected and irrelevant to the architectural claim.

#### Recommended hyperparameters (starting point)

Based on stock and text experiments:

* Context length: 100 (single-channel) or 10 (28-channel variant)
* Forget rate: 0.001–0.01 (low, to retain learned digit patterns across episodes)
* Error threshold: 0.3 (same as text memorization experiments)
* Merge threshold: 0.9

#### Phase 1 success criteria

* Test accuracy meaningfully above chance (>50% as a soft floor; 80%+ would be excellent)
* Training accuracy converges to >90% with sufficient repetition (mirroring the text memorization curve: 41% → 100% over 3–5 episodes)
* No architectural changes vs stock/text channels — same brain code

This is not an attempt to beat CNNs. It is a demonstration that a single prediction-only architecture, designed for temporal sequences, can learn visual recognition without any vision-specific components. The benchmark exists to make the architectural claim legible to the ML community using a universally understood task.

### Phase 2 — Split-MNIST (class-incremental continual learning)

This is the **headline experiment for external positioning**. Run only after Phase 1 confirms vanilla MNIST works.

#### Protocol

Standard class-incremental Split-MNIST as defined in the continual learning literature (van de Ven & Tolias 2019; Hsu et al. 2018):

* **5 sequential tasks**, each containing 2 digit classes:
    - Task 1: digits 0, 1
    - Task 2: digits 2, 3
    - Task 3: digits 4, 5
    - Task 4: digits 6, 7
    - Task 5: digits 8, 9
* **No task IDs at training or test time** — the brain is never told which task it's on. This is the hardest variant; domain-incremental and task-incremental are easier and excluded from the headline.
* **Strict sequential training**: train Task 1 to convergence, freeze the experiment (no more Task 1 episodes), train Task 2 to convergence, etc. The brain only ever sees one task's data at a time.
* **Action space remains 10 digits throughout** — the brain must learn to never output a digit it hasn't seen yet, and to retain old digits as new ones are added.

#### Why this should work architecturally

The reasoning that motivates running this experiment:

* Sensory neurons for digit-0 patterns and digit-2 patterns have **disjoint context fingerprints** — the pixel sequences are different, so they activate different patterns.
* When training Task 2, no Task 1 patterns fire (their contexts don't match), so their action connections aren't modified.
* **Higher-level patterns live exponentially longer than lower-level ones.** The decay schedule is stratified by level: base sensory neurons decay fastest, level-1 patterns slower, level-2 patterns slower still, and so on. Whatever digit-0 patterns the brain consolidated up the hierarchy during Task 1 training have decay timescales that comfortably exceed the duration of Tasks 2–5.
* This means the relevant retention question is not "does the forget rate eat Task 1 patterns" but "did Task 1 train long enough to push patterns up to durable levels." If yes, retention is essentially free.
* Action connections on those high-level patterns are similarly persistent — they were established when the patterns were the strong predictors of the digit-0 reward, and nothing in Tasks 2–5 fires those patterns to modify them.

The risk shifts accordingly: it's not "forget rate too aggressive" but "Task 1 trained too briefly to consolidate to durable levels." This is a training-schedule question, not a decay-parameter question.

#### Evaluation

After each task is trained, measure accuracy on the held-out test sets of **all tasks seen so far**. Report:

1. **Final retention matrix** (5×5): accuracy on Task i after training through Task j, for all i ≤ j. The diagonal is "just-trained" accuracy; the bottom row is "after all training" retention.
2. **Average accuracy after Task 5**: mean of the bottom row. This is the headline number.
3. **Forgetting metric**: for each task, max accuracy ever achieved minus final accuracy. Lower is better.

#### Baselines (cited, not re-run)

Reference numbers from the continual learning literature for class-incremental Split-MNIST without replay or task IDs:

* **Naive sequential MLP/transformer fine-tuning**: ~20% average accuracy after Task 5 (van de Ven & Tolias 2019; Hsu et al. 2018). This is the catastrophic-forgetting floor.
* **EWC (Elastic Weight Consolidation)**: ~20–25%. Regularization-based methods barely help in class-incremental setting.
* **Replay-based methods (iCaRL, GEM)**: 70–90%, but these require storing old data and are explicitly outside the "no replay" claim.
* **Joint training upper bound**: ~98% (all tasks trained together — not a continual learning method).

We cite these numbers rather than re-running. Reproduction is a separate effort if external review demands it.

#### Phase 2 success criteria

* **Headline**: average accuracy after Task 5 substantially above the ~20% catastrophic-forgetting floor, achieved with **no replay buffer, no task IDs, no regularizers** — purely from architectural retention.
* **Stretch**: retention competitive with replay-based methods (60%+) without using replay.
* **Diagnostic**: per-task accuracy degradation profile. If Task 1 accuracy is preserved through Task 5, the architectural claim holds; if it decays, forget-rate tuning is needed.

#### Hyperparameter notes for Phase 2

The critical knob is **per-task training duration**, not forget rate. Each task must train long enough for digit patterns to consolidate up the hierarchy to levels with decay timescales exceeding the remaining experiment duration. Under-training a task leaves its representations at low levels that decay during subsequent tasks; over-training is harmless beyond saturation.

Recommended starting point:
* Match the per-task episode count to whatever achieved >95% within-task accuracy in Phase 1 (vanilla MNIST). If Phase 1 converged in 30 episodes, use 30+ episodes per task in Phase 2.
* Forget rate stays at the Phase 1 value (0.001–0.01). The level-stratified decay schedule does the work.
* If retention is weak, the first diagnostic is to inspect the level distribution of digit-0 patterns after Task 1 training. If they're concentrated at low levels, train Task 1 longer. If they're at high levels and still being lost, then forget rate is the issue.

#### What this is NOT

Not an attempt to beat continual learning SOTA methods that use replay. The claim is structural: we get retention from the architecture, not from buffering or regularizing. A 60% result with no replay is a stronger paper than a 90% result with replay, because the former is a different category of system.

#### Dependencies

* Phase 1 (vanilla MNIST) complete and working
* Same Rust core, same channel code — no architectural changes
* Sequential task scheduler (trivial — just feed task data in order)
* Per-task evaluation harness (trivial)

---

## Documentation & Publish

### Update all documentation
- Sync docs with current architecture post-Rust migration
- Update README demos and examples

### npm package
- Prepare package for publication
- Publish to npm registry