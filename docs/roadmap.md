# Short-Term Roadmap

---

## MNIST: Vision + Continual Learning Benchmark

### Why this matters

MNIST is the most widely recognized benchmark in machine learning. We use it twice, in sequence, to make two distinct architectural claims:

1. **Vanilla MNIST** — demonstrates that one prediction-only architecture handles stocks, text, and vision with zero modifications. Validates the "one substrate, multiple domains" claim.
2. **Split-MNIST (class-incremental)** — demonstrates continual learning without replay buffers, task IDs, regularizers, or any of the workarounds gradient-based models require. This is the headline result for external positioning.

The deeper claim Split-MNIST validates: **Robot Brain is constitutionally immune to catastrophic forgetting.** Neurons aren't overwritten by gradient updates; they're added and decayed independently. Transformers and MLPs trained sequentially on disjoint class subsets collapse to ~20% accuracy on old tasks (Hsu et al. 2018; van de Ven & Tolias 2019). This architecture should retain old-task accuracy by construction — patterns formed for digits 0/1 are not modified when the system later sees digits 2/3, because their context fingerprints don't match.

### Phase 1 — Vanilla MNIST (validate vision works at all)

Before tackling continual learning, confirm the architecture handles 10-way digit classification at all.

#### Approach — retinotopic parallel channels with action-based classification

Rather than scanning pixels sequentially (which imposes an arbitrary temporal order on spatial data), the architecture treats every pixel position as its own parallel channel — analogous to a retinotopic map where each spatial position has a dedicated cortical column.

**Architecture:**

* **784 channels** (one per pixel position in the 28×28 grid), running in parallel
* Sensory neurons per channel depend on quantization level (see phased quantization below)
* **10 shared action neurons** for digit classification (digits 0–9), aggregating votes from all channels

This is analogous to how the visual cortex maps spatial positions to cortical columns. Each pixel-column doesn't "know" it's part of a grid — it only knows what value it sees and, through learned connections, what its neighbors' values are.

#### Phased sensory quantization — binary first

Robot Brain learns through co-activation frequency and combinatorial reuse, not gradient averaging. Sensory precision directly trades off against pattern stability: with 256 grayscale buckets, two handwritten "3"s that differ by a few brightness levels at a few pixels become completely different activation sets, fragmenting the representation. With binary, those same two "3"s likely collapse into the identical activation pattern, and one training example reinforces the next.

This mirrors biological vision: retinal ganglion cells don't transmit raw luminance — they transmit contrast, edges, on/off transitions. The brain aggressively compresses before pattern formation. If Robot Brain needs the same compression to work well, that's convergent design, not a limitation.

The connection density argument is decisive. With binary, each pixel has 2 possible neurons, so a connection between two pixels has 4 possible state combinations — all of which recur constantly, building strong statistics fast. With 256 buckets, the same two pixels have 65,536 combinations, most seen rarely or once. The system would need astronomically more training data to build stable connections.

**Phase A — Binary (2 buckets)**

Threshold MNIST to black/white only. This gives:

* **1,568 total sensory neurons** (784 × 2) — orders of magnitude more tractable than 200K
* Maximum overlap between examples of the same digit class
* Minimal entropy, fastest connection stabilization
* Smallest possible connection space — the architectural proof of concept

If binary fails, the issue is architectural. If binary succeeds, the core mechanism is validated.

**Phase B — 4 or 8 buckets**

Quantize to coarse levels (e.g., black / dark gray / light gray / white). This adds stroke thickness information, anti-aliasing structure, and soft edge detail without exploding the representation space. ~3,136 sensory neurons at 4 buckets, ~6,272 at 8.

**Phase C — 16 buckets**

Likely enough precision for anything useful in MNIST. ~12,544 sensory neurons. 256 buckets are unlikely to provide additional benefit for this architecture and may actively hurt generalization through fragmentation.

**The quantization curve itself is a publishable result.** It characterizes something fundamental about how non-gradient architectures interact with sensory resolution — the optimal quantization level reveals the architecture's natural operating point for balancing discriminative power against pattern stability.

#### Two-frame episode structure

A static image cannot drive connection-based learning in a single frame, because Robot Brain learns by observing temporal co-occurrences across frames. The solution:

* **Frame 1**: all 784 channels fire their respective grayscale values simultaneously. This populates the sensory state across the entire "retina."
* **Frame 2**: the identical image is presented again. Now each channel can observe what fired in the *previous* frame across all other channels, enabling inter-channel connection formation.
* After frame 2, the brain outputs a digit prediction via the shared action neurons.
* Reward is delivered: positive for correct digit, negative for incorrect.

The repetition is the minimal structure needed for Robot Brain's connection mechanism to operate on spatial data. Each pixel-column learns: "when I saw value X, and my neighbors saw values Y, Z, W in the previous frame, the correct digit was D."

#### What gets learned

The sensory neurons form connections to neurons in other channels based on co-activation across frames. In practice, connection creation is demand-driven (only when co-activation occurs), so the actual connection count is a fraction of the theoretical maximum. With binary quantization, the connection space is highly tractable — 1,568 neurons with ~2.5M possible connections, most of which will be frequently reinforced. The system learns:

* **Local spatial correlations**: pixel (14,14) being dark while pixel (14,15) is also dark is a learned association, not a geometric prior
* **Global digit signatures**: the full constellation of pixel activations that characterize each digit class
* **Discriminative features**: through reward, the system reinforces connections whose activation patterns reliably predict specific digits

The brain discovers that certain combinations of pixel values across specific spatial positions predict specific digits — without ever being told that pixels are arranged in a grid, or that adjacent pixels tend to co-vary.

#### Output mechanism — shared action neurons

A single set of 10 action neurons (digits 0–9) aggregates votes from all 784 channels. This is critical: individual pixel positions carry vastly different amounts of information about digit identity. A corner pixel that's always black has no discriminative power. The shared voting pool lets the system naturally weight contributions — channels with strong, reward-reinforced action connections dominate the vote; uninformative channels contribute noise that washes out.

This mirrors the stock experiment architecture where multiple channels (stocks) vote on a shared action space (up/down), and the consensus mechanism extracts signal from the aggregate.

#### Training and evaluation

1. Generate episodes from the 60,000 MNIST training images — each image becomes one 2-frame episode across 784 parallel channels
2. Run multiple training passes (episodes repeated 10–100 times) with low forget rate to build stable inter-channel representations
3. **Training accuracy**: percentage of training-set episodes where the brain's post-frame-2 action matches the correct digit
4. **Test accuracy (generalization)**: present the 10,000 held-out test images as new 2-frame episodes the brain has never seen. Brain continues learning during test (no freeze mode) — accuracy measured on **first exposure** to each test image, randomized order.

#### Compute requirements

Compute scales directly with quantization level:

* **Binary (Phase A)**: 1,568 sensory neurons, ~2.5M possible connections. 60,000 images × 100 passes × 2 frames = 12M episodes. With the small neuron count and high connection reuse, this should be highly tractable — fast enough to iterate on hyperparameters rapidly.
* **Phase B/C**: scales linearly with bucket count. 4 buckets ≈ 2× binary compute; 16 buckets ≈ 8× binary compute. Still far more tractable than the original 256-bucket design.
* Connection processing is the bottleneck: each channel checks connections to other channels' previous-frame activations. With binary, connection density saturates quickly (only 4 combinations per pixel pair), so per-frame cost stabilizes early.
* Requires Rust + Rayon threading. Start with Phase A on a small subset (1,000 images) to calibrate timing.
* **For comparison**: conventional CNNs train on MNIST in 2–5 minutes on GPU. The compute gap is expected and irrelevant to the architectural claim.

#### Recommended hyperparameters (starting point)

Based on stock and text experiments:

* Context length: 2 (only two frames per episode — context is spatial, not temporal)
* Forget rate: 0.001–0.01 (low, to retain learned digit patterns across episodes)
* Error threshold: 0.3 (same as text memorization experiments)
* Merge threshold: 0.9

#### Phase 1 success criteria

* **Phase A (binary)**: test accuracy meaningfully above chance (>50% as a soft floor; 70%+ would validate the mechanism). This is the gate — if binary fails, debug the architecture before adding precision.
* **Phases B/C**: test accuracy improves over binary as quantization adds discriminative detail, peaking at some optimal bucket count. 80%+ at the optimal level would be excellent.
* Training accuracy converges to >90% with sufficient repetition at the optimal quantization level
* No architectural changes vs stock/text channels — same brain code, same connection mechanism, just more channels
* Inter-channel connections demonstrably encode spatial structure (inspectable: which pixel positions form strong connections should roughly correspond to spatial proximity and shared digit-class membership)
* The quantization-vs-accuracy curve is documented as an empirical result characterizing the architecture's sensory resolution tradeoff

This is not an attempt to beat CNNs. It is a demonstration that a single prediction-only architecture, designed for temporal sequences, can learn visual recognition through spatial co-occurrence across parallel channels — without any vision-specific components. The retinotopic channel layout is the most biologically plausible framing: it mirrors cortical organization rather than imposing an arbitrary scan order. The benchmark exists to make the architectural claim legible to the ML community using a universally understood task.

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

* Sensory neurons for digit-0 patterns and digit-2 patterns have **disjoint context fingerprints** — the spatial co-activation patterns across 784 channels are different, so they activate different inter-channel connections.
* When training Task 2, no Task 1 patterns fire (their spatial contexts don't match), so their action connections aren't modified.
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

## Stock Metrics - Calculate up/down accuracy separately

- Report directional accuracy (up vs down) independently to identify prediction bias

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

## Documentation & Publish

### Update all documentation
- Sync docs with current architecture post-Rust migration
- Update README demos and examples

### npm package
- Prepare package for publication
- Publish to npm registry