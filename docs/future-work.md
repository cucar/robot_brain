# Future Work

## MNIST: Vision + Continual Learning Benchmark (post-Rust Phase 3)

### Why this matters

MNIST is the most widely recognized benchmark in machine learning. We use it twice, in sequence, to make two distinct architectural claims:

1. **Vanilla MNIST** — demonstrates that one prediction-only architecture handles stocks, text, and vision with zero modifications. Validates the "one substrate, multiple domains" claim.
2. **Split-MNIST (class-incremental)** — demonstrates continual learning without replay buffers, task IDs, regularizers, or any of the workarounds gradient-based models require. This is the headline result for external positioning.

The deeper claim Split-MNIST validates: **Robot Brain is constitutionally immune to catastrophic forgetting.** Neurons aren't overwritten by gradient updates; they're added and decayed independently. Transformers and MLPs trained sequentially on disjoint class subsets collapse to ~20% accuracy on old tasks (Hsu et al. 2018; van de Ven & Tolias 2019). This architecture should retain old-task accuracy by construction — patterns formed for digits 0/1 are not modified when the system later sees digits 2/3, because their context fingerprints don't match.

### Phase 1 — Vanilla MNIST (validate vision works at all)

Before tackling continual learning, confirm the architecture handles 10-way digit classification at all. This is the "baby step."

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
* Forget rate decays unused connections slowly; with low forget rate (0.001–0.01) and the timescale of 5 sequential tasks, Task 1 patterns should still be intact when re-tested after Task 5.
* Action-to-digit connections established during Task 1 are not overwritten by Task 2 training because the activating patterns are different.

The risk: if forget rate is too aggressive, old-task patterns decay during the long stretch of new-task training. This is the experiment's failure mode and the parameter to tune.

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

Forget rate is the critical parameter. Recommend running a small sweep (0.0001, 0.001, 0.01) to characterize the retention/plasticity tradeoff. The lower the forget rate, the better the retention but the slower new-task learning.

#### What this is NOT

Not an attempt to beat continual learning SOTA methods that use replay. The claim is structural: we get retention from the architecture, not from buffering or regularizing. A 60% result with no replay is a stronger paper than a 90% result with replay, because the former is a different category of system.

#### Dependencies

* Phase 1 (vanilla MNIST) complete and working
* Same Rust core, same channel code — no architectural changes
* Sequential task scheduler (trivial — just feed task data in order)
* Per-task evaluation harness (trivial)

## Python Bindings + PyPI (~1 week when prioritized)

Expose the Rust core to Python for broader adoption.

### Python bindings via PyO3/maturin
- Wrap `brain-core` Rust library with PyO3
- Pythonic API matching the Node.js wrapper patterns
- Build and publish to PyPI via maturin

### Python channel interface
- Python equivalent of Channel base class
- Example: stock channel in Python

---

## MPI Distribution (when multi-server budget available)

Distribute across multiple machines for large-scale workloads.

### Add MPI layer for inter-column communication
- Each MPI rank runs one `Region`
- MPI messages: vote broadcasts, neuron migration, consensus sync
- Use `rsmpi` crate for Rust MPI bindings

### Neuron metadata storage for MPI
- Each MPI rank's Region holds full metadata for its own neurons
- Read-only cache of metadata for foreign neurons it has connections to
- Cache populated on creation — when a new neuron is created, the creating rank broadcasts its metadata once via MPI
- No ongoing synchronization needed since all metadata is immutable after creation
- Lookup interface unchanged (`get_channel(neuron_id)`, etc.) — only backing storage is distributed

### Global consensus protocol
- Each column produces local vote aggregation
- MPI AllReduce or custom gather for global consensus
- Actions executed by rank 0 (or designated I/O rank)

### Neuron partitioning and migration
- Sensory neurons assigned to columns by channel/dimension hash
- Pattern neurons live on same column as parent
- Migration protocol for rebalancing load across columns

---

## Transfer Learning

### Experiment design
- Two sets of stocks: A and B
- Measure accuracy of B with no prior knowledge (baseline)
- Learn a few episodes on A
- Measure accuracy of B after learning from A
- To test: send B data as if it were A (same channel/dimension mapping) so learned patterns apply

### What this validates
- Whether temporal patterns learned from one stock transfer to another
- Whether the brain generalizes across similar time series or overfits to specific value ranges

---

## Pattern Efficiency

### Current behavior
- Error correction creates a higher-level pattern that captures the ENTIRE context and learns the ENTIRE inference across all channels/dimensions — even the ones the parent got right

### Proposed optimization
- When the parent gets a channel wrong, delegate inference of that channel to the higher-level child pattern
- When the parent gets a channel right, do NOT delegate — parent keeps inferring correctly
- This is more efficient but significantly more complicated
- Open question: does selective delegation improve accuracy or just efficiency?

---

## Better Debugging / Pattern Explanation

### Problem
- When printing a pattern, we should show it with its ancestors (which need to be active)
- Each node carries exponentially more sensory neurons as you go up the levels, representing different branches
- Not clear how to present this concisely

### Ideas
- Tree visualization of active pattern chains (sensory → level 1 → level 2)
- Collapse branches that share common sensory roots
- Highlight which dimensions each level added to the prediction

---

## Text Channel

### Core concept
- Verify the algorithm can memorize and reproduce text
- Events and actions both correspond to characters
- Each frame: one character neuron sent as event, one character neuron sent back as action
- During training, reward based on how closely returned actions match actual input text
- The system learns to "say things" in terms of how it encountered them before

### Training protocol
- Training data includes query and response
- During query phase: no actions expected
- During response phase: rewards based on matching the expected response
- Keep repeating until 100% accuracy on all training data
- Brain learns action sequences based on event sequences

### Key architectural insight
- Actions must behave exactly like events — they need to form patterns and infer what comes next
- Executed actions get fed back as events (feedback loop): when I output character X, I should see it as input Y
- This creates connections from action X to event Y — the foundation of action sequences
- Actions are rewarded based on the sequences they produce

### Math as text
- Same text channel handles arithmetic
- Learning the concept of "2": same things side by side, different frames, build associations
- Train on math equations — should be able to solve math problems from text
- One character at a time, input and output — it's a text interface to the brain

### Scaling implications
- When scaled up, this starts looking like conscience/inner monologue
- The input text is conversation/commands sent through screens remotely
- The actions are its responses
- Rewards can be given like "good job" or "don't do that again" via remote during training

### Integration with stock channel
- Test how text channel interacts with stock channel running simultaneously
- Both channels feed into the same brain — cross-channel pattern formation

---

## Vision Channel

### Core approach
- Reference: https://claude.ai/share/5ac65464-6293-4cad-9683-07f0bd135644
- Under the single-dimension-per-base-neuron rule, each (x, y) screen position becomes its own dimension (e.g. `pixel_12_5`), and the base neuron carries `{dimension: pixel_x_y, value: brightness}`
- Color (grayscale 0–256 to start) is the value; for RGB, each channel becomes its own dimension family (`pixel_x_y_r`, `_g`, `_b`)
- 100×100 grayscale camera = 10,000 dimensions × 256 values = ~2.5M possible base neurons
- No explicit relationships between inputs — the brain forms them via temporal prediction models

### Biological reference
- Inputs: 130M cells in retina + optic nerve
- Photoreceptor cells: rod cells (dark/grayscale brightness), cone cells (light/RGB color)
- Bipolar cells → Ganglion cells → optic nerve → visual cortex (V1, V2, V4)
- Horizontal cells between photoreceptors (light changes)
- Amacrine cells between ganglion cells

### Outputs (actions)
- Direction of eye (pupil movement / saccades)
- How much light to let in (brightness — iris)
- Near/far focus (zoom — lens)

### Open problems
- How to build a complete picture from small frames
- How to build depth perception from 2 eyes

### Spatial processing (spatio-temporal pooling)
- Channel emits one base neuron per pixel as `{dimension: pixel_x_y, value: brightness}` (or per-color-channel variant for RGB)
- Brain activates neurons corresponding to differences between these per-pixel dimensions
- Position is encoded structurally in the dimension name; brightness is the value — no multi-dim packing
- This is done BEFORE temporal recognition — spatial level processing within same frame and age
- Two-level processing: spatial levels and temporal levels
- For spatial pooling, add x and y distances to connections table alongside temporal distance
- Each neuron has 8 spatial connections (neighborhood)
- Apply peak/pattern detection recursively based on spatial distances

### Value encoding (applies to all channels)
- Channels should NOT do encoding like slope categorization or discretization
- Brain should automatically convert:
  1. Take all active neurons in frame, group by dimension
  2. Calculate differences of values between new and older neurons within the same dimension
  3. Activate neurons corresponding to those differences
- For stocks: neurons for differences in price and volume
- For vision: neurons for pixel differences across frames
- Open question: rounding/bucketing/discretization of differences — maybe a hyperparameter (difference match threshold)
- Consider: absolute neurons (with coordinates) vs relative neurons (representing connections between absolute neurons)
- Dynamic discretization using mean and variance matching
- Associative pooling — may not be used for temporal pooling (distance=0 not used) but likely useful for spatial pooling

### Training
- Video data: teach to focus on moving objects (evolutionary priority — watch out for moving things)
- Saccade/zoom training: when there's an object that needs to be recognized, zoom to it
- Training data: video zooming to an object, moving or rotating to better recognize it

---

## Audio Channel

### Core concept
- Microphone interface values become event neurons (amplitude and other metrics representing audio events)
- Speaker interface values become action neurons (commands to the speaker)
- Same value representation as other channels — brain handles the temporal patterns

---

## Extremities (Touch / Motor Control)

### Core concept
- Moving devices: arms, legs
- Event neurons: touch sensor interface values
- Action neurons: "muscle" contractions
- Touch carries rewards — can be great or very bad
- Brain learns to move and avoid danger through reward signals

---

## Hippocampal Region (Thinking, Long-Term Memory, Metacognition)

The largest remaining architectural component. A second organ — the **executor** — that operates on cortical neurons in parallel with the cortex's own perception/action loop. Implements long-term memory, thinking-as-action, and offline policy improvement.

See: [Hippocampus Design and Implementation Plan](./hippocampus.md)

### Summary

Robot Brain is a dual-system architecture. **Cortex splits reality into patterns; hippocampus groups patterns to generalize.**

* **One substrate, two operators.** Storage is always cortical. The hippocampus is the executor — it does not hold moments, it mints them.
* **Moment and class neurons live in cortical columns** alongside sensory and pattern neurons, addressed via the same thalamic id↔property bus. A moment is a tight-tolerance cortical pattern neuron minted in one shot from a salient frame; a class is a looser-tolerance neuron grouping similar moments.
* **Thinkability = reachability by the executor.** A neuron is thinkable iff the hippocampus can reach it — not a property of the neuron's location or kind.
* **Selective minting** via z-scored reward and prediction error — only salient frames produce moments. Novelty is not an input.
* **Hippocampus runs at a faster clock** than the cortex, executing many experiment frames per cortex frame.
* **Decided actions, not probabilistic ones.** Sensory and pattern neurons accumulate statistical action votes. Moment and class neurons carry *decided* actions written by the hippocampus after experimentation. When activated, a decided vote dominates probabilistic votes for the same action slot.
* **Action writeback closes the loop.** Experiment conclusions are written directly onto the moment (or class) neuron whose context they apply to. From then on, that neuron fires reflexively on context match and votes its decided action into cortex's action selection. **System 2 outputs become System 1 reflexes.** This is the architectural substrate for expertise development.
* **Recall is by context, not by handle.** The executor's primitive is "drive a partial context and let cortex pattern-complete."
* **Heterarchy at the moment layer** — one cortical neuron can be inside many moments and many classes; one moment can be a child of many classes.
* **Single-track experiments first; parallel later.** Phase 1 uses a stack (branching = push, terminate = pop, higher-priority think-actions interrupt). Concurrent experiment pools are deferred to Phase 9.
* **Same death ledger, different aging.** Moment and class neurons decay on their own timelines (classes slower than moments), but use one shared ledger.

### Why this matters

This is the architectural commitment that separates Robot Brain from sequence-model-based AI:

* Long-term memory as cortical neurons minted by deliberate experimentation — not a context window, not a separate store.
* Thinking as an action in the policy — not a hardcoded mode.
* Offline policy improvement via counterfactual replay through class siblings.
* Metacognitive control learned through reward signals — the agent learns when to think and when to react.
* **HM falls out as a direct architectural prediction.** Removing the executor leaves cued recall and skills intact but kills new-moment formation, deliberate recall, and counterfactual reasoning.

Built on hippocampal indexing theory (Teyler & DiScenna 1986; Teyler & Rudy 2007), complementary learning systems (McClelland et al. 1995), Mattar & Daw's expected-value-of-backup model (2018), and Kahneman dual-process theory implemented as architecture rather than metaphor.

### Implementation phases

1. Hippocampus skeleton with parallel-clock plumbing — moment and class neuron kinds wired into existing columns, decided-action writeback verified end-to-end
2. Salience module and selective minting (z-scored reward and prediction error)
3. Class formation (online agglomerative weighted Jaccard, multi-level hierarchy)
4. Temporal moment graph and basic replay
5. Counterfactual experiments via class siblings, decided-action writeback to moment/class neurons
6. Involuntary forecast pass (per-frame context-driven recall)
7. Voluntary think-actions and metacognitive reward learning (when to think)
8. Sleep/idle consolidation
9. (Deferred) Parallel experiment pool

Full details, neuron-kind specifications, executor operations, decided-vs-probabilistic vote semantics, testing strategy, and open questions in the [design document](./hippocampus.md).

---

## Robotics — Complete System

### Architecture

* Brain runs constantly (always-on processing)
* Thinking happens in the Hippocampal Region (see [Hippocampus design](./hippocampus.md)) — not as a feedback text channel, but as a second organ (the executor) running counterfactual experiments over moment and class neurons that live in cortical columns
* Inner monologue / stream of consciousness emerges from the always-on involuntary forecast pass plus voluntary think-actions, both surfacing and writing decided actions back onto moment/class neurons
* System 2 → System 1 enrichment: deliberated conclusions become cortical neurons that fire reflexively on context match — over a robot's lifetime, its experience-shaped reflexes are exactly what its hippocampus has decided
* Rewards given throughout experiences by owner/trainer via remote
* The Hippocampal Region is the substrate that lets the brain reflect on past rewards and find better future actions
* Resulting robots will likely develop personalities based on which moments get minted, which classes get formed, and which decided actions get written back to them

### Hardware
- The algorithm would need to be implemented in hardware (FPGA/ASIC) for real-time processing of all sensory channels simultaneously

