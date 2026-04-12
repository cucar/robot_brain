# Future Work

## MNIST Digit Recognition Benchmark (post-Rust Phase 6)

### Why this matters

MNIST is the most widely recognized benchmark in machine learning. Demonstrating competitive digit recognition with an architecture that has no backpropagation, no gradient descent, no convolution, and no explicit spatial encoding would validate Robot Brain as a fundamentally different computational paradigm — not just a curiosity on financial data.

The key claim: **one architecture, zero modifications, multiple domains** (stocks, text, vision). No other system makes this claim with the same underlying mechanism.

### Approach — sequential pixel reading with action-based classification

Unlike conventional image classification, Robot Brain cannot process static images. Each MNIST image (28×28 grayscale) is presented as a **784-frame episode**, reading pixels sequentially in raster scan order (left-to-right, top-to-bottom). Each frame contains a single grayscale value (0–255).

Classification is modeled as an **action selection problem**:

* The brain has 10 possible actions (digits 0–9)
* At each frame (after passing the context length threshold), the brain outputs a digit prediction as its action
* During training, the correct digit action receives positive reward; incorrect actions receive negative reward
* Over many episodes, the brain learns which pixel-value sequences correlate with which digit actions
* The brain discovers spatial structure implicitly — e.g., dark pixels at positions spaced 28 apart represent a vertical line — without any geometric encoding

This is analogous to the stock channel (sequential price events → directional actions with reward) and the text channel (sequential character events → character actions with reward). Same mechanism, different domain.

### Spatial feature discovery

The system has no knowledge that pixels are arranged in a 28×28 grid. All spatial relationships must be discovered through temporal prediction:

* **Horizontal adjacency**: consecutive pixels in the sequence (distance 1)
* **Vertical adjacency**: pixels 28 frames apart (distance 28) — requires sufficient context length
* **Diagonal features**: pixels at distance 27 or 29
* **Larger structures**: hierarchical pattern neurons combine lower-level detections across longer timescales

Context length directly determines what spatial features the system can discover. A context length of 100+ is recommended to capture cross-row relationships.

#### Parallelization variant — row-at-a-time (28 channels)

For practical compute reasons, a parallelized variant reads one full row per frame across 28 simultaneous channels (one per column position). This reduces episode length from 784 to 28 frames while preserving the need to discover vertical relationships across frames. This maps naturally to the multi-channel architecture already validated with 30 stock channels.

#### Training protocol

1. Generate episodes from the 60,000 MNIST training images — each image becomes one 784-frame episode (or 28-frame episode in the 28-channel variant)
2. Run multiple training passes (episodes repeated 10–100 times) with low forget rate to build stable representations
3. Classification accuracy measured as: percentage of episodes where the brain's final-frame action matches the correct digit

#### Evaluation protocol

* **Training accuracy**: measured on the 60,000 training images across episodes (expect improvement similar to text memorization — 41% → 100% over 3–5 episodes with tight parameters)
* **Test accuracy (generalization)**: present the 10,000 held-out MNIST test images as new episodes the brain has never seen. This is the number that matters for benchmark comparison. The brain continues learning during test (no freeze mode), so accuracy is measured on **first exposure** to each test image, with randomized presentation order.

#### Compute requirements

* **Single-channel (784 frames/episode)**: 50,000 images × 100 episodes × 784 frames = ~3.9B frames. At 0.007ms/frame (Rust target) ≈ 8 hours. Requires Rust + Rayon threading.
* **28-channel (28 frames/episode)**: 50,000 images × 100 episodes × 28 frames = ~140M frames. Significantly more tractable — potentially under 1 hour in Rust.
* **For comparison**: conventional CNNs train on MNIST in 2–5 minutes on GPU. The compute gap is expected and irrelevant to the architectural claim.

#### Recommended hyperparameters (starting point)

Based on stock and text experiments:

* Context length: 100 (single-channel) or 10 (28-channel variant)
* Forget rate: 0.001–0.01 (low, to retain learned digit patterns across episodes)
* Error threshold: 0.3 (same as text memorization experiments)
* Merge threshold: 0.9

#### Dependencies

* Requires Rust core (Phase 4+) for practical training times
* Multi-threaded Rust (Phase 5) needed for the single-channel variant
* No architectural changes to the brain algorithm — same code as stock/text channels
* MNIST dataset download and episode generation tooling (trivial)

#### What this is NOT

This is not an attempt to beat CNNs on their home turf. It is a demonstration that a single prediction-only architecture, designed for temporal sequences, can learn visual recognition without any vision-specific components. The benchmark exists to make the architectural claim legible to the ML community using a universally understood task.

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
- Each pixel neuron has dimensions: x, y, color (grayscale 0–256 to start)
- Coordinates relative to screen — each combination is a different neuron
- 100×100 camera with 256 color values = ~2.5M possible neurons
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
- Channel sends pixels with 5 dimensions: x, y, r, g, b
- Brain activates neurons corresponding to differences between these dimensions
- For each dimension, transform coordinates to relative values within the frame
- This is done BEFORE temporal recognition — spatial level processing within same frame and age
- Two-level processing: spatial levels and temporal levels
- For spatial pooling, add x and y distances to connections table alongside temporal distance
- Each neuron has 8 spatial connections (neighborhood)
- Apply peak/pattern detection recursively based on spatial distances

### Value encoding (applies to all channels)
- Channels should NOT do encoding like slope categorization or discretization
- Brain should automatically convert:
  1. Take all active neurons in frame, group by dimensions
  2. Calculate differences of coordinate values between new and older neurons
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

## Robotics — Complete System

### Architecture
- Brain runs constantly (always-on processing)
- Thinking channel: a feedback text channel — brain writes to it and reads from it simultaneously
- This is inner monologue / stream of consciousness
- Rewards given throughout experiences by owner/trainer via remote
- Brain reflects on rewards through the thinking channel — those are "thoughts"
- Resulting robots will likely develop personalities based on their experiences and internal dialogue

### Hardware
- The algorithm would need to be implemented in hardware (FPGA/ASIC) for real-time processing of all sensory channels simultaneously

