# Future Work

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

