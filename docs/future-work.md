# Future Work

Long-term plan, listed in implementation order.

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
- If positive, this is a strong demonstration that the architecture learns reusable temporal structure

---

## Chatbot

### Core concept
- Train the brain to hold conversations by learning when to speak and when to listen
- Events and actions both correspond to characters
- Each frame: one character neuron sent as event, one character neuron sent back as action
- During training, reward based on how closely returned actions match expected responses
- The system learns to "say things" in terms of how it encountered them before

### Training protocol
- Get conversation logs — therapy notes, interviews, etc.
- When the other person is speaking: punish all actions except "shut up" (silence action)
- When it's the target speaker's turn: reward actions that match what they said
- This teaches both WHAT to say and WHEN to speak/interrupt
- Keep repeating until high accuracy on all training data

### Training data structure
- Training data includes query and response
- During query phase: no actions expected (or only silence rewarded)
- During response phase: rewards based on matching the expected response

### Key architectural insight
- Actions are neurons like events: what ran is reported each frame, joins functions with the events around it, and
  is chosen by connections and their estimates
- Executed actions come back as what ran: when the brain says character X, the frame reports X as having run, and
  what follows it is recorded on its connections — the foundation of action sequences
- Actions are rewarded based on the sequences they produce

### From the text app to a chatbot
- **Now:** the text app (`apps/text`) is event-only: one channel with a `letter` event dimension of 256 values, no
  actions, no rewards. Each frame the brain infers the next letter as the event the most inferences name.
- **Next:** give the text channel an action dimension (`say`) beside its letter event, and reward what it says.
- **The two inferences become distinct and both useful:**
  - **Event inference** = "what am I *expected* to say at this moment": the next character the most inferences name
    (passive expectation).
  - **Action inference** = "what do I *want* to say at this moment": the character with the largest estimate (active
    intent). The character it says comes back as what ran.

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
- Test how the chatbot channel interacts with stock channel running simultaneously
- Both channels feed into the same brain — cross-channel pattern formation

---

## Separate Channels / Parallel Learning

### Core concept
- Ability to separate channels from each other — multiple independent memories
- Learn from parallel data streams simultaneously (e.g., scanning the Internet)
- Each channel or group of channels can operate on its own data stream without interference

### Why this matters
- Chatbot training at scale requires consuming many conversations in parallel
- Internet-scale data ingestion needs parallel processing pipelines
- Different data streams may have conflicting patterns that shouldn't interfere

### Relationship to existing architecture
- Current multi-channel (e.g., 10 stocks) already runs in one brain but shares one dictionary
- This extends to true isolation where needed — separate dictionaries for separate data streams
- Merge/transfer between isolated memories as a deliberate operation

---

## Vision Channel

### Core approach
- Reference: https://claude.ai/share/5ac65464-6293-4cad-9683-07f0bd135644
- A camera is one channel laid out over two spatial axes, x and y, as MNIST is now; each pixel reports its value in the
  channel's event dimension, at its position
- Color (grayscale to start) is the value; for RGB, one event dimension per color, or one channel per color
- The brain relates pixels by their offsets in space and time; a base neuron sees its neighbors one position away, and
  each level of functions sees one position beyond what its function covers
- Starts with the MNIST approach and scales from there

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

## MPI Distribution

Distribute across multiple machines for large-scale workloads.

### Add MPI layer for inter-region communication
- Each MPI rank runs one `Region` and its columns
- Every access to a neuron is already a batch call the thalamus fans out over the regions, with the results back in
  the order the tasks came (activate, write lifts, count relations, connect, infer, decommission): over MPI each batch
  becomes one message out and one back per rank
- Use `rsmpi` crate for Rust MPI bindings

### Neuron facts for MPI
- The thalamus holds the registry of each neuron's fixed facts (kind, dimension, value, channel, axes), the dictionary
  and the board; these stay on rank 0
- Each rank holds the neurons themselves: their histories, relation indexes and connections
- The facts never change after a neuron is created, so nothing needs synchronizing beyond the batches

### Selection
- Each rank returns its neurons' inferences in the infer batch
- Expansion and resolution run on rank 0, which outputs the actions and the events inferred

### Neuron partitioning and migration
- Neurons assigned to columns by id
- Migration protocol for rebalancing load across columns

### MPI transport optimization
- What a batch hands every neuron alike should be broadcast once per batch and referenced by handle, not serialized
  once per neuron
- This belongs in the MPI shim, not the core files

---

## Python Bindings + PyPI

Expose the Rust core to Python for broader adoption.

### Python bindings via PyO3/maturin
- Wrap `brain-core` Rust library with PyO3
- Pythonic API matching the Node.js wrapper patterns
- Build and publish to PyPI via maturin

### Python job interface
- Python equivalent of the Job class (`robot-brain`): declare channels, run frames, save and load
- Example: the stock trading job in Python

---

## Hippocampal Region (Thinking, Long-Term Memory, Metacognition)

The largest remaining architectural component. A second organ — the **executor** — that operates on cortical neurons in parallel with the cortex's own perception/action loop. Implements long-term memory, thinking-as-action, and offline policy improvement.

See: [Hippocampus Design and Implementation Plan](./hippocampus.md)

### Generalized replay and thinking

Replay/thinking should generalize to include all sensory neurons — not just text. The trigger question: what causes replay? Dreaming, re-thinking, and thinking in general are all forms of replay. The hippocampal executor is the mechanism, but the triggers need to be learned through reward signals (metacognitive control).

References:
- https://claude.ai/share/88a9e7ab-c9e0-4bb1-911e-2199c4102fd8
- https://claude.ai/share/13787bb6-37f2-4b91-96ef-e05496f30c7c

### Why this matters

This is the architectural commitment that separates Robot Brain from sequence-model-based AI:

* Long-term memory as cortical neurons minted by deliberate experimentation — not a context window, not a separate store.
* Thinking as an action in the policy — not a hardcoded mode.
* Offline policy improvement via counterfactual replay through class siblings.
* Metacognitive control learned through reward signals — the agent learns when to think and when to react.
* **HM falls out as a direct architectural prediction.** Removing the executor leaves cued recall and skills intact but kills new-moment formation, deliberate recall, and counterfactual reasoning.

Built on hippocampal indexing theory (Teyler & DiScenna 1986; Teyler & Rudy 2007), complementary learning systems (McClelland et al. 1995), Mattar & Daw's expected-value-of-backup model (2018), and Kahneman dual-process theory implemented as architecture rather than metaphor.

The design, its neuron kinds, the executor's operations and its implementation phases are in the
[design document](./hippocampus.md).

---

## Imitation Rewards

### Core concept
- Reward the brain for producing actions that match a target behavior
- "Like me" — the brain learns to imitate by being rewarded when its output matches the demonstrator's output
- This is the reward mechanism that enables learning from observation rather than explicit instruction

### Why it comes after hippocampus
- The hippocampus provides counterfactual experimentation — "what would have happened if I had done X?"
- Imitation rewards + hippocampal replay = offline imitation learning from stored moments
- The brain can replay observed behavior and experiment with matching it
- Without hippocampus, imitation rewards still work but only in real-time (no offline improvement)

### Applications
- Robotics: learn motor behaviors by watching demonstrations
- Chatbot: already partially addressed by the per-turn reward scheme, but imitation rewards generalize it
- Any domain where "do what that agent did" is a valid learning signal

References:
- https://claude.ai/share/e919bd9a-7370-4983-9d70-8134b07c7d15
- https://claude.ai/share/f9732e46-a95c-44d2-8dee-b7217392834c
- https://claude.ai/share/3067bd31-ec70-4ba9-a645-a7f4513cd9b7

---

## Debugging Tools

### Problem
- When printing a call, we should show what it covers, down to the base activations
- Each call covers more base activations as you go up the levels, along different branches
- Not clear how to present this concisely

### Ideas
- Tree visualization of the calls on the apex and what each covers (base → level 1 → level 2)
- Collapse branches that share common base activations
- Highlight which relations each level added

Reference: https://claude.ai/share/e319c25b-5323-4313-adf2-d1720e068b16

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
* Imitation rewards enable learning from human demonstrations — watch and reproduce

### Hardware
- The algorithm would need to be implemented in hardware (FPGA/ASIC) for real-time processing of all sensory channels simultaneously