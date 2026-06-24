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
- Actions must behave exactly like events — they need to form patterns and infer what comes next
- Executed actions get fed back as events (feedback loop): when I output character X, I should see it as input Y
- This creates connections from action X to event Y — the foundation of action sequences
- Actions are rewarded based on the sequences they produce

### Migration: event-only text app → action-based
- **Current state:** the text app (`apps/text`) is event-only — a single `text_char` *input* dim,
  passthrough resolution 256, no actions, no rewards, no `learn()`. It predicts the next character
  via the brain's *event* consensus (probability = strength-weighted voter share over the char dim).
- **Target state:** add a `text_char_out` *action* dim alongside the `text_char_in` event dim, and
  train with `learn()` (mirrors MNIST/stocks supervised wiring) so each voter→char connection carries
  a smoothed posterior.
- **The two inferences become distinct and both useful:**
  - **Event inference** = "what am I *expected* to say at this moment" — the predicted next character
    from the stream (passive expectation).
  - **Action inference** = "what do I *want* to say at this moment" — the character the brain chooses
    to emit (active intent). Executed actions feed back as events per the feedback loop above.
- **Consensus:** the action side wants **democratic** consensus (strength-weighted mean expected
  reward, argmax) — the natural fit for picking a single best character. **NB does not apply** until
  this reframing lands (it needs per-voter posteriors that only the action/reward path produces), and
  even then it is a poor fit for correlated temporal voters except on near-deterministic text.
- **Until the reframing lands, the text app is unaffected by the `--consensus` brain option** — that
  flag only switches the action path, and event-only text has no action votes.

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
- Current multi-channel (e.g., 30 stocks) already runs in parallel but shares pattern space
- This extends to true isolation where needed — separate pattern spaces for separate data streams
- Merge/transfer between isolated memories as a deliberate operation

---

## Vision Channel

### Core approach
- Reference: https://claude.ai/share/5ac65464-6293-4cad-9683-07f0bd135644
- Under the single-dimension-per-base-neuron rule, each (x, y) screen position becomes its own dimension (e.g. `pixel_12_5`), and the base neuron carries `{dimension: pixel_x_y, value: brightness}`
- Color (grayscale 0–256 to start) is the value; for RGB, each channel becomes its own dimension family (`pixel_x_y_r`, `_g`, `_b`)
- 100×100 grayscale camera = 10,000 dimensions × 256 values = ~2.5M possible base neurons
- No explicit relationships between inputs — the brain forms them via temporal prediction models
- Starts with MNIST-like approach and scales from there

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

## MPI Distribution

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

### MPI transport optimization
- levelContext is passed to every neuron.processFrame call in dispatchFrame. In-process that's a free reference; over MPI it serializes N times
- The transport layer should broadcast it once per (level, frame) and let workers reference by handle
- This belongs in the MPI shim, not the core files — current code is correct for that future

---

## Python Bindings + PyPI

Expose the Rust core to Python for broader adoption.

### Python bindings via PyO3/maturin
- Wrap `brain-core` Rust library with PyO3
- Pythonic API matching the Node.js wrapper patterns
- Build and publish to PyPI via maturin

### Python channel interface
- Python equivalent of Channel base class
- Example: stock channel in Python

---

## Pattern Efficiency

### Current behavior
- Error correction creates a higher-level pattern that captures the ENTIRE context and learns the ENTIRE inference across all channels/dimensions — even the ones the parent got right

### Proposed optimization
- When the parent gets a channel wrong, delegate inference of that channel to the higher-level child pattern
- When the parent gets a channel right, do NOT delegate — parent keeps inferring correctly
- This is more efficient but significantly more complicated
- Open question: does selective delegation improve accuracy or just efficiency?
- NOTE: if the neuron re-use works, this would be a conflicting optimization, and should not be done. 

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
- When printing a pattern, we should show it with its ancestors (which need to be active)
- Each node carries exponentially more sensory neurons as you go up the levels, representing different branches
- Not clear how to present this concisely

### Ideas
- Tree visualization of active pattern chains (sensory → level 1 → level 2)
- Collapse branches that share common sensory roots
- Highlight which dimensions each level added to the prediction

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