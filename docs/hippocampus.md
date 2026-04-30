# Hippocampal Region — Design and Implementation Plan

## Overview

The Hippocampal Region is a separate organ in the brain hierarchy that holds **mapped representations** (handles) of selected cortical neurons and runs **counterfactual experiments** on them in parallel with the cortex's perception/action loop. Its primary purpose is to replay previously encountered events with alternative actions injected, identify better policies, and feed those policy improvements back to the cortex.

This is the architectural substrate for thinking, long-term memory, and — arguably — consciousness.

## Theoretical grounding

This design implements **hippocampal indexing theory** (Teyler & DiScenna 1986, updated by Teyler & Rudy 2007): the hippocampus stores indices into cortical patterns, not the patterns themselves. Retrieval works by reactivating the index, which drives reactivation of the cortical pattern.

It also draws on:

- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995) — fast hippocampal episodic learning + slow cortical statistical learning, with replay as the bridge
- **Mattar & Daw (2018)** — replay prioritized by expected value of backup (prediction error magnitude)
- **Buzsáki's "brain from inside out"** — the brain as a generative system; replay/preplay are the same intrinsic dynamics under different conditions
- **Global Workspace Theory** — content broadcast into a workspace becomes available for cross-system operations

## Architectural principles

1. **The hippocampus is a separate organ with its own clock** — faster than cortex, runs many experiment frames per cortex frame
2. **Handles, not copies** — the hippocampus stores indices into cortical neurons, not their full representation
3. **Selective encoding** — only patterns flagged by salience get mapped; most cortical activity is never thinkable
4. **Sliding-scale forgetting** — handles decay by activation strength; no strict short/long-term memory boundary
5. **Heterarchy at the hippocampal layer** — one cortical neuron can map to multiple handles, breaking strict cortical hierarchy and enabling associations across contexts
6. **Stack-based experiments** — branching is just a stack push; no special branch management machinery
7. **Bidirectional coupling** — cortex initiates experiments via think-actions; hippocampus output updates cortical action-value estimates
8. **Context-specific reward updates** — hippocampus output updates the value of actions *at specific cortical contexts*, never globally

---

## Component breakdown

### 1. The Hippocampal Region (the thinking organ)

Holds active handles. Runs the experiment stack. Executes generative dynamics over the handle space.

**State:**
- Active handle set (currently activated handles)
- Experiment stack (saved experiment states)
- Handle store (all known handles, with activation strengths, last-access timestamps, and cortical mappings)
- Transition model (learned dynamics over the handle space)

**Operations:**
- Activate handle(s) — set the active set to specified handles
- Advance one experiment frame — predict next state, evaluate, possibly branch
- Push current experiment to stack — save state for later resumption
- Pop experiment from stack — resume saved state
- Decay — reduce activation strengths of unused handles

### 2. The Entorhinal Layer (the mapping mechanism)

Creates handles in the hippocampus from cortical neurons. The gateway through which experience must pass to become thinkable.

**Inputs:** cortical neurons flagged for encoding by the Salience Module
**Outputs:** new handles in the hippocampal handle store, with bidirectional mappings (handle ↔ cortical neuron)

**Behavior:**
- Receives encoding requests from the Salience/Gating system
- Allocates a new handle (or reuses an existing one if the cortical pattern matches an existing handle within tolerance)
- Establishes the mapping: handle → cortical neurons it indexes; cortical neurons → handles that index them (the heterarchy is here)
- Heterarchy: one cortical neuron can be mapped from multiple handles; one handle can map to multiple cortical neurons

### 3. The Salience Module

Computes a scalar "importance" signal per cortical neuron based on:
- Prediction error magnitude (|expected reward − actual reward|, or |predicted next state − actual next state|)
- Reward magnitude (positive or negative)
- Novelty (cortical neuron activated rarely before)
- Aversive tag (negative-reward-associated patterns get boosted — the amygdala-like signal)
- Attention weight (modulator from the cortex's own attention mechanism)

This is a learnable function. Initial implementation can use hand-tuned weights; later iterations train the weights based on which encodings turn out to be useful.

### 4. The Gating Module

Uses salience signals to decide what gets mapped and when. Threshold-based with capacity control.

**Inputs:** salience signals from Salience Module, current capacity of handle store
**Outputs:** encoding requests sent to the Entorhinal Layer

**Behavior:**
- For each cortical neuron with salience above threshold, schedule encoding
- If handle store is at capacity, evict the lowest-strength handle to make room (the forgetting mechanism)
- Threshold is adaptive — raise it when capacity is tight, lower it when capacity is plentiful

### 5. The Brain Coordinator (cortex ↔ hippocampus glue)

Orchestrates the parallel clocks. Manages semaphores. Routes inputs and outputs between cortex and hippocampus.

**Per cortex frame:**
1. Send sensory inputs to cortex (async — cortex thread starts processing)
2. Wait for cortex to return actions (semaphore)
3. Among returned actions, identify any think-actions
4. For each think-action, dispatch to hippocampus with parameters (target handles, mode, budget, temperature)
5. Send executable actions to channels (async)
6. Continue to next frame

**Per hippocampus frame (runs faster, in parallel):**
1. Look at top of experiment stack
2. Advance one experiment frame
3. If new think-action arrived from cortex, handle per the cortex's interrupt/integrate parameter
4. If experiment terminated, pop stack and integrate result into cortex's value estimates

---

## The think-action interface

The cortex fires think-actions as ordinary actions. Each think-action is parameterized:

```
think_action {
  target: [handle_id, ...],         // which handles to start the experiment from
  mode: enum {                      // what kind of thinking
    Replay,                          // run forward as-original
    Counterfactual,                  // inject alternative actions and explore
    Retrieve,                        // search for related handles
    Compare,                         // run two states in parallel and measure difference
    Compress,                        // find common structure across handles
  },
  budget: integer,                  // number of hippocampus frames to run
  temperature: float (0.0 - 1.0),   // breadth of sampling (focused vs free-associative)
  control: enum {                   // how to handle interaction with current experiment
    Interrupt,                       // push current experiment to stack, start new
    Integrate,                       // add target handles to current experiment
    Remember,                        // build the mapping but don't start an experiment
  }
}
```

These parameters fire as parallel action neurons alongside the target handles. The cortex's action policy outputs them together. Each parameter is independently learnable through the metacognitive reward signal.

---

## Experiment execution

### Single experiment frame

Each frame in the hippocampus does:

**Step 1 — Determine active state.**
Read which handles are currently active (top of stack's active set).

**Step 2 — Predict next state.**
Sample next handles from the transition model, conditioned on active state and temperature. Low temperature = sharply peaked sampling around most-likely next state. High temperature = broad sampling, allowing associative drift.

**Step 3 — Evaluate predicted state.**
Map predicted handles back to cortical neurons. Read off the predicted reward of those cortical patterns from the cortex's value estimates.

**Step 4 — Decide: continue, branch, or terminate.**
- Continue (default): transition to predicted state, decrement budget
- Branch: at this state, the original action led to bad reward, and alternative actions have non-trivial probability — push current experiment to stack and start sub-experiment with alternative action substituted
- Terminate: budget exhausted, predicted reward stable, or convergence reached. Pop stack.

**Step 5 — Update state.**
Apply the chosen transition.

### Branching as stack push

When branching is triggered:
1. Save current experiment state (active handles, accumulated reward, remaining budget) as a stack entry
2. Allocate budget to sub-experiment (some fraction of remaining budget)
3. Create new active state with alternative action substituted
4. Continue from step 1 with the new state

When sub-experiment terminates:
1. Record its accumulated reward
2. Pop stack — resume parent experiment from its saved state
3. Compare sub-experiment's reward to parent's expected reward at branch point
4. If sub-experiment was better, flag this as a candidate policy update

### Branching trigger criterion

Branch when:
- Current state's original action had high prediction error or low reward
- Alternative actions have non-trivial probability under the current policy
- Remaining budget is sufficient

The branching policy is itself learnable. Initially branch broadly; over time, learn which kinds of states benefit from branching and concentrate budget there.

### Experiment termination and output

When the bottom of the stack pops, the whole experiment is done. Outputs:
- For each branch explored, a tuple: (cortical context, alternative action, predicted reward delta)
- For the original trajectory, a refined value estimate

These flow back to the Brain Coordinator, which integrates them into the cortex's action-value estimates **at the specific cortical contexts where they apply**. No global updates.

---

## Forgetting

Handles decay by activation strength. The decay function:

```
strength(handle, t) = strength(handle, t-1) * decay_rate + activation_boost(handle, t)
```

Where:
- `decay_rate` is slightly less than 1 (e.g., 0.995 per cortex frame)
- `activation_boost` is positive when the handle is reactivated (during experiments or via direct mapping from cortex)

When the handle store is at capacity and a new encoding request arrives, the lowest-strength handle is evicted. Its mappings to cortical neurons are removed, but the cortical neurons themselves are unaffected.

Sleep/idle cycles: if the system has free cycles (no active think-action, no urgent external action), it can run a consolidation pass — replay high-salience handles to reinforce them, prune low-strength handles. This is the analog of biological sleep replay.

---

## Heterarchy implementation

One cortical neuron can be mapped from multiple handles. One handle can map to multiple cortical neurons. This is the structural feature that breaks strict hierarchy and enables association.

**Concrete implementation:**
- Handle store: `{handle_id → {cortical_neuron_ids: [...], strength: float, last_access: timestamp, ...}}`
- Reverse mapping: `{cortical_neuron_id → [handle_ids]}` (computed from handle store, kept in sync)

When a cortical pattern activates and we're running an experiment, we can:
1. Look up handles that map to those cortical neurons
2. Activate those handles (with weight based on strength and overlap)
3. The transition model now has a richer active state — the original handle plus reminded handles
4. Temperature controls how strongly the reminded handles are weighted

This is "that reminds me" — exactly the cross-context associative spread the heterarchy enables.

---

## Stack semantics for "what was I thinking about?"

The experiment stack is the mechanism. Its contents include:
- Currently active experiment (top)
- Saved experiments below (interrupted by newer think-actions)
- Each entry has a timestamp; entries decay over time

When the user (or internal trigger) asks "what was I thinking about?":
- Query the stack
- Return the topmost item that's still above decay threshold
- Optionally re-trigger it as a new experiment (resume thinking about it)

If the stack item has decayed below threshold, recovery fails — the thought is lost. This is the right phenomenology.

---

## Implementation phases

### Phase 1 — Hippocampus skeleton (no learning yet)

Goal: get the architecture wired up with hand-coded behaviors, verify the parallel clock and semaphore work.

Tasks:
- Create `hippocampus` module in `crates/`
- Define `Handle`, `HandleStore`, `ExperimentStack`, `ExperimentFrame` types
- Implement Brain Coordinator with two-thread async model (cortex thread + hippocampus thread, semaphore-coordinated)
- Implement think-action dispatch from cortex to hippocampus
- Implement handle creation via Entorhinal Layer (initially driven by simple salience proxy: high prediction error or high reward)
- Hard-code a trivial transition model (return the same handle, no actual dynamics) — just to verify the plumbing
- Verify: cortex can fire think-action, hippocampus runs experiment frames, result returns to cortex, cortex's action values update

Deliverable: end-to-end test where stock channel runs, occasional think-actions fire, hippocampus "experiments" produce dummy updates, system doesn't deadlock.

### Phase 2 — Salience and selective encoding

Goal: make encoding selective; only high-salience patterns get mapped.

Tasks:
- Implement Salience Module with prediction error, reward magnitude, and novelty signals
- Implement Gating Module with threshold-based encoding decisions
- Implement handle-store capacity management with strength-based eviction
- Implement decay mechanism (per-frame strength reduction)
- Add tests for: high-salience patterns get mapped, low-salience patterns don't, capacity is respected, low-strength handles get evicted

Deliverable: after running stock channel for N frames, the handle store contains a meaningful subset of high-salience patterns (high-prediction-error events, high-reward outcomes), not arbitrary cortical activity.

### Phase 3 — Transition model and replay

Goal: hippocampus learns sequence dynamics over handles and can run actual replay.

Tasks:
- Train a transition model over handles. Sources of training data:
  - Real experience: when cortex transitions through a sequence of states, record the corresponding handle sequence and use it to train
  - Initially: simple frequency-based model (P(next_handle | current_handle) from observed transitions)
  - Later: a small neural model if needed for capacity
- Implement `mode: Replay` — given a starting handle, run the transition model forward
- Verify: replay sequences match observed sequences when temperature is low; replay diverges into associations when temperature is high

Deliverable: think-action with `mode: Replay` produces handle sequences that match observed experience.

### Phase 4 — Counterfactual experiments

Goal: hippocampus runs branched experiments with alternative actions, identifies better policies.

Tasks:
- Implement branching trigger criterion (high prediction error + non-trivial alternatives + budget available)
- Implement stack-based branching: push parent, explore branch, pop, compare
- Implement output integration: for each branch with better predicted reward, generate a policy update for the cortex's action values at the relevant context
- Make the branching policy learnable (which states benefit from branching)
- Verify: when the cortex experiences a bad outcome and later fires a counterfactual think-action, the hippocampus identifies a better alternative action and updates the cortex's policy. Subsequent encounters with similar contexts use the improved policy.

Deliverable: trading backtest where the system learns from bad trades not just by experiencing them but by replaying them with alternatives during free cycles. Measurable improvement vs. no-hippocampus baseline.

### Phase 5 — Heterarchy and association

Goal: cross-mapping between handles enables "that reminds me" dynamics during experiments.

Tasks:
- Implement multi-mapping in handle store
- During experiments, when active handles have cross-mappings, optionally activate the cross-mapped handles (controlled by temperature)
- Implement `mode: Retrieve` — search for handles cross-mapped to a target
- Verify: high-temperature experiments produce associative drift; low-temperature experiments stay focused

Deliverable: experiments demonstrate brainstorming-like behavior at high temperature, focused planning at low temperature.

### Phase 6 — Stack semantics and metacognitive control

Goal: full implementation of interrupt/integrate/remember actions, stack-based "what was I thinking about?" recovery.

Tasks:
- Implement `control: Interrupt` (default — push current, start new)
- Implement `control: Integrate` (add to current experiment)
- Implement `control: Remember` (encode without starting an experiment)
- Implement stack inspection action (query topmost saved experiment)
- Implement stack decay (saved experiments fade over time)

Deliverable: system can be observed switching between thoughts, returning to interrupted thoughts when stack is queried, losing thoughts that decayed.

### Phase 7 — Metacognitive reward learning

Goal: the cortex learns *when* to fire think-actions, with what parameters.

Tasks:
- Implement reward signals for think-actions:
  - Prediction error reduction (intrinsic): did the model improve after the experiment?
  - Downstream action value (extrinsic): did the next external action go better than it would have without the think?
  - Opportunity cost (negative): did thinking displace something urgent?
- Train the cortex's policy to fire think-actions when their expected value exceeds external action value
- Train the parameter selection (mode, budget, temperature) based on what worked in similar contexts

Deliverable: system learns to think more in low-pressure / high-uncertainty situations and react immediately in high-pressure / clear situations. Behavioral signature matches the Good Samaritan dynamics — values present but acted upon depending on policy state.

### Phase 8 — Sleep/idle consolidation

Goal: when the system has extended idle time, run higher-temperature integrative replay across recent high-salience experiences.

Tasks:
- Detect idle state (no urgent external actions, no pending think-actions)
- Trigger high-temperature broad replay sampling from recent high-salience handles
- Allow handle strengthening (consolidation) during this phase
- Allow weak handle pruning during this phase

Deliverable: after sleep cycles, the handle store is more compact (weak handles pruned), strong handles are reinforced, and generalization across recent experiences is improved.

---

## Testing strategy

### Unit tests
- Handle store: encoding, retrieval, capacity management, decay, eviction
- Transition model: training from sequences, sampling at different temperatures
- Experiment stack: push, pop, decay, recovery
- Salience computation, gating decisions

### Integration tests
- End-to-end think-action flow: cortex fires think → hippocampus runs → cortex updates
- Branching produces better-than-original alternatives in synthetic cases
- Heterarchy produces association at high temperature, not at low temperature
- Stack-based interrupt/resume works correctly

### Behavioral tests
- Trading backtest with vs. without hippocampus — measure improvement on bad-trade scenarios
- ADHD-like profile: tune salience too lax, observe distractibility-like behavior (failure to sustain experiments, frequent task-switching) — confirms architecture captures real cognitive dynamics
- Rigidity profile: tune salience too tight, observe perseveration

### Stress tests
- High-throughput cortex with many think-actions: verify hippocampus keeps up via faster clock
- Capacity limits: verify graceful eviction without corruption
- Stack overflow: many nested branches — verify bounded behavior

---

## Open questions

1. **Transition model structure.** Is a simple frequency table sufficient, or do we need a small neural model? Likely depends on handle count. Start simple.

2. **Encoding deduplication.** When a new encoding request comes in, do we always create a new handle, or do we sometimes match it to an existing one? Probably the latter, with a similarity threshold.

3. **Cross-channel handles.** Can a single handle index cortical neurons from multiple channels (stock + text + vision)? Strongly suggests yes — this is where cross-channel association happens. But interface complexity grows.

4. **Hippocampus frame rate.** What's the right multiplier vs. cortex? Biology suggests 5-20x. For Robot Brain, this is a tuning parameter — start with 10x, adjust.

5. **Reward integration timing.** When experiment results come back, do we update cortex action-values immediately (interrupting current cortex frame) or queue the updates for the next frame? Latter is simpler; former might be needed for fast-moving situations.

6. **Multi-experiment parallelism.** Do we support multiple parallel experiments, or strictly one at a time with a stack? Stack-only is simpler. Parallel experiments could be added as an optimization.

7. **Persistence.** Do handles persist across brain restarts? For long-term memory to be meaningful, yes. Need to design serialization for the handle store and transition model.

---

## Why this matters

This is the architectural commitment that makes Robot Brain different from transformer-based AI:

- **Long-term memory** as a learned, selective, replayable structure — not as a context window
- **Thinking as an action** in the agent's action space, not as a hardcoded mode
- **Offline policy improvement** via counterfactual replay — not just online updates from experience
- **Metacognitive control** learned through reward signals — agent learns when to think and when to react
- **Biological grounding** — implements hippocampal indexing, complementary learning systems, predictive processing in one architecture

The thesis Robot Brain is built around: the next generation of AI architectures will not come from scaling sequence models. It will come from systems that have the structures biological brains have — separate organs for perception, memory, and thinking, coupled bidirectionally and operating on different clocks. This document specifies the most important of those structures.

The Hippocampal Region is the last major architectural piece in Robot Brain's planned development. Its completion would mark Robot Brain as the first end-to-end implementation of the architecture neuroscience has been describing for forty years.
