# Hippocampal Region — Design and Implementation Plan

## Overview

The Hippocampal Region is a separate organ in the brain hierarchy that holds **mapped representations** (moments) of selected cortical neurons and runs **counterfactual experiments** on them in parallel with the cortex's perception/action loop. Its primary purpose is to replay previously encountered events with alternative actions injected, identify better policies, and feed those policy improvements back to the cortex.

Thinking happens in two modes:

- **Involuntary** — every cortex frame ends with an automatic forecast pass: the brain takes the highest-level active cortical patterns, looks up matching moments in the hippocampus, and triggers short replays. This is the always-on background channel ("that reminds me," mind-wandering, gut-feel forecasting).
- **Voluntary** — the cortex can also emit explicit think-actions ("someone told me to think about this; I am thinking about it") with parameters controlling depth, breadth, and how to combine with whatever the hippocampus is already doing.

This is the architectural substrate for thinking, long-term memory, and — arguably — consciousness.

## Theoretical grounding

This design implements **hippocampal indexing theory** (Teyler & DiScenna 1986, updated by Teyler & Rudy 2007): the hippocampus stores indices into cortical patterns, not the patterns themselves. Retrieval works by reactivating the index, which drives reactivation of the cortical pattern.

It also draws on:

- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995) — fast hippocampal episodic learning + slow cortical statistical learning, with replay as the bridge
- **Mattar & Daw (2018)** — replay prioritized by expected value of backup (prediction error magnitude)
- **Buzsáki's "brain from inside out"** — the brain as a generative system; replay/preplay are the same intrinsic dynamics under different conditions
- **Global Workspace Theory** — content broadcast into a workspace becomes available for cross-system operations
- **Formal Concept Analysis** — moment-classes form a hierarchy of common cortical-neuron subsets, conceptually the same lattice FCA builds, approximated online by agglomerative clustering

## Architectural principles

1. **The hippocampus is a separate organ with its own clock** — runs faster than cortex, executes many experiment frames per cortex frame
2. **Moments, not copies** — a moment is a *weighted* set of cortical-neuron references — each entry is `(neuron_id, strength)` where the strength is that neuron's activation at the moment of encoding — plus the frame number, plus the reward/error that triggered remembering, plus the actions that were taken during the encoded context
3. **Selective encoding** — only frames whose reward or prediction error exceeds a z-score threshold get encoded; ordinary experience is never thinkable
4. **Two thinking modes** — involuntary per-frame forecast + voluntary cortex-issued think-actions, both running on the same experiment machinery
5. **Sliding-scale forgetting** — moments decay by activation strength; classes decay slower than instances; a death ledger tracks evictions
6. **Heterarchy at the moment layer** — one cortical neuron can belong to many moments and many classes, breaking strict cortical hierarchy and enabling cross-context association
7. **Hierarchical moment classes** — moments with overlapping cortical-neuron sets are abstracted into class moments by online agglomerative clustering; classes activate when their children do, the same way cortical pattern neurons activate from their children
8. **Stack-based experiments** — branching is just a stack push; a single experiment runs at a time initially (multi-experiment parallelism is a future optimization)
9. **Bidirectional coupling** — cortex initiates voluntary thinking via think-actions; hippocampus output updates cortical action-value estimates *at the specific cortical contexts where they apply*, never globally

---

## Component breakdown

### 1. The Hippocampus (the thinking organ)

Owns moments, classes, the temporal moment graph, the experiment stack, and the encoding/forgetting machinery. Encoding (what the biology calls the entorhinal pathway) is just a method on this class — there is no separate component for it.

**State:**
- `MomentIndex` — all known moments, keyed by id, each carrying:
  - `cortical_activations: Map<NeuronId, f32>` — what cortex was holding when this moment was minted, *with each neuron's activation strength at that instant*. These per-neuron strengths are the weights of the cortical → moment connections; they are what forgetting acts on at the link level.
  - `actions_taken: [ActionRecord]` — every action neuron that fired during the encoded context, each as `{action_neuron_id, frame_offset_within_context, activation_strength}`. This is the substrate for causal reasoning during counterfactual replay (which action was taken at this moment, what alternatives existed).
  - `frame_number: u64` — when it happened
  - `trigger: { reward: f32, prediction_error: f32 }` — the salience that caused encoding
  - `strength: f32` — moment-level activation strength (used for moment-level forgetting and eviction); distinct from the per-link strengths in `cortical_activations`
  - `last_access: Frame` — for decay calculations
  - `parent_classes: Set<MomentId>` — which class moments abstract this one
  - `temporal_edges: Map<MomentId, f32>` — weighted edges to neighboring moments in time
- `ClassIndex` — moment-class neurons, structurally identical to moments but with `children: Set<MomentId>` and a `centroid: Map<NeuronId, f32>` (a weighted neuron set, see "Moment classes" below)
- `ActiveSet` — currently activated moments (top of experiment stack's view)
- `ExperimentStack` — saved experiment states (single experiment at a time runs in Phase 1; stack stores branches and interrupts)
- `DeathLedger` — append-only log of evicted moment ids and the reason (low strength, cortical refs deleted, capacity pressure); used for debugging, observability, and to prevent immediate re-encoding of just-evicted patterns

**Operations:**
- `encode_moment(cortical_ids, frame, trigger) → MomentId` — creates a moment, attaches it to classes (creating new classes as needed), wires temporal edges to recent moments
- `activate(moment_ids)` — sets the active set; class moments activate automatically when enough of their children are active
- `advance_experiment_frame()` — predict next state via temporal graph, evaluate, possibly branch
- `push_branch() / pop_branch()` — stack manipulation for counterfactual exploration
- `decay()` — per-cortex-frame strength reduction, eviction pass, death ledger updates
- `match_to_active_cortex(top_level_active) → [MomentId]` — context-match moments for the involuntary forecast pass

### 2. The Salience Module

Computes a scalar "remember this" signal per cortex frame:

- `z_reward = (reward − μ_reward) / σ_reward` over a sliding window
- `z_pe = (prediction_error − μ_pe) / σ_pe` over a sliding window
- Encode iff `|z_reward| > θ_r` or `|z_pe| > θ_pe`

Novelty is **not** an input. Every observation contains some novelty; that does not make it worth remembering. What the literature calls "novelty effects" is in practice prediction error, and prediction error is already covered above.

The salience trigger fires `encode_moment(currently_active_cortical_ids, current_frame, {reward, prediction_error})`.

### 3. The Brain Coordinator (cortex ↔ hippocampus glue)

Orchestrates the parallel clocks. Manages semaphores. Routes inputs and outputs.

**Per cortex frame:**
1. Read sensory inputs; cortex thread propagates and picks actions
2. Wait on `SignalCortexDone`
3. If salience triggered, push an encoding request to the hippocampus
4. If cortex emitted think-actions, push them as voluntary experiments
5. Take the top-level active cortical patterns and push an involuntary forecast request
6. Drain any pending policy updates from the hippocampus into cortex action-values
7. Continue to next frame

**Per hippocampus frame (faster, in parallel):**
1. Service any pending encoding request (cheap, runs immediately)
2. If an experiment is on the stack, advance one frame
3. Otherwise pull the next request: voluntary think-actions take priority over involuntary forecasts
4. If idle, run consolidation: decay, eviction, death-ledger maintenance, optional class re-clustering

---

## Moment classes — online agglomerative clustering with Jaccard

Classes form a hierarchy of common cortical-neuron subsets across moments. The cortex's pattern neurons do this for sensory neurons; class moments do it for the cortical-neuron sets of moments. The mechanism:

**On each `encode_moment(cortical_activations, ...)`:**

1. Compute weighted Jaccard similarity between the new moment's neuron set (keys of `cortical_activations`) and every existing class centroid (keys of `centroid`):
   `J(A, B) = |A ∩ B| / |A ∪ B|`
   In practice this is bounded by an inverted index `cortical_neuron_id → [class_ids]` so we only compare against classes that share at least one neuron — the full pairwise scan is never needed.
2. Let `best_class` be the class with the highest Jaccard.
3. **Attach** if `J ≥ θ_attach` (e.g., 0.6): make the moment a child of `best_class`. Update the class centroid as the *weighted* running intersection of children's cortical activations: for each neuron `n` present in every child's `cortical_activations`, set `centroid[n] = mean(child.cortical_activations[n] for child in children)`. The set of keys is the intersection (a class represents *what its children share*); the values are the average strength of that shared neuron across the class. This makes the centroid both a membership signature and a strength profile.
4. **Merge classes** if attaching causes `best_class` and another class to now share `J ≥ θ_merge` (e.g., 0.85): merge into a single class, all children re-parented.
5. **Spawn a new class** if the new moment has Jaccard `≥ θ_spawn` (e.g., 0.5) with at least one existing *moment* (not class) but no class accepted it: create a new class with the new moment and that sibling moment as children, centroid keys = intersection of their neuron sets, centroid values = mean of paired activation strengths.
6. **Build hierarchy**: classes themselves are clustered the same way against other classes, at progressively lower thresholds. A class whose centroid is a subset of another class's centroid becomes its child. This produces the multi-level tree you wanted — instances at the leaves, increasingly abstract classes toward the root.
7. **Activation propagation**: when ≥ K of a class's children are in the active set, the class itself is activated. This is identical to cortical pattern activation: children activate parents.

8. **Class strength is independent, not derived.** Class moments carry their own `strength` field that decays on its own loop, exactly like cortical pattern neurons and instance moments do. Activation refreshes it: every time a class is activated — by its children firing, by being surfaced as a candidate in cue resolution, by participating in an experiment as a transition target, or by being a parent of an active class — its strength gets a boost. This is "thinking to keep remembering": classes that get used in experiments live longer.

   The centroid (`Map<NeuronId, f32>`) is recomputed whenever the set of surviving children changes — keys remain the running intersection, values the mean activation across surviving children. When the *last* child is evicted, the centroid is frozen at its last computed state and the class continues to live as an abstract signature with reward associations and connections to other classes. It can no longer be refined, but it can still be matched against future cues (its centroid neurons are real cortical-neuron ids), still be activated during experiments, and still participate in the temporal moment graph and class hierarchy. This matches the human experience of "I've had cases like this many times" without being able to bring up a single specific instance.

   Classes are evicted only when their own `strength` drops below the eviction threshold — never automatically because of child count.

**Idle re-clustering.** During consolidation, the hippocampus can revisit recently changed classes and rebalance: split classes whose centroids have drifted, merge classes that have converged, prune classes with fewer than 2 surviving children. This is the offline pass that Formal Concept Analysis would do exhaustively; we do it incrementally.

**Why this works.** Online agglomerative clustering with Jaccard distance is O(active classes per encoding), bounded by the inverted index. It's what the cortex already does in spirit, lifted one level. And it gives counterfactuals for free: sibling moments under a shared class differ in which action neuron fired, so the class *is* the counterfactual machinery.

---

## The temporal moment graph (replaces "transition model")

When a moment `M_new` is encoded at frame `f_new`, the hippocampus draws weighted edges to the K most recent moments encoded before it:

```
edge_weight(M_new, M_prev) = 1 / (f_new − f_prev)
```

If an edge already exists, its weight is incremented (Hebbian — co-occurrence reinforces). No model is trained; the graph *is* the transition model. Sampling a successor moment given the current active set means: walk outgoing edges weighted by `edge_weight × class_overlap × temperature_kernel`, draw one.

This mirrors the existing cortical context-distance machinery, which the codebase already implements.

---

## Actions stored with moments (causal substrate)

Every moment records *all* the action neurons that fired during its encoded context — not just the single action at the trigger frame. The field is:

```
actions_taken: [{ action_neuron_id, frame_offset_within_context, activation_strength }, ...]
```

The intent: causal relationships are formed by what actions were taken in the lead-up to a salient outcome, not by a single action at a single instant. A moment that ends in a large reward needs to remember the sequence of actions that led to it, not just the last one. Counterfactual replay then asks "what if action X had been suppressed at frame_offset −5, or replaced by Y?" — and the class hierarchy supplies sibling moments where exactly that variation actually happened.

This is **deliberately under-specified** at this stage. Open sub-questions:
- What is the "context window" — the last N cortex frames before encoding? The current cortex working-memory span? Something keyed off prediction-error rises?
- Are action records decayed at the link level the same way cortical activations are, or kept as a fixed sequence stamp?
- When the experiment substitutes an alternative action for replay, does it substitute at one frame_offset or rewrite the whole sequence?

These get pinned down once Phases 1–4 produce a real moment store to experiment against.

---

## The think-action interface (voluntary thinking)

The cortex fires think-actions as ordinary actions. Critically, **the cortex does not know about moments.** Its world is cortical neurons and patterns. A think-action says "think about *this cortical content*," and the hippocampus is responsible for translating that into the relevant moments via the inverted index `cortical_neuron_id → [moment_ids]` and class-membership lookup. This is the silicon analog of "remember the time when we…" — the cortex supplies the cue (the "when we…" pattern), the hippocampus finds the moment.

```
think_action {
  cue: [cortical_neuron_id, ...],   // cortical content to think about — patterns,
                                     //   sensory ids, action ids, or any mix.
                                     //   Hippocampus resolves these to candidate moments.
  mode: enum {
    Replay,                          // run forward as-original
    Counterfactual,                  // inject alternative actions and explore
    Retrieve,                        // search for related moments via class siblings
    Compare,                         // run two cued states in parallel and measure difference
    Compress,                        // find common structure across cued moments
  },
  budget: integer,                  // number of hippocampus frames to spend
  temperature: float (0.0 - 1.0),   // breadth of sampling (focused vs free-associative)
  control: enum {
    Interrupt,                       // push current experiment to stack, start new
    Integrate,                       // add cue to current experiment
    Remember,                        // build the mapping but don't start an experiment
  }
}
```

**Cue → moment resolution** (done inside the hippocampus, not the cortex):
1. Look up moments via the inverted index for each neuron in `cue`
2. Score each candidate moment by overlap with `cue`, boosted if the moment's parent class also matches the cue
3. Take the top-scoring moments above a match threshold as the experiment's starting `ActiveSet`

This is the same machinery as `match_to_active_cortex` used by the involuntary forecast pass — the only difference is the source of the cortical pattern (cortex's currently-active top-level patterns for involuntary, the explicit `cue` parameter for voluntary).

Each parameter is independently learnable through the metacognitive reward signal (Phase 7).

---

## Per-frame involuntary forecast pass

After cortex finishes its frame, the brain takes the top-level active cortical patterns and calls `Hippocampus.match_to_active_cortex(active)`. The match algorithm is the same as cortical context-matching but without the distance terms — pure set overlap, weighted by class-membership boost (a moment whose parent class is active gets a boost).

For each surfaced moment above a match threshold:
- Activate it
- Run a small-budget Replay or Counterfactual experiment via the temporal moment graph
- Push policy-update tuples to the result queue

This pass is unconditional and budget-bounded. It is the "always thinking in the background" channel. It fires every cortex frame even when the cortex emits no think-actions.

When a voluntary think-action arrives mid-forecast, it pushes the forecast onto the stack (`control: Interrupt` is the default) and runs. When it finishes, the forecast resumes — the same machinery as branch-push.

---

## Experiment execution

### Single experiment frame

**Step 1 — Determine active state.** Read `ActiveSet` from top of stack.

**Step 2 — Predict next state.** Sample a successor moment by walking the temporal moment graph from current active moments, weighted by edge weight × class overlap × temperature. Low temperature = peaked sampling around most-likely next moment. High temperature = broad sampling, allowing class siblings and weakly-connected moments to surface (associative drift).

**Step 3 — Evaluate predicted state.** Map predicted moment back to its cortical neurons. Read off the predicted reward of those cortical patterns from cortex's value estimates.

**Step 4 — Decide: continue, branch, or terminate.**
- Continue (default): transition, decrement budget
- Branch: original action led to bad reward and an alternative action is available — push parent, start sub-experiment with alternative action substituted
- Terminate: budget exhausted, predicted reward stable, or convergence reached. Pop stack.

**Step 5 — Update state.** Apply the chosen transition.

### Counterfactual actions via class siblings

When the experiment wants to ask "what if I had taken action B instead of action A at moment M?", it walks up to M's parent class, finds sibling moments under that class where action B was the active action, and replays forward from one of them. Class siblings differ in action and outcome but share the cortical context — this is exactly the substitution semantics counterfactuals require, and it falls out of the class hierarchy at no extra cost.

### Branching as stack push

When branching is triggered:
1. Save current experiment state (active moments, accumulated reward, remaining budget) as a stack entry
2. Allocate budget to sub-experiment (some fraction of remaining budget)
3. Switch to sibling moment with alternative action
4. Continue from step 1

When sub-experiment terminates: pop, compare, flag better-than-original alternatives as candidate policy updates.

### Output integration

Outputs flow back to the Brain Coordinator as `(cortical_context, alternative_action, predicted_reward_delta)` tuples and update cortex action-value estimates at the specific contexts where they apply. No global updates.

---

## Forgetting

Forgetting acts at two granularities — *links* (per-cortical-neuron connections inside a moment) and *moments* themselves — with classes derived from whatever survives below them.

**1. Per-link decay (inside each moment).**
Each entry in a moment's `cortical_activations` map decays independently:
```
cortical_activations[n][t] = cortical_activations[n][t-1] * decay_rate_link
                              + activation_boost_if_n_active_now
decay_rate_link ≈ 0.995 per cortex frame
```
The initial value is the cortical neuron's activation strength *at the moment of encoding* — so neurons that were strongly active when the moment was minted survive much longer in that moment than neurons that were barely active. This is what your design intent specified: "the strength values come from when the moments were captured." When a link's strength falls below `link_eviction_threshold`, that neuron is dropped from the moment.

**2. Moment-level decay.**
```
strength(M, t) = strength(M, t-1) * decay_rate_moment + activation_boost
decay_rate_moment ≈ 0.995 per cortex frame
```
A moment's overall `strength` is initialized from the salience that triggered encoding (large |z_reward| or |z_pe| → high initial strength). When `strength < eviction_threshold`, or when `MomentIndex` is at capacity, the lowest-strength moment is evicted. A moment is also evicted if its `cortical_activations` map drops below `min_cortical_refs` (e.g., 3) due to link-level eviction or cortical deletion.

**3. Class moments — independent decay, slower than instances.**
Classes have their own decay loop, like pattern neurons:
```
strength(C, t) = strength(C, t-1) * decay_rate_class + activation_boost(C, t)
decay_rate_class > decay_rate_moment   // classes age more slowly
```
`activation_boost` fires whenever the class is activated — by its children, by cue resolution, by being a transition target in an experiment, or by being a parent of an activated class. This is the "thinking to keep remembering" mechanism: a class that participates in many experiments stays alive indefinitely, even after every instance moment under it has been evicted.

When the last child is evicted, the centroid is frozen at its last computed state. The class persists as an abstract signature — still matchable by cue (its centroid neurons are real cortical-neuron ids), still carrying reward associations from its trigger history, still connected to other classes in the hierarchy and to other moments through the temporal graph. It can be activated and used in experiments; its predicted reward signifies "this is the kind of situation that tends to go well/badly," even when no specific case can be brought up. This matches the everyday experience of *"I've had cases like this many times"* with no recallable instance.

A class is evicted only when its own `strength` drops below the class eviction threshold — never automatically due to child count. Eviction is logged to the death ledger.

**4. Cascade-on-cortical-deletion.**
When a cortical neuron is deleted upstream, every moment removes that key from its `cortical_activations` map, and every class removes that key from its `centroid` map. If a moment's map drops below `min_cortical_refs`, the moment is evicted. Classes are *not* evicted on cortical deletion — they continue to live by their own strength even if their centroid shrinks, and even if their centroid empties entirely they remain as abstract reward-carrying nodes in the class graph until their own strength decays past the eviction threshold.

**The death ledger.** Every eviction appends `{moment_id, frame, reason, salience_at_birth}` to an append-only log, kept bounded by ring buffer. Uses:
- Debugging: "why don't I remember X?"
- Observability: shape of forgetting over time
- Re-encoding suppression: don't immediately re-encode something that was just evicted unless its salience is now substantially higher

**Sleep / idle consolidation.** When the hippocampus has free cycles, it runs a consolidation pass: replay high-salience moments to reinforce them, prune low-strength moments and orphan classes, and rebalance class clusters (merges, splits, re-parenting). This is the analog of biological sleep replay.

---

## Heterarchy implementation

One cortical neuron can be referenced by many moments and many classes. One moment can be a child of many classes.

**Concrete data:**
- `MomentIndex: {moment_id → Moment}`
- `ClassIndex: {class_id → Class}`
- Reverse mapping: `{cortical_neuron_id → [moment_ids]}` — the inverted index used by `match_to_active_cortex` and by the Jaccard search during encoding

This breaks strict cortical hierarchy and gives the system "that reminds me" — when active cortical patterns overlap a moment from a different context, it surfaces.

---

## Stack semantics for "what was I thinking about?"

The experiment stack is the mechanism. Its contents:
- Currently active experiment (top)
- Saved experiments below (interrupted by newer think-actions or by encoding events)
- Each entry has a timestamp; entries decay over time

Querying "what was I thinking about?" returns the topmost stack item still above decay threshold, optionally re-triggered as a new experiment. If everything has decayed, the thought is lost — the right phenomenology.

---

## Coordinator pseudo-code

```
// shared
SignalCortexDone:    semaphore (cortex → brain: frame done)
EncodeQueue:         channel<EncodeRequest>
ExperimentQueue:     channel<ThinkAction>     // voluntary, high priority
ForecastQueue:       channel<MomentSet>       // involuntary, low priority
ResultQueue:         channel<PolicyUpdate>

// cortex thread (1 frame per tick)
loop:
    inputs = read_sensory()
    cortex.activate(inputs)
    cortex.propagate_patterns()
    actions = cortex.pick_actions()              // may include think-actions
    cortex.emit(actions)
    SignalCortexDone.post()

// brain thread (orchestrator)
loop:
    SignalCortexDone.wait()
    if salience.should_remember(cortex.last_frame):
        EncodeQueue.push({active_ids, frame, trigger})
    for ta in actions.think_actions:
        ExperimentQueue.push(ta)
    ForecastQueue.push(cortex.top_level_active_patterns())
    integrate(ResultQueue.drain_nonblocking())   // apply pending policy updates

// hippocampus thread (free-running, faster clock)
loop:
    // 1. Encoding is cheap and time-sensitive: always serve first
    while req = EncodeQueue.try_pop():
        hippocampus.encode_moment(req)           // builds moment, attaches to classes,
                                                  // wires temporal edges

    // 2. If an experiment is in flight, keep advancing it
    if hippocampus.experiment_stack.has_active():
        result = hippocampus.advance_experiment_frame()
        if result.policy_update:
            ResultQueue.push(result.policy_update)
        if result.terminated:
            hippocampus.experiment_stack.pop()
        continue

    // 3. Voluntary think-actions take priority over involuntary forecasts
    if ta = ExperimentQueue.try_pop():
        hippocampus.start_experiment(ta)
        continue
    if forecast = ForecastQueue.try_pop():
        moments = hippocampus.match_to_active_cortex(forecast)
        if moments.non_empty():
            hippocampus.start_experiment(forecast_as_replay(moments, small_budget))
        continue

    // 4. Idle: run forgetting + consolidation
    hippocampus.decay()                          // strength reduction across moments+classes
    evicted = hippocampus.evict_below_threshold()
    for m in evicted:
        DeathLedger.append({m.id, current_frame, reason, m.trigger.salience})
    hippocampus.cascade_cortical_deletions()     // drop refs to deleted cortical neurons
    hippocampus.consolidate_classes()            // merge/split/re-parent if drifted
```

A high-priority voluntary think-action arriving mid-experiment pushes the current experiment onto the stack via `control: Interrupt` (the default) and runs. When it finishes, the older experiment resumes — same stack machinery as counterfactual branching.

Multi-experiment parallelism (running several experiments concurrently rather than one at a time with a stack) is deferred. Phase 1 is strictly one-experiment-at-a-time.

---

## Implementation phases

### Phase 1 — Hippocampus skeleton (no learning yet)

Goal: wire up the architecture with hand-coded behaviors, verify the parallel clocks and semaphores work.

Tasks:
- Create `hippocampus` module
- Define `Moment`, `Class`, `MomentIndex`, `ClassIndex`, `ExperimentStack`, `DeathLedger` types
- Implement Brain Coordinator with two-thread async model (cortex + hippocampus, semaphore-coordinated)
- Implement `encode_moment` driven by simple z-score salience (reward and prediction error only, no novelty)
- Hard-code a trivial temporal graph (just record edges) and a trivial replay (return current moment) — verify plumbing
- Verify: cortex fires think-action, hippocampus runs experiment frames, result returns, cortex action values update

### Phase 2 — Salience and selective encoding

- Implement z-scored reward and prediction error over sliding windows
- Implement gating (encode iff |z| > threshold)
- Implement strength-based decay and eviction
- Implement death ledger and re-encoding suppression
- Test: high-z events encoded, low-z events not, capacity respected, eviction logged

### Phase 3 — Moment classes (online agglomerative Jaccard clustering)

- Implement the inverted index `cortical_neuron_id → [class_ids]`
- Implement `attach / merge / spawn` decisions on each encode
- Implement class centroid as running intersection of children's cortical-neuron sets
- Implement multi-level hierarchy (classes clustered against classes at lower thresholds)
- Implement child→parent activation propagation
- Test: similar moments land in the same class, classes form a tree, class activation follows children

### Phase 4 — Temporal moment graph and replay

- On each encode, draw weighted edges to recent moments
- Implement edge-weighted sampling for next-state prediction
- Implement `mode: Replay` end-to-end
- Test: replay sequences match observed sequences at low temperature; diverge at high

### Phase 5 — Counterfactual experiments via class siblings

- Implement branch-via-class-sibling: find sibling under parent class with alternative action, replay forward
- Implement stack-based push/pop for branches
- Implement output integration into cortex action-values at specific contexts
- Test: replaying a bad outcome with alternative actions discovers better policies; subsequent encounters use the improved policy

### Phase 6 — Involuntary forecast pass

- Implement `match_to_active_cortex` (set overlap with class-membership boost, no distances)
- Wire the per-frame forecast push from brain to hippocampus
- Verify: even with zero think-actions, the hippocampus is producing background forecasts and small policy updates every frame

### Phase 7 — Voluntary think-actions and metacognitive control

- Implement full think-action parameters (mode, budget, temperature, control)
- Implement Interrupt / Integrate / Remember semantics
- Implement reward signals for think-actions (PE reduction, downstream action improvement, opportunity cost)
- Train cortex to fire think-actions when expected value exceeds external action value
- Test: system learns to think more in low-pressure / high-uncertainty situations and react immediately when pressed

### Phase 8 — Sleep / idle consolidation

- Detect idle state
- Run high-temperature integrative replay over recent high-salience moments
- Run class re-clustering (merge, split, re-parent)
- Reinforce strong moments, prune weak ones
- Test: after sleep, MomentIndex is more compact and class hierarchy has rebalanced

### Phase 9 (deferred) — Multi-experiment parallelism

Run multiple experiments concurrently rather than one-at-a-time with stack interrupts. Optimization, not correctness.

---

## Testing strategy

### Unit tests
- MomentIndex / ClassIndex: encoding, retrieval, capacity, decay, eviction, death ledger
- Jaccard clustering: attach, merge, spawn decisions; centroid as running intersection
- Temporal graph: edge construction, weighted sampling
- Experiment stack: push, pop, decay, recovery
- Salience: z-score thresholding over sliding windows

### Integration tests
- Full encode → class attach → temporal edge → replay → policy update flow
- Counterfactual via class sibling produces better-than-original alternatives
- Involuntary forecast fires every frame with no cortex prompting
- Voluntary think-action interrupts in-flight forecast and resumes after

### Behavioral tests
- Trading backtest with vs. without hippocampus — measure improvement on bad-trade scenarios
- ADHD-like profile: tune salience too lax, observe distractibility
- Rigidity profile: tune salience too tight, observe perseveration

### Stress tests
- High-throughput cortex with many think-actions
- Capacity limits and graceful eviction
- Cortical deletion cascades

---

## Open questions

1. **Jaccard thresholds.** `θ_attach`, `θ_merge`, `θ_spawn` need tuning. Start with 0.6 / 0.85 / 0.5 and adjust by observing class-tree shape on real data.

2. **Salience window length.** How many frames to compute σ over? Probably 100–1000 cortex frames; tunable.

3. **Hippocampus frame rate.** What's the right multiplier vs. cortex? Biology suggests 5–20×. Start with 10×.

4. **Reward integration timing.** Apply policy updates immediately (interrupting current cortex frame) or queue for next frame? Latter is simpler; former may be needed for fast-moving situations.

5. **Persistence.** Do moments persist across brain restarts? For long-term memory to be meaningful, yes. Need serialization for MomentIndex, ClassIndex, temporal graph, and death ledger.

6. **Cross-channel moments.** Can a single moment index cortical neurons from multiple channels (stock + text + vision)? Strongly suggests yes — this is where cross-channel association happens. Interface complexity grows.

7. **Multi-experiment parallelism.** Deferred to Phase 9. Single-experiment with stack is the Phase 1–8 contract.

---

## Why this matters

This is the architectural commitment that makes Robot Brain different from transformer-based AI:

- **Long-term memory** as a learned, selective, replayable structure — not a context window
- **Two thinking modes** — involuntary background forecasting + voluntary think-actions — both running on the same machinery
- **Offline policy improvement** via counterfactual replay through class siblings — not just online updates from experience
- **Metacognitive control** learned through reward signals — agent learns when to think and when to react
- **Forgetting that judges** — instances fade fast, classes persist, the death ledger remembers what was lost
- **Biological grounding** — implements hippocampal indexing, complementary learning systems, predictive processing, and FCA-style abstraction in one architecture

The thesis Robot Brain is built around: the next generation of AI architectures will not come from scaling sequence models. It will come from systems that have the structures biological brains have — separate organs for perception, memory, and thinking, coupled bidirectionally and operating on different clocks. This document specifies the most important of those structures.
