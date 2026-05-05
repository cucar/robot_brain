# Hippocampal Region — Design and Implementation Plan

## Overview

Robot Brain is a dual-system predictive architecture. Two organs run continuously in parallel over the same neuronal substrate:

- **Cortex (System 1)** — fast, reflexive, short-term. Observes the environment, splits reality into hierarchical patterns, predicts the next frame, picks short-term actions by habit. Probabilistic and statistical: many experiences smooth into action votes.
- **Hippocampus (System 2)** — slow, deliberate, long-term. Mints moment and class neurons in the cortical columns, runs counterfactual experiments over them, and writes *decided* long-term-optimum actions back onto those neurons.

The deep insight that organizes this design:

> **Cortex splits reality into patterns. Hippocampus groups patterns to generalize.**

Both organs operate on the same cortical neurons. The hippocampus is not a separate store. It is the **executor** that creates and operates on a particular kind of cortical neuron — one with tight enough context-matching tolerance to function as an episodic moment, or wider tolerance to function as a class. **Storage is always cortical. The hippocampus is the operator.**

This is the architectural commitment that makes Robot Brain different. A consequence: hippocampal experiment outputs become part of the cortex's vocabulary. System 2 continuously enriches System 1. This is the architectural substrate for expertise development — what cognitive psychology observes behaviorally as System 2 reflexes hardening into System 1 reflexes is, in this design, a structural inevitability.

### Thinkability = reachability by the executor

In the original cortex design, ordinary cortical neurons fire when their context matches and otherwise sit dormant — they are not deliberately summonable. Moment and class neurons are different: the hippocampus can reach them, reactivate them, traverse their connections, and run experiments over them. **A neuron is "thinkable" iff the hippocampal executor can reach it.** Thinkability is not a property of the neuron's location or type; it is a property of being operated on by the hippocampus.

Moment and class neurons live in cortical columns alongside sensory and pattern neurons. The thalamus is the bus that translates neuron ids ↔ properties (region/column-aware), so the hippocampus can address specific cortical neurons by id.

### The HM test

This architecture predicts HM directly. Removing the hippocampus removes the executor. After removal:

- Pre-existing moment and class neurons survive — they are cortical, with stable context-matching connections, reactivatable by sensory cues (cued recall).
- New moments cannot be minted — the operation that creates tight-tolerance neurons on demand is gone.
- Deliberate recall, planning, and counterfactual reasoning are gone — these all required executor reachability.
- Skills, perception, statistical learning, and cued recall continue normally — these are pure cortex.

If a Robot Brain with a fully populated cortex loses its hippocampus, that is the behavioral profile it should exhibit.

---

## Theoretical grounding

- **Hippocampal indexing theory** (Teyler & DiScenna 1986; Teyler & Rudy 2007) — the hippocampus indexes cortical patterns, doesn't store them. This design takes the further step: the indices themselves are also cortical (moment and class neurons), and the hippocampus is purely operator.
- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995) — fast experimental learning (System 2) + slow statistical learning (System 1), bridged by experiment-driven writeback.
- **Mattar & Daw (2018)** — replay prioritized by expected value of backup (prediction error magnitude).
- **Buzsáki's "brain from inside out"** — the brain as a generative system; replay/preplay are the same intrinsic dynamics under different conditions.
- **Kahneman dual-process theory** — Robot Brain implements System 1 and System 2 as separate organs over a shared substrate, rather than as a behavioral metaphor.
- **Recent hippocampal scenario-construction literature** — patients with hippocampal damage cannot imagine novel scenes either, not just remember old ones. The hippocampus is the executor; memory is one of its uses.

---

## Architectural principles

1. **One substrate, two operators.** Cortex and hippocampus operate on the same cortical neurons in the same columns. Thalamus mediates id↔property addressing.
2. **Cortex splits, hippocampus groups.** Cortex divides experience into hierarchical patterns through prediction-error-driven learning. Hippocampus groups moments into classes through deliberate clustering during encoding and consolidation.
3. **Moments and classes are cortical neurons.** A moment neuron is a tight-tolerance pattern neuron minted in one shot from a salient frame. A class neuron is a looser-tolerance pattern neuron grouping similar moments. Same substrate as sensory/pattern neurons; different minting process and different vote semantics (see below).
4. **Hippocampus is the executor, not the store.** It mints moment and class neurons, traverses them, runs experiments on them, and writes decided actions back onto them. It does not hold them.
5. **Recall is by context, not by handle.** The executor's primitive reactivation operation is "drive a partial context and let cortex pattern-complete." Handles are not part of the user-visible recall mechanism. (The hippocampus maintains transient experiment-local references; nothing more.)
6. **Decided actions, not probabilistic ones.** Sensory and pattern neurons accumulate action votes statistically across many experiences — probabilistic, smoothed. Moment and class neurons carry *decided* actions written by the hippocampus after experimentation. The decision is black-and-white: this is the action to take. When such a neuron is activated and votes, its decided action dominates the probabilistic votes from sensory/pattern neurons for the same action slot.
7. **Selective minting.** Only frames with reward or prediction error above a z-score threshold trigger moment minting. Ordinary experience never becomes thinkable.
8. **Heterarchy at the moment layer.** A cortical neuron can be inside many moments' contexts and many classes' centroids. Moments can be children of many classes. This breaks strict cortical hierarchy and enables cross-context association.
9. **Two clocks.** The hippocampus runs at a faster clock than the cortex, executing many experiment frames per cortex frame.
10. **Same death ledger, different aging.** Forgetting is unified through one death ledger, but moment and class neurons age and die on schedules distinct from sensory/pattern neurons because they represent different things on different timelines.
11. **Bidirectional coupling via thalamus.** Cortex emits think-actions that reach the hippocampus. Hippocampus reads currently active cortical state and writes decided actions back to specific moment/class neurons. Updates are local to the neuron written, never global.

---

## The cortical substrate (what lives in columns)

A column already contains sensory neurons and pattern neurons. To this design we add two additional neuron kinds, **living in the same columns**, addressable by the same thalamic translation layer:

- **Moment neuron** — a pattern neuron minted in one shot by the hippocampus from a salient frame. Tight context-matching tolerance: it fires when most (not necessarily all) of the encoded context is currently active. Carries:
  - Connections to the cortical neurons that constituted its encoded context, weighted by each neuron's activation strength at minting time. These weights are the substrate forgetting acts on at the link level.
  - Connections to action neurons, recording every action that fired during the encoded context (each with frame-offset and activation strength). This is the causal substrate for counterfactual replay.
  - Connections to other moment neurons (the temporal moment graph).
  - Membership in zero or more class neurons.
  - A **decided action** slot (initially empty), filled by the hippocampus after experiments converge.
  - Salience at birth, last-access frame, and a strength field that decays on the moment timeline.
- **Class neuron** — a pattern neuron at looser tolerance grouping similar moments. Centroid is the running weighted intersection of its children's contexts. Carries decided actions just like moments, applied to anything matching the class. Decays on the class timeline (slower than moments).

Both kinds participate in normal cortical activity: when their context matches the current frame within tolerance, they fire and contribute their votes. The difference from sensory/pattern neurons:

- They were minted by the hippocampus, not by gradual statistical learning.
- Their action votes are *decided*, not *probabilistic*. When a moment or class neuron is active and has a written decided action, that action wins over sensory/pattern probabilistic votes for the same slot.
- They are reachable by the hippocampal executor for experiments.

---

## The hippocampal executor

The hippocampus owns no permanent store. It owns:

- The salience module (windowed z-scoring of reward and prediction error).
- The experiment buffer (single experiment in flight in Phase 1; parallel pool in a later phase).
- A transient working set of currently-relevant moment/class neuron ids while an experiment runs.
- The death-ledger writer (the ledger itself is shared across cortex and hippocampus).

Its operations, in conceptual terms (the precise instruction set is left for implementation to discover — biology did not design its hippocampus all at once and we should not pretend to either):

- **Mint moment** — given the currently active cortical state and a salience trigger, create a tight-tolerance moment neuron in the appropriate column with connections to the active context, the active action neurons, and recent moments (temporal edges).
- **Group into class** — apply the class-formation rule (online weighted-Jaccard against existing class centroids) to attach the new moment to an existing class, merge classes, or spawn a new class.
- **Activate by context** — take a partial cortical context (either currently active patterns from the cortex's frame, or the cue field of a think-action) and drive cortex to pattern-complete the matching moment(s) and class(es). This is the recall primitive.
- **Traverse** — given an active moment, walk its temporal edges and class siblings to surface candidate next states.
- **Run an experiment** — replay or counterfactual exploration over the moment graph (see "Experiment execution").
- **Decide and write** — at the conclusion of an experiment, write decided actions to specific moment/class neurons.
- **Decay and evict** — apply moment-timeline and class-timeline decay; emit eviction records to the death ledger.

The cortex does not know about moments or classes. It just sees neurons firing and contributing votes. The hippocampus is the only thing that knows which neurons are moments, which are classes, and what experiments are running.

---

## The salience module

Computes a scalar "mint this moment" signal per cortex frame:

- `z_reward = (reward − μ_reward) / σ_reward` over a sliding window
- `z_pe = (prediction_error − μ_pe) / σ_pe` over a sliding window
- Mint iff `|z_reward| > θ_r` or `|z_pe| > θ_pe`

Novelty is not an input. Every observation contains some novelty; that does not make it worth remembering. What the literature calls "novelty effects" is in practice prediction error, already covered.

When the trigger fires, the hippocampus calls Mint over the currently active cortical state.

---

## Class formation — online agglomerative weighted Jaccard

When a moment is minted, the hippocampus groups it into the class hierarchy:

1. Compute weighted Jaccard between the new moment's context (the set of cortical neurons it connects to) and every existing class centroid. An inverted index `cortical_neuron_id → [class_ids]` bounds the comparison to classes that share at least one context neuron.
2. Let `best_class` be the class with the highest Jaccard.
3. **Attach** if `J ≥ θ_attach` (e.g., 0.6): make the moment a child of `best_class`. Update the class centroid as the weighted running intersection of children's contexts: keys = the intersection, values = mean activation strength across children for each shared neuron.
4. **Merge** classes if attaching causes `best_class` and another class to share `J ≥ θ_merge` (e.g., 0.85).
5. **Spawn** a new class if the new moment has Jaccard `≥ θ_spawn` (e.g., 0.5) with an existing moment (not class) but no class accepted it. New class's centroid keys = intersection, values = paired-strength means.
6. **Multi-level hierarchy** — classes are themselves clustered against other classes at progressively lower thresholds.
7. **Activation propagation** — when ≥ K of a class's children are active, the class itself activates. This is identical to ordinary cortical pattern activation: children activate parents.

The centroid is recomputed whenever the surviving child set changes. When the last child is evicted, the centroid is frozen at its last computed state and the class persists as an abstract signature — still matchable, still decay-eligible, still reward-carrying. This matches *"I've had cases like this many times"* with no recallable specific instance.

**Idle re-clustering** during consolidation revisits recently changed classes: split drifted ones, merge converged ones, prune classes with too few surviving children. This is the offline pass FCA would do exhaustively; we do it incrementally.

---

## The temporal moment graph

When a moment `M_new` is minted at frame `f_new`, the hippocampus draws weighted edges to the K most recent moments:

```
edge_weight(M_new, M_prev) = 1 / (f_new − f_prev)
```

Existing edges have their weight incremented (Hebbian — co-occurrence reinforces). No model is trained; the graph is the transition model. Sampling a successor moment given the current active set means walking outgoing edges weighted by `edge_weight × class_overlap × temperature_kernel`.

This mirrors the cortical context-distance machinery already in the codebase, lifted one level.

---

## Actions stored with moments

Each moment records all action neurons that fired during its encoded context window — not just a single action at the trigger frame:

```
actions_taken: [{ action_neuron_id, frame_offset_within_context, activation_strength }, ...]
```

A moment that ends in large reward needs to remember the *sequence* leading there, not just the last action. Counterfactual replay can then ask "what if action X had been suppressed at frame_offset −5, or replaced by Y?" — and class siblings under M's parent class supply moments where exactly that variation actually happened.

Sub-questions deferred until Phases 1–4 produce a real moment population:

- Context-window definition (last N frames, working-memory span, PE-rise-keyed).
- Are action records decayed at the link level the same way context links are, or kept as a fixed sequence stamp?
- When substituting an alternative action for replay, substitute at one offset or rewrite the sequence?

---

## Recall by context — the involuntary forecast pass

After cortex finishes its frame, the brain takes the top-level active cortical patterns and asks the hippocampus to pattern-complete any matching moments. Match algorithm: set overlap weighted by class-membership boost (a moment whose parent class is also active gets a boost). No distance terms — this is set-overlap context matching.

For each surfaced moment above threshold:

- Activate it (which lets it cast its decided-action vote, if any, into the cortex's action selection).
- Run a small-budget Replay or Counterfactual experiment from it.
- The experiment may write updated decided actions to moments it visits.

This pass fires every cortex frame, even with zero think-actions from the cortex. It is the always-on background channel — *"that reminds me…"*, mind-wandering, gut-feel forecasting.

When a voluntary think-action arrives mid-forecast, the forecast is pushed onto the experiment stack and the think-action runs. The forecast resumes after.

---

## The think-action interface (voluntary thinking)

The cortex fires think-actions as ordinary actions. Critically, **the cortex does not address moments by id**. Its world is neurons and patterns. A think-action says "think about *this cortical content*"; the hippocampus resolves the cue to candidate moments by activating the cue and reading off which moment/class neurons fire.

```
think_action {
  cue: [cortical_neuron_id, ...],   // cortical content to think about — patterns,
                                     //   sensory ids, action ids, or any mix.
  mode: enum {
    Replay,                          // run forward as-original
    Counterfactual,                  // inject alternative actions and explore
    Retrieve,                        // surface related moments via class siblings
    Compare,                         // run two cued states in parallel, measure delta
    Compress,                        // find common structure across cued moments
  },
  budget: integer,                  // hippocampus frames to spend
  temperature: float (0.0 - 1.0),   // breadth of sampling
  control: enum {
    Interrupt,                       // push current experiment, start new
    Integrate,                       // add cue to current experiment
    Remember,                        // mint without starting an experiment
  }
}
```

**Cue → moment activation** (inside the hippocampus):

1. Drive the cue neurons.
2. Let cortex pattern-complete; collect the moment/class neurons that fire above threshold.
3. The active set becomes the experiment's starting state.

This is the same machinery as the involuntary forecast pass. The only difference is the source of the cue — cortex's currently-active top-level patterns for involuntary, the explicit `cue` parameter for voluntary.

Each parameter is independently learnable through the metacognitive reward signal (Phase 7).

---

## Experiment execution

### Phase 1 — single-track

One experiment runs at a time. Branching is implemented as an experiment stack: branch = push, terminate = pop. A higher-priority think-action arriving mid-experiment pushes the current one and runs.

### Single experiment frame

1. **Determine active state.** Read the active set from the top of the stack.
2. **Predict next state.** Sample a successor moment by walking the temporal moment graph from the active moments, weighted by edge weight × class overlap × temperature. Low temperature = peaked; high = associative drift.
3. **Evaluate.** Map the predicted moment back to its cortical context. Read off predicted reward via the cortex's value estimates for those neurons, plus any decided actions already attached to the moment.
4. **Continue, branch, or terminate.**
   - Continue: transition, decrement budget.
   - Branch: original action led to bad reward and an alternative is available — push parent, start sub-experiment with the alternative substituted (see counterfactual via class siblings below).
   - Terminate: budget exhausted, predicted reward stable, or convergence reached. Pop stack.
5. **Update state.** Apply the chosen transition.

### Counterfactual via class siblings

When the experiment asks "what if I had taken action B at moment M instead of A?", it walks up to M's parent class, finds sibling moments under that class where action B was the active action, and replays forward from one. Class siblings differ in action and outcome but share context. This is exactly the substitution semantics counterfactuals require, and it falls out of the class hierarchy at no extra cost.

### Action writeback (the genius part)

When an experiment converges on a long-term-optimum action for a context, the hippocampus writes that action onto the moment neuron whose context it applies to. Concretely: the moment's decided-action slot (or the corresponding action-neuron connection) is set to the chosen action. From that point forward, whenever the moment is activated by context match — voluntarily through a think-action or involuntarily through the forecast pass — its decided action votes into the cortex's action selection at that frame.

If the conclusion generalizes across a class's children, the writeback lands on the class neuron, and any future context matching the class inherits the decision.

This is what makes System 2 enrich System 1. The hippocampus does deliberate analysis; the result becomes a cortical neuron that fires reflexively when its context recurs. Repeated experience of the same class hardens the habit.

---

## Decided vs. probabilistic action voting

This is the structural difference between moment/class neurons and ordinary sensory/pattern neurons:

- A **sensory or pattern neuron** votes for actions probabilistically — the votes are accumulated statistical evidence, smoothed across many experiences. Many neurons vote; the cortex's existing action-selection machinery aggregates.
- A **moment or class neuron** carries a *decided* action written by deliberate experimentation. When activated, it votes its decided action with high authority — black-and-white, the conclusion of analysis. A decided vote dominates probabilistic votes for the same action slot.

The two operations look the same from the cortex's perspective (a neuron firing contributes to action selection), but the underlying mechanism and timeline differ. Sensory/pattern neurons learn slowly, statistically, over long experience. Moment/class neurons learn one-shot, deliberately, via experiment.

---

## Forgetting

The same death-ledger mechanism applies to all neuron kinds, but moment and class neurons age on their own schedules, distinct from sensory/pattern neurons.

**1. Per-link decay (inside a moment's context connections).**
Each context-connection strength decays independently:
```
strength(link, t) = strength(link, t-1) * decay_rate_link + activation_boost_if_active_now
```
Initial value is the cortical neuron's activation at minting time. Strongly active context neurons survive far longer than weakly active ones. When a link's strength falls below `link_eviction_threshold`, that connection is dropped from the moment.

**2. Moment-level decay.** A moment's overall `strength` is initialized from its salience at minting and decays per cortex frame. Activation boosts it. Below `moment_eviction_threshold`, or when capacity is pressured, the lowest-strength moments are evicted. A moment is also evicted if its surviving context-connection count drops below `min_context_refs`.

**3. Class-level decay — slower.** Classes have their own decay loop with `decay_rate_class > decay_rate_moment`. Activation boosts apply whenever the class is activated — by its children, by cue resolution, by being a transition target in an experiment, or by being a parent of an activated class. This is "thinking to keep remembering": classes used in many experiments live indefinitely, even after every instance moment under them has been evicted. When the last child is evicted, the centroid is frozen at its last computed state and the class persists as an abstract reward-carrying signature.

**4. Cascade on cortical deletion.** When a context neuron is deleted, every moment removes that connection and every class removes that centroid key. Moments dropping below `min_context_refs` are evicted. Classes are not auto-evicted on cortical deletion — they keep living by their own strength.

**5. Death ledger.** Every eviction appends `{neuron_id, kind, frame, reason, salience_at_birth}` to the shared append-only ledger (ring-bounded). Uses:
- Debugging — "why don't I remember X?"
- Observability — shape of forgetting over time.
- Re-encoding suppression — don't immediately re-mint a just-evicted moment unless its salience is now substantially higher.

**6. Sleep / idle consolidation.** When the hippocampus has free cycles, it replays high-salience moments to reinforce them, prunes low-strength moment/class neurons, and rebalances class clusters (merge / split / re-parent). The biological-sleep-replay analog.

---

## The brain coordinator

Orchestrates the parallel clocks. Manages semaphores. Routes inputs and outputs.

**Per cortex frame:**
1. Read sensory inputs; cortex propagates patterns and picks actions.
2. Wait on `SignalCortexDone`.
3. If salience triggered, push a mint request to the hippocampus.
4. If cortex emitted think-actions, push them as voluntary experiments.
5. Push the top-level active cortical patterns as an involuntary forecast request.
6. Continue to the next frame. (Decided-action writebacks land directly on moment/class neurons via the thalamic bus; nothing to drain here.)

**Per hippocampus frame (faster, in parallel):**
1. Service any pending mint request (cheap, runs immediately).
2. If an experiment is on the stack, advance one frame.
3. Otherwise pop the next request: voluntary think-actions take priority over involuntary forecasts.
4. If idle, run consolidation: decay, eviction, death-ledger maintenance, optional class re-clustering.

---

## Coordinator pseudo-code

```
// shared
SignalCortexDone:    semaphore (cortex → brain: frame done)
MintQueue:           channel<MintRequest>
ExperimentQueue:     channel<ThinkAction>     // voluntary, high priority
ForecastQueue:       channel<MomentSet>       // involuntary, low priority

// cortex thread (1 frame per tick)
loop:
    inputs = read_sensory()
    cortex.activate(inputs)                  // sensory, pattern, moment, class neurons
                                              //   all fire by context match; moment/class
                                              //   decided votes dominate sensory/pattern
                                              //   probabilistic votes
    cortex.propagate_patterns()
    actions = cortex.pick_actions()          // may include think-actions
    cortex.emit(actions)
    SignalCortexDone.post()

// brain thread (orchestrator)
loop:
    SignalCortexDone.wait()
    if salience.should_mint(cortex.last_frame):
        MintQueue.push({active_ids, frame, trigger})
    for ta in actions.think_actions:
        ExperimentQueue.push(ta)
    ForecastQueue.push(cortex.top_level_active_patterns())

// hippocampus thread (free-running, faster clock)
loop:
    while req = MintQueue.try_pop():
        hippocampus.mint_moment(req)         // creates a moment neuron in the right column,
                                              //   wires context connections, attaches to
                                              //   class hierarchy, draws temporal edges

    if hippocampus.experiment_stack.has_active():
        hippocampus.advance_experiment_frame()
        // any decided-action writeback during advance lands directly on the target
        //   moment/class neuron via thalamic addressing
        continue

    if ta = ExperimentQueue.try_pop():
        hippocampus.start_experiment(ta)
        continue
    if forecast = ForecastQueue.try_pop():
        moments = hippocampus.activate_by_context(forecast)
        if moments.non_empty():
            hippocampus.start_experiment(forecast_as_replay(moments, small_budget))
        continue

    // idle
    hippocampus.decay()
    evicted = hippocampus.evict_below_threshold()
    for n in evicted:
        DeathLedger.append({n.id, n.kind, current_frame, reason, n.salience_at_birth})
    hippocampus.cascade_cortical_deletions()
    hippocampus.consolidate_classes()
```

---

## Heterarchy

A cortical neuron can be inside many moments' contexts and many classes' centroids. A moment can be a child of many classes. The reverse mapping `cortical_neuron_id → [moment_ids, class_ids]` is maintained for cue resolution and Jaccard search during minting.

This breaks strict cortical hierarchy and gives the system *"that reminds me"* — when active patterns overlap a moment from a different context, it surfaces.

---

## Stack semantics for "what was I thinking about?"

The experiment stack is the mechanism. Each entry has a timestamp; entries decay over time. Querying "what was I thinking about?" returns the topmost stack item still above decay threshold, optionally re-triggered as a new experiment. If everything has decayed, the thought is lost — the right phenomenology.

---

## Implementation phases

### Phase 1 — Skeleton and parallel clocks

- Create `hippocampus` module.
- Add moment-neuron and class-neuron kinds to the column/region types so they live alongside sensory/pattern neurons. Reuse the existing thalamic id-translation layer.
- Implement the brain coordinator with two-thread async (cortex + hippocampus, semaphore-coordinated).
- Implement `mint_moment` driven by simple z-score salience (reward and prediction error only, no novelty).
- Hard-code a trivial temporal graph (just record edges) and a trivial replay (return current moment).
- Verify: cortex fires a think-action, hippocampus runs experiment frames, decided action writeback lands on a moment neuron, that moment's decided action shows up in cortex's next action selection.

### Phase 2 — Salience and selective minting

- Z-scored reward and prediction error over sliding windows.
- Gating (mint iff `|z|` over threshold).
- Strength-based decay and eviction on the moment timeline.
- Death ledger and re-mint suppression.

### Phase 3 — Class formation (online agglomerative weighted Jaccard)

- Inverted index `cortical_neuron_id → [class_ids]`.
- `attach / merge / spawn` decisions on each mint.
- Class centroid as weighted running intersection of children's contexts.
- Multi-level hierarchy (classes clustered against classes at lower thresholds).
- Child→parent activation propagation.
- Class-timeline decay (slower than moment timeline).

### Phase 4 — Temporal moment graph and replay

- On each mint, draw weighted edges to recent moments.
- Edge-weighted sampling for next-state prediction.
- `mode: Replay` end-to-end.

### Phase 5 — Counterfactual experiments and decided-action writeback

- Branch via class sibling: find sibling under parent class with alternative action, replay forward.
- Stack-based push/pop for branches.
- Write decided actions to specific moment/class neurons at experiment conclusion.
- Verify: replaying a bad outcome with alternatives discovers a better policy; subsequent encounters of the same context use the improved policy via the decided-action vote dominating sensory/pattern probabilistic votes.

### Phase 6 — Involuntary forecast pass

- `activate_by_context` (set overlap with class-membership boost, no distances).
- Wire the per-frame forecast push.
- Verify: with zero think-actions, the hippocampus produces background forecasts and decided-action updates every frame.

### Phase 7 — Voluntary think-actions and metacognitive control

- Full think-action parameters (mode, budget, temperature, control).
- Interrupt / Integrate / Remember semantics.
- Reward signals for think-actions (PE reduction, downstream action improvement, opportunity cost).
- Train cortex to fire think-actions when expected value exceeds external action value.

### Phase 8 — Sleep / idle consolidation

- Detect idle.
- High-temperature integrative replay over recent high-salience moments.
- Class re-clustering (merge, split, re-parent).
- Reinforce strong moments, prune weak ones.

### Phase 9 (deferred) — Parallel experiments

Move from single-track-with-stack to a pool of N concurrent experiments sharing a "what's been visited" structure to prevent redundant exploration. Optimization, not correctness — Phase 1 is strictly single-track.

---

## Testing strategy

### Unit tests
- Moment / class neuron minting, retrieval, capacity, decay, eviction, death ledger.
- Jaccard clustering: attach, merge, spawn; centroid as running intersection.
- Temporal graph: edge construction, weighted sampling.
- Experiment stack: push, pop, decay, recovery.
- Salience: z-score thresholding over sliding windows.
- Decided-vote-dominates-probabilistic-vote action selection.

### Integration tests
- Full mint → class attach → temporal edge → replay → decided-action writeback flow.
- Counterfactual via class sibling produces a better-than-original alternative and the moment's decided action reflects it.
- Involuntary forecast fires every frame with no cortex prompting.
- Voluntary think-action interrupts in-flight forecast and resumes after.
- Hippocampus removal experiment: a fully populated cortex with the hippocampus disabled still does cued recall and skill execution but cannot mint new moments, plan, or run counterfactuals (HM profile).

### Behavioral tests
- Trading backtest with vs. without hippocampus on bad-trade scenarios.
- ADHD-like profile: salience too lax, observe distractibility.
- Rigidity profile: salience too tight, observe perseveration.

### Stress tests
- High-throughput cortex with many think-actions.
- Capacity limits and graceful eviction.
- Cortical deletion cascades.

---

## Open questions

1. **Jaccard thresholds.** `θ_attach`, `θ_merge`, `θ_spawn` need tuning. Start with 0.6 / 0.85 / 0.5; adjust by observing class-tree shape on real data.
2. **Salience window length.** Probably 100–1000 cortex frames; tunable.
3. **Hippocampus frame rate.** Multiplier vs. cortex. Biology suggests 5–20×; start at 10×.
4. **The actual instruction set the cortex uses to invoke the hippocampus.** This will emerge from implementation rather than be designed up front. The conceptual operations are listed above; their precise signatures and the cortex-side action neurons that trigger them are TBD.
5. **Persistence.** Do moment/class neurons persist across brain restarts? For long-term memory to be meaningful, yes. Need serialization for the column state plus the death ledger and temporal graph.
6. **Cross-channel moments.** Can a single moment's context include cortical neurons from multiple channels (stock + text + vision)? Strongly suggests yes — this is where cross-channel association happens. Interface complexity grows.
7. **Decided-vote dominance arithmetic.** Exactly how a decided vote suppresses or overrides probabilistic votes for the same action slot — full override, weighted override, or a multiplicative authority factor — needs implementation to settle.
8. **Search-policy for experiments.** Time-budgeted best-first is the default. The scoring function needs to fall out of existing primitives (prediction error against accumulated path context, predicted reward) rather than being a hand-tuned linear combination of heuristics.

---

## Why this matters

- **Long-term memory** as cortical neurons minted by deliberate experimentation — not as a separate store, not as a context window.
- **Two thinking modes** — involuntary background forecasting + voluntary think-actions — both running on the same machinery.
- **System 2 enriches System 1** — hippocampal experiment outputs become cortical neurons that fire reflexively. Expertise development is structural, not behavioral.
- **Decided actions** — moments and classes carry the conclusions of analysis, not statistics. Reflexes hardened by deliberation, voting with authority.
- **Forgetting that judges** — instances fade fast, classes persist, the death ledger remembers what was lost.
- **Biological grounding** — implements hippocampal indexing, complementary learning systems, predictive processing, scenario construction, and FCA-style abstraction in one architecture, with HM as a direct architectural prediction.
- **One substrate, two operators, thalamic bus.** Cortex is the store. Hippocampus is the executor. Thinkability is reachability.

The thesis: the next generation of AI architectures will not come from scaling sequence models. It will come from systems that have the structures biological brains have — separate organs for perception, memory, and thinking, coupled bidirectionally and operating on different clocks over a shared substrate.
