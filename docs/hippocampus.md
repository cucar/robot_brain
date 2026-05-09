# Hippocampal Region — Design and Implementation Plan

## Overview

Robot Brain is a dual-system predictive architecture. Two organs run continuously in parallel over the same cortical substrate:

- **Cortex (System 1)** — fast, reflexive, short-term. Builds patterns by intersection over recurring contexts. Predicts the next frame, picks short-term actions by accumulated statistical voting from firing patterns.
- **Hippocampus (System 2)** — slow, deliberate, long-term. Mints moments (multi-parent nodes formed by union over the active set at salience triggers), runs replay experiments over them, and reinforces action connections on moments along trajectories that yielded long-term-optimum rewards.

The deep insight that organizes this design:

> **Cortex splits reality by intersection. Hippocampus binds reality by union. Same substrate, two creation rules.**

There is one neuron kind that lives in cortical columns: a node with a context fingerprint, outgoing connections to other nodes and to action neurons, and metadata (mint frame, kind tag, salience-at-birth, strength). What distinguishes a moment from a pattern is *how it was created* and consequently *how many parents it has* — not what kind of neuron it is.

- **Patterns** are formed by *intersection*: the cortex finds the subset of co-active context that consistently predicts an outcome. Patterns typically have one or few parents in the abstraction hierarchy. Created gradually by statistical learning.
- **Moments** are formed by *union*: at a salience trigger, the hippocampus binds every currently-active high-level pattern as a parent. Moments have many parents by construction. Created in one shot.

Storage is always cortical. The hippocampus is the operator that mints moments and runs experiments — it owns no permanent store of its own.

### Moments age into classes

A moment is born with many parents (the union at mint time). Per-link decay applies independently to each parent connection. Use-driven reinforcement strengthens links that match real recurring contexts. Over time, weak (incidental) parent links drop out; strong (real-context) parent links survive and sharpen. The moment ends up connected only to its core parents — a class-shaped node, structurally indistinguishable from a statistically-formed pattern.

Classes are not a separate neuron kind. They are aged moments. The forgetting curve is the abstraction mechanism. This is the architectural analog of episodic-to-semantic consolidation as observed empirically: vivid specific memories lose incidental detail and survive as gist.

### Thinkability = reachability by the executor

Ordinary cortical neurons fire when their context matches and otherwise sit dormant — they are not deliberately summonable. Moments are different: the hippocampus can address them by id (via the thalamic translation layer), reactivate them, traverse their connections, and run experiments over them. **A neuron is "thinkable" iff the hippocampal executor can reach it.** Thinkability is not a property of the neuron's location or type; it is a property of being operated on by the hippocampus.

### The HM test

This architecture predicts HM directly. Removing the hippocampus removes the executor. After removal:

- Pre-existing moments survive — they are cortical, with stable connections, reactivatable by sensory cues (cued recall).
- New moments cannot be minted — the union-creation operation is gone.
- Deliberate recall, planning, and counterfactual reasoning are gone — these required executor reachability for replay.
- Skills, perception, statistical learning, and cued recall continue normally — these are pure cortex.

If a Robot Brain with a fully populated cortex loses its hippocampus, that is the behavioral profile it should exhibit.

---

## Theoretical grounding

- **Hippocampal indexing theory** (Teyler & DiScenna 1986; Teyler & Rudy 2007) — the hippocampus indexes cortical patterns, doesn't store them. This design takes the further step: the indices themselves are also cortical (moments), and the hippocampus is purely operator.
- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995) — fast experimental learning (System 2) + slow statistical learning (System 1), bridged by experiment-driven action reinforcement.
- **Mattar & Daw (2018)** — replay prioritized by expected value of backup (prediction error magnitude).
- **Buzsáki's "brain from inside out"** — the brain as a generative system; replay/preplay are the same intrinsic dynamics under different conditions.
- **Kahneman dual-process theory** — Robot Brain implements System 1 and System 2 as separate organs over a shared substrate, rather than as a behavioral metaphor.
- **Hippocampal scenario-construction literature** — patients with hippocampal damage cannot imagine novel scenes either, not just remember old ones. The hippocampus is the executor; memory is one of its uses.
- **Episodic-to-semantic consolidation** — empirical finding that specific episodic memories schematize into gist over time. In this architecture, this is a structural consequence of per-link decay on multi-parent moments under use-driven reinforcement.

---

## Architectural principles

1. **One substrate, two operators.** Cortex and hippocampus operate on the same cortical neurons in the same columns. Thalamus mediates id↔property addressing.
2. **Cortex splits by intersection, hippocampus binds by union.** Cortex divides experience into hierarchical patterns through prediction-error-driven statistical learning. Hippocampus binds salient instants into multi-parent moment nodes in one shot.
3. **One neuron kind in routing tables.** Patterns and moments are the same kind of node. They differ in how they were created (intersection vs union), in fan-in (typically few parents vs many parents), and in the temporal grain over which their action connections are organized.
4. **Hippocampus is the executor, not the store.** It mints moments, runs experiments over them, and updates action connections on moments. It does not hold them.
5. **Recall is by context.** A moment fires when its parent patterns fire with sufficient context overlap. This is normal cortical activation, not a special operation.
6. **One voting machinery.** All firing nodes vote for actions through their action connections, weighted by connection strength. Moments and patterns vote identically; experiment-driven updates strengthen the same connections that statistical learning strengthens. There is no separate "decided action" override mechanism.
7. **Selective minting.** Only frames with reward or prediction error above a z-score threshold trigger moment minting. Ordinary experience never becomes thinkable.
8. **Two clocks.** The hippocampus runs at a faster clock than the cortex, executing many experiment frames per cortex frame.
9. **Same death ledger, different aging.** Forgetting is unified through one death ledger. Moments age on a slower timeline than patterns because they represent rarer, more salient information.
10. **Bidirectional coupling via thalamus.** Cortex emits think-actions that reach the hippocampus. Hippocampus reads currently active cortical state and writes action-connection updates back to specific moment neurons. Updates are local to the neuron written, never global.

---

## The cortical substrate

A cortical node, regardless of whether it was minted as a pattern or as a moment, has:

- A **context fingerprint**: incoming connections from other nodes (sensory neurons, patterns, moments — anything in the substrate), each weighted by activation strength at the node's creation time and updated by use-driven reinforcement.
- **Outgoing connections to other nodes**: routing-table entries pointing to higher-level patterns the node participates in, and (for moments) to other moments via the temporal moment graph.
- **Outgoing connections to action neurons**, organized into temporal bins (see "Action connections and temporal bins" below).
- Metadata: mint frame, kind tag (pattern/moment), salience at birth, current strength, last-access frame.

When the node's context matches the current frame within tolerance, it fires and contributes its votes to action selection. The cortex does not distinguish patterns from moments at activation or voting time — both fire by the same rule, both vote into the same aggregation.

What distinguishes patterns from moments structurally:

- **Patterns** were minted by gradual statistical co-activation. They typically have few parents (the higher-level patterns they roll up into). Their action connections live on a *short-term temporal bin scheme* (linear, frame resolution, narrow horizon — e.g., 10 bins covering 1-10 frames).
- **Moments** were minted by the hippocampus by union over the active high-level pattern set at a salience trigger. They have many parents at birth. Their action connections live on a *long-term temporal bin scheme* (exponential, broad horizon — e.g., 10 bins covering 1 to 10^10 frames).

Both kinds participate in normal cortical activity. The hippocampus's special access is by id, not by kind.

---

## Action connections and temporal bins

Every node has outgoing action connections, organized by temporal bin. The bin scheme is a per-kind parameter.

### Bin scheme

- Patterns use **linear bins** at frame resolution. With N=10 bins, each bin holds the connection strength for actions taken at exactly that frame offset (1, 2, ..., 10) following the pattern firing. This matches cortex's existing short-term action prediction machinery.
- Moments use **exponential bins** over a wide horizon. With N=10 bins, the bin boundaries are 1, 10, 100, ..., 10^10 frames. Bin k holds the connection strength for actions whose effective offset from the moment's firing falls within (10^(k-1), 10^k] frames.

The number of bins (N) is a parameter, settable per kind. 10 is a reasonable default for both.

### Why exponential at long horizons

Linear bins at frame resolution would require billions of bins to cover meaningful long-term horizons. Heavy-tailed Δt distributions (which is what natural experience produces) want log-spaced binning. Discriminability of time intervals is logarithmic — humans don't distinguish "3000 frames ago" from "3050 frames ago" but sharply distinguish "yesterday" from "last week." Exponential bins are the architecture noticing this.

### Soft bin assignment

Hard bin boundaries cause fragility at edges (an action with effective offset 95 frames sits at the bin 1 / bin 2 boundary; small noise could flip it). Each action's vote is therefore distributed smoothly across adjacent bins via a kernel — typically a Gaussian over log-Δt — rather than placed in a single bin. Bin assignment is a distribution, not a categorical choice.

### Action intrinsic horizon

Each action neuron learns an **intrinsic horizon distribution**: the distribution of reward-arrival-Δts following the action's firing, aggregated over many firings. Some actions are unambiguously short-horizon (consequences arrive within a few frames). Some are unambiguously long-horizon (consequences arrive consistently far out). Some are mixed.

The action's intrinsic horizon determines which bin's connection strength is read when the action is being considered. A moment voting for "submit market order" pulls from its short-bin (or whichever bin matches "submit market order"'s intrinsic horizon). A moment voting for "shift portfolio strategy" pulls from its long-bin. The action's intrinsic horizon is the bin selector — there is no separate "current planning horizon" parameter held by the cortex.

This means strategic vs reflexive action selection happens automatically: the right moments fire because the right contexts arose, they have strong connections in the appropriate bins for the actions being considered, and the votes aggregate. No planning module required.

---

## The hippocampal executor

The hippocampus owns no permanent store. It owns:

- The salience module (windowed z-scoring of reward and prediction error).
- The experiment buffer (single experiment in flight in Phase 1; parallel pool in a later phase).
- A transient working set of currently-relevant moment ids while an experiment runs.
- The death-ledger writer (the ledger itself is shared across cortex and hippocampus).

Its operations:

- **Mint moment** — given the currently active cortical state and a salience trigger, create a moment node in the appropriate column with connections to every active high-level pattern (with weight proportional to each parent's activation strength), action connections recording every action that fired during the encoded context window (with frame offsets), and edges to recently-active moments (the temporal moment graph).
- **Replay** — wavefront walk through the moment graph, accumulating simulated time, reading rewards and action sequences along the trajectory.
- **Branch (for counterfactual)** — at any replay step, instead of sampling forward via edges, substitute an alternative action and walk to a sibling moment under shared parent patterns; continue from there.
- **Reinforce action connections** — at experiment conclusion, strengthen action connections on moments along high-reward trajectories, in the temporal bin matching the summed Δt at which the lesson applies.
- **Decay and evict** — apply moment-timeline decay; emit eviction records to the death ledger.

The cortex does not know about the moment/pattern distinction. It just sees nodes firing and contributing votes. The hippocampus is the only thing that knows which nodes are moments and what experiments are running.

---

## The salience module

Computes a scalar "mint this moment" signal per cortex frame:

- `z_reward = (reward − μ_reward) / σ_reward` over a sliding window
- `z_pe = (prediction_error − μ_pe) / σ_pe` over a sliding window
- Mint iff `|z_reward| > θ_r` or `|z_pe| > θ_pe`

Novelty is not an input. Every observation contains some novelty; that does not make it worth remembering. What the literature calls "novelty effects" is in practice prediction error, already covered.

When the trigger fires, the hippocampus calls Mint over the currently active cortical state.

In addition to salience-triggered minting, a **sampling rate** parameter mints occasional non-salient moments to provide a baseline graph of ordinary experience. This prevents the moment graph from being entirely composed of high-stakes nodes with no connective tissue between them. Sampled moments age and decay normally; if they don't get reactivated, they fade out without consequence.

---

## Moment creation

When a salience trigger fires, the hippocampus mints a new moment by union:

1. Read the currently active high-level patterns (those firing above some activation threshold).
2. Create a new moment node in the appropriate cortical column.
3. For each active pattern P, register the new moment as an outgoing connection in P's routing table, with link weight initialized to P's activation strength.
4. Record action connections for every action that fired during the encoded context window, with frame offsets, in the moment's long-term temporal bin scheme.
5. Draw edges to the K most recently-minted moments (the temporal moment graph), with edge weights initialized by `1/Δt` and Δt metadata recorded directly.
6. Set salience-at-birth from the triggering z-score; initialize moment strength accordingly.

The moment now exists as a multi-parent node in the cortical graph. Future cortical pattern-formation can incorporate this moment into its contexts, exactly like sensory or pattern neurons. This is the structural mechanism for "lessons from specific experiences becoming reflexes" — System 2 enriches System 1 by adding moment nodes that cortex then abstracts over.

### Cold-start gating

A freshly-minted moment has no replay history and no use-driven reinforcement on its links. It does not contribute to action voting until it has been activated K times (parameter, default K=3). Before that threshold, its action connections are silent — it accumulates evidence but does not project. This parallels the cold-start gating already used for newly-formed pattern neurons.

### Moments age into classes (mechanism)

After minting, per-link decay reduces parent-link strengths over time. When the moment reactivates because some of its parents fire with matching context, those matched links get strengthened by use-driven reinforcement. Unmatched links continue to decay. Over time:

- Incidental parent connections (parents that happened to be active at mint time but don't recur with this moment) decay away.
- Real parent connections (parents that capture the moment's actual context) get repeatedly reinforced.

The moment ends up sparsely connected to its core parents. Structurally, it has become a class — a node representing "this kind of situation" rather than "this specific instant." No separate machinery, no clustering pass, no centroid computation. The decay and reinforcement loops do the abstraction work over the moment's lifetime.

This means there is no class neuron type. There are only moments at various stages of life. Young moments are episodic; old moments are semantic. Use-driven selection on a multi-parent substrate is the consolidation mechanism.

Important: do not artificially renormalize per-link weights as link count drops. That would keep dead moments alive — moments that captured noise and never re-matched would still appear to fire because their few remaining links got boosted. The honest signal is "did this moment keep getting hit?" Useful moments survive and sharpen; useless moments fade. No compensation.

---

## The temporal moment graph

When a moment is minted, edges form to the K most recently minted moments. Edge weight is initialized by `1/Δt`. Δt is recorded as metadata.

Edges also form via co-reactivation: when two existing moments fire close in time during normal cortical operation or during replay, an edge is created (if absent) or reinforced (if present). Edge Δt metadata accumulates a small histogram over the long-term bin scheme (rather than a single mean), so an edge whose firings cluster bimodally at Δt=5 and Δt=500 carries mass in two bins.

Edges decay by disuse, like everything else in the substrate. Edges below a strength threshold are dropped.

The graph is directed. M_source's routing table points to M_target with Δt-histogram metadata. M_target does not necessarily point back. Backward traversal, when needed, requires explicit reverse edges (cheap to add at edge-creation time if desired).

This graph is the substrate replay walks. It is also what gives "this kind of situation tends to lead to that kind of situation" structure — the system's learned long-horizon transition model.

---

## Recall — the involuntary forecast pass

After cortex finishes its frame, any moment whose parents are firing with sufficient context overlap will already be active — this is normal cortical activation, not a special operation. The hippocampus reads the currently-firing moment set and, for each moment above an experiment-eligibility threshold:

- Notes that the moment is firing (it is already voting via its action connections; nothing extra needed).
- May initiate a small-budget replay experiment from the active moment set.

This pass fires every cortex frame, even with zero think-actions. It is the always-on background channel — *"that reminds me…"*, mind-wandering, gut-feel forecasting.

When a voluntary think-action arrives mid-forecast, the forecast is pushed onto the experiment stack and the think-action runs. The forecast resumes after.

---

## The think-action interface (voluntary thinking)

The cortex fires think-actions as ordinary actions. The cortex does not address moments by id — its world is nodes and patterns. A think-action says "think about *this cortical content*"; the hippocampus resolves the cue to candidate moments by driving the cue and reading off which moments fire.

```
think_action {
  cue: [cortical_neuron_id, ...],   // cortical content to think about — patterns,
                                     //   sensory ids, action ids, or any mix.
  mode: enum {
    Replay,                          // run forward as-original
    Counterfactual,                  // inject alternative actions and explore
    Retrieve,                        // surface related moments via shared parents
    Compare,                         // run two cued states in parallel, measure delta
    Compress,                        // find common structure across cued moments
  },
  budget: integer,                  // hippocampus frames to spend
  horizon: integer,                 // max simulated time (summed Δt) in frames
  temperature: float (0.0 - 1.0),   // breadth of sampling (low=peaked short-Δt;
                                     //   high=associative drift to long-Δt)
  control: enum {
    Interrupt,                       // push current experiment, start new
    Integrate,                       // add cue to current experiment
    Remember,                        // mint without starting an experiment
  }
}
```

**Cue → moment activation** (inside the hippocampus):

1. Drive the cue neurons.
2. Let cortex pattern-complete; collect the moments that fire above threshold.
3. The active set becomes the experiment's starting state.

This is the same machinery as the involuntary forecast pass. The only difference is the source of the cue — cortex's currently-active high-level patterns for involuntary, the explicit `cue` parameter for voluntary.

Each parameter is independently learnable through the metacognitive reward signal (Phase 7).

---

## Experiment execution

### Starting state

An experiment starts with a *set* of currently-active moments — whatever was firing in the surfacing context (involuntary) or whatever the cue resolved to (voluntary). The replay does not assume a single starting moment.

### Single replay frame (wavefront walk)

1. **Active set.** Read the currently active moment set from the top of the experiment stack.
2. **Step forward.** For each moment in the active set, sample outgoing edges weighted by `edge_strength × context_overlap × Δt_kernel(temperature)`. The Δt kernel concentrates on short-Δt edges at low temperature (near-future prediction at fine grain) and broadens to long-Δt edges at high temperature (far-future associative drift at coarse grain). The union of sampled targets, weighted by their firing strength, becomes the new active set.
3. **Accumulate simulated time.** Sum the Δts of edges traversed; this is the experiment's simulated horizon so far.
4. **Evaluate.** For each moment in the new active set, read off rewards (via the cortex's value estimates for the moment's context) and action information (the moment's action connections).
5. **Continue, branch, or terminate** (see termination conditions below).

### Termination conditions

The experiment stops when *any* of:

- **Budget exhausted** — the requested number of replay frames has been used.
- **Horizon exceeded** — the summed simulated Δt has passed the requested horizon. Useful for "simulate one hour ahead, stop" semantics.
- **Convergence** — predicted reward across additional steps stops changing meaningfully, or the wavefront has entered a loop (revisiting moments already in the trace).
- **Salience drop** — the running average of activated moment strengths drops below X% of the starting strength. The replay has drifted into uninteresting territory.

For involuntary experiments without an explicit budget or horizon, salience drop is the dominant terminator. The replay walks forward as long as each step keeps surfacing reasonably-strong moments; when the trail goes cold, it stops. This matches the phenomenology of mind-wandering: associative chains fade out when the activations weaken.

For voluntary think-actions, all four conditions apply, with budget and horizon caps coming from the request.

### Counterfactual via parent-pattern siblings

When the experiment asks "what if a different action had been taken at this moment in the trajectory?", it walks from the moment in question up to its parent patterns, then down to *other* moments under those same parents where a different action was taken. Replay forward from one of those sibling moments. Same parent patterns means similar context; different action means actual counterfactual.

If no sibling moments exist for the desired alternative action, the experiment can fall back to sampling actions from the cortex's probabilistic action votes for the moment's context — "what would cortex have done absent habit?" — but this is less precise than sibling substitution.

### Action reinforcement (writeback)

When an experiment converges on a long-term-optimum trajectory, the hippocampus reinforces action connections on moments along the trajectory. For each moment M_b in the trajectory at simulated-time-offset Δt from the start:

1. Determine the long-term temporal bin matching Δt under the soft bin assignment kernel.
2. Strengthen M_b's connection to the trajectory-optimum action in those bins.

The strength of the reinforcement is calibrated against ordinary statistical action-vote strength — strong enough to influence behavior, not so strong it overrides recent direct experience. The exact calibration constant is a tunable.

Generalization is left to the substrate. Don't try to write the lesson onto multiple moments simultaneously — write it onto the moment where the substitution applies. Over time, when M_b reactivates in similar contexts and votes its reinforced action successfully, the cortex builds patterns over the situations where that vote helped, and those patterns inherit the lesson through normal cortical abstraction.

### Wavefront writeback granularity (open)

When the trajectory is a wavefront rather than a single thread, multiple moments are active at each step. The current default is to reinforce on the strongest-firing moment at the relevant step, but the precise rule (all members weighted by firing strength? only those whose stored actions matched the substituted action?) needs to be specified in implementation. See Open Questions.

### Stack semantics

Branching is implemented as an experiment stack: branch = push, terminate = pop. A higher-priority think-action arriving mid-experiment pushes the current one and runs.

The stack also supports "what was I thinking about?" queries: each entry is timestamped; entries decay over time. Querying returns the topmost item still above decay threshold. If everything has decayed, the thought is lost — the right phenomenology.

---

## Forgetting

The same death-ledger mechanism applies to all node kinds. Moments age on their own (slower) timeline, distinct from patterns.

**1. Per-link decay.** Each context-connection strength decays independently:
```
strength(link, t) = strength(link, t-1) * decay_rate_link + activation_boost_if_active_now
```
Initial value is the parent neuron's activation at minting time. Strongly active context neurons survive far longer than weakly active ones. When a link's strength falls below `link_eviction_threshold`, the connection is dropped.

This is the mechanism that drives moments-becoming-classes. Do not apply renormalization or compensating weights as link count drops; let the use signal speak honestly.

**2. Moment-level decay.** A moment's overall strength is initialized from its salience at minting and decays per cortex frame. Activation boosts it. Below `moment_eviction_threshold`, or when capacity is pressured, the lowest-strength moments are evicted. A moment is also evicted if its surviving link count drops below `min_context_refs`.

**3. Action-connection decay.** Action connections in each temporal bin decay independently and are reinforced by use (either statistical reinforcement from observed action-reward pairs, or experiment-driven reinforcement from replay). Connections in bins that never get reinforced fade out.

**4. Cascade on cortical deletion.** When a context neuron is deleted, every node removes that connection. Moments dropping below `min_context_refs` are evicted.

**5. Death ledger.** Every eviction appends `{neuron_id, kind, frame, reason, salience_at_birth}` to the shared append-only ledger (ring-bounded). Uses:
- Debugging — "why don't I remember X?"
- Observability — shape of forgetting over time.
- Re-encoding suppression — don't immediately re-mint a just-evicted moment unless its salience is now substantially higher.

**6. Sleep / idle consolidation.** When the hippocampus has free cycles, it replays high-salience moments to reinforce them and prunes low-strength moments. This is the biological-sleep-replay analog. The class-rebalancing pass from the previous design is gone — there are no classes as separate entities to rebalance.

---

## The brain coordinator

Orchestrates the parallel clocks. Manages semaphores. Routes inputs and outputs.

**Per cortex frame:**
1. Read sensory inputs; cortex propagates patterns and picks actions. Currently-active moments fire normally and contribute their votes to action selection through the same machinery as patterns.
2. Wait on `SignalCortexDone`.
3. If salience triggered, push a mint request to the hippocampus.
4. If cortex emitted think-actions, push them as voluntary experiments.
5. Push the currently-active moment set as an involuntary forecast request.
6. Continue to the next frame.

**Per hippocampus frame (faster, in parallel):**
1. Service any pending mint request (cheap, runs immediately).
2. If an experiment is on the stack, advance one frame.
3. Otherwise pop the next request: voluntary think-actions take priority over involuntary forecasts. If the queue is full and a non-interrupt request arrives, drop it silently — this is the "deep in thought already, can't be bothered" state.
4. If idle, run consolidation: decay, eviction, death-ledger maintenance.

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
    cortex.activate(inputs)                  // all nodes (sensory, pattern, moment)
                                              //   fire by context match; all vote
                                              //   uniformly into action selection
    cortex.propagate_patterns()
    actions = cortex.pick_actions()          // may include think-actions
    cortex.emit(actions)
    SignalCortexDone.post()

// brain thread (orchestrator)
loop:
    SignalCortexDone.wait()
    if salience.should_mint(cortex.last_frame):
        MintQueue.push({active_high_level_patterns, frame, trigger})
    for ta in actions.think_actions:
        ExperimentQueue.push(ta)
    ForecastQueue.push(cortex.currently_active_moments())

// hippocampus thread (free-running, faster clock)
loop:
    while req = MintQueue.try_pop():
        hippocampus.mint_moment(req)         // creates moment node, registers as
                                              //   child in active patterns'
                                              //   routing tables, draws temporal
                                              //   edges, records initial actions

    if hippocampus.experiment_stack.has_active():
        hippocampus.advance_experiment_frame()
        // any action-connection writeback during advance lands directly on
        //   the target moment via thalamic addressing
        continue

    if ta = ExperimentQueue.try_pop():
        hippocampus.start_experiment(ta)
        continue
    if forecast = ForecastQueue.try_pop():
        if forecast.non_empty():
            hippocampus.start_experiment(forecast_as_replay(forecast, small_budget))
        continue

    // idle
    hippocampus.decay()
    evicted = hippocampus.evict_below_threshold()
    for n in evicted:
        DeathLedger.append({n.id, n.kind, current_frame, reason, n.salience_at_birth})
    hippocampus.cascade_cortical_deletions()
```

---

## Implementation phases

### Phase 1 — Skeleton and parallel clocks

- Create `hippocampus` module.
- Add the moment kind tag to the existing cortical node type so moments live alongside patterns in the same columns. Reuse the existing thalamic id-translation layer.
- Implement the brain coordinator with two-thread async (cortex + hippocampus, semaphore-coordinated).
- Implement `mint_moment` driven by simple z-score salience (reward and prediction error only, no novelty).
- Hard-code a trivial temporal graph (just record edges with Δt) and a trivial replay (return current moment set).
- Verify: cortex fires a think-action, hippocampus runs experiment frames, action-connection updates land on a moment, that moment's votes show up in cortex's next action selection.

### Phase 2 — Salience, selective minting, and cold-start gating

- Z-scored reward and prediction error over sliding windows.
- Gating (mint iff `|z|` over threshold).
- Sampling rate for non-salient baseline minting.
- Cold-start gating: new moments accumulate K activations before contributing to action votes.
- Strength-based decay and eviction on the moment timeline.
- Death ledger and re-mint suppression.

### Phase 3 — Action-connection temporal bins

- Implement linear short-term bins for patterns (existing cortex behavior, formalized).
- Implement exponential long-term bins for moments.
- Soft bin assignment via Gaussian-over-log-Δt kernel.
- Action intrinsic horizon: track reward-arrival-Δt distribution per action, use it to select which bin's connection strength is read at voting time.
- Verify: a moment with strong long-bin connections votes loudly when long-horizon actions are being considered, and vice versa.

### Phase 4 — Temporal moment graph and wavefront replay

- On each mint, draw weighted edges with Δt to recent moments.
- Co-reactivation edge formation (when two moments fire close in time, create or reinforce edge).
- Edge Δt-histogram over the long-term bin scheme.
- Wavefront replay: starting active set → step forward via edges weighted by `edge_strength × context_overlap × Δt_kernel(temperature)` → new active set → repeat.
- Termination conditions: budget, horizon, convergence, salience drop.
- `mode: Replay` end-to-end.

### Phase 5 — Counterfactual experiments and action reinforcement

- Branch via parent-pattern sibling: walk up to a moment's parent patterns, find sibling moments under shared parents with the alternative action, replay forward from one.
- Stack-based push/pop for branches.
- Action reinforcement: at experiment conclusion, strengthen action connections on moments along the trajectory in the temporal bin matching the summed Δt of the lesson.
- Calibration of writeback strength against ordinary statistical action-vote strength.
- Verify: replaying a bad outcome with alternatives discovers a better policy; subsequent encounters of the same context use the improved policy via the moment's reinforced action connections out-voting the old habit.

### Phase 6 — Involuntary forecast pass

- Wire the per-frame forecast push.
- Wire moment activation through normal cortical activation (no separate "activate by context" operation needed).
- Verify: with zero think-actions, the hippocampus produces background forecasts and action-connection updates every frame.

### Phase 7 — Voluntary think-actions and metacognitive control

- Full think-action parameters (mode, budget, horizon, temperature, control).
- Interrupt / Integrate / Remember semantics.
- Reward signals for think-actions (PE reduction, downstream action improvement, opportunity cost).
- Train cortex to fire think-actions when expected value exceeds external action value.

### Phase 8 — Sleep / idle consolidation

- Detect idle.
- High-temperature integrative replay over recent high-salience moments.
- Reinforce strong moments, prune weak ones.
- Observe moments-becoming-classes in real data — verify that aged moments retain only their core parent links.

### Phase 9 (deferred) — Parallel experiments

Move from single-track-with-stack to a pool of N concurrent experiments sharing a "what's been visited" structure to prevent redundant exploration. Optimization, not correctness — Phase 1 is strictly single-track.

### Phase 10 (deferred) — Higher-order moments

Extend the union-creation rule recursively: detect salience-density triggers over windows of moment-firings, mint higher-order nodes whose parents are sets of co-active moments. Same machinery, different temporal scale. Enables hierarchical episodic structure (instants → episodes → eras) for very-long-horizon planning. Deferred until single-level moments are validated and the limits of sequential traversal become apparent.

---

## Testing strategy

### Unit tests
- Moment minting with multi-parent registration: verify the new moment appears in every active pattern's routing table with weight proportional to that pattern's activation strength.
- Cold-start gating: a freshly-minted moment does not contribute to action votes for K activations.
- Per-link decay: weak parent links drop out faster than strong ones.
- Use-driven reinforcement: a parent link that matches recurring context strengthens; a parent link that doesn't fades.
- Moments-age-into-classes: simulate skewed reactivation over time, verify the moment ends up sparsely connected to its true core parents.
- Temporal moment graph: edge construction at mint and via co-reactivation; Δt-histogram accumulation; weighted sampling.
- Action-connection bins: linear (patterns) and exponential (moments); soft bin assignment kernel.
- Action intrinsic horizon: tracking reward-arrival-Δt and using it as the bin selector.
- Wavefront replay: starting from a multi-moment set, step forward, accumulate simulated time.
- Termination conditions: budget, horizon, convergence, salience drop — each fires correctly in isolation.
- Counterfactual via parent-pattern sibling: find the right sibling under shared parents.
- Action reinforcement writeback: lands in the correct temporal bin on the correct moment.
- Salience: z-score thresholding over sliding windows.
- Death ledger: eviction records, re-mint suppression.

### Integration tests
- Full mint → temporal edge → wavefront replay → action reinforcement flow.
- Counterfactual via parent-pattern sibling produces a better-than-original alternative; the moment's action connections reflect the new policy.
- Subsequent encounter of the same context uses the improved policy via the moment's reinforced votes out-weighing the old habit.
- Involuntary forecast fires every frame with no cortex prompting.
- Voluntary think-action interrupts in-flight forecast and resumes after.
- Non-interrupt think-action arriving with full queue is silently dropped (deep-in-thought state).
- Hippocampus removal: a fully populated cortex with the hippocampus disabled still does cued recall and skill execution but cannot mint new moments, plan, or run counterfactuals (HM profile).
- Moments-age-into-classes over a long simulation: verify aged moments end up structurally indistinguishable from statistically-formed patterns.

### Behavioral tests
- Trading backtest with vs. without hippocampus on bad-trade scenarios; verify hippocampus discovers better policies and they propagate to later similar contexts.
- ADHD-like profile: salience too lax, observe distractibility (too many moments minted, action votes diluted).
- Rigidity profile: salience too tight, observe perseveration (too few moments, can't escape habits).
- Rumination profile: high-magnitude negative-reward moments without escape counterfactuals; verify the salience-drop terminator still eventually fires (and verify that raising the terminator threshold or introducing competing high-salience moments shortens the rumination loop).

### Stress tests
- High-throughput cortex with many think-actions.
- Capacity limits and graceful eviction.
- Cortical deletion cascades.
- Long-running simulation to observe moments-becoming-classes at scale.

---

## Open questions

1. **Wavefront writeback granularity.** When the trajectory is a wavefront with multiple active moments at each step, which member(s) of the wavefront receive the action-reinforcement writeback? Strongest-firing? All members weighted by firing strength? Only those whose stored actions matched the substituted action? Specify during Phase 5 with real experiment data.
2. **Salience-drop computation.** What signal exactly is the running average computed over (mean activation strength of the active set?), and what threshold ratio terminates? Default proposal: drop below 30% of starting average. Tune from observation.
3. **Writeback strength calibration.** How strongly do experiment-driven action-connection updates compete with statistical updates? Too strong and one experiment overrides genuine recent experience; too weak and lessons get washed out. Initial proposal: writeback strength equivalent to N=10 statistical reinforcements. Tune from behavioral tests.
4. **Number of bins (short-term and long-term).** Default 10 each. Domain-dependent — algorithmic trading might want more long-term bins for finer-grain horizons; raw control tasks might want fewer.
5. **Cold-start K.** How many activations before a new moment contributes to votes? Default 3. Should be the same as the cortex's existing pattern cold-start gating, whatever that turns out to be.
6. **Moment sampling rate.** How often do we mint a non-salient moment to maintain baseline graph connectivity? Probably every few thousand cortex frames. Defer until we observe whether the temporal graph becomes too sparse with pure salience-triggered minting.
7. **Δt-kernel shape for replay step sampling.** Gaussian over log-Δt is the default. Alternative kernels could bias toward specific horizons. Implementation detail.
8. **Cross-channel moments.** Can a single moment's parent set include patterns from multiple channels (stock + text + vision)? With unification, the answer is structurally yes — any high-level pattern active at mint time becomes a parent regardless of channel. Worth verifying in implementation that cross-channel parent registration is permitted and that cross-channel moments behave sensibly under replay.
9. **Persistence.** Do moments persist across brain restarts? For long-term memory to be meaningful, yes. Need serialization for the column state plus the death ledger and temporal graph.
10. **Higher-order moments.** Whether and when to add the recursive layer (episodes-of-moments, eras-of-episodes). Deferred to Phase 10. The architectural shape is clear (same union-creation rule, lifted one level); the trigger criteria at higher scales need design when the time comes.
11. **The actual instruction set the cortex uses to invoke the hippocampus.** Will emerge from implementation rather than be designed up front. The conceptual operations are listed above; their precise signatures and the cortex-side action neurons that trigger them are TBD.

---

## Why this matters

- **Long-term memory** as cortical neurons minted by deliberate experimentation — not as a separate store, not as a context window.
- **Episodic-to-semantic consolidation** as a structural consequence of per-link decay on multi-parent nodes, not as a separate process.
- **Two thinking modes** — involuntary background forecasting + voluntary think-actions — both running on the same machinery.
- **System 2 enriches System 1** — moments enter the cortical substrate and become raw material for further pattern formation. Expertise development is structural, not behavioral.
- **One activation rule, one voting rule** — patterns and moments share the same machinery once they exist; the hippocampus's distinct role is purely in *creating* moments and *running experiments* to update their action connections.
- **Strategic vs reflexive action selection emerges from the data**, not from a planning module — actions carry their own intrinsic horizon, moments hold action evidence in matching bins, and the votes aggregate naturally at whatever horizon the current context engages.
- **Forgetting that judges** — uses moments fade or sharpen by their honest match with recurring reality; the death ledger remembers what was lost.
- **Biological grounding** — implements hippocampal indexing, complementary learning systems, predictive processing, scenario construction, and episodic-semantic consolidation in one architecture, with HM and rumination as direct architectural predictions.
- **One substrate, two operators, thalamic bus.** Cortex builds by intersection. Hippocampus binds by union. Activation, voting, and decay are unified across both.

The thesis: the next generation of AI architectures will not come from scaling sequence models. It will come from systems that have the structures biological brains have — separate organs for perception, memory, and thinking, coupled bidirectionally and operating on different clocks over a shared substrate, with one substrate underneath that both organs act on through different creation rules.