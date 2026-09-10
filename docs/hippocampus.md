# The hippocampus

An addendum to [algorithm.md](algorithm.md). The machine there — UCAR — is one organ, the cortex: it compresses
what it observes into a hierarchy of patterns and acts on the best reward estimate it holds. This document adds
the second organ, and it is written in the spec's terms and cites the spec's keys. Where it adds a rule of its
own the rule carries an **H** key. It comes after UCAR, and where it departs from the spec it says so where it
departs.

## Overview

Two organs run continuously in parallel over the same neurons:

- **Cortex (System 1)** — fast, reflexive, short-term. Builds patterns by the collapse over recurring
  activations (R7), and every frame executes the action with the best estimate among the situations on the apex
  (R36).
- **Hippocampus (System 2)** — slow, deliberate, long-term. Mints moments — neurons that bind everything on the
  apex at a salient instant — runs replay experiments over them, and reinforces the action connections of the
  moments along trajectories that yielded long-term-optimum rewards.

The deep insight that organizes this design:

> **Cortex splits reality by intersection. Hippocampus binds reality by union. Same substrate, two creation rules.**

There is one neuron kind (D2): a symbol that holds a table of patterns, a history, and connections (D16). A
moment is a neuron with a kind tag. What distinguishes a moment from a pattern's child is *how it was created*
and consequently *how much it names* — not what kind of neuron it is.

- **Patterns** are formed by *intersection*: a pattern names what recurs across the activations it covers — the
  collapse, a majority per neighbor (R7). It starts narrow and stays narrow. Created gradually, by counting.
- **Moments** are formed by *union*: at a salience trigger, the hippocampus binds every neuron on the apex as a
  neighbor of the moment. A moment names a great deal at birth. Created in one shot.

Storage is always cortical. The hippocampus is the operator that mints moments and runs experiments — it owns no
permanent store of its own.

### Why two organs: the hippocampus trains the cortex

The deeper reason for the split, beyond "two creation rules," is that a slow intersection-learner cannot safely absorb a rare single-shot event. One occurrence is too few samples to abstract from — forced to learn it on the spot, the cortex would either overfit the incidental detail or interfere with existing structure (catastrophic interference). The hippocampus exists to capture that instant losslessly in one shot and re-present it (replay) until the cortex has effectively seen it enough times to abstract it safely. In one line: **the hippocampus trains the cortex.** This is the complementary-learning-systems rationale, re-derived from the architecture's own mechanics rather than imported.

"Train" bundles two distinct jobs, with different reward signals and different failure modes; keep them separate:

- **It trains the cortex's representations.** Minting moments that the cortex abstracts into patterns and classes (the moments-age-into-classes mechanism). Consolidating *what is*. This is CLS proper.
- **It trains the cortex's policy.** Counterfactual and imagined replay discover better action→outcome links and write them onto moments, which the cortex inherits through normal abstraction. Improving *what to do*. This is Dyna-style model-based reinforcement, not CLS.

"Trainer" is only the offline half. The hippocampus is also a live participant: its moments vote in the current frame (the involuntary forecast, the gut-feel) before any consolidation has happened. **Teacher and scout** — it generates curriculum for slow consolidation *and* runs live forecasts that bias the current action. Collapsing it to "the cortex's trainer" loses the real-time prospection that is the other half of its value.

Division of credit (relevant to continual learning): by design, the cortex is natively continual — the collapse does not catastrophically interfere the way gradient descent does, so it holds old classes without replay. The hippocampal "training" is an additive single-shot episodic mechanism layered on top, not the thing carrying class-incremental performance. The cortex solves continual learning structurally; the hippocampus adds brain-like fast episodic learning.

## Core Principles

The hippocampus is not a database. Memories are not retrieved through global similarity search.
Instead, memories are re-instantiated by neurons firing on what they name.

The cortex continuously forms sparse hierarchical patterns.
The hippocampus binds those patterns into moments.
Replay corrects the estimates along the routes it walks; nothing weakens a route for not being walked (R34).
Forgetting is not cleanup; forgetting is abstraction.

Thought emerges from replay traversal through compressed contextual structure.

### Moments age into classes

A moment is born naming a great deal: the apex at the instant it was minted, all of it (R27).
It keeps a history of its last `H` activations like every neuron (D18), and what it names is the collapse over
that history — a neuron stays named while it is present in more than half of the moment's activations (R7).
At birth the history holds one activation, and a majority over one activation is everything, so the moment is
born as broad as the instant was.
Every later activation is another instant; the neurons those instants share keep their majority and the
incidental ones lose it.
The moment ends up naming only its core — a class-shaped neuron, structurally indistinguishable from a pattern's
child. No link decays, no link is boosted, and no threshold is set: the history is the whole mechanism, and it
moves at the rate the moment fires rather than at the rate of the clock.

> **H1 — A moment.** A neuron (D2) of the event kind that holds its own pattern: the set of neurons it names, at
> their offsets, the collapse over its history (R7). No parent's table holds it (R16) — that is the structural
> difference from a pattern's child, which exactly one table holds, prices and offers. A moment is priced by
> nothing, offered by nothing and elected by nothing; it holds a history and connections exactly as any neuron
> does (D16, D25), and the rest of this document is what it does with them.

Two rules say what reaches a moment and what fires it, and they are different.
**Any neighbor reaches.** A single active neuron the moment names makes it addressable by the executor: it can
be recalled from a partial cue, replayed and thought from.
**A majority fires.** The moment fires, and the current apex enters its history, only when more than half of
what it names is present — the cortex's own test for whether a pattern applies (R20 step 7), run by the moment
on itself since no table runs it for it.
Reaching writes an imagined activation carrying the moment's own reconstructed instant; when the moment fires,
it writes the real one. Both narrow the moment; only the second lets the present overwrite it, and only when the present
resembles it.

> **H2 — A moment fires on its own majority.** Where a pattern's child fires only when the election accepts its
> parent's bid (R24), a moment fires when more than half of what it names is active, with no bid and no
> election. Having fired, it covers what it names, as an elected child covers what its pattern names (D10): the
> neurons under it stop speaking and stop writing connections until their windows close. The moment stands on
> the apex in their place, at a reach set by its level like any neuron's (D4).

Moments are not stored as immutable records.
A recalled moment is a reconstruction from the neurons its history still gives a majority to.
As rehearsal and recurrence drop the incidental neighbors, moments gradually lose episodic specificity and
become semantic abstractions (classes) — and the cue that fires them widens with it, from instants very like the
original to anything carrying the gist.

Classes are not a separate neuron kind.
They are rehearsed moments.
The majority over the history is the abstraction mechanism.
This is the architectural analog of episodic-to-semantic consolidation as observed empirically: vivid specific
memories lose incidental detail and survive as gist.

### The three-tier graph

![The Three-Tier Graph: The Brain](../images/graph.png)

The full architecture forms a graph with three tiers:

- **Bottom tier: base event and action neurons.** Input and output. The raw interface with the environment (D1).
- **Middle tier: pattern neurons.** Created by the cortex through intersection — the collapse over recurring
  activations (R7). Many levels, each at a doubled reach (D4). A pattern names neighbors one level down and is
  itself named by patterns one level up. The cortex builds this tier bottom-up.
- **Top tier: moment/class neurons.** Created by the hippocampus through union — one-shot binding of the apex at
  salience triggers. Moments name patterns as neighbors, and their connections hold what action followed them
  (D25).

The cortex and hippocampus perform symmetric but opposite operations on this graph:

- **Cortex (intersection):** selectivity at creation. Observes many instances, finds what recurs, creates a
  narrow neuron representing only the common signal. Patterns start narrow and stay narrow.
- **Hippocampus (union):** selectivity deferred to the history. Observes one salient instant, binds everything
  active, creates a broad neuron. Moments start broad and narrow as later instants enter the history and the
  majority drops the neighbors they do not share. A moment converges toward a class — structurally identical to
  a pattern's child that was narrow from the start.

Both create neurons in the same substrate. Both fire on a majority of what they name. The difference is when
selectivity happens: at birth (intersection) or through life (union, narrowed by the history).

### Multi-level moment hierarchy (union of unions)

The pattern hierarchy has many levels: level-1 patterns name base neurons, level-2 patterns name level-1
children, and so on — each level at a doubled reach (D4). The same recursive structure applies to moments.

- **Level-1 moments** bind the apex at a single salient instant. This is what the rest of this document
  describes as the baseline system.
- **Level-2 moments** bind co-active level-1 moments that fired within a level-2 reach, at a level-2 salience
  trigger. A level-2 moment represents an episode — "this sequence of important instants forms a recognizable
  whole."
- **Level-3 moments** bind co-active level-2 moments over a longer reach still. An era — "this sequence of
  episodes forms a recognizable phase."
- **Level-N moments** follow the same rule recursively. Each level applies union at its own reach.

The operation at every level is the same: union over the currently-active lower-level moments when salience
triggers at that level's reach. A moment's offsets are D6's and its reach is D4's, so a level-N moment's
connections reach `2^N` frames forward, exactly as a level-N pattern's do.

Coverage (H2) applies across moment levels: when a level-2 moment fires, it covers the level-1 moments it
names, and they stop voting. This means the system automatically selects the appropriate planning horizon — a
level-2 moment's action connections reach farther, and coverage ensures they aren't drowned out by level-1
noise. If a level-2 moment stops firing (because aging narrowed what it names beyond the current situation),
coverage does not arrive and level-1 moments speak again. The system gracefully degrades down the hierarchy.

This is the architectural realization of hierarchical temporal abstraction: instants → episodes → eras →
life-chapters, with each level binding the one below by union and projecting actions at its own reach. No
planning module, no horizon selector — coverage across levels *is* the horizon selector.

Implementation note: the multi-level extension is architecturally locked in but the mechanics at level 2+
(salience triggers, co-activation windows, replay across levels) are best discovered empirically from a working
single-level system. See "Open questions" and "Phase 10" for specifics.

### Thinkability = reachability by the executor

Ordinary cortical neurons fire when their context matches and otherwise sit dormant — they are not deliberately summonable. Moments are different: the hippocampus can address them by id (via the thalamic translation layer), reactivate them, traverse their connections, and run experiments over them. **A neuron is "thinkable" iff the hippocampal executor can reach it.** Thinkability is not a property of the neuron's location or type; it is a property of being operated on by the hippocampus.

What makes a moment reachable in a frame is any one of the neurons it names being active — a partial cue is enough, which is pattern completion. Reaching is not firing: a reached moment can be recalled and replayed, but it fires, and takes the present into its history, only on a majority of what it names (H2).

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
- **Episodic-to-semantic consolidation** — empirical finding that specific episodic memories schematize into gist over time. In this architecture, this is a structural consequence of the majority over a broadly-named moment's history as later instants, real and rehearsed, enter it.

---

## Architectural principles

1. **One substrate, two operators.** Cortex and hippocampus operate on the same neurons. The thalamus mediates
   id↔property addressing.
2. **Cortex splits by intersection, hippocampus binds by union.** Cortex divides experience into hierarchical
   patterns by the collapse over recurring activations (selectivity at creation — only what recurs is built).
   Hippocampus binds salient instants into moments in one shot (selectivity deferred to the history — everything
   enters, and later instants vote the incidental neighbors out).
3. **One neuron kind.** Patterns' children and moments are the same kind of neuron (D2, D16). They differ in how
   they were created (intersection vs union) and in how much they name at birth (few neighbors vs the whole apex).
4. **Hippocampus is the executor, not the store.** It mints moments, runs experiments over them, and updates
   action connections on moments. It does not hold them.
5. **Recall is by context, and reaching is not firing.** Any active neuron a moment names makes it reachable by
   the executor. A moment fires when more than half of what it names is present (H2), and that is normal
   cortical activation, not a special operation.
6. **One voting machinery.** Every apex activation votes for actions through its neuron's action connections
   (D25, R36). A connection carries a *strength*, its exposures, and an *estimate*, the mean reward over them
   (R31); the largest estimate wins (R36). Moments and patterns' children vote identically, and there is no
   separate "decided action" override mechanism and no second kind of update.
7. **Selective minting.** A moment is minted on prediction error, on reward, and on the clock (H3). Ordinary
   experience becomes thinkable at the clock's rate and no faster.
8. **Two clocks.** The hippocampus runs at a faster clock than the cortex, executing many experiment frames per
   cortex frame.
8b. **Parallel fan-out replay.** Replay is a pool of N concurrent trajectories from a shared starting state, not
   a single-track walk. After fan-out, the winning trajectory's actions are written back to the moments along
   it. This is the core replay mechanism, not an optimization.
8c. **Salience-modulated lookback.** Before fan-out, the starting state is rewound backward through what the
   active moments name at negative offsets, by a depth derived from triggering salience. Small surprises rewind
   shallowly; large surprises rewind deep. The brain trades short-term thinking for long-term re-evaluation at
   the scope the trigger warrants.
9. **Same death ledger, same aging rule, different rate.** Forgetting is unified through one death ledger
   (R18), and a neuron's evidence expires on one rule: its history holds its last `H` activations (D18). A moment
   fires rarely, so its history turns over slowly and it remembers for as long as its rarity makes it. Nothing
   declares a slower timeline for moments; the rate is the neuron's own.
10. **Bidirectional coupling via thalamus.** Cortex emits think-actions that reach the hippocampus.
    Hippocampus reads currently active cortical state and writes action-connection updates back to specific
    moment neurons. Updates are local to the neuron written, never global.
11. **Multi-level moment hierarchy.** The union-creation rule applies recursively: level-1 moments bind patterns,
    level-2 moments bind level-1 moments, level-N moments bind level-(N-1) moments. Each level at a doubled
    reach, mirroring the pattern hierarchy (D4). The same machinery (a pattern of its own, a history, action
    connections at D6's offsets, coverage) applies at every moment level.
12. **Coverage selects the planning horizon.** When a higher-level moment fires, it covers the lower-level
    moments it names and they stop voting (H2, D10). The highest-level moment that matches the current context
    dominates action selection at its reach. No separate planning module or horizon selector is needed.

---

## Context overlap

Context overlap is not a global semantic similarity operation, and it is not a tolerance.
What a moment names is a majority statement over its history, and an instant matches it when it agrees with the
majority of that statement: more than half of what it names present (H2, R20 step 7).
That is one comparison per moment against the current apex, made where the moment lives, with no global search.
A moment that has narrowed to a few core neighbors is matched by any instant carrying those few; a fresh moment
is matched only by an instant much like its own.
Memory retrieval therefore emerges from many independent local matches, each exactly as wide as that moment's
history has made it.

---

## The cortical substrate

A neuron, whether minted as a pattern's child or as a moment, holds (D16):

- **What it names**: for a pattern's child, the line in its parent's table (D15); for a moment, its own pattern
  (H1). Either is the collapse over a history (R7), and nothing on it is weighted; a neuron is in the set or not.
- **A history**: its last `H` activations, each with the neighborhood it observed (D18).
- **Connections**: per `(action neuron, offset > 0)`, a strength and an estimate — what action followed the
  neuron's activations and what it earned, written apex to apex while the activation is uncovered (D25, R31).
- For a moment, a kind tag and its salience at birth.

When more than half of what a neuron names is present, it fires — for a pattern's child through its parent's
bid and the election (R20, R24), for a moment on its own (H2) — and, if on the apex, contributes its
inferences (R36). The cortex does not distinguish patterns' children from moments at activation or voting time
— both fire on a majority of what they name, both vote into the same resolution.

What distinguishes them structurally is only their origin: a pattern's child was minted by its parent on a
candidate that paid (R14, R15) and names what recurred; a moment was minted by the hippocampus by union over
the apex at a salience trigger and names what was there. Both hold connections at D6's offsets out to their
level's reach (D4). The hippocampus's special access is by id, not by kind.

---

## Coverage

A neuron an accepted bid covers stops speaking and stops writing connections (D10). A moment that fires does
the same to what it names (H2). This is the one inhibition in the design, and it runs in three places.

### Coverage across the pattern hierarchy

When a level-3 pattern's child is bought, it covers the level-2 and level-1 neurons its pattern names (R24).
They don't independently vote — the higher neuron already captures their information more precisely in
context — and they stop writing connections, since the coverer takes that sample over its narrower situation
(D25).

### Coverage across the moment hierarchy

The same rule applies across moment levels. When a level-2 moment fires (an episode recognized), it covers the
level-1 moments it names. When a level-3 moment fires (an era recognized), it covers the level-2 episodes below
it.

The consequence for action selection: each moment level's action connections reach as far as its level does.
Level-2 moments vote at longer offsets than level-1 moments. When a level-2 moment fires and covers what it
names, short-horizon action votes from level-1 go quiet and long-horizon action votes from level-2 dominate.
The system automatically selects the appropriate planning horizon based on which level of moment recognition
fires.

When a higher-level moment ages and its history drops neighbors, it may stop firing in contexts where it used to
fire. When it stops, coverage does not arrive, and the lower-level moments below it speak again with their
shorter-horizon actions. The system gracefully falls back down the hierarchy as abstractions sharpen and narrow.

### Coverage from moments to the patterns they name

A moment is minted by union over the apex; the patterns' children it names are its neighbors. The same coverage
that runs pattern→pattern and moment→moment therefore also runs moment→pattern: when a moment fires, the
neurons it names stop voting. This is the structural form of System 2 overriding System 1. The moment's
long-horizon action votes replace its neighbors' short-horizon reflexes — a deliberately-reasoned conclusion
suppresses the scattered habit, earned structurally rather than bolted on as a weight.

This preserves the one-voting-rule principle exactly. Coverage gates *who* votes, not *how much* a vote counts.
When a moment and a pattern's child both stand on the apex, they vote identically; there is no
moment-supremacy multiplier. The override is entirely a consequence of which neurons are on the apex, not of any
asymmetry in vote strength — which is why no separate "decided action" mechanism is needed.

Coverage is categorical, not graded. A moment fires only when more than half of what it names is active, so
selectivity already lives at the activation rule: by the time a moment fires, the question of whether a neighbor
really belongs to this context has been answered by the history. Grading coverage by anything would re-apply
that same selectivity a second time. A neighbor that is incidental to the moment is, by definition, usually not
active when the moment fires — so there is nothing to cover; and in the rare frame where it is co-active, it is
part of the current scene and covering it is correct. Fire-or-not carries the nuance; coverage is binary
downstream of it.

### Coverage as the horizon selector

No planning module is needed to choose between tactical and strategic action. The hierarchy does it:

- Novel situation, only level-1 moments match → short-horizon tactical behavior.
- Recognized episode, level-2 fires → long-horizon strategic action dominates.
- Recognized era, level-3 fires → very-long-horizon actions dominate.

Coverage across levels *is* the horizon selector.

Inhibition may also emerge behaviorally through competing actions, such as suppressive or muting actions.

---

## Action connections

A moment holds action connections exactly as any event neuron does, and nothing of the addendum's own: per
`(action neuron, offset > 0)`, a strength and an estimate (D25); at D6's offsets — `1, 2, 4, 8, …` — out to its
level's reach (D4); written apex to apex while the moment's activation is uncovered (R31); read at every offset
beyond the age when the moment stands on the apex, each connection placed at its completion `offset − age`
frames ahead and expanded to base actions (R28, R36).

So there is no bin scheme. The exponential offsets are D6's, which is what lets a high moment name what
followed it far out and coarsely, and a low one near and finely; the reach is D4's, which is why a level-N
moment sees `2^N` frames ahead. Nothing assigns an exposure to more than one offset, and nothing chooses which
offset to read for an action: an exposure is written at the offset its age rounds to and read back at the age
from which that offset places a completion one frame ahead (R31).

---

## The hippocampal executor

The hippocampus owns no permanent store. It owns:

- The salience module (H3: windowed z-scoring of reward and prediction error, and the clock).
- A **parallel experiment pool** — N concurrent trajectories explored simultaneously from a shared starting state, with a shared "what's been visited" structure to prevent redundant exploration. Parallel fan-out is the core replay mechanism, not an optimization.
- A transient working set of currently-relevant moment ids while experiments run.
- The death-ledger writer (the ledger itself is shared across cortex and hippocampus).

Its operations:

- **Mint moment** — given the apex and a salience trigger, create a moment neuron whose first activation is the instant: every neuron on the apex, and the actions that ran within reach at their offsets, as what it names (H1). Its connections form on their own as the activation stays open (D25).
- **Rewind** — walk backward through what the active moments name at negative offsets, from the trigger point, to select a past set as the starting state. Rewind depth is salience-modulated: small triggers rewind shallowly (local "what could I have done differently here?"); large triggers rewind deep (broader "what could I have done differently further upstream?"). This is how the brain trades short-term thinking for long-term re-evaluation.
- **Fan out parallel trajectories** — from the chosen starting state, dispatch N concurrent experiments. Each substitutes a different alternative action (or runs as-original) and walks forward independently. Trajectories share a visited-moments structure to prevent redundant exploration.
- **Replay (per trajectory)** — imagined frames, each an event step that infers the next action from the active set's action connections (D25, R36) and an action step that reads what follows that action from its event connections (H4), accumulating simulated time and the estimates earned along the trajectory.
- **Branch (for counterfactual)** — at any event step, substitute an alternative action for the one the vote would have chosen, and take the action step from that action's event connections; continue from there.
- **Select winner and write it back as activations** — when the fan-out converges or budget expires, compare trajectories by accumulated reward over the simulated horizon. Walk the winning trajectory backward and re-present each step to the moments in it as an imagined activation carrying the action taken and the step's return-to-go, at the offset the summed distance names; their connections are then re-read from their histories. Losing trajectories leave no trace beyond their contribution to the visited structure.
- **Evict** — moments that name nothing go to the death ledger (H5).

The cortex does not know about the moment/pattern distinction. It just sees neurons firing and contributing votes. The hippocampus is the only thing that knows which neurons are moments and what experiments are running.

Replay is intentionally approximate and resource bounded.

The goal is not exhaustive planning but salience-guided exploration.

Replay may:
- loop,
- terminate early,
- become distracted,
- or converge on emotionally reinforced regions.

This behavior is considered a feature rather than a defect and mirrors many properties of biological cognition.

---

## The salience module

Computes a "mint this moment" signal per cortex frame, from three sources:

> **H3 — What mints a moment.** Prediction error, reward, and the clock.
> ```
> prediction error   what the action that ran said would follow it — its event connections at the offset
>                    ahead (H4) — against what arrived; the misses are the error
> reward             what arrived with the frame (R33)
> the clock          every `T` frames, whatever the other two say
> ```

- `z_reward = (reward − μ_reward) / σ_reward` over a sliding window
- `z_pe = (prediction_error − μ_pe) / σ_pe` over a sliding window
- Mint iff `|z_reward| > θ_r` or `|z_pe| > θ_pe`, or the clock says so

The machine has no expectation output of its own, and no prediction error (D12). The hippocampus has both,
because it holds the one record the machine does not read: what events followed an action (H4). The error is
that record against the frame.

Novelty is not an input. Every observation contains some novelty; that does not make it worth remembering. What
the literature calls "novelty effects" is in practice prediction error, already covered.

When the trigger fires, the hippocampus calls Mint over the apex.

The clock is not a baseline for the other two triggers; it is how the robot remembers what it is doing. A moment
minted on schedule holds an ordinary instant, and the record between two salient moments is made of them.
Timed moments age and narrow like any other; one that is never reactivated stops naming anything and dies
(below).

---

## Moment creation

When a salience trigger fires, the hippocampus mints a new moment by union:

1. Read the apex (R27): every active neuron no accepted bid covers, at every level at once, and the actions that
   ran within reach at their offsets.
2. Mint a moment neuron (R16's create, with no parent), of the event kind, at a level of the hippocampus's
   choosing.
3. Write the instant into the moment's history as its first activation: the apex is its neighborhood. What the
   moment names is the collapse over that one activation, which is the whole set (H1). The moment names neurons
   at many levels and names actions beside events, both of which no pattern does (D5); a moment is not a pattern.
4. Nothing is wired forward. The moment's activation is open from this frame, and its connections form as the
   actions that follow run, at their offsets (D9, D25) — born holding nothing, like any neuron (R16).
5. Nothing links it to earlier or later moments by hand. What the moment names at negative offsets already holds
   the moments and patterns active before it within its reach, and what follows it is written as it happens.
   That, with H4, is the transition model replay walks (see "The transition model").
6. Record its salience at birth from the triggering z-score, or that it was minted on the clock.

The moment now exists as a neuron in the cortical graph. Future cortical pattern-formation can name this moment
as a neighbor, exactly like a base or pattern neuron. This is the structural mechanism for "lessons from
specific experiences becoming reflexes" — System 2 enriches System 1 by adding moment neurons that the cortex
then abstracts over.

### A fresh moment infers nothing

A freshly minted moment holds no connections, so it infers nothing until something has run under it (R35): it
fires, and covers, from its first majority, and what it then says about the next action is whatever was
executed under it. That is the spec's only gate on participation — the absence of connections — and the
addendum adds none. An earlier draft counted a moment's activations and let it neither vote nor cover until it
had `K` of them; the count is gone. What that count was protecting, that a purely imagined conclusion should
not drive behavior on its own, is addressed where imagined scenarios are constructed (below).

### Moments age into classes (mechanism)

After minting, the moment's history fills from two sources, and what it names is re-collapsed after each (R7,
R8):

- **Real activations.** An instant in which more than half of what the moment names is present fires the
  moment (H2), and the apex of that instant enters the history. Neighbors shared with the original survive the
  majority; neighbors peculiar to the original do not.
- **Imagined activations.** Every experiment that reaches the moment — any active neighbor suffices to reach it
  — writes the reconstructed instant it visited into the history ("Action reinforcement" below).
  Reconstructions vary across trajectories, cues and sibling substitutions, so across many replays the
  neighbors consistent among them keep the majority and the incidental ones lose it. This is what narrows a
  moment too broad to be fired by anything but a near copy of its own instant; idle consolidation, which
  replays high-salience moments, is therefore the abstraction mechanism and not a reinforcement of it.

The moment ends up naming only its core. Structurally, it has become a class — a neuron representing "this kind
of situation" rather than "this specific instant." No separate machinery, no clustering pass, no centroid
computation, no decay rate and no boost. The majority over the history does the abstraction work over the
moment's lifetime.

This means there is no class neuron type. There are only moments at various stages of rehearsal. Young moments
are episodic; rehearsed moments are semantic. The history on a broadly-named neuron is the consolidation
mechanism.

Why the activation rule is a majority and not any neighbor: a moment fired by any one neuron it names would fire on
every instant of its most common neighbor, its history would fill with that neighbor's ordinary company, and the
majority over it would converge on that neighbor's usual context — a duplicate of a pattern the cortex already
has, not the class of the salient situation. Any neighbor reaches; a majority fires; the history is the class.

A moment that captured noise and never recurs keeps its one activation and its one instant. It is never fired by
the present, and it is never narrowed by it; only rehearsal or eviction touches it. Nothing keeps it alive
artificially and nothing compensates for its breadth.

---

## The transition model

The machine keeps one thing about what follows an activation: the action that ran, on the event neuron, with
what it earned (D25). Replay needs one more — what an action leads to — and the addendum adds it on the action
neuron.

> **H4 — Action neurons hold event connections.** An uncovered action activation connects to the apex events
> that follow it, apex to apex, at the offset its age names, exactly as an uncovered event activation connects
> to the apex action (D25, R31): per `(event neuron, offset > 0)`, a strength — the times an activation of the
> action saw that event follow at that offset — and no estimate. It is written in the `process actions` call
> beside the action connections (§18), one exposure per frame per event dimension, at the offset the age rounds
> to; a coarse offset pools the events of every frame in its group (D6). It is never in the file and enters no
> test (D12), nothing collapses it (R7), nothing weakens it (R31), and nothing in the machine reads it: it is not
> an output. A pattern's child of the action kind holds it from the frame after its mint on (R17); a base action
> holds the marginal over every situation it ran in. The hippocampus reads it, and nothing else does.

An experiment alternates the two kinds, and that alternation is the transition model:

```
event step    from the active set, read its neurons' action connections at the offset ahead:
              what to do, and what it is estimated to earn                                   D25, R36
action step   from the action chosen, read its event connections at the offset ahead:
              what follows — the apex events, which are the next active set                  H4
repeat
```

Nothing is built for replay that the neurons do not already hold. The record is collected by the one call for
every neuron, event and action alike (§18), and it holds both directions replay needs: what a situation is
followed by, as an action with an estimate, and what an action is followed by, as events. Replay reads it and
writes nothing into it except through the writeback ("Action reinforcement"). An earlier draft kept a temporal
moment graph beside the moments — edges drawn at mint to the most recent moments, weighted by one over Δt,
reinforced by co-reactivation, decayed by disuse, dropped below a threshold, carrying a Δt histogram, and
sampled through a temperature-controlled kernel. None of it is needed, because the two connection kinds are the
thing the graph was approximating:

- The **offset** is the Δt. Offsets are D6's, so a moment at a high level names what followed it coarsely and
  far, and the histogram over long-term bins the graph carried is the per-offset structure, already there.
- The **strength** is the edge weight, and the **estimate** is what the edge is worth. Nothing is initialized,
  reinforced, decayed or thresholded (R31, R34).
- **The backward direction** is what the moment names. What was active before it, within its reach, is in its
  own pattern at negative offsets (H1), so rewind reads the same neuron in the other direction with nothing
  added at mint time.
- **Co-reactivation needs no rule.** A moment that fires shortly after another names it at a negative offset by
  the ordinary accrual, and the action that ran between them names the second at a positive one (H4).

---

## Recall — the involuntary forecast pass

After cortex finishes its frame, any moment more than half of whose neighbors are active will already have fired (H2) — this is normal cortical activation, not a special operation. Moments with fewer of their neighbors active are not firing but are reachable, and the executor may pull them into an experiment from the cue. The hippocampus reads the currently-firing moment set and, for each moment above an experiment-eligibility threshold:

- Notes that the moment is firing (it is already voting via its action connections; nothing extra needed).
- May initiate a small-budget parallel replay fan-out. The rewind depth before fan-out is derived from the triggering salience: low-magnitude triggers start from the currently-active moment set (no rewind); high-magnitude triggers rewind further back through what the moments name at negative offsets before branching forward. This is the involuntary "what should I have done differently?" reflex — its scope automatically matches how bad (or good) the trigger was.

This pass fires every cortex frame, even with zero think-actions. It is the always-on background channel — *"that reminds me…"*, mind-wandering, gut-feel forecasting.

The forecast itself is predictive, not yet counterfactual; its purpose is to surface good or bad scenarios worth evaluating. The moment it flags a salient outcome it spawns the rewind-and-fan-out that is thinking proper. In that sense forecasting exists in service of counterfactual evaluation: pure forward prediction is recall, and thinking proper begins when a predicted outcome is worth asking "what should I do instead?" about.

When a voluntary think-action arrives mid-forecast, the forecast is pushed onto the experiment stack and the think-action runs. The forecast resumes after.

---

## The think-action interface (voluntary thinking)

The cortex fires think-actions as ordinary actions: "remember", "think of" and the rest are members of the declared action alphabet in a channel of their own, so selection reaches them like any action and the exploration walk can try them ([algorithm.md](algorithm.md), R35, R37). Each of them means "run an experiment." The cortex does not address moments by id — its world is neurons and patterns. A think-action takes the currently active neighborhood as its cue, or the content it names; the hippocampus resolves the cue to candidate moments by driving the cue and reading off which moments fire, and the experiment begins with an event step from that set.

```
think_action {
  cue: [cortical_neuron_id, ...],   // cortical content to think about — patterns,
                                     //   sensory ids, action ids, or any mix.
  mode: enum {
    Replay,                          // run forward as-original
    Counterfactual,                  // inject alternative actions and explore
    Retrieve,                        // surface related moments via shared neighbors
    Compare,                         // run two cued states in parallel, measure delta
    Compress,                        // find common structure across cued moments
  },
  budget: integer,                  // hippocampus frames to spend
  lookback: integer,                // steps to rewind backward through what the
                                     //   active moments name at negative offsets before fanning
                                     //   forward; 0 = branch from here; larger =
                                     //   re-evaluate decisions further upstream.
  horizon: integer,                 // max simulated time in imagined frames
  fanout: integer,                  // number of parallel trajectories to dispatch
                                     //   from the rewound starting state (default
                                     //   pool size; may be capped by capacity)
  control: enum {
    Interrupt,                       // push current experiment, start new
    Integrate,                       // add cue to current experiment
    Remember,                        // mint without starting an experiment
  }
}
```

**Cue → moment activation** (inside the hippocampus):

1. Drive the cue neurons.
2. Let cortex pattern-complete; collect the moments that fire (H2).
3. The active set becomes the experiment's starting state.

This is the same machinery as the involuntary forecast pass. The only difference is the source of the cue — cortex's currently-active high-level patterns for involuntary, the explicit `cue` parameter for voluntary.

Each parameter is independently learnable through the metacognitive reward signal (Phase 7).

Think-actions are replay-control directives.

They bias:
- replay initialization,
- traversal direction,
- salience weighting,
- reinforcement,
- or memory persistence.

Language may trigger think-actions directly, allowing external instructions
such as "remember this" or "think about this" to shape replay behavior.

---

## Experiment execution

### Starting state

An experiment starts with a *set* of currently-active moments — whatever was firing in the surfacing context (involuntary) or whatever the cue resolved to (voluntary). The replay does not assume a single starting moment.

### Rewind — salience-modulated lookback

Before fan-out, the starting state may be rewound `lookback` steps backward. A step back reads what the active moments name at negative offsets — the moments and patterns present before them, within their reach (H1) — and takes those as the earlier active set. Lookback zero starts from the surfacing context; positive lookback lands on an earlier set along the recent trajectory, at D6's offsets at the moments' own level.

For involuntary experiments, lookback is derived from the triggering salience magnitude — bigger surprise rewinds further back. The mapping (proposed: `lookback = floor(k · |z|)` capped by the available trajectory length) is parametric and tuned empirically. The intuition: short-term thinking does not always guarantee long-term optimum results; a bad outcome may have its real branch point several moments upstream. Salience-modulated lookback lets the brain re-evaluate at the scope that the surprise warrants — local for small mistakes, deep for large ones.

For voluntary think-actions, lookback is supplied explicitly.

### Parallel fan-out

From the (possibly rewound) starting state, the hippocampus dispatches `fanout` concurrent trajectories. Each trajectory independently:

- Substitutes a candidate alternative action at its branch point (one that holds event connections, so that something says what follows it — see "Counterfactual" below), or runs as-original for a baseline.
- Walks forward step by step, accumulating simulated time and reward, until a termination condition fires.

Constraint on alternatives: an action can be substituted only where its event connections say what follows it (H4). Those connections are over every situation the action ran in — "what follows this action, as far as is known" — and they are present once the action has ever run. An action that has never run holds no event connections and cannot be stepped through; it can only be tried in the world, which is the walk's job (R37).

Trajectories share a visited-moments structure so that exploration doesn't redundantly retrace the same forward walks. Each trajectory's per-step state (active moment set, accumulated reward, accumulated Δt) is independent.

### One imagined frame — an event step and an action step

An experiment advances by imagined frames, and an imagined frame is the machine's own two-frame cycle run over connections instead of input: infer, then execute, then read the consequence ([algorithm.md](algorithm.md), R29). Every read is a connection, and every connection was written the same way — while a neuron's activation is open and uncovered it connects to what follows it, an event neuron to the action that ran with its reward (D25, R31) and an action neuron to the events that followed (H4), in the one `process actions` call (§18). Nothing is built for replay that the neuron did not already hold.

1. **Active set.** Read the currently active set — moments and patterns — from the top of the experiment stack.
2. **Event step: what should I do next?** Read the action connections of every member of the active set at the offset ahead, place and expand them to base actions carrying their estimates, and resolve one action per action dimension by largest estimate — the same vote selection runs on a real frame (R36). A trajectory running as-original takes that winner; a counterfactual trajectory takes its substituted alternative instead. **Score** the step: the winner's estimate is what this step is expected to earn, and it is added to the trajectory's running total. This is the event→action direction of the record, read as a question.
3. **Action step: what will happen as a result?** The chosen action now runs in the imagined frame, exactly as a real action fires beside the events it runs alongside (D8, R30). Read what follows it: its event connections at the offset ahead (H4), each at its strength. The events they name, placed and expanded to the level the experiment steps at (R28), are the next active set. This is the action→event direction of the record, read as a question.
4. **Accumulate simulated time.** One imagined frame per step; a step at a moment level advances the horizon by that level's offset (D6), so a long horizon is many steps at coarse offsets.
5. **Continue, branch, or terminate** (see termination conditions below).

Every step lands imagined activations on the members of the set it read from — what the step reconstructs at negative offsets, the action taken, and the estimate earned beside it — so that when the trajectory is chosen for writeback the activations are already in hand ("Action reinforcement" below). Losing trajectories' activations are discarded with them.

### Termination conditions

The experiment stops when *any* of:

- **Budget exhausted** — the requested number of replay frames has been used.
- **Horizon exceeded** — the summed simulated Δt has passed the requested horizon. Useful for "simulate one hour ahead, stop" semantics.
- **Convergence** — the estimate accumulated across additional steps stops changing meaningfully, or the walk has entered a loop (revisiting moments already in the trace).
- **Salience drop** — the running average of the visited moments' salience at birth drops below X% of the starting average. The replay has drifted into uninteresting territory.

For involuntary experiments without an explicit budget or horizon, salience drop is the dominant terminator. The replay walks forward as long as each step keeps surfacing reasonably salient moments; when the trail goes cold, it stops. This matches the phenomenology of mind-wandering: associative chains fade out when the activations weaken.

For voluntary think-actions, all four conditions apply, with budget and horizon caps coming from the request.

### Counterfactual

When the experiment asks "what if a different action had been taken at this step?", it substitutes the
alternative at the event step and reads its consequence at the action step from the alternative's own event
connections (H4): what has followed this action across every situation it ran in. That is "what would happen if
I did this, as far as I know" — a marginal over situations, honest about what is known. An earlier draft read the
consequence from a child whose parent pattern named the action beside the active events, the joint conditional;
no pattern names both kinds (D5), so no such child exists, and the marginal is what there is.

### Imagined scenarios — construction beyond recombination

The counterfactual mechanism above recombines only *known* action→outcome records: it requires a record that already holds the alternative beside this situation, or at least the action's own record. It cannot evaluate a situation that never occurred, or an action that has never run. That makes the counterfactual machinery a recombiner of past experience — narrower than the scenario-construction capacity the architecture claims biologically, where hippocampal damage abolishes imagining novel scenes, not just recalling old ones. This subsection closes that gap.

The general operation is union-mint with a supplied active set. The hippocampus can construct a **hypothetical moment** by union over patterns and lower moments that need never have co-fired from sensation. This is the same union-creation used at salience triggers, except the active set comes from a cue (a think-action) rather than from the current frame. "If this co-occurs with that, and then this other thing happens…" is literally a constructed union — a situation assembled from familiar parts that were never assembled together by the world.

Forward simulation runs as ordinary replay. The constructed moment has no connections of its own yet, so its first imagined frame reads the action connections of its constituents — the patterns and moments the cue bound — and steps from their union exactly as a normal forward replay. Forward simulation over a never-observed starting state is possible because the connections are over patterns and moments, not over specific episodes: a novel combination of familiar parts inherits the forward dynamics of those parts. The same termination conditions, reward accumulation, and winner selection apply.

Writeback targets the hypothetical moment. When the forward simulation concludes that action X is good in the constructed situation Y, the policy is written onto Y itself — "I will do X if I find myself in Y," for a Y that may never have happened before. When reality later produces situation Y, the moment fires on a majority of what it names (H2) — it is just a neuron, matched by what it names like any other — and it contributes its stored action through the normal voting machinery.

The firing rule is the safeguard here. A constructed moment votes only when it fires, it fires only on a majority of what it names (H2), and what it names is a situation the world has not presented. So a purely imagined conclusion cannot drive behavior on its own — the world has to actually present the situation before its policy projects, and every real occurrence enters its history and narrows it toward what actually recurs. **Imagination proposes; recurrence licenses.** This bounds the replay pathologies directly: the system can rehearse arbitrary hypotheticals cheaply, but only hypotheticals that reality subsequently presents acquire a vote, and only ones it keeps presenting keep it. An earlier draft counted activations before a moment could vote; the count is gone (R4), and the rule that replaces it is the one every neuron already obeys.

This is strictly a generalization of the counterfactual mechanism. Recombination is the special case where the constructed situation already exists as a moment; imagined construction is the same operation when it does not.

### Action reinforcement (writeback)

When the parallel fan-out completes (all trajectories terminated, or global budget exhausted), the hippocampus selects the winning trajectory — the one with the highest accumulated reward over its simulated horizon. The winner's taken actions are what get written back. Losing trajectories contribute nothing beyond their entries in the shared visited structure.

The as-original baseline is the reference for whether thinking changed the *recommended* action, not a gate on whether writeback happens. Even when the as-original trajectory wins — thinking found nothing better — its returns still refresh the standard action's estimate, possibly correcting it downward. The goal is an accurate estimate, not only an improvement.

Writeback walks the winning trajectory **backward** from its end. Walking back is what gives each step its **return-to-go** — that step's own reward plus everything downstream — rather than the trajectory's flat total. This matters: a moment ten steps before the payoff and one right before it must record *different* values, or the offset structure is corrupted. Return-to-go is computed naturally by the backward walk (each step = its immediate reward + the running downstream sum), and it lands at the offset the step's distance from the moment rounds to (D6) — the offset a real reward for that action would have reached.

**Replay writes activations, not estimates.** For every moment in each step, the step is re-presented to that moment as an activation: an entry in its history carrying what the step reconstructs at negative offsets, the action the trajectory took at the offset it ran, and this step's return-to-go beside that action as the reward. The activation is tagged as imagined, and otherwise it is an activation like any other — it enters the history, evicts the oldest, and the moment's connections are re-read from the history ([algorithm.md](algorithm.md), R8, R31). No scalar on any connection is written directly. What the vote then sees follows from the history:

- The connection's **strength** is the number of activations in the history, real or imagined, in which that action followed at that offset. It grows with every exposure and falls only when an exposure is evicted; nothing lowers it for an action not being chosen.
- The connection's **estimate** is the plain mean of the rewards those activations carry. Every exposure counts equally, real and imagined alike, so the estimate converges on the action's expected return and moves in **either** direction. This is what lets the hippocampus learn *not* to do something: a tempting action whose rollouts end badly gathers imagined activations with bad returns, and its estimate falls until the moment stops voting for it. The hippocampus is a value-estimate *corrector*, not a positive-reward seeker — finding a bad outcome and correcting an estimate down is the same machinery as finding a good one, run symmetrically.

All moments in the step receive the activation, every step, on the backward walk — there is no member-selection rule and no firing-strength weighting. The "which member of the step receives writeback" question dissolves: selectivity already lived at the activation threshold (a moment is in this step only because it fired), so nothing remains to weight downstream.

This is the complementary-learning-systems rationale made literal: the hippocampus re-presents the instant until the cortex has effectively seen it enough times. It replaces both earlier rules — the max gate, which could only raise an estimate and so could never represent "I used to think this was good; it isn't", and the running-mean writeback, which smoothed a scalar the history did not hold and so could not expire. A single unlucky trajectory still cannot condemn a good action, since one imagined activation among `H` moves a mean by one part in `H`; a habit worn in by many real activations takes many consistent bad rollouts to overturn, and it does get overturned within `H` of them, which a lifetime mean would not promise.

**What imagined activations cost.** They occupy the history. A moment that is replayed often holds a history that is mostly imagined, and its real exposures are evicted by its rehearsals. That is the price of re-presentation and it is bounded by `H`; whether replay should be rationed per moment so real evidence keeps a share of the history is open (see "Open questions").

**Estimate vs consensus.** Keep the two readings distinct. The *estimate* above is stored nowhere: it is read off the moment's history at voting time. The *consensus* is computed fresh each frame from the apex voters' connections, placed and expanded to base actions, one winner per action dimension by largest estimate ([algorithm.md](algorithm.md), R36). Replay changes histories; the consensus only reads them.

Generalization is left to the substrate. Don't write the lesson onto multiple moments simultaneously — write it onto the moments the trajectory actually visited. Over time, when those moments reactivate in similar contexts and vote their corrected actions successfully, the cortex builds patterns over the situations where the vote helped, and those patterns inherit the lesson through normal cortical abstraction.

The payoff of storing the correction as state on the moment: when the cortex later auto-activates that moment by recognition — no deliberation — it already votes with the corrected value. Past thinking is available reflexively. Intuition is compiled deliberation; the gut feeling that something is a bad idea is a moment voting with an estimate that a prior experiment's imagined activations pulled down.

### Stack semantics

Branching is implemented as an experiment stack: branch = push, terminate = pop. A higher-priority think-action arriving mid-experiment pushes the current one and runs.

The stack also supports "what was I thinking about?" queries: each entry is timestamped; entries decay over time. Querying returns the topmost item still above decay threshold. If everything has decayed, the thought is lost — the right phenomenology.

---

## Forgetting

The same death ledger applies to every neuron kind (R18), and the same evidence rule: a neuron's history holds
its last `H` activations and nothing older (D18). Moments age slower than patterns' children only because they
fire more rarely, so their histories turn over more slowly; no separate timeline is declared.

**1. No per-link decay.** What a moment names is the majority over its history (H1). A neighbor leaves when the
moment's later activations, real or imagined, stop holding it, and at no other time. There is no link strength,
no decay rate, no boost and no eviction threshold; the history's `H` is the only horizon, as everywhere else
(R4).

**2. A moment dies when it names nothing.** Moments are not in the file, so the one test cannot price them
(R12), and no clock touches them: nothing anywhere is measured in frames (D18). A moment's evidence adapts the
way a pattern's does — its history slides, what it names is re-collapsed (R7, R8) — and a moment whose history
gives no neighbor a majority names nothing, can never be fired or reached, and is dead by that fact.

> **H5 — Moment death.** A moment that names nothing goes on the death ledger, and the ledger takes it on its
> next pass, subtree and all, exactly as it takes a retired pattern's child (R18). Nothing else retires a moment.

**3. Connections do not decay, and nothing weakens them.** A connection is a lifetime total: its strength is
the number of exposures it has had and its estimate the mean of the reward shares they received, with no cap,
no rate and no horizon (R31, R34). Nothing weakens a connection for an action not being chosen: the machine
executes one action per dimension per frame, so an unchosen action wasn't tried, and its connection keeps
exactly the value it had. What moves a connection is an exposure entering it, real or imagined, and an estimate
is exactly as current as its exposures make it: a changed worth is absorbed at `1 / strength` per exposure.
Specificity comes from who holds the connection, not from when it was written (R35): a moment's connections
are over the situations the moment fires in, and a changed situation is answered by a new moment, never by
forgetting.

Reinforcement comes from two sources and is one operation: real frames write exposures into the connections
(the cortex, §18), and replay writes imagined ones into the same connections (the hippocampus, "Action
reinforcement" below). A neuron holds a connection for every action that followed it, and for whatever the
exploration walk has wired, at strength 1 and estimate 0 until tried (R37); the largest estimate wins votes
when the neuron stands on the apex.

**4. Cascade on cortical deletion.** When a neuron is deleted, its name is scrubbed from every pattern and
saved activation that holds it (R18), and what named it is re-collapsed. A moment left naming nothing is dead
as in item 2.

**5. Death ledger.** Every eviction appends `{neuron_id, kind, reason, salience_at_birth}` to the shared
append-only ledger (ring-bounded). Uses:
- Debugging — "why don't I remember X?"
- Observability — shape of forgetting over time.
- Re-encoding suppression — don't immediately re-mint a just-evicted moment unless its salience is now
  substantially higher.

**6. Sleep / idle consolidation.** When the hippocampus has free cycles, it replays high-salience moments. This
is the biological-sleep-replay analog, and it is the abstraction mechanism (above): rehearsal narrows a moment
toward its core. Nothing is pruned by strength; a moment that stops naming anything dies (H5). The
class-rebalancing pass from the previous design is gone — there are no classes as separate entities to
rebalance.

---

## Forgetting as Compression

The system is designed around the assumption that intelligence requires continuous forgetting.

Without forgetting:
- contextual noise accumulates,
- tables explode,
- memories become overly specific,
- and generalization fails.

Forgetting is the collapse over a sliding history (R7, D18): what stops recurring stops being named, and what
keeps recurring is all that is kept.

Over time, this transforms detailed episodic moments into compressed semantic classes.

---

## Attention

The cortex continuously processes sensory and internal activation.

Attention is defined as whatever replay process currently occupies the hippocampus.

In this model:
- attention,
- working memory,
- deliberate thought,
- imagination,
- and planning

are all forms of constrained replay occupancy.

---

## The brain coordinator

Orchestrates the parallel clocks. Manages semaphores. Routes inputs and outputs.

**Per cortex frame:**
1. Read the frame; the cortex runs its levels and picks actions (§2). Moments whose neighbors carry a
   majority fire (H2) and contribute their votes to action selection through the same machinery as patterns'
   children.
2. Wait on `SignalCortexDone`.
3. If salience triggered (H3), push a mint request to the hippocampus.
4. If cortex emitted think-actions, push them as voluntary experiments.
5. Push the currently-active moment set as an involuntary forecast request.
6. Continue to the next frame.

**Per hippocampus frame (faster, in parallel):**
1. Service any pending mint request (cheap, runs immediately).
2. If a parallel experiment pool is active, advance every trajectory in the pool by one frame. When all
   trajectories have terminated, select the winner and run writeback.
3. Otherwise pop the next request: voluntary think-actions take priority over involuntary forecasts. Rewind by
   `lookback`, then dispatch `fanout` trajectories from the rewound starting state. If the queue is full and a
   non-interrupt request arrives, drop it silently — this is the "deep in thought already, can't be bothered"
   state.
4. If idle, run consolidation: replay of high-salience moments, eviction of moments that name nothing (H5),
   death-ledger maintenance.

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
    frame = read_frame()
    cortex.process_frame(frame)              // every level, the election, process actions (§2);
                                              //   moments fire on their own majority (H2);
                                              //   every apex activation votes (R36)
    actions = cortex.resolve()               // may include think-actions
    cortex.emit(actions)
    SignalCortexDone.post()

// brain thread (orchestrator)
loop:
    SignalCortexDone.wait()
    if salience.should_mint(cortex.last_frame):        // H3: error, reward, or the clock
        MintQueue.push({apex, trigger})
    for ta in actions.think_actions:
        ExperimentQueue.push(ta)
    ForecastQueue.push(cortex.currently_active_moments())

// hippocampus thread (free-running, faster clock)
loop:
    while req = MintQueue.try_pop():
        hippocampus.mint_moment(req)         // creates the moment, naming the apex and the
                                              //   actions within reach as its first activation (H1)

    if hippocampus.experiment_pool.has_active():
        hippocampus.advance_all_trajectories()    // advance every trajectory in the
                                                   //   parallel pool by one frame
        if hippocampus.experiment_pool.all_terminated():
            winner = hippocampus.select_winner()  // highest accumulated reward
            hippocampus.writeback_along(winner.trajectory)  // imagined activations along the winner,
                                                            //   return-to-go beside the action taken;
                                                            //   baseline only flags whether
                                                            //   the recommended action changed
            hippocampus.experiment_pool.clear()
        continue

    if ta = ExperimentQueue.try_pop():
        start = hippocampus.rewind(ta.cue_resolved_moments, ta.lookback)
        hippocampus.experiment_pool.dispatch(start, ta.fanout, ta)
        continue
    if forecast = ForecastQueue.try_pop():
        if forecast.non_empty():
            lookback = salience.derive_lookback(forecast.trigger_z)
            start = hippocampus.rewind(forecast.moments, lookback)
            hippocampus.experiment_pool.dispatch(start, default_fanout,
                                                forecast_as_replay(small_budget))
        continue

    // idle
    hippocampus.replay_high_salience_moments()
    for n in hippocampus.moments_naming_nothing():     // H5
        DeathLedger.append({n.id, n.kind, reason, n.salience_at_birth})
    hippocampus.cascade_cortical_deletions()
```

---

## Implementation phases

Built on the substrate [algorithm-implementation.md](algorithm-implementation.md) describes, after its Stage 2
— actions and rewards — has landed, since every phase below reads action connections.

### Phase 1 — Skeleton and parallel clocks

- Create `hippocampus` module.
- Add the moment kind tag to the neuron (D16) so moments live alongside patterns' children in the same columns.
  Reuse the existing thalamic id-translation layer.
- Implement the brain coordinator with two-thread async (cortex + hippocampus, semaphore-coordinated).
- Implement `mint_moment` driven by H3: z-scored reward, the clock, and prediction error once H4 lands (Phase 3).
- Let moments hold action connections like any neuron (D25), and hard-code a trivial replay (return the
  current moment set).
- Verify: cortex fires a think-action, hippocampus runs experiment frames, action-connection updates land on a
  moment, that moment's votes show up in cortex's next action selection.

### Phase 2 — Salience and selective minting

- Z-scored reward and prediction error over sliding windows, and the clock (H3).
- Gating (mint iff `|z|` over threshold, or the clock says so).
- Moment death when nothing it names holds a majority (H5); death ledger and re-mint suppression.

### Phase 3 — Event connections on action neurons

- In the `process actions` call, every uncovered action activation connects to the apex events that follow it,
  at the offset its age names, beside the action connections the event activations write (H4, §18).
- Reading them by id without firing.
- Prediction error from them (H3): what the action that ran said would follow, against what arrived.
- Verify: an action that has run holds, per event and offset, how often that event followed; one that never
  ran holds nothing.

### Phase 4 — Replay

- A moment's activation stays open and connects like any neuron: the apex action at its offsets, the reward
  beside it (D25, R31).
- Step replay: starting active set → event step (infer the action from action connections, R36) → action step
  (read what follows it from the action's event connections, H4) → new active set → repeat.
- Termination conditions: budget, horizon, convergence, salience drop.
- `mode: Replay` end-to-end.

### Phase 5 — Counterfactual experiments, parallel fan-out, salience-modulated lookback, and action reinforcement

- Branch: substitute an alternative action that holds event connections, and step through them (H4).
- **Parallel fan-out pool**: dispatch N concurrent trajectories from the starting state, each substituting a
  different alternative action the active set's connections name (plus one as-original baseline). Shared
  visited structure across trajectories prevents redundant exploration.
- **Rewind by salience-modulated lookback** before fan-out: step backward through what the active moments name
  at negative offsets `lookback` times to land on an earlier moment set. For involuntary forecasts, derive
  `lookback` from the triggering z-score magnitude; for voluntary think-actions, use the supplied parameter.
- **Winner selection**: at pool termination (all trajectories ended, or global budget exhausted), pick the
  trajectory with the highest accumulated reward. Write its taken actions back regardless of the baseline — the
  as-original baseline only flags whether the *recommended* action changed; even a winning as-original
  trajectory refreshes the standard action's estimate.
- Action update along the winning trajectory: walk the trajectory backward; for every moment in each step,
  write an imagined activation into its history carrying the action it took at the offset the step's distance
  names (D6) and the step's return-to-go as the reward. Strength and estimate are re-read from the history;
  nothing is smoothed and there is no max-gate.
- Verify: replaying a bad outcome with parallel alternatives discovers a better policy; subsequent encounters of
  the same context use the improved policy via the moment's reinforced action connections out-voting the old
  habit. Verify that increasing the triggering salience produces a deeper rewind and re-evaluates earlier branch
  points.

### Phase 6 — Involuntary forecast pass

- Wire the per-frame forecast push.
- Wire moment activation through normal cortical activation (no separate "activate by context" operation
  needed).
- Verify: with zero think-actions, the hippocampus produces background forecasts and action-connection updates
  every frame.

### Phase 7 — Voluntary think-actions and metacognitive control

- Full think-action parameters (mode, budget, horizon, lookback, fanout, control).
- Interrupt / Integrate / Remember semantics.
- Reward signals for think-actions (prediction-error reduction, downstream action improvement, opportunity
  cost).
- Train cortex to fire think-actions when expected value exceeds external action value.
- **Imagined-scenario construction**: a cue-driven think-action mints a hypothetical moment by union over a
  supplied set of patterns/moments that need not have co-fired, forward-simulates it through its constituents'
  connections, and writes the trajectory-optimum policy onto the hypothetical moment as imagined activations.
  The hypothetical moment votes only once the world presents the situation and it fires (H2).
- Verify: a hypothetical situation assembled from never-co-fired parts simulates forward sensibly; its
  discovered policy stays silent until the situation actually arises, then projects through normal voting.

### Phase 8 — Sleep / idle consolidation

- Detect idle.
- Coarse-grained integrative replay over recent high-salience moments, stepping at high-level offsets.
- Moments that name nothing die (H5); nothing else is pruned.
- Observe moments-becoming-classes in real data — verify that aged moments name only their core neighbors.

### Phase 9 — Stack-based interrupt semantics for parallel pools

Parallel fan-out itself lands in Phase 5 as the core replay mechanism. This phase adds the orchestration
polish: how a higher-priority think-action arriving mid-pool pushes the current pool, runs its own pool to
completion, then resumes the original. Stack of pools rather than stack of single experiments. Includes the
"what was I thinking about?" query semantics over the stack.

### Phase 10 — Multi-level moment hierarchy (union of unions)

The architectural principle is locked in: the union-creation rule applies recursively at doubling reaches (D4).
Level-1 moments bind patterns. Level-2 moments bind level-1 moments. Level-N moments bind level-(N-1) moments.
Coverage across moment levels selects the planning horizon automatically (H2).

Deferred until single-level moments (Phases 1-8) are validated, because the mechanics at level 2+ are best
discovered empirically from a working level-1 system.

**What is clear:**

- Each moment level uses the same structural machinery: a pattern of its own read off a history (H1), action
  connections at D6's offsets out to the level's reach, any-neighbor reach and majority firing (H2), coverage
  of the level below.
- A level-N moment is minted by union over co-active level-(N-1) moments when a level-N salience trigger fires.
- A level-N moment's action connections are at level-N offsets, so they reach `2^N` frames — much farther than
  level-(N-1)'s. Which offset is read for an action is R36's placement: a connection at offset `b` read at age
  `a` completes `b − a` frames ahead.
- Coverage: when a level-2 moment fires, the level-1 moments it names don't independently vote. The
  highest-level moment that matches the current context dominates action selection at its reach.
- Graceful degradation: when a higher-level moment narrows past a context (its history's majority drops the
  neighbors that context supplied), coverage does not arrive and lower-level moments resume voting. The system
  falls back down the hierarchy.

**What single-level implementation will inform (open questions for Phase 10):**

1. **Level-2+ salience triggers.** At level 1, salience is H3: z-scored reward, prediction error, and the clock,
   per cortex frame. At level 2, the natural candidate is H3's prediction error read at level-2 offsets: what
   the apex action's event connections said would follow, coarsely and far, against what arrived. This requires
   patterns to form over moment-sequences (which should happen naturally — moments are neurons that patterns
   can name). Verify empirically that this falls out from the existing machinery rather than requiring a
   separate salience module per level.

2. **Co-activation window at each level.** At level 1, union binds everything active at one instant. At level
   2, union binds level-1 moments that fired within a level-2 reach. The width follows D4 — `2^level` frames.
   The precise rule (the reach alone? pattern-driven boundary detection?) needs empirical tuning.

3. **Replay across moment levels.** Level-1 replay steps at level-1 offsets; level-2 replay steps at level-2
   offsets, which are coarser, so each step covers more simulated time. Whether this uses the same hippocampal
   clock or needs a clock per level is an implementation question. The simplest approach: one clock, and the
   level of the stepping set decides the grain.

4. **Where level-2+ moments live.** Level-1 moments live in cortical columns alongside patterns' children.
   Level-2 moments should live in the same substrate (same neuron kind, same columns). What they name is
   level-1 moments rather than patterns. Verify that the existing column infrastructure handles this without
   modification.

5. **Number of levels.** Unlike the pattern hierarchy where levels emerge from the cortex's own contraction,
   moment levels are minted by the hippocampus. How many levels should exist? Likely determined by the system's
   experience horizon — a system that has only seen minutes of data won't have level-3 moments. The number of
   levels should emerge from the data, not be configured. A level-N moment is only minted when enough
   level-(N-1) moments exist and co-activate with sufficient salience.

---

## Testing strategy

### Unit tests
- Moment minting: verify the new moment names every neuron on the apex at the instant, and the actions that ran
  within reach, at their offsets (H1).
- The collapse on a moment: a neighbor absent from more than half of the moment's history leaves what it names
  (R7); one present in more than half stays.
- Moments-age-into-classes: simulate skewed reactivation over time, verify the moment ends up naming only its
  true core neighbors.
- Connections on moments: written while the activation is open and uncovered (D25, R31); reading by id without
  firing; the event step and the action step of an imagined frame.
- Event connections on actions (H4): an action that ran holds, per event and offset, how often that event
  followed; nothing is collapsed, nothing weakened; an action that never ran holds nothing.
- Step replay: starting from a multi-moment set, step forward, accumulate simulated time.
- Termination conditions: budget, horizon, convergence, salience drop — each fires correctly in isolation.
- Counterfactual: substitute an action that holds event connections and step through them; an action that never
  ran cannot be substituted.
- Connection updates: an imagined activation enters the history and the connection is re-read — strength is the
  exposures the history holds, estimate their mean. A sequence of bad returns pulls a previously-high estimate
  down until the action stops winning votes; a habit worn in by many real activations moves slowly and is
  overturned within `H` consistent bad rollouts.
- Salience: z-score thresholding over sliding windows, and the clock (H3).
- Death ledger: a moment that names nothing is evicted on the next pass (H5); re-mint suppression.

### Integration tests
- Full mint → accrual → replay → writeback-as-activations flow.
- Counterfactual produces a better-than-original alternative; the moment's connections, re-read from its
  history after writeback, reflect the new policy.
- Subsequent encounter of the same context uses the improved policy via the moment's reinforced votes
  out-weighing the old habit.
- Imagined-scenario construction: a hypothetical moment built by union over never-co-fired parts forward-simulates
  sensibly, its discovered policy is silent until the world presents the situation and the moment fires (H2),
  then projects through normal voting.
- Involuntary forecast fires every frame with no cortex prompting.
- Voluntary think-action interrupts in-flight forecast and resumes after.
- Non-interrupt think-action arriving with full queue is silently dropped (deep-in-thought state).
- Hippocampus removal: a fully populated cortex with the hippocampus disabled still does cued recall and skill
  execution but cannot mint new moments, plan, or run counterfactuals (HM profile).
- Moments-age-into-classes over a long simulation: verify aged moments end up structurally indistinguishable from
  patterns' children.

### Behavioral tests
- Trading backtest with vs. without hippocampus on bad-trade scenarios; verify hippocampus discovers better
  policies and they propagate to later similar contexts.
- ADHD-like profile: salience too lax, observe distractibility (too many moments minted, action votes diluted).
- Rigidity profile: salience too tight, observe perseveration (too few moments, can't escape habits).
- Rumination profile: high-magnitude negative-reward moments without escape counterfactuals; verify the
  salience-drop terminator still eventually fires (and verify that raising the terminator threshold or
  introducing competing high-salience moments shortens the rumination loop).

### Multi-level moment hierarchy tests (Phase 10)
- Level-2 moment minting: verify a level-2 moment is created when level-1 moments co-activate within the level-2
  reach and level-2 salience triggers.
- Level-2 coverage: when a level-2 moment fires, verify the level-1 moments it names stop voting (H2).
- Graceful degradation: when a level-2 moment narrows past a context (its history's majority drops the neighbors
  that context supplied), verify coverage does not arrive and level-1 moments resume voting.
- Horizon selection: verify that level-2 moments vote at farther offsets than level-1 moments, and that coverage
  causes the appropriate horizon to dominate.
- Level emergence: verify that the number of moment levels emerges from data — systems with short experience
  don't mint level-2+, systems with long diverse experience do.
- Cross-level replay: verify that level-2 replay steps at level-2 offsets.

### Stress tests
- High-throughput cortex with many think-actions.
- Capacity limits and graceful eviction.
- Cortical deletion cascades.
- Long-running simulation to observe moments-becoming-classes at scale.

---

## Open questions

1. **Salience-drop computation.** What signal exactly is the running average computed over (mean salience at
   birth of the visited set?), and what threshold ratio terminates? Default proposal: drop below 30% of
   starting average. Tune from observation.
2. **The clock's period.** How often is a moment minted on schedule (H3), so that salient moments have ordinary
   moments between them to step through? Probably every few thousand cortex frames. Whether the period can be
   derived rather than set is open.
3. **How far one step reaches.** A step reads a connection at one offset of the stepping neuron's level, so one
   step at level `k` advances the horizon by that level's offset and no further ([algorithm.md](algorithm.md),
   D4, D6). A long horizon is many steps, and a coarse one is a walk over coarse moments. Whether an experiment
   should be allowed to choose which level it steps at, and so its grain, is open.
4. **Cross-channel moments.** Can a single moment name patterns from multiple channels (stock + text + vision)?
   Structurally yes — anything on the apex at mint time is named regardless of channel (H1). Worth verifying in
   implementation that cross-channel moments behave sensibly under replay.
5. **Persistence.** Do moments persist across brain restarts? For long-term memory to be meaningful, yes. Need
   serialization for the column state plus the death ledger.
6. **Multi-level salience triggers.** How does the system detect salience at level 2+ without a dedicated
   salience module per level? The hypothesis: patterns form over moment-sequences (moments are neurons), H3's
   prediction error read at level-2 offsets serves as level-2 salience, and the same process recurses. Verify
   empirically in Phase 10.
7. **Multi-level co-activation window.** At level 1, union binds everything active at one instant. At level 2+,
   union binds lower-level moments that fired within the level's reach, `2^level` frames (D4). Whether the reach
   alone is the boundary, or a pattern-driven one is needed, needs empirical tuning from a working level-1
   system.
8. **Replay across moment levels.** A level-2 moment's connections are at level-2 offsets, so level-2 replay
   steps at a coarser grain; there is no separate graph to walk. Whether one experiment steps at one level
   throughout, or descends when a coarse step needs a fine consequence, is open. Simplest hypothesis: one
   clock, and the level of the stepping set decides the grain.
9. **Number of moment levels.** Should emerge from data, not configuration. A level-N moment is only minted when
   enough level-(N-1) moments exist and co-activate with sufficient salience. Systems with short experience
   horizons will have fewer levels.
10. **The actual instruction set the cortex uses to invoke the hippocampus.** Will emerge from implementation
    rather than be designed up front. The conceptual operations are listed above; their precise signatures and
    the cortex-side action neurons that trigger them are TBD.
11. **Salience-to-lookback mapping.** Proposal: `lookback = floor(k · |z|)` capped by available trajectory
    length. Tune `k` empirically from the first salience-driven rewind experiments. Possible refinement: separate
    coefficients for reward-z vs prediction-error-z (a large reward miss may want a different rewind depth than
    a large surprise).
12. **Default fan-out width.** How many concurrent trajectories does a pool dispatch? Probably small (4-8) for
    involuntary forecasts and larger (up to capacity) for voluntary think-actions where the user explicitly
    requested deep deliberation. Capped by hippocampal compute budget.
13. **Trajectory diversity.** When `fanout` exceeds the number of alternatives the active set's action
    connections name, how are the extra slots filled? Stepping through actions the set has never seen but that
    hold event connections from elsewhere (H4)? Leaving them empty? Empirical question.
14. **Hypothetical-union assembly.** For imagined-scenario construction, what determines which patterns/moments
    a cue binds into the hypothetical moment — the literal cue set only, or the cue plus what its neighbors
    reach by pattern-completion? Over-completion risks reconstructing a familiar real situation instead of the
    intended novel one; under-completion risks a hypothetical too sparse to forward-simulate. And since the
    constructed union has no connections of its own, forward simulation reads its constituents' — confirm this
    composes sensibly rather than producing incoherent forward walks. Tune from the first imagined-construction
    experiments in Phase 7.
15. **Rationing replay per moment.** Replay writes imagined activations into the same history real frames do, so
    a moment rehearsed often holds a history that is mostly imagined and its real exposures are evicted by its
    rehearsals. Whether replay should be rationed per moment — a cap on imagined activations per real one, or a
    rule that an imagined activation evicts the oldest imagined activation first — is open. The cost of not
    rationing is that a moment's estimate can come to rest on rollouts alone; the cost of rationing is a second
    horizon beside `H`, which the spec forbids everywhere else. Measure first: the imagined share of histories,
    per moment, over exposure.
16. **Writeback against the spec's connections.** The spec's connection is a lifetime total that nothing leaves
    (R31, R34); the writeback above re-reads strength and estimate from a sliding history, which is what lets a
    rollout overturn a habit within `H`. The two readings disagree, and which one a moment's vote should use —
    or whether an imagined step should instead be an exposure with a reward share, folded in at `1 / strength`
    like a real one — is not settled here.

---

## Replay Pathologies

Because replay reinforces future routing and behavior, the system may develop:
- obsessive replay loops,
- irrational associations,
- overgeneralizations,
- false abstractions,
- or self-reinforcing beliefs.

These are considered expected emergent properties of replay-driven cognition rather than implementation bugs.

The architecture assumes that environmental interaction, prediction error, and sensory correction will counterbalance pathological replay over time.

---

## Why this matters

The architecture does not use:
- vector embeddings,
- transformer attention,
- symbolic logic trees,
- immutable memory records,
- exhaustive planning,
- centralized retrieval,
- or static semantic representations.

All cognition emerges from:
- sparse hierarchical pattern formation,
- contextual routing,
- replay traversal,
- reinforcement,
- and forgetting.

- **Long-term memory** as cortical neurons minted at salient instants and on the clock — not as a separate store, not as a context window.
- **Episodic-to-semantic consolidation** as a structural consequence of the majority over a broadly-named neuron's history, not as a separate process.
- **Two thinking modes** — involuntary background forecasting + voluntary think-actions — both running on the same machinery.
- **Imagination as union-mint with a supplied cue** — the hippocampus constructs hypothetical situations from never-co-fired parts, forward-simulates them through the learned transition model, and caches a conditional policy onto the hypothetical that real recurrence later licenses. Scenario construction is a mechanism here, not just a cited capacity; imagination proposes, recurrence licenses.
- **System 2 enriches System 1** — moments enter the cortical substrate and become raw material for further pattern formation. Expertise development is structural, not behavioral.
- **One activation rule, one voting rule** — patterns and moments share the same machinery once they exist; the hippocampus's distinct role is purely in *creating* moments and *running experiments* to update their action connections.
- **Strategic vs reflexive action selection emerges from the data**, not from a planning module — moments hold action connections at their level's offsets, and the votes aggregate naturally at whatever horizon the current context engages.
- **Union of unions** — the moment hierarchy recurses: level-1 moments bind patterns, level-2 moments bind level-1 moments, level-N moments bind level-(N-1) moments. Each level at a doubled reach. Coverage across levels makes the hierarchy itself the planning-horizon selector — no separate mechanism needed.
- **Symmetric hierarchies, opposite operations** — cortex builds up from sensory through pattern levels by intersection (compression). Hippocampus builds up from moments through moment levels by union (binding). Same exponential timescale structure, opposite information operations. The full graph has sensory at the bottom, patterns in the middle, moments at the top, with both hierarchies speaking the same temporal language.
- **Forgetting that judges** — moments fade or sharpen by their honest match with recurring reality; the death ledger remembers what was lost.
- **Biological grounding** — implements hippocampal indexing, complementary learning systems, predictive processing, scenario construction, and episodic-semantic consolidation in one architecture, with HM and rumination as direct architectural predictions.
- **One substrate, two operators, thalamic bus.** Cortex builds by intersection. Hippocampus binds by union. Activation, voting, and forgetting are unified across both.

The thesis: the next generation of AI architectures will not come from scaling sequence models. It will come from systems that have the structures biological brains have — separate organs for perception, memory, and thinking, coupled bidirectionally and operating on different clocks over a shared substrate, with one substrate underneath that both organs act on through different creation rules.

Intelligence emerges not from storing information, but from continuously reconstructing compressed experience through replay.