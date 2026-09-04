# Algorithm Implementation

The implementation-facing half of [algorithm.md](algorithm.md): the per-neuron state and calls, the staged
build plan, and the deltas against the current code. The algorithm is specified there; nothing here changes
it. Where the design is settled but its data structures are not, this document says so rather than inventing
them.

## Neuron state

The complete per-neuron state, D20 read as storage. Sets are sorted id lists; nothing stores a frame number.

```
id                                          // (dim, bucket) at base; opaque id above

// the dictionary lines — the only stored definitions
patterns:     Map<pattern_id, {
                 neighborhood,              // sorted set of (neuron, offset ≤ 0); the line; moves at re-center
                 child,                     // the pattern neuron one level up
                 counts }>                  // per (neuron, offset ≤ 0) over the firings it covers, credited only

// the evidence
history:
  ring:       FIFO<firing>                  // capacity H; arrival order = eviction order
  firing:     { position,
                backward:  set of (neuron, offset ≤ 0),
                forward:   set of (neuron, offset > 0) as landed, and beside each action the reward,
                cover:     the patterns covering it, held (R6),
                assignment: which pattern of the cover holds each present backward neighbor }

// the forward record — a total over the ring; recomputable, in no line of the file
forward:      Map<(neuron, offset > 0), count>                       // event slots; expected iff 2·count > n
actions:      Map<(action_neuron, offset > 0), { strength, estimate }>  // action slots; born with the default
```

**What must always hold.** Each is checkable in a test by recomputing from the ring and comparing against the
incrementally maintained state:

- Every pattern's `counts` equal a from-scratch recount over the firings whose held cover holds it, restricted
  to the neighbors assigned to it. Its `neighborhood` is R4's collapse over those counts, with the line
  charged and equality held.
- Every firing's `cover` is one R18 could have produced against some past table, and no re-derivation against
  the current table is strictly cheaper than it (R6).
- `forward` and `actions` equal a from-scratch sum over the ring's forward halves. A slot no firing in the
  ring still holds is absent.
- Every pass of the bill leaves the neuron's file (T6's `L_N`) no longer than it found it.

**Not yet designed.** How the residual per firing and the seed tally per bill are kept so R14 is one pass
rather than a rescan; whether `cover` is stored as pattern ids or as an index the way the old histogram stored
servers; and what the three-way comparison in R6 costs when a candidate is installed against a full ring. The
old histogram, `normal`, `fallback` and running-benefit structures are retired: the spec has no default
pattern (D22) and no per-firing server, and the benefit is R12's margin read off `counts`.

## The machine–neuron interface

**The machine owns the open activations; the neuron owns its table, its history and its record** (D6, D20).
An open activation is `(position, age, its firing)`, held one per `(neuron, age, position)` on the machine
side. Nothing about a frame lives in the neuron.

There are two calls and no others:

```
process frame   — made at age 0 only, once per neuron per frame, with every activation that fired
                in:  each activation's backward half
                out: per activation, a bid for every pattern that applies (R18 step 3)
                     plus one request: the candidate that paid, and the pattern that retired (R19)

process actions — made once per frame after every level has run, with every open activation the machine holds
                in:  per activation, the neurons of its level that fired this frame (its forward neighbors at
                     offset = age), and any reward share for an action it already recorded (R33)
                out: per activation on the apex, its expectations and its inferences at offsets age+1 … reach,
                     in its own level's alphabet; covered activations return nothing
```

**The bill runs inside `process frame`, before the offer** (R19). The neuron covers and folds the new firing,
re-centers once, builds one candidate, retires one pattern, then offers. The election runs after the call
returns and reports nothing back (R23). The machine returns the requested child's identity on the next call or
as a separate reply; either way the pattern is in the table from the next frame (R13).

**`process actions` is age-blind by construction.** It walks every open activation the machine holds and hands
each what landed. A neuron with reach `r` is therefore reached `r + 1` times per activation on the forward
side — once per frame it is open — and each visit is a write into one firing plus, on the apex, a read of the
record. Reads move nothing (T8).

**Coverage inhibits on the machine side, not in the neuron** (D7). The machine knows which activations the
coverage set holds; it skips their speech and still delivers their forward neighbors and rewards.

## The bill, as methods

R19's five passes, in order. All prices are D16's fit over `O⁻`; all sums run over the ring.

**`cover_and_fold(O)`** — pass 1. R18 steps 1 and 2 over the current table: the greedy cover by ratio, the
assignment by first-namer. Push the firing with its cover and assignment; if the ring was full, pop the oldest
and subtract its assigned neighbors from its cover's counts and its landed forward half from the record.
Add the new firing's assigned neighbors to its cover's counts.

**`recenter()`** — pass 2. Every pattern whose counts moved re-collapses per slot with the line charged and
equality held (R4). Every firing whose table moved under it re-derives its cover and keeps the cheaper (R6).

**`build_one() → Option<Request>`** — pass 3. R14: tally the residual per neighbor over the ring, seed on the
largest (ties to declaration order then the nearer offset), take the firings whose residual holds the seed as
the population, collapse per slot with the same abstention. R15: price it over the firings whose cover it
would join, on residual neighbors only, against `1 + |C|`. If it pays, return the request with the definition
`C` carries at the end of the bill.

**`retire_one() → Option<pattern_id>`** — pass 4. Read every margin (R12), this bill's candidate included;
retire the smallest if strictly negative (R17). It leaves the table now; the firings it covered re-derive
(R6); its child goes on the request as a delete.

**`offer(O) → bids`** — pass 5. A bid for every pattern with more than half its neighbors present in `O⁻`,
less the candidate just requested. Each bid is the child id and the neighborhood.

**`register_child(id)`** — on the reply. Bind the pending pattern to its child id. Then every firing takes the
cheapest of its held cover, its held cover with the newcomer appended, and its cover re-derived (R6), and the
newcomer's counts are whatever those covers assign it.

**`accrue(age, arrivals, reward)`** — the forward call. Write the arrivals into the firing at offset `age`,
the reward beside the action it names, and add both to the record. If on the apex, return the record's slots
at offsets `age + 1` onward: event slots that clear the majority with their counts, action slots with strength
and estimate.

## The forward side — the code against the design

The temporal side of the current brain (`neuron.rs`, `thalamus.rs`, `brain.rs`) already has the shape D20, R31
and R36 describe. What matches, and what has to change:

**Already the design.**

- Connections live on the neuron per distance, and a neuron active at age `k` learns a distance-`k` connection
  toward each current active (`learn_temporal_connections`), so every level learns its own record from its own
  open activations.
- `vote(age)` reads `temporal_connections[age + 1]`, which is R36's offset.
- Ages that activated a child pattern are suppressed and do not vote (`get_suppressed_ages`), which is D7's
  silencing; they still learn.
- Strength is a count and the reward is the exact mean via `1 / strength` (`strengthen_connection`), which is
  R31. A negative mean wires the next untried action in the channel at neutral reward
  (`upsert_connection` → `find_alternative_action`), which is R37.
- `aggregate_votes` normalizes each voter to one unit per `(dimension, distance)` split by strength, events win
  by share and actions by reward, and level appears nowhere in it. That is §13 and R36's base-level vote.

**Deltas.**

1. **Targets are the voter's own level, not the base.** `process_temporal_levels` hands every level the
   level-0 active set as `sensory_neurons`, and `aggregate_votes` panics on a pattern-neuron target. Each level
   is handed its own age-0 actives instead, connections may target pattern neurons, and the panic goes.
2. **Expansion before resolution.** A new pass between `collect_votes` and `infer_neurons`: every vote whose
   target is a pattern neuron expands through dictionary lines to base symbols at composed offsets (R27), each
   carrying the vote's strength and reward unchanged; symbols landing at or before the current frame are
   dropped (R28). The expansion exists for spatial patterns already and is reused. `aggregate_votes` then runs
   on base targets only, as it does today.
3. **A majority over a ring, not weights that never leave.** `temporal_connections` keeps every target that
   ever followed and never weakens one. The record becomes a total over the ring: eviction subtracts the
   evicted firing's forward half, a slot's strength is the number of ring firings holding it, an event slot is
   expected only when `2·count > n` (R4), and an action slot's estimate is the mean over the exposures the
   ring holds, recomputed exactly on eviction rather than smoothed. Slots the bootstrap or the walk wired stand
   at strength 0 until tried (R31). This is the change that lets a neuron answer a changed world within `H` of
   its own firings.
4. **One record, one firing.** The forward half is stored per firing (R8) and the record is its sum; today the
   connection map is the only copy and nothing per firing exists to subtract.
5. **Rewards land beside the neighbor, shaped by R33.** Today every open age is handed the frame's reward whole
   and unscoped (`decorate_temporal_actives`), and a second path attributes `rewards[age − distance]`. Both go.
   A reward names channels and a span, shares fall linearly over the span, and each share is written beside the
   action neighbor at the offset the distance names, in every open activation's firing.
6. **Base neurons vote.** They already do in the code; the design keeps it. Nothing to change, and the risk it
   carries is in [algorithm-evaluation.md](algorithm-evaluation.md).
7. **Patterns are minted at the bill, never on a missed vote.** `recognize_temporal_patterns`,
   `correct_errors`, `evaluate_vote_error` and the error-pattern allocation are the old error-driven path and
   are retired, not ported. Prediction mints nothing (D9). The temporal pattern hierarchy is the same bill as
   the spatial one at `reach_t > 1`, which is the reason for one stack (R25).

## The build plan

Each phase lands independently and is measured before the next. The build order follows the spec's parts:
the neuron's bill, then contraction, then the forward side and actions. The headline metric through the
event stages is the objective itself — **apex neurons per level per frame, paired with the dictionary size
that bought them** — tracked **as a function of exposure**: on recurring data both curves should fall and
flatten. Alongside: churn (builds + retirements per thousand firings, which should decay as tables settle),
task accuracy (train and held-out), neuron counts per level, and wall-clock per frame.

### Stage 1 — the bill

**Phase 1 — substrate, evidence, and the table.** Sparse activation (a dimension with nothing happening
supplies no symbol); the ring of firings with held covers and assignments; the greedy cover; the five-pass bill
with one build and one retirement per firing; the request-and-reply mint. The invariants above become
`debug_assertions` that recompute from the ring on every mutation. Gate on the exposure curves: dictionary
size sublinear in exposures, apex per frame falling, churn decaying. Also measure history memory and per-frame
wall-clock. Cap at one level so the recursion is not a variable yet.

**Phase 2 — contraction.** The wide offer, the election, and the level-above construction on the sparse
substrate. Measured by the per-level reduction factor actually achieved and the depth at which it settles.

**Phase 3 — the readout gate.** Compare held-out accuracy against what the level counts justify. Do not build
past this phase if the answer is no.

### Stage 2 — the forward side

The forward record on every neuron, the forward call, expansion before resolution, and the base-level vote —
deltas 1 through 4 above. Gate: held-out prediction on sequence data, and the same exposure curves on the
temporal dictionaries. The temporal pattern hierarchy is Stage 1's bill run at `reach_t > 1` and is not a
separate mechanism.

### Stage 3 — actions and rewards

Action dimensions in the channels; the default as a slot on every neuron; R33's shaped rewards (delta 5);
exploration on a negative estimate; the standing inference and top-down expansion of a selected higher action
(R30, R36). Gate: a closed-loop environment in which a learned action sequence answers a learned event
sequence.

Variable-length pricing lands on its own track, specified in [forgetting.md](forgetting.md).

## Changes required in the current code (not yet implemented)

The current implementation predates most of the above. Each delta is tagged with the phase it lands in. The
forward-side deltas are the numbered list in the section above and land in Stages 2 and 3.

**Phase 1 — the substrate and the evidence.**

1. **Sparse emission is an encoder change only.** `build_frame` iterates whatever the inputs map holds, so
   omitting a dimension — or a whole channel — already works with no change in the brain. On MNIST it is the
   encoder skipping off pixels.
2. **Rebuild the history.** `SpatialHistory` becomes a FIFO ring of firings, each carrying its backward half,
   its forward half as it lands, its held cover and its assignment. The per-config `frames: Vec<FrameNumber>`
   and the absolute-frame `age_spatial_history` cutoff both go: capacity is `H` in the neuron's own firings,
   and eviction is one-out-one-in off the ring. `SpatialHistory::rebase` goes with the frame numbers. The
   histogram keyed on identical contexts goes: covers are held per firing, so identical backward halves no
   longer share one (R6).
3. **Replace the server with the cover.** Routing chooses one closest entry today; it becomes R18's greedy
   cover by ratio with the first-namer assignment, and both are written into the firing.
4. **Delete the normal.** `spatial_normal_config`, `refresh_normal_config`, `served`, `spatial_target_channels`
   and the channel plumbing that feeds it go. The spec has no default pattern (D22); what no pattern covers is
   the residual, one line each.
5. **Replace the delete scan with R17.** `spatial_delete_candidate` and `spatial_delete_candidate_uncached` are
   deleted outright. Retire becomes: read every margin off `counts`, retire the smallest if strictly negative,
   at most one per bill.
6. **Replace the greedy growth with R14.** `spatial_add_pays` and its candidate construction become seed,
   population, collapse, and one price, once per bill.
7. **Remove the new-child bid path.** Delete the `NEW_CHILD_BID` sentinel and every branch keyed on it: the
   request construction in `process_spatial_frame`, the split into `recognized` / `new_child_parents` in
   `process_spatial_level`, and the mid-frame create-install-activate block. New children stop competing in
   `elect_spatial_bids` entirely.
8. **Move minting to the bill, as a request and a reply.** The bill runs before the offer, inside the same
   call; the request leaves with the bids; the thalamus batch-allocates and dispatches `register_child` back;
   the parent binds the pending pattern to the returned id. The newborn is **not** activated: it is offered
   first on the next firing (R13).
9. **Delete the birth special cases** — the newborn's insertion into `new_error_pattern_ids` for the level
   above, the no-subsume-on-birth-frame rule, and the fires-but-does-not-record state.
10. **Record unconditionally.** Every active neuron folds its firing, as covered: bought or not, covered or
    not. Today a losing bid commits nothing and the frame vanishes — that goes.
11. **Delete the evidence coupling to the election.** Remove `prune_inhibited_spatial_history` /
    `drop_inhibited_spatial_frame` and the subsumed-set plumbing from the evidence path entirely. The subsumed
    set survives only where it belongs: deciding what the level above sees and who speaks. The election writes
    nothing into any neuron, and there is no report back (R23). With nothing left to commit conditionally, the
    decide-then-commit round trip goes too: `SpatialCommitOp`, `commit_spatial_frame`, and the
    thalamus/region/column `commit_spatial_frames` chain.
12. **Remove the completeness gate.** `observed.len() >= spatial_capacity` in `process_spatial_frame` has no
    counterpart in the design, and under sparse activation it would rarely pass. `spatial_capacity` and its
    plumbing go with it.
13. **Drop the spatial recency machinery.** `activation_strength`, `last_activation_frame`, the lazy decay,
    `forget_rate` and `can_delete_child` describe nothing in the design — patterns live and die by the one
    test. Death frames stay, re-derived as R17 states them: a retired pattern's child dies when its last open
    activation closes.
14. **Backups break.** The persisted history changes shape, so old backups are unloadable. Accepted: no
    migration path, bump the format and fail loudly on an old file.
15. **The invariants become debug assertions.** The four statements under "What must always hold", each
    recomputed from the ring and compared against the incrementally maintained state.

**Phase 2 — contraction.**

16. **Price named-and-absent in bids.** Carry `1 + |e \ O⁻|` on the bid as its price and change the survival
    test in `spatial_survivors` from a flat `≥ 2` to R23's `covers > price`.
17. **Offer every pattern that applies.** Today a neuron bids its routed entry only. R18 step 3 sends a bid for
    every pattern with more than half its neighbors present, so the election sees the catalog.
18. **Replace the above-base adjacency heuristics with the reach.** The eight-sector nearest rule
    (`directional_neighbors`) and the reuse-the-last-declared-set fallback both go: adjacency is D4's box at
    D14's reach, at every level. `channel_positions` and `set_channel_position` serve only the sector rule and
    go with it.

**Phase 3 — the readout gate.**

19. **Watch for the one expected regression:** a child's own record starts empty at its mint (D17), so a newly
    bought child expects nothing and infers the default until its ring fills.

## The MNIST frame protocol

MNIST runs on R29's chain — infer, execute, reward — one example per three frames. Nothing in
[algorithm.md](algorithm.md) changes for it. A base event neuron's reach in time is 1 (D14), so its firing
spans the frame before and the frame after its own.

```
frame     carries              what happens
-----     -------              ------------
f         events only          base event neurons fire, bill, and offer (R18, R19); contraction
                               runs; process actions delivers nothing new and collects the
                               apex's inferences; the vote at the base commits the digit call
                               for f + 1 (R36)
f + 1     the action only      the digit call executes and its neuron fires; events are
                               silent. Process actions runs and every activation open here
                               writes the action into its firing at its own age (R31)
f + 2     the reward only      the label arrives as input, not as a symbol (§15), and is
                               written beside the action in every open firing, from where it
                               enters the neuron's estimate (R31, R33). Nothing fires.
f + 3     next example         = the next example's f
```

**The image is presented once, at `f`.** Frames `f + 1` and `f + 2` carry no event symbols at all. Every event
dimension is silent in them (D5).

### What follows from that

**Base event processing is spatial.** The backward slot at `−1` lands on the previous example's reward frame
and the forward slot at `+1` on the action frame; both are silent. So every neighbor a base event neuron names
sits at temporal offset `0`. The temporal slots are voted out for want of a majority (R4, R11) and cost
nothing in `|e|`.

**No action patterns form.** An action neuron's own backward and forward slots land on frames carrying no
actions, so the action hierarchy stays flat. R32's apex active action is therefore always the base action,
which R32 states explicitly holds before any action pattern exists.

**Action slots are recorded per frame, never at the bill.** R31 records one at every age a neuron is open at,
and the reward lands a frame later. Neither is gated on anything completing, so the reward path does not wait
on the window and does not vary with level.

**This is why `process actions` is a second call.** `process frame` reaches a neuron at age 0 only, so the
forward half cannot ride on it — an activation at age 3 of a reach-8 span would never be reached. `process
actions` walks every open activation the machine holds and hands each one what landed. It also runs after the
stack has settled rather than during a level, because what ran is not known until then.

**Classification is selection at the base.** The digit call is an action chosen by R35 and R36: every apex
activation's action slots at offset `age + 1`, expanded to base actions, resolved per action dimension by
largest estimate with the code's per-voter normalization breaking ties. The current brain's per-dimension
vote in `aggregate_votes` is that resolution and stays. The naive-Bayes readout over active neurons has no
counterpart in [algorithm.md](algorithm.md) and is retired, not ported.

### Encoder changes

- Emit the image on one frame only; emit nothing on the action and reward frames.
- Sparse emission: an off pixel supplies no symbol (D5). No neuron is emitted for it.
- Declare the digit calls as an action dimension, in a fixed order — R37 walks that order for exploration, so
  it is part of the problem statement.
