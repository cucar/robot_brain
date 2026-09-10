# Algorithm Implementation

The implementation-facing half of [algorithm.md](algorithm.md): the per-neuron state and calls, the staged
build plan, and the deltas against the current code. The algorithm is specified there; nothing here changes
it. Where the design is settled but its data structures are not, this document says so rather than inventing
them.

## Neuron state

The complete per-neuron state, D16 read as storage. Sets are sorted id lists. Nothing stores a coordinate:
expiry was the frame number's only reader and expiry is a FIFO depth (R3), and no comparison, price, count or
vote ever reads a position (D11). The coordinate is the machine's, on the open activation (D9).

```
id                                          // (dim, bucket) at base; opaque id above

// the dictionary lines — the only stored definitions
patterns:     Map<pattern_id, {
                 neighborhood,              // sorted set of (neuron, offset ≤ 0); the line; moves at re-center
                 child,                     // the pattern neuron one level up
                 counts }>                  // present, held and n per (neuron, offset ≤ 0) — see Pattern counts

// the evidence
history:
  ring:       FIFO<activation>                  // capacity H; arrival order = eviction order
  activation:     { backward:  set of (neuron, offset ≤ 0),
                cover:     the patterns covering it, held (R10),
                assignment: which pattern of the cover holds each present backward neighbor }

// the connections — lifetime totals; in no line of the file, and not recomputable from anything
actions:      Map<(action_neuron, offset > 0), { strength, estimate }>  // event neurons only; born empty (R35)
```

**What must always hold.** Each is checkable in a test by recomputing from the ring and comparing against the
incrementally maintained state:

- Every pattern's `counts` equal a from-scratch recount over the activations whose held cover holds it — `present`
  and `held` per slot as defined under Pattern counts below. Its `neighborhood` is R7's collapse over those counts,
  with the line charged and equality held.
- Every activation's `cover` is one R9 could have produced against some past table, and no re-derivation against
  the current table is strictly cheaper than it (R10).
- `actions` only ever grows: no strength falls, and a connection leaves only with the death of either of its
  ends (R31). Its estimate is the mean of the shares it has received, over its strength.
- Every pass of the bill leaves the neuron's file (T7's `L_N`) no longer than it found it.

**Not yet designed.** How the residual per activation and the seed tally per bill are kept so R14 is one pass
rather than a rescan; whether `cover` is stored as pattern ids or as an index the way the old histogram stored
servers; and what the three-way comparison in R10 costs when a candidate is installed against a full ring. The
old histogram, `normal`, `fallback` and running-benefit structures are retired: the spec has no default
pattern (D21) and no per-activation server, and the benefit is R12's margin read off `counts`.

## Pattern counts

The collapse (R7) reads two numbers per pattern per slot: how many activations are in the population there, and how
many of them name the neuron there. Both are sums over the activations the pattern covers, and the code keeps them
as running tallies so that re-centering never re-reads the ring. Nothing in this section is design: every tally
equals a from-scratch recount, and the invariant above checks that it does.

**What a pattern keeps.** Over exactly the activations it covers, two sparse tallies per `(neuron, offset)` —
indexed by the neighbors actually seen, not by everything the box admits — and `n`, the number of activations it
covers:
```
present(p)   the covered activations in which p fired and no other pattern of that activation's cover
             holds it — its own share plus the residual
held(p)      the covered activations in which another pattern of the cover holds p
```
At slot `p` the collapse's population is `n − held(p)` and its count is `present(p)`. A pattern therefore tallies
neighbors it does not name, because whether it should name them is the question re-centering asks (R8), and a
slot only the residual has ever held is how a pattern grows. A neighbor another pattern of the same cover holds is
that pattern's evidence, not this one's, and is counted here only as an abstention; a neighbor in the residual is
nobody's yet, and is evidence for every pattern of the cover.

**What moves them.** The three events R8 names, and every one of them moves a whole activation's worth:
```
an activation is saved          every pattern of its cover adds the activation's contribution — one to `n`,
                                `present` for each neighbor it holds or the residual holds, `held` for
                                each neighbor another pattern of the cover holds
an activation is evicted        every pattern of its cover subtracts the same; no connection is
                                touched (R31)
an activation's cover changes   every pattern of the old cover subtracts the activation's old contribution,
                                every pattern of the new cover adds its new one
```
A pattern that stays in a changed cover still subtracts and re-adds, because what the other patterns took or gave
back moves its `present` and `held`. An activation whose cover has changed takes its counts with it, so the pattern
that received an activation's share is always the pattern that gives it back, and a share moves whole, so a
pattern joining or leaving a cover transfers it in `O(offsets)`. Counts move only in `process frame`, so
re-centering costs nothing to trigger: the counts it reads are current by the time the call reaches it.

**In dependency order**, so the list also says what to recompute when something moves:
```
activation.cover          =  the patterns covering its neighborhood, chosen by R9 and held by R10
activation.assignment[n]  =  the pattern of the cover credited with present neighbor n — none, when
                             n is in the residual
pattern.counts            =  Σ over the activations it covers: its share and the residual as
                             `present`, what other patterns of the cover hold as `held`
```

**The ring makes eviction exact.** Removing the oldest activation means subtracting the neighbors *it* contributed,
which a tally cannot recover, so each activation keeps its own neighborhood and a pattern's counts are the cached
aggregate over them. Eviction reads a neighborhood whole; everything else reads it per slot, off the counts.

**Why records and not a summary.** A total cannot answer retirement: when a pattern goes, its neighbors
have to be re-covered from the table, which needs the activations and what each holds against each pattern — a
single number per pattern could not produce it. Keeping the ring is not a storage saving; it buys that both
tests scan distinct backward contexts and read pre-summed counts.

**Why a pattern tallies neighbors it does not name.** A pattern used to count only the neighbors
assigned to it. That is enough to decide whether to *keep* a named slot and never enough to decide whether to
*enter* one: a neighbor the pattern does not name is never assigned to it, so its count was identically zero
and R7 could never take it. R7's abstention paragraph says a pattern grows into the residual, and the state as
defined could not support that sentence; R8's claim that re-centering needs no pass of its own was not true of
the one count that growth depends on.

The fix is the smallest that makes R7 exact. At each slot R7 wants two numbers: how many covered activations had
the neighbor there and unclaimed by another pattern of the cover, and how many abstain because another pattern
holds it. Those are `present(p)` and `held(p)`, and the population is `n − held(p)`. Both are sparse — indexed
by neighbors actually seen — and both move a whole activation's worth at a time, so the granularity above is
unchanged. The one new obligation is the third event above: when an activation's cover changes, a pattern that
*stays* in the cover still subtracts and re-adds, because what another pattern took from the residual moves this
pattern's `present` to `held`, or back.

**Worked case.** `e = {b@0, c@0}` covers ten activations and `d@0` begins to appear. `d` is residual in every
activation that has it, so `present_e(d)` climbs by one per such activation while `held_e(d)` stays zero. At
`2 · present_e(d) > 11` — six of ten — re-centering enters `d` and `e` becomes `{b, c, d}`. The four activations
without `d` now price `e` at 2, which is what they were paying before (line plus one residual), so R10 lets them
keep it, and they evict in turn. **No candidate could have done this**: a candidate is built on the residual alone (R14), `b` and
`c` are held by `e` in those activations, and `{d}` alone saves nothing (R15).

## The machine–neuron interface

**The machine owns the open activations; the neuron owns its table, its history and its connections** (D9, D16).
An open activation is `(its activation, coordinate, age, covered at)`, held one per `(neuron, age, position)` on the
machine side. Nothing about a frame lives in the neuron.

There are two calls and no others:

```
process frame   — made at age 0 only, once per neuron per frame, with every activation that fired
                in:  each activation's backward half
                out: per activation, a bid for every pattern that applies (R9 step 3)
                     plus one request: the candidate that paid, and the pattern that retired (R20)

process actions — made once per frame after every level has run, with every open activation the machine holds
                in:  per activation still uncovered, the apex action that ran this frame in each action
                     dimension, at offset = age; per activation, any reward share for a frame it already
                     wrote (R33)
                out: per activation on the apex, its inferences: its connections at every offset beyond its
                     age, each placed at its completion; covered activations return nothing
```

**The bill runs inside `process frame`, before the offer** (R20). The neuron covers and folds the new activation,
re-centers once, builds one candidate, retires one pattern, then offers. The election runs after the call
returns and reports nothing back (R24). The machine returns the requested child's identity on the next call or
as a separate reply; either way the pattern is in the table from the next frame (R17).

**`process actions` is age-blind by construction.** It walks every open activation the machine holds and hands
each what landed. A neuron with reach `r` is therefore reached `r + 1` times per activation on the forward
side — once per frame it is open — and each visit, while the activation is uncovered, is a write into the
connections plus a read of them. Reads move nothing (T8).

**Coverage inhibits on the machine side, not in the neuron** (D10). The machine knows which activations the
coverage set holds and the age coverage arrived at; it skips their speech and their apex action, and still
delivers a reward share for any frame they wrote (R33).

## The bill, as methods

R20's five passes, in order. All prices are D22's fit over `O⁻`; all sums run over the ring.

**`cover_and_fold(O)`** — pass 1. R9 steps 1 and 2 over the current table: the greedy cover by ratio, the
assignment by first-namer. Push the activation with its cover and assignment; if the ring was full, pop the oldest
and subtract its contribution from its cover's counts (Pattern counts). The connections are untouched. Add the new activation's
contribution to its cover's counts.

**`recenter()`** — pass 2. Every pattern whose counts moved re-collapses per slot with the line charged and
equality held (R7). Every activation whose table moved under it re-derives its cover and keeps the cheaper (R10).

**`build_one() → Option<Request>`** — pass 3. R14: tally the residual per neighbor over the ring, seed on the
largest (ties to declaration order then the nearer offset), take the activations whose residual holds the seed as
the population, collapse per slot with the same abstention. R15: price it over the activations whose cover it
would join, on residual neighbors only, against `1 + |C|`. If it pays, return the request with the definition
`C` carries at the end of the bill.

**`retire_one() → Option<pattern_id>`** — pass 4. Read every margin (R12), this bill's candidate included;
retire the smallest if strictly negative (R18). It leaves the table now; the activations it covered re-derive
(R10); its child goes on the request as a delete.

**`offer(O) → bids`** — pass 5. A bid for every pattern with more than half its neighbors present in `O⁻`,
less the candidate just requested. Each bid is the child id and the neighborhood.

**`register_child(id)`** — on the reply. Bind the pending pattern to its child id. Then every activation takes the
cheapest of its held cover, its held cover with the newcomer appended, and its cover re-derived (R10), and the
newcomer's counts are whatever those covers assign it. The machine, for its part, opens an activation for the
child at the parent's coordinate at age 0 — accrual only: `process actions` reaches it, nothing else does
(R17).

**`accrue(age, apex_action, reward)`** — the `process actions` call. If the activation is uncovered, increment
the connection at `(apex action, age)` for each action dimension, creating it at strength 1. Fold each reward
share into the estimate of the connection it names, weighted `1 / strength`, whether or not the activation is
still uncovered; if an estimate is now negative, create the next untried action of that channel at that offset
at strength 1 and estimate 0 (R37). Nothing is written into the activation. If uncovered, return the connections at
every offset beyond `age`, each with its strength and estimate.

## The forward side — the code against the design

The temporal side of the current brain (`neuron.rs`, `thalamus.rs`, `brain.rs`) is the older model: a neuron
active at age `k` learns a distance-`k` connection toward every current active, event and action alike,
predicts the next frame's events from those connections, scores the prediction, and mints on the misses. The
design keeps one piece of that — the action connection — and none of the rest. What matches, and what has to
change:

**Already the design.**

- A connection is created at strength 1 or incremented, and never decremented or deleted except with a neuron
  (`strengthen_or_create_connection`, `delete_patterns`). Nothing per activation exists forward and nothing is
  subtracted on eviction. That is R31's "nothing leaves".
- Strength is a count and the reward is the exact mean via `alpha = 1 / strength` (`strengthen_connection`),
  which is R31's estimate and R34's plain average. A negative mean wires the first action in the channel's
  declared order that has no connection at that distance, at strength 1 and reward 0
  (`upsert_connection` → `find_alternative_action`), which is R37.
- `vote(age)` returns every connection with strength above zero — the whole distribution, no majority. That is
  R7's "connections are never collapsed", read at one offset where R36 reads every offset beyond the age.
- Ages that activated a child pattern are suppressed and do not vote (`get_suppressed_ages`), which is D10's
  silencing of speech.
- `aggregate_votes` normalizes each voter to one vote per `(dimension, distance)` split by strength;
  `determine_dimension_winners` takes actions by the share-weighted mean of the voters' rewards, ties to larger
  strength then lower id, and level appears nowhere in it. That is R36's base-level vote for actions. Its event
  half, the `Nb` consensus mode and the supervised `Brain.learn` wiring are not in the design and go: MNIST runs
  on three frames with ordinary rewards (below), and nothing wires a voter to an action but an action running.

**Deltas.**

1. **Event connections go.** `learn_temporal_connections` learns a connection toward every active neuron of the
   level below. It learns toward the apex action only (D25), and the event targets, the event vote,
   `track_inference_performance`, MAPE and the continuous-error path go with them. The design has no
   expectation output.
2. **Targets are the apex action, not the base set.** `process_temporal_levels` hands every level the level-0
   active set. Each open activation is handed the apex action of each action dimension instead — the highest
   action pattern that fired there this frame — so a connection may target a pattern neuron, and the panic on
   a pattern target in `aggregate_votes` goes.
3. **Read every offset beyond the age, and expand before resolution.** `vote(age)` reads one distance; R36
   reads every offset beyond the age and places each connection's completion `offset − age` frames ahead
   (R28). A new pass between `collect_votes` and `infer_neurons` expands every vote whose target is a pattern
   neuron through dictionary lines to base actions at composed offsets, carrying the vote's strength and reward
   unchanged; what lands at the frame ahead is resolved, and the rest of a winning pattern's placements stand
   as standing inferences (R36). The expansion exists for spatial patterns already and is reused.
   `aggregate_votes` then runs on base targets only, as it does today.
4. **Coverage stops learning, not only speech.** `get_suppressed_ages` silences a covered age's vote and lets it
   keep learning. Under D10 a covered activation writes nothing from the frame coverage arrived; the open
   activation carries the age it was covered at, and a reward share for an earlier frame still lands on the
   connection written then (R33).
5. **Rewards are scoped and shaped by R33; the fold is unchanged.** Today the reward given with a frame is
   attached whole to every action active in that frame, on every open age (`decorate_temporal_actives`), and a
   minted pattern is pre-wired with `rewards[age − distance]` on a second path. Both go. A reward names channels
   and a span, shares fall linearly over the span, and each share is folded into the connection at the offset
   the distance names, in every neuron whose activation wrote it, at `1 / strength` exactly as today. The
   current harness's whole-reward-in-the-frame-the-action-ran is R33's span-of-one case and keeps working
   unchanged.
6. **Base neurons vote.** They already do in the code; the design keeps it. Nothing to change, and the risk it
   carries is in [algorithm-evaluation.md](algorithm-evaluation.md).
7. **Patterns are minted at the bill, never on a missed vote.** `recognize_temporal_patterns`,
   `correct_errors`, `evaluate_vote_error` and the error-pattern allocation are the old error-driven path and
   are retired, not ported. Nothing forward mints anything (D12). The temporal pattern hierarchy is the same
   bill as the spatial one at `reach_t > 1`, which is the reason for one stack (R26).
8. **A minted child starts with no connections, because it is minted at age 0.** The code mints on a missed
   vote, at whatever age the miss was seen, so the pattern is born already behind, and
   `allocate_temporal_pattern_neuron` backfills the frames between the parent's activation and the miss with the
   actuals that followed and their rewards, at strength 1. The design mints at the bill, at age 0, on
   justification (R14, R15), so a child is born with nothing behind it to backfill. It is born empty, is given
   an activation at the parent's coordinate in the mint frame, connects from the next frame on, and keeps
   connecting after the parent's own activation has closed, since children outlive their parents (R17, R18).
9. **The default is not wired at birth.** `Column::create_neurons` gives every neuron a connection to each
   channel's default action at every voting distance, strength 1 and reward 0. That goes: a dimension no
   inference reaches runs the declared default, which is then the apex action of the frame and is learned by
   the ordinary path (R35).

## The build plan

Each phase lands independently and is measured before the next. The build order follows the spec's parts:
the neuron's bill, then contraction, then actions and rewards. The headline metric through the
event stages is the objective itself — **apex neurons per level per frame, paired with the dictionary size
that bought them** — tracked **as a function of exposure**: on recurring data both curves should fall and
flatten. Alongside: churn (builds + retirements per thousand activations, which should decay as tables settle),
task accuracy (train and held-out), neuron counts per level, and wall-clock per frame.

### Stage 1 — the bill

**Phase 1 — substrate, evidence, and the table.** Sparse activation (a dimension with nothing happening
supplies no symbol); the history of activations with held covers and assignments; the greedy cover; the five-pass bill
with one build and one retirement per activation; the request-and-reply mint. The invariants above become
`debug_assertions` that recompute from the ring on every mutation. Gate on the exposure curves: dictionary
size sublinear in exposures, apex per frame falling, churn decaying. Also measure history memory and per-frame
wall-clock. Cap at one level so the recursion is not a variable yet.

**Phase 2 — contraction.** The wide offer, the election, and the level-above construction on the sparse
substrate. Measured by the per-level reduction factor actually achieved and the depth at which it settles.

**Phase 3 — the readout gate.** Compare held-out accuracy against what the level counts justify. Do not build
past this phase if the answer is no.

### Stage 2 — actions and rewards

The temporal pattern hierarchy is Stage 1's bill run at `reach_t > 1` and is not a separate mechanism; gate it
on the same exposure curves over the temporal dictionaries. Then the forward side, which is actions and nothing
else: action dimensions in the channels; the connections on every event neuron, written from the frontier and
read at every offset beyond the age; expansion before resolution and the base-level vote; R33's shaped rewards;
exploration on a negative estimate; the standing inference and top-down expansion of a selected higher action
(R30, R36) — deltas 1 through 9 above. Gate: a closed-loop environment in which a learned action sequence
answers a learned event sequence.

Variable-length pricing lands on its own track, specified in [forgetting.md](forgetting.md).

## Changes required in the current code (not yet implemented)

The current implementation predates most of the above. Each delta is tagged with the phase it lands in. The
forward-side deltas are the numbered list in the section above and land in Stage 2.

**Phase 1 — the substrate and the evidence.**

1. **Sparse emission is an encoder change only.** `build_frame` iterates whatever the inputs map holds, so
   omitting a dimension — or a whole channel — already works with no change in the brain. On MNIST it is the
   encoder skipping off pixels.
2. **Rebuild the history.** `SpatialHistory` becomes a FIFO history of activations, each carrying its backward half,
   its held cover and its assignment. The per-config `frames: Vec<FrameNumber>`
   and the absolute-frame `age_spatial_history` cutoff both go: capacity is `H` in the neuron's own activations,
   and eviction is one-out-one-in off the ring. `SpatialHistory::rebase` goes with the frame numbers. The
   histogram keyed on identical contexts goes: covers are held per activation, so identical backward halves no
   longer share one (R10).
3. **Replace the server with the cover.** Routing chooses one closest entry today; it becomes R9's greedy
   cover by ratio with the first-namer assignment, and both are written into the activation.
4. **Delete the normal.** `spatial_normal_config`, `refresh_normal_config`, `served`, `spatial_target_channels`
   and the channel plumbing that feeds it go. The spec has no default pattern (D21); what no pattern covers is
   the residual, one line each.
5. **Replace the delete scan with R18.** `spatial_delete_candidate` and `spatial_delete_candidate_uncached` are
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
   first on the next activation (R17).
9. **Delete the birth special cases** — the newborn's insertion into `new_error_pattern_ids` for the level
   above, the no-subsume-on-birth-frame rule, and the fires-but-does-not-record state.
10. **Record unconditionally.** Every active neuron folds its activation, as covered: bought or not, covered or
    not. Today a losing bid commits nothing and the frame vanishes — that goes.
11. **Delete the evidence coupling to the election.** Remove `prune_inhibited_spatial_history` /
    `drop_inhibited_spatial_frame` and the subsumed-set plumbing from the evidence path entirely. The subsumed
    set survives only where it belongs: deciding what the level above sees and who speaks. The election writes
    nothing into any neuron, and there is no report back (R24). With nothing left to commit conditionally, the
    decide-then-commit round trip goes too: `SpatialCommitOp`, `commit_spatial_frame`, and the
    thalamus/region/column `commit_spatial_frames` chain.
12. **Remove the completeness gate.** `observed.len() >= spatial_capacity` in `process_spatial_frame` has no
    counterpart in the design, and under sparse activation it would rarely pass. `spatial_capacity` and its
    plumbing go with it.
13. **Drop the spatial recency machinery.** `activation_strength`, `last_activation_frame`, the lazy decay,
    `forget_rate` and `can_delete_child` describe nothing in the design — patterns live and die by the one
    test. Death frames stay, re-derived as R18 states them: a retired pattern's child dies when its last open
    activation closes.
14. **Backups break.** The persisted history changes shape, so old backups are unloadable. Accepted: no
    migration path, bump the format and fail loudly on an old file.
15. **The invariants become debug assertions.** The four statements under "What must always hold", each
    recomputed from the ring and compared against the incrementally maintained state.

**Phase 2 — contraction.**

16. **Price named-and-absent in bids.** Carry `1 + |e \ O⁻|` on the bid as its price and change the survival
    test in `spatial_survivors` from a flat `≥ 2` to R24's `covered > price`.
17. **Offer every pattern that applies.** Today a neuron bids its routed entry only. R9 step 3 sends a bid for
    every pattern with more than half its neighbors present, so the election sees the catalog.
18. **Replace the above-base adjacency heuristics with the reach.** The eight-sector nearest rule
    (`directional_neighbors`) and the reuse-the-last-declared-set fallback both go: adjacency is D4's box at
    D14's reach, at every level. `channel_positions` and `set_channel_position` serve only the sector rule and
    go with it.

**Phase 3 — the readout gate.**

19. **Watch for the one expected regression:** a child's connections start empty at its mint (D25) and hold
    one window's worth by its first purchase (R17), so a newly bought child infers from very little, or the
    default, until it has been bought enough times to hold more.

## The MNIST frame protocol

MNIST runs on R29's chain — infer, execute, reward — one example per three frames. Nothing in
[algorithm.md](algorithm.md) changes for it. A base event neuron's reach in time is 1 (D14), so its activation
spans the frame before and the frame after its own.

```
frame     carries              what happens
-----     -------              ------------
f         events only          base event neurons fire, bill, and offer (R9, R20); contraction
                               runs; process actions delivers nothing new and collects the
                               apex's inferences; the vote at the base commits the digit call
                               for f + 1 (R36)
f + 1     the action only      the digit call executes and its neuron fires; events are
                               silent. Process actions runs and every uncovered event activation
                               open here increments its neuron's connection to the action at
                               its own age (R31)
f + 2     the reward only      the label arrives as input, not as a symbol (§20), and is
                               folded into that connection's estimate in every neuron that
                               wrote it (R31, R33). Nothing fires.
f + 3     next example         = the next example's f
```

**The image is presented once, at `f`.** Frames `f + 1` and `f + 2` carry no event symbols at all. Every event
dimension is silent in them (D5).

### What follows from that

**Base event processing is spatial.** The backward slot at `−1` lands on the previous example's reward frame,
which is silent. So every neighbor a base event neuron names
sits at temporal offset `0`. The temporal slots are voted out for want of a majority (R7, R5) and cost
nothing in `|e|`.

**No action patterns form.** An action neuron's own backward slots land on frames carrying no actions, so the action hierarchy stays flat. R32's apex active action is therefore always the base action,
which R32 states explicitly holds before any action pattern exists.

**Connections are written per frame, never at the bill.** R31 writes one at every age a neuron is open at,
and the reward lands a frame later. Neither is gated on anything completing, so the reward path does not wait
on the window and does not vary with level.

**This is why `process actions` is a second call.** `process frame` reaches a neuron at age 0 only, so the
forward half cannot ride on it — an activation at age 3 of a reach-8 span would never be reached. `process
actions` walks every open activation the machine holds and hands each one what landed. It also runs after the
stack has settled rather than during a level, because what ran is not known until then.

**Classification is selection at the base.** The digit call is an action chosen by R35 and R36: every apex
activation's connections at every offset beyond its age, placed and expanded to base actions, resolved per action
dimension by
largest estimate with the code's per-voter normalization breaking ties. The current brain's per-dimension
vote in `aggregate_votes` is that resolution and stays. The naive-Bayes readout over active neurons has no
counterpart in [algorithm.md](algorithm.md) and is retired, not ported.

### Encoder changes

- Emit the image on one frame only; emit nothing on the action and reward frames.
- Sparse emission: an off pixel supplies no symbol (D5). No neuron is emitted for it.
- Declare the digit calls as an action dimension, in a fixed order — R37 walks that order for exploration, so
  it is part of the problem statement.
