# Algorithm Implementation

The implementation-facing half of [algorithm.md](algorithm.md): the per-neuron state and methods, the staged
build plan, and the deltas against the current code. The algorithm is specified there; nothing here changes it.

## Neuron state

The complete per-neuron state on the spatial axis. Sets are sorted id lists; nothing stores a frame number.

```
id                                          // (dim, bucket) at base; opaque id above

// the dictionary line — the only stored definitions
children:     Map<child_id, definition>     // definition = sorted set of neuron ids; frozen at mint

// derived state — tallies over the history below; recomputable, costs nothing in the file
connections:  Map<neuron_id, count>         // counts over the normal-served records in the history
served:       int                           // number of normal-served records (majority denominator)
normal:       definition                    // cached majority set { n : 2·count(n) > served }

// the evidence
history:
  ring:       FIFO<context_ref>             // capacity = the horizon; arrival order = eviction order
  histogram:  Map<context, {
                 count,                     // how many remembered records are this context
                 server,                    // Normal | Child(id) — closest entry under the CURRENT entry set
                 best_distance,             // d(context, server's definition)
                 fallback,                  // second-closest entry
                 fallback_distance }        // d(context, fallback's definition)

// the running one test
benefits:     Map<child_id, benefit>        // Σ (fallback_distance − best_distance)·count over records served
```

**What must always hold.** Every method may do what it wants while it runs, but by the time it returns these
four statements must be true again. They are the design's debug assertions — each one is checkable in a test
by recomputing from the raw history and comparing against the incrementally-maintained state:

- `connections` / `served` describe exactly the histogram entries with `server == Normal`, weighted by count —
  a from-scratch recount of the normal-served records always matches. `normal` is their element-wise majority.
- Every histogram entry's `server` / `fallback` really are the closest and second-closest entries under the
  *current* entry set. Adds, deletes, and normal movement re-derive them; this is what lets routing read the
  stored assignment instead of rescanning when the context is already remembered.
- `benefits[child]` always equals what a from-scratch recomputation would give:
  `Σ (fallback_distance − best_distance)·count` over the entries it serves. The delete test is then just
  "delete iff `benefits[child] < 1 + |children[child]|`" — strictly below cost, kept at equality — checked
  only when an event moved the benefit.
- Every committed structural change strictly decreases `L`. Since `L` is a non-negative whole number, cascades
  terminate and nothing can churn.

The normal has no margin: no storage line, never deleted.

On the temporal axis the same shape holds with two statistic tables per entry — context-side counts and
outcome-side counts — both moved together when a record changes hands, and records completed one frame late
(algorithm.md, "The history").

## Neuron methods

In call order within a frame. All distances are the fit (algorithm.md, "The fit"); all sums run over the
histogram, which never holds more entries than the horizon.

**`evict_if_full()`** — frame step 1. If the ring is at capacity: pop the oldest ref; decrement its histogram
entry's count (drop the entry at zero). If its server was the normal: subtract the context from
`connections`, decrement `served`, mark the normal dirty. If a child: `benefits[child] -= (fallback_distance −
best_distance)`; if the benefit falls strictly below that child's cost, `delete_child(child)`.

**`route_and_record(O) → (server, best_d, fallback, fb_d)`** — frame step 2, one operation. If `O` is already
in the histogram, take `argmin` over its stored distances — the assignment is re-derived here, so a stale
server corrects itself the moment the context recurs; otherwise scan the normal and every child for the two
smallest distances, ties to the older entry. Then write the record from the
same scan: push the ref, upsert the histogram entry — the server is routing's choice, promoted or not, covered
or not. If the normal serves: add `O` to `connections`, increment `served`, mark dirty. Nothing recorded here
is ever revoked — there is no pending state and no retraction.

**`serve(server)`** — frame step 3. Child: **activate it** — it fires, and the thalamus gets the recognition
bid: the active neurons the child's definition names correctly, bidder included, plus `f` = its
named-but-absent count. **Then return — the neuron is done for the frame.** The record already carries the
served distance as priced demand (a badly-served child frame is demand a future add can win); the add test
never runs on this path — the context was recognized, and the description job went up a level with the
child. Normal: no activation, no bid; fall through to the steps below.

*(The election runs in the thalamus, concurrently as far as the neuron is concerned — see algorithm.md,
"Contraction". It reads the bids and writes only the level above; none of the methods below depend on its
outcome.)*

**`infer()`** — frame step 4, normal path only, at frame `f`. Predict the inference set from `connections`,
used as the per-neighbor distribution they are. Spatially the inference is the neighborhood itself; temporally
it is the vote for the frame ahead.

**`review(best_d) → error`** — frame step 5, when the inference resolves: spatially `error = best_d`,
immediately; temporally at `f+1`, when the actuals arrive — and `add_test` below runs then too.

**`add_test(O) → Option<MintRequest>`** — frame step 6, only on a nonzero error. Solo benefit = `Σ` over
histogram entries strictly closer to `O` than to their current server of `(best_distance − d)·count` — the
triggering frame is already recorded, so its error enters through its own entry, no special term. Pass iff
`benefit > 1 + |O|`. A pass inserts `C` into `children` under a **pending id**, seeds `benefits[C] = 0`, runs
`settle(C)`, and returns a **request**, not a child: `{definition: C, level: own + 1, channel: own}` —
everything the allocation needs, decided here so the allocator decides nothing. The definition on the request is
read *after* `settle`, so it is the one `C` ends the frame with rather than the one the test priced.

**No joint add-and-delete test.** `C` is priced on its own wins while every incumbent is still paid for, and an
incumbent left worthless by the takeover fails its own delete check in `settle` step 3, in the same move
(algorithm.md, "Delete — pruning the table"). What the sequence cannot reach is a candidate that would pay only
if some child's storage were refunded first; that case is given up deliberately.

**`settle(C)`** — inline, immediately on a passing add test:
1. For each histogram entry `C` wins (`d < best_distance`): old server's benefit drops by its gap; `fallback ←`
   old server, `server ← C`, distances update; `benefits[C]` gains the new gap; if the old server was the
   normal, subtract `count` copies of the context from `connections`, decrement `served`, mark dirty.
2. For each entry where `C` is closer than the stored fallback but not the server: replace the fallback,
   adjust the server's benefit by the change in gap.
3. Any benefit strictly below its cost: `delete_child`, one at a time, re-checking after each.
4. If dirty: `refresh_normal()`. **This is the frame's only re-center for structure** — the deletes in step 3
   mark dirty and do not refresh, so the normal is recomputed once, after every handoff has landed.

**`refresh_normal()`** — recompute `normal = { n : 2·count(n) > served }`. If it changed: re-derive `server` /
`fallback` for the histogram entries (bounded by the horizon); apply any handoffs exactly as in `settle`
step 1 — records leaving the normal subtract counts, records arriving add them — and re-check touched
benefits. A record changes hands only for a **strictly** smaller distance, ties keep the incumbent. **One
sweep, not a loop**: those handoffs move `connections` again, so the definition ends the frame one step behind
its own served set, and `route_and_record` re-derives the assignment when the context next recurs (algorithm.md,
"The bill's pass").

**The round trip.** `settle` ends the neuron's first pass, so the request leaves with every local decision
already made. The thalamus collects the frame's requests, batch-allocates a pattern neuron per request (level
and channel off the request, no connections), creates the objects so the level above can dispatch to them, and
dispatches `register_child` back to each requesting parent. This is the same shape as today's install path,
moved after the review. The same result carries the ids released by any deletes, so a frame touches the
allocator once, in one direction or both.

**`register_child(id)`** — on the returned identity: rebind the pending entry in `children` and `benefits` to
`id`. Nothing else moves — the table settled before the request went out. **The newborn does not fire this
frame**: it is a routing-table entry that first serves, bids, and subsumes on the next frame its context
recurs.

**`delete_child(X)`** — remove from `children` and `benefits`, and report `X` on the pass's result so the
thalamus can release the pattern neuron and scrub its cross-neuron references. Unlike a mint, a delete needs no
round trip: the neuron already holds everything the reassignment requires, and the released id is only of
interest outside. For each entry served by `X`: `server ← fallback`, `best_distance ← fallback_distance`, recompute a
fresh fallback against the survivors; entries landing on the normal add their counts (dirty). For each entry
whose *fallback* was `X`: recompute the fallback, adjust the server's benefit. Re-check any benefit the
reassignments moved and cascade sequentially. **It does not refresh the normal** — it only marks dirty, and
`settle` step 4 does that once for the whole frame.

**Wall-clock shape:** no test ever scans the history — the add test is one pass over the histogram and only
runs on a normal-served error, rare for a settled neuron; `settle` and deletion are bounded by the
histogram; the running benefits make every delete decision O(1) per event. What *does* scale with the
dictionary is routing itself: a novel context is one distance against the normal and every child, so the
per-frame cost grows with the child count. The one test is what bounds that count — only children that pay
for their storage survive — but how large it gets in practice is an open question (algorithm.md).

## The build plan

Each phase lands independently and is measured before the next. The build order follows the axes: spatial
event processing first, then temporal event processing, then actions and rewards. The headline metric through
the event stages is the objective itself — **apex neurons per level per frame, paired with the dictionary
size that bought them** — tracked **as a function of exposure**: on recurring data both curves should fall
and flatten. Alongside: churn (creates + deletes per thousand frames, which should decay as dictionaries
settle), task accuracy (train and held-out), neuron counts per level, and wall-clock per frame.

### Stage 1 — spatial event processing

**Phase 1 — substrate, evidence, and the dictionary lifecycle.** Two halves that could have landed separately
but do not, because the first has nothing to measure on its own: a brain that records evidence and builds no
structure produces no exposure curves, and the accuracy it reports is the readout's, not the algorithm's.

- *Substrate and evidence.* Sparse activation (a dimension with nothing happening supplies no symbol; no neuron
  is emitted for it); the FIFO history with the histogram, stored fallbacks, and running benefits; routing and
  serving; unconditional recording.
- *The lifecycle.* The error-triggered add test, `settle`, and event-driven deletes. This is the heart.

The evidence half is verified rather than measured: the four invariants under "What must always hold" become
`debug_assertions` that recompute from the raw history and compare against the incrementally-maintained state
on every mutation, so a divergence trips at the frame that caused it instead of showing up as a bad curve
later. Gate the phase on the exposure curves: dictionary size sublinear in exposures, apex per frame falling,
churn decaying. Also measure history memory and per-frame wall-clock. Cap at one level so the recursion is not
a variable yet.

**Phase 2 — contraction.** The bids, the election, the neighbor filter rule, and the level-above
construction, adapted to the sparse substrate (only active neurons need cover). Measured by the per-level
reduction factor actually achieved and the depth at which it settles.

**Phase 3 — the readout gate.** Compare held-out accuracy against what the level counts justify. Do not build
past this phase if the answer is no.

### Stage 2 — temporal event processing

The same neuron mechanism at `d > 0`. The recognition context and the inference connections come apart into
two statistic tables per entry, records complete one frame late, and the election stays recognition-only —
the frame order is already built for this (algorithm.md, "The frame, step by step"). Gate: held-out
prediction on sequence data, and the same exposure curves on the temporal dictionaries.

### Stage 3 — actions and rewards

Action dimensions in the channels; default actions as the bootstrap; reward learned on the apex active action
([global-rewards.md](global-rewards.md)); exploration when learned rewards are negative; top-down unfolding
of a selected higher action into its constituent actions over the coming frames. Gate: a closed-loop
environment in which a learned action sequence answers a learned event sequence.

Variable-length pricing lands on its own track, specified in [forgetting.md](forgetting.md).

## Changes required in the current code (not yet implemented)

The current implementation (`thalamus.rs` / `neuron.rs` on this branch) predates most of the above. Each delta
is tagged with the phase it lands in.

**Phase 1 — the substrate and the evidence.**

1. **Sparse emission is an encoder change only.** `build_frame` iterates whatever the inputs map holds, so
   omitting a dimension — or a whole channel — already works with no change in the brain. On MNIST it is the
   encoder skipping off pixels.
2. **Rebuild the history.** `SpatialHistory` becomes a FIFO ring of context refs plus a histogram carrying
   `count`, `server`, `best_distance`, `fallback`, `fallback_distance`. The per-config `frames: Vec<FrameNumber>`
   and the absolute-frame `age_spatial_history` cutoff both go: capacity is the horizon in the neuron's own
   activations, and eviction is one-out-one-in off the ring. `SpatialHistory::rebase` goes with the frame
   numbers.
3. **Store the fallback.** Routing returns the two smallest distances from one scan instead of the winner only,
   and writes both into the record. This is what every later item reads.
4. **Replace the delete scan with running benefits.** `spatial_delete_candidate` and
   `spatial_delete_candidate_uncached` are deleted outright — not ported. Delete becomes
   `benefits[child] < 1 + |definition|`, checked only where an event moved the benefit (eviction, a mint's
   handoffs), strictly below cost, all failing children sequentially with cascade. Today's version scans the histogram on
   every active neuron every frame, deletes at most one, and deletes at equality; all three change.
5. **Switch the normal to element-wise majority.** `spatial_normal_config`'s per-channel argmax becomes
   `{ n : 2·count(n) > served }`, which needs a new `served` counter. `spatial_target_channels` and the channel
   plumbing that feeds it go — the normal is channel-free.
6. **Add `settle`.** New. `reassign_after_mint` is its step 1 only and is subsumed by it.
7. **Remove the new-child bid path.** Delete the `NEW_CHILD_BID` sentinel (`neuron.rs`) and every branch keyed
   on it: the request construction in `process_spatial_frame`, the split into `recognized` /
   `new_child_parents` in `process_spatial_level`, and the mid-frame create-install-activate block. New
   children stop competing in `elect_spatial_bids` entirely.
8. **Move minting after the review, as a request and a reply.** The add test runs for normal-served neurons
   with a nonzero error, after the record is written — no election input of any kind. It settles the table
   against a pending child and returns a request; the thalamus batch-allocates and dispatches `register_child`
   back; the parent rebinds the pending entry to the returned id. The
   existing install path (`allocate_spatial_pattern_neuron`, `install_spatial_corrections`) is the right shape
   already and moves after the review. The newborn is **not** activated: it fires first on its next
   recognition.
9. **Delete the birth special cases** — they exist only to patch mid-frame minting: the newborn's insertion
   into `new_error_pattern_ids` for the level above, the no-subsume-on-birth-frame rule, and the
   fires-but-does-not-record state.
10. **Record unconditionally.** Every active neuron commits its frame, as routed: winners, losing recognizers
    (server = the child, at its routed distance), normal-serves, covered or not. Today a losing bid commits
    nothing and the frame vanishes — that goes.
11. **Delete the evidence coupling to the election.** Remove `prune_inhibited_spatial_history` /
    `drop_inhibited_spatial_frame` and the subsumed-set plumbing from the evidence path entirely. The subsumed
    set survives only where it belongs: deciding what the level above (and the apex handoff) sees. The
    election writes nothing into any neuron. With nothing left to commit conditionally, the whole
    decide-then-commit round trip goes too: `SpatialCommitOp`, `commit_spatial_frame`, and the
    thalamus/region/column `commit_spatial_frames` chain.
12. **Remove the completeness gate.** `observed.len() >= spatial_capacity` in `process_spatial_frame` has no
    counterpart in the design, and under sparse activation it would rarely pass. `spatial_capacity` and its
    plumbing go with it.
13. **Drop the spatial recency machinery.** `activation_strength`, `last_activation_frame`, the lazy decay,
    death frames, `forget_rate`, and `can_delete_child` describe nothing in the design on the spatial side —
    children live and die by the one test, not by activation recency.
14. **Backups break.** The persisted history changes shape (ring, fallbacks, `served`), so old backups are
    unloadable. Accepted: no migration path, bump the format and fail loudly on an old file.
15. **The invariants become debug assertions.** The four statements under "What must always hold", each
    recomputed from the raw history and compared against the incrementally-maintained state.

**Phase 2 — contraction.**

16. **Price false positives in bids.** `covered` stays correct-names-only; carry `f = |definition \ observed|`
    on the bid and change the survival test in `spatial_survivors` from a flat `≥ 2` to `k ≥ 2 + f`. This is a
    known gap between the doc and the code.
17. **Replace the above-base adjacency heuristics with the filter rule.** The eight-sector nearest rule
    (`directional_neighbors`) and the reuse-the-last-declared-set fallback both go: while a neighbor filter is
    declared for a level, it applies; above the declared levels, the neighborhood is the level's active set.
    `channel_positions` and `set_channel_position` are only there to serve the sector rule and go with it;
    `set_spatial_neighbor_levels` stays — the declared per-level sets *are* the filter.

**Phase 3 — the readout gate.**

18. **Watch for the one expected regression:** action wiring reaches a new child one recurrence later than
    today, since newborns no longer fire on their birth frame.

## The MNIST frame protocol

MNIST runs on R35's chain — infer, execute, reward — one example per three frames. Nothing in
[algorithm.md](algorithm.md) changes for it. A base event neuron's reach in time is 1 (D15), so its
observation spans the frame before and the frame after its own.

```
frame     carries              what happens
-----     -------              ------------
f         events only          base event neurons fire, bet and bid (§9.1);
                               contraction runs; the assertion resolves (§12),
                               committing the digit-call action for f + 1
f + 1     the action only      the digit call executes and its neuron fires;
                               events are silent. Every neuron open here
                               records a connection to what ran, at the
                               distance of its own age (R35a)
f + 2     the reward only      the label arrives as input, not as a symbol
                               (§13). Nothing fires. It folds into the
                               connection's running mean (R35a)
f + 3     next example         = the next example's f
```

**The image is presented once, at `f`.** Frames `f + 1` and `f + 2` carry no event symbols at all. Every event
dimension is silent in them (D5).

### What follows from that

**Base event processing is spatial.** The backward slot at `−1` lands on the previous example's reward frame
and the forward slot at `+1` on the action frame; both are silent. So every neighbor a base event neuron names
sits at temporal offset `0`. The temporal slots are voted out for want of a majority (R4, R11) and cost nothing
in `|e|`.

**No action patterns form.** An action neuron's own backward and forward slots land on frames carrying no
actions, so the action hierarchy stays flat. R36's apex active action is therefore always the base action,
which R36 states explicitly holds before any action pattern exists.

**Connections are recorded per frame, never at the bill.** R35a records one at every age a neuron is open at,
and the reward folds in a frame later. Neither is gated on the observation completing, so the reward path does
not wait on the window and does not vary with level. Do not couple connection recording to §9.3.

**Classification is selection, not a readout.** The digit call is an action chosen by R38 off the event→action
connections, scored by the reward at `f + 2`. R39 supplies exploration while a situation's reward is negative.
The naive-Bayes consensus and the per-dimension vote in the current code have no counterpart in
[algorithm.md](algorithm.md) and are retired, not ported.

### Encoder changes

- Emit the image on one frame only; emit nothing on the action and reward frames.
- Sparse emission: an off pixel supplies no symbol (D5). No neuron is emitted for it.
- Declare the digit calls as an action dimension, in a fixed order — R39 walks that order for exploration, so
  it is part of the problem statement.
