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
in the histogram, read the stored assignment (settlement and deletion keep it current); otherwise scan the
normal and every child for the two smallest distances, ties to the older entry. Then write the record from the
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

**`add_test(O)`** — frame step 6, only on a nonzero error. Solo benefit = `Σ` over histogram entries strictly
closer to `O` than to their current server of `(best_distance − d)·count` — the triggering frame is already
recorded, so its error enters through its own entry, no special term. Pass iff `benefit > 1 + |O|`, else
`swap_test(O)`.

**`swap_test(C)`** — for the child `X` most overlapped by `C`'s would-be wins (found during the solo pass; a
heuristic — see algorithm.md, "The swap"): price `{add C, delete X}` jointly — reassign `X`'s records to the
best of the surviving entries plus `C`, sum the changes, add `C`'s wins over the records `X` did **not** serve
(the two sums are disjoint), refund `1 + |C(X)|`, charge `1 + |C|`; the normal frozen.

**`commit_add(C)`** — on a passing test: mint through the thalamus (pattern neuron one level up, parent's
channel, no connections); insert into `children`; then `settle(C)`. **The newborn does not fire this frame** —
it is a routing-table entry that first serves, bids, and subsumes on the next frame its context recurs.
For a swap, `delete_child(X)` runs in the same move.

**`settle(C)`** — frame step 6, after a commit:
1. For each histogram entry `C` wins (`d < best_distance`): old server's benefit drops by its gap; `fallback ←`
   old server, `server ← C`, distances update; `benefits[C]` gains the new gap; if the old server was the
   normal, subtract `count` copies of the context from `connections`, decrement `served`, mark dirty.
2. For each entry where `C` is closer than the stored fallback but not the server: replace the fallback,
   adjust the server's benefit by the change in gap.
3. If dirty: `refresh_normal()`.
4. Any benefit strictly below its cost: `delete_child`, one at a time, re-checking after each.

**`refresh_normal()`** — recompute `normal = { n : 2·count(n) > served }`. If it changed: re-derive `server` /
`fallback` for the histogram entries (bounded by the horizon); apply any handoffs exactly as in `settle`
step 1 — records leaving the normal subtract counts, records arriving add them — and re-check touched
benefits. A record changes hands only for a **strictly** smaller distance — ties keep the incumbent — so every
handoff shortens the file by at least one symbol and the pass cannot cycle; when no strict improvement
remains, it stops.

**`delete_child(X)`** — remove from `children` and `benefits`; release the pattern neuron through the
thalamus. For each entry served by `X`: `server ← fallback`, `best_distance ← fallback_distance`, recompute a
fresh fallback against the survivors; entries landing on the normal add their counts (dirty). For each entry
whose *fallback* was `X`: recompute the fallback, adjust the server's benefit. `refresh_normal()` if dirty;
re-check any benefit the reassignments moved — cascade sequentially.

**Wall-clock shape:** no test ever scans the history — the add test is one pass over the histogram and only
runs on a normal-served error, rare for a settled neuron; settlement and deletion are bounded by the
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

**Phase 1 — substrate and evidence.** Sparse activation (a dimension with nothing happening supplies no
symbol; no neuron is emitted for it); the FIFO history with the histogram, stored fallbacks, and running
benefits; routing and serving; unconditional recording. No structural moves yet — verify the running benefits
agree with a brute-force recomputation on a fixed run, and measure history memory and per-frame wall-clock.

**Phase 2 — the dictionary lifecycle.** The error-triggered add test, the swap, event-driven deletes, and
settlement. This is the heart. Gate on the exposure curves: dictionary size sublinear in exposures, apex per
frame falling, churn decaying. Cap at one level so the recursion is not a variable yet.

**Phase 3 — contraction.** The bids, the election, the neighbor filter rule, and the level-above
construction, adapted to the sparse substrate (only active neurons need cover). Measured by the per-level
reduction factor actually achieved and the depth at which it settles.

**Phase 4 — the readout gate.** Compare held-out accuracy against what the level counts justify. Do not build
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

The current implementation (`thalamus.rs` / `neuron.rs` on this branch) predates the recognition-only
election. Bringing it in line is its own session; the deltas are:

1. **Remove the new-child bid path.** Delete the `NEW_CHILD_BID` sentinel (`neuron.rs`) and every branch keyed
   on it: the request construction in `process_spatial_frame`, the split into `recognized` /
   `new_child_parents` in `process_spatial_level`, and the mid-frame create-install-activate block. New
   children stop competing in `elect_spatial_bids` entirely.
2. **Move minting after the review.** The add test (and swap) runs for normal-served neurons with a nonzero
   error, after the frame's records commit — no election input of any kind. The mint allocates and installs
   the child but does **not** activate it: it fires first on its next recognition.
3. **Delete the birth special cases** — they exist only to patch mid-frame minting: the newborn's insertion
   into `new_error_pattern_ids` for the level above, the no-subsume-on-birth-frame rule, and the
   fires-but-does-not-record state.
4. **Record unconditionally.** Every active neuron commits its frame, as routed: winners, losing recognizers
   (server = the child, at its routed distance), normal-serves, covered or not. Today a losing bid commits
   nothing and the frame vanishes — that goes.
5. **Delete the evidence coupling to the election.** Remove `prune_inhibited_spatial_history` /
   `drop_inhibited_spatial_frame` and the subsumed-set plumbing from the evidence path entirely. The subsumed
   set survives only where it belongs: deciding what the level above (and the apex handoff) sees. The
   election writes nothing into any neuron.
6. **Price false positives in bids.** `covered` stays correct-names-only; carry `f = |definition \ observed|`
   on the bid and change the survival test in `spatial_survivors` from a flat `≥ 2` to `k ≥ 2 + f`. This is a
   known gap between the doc and the code.
7. **Watch the MNIST gate for the one expected regression:** action wiring reaches a new child one recurrence
   later than today, since newborns no longer fire on their birth frame.
8. **Replace the above-base adjacency heuristics with the filter rule.** The eight-sector nearest rule
   (`directional_neighbors`) and the reuse-the-last-declared-set fallback both go: while a neighbor filter is
   declared for a level, it applies; above the declared levels, the neighborhood is the level's active set.
