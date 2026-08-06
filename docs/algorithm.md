# Universal Compression with Actions and Rewards (UCAR)

UCAR is the design for the full pattern lifecycle: recognition of existing patterns, creation of new ones, and
their removal when they stop paying. The substrate is an online compression engine whose dictionary entries are
situations — "universal" in the sense that nothing in it assumes a particular source.

The structural unit is the **neighborhood**. A neuron looks at its whole neighborhood, and a recurring
neighborhood that pays for itself becomes a pattern one level up. Resolution shrinks going up by
**contraction**: the thalamus keeps only the fewest units that still reconstruct the level below.

Every structural decision a neuron makes is [one test](#the-one-test), evaluated exactly against the
[frames it remembers](#the-history) — and structure is only ever reconsidered in response to an **error**.
A neuron whose situations are all explained builds nothing and deletes nothing.

This document specifies the spatial axis (`d = 0`, same frame). The event axis (`d > 0`) and action axis
(`d < 0`) are [open ports](#temporal-the-open-port).

## The substrate

The encoder declares **channels**; each channel declares **dimensions**; each dimension declares a resolution
(its bucket count) and a **rest state** — the value that means "nothing happening": 0 for a binary pixel, the
no-change bucket for a return, baseline for a sensor. Every frame, every dimension quantizes its input, and:

> **A neuron fires only on departure from rest. At most one neuron is active per dimension per frame; a
> dimension at rest is silent.**

Rest is not a signal to be stored — it is what the decoder assumes for anything the file does not state. This
buys three things at once:

- **Blank regions cost nothing.** No activations, no history, no dictionary, no compute — a neuron that is not
  active does not process the frame at all.
- **The neighborhood is a set.** Variable-size, containing only what is actually happening, which is what the
  [fit](#the-fit) needs and nothing more.
- **Absence still discriminates.** A pattern naming a ring of pixels is measurably wrong on a filled disk: the
  active neurons inside the ring are present-but-unnamed and the distance counts them. Offness carries
  information through the distance without ever costing storage.

A dimension with no meaningful rest state simply stays dense; nothing below depends on sparsity — it only
profits from it.

A neuron's coordinate is `(dim_id, bucket_id)`; its channel is the channel owning that dimension.

**Identity is absolute.** A neuron is bound to its position, so the same shape appearing at two positions is two
different neighborhoods over two different neurons, learned independently. Nothing shares a shape across
positions the way a convolution shares a filter — a retinotopic layout cannot, and the design does not try to.
Each position pays for its own patterns out of its own [history](#the-history), so the redundancy is bounded by
what actually recurs there. Sharing structure across contexts is a concern of the event axis, not this one.

The encoder also declares which channels are neighbors of which. This declaration is for the base level only;
higher levels derive their own adjacency ([contraction](#contraction-building-the-level-above)). Receptive
fields grow with depth through composition. This is the only place topology enters the design.

## The unit: a neuron and its neighborhood

A neuron's **neighborhood** is the set of active neurons in its declared neighbor channels. A spatial pattern is
a **named neighborhood**: the neuron looks around and reports what it sees upward.

**On the spatial axis, neighborhood, context, and inferences are one set.** The neurons a neuron infers are
exactly the neurons it uses as context, which are exactly its neighbors — there is no separate target set and no
separate conditioning set. This document says **neighborhood** throughout. (On the event axis context and
inference come apart — see [the open port](#temporal-the-open-port).)

**Channels gate eligibility, nothing else.** Two children of the same neuron may name overlapping but different
channel sets — `{A, B, C}` and `{B, C, D}` are just two sets. No mechanism below ever needs to know which
channel a neuron belongs to; the declared neighbor graph decides who *can* appear in a neighborhood, and the
[fit](#the-fit) does the rest without channel structure.

**One neighborhood per neuron per frame**, and **at most one child fires per neuron per frame**. If a neuron
could activate several children, each channel would carry several active units, those would all become neighbors
one level up, and the count would square at every level.

## The fit

A stored configuration names a set; the neuron observed a set. The distance between them is the symmetric
difference:

```
d(O, C) = |O △ C|   =   neurons present that C does not name
                      + neurons C names that are not present
```

This is not a separate notion invented for matching — it is literally the corrections that would follow this
activation in the file. The service cost and the match distance are the same number. It is also channel-free:
it never asks which channel a neuron belongs to, which is what makes variable-channel children unproblematic.

## The base model: what a neuron expects

Before any pattern exists, a neuron already has a cheap expectation about its neighborhood, and it gets it by
counting. The counters are its **connections**, and they are governed by one rule that must always hold:

> **The connection counts are the sufficient statistics of exactly the frames the normal currently serves,
> within the history.**

Every event that changes *which frames the normal serves* moves counts, in both directions: the normal serves a
frame → its neighborhood's counts increment; a normal-served record leaves the history → decrement; a new child
captures remembered frames from the normal → decrement; a deleted child's frames fall back onto the normal →
increment. Nothing else touches the counts. **A neuron updates its connections only on frames where no child
fired** — a neuron either infers from its own connections or delegates to a child, so every frame is explained
by exactly one of them and neither learns from the other's frames.

The neuron infers with the counts as the distribution they are — per neighbor, how often it was present on the
frames the normal served — and never collapses them into a single joint expectation for *inference*. What the
counters cannot do is hold a joint at all: they record that the left neighbor is usually on and the right
neighbor is usually on, but never that they are on *together*. That gap is what patterns exist to fill:

```
creation  → connections → child patterns
```

Connections are not free — they are part of the neuron's line in the dictionary and are counted in its
[cost](#the-cost-it-is-all-one-file). What separates them from patterns is that their count is capped at the
neighbor set's size and stops growing, while a neuron's children have no such ceiling.

**Cold start is silence.** A neuron with no neighbors seen yet has no connections and infers nothing. That is
the initial case, not an error.

### The normal

The **normal** is the connection counts resolved to a single configuration — the element-wise majority:

```
normal = { n : 2 · count(n) > served }
```

where `served` is the number of normal-served records in the history. A neuron is in the normal iff it was
present in more than half of the frames the normal serves. This is the storage-free entry: it has no pattern
neuron, it never propagates, and it is never deleted — its storage is the connections, which are paid for
already. It competes for frames like any other entry and appears in the history like any other server. Before a
neuron has any child, it is the only candidate.

The normal is **not** a frozen first observation. It is recomputed as the counts move — and it *sharpens as
children are created*: every child that takes a cluster of frames away subtracts that cluster's counts, so the
normal converges on the genuinely typical neighborhood instead of a blur of everything the children have not
yet claimed. A moved normal changes distances, so remembered frames may change hands between it and the
children — [settlement](#settlement) handles this exactly.

A neuron whose world is genuinely stable serves from its normal forever and creates nothing.

## The history

A neuron remembers a fixed number of frames it was active for — its **horizon**, measured in its own
activations, not in absolute time. **There are no frame numbers anywhere in the history and nothing needs
them**: benefit is a sum over records, cost is a count of stored neurons, and the only role time ever played
was eviction order, which a FIFO provides by construction. This also makes the one test uniform across neurons
under sparse activation: a busy neuron and a rarely-active neuron judge over the same amount of *evidence*,
not the same amount of someone else's clock.

Each record is:

```
(O, server, best_distance, fallback, fallback_distance)
```

— the neighborhood observed, the entry that served it, that entry's distance, the runner-up entry, and the
runner-up's distance. Records with the same neighborhood have identical distances to every entry, so the
storage is a **histogram** — distinct neighborhoods with a count and one shared assignment — plus a FIFO ring
of references in arrival order for eviction.

**Aging is one-out-one-in.** Once the ring is full, recording this frame evicts exactly the oldest record —
"age before add" is the data structure, not a rule to remember. Eviction has consequences
([deletion](#deletion)). A neuron's evidence is exactly the frames it was active for — recording is
unconditional, and no election outcome ever edits a history ([the frame](#the-frame-step-by-step)).

**Why the fallback is stored.** The delete test asks what an entry's frames would cost without it, and the
answer is the fallback. Storing it (and its distance) is what lets every entry's benefit be maintained as a
**running benefit**, updated only by events — so no test ever scans.

**It has to be the frames, not a summary.** A running error total cannot answer what deletion asks: when an
entry goes, its frames fall to their fallbacks, and the *new* runner-up for those frames must be recomputed
from the frames themselves. Keeping them is what makes every decision exact rather than estimated.

**The trade this makes, stated honestly:** wall-clock staleness is unbounded. A neuron dormant for a million
frames wakes with its old model fully intact. That is the intended bet — absence of evidence is not evidence
the structure died, and if the world did change while the neuron slept, the errors start arriving immediately
and restructuring proceeds at the pace of new evidence, which is the only pace a neuron can honestly claim to
know anything at.

The horizon — the history's capacity — is the design's **only free parameter**.

## The routing table is a facility location problem

**The plain problem.** Customers are spread over a map. You may open warehouses. Opening one costs a fixed
amount; serving a customer costs in proportion to how far it is from the warehouse serving it. Minimize the
sum. Open too few and customers are served from far away; open too many and you pay to open them.

**The mapping is direct:**

| Facility location | Here |
|---|---|
| A customer, i.e. one unit of demand | One record in the [history](#the-history) |
| A facility | A child |
| Opening a facility | Creating a child, and the pattern neuron it mints one level up |
| Opening cost | `1 + |C|` — the neuron created, plus the configuration stored |
| Serving a customer from a facility | Describing that frame's neighborhood through that child |
| Service cost, i.e. distance | The [fit](#the-fit) — the neurons the child got wrong |
| The assignment of customers to facilities | The routing — closest entry serves |

The opening cost is not a tax bolted on — if opening were free you would put a warehouse on top of every
customer: perfect service, zero distance, one facility per frame ever seen. It is the only thing standing
between the design and memorizing every frame.

The local search over this problem has exactly three moves, and each appears below as an exact test against
the history:

- **Add** — open a child at this frame's observation ([creating a child](#creating-a-child)).
- **Delete** — close a child whose benefit no longer covers its storage ([deletion](#deletion)).
- **Swap** — open one and close one, priced jointly ([the swap](#the-swap)). Without it, the search can lock:
  two mediocre children can jointly block the one good one, with no single add or delete able to fix it.

**Move**, the fourth classical operation, lives entirely on the normal: it is recomputed from its counts, so it
shifts the moment its demand does. Children never move — a child's configuration is frozen at mint, and
consolidation happens through swaps, not drift. **Split** and **merge** need no machinery of their own: split
is what add does to an overloaded entry's demand, merge is what swap does to redundant entries.

## The one test

There is exactly one structural rule in the design:

> **An entry earns its place when the errors it removes from the history exceed what it costs to store.**

Both halves are counts of neurons, and both are read off the history:

```
cost(e)     =  1 + |C(e)|
benefit(e)  =  Σ over the records e serves:  (fallback_distance − best_distance) · count
```

Benefit is maintained as a **running benefit** per child. It changes only on events — the child gains a
record, loses a record to eviction, or an entry appears, disappears, or moves nearby — so the delete test is
O(1) per event and no test ever scans the history.

To keep the words unambiguous: **benefit** is the sum above, **cost** is `1 + |C(e)|`, and **margin** always
means `benefit − cost`. The rule is strict on both sides: an entry is **added** only when its margin is
strictly positive, and **deleted** only when its margin is strictly negative. At exact equality nothing
happens — an existing entry is kept, a candidate is rejected — so the boundary can never flip-flop.

**Stability is a potential argument.** Every committed structural change — add, swap, delete — strictly
decreases the file length `L` over the horizon, by the strict-margin rule above. `L` is a non-negative
integer, so cascades terminate, nothing oscillates, and no entry can churn. This replaces any need for
thresholds, probation periods, or hysteresis.

## The frame, step by step

Two processes run over the same frame, and they are **independent**. The neuron's own pipeline — route, serve,
record, review, reconsider structure — reads and writes only the neuron's own state, and depends on nothing
the election decides. Contraction reads the level's recognition bids and decides only what is represented
above; it has **zero side effects on neuron state**. Creation still follows the review rather than
participating in recognition — that ordering is forced by the event axis, where recognition happens at frame
`f` but the error is only known at `f+1`, so a new child can never bid in the election that judged the frame
that created it.

For an active neuron with observed neighborhood `O`:

1. **Age.** If the history ring is full, evict the oldest record. Its histogram count decrements; if the normal
   served it, its neighborhood leaves the connection counts; if a child served it, that child's benefit drops —
   and a child whose benefit falls strictly below its cost is deleted now ([deletion](#deletion)).
2. **Route.** The closest entry serves — the normal and every child compete by `d(O, ·)`, ties to the older
   entry. Both the best and the runner-up are kept: **server** and **fallback**.
3. **Serve and bid.** A child serves by firing; the neuron hands its **recognition bid** to the thalamus — an
   offer to represent its patch at the level above. The normal serves by inferring from the connections; it
   makes no bid. **Only recognition bids exist** — creation never bids
   ([contraction](#contraction-building-the-level-above)). The bid is the pipeline's only output to the
   election, and the election sends nothing back.
4. **Record.** Every active neuron appends `(O, server, best_distance, fallback, fallback_distance)` — the
   server is whatever routing chose, promoted or not, covered or not. If the normal served, fold `O` into the
   connection counts. A neuron's evidence is the frames it was active for — full stop; no election outcome
   ever edits a history.
5. **Review.** The error of the inference — the served distance, the neurons the server got wrong. Spatially it
   is known immediately; on the event axis the actuals arrive next frame and the review — and everything after
   it — runs then.
6. **Add test and settle** — only when the normal served and the error is nonzero. The candidate is this
   frame's observation, exactly; if the solo test fails, price the swap. At most one add per frame. A passing
   test mints the child **now, for later**: it is installed in the routing table but fires for the first time
   on the next frame its neighborhood recurs — it never covers, propagates, or serves on its birth frame
   ([creating a child](#creating-a-child)). Then [settlement](#settlement) runs: reassignments, count
   subtraction, normal recomputation, benefit updates, and any deletes they trigger.

Meanwhile — before, after, or alongside; the pipeline cannot tell — contraction runs its election over the
level's bids and promotes the survivors to the level above. Subsumption is purely a statement about
representation: a covered neuron is spoken for *up there*, this frame; nothing about it changes *down here*.
Because no write ever depends on the election, there is no pending state, no retraction, and no
covered-or-not branch — the entire transaction protocol this document once needed is gone, not fixed.

## Creating a child

**The trigger is an error, nothing else.** No background process proposes candidates and no accumulator waits
to fire. The neuron served from its normal, inferred, and the inference was wrong — that error is the only
moment structure is considered, on the spatial axis in the same frame, on the event axis when the actuals
arrive. The error is the neuron's own — the served distance, read off its own routing; no election outcome
enters into it. A neighborhood the connections already predict costs nothing to serve no matter how often it
recurs; surprise is the size of the bill, not a gate on it.

**Creation happens after recognition, never inside it.** A new child is minted after the review and fires for
the first time on the next frame its neighborhood recurs. It does not serve, cover, or propagate on its birth
frame — that frame was already encoded, corrections and all. This is forced by axis uniformity (on the event
axis the error arrives a frame after the election it would have needed to bid in, so same-frame creation is
impossible there) and it is the founding criterion enforced at the finest grain: **structure never pays off on
the evidence that created it, only on recurrence.** Nothing compresses at first exposure. It also keeps the
election single-purpose — recognition bids only — with no birth-frame special cases: no newborn subsuming
neurons it hasn't earned, no newborn processed mid-cascade at its own level.

**The candidate is the observation.** `C = O`, exactly. Whenever that neighborhood recurs, the child serves it
at zero error — a child can never be created unable to serve the demand that justified it; it just serves it
starting next time. **A candidate rejected today is not lost.** Every future error offers a new candidate —
its own observation — so the neuron samples candidate positions from its own demand. A cluster worth a child
will sooner or later present a frame near its middle, and that candidate passes where an off-center one
failed; the [swap](#the-swap) then retires the off-center early mint.

**The solo test.** Would a child at `O` pay for its storage?

```
benefit  =  Σ over histogram entries e with d(e.O, O) < e.best_distance:
                (e.best_distance − d(e.O, O)) · e.count     // the records it would win
commit iff  benefit > 1 + |O|
```

There is no special term for the triggering frame: it was recorded in step 4, so it sits in the histogram as a
normal-served entry at distance `error`, and the candidate wins it at distance 0 like any other record. One
pass over the histogram — never more distinct neighborhoods than the horizon holds. Note the records it wins
may include frames currently served *badly by children*, not only by the normal: an add repairs whatever
demand it sits closest to.

### The swap

If the solo test fails, the candidate gets one more chance, jointly with a deletion. Take the child `X` whose
served records `C` would win the most of (the overlap is visible in the solo pass; in practice one or two
children qualify), and price the joint move `{add C, delete X}` exactly. Examining only the most-overlapped
child is a **heuristic**, not an exhaustive search over one-for-one swaps — in principle a barely-overlapping
child with a large storage refund could be the better joint move. Overlap is where the money almost always is,
and the exact variant (price the swap against every child) stays affordable if diagnostics ever justify it,
since the test only runs on a normal-served error:

```
Δ =   Σ over X's records:  cost served by X  −  cost served by best of (surviving entries + C)
   +  Σ over records NOT served by X that C wins:  (best_distance − d) · count
   +  (1 + |C(X)|)          // X's storage refunded
   −  (1 + |C|)             // C's storage charged
commit iff  Δ > 0
```

The two sums are over **disjoint** record sets — the first reprices everything `X` served, the second counts
`C`'s gains everywhere else — so no improvement is ever counted twice.

The normal is priced **frozen** at its current configuration. That is conservative: the recomputation
afterward only improves things, so a positive frozen score understates the realized gain and is always safe to
commit.

**Why the swap is necessary.** Two off-center children can each serve half of a cluster whose true center
would serve all of it: the center's solo add gains too little (each frame improves only slightly), and neither
incumbent fails its delete test while the other stands. Locked — forever, under add and delete alone. The swap
prices "center in, one incumbent out" as a single move; it pays, commits, and the second incumbent then fails
the ordinary delete test and follows. This is how early one-shot structure consolidates into fewer, better
entries as exposure accumulates.

**What the child is, at birth.** The pattern **inherits its parent's channel** and mints **one level above its
parent's level** — both are what the `1 + |O|` paid for. It is **created with no connections**: its connections
belong to its own level, which has not been observed at the moment of creation; it learns them there by
ordinary counting. Its configuration is **frozen at mint** — the normal is the entry that tracks the data;
children are consolidated by swaps, not by drifting. Its own structure — those connections, children of its
own — is judged by its own tests at its own level. A neuron's *existence* is decided by its parent; a neuron's
*structure* is decided by itself.

*(A measured optimization, deliberately not part of the design: if diagnostics ever show families of
near-duplicate children each serving a slice of one noisy cluster, a second candidate — the element-wise
majority of the frames `O` would win — can be priced alongside `O` for one extra counting pass. Ship without
it.)*

## Deletion

An entry dies the moment its benefit no longer covers its storage — and that moment is only ever caused by an
event, so **there is no delete scan**. Exactly two kinds of event can push a margin negative:

1. **Eviction** (frame step 1). Served records leave the history one at a time as demand drifts; the margin
   slides; when benefit falls strictly below cost, the child is starved and deleted. Without this, entries whose
   demand vanished would live forever.
2. **Settlement** (frame step 6). A committed add or swap either takes records from a child directly, or
   plants a closer fallback under records it keeps — **fallback collapse**: the child still serves its frames,
   but the error it spares shrank because the frames now have somewhere better to fall.

**Deleting a child, exactly:** its configuration leaves the routing table and the pattern neuron one level up
is released. Each record it *served* reassigns to its stored fallback (now its best), and a fresh fallback for
those records is computed against the surviving entries. Each record where the dead child was the *fallback*
keeps its server but recomputes its fallback, and that server's margin updates. Records landing on the normal
rejoin the connection counts, and the normal recomputes. **Deletions are sequential** — one at a time,
re-checking margins after each — because two children covering the same demand each look redundant while the
other stands. Every deletion strictly shortens the file, so cascades settle.

Nothing irreplaceable dies. The evidence lives in the history, not in the entry — if the same configuration is
justified again, the add test rebuilds it from the same frames.

## Settlement

What follows a committed add or swap, in order — each step exact against the history:

1. **The won records change hands.** For each histogram entry the new child `C` wins: the old server's margin
   drops by the gap it was earning; the old server becomes the entry's fallback (the previous best is by
   construction the new runner-up); `server = C`, distances update; `C`'s margin gains the new gap.
2. **Fallback collapse propagates.** For each entry `C` did *not* win but where `C` is closer than the stored
   fallback: the fallback is replaced and the server's margin adjusts by the change in the gap.
3. **The normal purifies.** Records captured from the normal leave the connection counts. The normal
   recomputes; if it moved, distances to it changed — server and fallback re-derive for the remembered
   records (the histogram is bounded by the horizon), and any handoffs move counts again.
4. **Margins that went negative trigger deletion**, sequentially, cascading as needed.

Every step strictly decreases `L`, so settlement terminates. Its work is bounded by the histogram size, which
is bounded by the horizon.

## The cost: it is all one file

Take the whole run and write it as a single file a decoder could read back to reproduce every frame exactly.
The file has two parts:

1. **The dictionary.** For each neuron, what it is: itself, the neurons its connections reach, and the
   configuration of each of its children.
2. **The frames.** For each frame, its apex neurons, followed by whatever those apex neurons got wrong about
   the level below them. Anything unstated is at rest.

Every cost in this design is a part of that file, counted in neurons — one symbol is one neuron:

```
activating an apex neuron  =  1                             a line in the frames
the errors it made         =  the neurons it got wrong      the corrections after it
having a child             =  1 + |configuration|           a line in the dictionary
```

This is a **fixed-length code**: every neuron costs one symbol regardless of how often it is used. Pricing
symbols by actual occurrence is a variable-length code over the same file, an experiment rather than the
design — see [forgetting.md](forgetting.md).

**The horizon is what makes the two parts comparable.** The dictionary is written once; the frames are written
over and over. Over one horizon — for each neuron, the activations its history holds — the file is

```
L  =  Σ over entries (1 + |configuration|)          written once
   +  Σ over remembered frames (1 + errors)         written every frame
```

and the one test is nothing more than the derivative of this: an entry belongs in `L` when removing it would
lengthen the file more than deleting its line would shorten it. **Nothing is amortized and nothing is
estimated** — the neuron holds the frames, so it evaluates the sum.

**Which file `L` prices — the convention, stated once.** `L` is the cost of re-encoding the **remembered
window under the current dictionary** — not the transcript that was actually emitted. It has to be: the delete
test reprices frames that went out the door long ago, under entries that did not exist when they were emitted.
Past emissions are sunk; the window is the pricing model. A structural move is judged by the window it leaves
behind, never by the frames it missed — so a child minted this frame is credited for the triggering frame
through that frame's *record* (reassigned to it in settlement), even though the emitted frame itself carried
the corrections.

**Where the connections sit in this accounting.** The connections' dictionary line is defined as **fixed**: one
counter per eligible neighbor, present even at zero — a preallocated vector in the pricing model, whatever the
code stores. (Counts *move* as frames change hands, and a count may sit at zero, but the line's length never
changes.) A line that exists whether or not the neuron has children, with a length no add, swap, or delete can
change, is the same on both sides of every comparison and drops out of the decision — which is why the sums
above price only children and frames. At higher levels the eligible alphabet grows over time, but that growth
is caused and priced by mints at the level below, not by this neuron's decisions, so the argument survives
there. "Storage-free" for the normal means exactly this: its storage is the connections' fixed line, already
paid for — not that it costs nothing in the file.

**A short horizon overfits.** With too few frames in view, a neuron builds structure for coincidences that
have not proven themselves beyond the window. There is no smoothing to hide behind, so sensitivity to the
horizon should be expected to be sharp, and measured early.

**What shrinking the file looks like.** The frame part dominates, because it is written every frame.
Shortening it means fewer apex neurons per frame: 784 names on a 28×28 input if nothing compressed, 200 if
level 1 covers the frame with 200 apex units, 50 if level 2 covers it with 50. But the frame part alone has a
degenerate optimum — memorize each frame as one top-level pattern — which the dictionary term prevents:
784 → 200 → 50 is a win only when the patterns that achieved it recur across many frames.

**Every decision is local.** Each neuron evaluates `L` over its own neighborhood only. Local evaluations gate
local decisions cheaply and in parallel; contraction stops the redundancy from multiplying up the levels. The
global quantity is measured directly rather than summed: **apex neurons per level per frame, and the
dictionary size that bought them.** That pair is the standing metric — and under this design it should *fall
with exposure* on recurring data: early structure is provisional, swaps consolidate it, the normal purifies,
and a single frame can never out-bid the opening cost on its own.

## Contraction: building the level above

A neuron firing a child is not yet compression. If 49 active neurons all fire children, level 1 has 49 active
units and nothing has shrunk. **Contraction is the thalamus choosing, each frame, the fewest units at the
level above that reconstruct the active neurons below.** It runs between levels during spatial processing:
level 0 is processed, contraction decides what propagates to level 1, level 1 is processed, and so on. Nothing
it decides persists — it is this frame's grouping, recomputed next frame.

Only **active** neurons need accounting for; a dimension at rest is unstated and costs nothing to leave
uncovered.

**Recognition bids only.** The election's input is one **recognition bid** per neuron whose routing chose a
child: the active neurons that child's configuration names correctly this frame, the bidder included. The
configuration names only the neighborhood; the bidder itself is implied, because a child *is* its parent in
that neighborhood — the entry lives in the parent's routing table, so expanding the unit recovers the parent
along with the neighbors it names. A neuron that served from its normal makes no bid; it can be covered by a
neighbor's bid but never propagates one of its own. **Creation never bids.** A child minted this frame (step 6)
does not exist for this frame's election — it first competes at its next recognition. This is what keeps the
two axes identical: on the event axis the error arrives one frame after the election it would have needed to
bid in, so same-frame creation bids are impossible there, and the spatial axis does not exploit the
coincidence that its inference resolves in-frame.

**The election decides promotion — and nothing else.** Routing decided who serves; the election decides what
is *represented above*, and it has **zero side effects on neuron state**. A child that matched but whose bid
loses still serves its frame in the history — the neuron simply is not represented at the level above and
stands as a correction at this one. A neuron covered by a neighbor's winning bid is **subsumed**: spoken for
at the level above this frame — and that is the whole of it. It still records its frame, learns, and runs its
tests exactly as if the election had gone the other way. Subsumption is a fact about the level above, never
about the neuron's evidence. (The cost of this independence, accepted deliberately: a neuron that is reliably
covered can still mint a child from its own local demand, and that child may rarely win elections yet stay
paid-for locally — see [risks](#risks).)

**A bid's price.** If the configuration names neurons that are **not** active this frame, the decoder
expanding the unit would assert them, and corrections must turn them off. So a bid's cost is `1 + f`, where
`f` is its named-but-absent count this frame — not a flat 1. Routing keeps `f` small (the closest entry
served), but it must be charged, or a large sloppy pattern looks artificially cheap.

**The objective.** The thalamus accepts a subset of the bids. Each accepted bid propagates one unit upward at
its cost `1 + f`. Every active neuron not covered by an accepted bid is a correction — stated as itself, at
cost 1. The thalamus minimizes `Σ accepted (1 + f) + corrections`. This is prize-collecting set cover, in the
same currency as everything else, restricted to one frame.

**The election, in detail.** Set cover is NP-hard, but contraction mints nothing that lasts — a grouping a few
units short of optimal costs a few extra symbols on one frame and is gone the next — so it is settled by a
cheap election:

- **Voters** are every active neuron named by at least one bid, including neurons that made no bid themselves.
  Voters are walked in sorted id order and bids in a fixed order, so the outcome is independent of dispatch
  order. Active neurons no bid names are corrections from the start and never vote.
- **The ranking rule**: among a pool of bids, the best is the one covering the most neurons this frame; ties
  go to the **older pattern id**. Ranking is a total, deterministic order, so no vote can ever flip-flop on a
  tie.
- **Rounds**: (1) compute the **survivors** — bids whose current voters outweigh their price, `k ≥ 2 + f`
  (covering `k` at cost `1 + f` changes the total by `1 + f − k`; for a clean bid this is the familiar "two
  or more" — the threshold is derived, not chosen). On the first pass nobody has voted, so there are no
  survivors yet. (2) Each voter looks at the bids naming it: if any survived, it elects the best *among the
  survivors* — herding onto coverage already paid for; if none survived, it elects the best among *all* of
  them again — which means a dropped bid can be **resurrected** when enough orphaned voters re-converge on
  it. (3) Repeat until a full pass changes no vote, with a round cap at the bid count as a backstop —
  resurrection makes the objective's descent non-strict in pathological cases, and a per-frame throwaway
  grouping slightly short of optimal is acceptable.
- **Outcome**: the final survivors are promoted, one unit each at the level above; their covered neurons are
  subsumed. Voters whose final pick did not survive, and actives covered by nothing, are the corrections.

The election runs sequentially in the thalamus; the instance is small — bids only interact within about two
grid hops — and settles in a handful of rounds. The older-id tiebreak does double duty: deterministic outcomes
(a recurring input yields a recurring group for the level above to latch onto), and steady pressure toward
established patterns over interchangeable new ones.

**What this builds.** Each surviving bid contributes one unit to the level above. Adjacency there is
**derived**: two units are neighbors iff any of their members were neighbors below. Reach is radius 1 by
construction — a configuration cannot name past its own neighborhood — and receptive fields grow by
composition. The reduction is set by the data, not the topology: a frame the patterns predict well collapses
hard; a frame full of surprise barely collapses at all, which is the honest thing for it to do.

## Neuron state

The complete per-neuron state on the spatial axis. Sets are sorted id lists; nothing stores a frame number.

```
id                                          // (dim, bucket) at base; opaque id above

// the dictionary line
children:     Map<child_id, config>         // config = sorted set of neuron ids; frozen at mint
connections:  Map<neuron_id, count>         // counts over the normal-served records in the history
served:       int                           // number of normal-served records (majority denominator)
normal:       config                        // cached majority set { n : 2·count(n) > served }

// the evidence
history:
  ring:       FIFO<config_ref>              // capacity = the horizon; arrival order = eviction order
  histogram:  Map<config, {
                 count,                     // how many remembered records are this neighborhood
                 server,                    // Normal | Child(id) — closest entry under the CURRENT entry set
                 best_distance,             // d(config, server's configuration)
                 fallback,                  // second-closest entry
                 fallback_distance }        // d(config, fallback's configuration)

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
  stored assignment instead of rescanning when the neighborhood is already remembered.
- `benefits[child]` always equals what a from-scratch recomputation would give:
  `Σ (fallback_distance − best_distance)·count` over the entries it serves. The delete test is then just
  "delete iff `benefits[child] < 1 + |children[child]|`" — strictly below cost, kept at equality — checked
  only when an event moved the benefit.
- Every committed structural change strictly decreases `L`. Since `L` is a non-negative whole number, cascades
  terminate and nothing can churn.

The normal has no margin: no storage line, never deleted.

On the event axis the same shape holds with two statistic tables per entry — context-side counts and
outcome-side counts — both moved together when a record changes hands, and records completed one frame late
([the open port](#temporal-the-open-port)).

## Neuron methods

In call order within a frame. All distances are the [fit](#the-fit); all sums run over the histogram, which
never holds more entries than the horizon.

**`evict_if_full()`** — frame step 1. If the ring is at capacity: pop the oldest ref; decrement its histogram
entry's count (drop the entry at zero). If its server was the normal: subtract the neighborhood from
`connections`, decrement `served`, mark the normal dirty. If a child: `benefits[child] -= (fallback_distance −
best_distance)`; if the benefit falls strictly below that child's cost, `delete_child(child)`.

**`route(O) → (server, best_d, fallback, fb_d)`** — frame step 2. If `O` is already in the histogram, read the
stored assignment (settlement and deletion keep it current). Otherwise scan the normal and every child for the
two smallest distances; ties to the older entry.

**`serve_and_bid(server)`** — frame step 3. Child: fire it and hand the thalamus a recognition bid — the
active neurons the child's configuration names correctly, bidder included, plus `f` = its named-but-absent
count. Normal: infer from `connections`, used as the per-neighbor distribution they are; no bid.

*(The election runs in the thalamus, concurrently as far as the neuron is concerned — see
[contraction](#contraction-building-the-level-above). It reads the bids and writes only the level above; none
of the methods below depend on its outcome.)*

**`record(O, server, best_d, fallback, fb_d)`** — frame step 4, every active neuron. Push the ref; upsert the
histogram entry — the server is routing's choice, promoted or not, covered or not. If the normal served: add
`O` to `connections`, increment `served`, mark dirty. Nothing recorded here is ever revoked — there is no
pending state and no retraction.

**`review(server, best_d) → error`** — frame step 5. Spatial: `error = best_d`, immediately. A child-served
error is recorded and priced (it is demand a future add can win) but does not run the add test — the
neighborhood was recognized, and the description job went up a level with the child.

**`add_test(O)`** — frame step 6, only when `server == Normal` and the error is nonzero. Solo benefit = `Σ`
over histogram entries strictly closer to `O` than to their current server of `(best_distance − d)·count` —
the triggering frame is already recorded, so its error enters through its own entry, no special term. Pass iff
`benefit > 1 + |O|`, else `swap_test(O)`.

**`swap_test(C)`** — for the child `X` most overlapped by `C`'s would-be wins (found during the solo pass; a
heuristic — see [the swap](#the-swap)): price `{add C, delete X}` jointly — reassign `X`'s records to the best
of the surviving entries plus `C`, sum the changes, add `C`'s wins over the records `X` did **not** serve (the
two sums are disjoint), refund `1 + |C(X)|`, charge `1 + |C|`; the normal frozen.

**`commit_add(C)`** — on a passing test: mint through the thalamus (pattern neuron one level up, parent's
channel, no connections); insert into `children`; then `settle(C)`. **The newborn does not fire this frame** —
it is a routing-table entry that first serves, bids, and subsumes on the next frame its neighborhood recurs.
For a swap, `delete_child(X)` runs in the same move.

**`settle(C)`** — frame step 6, after a commit:
1. For each histogram entry `C` wins (`d < best_distance`): old server's benefit drops by its gap; `fallback ←`
   old server, `server ← C`, distances update; `benefits[C]` gains the new gap; if the old server was the
   normal, subtract `count` copies of the neighborhood from `connections`, decrement `served`, mark dirty.
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
dictionary is routing itself: a novel neighborhood is one distance against the normal and every child, so the
per-frame cost grows with the child count. The one test is what bounds that count — only children that pay
for their storage survive — but how large it gets in practice is the open question below.

## Estimation

**No probability sets a cost.** Distance is a count of neurons, cost is a count of neurons, savings is a count
of neurons — the pricing never leaves whole numbers, so no estimator, smoothing, or boundary correction is
needed. The only frequencies in the design are the connection counts, and they are used raw: the majority rule
only has to pick members, not price them. A new child's configuration is the exact observation that created
it, so it serves that observation with zero error from its first frame.

Estimators return only in [forgetting.md](forgetting.md), where the file is re-priced by how often each neuron
occurs — the one variable-length code in which probabilities set costs.

## One frame, in order

```mermaid
flowchart TD
    A["Frame: active neurons = departures from rest"] --> B["Age: if the ring is full, evict the oldest record<br/>(counts, benefits update; starved child deletes)"]
    B --> C["Route: closest of normal + children serves;<br/>keep server AND fallback"]
    C --> D{"Is the server a child?"}
    D -->|yes| E["Fire it — hand the thalamus a recognition bid"]
    D -->|no| F["The normal infers from the connections; no bid"]
    E --> J["Record the frame (O, server, fallback, distances) —<br/>unconditional: no election outcome edits a history"]
    F --> J
    E -.->|"bid (the only output;<br/>nothing comes back)"| X["Contraction, independently: election over<br/>recognition bids; survivors are the level above.<br/>Promotion only — zero side effects on neuron state"]
    J --> K["Review: error = the neurons the server got wrong<br/>(same frame spatially; next frame on the event axis)"]
    K --> L{"Normal served AND error > 0?"}
    L -->|yes| M["Add test: candidate = this frame's observation.<br/>Solo benefit > 1+|O|? Else price the swap.<br/>A mint serves NEXT recurrence, never its birth frame.<br/>Settle: reassign wins, purify the normal, update benefits,<br/>delete anything that stopped paying — cascade until quiet"]
    L -->|no| N["Nothing to reconsider"]
    M --> O["Process the next level up"]
    N --> O
    X --> O
```

## Implementation plan

Each phase lands independently and is measured before the next. The headline metric is the objective itself —
**apex neurons per level per frame, paired with the dictionary size that bought them** — now tracked **as a
function of exposure**: on recurring data both curves should fall and flatten. Alongside: churn (creates +
deletes per thousand frames, which should decay as dictionaries settle), MNIST accuracy (train and held-out),
neuron counts per level, and wall-clock per frame.

**Phase 1 — substrate and evidence.** Sparse activation (rest states declared in the encoder; rest-bucket
neurons no longer emitted); the FIFO history with the histogram, stored fallbacks, and running benefits;
routing and serving; unconditional recording. No structural moves yet — verify the running benefits agree with a
brute-force recomputation on a fixed run, and measure history memory and per-frame wall-clock.

**Phase 2 — the dictionary lifecycle.** The error-triggered add test, the swap, event-driven deletes, and
settlement. This is the heart. Gate on the exposure curves: dictionary size sublinear in exposures, apex per
frame falling, churn decaying. Cap at one level so the recursion is not a variable yet.

**Phase 3 — contraction.** The bids, the election, derived adjacency, and the level-above construction,
adapted to the sparse substrate (only active neurons need cover). Measured by the per-level reduction factor
actually achieved and the depth at which it settles.

**Phase 4 — the readout gate.** Compare held-out accuracy against what the level counts justify. Do not build
past this phase if the answer is no.

Variable-length pricing lands on its own track, specified in [forgetting.md](forgetting.md).

### Changes required in the current code (not yet implemented)

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
6. **Price false positives in bids.** `covered` stays correct-names-only; carry `f = |config \ observed|` on
   the bid and change the survival test in `spatial_survivors` from a flat `≥ 2` to `k ≥ 2 + f`. This is a
   known gap between the doc and the code.
7. **Watch the MNIST gate for the one expected regression:** action wiring reaches a new child one recurrence
   later than today, since newborns no longer fire on their birth frame.

## Risks

- **The readout is unvalidated.** Compressing harder can produce a worse classifier, because a readout can be
  living on exactly the position-and-class-specific duplicates that compression deletes. Phase 4 is the gate.
- **Horizon sensitivity is sharp.** Every decision is exact with respect to the horizon and blind beyond it; there
  is no smoothing anywhere. Too small and neurons build structure for coincidences; too large and they are
  slow to follow a drifting source. Measure early.
- **Cold-start churn.** With a nearly-empty history, early tests are decided by very little evidence and
  children may be created and deleted in bursts. The error-only trigger and the strict-margin rule bound this
  — every commit still shortens the file that exists — but measure churn over the first thousand frames rather
  than settled counts.
- **Redundant, never-promoted children.** Evidence is independent of the election, so a neuron reliably
  covered by a neighbor's unit still mints children from its own local error demand — and those children may
  rarely win elections yet stay paid-for by their local benefit. Accepted deliberately: it is the same local
  double-witnessing the design accepts everywhere ("the same base event is witnessed by every neuron whose
  neighborhood contains it"), contraction keeps it out of the frame part, the standing metric (dictionary size
  against apex reduction) makes it visible, and the variable-length track in
  [forgetting.md](forgetting.md) is the principled reaper for symbols that exist but never earn use.
- **Child-served error mass is invisible to the add test.** The add test triggers only on normal-served
  errors, so a cluster captured by a child but served badly is repaired only when a nearby normal-served error
  offers a candidate that wins those frames, or when the child starves. Watch persistent child-served error in
  diagnostics; if it accumulates, the trigger may need to widen to any served error.
- **Medoid-only candidates.** Children are always observed frames, never synthesized centers, so incoherent
  per-frame noise on a recurring core can produce families of near-duplicate children. The swap consolidates
  them over exposure; if diagnostics still show the pattern, the one-step majority candidate (noted under
  [creating a child](#creating-a-child)) is the measured remedy.
- **Dormant staleness.** Neuron-relative time means a long-dormant neuron wakes with its old model intact. If
  the world changed meanwhile, it must first accumulate errors before restructuring. Accepted as the right
  default; worth remembering when reading diagnostics after distribution shifts.

## Open questions

- **Configuration space beyond the small case.** Eight binary neighbors is 256 possible neighborhoods — finite
  and countable, and sparsity shrinks it further in practice. Radius 2 and richer alphabets multiply it. The
  histogram bounds what is *remembered* at the horizon, but what bounds the number of *distinct* neighborhoods worth
  remembering when the space is large?
- **Configuration space at higher levels.** Above level 0 the neighbors are patterns, and the per-channel
  alphabet grows as patterns are created. At-most-one-active per neuron holds each channel to a single state,
  but the space still expands with the structure. What bounds the child count there?
- **Parallelism.** The per-neuron passes are independent across neurons and could run at once. Contraction's
  election is sequential and deliberately so — small enough to settle in a handful of rounds — but on much
  larger inputs than MNIST that judgement would need revisiting.

## Temporal: the open port

The event axis (`d > 0`) is not specified here. What is known about where it lands:

- **The mechanism is the same on both axes.** A neuron holds a normal and children, keeps a FIFO history of
  the frames it was active for, routes each frame to the closest entry, and creates or deletes children by the
  one test — error-triggered, swap included, margins maintained by events. Nothing in that needs the context
  and the inference to be the same set — it needs only a distance and a storage cost.
- **Context and inference come apart.** A temporal entry carries two statistic tables — context-side (what the
  situation looked like) and outcome-side (what followed) — and both move together when a record changes hands
  between entries. The record cannot be scored until the next frame arrives; it is completed one frame late.
  That is a bookkeeping delay, not a structural difference.
- **Its election runs at recognition time, over recognition bids only** — at frame `f`, when the context
  matches. The error exists only at `f+1`, one frame after the election it would have needed to bid in, so
  creation structurally cannot compete. This is the constraint the spatial frame order is copied from
  ([the frame](#the-frame-step-by-step)): recognition elects, creation follows the review, the newborn first
  competes at its next recognition — identical on both axes.
- **A temporal pattern infers its own level**, so at the moment of its creation its event connections are
  empty — the next frame has not been observed yet. It learns them once that frame arrives, the same rule as
  the spatial side.
- **One context can lead to several outcomes.** Spatially the demand point fully specifies the child, because
  the neighborhood is both the match and the prediction. Temporally the demand point is a (context, outcome)
  pair, and two frames sharing a context but differing in outcome cannot both be served by a context-keyed
  child. The economics degrades gracefully — the normal predicts the dominant outcome, eats the error on the
  rest, and the test creates children until whatever genuinely distinguishes the contexts is found. What
  remains after that is irreducible uncertainty, which no model removes. Whether the event axis needs more
  than this — a tiebreak among same-context children, or a child carrying several outcomes — is the open
  decision.
