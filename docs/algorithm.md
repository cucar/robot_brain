# Universal Compression with Actions and Rewards (UCAR)

UCAR is the design for the full pattern lifecycle: recognition of existing patterns, creation of new ones, and
their refinement after birth. The substrate is an online compression engine whose dictionary entries are
situations — "universal" in the sense that nothing in it assumes a particular source.

The structural unit is the **neighborhood configuration**. A neuron looks at its whole neighborhood, and a
recurring configuration that pays for itself is promoted to a pattern one level up. Resolution shrinks going up
by **contraction**: neighbors inhibit each other so only a spread-out subset propagates upward.

This document specifies the spatial axis (`d = 0`, same frame). The event axis (`d > 0`) and action axis (`d < 0`) 
are [open ports](#temporal-the-open-port).

## The substrate

The encoder declares **channels**; each channel declares **dimensions**; each dimension declares a resolution
(its bucket count). Every frame, every declared dimension quantizes its input to exactly one bucket, so:

> **Exactly one neuron is active per dimension per frame.**

A neuron's coordinate is `(dim_id, bucket_id)`; its channel is the channel owning that dimension. 
The "off" state of a binary pixel is its bucket-0 neuron firing, not an absence.

The encoder also declares which channels are neighbors of which. This declaration is for base level only. 
Neighborhood is detected dynamically in higher levels. Receptive fields grow with depth through composition. 
This is the only place topology enters the design.

## The unit: a neuron and its neighborhood

A neuron's **neighborhood** is the active neurons of its declared neighbor channels, and the assignment of values
it sees across them is its **configuration**. A spatial pattern is a **named configuration**: the neuron looks
around and reports what it sees upward.

**On the spatial axis, neighborhood, context, and inferences are one set.** The neurons a neuron infers are
exactly the neurons it uses as context, which are exactly its neighbors — there is no separate target set and no
separate conditioning set. Whether the inference comes from connections or from a fired child, it covers that one
set and nothing else. This document says **neighborhood** throughout.

**One configuration per neuron per frame.** The neighborhood is an assignment, so at any frame a neuron sees
exactly one configuration. At radius 1 with binary channels there are 2⁸ = 256 possible configurations — a
finite, small space.

The substrate's "exactly one active per dimension" has to survive going up, which requires **at most one active
unit per neuron per frame**. If a neuron could activate several children, each channel would carry several active
units, those would all become neighbors one level up, and the count would square at every level. [Recognition](#recognition)
is what makes this hold.

## The base model: what a neuron expects

Before any pattern exists, a neuron already has a cheap expectation about each neighbor, and it gets it by
counting.

Every frame the neuron is active, it notes which bucket each neighbor dimension is in and adds one to a counter.
Those counters are its **connections**. After enough frames they answer, for each neighbor dimension separately:
how often is that dimension in each of its buckets, on the frames where I am active?

A pixel neuron that is on might find that across the 100 frames where it was on, its left neighbor was on in 60
and off in 40. Its connections hold 60/40 — a distribution, not a verdict. The neuron infers with the whole
distribution and never collapses it to a single expected value:

```
p(D = b | P) = strength(P → D_b) / Σ_b' strength(P → D_b')
```

**There is no default configuration.** Combining the per-neighbor favorites into one expected neighborhood would
build a joint out of parts that cannot express a joint, and the result need not be anything the neuron has ever
seen — if the left neighbor is usually on and the right neighbor is usually on but never both at once, that
combination is a neighborhood that never occurs. Connections are used as the distribution they are.

**What makes it free is what it cannot do.** The counters are one number per neighbor bucket, so the base model
is bounded at `neighbor dimensions × buckets` and never grows with experience. And it stores each neighbor
separately: it records that the left neighbor is usually on and that the right neighbor is usually on, but never
that they are on *together*. It cannot represent a joint at all.

That gap is what patterns exist to fill, and it is why connections never compete with the dictionary — a pattern
is not a better version of the same thing, it is the only thing that can hold a configuration:

```
marginal  →  pairwise (connections, free)  →  configurations (patterns, priced)
```

**A neuron updates its connections only on frames where no child fired.** The two are exclusive: a neuron either
infers from its own connections or delegates to a child, so every frame is explained by exactly one of them and
neither learns from the other's frames. Connections stay the model of the typical neighborhood; patterns hold
what pairwise cannot reach.

**Cold start is silence.** A neuron with no neighbors seen yet has no connections and infers nothing. That is the
base case, not an error.

## What a neuron does with a frame

An active neuron does one of two things:

1. **The neighborhood matches an entry that has a pattern.** Fire it. Inference is delegated to that child, and
   nothing else happens — connections do not update and no evidence accumulates.
2. **Anything else.** The connections infer the neighborhood, the connections update, and evidence accumulates on
   the matched entry **sized by how badly the connections did**.

Both halves of the second branch happen every time; they are not alternatives. There is no test for whether a
frame was surprising enough to be worth collecting. **Surprise is the size of the deposit, not a gate on it.** A
neighborhood the connections already predict well contributes almost nothing no matter how often it recurs — not
because a threshold rejected it, but because there was nothing left to explain. A neighborhood they badly miss
contributes a lot.

That is also why connections learning from surprising frames is not a leak. Whatever pairwise can absorb, it
absorbs, and each frame it absorbs makes the same neighborhood cheaper next time. What never shrinks is the part
pairwise structurally cannot reach — the joint — and that residual is exactly what a pattern is for.

## Recognition

A neuron keeps **one routing table**. Each entry is a configuration the neuron has observed, together with the
statistics it has accumulated on that configuration. Some entries have a pattern neuron born from them; the rest
do not, yet. That is the only difference between them — **an embryo is a routing table entry whose pattern has
not been created yet.**

**A neuron matches its observed neighborhood against every entry, and the best match wins.** This is a match, not
an economic test. Matching is partial: an observed configuration need not equal a stored one exactly, which is
what lets minor variations of a shape be recognized as that shape. **It is one algorithm over the whole table**,
applied identically whether the entry it lands on is born or unborn. The exact rule is [open](#open-questions).

What happens next depends only on whether the winning entry has a pattern:

- **Born.** Fire it. Firing is **delegation**: the neuron hands the job of describing this neighborhood to the
  child rather than doing it from its own connections. **At most one child fires**, because a job is handed to one
  delegate and not split among several — which is what preserves one-active-per-channel all the way up. On a fire
  the entry's statistics strengthen; this is [refinement](#refinement).
- **Unborn.** The entry accumulates evidence, and the neuron infers from its own connections as usual.

If nothing matches, the observed configuration opens a new entry, unborn.

## The womb: the unborn entries

The **womb** is the unborn part of the routing table. It is not a separate structure and it does not have its own
matching rule — it is the entries that have accumulated evidence but do not have a pattern yet.

At radius 1 binary, 256 configurations are possible. Opening one entry per configuration would be an explosion,
but only a small subset both recurs and stays unexplained by the connections: for a neuron sitting on a "1",
vertical, horizontal and cross configurations recur far more than the rest.

**The womb holds configurations only.** It does not cluster connections.

**How few entries there are is decided by the match.** Two neighborhoods differing in one position should usually
land on the same entry, or the table degenerates into one entry per distinct configuration ever seen. This is the
same partial match [recognition](#recognition) uses — the tolerance that lets a born pattern fire on a variation
is the tolerance that keeps two variations from opening two entries. One rule, two consequences.

The shape of that rule is [facility location](#the-routing-table-is-a-facility-location-problem).

**Promotion.** When an entry's accumulated service cost covers a horizon's worth of its
[rent](#rent-what-an-entry-costs), a pattern is created for it, one at a time. The entry does not move: it stays
where it is and gains a child.

## The routing table is a facility location problem

**The plain problem.** Customers are spread over a map. You may open warehouses. Opening one costs a fixed
amount; serving a customer costs in proportion to how far it is from the warehouse serving it. Minimize the sum.
Open too few and customers are served from far away; open too many and you pay to open them. The optimum balances
the two. The half that decides which customer is served by which warehouse is the **demand routing** problem —
the two are always solved together, because you cannot price a warehouse without knowing what it will serve.

**The mapping is direct:**

| Facility location | Here |
|---|---|
| A customer, i.e. one unit of demand | One frame's observed neighborhood |
| A facility | A routing table entry |
| Opening a facility | Creating an entry, and eventually a pattern for it |
| Opening cost | The entry's [rent](#rent-what-an-entry-costs) over one horizon |
| Serving a customer from a facility | Describing this frame's neighborhood through that entry |
| Service cost, i.e. distance | What the mismatch costs — the positions the entry got wrong |
| The assignment of customers to facilities | The [match](#recognition) |

So **matching is the routing half and minting is the opening half**, and they are one problem. Every frame is a
customer arriving. It is served by the entry it matches, at a cost equal to how wrong that entry was. When the
demand piling up on a poorly-served region would have paid for a warehouse, a warehouse opens there.

**What the opening cost is here.** Serving a customer badly is easy to picture — the entry got some positions
wrong, and the mismatch costs something. Opening is less obvious, because nothing is physically built. What it
costs is **holding the entry**: recording the configuration once, and thereafter making it harder every frame to
say which entry served that frame. Charged per frame that is [rent](#rent-what-an-entry-costs); over a horizon it
is what a warehouse cost to build.

It is not a tax bolted on — without it the problem has a trivial and useless answer. If opening were free you
would open a warehouse on top of every customer: perfect service, zero distance, and one facility per frame ever
seen. The opening cost is the only thing standing between the design and memorizing every frame.

### Why this replaces a threshold

A threshold asks: *is this neighborhood within θ of that entry?* It needs θ, and θ is not derivable from
anything.

Facility location never asks that. It asks: **is it cheaper to serve this neighborhood from an entry that already
exists, or to open a new one?** That comparison has no free parameter — the opening cost is the price, which is
already defined, and the service cost is the mismatch, which is already measurable. A crowded neuron has a higher
price, so it tolerates worse matches rather than opening again. A sparse neuron opens more readily. The tolerance
falls out of the neuron's own circumstances instead of being set.

Distance is measured **in bits**, not in the count of differing positions: a mismatch in a position that is almost
always the same costs more than a mismatch in a position that varies anyway.

### Arriving one at a time

Textbook facility location is offline — all customers are known, then you solve. Here demand arrives one frame at
a time and there is no going back. That is **online** facility location, and the standard treatment is: when a
customer arrives far from every open facility, open a new one with probability proportional to distance over
opening cost.

The design uses the deterministic form of the same idea. Rather than opening with probability `distance / price`,
an entry **accumulates** the distance and opens when the accumulated total reaches one horizon's rent. That is
where the promotion rule comes from: **evidence is accumulated service cost** — demand that recurred in a place no
warehouse served well — and what it has to cover is what holding a warehouse there would cost over a horizon.

### Merge and split need no machinery of their own

The standard local search for facility location has three moves, and each one is already something the design
does for other reasons:

- **Close.** Shut a facility and reassign its demand. This is death: an entry whose service saving has fallen
  below its rent is released, and the frames it used to serve match elsewhere or open something new. Rent is
  precisely how this move is made affordable — the exact version would re-solve the whole assignment to ask
  whether closing helps, and rent asks the same question a little at a time, locally.
- **Move.** Relocate a facility toward the customers it actually serves. This is [refinement](#refinement): the
  stored configuration drifts toward the recurring core of the frames that match it.
- **Split.** Divide a facility serving two populations. This is **move plus open** — refinement pulls the existing
  entry toward whichever population dominates, the other population is then served badly, its service cost
  accumulates, and it opens an entry of its own. Merge is likewise **close plus reassignment**.

So neither merge nor split is a move the design has to add. They are what the ordinary operations compose into.

**This holds on one condition:** the match has to be the serve-or-open comparison, not nearest-entry-wins. If a
frame is always captured by its closest entry, then a badly-served frame is still *served*, its demand is
absorbed silently, and nothing ever accumulates to open the second entry — the split can never happen. The
comparison has to be able to answer "none of these, open a new one," and it has to answer it before an entry with
a pattern fires, since [firing delegates and accumulates nothing](#what-a-neuron-does-with-a-frame).

## Rent: what an entry costs

There is **one cost**, and it is a rate. An entry costs something every frame it exists, and that per-frame charge
is its **rent**. What was called an opening price is nothing separate: it is simply rent over one horizon.

### Why it is the same unit as service cost

Describing one frame, once a neuron holds entries, means saying two things: **which entry served this frame**, and
**what that entry got wrong about it**. Both are parts of one description of one frame, so both are counted the
same way — in things you have to name.

The second of those is the service cost. The first is what rent pays for, and it is genuinely per-frame: the
neuron identifies a serving entry on every frame, forever, and that identification gets harder the more entries
it holds. Nothing about it is amortized from a one-time event.

Only the write-down is one-time — recording the configuration itself happened once — and that is the single place
the horizon enters, as the rate at which the write-down is paid off.

So the neuron's bill for a frame is: rent for every entry standing, plus the mismatch on the entry that served it.
Three charges, one description, one unit. They add.

### What rent is made of

```
rent = name_bits(|configuration|, alphabet) / horizon        the write-down, spread out
     + log2(entries(P) + 1) - log2(entries(P))               this entry's share of making identification harder

name_bits(k, n) = log2( C(n, k) )  =  Σ_{i=0}^{k-1} log2( (n-i) / (k-i) )

alphabet   = max(neighbors ever witnessed, structural neighborhood cardinality)
entries(P) = how many entries this neuron already holds
```

The configuration is named as what it physically is — an unordered subset of the alphabet — rather than `k`
independent pointers, which would waste `log2(k!)` bits on order and overcharge dense configurations.

The alphabet is floored by the **structural** neighborhood cardinality. At cold start the witnessed set is tiny
and the first recurring configuration equals it exactly, which would name it for nearly nothing and open for free;
flooring charges it against the positions the decoder already knows exist.

The crowding term is written as a **marginal** cost — what this entry adds to the difficulty of identifying a
server — because identification is paid once per frame in total, not once per entry. A crowded neuron therefore
charges more for the next entry than a sparse one does.

### Open and close are one comparison

Because everything is a rate, the two structural decisions are the same test with opposite signs:

- **Open** an entry when the service cost it would remove per frame exceeds the rent it would add per frame.
- **Close** an entry when the service cost it removes per frame has fallen below the rent it costs per frame.

Nothing else is needed. Promotion's "evidence covers price" is this same comparison written over a horizon rather
than per frame — divide both sides by the horizon and it is rate against rate. [forgetting.md](forgetting.md)
specifies how the close side is measured online.

## Birth

A promoted entry gains a pattern neuron **one level above the neuron's own level**.

- The pattern **inherits its parent's channel**. That channel then infers its neighbor channels at the higher
  level.
- Its trigger is the entry's configuration, and the entry keeps accumulating on it — what was evidence before
  birth is [refinement](#refinement) after.
- **It is born with no connections.** Its connections belong to its own level, which has not been observed at
  the moment of birth. As the level above populates, the newborn learns its connections there by ordinary
  co-occurrence counting.
- It opens with the evidence that promoted it, which is what [forgetting.md](forgetting.md) charges against.

## Contraction: building the level above

A neuron firing a child is not yet compression. If all 49 neurons of a 7×7 grid fire a child, level 1 has 49
active units and nothing has been compressed. **Contraction is what makes the level above smaller than the level
below.**

Contraction runs **in the thalamus**, **dynamically between levels** during the level sweep: level 0 is
processed, contraction determines what propagates to level 1, level 1 is processed, and so on. It is this
frame's routing, recomputed each frame — patterns persist, owned by a channel; only which of them propagates
upward is decided here.

Each neuron carries an **activation strength**: how established the pattern it fired is. Contraction needs only
an ordering over neighbors, so what exactly that quantity is remains [open](#open-questions).

**The passes**, executed by the thalamus since these are cross-neuron comparisons:

1. **Survivors.** A neuron survives if its activation strength is the local maximum among its **immediate
   neighbors** — radius 1 only, never further. A survivor inhibits those neighbors.
2. **Coverage.** Among neurons that neither survived nor are adjacent to a survivor, take the local maxima of
   that remaining set; they survive too. Repeat until every neuron survives or touches a survivor.
3. **Groups.** Each inhibited neuron is absorbed by the adjacent survivor with the **highest activation
   strength**, ties broken by **neuron id**.

Nothing propagates. A survivor only ever inhibits its immediate neighbors, so a group is a survivor plus some of
its neighbors — **at most 9 cells**. The extra passes add survivors in uncovered areas; they never stretch an
existing group.

**The level above.** Each group contributes one unit. The adjacency at that level is **derived**: two units are
neighbors iff any of their members were neighbors below. The neighbor relation is declared once, at level 0, and
every level above computes its own.

**Termination is structural.** Because no two survivors can be adjacent, the survivor count is capped by the
maximum independent set of the grid — on 7×7 with 8-connectivity, every other row and column, i.e. 16:

```
49  →  ≤16  →  ≤4  →  1
```

The reduction is enforced by the topology, not by tuning and not by the data.

**Receptive fields grow by composition.** A level-1 unit covers its group; a level-2 unit covers a group of
groups. The declared radius stays 1 forever.

## Estimation

- **Model side: Krichevsky–Trofimov.** `p_c = (count + 1/2) / (n + 1)` — the minimax-regret universal estimator,
  derived rather than tuned. It keeps probabilities off the 0/1 boundary without an arbitrary cap.
- **Background side: raw MLE with the skip rule.** Background frequencies are plain counts over frames,
  unsmoothed. An outcome the background has never witnessed cannot be priced against it, so its term is skipped
  rather than smoothed into a finite surprise.

**The two must agree wherever they are compared.** Any test that admits a member to a pattern must use the same
estimate recognition will later apply to it, or a pattern can be born unable to fire on its own configuration.

## Refinement

On a fire, matched entries strengthen and absent entries dilute relatively as the trial count grows. This is an
online mode collapse: the stored configuration converges to the recurring core of the frames the pattern
actually fires on, so a shape whose boundary was drawn slightly wrong at birth converges to the right one
through ordinary fires. Low-share members are born low and fade, so no explicit prune rule is needed.

Connections carry no refinement — they are plain accumulated counts.

## One frame, in order

```mermaid
flowchart TD
    A[Frame: every dimension fires one bucket] --> B[For each active neuron at this level]
    B --> C[Observe the whole neighborhood configuration]
    C --> D{Best-matching routing table entry: does it have a pattern?}
    D -->|yes| E[Fire that ONE child: inference is delegated to it]
    D -->|no| F[Connections infer the neighborhood, and update]
    F --> H[Evidence accumulates on the entry,<br/>sized by how badly they did]
    H --> I{Evidence covers price?}
    I -->|yes| J[Promote: pattern one level up, inherits channel,<br/>born with no connections]
    E --> K[Contraction: thalamus runs the inhibition passes]
    F --> K
    K --> L[Each group contributes one unit to the level above<br/>adjacency derived from the grouping]
    L --> M[Process the next level up]
```

## The cost

Everything above is mechanism. This is what the mechanism is for.

**A frame is described by its apex neurons — the ones that are not inhibited.** What describing a frame costs is
the length of that list: 784 names on a 28×28 input if nothing compressed, 200 if level 1 covers the frame with
200 apex neurons, 50 if level 2 covers it with 50. **Shrinking that list, while still covering the frame, is the
goal.** Everything else in this document exists to make it shorter.

Shortening it is not free. A pattern is a stored list of members, and remembering that list costs about what
saying those members once costs — nine members remembered is about as expensive as nine names said. This is what
makes the two sides comparable: **both are counted in things you have to name**, one for each frame and one for
the dictionary that describes frames. They subtract directly.

So a pattern naming nine members costs about nine, once, and saves about eight every frame it fires.

Take the frame list alone as the objective and it has a degenerate optimum: memorize each frame as a single
top-level pattern. One apex neuron, full coverage, the shortest possible list — and a dictionary holding one entry
per frame ever seen, which is longer than the input it replaced. The target 784 → 200 → 50 is a win only when the
patterns that achieved it are **reused across many frames** rather than minted per frame.

Every structural decision is one question: does the move shorten the total, dictionary included?
**No similarity thresholds exist anywhere in the design.**

### The ledgers are local

Each neuron keeps its own ledger over its own neighborhood, so the sum of local ledgers is not the total — the
same base event is witnessed by every neuron whose neighborhood contains it. This is accepted: local ledgers gate
local decisions cheaply and in parallel. Contraction is what stops that local redundancy from multiplying up the
levels.

The global quantity is measured directly instead of summed: **apex neurons per level per frame, and the
dictionary size that bought them.** That pair is the standing metric for whether the design is working, and the
local ledgers are only a cheap parallel proxy for it.

### Everything is local

Both decisions a neuron makes — open an entry, close an entry — are settled entirely from what that neuron holds.
[Rent](#rent-what-an-entry-costs) scales with the neuron's own entry count, never with the size of the brain, and
service cost is measured against its own neighborhood. Nothing consults a global total.

A naive neuron therefore opens cheaply, and that is its exploration budget rather than a leak: an entry that stops
earning its rent is closed by the same local comparison that opened it. What bounds the system is not a global
account but the fact that rent rises as a neuron fills and closes happen wherever demand moves away.

## Implementation plan

Each phase lands independently and is measured before the next. The headline metric is the objective itself:
**survivors per level per frame, paired with the dictionary size that bought them.** Alongside it: MNIST accuracy
(train and held-out), neuron counts per level, patterns per neuron, and wall-clock per frame.

**Phase 1 — the configuration loop at level 0.** The whole-neighborhood configuration; inference from the
pairwise connection distributions; deposits sized by how badly those did; the per-neuron womb over configurations
with partial matching; promotion when evidence covers price; recognition as a configuration match with at most
one child per neuron. Capped at one level so the recursion is not a variable yet.

**Phase 2 — contraction.** The inhibition passes, groups, derived adjacency, and the level-above construction.
Measured by the per-level reduction factor and the depth at which it terminates; the prediction is
49 → ≤16 → ≤4 → 1.

**Phase 3 — the readout gate.** Once growth is bounded and depth settles, compare held-out accuracy against
what the level counts justify. Do not build past this phase if the answer is no.

Forgetting lands on its own track, specified and phased in [forgetting.md](forgetting.md).

## Risks

- **The readout is unvalidated.** Compressing harder can produce a *worse* classifier, because a Naive-Bayes
  readout can be living on exactly the position-and-class-specific duplicates that compression deletes. Phase 3
  is the gate.
- **Cold-start churn.** Tiny early alphabets mean near-zero prices, so a burst of cheap early patterns appears
  with nothing in this document to remove them. Measure churn in the first thousand frames, not just settled
  counts.
- **Dictionary redundancy.** Contraction shrinks the width of each level; it does not shrink the dictionary. The
  same shape is learned separately at every position.

## Open questions

- **The match rule.** That it is [facility location](#the-routing-table-is-a-facility-location-problem) with bits
  as the distance is settled; what exactly the distance is, and therefore what lands on the same entry versus
  opens a new one, is not. It decides whether the routing table stays bounded.
- **Ranking born against unborn.** One match runs over the whole table, so an unborn entry can outscore a born
  one — the neuron would decline to fire a child it has in order to accumulate on an embryo instead. Whether that
  is correct is undecided.
- **Configuration space beyond the small case.** Eight binary neighbors is 256 configurations — finite and
  countable. Radius 2 is 2²⁴, and more buckets multiply it further. Exact-configuration counting is bounded
  precisely in the regime where it is least needed. What replaces it when the neighborhood is larger?
- **Configuration space at higher levels.** Above level 0 the neighbors are patterns, and the per-channel
  alphabet grows as patterns are promoted, so the configuration space is not fixed. One-active-per-neuron holds
  each channel to a single state, but the space still expands with the structure. What bounds the womb there?
- **Reuse.** Identity is keyed by absolute dimension, so nothing shares a shape across positions. Contraction
  fixes the width of each level, not the redundancy of the dictionary.
- **Activation strength.** What is a neuron's strength when it fired no pattern at all, and do zero-strength
  neurons participate in the inhibition passes or survive by default?
- **Parallelism.** The inhibition passes are cross-neuron comparisons and run in the thalamus. Whether they can
  be parallelised, or pushed into neurons given a shipped view of neighbor strengths, is undetermined.

## Still to write

Sections known to be missing or known to be placeholders, in the order they should be settled:

1. **The level transition.** Contraction, grouping, reuse, and merging are one operation, run by the thalamus at
   the boundary between levels, and together they determine which neurons are active at the level above. §Contraction
   currently describes only the inhibition half. Two questions inside it: whether reuse is keyed on the group's
   shape rather than on any single neuron's configuration, and which unit earns the saving once a group collapses.
2. **§The cost, rewritten.** It is a placeholder standing on mechanism that item 1 has not settled yet. Once the
   level transition is written, the cost section should be rebuilt on top of it rather than patched.
3. **Terminology.** §Contraction says "survivor" and §The cost says "apex neuron" for the same set. Pick one.
4. **Dictionary redundancy.** Currently filed under Risks, describing the same shape learned separately at every
   position. That is item 1's subject, not a risk, and should move once item 1 exists.

## Temporal: the open port

The event axis (`d > 0`) is not specified here. Two things are known about where it lands:

- **The context is the active neurons at `d > 0`**, used to infer the next frame.
- **A temporal pattern infers its own level**, not the one below, so at the moment of its creation its event
  connections are empty — the next frame has not been observed yet. It learns them once that frame arrives, the
  same rule as the spatial side: a newborn is born with no connections and learns them at its own level.
