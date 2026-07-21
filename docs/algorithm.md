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

**What it cannot do.** The counters are one number per neighbor bucket, so the base model
is bounded at `neighbor dimensions × buckets` and never grows with experience. And it stores each neighbor
separately: it records that the left neighbor is usually on and that the right neighbor is usually on, but never
that they are on *together*. It cannot represent a joint at all.

That gap is what patterns exist to fill, and it is why connections never compete with the dictionary — a pattern
is not a better version of the same thing, it is the only thing that can hold a configuration:

```
marginal  →  pairwise (connections, bounded)  →  configurations (patterns, unbounded)
```

Connections are not free — they are part of the neuron's line in the dictionary and are counted in its
[cost](#the-cost). What separates them from patterns is that their count is capped at
`neighbor dimensions × buckets` and stops growing, while a routing table has no such ceiling. A neuron's rent
therefore rises with the patterns it opens, never with how long it has been counting.

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

**Promotion.** An entry becomes a pattern when its [balance](#the-balance-is-the-whole-lifecycle) crosses the
opening line at one horizon of rent. The entry does not move: it stays where it is and gains a child. How that
balance is banked, and how a badly-served entry opens a *new* one at the demand point, is the
[facility location](#the-routing-table-is-a-facility-location-problem) machinery below.

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
| Opening cost | The entry's rent over one horizon |
| Serving a customer from a facility | Describing this frame's neighborhood through that entry |
| Service cost, i.e. distance | What the mismatch costs — the positions the entry got wrong |
| The assignment of customers to facilities | The [match](#recognition) |

So **matching is the routing half and minting is the opening half**, and they are one problem. Every frame is a
customer arriving. It is served by the entry it matches, at a cost equal to how wrong that entry was. When the
demand piling up on a poorly-served region would have paid for a warehouse, a warehouse opens there.

**What the opening cost is here.** Serving a customer badly is easy to picture — the entry got some positions
wrong, and the mismatch costs something. Opening is less obvious, because nothing is physically built. What it
costs is **holding the entry**: recording the configuration once, and thereafter making it harder every frame to
say which entry served that frame. Charged per frame that is rent; over a horizon it
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

Distance is the **number of neurons the entry would get wrong** — the same count that appears in the file as the
corrections following an activation. Nothing weights one position above another.

### The distance

An entry names a configuration; the neuron observed one. The distance between them is the number of neurons the
entry would get wrong, counted in both directions:

```
d(O, e) = |O △ C(e)|     neurons present that the entry does not name
                       + neurons the entry names that are not present
```

This is Hamming distance over the neighborhood, and it is not a separate notion invented for matching — it is
literally the corrections that would follow this activation in the file. The service cost and the match distance
are the same number.

### The closest entry serves; errors accumulate

**The closest entry always serves.** There is no per-frame decision about whether to open instead, because a
single frame cannot answer that question. One frame's error is a real cost, incurred once. Rent is an opening cost
sliced into `horizon` pieces. Comparing them directly compares one real thing against a fraction of a hypothetical
one, and the real thing wins every time — which would open something on nearly every imperfect frame.

Comparing like with like means **accumulating**. The neuron carries one running total: the service cost it has
incurred, meaning every neuron gotten wrong by whatever did the inferring, summed over every frame. When that
total covers what a new pattern would cost at the current observation, opening has already paid for itself, and
one is created there.

An entry wrong by one neuron per frame, against an opening cost of ten, opens after ten such frames rather than on
the first. An entry that serves perfectly accumulates nothing and never triggers anything, however long it runs.

**Nearest-wins is safe under accumulation.** [Merge and split](#merge-and-split-need-no-machinery-of-their-own)
required that badly-served demand not be absorbed silently, or a split could never happen. Accumulation is what
satisfies that: the closest entry does serve, but what it got wrong is still counted, and it eventually forces an
opening.

**The new pattern is created at the current observation** — the demand point that triggered it. That observation
need not be representative of everything served badly; [refinement](#refinement) is what moves a pattern toward
the frames it actually serves, so an imperfect placement corrects itself through ordinary use.

### The frame, step by step

For an active neuron with observed neighborhood `O` and routing table `E`:

1. **Measure.** Compute `d(O, e)` for every entry, and take the closest as `e*`.
2. **Serve.** `e*` serves. If it has a pattern, fire it — inference is delegated. If it does not, the connections
   infer the neighborhood and update. With an empty table, the connections serve.
3. **Bank.** The serving entry's balance rises by what it spared: the members it covered, less one for naming it,
   less the neurons it got wrong.
4. **Accumulate.** `e*` — the entry that served — adds the neurons it got wrong this frame to its own error
   total.
5. **Open.** If `e*`'s error total covers `1 + |O|`, create a pattern whose configuration is `O`, and subtract
   that cost from `e*`'s total.
6. **Charge.** Every entry in the table drains its own rent, served or not.

Step 3 is where the numbers close. An entry covering nine neurons with nothing wrong spares `9 − 1 − 0 = 8`,
against a cost of `1 + 9 = 10` to hold, so it needs to serve about twice within a horizon to stay solvent.

The error total is kept **per entry**: the entry that served is the one whose inadequacy justifies a neighbor, so
it is the one that accumulates. This is the split move — a broad entry serving a spread-out population accumulates
its own error and spawns a sub-entry for the part it serves worst. An observation far from every entry makes the
nearest one accumulate a large chunk at once, so novelty opens a pattern quickly.

**The base model is the entry of last resort.** When the table is empty, or when the connections themselves do
the inferring, their errors accumulate the same way — that is what buys the first pattern before any entry exists.

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

## The cost

### One file, one per-frame ledger

Take the whole run — every frame the brain has seen — and write it as a single file a decoder could read back to
reproduce every frame exactly. The file has two parts:

1. **The dictionary.** For each pattern, the configuration it holds — which neighbors it names.
2. **The history.** For each frame, its apex neurons, followed by whatever those apex neurons got wrong about the
   level below them.

You could write the encoder and decoder, run them, and measure the result. Its length is one number, and **every
cost in this design is a part of it, counted in neurons** — one symbol is one neuron:

```
activating an apex neuron  =  1                               a line in the history
the errors it made         =  the neurons it got wrong        the corrections after it
having a pattern           =  1 + |connections| + Σ over its born entries |configuration|   a line in the dictionary
```

A pattern's dictionary line cannot be its id alone — an id reconstructs nothing. The decoder has to be told what
the neuron does, and what it does is infer its neighborhood two ways: through its connections, and through the
entries in its routing table. So its line is itself, the neurons its connections reach, and the members of each
born entry.

**The two parts look like different kinds of cost — one paid once, two paid every frame — but they are the same
unit.** Divide the whole file by the `H` frames it covers, and all three land in one per-frame ledger:

```
per-frame cost  =  Σ rent          having, spread: (pattern cost) / H, over every pattern held
                +  activations      1 per apex neuron this frame
                +  errors           neurons gotten wrong this frame
```

The dictionary does not sit outside the history; it **enters it as rent.** Rent is not a second kind of cost laid
over the design — it is the one-time having-cost seen from inside the per-frame ledger, which is the only place a
one-time cost and a recurring one can be compared. Every decision reads from this one ledger: serving pays
activation plus errors, holding pays rent, and a pattern earns its keep when the errors it removes per frame beat
the rent it adds.

**Cost is whatever the neuron currently is**, not what it was at birth: as a pattern grows its own connections and
opens its own entries, its dictionary line lengthens and its rent rises with it. This is also why an established
neuron demands more before adding structure — a longer line means higher rent means the next entry has more to
beat. There is no crowding term anywhere; there is just more structure to write down.

### The horizon is the one free choice

`H` is the only number not fixed by counting. Offline it is simply how long the file is. Online there is no known
end and the source drifts, so `H` stands for how long the world is assumed to hold still — the payback period a
pattern is bet to survive. Rent is how that bet is collected, one frame at a time. That is the single place the
horizon enters the design.

The bet can be lost, and losing it is not an accounting error. In the real file a pattern's dictionary line is
paid in full once, however briefly it lives; a pattern that dies after five frames still cost its whole line but
paid only five frames of rent. Being online means not knowing the future, so the design bets a horizon, charges
rent against the bet, and lets the balance delete what does not pay.

**Embryos are never in the file.** An entry with no pattern is the neuron's own bookkeeping — nothing about it is
transmitted and a decoder never learns it existed. It is charged rent all the same, against the pattern it is
proposing, so an unborn entry has to earn like a pattern before it becomes one. This is a **fixed-length code**:
every neuron costs one symbol regardless of how often it is used. Pricing symbols by actual occurrence is a
variable-length code over the same file, an experiment rather than the design — see [forgetting.md](forgetting.md).

### The balance is the whole lifecycle

Rent is charged **in the routing table**, and every entry carries exactly one number: its **balance**. This is the
same quantity [contraction](#contraction-building-the-level-above) reads as activation strength.

- It **rises** when the entry serves a frame, by what having it spared.
- It **drains** every frame by the entry's rent.

Where the balance sits decides what the entry is. It is one number climbing from zero toward a cap of **two
horizons of rent**, and the opening line at one horizon is the midpoint it crosses on the way:

```
2 horizons of rent                 the cap
[1 horizon, 2 horizons] of rent    the entry has a pattern
(0, 1 horizon) of rent             the entry is an embryo
0                                  the entry is deleted
```

**Nothing happens to the balance at birth.** It accumulates continuously; crossing the midpoint simply changes
what the entry is called, from embryo to pattern. There is no grant and no discontinuity — the second horizon is
headroom a pattern earns by continuing to serve, and the cap is what stops it banking more life than two horizons.

**The crossing is reversible.** A pattern near the midpoint that stops serving drains below it, is withdrawn, and
is an embryo again — holding everything it learned. Serving again re-crosses and it is reborn without having lost
its statistics. Flickering across the line is harmless because the crossing costs nothing; only draining all the
way to zero deletes it.

That is why a pattern never falls off a cliff. The old distinction between "not yet born" and "died" collapses:
both are the same entry at different points on one axis, and the axis is the balance.

**Forgetting is not a separate mechanism.** Charging rent in the routing table *is* aging. What
[forgetting.md](forgetting.md) still carries is only the question of whether this beats a uniform forget rate,
which is an experiment rather than a mechanism.

### What shrinking the file actually looks like

The frame part of the file dominates, because it is written `H` times. Shortening it means **fewer apex neurons
per frame**: 784 names on a 28×28 input if nothing compressed, 200 if level 1 covers the frame with 200 apex
neurons, 50 if level 2 covers it with 50. That reduction is what the whole design is for.

But the frame part alone is not the objective, and taking it as one has a degenerate optimum: memorize each frame
as a single top-level pattern. One apex neuron, full coverage, the shortest possible frame part — and a dictionary
holding one line per frame ever seen, which is longer than the input it replaced. So 784 → 200 → 50 is a win only
when the patterns that achieved it are **reused across many frames** rather than opened per frame.

Every structural decision is the same question: does the move shorten the file, dictionary included?
**No similarity thresholds exist anywhere in the design.**

### Every decision is local

Each neuron charges rent and measures service cost over its own neighborhood only, so the sum of local ledgers is
not the length of the file — the same base event is witnessed by every neuron whose neighborhood contains it.
This is accepted: local ledgers gate local decisions cheaply and in parallel, and [contraction](#contraction-building-the-level-above)
is what stops that redundancy from multiplying up the levels.

Nothing consults a global total. Rent scales with the neuron's own entry count, never with the size of the brain,
which is why a naive neuron opens cheaply — its exploration budget, not a leak, since the same local comparison
that opened an entry closes it when demand moves away.

The global quantity is therefore measured directly rather than summed: **apex neurons per level per frame, and
the dictionary size that bought them.** That pair is the standing metric, and the local ledgers are only a cheap
parallel proxy for it.

## Birth

An entry whose balance crosses one horizon of rent gains a pattern neuron **one level above the neuron's own
level**. If the balance later falls back below that line the pattern is withdrawn and the entry is an embryo
again, so this is a crossing rather than a one-way event.

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

Each neuron carries an **activation strength**, and it is not a separate quantity: it is the
[balance](#the-balance-is-the-whole-lifecycle) of the entry that fired. An entry deep into solvency is one whose
demand keeps returning, so ranking by balance ranks by how established a pattern is.

A neuron that fired no pattern — one served by its connections — has no such balance, and does not take part.
Only neurons that delegated to a child are candidates here, which is consistent with contraction being what
turns fired children into the level above.

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

**No probability sets a cost.** Distance is a count of neurons, cost is a count of neurons, savings is a count of
neurons — the pricing never leaves whole numbers, so no estimator, smoothing, or boundary correction is needed.

The only frequencies in the design are the **connection counts**, and they are used raw. The base model predicts
each neighbor by the argmax of its counts; a prediction only has to name a bucket, not price one, so nothing is
smoothed. A newborn's configuration is the exact observation that opened it and it serves that observation with
zero error, so the old worry that a pattern could be born unable to fire on its own configuration cannot arise.

Estimators return only in [forgetting.md](forgetting.md), where the file is re-priced by how often each neuron
occurs — the one variable-length code in which probabilities set costs.

## Refinement

On each serve, the members present strengthen and the absent ones dilute relatively as the serve count grows.
This is an online mode collapse: the stored configuration converges to the recurring core of the frames the entry
actually serves, so a shape whose boundary was drawn slightly wrong at birth converges to the right one through
ordinary use. Low-share members are born low and fade, so no explicit prune rule is needed. This is facility
location's **move**: a pattern relocating toward the demand it serves.

Connections carry no refinement — they are plain accumulated counts.

## One frame, in order

```mermaid
flowchart TD
    A[Frame: every dimension fires one bucket] --> B[For each active neuron: observe neighborhood O]
    B --> C[Find the nearest routing entry e*]
    C --> D{Does e* have a pattern?}
    D -->|yes| E[Fire it — inference delegated to the child]
    D -->|no| F[Connections infer the neighborhood, and update]
    E --> G[Bank: e* balance += members − 1 − errors]
    F --> G
    G --> H[Accumulate: e* error total += errors this frame]
    H --> I{e* error total ≥ 1 + |O|?}
    I -->|yes| J[Open a new pattern whose configuration is O]
    I -->|no| K[Charge rent on every entry; drain, delete at zero]
    J --> K
    K --> L[Contraction: thalamus runs the inhibition passes]
    L --> M[Each group contributes one unit to the level above]
    M --> N[Process the next level up]
```

## Implementation plan

Each phase lands independently and is measured before the next. The headline metric is the objective itself:
**survivors per level per frame, paired with the dictionary size that bought them.** Alongside it: MNIST accuracy
(train and held-out), neuron counts per level, patterns per neuron, and wall-clock per frame.

**Phase 1 — the configuration loop at level 0.** The whole-neighborhood configuration; inference from the
pairwise connection distributions; the routing table with partial matching; the nearest entry serving, banking to
its balance, and accumulating error to open new entries; promotion when the balance crosses the opening line;
recognition as a match with at most one child per neuron. Capped at one level so the recursion is not a variable yet.

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
- **Cold-start churn.** Early on nearly every neighborhood is novel, so patterns open in bursts. The balance
  drains the ones that do not recur, but the transient can be large. Measure churn in the first thousand frames,
  not just settled counts.
- **Dictionary redundancy.** Contraction shrinks the width of each level; it does not shrink the dictionary. The
  same shape is learned separately at every position — this is the [reuse](#open-questions) question, and it is
  the largest source of waste the design does not yet address.

## Open questions

Ordered by when they should be settled: the routing table first, then level processing.

### Routing table

- **The match tolerance.** That the match is [facility location](#the-routing-table-is-a-facility-location-problem)
  with Hamming distance — the count of neurons an entry gets wrong — is settled. What partial-matching tolerance
  rides on top, and therefore what lands on the same entry versus opens a new one, is not. It decides whether the
  routing table stays bounded.
- **Ranking born against unborn.** One match runs over the whole table, so an unborn entry can outscore a born
  one — the neuron would decline to fire a child it has in order to serve from an embryo instead. Whether that is
  correct is undecided.
- **Birth-frame attribution.** A newborn's starting balance is settled — it opens at zero and banks its first
  serve like any entry, landing at `|O| − 1`, about two short of the opening line, so it becomes a pattern on its
  second serve. What is not settled: on the birth frame the observation `O` was already served by `e*`, and now
  the newborn serves it too. If both bank, the frame is double-counted. The likely rule is that the newborn takes
  over serving `O` and banks it, while `e*` — whose accumulated error triggered the split — banks nothing that
  frame, but this is not decided.

### Level processing

- **The level transition.** Contraction, grouping, reuse, and merging are one operation, run by the thalamus at
  the boundary between levels, and together they decide which neurons are active at the level above. §Contraction
  writes only the inhibition half. The two questions inside it: whether reuse is keyed on the group's shape rather
  than on any single neuron's configuration, and which unit earns the saving once a group collapses. This is the
  biggest gap in the document.
- **Configuration space beyond the small case.** Eight binary neighbors is 256 configurations — finite and
  countable. Radius 2 is 2²⁴, and more buckets multiply it further. Exact-configuration counting is bounded
  precisely in the regime where it is least needed. What replaces it when the neighborhood is larger?
- **Configuration space at higher levels.** Above level 0 the neighbors are patterns, and the per-channel
  alphabet grows as patterns are promoted, so the configuration space is not fixed. One-active-per-neuron holds
  each channel to a single state, but the space still expands with the structure. What bounds the routing table there?
- **Reuse across positions.** Identity is keyed by absolute dimension, so nothing shares a shape across positions —
  the same configuration is learned separately everywhere it occurs. Contraction fixes the width of each level, not
  this redundancy of the dictionary. It is filed under Risks today and belongs with the level transition.

### Independent

- **Parallelism.** The inhibition passes are cross-neuron comparisons and run in the thalamus. Whether they can be
  parallelised, or pushed into neurons given a shipped view of neighbor strengths, is undetermined.

## Still to write

- **Reconcile the early sections with the serve/bank/accumulate/open model.** §What a neuron does with a frame,
  §Recognition, and §The womb were written before [facility location](#the-routing-table-is-a-facility-location-problem)
  and still carry the older "deposit / promotion" framing. The concrete conflicts: §What a neuron does with a frame
  says a firing pattern does "nothing else" — but a firing pattern banks savings, drains rent, and (open question)
  may accumulate its own error to split; and "evidence sized by how badly the connections did" needs to read as the
  neurons the server got wrong. These sections need one pass to match the model that §The cost and the facility
  location section now define.
- **The level-transition section.** Contraction, reuse, grouping, and merging as one thalamic operation. Everything
  else in the document stands on top of it.
- **Terminology.** §Contraction says "survivor" and §The cost says "apex neuron" for the same set. Pick one.
- **Move dictionary redundancy out of Risks** and into the level-transition section once that exists.

## Temporal: the open port

The event axis (`d > 0`) is not specified here. Two things are known about where it lands:

- **The context is the active neurons at `d > 0`**, used to infer the next frame.
- **A temporal pattern infers its own level**, not the one below, so at the moment of its creation its event
  connections are empty — the next frame has not been observed yet. It learns them once that frame arrives, the
  same rule as the spatial side: a newborn is born with no connections and learns them at its own level.
