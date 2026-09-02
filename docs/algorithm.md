# Universal Compression with Actions and Rewards (UCAR)

UCAR is a machine that compresses what it observes by building a hierarchical dictionary of patterns, and
learns what to do by observing rewards. It is defined by two alphabets, like a Turing machine — the **event
alphabet** it can observe and the **action alphabet** it can execute — and above each it forms symbols of its
own. Every symbol, base or learned, event or action, is a **neuron**.

This document is the specification, and nothing else. **D** is a definition and **R** a rule; together they
are the machine. Theorems, worked examples and all commentary on why the design is shaped this way live in
[algorithm-remarks.md](algorithm-remarks.md), keyed by the same D and R numbers; risks, diagnostics and open
questions live in [algorithm-evaluation.md](algorithm-evaluation.md).

**How it is ordered.** Section 2 states the objective everything else is derived from, so it comes before the
mechanisms that optimize it, and section 3 maps one frame end to end so every rule below has a place to sit.
Part I is a neuron on its own evidence, Part II the machine over a level, Part III the action side. Nothing is
stated twice: where two sections need the same fact, one states it and the other cites it.

**Notation.**

```
f, g, h    frame numbers
δ          a raw coordinate difference, before D16 resolves it to an offset

reach(k)   the reach at level k, derived                                        D15
reach_t    the time dimension's reach at the neuron's own level — the window    D15
W          the frame buffer's depth, reach_t + 1 — 2 at the base                D15

O          an observation — what a neuron saw around one firing                 D19
O⁻         its backward half, complete the frame the neuron fires               D20
C          a candidate neighborhood, grown a backward neighbor at a time        R14
nbhd(e)    an entry's neighborhood, |e| its size                                D19
covers     the neurons a neighborhood names that fired                          D17
price      1 + the neurons it names that did not                                D17
residual   the neurons of an observation no entry covers                        D17
n          the size of a collapse population                                    R4

L          the file length                                                      D13
D          the highest level the stack currently holds                          R17, T13
H          the history size                                                     D25
```

---

# 1. The machine

## 1.1 The substrate

> **D1 — Declaration.** The machine declares **channels**, and a channel declares two kinds of dimension.
>
> **Neuron dimensions are structural**: they say what a symbol *is*. Each channel declares at most one **event
> dimension** and at most one **action dimension**, each with its **resolution**, its bucket count. A base
> symbol is a dimension–bucket pair, so this declaration *is* the alphabet.
>
> **Activation dimensions are fleeting**: they say where one instance of a symbol happened. Every channel has
> **time**; what else it has is the shape of its input — an image channel declares two more, a stream of prices
> one more. Which kind a dimension is follows from the side of the input it sits on: the input is *laid out
> over* its activation dimensions and *reports* its neuron dimensions at each point of that layout.

> **D2 — Type and instance.** A **neuron is a type** and an **activation is an instance of it**, and each
> carries the dimensions of its own kind (D1).
> ```
> neuron coordinate       (dim_id, bucket_id)           structural and defining
> activation coordinate   frame, and one position per   fleeting; two activations of one neuron
>                         activation dimension          differ in nothing else
> ```
> A neuron's channel is the channel owning its dimension. A neuron minted as a pattern inherits its parent's
> channel and dimension and sits one level above it.
>
> **A child's activation inherits the parent activation's coordinate** — the frame, and the position in every
> activation dimension — and never an average over what it covers.

> **D3 — Channels are declared, never created.** No mechanism mints a channel; what grows is the population
> inside one, level by level and without bound. The channel set, and with it the dimension set, is a fixed
> enumerable index over the whole run, which is what lets `(dimension, offset)` name a slot at any level.

> **D4 — Adjacency.** Two activations are neighbors when they are **within reach in every activation
> dimension they share** (D1, D15). It is a conjunction, so a neighborhood is a box.
>
> **Nothing declares which channels may see which**: what a channel is laid out over settles it. Two channels
> sharing only time are related in time alone, and a channel whose one activation dimension is time — a stream
> of characters — stands in no spatial relation to anything.
>
> Neighbors are always at the neuron's own level, since a neighborhood is written over the symbols that level
> offers. **The rule is identical at every level**; what differs is the reach (D15).

## 1.2 Firing

Every frame, each event dimension quantizes what was observed — if anything was — and each action dimension
carries the action executing in that same frame — if one is.

> **D5 — Firing.** A neuron fires only when something happens: an event neuron input, or an action neuron
> output when its action is executed.
>
> **At the base, at most one neuron fires per dimension per position in a frame.** The input reports one symbol
> per channel at each point of its layout, and a dimension with nothing to report there is silent. That bound
> is a property of the input, not a rule the machine imposes.
>
> **Above the base it does not hold.** A neuron covers one observation with a *set* of entries (R18), so
> several of its children may be promoted at one coordinate, and a `(dimension, position)` at a pattern level
> can carry several firings in a frame. D16 already admits several neighbors at one offset; this is the same
> situation, and `|e|` counts them the same way.
>
> **A neuron may fire many times in one frame**, once per position, and those firings are instances of one
> type (D2).
>
> **A frame's two halves are simultaneous.** What is observed and what is executed occupy the same column of
> the grid, so an action runs alongside that frame's events and both are in hand together (§14).

> **D6 — An activation stays open, and what it does while open.** A firing at frame `f` remains open through
> `f + reach_t`, the neuron's own reach in time (D15) — 1 at the base, wider above it.
>
> **Nothing the neuron decides waits for that.** Its backward half is complete the frame it fires (D20), and
> every structural act — which entries cover it, which children bid, what is added, what retires — happens then
> and there (R13). **An activation is open for one reason: to accrue.** Two things arrive over the window and
> neither is a decision:
> ```
> the forward half   the neighbors at Δt > 0, one frame at a time, into its bin's tallies    D22
> the adjustment     what other accepted bids covered, as coverage settles                   D28
> ```
> Both fold into the bin the activation joined at age 0 (R19), never into a decision the neuron is holding open.
>
> **The machine holds the open activation.** The arriving neighbors are its own frame data on the way in, and
> it accrues them on the neuron's behalf; the neuron is not called for it.
>
> **Age is per activation, and a neuron carries several ages at once.** An activation's age is the frames
> elapsed since it fired, `0` through `reach_t`, and it is the machine's counter. Nothing distinguishes
> activations but age: a new firing is the activation whose age is 0. **Age is read, not just counted** — it is
> the distance at which that activation holds its connections and votes (R37, R42).

> **D7 — Exclusion is per level, and about coverage.** A neuron an accepted bid covers is silenced in the
> machine: it does not stand in the file and it does not vote (R30, R31). That is the whole of inhibition, and
> it bounds nothing about how many neurons fire at a coordinate.
>
> **The design's only exclusivity is assignment, and it appears twice.** Every neuron a bid covers is credited
> to exactly one bid (R27), and every neighbor of an observation is covered by exactly one of the entries
> covering it (R18). Both are partitions of credit, and neither is a limit on firing.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what the decoder
assumes for anything the file does not state.

> **D8 — Identity is absolute.** A neuron's identity is fixed entirely by its neuron dimensions, so nothing
> about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: their observations carry the same relative neighborhood, so they land in one bin (D21) and one
> cover serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

---

# 2. The objective: it is all one file

Write every frame observed so far as a single file a decoder could read back to reproduce each of them
exactly. **The file is the run.** There is no window and no horizon: nothing is dropped from the objective
because it happened long ago, and everything the machine's structure is worth is measured against the whole of
what it has seen.

> **D9 — Predictive coding.** The decoder runs the same model as the encoder, so it already knows what each
> active unit predicts about the frames ahead. **The encoder writes only the surprise.**
> ```
> asserted {␣} at +1, actual {␣}       →  0 symbols
> asserted {␣} at +1, actual {e}       →  2 symbols  (turn off ␣, turn on e)
> asserted nothing at +1, actual {e}   →  1 symbol   (turn on e)
> ```
> **Being wrong costs twice what saying nothing costs.** That asymmetry decides when the machine should commit
> to a symbol at all, and it is derived from again in §6.2 and §11.4.

> **D10 — The file.** Two parts, both over the whole run. **The dictionary**: the neighborhoods needed to expand
> what the history states. **The history**: the apex units, and the corrections where they were wrong. Anything
> unstated is silence.
>
> **It is the current optimum encoding of the run.** The file is re-derived from the structure as it now
> stands, exactly as a price is (R2), so no past decision has to be honored and nothing is retained that the
> machine could not expand.
>
> **Nothing ever materializes it.** D27's two windows span `2·reach(k) + 1` in each activation dimension and a
> neuron's history is `H` observations deep (D25), so no pass anywhere walks the run.

**One file, one dictionary.** A neuron's history (§7) is its own record of what it saw. An entry in a routing
table is a **standing claim on a dictionary line**, and whether that claim is honored is settled where the
file is (R12).

> **D11 — Prices.** Every cost in the design is part of a file, counted in symbols:
> ```
> activating an apex unit  =  1                    a line in the history
> what it got wrong        =  the neighbors wrong  the corrections after it
> having a child           =  1 + |e|              a line in the dictionary
> ```
> This is a fixed-length code: a symbol costs one regardless of how often it is used.

> **D12 — What the file holds.** The dictionary, and the history encoded against it. A child's neighborhood
> must be written: it is the collapse of observations no ring still holds, so nothing in the file recovers it.
> **A symbol is not one line per frame** — a unit firing at `h` names `[h − reach_t, h + reach_t]`, so writing it
> discharges up to `2·reach_t + 1` frames of the run at once. What the machine holds and the file does not is
> search state — counts, tallies, distances, margins — because expanding an apex unit needs the neighborhoods
> and nothing else.

> **D13 — File length, and what a neighborhood is worth.** Over the run the file is
> ```
> L  =  Σ over dictionary (1 + |e|)  written once
>    +  Σ over history (1 + errors)  written every activation
> ```
> summing D11's prices over what D12 says the file holds. **There is one `L`**, and every neuron's structure is
> priced against it.
>
> **`L` grows without bound and nothing in the design ever computes it.** No structure holds a file length and
> no test compares two of them. Every quantity actually used is a **difference** in `L` — whether the file is
> longer or shorter for holding one neighborhood — which is finite however long the run is, and local: a count
> of neighbors over the observations that neighborhood covers (D17).
>
> **What one neighborhood is worth over one observation.** It accounts for the neurons it names that fired, at
> the cost of its own line and the neurons it names that did not:
> ```
> margin(e, O)  =  covers  −  price  =  |O ∩ e|  −  ( 1 + |e \ O| )
> ```
> **A neuron `e` does not name is not `e`'s business.** It costs one symbol whether or not `e` exists — its own
> line if nothing covers it, a turn-on if something does (D9) — so it is on neither side of this. **Naming a
> neighbor wrongly is what is charged; leaving one out is not.**
>
> **This is the only valuation in the design.** The one test reads it over `H` observations (R12) and the
> election reads it over one frame (R21), and they are the same expression over different populations.
>
> **The adjustment is what makes this local sum honest.** The one thing the neuron cannot know is overlap: a
> bid claims to cover `{a, b, c, d}` when `a` and `b` were already covered by another accepted bid. **So the
> machine reports the overlap and the neuron records it** (D28) — the *fact* that these neighbors were not
> credited, never the number that came out of it. A neighbor another bid took leaves `covers` and enters
> nobody's price.
>
> **Every term but the adjustment is measured now** (R2): `nbhd(e)` is wherever re-centering has put it, and
> what it covers follows. **§8's test is the derivative of `L` with respect to holding one entry**, read over
> the observations the neuron holds.

---

# 3. A frame, in outline

Nothing below is stated here: this is the order the rest of the document is read in, and every step names
where it is specified. A frame arrives carrying what each event dimension observed and what each action
dimension is executing (D5), and the machine works **up one stack, a level at a time**.

```
per level, in this order and no other

  cover      each neuron that fired covers its backward half with a set of entries; every
             entry in that set that has a child returns a recognition, which is its bid       R18
  election   the machine assigns each covered slot to one bid and accepts the bids that
             cover more than they cost; the level's uncovered neurons are corrections         §11.4
  bill       each activation that fired this frame folds its backward half into its bin,
             and the neuron asks what to add and what to retire                               §10.2

  the level above is built out of what the election accepted, and it happens again;
  a level that accepts nothing has no level above it this frame                               §12

then, once the last level has billed

  ledger     delete every entry and pattern the bills retired that is now unreachable         §9.2
  actions    every open activation records a connection to the action that ran, and
             returns the inferences competing for the next action slot                        §15, §16
  assert     the uncovered neurons' standing recognitions resolve into one owner per slot     §13
  accrue     every open activation writes this frame's arrivals into its bin — the forward
             neighbors that landed, and the coverage that settled                             §10.3
```

That last pass commits the action for the frame ahead; the reward for it arrives the frame after (R35).

**Every decision above is made on complete evidence.** A neuron's backward half is whole the frame it fires
(D20), and nothing structural reads anything else — so no step is a bet, nothing is committed early, and
nothing is revisited. What the remaining `reach_t` frames deliver is evidence for the *next* decision, not a
verdict on this one.

# Part I — A neuron

# 4. Neighborhoods and distance

## 4.1 The neighborhood

> **D14 — Neighborhood.** An activation observes the active neurons of its own level that adjacency admits
> (D4), each tagged with its **offset** — the difference of activation coordinates, one component per
> activation dimension the two share (D1, D2):
> ```
> O = { (p, −4), (a, −3), (r, −2), (i, −1), (␣, +1) }        a stream:  one component, time
> O = { (k, 0, −1, 0), (k, 0, +1, 0), (m, 0, 0, −2) }        an image:  three, time and two axes
> ```
> the first for a neuron `s` in a stream reading `p a r i s ␣`. At temporal offset 0 a neighbor co-occurs, at
> negative offsets it led here, at positive offsets it followed; **all three are the same kind of thing**, and
> a spatial component is one more of the same.
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other
> at a nonzero spatial offset. Only offset zero in every component is the activation itself, and that is the
> center.
>
> **One neighborhood per firing, and a set of entries covers it** (R18), each answering for the neighbors it
> was assigned and none of them for the rest (D17).

A neighborhood is a set of neighbors, each at its own offset. Drawn with a row per dimension and a column per
offset, with one activation dimension and a reach of 1:

```
                          offset
                   −1      0     +1
        dim A       a      ◉      ·        ◉  the firing neuron
        dim B       b      ·      ·        a  a named neighbor
        dim C       ·      ·      c        ·  silent
        dim D       ·      ·      d
                  ╰──── O⁻ ─────╯╰──── O⁺ ────╯
                   whole at age 0          lands over the next reach_t frames
                   covered, priced,        asserted, scored, accrued —
                   added and retired here  and never priced       D18
```

> **D15 — Radius and reach.** The radius is **2**, in every activation dimension of every channel (D1).
> Nothing declares it. The reach it gives is **1** either way along every dimension at the base, and it
> **doubles every level**:
> ```
> reach(k)   =   2^k          every activation dimension
> reach(0)   =   1            the base
> ```
> **No level declares anything, and there is no rate to choose.**
>
> In time, `W = reach_t + 1` is the depth of the frame buffer — 2 at the base. The buffer is a sliding window:
> an activation sits at its newest edge when it fires, with `reach_t` frames of context behind it, and slides
> to the oldest edge as its neighborhood completes.
>
> **The reach and the granularity are the same power of two.** D16 resolves offsets in base 2, so a level
> reaching `2^k` names its outermost offsets in groups of `2^k`.

> **D16 — Offsets are resolved logarithmically.** An offset component is the coordinate difference **kept to
> one significant digit in base 2**, the radius (D15):
> ```
> offset(0)   =   0
> offset(δ)   =   sign(δ) · 2^g · ⌊ |δ| / 2^g ⌋          δ ≠ 0,  g = ⌊ log_2 |δ| ⌋
> ```
> **Offset zero is the center and is given, not computed** — it is the only case `log_2 |δ|` does not reach.
> Every other `δ` is a nonzero integer difference of activation coordinates (D2), so `|δ| ≥ 1` and `g ≥ 0`
> without a floor of its own, and above `g = 0` the multiplier is always 1. The reachable offsets are
> `0, ±1, ±2, ±4, ±8, ±16, …`; `G` groups give `2 + G` offsets per direction across a reach of `2^G`, and
> `reach(k) = 2^k`, so `G = k`.
>
> **Near offsets come out exact and distant ones are named coarsely. Nothing is cut off; precision decays
> instead.**
>
> **A coarse offset may carry more than one neighbor**, since it spans a range and several activations of one
> dimension can fall inside it. A count is per `(neuron, offset)` (D24) and `|e|` counts every neighbor named
> there. Above the base a `(dimension, position)` may itself carry several firings (D5), which this handles the
> same way and needs no rule of its own.

## 4.2 The fit

> **D17 — What a neighborhood accounts for, and what it costs.** An entry `e` measured against an observation
> `O` gives three sets, and every price in the design is counted off them:
> ```
> covers    O ∩ e     the neurons it names that fired — what it accounts for
> price     1 + |e \ O|   its own line, and the neurons it names that did not fire
> residual  O \ (the union of the entries covering O)   the neurons nothing accounts for
> ```
> **The residual is not an error.** Each of its neurons stands in the file as its own line, at cost 1 (D11),
> exactly as it would if no entry existed — so it is charged to no entry and credited to none (D9, D13).
>
> **An observation is covered by a set of entries, not by one** (R18). They partition what they cover: a
> neuron is `covers` for exactly one of them, so nothing is accounted for twice. What is left over is the
> residual.
>
> **The cost of an observation** is therefore what its cover costs plus what nothing covers:
> ```
> cost(O)  =  Σ over the entries covering O ( 1 + |e \ O| )  +  |residual(O)|
> ```
> and an observation nothing covers costs `1 + |O|`, the whole chunk stated flat.
>
> **Distance is a reading of the same three sets.** Where one entry is measured against the whole of an
> observation, `d(O, e) = |O △ e|` and `margin = |O| − d`; that is the identical number D13 gives, written
> against a flat baseline instead of a subset one. **The design uses the subset form everywhere**, because an
> observation's cover is a set and only the subset form adds up over one.

> **D18 — The backward half decides; the forward half predicts.** Cut on the temporal component alone (D20).
> ```
> O⁻   Δt ≤ 0    complete the frame the neuron fires   every structural decision is made on it
> O⁺   Δt > 0    arrives over the next reach_t frames  it is the prediction, and it decides nothing
> ```
> **The cut is availability, and it is not a compromise.** `O⁻` is whole when the neuron fires, so covering it,
> pricing it, adding and retiring are all done on complete evidence and none of them is a bet. `covers`,
> `price` and `residual` are read over `O⁻` and over nothing else.
>
> **What the forward half is for.** It is what an entry asserts (§13), what its child's expansion places when
> it is an action (R36), and what the machine writes corrections against when it is wrong (R34). It enters no
> test. **It is not chosen — it is measured**: the collapse over what actually followed (R4) is the best
> statement of it available, so there is nothing about it to decide and nothing to price.
>
> **The cut is on time alone.** Spatial components never enter it: a neighbor three positions to the right
> arrives in the same frame as one three positions to the left, so both are in `O⁻`.

> **R1 — One comparison.** There is no second one. An observation is covered on `O⁻`, priced on `O⁻`, and the
> tests that add and retire read the same numbers over the history. Nothing anywhere compares a neighborhood
> against a forward half, and no quantity in the design waits for one.
>
> **What arrives later is evidence, not a verdict.** The forward neighbors and the settling coverage accrue
> into the bin (D6) and are read by the *next* decision the neuron makes, along with everything else its
> history holds.

> **R2 — A price is a measurement, not a record.** What an observation costs is read off its cover as that
> cover now stands (D17), and the cover can change: re-centering (R5) moves a neighborhood, which moves what it
> covers in every bin, and those are what the bin's observations then cost (D22). **An observation is fixed but
> its cost is not**, and it stops moving when its cover stops moving.
>
> **The adjustment is a record, and it is not a price.** What the machine reported about a frame (D28) cannot
> be re-derived, so it is stamped once and never revisited. **It is evidence, exactly like the observation it
> rides on**: a fact about what happened, which prices are then measured against.

# 5. State

Only two things a neuron holds cannot be recomputed: the history of what it observed, and the entries it has
decided to keep. Everything else it holds is a total.

> **D19 — Observed and named.** Two things have the shape of a neighborhood and must not be confused. Both
> span the whole box D4 admits, and D17 measures one against the other.
> ```
> an observation   what the neuron SAW around one firing; recorded once, evicted `H` firings later
> a neighborhood   what an entry NAMES; the collapse of what it covers (R4), moving as that moves (R5)
> ```
> An observation is a fact, a neighborhood a claim. Neither is a frame: a frame is one column, an observation
> the whole window.

> **D20 — Halves.** Cut at the firing frame, on the temporal component alone. The **backward half** is the
> neighbors at `Δt ≤ 0` and the **forward half** those at `Δt > 0`. An entry's neighborhood has the same two
> halves, and D18 says what each is for.

> **D21 — Neuron state.** `°` marks a total: recoverable by a walk, kept to avoid one.
> ```
> neuron           = (coordinate, routing table, history, connections)
>
> routing table    = set of entries
> entry            = (id, neighborhood, child, counts°, covered bins°)
>
> history          = (bins, ring)
> ring             = observations, oldest first
> observation      = (position, forward half, adjustment)     both filled in over the window (D6)
> adjustment       = (covered:    which backward neighbors another unit took,
>                     superseded: which forward slots this activation does not own)
> bin              = (backward half, observation count,
>                     cover°, assignment°, price°, residual°,
>                     forward tallies°, covered tallies°, superseded tallies°)
>
> connections      = one per (distance, action neuron) toward what has followed (R37)
> connection       = (strength, estimate)               exposures, and the mean reward over them
>
> held by the machine, not the neuron (D6):
> open activations = one per (neuron, age, position)   still accruing
> open activation  = (position, age, its bin)
> ```
> An `id` is creation order, a handle that survives re-centering and the tie-break R18, R24 and R27 reach for.
>
> **An observation stores no backward half**: every observation in a bin carries that bin's key exactly (R7),
> so the bin holds it once. **No observation carries a frame number**, and nothing anywhere holds absolute
> time: an open activation's `age` is a counter.
>
> **An open activation carries no commitment.** It holds the bin it joined at age 0 and nothing else, because
> everything it was going to decide was decided then (D6). What it does for the rest of its life is write
> arrivals into that bin.
>
> **Connections hang off the neuron, beside the routing table.** **Nothing an entry does reaches one** — a
> connection dies only when one of its two neurons does (R37).

> **D22 — What the totals owe.** In dependency order, so the list also says what to recompute when something
> moves.
> ```
> bin.cover          =  the entries covering this bin's key, chosen by R18 over the key alone
> bin.assignment[n]  =  which entry of the cover holds neighbor n            a partition of the key
> bin.price          =  Σ over the cover ( 1 + |e⁻ \ key| )
> bin.residual       =  the key's neurons no entry of the cover names
>
> entry.covered bins =  { b : this entry is in b.cover }
> entry.counts       =  Σ over covered bins:  observation count × the neighbors assigned to it   (backward)
>                       Σ over covered bins:  their forward tallies                              (forward)
>
> bin.forward tallies    =  Σ over its observations, per (neuron, forward offset)
> bin.covered[n]         =  # observations whose adjustment took key neighbor n     (D28)
> bin.superseded[o]      =  # observations whose adjustment took forward slot o
> ```
> Credited neighbors are the key's neighbors less `covered`, and the priced forward slots are the offsets less
> `superseded`.
>
> **The cover is a property of the bin, not of an observation** (R7). Every observation with that key was
> covered the same way, because R18 reads the key and nothing else.
>
> **Three of these accrue rather than being computed.** The forward tallies, `covered` and `superseded` are
> written into the bin by open activations over their windows (D6) and are never derived from anything the
> neuron holds. The rest are totals over the key and the table, and move when either does.
>
> **The tallies are sparse**: indexed by the neighbors actually seen, not by everything the box admits. Only
> the forward half needs tallies; backward, every observation carries the key, so the count *is* the tally.

> **D23 — The residual is not an entry.** What no entry covers is not routed anywhere and has no line to pay:
> it is a set of neurons, each standing in the file as itself (D17). **There is no default entry, no fallback
> and no empty neighborhood** — a routing table may be empty, and an observation it covers nothing of costs
> `1 + |O⁻|`, which is what an uncompressed chunk costs.
>
> The **neighborhood** is the whole of what an entry knows — neighbors behind the neuron and ahead of it, with
> no separate notion of context or of what it infers.

# 6. Counts, the collapse, re-centering

## 6.1 Counts

> **D24 — Counts.** Each entry keeps, over exactly the observations it covers, `count(neuron, offset)` = how
> often that neighbor was present.
>
> **Backward, it counts only what is assigned to it** (D22). A neighbor of the observation held by another
> entry of the same cover is that entry's evidence, not this one's — the same rule the adjustment applies
> across neurons (D28), applied inside one table.
>
> **Forward, it counts the whole tally.** Nothing partitions the forward half: every entry covering a bin
> asserts over the same future, and which of them owns a slot is settled where assertions are settled (R32),
> not here. Two entries covering one bin still differ, because each collapses over the whole set of bins it
> covers and no two of those sets are the same.

> **R3 — What moves counts.** Three events, and every one of them moves a whole bin's worth.
> ```
> an observation joins a bin        every entry of the cover increments, by the neighbors assigned to it
> an observation is evicted         every entry of the cover decrements by the same
> a bin's cover changes             an entry leaving decrements by the bin's tallies, one arriving increments
> ```
> A cover changes when the table changes — an entry added, an entry retired, or a neighborhood moved so that
> R18 reads the key differently (R6).

**An entry counts only its own share.** Two of R3's three moves transfer a whole bin, which is arithmetic on
tallies — `O(offsets)`, not `O(observations)` — because a bin moves whole.

## 6.2 The collapse

The collapse is the only operation anywhere that decides what goes in a slot.

> **R4 — The collapse.** Over a population that each has something to say about one neuron at one offset, let
> `n` be the size of that population and `count(p)` the number of it naming `p` there. **`p` is taken exactly
> when `count(p) > n / 2`; otherwise it is left out.**
>
> Nothing is divided: `2 · count(p) > n` is an integer comparison. The alternative is `p`'s absence, which the
> remaining `n − count(p)` vote for, so the two are decided against one denominator.
>
> **An observation cannot abstain; a claim can.** An observation held that neuron at that offset or it did not,
> and either way it is in the population. A unit naming nothing at a slot has **no opinion about it**, so a
> unit that could have claimed a slot and did not is not in `n` at all.
>
> **The three populations.** One arithmetic, three times, and nothing else in the design decides what a
> neighborhood names or what a slot holds.
> ```
> an entry's neighborhood   the observations it covers          R4, at every bill (R5)
> a candidate's             the observations it would improve   R14, at every add test
> a slot's owner            the claims landing on it            R24, when the assertion resolves
> ```
> **Backward, a neighbor another entry of the same cover holds votes against.** It is present in the
> observation but it is not this entry's to name, so it counts toward `n` and not toward `count(p)`. An entry
> therefore grows into the residual and retreats from ground another entry holds, and two entries covering the
> same bins cannot converge on one neighborhood.
>
> **Forward, every observation votes plainly.** There is no assignment ahead of the firing frame to respect.

**One denominator, every offset.** Every observation an entry covers has something to say at every offset — a
neuron or a silence — so the outermost forward offset is decided by the same population as offset 0. **No
threshold, smoothing or probability estimate enters any of this**, and no denominator is ever shared between
two populations.

## 6.3 Re-centering

> **R5 — Re-centering.** An entry **re-centers** by running the collapse (R4) over the observations it now
> covers, and the bins it reads then re-derive their covers with it (R6).
>
> **An entry re-centers whenever its counts move** — no test, no gate, and free, because the counts are already
> maintained. Counts move only at a bill or an accrual (R3), and every move there disturbs a whole bin's
> worth, so a re-center is always over all offsets and never per observation.
>
> **A bill re-centers twice, once after each kind of count movement** (R19): after **evidence** moves counts —
> an observation joining a bin, another being evicted — before either test prices against the result; and after
> **structure** moves them — an add taking neighbors, a retire releasing them — once both tests have run.

> **R6 — Covers are re-derived, not patched.** A moved neighborhood changes what it covers in every bin. What
> is maintained is **one entry's covers-and-price against every bin key**: an entry that re-centers recomputes
> those, and nothing else is repaired. A bin's cover is then taken from the table by R18 whenever that bin is
> used — by recognition when it fires, and by the tests when they scan the history (R19).
>
> **A bin whose cover has changed takes its counts with it** (R3), so the entry that received a bin's share is
> always the entry that gives it back.

**Cold start is silence.** An entry with no observations has no counts and no neighborhood, and a neuron with
an empty table covers nothing and bids nothing.

# 7. The history

> **D25 — History size.** A neuron remembers its last `H` observations and no more. `H` is declared once for
> the machine and is the same for every neuron; it counts that neuron's **own firings**, not a stretch of run,
> so a neuron that fires constantly and one that fires rarely weigh their entries against the same amount of
> evidence. **The ring is exactly `H` deep once filled**, and how much run it spans is whatever that neuron's
> rate makes it. Nothing else anywhere is measured in frames.

> **R7 — Keyed on the backward half — both sides.** Two observations with identical backward halves present
> R18 with identical input, so they are covered identically; that is what makes the backward half safe to share
> a cover across, and the forward half cannot key anything because it does not exist when the cover is chosen.
>
> **An entry is keyed the same way**, by its own backward half. Two entries with equal backward halves are
> indistinguishable to R18, so whichever it takes first covers everything the other would, leaving the second
> with nothing assigned — and an entry covering nothing fails the retire test at the next bill (R17). **The
> table does not need a rule against duplicates; the tests remove them.**

A **bin** is the aggregate over its observations exactly as an entry is the aggregate over its bins. The cover
and its partition are properties of the bin's key; the forward half is what followed this context, per slot,
tallied.

> **R8 — The ring makes eviction exact.** Removing the oldest observation means subtracting the neurons *it*
> contributed, which a tally cannot recover, so each observation keeps its own forward half and the bin is the
> cached aggregate over them.
>
> **Two things read a forward half whole**: eviction, and the collapse that builds a candidate around a forward
> neighbor (R14). Everything else reads it per slot, off the tallies.

> **R9 — Aging is by count, and an observation may still be filling.** The ring is a FIFO `H` deep: an arriving
> observation evicts the oldest, and only then. Nothing compares a frame number, nothing accumulates arrears,
> and nothing sweeps the population per frame — a neuron that does not fire evicts nothing.
>
> **An observation enters the ring at age 0 and keeps arriving for `reach_t` frames** (D6), so the ring holds
> observations that are still accruing. **Eviction subtracts what was accrued, not what would have been**: the
> forward half and adjustment the observation had written into its bin leave with it, and if its window had not
> closed it simply wrote less. A neuron firing faster than its own reach evicts partial futures and prices
> against partial tallies, which is a smaller sample and not a wrong one.
>
> **An evicted observation stops accruing.** Its open activation is closed by the eviction, and closing one is
> the only thing eviction does besides the subtraction.
>
> Recording is unconditional, and **no election outcome ever edits a history** — deletion is the one thing that
> reaches into a folded observation, and it only removes names of neurons that no longer exist (R17).

> **R10 — Free parameter: the history size `H`.** It is the only one. The alphabet — channels, dimensions,
> resolutions — is the problem statement rather than a knob; adjacency is not declared (D4), the reach per
> level is not declared (D15), and neither is the offset alphabet (D16). **Nothing else in the design is
> tuned, and nothing anywhere is capped.**

> **R11 — `H` and the reach constrain nothing in each other.** `H` counts observations and the reach sets how
> wide one observation is. **What the two share is the collapse's evidence**: R4 votes per offset slot over the
> same `H` observations, so every slot — innermost and outermost alike — is decided on the same count, and a
> reach wider than the data supports finds no majority in its outer slots and they drop.

# 8. The one test

> **R12 — The one test.** An entry earns its dictionary line when the file is shorter for holding it than it
> costs to state. **Nothing measures a file to find that out**: both terms are counts over what the neuron
> already holds, so the margin is the difference in `L` reached directly (§2).
> ```
> benefit(e)  =  Σ over the observations e covers:  covers − price   (adjusted, D28)
> cost(e)     =  1 + |e|                                              the line  (D11)
> margin(e)   =  benefit(e) − cost(e)
> ```
> An entry is **added** only when its margin is strictly positive and **retired** only when strictly negative
> (R15, R17). At equality nothing happens, so the boundary cannot flip-flop.
>
> **`covers` is what nothing else would have covered.** An entry is worth what it saves over what would account
> for those neurons if it were gone — another entry of the same cover if one names them, and the residual
> otherwise, where each stands as its own line (D17). A saving some other neighborhood already delivers is not
> this one's, which is what the adjustment says across neurons (D28) and what this says inside one table.
>
> **The same expression prices a bid over one frame** (R21). There is one valuation in the design (D13); the
> two tests differ in the population they sum it over and in whether the dictionary line is in the sum.
>
> **Benefit is a measurement, so it moves when anything under it moves** — an observation joined or evicted, a
> neighborhood re-centered, another entry taking a neighbor. **No test needs a pass of its own.**

# 9. The two moves

A neuron can do exactly two things to its routing table: **add** an entry and **retire** one. Re-centering is
neither — it is what moving counts means (R5). So the whole of restructuring is two tests, asked in that
order, at a bill and nowhere else (§10.2). **Both are R12 over different sets** — one margin, read over the
neurons a candidate would take out of the residual and over the neurons an entry holds — and there is no
second currency anywhere in the design.

## 9.1 Add — creating a child

> **R13 — One decision point: the firing frame.** A neuron is called once per activation, at age 0, and
> everything structural happens in that call: it covers its backward half (R18), returns a recognition per
> entry of the cover, and — once the level's election has run — folds and asks both tests (R19).
>
> **There is nothing to wait for.** `O⁻` is complete when the neuron fires (D18), so no decision here is made
> on partial evidence, none is committed for later, and none is revisited. The neuron remembers nothing
> between one firing and the next beyond what is in its history.
>
> A child minted at a bill first fires on the next activation whose key its parent's cover reaches; it does not
> cover, bid or propagate on the activation that created it. **Structure never pays off on the evidence that
> created it, only on recurrence.**
>
> **The rest of the activation's life is accrual** (D6, §10.3) — the forward half and the settling coverage,
> written into the bin for the next decision to read.

> **R14 — Where a candidate comes from.** A candidate is built one backward neighbor at a time, taking at every
> step the neighbor that shortens the file most, and stopping when none of them does. **Nothing seeds it and
> nothing outside the file's length chooses anything**: the same history builds the same `C`.
>
> **Only the backward half is grown.** A neighborhood's forward half is measured rather than chosen (D18), so
> once `C⁻` is settled the forward half is the collapse (R4) over the forward halves of the observations `C`
> would cover — which the ring already holds, so `|C|` is known before the entry exists.
>
> **What each observation would save.** `C` takes neurons out of the residual and pays for its own line and for
> what it names that did not fire:
> ```
> saving(o)  =  max( 0,  |residual(o) ∩ C⁻|  −  ( 1 + |C⁻ \ o⁻| ) )
> ```
> **A candidate is only ever credited the residual.** A neuron an entry already covers is not `C`'s to take —
> a candidate that fits a chunk beautifully earns nothing for it if something already accounts for it. The
> floor is R18: an observation whose cover `C` would not join contributes nothing, and `C` will not join a
> cover it does not pay in.
>
> **What one neighbor does.** Adding `q` to `C⁻` moves each observation's saving by
> ```
> Δ(q)  =  #{ o : q is in the residual, and C pays there }      it takes one more neuron
>       −  #{ o : q did not fire, and C pays there }            it names one more absence
>       −  1                                                    q's own place in the line
> ```
> A neighbor that fired but is already covered moves nothing: it is not `C`'s to claim and it is not an error
> for `C` to name. **The three cases are the three sets of D17**, and every neighbor is in exactly one.
> ```
> C⁻ ← ∅                         every saving starts at zero
> repeat
>     take the neighbor with the largest Δ, and stop if that Δ ≤ 0
>     add it to C⁻, and move every saving
> ```
> **The repetition is not a search over orderings.** A neighbor's worth depends on which observations `C` is
> paying in, and that changes with every addition. One neighbor's worth of information comes out of each pass,
> and the pass that follows is measured on what the last one left.
>
> **It terminates, and the count is bounded.** Every round strictly increases the total by at least one and adds
> a neighbor that is never taken back, and the total can never exceed the residual the history is carrying. So
> ```
> |C⁻|  ≤  ½ · Σ over the history |residual(o)|
> ```
> **The residual bounds the loop and nothing else does** — a table that already covers everything admits no
> candidate at all.
>
> **What it is not.** It is greedy, so it finds a local best and not the best `C` — choosing the subset is the
> facility-location problem and is not solvable exactly at any useful size. What it has instead is no free
> choice anywhere in it.

> **R15 — The solo test.** R12, asked of a table `C` is not in yet.
> ```
> benefit  =  Σ over the whole history:  saving(o)      (R14)
> commit iff  benefit > 1 + |C|
> ```
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every observation in the ring has a whole backward half, so the
> question is asked over the same evidence R18 will use when `C` next competes for a cover. Nothing here is
> decided on half an observation and nothing later can hand `C` less than the test counted.
>
> **Candidates are grown until one fails.** Grow, price, install if it pays, and grow again against the
> residual the installation leaves. The pass ends when a candidate does not clear its line. It terminates for
> the same reason the retire pass does: every accepted add strictly shortens the neuron's file, and that file
> is bounded below by the count of its observations.

> **R16 — What a child is at birth.** The parent requests; the machine creates. The pattern inherits its
> parent's channel and dimension and mints one level above it, all carried on the request. It is created with
> **no counts**: its own neighborhood belongs to its own level, which it has not observed yet. Its *existence*
> is decided by its parent, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each is one entry's child, each covers the part
> of an observation its entry was assigned, and several of them may be promoted at one coordinate (D5). What
> they share is a parent and a coordinate, not a slot.
>
> **Two objects, one word.** The **neuron** minted one level up has no counts. The **entry** the parent now
> holds is a different object and takes counts at once — R19 hands it its share of every bin whose cover it
> joins.
>
> **Release is the same shape reversed**: the parent retires, the machine reclaims. A retired entry goes back
> on the same request that carries the add (R19) and the machine reclaims its pattern neuron at the death frame
> (R17), so a bill touches the alphabet once — in one direction, both, or neither.

## 9.2 Retire — pruning the table

> **R17 — Retire, then delete.** Scan the whole routing table, every candidate the add pass just installed
> included. Every entry whose margin is strictly negative (R12) is **retired**.
> ```
> benefit  =  Σ over the observations it covers:  |neighbors only this entry names|  −  ( 1 + |e⁻ \ o⁻| )
> retire iff  benefit < 1 + |e|
> ```
> **Without the entry its neighbors fall where D17 puts them** — to another entry of the same cover that names
> them, at no extra cost to that entry, or into the residual at one line each. It is the same difference the
> add test reads with the roles swapped: the add asks what a neighborhood that is not there would take out of
> the residual, the retire asks what one that is there is still keeping out of it.
>
> Retirement is one at a time, re-checking the rest after each — **two entries naming the same neurons are each
> worth nothing while the other stands**, and taking both on one reading would return that demand to the
> residual with nothing left to cover it. The pass is bounded: every retirement removes an entry from
> competition and creates none.
>
> **Retiring is a deletion in the parent.** The entry leaves the routing table that instant. It stops competing
> for a place in any cover, so no further activation can bid it, and the neurons it held fall to whatever D17
> gives them next. Having nothing to cover it has no margin and nothing to re-center — **the neighborhood it
> held stops moving** — and it is not a candidate for anything again. What leaves the table rides the bill's
> return to the machine (§10.2), as a request to delete it. **The neuron keeps no retired state and re-checks
> nothing.**
>
> **The death frame is the machine's to set, and it can be this frame.** The neuron asks for the child to go
> and says nothing about when; it sees its own open activations and not the units promoted off them, while the
> machine sees both.
> ```
> death frame   =   when the last activation still able to name the entry closes
>               =   this frame, when none is open
> ```
> That set only shrinks. A pattern neuron has one parent (D2), so once that parent stops covering with it
> nothing can fire it again, and no level built afterward can name it either, because a level is built out of
> what is firing (R29). Reach grows with the level (D15), so the last to close is the highest one and the wait
> is at most `reach_t(D)` frames, `D` being the highest level the stack currently holds.
>
> **Deleting** is the machine's, on a pass of its own: **every frame, once the last level has billed**, it
> reads the **death ledger** and takes everything due — the entry, its pattern neuron and that neuron's subtree
> together, and what named them scrubbed with them. An entry retired this frame with nothing open dies on this
> frame's pass; one a standing recognition still names waits exactly as long as the stack above it needs, and
> not a frame longer. **Nothing traces who is naming what**: the machine settles the question off the board it
> already keeps.
>
> **The ledger holds the entry, not a handle to it.** A pattern's neighborhood is stated in one place, the
> parent's line for it (D10), and a claim on that pattern is expanded through that line (R32). Until the death
> frame, units at its own level still name it and units above it still cover it, so the definition has to stay
> readable after the table stops covering with it.

**A recognition survives its entry's retirement.** A standing recognition keeps the neighborhood it already
returned — read once, and the line it is stated in is in the ledger, not gone — and if that bid was promoted,
the unit above is still live and still expands through that same neighborhood (R32).

**A deletion takes the subtree, and takes it at once.** There is no staged cascade and nothing to wait on at
any level.

**Nothing is retired for having been wrong about the future.** A forward half is measured, not claimed (D18),
so an entry is never charged for what followed it — only for what it names that did not fire beside it.

# 10. The frame, per neuron

**The neuron prices, the machine holds the board.** Every test the neuron runs is its own arithmetic over its
own evidence; the one thing it cannot see is what the rest of the level already covered, and the machine hands
that over when it reports the assignment (D28).

The machine calls a neuron twice in a frame. `process frame` reaches every activation that fired **this**
frame — it covers, it returns recognitions, and after the election it bills. `process actions` runs after
every level has finished and reaches every **open** activation at whatever age it stands at; it is where
connections are recorded and inferences returned (§15, §16). Between them, the machine accrues into bins on
its own (§10.3) and does not call the neuron at all.

## 10.1 The cover

The neuron fired this frame, and the backward half of its observation, `O⁻`, is in hand — whole (D18).

> **R18 — The cover pass.** **Every activation that fired covers its own backward half**, and the neuron
> answers for each separately. The machine calls once with all of them, because they are activations of one
> type at one age differing only in position (D2, D8), and one call is cheaper than many.
> 1. **Cover.** Over the entries of the table, repeatedly take the one with the highest
>    ```
>    ( neurons of the residual it names )  /  ( 1 + | e⁻ \ O⁻ | )
>    ```
>    and take it **iff it names strictly more of the residual than its price**, ties to the older `id`. Each
>    round is measured on the residual the last one left. **Stop when no entry pays.** The result is the
>    **cover**, and every entry outside it is simply not used here.
> 2. **Assign.** A neuron of `O⁻` belongs to the entry that took it, which is the first one to name it, since
>    every round takes only out of the residual. That partition is the bin's assignment (D22), and it is what
>    each entry counts and re-centers on (D24, R4).
> 3. **Return one recognition per entry of the cover.** A recognition is that entry's whole neighborhood and
>    the id of its child if it has one, and **a recognition carrying a child id *is* a bid** (R20). An entry
>    with no child returns a neighborhood and bids nothing; the neighborhood is returned either way because the
>    machine needs it for the assertion (§13). **Creation never bids.**
>
> **This is R27, run by one chooser instead of many.** The machine resolves a frame's slots among bids that
> arrive independently, so it assigns first and lets each bid answer for what it was left. A neuron picks its
> own cover, so it takes and assigns in one motion. **The criterion is the same in both** — what a neighborhood
> covers against what it costs to state (D13) — and so is the accept test.
>
> **A neuron with an empty table covers nothing**, bids nothing, and its whole observation is residual (D23).
> That is the shortest file available to it, not a failure.
>
> **If `O⁻` has been seen before it already has a bin**, whose cover is exactly this computation (R7), and this
> is the read that settles it — nothing is stored, the cover being re-derived off the table every time the bin
> is used (R6).

## 10.2 The bill

The level's election has run, so the machine can report what the rest of the level covered (D28). **The bill
is the same call, continued**: the neuron folds this frame's observations and asks both tests.

> **R19 — The bill's pass.** Once per bill, in order:
> 1. **Fold.** The observation joins the bin its key names, **opening that bin if this is the first time that
>    context has been seen** — bins are created here and nowhere else, and destroyed in step 2 when the last
>    observation they hold is evicted. A bin that opens takes its cover from R18. The bin's count rises by one,
>    **every entry of the cover increments by the neighbors assigned to it** (R3), and the observation takes
>    the part of its adjustment that is already settled — what *this* frame's election covered. **The rest of
>    the adjustment, and the whole forward half, arrive later** (§10.3). **The fold is unconditional**: a
>    neuron another unit subsumed records exactly as one that was promoted does (R28).
> 2. **Age.** A full ring evicts its oldest observation, one out for one in (D25, R9). Whatever it had accrued
>    leaves its bin with it, and its open activation closes. **Nothing is priced here.**
> 3. **Re-center on the evidence.** Every entry the first two steps moved counts on re-centers, once, over the
>    totals they leave (R5, R6). **Once for the bill, not once per observation**, so the center does not turn
>    on which activation the inputs happened to give first — and that center is what the two tests below price
>    against.
> 4. **Add.** Grow a candidate against the residual the history is carrying — a backward neighbor at a time,
>    largest `Δ` first, until none pays (R14) — and price it (R15). If it pays, it enters the table and takes
>    its share of every bin whose cover it joins. **Grow again against what it left**, and stop at the first
>    candidate that does not clear its line.
> 5. **Retire**, at every bill and on no condition at all: a bill always moves counts, so every margin is a
>    different number than it was. The pass runs over the whole table, this bill's new entries included —
>    retire every strictly negative margin, sequentially, re-checking the rest after each (R17). Each leaves
>    the table at once, its neurons falling where D17 puts them and the neighborhood it held freezing. **The
>    neuron deletes nothing**: the machine takes the entry, its pattern neuron and that neuron's subtree on the
>    ledger pass at the death frame, which is this frame whenever nothing is open to name it.
> 6. **Re-center on the structure.** Every entry whose covered set moved in step 4 or 5 re-centers again (R5).
>    Step 3 already centered what the evidence moved, so between the two nothing is left stale.
> 7. **One request, and the bill returns.** Every candidate that survived, and every entry step 5 retired, in
>    one interaction, after both tests have run *and* the table has re-centered. A passing test **requests**
>    rather than creates, because a pattern is a symbol at the level above and that alphabet belongs to the
>    machine; the machine returns its identity and the neuron registers it as an entry. **Sending last settles
>    what the request carries**: a candidate can inherit neurons from an entry the retire pass took, so each
>    definition is the final one rather than the one the add test happened to price. The newborn is installed
>    **now, for later** (R13). Nothing else in the bill reaches another level, and the bill ends on it.

**The bill is processed after the level's election**, since step 1 reads a board that this frame's election is
the last thing to move (T10).

## 10.3 The rest of the window — accrual

**The neuron is not called.** For every open activation whose next forward frame has arrived, the machine
writes what landed into the bin that activation joined:

```
forward neighbors   the neurons at this offset go into the bin's forward tallies       D22
coverage            a neighbor of the key that an accepted bid has now taken goes
                    into the bin's covered tallies                                     D28
```

**Nothing is decided, priced or compared here**, and no test is waiting on any of it. The accrual is evidence
arriving for whatever the neuron decides next — its own next firing, or another activation's bill in the same
bin.

**Coverage settles over the window and only ever grows** (R31), which is why it accrues rather than being read
once: the last bid that can cover an activation at `g` fires at `g + reach_t` (T10), so the covered tallies
stop moving exactly when the activation closes.

**An activation closes at age `reach_t`, or when eviction closes it** (R9). Closing does nothing but stop the
accrual — there is no second bill and nothing is folded twice.

**What does happen in these frames outside this band**: an open activation records a connection to whatever
action ran, and offers an inference on the next action slot, in the pass that follows every level (R37, R42).
That is what the window is otherwise for.

# Part II — The machine

# 11. Contraction

> **D26 — Contraction.** The machine covers the level below with units from the level above, each taken when
> it covers more neurons than its bid costs to state — `1 + |e \ O⁻|`, never the dictionary line (R21, R27).
>
> **Covering everything is not the goal.** A neuron no unit covers stays in the file as itself, at cost 1
> (D11), and that is the shorter file whenever no unit could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a neighborhood names neighbors at offsets, so a promoted unit replaces a chunk of
> spacetime. Spatial contraction is the case where every offset is zero.

## 11.1 Bids

> **R20 — A bid is a neighborhood and a name.** A recognition whose entry has a child *is* a bid, and it
> carries two things and no others:
> ```
> the neighborhood   the entry's whole span, backward and forward
> the child          the id of the pattern neuron this entry would promote
> ```
> The **whole** neighborhood travels, because it *is* the dictionary line for the symbol being proposed (D10),
> and the bidder is implied, because a child *is* its parent in that neighborhood.
>
> **One activation may send several**, one per entry of its cover (R18), and they are independent bids: each
> answers for the part of the observation its entry was assigned, and the machine has no reason to know they
> came from one neuron. An entry with no child sends a recognition and no bid; a neuron covering nothing sends
> neither.
>
> **Nothing else is sent, because nothing else is the neuron's to know.** Which of the named neighbors
> actually fired, what this bid is worth against them, and what another bid has already taken are facts about
> the frame — and the machine is holding the frame. It reads the neighborhood against its own board and
> derives the rest (R21).

> **R21 — What a bid covers, and what it costs.** The neuron sends the complete neighborhood (R20) and nothing
> else. The machine holds the frame, so it reads that one object against what fired and derives both numbers.
> ```
> the bid   the entry's whole neighborhood, backward and forward, and the child's id      (R20)
>
> covers    the neurons it names that have fired — the slots it asks to subsume, the bidder among them
> price     1 + |e \ O⁻|   its own line in the history, and the neurons it names in those
>                          same frames that did not fire
> ```
> **This is D13's contribution, and it is the number routing already produced.** `covers − price` is
> `|O⁻| − d⁻`, the saving over stating the chunk flat, so a bid is worth at the election exactly what its entry
> was taken into its cover on (R18). Nothing is re-derived and no second valuation exists.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as its own
> line if nothing covers it (D26) and costs a turn-on if this unit is promoted — one symbol either way, so it
> cancels before the test begins, and charging it here would count it twice against the uncovered term of the
> same sum (§11.4). **The two-sided error is the assertion's** (D9): a slot named wrongly costs a turn-off and
> a turn-on, and that is scored forward as frames arrive (R34), not here.
>
> **The bid is whole; only the half that has happened can be settled.** Nothing at `Δt > 0` has fired, so the
> forward half of the neighborhood covers nothing and costs nothing *here* — it is not absent from the bid, it
> is unsettled. It is what the machine asserts out of the standing recognitions (§13, R32), what reality
> settles as each frame arrives (R34), and what the parent pays for at its bill (R19). **The cut is D20's
> availability, and it is a fact about the moment rather than about the bid.**
>
> **What it covers is re-measured on what the election leaves the bid; the price is not** (R27 step 2). A slot
> another bid holds is that bid's to answer for and leaves this bid's tally — but it fired, so it was never
> among the neurons named and absent. **What a neighborhood gets wrong about a frame is a fact about the two,
> and no assignment moves it.**
>
> **This is a price for one bid, not for the symbol.** The dictionary line `1 + |e|` is weighed by the one test
> (R12) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R21).

## 11.2 Slots and claims

> **D27 — Two windows, not one.** A bid at level `k` reaches `reach(k)` either way in every activation dimension
> (D15), and its two halves are different objects the machine keeps separately.
> ```
> coverage set    per level, backward    which accepted bid holds each subsumed active activation
>                 an assignment          one holder per activation; settled slots are never re-assigned
>
> assertion map   global, forward        which unit owns each base (dimension, frame, position) slot
>                 exclusive              one owner per slot, re-resolved every frame (§13)
> ```
> **A slot is named by a full coordinate**, dimension and position together, so two activations of one neuron
> at two positions are two slots and never contend. Level `k`'s coverage set spans `2·reach(k) + 1` in each
> activation dimension and ages out with it. **The machine holds nothing on the scale of the run.**
>
> Backward, the assignment is about **credit**: a neuron is a fact that needs accounting for exactly once, so it
> is settled once and never revisited (R22). Forward, it is about **truth**: two predictions of one slot is an
> ambiguity the decoder cannot resolve, so it keeps re-resolving until nothing can reach it (R24).

> **D28 — The adjustment.** At its bill, and nowhere else, an activation is told what it is not to be credited
> for. It is a lookup in the two windows of D27, made by the machine and handed over with the completed
> observation — **nothing is computed for it and nothing is held on its behalf.**
> ```
> covered      backward, off the coverage set:  neighbors of this activation's neighborhood — its own slot,
>              the one holding the neuron itself, among them — that the assignment gave to an accepted
>              bid other than this one
>
> superseded   forward, off the assertion map:  slots this activation's assertion does not own, held
>              by a level above, or by the majority claim at its own level (R24, R32)
> ```
> **Both halves are final when the bill reads them.** The slots at issue are `f+1` through `f + reach_t`, and
> an assertion reaches strictly forward (R33), so the last one that can name any of them resolved at the end of
> frame `f + reach_t − 1` — before this frame's stack runs, and before this frame's own assertion pass, which
> names `f + reach_t + 1` onward (T17). The coverage half closes on the same frame for its own reason (T10).
>
> **It accrues rather than being read once.** The coverage half settles over the activation's window and only
> ever grows (R31), so the machine writes each neighbor into the bin's covered tallies at the frame an accepted
> bid takes it (§10.3), starting with whatever this frame's own election settled. The superseded half accrues
> the same way, one slot at a time as the assertion resolves them. **Nothing waits for either**: the neuron's
> tests read the tallies as they stand, and a bin whose newest observations are still filling is a smaller
> sample rather than a wrong one (R9).
>
> **The machine reports facts, never numbers**: what the overlap is worth depends on a neighborhood that is
> still moving, so the worth is re-derived at every reading (D13) and only the fact is kept.
>
> **Every active neuron reads one, bidding or not.** Coverage is a fact about a **neuron**, so what it records
> applies to any entry later priced against that observation.

> **R22 — This frame's bids against both windows as they stand.** Against the **coverage set**: only neurons no
> earlier frame's election has assigned are in play, so a chunk already paid for is not paid for twice. **No
> earlier promotion is ever re-scored.** Re-assignment happens only inside the frame that is electing, and only
> among that frame's own bids (R27 step 3). Against the **assertion map**: a slot no live unit can still reach
> takes no more claims; an open one is held by whichever claim currently wins (§13).

> **R23 — Claims persist.** A promoted unit has spoken for the frames its neighborhood names, including frames
> ahead, and those claims stand until the span completes. So the election settles the future along with the
> present: the unit holding a slot is the one whose prediction counts there.

> **R24 — A slot's owner is whatever the resolution currently says.** §13 re-resolves every slot every frame
> from the live active set, so a unit that fires later takes a slot by winning that resolution — nothing is
> displaced, because nothing was being held.
>
> **Within a level, the collapse decides it** (R4). The population is the claims of that level landing on that
> slot; `n` is how many there are and `count(p)` how many name `p`; the slot takes `p` iff `count(p) > n/2`.
>
> **A level with no majority is silent there, and the contest passes down** (R32). The slot goes to whichever
> level first produces a majority, so the level structure chooses *which population votes*. **No tie-break is
> needed and none exists.**
>
> A slot stops changing when no live unit can still reach it. **This rule is about forward slots only** —
> backward coverage is never exclusive and never displaced (R22). A unit that loses a slot is neither right nor
> wrong there: the file says nothing on its behalf, so it pays no correction and earns no credit, which is what
> the `superseded` half of the adjustment records (D28).

## 11.3 When a slot is settled

Settlement is a property of one slot at one full coordinate. **Nothing here delays anything the machine does**
— no pass blocks on it and no decision is deferred by it. **The only consumer is measurement**: when `L` or
apex-units-per-frame is read, the settled frames are the ones whose numbers are final.

> **R25 — Settlement is a condition to detect, not a schedule to predict.**
>
> **Frontier membership settles one level, in `reach_t` frames.** Whether an activation at frame `h` is covered
> is decided by bids firing no later than `h + reach_t` (T10).
>
> **A frame's encoding settles at the top of whatever stack reached it.** A unit one level up, firing later,
> can name a lower unit that names frame `g`. **Frame `g` is settled when no level holds a live unit that could
> still join or leave that set** — a closure over the levels, evaluated upward.
>
> **`D` is reached, not known.** The walk stops where a level accepts no bids and therefore produces none above
> it, so `Σ_(k<D) reach_t(k)` bounds a condition rather than counting out a delay.

> **R26 — Best-effort promotion.** A unit is promoted on its backward match and its recognition names forward
> neighbors on faith. When the future disagrees, corrections are appended and price the completed claim; they
> do not revise the election that made it. **There is no retraction and nothing is held back waiting** — a
> wrong assertion is simply a longer file.

## 11.4 The election

**The file over one frame is the units promoted plus what they got wrong**: `Σ over the accepted (1 + |e \ O⁻|)
+ the neurons no unit covered`, the history half of `L` (D13) over the frames the election can see. **The two
terms do not overlap**: a neuron a promoted unit fails to name is in the second and not the first, which is why
the price counts only the neighbors named and absent (R21). The dictionary
half is R12's, and neither test touches the other's sum.

**That sum is the objective; R27 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — every slot resolves at once on profitability, the bids that do not clear their price are
dropped, and their slots go back to the accepted bids that named them. **The election does not minimize the
sum**: it resolves each slot locally and returns a good assignment, not a proved minimum. **Every neuron a bid
covers ends up credited to exactly one bid**, which is what stops a chunk being paid for twice.

> **R27 — The election assigns slots, then bids settle up.** Two decisions and a settling, each over every slot
> or every bid at once, and none of them runs twice.
>
> 1. **Resolve each free slot.** A slot here is one active activation of the level below, at its own full
>    coordinate — frame and position — that some bid this frame claims. Bids arrive naming relative offsets, so
>    a bid's claims are resolved against its own coordinate before this pass runs, and slots already assigned
>    by an earlier election are not in play (R22). **The slot goes to the claimant with the highest
>    `slots covered / price`**. **Ties go to the older symbol, then to the earlier coordinate** — creation order for a
>    pattern and declaration order (D1) for a base neuron, then frame, then position. **Every slot resolves
>    independently of every other.**
> 2. **Tally and test.** Each bid holds whatever step 1 assigned it, and is accepted iff it holds strictly more
>    slots than its price, **the tally measured on what step 1 left it and the price on the neighborhood
>    alone** (R21). The resolution may have taken slots away, so what a bid delivers is not what it claimed —
>    a slot another bid holds is that bid's to answer for and leaves this bid's tally, and the price does not
>    follow it, because a neuron that fired was never among the neighbors named and absent.
>
>    **The tally counts only what would otherwise be a correction.** A slot another accepted bid also names
>    would be picked up by that bid for free if this one were dropped (step 3), so it is no saving of this
>    bid's and is not in its tally. **A bid is worth what the file loses without it**, which is R12's baseline
>    and not a second rule — and the flat baseline is what that question returns when nothing else names the
>    slot (D13).
> 3. **Settle the assignment over the accepted.** A slot whose holder was rejected passes to the best-rationed
>    accepted bid that names it, by step 1's rule; a slot no accepted bid names is a correction. This decides
>    nothing: an accepted bid can only gain here, gaining cannot un-accept, and a rejected bid is never looked
>    at again.
>
> **The assignment is a partition of the neurons the accepted bids name**, and that is the whole of the
> inhibition — no bid is ever edited or forbidden, and **overlap is legal and priced**. **Held by an accepted
> bid** and **named by an accepted bid** are therefore the same set, so coverage, credit and the apex frontier
> (R30) are one question with one answer.
>
> **Outcome**: accepted bids are promoted, one unit each, the neurons assigned to them are subsumed, and every
> active neuron no accepted bid covers is a correction. **The election delivers nothing to anyone**: it writes
> the assignment and stops, and a neuron reads what it makes of that for itself when a span closes (D28).

> **R28 — Subsumption is recorded beside a neuron's evidence, never inside it.** A neuron covered by a
> neighbor's winning bid records and re-centers exactly as if the election had gone the other way, so the
> collapse stays the L1 center of what the neuron *saw* rather than of what it was elected on.
>
> **What subsumption governs is every price read against that observation.** It removes the covered neighbors
> from any entry's claim, so nothing is credited for describing a chunk the file already states some other way.
> **The observation says what was seen; the adjustment says what was already spoken for; every price is
> measured against both.**

# 12. The order of a frame

> **R29 — One stack, at the derived reach.** Base neurons build their backward halves, cover them and bid;
> contraction settles which bids propagate; the neurons then bill, reading off the settled board and minting
> children where economical. The survivors are level 1 — the fewest that cover the active base neurons — and it
> happens again. **When a level's active neurons promote no children, nothing propagates and there is no level
> above it on this frame.** Nothing declares the depth and nothing caps it.
>
> **Within a level the order is cover, election, bill**, and it cannot be otherwise: a bid is made before the
> election because that is what the election is over, and the bill comes after it because it reads a board that
> this frame's election is the last to move (T10). Nothing in that order leaves the level or the frame (T16).
>
> **All three are one call on one activation** (R13). The neuron covers, the machine elects, the neuron folds
> and restructures — and the activation that did all of it fired this frame, on evidence that was complete when
> it started.
>
> **The frame ends with the machine's own pass.** Once the last level has billed, the machine reads the death
> ledger and deletes everything due (R17), which is why a retirement is usually collected in the frame that
> made it.
>
> Every level runs the same rule at the reach one expression gives it (D15). **Compression is spatio-temporal
> at every level, in one pass**: a pattern at any level may name neighbors in its own frame, in earlier ones,
> in later ones, beside it in space, or in a mix.

> **R30 — The apex is a frontier, not a level.** It is every active neuron **no accepted bid covers** — the
> uncovered set of D28, at every level at once — so a base neuron nothing found worth chunking stands in it
> beside a level-4 pattern. This is the same frontier the file's history writes, the same one rewards credit,
> and **the same one that asserts** (R31): the uncovered set does all three, and coverage silences a neuron in
> every one of them at once. Everything underneath it is recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that fired a child and had its bid declined with nothing else covering
> it is still on the frontier, and the election charges it as a correction. Coverage is the criterion, and
> being covered is exactly what the machine reports (D28).

The frontier cuts across levels, not along one:

```
   level 3                          ┌────── ▣ ──────┐
   level 2                ┌─── ▣ ───┐               │
   level 1      ┌─ ▣ ─┐   │         │       │       │
   level 0      a     b   c         d   e   f   g   h        i     j
                                                             ▣     ▣

   frontier  =  { L1 over (a,b),  L2 over (c,d),  L3 over (e,f,g,h),  i,  j }
```

**Events and actions run in parallel within a level.** An action fires in the same column as the events it runs
alongside (D5), so it is recognized and chunked by the rule they are. **The event→action connection is not
formed here**: it names the action pattern that ends up in control of the dimension, which stays open until
every level has settled, so it is recorded in the `process actions` pass instead (R37).

# 13. The assertion

**Asserting is the machine's act and nothing else's.** A neuron returns a recognition (R18); the machine
resolves the standing recognitions into one assertion, one owner per base slot, because the file scores
corrections against an asserted set (D27).

**The electorate is the uncovered set, at every level at once** — R30's frontier. A recognition returned at an
earlier frame still stands if its span has not closed (R23), and it is withdrawn the moment an accepted bid
covers its neuron.

> **R31 — Coverage inhibits.** A covered neuron's recognition takes no part in the resolution, and neither do
> its inferences (R42).
>
> **Coverage silences a neuron in the machine, not in its own model.** The inhibited neuron still bills, still
> folds its observation, still reads its adjustment (D28), and still records its connection to whatever ran
> (R37). It is denied a vote, not a life.
>
> **Coverage is acquired late and never revoked.** An activation firing at `g` is uncovered until some bid
> takes it, and by T10 the last bid that can fires at `g + reach_t`, so the electorate over the slots it names
> only ever shrinks and stops moving when everything else about that activation does. **No accepted bid is ever
> dropped** (R22, D27).

> **R32 — Expand, then precedence.** A claim at level `k` names level-`k` units, which are not yet anything the
> file can be wrong about. Expanding a unit recovers the neighbors its neighborhood names one level down, at
> that unit's offset plus theirs — offsets compose because each is a difference of activation coordinates (D2)
> — repeated to base symbols. Every claim then has the shape `(dimension, frame, position, symbol)`.
> ```
> a's recognition names  (b, +1)                 → b's dimension at f+1
> A's recognition names  (C, +2)  expand it:
>   C names {(p, 0), (q, +1)}                    → p's dimension at f+2, q's dimension at f+3
> ```
> **For each `(dimension, frame, position)` slot, work down from the highest level that claims it. Within a
> level, the collapse over the claims landing there decides the slot (R4, R24). A level with no majority is
> silent, and the next level down resolves it. A slot no level carries is a correction.** **An action slot
> resolves by this same procedure and no other** — what executes there is chosen elsewhere and never enters it
> (R34, R42).
>
> **Precedence chooses the electorate; the collapse decides the outcome.** **Nothing here consults counts**:
> the vote counts claims, not observations, so the decoder reproduces the procedure exactly from the active
> set.

> **R33 — The assertion only reaches forward.** Expansion reaches both directions, so a claim at `+2` whose
> neighborhood reaches `−3` lands at `−1`. Claims landing at or before the recognizing neuron's own frame are
> discarded, and nothing is lost: the decoder builds each frame from what the machine asserted *before* it.

> **R34 — Events and actions, one procedure.** The rule is identical on both sides, and so is the consumer.
> **An assertion is a prediction, and what settles it is a firing** (D5). The asserted **event** set is what
> the machine expects to observe, settled by what the input reports; the asserted **action** set is what it
> expects itself to do, settled by the action that actually ran. Both are scored as frames arrive.
>
> **An action assertion commits nothing.** What runs is chosen by inference (R42), so a habit that predicts the
> machine's own behavior compresses the action stream when it is right and pays corrections when it is not.

As each frame arrives, the part of the assertion that came due is scored, on both sides alike: what it named
correctly is free, what it got wrong is written as corrections. An event slot is settled by what was observed
there and an action slot by what ran there, and neither is settled by the assertion itself (R34).

---

# Part III — Actions, reward, and selection

# 14. Actions

An action dimension carries what the machine executes, and it is compressed by the same hierarchy its events
are (D1, D5).

> **R35 — Three frames: infer, execute, reward.** What is chosen in one frame runs in the next and is paid for
> after that.
> ```
> f      infer     the frame's events are recognized, `process actions` returns the inferences,
>                  the inference resolves (R42), committing an action for the frame ahead
> f + 1  execute   the action runs, and its neuron fires in this frame's column alongside
>                  this frame's events — the connection is made here
> f + 2  reward    what the action earned arrives as input, and updates that connection
> ```
> **The connection is not a neighbor**: it crosses kinds, it is temporal only, its two ends need not sit at the
> same level, and it is formed after every level has settled.

> **R36 — Execution is an expansion**, of the selected pattern rather than the asserted one (R42). A high
> action pattern becomes its constituent actions at the distances its neighborhood recorded, down to base
> actions that execute. Execution is not a second mechanism; it is this expansion read as a program. Each base
> action executes in the frame its expansion places it in, the nearest being `+1` (R35).

# 15. Reward

A reward is an input, not a symbol: alongside what it reports observed, a frame may carry rewards for actions
already executed. They reach the machine through one object and one only — the event→action connection, which
no structural test can see (R40).

> **R37 — A connection is made when the action fires, at the distance the observer is open at.** What executes
> at `f + 1` is not known at `f`: a committed action still has to survive R32, and which action pattern ends up
> in control of the dimension is settled only once every level has run. The connection is therefore recorded at
> `f + 1`, after the stack finishes, against what actually ran — **a neuron that argued for a different action
> still learns from the one that ran.**
>
> **This is `process actions`, the machine's second call on a neuron and the only one that reaches an
> activation after its firing frame.** `process frame` reaches a neuron at age 0 alone (R13); this one reaches
> every open activation it has, at whatever age each stands at, and does both action-side jobs at once — the neuron records its
> connections to what ran and returns the inferences that contend for the next action slot (R42). **It runs
> before §13 resolves**: the connections it records are to the action already executing this frame, decided a
> frame ago, while the inferences it returns are candidates for the slot §13 is about to settle.
>
> **Every neuron active in that frame records one, at every age it is open at.** The distance is the age — the
> offset from the frame that activation opened to the frame the action ran — so a neuron open at ages 1, 2 and
> 3 holds three connections to the same action, one per distance. **Recording and reading sit one frame
> apart**: selection is choosing an action that will run *next* frame, so it reads the distance one beyond the
> age it stands at now (R42). Fan-out is bounded — a neuron connects only toward the channels its neighborhood
> already names.
>
> **Making and strengthening are one operation.** The first co-occurrence creates the connection at strength 1
> and every later one increments it; the share of reward due to it folds into the running mean when that reward
> arrives (R39), weighted by `1 / strength`, so the stored estimate is the exact average over the connection's
> exposures and no rate is chosen.
>
> **Every level holds them, not only the apex.** No operation derives what a pattern's children were worth from
> what the pattern earned, so a connection held only at the frontier would be lost at the next mint.
>
> **A covered neuron still learns.** Coverage decides which neuron speaks, not what counts as evidence: the
> action ran, the reward arrived, and that reward is a sample of what the action is worth when this neuron
> fires, whichever claim selected it.
>
> **A connection dies with either of its ends.** Nothing else removes one, and no window ages the estimate it
> carries (R40).

> **R38 — Credit lands on the apex active action, and covers its whole span.** The credited object is the
> highest action pattern in control of its dimension at the frame the action ran, not at the frame the reward
> arrives in, falling back to the base action when nothing higher covers it. Before any action pattern exists
> the apex is the base action, so the rule holds across all of development.
>
> **A pattern that occupies its dimension for several frames is credited with every reward that arrives while
> it runs.** An expansion places base actions at `+1`, `+2` and beyond (R36), so a pattern has a span rather
> than a frame, and the credit lands on the pattern rather than on any one of those frames.

> **R39 — A reward names what it pays for, and the machine fills in the rest.** A frame carries an array of
> rewards, each of them
> ```
> { reward, channels, frames }
> ```
> **`channels`** is the set of action channels the reward pays and **`frames`** the span of the window it pays
> over. **Both are optional, and an omitted one means all of them**: no channels named pays every channel, no
> span named pays the whole window.
>
> **What reaches distance `d` falls linearly with `d`**, because the further back a frame is the less likely it
> is to have caused what arrived. Over a span of `S` frames,
> ```
> share(d)  =  reward · (S + 1 − d) / S            d = 1 … S
> ```
> so the nearest frame takes the whole reward, the far end of the span takes `reward / S`, and nothing along
> the way takes nothing. A span of one frame is the degenerate case, `share(1) = reward`.
>
> **Nothing is being divided up.** Every distance in the span is paid, and the shares are not a partition of
> the reward — a reward is not in short supply, and the point of spreading it is not to conserve it but to say
> how likely each frame is to have earned it. What is genuinely responsible recurs and accumulates; what is not
> is sampled once and averaged away (R36). Each share goes into the running mean of the
> connections pointing at the apex action of a paid channel at that distance (R37, R38).
>
> **Rewards in one frame are independent.** A connection at one `(channel, distance)` takes the sum of the
> shares that reach it, and nothing coordinates one reward with another.
>
> **The unscoped form is the general case, and the machine sorts out the attribution itself.** An environment
> that knows what it is paying for names it — a stock channel, and the one frame the buy or sell ran in — and
> the credit is exact. One that does not names nothing, and the reward spreads over every channel and the
> whole window: the shares landing on a channel or a distance that had nothing to do with the outcome are
> noise around zero and average out over exposures, while the shares landing on the ones that did accumulate.
> **No structure is priced on either** (R40), so a coarse reward costs accuracy in the estimate and nothing
> else.

> **R40 — Two objectives, meeting at one place.** Everything structural is priced in file length; reward prices
> nothing structural and cannot. The machine runs **two** objectives: compression, which decides what structure
> exists, and reward, which decides which of it is executed. They meet at exactly one place, the event→action
> connection, and those connections are not in the file and not priced by the one test.
>
> **Connections are not forgotten.** A connection's reward is the plain average over its exposures, with no cap
> and no decay.

# 16. Selection

> **R41 — Selection.** No fit says which action to take; it says only what an action set looks like. Choosing
> comes from the connections held toward the action patterns that have followed, each carrying the reward that
> arrived averaged over its exposures, and **the machine executes the best. Nothing else decides it.**
>
> **A situation is a set of active event patterns** — any one of them, and any subset of them. Situations
> intersect, and **nothing ever materializes one**: a situation is what a voter fires in, never an object the
> machine holds.
>
> **A voter is one active neuron at one age**, and a voter is what holds connections and offers an inference.
> Its estimate is a marginal over every situation it has fired in, so the same frame reads differently to a
> base neuron, to the level-4 pattern covering it, and to that same neuron two ages later. R42 reconciles
> them, and **a minted pattern is how one recurring situation acquires an estimate of its own** — which is why
> a covered neuron is silenced (R31) and its coverer speaks instead.
>
> **The asymmetry is in the targets, not the holders.** Any active neuron holds connections, an action neuron
> included; what an action neuron may not do is point one at an event. **Events infer actions and actions never
> infer events.** An action's connection to a following action carries an ordinary estimate and selects on
> equal footing with an event's.
>
> **The bootstrap is a connection, not a fallback.** Every neuron is born connected to the declared default
> action at every distance it can vote at, at neutral reward, so there is no separate no-history path. It is
> the only action wired in advance; R43's walk supplies the rest, one at a time and only where the default has
> been judged and found wanting.

> **R42 — The inference decides, the assertion predicts.** Neurons offer recognitions and inferences on an
> action slot, and the two are resolved separately. **The recognitions collapse into an assertion**, which says
> what the machine expects to do and is scored against what it did (R34), resolving the way every assertion
> resolves (R32). **The inferences decide what actually runs**, and nothing about the assertion enters that.
> Neither side supplies the other's candidates — the recognitions come from what the action hierarchy
> recognizes, the inferences from the connections the voter holds — which is the whole of it, because R43's
> walk enters by creating one.
>
> **A selected pattern stands until its span runs out.** Execution is an expansion (R36), so choosing a
> pattern at `f` places base actions at `f+1`, `f+2` and beyond — and those placements are **standing
> inferences**, contending for their slots exactly as this frame's fresh ones do. They are resolved by the same
> two keys below: **a plan holds because it keeps winning**, a higher-level selection displaces it, an
> equal-level one with a better estimate displaces it, and when its span ends it is simply gone. Nothing is
> retracted and nothing is held.
>
> **A standing inference is not an assertion.** It is sourced from connections, decides what runs and is not
> in the file; the assertion is sourced from recognitions, is scored, and decides nothing (R34, R40). Both
> stand for the span of the pattern that produced them and both re-resolve every frame, and neither is ever
> read by the other.
>
> **The electorate, explicitly.** At frame `f` the voters entitled to an inference on the slot `f + 1` (R41)
> are:
> ```
> every standing inference that places a base action at f + 1                          R36
> plus  every open activation the machine holds at f  one per (neuron, age, position)  (D21)
>   less  every activation an accepted bid covers                                      (R31)
>   less  every activation at age reach_t            it will not be open at f + 1
>   read as (neuron, age)                            position carries no connection    (D8)
> ```
> Each reads its connections at **distance `age + 1`**, the distance at which it will stand from the action it
> is choosing (R35). An activation at age `reach_t` has no connection to read, which is why the distances a
> neuron votes at run `1` through `reach_t` and the bootstrap fills exactly those (R41). Every connection at
> that distance is one inference, naming an action pattern and carrying an estimate.
>
> **Position drops out.** Two activations of one neuron at two positions are one type (D8) reading one set of
> connections, so they offer the identical inference and the argmax is indifferent to the duplicate. Two *ages*
> are two voters and do not collapse together — they read different distances, so they can name different
> actions.
>
> **Inferences resolve by level, then by estimate.** An inference's level is that of the action pattern it
> selects, so R32's precedence carries over unchanged: work down from the highest level that reaches the slot,
> then expand to base actions (R36). **Several inferences at one level take the largest estimate.** **Nothing
> corrects for how many exposures an estimate rests on**, so a sharp estimate on three exposures outranks a
> coarse one on two hundred.
>
> **Selection is never empty.** Every neuron is born connected to the declared default at every distance it can
> vote at (R41), so some inference always reaches the slot.
>
> **A covered neuron supplies neither** (R31). A new pattern therefore starts with no estimate and explores.

> **R43 — Exploration.** The default policy resolves explore–exploit without randomness: **the action alphabet
> is declared in order**, and **a connection whose estimate turns negative wires the next action in that
> order** — the first one the voter holds no connection to at that distance — at neutral reward.
>
> **The walk is in the same currency as everything else**: an untried action becomes a candidate by becoming a
> connection, so R42 enumerates it like any other and needs no second source of inferences. The trigger is one
> connection at one distance, never a reading over the voter's whole set.
>
> **A reward is signed, and zero is where everything starts.** The environment reports an action as good or bad
> — strictly greater than zero or strictly less — and zero is neither. It is also the estimate every connection
> is created at, the declared default's included, which is what makes the walk work: a connection at zero has
> not been judged, so it outranks anything negative and yields to anything positive.
>
> **The walk ends when the alphabet does.** Once a voter holds a connection to every action in the channel at
> that distance there is nothing left to wire, and selection takes the largest estimate, which is the least
> bad.
