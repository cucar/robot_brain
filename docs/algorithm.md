# Universal Compression with Actions and Rewards (UCAR)

UCAR is a machine that compresses what it observes by building a hierarchical dictionary of patterns, and
learns what to do by observing rewards. It is defined by two alphabets, like a Turing machine — the **event
alphabet** it can observe and the **action alphabet** it can execute — and above each it forms symbols of its
own. Every symbol, base or learned, event or action, is a **neuron**.

It has two inputs and two outputs. In: the events observed and the rewards earned. Out: the events it expects
next and the actions it executes. Both outputs are written in the base alphabets.

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
δ          a raw coordinate difference, before D15 resolves it to an offset

reach(k)   the reach at level k, derived                                        D14
reach_t    the time dimension's reach at the neuron's own level — the window    D14
W          the frame buffer's depth, reach_t + 1 — 2 at the base                D14

O          the neighborhood one activation observed, both halves                    D13
O⁻         its backward half, complete the frame the neuron fires              D19
C          a candidate neighborhood, the collapse over its seed's population   R14
nbhd(e)    a pattern's neighborhood, |e| its size — backward neighbors only     D18
covers     the neurons a neighborhood names that fired                         D16
price      1 + the neurons it names that did not                               D16
residual   the neurons of an activation no pattern covers                           D16
n          the size of a collapse population                                   R4

L          the file length                                                     D12
D          the highest level the stack currently holds                         R17, R24
H          the history size                                                    D24
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
> dimension they share** (D1, D14). It is a conjunction, so a neighborhood is a box.
>
> **Nothing declares which channels may see which**: what a channel is laid out over settles it. Two channels
> sharing only time are related in time alone, and a channel whose one activation dimension is time — a stream
> of characters — stands in no spatial relation to anything.
>
> Neighbors are always at the neuron's own level, since a neighborhood is written over the symbols that level
> offers. **The rule is identical at every level**; what differs is the reach (D14).

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
> **Above the base it does not hold.** A neuron covers one activation with a *set* of patterns (R18), so several of
> its children may be promoted at one coordinate, and a `(dimension, position)` at a pattern level can carry
> several activations in a frame. D15 already admits several neighbors at one offset; this is the same situation,
> and `|e|` counts them the same way.
>
> **A neuron may fire many times in one frame**, once per position, and those activations are instances of one
> type (D2).
>
> **A frame's two halves are simultaneous.** What is observed and what is executed occupy the same column of
> the grid, so an action runs alongside that frame's events and both are in hand together (§14).

> **D6 — An activation stays open, and what it does while open.** An activation at frame `f` remains open through
> `f + reach_t`: what follows it fires one frame at a time through that frame (D17), and an action's reward
> arrives in the frame the action runs in (R29), so nothing lands after the last of them.
>
> **Nothing the neuron decides waits for that.** Its backward half is complete the frame it fires (D19), and every
> structural act — which patterns cover it, which are offered, what is added, what retires — happens then and there
> (R13). **An activation is open for two reasons, and neither is a decision about structure.** It **connects**: each
> neuron of its level that fires while it is open strengthens its neuron's connection to it at the offset its age
> names, and the rewards that follow the actions among them move the estimates on those connections (§10.3). And while
> it stands on the apex (R26) it **speaks**: it expects from its neuron's connections (§13) and votes on the next
> action from the same connections (§16).
>
> **The machine holds the open activation.** The arriving neighbors are its own frame data on the way in, and
> it delivers them to the neuron once per frame (§10.3).
>
> **Age is per activation, and a neuron carries several ages at once.** An activation's age is the frames
> elapsed since it fired, `0` through `reach_t`, and it is the machine's counter. Nothing distinguishes
> activations but age: a new activation is the one whose age is 0. **Age is read, not just counted** — it is
> the offset at which the activation strengthens connections and reads what comes next (R31, R36).

> **D7 — Exclusion is per level, and about coverage.** A neuron an accepted bid covers is silenced in the
> machine: it does not stand in the file, it does not predict and it does not vote (R26). That is the whole of
> inhibition, and it bounds nothing about how many neurons fire at a coordinate.
>
> **The design's only exclusivity is credit, and it appears twice.** At the election, every neuron the accepted
> bids name is paid for once, credited to exactly one bid (R23); inside a neuron, every present neighbor of an
> activation is credited to exactly one pattern of its cover (R18). Both are partitions of credit, and **neither is
> a limit on naming**: two accepted bids may both name one neuron and both expand to it. The file pays for it
> once, and the second bid is neither credited nor charged for it (R21).

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what the decoder
assumes for anything the file does not state.

> **D8 — Identity is absolute.** A neuron's identity is fixed entirely by its neuron dimensions, so nothing
> about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: their activations carry the same relative neighborhood, so the same patterns cover them and one
> pattern serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

---

# 2. The objective: it is all one file

Write every frame observed so far as a single file a decoder could read back to reproduce each of them
exactly. **The file is the run.** There is no window and no horizon: nothing is dropped from the objective
because it happened long ago, and everything the machine's structure is worth is measured against the whole of
what it has seen.

> **D9 — The file.** Two parts, both over the whole run. **The dictionary**: one line per pattern, its
> neighborhood — the backward neighbors that define it (D18). **The history**: the apex units (R26), each
> followed by the neighbors it names that did not fire, and the neurons no unit covers, each standing as
> itself. Anything unstated is silence.
>
> **It is the current optimum encoding of the run.** The file is re-derived from the structure as it now
> stands, exactly as a price is (R2), so no past decision has to be honored and nothing is retained that the
> machine could not expand.
>
> **Nothing ever materializes it.** The coverage set spans `reach(k) + 1` frames (D26) and a neuron's history
> is `H` activations deep (D24), so no pass anywhere walks the run.
>
> **The file holds nothing about the future.** A neuron's connections — what followed its activations — are in
> no dictionary line and not in the history (D17). What the machine expects to see next is its output
> (§13), and nothing inside the machine scores it.

**One file, one dictionary.** A neuron's history (§7) is its own connections of what it saw. A pattern in a
neuron's table is a **standing claim on a dictionary line**, and whether that claim is honored is settled
where the file is (R12).

> **D10 — Prices.** Every cost in the design is part of a file, counted in symbols:
> ```
> activating an apex unit  =  1                 a line in the history
> what it got wrong        =  the neighbors it names that did not fire, one each
> a neuron no unit covers  =  1                 its own line
> having a pattern         =  1 + |e|           a line in the dictionary, |e| its backward neighbors
> ```
> This is a fixed-length code: a symbol costs one regardless of how often it is used.

> **D11 — What the file holds.** The dictionary, and the history encoded against it. A pattern's neighborhood
> must be written: it is the collapse of activations no ring still holds, so nothing in the file recovers it.
> **A symbol is not one line per frame** — a unit firing at `h` names `[h − reach_t, h]`, so writing it
> discharges up to `reach_t + 1` frames of the run at once. What the machine holds and the file does not is
> search state — counts, tallies, estimates, margins — because expanding an apex unit needs the neighborhoods
> and nothing else.

> **D12 — File length, and what a neighborhood is worth.** Over the run the file is
> ```
> L  =  Σ over the dictionary  ( 1 + |e| )                     written once
>    +  Σ over the apex units  ( 1 + the neighbors named and absent )
>    +  the neurons no unit covers, one each
> ```
> summing D10's prices over what D11 says the file holds. **There is one `L`**, and every neuron's structure is
> priced against it.
>
> **`L` grows without bound and nothing in the design ever computes it.** No structure holds a file length and
> no test compares two of them. Every quantity actually used is a **difference** in `L` — whether the file is
> longer or shorter for holding one neighborhood — which is finite however long the run is, and local: a count
> of neighbors over the activations that neighborhood covers (D16).
>
> **What one neighborhood is worth over one activation.** It accounts for the neurons it names that fired, at the
> cost of its own line and the neurons it names that did not:
> ```
> margin(e, O)  =  covers  −  price  =  |O ∩ e|  −  ( 1 + |e \ O| )
> ```
> **A neuron `e` does not name is not `e`'s business.** It costs one symbol whether or not `e` exists — its own
> line if nothing covers it, a turn-on if something does — so it is on neither side of this. **Naming a
> neighbor wrongly is what is charged; leaving one out is not.**
>
> **This is the only valuation in the design, and it has two readers.** The neuron reads it over its own `H`
> activations to decide which patterns to hold and which to offer (R12, R18); the machine reads it over the window
> to decide which bids to buy (R21, R23). They are one expression over two populations — the neuron's activations
> against the machine's board, where neighbors earlier frames already paid for are not in play (R22) — so **the
> two numbers differ, and are meant to.** Neither is ever handed to the other: the neuron is not told what the
> machine paid for, and the machine never reads what the neuron priced.
>
> **Every term is measured now** (R2): `nbhd(e)` is wherever re-centering has put it, and what it covers
> follows. **§8's test is the derivative of `L` with respect to holding one pattern**, read over the activations
> the neuron holds.

---

# 3. A frame, in outline

Nothing below is stated here: this is the order the rest of the document is read in, and every step names
where it is specified. A frame arrives carrying what each event dimension observed, what each action
dimension is executing (D5), and any rewards for actions already run (R33). The machine works **up one stack,
a level at a time**.

```
per level, in this order and no other

  bill       each neuron that fired covers this frame's activation and saves it, re-centers, builds
             one candidate and retires one pattern                                              R19
  offer      and returns, in the same call, a bid for every pattern that applies to the
             activation, and its requests                                                           R18
  election   the machine takes bids one at a time by what they cover per line they cost, each
             credited the free neurons it names, until the best left covers no more than it
             costs; the level's uncovered neurons stand as themselves                           R23

  the level above is built out of what the election accepted, and it happens again;
  a level that accepts nothing has no level above it this frame                                 §12

then, once the last level has run

  ledger     delete every pattern the bills retired whose child has nothing open                §9.2
  forward    every open activation strengthens its neuron's connections to what fired, at the
             offset its age names, and moves an estimate by any reward at its distance; the
             ones on the apex return what they expect and what they infer, each in its own
             level's alphabet                                                                  §10.3
  predict    the expectations expand to base events and resolve one per event dimension —
             the machine's first output                                                         §13
  select     the inferences expand to base actions and resolve one per action dimension —
             its second                                                                         §16
```

The last pass commits the action for the frame ahead; the reward for it arrives with that frame (R29).

**Every decision above is made on complete evidence.** A neuron's backward half is whole the frame it fires
(D19), and nothing structural reads anything else — so no step is a bet, nothing is committed early, and
nothing is revisited. What the remaining frames deliver is what the situation was followed by, and it is read
only when a unit on the apex is expanded.

# Part I — A neuron

# 4. Neighborhoods and distance

## 4.1 The neighborhood

> **D13 — Neighborhood.** An activation observes the active neurons of its own level that adjacency admits
> (D4), each tagged with its **offset** — the difference of activation coordinates, one component per
> activation dimension the two share (D1, D2):
> ```
> O = { (p, −4), (a, −3), (r, −2), (i, −1), (␣, +1) }        a stream:  one component, time
> O = { (k, 0, −1, 0), (k, 0, +1, 0), (m, 0, 0, −2) }        an image:  three, time and two axes
> ```
> the first for a neuron `s` in a stream reading `p a r i s ␣`. At temporal offset 0 a neighbor co-occurs, at
> negative offsets it led here, at positive offsets it followed; **all three are the same kind of thing**, and
> a spatial component is one more of the same. **A neighbor is a neuron at an offset.**
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other
> at a nonzero spatial offset. At the base, offset zero in every component is the activation itself, and that
> is the center; above the base several children promoted at one coordinate (D5) are each other's neighbors at
> offset zero.
>
> **One neighborhood per activation, and a set of patterns covers its backward half** (R18), each answering for
> the neighbors it was assigned and none of them for the rest (D16).

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
                   whole at age 0          fires over the next reach_t frames
                   covered, priced,        strengthens connections —
                   added and retired here  never saved, never priced, never in the file    D17
```

> **D14 — Radius and reach.** The radius is **2**, in every activation dimension of every channel (D1).
> Nothing declares it. The reach it gives is **1** either way along every dimension at the base, and it
> **doubles every level**:
> ```
> reach(k)   =   2^k          every activation dimension
> reach(0)   =   1            the base
> ```
> **No level declares anything, and there is no rate to choose.**
>
> In time, `W = reach_t + 1` is the depth of the frame buffer — 2 at the base. The buffer is a sliding window:
> an activation sits at its newest edge when it fires, with `reach_t` frames of context behind it, and its
> backward half is read there and never again.
>
> **The reach and the granularity are the same power of two.** D15 resolves offsets in base 2, so a level
> reaching `2^k` names its outermost offsets in groups of `2^k`.

> **D15 — Offsets are resolved logarithmically.** An offset component is the coordinate difference **kept to
> one significant digit in base 2**, the radius (D14):
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
> dimension can fall inside it. A count is per `(neuron, offset)` (D23) and `|e|` counts every neighbor named
> there. Above the base a `(dimension, position)` may itself carry several activations (D5), which this handles the
> same way and needs no rule of its own.

## 4.2 The fit

> **D16 — What a neighborhood accounts for, and what it costs.** A pattern `e` measured against an activation `O`
> gives three sets, and every price in the design is counted off them:
> ```
> covers    O ∩ e         the neurons it names that fired — what it accounts for
> price     1 + |e \ O|   its own line, and the neurons it names that did not fire
> residual  O \ (the union of the patterns covering O)   the neurons nothing accounts for
> ```
> **The residual is not an error.** Each of its neurons stands in the file as its own line, at cost 1 (D10),
> exactly as it would if no pattern existed — so it is charged to no pattern and credited to none (D12).
>
> **An activation is covered by a set of patterns, not by one** (R18). They partition what they cover: a neuron is
> `covers` for exactly one of them, so nothing is accounted for twice. What is left over is the residual.
>
> **The cost of an activation** is therefore what its cover costs plus what nothing covers:
> ```
> cost(O)  =  Σ over the patterns covering O ( 1 + |e \ O| )  +  |residual(O)|
> ```
> and an activation nothing covers costs `1 + |O|`, the whole chunk stated flat.
>
> **Distance is a reading of the same three sets.** Where one pattern is measured against the whole of an
> activation, `d(O, e) = |O △ e|` and `margin = |O| − d`; that is the identical number D12 gives, written against
> a flat baseline instead of a subset one. **The design uses the subset form everywhere**, because an activation's
> cover is a set and only the subset form adds up over one.

> **D17 — The backward half decides; the forward half is what followed.** Cut on the temporal component alone
> (D19).
> ```
> O⁻   Δt ≤ 0    complete the frame the neuron fires   every structural decision is made on it
> O⁺   Δt > 0    arrives over the next reach_t frames  what the situation was followed by; it decides nothing
> ```
> **The cut is availability, and it is not a compromise.** `O⁻` is whole when the neuron fires, so covering it,
> pricing it, adding and retiring are all done on complete evidence and none of them is a bet. `covers`,
> `price` and `residual` are read over `O⁻` and over nothing else.
>
> **What the forward half is for.** It is measured, never chosen, and never saved. Each neighbor in it strengthens
> one of the neuron's **connections** as it fires (D20). A connection is a distribution, not a record: per
> `(neuron, offset > 0)`, how often that neuron has followed, and for an action what it earned (R31), and nothing
> about any one activation is kept. It is not in the bid (R20), not in any dictionary line (D10), and it enters
> no test. Connections are read in one place — **when the neuron stands on the apex** (R26): its event connections
> are what the neuron expects (§13), its action connections are the inferences that choose the next action (§16).
> A child neuron fires only when its parent's pattern was bought, so a child's connections are the future of that
> situation and nothing else, forming from the frame after its mint on (R13); a base neuron's are the future of
> the symbol over every situation it fires in. Either speaks only when nothing more specific covers it (D7).
>
> **The cut is on time alone.** Spatial components never enter it: a neighbor three positions to the right
> arrives in the same frame as one three positions to the left, so both are in `O⁻`.

> **R1 — One comparison.** There is no second one. An activation is covered on `O⁻`, priced on `O⁻`, and the tests
> that add and retire read the same numbers over the history. Nothing anywhere compares a neighborhood against
> a forward half, and no quantity in the design waits for one.
>
> **What arrives later is evidence, not a verdict.** The forward neighbors and rewards strengthen the neuron's
> connections (D6), which are read by the next expansion of a unit on the apex, never by a test.

> **R2 — A price is a measurement, not a record.** What an activation costs is read off its cover as that cover now
> stands (D16), and the cover can change: re-centering (R5) moves a neighborhood, which moves what it covers in every
> activation, and those are what the activations then cost. **An activation is fixed but its cost is not**, and it
> stops moving when its cover stops moving.

# 5. State

Only three things a neuron holds cannot be recomputed: the history of what it saw, the patterns it has
decided to keep, and its connections, what followed it. Everything else it holds is a total.

> **D18 — Observed and named.** Two things have the shape of a neighborhood and must not be confused. Both
> span the whole box D4 admits, and D16 measures one against the other.
> ```
> an activation             what the neuron SAW where it fired; saved once, evicted `H` activations later
> a pattern's neighborhood  what it NAMES; the collapse of the activations it covers (R4), moving as that moves (R5)
> ```
> An activation is a fact, a neighborhood a claim. Neither is a frame: a frame is one column, an activation the
> whole window. **A pattern's neighborhood is its backward half only** — the dictionary line — and a pattern has no
> forward half. What followed is the neuron's connections (D20).

> **D19 — Halves.** Cut at the frame the neuron fires, on the temporal component alone. The **backward half** is the
> neighbors at `Δt ≤ 0` and the **forward half** those at `Δt > 0`. An activation has both; a pattern names only
> the first; the neuron's connections are the distribution of the second over every activation it has ever had; and
> D17 says what each is for.

> **D20 — Neuron state.** `°` marks a total: recoverable by a walk, kept to avoid one.
> ```
> neuron           = (coordinate, patterns, history, connections)
>
> pattern          = (id, neighborhood, child, counts°)
> neighborhood     = the backward neighbors it names — the dictionary line
>
> history          = the last H activations, oldest first
> activation       = (position, backward half, cover°, assignment°)
>
> connections      = event connections    per (neuron, offset > 0), a strength                   R31
>                    action connections   per (action neuron, offset > 0), (strength, estimate)  R31
>                    a distribution, strengthened one exposure at a time; nothing per activation  §10.3
>
> held by the machine, not the neuron (D6):
> open activation  = (the activation, its age)             one per (neuron, age, position)
> ```
> An `id` is creation order, a handle that survives re-centering and the tie-break R18 and R23 reach for.
>
> **No activation carries a frame number**, and nothing anywhere holds absolute time: an open activation's `age`
> is a counter.
>
> **An open activation carries no commitment.** It holds its backward half and nothing else, because
> everything it was going to decide was decided at age 0 (D6). What it does for the rest of its life is strengthen
> its neuron's connections, and speak from them while it is on the apex.
>
> **Every neuron holds connections, base neurons included.** They are the distribution of what has followed its
> activations and nothing else: an open activation strengthens one each time a neuron of its level fires at its
> age (§10.3), and nothing ever weakens one (R31). A pattern holds none. A child neuron's connections are the
> future of the situation its parent's pattern names, because the child is open nowhere else (D17, R13).
>
> **An activation's cover is held, not derived on demand.** It is chosen when the activation is saved and replaced
> only by a strictly cheaper one (R6), so two activations with identical backward halves can be covered differently,
> and the history carries each cover with its activation.

> **D21 — What the totals owe.** In dependency order, so the list also says what to recompute when something
> moves.
> ```
> activation.cover          =  the patterns covering its backward half, chosen by R18 and held by R6
> activation.assignment[n]  =  which pattern of the cover holds present neighbor n     a partition of O⁻

> pattern.counts        =  Σ over the activations it covers: its share and the residual as       backward only
>                          `present`, what other patterns of the cover hold as `held`      D23
> ```
> **A pattern is credited only its own share, but tallies everything it could still name.** A neighbor another
> pattern of the same cover holds is that pattern's evidence, not this one's, and is counted here only as an
> abstention (`held`); a neighbor in the residual is nobody's yet, and is evidence for every pattern of the
> cover.
>
> **Connections are not a total and belong to no pattern.** Nothing held recomputes them: every activation of the
> neuron strengthens them, whichever patterns covered it (R31). Several children promoted at one coordinate are
> several neurons, so each holds its own connections from birth; while they are always bought together their
> connections agree, and they diverge the first time one is bought without the other.
>
> **The tallies are sparse**: indexed by the neighbors actually seen, not by everything the box admits.

> **D22 — The residual is not a pattern.** What no pattern covers is not routed anywhere and has no line to
> pay: it is a set of neurons, each standing in the file as itself (D16). **There is no default pattern, no
> fallback and no empty neighborhood** — a table may be empty, and an activation it covers nothing of costs
> `1 + |O⁻|`, which is what an uncompressed chunk costs.

# 6. Counts, the collapse, re-centering

## 6.1 Counts

> **D23 — Counts.** Each pattern keeps, over exactly the activations it covers, two tallies per `(neuron, offset)`,
> both backward only and both sparse:
> ```
> present(p)   the covered activations in which p fired and no other pattern of that activation's cover holds it —
>              its own share (D21) plus the residual
> held(p)      the covered activations in which another pattern of the cover holds p
> ```
> and `n`, the number of activations it covers. **These are exactly what the collapse reads** (R4): at slot `p`
> the population is `n − held(p)` and the count is `present(p)`. A pattern therefore tallies neighbors it does
> not name, because whether it should name them is the question re-centering asks (R5), and a slot only the
> residual has ever held is how a pattern grows.

> **R3 — What moves counts.** Three events, and every one of them moves a whole activation's worth.
> ```
> an activation is saved          every pattern of its cover adds the activation's contribution — one to `n`,
>                                 `present` for each neighbor it holds or the residual holds, `held` for
>                                 each neighbor another pattern of the cover holds
> an activation is evicted        every pattern of its cover subtracts the same; no connection is
>                                 touched (R31)
> an activation's cover changes   every pattern of the old cover subtracts the activation's old contribution,
>                                 every pattern of the new cover adds its new one
> ```
> A cover changes when the table changes under it — a pattern added, retired, or re-centered so that R18 reads
> the activation differently — and only when the new cover is cheaper (R6). A pattern that stays in a changed cover
> still subtracts and re-adds, because what the other patterns took or gave back moves its `present` and `held`.

## 6.2 The collapse

The collapse is the only operation anywhere that decides what a neighborhood names.

> **R4 — The collapse.** Over a population that each has something to say about one neuron at one offset, let
> `n` be the size of that population and `count(p)` the number of it naming `p` there.
>
> **Backward, `p` is taken exactly when `2 · count(p) > n + 1`.** Naming it covers it in `count(p)` activations,
> states it wrongly in the other `n − count(p)`, and costs one in the dictionary line (D10), so naming it
> shortens the file by `2 · count(p) − n − 1` and it is taken when that is positive. **At `2 · count(p) = n + 1`
> the slot keeps what it had**: a pattern that names `p` keeps it, one that does not leaves it out, and a
> candidate, which has nothing, leaves it out. Nothing is divided, and over a fixed population nothing flip-flops
> at the boundary. **The ring is not fixed**: an activation enters and one leaves at every bill, so a slot whose
> count sits at the boundary follows them, and naming it raises the pattern's price wherever it is absent, which
> can cost the pattern a cover and re-decide every other slot on the smaller population. Each step is the right
> response to the evidence, and what it adds up to on stationary input is flicker around a fixed point, confined
> to slots at the boundary — an amount that is measured, not proved.
>
> **Connections are never collapsed.** No line is paid for one (D9), so nothing has to be made into a set: a
> connection keeps how often its neuron followed, and for an action what it earned (R31), and prediction and
> selection read the whole distribution (§13, R36).
>
> **An activation abstains only where the answer is already settled for it.** Backward, a neighbor another pattern
> of the same cover holds is out of the population at that slot entirely. Naming it would move nothing — it is
> already accounted for, so this pattern gains no `covers` by naming it, and it fired, so this pattern pays no
> `price` for not naming it (D16). The slot's population is the activations where that neighbor was in the
> residual or was not there at all, and the majority is over those: for a pattern, `n − held(p)` activations and
> `present(p)` of them naming it (D23). **A pattern therefore grows into the residual and never into ground
> another pattern holds**, and two patterns covering the same activations cannot converge on one neighborhood.
> Otherwise an activation held that neuron at that offset or it did not, and either way it is in the population.
>
> **This is the only abstention in the design, and it is per slot rather than per activation.** An activation still
> says something about every other offset; it is silent only where the question has already been answered for
> it.
>
> **Two populations.** One arithmetic, twice, and nothing else in the design decides what a neighborhood
> names.
> ```
> a pattern's         the activations it covers, less those abstaining    R5, at every bill      backward
> a candidate's       the activations whose residual holds its seed      R14, once per bill     backward
> ```

**One denominator, every offset.** Every activation a pattern covers has something to say at every offset — a
neuron or a silence — so the outermost offset is decided by the same population as offset 0, less only the
activations that abstain there. **No threshold, smoothing or probability estimate enters any of this**, and no
denominator is ever shared between two populations.

## 6.3 Re-centering

> **R5 — Re-centering.** A pattern **re-centers** by running the collapse (R4) over the activations it now covers,
> and the activations it covers then re-derive their covers with it (R6).
>
> **A pattern re-centers whenever its counts move** — no test, no gate, and free, because the counts are
> already maintained. Counts move only at a bill (R3), and every move there disturbs a whole activation's worth, so
> a re-center is always over all offsets and never per activation.
>
> **A bill re-centers once**, after the activation is saved and before the tests (R19). What the tests then move
> — a candidate joining covers, a retired pattern leaving them — is re-centered at the next bill, so the center
> never turns on the order two moves happened to run in.

> **R6 — Covers are held, not patched.** A moved neighborhood changes what it covers. What is maintained is
> **one pattern's covers-and-price against every activation**: a pattern that re-centers recomputes those, and
> nothing else is repaired. An activation whose table changed under it — a pattern re-centered, added or retired —
> re-derives its cover by R18 over its backward half, **and the re-derived cover replaces the one it holds only
> when it is strictly cheaper** (D16). R18 is greedy, so re-deriving can cost more than what stands; holding
> the cheaper is what makes every move a descent (R12). A retired pattern leaves every cover it was in at once,
> and the cover without it is the one the re-derivation has to beat. **A pattern that was just added gives an
> activation three options, not two**: the cover it holds, that cover with the newcomer appended and taking the
> residual it names, and the cover re-derived from scratch — and the activation takes the cheapest. The appended
> cover is what R15 priced, so what adding the pattern realizes is never less than what the test counted.
>
> **An activation whose cover has changed takes its counts with it** (R3), so the pattern that received an
> activation's share is always the pattern that gives it back.

**Cold start is silence.** A pattern with no activations has no counts and no neighborhood, and a neuron with an
empty table covers nothing and bids nothing.

# 7. The history

> **D24 — History size.** A neuron's history holds its last `H` activations and no more; its connections are
> not in the history and `H` does not bound them (R31). `H` is declared once for the
> machine and is the same for every neuron; it counts that neuron's **own activations**, not a stretch of run, so a
> neuron that fires constantly and one that fires rarely weigh their patterns against the same amount of
> evidence. **The ring is exactly `H` deep once filled**, and how much run it spans is whatever that neuron's
> rate makes it. Nothing else anywhere is measured in frames.

> **R7 — The table needs no rule against duplicates.** Two patterns with equal neighborhoods present R18 with
> identical input. It takes the older first, so the older covers everything the younger would, the younger is
> assigned nothing anywhere, and a pattern covering nothing fails R17. The tests remove them.

> **R8 — The ring makes eviction exact.** Removing the oldest activation means subtracting the neighbors *it*
> contributed, which a tally cannot recover, so each activation keeps its own backward half and a pattern's counts
> are the cached aggregate over them. Eviction reads a backward half whole; everything else reads it per slot,
> off the counts.

> **R9 — Aging is by count.** The ring is a FIFO `H` deep: an arriving activation evicts the oldest, and only then.
> Nothing compares a frame number, nothing accumulates arrears, and nothing sweeps the population per frame — a
> neuron that does not fire evicts nothing.
>
> **An activation enters the ring whole and never changes.** Its backward half is complete at age 0 (D17), and that
> is all the ring holds of it: what follows the activation strengthens the neuron's connections and is saved
> nowhere (R31), so eviction touches no connection. **Eviction does not close the activation** — the machine holds
> it (D6), and it keeps strengthening connections and speaking from the apex until its own window ends.
>
> Recording is unconditional, and **no election outcome ever edits a history** — deletion is the one thing that
> reaches into a saved activation, and it only removes names of neurons that no longer exist (R17).

> **R10 — Free parameter: the history size `H`.** It is the only one. The alphabet — channels, dimensions,
> resolutions — is the problem statement rather than a knob; adjacency is not declared (D4), the reach per
> level is not declared (D14), and neither is the offset alphabet (D15). **Nothing else in the design is
> tuned, and nothing anywhere is capped.**

> **R11 — `H` and the reach constrain nothing in each other.** `H` counts activations and the reach sets how wide
> one activation is. **What the two share is the collapse's evidence**: R4 votes per offset slot over the same `H`
> activations, so every slot — innermost and outermost alike — is decided on the same count, and a reach wider than
> the data supports finds no majority in its outer slots and they drop.

# 8. The one test

> **R12 — The one test.** A pattern earns its dictionary line when the file is shorter for holding it than it
> costs to state. **Nothing measures a file to find that out**: both terms are counts over what the neuron
> already holds, so the margin is the difference in `L` reached directly (§2).
> ```
> benefit(e)  =  Σ over the activations e covers:  covers − price
> cost(e)     =  1 + |e|                                              the line  (D10)
> margin(e)   =  benefit(e) − cost(e)
> ```
> A pattern is **added** only when its margin is strictly positive and **retired** only when strictly negative
> (R15, R17). At equality nothing happens, so the boundary cannot flip-flop.
>
> **`covers` is what nothing else would have covered.** A pattern is worth what it saves over what would
> account for those neurons if it were gone — another pattern of the same cover if one names them, and the
> residual otherwise, where each stands as its own line (D16). A saving some other neighborhood already
> delivers is not this one's.
>
> **The same expression prices a bid over one frame** (R21). There is one valuation in the design (D12); the
> two readings differ in the population they sum it over and in whether the dictionary line is in the sum.
>
> **Benefit is a measurement, so it moves when anything under it moves** — an activation saved or evicted, a
> neighborhood re-centered, a cover re-derived. **No test needs a pass of its own.**

# 9. The two moves

A neuron can do exactly two things to its table: **add** a pattern and **retire** one. Re-centering is neither
— it is what moving counts means (R5). So the whole of restructuring is two tests, asked in that order, at a
bill and nowhere else, **and each is asked once per bill: one candidate built and priced, one pattern retired
at most** (R19). **Both are R12 over different sets** — one margin, read over the neurons a candidate would
take out of the residual and over the neurons a pattern holds — and there is no second currency anywhere in
the design.

## 9.1 Add — creating a child

> **R13 — One decision point: the frame it fires.** A neuron is called once per activation, at age 0, and
> everything structural happens in that call: it covers its backward half and saves it, re-centers, builds and
> prices one candidate, retires at most one pattern, and returns a bid for every pattern that applies together
> with its requests (R18, R19).
>
> **There is nothing to wait for.** `O⁻` is complete when the neuron fires (D17), so no decision here is made
> on partial evidence, none is committed for later, and none is revisited. The neuron remembers nothing
> between one activation and the next beyond what is in its history.
>
> **A child requested in a call is not offered in it.** It enters the table when the machine returns its
> identity, after the response, and it is first *bought* — joins its level's frame, covers, bids, speaks — on
> the next activation of its parent whose cover its pattern takes and whose bid the election accepts.
> **Structure never pays off on the evidence that created it, only on recurrence.**
>
> **But its connections start forming at once.** When the machine registers the child it opens an activation for
> it at the parent's coordinate, at age 0, in the mint frame, and that activation connects like any other
> (§10.3): from the next frame on, every neuron of the child's level that fires strengthens the child's connection
> to it at the offset its age names, until the window closes — after the parent's own activation has
> (R17). **A mint activation connects and does nothing else.** It is not in its level's frame, it holds no
> activation, covers nothing, bids nothing, and neither speaks nor votes (R26, R36); it exists so that a child is
> not born a frame behind the situation it was minted for.
>
> **The rest of the activation's life is connecting and speaking** (D6, §10.3) — the neuron's connections
> strengthened by what fires, and while on the apex, an expectation and an inference read off them.

> **R14 — Where a candidate comes from.** Three fixed steps. Nothing seeds it from outside, nothing grows it a
> neighbor at a time, and nothing repeats until a condition holds.
> ```
> residual(o)  =  the present neighbors of o⁻ no pattern of its cover names                     D16
> seed         =  the neighbor in the most activations' residuals — ties to declaration order (D1),
>                 then to the nearer offset
> population   =  the activations whose residual holds the seed
> C⁻           =  the collapse (R4) over that population, backward, per slot
> ```
> The seed is in every activation of the population, so `C` names it once the population holds two activations; over
> one activation `2 · 1 > 2` fails and the collapse names nothing, so **nothing is ever built on a single
> occurrence**. Every other neighbor `C` names is present, and in the residual, in more than half of the
> population: R4's abstention applies as it does everywhere, so a neighbor a pattern of the cover holds in an
> activation is out of that slot's population, and **a candidate is built on the residual and nothing else.** The
> seed is the neighbor the table is failing on most, and the collapse settles every other slot at once — the
> seed chooses the population, and the population decides every slot.
>
> **Only the backward half is built, because a pattern has only a backward half.** What the child will be followed by
> is the child's own connections, formed by the child's own activations once it exists (D20). Nothing about it is
> decided here and nothing about it is priced.
>
> **The same history under the same covers yields the same `C`.** Covers are held rather than derived (R6),
> so the residual, and with it the seed, is a function of the ring and the covers it carries together.
>
> **What `C` is worth.** Against an activation, `C` takes neurons out of the residual and names some that did not
> fire:
> ```
> reach(o)   =  |residual(o) ∩ C⁻|  −  |C⁻ \ o⁻|          what C is worth there, before its line
> saving(o)  =  max( 0,  reach(o) − 1 )                    what it is worth once the line is paid
> ```
> **A candidate is only ever credited the residual.** A neuron a pattern already covers is not `C`'s to take —
> a candidate that fits a chunk beautifully earns nothing for it if something already accounts for it.
>
> **The floor is R18 and belongs to pricing only.** An activation whose cover `C` would not join contributes
> nothing to the benefit: a candidate has to be able to reach an activation before it pays in it.
>
> **What it is not.** It finds a local best and not the best `C` — choosing the pattern set is the
> facility-location problem and is not solvable exactly at any useful size. What it has instead is no free
> choice anywhere in it, and one candidate per bill, which is the rhythm the machine keeps with one election
> per frame.

> **R15 — The solo test.** R12, asked of a table `C` is not in yet.
> ```
> benefit  =  Σ over the whole history:  saving(o)      (R14, floored — one line per activation already in it)
> commit iff  benefit > 1 + |C⁻|
> ```
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every activation in the ring has a whole backward half, so the question is
> asked over the same evidence R18 will use when `C` next competes for a cover. Nothing here is decided on half
> an activation and nothing later can hand `C` less than the test counted.
>
> **One candidate per bill, whether it pays or not.** What a single candidate leaves uncovered is the next
> bill's residual, and the next bill's seed is whatever is then failing most.

> **R16 — What a child is at birth.** The parent requests; the machine creates. The pattern inherits its
> parent's channel and dimension and mints one level above it, all carried on the request. It is created with
> **no counts**: its own neighborhood belongs to its own level, which it has not observed yet. Its *existence*
> is decided by its parent, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each is one pattern's child, each covers the
> part of an activation its pattern was assigned, and several of them may be promoted at one coordinate (D5). What
> they share is a parent and a coordinate, not a slot.
>
> **Two objects, one word.** The **neuron** minted one level up has no counts. The **pattern** the parent now
> holds is a different object and takes counts at once — its share of every activation whose cover it joins (R6).
>
> **Release is the same shape reversed**: the parent retires, the machine reclaims. A retired pattern goes back
> on the same request that carries the candidate (R19) and the machine reclaims its child neuron at the death frame
> (R17), so a bill touches the alphabet once — in one direction, both, or neither.

## 9.2 Retire — pruning the table

> **R17 — Retire one, then delete.** Read every margin in the table (R12), this bill's candidate included:
> ```
> benefit  =  Σ over the activations it covers:  |neighbors only this pattern names|  −  ( 1 + |e⁻ \ o⁻| )
> retire the pattern with the smallest margin, iff  benefit < 1 + |e|
> ```
> **One per bill, and no other.** Two patterns naming the same neurons are each worth nothing while the other
> stands, so retiring both on one reading would return their neighbors to the residual with nothing left to
> cover them. Retiring the worst lets the survivor take full credit at the next bill, and the next bill reads
> every margin again.
>
> **Without the pattern its neighbors fall where D16 puts them** — to another pattern of the same cover that
> names them, at no extra cost to that pattern, or into the residual at one line each. It is the same
> difference R15 reads with the roles swapped: adding a pattern asks what a neighborhood that is not there
> would take out of the residual, retiring one asks what one that is there is still keeping out of it.
>
> **Retiring is a deletion in the parent.** The pattern leaves the table that instant. It stops competing for a
> place in any cover, so no further activation can bid it, and the neurons it held fall to whatever D16 gives
> them next (R6). Having nothing to cover it has no margin and nothing to re-center — **the neighborhood it
> held stops moving** — and it is not a candidate for anything again. What leaves the table rides the bill's
> return to the machine (R19), as a request to delete it. **The neuron keeps no retired state and re-checks
> nothing.**
>
> **The death frame is the machine's to set, and it can be this frame.** The neuron asks for the child to go
> and says nothing about when; it sees its own open activations and not the units promoted off them, while the
> machine sees both.
> ```
> death frame   =   when the child's last open activation closes
>               =   this frame, when none is open
> ```
> That set only shrinks. A pattern neuron has one parent (D2), so once that parent stops covering with it
> nothing can fire it again, and no level built afterward can name it either, because a level is built out of
> what is firing (R25). Reach grows with the level (D14), so the last to close is the highest one and the wait
> is at most `reach_t(D)` frames, `D` being the highest level the stack currently holds.
>
> **Deleting** is the machine's, on a pass of its own: **every frame, once the last level has run**, it reads
> the **death ledger** and takes everything due — the pattern, its child neuron and that neuron's subtree
> together, and what named them scrubbed with them. A pattern retired this frame with nothing open dies on
> this frame's pass; one whose child is still open waits exactly as long as the stack above it needs, and not
> a frame longer. **Nothing traces who is naming what**: the machine settles the question off the board it
> already keeps.
>
> **The ledger holds the pattern, not a handle to it.** A child's neighborhood is stated in one place, the
> parent's line for it (D9), and the child is expanded through that line (R27). Until the death frame, units
> above it still cover it and the apex may still expand it, so the definition has to stay readable after the
> table stops covering with it.

**A deletion takes the subtree, and takes it at once.** There is no staged cascade and nothing to wait on at
any level.

**Nothing is retired for what followed it.** What followed is measured, not claimed (D17), so a pattern is
never charged for what came after — only for what it names that did not fire beside it.

# 10. The frame, per neuron

**The neuron prices, the machine holds the board.** Every test the neuron runs is its own arithmetic over its
own evidence, and it is never told what the board did with its bids (R23).

The machine calls a neuron twice in a frame. `process frame` reaches every activation that fired **this**
frame — it bills, covers and offers in one call. `process actions` runs after every level has finished and
reaches every **open** activation at whatever age it stands at; it delivers what landed and collects what the
apex speaks (§10.3). Nothing is written into a neuron outside those two calls.

## 10.1 The cover, and the offer

The neuron fired this frame, and the backward half of its activation, `O⁻`, is in hand — whole (D17). R18 defines
three operations on it; R19 says the order they run in.

> **R18 — Cover, assign, offer.** **Every activation that fired covers its own backward half**, and the neuron
> answers for each separately. The machine calls once with all of them, because they are activations of one
> type at one age differing only in position (D2, D8), and one call is cheaper than many.
> 1. **Cover.** Over the patterns of the table, repeatedly take the one with the highest
>    ```
>    ( neurons of the residual it names )  /  ( 1 + | e⁻ \ O⁻ | )
>    ```
>    and take it **iff it names strictly more of the residual than its price**, ties to the older `id`. Each
>    round is measured on the residual the last one left. **Stop when no pattern pays.** The result is the
>    **cover**, and it is what the activation holds (R6).
> 2. **Assign.** A neuron of `O⁻` belongs to the pattern that took it — the first pattern the rounds took that
>    names it, since every round takes only out of the residual. That partition is the activation's assignment
>    (D21), and it is what each pattern counts and re-centers on (D23, R4). **The cover is exclusive because
>    credit is.**
> 3. **Offer.** Return a bid for every pattern of the table that **applies** to `O⁻`: more than half of its
>    neighbors are present,
>    ```
>    2 · | e⁻ ∩ O⁻ |  >  | e⁻ |
>    ```
>    whether or not the cover took it. A bid is the child's id and the pattern's neighborhood (R20).
>
> **The offer is not the cover.** The cover is one partition, chosen on the neuron's residual; the offer is
> every pattern the machine could conceivably buy, because the machine's residual is not the neuron's (R22) and
> a pattern the cover passed over may be the machine's best purchase. A pattern that does not apply cannot be
> bought on any board: its present neighbors do not outnumber its absent ones, so `covers − price ≤ −1` however
> the board stands (R21). **The offer is the loosest set that drops nothing the machine could buy, and it is
> the collapse read backwards**: a pattern is a majority statement over the activations it covers, and an activation
> agrees with it when it agrees with the majority of it. The offer is not exclusive because the machine
> chooses; two bids from one neuron can both be bought.
>
> **Step 1 and R23 are one procedure over two populations.** The neuron runs it over its table against one
> activation's residual; the machine runs it over a frame's bids against the free slots of the board. Both take
> the best ratio, re-measure what is left, and stop when the best remaining does not pay. **The criterion is
> the same in both** — what a neighborhood covers against what it costs to state (D12) — and the price is
> the same expression on both sides, `1 + |e⁻ \ O⁻|`.
>
> **Two things differ, and neither is the procedure.** The populations, so the numbers do (D12) — and in
> particular the activation's own neuron is not in `O⁻`, so the cover can never take it, while the bidding
> activation *is* a slot of the board and is the first thing a bought bid subsumes (R21). And what each side
> may do about a poor result: the neuron may mint a pattern and retire one (R14, R17), the machine may only
> take what it is offered. **Recognition is one algorithm; only the neuron writes the dictionary.**
>
> **A neuron with an empty table covers nothing**, offers nothing, and the whole of its activation is residual (D22).
> That is the shortest file available to it, not a failure.

## 10.2 The bill

**The bill is the same call**, and it runs first: the neuron saves this frame's activation, restructures, and only
then offers, so the offer is over a table that has already seen this frame. Five fixed passes, none of which
repeats until a condition holds.

> **R19 — The bill's passes.** Once per call, in order:
> 1. **Cover and save.** The new activation is covered and assigned (R18 steps 1 and 2) over the table as it stands
>    and joins the ring; a full ring evicts its oldest activation, one out for one in (D24, R9). Every pattern of
>    either cover moves counts by its share (R3). **Saving is
>    unconditional**: a neuron another unit will subsume this frame records exactly as one that will be
>    promoted does, because the neuron does not know which it will be and never learns.
> 2. **Re-center.** Every pattern the first pass moved counts on re-centers, once, over the totals it leaves
>    (R5); every activation whose table moved under it re-derives its cover and keeps the cheaper (R6). **Once for
>    the bill, not once per activation**, so the center does not turn on which activation the inputs happened to
>    give first — and that center is what the two tests below price against.
> 3. **Build one candidate.** Seed, population, collapse (R14), then price it (R15). If it pays it is
>    requested, and when the machine returns it every activation takes the cheapest of its held cover, its held
>    cover with the newcomer appended, and its cover re-derived (R6), so the newcomer's share is at least what
>    the test counted.
> 4. **Retire one.** Read every margin, this bill's candidate included, and retire the worst if it is strictly
>    negative (R17). It leaves the table at once, its neurons falling where D16 puts them and the neighborhood
>    it held freezing. **The neuron deletes nothing**: the machine takes the pattern, its child and that
>    child's subtree on the ledger pass at the death frame, which is this frame whenever nothing is open.
> 5. **Offer, and return.** A bid for every pattern that applies (R18 step 3), less the candidate just
>    requested (R13), and one request carrying the candidate that survived and the pattern that retired. A
>    passing test **requests** rather than creates, because a pattern is a symbol at the level above and that
>    alphabet belongs to the machine; the machine returns its identity and the neuron registers it. **Sending
>    last settles what the request carries**: the candidate's definition is the final one, and the table the
>    offer was read from is the table the next frame will see.

**The bill runs before the election**, on evidence that was complete when the neuron fired (D17). The election is over
bids from a table that has already saved this frame, and nothing about its outcome comes back to the neuron.

## 10.3 The rest of the window — `process actions`

**Once every level has run, the machine calls every open activation**, at whatever age it stands at, with two
things:

```
the forward neighbors   the neurons of its level that fired this frame, at offset +age              D13
the reward              any reward that arrived, for the action that ran this frame and for any earlier
                        frame the reward spans, at the distance each one names                     R33
```

The neuron strengthens a connection for each and saves nothing: the connection at `(neighbor, age)`, created at
strength 1 or incremented (R31), and for each reward share the estimate of the action connection it names (R33) — for
the action that ran this frame, the connection just strengthened, in the same write. Nothing is written into the
activation. **Nothing is decided, priced or compared here**, and no test is waiting on any of it. Covered activations
strengthen exactly as uncovered ones do (D7).

**If the activation stands on the apex (R26), the call returns what it speaks.** It reads its own neuron's
connections at the offsets ahead of its age — `age + 1` for the frame ahead, out to its reach — and returns
two things in its own level's alphabet: the event connections there, each with its strength, are its
**expectations** (§13), and the action connections there, each with its strength and estimate, are its
**inferences** (§16). A base
neuron on the apex speaks from its own connections like any other; what it expects is the marginal over every
situation it fires in, and it speaks only because nothing more specific covers it.

**An activation closes at age `reach_t`** (D6), once that frame is taken. Closing does nothing but stop the
strengthening — there is no second bill and nothing is saved twice.

# Part II — The machine

# 11. Contraction

> **D25 — Contraction.** The machine covers the level below with units from the level above, each taken when
> it covers more neurons than its bid costs to state — `1 + |e \ O⁻|`, never the dictionary line (R21, R23).
>
> **Covering everything is not the goal.** A neuron no unit covers stays in the file as itself, at cost 1
> (D10), and that is the shorter file whenever no unit could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a neighborhood names neighbors at offsets, so a promoted unit replaces a chunk of
> spacetime. Spatial contraction is the case where every offset is zero.

## 11.1 Bids

> **R20 — A bid is a neighborhood and a name.** A bid carries two things and no others:
> ```
> the neighborhood   the pattern's backward neighbors — its dictionary line (D9)
> the child          the id of the pattern neuron this bid would promote
> ```
> The neighborhood travels because it *is* the line for the symbol being proposed, and the bidder is implied,
> because a child *is* its parent in that neighborhood. **No connection travels**: nothing at
> `Δt > 0` has fired, and the file has no line for what follows (D9).
>
> **One activation may send several**, one per pattern that applies (R18), and they are independent bids: each
> answers for what the election leaves it, and the machine has no reason to know they came from one neuron. A
> neuron covering nothing sends nothing.
>
> **Nothing else is sent, because nothing else is the neuron's to know.** Which of the named neighbors actually
> fired, what this bid is worth against them, and what another bid has already taken are facts about the frame
> — and the machine is holding the frame. It reads the neighborhood against its own board and derives the rest
> (R21).

> **R21 — What a bid covers, and what it costs.** The neuron sends the neighborhood (R20) and nothing else.
> The machine holds the frame, so it reads that one object against what fired and derives both numbers.
> ```
> the bid   the pattern's backward neighborhood, and the child's id                          (R20)
> covers    the neurons it names that fired and no earlier unit covers — the slots it asks to subsume,
>           the bidder among them
> price     1 + |e \ O⁻|   its own line in the history, and the neurons it names in those
>                          same frames that did not fire
> ```
> `covers − price` is the saving over stating the chunk flat, D12's expression over the machine's population.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as its own
> line if nothing covers it (D25) and costs a turn-on if this unit is promoted — one symbol either way, so it
> cancels before the test begins, and charging it here would count it twice against the uncovered term of the
> same sum (§11.4).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another unit
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged one whether or not another unit is right at
> that slot — another unit's expansion being right there does not make this one's wrong name free. **What a
> neighborhood gets wrong about a frame is a fact about the two, and no assignment moves it.**
>
> **This is the neuron's arithmetic over the machine's population, and the number is not the neuron's.** The
> neuron took the pattern into its cover, or offered it, on its own residual (R18); the machine tallies on a
> board where earlier frames' credit stands (R22) and this frame's other bids contend (R23). The two numbers
> differ, and are meant to (D12).
>
> **This is a price for one bid, not for the symbol.** The dictionary line `1 + |e|` is weighed by the one test
> (R12) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R21).

## 11.2 The board

> **D26 — The coverage set.** Per level, the machine keeps which accepted bid was credited each covered
> activation:
> ```
> coverage set    per level, backward    which accepted bid holds each subsumed active activation
>                 an assignment          one holder per activation; settled slots are never re-assigned
> ```
> **A slot is named by a full coordinate**, dimension and position together, so two activations of one neuron
> at two positions are two slots and never contend. Level `k`'s coverage set spans `reach(k) + 1` frames — a
> bid reaches `reach(k)` back and no further — and the box every other activation dimension gives, and ages
> out with it. **The machine holds nothing on the scale of the run.**
>
> The assignment is about **credit**: a neuron is a fact that needs paying for exactly once, so it is settled
> once and never revisited (R22). It is not about naming: a unit expands to everything its neighborhood names,
> credited or not (R23).

> **R22 — This frame's bids against the board as it stands.** Only neurons no earlier frame's election has
> credited are in play, so a chunk already paid for is not paid for twice. **No earlier promotion is ever
> re-scored.** Within the electing frame a slot is credited once, to the first accepted bid that names it
> (R23), and never moves.
>
> **Earlier bidders have priority.** A bid at `f` wins a neuron at `f − 2` before a better bid at `f + 1` can
> name it, because the earlier election settled it and nothing re-elects the past. That is the price of never
> revisiting a frame, and the design pays it.

## 11.3 The election

**The file over one frame is the units promoted plus what they got wrong**: `Σ over the accepted (1 + |e \ O⁻|)
+ the neurons no unit covered`, the history half of `L` (D12) over the frames the election can see. **The two
terms do not overlap**: a neuron a promoted unit fails to name is in the second and not the first, which is why
the price counts only the neighbors named and absent (R21). The dictionary half is R12's, and neither test
touches the other's sum.

**That sum is the objective; R23 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — bids are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good assignment, not a proved minimum. **Every neuron a bid covers ends up credited to exactly one
bid**, which is what stops a chunk being paid for twice.

> **R23 — The election is R18 run by one chooser.** The candidates are this frame's bids; the residual is the
> **free set**: every active activation of the level below, at its own full coordinate — frame and position —
> that some bid names and no earlier election has credited (R22). Bids arrive naming relative offsets, so each
> is resolved against its own coordinate before the first round. Rounds, until one fails:
>
> 1. **Measure.** For every bid not yet taken, `covers` is the free slots it names and `price` is
>    `1 + |e \ O⁻|` (R21). **Price is the same in every round**, because which neurons fired is a fact about
>    the frame and no other bid changes it; **covers only falls**, because the free set only shrinks.
> 2. **Take.** The bid with the highest `covers / price` — **ties to the older symbol, then to the earlier
>    coordinate**: creation order for a pattern and declaration order (D1) for a base neuron, then frame, then
>    position — is accepted **iff `covers > price`**. Its covers leave the free set, credited to it. **If the
>    best bid does not pay, no bid does**, and the election stops.
> 3. Return to 1 over the smaller free set.
>
> **The bound is structural.** An accepted bid takes at least two slots, since `covers > price ≥ 1`, so the
> rounds are at most half the free set and never more than the bids.
>
> **A round costs one re-measure, not a re-scan.** `price` is fixed and `covers` only falls, so no bid's ratio
> can rise between rounds: a bid measured in an earlier round holds an upper bound on what it is worth now. So
> the leader is re-measured alone, and if it still beats every other bid's last measurement it is the true
> maximum and is taken without touching them. Step 1 is what the rounds mean; this is what they cost.
>
> **A bid that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> a slot it named is either credited to a bid that did pay or stands as its own line.
>
> **The assignment is a partition of the neurons the accepted bids name**, and that is the whole of the
> inhibition — no bid is ever edited or forbidden, and **overlap is legal and priced**: a bid that names a slot
> an earlier round credited gains nothing for it and pays nothing for it (R21). **Held by an accepted bid** and
> **named by an accepted bid** are therefore the same set, so coverage, credit and the apex frontier (R26) are
> one question with one answer.
>
> **Outcome**: accepted bids are promoted, one unit each and **whole** — a child expands to everything its
> neighborhood names, credited or not — the neurons credited to them are subsumed, and every active neuron no
> accepted bid covers stands as itself. **The election delivers nothing to anyone**: it writes the coverage set
> and stops. No neuron is told which of its bids were bought, what they were credited, or what they lost; a
> neuron's history is what it saw, and the board is the machine's.

## 11.4 When a slot is settled

Settlement is a property of one slot at one full coordinate. **Nothing here delays anything the machine does**
— no pass blocks on it and no decision is deferred by it. **The only consumer is measurement**: when `L` or
apex-units-per-frame is read, the settled frames are the ones whose numbers are final.

> **R24 — Settlement is a condition to detect, not a schedule to predict.**
>
> **Frontier membership settles one level, in `reach_t` frames.** Whether an activation at frame `h` is covered
> is decided by bids firing no later than `h + reach_t`, since a bid reaches `reach_t` back and no further.
>
> **A frame's encoding settles at the top of whatever stack reached it.** A unit one level up, firing later,
> can name a lower unit that names frame `g`. **Frame `g` is settled when no level holds a live unit that could
> still join or leave that set** — a closure over the levels, evaluated upward.
>
> **`D` is reached, not known.** The walk stops where a level accepts no bids and therefore produces none above
> it, so `Σ_(k<D) reach_t(k)` bounds a condition rather than counting out a delay.

# 12. The order of a frame

> **R25 — One stack, at the derived reach.** Base neurons bill, cover and offer; the election settles which bids
> are bought. The survivors are level 1 — the fewest that cover the active base neurons — and it happens
> again. **When a level's active neurons promote no children, nothing propagates and there is no level above
> it on this frame.** Nothing declares the depth and nothing caps it.
>
> **Within a level the order is bill, offer, election**, and it cannot be otherwise: the bill saves this
> frame's activation before the offer reads the table, and the offer is what the election is over. Nothing in that
> order leaves the level or the frame.
>
> **Bill and offer are one call on one activation** (R13). The neuron saves, restructures and offers, the
> machine elects — and the activation that did all of it fired this frame, on evidence that was complete when
> it started.
>
> **The frame ends with the machine's own passes.** Once the last level has run, the machine reads the death
> ledger and deletes everything due (R17), calls every open activation (§10.3), expands what the apex expects
> (§13) and resolves what it infers (§16).
>
> Every level runs the same rule at the reach one expression gives it (D14). **Compression is spatio-temporal
> at every level, in one pass**: a pattern at any level may name neighbors in its own frame, in earlier ones,
> beside it in space, or in a mix.

> **R26 — The apex is a frontier, not a level.** It is every active neuron **no accepted bid covers** — the
> uncovered set, at every level at once — so a base neuron nothing found worth chunking stands in it beside a
> level-4 pattern. This is the frontier the file's history writes, **the one that predicts** (§13), **the one
> that votes** (§16), and **the one reward credits** (R32): the uncovered set does all four, and coverage
> silences a neuron in every one of them at once (D7). Everything underneath it is recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that offered a child and had its bid declined with nothing else
> covering it is still on the frontier, and stands in the file as its own line.
>
> **Coverage is acquired late and never revoked.** An activation firing at `g` is uncovered until some bid
> takes it, and the last bid that can fires at `g + reach_t` (R24), so an activation may speak for a few frames
> and then fall silent. **No accepted bid is ever dropped** (R22).

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
alongside (D5), so it is recognized and chunked by the rule they are. **The action connection is not formed here**:
it names the action pattern that ends up in control of the dimension, which stays open until every level has
settled, so it is recorded in the `process actions` pass instead (R31).

# 13. The prediction

**Predicting is the machine's act, and it is an output, not a claim.** Every activation on the apex returns
its expectations in its own level's alphabet (§10.3); the machine expands each to base symbols and resolves
one expected symbol per event dimension per frame ahead. That set is the machine's first output. **Nothing
inside the machine scores it**: the file holds no line for what was expected (D9), so a wrong expectation costs
nothing, retires nothing, and is not a correction. What settles it is the input, and only whoever reads the
output is the wiser.

**Expectation and inference are one read, resolved one way.** An apex activation reads one set of connections; its
event connections expand to base events and its action connections to base actions; both are resolved at the base, one
winner per dimension, by a vote in which every apex activation is one voter and **the level it stands at is never
read**. What differs between the two is only what wins a dimension: an event by how many voters expect it, an action
by what it is estimated to earn (R36).

> **R27 — Expansion.** A connection at level `k` names a level-`k` neuron, which is not yet anything in the base
> alphabet. Expanding a unit recovers the neighbors its neighborhood names one level down, at that unit's
> offset plus theirs — offsets compose because each is a difference of activation coordinates (D2) — repeated
> to base symbols. Every expectation then has the shape `(dimension, frame, position, symbol)`.
> ```
> a's connections name  (b, +1)                → b's dimension at f+1
> A's connections name  (C, +2)  expand it:
>   C's line is {(p, 0), (q, −1)}                → p's dimension at f+2, q's dimension at f+1
> ```
> **This is the same expansion that recovers the run from the file** (D11) and that turns a selected action
> pattern into a program (R30). There is one expansion in the design, and it reads dictionary lines only.
>
> **What travels down with a symbol is its strength.** An expected connection carries its strength (R31), and every
> base symbol its expansion places carries that strength as the vote's strength. Nothing
> is re-weighted on the way down: a level-3 unit expecting one level-3 symbol places every base symbol of its
> line at the same strength.
>
> **One winner per event dimension, by share of voters.** For each `(dimension, frame, position)`, every apex
> activation whose expansion placed a symbol there is one voter, contributing one unit split across the
> symbols it placed in that dimension in proportion to their strengths, so a voter that hedges between two
> symbols counts as one voter and not two. The symbol with the largest share wins the dimension; ties go to
> the older symbol. **Level is not read**: a level-3 voter and a base voter are one unit each. A dimension no
> voter reaches expects silence.
>
> **An action connection in the expectation is what the machine expects itself to do**; what it does is chosen from
> the inferences (R36), and neither read enters the other.

> **R28 — The expectation only reaches forward.** Expansion reaches both directions, so a connection at `+2` whose
> line reaches `−3` lands at `−1`. Symbols landing at or before the expecting unit's own frame are discarded,
> and nothing is lost: the output for a frame is what the machine expected *before* it.

---

# Part III — Actions, reward, and selection

# 14. Actions

An action dimension carries what the machine executes, and it is compressed by the same hierarchy its events
are (D1, D5).

> **R29 — Two frames: infer, then execute and reward.** What is chosen in one frame runs in the next, and what
> it earned arrives with that frame.
> ```
> f      infer     the frame's events are recognized, `process actions` returns the inferences,
>                  the inference resolves (R36), committing an action for the frame ahead
> f + 1  execute   the action runs, and its neuron fires in this frame's column alongside
>        reward    this frame's events — every open activation sees it as a forward neighbor;
>                  what the action earned arrives as this frame's input, and the connection
>                  that neighbor strengthens takes the reward in the same write (R31)
> ```
> **The reward is part of the frame the action ran in.** The environment reports what it observed and what the
> action in effect during that frame earned together, so an action is never on the books without its outcome,
> and no activation has to stay open for a reward that arrives later than the action it pays for (D6).

> **R30 — Execution is an expansion**, of the selected pattern (R36) through its dictionary line (R27). A high
> action pattern becomes its constituent actions at the offsets its line records, down to base actions that
> execute. Execution is not a second mechanism; it is this expansion read as a program. Each base action
> executes in the frame its expansion places it in, the nearest being `+1` (R29).

# 15. Reward

A reward is an input, not a symbol: alongside what it reports observed, a frame may carry rewards for actions
already executed. They reach the machine through one object and one only — the **action connection**, which no
structural test can see (R34).

> **R31 — An action connection is a forward neighbor with an estimate.** What executes at `f + 1` is not known at `f`:
> which action pattern ends up in control of the dimension is settled only once every level has run. So the action
> that ran fires in its dimension at `f + 1` (D5), with its reward beside it (R29), and **every activation open at
> that frame sees it as a forward neighbor at its own age** (D13, §10.3). That neighbor strengthens the connection
> between what the neuron stands for and what the machine did — formed against what actually ran, so **a neuron that
> argued for a different action still learns from the one that ran.**
>
> **Every open activation connects to it, at every age it is open at.** The offset is the age — the distance from
> the frame the activation opened to the frame the action ran — so a neuron open at ages 1, 2 and 3 holds the
> same action at three offsets. **Strengthening and reading sit one frame apart**: selection is choosing an action
> that will run *next* frame, so an activation reads the offset one beyond the age it stands at (R36). Fan-out
> is bounded — a neuron names actions only in the channels its activations have seen follow.
>
> **Making and strengthening are one operation.** A neuron's action connection at `(action, offset)` has a
> **strength**, the number of its exposures — the times an activation of the neuron saw that action follow at that
> offset — and an **estimate**, the mean of the reward those exposures received (R33). The first exposure creates the
> connection at strength 1 and every later one increments it; each share of reward folds into the mean weighted by `1
> / strength`, so the estimate is the exact average over the connection's exposures and no rate is chosen. **Nothing
> leaves.** No exposure is ever subtracted, no strength ever falls, and no window bounds a connection: it is the whole
> of what the neuron has seen follow it at that offset, over its life. An event connection is the same object without
> the estimate.
>
> **Every level holds them, base neurons included.** No operation derives what a child was worth from what its
> parent earned or the reverse, so a connection held only at the frontier would be lost at the next mint. Every
> neuron keeps its own connections over its own activations (D20).
>
> **A covered neuron still learns.** Coverage decides which neuron speaks, not what counts as evidence: the
> action ran, the reward arrived, and that reward is a sample of what the action is worth when this neuron
> fires, whichever unit selected it.
>
> **A connection dies with either of its ends** — the neuron, or the action neuron. Nothing else removes one (R34).
>
> **A connection wired ahead of any exposure is created at strength 1 and estimate 0.** The bootstrap (R35) and the
> walk (R37) create connections nothing has yet been seen to follow; each is created exactly as a first exposure at
> neutral reward would create it, and from then on it is a connection like any other.

> **R32 — Credit lands on the apex active action, and covers its whole span.** The credited object is the
> highest action pattern in control of its dimension at the frame the action ran, not at the frame the reward
> arrives in, falling back to the base action when nothing higher covers it. Before any action pattern exists
> the apex is the base action, so the rule holds across all of development.
>
> **A pattern that occupies its dimension for several frames is credited with every reward that arrives while
> it runs.** An expansion places base actions at `+1`, `+2` and beyond (R30), so a pattern has a span rather
> than a frame, and the credit lands on the pattern rather than on any one of those frames.

> **R33 — A reward names what it pays for, and the machine fills in the rest.** A frame carries an array of
> rewards, each of them
> ```
> { reward, channels, frames }
> ```
> **`channels`** is the set of action channels the reward pays and **`frames`** the span of the window it pays
> over. **Both are optional, and an omitted one means all of them**: no channels named pays every channel, no
> span named pays the whole window.
>
> **Distance is counted back from the frame the reward arrives in**, `0` being that frame itself — the frame
> whose action the reward is for in the two-frame cycle (R29). **What reaches distance `d` falls linearly with
> `d`**, because the further back a frame is the less likely it is to have caused what arrived. Over a span of
> `S` frames,
> ```
> share(d)  =  reward · (S − d) / S            d = 0 … S − 1
> ```
> so the frame the reward arrived in takes the whole reward, the far end of the span takes `reward / S`, and
> nothing along the way takes nothing. A span of one frame is the degenerate case, `share(0) = reward`, and it
> is what an environment that pays each frame's action as it goes reports.
>
> **The whole window is the deepest open one.** When no span is named, `S = reach_t(D)`: an activation at the
> highest level `D` is open through `reach_t(D)` frames (D6), so the furthest back any open activation could have
> seen an action run is `reach_t(D) − 1` frames, and no share can reach further than the connections that would
> take it. At the base that is `S = 1`, the degenerate case above.
>
> **Nothing is being divided up.** Every distance in the span is paid, and the shares are not a partition of
> the reward — a reward is not in short supply, and the point of spreading it is not to conserve it but to say
> how likely each frame is to have earned it. What is genuinely responsible recurs and accumulates; what is not
> is sampled once and averaged away. Each share is delivered, in the `process actions` call (§10.3), to every
> open activation that saw the apex action of a paid channel (R32) run at the frame that distance names, into
> the connection at the age the activation stood at then: at distance `0` the connection this call strengthens,
> further back one strengthened `d` frames ago (R31).
>
> **Rewards in one frame are independent.** A connection at one `(channel, offset)` takes the sum of the shares that
> reach it, and nothing coordinates one reward with another.
>
> **The unscoped form is the general case, and the machine sorts out the attribution itself.** An environment
> that knows what it is paying for names it — a stock channel, and the one frame the buy or sell ran in — and
> the credit is exact. One that does not names nothing, and the reward spreads over every channel and the
> whole window: the shares landing on a channel or a distance that had nothing to do with the outcome are
> noise around zero and average out over exposures, while the shares landing on the ones that did accumulate.
> **No structure is priced on either** (R34), so a coarse reward costs accuracy in the estimate and nothing in
> the file. What it does move is the walk: an unscoped reward carries its sign into every estimate it reaches, so
> the sign test R37 reads shifts the same way for all of them, and the walk runs while the world is going badly
> overall and rests while it is going well. That is the meaning of a signed reward, and it is intended.

> **R34 — Two objectives, meeting at one place.** Everything structural is priced in file length; reward prices
> nothing structural and cannot. The machine runs **two** objectives: compression, which decides what structure
> exists, and reward, which decides which of it is executed. They meet at exactly one place, a neuron's action
> connections, and connections are not in the file and not priced by the one test.
>
> **Estimates are not decayed, and not windowed.** A connection's reward is the plain average over every exposure it
> has had, with no cap, no rate and no horizon; nothing ever leaves it (R31). **Non-choice moves nothing**: an
> action not taken keeps the value it had. What makes an estimate specific is the neuron that holds it, not
> how recent its exposures are — a child's connections are over the one situation its parent's pattern names (R35),
> and a changed situation is answered by a new child, never by forgetting.

# 16. Selection

> **R35 — Selection.** No fit says which action to take; it says only what a situation was followed by.
> Choosing comes from the action connections of the neurons standing on the apex, each carrying the reward that
> arrived averaged over its exposures, and **the machine executes the best. Nothing else decides it.**
>
> **A situation is a set of active event neurons** — any one of them, and any subset of them. Situations
> intersect, and **nothing ever materializes one**: a situation is what a voter fires in, never an object the
> machine holds.
>
> **A voter is one apex activation at one age**, reading its own neuron's action connections at the offset ahead. A
> base neuron's estimate is a marginal over every situation it fires in; a child's is over the one situation
> its parent's pattern names; so the same frame reads differently to a base neuron, to the level-1 child
> covering a chunk of it and to the level-4 child covering the whole, and differently again to any of them two
> ages later. **A minted pattern is how one recurring situation acquires an estimate of its own** — which is
> why a covered neuron is silenced (D7) and its coverer speaks instead, and the only reason a base neuron's
> marginal ever speaks is that nothing more specific was bought over it.
>
> **The asymmetry is in what is chosen, not in who holds the connections.** Every neuron holds action connections,
> action neurons included, and any apex activation may select an action. Nothing selects an event: what the machine
> expects to observe is an output (§13), never a choice.
>
> **The bootstrap is a connection, not a fallback.** Every neuron is born holding the declared default action at
> every forward offset it can vote at, at strength 1 and neutral estimate (R31), so there is no separate
> no-history path. It is the
> only action wired in advance; R37's walk supplies the rest, one at a time and only where the default has been
> judged and found wanting. **An action dimension no inference reaches runs the declared default**, which can
> only happen when nothing at all is on the apex.

> **R36 — The inference decides.** An inference is one action connection read by one voter: it names an action
> neuron at the voter's own level and carries a strength and an estimate. What runs is chosen from the
> inferences and nothing else; the expectation (§13) is never read here, and what runs is never read there.
>
> **Inferences expand before they are resolved.** An inferred action at level `k` is expanded through its
> dictionary line to the base actions it places at the frames ahead (R30), and every base action placed
> carries the strength and the estimate of the connection it came from. Resolution then runs at the base and only
> there.
>
> **A selected pattern stands until its span runs out.** Choosing an action pattern at `f` places base actions
> at `f+1`, `f+2` and beyond — and those placements are **standing inferences**, contending for their
> dimensions exactly as this frame's fresh ones do, at the strength and estimate they were selected on. **A
> plan holds because it keeps winning**: a better estimate displaces it, and when its span ends it is simply
> gone. Nothing is retracted and nothing is held.
>
> **The electorate, explicitly.** At frame `f` the voters entitled to an inference on the actions at `f + 1`
> (R35) are:
> ```
> every standing inference that places a base action at f + 1                          R30
> plus  every open activation the machine holds at f                                    (D20)
>   less  every activation an accepted bid covers                                       (D7)
>   less  every mint activation                        it connects only                   (R13)
>   less  every activation at age reach_t             it has no offset left to read
>   read as (neuron, age)                             position carries no connection          (D8)
> ```
> Each reads its neuron's action connections at **offset `age + 1`**, the distance at which it will stand from the
> action it is choosing (R29). Every connection at that offset is one inference, naming an action neuron and
> carrying a strength and an estimate, and it expands as above.
>
> **Position drops out.** Two activations of one neuron at two positions read one set of connections, so they offer
> the identical inference and the argmax is indifferent to the duplicate. Two *ages* are two voters and do not
> collapse together — they read different offsets, so they can name different actions.
>
> **One winner per action dimension, by estimate.** For each action dimension at `f + 1`, every base action
> some voter's expansion placed there is a candidate. **A candidate's estimate is the mean of the estimates its
> voters placed it with, each weighted by that voter's share** — one unit per voter, split across the actions
> it placed in that dimension by strength, exactly as §13 counts expectations — and **the candidate with the
> largest estimate runs**. Ties go to the larger share of voters, and then to the older action. **Level is
> not read**: a level-4 voter's inference and a base voter's meet on the estimate alone, and a specific
> situation wins over a general one only by being right about what pays, never by rank. **Nothing corrects for
> how many exposures an estimate rests on**, so a sharp estimate on three exposures outranks a coarse one on
> two hundred.
>
> **A covered neuron supplies nothing** (D7). A newly minted child therefore starts with no estimate but the
> default's and explores.

> **R37 — Exploration.** The default policy resolves explore–exploit without randomness: **the action alphabet
> is declared in order**, and **an action connection whose estimate turns negative wires the next action in that
> order** — the first one the neuron holds no connection to at that offset — at strength 1 and neutral estimate
> (R31).
>
> **The walk is in the same currency as everything else**: an untried action becomes a candidate by becoming a
> connection, so R36 enumerates it like any other and needs no second source of inferences. The trigger is one
> connection at one offset, never a reading over the neuron's whole set.
>
> **A reward is signed, and zero is where everything starts.** The environment reports an action as good or bad
> — strictly greater than zero or strictly less — and zero is neither. It is also the estimate every connection is
> created at, the declared default's included, which is what makes the walk work: a connection at zero has not been
> judged, so it outranks anything negative and yields to anything positive.
>
> **The walk ends when the alphabet does.** Once a neuron holds a connection to every action in the channel at that
> offset there is nothing left to wire, and selection takes the largest estimate, which is the least bad.
