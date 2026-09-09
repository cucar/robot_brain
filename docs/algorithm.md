# Universal Compression with Actions and Rewards (UCAR)

UCAR is the design for a machine that compresses what it observes by building a hierarchical dictionary of patterns, and
learns what to do by observing rewards. It is defined by two alphabets, like a Turing machine: the **event
alphabet** it can observe and the **action alphabet** it can execute. Above each, it forms symbols of its
own. Every symbol, base or learned, event or action, is a **neuron**.

It has two inputs and one output. Inputs: the events observed and the rewards earned. Output: the actions it
executes, written in the base alphabet.

This document is the specification, and nothing else. **D** is a definition and **R** a rule; together they
are the machine. Theorems, worked examples and all commentary on why the design is shaped this way live in
[algorithm-remarks.md](algorithm-remarks.md), keyed by the same D and R numbers; risks, diagnostics and open
questions live in [algorithm-evaluation.md](algorithm-evaluation.md).

---

# The vocabulary

Every term below is defined where its key says, and this table is a lookup rather than a definition. The density
is the difficulty: the arithmetic is counting, and what has to be held is which object a word names.

**The alphabet.** What the machine is given.

| term | meaning                                                                                        | key |
|---|------------------------------------------------------------------------------------------------|---|
| channel | a declared input, carrying at most one event neuron dimension and at most one action neuron dimension | D1 |
| neuron dimension | structural — it says what a symbol *is*; a dimension and a bucket are a base symbol            | D1 |
| activation dimension | fleeting — it says where one instance happened; time, and whatever the input's layout adds     | D1 |
| event / action | the two kinds. Events are observed, actions are executed, and each has its own alphabet        | D1 |

**The neuron.** What a neuron is, and what it holds.

| term              | meaning                                                                              | key |
|-------------------|--------------------------------------------------------------------------------------|---|
| neuron            | a symbol, and a type. Base or learned, event or action                               | D2 |
| activation        | one occurrence of a neuron, at a frame and a position                                | D2 |
| pattern           | a set of past and present neighbors, held in one neuron's table; it promotes a child | D15, D16 |
| child             | the neuron one level up that a pattern promotes                                      | R16 |
| neighbor          | a neuron at an offset — behind or beside, never after                                | D7 |
| offset            | a coordinate difference, kept to one significant digit in base 2                     | D6 |
| reach             | how far a neuron sees in every activation dimension — `2^k`, doubling every level    | D4 |
| neighborhood, `O` | the past and present neighbors one activation **observed** at age 0                  | D7, D15 |
| history           | the last `H` activations of one neuron, oldest first. The only free parameter is `H` | D18, R4 |
| age               | frames since an activation fired, `0` through `reach_t`. Read, not just counted      | D9 |
| open activation   | one the machine still holds: while uncovered it connects to the action that ran, and speaks | D9 |

**Compression.** One valuation, read over two populations.

| term | meaning | key |
|---|---|---|
| cover | **a set of patterns** — the ones chosen to explain one activation. Held by the activation | D20, R9 |
| covered | **a set of neurons** — the ones one pattern names that fired. One per pattern of the cover | D20 |
| residual | **a set of neurons** — the ones no pattern of the cover names. Not an error, and not a pattern | D20, D21 |
| assignment | what R9 credited: one pattern per covered neuron; the residual is credited to none | R9, D16 |
| price | `1 + \|e \ O\|` — its own line, and the neurons it names that did not fire | D22 |
| margin | `covered − price`. The only valuation in the design | D22 |
| collapse | the majority over a population that decides what a pattern names | R7 |
| candidate | the one pattern built per call, out of the residual and nothing else | R14 |
| file, `L` | the yardstick the run would be written to: the dictionary, and the body | D12, D14 |

**The frame.** What runs, and in what order.

| term | meaning | key |
|---|---|---|
| `process frame` | the call in its level's turn: everything structural for what fired | R20 |
| bid | a pattern and its child's id, and nothing else | R21 |
| free set | the active neurons of the level below that no earlier election has credited | R24 |
| election | R9, run by the machine over a frame's bids against the free set | R24 |
| apex | every active neuron **no accepted bid covers** — at every level at once. A frontier, not a level | R27 |
| expansion | recovering what a neuron names, level by level, down to base symbols | R28 |

**The future.** What is measured rather than named.

| term | meaning | key |
|---|---|---|
| connection | per `(action neuron, offset > 0)`, on an event neuron: how often that action followed, and what it earned. Never in the file, never in a test | D25 |
| strength | a connection's exposures — the times an activation saw that action run at that offset | R31 |
| estimate | the mean reward a connection's exposures received. Never decayed, never windowed | R31 |
| inference | a connection read by one voter. The machine's output | R36 |
| voter | one apex activation at one age, reading its own neuron's connections | R35 |

---

# 1. The objectives

There are two objectives: **compress what is observed**, and **act on the best reward estimate**. 
The compression decides what the current situation *is*, and that's provided to the second objective as input. 

## 1.1 Compression: classify the situation

**The machine compresses by naming.** A pattern is a set of lower level symbols or patterns that keep occurring
together — past and present neighbors of one another (D7) — and one symbol stands for all of them. The run can be shortened by using
these patterns. Patterns can name base symbols 
or other patterns, so one symbol high in the stack can stand for a long stretch of the run. 
This substitution is the whole mechanism for compression.

**The compression is lossy, in two places.**
```
placement   an offset is kept to one significant digit in base 2, so a far neighbor is placed
            only to within the power of two it rounds to                                        D6
evidence    a neuron decides its structure over its last H activations and the history slides, so
            the structure that would restate a frame long past is neither held nor recoverable  D18
```

Both losses are why the file is a yardstick and not an artifact (D12). Imagine the run written out under the
structure as it now stands, and read the objective as: make that shorter.

## 1.2 Execution: the best estimate for the situation

**There is no return, no horizon and no value function.** What the machine holds is one number per situation,
distance and action: the mean reward that action received when it ran at that distance from that situation
(R31). It is an exact mean over every exposure the connection has ever had, never discounted, never decayed and
never windowed. Every frame, each action dimension runs the action with the largest such mean among the
situations then standing on the apex (R36). That is the whole of it — **act on the best estimate you hold, for
the situation you are actually in.**

**Credit is attributed, never propagated.** A reward names the frames it pays for and reaches each of them at a
strength falling with distance (R33). No estimate is ever computed from another estimate, so nothing bootstraps
and nothing has to converge before anything else can be read.

**The two objectives meet in the neuron, and nowhere else.** An estimate is held by a neuron, so which
situations can hold an estimate at all is settled entirely by the compression objective: a situation acquires an
estimate of its own exactly when a pattern is minted for it, and until then only the coarser situations
containing it have anything to say (R35). Compression is not preprocessing for control; it is what defines the
space control acts over.

---

# 2. Frame processing

A frame arrives carrying what each event dimension observed, what each action dimension executed (D8), and any 
rewards for actions already run (R33). The machine works **up one stack, a level at a time**.

```
per level, in this order and no other

  bids + calls (R20)  
  the machine calls every neuron that fired at that level to process the frame. 
  inputs: past and present neighbors at age=0.
  output (age=0): patterns to cover given neighborhood, add/delete child requests
  
  election (R24)
  the machine greedily covers the residual based on their cost. 
  each pattern is credited the free neurons it names, until the best left covers no more than it costs.
  the level's uncovered neurons stand as themselves.
  
  allocation (R16, R17) 
  the machine allocates every child requested at this level: 
  neuron id, its parent, its level and the coordinate it inherits. 
  then, it activates it one level up beside the election's winners. 
  it does no work of its own until it next fires.        

  the level above is built out of what the election accepted and what allocation added, and
  it happens again; a level that produced neither has no level above it this frame (§15)

then, once the last level has run

  ledger (R16, R18)    the machine builds the neurons it allocated this frame, and deletes every retired
                       pattern now due                                                           
  learn (§17, R33)     every uncovered event activation connects to the apex action that ran, and a
                       reward moves the estimate of the connection at its distance
  infer (§20)          the apex reads its connections — only events infer — and every inference
                       expands down to base actions
  consensus (R28, R36) one winner per action dimension, at the base, for the frame ahead, by the estimate
                       it carries
```

The last pass commits the action for the frame ahead; the reward for it arrives with that frame (R29).

---

# 3. The machine

## 3.1 The substrate

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
> carries the dimensions of its own kind.
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

**Three objects:**

```
NEURON       a symbol, and a type. Sits at one level, in one dimension of one channel.
             Holds a table of patterns, a history, and connections.                      D16

PATTERN      a set of past and present neighbors, one line of one neuron's table, and the
             child neuron it promotes. Lives in its parent.                              D16

ACTIVATION   one occurrence of a neuron, at a frame and a position. Holds the
             neighborhood it observed, and the cover chosen for it.                      D16
```

**A pattern is a pointer to a child, and a child is a neuron.** One add request creates both: the parent
gains a **pattern**, a line in its own table that may enter a cover at once, and the machine mints the **child** it
points to, a neuron one level up (R16), and wires it to the parent.

What fires is an activation; what a level elects is a bid for a pattern's child; 
what the dictionary writes is a pattern.

> **D3 — Channels and dimensions.** No mechanism mints a channel; what grows is the population
> inside one, level by level and without bound. The channel set, and with it the dimension set, is a fixed
> enumerable index over the whole run, which is what lets `(dimension, offset)` name a slot at any level.

## 3.2 Space

> **D4 — Reach.** How far a neuron sees, either way, in every activation dimension of every channel (D1). It
> is **1** at the base and **doubles every level**:
> ```
> reach(k)   =   2^k          every activation dimension
> ```
> `reach_t` is this reach in the time dimension, at the neuron's own level — its window. In time,
> `W = reach_t + 1` is the depth of the frame buffer — 2 at the base. The buffer is a sliding window:
> an activation sits at its newest edge when it fires, with `reach_t` frames of context behind it, and reads
> them there and never again.

> **D5 — Adjacency.** Two activations are neighbors when they are **within reach in every activation dimension
> they share** (D4), and **adjacency reaches only toward the past**: what an activation's neighborhood holds is
> what fired beside it and what led to it. In space both directions count — a neighbor three positions to the
> right arrives in the same frame as one three positions to the left, and both are in the neighborhood. In time
> only the past is, because compression only reads the past. **What fires after an activation is not a
> neighbor.** The one thing about it the machine keeps — the action that ran — is recorded on the neuron as a
> connection (D25), and it is never in the neighborhood and never in a pattern.
>
> Neighbors are always at the neuron's own level, since the symbols a level offers are what its neurons draw
> neighbors from, **and always of the neuron's own kind**: an event's neighbors are events and an action's are
> actions.

> **D6 — Offsets.** An offset component is the coordinate difference `x` with its magnitude **rounded down to
> a power of two**:
> ```
> offset(x)   =   sign(x) · 2^floor(log2 |x|)          x ≠ 0
> offset(0)   =   0
> ```
> So 5 and 7 become 4, and −13 becomes −8. The reachable offsets are
> `0, ±1, ±2, ±4, ±8, ±16, …`; `G` groups give `2 + G` offsets per direction across a reach of `2^G`, and
> `reach(k) = 2^k`, so `G = k`. **The reach and the granularity are therefore the same power of two**: a level
> reaching `2^k` names its outermost offsets in groups of `2^k`.
>
> **A coarse offset may carry more than one neighbor**, since it spans a range and several activations of one
> dimension can fall inside it. A pattern names per `(neuron, offset)` (D15) and `|e|` counts every neighbor named
> there. Above the base a `(dimension, position)` may itself carry several activations (D8), which this handles the
> same way and needs no rule of its own.

> **D7 — Neighborhood.** An activation observes the active neurons that adjacency admits (D5), each tagged
> with its **offset** — the difference of activation coordinates, one component per activation dimension the two
> share (D1, D2). That set is the activation's **neighborhood**, written `O`:
> ```
> O = { (p, −4), (a, −3), (r, −2), (i, −1) }             a stream:  one component, time
> O = { (k, 0, −1, 0), (k, 0, +1, 0), (m, 0, 0, −2) }    an image:  three, time and two axes
> ```
> the first for a neuron `s` in a stream reading `p a r i s`. At temporal offset 0 a neighbor co-occurs and at
> negative offsets it led here; **both are the same kind of thing**, and a spatial component is one more of the
> same. **A neighbor is a neuron at an offset**, whichever way the offset points in space; in time it is at or before
> the frame, and nothing after the frame is a neighbor (D5).
>
> **A neighborhood is whole the frame the neuron fires**, since nothing in it is later than that frame (D5), and
> every structural decision is made on it.
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other
> at a nonzero spatial offset. At the base, offset zero in every component is the activation itself, and that
> is the center; above the base several children promoted at one coordinate (D8) are each other's neighbors at
> offset zero.

A neighborhood is a set of neighbors, each at its own offset. Drawn with a row per dimension and a column per
offset, with one activation dimension and a reach of 1:

```
                       offset
                   −1      0
        dim A       a      ◉        ◉  the activation
        dim B       b      ·        a  a named neighbor
        dim C       ·      c        ·  silent
        dim D       d      ·
                  ╰──── O ────╯
                   whole at age 0 — covered, priced, added and retired here
```

## 3.3 Activations

Every frame, each event dimension quantizes what was observed (if anything) and each action dimension
carries the action executing in that same frame (if there is one).

> **D8 — Activation.** A neuron is activated (fires) only when something happens: an event neuron input, or an action neuron
> output when its action is executed.
>
> **A neuron may fire many times in one frame**, and those activations are instances of one type.
>
> **At the base level, at most one neuron fires per dimension per position in a frame.** The input reports one symbol
> per channel at each point of its layout, and a dimension with nothing to report there is silent. That bound
> is a property of the input, not a rule the machine imposes.
>
> **Above the base level, multiple neurons can fire for the same dimension and position.** A neuron can choose to cover with
> multiple patterns (R9), so several of its children may be active at one coordinate, since they inherit coordinates. 

> **D9 — Activation window.** An activation at frame `f` stays open through `f + reach_t`, the reach in
> time at its neuron's own level (D4). Two things reach it over that span, one frame at a time, after every
> level has run:
> ```
> the apex action  the action that ran this frame, in each action dimension — only while the
>                  activation is uncovered (D10)                                              D25, R31
> the rewards      for that action, and for the actions of earlier frames the reward spans,
>                  each at the distance its frame names                                       R29, R33
> ```
> The machine holds the activation open, not the neuron.
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since it fired, `0` through `reach_t`; a new activation is the one at age 0. **Age is read, not just
> counted** — rounded as every offset is (D6), it is the offset at which the activation strengthens connections,
> and every offset beyond it is read against it for what comes next (R31, R36).
>
> **Covered at** is the age at which an accepted bid first covered the activation, none until one does.
> Coverage is acquired late and never revoked (R27), so this one number says for every frame of the window
> whether the activation stood on the apex there: it did at every age before it, and at none from it on.

> **D10 — Inhibition.** A neuron an accepted bid covers does not stand in the file, does not infer actions,
> and does not connect to the action that runs. **A neuron is covered only after it has fired** (R27), so
> coverage acquired at a later age stops all three from that frame on and revokes nothing before it: the
> exposures the activation wrote while uncovered stand, and a reward for one of those frames still reaches the
> connection it wrote (R33). An activation covered already at age 0 writes nothing, ever (R17). That is the
> whole of inhibition in the design.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what the decoder
assumes for anything the file does not state.

> **D11 — Identity.** A neuron's identity is fixed entirely by its neuron dimensions, so nothing
> about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: both carry the same relative neighborhood, so the same patterns cover them and one pattern
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

## 3.4 The file

> **D12 — The file.** Two parts, both spanning the whole run. **The dictionary**: one line per pattern, the
> neighbors it consists of (D15). **The body**: every neuron no
> accepted bid covers (D10) — each pattern among them followed by the neighbors it names that did not fire
> (error correction), and each bare neuron standing as itself.
>
> **Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
> file exactly as an observed event does, compressed by the same hierarchy (§18).
>
> **It holds nothing about the future.** A neuron's connections — what actions followed its activations, and what
> they earned — are in no
> dictionary line and not in the body (D25), and neither is an inference that did not run (R32).
>
> **It holds nothing about the search either.** Populations, estimates and margins are the machine's and
> never the file's, because expanding an apex neuron needs the patterns and nothing else.

> **D13 — Prices.** Every cost in the design is part of a file, counted in symbols:
> ```
> a child on the apex          =  1                 a line in the body
> what it got wrong            =  number of neighbors its pattern names that did not fire (error correction)
> a base neuron on the apex    =  1                 its own line
> having a pattern             =  1 + |e|           a line in the dictionary, |e| the neighbors it names
> ```
> This is a fixed-length code: a symbol costs one regardless of how often it is used.

> **D14 — File length.** Over the run the file is
> ```
> L  =  Σ over the dictionary  ( 1 + |e| )                                    written once
>    +  Σ over the base neurons on the apex  ( 1 )                            the body
>    +  Σ over the children on the apex  ( 1 + the neighbors named and absent )  error correction
> ```
> summing D13's prices over the two parts D12 gives. **There is one `L`**, and every neuron's structure is
> priced against it. Nothing computes it: every quantity the design uses is a **difference** in `L`, which is
> finite however long the run is (R12).

---

# Part I — The past and present: a neuron

# 4. The neuron

A neuron is a symbol, and a type (D2). It holds a table of patterns and a history of past/present observations (D16).

**The machine reaches a neuron through five calls, and nothing else writes into one.**
```
create neuron            the machine mints it once the last level has run, one level above its
                         parent, holding nothing                                                R16
process frame            in its level's turn: everything structural for the activations that
                         fired this frame                                                       R20
wire child to pattern    the machine hands back the child it minted for the candidate the call
                         requested, and the pattern points to it                            R16, R17
delete pattern neighbor  the machine removes a neuron that no longer exists from every pattern
                         and saved activation that names it                                     R18
process actions          after every level has finished, reaching every open activation at
                         whatever age it stands at: it delivers the apex action to those still
                         uncovered and any reward, and collects what the apex speaks            §17
```

Every test the neuron runs is its own arithmetic over its own evidence, and it is never told what the board did
with its bids (R24).

Part I covers `process frame`: what a neuron holds, and what it does in the frame it fires. 
Part III covers the `process actions` call, where a neuron learns what action followed and infers the next.

> **R1 — One decision point: the frame it fires.** A neuron is called once per activation, at age 0, and
> everything structural happens in that call: it covers its neighborhood and saves it, re-centers, builds and
> prices one candidate, retires at most one pattern, and returns a bid for every pattern that applies together
> with its requests (R20).

# 5. State

> **D15 — The pattern.** A set of past and present neighbors that a neuron **names**: one line of its table,
> spanning the same box a neighborhood does (D5), and shaped exactly like one (D7). It is the collapse of the
> neighborhoods it covers (R7), moves as they move (R8), and promotes a child (R16).
>
> **A pattern is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a pattern is what the neuron claims. A neighborhood is a fact, a pattern a claim.
> Neither is a frame: a frame is one column, either of these the whole window. What action followed is in
> neither, and is held as the neuron's connections (D25).

> **D16 — Neuron state.**
> ```
> neuron           = (coordinate, patterns, history)      and its connections, defined in Part III  D25
>
> pattern          = (id, neighbors, child)               the neighbors are what it is
>
> history          = the last H activations, oldest first
> activation       = (position, neighborhood, cover, assignment)
>
> held by the machine, not the neuron (D9):
> open activation  = (the activation, its age, covered at)  one per (neuron, age, position)
> ```
> An `id` is creation order, a handle that survives re-centering and the tie-break R9 and R24 reach for.
>
> **No activation carries a frame number**, and nothing anywhere holds absolute time: an open activation's `age`
> is a counter, and `covered at` is the age at which an accepted bid first covered it (D9).
>
> **An open activation carries no commitment beyond that one number.** It holds its neighborhood, the age it
> was covered at, and nothing else, because everything else it was going to decide was decided at age 0 (D9).
> What it does for the rest of its life is connect to what ran and speak from its connections while it is on
> the apex, and that is Part III.
>
> **The cover is the set of patterns chosen to explain an activation's neighborhood**, chosen once, when the
> activation is saved (R9). Each pattern of the cover is credited the neurons it names that fired — its `covered`
> set — and every present neighbor is credited to exactly one of them. What no pattern of the cover names is the
> **residual**: a set of neurons, not a pattern, each standing in the file as its own line (D20, D21). Every price
> in the design is counted off those three sets, and §9 prices them. The **assignment** is the record of that
> credit: for each present neighbor, the pattern of the cover it was credited to, and none when it is in the
> residual.
>
> **An activation's cover is held, not derived on demand.** It is chosen when the activation is saved and replaced
> only by a strictly cheaper one (R10), so two activations with identical neighborhoods can be covered differently,
> and the history carries each cover with its activation.

# 6. The history

> **D18 — History size.** A neuron's history holds its last `H` activations and no more; its connections are
> not in the history and `H` does not bound them (R31). `H` is declared once for the
> machine and is the same for every neuron; it counts that neuron's **own activations**, not a stretch of run, so a
> neuron that fires constantly and one that fires rarely weigh their patterns against the same amount of
> evidence. **The ring is exactly `H` deep once filled**, and how much run it spans is whatever that neuron's
> rate makes it. Nothing else anywhere is measured in frames.

> **R3 — Aging is by count.** The ring is a FIFO `H` deep: an arriving activation evicts the oldest, and only then.
> Nothing compares a frame number, nothing accumulates arrears, and nothing sweeps the population per frame — a
> neuron that does not fire evicts nothing.
>
> **An activation enters the ring whole and never changes.** Its neighborhood is complete at age 0 (D7), and that
> is all the ring holds of it: the action that follows the activation strengthens the neuron's connections and
> is saved nowhere (R31), so eviction touches no connection. **Eviction does not close the activation** — the
> machine holds it (D9), and it keeps connecting and speaking from the apex until its window ends or coverage
> arrives.
>
> Recording is unconditional, and **no election outcome ever edits a history** — deletion is the one thing that
> reaches into a saved activation, and it only removes names of neurons that no longer exist (R18).

> **R4 — Free parameter: the history size `H`.** It is the only one. The alphabet — channels, dimensions,
> resolutions — is the problem statement rather than a knob; adjacency is not declared (D5), the reach per
> level is not declared (D4), and neither is the offset alphabet (D6). **Nothing else in the design is
> tuned, and nothing anywhere is capped.**

> **R5 — `H` and the reach constrain nothing in each other.** `H` counts activations and the reach sets how wide
> one activation is. **What the two share is the collapse's evidence**: R7 votes per offset slot over the same `H`
> activations, so every slot — innermost and outermost alike — is decided on the same count, and a reach wider than
> the data supports finds no majority in its outer slots and they drop.

# 8. Collapse and re-centering

The collapse is the only operation anywhere that decides what a pattern names.

> **R7 — The collapse.** Over a population that each has something to say about one neuron at one offset, let
> `n` be the size of that population and `count(p)` the number of it naming `p` there.
>
> **Backward, `p` is taken exactly when `2 · count(p) > n + 1`.** Naming it covers it in `count(p)` activations,
> states it wrongly in the other `n − count(p)`, and costs one in the dictionary line (D13), so naming it
> shortens the file by `2 · count(p) − n − 1` and it is taken when that is positive. **At `2 · count(p) = n + 1`
> the slot keeps what it had**: a pattern that names `p` keeps it, one that does not leaves it out, and a
> candidate, which has nothing, leaves it out. Nothing is divided, and over a fixed population nothing flip-flops
> at the boundary. **The ring is not fixed**: an activation enters and one leaves at every call, so a slot whose
> count sits at the boundary follows them, and naming it raises the pattern's price wherever it is absent, which
> can cost the pattern a cover and re-decide every other slot on the smaller population. Each step is the right
> response to the evidence, and what it adds up to on stationary input is flicker around a fixed point, confined
> to slots at the boundary — an amount that is measured, not proved.
>
> **Connections are never collapsed** (D25). A connection keeps how often its action followed and what it
> earned (R31); selection reads the whole distribution (R36).
>
> **An activation abstains only where the answer is already settled for it.** Backward, a neighbor another pattern
> of the same cover holds is out of the population at that slot entirely. Naming it would move nothing — it is
> already accounted for, so this pattern gains no `covered` by naming it, and it fired, so this pattern pays no
> `price` for not naming it (D20). The slot's population is the activations where that neighbor was in the
> residual or was not there at all, and the majority is over those: the activations the pattern covers, less the
> ones abstaining there, against the ones among them that saw that neuron there. **A pattern therefore grows into
> the residual and never into ground another pattern holds**, and two patterns covering the same activations
> cannot converge on one set of neighbors. Otherwise an activation held that neuron at that offset or it did not,
> and either way it is in the population.
>
> **This is the only abstention in the design, and it is per slot rather than per activation.** An activation still
> says something about every other offset; it is silent only where the question has already been answered for
> it.
>
> **Two populations.** One arithmetic, twice, and nothing else in the design decides what a pattern
> names.
> ```
> a pattern's         the activations it covers, less those abstaining    R8, at every call
> a candidate's       the activations whose residual holds its seed      R14, once per call
> ```

**One denominator, every offset.** Every activation a pattern covers has something to say at every offset — a
neuron or a silence — so the outermost offset is decided by the same population as offset 0, less only the
activations that abstain there. **No threshold, smoothing or probability estimate enters any of this**, and no
denominator is ever shared between two populations.

> **R8 — Re-centering.** A pattern **re-centers** by running the collapse (R7) over the activations it now covers,
> and the activations it covers then re-derive their covers with it (R10).
>
> **A pattern re-centers whenever its population moves** — no test and no gate. Three things move it, all in
> `process frame` (R20): an activation it covers is saved, one is evicted, or an activation's cover changes so
> that the pattern joins it, leaves it, or is credited differently within it. A cover changes when the table
> changes under it — a pattern added, retired, or re-centered so that R9 reads the activation differently — and
> only when the new cover is cheaper (R10). Every move disturbs a whole activation's worth, so a re-center is
> always over all offsets and never per activation.
>
> **A call re-centers once**, after the activation is saved and before the tests (R20). What the tests then move
> — a candidate joining covers, a retired pattern leaving them — is re-centered at the next call, so the center
> never turns on the order two moves happened to run in.

# 9. Recognition

A neuron recognizes an activation by covering it: every pattern of its table is measured against the
neighborhood, and the ones that pay are taken. What one pattern is worth over one activation comes first, then
the procedure that chooses the set.

> **D20 — Coverage and residual.** A pattern `e` measured against an activation `O` divides it in two, and
> every price in the design (D13) is counted off them:
> ```
> covered   O ∩ e                                        the neurons it names that fired
> residual  O \ (the union of the patterns covering O)   the neurons nothing accounts for
> ```
> **The residual is not an error.** Each of its neurons stands in the file as its own line, at cost 1 (D13),
> exactly as it would if no pattern existed — so it is charged to no pattern and credited to none (D22).
>
> **An activation is covered by a set of patterns, not by one** (R9) — that set is its **cover**. The patterns of
> a cover partition what they cover: each present neuron is `covered` by exactly one of them, so nothing is
> accounted for twice. What is left over is the residual.
>
> **`cover` and `covered` are different objects.** The cover is a set of patterns, held by the activation; `covered`
> is a set of neurons, one per pattern, and it is the quantity the margin is read off (D22).
>

> **D21 — The residual is not a pattern.** What no pattern covers is not routed anywhere and has no line to
> pay: it is a set of neurons, each standing in the file as itself (D20). **There is no default pattern, no
> fallback and no empty pattern** — a table may be empty, and an activation it covers nothing of costs
> `1 + |O|`, which is what an uncompressed chunk costs.

> **D22 — Margin.** What one pattern is worth over one activation: what it covers (D20), less what it
> costs — its own line, and the neurons it names that did not fire (D13).
> ```
> covered(e, O)  =  | O ∩ e |       how many of the neurons it names fired
> price(e, O)    =  1 + | e \ O |   its own line, and the neurons it names that did not
> margin(e, O)   =  covered(e, O)  −  price(e, O)
> ```
> A neuron `e` does not name is on neither side of this: it costs one symbol whether or not `e` exists.
>
> **The cost of an activation** is what its cover costs plus what nothing covers:
> ```
> cost(O)  =  Σ over the patterns covering O ( price(e, O) )  +  |residual(O)|
> ```
> and an activation nothing covers costs `1 + |O|`, the whole chunk stated flat.
>
> **Distance is a reading of the same three sets.** Where one pattern is measured against the whole of an
> activation, `d(O, e) = |O △ e|` and `margin = |O| − d`; that is the identical number, written against a flat
> baseline instead of a subset one. **The design uses the subset form everywhere**, because an activation's
> cover is a set and only the subset form adds up over one.
>
> **This is the only valuation in the design**, and it is read over two different populations — the neuron's own
> activations (R12, R9) and the machine's board (R22, R24) — so the two numbers differ, and are meant to.

One procedure appears twice in the design — once inside a neuron, over its own table, and once inside the
machine, over a frame's bids. It is stated here and cited from both.

> **R9 — The greedy cover.** One procedure with two callers, stated here once and run nowhere else. Given a set
> of **claimants**, each naming some neurons, and a set of neurons to cover, repeat:
>
> 1. **Measure** every claimant not yet taken against the neurons still uncovered:
>    ```
>    covered  =  | the still-uncovered neurons it names |
>    price    =  1 + | the neurons it names that did not fire |            D22
>    ```
> 2. **Take** the one with the highest `covered / price`, **iff `covered > price`**. The neurons it was measured
>    on are credited to it and leave the uncovered set.
> 3. **Stop** when the best remaining claimant does not pay. Otherwise return to 1 over the smaller set.
>
> **What it returns is the taken set and the credit.** Every neuron a taken claimant covers is credited to
> exactly one of them — the round that took it — so nothing is accounted for twice, and what no round took is
> the **residual** (D20). `price` is fixed by what fired and cannot change between rounds; `covered` only falls.
>
> **The two callers differ in population and tie-break, in nothing else.**
> ```
> caller       claimants               to cover                    ties
> R20 step 3   a neuron's patterns     one activation's `O`        the older `id`
> R24          a frame's bids          the free set of the board   the older symbol, then the
>                                                                  earlier coordinate
> ```
>

> **R10 — Covers are held, not patched.** A moved pattern changes what it covers. What is maintained is
> **one pattern's `covered`-and-price against every activation**: a pattern that re-centers recomputes those, and
> nothing else is repaired. An activation whose table changed under it — a pattern re-centered, added or retired —
> re-derives its cover by R9 over its neighborhood, **and the re-derived cover replaces the one it holds only
> when it is strictly cheaper** (D20). R9 is greedy, so re-deriving can cost more than what stands; holding
> the cheaper is what makes every move a descent (R12). A retired pattern leaves every cover it was in at once,
> and the cover without it is the one the re-derivation has to beat. **A pattern that was just added gives an
> activation three options, not two**: the cover it holds, that cover with the newcomer appended and taking the
> residual it names, and the cover re-derived from scratch — and the activation takes the cheapest. The appended
> cover is what R15 priced, so what adding the pattern realizes is never less than what the test counted.

> **R11 — A price is a measurement, not a record.** What an activation costs is read off its cover as that cover now
> stands (D20), and the cover can change: re-centering (R8) moves a pattern, which moves what it covers in every
> activation, and those are what the activations then cost. **An activation is fixed but its cost is not**, and it
> stops moving when its cover stops moving.

**Cold start is silence.** A pattern covering no activations names nothing, and a neuron with an
empty table covers nothing and bids nothing.

# 10. The one test

> **R12 — The one test.** A pattern earns its dictionary line when the file is shorter for holding it than it
> costs to state. **Nothing measures a file to find that out**: both terms are counts over what the neuron
> already holds, so the margin is the difference in `L` reached directly (§1).
> ```
> benefit(e)  =  Σ over the activations e covers:  covered − price
> cost(e)     =  1 + |e|                                              the line  (D13)
> margin(e)   =  benefit(e) − cost(e)
> ```
> A pattern is **added** only when its margin is strictly positive and **retired** only when strictly negative
> (R15, R18). At equality nothing happens, so the boundary cannot flip-flop.
>
> **`covered` is what nothing else would have covered.** A pattern is worth what it saves over what would
> account for those neurons if it were gone — another pattern of the same cover if one names them, and the
> residual otherwise, where each stands as its own line (D20). A saving some other pattern already
> delivers is not this one's.
>
> **The same expression prices a bid over one frame** (R22). There is one valuation in the design (D22); the
> two readings differ in the population they sum it over and in whether the dictionary line is in the sum.
>
> **Benefit is a measurement, so it moves when anything under it moves** — an activation saved or evicted, a
> pattern re-centered, a cover re-derived. **No test needs a pass of its own.**

> **R13 — One comparison.** There is no second one. An activation is covered on `O`, priced on `O`, and the tests
> that add and retire read the same numbers over the history. No quantity in the design waits for anything.
>
> **What arrives later is evidence, not a verdict.** The action that runs after an activation, and the reward
> with it, strengthen the neuron's connections (D25), which the next inference from the apex reads, never a test.

# 11. Creating a pattern

> **R14 — Where a candidate comes from.** Three fixed steps. Nothing seeds it from outside, nothing grows it a
> neighbor at a time, and nothing repeats until a condition holds.
> ```
> residual(o)  =  the present neighbors of o no pattern of its cover names                     D20
> seed         =  the neighbor in the most activations' residuals — ties to declaration order (D1),
>                 then to the nearer offset
> population   =  the activations whose residual holds the seed
> C           =  the collapse (R7) over that population, per slot
> ```
> `C` is the candidate so built — a pattern not yet in any table (D15).
> The seed is in every activation of the population, so `C` names it once the population holds two activations; over
> one activation `2 · 1 > 2` fails and the collapse names nothing, so **nothing is ever built on a single
> occurrence**. Every other neighbor `C` names is present, and in the residual, in more than half of the
> population: R7's abstention applies as it does everywhere, so a neighbor a pattern of the cover holds in an
> activation is out of that slot's population, and **a candidate is built on the residual and nothing else.** The
> seed is the neighbor the table is failing on most, and the collapse settles every other slot at once — the
> seed chooses the population, and the population decides every slot.
>
> **Only a set of neighbors is built, because that is all a pattern is.** What actions the child will be followed by
> is the child's own connections, formed by the child's own activations once it exists (D16). Nothing about it is
> decided here and nothing about it is priced.
>
> **The same history under the same covers yields the same `C`.** Covers are held rather than derived (R10),
> so the residual, and with it the seed, is a function of the ring and the covers it carries together.
>
> **What `C` is worth.** Against an activation, `C` takes neurons out of the residual and names some that did not
> fire:
> ```
> reach(o)   =  |residual(o) ∩ C|  −  |C \ o|          what C is worth there, before its line
> saving(o)  =  max( 0,  reach(o) − 1 )                    what it is worth once the line is paid
> ```
> **A candidate is only ever credited the residual.** A neuron a pattern already covers is not `C`'s to take —
> a candidate that fits a chunk beautifully earns nothing for it if something already accounts for it.
>
> **The floor is the cover test and belongs to pricing only.** An activation whose cover `C` would not join contributes
> nothing to the benefit: a candidate has to be able to reach an activation before it pays in it.
>
> **What it is not.** It finds a local best and not the best `C` — choosing the pattern set is the
> facility-location problem and is not solvable exactly at any useful size. What it has instead is no free
> choice anywhere in it, and one candidate per call, which is the rhythm the machine keeps with one election
> per frame.

> **R15 — The solo test.** R12, asked of a table `C` is not in yet.
> ```
> benefit  =  Σ over the whole history:  saving(o)      (R14, floored — one line per activation already in it)
> commit iff  benefit > 1 + |C|
> ```
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every activation in the ring has a whole neighborhood, so the question is
> asked over the same evidence the cover will use when `C` next competes for one. Nothing here is decided on half
> an activation and nothing later can hand `C` less than the test counted.
>
> **One candidate per call, whether it pays or not.** What a single candidate leaves uncovered is the next
> call's residual, and the next call's seed is whatever is then failing most.

> **R16 — What a child is at birth.** The parent requests; the machine creates. The child inherits its
> parent's channel and dimension and is minted one level above it, all carried on the request. It is created
> with **an empty table**: its own patterns belong to its own level, which it has not observed yet. Its
> *existence* is decided by its parent, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each is one pattern's child, each covers the
> part of an activation its pattern was assigned, and several of them may be promoted at one coordinate (D8). What
> they share is a parent and a coordinate, not a slot.
>
> **Release is the same shape reversed**: the parent retires, the machine reclaims. A retired pattern goes back
> on the same request that carries the candidate (R20), so a call touches the alphabet once — in one direction,
> both, or neither.
>
> **Allocating and building are separate, and only building waits.** The machine allocates the child in the
> level the request came from — id, parent, level, inherited coordinate — which is all the frame needs to
> activate it and route to it (R17). Once the last level has run it builds the neurons it allocated and
> reclaims every retired pattern's child now due (R18), in one pass over its own alphabet. **The wait costs
> nothing** because a child runs no call of its own in its mint frame (R17), and a retired pattern's child
> cannot be reclaimed sooner in any case (R18).

> **R17 — A child requested in a call is not offered in it.** The two objects part company here (R16): the **pattern**
> is the parent's and joins its table in the call, so this activation's cover may already take it (R10); the
> **child neuron** is the machine's and is minted in the machine's pass, once the last level has run (R16).
> Nothing in the frame reads it, so nothing waits on it. It is first *bought* — joins its level's frame, covers,
> bids, speaks — on the next activation of its parent whose cover its pattern takes and whose bid the election
> accepts. **Structure never pays off on the evidence that created it, only on recurrence.**
>
> **A child is allocated in the level and built after it.** The machine allocates the identity where the
> request is made — an id, its parent, its level and the coordinate it inherits (D2) — which is all the frame
> needs to activate it and route to it. What waits for the machine's pass is the neuron itself (R16).
>
> **It is born holding nothing.** A child is minted at its parent's age 0, so there is no span behind it to
> wire: no patterns, no history and no connections (R16). Everything it comes to hold is over the situations its
> parent's pattern actually took (D25).
>
> **It fires in the frame it is minted, and does no work in it.** It is activated one level above its parent,
> beside the neurons the election promoted, so the level above sees it and may cover it, and its parent is
> subsumed under it as under any activated pattern (D10). What it does not do is run a call of its own: it
> covers no neighborhood, bids nothing, mints nothing, retires nothing, learns nothing, and neither speaks nor
> votes (R20, R27, R36) — it has no history at its own level to do any of it on. **Without that exclusion one
> mint would cascade up the levels inside a single frame.**
>
> **The mint frame costs one exposure, and that is deliberate.** A parent whose candidate is accepted is
> covered by its fresh child in the same frame — covered at age 0 — so the parent writes nothing for that
> activation (D9, D10). The child, being uncalled in its mint frame, writes no connection until the frame after
> (D25). Neither one records the action that ran or the reward that arrived for that single frame: the parent
> because coverage arrived before age 0 closed, the child because it was not yet open. One lost exposure per
> mint, and no more.
>
> **Its own work begins the next time it fires.** From then it is called with its level like any neuron (R20),
> and what it comes to hold is over the situations its parent's pattern actually took (D25).
>
> **The rest of the activation's life is connecting and speaking** (D9, §17) — the neuron's connections
> strengthened by what runs, and while on the apex, an inference read off them.

# 12. Deleting a pattern

> **R18 — Retire one, then delete.** Read every margin in the table (R12) — the table as re-centering left
> it, and before this call's candidate is built (R20):
> ```
> benefit  =  Σ over the activations it covers:  |neighbors only this pattern names|  −  ( 1 + |e \ o| )
> retire the pattern with the smallest margin, iff  benefit < 1 + |e|
> ```
> **One per call, and no other.** Two patterns naming the same neurons are each worth nothing while the other
> stands, so retiring both on one reading would return their neighbors to the residual with nothing left to
> cover them. Retiring the worst lets the survivor take full credit at the next call, and the next call reads
> every margin again.
>
> **Without the pattern its neighbors fall where D20 puts them** — to another pattern of the same cover that
> names them, at no extra cost to that pattern, or into the residual at one line each. It is the same
> difference R15 reads with the roles swapped: adding a pattern asks what one that is not there
> would take out of the residual, retiring one asks what one that is there is still keeping out of it.
>
> **Retiring is a deletion in the parent.** The pattern leaves the table that instant. It stops competing for a
> place in any cover, so no further activation can bid it, and the neurons it held fall to whatever D20 gives
> them next (R10). Having nothing to cover it has no margin and nothing to re-center — **the neighbors it
> held stop moving** — and it is not a candidate for anything again. What leaves the table rides the call's
> return to the machine (R20), as a request to delete it. **The neuron keeps no retired state and re-checks
> nothing.**
>
> **The death frame is the machine's to set, and it can be this frame.** The neuron asks for the child to go
> and says nothing about when; it sees its own open activations and not the children promoted off them, while the
> machine sees both.
> ```
> death frame   =   when the child's last open activation closes
>               =   this frame, when none is open
> ```
> That set only shrinks. A child has one parent (D2), so once that parent stops covering with it
> nothing can fire it again, and no level built afterward can name it either, because a level is built out of
> what is firing (R26). Reach grows with the level (D4), so the last to close is the highest one and the wait
> is at most `reach_t(D)` frames, `D` being the highest level the stack currently holds.
>
> **Deleting** is the machine's, on the same pass that mints (R16): **every frame, once the last level has run**,
> it reads the **death ledger** and takes everything due — the pattern, its child neuron and that neuron's subtree
> together, and what named them scrubbed with them. A pattern retired this frame with nothing open dies on
> this frame's pass; one whose child is still open waits exactly as long as the stack above it needs, and not
> a frame longer. **Nothing traces who is naming what**: the machine settles the question off the board it
> already keeps.
>
> **The ledger holds the pattern, not a handle to it.** A child is stated in one place, its parent's pattern
> for it (D12), and the child is expanded through that pattern (R28). Until the death frame, neurons
> above it still cover it and the apex may still expand it, so the definition has to stay readable after the
> table stops covering with it.


> **R19 — The table needs no rule against duplicates.** Two patterns with equal neighbors present R9 with
> identical input. It takes the older first, so the older covers everything the younger would, the younger is
> assigned nothing anywhere, and a pattern covering nothing fails R18. The tests remove them.

**A deletion takes the subtree, and takes it at once.** There is no staged cascade and nothing to wait on at
any level.

**Nothing is retired for what followed it.** What followed is measured, not claimed (D25), so a pattern is
never charged for what came after — only for what it names that did not fire beside it.

# 13. The process frame call

One call per level per frame: everything structural for the activations that fired this frame.

> **R20 — The call, in order.** Once per level per frame, the machine asks one neuron for everything it owes
> that frame. **One population answers**: the activations that fired *this* frame, at age 0 (R1). What an open
> activation learns of what followed names the apex, which no level knows, so it is written after every level
> has run (§17) and never here. The right-hand column says which steps are per activation and which per neuron.
> ```
>                                                                                        over
>  1  evict      a full ring drops its oldest activation, and every pattern of that       each new
>                activation's cover loses it                                  D18, R3     activation
>  2  admit      the new activation joins the ring, whole (D7). No pattern has taken      each new
>                anything yet, so the whole of `O` is residual                    D20     activation
>  3  cover      run R9 over the table against `O`. What it takes is the **cover**      each new
>                and what it credits is the **assignment**             R9, D20, D16      activation
>
>  4  re-center  every pattern whose population moved — an activation joined it,             the
>                left it, or is credited differently — re-centers, once; every             neuron
>                activation whose table moved under it re-derives its cover and
>                keeps the cheaper                                             R8, R10
>  5  retire     read every margin in the table; retire the worst if it is strictly          the
>                negative. It leaves the table at once, and the activations it              neuron
>                covered re-derive without it                                R18, R10
>
>  6  offer      a bid for every pattern of the table more than half of whose           each new
>                neighbors are present — `2 · |e ∩ O| > |e|` — whether or not the        activation
>                cover took it. A bid is the child's id and the pattern           R21
>  7  build      seed, population, collapse (R14), then price it (R15)                       the
>                                                                                           neuron
>  8  return     the bids, and one request carrying the candidate that passed and             the
>                the pattern that retired                                    R16, R21       neuron
> ```

**The call runs before the election**, and the election is over bids from a table that has already saved this
frame, re-centered on it, and retired against it.

# Part II — The past and present: the machine

# 14. Contraction

> **D23 — Contraction.** The machine covers the level below with neurons from the level above, each taken when
> it covers more neurons than its bid costs to state — `1 + |e \ O|`, never the dictionary line (R22, R24).
>
> **Covering everything is not the goal.** A neuron no accepted bid covers stays in the file as itself, at cost 1
> (D13), and that is the shorter file whenever no neuron could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a pattern names neighbors at offsets, so a promoted neuron replaces a chunk of
> spacetime. Spatial contraction is the case where every offset is zero.

## 14.1 Bids

> **R21 — A bid is a pattern and a name.** A bid carries two things and no others:
> ```
> the pattern   its neighbors — the dictionary line (D12)
> the child     the id of the child this pattern would promote
> ```
> The pattern travels because it *is* the line for the symbol being proposed, and the bidder is implied,
> because a child *is* its parent in that pattern. **No connection travels**: nothing at
> `Δt > 0` has fired, and the file has no line for what follows (D12).
>
> **One activation may send several**, one per pattern that applies (R20), and they are independent bids: each
> answers for what the election leaves it, and the machine has no reason to know they came from one neuron. A
> neuron covering nothing sends nothing.
>
> **Nothing else is sent, because nothing else is the neuron's to know.** Which of the named neighbors actually
> fired, what this bid is worth against them, and what another bid has already taken are facts about the frame
> — and the machine is holding the frame. It reads the pattern against its own board and derives the rest
> (R22).

> **R22 — What a bid covers, and what it costs.** The neuron sends the pattern (R21) and nothing else.
> The machine holds the frame, so it reads that one object against what fired and derives both numbers.
> ```
> the bid   the pattern, and the child's id                                        (R21)
> covered   the neurons it names that fired and no earlier bid covers — the slots it asks to subsume,
>           the bidder among them
> price     1 + |e \ O|   its own line in the body, and the neurons it names in those
>                          same frames that did not fire
> ```
> `covered − price` is the saving over stating the chunk flat, D22's expression over the machine's population.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as its own
> line if nothing covers it (D23) and costs a turn-on if this child is promoted — one symbol either way, so it
> cancels before the test begins, and charging it here would count it twice against the uncovered term of the
> same sum (§14.4).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another bid
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged one whether or not another neuron is right at
> that slot — another neuron's expansion being right there does not make this one's wrong name free. **What a
> pattern gets wrong about a frame is a fact about the two, and no assignment moves it.**
>
> **This is the neuron's arithmetic over the machine's population, and the number is not the neuron's.** The
> neuron took the pattern into its cover, or offered it, on its own residual (R20); the machine tallies on a
> board where earlier frames' credit stands (R23) and this frame's other bids contend (R24). The two numbers
> differ, and are meant to (D22).
>
> **This is a price for one bid, not for the symbol.** The dictionary line `1 + |e|` is weighed by the one test
> (R12) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R22).

## 14.2 The board

> **D24 — The coverage set.** Per level, the machine keeps which accepted bid was credited each covered
> activation:
> ```
> coverage set    per level    which accepted bid holds each subsumed active activation
>                 an assignment          one holder per activation; settled slots are never re-assigned
> ```
> **A slot is named by a full coordinate**, dimension and position together, so two activations of one neuron
> at two positions are two slots and never contend. Level `k`'s coverage set spans `reach(k) + 1` frames — a
> bid reaches `reach(k)` back and no further — and the box every other activation dimension gives, and ages
> out with it. **The machine holds nothing on the scale of the run.**
>
> The assignment is about **credit**: a neuron is a fact that needs paying for exactly once, so it is settled
> once and never revisited (R23). It is not about naming: a neuron expands to everything its pattern names,
> credited or not (R24).

> **R23 — This frame's bids against the board as it stands.** Only neurons no earlier frame's election has
> credited are in play, so a chunk already paid for is not paid for twice. **No earlier promotion is ever
> re-scored.** Within the electing frame a slot is credited once, to the first accepted bid that names it
> (R24), and never moves.
>
> **Earlier bidders have priority.** A bid at `f` wins a neuron at `f − 2` before a better bid at `f + 1` can
> name it, because the earlier election settled it and nothing re-elects the past. That is the price of never
> revisiting a frame, and the design pays it.

## 14.3 The election

**The file over one frame is the neurons promoted plus what they got wrong**: `Σ over the accepted (1 + |e \ O|)
+ the neurons no bid covered`, the body half of `L` (D14) over the frames the election can see. **The two
terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first, which is why
the price counts only the neighbors named and absent (R22). The dictionary half is R12's, and neither test
touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — bids are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good assignment, not a proved minimum. **Every neuron a bid covers ends up credited to exactly one
bid**, which is what stops a chunk being paid for twice.

> **R24 — The election is R9, run by the machine.** The claimants are this frame's bids and what they cover is
> the **free set**: every active activation of the level below, at its own full coordinate — frame and position —
> that some bid names and no earlier election has credited (R23).
>
> Bids arrive naming relative offsets, so each is resolved against its own coordinate before the first round.
> `price` is `1 + |e \ O|` (R22). **Ties go to the older symbol, then to the earlier coordinate** — creation
> order for a pattern and declaration order (D1) for a base neuron, then frame, then position. Then R9 runs,
> and what it takes are the accepted bids.
>
> **The bound is structural.** An accepted bid takes at least two slots, since `covered > price ≥ 1`, so the
> rounds are at most half the free set and never more than the bids.
>
> **A bid that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> a slot it named is either credited to a bid that did pay or stands as its own line.
>
> **The assignment is a partition of the neurons the accepted bids name**, and that is the whole of the
> inhibition — no bid is ever edited or forbidden, and **overlap is legal and priced**: a bid that names a slot
> an earlier round credited gains nothing for it and pays nothing for it (R22). **Held by an accepted bid** and
> **named by an accepted bid** are therefore the same set, so coverage, credit and the apex frontier (R27) are
> one question with one answer.
>
> **Outcome**: accepted bids are promoted, one neuron each and **whole** — a child expands to everything its
> pattern names, credited or not — the neurons credited to them are subsumed, and every active neuron no
> accepted bid covers stands as itself. **The election delivers nothing to anyone**: it writes the coverage set
> and stops. No neuron is told which of its bids were bought, what they were credited, or what they lost; a
> neuron's history is what it saw, and the board is the machine's.

## 14.4 When a slot is settled

Settlement is a property of one slot at one full coordinate. **Nothing here delays anything the machine does**
— no pass blocks on it and no decision is deferred by it. **The only consumer is measurement**: when `L` or
apex-neurons-per-frame is read, the settled frames are the ones whose numbers are final.

> **R25 — Settlement is a condition to detect, not a schedule to predict.**
>
> **Frontier membership settles one level, in `reach_t` frames.** Whether an activation at frame `h` is covered
> is decided by bids firing no later than `h + reach_t`, since a bid reaches `reach_t` back and no further.
>
> **A frame's encoding settles at the top of whatever stack reached it.** A neuron one level up, firing later,
> can name a lower neuron that names frame `g`. **Frame `g` is settled when no level holds an open activation that could
> still join or leave that set** — a closure over the levels, evaluated upward.
>
> **`D` is reached, not known.** The walk stops where a level accepts no bids and therefore produces none above
> it, so `Σ_(k<D) reach_t(k)` bounds a condition rather than counting out a delay.

# 15. The order of a frame

> **R26 — One stack, at the derived reach.** Base neurons run `process frame` and offer; the election settles which bids
> are bought. The survivors are level 1 — the fewest that cover the active base neurons — and it happens
> again. **When a level's active neurons promote no children, nothing propagates and there is no level above
> it on this frame.** Nothing declares the depth and nothing caps it.
>
> **Within a level the order is cover, offer, election**, and it cannot be otherwise: the call saves this
> frame's activation before the offer reads the table, and the offer is what the election is over. Nothing in that
> order leaves the level or the frame.
>
> **The whole of it is one call** (R1). The neuron saves, restructures and offers, the
> machine elects — and the activation that did all of it fired this frame, on evidence that was complete when
> it started.
>
> **The frame ends with the machine's own passes.** Once the last level has run, the machine builds the neurons
> it allocated this frame and deletes everything the death ledger has due (R16, R18), calls every open activation
> (§17), then expands what the apex infers (§20) and resolves one winner per action dimension at the base
> (R28, R36).
>
> Every level runs the same rule at the reach one expression gives it (D4). **Compression is spatio-temporal
> at every level, in one pass**: a pattern at any level may name neighbors in its own frame, in earlier ones,
> beside it in space, or in a mix.

> **R27 — The apex is a frontier, not a level.** It is every active neuron **no accepted bid covers** — the
> uncovered set, at every level at once — so a base neuron nothing found worth chunking stands in it beside a
> level-4 pattern. This is the frontier the file's body writes, **the one that learns what ran** (D25), and
> **the one that votes** (§20): the uncovered set does all three, and coverage silences a neuron in every one of
> them at once (D10). A reward for a frame already written reaches its connection regardless (R33). Everything
> underneath the current frontier is recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that offered a child and had its bid declined with nothing else
> covering it is still on the frontier, and stands in the file as its own line.
>
> **Coverage is acquired late and never revoked.** An activation firing at `g` is uncovered until some bid
> takes it, and the last bid that can fires at `g + reach_t` (R25), so an activation may speak for a few frames
> and then fall silent. **No accepted bid is ever dropped** (R23).

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
alongside (D8), so it is recognized and chunked by the rule they are, over neighbors of its own kind (D5). **The
connection is not formed here**: it names the apex action, which is known only once every level has settled, so
it is recorded in the `process actions` pass instead (R31).

---

# Part III — The future: action and reward

# 16. Connections

Everything before this point read the past and the present: what an activation observed, and what the file
pays to state it. What follows an activation is never in the file (D12) and enters no test. One thing about it
is held — which actions followed, and what they earned — as connections on the event neuron, read only from
the apex, and it is the whole of what the machine does.

> **D25 — Connections.** An event neuron holds, per `(action neuron, offset > 0)`, one **action connection**:
> a **strength**, the number of times an activation of the neuron saw that action run at that offset, and an
> **estimate**, the mean reward those runs received (R31). It is a distribution, not a record: the offset is a
> D6 offset like every other, so a coarse offset pools the runs of every frame in its group, and nothing about any
> one activation is kept.
> ```
> connections   per (action neuron, offset > 0), (strength, estimate)        held by event neurons only
>               strengthened one exposure at a time; nothing per activation  R31
> ```
> **They are written apex to apex, and no level is read.** An uncovered event activation connects to the apex
> action that ran — the highest action pattern that fired in that dimension that frame, the base action when
> none did — at whatever level either stands, so a level-8 event can learn to name a level-5 action. **Whether
> an activation writes is decided frame by frame, by the frontier** (R27): it writes an exposure at a frame iff
> no accepted bid covers it at that frame. Coverage is acquired late and never revoked (R27), so an activation
> writes from age 0 until the frame it is covered, and never again; one covered at age 0 never writes (R17).
>
> **They are measured, never chosen.** A connection is not in the bid (R21), not in any dictionary line (D13),
> and it enters no test. Connections are read in one place — **when the activation stands on the apex** (R27) —
> and what it reads there are the inferences that choose the next action (§20). A child neuron fires only when
> its parent's pattern was bought, so a child's connections are the future of that situation and nothing else,
> forming from the frame after its mint on (R17); a base neuron's are the marginal over every situation it fires
> in, and speak only where nothing more specific covers it (D10).
>
> **Every event neuron holds them, base neurons included, and a pattern holds none.** Nothing held recomputes
> them: every activation of the neuron strengthens them, whichever patterns covered it, and nothing ever weakens
> one (R31). Several children promoted at one coordinate are several neurons, so each holds its own connections
> from birth; while they are always bought together their connections agree, and they diverge the first time one
> is bought without the other.
>
> **An action neuron holds no connections.** What actions follow an action is a chunk, and the action hierarchy
> writes it as a pattern (D5); nothing the machine executes is chosen from an action neuron (R35). An action
> neuron is a symbol that fires when its program runs, and it is chunked like any other (§18).

# 17. Once the levels are done — `process actions`

**The machine calls every open activation once more**, at whatever age it stands at, with two things:

```
the apex action    the action that ran this frame in each action dimension — the highest pattern that
                   fired there, the base action when none did. **Only if the activation is uncovered
                   at this frame** — one an accepted bid covers writes nothing more     D10, D25, R31
the reward         any reward that arrived, for the action that ran this frame and for any earlier
                   frame the reward spans, at the distance each one names                       R33
```

The neuron strengthens a connection for the action and saves nothing: the connection at `(action, age)`, created
at strength 1 or incremented (R31), and for each reward share the estimate of the connection it names (R33) — for
the action that ran this frame, the connection just strengthened, in the same write. A share for an earlier frame
reaches the connection the activation wrote to at that frame, whether or not coverage has arrived since. Nothing
is written into the activation. **Nothing is decided, priced or compared here**, and no test is waiting on any of
it.

**If the activation is uncovered, the call returns what it speaks.** It reads its own neuron's connections at
every offset beyond its age, out to its reach — a connection at offset `b` read at age `a` is a claim about an action
completing `b − a` frames ahead (R28) — and returns each with its strength and estimate. Those are its
**inferences** (§20), each naming an action neuron at whatever level that neuron stands (D25), and only what
reaches the frame ahead is resolved. A base neuron on the apex speaks from its own connections like any other;
what it infers is the marginal over every situation it fires in, and it speaks only because nothing more
specific covers it.

**An activation closes at age `reach_t`** (D9), once that frame's exposure is taken. Closing does nothing but
stop the writing — there is no second call and nothing is saved twice.

# 18. Actions

An action dimension carries what the machine executes, and it is compressed by the same hierarchy its events
are (D1, D8).

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> the neighbors its pattern names one level down, at that neuron's offset plus theirs — offsets compose because
> each is a difference of activation coordinates (D2) — repeated to base symbols, one level fewer than the
> neuron's height.
> ```
> A placed at f:      A's line is {(p, 0), (q, −1)}                 → p at f,   q at f−1
> P placed at f+4:    P's line is {(B, 0), (C, −2)},
>                     C's line is {(c, 0), (d, −1)}                  → B at f+4, c at f+2, d at f+1
> ```
> **A coarse offset expands to its rounded coordinate.** A neighbor named at `sign · 2^g` is placed at exactly
> that distance whatever distance in the group it fired at, and several neighbors at one coarse offset (D6) are
> each placed there. **The rounding composes.** Each step down adds its own group's slack, so a level-`k` neuron
> places the base symbols of its farthest neighbors to within the sum of the groups along the path: the higher
> the neuron, the coarser its far placements. That is the loss the file carries (§1), and expansion reports it
> faithfully rather than hiding it.
>
> **This is the one expansion in the design.** It recovers the run from the file (D12) and it turns a selected
> action pattern into a program (R30), and it reads dictionary lines only. **What travels down with a symbol is
> what the connection carried**: every base action an inference's expansion places carries the strength and the
> estimate of the connection it came from, and nothing is re-weighted on the way down.
>
> **A connection is placed the way a neighbor is.** A connection at offset `b`, read by an activation at age `a`,
> puts the action neuron it names `b − a` frames ahead — that is where it completes — and its expansion hangs from
> there, its base actions at the frames back from it. **A coarse offset is not a window.** Its action completes at
> `b`, not somewhere in the group `b` stands for, exactly as a neighbor named at `−b` is placed at `−b` and nowhere
> else. So the steps of a long program reach the frame ahead one at a time, in order, each at exactly one age,
> from one connection and with nothing held: what an activation places beyond the frame ahead it places again
> next frame, one frame nearer.

> **R29 — Two frames: infer, then execute and reward.** What is chosen in one frame runs in the next, and what
> it earned arrives with that frame.
> ```
> f      infer     the frame's events are recognized, `process actions` returns the inferences,
>                  the inference resolves (R36), committing an action for the frame ahead
> f + 1  execute   the action runs, and its neuron fires in this frame's column alongside
>        reward    this frame's events — every uncovered event activation connects to it;
>                  what the action earned arrives as this frame's input, and the connection
>                  it strengthens takes the reward in the same write (R31)
> ```
> **The reward is part of the frame the action ran in.** The environment reports what it observed and what the
> action in effect during that frame earned together, so an action is never on the books without its outcome,
> and no activation has to stay open for a reward that arrives later than the action it pays for (D9).

> **R30 — Execution is an expansion**, of the selected pattern (R36) through its dictionary line (R28). A high
> action pattern becomes its constituent actions at the offsets its line records, down to base actions that
> execute. Execution is not a second mechanism; it is this expansion read as a program. Each base action
> executes in the frame its expansion places it in, the nearest being `+1` (R29).
>
> **Execution activates what it expands.** The base actions fire the frames they run (D8), and every action
> pattern the expansion passed through fires when its expansion completes, at the coordinate the expansion
> placed it — the last frame of its chunk, which is the coordinate recognition would have given it. An executed
> pattern is therefore on its level's frame like a bought one: it runs `process frame`, is chunked upward (§15),
> and every uncovered event activation open at that frame connects to it (D25). While its program runs, the apex
> action at each frame is the base action, or a lower pattern as it completes; the whole is the apex action only
> in the frame it completes, and what the program earned reaches it through the span a reward names (R33).

# 19. Reward

A reward is an input, not a symbol: alongside what it reports observed, a frame may carry rewards for actions
already executed. They reach the machine through one object and one only — the **action connection**, which no
structural test can see (R34).

> **R31 — An action connection carries an estimate.** What executes at `f + 1` is not known at `f`:
> it is settled only once every level has run and the inferences resolve (R36). So the action that ran fires in
> its dimension at `f + 1` (D8), with its reward beside it (R29), and **every uncovered event activation open at
> that frame connects to the apex action at its own age** (D25, §17) — the highest action pattern that fired in
> that dimension that frame, the base action when none did. That connection binds what the neuron stands for to
> what the machine did — formed against what actually ran, so **a neuron that inferred a different action, or
> none, learns from the one that ran.**
>
> **Every uncovered open activation connects to it, at every age it is open at.** The offset is the age — the
> distance from the frame the activation opened to the frame the action ran — rounded as every offset is (D6),
> so a neuron open at ages 1, 2 and 3 holds the same action at two offsets, `1` and `2`, and the exposure at age
> 3 strengthens the second. **A coarse offset takes one exposure per frame of its group**, so the outer offsets pool the
> apex actions of many frames, each at its own strength, as a coarse offset carries several neighbors (D6). An
> action that ran twice inside one group over one activation is two exposures: each run has its own reward, and
> the connection keeps nothing that could tell the two apart (D25).
>
> **Strengthening and reading are inverses**: an exposure is written at the offset its age rounds to, and read
> back at the age from which that offset places a completion one frame ahead (R28, R36), so what a replay earns
> lands on the connection it was learned from. Fan-out is bounded — a neuron names actions only in the channels its activations have seen follow, and one
> exposure per frame per action dimension means a level-`k` neuron holds at most `k + 1` connections per action
> it has seen.
>
> **Making and strengthening are one operation.** A neuron's action connection at `(action, offset)` has a
> **strength**, the number of its exposures — the times an activation of the neuron saw that action run at that
> offset — and an **estimate**, the mean of the reward those exposures received (R33). The first exposure creates
> the connection at strength 1 and every later one increments it; each share of reward folds into the mean
> weighted by `1 / strength`, so the estimate is the exact average over the connection's exposures and no rate
> is chosen. **Nothing leaves.** No exposure is ever subtracted, no strength ever falls, and no window bounds a
> connection: it is the whole of what the neuron has seen follow it at that offset, over its life.
>
> **Every level holds them, base neurons included.** No operation derives what a child was worth from what its
> parent earned or the reverse, so a connection held only at the frontier would be lost at the next mint. Every
> neuron keeps its own connections over its own activations (D16).
>
> **A connection dies with either of its ends** — the neuron, or the action neuron. Nothing else removes one (R34).
>
> **A connection wired ahead of any exposure is created at strength 1 and estimate 0.** The walk (R37) is the one
> thing that creates a connection nothing has yet been seen to follow; it is created exactly as a first exposure
> at neutral reward would create it, and from then on it is a connection like any other.

> **R32 — What is learned is what ran.** No inference is credited and nothing is in control: the action that
> executed is the one every uncovered activation connects to and the one its reward lands on (R31), whether
> that activation inferred it, inferred something else, or inferred nothing. Before any action pattern exists
> the apex action is the base action, so the rule holds across all of development.
>
> **A pattern earns what its program earned.** It fires when its expansion completes (R30), so what arrived in
> that frame lands on it directly, and what arrived over the frames its program occupied reaches it through
> the span a reward names (R33). That is what makes a multi-frame candidate comparable to a single-frame one.

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
> highest level `D` is open through `reach_t(D)` frames (D9), so the furthest back any open activation could have
> seen an action run is `reach_t(D) − 1` frames, and no share can reach further than the connections that would
> take it. At the base that is `S = 1`, the degenerate case above.
>
> **Nothing is being divided up.** Every distance in the span is paid, and the shares are not a partition of
> the reward — a reward is not in short supply, and the point of spreading it is not to conserve it but to say
> how likely each frame is to have earned it. What is genuinely responsible recurs and accumulates; what is not
> is sampled once and averaged away. Each share is delivered, in the `process actions` call (§17), to every open
> activation that wrote an exposure at the frame that distance names — one uncovered at that frame (D25) — into
> the connection it wrote to then, whether or not coverage has arrived since: at distance `0` the connection this call
> strengthens, further back one strengthened `d` frames ago (R31).
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

# 20. Selection

Five steps, and the last is the only one that compares anything:

```
1. read      every voter — one uncovered open activation at one age — reads its own neuron's
             action connections at every offset beyond its age, out to its reach. Each such
             connection is one inference                                                         R36
2. expand    each inference is placed at its completion, offset minus age frames ahead, and expands
             through its dictionary line to the base actions it places at the frames back from
             there, carrying its strength and its estimate                            R28, R30, R36
3. drop      a base action placed at `f` or before is gone, and one placed beyond `f + 1` is not
             resolved; the placements ahead of a selected pattern stand as standing inferences,
             contending again each frame                                                         R36
4. resolve   per action dimension at `f + 1`: one vote per voter, split by strength; the
             candidate with the largest estimate runs                                            R36
5. execute   it runs next frame, its neuron fires in that column, and its reward arrives
             with it                                                                        R29, R30
```

> **R35 — Selection.** No fit says which action to take; it says only what a situation was followed by.
> Choosing comes from the action connections of the neurons standing on the apex, each carrying the reward that
> arrived averaged over its exposures, and **the machine executes the best. Nothing else decides it.**
>
> **A situation is a set of active event neurons** — any one of them, and any subset of them. Situations
> intersect, and **nothing ever materializes one**: a situation is what a voter fires in, never an object the
> machine holds.
>
> **A voter is one apex activation at one age**, reading its own neuron's action connections at the offset ahead
> (R36). A base neuron's estimate is a marginal over every situation it fires in; a child's is
> over the one situation its parent's pattern names; so the same frame reads differently to a base neuron, to
> the level-1 child covering a chunk of it and to the level-4 child covering the whole, and differently again to
> any of them two ages later. **A minted pattern is how one recurring situation acquires an estimate of its own**
> — which is why a covered neuron is silenced (D10) and its coverer speaks instead, and the only reason a base
> neuron's marginal ever speaks is that nothing more specific was bought over it.
>
> **Only events choose.** An event neuron holds action connections (D25) and any apex event activation may select
> an action. An action neuron holds none: what it did next is a chunk the action hierarchy writes as a pattern,
> not a policy, and a voter that knew only the last action and not the situation would be a habit. Nothing
> selects an event.
>
> **The default runs; it is not wired.** An action dimension no inference reaches runs the declared default
> action. Nothing holds it in advance: when it runs it is the apex action of that frame, every uncovered
> activation connects to it with its reward by the ordinary path (R31), and from then on it is an action like
> any other. A neuron is born holding no action connection at all — a base neuron at cold start and a freshly
> minted pattern alike say nothing about the next action until something has run under them, and what they
> then hold is whatever was executed. R37's walk is the only thing that wires an action ahead of its running,
> one at a time and only where the best known has been judged and found wanting.

> **R36 — The inference decides.** An inference is one action connection read by one voter: it names an action
> neuron, at whatever level that neuron stands (D25), and carries a strength and an estimate. What runs is chosen
> from the inferences and nothing else.
>
> **An inference is placed at its completion, and a connection means what it recorded.** A connection at offset
> `b` was written when that action completed `b` frames after the situation opened (R31), so a voter at age `a`
> places its completion `b − a` frames ahead (R28) and expands it through its dictionary line to the base actions
> it places at the frames back from there (R30), every one carrying the strength and the estimate of the connection
> it came from. What lands at `f + 1` is what the voter proposes: at distance 1 a pattern's last step, at distance
> ten the step ten from its end, and its first step only when its completion is placed its whole length ahead.
> Resolution then runs at the base and only there. **This is what launches a program**, and it is why the read is
> over every offset beyond the age and not the next one alone: a voter can start a program only from an offset
> at least as far out as the program is long, and can carry the tail of a longer one to completion from any offset.
> A situation whose reach is shorter than a program never starts it, and that is correct — it cannot see that far.
>
> **A selected pattern stands until its span runs out.** When the action that wins `f + 1` was placed by a
> pattern's expansion, the rest of that expansion — the members its line places beyond `f + 1`; one at `f` or
> before is dropped, since it has run or cannot — are **standing inferences**, contending for their dimensions exactly as this frame's fresh ones do, at the strength and
> estimate they were selected on. **A plan holds because it keeps winning**: a better estimate displaces it, and
> when its span ends it is simply gone. Nothing is retracted and nothing is held.
>
> **The electorate, explicitly.** At frame `f` the voters entitled to an inference on the actions at `f + 1`
> (R35) are:
> ```
> every standing inference that places a base action at f + 1                          R30
> plus  every open activation the machine holds at f                                    (D16)
>   less  every activation an accepted bid covers                                       (D10)
>   less  every activation at age reach_t             it has no offset left to read
>   read as (neuron, age)                             position carries no connection          (D11)
> ```
> Each reads its neuron's action connections at **every offset beyond its age**, out to its reach. Every
> connection there is one inference, naming an action neuron and carrying a strength and an estimate, and it
> expands as above; what its expansion puts at `f + 1` is what it proposes, and a connection whose expansion puts
> nothing there proposes nothing this frame. A base action is proposed by the one offset the age places one frame
> ahead (R29); a pattern's first step by a farther one.
>
> **Position drops out.** Two activations of one neuron at two positions read one set of connections, so they offer
> the identical inference and the argmax is indifferent to the duplicate. Two *ages* are two voters and do not
> collapse together — they read different offsets, so they can name different actions.
>
> **One winner per action dimension, by estimate.** For each action dimension at `f + 1`, every base action
> some voter's expansion placed there is a candidate. **A candidate's estimate is the mean of the estimates its
> voters placed it with, each weighted by that voter's share** — one vote per voter, split across the actions
> it placed in that dimension by strength, so a voter that hedges between two actions counts as one voter and
> not two — and **the candidate with the largest estimate runs**. Ties go to the larger share of voters, and
> then to the older action. **Level is not read**: a level-4 voter's inference and a base voter's meet on the
> estimate alone, and a specific situation wins over a general one only by being right about what pays, never
> by rank. **Nothing corrects for how many exposures an estimate rests on**, so a sharp estimate on three
> exposures outranks a coarse one on two hundred.
>
> **A covered neuron supplies nothing** (D10). A newly minted child therefore starts with no inference at all; the
> other voters, or the default, decide until something has run under it, and it learns what ran (R35).

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
> created at, which is what makes the walk work: a connection at zero has not been
> judged, so it outranks anything negative and yields to anything positive.
>
> **The walk ends when the alphabet does.** Once a neuron holds a connection to every action in the channel at that
> offset there is nothing left to wire, and selection takes the largest estimate, which is the least bad.
