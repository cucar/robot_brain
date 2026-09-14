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

# 1. The objectives

There are two objectives: **compress what is observed**, and **act on the best reward estimate**. 
The compression decides what the current situation *is*, and that's provided to the second objective as input. 

## 1.1 Compression: classify the situation

**The machine compresses by naming.** A pattern is a set of lower level symbols or patterns that keep occurring
together — past and present neighbors of one another (D26) — and one symbol stands for all of them. The run can be shortened by using
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

  calls (R20)
  the machine calls every neuron that fired at that level. each retires its worst pattern, explains the
  new neighborhood with its table, re-centers, builds one candidate, and returns its bids and its requests.

  creation (R16, R17)
  the machine creates every child requested at this level: id, parent, level, inherited coordinate, empty
  table. its pattern's bid is on the board like any other.

  election (R24)
  the machine greedily covers the level's active neurons with the bids by cost. each bid is credited the
  free neurons it names, until the best left covers no more than it costs. every winning bid activates its
  child one level up; the level's uncovered neurons stand as themselves.

  the level above is built out of what the election accepted, and it happens again; a level that produced
  nothing has no level above it this frame (§18)

then, once the last level has run

  ledger (R16, R18)    the machine wires every child created this frame to its pattern, and deletes every
                       retired pattern now due
  learn (§20, R33)     every uncovered event activation connects to the apex action that ran, and a
                       reward moves the estimate of the connection at its distance
  infer (§23)          the apex reads its connections — only events infer — and every inference
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
             Holds a table of patterns, a history, and connections.           D15, D18, D25

PATTERN      a set of past and present neighbors, one line of one neuron's table, and the
             child neuron it promotes. Lives in its parent.                              D15

ACTIVATION   one occurrence of a neuron, at a frame and a position. Holds the
             neighborhood it observed, and the cover chosen for it.                  D7, D17
```

**A pattern is a pointer to a child, and a child is a neuron.** One add request creates both: the parent
gains a **pattern**, a line in its own table that may enter a cover at once, and the machine mints the **child** it
points to, a neuron one level up (R16), and wires it to the parent.

What fires is an activation; what a level elects is a bid for a pattern's child; 
what the dictionary writes is a pattern.

> **D3 — Channels and dimensions.** No mechanism mints a channel; what grows is the population
> inside one, level by level and without bound. The channel set, and with it the dimension set, is a fixed
> enumerable index over the whole run, which is what lets `(dimension, offset)` name a neighbor at any level.

## 3.2 Space

> **D4 — Reach.** The farthest a neuron sees, either way, in every activation dimension of every channel (D1).
> It is **1** at the base and **doubles every level**:
> ```
> reach(k)   =   2^k          every activation dimension
> ```
> It is a bound, not a distance: a neuron sees every distance up to it, bucketed by powers of two (D6), so a
> level-5 neuron names its neighbors at 1, 2, 4, 8, 16 and 32. `reach_t` is this reach in the time dimension,
> at the neuron's own level — its window.

> **D6 — Offsets.** The offset between two activations is the difference of their coordinates, one component per
> activation dimension they share (D1, D2), each with its magnitude **rounded down to a power of two**:
> ```
> offset(x)   =   sign(x) · 2^floor(log2 |x|)          x ≠ 0
> offset(0)   =   0
> ```
> So 5 and 7 become 4, and −13 becomes −8. The reachable offsets are
> `0, ±1, ±2, ±4, ±8, ±16, …`; `G` groups give `2 + G` offsets per direction across a reach of `2^G`, and
> `reach(k) = 2^k`, so `G = k`: a level reaching `2^k` names offset 0 and every power of two up to `2^k`, in
> both directions.

> **D5 — Adjacency.** Two activations are adjacent when they are **at the same level, of the same kind, within
> reach in every activation dimension they share** (D4), **and the second is not later than the first**. The
> level, since the symbols a level offers are what its neurons draw from; the kind, since an event's neighbors
> are events and an action's are actions. In space both directions count — an activation three positions to the
> right arrives in the same frame as one three positions to the left. In time only the past does, because
> compression only reads the past. **What fires after an activation is not adjacent to it.** The one thing about
> it the machine keeps — the action that ran — is recorded on the neuron as a connection (D25), and it is never in
> a neighborhood and never in a pattern.

> **D26 — Neighbor.** **A neighbor is a neuron at an offset**: the neuron of an adjacent activation (D5), at its
> offset (D6). At temporal offset 0 a neighbor co-occurs and at negative offsets it led here; **both are the same
> kind of thing**, and a spatial component is one more of the same.
>
> **A coarse offset may carry more than one neighbor**, since it spans a range and several activations of one
> dimension can fall inside it. A pattern names per `(neuron, offset)` (D15) and `|p|` counts every neighbor named
> there. Above the base a `(dimension, position)` may itself carry several activations (D8), which this handles the
> same way and needs no rule of its own.
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other
> at a nonzero spatial offset. Offset zero in every component is where the activation itself sits, the center
> of the box, and an activation is not its own neighbor there. Above the base several children promoted at one
> coordinate (D8) are each other's neighbors at offset zero.

> **D7 — Neighborhood.** The set of neighbors (D26) one activation observes is its **neighborhood**, written `O`
> for what it observed:
> ```
> O = { (i, −1) }                                        text stream:  time
> O = { (k, 0, −1, 0), (k, 0, +1, 0), (m, 0, 0, −1) }    image stream:  time, x and y
> ```
> the first could be for a neuron `s` in a stream reading `p a r i s`: at the base the reach is 1 (D4), so `i` is its one
> neighbor and `p a r` are out of reach.
>
> **A neighborhood is the frame around the fired neuron**, since adjacency admits nothing later (D5), and every
> structural decision is made on it. The activation itself is not in it: its own instance is what a pattern
> covering it replaces, and is counted there (D22).

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
> multiple patterns (D28), so several of its children may be active at one coordinate, since they inherit coordinates. 

> **D9 — Activation window.** An activation at frame `f` stays open through `f + reach_t`, the reach in
> time at its neuron's own level (D4). Two things reach it over that span, one frame at a time, after every
> level has run:
> ```
> the apex action  the action that ran this frame, in each action dimension — only while the
>                  activation is uncovered (D10)                                              D25, R31
> the rewards      for that action, and for the actions of earlier frames the reward spans,
>                  each at the distance its frame names                                       R29, R33
> ```
> The machine holds the activation open, not the neuron:
> ```
> open activation  =  (the activation, its coordinate, its age, covered at)      one per (neuron, age, position)
> ```
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since it fired, `0` through `reach_t`; a new activation is the one at age 0. **Age is read, not just
> counted** — rounded as every offset is (D6), it is the offset at which the activation strengthens connections,
> and every offset beyond it is read against it for what comes next (R31, R36).
>
> **Covered at** is the age at which an accepted bid first covered the activation, none until one does.
> Coverage is acquired late and never revoked (R27), so this one number says for every frame of the window
> whether the activation stood on the apex there: it did at every age before it, and at none from it on.
>
> **An open activation carries no commitment beyond that one number.** It holds its neighborhood, the age it
> was covered at, and nothing else, because everything else it was going to decide was decided at age 0 (R1).
> What it does for the rest of its life is connect to what ran and speak from its connections while it is on
> the apex, and that is Part III.

> **D10 — Inhibition.** A neuron an accepted bid covers does not stand in the file, does not infer actions,
> and does not connect to the action that runs. **A neuron is covered only after it has fired** (R27), so
> coverage acquired at a later age stops all three from that frame on and revokes nothing before it: the
> exposures the activation wrote while uncovered stand, and a reward for one of those frames still reaches the
> connection it wrote (R33). An activation covered already at age 0 writes nothing, ever (R27). That is the
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
> file exactly as an observed event does, compressed by the same hierarchy (§21).
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
> having a pattern             =  1 + |p|           a line in the dictionary, |p| the neighbors pattern p names
> ```
> This is a fixed-length code: a symbol costs one regardless of how often it is used.

> **D14 — File length.** Over the run the file is
> ```
> L  =  Σ over the dictionary  ( 1 + |p| )                                    written once
>    +  Σ over the base neurons on the apex  ( 1 )                            the body
>    +  Σ over the children on the apex  ( 1 + the neighbors named and absent )  error correction
> ```
> summing D13's prices over the two parts D12 gives. **There is one `L`**, and every neuron's structure is
> priced against it. Nothing computes it: every quantity the design uses is a **difference** in `L`, which is
> finite however long the run is (D30).

---

# Part I — The past and present: a neuron

# 4. The interface

A neuron is a symbol, and a type (D2). It holds a table of patterns, a history of past and present observations,
and its connections (D25). What it holds is defined in §5 through §7, and what it does with it in the
sections after.

**The machine reaches a neuron through five calls, and nothing else writes into one.** Each is specified where
it is used.
```
create neuron            the machine creates it when the call requesting it returns, one level above
                         its parent, holding nothing                                            R16
process frame            in its level's turn: everything structural for the activations that
                         fired this frame                                                  §16, R20
wire child to pattern    once the last level has run, the machine points the pattern at the child it
                         created for it                                                    R16, R17
delete pattern neighbor  the machine removes a neuron that no longer exists from every pattern
                         and saved activation that names it                                     R18
process actions          after every level has finished, reaching every open activation at
                         whatever age it stands at: it delivers the apex action to those still
                         uncovered and any reward, and collects what the apex speaks            §20
```

Every test the neuron runs is its own arithmetic over its own evidence, and it is never told what the board did
with its bids (R24).

Part I covers `process frame`: what a neuron holds, and what it does in the frame it fires. 
Part III covers the `process actions` call, where a neuron learns what action followed and infers the next.

> **R1 — One decision point: the frame it fires.** A neuron is called once per activation, at age 0, and
> everything structural happens in that call: it covers its neighborhood and saves it, re-centers, builds and
> prices one candidate, retires at most one pattern, and returns a bid for every pattern that applies together
> with its requests (R20).

# 5. The pattern

> **D15 — The pattern.** A set of past and present neighbors that a neuron **names**: one line of its table,
> spanning the same box a neighborhood does (D5), and shaped exactly like one (D7). It is the collapse of the
> neighborhoods it covers (D27), moves as they move (D29), and promotes a child (R16). Its `id` is its creation
> order, a handle that survives re-centering and the tie-break D28 and R24 reach for.
>
> **A pattern is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a pattern is what the neuron claims. A neighborhood is a fact, a pattern a claim.
> Neither is a frame: a frame is one column, either of these the whole window. What action followed is in
> neither, and is held as the neuron's connections (D25).

# 6. The history

> **D18 — The history.** A neuron's record of its own activations: the last `H` of them, oldest first, each
> holding the neighborhood it observed in the frame it fired (D7). It is the evidence every structural decision
> the neuron makes is read over, and it holds nothing else: the neuron's connections are not in the history and
> `H` does not bound them (R31).
>
> **`H` is declared once for the machine and is the same for every neuron.** It counts that neuron's **own
> activations**, not a stretch of run, so a neuron that fires constantly and one that fires rarely weigh their
> patterns against the same amount of evidence. **The ring is exactly `H` deep once filled**, and how much run
> it spans is whatever that neuron's rate makes it. Nothing else anywhere is measured in frames.
>
> **Every activation is recorded, covered or not.** An arriving activation evicts the oldest, and only then.

> **R4 — `H` is the only free parameter.** The alphabet (D1) states the problem and is not a knob. The reach
> (D4), adjacency (D5) and the offsets (D6) are derived, not declared. **No rule introduces a constant, a
> threshold, a window or a cap of its own.**

# 7. The cover

A neuron explains each activation's neighborhood with patterns from its table. This section defines that
explanation: the cover, the owner of each neighbor, and the two readings of that, the coverage and the
residual. Every price in the design (D13) is counted off them.

> **D17 — The cover.** One per activation: the set of patterns chosen to explain its neighborhood. A pattern
> is a set of neighbors (D15), so it explains a subset of the neighbors of the neighborhood it names.

> **D19 — The owner.** One per neighbor of the activation: the pattern of the cover credited with it. A
> neighbor no pattern of the cover names has no owner. Over the activation the owners are a map keyed by its
> neighbors, `(neuron, offset)`: it has the shape of the neighborhood, what fired, and beside each entry which
> pattern is paid for it (the owner).

> **D20 — The coverage.** What one pattern of the cover is credited with: the activation itself, and the
> neighbors it owns (D19).
> ```
> coverage(p, O)   =   the activation, and the neighbors of O owned by p
> ```

> **D21 — The residual.** What no pattern of the cover is credited with: the neighbors with no owner (D19).
> ```
> residual(O)   =   the neighbors of O with no owner
> ```

# 8. The saving

> **D22 — Saving.** What one pattern saves over one activation: what it is credited with (D20), less what it
> costs — its own line, and the neurons it names that did not fire (D13).
> ```
> coverage(p, O)  =  1 + | the neighbors of O owned by p |   the activation itself, and what it owns
> price(p, O)    =  1 + | p \ O |                            its own line, and the neurons it names that did not
> saving(p, O)   =  coverage(p, O)  −  price(p, O)
> ```
> A child on the apex stands in for the activation it covers and for the neighbors its pattern owns there;
> that is what it saves, and that is the coverage. What it costs is its own line, plus a turn-off for every
> neuron the pattern names that did not fire. A neuron the pattern does not name is not in the account at all:
> it costs its own line whether the pattern exists or not.
>
> **This is the only valuation in the design**, and it is read over two different sets — the neuron's own
> activations (D30, D28) and the machine's board (R22, R24) — so the two numbers differ, and are meant to.

# 9. The greedy cover

One procedure appears twice in the design — once inside a neuron, over its own table, and once inside the
machine, over a frame's bids. It is stated here and cited from both.

> **D28 — The greedy cover.** The operation that chooses, from a set of **claimants** each naming some neurons,
> the ones that cover a set of neurons, and credits each covered neuron to exactly one of them. It repeats:
>
> 1. **Measure** every claimant not yet taken against the neurons still uncovered:
>    ```
>    coverage =  1 + | the still-uncovered neurons it names |     the activation itself, and what it names
>    price    =  1 + | the neurons it names that did not fire |            D22
>    ```
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It becomes
>    the owner (D19) of the neurons it was measured on, which leave the uncovered set, and the round repeats from 1.
>
> **What it returns is the cover and the owners.** The cover is the claimants taken (D17); the owner of a
> covered neuron is the claimant of the round that took it (D19), so each is credited once; what no round took
> is the residual (D21). `price` is fixed by what fired and cannot change between rounds; `coverage` only falls.

**The two callers differ in what they cover and in tie-break, in nothing else.**

| caller                     | claimants | to cover                      | ties                                               |
|----------------------------|---|-------------------------------|----------------------------------------------------|
| neuron - recognition (§12) | patterns | the history (D18) | the older `pattern id`                             |
| machine - election (R24)  | bids | the board (§17.2) | the older `neuron id`, then the earlier coordinate |

# 10. The collapse

> **D27 — The collapse.** The operation that collapses a set of neighborhoods into a pattern, returning the
> set of neighbors the pattern names. Every neighbor is decided independently, from the input alone, so the
> result does not depend on the order they are decided in.
>
> For a neighbor `n`, the neighborhoods that count are those in which `n` is not owned by another pattern of
> the cover (D19). Let `s` be their number and `count(n)` how many of them have `n`. Naming `n` covers it in
> `count(n)` of them, states it wrongly in the other `s − count(n)`, and costs one symbol in the dictionary
> line (D13), so it shortens the file by `2 · count(n) − s − 1`.
>
> **`n` is taken exactly when `2 · count(n) > s + 1`.** Nothing is divided.

The collapse is the only operation that decides what a pattern names. It runs in two places:

- **Re-centering** (§11, D29) collapses the neighborhoods of the activations an existing pattern covers, at every
  call.
- **Creating a pattern** (§15, R14) collapses the neighborhoods whose residual holds the candidate's seed, once
  per call. The candidate is in no cover, so every owned neighbor is skipped and it is built on the residual
  alone.

Skipping owned neighbors is what keeps patterns apart: a pattern grows only into the residual, never into what
another pattern owns, so two patterns covering the same activations cannot converge on one set of neighbors.

**Every offset is decided by the same neighborhoods.** An activation says something at every offset, a neuron
or a silence, so the outermost offset is decided over the same `s` as offset 0, less only the neighborhoods
skipped there. Each pattern's collapse has its own `s`; nothing is pooled across patterns, and no threshold,
smoothing or probability estimate enters.

# 11. Re-centering

> **D29 — Re-centering.** The operation that re-decides the neighbors a pattern names (D15) once the activations
> it covers have changed: the oldest evicted when the history is full (D18), or a new one admitted and covered
> with it (D28). It runs the collapse (D27) over the activations the pattern now covers (residual included),
> and updates the owners (D19) to what the pattern now names.

# 12. Recognition

Recognition is the procedure that chooses a cover for a new activation/neighborhood (D17): the greedy cover
(D28) over the residual of the history (D21).

**Cold start is silence.** A pattern covering no activations names nothing, and a neuron with an
empty table covers nothing and bids nothing.

# 13. The margin

> **D30 — Margin.** What a pattern is worth over the history: its saving over each activation it covers (D22),
> summed, less its dictionary line (D13).
> ```
> benefit(p)  =  Σ over the activations p covers:  saving(p, O)                     D22
> cost(p)     =  1 + |p|                                                            D13
> margin(p)   =  benefit(p) − cost(p)
> ```
> It is the difference in `L` (D14) between the file with the pattern and the file without it, and both terms
> are counts over what the neuron already holds.

A pattern is added only when its margin is strictly positive (R15) and retired only when strictly negative (R18).

> **R13 — One comparison.** There is no second one. An activation is covered on `O`, priced on `O`, and the tests
> that add and retire read the same numbers over the history. No quantity in the design waits for anything.
>
> **What arrives later is evidence, not a verdict.** The action that runs after an activation, and the reward
> with it, strengthen the neuron's connections (D25), which the next inference from the apex reads, never a test.

# 14. Deleting a pattern

> **R18 — Retire one, then delete.** Read every margin in the table (D30) — the table as the last call left
> it, less the activation just evicted, and before this call's activation is admitted (R20):
> ```
> benefit  =  Σ over the activations it covers:  |neighbors only this pattern names|  −  ( 1 + |p \ o| )
> retire the pattern with the smallest margin, iff  benefit < 1 + |p|
> ```
> **A pattern that names no neighbor retires at once**, whatever else the call retires: the collapse can drop
> every neighbor a pattern had, and a pattern naming nothing covers nothing and has nothing to earn its line.
>
> **One per call, and no other.** Two patterns naming the same neurons are each worth nothing while the other
> stands, so retiring both on one reading would return their neighbors to the residual with nothing left to
> cover them. Retiring the worst lets the survivor take full credit at the next call, and the next call reads
> every margin again.
>
> **Without the pattern its neighbors fall where D19 puts them** — to another pattern of the same cover that
> names them, at no extra cost to that pattern, or into the residual at one line each. It is the same
> difference R15 reads with the roles swapped: adding a pattern asks what one that is not there
> would take out of the residual, retiring one asks what one that is there is still keeping out of it.
>
> **Retiring is a deletion in the parent.** The pattern leaves the table that instant. It stops competing for a
> place in any cover, so no further activation can bid it, and the neurons it held fall to the residual
> (D21). Having nothing to cover it has no margin and nothing to re-center — **the neighbors it
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


> **R19 — The table needs no rule against duplicates.** Two patterns with equal neighbors present D28 with
> identical input. It takes the older first, so the older covers everything the younger would, the younger
> owns nothing anywhere, and a pattern covering nothing fails R18. The tests remove them.

**A deletion takes the subtree, and takes it at once.** There is no staged cascade and nothing to wait on at
any level.

**Nothing is retired for what followed it.** What followed is measured, not claimed (D25), so a pattern is
never charged for what came after — only for what it names that did not fire beside it.

# 15. Creating a pattern

> **R14 — Where a candidate comes from.** Three fixed steps. Nothing seeds it from outside, nothing grows it a
> neighbor at a time, and nothing repeats until a condition holds.
> ```
> residual(o)  =  the neighbors of o with no owner                                              D21
> seed         =  the neighbor in the most activations' residuals — ties to declaration order (D1),
>                 then to the nearer offset
> input        =  the neighborhoods whose residual holds the seed
> C            =  the collapse (D27) over them
> ```
> `C` is the candidate so built — a pattern not yet in any table (D15).
> The seed is in every one of them, so `C` names it once there are two; over one neighborhood `2 · 1 > 2`
> fails and the collapse names nothing, so **nothing is ever built on a single occurrence**. Every other
> neighbor `C` names is present, and in the residual, in more than half of them: D27 skips owned neighbors as
> it does everywhere, so **a candidate is built on the residual and nothing else.** The seed is the neighbor
> the table is failing on most, and the collapse settles every other neighbor at once — the seed chooses the
> neighborhoods, and the neighborhoods decide every neighbor.
>
> **Only a set of neighbors is built, because that is all a pattern is.** What actions the child will be followed by
> is the child's own connections, formed by the child's own activations once it exists (D25). Nothing about it is
> decided here and nothing about it is priced.
>
> **The same history under the same covers yields the same `C`.** Covers are carried by the activations (D17),
> so the residual, and with it the seed, is a function of the ring and the covers it carries together.
>
> **What `C` is worth.** Against an activation, `C` takes neurons out of the residual and names some that did not
> fire:
> ```
> saving(o)  =  |residual(o) ∩ C|  −  |C \ o|         D22, with C credited the residual only
>               and 0 where that is negative           C joins a cover only where it pays (D28)
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

> **R15 — The add test.** The margin (D30) of a candidate, which is in no cover: it is credited the residual
> only, and only in the activations where it pays (R14).
> ```
> benefit  =  Σ over the history:  saving(o)                                          R14
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
> part of an activation its pattern owns, and several of them may be promoted at one coordinate (D8). What
> they share is a parent and a coordinate, not a slot.
>
> **Release is the same shape reversed**: the parent retires, the machine reclaims. A retired pattern goes back
> on the same request that carries the candidate (R20), so a call touches the alphabet once — in one direction,
> both, or neither.
>
> **Creating and wiring are separate, and only wiring waits.** The machine creates the child when the call
> returns, before the election — an id, its parent, its level, the coordinate it inherits (D2) and an empty
> table — which is all the frame needs to elect it, activate it and call it (R17). Once the last level has run
> it points every pattern at the child created for it and reclaims every retired pattern's child now due
> (R18), in one pass over its own alphabet. Nothing in the frame reads that pointer except a later bid, so
> **the wait costs nothing**.

> **R17 — A child requested in a call is offered in it.** The pattern joins the parent's table in the call
> and is offered by the ordinary rule (R20); the machine creates the child when the call returns, before the
> election (R16), so the bid competes like any other. If it wins, the child is activated one level up and
> called with that level, where it records its first neighborhood and nothing more: its table is empty, and a
> candidate needs two neighborhoods (R14). If it loses, the child exists all the same, as for any pattern whose
> bid loses, and is bought on a later activation.
>
> **It is born holding nothing.** No patterns, no history and no connections (R16). Everything it comes to
> hold is over the situations its parent's pattern actually took (D25), from its first activation on.

# 16. The process frame call

One call per level per frame: everything structural for the activations that fired this frame.

> **R20 — The call, in order.** Once per level per frame, the machine asks one neuron for everything it owes
> that frame. **One set of activations answers**: the ones that fired *this* frame, at age 0 (R1). What an open
> activation learns of what followed names the apex, which no level knows, so it is written after every level
> has run (§20) and never here. The right-hand column says which steps are per activation and which per neuron.
> ```
>                                                                                        over
>  1  evict      a full ring drops its oldest activation; every pattern of that           each new
>                activation's cover loses it and re-centers                     D18, D29   activation
>  2  retire     read every margin in the table; retire the worst if it is strictly          the
>                negative, and any pattern naming nothing. It leaves the table at           neuron
>                once, and the neighbors it owned fall to the residual         R18, D21
>  3  cover      the new activation joins the ring, whole (D7), and recognition runs      each new
>                over the history: what it takes is the **cover** and what it            activation
>                credits is each neighbor's **owner**; every pattern whose covered
>                activations changed re-centers                         D28, D17, D19, D29
>  4  build      seed, neighborhoods, collapse (R14), then price it (R15). A                the
>                candidate that pays joins the table                                        neuron
>  5  offer      a bid for every pattern of the table more than half of whose           each new
>                neighbors are present — `2 · |p ∩ O| > |p|` — whether or not the        activation
>                cover took it, this call's candidate included. A bid is the child's
>                id and the pattern; for the candidate, the request stands in for
>                the id until the machine allocates one                       R21, R17
>  6  return     the bids, and one request carrying the candidate that passed and             the
>                the pattern that retired                                    R16, R21       neuron
> ```

**The call runs before the election**, and the election is over bids from a table that has already retired
against this frame, saved it, re-centered on it, and built from it.

# Part II — The past and present: the machine

# 17. Contraction

> **D23 — Contraction.** The machine covers the level below with neurons from the level above, each taken when
> it covers more neurons than its bid costs to state — `1 + |p \ O|`, never the dictionary line (R22, R24).
>
> **Covering everything is not the goal.** A neuron no accepted bid covers stays in the file as itself, at cost 1
> (D13), and that is the shorter file whenever no neuron could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a pattern names neighbors at offsets, so a promoted neuron replaces a chunk of
> spacetime. Spatial contraction is the case where every offset is zero.

## 17.1 Bids

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
> covered   the bidder, and the neurons it names that fired and no earlier bid covers — the slots it
>           asks to subsume
> price     1 + |p \ O|   its own line in the body, and the neurons it names in those
>                          same frames that did not fire
> ```
> `coverage − price` is the saving over stating the chunk flat, D22's expression over the machine's population.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as its own
> line if nothing covers it (D23) and costs a turn-on if this child is promoted — one symbol either way, so it
> cancels before the test begins, and charging it here would count it twice against the uncovered term of the
> same sum (§17.4).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another bid
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged one whether or not another neuron is right at
> that slot — another neuron's expansion being right there does not make this one's wrong name free. **What a
> pattern gets wrong about a frame is a fact about the two, and ownership does not move it.**
>
> **This is the neuron's arithmetic over the machine's population, and the number is not the neuron's.** The
> neuron took the pattern into its cover, or offered it, on its own residual (R20); the machine tallies on a
> board where earlier frames' credit stands (R23) and this frame's other bids contend (R24). The two numbers
> differ, and are meant to (D22).
>
> **This is a price for one bid, not for the symbol.** The dictionary line `1 + |p|` is weighed by the one test
> (D30) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R22).

## 17.2 The board

> **D24 — The coverage set.** Per level, the machine keeps which accepted bid was credited each covered
> activation:
> ```
> coverage set    per level    which accepted bid holds each subsumed active activation
>                 the owners             one owner per activation; a settled slot never changes owner
> ```
> **A slot is named by a full coordinate**, dimension and position together, so two activations of one neuron
> at two positions are two slots and never contend. Level `k`'s coverage set spans `reach(k) + 1` frames — a
> bid reaches `reach(k)` back and no further — and the box every other activation dimension gives, and ages
> out with it. **The machine holds nothing on the scale of the run.**
>
> Ownership is about **credit**: a neuron is a fact that needs paying for exactly once, so it is settled
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

## 17.3 The election

**The file over one frame is the neurons promoted plus what they got wrong**: `Σ over the accepted (1 + |p \ O|)
+ the neurons no bid covered`, the body half of `L` (D14) over the frames the election can see. **The two
terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first, which is why
the price counts only the neighbors named and absent (R22). The dictionary half is D30's, and neither test
touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — bids are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good solution, not a proved minimum. **Every neuron a bid covers ends up credited to exactly one
bid**, which is what stops a chunk being paid for twice.

> **R24 — The election is D28, run by the machine.** The claimants are this frame's bids and what they cover is
> the **free set**: every active activation of the level below, at its own full coordinate — frame and position —
> that some bid names and no earlier election has credited (R23).
>
> Bids arrive naming relative offsets, so each is resolved against its own coordinate before the first round.
> `price` is `1 + |p \ O|` (R22). **Ties go to the older symbol, then to the earlier coordinate** — creation
> order for a pattern and declaration order (D1) for a base neuron, then frame, then position. Then D28 runs,
> and what it takes are the accepted bids.
>
> **The bound is structural.** An accepted bid takes at least two slots, since `covered > price ≥ 1`, so the
> rounds are at most half the free set and never more than the bids.
>
> **A bid that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> a slot it named is either credited to a bid that did pay or stands as its own line.
>
> **Ownership is a partition of the neurons the accepted bids name**, and that is the whole of the
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

## 17.4 When a slot is settled

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

# 18. The order of a frame

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
> **The frame ends with the machine's own passes.** Once the last level has run, the machine wires every child
> created this frame to its pattern and deletes everything the death ledger has due (R16, R18), calls every open activation
> (§20), then expands what the apex infers (§23) and resolves one winner per action dimension at the base
> (R28, R36).
>
> Every level runs the same rule at the reach one expression gives it (D4). **Compression is spatio-temporal
> at every level, in one pass**: a pattern at any level may name neighbors in its own frame, in earlier ones,
> beside it in space, or in a mix.

> **R27 — The apex is a frontier, not a level.** It is every active neuron **no accepted bid covers** — the
> uncovered set, at every level at once — so a base neuron nothing found worth chunking stands in it beside a
> level-4 pattern. This is the frontier the file's body writes, **the one that learns what ran** (D25), and
> **the one that votes** (§23): the uncovered set does all three, and coverage silences a neuron in every one of
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

# 19. Connections

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
> an activation connects is decided frame by frame, by the frontier** (R27): it connects at a frame iff no
> accepted bid covers it at that frame. Coverage is acquired late and never revoked (R27), so an activation
> stands on the apex from age 0 until the frame it is covered, and never again; one covered at age 0 never
> stands on it (R17).
>
> **They are measured, never chosen.** A connection is not in the bid (R21), not in any dictionary line (D13),
> and it enters no test. Connections are read in one place — **when the activation stands on the apex** (R27) —
> and what it reads there are the inferences that choose the next action (§23). A child neuron fires only when
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
> neuron is a symbol that fires when its program runs, and it is chunked like any other (§21).

# 20. The process actions call

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
**inferences** (§23), each naming an action neuron at whatever level that neuron stands (D25), and only what
reaches the frame ahead is resolved. A base neuron on the apex speaks from its own connections like any other;
what it infers is the marginal over every situation it fires in, and it speaks only because nothing more
specific covers it.

**An activation closes at age `reach_t`** (D9), once that frame's exposure is taken. Closing does nothing but
stop the writing — there is no second call and nothing is saved twice.

# 21. Actions

An action dimension carries what the machine executes, and it is compressed by the same hierarchy its events
are (D1, D8).

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> the neighbors its pattern names one level down, at that neuron's offset plus theirs — offsets compose because
> each is a difference of activation coordinates (D2) — repeated to base symbols, one level fewer than the
> neuron's height.
> ```
> A placed at f:      A's line is {(u, 0), (v, −1)}                 → u at f,   v at f−1
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
> pattern is therefore on its level's frame like a bought one: it runs `process frame`, is chunked upward (§18),
> and every uncovered event activation open at that frame connects to it (D25). While its program runs, the apex
> action at each frame is the base action, or a lower pattern as it completes; the whole is the apex action only
> in the frame it completes, and what the program earned reaches it through the span a reward names (R33).

# 22. Reward

A reward is an input, not a symbol: alongside what it reports observed, a frame may carry rewards for actions
already executed. They reach the machine through one object and one only — the **action connection**, which no
structural test can see (R34).

> **R31 — An action connection carries an estimate.** What executes at `f + 1` is not known at `f`:
> it is settled only once every level has run and the inferences resolve (R36). So the action that ran fires in
> its dimension at `f + 1` (D8), with its reward beside it (R29), and **every uncovered event activation open at
> that frame connects to the apex action at its own age** (D25, §20) — the highest action pattern that fired in
> that dimension that frame, the base action when none did. That connection binds what the neuron stands for to
> what the machine did — formed against what actually ran, so **a neuron that inferred a different action, or
> none, learns from the one that ran.**
>
> **One activation connects to the action of every frame it is open through, and one action is connected to by
> every uncovered activation open when it ran.** The offset is the age — the distance from the frame the
> activation opened to the frame the action ran — rounded as every offset is (D6),
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
> neuron keeps its own connections over its own activations (D25).
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
> is sampled once and averaged away. Each share is delivered, in the `process actions` call (§20), to every open
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

# 23. Selection

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
> plus  every open activation the machine holds at f                                     (D9)
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
