# Universal Compression with Actions and Rewards (UCAR)

UCAR is the design for a machine that compresses what it observes by building a hierarchical dictionary of patterns, and
learns what to do by observing rewards. It is defined by two alphabets, like a Turing machine: the **event
alphabet** it can observe and the **action alphabet** it can execute. Above each, it forms symbols of its
own. Every symbol, base or learned, event or action, is a **neuron**.

It has two inputs and one output. Inputs: the events observed, the actions that ran, and the rewards earned.
Output: the actions it infers, written in the base alphabet; the environment executes what it will of them and
reports what ran.

This document is the specification, and nothing else. **D** is a definition and **R** a rule; together they
are the machine. Theorems, worked examples and all commentary on why the design is shaped this way live in
[algorithm-remarks.md](algorithm-remarks.md), keyed by the same D and R numbers; risks, diagnostics and open
questions live in [algorithm-evaluation.md](algorithm-evaluation.md).

---

# Part I — The structure

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

| Loss      | Description                                                                                                                                       | Reference |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| placement | An offset is kept to one significant digit in base 2, so a far neighbor is placed only to within the power of two it rounds to.                   | D6        |
| evidence  | A neuron decides its structure over its last `H` activations and the history slides, so the structure that would restate a frame long past is neither held nor recoverable. | D18       |

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
rewards for actions already run (D35). The machine works **up one stack, a level at a time**: at each level it
calls every neuron that fired, elects over their bids, and gives the accepted bids the pattern children and value
neurons they need, reused or new, and the level above is built out of what the election
accepted (§7). Once the last level has run it wires and deletes children, delivers the action that ran and its
reward to every open activation, and resolves one action per dimension for the frame ahead from what the apex
infers (§8). The reward for that action arrives with the next frame (R29).

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
>
> **A channel declares its neighborhood policy**: whether its base actions are neighbors of its events (D5). A
> channel whose actions do not move its events, a price and the decision to hold it, declares that they are not,
> and its patterns name events alone; every other channel lets a body name both.

> **D2 — Type and instance.** A **neuron is a type** and an **activation is an instance of it**.
>
> | Coordinate | Components                                        | Nature                                                              |
> |------------|---------------------------------------------------|---------------------------------------------------------------------|
> | base neuron | `(dim_id, bucket_id)`                            | structural and defining: the alphabet                               |
> | higher neuron | a level and an id                              | a pattern child or a value neuron; no dimension and no kind |
> | activation | frame, and one position per activation dimension  | fleeting; two activations of one neuron differ in nothing else      |
>
> **Neuron dimensions belong to the base alphabet and to nothing above it.** A base neuron is an event or an action
> of one dimension of one channel. A higher neuron, whether a pattern child or a value neuron, has a
> level and a name and nothing else structural: it is not an event or an action, it belongs to no dimension and no
> channel, and its body or its values may hold either kind. A child sits one level above the parent whose bid it
> was created for, other patterns may come to share it (R43), and its parents may be of either kind. Only
> activation dimensions exist above the base, and an activation of a higher neuron has exactly the coordinate it
> inherits.
>
> **A child's activation inherits the parent activation's coordinate** — the frame, and the position in every
> activation dimension — and never an average over what it covers.

**Three objects:**

| Object     | Description                                                                                                                          | References    |
|------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------|
| neuron     | A symbol, and a type. Sits at one level; a base neuron in one dimension of one channel, a higher one in none. Holds a patterns table, a class table, a parameters table, a history, and connections. | D32, D41, D44, D18, D25 |
| pattern    | A set of past and present neighbors, one line of one neuron's table, and the child neuron it promotes. Lives in its parent.          | D15           |
| activation | One occurrence of a neuron, at a frame and a position. Holds the neighborhood it observed, and the cover chosen for it.              | D7, D17       |

**A pattern is a pointer to a child, and a child is a neuron.** The two are made apart. A neuron adds a
**pattern**, a line in its own table that may enter a cover at once, and bids it with no child. The first
time a bid of that pattern is accepted the machine gives it a **child**, a neuron one level up: one it mints, or
one another pattern already has for the same chunk (R16, R43). **Lines stay with the neuron, and neurons belong
to the machine.**

What fires is an activation; what a level elects is a bid for a pattern child; 
what the dictionary writes is a pattern.

> **D3 — Channels and dimensions.** No mechanism mints a channel; what grows is the population
> inside one, level by level and without bound. The channel set, and with it the dimension set, is a fixed
> enumerable set over the whole run, which is what lets `(dimension, offset)` name a neighbor at any level.

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

> **D5 — Adjacency.** Two activations are adjacent when they are **at the same level, within reach in every
> activation dimension they share** (D4), **and the second is not later than the first**. The level, since the
> symbols a level offers are what its neurons draw from. Events and actions are adjacent to one another like
> anything else, unless their channel declares otherwise (D1): what was done a frame ago is context as much as
> what was seen. In space both directions count — an activation three positions to the right arrives in the same
> frame as one three positions to the left. In time only the past does, because compression only reads the
> past. **What fires after an activation is not adjacent to it.** What the machine keeps about it is recorded
> on the neuron as a connection (D25), and it is never in a neighborhood and never in a pattern.

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
carries the action the environment reports as run in that same frame (if there is one), or, for a dimension
the environment does not have, the action the machine inferred there, weak (D40).

> **D8 — Activation.** A neuron is activated (fires) only when something happens: an event neuron when its event
> is observed, or an action neuron when its action runs.
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
>
> | Input           | Description                                                                                                 | References |
> |-----------------|-------------------------------------------------------------------------------------------------------------|------------|
> | the apex       | What stands on the apex of that frame (R27), only while the activation is uncovered (D10).   | D25, R31   |
> | the rewards     | For the actions of that frame, and of earlier frames the reward spans, each at the distance its frame names. | R29, R33   |
>
> The machine holds the activation open, not the neuron:
> ```
> open activation  =  (the activation, its coordinate, its age, covered at)      one per (neuron, age, position)
> ```
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since it fired, `0` through `reach_t`; a new activation is the one at age 0.
>
> **Covered at** is the age at which an accepted bid first covered the activation, none until one does.
> Coverage is acquired late and never revoked (R27), so this one number says for every frame of the window
> whether the activation stood on the apex there: it did at every age before it, and at none from it on.
>
> **An open activation carries no commitment beyond that one number.** It holds its neighborhood, the age it
> was covered at, and nothing else, because everything else it was going to decide was decided at age 0 (R1).
> What it does for the rest of its life is connect to what ran and speak from its connections while it is on
> the apex, and that is Part IV.

> **D10 — Inhibition.** A neuron an accepted bid covers does not stand in the file, does not infer, and does
> not connect to what follows. **A neuron is covered only after it has fired** (R27), so
> coverage acquired at a later age stops all three from that frame on and revokes nothing before it: the
> exposures the activation wrote while uncovered stand, and a reward for one of those frames still reaches the
> connection it wrote (R33). An activation covered already at age 0 writes nothing, ever (R27). That is the
> whole of inhibition in the design.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what the decoder
assumes for anything the file does not state.

> **D11 — Identity.** A neuron's identity is its type alone, a base symbol or a higher neuron's id (D2), so
> nothing about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: both carry the same relative neighborhood, so the same patterns cover them and one pattern
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

> **D40 — Strength.** An activation is **strong** when the environment reported it, and **weak** when an
> inference placed it (§8.4). That holds for events and actions alike: an event the machine expects and an action
> it infers are both weak, and the environment's report replaces both. The neuron is the same either way: a
> weak activation is an activation of the neuron the environment would have reported.
>
> **A weak activation is an input of its frame like a strong one.** It is in the neighborhood of everything that
> fires within reach of it, at whatever offset, so it enters histories, and patterns name it and price it like any
> cell; its neuron is called with it, it may be covered, and it may stand on the apex and vote once (R36), with
> the connections its neuron's strong activations wrote. **It differs in one thing: it lasts its one frame.** It
> is not open afterward: it writes no connection, nothing connects to it, and no reward reaches it (R31). **The
> report replaces it**: where a strong activation of the same neuron stands at the same coordinate, at any level,
> the weak one is dropped. Where nothing reports, the inference peeks through: an inferred action nothing reports
> as run, and an expected event nothing reports as seen, are what the machine had in that frame, and a dimension
> the environment does not have holds nothing else.
>
> **A child is as strong as what it covers.** An accepted bid's child is strong when any activation it covers
> is strong, and weak when every one of them is: a situation made entirely of expected inputs is expected. A
> value neuron is as strong as the value it stands for (D45).

## 3.4 The file

> **D12 — The file.** Two parts, both spanning the whole run. **The dictionary**: every neuron's tables, its
> constants, classes and parameters, and one line per pattern, the constants and variables it refers to (D38).
> **The body**: every neuron no accepted bid covers (D10) — each pattern child among them written as its call, its
> owner and which of its patterns, the member each class held and the value each parameter held, and the cells it
> names that did not hold (error correction); and each bare neuron standing as itself. The level that made a call
> writes it that way; the level above reads the same pattern child and value neurons as neurons of its own
> (D14).
>
> **Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
> file exactly as an observed event does, compressed by the same patterns (§3.5).
>
> **It holds nothing about the future.** A neuron's connections — what actions followed its activations, and what
> they earned — are in no
> dictionary line and not in the body (D25), and neither is an inference that did not run (R32).
>
> **It holds nothing about the search either.** Populations, estimates and margins are the machine's and
> never the file's, because expanding an apex neuron needs the patterns and nothing else.
>
> **It holds the member.** Where a pattern names a class (D38), the body writes which member stood there, a
> choice among the class's members (D13), and expansion puts that member back (R28). What a class gives up is
> nothing the file holds: the pattern child it is part of has one set of connections for every member, and the
> value neuron beside it holds the connections for that member (D45).

> **D13 — Prices.** Every cost in the design is part of a file, counted in **bits**. Each thing the file writes
> is a **field**, a choice among the options of its own alphabet, and a field costs `log₂` of its alphabet's
> size however often it is used: a fixed-length code, with one alphabet per kind of field.
> ```
> a neuron                  log₂ n                                   among the n neurons of the level
> an offset                 Σ over its dimensions  log₂ b            b the buckets of that dimension at the level (D6):
>                                                                    k + 2 in time and 2k + 3 in any other, at level k
> a constant, in a line     log₂ |constants|                         which entry of the owner's constants table
> a class, in a line        log₂ |classes|  +  an offset             which entry of the owner's class table, and where
> a parameter, in a line    log₂ |parameters|                        which entry of the owner's parameters table
> which pattern             log₂ |patterns|                          which of the owner's patterns
> which member              log₂ |K|                                 among the members of the class K
> which value               log₂ |V|                                 among the values its parameter has passed
> which cell failed         log₂ cells                               among the cells the pattern names
> ```
> What each thing costs is the sum of the fields it writes:
> ```
> a neuron standing as itself    =  a neuron
> a constant, in its table       =  a neuron + an offset                              written once
> a class, in its table          =  a neuron per member                               written once
> a parameter, in its table      =  an offset per offset it spans + a neuron per value it has passed
>                                                                                     written once
> a pattern's line               =  a reference per constant or variable it names     written once
> a pattern, on one activation   =  a neuron + which pattern                          its owner, and which of its patterns
>                                +  which member, per class it names
>                                +  which value, per parameter it names
>                                +  which cell failed, per cell that did not hold
> ```
> **Each constant and variable pays for what it says.** A constant says which neuron and where, once, in its table.
> A class says which set, once, in its table, says where in every line that refers to it, and leaves which member
> to each occurrence. A parameter says where, once, in its table, and leaves which value to each occurrence. A
> class of every neuron of the level leaves the whole neuron open and saves nothing; a class of one member is a
> constant.
>
> **The layout is not priced.** Where a neuron stands is its place in the body, and how many things stand at one
> place or follow one instance is the layout's; no field says either. **Nothing about frequency enters**: a
> field costs the same however often its value is written.
>
> **`n` is the machine's count of the neurons at the level**, raised when a neuron of that level is created and
> lowered when one is deleted. The machine hands it to every neuron of the level with its call (R20), so every
> neuron of the level prices against the same `n`, and so does the election over their bids (R22). Every other
> alphabet is the owner's own: its tables, a class's members, a parameter's values.

> **D14 — File length.** Each level reads the file from its own side, in its own alphabet: its activations, each
> standing as a neuron of the level or covered by one of the level's calls, and the tables of the level's neurons.
> Over the run, at level `k`:
> ```
> L_k  =  Σ over the tables of the level's neurons:  constants, variables and pattern lines      the dictionary
>      +  Σ over the level's activations no call covers:  a neuron                               the body
>      +  Σ over the level's accepted calls:  the pattern on one activation (D13)
> ```
> summing D13's prices over the two parts D12 gives. **Each level is its own coder.** A call's pattern child and
> value neurons are activations of the level above, and that level reads them as neurons of its own, regular
> choices among its `n`, knowing nothing of how the values were distributed below. So the two levels price one
> call differently, and are meant to: each compresses what it sees with what it knows. Nothing computes any `L_k`:
> every quantity the design uses is a **difference** in the reading of the level that makes it, which is finite
> however long the run is (D30).

---

## 3.5 Actions

**An action is a function, and its activation is a call.** An action dimension carries what ran in it
(D1), and it is compressed in the same patterns its events are (D8): above the base there is one kind of
pattern, and its body may name what was seen and what was done together (D5).

**A base action takes no arguments and has no position of its own.** Nothing is declared about it but its place
in the alphabet (D1). Where it acts is state the environment holds, a **focus**: base actions move it as they do
anything else, and the environment reports it to the machine as events like anything else.

**An action the environment executes on its own is a lesson.** It arrives strong, with its reward, whether or not
the machine inferred it (D40): patterns form over it beside the events it ran with, and what stood on the apex
connects to it with what it earned (R31), so in the same situation the machine comes to infer it (R36). That is
teaching by demonstration, and it needs no rule of its own. Exploration is the other way an action is first
tried (R37).

> **D37 — The call.** An activation of an action (D8). It fires in the frame it runs, at the coordinate of the
> activation that inferred it. Where the action's body names a parameter, expanding the call binds it from what
> stands at those of its offsets that have happened and places its value at the rest (D44, R28), so a call
> carries nothing of its own. **An
> action dimension is a set of functions that contend with each other**: one call per dimension runs in a
> frame. The machine outputs the calls it infers, and a call has run when the environment reports it (D40). The
> environment executes what it will of them, and calls of its own, and reports what ran. In a dimension the
> environment does not have, nothing is output: the machine's inference feeds back as a weak activation, as an
> expected event does, and nothing outside the machine sees it.

> **D34 — The apex action.** The base action reported as run this frame in an action dimension, as the highest
> uncovered activation whose expansion placed it (R27), or the report itself where nothing placed it. An action
> dimension no inference reaches is output its declared **default action**, a proposal like any other, and what
> the report names there is the apex action of the frame.

**A pattern is matched or inferred, and both kinds of member take part in both.** Matched, a pattern says that
its events stood and its actions ran. Inferred, it is expanded (R28): its actions are placed as the
machine's output and its events as what the actions are expected to bring, both weakly, and the report makes
strong what ran and what was seen (D40). What follows any activation is held on its neuron as
connections (D25), one kind, and an inference is a connection read.

Four cases are worked by hand on these definitions: [adding two binary numbers](algorithm-addition.md), where
the carry is an action nothing outside the machine sees, recognized a frame after it ran;
[copying what was seen](algorithm-copy.md), where a parameter binds a value now and places it two frames on; [hitting whatever comes at you](algorithm-hit.md), where offsets from
the body are the reference frame; and [an alternating pair](algorithm-xy.md), where a neuron that keeps seeing a
different pair writes it as one pattern of two parameters, and holds the alternation once.

## 3.6 Rewards

> **D35 — The reward.** An input, not a symbol: a frame carries an array of rewards for actions already
> executed, each
> ```
> { reward, channels, frames }
> ```
> `reward` is signed: strictly greater than zero is good, strictly less is bad, and zero is neither.
> `channels` is the set of action channels it pays and `frames` the span of frames it pays over, counted back
> from the frame it arrives in. Both are optional, and an omitted one means all of them.

# 4. The pattern and the cover

What this section defines is shared: a neuron reads it over its own history and the machine over a frame.

## 4.1 The pattern

> **D15 — The pattern.** A set of past and present neighbors that a neuron **names**, spanning the same box a
> neighborhood does (D5), and shaped exactly like one (D7). It is the collapse of the
> neighborhoods it covers (D27), moves as they move (D29), and promotes a child (R16). Its `id` is its creation
> order, a handle that survives re-centering and the tie-break D28 and R24 reach for.
>
> **A pattern is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a pattern is what the neuron claims. A neighborhood is a fact, a pattern a claim.
> Neither is a frame: a frame is one column, either of these the whole window. What action followed is in
> neither, and is held as the neuron's connections (D25).

**A pattern is a function, and its variables are its arguments.** A pattern with no variable is the
parameterless one, a set of neighbors replaced by one symbol. Events and actions form them the same way. A
parameter says that several offsets hold one neuron, without naming it; a class says that some member of a set
stands at an offset. **An accepted bid is a call of it**: its child fires one level up, and beside it a value
neuron for each of its variables, carrying what the variable held (§7.4).

> **D38 — The body.** A pattern's line is its body: a set of references to **constants** and **variables**,
> each held in its own table and each covering cells of a neighborhood, a cell being a neuron at an offset (D26).
> A class and a parameter are the two kinds of variable: what stands in them changes from one occurrence to the
> next.
>
> | Named       | Table       | Covers                                   | Written once, in its table | In the line | Left open on each row |
> |-------------|-------------|------------------------------------------|----------------------------|-------------|-----------------------|
> | a constant  | constants   | one cell: neuron `n` at `δ`              | a neuron and an offset     | which constant | nothing            |
> | a class     | classes     | one cell: any member of `K` at `δ` (D41) | a neuron per member        | which class, and its offset | which member, `log₂|K|` |
> | a parameter | parameters  | several cells: one neuron at every offset of `P` (D44) | an offset per offset, a neuron per value | which parameter | which value, `log₂|V|` |
>
> A constant says which neuron and where. A class says where, and which set the neuron is from. A parameter says
> where only, at several offsets at once, and that the neuron is the same at all of them; it names none, and
> holds no class. A pattern owns none of what it names: every pattern that refers to a constant or a variable
> refers to the same one, and it is paid for once, in its table. On every row it covers, the pattern writes its
> owner and which of its patterns, what its variables leave open, and which of its cells did not hold (D13, D22);
> the row's owner and every cell it names are then covered. Both kinds of variable write what stood in them,
> and both pass it up, as a value neuron (D45).
> A body holds nothing else: no branch, no loop and no argument list. What branches is which situation fires;
> what loops is the frame; what a variable held is on the level above, in its value neuron.

## 4.2 The cover

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

## 4.3 The saving

> **D22 — Saving.** What one pattern saves over one activation: what it is credited with (D20), less what it
> costs, the fields its occurrence writes (D13).
> ```
> coverage(p, O)  =  log₂ n · (1 + the cells of O owned by p)       the activation itself, and what it owns
> price(p, O)     =  log₂ n  +  which pattern                        its owner, and which of its patterns
>                 +  Σ over p's classes     log₂|K|                  which member stood
>                 +  Σ over p's parameters  log₂|V|                  which value
>                 +  failed · log₂ cells                             which of its cells did not hold
> saving(p, O)    =  coverage(p, O)  −  price(p, O)
> ```
> A cell holds or fails as D38 says: a constant fails when its neuron is absent, a class when no member stands
> at its offset, and a parameter at each of its offsets where what stands differs from the neuron most of its
> offsets hold. A value is paid once however many offsets its parameter spans, so its first offset saves
> `log₂ n − log₂|V|` and every further offset saves `log₂ n`.
> A child on the apex stands in for the activation it covers and for the cells its pattern owns there; that is
> what it saves, and that is the coverage. What it costs is its owner and which of its patterns, what its variables leave
> open, and which of its cells failed; the owner's bits and the activation's cancel. A neuron the pattern does not
> name is not in the account at all: it costs `log₂ n` whether the pattern exists or not.
>
> **This is the only valuation in the design**, and it is read over two different sets — the neuron's own
> activations (D30, D28) and the machine's board (R22, R24) — so the two numbers differ, and are meant to.

## 4.4 The greedy cover

One procedure appears twice in the design — once inside a neuron, over its own table, and once inside the
machine, over a frame's bids. It is stated here and cited from both.

> **D28 — The greedy cover.** The operation that covers a set of activations with **claimants**, each a set of
> named neurons read against one of those activations, and credits each covered neuron to exactly one claimant.
> It repeats:
>
> 1. **Measure** every claimant not yet taken against the still-uncovered neurons of its activation:
>    ```
>    coverage =  log₂ n · (1 + | the still-uncovered neurons of its activation it names |)   the activation, and what it names
>    price    =  the fields its occurrence writes there: its owner and which pattern, what its variables  D22
>                leave open, and which of its cells did not fire
>    ```
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It joins
>    its activation's cover (D17) and becomes the owner (D19) of the neurons it was measured on, which leave the
>    uncovered set; the round repeats from 1.
>
> **What it returns is the covers and the owners.** A cover is the claimants taken for that activation (D17);
> the owner of a covered neuron is the claimant of the round that took it (D19), so each is credited once; what
> no round took is the residual (D21). `price` is fixed by what fired and cannot change between rounds;
> `coverage` only falls.

**The two callers differ in what they cover and in tie-break, in nothing else.**

| caller                     | claimants | to cover                      | ties                                               |
|----------------------------|---|-------------------------------|----------------------------------------------------|
| neuron - recognition (§6.2) | each pattern, against each activation | the residual of the history (D21) | the older `pattern id`                             |
| machine - election (R24)  | bids | the board (§7.2) | the older `neuron id`, then the earlier coordinate |

## 4.5 The bid

> **D31 — The bid.** What an activation offers the machine for one pattern of its cover (D17), as the neuron
> holds it: the pattern's id, its neighbors, the child it promotes, and for each variable the value it
> held.
>
> | Component  | Description                                                                      |
> |------------|----------------------------------------------------------------------------------|
> | pattern id | its creation order (D15)                                                         |
> | neighbors  | the dictionary line (D12)                                                        |
> | child      | the id of the child this pattern promotes; none until a bid of the pattern has been accepted (R16) |
> | values     | per variable, the value it held and that value's value neuron, none until the machine has created it (D45) |

# 5. The neuron

What this section defines belongs to one neuron and is read nowhere else.

## 5.1 The interface

A neuron is a symbol, and a type (D2). It holds a patterns table, a class table and a parameters table, a
history of past and present observations, and its connections (D25). What it holds is defined in §4, §5.2 and §5.3, and what it does with it in the
sections after.

**The machine reaches a neuron through five calls, and nothing else writes into one.** Each is specified where
it is used.

| Call                    | Description                                                                                                                                                       | References |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| create neuron           | The machine creates it when an accepted bid needs one and none can be reused, one level above the bidder, holding nothing: a pattern child, or a value neuron for a variable's value. | R16, R43, R44 |
| process functions       | In its level's turn: everything structural for the activations that fired this frame.                                                                             | §6, R20    |
| wire child to pattern   | Once the last level has run, the machine points the pattern at the child it created or chose for it, and each variable's value at its value neuron.                                             | R16, R17, R44 |
| delete pattern neighbor | The machine removes a neuron that no longer exists from every pattern and saved activation that names it.                                                          | R38        |
| process actions         | After every level has finished, reaching every open activation at whatever age it stands at: it delivers what stood on the apex this frame and any reward, and collects what the apex infers.                          | §8.1       |

Every test the neuron runs is its own arithmetic over its own evidence, and it is never told what the board did
with its bids (R24).

Part II covers `process functions`: what a neuron does in the frame it fires.
Part IV covers the `process actions` call, where a neuron learns what action followed and infers the next.

> **R1 — One decision point: the frame it fires.** A neuron is called once per frame it fires in, for every
> activation of that frame together, at age 0, and everything structural happens in that call: it refreshes
> its history, recognizes the frame's neighborhoods, retires what no longer pays, builds every candidate that
> pays, merges what became identical, and returns a bid for every pattern of each cover together with
> everything it retired (R20).

## 5.2 The patterns table

> **D32 — The tables.** A neuron holds four: the **patterns table**, the set of patterns (D15) it holds, one line
> per pattern, **the table** for short; and the three tables of the constants and variables its patterns refer to (D38), the
> **constants table**, the **class table** (D41) and the **parameters table** (D44). Recognition reads them (D28);
> adding, retiring and merging write them (R15, R18, D43).

## 5.3 The history

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
> (D4), adjacency (D5) and the offsets (D6) are derived, not declared; the `n` every price reads is the
> machine's count of the neurons at the level (D13). **No rule introduces a constant, a
> threshold, a window or a cap of its own.**

## 5.4 The relations and the collapse

> **D47 — The relation.** Two cells of the residual (D21) that bear on one another, counted over the history
> (D18). It is one of three kinds, and each is the evidence for one kind of candidate:
> ```
> two neurons standing together in one row    a at δ₁ and b at δ₂          a pattern of constants
> one neuron at two offsets of one row        n at δ₁ and n at δ₂          a parameter over δ₁ and δ₂ (D44)
> two neurons at one offset, in two rows      a at δ in one, b at δ in another   a class of a and b (D41)
> ```
> The first two are within a row and are counted by the rows that hold them. The third is across rows: `a` and
> `b` never stand together at one offset, they stand there instead of each other, and it is counted by the
> residual cells holding either of them at an offset where both have stood, pooled over every offset. Once variables
> exist, the first kind also holds between them: two things standing together in a row, each a neuron at an
> offset or a variable holding there. The counts move as rows enter and leave the history and as
> cells are claimed and covered; how they are kept is the implementation's.

> **D27 — The collapse.** The operation that grows a seed relation (D47) into a candidate. The seed fixes the
> candidate's **uses**, the neighborhoods or cells where it holds, and every extension of it is decided by its
> price over those uses (D13), from the uses alone. There is one collapse per kind of candidate.
>
> **A pattern.** The uses are the neighborhoods whose residual holds the seed, and the collapse returns what
> the pattern names, constants and variables alike. Every offset is decided independently, from the
> input alone, so the result does not depend on the order they are decided in. For a constant or a variable `x` at offset `δ`,
> the neighborhoods that count are those in which `δ` is not owned by another pattern of the cover (D19). Let `s`
> be their number and `count(x)` how many of them `x` holds in (D38). Naming `x` saves `worth(x)` in each of
> those, costs a failed cell in each of the other `s − count(x)`, and costs its reference in the line, with its
> entry in its table if it is not there yet (D13):
> ```
> worth(constant)      =  log₂ n
> worth(class K)       =  log₂ n − log₂|K|                     what the class narrows
> worth(parameter)     =  offsets · log₂ n − log₂|V|           over all its offsets at once: its value is paid once
> ```
> **`x` is taken exactly when `count(x) · worth(x) > (s − count(x)) · log₂ cells + reference(x)`.** A pattern of
> constants memorizes its neighborhoods; a pattern of classes and parameters equalizes them; the rule is the
> same for all three, and a body may hold all three. A parameter is named whole, so it is weighed over all its
> offsets at once, and it fails at each of them where its value does not stand. **Where more than one
> qualifies**, a constant, a class or a parameter, the one that saves more is taken,
> `count(x) · worth(x) − (s − count(x)) · log₂ cells`, a parameter against the best of the others at each of
> its offsets, summed. A constant is worth more where it holds, but holds in fewer neighborhoods than a class of
> it and its alternatives; so a pattern names the specific neuron only when that neuron dominates enough to save
> more than the class, and turns general again when it no longer does. Re-centering decides this afresh every
> time it runs, on the same rows.
>
> **What an offset offers.** At each offset the candidates are the neuron standing there, every class of the
> class table it is a member of (D41), and every parameter of the parameters table that spans the offset and
> holds there (D44). A pattern's collapse chooses among the variables that exist and makes none but
> constants: variables are built by their own collapse.
>
> **A branch.** Where two offsets hold a fixed pairing of different neurons in a majority, `paired(δ₁, δ₂)`, `m`
> whenever `d` and `n` whenever `e`, the variation at one is explained by the other. The neighborhoods are split
> on the first into one candidate per pairing, each with constants where the parameter would have been, and the
> candidates are priced against the one; the shorter file is kept. That is dispatch, and it is where a case keeps
> a lesson of its own.
>
> **A parameter.** The uses are the rows where the seed's two offsets hold one neuron, which is that row's value.
> Every other offset `δ` is counted by the uses in which it holds the same value, and each is decided on its own,
> since the value is paid once however many offsets hold it:
> ```
> δ is taken when   count(δ) · log₂ n   >   (uses − count(δ)) · log₂ cells   +   an offset
> ```
> An offset leaves by the same test read the other way, and the parameter is dropped when fewer than two remain.
>
> **A class.** The uses are the residual cells holding a member at an offset where two members have stood. The
> neurons that stood at those offsets in the other rows are its candidates, and they are taken in order of their
> count `c` there, each while what it saves beats its entry and the wider choice it gives every use the class
> already has:
> ```
> c · log₂(n / (|K| + 1))   >   log₂ n  +  uses · log₂((|K| + 1) / |K|)
> ```
> Down the order the counts only fall and the widening only grows, so the first candidate that fails ends it.
> Where a pattern names the class, a non-member standing at its offset is a failed cell there, and joining saves
> that too, `log₂ cells` more on each of the candidate's uses. A member leaves by the same test read the other
> way. A class with fewer than two members is deleted (R41).

The collapse is the only operation that decides what a pattern names, what a class holds and where a parameter
stands. It runs in three places:

- **Re-centering** (§5.5, D29) collapses the neighborhoods of the activations an existing pattern covers, at every
  call.
- **Maintaining a variable.** A class's members and a parameter's offsets are re-decided by their own collapse over
  their uses whenever those change (R20).
- **Creating** (§6.4, D33) collapses a seed relation into a variable or a pattern. A candidate is in no cover, so
  every owned neighbor is skipped and it is built on the residual alone.

Skipping owned neighbors is what keeps patterns apart: a pattern grows only into the residual, never into what
another pattern owns, so two patterns covering the same activations cannot converge on one set of neighbors.

**Every offset is decided by the same neighborhoods.** An activation says something at every offset, a neuron
or a silence, so the outermost offset is decided over the same `s` as offset 0, less only the neighborhoods
skipped there. Each pattern's collapse has its own `s`; nothing is pooled across patterns, and no threshold,
smoothing or probability estimate enters.

## 5.5 Re-centering

> **D29 — Re-centering.** The operation that re-decides the neighbors a pattern names (D15) once the activations
> it covers have changed: the oldest evicted when the history is full (D18), or a new one admitted and covered
> with it (D28). It runs the collapse (D27) over the activations the pattern now covers (residual included),
> and updates the owners (D19) to what the pattern now names.

## 5.6 The margin

> **D30 — Margin.** What a pattern is worth over the history: its saving over each activation it covers (D22),
> summed, less its dictionary line (D13).
> ```
> benefit(p)  =  Σ over the activations p covers:  saving(p, O)                     D22
> cost(p)     =  its line                                                           D13
> margin(p)   =  benefit(p) − cost(p)
> ```
> It is the difference in its level's reading of the file (D14) between the file with the pattern and the file
> without it, and both terms are sums over what the neuron already holds.

A pattern is added only when its margin is strictly positive (R15) and retired only when strictly negative (R18).

## 5.7 The greedy pick

> **D33 — The greedy pick.** The operation that builds new variables and patterns out of the residual of the
> history (D21), in two stages, one candidate at a time. In each, a relation (D47) is the seed, the collapse of
> its kind grows it (D27), and it is priced:
>
> 1. **Variables.** Of the parameter and class relations, the one that would save most as it stands, its price as a
>    variable (D13), is taken first; ties go to the higher count, then to declaration order (D1), then to the nearer
>    offset. The collapse grows it into a parameter or a class, and it is kept if its price over its uses is
>    positive:
>    ```
>    a parameter    Σ over its uses ( agreeing cells · log₂ n − log₂|V| − failed · log₂ cells )  −  its entry
>    a class        uses · log₂(n / |K|)  −  its entry
>    ```
>    its entry as D13 prices it. A class or a parameter already in its table with those members or offsets is
>    used rather than a new one. A variable that is kept **claims** the cells of its uses, and every relation from
>    then on counts only unclaimed cells, so no cell is evidence for two variables. Then the next relation.
> 2. **Patterns.** Each residual row is now read in variables: a claimed cell as its variable, an
>    unclaimed cell as a neuron at its offset. Of the relations between two things standing together in a row,
>    the one that would save most is taken first, with the same ties. The collapse grows it into a pattern over
>    the neighborhoods whose residual holds it, choosing at each offset among the variables that exist, and it is
>    priced by its margin (D30), read with the candidate credited the residual only, over the activations where
>    its saving there is positive (D22), with the entry of every constant it brings new. A candidate that pays
>    joins the table and the covers it was priced on, owning the residual it names there (D19). Then the next
>    relation.
>
> A relation counted fewer than twice seeds nothing. Each stage ends when every relation of its kind has been tried
> or no residual is left. **A variable that no pattern names by the end of the pick is dropped**: a variable covers
> only through a pattern (D41, D44), and its uses stay potential until one names it. Its relation is tried again at
> the next call. **What the pick returns is the patterns added, and the classes and parameters they name.**

> **D43 — Merging.** Whenever a pattern, a class or a parameter is added or re-centered, it is compared with the
> table it is in, exactly, and of each identical group the oldest is kept and the rest are retired (§6.5):
>
> - Two patterns with the same neighbors are one pattern, their covered activations joined.
> - Two classes with the same members are one class, and their value neurons for each member are one.
> - Two parameters with the same offsets are one parameter, and their value neurons for each
>   value are one.
>
> Nothing else is merged. A pattern never makes a class or a parameter: a general pattern is built over the
> residual that the specific ones left, from a class or a parameter found there on its own (D33).

## 5.8 Connections

What follows an activation is never in the file (D12) and enters no test. It is held only here, as connections:
what stood on the apex in the frames after it, event and action alike.

> **D25 — Connections.** A neuron holds, per `(neuron, offset)`, one **connection**: a **strength**, the number
> of times an activation of the neuron saw that neuron stand on the apex at that offset, and an **estimate**, the
> mean reward the actions of those frames received. The offset is a full D6 offset from the holder's activation:
> positive in time, because what is connected to comes after, and signed in every other activation dimension.
> A coarse offset pools the exposures of every frame in its group. The neuron connected to may be a value
> neuron, which fires only beside its pattern child and is inferred with it (D45).
> Nothing about any one activation is kept.
>
> | Component | Description                                                                              |
> |-----------|------------------------------------------------------------------------------------------|
> | key       | `(neuron, offset)`, the offset strictly positive in time                                 |
> | strength  | the number of exposures: the times an activation saw that neuron on the apex at that offset |
> | estimate  | the mean reward those exposures received                                                 |
>
> Every neuron holds them, base events and base actions included. A connection to a neuron whose expansion
> places actions is a plan; one to a neuron whose expansion places events alone is a prediction; the two are
> not told apart.

## 5.9 Variables and value neurons

> **D41 — The class.** A set of neurons, its **members**, held in the **class table** of the neuron that formed
> it (D32), with no offset of its own; each member is kept while it pays for its entry (D27). A
> class is a variable (D38): a pattern refers to it at an offset, and then any member standing there fits. A pattern
> child says "these happened together"; a class inside it says "one of these stood here".
>
> **A class is not a neuron, and it covers only through a pattern**, as a parameter does (D44). A pattern's
> occurrence writes its owner and which pattern once for all it names, and its line says where each variable
> stands (D13); a variable standing alone would have to say both on every use, and would say nothing a pattern of
> the owner and that variable does not. What a class saves is the choice among its members rather than among the
> level, `log₂(n / |K|)` on every use (D13). It is built on its own uses (D27, D33), and a class no pattern refers
> to is deleted (R41). Beside the pattern child of a call that names it, it fires a value neuron for the member
> that stood (D45), and expansion puts that member back (R28).

> **D44 — The parameter.** A set of offsets from its owner's activation that hold one and the same neuron,
> naming none and holding no class: whatever agrees at its offsets fits. A parameter is held in its owner's
> **parameters table**, beside the patterns table and the class table: its offsets, and the values it has passed,
> each with its value neuron (D45). A pattern names a parameter as one of its neighbors
> (D38), and every pattern of the owner that names the same offsets names the same parameter.
>
> **Its value** is the neuron that stood at its offsets. Where its offsets disagree, the value is the neuron at most
> of them, the nearest on a tie, and every offset that holds something else is a correction (D22).
>
> **It is found on its own, independent of any pattern.** The neuron counts, over its residual, how often two
> offsets hold one neuron (D47), and where that recurrence pays for a parameter, its collapse makes one (D27,
> D33). **It covers only through a pattern**, as a class does (D41): a pattern that names it writes its value,
> `log₂|V|`, and the pattern's owner and which pattern are written once for all it names (D13).
>
> **It is grown and dropped by its own collapse** (D27), over the uses where it holds: an offset that comes to
> hold its value joins it, one that stops leaves, and it is dropped when fewer than two offsets remain. Two
> parameters with the same offsets are one (D43). A parameter no pattern names is deleted, and its value
> neurons with it (R44).
>
> **Bound from what has happened, placed at what has not.** When a pattern is expanded (R28), a parameter whose
> offsets fall partly in frames that have happened and partly in frames to come takes its value from what stands
> at the first and places it at the second. That is how a value is carried forward in time, and it is the whole
> of copying.

> **D45 — The value neuron.** The neuron of a variable for one value: one per variable and value. The owner's
> class table records which member each of a class's value neurons stands for, as its list of members, and the
> parameters table which value each of a parameter's stands for. Two variables that hold the same neuron in one
> activation fire two value neurons, and one variable that holds it again fires the same one. A value neuron sits
> one level above its variable's owner, has a level and an id like any higher neuron (D2), and holds a table, a
> history and connections of its own.
>
> It fires one level up when a pattern's bid naming its variable with that value is accepted (§7.4), at the
> bidder's coordinate, beside the pattern child, and never without it. It tells the level above what the call
> was given, which the pattern child alone does not: the pattern child says which function ran, its value
> neurons say with what. The pattern child's connections are one set for every value; a value neuron holds
> connections of its own for its value, so the level above can learn about one value where that matters, by a
> pattern naming the pattern child and the value neuron together. It costs what the value's choice costs inside
> the pattern (D13). The value neuron gives the content and the pattern the position: it is expanded through the
> pattern whose child stands beside it, its value placed where that pattern names its variable (R28).

# Part II — The past and present: a neuron

# 6. The process functions call

> **R20 — The call, in order.** Once per level per frame, the machine asks one neuron for everything it owes
> that frame, handing it the count `n` of neurons at its level (D13) and **every activation of it that fired
> this frame with a neighborhood that is not empty**,
> each with its neighborhood, at age 0 (R1). They are processed together. A neuron none of whose activations
> has a neighbor is not called; such an activation is still open (D9) and on the apex (R27) like any uncovered
> activation.
>
> | Step               | Description                                                                                                       |
> |--------------------|-------------------------------------------------------------------------------------------------------------------|
> | refresh history    | Evict as many of the oldest activations as the frame's need, then admit the frame's.                              |
> | recognize          | Cover the residual of the history with the tables, a class fitting where a member stands and a parameter where its offsets hold one neuron, and re-center every pattern, class and parameter whose evidence changed. |
> | delete             | Retire every pattern whose margin is negative, and every class and parameter that no longer pays or that nothing names. |
> | create             | Build variables, then patterns over them: seed each on a relation, grow it by the collapse of its kind, price it. |
> | merge              | Compare everything added or re-centered this call against its table, exactly; keep the oldest of each identical group and retire the rest. |
> | return             | The bids, each with its classes' members and its parameters' values; and everything retired. |

The call runs before the election (R24).

## 6.1 Refresh history

The history holds `H` activations (D18). A frame bringing `A` activations evicts the `A` oldest once the history is
full, and fewer before, as many as it takes to make room; every pattern of an evicted cover loses it. The frame's
activations then join the history, whole (D7) and wholly residual (D21).

## 6.2 Recognize

Recognition is the procedure that chooses a cover for a new activation/neighborhood (D17): the greedy cover
(D28) over the residual of the history (D21).

**A class fits where a member stands; a parameter where its offsets agree.** Measuring a pattern against a
neighborhood, a constant holds when its neuron stands at its offset, a class when a member of it stands there,
and a parameter when one neuron stands at all its offsets (D38); each
offset that fails is a correction (D22). What each class and each parameter held is what the bid carries (D31).
Neither a class nor a parameter enters the cover on its own (D41, D44).

**Cold start is silence.** A pattern covering no activations names nothing, and a neuron with an
empty table covers nothing and bids nothing.

Every pattern whose covered activations changed, by eviction or by cover, re-centers (D29).

## 6.3 Delete

> **R18 — Retire.** After the frame's activations are recognized and the patterns re-centered (R20), retire
> every pattern whose margin (D30) is strictly negative:
> ```
> retire p  iff  margin(p) < 0
> ```
> Retiring is deletion from the table (D32). The pattern leaves that instant: it stops competing for a place in
> any cover, and the neurons it held fall to the residual (D21), where the next call's recognition may re-cover
> them (R20). The retired patterns go on the return (§6.6), and the machine deletes their children (§7.5). The
> neuron keeps no retired state.

## 6.4 Create

The greedy pick (D33) runs over the residual of the history in two stages. First the variables: each class and
parameter is seeded on its relation, grown by its own collapse and priced on its own uses, and each that pays
joins its table and claims its cells. Then the patterns over the variables: each is seeded on two things that stand
together, grown by the pattern's collapse and priced by its margin, and each that pays joins the table and the
covers it was priced on. Every relation is tried, and a variable that no pattern names by the end is dropped.

> **R14 — The candidate.** A candidate `C` is what one round of the greedy pick's second stage builds (D33):
> the collapse over the neighborhoods whose residual holds the seed relation. A variable is built the same way, by
> its own collapse over its own uses (D27). Nothing seeds it from outside, and nothing grows it a
> neighbor at a time. The seed is in every one of those neighborhoods and the relation has a count
> of at least two (D33), so **nothing is ever built on a single occurrence**, and the collapse takes the seed
> where it pays there (D27). Everything else `C` names holds, in the residual, in enough of them to pay for
> itself: D27 skips owned neighbors as it does everywhere, so **a candidate is built on the residual and nothing
> else.** The
> seed is the neighbor the table is failing on most, and the collapse settles every other neighbor at once — the
> seed chooses the neighborhoods, and the neighborhoods decide every neighbor.
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
> saving(o)  =  D22's saving, with C credited only the cells of residual(o) it names
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
> choice anywhere in it.

> **R15 — The add test.** The margin (D30) of a candidate, which is in no cover: it is credited the residual
> only, and only in the activations where it pays (R14).
> ```
> benefit  =  Σ over the history:  saving(o)                                          R14
> commit iff  benefit > C's line, and the entry of every constant it brings new                         D13
> ```
> **An accepted candidate joins the cover of every activation where its saving is positive**, as the owner of
> the residual it names there (D19), so its margin at birth is what the test counted.
>
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every activation in the ring has a whole neighborhood, so the question is
> asked over the same evidence the cover will use when `C` next competes for one. Nothing here is decided on half
> an activation and nothing later can hand `C` less than the test counted.
>
> **A variable is kept by its own price** over its uses (D33), and a pattern by this test.
>
> **The pick tries every relation** (D33). What a candidate that does not pay leaves uncovered is the next
> call's residual, and its relation is tried again then.

> **R16 — What a child is at birth.** The neuron proposes, the election decides, and the machine creates. A
> pattern is the neuron's own: it is added, used in covers and bid on the neuron's evidence alone. A pattern
> with no child is bid as it is, and the bid asks for a child if it is accepted (D31). **The machine gives a
> child only to an accepted bid**, so no neuron exists for a pattern the board never bought, and it reuses a
> child before it creates one (R43). A new child has a level and an id and no dimension (D2), and is minted one
> level above its parent. It is created with
> **an empty table**: its own patterns belong to its own level, which it has not observed yet. Its *existence*
> is decided by the election, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each covers the part of an activation its
> pattern owns, and several of them may be promoted at one coordinate (D8). What they share is a parent and a
> coordinate, not a place on the board. **A child may have many parents**: every pattern wired to it promotes
> it, each from its own neuron's position (R43).
>
> **Release is the reverse**: the parent retires, the machine reclaims. The retired patterns go on the return
> (R20). A retired pattern that never had a child leaves nothing to reclaim, and one whose child other patterns
> still promote leaves the child alone (R38).
>
> **Creating and wiring are separate, and only wiring waits.** The machine creates or chooses the child the
> moment the bid is accepted — an id, its parent, its level, the coordinate it inherits (D2) and an empty table — which
> is all the frame needs to activate it and call it (R17). Once the last level has run it points every pattern
> at the child created or chosen for it and reclaims every retired pattern's child now due (R18), in one pass over its
> own alphabet. Nothing in the frame reads that pointer except a later bid, so **the wait costs nothing**.

> **R17 — A pattern added in a call is bid in it.** The candidate joins the parent's table and the covers
> it was priced on in the call (R15), so every activation of the frame whose cover it joined bids it (R20), with
> no child, and the bid competes like any other. If it wins, the machine gives it a child, reused or new (R43),
> activates it one level up and calls it with that level. A new child records its first neighborhood there and
> nothing more: its table is empty, and a candidate needs two neighborhoods (R14). If it loses, there is no child; the pattern stays in the
> table on its own margin (D30) and is bid again whenever it is in a cover.
>
> **It is born holding nothing.** No patterns, no history and no connections (R16). Everything it comes to
> hold is over the situations its parent's pattern actually took (D25), from its first activation on.

> **R41 — The life of a class.** **Born** from a class relation, two neurons that stood at one offset in different
> rows, when its collapse and its price pay on its own uses (D27, D33), with the members the collapse took; it is
> kept if a pattern of the same pick names it. **Referred to** by any pattern of the owner whose re-centering finds
> it saves more at an offset than what else fits there (D27). **Widened and narrowed** by the membership test
> (D27): a neuron joins when what it saves pays for its entry, and leaves when it no longer does. **Dead** when it
> has fewer than two members or no pattern of the table refers to it: it leaves the class table.

> **R44 — The life of a value neuron.** **Given** when an accepted bid first passes that value for that
> variable: the bid carries it with no value neuron, as it carries no pattern child, and the machine creates one
> (R16) and wires it to the variable's entry for that value (§5.1). **Fired** by the election and by nothing
> else (§7.4). **Dead** when its variable is deleted, when its member leaves a class, or when the neuron it
> stands for is deleted:
> it goes on the death ledger and is deleted like any neuron nothing can fire again (R38).

## 6.5 Merge

Recognition re-centers, and creation adds, so either can leave two things in a table that are the same: two
patterns with the same neighbors, two classes with the same members, two parameters with the same offsets.
Merging compares everything added or re-centered this call against its own table, exactly (D43), and keeps
the oldest of each identical group, the lowest creation order, because the oldest has already accumulated its
children's histories, connections and estimates. The rest are retired:

| identical | kept | retired |
|---|---|---|
| patterns | the oldest, taking the others' covered activations | the newer patterns; each pattern child dies when nothing points to it and its last open activation closes (R38) |
| classes | the oldest and, for each member, its value neuron; patterns that referred to a newer one now refer to it | the newer classes and their value neurons |
| parameters | the oldest and, for each value both passed, its value neuron | the newer parameters and their value neurons for values the oldest also passed; a value only a newer one passed moves to the oldest with its value neuron |

What merging retires goes on the return with everything else retired this call (§6.6), so the machine deletes
the children on its ordinary pass.

## 6.6 Return

The call returns two lists: **the bids**, one per pattern of each of the frame's activations' covers (D31, R21),
and **everything retired** this call, by deletion (R18) or by merging (§6.5). A bid is a line of the table, the
pattern's id, its neighbors and its child (D32), with its classes' members and its parameters' values (D31). A
pattern that has never won a bid has no child, and its bid says so: if it is accepted the machine gives it a
pattern child and a value neuron for each value it passed that has none,
reused or new (R16, R43, R44), and wires them by the wire call (§5.1). A pattern retired is named by its id, and
the machine deletes its pattern child, if it has one, when it is due (R18).

> **R21 — One bid per pattern of the cover.** An activation sends one bid (D31) per pattern of its cover. A
> neuron covering nothing sends nothing.

# Part III — The past and present: the machine

# 7. Contraction

> **D23 — Contraction.** The machine covers the level below with neurons from the level above, each taken when
> the neurons it covers cost more than its bid costs to state, never counting the dictionary line (R22, R24).
>
> **Covering everything is not the goal.** A neuron no accepted bid covers stays in the file as itself, at a neuron's
> price (D13), and that is the shorter file whenever no neuron could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a pattern names neighbors at offsets, so a promoted neuron replaces a chunk of
> spacetime. Spatial contraction is the case where every offset is zero.

**The machine compresses a frame in this order and no other.** For each level, from the base up, until a level
accepts no bid:

| Step              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| process functions | Call every neuron with an activation at this level, value neurons included (§6).                              |
| elect bids        | Cover the level's uncovered activations with the bids, by the greedy cover over the board.                    |
| create children   | Give every accepted bid that has no pattern child one, and every value it passed that has no value neuron one: a pattern child reused where another bid of the election tied it, created otherwise (R43). |
| activate children | Every accepted bid activates its pattern child one level up, and a value neuron for each of its variables; the uncovered stand on the apex. |

Then, once the last level has run:

| Step              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| delete children   | Wire every child created this frame to its pattern, and delete every child that is due.                       |

## 7.1 Process levels

**A level is explained as a set of function calls with their arguments.** Every accepted bid is a call, a
pattern child, and every variable it names is an argument, whose value neuron fires beside the pattern child with
what the variable held (D45). The level above is made of the calls and their arguments (§7.4). A pattern with no
variable is a call with no arguments.

> **R26 — One stack, at the derived reach.** Base neurons run `process functions` and offer; the election settles which bids
> are bought. The survivors are level 1 — the fewest that cover the active base neurons — and it happens
> again. **When a level's active neurons promote no children, nothing propagates and there is no level above
> it on this frame.** Nothing declares the depth and nothing caps it.
>
> **Within a level the order is call, elect, create, activate**, and it cannot be otherwise: the bids are what
> the election is over, a child is created only for a bid the election accepted (R16), and what is activated is
> what was elected.
> Nothing in that order leaves the level or the frame.
>
> Every level runs the same rule at the reach one expression gives it (D4). **Compression is spatio-temporal
> at every level, in one pass**: a pattern at any level may name neighbors in its own frame, in earlier ones,
> beside it in space, or in a mix.

## 7.2 The election

> **R22 — What a bid covers, and what it costs.** The neuron sends the pattern with its classes' members and its
> parameters' values, and nothing else. The machine holds the frame, so it reads that one object against what fired
> and derives both numbers.
> ```
> the bid   the pattern, the child's id or none, its classes' members and its parameters' values      (D31)
> covered   the bidder, and the neurons it names that fired and no earlier bid covers — the activations it
>           asks to subsume, what its classes and parameters were fit by among them — a neuron each
> price     the fields its call writes (D13): its owner and which pattern, what its classes and parameters
>           leave open, and which of the neurons it names in those same frames did not fire, against the n the
>           machine handed the level's neurons
> ```
> `coverage − price` is the saving over stating the chunk flat, D22's expression over the machine's population.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as itself if
> nothing covers it (D23), and the same way if this child is promoted, since the child does not name it — a neuron
> either way, so it cancels before the test begins, and charging it here would count it twice against the uncovered
> term of the same sum (§7.2).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another bid
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged a failed cell whether or not another neuron is
> right at that coordinate — another neuron's expansion being right there does not make this one's wrong name free. **What a
> pattern gets wrong about a frame is a fact about the two, and ownership does not move it.**
>
> **This is the neuron's arithmetic over the machine's population, and the number is not the neuron's.** The
> neuron took the pattern into its cover, or offered it, on its own residual (R20); the machine tallies on a
> board where earlier frames' credit stands (R23) and this frame's other bids contend (R24). The two numbers
> differ, and are meant to (D22).
>
> **This is a price for one bid, not for the symbol.** The dictionary line is weighed by the one test
> (D30) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R22).

> **D24 — The coverage set.** Per level, the machine keeps which accepted bid was credited each covered
> activation:
> ```
> coverage set    per level    which accepted bid holds each subsumed active activation
>                 the owners             one owner per activation; a settled activation never changes owner
> ```
> **An activation is named by its full coordinate**, dimension and position together, so two activations of one
> neuron at two positions never contend. Level `k`'s coverage set spans `reach(k) + 1` frames — a
> bid reaches `reach(k)` back and no further — and the box every other activation dimension gives, and ages
> out with it. **The machine holds nothing on the scale of the run.**
>
> Ownership is about **credit**: a neuron is a fact that needs paying for exactly once, so it is settled
> once and never revisited (R23). It is not about naming: a neuron expands to everything its pattern names,
> credited or not (R24).

> **R23 — This frame's bids against the board as it stands.** Only neurons no earlier frame's election has
> credited are in play, so a chunk already paid for is not paid for twice. **No earlier promotion is ever
> re-scored.** Within the electing frame an activation is credited once, to the first accepted bid that names it
> (R24), and never moves.
>
> **Earlier bidders have priority.** A bid at `f` wins a neuron at `f − 2` before a better bid at `f + 1` can
> name it, because the earlier election settled it and nothing re-elects the past. That is the price of never
> revisiting a frame, and the design pays it.

**The file over one frame is the calls promoted plus the neurons no bid covered**: the price of every accepted bid,
and a neuron for every neuron no bid covered, the body half of `L` (D14) over the frames the election can see.
**The two terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first,
which is why the price counts only the neighbors named and absent (R22). The dictionary half is D30's, and neither
test touches the other's sum.

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
> `price` is R22's. **Ties go to the older symbol, then to the earlier coordinate** — creation
> order for a pattern and declaration order (D1) for a base neuron, then frame, then position. Then D28 runs,
> and what it takes are the accepted bids.
>
> **The bound is structural.** An accepted bid takes at least two activations: its price includes its owner, a
> neuron, so it must cover more than its owner. So the rounds are at most half the free set and never more than
> the bids.
>
> **A bid that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> an activation it named is either credited to a bid that did pay or stands as its own line.
>
> **Ownership is a partition of the neurons the accepted bids name**, and that is the whole of the
> inhibition — no bid is ever edited or forbidden, and **overlap is legal and priced**: a bid that names an activation
> an earlier round credited gains nothing for it and pays nothing for it (R22). **Held by an accepted bid** and
> **named by an accepted bid** are therefore the same set, so coverage, credit and the apex frontier (R27) are
> one question with one answer.
>
> **Outcome**: accepted bids are promoted, one neuron each and **whole** — a child expands to everything its
> pattern names, credited or not — the neurons credited to them are subsumed, and every active neuron no
> accepted bid covers stands as itself. **The election delivers nothing to anyone**: it writes the coverage set
> and stops. No neuron is told which of its bids were bought, what they were credited, or what they lost; a
> neuron's history is what it saw, and the board is the machine's.

## 7.3 Create children

Every accepted bid whose pattern has no pattern child is given one, and every value it passed that has no value
neuron is given one. A bid that lost is given
nothing.

> **R43 — Reuse or new.** The candidates for reuse are the bids already on the board. Every neuron the accepted
> bid covered was called this frame and bid every line of its cover (R21), so a line that describes this ground
> from another of its neurons is among this election's bids, measured on the same board at the same prices
> (R22).
>
> **A child is reused on a tie.** If another bid of this election carries a child, and names, its bidder
> included, exactly the activations the accepted bid was credited, at the same price, the two lines state one
> chunk from two of its neurons, and the accepted bid's pattern is wired to that child. Between several such
> bids the older child is taken. A bid that measured better than the accepted one would have been taken before
> it (D28), so reuse is only ever a tie; a bid that does better than every bid carrying a child states a
> different chunk, and gets a child of its own.
>
> **A value neuron is its variable's.** It is reused whenever the variable passes that value again, and never
> shared with another variable (D45).
>
> **Otherwise the machine creates one**: an id, its parent, its level, the coordinate it inherits (D2) and an
> empty table (R16); a value neuron at the pattern child's level, with no dimension (R44).
>
> **Nothing is sent back.** The pattern stays the neuron's own line and keeps re-centering on its own history
> (D29); only the pointer is shared. Lines wired to one child may drift apart, as one line drifts over time,
> and the child stands for all of them. The neuron optimizes its history and the machine its window.

Once the last level has run, the machine points every pattern at the child created or chosen for it, and every
variable's value at its value neuron, by the wire call (§5.1).

## 7.4 Activate children

Every accepted bid activates its child one level up, at the bidder's coordinate (D2). The activation records the
pattern whose bid fired it, since a child may have several (R43), and is expanded through that pattern (R28). For
each variable the pattern names it also activates that variable's value neuron for the value the bid carried (D45),
at the same level and the same coordinate as the child. Every activation no accepted bid covers stands as itself.

> **R27 — The apex is a frontier, not a level.** It is every active neuron **no accepted bid covers** — the
> uncovered set, at every level at once — so a base neuron nothing found worth chunking stands in it beside a
> level-4 pattern. This is the frontier the file's body writes, **the one that learns what ran** (D25), and
> **the one that votes** (§8.3): the uncovered set does all three, and coverage silences a neuron in every one of
> them at once (D10). A reward for a frame already written reaches its connection regardless (R33). Everything
> underneath the current frontier is recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that offered a child and had its bid declined with nothing else
> covering it is still on the frontier, and stands in the file as its own line.
>
> **Coverage is acquired late and never revoked.** An activation firing at `g` is uncovered until some bid
> takes it, and the last bid that can fires at `g + reach_t` (D4), so an activation may speak for a few frames
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

**Events and actions run together within a level.** An action fires in the same column as the events it runs
alongside (D8), so it is recognized and chunked by the rule they are, in the same patterns (D5). **The
connection is not formed here**: it names what stands on the apex, which is known only once every level has
settled, so it is recorded in the `process actions` pass instead (R31).

---

## 7.5 Delete children

> **R38 — A child dies when its last parent has retired and its last open activation closes.** A pattern the
> neuron retired (R18) names a child the machine still holds, unless no bid of it was ever accepted and there
> is nothing to delete (R16). The machine wired every pattern that promotes the child (R43), so it knows when
> the retired one was the last; while another remains the child lives and only the retired pattern goes. The
> neuron says nothing about when the child should go: it sees its own open activations and not the children
> promoted off them, while the machine sees both.
> ```
> death frame   =   when the child's last open activation closes
>               =   this frame, when none is open
> ```
> That set only shrinks. Once every pattern wired to the child has retired nothing can fire it again, and no level built afterward can name it either, because a level is built out of
> what is firing (R26). Reach grows with the level (D4), so the last to close is the highest one and the wait
> is at most `reach_t(D)` frames, `D` being the highest level the stack currently holds.
>
> **Deleting is on the same pass that wires** (R16): **every frame, once the last level has run**, the machine
> reads the **death ledger** and takes everything due — the pattern, its child neuron and that neuron's subtree
> together, at once, and what named them scrubbed with them by the delete call (§5.1). A pattern retired this frame with nothing open dies on
> this frame's pass; one whose child is still open waits exactly as long as the stack above it needs, and not
> a frame longer. **Nothing traces who is naming what**: the machine settles the question off the board it
> already keeps.
>
> **The ledger holds the pattern, not a handle to it.** A child is stated by the patterns wired to it (D12),
> and each activation of it is expanded through the pattern whose bid fired it (R28, §7.4). Until the
> activations a retired pattern fired have closed, neurons above them still cover them and the apex may still
> expand them, so that pattern has to stay readable after the table stops covering with it, whether or not the
> child lives on.

# Part IV — The future: action and reward

# 8. The process actions call

Once the last level has run, the machine works from the apex down to the base alphabet, in this order and no
other:

| Step               | Description                                                                                                                  |
|--------------------|------------------------------------------------------------------------------------------------------------------------------|
| process actions    | Call every open activation with what stands on the apex this frame and any reward at its distance; collect what those on the apex infer. |
| expand inferences  | Place each inference at its completion and expand it through the dictionary to the base symbols it names.                    |
| resolve actions    | One winner per action dimension at the frame ahead, by estimate; it is output for that frame, and what ran arrives with that frame's report and its reward. |
| place inferences   | The actions and events the winning inferences' expansions place at the frame ahead fire weakly there (D40). |

> **R29 — Two frames: infer, then execute and reward.** What is chosen in one frame is output for the next, and
> what ran, with what it earned, arrives with that frame.
> ```
> f      infer     the frame's events are recognized, `process actions` returns the inferences,
>                  the inference resolves (R36), and an action is output for the frame ahead
> f + 1  execute   the report names what ran, and its neuron fires strong in this frame's column
>        reward    alongside this frame's events — every uncovered activation open connects to the
>                  apex; what the action earned arrives as this frame's input, and the connection
>                  it strengthens takes the reward in the same write (R31)
> ```
> **The reward is part of the frame the action ran in.** The environment reports what it observed and what the
> action in effect during that frame earned together, so an action is never on the books without its outcome,
> and no activation has to stay open for a reward that arrives later than the action it pays for (D9).

## 8.1 Process actions

**The machine calls every open activation once more**, at whatever age it stands at, with two things:

```
the apex          every activation standing on the apex this frame (R27). **Only if the
                   activation is uncovered at this frame** — one an accepted bid covers writes
                   nothing more                                                           D10, D25, R31
the reward         any reward that arrived, for the actions of this frame and for any earlier
                   frame the reward spans, at the distance each one names                       R33
```

The neuron strengthens a connection per apex activation and saves nothing: the connection at `(neuron, age)`,
created at strength 1 or incremented (R31), and for each reward share the estimate of the connections it names
(R33) — for this frame, the connections just strengthened, in the same write. A share for an earlier frame
reaches the connection the activation wrote to at that frame, whether or not coverage has arrived since. Nothing
is written into the activation. **Nothing is decided, priced or compared here**, and no test is waiting on any of
it.

**If the activation is uncovered, the call returns what it speaks.** It reads its own neuron's connections at
every offset beyond its age, out to its reach — a connection at offset `b` read at age `a` is a claim about a
neuron completing `b − a` frames ahead (R28) — and returns each with its strength and estimate. Those are its
**inferences** (§8.3), each naming a neuron at whatever level it stands (D25), and only what reaches the frame
ahead is resolved. A base neuron on the apex speaks from its own connections like any other;
what it infers is the marginal over every situation it fires in, and it speaks only because nothing more
specific covers it.

**An activation closes at age `reach_t`** (D9), once that frame's exposure is taken. Closing does nothing but
stop the writing — there is no second call and nothing is saved twice.

> **R31 — A connection carries an estimate.** What stands on the apex at `f + 1` is not known at `f`: the
> action is settled only when the report names what ran, and the events only when the environment reports them
> (D40). So at `f + 1`, once the last level has run, **every uncovered activation open at that frame connects to
> every strong activation standing on the apex of that frame** (R27) at its own age (D25, §8.1), and the reward
> of that frame's actions goes into each. A weak activation is connected to by nothing: it is a proposal, not a
> fact. That connection binds what the neuron stands for
> to what followed — formed against what actually stood, so **a neuron that inferred something else, or nothing,
> learns from what happened.**
>
> **One activation connects to the apex of every frame it is open through, and one strong apex activation is
> connected to by every uncovered activation open when it stood.** The offset is the age — the distance from the frame the
> activation opened to the frame the call ran — rounded as every offset is (D6), so a neuron open at ages 1, 2
> and 3 holds the same neuron at two offsets, `1` and `2`, and the exposure at age 3 strengthens the second.
> **A coarse offset takes one exposure per frame of its group**, so the outer offsets pool the apexes of many
> frames, each at its own strength, as a coarse offset carries several neighbors (D6). A neuron that stood twice
> inside one group over one activation is two exposures: each frame has its own reward, and the connection keeps
> nothing that could tell the two apart (D25).
>
> **Strengthening and reading are inverses**: an exposure is written at the offset its age rounds to, and read
> back at the age from which that offset places a completion one frame ahead (R28, R36), so what a replay earns
> lands on the connection it was learned from. Fan-out is the apex: a neuron connects to what stood uncovered
> in the frames after it and to nothing beneath that, and one exposure per apex activation per frame means a
> level-`k` neuron holds at most `k + 1` offsets per neuron it has seen.
>
> **Making and strengthening are one operation.** A neuron's connection at `(neuron, offset)` has a
> **strength**, the number of its exposures — the times an activation of the neuron saw that neuron on the apex
> at that offset — and an **estimate**, the mean of the reward those exposures received (R33). The first exposure creates
> the connection at strength 1 and every later one increments it; each share of reward folds into the mean
> weighted by `1 / strength`, so the estimate is the exact average over the connection's exposures and no rate
> is chosen. **Nothing leaves.** No exposure is ever subtracted, no strength ever falls, and no window bounds a
> connection: it is the whole of what the neuron has seen follow it at that offset, over its life.
>
> **Every level holds them, base neurons included.** No operation derives what a child was worth from what its
> parent earned or the reverse, so a connection held only at the frontier would be lost at the next mint. Every
> neuron keeps its own connections over its own activations (D25).
>
> **A connection dies with either of its ends.** Nothing else removes one (R34).
>
> **A connection wired ahead of any exposure is created at strength 1 and estimate 0.** The walk (R37) is the one
> thing that creates a connection nothing has yet been seen to follow; it is created exactly as a first exposure
> at neutral reward would create it, and from then on it is a connection like any other.

> **R32 — What is learned is what happened.** No inference is credited and nothing is in control: what stood
> on the apex is what every uncovered activation connects to and what its reward lands on (R31), whether that
> activation inferred it, inferred something else, or inferred nothing. Before any pattern exists the apex is
> the base symbols, so the rule holds across all of development.
>
> **A pattern earns what its program earned.** It fires when its expansion completes (R30), so what arrived in
> that frame lands on it directly, and what arrived over the frames its program occupied reaches it through
> the span a reward names (R33). That is what makes a multi-frame candidate comparable to a single-frame one.

> **R33 — A reward names what it pays for, and the machine fills in the rest.** A reward (D35) pays the channels
> it names over the span it names, every channel and the whole window when it names none.
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
> is sampled once and averaged away. Each share is delivered, in the `process actions` call (§8.1), to every open
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

## 8.2 Expansion

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> the neighbors named one level down by its pattern, the one whose bid fired the activation (§7.4), at that neuron's offset plus theirs — offsets compose because
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
> **This is the one expansion in the design.** It recovers the run from the file (D12): a class expands to the
> member its value neuron names, and where an inferred call brought none, to its most frequent member, the
> oldest on a tie. And it turns an inferred
> pattern into a program and an expectation at once (R30): the actions it places run, and the events it places
> fire weakly as what those actions are expected to bring (D40). It reads dictionary lines only. Recognition runs it backward: a pattern fit by what
> fired says that this function ran here, with these arguments, and the level above writes the call and its
> values. **What
> travels down with a symbol is
> what the connection carried**: every base symbol an inference's expansion places carries the strength and the
> estimate of the connection it came from, and nothing is re-weighted on the way down.
>
> **Expansion substitutes the arguments.** A body is written one level below the neuron whose pattern it is, and a
> call's value neurons stand one level above that, beside the pattern child. Where the body names a parameter, the
> machine places the parameter's value, read from its value neuron, at every one of the parameter's offsets: the
> function run with its argument substituted in. Where an offset of the parameter falls in a frame that has already
> happened, the value is read from what stands there and placed at the offsets still to come (D44): that is how a
> value is carried forward. A class is placed as the member its value neuron names, or as its most frequent member
> where none came. A neighbor that is itself a higher neuron is expanded the same way from where it is placed.
> Recognition is this substitution run backward: the body is fit by `t` standing at every offset of a parameter,
> and the level above records that parameter's value neuron for `t` (D45). **Two words, never one.** A pattern is
> **expanded**, downward, into the level below; a connection is **inferred**, forward, into the frames ahead (D25);
> what a call does and what follows it are different directions.
>
> **A connection is placed the way a neighbor is.** A connection at offset `b`, read by an activation at age `a`,
> puts the neuron it names `b − a` frames ahead — that is where it completes — and its expansion hangs
> from there, its base symbols at the frames back from it. **A coarse
> offset is not a window.** Its action completes at
> `b`, not somewhere in the group `b` stands for, exactly as a neighbor named at `−b` is placed at `−b` and nowhere
> else. So the steps of a long program reach the frame ahead one at a time, in order, each at exactly one age,
> from one connection and with nothing held: what an activation places beyond the frame ahead it places again
> next frame, one frame nearer.

> **R30 — Execution is an expansion**, of the selected pattern (R36) through its dictionary line (R28). A high
> pattern becomes its members at the offsets its line records, down to base symbols. Its base actions are output,
> each in the frame its expansion places it in, the nearest being `+1` (R29); its base events are placed in
> theirs, as what the program is expected to see. Both fire weakly (D40), and the report replaces them: the
> actions it names as run and the events it names as seen become strong, and the rest last their frame.
> Execution is not a second mechanism; it is this expansion read as a program, and expectation is the same
> expansion read as a prediction. A weak pattern's expansion places no action: what is output is chosen by
> estimate, never by expecting itself to act.
>
> **Execution activates what it expands.** The base actions fire weakly in the frames they are placed, and every
> pattern the expansion passed through fires weakly when its expansion completes, at the coordinate the
> expansion placed it — the last frame of its chunk, which is the coordinate recognition would have given it.
> Where the report makes its actions strong, recognition builds the strong activation of the same pattern at
> the same coordinate, and it replaces the weak one (D40). An executed pattern is therefore on its level's frame
> like a bought one: it runs `process functions`, is chunked upward (§7.1), and every uncovered activation open
> at that frame connects to it (D25). While its
> program runs, the apex action at each frame is the base action, or a lower pattern as it completes; the whole
> stands on the apex only in the frame it completes, and what the program earned reaches it through the span a
> reward names (R33).

## 8.3 Resolution

> **R35 — Selection.** No fit says which action to take; it says only what a situation was followed by.
> Choosing comes from the connections of the neurons standing on the apex, each carrying the reward that
> arrived averaged over its exposures, and **the machine outputs the best. Nothing else decides it.**
>
> **A situation is a set of active neurons** — any one of them, and any subset of them. Situations
> intersect, and **nothing ever materializes one**: a situation is what a voter fires in, never an object the
> machine holds.
>
> **A voter is one apex activation at one age**, reading its own neuron's connections at the offset ahead
> (R36). A base neuron's estimate is a marginal over every situation it fires in; a child's is
> over the one situation its parent's pattern names; so the same frame reads differently to a base neuron, to
> the level-1 child covering a chunk of it and to the level-4 child covering the whole, and differently again to
> any of them two ages later. **A minted pattern is how one recurring situation acquires an estimate of its own**
> — which is why a covered neuron is silenced (D10) and its coverer speaks instead, and the only reason a base
> neuron's marginal ever speaks is that nothing more specific was bought over it.
>
> **Every apex activation votes**, base actions included. A pattern that names what was done beside what was
> seen votes knowing the situation; a chunk of actions alone, or a base action, votes what followed it, a habit,
> and reward prices the habit like anything else. Nothing selects an event: what an inference places in an
> event dimension is expected, not chosen (R30).
>
> **The default is output; it is not wired.** An action dimension no inference reaches is output the declared
> default action. Nothing holds it in advance: where the report names it as run it is the apex action of that
> frame, every uncovered activation connects to it with its reward by the ordinary path (R31), and from then on
> it is an action like any other. A neuron is born holding no connection at all — a base neuron at cold start and a freshly
> minted pattern alike say nothing about the next action until something has run under them, and what they
> then hold is whatever was executed. R37's walk is the only thing that wires an action ahead of its running,
> one at a time and only where the best known has been judged and found wanting.

> **R36 — The inference decides.** An inference is one connection read by one voter: it names a neuron, at
> whatever level it stands (D25), and carries a strength and an estimate. What runs is chosen from the
> inferences and nothing else; what its expansion places in event dimensions is expected (R30).
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
> Each reads its neuron's connections at **every offset beyond its age**, out to its reach. Every
> connection there is one inference, naming an action neuron and carrying a strength and an estimate, and it
> expands as above; what its expansion puts at `f + 1` is what it proposes, and a connection whose expansion puts
> nothing there proposes nothing this frame. A base action is proposed by the one offset the age places one frame
> ahead (R29); a pattern's first step by a farther one.
>
> **An inference is a call**: the action a connection names (D25), and the call fires at the voter's coordinate
> (D37), where its parameters are bound from what stands within reach (D44).
>
> **Position drops out.** Two activations of one neuron at two positions read one set of connections, so they offer
> the identical inference and the argmax is indifferent to the duplicate. Two *ages* are two voters and do not
> collapse together — they read different offsets, so they can name different actions.
>
> **One winner per action dimension, by estimate.** For each action dimension at `f + 1`, every base action
> some voter's expansion placed there is a candidate (D37); what the same expansions place in event dimensions
> fires weakly there once the winners are settled, the expectation of the winning plans (R30). **A candidate's estimate is the mean of the estimates
> its voters placed it with, each weighted by that voter's share** — one vote per voter, split across the actions
> it placed in that dimension by strength, so a voter that hedges between two actions counts as one voter and
> not two — and **the candidate with the largest estimate is output**. Ties go to the larger share of voters, and
> then to the older action. **Level is not read**: a level-4 voter's inference and a base voter's meet on the
> estimate alone, and a specific situation wins over a general one only by being right about what pays, never
> by rank. **Nothing corrects for how many exposures an estimate rests on**, so a sharp estimate on three
> exposures outranks a coarse one on two hundred.
>
> **A covered neuron supplies nothing** (D10). A newly minted child therefore starts with no inference at all; the
> other voters, or the default, decide until something has run under it, and it learns what ran (R35).

> **R37 — Exploration.** The default policy resolves explore–exploit without randomness: **the action alphabet
> is declared in order**, and **a connection whose estimate turns negative wires the next action in that
> order** — the first one the neuron holds no connection to at that offset — at strength 1 and neutral estimate
> (R31). The walk is over the actions a neuron can name; what a call's parameters hold is never searched, it is
> whatever stands at their offsets when the call runs (D44).
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

## 8.4 Expectations

> **R40 — An inference is weak until the report makes it strong.** The winning inferences' expansions place
> events as well as actions (R30), and both fire weakly in the frame they are placed (D40): the action as the
> machine's output, the event as what it expects. Where the report names that dimension at that coordinate, the
> report stands and the weak activation is dropped; an action nothing reports as run, or an event nothing
> reports as seen, lasts its frame and is gone. Where the expansion names a parameter, what fires is its value,
> bound from what has happened (D44). An expected event above the base is expanded in turn, one frame at a time
> as its members come due, so what a plan expects reaches the base as expected inputs exactly as what it does
> reaches the base as output. A weak activation is an input of its frame: it is recognized as a neighbor, it may
> be covered or stand on the apex and vote once, and then it is gone. It carries no reward, connects to nothing,
> and is connected to by nothing (R31).
