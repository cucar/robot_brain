# Universal Compression with Actions and Rewards (UCAR)

UCAR is an algorithm that detects patterns in space and time, and gives each pattern that keeps recurring a
neuron of its own. Those neurons are symbols too, so the patterns they form get neurons in turn, level by
level, and what the machine sees comes to be stated more and more briefly. A pattern is held as how its parts
stand relative to one another, never where they stood, so a pattern learned at one position is known
everywhere. Every neuron also remembers what followed it and what reward that earned. Each frame, the neurons
that describe the present say what usually comes next and how well it paid, and the machine acts on the best
estimate. That is the whole algorithm: it compresses what it sees into names, and acts on what the names have
learned.

A machine running it is a **UCAR machine**, and like a Turing machine it is defined by two alphabets: the
**event alphabet** it can observe and the **action alphabet** it can execute. Every symbol, base or learned,
event or action, is a **neuron**.

Its inputs are the events observed, the actions that ran and the rewards earned. Its output is what it infers
at the frame ahead, written in the base alphabet: the actions and the events it infers.

**What it compresses is a graph.** Every activation of a neuron is a node. Every relation between two activations,
one at an offset from the other in time and space together, is an edge. The graph spans time and every dimension
of space, and it is the only record the machine keeps of what happened: it holds offsets, never coordinates. A
neuron sees the edges around each of its own activations and reports them with how often each has held. The
machine groups the edges into **functions**, graphs over neurons that keep recurring, names each with an
interneuron, and from then on the name stands where the graph recurs. A function in time is recognized as it
unfolds, over the nodes that have come, and its call is one activation of the interneuron in a **phase**: which
node the fit ran from and to, and how many laps. A neuron in a phase is a **pose**, and a relation is between
two poses. The call stands for what it covers, and the relations of it that no call covers are **lifted** to
the call; what always followed it is a function at the level above, over its call.
Where a function is general it leaves a **variable**, a node whose neuron changes from one occurrence to the
next and is filled by what stands there. The interneurons are nodes like any other, so the next level groups
them in turn, and the dictionary is a hierarchy of named graphs.

**A function is a procedure in a very small language.** Its instructions are the base symbols, each at a
position in space and time; a line may call another procedure at a position; a variable is bound where it is
first read and tested where it is read again; a cycle is a goto to a line already read, run while its test
holds. There is no return value: a procedure's effect is what it expands to, and what follows
it is held outside it, as connections from its call. Recognition parses the text backward once its last line
has run, and expansion interprets it forward.

What no function covers is the **apex**: the machine's state at a frame, the situation it is in, and the set
that infers the frame to come. Every neuron records what followed its activations along with their rewards.
Each frame, every activation still on the apex connects to what stands on the apex
now, building a network of apex-to-apex connections. The machine expands the inferences those connections make
and resolves them at the base, matching them against what is observed. One winner is resolved per dimension, and
the machine outputs them: the actions and the events it infers. The environment executes the actions, and
reports what ran, what was observed and what it earned.

**D** is a definition and **R** a rule; together they are the machine.

---

# Part I — Structure

# 1. Objectives

There are two objectives: **compress what is observed**, and **act on the best reward estimate**. The
compression decides what the current situation *is*, and that is handed to the second objective as its input.

## 1.1 Compression: classify the situation

**The machine compresses by naming.** A function is a graph over neurons that keeps recurring, and its
interneuron stands for it wherever it does. The graph of the run is stated shorter by the name standing for the
graph. Functions may name base symbols or the interneurons of other functions. This substitution is the whole
mechanism of compression.

**A function is what has happened, and a connection is what followed.** A function in time is recognized as
it unfolds: in each frame one of its nodes fires, over the nodes that have come, its call standing in the
phase that ends at that node (D54). A relation
joins a function where it pays over the occasions the neurons remember (R50). A function's latest node never
moves, so it gains nodes toward its past only, and what followed it is a function at the level above, over its
call. What followed any activation, its function's later nodes
included, is on the neurons as connections with an estimate (D25), and the reward lives there, where the choice
is.

**The compression is lossy in two ways, and both are limits of the neuron.**

| Loss      | Description                                                                                                                                       | Reference |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| memory    | A neuron has limited memory: it remembers only its most recent activations, and reports its relations over those, so what it saw earlier no longer shapes the dictionary. | D18       |
| laps      | A call's phase says how many laps its cycle ran, and a relation to the call names those laps, so the dictionary holds a stretch at each length it recurred at: the same shape at two lengths is two functions until a variable over laps and offsets joins them. | D15, D37, D34 |

**A neuron's vision is a limit, not a loss.** A base neuron sees its neighbors one position away in each
dimension and nothing beyond (D4). What lies farther reaches the dictionary through the hierarchy: an interneuron
sees one position beyond what its function covers, so each level relates things the level below could not see
together. Nothing is rounded anywhere: every offset the machine holds is exact.

The loss is what lets the machine generalize: a neuron that remembers recently reports what recurs, not the
instances it happened to meet. Imagine the graph of the run stated under the functions as they now stand, and
read the objective as: make that shorter.

## 1.2 Execution: the best estimate for the situation

A connection of a neuron names the neuron that stood on the apex at an offset after it (D25). It holds two
numbers: strength and earnings. Strength is the number of times the connection was seen. Earnings are the total
reward paid in those frames (R31). Its estimate is its earnings over its strength, the average reward per time
seen. Every frame, each neuron standing on the apex infers from its connections what follows it (R36). If an
inference names a neuron above the base, it is expanded down to the base neurons it stands for (R28). The base
neurons inferred at the frame ahead are then resolved into the machine's output. In each action dimension the
machine outputs the action inferred there with the largest estimate (R47). In each event dimension it outputs the
event the most inferences name there, and reads no estimate. That is the whole of it: **act on the best
estimate you hold, for the situation you are actually in.**

**Inference does not inhibit.** What a connection infers covers nothing. Only a call recognized at the present
covers, and only what has happened (D54, R40). An action is chosen by connections and their estimates, never
by a function: a function records what was done, afterward.

**A reward pays frames and channels.** It arrives with a frame and may name the action channels it pays and the
span of frames it pays over (D35). A reward that names no channels pays every action channel. One that names no
span pays as far back as any neuron can still see, the longest reach in time the machine holds. The share of it
reaching each frame of the span falls linearly with that frame's distance from the one it arrived in, and lands
on the connections written at that frame (R33).

**Compression decides which neurons exist, and the neurons hold the connections.** A situation that recurs is
named by a function. The function's interneuron then stands on the apex for it, and holds connections of
its own. Until then, the neurons that make up the situation each hold their own (R35).

---

# 2. World

The world is everything outside the machine. The two meet once per frame. The world reports what it observed,
what actions ran and what they earned. The machine outputs the actions and the events it infers for the frame
ahead. The actions run in the world, and the next frame reports what came of them.

## 2.1 Dimensions

> **D1 — Dimensions.** There are two types of dimensions.
>
> A **neuron dimension** says what a symbol is. It is an **event dimension** or an **action dimension**, and is
> declared with its **resolution**, its bucket count. A base symbol is a dimension–bucket pair. The neuron
> dimensions together are the alphabet.
>
> An **activation dimension** says where one instance of a symbol happened relative to the others. It is an axis
> of the world: time or an axis of space. Every world has time. An input arrives laid out over the activation
> dimensions of its channel, and the layout says which inputs are beside which, and at what distance. The
> activation dimensions are declared in a fixed order, time first.

For example, a pixel's color could be a neuron dimension, an event dimension of 256 buckets. A motor's
direction could be another, an action dimension of three buckets: forward, back and stop.

An image's `x` and `y` are activation dimensions, and so is time. The world lays an image out as pixels at
coordinates, and lays frames out in order. The machine lays a frame out the same way, so that every activation
has a position on the board, and reads the layout for one thing: the offset between any two activations in each
dimension (D6). The graph keeps the offsets and nothing of the positions.

## 2.2 Channels

> **D3 — Channels.** A channel is defined by an event dimension, an action dimension, and spatial activation
> dimensions. A channel has an event dimension, an action dimension or both, and owns them exclusively. The same
> activation dimensions may be shared by several channels. Each frame, the channel reports a symbol of each of
> its neuron dimensions at each point of its layout.

For example, a vision channel is laid out over `x`, `y` and time. A stream of prices could use time alone.

Three color channels of one camera use the same `x` and `y`, so a red pixel and a green pixel can stand at the
same position in one camera. Two cameras with separate screens name separate axes.

The machine declares its channels once, and the set is fixed over the whole run.

## 2.3 Frame

A frame arrives carrying what each event dimension observed, what each action dimension executed (D8), and any
rewards for actions already run (D35). An event dimension carries what was observed, quantized to its buckets.
An action dimension carries the action that ran.

An event is a base symbol of an event dimension (D1), and it fires when the environment reports that it was
observed (D8). An action is a base symbol of an action dimension, and it fires when the environment reports
that it ran.

At most one symbol is reported per dimension per point of the layout in a frame (D8): one event observed, one
action run. The machine outputs what it infers for the frame ahead, one symbol per dimension per point, an
event in each event dimension and an action in each action dimension (R47). The actions run in the world. The
next frame reports what was observed and what ran, and a symbol has happened when the report names it.

## 2.4 Reward

> **D35 — Reward.** A frame carries an array of rewards for actions already executed. Each has three fields:
>
> | Field | What it holds |
> |---|---|
> | `reward` | a signed number |
> | `channels` | the set of action channels it pays |
> | `frames` | the span of frames it pays over, counted back from the frame it arrives in |
>
> `channels` and `frames` are optional, and an omitted one means all of them.

A reward is an input, not a symbol: nothing fires for it and no function names it. Zero is neutral, a positive
reward is good and a negative one is bad. The numbers are the world's to choose. The only condition is that
they are consistent with one another, a larger reward always meaning a better outcome, since the machine only
ever compares them.

A reward is **scoped** when it names channels, a span, or both, and **unscoped** when it names neither. When the
world knows what it is paying for, it scopes the reward. Example: a stock channel, and the frames a position was
held. When the world does not know, the reward pays every action channel over the longest span any neuron is
active for. A scoped reward reaches only the base actions of the channels it names.

# 3. Machine

The machine is a processor with its own memory and a population of **neurons**, each a processor with a memory
of its own.

**The machine** carries what passes between the world and the neurons, holds the dictionary, and weaves the
connections between neurons. It runs two procedures once per frame, **recognition** and **inference**. Each
frame is one wave through the neurons. It rises through the frame as recognition, runs forward into the frame
ahead as inference, and comes down to the base as expansion, where one winner per dimension is output.

**The graph is the machine's own memory.** Nothing is written out and nothing is stored twice: the neurons hold
the relations they saw, the machine holds the dictionary and the board of the frame, and every cost in the
design is the number of edges it would take to state the graph of the run with the functions as they stand
(D12, D13). The cost is how the machine compares one way of covering the board with another, and it is never
computed as a whole (§3.12).

**It has two capacities, both the hardware's.** Its **precision** `P` is the range of one number: every id and
offset it holds is a number below `P`. Its **neuron capacity** `N` is the neurons it holds, fewer than `P`,
since each has an id.

## 3.1 Neurons

> **D2 — Neuron.** A neuron is a symbol, and a type: its activations are its instances (D8). It is identified
> by its id, fixed at allocation. The machine keeps the order neurons were allocated in, so an older neuron is
> one allocated earlier, and a decommissioned neuron allocated again is the youngest. It is a processor with a
> memory of its own, and that memory holds a history (D18), a relation index over it (D47) and connections
> (D25). It is a base neuron (D52) or an interneuron (D53).

Older neurons were allocated earlier. That is how ties are broken.

A neuron records what it sees and returns it when the machine calls it: `activate`, `connect` and `infer`
(§4.1). It knows nothing of functions, of the dictionary or of what costs.

The machine holds an index of its neurons by id. Every function, connection, activation and deletion that names
a neuron names it by id, and reaches it through this index.

## 3.2 Base neurons

> **D52 — Base neuron.** A symbol of the alphabet: an event or an action of one dimension of one channel (D1,
> D3). At declaration the machine maps each base symbol `(dimension, bucket)` to a base neuron, in declaration
> order, and keeps the map. The base neurons are the first neurons the machine holds.

Neuron dimensions belong to the base alphabet and to nothing above it.

The map is the machine's index of its base neurons by base symbol: a symbol a frame reports reaches the neuron
that fires through it (D8), and expansion ends on base symbols read back through it (R28).

## 3.3 Activations

> **D8 — Activation.** A neuron is activated (fires) only when something happens: an event neuron when its event
> is observed, an action neuron when its action runs, and an interneuron when a fit of its function is accepted
> (R24) or an inference names it (R36).
>
> **A neuron may fire many times in one frame**, and those activations are instances of one type.
>
> **At the base, at most one neuron fires per dimension per point of the layout in a frame.** The input reports
> one symbol per channel at each point, and a dimension with nothing to report there is silent. That bound is a
> property of the input, not a rule the machine imposes.
>
> **Above the base, several neurons can fire at one position.** Several accepted fits may land on one activation
> (D28), so several interneurons may stand at one position (D53), and one function fitting at several positions is
> one interneuron that fires at each (D15). The relation between two activations at one position is an
> association (D34).
>
> **An activation has a position and a pose.** A **pose** is a neuron in a **phase**. A base neuron has one
> phase, so its pose is the neuron, and a base activation stands where the input laid it. A call is one
> activation of the interneuron, at the position of the latest activation the fit landed on, in the phase the
> fit read (D37, D53). A phase is the node the fit started at, the node it ended at, and its **laps**, how many
> times it went round. A fit reads a cycle one way, the way its relations' offsets advance (D15), so the phase
> names no direction. The phase names its two nodes by their neurons: a node is a pose (D15), so the neuron and
> its phase say which node. So a phase is two nodes and a number. A ring of calls, each ending at the call
> before it, keeps the same names lap after lap. Everything the fit landed on has already happened (D54). A
> pose is a name: a relation to the activation names its pose (D34), and nothing in it costs (D13).
>
> | | is |
> |---|---|
> | phase | the node the fit started at, the node it ended at, and the laps |
> | pose | a neuron in a phase |
> | activation | a pose at a position |
> | relation | two poses and the offset between them (D34) |

> **D9 — Activation window.** An activation at position `f` in time has two spans. It **stands on the board**
> through `f + S`, where `S` is the longest reach in time among the neurons the machine holds, one beyond the
> farthest node any function reaches from its latest (D4, D24): for that long a fit may land on it, and an
> interneuron whose reach extends to it sees it. `S` moves with the dictionary, up when a function gains a node
> toward its past and down when the last function of that depth leaves. It is **open**, connecting, through
> `f + reach_t`, its own reach in time (D4): a neuron learns what followed only as far as it sees. Two things
> reach it while it is open, one frame at a time, by the connect call after the frame's last level (§7.1):
>
> | Input           | Description                                                                                                 | References |
> |-----------------|-------------------------------------------------------------------------------------------------------------|------------|
> | the apex        | What stands on the apex of that frame within reach (R27, D25), while the activation is uncovered; in the frame it is covered, the neuron that covered it (D10). | D25, R31   |
> | the rewards     | For the actions of that frame, and of earlier frames the reward spans, each at the distance its frame names. | R29, R33   |
>
> The machine holds the activation open, not the neuron. There is one open activation per neuron and position, and
> it holds: the activation, its position on the board, its age, and the age it was covered at, if it has been. An
> interneuron's activation holds besides what expansion reads (R28): the function that fired it, its phase, which
> of the function's relations did not come true, and the activations its fit landed on and bound, credited or
> not.
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since the frame of its position, `0` through `reach_t`; a new activation is the one at age 0, base and
> interneuron alike.
>
> **Covered at** is the age at which the activation was covered (D24), none until then. Where its only
> neighbors are in time it is never 0: it waits for what follows, and what follows has not fired.
> Where it has neighbors in its own frame it may be 0: a call inside the frame covers every activation it
> landed on (D24). Covering comes
> after the activation has fired (R27), and nothing undoes it: later calls may land on the activation too,
> and a later function may name the call that covered it (D10), but it never returns to the apex. So this
> one number says for every
> frame of the window whether the activation stood on the apex there: it did at every age before it, and at
> none from it on. Which of its relations are covered, and by whom, is the coverage set's (D24).
>
> **An open activation carries no commitment beyond that.** It infers once, at age 0, and the machine keeps
> what it inferred (R45); what it does for the rest of its life is connect to what ran, in the `connect` call.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what expansion
assumes for anything no node states.

> **D40 — Strength of an activation.** An activation is **strong** when the environment reported it, and
> **weak** when an inference named it (§8.5). That holds for events and actions alike: an event and an action
> the machine infers are both weak, and the environment's report replaces both. The neuron is the same either
> way: a weak activation is an activation of the neuron the environment would have reported.
>
> **A weak activation is an input of its frame like a strong one.** It is in the neighborhood of everything that
> fires within reach of it, at whatever offset, so it enters histories, and functions name it and price it like
> any neighbor; its neuron is called with it, it may be covered, and it may stand on the apex and vote once
> (R46), with the connections its neuron's strong activations strengthened. **It differs in one thing: it lasts
> its one frame.** It is not open afterward: it writes no connection, nothing connects to it, and no reward
> reaches it (R31). **It covers nothing** (R40). **The report replaces it**: where a strong activation of the
> same neuron stands at the same position, the weak one is dropped. Where nothing reports, the inference peeks
> through: an inferred action nothing reports as run, and an inferred event nothing reports as seen, are what
> the machine had in that frame.
>
> **An interneuron is as strong as what it covers.** An accepted function's interneuron is strong when any
> activation it covers or binds is strong, and weak when every one of them is: a situation made entirely of
> inferred inputs is inferred.

> **D37 — Call.** One activation of a call neuron (D48), in a phase (D8), together with its **arguments**: the
> activations bound in its function's variables on that occasion (D10, D38). It is made in one of two ways and
> is the same thing either way. **Recognized**: the election accepts a fit (R24), and the call neuron fires once,
> at the position of the latest activation the fit landed on, at age 0, in the phase the fit read, as strong as what
> it covers (D40, D53); among several at one frame the latest is the one at the least position in the order the
> dimensions are declared, as the origin is (D15), so a row's call stands at its least `x`. It costs nothing of
> its own (D13);
> its arguments are what stood in the variables. **Inferred**: a
> connection names the call neuron in a phase at an offset, and an expansion reads the call there, at its
> completion, weakly (R36), its nodes at the frames back from it; the part that lies ahead is what it proposes,
> for the report to make strong what it inferred (R30), and its arguments are bound from what has happened where
> a variable's node has happened, and expanded where it has not (D44, R28). A call carries its pose and nothing
> else: what it is stands in its function, and what its arguments are stands on the apex beside it. What it
> records is what expansion needs (D9).

**A function is recognized or inferred, and both kinds of member take part in both.** Recognized, a function
says that its events stood and its actions ran. Inferred, it is expanded (R28): its actions stand inferred as the
machine's output and its events beside them, both weakly, and the report makes strong what ran and what was
seen. What follows any activation is held on its neuron as connections (D25), one kind, and an inference is a
connection read.

## 3.4 Spacetime

> **D4 — Reach.** How far an activation sees, in each activation dimension (D1), and in all of them at once. A
> base activation sees one position: the positions within one of it in every dimension of space, in this frame and
> the frame before. It sees what stands on the board there (D24): an activation that is not covered, as itself.
> At a covered position it sees each call that landed on the activation there, and it sees nothing through a
> call. A call sees the calls that landed on an activation it landed on (D26). It also sees what the relations
> it took from the activations it covered end at (D10): one position beyond the farthest of everything it
> landed on, at any depth (D53, R28), and no further. The call's own
> position says nothing about how far it sees. In time its position is the latest activation it
> landed on, so it sees back to the earliest frame of anything it stands for, and one frame more. Where what
> it stands for comes back to an interneuron already expanded, it stops there (D36), and so does its reach. How
> far an activation sees is one thing, and where its neighbors stand is another. A call keeps every neighbor
> of what it covered (D10), and some of those stand farther away than the call sees. That is fine: reach only
> says what an activation sees when it fires.
>
> In space the reach runs either way; in time it runs back, and the same number runs forward as the span over
> which the activation's row takes what followed it and its neuron connects (D9, D18). `reach_t` is the reach
> in time.

> **D6 — Offsets.** The offset between two activations is the difference of their positions on the board, one
> component per activation dimension they share (D1). Every offset is exact, in a neighborhood (D18), in a
> function (D15), in a connection (D25) and where an expansion reads a node (R28). A neighbor may differ from
> the activation in several dimensions at once (D5), so a relation's offset is the full vector, one component
> per dimension, the same shape as a connection's (D25), and it is one number (D13). A base activation holds its
> neighbors at components of `0` or `−1` in time and `−1`, `0` or `+1` in each dimension of space. An
> interneuron holds them out to its reach.

> **D5 — Adjacency.** An activation `b` is adjacent to an activation `a` when the following conditions hold:
>
> - `b` is in the same frame as `a` or in an earlier one.
> - `b` is within `a`'s reach in every activation dimension they share (D4), and is not `a` itself. Two
>   activations at one position are neighbors at offset zero: base neurons of different channels, and
>   calls standing at one position (D53, D10). That relation is an association (D34).
> - `b` stands on the board: it is not covered (D24). An activation a call in time has landed on is a neighbor as
>   itself until it is covered. Where its only neighbors are in time, it waits for what follows
>   it, so the activation just before `a` in time is a neighbor as itself. A covered activation is no
>   neighbor: each call that landed on it is, at its own position. Two calls are neighbors where they landed
>   on one activation, or where a relation no call covers runs from a covered activation under one to a
>   covered activation under the other, and not otherwise (D26). A call takes every relation of a covered
>   activation it landed on that its fit did not cover (D53, D54, D10).
> - Where `a` is a call, `b` is nothing its fit landed on, nor anything under what it landed on, at any depth:
>   a call never sees what it stands for (D24, R28).
> - The neighborhood policy allows `b`'s kind beside `a`'s (D50).
>
> For example, in an image channel a pixel's neighbors are the eight pixels around it, and the nine at those
> positions and its own in the frame before. A pixel one step away in both `x` and `y` is one neighbor at one
> offset (D6). In space adjacency runs both ways. In time it runs only backward: what fires after `a` is never
> adjacent to it, and is held as a connection instead (D25).

Events and actions are compressed in the same functions: above the base a function may name what was seen and
what was done together.

> **D50 — Neighborhood policy.** Declared once for the machine, as the alphabet is (D1). For it a neuron is of
> one of three kinds: an **event** or an **action**, a base neuron of that kind of dimension (D1), or an
> **interneuron**, whatever its function names (D2). The policy is two tables of booleans, the kind of the
> holder by the kind of the target, nine entries each:
>
> | Table  | An entry on says                                                          | And so decides            |
> |--------|---------------------------------------------------------------------------|---------------------------|
> | past   | a target of that kind may stand in a holder of that kind's neighborhood (D5) | what functions are built from |
> | future | a holder of that kind may hold a connection to a target of that kind (D25)   | what inference reads      |
>
> With every entry on, anything may be a neighbor of anything and anything may connect to anything. An entry
> off removes that kind of target from that kind of holder's neighborhoods or connections, and nothing else:
> what stands on the apex infers in every setting (D49), and the policy restricts only what it reaches.

> **D26 — Neighbor.** **A neighbor is a pose at an offset**: the pose of an adjacent activation (D5, D8), at
> its offset (D6). At temporal offset 0 a neighbor co-occurs and at
> negative offsets it led here; **both are the same kind of thing**, and a neighbor in space, or in space and
> time at once, is one more of the same.
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other at a
> nonzero spatial offset, and in time a neuron stands beside its own earlier activation. Offset zero in every
> component is where the activation itself sits, and an activation is not its own neighbor there. Above the base
> two calls that landed on one activation are each other's neighbors, at the offset between their positions,
> and the relation between them is a relation like any other (D34). So are two calls where a
> relation no call covers ran from a covered activation under one to a covered activation under the other:
> that relation is lifted, and a relation between the two calls stands there instead. However many are lifted
> between the same two calls, one relation stands there for all of them (D55, D10). Two calls are neighbors in
> no other way.

> **D7 — Neighborhood.** The set of neighbors (D26) one activation observes is its **neighborhood**. In a text
> stream, with time its one dimension, the neighborhood of `s` in `p a r i s` is `i` at −1, and `p a r` are out
> of reach (D5). In an image stream, with time, `x` and `y`, a
> neighborhood holds the eight pixels around the
> activation and the nine at those positions and its own in the frame before.
>
> **A neighborhood is the frame around the fired neuron**, what stood before it and beside it, since adjacency
> admits nothing later (D5). The row that records it takes what followed afterward, frame by frame, so that in
> the history the frame around an activation is whole on both sides in time, as it is in space (D18). The
> activation itself is in neither: its own node is what a function covering it replaces, and is paid there
> (D22).
>
> **An activation bound in a variable is a neighbor like any other** (D10): the neuron that stood, at its own
> position. Which variable it stood in is the call's business, not the neighbor's.

> **D34 — Relation.** Two poses and the offset between them, one component per activation dimension the two
> share (D1, D6): the same shape as a connection (D25). A pose is a neuron in a phase (D8). There is one kind of relation in the
> design: what a neuron returns is a relation, what a function holds is a relation, and they are the same
> object. Two nodes that differ in several dimensions are in one relation, with a component in each. In a
> neuron's relation index, one end is the neuron's own activation and the other a neighbor. In a function, both
> ends are its nodes, and a relation from a pose to itself joins one node to itself. It is one relation, read
> in the one direction its offset gives: the row that holds the same neighbor at the opposite offset is the
> other end's reading of the same relation, not a second one (D47, R50).
>
> **By its offset a relation is of one of two kinds: in time, or inside the frame.** It is **in time** where
> its two ends are in different frames. It is **inside the frame** where its two ends are in one frame. Inside
> the frame it is in space where its two ends are at different positions. It is an
> **association** where every component of its offset is zero: its two ends stand at one position in one
> frame, as base neurons of different channels do, and calls standing at one position (D5). An association
> has no direction and no distance. It says only that the two were there together. A relation in space and an
> association are one kind. The two kinds are the same object and cost the same (D13).
>
> **The two kinds are held apart, for one reason.** A frame is complete when it fires, and time is not: what
> stands with an activation is all there, and what follows it is still to come. So the two kinds are counted
> apart where an activation is covered, and the first covered relation of an activation decides which kind
> covers it (D24). A function holds one kind (D15). The frame runs up all its levels before time runs at any (R26).
>
> **A call's phase is part of its name.** A relation to an interneuron names it in the phase its activation
> fired in (D8, D53): which node the fit ran from and to, each by its neuron, and how many laps. A base neuron
> has one phase, so at the base a relation is two neurons and an offset. Two calls of one interneuron in
> different phases are two poses. They are different ends for a relation, as two neurons are, and two in one
> phase at one offset are one relation. A pose is a name, and costs nothing (D13).
>
> **A relation has one form, and in time it runs backward.** Where the offset has a component in time, the
> relation runs from the later neuron to the earlier one, at a negative offset there: that is how the neuron
> that fired saw it, what stood before it. The earlier neuron sees it afterward, when its row takes what
> followed (D18), and from then on holds it as the later neuron does. Where the component in time is zero, the
> two neurons are ordered so that the first nonzero component of the offset, in the order the dimensions are
> declared, is positive, and where every component is zero, the older neuron first (D2). Either way each end
> counts the relation in its own index (D47), the machine reads both as the one relation, and takes the larger
> of the two strengths, since each neuron's history is a window that can only miss occasions (D18, R21).
>
> **Relations compose.** Given the offset from `a` to `b` and from `b` to `c`, the offset from `a` to `c`
> follows, component by component. Where `a` and `c` are neighbors, the relation from `a` to `c` is still a
> relation of its own: the neurons return it, and a fit may land on it. It is refused where calls already join
> `a` and `c`, since it would close a deadlock (D24, D55).

> **D11 — Identity.** A neuron's identity is its type alone, a base symbol or an interneuron's id (D2), so
> nothing about where it occurred is part of it. **The same shape at two positions is two activations of one
> neuron**, and they pool: both carry the same relations, so the same functions cover them and one function
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

## 3.5 The graph

> **D55 — Graph.** What a run leaves: a set of **nodes**, each an activation of a neuron (D8), and a set of
> **edges**, each a relation between two nodes (D34). A node carries its pose, and nothing
> else. An edge carries an offset, one component per dimension (D6). Two nodes are joined by at most one edge. A
> node's position in the world is not in the graph: it is wherever the edges put it relative to the nodes it is
> joined to, and two nodes joined by no path stand in no particular relation.
>
> **The graph is what every neuron saw.** An edge exists between two nodes where one was in the other's
> neighborhood when it fired (D7), so the graph has no edge longer than a reach (D4). Two activations farther
> apart than any neuron sees are joined by no edge, and the graph does not say where one is relative to the
> other. That is right, since no neuron could ever have read it.
>
> **A piece of the graph is stated as nodes and edges, nothing more.** A piece, in the dictionary or on the
> board, is its nodes and its edges, each between two of those nodes, with an offset. Two parts joined by no
> edge need nothing between them. The cost of a piece is its edges, one each: a node, base or call, is in the
> graph through its edges and costs nothing of its own (D13).
>
> **A call covers relations, and inside the frame the nodes it landed on.** On the board a
> call covers the relations its fit landed on, and they cost nothing more. Inside the frame it covers every
> node it landed on as well, and they leave the board (D24). Expansion puts the covered relations back (R28).
>
> ```
>   ( a ) ──── ( b ) ──── ( c ) ──── ( d )      without the call
>
>   ( a ) ──────── ( bc ) ──────── ( d )        the call bc covers b–c, and b and c with it
> ```
>
> | | edges on the board | count |
> |---|---|---|
> | without the call | `a–b`, `b–c`, `c–d` | 3 |
> | with the call `bc` | `a–bc`, `bc–d` | 2 |
>
> The dictionary holds `b–c` as a function, and its call pays for that relation. `b` and `c` are covered as
> the call is taken. No call covers `a–b`, so it is lifted: `a–bc` stands there instead. `c–d` is lifted the
> same way. Nothing is left on the board beside the call that the call landed on.
>
> **In time a node stays until it is covered on both sides.** A node a call in time landed on stays in the
> graph until one relation of its own to an earlier frame and one to a later frame are covered (D24). Until
> then an edge to it ends at the node itself, never at the call. Where `a`, `b`, `c` and `d` follow one
> another in time, `b` is still on the board after the call `bc`: its relation to `a` is not covered. Where
> `a–b` recurs and becomes a function, its call covers `a–b`. That relation and `b–c` have `b` in common, so
> the two calls are neighbors, one level up (D26), and `b` has both its relations covered and leaves the
> board.
>
> **Lifting.** When a node is covered, each relation of its own that no call covers leaves the board. A
> relation to the call that landed on the node stands there instead. It has the same other end, and the call's
> pose where the covered node's pose was (D8). The relation is **lifted** to the call. Inside the frame, the
> row at its other end takes the call (D18). The other end is what stands there on the board: an activation
> no call has landed on, as itself, or, where that end was covered first, the call the relation already stood
> lifted to there.
>
> **A lifted relation is between what stands.** Where the call that landed on a covered end is itself covered,
> the relation goes on up, to the call that stands over it: from the covered activation, call above call, to
> each call that no call covers. This holds at both ends, and whichever end is covered first. So a relation is
> never lifted to a call that has left the board, and no relation stands between a covered call and anything.
>
> ```
>   ( p ) ──── ( q ) ──── ( r )      the call A covers p–q and q–r, so q is covered
>                │
>   ( s ) ──── ( t ) ──── ( u )      the call B covers s–t and t–u, so t is covered
> ```
>
> No call covers `q–t`. Once `q` is covered, `q–t` leaves the board and `A–t` stands there instead. Once `t`
> is covered too, `A–t` leaves and `A–B` stands there instead, one level up (D26). Whichever end is covered
> first, what stands in the end is `A–B`. Where `q` is covered first, `t`'s row takes `A`, and then `A`'s
> row takes `B`. Where `t` is covered first, `q`'s row takes `B`, and then `B`'s row takes `A`: the row that
> takes a call is always the row of what stood at the other end when the relation was lifted, never the row
> of an activation a call had already covered there (D18). Two nodes are joined by at most one edge. So however many relations
> are lifted between `A` and `B`, one relation stands there for all of them. A refused relation is never
> lifted (D24).
>
> **A lifted relation is a relation like any other.** It is two poses and the offset between their positions
> (D34). It is not the relation that left the board. That one named the covered node's pose, and this one
> names the call's. A function fits only relations that stand on the board (D31). So a function built with the
> relation that left has nothing to fit there, and it gets no call. A function built with the relation that
> stands instead fits there. This is the same at every level.
>
> **Every edge costs one.** No neuron sees a path, only its own neighbors (D4). So an edge between two nodes is
> paid whether or not a longer path of other edges already fixes the offset between them.
>
> **A deadlock is a chain of calls that closes.** Each call shares a node with the next, and the last shares
> one with the first. It is of any length, and no function holds it: it is made of the calls of different
> fits. `a–b`, `b–c` and `c–a`, each covered by its own call, is the shortest. Left alone it never ends. One
> level up the calls are neighbors round the same chain, as many calls cover them, and so at every level
> after (D26). A deadlock is not a cycle. A cycle is one function whose own relations lead back to where they
> started, one call covers it, and it is kept (D15). A deadlock is never allowed to close: the edge that
> would close it is **refused**. No call covers a refused edge, it costs nothing, and it is never lifted
> (D24).

**Everything in the design is this one object.** What a neuron returns is the edges at one node, with how often
each has held (R21). A function is a graph over neurons that recurs (D15). The dictionary is those graphs, held
once each. The board is the graph of the frame as it stands, with the calls standing for the pieces they cover
(D12). Covering is substituting a name for a piece. Expansion is putting the piece back (R28).

## 3.6 Functions

> **D15 — Function.** A graph over neurons that recurs: a set of relations (D34), each joining two **nodes**,
> with **one node per pose**: a node is a pose (D8), or an open node, a variable (D38), and a
> relation names its two nodes by naming its two poses (D34). A base neuron has one phase, so
> at the base a node is a neuron. An interneuron in two phases is two nodes, so one function may hold it twice.
> In one phase it is one node, so a cycle closes on it. A function is **connected**: every node is reached from every other along
> its relations. It is the center the machine keeps for the relations it has grouped (R50): it exists from its
> first relation, with or without an interneuron, and has one interneuron once a fit of it has been accepted
> (R16, D48).
>
> **Its origin** is its latest node, listed first in the dictionary (D56): the node later than every other in
> time, and among several at one frame, the one at the least position in the order the dimensions are declared
> (D1). The call fires at the position of the latest activation the fit landed on, which is the origin's where
> the fit reached it (D37, D53), and expansion runs from the node the phase starts at (R28). **Nothing joins a
> function later than its origin.** A function is recognized through to its origin in the frame its origin
> fires, and its call stands from then on in that phase, so what fires later sees the call, and its relation
> is a function over the call (R50).
> The origin is no more than the latest node the function has: where the relation at the origin leaves, the
> latest of the nodes that remain is the origin (R54). A fit enters the function at a node that stands in the
> present frame (D54). In a function in time that is any node that fires in the frame: the nodes later than
> it are still to come, and the fit leaves them out. A function
> that is a cycle has no fixed origin, since the last lap may stop at any node of it; its
> call's phase says which node the fit started at and which it ended at (D37).
>
> **A function is listed from its origin.** Node after node, each relation leads from the node before to the
> next. A node is listed once. An open node is listed as a neuron is (D38).
>
> **A function holds one kind of relation.** Its relations are all in time, or all inside the frame (D34). A
> relation of the other kind does not join it (R50). Where a pattern inside the frame and a pattern in time
> recur together, the level above relates their calls.
>
> **A cycle is a function of its own.** In time, the moment a node comes again, the relation that reached it
> closes the cycle. The frame has passed, so what comes round is the pose, and any relation back to it closes.
> Inside the frame, a relation back to a pose the function has closes the cycle only where it reaches the
> activation the fit landed on at that node. There its offset is the one the function's relations compose to
> between its two nodes (D34). At any other offset it reaches another activation of that pose, and it closes
> nothing. A relation of a pose to itself closes one alone, in time and inside the frame. A relation between
> one neuron in two phases does not: it joins two nodes (D8). A cycle advances by one offset, the one its
> relations compose to round a lap, in whatever direction that runs, time included. A function that is a cycle
> holds the relations of the cycle and no other. No relation joins it, and a relation closes a cycle only
> where every relation of the function is then on the cycle (R50). So a function is a cycle and nothing else,
> or it has no cycle. `y` after `x` and `x` after `y` is `… x y x y` for as long as it runs, entered at either
> letter. `a` after `P` and `P` after `a` is `P a P a …` for as long as it holds, `P` a parameter (D44). What
> comes after a cycle sees its call, and the relation to the call is a function at the level above. A cycle
> is recognized wherever its relations hold, in the frame the latest of them fires, since nothing else has to
> hold with it (D54). How a function is read against the board is the fit's (D54).
>
> **A function is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a function is what the machine holds. A neighborhood is a fact, a function a
> claim. What followed is in neither, and is held as connections (D25).
>
> **A function is a procedure.** Read as code, its fixed nodes are instructions, each a base symbol or a call at a
> position; its variables are assignments, a class a variable of a declared type and a parameter one of any type,
> each bound where the fit first reads it and tested where it reads it again (D41, D44); its cycle is a goto to a
> line already read, run while the relation that closes it lands (D54); and its origin is the last line, which the
> parse starts from once it has run. It returns nothing: what it does is what its expansion reads
> (R28), and what comes after it is not in it but in the connections from its call (D25). The language holds
> straight-line code, or one loop and nothing else, and one assignment per variable before its reads. A loop
> with code after it, two loops, a branch, or a variable assigned twice have no single function: the machine
> holds them as procedures over each other's calls.
>
> **In the dictionary it is** its nodes, each named once, and its relations, as D55 states a piece of the graph
> and as the dictionary stores it (D56). What it costs there is that statement's (D13).

**A function's body is its relations, and the nodes they name.** Some nodes are fixed, a particular neuron;
some are **variables**, nodes where the function is general. A variable is a hole in the graph: it names the
position and leaves open what stands in it. What stands in a variable on one occasion is the variable's **value**,
and the values of a call's variables are its arguments (D37). A variable exists only inside a function, is
never fit alone, and promotes no interneuron: what stands in it stays on the apex, bound (D10).

> **D38 — Variables.** A variable is an open node: a node of a function that leaves open which neuron stands at
> its position. The machine makes variables, never a neuron (R52): a variable says what is general across several
> functions, which no one activation shows. It is of one of two types, closed in opposite ways:
>
> | Variable | In the dictionary, once | In the call, each time | Closed on |
> |---|---|---|---|
> | a class (D41) | which class, and the class's members | the member that stood, one number | its members: a neuron that is not a member does not fit, and any function may use the class at any node |
> | a parameter (D44) | the node, open | the neuron that stood, one number | nothing: any neuron fits, seen before or not |
>
> **A variable is one node, like any node.** Where the same thing has to stand at two positions, the function says
> so the way it says anything twice: the node is reached again, by a cycle (D15). So a parameter is listed at
> most twice, the second time as the node that closes the cycle, and the fit reads it as many times as the
> cycle runs: `a` after `P` and `P` after `a` is `P a P a …` for as long as it holds. Nothing ties positions but
> the graph.
>
> **A function holds nothing but its nodes and relations**: no branch, no loop but a cycle. What branches is
> which function fires; what loops is the frame; what a variable held is on the apex, bound beside the call
> neuron. On every occasion it is priced as part of the call (D12, D22).

> **D41 — Class.** A set of neurons the dictionary holds, its **members**, and nothing else: **one of these
> stands here**. In the dictionary it costs nothing (D13). A function has it at a node as a variable
> (D38); several functions may have one class, and a function may have one class at several nodes, a member
> bound at each. Members stand for one another: two neurons that have stood together where the class is
> used are never both members.
>
> **Its value**, at a node, is the member that stood there, which stays on the apex, bound (D10). Where no
> member stands at a node the class is at, that node's relations did not come true (D22), and the variable does
> not hold there.
>
> **A class is not a neuron.** It is made by generalizing functions that differ at that node (R52). A neuron
> joins it when it stands at the node in an occurrence where the rest of a function using the class holds, and
> leaves when that no longer recurs in the histories (R18). So what is
> learned about a member at one position holds at every other. A class is deleted when no function has it, or when
> it is left with one member, which is an ordinary fixed node.

> **D44 — Parameter.** An open node of one function that any neuron may fill: **whatever stands here**. Beside
> it the function keeps its **members**, the neurons that have stood there within the window. The list is a
> record and not a condition. It says nothing about what may stand, and it prices nothing (D13).
>
> **Its value** is the neuron that stood at the node. It stays on the apex, bound (D10). Where the node is
> reached again by a cycle and holds something else, the value is the neuron that stood the first time, and the
> relation that reached it again did not come true (D22).
>
> **It is open on its members and closed on where.** Any neuron fits, seen before or not: a thing that comes for
> the first time is the parameter's value the first time, and is a member from then on. It is made by
> generalizing functions that differ at that node (R52).
>
> **Bound from what has happened, expanded at what has not.** When an inferred call is expanded with no argument
> for a parameter (R28), and the node is reached once in frames that have happened and again in frames to come,
> it takes its value from what stood the first time and expands to it the second. That is how a value is carried
> forward in time, and it is the whole of copying.

## 3.7 Interneurons

> **D53 — Interneuron.** The neuron the machine allocates for a function, its **call neuron** (D48). It is
> allocated the first time a fit of the function is accepted and a neuron is free (R16). It has an id and no
> dimension and no channel. **It fires once per accepted fit, in a phase** (D37): at the position of the latest
> activation the fit landed on, in the frame the fit is accepted, at age 0. The phase is read off the fit: the
> node the fit started at, the node it ended at, each by its neuron (D8), and how many laps it went round
> (D54). The call stands for
> everything the fit landed on, the relations of it that no call covers are lifted to it (D55, D10), and it
> sees what it saw (D4). Where an arriving activation reads a relation that is the call's own function going round its cycle,
> the call goes on: it fires at the new position, the same call, its phase that much longer, so a stretch is
> one call however long it runs (D54, R50). The call costs nothing of its own:
> its failures and its bindings are its cost (D13).

An interneuron is not an event or an action, though what it stands for may be either. Its phase is read off the
fit, so that other neurons hold it as a neighbor in that phase and expansion reads it. Nothing in the pricing
reads where it stands.

A neuron is never made or destroyed. It is **allocated** when an interneuron is needed, and **decommissioned**
when its last function leaves the dictionary, to be allocated again. When none of the machine's `N` neurons is
free, no interneuron is granted (R16, R24).

> **D56 — Dictionary.** What the machine holds about functions, field by field. Three kinds of thing are in it,
> and every cost in the design is read over these fields (D13).
>
> | Thing | Field | What it holds | Costs |
> |---|---|---|---|
> | a function | relations | its relations, each a neuron relation as the neurons return it: two poses and an offset (D34), and which of its two ends it attached to the function at, where its worth is read, none for a relation of a cycle (D57); together they name its nodes, one per pose | one per relation; nothing per node or attachment |
> | | entry | that the function is in the dictionary | one |
> | | origin | which of its nodes is the origin, its latest (D15); none where the function is a cycle | nothing: the node list begins with it |
> | | open nodes | which of its nodes are variables, each a class by id or a parameter (D38) | nothing |
> | | interneuron | the neuron allocated for it, or none (D53) | nothing: the function is reached through it, not stated by it |
> | | members | per parameter, the neurons that have stood there within the window, a record (D44) | nothing |
> | a class | members | its member neurons, by id (D41) | nothing |
> | an interneuron | functions | the functions it fires for, one or more where a stretch came round (R43); and nothing of any earlier form (R54) | nothing beyond the functions' own |
>
> **Two indexes are kept over it and rebuilt from it**, so they cost nothing: by relation, each neuron and
> offset with the other end open or not, to the functions holding it, for finding (R53); and by what an
> interneuron stands for, its expansion to base symbols, for reuse (R43) and for generalizing (R52).
>
> So a function of `m` relations costs `m + 1`, whatever its nodes, and a class and a parameter cost nothing.
> Nothing here is the implementation's: these are the fields, and the costs are read over them.

> **D36 — Interneuron index.** The dictionary (D56) reached three ways:
>
> | By | Reaches |
> |---|---|
> | a function | its interneuron, and back, for expansion (R28) |
> | a relation (D34), or one of its neurons and its offset with the other end open | every function that holds it; or every function that holds a relation with that neuron at that end and that offset and a variable at the other, a class or a parameter (D38) |
> | what an interneuron stands for | the interneuron, for reuse when a stretch comes round (R43), and the functions of the same shape, for generalizing (R52) |
>
> **The same relations are the same function.** A function is its set of relations, so the same neurons at the
> same offsets are one function whichever neurons returned them. Each function has one interneuron. An
> interneuron may hold more than one function where a stretch comes round (R43): it fires for each, and an
> activation of it expands to what its fit landed on (R28).
>
> **A function names other interneurons by id**, and through them it may come back to its own: a function whose
> node is the interneuron that fired a frame before, whose function names the one before that, and so on, can
> reach its own interneuron again (§6.7). **What an interneuron stands for** is its expansion (R28): its
> function's nodes with the variables left open, each node in turn for what it stands for, down to base symbols.
> Where that comes back to an interneuron already expanded, it stops there.
>
> **The index is searched by what an interneuron would stand for** (R43): a function's expansion, read from its
> origin. The search finds an interneuron when three things hold: reading that expansion, it meets the
> interneuron no further back than the interneuron's own length, that far included; over exactly that length
> the two expansions are the same, a cycle's length being one lap; and the meeting is not at the origin. That
> is how something that comes round again finds the interneuron it had the first time.

**A function and its interneuron are held together.** The machine holds both in its dictionary. No neuron holds
a function or points at an interneuron: a neuron records what it sees and returns it (§5), and the machine
groups, fits and keeps (§6).

> **D48 — Call neuron.** A function's interneuron: the neuron the machine allocates for a function the first
> time a fit of it is accepted (R16). It fires when a fit of its function is accepted, once, in the phase the
> fit read (D53), covering the function's fixed nodes the fit landed on (§6.4); what stands in the function's
> variables is bound beside it, on the apex (D10). It says which function ran, and its phase says how. Its
> connections are one set per phase it has fired in, and for every value its variables have taken, so a
> function above learns about it regardless of its arguments by naming the call neuron in a phase alone, and
> about one argument by naming the call neuron and the activation bound in that variable. It holds a history,
> a relation index and connections of its own, like any neuron, and an activation of it expands to what its
> fit landed on and bound (R28).

## 3.8 Apex

> **D49 — Apex.** The activations of a frame that no accepted function covers, base neurons and interneurons
> alike: a base neuron no function found worth naming stands in it beside an interneuron many functions deep.
> It is the machine's whole state at that frame, and it has four names for the four things it is. It is the
> **frontier**, looking back: everything underneath it is recovered by expanding it, so it is what the cost
> is read over (D12). It is the **situation**: what every neuron sees as its neighborhood (D5), what connects to what
> ran, and what votes, less what is inhibited (D24). It is the **wavefront** of recognition: where the frame's levels stopped (R26), and from
> where the wave runs forward, each activation of it reading its own connections ahead. And it is the apex, the
> top of the hierarchy as it stands. The four are one set: an activation that is covered leaves every
> role at once (D10, D24). One role ends sooner. An activation a call has landed on is inhibited: it still
> stands on the apex, and it does not vote while a call over it speaks (D24).
>
> **An expected input never reaches the apex.** An activation a function in the dictionary names at its latest
> node has its relations covered by that function's call, recognized in the frame it fires, and is covered once what
> follows it has fired (D24). Where the function is a
> cycle, the next lap is the same call, its phase that much longer (D53);
> where a function at the level above names the call, its call covers it (D10). So where a function keeps coming true
> the apex holds one call, the latest, and what arrives is covered as it arrives.
>
> **Partway through the frame it is the board.** While the levels run, the apex as it stands at a level is the
> board the next level is fit and elected on (§6). The apex of the frame is the board once the last level has
> run.

> **D10 — Inhibition.** A call that lands on an activation as a fixed node does three things to it. The
> second, that it is not heard, is from the frame the call lands. The other two are from the frame it is
> covered in (D24). It takes it out of the cost: the relations of it the fit landed on are covered,
> paid through the call, and every other relation of it that no call covers is lifted to the call, which
> stands for it, at the offset from the call's position (D55, D12, D5). Those are the call's neighbors from
> then on. Every row that saw the
> covered activation as on the board, in the history of each neuron outside the call, keeps it as it saw it.
> The row takes no call for it: a row holds what stood on the board when it was written, and no relation to a
> call that landed on a neighbor afterward is counted (D18, D47). Nothing in a history is ever changed to a call. An activation
> that fires later sees the calls that landed on it, and not the covered activation (D5). The rows of the activations the call stands for take nothing: what they saw of the call's other parts
> are the function's own relations, read from the rows that saw them (D57). Where several calls land on one
> activation, each covers relations of
> its own and the rest are lifted to each; a lifted relation is paid once, between positions, however many
> calls hold it; the calls are neighbors, at their two poses, and the relation between them is on the board at the level above
> (D26). It is not heard, from the frame the call lands on it, covered or not: it is inhibited, and
> resolution reads the call's standing inferences, not its own (D24, R46). And it stops connecting: from the frame after it is
> covered it writes nothing about what follows, though its row still takes what followed, since the history
> is what happened and not what was heard (D18). Every call that landed on it expands to it (R28); nothing
> lands on a relation a call covers (D28).
>
> **A call is inhibited where a function names it**, as any activation is.
>
> **The last connection.** In the frame it is covered, the activation connects once more, to the neuron that
> covered it, the function's interneuron, and to each where two cover it. The connection is at the offset
> from the activation to its coverer (D6), in time and in every other activation dimension, as every
> connection is (D25), and it takes that frame's
> reward (R31). Then the activation writes nothing more. Where a
> later function names its coverer (D53), the
> coverer connects once more the same way, to that function's call, and the activation under it stays silent. Without
> this connection the activation could
> never learn the neuron built over it: the interneuron did not exist in the frames it was uncovered, and it is
> inhibited from the frame the interneuron first fires. So a situation beside which a function has named what
> followed it learns that one neuron, the interneuron, and from then on infers it; the interneuron learns what
> follows.
>
> **The latest of a sequence is not covered.** A call recognized at the present covers relations of the
> activation at its position (D54). Where that activation's only neighbors are in time, it waits for
> what follows it, still to come (D24). So it stands on the apex in the frame it fires, beside the call
> that landed on it, and what fires next reads it as itself. It is inhibited all the same: while the call speaks,
> it does not (D24). An activation with neighbors in its own frame is covered in the frame it fires
> where a call inside the frame lands on it. What fires next then reads the call, and not the activation (D5). Only a recognized call covers. What an inference named covers nothing (R40).
>
> **What stands before coverage stands.** A neuron is covered only after it has fired (R27). Coverage acquired at
> a later age revokes nothing before it: the connections the activation strengthened while uncovered stand, and
> a reward for one of those frames still reaches them (R33).
>
> **One exception, on being heard alone.** Every activation infers at its age 0, inhibited or not, and the
> machine keeps what it inferred (R45). An inhibited activation's inferences are heard in a neuron
> dimension only where no call over it has anything for the frame ahead in that dimension (D24, R48), as a newly allocated interneuron has nothing until something has run under
> it. A covered activation still writes nothing and connects to nothing, heard or not; only its coverer learns
> what ran.
> That is the whole of inhibition in the design.
>
> **Bound is half covered.** An activation standing in a variable of an accepted function (D38) is **bound**.
> The call covers the relations to it: the function's relations are in the dictionary once. Its binding is
> one edge, paid with the call (D12, D13). It stays on the apex, it is a neighbor, it is heard, and it
> connects to what follows. A call hides what is the same on every occasion, its fixed nodes, and leaves visible
> what varies.
>
> **Above its level, a bound activation is a neighbor.** Its neuron is called once, at age 0 (R1). At the
> levels above, the interneurons that fire see it as a neighbor, and their relations to it are the ones it takes
> part in. A relation to it the call that bound it did not cover stands at its cost, so a function above can
> cover it. One the call covered costs nothing, so a relation above that lands on it holds a node and covers
> nothing (D22). Its neuron is still paid, as the value in the call that bound it, so a function above that
> names that call and this neuron fixes the value, and saves it (D24).

## 3.9 Costs

> **D12 — Cost of the graph.** What it takes to state the graph of the run as it stands on the apex, with the
> accepted calls standing for what they cover: every edge between activations on the apex, the calls among
> them. Nothing joins a node to the graph but its edges, and a node costs nothing of its own, base or call: a
> base node its neuron related to nothing has no part in the cost, and neither has a call with no failure and
> no binding.
>
> | On the apex | Costs |
> |---|---|
> | a bare activation | nothing of its own |
> | a relation between two activations on the apex | its offset, once per pair, and nothing where it is refused (D24) |
> | a call | nothing for the activation; each relation of its function that did not come true, one each; and each of its arguments, the edge that binds the activation to it, one each |
> | an activation bound in a variable | nothing of its own: its binding is paid with the call, and the relations the function fixes are the call's |
>
> **What a call takes out of the board** is the relations its true relations landed on, which its function
> holds once in the dictionary, and the nodes they joined, which it stands for. A node at a variable
> stays on the apex, bound, and the edge that binds it is paid with the call. Every other edge of a node the
> call stands for stands as it was, from the call: it joins the call to the rest of the graph, and where two
> calls stand for one node it is paid once (D10). The call adds nothing of its own.

**A call pays its differences and nothing else.** The function's relations are in the dictionary. Each call
pays for which of its relations did not come true and for what it binds. What the relations fix is not paid
again: the offsets among the nodes they join.

**A failure is a relation that did not come true.** It is paid with the call, one edge. Whatever stood
where the relation said is on the apex, uncovered, with its own relations. So a wrong neuron costs the failure
and whatever joins it to the rest, and nothing there costs the relation alone.

**Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
cost exactly as an observed event does, compressed by the same functions (D5).

**It holds nothing about the future.** A function is what has happened. A neuron's connections, what followed
its activations, event and action alike, and what the actions earned, are in neither the graph nor the
dictionary (D25), and neither is an inference that did not run (R32).

**It holds nothing about the search either.** Strengths, fits, estimates and worths are the machine's and
the neurons' and enter no cost, because expanding an apex neuron needs the functions and nothing else.

**It holds the bindings.** What stands in a variable stays on the apex, bound, the edge from it to the call
paid with the call (D13), and expansion puts it back at its node (R28).

**Two runs that differ only in where things stand cost the same.** The graph has no coordinates, so a shape at
one position and the same shape at another cost the same, and nothing in the costs reads a coordinate
(D11).

> **D13 — Cost.** Every cost in the design is a number of edges of the graph, a relation between two activations
> being one (D55, D34): how many it takes to state a piece of the graph, over the fields the dictionary holds
> (D56). Nothing is stored for it; the
> cost is how the machine compares choices. Every edge costs the same, `1`, and a name costs nothing: a node
> is in the graph through its edges, and a pose is a name (D8). Three things cost on the apex: a relation
> standing there; a failure, a relation of the function that did not come true where its call stands; and a
> binding, the edge from a bound activation to the call it is an argument of. A call is a node like any other:
> it is in the graph through its edges and costs nothing of its own, whatever it covers (D53). In the
> dictionary a function costs its relations and one edge more, for its entry: the edge the dictionary holds it
> by (D56). So one function of two relations costs less than two functions of one.
>
> | What | Edges |
> |---|---|
> | a bare activation on the apex | nothing of its own |
> | a relation on the apex | `1` |
> | a call on the apex | one per relation that did not come true, one per argument; nothing for itself (D12) |
> | a function in the dictionary, of `m` relations | `m + 1`: its relations, and one for its entry (D56) |
> | a class or a parameter, in the dictionary | nothing |

The precision `P` and the neuron capacity `N` are the hardware's (§3), not prices. Neither enters a cost. They
set what can exist: how many neurons there are to allocate (R16), and how far an offset can reach.

**Every saving is a comparison of two costs.** Each savings and each worth in the design is the cost of stating
the same thing two ways, A less B, and nothing else: a piece of the graph stated bare against stated through a
function (D22), a relation paid on the apex at every occurrence against held in the dictionary once (D57).
Nothing is priced that is not in A and left out of B.

**Covering an activation saves exactly what it cost.** A fixed node that came true saves the relations the
function fixes at it. A variable that came true saves them too, and its binding is paid with the call.

**Nothing about frequency enters a price.** A symbol costs the same however often it occurs. How often it
occurs is what the worth measures (D57).

## 3.10 Savings

> **D54 — Fit.** A function fits where stating what it names through it costs less than stating it bare: its
> savings there are positive (D22). A fit is a match of relations to relations. The board is the relations the
> neurons returned (R21), and a fit of a function is one board relation per relation of the function, with the
> neurons, the dimension and the offset the function's relation states, two relations of the function that share
> a node landing on two board relations that share an end. A board relation is read as it stands: its ends are
> the neurons that fired, and an interneuron on the apex is a neuron as itself. Nothing is reserved: two
> functions may land on the same board relation. A function may fit at several positions in one frame, each fit
> one activation of its interneuron, and each fit is priced as it falls (D22).
>
> **A fit is read at the present, over what has happened.** It enters at a node of the function, on an
> activation at the frame being recognized that the node admits, its own pose,
> and reads the function's relations from there (D31). **A function in time is fit as it unfolds, as a cycle
> is.** It is entered at any of its nodes that fires in the present frame. The nodes later than that node are
> still to come. The fit leaves them out, with anything reached only through them. Nothing is charged for
> them, now or if they never come: stopping is not a failure. What lies behind the node entered at is read as
> any fit reads it, each relation true or failed. So the call stands in the phase that ends at the node
> entered at, and it stays in the phase it reached (D37). It stands as any call stands, whether or not its
> later nodes come. Where a later node fires, and the fit from it reaches an activation that stands under an
> earlier call of the same function, uncovered, whose latest node is that node, that call takes the new node:
> it is the same call, its phase that much longer, and no new call is made (D53). A fit needs one relation to
> land, so a function's earliest node alone is no fit. A function inside the frame is entered at any
> of its nodes, since all of them stand in the present frame. A cycle is entered at any node too: its fit ends
> where the cycle stops landing. Every node the fit reads is then at or before the
> present, so every
> relation it reads is true or failed when the function is fit, and nothing is pending. Its phase is the
> node of the latest activation it lands on, the node of the earliest, and the laps it read (D37). Without an
> activation at the present that some node admits there is no fit.
>
> **A cycle is read round, lap after lap.** From the node the fit entered at, the cycle's relations are
> matched one after another, each from the node the one before reached, round the cycle and round again
> (D15). The fit goes round while they land, and ends at the first that does not. That relation is not a
> failure, whichever of the cycle's relations it is: every fit of a cycle ends where the cycle stops landing.
> So a cycle's call may stand on part of a lap, and its phase says the node it started at and the node it ended
> at (D12, D37). A node of the cycle may land on several activations, one per lap; how many is the call's laps
> (D53). Where the activation it reaches stands under an earlier call of the same function, the relation is
> that call's own function going round, and the call takes the lap: it is the same call, its phase that much
> longer, and no new call is made (D53). Nothing is read past
> the cycle.
>
> **Coming true.** Each relation of a function comes true or does not. It comes true where the board holds a
> relation of its dimension and offset whose near end is the end already landed and whose far end is what the
> function's far node admits:
>
> | Relation, at its far node | Comes true where the far end is |
> |---|---|
> | a fixed node | its neuron, standing on the board at that position (D5, D24) |
> | a class | a member; it takes members only (D41) |
> | a parameter | any neuron where the node is first reached, reading from the node the fit entered at, and the value already bound where it is reached again (D44) |
>
> A relation that does not come true lands on no board relation and is paid with the call (D12). A node no
> relation that came true reaches stands nowhere: the call pays nothing for it, and expansion reads it from
> the relations that did (R28).

> **D22 — Savings.** What one fit saves where it is read: the cost of stating bare what it lands on and came
> true (A), less the cost of stating it through the function (B).
>
> | | Edges |
> |---|---|
> | A | for each relation of the fit that came true, the board relation it stands in for, one (D12, D54) |
> | B | the call's cost (D12): one failure per relation that did not come true, one binding per argument, nothing for the call |
> | savings | A less B |
>
> A relation that did not come true is in neither: whatever stands there costs as bare either way. What a
> relation is worth is always what it cost (D13).
>
> **A call's cost stays with the call.** A fit that names a call is
> credited the relations it read and nothing more: the call's failures and bindings are paid where it stands
> (D12). What a call covers went with it: a relation it covers is there for no other fit to
> land on, and what was lifted to it stands, at the call, for a later fit to read (D24, D55). So
> a function saves what it takes out of the cost, and that is relations.

**What a function's call stands in for is what it covers**; that is A. What it costs is its cost; that is B.
A neuron the function does not name is in neither: it costs what it cost whether the function exists or not.

> **D57 — Worth.** A relation saves nothing by itself: only a call takes anything out of the graph, the board
> relations its function's relations land on, for its failures and bindings (D12). The worth is that
> saving shared out among the function's relations, since grouping decides relation by relation which belong
> to a function (R50, R54). The machine keeps nothing for it: one relation's share, in one function, is read
> at the node it attached to the function at (D56), over that neuron's history (D18, D47, R50):
>
> | | |
> |---|---|
> | with | the activations of that neuron at which the function's other relations at it held, and this relation held with them |
> | without | the activations at which the function's other relations at it held, and this relation did not |
> | worth of a relation | `with − without − 1`: one relation saved at each occasion it held with the function, one failure at each it did not, and one relation in the dictionary (D13). **A relation counts in a function only where that function lands.** In a function that is a cycle (D15), that is every occurrence: each relation is counted at every occurrence, its strength less one, with no `without`, since the cycle's call covers it wherever it holds, on part of a lap as on a whole one, and where the next relation does not land the lap ends, which is never a failure (D54) |
> | worth of a function | the sum of its relations' worths, less one for its entry in the dictionary, less one binding for each variable at each occasion its node held with the function (D13). A call costs nothing of its own, and what it fails is in its relations' `without` |
>
> **No neuron counts the function.** A neuron's rows hold the relations at its own activations and nothing
> beyond (D18), so an end of a relation sees of the function only the relations the function has at that
> node. Its `with` says those held together with the candidate, whether or not the rest of the function held
> at the same occasion: a partial reading, from one node. The function's worth is those readings summed over
> its relations, each from the neuron that holds it, and no neuron and no reading sees the whole.
>
> **A call costs nothing, so every relation saves one at each occasion.** A function of one relation is worth
> its occasions less two, one for its relation and one for its entry. It is worth nothing at two occasions, so
> it starts at its third (R50). Its call covers one relation for nothing wherever it is taken (R24), and the
> function stays until its relation's strength slides below 3 (R18). Nothing is built that is worth nothing.
>
> **A relation's contribution to a function** is the function's worth with it less its worth without it, every
> other relation priced with it and without it, since each is conditioned on the others. Alone, in a function
> of its own, its contribution is that function's worth, its strength less two. A relation's own worth in a
> function and its contribution
> differ: a relation can be worth something beside another while the two together are worth less than that
> other alone. **Every move is priced by the dictionary's worth**, the sum of its functions' worths, before and
> after, over every function the move touches. A move that takes a function's last relation touches that
> function and its subtree, every function that leaves the dictionary with its interneuron (R38). A move is
> made where that total rises and nowhere else, since
> the machine does nothing that does not help (R54). So no sequence of moves on the same histories comes back
> to where it started.
> The neurons' histories are the whole record, over the last `H` activations of each (D18), so a relation that
> stops recurring sees its `with` and its strength slide out, its contribution fall below zero everywhere, and
> leaves (R18). Worth measures what a
> function saves wherever it fits.

**This is the only valuation in the design**, and it is read by three rules, so the three numbers differ, and
are meant to:

| Read by                               | Over                  | Against                                  | Settles                            |
|---------------------------------------|-----------------------|------------------------------------------|------------------------------------|
| the machine, in the election (R22, R24) | the fits             | the board                                | what is credited, and so the cost |
| the machine, keeping the dictionary (R50, R54, R18) | one relation in one function | the histories of the neurons the function stands in | which relations the dictionary holds, and in which function |
| the machine, in the match (R49)       | one inferred function | the frames it reaches back into          | whether the inference is proposed  |

One rule decides what a set of functions covers for a set of relations. The election runs it over the board. It
is stated here and cited from there.

> **D28 — Greedy cover.** The operation that covers a board with **fits**, each landing on relations, and pays
> each relation to exactly one fit; an activation may be landed on by several fits, each covering relations
> of its own, until it is covered (D24, D10). It repeats:
>
> 1. **Measure** every fit not yet taken against the board as the taken fits have left it. A fit lands on no
>    relation a taken fit covers and on no activation the taken fits have covered, so one read through either
>    is read again over what stands, and a
>    cycle's laps past it are a fit of their own. Its coverage is what it would take off the board: the
>    relations it lands on, less any that are refused or blocked (D24, D54). Its price is its call's cost there
>    (D22): its failures and bindings, and nothing else (R22).
> 2. **Take** one fit, and **stop** where none covers more than it costs. A fit is inside the frame or in
>    time, as its function is (D15), and the cover runs over one kind at a time, the frame before time (R24).
>    The fit whose `coverage` exceeds its `price` by the most is taken. It covers what it was measured on. A relation it covers decides the kind of
>    each of its ends that had none, and that activation's relations of the other kind are blocked. Every
>    relation whose two ends the fits taken at this level now join is refused. A fit inside the frame
>    covers every activation it landed on, at a relation it covered (D24). An
>    activation that is covered once the fit is taken is off the board from the
>    next round (D24): no fit lands on it, and each relation of its own that no call covers, and that is not
>    refused, is lifted to the calls that landed on it (D55).
>    An activation at one of its variables stays on the apex, bound, and the relations the function does not
>    fix at it stay in the set, so a later round may cover them (D10). The round repeats from 1.
>
> **What it returns is what was covered and by whom**: the fits taken, what each covered, and what no round
> covered, which stands bare. A fit read again is priced again; otherwise `price` is fixed by what fired, and `coverage` only
> falls.

## 3.11 Generalizing

> **R52 — Generalizing.** Two functions are compared as graphs, and the comparison is a fit of one onto the
> other with the names ignored (D15, D31): grown from one relation of each with the same dimension and offset,
> through the nodes the relations share, until every relation of each has landed on one of the other. Where
> that holds, the two have the same shape, and their nodes are paired. What is compared is what each
> interneuron stands for, its expansion to base symbols (D36), since two functions that stand for the same
> shape may hold it through different calls.
>
> For each pair of nodes:
>
> - where both hold the same neuron, the node is fixed;
> - where they differ, the node is a parameter where the two neurons have each stood there with nothing in
>   common but the position (D44), and a class whose members are the two neurons otherwise (D41).
>
> The general function is a new function over base symbols, with those variables at those nodes and the shape's
> relations. It **stands for** every function in the dictionary of its shape that holds, at each node, the
> neuron it holds there or a neuron its variable takes there: the two it was read from, and every other that
> differs from them only at those nodes.
>
> **It is taken where it saves more than the functions it stands for save together.** What each saves is its
> worth, read over the same histories (D57). At one activation the general function covers what one of them
> covers and pays one binding for each variable besides (D13), so it saves less there than they do. It has
> more activations than they have: every occasion one of them held, and every occasion its relations held
> with a neuron at a variable's node that none of them holds there (D47). And it is one entry in the
> dictionary where they are several. Where its worth is higher than their worths together, it joins the
> dictionary and they leave it. Its one entry against their several is in that comparison (D13). Where its
> worth is not higher, it is not made, and they stay (R54).
>
> So a general function that saves nothing at an activation is never taken: one relation for one binding, or
> two relations for two. A class fits only its members (D41), so a function over a class has no occasion the
> functions it stands for do not have, and is taken only where the entries it saves outweigh its bindings.
>
> **The comparison is made when an interneuron fires and the index gives a function of the same shape**
> (D36), over the histories as they then stand. A general function that was not taken is compared again at
> the next such activation. Where the dictionary already holds the general function, it is not made again. A
> function grouped beside it that it stands for is taken like any function, and leaves the dictionary as soon
> as its interneuron fires and the index gives the general function, which already stands for it. The general
> function is held like any function: find reaches it through its open ends (R53), the election takes it
> where it pays (R24), and its worth is read like any function's (D57).
>
> **The pairing starts at the origins.** Two functions of the same shape are paired from their origins, and
> the pairing follows the relations from there. Two functions whose latest nodes are in their cycles have
> none (D15): they are paired from any node, every way the cycle allows, and the pairing that lands every
> relation is taken.

Generalizing needs two functions to start from, each built on something that recurred. After that, a
new thing fits the general function directly: any neuron for a parameter, and a member for a class.

## 3.12 The objective

The cost of the run is the cost of two things (D13): the dictionary, every function's relations; and the
graph of the run, its edges with the calls standing for what they cover. There is one reading of it. A call is
its cost, what the call covers stands inside it, and what it bound stands beside it. A neuron above prices
an interneuron at what it cost below, so nothing is saved by naming it again and nothing is gained by pricing
it twice. The election takes nodes off the frontier and puts
a call standing for them (R22). Grouping and the moves do the same to the dictionary, one relation at a time (R50,
R54). Nothing computes the total: every quantity the design uses is a **difference**, which is finite however
long the run is.

## 3.13 Parameters

> **R4 — `H` is the only free parameter.** One thing is the designer's to pick: `H`, the depth of every
> neuron's history (D18). Nothing else is. The alphabet (D1) states the
> problem and is not a knob. The reach (D4), adjacency (D5) and the offsets (D6) are fixed: a base neuron sees
> one position in each dimension, and an interneuron one beyond what its function covers, each through the
> calls that stand there (D4). The precision `P` and
> the neuron capacity `N` are the hardware's (§3). The neighborhood policy (D50) and the action policy (D51) are
> declarations like the alphabet: each states the kind of world the machine is in, and neither is a knob. **No
> rule introduces a constant, a threshold, a window or a cap of its own.**

A base neuron sees one position because that is where every longer distance comes from. A relation across a
stretch is a chain of relations of one, each a function of its own or part of one, and the interneuron over the
chain sees one position beyond it, so its relations span the stretch in one offset. Every distance the
dictionary holds is built this way, level by level, and none is declared. Where a stretch is unpredictable
in its middle, a variable has to form there before the stretch can be covered (R52), and that is
the price of seeing one position: a deeper hierarchy, never a number nobody can derive.

# 4. Neuron

What this section defines belongs to one neuron: what it holds, and what it counts. A neuron records, counts
and returns. It decides nothing about what is covered, and it does not know that functions exist.

## 4.1 Interface

A neuron is a symbol, and a type (D2). It holds a history of its activations, a relation index over it, and its
connections (D25).

**The machine reaches a neuron through six calls, and nothing else writes into one.** Each is specified where
it is used.

| Call              | Description                                                                                                       | References |
|-------------------|-------------------------------------------------------------------------------------------------------------------|------------|
| allocate neuron   | The machine allocates it the first time a fit of a function is accepted and a neuron is free, holding nothing: the function's call neuron. | R16 |
| activate          | At its level: record the frame's activations, and return the relations from each with their strengths.             | §5, R20    |
| count             | At its level, after find: given the functions it stands in and their candidates, return with and without for each relation and each candidate, over its history. | R50, D47   |
| forget neuron     | The machine removes a decommissioned neuron from every row and relation of the history that names it.              | R38        |
| connect           | After the frame's last level, reaching every open activation at age 1 and beyond: it delivers what stood on the apex this frame, calls with what they cover, and any reward; the activation's row takes what stood as what followed it, its connections strengthen while it is uncovered, and the call collects nothing. | §7.1, D18 |
| infer             | After the frame's last level, reaching every activation at age 0: it delivers nothing and collects what the activation infers. | §7.2 |

In `activate` a neuron records and returns, in the frame it fires. In `count` it counts over its history. In
recognition the machine fits what the dictionary holds, moves relations between functions, elects and
allocates. In `connect` and `infer` a
neuron learns what followed and infers what comes next. In selection the machine chooses among those inferences
one winner per dimension.

> **R1 — One decision point: the frame it fires.** A neuron is called once per level it fires at, for every
> activation of that level together, each at age 0 (D53): it refreshes its history and its relation index, and
> returns the relations from each activation (R20). Nothing else reads it
> until its next activation.

## 4.2 History

> **D18 — History.** A neuron's record of its own activations: the last `H` of them, oldest first, each holding
> the phase it fired in (D8), the neighborhood it observed in the frame it fired (D7), each neighbor a pose
> at its offset (D6, D26), and what followed it within its reach, each at its offset too. A neighbor is held as what stood on the board, as
> itself (D5). **A row is written
> when the activation fires and takes what followed as it comes**: at every frame of the activation's window
> the connect call hands the neuron what
> stood on the apex (§7.1, D5), and the row takes it at the
> activation's age, whether the activation is covered or not (D10). **Nothing a row holds is ever replaced.**
> Where a call lands on an activation the row holds, the row keeps that activation (D10). Inside the frame,
> where the row's relation to that activation is lifted to the call (D55), the row takes the call as well, at
> the call's offset. The lifted relation is a relation of the row's own activation, so its row holds it, as
> the call's row does. Where the row is not yet written when the relation is lifted, it holds the call when it
> is written. A row takes one neighbor for a call, however many relations were lifted to it. The row that
> takes the call is the row of what stands at the relation's other end on the board. Where a call had already
> landed on that end, the relation stood lifted to that call, so it is that call's row, and the row of the
> activation under it takes nothing (D55). The row of the call the relation is lifted to takes nothing for
> the lift: it is written when the call fires, with what stands on the board then (D5).
> In time a row takes what the connect call hands it and nothing else. So in the history the frame around an activation is whole on both sides in time, as it is in
> space, and the relation between two activations is held by both of them, each from its own side (D34). It is
> the evidence every strength is counted over, and it holds nothing else: the neuron's connections are not in
> the history and `H` does not bound them (R31).
>
> **`H` is declared once for the machine and is the same for every neuron.** It counts that neuron's **own
> activations**, not a stretch of run, so a neuron that fires constantly and one that fires rarely report their
> relations over the same amount of evidence. **The ring is exactly `H` deep once filled.** A neuron may fire
> many times in one frame and not at all in the next (D8), so how many frames its `H` activations span is
> whatever its rate makes it, from part of one frame to a long stretch of the run. Nothing else anywhere is
> measured in frames.
>
> **Every activation the call hands it is recorded, covered or not** (R20). An arriving activation evicts the
> oldest, and only then.

## 4.3 Relations

**A neuron records one kind of relation:** a particular neighbor at an offset from itself.
What is general across neighbors, a position where different neurons stand, is not seen in any one activation, so
no neuron records it. The machine makes it (R52).

> **D47 — Relation index.** What a neuron counts over its history (D18): every relation from its own activation
> that its rows hold, each with its **strength**, the number of rows that hold it. A row holds every neighbor
> as what stood on the board when the row was written (D18). So a relation to a neuron is counted in the rows that hold that neuron at that offset,
> whatever calls had landed on it. It is updated
> when a row is admitted, when a row takes what followed (D18), and when a row is
> evicted, and at no other time.
> Handed a set
> of relations it is an end of and one more, it returns how many rows hold the set with the one more and how many
> hold the set without it, over the rows that have taken what followed (R50).
>
> **A relation with a variable at an end is counted across the neurons the variable takes**, any neuron for a
> parameter and a member for a class (D38). With the variable at the far end, a row holds the relation where
> it holds one of those neurons at that offset. With the variable at the near end, the relation is counted in
> the rows of each of those neurons. In one row a variable is one neuron: where it stands at several ends of
> the set, the row holds the set where the same neuron stands at each. So a general function is read over
> rows written before it was made (R52).

How the index is laid out is the implementation's. What it holds is not: a relation's strength is the number of
rows of the history that hold it, and which rows those are leaves the neuron for nothing.

## 4.4 Connections

What follows an activation enters no cost (D12) and no test. It is held only here, as connections: what stood
on the apex in the frames after it, event and action alike.

> **D25 — Connections.** A neuron holds, per phase of its own and per `(neuron, offset)`, the neuron in its
> phase, one **connection**: a **strength**, the number of times an activation of the neuron in that phase saw
> that neuron stand on the apex at that offset, and its **earnings**, the total reward the actions of those
> frames received. Its **estimate** is its earnings over its strength,
> and is read, never stored. The offset is a full D6 offset from the holder's activation's position: positive in
> time, because what is connected to comes after, and signed in every other activation dimension. A connection
> to a call names it in its phase and is at its offset (D53, D34). **A connection reaches as far as a
> neighbor does, in every dimension at once** (D4): an activation connects to what stands on the apex within
> its reach in every activation dimension the two share, the window in time (D9) and the same reach in every
> other, and to nothing beyond it. What is connected to may differ from the activation in several dimensions,
> as a neighbor may (D5), so a thing that moved is connected to where it went: that is where displacement is
> learned. **A connection is written only where the future table of the policy admits the target's kind among
> the holder's kind's connections** (D50). Nothing about any one activation is kept. A call's connections are
> keyed by its phase less its laps, the node it started at and the node it stands at, and a call a connection
> names is named the same way. So a cycle's call standing at one node of the cycle learns apart from one
> standing at another, in one neuron (D48), and a running cycle learns what follows each of its nodes once,
> for every length it reaches. A function in time that is not a cycle does the same: its call in the phase
> that ends at one node learns and infers apart from its call in the phase that ends at another (D54).
>
> | Component | Description                                                                              |
> |-----------|------------------------------------------------------------------------------------------|
> | key       | the holder's phase less its laps, and `(neuron, offset)`, the neuron in its phase less its laps, the offset strictly positive in time |
> | strength  | the number of exposures: the times an activation saw that neuron on the apex at that offset |
> | earnings  | the total reward those exposures received                                                |
> | estimate  | earnings / strength, read whenever the connection is read                                |
>
> Every neuron holds them, base events and base actions included.

Nine cases are worked by hand on these definitions, listed in Part VI.

# Part II — Past and present: a neuron

# 5. Activate call

> **R20 — Call, in order.** Once per level it fires at, the machine hands a neuron **every activation of it that
> fired this frame at that level**, all at once, each with its neighborhood, at age 0 (R1). They are processed
> together, and what the call returns for them is returned together.
>
> | Step            | Description                                                                                        |
> |-----------------|----------------------------------------------------------------------------------------------------|
> | refresh history | Evict the oldest rows as the frame needs, and admit the frame's rows (D18).                         |
> | refresh index   | Add the relations of the admitted rows, and take away those of the evicted ones (D47).               |
> | return          | The relations of every activation handed in, each with its strength, in one return, divided by activation (R21). |

The call runs before the machine fits and groups (R53, R50). An activation a call will land on
(D10) is handed in like any other: its neuron records it and returns its relations, and what the machine does
with them is the machine's.

## 5.1 Refresh history

The history holds `H` activations (D18). A frame bringing `A` activations evicts the `A` oldest once the
history is full, and fewer before, as many as it takes to make room. The frame's activations then join the
history, whole (D7). A frame bringing more activations of one neuron than `H` fills the history with the first
`H` of them, in the order the input lays them out, and the rest are not recorded; the machine reports it. `H` is
the designer's to pick (R4), and a report says it was picked below the most activations a neuron can have in one
frame.

## 5.2 Refresh index

The index follows the rows (D47).

**An admitted row adds its relations.** Every neighbor in the row gives one relation: that neuron, at its offset
from the activation (D6). The relation's strength goes up by one.

**A bound neighbor gives worthless relations where its binding covered it.** A neighbor bound in a variable at a
lower level costs nothing in each relation its binding covered, so that relation is marked: it holds a node and
covers nothing (D10, D22). A relation to it that its binding left open costs an offset, and costs like any
other.

**An evicted row takes its relations away.** Each relation it held loses one from its strength, and a relation
whose strength is zero leaves the index.

**A decommissioned neuron is forgotten.** The forget neuron call (§4.1) removes every relation that names it, and
every row's entry for it. Nothing else touches the index.

## 5.3 Return

> **R21 — What the call returns.** One return for every activation the machine handed in, divided by
> activation. For each activation it holds one relation per neighbor (D34), each with its strength, the number
> of rows of the neuron's history that hold it, this one among them. An
> activation with no neighbor has nothing in its part.
>
> **The machine reads every relation in its one form** (D34). Every occurrence of a relation is on two rows,
> one at each of its ends, so each end counts it, and the machine takes the larger of the two counts as how
> often it occurred: a history is a window that can only miss occasions (D18, D34). A relation between two
> activations of this level comes back from both in this return. A relation to an activation of an earlier
> level or frame comes back from the activation that fired now, and the machine asks the earlier neuron for
> its count at the count step (§6.1). That neuron's row takes this occurrence afterward (D18, R50), and the
> machine counts it there already: the occurrence is one, on two rows.

A relation held in one row cannot recur yet, so **nothing is ever built on a single occurrence** (D57, R50); it
is still returned, since a function already in the dictionary may hold it (R53). What the call returns is the
neuron's whole say in what is covered: what it saw, and how often. Which rows hold what never leaves the neuron.

## 5.4 Cost of the call

With `A` activations this frame and `M` neighbors per row:

| Step | Work |
|---|---|
| refresh history | `A` rows admitted, as many evicted |
| refresh index | `A · M` relations counted in and out |
| return | `A · M` relations, each with one number |

# Part III — Past and present: the machine

# 6. Recognition

> **D23 — Recognition.** The machine covers the board with interneurons, each fit taken when the relations it
> covers cost more than its call's cost, never adding what its function costs in the dictionary (R22, R24).
>
> **Covering everything is not the goal.** An activation no accepted function covers stays in the cost as
> itself, at what it cost (D13), and that is the lower cost whenever no function could hold it for less.
> **What coverage varies is the cost, never the fidelity.**
>
> It is **axis-general**: a function names nodes at offsets, so an interneuron replaces a piece of the graph
> however it lies across time and space. A purely spatial function is the case where every offset in time is
> zero. A temporal function is recognized at its latest node, in the frame that node fires, over what has
> happened.

**The board** is the apex as it stands at a level, partway through the frame: what the levels below have left
uncovered so far, with the activations of earlier frames still on the board (D9), and the relations the neurons
returned among all of them. It is a piece of the graph (D55), and it changes with every level. The apex of the
frame is the board once the last level has run.

**The machine compresses a frame in this order and no other.** The frame goes first, up all its levels.
The first level is over what the frame reported, each after it over what the level before accepted, until a
level takes and starts nothing:

| Step       | Description |
|------------|-------------|
| activate   | Fire every interneuron the level below accepted, once, in its fit's phase (D53), and call `activate` on every neuron with an activation at this level (§5). They return their relations with strengths. |
| the frame  | Run the six steps below over the relations inside the frame and the functions that hold them (R24). |
| allocate   | Give each accepted function with no interneuron one, reused where a stretch comes round (R43), allocated otherwise where a neuron is free (R16). It fires at the next level. |

Then the levels run again from the first, on the board as the frame left it, until a level holds no
activation in this frame. A level whose activations all came from the frame's own run still runs for time:

| Step       | Description |
|------------|-------------|
| activate   | Fire every interneuron accepted at the level below since this level last ran, once, and call `activate` on it. An activation is activated once in a frame. |
| the frame  | Run the six steps over the relations inside the frame that those activations returned (R24). Functions are fit and start from those relations only. Group reads the whole board for its moves. |
| time       | Run the same six steps over the relations in time and the functions that hold them (R24). |
| allocate   | As above. |

The six steps, each run over one kind of relation (D34):

| Step       | Description |
|------------|-------------|
| find       | Read every returned relation in the index, as it is and with either end open, and fit every function it reaches on the board wherever its relations match, one fit per position (R53, D31). |
| elect      | Take the fits, the largest coverage less price each round, stopping when no fit pays (R24). The dictionary as it stands covers what it knows before anything in it changes. |
| count      | Hand each neuron every function it stands in, fit at this level or not, with the candidates of the ones that fit, and call `count` on it: it returns with and without for each relation and each candidate (R50, D47). |
| group      | Over those counts, move each relation where the dictionary's worth rises most, to a function it is a candidate of, one that closes a cycle before one that adds a node, the nearest on a tie, or out of a function it does not pay in (R50, R54, R18). |
| elect      | Fit every function that changed in group on what the first election left, and elect again over those fits the same way (R24). |
| start      | One function at a time. Of the relations no function holds that are worth a function and still stand on the board, start a function from the one with the most instances standing, elect its fits, and look again at what stands. Stop when none is left (R50). |

Then, once the last level has run:

| Step         | Description |
|--------------|-------------|
| decommission | After the frame's connect and infer calls (§7), decommission every interneuron that is due (R38). |

## 6.1 Process levels

**A level is explained as a set of calls with their arguments** (D37). Every accepted fit is a call: a
function's interneuron, and beside it, bound on the apex, what stood in each of the function's variables. What
stands above is made of the calls and their arguments (§6.4): a function there over an interneuron alone holds
for every argument the call was given, and one over the interneuron and a bound activation holds for that one
argument.

> **R26 — Levels.** A frame is compressed one level at a time. The first level is every base activation the
> frame reported, all of them together: they are activated and return relations, the machine fits what the
> dictionary holds and groups, and the election settles which fits are accepted. The next level is the
> interneurons of those fits:
> they fire, are activated and return relations, and so on upward. **When a level holds no activation in this
> frame, the frame's levels end there.** Nothing declares the depth and nothing caps it. A level belongs to a frame and not to a
> neuron: a neuron fires at whatever level accepts its function, and one neuron may fire at different levels of
> different frames.
>
> **The frame runs up all its levels before time runs at any.** At each level of the frame the order is
> activate; then find, elect, count, group, elect and start over the relations inside the frame; then
> allocate. It ends at the level that takes and starts nothing. Then the levels run again from the first,
> every level that holds an activation in this frame, whether or not anything new fires there. At
> each the order is activate, for what was accepted since the level last ran; the six steps over the
> relations inside the frame that those returned, where functions are fit and start from those relations
> only and group reads the whole board for its moves; the same six over the relations in time; then allocate. So
> a call made in time is still related inside the frame to what stands at its position. Nothing in time is
> fit, joined, moved or started, at any level, until the frame has nothing left to do at every level (R24).
> What fires is what
> the level before accepted; what a neuron returns is what is fit; the first election takes the best fits of
> the dictionary as it stands, so what is known covers its pixels before anything in the dictionary changes;
> the neurons count over what was fit; the moves read their counts; a function that changed in
> group is fit on what the first election left, and the second election is over those fits; functions then
> start one at a time, each elected before the next is chosen (R50); and an
> interneuron is allocated only for a fit an election accepted (R16). A function cannot form over the relations the
> first election covered: a function that wins there keeps them until its relations leave on their
> worth (R18, R54). The dictionary changes in group and in start, and the index is read before and after them.
>
> Every neuron runs the same rule at its own reach (D4). **Compression is spatio-temporal for every neuron, in
> one pass**: a function names neighbors in its own frame or in earlier ones, one kind per function (D15), and
> where the two recur together a function at the level above relates their calls.

> **R53 — Finding.** For each relation `(a, b, δ)` a neuron returned, the machine reads the index (D36) three
> ways: as it is; with `b` open, `(a, ?, δ)`; and with `a` open, `(?, b, δ)`. The reading as it is gives the
> functions that hold that relation. A covered neighbor comes back as itself, with the relation to its call
> beside it (D5, R21), so a function that holds either is found. An open reading gives the functions that
> hold a relation with
> that neuron at that end and that offset and a variable at the other end, a class or a parameter (D38), and
> no function whose node there is fixed: a fixed node is found as it is or not at all. Where the open end
> holds a class the observed neuron is a member of, or a parameter, the dictionary function is fit by the
> observed relation the way D54 reads it, and this is where variables are matched.
>
> Every function any reading reaches is **fit** on the board (D31), from the node the reading matched: a
> reading tells the machine the function may fit, and the fit begins on an activation at the present frame
> that can stand at that node, or nowhere. Matched relation by relation out from there, those of its relations
> that land on a board relation are true, those that do not are its failures (D22). A function is fit once
> per activation at the present it can enter at, in time at each of its nodes that fires in the frame
> (D54), and a fit reached from two of its relations is one fit. A relation no
> reading reaches is left for grouping (R50). Everything find fits goes to the election once every relation
> of the level has been read (R24).

> **D31 — Fitting a function.** A function find reached (R53) is fit on the board, on the board relations it
> matches (D15). Competing matches over the same board relations are all fit, and the election picks among
> them (R24). The same function matched to the same board relations from two of its relations is one fit.
>
> **A function is fit whole over what has come.** The machine takes it or leaves it, its variables with it:
> it cannot hold where they do not. In time the nodes still to come are left out of the fit, and the call's
> phase says how far it has got (D54).
>
> **A function is fit only on relations that stand on the board.** Where a relation of it has been lifted, the
> relation that stands there instead names another pose (D55). So the function has nothing to fit there, and
> it gets no call. That alone does not remove it from the dictionary (R18, R54). It is fit again wherever its
> own relations stand on the board.
>
> **The fitting rule.** A fit is anchored at the node it enters at. In time the nodes later than that node
> are left out, and nothing is charged for them (D54). The machine lands that node on an activation
> at the present frame the node admits, its own pose,
> or one admitted at it where the node is a variable (D54), and extends the match relation by
> relation from there: each next relation of the function that shares a node with
> one already landed is matched to a board relation that shares that end and reads as it says. A parameter's value
> is the far end where its node is first reached, and a relation reaching the node again is matched to that neuron
> (D44). A relation that closes the cycle is matched to the node it reaches, and the match goes round from there,
> lap after lap, while that relation lands (D54). Where several board relations could answer one relation of the
> function, the one no accepted call covers is taken first, then the older one (R24). A relation no board relation
> answers does not come true, and the match goes on from the function's other relations. Every relation can fail
> but the one thing the fit rests on: where no activation at the present can stand at the node it entered at,
> there is no fit and nothing fires (D15). What stands in each variable on that occasion is the call's arguments
> (D37). Expansion reads a call back by the same rule (R28).

> **R50 — Grouping.** Every function the index holds is a center for the relations grouped into it, and every
> fit of it (R53) is one reading of it. The machine keeps nothing for it: what grouping reads, it asks the
> neurons at the count step (§6.1), over their histories (D18, D47).
>
> **A candidate** is a board relation the neurons returned with strength 2 or more, this frame or at one
> before while its activations still stand (D9), that the fit did not land on, and that the fit touches in
> one of two ways. The first election left it on the board, and it shares a neuron with an activation the fit
> landed on or with the call. Or a call covers it, and the fit touches that call: the fit landed on an
> activation that call also landed on, and the relation shares that activation. So a relation another function
> holds is offered where the fit landed on an activation that function's call also landed on, and whether it
> moves is decided by its gain here against its contribution there (R54). Where the fit landed on a call as a
> node, the relations under that call are no candidates: a function that names a call is offered nothing from
> under it, and its relation to the call stays a relation to the call. A relation under a call the fit does not
> touch is no candidate. Its
> other end is a
> neuron the function does not name, which would be a new node, or a node the function has, which would close
> a cycle (D15). An activation returns the relation to what stands on the board,
> as itself (D5). Where the fit
> already lands on it, it is no candidate. It is no candidate where its new node would be the origin, later in time or at a
> lesser position in the same frame, since nothing joins a function later than its origin (D15).
>
> **Three more things are no candidate** (D15, D34). A relation of the other kind than the function's
> relations is none: a function holds relations in time or relations inside the frame, never both. No relation
> is a candidate of a function that is a cycle. A relation that would close a cycle is a candidate only where
> every relation of the function would then be on the cycle. Inside the frame it is a candidate only where
> its offset is the one the function's relations compose to between its two nodes, so that it reaches the
> activation the fit landed on there. In time it is a candidate at any offset. And a relation of a pose to itself is a
> candidate of no function, since it is a cycle alone: it starts a function of its own. A relation between one
> neuron in two phases is between two nodes, and it is a candidate like any other (D8).
>
> **What the neurons count.** The machine hands each neuron every function it stands in, which the index
> gives by neuron (D36), fit at this level or not, with the candidates of the ones that fit. A
> function nothing reaches any more, one naming a call that no longer fires, is still handed to the neurons
> that stand in it, so its worth is still read and it can still leave (R18). For each relation of the
> function the neuron is an end of, and for each
> candidate it shares, the neuron returns two numbers over its history (D47): **with**, how many of its
> activations hold the function's other relations at it, the ones it is an end of, together with this one,
> and **without**, how many hold them and not this one. A neuron sees no more of the function than the
> relations at it. Every activation of an interneuron is a fit of its function, so it counts over all its
> activations, for any candidate its rows hold. A row holds the frame around the activation on both sides in
> time (D18), so a candidate at either end of a function is counted by the node it attaches at: for a
> candidate at the past end, the node sees the function's relation after it and the candidate before it in
> one row. **A relation is counted at the node it attached to the function at**, which the function keeps with
> it (D56), at the offset that node's rows hold it (D18): that is the node the function already had
> relations at when it joined, so the reading is conditioned on them, and it does not move as the histories
> slide. A function's first relation attached at no node: it is read as a returned relation is, at the larger
> of its two ends' counts (R21), until a second relation joins, and from then on at the node that relation
> joined at. A relation
> alone is read at the node it would attach at in the function it is compared with, over the same rows, so
> joining and leaving are one comparison at one neuron, and a function's worth with a candidate and its worth
> without it are read at the same nodes, so a function of one relation is read, in both, at the node the
> candidate would attach at; a relation no function is a candidate of is read as a returned relation is, at
> the larger of its two ends' counts (R21). A relation of a
> pose to itself is read in one direction, from an activation toward the one its offset reaches (D34): a row
> holds it where that neighbor stands, whether or not the same neighbor stands at the opposite offset, so the
> ends of a run count as its middle does.
>
> **A relation counts in a function only where that function lands** (D57). A function that is a cycle lands
> on every occurrence of its relations (D15). So a relation
> of it is counted at no one node: every occurrence of it counts, and its count is its strength, the larger of
> its two ends' counts (R21). Where both its ends are one node there is
> one reading. A candidate is read by what the function would be with it.

> **A relation forward in time is read once the row has taken what followed** (D18). A relation backward in
> time from an activation is on its row when the row is written, and is read at once. A relation forward in
> time is on the row only once the row has taken what followed it, at the connect step of the frame after
> (§7.1), so it is read from two frames after the row's own. Until then it has neither held nor failed there,
> and the row is neither `with` nor `without` for it and counts nothing toward its strength alone.
>
> **A relation is counted as the row saw it.** A row holds every neighbor as what stood on the board when the
> row was written: an activation as itself, whatever calls had landed on it (D18, D5). A row that saw a neuron before a call landed on it keeps the neuron (D10).
> Inside the frame it takes the call too, where its relation to that neuron is lifted to the call (D18, D55).
> So a relation to a neuron is in the same rows whether or not a call had landed on it,
> and whether or not that call's function is still in the dictionary. And a relation lifted inside the frame
> is in the rows of both its ends, so each end counts it. The count reads those rows and nothing
> else: it never expands a call. A relation to a call
> is counted the same way, in the rows that hold that call in that phase at that offset (D34).
>
> **Distance and worth.** A candidate's distance to a function is `1 − with / (with + without)`: zero where it
> stood at every occasion the function did, one where it never did. Its worth in the function is
> `with − without − 1`: one relation saved at each occasion it held with the function, a failure at each it did
> not, and its one relation in the dictionary. Where a candidate closes a cycle, it and every relation of the
> cycle from then on are counted at every occurrence,
> their strength less one: the cycle's call covers them wherever they hold, and a lap's end is not a failure
> (D57, D54). A candidate joins the function where its
> gain is highest, the function's worth with it less without it. A candidate no function holds joins where
> that gain is above zero, and nothing else is asked of it. It is not weighed against a function of its own:
> it has none, and the rows where it is missing already count against it. A candidate a function holds joins
> where that gain is higher than its contribution in the function it is in (R54). Among the candidates of one function
> that pay, one that leads back to a node the function has, closing its cycle, joins before one that would add
> a node (D15). Where two functions tie, it joins the one it is nearest to (D57, R54). A relation in a
> function of more than one relation leaves it where it does not pay there: where its own worth in the
> function, its `with` less its `without` less one, read at the node it is attached at, is not above zero.
> It is not weighed against a function of its own. Where several relations of one function do not pay, the
> one worth least leaves first, and the function is priced again. A relation that leaves this way is held by
> no function: it joins where its gain is above zero, or starts a function as any such relation may. A
> relation of a cycle is not read this way, since it has no `without`: it leaves for a function of its own
> where its worth alone is higher than its contribution to the cycle. A relation leaves for nothing once it
> is worth nothing: the neurons return it with strength below 2, or below 3 where it is its function's only
> relation and the entry is its to pay (R54, R18). That asks whether it recurs at all, so it is read over
> every row that holds it (D47), where a worth is read over the rows that have taken what followed. A relation
> the neurons returned with strength 3 or more that no function
> holds may start a function of its own, two nodes and an offset, worth its strength less two
> (D57). A relation of a pose to itself
> starts one the same way: one node and an offset, a cycle from its first relation (D15). A relation held less often is left where it is:
> nothing is built that is worth nothing (R21).
>
> **Functions start one at a time, and only from a relation that still stands.** A relation **stands** where
> at least one instance of it on the board is covered by no call and is not lifted, refused or blocked (D24,
> D55). Starting comes last in its half of the level, after the fits of the functions that exist and after the
> moves (R26). Of the relations that may start a function and stand, the one with the most instances standing
> starts first, and on a tie the one whose standing instance is oldest. Its function is fit, and the election
> takes its fits (R24). Covering, lifting, refusing and blocking follow as each fit is taken. Then the machine
> looks again at what stands, and starts the next. It stops when no relation that may start a function stands.
>
> **So nothing starts from a relation that has already gone to the calls.** In an image, the row is started
> first and its fits cover the pixels. The relations those pixels had to the pixels beside the row are lifted
> to the row's calls, so no function between two bare pixels starts from them. The relation that stands
> instead, between the calls, starts a function at the level above (D55). What stops a relation from starting
> a function is that no instance of it stands, never that a call has landed on one of its ends.
>
> **Which relation it is, and where none is built.** A relation has one reading. Where its far end stands on
> the board, it is the relation to that activation as itself, whatever calls have landed on it. Where its far
> end is covered, it is the relation to each call that stands over it, the call no call covers (D24, D5, D55). It is counted in the rows that hold it (D47) and may start a function at its own strength of 3 where it
> still stands. So one pattern has one form, and nothing is built on a call while the activation under it can
> still be read (R18). A relation to
> a call that is the call's own function going round its cycle starts nothing: it is a lap of that call (D53),
> so a pattern that repeats adds laps and never a function over itself. Two calls that landed on one
> activation relate like any two neighbors (D26): the relation between them starts a function at strength 3,
> as any relation does.
>
> **Relations join only through a node, and a function grows toward its past.** A candidate is attached at the
> node it shares, so every function stays connected (D15). Where both its ends are nodes of the function it
> closes the cycle, and it is attached at no one node: it is counted at every occurrence (D57). From then on
> the function is closed, and nothing joins it.
> Nothing joins a function at a later frame than its
> origin: an activation after the call sees the call, standing at its position (D5), and the relation it brings,
> from itself back to the call, is a function at the level above, over the call (R26). Two relations that never
> touch are never one function: where they recur together, the level above relates their two calls.
>
> **The fits the first election chooses among** are every fit of this level of its kind, of functions with an
> interneuron and without (R24). A function that changed at this level is fit again before the second
> election, with its relations as they now stand, on what the first election left. A function that starts is
> fit as it starts, on what the elections before it left.

> **R54 — Moving.** A function changes by relations moving, one at a time, and by nothing else. Each
> move is the same decision, whether the dictionary's worth rises (D57), read from the neurons the
> functions stand in (R50). A relation's **gain** in a function is that function's worth with it less without
> it; its **contribution** where it stands is the same for the function it is in, and alone its strength less
> two, its relation and its entry. A relation no function holds contributes nothing where it stands, so it
> joins where its gain is above zero. A relation in a function of more than one relation stays while its own
> worth there is above zero, and leaves where it is not. Neither is weighed against a function of its own
> that it does not have. Where the relation is its function's last, its contribution there is that
> function's worth and the worth of every function in its subtree, since all of them leave with it (R38).
> The subtree is read from the dictionary as it stands, the function the relation would join included.
> Nothing the move would change is read.
> Every number is read at the node it attaches or would attach at, over the rows
> that have taken what followed, and a relation of a function that is a cycle at every
> occurrence (R50):
>
> | Move | When |
> |---|---|
> | a relation joins | no function holds it and its gain in the function is above zero; or a function holds it and its gain is higher than its contribution there; and in either case higher than its gain in any other function it is a candidate of, the nearest on a tie (R50) |
> | a relation leaves its function | the function holds more than one relation and no cycle, and the relation's own worth there, its `with` less its `without` less one at its node, is not above zero; it is then held by no function |
> | a relation leaves a cycle for a function of its own | its worth alone is higher than its contribution to the cycle, and higher than its gain in any function it is a candidate of |
> | a relation leaves for nothing | the neurons return it with strength below 2, or below 3 where it is its function's only relation, over every row that holds it (R18) |
> | two functions merge | a relation of one is a candidate of the other and joins it, and the rest follow one at a time, each where its gain is higher; the last gains the entry its function no longer pays, and is weighed against the worth of every function that leaves with it (R38) |

> **Nothing moves on a tie.** The machine does nothing that does not help: a move is made where the
> dictionary's worth rises and nowhere else. Two relations that save the same edges in one function as in two
> join all the same, for the entry: one function pays one where two pay two (D13).
>
> **A relation leaves only from an end.** A relation leaves its function, for another function, for one of
> its own or for nothing, only where the function's other relations still hold together without it (D15): at
> an end of a chain, or in a cycle, which its leaving opens. Either end of a chain is an end: where the
> relation at the origin leaves, the origin's node goes with it, and the latest of the nodes that remain is
> the function's origin from then on (D15). A relation whose leaving would split the function
> in two is not priced for leaving and does not leave. It stays until what lies beyond it on one side has
> left, one relation at a time from that end. So relations leave as they join, at the ends, and no function
> is ever priced in two parts.
>
> A relation that leaves its function of one relation leaves that function empty, and the function goes with
> it (R18). A function keeps its interneuron through every move: what the interneuron stands for moves with its
> function. A call that fired under an earlier form of the function expands to what it covered, not through
> the function (R28), so nothing of the earlier form is kept. A function that a move makes the same as
> another is that other function (D36),
> and the two are one from then on, the older interneuron standing for both (R43). A moved function is read
> against the functions of the same shape, and compared with them for a general function (R52).

> **R18 — Prune.** A relation leaves its function where it does not pay there, or where its contribution
> there is lower than its gain in another function (R54), and leaves for nothing once the neurons return it
> with strength below 2, or below 3 where it is its function's only relation and worth nothing there, over
> every row that holds it (D47, D57), read at every level a neuron standing in the function fires at, fit or
> not (R50). A relation that leaves for
> nothing stands bare wherever it holds. A function leaves the dictionary when its last relation does.
> **Relations leave one at a time**, the one that gains most by leaving first, and the function's other
> relations are priced again before the next leaves. **A function stays connected** (D15): a relation leaves
> only from an end, where the function's other relations still hold together without it, and one whose
> leaving would split the function stays until what lies beyond it on one side has left (R54). A
> class no function has any longer goes with its last function. A function's interneuron is decommissioned when due (R38).

## 6.2 Election

> **R22 — What a fit covers on the board, and what it costs.** A function is fit on the board (D31). The
> machine reads the fit against what has fired and derives both numbers.
>
> | | |
> |---|---|
> | the fit | the function and its interneuron or none (D31), its relations landed on the board's relations, with what each variable binds |
> | coverage | A: the board relations its true relations stand in for, where no accepted call has covered them (D54, D24) |
> | price | B: its call's cost (D12, D22): one failure per relation that did not come true, one binding per argument, nothing for the call |
>
> The machine fits the function by D31's rule. Coverage less price is the savings over stating what it covers
> bare, D22's expression over the board.
>
> **A fit is not charged for a relation its own call brings.** A relation that comes into being only when the
> fit's interneuron fires is no part of its price. The relation between its call and another call that stands
> for an activation it lands on is one (D26). Once the interneuron has fired, that relation is on the board at
> the level above. It costs one there, like any relation, and a function there may cover it.
>
> **A relation is covered once.** A fit whose relation lands on one an earlier accepted call already covers is
> credited nothing for it (D10, D24).
>
> **A node that fired and the fit does not land on belongs to neither side.** Its relations stand in the
> cost as they are if nothing covers them (D23), and the same way if this fit is accepted, since the fit
> does not land on them: their own cost either way, so they cancel before the test begins.
>
> **Coverage changes the credit and never the price.** A relation that came true on a board relation another
> fit of this election already covers is credited to no one and charged nothing (R24): it came true, so it is
> not a failure. A relation that did not come true is charged whether or not
> another call covers something at that position. **What a function gets wrong about a frame is a fact about the
> two, and ownership does not move it.**
>
> **This is the fit's arithmetic over the board, and it differs from the worth.** The worth reads a relation
> over the neurons' histories (D57); the election reads the board, where earlier frames' credit stands (R23)
> and this frame's other fits contend (R24). The two numbers differ, and are meant to (D22).
>
> **This is a price for one fit, not for the function.** What the function costs in the dictionary is weighed
> by the worth (D57) and appears nowhere in this price and nowhere in the election.

**The election proposes nothing.** Every fit it chooses among comes from find, over the functions as grouping
left them (R53, R50). The election only accepts or declines one: it never edits a function, merges two, or
invents a third. What it does do is *measure* one, and everything a fit is worth on the board this frame the
machine works out itself (R22).

> **D24 — Coverage set.** The machine keeps, over the board and the frames within reach, who covers what:
>
> | | |
> |---|---|
> | per relation | the accepted call whose fit landed on it, if any |
> | per node | the accepted calls whose fits landed on it, which stand for it, if any (D8); the call it is bound to as an argument, if any |
>
> **Covered, inhibited and bound are read off it.** An activation is **covered** when the covered relations
> of its own, of one kind, reach that kind's count (D34):
>
> | kind | count |
> |---|---|
> | inside the frame | one, in space or at one position |
> | in time | one relation to an earlier frame and one to a later frame, each to any channel |
>
> The counts are separate: a relation inside the frame and a relation in time never make a count together.
> In time one call may cover both relations, or two calls one each. It is covered from the round of the election in
> which a count is reached, and not from the election's end (D28).
>
> **Inside the frame a call covers every activation it landed on.** A relation in space and an association
> count alike. Each activation a fit inside the frame landed on, at a relation that fit covered, is covered
> as the fit is taken. So a row of five pixels under one call is that call and nothing beside it: the two
> end pixels are covered as the three between them are. Each relation of an end pixel to what lies beyond
> the row is lifted to the call (D55). Two activations at one position are covered by one call over their
> association, in the frame they fire. A frame is complete when it fires, so nothing more can come to an
> activation there, and it does not wait.
>
> **In time.** Time is not complete: what follows an activation has not fired. So an activation waits for a
> covered relation on each side. Where the layout has no frame before an activation, one covered relation to
> a later frame is enough (D3). The first activation of a run is such a case: nothing is before it. The
> latest activation is not: the layout does not end after it, so it waits for what follows (D9).
>
> **The first covered relation decides the kind.** From the round of the election in which a relation of an
> activation is first covered, the activation is covered through that relation's kind and no other. Every
> relation of its own of the other kind is **blocked**. A blocked relation is never covered,
> and no fit is credited it. It is not counted among the relations its other end has. So an activation is
> taken inside its frame or in time, never both, and the calls over it are of one kind.
>
> **A deadlock is broken where it would close.** A deadlock is a chain of calls, each sharing an activation
> with the next, where the last shares one with the first (D55). It is of any length. At one level of one
> frame, two activations are **joined** when a chain of relations that the calls taken so far cover leads
> from one to the other. Only relations covered at this level in this frame join them, across every election
> of the level (R24). A relation whose two ends are joined is **refused**, from that round, inside the frame
> or in time. It makes no difference whether a fit offers it, whether any function holds it, whether one call
> joined its ends or several, or whether the function that would cover it is a cycle. A fit's own relations
> do not refuse one another: a function that holds every relation among its nodes covers them with one call.
>
> **A refused relation is eliminated.** It is never covered, and no fit is credited it: a fit that would
> cover nothing else is not taken (D28). It is not counted among the relations either end has. It is never
> lifted, at any level (D55). Where an end was covered before the refusal and the relation had been lifted,
> the relation that stood there instead leaves the board, and every relation that one was lifted to above. A
> relation between two calls that also stands for a relation that is not refused stays, for that relation. A relation stays refused for as long as its
> two activations exist.
>
> **Three at one position.** The first call covers one association and both of its activations. The third
> activation's associations to those two are lifted to the call, and one relation stands there for both
> (D55): the third beside the call, one level up. The frame runs that level before time runs at any (R26),
> so where the relation recurs a function holds it, and its call covers the third activation and the first
> call. With four at one position, two calls cover two associations and all four activations, and the two
> calls stand at one position with one association between them.
>
> Until an activation is
> covered it stands on the board, whatever calls have landed on it, and a neighbor reads it as itself. Once
> covered it has left the apex and the graph (D55). A relation of its own that no call covers is not lost:
> it is **lifted** (D55). It leaves the board, and a relation to each call that stands over the activation
> stands there instead, at the offset from that call's position (D10, D5). Where the call that landed on the
> activation is itself covered, that is the call standing above it, never the covered one (D55). A blocked relation is lifted the
> same way. A refused relation is never lifted.
>
> An activation is **inhibited** when one relation of its own is covered. That is sooner than it is covered,
> and may be in the frame it fires. The calls over it are every call that covers a relation of its own, and
> every call over one of those, at any height. An inhibited activation is not heard in a neuron dimension
> while any call over it has an inference for the frame ahead in that dimension. Where none has, it is heard
> there (D10, R48). The dimension meant is a neuron dimension, what is inferred, an action or an event (D1).
> It is not an activation dimension such as time, `x` or `y`.
>
> A call is covered and inhibited the same way, one level up, and its kind is decided the same way: an
> association between two calls counts as one between two base activations does. An activation is **bound**
> when a call binds it
> as an argument: it stays on the apex, its binding is paid with the call, and that call does not inhibit it
> (D10). It is on the apex, bare, when none of these holds.
>
> **Credit is what blocks later payment, and coverage is what hides.** A relation a call covers is paid for
> through that call, and no later fit is credited it. A later fit lands on the activation's other relations
> while it stands on the board, and on its call's once it is covered, and is credited those alone (D5, R23). A bound activation is
> still on the apex and is a neighbor to every later function. A later fit that lands on it covers the relations of its own
> that no call has covered, and nothing else (D10). So every relation is paid for once. The coverage set spans one
> frame more than the longest reach in time among the neurons the machine holds, no function reaches further
> back, which is the span an activation stands on the board (D9), and the box every other activation dimension
> gives, and ages out with it, and over the same span the
> machine keeps the action that ran at each frame in each action channel, for the shares a reward sends back
> (R33). **The machine holds
> nothing on the scale of the run.**
>
> The coverage set is about **credit**: a relation is a fact that needs paying for exactly once, so one a call
> covers is paid for through that call from then on (R23). It is not about naming: an
> interneuron expands to everything its function names, credited or not (R24).

> **R23 — This frame's fits against the board as it stands.** Only what costs is in play: a bare node's
> relations at their cost, a bound node's binding, and the relations a call covers at that call's cost,
> together. So no relation is paid for twice. Covering takes effect inside the election, as each fit is accepted
> (D28). Inside the frame an accepted fit covers every activation it landed on. In time an activation an
> accepted fit has landed on stays on the board while it is not covered, and another
> fit of that election may land on it and cover a relation of its own. Once the accepted fits cover it, it is
> off the board: no later fit of that election lands on it, each relation of its own that no call covers is
> lifted to the calls that landed on it, and only those calls stand there (D24, D55). A later election of a level
> fits what the ones before it left, so a relation they covered is not there for it, nor an activation they covered.
>
> **A later frame reads the past as it stands on the board.** An activation of an earlier frame that is not
> covered is returned as itself, and a fit at `f` lands on the relations to it that no call has covered. A
> covered one is not returned; the calls that landed on it are (D5, D24). Where the relation read is the call's own function going round its cycle, the call takes
> the lap and no new call is made (D53). What that interneuron learned stands.

**The cost over one frame is the calls made plus the relations no call covered**: the price of every
accepted fit, and one for each relation on the board no call covers, the graph's half of the objective (§3.12)
over the frames the election can see. **The two terms do not overlap**: a neuron a promoted neuron fails to name is
in the second and not the first, which is why the price takes only the neighbors named and absent (R22). The
dictionary half is the worth's (D57), and neither test touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of
fits and scores it**: fits are taken one at a time, each measured against what the ones before it left, and
the election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy,
and returns a good solution, not a proved minimum. **Every activation a fit covers ends up credited to exactly
one fit**, which is what stops an activation being paid for twice.

> **R24 — The election is D28, run by the machine.** It runs for the frame at every level, and after that
> for time at every level, over the fits of that kind (R26). In each it runs over three sets of fits. First, everything find fit at this
> level, over the dictionary as it stands (R53). Second, the fits of the functions grouping changed, on what
> the first left (R50, R52). Third, the fits of each function that starts, one function at a time (R50). Each
> time, less
> any whose function would need a neuron when none is free (R16), and what they cover is the **free set**:
> every activation on the board that some fit lands on and no accepted call covers, with its relations
> (D54).
>
> Each fit stands on the board before the first round (D31), and `price` is R22's. Then D28 runs: the largest
> `coverage` less `price` first, each while `coverage > price` against the board as it then stands, and it
> stops when nothing on the board is left that a fit could cover for less than it costs.
>
> **The frame is settled before time.** The frame runs first, up all its levels: its fits, its moves, and the
> functions that start inside it. Only then does anything run for time, on the board as the frame left it. What
> stands with an activation in its frame is all there when the frame fires, and what follows it is still to
> come (D34). So nothing in time is fit, joined, moved or started at any level while anything inside the
> frame is left to do at any level. A relation inside the frame is therefore never blocked by a fit in time
> before it has had its turn to start a function. That holds for a relation lifted to a call, whose turn is
> one level up: an activation left beside a call that covered its neighbors has that relation inside the
> frame, and the frame reaches it before time reaches the activation. Once a fit inside the frame
> covers a relation of an activation, that activation is taken inside the frame, and its relations in time are
> blocked (D24). They are lifted to the calls, and the level above reads them there (D55).
>
> **Two fits that measure the same are told apart in this order**: the fit whose
> calls stand for more of the same base activations, since an activation that stood under two calls then stands
> under one (D10); a function with an interneuron before one without; between two with interneurons, the older
> interneuron; between two without, the function that joined the dictionary first; and between two fits of one
> function, the one reached from the older activation.
>
> **The bound is structural.** A fit is taken only while it covers more than it costs. A pair whose relation
> came true covers one for nothing, so a pair in the dictionary is taken where it stands bare. Each round
> takes one fit, so the rounds are never more than the fits.
>
> **A fit that never reached the top held nothing.** There is nothing to hand back and nothing to settle: an
> activation it landed on is either covered by a fit that did pay or stands bare.
>
> **Ownership is a partition of the relations the accepted fits land on**, and that is the whole of the
> inhibition: no function is ever edited or forbidden. An activation is not owned: several calls may land on
> one (D8), each covering relations of its own, and the calls are neighbors at the level above (D26). A fit that lands on
> a relation an earlier round covered is read again without it (D28), and one that binds a bound node again
> gains nothing (D10, R22). **Covered by an accepted fit** and **landed on by an accepted fit** are therefore
> the same set, so coverage, credit and the apex frontier (R27) are one question with one answer.
>
> **Outcome**: accepted fits are promoted, each **whole**, and an interneuron expands to everything its function
> names, credited or not; the activations they land on as fixed nodes leave the apex, the calls standing in
> their position with what the fits did not cover, and what they bind stays on it (D24), and every active neuron
> no accepted fit lands on stands as itself. **The election delivers nothing to
> anyone**: it writes the coverage set and stops. No neuron is told what was bought; a neuron's history is what
> it saw, and the board is the machine's.

## 6.3 Allocate interneurons

Every accepted fit of a function with no interneuron gets the function one where a neuron is free. A fit that
lost is given nothing.

> **R16 — What an interneuron is when allocated.** The machine allocates a function's call neuron the first time
> a fit of the function is accepted, where a neuron is free (§3), unless a stretch comes round and an
> interneuron already stands for it (R43). A new interneuron has an id and no dimension (D2), and holds nothing:
> no history, no relation index, no connections. Everything it comes to hold is over the situations its function
> actually took (D25), from its first activation on. Its *existence* is decided by the election, its *structure*
> by its function.
>
> **No free neuron, no interneuron.** Where every neuron is allocated, a function with no interneuron cannot be
> granted one, so its fits are not in the election (R24). It stays in the dictionary on its worth, is fit and
> priced as before, and is elected again when a neuron is free.
>
> **Release is the reverse**: the function leaves the dictionary, and the machine decommissions its interneuron
> once no other function of it remains (R38).
>
> **Allocating is immediate.** The machine allocates or reuses the interneuron the moment the fit is accepted,
> which is all the next level needs to activate it in the fit's phase (D53) and call it.

> **R43 — Reuse.** An accepted fit fires its function's own interneuron where it has one. Where it has none,
> the machine looks the function up in the interneuron index, by what its interneuron would stand for (D36).
> Where one already stands for that, the function is added to it, and that interneuron fires. Otherwise the
> machine allocates one. **An interneuron never holds two functions whose nodes are the same poses**: the
> two expand to one stretch, and when the second comes the older gives way, its relations leaving with it
> (R54, R18). So the two neurons a phase names pick out one function of the interneuron, and a call in that phase
> expands through it (R28).
>
> **Otherwise the machine allocates one** (R16): a free neuron, its first activation at the fit, in its phase
> (D53). Where none is free the fit was not in the election (R24).

## 6.4 Activate interneurons

An interneuron fires at the level after the one its fit was accepted at, before that level is called. It fires
once per accepted fit, at the position of the latest activation the fit landed on, at age 0, in the phase the fit
read (D53), and what stood in its function's variables stays on the apex beside it, bound (D10). Nothing sees an
interneuron until it fires (D5). An interneuron's activation records the function that fired it, since it may
have several (R43), its phase, which of that function's relations did not come true, and what its fit landed on
and bound, credited or not, which is what it expands to (R28). Every activation no accepted fit covers stands as itself.

> **R27 — The apex is a frontier.** It is every active neuron **that is not covered** (D49, D24), base neurons and interneurons alike, so a base neuron no function found worth naming
> stands in it beside an interneuron many functions deep. This is the frontier the cost is read over (D12), **the one every
> neuron sees as its neighborhood** (D5), **the one that learns what ran** (D25), and **the one that votes**
> (§8.4): the uncovered set does all four, and covering a neuron silences it in every one of them at
> once, its one last connection aside (D10). A reward for a frame
> already priced reaches its connection regardless (R33). Everything underneath the current frontier is
> recovered by expanding it.
>
> **Named, not covered.** A neuron a declined fit named, with nothing else covering it, is still on the
> frontier, and stands in the cost as itself.
>
> **Coverage is acquired late.** An activation that fires at `g` is on the apex until it is covered
> (D24), and a fit can land on it for as long as it stands on the board (D9). It is
> inhibited sooner, from the frame a call first lands on it, which may be the frame it fires (D24). A later function may name the
> call that covered it (D10); it does not return to the apex.

The frontier cuts across the hierarchy:

```
                                    ┌────── ▣ ──────┐
                          ┌─── ▣ ───┐               │
                ┌─ ▣ ─┐   │         │       │       │
                a     b   c         d   e   f   g   h        i     j
                                                             ▣     ▣

   frontier  =  { the interneuron over (a,b),  the interneuron over (c,d),  the interneuron over (e,f,g,h),  i,  j }
```

**Events and actions run together within a level.** An action fires in the same column as the events it runs
alongside (D8), so it is recognized and covered by the rule they are, in the same functions (D5). **The
connection is not formed here**: it names what stands on the apex, which is known only once the frame's last
level has settled, so it is recorded in the `connect` call instead (R31).

---

## 6.5 Decommission interneurons

> **R38 — An interneuron dies when its last function leaves the dictionary and its last activation leaves
> the board.** Dying is decommissioning: the neuron is free to be allocated again. Once its last function has
> left, nothing can fire it again. Every function that names it as a node leaves the dictionary with it, since
> it can no longer hold, and so on up: that is its **subtree**. The death frame is the frame in which the last
> activation of the interneuron, or of any neuron in its subtree, leaves the board, and this frame when none
> stands on it. That set only shrinks. An activation stands on the board through the machine's longest reach
> in time (D9), which is longer than it connects, so a neuron is never allocated again while a fit could still
> land on one of its activations or an interneuron could still see it.
>
> **Decommissioning is after the frame's connect and infer calls** (§7), so the last exposure an activation
> writes at age `reach_t` is taken before its neuron goes. Every frame, after those calls, the machine takes
> everything due, and calls forget neuron on every neuron whose history names one (§4.1).
>
> **A call outlives its function on the board.** Until the activations a function fired have left the board,
> calls above them may still expand them; expansion reads what they covered, not the dictionary (R28), so the
> function need not be kept for it.

## 6.6 Cost per level

With `G` relations returned per activation, `F` functions reached per relation, `L` relations per function and
`C` candidates attached per fit:

| Step | Work |
|---|---|
| find | per relation, three index reads, widened by the classes its far neuron is in; per function reached, `L` relations matched per fit |
| count | per neuron, per function it stands in, fit or not, `L + C` counts, each a pass over its `H` rows |
| group | per candidate, the worth of every relation of the function it would join, with it and without it, and the same for the function it stands in; one worth per relation of each function fit |
| elect | the fits, a greedy cover over the board, stopping when it is covered |

Nothing writes a row and nothing waits on a merge. Everything is bounded by what the level returned and by `H`,
and nothing scales with the run. What is cached, and how, is the implementation's.

## 6.7 Cycles

Two things that are cycles, and one that is not, told apart, because the design holds them apart.

**Within a function, a cycle is the pattern repeating.** A function has one node per pose, so in time a relation back to
a node already in it closes the cycle. Inside the frame it closes the cycle only where it reaches the
activation the fit landed on at that node (D15). The function is then that cycle and nothing else; the function says that the pattern comes
round: `x` to `y` at 1 and `y` to `x` at 1 is `x y x y …` for as long as it runs (D15). A fit at either letter reads back
round the cycle while the relation that closes it lands; where the call of the lap before stands there, that
call takes the lap (D54); nothing is read past the cycle, and the call's
phase says how far it went (D53). What comes after
the pattern is no part of the cycle's function: it sees the call, and a function at the level above holds
its relation to the call (R50).

**Through interneurons, recursion.** A period no single function can hold, because the same neurons come in an
order a graph over them does not fix, is held as a chain: each function "this symbol, and the call before it",
two distinct neurons, each covering the call before it, and the last naming the first through the dictionary.
Six functions for six bars whose moves repeat. That is a cycle in the dictionary and not in any function:
unrolled over the run it is a chain, since the call at one frame and at the next are different nodes.
Expansion follows what fired, frame by frame, and stops at reach (R28), so it never runs round. It is how a
periodic world is covered for a fixed dictionary: each call covers the one before it, the whole run stands as
one node at the end, and the dictionary says the rest, with one failure wherever the chain breaks.

**A deadlock is not a cycle.** A cycle is a function: its own relations lead back to where they started, one
call covers it, and the machine keeps it. A deadlock is made of the calls of different fits in one frame: each
shares an activation with the next, and the last with the first (D55). No function holds it. Three calls round
three activations come back, one level up, as three calls round three calls, and so without
end. The election never lets one close: the relation that would close it is refused and eliminated (D24).
Inside the frame a call covers every activation it landed on, so two calls there never share one (D24).

# Part IV — Future: a neuron

# 7. Connect and infer calls

Once the frame's last level has run, the machine calls every open activation once more, and what the neuron does
with that call is the whole of this part. Which call it is depends on the activation's age, and the two never
meet on one visit:

| Call    | Description                                                                                                                            |
|---------|----------------------------------------------------------------------------------------------------------------------------------------|
| connect | At every age from 1 to `reach_t`: the activation connects to what stands on the apex this frame, with any reward at its distance (R31, R33). It returns nothing (§7.1). |
| infer   | At age 0, and only then: the activation returns what its neuron is connected to at every offset, each with its strength and estimate: its inferences (R36). It writes nothing (§7.2). |

**Nothing is written at age 0 and nothing is returned after it.** A connection's offset is the activation's age,
and an offset is strictly positive in time (D25), so the visit at age 0 has nothing to write, and a reward
share at distance `d` lands at offset `a − d`, which at age 0 is offset 0 again (R33). The visits after age 0
have nothing to say: the inferences are read once, in the frame the activation fires (R36).

The neuron chooses nothing in either call and is told nothing of what came of its inferences. Choosing among the
inferences of every activation is the machine's, and is selection.

> **R29 — Two frames: infer, then execute and reward.** What is chosen in one frame is output for the next, and
> what ran, with what it earned, arrives with that frame.
>
> | Frame | | |
> |---|---|---|
> | `f` | infer | the frame's events are recognized, `infer` returns the inferences, the inference resolves (R47), and an action is output for the frame ahead |
> | `f + 1` | execute | the report names what ran, and its neuron fires strong in this frame's column |
> | `f + 1` | reward | alongside this frame's events, every uncovered activation open connects to the apex; what the action earned arrives as this frame's input, and the connection it strengthens takes the reward in the same write (R31) |
>
> **The reward is part of the frame the action ran in.** The environment reports what it observed and what the
> action in effect during that frame earned together, so an action is never on the books without its outcome,
> and no activation has to stay open for a reward that arrives later than the action it pays for (D9).

## 7.1 Connection

**The machine calls `connect` on every open activation at age 1 and beyond**, once per frame of its window, with
two things:

| | |
|---|---|
| the apex | every activation standing on the apex this frame within the activation's reach in every dimension but time (R27, D25). The activation's row takes each at its age, as what followed it, covered or not (D18). Its connections take them only while it is uncovered: one an accepted function covered this frame connects to the neuron that covered it and to nothing else, and one covered earlier connects to nothing more (D10, D25, R31). |
| the reward | any reward that arrived, for the actions of this frame and for any earlier frame the reward spans, at the distance each one names, with the action that ran at that frame in each channel it pays (D24, R33). |

The neuron writes each apex activation into the row at its offset (D18) and, while uncovered, strengthens a
connection per apex activation: the connection at `(neuron, age)` under its own phase, created at strength 1 or
incremented (R31), and
for each reward share the earnings of the connections it names (R33). For this frame, the connections just
strengthened, in the same write. A share for an earlier frame, at distance `d`, goes into the connection at offset
`a − d`, the activation's age less the distance, to the action the machine hands it for that frame, where the
activation was uncovered then (D9). The activation keeps nothing of it. **Nothing is decided, priced or compared
here**, and no test is waiting on any of it.

> **R31 — A connection carries an estimate.** What stands on the apex at `f + 1` is not known at `f`: the action
> is settled only when the report names what ran, and the events only when the environment reports them (D40).
> So at `f + 1`, once the frame's last level has run, **every uncovered activation open at that frame connects to
> every strong activation standing on the apex of that frame within its reach that the policy admits** (R27,
> D25, D50) at its own age (§7.1), and the reward of that frame's actions goes into
> each. An activation covered in that frame connects once more, to the neuron that covered it where that is
> strong, and to nothing else (D10). A weak activation is connected to by nothing: it is a proposal, not a fact.
> That connection binds what the neuron stands for to what followed, formed against what actually stood, so
> **a neuron that inferred something else, or nothing, learns from what happened.**
>
> **A call connects to what follows it like any activation**, its own function's later nodes included once a
> longer function holds them: a connection and a relation to the same node are one fact priced in two
> currencies, the relation in the cost and the connection in the reward, and the reward is learned on the
> connection (R34).
>
> **One activation connects to the apex of every frame it is open through, and one strong apex activation is
> connected to by every uncovered activation open when it stood.** The offset is the age, the distance from the
> activation's position to the frame the call ran, exact (D6), so a neuron open at ages 1, 2 and 3 holds the same
> neuron at three offsets, and the exposure at each age strengthens its own. A neuron that stood at two ages
> over one activation is two exposures: each frame has its own reward, and each connection keeps its own (D25).
> A connection to a call names it in its phase and is at its offset (D34).
>
> **An exposure is written at its age, and an inference stands at its offset.** An activation that saw a
> neuron at age 2 strengthens the connection at offset 2, and an inference at that offset stands exactly 2
> frames after the activation that reads it (R36). So a replay of what was seen at age 2 completes at 2, and
> what it earns lands on the connection it was learned from. Fan-out is the apex within reach: a neuron connects
> to what stood uncovered in the frames after it, as far as it sees in every other dimension, and to nothing
> beneath that, and one exposure per apex activation per frame means a neuron of reach `r` in time holds at
> most `r` offsets per neuron it has seen.
>
> **Making and strengthening are one operation.** A neuron's connection at `(neuron, offset)` has a
> **strength**, the number of its exposures, the times an activation of the neuron saw that neuron on the apex
> at that offset, and its **earnings**, the total reward those exposures received (R33). The first exposure
> creates the connection at strength 1 and earnings 0 and every later one adds 1 to the strength; each share of
> reward adds to the earnings. The estimate is the earnings over the strength, the exact average over the
> connection's exposures, and no rate is chosen. **Nothing leaves.** No exposure is ever subtracted, no strength
> ever falls, and no window bounds a connection: it is the whole of what the neuron has seen follow it at that
> offset, over its life.
>
> **Every neuron holds them, base neurons included.** No operation derives what an interneuron was worth from
> what the neurons under it earned or the reverse, so a connection held only at the frontier would be lost at
> the next allocation. Every neuron keeps its own connections over its own activations (D25).
>
> **A connection dies with either of its ends.** Nothing else removes one (R34).
>
> **A connection wired ahead of any exposure is created at strength 1 and earnings 0.** The walk (R37) is the
> one thing that creates a connection nothing has yet been seen to follow; it is created exactly as a first
> exposure at neutral reward would create it, and from then on it is a connection like any other.

> **R32 — What is learned is what happened.** No inference is credited and nothing is in control: what stood on
> the apex is what every uncovered activation connects to and what its reward lands on (R31), whether that
> activation inferred it, inferred something else, or inferred nothing. Before any function exists the apex is
> the base symbols, so the rule holds across all of development.
>
> **A function earns what its program earned.** It fires when its expansion completes (R30), so what arrived in
> that frame lands on it directly, and what arrived over the frames its program occupied reaches it through the
> span a reward names (R33). That is what makes a multi-frame candidate comparable to a single-frame one.

> **R33 — A reward names what it pays for, and the machine fills in the rest.** A reward (D35) pays the
> channels it names over the span it names, every channel and the whole window when it names none.
>
> **Distance is counted back from the frame the reward arrives in**, `0` being that frame itself, the frame
> whose action the reward is for in the two-frame cycle (R29). **What reaches distance `d` falls linearly with
> `d`**, because the further back a frame is the less likely it is to have caused what arrived. Over a span of
> `S` frames the share at distance `d` is `reward · (S − d) / S`, for `d` from `0` to `S − 1`, so the frame the
> reward arrived in takes the whole reward, the far end of the span takes `reward / S`, and nothing along the
> way takes nothing. A span of one frame is the degenerate case, `share(0) = reward`, and it is what an
> environment that pays each frame's action as it goes reports.
>
> **The whole window is the longest open one.** When no span is named, `S` is the longest reach in time among
> the neurons the machine holds: an activation of that neuron is open through `S` frames (D9), so the furthest
> back any open activation could have seen an action run is `S − 1` frames, and no share can reach further than
> the connections that would take it. With base neurons alone that is `S = 1`, the degenerate case above.
>
> **Nothing is being divided up.** Every distance in the span is paid, and the shares are not a partition of the
> reward: a reward is not in short supply, and the point of spreading it is not to conserve it but to say how
> likely each frame is to have earned it. What is genuinely responsible recurs and accumulates; what is not is
> sampled once and averaged away. **Each share is delivered, in the `connect` call (§7.1), to every open
> activation past age 0, and the activation picks the connection from its age and the reward's channels.** The
> machine hands it, for the frame each distance names, the action that ran there in each channel the reward
> pays, from the window it keeps (D24). An activation at age `a` takes the share at distance `d`, `d` at most
> `a`, into its neuron's connection at offset `a − d`, to that action, where it was uncovered at that frame
> (D9): at distance `0` the connection this call strengthens, in the same write; further back one strengthened
> `d` frames ago (R31). An activation whose window has closed is not called, so a share reaches no further back
> than the open activations that would take it.
>
> **A scoped reward reaches base actions of the channels it names, and nothing else.** A connection names a
> neuron (D25). A share of a reward that names channels is delivered only to a connection whose neuron is a base
> action of one of them; a connection to an event or to an interneuron takes none of it. An interneuron belongs
> to no channel (D2), and a share read through its expansion would give a call spanning two channels one
> estimate, read in both at resolution (R47). An unscoped reward reaches every connection written at the frames
> it pays. A machine whose policy lets interneurons be connected to (D50) and whose rewards name channels leaves
> those connections unpriced by them; the two are not declared together.
>
> **Rewards in one frame are independent.** A connection at one `(neuron, offset)` takes the sum of the shares
> that reach it, and nothing coordinates one reward with another.
>
> **The unscoped form is the general case, and the machine sorts out the attribution itself.** An environment
> that knows what it is paying for names it, a stock channel, and the frames a position was held, and the credit
> is exact. One that does not names nothing, and the reward spreads over every channel and the whole window: the
> shares landing on a channel or a distance that had nothing to do with the outcome are noise around zero and
> average out over exposures, while the shares landing on the ones that did accumulate. **No structure is priced
> on either** (R34), so a coarse reward costs accuracy in the estimate and nothing in the cost. What it does
> move is the walk: an unscoped reward carries its sign into every estimate it reaches, so the sign test R37
> reads shifts the same way for all of them, and the walk runs while the world is going badly overall and rests
> while it is going well. That is the meaning of a signed reward, and it is intended.

> **R34 — Two objectives, meeting at one point.** Everything structural is priced in cost; reward prices
> nothing structural and cannot. The machine runs **two** objectives: compression, which decides what structure
> exists, and reward, which decides which of it is executed. They meet at exactly one position, a neuron's action
> connections, and connections enter no cost and are not priced by the one valuation.
>
> **Estimates are not decayed, and not windowed.** A connection's estimate is its earnings over its strength,
> the plain average over every exposure it has had, with no cap, no rate and no horizon; nothing ever leaves it
> (R31). **Non-choice moves nothing**: an action not taken keeps the value it had. What makes an estimate
> specific is the neuron that holds it, not how recent its exposures are: an interneuron's connections are over
> the one situation its function names (R35), and a changed situation is answered by a new interneuron, never by
> forgetting.

## 7.2 Inference

**At age 0, and only then, the machine calls `infer`, and the call returns what the activation infers.** It
reads its own neuron's connections at every offset, out to its reach, and returns each with its strength and
estimate, covered or not. Those are its **inferences** (R36), each naming a neuron, base or interneuron (D25),
and the machine keeps them for as long as the activation is open (R45); at every later age the machine calls
`connect` instead (§7.1), which delivers and asks nothing. A base neuron infers from its own connections like
any other; what it infers is the marginal over every situation it fires in, and it is heard only because
nothing more specific has landed on it. A newly allocated
interneuron has no connections and infers nothing: what it landed on speaks for it (R48).

**An activation's window closes at age `reach_t`** (D9), once that frame's exposure is taken. The window closing
stops the connect calls and drops its standing inferences (R45), and nothing else: no call marks it and nothing
is saved twice.

> **R36 — Inference.** One connection read by one activation at age 0: it names a neuron, base or interneuron
> (D25), and carries the strength and the estimate the connection held then. An activation reads every
> connection its neuron holds, at every offset out to its reach, once, in the frame it fires (§7.2), and never
> again: what the walk wires or a reward moves afterwards is read by the neuron's next activation. The winner of
> every dimension is chosen from the inferences and nothing else (R47).
>
> **An inference stands at its completion, and a connection means what it recorded.** A connection at offset
> `b` was written when that neuron stood on the apex `b` frames after the activation's frame (R31), so an
> inference at offset `b` stands `b` frames after the activation that made it, and at the offset's components
> in every other activation dimension from its position, as a neighbor is read (R28). An inference that names a
> base symbol infers it there. One that names a call neuron infers a call there, in the phase it names, weak
> (D37), and the call's function hangs from it (R28): its nodes at the frames back from it, every one carrying
> the strength and the estimate of the connection it came from, its parameters bound from what stands at their
> nodes where those have happened (D44). What its expansion puts at the frame ahead is what it proposes there,
> and the machine expands no more of it than that (R45): at distance 1 the last thing a function reaches, at
> distance ten the one ten from its end, and the first only when its completion stands its whole length
> ahead. An inference is proposed only where its match holds (R49), and resolution runs at the base and only
> there (R47). **This is what launches a program**, and it is why every offset is read and not the next one
> alone: an activation can start a program only from an offset at least as far out as the program is long, and
> can carry the tail of a longer one to completion from any offset. A situation whose reach is shorter than a
> program never starts it, and that is correct: it cannot see that far.
>
> **Two positions are two voters.** Two activations of one neuron read one set of connections (D11) and name the
> same inference from two positions: in a channel whose only activation dimension is time that is one proposal
> twice, and in a channel with a layout it is a proposal at each of two positions, each resolved where it lands
> (R47).

# Part V — Future: the machine

# 8. Selection

Once every open activation has been called (§7), the machine holds what the apex infers, and it works from
there down to the base alphabet, in this order and no other:

| Step     | Description                                                                                                                                  |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| keep     | Hold every inference an activation returned at age 0 as a standing inference, until that activation's window closes (R45).                     |
| expand   | Expand of every standing inference what comes due: the base symbols it expands to at the frame ahead are what it proposes (R28).                   |
| match    | Read each standing inference's function against the frames it reaches back into; propose it where the function would pay there, strike it where it would not (R49). |
| resolve  | One winner per dimension at the frame ahead: in each action dimension by estimate, in each event dimension by share of voters (R47).           |
| activate | The winner of every dimension fires weakly at the frame ahead (D40), action and event alike, and both are output; the world's report replaces them, in part or in whole, and what ran arrives with its reward (R40). |

No neuron takes part in any of it. What an activation returned is kept, fit, expanded, matched, resolved and
fired by the machine alone, as the election is the machine's alone (R24), and a neuron learns of the outcome only what
happened (R32).

## 8.1 Standing inferences

> **R45 — Standing inferences.** The machine holds every inference of every open activation, from the
> activation's age 0 until its window closes (D9), and nothing is read twice: these are the **standing
> inferences**, and they are what votes (R46). Each is held as what its connection named, the neuron and its
> completion, with the strength and the estimate it was read with, and the machine expands of it only what
> comes due: at age 0 the part that falls at or before the frame, which is matched against what stood (R49),
> and in each frame after the part that lands at the frame ahead, which is what it proposes there (R47). A
> program a hundred frames long is one standing inference, unrolled a frame at a time. **A standing inference is
> dropped** when its activation's window closes, and when its match fails (R49): a plan holds because it keeps
> being what happened, and nothing is retracted. An inference stands whether its activation is inhibited or not;
> inhibition decides whether it is heard (R46, D24).
>
> **What stands is what was read.** A reward may move an estimate after an activation's age 0, and the walk may
> wire a connection. The neuron's next activation reads them (R36). This one does not. A situation that launched
> a plan keeps proposing it at the estimate it read, until its window closes or its match fails. The rewards the
> plan's actions bring as they run fold into the connection's estimate on the neuron (R31). The standing
> inference does not read it again.

## 8.2 Expansion

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> what its function covered and bound, down to base symbols.
>
> **A recognized call expands to what its fit landed on.** The machine holds, with every call on the board,
> the activations its fit landed on and bound, credited or not (D9, R24). Expanding the call is reading them:
> each landed activation that is an interneuron is expanded the same way, through what its own fit landed on,
> and each bound activation is the argument, the neuron that stood there (D37). Nothing is read from the dictionary,
> so a function that has changed since the call fired changes nothing about what the call expands to:
> expansion follows what actually fired, frame by frame, back to where it began, and it ends there, even
> where a function comes back to its own interneuron (§6.7).
>
> **An inferred call expands through its function**, since it covers nothing yet (R36): the function of its
> interneuron whose nodes its phase names (R43). It follows the function's relations back from the node it
> entered at, one dimension at a time:
>
> 1. **Start** at the node the call's phase starts at (R36, D37), at the call's position in every dimension.
> 2. **Follow** each relation from an expanded node to the node at its far end, at that relation's offset, to
>    the node the phase ends at. Offsets compose because each is a difference (D34). Round a cycle the laps the
>    phase says; where the phase leaves the laps open, round while the node it reaches has something to land on in
>    the frames that have happened, and beyond them once more (D54, D15). Repeat until no relation leads to a
>    node not yet expanded, in any dimension.
> 3. **Expand** each expanded node that is an interneuron the same way, through its function, in its own phase.
>    Expand each variable to its argument where one is bound, and otherwise as the next paragraph says.
>
> A fixed node expands to its neuron; a variable expands to its argument.
>
> **Every expansion is exact.** A node is read at exactly the offset its relation names, composed along the path
> from the node the expansion started at, and an expansion many levels deep reads its farthest base symbols
> exactly where the relations along the path put them. Nothing is rounded on the way down.
>
> **Fitting and expansion are one rule read in two directions.** Fitting (D31) lands the function's relations on
> the board from the node the fit enters at. Expansion starts at a node and lands the relations back on the board.
> Both read the function and what the call recorded and nothing else, so what one covered the other recovers.
>
> **This is the one expansion in the design.** It recovers the run from the dictionary and the calls (D12): a
> class to the member bound at its node, a parameter to the neuron bound at its node. And it turns an inferred
> function into a program (R30): the actions it expands to run, and the events it expands to fire weakly beside them
> (D40). It reads the dictionary only. **What travels down with a symbol is what the connection carried**: every
> base symbol an inference's expansion reads carries the strength and the estimate of the connection it came
> from, and nothing is re-weighted on the way down.
>
> **Expansion substitutes the arguments.** A recognized call's arguments stand on the apex, bound, and need no
> substituting. An inferred call brought none, so its variables are bound from what has happened: a parameter
> from what stands at its node where that has happened, and expanded where the node is reached again, which is
> how a value is carried forward; where it has not happened it expands to its most frequent member. A class with no
> argument expands to its most frequent member, the oldest on a tie. A function above that names an interneuron
> alone expands to the call with its arguments left to be bound, and one that names the interneuron and a bound
> activation expands to the call with that argument. **Two words, never one.** A function is **expanded**,
> downward, into what it covers; a connection is **inferred**, forward, into the frames ahead (D25); what a call
> does and what follows it are different directions.
>
> **A connection is read the way a neighbor is.** A connection at offset `b`, read by an activation at its age
> 0 (R36), puts the neuron it names `b` frames after the activation's frame, that is where it completes, and its
> expansion hangs from there, its base symbols at the frames back from it. At a later age `a`, the frames
> elapsed (D9), that completion is `b − a` frames ahead. So the base symbols of a long program reach the frame
> ahead one at a time, in order, each at exactly one age, from one connection and with nothing held: what an
> activation infers beyond the frame ahead it infers again next frame, one frame nearer.

> **R30 — Execution is an expansion**, of the selected function (R47) through its nodes and relations in the
> dictionary (R28). A high function becomes its nodes at the offsets it records, down to base symbols. Its base
> actions are output, each in the frame its expansion reads it in, the nearest being `+1` (R29); its base
> events stand in theirs. Both fire weakly (D40), and the report replaces them: the actions it names as run
> and the events it names as seen become strong, and the rest last their frame. Execution is not a second
> mechanism; it is this expansion read as a program. A weak function's expansion reads no action: what is
> output is chosen by estimate, never by a function that was itself inferred.
>
> **Execution activates what it expands, where it won.** A base action or a base event fires weakly in the frame
> R47 selected it, and an interneuron the expansion passed through fires weakly when its expansion completes, at
> its position, the last frame of what it covers, which is the position recognition would have given it, provided the
> standing inference that named it still stands (R45) and what it named at that frame was selected (R47). An
> inference that lost its dimensions, or whose match failed on the way, fires nothing. Where the report makes
> its actions strong, recognition builds the strong activation of the same function at the same position, and it
> replaces the weak one (D40). An executed function is therefore in its frame's levels like a bought one: it is
> activated, may be covered (§6.1), and every uncovered activation open at that frame connects to it (D25).
> While its program runs, what stands on the apex at each frame is the base action it named there, or a lower
> function as it completes; the whole stands on the apex only in the frame it completes, and what the program
> earned reaches it through the span a reward names (R33).

## 8.3 Match

> **R49 — Match.** The machine reads one standing inference (R45) against the frames its function reaches back
> into, and proposes it where the function would pay there and strikes it where it would not. It runs every
> frame the inference stands, over the part of the function that has come to lie in the past. Nothing competes
> in it, nothing is credited and nothing is covered: it is the one valuation (D22) read over its third set
> (§3.10).
>
> **What is read.** The inferred interneuron's function, as it stands in the dictionary (R28): its fixed nodes,
> each at an offset, and its variables, each at its node, read back from the completion the inference names,
> one expansion down and no deeper. Whatever is read at the frame ahead or beyond is what the inference
> proposes, and is not judged. Against them stands the apex of each frame within the longest reach in time, as
> the machine keeps it (D24).
>
> **What each expanded node is worth.**
>
> | Expanded node | Worth |
> |---|---|
> | a neighbor standing at its offset | credits what it cost |
> | a neighbor standing nowhere at its offset | charges its relation, one failure (D13) |
> | a variable | bound from what stands at its node where that has happened (D44), crediting its relations; a node holding nothing it could take charges its relation |
>
> An offset is read exactly (D6): a neighbor stands at its offset or it does not, and a neuron of the right
> kind at another position is not it.
>
> **The test.** The inference is proposed when what the expanded nodes credit exceeds what they charge, the
> relations that did not come true included, and struck otherwise. The call's cost is not charged: the match
> covers nothing, it only asks whether the frames that have happened bear the inference out, so it is D22's test
> over the nodes that have happened, with B holding the relations that did not come true and nothing else.
>
> **What it decides.** Proposed, the inference's nodes ahead contend for their positions (R46, R47). Struck,
> it leaves the standing inferences (R45). An inference that was proposed and stops matching as its nodes
> come due is struck then, which is the whole of how a plan ends before its span does.
>
> **Where it stands among the three.** The election reads the fits against the board and credits them
> (R24). The worth reads a relation against the neurons' histories (D57). The match reads one inferred
> function against the frames it points back into and admits or strikes it. The prices are the same; the
> questions are not.

## 8.4 Resolution

> **R35 — Selection.** No fit says which action to take; it says only what a situation was followed by.
> Choosing comes from the connections of the neurons standing on the apex, each carrying the reward that arrived
> averaged over its exposures, and **the machine outputs the best. Nothing else decides it.**
>
> **A situation is the apex** (D49): what stands uncovered at the frame, base neurons and interneurons alike. No
> neuron stands for it whole unless a function has covered it, so it is read a neuron at a time: a voter is one
> activation of it, and what it holds is its own neuron's record of what followed.
>
> **A voter is one apex activation that is heard** (D24), speaking from the connections its neuron held when it fired (R36). A base
> neuron's estimate is a marginal over every situation it has stood in; an interneuron's is over those its
> function recurs in, one situation; so the same frame reads
> differently to a base neuron, to the interneuron covering part of it and to the interneuron covering the
> whole. **A function's interneuron is how one recurring situation acquires an estimate of its own**, which is
> why an inhibited neuron is silenced (D24) and the call over it speaks instead, and the only reason a base neuron's
> marginal ever speaks is that nothing more specific was bought over it.
>
> **Every apex activation that is heard votes**, base actions included. A function that names what was done beside what was
> seen votes knowing the situation; an interneuron over actions alone, or a base action, votes what followed it,
> a habit, and reward prices the habit like anything else. What an inference names in an event dimension is
> resolved the same way, and the winner is output (R47, R30).
>
> **Nothing runs where nothing infers.** An action dimension no inference reaches is output nothing: no call
> runs, nothing fires in it, and the frame carries silence there as an event dimension with nothing to report
> does. The first action in a dimension is the world's. An action the environment executes on its own is a
> lesson: it stands on the apex of its frame, every uncovered activation connects to it with its reward by the
> ordinary path (R31), and from then on it is an action the machine can infer. A neuron is born holding no
> connection at all: a base neuron at cold start and a newly allocated interneuron alike say nothing about the
> next action until something has run under them, and what they then hold is whatever was executed. R37's walk
> is the only thing that wires an action ahead of its running, one at a time and only where the best known has
> been judged and found wanting.

> **R46 — Electorate.** At frame `f` the voters on the frame ahead, `f + 1` (R35), are the open activations the
> machine holds at `f` (D9), less every inhibited activation (D24). An inhibited activation
> is a voter in each neuron dimension in which no call over it, at any height, has an inference for `f + 1`
> (R48). What
> each voter proposes is what its standing inferences name at `f + 1` (R45): a base action by the inference at
> the offset one ahead of its age (R29), a base symbol a function reaches by a farther one, and nothing where
> none of them lands there. A
> voter is read as `(neuron, position)`: the inferences are its neuron's (D11), and the inference is from its position
> (R36).

> **R47 — Resolution.** One winner per dimension per position at `f + 1`, action and event alike (R30), and
> nothing about where a voter stands in the hierarchy is read in it: an interneuron's inference and a base
> voter's meet on the estimate or the share alone, and a specific situation wins over a general one only by
> being right about what pays, never by rank.
>
> **An action dimension resolves by estimate.** At each position of it, every base action some voter's expansion
> named there is a candidate (D8). **A candidate's estimate is the mean of the estimates its voters named it
> with, each weighted by that voter's share**, one vote per voter, split across the actions it named at that
> position by strength, so a voter that hedges between two actions counts as one voter and not two, and **the
> candidate with the largest estimate is output**, unless every candidate's estimate is negative, where the
> action policy decides between the least bad and nothing (D51). Ties go to the larger share of voters, and then
> to the older action. **Nothing corrects for how many exposures an estimate rests on**, so a sharp estimate on
> three exposures outranks a coarse one on two hundred.
>
> **An event dimension resolves by share of voters.** At each position of it, every base event some voter's
> expansion read there is a candidate, and every voter whose expansion read one is one vote, split across
> the events it read at that position by strength, as above. **The candidate with the largest share is output**
> (R40). The estimate is not read here: it is the reward of the actions of those frames, and an event is what
> followed, not what paid. Ties go to the older event. A position no voter reaches outputs nothing.
>
> **A bad plan is worn down, not cut.** A standing inference carries the estimate it read at age 0 (R45). Each
> frame it names what comes due at the frame ahead: actions in their dimensions, events in theirs, at their
> positions, all at that estimate. Activations that fired after a reward moved the estimate name candidates at the
> moved number, or the walk's untried action at zero (R37). Where the largest estimate is one of those, the
> plan's action does not run; where the world reports an event other than the one the plan named, that event
> did not stand. Each such node is a failed neighbor to the plan's match (R49). The plan stands while what
> it found still pays for what it missed, and is dropped when it no longer does.
>
> **Each dimension is resolved on its own**: a plan that loses its action dimension still counts for the events
> it named, and one that wins is not favored for them, so what an event dimension outputs is what the most
> voters named, whichever inference named it, and an inference that names events alone is resolved like any
> other.

> **D51 — Action policy.** Declared once for the machine, as the alphabet and the neighborhood policy are (D1,
> D50): what an action dimension outputs when every candidate in it carries a negative estimate (R47), **the
> least bad, or nothing**. The first is for a world that acts whether or not the machine does, where silence
> hands the frame to the world's own action and the machine learns from that (R35); the second is for a world
> in which silence is an action in its own right and costs nothing. The walk is the same under either (R37); the
> policy decides only what runs once the walk has nothing left to wire in a dimension.

> **R48 — Speaking falls through the call.** An inhibited neuron supplies nothing in a neuron dimension
> while any call over it, at any height, speaks in that dimension (D24, D10). Where no call over it has an
> inference for the frame ahead in that dimension, it speaks there. So where a call has nothing, what it
> inhibits speaks for it, and what they inhibit in theirs, down to the base; nothing is output where nothing down to the base has anything. A newly
> allocated interneuron therefore starts with the voices of what it landed on, the neuron at its start among them, with
> the standing inferences each made at its age 0: they say what usually followed them, it learns what ran under it
> (R35), and from then on it speaks for itself. At the base a neuron with nothing outputs nothing, and the walk
> (R37) wires the next action where an estimate turns negative.

> **R37 — Exploration.** Explore–exploit is resolved without randomness: **the action alphabet is declared in
> order**, and **a connection whose estimate turns negative wires the next action in that order**, the first one
> the neuron holds no connection to at that offset, at strength 1 and earnings 0 (R31). **The walk runs for
> every neuron, and its dimensions are the connection's.** A neuron above the base is not an alphabet: it was
> created when a function paid, it has no declared order, and nothing could name the next one. The base actions
> are the one ordered alphabet there is, so what a negative connection wires is base actions, in the dimensions
> its target acted in, read off the target's expansion (R28): a base action, its own dimension; an interneuron,
> every action dimension its expansion reads an action in, one untried base action each; an event, or an
> interneuron that positions no action, nothing. An interneuron whose plan hurt therefore tries, at that offset,
> one other action in each dimension the plan acted in, and nothing at any other offset. **What the walk finds,
> compression names.** The untried action is a base action under a situation an interneuron stands for, named
> by the neuron that walked; where it pays it recurs, where it recurs the activations under that situation
> return the relation to it, and it joins a function (R50, R54), so that function's interneuron has the walked
> action inside it, found as a lesson's action is found, one base action at a time and then named. What stands
> in a call's variables is never searched: it is whatever stands at their nodes when the call runs (D41, D44).
>
> **The walk is in the same currency as everything else**: an untried action becomes a candidate by becoming a
> connection, so R46 enumerates it like any other and needs no second source of inferences. The trigger is one
> connection at one offset, never a reading over the neuron's whole set.
>
> **A reward is signed, and zero is where everything starts.** The environment reports an action as good or bad,
> strictly greater than zero or strictly less, and zero is neither. It is also the estimate every connection is
> created at, which is what makes the walk work: a connection at zero has not been judged, so it outranks
> anything negative and yields to anything positive.
>
> **The walk ends when the alphabet does.** Once a neuron holds a connection to every action in a dimension at
> that offset there is nothing left to wire there, and what the dimension does when every candidate in it is
> negative is the action policy's (D51): the least bad, or nothing.

## 8.5 Activation

> **R40 — An inference is weak until the report makes it strong.** Resolution leaves one winner per dimension,
> action and event alike (R47), and each fires weakly in the frame it was named in (D40), the action to run and
> the event to come, and both are the machine's output. Where the report names that dimension at that position, the
> report stands and the weak activation is dropped; an action nothing reports as run, or an event nothing
> reports as seen, lasts its frame and is gone. Where the expansion reaches a parameter, what fires is its value,
> bound from what has happened (D44). An inferred event above the base is expanded in turn, one frame at a time
> as its nodes come due, so the events a plan infers reach the base as inputs exactly as its actions reach the
> base as output. A weak activation is an input of its frame: it is recognized as a neighbor, it may be covered
> or stand on the apex and vote once, and then it is gone. **It covers nothing**: inference does not inhibit,
> and only a recognized call covers the present (D10). It carries no reward, connects to nothing, and is
> connected to by nothing (R31).

# Part VI — Beyond the specification

# 9. Examples

Nine cases, each worked by hand in a file of its own; each follows the definitions through one small job. Each
is to be rewalked under this document, with its graph and its costs written out.

| Case | File | The job |
|---|---|---|
| Alternation | [algorithm-xy.md](algorithm-xy.md) | A stream in which a pair of letters alternates: the two-node cycle it becomes, entered at either letter, one call per frame over the whole stretch, and why nothing general is made of the pairs. |
| Addition | [algorithm-addition.md](algorithm-addition.md) | A request with each column, a function per case that writes the answer, and a carry the world shows while it teaches and the machine infers afterwards. |
| Copy | [algorithm-copy.md](algorithm-copy.md) | Shown a digit and asked, the machine says it, by a parameter tying the digit shown to the digit said. |
| Equal | [algorithm-equal.md](algorithm-equal.md) | Shown four digits in one frame and asked, the machine says the third where the first two are the same and the fourth where they differ. |
| Hit | [algorithm-hit.md](algorithm-hit.md) | The machine hits whatever comes at it, wherever the two of them are, since a function is written in offsets and nothing else. |
| Loop | [algorithm-loop.md](algorithm-loop.md) | The machine taps in the frame after every `go` and runs `done` after a `stop`: the cues a cycle, the second cue a function over its call, the actions held as connections under a neighborhood policy that makes no action a neighbor, with no counter. |
| MNIST | [algorithm-mnist.md](algorithm-mnist.md) | An image channel with two spatial dimensions and two base neurons, black and white, every pixel firing one of them; runs of one color become lines, lines become strokes and corners, and so on up; a digit action taught by demonstration and resolved by a vote. |
| Stocks | [algorithm-stocks.md](algorithm-stocks.md) | A price channel per instrument with own and disown as its actions, a world that runs the right action wherever the machine has none, and a reward scoped to the channel and the frame. |
| Text | [algorithm-text.md](algorithm-text.md) | A stream of letters: triples at the base, a pair never paying, and stretches of any length, each letter's function naming the interneuron a frame before. |

# 10. Further reading

Each document is written in the terms of this specification and references its keys.

| Document | What it holds |
|---|---|
| [algorithm-remarks.md](algorithm-remarks.md) | The theorems, and the commentary on why the design is shaped this way, keyed by the same D and R numbers. |
| [algorithm-evaluation.md](algorithm-evaluation.md) | The falsifiable claim, the risks with what would be done about each, and the open questions. |
| [algorithm-implementation.md](algorithm-implementation.md) | The implementation plan: the per-neuron state and calls, the build stages, and what is cached. |
| [hippocampus.md](hippocampus.md) | An addendum: the second organ, running beside this machine over the same neurons, with rules keyed **H**. |
