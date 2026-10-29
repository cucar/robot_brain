# Universal Compression with Actions and Rewards (UCAR)

UCAR is an algorithm that compresses what it observes and acts on the best reward estimate it holds. A machine
running it is a **UCAR machine**, and like a Turing machine it is defined by two alphabets: the **event alphabet**
it can observe and the **action alphabet** it can execute. Every symbol, base or learned, event or action, is a
**neuron**.

Its inputs are the events observed, the actions that ran and the rewards earned. Its output is what it places
at the frame ahead, written in the base alphabet: the actions and the events it infers.

**What it compresses is a graph.** Every activation of a neuron is a node. Every relation between two
activations, one at an offset from the other in one dimension of the world, is an edge. The graph spans time and
every dimension of space together, and it is the only record the machine keeps of what happened: it holds
offsets, never coordinates. A neuron sees the edges around each of its own activations and reports them with
how often each has held. The machine groups the edges into **functions**, graphs over neurons that keep
recurring, names each with an interneuron, and from then on the name stands where the graph recurs. A function is
recognized at its latest node, in the frame that node fires, and covers back over what has happened; a relation
that always followed it joins it, and the function is recognized again at its new last node. Where a function
is general
it leaves a **variable**, a node whose neuron changes from one occurrence to the next and is filled by what
stands there. The interneurons are nodes like any other, so the next level groups them in turn, and the
dictionary is a hierarchy of named graphs.

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
interneuron stands for it wherever it does. The graph of the run is stated shorter by the name in place of the
graph. Functions may name base symbols or the interneurons of other functions. This substitution is the whole
mechanism of compression.

**A function is what has happened, and a connection is what followed.** A function is recognized at its latest
node, once every node of it has fired (D54). A relation joins a function only where it has come true at every
placement the machine accepted (R50), so a function holds what always followed it, and grows at its latest end
as the run bears it out. What followed any activation, its function's later nodes included, is on the neurons
as connections with an estimate (D25), and the reward lives there, where the choice is.

**The compression is lossy in one place, and it is a limit of the neuron.**

| Loss      | Description                                                                                                                                       | Reference |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| memory    | A neuron has limited memory: it remembers only its most recent activations, and reports its relations over those, so what it saw earlier no longer shapes the dictionary. | D18       |

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
neurons placed at the frame ahead are then resolved into the machine's output. In each action dimension the
machine outputs the action placed there with the largest estimate (R47). In each event dimension it outputs the
event the most inferences place there, and reads no estimate. That is the whole of it: **act on the best
estimate you hold, for the situation you are actually in.**

**Inference does not inhibit.** What a connection places covers nothing. Only a call recognized at the present
covers, and only what has happened (D54, R40). An action is chosen by connections and their estimates, never
by a function: a function records what was done, afterward.

**A reward pays frames and channels.** It arrives with a frame and may name the action channels it pays and the
span of frames it pays over (D35). A reward that names no channels pays every action channel. One that names no
span pays as far back as any neuron can still see, the longest reach in time the machine holds. The share of it
reaching each frame of the span falls linearly with that frame's distance from the one it arrived in, and lands
on the connections written at that frame (R33).

**Compression decides which neurons exist, and the neurons hold the connections.** A situation that recurs is
named by a function. The function's interneuron then stands on the apex in its place, and holds connections of
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
has a place on the board, and reads the layout for one thing: the offset between any two activations in each
dimension (D6). The graph keeps the offsets and nothing of the places.

## 2.2 Channels

> **D3 — Channels.** A channel is defined by an event dimension, an action dimension, and spatial activation
> dimensions. A channel has an event dimension, an action dimension or both, and owns them exclusively. The same
> activation dimensions may be shared by several channels. Each frame, the channel reports a symbol of each of
> its neuron dimensions at each point of its layout.

For example, a vision channel is laid out over `x`, `y` and time. A stream of prices could use time alone.

Three color channels of one camera use the same `x` and `y`, so a red pixel and a green pixel can stand at the
same place in one camera. Two cameras with separate screens name separate axes.

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
design is a count of the numbers it would take to state the graph of the run with the functions as they stand
(D12, D13). The count is how the machine compares one way of covering the board with another, and it is never
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
(§4.1). It knows nothing of functions, of the dictionary or of what is counted.

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
> (R24) or an inference places it (R36).
>
> **A neuron may fire many times in one frame**, and those activations are instances of one type.
>
> **At the base, at most one neuron fires per dimension per point of the layout in a frame.** The input reports
> one symbol per channel at each point, and a dimension with nothing to report there is silent. That bound is a
> property of the input, not a rule the machine imposes.
>
> **Above the base, several neurons can fire at one place.** Several fits may be accepted over overlapping
> ground (D28), so several interneurons may stand at one place (D53), and one function fitting at several places
> is one interneuron that fires at each (D15).
>
> **An activation has a place, and a call's activation covers back from it.** A base activation stands where
> the input laid it. An interneuron's activation stands at its origin, the fit's latest node, in the frame it is
> accepted (D37), and covers from there everything its function's relations landed on, all of it already
> happened (D54).

> **D9 — Activation window.** An activation at place `f` in time stays open through `f + reach_t`, its reach in
> time (D4). Two things reach it over that span, one frame at a time, by the connect call after the frame's last
> level (§7.1):
>
> | Input           | Description                                                                                                 | References |
> |-----------------|-------------------------------------------------------------------------------------------------------------|------------|
> | the apex        | What stands on the apex of that frame within reach (R27, D25), while the activation is uncovered; in the frame it is covered, the neuron that covered it (D10). | D25, R31   |
> | the rewards     | For the actions of that frame, and of earlier frames the reward spans, each at the distance its frame names. | R29, R33   |
>
> The machine holds the activation open, not the neuron. There is one open activation per neuron and place, and
> it holds: the activation, its place on the board, its age, and the age it was covered at, if it has been. An
> interneuron's activation holds besides what expansion reads (R28): the function that fired it, which of the
> function's relations did not come true, and how far back it ran round a cycle.
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since the frame it fired in, `0` through `reach_t`; a new activation is the one at age 0, base and
> interneuron alike.
>
> **Covered at** is the age at which an accepted function first covered the activation, none until one does.
> Coverage is acquired late (R27), and a later call may take the activation from its coverer, covered still
> (R23), so this one number says for every frame of the window whether the activation stood on the apex there:
> it did at every age before it, and at none from it on.
>
> **An open activation carries no commitment beyond that.** It infers once, at age 0, and the machine keeps
> what it inferred (R45); what it does for the rest of its life is connect to what ran, in the `connect` call.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what expansion
assumes for anything no node states.

> **D40 — Strength of an activation.** An activation is **strong** when the environment reported it, and
> **weak** when an inference placed it (§8.5). That holds for events and actions alike: an event and an action
> the machine infers are both weak, and the environment's report replaces both. The neuron is the same either
> way: a weak activation is an activation of the neuron the environment would have reported.
>
> **A weak activation is an input of its frame like a strong one.** It is in the neighborhood of everything that
> fires within reach of it, at whatever offset, so it enters histories, and functions name it and price it like
> any neighbor; its neuron is called with it, it may be covered, and it may stand on the apex and vote once
> (R46), with the connections its neuron's strong activations strengthened. **It differs in one thing: it lasts
> its one frame.** It is not open afterward: it writes no connection, nothing connects to it, and no reward
> reaches it (R31). **It covers nothing** (R40). **The report replaces it**: where a strong activation of the
> same neuron stands at the same place, the weak one is dropped. Where nothing reports, the inference peeks
> through: an inferred action nothing reports as run, and an inferred event nothing reports as seen, are what
> the machine had in that frame.
>
> **An interneuron is as strong as what it covers.** An accepted function's interneuron is strong when any
> activation it covers or binds is strong, and weak when every one of them is: a situation made entirely of
> inferred inputs is inferred.

> **D37 — Call.** One activation of a call neuron (D48), together with its **arguments**: the activations bound
> in its function's variables on that occasion (D10, D38). It is made in one of two ways and is the same thing
> either way. **Recognized**: the election accepts a fit (R24), and the call neuron fires at the fit's
> **origin**, its latest node (D15), in the frame it is accepted, at age 0, as strong as what it covers (D40);
> its arguments are what stood in the variables. **Inferred**: a connection names the call neuron at an offset,
> and an expansion places the call there, at its completion, weakly (R36), its nodes at the frames back from it;
> the part that lies ahead is what it proposes, for the report to make strong what it placed (R30), and its
> arguments are bound from what has happened where a variable's node has happened, and placed where it has not
> (D44, R28). A call carries nothing of its own: what it is stands in its function, and what its arguments are
> stands on the apex beside it. What it records is what expansion needs (D9).

**A function is recognized or inferred, and both kinds of member take part in both.** Recognized, a function
says that its events stood and its actions ran. Inferred, it is expanded (R28): its actions are placed as the
machine's output and its events beside them, both weakly, and the report makes strong what ran and what was
seen. What follows any activation is held on its neuron as connections (D25), one kind, and an inference is a
connection read.

## 3.4 Spacetime

> **D4 — Reach.** How far an activation sees, in each activation dimension (D1). A base activation sees one
> position: the neighbors beside it in each dimension of space, and the frame before in time. An interneuron's
> activation sees one position beyond the farthest node its function places from its origin (D15, R28), in each
> dimension. In time the origin is the latest node, so the activation sees back over all of what it stands for,
> and one frame more. Where what it stands for comes back to an interneuron already expanded, it stops there
> (D36), and so does its reach.
>
> In space the reach runs either way; in time it runs back, and the same number runs forward as the activation's
> window (D9). `reach_t` is the reach in time, the activation's window.

> **D6 — Offsets.** The offset between two activations is the difference of their places on the board, one
> component per activation dimension they share (D1). Every offset is exact, in a neighborhood (D18), in a
> function (D15), in a connection (D25) and where an expansion places a node (R28). A base activation holds its
> neighbors at offsets of one: `−1` in time, and `−1`, `0` or `+1` in each dimension of space. An interneuron
> holds them out to its reach.

> **D5 — Adjacency.** An activation `b` is adjacent to an activation `a` when the following conditions hold:
>
> - `b` is in the same frame as `a` or in an earlier one.
> - `b` is within `a`'s reach in every activation dimension they share, time among them (D4).
> - `a` stands on the apex (R27). `b` stands on it, or is covered by a call on it other than `a`'s own (D10).
> - The neighborhood policy allows `b`'s kind beside `a`'s (D50).
>
> For example, in an image channel the pixels of one frame are adjacent to one another within reach. In space
> adjacency runs both ways. In time it runs only backward: what fires after `a` is never adjacent to it, and is
> held as a connection instead (D25).

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

> **D26 — Neighbor.** **A neighbor is a neuron at an offset**: the neuron of an adjacent activation (D5), at its
> offset (D6). At temporal offset 0 a neighbor co-occurs and at negative offsets it led here; **both are the same
> kind of thing**, and a spatial component is one more of the same.
>
> **A call is a neighbor as itself and as what it covers.** An activation that sees a call within reach sees it
> as itself, at its origin's offset, and as each base symbol under it that is within reach, each at that
> symbol's own offset (D47). So a base activation that sees one frame back sees the call standing there and the
> symbol the call's origin is, and a function may name either.
>
> **A neuron can be its own neighbor.** Two activations of one type at different places each name the other at a
> nonzero spatial offset, and in time a neuron stands beside its own earlier activation. Offset zero in every
> component is where the activation itself sits, and an activation is not its own neighbor there. Above the base
> several interneurons standing at one place are each other's neighbors at offset zero.

> **D7 — Neighborhood.** The set of neighbors (D26) one activation observes is its **neighborhood**. In a text
> stream, with time its one dimension, the neighborhood of `s` in `p a r i s` is `i` at −1, and `p a r` are out
> of reach. In an image stream, with time, `x` and `y`, a neighborhood holds the pixels one away in `x` and in
> `y`, and the pixels of the frame before at the same distance.
>
> **A neighborhood is the frame around the fired neuron**, since adjacency admits nothing later (D5). The
> activation itself is not in it: its own node is what a function covering it replaces, and is counted there
> (D22).
>
> **An activation bound in a variable is a neighbor like any other** (D10): the neuron that stood, at its own
> place. Which variable it stood in is the call's business, not the neighbor's.

> **D34 — Relation.** Two neurons and the offset between them in one activation dimension (D1, D6). There is one
> kind of relation in the design: what a neuron returns is a relation, what a function holds is a relation, and
> they are the same object. A node joined to the graph in several dimensions is in several relations, one per
> dimension. In a neuron's relation index, one end is the neuron's own activation and the other a neighbor. In a
> function, both ends are its nodes, and a relation from a neuron to itself joins one node to itself.
>
> **A relation has one form, and in time it runs backward.** In time a relation runs from the later neuron to
> the earlier one, at a negative offset: that is how the neuron that fired saw it, what stood before it, and the
> earlier neuron never sees the later. In a dimension of space the two neurons are ordered so that the offset
> from the first to the second is positive, and where it is zero, the older neuron first (D2); there each end
> sees the relation from its own side and counts it in its own index (D47), the machine reads both returns as
> the one relation, and takes the larger of the two strengths, since each neuron's history is a window that can
> only miss occasions (D18, R21).
>
> **Relations compose.** Given the offset from `a` to `b` and from `b` to `c` in one dimension, the offset from
> `a` to `c` follows, so a relation implied by composing others along a path says nothing new and is never held
> (D15).

> **D11 — Identity.** A neuron's identity is its type alone, a base symbol or an interneuron's id (D2), so
> nothing about where it occurred is part of it. **The same shape at two places is two activations of one
> neuron**, and they pool: both carry the same relations, so the same functions cover them and one function
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

## 3.5 The graph

> **D55 — Graph.** What a run leaves: a set of **nodes**, each an activation of a neuron (D8), and a set of
> **edges**, each a relation between two nodes (D34). A node carries its neuron's name and nothing else. An edge
> carries a dimension and an offset. Two nodes are joined by at most one edge per dimension. A node's place in
> the world is not in the graph: it is wherever the edges put it relative to the nodes it is joined to, and two
> nodes joined by no path in a dimension stand in no particular relation there.
>
> **The graph is what every neuron saw.** An edge exists between two nodes where one was in the other's
> neighborhood when it fired (D7), so the graph has no edge longer than a reach (D4). Two activations farther
> apart than any neuron sees are joined by no edge, and the graph does not say where one is relative to the
> other. That is right, since no neuron could ever have read it.
>
> **A piece of the graph is stated as nodes and edges, nothing more.** A piece, in the dictionary or on the
> board, is its nodes, each named once, and its edges, each between two of those nodes, with a dimension and an
> offset. A node several edges touch is named once. Two parts joined by no edge need nothing between them. The
> cost of a piece is its names and its offsets, one number each (D13).
>
> **An edge implied by others is not counted.** Where a path of edges in one dimension fixes the offset between
> two nodes, an edge between them in that dimension says nothing new (D34), and the count holds one of them.
> Across dimensions there is no such constraint: `a–b` in `x`, `b–c` in `y` and `c–a` in `x` are three
> independent facts.

**Everything in the design is this one object.** What a neuron returns is the edges at one node, with how often
each has held (R21). A function is a graph over neurons that recurs (D15). The dictionary is those graphs, held
once each. The board is the graph of the frame as it stands, with the calls in place of the pieces they cover
(D12). Covering is substituting a name for a piece. Expansion is putting the piece back (R28).

## 3.6 Functions

> **D15 — Function.** A graph over neurons that recurs: a set of relations (D34), each joining two **nodes**,
> with **one node per neuron**: a node is a neuron, or an open node, a variable (D38), and a relation names its
> two nodes by naming its two neurons. A function is **connected**: every node is reached from every other along
> its relations. It is the center the machine keeps for the relations it has grouped (R50): it exists from its
> first relation, with or without an interneuron, and has one interneuron once a fit of it has been accepted
> (R16, D48).
>
> **Its origin** is one of its nodes, fixed in the dictionary (D56): the node at age 0 when the function is
> recognized, its latest node in time, and among several at that frame, the one at the least place in the
> order the dimensions are declared (D1). The call fires there (D37, D53), and expansion runs back from there
> (R28). Two functions with the same relations and different origins are two functions.
>
> **A function may hold cycles.** A node is listed once, and in each dimension a relation references it at
> most twice: once as the later end, its one step back, and once as the earlier end, its one step forward. A
> relation whose far end is a node already referenced closes a cycle, in any dimension, time included, and the
> cycle may run through any nodes, the origin among them or not: `y` after `x` and `x` after `y`, origin `y`, is
> `… x y x y` for as long as it runs; `a` then `b c b c …` is a tail into a cycle, and `b c b c d` a cycle with
> an exit. A function holds at most one cycle per dimension, since a third reference to a node in one dimension
> would need a second step back or forward, and there is none. A relation that a path of the function's other
> relations already implies is never held (D34).
>
> **A fit is a match of relations to relations.** The board is the relations the neurons returned (R21), and a
> fit of a function is one board relation per relation of the function, per traversal of a cycle, with the
> neurons, the dimension and the offset the function's relation states, two relations of the function that share
> a node landing on two board relations that share an end. A node may land on several activations, one per
> traversal. A fit starts at the function's origin, on an activation at the present frame, and nowhere else:
> a stretch that ends on another node of a cycle is a fit of the function that has that node as its origin. A function may
> fit at several places in one frame, each fit one activation of its interneuron, and each fit is priced as it
> falls (D22).
>
> **A function is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a function is what the machine holds. A neighborhood is a fact, a function a
> claim. What followed is in neither, and is held as connections (D25).
>
> **In the dictionary it is** its nodes, each named once, and its relations, as D55 states a piece of the graph
> and as the dictionary stores it (D56). What it costs there is that statement's (D13).

**A function's body is its relations, and the nodes they name.** Some nodes are fixed, a particular neuron;
some are **variables**, nodes where the function is general. A variable is a hole in the graph: it names the
place and leaves open what stands in it. What stands in a variable on one occasion is the variable's **value**,
and the values of a call's variables are its arguments (D37). A variable exists only inside a function, is
never placed alone, and promotes no interneuron: what stands in it stays on the apex, bound (D10).

> **D38 — Variables.** A variable is an open node: a node of a function that leaves open which neuron stands at
> its place. The machine makes variables, never a neuron (R52): a variable says what is general across several
> functions, which no one activation shows. It is of one of two types, closed in opposite ways:
>
> | Variable | In the dictionary, once | In the call, each time | Closed on |
> |---|---|---|---|
> | a class (D41) | which class, and the class's members | the member that stood, one number | its members: a neuron that is not a member does not fit, and any function may use the class at any node |
> | a parameter (D44) | the node, open | the neuron that stood, one number | nothing: any neuron fits, seen before or not |
>
> **A variable is one node, like any node.** Where the same thing has to stand at two places, the function says
> so the way it says anything twice: the node is reached again, by a cycle (D15). Nothing ties places but the
> graph.
>
> **A function holds nothing but its nodes and relations**: no branch, no loop but a cycle. What branches is
> which function fires; what loops is the frame; what a variable held is on the apex, bound beside the call
> neuron. On every occasion it is priced as part of the call (D12, D22).

> **D41 — Class.** A set of neurons the dictionary holds, its **members**, and nothing else: **one of these
> stands here**. In the dictionary it costs one name per member (D13). A function has it at a node as a variable
> (D38); several functions may have one class, and a function may have one class at several nodes, a member
> bound at each. Members stand in place of one another: two neurons that have stood together where the class is
> used are never both members.
>
> **Its value**, at a node, is the member that stood there, which stays on the apex, bound (D10). Where no
> member stands at a node the class is at, that node's relations did not come true (D22), and the variable does
> not hold there.
>
> **A class is not a neuron.** It is made by generalizing functions that differ at that node (R52). A neuron
> joins it when it stands at the node in an occurrence where the rest of a function using the class holds, and
> what it covers there pays for its one name in the dictionary; it leaves when it no longer does. So what is
> learned about a member at one place holds at every other. A class is deleted when no function has it, or when
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
> **Bound from what has happened, placed at what has not.** When an inferred call is expanded with no argument
> for a parameter (R28), and the node is reached once in frames that have happened and again in frames to come,
> it takes its value from what stood the first time and places it the second. That is how a value is carried
> forward in time, and it is the whole of copying.

## 3.7 Interneurons

> **D53 — Interneuron.** The neuron the machine allocates for a function, its **call neuron** (D48). It is
> allocated the first time a fit of the function is accepted and a neuron is free (R16). It has an id and no
> dimension and no channel. **It fires at the fit's origin** (D15): the fit's latest node, in the frame the fit
> is accepted, at age 0. One activation per fit, and the activation covers back from there everything the fit
> landed on (D54).

An interneuron is not an event or an action, though what it stands for may be either. Where it stands is read
off the fit, so that other neurons can hold it as a neighbor and expansion can start from one place. Nothing in
the pricing reads it.

A neuron is never made or destroyed. It is **allocated** when an interneuron is needed, and **decommissioned**
when its last function leaves the dictionary, to be allocated again. When none of the machine's `N` neurons is
free, no interneuron is granted (R16, R24).

> **D56 — Dictionary.** What the machine holds about functions, field by field. Three kinds of thing are in it,
> and every cost in the design is a count over these fields (D13).
>
> | Thing | Field | What it holds | Counted |
> |---|---|---|---|
> | a function | relations | its relations, each a neuron relation as the neurons return it: two neurons, a dimension, an offset (D34); together they name its nodes, one per neuron | one number per node and one per relation |
> | | origin | which of its nodes is the origin, the node it is recognized at (D15) | within the node's number: the node list begins with it |
> | | open nodes | which of its nodes are variables, each a class by id or a parameter (D38) | within the node's number: a class costs its id, a parameter its node |
> | | interneuron | the neuron allocated for it, or none (D53) | nothing: the function is reached through it, not stated by it |
> | | counts | its last `H` accepted placements, and per relation and per candidate at how many of those it held, as the worth reads them (R50, D57) | nothing: counts are the search's, never a statement of the graph |
> | | members | per parameter, the neurons that have stood there within the window, a record (D44) | nothing |
> | a class | members | its member neurons, by id (D41) | one number per member |
> | an interneuron | functions | the functions it fires for, one or more where a stretch came round (R43); and each form a function had before a move, kept until the last window over the activations that form fired has closed (R54, R38) | nothing beyond the functions' own |
>
> **Two indexes are kept over it and rebuilt from it**, so they cost nothing: by relation, each neuron and
> offset with the other end open or not, to the functions holding it, for finding (R53); and by what an
> interneuron stands for, its expansion to base symbols, for reuse (R43) and for generalizing (R52).
>
> So a function of `n` nodes and `m` relations costs `n + m` numbers, a class costs one per member, and a
> parameter costs nothing beyond its node. Nothing here is the implementation's: these are the fields, and the
> costs are counted over them.

> **D36 — Interneuron index.** The dictionary (D56) reached three ways:
>
> | By | Reaches |
> |---|---|
> | a function | its interneuron, and back, for expansion (R28) |
> | a relation (D34), or one of its neurons and its offset with the other end open | every function that holds it, or holds any relation with that neuron at that end and that offset |
> | what an interneuron stands for | the interneuron, for reuse when a stretch comes round (R43), and the functions of the same shape, for generalizing (R52) |
>
> **The same relations are the same function.** A function is its set of relations, so the same neurons at the
> same offsets are one function whichever neurons returned them. Each function has one interneuron. An
> interneuron may hold more than one function where a stretch comes round (R43): it fires for each, and an
> activation of it is expanded through the one that fired it.
>
> **A function names other interneurons by id**, and through them it may come back to its own: a function whose
> node is the interneuron that fired a frame before, whose function names the one before that, and so on, can
> reach its own interneuron again (§6.7). **What an interneuron stands for** is its expansion (R28): its
> function's nodes with the variables left open, each node in turn for what it stands for, down to base symbols.
> Where that comes back to an interneuron already expanded, it stops there.
>
> **The index is searched by what an interneuron would stand for** (R43): a function's expansion, read from its
> origin. The search finds an interneuron when three things hold: reading that expansion, it meets the
> interneuron within the interneuron's own length; over exactly that length the two expansions are the same; and
> the meeting is not at the origin. That is how something that comes round again finds the interneuron it had the
> first time.

**A function and its interneuron are held together.** The machine holds both in its dictionary. No neuron holds
a function or points at an interneuron: a neuron records what it sees and returns it (§5), and the machine
groups, places and keeps (§6).

> **D48 — Call neuron.** A function's interneuron: the neuron the machine allocates for a function the first
> time a fit of it is accepted (R16). It fires when a fit of its function is accepted, at the fit's origin
> (D15), once per fit, covering the function's fixed nodes (§6.4); what stands in the function's
> variables is bound beside it, on the apex (D10). It says which function ran. Its connections are one set for
> every value its variables have taken, so a function above learns about it regardless of its arguments by
> naming the call neuron alone, and about one argument by naming the call neuron and the activation bound in
> that variable. It holds a history, a relation index and connections of its own, like any neuron, and is
> expanded through the function that fired it (R28).

## 3.8 Apex

> **D49 — Apex.** The activations of a frame that no accepted function covers, base neurons and interneurons
> alike: a base neuron no function found worth naming stands in it beside an interneuron many functions deep.
> It is the machine's whole state at that frame, and it has four names for the four things it is. It is the
> **frontier**, looking back: everything underneath it is recovered by expanding it, so it is what the cost
> counts (D12). It is the **situation**: what every neuron sees as its neighborhood (D5), what connects to what
> ran, and what votes. It is the **wavefront** of recognition: where the frame's levels stopped (R26), and from
> where the wave runs forward, each activation of it reading its own connections ahead. And it is the apex, the
> top of the hierarchy as it stands. The four are one set, and coverage takes a neuron out of it in every role
> at once.
>
> **An expected input never reaches the apex.** An activation a function in the dictionary names at its latest
> node is covered at age 0 by that function's call, recognized in the frame it fires, and the call takes back
> the call that stood for the function's earlier part (R23). So where a function keeps coming true the apex
> holds one call, the latest, and what arrives is covered as it arrives.
>
> **Partway through the frame it is the board.** While the levels run, the apex as it stands at a level is the
> board the next level is placed and elected on (§6). The apex of the frame is the board once the last level has
> run.

> **D10 — Inhibition.** Coverage does three things to an activation, all from the frame it is covered in. It
> leaves the count: the cost counts its coverer's call, not it (D12). It is not heard: resolution reads its
> coverer's standing inferences, not its own (R46). And it stops connecting: from the frame after it is covered
> it writes nothing about what follows.
>
> **The last connection.** In the frame it is covered, the activation connects once more, to the neuron that
> covered it, the function's interneuron. The connection is at the offset from the activation to its coverer
> (D6), in time and in every other activation dimension, as every connection is (D25), and it takes that frame's
> reward (R31). Then the activation is silent. An activation standing at its coverer's own place, the origin
> (D15), writes nothing, as one covered at age 0 does. Where a later call takes the activation from its coverer
> (R23), it connects once more the same way, to the new coverer. Without this connection the activation could
> never learn the neuron built over it: the interneuron did not exist in the frames it was uncovered, and it is
> inhibited from the frame the interneuron first fires. So a situation beside which a function has named what
> followed it learns that one neuron, the interneuron, and from then on infers it; the interneuron learns what
> follows.
>
> **An origin is covered at age 0.** A call recognized at the present covers its own origin in the frame it
> fires (D54). That activation never stands on the apex: it writes nothing but its last connection, to the
> call. Only a recognized call covers. What an inference placed covers nothing (R40).
>
> **What stands before coverage stands.** A neuron is covered only after it has fired (R27). Coverage acquired at
> a later age revokes nothing before it: the connections the activation strengthened while uncovered stand, and
> a reward for one of those frames still reaches them (R33). An activation covered already at age 0 writes
> nothing but its last connection, to its coverer and again to a call that takes it (R23): a connection is to
> what comes after (D25), and by then it is inhibited.
>
> **One exception, on being heard alone.** Every activation infers at its age 0, covered or not, and the
> machine keeps what it inferred (R45). A covered activation's inferences are heard only where its coverer has
> nothing for the frame ahead (R48), as a newly allocated interneuron has nothing until something has run under
> it. The covered activation still writes nothing and connects to nothing; only its coverer learns what ran.
> That is the whole of inhibition in the design.
>
> **Bound is half covered.** An activation standing in a variable of an accepted function (D38) is **bound**.
> The call covers the relations to it: the function's relations are in the dictionary once. Its neuron is
> counted inside the call, one number (D12, D13). It stays on the apex, it is a neighbor, it is heard, and it
> connects to what follows. A call hides what is the same on every occasion, its fixed nodes, and leaves visible
> what varies.
>
> **Above its level, a bound activation is a neighbor.** Its neuron is called once, at age 0 (R1). At the
> levels above, the interneurons that fire see it as a neighbor, and their relations to it are the ones it takes
> part in. A relation to it the call that bound it did not cover stands at its cost, so a function above can
> cover it. One the call covered costs nothing, so a relation above that lands on it holds a node and covers
> nothing (D22). Its neuron is still counted, as the value in the call that bound it, so a function above that
> names that call and this neuron fixes the value, and saves it (D24).

## 3.9 Costs

> **D12 — Cost of the graph.** What it takes to state the graph of the run as it stands on the apex, with the
> accepted calls in place of what they cover: the nodes on the apex, and the relations among them. Nothing
> joins a node to the graph but the relations the neurons returned: a node its neuron related to nothing is a
> name alone, and no node has a place in the count.
>
> | On the apex | Counted as |
> |---|---|
> | a bare activation | its name |
> | a relation between two activations on the apex | its offset, once per pair and dimension, and not where other relations on the apex imply it (D55) |
> | a call | the interneuron's name; then each relation of its function that did not come true, one number each; then the value of each variable, one number each; then, where the function holds a cycle, how far back it ran round it, one number |
> | an activation bound in a variable | nothing of its own: its neuron is the value counted in the call, and the relations the function fixes are the call's |
>
> **A relation to a covered activation is a relation to its coverer.** The call stands in place of what it
> covers, so a board relation from an activation on the apex to a node inside a call joins that activation to the
> call, read at the node it names (D26). It is counted once, however many nodes of the call it could be read at.
>
> **What a call takes out of the count** is the relations among the nodes its true relations landed on, which
> its function holds once in the dictionary, and the names of the nodes its true relations landed on as fixed
> nodes. A node at a variable keeps its name, counted as the call's value. A relation between a node inside the
> call and one outside it stands as it was: it joins the call to the rest of the graph, and the call adds no
> relation of its own.

**A call counts its body once and its differences every time.** The function's nodes and relations are in the
dictionary. Each call counts which of its relations did not come true, what stood in its variables, and how
far it ran round a cycle. What the relations fix is not counted again: the names of its fixed nodes, and the
offsets among the nodes they join.

**A failure is a relation that did not come true.** It is counted with the call, one number. Whatever stood
where the relation said is on the apex, uncovered, and is counted as itself. So a wrong neuron costs the
relation and a node, and nothing there costs the relation alone.

**Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
count exactly as an observed event does, compressed by the same functions (D5).

**It holds nothing about the future.** A function is what has happened. A neuron's connections, what followed
its activations, event and action alike, and what the actions earned, are in neither the graph nor the
dictionary (D25), and neither is an inference that did not run (R32).

**It holds nothing about the search either.** Strengths, placements, estimates and worths are the machine's and
the neurons' and enter no count, because expanding an apex neuron needs the functions and nothing else.

**It holds the values.** What stands in a variable is counted inside the call that bound it, as its neuron
(D13), and expansion puts it back at its node (R28).

**Two runs that differ only in where things stand cost the same.** The graph has no coordinates, so a shape at
one place and the same shape at another are counted the same, and nothing in the costs reads a coordinate
(D11).

> **D13 — Number.** Every cost in the design is a count of numbers: how many it takes to state a piece of the
> graph (D55), over the fields the dictionary holds (D56). Nothing is stored for it; the count is how the
> machine compares choices. A number is one field at the machine's precision `P` (§3), and every number costs
> the same, `1`. A name, an offset in one dimension, a failure and a value are each one number.
>
> | What | Numbers |
> |---|---|
> | a bare activation on the apex | its name: `1` |
> | a relation on the apex | its offset: `1` |
> | a call on the apex | its name, `1`; plus one per relation that did not come true; plus one per variable, its value; plus, where the function holds a cycle, how far back round it the fit ran, one number (D12) |
> | a function in the dictionary, of `n` nodes and `m` relations | `n` names and `m` offsets (D56) |
> | a class at a node, in the dictionary | `1`, which class |
> | a class, in the dictionary | one per member |

The precision `P` and the neuron capacity `N` are the hardware's (§3), not prices. Neither enters a cost, since
every number costs `1` at whatever precision. They set what can exist: how many neurons there are to allocate
(R16), and how far an offset can be counted.

**Every saving is a comparison of two counts.** Each savings and each worth in the design is the cost of stating
the same thing two ways, A less B, and nothing else: a piece of the graph stated bare against stated through a
function (D22), a relation counted on the apex at every occurrence against held in the dictionary once (D57).
Nothing is counted that is not in A and left out of B.

**Covering an activation saves exactly what it cost.** A fixed node that came true saves its name, and one
offset for each dimension a relation joins it in. A variable that came true saves the offsets, and its neuron is
counted inside the call.

**Nothing about frequency enters a price.** A symbol costs the same however often it occurs. How often it
occurs is what the worth counts (D57).

## 3.10 Savings

> **D54 — Fit.** A function fits where stating what it names through it costs less than stating it bare: its
> savings there are positive (D22). A fit is a match of relations (D15): each relation of the function against
> one board relation of the same dimension and offset, with the two sharing an end wherever the function's
> relations share a node. A board relation is read as it stands: its ends are the neurons that fired, and an
> interneuron on the apex is a neuron as itself. Nothing is reserved: two functions may land on the same board
> relation.
>
> **A fit is read at the present, over what has happened.** It starts at the function's origin, on an
> activation at the frame being recognized, and every other node is at a negative offset from it in time (D15),
> so every relation of the function is true or failed when the fit is placed, and nothing is pending. Without
> an activation at the origin there is no fit (D31). A function with a cycle is read
> back round it as far as its relations keep coming true within reach, and the call counts how far (D12). A
> relation of a cycle came true when it held anywhere in the fit, and did not when it held nowhere.
>
> **Coming true.** Each relation of a function comes true or does not. It comes true where the board holds a
> relation of its dimension and offset whose near end is the end already landed and whose far end is what the
> function's far node admits:
>
> | Relation, at its far node | Comes true where the far end is |
> |---|---|
> | a fixed node | its neuron, or an interneuron standing on the apex that stands for it there (D22) |
> | a class | a member; it takes members only (D41) |
> | a parameter | any neuron where the node is first reached, reading back from the origin, and the value already bound where it is reached again (D44) |
>
> A relation that does not come true lands on no board relation and is counted with the call (D12). A node no
> relation that came true reaches stands nowhere: the call counts nothing for it, and expansion places it from
> the relations that did (R28).

> **D22 — Savings.** What one fit saves where it is read: the cost of stating bare what it lands on and came
> true (A), less the cost of stating it through the function (B).
>
> | | Numbers |
> |---|---|
> | A | for each relation of the fit that came true, the board relation it stands in for, one offset; for each node a true relation lands on as a fixed node, its name; and for the relations and names a lower call covers, that call's whole count, once, where the fit lands on every relation that call covers, and nothing where it lands on some (R23) |
> | B | the call's count (D12): its name, `1`; one value per variable; one failure per relation that did not come true; and, where the function holds a cycle, how far the fit ran round it |
> | savings | A less B |
>
> A relation that did not come true is in neither: whatever stands there is counted bare either way. What a
> relation or a name is worth is always what it cost (D13).
>
> **A call is taken back whole or not at all.** Where a function's relations land on every relation an earlier
> call covers, and on the names it covers, it takes the call back: the call leaves the count, the interneuron's
> activation leaves the apex, and the function is credited the call's count, since that is what stood for them
> (R23). Where it lands on some of them, those relations hold, the function is credited nothing for them, and
> the call stands: its count stands either way. So a function saves what it takes out of the count, bare
> relations and names, and whole calls.

**What a function's call stands in for is what it covers**; that is A. What it costs is its count; that is B.
A neuron the function does not name is in neither: it costs what it cost whether the function exists or not.

> **D57 — Worth.** A relation saves nothing by itself: only a call takes anything out of the graph, the board
> relations its function's relations land on and the names at their nodes, for the call's one name (D12). The
> worth is that saving shared out among the function's relations, since grouping decides relation by relation
> which belong to a function (R50, R54). One relation's share, in one function, over the function's last `H`
> accepted placements (R50):
>
> | | |
> |---|---|
> | held | the accepted placements at which the relation came true; for a candidate, those at which it was attached |
> | failed | the accepted placements at which it did not come true, counted from the first placement it was seen at |
> | saves | one offset, and one name where the relation brings a node no other relation of the function names |
> | costs | one offset in the dictionary, and that name where it brings the node (D13) |
> | worth | `held · saves − failed − costs` |
>
> **A function's worth is the sum of its relations' worths**, with a name counted once however many relations
> touch the node. The origin's name is credited to no relation: it is what pays for the call's name. A function
> of one relation, two nodes and an offset: its call takes three numbers out of the graph and counts one, saving
> 2 per occasion, against 3 in the dictionary, so it is worth `2 · held − 3`, which turns positive once it has
> held twice: nothing is built on one occurrence (R21).
>
> **A relation's worth where it would be** is read with the same counts, as a candidate of that function (R50),
> and a relation goes where its worth is highest (R54). The counts are the machine's own, taken at every
> placement of the function once the election has ruled on it, so they exist from the function's first
> placement, interneuron or not. **A placement the election declined saves nothing and counts for nothing**
> but its place in the window: a function that is placed and never taken sees its held count slide out, its
> worth fall below zero, and is pruned (R18). Worth measures what a function actually saves.

**This is the only valuation in the design**, and it is read in three places, so the three numbers differ, and
are meant to:

| Read by                               | Over                  | Against                                  | Settles                            |
|---------------------------------------|-----------------------|------------------------------------------|------------------------------------|
| the machine, in the election (R22, R24) | the placed fits      | the board                                | what is credited, and so the count |
| the machine, keeping the dictionary (R50, R54, R18) | one relation in one function | the function's placements | which relations the dictionary holds, and in which function |
| the machine, in the match (R49)       | one inferred function | the frames it reaches back into          | whether the inference is proposed  |

One rule decides what a set of functions covers for a set of relations and names. The election runs it over the
board. It is stated here and cited from there.

> **D28 — Greedy cover.** The operation that covers a set of relations and names with **fits**, each landing
> its relations on them and its fixed nodes on the names, and pays each relation and each name to exactly one
> fit. It repeats:
>
> 1. **Measure** every fit not yet taken against the relations and names no taken fit has covered: its
>    coverage is what those it would cover cost, and the count of each call it takes back whole (R23), and
>    its price is its call's count there (D22).
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It
>    covers the relations and names it was measured on. A name its true relation lands on as a fixed node is
>    covered by it and leaves the set. A name at one of its variables is counted as its value and stays in the
>    set, so a later round may cover it (D10). The round repeats from 1.
>
> **What it returns is what was covered and by whom**: the fits taken, what each covered, and what no round
> covered, which stands bare. `price` is fixed by what fired and cannot change between rounds; `coverage` only
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
>   common but the place (D44), and a class whose members are the two neurons otherwise (D41).
>
> The general function is a new function over base symbols, with those variables at those nodes and the shape's
> relations, and it is held like any function: find reaches it through its open ends (R53), the election takes
> it where it pays (R24), and its worth is its own accepted placements (D57). It replaces nothing: the two it
> was made from stand where they are cheaper, which is wherever they recur. It is made when an interneuron
> fires and the index gives a function of the same shape (D36), and is not made again where the dictionary
> already holds it.
>
> **The pairing starts at the origins.** Two functions of the same shape are paired from their origins, and
> the pairing follows the relations from there, so two cycles over different neurons pair node for node in one
> way, the way they end, and a cycle read from another node is another function with another origin (D15).

Generalizing needs two functions to start from, each built on something that came round twice. After that, a
new thing fits the general function directly: any neuron for a parameter, and for a class once the member pays
its one name.

## 3.12 The objective

The cost of the run is the cost of two things (D13): the dictionary, every function's nodes and relations and
every class's members; and the graph of the run, its nodes and edges with the calls in place of what they
cover. There is one reading of it. A call is its count, what the call covers stands inside it, and what it
bound stands beside it. A neuron above counts an interneuron at what it cost below, so nothing is saved by
naming it again and nothing is gained by counting it twice. The election takes nodes off the frontier and puts
a call in their place (R22). Grouping and the moves do the same to the dictionary, one relation at a time (R50,
R54). Nothing computes the total: every quantity the design uses is a **difference**, which is finite however
long the run is.

## 3.13 Parameters

> **R4 — `H` is the only free parameter.** One thing is the designer's to pick: `H`, the depth of every
> neuron's history (D18) and of every function's placements (R50). Nothing else is. The alphabet (D1) states the
> problem and is not a knob. The reach (D4), adjacency (D5) and the offsets (D6) are fixed: a base neuron sees
> one position in each dimension, and an interneuron one beyond what its function covers. The precision `P` and
> the neuron capacity `N` are the hardware's (§3). The neighborhood policy (D50) and the action policy (D51) are
> declarations like the alphabet: each states the kind of world the machine is in, and neither is a knob. **No
> rule introduces a constant, a threshold, a window or a cap of its own.**

A base neuron sees one position because that is where every longer distance comes from. A relation across a
stretch is a chain of relations of one, each a function of its own or part of one, and the interneuron over the
chain sees one position beyond it, so its relations span the stretch in one offset. Every distance the
dictionary holds is built this way, level by level, and none is declared. Where a stretch is unpredictable
between its two ends, a variable has to form in the middle before the stretch can be covered (R52), and that is
the price of seeing one position: a deeper hierarchy, never a number nobody can derive.

# 4. Neuron

What this section defines belongs to one neuron: what it holds, and what it answers. A neuron records, counts
and returns. It decides nothing about what is covered, and it does not know that functions exist.

## 4.1 Interface

A neuron is a symbol, and a type (D2). It holds a history of its activations, a relation index over it, and its
connections (D25).

**The machine reaches a neuron through five calls, and nothing else writes into one.** Each is specified where
it is used.

| Call              | Description                                                                                                       | References |
|-------------------|-------------------------------------------------------------------------------------------------------------------|------------|
| allocate neuron   | The machine allocates it the first time a fit of a function is accepted and a neuron is free, holding nothing: the function's call neuron. | R16 |
| activate          | At its level: record the frame's activations, and return the relations from each with their strengths.             | §5, R20    |
| forget neuron     | The machine removes a decommissioned neuron from every row and relation of the history that names it.              | R38        |
| connect           | After the frame's last level, reaching every open activation at age 1 and beyond: it delivers what stood on the apex this frame and any reward, and collects nothing. | §7.1 |
| infer             | After the frame's last level, reaching every activation at age 0: it delivers nothing and collects what the activation infers. | §7.2 |

In `activate` a neuron records and returns, in the frame it fires. In recognition the machine places what the
dictionary holds, moves relations between functions, elects, allocates and counts. In `connect` and `infer` a
neuron learns what followed and infers what comes next. In selection the machine chooses among those inferences
one winner per dimension.

> **R1 — One decision point: the frame it fires.** A neuron is called once per level it fires at, for every
> activation of that level together, each at age 0: it refreshes its history and its relation index, and
> returns the relations from each activation (R20). Nothing else reads it until its next activation.

## 4.2 History

> **D18 — History.** A neuron's record of its own activations: the last `H` of them, oldest first, each holding
> the neighborhood it observed in the frame it fired (D7), each neighbor at its offset (D6) and with what it
> cost, and the activation's own cost beside them, since a function covers the activation too (D13, D22). It is
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

**A neuron records one kind of relation:** a particular neighbor at an offset from itself, in one dimension.
What is general across neighbors, a place where different neurons stand, is not seen in any one activation, so
no neuron records it. The machine makes it (R52).

> **D47 — Relation index.** What a neuron counts over its history (D18): every relation from its own activation
> that its rows hold, each with its **strength**, the number of rows that hold it. It is updated when a row is
> admitted and when a row is evicted, and at no other time.
>
> **An interneuron stands for what is under it.** An interneuron on the apex is a neighbor as itself, and as
> each of the base symbols it stands for, each at its own offset (D26, R28). As itself it costs what it cost. As
> the base symbols under it, together, it costs the same: a function that lands on all of them takes its call
> back, and one that lands on some of them holds there and saves nothing (D22, R23).

How the index is laid out is the implementation's. What it holds is not: a relation's strength is the number of
rows of the history that hold it, and which rows those are leaves the neuron for nothing.

## 4.4 Connections

What follows an activation enters no cost (D12) and no test. It is held only here, as connections: what stood
on the apex in the frames after it, event and action alike.

> **D25 — Connections.** A neuron holds, per `(neuron, offset)`, one **connection**: a **strength**, the number
> of times an activation of the neuron saw that neuron stand on the apex at that offset, and its **earnings**,
> the total reward the actions of those frames received. Its **estimate** is its earnings over its strength,
> and is read, never stored. The offset is a full D6 offset from the holder's activation's place: positive in
> time, because what is connected to comes after, and signed in every other activation dimension. A connection
> to a call is at the offset of the call's origin, where it stands (D53). **A connection reaches as far as a
> neighbor does** (D5): an activation connects to what stands on the apex within its reach in every activation
> dimension the two share (D4), the window in time (D9) and the same reach in every other, and to nothing
> beyond it. **A connection is written only where the future table of the policy admits the target's kind among
> the holder's kind's connections** (D50). Nothing about any one activation is kept.
>
> | Component | Description                                                                              |
> |-----------|------------------------------------------------------------------------------------------|
> | key       | `(neuron, offset)`, the offset strictly positive in time                                 |
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

The call runs before the machine places and groups (R53, R50). An activation that will be covered at age 0
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

**An admitted row adds its relations.** Every neighbor in the row gives one relation per dimension it shares with
the activation: that neuron, at its offset from the activation in that dimension. The relation's strength goes
up by one.

**An interneuron gives its relations twice.** A call standing on the apex is a neighbor as itself, at its
origin's offset, and as each base symbol under it at that symbol's own offset (D26, D47). Both kinds join the
index. The second kind is worth the interneuron's count together and nothing singly (D47).

**A bound neighbor gives worthless relations where its binding covered it.** A neighbor bound in a variable at a
lower level costs nothing in each relation its binding covered, so that relation is marked: it holds a node and
covers nothing (D10, D22). A relation to it that its binding left open costs an offset, and counts like any
other.

**An evicted row takes its relations away.** Each relation it held loses one from its strength, and a relation
whose strength is zero leaves the index.

**A decommissioned neuron is forgotten.** The forget neuron call (§4.1) removes every relation that names it, and
every row's entry for it. Nothing else touches the index.

## 5.3 Return

> **R21 — What the call returns.** One return for every activation the machine handed in, divided by
> activation. For each activation it holds one relation per neighbor per dimension the two share (D34), each
> with its strength, the number of rows of the neuron's history that hold it, this one among them. An
> activation with no neighbor has nothing in its part.
>
> **The machine reads every relation in its one form** (D34). The neuron at each end returns it from its own
> side, so a relation between two activations of this level comes back twice, and the machine keeps the larger
> strength. A relation to an activation of an earlier level or frame comes back once, from the activation that
> fired now.

A relation held in one row cannot recur yet, so **nothing is ever built on a single occurrence** (D57, R50); it
is still returned, since a function already in the dictionary may hold it (R53). What the call returns is the
neuron's whole say in what is covered: what it saw, and how often. Which rows hold what never leaves the neuron.

## 5.4 Cost of the call

With `A` activations this frame, `M` neighbors per row and `d` spatial dimensions:

| Step | Work |
|---|---|
| refresh history | `A` rows admitted, as many evicted |
| refresh index | `A · M · (d + 1)` relations counted in and out |
| return | `A · M · (d + 1)` relations, each with one number |

# Part III — Past and present: the machine

# 6. Recognition

> **D23 — Recognition.** The machine covers the board with interneurons, each fit taken when the relations and
> names it covers cost more than its call's count, never counting what its function costs in the dictionary
> (R22, R24).
>
> **Covering everything is not the goal.** An activation no accepted function covers stays in the count as
> itself, at what it cost (D13), and that is the shorter count whenever no function could hold it for less.
> **What coverage varies is the cost, never the fidelity.**
>
> It is **axis-general**: a function names nodes at offsets, so an interneuron replaces a piece of the graph
> however it lies across time and space. A purely spatial function is the case where every offset in time is
> zero. A temporal function is recognized at its latest node, in the frame that node fires, over what has
> happened.

**The board** is the apex as it stands at a level, partway through the frame: what the levels below have left
uncovered so far, with the open activations of earlier frames still within reach, and the relations the neurons
returned among all of them. It is a piece of the graph (D55), and it changes with every level. The apex of the
frame is the board once the last level has run.

**The machine compresses a frame in this order and no other.** One level at a time: the first over what the
frame reported, each after it over what the level before accepted, until a level fires nothing:

| Step       | Description |
|------------|-------------|
| activate   | Fire every interneuron the level below accepted, at its fit's origin (D15, D53), and call `activate` on every neuron with an activation at this level (§5). They return their relations with strengths. |
| find       | Read every returned relation in the index, as it is and with either end open, and place every function it reaches on the board wherever its relations match, one fit per place (R53, D31). |
| group      | Over the counts as they stand, move each relation to the function where its worth is highest, drop each whose worth has turned negative, and start a function for each recurring relation no function holds (R50, R54, R18). |
| elect      | Take the fits best first, the highest coverage over price each round, stopping when the board is covered (R24). |
| allocate   | Give each accepted function with no interneuron one, reused where a stretch comes round (R43), allocated otherwise where a neuron is free (R16). It fires at the next level. |
| count      | Record each fit as a placement of its function, with the election's verdict on it: for an accepted fit, which of its relations came true; for a declined one, nothing (R50). |

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
> frame reported, all of them together: they are activated and return relations, the machine places what the
> dictionary holds and groups, and the election settles which fits are accepted. The next level is the
> interneurons of those fits:
> they fire, are activated and return relations, and so on upward. **When a level fires nothing, the frame's
> levels end there.** Nothing declares the depth and nothing caps it. A level belongs to a frame and not to a
> neuron: a neuron fires at whatever level accepts its function, and one neuron may fire at different levels of
> different frames.
>
> **Within a level the order is activate, find, group, elect, allocate, count.** It cannot be otherwise: what
> fires is what the level before accepted; what a neuron returns is what is placed; the moves read the counts
> as they stand; the election is over everything placed, as the functions now stand, since a function changed
> last may cover the board better than any placed first, so nothing is taken until the whole board has been
> read; an interneuron is allocated only for a fit the election accepted (R16); and a placement is counted with
> the election's verdict on it, which is known last. The dictionary changes in group, and the index is read
> before and after it.
>
> Every neuron runs the same rule at its own reach (D4). **Compression is spatio-temporal for every neuron, in
> one pass**: a function may name neighbors in its own frame, in earlier ones, beside it in space, or in a mix.

> **R53 — Finding.** For each relation `(a, b, δ)` a neuron returned, the machine reads the index (D36) three
> ways: as it is; with `b` open, `(a, ?, δ)`; and with `a` open, `(?, b, δ)`. The reading as it is gives the
> functions that hold that relation. An open reading gives the functions that hold any relation with that
> neuron at that end and that offset, whatever stands at the other end: a neuron, a class or a parameter. Where
> the open end holds a class the observed neuron is a member of, or a parameter, the dictionary function is fit
> by the observed relation the way D54 reads it, and this is where variables are matched.
>
> Every function any reading reaches is **placed** on the board (D31), from its origin: a reading tells the
> machine the function may fit, and the fit begins on an activation at the present frame that can stand at the
> function's origin, or nowhere. Matched relation by relation out from the origin, those of its relations that
> land on a board relation are true, those that do not are its failures (D22). A function is placed once per
> activation that can stand at its origin, and as many times as there are such activations. A relation no
> reading reaches is left for grouping (R50). Everything find places goes to the election once every relation
> of the level has been read (R24).

> **D31 — Placing a function.** A function find reached (R53) is placed on the board, on the board relations it
> matches (D15). Competing matches over the same board relations are all placed, and the election picks among
> them (R24). The same function matched to the same board relations from two of its relations is placed once.
>
> **A function is placed whole.** The machine takes it or leaves it, its variables with it: it cannot hold where
> they do not.
>
> **The placing rule.** A fit is anchored at the origin. The machine lands the function's origin on an
> activation at the present frame whose neuron is the origin's, or is admitted at it where the origin is a
> variable (D54), and extends the match relation by relation from there: each next relation of the function
> that shares a node with one already landed is matched to a board relation that shares that end and reads as
> it says. A parameter's value is the far end where its node is first reached, and a relation reaching the node
> again is matched to that neuron (D44). Where several board relations could answer one relation of the
> function, the one no accepted call covers is taken first, then the older one (R24). A relation no board
> relation answers does not come true, and the match goes on from the function's other relations. Every
> relation can fail but the one thing the fit rests on: where no activation can stand at the origin, there is
> no fit and nothing fires (D15). What stands in each variable on that occasion is the call's arguments (D37).
> Expansion reads a call back by the same rule (R28).

> **R50 — Grouping.** Every function the index holds is a center for the relations grouped into it, and every
> fit of it (R53) is one **placement**. The machine keeps, per function, the counts the worth is read over
> (D57), recorded once the election has ruled on the fit (R24):
>
> | | |
> |---|---|
> | the function | one more placement per fit, accepted or declined |
> | each relation of the function | at an accepted placement: came true, anywhere in the fit, or did not; at a declined one, nothing |
> | each candidate | a board relation the neurons returned with strength 2 or more, that the function does not hold and that shares a neuron with an activation the fit landed on, read over the frames the call's activation is open: at an accepted placement, attached; a candidate seen at an earlier accepted placement and not attached at this one, absent; at a declined one, nothing |
>
> Each count runs over the function's last `H` placements, as a history runs over a neuron's last `H`
> activations (D18), and nothing else is kept. Grouping reads the counts as they stand before the election,
> and the relations move (R54): a candidate joins the function where its worth is highest, when that is higher
> than its worth where it is; a relation whose worth in its function has turned negative leaves it (R18). A
> relation the neurons returned with strength 2 or more that no function holds starts a function of its own,
> two nodes and an offset, credited its strength as accepted placements it came true at, since each occurrence
> of it would have been one (D57). A relation held once is left where it is: nothing is built on one occurrence
> (R21).
>
> **Relations join only through a node, and a function grows at its latest end.** A candidate is attached where
> it shares a neuron with an activation the fit landed on, so every function stays connected (D15). The relation
> that extends a function in time arrives in a later frame, from the new activation back to the call's origin or
> to a node it covers, and is attached to the call while its activation is open; so what joins is what always
> followed, the function's origin moves to the new node, and the next time the function is recognized it is
> recognized there and takes the earlier call back (R23). A candidate whose other neuron is already a node of
> the function closes a cycle. Two relations that never touch are never one function: where they recur
> together, the level above relates their two calls.
>
> **The fits the election chooses among** are every placement of this level, of functions with an interneuron
> and without (R24). A function that changed at this level is placed again before the election, with its
> relations as they now stand.

> **R54 — Moving.** A function changes by relations moving, one at a time, and by nothing else. There are three
> moves, and each is the same decision, a relation's worth in a function (D57), read on the function's
> placements (R50):
>
> | Move | When |
> |---|---|
> | a relation joins | a candidate's worth in the function is higher than where it is: in its own function of one relation, or in another function |
> | a relation leaves | its worth in its function is negative (R18) |
> | two functions merge | a relation of one is a candidate of the other and joins it, and the rest follow one at a time, each where its worth is higher |
>
> A relation that leaves its function of one relation leaves that function empty, and the function goes with
> it (R18). A function keeps its interneuron through every move: what the interneuron stands for moves with its
> function. The relations it held before a move stay readable through the dictionary until the windows over the
> activations the old form fired have closed (D56, R38), so expansion reads each call through the function as
> it was when the call fired. A function that a move makes the same as another is that other function (D36),
> and the two are one from then on, the older interneuron standing for both (R43). A moved function is read
> against the functions of the same shape, and generalized with them (R52).

> **R18 — Prune.** A relation leaves its function when its worth is negative (D57), read at each placement of
> the function, accepted or declined (R50). A relation that leaves with no function stands bare wherever it
> holds. A function leaves the dictionary when its last relation does. **Relations leave one at a time**, the
> most negative first, and the function's other relations are priced again before the next leaves, since a node
> two relations bring is charged to whichever is read first. A class no function has any longer goes with its
> last function. A function's interneuron is decommissioned when due (R38).

## 6.2 Election

> **R22 — What a fit covers on the board, and what it costs.** A function is placed on the board (D31). The
> machine reads the fit against what has fired and derives both numbers.
>
> | | |
> |---|---|
> | the fit | the function and its interneuron or none (D31), its relations landed on the board's relations, with what each variable binds |
> | coverage | A: the board relations its true relations stand in for and the names its fixed nodes land on, where no accepted call has covered them, and the count of each call it takes back whole (D24, R23) |
> | price | B: its call's count (D12, D22): its name, one failure per relation that did not come true, one value per variable, and, where the function holds a cycle, how far the fit ran round it |
>
> The machine places the function by D31's rule. Coverage less price is the savings over stating what it covers
> bare, D22's expression over the board.
>
> **A relation is covered once.** A fit whose relation lands on one an earlier accepted call already covers is
> credited nothing for it (D10, D24).
>
> **A node that fired and the fit does not land on belongs to neither side.** Its name and relations stand in
> the count as they are if nothing covers them (D23), and the same way if this fit is accepted, since the fit
> does not land on them: their own cost either way, so they cancel before the test begins.
>
> **Coverage changes the credit and never the price.** A relation that came true on a board relation another
> call already covers is credited to no one and charged nothing, unless the fit takes that call back whole
> (R23): it came true, so it is not a failure. A relation that did not come true is charged whether or not
> another call covers something at that place. **What a function gets wrong about a frame is a fact about the
> two, and ownership does not move it.**
>
> **This is the fit's arithmetic over the board, and it differs from the worth.** The worth reads a relation
> over the function's placements (D57); the election reads the board, where earlier frames' credit stands (R23)
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
> | per relation on the board | the accepted call that covers it, or none, in which case it is counted bare |
> | per node | whether its name is covered by a call, as a fixed node; counted as a value, inside a call that binds it; or bare |
>
> **Covered and bound are read off it.** An activation is **covered** when a call covers its name: it has left
> the apex (D10). It is **bound** when a call covers its relations and its name is a value: it stays on the apex
> at its name's cost. It is on the apex, bare, when nothing covers its name.
>
> **Credit is what blocks later payment, and coverage is what hides.** A relation a call covers is paid for
> through that call. It is in a later fit's coverage only together with everything that call covers: the fit
> takes the call back and is paid the call's count (R23). A fit that lands on it afterward without the rest is
> neither paid for it nor charged a failure for it, since it came true. A bound activation's name is still on
> the apex and is a neighbor to every later function. A later fit that lands on it as a fixed node covers its
> name, which is all that is left of it in the count. A later fit that binds it again covers nothing new (D10).
> So every relation and every name is paid for once. The coverage set spans one frame more than the longest
> reach in time among the neurons the machine holds, no function reaches further back, and the box every other
> activation dimension gives, and ages out with it, and over the same span the machine keeps the action that
> ran at each frame in each action channel, for the shares a reward sends back (R33). **The machine holds
> nothing on the scale of the run.**
>
> The coverage set is about **credit**: a relation or a name is a fact that needs paying for exactly once, so
> one a call covers is paid for through that call until a later call takes that call back whole (R23). It is
> not about naming: an interneuron expands to everything its function names, credited or not (R24).

> **R23 — This frame's fits against the board as it stands.** Only what is counted is in play: a bare node's
> name and relations at their cost, a bound node's name, and the relations and names a call covers at that
> call's count, together. So no relation and no name is paid for twice. Within the electing frame a relation is
> covered once, by the first accepted fit whose relation lands on it (R24), and never moves.
>
> **A later frame takes the past back where it covers it shorter.** A fit at `f` whose relations land on every
> relation a call at an earlier frame within its reach covers takes that call back: the new call covers them,
> the old call leaves the count, and its interneuron's activation leaves the apex. What that interneuron
> learned stands. A fit that lands on some of what a call covers takes nothing: a call is whole. So the past is
> re-elected only by a fit that covers a call entire, and only where that pays on the board (D28).

**The cost over one frame is the calls made plus the activations no function covered**: the price of every
accepted fit, and what each activation no function covered cost, the graph's half of the objective (§3.12) over
the frames the election can see. **The two terms do not overlap**: a neuron a promoted neuron fails to name is
in the second and not the first, which is why the price counts only the neighbors named and absent (R22). The
dictionary half is the worth's (D57), and neither test touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of
fits and scores it**: fits are taken one at a time, each measured against what the ones before it left, and
the election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy,
and returns a good solution, not a proved minimum. **Every activation a fit covers ends up credited to exactly
one fit**, which is what stops an activation being paid for twice.

> **R24 — The election is D28, run by the machine.** The fits are everything find placed at this level, over
> the functions as grouping left them (R53, R50, R52), less any whose function would need a neuron when none is
> free (R16), and what they cover is the **free set**: every relation and every name on the board that some fit
> lands on and no accepted call covers, and every call within reach whose every covered relation a fit lands
> on, at that call's count (R23).
>
> Each fit stands on the board before the first round (D31), and `price` is R22's. Then D28 runs: the highest
> `coverage / price` first, each while `coverage > price` against the board as it then stands, and it stops
> when nothing on the board is left that a fit could cover for less than it costs.
>
> **Two fits that measure the same are told apart in this order**: a function with an interneuron before one
> without; between two with interneurons, the older interneuron; between two without, the function that joined
> the dictionary first; and between two fits of one function, the one reached from the older activation.
>
> **The bound is structural.** A fit's price includes its call's name, so it is taken only when it covers more
> than one name's worth: at least one relation, two nodes. Each round takes one fit, so the rounds are never
> more than the fits.
>
> **A fit that never reached the top held nothing.** There is nothing to hand back and nothing to settle: a
> relation or a name it landed on is either covered by a fit that did pay or stands bare.
>
> **Ownership is a partition of the relations and names the accepted fits land on**, and that is the whole of
> the inhibition: no function is ever edited or forbidden, and **overlap is legal and priced**: a fit that lands
> on a relation an earlier round covered gains nothing for it and pays nothing for it, unless it takes that
> call back whole (R23). One that lands on a bound node as a fixed node gains its name, and one that binds it
> again gains nothing (D10, R22). **Covered by an accepted fit** and **landed on by an accepted fit** are
> therefore the same set, so coverage, credit and the apex frontier (R27) are one question with one answer.
>
> **Outcome**: accepted fits are promoted, each **whole**, and an interneuron expands to everything its function
> names, credited or not; the activations whose names they cover leave the apex and what they bind stays on it
> (D24), and every active neuron no accepted fit covers stands as itself. **The election delivers nothing to
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
> granted one, so its fits are not in the election (R24). It stays in the dictionary on its worth, is placed and
> counted as before, and is elected again when a neuron is free.
>
> **Release is the reverse**: the function leaves the dictionary, and the machine decommissions its interneuron
> once no other function of it remains (R38).
>
> **Allocating is immediate.** The machine allocates or reuses the interneuron the moment the fit is accepted,
> which is all the next level needs to activate it at the fit's origin (D15) and call it.

> **R43 — Reuse.** An accepted fit fires its function's own interneuron where it has one. Where it has none,
> the machine looks the function up in the interneuron index, by what its interneuron would stand for (D36).
> Where one already stands for that, the function is added to it, and that interneuron fires. Otherwise the
> machine allocates one.
>
> **Otherwise the machine allocates one** (R16): a free neuron, its first activation at the fit's origin (D15).
> Where none is free the fit was not in the election (R24).

## 6.4 Activate interneurons

An interneuron fires at the level after the one its fit was accepted at, before that level is called. It fires
at the fit's origin (D15, D53), once per accepted fit, at age 0, and what stood in its function's variables
stays on the apex beside it, bound (D10). Nothing sees an interneuron until it fires
(D5). An interneuron's activation records the function that fired it, since it may have several (R43), which
of that function's relations did not come true, and how far back round a cycle it ran, and is expanded through
that function (R28). Every activation no accepted fit covers stands as itself.

> **R27 — The apex is a frontier.** It is every active neuron **no accepted function covers** (D49), the
> uncovered set, base neurons and interneurons alike, so a base neuron no function found worth naming stands in
> it beside an interneuron many functions deep. This is the frontier the cost counts, **the one every neuron
> sees as its neighborhood** (D5), **the one that learns what ran** (D25), and **the one that votes** (§8.4): the
> uncovered set does all four, and coverage silences a neuron in every one of them at once, its one last
> connection aside (D10). A reward for a frame already counted reaches its connection regardless (R33).
> Everything underneath the current frontier is recovered by expanding it.
>
> **Named, not covered.** A neuron a declined fit named, with nothing else covering it, is still on the
> frontier, and stands in the count as itself.
>
> **Coverage is acquired late and passes only whole.** An activation that fires at `g` is uncovered until some
> fit takes it, and a fit can take it for as long as it lies within reach of its origin (D4), so an activation
> may speak for a few frames and then fall silent. A later call may take it from its coverer, with everything
> that coverer covered, and it stays covered (R23).

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

> **R38 — An interneuron dies when its last function leaves the dictionary and the last window over it
> closes.** Dying is decommissioning: the neuron is free to be allocated again. Once its last function has
> left, nothing can fire it again. Every function that names it as a node leaves the dictionary with it, since
> it can no longer hold, and so on up: that is its **subtree**. The death frame is the frame in which the last
> open activation of the interneuron, or of any neuron in its subtree, closes its window, and this frame when
> none is open. That set only shrinks. An activation's window closes at its reach in time (D9), so the wait is
> at most the longest reach in time among the interneuron and the neurons built on it.
>
> **Decommissioning is after the frame's connect and infer calls** (§7), so the last exposure an activation
> writes at age `reach_t` is taken before its neuron goes. Every frame, after those calls, the machine takes
> everything due, and calls forget neuron on every neuron whose history names one (§4.1).
>
> **The dictionary holds the function.** Until the windows of the activations a function fired have closed,
> calls above them may still expand them, so the function stays readable after it leaves the dictionary, and so
> does each form a function had before a move (D56, R54).

## 6.6 Cost per level

With `G` relations returned per activation, `F` functions reached per relation, `L` relations per function and
`C` candidates attached per fit:

| Step | Work |
|---|---|
| find | per relation, three index reads, widened by the classes its far neuron is in; per function reached, `L` relations matched per fit |
| group | per candidate, one worth where it is and one where it would join; one worth per relation whose count changed |
| elect | the fits placed, a greedy cover over the board, stopping when it is covered |
| count | per fit, `L` counts and `C` candidate counts |

Nothing writes a row and nothing waits on a merge. Everything is bounded by what the level returned and by `H`,
and nothing scales with the run. What is cached, and how, is the implementation's.

## 6.7 Cycles

Two things that are cycles, told apart, because the design holds them in two different places.

**Within a function, a cycle is the pattern repeating.** A function has one node per neuron, so a relation
back to a node already in it closes a cycle, and the function says that the pattern comes round: `x` to `y` at
1 and `y` to `x` at 1 is `x y x y …` for as long as it runs (D15). A fit at the latest letter reads back round
the cycle as far as its relations keep coming true, and the call counts how far it went (D12). A relation a path of the
function's relations already implies is never held (D34), so a cycle is never a fact paid for twice.

**Through interneurons, recursion.** A period no single function can hold, because the same neurons come in an
order a graph over them does not fix, is held as a chain: each function "this symbol, and the call before it",
two distinct neurons, each covering the call before it, and the last naming the first through the dictionary.
Six functions for six bars whose moves repeat. That is a cycle in the dictionary and not in any function:
unrolled over the run it is a chain, since the call at one frame and at the next are different nodes.
Expansion follows what fired, frame by frame, and stops at reach (R28), so it never runs round. It is how a
periodic world is covered for a fixed dictionary: each call covers the one before it, the whole run stands as
one node at the end, and the dictionary says the rest, with one failure wherever the chain breaks.

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
| the apex | every activation standing on the apex this frame within the activation's reach in every dimension but time (R27, D25), and only if the activation is uncovered at this frame. One an accepted function covered this frame is handed the neuron that covered it and nothing else; one covered earlier writes nothing more (D10, D25, R31). |
| the reward | any reward that arrived, for the actions of this frame and for any earlier frame the reward spans, at the distance each one names, with the action that ran at that frame in each channel it pays (D24, R33). |

The neuron strengthens a connection per apex activation and saves nothing: the connection at `(neuron, age)`,
created at strength 1 or incremented (R31), and for each reward share the earnings of the connections it names
(R33). For this frame, the connections just strengthened, in the same write. A share for an earlier frame, at
distance `d`, goes into the connection at offset `a − d`, the activation's age less the distance, to the action
the machine hands it for that frame, where the activation was uncovered then (D9). The activation keeps nothing
of it. **Nothing is decided, priced or compared here**, and no test is waiting on any of it.

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
> currencies, the relation in the count and the connection in the reward, and the reward is learned on the
> connection (R34).
>
> **One activation connects to the apex of every frame it is open through, and one strong apex activation is
> connected to by every uncovered activation open when it stood.** The offset is the age, the distance from the
> activation's place to the frame the call ran, exact (D6), so a neuron open at ages 1, 2 and 3 holds the same
> neuron at three offsets, and the exposure at each age strengthens its own. A neuron that stood at two ages
> over one activation is two exposures: each frame has its own reward, and each connection keeps its own (D25).
> A connection to a call is at the offset of the call's origin, the frame the call fired in.
>
> **An exposure is written at its age, and an inference is placed at its offset.** An activation that saw a
> neuron at age 2 strengthens the connection at offset 2, and an inference at that offset is placed exactly 2
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
> to no channel (D2), and a share placed through its expansion would give a call spanning two channels one
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
> on either** (R34), so a coarse reward costs accuracy in the estimate and nothing in the count. What it does
> move is the walk: an unscoped reward carries its sign into every estimate it reaches, so the sign test R37
> reads shifts the same way for all of them, and the walk runs while the world is going badly overall and rests
> while it is going well. That is the meaning of a signed reward, and it is intended.

> **R34 — Two objectives, meeting at one place.** Everything structural is priced in cost; reward prices
> nothing structural and cannot. The machine runs **two** objectives: compression, which decides what structure
> exists, and reward, which decides which of it is executed. They meet at exactly one place, a neuron's action
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
nothing more specific covers it. A newly allocated
interneuron has no connections and infers nothing: what it covers speaks for it (R48).

**An activation's window closes at age `reach_t`** (D9), once that frame's exposure is taken. The window closing
stops the connect calls and drops its standing inferences (R45), and nothing else: no call marks it and nothing
is saved twice.

> **R36 — Inference.** One connection read by one activation at age 0: it names a neuron, base or interneuron
> (D25), and carries the strength and the estimate the connection held then. An activation reads every
> connection its neuron holds, at every offset out to its reach, once, in the frame it fires (§7.2), and never
> again: what the walk wires or a reward moves afterwards is read by the neuron's next activation. The winner of
> every dimension is chosen from the inferences and nothing else (R47).
>
> **An inference is placed at its completion, and a connection means what it recorded.** A connection at offset
> `b` was written when that neuron stood on the apex `b` frames after the activation's frame (R31), so an
> inference at offset `b` is placed `b` frames after the activation that made it, and at the offset's components
> in every other activation dimension from its place, as a neighbor is placed (R28). An inference that names a
> base symbol places it there. One that names a call neuron places a call there, at its origin, weak (D37), and
> the call's function hangs back from it (R28): its nodes at the frames back from the origin, every one carrying
> the strength and the estimate of the connection it came from, its parameters bound from what stands at their
> nodes where those have happened (D44). What its expansion puts at the frame ahead is what it proposes there,
> and the machine expands no more of it than that (R45): at distance 1 the last thing a function places, at
> distance ten the one ten from its end, and the first only when its completion is placed its whole length
> ahead. An inference is proposed only where its match holds (R49), and resolution runs at the base and only
> there (R47). **This is what launches a program**, and it is why every offset is read and not the next one
> alone: an activation can start a program only from an offset at least as far out as the program is long, and
> can carry the tail of a longer one to completion from any offset. A situation whose reach is shorter than a
> program never starts it, and that is correct: it cannot see that far.
>
> **Two places are two voters.** Two activations of one neuron read one set of connections (D11) and place the
> same inference from two places: in a channel whose only activation dimension is time that is one proposal
> twice, and in a channel with a layout it is a proposal at each of two places, each resolved where it lands
> (R47).

# Part V — Future: the machine

# 8. Selection

Once every open activation has been called (§7), the machine holds what the apex infers, and it works from
there down to the base alphabet, in this order and no other:

| Step     | Description                                                                                                                                  |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| keep     | Hold every inference an activation returned at age 0 as a standing inference, until that activation's window closes (R45).                     |
| expand   | Expand of every standing inference what comes due: the base symbols it places at the frame ahead are what it proposes (R28).                   |
| match    | Read each standing inference's function against the frames it reaches back into; propose it where the function would pay there, strike it where it would not (R49). |
| resolve  | One winner per dimension at the frame ahead: in each action dimension by estimate, in each event dimension by share of voters (R47).           |
| activate | The winner of every dimension fires weakly at the frame ahead (D40), action and event alike, and both are output; the world's report replaces them, in part or in whole, and what ran arrives with its reward (R40). |

No neuron takes part in any of it. What an activation returned is kept, placed, expanded, matched, resolved and
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
> being what happened, and nothing is retracted. An inference stands whether its activation is covered or not;
> coverage decides whether it is heard (R46).
>
> **What stands is what was read.** A reward may move an estimate after an activation's age 0, and the walk may
> wire a connection. The neuron's next activation reads them (R36). This one does not. A situation that launched
> a plan keeps proposing it at the estimate it read, until its window closes or its match fails. The rewards the
> plan's actions bring as they run fold into the connection's estimate on the neuron (R31). The standing
> inference does not read it again.

## 8.2 Expansion

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> what its function covered and bound: the nodes its function names, each where the function's relations put it
> from the origin, repeated down to base symbols. It follows the function's relations back from the origin, one
> dimension at a time:
>
> 1. **Start** at the origin: the call's place, in every dimension, where the function's origin stands (D15).
> 2. **Follow** each relation that came true, from a placed node to the node at its far end, in that relation's
>    dimension. Offsets compose because each is a difference (D34). Round a cycle as many times as the call
>    records (D12). Repeat until no true relation leads to an unplaced node in any dimension.
> 3. **Leave** a node no true relation reaches where nothing is: whatever stood there was counted bare beside
>    the call, and the call says nothing about it.
> 4. **Expand** each placed node that is an interneuron the same way, through the function that fired it (§6.4),
>    from its own origin, which is where it was placed. Expand each variable to its argument, the neuron bound
>    there (D37).
>
> Each activation is expanded through the function that fired it, so expansion follows what actually fired,
> frame by frame, back to where it began, and it ends there, even where a function comes back to its own
> interneuron (§6.7). A fixed node expands to its neuron; a variable expands to its argument.
>
> **Every placement is exact.** A node is placed at exactly the offset its relation names, composed along the
> path from the origin, and an expansion many levels deep places its farthest base symbols exactly where the
> relations along the path put them. Nothing is rounded on the way down.
>
> **Placing and expansion are one rule read in two directions.** Placing (D31) lands the function's relations on
> the board back from its origin. Expansion starts at the origin and lands the relations back on the board.
> Both read the function and what the call recorded and nothing else, so what one covered the other recovers.
>
> **This is the one expansion in the design.** It recovers the run from the dictionary and the calls (D12): a
> class to the member bound at its node, a parameter to the neuron bound at its node. And it turns an inferred
> function into a program (R30): the actions it places run, and the events it places fire weakly beside them
> (D40). It reads the dictionary only. **What travels down with a symbol is what the connection carried**: every
> base symbol an inference's expansion places carries the strength and the estimate of the connection it came
> from, and nothing is re-weighted on the way down.
>
> **Expansion substitutes the arguments.** A recognized call's arguments stand on the apex, bound, and need no
> substituting. An inferred call brought none, so its variables are bound from what has happened: a parameter
> from what stands at its node where that has happened, and placed where the node is reached again, which is
> how a value is carried forward; where it has not happened it places its most frequent member. A class with no
> argument places its most frequent member, the oldest on a tie. A function above that names an interneuron
> alone expands to the call with its arguments left to be bound, and one that names the interneuron and a bound
> activation expands to the call with that argument. **Two words, never one.** A function is **expanded**,
> downward, into what it covers; a connection is **inferred**, forward, into the frames ahead (D25); what a call
> does and what follows it are different directions.
>
> **A connection is placed the way a neighbor is.** A connection at offset `b`, read by an activation at its age
> 0 (R36), puts the neuron it names `b` frames after the activation's frame, that is where it completes, and its
> expansion hangs from there, its base symbols at the frames back from it. At a later age `a`, the frames
> elapsed (D9), that completion is `b − a` frames ahead. So the base symbols of a long program reach the frame
> ahead one at a time, in order, each at exactly one age, from one connection and with nothing held: what an
> activation places beyond the frame ahead it places again next frame, one frame nearer.

> **R30 — Execution is an expansion**, of the selected function (R47) through its nodes and relations in the
> dictionary (R28). A high function becomes its nodes at the offsets it records, down to base symbols. Its base
> actions are output, each in the frame its expansion places it in, the nearest being `+1` (R29); its base
> events are placed in theirs. Both fire weakly (D40), and the report replaces them: the actions it names as run
> and the events it names as seen become strong, and the rest last their frame. Execution is not a second
> mechanism; it is this expansion read as a program. A weak function's expansion places no action: what is
> output is chosen by estimate, never by a function that was itself inferred.
>
> **Execution activates what it expands, where it won.** A base action or a base event fires weakly in the frame
> R47 selected it, and an interneuron the expansion passed through fires weakly when its expansion completes, at
> its origin, the last frame of what it covers, which is the place recognition would have given it, provided the
> standing inference that placed it still stands (R45) and what it placed at that frame was selected (R47). An
> inference that lost its dimensions, or whose match failed on the way, fires nothing. Where the report makes
> its actions strong, recognition builds the strong activation of the same function at the same place, and it
> replaces the weak one (D40). An executed function is therefore in its frame's levels like a bought one: it is
> activated, may be covered (§6.1), and every uncovered activation open at that frame connects to it (D25).
> While its program runs, what stands on the apex at each frame is the base action it placed there, or a lower
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
> each at an offset, and its variables, each at its node, placed back from the completion the inference names,
> one expansion down and no deeper. Whatever is placed at the frame ahead or beyond is what the inference
> proposes, and is not judged. Against them stands the apex of each frame within the longest reach in time, as
> the machine keeps it (D24).
>
> **What each placed node is worth.**
>
> | Placed node | Worth |
> |---|---|
> | a neighbor standing at its offset | credits what it cost |
> | a neighbor standing nowhere at its offset | charges its relation, one failure (D13) |
> | a variable | bound from what stands at its node where that has happened (D44), crediting its relations; a node holding nothing it could take charges its relation |
>
> An offset is read exactly (D6): a neighbor stands at its offset or it does not, and a neuron of the right
> kind at another place is not it.
>
> **The test.** The inference is proposed when what the placed nodes credit exceeds what they charge, the
> relations that did not come true included, and struck otherwise. The call's count is not charged: the match
> covers nothing, it only asks whether the frames that have happened bear the inference out, so it is D22's test
> over the nodes that have happened, with B holding the relations that did not come true and nothing else.
>
> **What it decides.** Proposed, the inference's placements ahead contend for their places (R46, R47). Struck,
> it leaves the standing inferences (R45). An inference that was proposed and stops matching as its placements
> come due is struck then, which is the whole of how a plan ends before its span does.
>
> **Where it stands among the three.** The election reads the placed fits against the board and credits them
> (R24). The worth reads a relation against its function's placements (D57). The match reads one inferred
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
> **A voter is one apex activation**, speaking from the connections its neuron held when it fired (R36). A base
> neuron's estimate is a marginal over every situation it has stood in; an interneuron's is over those its
> function recurs in, one situation; so the same frame reads
> differently to a base neuron, to the interneuron covering part of it and to the interneuron covering the
> whole. **A function's interneuron is how one recurring situation acquires an estimate of its own**, which is
> why a covered neuron is silenced (D10) and its coverer speaks instead, and the only reason a base neuron's
> marginal ever speaks is that nothing more specific was bought over it.
>
> **Every apex activation votes**, base actions included. A function that names what was done beside what was
> seen votes knowing the situation; an interneuron over actions alone, or a base action, votes what followed it,
> a habit, and reward prices the habit like anything else. What an inference places in an event dimension is
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
> machine holds at `f` (D9), less every activation an accepted function covers (D10), plus every covered
> activation whose coverer has nothing for `f + 1`, and so on down through the covers to the base (R48). What
> each voter proposes is what its standing inferences place at `f + 1` (R45): a base action by the inference at
> the offset one ahead of its age (R29), a base symbol a function places by a farther one, and nothing where
> none of them lands there. A
> voter is read as `(neuron, place)`: the inferences are its neuron's (D11), and the placement is from its place
> (R36).

> **R47 — Resolution.** One winner per dimension per place at `f + 1`, action and event alike (R30), and
> nothing about where a voter stands in the hierarchy is read in it: an interneuron's inference and a base
> voter's meet on the estimate or the share alone, and a specific situation wins over a general one only by
> being right about what pays, never by rank.
>
> **An action dimension resolves by estimate.** At each place of it, every base action some voter's expansion
> placed there is a candidate (D8). **A candidate's estimate is the mean of the estimates its voters placed it
> with, each weighted by that voter's share**, one vote per voter, split across the actions it placed at that
> place by strength, so a voter that hedges between two actions counts as one voter and not two, and **the
> candidate with the largest estimate is output**, unless every candidate's estimate is negative, where the
> action policy decides between the least bad and nothing (D51). Ties go to the larger share of voters, and then
> to the older action. **Nothing corrects for how many exposures an estimate rests on**, so a sharp estimate on
> three exposures outranks a coarse one on two hundred.
>
> **An event dimension resolves by share of voters.** At each place of it, every base event some voter's
> expansion placed there is a candidate, and every voter whose expansion placed one is one vote, split across
> the events it placed at that place by strength, as above. **The candidate with the largest share is output**
> (R40). The estimate is not read here: it is the reward of the actions of those frames, and an event is what
> followed, not what paid. Ties go to the older event. A place no voter reaches outputs nothing.
>
> **A bad plan is worn down, not cut.** A standing inference carries the estimate it read at age 0 (R45). Each
> frame it places what comes due at the frame ahead: actions in their dimensions, events in theirs, at their
> places, all at that estimate. Activations that fired after a reward moved the estimate place candidates at the
> moved number, or the walk's untried action at zero (R37). Where the largest estimate is one of those, the
> plan's action does not run; where the world reports an event other than the one the plan placed, that event
> did not stand. Each such placement is a failed neighbor to the plan's match (R49). The plan stands while what
> it found still pays for what it missed, and is dropped when it no longer does.
>
> **Each dimension is resolved on its own**: a plan that loses its action dimension still counts for the events
> it placed, and one that wins is not favored for them, so what an event dimension outputs is what the most
> voters placed, whichever inference placed it, and an inference that places events alone is resolved like any
> other.

> **D51 — Action policy.** Declared once for the machine, as the alphabet and the neighborhood policy are (D1,
> D50): what an action dimension outputs when every candidate in it carries a negative estimate (R47), **the
> least bad, or nothing**. The first is for a world that acts whether or not the machine does, where silence
> hands the frame to the world's own action and the machine learns from that (R35); the second is for a world
> in which silence is an action in its own right and costs nothing. The walk is the same under either (R37); the
> policy decides only what runs once the walk has nothing left to wire in a dimension.

> **R48 — Speaking falls through the cover.** A covered neuron supplies nothing while its coverer speaks (D10).
> Where the coverer has no inference for the frame ahead, what it covers speaks in its place, and what they
> cover in theirs, down to the base; nothing is output where nothing down to the base has anything. A newly
> allocated interneuron therefore starts with the voices of what it covers, the origin neuron among them, with
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
> every action dimension its expansion places an action in, one untried base action each; an event, or an
> interneuron that places no action, nothing. An interneuron whose plan hurt therefore tries, at that offset,
> one other action in each dimension the plan acted in, and nothing at any other offset. **What the walk finds,
> compression names.** The untried action is a base action under a situation an interneuron stands for, placed
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
> action and event alike (R47), and each fires weakly in the frame it was placed in (D40), the action to run and
> the event to come, and both are the machine's output. Where the report names that dimension at that place, the
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
| Alternation | [algorithm-xy.md](algorithm-xy.md) | A stream in which a pair of letters alternates: the two functions it becomes, one per origin, each a two-node cycle, one call over the whole stretch at any moment, and the general function the pairs generalize into. |
| Addition | [algorithm-addition.md](algorithm-addition.md) | A request with each column, a function per case that writes the answer, and a carry the world shows while it teaches and the machine infers afterwards. |
| Copy | [algorithm-copy.md](algorithm-copy.md) | Shown a digit and asked, the machine says it, by a parameter tying the digit shown to the digit said. |
| Equal | [algorithm-equal.md](algorithm-equal.md) | Shown four digits in one frame and asked, the machine says the third where the first two are the same and the fourth where they differ. |
| Hit | [algorithm-hit.md](algorithm-hit.md) | The machine hits whatever comes at it, wherever the two of them are, since a function is written in offsets and nothing else. |
| Loop | [algorithm-loop.md](algorithm-loop.md) | Shown a cue, the machine taps in each of the next three frames and then stops, with no counter: each tap's function names what stood a frame back. |
| MNIST | [algorithm-mnist.md](algorithm-mnist.md) | An image channel with two spatial dimensions and one base neuron, ink; functions over its relations, tiling the image at every level; a digit action taught by demonstration and resolved by a vote. |
| Stocks | [algorithm-stocks.md](algorithm-stocks.md) | A price channel per instrument with own and disown as its actions, a world that runs the right action wherever the machine has none, and a reward scoped to the channel and the frame. |
| Text | [algorithm-text.md](algorithm-text.md) | A stream of letters: pairs at the base, and stretches of any length, each letter's function naming the interneuron a frame before. |

# 10. Further reading

Each document is written in the terms of this specification and references its keys.

| Document | What it holds |
|---|---|
| [algorithm-remarks.md](algorithm-remarks.md) | The theorems, and the commentary on why the design is shaped this way, keyed by the same D and R numbers. |
| [algorithm-evaluation.md](algorithm-evaluation.md) | The falsifiable claim, the risks with what would be done about each, and the open questions. |
| [algorithm-implementation.md](algorithm-implementation.md) | The implementation plan: the per-neuron state and calls, the build stages, and what is cached. |
| [hippocampus.md](hippocampus.md) | An addendum: the second organ, running beside this machine over the same neurons, with rules keyed **H**. |
