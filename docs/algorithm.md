# Universal Compression with Actions and Rewards (UCAR)

UCAR is an algorithm that compresses what it observes and acts on the best reward estimate it holds. A machine
running it is a **UCAR machine**, and like a Turing machine it is defined by two alphabets: the **event alphabet**
it can observe and the **action alphabet** it can execute. Every symbol, base or learned, event or action, is a
**neuron**.

Its inputs are the events observed, the actions that ran and the rewards earned. Its output is what it places
at the frame ahead, written in the base alphabet: the actions and the events it infers.

Above the alphabets it builds a hierarchical dictionary of **patterns**. A pattern is a **function**: a set of
neurons, each at an offset from the others, that keep standing together. Where a function is general it leaves
a **variable**, a place whose content changes from one occurrence to the next and is filled by what stands there.
A function has a single interneuron, which stands for every occurrence of it.
Each frame, the machine compresses what arrived one level at a time. The neurons that fired return the relations
they see, the machine groups them into functions and variables and matches them against its dictionary, an election takes the
functions that pay, and each accepted function's interneuron fires at the next level.

What no pattern covers is the **apex**: the machine's state at a frame, the situation it is in, and the set that
infers the frame to come. Every neuron records what followed its activations along with their rewards.
Each frame, every activation still on the apex connects to what stands on the apex now, building a network of apex-to-apex connections.

The machine expands the inferences based on those connections and resolves them at the base, while matching them to observed reality.
One winner is resolved per dimension, and the machine outputs them: the actions and the events it infers.
The environment executes the actions, and reports what ran, what was observed and what it earned.

**D** is a definition and **R** a rule; together they are the machine.

---

# Part I — Structure

# 1. Objectives

There are two objectives: **compress what is observed**, and **act on the best reward estimate**.
The compression decides what the current situation *is*, and that's provided to the second objective as input.

## 1.1 Compression: classify the situation

**The machine compresses by naming.** A pattern is a set of symbols that keep occurring together, and a
symbol stands for them wherever they do. The run can be shortened by using these patterns. Patterns can name
base symbols or the interneurons of other patterns. This substitution is the whole mechanism for compression.

**The compression is lossy, in two places, and both are limits of the neuron.**

| Loss      | Description                                                                                                                                       | Reference |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| vision    | A neuron has limited vision: it sees its neighbors exactly only up to a declared distance, and beyond that only roughly, so a far neighbor is placed only to within the distance it rounds to. | D6        |
| memory    | A neuron has limited memory: it remembers only its most recent occurrences, and decides its patterns over those, so what it saw earlier no longer shapes its patterns. | D18       |

Both losses are what lets the machine generalize: a neuron that saw roughly and remembers recently names what
recurs, not the instances it happened to meet. Imagine the run written out under the patterns as they now
stand, and read the objective as: make that shorter.

## 1.2 Execution: the best estimate for the situation

A connection of a neuron contains the neuron that stood on the apex at an offset after it (D25).
It holds two numbers: strength and earnings.
Strength is the number of times that the connection was seen. Earnings are the total reward paid in those frames (R31).
Its estimate is its earnings over its strength, the average reward per time seen.
Every frame, each neuron standing on the apex infers from its connections what follows it (R36).
If an inference names a neuron above the base, it is expanded down to the base neurons it stands for (R28).
The base neurons placed at the frame ahead are then resolved into the machine's output.
In each action dimension the machine outputs the action placed there with the largest estimate (R47).
In each event dimension it outputs the event the most inferences place there, and reads no estimate.
That is the whole of it: **act on the best estimate you hold, for the situation you are actually in.**

**A reward pays frames and channels.** It arrives with a frame and may name the action channels it pays and the span of
frames it pays over (D35). A reward that names no channels pays every action channel. One that names no span
pays as far back as any neuron can still see, the longest reach in time the machine holds. The share of it
reaching each frame of the span falls linearly with that frame's distance from the one it arrived in, and lands
on the connections written at that frame (R33).

**Compression decides which neurons exist, and the neurons hold the connections.** A situation that recurs is
named by a pattern. The pattern's interneuron then stands on the apex in its place, and holds connections of its own.
Until then, the neurons that make up the situation each hold their own (R35).

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
> An **activation dimension** says where one instance of a symbol happened. It is an axis of the world: time
> or an axis of space. Every world has time. An input arrives with its position in each activation dimension
> it is laid out over.

For example, a pixel's color could be a neuron dimension, an event dimension of 256 buckets. A motor's
direction could be another, an action dimension of three buckets: forward, back and stop.

An image's `x` and `y` are activation dimensions, and so is time. A position in `x` or `y` is the pixel's
coordinate within the image. A position in time is the frame number, a counter that goes up by one with every
frame that arrives.

## 2.2 Channels

> **D3 — Channels.** A channel is defined by an event dimension, an action dimension, and spatial activation dimensions.
> Channels must have an event and/or an action dimension, and they own them exclusively.
> But the same activation dimensions may be shared by multiple channels.
> Each frame, the channel reports a symbol of each of its neuron dimensions at each position.

For example, a vision channel is laid out over `x`, `y`, and time. A stream of prices could use time alone.

Three color channels of one camera use the same `x` and `y`, so a red pixel and a green pixel could occupy the same
space in one camera. Two cameras with separate screens name separate axes.

The machine declares its channels once, and the set is fixed over the whole run. A channel cannot be created dynamically.

## 2.3 Frame

A frame arrives carrying what each event dimension observed, what each action dimension executed (D8), and any
rewards for actions already run (D35). Each dimension carries what the environment reports in it. An event
dimension carries what was observed, quantized to its buckets. An action dimension carries the action that ran.

An event is a base symbol of an event dimension (D1), and it fires when the environment reports that it was
observed (D8). An action is a base symbol of an action dimension, and it fires when the environment reports
that it ran.

At most one symbol is reported per dimension per position in a frame (D8): one event observed, one action
run. The machine outputs what it infers for the frame ahead, one symbol per dimension per position, an event
in each event dimension and an action in each action dimension (R47). The actions run in the world. The next
frame reports what was observed and what ran, and a symbol has happened when the report names it.

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

A reward is an input, not a symbol: nothing fires for it and no pattern names it. Zero is neutral, a positive
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

**The machine** carries what passes between the world and the neurons, and weaves the connections between
neurons. It runs two procedures once per frame, **recognition** and **inference**. Each frame
is one wave through the neurons. It rises through the frame as recognition, runs forward into the frame ahead as
inference, and comes down to the base as expansion, where one winner per dimension is output.

**It has two capacities, both the hardware's.** Its **precision** `P` is the range of one number: every id, frame,
position and offset it writes is a number below `P`. Its **neuron capacity** `N` is the neurons it holds, fewer
than `P`, since each has an id.

## 3.1 Neurons

> **D2 — Neuron.** A neuron is a symbol, and a type: its activations are its instances (D8). It is identified
> by its id, fixed at allocation. The machine keeps the order neurons were allocated in, so an older neuron is
> one allocated earlier, and a decommissioned neuron allocated again is the youngest. It is a processor with a
> memory of its own, and that memory holds a history (D18), a relation index over it (D47) and connections (D25). It
> is a base neuron (D52) or an interneuron (D53).

Older neurons were allocated earlier. That's how ties are broken.

A neuron records what it sees and returns it when the machine calls it: `activate`, `connect` and `infer` (§4.1).

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
> is observed, or an action neuron when its action runs.
>
> **A neuron may fire many times in one frame**, and those activations are instances of one type.
>
> **At the base, at most one neuron fires per dimension per position in a frame.** The input reports one symbol
> per channel at each point of its layout, and a dimension with nothing to report there is silent. That bound
> is a property of the input, not a rule the machine imposes.
>
> **Above the base, multiple neurons can fire for the same dimension and position.** Several functions may be
> accepted over overlapping ground (D28), so several interneurons may stand at one coordinate (D53).

> **D9 — Activation window.** An activation at frame `f` stays open through `f + reach_t`, its reach
> in time (D4). Two things reach it over that span, one frame at a time, by the connect call after the
> frame's last level (§7.1):
>
> | Input           | Description                                                                                                 | References |
> |-----------------|-------------------------------------------------------------------------------------------------------------|------------|
> | the apex       | What stands on the apex of that frame within reach (R27, D25), while the activation is uncovered; in the frame it is covered, the neuron that covered it (D10). | D25, R31   |
> | the rewards     | For the actions of that frame, and of earlier frames the reward spans, each at the distance its frame names. | R29, R33   |
>
> The machine holds the activation open, not the neuron. There is one open activation per neuron, age and
> position, and it holds four things: the activation, its coordinate, its age, and the age it was covered at,
> if it has been.
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since the frame of its coordinate, `0` through `reach_t`; a new activation is the one at age 0.
>
> **Covered at** is the age at which an accepted function first covered the activation, none until one does.
> Coverage is acquired late (R27), and a later call may take the activation from its coverer, covered still
> (R23), so this one number says for every frame of the window whether the activation stood on the apex there:
> it did at every age before it, and at none from it on.
>
> **An open activation carries no commitment beyond that one number.** It holds its neighborhood, the age it
> was covered at, and nothing else, because everything else it was going to decide was decided at age 0 (R1).
> It infers once, at age 0, and the machine keeps what it inferred (R45); what it does for the rest of its life
> is connect to what ran, in the `connect` call.

There is no rest value. A dimension where nothing happens supplies no symbol, and silence is what the decoder
assumes for anything the file does not state.

> **D40 — Strength.** An activation is **strong** when the environment reported it, and **weak** when an
> inference placed it (§8.5). That holds for events and actions alike: an event and an action the machine
> infers are both weak, and the environment's report replaces both. The neuron is the same either way: a
> weak activation is an activation of the neuron the environment would have reported.
>
> **A weak activation is an input of its frame like a strong one.** It is in the neighborhood of everything that
> fires within reach of it, at whatever offset, so it enters histories, and patterns name it and price it like any
> neighbor; its neuron is called with it, it may be covered, and it may stand on the apex and vote once (R46), with
> the connections its neuron's strong activations strengthened. **It differs in one thing: it lasts its one frame.** It
> is not open afterward: it writes no connection, nothing connects to it, and no reward reaches it (R31). **The
> report replaces it**: where a strong activation of the same neuron stands at the same coordinate, the weak
> one is dropped. Where nothing reports, the inference peeks through: an inferred action nothing reports
> as run, and an inferred event nothing reports as seen, are what the machine had in that frame, and a dimension
> the environment does not have holds nothing else.
>
> **An interneuron is as strong as what it covers.** An accepted function's interneuron is strong when any activation it covers
> or binds is strong, and weak when every one of them is: a situation made entirely of inferred inputs is
> inferred.

> **D37 — Call.** One activation of a call neuron (D48), together with its **arguments**: the activations bound
> in its function's variables on that occasion (D10, D38). It is made in one of two ways and is the same thing either
> way. **Recognized**: the election accepts the function, the call neuron fires where what it covers stands
> (D53), as strong as what it covers (D40), and the arguments are what stood in the variables. **Inferred**:
> an expansion places the call neuron at its completion, the voter's coordinate plus the connection's offset in
> time and in every other activation dimension (R28, R36), and it fires weakly there, for the report to make
> strong what it placed (R30); its arguments are bound from what has happened where a variable's places have
> happened, and placed where they have not (D44, R28). A call carries nothing of its own: what it is stands in
> its function, and what its arguments are stands on the apex beside it. Recognition builds calls out of what
> fired, and expansion unwinds a call back to the base symbols it stands for.

**A pattern is matched or inferred, and both kinds of member take part in both.** Matched, a pattern says that
its events stood and its actions ran. Inferred, it is expanded (R28): its actions are placed as the machine's
output and its events beside them, both weakly, and the report makes strong what ran and what was seen. What
follows any activation is held on its neuron as connections (D25), one kind, and an inference is a connection
read.

## 3.4 Spacetime

> **D4 — Reach.** How far an activation sees, in each activation dimension (D1). A base activation sees `R`, the
> radius of that dimension. An interneuron's activation sees `R` beyond the farthest base symbol it stands for
> (D36), measured from where it stands (D53). In space it stands at the centroid of what it covers and binds, so
> it sees about half of what it stands for each way, and `R` more. In time it stands at the frame it fires in, the
> latest, so it sees back over all of what it stands for, and `R` more. Where what it stands for comes back to an
> interneuron already expanded, it stops there (D36), and so does its reach.
>
> In space the reach runs either way; in time it runs back, and the same number runs forward as the activation's
> window (D9). An activation sees every distance up to its reach, exact within `R` and rounded to one significant
> digit beyond when written (D6). `reach_t` is the reach in time, the activation's
> window.

> **D6 — Offsets.** The offset between two activations is the difference of their coordinates, one component per
> activation dimension they share (D1, D2). A neighborhood keeps it exact (D18). Written, in a pattern (D15), in a
> connection (D25) or where an expansion places a neighbor (R28), each component is **rounded down to one
> significant digit in base `R + 1`**, `R` being the radius of that dimension (D1). Zero stays zero, and any
> other component `x` becomes `sign(x) · floor(|x| / (R + 1)^k) · (R + 1)^k`, with `k = floor(log_(R + 1) |x|)`.
> The single digits of base `R + 1` are `0` through `R`, so every distance up to `R` is exact, and beyond it the
> distance is kept to one digit. At `R = 1`, 5 and 7 become 4 and −13 becomes −8; at `R = 10`, 7 stays 7, 15
> becomes 11, 95 becomes 88 and −250 becomes −242. The reachable offsets are `0`, every distance up to `±R`, then
> `±(R + 1), ±2(R + 1), … ±R(R + 1), ±(R + 1)², ±2(R + 1)², …` up to the largest not above the reach, and the last
> group is cut off at the reach: at `R = 1` a reach of 38 names 1, 2, 4, 8, 16 and 32, the last standing for 32
> through 38; at `R = 10` a reach of 250 names 1 through 10, 11 through 110 by elevens, 121 and 242, the last
> standing for 242 through 250. A reach of `r` holds `offsets(r) = R · k + floor(r / (R + 1)^k)` distances from
> 1 to `r`, with `k = floor(log_(R + 1) r)`, so `offsets(r) + 1` offsets in time, zero and the past, and
> `2 · offsets(r) + 1` in any other dimension.

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

Events and actions are compressed in the same patterns: above the base a pattern may name what was seen and
what was done together.

> **D50 — Neighborhood policy.** Declared once for the machine, as the alphabet is (D1). For it a neuron is of
> one of three kinds: an **event** or an **action**, a base neuron of that kind of dimension (D1), or an
> **interneuron**, whatever its function names (D2). The policy is two tables of booleans,
> the kind of the holder by the kind of the target, nine entries each:
>
> | Table  | An entry on says                                                          | And so decides            |
> |--------|---------------------------------------------------------------------------|---------------------------|
> | past   | a target of that kind may stand in a holder of that kind's neighborhood (D5) | what patterns are built from |
> | future | a holder of that kind may hold a connection to a target of that kind (D25)   | what inference reads      |
>
> With every entry on, anything may be a neighbor of anything and anything may connect to anything. An entry
> off removes that kind of target from that kind of holder's neighborhoods or connections, and nothing else:
> what stands on the apex infers in every setting (D49), and the policy restricts only what it reaches.

> **D26 — Neighbor.** **A neighbor is a neuron at an offset**: the neuron of an adjacent activation (D5), at its
> offset (D6). At temporal offset 0 a neighbor co-occurs and at negative offsets it led here; **both are the same
> kind of thing**, and a spatial component is one more of the same.
>
> **A coarse offset in a pattern may be held by more than one neighbor**, since it spans a range and several
> activations of one dimension can fall inside it (D6). A relation's offset is written rounded (D34), every
> neighbor whose exact offset from the relation's other node falls in it holds there, and `|p|` counts every
> node named. Above the base a `(dimension, position)` may itself carry several activations (D8), which this
> handles the same way and needs no rule of its own.
>
> **A neuron can be its own neighbor.** Two activations of one type at different positions each name the other
> at a nonzero spatial offset. Offset zero in every component is where the activation itself sits, the center
> of the box, and an activation is not its own neighbor there. Above the base several interneurons promoted at one
> coordinate (D8) are each other's neighbors at offset zero.

> **D7 — Neighborhood.** The set of neighbors (D26) one activation observes is its **neighborhood**, written `O`
> for what it observed. In a text stream, with time its one dimension, `O = { (i, −1) }` could be the
> neighborhood of a neuron `s` in a stream reading `p a r i s`: at the base the reach is the radius, here 1
> (D4), so `i` is its one neighbor and `p a r` are out of reach. In an image stream, with time, x and y,
> `O = { (k, 0, −1, 0), (k, 0, +1, 0), (m, 0, 0, −1) }` is a neighborhood of three.
>
> **A neighborhood is the frame around the fired neuron**, since adjacency admits nothing later (D5), and every
> structural decision is made on it. The activation itself is not in it: its own instance is what a pattern
> covering it replaces, and is counted there (D22).
>
> **An activation bound in a variable is a neighbor like any other** (D10): the neuron that stood, at its own
> coordinate. Which variable it stood in is the call's business, not the neighbor's.

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

> **D34 — Relation.** Two neurons and the offset between them in one activation dimension (D1, D6). A node pinned
> in several dimensions is in several relations, one per dimension. In a neuron's relation index, one end is the
> neuron's own activation and the other a neighbor. In a function, both ends are its nodes, and the relation is
> written from the older of the two (D36). In a function the offset is written rounded, as every written offset is,
> and the relation holds where its ends stand at any exact offset in the written offset's group. Relations
> compose: given the offset from `a` to `b` and from `b` to `c` in one dimension, the offset from `a` to `c`
> follows, so a set of relations that connects its nodes in a dimension fixes every other offset among them there.

> **D11 — Identity.** A neuron's identity is its type alone, a base symbol or an interneuron's id (D2), so
> nothing about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: both carry the same relative neighborhood, so the same functions cover them and one function
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

## 3.5 Functions

> **D15 — Function.** A set of relations (D34). Each relation names a **node** at each end (D16) and the offset
> between them in one dimension. A function holds where its relations hold: each one lands on activations that
> stand as it says. It has at least one fixed node in its latest frame, since the evidence for it is the history
> of a neuron that sees all of it, and reach in time runs back (D30). It
> is written once, in the dictionary (D36), and has one interneuron, its call neuron (D48).
>
> **A fit places relations, not a fixed set of activations.** Each time a function fits, every relation lands on
> the activations that satisfy it. Two relations that name the same neuron may land on one activation, or on two.
> So one function may fix one activation's `x` and `y` in one row, and the `x` of one activation and the `y` of
> another in the next. It is the same function either way, and each fit is priced as it falls (D22): an activation
> the relations land on is covered in the dimensions they fix, and its other positions are written.
>
> **Relations, not positions, are stored.** A neuron returns relations from itself to its neighbors (D47). The
> machine stores a function as its relations, each written from the older of its two nodes, so a group a neuron
> returned is rewritten from each relation's older end before it is stored. To match a stored function against
> what a neuron returned, the machine puts that neuron at its node and composes: a stored relation between two of
> its neighbors is checked against that neuron's relations to each of them.
>
> **Where the call stands follows from what it lands on.** The call fires at the rounded centroid of the
> activations it covers and binds (D53). In each dimension, the relations of a fit fix the offsets among the
> activations they land on there, and the positions they leave open are written from where the call stands. So
> every activation's offset from the call follows from the relations and the call's record. Expansion runs from the
> call's position out to each activation (R28), and the decoder recovers the centroid the same way.
>
> **A function is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a function is what the machine writes. A neighborhood is a fact, a function a
> claim. What followed is in neither, and is held as connections (D25).
>
> **Its entry in the dictionary** is its relations and the neurons of its fixed nodes, each at what its kind
> costs (D13). A neuron named by several relations is named once.

> **D16 — Node.** What a relation names at one of its ends: a **fixed node**, a particular neuron (D2), or a variable
> (D38). A node is what a function names, never an activation. Within one fit, each relation lands its nodes on
> activations, and nodes of different relations that name the same neuron may land on the same activation or on
> different ones (D15).

**A function's body is its relations, and the nodes they name.** Some nodes are fixed, a particular neuron at
a place; some are **variables**, places where the function is general. A variable is where the function fits more
activations than any one neuron there would let it: it names the place and leaves open what stands in it. What
stands in a variable on one occasion is the variable's **value**, and the values of a call's variables are its
arguments (D37). **Classes are a function's local variables and parameters are its arguments**: a class is a place
whose content is decided where the function runs, among what has stood there before, and a parameter is one value
handed in and used at every one of its places. A variable exists only inside a function, is never placed alone, and
promotes no interneuron: what stands in it stays on the apex, bound (D10).

> **D38 — Variables.** A variable is a node that leaves open which neuron stands at its place. The machine makes
> variables, never a neuron (R52): a variable says what is general across several functions, which no one
> activation shows. It is of one of two types:
>
> | Variable | In the dictionary, once | In the call, each time | What it covers |
> |---|---|---|---|
> | a class (D41) | which class, a set of neurons, its members, and the node's relations | the member that stood, one number | the member's position in each dimension it is pinned in; the member stands bound (D10) |
> | a parameter (D44) | its nodes' relations, all reading one neuron | the neuron that stood, one number for all its nodes | the position at each node, and the neuron at every node after the first; the neuron stands bound at each (D10) |
>
> **The two types are closed in opposite ways.** A class is closed on its members and open on where: a
> neuron that is not a member does not fit, and any function may use the class at any place. A
> parameter is open on its members and closed on where: any neuron fits, seen before or not, and only
> where it stands at every node the variable ties.
>
> **A function holds nothing but its nodes and relations**: no branch, no loop. What branches is which function
> fires; what loops is the frame; what a variable held is on the apex, bound beside the call neuron. On every
> occasion it is priced as its call's record (D12, D22).

> **D41 — Class.** A set of neurons the dictionary holds, its **members**, and nothing else: **one of these stands
> here**. In the dictionary it costs one number per member (D13). A function uses it as a variable at a node
> (D38); several functions may use one class, and a function may use one class at several nodes, a member bound at
> each. Members stand in place of one another: two neurons that have stood together where the class is used are
> never both members.
>
> **Its value**, at a node, is the member that stood there, which stays on the apex, bound (D10). Where no
> member stands at a node the class is used, that node's relations did not come true (D22), and the variable does
> not hold there.
>
> **A class is not a neuron.** It is made by generalizing functions that differ at that node (R52). A neuron joins
> it when it stands at the node in a row where the rest of a function using the class holds, and what it covers
> there pays for its one number in the dictionary; it leaves when it no longer does. So what is learned about a
> member at one place holds at every other. A class is deleted when no function uses it, or when it is left with
> one member, which is an ordinary fixed node.

> **D44 — Parameter.** A variable of one function that ties two or more of its nodes together: **the same one
> stands at all of these nodes**, whichever it is. Beside it the function keeps its **members**, the neurons that
> have stood at all of its nodes within the history. The list is a record and not a condition. It says nothing
> about what may stand, and it prices nothing (D13).
>
> **Its value** is the neuron that stood at its nodes. It stays on the apex, bound, at each of them, written as
> itself (D10). Where the nodes disagree, the value is the neuron at the earliest of them, the smaller position on
> a tie, and every other node that holds something else did not come true (D22).
>
> **It is open on its members and closed on where.** Any neuron fits, seen before or not, and only where it
> stands at every node: a thing that comes for the first time is the parameter's value the first time, and is a
> member from then on. It is made by generalizing functions in which some nodes hold the same neuron within each
> function and a different one across them (R52).
>
> **Bound from what has happened, placed at what has not.** When a function is expanded with no argument for a
> parameter (R28), and the variable's nodes fall partly in frames that have happened and partly in frames to
> come, it takes its value from what stands at the first and places it at the second. That is how a value is
> carried forward in time, and it is the whole of copying.

## 3.6 Interneurons

> **D53 — Interneuron.** The neuron the machine allocates for a function, its **call neuron** (D48). It is allocated the first time
> the function is accepted and a neuron is free (R16). It has an id and no dimension and no channel. **It fires
> where what it covers stands**: in time, at the frame it is accepted in, age 0; in each spatial dimension, at the
> centroid of the positions of the activations it covers and binds that its function pins in that dimension,
> rounded, half toward the smaller position.

An interneuron is not an event or an action, though what it stands for may be either. Its position comes from
what it covers, not from any one neuron that returned it. So a shape's call stands at the shape's center, and a
call over several shapes at the center of theirs. The centroid is not stored: it is computed where the call fires,
and the decoder recovers it the same way (D15).

A neuron is never made or destroyed. It is **allocated** when an interneuron is needed, and **decommissioned** when its
last function leaves the dictionary, to be allocated again. When none of the machine's `N` neurons is free, no
interneuron is granted (R16, R24).

> **D36 — Interneuron index.** The machine's dictionary: every function (D15), each with its interneuron once it
> has one. It sits beside the index by id (D2) and the one by base symbol (D52), and is reached three ways:
>
> | By | Reaches |
> |---|---|
> | a function | its interneuron, and back, for expansion (R28) |
> | a relation (D34), or one of its neurons and its offset with the other end open | every function that writes it, or writes any relation with that neuron at that end and that offset |
> | what an interneuron stands for | the interneuron, for reuse when a stretch comes round (R43) |
>
> **The same relations are the same function.** A function is its set of relations, each written from the older of
> its two nodes, so the same neurons at the same distances are one function whichever neuron returned them. Each
> function has one interneuron. An interneuron may hold more than one function where a stretch comes round (R43):
> it fires for each, and an activation of it is expanded through the one that fired it.
>
> **A function names other interneurons by id**, and through them it may come back to its own: a function whose node
> is the interneuron that fired a frame before, whose function names the one before that, and so on, can reach its
> own interneuron again. **What an interneuron stands for** is its expansion (R28): its function's nodes with the
> variables left open, each node in turn for what it stands for, down to base symbols. Where that comes back to an
> interneuron already expanded, it stops there.
>
> **The index is searched by what an interneuron would stand for** (R43): a function's expansion, read back from
> where its call would stand. The search finds an interneuron when three things hold: reading that expansion back,
> it meets the interneuron within the interneuron's own length; over exactly that length the two expansions are the
> same; and the meeting is not where the call would stand. That is how something that comes round again finds the
> interneuron it had the first time. A search by relations alone would not find it: a function whose node is the
> interneuron that fired a frame before names a different interneuron every frame, while what it stands for, over
> that length, is the same.

**A function and its interneuron are one entry.** The machine holds both in its dictionary. No neuron holds a
function or points at an interneuron: a neuron records what it sees and returns it (§5), and the machine groups, matches and keeps
(§6).

> **D48 — Call neuron.** A function's interneuron: the neuron the machine allocates for a function the first time
> it is accepted (R16). It fires when its function is accepted, where what it covers stands (D53), once, covering the
> function's fixed nodes (§6.4); what stands in the function's variables is bound beside it, on the apex (D10). It
> says which function ran. Its connections are one set for every value its variables have taken, so a function
> above learns about it regardless of its arguments by naming the call neuron alone, and about one argument by
> naming the call neuron and the activation bound in that variable. It holds a history, a relation index and
> connections of its own, like any neuron, and is expanded through the function that fired it (R28).

## 3.7 Apex

> **D49 — Apex.** The activations of a frame that no accepted function covers, base neurons and interneurons alike:
> a base neuron no pattern found worth naming stands in it beside an interneuron many patterns deep. It is the machine's whole state at
> that frame, and it has four names for the four things it is. It is the **frontier**, looking back: everything
> underneath it is recovered by expanding it, so it is what the file counts. It is the **situation**: what
> every neuron sees as its neighborhood (D5), what connects to what ran, and what votes. It is the
> **wavefront** of recognition: where the frame's levels stopped (R26), and from where the wave runs
> forward, each activation of it reading its own connections ahead, which it can do only because it is also
> the frontier, the neighborhood every neuron was built on. And it is the apex, the top of the hierarchy as it
> stands. The four are one set, and coverage takes a neuron out of it in every role at once.
>
> **Partway through the frame it is the board.** While the levels run, the apex as it stands at a level is the
> board the next level is matched and elected on (§6). The apex of the frame is the board once the last level has
> run.

> **D10 — Inhibition.** Coverage does three things to an activation, all from the frame it is covered in. It
> leaves the file: the file counts its coverer's call, not it. It is not heard: resolution reads its coverer's
> standing inferences, not its own (R46). And it stops connecting: from the frame after it is covered it writes
> nothing about what follows.
>
> **The last connection.** In the frame it is covered, the activation connects once more, to the neuron that
> covered it, the function's interneuron. The connection is at the
> offset from the activation's coordinate to its coverer's (D6), in time and in every other activation
> dimension, as every connection is (D25), and it takes that frame's reward (R31). Then the activation is
> silent. An activation standing at its coverer's own coordinate (D53) writes nothing, as one covered at age 0
> does. Where a later call takes the activation from its coverer (R23), it connects once more the same way, to
> the new coverer. Without this connection the activation could never learn the neuron built over it:
> the interneuron did not exist in the frames it was uncovered, and it is inhibited from the frame the interneuron first
> fires. So a situation beside which a pattern has named what followed it learns that one neuron, the interneuron, and
> from then on infers it; the interneuron learns what follows.
>
> **What stands before coverage stands.** A neuron is covered only after it has fired (R27). Coverage acquired at
> a later age revokes nothing before it: the connections the activation strengthened while uncovered stand, and
> a reward for one of those frames still reaches them (R33). An activation covered already at age 0 writes
> nothing but its last connection, to its coverer and again to a call that takes it (R23): a connection is to
> what comes after (D25), and by then it is inhibited.
>
> **One exception, on being heard alone.** Every activation infers at its age 0, covered or not, and the machine
> keeps what it inferred (R45). A covered activation's inferences are heard only where its coverer has nothing
> for the frame ahead (R48), as a newly allocated interneuron has nothing until something has run under it. The covered
> activation still writes nothing and connects to nothing; only its coverer learns what ran. That is the whole
> of inhibition in the design.
>
> **Bound is half covered.** An activation standing in a variable of an accepted function (D38) is **bound**. The call
> covers its positions: the function's relation says where it stands, so they are not written again. Its neuron
> is written inside the call's record, one number (D12, D13). It stays on the apex, it is a neighbor, it is
> heard, and it connects to what follows. A call hides what is the same on every occasion, its fixed nodes, and
> leaves visible what varies. A parameter binds each of its places the same way, and writes its value once.
>
> **Above its level, a bound activation is a neighbor.** Its neuron is called once, at age 0 (R1). At the levels
> above, the interneurons that fire see it as a neighbor, and their relations to it are the ones it takes part in.
> In a dimension its binding left uncovered, its position is still written bare, so a function above can cover it
> there. In a dimension its binding covered, its position costs nothing, so a relation to it there holds a node and
> covers nothing (D22). Its neuron is still written, as the value in the call that bound it, so a function above
> that names that call and this neuron fixes the value, and saves it (D24).

## 3.8 File

> **D12 — File.** Two parts, both spanning the whole run, both lists of numbers (D13). **The dictionary**: every
> function of the interneuron index (D36), written as its relations, and the classes they use (D38). **The transcript**:
> the apex of every frame (D49), each activation as its **record**:
>
> | Activation | Its record |
> |---|---|
> | a bare neuron | the neuron, the frame, and a position per spatial dimension: `d + 2` numbers |
> | a call | the interneuron, the frame, and a position per spatial dimension, `d + 2` numbers; then each relation of its function that did not come true; then the value of each variable that did, one neuron each |
> | an activation bound in a variable | its neuron, written inside the call's record; and a position number for each dimension its node is not pinned in |
>
> The file names neurons and relations in one range of numbers, so a number says which it is. A record is read
> in order: a neuron begins it, the frame and the positions follow, and for a call the relations that did not
> come true and then the values, as many as the function says.

**A call writes its body once and its differences every time.** The function's relations are in the dictionary.
Each call writes where it stood, which of its relations did not come true, and what stood in its variables.
What the relations fix is not written again: the neurons of its fixed nodes, and the positions of every node it
names, its variables' included. What a variable leaves open is written: the value's neuron, and any position
number the relation does not fix.

**An error is a relation that did not come true.** It is written with the call, one number. Whatever stood where
the relation said is on the apex, uncovered, and is written as itself. So a wrong neuron costs the relation and
a record, and nothing there costs the relation alone.

**Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
file exactly as an observed event does, compressed by the same patterns (D5).

**It holds nothing about the future.** A neuron's connections — what followed its activations, event and
action alike, and what the actions earned — are in neither part (D25), and neither is an inference that did
not run (R32).

**It holds nothing about the search either.** Populations, estimates and margins are the machine's and
never the file's, because expanding an apex neuron needs the patterns and nothing else.

**It holds the values.** What stands in a variable is written inside the call that bound it, as its neuron
(D13), and expansion puts it back at its place (R28).

## 3.9 Prices

> **D13 — Number.** Every cost in the design is a count of numbers in the file, and nothing is ever written: the
> file is the yardstick (§1.1). A number is one field at the machine's precision `P` (§3), and every number
> costs the same, `1`. A neuron, a relation, a class, a frame, a position and an offset in one dimension are each one
> number. With `d` spatial dimensions in the activation's layout:
>
> | What | Numbers |
> |---|---|
> | the record of a bare activation | `d + 2` |
> | the record of a call | `d + 2`, plus one per relation that did not come true, plus one per variable that did |
> | a value bound in a variable | `1`, inside the call's record, plus one per dimension its node is not pinned in |
> | a fixed node, in the dictionary | `1`, its neuron |
> | a class at a node, in the dictionary | `1`, which class |
> | a class, in the dictionary | one per member |
> | a relation, in the dictionary | `1`, its offset; its two nodes are named within the function |

The precision `P` and the neuron capacity `N` are the hardware's (§3), not prices. Neither enters a cost, since
every number costs `1` at whatever precision. They set what can exist: how many neurons there are to allocate
(R16), how far a frame is counted, and how fine a position can be written.

**Every saving is a comparison of two files.** Each savings and each margin in the design is the cost of writing
the same thing two ways, A less B, and nothing else: a row written bare against written through a function
(D22), a neuron's history written by the dictionary without a function against with it (D30). Nothing is counted that is
not written in A and left out of B.

**An activation costs what its record takes to state** (D12), and it is the same wherever the activation is
counted. The machine keeps the cost with the activation, and a saved neighborhood keeps it with each neighbor
(D18).

**Covering an activation saves exactly what it cost.** A fixed node that came true saves its neuron, and one
number for each dimension it is pinned in. A variable that came true saves one number for each dimension it is
pinned in, and its neuron is written inside the call. A parameter writes its value once, so every node it ties
after the first saves its neuron as well.

**Where an activation stands is written.** The transcript lists what stood, and silence is what it does not
list, so each record says where it stood: the frame, and a position per spatial dimension. A frame is counted within `P` and
comes round. The window is short against that, so a record's frame is read against the frames around it, and
nothing is ambiguous. How many things
stand at one place, or follow one instance, needs no number: each is a record of its own. **Nothing about
frequency enters**: a number costs the same however often it is used.

## 3.10 Savings

> **D54 — Fit.** A function fits where writing what it names through it costs less than writing it bare: its
> savings there are positive (D22). It is read against a row of a neuron's history, to price the function (D30),
> and against what an activation returned, to find it (R53). Each node is read as it stands: every neighbor is the neuron at its
> offset, and an interneuron on the apex is a neuron as itself. Nothing is reserved: two functions may fit on the same
> neighbors. The rows of a neuron's history a function fits are its **uses** there.
>
> **Coming true.** Measured where it is read, each relation of a function comes true or does not:
>
> | Relation, at its far node | Comes true where |
> |---|---|
> | a fixed node | its neuron stands at the offset, or an interneuron standing on the apex stands for it there (D22) |
> | a class | a member stands at the offset; it takes members only (D41) |
> | a parameter | the parameter's value stands at the offset; it takes any neuron, and one that was not a member is one from then on (D44) |
>
> A relation that does not come true is written with the call (D12). Where its node would have stood follows from
> the relations that did.

> **D22 — Savings.** What one function saves where it is read: the cost of writing bare what it names and came
> true (A), less the cost of writing it through the function (B).
>
> | | Numbers |
> |---|---|
> | A | for each activation the fit's relations land on, fixed or bound in a variable, its record: its neuron and its position in every dimension, `d + 2`; for the activations a lower call covers, that call's record, once, where the fit lands on every one of them, and nothing where it lands on some (R23) |
> | B | the call's record: `d + 2`; one per value; one per position number of such an activation in a dimension no relation that lands on it fixes; and one per relation that did not come true |
> | savings | A less B |
>
> A node that did not come true is in neither: whatever stands there is written bare either way. What a neuron is
> worth is always what it cost (D13), read from the saved neighborhood.
>
> **A call is taken back whole or not at all.** Where a function names neurons an interneuron standing on the
> apex stands for (D47), and it lands on everything that interneuron's call covers, it covers them: the call
> leaves the file, the interneuron's activation leaves the apex, and the function is credited the call's record,
> since that is what the file held for them (R23). Where it lands on some of them, the nodes hold, the function
> is credited nothing for them, and the interneuron is not covered: the file holds its record either way. So a
> function saves what it takes out of the file, bare records and whole calls.

**What a function's call stands in for is what it covers**; that is A. What it costs is its record; that is B. A
neuron the function does not name is in neither: it costs what it cost whether the function exists or not.

**This is the only valuation in the design**, and it is read over three different sets, so the three numbers
differ, and are meant to:

| Read by                               | Over                  | Against                                  | Settles                            |
|---------------------------------------|-----------------------|------------------------------------------|------------------------------------|
| the machine, pricing a function (D30) | the dictionary's functions | each row of a neuron's history      | which functions the dictionary keeps |
| the machine, in the election (R22, R24) | the matched functions | the board                              | what is credited, and so the file  |
| the machine, in the match (R49)       | one inferred function | the frames it reaches back into          | whether the inference is proposed  |

One rule decides what a set of functions writes for a set of activations. The election runs it over the board,
and the machine runs it over a row of a neuron's history to price a function (D30). It is stated here and cited
from both.

> **D28 — Greedy cover.** The operation that covers a set of activations with **functions**, each read against
> one of those activations, and pays each covered neuron to exactly one function. It repeats:
>
> 1. **Measure** every function not yet taken against the still-uncovered neurons: its coverage is what the
>    still-uncovered neurons it covers cost, and the record of each call it takes back whole (R23), and its
>    price is its call's record there (D22).
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It is
>    paid for the neurons it was measured on. Those it covers leave the uncovered set. Those it binds stay in it
>    at their neuron's cost, since only their positions are covered, so a later round may cover them (D10). The
>    round repeats from 1.
>
> **What it returns is what was written and by whom**: the functions taken, what each was paid for, and what no
> round took, which stands as itself. `price` is fixed by what fired and cannot change between rounds;
> `coverage` only falls.

**The two callers differ in what they cover and in how they break ties, in nothing else.** Each states its
tie-break where it runs (R24, D30).

| caller                     | functions | to cover                      |
|----------------------------|---|-------------------------------|
| machine - margin (D30) | the dictionary's functions that fit the row (D54) | one row of a neuron's history, whole; priced and kept nowhere |
| machine - election (R24)  | the functions matched at this level, each placed (D31) | the board (§6.2) |

## 3.11 File length

The file's length `L` is two counts of numbers (D13): the dictionary, every relation of every function and
every member of every class; and the transcript, every record on the frontier. There is one reading of it. A
call is its record, what the call fired stands on the frontier at exactly that, and what it bound stands inside
it. A neuron above counts an interneuron at what it cost below, so nothing is saved by naming it again and nothing is
gained by counting it twice. The election takes records off the frontier and puts a call's record in their
place (R22). A neuron's margin does the same over its own history (D30). Nothing computes `L`: every quantity
the design uses is a **difference**, which is finite however long the run is.

## 3.12 Parameters

> **R4 — `H` and `R` are the only free parameters.** Two things are the designer's to pick: `H`, the depth of
> every neuron's history (D18), and `R`, one radius per activation dimension (D1), how far a base neuron sees in
> it. Nothing else is. The alphabet (D1) states the problem and is not a knob. The reach (D4),
> adjacency (D5) and the offsets (D6) are derived from `R`, not declared. The precision `P` and the neuron
> capacity `N` are the hardware's (§3), and no price reads them (D13). The neighborhood policy (D50) and the action policy (D51) are
> declarations like the alphabet: each states the kind of world the machine is in, and neither is a knob. **No
> rule introduces a constant, a threshold, a window or a cap of its own.**

The machine gives each activation dimension (D1) a **radius** `R`, a count of positions, `1` or more: how far a
base neuron sees in it (D4), and how every distance in it is rounded (D6). The radius is the resolution of the
layout, as the bucket count is the resolution of what is reported on it. For example, an image whose pixel is
to be told from the next has `R = 1` in `x` and `y`. A stream in which the fourth frame back matters as exactly
as the first has `R = 4` in time. Where this document writes a distance with no radius given, `R = 1`.

# 4. Neuron

What this section defines belongs to one neuron: what it holds, and what it answers. A neuron records, counts
and returns. It decides nothing about what is written.

## 4.1 Interface

A neuron is a symbol, and a type (D2). It holds a history of its activations, a relation index over it, and its
connections (D25).

**The machine reaches a neuron through six calls, and nothing else writes into one.** Each is specified where
it is used.

| Call              | Description                                                                                                       | References |
|-------------------|-------------------------------------------------------------------------------------------------------------------|------------|
| allocate neuron   | The machine allocates it the first time a function is accepted and a neuron is free, holding nothing: the function's call neuron. | R16 |
| activate          | At its level: record the frame's activations, and return the relations from each.                                 | §5, R20    |
| count             | Asked about a set of relations from itself, return the rows of its history where every one of them holds, or every one but `k`. | D47 |
| forget neuron     | The machine removes a decommissioned neuron from every row and relation of the history that names it.              | R38        |
| connect           | After the frame's last level, reaching every open activation at age 1 and beyond: it delivers what stood on the apex this frame and any reward, and collects nothing. | §7.1 |
| infer             | After the frame's last level, reaching every activation at age 0: it delivers nothing and collects what the activation infers. | §7.2 |

In `activate` a neuron records and returns, in the frame it fires. In recognition the machine groups
what each neuron returns, finds it in the index, generalizes, elects, allocates and prunes, and asks a neuron to
count. In `connect` and `infer` a neuron learns what
followed and infers what comes next. In selection the machine chooses among those inferences one winner per
dimension.

> **R1 — One decision point: the frame it fires.** A neuron is called once per level it fires at, for every
> activation of that level together, each at age 0: it refreshes its history and its relation index, and returns
> the relations from each activation (R20). Between those calls the machine may ask it to count, which changes
> nothing in it.

## 4.2 History

> **D18 — History.** A neuron's record of its own activations: the last `H` of them, oldest first, each at its
> coordinate, holding the neighborhood it observed in the frame it fired (D7), each neighbor at its exact
> offset (D6) and with what it cost, and the activation's own cost beside them, since a function
> covers the activation too (D13, D22). It is the evidence every structural decision is read over, by the
> machine, and it holds nothing else: the neuron's connections are not in the history and `H` does not bound them
> (R31).
>
> **`H` is declared once for the machine and is the same for every neuron.** It counts that neuron's **own
> activations**, not a stretch of run, so a neuron that fires constantly and one that fires rarely weigh their
> functions against the same amount of evidence. **The ring is exactly `H` deep once filled.** A neuron may fire
> many times in one frame and not at all in the next (D8), so how many frames its `H` activations span is whatever
> its rate makes it, from part of one frame to a long stretch of the run. Nothing else anywhere is measured in
> frames.
>
> **Every activation the call hands it is recorded, covered or not** (R20). An arriving activation evicts the oldest,
> and only then.

## 4.3 Relations

**A neuron records one kind of relation:** a particular neighbor at an offset from itself, in one dimension.
What is general across neighbors, a place where different neurons stand, or two places that hold the same
neuron whichever it is, is not seen in any one activation, so no neuron records it. The machine makes it (R52).

> **D47 — Relation index.** What a neuron counts over its history (D18): every relation from its own activation
> that its rows hold, each with the rows that hold it. It is updated when a row is admitted and when a row is
> evicted, and at no other time. It answers three questions:
>
> | Question | Answer |
> |---|---|
> | what holds in this row | every relation from the neuron's activation there |
> | which rows hold this relation | its row list |
> | which rows hold every relation of this set, or every one but `k` | the intersection of their row lists |
>
> **An interneuron stands for what is under it.** An interneuron on the apex is a neighbor as itself, and as each of the base
> symbols it stands for, each at its own offset (R28). As itself it costs what it cost. As the base symbols under
> it, together, it costs the same: a function that lands on all of them takes its call back, and one that lands
> on some of them holds there and saves nothing (D22, R23).

How the index is laid out is the implementation's. What it holds is not: a relation's count is the number of
rows of the history that hold it.

## 4.4 Connections

What follows an activation is never in the file (D12) and enters no test. It is held only here, as connections:
what stood on the apex in the frames after it, event and action alike.

> **D25 — Connections.** A neuron holds, per `(neuron, offset)`, one **connection**: a **strength**, the number
> of times an activation of the neuron saw that neuron stand on the apex at that offset, and its **earnings**,
> the total reward the actions of those frames received. Its **estimate** is its earnings over its strength,
> and is read, never stored. The offset is a full D6 offset from the holder's activation:
> positive in time, because what is connected to comes after, and signed in every other activation dimension.
> **A connection reaches as far as a neighbor does** (D5): an activation connects to what stands on the apex
> within its reach in every activation dimension the two share (D4), the window in time (D9) and the
> same reach in every other, and to nothing beyond it. **A connection is written only where the future table of
> the policy admits the target's kind among the holder's kind's connections** (D50). A coarse offset pools the
> exposures of every frame in its group. Nothing about any one activation is kept.
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
> | return          | The relations of every activation handed in, in one return, divided by activation (R21).            |

The call runs before the machine groups and finds (R50, R53).

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
the activation: that neuron, at its offset from the activation in that dimension, written rounded as D6 writes
it. The row joins the row list of each such relation. A row that holds one relation through two neighbors, since
a coarse offset spans a group (D6), is on its list once.

**An interneuron gives its relations twice.** An interneuron standing on the apex is a neighbor as itself, and as
each base symbol under it at that symbol's own offset (D47). Both kinds join the index. The second kind is
worth the interneuron's record together and nothing singly (D47).

**A bound neighbor gives worthless relations where its binding covered it.** A neighbor bound in a variable at a
lower level costs nothing in each dimension its binding covered, so its relation in that dimension is marked:
it holds a node and covers nothing (D10, D22). In a dimension its binding left uncovered it costs a
number, and its relation counts like any other.

**An evicted row takes its relations away.** It leaves the row list of every relation it held, and a relation
whose list is empty leaves the index.

**A decommissioned neuron is forgotten.** The forget neuron call (§4.1) removes every relation that names it, and
every row's entry for it. Nothing else touches the index.

## 5.3 Return

> **R21 — What the call returns.** One return for every activation the machine handed in, divided by activation.
> For each activation it holds one relation per neighbor per dimension the two share (D34), each with the rows
> of the neuron's history that hold it, this one among them. An activation with no neighbor has nothing in its
> part.

A relation held in one row cannot recur yet, so **nothing is ever built on a single occurrence** (R50); it is
still returned, since a function already in the dictionary may fit it (R53). What the call returns is the
neuron's whole say in what is written: what it saw, and how often.
Between calls the machine may ask it to count a set of relations (D47), which changes nothing in it.

## 5.4 Cost of the call

With `A` activations this frame, `M` neighbors per row and `d` spatial dimensions:

| Step | Work |
|---|---|
| refresh history | `A` rows admitted, as many evicted |
| refresh index | `A · M · (d + 1)` relations counted in and out |
| return | `A · M · (d + 1)` relations, each with its row list |
| count, when asked | the intersection of the row lists of the relations asked about |

# Part III — Past and present: the machine

# 6. Recognition

> **D23 — Recognition.** The machine covers the board with interneurons, each taken when the neurons it covers
> cost more than its call's record, never counting its entry in the dictionary (R22, R24).
>
> **Covering everything is not the goal.** A neuron no accepted function covers stays in the file as itself, at what it
> cost (D13), and that is the shorter file whenever no neuron could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a pattern names neighbors at offsets, so a promoted neuron replaces a stretch of
> spacetime. Spatial recognition is the case where every offset is zero.

**The board** is the apex as it stands at a level, partway through the frame: what the levels below have left
uncovered so far, with the open activations of earlier frames still within reach. It changes with every level.
The apex of the frame is the board once the last level has run (D49).

**The machine compresses a frame in this order and no other.** One level at a time: the first over what the
frame reported, each after it over what the level before accepted, until a level fires nothing:

| Step       | Description |
|------------|-------------|
| activate   | Fire every interneuron the level below accepted, where what it covers stands (D53), and call `activate` on every neuron with an activation at this level (§5). They return their relations. |
| recognize  | One activation at a time, the three steps below. Every activation is visited, so every function the level can build is built before any is placed. The dictionary changes only here, between one activation and the next. |
| elect      | Place every function recognize gave on the board, and take them best first, the highest coverage over price each round, stopping when the board is covered (R24). |
| allocate   | Give each accepted function with no interneuron one, reused where a stretch comes round (R43), allocated otherwise where a neuron is free (R16). It fires at the next level. |
| prune      | Price every function whose evidence changed at this level, and drop each that no longer pays (R18). |

For each activation, recognize is:

| Step       | Description |
|------------|-------------|
| group      | Group the relations it returned in two or more rows into functions of fixed nodes, and keep each that pays (R50). |
| find       | Read each kept function's relations in the index, as they are and with either end open. An interneuron writing all of them as they are is that function. One writing all of them, some only through an open end, differs from it only there. One writing some is a partial match, the rest its errors. None writing all: the function is new and joins the dictionary. Relations held in one row are read the same way, to find what already fits them (R53). |
| generalize | Make, from each new function and each that differs from it only at an open end, the function that fits both, and add it where it is cheaper than the two (R52). |

Then, once the last level has run:

| Step         | Description |
|--------------|-------------|
| decommission | After the frame's connect and infer calls (§7), decommission every interneuron that is due (R38). |

## 6.1 Process levels

**A level is explained as a set of calls with their arguments** (D37). Every accepted function is a call: a
function's interneuron, and beside it, bound on the apex, what stood in each of the function's variables. What stands
above is made of the calls and their arguments (§6.4): a function there over an interneuron alone holds for every
argument the call was given, and one over the interneuron and a bound activation holds for that one argument.

> **R26 — Levels.** A frame is compressed one level at a time. The first level is every base activation the
> frame reported, all of them together: they are activated and return relations, the machine groups and
> matches, and the election settles which functions are accepted. The next level is the interneurons of those
> functions:
> they fire, are activated and return relations, and so on upward. **When a level fires nothing, the frame's
> levels end there.** Nothing declares the depth and nothing caps it. A level belongs to a frame and not to a
> neuron: a neuron fires at whatever level accepts its function, and one neuron may fire at different levels of
> different frames.
>
> **Within a level the order is activate, recognize, elect, allocate, prune**, and within recognize, for one
> activation at a time, group, find, generalize. It cannot be otherwise: what fires is what the level before
> accepted; what a neuron returns is what is grouped; whether a grouped function is new is known only once it is
> looked up, and the lookup is what finds the dictionary's functions it fits and the ones it is generalized
> with; the election is over everything recognize found or built, since a function built from the last activation
> may cover the board better than any found from the first, so nothing is placed until every activation has been
> recognized; an interneuron is allocated only for a function the election accepted (R16); and the dictionary is
> priced once the level has added to it. Nothing in that order leaves the level or the frame, and nothing reads
> the dictionary while it changes: it changes in recognize, between one activation and the next.
>
> Every neuron runs the same rule at its own reach (D4). **Compression is spatio-temporal for every neuron, in
> one pass**: a function may name neighbors in its own frame, in earlier ones, beside it in space, or in a mix.

> **D30 — Margin.** What a function `f` is worth over a neuron's history, less its entry in the dictionary: the
> cost of writing that history by the dictionary without `f` (A), less with `f` (B).
>
> | | |
> |---|---|
> | writing a row | what the election would write for it from the dictionary: the greedy cover (D28) of the row by the functions that fit it (D54), each taken function as its call's record and what no function covers as its own record |
> | A | writing every row of the history by the dictionary without `f` |
> | B | writing every row by the dictionary with `f`, plus `f`'s entry (D15) |
> | margin of `f` | A less B |
>
> **Whose history.** `f` is read over the history of each neuron at a fixed node of `f` whose reach covers every
> node of `f` (D4), since only that neuron's rows hold all of it. Reach in time runs back, so that is a neuron at
> `f`'s latest frame: every function has a fixed node there (D15, R52). `f` is kept while its margin is positive
> over at least one such history. The neuron answers which of its rows hold `f` (D47); the machine does the
> pricing.
>
> **Without `f`**, a row is written by the rest of the dictionary, so another function may take what `f` held. A
> function is worth what it adds over the best the dictionary does without it, and where another function does
> better in a row, `f` adds nothing there. Only `f`'s uses can differ between A and B, so only they are read. A
> class has no margin of its own: it is worth what the functions using it are worth with it in them.
>
> **A and B are computed when the margin is asked, and kept nowhere.** No neuron holds a cover or a function.

> **R50 — Grouping.** For each activation, the machine groups the relations its neuron returned in two or more
> rows (R21) into functions. It starts with each such relation as a group of its own: the activation and one
> neighbor, pinned in one dimension. It then merges, repeatedly, the two groups whose merge most lowers the cost of
> writing that neuron's history (D30), and stops when no merge lowers it. A group's rows are the rows that
> hold every relation of it, the intersection of their row lists (D47). A merge writes one call where two were
> written, in the rows that hold both groups, and gives up the rows that hold only one; the cost of the history
> says which way that comes out.
>
> Each group left is a function whose nodes are all fixed: the activation and the neighbors it pins. Merging
> priced only merges, so each group left is priced once on its own, by its margin (D30), and kept where that is
> positive. The machine rewrites each kept group's relations from the older of their two nodes (D15) and hands it
> to find (R53). The work is bounded by the activation's own relations.

> **R53 — Finding.** For each function grouping kept, the machine reads each of its relations `(a, b, δ)` in the
> index (D36) three ways: as it is; with `b` open, `(a, ?, δ)`; and with `a` open, `(?, b, δ)`. The reading as it
> is gives the interneurons whose function writes that relation. An open reading gives the interneurons whose
> function writes any relation with that neuron at that end and that offset, whatever stands at the other end: a
> neuron, a class or a parameter.
>
> | The interneurons in | Are |
> |---|---|
> | every relation's reading as it is | the function itself: it is already in the dictionary, and nothing more is done with it |
> | every relation's readings, some only open | dictionary functions that differ from the observed one only at the open ends, which it is generalized with (R52) |
> | some relations' readings | **partial matches**: dictionary functions that fit the observed one in part; the relations of theirs that nothing observed fits are their errors (D22) |
> | no reading | nothing |
>
> Where no interneuron is in every relation's reading as it is, the function is **new**. It joins the dictionary,
> with no interneuron until it is accepted (R16), and is generalized at once (R52) with the functions that differ
> from it only at the open ends. A function with no neuron and offset in common is never reached, and R52 could
> make nothing of it. Where the open end holds a class the observed neuron is a member of, or a parameter, the
> dictionary function is fit by the observed one the way D54 reads it, and this is where variables are matched.
> The machine reads the activation's relations held in one row the same three ways, to find the dictionary
> functions that fit them: those are placed like the rest, and nothing is built from them (R21). Everything find
> gives, the function itself, each partial match with its errors, and the new function, goes to the election
> once every activation of the level has been recognized (R24).

> **R52 — Generalizing.** For each new function and each function find gave it that differs from it only at an
> open end (R53), two functions that differ only in which neuron stands at some of their nodes, the machine makes
> the function that fits both, before the next activation is grouped, and what it makes goes to the election with
> the rest:
>
> - where they have the same neuron, that node is fixed;
> - where they differ, and two or more such nodes hold the same neuron within each function, those nodes are one
>   parameter (D44);
> - where they differ otherwise, the node is a class whose members are the neurons that stood there (D41).
>
> The general function is priced as a replacement (D30): the history is written with it in place of the two it
> came from, against the history written with those two, and it joins the dictionary where that is cheaper. It
> needs a fixed node in its latest frame, so two functions that differ at every node, or at every node of the
> latest frame, make nothing. **Where it joins, the two it came from leave the dictionary.** The general function
> stands for both, and every variant is written through it, its value beside the call (D48). Each occurrence pays
> one number more than through a function of its own, and the dictionary pays once less, so the general function
> joins where the dictionary saving outweighs the occurrences: long functions and many members, not a single
> relation over a few neurons. A grouped function that a general function already fits is not new, since find
> reads it through the general function's open ends (R53), so the specific form is not built again. This is least
> general generalization: the most specific function that still fits both.

Generalizing needs two functions to start from, each built on a variant that came round twice. After that, a rare
variant fits the general function directly: any neuron for a parameter, and for a class once the member pays its
one number.

> **R18 — Prune.** After the election, every function whose evidence changed at this level is priced: each
> function that names, at a fixed node, a neuron activated at this level, whose history admitted or evicted rows
> (D30), and each function a new one competes with. **They leave one at a time.** The one with the lowest margin
> leaves first, where that margin is negative over every history it is read over, and the rest are priced again
> before the next leaves, since two functions that write the same rows equally well are each worth nothing while
> the other stands, and pricing them together would drop both. Between two with the same margin, the one that
> joined the dictionary later leaves. A class no function uses any longer goes with its last function. A
> function's interneuron is decommissioned when due (R38).

## 6.2 Election

> **D31 — Placing a function.** A function find gave for an activation (R53) is placed on the board, at
> the activations it would cover. Competing explanations of the same activations are all placed, and
> the election picks among them (R24). The same function placed over the same activations from two activations is
> placed once.
>
> **A function is placed whole.** The machine takes it or leaves it, its variables with it: it cannot hold where they
> do not.
>
> **The placing rule.** The machine puts the activation at its node and each other node at its offset by the
> relations. A coarse offset spans a group of exact offsets (D6), and the node is read over every activation in the
> group. A fixed node holds where its neuron stands there, or an interneuron standing for it (D22). A class node holds
> where a member stands there. A parameter's value is the neuron standing at the earliest of its nodes, the smaller
> position on a tie, and it holds at each node where that neuron stands (D44). Where several activations could
> answer one node, the one not yet covered is taken first, then the one nearest that activation, then the
> earlier coordinate (R24). What stands in each variable on that occasion is the call's arguments (D37). Expansion
> reads a call back by the same rule (R28).

> **R22 — What a function covers on the board, and what it costs.** A function is placed on the board (D31). The machine reads
> it against what fired and derives both numbers.
>
> | | |
> |---|---|
> | the function | and its interneuron or none (D31), placed where each node stands, with what each variable binds |
> | coverage | A: the records of the activations it covers or binds that stand on the apex, and the record of each call it takes back whole (D24, R23) |
> | price | B: its call's record (D12, D22): `d + 2`, one per relation that did not come true in those same frames, one per value, and one per position number a relation leaves open |
>
> The machine places the function by D31's rule. Coverage less price is the savings
> over stating what it covers flat, D22's expression over the machine's population.
>
> **A position is covered once.** A function that binds an activation whose position an earlier accepted function covered
> is credited nothing for it (D10, D24).
>
> **A neuron that fired and the function does not name belongs to neither side.** It stands in the file as itself if
> nothing covers it (D23), and the same way if this interneuron is promoted, since the interneuron does not name it — its own
> cost either way, so it cancels before the test begins, and charging it here would count it twice against the
> uncovered term of the same sum (§6.2).
>
> **Coverage changes the credit and never the price.** A neuron the function names that fired and another function
> already covers is credited to no one and charged nothing, unless the function takes that call back whole
> (R23): it fired, so it was never among the neurons named and absent. A neuron the function names that did not fire is charged its relation whether or not another neuron is
> right at that coordinate — another neuron's expansion being right there does not make this one's wrong name free. **What a
> pattern gets wrong about a frame is a fact about the two, and ownership does not move it.**
>
> **This is the function's arithmetic over the board, and it differs from its margin.** The margin reads a
> neuron's history (D30); the election reads the board, where earlier frames' credit stands (R23) and this frame's
> other matched functions contend (R24). The two numbers differ, and are meant to (D22).
>
> **This is a price for one placement, not for the function.** The entry in the dictionary is weighed by the one test
> (D30) and appears nowhere in this price and nowhere in the election.

**The election proposes nothing.** Every function it chooses among comes from find (R53). The election only
accepts or declines one: it never edits a function, merges two, or invents a third. What it does do is *measure*
one, and everything a function is worth on the board this frame the machine works out itself (R22).

> **D24 — Coverage set.** The machine keeps, for each activation an accepted function was credited, two things:
>
> | | |
> |---|---|
> | credited to | the accepted function that was last credited the activation, by covering it or by binding it (R22) |
> | covered | whether that function covered the activation, so that it left the apex, or bound it, so that it stays with its position covered (D10) |
>
> **Credit is what blocks later payment, and coverage is what hides.** A covered activation is paid for through
> its call. It is in a later function's coverage only together with everything that call covers: the function
> takes the call back and is paid the call's record (R23). A function that names it afterward without the rest is
> neither paid for it nor charged a failure for it, since it fired. A bound activation stays on the apex at its neuron's cost and is a neighbor to every later
> function. A later function that covers it is credited its neuron, which is all that is left of it in the file. A
> later function that binds it again is credited nothing, since its position is covered already (D10).
> So every number of an activation is paid for once.
> **An activation is named by its full coordinate**, dimension and position together, so two activations of one
> neuron at two positions never contend. The coverage set spans one frame more than the longest reach in time
> among the neurons the machine holds — no function reaches further back — and the box every other activation
> dimension gives, and ages out with it, and over the same span the machine keeps the action that ran at each
> frame in each action channel, for the shares a reward sends back (R33). **The machine holds nothing on the
> scale of the run.**
>
> The record is about **credit**: an activation is a fact that needs paying for exactly once, so a covered one is
> paid for through its call until a later call takes that call back whole (R23). It is not about naming: an interneuron expands to everything its function names,
> credited or not (R24).

> **R23 — This frame's functions against the board as it stands.** Only what the file holds is in play: a bare
> activation at its cost, a bound one at its neuron's cost, and the activations a call covers at that call's
> record, together. So no number of an activation is paid for twice. Within the electing frame an activation is
> covered once, by the first accepted function that covers it (R24), and never moves.
>
> **A later frame takes the past back where it writes it shorter.** A function at `f` that lands on everything a
> call at an earlier frame within its reach covers takes that call back: its activations are covered by the new call, the old call leaves
> the file, and its interneuron's activation leaves the apex. What that interneuron learned stands. A function
> that lands on some of what a call covers takes nothing: a call is whole. So the past is re-elected only by a
> function that covers a call entire, and only where that pays on the board (D28).

**The file over one frame is the calls made plus the neurons no function covered**: the price of every accepted
function, and what each neuron no function covered cost, the transcript half of `L` (§3.11) over the frames the election can see.
**The two terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first,
which is why the price counts only the neighbors named and absent (R22). The dictionary half is D30's, and neither
test touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of functions
and scores it** — functions are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good solution, not a proved minimum. **Every neuron a function covers ends up credited to exactly
one function**, which is what stops an activation being paid for twice.

> **R24 — The election is D28, run by the machine.** The functions are everything recognize found or built at
> this level (R53, R52), less any that would need a neuron when none is free (R16), and what they cover is the
> **free set**: every activation on the frontier (R27), at its own full coordinate, frame and position, that some
> placed function names, a bound one at its neuron's cost, and every call within reach whose covered activations
> a placed function lands on entire, at that call's record (R23).
>
> Each function is placed on the board before the first round (D31), and `price` is R22's. Then D28 runs: the
> highest `coverage / price` first, each while `coverage > price` against the board as it then stands, and it
> stops when nothing on the board is left that a placed function could cover for less than it costs.
>
> **Two placements that measure the same are told apart in this order**: a function with
> an interneuron before one without; between two with interneurons, the older interneuron; between two without,
> the function that joined the dictionary first; and between two placements of one function, the earlier
> coordinate, the earlier frame first and then the smaller position, the dimensions in the order the machine
> declares them (D1). Older is earlier allocated (D2).
>
> **The bound is structural.** A function's price includes its call's record, so it is taken only when it covers
> more than one activation's record: at least two nodes. Each round takes one function, so the rounds are never
> more than the placements.
>
> **A function that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> an activation it named is either credited to a function that did pay or stands as itself.
>
> **Ownership is a partition of the neurons the accepted functions name**, and that is the whole of the
> inhibition — no function is ever edited or forbidden, and **overlap is legal and priced**: a function that names an activation
> an earlier round covered gains nothing for it and pays nothing for it, unless it takes that call back whole (R23). One that covers a bound activation gains
> its neuron's cost, and one that binds it again gains nothing (D10, R22). **Held by an accepted function** and
> **named by an accepted function** are therefore the same set, so coverage, credit and the apex frontier (R27) are
> one question with one answer.
>
> **Outcome**: accepted functions are promoted, each **whole**, and an interneuron expands to everything its function
> names, credited or not; the activations they cover leave the apex and what they bind stays on it (D24), and
> every active neuron no accepted function covers stands as itself. **The election delivers
> nothing to anyone**: it writes the coverage set and stops. No neuron is told what was bought; a neuron's history
> is what it saw, and the board is the machine's.

## 6.3 Allocate interneurons

Every accepted function with no interneuron is given one where a neuron is free. A function that lost is given nothing.

> **R16 — What an interneuron is when allocated.** The machine allocates a function's call neuron the first time the function is
> accepted, where a neuron is free (§3), unless a stretch comes round and an interneuron already stands for it
> (R43). A new interneuron has an id and no dimension (D2), and holds nothing: no history, no relation index, no
> connections. Everything it comes to hold is over the situations its function actually took (D25), from its first
> activation on. Its *existence* is decided by the election, its *structure* by its function.
>
> **No free neuron, no interneuron.** Where every neuron is allocated, a function with no interneuron cannot be granted one, so
> it is not in the election (R24). It stays in the dictionary on its own margin, and is matched again when a neuron is
> free.
>
> **Release is the reverse**: the function leaves the dictionary, and the machine decommissions its interneuron once no
> other function of it remains (R38).
>
> **Allocating is immediate.** The machine allocates or reuses the interneuron the moment the function is
> accepted, which is all the next level needs to activate it where what it covers stands (D53) and call it.
> Grouping, generalizing and pruning have already run within the level, before matching (R26).

> **R43 — Reuse.** An accepted function fires its own interneuron where it has one. Where it has none, the
> machine looks it up in the interneuron index, by what its interneuron would stand for (D36). Where one already
> stands for that, the function is added to it, and that interneuron fires. Otherwise the machine allocates one.
>
> **Otherwise the machine allocates one** (R16): a free neuron, its first activation where what it covers stands
> (D53). Where none is free the function was not in the election (R24).

## 6.4 Activate interneurons

An interneuron fires at the level after the one its function was accepted at, before that level is called. It fires
where what it covers stands (D53), and what stood in its function's variables stays on the apex beside it, bound
(D10). Nothing sees an interneuron until it fires (D5). An interneuron's activation records the function that fired it, since
it may have several (R43), and is expanded through that one (R28). Every activation no accepted function covers
stands as itself.

> **R27 — The apex is a frontier.** It is every active neuron **no accepted function covers** (D49) — the
> uncovered set, base neurons and interneurons alike — so a base neuron no pattern found worth naming stands in it beside an
> interneuron many patterns deep. This is the frontier the file counts, **the one every neuron sees as its neighborhood**
> (D5), **the one that learns what ran** (D25), and **the one that votes** (§8.4): the uncovered set does all four,
> and coverage silences a neuron in every one of them at once, its one last connection aside (D10). A reward for a
> frame already written reaches its connection regardless (R33). Everything underneath the current frontier is
> recovered by expanding it.
>
> **Named, not covered.** A neuron a declined function named, with nothing else covering it, is still on the
> frontier, and stands in the file as itself.
>
> **Coverage is acquired late and passes only whole.** An activation firing at `g` is uncovered until some function
> takes it, and a function can take it for as long as it lies within reach of where its call stands (D4), so
> an activation may speak for a few frames and then fall silent. A later call may take it from its coverer, with
> everything that coverer covered, and it stays covered (R23).

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
alongside (D8), so it is recognized and covered by the rule they are, in the same patterns (D5). **The
connection is not formed here**: it names what stands on the apex, which is known only once the frame's last level has
settled, so it is recorded in the `connect` call instead (R31).

---

## 6.5 Decommission interneurons

> **R38 — An interneuron dies when its last function leaves the dictionary and the last window over it closes.** Dying is
> decommissioning: the neuron is free to be allocated again. Once its last function has left, nothing can fire it
> again. Every function that names it as a node leaves the dictionary with it, since it can no longer hold, and so
> on up: that is its **subtree**. The death frame is the frame in which the last open activation of the interneuron, or
> of any neuron in its subtree, closes its window, and this frame when none is open. That set only shrinks. An
> activation's window closes at its reach in time (D9), so the wait is at most the longest reach in time among the
> interneuron and the neurons built on it.
>
> **Decommissioning is after the frame's connect and infer calls** (§7), so the last exposure an activation writes
> at age `reach_t` is taken before its neuron goes. Every frame, after those calls, the machine takes everything
> due, and calls forget neuron on every neuron whose history names one (§4.1).
>
> **The ledger holds the function.** Until the windows of the activations a function fired have closed, calls
> above them may still expand them, so the function stays readable after it leaves the dictionary.

## 6.6 Cost per level

With `G` relations returned per activation, `F` functions reached per activation, `L` relations per function
and `U` the uses of a function:

| Step | Work |
|---|---|
| group | per activation, up to `G²` merges, each an intersection of row lists; then one margin per group kept |
| find | per kept function, one index read per relation, widened by the classes its far neuron is in; one count per interneuron reached |
| generalize | per new function, each function differing from it only at an open end, `L` nodes compared |
| elect | the functions recognize gave, a greedy cover over the board, stopping when it is covered |
| prune | per function whose evidence changed, writing its `U` uses with and without it; writing a row is a greedy cover |

Grouping and pruning dominate. Within recognize nothing waits on anything but the activation before it, and the
dictionary is read only between its changes; the election waits for recognize, since the best function may be the
last one built. Everything is bounded by what the level changed and by `H`, and nothing scales with the
run. What is cached, and how, is the implementation's.

# Part IV — Future: a neuron

# 7. Connect and infer calls

Once the frame's last level has run, the machine calls every open activation once more, and what the neuron does with
that call is the whole of this part. Which call it is depends on the activation's age, and the two never meet on
one visit:

| Call    | Description                                                                                                                            |
|---------|----------------------------------------------------------------------------------------------------------------------------------------|
| connect | At every age from 1 to `reach_t`: the activation connects to what stands on the apex this frame, with any reward at its distance (R31, R33). It returns nothing (§7.1). |
| infer   | At age 0, and only then: the activation returns what its neuron is connected to at every offset, each with its strength and estimate: its inferences (R36). It writes nothing (§7.2). |

**Nothing is written at age 0 and nothing is returned after it.** A connection's offset is the activation's age, and an
offset is strictly positive in time (D25), so the visit at age 0 has nothing to write, and a reward share at distance `d`
lands at offset `a − d`, which at age 0 is offset 0 again (R33). The visits after age 0 have nothing to say: the
inferences are read once, in the frame the activation fires (R36).

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
(R33) — for this frame, the connections just strengthened, in the same write. A share for an earlier frame,
at distance `d`, goes into the connection at the offset the activation's age less `d` rounds to, to the action
the machine hands it for that frame, where the activation was uncovered then (D9). The activation keeps nothing
of it. **Nothing is decided, priced or compared here**, and no test is waiting on any of
it.

> **R31 — A connection carries an estimate.** What stands on the apex at `f + 1` is not known at `f`: the action is
> settled only when the report names what ran, and the events only when the environment reports them (D40). So at
> `f + 1`, once the frame's last level has run, **every uncovered activation open at that frame connects to every strong
> activation standing on the apex of that frame within its reach that the policy admits** (R27, D25, D50) at its
> own age (§7.1), and the reward of that frame's
> actions goes into each. An activation covered in that frame connects once more, to the neuron that covered it
> where that is strong, and to nothing else (D10). A weak activation is connected to by nothing: it is a proposal,
> not a fact. That connection binds what the neuron stands for to what followed — formed against what actually
> stood, so **a neuron that inferred something else, or nothing, learns from what happened.**
>
> **One activation connects to the apex of every frame it is open through, and one strong apex activation is
> connected to by every uncovered activation open when it stood.** The offset is the age — the distance from the frame the
> activation opened to the frame the call ran — rounded as every offset is (D6), so at `R = 1` a neuron open at
> ages 1, 2 and 3 holds the same neuron at two offsets, `1` and `2`, and the exposure at age 3 strengthens the
> second; at `R = 4` it holds it at three.
> **A coarse offset takes one exposure per frame of its group**, so the outer offsets pool the apexes of many
> frames, each at its own strength, as a coarse offset carries several neighbors (D6). A neuron that stood twice
> inside one group over one activation is two exposures: each frame has its own reward, and the connection keeps
> nothing that could tell the two apart (D25).
>
> **Writing rounds, reading does not.** An exposure is written at the offset its age rounds to (D6): at `R = 1`
> an activation that saw a neuron at age 2 and one that saw it at age 3 strengthen the same connection, at
> offset 2. An inference at that offset is placed at exactly that offset, the lower edge of its group, `b`
> frames after the activation that reads it (R36). So a replay of what was seen at age 3 completes at 2, and
> what it earns lands on the connection it was learned from, within the group's slack, which is the loss D6
> carries everywhere else. The match reads the other way: what an expansion places into the past is looked for
> anywhere in its offset's group (R49). Writing rounds to the group, matching reads the whole group, and only
> the placement ahead lands on one frame, the group's lower edge, because a base symbol has to be put somewhere. Fan-out is the apex within reach: a neuron connects to what stood
> uncovered in the frames after it, as far as it sees in every other dimension, and to nothing beneath that, and one exposure per apex activation per frame means a
> neuron of reach `r` holds at most `offsets(r)` offsets per neuron it has seen (D6).
>
> **Making and strengthening are one operation.** A neuron's connection at `(neuron, offset)` has a
> **strength**, the number of its exposures — the times an activation of the neuron saw that neuron on the apex
> at that offset — and its **earnings**, the total reward those exposures received (R33). The first exposure
> creates the connection at strength 1 and earnings 0 and every later one adds 1 to the strength; each share of
> reward adds to the earnings. The estimate is the earnings over the strength, the exact average over the
> connection's exposures, and no rate is chosen. **Nothing leaves.** No exposure is ever subtracted, no strength ever falls, and no window bounds a
> connection: it is the whole of what the neuron has seen follow it at that offset, over its life.
>
> **Every neuron holds them, base neurons included.** No operation derives what an interneuron was worth from what the
> neurons under it earned or the reverse, so a connection held only at the frontier would be lost at the next allocation. Every
> neuron keeps its own connections over its own activations (D25).
>
> **A connection dies with either of its ends.** Nothing else removes one (R34).
>
> **A connection wired ahead of any exposure is created at strength 1 and earnings 0.** The walk (R37) is the one
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
> `S` frames the share at distance `d` is `reward · (S − d) / S`, for `d` from `0` to `S − 1`, so the frame the reward arrived in takes the whole reward, the far end of the span takes `reward / S`, and
> nothing along the way takes nothing. A span of one frame is the degenerate case, `share(0) = reward`, and it
> is what an environment that pays each frame's action as it goes reports.
>
> **The whole window is the longest open one.** When no span is named, `S` is the longest reach in time among
> the neurons the machine holds: an activation of that neuron is open through `S` frames (D9), so the furthest
> back any open activation could have seen an action run is `S − 1` frames, and no share can reach further than
> the connections that would take it. With base neurons alone that is `S = 1`, the degenerate case above.
>
> **Nothing is being divided up.** Every distance in the span is paid, and the shares are not a partition of
> the reward — a reward is not in short supply, and the point of spreading it is not to conserve it but to say
> how likely each frame is to have earned it. What is genuinely responsible recurs and accumulates; what is not
> is sampled once and averaged away. **Each share is delivered, in the `connect` call (§7.1), to every open
> activation past age 0, and the activation picks the connection from its age and the reward's channels.** The machine hands
> it, for the frame each distance names, the action that ran there in each channel the reward pays, from the
> window it keeps (D24). An activation at age `a` takes the share at distance `d`, `d` at most `a`, into its
> neuron's connection at the offset `a − d` rounds to, to that action, where it was uncovered at that frame (D9):
> at distance `0` the connection this call strengthens, in the same write; further back one strengthened `d`
> frames ago (R31). An activation whose window has closed is not called, so a share reaches no further back than the
> open activations that would take it.
>
> **A scoped reward reaches base actions of the channels it names, and nothing else.** A connection names a
> neuron (D25). A share of a reward that names channels is delivered only to a connection whose neuron is a base
> action of one of them; a connection to an event or to a pattern takes none of it. A pattern belongs to no
> channel (D2), and a share placed through its expansion would give a call spanning two channels one estimate,
> read in both at resolution (R47). An unscoped reward reaches every connection written at the frames it pays. A
> machine whose policy lets patterns be connected to (D50) and whose rewards name channels leaves those
> connections unpriced by them; the two are not declared together.
>
> **Rewards in one frame are independent.** A connection at one `(neuron, offset)` takes the sum of the shares that
> reach it, and nothing coordinates one reward with another.
>
> **The unscoped form is the general case, and the machine sorts out the attribution itself.** An environment
> that knows what it is paying for names it — a stock channel, and the frames a position was held — and
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
> **Estimates are not decayed, and not windowed.** A connection's estimate is its earnings over its strength,
> the plain average over every exposure it has had, with no cap, no rate and no horizon; nothing ever leaves it
> (R31). **Non-choice moves nothing**: an
> action not taken keeps the value it had. What makes an estimate specific is the neuron that holds it, not
> how recent its exposures are — an interneuron's connections are over the one situation its function names (R35),
> and a changed situation is answered by a new interneuron, never by forgetting.

## 7.2 Inference

**At age 0, and only then, the machine calls `infer`, and the call returns what the activation infers.** It reads its own neuron's connections
at every offset, out to its reach, and returns each with its strength and estimate, covered or not. Those are
its **inferences** (R36), each naming a neuron, base or interneuron (D25), and the machine keeps them for
as long as the activation is open (R45); at every later age the machine calls `connect` instead (§7.1), which
delivers and asks nothing. A base neuron
infers from its own connections like any other; what it infers is the marginal over every situation it fires
in, and it is heard only because nothing more specific covers it.

**An activation's window closes at age `reach_t`** (D9), once that frame's exposure is taken. The window closing stops the
connect calls and drops its standing inferences (R45), and nothing else — no call marks it and nothing is
saved twice.

> **R36 — Inference.** One connection read by one activation at age 0: it names a neuron, base or
> interneuron (D25), and carries the strength and the estimate the connection held then. An activation reads every
> connection its neuron holds, at every offset out to its reach, once, in the frame it fires (§7.2), and never
> again: what the walk wires or a reward moves afterwards is read by the neuron's next activation. The winner of
> every dimension is chosen from the inferences and nothing else (R47).
>
> **An inference is placed at its completion, and a connection means what it recorded.** A connection at offset
> `b` was written when that neuron completed `b` frames after the situation opened (R31), so an inference at
> offset `b` completes `b` frames after the activation that made it, and at the offset's components in every
> other activation dimension from its position, as a neighbor is placed (R28). Expanded through its function
> it places base symbols at the frames back from there (R30), every one carrying the strength and the
> estimate of the connection it came from. A connection to a call neuron places a call (D37), weak, its
> parameters bound from what stands at those of their places that have happened (D44). What its expansion
> puts at the frame ahead is what it proposes there, and the machine expands no more of it than that (R45): at
> distance 1 the last thing a pattern places, at distance ten the one ten from its end, and the first only when its
> completion is placed its whole length ahead. An inference is proposed only where its match holds (R49), and
> resolution runs at the base and only there (R47). **This is what launches a program**, and it is why every offset is read and not the next one alone: an activation can start
> a program only from an offset at least as far out as the program is long, and can carry the tail of a longer
> one to completion from any offset. A situation whose reach is shorter than a program never starts it, and
> that is correct — it cannot see that far.
>
> **Two positions are two voters.** Two activations of one neuron read one set of connections (D11) and place
> the same inference from two places: in a channel whose only activation dimension is time that is one proposal
> twice, and in a channel with a layout it is a proposal at each of two positions, each resolved where it lands
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

No neuron takes part in any of it. What an activation returned is kept, placed, expanded, matched, resolved
and fired by the machine alone, as the election is the machine's alone (R24), and a neuron learns of the outcome
only what happened (R32).

## 8.1 Standing inferences

> **R45 — Standing inferences.** The machine holds every inference of every open activation, from the
> activation's age 0 until its window closes (D9), and nothing is read twice: these are the **standing inferences**, and
> they are what votes (R46). Each is held as what its connection named, the neuron and its completion, with the
> strength and the estimate it was read with, and the machine expands of it only what comes due: at age 0 the
> part that falls at or before the frame, which is matched against what stood (R49), and in each frame after
> the part that lands at the frame ahead, which is what it proposes there (R47). A program a hundred frames long
> is one standing inference, unrolled a frame at a time. **A standing inference is dropped** when its
> activation's window closes, and when its match fails (R49): a plan holds because it keeps being what
> happened, and nothing is retracted. An inference stands whether its activation is covered or
> not; coverage decides whether it is heard (R46).
>
> **What stands is what was read.** A reward may move an estimate after an activation's age 0, and the walk may
> wire a connection. The neuron's next activation reads them (R36). This one does not. A situation that launched
> a plan keeps proposing it at the estimate it read, until its window closes or its match fails. The rewards the
> plan's actions bring as they run fold into the connection's estimate on the neuron (R31). The standing
> inference does not read it again.

## 8.2 Expansion

> **R28 — Expansion.** A neuron above the base is not yet anything in the base alphabet. Expanding it recovers
> what its function covered and bound: the nodes its function names, the one that fired the activation (§6.4), each
> at its offset from where the call stands, by the relations (D15, D34) — offsets compose because each is a
> difference of activation coordinates (D2) — repeated down to base symbols. Each activation is expanded through
> the function that fired it (§6.4), so expansion follows what actually fired, frame by frame, back to where it
> began, and it ends there, even where a function comes back to its own interneuron. A fixed node expands to its neuron; a variable expands to its argument, the neuron bound there (D37).
> So `A` placed at `f`, whose function is `v` with `u` one frame before it, expands to `v` at `f` and `u` at
> `f − 1`; and `P` placed at `f + 4`, whose function is `b` with `C` two frames before it, `C`'s function being `c`
> with `d` one frame before it, expands to `b` at `f + 4`, `c` at `f + 2` and `d` at `f + 1`.
> **A coarse offset expands to its rounded coordinate.** A neighbor named within the radius is placed where it
> fired. One named beyond the radius is placed at exactly the distance its offset names, whatever distance in
> the group it fired at, and several neighbors at one coarse offset (D6) are each placed there. **The rounding
> composes.** Each expansion down adds its own group's slack, so a neuron places the base symbols of its
> farthest neighbors to within the sum of the groups along the path: the farther it reaches, the coarser its far
> placements. That is the loss the file carries (§1), and expansion reports it faithfully rather than hiding it.
>
> **This is the one expansion in the design.** It recovers the run from the file (D12): a class to the
> member bound at its place, a parameter to the neuron bound at every one of its places. And it turns an
> inferred function into a program (R30): the actions it places run, and the events it places fire weakly
> beside them (D40). It reads the dictionary only. Recognition runs it backward: a function fit by what fired
> says that this function ran here, with these arguments, and the neuron above sees the call neuron and the
> arguments bound beside it. **What travels down with a symbol is what the connection carried**: every base
> symbol an inference's expansion places carries the strength and the estimate of the connection it came from,
> and nothing is re-weighted on the way down.
>
> **Expansion substitutes the arguments.** A recognized call's arguments stand on the apex, bound, and need no
> substituting. An inferred call brought none, so its variables are bound from what has happened: a parameter
> from what stands at those of its places that have happened, and placed at the rest, which is how a value is
> carried forward; where none of its places has happened it places its most frequent member. A class with
> no argument places its most frequent member, the oldest on a tie. A function above that names an interneuron alone
> expands to the call with its arguments left to be bound, and one that names the interneuron and a bound activation
> expands to the call with that argument. A neighbor that is itself an interneuron is expanded the same way
> from where it is placed. **Two words, never one.** A function is **expanded**, downward, into what it covers;
> a connection is **inferred**, forward, into the frames ahead (D25); what a call does and what follows it are
> different directions.
>
> **A connection is placed the way a neighbor is.** A connection at offset `b`, read by an activation at its age
> 0 (R36), puts the neuron it names `b` frames after the activation's frame — that is where it completes — and
> its expansion hangs from there, its base symbols at the frames back from it. At a later age `a`, the raw count
> of frames elapsed (D9), that completion is `b − a` frames ahead: `b` is the connection's offset, a D6-rounded
> number, and `a` is never rounded. **A coarse offset is not a window.** Its action completes at `b`, not
> somewhere in the group `b` stands for, exactly as a neighbor named at `−b` is placed at `−b` and nowhere else.
> So the base symbols of a long program reach the frame ahead one at a time, in order, each at exactly one age,
> from one connection and with nothing held: what an activation places beyond the frame ahead it places again
> next frame, one frame nearer.

> **R30 — Execution is an expansion**, of the selected function (R47) through its entry in the dictionary
> (R28). A high function becomes its nodes at the offsets it records, down to base symbols. Its base actions are output,
> each in the frame its expansion places it in, the nearest being `+1` (R29); its base events are placed in
> theirs. Both fire weakly (D40), and the report replaces them: the actions it names as run and the events it
> names as seen become strong, and the rest last their frame. Execution is not a second mechanism; it is this
> expansion read as a program. A weak pattern's expansion places no action: what is output is chosen by
> estimate, never by a pattern that was itself inferred.
>
> **Execution activates what it expands, where it won.** A base action or a base event fires weakly in the frame R47
> selected it, and a pattern the expansion passed through fires weakly when its
> expansion completes, at the coordinate the expansion placed it — the last frame of what it covers, which is the
> coordinate recognition would have given it — provided the standing inference that placed it still stands
> (R45) and what it placed at that frame was selected (R47). An inference that lost its dimensions, or whose
> match failed on the way, fires nothing.
> Where the report makes its actions strong, recognition builds the strong activation of the same pattern at
> the same coordinate, and it replaces the weak one (D40). An executed pattern is therefore in its frame's levels
> like a bought one: it is activated, may be covered (§6.1), and every uncovered activation open
> at that frame connects to it (D25). While its
> program runs, what stands on the apex at each frame is the base action it placed there, or a lower pattern as
> it completes; the whole stands on the apex only in the frame it completes, and what the program earned reaches it through the span a
> reward names (R33).

## 8.3 Match

> **R49 — Match.** The machine reads one standing inference (R45) against the frames its function reaches
> back into, and proposes it where the function would pay there and strikes it where it would not. It runs
> every frame the inference stands, over the part of the function that has come to lie in the past. Nothing competes in it,
> nothing is credited and nothing is written: it is the one valuation (D22) read over its third set (§3.10).
>
> **What is read.** The inferred interneuron's function, as it stands in the dictionary (R28): its fixed nodes, each at
> an offset, and its variables, each at its places, placed back from the completion the inference names, one
> expansion down and no deeper. Whatever is placed at the frame ahead or beyond is what the inference proposes,
> and is not judged. Against them stands the apex of each frame within the longest reach in
> time, as the machine keeps it (D24).
>
> **What each placed node is worth.**
>
> | Placed node | Worth |
> |---|---|
> | a neighbor standing at its offset | credits what it cost |
> | a neighbor standing within reach at another offset | credits what it cost, and charges an offset (D13) |
> | a neighbor standing nowhere within reach | charges its relation (D13) |
> | a variable | bound from what stands at its places that have happened (D44), crediting their positions; a place holding nothing it could take charges its relation |
>
> An offset is read as D6 reads it: exact within the radius, and beyond it as its whole group, composed down the
> expansion as the placement is, so a neighbor named at a coarse offset stands at its offset anywhere in the
> group. A neighbor found at a different distance is a correction that says where, and it costs what an offset
> costs, near or far: what makes a near miss cheap is that within the group it costs nothing.
>
> **The test.** The inference is proposed when what the placed nodes credit exceeds what they charge, the
> relations that did not come true included, and struck otherwise. The call's record is not charged: the match
> writes nothing, it only asks whether the frames that have happened bear the inference out, so it is D22's test
> over the nodes that have happened, with B holding the relations that did not come true and nothing else.
>
> **What it decides.** Proposed, the inference's placements ahead contend for their positions (R46, R47).
> Struck, it leaves the standing inferences (R45). An inference that was proposed and stops matching as its
> placements come due is struck then, which is the whole of how a plan ends before its span does.
>
> **Where it stands among the three.** The machine reads a function against a neuron's rows and prices it
> (D30). The election reads the matched functions against the board and credits them (R24). The match reads one
> inferred pattern against the frames it points back into and admits or strikes it. The prices are the same;
> the questions are not.

## 8.4 Resolution

> **R35 — Selection.** No fit says which action to take; it says only what a situation was followed by.
> Choosing comes from the connections of the neurons standing on the apex, each carrying the reward that
> arrived averaged over its exposures, and **the machine outputs the best. Nothing else decides it.**
>
> **A situation is the apex** (D49): what stands uncovered at the frame, base neurons and interneurons alike. No neuron stands for
> it whole unless a pattern has covered it, so it is read a neuron at a time: a voter is one activation of it,
> and what it holds is its own neuron's record of what followed.
>
> **A voter is one apex activation**, speaking from the connections its neuron held when it fired (R36). A base
> neuron's estimate is a marginal over every situation it has stood in; an interneuron's is over those its function
> recurs in, one situation; so the same frame reads differently to a base neuron, to the interneuron covering
> part of it and to the interneuron covering the whole. **A function's interneuron is how one recurring situation acquires an estimate of its own**
> — which is why a covered neuron is silenced (D10) and its coverer speaks instead, and the only reason a base
> neuron's marginal ever speaks is that nothing more specific was bought over it.
>
> **Every apex activation votes**, base actions included. A pattern that names what was done beside what was
> seen votes knowing the situation; an interneuron over actions alone, or a base action, votes what followed it, a habit,
> and reward prices the habit like anything else. What an inference places in an event dimension is resolved
> the same way, and the winner is output (R47, R30).
>
> **Nothing runs where nothing infers.** An action dimension no inference reaches is output nothing: no call
> runs, nothing fires in it, and the frame carries silence there as an event dimension with nothing to report
> does. The first action in a dimension is the world's. An action the environment executes on its own is a
> lesson: it stands on the apex of its frame, every uncovered activation connects to it with its reward by
> the ordinary path (R31), and from then on it is an action the machine can infer. A neuron is born holding no
> connection at all — a base neuron at cold start and a newly
> allocated interneuron alike say nothing about the next action until something has run under them, and what they
> then hold is whatever was executed. R37's walk is the only thing that wires an action ahead of its running,
> one at a time and only where the best known has been judged and found wanting.

> **R46 — Electorate.** At frame `f` the voters on the frame ahead, `f + 1` (R35), are the open activations
> the machine holds at `f` (D9), less every activation an accepted function covers (D10), plus every covered
> activation whose coverer has nothing for `f + 1`, and so on down through the covers to the base (R48). What
> each voter proposes is what its standing inferences place at `f + 1` (R45): a base action by the inference at
> the offset one ahead of its age (R29), a base symbol a pattern places by a farther one, and nothing where none of them lands
> there. A voter is read as `(neuron, position)`: the inferences are its neuron's (D11), and the placement is
> from its position (R36).

> **R47 — Resolution.** One winner per dimension per position at `f + 1`, action and event alike (R30), and
> nothing about where a voter stands in the hierarchy is read in it: an interneuron's inference and a base voter's
> meet on the estimate or the share alone, and a specific situation wins over a general one only by being right about what pays, never by
> rank.
>
> **An action dimension resolves by estimate.** At each position of it, every base action some voter's expansion
> placed there is a candidate (D8). **A candidate's estimate is the mean of the estimates its voters placed it
> with, each weighted by that voter's share** — one vote per voter, split across the actions it placed at that
> position by strength, so a voter that hedges between two actions counts as one voter and not two — and **the
> candidate with the largest estimate is output**, unless every candidate's estimate is negative, where the
> action policy decides between the least bad and nothing (D51). Ties go to the larger share of voters, and
> then to the older action. **Nothing corrects for how many exposures an estimate rests on**, so a sharp estimate on three
> exposures outranks a coarse one on two hundred.
>
> **An event dimension resolves by share of voters.** At each position of it, every base event some voter's
> expansion placed there is a candidate, and every voter whose expansion placed one is one vote, split across the
> events it placed at that position by strength, as above. **The candidate with the largest share is output**
> (R40). The estimate is not read here: it is the reward of the actions of those frames, and an event is what
> followed, not what paid. Ties go to the older event. A position no voter reaches outputs nothing.
>
> **A bad plan is worn down, not cut.** A standing inference carries the estimate it read at age 0 (R45). Each
> frame it places what comes due at the frame ahead: actions in their dimensions, events in theirs, at
> their positions, all at that estimate. Activations that fired after a reward moved the estimate place
> candidates at the moved number, or the walk's untried action at zero (R37). Where the largest estimate is one
> of those, the plan's action does not run; where the world reports an event other than the one the plan
> placed, that event did not stand. Each such placement is a failed neighbor to the plan's match (R49). The
> plan stands while what it found still pays for what it missed, and is dropped when it no longer does.
>
> **Each dimension is resolved on its own**: a plan that loses its action dimension still counts for the events it
> placed, and one that wins is not favored for them, so what an event dimension outputs is what the most voters placed, whichever
> inference placed it, and an inference that places events alone is resolved like any other.

> **D51 — Action policy.** Declared once for the machine, as the alphabet and the neighborhood policy are
> (D1, D50): what an action dimension outputs when every candidate in it carries a negative estimate (R47),
> **the least bad, or nothing**. The first is for a world that acts whether or not the machine does, where
> silence hands the frame to the world's own action and the machine learns from that (R35); the second is for a
> world in which silence is an action in its own right and costs nothing. The walk is the same under either
> (R37); the policy decides only what runs once the walk has nothing left to wire in a dimension.

> **R48 — Speaking falls through the cover.** A covered neuron supplies nothing while its coverer speaks (D10).
> Where the coverer has no inference for the frame ahead, what it covers speaks in its place, and what they cover
> in theirs, down to the base; nothing is output where nothing down to the base has anything. A newly allocated
> interneuron therefore starts with the voices of what it covers: they say what usually followed them, it learns what ran under
> it (R35), and from then on it speaks for itself. At the base a neuron with nothing outputs nothing, and the
> walk (R37) wires the next action where an estimate turns negative.

> **R37 — Exploration.** Explore–exploit is resolved without randomness: **the action alphabet
> is declared in order**, and **a connection whose estimate turns negative wires the next action in that
> order** — the first one the neuron holds no connection to at that offset — at strength 1 and earnings 0
> (R31). **The walk runs for every neuron, and its dimensions are the connection's.** A neuron above the base is
> not an alphabet: it was created when a pattern paid, it has no declared order, and nothing could name the next
> one. The base actions are the one ordered alphabet there is, so what a negative connection wires is base
> actions, in the dimensions its target acted in, read off the target's expansion (R28): a base action, its own
> dimension; a pattern, every action dimension its expansion places an action in, one untried base action each;
> an event, or a pattern that places no action, nothing. An interneuron whose plan hurt therefore tries,
> at that offset, one other action in each dimension the plan acted in, and nothing at any other offset. **What the
> walk finds, compression names.** The untried action is a base action under a situation an interneuron stands
> for, placed by the neuron that walked; where it pays it recurs, where it recurs the activations under that
> situation name it in a function (R50), and that function's interneuron is new, with the walked action
> inside it, found as a lesson's action is found, one base action at a time and then named. What stands in a
> call's variables is never searched: it is whatever stands at their places when the call runs (D41, D44).
>
> **The walk is in the same currency as everything else**: an untried action becomes a candidate by becoming a
> connection, so R46 enumerates it like any other and needs no second source of inferences. The trigger is one
> connection at one offset, never a reading over the neuron's whole set.
>
> **A reward is signed, and zero is where everything starts.** The environment reports an action as good or bad
> — strictly greater than zero or strictly less — and zero is neither. It is also the estimate every connection is
> created at, which is what makes the walk work: a connection at zero has not been
> judged, so it outranks anything negative and yields to anything positive.
>
> **The walk ends when the alphabet does.** Once a neuron holds a connection to every action in a dimension at
> that offset there is nothing left to wire there, and what the dimension does when every candidate in it is
> negative is the action policy's (D51): the least bad, or nothing.

## 8.5 Activation

> **R40 — An inference is weak until the report makes it strong.** Resolution leaves one winner per dimension,
> action and event alike (R47), and each fires weakly in the frame it was placed in (D40), the action to run and
> the event to come, and both are the machine's output. Where the report names that
> dimension at that coordinate, the report stands and the weak activation is dropped; an action nothing reports as run, or an event nothing
> reports as seen, lasts its frame and is gone. Where the expansion reaches a parameter, what fires is its value,
> bound from what has happened (D44). An inferred event above the base is expanded in turn, one frame at a time
> as its members come due, so the events a plan infers reach the base as inputs exactly as its actions reach
> the base as output. A weak activation is an input of its frame: it is recognized as a neighbor, it may
> be covered or stand on the apex and vote once, and then it is gone. It carries no reward, connects to nothing,
> and is connected to by nothing (R31).

# Part VI — Beyond the specification

# 9. Examples

Nine cases, each worked by hand in a file of its own; each follows the definitions through one small job.

| Case | File | What it shows |
|---|---|---|
| Alternation | [algorithm-xy.md](algorithm-xy.md) | A stream at radius 4. A neuron that keeps seeing a different pair of letters gets one function with two parameters tying the letters two frames apart, generalized from two pairs that each came round twice and replacing their functions: its interneuron is the alternation, one neuron for every such pair, the letters bound beside it, and a pair never seen fits it the first time. A pair that comes often is covered by its own letters' functions first, and gets a function naming that interneuron. |
| Addition | [algorithm-addition.md](algorithm-addition.md) | A request with each column, a neuron for each case that can be asked, a function per case, and a carry the world shows while it teaches and the machine infers afterwards. |
| Copy | [algorithm-copy.md](algorithm-copy.md) | Shown a digit and asked, the machine says it: one function over the answer with a parameter tying the digit shown to the digit said, generalized from two digits' answers and replacing them, so a digit never said before is said the first time; it reads the shown digit through the call that covers the request, and takes that call back. |
| Equal | [algorithm-equal.md](algorithm-equal.md) | Shown four digits in one frame and asked, the machine says the third where the first two are the same and the fourth where they differ: equality is a parameter over two positions of the request, fitting an unequal request too with one relation that did not come true; the answer is a copy by a parameter over a position of the request and the position of the answer; and the answer at the first position is decided by a vote the interneuron that knows the situation wins, while the other positions get answers the world's reward decides on. |
| Hit | [algorithm-hit.md](algorithm-hit.md) | Offsets from where the world reports something approaching as the reference frame, a parameter that takes whatever comes, bound beside the call, found through its open ends from relations seen once, and a thing never seen answered by the lesson of the call neuron, which stands for anything coming that way. |
| Loop | [algorithm-loop.md](algorithm-loop.md) | Shown a cue, the machine taps in each of the next three frames and then stops. There is no counter: each tap's function names what stood a frame back, so the states of the loop are three interneurons each built over the one before, one per run, each connected to the one after it and the last to the event the world shows at the end. Stopping is a thing that follows, since speaking falls through the cover; taught to stop on a cue instead of a count, the same machine is a while loop. |
| MNIST | [algorithm-mnist.md](algorithm-mnist.md) | An image channel with two spatial dimensions and one base neuron, ink; relations one per dimension, grouped into functions whose calls carry the positions no relation fixes, tiling the image at every level; a digit action taught by demonstration and resolved by a vote. |
| Stocks | [algorithm-stocks.md](algorithm-stocks.md) | A price channel per instrument with own and disown as its actions, under a policy where an action is no one's neighbor and only events and patterns connect, to actions alone; patterns over frames; a world that runs the right action wherever the machine has none; and a reward scoped to the channel and the frame, each frame. |
| Text | [algorithm-text.md](algorithm-text.md) | A stream of letters at reach 1: pairs at the base, and no class over the letters before a letter, since over one relation it never pays; stretches of any length, each letter's function naming the interneuron a frame before, one neuron per stretch that recurs rather than per word; a frequent doubled letter as a pair like any other. |

# 10. Further reading

Each document is written in the terms of this specification and references its keys.

| Document | What it holds |
|---|---|
| [algorithm-remarks.md](algorithm-remarks.md) | The theorems, and the commentary on why the design is shaped this way, keyed by the same D and R numbers. |
| [algorithm-evaluation.md](algorithm-evaluation.md) | The falsifiable claim, the risks with what would be done about each, and the open questions. |
| [algorithm-implementation.md](algorithm-implementation.md) | The implementation plan: the per-neuron state and calls, the build stages, and the deltas against the current code. |
| [hippocampus.md](hippocampus.md) | An addendum: the second organ, running beside this machine over the same neurons, with rules keyed **H**. |